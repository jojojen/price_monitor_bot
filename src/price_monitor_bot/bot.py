from __future__ import annotations

import json
import logging
import mimetypes
import ssl
import tempfile
import time
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

if TYPE_CHECKING:
    from sns_monitor.storage import SnsDatabase

from hashlib import sha1

from market_monitor.http import HttpClient
from market_monitor.storage import MercariWatch, MonitorDatabase
from tcg_tracker.hot_cards import HotCardBoard, TcgHotCardService
from tcg_tracker.image_lookup import (
    TcgImageLookupOutcome,
    TcgImagePriceService,
    TcgVisionSettings,
    _sanitize_image_title_hint,
)

from .commands import lookup_card
from .formatters import format_jpy, format_lookup_result_telegram
from .logging_utils import mask_identifier, trim_for_log
from .natural_language import (
    TelegramNaturalLanguageIntent,
    TelegramNaturalLanguageRouter,
    fallback_route_telegram_natural_language,
)

LookupRenderer = Callable[["TelegramLookupQuery"], str]
PhotoLookupRenderer = Callable[["TelegramPhotoQuery"], str]
ReputationRenderer = Callable[["TelegramReputationQuery"], object]
BoardLoader = Callable[[], tuple[HotCardBoard, ...]]
CatalogRenderer = Callable[[], str]
WatchlistStore = object  # MonitorDatabase or None

PRICE_LOOKUP_COMMANDS = {"/lookup", "/price"}
TREND_BOARD_COMMANDS = {"/trend", "/trending", "/hot", "/heat", "/liquidity"}
PHOTO_SCAN_COMMANDS = {"/scan", "/image", "/photo"}
REPUTATION_SNAPSHOT_COMMANDS = {"/snapshot", "/proof", "/repcheck", "/reputation"}
WATCH_COMMANDS = {"/watch"}
WATCHLIST_COMMANDS = {"/watchlist", "/watches"}
UNWATCH_COMMANDS = {"/unwatch", "/stopwatch"}
SET_PRICE_COMMANDS = {"/setprice", "/updatewatch"}
SNS_ADD_COMMANDS = {"/snsadd", "/sns_add"}
SNS_LIST_COMMANDS = {"/snslist", "/sns_list"}
SNS_DELETE_COMMANDS = {"/snsdelete", "/sns_delete"}
SNS_BUZZ_COMMANDS = {"/snsbuzz", "/sns_buzz"}
HEAVY_COMMANDS = PRICE_LOOKUP_COMMANDS | TREND_BOARD_COMMANDS | REPUTATION_SNAPSHOT_COMMANDS

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class TelegramLookupQuery:
    game: str
    name: str
    card_number: str | None = None
    rarity: str | None = None
    set_code: str | None = None


@dataclass(frozen=True, slots=True)
class TelegramPhotoQuery:
    chat_id: str | int
    image_path: Path
    caption: str | None = None
    game_hint: str | None = None
    title_hint: str | None = None
    file_id: str | None = None


@dataclass(frozen=True, slots=True)
class TelegramReputationQuery:
    query_url: str


@dataclass(frozen=True, slots=True)
class TelegramFileAttachment:
    kind: str
    path: Path
    caption: str | None = None


@dataclass(frozen=True, slots=True)
class TelegramReputationDelivery:
    summary_text: str
    attachments: tuple[TelegramFileAttachment, ...] = ()
    cleanup_paths: tuple[Path, ...] = ()


@dataclass(frozen=True, slots=True)
class TelegramTextReplyPlan:
    ack: str | None
    reply: str | None
    reply_factory: Callable[[], str] | None = None
    reputation_delivery_factory: "Callable[[], TelegramReputationDelivery] | None" = None

    def execute(self) -> str | None:
        if self.reply is not None:
            return self.reply
        if self.reply_factory is not None:
            return self.reply_factory()
        return None


class TelegramBotClient:
    def __init__(
        self,
        token: str,
        *,
        timeout_seconds: float = 35.0,
        ssl_context: ssl.SSLContext | None = None,
    ) -> None:
        self._token = token
        self._base_url = f"https://api.telegram.org/bot{token}/"
        self._file_base_url = f"https://api.telegram.org/file/bot{token}/"
        self._timeout_seconds = timeout_seconds
        self._ssl_context = ssl_context

    def get_me(self) -> dict[str, object]:
        return self._call("getMe")

    def get_updates(self, *, offset: int | None = None, timeout: int = 20) -> list[dict[str, object]]:
        payload: dict[str, object] = {
            "timeout": timeout,
            "allowed_updates": ["message"],
        }
        if offset is not None:
            payload["offset"] = offset
        result = self._call("getUpdates", payload)
        return result if isinstance(result, list) else []

    def send_message(self, *, chat_id: str | int, text: str) -> dict[str, object]:
        return self._call(
            "sendMessage",
            {
                "chat_id": str(chat_id),
                "text": text[:4096],
                "disable_web_page_preview": True,
            },
        )

    def send_photo(self, *, chat_id: str | int, photo_path: Path, caption: str | None = None) -> dict[str, object]:
        return self._call_multipart(
            "sendPhoto",
            fields={"chat_id": str(chat_id), "caption": caption},
            file_field="photo",
            file_path=photo_path,
        )

    def send_document(self, *, chat_id: str | int, document_path: Path, caption: str | None = None) -> dict[str, object]:
        return self._call_multipart(
            "sendDocument",
            fields={"chat_id": str(chat_id), "caption": caption},
            file_field="document",
            file_path=document_path,
        )

    def get_file(self, *, file_id: str) -> dict[str, object]:
        result = self._call("getFile", {"file_id": file_id})
        return result if isinstance(result, dict) else {}

    def download_file(self, *, file_path: str) -> bytes:
        request = Request(self._file_base_url + file_path, method="GET")
        try:
            with urlopen(request, timeout=self._timeout_seconds, context=self._ssl_context) as response:
                return response.read()
        except HTTPError as exc:  # pragma: no cover - network-dependent.
            raise RuntimeError(f"Telegram file download HTTP {exc.code} for {file_path}.") from exc
        except URLError as exc:  # pragma: no cover - network-dependent.
            raise RuntimeError(f"Telegram file download failed for {file_path}: {exc.reason}") from exc

    def _call(self, method: str, payload: dict[str, object] | None = None) -> dict[str, object] | list[dict[str, object]]:
        request_body = None if payload is None else json.dumps(payload).encode("utf-8")
        request = Request(
            self._base_url + method,
            data=request_body,
            headers={"Content-Type": "application/json; charset=utf-8"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self._timeout_seconds, context=self._ssl_context) as response:
                response_payload = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:  # pragma: no cover - network-dependent.
            raise RuntimeError(f"Telegram API HTTP {exc.code} for {method}.") from exc
        except URLError as exc:  # pragma: no cover - network-dependent.
            raise RuntimeError(f"Telegram API request failed for {method}: {exc.reason}") from exc

        if not response_payload.get("ok"):
            description = response_payload.get("description", "Unknown Telegram API error.")
            raise RuntimeError(f"Telegram API {method} failed: {description}")
        return response_payload.get("result", {})

    def _call_multipart(
        self,
        method: str,
        *,
        fields: dict[str, str | None],
        file_field: str,
        file_path: Path,
    ) -> dict[str, object]:
        boundary = f"----OpenClawBoundary{uuid.uuid4().hex}"
        content_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
        body = _encode_multipart_body(
            boundary=boundary,
            fields=fields,
            file_field=file_field,
            file_path=file_path,
            content_type=content_type,
        )
        request = Request(
            self._base_url + method,
            data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self._timeout_seconds, context=self._ssl_context) as response:
                response_payload = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:  # pragma: no cover - network-dependent.
            raise RuntimeError(f"Telegram API HTTP {exc.code} for {method}.") from exc
        except URLError as exc:  # pragma: no cover - network-dependent.
            raise RuntimeError(f"Telegram API request failed for {method}: {exc.reason}") from exc

        if not response_payload.get("ok"):
            description = response_payload.get("description", "Unknown Telegram API error.")
            raise RuntimeError(f"Telegram API {method} failed: {description}")
        result = response_payload.get("result", {})
        return result if isinstance(result, dict) else {}


class TelegramCommandProcessor:
    def __init__(
        self,
        *,
        lookup_renderer: LookupRenderer,
        board_loader: BoardLoader,
        catalog_renderer: CatalogRenderer,
        reputation_renderer: ReputationRenderer | None = None,
        natural_language_router: TelegramNaturalLanguageRouter | None = None,
        allowed_chat_ids: frozenset[str] | None = None,
        status_renderer: Callable[[], str] | None = None,
        watch_db: MonitorDatabase | None = None,
        sns_db: SnsDatabase | None = None,
        sns_buzz_fn: Callable[[str], str] | None = None,
    ) -> None:
        self._lookup_renderer = lookup_renderer
        self._board_loader = board_loader
        self._catalog_renderer = catalog_renderer
        self._reputation_renderer = reputation_renderer
        self._natural_language_router = natural_language_router
        self._allowed_chat_ids: frozenset[str] = allowed_chat_ids or frozenset()
        self._status_renderer = status_renderer
        self._watch_db = watch_db
        self._sns_db = sns_db
        self._sns_buzz_fn = sns_buzz_fn

    def is_allowed_chat(self, chat_id: str | int) -> bool:
        if not self._allowed_chat_ids:
            return True
        return str(chat_id) in self._allowed_chat_ids

    def build_reply(self, *, chat_id: str | int, text: str | None) -> str | None:
        return self.build_reply_plan(chat_id=chat_id, text=text).execute()

    def build_reply_plan(self, *, chat_id: str | int, text: str | None) -> TelegramTextReplyPlan:
        logger.info(
            "Telegram message received chat_id=%s text=%s",
            mask_identifier(chat_id),
            trim_for_log(text or "", limit=320),
        )
        if text is None:
            return TelegramTextReplyPlan(ack=None, reply=None)
        if not self.is_allowed_chat(chat_id):
            logger.warning("Rejected Telegram message from unauthorized chat_id=%s", mask_identifier(chat_id))
            return TelegramTextReplyPlan(ack=None, reply=None)

        content = text.strip()
        if not content:
            return TelegramTextReplyPlan(ack=None, reply="Empty command. Use /help to see supported commands.")

        command = _extract_command_name(content)
        remainder = _extract_command_remainder(content)
        logger.debug("Telegram command parsed command=%s remainder=%s", command, trim_for_log(remainder, limit=240))

        if command in {"/start", "/help"}:
            return TelegramTextReplyPlan(ack=None, reply=self._help_text())
        if command == "/ping":
            return TelegramTextReplyPlan(ack=None, reply="pong")
        if command == "/status":
            return TelegramTextReplyPlan(ack=None, reply=self._status_text())
        if command == "/tools":
            return TelegramTextReplyPlan(ack=None, reply=self._catalog_renderer())
        if command in PRICE_LOOKUP_COMMANDS:
            return TelegramTextReplyPlan(
                ack=build_processing_ack(text=content),
                reply=None,
                reply_factory=lambda remainder=remainder: self._handle_lookup(remainder),
            )
        if command in TREND_BOARD_COMMANDS:
            return TelegramTextReplyPlan(
                ack=build_processing_ack(text=content),
                reply=None,
                reply_factory=lambda remainder=remainder: self._handle_liquidity(remainder),
            )
        if command in PHOTO_SCAN_COMMANDS:
            return TelegramTextReplyPlan(
                ack=None,
                reply="Send a card photo with the caption /scan pokemon or /scan ws, and I will parse it and then look up the price.",
            )
        if command in REPUTATION_SNAPSHOT_COMMANDS:
            return TelegramTextReplyPlan(
                ack=build_processing_ack(text=content),
                reply=None,
                reply_factory=lambda remainder=remainder: self._handle_reputation_snapshot(remainder),
            )
        if command in WATCH_COMMANDS:
            return TelegramTextReplyPlan(
                ack="收到追蹤指令，正在設定…",
                reply=None,
                reply_factory=lambda remainder=remainder, cid=chat_id: self._handle_watch(remainder, str(cid)),
            )
        if command in WATCHLIST_COMMANDS:
            return TelegramTextReplyPlan(ack=None, reply=self._handle_watchlist())
        if command in UNWATCH_COMMANDS:
            return TelegramTextReplyPlan(
                ack=None,
                reply=self._handle_unwatch(remainder),
            )
        if command in SET_PRICE_COMMANDS:
            return TelegramTextReplyPlan(
                ack=None,
                reply=self._handle_set_price(remainder),
            )
        if command in SNS_ADD_COMMANDS:
            return TelegramTextReplyPlan(
                ack="收到 X 追蹤指令，正在設定…",
                reply=None,
                reply_factory=lambda remainder=remainder, cid=chat_id: self._handle_sns_add(remainder, str(cid)),
            )
        if command in SNS_LIST_COMMANDS:
            return TelegramTextReplyPlan(ack=None, reply=self._handle_sns_list())
        if command in SNS_DELETE_COMMANDS:
            return TelegramTextReplyPlan(ack=None, reply=self._handle_sns_delete(remainder))
        if command in SNS_BUZZ_COMMANDS:
            return TelegramTextReplyPlan(
                ack="收到，正在抓取 X 熱門討論並交給 LLM 整理…",
                reply=None,
                reply_factory=lambda remainder=remainder: self._handle_sns_buzz(remainder),
            )
        if not content.startswith("/"):
            intent = self._route_natural_language(content)
            natural_language_plan = self._build_natural_language_reply_plan(intent, chat_id=chat_id)
            if natural_language_plan is not None:
                return natural_language_plan

        logger.info("Telegram unknown command command=%s", command)
        return TelegramTextReplyPlan(
            ack=None,
            reply="Unknown command. Use /help, /price, /trend, /snapshot, or send a photo with /scan. You can also ask in natural language.",
        )

    def _handle_lookup(self, raw: str) -> str:
        try:
            query = parse_lookup_command(raw)
        except ValueError as exc:
            logger.warning("Telegram lookup parse failed raw=%s error=%s", trim_for_log(raw), exc)
            return f"{exc}\nExample: /price pokemon | Pikachu ex | 132/106 | SAR | sv08"

        try:
            logger.info(
                "Telegram lookup parsed game=%s name=%s card_number=%s rarity=%s set_code=%s",
                query.game,
                query.name,
                query.card_number,
                query.rarity,
                query.set_code,
            )
            return self._lookup_renderer(query)
        except Exception as exc:  # pragma: no cover - source/network-dependent.
            logger.exception("Telegram lookup failed game=%s name=%s", query.game, query.name)
            return f"Lookup failed: {exc}"

    def _handle_liquidity(self, raw: str) -> str:
        parts = [part for part in raw.split() if part]
        if not parts:
            return "Specify a game, for example: /trend pokemon"

        game = parts[0].lower()
        if game not in {"pokemon", "ws"}:
            return "Unsupported game. Use pokemon or ws."

        limit = 5
        if len(parts) >= 2 and parts[1].isdigit():
            limit = max(1, min(10, int(parts[1])))

        try:
            board = next(board for board in self._board_loader() if board.game == game)
        except StopIteration:
            logger.warning("Telegram liquidity board unavailable game=%s", game)
            return f"No liquidity board is available for {game}."
        except Exception as exc:  # pragma: no cover - source/network-dependent.
            logger.exception("Telegram liquidity load failed game=%s", game)
            return f"Trend board failed: {exc}"

        logger.info("Telegram liquidity board loaded game=%s limit=%s items=%s", game, limit, len(board.items))
        return format_liquidity_board(board, limit=limit)

    def _handle_reputation_snapshot(self, raw: str) -> str:
        return self.build_reputation_delivery(raw).summary_text

    def _handle_watch(self, raw: str, chat_id: str) -> str:
        if self._watch_db is None:
            return "追蹤功能尚未啟用（watch_db 未設定）。"
        try:
            query, threshold = parse_watch_command(raw)
        except ValueError as exc:
            return f"{exc}\n格式範例：/watch 想いが重なる場所で 初音ミク SSP on 300000"
        watch_id = sha1(f"{chat_id}|{query}".encode()).hexdigest()[:16]
        watch = MercariWatch(
            watch_id=watch_id,
            query=query,
            price_threshold_jpy=threshold,
            enabled=True,
            chat_id=chat_id,
            last_checked_at=None,
            created_at="",
            updated_at="",
        )
        self._watch_db.add_mercari_watch(watch)
        logger.info("Watch added watch_id=%s query=%s threshold=%d chat_id=%s", watch_id, query, threshold, chat_id)
        return (
            f"已新增追蹤\n"
            f"ID: {watch_id}\n"
            f"關鍵字：{query}\n"
            f"價格上限：¥{threshold:,}\n"
            f"將每分鐘在 Mercari 搜尋，發現新商品時通知你。"
        )

    def _handle_watchlist(self) -> str:
        if self._watch_db is None:
            return "追蹤功能尚未啟用（watch_db 未設定）。"
        watches = self._watch_db.list_mercari_watchlist()
        if not watches:
            return "目前沒有任何追蹤項目。\n使用 /watch 關鍵字 on 價格 來新增。"
        lines = [f"目前追蹤清單（共 {len(watches)} 項）："]
        for w in watches:
            status = "✓ 啟用" if w.enabled else "✗ 停用"
            checked = _format_local_time(w.last_checked_at) if w.last_checked_at else "尚未檢查"
            lines.append(
                f"\n[{w.watch_id}] {status}\n"
                f"  關鍵字：{w.query}\n"
                f"  上限：¥{w.price_threshold_jpy:,}\n"
                f"  最後檢查：{checked}"
            )
        lines.append("\n/unwatch <ID> 可移除追蹤")
        return "\n".join(lines)

    def _handle_unwatch(self, raw: str) -> str:
        if self._watch_db is None:
            return "追蹤功能尚未啟用（watch_db 未設定）。"
        watch_id = raw.strip()
        if not watch_id:
            return "請提供追蹤 ID。格式：/unwatch <ID>\n可用 /watchlist 查看所有 ID。"
        deleted = self._watch_db.delete_mercari_watch(watch_id)
        if deleted:
            logger.info("Watch deleted watch_id=%s", watch_id)
            return f"已移除追蹤 [{watch_id}]"
        return f"找不到追蹤 ID [{watch_id}]，請用 /watchlist 確認。"

    def _handle_set_price(self, raw: str) -> str:
        if self._watch_db is None:
            return "追蹤功能尚未啟用（watch_db 未設定）。"
        try:
            watch_id, new_price = parse_set_price_command(raw)
        except ValueError as exc:
            return f"{exc}\n格式範例：/setprice abc12345 50000"
        watch = self._watch_db.get_mercari_watch(watch_id)
        if watch is None:
            return f"找不到追蹤 ID [{watch_id}]，請用 /watchlist 確認。"
        old_price = watch.price_threshold_jpy
        self._watch_db.update_mercari_watch(watch_id, price_threshold_jpy=new_price)
        logger.info("Watch price updated watch_id=%s old=%d new=%d", watch_id, old_price, new_price)
        return (
            f"已更新追蹤目標價\n"
            f"ID: {watch_id}\n"
            f"關鍵字：{watch.query}\n"
            f"價格上限：¥{old_price:,} → ¥{new_price:,}"
        )

    def build_reputation_delivery(self, raw: str) -> TelegramReputationDelivery:
        if self._reputation_renderer is None:
            return TelegramReputationDelivery(
                summary_text="Reputation snapshot integration is not configured in this OpenClaw runtime."
            )

        try:
            query = parse_reputation_snapshot_command(raw)
        except ValueError as exc:
            return TelegramReputationDelivery(
                summary_text=f"{exc}\nExample: /snapshot https://jp.mercari.com/item/m123456789"
            )

        try:
            rendered = self._reputation_renderer(query)
            if isinstance(rendered, TelegramReputationDelivery):
                return rendered
            return TelegramReputationDelivery(summary_text=str(rendered))
        except Exception as exc:  # pragma: no cover - network-dependent.
            logger.exception("Telegram reputation snapshot failed query_url=%s", query.query_url)
            return TelegramReputationDelivery(summary_text=f"Snapshot failed: {exc}")

    def _build_natural_language_reply_plan(
        self,
        intent: TelegramNaturalLanguageIntent | None,
        *,
        chat_id: str | int = "",
    ) -> TelegramTextReplyPlan | None:
        if intent is None:
            return None
        if intent.intent == "help":
            logger.info("Telegram natural-language routed intent=help")
            return TelegramTextReplyPlan(ack=None, reply=self._help_text())
        if intent.intent == "status":
            logger.info("Telegram natural-language routed intent=status")
            return TelegramTextReplyPlan(ack=None, reply=self._status_text())
        if intent.intent == "tools":
            logger.info("Telegram natural-language routed intent=tools")
            return TelegramTextReplyPlan(ack=None, reply=self._catalog_renderer())
        if intent.intent == "scan_help":
            logger.info("Telegram natural-language routed intent=scan_help")
            return TelegramTextReplyPlan(
                ack=None,
                reply="Send a card photo with the caption /scan pokemon or /scan ws, and I will parse it and then look up the price.",
            )
        if intent.intent == "trend_board":
            if intent.game not in {"pokemon", "ws"}:
                return TelegramTextReplyPlan(
                    ack=None,
                    reply="I understood that you want the hot board, but I still need the game: pokemon or ws.",
                )
            limit = 5 if intent.limit is None else max(1, min(10, intent.limit))
            logger.info(
                "Telegram natural-language routed intent=trend_board game=%s limit=%s confidence=%s",
                intent.game,
                limit,
                intent.confidence,
            )
            return TelegramTextReplyPlan(
                ack=f"已理解查詢內容，相當於 /trend {intent.game} {limit}，開始整理資料。",
                reply=None,
                reply_factory=lambda game=intent.game, limit=limit: self._handle_liquidity(f"{game} {limit}"),
            )
        if intent.intent == "lookup_card":
            if intent.game not in {"pokemon", "ws"}:
                return TelegramTextReplyPlan(
                    ack=None,
                    reply="I understood that you want a card lookup, but I still need the game: pokemon or ws.",
                )
            resolved_name = intent.name or intent.card_number
            if not resolved_name:
                return TelegramTextReplyPlan(
                    ack=None,
                    reply="I understood that you want a card lookup, but I still need the card name.",
                )
            query = TelegramLookupQuery(
                game=intent.game,
                name=resolved_name,
                card_number=intent.card_number,
                rarity=intent.rarity,
                set_code=intent.set_code,
            )
            logger.info(
                "Telegram natural-language routed intent=lookup_card game=%s name=%s card_number=%s rarity=%s set_code=%s confidence=%s",
                query.game,
                query.name,
                query.card_number,
                query.rarity,
                query.set_code,
                intent.confidence,
            )
            try:
                return TelegramTextReplyPlan(
                    ack=f"已理解查詢內容，相當於 {_format_lookup_ack_command(query)}，開始查價。",
                    reply=None,
                    reply_factory=lambda query=query: self._lookup_renderer(query),
                )
            except Exception as exc:  # pragma: no cover - source/network-dependent.
                logger.exception("Telegram natural-language lookup failed game=%s name=%s", query.game, query.name)
                return TelegramTextReplyPlan(
                    ack=None,
                    reply=f"Lookup failed: {exc}",
                )
        if intent.intent == "add_watch":
            if not intent.watch_query or not intent.watch_price_threshold:
                return TelegramTextReplyPlan(
                    ack=None,
                    reply="請提供追蹤關鍵字和價格上限。例如：追蹤 初音ミク SSP 5萬以下",
                )
            q = intent.watch_query
            p = intent.watch_price_threshold
            logger.info(
                "Telegram natural-language routed intent=add_watch watch_query=%s watch_price_threshold=%d confidence=%s",
                q, p, intent.confidence,
            )
            return TelegramTextReplyPlan(
                ack=f"已理解查詢內容，相當於 /watch {q} on {p}，開始設定。",
                reply=None,
                reply_factory=lambda q=q, p=p, cid=chat_id: self._handle_watch(f"{q} on {p}", str(cid)),
            )
        if intent.intent == "list_watches":
            logger.info("Telegram natural-language routed intent=list_watches confidence=%s", intent.confidence)
            return TelegramTextReplyPlan(ack=None, reply=self._handle_watchlist())
        if intent.intent == "remove_watch":
            if not intent.watch_id:
                return TelegramTextReplyPlan(
                    ack=None,
                    reply="請提供要取消的追蹤 ID。例如：取消追蹤 abc12345\n可用 /watchlist 查看所有 ID。",
                )
            wid = intent.watch_id
            logger.info(
                "Telegram natural-language routed intent=remove_watch watch_id=%s confidence=%s",
                wid, intent.confidence,
            )
            return TelegramTextReplyPlan(
                ack=f"已理解查詢內容，相當於 /unwatch {wid}，正在移除。",
                reply=None,
                reply_factory=lambda wid=wid: self._handle_unwatch(wid),
            )
        if intent.intent == "update_watch_price":
            if not intent.watch_id or not intent.watch_price_threshold:
                return TelegramTextReplyPlan(
                    ack=None,
                    reply="請提供追蹤 ID 和新的目標價。例如：把 abc12345 改成 4萬\n可用 /watchlist 查看所有 ID。",
                )
            wid = intent.watch_id
            p = intent.watch_price_threshold
            logger.info(
                "Telegram natural-language routed intent=update_watch_price watch_id=%s new_price=%d confidence=%s",
                wid, p, intent.confidence,
            )
            return TelegramTextReplyPlan(
                ack=f"已理解查詢內容，相當於 /setprice {wid} {p}，正在更新。",
                reply=None,
                reply_factory=lambda wid=wid, p=p: self._handle_set_price(f"{wid} {p}"),
            )
        if intent.intent == "reputation_snapshot":
            if not intent.query_url:
                return TelegramTextReplyPlan(
                    ack=None,
                    reply="請提供商品或賣家的網址。\n例如：查詢信用 https://jp.mercari.com/item/m123456789",
                )
            url = intent.query_url
            logger.info(
                "Telegram natural-language routed intent=reputation_snapshot query_url=%s confidence=%s",
                trim_for_log(url, limit=240),
                intent.confidence,
            )
            return TelegramTextReplyPlan(
                ack=f"已理解查詢內容，相當於 /snapshot {url}，正在建立信譽快照…",
                reply=None,
                reputation_delivery_factory=lambda url=url: self.build_reputation_delivery(url),
            )

        # ── SNS intents ────────────────────────────────────────────────────────
        if intent.intent == "sns_add_account":
            if not intent.sns_handle:
                return TelegramTextReplyPlan(
                    ack=None,
                    reply="請告訴我要追蹤哪個 X 帳號，例如：追蹤 @elonmusk",
                )
            handle = intent.sns_handle
            logger.info("Telegram NL routed intent=sns_add_account handle=%s", handle)
            cid = str(chat_id)
            return TelegramTextReplyPlan(
                ack=f"已理解：相當於 /snsadd @{handle}，正在新增 X 追蹤…",
                reply=None,
                reply_factory=lambda h=handle, c=cid: self._handle_sns_add(f"@{h}", c),
            )
        if intent.intent == "sns_add_keyword":
            if not intent.sns_keyword:
                return TelegramTextReplyPlan(ack=None, reply="請提供 X 關鍵字，例如：監控關鍵字 機動戰士")
            kw = intent.sns_keyword
            logger.info("Telegram NL routed intent=sns_add_keyword keyword=%s", kw)
            cid = str(chat_id)
            return TelegramTextReplyPlan(
                ack=f"已理解：相當於 /snsadd keyword:{kw}",
                reply=None,
                reply_factory=lambda k=kw, c=cid: self._handle_sns_add(f"keyword:{k}", c),
            )
        if intent.intent == "sns_list":
            logger.info("Telegram NL routed intent=sns_list")
            return TelegramTextReplyPlan(ack=None, reply=self._handle_sns_list())
        if intent.intent == "sns_delete":
            target = (intent.sns_handle and f"@{intent.sns_handle}") or intent.watch_id
            if not target:
                return TelegramTextReplyPlan(
                    ack=None,
                    reply="請告訴我要移除哪個 X 帳號或規則 ID，例如：刪除追蹤 @elonmusk",
                )
            logger.info("Telegram NL routed intent=sns_delete target=%s", target)
            return TelegramTextReplyPlan(
                ack=f"已理解：相當於 /snsdelete {target}",
                reply=None,
                reply_factory=lambda t=target: self._handle_sns_delete(t),
            )
        if intent.intent == "sns_buzz":
            q = intent.sns_buzz_query
            if not q:
                return TelegramTextReplyPlan(
                    ack=None,
                    reply="請告訴我關鍵字，例如：整理一下 amd 最近的熱門討論",
                )
            logger.info("Telegram NL routed intent=sns_buzz query=%s", q)
            return TelegramTextReplyPlan(
                ack=f"已理解：相當於 /snsbuzz {q}，正在抓 Reddit 熱門討論並交給 LLM 整理…",
                reply=None,
                reply_factory=lambda x=q: self._handle_sns_buzz(x),
            )

        return None

    def _handle_sns_add(self, raw: str, chat_id: str) -> str:
        """Handle /snsadd command to add an X (Twitter) account, keyword, or trend watch."""
        if self._sns_db is None:
            return "SNS 監控尚未啟用（sns_db 未設定）。"
        from sns_monitor.models import AccountWatch, KeywordWatch, TrendWatch
        from sns_monitor.storage import SnsDatabase

        try:
            raw = raw.strip()
            if not raw:
                return "用法：/snsadd @username 或 /snsadd keyword:搜尋詞 或 /snsadd trend:trending"

            # Parse: "@username" or "keyword:xxx" or "trend:xxx"
            if raw.startswith("@"):
                # Account watch
                screen_name = raw.lstrip("@").split()[0]
                rule_id = SnsDatabase._watch_rule_id("account", screen_name)
                rule = AccountWatch(
                    rule_id=rule_id,
                    screen_name=screen_name,
                    user_id=None,
                    label=f"@{screen_name}",
                    enabled=True,
                    schedule_minutes=15,
                    chat_id=chat_id,
                    last_checked_at=None,
                )
                self._sns_db.save_watch_rule(rule)
                logger.info("SNS account watch added screen_name=%s chat_id=%s", screen_name, chat_id)
                return f"✓ 已新增 X 帳號追蹤：@{screen_name}\nID: {rule_id[:8]}…"
            elif raw.startswith("keyword:"):
                # Keyword watch
                query = raw[8:].strip()
                if not query:
                    return "請提供搜尋關鍵字。例如：/snsadd keyword:機動戰士"
                rule_id = SnsDatabase._watch_rule_id("keyword", query)
                rule = KeywordWatch(
                    rule_id=rule_id,
                    query=query,
                    label=f'"{query}"',
                    enabled=True,
                    schedule_minutes=30,
                    chat_id=chat_id,
                    last_checked_at=None,
                )
                self._sns_db.save_watch_rule(rule)
                logger.info("SNS keyword watch added query=%s chat_id=%s", query, chat_id)
                return f'✓ 已新增 X 關鍵字追蹤："{query}"\nID: {rule_id[:8]}…'
            elif raw.startswith("trend:"):
                # Trend watch
                category = raw[6:].strip()
                if category not in {"trending", "for-you", "news", "sports", "entertainment"}:
                    return "不支援的分類。請使用：trending, for-you, news, sports, 或 entertainment"
                rule_id = SnsDatabase._watch_rule_id("trend", category)
                rule = TrendWatch(
                    rule_id=rule_id,
                    category=category,
                    label=f"Trend: {category}",
                    enabled=True,
                    schedule_minutes=60,
                    chat_id=chat_id,
                    last_checked_at=None,
                )
                self._sns_db.save_watch_rule(rule)
                logger.info("SNS trend watch added category=%s chat_id=%s", category, chat_id)
                return f"✓ 已新增 X 熱門話題追蹤：{category}\nID: {rule_id[:8]}…"
            else:
                return "不認識的格式。用法：/snsadd @username 或 /snsadd keyword:搜尋詞 或 /snsadd trend:trending"
        except Exception as exc:
            logger.exception("SNS add failed raw=%s chat_id=%s", raw, chat_id)
            return f"新增失敗：{exc}"

    def _handle_sns_list(self) -> str:
        """Handle /snslist command to list all SNS watch rules."""
        if self._sns_db is None:
            return "SNS 監控尚未啟用（sns_db 未設定）。"
        try:
            rules = self._sns_db.list_watch_rules()
            if not rules:
                return "尚無 SNS 監控規則。\n用法：/snsadd @username"

            lines = [f"📋 SNS 監控規則 ({len(rules)} 項)："]
            for rule in rules:
                status = "✓" if rule.enabled else "✗"
                if rule.__class__.__name__ == "AccountWatch":
                    info = f"@{rule.screen_name}"
                elif rule.__class__.__name__ == "KeywordWatch":
                    info = f'"{rule.query}"'
                elif rule.__class__.__name__ == "TrendWatch":
                    info = f"Trend:{rule.category}"
                else:
                    info = "Unknown"
                lines.append(f"  {status} {info} ({rule.rule_id[:8]}…)")

            return "\n".join(lines)
        except Exception as exc:
            logger.exception("SNS list failed")
            return f"列表失敗：{exc}"

    def _handle_sns_delete(self, raw: str) -> str:
        """Handle /snsdelete to remove an SNS rule by rule_id, @handle, or keyword:xxx."""
        if self._sns_db is None:
            return "SNS 監控尚未啟用（sns_db 未設定）。"
        try:
            target = raw.strip()
            if not target:
                return "請提供 @帳號 或規則 ID。例如：/snsdelete @elonmusk 或 /snsdelete abc12345"

            rule_id = self._resolve_sns_rule_id(target)
            if rule_id is None:
                return f"找不到對應的 SNS 規則：{target}"

            label = self._describe_sns_rule(rule_id)
            found = self._sns_db.delete_watch_rule(rule_id)
            if found:
                logger.info("SNS rule deleted rule_id=%s target=%s", rule_id, target)
                return f"✓ 已刪除 SNS 監控：{label}"
            return f"找不到規則 {target}"
        except Exception as exc:
            logger.exception("SNS delete failed raw=%s", raw)
            return f"刪除失敗：{exc}"

    def _resolve_sns_rule_id(self, target: str) -> str | None:
        """Resolve a user-supplied target to a SNS rule_id.

        Accepts: a hex rule_id prefix, '@handle' for account watches,
        or 'keyword:xxx' for keyword watches.
        """
        if self._sns_db is None:
            return None
        cleaned = target.strip()
        if not cleaned:
            return None

        rules = list(self._sns_db.list_watch_rules())

        # 1) Exact / prefix match on rule_id
        for rule in rules:
            if rule.rule_id == cleaned or rule.rule_id.startswith(cleaned):
                return rule.rule_id

        # 2) @handle → account watch
        if cleaned.startswith("@"):
            handle = cleaned.lstrip("@").lower()
            for rule in rules:
                if getattr(rule, "screen_name", "").lower() == handle:
                    return rule.rule_id
            return None

        # 3) keyword:xxx → keyword watch
        if cleaned.lower().startswith("keyword:"):
            query = cleaned.split(":", 1)[1].strip().lower()
            for rule in rules:
                if getattr(rule, "query", "").lower() == query:
                    return rule.rule_id
            return None

        # 4) Bare token: try as a handle (without @) since users often forget it
        bare = cleaned.lstrip("@").lower()
        for rule in rules:
            if getattr(rule, "screen_name", "").lower() == bare:
                return rule.rule_id

        return None

    def _describe_sns_rule(self, rule_id: str) -> str:
        if self._sns_db is None:
            return rule_id[:8]
        for rule in self._sns_db.list_watch_rules():
            if rule.rule_id != rule_id:
                continue
            screen_name = getattr(rule, "screen_name", None)
            if screen_name:
                return f"@{screen_name}"
            query = getattr(rule, "query", None)
            if query:
                return f"關鍵字「{query}」"
            return rule.rule_id[:8]
        return rule_id[:8]

    def _handle_sns_buzz(self, raw: str) -> str:
        """Handle /snsbuzz <keyword> command: summarize X buzz on a topic via LLM."""
        if self._sns_buzz_fn is None:
            return "SNS Buzz 功能尚未啟用（需要 X 客戶端與 LLM endpoint）。"
        query = raw.strip()
        if not query:
            return "請提供關鍵字。例如：/snsbuzz amd"
        try:
            return self._sns_buzz_fn(query)
        except Exception as exc:
            logger.exception("SNS buzz failed query=%s", query)
            return f"熱門整理失敗：{exc}"

    def _route_natural_language(self, text: str) -> TelegramNaturalLanguageIntent | None:
        if self._natural_language_router is not None:
            try:
                intent = self._natural_language_router.route(text)
            except Exception:
                logger.exception("Telegram natural-language router failed, falling back to keyword rules text=%s", trim_for_log(text, limit=240))
            else:
                # LLM responded successfully — trust its answer, even if "unknown"
                if intent is not None and intent.intent != "unknown":
                    return intent
                logger.debug("Telegram natural-language LLM returned unknown, not overriding with keyword rules")
                return None

        # LLM not configured, or LLM threw an exception — use keyword fallback as last resort
        fallback_intent = fallback_route_telegram_natural_language(text)
        if fallback_intent is not None and fallback_intent.intent != "unknown":
            logger.info(
                "Telegram natural-language fallback (no LLM) intent=%s confidence=%s",
                fallback_intent.intent,
                fallback_intent.confidence,
            )
            return fallback_intent
        return None

    def _status_text(self) -> str:
        if self._status_renderer is not None:
            return self._status_renderer()
        return "OpenClaw Telegram bot is online."

    def _help_text(self) -> str:
        return "\n".join(
            [
                "OpenClaw Telegram bot",
                "/ping",
                "/status",
                "/tools",
                "/price pokemon Pikachu ex",
                "/price pokemon | Pikachu ex | 132/106 | SAR | sv08",
                "/price ws | Hatsune Miku | PJS/S91-T51 | TD | pjs",
                "/trend pokemon",
                "/trend ws 5",
                "/hot pokemon",
                "/liquidity ws 5",
                "/snapshot https://jp.mercari.com/item/m123456789",
                "Send a photo with caption: /scan pokemon",
                "--- Mercari 追蹤 ---",
                "/watch 想いが重なる場所で 初音ミク SSP on 300000",
                "/watchlist",
                "/unwatch <ID>",
                "/setprice <ID> <新價格>",
                "--- SNS (X/Twitter) 監控 ---",
                "/snsadd @username",
                "/snsadd keyword:搜詞",
                "/snsadd trend:trending",
                "/snslist",
                "/snsdelete <rule_id>",
                "/snsbuzz amd",
                "You can also ask things like: 幫我查 pokemon Pikachu ex 132/106",
                "Or: pokemon 熱門前 5",
            ]
        )


def _format_local_time(ts: str) -> str:
    """Convert a UTC ISO timestamp from DB to local timezone for display."""
    try:
        return datetime.fromisoformat(ts).astimezone().strftime("%Y-%m-%d %H:%M")
    except Exception:
        return ts[:16].replace("T", " ")


def parse_watch_command(raw: str) -> tuple[str, int]:
    """Parse '/watch <query> on <price>' → (query, price_threshold_jpy).

    Accepts both bracket notation from the docs:
      /watch [商品名] on [300000]
    and plain:
      /watch 商品名 on 300000
    """
    body = raw.strip().strip("[]")
    # Remove surrounding brackets from whole body if present
    lower = body.lower()
    # Find last occurrence of " on " to split
    sep = " on "
    idx = lower.rfind(sep)
    if idx == -1:
        raise ValueError("格式錯誤：需要 'on 價格' 部分。")
    query_part = body[:idx].strip().strip("[]").strip()
    price_part = body[idx + len(sep):].strip().strip("[]").strip()
    if not query_part:
        raise ValueError("關鍵字不可為空。")
    try:
        threshold = int(price_part.replace(",", "").replace("，", ""))
    except ValueError:
        raise ValueError(f"價格格式錯誤：'{price_part}'，請填入整數日幣金額。")
    if threshold <= 0:
        raise ValueError("價格上限必須大於 0。")
    return query_part, threshold


def parse_set_price_command(raw: str) -> tuple[str, int]:
    """Parse '/setprice <watch_id> <new_price>' → (watch_id, new_price_jpy)."""
    parts = raw.strip().split()
    if len(parts) < 2:
        raise ValueError("格式錯誤：需要追蹤 ID 和新價格。")
    watch_id = parts[0]
    price_str = "".join(parts[1:]).replace(",", "").replace("，", "")
    try:
        threshold = int(price_str)
    except ValueError:
        raise ValueError(f"價格格式錯誤：'{parts[1]}'，請填入整數日幣金額。")
    if threshold <= 0:
        raise ValueError("價格上限必須大於 0。")
    return watch_id, threshold


def parse_lookup_command(raw: str) -> TelegramLookupQuery:
    body = raw.strip()
    if not body:
        raise ValueError("Lookup command requires at least a game and a name.")

    if "|" in body:
        parts = [part.strip() for part in body.split("|")]
        if len(parts) < 2 or not parts[0] or not parts[1]:
            raise ValueError("Pipe format requires at least game and name.")
        game = parts[0].lower()
        name = parts[1]
        if game not in {"pokemon", "ws"}:
            raise ValueError("Unsupported game. Use pokemon or ws.")
        return TelegramLookupQuery(
            game=game,
            name=name,
            card_number=_value_or_none(parts, 2),
            rarity=_value_or_none(parts, 3),
            set_code=_value_or_none(parts, 4),
        )

    tokens = body.split()
    if len(tokens) < 2:
        raise ValueError("Lookup command requires at least game and name.")
    game = tokens[0].lower()
    if game not in {"pokemon", "ws"}:
        raise ValueError("Unsupported game. Use pokemon or ws.")
    name = " ".join(tokens[1:]).strip()
    if not name:
        raise ValueError("Lookup name cannot be empty.")
    return TelegramLookupQuery(game=game, name=name)


def parse_reputation_snapshot_command(raw: str) -> TelegramReputationQuery:
    query_url = raw.strip()
    if not query_url:
        raise ValueError("Snapshot command requires a Mercari item or profile URL.")
    return TelegramReputationQuery(query_url=query_url)


def format_liquidity_board(board: HotCardBoard, *, limit: int = 5) -> str:
    lines = [board.label]
    for item in board.items[:limit]:
        price_text = "price n/a" if item.price_jpy is None else format_jpy(item.price_jpy)
        bid_text = "bid n/a" if item.best_bid_jpy is None else f"bid {format_jpy(item.best_bid_jpy)}"
        ask_text = "ask n/a" if item.best_ask_jpy is None else f"ask {format_jpy(item.best_ask_jpy)}"
        ratio_text = "ratio n/a" if item.bid_ask_ratio is None else f"ratio {item.bid_ask_ratio:.0%}"
        score_text = f"liq {item.hot_score:.2f}"
        attention_text = f"attn {item.attention_score:.2f}"
        momentum_text = "" if item.momentum_boost_score <= 0 else f" | boost {item.momentum_boost_score:.2f}"
        meta = " / ".join(
            value
            for value in (
                item.card_number or "",
                item.rarity or "",
                item.set_code or "",
                "buy-up" if item.buy_signal_label == "priceup" else "",
                "graded" if item.is_graded else "",
            )
            if value
        )
        lines.append(f"{item.rank}. {item.title}")
        lines.append(f"   {price_text} | {bid_text} | {ask_text} | {ratio_text}")
        lines.append(f"   support {item.buy_support_score:.2f}{momentum_text} | {score_text} | {attention_text}")
        if item.social_post_count is not None:
            lines.append(f"   sns {item.social_post_count} posts / {item.social_engagement_count or 0} engagement")
        if meta:
            lines.append(f"   {meta}")
        if item.references:
            lines.append(f"   {item.references[0].url}")
    return "\n".join(lines)


def format_photo_lookup_result(outcome: TcgImageLookupOutcome) -> str:
    if outcome.status == "unavailable":
        return "尚未設定 OCR，無法解析卡片圖片。"
    if outcome.status == "unresolved" or outcome.lookup_result is None:
        return "無法從圖片提取足夠資訊進行查價。"
    return format_lookup_result_telegram(outcome.lookup_result)


def default_lookup_renderer(db_path: str | Path | None = None) -> LookupRenderer:
    def render(query: TelegramLookupQuery) -> str:
        logger.debug(
            "Telegram lookup renderer executing game=%s name=%s card_number=%s rarity=%s set_code=%s",
            query.game,
            query.name,
            query.card_number,
            query.rarity,
            query.set_code,
        )
        result = lookup_card(
            db_path=db_path or "data/monitor.sqlite3",
            game=query.game,
            name=query.name,
            card_number=query.card_number,
            rarity=query.rarity,
            set_code=query.set_code,
            persist=False,
        )
        logger.info(
            "Telegram lookup renderer completed game=%s name=%s offers=%s fair_value=%s",
            query.game,
            query.name,
            len(result.offers),
            None if result.fair_value is None else result.fair_value.amount_jpy,
        )
        logger.debug(
            "Telegram lookup result detail game=%s name=%s notes=%s confidence=%s offers=%s",
            query.game,
            query.name,
            list(result.notes),
            None if result.fair_value is None else f"{result.fair_value.confidence:.2f}",
            [
                {
                    "source": o.source,
                    "kind": o.price_kind,
                    "jpy": o.price_jpy,
                    "num": o.attributes.get("card_number"),
                    "rarity": o.attributes.get("rarity"),
                    "score": o.score,
                    "url": o.url,
                }
                for o in result.offers[:10]
            ],
        )
        return format_lookup_result_telegram(result)

    return render


def default_photo_renderer(
    *,
    db_path: str | Path | None = None,
    tesseract_path: str | None = None,
    tessdata_dir: str | None = None,
    vision_settings: TcgVisionSettings | None = None,
) -> PhotoLookupRenderer:
    image_service = TcgImagePriceService(
        db_path=db_path,
        tesseract_path=tesseract_path,
        tessdata_dir=tessdata_dir,
        vision_settings=vision_settings,
    )

    def render(query: TelegramPhotoQuery) -> str:
        logger.info(
            "Telegram photo renderer executing chat_id=%s file_id=%s path=%s caption=%s game_hint=%s title_hint=%s",
            mask_identifier(query.chat_id),
            query.file_id,
            query.image_path,
            trim_for_log(query.caption or "", limit=200),
            query.game_hint,
            query.title_hint,
        )
        outcome = image_service.lookup_image(
            query.image_path,
            caption=query.caption,
            game_hint=query.game_hint,
            title_hint=query.title_hint,
            persist=False,
        )
        logger.info(
            "Telegram image scan result chat_id=%s file_id=%s status=%s game=%s title=%s card_number=%s rarity=%s set_code=%s extracted_lines=%s raw_text=%s",
            mask_identifier(query.chat_id),
            query.file_id,
            outcome.parsed.status,
            outcome.parsed.game,
            outcome.parsed.title,
            outcome.parsed.card_number,
            outcome.parsed.rarity,
            outcome.parsed.set_code,
            list(outcome.parsed.extracted_lines[:12]),
            trim_for_log(outcome.parsed.raw_text, limit=600),
        )
        logger.info(
            "Telegram photo renderer completed status=%s title=%s game=%s card_number=%s rarity=%s set_code=%s offers=%s warnings=%s",
            outcome.status,
            outcome.parsed.title,
            outcome.parsed.game,
            outcome.parsed.card_number,
            outcome.parsed.rarity,
            outcome.parsed.set_code,
            0 if outcome.lookup_result is None else len(outcome.lookup_result.offers),
            list(outcome.warnings),
        )
        return format_photo_lookup_result(outcome)

    return render


_BOARD_CACHE_TTL_SECONDS = 20 * 60
_board_cache_lock = threading.Lock()
_board_cache: tuple[HotCardBoard, ...] | None = None
_board_cache_ts: float = 0.0


def default_board_loader(ssl_context: ssl.SSLContext | None = None) -> tuple[HotCardBoard, ...]:
    global _board_cache, _board_cache_ts
    now = time.time()
    with _board_cache_lock:
        if _board_cache is not None and now - _board_cache_ts < _BOARD_CACHE_TTL_SECONDS:
            logger.debug("Board cache hit age_seconds=%.0f", now - _board_cache_ts)
            return _board_cache

    client = HttpClient(ssl_context=ssl_context)
    boards = TcgHotCardService(http_client=client).load_boards()
    with _board_cache_lock:
        _board_cache = boards
        _board_cache_ts = time.time()
    logger.info("Board cache refreshed boards=%s", len(boards))
    return boards


def build_processing_ack(*, text: str | None = None, has_photo: bool = False) -> str | None:
    if has_photo:
        return "收到圖片，開始解析與查價。"
    command = _extract_command_name(text)
    if command in PRICE_LOOKUP_COMMANDS:
        return "收到查價指令，開始處理。"
    if command in TREND_BOARD_COMMANDS:
        return "收到趨勢榜查詢，開始整理資料。"
    if command in REPUTATION_SNAPSHOT_COMMANDS:
        return "收到信譽快照查詢，先檢查既有 proof，必要時建立新快照。"
    return None


def run_telegram_polling(
    *,
    token: str,
    lookup_renderer: LookupRenderer,
    board_loader: BoardLoader,
    catalog_renderer: CatalogRenderer,
    photo_renderer: PhotoLookupRenderer | None = None,
    reputation_renderer: ReputationRenderer | None = None,
    natural_language_router: TelegramNaturalLanguageRouter | None = None,
    ssl_context: ssl.SSLContext | None = None,
    allowed_chat_ids: frozenset[str] | None = None,
    status_renderer: Callable[[], str] | None = None,
    watch_db: MonitorDatabase | None = None,
    sns_db: SnsDatabase | None = None,
    sns_buzz_fn: Callable[[str], str] | None = None,
    poll_timeout: int = 20,
    notify_startup: bool = False,
    drop_pending_updates: bool = True,
) -> int:
    client = TelegramBotClient(token, ssl_context=ssl_context)
    me = client.get_me()
    username = me.get("username", "<unknown>")
    logger.info(
        "Telegram polling starting username=%s notify_startup=%s drop_pending_updates=%s allowed_chats=%s",
        username,
        notify_startup,
        drop_pending_updates,
        sorted(allowed_chat_ids or []),
    )

    offset: int | None = None
    if drop_pending_updates:
        pending_updates = client.get_updates(timeout=0)
        if pending_updates:
            offset = int(pending_updates[-1]["update_id"]) + 1

    processor = TelegramCommandProcessor(
        lookup_renderer=lookup_renderer,
        board_loader=board_loader,
        catalog_renderer=catalog_renderer,
        reputation_renderer=reputation_renderer,
        natural_language_router=natural_language_router,
        allowed_chat_ids=allowed_chat_ids,
        status_renderer=status_renderer,
        watch_db=watch_db,
        sns_db=sns_db,
        sns_buzz_fn=sns_buzz_fn,
    )
    resolved_photo_renderer = photo_renderer or default_photo_renderer()

    print(f"OpenClaw Telegram bot polling as @{username}")
    if notify_startup and allowed_chat_ids:
        for cid in allowed_chat_ids:
            client.send_message(chat_id=cid, text="OpenClaw Telegram bot is online.")
            logger.info("Telegram startup notification sent chat_id=%s", mask_identifier(cid))

    try:
        while True:
            updates = client.get_updates(offset=offset, timeout=poll_timeout)
            for update in updates:
                offset = int(update["update_id"]) + 1
                message = update.get("message")
                if not isinstance(message, dict):
                    continue
                handle_telegram_message(
                    client=client,
                    processor=processor,
                    photo_renderer=resolved_photo_renderer,
                    message=message,
                )
    except KeyboardInterrupt:
        logger.info("Telegram polling stopped by KeyboardInterrupt")
        print("Telegram polling stopped.")
    return 0


def handle_telegram_message(
    *,
    client: TelegramBotClient,
    processor: TelegramCommandProcessor,
    photo_renderer: PhotoLookupRenderer,
    message: dict[str, object],
) -> tuple[str, ...]:
    chat = message.get("chat")
    if not isinstance(chat, dict):
        return ()
    chat_id = chat.get("id")
    if chat_id is None:
        return ()
    if not processor.is_allowed_chat(chat_id):
        logger.warning("Rejected Telegram message from unauthorized chat_id=%s", mask_identifier(chat_id))
        return ()

    replies: list[str] = []
    photo_items = message.get("photo")
    if isinstance(photo_items, list) and photo_items:
        ack = build_processing_ack(has_photo=True)
        if ack:
            client.send_message(chat_id=chat_id, text=ack)
            replies.append(ack)
        final_reply = _handle_photo_message(
            client=client,
            photo_renderer=photo_renderer,
            chat_id=chat_id,
            message=message,
        )
        client.send_message(chat_id=chat_id, text=final_reply)
        replies.append(final_reply)
        return tuple(replies)

    text = message.get("text")
    text_value = text if isinstance(text, str) else None
    if text_value is not None and _extract_command_name(text_value) in REPUTATION_SNAPSHOT_COMMANDS:
        if not processor.is_allowed_chat(chat_id):
            return ()
        ack = build_processing_ack(text=text_value)
        if ack:
            client.send_message(chat_id=chat_id, text=ack)
            replies.append(ack)
        delivery = processor.build_reputation_delivery(_extract_command_remainder(text_value))
        _send_reputation_delivery(client=client, chat_id=chat_id, delivery=delivery)
        replies.append(delivery.summary_text)
        return tuple(replies)

    plan = processor.build_reply_plan(chat_id=chat_id, text=text_value)
    if plan.ack:
        client.send_message(chat_id=chat_id, text=plan.ack)
        replies.append(plan.ack)

    if plan.reputation_delivery_factory is not None:
        delivery = plan.reputation_delivery_factory()
        _send_reputation_delivery(client=client, chat_id=chat_id, delivery=delivery)
        replies.append(delivery.summary_text)
        return tuple(replies)

    reply = plan.execute()
    if reply:
        logger.debug(
            "Telegram reply sending chat_id=%s text=%s",
            mask_identifier(chat_id),
            trim_for_log(reply, limit=320),
        )
        client.send_message(chat_id=chat_id, text=reply)
        replies.append(reply)
    return tuple(replies)


def send_telegram_test_message(
    *,
    token: str,
    chat_id: str,
    message: str,
    ssl_context: ssl.SSLContext | None = None,
) -> int:
    client = TelegramBotClient(token, ssl_context=ssl_context)
    logger.info("Telegram test message sending chat_id=%s text=%s", mask_identifier(chat_id), trim_for_log(message))
    client.send_message(chat_id=chat_id, text=message)
    print(f"Sent Telegram test message to chat {chat_id}.")
    return 0


def _send_reputation_delivery(
    *,
    client: TelegramBotClient,
    chat_id: str | int,
    delivery: TelegramReputationDelivery,
) -> None:
    cleanup_paths = list(delivery.cleanup_paths)
    try:
        logger.debug(
            "Telegram reputation delivery sending chat_id=%s text=%s attachments=%s",
            mask_identifier(chat_id),
            trim_for_log(delivery.summary_text, limit=320),
            [attachment.path.name for attachment in delivery.attachments],
        )
        client.send_message(chat_id=chat_id, text=delivery.summary_text)
        for attachment in delivery.attachments:
            if attachment.kind == "document":
                client.send_document(chat_id=chat_id, document_path=attachment.path, caption=attachment.caption)
            elif attachment.kind == "photo":
                client.send_photo(chat_id=chat_id, photo_path=attachment.path, caption=attachment.caption)
            else:
                logger.warning("Unknown Telegram attachment kind=%s path=%s", attachment.kind, attachment.path)
    finally:
        for path in cleanup_paths:
            try:
                path.unlink(missing_ok=True)
            except Exception:
                logger.debug("Could not remove temporary reputation artifact path=%s", path)


def _handle_photo_message(
    *,
    client: TelegramBotClient,
    photo_renderer: PhotoLookupRenderer,
    chat_id: str | int,
    message: dict[str, object],
) -> str:
    caption = message.get("caption")
    caption_text = caption if isinstance(caption, str) else None
    photo_items = message.get("photo")
    if not isinstance(photo_items, list) or not photo_items:
        return "No image was attached."

    candidates = [item for item in photo_items if isinstance(item, dict) and item.get("file_id")]
    if not candidates:
        return "Could not resolve the Telegram file metadata for this image."

    best_item = max(
        candidates,
        key=lambda item: int(item.get("file_size") or 0),
    )
    file_id = best_item.get("file_id")
    if not isinstance(file_id, str):
        return "Could not resolve the Telegram file id for this image."

    game_hint, title_hint = _parse_photo_caption_for_lookup(caption_text)

    try:
        file_info = client.get_file(file_id=file_id)
        file_path = file_info.get("file_path")
        if not isinstance(file_path, str) or not file_path:
            return "Telegram did not return a downloadable file path for this image."
        payload = client.download_file(file_path=file_path)
        suffix = Path(file_path).suffix or ".jpg"
        temp_root = Path.cwd() / ".openclaw_tmp"
        temp_root.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="wb",
            suffix=suffix,
            prefix="telegram-upload-",
            dir=temp_root,
            delete=False,
        ) as handle:
            handle.write(payload)
            local_path = Path(handle.name)
        try:
            query = TelegramPhotoQuery(
                chat_id=chat_id,
                image_path=local_path,
                caption=caption_text,
                game_hint=game_hint,
                title_hint=title_hint,
                file_id=file_id,
            )
            return photo_renderer(query)
        finally:
            try:
                local_path.unlink(missing_ok=True)
            except PermissionError:
                logger.debug("Could not remove temporary Telegram photo path=%s", local_path)
    except Exception as exc:  # pragma: no cover - network-dependent.
        logger.exception("Telegram photo handling failed chat_id=%s file_id=%s", mask_identifier(chat_id), file_id)
        return f"Image lookup failed: {exc}"


def _parse_photo_caption_for_lookup(caption: str | None) -> tuple[str | None, str | None]:
    if caption is None:
        return None, None
    content = caption.strip()
    if not content:
        return None, None
    for prefix in PHOTO_SCAN_COMMANDS:
        if content.lower().startswith(prefix):
            content = content[len(prefix):].strip()
            break
    if not content:
        return None, None
    tokens = content.split()
    if not tokens:
        return None, None
    first = tokens[0].lower()
    if first in {"pokemon", "ws"}:
        remainder = " ".join(tokens[1:]).strip()
        return first, _sanitize_image_title_hint(remainder or None)
    return None, _sanitize_image_title_hint(content)


def _format_lookup_ack_command(query: TelegramLookupQuery) -> str:
    metadata = [query.card_number, query.rarity, query.set_code]
    if any(metadata):
        parts = [query.game, query.name, *(value or "" for value in metadata)]
        return f"/price {' | '.join(parts)}"
    return f"/price {query.game} {query.name}"


def _encode_multipart_body(
    *,
    boundary: str,
    fields: dict[str, str | None],
    file_field: str,
    file_path: Path,
    content_type: str,
) -> bytes:
    body = bytearray()
    for key, value in fields.items():
        if value is None:
            continue
        body.extend(f"--{boundary}\r\n".encode("utf-8"))
        body.extend(f'Content-Disposition: form-data; name="{key}"\r\n\r\n'.encode("utf-8"))
        body.extend(str(value).encode("utf-8"))
        body.extend(b"\r\n")

    body.extend(f"--{boundary}\r\n".encode("utf-8"))
    body.extend(
        (
            f'Content-Disposition: form-data; name="{file_field}"; filename="{file_path.name}"\r\n'
            f"Content-Type: {content_type}\r\n\r\n"
        ).encode("utf-8")
    )
    body.extend(file_path.read_bytes())
    body.extend(b"\r\n")
    body.extend(f"--{boundary}--\r\n".encode("utf-8"))
    return bytes(body)


def _extract_command_name(text: str | None) -> str | None:
    if text is None:
        return None
    content = text.strip()
    if not content or not content.startswith("/"):
        return None
    command, *_ = content.split(maxsplit=1)
    return command.split("@", 1)[0].lower()


def _extract_command_remainder(text: str | None) -> str:
    if text is None:
        return ""
    content = text.strip()
    if not content:
        return ""
    _, _, remainder = content.partition(" ")
    return remainder.strip()


def _value_or_none(parts: list[str], index: int) -> str | None:
    if index >= len(parts):
        return None
    value = parts[index].strip()
    return value or None
