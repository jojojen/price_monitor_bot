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
from pathlib import Path
from typing import Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from market_monitor.http import HttpClient
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

PRICE_LOOKUP_COMMANDS = {"/lookup", "/price"}
TREND_BOARD_COMMANDS = {"/trend", "/trending", "/hot", "/heat", "/liquidity"}
PHOTO_SCAN_COMMANDS = {"/scan", "/image", "/photo"}
REPUTATION_SNAPSHOT_COMMANDS = {"/snapshot", "/proof", "/repcheck", "/reputation"}
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
        allowed_chat_id: str | None = None,
        status_renderer: Callable[[], str] | None = None,
    ) -> None:
        self._lookup_renderer = lookup_renderer
        self._board_loader = board_loader
        self._catalog_renderer = catalog_renderer
        self._reputation_renderer = reputation_renderer
        self._natural_language_router = natural_language_router
        self._allowed_chat_id = allowed_chat_id
        self._status_renderer = status_renderer

    def is_allowed_chat(self, chat_id: str | int) -> bool:
        if self._allowed_chat_id is None:
            return True
        return str(chat_id) == str(self._allowed_chat_id)

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
        if not content.startswith("/"):
            intent = self._route_natural_language(content)
            natural_language_plan = self._build_natural_language_reply_plan(intent)
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
    ) -> TelegramTextReplyPlan | None:
        if intent is None:
            return None
        if intent.intent == "help":
            logger.info("Telegram natural-language routed intent=help")
            return TelegramTextReplyPlan(ack=None, reply=self._help_text())
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
        return None

    def _route_natural_language(self, text: str) -> TelegramNaturalLanguageIntent | None:
        if self._natural_language_router is not None:
            try:
                intent = self._natural_language_router.route(text)
            except Exception:
                logger.exception("Telegram natural-language router failed text=%s", trim_for_log(text, limit=240))
            else:
                if intent is not None and intent.intent != "unknown":
                    return intent
        fallback_intent = fallback_route_telegram_natural_language(text)
        if fallback_intent is not None and fallback_intent.intent != "unknown":
            logger.info(
                "Telegram natural-language fallback intent=%s game=%s name=%s limit=%s confidence=%s",
                fallback_intent.intent,
                fallback_intent.game,
                fallback_intent.name,
                fallback_intent.limit,
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
                "You can also ask things like: 幫我查 pokemon Pikachu ex 132/106",
                "Or: pokemon 熱門前 5",
            ]
        )


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
    allowed_chat_id: str | None = None,
    status_renderer: Callable[[], str] | None = None,
    poll_timeout: int = 20,
    notify_startup: bool = False,
    drop_pending_updates: bool = True,
) -> int:
    client = TelegramBotClient(token, ssl_context=ssl_context)
    me = client.get_me()
    username = me.get("username", "<unknown>")
    logger.info(
        "Telegram polling starting username=%s notify_startup=%s drop_pending_updates=%s allowed_chat=%s",
        username,
        notify_startup,
        drop_pending_updates,
        mask_identifier(allowed_chat_id),
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
        allowed_chat_id=allowed_chat_id,
        status_renderer=status_renderer,
    )
    resolved_photo_renderer = photo_renderer or default_photo_renderer()

    print(f"OpenClaw Telegram bot polling as @{username}")
    if notify_startup and allowed_chat_id is not None:
        client.send_message(chat_id=allowed_chat_id, text="OpenClaw Telegram bot is online.")
        logger.info("Telegram startup notification sent chat_id=%s", mask_identifier(allowed_chat_id))

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
