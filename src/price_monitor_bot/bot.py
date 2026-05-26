from __future__ import annotations

import json
import logging
import mimetypes
import re
import ssl
import tempfile
import time
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

if TYPE_CHECKING:
    from sns_monitor.storage import SnsDatabase

from hashlib import sha1

from market_monitor.http import HttpClient
from market_monitor.mercari_search import MERCARI_CONDITION_LABELS
from market_monitor.storage import (
    MarketplaceWatch,
    MonitorDatabase,
    build_marketplace_watch_id,
)
from tcg_tracker.catalog import normalize_game_key, supported_game_hint
from tcg_tracker.hot_cards import HotCardBoard, TcgHotCardService
from tcg_tracker.image_lookup import (
    TcgImageLookupOutcome,
    TcgImagePriceService,
    TcgVisionSettings,
    _sanitize_image_title_hint,
)

from .commands import lookup_card
from .formatters import build_lookup_feedback_keyboard, format_jpy, format_lookup_result_telegram
from .logging_utils import mask_identifier, trim_for_log
from .natural_language import (
    TelegramNaturalLanguageIntent,
    TelegramNaturalLanguageRouter,
    fallback_route_telegram_natural_language,
    _recover_lookup_fields,
)

LookupRenderer = Callable[["TelegramLookupQuery"], "str | tuple[str, dict[str, object] | None]"]
# Photo renderer can return either a bare reply string (existing behaviour)
# or a PhotoLookupReply when the renderer also wants to install a pending
# clarification state — used by the unresolved/rejected_sanity path.
PhotoLookupRenderer = Callable[["TelegramPhotoQuery"], "str | PhotoLookupReply"]
PhotoIntentAnalyzer = Callable[["TelegramPhotoQuery"], "TelegramPhotoIntentAnalysis"]
ReputationRenderer = Callable[["TelegramReputationQuery"], object]
ResearchRenderer = Callable[["TelegramResearchQuery"], str]
OpportunityTargetRemover = Callable[[str], str]
# (selector, kind, action, names) -> reply text. kind in {"aliases","related"}, action in {"add","remove"}.
OpportunityAliasUpdater = Callable[[str, str, str, list[str]], str]
# (name) -> reply text. Marks a candidate (existing or newly created) as 🎯 Target.
OpportunityTargetPinner = Callable[[str], str]
# (selector) -> reply text. Removes the 🎯 Target flag without dismissing the candidate.
OpportunityTargetUnpinner = Callable[[str], str]
BoardLoader = Callable[[], tuple[HotCardBoard, ...]]
CatalogRenderer = Callable[[], str]
WatchlistStore = object  # MonitorDatabase or None

PRICE_LOOKUP_COMMANDS = {"/lookup", "/price"}
TREND_BOARD_COMMANDS = {"/trend", "/trending", "/hot", "/heat", "/liquidity"}
PHOTO_SCAN_COMMANDS = {"/scan", "/image", "/photo"}
REPUTATION_SNAPSHOT_COMMANDS = {"/snapshot", "/proof", "/repcheck", "/reputation"}
WEB_RESEARCH_COMMANDS = {"/search", "/research", "/web"}
WATCH_COMMANDS = {"/watch"}
WATCHLIST_COMMANDS = {"/watchlist", "/watches"}
UNWATCH_COMMANDS = {"/unwatch", "/stopwatch"}

# Display name + emoji per marketplace source (kept here so the UI layer
# doesn't need to import from market_monitor). Adding a new source = one line.
_MARKETPLACE_SOURCE_DISPLAY: dict[str, tuple[str, str]] = {
    "mercari": ("Mercari", "🛒"),
    "rakuma": ("Rakuma", "🟣"),
    "yuyutei": ("遊々亭", "📚"),
}

# The default set of markets a new /watch covers when the user doesn't pin
# the subset. Future markets get added here so existing users automatically
# pick them up.
DEFAULT_MARKETS: tuple[str, ...] = ("mercari", "rakuma", "yuyutei")
# Aliases users may type for each canonical market id.
_MARKET_ALIASES: dict[str, str] = {
    "mercari": "mercari", "メルカリ": "mercari",
    "rakuma": "rakuma", "ラクマ": "rakuma", "フリル": "rakuma", "fril": "rakuma",
    "yuyutei": "yuyutei", "遊々亭": "yuyutei", "yuyu亭": "yuyutei", "yuyu-tei": "yuyutei",
}


def _marketplace_source_display(source: str) -> tuple[str, str]:
    return _MARKETPLACE_SOURCE_DISPLAY.get(source, (source.capitalize(), "📦"))


def _condition_ids_from_options(source_options: dict[str, object]) -> tuple[int, ...]:
    """Extract Mercari condition_ids from a per-market options dict."""
    raw = source_options.get("condition_ids") if source_options else None
    if isinstance(raw, (list, tuple)):
        out: list[int] = []
        for value in raw:
            try:
                out.append(int(value))
            except (TypeError, ValueError):
                continue
        if out:
            return tuple(out)
    return (1, 2, 3)


def _normalize_markets(values: object) -> tuple[str, ...]:
    """Canonicalise a list/tuple of market names. Unknown names are dropped;
    duplicates collapsed; order preserved. Empty input returns ()."""
    if not isinstance(values, (list, tuple)):
        return ()
    out: list[str] = []
    seen: set[str] = set()
    for raw in values:
        if not isinstance(raw, str):
            continue
        normal = _MARKET_ALIASES.get(raw.strip().lower())
        if normal is None or normal in seen:
            continue
        seen.add(normal)
        out.append(normal)
    return tuple(out)
SET_PRICE_COMMANDS = {"/setprice", "/updatewatch"}
SNS_ADD_COMMANDS = {"/snsadd", "/sns_add"}
SNS_LIST_COMMANDS = {"/snslist", "/sns_list"}
SNS_DELETE_COMMANDS = {"/snsdelete", "/sns_delete"}
SNS_BUZZ_COMMANDS = {"/snsbuzz", "/sns_buzz"}
KNOWLEDGE_COMMANDS = {"/knowledge", "/kb"}
HUNT_COMMANDS = {"/hunt", "/opportunity"}
HEAVY_COMMANDS = PRICE_LOOKUP_COMMANDS | TREND_BOARD_COMMANDS | REPUTATION_SNAPSHOT_COMMANDS | WEB_RESEARCH_COMMANDS

logger = logging.getLogger(__name__)
PHOTO_CLARIFICATION_TTL_SECONDS = 15 * 60
TEXT_CLARIFICATION_TTL_SECONDS = 15 * 60
TEXT_AMBIGUITY_CONFIDENCE_THRESHOLD = 0.55

LIST_VIEW_PAGE_SIZE = 5
LIST_VIEW_MODE_READ = "r"
LIST_VIEW_MODE_EDIT = "e"


@dataclass(frozen=True, slots=True)
class _ListRow:
    """One row of a paginated /snslist // /watchlist // /hunt view.

    `id` lands in the delete-button's callback_data, so it must be short
    enough to keep `del:<kind>:<id>` under Telegram's 64-byte limit.
    `text` is the rendered multi-line block for read mode; `short_label`
    is the truncated label shown on the delete button in edit mode.
    `extra_buttons` is appended next to the delete button on the same row
    (used by /watchlist to add a "🎛 設定狀態" button).
    `label_button` when set is inserted as a full-width row directly above
    the action buttons in edit mode, so the delete button appears below the
    item label rather than across from it.
    """

    id: str
    text: str
    short_label: str
    extra_buttons: tuple[dict[str, object], ...] = ()
    label_button: dict[str, object] | None = None


_CONDITION_PICKER_TITLE = "🎛 設定狀態"
_WPRC_TAG_RE = re.compile(r"\[wprc:([a-f0-9]{40})\]")
_FBPRC_TAG_RE = re.compile(r"\[fbprc:([a-zA-Z0-9\-]{1,80})\]")


def _render_watch_edit_view(
    processor: "TelegramCommandProcessor", watch_id: str
) -> tuple[str, dict[str, object]] | None:
    """Build the single-watch detailed edit view (text + inline keyboard)."""
    watch_db = getattr(processor, "_watch_db", None)
    if watch_db is None:
        return None
    watch = watch_db.get_marketplace_watch(watch_id)
    if watch is None:
        return None

    market_chips = " ".join(
        f"{_marketplace_source_display(m)[1]}{_marketplace_source_display(m)[0]}"
        for m in watch.markets
    ) or "(無)"

    lines = [
        f"✏️  {watch.query}",
        "",
        f"💰 上限：¥{watch.price_threshold_jpy:,}",
        f"🛍 平台：{market_chips}",
    ]
    if "mercari" in watch.markets:
        condition_ids = _condition_ids_from_options(watch.options_for("mercari"))
        lines.append(f"🎛 Mercari 狀態：{_summarize_condition_ids_short(condition_ids)}")

    keyboard: list[list[dict[str, object]]] = []

    keyboard.append([{"text": "💰 修改上限", "callback_data": f"wprc:{watch_id}"}])

    mkt_row = []
    for m in DEFAULT_MARKETS:
        name, emoji = _marketplace_source_display(m)
        check = "✓" if m in watch.markets else "☐"
        mkt_row.append({
            "text": f"{check} {emoji}{name}",
            "callback_data": f"wmkt:{watch_id}:{m}",
        })
    keyboard.append(mkt_row)

    if "mercari" in watch.markets:
        keyboard.append([{"text": "🎛 Mercari 狀態", "callback_data": f"cond:{watch_id}:open"}])

    keyboard.append([{"text": "← 返回清單", "callback_data": "wback:"}])

    return "\n".join(lines), {"inline_keyboard": keyboard}


def _summarize_condition_ids_short(condition_ids: tuple[int, ...]) -> str:
    """One-line summary for /watchlist's per-watch row."""
    if not condition_ids:
        return "未設定"
    if condition_ids == (1, 2, 3):
        return "目立った傷や汚れなし以上（預設）"
    return " / ".join(
        MERCARI_CONDITION_LABELS.get(cid, f"ID{cid}") for cid in condition_ids
    )


def _build_condition_picker_view(
    *, watch_id: str, query: str, condition_ids: tuple[int, ...]
) -> tuple[str, dict[str, object]]:
    """Render the per-watch condition checkbox picker.

    Each of the six Mercari conditions appears as one row with a checkbox
    indicator. Tapping a row toggles that condition in the watch's
    condition_ids set. The "完成並返回清單" row returns the user to /watchlist.
    """
    active = set(condition_ids)
    lines = [
        f"{_CONDITION_PICKER_TITLE}：{query}",
        f"目前接受：{_summarize_condition_ids_short(condition_ids)}",
        "",
        "勾選你想保留的狀態（至少要留一個）：",
    ]
    keyboard: list[list[dict[str, object]]] = []
    for cid in sorted(MERCARI_CONDITION_LABELS.keys()):
        label = MERCARI_CONDITION_LABELS[cid]
        marker = "☑" if cid in active else "☐"
        keyboard.append([{
            "text": f"{marker} {label}",
            "callback_data": f"cond:{watch_id}:t:{cid}",
        }])
    keyboard.append([{
        "text": "💾 完成並返回清單",
        "callback_data": f"cond:{watch_id}:done",
    }])
    return "\n".join(lines), {"inline_keyboard": keyboard}


def _build_list_view(
    *,
    list_kind: str,
    items: list[_ListRow],
    page: int,
    mode: str,
    list_title: str,
    empty_message: str,
) -> tuple[str, dict[str, object] | None, int]:
    """Render a paginated list view. Returns (text, reply_markup, clamped_page).

    `clamped_page` is `page` snapped into `[0, total_pages)`; callers that
    re-render after deletion can pass `page=current_page` and read back the
    page actually shown (it may shift up by one if they just removed the
    last item on the last page).
    """
    if not items:
        return empty_message, None, 0

    total = len(items)
    total_pages = max(1, (total + LIST_VIEW_PAGE_SIZE - 1) // LIST_VIEW_PAGE_SIZE)
    clamped = max(0, min(page, total_pages - 1))
    start = clamped * LIST_VIEW_PAGE_SIZE
    visible = items[start : start + LIST_VIEW_PAGE_SIZE]

    header = f"{list_title}  第 {clamped + 1}/{total_pages} 頁（共 {total} 筆）"
    body_lines = [row.text for row in visible if row.text]
    text = "\n".join([header, "", *body_lines] if body_lines else [header])

    keyboard: list[list[dict[str, object]]] = []
    if mode == LIST_VIEW_MODE_EDIT:
        for row in visible:
            if row.label_button is not None:
                keyboard.append([row.label_button])
            btn_label = f"❌ 刪除 {row.short_label}".strip() if row.short_label else "❌ 刪除"
            row_buttons: list[dict[str, object]] = [{
                "text": btn_label,
                "callback_data": f"del:{list_kind}:{row.id}",
            }]
            row_buttons.extend(row.extra_buttons)
            keyboard.append(row_buttons)

    nav: list[dict[str, object]] = []
    if clamped > 0:
        nav.append({"text": "⬅️ 上頁", "callback_data": f"pg:{list_kind}:{clamped - 1}:{mode}"})
    if mode == LIST_VIEW_MODE_READ:
        nav.append({"text": "✏️ 編輯", "callback_data": f"pg:{list_kind}:{clamped}:{LIST_VIEW_MODE_EDIT}"})
    else:
        nav.append({"text": "✓ 完成", "callback_data": f"pg:{list_kind}:{clamped}:{LIST_VIEW_MODE_READ}"})
    if clamped < total_pages - 1:
        nav.append({"text": "下頁 ➡️", "callback_data": f"pg:{list_kind}:{clamped + 1}:{mode}"})
    nav.append({"text": "✖️ 關閉", "callback_data": f"close:{list_kind}"})
    keyboard.append(nav)

    return text, {"inline_keyboard": keyboard}, clamped


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
    item_kind_hint: str | None = None
    file_id: str | None = None


@dataclass(frozen=True, slots=True)
class TelegramReputationQuery:
    query_url: str


@dataclass(frozen=True, slots=True)
class TelegramResearchQuery:
    query: str


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
    reply_factory: "Callable[[], str | tuple[str, dict[str, object] | None]] | None" = None
    reputation_delivery_factory: "Callable[[], TelegramReputationDelivery] | None" = None
    reply_markup: dict[str, object] | None = None  # optional inline keyboard for list views

    def execute(self) -> str | None:
        text, _ = self._execute_unpacked()
        return text

    def _execute_unpacked(self) -> tuple[str | None, dict[str, object] | None]:
        """Return (text, reply_markup_from_factory). Factory may return either
        a bare string (legacy) or a tuple (text, markup-override). Callers
        merge the override with self.reply_markup at send time."""
        if self.reply is not None:
            return self.reply, None
        if self.reply_factory is not None:
            result = self.reply_factory()
            if isinstance(result, tuple):
                return result[0], result[1]
            return result, None
        return None, None


@dataclass(frozen=True, slots=True)
class TelegramPhotoIntentOption:
    option_number: int
    action_key: str
    prompt: str
    synthetic_caption: str


@dataclass(frozen=True, slots=True)
class TelegramPhotoIntentAnalysis:
    options: tuple[TelegramPhotoIntentOption, ...]
    parsed_game: str | None = None
    parsed_item_kind: str | None = None
    parsed_title: str | None = None
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class PhotoLookupReply:
    """Return type for `photo_renderer` callables that need to install a
    pending photo clarification (e.g. when the vision pipeline could not
    confidently identify the card and is falling back to user clarification).
    Plain `str` returns are still supported for back-compat — `_handle_photo_message`
    wraps them automatically.
    `reply_markup` carries the optional inline keyboard for the bot to attach
    (e.g. the price-feedback ❌ button on successful lookups)."""
    text: str
    pending_clarification: "PendingTelegramPhotoClarification | None" = None
    ack: str | None = None
    reply_markup: dict[str, object] | None = None


@dataclass(slots=True)
class PendingTelegramPhotoClarification:
    chat_id: str
    image_path: Path
    caption: str | None
    file_id: str | None
    options: tuple[TelegramPhotoIntentOption, ...]
    parsed_game: str | None = None
    parsed_item_kind: str | None = None
    parsed_title: str | None = None
    created_at: float = field(default_factory=time.monotonic)

    def is_expired(self) -> bool:
        return (time.monotonic() - self.created_at) > PHOTO_CLARIFICATION_TTL_SECONDS


@dataclass(frozen=True, slots=True)
class TelegramTextIntentOption:
    option_number: int
    action_key: str
    prompt: str
    intent: "TelegramNaturalLanguageIntent"


@dataclass(slots=True)
class PendingTelegramTextClarification:
    chat_id: str
    original_text: str
    options: tuple[TelegramTextIntentOption, ...]
    top_intent: "TelegramNaturalLanguageIntent | None" = None
    created_at: float = field(default_factory=time.monotonic)

    def is_expired(self) -> bool:
        return (time.monotonic() - self.created_at) > TEXT_CLARIFICATION_TTL_SECONDS


PRICE_FEEDBACK_TTL_SECONDS = 10 * 60


@dataclass(slots=True)
class PendingTelegramPriceFeedback:
    """User clicked '不合理' on a lookup result; we sent a ForceReply asking
    for a reference URL. This row is consumed when the user replies with the
    URL. Mirrors PendingTelegramTextClarification pattern."""
    chat_id: str
    item_id: str
    original_fair_value_jpy: int | None
    created_at: float = field(default_factory=time.monotonic)

    def is_expired(self) -> bool:
        return (time.monotonic() - self.created_at) > PRICE_FEEDBACK_TTL_SECONDS


@dataclass(slots=True)
class PendingTelegramSnsBulkUpdate:
    """A SNS bulk-rule update that's been previewed to the user and is
    waiting on a confirm / cancel inline-button tap.

    The ``action`` field selects which apply helper runs on confirm:
    - ``"add"``           → apply_bulk_keyword_filter_add (uses keywords)
    - ``"remove"``        → apply_bulk_keyword_filter_remove (uses keywords)
    - ``"set_schedule"``  → apply_bulk_schedule_update (uses schedule_minutes)
    """
    chat_id: str
    bulk_target_domain: str
    keywords: tuple[str, ...]
    affected_rule_ids: tuple[str, ...]
    action: str = "add"
    schedule_minutes: int | None = None
    created_at: float = field(default_factory=time.monotonic)

    def is_expired(self) -> bool:
        return (time.monotonic() - self.created_at) > TEXT_CLARIFICATION_TTL_SECONDS


def default_photo_intent_analyzer(
    *,
    db_path: str | Path | None = None,
    tesseract_path: str | None = None,
    tessdata_dir: str | None = None,
    vision_settings: TcgVisionSettings | None = None,
) -> PhotoIntentAnalyzer:
    image_service = TcgImagePriceService(
        db_path=db_path,
        tesseract_path=tesseract_path,
        tessdata_dir=tessdata_dir,
        vision_settings=vision_settings,
    )

    def analyze(query: TelegramPhotoQuery) -> TelegramPhotoIntentAnalysis:
        parsed = image_service.parse_image(
            query.image_path,
            caption=query.caption,
            game_hint=query.game_hint,
            title_hint=query.title_hint,
            item_kind_hint=query.item_kind_hint,
        )
        return TelegramPhotoIntentAnalysis(
            options=_build_photo_intent_options(parsed_game=parsed.game, item_kind=parsed.item_kind),
            parsed_game=parsed.game,
            parsed_item_kind=parsed.item_kind,
            parsed_title=parsed.title,
            warnings=parsed.warnings,
        )

    return analyze


def _build_photo_intent_options(*, parsed_game: str | None, item_kind: str | None) -> tuple[TelegramPhotoIntentOption, ...]:
    preferred: list[str]
    if parsed_game == "pokemon" and item_kind == "sealed_box":
        preferred = ["pokemon_box_price", "pokemon_card_price", "yugioh_card_price"]
    elif parsed_game == "pokemon":
        preferred = ["pokemon_card_price", "pokemon_box_price", "yugioh_card_price"]
    elif parsed_game == "yugioh":
        preferred = ["yugioh_card_price", "pokemon_card_price", "pokemon_box_price"]
    elif parsed_game == "ws":
        preferred = ["ws_card_price", "pokemon_card_price", "yugioh_card_price"]
    elif parsed_game == "union_arena":
        preferred = ["union_arena_card_price", "pokemon_card_price", "yugioh_card_price"]
    else:
        preferred = ["pokemon_card_price", "yugioh_card_price", "pokemon_box_price"]

    labels = {
        "pokemon_card_price": ("要我查這張寶可夢卡市價嗎？", "/scan pokemon"),
        "yugioh_card_price": ("要我查這張遊戲王卡市價嗎？", "/scan yugioh"),
        "pokemon_box_price": ("要我查這個寶可夢卡盒市價嗎？", "/scan pokemon"),
        "ws_card_price": ("要我查這張 Weiss Schwarz 卡市價嗎？", "/scan ws"),
        "union_arena_card_price": ("要我查這張 Union Arena 卡市價嗎？", "/scan union_arena"),
    }
    options: list[TelegramPhotoIntentOption] = []
    seen: set[str] = set()
    for action_key in preferred:
        if action_key in seen or action_key not in labels:
            continue
        seen.add(action_key)
        prompt, caption = labels[action_key]
        options.append(
            TelegramPhotoIntentOption(
                option_number=len(options) + 1,
                action_key=action_key,
                prompt=prompt,
                synthetic_caption=caption,
            )
        )
    return tuple(options)


def _build_clarification_keyboard(
    prefix: str, options: tuple
) -> dict[str, object] | None:
    """Build the inline keyboard for a numbered-options clarification message.

    Each option becomes one row: ``[N. <prompt>]`` with callback_data
    ``<prefix>:<N>``. The "都不是 / 否，..." free-form fallback stays as a
    text-only instruction (Telegram inline buttons can't capture free text).
    Returns None if there are no options to render.
    """
    if not options:
        return None
    keyboard: list[list[dict[str, object]]] = []
    for opt in options:
        prompt = str(opt.prompt or "").strip()
        # Telegram renders button text on a single line; cap to keep it readable.
        if len(prompt) > 48:
            prompt = prompt[:47] + "…"
        keyboard.append([{
            "text": f"{opt.option_number}. {prompt}",
            "callback_data": f"{prefix}:{opt.option_number}",
        }])
    return {"inline_keyboard": keyboard}


def _build_photo_clarification_reply(
    analysis: TelegramPhotoIntentAnalysis,
) -> tuple[str, dict[str, object] | None]:
    context = _describe_photo_analysis_context(
        parsed_game=analysis.parsed_game,
        parsed_item_kind=analysis.parsed_item_kind,
        parsed_title=analysis.parsed_title,
    )
    lines = [context, "請點按鈕（或回覆數字）："]
    for option in analysis.options:
        lines.append(f"{option.option_number}. {option.prompt}")
    lines.append(f"{len(analysis.options) + 1}. 都不是，請回答：否，[您的意圖]")
    return "\n".join(lines), _build_clarification_keyboard("popt", analysis.options)


def _describe_photo_analysis_context(*, parsed_game: str | None, parsed_item_kind: str | None, parsed_title: str | None) -> str:
    if parsed_title and parsed_game:
        item_label = "卡盒" if parsed_item_kind == "sealed_box" else "卡片"
        return f"我先看了一下，這張圖看起來比較像「{_display_game_name(parsed_game)}{item_label}」{parsed_title}，但我還不想先亂猜你的意圖。"
    if parsed_game:
        item_label = "卡盒" if parsed_item_kind == "sealed_box" else "卡片"
        return f"我先看了一下，這張圖看起來比較像「{_display_game_name(parsed_game)}{item_label}」，但我還不想先亂猜你的意圖。"
    return "我先看了一下這張圖，但我還不想先亂猜你的意圖。"


def _display_game_name(game: str) -> str:
    labels = {
        "pokemon": "寶可夢",
        "yugioh": "遊戲王",
        "ws": "Weiss Schwarz",
        "union_arena": "Union Arena",
    }
    return labels.get(game, game)


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
            "allowed_updates": ["message", "callback_query"],
        }
        if offset is not None:
            payload["offset"] = offset
        result = self._call("getUpdates", payload)
        return result if isinstance(result, list) else []

    def send_message(
        self,
        *,
        chat_id: str | int,
        text: str,
        reply_markup: dict[str, object] | None = None,
    ) -> dict[str, object]:
        payload: dict[str, object] = {
            "chat_id": str(chat_id),
            "text": text[:4096],
            "disable_web_page_preview": True,
        }
        if reply_markup is not None:
            payload["reply_markup"] = reply_markup
        return self._call("sendMessage", payload)

    def edit_message_text(
        self,
        *,
        chat_id: str | int,
        message_id: int,
        text: str,
        reply_markup: dict[str, object] | None = None,
    ) -> dict[str, object]:
        payload: dict[str, object] = {
            "chat_id": str(chat_id),
            "message_id": int(message_id),
            "text": text[:4096],
            "disable_web_page_preview": True,
        }
        if reply_markup is not None:
            payload["reply_markup"] = reply_markup
        return self._call("editMessageText", payload)

    def answer_callback_query(
        self,
        *,
        callback_query_id: str,
        text: str | None = None,
        show_alert: bool = False,
    ) -> dict[str, object]:
        payload: dict[str, object] = {
            "callback_query_id": callback_query_id,
            "show_alert": show_alert,
        }
        if text is not None:
            payload["text"] = text[:200]
        return self._call("answerCallbackQuery", payload)

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
        photo_intent_analyzer: PhotoIntentAnalyzer | None = None,
        reputation_renderer: ReputationRenderer | None = None,
        research_renderer: ResearchRenderer | None = None,
        natural_language_router: TelegramNaturalLanguageRouter | None = None,
        allowed_chat_ids: frozenset[str] | None = None,
        status_renderer: Callable[[], str] | None = None,
        watch_db: MonitorDatabase | None = None,
        sns_db: SnsDatabase | None = None,
        sns_buzz_fn: Callable[[str], str] | None = None,
        opportunity_status_renderer: Callable[[], str] | None = None,
        opportunity_target_remover: OpportunityTargetRemover | None = None,
        opportunity_list_provider: Callable[[], list[dict[str, object]]] | None = None,
        opportunity_alias_updater: OpportunityAliasUpdater | None = None,
        opportunity_target_pinner: OpportunityTargetPinner | None = None,
        opportunity_target_unpinner: OpportunityTargetUnpinner | None = None,
        knowledge_handler: Callable[[str, str], str] | None = None,
        collab_backfiller: "object | None" = None,
        feedback_service: "object | None" = None,
    ) -> None:
        self._lookup_renderer = lookup_renderer
        self._board_loader = board_loader
        self._catalog_renderer = catalog_renderer
        self._photo_intent_analyzer = photo_intent_analyzer or default_photo_intent_analyzer()
        self._reputation_renderer = reputation_renderer
        self._research_renderer = research_renderer
        self._natural_language_router = natural_language_router
        self._allowed_chat_ids: frozenset[str] = allowed_chat_ids or frozenset()
        self._status_renderer = status_renderer
        self._watch_db = watch_db
        self._sns_db = sns_db
        self._sns_buzz_fn = sns_buzz_fn
        self._opportunity_status_renderer = opportunity_status_renderer
        self._opportunity_target_remover = opportunity_target_remover
        self._opportunity_list_provider = opportunity_list_provider
        self._opportunity_alias_updater = opportunity_alias_updater
        self._opportunity_target_pinner = opportunity_target_pinner
        self._opportunity_target_unpinner = opportunity_target_unpinner
        self._knowledge_handler = knowledge_handler
        self._collab_backfiller = collab_backfiller
        self._feedback_service = feedback_service
        self._pending_photo_clarifications: dict[str, PendingTelegramPhotoClarification] = {}
        self._pending_text_clarifications: dict[str, PendingTelegramTextClarification] = {}
        self._pending_sns_bulk_updates: dict[str, PendingTelegramSnsBulkUpdate] = {}
        self._pending_price_feedbacks: dict[str, PendingTelegramPriceFeedback] = {}

    def is_allowed_chat(self, chat_id: str | int) -> bool:
        if not self._allowed_chat_ids:
            return True
        return str(chat_id) in self._allowed_chat_ids

    def get_pending_photo_clarification(self, chat_id: str | int) -> PendingTelegramPhotoClarification | None:
        key = str(chat_id)
        pending = self._pending_photo_clarifications.get(key)
        if pending is None:
            return None
        if not pending.is_expired():
            return pending
        self.clear_pending_photo_clarification(chat_id)
        return None

    def set_pending_photo_clarification(self, clarification: PendingTelegramPhotoClarification) -> None:
        existing = self._pending_photo_clarifications.get(clarification.chat_id)
        if existing is not None:
            self._cleanup_pending_photo_path(existing.image_path)
        self._pending_photo_clarifications[clarification.chat_id] = clarification

    def pop_pending_photo_clarification(self, chat_id: str | int) -> PendingTelegramPhotoClarification | None:
        return self._pending_photo_clarifications.pop(str(chat_id), None)

    def clear_pending_photo_clarification(self, chat_id: str | int) -> None:
        pending = self._pending_photo_clarifications.pop(str(chat_id), None)
        if pending is not None:
            self._cleanup_pending_photo_path(pending.image_path)

    def get_pending_text_clarification(self, chat_id: str | int) -> PendingTelegramTextClarification | None:
        key = str(chat_id)
        pending = self._pending_text_clarifications.get(key)
        if pending is None:
            return None
        if not pending.is_expired():
            return pending
        self._pending_text_clarifications.pop(key, None)
        return None

    def set_pending_text_clarification(self, clarification: PendingTelegramTextClarification) -> None:
        self._pending_text_clarifications[clarification.chat_id] = clarification

    def pop_pending_text_clarification(self, chat_id: str | int) -> PendingTelegramTextClarification | None:
        return self._pending_text_clarifications.pop(str(chat_id), None)

    def clear_pending_text_clarification(self, chat_id: str | int) -> None:
        self._pending_text_clarifications.pop(str(chat_id), None)

    def get_pending_price_feedback(self, chat_id: str | int) -> PendingTelegramPriceFeedback | None:
        key = str(chat_id)
        pending = self._pending_price_feedbacks.get(key)
        if pending is None:
            return None
        if not pending.is_expired():
            return pending
        self._pending_price_feedbacks.pop(key, None)
        return None

    def set_pending_price_feedback(self, pending: PendingTelegramPriceFeedback) -> None:
        self._pending_price_feedbacks[pending.chat_id] = pending

    def pop_pending_price_feedback(self, chat_id: str | int) -> PendingTelegramPriceFeedback | None:
        return self._pending_price_feedbacks.pop(str(chat_id), None)

    def clear_pending_price_feedback(self, chat_id: str | int) -> None:
        self._pending_price_feedbacks.pop(str(chat_id), None)

    def get_pending_sns_bulk_update(self, chat_id: str | int) -> PendingTelegramSnsBulkUpdate | None:
        key = str(chat_id)
        pending = self._pending_sns_bulk_updates.get(key)
        if pending is None:
            return None
        if not pending.is_expired():
            return pending
        self._pending_sns_bulk_updates.pop(key, None)
        return None

    def set_pending_sns_bulk_update(self, pending: PendingTelegramSnsBulkUpdate) -> None:
        self._pending_sns_bulk_updates[pending.chat_id] = pending

    def pop_pending_sns_bulk_update(self, chat_id: str | int) -> PendingTelegramSnsBulkUpdate | None:
        return self._pending_sns_bulk_updates.pop(str(chat_id), None)

    def analyze_photo_intent(self, query: TelegramPhotoQuery) -> TelegramPhotoIntentAnalysis:
        return self._photo_intent_analyzer(query)

    @staticmethod
    def _cleanup_pending_photo_path(path: Path) -> None:
        try:
            path.unlink(missing_ok=True)
        except PermissionError:
            logger.debug("Could not remove pending Telegram photo path=%s", path)

    def build_reply(self, *, chat_id: str | int, text: str | None) -> str | None:
        return self.build_reply_plan(chat_id=chat_id, text=text).execute()

    def build_pending_photo_reply_plan(
        self,
        *,
        chat_id: str | int,
        text: str | None,
        photo_renderer: PhotoLookupRenderer,
    ) -> TelegramTextReplyPlan | None:
        pending = self.get_pending_photo_clarification(chat_id)
        if pending is None or text is None:
            return None

        content = text.strip()
        if not content or content.startswith("/"):
            return None

        if content == str(len(pending.options) + 1):
            return TelegramTextReplyPlan(
                ack=None,
                reply="好，請直接回答：否，[您的意圖]",
            )

        selected_option = _match_photo_clarification_option(content, pending.options)
        if selected_option is not None:
            resolved = self.pop_pending_photo_clarification(chat_id)
            assert resolved is not None
            return TelegramTextReplyPlan(
                ack=f"收到，我就照第 {selected_option.option_number} 個方式處理。",
                reply=None,
                reply_factory=lambda resolved=resolved, option=selected_option: _execute_pending_photo_lookup(
                    pending=resolved,
                    option=option,
                    photo_renderer=photo_renderer,
                ),
            )

        override_text = _extract_photo_clarification_override(content)
        if override_text is not None:
            override_option = _resolve_photo_override_to_option(override_text, pending.options)
            if override_option is not None:
                resolved = self.pop_pending_photo_clarification(chat_id)
                assert resolved is not None
                return TelegramTextReplyPlan(
                    ack=f"收到，我改照你補充的意思處理：{override_text}",
                    reply=None,
                    reply_factory=lambda resolved=resolved, option=override_option: _execute_pending_photo_lookup(
                        pending=resolved,
                        option=option,
                        photo_renderer=photo_renderer,
                    ),
                )
            identify_reply = _build_photo_identify_reply(
                pending=pending,
                override_text=override_text,
            )
            if identify_reply is not None:
                self.clear_pending_photo_clarification(chat_id)
                return TelegramTextReplyPlan(ack=None, reply=identify_reply)
            return TelegramTextReplyPlan(
                ack=None,
                reply=(
                    "我還沒完全理解你的意思。請直接回答像這樣：\n"
                    "否，查這張寶可夢卡市價\n"
                    "否，查這張遊戲王卡市價\n"
                    "否，查這個寶可夢卡盒市價"
                ),
            )

        retry_text, retry_kb = _build_pending_photo_retry_reply(pending.options)
        return TelegramTextReplyPlan(
            ack=None,
            reply=retry_text,
            reply_markup=retry_kb,
        )

    def build_pending_text_reply_plan(
        self,
        *,
        chat_id: str | int,
        text: str | None,
    ) -> TelegramTextReplyPlan | None:
        pending = self.get_pending_text_clarification(chat_id)
        if pending is None or text is None:
            return None

        content = text.strip()
        if not content or content.startswith("/"):
            return None

        # "都不是" sentinel option (last numbered choice = len(options) + 1)
        if content == str(len(pending.options) + 1):
            return TelegramTextReplyPlan(
                ack=None,
                reply="好，請直接回答：否，[您的意圖]",
            )

        selected = _match_text_clarification_option(content, pending.options)
        if selected is not None:
            self.pop_pending_text_clarification(chat_id)
            plan = self._build_natural_language_reply_plan(selected.intent, chat_id=chat_id)
            ack_prefix = f"收到，我就照第 {selected.option_number} 個方式處理。"
            if plan is not None:
                combined_ack = ack_prefix if not plan.ack else f"{ack_prefix}\n{plan.ack}"
                return TelegramTextReplyPlan(
                    ack=combined_ack,
                    reply=plan.reply,
                    reply_factory=plan.reply_factory,
                    reputation_delivery_factory=plan.reputation_delivery_factory,
                )
            return TelegramTextReplyPlan(
                ack=None,
                reply=f"{ack_prefix}\n但暫時無法執行該動作，請告訴我更詳細的需求。",
            )

        override_text = _extract_photo_clarification_override(content)
        if override_text is not None:
            self.pop_pending_text_clarification(chat_id)
            if not override_text:
                return TelegramTextReplyPlan(
                    ack=None,
                    reply="好，請直接告訴我你的意圖。",
                )
            rerouted_plan = self.build_reply_plan(chat_id=chat_id, text=override_text)
            new_pending = self.get_pending_text_clarification(chat_id)
            if new_pending is not None:
                # Re-routing produced a fresh clarification — let it speak for itself.
                return rerouted_plan
            ack = f"收到，我改照你補充的意思處理：{override_text}"
            combined_ack = ack if not rerouted_plan.ack else f"{ack}\n{rerouted_plan.ack}"
            return TelegramTextReplyPlan(
                ack=combined_ack,
                reply=rerouted_plan.reply,
                reply_factory=rerouted_plan.reply_factory,
                reputation_delivery_factory=rerouted_plan.reputation_delivery_factory,
            )

        retry_text, retry_kb = _build_pending_text_retry_reply(pending.options)
        return TelegramTextReplyPlan(
            ack=None,
            reply=retry_text,
            reply_markup=retry_kb,
        )

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
        if command in HUNT_COMMANDS:
            action = remainder.strip().lower()
            # `/hunt`、`/hunt list`、空動作 → 進入分頁清單（若 provider 有設定）。
            # `/hunt status` 維持舊「狀態 + 推薦紀錄」純文字輸出。
            if (
                self._opportunity_list_provider is not None
                and action in {"", "list", "candidates", "targets", "目標", "候選"}
            ):
                text, reply_markup, _ = self.render_huntlist_view()
                return TelegramTextReplyPlan(ack=None, reply=text, reply_markup=reply_markup)
            return TelegramTextReplyPlan(ack=None, reply=self._handle_hunt(remainder))
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
        if command in WEB_RESEARCH_COMMANDS:
            return TelegramTextReplyPlan(
                ack=build_processing_ack(text=content),
                reply=None,
                reply_factory=lambda remainder=remainder: self._handle_web_research(remainder),
            )
        if command in WATCH_COMMANDS:
            return TelegramTextReplyPlan(
                ack="收到追蹤指令，正在設定…",
                reply=None,
                reply_factory=lambda remainder=remainder, cid=chat_id: self._handle_watch(remainder, str(cid)),
            )
        if command in WATCHLIST_COMMANDS:
            text, reply_markup, _ = self.render_watchlist_view()
            return TelegramTextReplyPlan(ack=None, reply=text, reply_markup=reply_markup)
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
            text, reply_markup, _ = self.render_snslist_view()
            return TelegramTextReplyPlan(ack=None, reply=text, reply_markup=reply_markup)
        if command in SNS_DELETE_COMMANDS:
            return TelegramTextReplyPlan(ack=None, reply=self._handle_sns_delete(remainder))
        if command in SNS_BUZZ_COMMANDS:
            return TelegramTextReplyPlan(
                ack="收到，正在抓取 X 熱門討論並交給 LLM 整理…",
                reply=None,
                reply_factory=lambda remainder=remainder: self._handle_sns_buzz(remainder),
            )
        if command in KNOWLEDGE_COMMANDS:
            return TelegramTextReplyPlan(
                ack=None,
                reply=self._handle_knowledge(remainder, str(chat_id)),
            )
        if not content.startswith("/"):
            intent = self._route_natural_language(content)
            if _is_text_intent_ambiguous(intent):
                clarification_plan = self._build_text_clarification_plan(
                    chat_id=chat_id,
                    text=content,
                    top_intent=intent,
                )
                if clarification_plan is not None:
                    return clarification_plan
            natural_language_plan = self._build_natural_language_reply_plan(intent, chat_id=chat_id)
            if natural_language_plan is not None:
                return natural_language_plan
            clarification_plan = self._build_text_clarification_plan(
                chat_id=chat_id,
                text=content,
                top_intent=intent,
            )
            if clarification_plan is not None:
                return clarification_plan

        logger.info("Telegram unknown command command=%s", command)
        return TelegramTextReplyPlan(
            ack=None,
            reply="Unknown command. Use /help, /price, /trend, /snapshot, /search, or send a photo with /scan. You can also ask in natural language.",
        )

    def _build_text_clarification_plan(
        self,
        *,
        chat_id: str | int,
        text: str,
        top_intent: "TelegramNaturalLanguageIntent | None",
    ) -> TelegramTextReplyPlan | None:
        options = _build_text_intent_candidates(text, top_intent)
        if not options:
            return None
        self.set_pending_text_clarification(
            PendingTelegramTextClarification(
                chat_id=str(chat_id),
                original_text=text,
                options=options,
                top_intent=top_intent,
            )
        )
        clar_text, clar_kb = _build_text_clarification_reply(text, options, top_intent)
        return TelegramTextReplyPlan(
            ack=None,
            reply=clar_text,
            reply_markup=clar_kb,
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

        game = normalize_game_key(parts[0])
        if game is None:
            return f"Unsupported game. Use {supported_game_hint()}."

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

    def _handle_web_research(self, raw: str) -> str:
        if self._research_renderer is None:
            return "這個 OpenClaw 執行環境尚未設定網路搜尋摘要功能。"
        query = raw.strip()
        if not query:
            return "請提供要搜尋整理的問題。例如：/search 為什麼皮卡丘寶可夢卡這麼受歡迎"
        try:
            logger.info("Telegram web research requested query=%s", trim_for_log(query, limit=240))
            return self._research_renderer(TelegramResearchQuery(query=query))
        except Exception as exc:  # pragma: no cover - source/network-dependent.
            logger.exception("Telegram web research failed query=%s", trim_for_log(query, limit=240))
            return f"網路搜尋摘要失敗：{exc}"

    def _handle_watch(
        self, raw: str, chat_id: str, *, markets: tuple[str, ...] | None = None,
    ) -> str:
        if self._watch_db is None:
            return "追蹤功能尚未啟用（watch_db 未設定）。"
        try:
            query, threshold, parsed_markets = parse_watch_command(raw)
        except ValueError as exc:
            return (
                f"{exc}\n"
                f"格式範例：/watch 想いが重なる場所で 初音ミク SSP on 300000\n"
                f"指定平台：/watch アビスアイ box on 8000 markets:rakuma\n"
                f"指定多平台：/watch ピカチュウ on 5000 markets:mercari,rakuma"
            )
        # Resolution order: NL-provided markets → command suffix → DEFAULT_MARKETS.
        chosen = markets or parsed_markets or DEFAULT_MARKETS
        chosen = _normalize_markets(chosen) or DEFAULT_MARKETS
        return self._add_marketplace_watch_row(
            chat_id=chat_id, query=query, threshold=threshold, markets=chosen,
        )

    def _add_marketplace_watch_row(
        self,
        *,
        chat_id: str,
        query: str,
        threshold: int,
        markets: tuple[str, ...],
    ) -> str:
        watch_id = build_marketplace_watch_id(chat_id=chat_id, query=query)
        market_options: dict[str, dict[str, object]] = {}
        for market in markets:
            if market == "mercari":
                market_options[market] = {"condition_ids": [1, 2, 3]}
            else:
                market_options[market] = {}
        watch = MarketplaceWatch(
            watch_id=watch_id,
            query=query,
            price_threshold_jpy=threshold,
            markets=markets,
            enabled=True,
            chat_id=chat_id,
            last_checked_at=None,
            created_at="",
            updated_at="",
            market_options=market_options,
        )
        if self._watch_db is not None:
            self._watch_db.add_marketplace_watch(watch)
        logger.info(
            "Watch added watch_id=%s query=%s threshold=%d markets=%s chat_id=%s",
            watch_id, query, threshold, list(markets), chat_id,
        )
        market_chips = ", ".join(
            f"{_marketplace_source_display(m)[1]} {_marketplace_source_display(m)[0]}"
            for m in markets
        )
        return (
            f"✅ 已新增追蹤\n"
            f"ID: {watch_id}\n"
            f"關鍵字：{query}\n"
            f"價格上限：¥{threshold:,}\n"
            f"監控平台：{market_chips}\n"
            f"將每分鐘在以上平台搜尋，發現新商品時通知你。"
        )

    def _handle_watchlist(self) -> str:
        text, _, _ = self.render_watchlist_view(page=0, mode=LIST_VIEW_MODE_READ)
        return text

    def render_watchlist_view(
        self, *, page: int = 0, mode: str = LIST_VIEW_MODE_READ
    ) -> tuple[str, dict[str, object] | None, int]:
        """Render the paginated multi-source marketplace watch view.
        Returns (text, reply_markup, page)."""
        if self._watch_db is None:
            return "追蹤功能尚未啟用（watch_db 未設定）。", None, 0
        watches = list(self._watch_db.list_marketplace_watchlist())
        watches.sort(key=lambda w: (not w.enabled, w.watch_id))

        _CIRCLED = "①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳"

        items: list[_ListRow] = []
        for i, w in enumerate(watches):
            num = _CIRCLED[i] if i < len(_CIRCLED) else str(i + 1)
            status = "✓ 啟用" if w.enabled else "✗ 停用"
            checked = _format_local_time(w.last_checked_at) if w.last_checked_at else "尚未檢查"
            market_chips_full = " ".join(
                f"{_marketplace_source_display(m)[1]}{_marketplace_source_display(m)[0]}"
                for m in w.markets
            ) or "(無)"
            market_emojis = "".join(
                _marketplace_source_display(m)[1] for m in w.markets
            ) or "—"
            extra_lines: list[str] = []
            extra_buttons = (
                {"text": "✏️ 詳細編輯", "callback_data": f"wedit:{w.watch_id}"},
            )
            if "mercari" in w.markets:
                condition_ids = _condition_ids_from_options(w.options_for("mercari"))
                extra_lines.append(f"  Mercari 狀態：{_summarize_condition_ids_short(condition_ids)}")
            if mode == LIST_VIEW_MODE_EDIT:
                # Edit mode: text body is empty (label_button carries item info)
                text_block = ""
                label_btn: dict[str, object] | None = {
                    "text": f"📌 {w.query}  ¥{w.price_threshold_jpy:,}  {market_emojis}",
                    "callback_data": "noop",
                }
            else:
                label_btn = None
                text_block = (
                    f"{num} {status}  {w.query}\n"
                    f"  上限：¥{w.price_threshold_jpy:,}  ·  {market_chips_full}\n"
                    + ("\n".join(extra_lines) + "\n" if extra_lines else "")
                    + f"  最後檢查：{checked}"
                )
            items.append(_ListRow(
                id=w.watch_id,
                text=text_block,
                short_label="",
                extra_buttons=extra_buttons,
                label_button=label_btn,
            ))

        return _build_list_view(
            list_kind="wl",
            items=items,
            page=page,
            mode=mode,
            list_title="📋 Marketplace 追蹤（多站）",
            empty_message=(
                "目前沒有任何追蹤項目。\n"
                "使用 /watch 關鍵字 on 價格 新增追蹤（預設監控 Mercari + Rakuma）。\n"
                "指定平台：/watch 關鍵字 on 價格 markets:rakuma"
            ),
        )

    def delete_mercari_watch_by_id(self, watch_id: str) -> bool:
        """Backward-compat alias — the row deleter map still calls this name.
        Internally delegates to ``delete_marketplace_watch`` so it works for
        any source (Mercari / Rakuma / future)."""
        return self.delete_marketplace_watch_by_id(watch_id)

    def delete_marketplace_watch_by_id(self, watch_id: str) -> bool:
        if self._watch_db is None:
            return False
        try:
            return bool(self._watch_db.delete_marketplace_watch(watch_id))
        except Exception:
            logger.exception("Marketplace watch delete failed watch_id=%s", watch_id)
            return False

    def _handle_unwatch(self, raw: str) -> str:
        if self._watch_db is None:
            return "追蹤功能尚未啟用（watch_db 未設定）。"
        watch_id = raw.strip()
        if not watch_id:
            return "請提供追蹤 ID。格式：/unwatch <ID>\n可用 /watchlist 查看所有 ID。"
        deleted = self._watch_db.delete_marketplace_watch(watch_id)
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
        watch = self._watch_db.get_marketplace_watch(watch_id)
        if watch is None:
            return f"找不到追蹤 ID [{watch_id}]，請用 /watchlist 確認。"
        old_price = watch.price_threshold_jpy
        self._watch_db.update_marketplace_watch(watch_id, price_threshold_jpy=new_price)
        logger.info("Watch price updated watch_id=%s old=%d new=%d", watch_id, old_price, new_price)
        return (
            f"已更新追蹤目標價\n"
            f"ID: {watch_id}\n"
            f"關鍵字：{watch.query}\n"
            f"價格上限：¥{old_price:,} → ¥{new_price:,}"
        )

    def _handle_hunt(self, raw: str) -> str:
        action = raw.strip().lower()
        if action in {"", "status", "candidates", "list", "targets", "目標", "候選"}:
            if self._opportunity_status_renderer is None:
                return "Opportunity agent status is not configured in this runtime."
            return self._opportunity_status_renderer()
        if _is_hunt_remove_action(raw):
            if self._opportunity_target_remover is None:
                return "Opportunity target removal is not configured in this runtime."
            target = _extract_hunt_remove_target(raw)
            if not target:
                return "請提供要移除的目標，例如：/hunt remove 2 或 /hunt remove Umbreon ex SAR"
            return self._opportunity_target_remover(target)
        if _is_hunt_pin_action(raw):
            if self._opportunity_target_pinner is None:
                return "Opportunity target pinner is not configured in this runtime."
            name = _extract_hunt_pin_target(raw)
            if not name:
                return "請提供要釘為目標的商品名，例如：/hunt pin アビスアイ box"
            return self._opportunity_target_pinner(name)
        if _is_hunt_unpin_action(raw):
            if self._opportunity_target_unpinner is None:
                return "Opportunity target unpinner is not configured in this runtime."
            selector = _extract_hunt_unpin_target(raw)
            if not selector:
                return "請提供要從目標清單移除的編號或名稱，例如：/hunt unpin 1"
            return self._opportunity_target_unpinner(selector)
        alias_reply = self._maybe_handle_hunt_alias(raw)
        if alias_reply is not None:
            return alias_reply
        return (
            "可用格式：\n"
            "  /hunt status\n"
            "  /hunt pin <商品名>          ← 🎯 主動加入目標清單\n"
            "  /hunt unpin <編號或名稱>     ← 從目標清單移除（保留 candidate）\n"
            "  /hunt remove <編號或名稱>   ← 永久封殺該 candidate\n"
            "  /hunt alias <編號或id> add 別名A, 別名B\n"
            "  /hunt alias <編號或id> remove 別名A\n"
            "  /hunt related <編號或id> add 關鍵字"
        )

    def _maybe_handle_hunt_alias(self, raw: str) -> str | None:
        """Parse `/hunt alias|related <selector> add|remove <names>`.

        Returns the reply text when matched, None otherwise so the caller falls
        back to the generic usage hint. Names accept ','/'、'/'，' separators.
        """
        import re as _re

        match = _re.match(
            r"^\s*(?P<kind>alias|aliases|related|related_keywords)\s+(?P<rest>.+)$",
            raw,
            _re.IGNORECASE,
        )
        if match is None:
            return None
        kind_raw = match.group("kind").lower()
        kind = "aliases" if kind_raw.startswith("alias") else "related"
        rest = match.group("rest").strip()
        action_match = _re.search(r"\b(add|remove|rm|del|delete)\b", rest, _re.IGNORECASE)
        if action_match is None:
            return f"請指定 add 或 remove。例如：/hunt {kind_raw} <編號或id> add 別名A"
        selector = rest[: action_match.start()].strip()
        if not selector:
            return f"請提供候選編號或 id。例如：/hunt {kind_raw} 2 add 別名A"
        action_word = action_match.group(1).lower()
        action = "remove" if action_word in {"remove", "rm", "del", "delete"} else "add"
        tail = rest[action_match.end():].strip()
        if not tail:
            return f"請提供至少一個名稱。例如：/hunt {kind_raw} {selector} {action} ピカチュウex SAR"
        names = _split_alias_names(tail)
        if not names:
            return "請提供至少一個有效的名稱。"
        if self._opportunity_alias_updater is None:
            return "Opportunity alias updater is not configured in this runtime."
        return self._opportunity_alias_updater(selector, kind, action, names)

    def render_huntlist_view(
        self, *, page: int = 0, mode: str = LIST_VIEW_MODE_READ
    ) -> tuple[str, dict[str, object] | None, int]:
        """Render the paginated Opportunity candidate view. Returns (text, reply_markup, page)."""
        if self._opportunity_list_provider is None:
            return "Opportunity 候選清單未啟用（list provider 未設定）。", None, 0
        try:
            candidates = list(self._opportunity_list_provider())
        except Exception as exc:
            logger.exception("Hunt list provider failed")
            return f"列表失敗：{exc}", None, 0

        items: list[_ListRow] = []
        for candidate in candidates:
            candidate_id = str(candidate.get("candidate_id") or "")
            if not candidate_id:
                continue
            game = str(candidate.get("game") or "?")
            product_type = str(candidate.get("product_type") or "other")
            title = str(candidate.get("title") or "(no title)")
            heat = candidate.get("heat_score")
            heat_text = f"{float(heat):.0f}" if heat is not None else "?"
            search_query = str(candidate.get("search_query") or "")
            text_block = (
                f"[{game} / {product_type}] {title}\n"
                f"  heat={heat_text}  search: {search_query}"
            )
            short = f"[{game}] {title[:18]}"
            items.append(_ListRow(id=candidate_id, text=text_block, short_label=short))

        return _build_list_view(
            list_kind="hl",
            items=items,
            page=page,
            mode=mode,
            list_title="📋 Opportunity 候選",
            empty_message="目前沒有 Opportunity 候選。等下一輪 agent tick 收集到目標後再看。",
        )

    def delete_huntlist_item_by_id(self, candidate_id: str) -> bool:
        """Dismiss an opportunity candidate by id via the existing remover.

        Returns True only on confirmed removal (the remover's reply starts
        with the success prefix from ``dismiss_opportunity_target``).
        """
        if self._opportunity_target_remover is None:
            return False
        try:
            reply = self._opportunity_target_remover(candidate_id)
        except Exception:
            logger.exception("Hunt delete failed candidate_id=%s", candidate_id)
            return False
        return isinstance(reply, str) and reply.startswith("已從機會清單移除")

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
            if normalize_game_key(intent.game) is None:
                return TelegramTextReplyPlan(
                    ack=None,
                    reply=f"I understood that you want the hot board, but I still need the game: {supported_game_hint()}.",
                )
            game = normalize_game_key(intent.game) or intent.game
            limit = 5 if intent.limit is None else max(1, min(10, intent.limit))
            logger.info(
                "Telegram natural-language routed intent=trend_board game=%s limit=%s confidence=%s",
                game,
                limit,
                intent.confidence,
            )
            return TelegramTextReplyPlan(
                ack=f"已理解查詢內容，相當於 /trend {game} {limit}，開始整理資料。",
                reply=None,
                reply_factory=lambda game=game, limit=limit: self._handle_liquidity(f"{game} {limit}"),
            )
        if intent.intent == "lookup_card":
            if normalize_game_key(intent.game) is None:
                return TelegramTextReplyPlan(
                    ack=None,
                    reply=f"I understood that you want a card lookup, but I still need the game: {supported_game_hint()}.",
                )
            game = normalize_game_key(intent.game) or intent.game
            resolved_name = intent.name or intent.card_number
            if not resolved_name:
                return TelegramTextReplyPlan(
                    ack=None,
                    reply="I understood that you want a card lookup, but I still need the card name.",
                )
            query = TelegramLookupQuery(
                game=game,
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
            markets = _normalize_markets(intent.watch_markets) or DEFAULT_MARKETS
            market_chips = ", ".join(_marketplace_source_display(m)[0] for m in markets)
            logger.info(
                "Telegram natural-language routed intent=add_watch markets=%s watch_query=%s watch_price_threshold=%d confidence=%s",
                list(markets), q, p, intent.confidence,
            )
            return TelegramTextReplyPlan(
                ack=f"已理解，相當於 /watch {q} on {p}（平台：{market_chips}），開始設定。",
                reply=None,
                reply_factory=lambda q=q, p=p, mk=markets, cid=chat_id: self._handle_watch(
                    f"{q} on {p}", str(cid), markets=mk,
                ),
            )
        if intent.intent == "list_watches":
            logger.info("Telegram natural-language routed intent=list_watches confidence=%s", intent.confidence)
            text, reply_markup, _ = self.render_watchlist_view()
            return TelegramTextReplyPlan(ack=None, reply=text, reply_markup=reply_markup)
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
        if intent.intent == "web_research":
            query = intent.research_query
            if not query:
                return TelegramTextReplyPlan(
                    ack=None,
                    reply="I understood that you want web research, but I still need the topic or question.",
                )
            logger.info(
                "Telegram natural-language routed intent=web_research query=%s confidence=%s",
                trim_for_log(query, limit=240),
                intent.confidence,
            )
            return TelegramTextReplyPlan(
                ack=f"已理解：相當於 /search {query}，正在搜尋資料來源並整理答案…",
                reply=None,
                reply_factory=lambda q=query: self._handle_web_research(q),
            )
        if intent.intent == "opportunity_remove":
            target = intent.opportunity_target
            if not target:
                return TelegramTextReplyPlan(
                    ack=None,
                    reply="請告訴我要移除哪個機會目標，例如：移除機會目標 2 或 不感興趣 Umbreon ex SAR",
                )
            logger.info(
                "Telegram natural-language routed intent=opportunity_remove target=%s confidence=%s",
                trim_for_log(target, limit=160),
                intent.confidence,
            )
            return TelegramTextReplyPlan(
                ack=f"已理解：相當於 /hunt remove {target}，正在移除。",
                reply=None,
                reply_factory=lambda target=target: self._handle_hunt(f"remove {target}"),
            )

        # ── SNS intents ────────────────────────────────────────────────────────
        if intent.intent == "sns_add_account":
            if not intent.sns_handle:
                return TelegramTextReplyPlan(
                    ack=None,
                    reply="請告訴我要追蹤哪個 X 帳號，例如：追蹤 @elonmusk",
                )
            handle = intent.sns_handle
            include_keywords = tuple(intent.sns_include_keywords)
            schedule_minutes = intent.sns_schedule_minutes
            logger.info(
                "Telegram NL routed intent=sns_add_account handle=%s include_keywords=%s schedule_minutes=%s",
                handle,
                include_keywords,
                schedule_minutes,
            )
            cid = str(chat_id)
            filter_suffix = f" {json.dumps(list(include_keywords), ensure_ascii=False)}" if include_keywords else ""
            schedule_suffix = f" schedule:{schedule_minutes}" if schedule_minutes else ""
            remainder = f"@{handle}{filter_suffix}{schedule_suffix}"
            ack_extras: list[str] = []
            if include_keywords:
                ack_extras.append(f"並加上篩選 {', '.join(include_keywords)}")
            if schedule_minutes:
                ack_extras.append(f"排程 {schedule_minutes} 分鐘")
            ack_suffix = ("，" + "、".join(ack_extras)) if ack_extras else ""
            return TelegramTextReplyPlan(
                ack=f"已理解：相當於 /snsadd {remainder}，正在新增 X 追蹤{ack_suffix}…",
                reply=None,
                reply_factory=lambda r=remainder, c=cid: self._handle_sns_add(r, c),
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
            text, reply_markup, _ = self.render_snslist_view()
            return TelegramTextReplyPlan(ack=None, reply=text, reply_markup=reply_markup)
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

        if intent.intent == "sns_bulk_add_filter":
            return self._build_sns_bulk_add_filter_plan(
                chat_id=chat_id,
                target_domain=intent.bulk_target_domain or "",
                keywords=intent.bulk_filter_keywords,
            )

        if intent.intent == "sns_bulk_remove_filter":
            return self._build_sns_bulk_remove_filter_plan(
                chat_id=chat_id,
                target_domain=intent.bulk_target_domain or "",
                keywords=intent.bulk_filter_keywords,
            )

        if intent.intent == "sns_bulk_update_schedule":
            return self._build_sns_bulk_update_schedule_plan(
                chat_id=chat_id,
                target_domain=intent.bulk_target_domain or "",
                minutes=intent.sns_schedule_minutes,
            )

        if intent.intent == "sns_clear_filter":
            if not intent.sns_handle:
                return TelegramTextReplyPlan(
                    ack=None,
                    reply="請告訴我要清空哪個 @ 帳號的 filter，例如：把 @elonmusk 的 filter 拿掉。",
                )
            handle = intent.sns_handle
            logger.info("Telegram NL routed intent=sns_clear_filter handle=%s", handle)
            return TelegramTextReplyPlan(
                ack=f"已理解：把 @{handle} 的 filter 清空（保留追蹤）",
                reply=None,
                reply_factory=lambda h=handle: self._handle_sns_clear_filter(h),
            )

        return None

    def _handle_sns_add(self, raw: str, chat_id: str) -> str:
        """Handle /snsadd command to add an X/Reddit account, keyword, or trend watch."""
        if self._sns_db is None:
            return "SNS 監控尚未啟用（sns_db 未設定）。"
        from sns_monitor.filters import (
            extract_labeled_brackets,
            extract_schedule_minutes,
            parse_account_watch_text,
            split_source_prefix,
        )
        from sns_monitor.models import AccountWatch, KeywordWatch, TrendWatch
        from sns_monitor.storage import SnsDatabase

        # Per-source defaults — kept in sync with sources/*.py default_*_schedule_minutes.
        # Inlined here to avoid import cycles via sources -> reddit_buzz -> subprocess.
        default_schedules: dict[tuple[str, str], int] = {
            ("x", "account"): 15, ("x", "keyword"): 30, ("x", "trend"): 60,
            ("reddit", "account"): 30, ("reddit", "keyword"): 60,
        }
        source_label = {"x": "X", "reddit": "Reddit"}

        try:
            raw = raw.strip()
            if not raw:
                return (
                    '用法：/snsadd x:@username\n'
                    '     /snsadd x:@username filter[抽選] domain[pokemon] schedule:30\n'
                    '     /snsadd x:keyword:搜尋詞 domain[gundam]\n'
                    '     /snsadd x:trend:trending\n'
                    '     /snsadd reddit:r/PokemonTCG domain[pokemon] schedule:30\n'
                    '     /snsadd reddit:keyword:Umbreon ex domain[pokemon]'
                )

            # 1. Pull `schedule:NN` out of the raw text first so it doesn't
            #    leak into source / kind parsing downstream.
            schedule_override, raw = extract_schedule_minutes(raw)

            # 2. Strip `<source>:` prefix. No prefix → backcompat default "x".
            source, body = split_source_prefix(raw)

            # 3. Reddit trend watches don't exist — fail loudly.
            if source == "reddit" and body.lower().startswith("trend:"):
                return "Reddit 來源不支援 trend 追蹤。請改用 reddit:r/<subreddit> 或 reddit:keyword:<關鍵字>。"

            # Reddit account form is `r/<subreddit>` (parse_account_watch_text
            # already handles both `@xxx` and `r/xxx`); X account form is `@<handle>`.
            account_target = parse_account_watch_text(body)
            if account_target is not None:
                screen_name, include_keywords, domains = account_target
                rule_id = SnsDatabase._watch_rule_id("account", screen_name, source)
                existing_rule = self._sns_db.get_watch_rule(rule_id)
                resolved_domains = (
                    domains if domains is not None else getattr(existing_rule, "domains", ())
                )
                schedule_minutes = (
                    schedule_override
                    if schedule_override is not None
                    else getattr(existing_rule, "schedule_minutes", None)
                    or default_schedules.get((source, "account"), 30)
                )
                display = f"r/{screen_name}" if source == "reddit" else f"@{screen_name}"
                rule = AccountWatch(
                    rule_id=rule_id,
                    screen_name=screen_name,
                    user_id=getattr(existing_rule, "user_id", None),
                    label=getattr(existing_rule, "label", None) or display,
                    include_keywords=include_keywords,
                    domains=resolved_domains,
                    enabled=True,
                    schedule_minutes=schedule_minutes,
                    chat_id=chat_id,
                    last_checked_at=getattr(existing_rule, "last_checked_at", None),
                    source=source,
                )
                self._sns_db.save_watch_rule(rule)
                logger.info(
                    "SNS account watch added source=%s target=%s chat_id=%s include_keywords=%s domains=%s schedule=%dm",
                    source, screen_name, chat_id, include_keywords, resolved_domains, schedule_minutes,
                )
                filter_line = f"\n篩選：{', '.join(include_keywords)}" if include_keywords else ""
                domain_line = f"\n領域：{', '.join(resolved_domains)}" if resolved_domains else ""
                kind_label = "subreddit" if source == "reddit" else "帳號"
                return (
                    f"✓ 已新增 {source_label.get(source, source)} {kind_label}追蹤：{display}"
                    f"{filter_line}{domain_line}\n排程：每 {schedule_minutes} 分鐘\nID: {rule_id[:8]}…"
                )
            elif body.lower().startswith("keyword:"):
                explicit_filter, parsed_domains, body_clean = extract_labeled_brackets(body[len("keyword:"):])
                query = body_clean.strip()
                if not query:
                    return "請提供搜尋關鍵字。例如：/snsadd x:keyword:機動戰士 domain[gundam]"
                rule_id = SnsDatabase._watch_rule_id("keyword", query, source)
                existing_rule = self._sns_db.get_watch_rule(rule_id)
                resolved_domains = (
                    parsed_domains if parsed_domains is not None else getattr(existing_rule, "domains", ())
                )
                schedule_minutes = (
                    schedule_override
                    if schedule_override is not None
                    else getattr(existing_rule, "schedule_minutes", None)
                    or default_schedules.get((source, "keyword"), 60)
                )
                rule = KeywordWatch(
                    rule_id=rule_id,
                    query=query,
                    label=f'"{query}"',
                    domains=resolved_domains,
                    enabled=True,
                    schedule_minutes=schedule_minutes,
                    chat_id=chat_id,
                    last_checked_at=None,
                    source=source,
                )
                self._sns_db.save_watch_rule(rule)
                logger.info(
                    "SNS keyword watch added source=%s query=%s chat_id=%s domains=%s schedule=%dm",
                    source, query, chat_id, resolved_domains, schedule_minutes,
                )
                domain_line = f"\n領域：{', '.join(resolved_domains)}" if resolved_domains else ""
                return (
                    f'✓ 已新增 {source_label.get(source, source)} 關鍵字追蹤："{query}"{domain_line}'
                    f"\n排程：每 {schedule_minutes} 分鐘\nID: {rule_id[:8]}…"
                )
            elif body.lower().startswith("trend:"):
                _, parsed_domains, body_clean = extract_labeled_brackets(body[len("trend:"):])
                category = body_clean.strip()
                if category not in {"trending", "for-you", "news", "sports", "entertainment"}:
                    return "不支援的分類。請使用：trending, for-you, news, sports, 或 entertainment"
                rule_id = SnsDatabase._watch_rule_id("trend", category, source)
                existing_rule = self._sns_db.get_watch_rule(rule_id)
                resolved_domains = (
                    parsed_domains if parsed_domains is not None else getattr(existing_rule, "domains", ())
                )
                schedule_minutes = (
                    schedule_override
                    if schedule_override is not None
                    else getattr(existing_rule, "schedule_minutes", None)
                    or default_schedules.get((source, "trend"), 60)
                )
                rule = TrendWatch(
                    rule_id=rule_id,
                    category=category,
                    label=f"Trend: {category}",
                    domains=resolved_domains,
                    enabled=True,
                    schedule_minutes=schedule_minutes,
                    chat_id=chat_id,
                    last_checked_at=None,
                    source=source,
                )
                self._sns_db.save_watch_rule(rule)
                logger.info(
                    "SNS trend watch added source=%s category=%s chat_id=%s schedule=%dm",
                    source, category, chat_id, schedule_minutes,
                )
                return (
                    f"✓ 已新增 {source_label.get(source, source)} 熱門話題追蹤：{category}"
                    f"\n排程：每 {schedule_minutes} 分鐘\nID: {rule_id[:8]}…"
                )
            else:
                return (
                    '不認識的格式。用法：\n'
                    '  /snsadd x:@username [filter[…] domain[…] schedule:NN]\n'
                    '  /snsadd x:keyword:搜尋詞 / x:trend:trending\n'
                    '  /snsadd reddit:r/<subreddit> / reddit:keyword:<關鍵字>'
                )
        except Exception as exc:
            logger.exception("SNS add failed raw=%s chat_id=%s", raw, chat_id)
            return f"新增失敗：{exc}"

    def _handle_sns_list(self) -> str:
        """Backward-compatible thin wrapper. Returns plain text only.

        The interactive paginated + edit-mode keyboard lives on
        ``render_snslist_view`` and is reached via the
        TelegramCommandProcessor dispatch path that consults it directly.
        """
        text, _, _ = self.render_snslist_view(page=0, mode=LIST_VIEW_MODE_READ)
        return text

    def render_snslist_view(
        self, *, page: int = 0, mode: str = LIST_VIEW_MODE_READ
    ) -> tuple[str, dict[str, object] | None, int]:
        """Render the paginated SNS-list view. Returns (text, reply_markup, page)."""
        if self._sns_db is None:
            return "SNS 監控尚未啟用（sns_db 未設定）。", None, 0
        try:
            rules = list(self._sns_db.list_watch_rules())
        except Exception as exc:
            logger.exception("SNS list failed")
            return f"列表失敗：{exc}", None, 0

        # Stable order: enabled-first (so deactivated rules sink to the bottom),
        # then by rule_id so the page slice doesn't shift mid-session.
        rules.sort(key=lambda r: (not r.enabled, r.rule_id))

        items: list[_ListRow] = []
        for rule in rules:
            status = "✓" if rule.enabled else "✗"
            source = getattr(rule, "source", "x")
            source_tag = f"[{source}] "
            screen_name = getattr(rule, "screen_name", None)
            query_text = getattr(rule, "query", None)
            category = getattr(rule, "category", None)
            if screen_name:
                handle_display = f"r/{screen_name}" if source == "reddit" else f"@{screen_name}"
                include_kw = getattr(rule, "include_keywords", ()) or ()
                filters = f" filter[{', '.join(include_kw)}]" if include_kw else ""
                info = f"{handle_display}{filters}"
                short = handle_display
            elif query_text:
                info = f'"{query_text}"'
                short = f'"{query_text[:18]}"'
            elif category:
                info = f"Trend:{category}"
                short = f"Trend:{category}"
            else:
                info = "Unknown"
                short = rule.rule_id[:8]
            domains = getattr(rule, "domains", ())
            domain_segment = f" domain[{', '.join(domains)}]" if domains else " domain[?]"
            schedule_segment = f" schedule:{rule.schedule_minutes}m" if getattr(rule, "schedule_minutes", None) else ""
            text_block = f"  {status} {source_tag}{info}{domain_segment}{schedule_segment} ({rule.rule_id[:8]}…)"
            items.append(_ListRow(id=rule.rule_id, text=text_block, short_label=short))

        text, reply_markup, clamped = _build_list_view(
            list_kind="sl",
            items=items,
            page=page,
            mode=mode,
            list_title="📋 SNS 監控規則",
            empty_message="尚無 SNS 監控規則。\n用法：/snsadd @username",
        )
        return text, reply_markup, clamped

    def delete_sns_rule_by_id(self, rule_id: str) -> bool:
        """Delete a SNS watch rule by its full rule_id. Returns True if a row was removed."""
        if self._sns_db is None:
            return False
        try:
            return bool(self._sns_db.delete_watch_rule(rule_id))
        except Exception:
            logger.exception("SNS delete by id failed rule_id=%s", rule_id)
            return False

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

        Accepts: a hex rule_id prefix, '@handle' / 'r/sub' for account watches,
        'keyword:xxx' for keyword watches, and the source-prefixed forms
        ('reddit:r/sub', 'reddit:keyword:xxx', 'x:@handle', ...).
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

        # 2) Optional `<source>:` prefix narrows the search. Inline a small
        #    prefix-split here to avoid a hard sns_monitor dependency in the
        #    test path (test venv may not have sns_monitor installed).
        source_filter = "x"
        had_source_prefix = False
        body = cleaned
        for src in ("reddit:", "x:"):
            if cleaned.lower().startswith(src):
                source_filter = src[:-1]
                had_source_prefix = True
                body = cleaned[len(src):].strip()
                break

        def _source_matches(rule) -> bool:
            return (not had_source_prefix) or getattr(rule, "source", "x") == source_filter

        # 3) @handle → X account watch
        if body.startswith("@"):
            handle = body.lstrip("@").lower()
            for rule in rules:
                if (
                    getattr(rule, "screen_name", "").lower() == handle
                    and _source_matches(rule)
                ):
                    return rule.rule_id
            return None

        # 4) r/<sub> → Reddit subreddit account watch
        if body.lower().startswith("r/"):
            sub = body[2:].strip().lower()
            for rule in rules:
                if (
                    getattr(rule, "screen_name", "").lower() == sub
                    and (getattr(rule, "source", "x") == "reddit" if not had_source_prefix else _source_matches(rule))
                ):
                    return rule.rule_id
            return None

        # 5) keyword:xxx → keyword watch
        if body.lower().startswith("keyword:"):
            query = body.split(":", 1)[1].strip().lower()
            for rule in rules:
                if (
                    getattr(rule, "query", "").lower() == query
                    and _source_matches(rule)
                ):
                    return rule.rule_id
            return None

        # 6) Bare token → try as handle (without @ or r/)
        bare = body.lstrip("@").lower()
        for rule in rules:
            if (
                getattr(rule, "screen_name", "").lower() == bare
                and _source_matches(rule)
            ):
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
                include_keywords = getattr(rule, "include_keywords", ())
                filters = f" [{', '.join(include_keywords)}]" if include_keywords else ""
                return f"@{screen_name}{filters}"
            query = getattr(rule, "query", None)
            if query:
                return f"關鍵字「{query}」"
            return rule.rule_id[:8]
        return rule_id[:8]

    def _handle_sns_clear_filter(self, handle: str) -> str:
        """Clear include_keywords on an account watch while keeping the rule active."""
        if self._sns_db is None:
            return "SNS 監控尚未啟用（sns_db 未設定）。"
        from dataclasses import replace
        from sns_monitor.models import AccountWatch
        from sns_monitor.storage import SnsDatabase

        try:
            screen_name = handle.lstrip("@").strip()
            if not screen_name:
                return "請提供 @ 帳號，例如：把 @elonmusk 的 filter 拿掉。"
            rule_id = SnsDatabase._watch_rule_id("account", screen_name)
            existing_rule = self._sns_db.get_watch_rule(rule_id)
            if not isinstance(existing_rule, AccountWatch):
                return f"找不到 @{screen_name} 的 X 帳號追蹤規則（請先用 /snsadd 新增）。"
            if not existing_rule.include_keywords:
                return f"✓ @{screen_name} 目前沒有 filter，無需清空。"
            previous = existing_rule.include_keywords
            cleared = replace(existing_rule, include_keywords=())
            self._sns_db.save_watch_rule(cleared)
            logger.info(
                "SNS filter cleared screen_name=%s previous=%s", screen_name, previous,
            )
            return f"✓ 已清空 @{screen_name} 的 filter（追蹤仍啟用，原本：{', '.join(previous)}）。"
        except Exception as exc:
            logger.exception("SNS clear filter failed handle=%s", handle)
            return f"清空 filter 失敗：{exc}"

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

    def _handle_knowledge(self, raw: str, chat_id: str) -> str:
        """Dispatch /knowledge subcommands to the registered handler.

        The actual handler lives in aka_no_claw (it owns ``KnowledgeDatabase``).
        This method only does presence checking and forwards the remainder —
        the handler parses the action / entity / summary itself.
        """
        if self._knowledge_handler is None:
            return (
                "知識庫指令尚未啟用（需在 aka_no_claw 端註冊 knowledge_handler）。"
            )
        try:
            return self._knowledge_handler(raw, chat_id)
        except Exception as exc:
            logger.exception("knowledge handler failed raw=%r", raw)
            return f"知識庫指令失敗：{exc}"

    def _build_sns_bulk_add_filter_plan(
        self,
        *,
        chat_id: str | int,
        target_domain: str,
        keywords: tuple[str, ...],
    ) -> "TelegramTextReplyPlan":
        """Build the preview message for ``sns_bulk_add_filter`` and stash a
        PendingTelegramSnsBulkUpdate so the confirm/cancel inline buttons
        know what to do."""
        if self._sns_db is None:
            return TelegramTextReplyPlan(ack=None, reply="SNS 監控尚未啟用（sns_db 未設定）。")
        if not target_domain:
            return TelegramTextReplyPlan(ack=None, reply="我看不太懂你要對哪類帳號加 filter，請重講一次。")
        if not keywords:
            return TelegramTextReplyPlan(ack=None, reply="請告訴我要加哪個 filter 關鍵字。")

        try:
            from sns_monitor.bulk_filter import (
                find_accounts_matching_domain,
                resolve_target_domain_set,
            )
        except ImportError:
            logger.exception("sns_monitor.bulk_filter import failed")
            return TelegramTextReplyPlan(ack=None, reply="SNS bulk-filter 模組載入失敗。")

        target_set = resolve_target_domain_set(target_domain)
        accounts = find_accounts_matching_domain(self._sns_db, target_set)
        if not accounts:
            return TelegramTextReplyPlan(
                ack=None,
                reply=f"找不到任何 domain 跟 {target_domain} 有交集的 SNS 帳號。",
            )

        kw_display = "、".join(keywords)
        lines = [
            f"🎯 找到 {len(accounts)} 個 {target_domain} 相關帳號，要把 filter 加上：{kw_display}",
            "",
        ]
        for rule in accounts[:10]:
            existing = (
                f" 現有 filter[{', '.join(rule.include_keywords)}]"
                if rule.include_keywords else " 現有 filter[]"
            )
            domain_label = f" domain[{', '.join(rule.domains)}]" if rule.domains else ""
            lines.append(f"- @{rule.screen_name}{domain_label}{existing}")
        if len(accounts) > 10:
            lines.append(f"  …以及另外 {len(accounts) - 10} 筆")
        lines.append("")
        lines.append(f"確認要把「{kw_display}」加進這 {len(accounts)} 個帳號的 filter 嗎？")

        pending = PendingTelegramSnsBulkUpdate(
            chat_id=str(chat_id),
            bulk_target_domain=target_domain,
            keywords=keywords,
            affected_rule_ids=tuple(r.rule_id for r in accounts),
            action="add",
        )
        self.set_pending_sns_bulk_update(pending)

        reply_markup = {
            "inline_keyboard": [
                [{"text": f"✓ 全部修改 ({len(accounts)})", "callback_data": "bulk:c"}],
                [{"text": "✖️ 取消", "callback_data": "bulk:x"}],
            ]
        }
        return TelegramTextReplyPlan(
            ack=None,
            reply="\n".join(lines),
            reply_markup=reply_markup,
        )

    def _build_sns_bulk_remove_filter_plan(
        self,
        *,
        chat_id: str | int,
        target_domain: str,
        keywords: tuple[str, ...],
    ) -> "TelegramTextReplyPlan":
        """Preview for ``sns_bulk_remove_filter`` — symmetric to the add plan
        but the message text describes a removal, and the pending state
        carries ``action="remove"``."""
        if self._sns_db is None:
            return TelegramTextReplyPlan(ack=None, reply="SNS 監控尚未啟用（sns_db 未設定）。")
        if not target_domain:
            return TelegramTextReplyPlan(ack=None, reply="我看不太懂你要對哪類帳號移除 filter 關鍵字，請重講一次。")
        if not keywords:
            return TelegramTextReplyPlan(ack=None, reply="請告訴我要移除哪個 filter 關鍵字。")

        try:
            from sns_monitor.bulk_filter import (
                find_accounts_matching_domain,
                resolve_target_domain_set,
            )
        except ImportError:
            logger.exception("sns_monitor.bulk_filter import failed")
            return TelegramTextReplyPlan(ack=None, reply="SNS bulk-filter 模組載入失敗。")

        target_set = resolve_target_domain_set(target_domain)
        accounts = find_accounts_matching_domain(self._sns_db, target_set)
        if not accounts:
            return TelegramTextReplyPlan(
                ack=None,
                reply=f"找不到任何 domain 跟 {target_domain} 有交集的 SNS 帳號。",
            )

        drop_set = {kw.casefold() for kw in keywords if kw}
        affected_count = sum(
            1 for r in accounts if any(kw.casefold() in drop_set for kw in r.include_keywords)
        )

        kw_display = "、".join(keywords)
        lines = [
            f"🗑 找到 {len(accounts)} 個 {target_domain} 相關帳號，要從 filter 移除：{kw_display}",
            f"   其中 {affected_count} 個目前有這些 keyword（會被改），{len(accounts) - affected_count} 個沒有（不會動）",
            "",
        ]
        for rule in accounts[:10]:
            existing = (
                f" 現有 filter[{', '.join(rule.include_keywords)}]"
                if rule.include_keywords else " 現有 filter[]"
            )
            domain_label = f" domain[{', '.join(rule.domains)}]" if rule.domains else ""
            lines.append(f"- @{rule.screen_name}{domain_label}{existing}")
        if len(accounts) > 10:
            lines.append(f"  …以及另外 {len(accounts) - 10} 筆")
        lines.append("")
        lines.append(f"確認要從這 {len(accounts)} 個帳號的 filter 移除「{kw_display}」嗎？")

        pending = PendingTelegramSnsBulkUpdate(
            chat_id=str(chat_id),
            bulk_target_domain=target_domain,
            keywords=keywords,
            affected_rule_ids=tuple(r.rule_id for r in accounts),
            action="remove",
        )
        self.set_pending_sns_bulk_update(pending)

        reply_markup = {
            "inline_keyboard": [
                [{"text": f"✓ 全部移除 ({affected_count})", "callback_data": "bulk:c"}],
                [{"text": "✖️ 取消", "callback_data": "bulk:x"}],
            ]
        }
        return TelegramTextReplyPlan(
            ack=None,
            reply="\n".join(lines),
            reply_markup=reply_markup,
        )

    def _build_sns_bulk_update_schedule_plan(
        self,
        *,
        chat_id: str | int,
        target_domain: str,
        minutes: int | None,
    ) -> "TelegramTextReplyPlan":
        """Preview for ``sns_bulk_update_schedule`` — same pattern but the
        pending state carries ``action="set_schedule"`` and ``schedule_minutes``."""
        if self._sns_db is None:
            return TelegramTextReplyPlan(ack=None, reply="SNS 監控尚未啟用（sns_db 未設定）。")
        if not target_domain:
            return TelegramTextReplyPlan(ack=None, reply="我看不太懂你要對哪類帳號改 schedule，請重講一次。")
        if minutes is None or not (5 <= minutes <= 1440):
            return TelegramTextReplyPlan(ack=None, reply="請給一個有效的分鐘數 (5-1440)。")

        try:
            from sns_monitor.bulk_filter import (
                find_accounts_matching_domain,
                resolve_target_domain_set,
            )
        except ImportError:
            logger.exception("sns_monitor.bulk_filter import failed")
            return TelegramTextReplyPlan(ack=None, reply="SNS bulk-filter 模組載入失敗。")

        target_set = resolve_target_domain_set(target_domain)
        accounts = find_accounts_matching_domain(self._sns_db, target_set)
        if not accounts:
            return TelegramTextReplyPlan(
                ack=None,
                reply=f"找不到任何 domain 跟 {target_domain} 有交集的 SNS 帳號。",
            )
        already_count = sum(1 for r in accounts if r.schedule_minutes == minutes)
        will_change = len(accounts) - already_count

        lines = [
            f"⏱ 找到 {len(accounts)} 個 {target_domain} 相關帳號，要把 schedule 改成 {minutes} 分鐘",
            f"   其中 {already_count} 個本來就是 {minutes} 分鐘（不會動），{will_change} 個會被改",
            "",
        ]
        for rule in accounts[:10]:
            domain_label = f" domain[{', '.join(rule.domains)}]" if rule.domains else ""
            lines.append(f"- @{rule.screen_name}{domain_label} 目前 schedule={rule.schedule_minutes}m")
        if len(accounts) > 10:
            lines.append(f"  …以及另外 {len(accounts) - 10} 筆")
        lines.append("")
        lines.append(f"確認要把這 {len(accounts)} 個帳號的 schedule 改成 {minutes} 分鐘嗎？")

        pending = PendingTelegramSnsBulkUpdate(
            chat_id=str(chat_id),
            bulk_target_domain=target_domain,
            keywords=(),
            affected_rule_ids=tuple(r.rule_id for r in accounts),
            action="set_schedule",
            schedule_minutes=minutes,
        )
        self.set_pending_sns_bulk_update(pending)

        reply_markup = {
            "inline_keyboard": [
                [{"text": f"✓ 全部修改 ({will_change})", "callback_data": "bulk:c"}],
                [{"text": "✖️ 取消", "callback_data": "bulk:x"}],
            ]
        }
        return TelegramTextReplyPlan(
            ack=None,
            reply="\n".join(lines),
            reply_markup=reply_markup,
        )

    def _route_natural_language(self, text: str) -> TelegramNaturalLanguageIntent | None:
        llm_intent: TelegramNaturalLanguageIntent | None = None
        if self._natural_language_router is not None:
            try:
                llm_intent = self._natural_language_router.route(text)
            except Exception:
                logger.exception("Telegram natural-language router failed, falling back to keyword rules text=%s", trim_for_log(text, limit=240))

        fallback_intent = fallback_route_telegram_natural_language(text)

        # Bulk-filter sentences (e.g. "把每個跟 pokemon 相關的 sns 追蹤帳號 filter 都加上「抽選」")
        # are unreliable through the LLM — the structured fields it must populate are easy to drop.
        # When the deterministic regex fallback identifies this intent, prefer it over the LLM result.
        if (
            fallback_intent is not None
            and fallback_intent.intent == "sns_bulk_add_filter"
            and (llm_intent is None or llm_intent.intent != "sns_bulk_add_filter")
        ):
            logger.info(
                "Telegram natural-language fallback rescued sns_bulk_add_filter (llm_intent=%s)",
                getattr(llm_intent, "intent", None),
            )
            return fallback_intent

        # Clear-filter sentences (e.g. "把 @ARS_Arsales 的 filter 全部拿掉") get
        # routinely mis-mapped by the LLM to sns_delete because "拿掉" reads as
        # a remove-verb. The fallback's double-signal regex (filter-noun + clear-verb)
        # is the reliable detector — prefer it over the LLM when it fires.
        if (
            fallback_intent is not None
            and fallback_intent.intent == "sns_clear_filter"
            and (llm_intent is None or llm_intent.intent != "sns_clear_filter")
        ):
            logger.info(
                "Telegram natural-language fallback rescued sns_clear_filter (llm_intent=%s)",
                getattr(llm_intent, "intent", None),
            )
            return fallback_intent

        if llm_intent is not None and llm_intent.intent != "unknown":
            return llm_intent

        if fallback_intent is not None and fallback_intent.intent != "unknown":
            logger.info(
                "Telegram natural-language fallback intent=%s confidence=%s",
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
                "/price ygo | 青眼の白龍 | QCCP-JP001 | UR",
                "/price ua | 綾波レイ | UAPR/EVA-1-71",
                "/trend pokemon",
                "/trend ws 5",
                "/trend ygo 5",
                "/trend ua 5",
                "/hot pokemon",
                "/liquidity ws 5",
                "/snapshot https://jp.mercari.com/item/m123456789",
                "/search why Pikachu Pokemon cards are popular",
                "Send a photo with caption: /scan pokemon",
                "--- Mercari 追蹤 ---",
                "/watch 想いが重なる場所で 初音ミク SSP on 300000",
                "/watchlist",
                "/unwatch <ID>",
                "/setprice <ID> <新價格>",
                "--- SNS (X/Twitter) 監控 ---",
                "/snsadd @username",
                '/snsadd @username ["buy", "sell"]',
                "/snsadd keyword:搜詞",
                "/snsadd trend:trending",
                "/snslist",
                "/snsdelete <rule_id>",
                "/snsbuzz amd",
                "--- Opportunity Agent ---",
                "/hunt status",
                "/hunt remove 2",
                "You can also ask things like: 幫我查 pokemon Pikachu ex 132/106",
                "Or: pokemon 熱門前 5",
                "Or: why are Pikachu Pokemon cards so popular?",
                "Or: remove target 2 from the opportunity list",
            ]
        )


def _format_local_time(ts: str) -> str:
    """Convert a UTC ISO timestamp from DB to local timezone for display."""
    try:
        return datetime.fromisoformat(ts).astimezone().strftime("%Y-%m-%d %H:%M")
    except Exception:
        return ts[:16].replace("T", " ")


_MARKETS_SUFFIX_RE = re.compile(
    r"\bmarkets?\s*[:=]\s*([A-Za-z_,\s、，ーぁ-んァ-ヶー一-龥]+)\s*$",
    re.IGNORECASE,
)


def parse_watch_command(raw: str) -> tuple[str, int, tuple[str, ...] | None]:
    """Parse '/watch <query> on <price> [markets:<m1>,<m2>]' →
    (query, price_threshold_jpy, markets_or_None).

    ``markets`` is a tuple of canonical market names (or None if the user
    didn't specify any — caller applies the default). The suffix must come
    AFTER the price clause:

      /watch 想いが重なる場所で on 300000
      /watch アビスアイ box on 8000 markets:rakuma
      /watch ピカチュウ on 5000 markets:mercari,rakuma
    """
    body = raw.strip().strip("[]")
    markets: tuple[str, ...] | None = None
    markets_match = _MARKETS_SUFFIX_RE.search(body)
    if markets_match is not None:
        market_tokens = re.split(r"[,\s、，]+", markets_match.group(1).strip())
        normalized = _normalize_markets(market_tokens)
        if normalized:
            markets = normalized
        body = body[: markets_match.start()].rstrip()

    lower = body.lower()
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
    return query_part, threshold, markets


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
        game = normalize_game_key(parts[0])
        name = parts[1]
        if game is None:
            raise ValueError(f"Unsupported game. Use {supported_game_hint()}.")
        name, card_number, rarity, set_code = _recover_lookup_fields(
            name,
            _value_or_none(parts, 2),
            _value_or_none(parts, 3),
            _value_or_none(parts, 4),
        )
        if not name:
            raise ValueError("Lookup name cannot be empty.")
        return TelegramLookupQuery(
            game=game,
            name=name,
            card_number=card_number,
            rarity=rarity,
            set_code=set_code,
        )

    tokens = body.split()
    if len(tokens) < 2:
        raise ValueError("Lookup command requires at least game and name.")
    game = normalize_game_key(tokens[0])
    if game is None:
        raise ValueError(f"Unsupported game. Use {supported_game_hint()}.")
    name = " ".join(tokens[1:]).strip()
    name, card_number, rarity, set_code = _recover_lookup_fields(name, None, None, None)
    if not name:
        raise ValueError("Lookup name cannot be empty.")
    return TelegramLookupQuery(
        game=game,
        name=name,
        card_number=card_number,
        rarity=rarity,
        set_code=set_code,
    )


def parse_reputation_snapshot_command(raw: str) -> TelegramReputationQuery:
    query_url = raw.strip()
    if not query_url:
        raise ValueError("Snapshot command requires a Mercari item or profile URL.")
    return TelegramReputationQuery(query_url=query_url)


_HUNT_PIN_KEYWORDS: tuple[str, ...] = (
    "pin",
    "target",
    "watch",
    "track",
    "加入目標",
    "鎖定",
    "盯",
    "釘",
)

_HUNT_UNPIN_KEYWORDS: tuple[str, ...] = (
    "unpin",
    "untarget",
    "unwatch",
    "untrack",
    "取消鎖定",
    "解除目標",
    "解鎖",
)


def _action_keyword_match(raw: str, keywords: tuple[str, ...]) -> str | None:
    """Return the matching keyword if `raw` is exactly that keyword or starts
    with `keyword + space`. Used for "/hunt pin foo"-style action dispatch."""
    stripped = raw.strip()
    if not stripped:
        return None
    lowered = stripped.lower()
    for keyword in keywords:
        key_lower = keyword.lower()
        if lowered == key_lower or lowered.startswith(f"{key_lower} "):
            return keyword
        if stripped == keyword or stripped.startswith(f"{keyword} "):
            return keyword
    return None


def _strip_action_keyword(raw: str, keyword: str) -> str:
    stripped = raw.strip()
    lowered_stripped = stripped.lower()
    key_lower = keyword.lower()
    if lowered_stripped == key_lower or stripped == keyword:
        return ""
    if lowered_stripped.startswith(f"{key_lower} "):
        return stripped[len(keyword) :].strip()
    if stripped.startswith(f"{keyword} "):
        return stripped[len(keyword) :].strip()
    return stripped


def _is_hunt_pin_action(raw: str) -> bool:
    if _action_keyword_match(raw, _HUNT_UNPIN_KEYWORDS) is not None:
        return False  # unpin matches "unwatch" etc; check unpin first
    return _action_keyword_match(raw, _HUNT_PIN_KEYWORDS) is not None


def _extract_hunt_pin_target(raw: str) -> str:
    matched = _action_keyword_match(raw, _HUNT_PIN_KEYWORDS)
    if matched is None:
        return ""
    return _strip_action_keyword(raw, matched)


def _is_hunt_unpin_action(raw: str) -> bool:
    return _action_keyword_match(raw, _HUNT_UNPIN_KEYWORDS) is not None


def _extract_hunt_unpin_target(raw: str) -> str:
    matched = _action_keyword_match(raw, _HUNT_UNPIN_KEYWORDS)
    if matched is None:
        return ""
    return _strip_action_keyword(raw, matched)


def _is_hunt_remove_action(raw: str) -> bool:
    lowered = raw.strip().lower()
    if not lowered:
        return False
    return any(
        lowered == keyword or lowered.startswith(f"{keyword} ")
        for keyword in (
            "remove",
            "delete",
            "dismiss",
            "hide",
            "ignore",
            "drop",
            "not interested",
            "no interest",
            "移除",
            "刪除",
            "删除",
            "不要",
            "不感興趣",
            "不感兴趣",
            "沒興趣",
            "没兴趣",
            "外して",
            "削除",
        )
    )


def _extract_hunt_remove_target(raw: str) -> str:
    target = raw.strip()
    for phrase in (
        "not interested in",
        "not interested",
        "no interest in",
        "no interest",
        "remove",
        "delete",
        "dismiss",
        "hide",
        "ignore",
        "drop",
        "target",
        "candidate",
        "opportunity",
        "移除",
        "刪除",
        "删除",
        "不要",
        "不感興趣",
        "不感兴趣",
        "沒興趣",
        "没兴趣",
        "外して",
        "削除",
        "目標",
        "目标",
        "候選",
        "候选",
    ):
        target = re.sub(re.escape(phrase), " ", target, flags=re.IGNORECASE)
    target = re.sub(r"^(?:第)?\s*(\d{1,2})\s*(?:個|个|項|项|筆|笔|番)?$", r"\1", target.strip())
    target = re.sub(r"[，、。！？!?：:；;]", " ", target)
    return " ".join(target.split()).strip()


def _split_alias_names(tail: str) -> list[str]:
    """Split a /hunt alias names string into individual names.

    Accepts ASCII comma, full-width comma (，) and Japanese enumeration (、)
    as separators. When no separator is present, the entire tail is treated
    as a single name (so multi-word aliases like "テラスタル ピカチュウ sar"
    survive without manual quoting).
    """
    standardized = tail.replace("，", ",").replace("、", ",").strip()
    if not standardized:
        return []
    if "," in standardized:
        return [part.strip() for part in standardized.split(",") if part.strip()]
    return [standardized]


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
        return format_lookup_result_telegram(result), build_lookup_feedback_keyboard(result)

    return render


def default_photo_renderer(
    *,
    db_path: str | Path | None = None,
    tesseract_path: str | None = None,
    tessdata_dir: str | None = None,
    vision_settings: TcgVisionSettings | None = None,
    research_renderer: ResearchRenderer | None = None,
) -> PhotoLookupRenderer:
    image_service = TcgImagePriceService(
        db_path=db_path,
        tesseract_path=tesseract_path,
        tessdata_dir=tessdata_dir,
        vision_settings=vision_settings,
    )

    def render(query: TelegramPhotoQuery) -> str | PhotoLookupReply:
        logger.info(
            "Telegram photo renderer executing chat_id=%s file_id=%s path=%s caption=%s game_hint=%s title_hint=%s item_kind_hint=%s",
            mask_identifier(query.chat_id),
            query.file_id,
            query.image_path,
            trim_for_log(query.caption or "", limit=200),
            query.game_hint,
            query.title_hint,
            query.item_kind_hint,
        )
        outcome = image_service.lookup_image(
            query.image_path,
            caption=query.caption,
            game_hint=query.game_hint,
            title_hint=query.title_hint,
            item_kind_hint=query.item_kind_hint,
            persist=True,
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
        if outcome.status in {"unresolved", "rejected_sanity"}:
            return _build_unresolved_photo_reply(
                query=query,
                outcome=outcome,
                research_renderer=research_renderer,
            )
        text = format_photo_lookup_result(outcome)
        keyboard: dict[str, object] | None = None
        if outcome.lookup_result is not None:
            keyboard = build_lookup_feedback_keyboard(outcome.lookup_result)
        return PhotoLookupReply(text=text, reply_markup=keyboard)

    return render


def _build_unresolved_photo_reply(
    *,
    query: TelegramPhotoQuery,
    outcome: TcgImageLookupOutcome,
    research_renderer: ResearchRenderer | None,
) -> PhotoLookupReply:
    """Build a clarification reply for the photo path when vision either
    couldn't read the card (status=unresolved) or the post-lookup sanity
    check rejected the catalog match (status=rejected_sanity).

    Optionally asks the local /search pipeline for a human-readable hint
    about what the card might be, and stashes the photo as a pending
    clarification so the user's text response can re-target the lookup
    via the existing "否，[您的意圖]" override path.
    """
    parsed = outcome.parsed
    research_summary: str | None = None
    if research_renderer is not None and parsed.research_hint:
        try:
            research_summary = research_renderer(
                TelegramResearchQuery(query=parsed.research_hint)
            )
        except Exception:
            logger.exception(
                "Photo research-assist failed chat_id=%s file_id=%s",
                mask_identifier(query.chat_id),
                query.file_id,
            )
            research_summary = None

    if outcome.status == "rejected_sanity":
        head = "我看了一下，看起來和我查到的卡對不太上，所以先不亂下結論。"
    else:
        head = "我看不太清楚這張卡，先不亂猜。"

    pieces: list[str] = [head]
    if research_summary:
        pieces.append("以下是網路搜尋找到的線索：")
        pieces.append(research_summary.strip()[:1500])
    pieces.append(
        "可以直接告訴我這張卡是什麼嗎？例如：\n"
        "否，這張是寶可夢 ピカチュウ\n"
        "否，這張是遊戲王 青眼白龍"
    )
    text = "\n\n".join(pieces)

    pending = PendingTelegramPhotoClarification(
        chat_id=str(query.chat_id),
        image_path=query.image_path,
        caption=query.caption,
        file_id=query.file_id,
        options=(),  # no numeric options — only the "否，..." override path
        parsed_game=parsed.game,
        parsed_item_kind=parsed.item_kind,
        parsed_title=parsed.title,
    )
    return PhotoLookupReply(text=text, pending_clarification=pending)


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
    if command in WEB_RESEARCH_COMMANDS:
        return "收到搜尋問題，正在找資料來源並整理答案。"
    return None


def run_telegram_polling(
    *,
    token: str,
    lookup_renderer: LookupRenderer,
    board_loader: BoardLoader,
    catalog_renderer: CatalogRenderer,
    photo_renderer: PhotoLookupRenderer | None = None,
    photo_intent_analyzer: PhotoIntentAnalyzer | None = None,
    reputation_renderer: ReputationRenderer | None = None,
    research_renderer: ResearchRenderer | None = None,
    natural_language_router: TelegramNaturalLanguageRouter | None = None,
    ssl_context: ssl.SSLContext | None = None,
    allowed_chat_ids: frozenset[str] | None = None,
    status_renderer: Callable[[], str] | None = None,
    watch_db: MonitorDatabase | None = None,
    sns_db: SnsDatabase | None = None,
    sns_buzz_fn: Callable[[str], str] | None = None,
    opportunity_status_renderer: Callable[[], str] | None = None,
    opportunity_target_remover: OpportunityTargetRemover | None = None,
    opportunity_list_provider: Callable[[], list[dict[str, object]]] | None = None,
    opportunity_alias_updater: OpportunityAliasUpdater | None = None,
    opportunity_target_pinner: OpportunityTargetPinner | None = None,
    opportunity_target_unpinner: OpportunityTargetUnpinner | None = None,
    knowledge_handler: Callable[[str, str], str] | None = None,
    feedback_service: "object | None" = None,
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
        photo_intent_analyzer=photo_intent_analyzer,
        reputation_renderer=reputation_renderer,
        research_renderer=research_renderer,
        natural_language_router=natural_language_router,
        allowed_chat_ids=allowed_chat_ids,
        status_renderer=status_renderer,
        watch_db=watch_db,
        sns_db=sns_db,
        sns_buzz_fn=sns_buzz_fn,
        opportunity_status_renderer=opportunity_status_renderer,
        opportunity_target_remover=opportunity_target_remover,
        opportunity_list_provider=opportunity_list_provider,
        opportunity_alias_updater=opportunity_alias_updater,
        opportunity_target_pinner=opportunity_target_pinner,
        opportunity_target_unpinner=opportunity_target_unpinner,
        knowledge_handler=knowledge_handler,
        feedback_service=feedback_service,
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
                callback_query = update.get("callback_query")
                if isinstance(callback_query, dict):
                    handle_telegram_callback_query(
                        client=client,
                        processor=processor,
                        callback_query=callback_query,
                        photo_renderer=resolved_photo_renderer,
                    )
                    continue
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


def _handle_condition_callback(
    *,
    processor: TelegramCommandProcessor,
    watch_id: str,
    action: str,
    extra: str,
    original_text: str,
) -> tuple[str | None, str | None, dict[str, object] | None]:
    """Resolve a `cond:<watch_id>:<action>` callback.

    Returns ``(toast_text, new_message_text, new_reply_markup)``. When
    ``new_message_text`` is None the caller should NOT edit the message;
    the toast still fires so the user gets feedback.
    """
    watch_db = getattr(processor, "_watch_db", None)
    if watch_db is None or not watch_id:
        return "追蹤功能未啟用", None, None
    watch = watch_db.get_marketplace_watch(watch_id)
    if watch is None:
        return "找不到該追蹤", None, None
    if "mercari" not in watch.markets:
        # Condition picker only applies to Mercari (the only marketplace with
        # a condition enum). Rakuma/etc. watches surface no button.
        return "此追蹤未含 Mercari 平台，無狀態條件可設", None, None

    current_condition_ids = _condition_ids_from_options(watch.options_for("mercari"))

    if action == "open":
        text, kb = _build_condition_picker_view(
            watch_id=watch_id, query=watch.query, condition_ids=current_condition_ids,
        )
        return None, text, kb

    if action == "done":
        result = _render_watch_edit_view(processor, watch_id)
        if result:
            text, kb = result
        else:
            text, kb, _ = processor.render_watchlist_view(mode=LIST_VIEW_MODE_EDIT)
        return "✓ 已更新狀態條件", text, kb

    if action == "t" and extra.isdigit():
        cid = int(extra)
        if cid not in MERCARI_CONDITION_LABELS:
            return "未知狀態", None, None
        current = set(current_condition_ids)
        if cid in current:
            if len(current) == 1:
                # Re-render the picker with the same state — visual feedback only.
                text, kb = _build_condition_picker_view(
                    watch_id=watch_id, query=watch.query, condition_ids=current_condition_ids,
                )
                return "至少要保留一個狀態", text, kb
            current.remove(cid)
        else:
            current.add(cid)
        new_ids = tuple(sorted(current))
        merged_options = {
            market: dict(opts) for market, opts in watch.market_options.items()
        }
        mercari_opts = dict(merged_options.get("mercari") or {})
        mercari_opts["condition_ids"] = list(new_ids)
        merged_options["mercari"] = mercari_opts
        watch_db.update_marketplace_watch(watch_id, market_options=merged_options)
        # Re-fetch (defensive) and re-render picker with the new state.
        watch = watch_db.get_marketplace_watch(watch_id)
        next_condition_ids = (
            _condition_ids_from_options(watch.options_for("mercari"))
            if watch is not None else new_ids
        )
        text, kb = _build_condition_picker_view(
            watch_id=watch_id, query=(watch.query if watch else ""), condition_ids=next_condition_ids,
        )
        return None, text, kb

    return "未知操作", None, None


def _handle_sns_bulk_update_callback(
    *,
    processor: TelegramCommandProcessor,
    action: str,
    chat_id: str | int,
    original_text: str,
) -> tuple[str | None, str | None, dict[str, object] | None]:
    """Resolve a `bulk:c` / `bulk:x` callback (confirm / cancel of an SNS
    bulk filter-add preview). Returns (toast, edit_text, edit_reply_markup).

    The pending preview state is popped on either action so a second tap is
    harmless. Confirm applies the keyword merge via
    ``sns_monitor.bulk_filter.apply_bulk_keyword_filter_add`` against the
    rule ids captured at preview time — fresh DB lookups, so any concurrent
    deletion shows up as fewer updates.
    """
    pending = processor.pop_pending_sns_bulk_update(chat_id)
    if pending is None:
        return "操作已過期，請重新輸入", f"{original_text}\n\n（操作已過期）", None

    if action == "x":
        return "已取消", f"{original_text}\n\n（已取消）", None

    if action != "c":
        return "未知操作", None, None

    if processor._sns_db is None:
        return "SNS 監控未啟用", f"{original_text}\n\n（SNS 監控未啟用）", None

    try:
        from sns_monitor.bulk_filter import (
            apply_bulk_keyword_filter_add,
            apply_bulk_keyword_filter_remove,
            apply_bulk_schedule_update,
        )
    except ImportError:
        logger.exception("sns_monitor.bulk_filter import failed in bulk callback")
        return "套用失敗", f"{original_text}\n\n（套用失敗：模組載入錯誤）", None

    fresh_accounts = []
    for rule_id in pending.affected_rule_ids:
        rule = processor._sns_db.get_watch_rule(rule_id)
        if rule is not None:
            fresh_accounts.append(rule)

    if pending.action == "remove":
        updated = apply_bulk_keyword_filter_remove(
            processor._sns_db, fresh_accounts, pending.keywords
        )
        kw_display = "、".join(pending.keywords)
        if not updated:
            return (
                "沒有需要更新的帳號",
                f"{original_text}\n\n（沒有帳號需要更新，沒人有「{kw_display}」）",
                None,
            )
        toast = f"已修改 {len(updated)} 個帳號"
        new_text = f"{original_text}\n\n✓ 已從 {len(updated)} 個帳號 filter 移除「{kw_display}」"
        return toast, new_text, None

    if pending.action == "set_schedule":
        if pending.schedule_minutes is None:
            return "套用失敗", f"{original_text}\n\n（套用失敗：schedule_minutes 未設定）", None
        updated = apply_bulk_schedule_update(
            processor._sns_db, fresh_accounts, pending.schedule_minutes
        )
        if not updated:
            return (
                "沒有需要更新的帳號",
                f"{original_text}\n\n（沒有帳號需要更新，全部已經是 {pending.schedule_minutes} 分鐘）",
                None,
            )
        toast = f"已修改 {len(updated)} 個帳號"
        new_text = f"{original_text}\n\n✓ 已把 {len(updated)} 個帳號的 schedule 改成 {pending.schedule_minutes} 分鐘"
        return toast, new_text, None

    # Default / "add"
    updated = apply_bulk_keyword_filter_add(
        processor._sns_db, fresh_accounts, pending.keywords
    )

    kw_display = "、".join(pending.keywords)
    if not updated:
        return (
            "沒有需要更新的帳號",
            f"{original_text}\n\n（沒有帳號需要更新，全部已經有「{kw_display}」）",
            None,
        )
    toast = f"已修改 {len(updated)} 個帳號"
    new_text = f"{original_text}\n\n✓ 已修改 {len(updated)} 個帳號 filter 加上「{kw_display}」"
    return toast, new_text, None


def _list_view_renderer(processor: TelegramCommandProcessor, list_kind: str):
    """Return the ``render_*_view`` method for the given list kind, or None."""
    return {
        "sl": getattr(processor, "render_snslist_view", None),
        "wl": getattr(processor, "render_watchlist_view", None),
        "hl": getattr(processor, "render_huntlist_view", None),
    }.get(list_kind)


def _list_item_deleter(processor: TelegramCommandProcessor, list_kind: str):
    """Return ``(callable(item_id) -> bool, label)`` for the given list kind, or (None, None)."""
    if list_kind == "sl":
        return processor.delete_sns_rule_by_id, "SNS 規則"
    if list_kind == "wl":
        return processor.delete_marketplace_watch_by_id, "Marketplace 追蹤"
    if list_kind == "hl":
        return processor.delete_huntlist_item_by_id, "Opportunity 候選"
    return None, None


def handle_telegram_callback_query(
    *,
    client: TelegramBotClient,
    processor: TelegramCommandProcessor,
    callback_query: dict[str, object],
    photo_renderer: PhotoLookupRenderer | None = None,
) -> None:
    """Dispatch a Telegram callback_query (e.g. inline-button tap) to its handler.

    Callback-data dispatch table:
      - ``snsdel:<handle>``         → notification one-tap delete (legacy path
        used by the SNS auto-add notice).
      - ``pg:<list>:<page>:<mode>`` → repaginate / toggle a list view
        (read mode ↔ edit mode), where ``list`` is ``sl`` / ``wl`` / ``hl``.
      - ``del:<list>:<id>``         → delete one list row and re-render the
        same page in edit mode (drops back a page if the row was the last
        one on the last page).
      - ``close:<list>``            → clear the inline keyboard and mark the
        message as closed.
      - ``popt:<N>`` / ``topt:<N>`` → pick option N from a pending photo /
        text clarification — the same action the user would have triggered by
        typing the digit N as a normal message.
    """
    callback_id = callback_query.get("id")
    data = callback_query.get("data")
    message = callback_query.get("message")
    if not isinstance(callback_id, str) or not isinstance(data, str) or not isinstance(message, dict):
        return

    chat = message.get("chat")
    chat_id = chat.get("id") if isinstance(chat, dict) else None
    message_id = message.get("message_id")
    original_text = message.get("text") if isinstance(message.get("text"), str) else ""

    if chat_id is None or not isinstance(message_id, int):
        return

    if not processor.is_allowed_chat(chat_id):
        logger.warning(
            "Rejected Telegram callback_query from unauthorized chat_id=%s",
            mask_identifier(chat_id),
        )
        try:
            client.answer_callback_query(callback_query_id=callback_id)
        except Exception:
            logger.exception("answer_callback_query failed for unauthorized chat_id=%s", mask_identifier(chat_id))
        return

    prefix, _, payload = data.partition(":")
    toast: str | None = "未知按鈕"
    new_text: str | None = None
    new_reply_markup: dict[str, object] | None = None
    rerender = False

    if prefix == "snsdel" and payload:
        # Notification one-tap delete. The storage layer (delete_watch_rule)
        # automatically appends a polarity='negative' row to
        # sns_auto_discovery_feedback + ratchets the per-domain actionable
        # threshold up by DISCOVERY_TIGHTENING_STEP — see
        # sns_monitor.storage.SnsDatabase.record_auto_discovery_feedback.
        handle = payload.lstrip("@")
        reply = processor._handle_sns_delete(f"@{handle}")
        if reply.startswith("✓"):
            toast = f"已刪除 @{handle}"
            new_text = f"{original_text}\n\n✓ 已刪除 @{handle}"
            rerender = True
        elif reply.startswith("找不到"):
            toast = f"已經不在追蹤 @{handle}"
            new_text = f"{original_text}\n\n✓ 已刪除 @{handle}（先前已移除）"
            rerender = True
        else:
            toast = reply[:200]
    elif prefix == "snsaddok" and payload:
        # Notification one-tap positive feedback. Looks up the rule's
        # domains and writes polarity='positive' to the feedback table —
        # bumps keep_count for each domain, leaves the threshold untouched
        # (the learning loop tightens-only on negatives, never loosens).
        handle = payload.lstrip("@")
        sns_db = getattr(processor, "_sns_db", None)
        if sns_db is None:
            toast = "SNS monitor 未啟用，無法寫入回饋"
        else:
            try:
                from sns_monitor.models import AccountWatch as _AccountWatch
                rule = next(
                    (
                        r
                        for r in sns_db.list_watch_rules()
                        if isinstance(r, _AccountWatch)
                        and (r.screen_name or "").lower() == handle.lower()
                    ),
                    None,
                )
                domains = tuple(getattr(rule, "domains", ()) or ()) if rule else ()
                sns_db.record_auto_discovery_feedback(
                    screen_name=handle,
                    polarity="positive",
                    domains=domains,
                    chat_id=str(chat_id),
                )
                toast = f"👍 已記錄 @{handle}"
                new_text = f"{original_text}\n\n👍 已記錄為投資訊號帳號"
                rerender = True
            except Exception:
                logger.exception("snsaddok: positive feedback failed handle=@%s", handle)
                toast = "回饋寫入失敗"
    elif prefix == "snsfb" and payload:
        # Inline-button feedback on a single SNS post (account or keyword watch).
        # Payload format: "<kind>:<tweet_id>:<rule_id>" where kind ∈ {up,down,bought}.
        parts = payload.split(":", 2)
        if len(parts) != 3 or parts[0] not in {"up", "down", "bought"}:
            toast = "未知回饋"
        else:
            kind, tweet_id, rule_id = parts
            try:
                from sns_monitor.feedback import record_sns_feedback
                from sns_monitor.storage import SnsDatabase
                sns_db_path = (
                    processor._sns_db.path if processor._sns_db is not None
                    else None
                )
                if sns_db_path is None:
                    toast = "SNS monitor 未啟用，無法寫入回饋"
                else:
                    db = SnsDatabase(sns_db_path)
                    result = record_sns_feedback(
                        db=db, tweet_id=tweet_id, rule_id=rule_id,
                        chat_id=str(chat_id), kind=kind,
                    )
            except Exception:
                logger.exception(
                    "snsfb feedback failed tweet_id=%s rule_id=%s kind=%s",
                    tweet_id, rule_id, kind,
                )
                toast = "回饋寫入失敗，請看 log"
            else:
                if "result" not in locals() or result.get("status") != "ok":
                    reason = (result.get("reason", "unknown")
                              if "result" in locals() else "sns_db unavailable")
                    toast = f"記錄失敗：{reason}"
                else:
                    side_effects = list(result.get("side_effects") or ())
                    if kind == "up":
                        toast = "✓ 已記錄 👍"
                    elif kind == "bought":
                        if "rule_schedule_shortened" in side_effects:
                            new_minutes = result.get("new_schedule_minutes")
                            toast = f"✓ 已記錄 💰（rule 加速檢查 → {new_minutes} 分鐘）"
                        else:
                            toast = "✓ 已記錄 💰（schedule 已達 floor）"
                    else:  # down
                        if "rule_disabled" in side_effects:
                            toast = "✓ 已標記不感興趣（累計過閾值，rule 自動停用）"
                        else:
                            toast = "✓ 已標記不感興趣（24h cooldown）"
                    new_text = f"{original_text}\n\n{toast}"
                    new_reply_markup = None
                    rerender = True
    elif prefix == "oppfb" and payload:
        # Inline-button feedback on an Opportunity recommendation.
        # Payload format: "<kind>:<recommendation_id>" where kind ∈ {up,down,bought}.
        kind, _, rec_id = payload.partition(":")
        if kind not in {"up", "down", "bought"} or not rec_id:
            toast = "未知回饋"
        else:
            try:
                from openclaw_adapter.opportunity_feedback import (
                    record_opportunity_feedback,
                )
                result = record_opportunity_feedback(
                    recommendation_id=rec_id,
                    kind=kind,
                    collab_backfiller=self._collab_backfiller,
                )
            except Exception:
                logger.exception("oppfb feedback failed rec_id=%s kind=%s", rec_id, kind)
                toast = "回饋寫入失敗，請看 log"
            else:
                if result.get("status") != "ok":
                    toast = f"記錄失敗：{result.get('reason', 'unknown')}"
                else:
                    side_effects = list(result.get("side_effects") or ())
                    if kind == "up":
                        toast = "✓ 已記錄 👍" + (" + 升級為目標" if "promoted_to_target" in side_effects else "")
                    elif kind == "bought":
                        toast = "✓ 已記錄 💰" + (" + 升級為目標" if "promoted_to_target" in side_effects else "")
                    else:  # down
                        if "auto_dismissed" in side_effects:
                            toast = "✓ 已標記不感興趣（累計過閾值，自動 dismiss）"
                        else:
                            toast = "✓ 已標記不感興趣（24h cooldown）"
                    new_text = f"{original_text}\n\n{toast}"
                    new_reply_markup = None  # clear buttons; reply_markup=None means edit_message_text sets no keyboard
                    rerender = True
    elif prefix == "pg" and payload.count(":") == 2:
        list_kind, page_str, mode = payload.split(":", 2)
        renderer = _list_view_renderer(processor, list_kind)
        if renderer is None:
            toast = "未知清單"
        else:
            try:
                page = int(page_str)
            except ValueError:
                page = 0
            new_text, new_reply_markup, _ = renderer(page=page, mode=mode)
            toast = None
            rerender = True
    elif prefix == "del" and payload.count(":") == 1:
        list_kind, item_id = payload.split(":", 1)
        deleter, label = _list_item_deleter(processor, list_kind)
        renderer = _list_view_renderer(processor, list_kind)
        if deleter is None or renderer is None:
            toast = "未知清單"
        else:
            removed = deleter(item_id)
            toast = "✓ 已刪除" if removed else "已經不在清單"
            # Re-read the same page in edit mode; helper auto-clamps if the
            # last item on the last page just vanished.
            new_text, new_reply_markup, _ = renderer(page=_guess_current_page(original_text), mode=LIST_VIEW_MODE_EDIT)
            rerender = True
    elif prefix == "close" and payload:
        list_kind = payload
        if _list_view_renderer(processor, list_kind) is None:
            toast = "未知清單"
        else:
            new_text = f"{original_text}\n\n（已關閉）"
            new_reply_markup = None
            toast = None
            rerender = True
    elif prefix == "cond" and payload:
        # `cond:<watch_id>:open` | `cond:<watch_id>:t:<id>` | `cond:<watch_id>:done`
        parts = payload.split(":")
        watch_id = parts[0] if parts else ""
        action = parts[1] if len(parts) >= 2 else ""
        toast_out, edit_text, edit_kb = _handle_condition_callback(
            processor=processor,
            watch_id=watch_id,
            action=action,
            extra=parts[2] if len(parts) >= 3 else "",
            original_text=original_text,
        )
        if toast_out is not None:
            toast = toast_out
        if edit_text is not None:
            new_text = edit_text
            new_reply_markup = edit_kb
            rerender = True
    elif prefix == "bulk" and payload in ("c", "x"):
        # bulk SNS-filter update preview confirm / cancel.
        toast_out, edit_text, edit_kb = _handle_sns_bulk_update_callback(
            processor=processor,
            action=payload,
            chat_id=chat_id,
            original_text=original_text,
        )
        if toast_out is not None:
            toast = toast_out
        if edit_text is not None:
            new_text = edit_text
            new_reply_markup = edit_kb
            rerender = True
    elif prefix in ("popt", "topt") and payload.isdigit():
        option_n = int(payload)
        is_photo = prefix == "popt"
        # Peek at the pending state to find the prompt for this option — we
        # need it for the audit line before `build_pending_*_reply_plan`
        # consumes it.
        pending = (
            processor.get_pending_photo_clarification(chat_id) if is_photo
            else processor.get_pending_text_clarification(chat_id)
        )
        if pending is None:
            toast = "選項已處理或過期"
            new_text = f"{original_text}\n\n（選項已過期）"
            new_reply_markup = None
            rerender = True
        else:
            picked_prompt = next(
                (o.prompt for o in pending.options if o.option_number == option_n),
                None,
            )
            if picked_prompt is None:
                toast = "找不到該選項"
            else:
                new_text = f"{original_text}\n\n✓ 已選 {option_n}. {picked_prompt}"
                new_reply_markup = None
                rerender = True
                toast = f"已選 {option_n}"
                # Run the same plan the user would have triggered by typing N.
                if is_photo:
                    plan = processor.build_pending_photo_reply_plan(
                        chat_id=chat_id, text=str(option_n), photo_renderer=photo_renderer
                    )
                else:
                    plan = processor.build_pending_text_reply_plan(
                        chat_id=chat_id, text=str(option_n)
                    )
                if plan is not None:
                    try:
                        _send_text_reply_plan(client=client, chat_id=chat_id, plan=plan)
                    except Exception:
                        logger.exception(
                            "Sending clarification plan from callback failed chat_id=%s prefix=%s n=%d",
                            mask_identifier(chat_id), prefix, option_n,
                        )
    elif prefix == "wedit" and payload:
        # Open single-watch detailed edit view
        result = _render_watch_edit_view(processor, payload)
        if result:
            new_text, new_reply_markup = result
            rerender = True
        else:
            toast = "找不到該追蹤"

    elif prefix == "wmkt" and payload.count(":") == 1:
        # Toggle a marketplace on/off for a watch
        watch_id, market = payload.split(":", 1)
        watch_db = getattr(processor, "_watch_db", None)
        if watch_db and watch_id:
            watch = watch_db.get_marketplace_watch(watch_id)
            if watch:
                current = set(watch.markets)
                name, emoji = _marketplace_source_display(market)
                if market in current:
                    if len(current) == 1:
                        toast = "至少要保留一個平台"
                    else:
                        current.remove(market)
                        new_mkt = tuple(m for m in DEFAULT_MARKETS if m in current)
                        watch_db.update_marketplace_watch(watch_id, markets=new_mkt)
                        toast = f"已移除 {emoji}{name}"
                else:
                    current.add(market)
                    new_mkt = tuple(m for m in DEFAULT_MARKETS if m in current)
                    watch_db.update_marketplace_watch(watch_id, markets=new_mkt)
                    toast = f"已加入 {emoji}{name}"
                result = _render_watch_edit_view(processor, watch_id)
                if result:
                    new_text, new_reply_markup = result
                    rerender = True
            else:
                toast = "找不到該追蹤"

    elif prefix == "wprc" and payload:
        # Send ForceReply message asking for new price
        watch_db = getattr(processor, "_watch_db", None)
        if watch_db:
            watch = watch_db.get_marketplace_watch(payload)
            if watch:
                fr_text = (
                    f"請輸入「{watch.query}」的新目標上限（日圓整數，例：25000）：\n"
                    f"[wprc:{payload}]"
                )
                try:
                    client.send_message(
                        chat_id=chat_id,
                        text=fr_text,
                        reply_markup={"force_reply": True, "selective": True},
                    )
                except Exception:
                    logger.exception("wprc: ForceReply send failed watch_id=%s", payload)
                    toast = "發送失敗"
            else:
                toast = "找不到該追蹤"

    elif prefix == "fbprc" and payload:
        # Price-feedback callback: store pending state + send ForceReply asking for URL.
        feedback_service = getattr(processor, "_feedback_service", None)
        watch_db = getattr(processor, "_watch_db", None)
        if feedback_service is None or watch_db is None:
            toast = "回饋功能未啟用"
        else:
            item = watch_db.find_item(payload)
            if item is None:
                toast = "找不到該品項，可能已過期"
            else:
                fair_value = watch_db.latest_fair_value_for(payload)
                processor.set_pending_price_feedback(
                    PendingTelegramPriceFeedback(
                        chat_id=str(chat_id),
                        item_id=payload,
                        original_fair_value_jpy=fair_value,
                    )
                )
                fr_text = (
                    "請貼上你覺得合理價格的參考 URL（必須是公開可讀的網頁）：\n"
                    f"[fbprc:{payload}]"
                )
                try:
                    client.send_message(
                        chat_id=chat_id,
                        text=fr_text,
                        reply_markup={"force_reply": True, "selective": True},
                    )
                    toast = "好的，等你貼 URL"
                except Exception:
                    logger.exception("fbprc: ForceReply send failed item_id=%s", payload)
                    toast = "發送失敗"

    elif prefix == "fbpos" and payload:
        # Positive-feedback callback: one-tap thumbs-up, no ForceReply.
        # Writes a polarity='positive' row directly via feedback_service.
        feedback_service = getattr(processor, "_feedback_service", None)
        watch_db = getattr(processor, "_watch_db", None)
        if feedback_service is None or watch_db is None:
            toast = "回饋功能未啟用"
        else:
            item = watch_db.find_item(payload)
            if item is None:
                toast = "找不到該品項，可能已過期"
            else:
                fair_value = watch_db.latest_fair_value_for(payload)
                try:
                    from tcg_tracker.catalog import TcgCardSpec as _TcgCardSpec
                    spec = _TcgCardSpec.from_tracked_item(item)
                except Exception:
                    logger.exception("fbpos: failed to build spec from item_id=%s", payload)
                    toast = "解析品項失敗"
                else:
                    try:
                        feedback_service.submit_positive(
                            item=item,
                            spec=spec,
                            chat_id=chat_id,
                            original_fair_value_jpy=fair_value,
                        )
                        toast = "👍 已記錄"
                    except Exception:
                        logger.exception("fbpos: submit_positive failed item_id=%s", payload)
                        toast = "回饋寫入失敗"

    elif prefix == "wback":
        # Return from watch edit view to watchlist edit mode
        text, kb, _ = processor.render_watchlist_view(mode=LIST_VIEW_MODE_EDIT)
        new_text = text
        new_reply_markup = kb
        rerender = True

    elif prefix == "noop":
        pass  # label buttons — silently acknowledge, no action
    else:
        logger.warning("Unknown callback_query prefix=%r data=%r", prefix, data)

    if rerender and new_text is not None:
        try:
            client.edit_message_text(
                chat_id=chat_id,
                message_id=message_id,
                text=new_text,
                reply_markup=new_reply_markup,
            )
        except Exception:
            logger.exception(
                "edit_message_text failed chat_id=%s message_id=%s",
                mask_identifier(chat_id),
                message_id,
            )

    try:
        client.answer_callback_query(callback_query_id=callback_id, text=toast)
    except Exception:
        logger.exception("answer_callback_query failed callback_id=%s", callback_id)


_PAGE_HEADER_RE = re.compile(r"第\s*(\d+)\s*/\s*(\d+)\s*頁")


def _guess_current_page(message_text: str) -> int:
    """Pull the 0-based page index out of a list-view header line.

    The header looks like "📋 SNS 監控規則  第 2/3 頁（共 13 筆）"; we read
    the "2" and return it as a 0-based index (so 2 → 1). Returns 0 if no
    header is found.
    """
    match = _PAGE_HEADER_RE.search(message_text)
    if not match:
        return 0
    return max(0, int(match.group(1)) - 1)


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
    text = message.get("text")
    text_value = text if isinstance(text, str) else None
    photo_items = message.get("photo")
    has_photo = isinstance(photo_items, list) and bool(photo_items)

    # ForceReply price-feedback URL — check before intake ack so it doesn't misfire NLP
    if text_value and not has_photo:
        reply_to = message.get("reply_to_message")
        if isinstance(reply_to, dict):
            rt_text = reply_to.get("text") or ""
            fbprc_m = _FBPRC_TAG_RE.search(rt_text) if isinstance(rt_text, str) else None
            if fbprc_m:
                pending = processor.pop_pending_price_feedback(chat_id)
                url = text_value.strip()
                if not (url.startswith("http://") or url.startswith("https://")):
                    client.send_message(chat_id=chat_id, text="⚠️ 這看起來不是合法 URL，回饋已取消。")
                    return ("⚠️ 不是合法 URL",)
                if pending is None:
                    client.send_message(chat_id=chat_id, text="⚠️ 該回饋對話已過期，請重新點擊「不合理」。")
                    return ("⚠️ 過期",)
                feedback_service = getattr(processor, "_feedback_service", None)
                watch_db = getattr(processor, "_watch_db", None)
                if feedback_service is None or watch_db is None:
                    client.send_message(chat_id=chat_id, text="⚠️ 回饋服務未啟用。")
                    return ("⚠️ 未啟用",)
                item = watch_db.find_item(pending.item_id)
                if item is None:
                    client.send_message(chat_id=chat_id, text="⚠️ 找不到對應品項。")
                    return ("⚠️ 品項遺失",)
                # Build TcgCardSpec from TrackedItem
                try:
                    from tcg_tracker.catalog import TcgCardSpec as _TcgCardSpec
                    spec = _TcgCardSpec.from_tracked_item(item)
                except Exception as exc:
                    logger.exception("fbprc consumer: failed to build spec from item_id=%s: %s", pending.item_id, exc)
                    client.send_message(chat_id=chat_id, text="⚠️ 解析品項失敗。")
                    return ("⚠️ 解析失敗",)
                try:
                    outcome = feedback_service.submit(
                        item=item,
                        spec=spec,
                        chat_id=chat_id,
                        original_fair_value_jpy=pending.original_fair_value_jpy,
                        claimed_url=url,
                    )
                except Exception:
                    logger.exception("Feedback service failed for chat_id=%s url=%s", chat_id, url)
                    client.send_message(chat_id=chat_id, text="⚠️ 抓取／分析時出現錯誤，已記錄。")
                    return ("⚠️ 內部錯誤",)
                client.send_message(chat_id=chat_id, text=outcome.summary_for_user)
                return (outcome.summary_for_user,)

            wprc_m = _WPRC_TAG_RE.search(rt_text) if isinstance(rt_text, str) else None
            if wprc_m:
                watch_id = wprc_m.group(1)
                price_str = text_value.strip().replace(",", "").replace("¥", "").replace("円", "")
                watch_db = getattr(processor, "_watch_db", None)
                if price_str.isdigit() and watch_db:
                    new_price = int(price_str)
                    watch = watch_db.get_marketplace_watch(watch_id)
                    if watch:
                        watch_db.update_marketplace_watch(watch_id, price_threshold_jpy=new_price)
                        result = _render_watch_edit_view(processor, watch_id)
                        reply_text = f"✓ 已更新上限 ¥{new_price:,}"
                        reply_markup = result[1] if result else None
                        msg_text = f"{reply_text}\n\n{result[0]}" if result else reply_text
                        client.send_message(chat_id=chat_id, text=msg_text, reply_markup=reply_markup)
                        return (reply_text,)
                else:
                    client.send_message(chat_id=chat_id, text="⚠️ 請輸入純數字，例：25000")
                    return ("⚠️ 格式錯誤",)

    # Immediate intake ack — the downstream pipeline can take a while
    # (vision/OCR/LLM calls), so let the user know we've received their
    # message before we start the real work.
    if has_photo:
        intake_ack = "已收到圖片，開始解讀使用者意圖"
    elif text_value is not None:
        intake_ack = "已收到訊息，開始解讀使用者意圖"
    else:
        intake_ack = None
    if intake_ack is not None:
        try:
            client.send_message(chat_id=chat_id, text=intake_ack)
        except Exception:
            logger.exception("Intake ack send failed chat_id=%s", mask_identifier(chat_id))
        else:
            replies.append(intake_ack)

    pending_plan = processor.build_pending_photo_reply_plan(
        chat_id=chat_id,
        text=text_value,
        photo_renderer=photo_renderer,
    )
    if pending_plan is not None:
        replies.extend(_send_text_reply_plan(client=client, chat_id=chat_id, plan=pending_plan))
        return tuple(replies)

    if not has_photo:
        text_pending_plan = processor.build_pending_text_reply_plan(
            chat_id=chat_id,
            text=text_value,
        )
        if text_pending_plan is not None:
            replies.extend(_send_text_reply_plan(client=client, chat_id=chat_id, plan=text_pending_plan))
            return tuple(replies)

    if has_photo:
        final_reply, ack, reply_markup = _handle_photo_message(
            client=client,
            processor=processor,
            photo_renderer=photo_renderer,
            chat_id=chat_id,
            message=message,
        )
        if ack:
            client.send_message(chat_id=chat_id, text=ack)
            replies.append(ack)
        client.send_message(chat_id=chat_id, text=final_reply, reply_markup=reply_markup)
        replies.append(final_reply)
        return tuple(replies)
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
    replies.extend(_send_text_reply_plan(client=client, chat_id=chat_id, plan=plan))
    return tuple(replies)


def _send_text_reply_plan(
    *,
    client: TelegramBotClient,
    chat_id: str | int,
    plan: TelegramTextReplyPlan,
) -> list[str]:
    """Send a TelegramTextReplyPlan to Telegram and return the texts sent.

    Single-source-of-truth for the "ack → reputation delivery → reply"
    sequence. Used by both the message-handler and the callback-query
    dispatcher so they can't drift.
    """
    sent: list[str] = []
    if plan.ack:
        client.send_message(chat_id=chat_id, text=plan.ack)
        sent.append(plan.ack)
    if plan.reputation_delivery_factory is not None:
        delivery = plan.reputation_delivery_factory()
        _send_reputation_delivery(client=client, chat_id=chat_id, delivery=delivery)
        sent.append(delivery.summary_text)
        return sent
    reply, factory_markup = plan._execute_unpacked()
    if reply:
        logger.debug(
            "Telegram reply sending chat_id=%s text=%s",
            mask_identifier(chat_id),
            trim_for_log(reply, limit=320),
        )
        # Factory-provided markup (from e.g. lookup feedback keyboard) takes
        # precedence over plan-level reply_markup; falls back to it when None.
        markup = factory_markup if factory_markup is not None else plan.reply_markup
        client.send_message(chat_id=chat_id, text=reply, reply_markup=markup)
        sent.append(reply)
    return sent


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
    processor: TelegramCommandProcessor,
    photo_renderer: PhotoLookupRenderer,
    chat_id: str | int,
    message: dict[str, object],
) -> tuple[str, str | None, dict[str, object] | None]:
    caption = message.get("caption")
    caption_text = caption if isinstance(caption, str) else None
    photo_items = message.get("photo")
    if not isinstance(photo_items, list) or not photo_items:
        return "No image was attached.", None, None

    candidates = [item for item in photo_items if isinstance(item, dict) and item.get("file_id")]
    if not candidates:
        return "Could not resolve the Telegram file metadata for this image.", None, None

    best_item = max(
        candidates,
        key=lambda item: int(item.get("file_size") or 0),
    )
    file_id = best_item.get("file_id")
    if not isinstance(file_id, str):
        return "Could not resolve the Telegram file id for this image.", None, None

    game_hint, title_hint = _parse_photo_caption_for_lookup(caption_text)

    try:
        local_path = _download_telegram_photo_to_temp(client=client, file_id=file_id)
        query = TelegramPhotoQuery(
            chat_id=chat_id,
            image_path=local_path,
            caption=caption_text,
            game_hint=game_hint,
            title_hint=title_hint,
            item_kind_hint=None,
            file_id=file_id,
        )
        if _caption_requests_direct_photo_lookup(caption_text):
            raw_reply = photo_renderer(query)
            reply = _coerce_photo_lookup_reply(raw_reply)
            installed_pending = False
            try:
                if reply.pending_clarification is not None:
                    processor.set_pending_photo_clarification(reply.pending_clarification)
                    installed_pending = True
                    # When the renderer routed the user into a clarification
                    # we mint the same per-option keyboard the ambiguous-intake
                    # path uses, so direct-lookup fallbacks get one-tap selection too.
                    kb = _build_clarification_keyboard("popt", reply.pending_clarification.options)
                else:
                    # No clarification — pass through whatever keyboard the
                    # renderer attached (e.g. the price-feedback ❌ button).
                    kb = reply.reply_markup
                return reply.text, reply.ack or build_processing_ack(has_photo=True), kb
            finally:
                if not installed_pending:
                    try:
                        local_path.unlink(missing_ok=True)
                    except PermissionError:
                        logger.debug("Could not remove temporary Telegram photo path=%s", local_path)

        analysis = processor.analyze_photo_intent(query)
        processor.set_pending_photo_clarification(
            PendingTelegramPhotoClarification(
                chat_id=str(chat_id),
                image_path=local_path,
                caption=caption_text,
                file_id=file_id,
                options=analysis.options,
                parsed_game=analysis.parsed_game,
                parsed_item_kind=analysis.parsed_item_kind,
                parsed_title=analysis.parsed_title,
            )
        )
        text, kb = _build_photo_clarification_reply(analysis)
        return text, None, kb
    except Exception as exc:  # pragma: no cover - network-dependent.
        logger.exception("Telegram photo handling failed chat_id=%s file_id=%s", mask_identifier(chat_id), file_id)
        return f"Image lookup failed: {exc}", None, None


def _download_telegram_photo_to_temp(*, client: TelegramBotClient, file_id: str) -> Path:
    file_info = client.get_file(file_id=file_id)
    file_path = file_info.get("file_path")
    if not isinstance(file_path, str) or not file_path:
        raise RuntimeError("Telegram did not return a downloadable file path for this image.")
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
        return Path(handle.name)


def _execute_pending_photo_lookup(
    *,
    pending: PendingTelegramPhotoClarification,
    option: TelegramPhotoIntentOption,
    photo_renderer: PhotoLookupRenderer,
) -> "str | tuple[str, dict[str, object] | None]":
    query = TelegramPhotoQuery(
        chat_id=pending.chat_id,
        image_path=pending.image_path,
        caption=option.synthetic_caption,
        game_hint=_parse_photo_caption_for_lookup(option.synthetic_caption)[0],
        title_hint=None,
        item_kind_hint="sealed_box" if option.action_key == "pokemon_box_price" else "card",
        file_id=pending.file_id,
    )
    try:
        raw_reply = photo_renderer(query)
        reply = _coerce_photo_lookup_reply(raw_reply)
        # When the renderer attached a feedback keyboard, return a tuple so
        # the reply_factory plumbing in TelegramTextReplyPlan picks it up.
        if reply.reply_markup is not None:
            return reply.text, reply.reply_markup
        return reply.text
    finally:
        try:
            pending.image_path.unlink(missing_ok=True)
        except PermissionError:
            logger.debug("Could not remove pending Telegram photo path=%s", pending.image_path)


def _coerce_photo_lookup_reply(value: "str | PhotoLookupReply") -> PhotoLookupReply:
    if isinstance(value, PhotoLookupReply):
        return value
    return PhotoLookupReply(text=str(value))


def _build_pending_photo_retry_reply(
    options: tuple[TelegramPhotoIntentOption, ...],
) -> tuple[str, dict[str, object] | None]:
    lines = ["我現在在等你確認這張圖要怎麼處理，請點按鈕（或回覆數字）："]
    for option in options:
        lines.append(f"{option.option_number}. {option.prompt}")
    lines.append("或輸入：否，[您的意圖]")
    return "\n".join(lines), _build_clarification_keyboard("popt", options)


def _match_photo_clarification_option(
    content: str,
    options: tuple[TelegramPhotoIntentOption, ...],
) -> TelegramPhotoIntentOption | None:
    stripped = content.strip()
    for option in options:
        if stripped == str(option.option_number):
            return option
    return None


def _extract_photo_clarification_override(content: str) -> str | None:
    stripped = content.strip()
    for prefix in ("否，", "否,", "否:", "否：", "否 ", "都不是，", "都不是,", "都不是:", "都不是：", "不是，", "不是,", "no,", "no:"):
        if stripped.lower().startswith(prefix.lower()):
            remainder = stripped[len(prefix):].strip()
            return remainder or ""
    if stripped in {"否", "都不是", "不是", "no"}:
        return ""
    return None


def _resolve_photo_override_to_option(
    override_text: str,
    options: tuple[TelegramPhotoIntentOption, ...],
) -> TelegramPhotoIntentOption | None:
    lowered = override_text.lower()
    if not _text_requests_price_lookup(lowered):
        return None
    desired_action = None
    if "遊戲王" in override_text or "yugioh" in lowered or "yu-gi-oh" in lowered:
        desired_action = "yugioh_card_price"
    elif "weiss" in lowered or "ws" in lowered or "ヴァイス" in override_text:
        desired_action = "ws_card_price"
    elif "union arena" in lowered or "ユニオン" in override_text or "ua " in lowered:
        desired_action = "union_arena_card_price"
    elif "box" in lowered or "盒" in override_text or "ボックス" in override_text:
        desired_action = "pokemon_box_price"
    elif "寶可夢" in override_text or "pokemon" in lowered:
        desired_action = "pokemon_card_price"
    elif ("卡" in override_text or "card" in lowered) and options:
        return options[0]
    if desired_action is None:
        return None
    for option in options:
        if option.action_key == desired_action:
            return option
    return None


def _build_photo_identify_reply(
    *,
    pending: PendingTelegramPhotoClarification,
    override_text: str,
) -> str | None:
    lowered = override_text.lower()
    if not any(token in lowered for token in ("identify", "what is", "辨識", "辨认", "辨識", "是什麼", "这是什么", "這是什麼")):
        return None
    if pending.parsed_game is None:
        return "我目前只能先判斷這大概是 TCG 相關圖片，但還沒辦法穩定確認是哪個系列。"
    item_label = "卡盒" if pending.parsed_item_kind == "sealed_box" else "卡片"
    title_suffix = f"：{pending.parsed_title}" if pending.parsed_title else ""
    return f"我目前看起來比較像 {_display_game_name(pending.parsed_game)}{item_label}{title_suffix}"


def _is_text_intent_ambiguous(intent: "TelegramNaturalLanguageIntent | None") -> bool:
    if intent is None:
        return True
    if intent.intent == "unknown":
        return True
    if intent.confidence is None:
        return False
    return intent.confidence < TEXT_AMBIGUITY_CONFIDENCE_THRESHOLD


def _build_text_intent_candidates(
    text: str,
    top_intent: "TelegramNaturalLanguageIntent | None",
) -> tuple[TelegramTextIntentOption, ...]:
    raw_candidates: list[tuple[str, str, TelegramNaturalLanguageIntent]] = []
    seen_keys: set[str] = set()

    def add(key: str, prompt: str, intent: "TelegramNaturalLanguageIntent") -> None:
        if key in seen_keys:
            return
        seen_keys.add(key)
        raw_candidates.append((key, prompt, intent))

    lowered = text.lower()

    if top_intent is not None and top_intent.intent != "unknown":
        prompt = _describe_intent_for_clarification(top_intent, text)
        if prompt is not None:
            add(f"top_{top_intent.intent}", prompt, top_intent)

    game_hint = _infer_game_hint_from_text(text)
    if game_hint:
        lookup_guess = TelegramNaturalLanguageIntent(
            intent="lookup_card",
            game=game_hint,
            name=text.strip(),
            confidence=0.5,
        )
        prompt = _describe_intent_for_clarification(lookup_guess, text)
        if prompt is not None:
            add(f"lookup_{game_hint}", prompt, lookup_guess)

    handle_match = re.search(r"@([A-Za-z0-9_]{2,32})", text)
    if handle_match is not None:
        handle = handle_match.group(1)
        sns_guess = TelegramNaturalLanguageIntent(
            intent="sns_add_account",
            sns_handle=handle,
            confidence=0.5,
        )
        prompt = _describe_intent_for_clarification(sns_guess, text)
        if prompt is not None:
            add(f"sns_add_{handle.lower()}", prompt, sns_guess)

    if game_hint and any(token in lowered for token in ("trending", "hot", "熱門", "排行", "流動性", "trend")):
        trend_guess = TelegramNaturalLanguageIntent(
            intent="trend_board",
            game=game_hint,
            limit=5,
            confidence=0.5,
        )
        prompt = _describe_intent_for_clarification(trend_guess, text)
        if prompt is not None:
            add(f"trend_{game_hint}", prompt, trend_guess)

    url_match = re.search(r"https?://\S+", text)
    if url_match is not None:
        url = url_match.group(0)
        reputation_guess = TelegramNaturalLanguageIntent(
            intent="reputation_snapshot",
            query_url=url,
            confidence=0.5,
        )
        prompt = _describe_intent_for_clarification(reputation_guess, text)
        if prompt is not None:
            add("reputation_snapshot", prompt, reputation_guess)

    research_guess = TelegramNaturalLanguageIntent(
        intent="web_research",
        research_query=text.strip(),
        confidence=0.5,
    )
    research_prompt = _describe_intent_for_clarification(research_guess, text)
    if research_prompt is not None:
        add("web_research", research_prompt, research_guess)

    if len(raw_candidates) < 4:
        help_guess = TelegramNaturalLanguageIntent(intent="help", confidence=0.5)
        help_prompt = _describe_intent_for_clarification(help_guess, text)
        if help_prompt is not None:
            add("help", help_prompt, help_guess)

    if not raw_candidates:
        return ()

    trimmed = raw_candidates[:4]
    return tuple(
        TelegramTextIntentOption(
            option_number=index + 1,
            action_key=key,
            prompt=prompt,
            intent=intent,
        )
        for index, (key, prompt, intent) in enumerate(trimmed)
    )


def _describe_intent_for_clarification(
    intent: "TelegramNaturalLanguageIntent",
    original_text: str,
) -> str | None:
    fallback = trim_for_log(original_text.strip(), limit=60)
    kind = intent.intent
    if kind == "lookup_card":
        game_label = _display_game_name(intent.game) if intent.game else "TCG"
        name = intent.name or intent.card_number or fallback
        return f"查 {game_label} 卡片『{trim_for_log(name, limit=60)}』的市價 (相當於 /price)"
    if kind == "trend_board":
        game_label = _display_game_name(intent.game) if intent.game else "TCG"
        limit_text = f" {intent.limit}" if intent.limit else ""
        return f"看 {game_label} 的熱門卡排行 (相當於 /trend{limit_text})"
    if kind == "add_watch":
        q = intent.watch_query or fallback
        threshold = intent.watch_price_threshold
        if threshold:
            return f"追蹤 Mercari 商品『{trim_for_log(q, limit=40)}』在 {threshold} 円以下 (相當於 /watch)"
        return f"追蹤 Mercari 商品『{trim_for_log(q, limit=40)}』 (相當於 /watch)"
    if kind == "list_watches":
        return "列出 Mercari 追蹤清單 (相當於 /watchlist)"
    if kind == "remove_watch":
        target = intent.watch_id or fallback
        return f"取消 Mercari 追蹤 {trim_for_log(target, limit=40)} (相當於 /unwatch)"
    if kind == "update_watch_price":
        return f"更新 Mercari 追蹤 {intent.watch_id} 的目標價 (相當於 /setprice)"
    if kind == "reputation_snapshot":
        url = intent.query_url or fallback
        return f"建立賣家信譽快照：{trim_for_log(url, limit=40)} (相當於 /snapshot)"
    if kind == "web_research":
        q = intent.research_query or fallback
        return f"上網搜尋『{trim_for_log(q, limit=60)}』 (相當於 /search)"
    if kind == "opportunity_remove":
        target = intent.opportunity_target or fallback
        return f"從機會清單移除『{trim_for_log(target, limit=40)}』 (相當於 /hunt remove)"
    if kind == "help":
        return "顯示我能做什麼 (相當於 /help)"
    if kind == "status":
        return "顯示目前服務狀態 (相當於 /status)"
    if kind == "tools":
        return "列出工具目錄 (相當於 /tools)"
    if kind == "scan_help":
        return "說明如何用照片掃描卡片"
    if kind == "sns_add_account":
        handle = intent.sns_handle or fallback
        return f"追蹤 X 帳號 @{handle} (相當於 /snsadd)"
    if kind == "sns_add_keyword":
        keyword = intent.sns_keyword or fallback
        return f"監控 X 關鍵字『{trim_for_log(keyword, limit=40)}』 (相當於 /snsadd keyword:)"
    if kind == "sns_list":
        return "列出 SNS 監控規則 (相當於 /snslist)"
    if kind == "sns_delete":
        target = (intent.sns_handle and f"@{intent.sns_handle}") or intent.watch_id or fallback
        return f"取消 SNS 監控 {trim_for_log(target, limit=40)} (相當於 /snsdelete)"
    if kind == "sns_buzz":
        q = intent.sns_buzz_query or fallback
        return f"整理 X 熱門討論：{trim_for_log(q, limit=40)} (相當於 /snsbuzz)"
    return None


def _build_text_clarification_reply(
    original_text: str,
    options: tuple[TelegramTextIntentOption, ...],
    top_intent: "TelegramNaturalLanguageIntent | None",
) -> tuple[str, dict[str, object] | None]:
    if top_intent is not None and top_intent.intent != "unknown":
        head = "我看了一下你的訊息，但我還不想先亂猜你的意圖。"
    else:
        head = "我還不太確定你想叫我做什麼，所以先不亂動手。"
    lines = [head, "請點按鈕（或回覆數字）："]
    for option in options:
        lines.append(f"{option.option_number}. {option.prompt}")
    lines.append(f"{len(options) + 1}. 都不是，請回答：否，[您的意圖]")
    return "\n".join(lines), _build_clarification_keyboard("topt", options)


def _build_pending_text_retry_reply(
    options: tuple[TelegramTextIntentOption, ...],
) -> tuple[str, dict[str, object] | None]:
    lines = ["我現在在等你確認剛剛那則訊息要怎麼處理，請點按鈕（或回覆數字）："]
    for option in options:
        lines.append(f"{option.option_number}. {option.prompt}")
    lines.append("或輸入：否，[您的意圖]")
    return "\n".join(lines), _build_clarification_keyboard("topt", options)


def _match_text_clarification_option(
    content: str,
    options: tuple[TelegramTextIntentOption, ...],
) -> TelegramTextIntentOption | None:
    stripped = content.strip()
    for option in options:
        if stripped == str(option.option_number):
            return option
    return None


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
    first = normalize_game_key(tokens[0])
    if first is not None:
        remainder = " ".join(tokens[1:]).strip()
        return first, _sanitize_image_title_hint(remainder or None)
    inferred_game = _infer_game_hint_from_text(content)
    return inferred_game, _sanitize_image_title_hint(content)


def _caption_requests_direct_photo_lookup(caption: str | None) -> bool:
    if caption is None:
        return False
    content = caption.strip()
    if not content:
        return False
    lowered = content.lower()
    if any(lowered.startswith(prefix) for prefix in PHOTO_SCAN_COMMANDS):
        return True
    return _text_requests_price_lookup(lowered)


def _text_requests_price_lookup(content: str) -> bool:
    price_tokens = (
        "查價",
        "查价",
        "市價",
        "市价",
        "價格",
        "价格",
        "price",
        "value",
        "估價",
        "估价",
    )
    return any(token in content for token in price_tokens)


def _infer_game_hint_from_text(content: str) -> str | None:
    lowered = content.lower()
    if "寶可夢" in content or "宝可梦" in content or "pokemon" in lowered:
        return "pokemon"
    if "遊戲王" in content or "游戏王" in content or "yugioh" in lowered or "yu-gi-oh" in lowered:
        return "yugioh"
    if "weiss" in lowered or "ヴァイス" in content:
        return "ws"
    if "union arena" in lowered or "ユニオン" in content:
        return "union_arena"
    return None


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
