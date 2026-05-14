from __future__ import annotations

import json
import logging
import re
import ssl
from dataclasses import dataclass
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from tcg_tracker.catalog import normalize_game_key, supported_game_hint

logger = logging.getLogger(__name__)

_CHINESE_DIGIT_MAP: dict[str, int] = {
    "一": 1, "二": 2, "三": 3, "四": 4, "五": 5,
    "六": 6, "七": 7, "八": 8, "九": 9, "十": 10,
}

_ROUTER_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "intent": {"type": "string"},
        "game": {"type": ["string", "null"]},
        "name": {"type": ["string", "null"]},
        "card_number": {"type": ["string", "null"]},
        "rarity": {"type": ["string", "null"]},
        "set_code": {"type": ["string", "null"]},
        "limit": {"type": ["integer", "null"]},
        "confidence": {"type": ["number", "null"]},
        # watch fields
        "watch_query": {"type": ["string", "null"]},
        "watch_price_threshold": {"type": ["integer", "null"]},
        "watch_id": {"type": ["string", "null"]},
        # reputation snapshot field
        "query_url": {"type": ["string", "null"]},
        # SNS fields
        "sns_handle": {"type": ["string", "null"]},
        "sns_keyword": {"type": ["string", "null"]},
        "sns_buzz_query": {"type": ["string", "null"]},
    },
    "required": [
        "intent", "game", "name", "card_number", "rarity", "set_code", "limit", "confidence",
        "watch_query", "watch_price_threshold", "watch_id", "query_url",
        "sns_handle", "sns_keyword", "sns_buzz_query",
    ],
    "additionalProperties": False,
}

_LOOKUP_KEYWORDS = (
    "查",
    "估價",
    "價格",
    "價錢",
    "price",
    "lookup",
    "value",
)
_TREND_KEYWORDS = (
    "熱門",
    "排行",
    "趨勢",
    "熱度",
    "前",
    "trend",
    "trending",
    "hot",
    "heat",
    "liquidity",
)
_WATCH_ADD_KEYWORDS = (
    "追蹤",
    "監控",
    "盯",
    "watch",
    "提醒",
    "通知我",
    "低於",
    "以下",
    "以內",
    "watcher",
    "alert",
)
_WATCH_LIST_KEYWORDS = (
    "追蹤清單",
    "追蹤列表",
    "我的追蹤",
    "追蹤了什麼",
    "watchlist",
    "watches",
)
_WATCH_REMOVE_KEYWORDS = (
    "取消追蹤",
    "停止追蹤",
    "移除追蹤",
    "刪除追蹤",
    "unwatch",
    "stopwatch",
)
_WATCH_UPDATE_PRICE_KEYWORDS = (
    "改成",
    "改為",
    "更新",
    "調整",
    "修改",
    "setprice",
    "updatewatch",
)
_SNS_CONTEXT_KEYWORDS = (
    "推特",
    "推主",
    "推文",
    "twitter",
    "x.com",
    "sns",
    " x ",  # bare "X" as a word (with spaces)
    "@",
)
_SNS_LIST_KEYWORDS = (
    "推主追蹤",
    "x 追蹤",
    "x追蹤",
    "twitter 追蹤",
    "snslist",
    "sns 清單",
    "sns清單",
    "x 清單",
    "推特清單",
)
_SNS_BUZZ_KEYWORDS = (
    "snsbuzz",
    "熱門整理",
    "熱門討論",
    "整理一下",
    "什麼熱門",
    "最近熱門",
    "在 reddit",
    "buzz",
    "topic digest",
)
_X_HANDLE_RE = re.compile(r"@([a-zA-Z0-9_]{1,15})")
_REPUTATION_KEYWORDS = (
    "信用",
    "信譽",
    "信頼",
    "查信",
    "查賣家",
    "reputation",
    "repcheck",
    "snapshot",
    "快照",
    "proof",
)
_STATUS_KEYWORDS = (
    "status",
    "狀態",
    "状态",
    "目前狀況",
    "目前状态",
    "現在狀況",
    "現在状态",
    "模型",
    "運行",
    "运行",
    "健康",
    "health",
)
_TOOLS_KEYWORDS = (
    "tools",
    "tool",
    "工具",
    "功能清單",
    "功能列表",
    "工具清單",
    "工具列表",
    "所有工具",
    "可用工具",
    "capabilities",
    "catalog",
)
_SCAN_KEYWORDS = (
    "scan",
    "掃圖",
    "扫图",
    "圖片查價",
    "图片查价",
    "照片查價",
    "照片查价",
    "image lookup",
    "photo lookup",
    "ocr",
)
_URL_PATTERN = re.compile(r"https?://\S+")
_GENERIC_CARD_NUMBER_PATTERN = re.compile(
    r"\b(?:[A-Z0-9]+/[A-Z0-9]+(?:-[A-Z0-9]+)*-\d{1,3}|[A-Z0-9]{2,}-[A-Z]{1,4}\d{1,4})\b",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class TelegramNaturalLanguageIntent:
    intent: str
    game: str | None = None
    name: str | None = None
    card_number: str | None = None
    rarity: str | None = None
    set_code: str | None = None
    limit: int | None = None
    confidence: float | None = None
    # watch-specific fields
    watch_query: str | None = None
    watch_price_threshold: int | None = None
    watch_id: str | None = None
    # reputation snapshot field
    query_url: str | None = None
    # SNS-specific fields
    sns_handle: str | None = None       # for sns_add_account / sns_delete (@username)
    sns_keyword: str | None = None      # for sns_add_keyword (keyword watch)
    sns_buzz_query: str | None = None   # for sns_buzz (LLM topic digest)


class TelegramNaturalLanguageRouter:
    backend = "ollama"

    def __init__(
        self,
        *,
        endpoint: str,
        model: str,
        timeout_seconds: int,
        tool_spec: str | None = None,
        ssl_context: ssl.SSLContext | None = None,
    ) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.tool_spec = tool_spec.strip() if tool_spec else ""
        self._ssl_context = ssl_context if self.endpoint.startswith("https://") else None

    @property
    def descriptor(self) -> str:
        return f"{self.backend}:{self.model}"

    def route(self, text: str) -> TelegramNaturalLanguageIntent | None:
        content = text.strip()
        if not content:
            return None

        payload = {
            "model": self.model,
            "prompt": self._build_prompt(content),
            "format": _ROUTER_JSON_SCHEMA,
            "stream": False,
            "options": {"temperature": 0},
        }
        response_text = self._post_generate(payload)
        parsed = _load_json_fragment(response_text)
        if not isinstance(parsed, dict):
            raise RuntimeError(f"Natural-language router did not return a JSON object for {self.descriptor}.")
        return _normalize_intent(parsed)

    def _build_prompt(self, text: str) -> str:
        tool_spec_block = f"Tool spec:\n{self.tool_spec}\n\n" if self.tool_spec else ""
        return (
            "You route Telegram messages for a trading-card price assistant and must return only JSON.\n"
            "Allowed intents: lookup_card, trend_board, add_watch, list_watches, remove_watch, update_watch_price, reputation_snapshot, "
            "sns_add_account, sns_add_keyword, sns_list, sns_delete, sns_buzz, "
            "help, status, tools, scan_help, unknown.\n"
            + tool_spec_block +
            "Use lookup_card when the user wants the price, value, or card lookup of one specific card.\n"
            "Use trend_board when the user asks for hot, trending, liquidity, or ranking cards.\n"
            "Use add_watch when the user wants to track/monitor a MERCARI product and be notified below a price threshold.\n"
            "  Set watch_query to the product name/keywords, watch_price_threshold to the integer JPY limit.\n"
            "Use list_watches when the user wants to see the MERCARI watchlist / tracked items (no @ handle, no SNS context).\n"
            "Use remove_watch when the user wants to stop tracking a MERCARI watch by hex ID (e.g. abc12345). Set watch_id.\n"
            "Use update_watch_price when the user wants to change the price threshold of an existing Mercari watch. Set watch_id and watch_price_threshold.\n"
            "Use reputation_snapshot when the user wants to check a seller's reputation/trust/credit or take a snapshot of a URL.\n"
            "  Set query_url to the URL found in the message (Mercari item or profile URL).\n"
            "Use sns_add_account when the user wants to ADD / track / monitor an X (Twitter) account that starts with @.\n"
            "  Set sns_handle to the @username (without the @).\n"
            "Use sns_add_keyword when the user wants to add an X keyword/topic watch (not an @ account).\n"
            "  Set sns_keyword to the keyword phrase.\n"
            "Use sns_list when the user wants to see the SNS / X / Twitter watch rules (NOT the Mercari watchlist).\n"
            "Use sns_delete when the user wants to REMOVE / unfollow / unwatch / 取消追蹤 / 刪除 an X account or SNS rule, especially when the message references an @ handle.\n"
            "  Set sns_handle to the @username if mentioned (without the @). Set watch_id only if a hex SNS rule ID is given.\n"
            "Use sns_buzz when the user wants a Reddit/X topic digest, hot discussion, summary, 'what's buzzing about X'.\n"
            "  Set sns_buzz_query to the topic keyword.\n"
            "Use help when the user asks what the bot can do.\n"
            "Use status when the user asks about current runtime state, models, or service health.\n"
            "Use tools when the user explicitly asks for the full tool catalog or list of available tools.\n"
            "Use scan_help when the user asks how to scan a card from a photo or wants image-lookup instructions before sending a photo.\n"
            "Use unknown when the request is unrelated or too ambiguous.\n"
            "DISAMBIGUATION RULES:\n"
            "- Any mention of @username (e.g. @elonmusk, @aka_claw) means SNS, not Mercari. Pick sns_add_account / sns_delete / sns_list accordingly.\n"
            "- 'X', 'Twitter', '推特', '推文', '推主', '帳號' always indicate SNS intents.\n"
            "- '追蹤'/'tracking' alone is ambiguous: if it co-occurs with @handle/X/Twitter → SNS; if it co-occurs with a price (円/JPY/万) or Mercari URL → Mercari.\n"
            "- '取消'/'刪除'/'unfollow'/'unwatch' on an @ handle → sns_delete with sns_handle set, NOT remove_watch.\n"
            f'Game must be one of "{supported_game_hint()}" or null.\n'
            "Infer pokemon for wording like Pokemon, PTCG, 寶可夢, 寶可卡.\n"
            "Infer ws for wording like Weiss, WS, Weiß Schwarz, ヴァイス.\n"
            "Infer yugioh for wording like Yu-Gi-Oh, YGO, 遊戯王, 遊戲王.\n"
            "Infer union_arena for wording like Union Arena, Union Area, UA, ユニオンアリーナ.\n"
            "Extract only high-confidence structured fields.\n"
            "Do not invent card numbers, rarity, or set codes.\n"
            "For trend_board, limit should be 1-10 when specified, otherwise 5.\n"
            "For add_watch, watch_price_threshold is an integer JPY amount (e.g. 50000 for 5万).\n"
            "For fields not applicable to the intent, return null.\n"
            "Examples:\n"
            '- "幫我查寶可夢 リザードンex 201/165 SAR" -> lookup_card\n'
            '- "查遊戯王 青眼の白龍 QCCP-JP001 UR" -> lookup_card\n'
            '- "查 Union Arena UAPR/EVA-1-71 綾波レイ" -> lookup_card\n'
            '- "pokemon 熱門前5" -> trend_board\n'
            '- "追蹤 初音ミク SSP 5万以下" -> add_watch, watch_query="初音ミク SSP", watch_price_threshold=50000\n'
            '- "看我的追蹤清單" -> list_watches\n'
            '- "取消追蹤 abc12345" -> remove_watch, watch_id="abc12345"\n'
            '- "把 abc12345 改成 4萬" -> update_watch_price, watch_id="abc12345", watch_price_threshold=40000\n'
            '- "查詢信用 https://jp.mercari.com/item/m12345" -> reputation_snapshot, query_url="https://jp.mercari.com/item/m12345"\n'
            '- "追蹤 @elonmusk" -> sns_add_account, sns_handle="elonmusk"\n'
            '- "新增 X 監控 @aka_claw" -> sns_add_account, sns_handle="aka_claw"\n'
            '- "刪除追蹤 @elonmusk" -> sns_delete, sns_handle="elonmusk"\n'
            '- "取消追蹤 @aka_claw" -> sns_delete, sns_handle="aka_claw"\n'
            '- "unfollow @elonmusk" -> sns_delete, sns_handle="elonmusk"\n'
            '- "我的 X 追蹤清單" -> sns_list\n'
            '- "看一下推主追蹤" -> sns_list\n'
            '- "監控關鍵字 機動戰士" -> sns_add_keyword, sns_keyword="機動戰士"\n'
            '- "整理一下 amd 最近的熱門討論" -> sns_buzz, sns_buzz_query="amd"\n'
            '- "Trump 在 X 上最近怎樣" -> sns_buzz, sns_buzz_query="Trump"\n'
            '- "你會什麼" -> help\n'
            '- "你現在狀態如何" -> status\n'
            '- "列出所有工具" -> tools\n'
            '- "我要怎麼用照片查價" -> scan_help\n'
            '- "明天天氣如何" -> unknown\n'
            f"User message:\n{text}\n"
        )

    def _post_generate(self, payload: dict[str, object]) -> str:
        request = Request(
            _resolve_generate_url(self.endpoint),
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds, context=self._ssl_context) as response:
                body = response.read().decode("utf-8", errors="replace")
        except HTTPError as exc:
            raise RuntimeError(f"Natural-language router HTTP {exc.code}.") from exc
        except URLError as exc:
            raise RuntimeError(f"Natural-language router request failed: {exc.reason}") from exc

        payload_body = json.loads(body)
        response_text = payload_body.get("response", "")
        if isinstance(response_text, dict):
            return json.dumps(response_text, ensure_ascii=False)
        if not isinstance(response_text, str):
            raise RuntimeError(f"Natural-language router response type was {type(response_text).__name__}.")
        return response_text.strip()


def build_telegram_natural_language_router(
    *,
    endpoint: str,
    model: str | None = None,
    backend: str = "ollama",
    timeout_seconds: int = 180,
    tool_spec: str | None = None,
    ssl_context: ssl.SSLContext | None = None,
) -> TelegramNaturalLanguageRouter | None:
    if not model:
        return None
    resolved_backend = backend.strip().lower()
    if resolved_backend != "ollama":
        logger.warning("Unsupported Telegram natural-language router backend=%s", resolved_backend)
        return None
    return TelegramNaturalLanguageRouter(
        endpoint=endpoint,
        model=model,
        timeout_seconds=max(1, timeout_seconds),
        tool_spec=tool_spec,
        ssl_context=ssl_context,
    )


_KANJI_MAN: dict[str, int] = {
    "一": 1, "二": 2, "三": 3, "四": 4, "五": 5,
    "六": 6, "七": 7, "八": 8, "九": 9, "十": 10,
    "百": 100, "千": 1000,
}


def _parse_price_threshold(text: str) -> int | None:
    """Extract a JPY price threshold from natural language.

    Handles patterns like: 50000, 5万, 50,000, ¥300000, 三十万, 五万, 30万, etc.
    Returns the integer JPY value, or None if not found.
    """
    # Strip currency symbols and commas
    cleaned = text.replace("¥", "").replace("，", "").replace(",", "").replace("円", "").replace("日幣", "").replace("日元", "").replace("日圓", "").replace("以下", "").replace("以內", "").replace("以内", "").replace("低於", "").replace("不超過", "")

    # Pattern: digits optionally followed by 万/萬 (man = 10,000)
    man_match = re.search(r"(\d+(?:\.\d+)?)\s*[万萬]", cleaned)
    if man_match:
        return int(float(man_match.group(1)) * 10000)

    # Pattern: kanji number + 万/萬, e.g. 三十万, 五万
    kanji_man_match = re.search(r"([一二三四五六七八九十百千]+)\s*[万萬]", cleaned)
    if kanji_man_match:
        kanji_val = _parse_kanji_number(kanji_man_match.group(1))
        if kanji_val is not None:
            return kanji_val * 10000

    # Pattern: plain digits (must be >= 100 to avoid false positives)
    digit_match = re.search(r"\b(\d{3,7})\b", cleaned)
    if digit_match:
        return int(digit_match.group(1))

    return None


def _parse_kanji_number(kanji: str) -> int | None:
    """Parse a simple kanji numeral like 三十, 五, 百二十 into an integer."""
    if not kanji:
        return None
    result = 0
    current = 0
    for ch in kanji:
        val = _KANJI_MAN.get(ch)
        if val is None:
            return None
        if val >= 10:
            if current == 0:
                current = 1
            result += current * val
            current = 0
        else:
            current = val
    return result + current if result + current > 0 else None


def _extract_watch_query(text: str) -> str | None:
    """Strip watch-command keywords and price part, return the remaining product query."""
    # Remove trigger keywords
    stripped = text
    for kw in (*_WATCH_ADD_KEYWORDS, "幫我", "我想", "請", "幫", "要"):
        stripped = re.sub(re.escape(kw), " ", stripped, flags=re.IGNORECASE)
    # Remove price portion: digits + 万/萬 or plain digits near threshold words
    stripped = re.sub(r"\d+(?:\.\d+)?\s*[万萬]?", " ", stripped)
    stripped = re.sub(r"[一二三四五六七八九十百千]+\s*[万萬]", " ", stripped)
    # Remove stray punctuation
    stripped = re.sub(r"[，、。！？!?¥￥]", " ", stripped)
    query = " ".join(stripped.split()).strip()
    return query if len(query) >= 2 else None


def fallback_route_telegram_natural_language(text: str) -> TelegramNaturalLanguageIntent | None:
    content = text.strip()
    if not content:
        return None
    lowered = content.lower()

    # ── Reputation snapshot: URL present + reputation-related keyword ──────────
    url_match = _URL_PATTERN.search(content)
    if url_match and any(kw in lowered for kw in _REPUTATION_KEYWORDS):
        return TelegramNaturalLanguageIntent(
            intent="reputation_snapshot",
            query_url=url_match.group(0),
            confidence=0.85,
        )

    # ── SNS intents (must run before Mercari watch intents so @handle wins) ───

    handle_match = _X_HANDLE_RE.search(content)
    has_sns_context = any(kw in lowered for kw in _SNS_CONTEXT_KEYWORDS) or handle_match is not None

    # sns_delete: any remove/unfollow phrase combined with @handle or SNS context
    if handle_match and any(kw in lowered for kw in _WATCH_REMOVE_KEYWORDS):
        return TelegramNaturalLanguageIntent(
            intent="sns_delete",
            sns_handle=handle_match.group(1),
            confidence=0.85,
        )
    if "unfollow" in lowered and handle_match:
        return TelegramNaturalLanguageIntent(
            intent="sns_delete",
            sns_handle=handle_match.group(1),
            confidence=0.85,
        )

    # sns_list: explicit SNS list phrasing
    if any(kw in lowered for kw in _SNS_LIST_KEYWORDS):
        return TelegramNaturalLanguageIntent(intent="sns_list", confidence=0.75)

    # sns_buzz: digest-of-topic phrasing. Keep generic "最近什麼熱門排行"
    # available for the TCG trend router, where it should return None until a
    # game is provided.
    buzz_keywords = [kw for kw in _SNS_BUZZ_KEYWORDS if kw in lowered]
    generic_buzz_keywords = {"什麼熱門", "最近熱門"}
    generic_trend_without_game = (
        buzz_keywords
        and all(kw in generic_buzz_keywords for kw in buzz_keywords)
        and not has_sns_context
        and _infer_game(content) is None
        and any(kw in lowered for kw in _TREND_KEYWORDS)
    )
    if buzz_keywords and not generic_trend_without_game:
        return TelegramNaturalLanguageIntent(
            intent="sns_buzz",
            sns_buzz_query=_extract_buzz_query(content),
            confidence=0.6,
        )

    # sns_add_account: track/monitor with @handle
    if handle_match and any(kw in lowered for kw in _WATCH_ADD_KEYWORDS):
        return TelegramNaturalLanguageIntent(
            intent="sns_add_account",
            sns_handle=handle_match.group(1),
            confidence=0.8,
        )

    # ── Watch intents (check before generic lookup so keywords don't conflict) ──

    # remove_watch: "取消追蹤 abc123" / "unwatch abc123"
    if any(kw in lowered for kw in _WATCH_REMOVE_KEYWORDS):
        id_match = re.search(r"\b([0-9a-f]{8,16})\b", lowered)
        watch_id = id_match.group(1) if id_match else None
        return TelegramNaturalLanguageIntent(
            intent="remove_watch",
            watch_id=watch_id,
            confidence=0.7,
        )

    # update_watch_price: "把 abc12345 改成 4萬" / "setprice abc12345 40000"
    # Requires both a watch_id (hex) and an update keyword in the same message.
    _id_match = re.search(r"\b([0-9a-f]{8,16})\b", lowered)
    if _id_match and any(kw in lowered for kw in _WATCH_UPDATE_PRICE_KEYWORDS):
        threshold = _parse_price_threshold(content)
        if threshold:
            return TelegramNaturalLanguageIntent(
                intent="update_watch_price",
                watch_id=_id_match.group(1),
                watch_price_threshold=threshold,
                confidence=0.7,
            )

    # list_watches: "追蹤清單" / "我的追蹤" / "watchlist"
    if any(kw in lowered for kw in _WATCH_LIST_KEYWORDS):
        return TelegramNaturalLanguageIntent(intent="list_watches", confidence=0.75)

    # add_watch: "追蹤 初音ミク SSP 5萬以下" / "提醒我 xxx 低於 50000"
    if any(kw in lowered for kw in _WATCH_ADD_KEYWORDS):
        threshold = _parse_price_threshold(content)
        query = _extract_watch_query(content)
        if query or threshold:
            return TelegramNaturalLanguageIntent(
                intent="add_watch",
                watch_query=query,
                watch_price_threshold=threshold,
                confidence=0.55 if (query and threshold) else 0.35,
            )

    # ── Original intents ──────────────────────────────────────────────────────

    # scan_help is checked before help so "照片查價怎麼用" / "OCR 怎麼用" routes
    # to scan_help rather than being swallowed by the generic "怎麼用" help keyword.
    if any(keyword in lowered for keyword in _SCAN_KEYWORDS):
        return TelegramNaturalLanguageIntent(intent="scan_help", confidence=0.45)

    if any(keyword in lowered for keyword in ("help", "指令", "怎麼用", "會什麼")):
        return TelegramNaturalLanguageIntent(intent="help", confidence=0.35)

    if any(keyword in lowered for keyword in _TOOLS_KEYWORDS):
        return TelegramNaturalLanguageIntent(intent="tools", confidence=0.45)

    if any(keyword in lowered for keyword in _STATUS_KEYWORDS):
        return TelegramNaturalLanguageIntent(intent="status", confidence=0.45)

    if any(keyword in lowered for keyword in _TREND_KEYWORDS):
        game = _infer_game(content)
        if game is None:
            return None
        limit_match = re.search(r"(?:top|前)\s*(?P<limit>\d{1,2})", lowered)
        if limit_match:
            limit = int(limit_match.group("limit"))
        else:
            chinese_match = re.search(r"前\s*(?P<digit>[一二三四五六七八九十])", content)
            limit = _CHINESE_DIGIT_MAP.get(chinese_match.group("digit"), 5) if chinese_match else 5
        return TelegramNaturalLanguageIntent(
            intent="trend_board",
            game=game,
            limit=max(1, min(10, limit)),
            confidence=0.45,
        )

    if any(keyword in content for keyword in _LOOKUP_KEYWORDS) or _infer_game(content) is not None:
        game = _infer_game(content)
        if game is None:
            return None
        card_number_match = _extract_card_number_match(content)
        rarity_match = re.search(r"\b(SSP|SEC\+|SEC|SAR|CSR|CHR|UR|SR|AR|RRR|RR|PR\+|PR|SP|OFR|SS|R|U|C|MA|MUR)\b", content.upper())
        set_code_match = re.search(r"\b(SV\d{1,2}[A-Z]?|M\d{1,2}[A-Z]?|SM\d{1,2}[A-Z]?|S\d{1,2}[A-Z]?|SV-P|SM-P|S-P|M-P|BW-P|XY-P)\b", content.upper())
        stripped_name = content
        if card_number_match:
            stripped_name = stripped_name.replace(card_number_match.group(0), " ")
        for token in (
            "幫我查",
            "查一下",
            "查",
            "估價",
            "價格",
            "price",
            "pokemon",
            "ptcg",
            "ws",
            "weiss",
            "schwarz",
            "yugioh",
            "yu-gi-oh",
            "ygo",
            "union_arena",
            "union arena",
            "union area",
            "unionarea",
            "ua",
            "寶可夢",
            "寶可卡",
            "遊戯王",
            "遊戲王",
            "游戏王",
            "遊☆戯☆王",
            "ユニオンアリーナ",
        ):
            stripped_name = re.sub(re.escape(token), " ", stripped_name, flags=re.IGNORECASE)
        if rarity_match:
            stripped_name = stripped_name.replace(rarity_match.group(1), " ")
        if set_code_match:
            stripped_name = stripped_name.replace(set_code_match.group(1), " ")
        name = " ".join(stripped_name.split()).strip() or None
        return TelegramNaturalLanguageIntent(
            intent="lookup_card",
            game=game,
            name=name,
            card_number=None if card_number_match is None else card_number_match.group(0),
            rarity=None if rarity_match is None else rarity_match.group(1).upper(),
            set_code=_derive_set_code_from_card_number(None if card_number_match is None else card_number_match.group(0))
            if set_code_match is None
            else set_code_match.group(1).lower(),
            confidence=0.3,
        )
    return None


def _resolve_generate_url(endpoint: str) -> str:
    parsed = urlparse(endpoint)
    path = parsed.path.rstrip("/")
    if path.endswith("/api/generate"):
        return endpoint
    if path.endswith("/api"):
        return f"{endpoint.rstrip('/')}/generate"
    return f"{endpoint.rstrip('/')}/api/generate"


def _load_json_fragment(value: str) -> object:
    stripped = value.strip()
    if not stripped:
        return None
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", stripped, re.DOTALL)
        if match is None:
            return None
        return json.loads(match.group(0))


_ALLOWED_INTENTS = frozenset({
    "lookup_card", "trend_board",
    "add_watch", "list_watches", "remove_watch", "update_watch_price",
    "reputation_snapshot",
    "sns_add_account", "sns_add_keyword", "sns_list", "sns_delete", "sns_buzz",
    "help", "status", "tools", "scan_help", "unknown",
})


def _normalize_intent(payload: dict[str, object]) -> TelegramNaturalLanguageIntent:
    intent = str(payload.get("intent", "unknown")).strip().lower()
    if intent not in _ALLOWED_INTENTS:
        intent = "unknown"

    game = _normalize_game(payload.get("game"))
    name = _normalize_text_field(payload.get("name"))
    card_number = _normalize_text_field(payload.get("card_number"))
    rarity = _normalize_token(payload.get("rarity"), uppercase=True)
    set_code = _normalize_token(payload.get("set_code"), uppercase=False)
    limit = _normalize_limit(payload.get("limit"))
    confidence = _normalize_confidence(payload.get("confidence"))
    watch_query = _normalize_text_field(payload.get("watch_query"))
    watch_price_threshold = _normalize_price_threshold(payload.get("watch_price_threshold"))
    watch_id = _normalize_text_field(payload.get("watch_id"))
    query_url = _normalize_text_field(payload.get("query_url"))
    sns_handle = _normalize_handle(payload.get("sns_handle"))
    sns_keyword = _normalize_text_field(payload.get("sns_keyword"))
    sns_buzz_query = _normalize_text_field(payload.get("sns_buzz_query"))

    if intent == "trend_board" and limit is None:
        limit = 5
    return TelegramNaturalLanguageIntent(
        intent=intent,
        game=game,
        name=name,
        card_number=card_number,
        rarity=rarity,
        set_code=set_code,
        limit=limit,
        confidence=confidence,
        watch_query=watch_query,
        watch_price_threshold=watch_price_threshold,
        watch_id=watch_id,
        query_url=query_url,
        sns_handle=sns_handle,
        sns_keyword=sns_keyword,
        sns_buzz_query=sns_buzz_query,
    )


def _normalize_handle(value: object) -> str | None:
    """Strip leading @ and whitespace from an X handle."""
    text = _normalize_text_field(value)
    if not text:
        return None
    return text.lstrip("@").strip() or None


_BUZZ_STOP_PHRASES = (
    "幫我整理", "整理一下", "整理", "最近熱門討論", "熱門討論", "熱門整理",
    "最近怎樣", "怎麼樣", "在 reddit 上", "在 reddit", "buzz", "topic digest",
    "snsbuzz", "什麼熱門", "最近熱門", "看一下",
)


def _extract_buzz_query(text: str) -> str | None:
    """Strip framing phrases from a buzz request to expose the actual keyword."""
    cleaned = text
    for phrase in _BUZZ_STOP_PHRASES:
        cleaned = re.sub(re.escape(phrase), " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"[，、。！？!?]", " ", cleaned)
    cleaned = " ".join(cleaned.split()).strip()
    return cleaned or None


def _normalize_game(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    return normalize_game_key(value)


def _normalize_text_field(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = " ".join(value.split()).strip()
    return cleaned or None


def _normalize_token(value: object, *, uppercase: bool) -> str | None:
    text = _normalize_text_field(value)
    if text is None:
        return None
    return text.upper() if uppercase else text.lower()


def _normalize_limit(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return max(1, min(10, int(value)))
    if isinstance(value, str) and value.strip().isdigit():
        return max(1, min(10, int(value.strip())))
    return None


def _normalize_confidence(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def _normalize_price_threshold(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        v = int(value)
        return v if v > 0 else None
    if isinstance(value, str):
        try:
            v = int(value.strip().replace(",", "").replace("，", ""))
            return v if v > 0 else None
        except ValueError:
            return None
    return None


def _infer_game(text: str) -> str | None:
    lowered = text.lower()
    if any(token in lowered for token in ("pokemon", "ptcg", "寶可夢", "寶可卡")):
        return "pokemon"
    # \bws\b fails when "ws" is surrounded by CJK characters (which are also \w in Python).
    # Use ASCII-only lookaround so "熱門ws前三" is matched correctly.
    if any(token in lowered for token in ("weiss", "schwarz", "ヴァイス")) or re.search(r"(?<![a-z])ws(?![a-z])", lowered):
        return "ws"
    if (
        any(token in lowered for token in ("yugioh", "yu-gi-oh", "遊戯王", "遊戲王", "游戏王", "遊☆戯☆王"))
        or re.search(r"(?<![a-z])ygo(?![a-z])", lowered)
    ):
        return "yugioh"
    if (
        any(token in lowered for token in ("union arena", "union area", "union_arena", "unionarea", "ユニオンアリーナ"))
        or re.search(r"(?<![a-z])ua(?![a-z])", lowered)
    ):
        return "union_arena"
    return None


def _extract_card_number_match(text: str) -> re.Match[str] | None:
    return re.search(r"\b\d{1,3}/\d{1,3}\b", text) or _GENERIC_CARD_NUMBER_PATTERN.search(text.upper())


def _derive_set_code_from_card_number(card_number: str | None) -> str | None:
    if not card_number:
        return None
    if "/" in card_number:
        return card_number.split("/", 1)[0].lower()
    if "-" in card_number:
        return card_number.split("-", 1)[0].lower()
    return None
