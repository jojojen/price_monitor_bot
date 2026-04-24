from __future__ import annotations

import json
import logging
import re
import ssl
from dataclasses import dataclass
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

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
    },
    "required": ["intent", "game", "name", "card_number", "rarity", "set_code", "limit", "confidence"],
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


class TelegramNaturalLanguageRouter:
    backend = "ollama"

    def __init__(
        self,
        *,
        endpoint: str,
        model: str,
        timeout_seconds: int,
        ssl_context: ssl.SSLContext | None = None,
    ) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds
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
        return (
            "You route Telegram messages for a trading-card assistant and must return only JSON.\n"
            "Allowed intents: lookup_card, trend_board, help, unknown.\n"
            "Use lookup_card when the user wants the price, value, or card lookup of one specific card.\n"
            "Use trend_board when the user asks for hot, trending, liquidity, or ranking cards.\n"
            "Use help when the user asks what the bot can do.\n"
            "Use unknown when the request is unrelated or too ambiguous.\n"
            'Game must be "pokemon", "ws", or null.\n'
            "Infer pokemon for wording like Pokemon, PTCG, 寶可夢, 寶可卡.\n"
            "Infer ws for wording like Weiss, WS, Weiß Schwarz, ヴァイス.\n"
            "Extract only high-confidence structured fields.\n"
            "Do not invent card numbers, rarity, or set codes.\n"
            "For trend_board, name/card_number/rarity/set_code should be null and limit should be 1-10 when specified, otherwise 5.\n"
            "For lookup_card, keep the card name concise and leave missing metadata null.\n"
            "Examples:\n"
            '- "幫我查寶可夢 リザードンex 201/165 SAR" -> lookup_card\n'
            '- "pokemon 熱門前5" -> trend_board\n'
            '- "你會什麼" -> help\n'
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
        ssl_context=ssl_context,
    )


def fallback_route_telegram_natural_language(text: str) -> TelegramNaturalLanguageIntent | None:
    content = text.strip()
    if not content:
        return None
    lowered = content.lower()

    if any(keyword in lowered for keyword in ("help", "指令", "怎麼用", "會什麼")):
        return TelegramNaturalLanguageIntent(intent="help", confidence=0.35)

    if any(keyword in lowered for keyword in _TREND_KEYWORDS):
        game = _infer_game(content)
        if game is None:
            return None
        limit_match = re.search(r"(?:top|前)\s*(?P<limit>\d{1,2})", lowered)
        limit = int(limit_match.group("limit")) if limit_match else 5
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
        card_number_match = re.search(r"\b\d{1,3}/\d{1,3}\b", content)
        rarity_match = re.search(r"\b(SSP|SEC\+|SEC|SAR|CSR|CHR|UR|SR|AR|RRR|RR|PR\+|PR|SP|OFR|SS|R|U|C|MA|MUR)\b", content.upper())
        set_code_match = re.search(r"\b(SV\d{1,2}[A-Z]?|M\d{1,2}[A-Z]?|SM\d{1,2}[A-Z]?|S\d{1,2}[A-Z]?|SV-P|SM-P|S-P|M-P|BW-P|XY-P)\b", content.upper())
        stripped_name = content
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
            "寶可夢",
            "寶可卡",
        ):
            stripped_name = re.sub(re.escape(token), " ", stripped_name, flags=re.IGNORECASE)
        if card_number_match:
            stripped_name = stripped_name.replace(card_number_match.group(0), " ")
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
            set_code=None if set_code_match is None else set_code_match.group(1).lower(),
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


def _normalize_intent(payload: dict[str, object]) -> TelegramNaturalLanguageIntent:
    intent = str(payload.get("intent", "unknown")).strip().lower()
    if intent not in {"lookup_card", "trend_board", "help", "unknown"}:
        intent = "unknown"

    game = _normalize_game(payload.get("game"))
    name = _normalize_text_field(payload.get("name"))
    card_number = _normalize_text_field(payload.get("card_number"))
    rarity = _normalize_token(payload.get("rarity"), uppercase=True)
    set_code = _normalize_token(payload.get("set_code"), uppercase=False)
    limit = _normalize_limit(payload.get("limit"))
    confidence = _normalize_confidence(payload.get("confidence"))

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
    )


def _normalize_game(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    lowered = value.strip().lower()
    if lowered in {"pokemon", "ws"}:
        return lowered
    return None


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


def _infer_game(text: str) -> str | None:
    lowered = text.lower()
    if any(token in lowered for token in ("pokemon", "ptcg", "寶可夢", "寶可卡")):
        return "pokemon"
    if any(token in lowered for token in ("weiss", "schwarz", "ヴァイス")) or re.search(r"\bws\b", lowered):
        return "ws"
    return None
