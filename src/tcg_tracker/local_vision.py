from __future__ import annotations

import base64
import json
import logging
import socket
import ssl
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from market_monitor.normalize import normalize_card_number

logger = logging.getLogger(__name__)
_LOCAL_VISION_TIMEOUT_COOLDOWN_SECONDS = 15 * 60

CARD_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "item_kind": {"type": ["string", "null"]},
        "game": {"type": ["string", "null"]},
        "title": {"type": ["string", "null"]},
        "aliases": {
            "type": ["array", "null"],
            "items": {"type": "string"},
        },
        "card_number": {"type": ["string", "null"]},
        "rarity": {"type": ["string", "null"]},
        "set_code": {"type": ["string", "null"]},
        "confidence": {"type": ["number", "null"]},
    },
    "required": ["item_kind", "game", "title", "aliases", "card_number", "rarity", "set_code", "confidence"],
    "additionalProperties": False,
}


@dataclass(frozen=True, slots=True)
class LocalVisionCardCandidate:
    backend: str
    model: str
    game: str | None
    title: str | None
    aliases: tuple[str, ...]
    card_number: str | None
    rarity: str | None
    set_code: str | None
    item_kind: str | None = None
    confidence: float | None = None
    raw_response: str = ""
    warnings: tuple[str, ...] = ()

    @property
    def descriptor(self) -> str:
        return f"{self.backend}:{self.model}"


class LocalVisionTimeoutError(RuntimeError):
    def __init__(self, descriptor: str, *, timeout_seconds: int, detail: str) -> None:
        super().__init__(f"{descriptor} timed out after {timeout_seconds}s: {detail}")
        self.descriptor = descriptor
        self.timeout_seconds = timeout_seconds
        self.detail = detail


class OllamaLocalVisionClient:
    backend = "ollama"
    _timeout_backoff_until: dict[str, float] = {}

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

    @property
    def cooldown_key(self) -> str:
        return f"{self.endpoint}|{self.descriptor}"

    def cooldown_remaining_seconds(self) -> int:
        until = self._timeout_backoff_until.get(self.cooldown_key)
        if until is None:
            return 0
        remaining = int(round(until - time.monotonic()))
        return max(0, remaining)

    def is_temporarily_disabled(self) -> bool:
        return self.cooldown_remaining_seconds() > 0

    def mark_timeout_cooldown(self) -> None:
        self._timeout_backoff_until[self.cooldown_key] = time.monotonic() + _LOCAL_VISION_TIMEOUT_COOLDOWN_SECONDS

    def analyze_card_image(
        self,
        image_path: Path,
        *,
        game_hint: str | None = None,
        title_hint: str | None = None,
    ) -> LocalVisionCardCandidate | None:
        return self._analyze_with_prompt(
            image_path,
            prompt=self._build_prompt(game_hint=game_hint, title_hint=title_hint),
            game_hint=game_hint,
        )

    def analyze_card_image_text_focus(
        self,
        image_path: Path,
        *,
        game_hint: str | None = None,
        title_hint: str | None = None,
    ) -> LocalVisionCardCandidate | None:
        return self._analyze_with_prompt(
            image_path,
            prompt=self._build_text_focus_prompt(game_hint=game_hint, title_hint=title_hint),
            game_hint=game_hint,
        )

    def analyze_sealed_box_title_focus(
        self,
        image_path: Path,
        *,
        game_hint: str | None = None,
        title_hint: str | None = None,
    ) -> LocalVisionCardCandidate | None:
        return self._analyze_with_prompt(
            image_path,
            prompt=self._build_sealed_box_title_prompt(game_hint=game_hint, title_hint=title_hint),
            game_hint=game_hint,
        )

    def _analyze_with_prompt(
        self,
        image_path: Path,
        *,
        prompt: str,
        game_hint: str | None,
    ) -> LocalVisionCardCandidate | None:
        image_payload = base64.b64encode(image_path.read_bytes()).decode("ascii")
        payload = {
            "model": self.model,
            "prompt": prompt,
            "images": [image_payload],
            "format": CARD_JSON_SCHEMA,
            "stream": False,
            "options": {
                "temperature": 0,
            },
        }
        response_text = self._post_generate(payload)
        candidate_payload = _load_json_fragment(response_text)
        if not isinstance(candidate_payload, dict):
            raise RuntimeError(f"Ollama did not return a JSON object for {self.descriptor}.")

        return LocalVisionCardCandidate(
            backend=self.backend,
            model=self.model,
            item_kind=_normalize_item_kind(candidate_payload.get("item_kind")),
            game=_normalize_game(candidate_payload.get("game"), fallback=game_hint),
            title=_normalize_text_field(candidate_payload.get("title")),
            aliases=_normalize_aliases(candidate_payload.get("aliases")),
            card_number=_normalize_card_number_field(candidate_payload.get("card_number")),
            rarity=_normalize_token(candidate_payload.get("rarity"), uppercase=True),
            set_code=_normalize_token(candidate_payload.get("set_code"), uppercase=False),
            confidence=_normalize_confidence(candidate_payload.get("confidence")),
            raw_response=response_text,
        )

    def _build_prompt(self, *, game_hint: str | None, title_hint: str | None) -> str:
        hints: list[str] = []
        if game_hint in {"pokemon", "ws"}:
            hints.append(f"game_hint={game_hint}")
        if title_hint:
            hints.append(f"title_hint={title_hint}")
        hint_text = "\n".join(hints) if hints else "no external hints"
        return (
            "Identify the main trading card product in this image and return only JSON.\n"
            "The image may show either exactly one identifiable trading card or one sealed trading card box.\n"
            "If the image does not show one clear card or one clear sealed box, return null for every field instead of guessing.\n"
            "Use item_kind=card for a single card and item_kind=sealed_box for an unopened or box-style sealed product.\n"
            "For sealed boxes, return the full product title and keep card_number and rarity null.\n"
            "Do not merge multiple products from the same photo into one answer.\n"
            "Do not merge multiple cards from the same photo into one answer.\n"
            "Prefer the printed product name and pack branding over slab labels whenever they disagree.\n"
            "Ignore slab grades, cert numbers, copyright lines, attack text, and rule text.\n"
            "Never use slab grade text as the card rarity.\n"
            "For Japanese sealed Pokemon products, include the product line when visible, such as 強化拡張パック ポケモンカード151 or ハイクラスパック MEGAドリームex.\n"
            "Pokemon collector numbers should stay in full market-friendly form when visible, such as 201/165, 020/M-P, 085/SV-P, or 764/742.\n"
            "If a Pokemon promo card shows a set code and number separately, combine them into one card_number field like 085/SV-P.\n"
            "For Japanese Start Deck 100 Battle Collection slab labels such as MC JP #764, return 764/742 when the image supports that collector number.\n"
            'Use item_kind values "card", "sealed_box", or null.\n'
            'Use game values "pokemon", "ws", or null.\n'
            "Preserve the Japanese title when visible.\n"
            "Use null for unknown values instead of guessing.\n"
            "aliases should contain only high-confidence alternate names.\n"
            "Hints:\n"
            f"{hint_text}\n"
        )

    def _build_text_focus_prompt(self, *, game_hint: str | None, title_hint: str | None) -> str:
        hints: list[str] = []
        if game_hint in {"pokemon", "ws"}:
            hints.append(f"game_hint={game_hint}")
        if title_hint:
            hints.append(f"title_hint={title_hint}")
        hint_text = "\n".join(hints) if hints else "no external hints"
        return (
            "Read only the printed card identity text from this trading card image and return only JSON.\n"
            "Prioritize the printed card name near the top and the collector number, rarity, and set code near the bottom.\n"
            "Do not identify the artwork subject, attack names, slab labels, cert numbers, or promo guesses unless the printed card text supports them.\n"
            "If you are unsure about the title, keep title null but still return card_number, rarity, and set_code when visible.\n"
            "For Pokemon cards, prefer collector numbers like 201/165, 110/080, 085/SV-P, or 020/M-P.\n"
            'Use game values "pokemon", "ws", or null.\n'
            "Hints:\n"
            f"{hint_text}\n"
        )

    def _build_sealed_box_title_prompt(self, *, game_hint: str | None, title_hint: str | None) -> str:
        hints: list[str] = []
        if game_hint in {"pokemon", "ws"}:
            hints.append(f"game_hint={game_hint}")
        if title_hint:
            hints.append(f"title_hint={title_hint}")
        hint_text = "\n".join(hints) if hints else "no external hints"
        return (
            "Read only the sealed trading card box product title from this image and return only JSON.\n"
            "Assume the image shows one unopened trading card box or display box, not a single card.\n"
            "Focus on the big printed product name on the front of the box.\n"
            "Ignore chat captions, timestamps, UI chrome, seller overlays, copyright text, pack counts, and stock labels unless they are part of the official product title.\n"
            "For sealed boxes, set item_kind=sealed_box, keep card_number and rarity null, and keep title null instead of guessing.\n"
            "For Japanese Pokemon boxes, preserve the Japanese product line when visible, such as 強化拡張パック ポケモンカード151 or ハイクラスパック MEGAドリームex.\n"
            "If only a strong numeric title like 151 is clearly visible, you may return that as title instead of inventing missing words.\n"
            'Use item_kind values "sealed_box" or null.\n'
            'Use game values "pokemon", "ws", or null.\n'
            "aliases should contain only high-confidence alternate names.\n"
            "Hints:\n"
            f"{hint_text}\n"
        )

    def _post_generate(self, payload: dict[str, object]) -> str:
        target = _resolve_generate_url(self.endpoint)
        body = None
        for attempt in range(2):
            started_at = time.monotonic()
            request = Request(
                target,
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                },
                method="POST",
            )
            logger.info(
                "Local vision request starting target=%s model=%s attempt=%s timeout_seconds=%s",
                target,
                self.model,
                attempt + 1,
                self.timeout_seconds,
            )
            try:
                with urlopen(request, timeout=self.timeout_seconds, context=self._ssl_context) as response:
                    body = response.read().decode("utf-8", errors="replace")
                logger.info(
                    "Local vision request completed target=%s model=%s attempt=%s elapsed_seconds=%.2f bytes=%s",
                    target,
                    self.model,
                    attempt + 1,
                    time.monotonic() - started_at,
                    len(body.encode("utf-8", errors="replace")),
                )
                break
            except HTTPError as exc:
                if exc.code >= 500 and attempt == 0:
                    time.sleep(1)
                    continue
                raise RuntimeError(f"Ollama request failed with status {exc.code}.") from exc
            except URLError as exc:
                if isinstance(exc.reason, TimeoutError | socket.timeout):  # type: ignore[arg-type]
                    raise LocalVisionTimeoutError(
                        self.descriptor,
                        timeout_seconds=self.timeout_seconds,
                        detail=str(exc.reason) or "request timed out",
                    ) from exc
                if attempt == 0:
                    time.sleep(1)
                    continue
                raise RuntimeError(f"Ollama request failed: {exc.reason}.") from exc
            except (TimeoutError, socket.timeout) as exc:
                raise LocalVisionTimeoutError(
                    self.descriptor,
                    timeout_seconds=self.timeout_seconds,
                    detail=str(exc) or "request timed out",
                ) from exc

        if body is None:
            raise RuntimeError(f"Ollama request failed without a response body for {self.descriptor}.")

        payload = json.loads(body)
        response_text = payload.get("response", "")
        if isinstance(response_text, dict):
            return json.dumps(response_text, ensure_ascii=False)
        if not isinstance(response_text, str):
            raise RuntimeError(f"Ollama response field had unexpected type: {type(response_text).__name__}.")
        return response_text.strip()


def build_local_vision_client(
    *,
    endpoint: str,
    model_list: str | None,
    backend: str = "ollama",
    timeout_seconds: int = 180,
    ssl_context: ssl.SSLContext | None = None,
) -> OllamaLocalVisionClient | None:
    clients = build_local_vision_clients(
        endpoint=endpoint,
        model_list=model_list,
        backend=backend,
        timeout_seconds=timeout_seconds,
        ssl_context=ssl_context,
    )
    if not clients:
        return None
    return clients[0]


def build_local_vision_clients(
    *,
    endpoint: str,
    model_list: str | None,
    backend: str = "ollama",
    timeout_seconds: int = 180,
    ssl_context: ssl.SSLContext | None = None,
) -> tuple[OllamaLocalVisionClient, ...]:
    models = _parse_model_list(model_list)
    if not models:
        return ()

    resolved_backend = backend.strip().lower()
    if resolved_backend != "ollama":
        logger.warning("Unsupported local vision backend=%s", resolved_backend)
        return ()

    return tuple(
        OllamaLocalVisionClient(
            endpoint=endpoint,
            model=model,
            timeout_seconds=max(1, timeout_seconds),
            ssl_context=ssl_context,
        )
        for model in models
    )


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
        pass

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    return json.loads(stripped[start : end + 1])


def _normalize_game(value: object, *, fallback: str | None) -> str | None:
    normalized = _normalize_text_field(value)
    if normalized in {"pokemon", "ws"}:
        return normalized
    if fallback in {"pokemon", "ws"}:
        return fallback
    return None


def _normalize_item_kind(value: object) -> str | None:
    normalized = _normalize_text_field(value)
    if normalized in {"card", "sealed_box"}:
        return normalized
    return None


def _normalize_aliases(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    aliases: list[str] = []
    for raw in value:
        normalized = _normalize_text_field(raw)
        if normalized and normalized not in aliases:
            aliases.append(normalized)
    return tuple(aliases)


def _normalize_card_number_field(value: object) -> str | None:
    normalized = _normalize_text_field(value)
    if normalized is None:
        return None
    return normalize_card_number(normalized)


def _normalize_token(value: object, *, uppercase: bool) -> str | None:
    normalized = _normalize_text_field(value)
    if normalized is None:
        return None
    collapsed = "".join(character for character in normalized if character.isalnum() or character in {"/", "+", "-"})
    if not collapsed:
        return None
    return collapsed.upper() if uppercase else collapsed.lower()


def _normalize_text_field(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"null", "none", "unknown", "n/a"}:
        return None
    return text


def _normalize_confidence(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_model_list(value: object) -> tuple[str, ...]:
    normalized = _normalize_text_field(value)
    if normalized is None:
        return ()
    models: list[str] = []
    for raw in normalized.split(","):
        model = raw.strip()
        if model and model not in models:
            models.append(model)
    return tuple(models)
