from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import logging
import re
from typing import Callable

from market_monitor.mercari_search import search_mercari
from market_monitor.models import MarketOffer
from market_monitor.normalize import normalize_card_number

from .catalog import TcgCardSpec
from .matching import minimum_match_score, score_tcg_offer
from .search_terms import generic_card_number_variants

logger = logging.getLogger(__name__)

MercariSearchFn = Callable[..., list[dict[str, object]]]
_MERCARI_SUPPORTED_GAMES = {"yugioh", "union_arena"}
_DEFAULT_PRICE_MAX_JPY = 1_000_000
_DEFAULT_MAX_RESULTS = 12
_DEFAULT_TIMEOUT_MS = 30_000
_TCG_CODE_RE = re.compile(
    r"(?P<code>[A-Z0-9]+/[A-Z0-9]+(?:-[A-Z0-9]+)*-\d{1,3}|[A-Z0-9]{2,}-[A-Z]{1,4}\d{1,4})",
    re.IGNORECASE,
)
_RARITY_RE = re.compile(
    r"\b(?P<rarity>PSE|QCSE|SE|UR|UL|SR|R|N|SEC|SSP|SP|AP|U\*|U|C|R\*|SR\*)\b",
    re.IGNORECASE,
)


class MercariReferenceClient:
    def __init__(
        self,
        *,
        search_fn: MercariSearchFn = search_mercari,
        price_max_jpy: int = _DEFAULT_PRICE_MAX_JPY,
        max_results: int = _DEFAULT_MAX_RESULTS,
        timeout_ms: int = _DEFAULT_TIMEOUT_MS,
    ) -> None:
        self._search_fn = search_fn
        self._price_max_jpy = max(1, price_max_jpy)
        self._max_results = max(1, max_results)
        self._timeout_ms = max(1, timeout_ms)

    def lookup(self, spec: TcgCardSpec, *, minimum_score: float | None = None) -> list[MarketOffer]:
        if spec.game not in _MERCARI_SUPPORTED_GAMES:
            return []

        resolved_minimum_score = minimum_score if minimum_score is not None else minimum_match_score(spec)
        matched: list[MarketOffer] = []
        for offer in self._search_candidates(spec):
            score = score_tcg_offer(spec, offer)
            logger.debug("Mercari reference candidate scored score=%s offer=%s", score, _offer_summary(offer))
            if score >= resolved_minimum_score:
                matched.append(replace(offer, score=score))
        return matched

    def _search_candidates(self, spec: TcgCardSpec) -> list[MarketOffer]:
        offers: list[MarketOffer] = []
        seen: set[str] = set()
        for query in _build_mercari_queries(spec):
            logger.debug("Mercari reference search query=%s", query)
            results = self._search_fn(
                query,
                price_max=self._price_max_jpy,
                max_results=self._max_results,
                timeout_ms=self._timeout_ms,
            )
            logger.debug("Mercari reference raw query=%s count=%s", query, len(results))
            for raw in results:
                offer = _offer_from_mercari_item(raw, spec=spec)
                if offer is None or offer.url in seen:
                    continue
                seen.add(offer.url)
                offers.append(offer)
        return offers


def _build_mercari_queries(spec: TcgCardSpec) -> tuple[str, ...]:
    queries: list[str] = []
    title = spec.title.strip()
    context = _game_context_keyword(spec.game)
    if spec.card_number:
        card_numbers = (spec.card_number, *generic_card_number_variants(spec.card_number))
        for card_number in card_numbers:
            queries.append(f"{card_number} {title}".strip())
        if context:
            queries.append(f"{spec.card_number} {context} {title}".strip())
    if spec.rarity and title:
        queries.append(f"{context} {title} {spec.rarity}".strip() if context else f"{title} {spec.rarity}".strip())
    if title:
        queries.append(f"{context} {title}".strip() if context else title)

    deduped: list[str] = []
    for query in queries:
        cleaned = " ".join(query.split()).strip()
        if cleaned and cleaned not in deduped:
            deduped.append(cleaned)
    return tuple(deduped[:3])


def _game_context_keyword(game: str) -> str | None:
    if game == "union_arena":
        return "ユニオンアリーナ"
    if game == "yugioh":
        return "遊戯王"
    return None


def _offer_from_mercari_item(raw: dict[str, object], *, spec: TcgCardSpec) -> MarketOffer | None:
    item_id = str(raw.get("item_id") or "").strip()
    url = str(raw.get("url") or "").strip()
    title = str(raw.get("title") or "").strip()
    try:
        price_jpy = int(raw.get("price_jpy") or 0)
    except (TypeError, ValueError):
        return None
    if not item_id or not url or not title or price_jpy <= 0:
        return None

    card_number = _extract_matching_card_number(title, spec.card_number) or _extract_first_card_number(title) or ""
    rarity = _extract_rarity(title) or spec.rarity or ""
    attributes = {
        "card_number": card_number,
        "rarity": rarity,
        "version_code": _derive_set_code(card_number) or spec.set_code or "",
        "set_code": _derive_set_code(card_number) or spec.set_code or "",
        "image_alt": title,
    }
    thumbnail_url = str(raw.get("thumbnail_url") or "").strip()
    if thumbnail_url:
        attributes["thumbnail_url"] = thumbnail_url

    return MarketOffer(
        source="mercari",
        listing_id=item_id,
        url=url,
        title=title,
        price_jpy=price_jpy,
        price_kind="market",
        captured_at=datetime.now(timezone.utc),
        source_category="marketplace",
        attributes=attributes,
    )


def _extract_matching_card_number(title: str, requested: str | None) -> str | None:
    if not requested:
        return None
    requested_numbers = {normalize_card_number(requested)}
    requested_numbers.update(normalize_card_number(value) for value in generic_card_number_variants(requested))
    for match in _TCG_CODE_RE.finditer(title.upper()):
        code = match.group("code").upper()
        normalized = normalize_card_number(code)
        if normalized in requested_numbers:
            return code
    return None


def _extract_first_card_number(title: str) -> str | None:
    match = _TCG_CODE_RE.search(title.upper())
    if match is None:
        return None
    return match.group("code").upper()


def _extract_rarity(title: str) -> str | None:
    match = _RARITY_RE.search(title.upper())
    return None if match is None else match.group("rarity").upper()


def _derive_set_code(card_number: str) -> str | None:
    if not card_number:
        return None
    prefix = card_number.split("/", 1)[0].split("-", 1)[0].strip().lower()
    return prefix or None


def _offer_summary(offer: MarketOffer) -> dict[str, object]:
    return {
        "source": offer.source,
        "price_kind": offer.price_kind,
        "price_jpy": offer.price_jpy,
        "title": offer.title,
        "url": offer.url,
        "card_number": offer.attributes.get("card_number", ""),
        "rarity": offer.attributes.get("rarity", ""),
    }
