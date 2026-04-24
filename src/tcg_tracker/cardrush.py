from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import logging
import time
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup

from market_monitor.http import HttpClient
from market_monitor.models import MarketOffer

from .catalog import TcgCardSpec
from .hot_cards import CARDRUSH_BROWSER_HEADERS, _parse_cardrush_text
from .matching import minimum_match_score, score_tcg_offer
from .search_terms import build_lookup_terms

CARDRUSH_BASE_URL = "https://www.cardrush-pokemon.jp"
CARDRUSH_PRODUCT_LIST_URL = f"{CARDRUSH_BASE_URL}/product-list"
logger = logging.getLogger(__name__)


class CardrushPokemonClient:
    _cooldown_seconds = 600.0
    _disabled_until_monotonic = 0.0

    def __init__(self, http_client: HttpClient | None = None) -> None:
        self.http_client = http_client or HttpClient()

    def lookup(self, spec: TcgCardSpec, *, minimum_score: float | None = None) -> list[MarketOffer]:
        if spec.game != "pokemon":
            return []
        if self._is_temporarily_disabled():
            logger.debug("Cardrush lookup skipped because the source is temporarily disabled.")
            return []

        resolved_minimum_score = minimum_score if minimum_score is not None else minimum_match_score(spec)
        search_terms = build_lookup_terms(spec)
        logger.debug(
            "Cardrush lookup starting title=%s card_number=%s rarity=%s set_code=%s search_terms=%s minimum_score=%s",
            spec.title,
            spec.card_number,
            spec.rarity,
            spec.set_code,
            list(search_terms),
            resolved_minimum_score,
        )
        matched: list[MarketOffer] = []
        for offer in self._search_candidates(spec):
            score = score_tcg_offer(spec, offer)
            logger.debug("Cardrush candidate scored score=%s offer=%s", score, _offer_summary(offer))
            if score >= resolved_minimum_score:
                matched.append(replace(offer, score=score))
        logger.debug(
            "Cardrush matched offers count=%s matched=%s",
            len(matched),
            [_offer_summary(offer) for offer in matched[: _log_limit()]],
        )
        return matched

    def _search_candidates(self, spec: TcgCardSpec) -> list[MarketOffer]:
        offers: list[MarketOffer] = []
        seen: set[str] = set()
        for search_word in build_lookup_terms(spec):
            logger.debug("Cardrush search term=%s", search_word)
            try:
                html = self.http_client.get_text(
                    CARDRUSH_PRODUCT_LIST_URL,
                    params={"keyword": search_word},
                    headers=CARDRUSH_BROWSER_HEADERS,
                )
            except Exception as exc:
                self._temporarily_disable()
                logger.warning(
                    "Cardrush search failed term=%s; temporarily disabling the source for %.0f seconds. error=%s",
                    search_word,
                    self._cooldown_seconds,
                    exc,
                )
                break
            raw_offers = self._parse_search_page(html)
            logger.debug(
                "Cardrush raw candidates term=%s count=%s offers=%s",
                search_word,
                len(raw_offers),
                [_offer_summary(offer) for offer in raw_offers[: _log_limit()]],
            )
            for offer in raw_offers:
                if offer.url in seen:
                    continue
                seen.add(offer.url)
                offers.append(offer)
        return offers

    @classmethod
    def reset_temporary_disable(cls) -> None:
        cls._disabled_until_monotonic = 0.0

    @classmethod
    def _temporarily_disable(cls) -> None:
        cls._disabled_until_monotonic = time.monotonic() + cls._cooldown_seconds

    @classmethod
    def _is_temporarily_disabled(cls) -> bool:
        return time.monotonic() < cls._disabled_until_monotonic

    def _parse_search_page(self, html: str) -> list[MarketOffer]:
        soup = BeautifulSoup(html, "html.parser")
        offers: list[MarketOffer] = []
        for anchor in soup.select("ul.item_list li.list_item_cell div.item_data a[href]"):
            raw_text = " ".join(anchor.get_text(" ", strip=True).split())
            if not raw_text:
                continue

            detail_url = urljoin(CARDRUSH_BASE_URL, anchor["href"])
            parsed = _parse_cardrush_text(
                raw_text,
                detail_url=detail_url,
                board_url=CARDRUSH_PRODUCT_LIST_URL,
            )
            if parsed is None or parsed.price_jpy is None:
                continue

            listing_id = urlparse(detail_url).path.strip("/") or detail_url
            attributes = {
                "card_number": parsed.card_number or "",
                "rarity": parsed.rarity or "",
                "version_code": parsed.set_code or "",
                "set_code": parsed.set_code or "",
                "image_alt": raw_text,
            }
            if parsed.listing_count is not None:
                attributes["listing_count"] = str(parsed.listing_count)

            offers.append(
                MarketOffer(
                    source="cardrush_pokemon",
                    listing_id=listing_id,
                    url=detail_url,
                    title=parsed.title,
                    price_jpy=parsed.price_jpy,
                    price_kind="ask",
                    captured_at=datetime.now(timezone.utc),
                    source_category="specialty_store",
                    availability=None if parsed.listing_count is None else f"stock {parsed.listing_count}",
                    condition=parsed.condition,
                    attributes=attributes,
                )
            )
        return offers


def _offer_summary(offer: MarketOffer) -> dict[str, object]:
    return {
        "source": offer.source,
        "price_kind": offer.price_kind,
        "price_jpy": offer.price_jpy,
        "title": offer.title,
        "url": offer.url,
        "card_number": offer.attributes.get("card_number", ""),
        "rarity": offer.attributes.get("rarity", ""),
        "set_code": offer.attributes.get("version_code", "") or offer.attributes.get("set_code", ""),
    }


_LOG_LIMIT = 5


def _log_limit() -> int:
    return _LOG_LIMIT
