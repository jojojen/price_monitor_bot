from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import logging
from urllib.parse import urlencode, urljoin, urlparse

from bs4 import BeautifulSoup

from market_monitor.http import HttpClient
from market_monitor.models import MarketOffer

from .catalog import TcgCardSpec
from .hot_cards import _parse_magi_text
from .matching import minimum_match_score, score_tcg_offer
from .search_terms import build_lookup_terms

MAGI_BASE_URL = "https://magi.camp"
MAGI_PRODUCT_SEARCH_URL = f"{MAGI_BASE_URL}/products/search"
logger = logging.getLogger(__name__)


class MagiProductClient:
    def __init__(self, http_client: HttpClient | None = None) -> None:
        self.http_client = http_client or HttpClient()

    def lookup(self, spec: TcgCardSpec, *, minimum_score: float | None = None) -> list[MarketOffer]:
        resolved_minimum_score = minimum_score if minimum_score is not None else minimum_match_score(spec)
        search_terms = build_lookup_terms(spec)
        logger.debug(
            "Magi lookup starting title=%s card_number=%s rarity=%s set_code=%s search_terms=%s minimum_score=%s",
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
            logger.debug("Magi candidate scored score=%s offer=%s", score, _offer_summary(offer))
            if score >= resolved_minimum_score:
                matched.append(replace(offer, score=score))
        logger.debug(
            "Magi matched offers count=%s matched=%s",
            len(matched),
            [_offer_summary(offer) for offer in matched[: _log_limit()]],
        )
        return matched

    def _search_candidates(self, spec: TcgCardSpec) -> list[MarketOffer]:
        offers: list[MarketOffer] = []
        seen: set[str] = set()
        for search_word in build_lookup_terms(spec):
            logger.debug("Magi search term=%s", search_word)
            search_url = f"{MAGI_PRODUCT_SEARCH_URL}?{urlencode({'forms_search_items[keyword]': search_word})}"
            html = self.http_client.get_text(
                MAGI_PRODUCT_SEARCH_URL,
                params={"forms_search_items[keyword]": search_word},
            )
            raw_offers = self._parse_search_page(html, search_url=search_url)
            logger.debug(
                "Magi raw candidates term=%s count=%s offers=%s",
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

    def _parse_search_page(self, html: str, *, search_url: str) -> list[MarketOffer]:
        soup = BeautifulSoup(html, "html.parser")
        offers: list[MarketOffer] = []
        for anchor in soup.select("div.product-list__box a[href^='/products/']"):
            raw_text = " ".join(anchor.get_text(" ", strip=True).split())
            if not raw_text:
                continue

            detail_url = urljoin(MAGI_BASE_URL, anchor["href"])
            parsed = _parse_magi_text(
                raw_text,
                detail_url=detail_url,
                board_url=search_url,
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
            if parsed.is_graded:
                attributes["is_graded"] = "1"
                if parsed.grade_label:
                    attributes["grade_label"] = parsed.grade_label

            offers.append(
                MarketOffer(
                    source="magi",
                    listing_id=listing_id,
                    url=detail_url,
                    title=parsed.title,
                    price_jpy=parsed.price_jpy,
                    price_kind="market",
                    captured_at=datetime.now(timezone.utc),
                    source_category="marketplace",
                    availability=None
                    if parsed.listing_count is None
                    else f"active listings {parsed.listing_count}",
                    condition="graded" if parsed.is_graded else None,
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
