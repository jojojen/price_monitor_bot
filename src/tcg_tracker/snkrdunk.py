"""Snkrdunk (Sneaker Dunk) reference client.

Snkrdunk is a Japanese marketplace + price index that has become the de-facto
fair-price reference for sealed TCG boxes and other collectibles. Treated as
**Tier 1** (authoritative) in the new tiered lookup architecture — its search
HTML carries product price + title in plain ARIA labels so we can parse it
without JS or LLM.

Architecture note: this client ONLY runs the search → matched offers pipeline;
the per-product JSON-LD detail page extraction lives in
`market_monitor.llm_listing_extractor._rule_based_price_extraction` and is
used by the feedback flow on user-submitted URLs.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import logging
import re
from urllib.parse import quote

from market_monitor.http import HttpClient
from market_monitor.models import MarketOffer

from .catalog import TcgCardSpec
from .matching import minimum_match_score, score_tcg_offer
from .search_terms import build_lookup_terms
from .sealed_box_filters import looks_like_sealed_box_listing

logger = logging.getLogger(__name__)

SNKRDUNK_BASE_URL = "https://snkrdunk.com"
SNKRDUNK_SEARCH_URL = f"{SNKRDUNK_BASE_URL}/search"

# `aria-label="<title> - ¥<comma_price>"` lives on every product tile anchor.
# Also capture the href (absolute or relative) so we can build a stable URL.
_PRODUCT_TILE_RE = re.compile(
    r'<a[^>]+href="(?P<href>(?:https?://snkrdunk\.com)?/apparels/\d+)"[^>]*aria-label="(?P<aria>[^"]+)"',
    re.IGNORECASE | re.DOTALL,
)
# aria-label format: "<title> - ¥<price>"
_ARIA_PRICE_RE = re.compile(r"^(.+?)\s*-\s*¥\s*([\d,]+)\s*$")
_MAX_RESULTS = 12


class SnkrdunkClient:
    def __init__(self, http_client: HttpClient | None = None) -> None:
        self.http_client = http_client or HttpClient()

    def lookup(
        self,
        spec: TcgCardSpec,
        *,
        minimum_score: float | None = None,
    ) -> list[MarketOffer]:
        # snkrdunk indexes sealed boxes and sneakers well; cards less so.
        # For now we surface results for any TCG query — the score gate
        # filters non-matching results.
        resolved_minimum_score = (
            minimum_score if minimum_score is not None else minimum_match_score(spec)
        )

        matched: list[MarketOffer] = []
        for offer in self._search_candidates(spec):
            score = score_tcg_offer(spec, offer)
            logger.debug(
                "Snkrdunk candidate scored score=%s offer=%s", score, _offer_summary(offer)
            )
            if score >= resolved_minimum_score:
                matched.append(replace(offer, score=score))
        logger.debug(
            "Snkrdunk matched offers count=%s matched=%s",
            len(matched),
            [_offer_summary(o) for o in matched[: _MAX_RESULTS]],
        )
        return matched

    def _search_candidates(self, spec: TcgCardSpec) -> list[MarketOffer]:
        offers: list[MarketOffer] = []
        seen: set[str] = set()
        for term in build_lookup_terms(spec):
            url = f"{SNKRDUNK_SEARCH_URL}?keyword={quote(term)}"
            logger.debug("Snkrdunk search term=%s url=%s", term, url)
            try:
                html = self.http_client.get_text(url, timeout_seconds=12)
            except Exception as exc:
                logger.warning("Snkrdunk search failed term=%s error=%s", term, exc)
                continue
            raw = self._parse_search_html(html)
            logger.debug(
                "Snkrdunk raw candidates term=%s count=%s offers=%s",
                term, len(raw),
                [_offer_summary(o) for o in raw[: _MAX_RESULTS]],
            )
            for offer in raw:
                if offer.url in seen:
                    continue
                seen.add(offer.url)
                offers.append(offer)
                if len(offers) >= _MAX_RESULTS:
                    return offers
        return offers

    def _parse_search_html(self, html: str) -> list[MarketOffer]:
        """Pull `productTile` anchors out of the search-result HTML.
        Returns at most `_MAX_RESULTS` offers — snkrdunk shows the most
        relevant matches first, ranking shrinks past the top ~10."""
        now = datetime.now(timezone.utc)
        offers: list[MarketOffer] = []
        for match in _PRODUCT_TILE_RE.finditer(html):
            href = match.group("href")
            aria = match.group("aria")
            aria_match = _ARIA_PRICE_RE.match(aria.strip())
            if aria_match is None:
                continue
            title = aria_match.group(1).strip()
            price_str = aria_match.group(2).replace(",", "")
            if not price_str.isdigit():
                continue
            price = int(price_str)
            if price <= 0:
                continue
            url = href if href.startswith("http") else f"{SNKRDUNK_BASE_URL}{href}"
            listing_id = url.rsplit("/", 1)[-1]
            offers.append(
                MarketOffer(
                    source="snkrdunk",
                    listing_id=listing_id,
                    url=url,
                    title=title,
                    price_jpy=price,
                    price_kind="market",
                    captured_at=now,
                    source_category="marketplace",
                    attributes={
                        # Mark product kind via the shared classifier so the
                        # downstream sealed-box prefilter / matching code can
                        # rely on a single source of truth.
                        "product_kind": (
                            "sealed_box" if looks_like_sealed_box_listing(title) else "card"
                        ),
                        "image_alt": title,
                    },
                )
            )
            if len(offers) >= _MAX_RESULTS:
                break
        return offers


def _offer_summary(offer: MarketOffer) -> dict[str, object]:
    return {
        "source": offer.source,
        "price_kind": offer.price_kind,
        "price_jpy": offer.price_jpy,
        "title": offer.title,
        "url": offer.url,
        "product_kind": offer.attributes.get("product_kind", ""),
    }
