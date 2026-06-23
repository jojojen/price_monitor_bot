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

from dataclasses import dataclass
from dataclasses import replace
from datetime import datetime, timezone
import logging
import re
from urllib.parse import quote, urljoin, urlparse

from bs4 import BeautifulSoup, Tag

from market_monitor.http import HttpClient
from market_monitor.models import MarketOffer

from .catalog import TcgCardSpec
from .grading import looks_like_graded
from .matching import minimum_match_score, score_tcg_offer
from .search_terms import build_lookup_terms
from .sealed_box_filters import looks_like_sealed_box_listing

logger = logging.getLogger(__name__)

SNKRDUNK_BASE_URL = "https://snkrdunk.com"
SNKRDUNK_SEARCH_URL = f"{SNKRDUNK_BASE_URL}/search"

_PRODUCT_PATH_RE = re.compile(r"^/apparels/\d+(?:/used/\d+)?/?$")
_ARIA_PRICE_PATTERNS = (
    re.compile(r"^(.+?)\s*-\s*¥\s*([\d,]+)\s*$"),
    re.compile(r"^(.+?)\s*¥\s*([\d,]+)\s*$"),
    re.compile(r"^(.+?)\s*-\s*([\d,]+)\s*円\s*$"),
)
_CURRENCY_PRICE_RE = re.compile(r"(?:¥\s*(?P<yen>\d[\d,]*)|(?P<en>\d[\d,]*)\s*円)")
_MAX_RESULTS = 12


@dataclass(frozen=True, slots=True)
class SnkrdunkSearchTile:
    title: str
    url: str
    price_jpy: int
    image_url: str | None = None


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
        """Pull product tiles out of the search-result HTML.
        Returns at most `_MAX_RESULTS` offers — snkrdunk shows the most
        relevant matches first, ranking shrinks past the top ~10."""
        now = datetime.now(timezone.utc)
        offers: list[MarketOffer] = []
        for tile in iter_snkrdunk_search_tiles(html, limit=_MAX_RESULTS):
            listing_id = _listing_id(tile.url)
            attributes: dict[str, str] = {
                # Mark product kind via the shared classifier so the
                # downstream sealed-box prefilter / matching code can
                # rely on a single source of truth.
                "product_kind": (
                    "sealed_box" if looks_like_sealed_box_listing(tile.title) else "card"
                ),
                "image_alt": tile.title,
            }
            if tile.image_url:
                attributes["image_url"] = tile.image_url
            if looks_like_graded(tile.title):
                attributes["is_graded"] = "1"
            offers.append(
                MarketOffer(
                    source="snkrdunk",
                    listing_id=listing_id,
                    url=tile.url,
                    title=tile.title,
                    price_jpy=tile.price_jpy,
                    price_kind="market",
                    captured_at=now,
                    source_category="marketplace",
                    attributes=attributes,
                )
            )
            if len(offers) >= _MAX_RESULTS:
                break
        return offers


def iter_snkrdunk_search_tiles(
    html: str,
    *,
    limit: int | None = None,
) -> list[SnkrdunkSearchTile]:
    """Extract product tiles from SSR search HTML without depending on class names.

    The public search page currently carries both accessible labels and nested
    title/price spans. Use the ARIA label first, then fall back to visible tile
    text so small markup reshuffles do not zero out the source.
    """
    soup = BeautifulSoup(html, "html.parser")
    tiles: list[SnkrdunkSearchTile] = []
    seen_urls: set[str] = set()
    for anchor in soup.find_all("a", href=True):
        if not isinstance(anchor, Tag):
            continue
        url = _snkrdunk_product_url(str(anchor.get("href", "")))
        if url is None or url in seen_urls:
            continue
        parsed = _tile_title_price(anchor)
        if parsed is None:
            continue
        title, price_jpy = parsed
        if price_jpy <= 0:
            continue
        seen_urls.add(url)
        tiles.append(
            SnkrdunkSearchTile(
                title=title,
                url=url,
                price_jpy=price_jpy,
                image_url=_tile_image_url(anchor),
            )
        )
        if limit is not None and len(tiles) >= limit:
            break
    return tiles


def _snkrdunk_product_url(href: str) -> str | None:
    if not href:
        return None
    url = urljoin(SNKRDUNK_BASE_URL, href)
    parsed = urlparse(url)
    if parsed.netloc.lower() != "snkrdunk.com":
        return None
    path = parsed.path.rstrip("/") or parsed.path
    if _PRODUCT_PATH_RE.match(path) is None:
        return None
    return f"{parsed.scheme}://{parsed.netloc}{path}"


def _tile_title_price(anchor: Tag) -> tuple[str, int] | None:
    aria = anchor.get("aria-label")
    if isinstance(aria, str):
        parsed = _parse_labeled_price(aria)
        if parsed is not None:
            return parsed

    title = _tile_title(anchor)
    price = _tile_price(anchor)
    if title is None or price is None:
        return None
    return title, price


def _parse_labeled_price(text: str) -> tuple[str, int] | None:
    value = _clean_text(text)
    for pattern in _ARIA_PRICE_PATTERNS:
        match = pattern.match(value)
        if match is None:
            continue
        title = match.group(1).strip()
        price = _parse_digits(match.group(2))
        if title and price is not None:
            return title, price
    return None


def _tile_title(anchor: Tag) -> str | None:
    title_node = anchor.select_one('[class*="productName"], [data-testid*="title"]')
    if title_node is not None:
        title = _clean_text(title_node.get_text(" ", strip=True))
        if title:
            return title
    image = anchor.find("img", alt=True)
    if isinstance(image, Tag):
        alt = image.get("alt")
        if isinstance(alt, str):
            title = _clean_text(alt)
            if title:
                return title
    return None


def _tile_price(anchor: Tag) -> int | None:
    price_node = anchor.select_one('[class*="productPrice"], [data-testid*="price"]')
    if price_node is not None:
        price = _parse_digits(price_node.get_text(" ", strip=True))
        if price is not None:
            return price
    return _parse_currency_price(anchor.get_text(" ", strip=True))


def _tile_image_url(anchor: Tag) -> str | None:
    image = anchor.find("img", src=True)
    if not isinstance(image, Tag):
        return None
    src = image.get("src")
    if not isinstance(src, str) or not src.startswith("https://cdn.snkrdunk.com/"):
        return None
    return src


def _parse_currency_price(text: str) -> int | None:
    match = _CURRENCY_PRICE_RE.search(_clean_text(text))
    if match is None:
        return None
    return _parse_digits(match.group("yen") or match.group("en"))


def _parse_digits(text: str | None) -> int | None:
    if not text:
        return None
    digits = re.sub(r"[^0-9]", "", text)
    return int(digits) if digits else None


def _clean_text(text: str) -> str:
    return " ".join(text.split()).strip()


def _listing_id(url: str) -> str:
    path = urlparse(url).path.strip("/")
    if path.startswith("apparels/"):
        return path.removeprefix("apparels/")
    return path or url


def _offer_summary(offer: MarketOffer) -> dict[str, object]:
    return {
        "source": offer.source,
        "price_kind": offer.price_kind,
        "price_jpy": offer.price_jpy,
        "title": offer.title,
        "url": offer.url,
        "product_kind": offer.attributes.get("product_kind", ""),
    }
