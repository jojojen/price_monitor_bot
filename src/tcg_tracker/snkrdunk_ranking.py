"""Crawl popular TCG products from Snkrdunk's search results.

Snkrdunk's official ranking page is JS-rendered (no SSR product data), but
the SEARCH endpoint returns the same hot products at the top of its results
in plain SSR HTML — server-pre-rendered with full product tiles. We use
that as the de-facto "what's hot right now" feed.

Each result tile carries:
  - product URL (`https://snkrdunk.com/apparels/<id>`)
  - aria-label `"<title> - ¥<price>"`
  - thumbnail image URL on `cdn.snkrdunk.com/upload_bg_removed/...`

This module just extracts those tuples. The image-crawler downloads each
thumbnail, computes a perceptual hash, and persists into
`card_image_fingerprints` so future user uploads hit the fast path.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from urllib.parse import quote

from market_monitor.http import HttpClient

from .sealed_box_filters import looks_like_sealed_box_listing

logger = logging.getLogger(__name__)

SNKRDUNK_SEARCH_URL = "https://snkrdunk.com/search"

# Search keyword per game. Snkrdunk's index is broad — these umbrella terms
# surface the actually-trending products at the top of search results.
_GAME_SEED_QUERIES: dict[str, str] = {
    "pokemon": "ポケモンカードゲーム",
    "ws": "ヴァイスシュヴァルツ",
    "union_arena": "ユニオンアリーナ",
    "yugioh": "遊戯王",
}

_PRODUCT_TILE_RE = re.compile(
    r'<a[^>]+href="(?P<href>(?:https?://snkrdunk\.com)?/apparels/\d+)"[^>]*'
    r'aria-label="(?P<aria>[^"]+)"[^>]*>.*?'
    r'<img[^>]+src="(?P<img>https://cdn\.snkrdunk\.com/[^"]+)"',
    re.IGNORECASE | re.DOTALL,
)
_ARIA_PRICE_RE = re.compile(r"^(.+?)\s*-\s*¥\s*([\d,]+)\s*$")


@dataclass(frozen=True, slots=True)
class RankedProduct:
    title: str
    product_url: str
    image_url: str
    price_jpy: int
    item_kind: str  # "sealed_box" or "card"
    rank: int       # 1-based position in the search result list


def iter_ranked_products(
    *,
    game: str,
    http_client: HttpClient | None = None,
    limit: int = 50,
    image_size: str = "l",
) -> list[RankedProduct]:
    """Fetch Snkrdunk search results for the given game and parse the
    top-N tiles. Replaces JS-rendered ranking page with SSR-friendly
    search-result extraction (proven equivalent on 2026-05-26 inspection)."""
    query = _GAME_SEED_QUERIES.get(game)
    if query is None:
        logger.warning("No snkrdunk seed query configured for game=%s", game)
        return []
    client = http_client or HttpClient()
    url = f"{SNKRDUNK_SEARCH_URL}?keyword={quote(query)}&category=apparels"
    try:
        html = client.get_text(url, timeout_seconds=15)
    except Exception as exc:
        logger.warning("Snkrdunk ranking fetch failed game=%s error=%s", game, exc)
        return []

    products: list[RankedProduct] = []
    for idx, m in enumerate(_PRODUCT_TILE_RE.finditer(html), start=1):
        href = m.group("href")
        aria = m.group("aria")
        img = m.group("img")
        product_url = href if href.startswith("http") else f"https://snkrdunk.com{href}"
        aria_match = _ARIA_PRICE_RE.match(aria.strip())
        if aria_match is None:
            continue
        title = aria_match.group(1).strip()
        price_str = aria_match.group(2).replace(",", "")
        try:
            price = int(price_str)
        except ValueError:
            continue
        if price <= 0:
            continue
        # Bump image to higher resolution — the search HTML defaults to "size=m"
        # for layout density; "size=l" gives us better pixels for hashing.
        image_url = re.sub(r"\?size=\w+", f"?size={image_size}", img)
        item_kind = "sealed_box" if looks_like_sealed_box_listing(title) else "card"
        products.append(
            RankedProduct(
                title=title,
                product_url=product_url,
                image_url=image_url,
                price_jpy=price,
                item_kind=item_kind,
                rank=idx,
            )
        )
        if len(products) >= limit:
            break
    logger.info(
        "Snkrdunk ranking: game=%s parsed=%d (sealed_box=%d)",
        game, len(products), sum(1 for p in products if p.item_kind == "sealed_box"),
    )
    return products
