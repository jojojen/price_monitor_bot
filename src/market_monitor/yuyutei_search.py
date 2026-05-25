"""Yuyutei (yuyu-tei.jp) sell-listing search client.

Yuyu亭 is Japan's largest TCG card shop. This module implements
``MarketplaceSearchClient`` for the *sell* (ask) side only — yuyutei
publishes fixed sell prices directly on the web without authentication.

Avoids importing ``tcg_tracker`` (which imports ``market_monitor.http``) to
prevent a circular dependency. Only the fields needed by ``MarketplaceListing``
are extracted; card-specific metadata (rarity, card_number) is left to the
``tcg_tracker`` layer.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping
from typing import Any
from urllib.parse import urljoin, urlparse, urlencode

from bs4 import BeautifulSoup

from .http import HttpClient
from .marketplace_search import MarketplaceListing

logger = logging.getLogger(__name__)

YUYUTEI_BASE_URL = "https://yuyu-tei.jp"

# Game codes supported in the sell path: /sell/{code}/s/search
DEFAULT_GAME_CODES: tuple[str, ...] = ("ygo", "ws", "ua", "op")
# ygo = 遊戯王  ws = Weiss Schwarz  ua = Union Arena  op = One Piece Card Game

_PRICE_DIGITS_RE = re.compile(r"(\d[\d,]*)")


def _parse_jpy(text: str) -> int | None:
    if not text:
        return None
    match = _PRICE_DIGITS_RE.search(text)
    if match is None:
        return None
    try:
        return int(match.group(1).replace(",", ""))
    except ValueError:
        return None


def _parse_sell_listings(html: str, *, game_code: str, base_url: str = YUYUTEI_BASE_URL) -> list[dict]:
    """Parse yuyutei sell-search HTML and return raw dicts (title, price_jpy, url, item_id).

    Skips cards where ``label.cart_sell_zaiko`` contains 「在庫なし」."""
    soup = BeautifulSoup(html, "html.parser")
    results: list[dict] = []

    for card in soup.select("div.card-product"):
        # Skip out-of-stock cards
        zaiko = card.select_one("label.cart_sell_zaiko")
        if zaiko is not None and "在庫なし" in zaiko.get_text():
            continue

        anchors = [a for a in card.select("a[href]") if "/card/" in a.get("href", "")]
        title_el = card.find("h4")
        price_el = card.find("strong")

        if not anchors or title_el is None or price_el is None:
            continue

        href = urljoin(base_url, anchors[0]["href"])
        path_parts = [p for p in urlparse(href).path.split("/") if p]
        version_code = path_parts[-2] if len(path_parts) >= 2 else ""
        product_id = path_parts[-1] if path_parts else href
        item_id = f"{version_code}:{product_id}"

        title = title_el.get_text(" ", strip=True)
        price_jpy = _parse_jpy(price_el.get_text(" ", strip=True))
        if price_jpy is None or price_jpy <= 0:
            continue

        results.append({
            "item_id": item_id,
            "title": title,
            "price_jpy": price_jpy,
            "url": href,
        })

    return results


class YuyuteiMarketplaceSearchClient:
    """``MarketplaceSearchClient`` for Yuyutei sell listings.

    Iterates over ``game_codes``, fetches the search page for each, then
    deduplicates, filters, sorts, and caps the combined result.

    ``source_options`` key recognised: ``"game_code"`` (str) — when set, only
    that single game code is queried."""

    source_name: str = "yuyutei"

    def __init__(
        self,
        http_client: HttpClient | None = None,
        game_codes: tuple[str, ...] = DEFAULT_GAME_CODES,
    ) -> None:
        self._http = http_client or HttpClient()
        self._game_codes = game_codes

    def search(
        self,
        query: str,
        *,
        price_max: int,
        max_results: int = 30,
        timeout_ms: int = 30_000,
        source_options: Mapping[str, Any] | None = None,
    ) -> list[MarketplaceListing]:
        if not query.strip() or price_max <= 0:
            return []

        opts = dict(source_options or {})
        if "game_code" in opts:
            codes: tuple[str, ...] = (str(opts["game_code"]),)
        else:
            codes = self._game_codes

        params = urlencode({"search_word": query, "rare": "", "type": ""})
        seen_ids: set[str] = set()
        raw_hits: list[dict] = []

        for code in codes:
            url = f"{YUYUTEI_BASE_URL}/sell/{code}/s/search?{params}"
            try:
                html = self._http.get_text(url, timeout_seconds=timeout_ms / 1000.0)
            except Exception:
                logger.warning("YuyuteiMarketplaceSearchClient: HTTP failed code=%s url=%s", code, url)
                continue
            for hit in _parse_sell_listings(html, game_code=code):
                if hit["item_id"] in seen_ids:
                    continue
                if hit["price_jpy"] > price_max:
                    continue
                seen_ids.add(hit["item_id"])
                raw_hits.append(hit)

        raw_hits.sort(key=lambda h: h["price_jpy"])
        return [
            MarketplaceListing(
                source="yuyutei",
                item_id=h["item_id"],
                title=h["title"],
                price_jpy=h["price_jpy"],
                url=h["url"],
                thumbnail_url=None,
                stock_count=None,
                listing_kind="fixed_price",
            )
            for h in raw_hits[:max_results]
        ]
