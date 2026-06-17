"""Rakuma (フリル / fril.jp) search client — Rakuten's flea market.

Public search URL is HTML-rendered (not a SPA in the cards we need), so a
plain HTTP GET + BeautifulSoup parse is enough. If Rakuma later starts
rendering listings client-side, this file is the only place to upgrade to
Playwright — the ``RakumaSearchClient`` interface and Monitor wiring stay
the same.

Rakuma-specific things this file decides locally so the rest of the system
stays generic:
  - item_id extraction from /item/<id> URL slug
  - price parsing strips ¥ / 円 / 千分位
  - thumbnail URL lazy-load fallback (data-src → src)
  - no Rakuma condition filter (Rakuma's condition enum differs from Mercari;
    we skip it for the first cut)
"""

from __future__ import annotations

import logging
import re
import urllib.error
import urllib.request
from collections.abc import Mapping
from typing import Any
from urllib.parse import quote

from bs4 import BeautifulSoup

from . import browser_stealth as bs
from .http import host_cooldown_remaining, note_http_error, note_http_success
from .marketplace_search import MarketplaceListing

logger = logging.getLogger(__name__)

RAKUMA_SEARCH_BASE = "https://fril.jp/s"
RAKUMA_ITEM_PATH_PREFIX = "/item/"

_PRICE_RE = re.compile(r"(\d[\d,]*)")
# Rakuma item URLs look like https://fril.jp/<slug>/item/<id> OR
# https://fril.jp/item/<id> — we only need the trailing numeric id.
_ITEM_ID_RE = re.compile(r"/item/(\d+)")


def build_search_url(
    query: str,
    *,
    price_max: int,
    transaction: str = "selling",
    sort: str = "created_at",
    order: str = "desc",
) -> str:
    parts = [
        f"query={quote(query)}",
        f"max_price={price_max}",
        f"transaction={transaction}",
        f"sort={sort}",
        f"order={order}",
    ]
    return f"{RAKUMA_SEARCH_BASE}?{'&'.join(parts)}"


def _fetch_html(url: str, *, timeout_seconds: float) -> str | None:
    remaining = host_cooldown_remaining(url)
    if remaining > 0:
        logger.warning("Rakuma fetch short-circuited (host rate-limited %.0fs) url=%s", remaining, url)
        return None
    request = urllib.request.Request(url, headers=bs.http_headers())
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            charset = response.headers.get_content_charset() or "utf-8"
            text = response.read().decode(charset, errors="replace")
        note_http_success(url)
        return text
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
        note_http_error(url, exc)
        logger.exception("Rakuma search fetch failed url=%s", url)
        return None


def _parse_price(raw: str) -> int | None:
    if not raw:
        return None
    match = _PRICE_RE.search(raw)
    if match is None:
        return None
    try:
        return int(match.group(1).replace(",", ""))
    except ValueError:
        return None


def _extract_item_id(href: str) -> str | None:
    if not href:
        return None
    match = _ITEM_ID_RE.search(href)
    return match.group(1) if match else None


def _absolute_url(href: str) -> str:
    if not href:
        return ""
    if href.startswith("http"):
        return href
    if href.startswith("//"):
        return f"https:{href}"
    if href.startswith("/"):
        return f"https://fril.jp{href}"
    return href


def parse_rakuma_listings(html: str, *, query: str, price_max: int) -> list[MarketplaceListing]:
    """Pure parser — extracted so unit tests can pass canned HTML fixtures."""
    soup = BeautifulSoup(html, "html.parser")
    listings: list[MarketplaceListing] = []
    seen_ids: set[str] = set()

    # Rakuma renders listings inside <a> tags pointing at /item/<id>.
    # We're permissive about the surrounding markup since fril.jp's HTML
    # changes occasionally; we anchor on the href pattern and pull whatever
    # title/price/thumbnail are siblings/descendants.
    for anchor in soup.find_all("a", href=_ITEM_ID_RE):
        href = anchor.get("href", "")
        item_id = _extract_item_id(href)
        if not item_id or item_id in seen_ids:
            continue

        # Title — try the anchor's text first; if empty, look for a child img alt.
        title = " ".join(anchor.get_text(strip=True).split())
        if not title:
            img = anchor.find("img")
            if img is not None:
                title = (img.get("alt") or "").strip()
        if not title:
            continue

        # Price — search the anchor's ancestor element (item card) for a
        # price-like number. We use CSS selectors (select_one) because
        # BeautifulSoup's find() expects tag names/kwargs, not class selectors.
        # Walk up at most 3 ancestors to find a container that holds the price.
        price_text = ""
        for container in [anchor, anchor.parent, anchor.parent.parent if anchor.parent else None]:
            if container is None:
                continue
            for selector in (".item-box__item-price", ".item-price", "[class*=price]"):
                el = container.select_one(selector)
                if el is not None:
                    price_text = el.get_text(strip=True)
                    if price_text:
                        break
            if price_text:
                break
        if not price_text and anchor.parent is not None:
            price_text = anchor.parent.get_text(" ", strip=True)
        price = _parse_price(price_text)
        if price is None or price <= 0:
            continue
        if price > price_max:
            continue

        # Thumbnail
        thumb_src: str | None = None
        img = anchor.find("img")
        if img is not None:
            thumb_src = img.get("data-src") or img.get("src") or None
            if thumb_src and thumb_src.startswith("//"):
                thumb_src = f"https:{thumb_src}"

        seen_ids.add(item_id)
        listings.append(
            MarketplaceListing(
                source="rakuma",
                item_id=item_id,
                title=title,
                price_jpy=price,
                url=_absolute_url(href),
                thumbnail_url=thumb_src,
            )
        )
    return listings


def search_rakuma(
    query: str,
    *,
    price_max: int,
    max_results: int = 30,
    timeout_ms: int = 30_000,
) -> list[MarketplaceListing]:
    """Search Rakuma (fril.jp) for listings matching ``query`` priced ≤ ``price_max``.

    Returns up to ``max_results`` listings sorted newest first. Failures
    (network / parse) return an empty list — the caller (monitor) handles
    the absence gracefully and logs the exception."""
    if not query.strip() or price_max <= 0:
        return []
    url = build_search_url(query, price_max=price_max)
    logger.info("Rakuma search query=%s price_max=%d url=%s", query, price_max, url)
    html = _fetch_html(url, timeout_seconds=timeout_ms / 1000.0)
    if html is None:
        return []
    listings = parse_rakuma_listings(html, query=query, price_max=price_max)
    return listings[:max_results]


class RakumaSearchClient:
    """``MarketplaceSearchClient`` for Rakuma.

    Source-specific options recognised: (none yet — Rakuma condition enum
    differs from Mercari's, and the first cut doesn't expose it). Unknown
    keys in ``source_options`` are silently ignored."""

    source_name: str = "rakuma"

    def search(
        self,
        query: str,
        *,
        price_max: int,
        max_results: int = 30,
        timeout_ms: int = 30_000,
        source_options: Mapping[str, Any] | None = None,
    ) -> list[MarketplaceListing]:
        return search_rakuma(
            query,
            price_max=price_max,
            max_results=max_results,
            timeout_ms=timeout_ms,
        )
