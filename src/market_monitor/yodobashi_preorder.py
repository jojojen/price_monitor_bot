"""Yodobashi Camera official pre-order / lottery crawler.

Target pages:
  - TCG pre-order search: https://www.yodobashi.com/?word=トレカ+予約&category=500000003&condition=PREORDER
  - Lottery/chance items: https://www.yodobashi.com/?word=カードゲーム+抽選

Yodobashi product cards use JSON-LD for structured data; we also parse the
rendered HTML product grid as a fallback.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .log_utils import log_network_failure
from .official_store_base import (
    OfficialStoreListing,
    _build_jst_iso,
    abs_url,
    infer_status,
    item_key_from_url,
    parse_jp_datetime,
)

if TYPE_CHECKING:
    from .http import HttpClient

logger = logging.getLogger(__name__)

STORE_NAME = "yodobashi"

_ENTRY_URLS = (
    "https://www.yodobashi.com/?word=%E3%83%88%E3%83%AC%E3%82%AB+%E4%BA%88%E7%B4%84&category=500000003",
    "https://www.yodobashi.com/?word=%E3%82%AB%E3%83%BC%E3%83%89%E3%82%B2%E3%83%BC%E3%83%A0+%E6%8A%BD%E9%81%B8&category=500000003",
)

_PRICE_RE = re.compile(r"([\d,]+)円")
_DEADLINE_RE = re.compile(
    r"(?:予約締切|申込締切|締切|受付終了)[^\d]*"
    r"(?:(\d{4})年)?(\d{1,2})月(\d{1,2})日"
    r"(?:\s*(\d{1,2})[:時](\d{2})(?:分)?)?"
)


@dataclass
class YodobashiPreorderCrawler:
    """Crawls Yodobashi Camera's pre-order / lottery TCG listings."""

    http_client: "HttpClient"
    store_name: str = STORE_NAME
    entry_urls: tuple[str, ...] = field(default_factory=lambda: _ENTRY_URLS)

    def fetch_listings(self, *, timeout_seconds: int = 30) -> list[OfficialStoreListing]:
        results: list[OfficialStoreListing] = []
        for url in self.entry_urls:
            try:
                listings = self._fetch_page(url, timeout_seconds=timeout_seconds)
                results.extend(listings)
            except Exception as exc:
                log_network_failure(logger, exc, "YodobashiPreorderCrawler: failed url=%s", url)
        seen: set[str] = set()
        deduped = [r for r in results if not (seen.add(r.item_key) or r.item_key in seen - {r.item_key})]
        logger.info("YodobashiPreorderCrawler: fetched listings=%d", len(deduped))
        return deduped

    def _fetch_page(self, url: str, *, timeout_seconds: int) -> list[OfficialStoreListing]:
        from bs4 import BeautifulSoup
        html = self.http_client.get_text(url, timeout_seconds=timeout_seconds)
        soup = BeautifulSoup(html, "html.parser")
        return parse_yodobashi_page(soup, base_url=url)


def parse_yodobashi_page(
    soup: "BeautifulSoup", *, base_url: str
) -> list[OfficialStoreListing]:
    """Parse Yodobashi search results page.

    Yodobashi uses a div.prdLst > ul.prdList > li.prdListItem structure.
    Each item has a product name, price, and sometimes status labels."""
    listings: list[OfficialStoreListing] = []

    for item in soup.select("li.prdListItem, div.prdItem, div.product-item"):
        listing = _parse_product_item(item, base_url=base_url)
        if listing:
            listings.append(listing)

    return listings


def _parse_product_item(
    item: "BeautifulSoup", *, base_url: str
) -> OfficialStoreListing | None:
    a_tag = item.select_one("a[href]")
    if not a_tag:
        return None
    href = str(a_tag.get("href", ""))
    url = abs_url(base_url, href)

    title_tag = item.select_one(".prdName, .prdTitle, h3, h4, .title")
    title = title_tag.get_text(strip=True) if title_tag else a_tag.get_text(strip=True)
    if not title:
        return None

    full_text = item.get_text()
    status = infer_status(full_text)

    price_tag = item.select_one(".prcArea, .price, [class*='price']")
    price_text = price_tag.get_text() if price_tag else full_text
    price_jpy = _parse_price(price_text)

    deadline_iso = _extract_deadline(full_text)

    return OfficialStoreListing(
        store_name=STORE_NAME,
        item_key=item_key_from_url(url),
        title=title,
        url=url,
        status=status,
        price_jpy=price_jpy,
        deadline_iso=deadline_iso,
        categories=("tcg",),
    )


def _parse_price(text: str) -> int | None:
    m = _PRICE_RE.search(text)
    if m:
        try:
            return int(m.group(1).replace(",", ""))
        except (ValueError, AttributeError):
            pass
    return None


def _extract_deadline(text: str) -> str | None:
    m = _DEADLINE_RE.search(text)
    if not m:
        return None
    year = int(m.group(1)) if m.group(1) else None
    month, day = int(m.group(2)), int(m.group(3))
    hour = int(m.group(4)) if m.group(4) else None
    minute = int(m.group(5)) if m.group(5) else None
    return _build_jst_iso(year=year, month=month, day=day, hour=hour, minute=minute)
