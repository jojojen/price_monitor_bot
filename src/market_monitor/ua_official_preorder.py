"""UNION ARENA official store (ua-tcg.com) pre-order / lottery crawler.

Target pages:
  - 予約/抽選一覧: https://ua-tcg.com/information/

UA official uses a news/information page listing upcoming product info.
Each article card links to a product detail page.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .log_utils import log_network_failure
from .official_store_base import (
    COMING_SOON,
    OfficialStoreListing,
    _build_jst_iso,
    abs_url,
    infer_status,
    item_key_from_url,
)

if TYPE_CHECKING:
    from .http import HttpClient

logger = logging.getLogger(__name__)

STORE_NAME = "ua_official"

_ENTRY_URLS = (
    "https://ua-tcg.com/information/",
    "https://ua-tcg.com/product/",
)

_PRICE_RE = re.compile(r"([\d,]+)円")
_DEADLINE_RE = re.compile(
    r"(?:申込締切|抽選締切|締切|受付終了)[^\d]*"
    r"(?:(\d{4})年)?(\d{1,2})月(\d{1,2})日"
    r"(?:\s*(\d{1,2})[:時](\d{2})(?:分)?)?"
)
_OPEN_RE = re.compile(
    r"(?:申込開始|受付開始|抽選開始|発売日)[^\d]*"
    r"(?:(\d{4})年)?(\d{1,2})月(\d{1,2})日"
    r"(?:\s*(\d{1,2})[:時](\d{2})(?:分)?)?"
)

# UA product keywords for title filtering
_UA_PRODUCT_KW = re.compile(
    r"ブースター|パック|box|BOX|デッキ|エクストラ|スターター|"
    r"抽選|予約|限定|発売",
    re.IGNORECASE,
)


@dataclass
class UaOfficialPreorderCrawler:
    """Crawls UNION ARENA official site for pre-order / lottery listings."""

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
                log_network_failure(logger, exc, "UaOfficialPreorderCrawler: failed url=%s", url)
        seen: set[str] = set()
        deduped = []
        for r in results:
            if r.item_key not in seen:
                seen.add(r.item_key)
                deduped.append(r)
        logger.info("UaOfficialPreorderCrawler: fetched listings=%d", len(deduped))
        return deduped

    def _fetch_page(self, url: str, *, timeout_seconds: int) -> list[OfficialStoreListing]:
        from bs4 import BeautifulSoup
        html = self.http_client.get_text(url, timeout_seconds=timeout_seconds)
        soup = BeautifulSoup(html, "html.parser")
        return parse_ua_official_page(soup, base_url=url)


def parse_ua_official_page(
    soup: "BeautifulSoup", *, base_url: str
) -> list[OfficialStoreListing]:
    """Parse UA official site information/product pages.

    UA uses article or li elements for news/product listings."""
    listings: list[OfficialStoreListing] = []
    _SELECTORS = (
        "article.news-item",
        "article.product-item",
        "li.news-list__item",
        "li.product-list__item",
        "div.entry",
        "article",
    )
    for selector in _SELECTORS:
        cards = soup.select(selector)
        if not cards:
            continue
        for card in cards:
            listing = _parse_card(card, base_url=base_url)
            if listing:
                listings.append(listing)
        break
    return listings


def _parse_card(card: "BeautifulSoup", *, base_url: str) -> OfficialStoreListing | None:
    a_tag = card.select_one("a[href]")
    if not a_tag:
        return None
    href = str(a_tag.get("href", ""))
    url = abs_url(base_url, href)

    title_tag = card.select_one("h2, h3, h4, .title, .entry-title, .news-title")
    title = title_tag.get_text(strip=True) if title_tag else a_tag.get_text(strip=True)
    if not title:
        return None

    full_text = card.get_text()
    status = infer_status(full_text)
    if status == "unknown":
        if _UA_PRODUCT_KW.search(title):
            status = COMING_SOON

    price_jpy = _parse_price(full_text)
    deadline_iso = _extract_deadline(full_text)
    open_date_iso = _extract_open_date(full_text)

    return OfficialStoreListing(
        store_name=STORE_NAME,
        item_key=item_key_from_url(url),
        title=title,
        url=url,
        status=status,
        price_jpy=price_jpy,
        deadline_iso=deadline_iso,
        open_date_iso=open_date_iso,
        categories=("tcg", "union_arena"),
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


def _extract_open_date(text: str) -> str | None:
    m = _OPEN_RE.search(text)
    if not m:
        return None
    year = int(m.group(1)) if m.group(1) else None
    month, day = int(m.group(2)), int(m.group(3))
    hour = int(m.group(4)) if m.group(4) else None
    minute = int(m.group(5)) if m.group(5) else None
    return _build_jst_iso(year=year, month=month, day=day, hour=hour, minute=minute)
