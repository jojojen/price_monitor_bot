"""Joshin web shop official lottery / pre-order crawler.

Target pages:
  - 抽選販売: https://joshinweb.jp/campaign/lottery/
  - TCG category top: https://joshinweb.jp/tcg/

Crawl strategy: fetch the lottery campaign list, parse each product card for
title / status / price / deadline. Fall back to an empty list on any error so
the pipeline continues without this source.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .official_store_base import (
    LOTTERY_OPEN,
    PREORDER_OPEN,
    OfficialStoreListing,
    _build_jst_iso,
    abs_url,
    infer_status,
    item_key_from_url,
)

if TYPE_CHECKING:
    from .http import HttpClient

logger = logging.getLogger(__name__)

STORE_NAME = "joshin"

# Entry points — ordered by signal value. Lottery page first (highest priority).
_ENTRY_URLS = (
    "https://joshinweb.jp/campaign/lottery/",
    "https://joshinweb.jp/tcg/",
)

# Keywords that identify TCG / collectible card product titles.
_TCG_KW = re.compile(
    r"カード|トレカ|(?<!非)TCG|ブースター|パック|ボックス|box|デッキ|"
    r"ポケモン|UNION ARENA|ユニオンアリーナ|ワンピース|遊戯王|"
    r"ヴァイス|Weiss|バトスピ",
    re.IGNORECASE,
)


@dataclass
class JoshinPreorderCrawler:
    """Crawls Joshin's lottery / pre-order pages for TCG products."""

    http_client: "HttpClient"
    store_name: str = STORE_NAME
    entry_urls: tuple[str, ...] = field(
        default_factory=lambda: _ENTRY_URLS
    )

    def fetch_listings(self, *, timeout_seconds: int = 30) -> list[OfficialStoreListing]:
        results: list[OfficialStoreListing] = []
        for url in self.entry_urls:
            try:
                listings = self._fetch_page(url, timeout_seconds=timeout_seconds)
                results.extend(listings)
            except Exception:
                logger.exception("JoshinPreorderCrawler: failed to fetch url=%s", url)
        seen: set[str] = set()
        deduped = []
        for item in results:
            if item.item_key not in seen:
                seen.add(item.item_key)
                deduped.append(item)
        logger.info(
            "JoshinPreorderCrawler: fetched listings=%d (from %d entry urls)",
            len(deduped), len(self.entry_urls),
        )
        return deduped

    def _fetch_page(self, url: str, *, timeout_seconds: int) -> list[OfficialStoreListing]:
        from bs4 import BeautifulSoup
        html = self.http_client.get_text(url, timeout_seconds=timeout_seconds)
        soup = BeautifulSoup(html, "html.parser")
        return parse_joshin_page(soup, base_url=url)


def parse_joshin_page(soup: "BeautifulSoup", *, base_url: str) -> list[OfficialStoreListing]:
    """Parse a Joshin HTML page and extract pre-order / lottery listings.

    Handles two layouts:
      Layout A — campaign/lottery page: ``<div class="lot-item">`` cards
      Layout B — category page: ``<li class="list-item">`` or similar list items

    Only returns listings whose titles contain TCG-related keywords."""
    listings: list[OfficialStoreListing] = []

    # Layout A: lottery campaign cards
    for card in soup.select("div.lot-item, div.lottery-item, article.lot-product"):
        listing = _parse_lot_card(card, base_url=base_url)
        if listing and _is_tcg(listing.title):
            listings.append(listing)

    # Layout B: generic product list items (category pages)
    if not listings:
        for item in soup.select("li.list-item, div.product-list-item, div.item-box"):
            listing = _parse_list_item(item, base_url=base_url)
            if listing and _is_tcg(listing.title):
                listings.append(listing)

    return listings


def _parse_lot_card(card: "BeautifulSoup", *, base_url: str) -> OfficialStoreListing | None:
    """Parse a lottery campaign card block."""
    # Title + URL
    a_tag = card.select_one("a[href]")
    if not a_tag:
        return None
    href = a_tag.get("href", "")
    url = abs_url(base_url, str(href))
    title_tag = card.select_one(".lot-name, .product-name, h2, h3, h4")
    title = (title_tag.get_text(strip=True) if title_tag else a_tag.get_text(strip=True))
    if not title:
        return None

    # Status
    status_tag = card.select_one(".status, .lot-status, .badge, [class*='status']")
    status_text = status_tag.get_text(strip=True) if status_tag else ""
    status = infer_status(status_text or card.get_text())

    # Price
    price_jpy = _parse_price(card.get_text())

    # Deadline — look for 〜 or まで pattern
    full_text = card.get_text()
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
        categories=("tcg",),
    )


def _parse_list_item(item: "BeautifulSoup", *, base_url: str) -> OfficialStoreListing | None:
    """Parse a generic product list item."""
    a_tag = item.select_one("a[href]")
    if not a_tag:
        return None
    href = a_tag.get("href", "")
    url = abs_url(base_url, str(href))
    title_tag = item.select_one(".title, .product-name, .name, h3, h4")
    title = (title_tag.get_text(strip=True) if title_tag else a_tag.get_text(strip=True))
    if not title:
        return None

    full_text = item.get_text()
    status = infer_status(full_text)
    # Only include pre-order / lottery items from category pages
    if status not in (LOTTERY_OPEN, PREORDER_OPEN):
        return None

    return OfficialStoreListing(
        store_name=STORE_NAME,
        item_key=item_key_from_url(url),
        title=title,
        url=url,
        status=status,
        price_jpy=_parse_price(full_text),
        deadline_iso=_extract_deadline(full_text),
        open_date_iso=_extract_open_date(full_text),
        categories=("tcg",),
    )


# ── Text extraction helpers ──────────────────────────────────────────────────

_PRICE_RE = re.compile(r"([\d,]+)円")
_DEADLINE_RE = re.compile(
    r"(?:申込締切|応募締切|締切|受付終了|まで)[^\d]*"
    r"(?:(\d{4})年)?(\d{1,2})月(\d{1,2})日"
    r"(?:\s*(\d{1,2})[:時](\d{2})(?:分)?)?"
)
_OPEN_DATE_RE = re.compile(
    r"(?:申込開始|受付開始|開始)[^\d]*"
    r"(?:(\d{4})年)?(\d{1,2})月(\d{1,2})日"
    r"(?:\s*(\d{1,2})[:時](\d{2})(?:分)?)?"
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
    m = _OPEN_DATE_RE.search(text)
    if not m:
        return None
    year = int(m.group(1)) if m.group(1) else None
    month, day = int(m.group(2)), int(m.group(3))
    hour = int(m.group(4)) if m.group(4) else None
    minute = int(m.group(5)) if m.group(5) else None
    return _build_jst_iso(year=year, month=month, day=day, hour=hour, minute=minute)


def _is_tcg(title: str) -> bool:
    return bool(_TCG_KW.search(title))
