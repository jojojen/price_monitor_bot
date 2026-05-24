"""AmiAmi pre-order / new release crawler.

AmiAmi has a public search API endpoint:
  GET https://api.amiami.com/api/v1.0/items
    ?s_st_list_newitem_available=1&s_cate_id=TCG&lang=ja

Fields in response: items[].gcode, items[].gname, items[].thumb_url,
items[].min_price, items[].condition_flag, items[].list_price,
items[].preorder_discount_flg, items[].instock_flg

We also fall back to HTML scraping of the TCG new-release page if the API
returns an unexpected shape.

Reference: https://github.com/AmiAmi/amiami-api (community documentation)
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .official_store_base import (
    PREORDER_OPEN,
    STATUS_UNKNOWN,
    OfficialStoreListing,
    _build_jst_iso,
    abs_url,
    infer_status,
    item_key_from_url,
)

if TYPE_CHECKING:
    from .http import HttpClient

logger = logging.getLogger(__name__)

STORE_NAME = "amiami"

_API_URL = (
    "https://api.amiami.com/api/v1.0/items"
    "?s_st_list_newitem_available=1"
    "&s_cate_id=TCG"
    "&lang=ja"
    "&pagemax=30"
)
_FALLBACK_HTML_URL = "https://www.amiami.com/jpn/search/list/?s_st_list_newitem_available=1&s_cate_id=TCG"

_PRICE_RE = re.compile(r"([\d,]+)円")

_TCG_KW = re.compile(
    r"カード|トレカ|(?<!非)TCG|ブースター|パック|ボックス|box|デッキ|"
    r"ポケモン|UNION ARENA|ユニオンアリーナ|遊戯王|ヴァイス|Weiss|"
    r"ワンピース|バトスピ",
    re.IGNORECASE,
)


@dataclass
class AmiAmiPreorderCrawler:
    """Crawls AmiAmi new releases / pre-orders via their public API."""

    http_client: "HttpClient"
    store_name: str = STORE_NAME
    api_url: str = _API_URL
    fallback_html_url: str = _FALLBACK_HTML_URL

    def fetch_listings(self, *, timeout_seconds: int = 30) -> list[OfficialStoreListing]:
        try:
            return self._fetch_api(timeout_seconds=timeout_seconds)
        except Exception:
            logger.exception("AmiAmiPreorderCrawler: API failed, trying HTML fallback")
        try:
            return self._fetch_html(timeout_seconds=timeout_seconds)
        except Exception:
            logger.exception("AmiAmiPreorderCrawler: HTML fallback also failed")
        return []

    def _fetch_api(self, *, timeout_seconds: int) -> list[OfficialStoreListing]:
        raw = self.http_client.get_text(
            self.api_url,
            timeout_seconds=timeout_seconds,
            headers={"X-User-Key": "amiami_dev"},
        )
        data = json.loads(raw)
        items = data.get("items") or []
        listings: list[OfficialStoreListing] = []
        for item in items:
            listing = _api_item_to_listing(item)
            if listing and _is_tcg(listing.title):
                listings.append(listing)
        logger.info("AmiAmiPreorderCrawler: API returned listings=%d", len(listings))
        return listings

    def _fetch_html(self, *, timeout_seconds: int) -> list[OfficialStoreListing]:
        from bs4 import BeautifulSoup
        html = self.http_client.get_text(self.fallback_html_url, timeout_seconds=timeout_seconds)
        soup = BeautifulSoup(html, "html.parser")
        return parse_amiami_html(soup, base_url=self.fallback_html_url)


def _api_item_to_listing(item: dict) -> OfficialStoreListing | None:
    gcode = str(item.get("gcode", ""))
    gname = str(item.get("gname", ""))
    if not gcode or not gname:
        return None
    url = f"https://www.amiami.com/jpn/detail/?gcode={gcode}"
    price_raw = item.get("min_price") or item.get("list_price")
    price_jpy: int | None = None
    if price_raw is not None:
        try:
            price_jpy = int(str(price_raw).replace(",", ""))
        except ValueError:
            pass
    instock = item.get("instock_flg")
    is_preorder = item.get("preorder_discount_flg") or item.get("s_st_list_newitem_available")
    if is_preorder:
        status = PREORDER_OPEN
    elif instock:
        status = "available"
    else:
        status = STATUS_UNKNOWN
    return OfficialStoreListing(
        store_name=STORE_NAME,
        item_key=f"amiami/{gcode}",
        title=gname,
        url=url,
        status=status,
        price_jpy=price_jpy,
        categories=("tcg",),
    )


def parse_amiami_html(soup: "BeautifulSoup", *, base_url: str) -> list[OfficialStoreListing]:
    """Parse AmiAmi HTML search results (fallback)."""
    listings: list[OfficialStoreListing] = []
    for item in soup.select("li.newly_added_item, div.product-item, li.item"):
        listing = _html_item_to_listing(item, base_url=base_url)
        if listing and _is_tcg(listing.title):
            listings.append(listing)
    return listings


def _html_item_to_listing(
    item: "BeautifulSoup", *, base_url: str
) -> OfficialStoreListing | None:
    a_tag = item.select_one("a[href]")
    if not a_tag:
        return None
    href = str(a_tag.get("href", ""))
    url = abs_url(base_url, href)
    title_tag = item.select_one(".newly_added_item_title, .title, h3, h4")
    title = title_tag.get_text(strip=True) if title_tag else a_tag.get_text(strip=True)
    if not title:
        return None
    full_text = item.get_text()
    status = infer_status(full_text)
    price_jpy = _parse_price(full_text)
    return OfficialStoreListing(
        store_name=STORE_NAME,
        item_key=item_key_from_url(url),
        title=title,
        url=url,
        status=status,
        price_jpy=price_jpy,
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


def _is_tcg(title: str) -> bool:
    return bool(_TCG_KW.search(title))
