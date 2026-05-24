"""Tests for Yodobashi pre-order crawler (HTML fixture)."""

from __future__ import annotations

from pathlib import Path

from bs4 import BeautifulSoup

from market_monitor.yodobashi_preorder import STORE_NAME, parse_yodobashi_page
from market_monitor.official_store_base import PREORDER_CLOSED, PREORDER_OPEN

FIXTURES = Path(__file__).parent / "fixtures"
BASE_URL = "https://www.yodobashi.com/"


def _soup(name: str) -> BeautifulSoup:
    return BeautifulSoup((FIXTURES / name).read_text(encoding="utf-8"), "html.parser")


def test_parse_yodobashi_returns_listings():
    listings = parse_yodobashi_page(_soup("yodobashi_preorder_page.html"), base_url=BASE_URL)
    assert len(listings) == 3


def test_parse_yodobashi_ua_listing_preorder_open():
    listings = parse_yodobashi_page(_soup("yodobashi_preorder_page.html"), base_url=BASE_URL)
    ua = next(l for l in listings if "チェンソーマン" in l.title)
    assert ua.status == PREORDER_OPEN
    assert ua.price_jpy == 4180
    assert ua.deadline_iso is not None
    assert "06-20" in ua.deadline_iso


def test_parse_yodobashi_preorder_closed_listing():
    listings = parse_yodobashi_page(_soup("yodobashi_preorder_page.html"), base_url=BASE_URL)
    vs = next(l for l in listings if "推しの子" in l.title)
    assert vs.status == PREORDER_CLOSED


def test_parse_yodobashi_url_is_absolute():
    listings = parse_yodobashi_page(_soup("yodobashi_preorder_page.html"), base_url=BASE_URL)
    for l in listings:
        assert l.url.startswith("https://")


def test_parse_yodobashi_store_name():
    listings = parse_yodobashi_page(_soup("yodobashi_preorder_page.html"), base_url=BASE_URL)
    assert all(l.store_name == STORE_NAME for l in listings)


def test_parse_yodobashi_item_keys_unique():
    listings = parse_yodobashi_page(_soup("yodobashi_preorder_page.html"), base_url=BASE_URL)
    keys = [l.item_key for l in listings]
    assert len(keys) == len(set(keys))
