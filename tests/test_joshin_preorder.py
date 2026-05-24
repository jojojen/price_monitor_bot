"""Tests for Joshin pre-order / lottery crawler (uses HTML fixture)."""

from __future__ import annotations

from pathlib import Path

from bs4 import BeautifulSoup

from market_monitor.joshin_preorder import (
    STORE_NAME,
    _is_tcg,
    _parse_price,
    parse_joshin_page,
)
from market_monitor.official_store_base import LOTTERY_CLOSED, LOTTERY_OPEN

FIXTURES = Path(__file__).parent / "fixtures"
BASE_URL = "https://joshinweb.jp/campaign/lottery/"


def _load(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


def _soup(name: str) -> BeautifulSoup:
    return BeautifulSoup(_load(name), "html.parser")


# ── _is_tcg ──────────────────────────────────────────────────────────────────


def test_is_tcg_matches_union_arena():
    assert _is_tcg("UNION ARENA エクストラブースター チェンソーマン")


def test_is_tcg_matches_pokemon():
    assert _is_tcg("ポケモンカードゲーム スカーレット＆バイオレット")


def test_is_tcg_matches_weiss():
    assert _is_tcg("ヴァイスシュヴァルツ ブースターパック")


def test_is_tcg_rejects_non_tcg():
    assert not _is_tcg("その他非TCG商品 限定版")
    assert not _is_tcg("限定フィギュア 特典付き")


# ── _parse_price ─────────────────────────────────────────────────────────────


def test_parse_price_with_comma():
    assert _parse_price("4,180円（税込）") == 4180


def test_parse_price_plain():
    assert _parse_price("6050円") == 6050


def test_parse_price_returns_none_for_no_match():
    assert _parse_price("受付中") is None


# ── parse_joshin_page ─────────────────────────────────────────────────────────


def test_parse_joshin_page_returns_tcg_listings_only():
    soup = _soup("joshin_lottery_page.html")
    listings = parse_joshin_page(soup, base_url=BASE_URL)
    titles = [l.title for l in listings]
    # TCG items returned
    assert any("チェンソーマン" in t for t in titles)
    assert any("ポケモン" in t for t in titles)
    assert any("鬼滅" in t for t in titles)
    # Non-TCG item excluded
    assert not any("非TCG" in t for t in titles)


def test_parse_joshin_page_count():
    soup = _soup("joshin_lottery_page.html")
    listings = parse_joshin_page(soup, base_url=BASE_URL)
    assert len(listings) == 3


def test_parse_joshin_page_open_listing_has_lottery_open_status():
    soup = _soup("joshin_lottery_page.html")
    listings = parse_joshin_page(soup, base_url=BASE_URL)
    ua_listing = next(l for l in listings if "チェンソーマン" in l.title)
    assert ua_listing.status == LOTTERY_OPEN


def test_parse_joshin_page_closed_listing_has_lottery_closed_status():
    soup = _soup("joshin_lottery_page.html")
    listings = parse_joshin_page(soup, base_url=BASE_URL)
    kny_listing = next(l for l in listings if "鬼滅" in l.title)
    assert kny_listing.status == LOTTERY_CLOSED


def test_parse_joshin_page_price_parsed():
    soup = _soup("joshin_lottery_page.html")
    listings = parse_joshin_page(soup, base_url=BASE_URL)
    ua_listing = next(l for l in listings if "チェンソーマン" in l.title)
    assert ua_listing.price_jpy == 4180


def test_parse_joshin_page_deadline_parsed():
    soup = _soup("joshin_lottery_page.html")
    listings = parse_joshin_page(soup, base_url=BASE_URL)
    ua_listing = next(l for l in listings if "チェンソーマン" in l.title)
    assert ua_listing.deadline_iso is not None
    assert "06-15" in ua_listing.deadline_iso


def test_parse_joshin_page_open_date_parsed():
    soup = _soup("joshin_lottery_page.html")
    listings = parse_joshin_page(soup, base_url=BASE_URL)
    ua_listing = next(l for l in listings if "チェンソーマン" in l.title)
    assert ua_listing.open_date_iso is not None
    assert "06-01" in ua_listing.open_date_iso


def test_parse_joshin_page_url_is_absolute():
    soup = _soup("joshin_lottery_page.html")
    listings = parse_joshin_page(soup, base_url=BASE_URL)
    for listing in listings:
        assert listing.url.startswith("https://")


def test_parse_joshin_page_store_name():
    soup = _soup("joshin_lottery_page.html")
    listings = parse_joshin_page(soup, base_url=BASE_URL)
    assert all(l.store_name == STORE_NAME for l in listings)


def test_parse_joshin_page_item_key_stable():
    soup = _soup("joshin_lottery_page.html")
    listings = parse_joshin_page(soup, base_url=BASE_URL)
    keys = [l.item_key for l in listings]
    assert len(keys) == len(set(keys)), "item_keys must be unique"
