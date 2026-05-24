"""Tests for B4 crawlers: PokemonCenter, UA official, AmiAmi."""

from __future__ import annotations

import json
from pathlib import Path

from bs4 import BeautifulSoup

from market_monitor.pokecen_preorder import STORE_NAME as POKECEN, parse_pokecen_page
from market_monitor.ua_official_preorder import STORE_NAME as UA, parse_ua_official_page
from market_monitor.amiami_preorder import (
    STORE_NAME as AMIAMI,
    _api_item_to_listing,
    _is_tcg,
    parse_amiami_html,
)
from market_monitor.official_store_base import LOTTERY_OPEN, PREORDER_OPEN, COMING_SOON

FIXTURES = Path(__file__).parent / "fixtures"


def _soup(name: str) -> BeautifulSoup:
    return BeautifulSoup((FIXTURES / name).read_text(encoding="utf-8"), "html.parser")


# ── Pokemon Center ────────────────────────────────────────────────────────────


def test_pokecen_returns_two_listings():
    listings = parse_pokecen_page(_soup("pokecen_preorder_page.html"), base_url="https://www.pokemoncenter-online.com/special/yoyaku/")
    assert len(listings) == 2


def test_pokecen_preorder_open_listing():
    listings = parse_pokecen_page(_soup("pokecen_preorder_page.html"), base_url="https://www.pokemoncenter-online.com/")
    pack = next(l for l in listings if "拡張パック" in l.title)
    assert pack.status == PREORDER_OPEN
    assert pack.price_jpy == 6050
    assert pack.deadline_iso is not None
    assert "06-30" in pack.deadline_iso


def test_pokecen_lottery_open_listing():
    listings = parse_pokecen_page(_soup("pokecen_preorder_page.html"), base_url="https://www.pokemoncenter-online.com/")
    starter = next(l for l in listings if "スターター" in l.title)
    assert starter.status == LOTTERY_OPEN
    assert starter.deadline_iso is not None
    assert "06-20" in starter.deadline_iso
    assert starter.open_date_iso is not None
    assert "06-10" in starter.open_date_iso


def test_pokecen_urls_absolute():
    listings = parse_pokecen_page(_soup("pokecen_preorder_page.html"), base_url="https://www.pokemoncenter-online.com/")
    assert all(l.url.startswith("https://") for l in listings)


def test_pokecen_categories_include_pokemon():
    listings = parse_pokecen_page(_soup("pokecen_preorder_page.html"), base_url="https://www.pokemoncenter-online.com/")
    assert all("pokemon" in l.categories for l in listings)


# ── UA Official ───────────────────────────────────────────────────────────────


def test_ua_official_returns_two_listings():
    listings = parse_ua_official_page(_soup("ua_official_page.html"), base_url="https://ua-tcg.com/information/")
    assert len(listings) == 2


def test_ua_official_chainsaw_lottery_open():
    listings = parse_ua_official_page(_soup("ua_official_page.html"), base_url="https://ua-tcg.com/information/")
    csm = next(l for l in listings if "チェンソーマン" in l.title)
    assert csm.status == LOTTERY_OPEN
    assert csm.price_jpy == 4180
    assert csm.deadline_iso is not None
    assert "06-15" in csm.deadline_iso
    assert csm.open_date_iso is not None
    assert "06-01" in csm.open_date_iso


def test_ua_official_coming_soon():
    listings = parse_ua_official_page(_soup("ua_official_page.html"), base_url="https://ua-tcg.com/information/")
    spy = next(l for l in listings if "SPY" in l.title)
    assert spy.status == COMING_SOON


def test_ua_official_categories_include_union_arena():
    listings = parse_ua_official_page(_soup("ua_official_page.html"), base_url="https://ua-tcg.com/information/")
    assert all("union_arena" in l.categories for l in listings)


def test_ua_official_urls_absolute():
    listings = parse_ua_official_page(_soup("ua_official_page.html"), base_url="https://ua-tcg.com/information/")
    assert all(l.url.startswith("https://") for l in listings)


# ── AmiAmi API ────────────────────────────────────────────────────────────────


def test_amiami_is_tcg_accepts_ua():
    assert _is_tcg("UNION ARENA エクストラブースター チェンソーマン")


def test_amiami_is_tcg_rejects_figure():
    assert not _is_tcg("フィギュア 非TCG限定")


def test_amiami_api_item_to_listing_preorder():
    data = json.loads((FIXTURES / "amiami_api_response.json").read_text())
    ua_item = data["items"][0]
    listing = _api_item_to_listing(ua_item)
    assert listing is not None
    assert listing.status == PREORDER_OPEN
    assert listing.price_jpy == 4180
    assert "CARD-TCG-UA-CSM-001" in listing.item_key
    assert listing.store_name == AMIAMI


def test_amiami_api_item_to_listing_url_format():
    data = json.loads((FIXTURES / "amiami_api_response.json").read_text())
    listing = _api_item_to_listing(data["items"][0])
    assert listing is not None
    assert "gcode=CARD-TCG-UA-CSM-001" in listing.url


def test_amiami_api_filters_non_tcg():
    data = json.loads((FIXTURES / "amiami_api_response.json").read_text())
    items = data["items"]
    listings = [l for item in items if (l := _api_item_to_listing(item)) and _is_tcg(l.title)]
    titles = [l.title for l in listings]
    assert any("チェンソーマン" in t for t in titles)
    assert any("ポケモン" in t for t in titles)
    assert not any("非TCG" in t for t in titles)
    assert len(listings) == 2
