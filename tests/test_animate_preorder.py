"""Tests for Animate Online Shop pre-order crawler (HTML fixture)."""

from __future__ import annotations

from pathlib import Path

from bs4 import BeautifulSoup

from market_monitor.animate_preorder import STORE_NAME, _is_tcg, parse_animate_page
from market_monitor.official_store_base import PREORDER_OPEN

FIXTURES = Path(__file__).parent / "fixtures"
BASE_URL = "https://www.animate-onlineshop.jp/sp/fair/reserve/"


def _soup(name: str) -> BeautifulSoup:
    return BeautifulSoup((FIXTURES / name).read_text(encoding="utf-8"), "html.parser")


def test_is_tcg_accepts_union_arena():
    assert _is_tcg("UNION ARENA ブースターパック 呪術廻戦")


def test_is_tcg_rejects_figure():
    assert not _is_tcg("フィギュア 限定版（非TCG）")


def test_parse_animate_returns_tcg_only():
    listings = parse_animate_page(_soup("animate_preorder_page.html"), base_url=BASE_URL)
    titles = [l.title for l in listings]
    assert any("呪術廻戦" in t for t in titles)
    assert any("ポケモン" in t for t in titles)
    assert not any("フィギュア" in t for t in titles)


def test_parse_animate_count():
    listings = parse_animate_page(_soup("animate_preorder_page.html"), base_url=BASE_URL)
    assert len(listings) == 2


def test_parse_animate_status_preorder_open():
    listings = parse_animate_page(_soup("animate_preorder_page.html"), base_url=BASE_URL)
    assert all(l.status == PREORDER_OPEN for l in listings)


def test_parse_animate_price_parsed():
    listings = parse_animate_page(_soup("animate_preorder_page.html"), base_url=BASE_URL)
    jcc = next(l for l in listings if "呪術廻戦" in l.title)
    assert jcc.price_jpy == 4400


def test_parse_animate_deadline_parsed():
    listings = parse_animate_page(_soup("animate_preorder_page.html"), base_url=BASE_URL)
    jcc = next(l for l in listings if "呪術廻戦" in l.title)
    assert jcc.deadline_iso == "2026-06-30"


def test_parse_animate_url_is_absolute():
    listings = parse_animate_page(_soup("animate_preorder_page.html"), base_url=BASE_URL)
    for l in listings:
        assert l.url.startswith("https://")


def test_parse_animate_store_name():
    listings = parse_animate_page(_soup("animate_preorder_page.html"), base_url=BASE_URL)
    assert all(l.store_name == STORE_NAME for l in listings)
