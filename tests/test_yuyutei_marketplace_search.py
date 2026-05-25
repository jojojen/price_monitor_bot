"""Tests for YuyuteiMarketplaceSearchClient and _parse_sell_listings."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from market_monitor.yuyutei_search import (
    YuyuteiMarketplaceSearchClient,
    _parse_sell_listings,
)


# ── HTML fixtures ─────────────────────────────────────────────────────────────

def _make_card(
    *,
    title: str = "チェンソーマン SR",
    price: str = "¥1,200",
    href: str = "/card/ua/ver01/12345",
    out_of_stock: bool = False,
) -> str:
    zaiko = '<label class="cart_sell_zaiko">在庫なし</label>' if out_of_stock else ""
    return f"""
    <div class="card-product">
      {zaiko}
      <a href="{href}"><h4>{title}</h4></a>
      <strong>{price}</strong>
    </div>
    """


def _make_page(cards_html: str) -> str:
    return f"""
    <html><body>
      <div id="power">
        <div class="cards-list">
          <h3>Card List</h3>
          {cards_html}
        </div>
      </div>
    </body></html>
    """


_CARD_A = _make_card(title="Card A SR", price="¥500", href="/card/ua/v1/001")
_CARD_B = _make_card(title="Card B R",  price="¥1500", href="/card/ua/v1/002")
_CARD_SOLD_OUT = _make_card(title="Card C", price="¥300", href="/card/ua/v1/003", out_of_stock=True)

FIXTURE_PAGE = _make_page(_CARD_A + _CARD_B + _CARD_SOLD_OUT)


# ── _parse_sell_listings unit tests ──────────────────────────────────────────


def test_parse_returns_in_stock_cards() -> None:
    hits = _parse_sell_listings(FIXTURE_PAGE, game_code="ua")
    titles = [h["title"] for h in hits]
    assert "Card A SR" in titles
    assert "Card B R" in titles


def test_parse_skips_out_of_stock() -> None:
    hits = _parse_sell_listings(FIXTURE_PAGE, game_code="ua")
    titles = [h["title"] for h in hits]
    assert "Card C" not in titles


# ── YuyuteiMarketplaceSearchClient tests ──────────────────────────────────────


def _client_with_html(html: str) -> YuyuteiMarketplaceSearchClient:
    mock_http = MagicMock()
    mock_http.get_text.return_value = html
    return YuyuteiMarketplaceSearchClient(http_client=mock_http, game_codes=("ua",))


def test_search_returns_marketplace_listings() -> None:
    client = _client_with_html(FIXTURE_PAGE)
    results = client.search("チェンソーマン", price_max=10000)
    assert len(results) == 2
    assert results[0].title == "Card A SR"
    assert results[0].price_jpy == 500
    assert results[0].url.endswith("/card/ua/v1/001")


def test_search_filters_by_price_max() -> None:
    client = _client_with_html(FIXTURE_PAGE)
    results = client.search("test", price_max=800)
    assert len(results) == 1
    assert results[0].price_jpy == 500


def test_search_skips_out_of_stock() -> None:
    client = _client_with_html(FIXTURE_PAGE)
    results = client.search("test", price_max=99999)
    titles = [r.title for r in results]
    assert "Card C" not in titles


def test_source_name_is_yuyutei() -> None:
    client = YuyuteiMarketplaceSearchClient()
    assert client.source_name == "yuyutei"


def test_listing_source_field_is_yuyutei() -> None:
    client = _client_with_html(FIXTURE_PAGE)
    results = client.search("test", price_max=99999)
    assert all(r.source == "yuyutei" for r in results)


def test_search_http_error_skips_that_game_code() -> None:
    mock_http = MagicMock()
    # First code (ua) raises, second (ws) succeeds with one card
    ws_page = _make_page(_make_card(title="WS Card", price="¥800", href="/card/ws/v1/999"))
    mock_http.get_text.side_effect = [RuntimeError("timeout"), ws_page]

    client = YuyuteiMarketplaceSearchClient(http_client=mock_http, game_codes=("ua", "ws"))
    results = client.search("test", price_max=99999)
    assert len(results) == 1
    assert results[0].title == "WS Card"


def test_search_respects_game_code_override() -> None:
    mock_http = MagicMock()
    mock_http.get_text.return_value = FIXTURE_PAGE
    client = YuyuteiMarketplaceSearchClient(http_client=mock_http, game_codes=("ua", "ws", "ygo", "op"))

    client.search("test", price_max=99999, source_options={"game_code": "ygo"})
    assert mock_http.get_text.call_count == 1
    called_url = mock_http.get_text.call_args[0][0]
    assert "/sell/ygo/" in called_url


def test_search_deduplicates_across_game_codes() -> None:
    # Same item_id (/card/ua/v1/001) appears in both "ua" and "ws" responses
    shared_card = _make_card(title="Duplicate Card", price="¥400", href="/card/ua/v1/001")
    page = _make_page(shared_card)
    mock_http = MagicMock()
    mock_http.get_text.return_value = page

    client = YuyuteiMarketplaceSearchClient(http_client=mock_http, game_codes=("ua", "ws"))
    results = client.search("test", price_max=99999)
    item_ids = [r.item_id for r in results]
    assert len(item_ids) == len(set(item_ids))
    assert len(results) == 1
