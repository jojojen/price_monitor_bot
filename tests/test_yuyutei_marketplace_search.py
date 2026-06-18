"""Tests for YuyuteiMarketplaceSearchClient and _parse_sell_listings."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from market_monitor.yuyutei_search import (
    YuyuteiMarketplaceSearchClient,
    YuyuteiReferenceBand,
    _parse_buy_listings,
    _parse_sell_listings,
    resolve_sell_code,
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


@pytest.mark.parametrize(
    "query,expected",
    [
        ("ポケモン ピカチュウex SAR", "poc"),
        ("遊戯王 青眼の白龍", "ygo"),
        ("ヴァイス 初音ミク", "ws"),
        ("ユニオンアリーナ チェンソーマン", "ua"),
        ("ワンピース ルフィ", "op"),
        ("poc", "poc"),
        ("ygo", "ygo"),
        ("ピカチュウex SAR PSA10", None),
        ("初音ミク フィギュア", None),
        ("", None),
    ],
)
def test_resolve_sell_code(query: str, expected: str | None) -> None:
    assert resolve_sell_code(query) == expected


def test_auto_mode_routes_to_single_code_when_game_word_present() -> None:
    mock_http = MagicMock()
    mock_http.get_text.return_value = FIXTURE_PAGE
    client = YuyuteiMarketplaceSearchClient(http_client=mock_http)  # game_codes=None → auto
    client.search("遊戯王 ブルーアイズ", price_max=99999)
    assert mock_http.get_text.call_count == 1
    assert "/sell/ygo/" in mock_http.get_text.call_args[0][0]


def test_auto_mode_skips_when_no_game_identifiable() -> None:
    mock_http = MagicMock()
    client = YuyuteiMarketplaceSearchClient(http_client=mock_http)  # auto
    results = client.search("ピカチュウex SAR PSA10", price_max=99999)
    assert results == []
    assert mock_http.get_text.call_count == 0


def test_search_uses_fail_fast_http_options() -> None:
    mock_http = MagicMock()
    mock_http.get_text.return_value = FIXTURE_PAGE
    client = YuyuteiMarketplaceSearchClient(http_client=mock_http, game_codes=("ua",))
    client.search("test", price_max=99999)
    _, kwargs = mock_http.get_text.call_args
    assert kwargs.get("retries") == 1
    assert kwargs.get("curl_fallback") is False


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


# ── stock parsing (real yuyutei markup: 在庫 : ×  /  在庫 : N 点) ──────────────

def _make_card_with_stock(*, title: str, price: str, href: str, zaiko_text: str) -> str:
    return f"""
    <div class="card-product">
      <label class="cart_sell_zaiko">在庫 : {zaiko_text}</label>
      <a href="{href}"><h4>{title}</h4></a>
      <strong>{price}</strong>
    </div>
    """


def test_parse_sell_skips_x_out_of_stock_and_reads_count() -> None:
    page = _make_page(
        _make_card_with_stock(title="OOS", price="¥14,800", href="/card/poc/sv08/1", zaiko_text="×")
        + _make_card_with_stock(title="InStock", price="¥9,800", href="/card/poc/sv08/2", zaiko_text="3 点")
    )
    hits = _parse_sell_listings(page, game_code="poc")
    titles = [h["title"] for h in hits]
    assert "OOS" not in titles            # 在庫 : × must be dropped
    assert "InStock" in titles
    in_stock = next(h for h in hits if h["title"] == "InStock")
    assert in_stock["stock_count"] == 3


def test_parse_buy_keeps_all_cards_no_stock_gate() -> None:
    page = _make_page(
        _make_card(title="Buy A", price="¥10,000", href="/card/poc/sv08/1")
        + _make_card(title="Buy B", price="¥14,000", href="/card/poc/sv08/2")
    )
    hits = _parse_buy_listings(page)
    assert {h["title"] for h in hits} == {"Buy A", "Buy B"}


def test_reference_band_combines_buy_and_in_stock_sell() -> None:
    sell_page = _make_page(
        _make_card_with_stock(title="Sell A", price="¥9,800", href="/card/poc/sv08/2", zaiko_text="3 点")
        + _make_card_with_stock(title="Sell OOS", price="¥59,800", href="/card/poc/sv08/9", zaiko_text="×")
    )
    buy_page = _make_page(
        _make_card(title="Buy A", price="¥6,000", href="/card/poc/sv08/2")
        + _make_card(title="Buy B", price="¥8,000", href="/card/poc/sv08/3")
    )
    mock_http = MagicMock()
    mock_http.get_text.side_effect = [sell_page, buy_page]

    client = YuyuteiMarketplaceSearchClient(http_client=mock_http)
    band = client.reference_band("test card", price_max=99999, source_options={"game_code": "poc"})

    assert isinstance(band, YuyuteiReferenceBand)
    assert band.sell_prices == (9800,)            # OOS sell excluded
    assert set(band.buy_prices) == {6000, 8000}
    assert band.sell_stock_total == 3
    assert band.buy_reference == 7000             # median(6000, 8000)
    assert band.sell_reference == 9800
    # two requests: one /sell/, one /buy/, both fail-fast
    assert mock_http.get_text.call_count == 2
    for _, kwargs in mock_http.get_text.call_args_list:
        assert kwargs.get("retries") == 1
        assert kwargs.get("curl_fallback") is False


def test_reference_band_carries_matched_titles() -> None:
    sell_page = _make_page(
        _make_card_with_stock(title="リザードンex SAR 200/165", price="¥9,800", href="/card/poc/sv08/2", zaiko_text="3 点")
        + _make_card_with_stock(title="Sell OOS", price="¥59,800", href="/card/poc/sv08/9", zaiko_text="×")
    )
    buy_page = _make_page(
        _make_card(title="買取タイトル B", price="¥6,000", href="/card/poc/sv08/3")
    )
    mock_http = MagicMock()
    mock_http.get_text.side_effect = [sell_page, buy_page]

    client = YuyuteiMarketplaceSearchClient(http_client=mock_http)
    band = client.reference_band("test card", price_max=99999, source_options={"game_code": "poc"})

    assert band is not None
    # Verbatim titles carried for cache enrichment; in-stock sell first, OOS excluded.
    assert band.sample_titles == ("リザードンex SAR 200/165", "買取タイトル B")


def test_reference_band_exposes_min_max_ranges() -> None:
    sell_page = _make_page(
        _make_card_with_stock(title="Sell Lo", price="¥75,000", href="/card/poc/sv08/1", zaiko_text="3 点")
        + _make_card_with_stock(title="Sell Hi", price="¥85,000", href="/card/poc/sv08/2", zaiko_text="1 点")
    )
    buy_page = _make_page(
        _make_card(title="Buy Lo", price="¥50,000", href="/card/poc/sv08/1")
        + _make_card(title="Buy Hi", price="¥64,000", href="/card/poc/sv08/3")
    )
    mock_http = MagicMock()
    mock_http.get_text.side_effect = [sell_page, buy_page]

    client = YuyuteiMarketplaceSearchClient(http_client=mock_http)
    band = client.reference_band("test card", price_max=99999, source_options={"game_code": "poc"})

    assert band is not None
    assert band.buy_min == 50000 and band.buy_max == 64000
    assert band.sell_min == 75000 and band.sell_max == 85000


def test_reference_band_skips_non_tcg_query_without_network() -> None:
    mock_http = MagicMock()
    client = YuyuteiMarketplaceSearchClient(http_client=mock_http)
    band = client.reference_band("初音ミク フィギュア", price_max=99999)
    assert band is None
    mock_http.get_text.assert_not_called()
