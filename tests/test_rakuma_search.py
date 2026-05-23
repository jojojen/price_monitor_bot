"""Tests for the Rakuma (fril.jp) HTML parser.

The parser is pure (HTML in → MarketplaceListing list out) so we drive it with
canned HTML fixtures — no network. The network-facing ``search_rakuma`` is
covered by an integration test elsewhere (skipped by default)."""

from __future__ import annotations

from market_monitor.marketplace_search import MarketplaceListing
from market_monitor.rakuma_search import (
    RakumaSearchClient,
    _extract_item_id,
    _parse_price,
    parse_rakuma_listings,
)


def test_extract_item_id_from_relative_href() -> None:
    assert _extract_item_id("/item/12345678") == "12345678"
    assert _extract_item_id("/some/slug/item/99887766/xx") == "99887766"
    assert _extract_item_id("/profile/abc") is None
    assert _extract_item_id("") is None


def test_parse_price_strips_yen_and_commas() -> None:
    assert _parse_price("¥4,622") == 4622
    assert _parse_price("4622 円") == 4622
    assert _parse_price("4,622") == 4622
    assert _parse_price("not a price") is None
    assert _parse_price("") is None


def _fixture_html(items: list[tuple[str, str, int]]) -> str:
    """Build an HTML blob that mimics the structure parse_rakuma_listings
    expects: each ``<a href="/item/...">`` has a title in its text and a
    sibling-or-child price element."""
    rows = []
    for item_id, title, price in items:
        rows.append(
            f"""
            <div class="item-box">
              <a href="/item/{item_id}">
                <img data-src="//static.fril.jp/thumb/{item_id}.jpg" alt="{title}"/>
                <span class="title">{title}</span>
              </a>
              <p class="item-box__item-price">¥{price:,}</p>
            </div>
            """
        )
    return "<html><body>" + "\n".join(rows) + "</body></html>"


def test_parser_returns_one_listing_per_item_link() -> None:
    html = _fixture_html([
        ("10001", "ピカチュウex SAR", 12000),
        ("10002", "リザードンex SAR", 25000),
    ])
    out = parse_rakuma_listings(html, query="any", price_max=100_000)
    assert len(out) == 2
    assert all(isinstance(li, MarketplaceListing) for li in out)
    assert {li.item_id for li in out} == {"10001", "10002"}
    assert all(li.source == "rakuma" for li in out)


def test_parser_drops_items_above_price_max() -> None:
    html = _fixture_html([
        ("10001", "cheap", 1000),
        ("10002", "expensive", 50_000),
    ])
    out = parse_rakuma_listings(html, query="any", price_max=10_000)
    assert {li.item_id for li in out} == {"10001"}


def test_extract_item_id_only_matches_numeric_ids() -> None:
    """Rakuma item IDs are always numeric — non-numeric paths must not match."""
    assert _extract_item_id("/item/X1") is None
    assert _extract_item_id("/item/abc") is None


def test_parser_dedupes_by_item_id() -> None:
    """Same /item/<id> appearing twice (image link + title link) → one listing."""
    html = """
    <html><body>
      <div class="item-box">
        <a href="/item/11111"><img alt="abc"/></a>
        <a href="/item/11111">abc</a>
        <p class="item-box__item-price">¥500</p>
      </div>
    </body></html>
    """
    out = parse_rakuma_listings(html, query="any", price_max=10000)
    assert len(out) == 1
    assert out[0].item_id == "11111"


def test_parser_skips_listings_with_no_extractable_price() -> None:
    html = """
    <html><body>
      <a href="/item/22222"><img alt="no-price"/></a>
      <!-- no price element anywhere -->
    </body></html>
    """
    out = parse_rakuma_listings(html, query="any", price_max=10000)
    assert out == []


def test_parser_promotes_protocol_relative_thumbnail() -> None:
    html = _fixture_html([("33333", "Z", 100)])
    out = parse_rakuma_listings(html, query="any", price_max=10000)
    assert len(out) == 1
    assert out[0].thumbnail_url is not None
    assert out[0].thumbnail_url.startswith("https://"), out[0].thumbnail_url


def test_rakuma_search_client_returns_empty_on_failed_fetch(monkeypatch) -> None:
    """When the network helper returns None, the client returns []."""
    from market_monitor import rakuma_search as rs
    monkeypatch.setattr(rs, "_fetch_html", lambda url, *, timeout_seconds: None)
    client = RakumaSearchClient()
    assert client.search("any", price_max=10000) == []
