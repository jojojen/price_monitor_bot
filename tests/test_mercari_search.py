from __future__ import annotations

from pathlib import Path

import pytest

from market_monitor import mercari_search
from market_monitor.mercari_search import (
    DEFAULT_CONDITION_IDS,
    build_search_url,
    parse_detail_price,
    parse_search_html,
    search_mercari,
    search_mercari_sold,
)

FIXTURES = Path(__file__).parent / "fixtures"


def test_build_search_url_without_condition_ids_omits_param() -> None:
    url = build_search_url("test", price_max=5000)
    assert "item_condition_id" not in url


def test_build_search_url_with_condition_ids_appends_param_per_id() -> None:
    url = build_search_url("test", price_max=5000, condition_ids=(1, 2, 3))
    assert url.count("item_condition_id=") == 3
    for cid in (1, 2, 3):
        assert f"item_condition_id={cid}" in url


def test_default_condition_ids_is_top_three() -> None:
    assert DEFAULT_CONDITION_IDS == (1, 2, 3)


# ── Search-results parser ─────────────────────────────────────────────────────

def _load(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


def test_parse_search_html_extracts_item_cards_with_real_prices() -> None:
    items = parse_search_html(_load("mercari_search_eva_rei.html"), max_results=50)

    # Fixture is a live capture of jp.mercari.com results for
    # "綾波レイ ユニオンアリーナ プロモカード" price_max=4500.
    assert len(items) >= 3, items
    by_id = {item["item_id"]: item for item in items}

    # All three known items must be present with their actual list prices,
    # NOT the installment-estimate values that bit us in production.
    assert by_id["m67507032609"]["price_jpy"] == 2200
    assert by_id["m13000897914"]["price_jpy"] == 999
    assert by_id["m81947702369"]["price_jpy"] == 2888

    for item in items:
        assert item["price_jpy"] > 0
        # Reject any value that looks like a monthly-installment estimate
        # (Mercari's bug was ¥1,382 / ¥1,749 on ¥6,555 / ¥8,300 items).
        assert item["price_jpy"] >= 500, f"suspiciously low: {item}"
        assert item["url"].startswith("https://jp.mercari.com/item/m")
        assert item["title"], item


# ── Detail-page parser ────────────────────────────────────────────────────────

def test_parse_detail_price_known_items() -> None:
    # These two fixtures are the items the user got bad notifications for.
    # Their authoritative prices live in <meta name="product:price:amount">.
    assert parse_detail_price(_load("mercari_item_m18542743389.html")) == 6555
    assert parse_detail_price(_load("mercari_item_m85537287496.html")) == 8300


def test_parse_detail_price_missing_meta_returns_none() -> None:
    assert parse_detail_price("<html><head></head><body></body></html>") is None


# ── End-to-end: search + verify (no network) ──────────────────────────────────

def test_search_mercari_drops_candidates_whose_detail_price_exceeds_max(monkeypatch) -> None:
    """If a card slips through with a low search-page price but the detail
    page reports a price above price_max, the item must be dropped."""
    fake_html = _load("mercari_search_eva_rei.html")

    class _FakePage:
        def goto(self, *_a, **_k): ...
        def wait_for_selector(self, *_a, **_k): ...
        def evaluate(self, *_a, **_k): ...
        def wait_for_timeout(self, *_a, **_k): ...
        def content(self) -> str:
            return fake_html

    class _FakeContext:
        def new_page(self): return _FakePage()
        def add_init_script(self, *_a, **_k): ...
        def close(self): ...

    class _FakeBrowser:
        def new_context(self, **_k): return _FakeContext()
        def close(self): ...

    class _FakeChromium:
        def launch(self, **_k): return _FakeBrowser()

    class _FakePW:
        chromium = _FakeChromium()
        def __enter__(self): return self
        def __exit__(self, *_): return False

    monkeypatch.setattr(
        "playwright.sync_api.sync_playwright",
        lambda: _FakePW(),
    )

    # Detail-price stub: pretend m67507032609 is really ¥9,000 (above price_max
    # of 4500), the other two stay under the cap.
    detail_prices = {
        "m67507032609": 9000,
        "m13000897914": 999,
        "m81947702369": 2888,
    }
    monkeypatch.setattr(
        mercari_search,
        "fetch_item_detail_price",
        lambda item_id, **_k: detail_prices.get(item_id),
    )

    # Use "デッキ" — a token shared by all three card titles in the fixture so
    # the query filter doesn't pre-empt the detail-page verification step.
    results = search_mercari("デッキ", price_max=4500)
    ids = {r["item_id"] for r in results}
    assert "m67507032609" not in ids, "item with detail price > price_max should be dropped"
    assert "m13000897914" in ids
    assert "m81947702369" in ids


def test_search_mercari_replaces_search_price_with_detail_price(monkeypatch) -> None:
    """When search-page price differs from detail-page price, detail wins."""
    fake_html = _load("mercari_search_eva_rei.html")

    class _FakePage:
        def goto(self, *_a, **_k): ...
        def wait_for_selector(self, *_a, **_k): ...
        def evaluate(self, *_a, **_k): ...
        def wait_for_timeout(self, *_a, **_k): ...
        def content(self) -> str: return fake_html

    class _FakeContext:
        def new_page(self): return _FakePage()
        def add_init_script(self, *_a, **_k): ...
        def close(self): ...

    class _FakeBrowser:
        def new_context(self, **_k): return _FakeContext()
        def close(self): ...

    class _FakeChromium:
        def launch(self, **_k): return _FakeBrowser()

    class _FakePW:
        chromium = _FakeChromium()
        def __enter__(self): return self
        def __exit__(self, *_): return False

    monkeypatch.setattr("playwright.sync_api.sync_playwright", lambda: _FakePW())
    # Detail reports a different (still-valid) price for one item.
    monkeypatch.setattr(
        mercari_search,
        "fetch_item_detail_price",
        lambda item_id, **_k: {"m13000897914": 1234}.get(item_id, 0) or None,
    )

    results = search_mercari("デッキ", price_max=4500)
    matched = [r for r in results if r["item_id"] == "m13000897914"]
    assert matched and matched[0]["price_jpy"] == 1234


def test_search_mercari_sold_returns_verified_items(monkeypatch) -> None:
    fake_html = _load("mercari_search_eva_rei.html")

    class _FakePage:
        def goto(self, *_a, **_k): ...
        def wait_for_selector(self, *_a, **_k): ...
        def evaluate(self, *_a, **_k): ...
        def wait_for_timeout(self, *_a, **_k): ...
        def content(self) -> str:
            return fake_html

    class _FakeContext:
        def new_page(self): return _FakePage()
        def add_init_script(self, *_a, **_k): ...
        def close(self): ...

    class _FakeBrowser:
        def new_context(self, **_k): return _FakeContext()
        def close(self): ...

    class _FakeChromium:
        def launch(self, **_k): return _FakeBrowser()

    class _FakePW:
        chromium = _FakeChromium()
        def __enter__(self): return self
        def __exit__(self, *_): return False

    monkeypatch.setattr("playwright.sync_api.sync_playwright", lambda: _FakePW())
    monkeypatch.setattr(
        mercari_search,
        "fetch_item_detail_price",
        lambda item_id, **_k: {"m67507032609": 2200, "m13000897914": 999, "m81947702369": 2888}.get(item_id),
    )

    results = search_mercari_sold("デッキ", max_results=2)

    assert [item["item_id"] for item in results] == ["m67507032609", "m13000897914"]
    assert results[0]["price_jpy"] == 2200
    assert results[1]["url"].startswith("https://jp.mercari.com/item/")


@pytest.fixture
def _reset_title_matcher():
    """Ensure module-level matcher is cleared after each test using it."""
    mercari_search.set_title_matcher(None)
    try:
        yield
    finally:
        mercari_search.set_title_matcher(None)


def test_filter_by_query_delegates_to_registered_matcher(_reset_title_matcher) -> None:
    items = [{"item_id": "a", "title": "foo"}, {"item_id": "b", "title": "bar"}]
    calls: list[tuple[str, list]] = []

    def fake_matcher(query, batch):
        calls.append((query, batch))
        return [batch[1]]  # keep only the second, ignoring lexical tokens

    mercari_search.set_title_matcher(fake_matcher)

    result = mercari_search._filter_by_query(items, "nonmatching query tokens")

    assert result == [items[1]]
    assert calls == [("nonmatching query tokens", items)]


def test_filter_by_query_falls_back_to_lexical_when_matcher_raises(_reset_title_matcher) -> None:
    items = [
        {"item_id": "a", "title": "world card SP"},
        {"item_id": "b", "title": "unrelated"},
    ]

    def boom(query, batch):
        raise RuntimeError("embedder exploded")

    mercari_search.set_title_matcher(boom)

    result = mercari_search._filter_by_query(items, "world SP")

    # Lexical AND-substring filter keeps only the item containing every token.
    assert result == [items[0]]
