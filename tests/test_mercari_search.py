from __future__ import annotations

from pathlib import Path

import pytest

from market_monitor import mercari_search
from market_monitor.mercari_search import (
    _chromium_launch_options,
    parse_detail_price,
    parse_search_html,
    search_mercari,
)

FIXTURES = Path(__file__).parent / "fixtures"


def test_chromium_launch_options_use_configured_system_chromium(monkeypatch) -> None:
    monkeypatch.setenv("PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH", "/usr/bin/chromium")

    assert _chromium_launch_options() == {
        "headless": True,
        "executable_path": "/usr/bin/chromium",
    }


def test_chromium_launch_options_fall_back_to_playwright_bundle(monkeypatch) -> None:
    monkeypatch.delenv("PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH", raising=False)
    monkeypatch.setattr("market_monitor.mercari_search.shutil.which", lambda command: None)

    assert _chromium_launch_options() == {"headless": True}


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
