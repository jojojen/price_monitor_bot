"""Tests for the /watch fair-value verdict (公允價評語) — pure functions only.

These cover the two pure helpers added for the personal-use enhancement:
``_fair_value_verdict`` (price-vs-sold-average classification) and
``_format_notification`` (whether the verdict line is woven into the alert).
No Playwright / network — sold-price averages are passed in directly.
"""

from __future__ import annotations

from market_monitor.storage import MarketplaceWatch
from price_monitor_bot.watch_monitor import (
    _fair_value_verdict,
    _format_notification,
)


def _watch(*, query: str = "テスト", price_threshold_jpy: int = 10_000) -> MarketplaceWatch:
    return MarketplaceWatch(
        watch_id="w1",
        query=query,
        price_threshold_jpy=price_threshold_jpy,
        markets=("mercari",),
        enabled=True,
        chat_id="123",
        last_checked_at=None,
        created_at="2026-05-29T00:00:00+00:00",
        updated_at="2026-05-29T00:00:00+00:00",
    )


def _item(*, price: int, title: str = "商品", event: str = "new") -> dict:
    return {
        "price_jpy": price,
        "title": title,
        "url": "https://example.com/i",
        "_event": event,
    }


# ── _fair_value_verdict boundaries ───────────────────────────────────────────


def test_verdict_cheap_at_threshold() -> None:
    # ratio exactly 0.85 → 划算 (≤ cheap ratio)
    out = _fair_value_verdict(price_jpy=8_500, avg_jpy=10_000.0)
    assert "划算" in out
    assert "15%" in out  # 1 - 0.85 = 15%


def test_verdict_pricey_at_threshold() -> None:
    # ratio exactly 1.10 → 偏貴 (≥ pricey ratio)
    out = _fair_value_verdict(price_jpy=11_000, avg_jpy=10_000.0)
    assert "偏貴" in out
    assert "10%" in out


def test_verdict_reasonable_in_between() -> None:
    # ratio 1.0 → 合理
    out = _fair_value_verdict(price_jpy=10_000, avg_jpy=10_000.0)
    assert "合理" in out


def test_verdict_just_below_pricey_is_reasonable() -> None:
    # ratio 1.09 → still 合理 (strictly below 1.10)
    out = _fair_value_verdict(price_jpy=10_900, avg_jpy=10_000.0)
    assert "合理" in out


def test_verdict_empty_when_avg_unusable() -> None:
    assert _fair_value_verdict(price_jpy=10_000, avg_jpy=0.0) == ""
    assert _fair_value_verdict(price_jpy=10_000, avg_jpy=-5.0) == ""


# ── _format_notification with / without fair value ───────────────────────────


def test_notification_without_fair_value_has_no_verdict() -> None:
    text = _format_notification(
        watch=_watch(),
        market="mercari",
        new_or_changed=[_item(price=9_000)],
        fair_value_avg=None,
    )
    assert "划算" not in text
    assert "合理" not in text
    assert "偏貴" not in text
    assert "二手均價" not in text


def test_notification_with_fair_value_includes_verdict() -> None:
    text = _format_notification(
        watch=_watch(),
        market="mercari",
        new_or_changed=[_item(price=8_000)],
        fair_value_avg=10_000.0,
    )
    assert "划算" in text
    assert "二手均價 ¥10,000" in text


def test_notification_verdict_per_item() -> None:
    text = _format_notification(
        watch=_watch(),
        market="mercari",
        new_or_changed=[
            _item(price=8_000, title="便宜的"),
            _item(price=12_000, title="貴的"),
        ],
        fair_value_avg=10_000.0,
    )
    assert "划算" in text
    assert "偏貴" in text
