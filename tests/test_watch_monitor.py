"""Tests for the /watch fair-value verdict (公允價評語) — pure functions only.

These cover the two pure helpers added for the personal-use enhancement:
``_fair_value_verdict`` (price-vs-sold-average classification) and
``_format_notification`` (whether the verdict line is woven into the alert).
No Playwright / network — sold-price averages are passed in directly.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from market_monitor.storage import MarketplaceWatch
from price_monitor_bot.watch_monitor import (
    MarketplaceWatchMonitor,
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


# ── _throttled per-source cadence ─────────────────────────────────────────────


def _monitor(tmp_path, **kwargs) -> MarketplaceWatchMonitor:
    return MarketplaceWatchMonitor(
        db_path=tmp_path / "watch.sqlite3",
        clients={},
        notify_fn=MagicMock(),
        **kwargs,
    )


def test_throttle_defaults_yuyutei_to_daily(tmp_path) -> None:
    mon = _monitor(tmp_path)
    # First poll attempt records the time and is allowed through.
    assert mon._throttled("w1", "yuyutei") is False
    # A second attempt in the same tick is within 24h → skipped.
    assert mon._throttled("w1", "yuyutei") is True


def test_throttle_never_blocks_c2c_sources(tmp_path) -> None:
    mon = _monitor(tmp_path)
    for _ in range(5):
        assert mon._throttled("w1", "mercari") is False
        assert mon._throttled("w1", "rakuma") is False


def test_throttle_is_per_watch_and_market(tmp_path) -> None:
    mon = _monitor(tmp_path)
    assert mon._throttled("w1", "yuyutei") is False
    # Different watch → independent clock, still allowed.
    assert mon._throttled("w2", "yuyutei") is False
    # Same (watch, market) again → throttled.
    assert mon._throttled("w1", "yuyutei") is True


def test_throttle_records_attempt_even_on_failure_path(tmp_path) -> None:
    # The time is stored on the attempt (not on success), so a 429/timeout
    # still defers the next try — never amplify a rate-limited host.
    mon = _monitor(tmp_path)
    assert mon._throttled("w1", "yuyutei") is False
    assert ("w1", "yuyutei") in mon._last_polled


def test_throttle_override_disables_with_zero(tmp_path) -> None:
    mon = _monitor(tmp_path, source_min_interval_seconds={})
    for _ in range(3):
        assert mon._throttled("w1", "yuyutei") is False
