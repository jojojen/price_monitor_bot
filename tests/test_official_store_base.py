"""Tests for official_store_base: Protocol, helpers, date parsing."""

from __future__ import annotations

from market_monitor.official_store_base import (
    ACTIVE_STATUSES,
    LOTTERY_CLOSED,
    LOTTERY_OPEN,
    PREORDER_CLOSED,
    PREORDER_OPEN,
    COMING_SOON,
    STATUS_UNKNOWN,
    OfficialStoreListing,
    abs_url,
    infer_status,
    item_key_from_url,
    parse_jp_date,
    parse_jp_datetime,
)


# ── OfficialStoreListing ─────────────────────────────────────────────────────


def test_listing_defaults():
    listing = OfficialStoreListing(
        store_name="joshin",
        item_key="joshinweb.jp/tcg/chainsaw-ua",
        title="UNION ARENA チェンソーマン",
        url="https://joshinweb.jp/tcg/chainsaw-ua",
        status=LOTTERY_OPEN,
    )
    assert listing.price_jpy is None
    assert listing.deadline_iso is None
    assert listing.crawled_at != ""  # auto-populated


def test_listing_full():
    listing = OfficialStoreListing(
        store_name="joshin",
        item_key="joshinweb.jp/tcg/chainsaw-ua",
        title="チェンソーマン UA",
        url="https://joshinweb.jp/tcg/chainsaw-ua",
        status=PREORDER_OPEN,
        price_jpy=4180,
        deadline_iso="2026-06-15T23:59:00+09:00",
        open_date_iso="2026-06-01T10:00:00+09:00",
        product_code="UA-EXT-CSM-001",
        categories=("tcg", "union_arena"),
    )
    assert listing.price_jpy == 4180
    assert listing.categories == ("tcg", "union_arena")


# ── ACTIVE_STATUSES ──────────────────────────────────────────────────────────


def test_active_statuses_includes_open_and_available():
    assert LOTTERY_OPEN in ACTIVE_STATUSES
    assert PREORDER_OPEN in ACTIVE_STATUSES
    assert LOTTERY_CLOSED not in ACTIVE_STATUSES
    assert PREORDER_CLOSED not in ACTIVE_STATUSES


# ── parse_jp_date ────────────────────────────────────────────────────────────


def test_parse_jp_date_year_month_day():
    assert parse_jp_date("2026年6月1日") == "2026-06-01"


def test_parse_jp_date_month_day_only_uses_base_year():
    assert parse_jp_date("6月1日", base_year=2026) == "2026-06-01"


def test_parse_jp_date_slash_with_year():
    assert parse_jp_date("2026/06/01") == "2026-06-01"


def test_parse_jp_date_slash_month_day():
    assert parse_jp_date("6/1", base_year=2026) == "2026-06-01"


def test_parse_jp_date_embedded_in_text():
    assert parse_jp_date("抽選申込開始：2026年6月1日 10:00") == "2026-06-01"


def test_parse_jp_date_returns_none_for_garbage():
    assert parse_jp_date("") is None
    assert parse_jp_date("申し込み受付中") is None


# ── parse_jp_datetime ────────────────────────────────────────────────────────


def test_parse_jp_datetime_with_time():
    result = parse_jp_datetime("2026年6月1日 10:00")
    assert result == "2026-06-01T10:00:00+09:00"


def test_parse_jp_datetime_with_ji_notation():
    result = parse_jp_datetime("6月15日 23時59分", base_year=2026)
    assert result == "2026-06-15T23:59:00+09:00"


def test_parse_jp_datetime_no_time_returns_date_only():
    result = parse_jp_datetime("2026年6月1日", base_year=2026)
    assert result == "2026-06-01"


# ── infer_status ─────────────────────────────────────────────────────────────


def test_infer_status_lottery_open():
    assert infer_status("抽選申込受付中") == LOTTERY_OPEN


def test_infer_status_lottery_closed_beats_open():
    # Closed keyword takes precedence when both appear (edge case on pages
    # that show old "受付中" text alongside a "終了" badge).
    assert infer_status("抽選受付終了 当選者発表済") == LOTTERY_CLOSED


def test_infer_status_preorder_open():
    assert infer_status("予約受付中") == PREORDER_OPEN


def test_infer_status_sold_out():
    assert infer_status("完売") == PREORDER_CLOSED


def test_infer_status_coming_soon():
    assert infer_status("近日公開") == COMING_SOON


def test_infer_status_unknown():
    assert infer_status("詳細は後日公開") == STATUS_UNKNOWN


# ── item_key_from_url ────────────────────────────────────────────────────────


def test_item_key_strips_query_and_fragment():
    url = "https://joshinweb.jp/tcg/chainsaw-ua?from=top&ref=banner#section"
    assert item_key_from_url(url) == "joshinweb.jp/tcg/chainsaw-ua"


def test_item_key_stable_across_query_variants():
    u1 = "https://yodobashi.com/product/100000001004966025/?from=search"
    u2 = "https://yodobashi.com/product/100000001004966025/?from=recommend"
    assert item_key_from_url(u1) == item_key_from_url(u2)


# ── abs_url ──────────────────────────────────────────────────────────────────


def test_abs_url_relative():
    assert abs_url("https://joshinweb.jp/tcg/", "/products/123") == "https://joshinweb.jp/products/123"


def test_abs_url_already_absolute():
    full = "https://joshinweb.jp/products/123"
    assert abs_url("https://joshinweb.jp/tcg/", full) == full
