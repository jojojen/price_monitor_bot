"""Unit tests for natural_language.py pure-function layer.

Only tests deterministic helpers — no LLM, no network.
"""
from __future__ import annotations

import pytest

from price_monitor_bot.natural_language import (
    _extract_opportunity_target,
    _extract_sns_schedule_minutes,
    _extract_watch_query,
    _looks_like_opportunity_remove_request,
    _looks_like_web_research_question,
    _normalize_keyword_values,
    _parse_kanji_number,
    _parse_price_threshold,
    _recover_lookup_fields,
    _split_keyword_phrase,
    fallback_route_telegram_natural_language,
)


# ── _parse_price_threshold ────────────────────────────────────────────────────

@pytest.mark.parametrize("text,expected", [
    ("5万", 50000),
    ("50000", 50000),
    ("50,000", 50000),
    ("¥50000", 50000),
    ("30万円以下", 300000),
    ("三十万", 300000),
    ("五万", 50000),
    ("1.5万", 15000),
    ("300000日幣", 300000),
    ("低於2万", 20000),
])
def test_parse_price_threshold(text: str, expected: int) -> None:
    assert _parse_price_threshold(text) == expected


def test_parse_price_threshold_none_on_short_number() -> None:
    # Single digit — not a price
    assert _parse_price_threshold("5") is None


def test_parse_price_threshold_none_on_empty() -> None:
    assert _parse_price_threshold("") is None


# ── _parse_kanji_number ───────────────────────────────────────────────────────

@pytest.mark.parametrize("kanji,expected", [
    ("五", 5),
    ("十", 10),
    ("三十", 30),
    ("百", 100),
    ("百二十", 120),
    ("三", 3),
])
def test_parse_kanji_number(kanji: str, expected: int) -> None:
    assert _parse_kanji_number(kanji) == expected


def test_parse_kanji_number_invalid() -> None:
    assert _parse_kanji_number("abc") is None
    assert _parse_kanji_number("") is None


# ── _extract_watch_query ─────────────────────────────────────────────────────

def test_extract_watch_query_strips_keywords_and_price() -> None:
    query = _extract_watch_query("幫我監控 ピカチュウ 5万以下")
    assert query is not None
    assert "ピカチュウ" in query
    assert "5" not in query
    assert "万" not in query


def test_extract_watch_query_returns_none_on_short_result() -> None:
    # After stripping everything, result too short
    assert _extract_watch_query("監控") is None


# ── _extract_sns_schedule_minutes ────────────────────────────────────────────

@pytest.mark.parametrize("text,expected", [
    ("每 60 分鐘", 60),
    ("每720分鐘", 720),
    ("schedule 30 mins", 30),
    ("每 5 分鐘", 5),
])
def test_extract_sns_schedule_minutes_valid(text: str, expected: int) -> None:
    assert _extract_sns_schedule_minutes(text) == expected


def test_extract_sns_schedule_minutes_out_of_range() -> None:
    assert _extract_sns_schedule_minutes("每 2 分鐘") is None   # < 5
    assert _extract_sns_schedule_minutes("每 2000 分鐘") is None  # > 1440


def test_extract_sns_schedule_minutes_no_match() -> None:
    assert _extract_sns_schedule_minutes("add twitter account @foo") is None


# ── _normalize_keyword_values ────────────────────────────────────────────────

def test_normalize_keyword_values_deduplicates() -> None:
    result = _normalize_keyword_values(["抽選", "抽選", "lottery"])
    assert result.count("抽選") == 1


def test_normalize_keyword_values_strips_quotes() -> None:
    result = _normalize_keyword_values(['"抽選"', "'lottery'"])
    assert "抽選" in result
    assert "lottery" in result


def test_normalize_keyword_values_skips_empty() -> None:
    result = _normalize_keyword_values(["", "  ", "抽選"])
    assert "" not in result
    assert "  " not in result


# ── _split_keyword_phrase ─────────────────────────────────────────────────────

def test_split_keyword_phrase_comma_separated() -> None:
    parts = _split_keyword_phrase("抽選,lottery,ガチャ")
    assert "抽選" in parts
    assert "lottery" in parts
    assert "ガチャ" in parts


def test_split_keyword_phrase_space_separated() -> None:
    parts = _split_keyword_phrase("抽選 lottery")
    assert len(parts) == 2


def test_split_keyword_phrase_empty() -> None:
    assert _split_keyword_phrase("") == []


# ── _looks_like_web_research_question ────────────────────────────────────────

def test_looks_like_web_research_question_yes() -> None:
    assert _looks_like_web_research_question("pokemon SSP 現在多少錢?") is True


def test_looks_like_web_research_question_no_subject() -> None:
    assert _looks_like_web_research_question("你好嗎?") is False


def test_looks_like_web_research_question_no_question_shape() -> None:
    assert _looks_like_web_research_question("鏈鋸人 SSP 買了") is False


# ── _looks_like_opportunity_remove_request ────────────────────────────────────

def test_looks_like_opportunity_remove_request_yes() -> None:
    assert _looks_like_opportunity_remove_request("移除 鏈鋸人 機會清單") is True
    assert _looks_like_opportunity_remove_request("I'm not interested in this anymore") is True


def test_looks_like_opportunity_remove_request_no() -> None:
    assert _looks_like_opportunity_remove_request("監控 ピカチュウ") is False


# ── _extract_opportunity_target ───────────────────────────────────────────────

def test_extract_opportunity_target_strips_noise() -> None:
    result = _extract_opportunity_target("移除 鏈鋸人 從機會清單")
    assert result is not None
    assert "鏈鋸人" in result
    assert "移除" not in result


def test_extract_opportunity_target_numeric_index() -> None:
    result = _extract_opportunity_target("remove 第3個")
    assert result == "3"


# ── _recover_lookup_fields ────────────────────────────────────────────────────

def test_recover_lookup_fields_extracts_rarity_from_name() -> None:
    name, cn, rarity, sc = _recover_lookup_fields("ピカチュウ SSP", None, None, None)
    assert rarity == "SSP"


def test_recover_lookup_fields_no_op_when_already_present() -> None:
    name, cn, rarity, sc = _recover_lookup_fields("ピカチュウ", None, "SSP", None)
    assert rarity == "SSP"
    assert name == "ピカチュウ"


# ── fallback_route_telegram_natural_language ──────────────────────────────────

def test_fallback_returns_none_on_empty() -> None:
    assert fallback_route_telegram_natural_language("") is None


def test_fallback_routes_reputation_snapshot() -> None:
    text = "https://www.mercari.com/jp/items/123456 これの出品者信用できますか"
    intent = fallback_route_telegram_natural_language(text)
    assert intent is not None
    assert intent.intent == "reputation_snapshot"
    assert intent.query_url is not None


def test_fallback_routes_opportunity_remove() -> None:
    text = "移除 鏈鋸人 從目標清單"
    intent = fallback_route_telegram_natural_language(text)
    assert intent is not None
    assert intent.intent == "opportunity_remove"


def test_fallback_routes_watch_add() -> None:
    text = "幫我監控 ピカチュウ SSP 5万以下"
    intent = fallback_route_telegram_natural_language(text)
    assert intent is not None
    assert intent.intent == "add_watch"
    assert intent.watch_price_threshold == 50000
