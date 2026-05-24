"""Tests for LlmListingExtractor — all Ollama HTTP calls are mocked."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from market_monitor.llm_listing_extractor import LlmListingExtractor, _html_to_text
from market_monitor.official_store_base import STATUS_UNKNOWN


# ── Helper to build a fake Ollama response ───────────────────────────────────

def _make_ollama_response(listings_payload: object) -> bytes:
    """Return bytes that look like an Ollama /api/chat response."""
    content = json.dumps({"listings": listings_payload})
    response = {
        "model": "qwen3:4b",
        "message": {"role": "assistant", "content": content},
        "done": True,
    }
    return json.dumps(response).encode("utf-8")


def _mock_urlopen(response_bytes: bytes):
    """Return a context-manager mock whose .read() returns *response_bytes*."""
    cm = MagicMock()
    cm.__enter__ = MagicMock(return_value=cm)
    cm.__exit__ = MagicMock(return_value=False)
    cm.read = MagicMock(return_value=response_bytes)
    return cm


# ── test_extract_parses_valid_json_response ───────────────────────────────────

def test_extract_parses_valid_json_response():
    """A valid JSON array with one listing is parsed into an OfficialStoreListing."""
    listing_data = [
        {
            "title": "ポケモンカード スカーレット＆バイオレット ブースターパック",
            "url": "https://example.co.jp/products/poke-sv-booster",
            "status": "lottery_open",
            "price_jpy": 4180,
            "deadline_iso": "2026-06-15",
            "open_date_iso": "2026-06-01",
        }
    ]
    resp_bytes = _make_ollama_response(listing_data)

    extractor = LlmListingExtractor()
    with patch("urllib.request.urlopen", return_value=_mock_urlopen(resp_bytes)):
        results = extractor.extract(
            "<html><body>ポケモンカード 抽選受付中</body></html>",
            store_name="teststore",
            base_url="https://example.co.jp/lottery/",
        )

    assert len(results) == 1
    r = results[0]
    assert r.title == "ポケモンカード スカーレット＆バイオレット ブースターパック"
    assert r.url == "https://example.co.jp/products/poke-sv-booster"
    assert r.status == "lottery_open"
    assert r.price_jpy == 4180
    assert r.deadline_iso == "2026-06-15"
    assert r.open_date_iso == "2026-06-01"
    assert r.store_name == "teststore"


# ── test_extract_returns_empty_on_invalid_json ────────────────────────────────

def test_extract_returns_empty_on_invalid_json():
    """When the LLM returns {\"listings\": \"bad\"} (not a list), return []."""
    # "listings" is a string, not a list — should trigger the warning path
    content = json.dumps({"listings": "bad"})
    response = {
        "model": "qwen3:4b",
        "message": {"role": "assistant", "content": content},
        "done": True,
    }
    resp_bytes = json.dumps(response).encode("utf-8")

    extractor = LlmListingExtractor()
    with patch("urllib.request.urlopen", return_value=_mock_urlopen(resp_bytes)):
        results = extractor.extract(
            "<html><body>some content</body></html>",
            store_name="teststore",
            base_url="https://example.co.jp/",
        )

    assert results == []


# ── test_extract_returns_empty_on_http_error ──────────────────────────────────

def test_extract_returns_empty_on_http_error():
    """When urlopen raises an exception, extract() returns [] gracefully."""
    extractor = LlmListingExtractor()
    with patch("urllib.request.urlopen", side_effect=OSError("connection refused")):
        results = extractor.extract(
            "<html><body>some content</body></html>",
            store_name="teststore",
            base_url="https://example.co.jp/",
        )

    assert results == []


# ── test_html_to_text_strips_scripts_and_styles ───────────────────────────────

def test_html_to_text_strips_scripts_and_styles():
    """Script and style block content must not appear in the output."""
    html = """
    <html>
    <head>
      <style>.foo { color: red; } body { margin: 0; }</style>
      <script>var x = 1; alert('hello');</script>
    </head>
    <body>
      <p>ポケモンカード 抽選受付中</p>
    </body>
    </html>
    """
    result = _html_to_text(html)
    assert "color: red" not in result
    assert "var x = 1" not in result
    assert "alert" not in result
    assert "ポケモンカード" in result
    assert "抽選受付中" in result


# ── test_html_to_text_preserves_link_text ────────────────────────────────────

def test_html_to_text_preserves_link_text():
    """Anchor inner text must be preserved; href should also appear."""
    html = '<a href="/products/poke-sv">ポケモンカード ブースター</a>'
    result = _html_to_text(html)
    assert "ポケモンカード ブースター" in result
    # href should be embedded so the LLM can pick it up
    assert "/products/poke-sv" in result


# ── test_extract_resolves_relative_urls ──────────────────────────────────────

def test_extract_resolves_relative_urls():
    """If the LLM returns a relative URL like \"/path\", it must be resolved against base_url."""
    listing_data = [
        {
            "title": "チェンソーマン UA エクストラブースター",
            "url": "/products/chainsaw-ua",
            "status": "preorder_open",
            "price_jpy": 3850,
            "deadline_iso": None,
            "open_date_iso": None,
        }
    ]
    resp_bytes = _make_ollama_response(listing_data)

    extractor = LlmListingExtractor()
    with patch("urllib.request.urlopen", return_value=_mock_urlopen(resp_bytes)):
        results = extractor.extract(
            "<html><body>予約受付中</body></html>",
            store_name="animate",
            base_url="https://www.animate.co.jp/special/lottery/",
        )

    assert len(results) == 1
    assert results[0].url == "https://www.animate.co.jp/products/chainsaw-ua"


# ── Additional edge-case tests ────────────────────────────────────────────────

def test_extract_uses_unknown_status_for_unrecognised_value():
    """A status value not in the vocabulary should be normalised to 'unknown'."""
    listing_data = [
        {
            "title": "遊戯王 ブースター",
            "url": "https://example.co.jp/yugioh",
            "status": "weird_status_not_in_vocab",
            "price_jpy": None,
            "deadline_iso": None,
            "open_date_iso": None,
        }
    ]
    resp_bytes = _make_ollama_response(listing_data)

    extractor = LlmListingExtractor()
    with patch("urllib.request.urlopen", return_value=_mock_urlopen(resp_bytes)):
        results = extractor.extract(
            "<html><body>遊戯王</body></html>",
            store_name="teststore",
            base_url="https://example.co.jp/",
        )

    assert len(results) == 1
    assert results[0].status == STATUS_UNKNOWN


def test_extract_skips_items_without_title():
    """Items with no title (empty or missing) must be dropped."""
    listing_data = [
        {"title": "", "url": "/a", "status": "available", "price_jpy": None,
         "deadline_iso": None, "open_date_iso": None},
        {"url": "/b", "status": "available", "price_jpy": None,
         "deadline_iso": None, "open_date_iso": None},
        {"title": "ワンピース カードゲーム", "url": "/c", "status": "available",
         "price_jpy": 5500, "deadline_iso": None, "open_date_iso": None},
    ]
    resp_bytes = _make_ollama_response(listing_data)

    extractor = LlmListingExtractor()
    with patch("urllib.request.urlopen", return_value=_mock_urlopen(resp_bytes)):
        results = extractor.extract(
            "<html><body>ワンピース</body></html>",
            store_name="teststore",
            base_url="https://example.co.jp/",
        )

    assert len(results) == 1
    assert results[0].title == "ワンピース カードゲーム"


def test_html_to_text_collapses_whitespace():
    """Multiple consecutive blank lines should be reduced to one."""
    html = "<p>Line A</p>\n\n\n\n<p>Line B</p>"
    result = _html_to_text(html)
    # Should not have more than one consecutive blank line
    assert "\n\n\n" not in result
    assert "Line A" in result
    assert "Line B" in result
