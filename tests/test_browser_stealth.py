"""Unit tests for the shared human-like browser fingerprint module.

These cover the pure logic (headers, context kwargs, channel resolution) plus the
real-Chrome→bundled-Chromium launch fallback, using a fake playwright so the
tests never start a browser.
"""
from __future__ import annotations

import pytest

from market_monitor import browser_stealth as bs


def test_http_headers_wear_the_mac_chrome_identity() -> None:
    headers = bs.http_headers()
    assert "Macintosh" in headers["User-Agent"]
    assert headers["Accept-Language"].startswith("ja-JP")
    assert "text/html" in headers["Accept"]


def test_http_headers_merge_extra() -> None:
    headers = bs.http_headers({"Referer": "https://jp.mercari.com/"})
    assert headers["Referer"] == "https://jp.mercari.com/"
    assert "User-Agent" in headers  # base still present


def test_stealth_context_kwargs_defaults_and_override() -> None:
    kwargs = bs.stealth_context_kwargs()
    assert "Macintosh" in kwargs["user_agent"]
    assert kwargs["timezone_id"] == "Asia/Tokyo"
    assert kwargs["has_touch"] is False
    assert kwargs["viewport"]["height"] < 1200
    # Callers can override any field (e.g. a scraper that wants a taller canvas).
    overridden = bs.stealth_context_kwargs(viewport={"width": 1280, "height": 2000})
    assert overridden["viewport"] == {"width": 1280, "height": 2000}


def test_stealth_init_script_spoofs_key_tells() -> None:
    for token in ("webdriver", "MacIntel", "languages", "window.chrome"):
        assert token in bs.STEALTH_INIT_SCRIPT


def test_resolve_browser_channel_honors_env(monkeypatch) -> None:
    monkeypatch.setenv("OPENCLAW_BROWSER_CHANNEL", "chrome")
    assert bs.resolve_browser_channel() == "chrome"
    monkeypatch.setenv("OPENCLAW_BROWSER_CHANNEL", "")  # explicit force bundled
    assert bs.resolve_browser_channel() is None


class _FakeChromium:
    def __init__(self, *, fail_channel: bool) -> None:
        self.fail_channel = fail_channel
        self.calls: list[dict] = []

    def launch(self, **kwargs):
        self.calls.append(kwargs)
        if "channel" in kwargs and self.fail_channel:
            raise RuntimeError("chrome not available")
        return f"browser(channel={kwargs.get('channel')})"


class _FakePlaywright:
    def __init__(self, *, fail_channel: bool) -> None:
        self.chromium = _FakeChromium(fail_channel=fail_channel)


def test_launch_uses_chrome_channel_when_available(monkeypatch) -> None:
    monkeypatch.setenv("OPENCLAW_BROWSER_CHANNEL", "chrome")
    monkeypatch.delenv("PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH", raising=False)
    pw = _FakePlaywright(fail_channel=False)
    browser = bs.launch_stealth_chromium(pw)
    assert browser == "browser(channel=chrome)"
    assert pw.chromium.calls[0]["channel"] == "chrome"


def test_launch_falls_back_to_bundled_when_channel_fails(monkeypatch) -> None:
    monkeypatch.setenv("OPENCLAW_BROWSER_CHANNEL", "chrome")
    monkeypatch.delenv("PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH", raising=False)
    pw = _FakePlaywright(fail_channel=True)
    browser = bs.launch_stealth_chromium(pw)
    # First attempt with channel raises, second (no channel) succeeds.
    assert browser == "browser(channel=None)"
    assert pw.chromium.calls[0].get("channel") == "chrome"
    assert "channel" not in pw.chromium.calls[1]


def test_launch_executable_override_skips_channel(monkeypatch) -> None:
    monkeypatch.setenv("OPENCLAW_BROWSER_CHANNEL", "chrome")
    monkeypatch.setenv("PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH", "/usr/bin/chromium")
    pw = _FakePlaywright(fail_channel=False)
    bs.launch_stealth_chromium(pw)
    assert "channel" not in pw.chromium.calls[0]
    assert pw.chromium.calls[0]["executable_path"] == "/usr/bin/chromium"
