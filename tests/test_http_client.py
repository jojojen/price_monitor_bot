from __future__ import annotations

from market_monitor.http import HttpClient


def test_get_text_uses_curl_fallback_for_timeout(monkeypatch) -> None:
    client = HttpClient(user_agent="fixture", timeout_seconds=3)

    def fake_urlopen(*args, **kwargs):
        raise TimeoutError("timed out")

    monkeypatch.setattr("market_monitor.http.urlopen", fake_urlopen)
    monkeypatch.setattr(client, "_get_text_with_curl", lambda **kwargs: "fallback text")

    text = client.get_text("https://example.com/cards")

    assert text == "fallback text"
