from __future__ import annotations

from urllib.error import HTTPError

import pytest

from market_monitor.http import HostRateLimitedError, HttpClient, reset_circuit_breaker


def test_get_text_uses_curl_fallback_for_timeout(monkeypatch) -> None:
    reset_circuit_breaker()
    client = HttpClient(user_agent="fixture", timeout_seconds=3)

    def fake_urlopen(*args, **kwargs):
        raise TimeoutError("timed out")

    monkeypatch.setattr("market_monitor.http.urlopen", fake_urlopen)
    monkeypatch.setattr(client, "_get_text_with_curl", lambda **kwargs: "fallback text")

    text = client.get_text("https://example.com/cards")

    assert text == "fallback text"


def test_429_trips_circuit_and_subsequent_requests_short_circuit(monkeypatch) -> None:
    reset_circuit_breaker()
    client = HttpClient(user_agent="fixture", timeout_seconds=3)
    calls = {"n": 0}

    def fake_urlopen(*args, **kwargs):
        calls["n"] += 1
        raise HTTPError("https://ratelimited.example/a", 429, "Too Many Requests", {}, None)

    monkeypatch.setattr("market_monitor.http.urlopen", fake_urlopen)

    # First request reaches the network, gets 429, trips the circuit (fail fast).
    with pytest.raises(HTTPError):
        client.get_text("https://ratelimited.example/a", retries=1, curl_fallback=False)
    assert calls["n"] == 1

    # Any further request to the same host short-circuits WITHOUT a network call.
    with pytest.raises(HostRateLimitedError):
        client.get_text("https://ratelimited.example/b")
    assert calls["n"] == 1

    # A different host is unaffected.
    monkeypatch.setattr("market_monitor.http.urlopen", lambda *a, **k: (_ for _ in ()).throw(TimeoutError("x")))
    monkeypatch.setattr(client, "_get_text_with_curl", lambda **kwargs: "ok")
    assert client.get_text("https://other.example/c") == "ok"
    reset_circuit_breaker()
