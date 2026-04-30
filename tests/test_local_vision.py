from __future__ import annotations

import json

from tcg_tracker.local_vision import (
    LocalVisionTimeoutError,
    OllamaLocalVisionClient,
    build_local_vision_client,
    build_local_vision_clients,
)

CHARIZARD_JP = "リザードンex"


def test_build_local_vision_client_defaults_to_ollama_when_model_is_configured() -> None:
    client = build_local_vision_client(
        endpoint="http://127.0.0.1:11434",
        model_list="qwen2.5vl:3b",
        timeout_seconds=75,
    )

    assert client is not None
    assert client.backend == "ollama"
    assert client.model == "qwen2.5vl:3b"
    assert client.timeout_seconds == 75


def test_build_local_vision_clients_supports_comma_separated_models() -> None:
    clients = build_local_vision_clients(
        endpoint="http://127.0.0.1:11434",
        model_list="qwen2.5vl:3b, gemma3:4b, qwen2.5vl:3b",
        timeout_seconds=75,
    )

    assert [client.model for client in clients] == ["qwen2.5vl:3b", "gemma3:4b"]


def test_build_local_vision_clients_allows_disabled_backend_with_stale_model() -> None:
    clients = build_local_vision_clients(
        endpoint="http://127.0.0.1:11434",
        model_list="qwen2.5vl:3b",
        backend=None,
        timeout_seconds=75,
    )

    assert clients == ()


def test_ollama_local_vision_client_parses_structured_card_response(monkeypatch, tmp_path) -> None:
    image_path = tmp_path / "charizard.jpg"
    image_path.write_bytes(b"fake-image")

    client = OllamaLocalVisionClient(
        endpoint="http://127.0.0.1:11434",
        model="qwen2.5vl:3b",
        timeout_seconds=60,
    )

    payload = {
        "game": "pokemon",
        "title": CHARIZARD_JP,
        "aliases": ["Charizard ex"],
        "card_number": "201 / 165",
        "rarity": "sar",
        "set_code": "SV2A",
        "confidence": 0.94,
    }
    monkeypatch.setattr(client, "_post_generate", lambda request_payload: json.dumps(payload, ensure_ascii=False))

    candidate = client.analyze_card_image(image_path)

    assert candidate is not None
    assert candidate.backend == "ollama"
    assert candidate.model == "qwen2.5vl:3b"
    assert candidate.game == "pokemon"
    assert candidate.title == CHARIZARD_JP
    assert candidate.aliases == ("Charizard ex",)
    assert candidate.card_number == "201/165"
    assert candidate.rarity == "SAR"
    assert candidate.set_code == "sv2a"
    assert candidate.confidence == 0.94


def test_ollama_local_vision_client_raises_structured_timeout(monkeypatch, tmp_path) -> None:
    image_path = tmp_path / "charizard.jpg"
    image_path.write_bytes(b"fake-image")

    client = OllamaLocalVisionClient(
        endpoint="http://127.0.0.1:11434",
        model="qwen2.5vl:3b",
        timeout_seconds=45,
    )
    monkeypatch.setattr("tcg_tracker.local_vision.urlopen", lambda *args, **kwargs: (_ for _ in ()).throw(TimeoutError("timed out")))

    try:
        client.analyze_card_image(image_path)
    except LocalVisionTimeoutError as exc:
        assert exc.timeout_seconds == 45
        assert exc.descriptor == "ollama:qwen2.5vl:3b"
    else:
        raise AssertionError("expected LocalVisionTimeoutError")
