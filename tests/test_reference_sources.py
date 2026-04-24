from __future__ import annotations

from price_monitor_bot.commands import list_reference_sources


def test_reference_sources_include_official_and_marketplace_entries() -> None:
    sources = list_reference_sources()
    source_ids = {source.id for source in sources}

    assert "pokemon_official_card_search" in source_ids
    assert "mercari" in source_ids
    assert "yuyutei" in source_ids


def test_reference_sources_can_filter_by_game_and_role() -> None:
    pokemon_listing_sources = list_reference_sources(game="pokemon", reference_role="listing_price")
    source_ids = {source.id for source in pokemon_listing_sources}

    assert "mercari" in source_ids
    assert "magi_pokemon" in source_ids
    assert "magi_ws" not in source_ids
    assert all("pokemon" in source.games for source in pokemon_listing_sources)
