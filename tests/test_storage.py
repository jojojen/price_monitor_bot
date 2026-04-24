from __future__ import annotations

from datetime import datetime, timezone

from market_monitor.models import FairValueEstimate, MarketOffer, TrackedItem
from market_monitor.storage import MonitorDatabase


def test_storage_bootstrap_and_snapshot_roundtrip(tmp_path) -> None:
    database = MonitorDatabase(tmp_path / "monitor.sqlite3")
    database.bootstrap()

    item = TrackedItem(
        item_id="tcg-1",
        item_type="tcg_card",
        category="tcg",
        title="ピカチュウex",
        aliases=("Pikachu ex",),
        attributes={"game": "pokemon"},
    )
    database.upsert_item(item)
    database.save_offers(
        item.item_id,
        [
            MarketOffer(
                source="yuyutei",
                listing_id="sv08:10132",
                url="https://example.com/card",
                title="ピカチュウex",
                price_jpy=99800,
                price_kind="ask",
                captured_at=datetime.now(timezone.utc),
                source_category="poc",
                attributes={"card_number": "132/106", "rarity": "SAR"},
            )
        ],
    )
    database.save_snapshot(
        FairValueEstimate(
            item_id=item.item_id,
            amount_jpy=80000,
            confidence=0.72,
            sample_count=2,
            reasoning=("fixture test",),
        )
    )

    snapshot = database.latest_snapshot(item.item_id)
    assert snapshot is not None
    assert snapshot["fair_value_jpy"] == 80000
    assert snapshot["sample_count"] == 2
