from __future__ import annotations

from datetime import datetime, timezone

import sqlite3

from market_monitor.models import FairValueEstimate, MarketOffer, TrackedItem
from market_monitor.storage import MercariWatch, MonitorDatabase


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


def _make_watch(*, condition_ids: tuple[int, ...] = (1, 2, 3)) -> MercariWatch:
    return MercariWatch(
        watch_id="wid-test",
        query="test",
        price_threshold_jpy=5000,
        enabled=True,
        chat_id="123",
        last_checked_at=None,
        created_at="2026-01-01T00:00:00+00:00",
        updated_at="2026-01-01T00:00:00+00:00",
        condition_ids=condition_ids,
    )


def test_mercari_watch_default_condition_ids_persists(tmp_path) -> None:
    db = MonitorDatabase(tmp_path / "m.sqlite3")
    db.bootstrap()
    db.add_mercari_watch(_make_watch())
    loaded = db.get_mercari_watch("wid-test")
    assert loaded is not None
    assert loaded.condition_ids == (1, 2, 3)


def test_mercari_watch_custom_condition_ids_persists(tmp_path) -> None:
    db = MonitorDatabase(tmp_path / "m.sqlite3")
    db.bootstrap()
    db.add_mercari_watch(_make_watch(condition_ids=(1, 2, 3, 4)))
    loaded = db.get_mercari_watch("wid-test")
    assert loaded is not None and loaded.condition_ids == (1, 2, 3, 4)


def test_update_mercari_watch_changes_condition_ids(tmp_path) -> None:
    db = MonitorDatabase(tmp_path / "m.sqlite3")
    db.bootstrap()
    db.add_mercari_watch(_make_watch())
    changed = db.update_mercari_watch("wid-test", condition_ids=(1,))
    assert changed is True
    loaded = db.get_mercari_watch("wid-test")
    assert loaded is not None and loaded.condition_ids == (1,)


def test_mercari_watchlist_migration_adds_condition_ids_to_legacy_schema(tmp_path) -> None:
    """A DB created before condition_ids existed gets the column added on
    bootstrap; existing rows pick up the safe default."""
    db_path = tmp_path / "legacy.sqlite3"
    legacy = sqlite3.connect(db_path)
    legacy.execute(
        """
        CREATE TABLE mercari_watchlist (
            watch_id TEXT PRIMARY KEY,
            query TEXT NOT NULL,
            price_threshold_jpy INTEGER NOT NULL,
            enabled INTEGER NOT NULL DEFAULT 1,
            chat_id TEXT NOT NULL,
            last_checked_at TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )
    legacy.execute(
        "INSERT INTO mercari_watchlist VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        ("old-w", "old query", 1000, 1, "c", None, "2026-01-01", "2026-01-01"),
    )
    legacy.commit()
    legacy.close()

    db = MonitorDatabase(db_path)
    db.bootstrap()
    loaded = db.get_mercari_watch("old-w")
    assert loaded is not None
    assert loaded.condition_ids == (1, 2, 3)  # default applied via migration
