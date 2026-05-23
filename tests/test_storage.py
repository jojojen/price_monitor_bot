from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from market_monitor.models import FairValueEstimate, MarketOffer, TrackedItem
from market_monitor.storage import (
    MarketplaceWatch,
    MonitorDatabase,
    build_marketplace_watch_id,
)


def test_storage_bootstrap_and_snapshot_roundtrip(tmp_path: Path) -> None:
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


# ── Marketplace watch CRUD (v2: markets-array shape) ────────────────────────


def _make_watch(
    *,
    markets: tuple[str, ...] = ("mercari",),
    market_options: dict[str, dict[str, object]] | None = None,
    query: str = "test",
) -> MarketplaceWatch:
    if market_options is None:
        market_options = {}
        for m in markets:
            market_options[m] = {"condition_ids": [1, 2, 3]} if m == "mercari" else {}
    return MarketplaceWatch(
        watch_id=build_marketplace_watch_id(chat_id="123", query=query),
        query=query,
        price_threshold_jpy=5000,
        markets=markets,
        enabled=True,
        chat_id="123",
        last_checked_at=None,
        created_at="2026-01-01T00:00:00+00:00",
        updated_at="2026-01-01T00:00:00+00:00",
        market_options=market_options,
    )


def test_marketplace_watch_with_market_options_persists(tmp_path: Path) -> None:
    db = MonitorDatabase(tmp_path / "m.sqlite3")
    db.bootstrap()
    watch = _make_watch()
    db.add_marketplace_watch(watch)
    loaded = db.get_marketplace_watch(watch.watch_id)
    assert loaded is not None
    assert loaded.markets == ("mercari",)
    assert loaded.options_for("mercari") == {"condition_ids": [1, 2, 3]}


def test_marketplace_watch_with_multiple_markets_persists(tmp_path: Path) -> None:
    db = MonitorDatabase(tmp_path / "m.sqlite3")
    db.bootstrap()
    watch = _make_watch(
        markets=("mercari", "rakuma"),
        market_options={"mercari": {"condition_ids": [1, 2]}, "rakuma": {}},
    )
    db.add_marketplace_watch(watch)
    loaded = db.get_marketplace_watch(watch.watch_id)
    assert loaded is not None
    assert loaded.markets == ("mercari", "rakuma")
    assert loaded.options_for("mercari") == {"condition_ids": [1, 2]}
    assert loaded.options_for("rakuma") == {}


def test_watch_id_does_not_include_market(tmp_path: Path) -> None:
    """Same (chat_id, query) always produces the same watch_id regardless of
    which markets are targeted — that's what enables 'one watch row, many
    markets'."""
    a = build_marketplace_watch_id(chat_id="123", query="abc")
    b = build_marketplace_watch_id(chat_id="123", query="abc")
    assert a == b


def test_list_marketplace_watchlist_filters_by_market(tmp_path: Path) -> None:
    db = MonitorDatabase(tmp_path / "m.sqlite3")
    db.bootstrap()
    db.add_marketplace_watch(_make_watch(markets=("mercari",), query="a"))
    db.add_marketplace_watch(_make_watch(markets=("rakuma",), query="b"))
    db.add_marketplace_watch(_make_watch(markets=("mercari", "rakuma"), query="c"))
    assert len(db.list_marketplace_watchlist()) == 3
    # market="mercari" matches watches whose markets contains mercari
    mercari_only = db.list_marketplace_watchlist(market="mercari")
    assert {w.query for w in mercari_only} == {"a", "c"}
    rakuma_only = db.list_marketplace_watchlist(market="rakuma")
    assert {w.query for w in rakuma_only} == {"b", "c"}


def test_update_marketplace_watch_changes_market_options(tmp_path: Path) -> None:
    db = MonitorDatabase(tmp_path / "m.sqlite3")
    db.bootstrap()
    watch = _make_watch()
    db.add_marketplace_watch(watch)
    assert db.update_marketplace_watch(
        watch.watch_id,
        market_options={"mercari": {"condition_ids": [1]}},
    )
    loaded = db.get_marketplace_watch(watch.watch_id)
    assert loaded is not None
    assert loaded.options_for("mercari") == {"condition_ids": [1]}


def test_update_marketplace_watch_changes_markets(tmp_path: Path) -> None:
    db = MonitorDatabase(tmp_path / "m.sqlite3")
    db.bootstrap()
    watch = _make_watch(markets=("mercari",))
    db.add_marketplace_watch(watch)
    db.update_marketplace_watch(watch.watch_id, markets=("mercari", "rakuma"))
    loaded = db.get_marketplace_watch(watch.watch_id)
    assert loaded is not None and loaded.markets == ("mercari", "rakuma")


def test_record_marketplace_hits_writes_source_and_kind(tmp_path: Path) -> None:
    db = MonitorDatabase(tmp_path / "m.sqlite3")
    db.bootstrap()
    watch = _make_watch(markets=("rakuma",), query="q")
    db.add_marketplace_watch(watch)
    items = [
        {"item_id": "rk-001", "title": "abc", "price_jpy": 1234,
         "url": "https://fril.jp/item/1"},
    ]
    out = db.record_marketplace_hits(
        watch_id=watch.watch_id, source="rakuma", items=items,
    )
    assert out and out[0]["_event"] == "new"
    hits = db.list_marketplace_hits(watch.watch_id)
    assert len(hits) == 1
    assert hits[0].source == "rakuma"
    assert hits[0].source_item_id == "rk-001"
    assert hits[0].listing_kind == "fixed_price"


# ── Legacy mercari_* → marketplace_* v1 migration ───────────────────────────


def _seed_legacy_mercari_db(db_path: Path) -> None:
    """Create a DB with the legacy mercari_* tables (pre-marketplace schema)."""
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE mercari_watchlist (
            watch_id TEXT PRIMARY KEY,
            query TEXT NOT NULL,
            price_threshold_jpy INTEGER NOT NULL,
            enabled INTEGER NOT NULL DEFAULT 1,
            chat_id TEXT NOT NULL,
            last_checked_at TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            condition_ids TEXT NOT NULL DEFAULT '1,2,3'
        );
        CREATE TABLE mercari_watch_hits (
            hit_id TEXT PRIMARY KEY,
            watch_id TEXT NOT NULL,
            mercari_item_id TEXT NOT NULL,
            title TEXT NOT NULL,
            price_jpy INTEGER NOT NULL,
            url TEXT NOT NULL,
            thumbnail_url TEXT,
            first_seen_at TEXT NOT NULL,
            notified INTEGER NOT NULL DEFAULT 0,
            UNIQUE (watch_id, mercari_item_id)
        );
        """
    )
    conn.execute(
        "INSERT INTO mercari_watchlist VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ("old-w1", "水野愛 ssp", 25000, 1, "5631877240", None,
         "2026-01-01", "2026-01-01", "1,2"),
    )
    conn.execute(
        "INSERT INTO mercari_watchlist VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ("old-w2", "アビスアイ box", 8000, 1, "5631877240", "2026-04-01",
         "2026-02-01", "2026-04-01", "1,2,3"),
    )
    conn.execute(
        "INSERT INTO mercari_watch_hits VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ("hit-1", "old-w1", "m1234", "見つけた水野愛 ssp", 22000,
         "https://jp.mercari.com/item/m1234", None, "2026-04-01", 0),
    )
    conn.commit()
    conn.close()


def test_bootstrap_migrates_legacy_mercari_to_markets_array(tmp_path: Path) -> None:
    """End-to-end: legacy mercari_* → v1 marketplace_* → v2 marketplace_*
    runs through both migrations in one bootstrap call."""
    db_path = tmp_path / "legacy.sqlite3"
    _seed_legacy_mercari_db(db_path)

    db = MonitorDatabase(db_path)
    db.bootstrap()

    watches = db.list_marketplace_watchlist()
    assert len(watches) == 2
    queries = {w.query: w for w in watches}
    assert "水野愛 ssp" in queries
    assert "アビスアイ box" in queries
    # Each migrated watch has markets=["mercari"] and Mercari condition_ids
    # preserved in market_options["mercari"]["condition_ids"].
    for w in watches:
        assert w.markets == ("mercari",)
        assert "condition_ids" in w.options_for("mercari")
    assert queries["水野愛 ssp"].options_for("mercari")["condition_ids"] == [1, 2]


def test_bootstrap_migrates_hits_with_remapped_watch_id(tmp_path: Path) -> None:
    db_path = tmp_path / "legacy.sqlite3"
    _seed_legacy_mercari_db(db_path)
    db = MonitorDatabase(db_path)
    db.bootstrap()

    new_w1 = next(w for w in db.list_marketplace_watchlist() if w.query == "水野愛 ssp")
    hits = db.list_marketplace_hits(new_w1.watch_id)
    assert len(hits) == 1
    assert hits[0].source == "mercari"
    assert hits[0].source_item_id == "m1234"
    assert hits[0].price_jpy == 22000


def test_bootstrap_drops_legacy_mercari_tables(tmp_path: Path) -> None:
    db_path = tmp_path / "legacy.sqlite3"
    _seed_legacy_mercari_db(db_path)
    db = MonitorDatabase(db_path)
    db.bootstrap()

    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'mercari%'"
    ).fetchall()
    conn.close()
    assert rows == []


def test_bootstrap_creates_backup_file_before_migration(tmp_path: Path) -> None:
    db_path = tmp_path / "legacy.sqlite3"
    _seed_legacy_mercari_db(db_path)
    db = MonitorDatabase(db_path)
    db.bootstrap()

    backups = list(tmp_path.glob("legacy.sqlite3.bak_pre_*"))
    # At least one backup (could be 1 if both migrations run in single bootstrap,
    # but legacy_mercari migration creates a v1 schema directly so only the
    # pre_marketplace backup is created).
    assert len(backups) >= 1


def test_bootstrap_is_idempotent_on_second_run(tmp_path: Path) -> None:
    db_path = tmp_path / "legacy.sqlite3"
    _seed_legacy_mercari_db(db_path)

    db = MonitorDatabase(db_path)
    db.bootstrap()
    first_count = len(db.list_marketplace_watchlist())
    db.bootstrap()
    second_count = len(db.list_marketplace_watchlist())
    assert first_count == second_count


def test_bootstrap_no_op_on_fresh_db(tmp_path: Path) -> None:
    db = MonitorDatabase(tmp_path / "fresh.sqlite3")
    db.bootstrap()
    backups = list(tmp_path.glob("fresh.sqlite3.bak_*"))
    assert backups == []
    assert db.list_marketplace_watchlist() == []


# ── v1 marketplace_* (single-source) → v2 marketplace_* (markets array) ─────


def _seed_v1_marketplace_db(db_path: Path) -> None:
    """Create a DB that already has the v1 (single-source) marketplace_* shape,
    as it would look after only the legacy → v1 migration has run."""
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE marketplace_watchlist (
            watch_id TEXT PRIMARY KEY,
            source TEXT NOT NULL,
            query TEXT NOT NULL,
            price_threshold_jpy INTEGER NOT NULL,
            enabled INTEGER NOT NULL DEFAULT 1,
            chat_id TEXT NOT NULL,
            last_checked_at TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            source_options_json TEXT NOT NULL DEFAULT '{}'
        );
        CREATE TABLE marketplace_watch_hits (
            hit_id TEXT PRIMARY KEY,
            watch_id TEXT NOT NULL,
            source TEXT NOT NULL,
            source_item_id TEXT NOT NULL,
            title TEXT NOT NULL,
            price_jpy INTEGER NOT NULL,
            url TEXT NOT NULL,
            thumbnail_url TEXT,
            first_seen_at TEXT NOT NULL,
            notified INTEGER NOT NULL DEFAULT 0,
            stock_count INTEGER,
            listing_kind TEXT NOT NULL DEFAULT 'fixed_price',
            FOREIGN KEY (watch_id) REFERENCES marketplace_watchlist(watch_id) ON DELETE CASCADE,
            UNIQUE (watch_id, source, source_item_id)
        );
        """
    )
    # Same (chat_id, query) on two markets → after v2 migration should merge
    # into one row with markets=["mercari", "rakuma"].
    conn.execute(
        "INSERT INTO marketplace_watchlist VALUES "
        "(?, 'mercari', ?, ?, 1, ?, NULL, ?, ?, ?)",
        ("v1-merc", "綾波レイ", 4500, "c1", "2026-01-01", "2026-01-01",
         '{"condition_ids":[1,2,3]}'),
    )
    conn.execute(
        "INSERT INTO marketplace_watchlist VALUES "
        "(?, 'rakuma', ?, ?, 1, ?, NULL, ?, ?, ?)",
        ("v1-rak", "綾波レイ", 4500, "c1", "2026-01-01", "2026-01-01", "{}"),
    )
    # Hit attached to v1-merc — its watch_id must be remapped after v2 migration.
    conn.execute(
        "INSERT INTO marketplace_watch_hits VALUES "
        "(?, ?, 'mercari', ?, ?, ?, ?, NULL, ?, 0, NULL, 'fixed_price')",
        ("hit-merc", "v1-merc", "m1", "綾波 hit", 4400,
         "https://jp.mercari.com/item/m1", "2026-04-01"),
    )
    conn.commit()
    conn.close()


def test_v1_to_v2_merges_rows_sharing_chat_id_and_query(tmp_path: Path) -> None:
    db_path = tmp_path / "v1.sqlite3"
    _seed_v1_marketplace_db(db_path)
    db = MonitorDatabase(db_path)
    db.bootstrap()
    watches = db.list_marketplace_watchlist()
    assert len(watches) == 1
    w = watches[0]
    assert set(w.markets) == {"mercari", "rakuma"}
    assert w.query == "綾波レイ"
    assert w.options_for("mercari") == {"condition_ids": [1, 2, 3]}
    assert w.options_for("rakuma") == {}


def test_v1_to_v2_remaps_hits_watch_id(tmp_path: Path) -> None:
    db_path = tmp_path / "v1.sqlite3"
    _seed_v1_marketplace_db(db_path)
    db = MonitorDatabase(db_path)
    db.bootstrap()
    merged = db.list_marketplace_watchlist()[0]
    hits = db.list_marketplace_hits(merged.watch_id)
    assert len(hits) == 1
    assert hits[0].source == "mercari"
    assert hits[0].source_item_id == "m1"
