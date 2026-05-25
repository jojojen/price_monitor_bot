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


# ── Price-feedback loop tables (Phase 1) ────────────────────────────────────


def _make_item(item_id: str = "tcg-fb-1") -> TrackedItem:
    return TrackedItem(
        item_id=item_id,
        item_type="tcg_sealed_box",
        category="tcg",
        title="MEGA アビスアイ",
        attributes={"game": "pokemon", "item_kind": "sealed_box"},
    )


def test_save_and_find_feedback_event(tmp_path: Path) -> None:
    from market_monitor.models import PriceFeedbackEvent, utc_now
    db = MonitorDatabase(tmp_path / "fb.sqlite3")
    db.bootstrap()
    item = _make_item()
    db.upsert_item(item)
    now = utc_now()
    event = PriceFeedbackEvent(
        feedback_id="fbk-001",
        chat_id="12345",
        item_id=item.item_id,
        game="pokemon",
        item_kind="sealed_box",
        original_fair_value_jpy=16800,
        claimed_url="https://yuyu-tei.jp/sealed/abc",
        claimed_domain="yuyu-tei.jp",
        url_hash="abc123",
        extracted_price_jpy_pass1=16500,
        extracted_price_jpy_pass2=17100,
        consistency_pct=3.6,
        consensus_pct=0.0,
        extraction_confidence="high",
        raw_html_gzipped=b"\x1f\x8b\x08\x00fakegzip",
        llm_notes_json='{"pass1": {"price_jpy": 16500}}',
        status="analyzed",
        created_at=now, updated_at=now,
    )
    db.save_price_feedback(event)
    found = db.find_feedback_by_url_hash(chat_id="12345", url_hash="abc123")
    assert found is not None
    assert found["feedback_id"] == "fbk-001"
    assert found["extraction_confidence"] == "high"


def test_bump_domain_trust_creates_then_updates(tmp_path: Path) -> None:
    db = MonitorDatabase(tmp_path / "fb.sqlite3")
    db.bootstrap()

    # First bump: creates row, succeeds
    t1 = db.bump_domain_trust(
        game="pokemon", item_kind="sealed_box", domain="yuyu-tei.jp", success=True,
    )
    assert t1.vote_count == 1
    assert t1.consensus_success_count == 1
    assert t1.consensus_fail_count == 0
    # Bayes with prior_belief=5: (1*1 + 5*0.5) / (1+0+5) = 3.5/6 = 0.583
    assert abs(t1.bayes_accuracy_score - (3.5 / 6.0)) < 0.001

    # Three more successes
    for _ in range(3):
        db.bump_domain_trust(
            game="pokemon", item_kind="sealed_box", domain="yuyu-tei.jp", success=True,
        )
    # One failure
    t2 = db.bump_domain_trust(
        game="pokemon", item_kind="sealed_box", domain="yuyu-tei.jp", success=False,
    )
    # 4 succ, 1 fail, prior 5 → (4 + 2.5) / (4 + 1 + 5) = 6.5/10 = 0.65
    assert t2.vote_count == 5
    assert t2.consensus_success_count == 4
    assert t2.consensus_fail_count == 1
    assert abs(t2.bayes_accuracy_score - 0.65) < 0.001


def test_list_learned_reference_sites_filters_by_votes_and_score(tmp_path: Path) -> None:
    db = MonitorDatabase(tmp_path / "fb.sqlite3")
    db.bootstrap()
    # one promoted domain (votes >= 1, score >= 0.5)
    db.bump_domain_trust(
        game="pokemon", item_kind="sealed_box", domain="good.example", success=True,
    )
    # one seeded-only domain (vote_count == 0) — should NOT surface
    db.upsert_domain_trust_seed(
        game="pokemon", item_kind="sealed_box", domain="seed-only.example", initial_score=0.9,
    )
    rows = db.list_learned_reference_sites(
        game="pokemon", item_kind="sealed_box", limit=10,
    )
    domains = [r.domain for r in rows]
    assert "good.example" in domains
    assert "seed-only.example" not in domains  # zero votes filter


def test_extraction_examples_returns_most_recent_first(tmp_path: Path) -> None:
    from market_monitor.models import ExtractionExample, utc_now
    from datetime import timedelta
    db = MonitorDatabase(tmp_path / "fb.sqlite3")
    db.bootstrap()
    item = _make_item()
    db.upsert_item(item)
    # Need feedback rows for FK
    base = utc_now()
    for idx, (price, age_s) in enumerate(((1000, 300), (2000, 200), (3000, 100))):
        ts = base - timedelta(seconds=age_s)
        # Quick-and-dirty feedback row so the FK is satisfied
        from market_monitor.models import PriceFeedbackEvent
        db.save_price_feedback(PriceFeedbackEvent(
            feedback_id=f"fbk-{idx}", chat_id=None,
            item_id=item.item_id, game="pokemon", item_kind="sealed_box",
            original_fair_value_jpy=None, claimed_url="https://x", claimed_domain="x",
            url_hash=f"h{idx}",
            extracted_price_jpy_pass1=price, extracted_price_jpy_pass2=price,
            consistency_pct=0.0, consensus_pct=None, extraction_confidence="high",
            raw_html_gzipped=None, llm_notes_json="{}", status="analyzed",
            created_at=ts, updated_at=ts,
        ))
        db.save_extraction_example(ExtractionExample(
            example_id=f"ex-{idx}", game="pokemon", item_kind="sealed_box",
            domain="x", title=f"title-{idx}", price_jpy=price,
            captured_from_feedback_id=f"fbk-{idx}", captured_at=ts,
        ))
    examples = db.recent_extraction_examples(game="pokemon", item_kind="sealed_box", limit=2)
    assert len(examples) == 2
    # Most-recent first → price 3000 then 2000
    assert examples[0].price_jpy == 3000
    assert examples[1].price_jpy == 2000


def test_seed_domain_trust_from_reference_sources_idempotent(tmp_path: Path) -> None:
    db = MonitorDatabase(tmp_path / "fb.sqlite3")
    db.bootstrap()
    entries = [
        {"id": "yt", "url": "https://yuyu-tei.jp/", "games": ["pokemon"],
         "trust_score": 0.93, "price_weight": 0.9},
        {"id": "meta", "url": "https://pokemon-card.com/", "games": ["pokemon"],
         "trust_score": 1.0, "price_weight": 0.0},  # metadata-only — should skip
    ]
    n1 = db.seed_domain_trust_from_reference_sources(entries)
    n2 = db.seed_domain_trust_from_reference_sources(entries)  # idempotent
    rows = db.list_learned_reference_sites(
        game="pokemon", item_kind="sealed_box", min_votes=0, limit=20,
    )
    domains = {r.domain for r in rows}
    assert "yuyu-tei.jp" in domains
    assert "pokemon-card.com" not in domains
    # n1 considered both item_kinds for one entry (card + sealed_box) → 2 rows
    assert n1 == 2
    assert n2 == 2  # same value (entries considered), but no extra rows created
    # Confirm card kind also seeded
    rows_card = db.list_learned_reference_sites(
        game="pokemon", item_kind="card", min_votes=0, limit=20,
    )
    assert "yuyu-tei.jp" in {r.domain for r in rows_card}
