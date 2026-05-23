from __future__ import annotations

import json
import shutil
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from hashlib import sha1
from pathlib import Path
from typing import Any, Iterator

from .models import FairValueEstimate, MarketOffer, TrackedItem, WatchRule, utc_now

SCHEMA_BASE = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS tracked_items (
    item_id TEXT PRIMARY KEY,
    item_type TEXT NOT NULL,
    category TEXT NOT NULL,
    title TEXT NOT NULL,
    attributes_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS item_aliases (
    item_id TEXT NOT NULL,
    alias TEXT NOT NULL,
    source TEXT NOT NULL DEFAULT 'manual',
    PRIMARY KEY (item_id, alias, source),
    FOREIGN KEY (item_id) REFERENCES tracked_items(item_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS watch_rules (
    rule_id TEXT PRIMARY KEY,
    item_id TEXT NOT NULL,
    source_scope TEXT NOT NULL,
    discount_threshold_pct REAL NOT NULL,
    enabled INTEGER NOT NULL,
    schedule_minutes INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY (item_id) REFERENCES tracked_items(item_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS source_offers (
    offer_id TEXT PRIMARY KEY,
    item_id TEXT NOT NULL,
    source TEXT NOT NULL,
    source_category TEXT NOT NULL,
    price_kind TEXT NOT NULL,
    title TEXT NOT NULL,
    url TEXT NOT NULL,
    price_jpy INTEGER NOT NULL,
    captured_at TEXT NOT NULL,
    availability TEXT,
    condition_text TEXT,
    raw_attributes_json TEXT NOT NULL DEFAULT '{}',
    score REAL,
    FOREIGN KEY (item_id) REFERENCES tracked_items(item_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS price_snapshots (
    snapshot_id TEXT PRIMARY KEY,
    item_id TEXT NOT NULL,
    fair_value_jpy INTEGER NOT NULL,
    confidence REAL NOT NULL,
    sample_count INTEGER NOT NULL,
    reasoning_json TEXT NOT NULL DEFAULT '[]',
    computed_at TEXT NOT NULL,
    FOREIGN KEY (item_id) REFERENCES tracked_items(item_id) ON DELETE CASCADE
);
"""

# Multi-source marketplace tables (Mercari / Rakuma / future Yuyutei et al.).
# Replaces the legacy `mercari_watchlist` and `mercari_watch_hits` tables that
# were tightly bound to a single source.
SCHEMA_MARKETPLACE = """
CREATE TABLE IF NOT EXISTS marketplace_watchlist (
    watch_id TEXT PRIMARY KEY,
    query TEXT NOT NULL,
    price_threshold_jpy INTEGER NOT NULL,
    markets_json TEXT NOT NULL DEFAULT '[]',
    market_options_json TEXT NOT NULL DEFAULT '{}',
    enabled INTEGER NOT NULL DEFAULT 1,
    chat_id TEXT NOT NULL,
    last_checked_at TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS marketplace_watch_hits (
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


_DEFAULT_MERCARI_CONDITION_IDS: tuple[int, ...] = (1, 2, 3)  # 目立った傷や汚れなし以上


@dataclass
class MarketplaceWatch:
    """A single search-keyword watch that may target multiple marketplaces.

    One ``(chat_id, query)`` pair → one watch row → fan out across every entry
    in ``markets``. ``market_options[source]`` carries per-market params (e.g.
    Mercari's ``condition_ids``); markets not in the dict use empty options.
    """

    watch_id: str
    query: str
    price_threshold_jpy: int
    markets: tuple[str, ...]
    enabled: bool
    chat_id: str
    last_checked_at: str | None
    created_at: str
    updated_at: str
    market_options: dict[str, dict[str, Any]] = field(default_factory=dict)

    def options_for(self, market: str) -> dict[str, Any]:
        """Return the per-market options dict for ``market``, or {} if absent."""
        opts = self.market_options.get(market)
        return dict(opts) if isinstance(opts, dict) else {}


@dataclass
class MarketplaceHit:
    hit_id: str
    watch_id: str
    source: str
    source_item_id: str
    title: str
    price_jpy: int
    url: str
    thumbnail_url: str | None
    first_seen_at: str
    notified: bool
    stock_count: int | None = None        # B2C only (Yuyutei et al.); C2C 一律 None
    listing_kind: str = "fixed_price"     # "fixed_price" / future "auction"


def build_marketplace_watch_id(*, chat_id: str, query: str) -> str:
    """Stable hash of (chat_id, query). One watch row = one query, regardless
    of how many markets it targets — markets are stored as a list in the row."""
    payload = f"{chat_id}|{query}"
    return sha1(payload.encode("utf-8")).hexdigest()


def _json(data: object) -> str:
    return json.dumps(data, ensure_ascii=False, sort_keys=True)


def _decode_condition_ids(raw: str | None) -> tuple[int, ...]:
    """Legacy decoder for the migration path; kept so we can read existing
    `mercari_watchlist.condition_ids` cells (comma-separated integers)."""
    if not raw:
        return _DEFAULT_MERCARI_CONDITION_IDS
    out: list[int] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            value = int(token)
        except ValueError:
            continue
        if 1 <= value <= 6:
            out.append(value)
    return tuple(out) if out else _DEFAULT_MERCARI_CONDITION_IDS


class MonitorDatabase:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def bootstrap(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as connection:
            connection.executescript(SCHEMA_BASE)
            # Two cascading migrations:
            #   (a) legacy mercari_* tables    → marketplace_* v1 (single-source rows)
            #   (b) marketplace_* v1            → marketplace_* v2 (markets-array rows)
            # Each is idempotent; on a fresh DB both no-op.
            self._migrate_mercari_to_marketplace_v1(connection)
            self._migrate_marketplace_v1_to_v2(connection)
            connection.executescript(SCHEMA_MARKETPLACE)

    def _backup_db_file(self, *, suffix: str) -> None:
        """Make a timestamped copy of the DB file before destructive migration.
        Raises if the copy fails — callers must abort migration on failure."""
        if not self.path.exists():
            return
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        backup_path = self.path.with_name(f"{self.path.name}.bak_{suffix}_{stamp}")
        shutil.copy2(self.path, backup_path)

    def _migrate_mercari_to_marketplace_v1(self, connection: sqlite3.Connection) -> None:
        """First migration: legacy ``mercari_*`` tables → ``marketplace_*`` v1
        with a ``source`` column per row. Idempotent guards:
        - if ``marketplace_watchlist`` already exists → already migrated, return
        - if ``mercari_watchlist`` is absent → fresh DB, return
        """
        if _table_exists(connection, "marketplace_watchlist"):
            return
        if not _table_exists(connection, "mercari_watchlist"):
            return

        self._backup_db_file(suffix="pre_marketplace")

        # Build a transitional v1 schema inline (NOT the current v2 schema)
        # — we want the v2 migration to handle the reshape uniformly.
        connection.executescript(
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

        id_remap: dict[str, str] = {}
        old_watches = connection.execute("SELECT * FROM mercari_watchlist").fetchall()
        for row in old_watches:
            # v1 watch_id = sha1("mercari|chat_id|query")
            new_watch_id = sha1(
                f"mercari|{row['chat_id']}|{row['query']}".encode("utf-8")
            ).hexdigest()
            id_remap[row["watch_id"]] = new_watch_id
            condition_ids = _decode_condition_ids(row["condition_ids"])
            connection.execute(
                """
                INSERT OR IGNORE INTO marketplace_watchlist (
                    watch_id, source, query, price_threshold_jpy, enabled,
                    chat_id, last_checked_at, created_at, updated_at,
                    source_options_json
                )
                VALUES (?, 'mercari', ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    new_watch_id, row["query"], int(row["price_threshold_jpy"]),
                    int(row["enabled"]), row["chat_id"], row["last_checked_at"],
                    row["created_at"], row["updated_at"],
                    _json({"condition_ids": list(condition_ids)}),
                ),
            )

        old_hits = connection.execute("SELECT * FROM mercari_watch_hits").fetchall()
        for row in old_hits:
            new_watch_id = id_remap.get(row["watch_id"])
            if new_watch_id is None:
                continue
            new_hit_id = sha1(
                f"mercari|{new_watch_id}|{row['mercari_item_id']}".encode("utf-8")
            ).hexdigest()
            connection.execute(
                """
                INSERT OR IGNORE INTO marketplace_watch_hits (
                    hit_id, watch_id, source, source_item_id, title, price_jpy,
                    url, thumbnail_url, first_seen_at, notified,
                    stock_count, listing_kind
                )
                VALUES (?, ?, 'mercari', ?, ?, ?, ?, ?, ?, ?, NULL, 'fixed_price')
                """,
                (
                    new_hit_id, new_watch_id, row["mercari_item_id"], row["title"],
                    int(row["price_jpy"]), row["url"], row["thumbnail_url"],
                    row["first_seen_at"], int(row["notified"]),
                ),
            )

        connection.execute("DROP TABLE IF EXISTS mercari_watch_hits")
        connection.execute("DROP TABLE IF EXISTS mercari_watchlist")

    def _migrate_marketplace_v1_to_v2(self, connection: sqlite3.Connection) -> None:
        """Second migration: ``marketplace_watchlist`` from v1 shape (one row
        per ``(source, chat_id, query)``) to v2 shape (one row per
        ``(chat_id, query)`` with a ``markets`` array). Idempotent guards:
        - if the v2 ``markets_json`` column already exists → already v2, return
        - if ``marketplace_watchlist`` doesn't exist → fresh DB, return

        Rows that share the same ``(chat_id, query)`` are merged into one
        watch with both markets in the array; their per-source options are
        keyed by market name in ``market_options_json``. watch_id is
        regenerated (no source component) and the FK on
        ``marketplace_watch_hits.watch_id`` is updated via id remap.
        """
        if not _table_exists(connection, "marketplace_watchlist"):
            return
        cols = {row[1] for row in connection.execute(
            "PRAGMA table_info(marketplace_watchlist)"
        )}
        if "markets_json" in cols:
            return  # already v2

        self._backup_db_file(suffix="pre_markets_array")

        # Disable FK enforcement for the duration of this migration: we have
        # to rename the parent table (`marketplace_watchlist` → `..._v1`),
        # then create a new parent, then UPDATE child rows' watch_id to the
        # new ids. SQLite's FK is enforced eagerly per-statement, so without
        # this the rename or the UPDATE will fail. Re-enabled at the bottom.
        # PRAGMA only takes effect outside a transaction → commit any pending
        # work before toggling.
        connection.commit()
        connection.execute("PRAGMA foreign_keys = OFF")

        # Pull all v1 rows; group by (chat_id, query).
        rows = connection.execute(
            "SELECT * FROM marketplace_watchlist ORDER BY created_at"
        ).fetchall()
        groups: dict[tuple[str, str], dict[str, Any]] = {}
        # old_watch_id → new_watch_id, for FK remap on hits
        id_remap: dict[str, str] = {}
        for row in rows:
            key = (str(row["chat_id"]), str(row["query"]))
            new_watch_id = build_marketplace_watch_id(
                chat_id=row["chat_id"], query=row["query"],
            )
            id_remap[row["watch_id"]] = new_watch_id
            source = str(row["source"])
            try:
                source_options = json.loads(row["source_options_json"] or "{}")
                if not isinstance(source_options, dict):
                    source_options = {}
            except (TypeError, ValueError, json.JSONDecodeError):
                source_options = {}
            entry = groups.get(key)
            if entry is None:
                entry = {
                    "watch_id": new_watch_id,
                    "query": row["query"],
                    "price_threshold_jpy": int(row["price_threshold_jpy"]),
                    "markets": [source],
                    "market_options": {source: source_options},
                    "enabled": int(row["enabled"]),
                    "chat_id": row["chat_id"],
                    "last_checked_at": row["last_checked_at"],
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"],
                }
                groups[key] = entry
            else:
                if source not in entry["markets"]:
                    entry["markets"].append(source)
                entry["market_options"][source] = source_options
                # If any source-row is enabled, the merged watch is enabled.
                entry["enabled"] = max(entry["enabled"], int(row["enabled"]))
                # Take the max price_threshold (defensive — they should match).
                entry["price_threshold_jpy"] = max(
                    entry["price_threshold_jpy"], int(row["price_threshold_jpy"]),
                )

        # Rebuild marketplace_watchlist with the v2 schema.
        connection.execute("ALTER TABLE marketplace_watchlist RENAME TO marketplace_watchlist_v1")
        connection.executescript(SCHEMA_MARKETPLACE)

        for entry in groups.values():
            connection.execute(
                """
                INSERT OR IGNORE INTO marketplace_watchlist (
                    watch_id, query, price_threshold_jpy, markets_json,
                    market_options_json, enabled, chat_id, last_checked_at,
                    created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    entry["watch_id"], entry["query"], entry["price_threshold_jpy"],
                    _json(entry["markets"]), _json(entry["market_options"]),
                    entry["enabled"], entry["chat_id"], entry["last_checked_at"],
                    entry["created_at"], entry["updated_at"],
                ),
            )

        # Update FK on hits to point at the new watch_ids.
        for old_id, new_id in id_remap.items():
            if old_id == new_id:
                continue
            connection.execute(
                "UPDATE marketplace_watch_hits SET watch_id = ? WHERE watch_id = ?",
                (new_id, old_id),
            )

        connection.execute("DROP TABLE marketplace_watchlist_v1")
        connection.commit()  # close migration transaction before flipping FK
        connection.execute("PRAGMA foreign_keys = ON")

    # ── Tracked items / watch rules / offers (unchanged from before) ─────────

    def upsert_item(self, item: TrackedItem) -> None:
        timestamp = utc_now().isoformat()
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO tracked_items (item_id, item_type, category, title, attributes_json, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(item_id) DO UPDATE SET
                    item_type=excluded.item_type,
                    category=excluded.category,
                    title=excluded.title,
                    attributes_json=excluded.attributes_json,
                    updated_at=excluded.updated_at
                """,
                (
                    item.item_id,
                    item.item_type,
                    item.category,
                    item.title,
                    _json(dict(item.attributes)),
                    timestamp,
                    timestamp,
                ),
            )
            connection.execute("DELETE FROM item_aliases WHERE item_id = ?", (item.item_id,))
            connection.executemany(
                "INSERT INTO item_aliases (item_id, alias, source) VALUES (?, ?, ?)",
                [(item.item_id, alias, "manual") for alias in item.aliases],
            )

    def save_watch_rule(self, rule: WatchRule) -> None:
        timestamp = utc_now().isoformat()
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO watch_rules (
                    rule_id, item_id, source_scope, discount_threshold_pct, enabled, schedule_minutes, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(rule_id) DO UPDATE SET
                    source_scope=excluded.source_scope,
                    discount_threshold_pct=excluded.discount_threshold_pct,
                    enabled=excluded.enabled,
                    schedule_minutes=excluded.schedule_minutes,
                    updated_at=excluded.updated_at
                """,
                (
                    rule.rule_id,
                    rule.item_id,
                    rule.source_scope,
                    rule.discount_threshold_pct,
                    int(rule.enabled),
                    rule.schedule_minutes,
                    timestamp,
                    timestamp,
                ),
            )

    def save_offers(self, item_id: str, offers: list[MarketOffer] | tuple[MarketOffer, ...]) -> None:
        with self.connect() as connection:
            connection.executemany(
                """
                INSERT OR REPLACE INTO source_offers (
                    offer_id, item_id, source, source_category, price_kind, title, url, price_jpy, captured_at,
                    availability, condition_text, raw_attributes_json, score
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        self._offer_id(item_id, offer),
                        item_id,
                        offer.source,
                        offer.source_category,
                        offer.price_kind,
                        offer.title,
                        offer.url,
                        offer.price_jpy,
                        offer.captured_at.isoformat(),
                        offer.availability,
                        offer.condition,
                        _json(dict(offer.attributes)),
                        offer.score,
                    )
                    for offer in offers
                ],
            )

    def save_snapshot(self, snapshot: FairValueEstimate) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO price_snapshots (
                    snapshot_id, item_id, fair_value_jpy, confidence, sample_count, reasoning_json, computed_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    self._snapshot_id(snapshot),
                    snapshot.item_id,
                    snapshot.amount_jpy,
                    snapshot.confidence,
                    snapshot.sample_count,
                    _json(list(snapshot.reasoning)),
                    snapshot.computed_at.isoformat(),
                ),
            )

    def latest_snapshot(self, item_id: str) -> sqlite3.Row | None:
        with self.connect() as connection:
            return connection.execute(
                """
                SELECT *
                FROM price_snapshots
                WHERE item_id = ?
                ORDER BY computed_at DESC
                LIMIT 1
                """,
                (item_id,),
            ).fetchone()

    # ── Marketplace watchlist CRUD ───────────────────────────────────────────

    def add_marketplace_watch(self, watch: MarketplaceWatch) -> None:
        timestamp = utc_now().isoformat()
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO marketplace_watchlist (
                    watch_id, query, price_threshold_jpy, markets_json,
                    market_options_json, enabled, chat_id, last_checked_at,
                    created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(watch_id) DO UPDATE SET
                    query=excluded.query,
                    price_threshold_jpy=excluded.price_threshold_jpy,
                    markets_json=excluded.markets_json,
                    market_options_json=excluded.market_options_json,
                    enabled=excluded.enabled,
                    chat_id=excluded.chat_id,
                    updated_at=excluded.updated_at
                """,
                (
                    watch.watch_id,
                    watch.query,
                    watch.price_threshold_jpy,
                    _json(list(watch.markets)),
                    _json({k: dict(v) for k, v in watch.market_options.items()}),
                    int(watch.enabled),
                    watch.chat_id,
                    watch.last_checked_at,
                    timestamp,
                    timestamp,
                ),
            )

    def delete_marketplace_watch(self, watch_id: str) -> bool:
        with self.connect() as connection:
            cursor = connection.execute(
                "DELETE FROM marketplace_watchlist WHERE watch_id = ?", (watch_id,)
            )
            return cursor.rowcount > 0

    def toggle_marketplace_watch(self, watch_id: str, *, enabled: bool) -> bool:
        timestamp = utc_now().isoformat()
        with self.connect() as connection:
            cursor = connection.execute(
                "UPDATE marketplace_watchlist SET enabled = ?, updated_at = ? WHERE watch_id = ?",
                (int(enabled), timestamp, watch_id),
            )
            return cursor.rowcount > 0

    def update_marketplace_watch(
        self,
        watch_id: str,
        *,
        query: str | None = None,
        price_threshold_jpy: int | None = None,
        markets: tuple[str, ...] | list[str] | None = None,
        market_options: dict[str, dict[str, Any]] | None = None,
    ) -> bool:
        timestamp = utc_now().isoformat()
        sets: list[str] = []
        params: list[object] = []
        if query is not None:
            sets.append("query = ?")
            params.append(query)
        if price_threshold_jpy is not None:
            sets.append("price_threshold_jpy = ?")
            params.append(price_threshold_jpy)
        if markets is not None:
            sets.append("markets_json = ?")
            params.append(_json(list(markets)))
        if market_options is not None:
            sets.append("market_options_json = ?")
            params.append(_json({k: dict(v) for k, v in market_options.items()}))
        if not sets:
            return False
        sets.append("updated_at = ?")
        params.append(timestamp)
        params.append(watch_id)
        with self.connect() as connection:
            cursor = connection.execute(
                f"UPDATE marketplace_watchlist SET {', '.join(sets)} WHERE watch_id = ?",  # noqa: S608
                params,
            )
            return cursor.rowcount > 0

    def list_marketplace_watchlist(
        self, *, market: str | None = None
    ) -> list[MarketplaceWatch]:
        """List all watches, optionally filtered to those whose ``markets``
        array contains ``market``."""
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM marketplace_watchlist ORDER BY created_at DESC"
            ).fetchall()
        watches = [_row_to_marketplace_watch(row) for row in rows]
        if market is None:
            return watches
        return [w for w in watches if market in w.markets]

    def get_marketplace_watch(self, watch_id: str) -> MarketplaceWatch | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM marketplace_watchlist WHERE watch_id = ?", (watch_id,)
            ).fetchone()
        return _row_to_marketplace_watch(row) if row else None

    def mark_watch_checked(self, watch_id: str) -> None:
        timestamp = utc_now().isoformat()
        with self.connect() as connection:
            connection.execute(
                "UPDATE marketplace_watchlist SET last_checked_at = ?, updated_at = ? "
                "WHERE watch_id = ?",
                (timestamp, timestamp, watch_id),
            )

    def record_marketplace_hits(
        self,
        *,
        watch_id: str,
        source: str,
        items: list[dict[str, object]],
    ) -> list[dict[str, object]]:
        """Upsert hits; return items that are NEW or have a PRICE CHANGE.

        Each input item is a dict with at least: item_id (source-local),
        title, price_jpy, url. Optional: thumbnail_url, stock_count,
        listing_kind. Each returned item carries extra keys:
          _event: "new" | "price_changed"
          _old_price: int  (only when _event == "price_changed")
        """
        now = utc_now().isoformat()
        new_or_changed: list[dict[str, object]] = []
        with self.connect() as connection:
            for item in items:
                source_item_id = str(item.get("item_id", ""))
                if not source_item_id:
                    continue
                hit_id = sha1(
                    f"{source}|{watch_id}|{source_item_id}".encode()
                ).hexdigest()
                price_jpy = int(item.get("price_jpy", 0))
                title = str(item.get("title", ""))
                url = str(item.get("url", ""))
                thumbnail = item.get("thumbnail_url")
                stock_count = item.get("stock_count")
                listing_kind = str(item.get("listing_kind", "fixed_price"))

                existing = connection.execute(
                    "SELECT price_jpy FROM marketplace_watch_hits "
                    "WHERE watch_id = ? AND source = ? AND source_item_id = ?",
                    (watch_id, source, source_item_id),
                ).fetchone()

                if existing is None:
                    try:
                        connection.execute(
                            """
                            INSERT INTO marketplace_watch_hits
                                (hit_id, watch_id, source, source_item_id, title,
                                 price_jpy, url, thumbnail_url, first_seen_at,
                                 notified, stock_count, listing_kind)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?)
                            """,
                            (
                                hit_id, watch_id, source, source_item_id, title,
                                price_jpy, url, thumbnail, now,
                                stock_count, listing_kind,
                            ),
                        )
                        new_or_changed.append({**item, "_event": "new"})
                    except sqlite3.IntegrityError:
                        pass  # race condition; skip
                elif int(existing["price_jpy"]) != price_jpy:
                    old_price = int(existing["price_jpy"])
                    connection.execute(
                        """
                        UPDATE marketplace_watch_hits
                        SET price_jpy = ?, title = ?, thumbnail_url = ?,
                            first_seen_at = ?, notified = 0,
                            stock_count = ?, listing_kind = ?
                        WHERE watch_id = ? AND source = ? AND source_item_id = ?
                        """,
                        (
                            price_jpy, title, thumbnail, now,
                            stock_count, listing_kind,
                            watch_id, source, source_item_id,
                        ),
                    )
                    new_or_changed.append(
                        {**item, "_event": "price_changed", "_old_price": old_price}
                    )
                # else: same price, already handled → skip
        return new_or_changed

    def mark_marketplace_hits_notified(
        self, *, watch_id: str, source: str, source_item_ids: list[str]
    ) -> None:
        if not source_item_ids:
            return
        with self.connect() as connection:
            connection.executemany(
                "UPDATE marketplace_watch_hits SET notified = 1 "
                "WHERE watch_id = ? AND source = ? AND source_item_id = ?",
                [(watch_id, source, sid) for sid in source_item_ids],
            )

    def list_marketplace_hits(
        self, watch_id: str, *, limit: int = 10
    ) -> list[MarketplaceHit]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM marketplace_watch_hits
                WHERE watch_id = ?
                ORDER BY first_seen_at DESC
                LIMIT ?
                """,
                (watch_id, limit),
            ).fetchall()
        return [_row_to_marketplace_hit(row) for row in rows]

    @staticmethod
    def _offer_id(item_id: str, offer: MarketOffer) -> str:
        payload = "|".join(
            [
                item_id,
                offer.source,
                offer.price_kind,
                offer.url,
                offer.captured_at.isoformat(),
                str(offer.price_jpy),
            ]
        )
        return sha1(payload.encode("utf-8")).hexdigest()

    @staticmethod
    def _snapshot_id(snapshot: FairValueEstimate) -> str:
        payload = "|".join([snapshot.item_id, snapshot.computed_at.isoformat(), str(snapshot.amount_jpy)])
        return sha1(payload.encode("utf-8")).hexdigest()


def _table_exists(connection: sqlite3.Connection, table_name: str) -> bool:
    row = connection.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (table_name,),
    ).fetchone()
    return row is not None


def _row_to_marketplace_watch(row: sqlite3.Row) -> MarketplaceWatch:
    try:
        markets_raw = row["markets_json"]
    except (IndexError, KeyError):
        markets_raw = "[]"
    try:
        markets_list = json.loads(markets_raw) if markets_raw else []
    except (TypeError, ValueError, json.JSONDecodeError):
        markets_list = []
    if not isinstance(markets_list, list):
        markets_list = []
    markets = tuple(str(m) for m in markets_list if m)

    try:
        opts_raw = row["market_options_json"]
    except (IndexError, KeyError):
        opts_raw = "{}"
    try:
        opts_parsed = json.loads(opts_raw) if opts_raw else {}
    except (TypeError, ValueError, json.JSONDecodeError):
        opts_parsed = {}
    market_options: dict[str, dict[str, Any]] = {}
    if isinstance(opts_parsed, dict):
        for k, v in opts_parsed.items():
            if isinstance(v, dict):
                market_options[str(k)] = v

    return MarketplaceWatch(
        watch_id=row["watch_id"],
        query=row["query"],
        price_threshold_jpy=int(row["price_threshold_jpy"]),
        markets=markets,
        enabled=bool(row["enabled"]),
        chat_id=row["chat_id"],
        last_checked_at=row["last_checked_at"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        market_options=market_options,
    )


def _row_to_marketplace_hit(row: sqlite3.Row) -> MarketplaceHit:
    try:
        stock_raw = row["stock_count"]
    except (IndexError, KeyError):
        stock_raw = None
    try:
        listing_kind = row["listing_kind"] or "fixed_price"
    except (IndexError, KeyError):
        listing_kind = "fixed_price"
    return MarketplaceHit(
        hit_id=row["hit_id"],
        watch_id=row["watch_id"],
        source=row["source"],
        source_item_id=row["source_item_id"],
        title=row["title"],
        price_jpy=int(row["price_jpy"]),
        url=row["url"],
        thumbnail_url=row["thumbnail_url"],
        first_seen_at=row["first_seen_at"],
        notified=bool(row["notified"]),
        stock_count=int(stock_raw) if stock_raw is not None else None,
        listing_kind=listing_kind,
    )
