from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from hashlib import sha1
from pathlib import Path
from typing import Iterator

from .models import FairValueEstimate, MarketOffer, TrackedItem, WatchRule, utc_now

SCHEMA = """
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

CREATE TABLE IF NOT EXISTS mercari_watchlist (
    watch_id TEXT PRIMARY KEY,
    query TEXT NOT NULL,
    price_threshold_jpy INTEGER NOT NULL,
    enabled INTEGER NOT NULL DEFAULT 1,
    chat_id TEXT NOT NULL,
    last_checked_at TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS mercari_watch_hits (
    hit_id TEXT PRIMARY KEY,
    watch_id TEXT NOT NULL,
    mercari_item_id TEXT NOT NULL,
    title TEXT NOT NULL,
    price_jpy INTEGER NOT NULL,
    url TEXT NOT NULL,
    thumbnail_url TEXT,
    first_seen_at TEXT NOT NULL,
    notified INTEGER NOT NULL DEFAULT 0,
    FOREIGN KEY (watch_id) REFERENCES mercari_watchlist(watch_id) ON DELETE CASCADE,
    UNIQUE (watch_id, mercari_item_id)
);
"""


@dataclass
class MercariWatch:
    watch_id: str
    query: str
    price_threshold_jpy: int
    enabled: bool
    chat_id: str
    last_checked_at: str | None
    created_at: str
    updated_at: str


@dataclass
class MercariHit:
    hit_id: str
    watch_id: str
    mercari_item_id: str
    title: str
    price_jpy: int
    url: str
    thumbnail_url: str | None
    first_seen_at: str
    notified: bool


def _json(data: object) -> str:
    return json.dumps(data, ensure_ascii=False, sort_keys=True)


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
            connection.executescript(SCHEMA)

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

    def add_mercari_watch(self, watch: MercariWatch) -> None:
        timestamp = utc_now().isoformat()
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO mercari_watchlist (
                    watch_id, query, price_threshold_jpy, enabled, chat_id,
                    last_checked_at, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(watch_id) DO UPDATE SET
                    query=excluded.query,
                    price_threshold_jpy=excluded.price_threshold_jpy,
                    enabled=excluded.enabled,
                    chat_id=excluded.chat_id,
                    updated_at=excluded.updated_at
                """,
                (
                    watch.watch_id,
                    watch.query,
                    watch.price_threshold_jpy,
                    int(watch.enabled),
                    watch.chat_id,
                    watch.last_checked_at,
                    timestamp,
                    timestamp,
                ),
            )

    def delete_mercari_watch(self, watch_id: str) -> bool:
        with self.connect() as connection:
            cursor = connection.execute(
                "DELETE FROM mercari_watchlist WHERE watch_id = ?", (watch_id,)
            )
            return cursor.rowcount > 0

    def toggle_mercari_watch(self, watch_id: str, *, enabled: bool) -> bool:
        timestamp = utc_now().isoformat()
        with self.connect() as connection:
            cursor = connection.execute(
                "UPDATE mercari_watchlist SET enabled = ?, updated_at = ? WHERE watch_id = ?",
                (int(enabled), timestamp, watch_id),
            )
            return cursor.rowcount > 0

    def update_mercari_watch(self, watch_id: str, *, query: str | None = None, price_threshold_jpy: int | None = None) -> bool:
        timestamp = utc_now().isoformat()
        sets = []
        params: list[object] = []
        if query is not None:
            sets.append("query = ?")
            params.append(query)
        if price_threshold_jpy is not None:
            sets.append("price_threshold_jpy = ?")
            params.append(price_threshold_jpy)
        if not sets:
            return False
        sets.append("updated_at = ?")
        params.append(timestamp)
        params.append(watch_id)
        with self.connect() as connection:
            cursor = connection.execute(
                f"UPDATE mercari_watchlist SET {', '.join(sets)} WHERE watch_id = ?",  # noqa: S608
                params,
            )
            return cursor.rowcount > 0

    def list_mercari_watchlist(self) -> list[MercariWatch]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM mercari_watchlist ORDER BY created_at DESC"
            ).fetchall()
        return [_row_to_watch(row) for row in rows]

    def get_mercari_watch(self, watch_id: str) -> MercariWatch | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM mercari_watchlist WHERE watch_id = ?", (watch_id,)
            ).fetchone()
        return _row_to_watch(row) if row else None

    def mark_watch_checked(self, watch_id: str) -> None:
        timestamp = utc_now().isoformat()
        with self.connect() as connection:
            connection.execute(
                "UPDATE mercari_watchlist SET last_checked_at = ?, updated_at = ? WHERE watch_id = ?",
                (timestamp, timestamp, watch_id),
            )

    def record_mercari_hits(
        self,
        watch_id: str,
        items: list[dict[str, object]],
    ) -> list[dict[str, object]]:
        """Upsert hits; return items that are NEW or have a PRICE CHANGE.

        Each returned item may carry extra keys:
          _event: "new" | "price_changed"
          _old_price: int  (only when _event == "price_changed")
        """
        now = utc_now().isoformat()
        new_or_changed: list[dict[str, object]] = []
        with self.connect() as connection:
            for item in items:
                mercari_item_id = str(item.get("item_id", ""))
                if not mercari_item_id:
                    continue
                hit_id = sha1(f"{watch_id}|{mercari_item_id}".encode()).hexdigest()
                price_jpy = int(item.get("price_jpy", 0))
                title = str(item.get("title", ""))
                url = str(item.get("url", ""))
                thumbnail = item.get("thumbnail_url")

                existing = connection.execute(
                    "SELECT price_jpy FROM mercari_watch_hits WHERE watch_id = ? AND mercari_item_id = ?",
                    (watch_id, mercari_item_id),
                ).fetchone()

                if existing is None:
                    # Brand-new item
                    try:
                        connection.execute(
                            """
                            INSERT INTO mercari_watch_hits
                                (hit_id, watch_id, mercari_item_id, title, price_jpy, url,
                                 thumbnail_url, first_seen_at, notified)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0)
                            """,
                            (hit_id, watch_id, mercari_item_id, title, price_jpy, url, thumbnail, now),
                        )
                        new_or_changed.append({**item, "_event": "new"})
                    except sqlite3.IntegrityError:
                        pass  # race condition; skip
                elif int(existing["price_jpy"]) != price_jpy:
                    # Same item, price changed → reset notified so it fires again
                    old_price = int(existing["price_jpy"])
                    connection.execute(
                        """
                        UPDATE mercari_watch_hits
                        SET price_jpy = ?, title = ?, thumbnail_url = ?,
                            first_seen_at = ?, notified = 0
                        WHERE watch_id = ? AND mercari_item_id = ?
                        """,
                        (price_jpy, title, thumbnail, now, watch_id, mercari_item_id),
                    )
                    new_or_changed.append({**item, "_event": "price_changed", "_old_price": old_price})
                # else: same price, already handled → skip
        return new_or_changed

    def mark_hits_notified(self, watch_id: str, mercari_item_ids: list[str]) -> None:
        if not mercari_item_ids:
            return
        with self.connect() as connection:
            connection.executemany(
                "UPDATE mercari_watch_hits SET notified = 1 WHERE watch_id = ? AND mercari_item_id = ?",
                [(watch_id, mid) for mid in mercari_item_ids],
            )

    def list_mercari_hits(self, watch_id: str, *, limit: int = 10) -> list[MercariHit]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM mercari_watch_hits
                WHERE watch_id = ?
                ORDER BY first_seen_at DESC
                LIMIT ?
                """,
                (watch_id, limit),
            ).fetchall()
        return [_row_to_hit(row) for row in rows]

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


def _row_to_watch(row: sqlite3.Row) -> MercariWatch:
    return MercariWatch(
        watch_id=row["watch_id"],
        query=row["query"],
        price_threshold_jpy=int(row["price_threshold_jpy"]),
        enabled=bool(row["enabled"]),
        chat_id=row["chat_id"],
        last_checked_at=row["last_checked_at"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _row_to_hit(row: sqlite3.Row) -> MercariHit:
    return MercariHit(
        hit_id=row["hit_id"],
        watch_id=row["watch_id"],
        mercari_item_id=row["mercari_item_id"],
        title=row["title"],
        price_jpy=int(row["price_jpy"]),
        url=row["url"],
        thumbnail_url=row["thumbnail_url"],
        first_seen_at=row["first_seen_at"],
        notified=bool(row["notified"]),
    )
