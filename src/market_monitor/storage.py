from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
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
"""


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
