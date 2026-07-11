"""Schema version stamping and probe tests for price_monitor_bot.

Verifies:
- bootstrap() stamps PRAGMA user_version=1 on fresh DB (idempotent if already set)
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from market_monitor.storage import MonitorDatabase, SCHEMA_VERSION


def test_bootstrap_stamps_user_version_on_fresh_db(tmp_path: Path) -> None:
    """Fresh DB should have PRAGMA user_version=1 after bootstrap."""
    db = MonitorDatabase(tmp_path / "fresh.db")
    db.bootstrap()
    with sqlite3.connect(db.path) as conn:
        version = conn.execute("PRAGMA user_version").fetchone()[0]
    assert version == 1


def test_bootstrap_idempotent_does_not_downgrade_version(tmp_path: Path) -> None:
    """Calling bootstrap() twice should not downgrade version; stay at 1."""
    db = MonitorDatabase(tmp_path / "twice.db")
    db.bootstrap()
    with sqlite3.connect(db.path) as conn:
        version1 = conn.execute("PRAGMA user_version").fetchone()[0]

    db.bootstrap()  # second call
    with sqlite3.connect(db.path) as conn:
        version2 = conn.execute("PRAGMA user_version").fetchone()[0]

    assert version1 == 1
    assert version2 == 1


def test_bootstrap_preserves_existing_version(tmp_path: Path) -> None:
    """If a DB already has user_version set, bootstrap should not change it."""
    db_path = tmp_path / "existing_version.db"
    # Create DB with explicit user_version=1
    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA user_version = 1")

    db = MonitorDatabase(db_path)
    db.bootstrap()

    with sqlite3.connect(db.path) as conn:
        version = conn.execute("PRAGMA user_version").fetchone()[0]
    assert version == 1


def test_schema_version_constant_is_defined() -> None:
    """SCHEMA_VERSION should be exported from storage module."""
    assert SCHEMA_VERSION == 1
