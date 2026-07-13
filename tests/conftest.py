"""Pytest conftest for live image-lookup regression tests.

Pre-populates `card_image_fingerprints` with one row per fixture case so the
regression suite can verify the fast-path round-trips known images
correctly. This decouples the suite from:

  - live network (yuyutei 429s, snkrdunk reachability)
  - the local Ollama vision LLM (slow, model-version-dependent)
  - the crawler running and having populated the cache

The test then exercises:
  1. dHash compute on the fixture image
  2. Nearest-neighbour scan in card_image_fingerprints
  3. Return-from-cache fast-path in TcgImagePriceService.parse_image
  4. Spec → parsed normalization (parenthetical stripping, promo-rarity expansion)
  5. Status / partial demotion logic

Cases without expected `parsed.title` etc. (e.g. partial-pikachu) are
intentionally skipped — those test the OCR-only / unresolved-detection
paths and should NOT have a pre-warmed cache entry.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from pathlib import Path

import pytest

from market_monitor.host_budget import reset_host_budget
from market_monitor.models import CardImageFingerprint, utc_now
from market_monitor.storage import MonitorDatabase
from tcg_tracker.image_fingerprint import DEFAULT_ALGO, compute_dhash
from tests.image_lookup_case_fixtures import iter_image_lookup_live_cases


def _seed_fingerprint_db(db_path: Path) -> int:
    """Insert one fingerprint per regression case so the fast path can
    resolve them deterministically. Returns the number of rows inserted."""
    db = MonitorDatabase(db_path)
    db.bootstrap()
    inserted = 0
    for case in iter_image_lookup_live_cases():
        expected = case.payload.get("expected") or {}
        if not isinstance(expected, dict):
            continue
        parsed = expected.get("parsed")
        lookup_spec = expected.get("lookup_spec") or {}
        if not isinstance(parsed, dict):
            continue
        # Only seed cases that have full enough metadata to round-trip.
        # Cases like partial-pikachu deliberately have minimal expected
        # fields — those are testing the OCR-failure / partial-status path
        # and must NOT hit the fast path.
        if expected.get("status") != "success":
            continue
        # Cases checking only game / item_kind (sealed boxes) use
        # title_contains_any — we still seed those because the fast path
        # populates a title from the fingerprint record.
        title = lookup_spec.get("title") or parsed.get("title")
        if not title:
            # Allow sealed boxes where the title comes from title_contains_any
            tca = expected.get("title_contains_any")
            if not isinstance(tca, list) or not tca:
                continue
            title = tca[0]  # first listed substring as the canonical title
        target_hash = compute_dhash(image_path=case.image_path)
        if not target_hash:
            continue
        fp = CardImageFingerprint(
            fingerprint_id=MonitorDatabase._fingerprint_id(target_hash, DEFAULT_ALGO),
            game=parsed.get("game") or lookup_spec.get("game") or "pokemon",
            item_kind=parsed.get("item_kind") or lookup_spec.get("item_kind") or "card",
            title=title,
            card_number=lookup_spec.get("card_number") or parsed.get("card_number"),
            rarity=lookup_spec.get("rarity") or parsed.get("rarity"),
            set_code=lookup_spec.get("set_code") or parsed.get("set_code"),
            source_url=f"fixture://{case.case_id}",
            image_url=f"fixture://{case.case_id}/image",
            perceptual_hash=target_hash,
            fingerprint_algo=DEFAULT_ALGO,
            confidence_source="verified",
            captured_at=utc_now(),
            last_seen_at=utc_now(),
        )
        db.upsert_card_image_fingerprint(fp)
        inserted += 1
    return inserted


@pytest.fixture(scope="session")
def regression_db_path(tmp_path_factory) -> Path:
    """Session-scoped SQLite path with pre-seeded fingerprints. Used by the
    live image-lookup regression suite to give the fast path deterministic
    metadata for every known fixture case."""
    db_path = tmp_path_factory.mktemp("image_lookup_db") / "regression.sqlite3"
    _seed_fingerprint_db(db_path)
    return db_path


# Allow live-regression to override db_path via the env var — easier than
# threading a fixture through @parametrize.
@pytest.fixture(scope="session", autouse=True)
def _wire_regression_db_path(regression_db_path: Path) -> None:
    os.environ.setdefault("OPENCLAW_REGRESSION_DB_PATH", str(regression_db_path))


# Thread names owned by the poll-loop machinery (telegram_core.polling). Tests
# must not leave any of these alive after they finish — a leaked "poll-watchdog"
# keeps logging "poll loop likely wedged" into later tests (issue #3).
_PROJECT_WORKER_THREAD_NAMES = ("poll-watchdog", "hb-beacon")


def _project_worker_threads() -> list[str]:
    return [
        t.name
        for t in threading.enumerate()
        if t.is_alive() and t.name in _PROJECT_WORKER_THREAD_NAMES
    ]


@pytest.fixture
def poll_watchdog_factory():
    """Create PollWatchdog instances that are guaranteed to be stopped+joined.

    Centralizes worker creation so every started watchdog has an owner and a
    cleanup path (issue #3, Phase 2). The teardown runs unconditionally in
    ``finally``, so even a failing/raising test cannot leak the daemon thread.
    """
    from price_monitor_bot.bot import PollWatchdog

    created: list[PollWatchdog] = []

    def _make(**kwargs) -> PollWatchdog:
        wd = PollWatchdog(**kwargs)
        created.append(wd)
        return wd

    try:
        yield _make
    finally:
        for wd in created:
            wd.stop()
            wd.join(timeout=5.0)


@pytest.fixture(scope="session", autouse=True)
def _assert_no_leaked_workers():
    """Session-end guard: no project-owned worker thread may survive the run.

    Scoped to project-owned thread names only (not "all Python threads"), as the
    issue requires, so pytest's own internal threads don't trip it."""
    yield
    # Give any just-stopped thread a brief grace to unwind before asserting.
    deadline = time.monotonic() + 2.0
    while _project_worker_threads() and time.monotonic() < deadline:
        time.sleep(0.05)
    leaked = _project_worker_threads()
    assert not leaked, (
        f"project-owned worker threads survived the test session: {leaked!r} — "
        "a poll watchdog / heartbeat beacon was started without a shutdown path"
    )


@pytest.fixture(autouse=True)
def _isolate_host_budget_db(tmp_path, monkeypatch):
    """Point the #24 host-budget store at a throwaway per-test SQLite file.

    Without this, tests open the shared default `/tmp/openclaw_host_budget.sqlite3`
    that the live stack writes to — a stale cooldown row there can short-circuit
    requests before they reach a test's stubbed network (flaky failures), and
    tests can pollute the live host's budget state. Isolating per test keeps the
    circuit-breaker / budget behaviour hermetic in both directions.
    """
    monkeypatch.setenv("OPENCLAW_HOST_BUDGET_DB", str(tmp_path / "host_budget.sqlite3"))
    reset_host_budget()
    yield
    reset_host_budget()
