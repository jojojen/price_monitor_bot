"""Shared Host Budget state & policy registry (issue #22).

#19 fixed the Yuyutei rate-limit amplification with a tempdir cooldown marker.
That hotfix is fine as a stopgap but is not a durable host-traffic coordination
layer. This module is the foundation it grows into: a SQLite-backed, inspectable
store of per-host *policies*, *cooldowns*, and *request events* that is shared
across the several independent OpenClaw processes on one machine (the
opportunity-agent, the Telegram /research worker, scrape subprocesses) and
survives process restarts.

This issue is intentionally state + API only. It does NOT integrate HttpClient,
change crawler behavior, implement queueing, or replace the #19 hotfix — those
are #24/#25. Future request scheduling can depend on the API defined here.
"""
from __future__ import annotations

import logging
import os
import sqlite3
import tempfile
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from hashlib import sha1
from pathlib import Path
from typing import Callable, Iterator, Mapping
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# ── Deliverable 1: host policy model ──────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class HostPolicy:
    host: str
    requests_per_minute: int
    min_interval_seconds: float
    max_concurrency: int
    cooldown_seconds: int
    enabled: bool = True


# A deliberately conservative fallback for hosts we have no explicit policy for:
# slow, low-concurrency, with a real cooldown so an unknown marketplace can't be
# hammered just because it isn't in the registry.
DEFAULT_HOST_POLICY = HostPolicy(
    host="__default__",
    requests_per_minute=30,
    min_interval_seconds=2.0,
    max_concurrency=2,
    cooldown_seconds=120,
    enabled=True,
)

YUYUTEI_HOST = "yuyu-tei.jp"

# Built-in policies. Yuyutei is the #19 amplification source — keep it strict:
# one request at a time, ≥10s apart, 5-minute cooldown after a 429.
DEFAULT_POLICIES: dict[str, HostPolicy] = {
    YUYUTEI_HOST: HostPolicy(
        host=YUYUTEI_HOST,
        requests_per_minute=6,
        min_interval_seconds=10.0,
        max_concurrency=1,
        cooldown_seconds=300,
        enabled=True,
    ),
}


def normalize_host(value: str) -> str:
    """Lowercase bare hostname. Accepts a full URL or a bare host; mirrors the
    #19 ``_host_of`` so cooldown state lines up across both layers."""
    if not value:
        return "unknown"
    raw = value.strip()
    host = urlparse(raw).hostname if "//" in raw or raw.startswith("http") else None
    return (host or raw).lower()


def policy_for(host: str) -> HostPolicy:
    """Built-in policy for a host, or the safe default stamped with the host."""
    h = normalize_host(host)
    known = DEFAULT_POLICIES.get(h)
    return known if known is not None else replace(DEFAULT_HOST_POLICY, host=h)


# ── Deliverable 3/4: cooldown + request-event records ─────────────────────────
@dataclass(frozen=True, slots=True)
class HostCooldown:
    host: str
    expires_at: float          # wall-clock epoch seconds (cross-process safe)
    reason: str | None = None
    requester: str | None = None
    last_status: int | None = None
    tripped_at: str | None = None

    @property
    def remaining_seconds(self) -> float:
        return max(0.0, self.expires_at - time.time())

    @property
    def active(self) -> bool:
        return self.remaining_seconds > 0.0


@dataclass(frozen=True, slots=True)
class RequestEvent:
    event_id: str
    host: str
    requester: str | None
    priority: str | None
    url_hash: str | None
    decision: str
    wait_seconds: float
    reason: str | None
    created_at: str


# ── request-decision vocabulary (Deliverable 4) ───────────────────────────────
DECISION_GRANTED = "granted"
DECISION_WAITED_THEN_GRANTED = "waited_then_granted"
DECISION_SKIPPED_COOLING_DOWN = "skipped_cooling_down"
DECISION_SKIPPED_BUDGET_EXHAUSTED = "skipped_budget_exhausted"
DECISION_SKIPPED_CONCURRENCY_LIMIT = "skipped_concurrency_limit"
DECISION_LIVE_429 = "live_429"
DECISION_MANUAL_WAIT_TIMEOUT = "manual_wait_timeout"
DECISIONS: tuple[str, ...] = (
    DECISION_GRANTED, DECISION_WAITED_THEN_GRANTED, DECISION_SKIPPED_COOLING_DOWN,
    DECISION_SKIPPED_BUDGET_EXHAUSTED, DECISION_SKIPPED_CONCURRENCY_LIMIT,
    DECISION_LIVE_429, DECISION_MANUAL_WAIT_TIMEOUT,
)


def default_host_budget_path() -> Path:
    """Shared default store path. Honors ``OPENCLAW_HOST_BUDGET_DB`` (so the path
    is configurable per deployment / isolated in tests); otherwise lives in the
    system temp dir so every OpenClaw process on the host opens the *same* SQLite
    file (matching #19's cross-process model) and it survives process restarts."""
    env = os.environ.get("OPENCLAW_HOST_BUDGET_DB")
    if env:
        return Path(env)
    return Path(tempfile.gettempdir()) / "openclaw_host_budget.sqlite3"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _hash_url(url: str | None) -> str | None:
    """Store URLs as a short hash only — never the raw full URL (Deliverable 4)."""
    if not url:
        return None
    return sha1(url.encode("utf-8")).hexdigest()[:16]


_SCHEMA = """
CREATE TABLE IF NOT EXISTS host_policies (
    host                 TEXT PRIMARY KEY,
    requests_per_minute  INTEGER NOT NULL,
    min_interval_seconds REAL NOT NULL,
    max_concurrency      INTEGER NOT NULL,
    cooldown_seconds     INTEGER NOT NULL,
    enabled              INTEGER NOT NULL DEFAULT 1,
    updated_at           TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS host_cooldowns (
    host        TEXT PRIMARY KEY,
    expires_at  REAL NOT NULL,
    reason      TEXT,
    requester   TEXT,
    last_status INTEGER,
    tripped_at  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS host_request_events (
    event_id     TEXT PRIMARY KEY,
    host         TEXT NOT NULL,
    requester    TEXT,
    priority     TEXT,
    url_hash     TEXT,
    decision     TEXT NOT NULL,
    wait_seconds REAL NOT NULL DEFAULT 0,
    reason       TEXT,
    created_at   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_host_events_host ON host_request_events(host);
CREATE INDEX IF NOT EXISTS idx_host_events_created ON host_request_events(created_at);
"""


# ── Deliverable 2: SQLite host-budget store ───────────────────────────────────
class HostBudgetStore:
    """Durable, cross-process host-budget state. WAL + wall-clock expiries make
    the cooldown visible to peer processes; the schema bootstrap is idempotent so
    every process can open the same file safely."""

    def __init__(self, path: str | Path | None = None, *, seed_defaults: bool = True) -> None:
        self.path = Path(path) if path is not None else default_host_budget_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.bootstrap()
        if seed_defaults:
            self.seed_default_policies()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def bootstrap(self) -> None:
        with self.connect() as conn:
            conn.executescript(_SCHEMA)

    # ── Deliverable 1: policy load / lookup ───────────────────────────────────
    def seed_default_policies(self) -> None:
        """Persist the built-in policies if absent (idempotent). Existing rows —
        which may have been hand-tuned — are left untouched."""
        for policy in DEFAULT_POLICIES.values():
            if self._get_policy_row(policy.host) is None:
                self.upsert_policy(policy)

    def upsert_policy(self, policy: HostPolicy) -> HostPolicy:
        host = normalize_host(policy.host)
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO host_policies (
                    host, requests_per_minute, min_interval_seconds,
                    max_concurrency, cooldown_seconds, enabled, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(host) DO UPDATE SET
                    requests_per_minute = excluded.requests_per_minute,
                    min_interval_seconds = excluded.min_interval_seconds,
                    max_concurrency = excluded.max_concurrency,
                    cooldown_seconds = excluded.cooldown_seconds,
                    enabled = excluded.enabled,
                    updated_at = excluded.updated_at
                """,
                (host, int(policy.requests_per_minute), float(policy.min_interval_seconds),
                 int(policy.max_concurrency), int(policy.cooldown_seconds),
                 1 if policy.enabled else 0, _utc_now_iso()),
            )
        return replace(policy, host=host)

    def _get_policy_row(self, host: str) -> sqlite3.Row | None:
        with self.connect() as conn:
            return conn.execute(
                "SELECT * FROM host_policies WHERE host = ?", (normalize_host(host),)
            ).fetchone()

    def get_policy(self, host: str) -> HostPolicy:
        """Stored policy for a host; falls back to the built-in default registry,
        then the conservative ``DEFAULT_HOST_POLICY`` (never raises)."""
        row = self._get_policy_row(host)
        if row is None:
            return policy_for(host)
        return _row_to_policy(row)

    # ── Deliverable 3: cooldown API ───────────────────────────────────────────
    def get_host_cooldown(self, host: str) -> HostCooldown | None:
        """Active cooldown for a host, or ``None`` if absent/expired. Expired rows
        are reported as ``None`` but left for ``clear_expired_cooldowns`` to reap,
        so a read never deletes state a peer process might still inspect."""
        h = normalize_host(host)
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM host_cooldowns WHERE host = ?", (h,)
            ).fetchone()
        if row is None:
            return None
        cooldown = _row_to_cooldown(row)
        return cooldown if cooldown.active else None

    def trip_host_cooldown(
        self,
        host: str,
        *,
        reason: str | None = None,
        requester: str | None = None,
        cooldown_seconds: float | None = None,
        last_status: int | None = None,
    ) -> HostCooldown:
        """Open (or extend) a host's cooldown. A shorter cooldown NEVER overwrites
        a longer active one — like the #19 marker, we always bias toward backing
        off. Records who tripped it, why, and the optional last HTTP status."""
        h = normalize_host(host)
        secs = cooldown_seconds if cooldown_seconds is not None else self.get_policy(h).cooldown_seconds
        new_expiry = time.time() + max(0.0, float(secs))
        now_iso = _utc_now_iso()
        with self.connect() as conn:
            existing = conn.execute(
                "SELECT expires_at FROM host_cooldowns WHERE host = ?", (h,)
            ).fetchone()
            if existing is not None and float(existing["expires_at"]) >= new_expiry:
                # A longer (or equal) cooldown already stands — don't shorten it.
                row = conn.execute(
                    "SELECT * FROM host_cooldowns WHERE host = ?", (h,)
                ).fetchone()
                return _row_to_cooldown(row)
            conn.execute(
                """
                INSERT INTO host_cooldowns (
                    host, expires_at, reason, requester, last_status, tripped_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(host) DO UPDATE SET
                    expires_at = excluded.expires_at,
                    reason = excluded.reason,
                    requester = excluded.requester,
                    last_status = excluded.last_status,
                    tripped_at = excluded.tripped_at
                """,
                (h, new_expiry, reason, requester, last_status, now_iso),
            )
        return HostCooldown(host=h, expires_at=new_expiry, reason=reason,
                            requester=requester, last_status=last_status, tripped_at=now_iso)

    def clear_expired_cooldowns(self) -> int:
        """Delete cooldown rows whose wall-clock expiry has passed. Returns the
        number cleared. Safe to call from any process."""
        with self.connect() as conn:
            cur = conn.execute(
                "DELETE FROM host_cooldowns WHERE expires_at <= ?", (time.time(),)
            )
            return cur.rowcount

    # ── Deliverable 4: request event logging ──────────────────────────────────
    def log_request_event(
        self,
        *,
        host: str,
        decision: str,
        requester: str | None = None,
        priority: str | None = None,
        url: str | None = None,
        wait_seconds: float = 0.0,
        reason: str | None = None,
    ) -> RequestEvent:
        """Append a request decision for diagnostics. The URL is stored as a hash
        only — never the raw full URL."""
        evt = RequestEvent(
            event_id="he_" + uuid.uuid4().hex[:16],
            host=normalize_host(host),
            requester=requester,
            priority=priority,
            url_hash=_hash_url(url),
            decision=decision,
            wait_seconds=float(wait_seconds),
            reason=reason,
            created_at=_utc_now_iso(),
        )
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO host_request_events (
                    event_id, host, requester, priority, url_hash,
                    decision, wait_seconds, reason, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (evt.event_id, evt.host, evt.requester, evt.priority, evt.url_hash,
                 evt.decision, evt.wait_seconds, evt.reason, evt.created_at),
            )
        return evt

    def recent_events(
        self, *, host: str | None = None, limit: int = 100
    ) -> list[RequestEvent]:
        clauses, params = [], []
        if host:
            clauses.append("host = ?")
            params.append(normalize_host(host))
        sql = "SELECT * FROM host_request_events"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY created_at DESC, event_id LIMIT ?"
        params.append(int(limit))
        with self.connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [_row_to_event(r) for r in rows]


def _row_to_policy(row: Mapping) -> HostPolicy:
    return HostPolicy(
        host=row["host"],
        requests_per_minute=row["requests_per_minute"],
        min_interval_seconds=row["min_interval_seconds"],
        max_concurrency=row["max_concurrency"],
        cooldown_seconds=row["cooldown_seconds"],
        enabled=bool(row["enabled"]),
    )


def _row_to_cooldown(row: Mapping) -> HostCooldown:
    return HostCooldown(
        host=row["host"],
        expires_at=float(row["expires_at"]),
        reason=row["reason"],
        requester=row["requester"],
        last_status=row["last_status"],
        tripped_at=row["tripped_at"],
    )


def _row_to_event(row: Mapping) -> RequestEvent:
    return RequestEvent(
        event_id=row["event_id"],
        host=row["host"],
        requester=row["requester"],
        priority=row["priority"],
        url_hash=row["url_hash"],
        decision=row["decision"],
        wait_seconds=row["wait_seconds"],
        reason=row["reason"],
        created_at=row["created_at"],
    )


# ══════════════════════════════════════════════════════════════════════════════
# Runtime coordinator (issue #24): the layer HttpClient asks before a live fetch.
# State lives in HostBudgetStore (#22, cross-process); concurrency is gated by an
# in-process per-host semaphore (one process's ThreadPool can't exceed a host's
# policy). Priority handling is deliberately minimal here — the richer scheduling
# (reserved manual capacity, bounded queueing) is layered on in #25.
# ══════════════════════════════════════════════════════════════════════════════

# Stable requester identities (Deliverable 2).
REQUESTER_RESEARCH = "research_command"
REQUESTER_HOT_CARDS = "tcg_hot_cards"
REQUESTER_OPPORTUNITY = "opportunity_agent"
REQUESTER_CACHE_REFRESH = "cache_refresh"
REQUESTER_MANUAL_DEBUG = "manual_debug"

# Priority classes. Manual classes may wait briefly for a slot; everything else
# fails fast so background fan-out can never starve a manual request.
PRIORITY_MANUAL_RESEARCH = "manual_research"
PRIORITY_USER_COMMAND = "user_command"
PRIORITY_SCHEDULED_OPPORTUNITY = "scheduled_opportunity"
PRIORITY_BACKGROUND_ENRICHMENT = "background_enrichment"
PRIORITY_CACHE_REFRESH = "cache_refresh"
PRIORITIES: tuple[str, ...] = (
    PRIORITY_MANUAL_RESEARCH, PRIORITY_USER_COMMAND, PRIORITY_SCHEDULED_OPPORTUNITY,
    PRIORITY_BACKGROUND_ENRICHMENT, PRIORITY_CACHE_REFRESH,
)
# Unknown priority falls back to the lowest (background) — never to a manual one.
DEFAULT_PRIORITY = PRIORITY_BACKGROUND_ENRICHMENT
_MANUAL_PRIORITIES: frozenset[str] = frozenset({PRIORITY_MANUAL_RESEARCH, PRIORITY_USER_COMMAND})

# Hard cap on how long a manual caller may block for a concurrency slot, so a
# user command can never hang behind a stuck background fetch. Configurable per
# HostBudget instance (#25 D4); this is the safe default.
MANUAL_WAIT_CAP_SEC = 5.0


def normalize_priority(value: str | None) -> str:
    return value if value in PRIORITIES else DEFAULT_PRIORITY


def reserved_manual_slots(max_concurrency: int) -> int:
    """How many of a host's concurrency slots are held back for manual callers
    (#25 D3). Reserve one slot when the host allows ≥2 concurrent requests so a
    background fan-out can never consume *all* capacity and starve /research. A
    single-slot host (e.g. Yuyutei) can't reserve without blocking background
    entirely, so it reserves none — background still runs but fails fast, leaving
    the slot free the instant a manual caller wants it."""
    return 1 if int(max_concurrency) >= 2 else 0


@dataclass
class FetchPermit:
    """Outcome of an ``acquire_fetch_slot`` call. When ``granted`` is True the
    caller MUST call ``release()`` (or use it as a context manager) after the
    request so the host's concurrency slot is returned."""
    granted: bool
    host: str
    decision: str
    reason: str | None = None
    wait_seconds: float = 0.0
    _release: Callable[[], None] | None = field(default=None, repr=False)

    def release(self) -> None:
        if self._release is not None:
            try:
                self._release()
            finally:
                self._release = None

    def __enter__(self) -> "FetchPermit":
        return self

    def __exit__(self, *exc: object) -> None:
        self.release()


class HostBudget:
    """Runtime gate over a ``HostBudgetStore``. Fails OPEN: any internal error in
    the budget bookkeeping grants the request rather than blocking crawling."""

    def __init__(
        self, store: HostBudgetStore, *, manual_wait_cap_seconds: float = MANUAL_WAIT_CAP_SEC,
    ) -> None:
        self.store = store
        # Configurable bound on how long a manual caller may queue for a slot
        # (#25 D4). Background callers never queue (they fail fast), so this is
        # the only place waiting happens — keeping the queue bounded by design.
        self.manual_wait_cap_seconds = float(manual_wait_cap_seconds)
        self._sema_lock = threading.Lock()
        # Per host: (host_sema capping total concurrency, bg_sema capping how many
        # of those slots background priorities may hold — the rest are reserved
        # for manual callers, #25 D3).
        self._semaphores: dict[str, tuple[threading.BoundedSemaphore, threading.BoundedSemaphore]] = {}
        self._policy_cache: dict[str, HostPolicy] = {}

    # concurrency is enforced only for hosts we have an explicit policy for, so
    # untargeted crawls keep their existing behavior (the universal protection is
    # the cooldown, which every host gets).
    def _is_known(self, host: str) -> bool:
        if host in DEFAULT_POLICIES:
            return True
        return self.store._get_policy_row(host) is not None

    def _policy(self, host: str) -> HostPolicy:
        cached = self._policy_cache.get(host)
        if cached is None:
            cached = self.store.get_policy(host)
            self._policy_cache[host] = cached
        return cached

    def _semaphores_for(
        self, host: str, max_concurrency: int
    ) -> tuple[threading.BoundedSemaphore, threading.BoundedSemaphore]:
        with self._sema_lock:
            pair = self._semaphores.get(host)
            if pair is None:
                total = max(1, int(max_concurrency))
                bg_slots = max(1, total - reserved_manual_slots(total))
                pair = (threading.BoundedSemaphore(total), threading.BoundedSemaphore(bg_slots))
                self._semaphores[host] = pair
            return pair

    def acquire_fetch_slot(
        self,
        *,
        url: str,
        requester: str | None = None,
        priority: str | None = None,
        timeout_seconds: float | None = None,
    ) -> FetchPermit:
        host = normalize_host(url)
        prio = normalize_priority(priority)
        try:
            policy = self._policy(host)
            if not policy.enabled:
                self._log(host, DECISION_SKIPPED_BUDGET_EXHAUSTED, requester, prio, url,
                          reason="host disabled by policy")
                return FetchPermit(False, host, DECISION_SKIPPED_BUDGET_EXHAUSTED,
                                   "host disabled by policy")

            # Cooldown blocks EVERY priority (cross-process; the #19 generalization).
            cooldown = self.store.get_host_cooldown(host)
            if cooldown is not None and cooldown.active:
                self._log(host, DECISION_SKIPPED_COOLING_DOWN, requester, prio, url,
                          reason=cooldown.reason or "cooling down")
                return FetchPermit(False, host, DECISION_SKIPPED_COOLING_DOWN,
                                   cooldown.reason or "cooling down", cooldown.remaining_seconds)

            if not self._is_known(host):
                # Untargeted host: cooldown already checked; don't cap concurrency.
                self._log(host, DECISION_GRANTED, requester, prio, url)
                return FetchPermit(True, host, DECISION_GRANTED)

            host_sema, bg_sema = self._semaphores_for(host, policy.max_concurrency)
            if prio in _MANUAL_PRIORITIES:
                # Manual callers may queue briefly for ANY of the host's slots,
                # including the one reserved away from background (#25 D3/D4).
                cap = self.manual_wait_cap_seconds
                if timeout_seconds is not None:
                    cap = min(cap, float(timeout_seconds))
                start = time.monotonic()
                acquired = host_sema.acquire(timeout=cap) if cap > 0 else host_sema.acquire(blocking=False)
                waited = time.monotonic() - start
                if not acquired:
                    self._log(host, DECISION_MANUAL_WAIT_TIMEOUT, requester, prio, url,
                              wait_seconds=waited, reason="no concurrency slot before timeout")
                    return FetchPermit(False, host, DECISION_MANUAL_WAIT_TIMEOUT,
                                       "no concurrency slot before timeout", waited)
                decision = DECISION_WAITED_THEN_GRANTED if waited > 0.05 else DECISION_GRANTED
                self._log(host, decision, requester, prio, url, wait_seconds=waited)
                return FetchPermit(True, host, decision, None, waited, host_sema.release)

            # Background priorities never queue (no unbounded growth, #25 D4) and
            # may only use the un-reserved slots: take the background limiter
            # first, then a real host slot — releasing both on failure so manual
            # capacity stays intact.
            if not bg_sema.acquire(blocking=False):
                self._log(host, DECISION_SKIPPED_CONCURRENCY_LIMIT, requester, prio, url,
                          reason="manual capacity reserved")
                return FetchPermit(False, host, DECISION_SKIPPED_CONCURRENCY_LIMIT,
                                   "manual capacity reserved")
            if not host_sema.acquire(blocking=False):
                bg_sema.release()
                self._log(host, DECISION_SKIPPED_CONCURRENCY_LIMIT, requester, prio, url,
                          reason=f"at max_concurrency={policy.max_concurrency}")
                return FetchPermit(False, host, DECISION_SKIPPED_CONCURRENCY_LIMIT,
                                   f"at max_concurrency={policy.max_concurrency}")

            def _release_both() -> None:
                host_sema.release()
                bg_sema.release()

            self._log(host, DECISION_GRANTED, requester, prio, url)
            return FetchPermit(True, host, DECISION_GRANTED, None, 0.0, _release_both)
        except Exception:  # never let budget bookkeeping break a fetch
            logger.debug("host budget acquire failed open host=%s", host, exc_info=True)
            return FetchPermit(True, host, DECISION_GRANTED, "budget-error-fail-open")

    def record_result(
        self,
        *,
        url: str,
        status: int | None = None,
        requester: str | None = None,
        retry_after_seconds: float | None = None,
    ) -> None:
        """Feed a request outcome back. A live 429 trips the durable cooldown (so
        peer processes back off) and is logged for diagnostics."""
        if status != 429:
            return
        host = normalize_host(url)
        try:
            policy = self._policy(host)
            secs = policy.cooldown_seconds
            if retry_after_seconds and retry_after_seconds > secs:
                secs = retry_after_seconds
            self.store.trip_host_cooldown(host, reason="live_429", requester=requester,
                                          cooldown_seconds=secs, last_status=429)
            self._log(host, DECISION_LIVE_429, requester, None, url, reason="HTTP 429")
        except Exception:
            logger.debug("host budget record_result failed host=%s", host, exc_info=True)

    def _log(
        self, host: str, decision: str, requester: str | None, priority: str | None,
        url: str | None, *, wait_seconds: float = 0.0, reason: str | None = None,
    ) -> None:
        # Structured diagnostics line (#25 D5): every decision carries requester,
        # priority, host, decision, wait_seconds — enough for a future dashboard
        # to answer "why was this request skipped?". Skips log at INFO so they're
        # visible; grants stay at DEBUG to avoid noise on the happy path.
        level = logging.DEBUG if decision in (DECISION_GRANTED, DECISION_WAITED_THEN_GRANTED) else logging.INFO
        logger.log(
            level,
            "host-budget decision host=%s requester=%s priority=%s decision=%s wait_seconds=%.3f reason=%s",
            host, requester, priority, decision, wait_seconds, reason,
        )
        try:
            self.store.log_request_event(
                host=host, decision=decision, requester=requester, priority=priority,
                url=url, wait_seconds=wait_seconds, reason=reason,
            )
        except Exception:
            logger.debug("host budget event log failed host=%s decision=%s", host, decision,
                         exc_info=True)

    # ── Deliverable 5: diagnostics / observability ────────────────────────────
    def recent_decisions(
        self, *, host: str | None = None, limit: int = 100
    ) -> list[RequestEvent]:
        """Recent scheduling decisions (newest first), for inspection / a future
        dashboard. Thin pass-through to the durable event log."""
        return self.store.recent_events(host=host, limit=limit)

    def decision_summary(
        self, *, host: str | None = None, limit: int = 500
    ) -> dict[str, int]:
        """Counts of recent decisions keyed by decision type — a compact health
        snapshot (e.g. how often manual callers time out, how often a host is
        cooling down). Dashboard-friendly aggregate over the event log."""
        summary: dict[str, int] = {}
        for evt in self.store.recent_events(host=host, limit=limit):
            summary[evt.decision] = summary.get(evt.decision, 0) + 1
        return summary


# ── process-wide singleton (HttpClient + crawlers share one coordinator) ──────
_global_budget: HostBudget | None = None
_global_budget_lock = threading.Lock()


def get_host_budget() -> HostBudget:
    global _global_budget
    if _global_budget is None:
        with _global_budget_lock:
            if _global_budget is None:
                _global_budget = HostBudget(HostBudgetStore())
    return _global_budget


def set_host_budget(budget: HostBudget | None) -> None:
    """Override the process-wide coordinator (tests / custom wiring)."""
    global _global_budget
    with _global_budget_lock:
        _global_budget = budget


def reset_host_budget() -> None:
    set_host_budget(None)
