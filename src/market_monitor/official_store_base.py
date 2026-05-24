"""Protocol and shared helpers for official store pre-order / lottery crawlers.

Adding a new store = (a) write a class that implements OfficialStoreCrawler,
(b) implement ``fetch_listings()`` returning list[OfficialStoreListing],
(c) register it in OfficialStoreCandidateProvider.

Status vocabulary (used in OfficialStoreListing.status):
  "lottery_open"   — 抽選申込受付中 (deadline in the future)
  "lottery_closed" — 抽選終了 / 抽選結果待ち
  "preorder_open"  — 予約受付中
  "preorder_closed"— 予約終了 / 在庫切れ
  "available"      — 通常販売中 (in stock, no lottery needed)
  "coming_soon"    — 発売予定 / 近日公開
  "unknown"        — status could not be determined
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Protocol
from urllib.parse import urljoin

if TYPE_CHECKING:
    from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# All recognised status codes so callers can compare without magic strings.
LOTTERY_OPEN = "lottery_open"
LOTTERY_CLOSED = "lottery_closed"
PREORDER_OPEN = "preorder_open"
PREORDER_CLOSED = "preorder_closed"
AVAILABLE = "available"
COMING_SOON = "coming_soon"
STATUS_UNKNOWN = "unknown"

ACTIVE_STATUSES = frozenset({LOTTERY_OPEN, PREORDER_OPEN, AVAILABLE})


@dataclass(frozen=True)
class OfficialStoreListing:
    """Normalised listing returned by every OfficialStoreCrawler.

    ``item_key`` is a stable, store-scoped identifier derived from the URL or
    product code. Used for dedup / change-detection between runs."""

    store_name: str           # e.g. "joshin", "yodobashi", "pokecen"
    item_key: str             # stable id within this store
    title: str
    url: str
    status: str               # one of the STATUS_* constants above
    price_jpy: int | None = None
    deadline_iso: str | None = None   # ISO8601 if extractable
    open_date_iso: str | None = None  # when lottery/preorder opens
    product_code: str | None = None
    categories: tuple[str, ...] = field(default_factory=tuple)
    crawled_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    )


class OfficialStoreCrawler(Protocol):
    """Protocol every official-store pre-order/lottery crawler implements."""

    store_name: str

    def fetch_listings(
        self,
        *,
        timeout_seconds: int = 30,
    ) -> list[OfficialStoreListing]: ...


# ── Japanese date parsing helpers ───────────────────────────────────────────

# Matches patterns like "6月1日", "2026年6月1日", "6/1", "2026/6/1"
_JP_DATE_RE = re.compile(
    r"(?:(\d{4})年\s*)?(\d{1,2})月\s*(\d{1,2})日"
    r"|(?:(\d{4})/)?(\d{1,2})/(\d{1,2})"
)
# Matches HH:MM or HH時MM分
_TIME_RE = re.compile(r"(\d{1,2})[:時](\d{2})(?:分)?")


def parse_jp_date(text: str, *, base_year: int | None = None) -> str | None:
    """Parse a Japanese date string into ISO8601 date (no time).

    Returns None if no recognisable pattern found. base_year is used when
    the year is absent from the pattern (defaults to current JST year)."""
    if not text:
        return None
    m = _JP_DATE_RE.search(text)
    if not m:
        return None
    g = m.groups()
    if g[0] is not None:
        year, month, day = int(g[0]), int(g[1]), int(g[2])
    elif g[3] is not None:
        year, month, day = int(g[3]), int(g[4]), int(g[5])
    else:
        year = base_year or _current_jst_year()
        # g[1]/g[2] for 年月日, g[4]/g[5] for slash form without year
        month = int(g[1] or g[4])
        day = int(g[2] or g[5])
    try:
        return f"{year:04d}-{month:02d}-{day:02d}"
    except (ValueError, TypeError):
        return None


def parse_jp_datetime(text: str, *, base_year: int | None = None) -> str | None:
    """Parse a Japanese datetime string into ISO8601 (JST assumed).

    Returns None if date not found. If time is present, appends T + time + +09:00."""
    date_part = parse_jp_date(text, base_year=base_year)
    if not date_part:
        return None
    tm = _TIME_RE.search(text)
    if tm:
        hour, minute = int(tm.group(1)), int(tm.group(2))
        return f"{date_part}T{hour:02d}:{minute:02d}:00+09:00"
    return date_part


def _current_jst_year() -> int:
    # JST = UTC+9 — close enough; within 9h of rollover it may be off by 1.
    from datetime import timezone, timedelta
    jst = timezone(timedelta(hours=9))
    return datetime.now(jst).year


def _build_jst_iso(
    *, year: int | None, month: int, day: int,
    hour: int | None, minute: int | None,
) -> str:
    """Build an ISO8601 string in JST from parsed date/time components."""
    y = year or _current_jst_year()
    if hour is not None and minute is not None:
        return f"{y:04d}-{month:02d}-{day:02d}T{hour:02d}:{minute:02d}:00+09:00"
    return f"{y:04d}-{month:02d}-{day:02d}"


# ── HTML fetch helper ────────────────────────────────────────────────────────

def fetch_soup(
    url: str,
    *,
    http_client: "HttpClientProtocol",
    timeout_seconds: int = 30,
) -> "BeautifulSoup":
    """Fetch URL and parse with BeautifulSoup (html.parser).

    Raises on network/HTTP errors — callers should catch and log."""
    from bs4 import BeautifulSoup
    html = http_client.get_text(url, timeout_seconds=timeout_seconds)
    return BeautifulSoup(html, "html.parser")


class HttpClientProtocol(Protocol):
    def get_text(
        self,
        url: str,
        *,
        timeout_seconds: int = 20,
        headers: "dict[str, str] | None" = None,
    ) -> str: ...


# ── URL normalisation helper ─────────────────────────────────────────────────

def abs_url(base: str, href: str) -> str:
    """Return absolute URL, joining relative hrefs against base."""
    return urljoin(base, href)


# ── Keyword helpers for status inference ────────────────────────────────────

_LOTTERY_OPEN_KW = ("抽選申込", "抽選受付", "抽選販売受付", "申し込み受付中", "エントリー受付中")
_LOTTERY_CLOSED_KW = ("抽選終了", "抽選受付終了", "抽選結果", "当選", "落選")
_PREORDER_OPEN_KW = ("予約受付中", "予約販売", "ご予約", "予約はこちら")
_PREORDER_CLOSED_KW = ("予約終了", "予約受付終了", "在庫なし", "売り切れ", "完売", "SOLD OUT")
_COMING_SOON_KW = ("近日公開", "発売予定", "coming soon", "受付開始前")


def infer_status(text: str) -> str:
    """Heuristic: classify a page / badge text into a STATUS_* constant."""
    t = text.lower()
    for kw in _LOTTERY_CLOSED_KW:
        if kw.lower() in t:
            return LOTTERY_CLOSED
    for kw in _LOTTERY_OPEN_KW:
        if kw.lower() in t:
            return LOTTERY_OPEN
    for kw in _PREORDER_CLOSED_KW:
        if kw.lower() in t:
            return PREORDER_CLOSED
    for kw in _PREORDER_OPEN_KW:
        if kw.lower() in t:
            return PREORDER_OPEN
    for kw in _COMING_SOON_KW:
        if kw.lower() in t:
            return COMING_SOON
    return STATUS_UNKNOWN


def item_key_from_url(url: str) -> str:
    """Derive a stable store-scoped item key from a product URL.

    Strips query string and fragment, keeps path so it's unique per SKU."""
    from urllib.parse import urlparse
    parsed = urlparse(url)
    return f"{parsed.netloc}{parsed.path}".rstrip("/")
