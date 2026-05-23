"""Background thread that polls every marketplace watch and sends notifications.

Dispatches by ``MarketplaceWatch.source`` to the right ``MarketplaceSearchClient``
(Mercari / Rakuma / future Yuyutei …). The monitor itself stays source-agnostic
— per-source quirks live inside each client.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Callable

from market_monitor.marketplace_search import (
    MarketplaceListing,
    MarketplaceSearchClient,
    MercariSearchClient,
    listing_to_record,
)
from market_monitor.mercari_search import MERCARI_CONDITION_LABELS
from market_monitor.storage import MarketplaceWatch, MonitorDatabase

logger = logging.getLogger(__name__)

NotifyFn = Callable[[str, str], None]  # (chat_id, text)
SnapshotFn = Callable[[str, list[str]], None]  # (chat_id, item_urls)


# (display name, emoji) per source. Adding a new source = one line here.
# `default` is the fallback when a watch references a source we don't have
# a client for; the monitor logs a warning and the notification uses the
# raw source string.
_SOURCE_DISPLAY: dict[str, tuple[str, str]] = {
    "mercari": ("Mercari", "🛒"),
    "rakuma": ("Rakuma", "🟣"),
    # Future:
    # "paypay_fleamarket": ("PayPay フリマ", "💴"),
    # "yahoo_auctions":    ("Yahoo Auctions", "⏰"),
    # "yuyutei":           ("Yuyutei", "📚"),
}


def _source_display(source: str) -> tuple[str, str]:
    return _SOURCE_DISPLAY.get(source, (source.capitalize(), "📦"))


class MarketplaceWatchMonitor:
    def __init__(
        self,
        *,
        db_path: str | Path,
        clients: Mapping[str, MarketplaceSearchClient],
        notify_fn: NotifyFn,
        snapshot_fn: SnapshotFn | None = None,
        interval_seconds: int = 60,
    ) -> None:
        self._db = MonitorDatabase(db_path)
        self._clients = dict(clients)
        self._notify_fn = notify_fn
        self._snapshot_fn = snapshot_fn
        self._interval = interval_seconds
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, name="marketplace-watch-monitor", daemon=True,
        )
        self._thread.start()
        logger.info(
            "MarketplaceWatchMonitor started interval_seconds=%d sources=%s",
            self._interval, sorted(self._clients.keys()),
        )

    def stop(self) -> None:
        self._stop.set()

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _loop(self) -> None:
        # Run immediately on first tick, then wait for interval
        self._tick()
        while not self._stop.wait(self._interval):
            self._tick()

    def _tick(self) -> None:
        try:
            watches = self._db.list_marketplace_watchlist()
        except Exception:
            logger.exception("MarketplaceWatchMonitor: failed to list watchlist")
            return

        enabled = [w for w in watches if w.enabled]
        if not enabled:
            return

        logger.debug("MarketplaceWatchMonitor: checking %d active watches", len(enabled))
        for watch in enabled:
            try:
                self._check_watch(watch)
            except Exception:
                logger.exception(
                    "MarketplaceWatchMonitor: check failed source=%s watch_id=%s query=%s",
                    watch.source, watch.watch_id, watch.query,
                )

    def _check_watch(self, watch: MarketplaceWatch) -> None:
        if not watch.markets:
            logger.warning(
                "MarketplaceWatchMonitor: watch has no markets configured "
                "watch_id=%s query=%s — skipping",
                watch.watch_id, watch.query,
            )
            return
        is_first_scan = watch.last_checked_at is None
        any_market_checked = False
        for market in watch.markets:
            checked = self._check_watch_on_market(
                watch=watch, market=market, is_first_scan=is_first_scan,
            )
            any_market_checked = any_market_checked or checked
        if any_market_checked:
            self._db.mark_watch_checked(watch.watch_id)

    def _check_watch_on_market(
        self, *, watch: MarketplaceWatch, market: str, is_first_scan: bool,
    ) -> bool:
        """Search ``market`` for ``watch.query`` and record any new/changed
        hits. Returns True if the search ran (even with zero hits) so the
        caller knows to update last_checked_at."""
        client = self._clients.get(market)
        if client is None:
            logger.warning(
                "MarketplaceWatchMonitor: no client registered for market=%s "
                "(watch_id=%s query=%s) — skipping",
                market, watch.watch_id, watch.query,
            )
            return False

        options = watch.options_for(market)
        logger.info(
            "MarketplaceWatchMonitor: searching market=%s query=%s price_max=%d watch_id=%s options=%s",
            market, watch.query, watch.price_threshold_jpy,
            watch.watch_id, options,
        )
        try:
            listings = client.search(
                watch.query,
                price_max=watch.price_threshold_jpy,
                source_options=options,
            )
        except Exception:
            logger.exception(
                "MarketplaceWatchMonitor: client search failed market=%s watch_id=%s",
                market, watch.watch_id,
            )
            return False

        records = [listing_to_record(li) for li in listings]
        new_or_changed = self._db.record_marketplace_hits(
            watch_id=watch.watch_id, source=market, items=records,
        )

        if is_first_scan:
            all_ids = [str(r.get("item_id", "")) for r in records if r.get("item_id")]
            self._db.mark_marketplace_hits_notified(
                watch_id=watch.watch_id, source=market, source_item_ids=all_ids,
            )
            logger.info(
                "MarketplaceWatchMonitor: first scan baseline recorded market=%s watch_id=%s items=%d",
                market, watch.watch_id, len(all_ids),
            )
            return True

        if not new_or_changed:
            logger.debug(
                "MarketplaceWatchMonitor: no new or changed items market=%s watch_id=%s",
                market, watch.watch_id,
            )
            return True

        new_count = sum(1 for i in new_or_changed if i.get("_event") != "price_changed")
        changed_count = sum(1 for i in new_or_changed if i.get("_event") == "price_changed")
        logger.info(
            "MarketplaceWatchMonitor: new=%d price_changed=%d market=%s watch_id=%s query=%s",
            new_count, changed_count, market, watch.watch_id, watch.query,
        )

        text = _format_notification(watch=watch, market=market, new_or_changed=new_or_changed)
        try:
            self._notify_fn(watch.chat_id, text)
        except Exception:
            logger.exception(
                "MarketplaceWatchMonitor: notification failed market=%s watch_id=%s chat_id=%s",
                market, watch.watch_id, watch.chat_id,
            )
            return True

        self._db.mark_marketplace_hits_notified(
            watch_id=watch.watch_id,
            source=market,
            source_item_ids=[
                str(item.get("item_id", "")) for item in new_or_changed
                if item.get("item_id")
            ],
        )
        if self._snapshot_fn:
            new_urls = [
                str(item.get("url") or "")
                for item in new_or_changed[:3]
                if item.get("_event") != "price_changed" and item.get("url")
            ]
            if new_urls:
                try:
                    self._snapshot_fn(watch.chat_id, new_urls)
                except Exception:
                    logger.exception(
                        "MarketplaceWatchMonitor: snapshot callback failed market=%s watch_id=%s",
                        market, watch.watch_id,
                    )
        return True


def _summarize_mercari_condition_ids(condition_ids: list[int] | tuple[int, ...] | None) -> str:
    if not condition_ids:
        return "未設定"
    return " / ".join(
        MERCARI_CONDITION_LABELS.get(int(cid), f"ID{cid}") for cid in condition_ids
    )


def _format_notification(
    *,
    watch: MarketplaceWatch,
    market: str,
    new_or_changed: list[dict[str, object]],
) -> str:
    new_count = sum(1 for i in new_or_changed if i.get("_event") != "price_changed")
    changed_count = sum(1 for i in new_or_changed if i.get("_event") == "price_changed")

    summary_parts: list[str] = []
    if new_count:
        summary_parts.append(f"新商品 {new_count} 筆")
    if changed_count:
        summary_parts.append(f"價格更新 {changed_count} 筆")

    display_name, emoji = _source_display(market)

    lines = [
        f"{emoji} {display_name} 通知",
        f"追蹤條件：{watch.query}",
        f"價格上限：¥{watch.price_threshold_jpy:,}",
    ]
    # Mercari-specific: surface the condition filter so the user can verify
    # whether mismatched items are slipping through. Other markets don't
    # have an equivalent today, so we only add this line for Mercari.
    if market == "mercari":
        opts = watch.options_for(market)
        lines.append(f"狀態條件：{_summarize_mercari_condition_ids(opts.get('condition_ids'))}")
    lines.append(f"發現：{' / '.join(summary_parts) if summary_parts else '(無)'}")

    for item in new_or_changed[:5]:
        price = int(item.get("price_jpy") or 0)
        title = str(item.get("title") or "").strip() or "（無標題）"
        url = str(item.get("url") or "")
        event = item.get("_event", "new")
        old_price = item.get("_old_price")
        if event == "price_changed" and isinstance(old_price, int):
            tag = f"[価格更新 ¥{old_price:,}→¥{price:,}]"
        else:
            tag = "[新商品]"
        lines.append(f"・{tag} {title}")
        lines.append(f"  ¥{price:,}  {url}")
    if len(new_or_changed) > 5:
        lines.append(f"  …以及另外 {len(new_or_changed) - 5} 筆")
    return "\n".join(lines)


_monitor_lock = threading.Lock()
_monitor: MarketplaceWatchMonitor | None = None


def default_marketplace_clients() -> dict[str, MarketplaceSearchClient]:
    """The standard set of clients shipped with the bot. Future sources are
    added here so callers don't need to know which clients exist."""
    from market_monitor.rakuma_search import RakumaSearchClient
    return {
        "mercari": MercariSearchClient(),
        "rakuma": RakumaSearchClient(),
    }


def ensure_monitor(
    *,
    db_path: str | Path,
    notify_fn: NotifyFn,
    snapshot_fn: SnapshotFn | None = None,
    interval_seconds: int = 60,
    clients: Mapping[str, MarketplaceSearchClient] | None = None,
) -> tuple[MarketplaceWatchMonitor, bool]:
    """Return the running monitor singleton, starting it if needed.
    Returns (monitor, started_now)."""
    global _monitor
    with _monitor_lock:
        if _monitor is not None and _monitor.is_running():
            return _monitor, False
        _monitor = MarketplaceWatchMonitor(
            db_path=db_path,
            clients=clients if clients is not None else default_marketplace_clients(),
            notify_fn=notify_fn,
            snapshot_fn=snapshot_fn,
            interval_seconds=interval_seconds,
        )
        _monitor.start()
        return _monitor, True


# ── Backward-compat alias for callers that haven't been renamed yet ─────────
# (logs at import time so we know who's still using it).
MercariWatchMonitor = MarketplaceWatchMonitor
