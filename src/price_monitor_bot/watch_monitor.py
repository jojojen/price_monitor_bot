"""Background thread that polls the Mercari watchlist every minute and sends notifications."""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Callable

from market_monitor.mercari_search import search_mercari
from market_monitor.storage import MercariWatch, MonitorDatabase

logger = logging.getLogger(__name__)

NotifyFn = Callable[[str, str], None]  # (chat_id, text)
SnapshotFn = Callable[[str, list[str]], None]  # (chat_id, item_urls)


class MercariWatchMonitor:
    def __init__(
        self,
        *,
        db_path: str | Path,
        notify_fn: NotifyFn,
        snapshot_fn: SnapshotFn | None = None,
        interval_seconds: int = 60,
    ) -> None:
        self._db = MonitorDatabase(db_path)
        self._notify_fn = notify_fn
        self._snapshot_fn = snapshot_fn
        self._interval = interval_seconds
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="mercari-watch-monitor", daemon=True)
        self._thread.start()
        logger.info("MercariWatchMonitor started interval_seconds=%d", self._interval)

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
            watches = self._db.list_mercari_watchlist()
        except Exception:
            logger.exception("MercariWatchMonitor: failed to list watchlist")
            return

        enabled = [w for w in watches if w.enabled]
        if not enabled:
            return

        logger.debug("MercariWatchMonitor: checking %d active watches", len(enabled))
        for watch in enabled:
            try:
                self._check_watch(watch)
            except Exception:
                logger.exception("MercariWatchMonitor: check failed watch_id=%s query=%s", watch.watch_id, watch.query)

    def _check_watch(self, watch: MercariWatch) -> None:
        logger.info(
            "MercariWatchMonitor: searching query=%s price_max=%d watch_id=%s",
            watch.query,
            watch.price_threshold_jpy,
            watch.watch_id,
        )
        results = search_mercari(watch.query, price_max=watch.price_threshold_jpy)

        # If this watch has never been checked before, treat the first scan as a
        # baseline snapshot — record all current items as already-seen so we only
        # notify about items that appear *after* the watch was created.
        is_first_scan = watch.last_checked_at is None
        new_or_changed = self._db.record_mercari_hits(watch.watch_id, results)
        self._db.mark_watch_checked(watch.watch_id)

        if is_first_scan:
            # Baseline scan: record all current items as already-seen (notified=1).
            # Price changes detected on the NEXT scan will still fire normally.
            all_ids = [str(item.get("item_id", "")) for item in results if item.get("item_id")]
            self._db.mark_hits_notified(watch.watch_id, all_ids)
            logger.info(
                "MercariWatchMonitor: first scan baseline recorded watch_id=%s items=%d",
                watch.watch_id,
                len(all_ids),
            )
            return

        if new_or_changed:
            new_count = sum(1 for i in new_or_changed if i.get("_event") != "price_changed")
            changed_count = sum(1 for i in new_or_changed if i.get("_event") == "price_changed")
            logger.info(
                "MercariWatchMonitor: new=%d price_changed=%d watch_id=%s query=%s",
                new_count,
                changed_count,
                watch.watch_id,
                watch.query,
            )
            text = _format_notification(watch, new_or_changed)
            try:
                self._notify_fn(watch.chat_id, text)
            except Exception:
                logger.exception(
                    "MercariWatchMonitor: notification failed watch_id=%s chat_id=%s",
                    watch.watch_id,
                    watch.chat_id,
                )
            else:
                self._db.mark_hits_notified(
                    watch.watch_id,
                    [str(item.get("item_id", "")) for item in new_or_changed
                     if item.get("item_id")],
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
                                "MercariWatchMonitor: snapshot callback failed watch_id=%s",
                                watch.watch_id,
                            )
        else:
            logger.debug("MercariWatchMonitor: no new or changed items watch_id=%s", watch.watch_id)


def _format_notification(watch: MercariWatch, new_or_changed: list[dict[str, object]]) -> str:
    new_count = sum(1 for i in new_or_changed if i.get("_event") != "price_changed")
    changed_count = sum(1 for i in new_or_changed if i.get("_event") == "price_changed")

    summary_parts = []
    if new_count:
        summary_parts.append(f"新商品 {new_count} 筆")
    if changed_count:
        summary_parts.append(f"價格更新 {changed_count} 筆")

    lines = [
        "Mercari 通知",
        f"追蹤條件：{watch.query}",
        f"價格上限：¥{watch.price_threshold_jpy:,}",
        f"發現：{' / '.join(summary_parts)}",
    ]
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
_monitor: MercariWatchMonitor | None = None


def ensure_monitor(
    *,
    db_path: str | Path,
    notify_fn: NotifyFn,
    snapshot_fn: SnapshotFn | None = None,
    interval_seconds: int = 60,
) -> tuple[MercariWatchMonitor, bool]:
    """Return the running monitor singleton, starting it if needed. Returns (monitor, started_now)."""
    global _monitor
    with _monitor_lock:
        if _monitor is not None and _monitor.is_running():
            return _monitor, False
        _monitor = MercariWatchMonitor(
            db_path=db_path,
            notify_fn=notify_fn,
            snapshot_fn=snapshot_fn,
            interval_seconds=interval_seconds,
        )
        _monitor.start()
        return _monitor, True
