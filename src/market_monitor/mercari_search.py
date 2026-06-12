"""Mercari Japan search client using Playwright for keyword + price-max queries.

Two-stage pricing:
  1. Playwright loads the search results page and we parse it with BeautifulSoup
     to find candidate items matching the query and price_max.
  2. For each candidate we then fetch its detail page via plain HTTP and read
     ``<meta name="product:price:amount">`` as the authoritative price. The
     search-page card is unreliable because Mercari sometimes renders an
     installment / monthly-payment estimate alongside the real price; only the
     detail-page meta tag tells us the actual list price.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import urllib.error
import urllib.request
from typing import Any
from urllib.parse import quote

from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

MERCARI_SEARCH_BASE = "https://jp.mercari.com/search"
MERCARI_ITEM_BASE = "https://jp.mercari.com/item"

_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)

_INSTALLMENT_MARKERS = ("月々", "分割", "/月", "～", "〜")

# Mercari Japan item-condition (商品の状態) enum. IDs match what the public
# search URL accepts as `item_condition_id`.
MERCARI_CONDITION_NEW = 1            # 新品、未使用
MERCARI_CONDITION_LIKE_NEW = 2       # 未使用に近い
MERCARI_CONDITION_GOOD = 3           # 目立った傷や汚れなし
MERCARI_CONDITION_FAIR = 4           # やや傷や汚れあり
MERCARI_CONDITION_POOR = 5           # 傷や汚れあり
MERCARI_CONDITION_VERY_POOR = 6      # 全体的に状態が悪い

MERCARI_CONDITION_LABELS: dict[int, str] = {
    MERCARI_CONDITION_NEW: "新品、未使用",
    MERCARI_CONDITION_LIKE_NEW: "未使用に近い",
    MERCARI_CONDITION_GOOD: "目立った傷や汚れなし",
    MERCARI_CONDITION_FAIR: "やや傷や汚れあり",
    MERCARI_CONDITION_POOR: "傷や汚れあり",
    MERCARI_CONDITION_VERY_POOR: "全体的に状態が悪い",
}

# "Good and above" — the default filter applied to user watches and the
# opportunity hunter. Anything dirtier than this clutters notifications.
DEFAULT_CONDITION_IDS: tuple[int, ...] = (
    MERCARI_CONDITION_NEW,
    MERCARI_CONDITION_LIKE_NEW,
    MERCARI_CONDITION_GOOD,
)


def build_search_url(
    query: str,
    *,
    price_max: int,
    condition_ids: tuple[int, ...] | None = None,
    sold: bool = False,
) -> str:
    status = "sold_out" if sold else "on_sale"
    parts = [
        f"keyword={quote(query)}",
        f"price_max={price_max}",
        f"status={status}",
        "sort=updated_time",
        "order=desc",
    ]
    if condition_ids:
        # Mercari expects the param to repeat once per condition id.
        for cid in condition_ids:
            parts.append(f"item_condition_id={cid}")
    return f"{MERCARI_SEARCH_BASE}?{'&'.join(parts)}"


def search_mercari(
    query: str,
    *,
    price_max: int,
    max_results: int = 30,
    timeout_ms: int = 30000,
    condition_ids: tuple[int, ...] | None = None,
) -> list[dict[str, object]]:
    """Search Mercari Japan and return items below price_max that match ALL query tokens.

    ``condition_ids`` is the Mercari item-condition (商品の状態) filter; pass
    e.g. ``(1, 2, 3)`` to limit results to 目立った傷や汚れなし以上. When
    omitted no condition filter is applied.

    Returns a list of dicts with keys:
      item_id, title, price_jpy, url, thumbnail_url
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise RuntimeError(
            "playwright is not installed; run: pip install playwright && playwright install chromium"
        ) from exc

    url = build_search_url(query, price_max=price_max, condition_ids=condition_ids)
    logger.info(
        "Mercari search query=%s price_max=%d condition_ids=%s url=%s",
        query,
        price_max,
        condition_ids,
        url,
    )

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(**_chromium_launch_options())
        context = browser.new_context(
            locale="ja-JP",
            user_agent=_USER_AGENT,
            viewport={"width": 1280, "height": 900},
        )
        page = context.new_page()
        try:
            page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
            try:
                page.wait_for_selector(
                    'li[data-testid="item-cell"], mer-item-thumbnail, li[data-item-id]',
                    timeout=15000,
                )
            except Exception:
                logger.debug("Mercari: item selector timed out, trying JS scroll")
                page.evaluate("window.scrollTo(0, 300)")
                page.wait_for_timeout(2000)

            html = page.content()
        finally:
            context.close()
            browser.close()

    raw_items = parse_search_html(html, max_results=max_results * 3)
    query_matched = _filter_by_query(raw_items, query)
    candidates = [item for item in query_matched if int(item.get("price_jpy") or 0) <= price_max]
    dropped_price = len(query_matched) - len(candidates)

    verified = _verify_candidates_via_detail_pages(candidates, price_max=price_max)

    logger.info(
        "Mercari search raw=%d matched=%d price_filtered=%d verified=%d query=%s",
        len(raw_items),
        len(candidates),
        dropped_price,
        len(verified),
        query,
    )
    return verified[:max_results]


def search_mercari_sold(
    query: str,
    *,
    max_results: int = 20,
    timeout_ms: int = 45_000,
) -> list[dict[str, object]]:
    """Search Mercari sold listings and return matched items with URLs/prices.

    Uses the same rendered-results parsing as ``search_mercari`` and keeps
    detail-page verification enabled so the returned sold prices stay aligned
    with the authoritative item-page meta tag.
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise RuntimeError(
            "playwright is not installed; run: pip install playwright && playwright install chromium"
        ) from exc

    price_max = 9_999_999
    url = build_search_url(query, price_max=price_max, sold=True)
    logger.info("Mercari sold search query=%s url=%s", query, url)

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(**_chromium_launch_options())
        context = browser.new_context(
            locale="ja-JP",
            user_agent=_USER_AGENT,
            viewport={"width": 1280, "height": 900},
        )
        page = context.new_page()
        try:
            page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
            try:
                page.wait_for_selector(
                    'li[data-testid="item-cell"], mer-item-thumbnail, li[data-item-id]',
                    timeout=15000,
                )
            except Exception:
                logger.debug("Mercari sold: item selector timed out, trying JS scroll")
                page.evaluate("window.scrollTo(0, 300)")
                page.wait_for_timeout(2000)
            html = page.content()
        finally:
            context.close()
            browser.close()

    raw_items = parse_search_html(html, max_results=max_results * 3)
    matched = _filter_by_query(raw_items, query)
    verified = _verify_candidates_via_detail_pages(matched, price_max=price_max)
    logger.info(
        "Mercari sold search raw=%d matched=%d verified=%d query=%s",
        len(raw_items),
        len(matched),
        len(verified),
        query,
    )
    return verified[:max_results]


def fetch_avg_sold_price(
    query: str,
    *,
    min_results: int = 3,
    max_results: int = 20,
    timeout_ms: int = 45_000,
) -> float | None:
    """Return the average sold price on Mercari Japan for *query*, or None.

    Searches sold (``status=sold_out``) listings, parses prices from the search
    results HTML, and returns the mean price in JPY.  Returns None when fewer
    than *min_results* sold items are found.

    Uses the same Playwright infrastructure as :func:`search_mercari`.  The
    *price_max* parameter is set very high (9_999_999) so that no sold item is
    filtered out by price.
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise RuntimeError(
            "playwright is not installed; run: pip install playwright && playwright install chromium"
        ) from exc

    _PRICE_MAX_SENTINEL = 9_999_999
    url = build_search_url(query, price_max=_PRICE_MAX_SENTINEL, sold=True)
    logger.info(
        "Mercari sold-price fetch query=%s url=%s",
        query,
        url,
    )

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(**_chromium_launch_options())
        context = browser.new_context(
            locale="ja-JP",
            user_agent=_USER_AGENT,
            viewport={"width": 1280, "height": 900},
        )
        page = context.new_page()
        try:
            page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
            try:
                page.wait_for_selector(
                    'li[data-testid="item-cell"], mer-item-thumbnail, li[data-item-id]',
                    timeout=15000,
                )
            except Exception:
                logger.debug("Mercari sold: item selector timed out, trying JS scroll")
                page.evaluate("window.scrollTo(0, 300)")
                page.wait_for_timeout(2000)
            html = page.content()
        finally:
            context.close()
            browser.close()

    raw_items = parse_search_html(html, max_results=max_results * 3)
    matched = _filter_by_query(raw_items, query)
    prices = [int(item["price_jpy"]) for item in matched if item.get("price_jpy")]

    if len(prices) < min_results:
        logger.info(
            "Mercari sold-price: insufficient results query=%s found=%d min=%d",
            query, len(prices), min_results,
        )
        return None

    avg = sum(prices[:max_results]) / len(prices[:max_results])
    logger.info(
        "Mercari sold-price query=%s n=%d avg=%.0f",
        query, len(prices[:max_results]), avg,
    )
    return avg


# ── HTML parsing ──────────────────────────────────────────────────────────────

def parse_search_html(html: str, *, max_results: int) -> list[dict[str, object]]:
    """Parse rendered Mercari search-results HTML into item dicts.

    Pure function so unit tests can call it against saved fixtures without
    spinning up Playwright.
    """
    soup = BeautifulSoup(html, "html.parser")
    items: list[dict[str, object]] = []
    seen_ids: set[str] = set()

    for li in soup.select('li[data-testid="item-cell"]'):
        anchor = li.select_one('a[href*="/item/"]')
        if not anchor:
            continue
        href = anchor.get("href") or ""
        m = re.search(r"/item/(m[0-9a-zA-Z]+)", href)
        if not m:
            continue
        item_id = m.group(1)
        if item_id in seen_ids:
            continue

        price = _extract_price_from_card(li)
        if price is None:
            logger.info("Mercari: no clean price in card item_id=%s — skipping", item_id)
            continue

        title = _extract_title_from_card(li)
        img = li.select_one("img")
        thumb = img.get("src") if img else None
        url = href if href.startswith("http") else f"https://jp.mercari.com{href}"

        items.append({
            "item_id": item_id,
            "title": _clean_title(title),
            "price_jpy": price,
            "url": url,
            "thumbnail_url": thumb,
        })
        seen_ids.add(item_id)
        if len(items) >= max_results:
            break

    return items


def _extract_price_from_card(card: Any) -> int | None:
    """Pull the displayed list price from an item card, rejecting installment estimates."""
    # Prefer the canonical .merPrice component (its text is "¥X,XXX")
    price_el = card.select_one(".merPrice")
    if price_el is None:
        return None
    text = price_el.get_text(strip=True)
    if any(marker in text for marker in _INSTALLMENT_MARKERS):
        return None
    return _parse_yen_text(text)


def _extract_title_from_card(card: Any) -> str:
    name_el = card.select_one('[data-testid="thumbnail-item-name"]')
    if name_el is not None:
        return name_el.get_text(strip=True)
    img = card.select_one("img[alt]")
    if img is not None:
        return img.get("alt") or ""
    return ""


def _parse_yen_text(text: str) -> int | None:
    cleaned = text.replace("¥", "").replace(",", "").replace("，", "").strip()
    m = re.search(r"(\d+)", cleaned)
    return int(m.group(1)) if m else None


# ── Detail-page verification ──────────────────────────────────────────────────

_PRICE_META_RE = re.compile(
    r'<meta[^>]+name=["\']product:price:amount["\'][^>]+content=["\'](\d+)["\']',
    re.IGNORECASE,
)


def fetch_item_detail_price(item_id: str, *, timeout: float = 15.0) -> int | None:
    """Fetch the authoritative price for a Mercari item via plain HTTP.

    Reads ``<meta name="product:price:amount">`` from the SSR'd HTML.
    Returns the price in JPY or None on any failure.
    """
    url = f"{MERCARI_ITEM_BASE}/{item_id}"
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": _USER_AGENT,
            "Accept-Language": "ja-JP,ja;q=0.9",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status != 200:
                logger.warning("Mercari detail fetch HTTP %s for %s", resp.status, item_id)
                return None
            html = resp.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, TimeoutError) as exc:
        logger.warning("Mercari detail fetch error for %s: %s", item_id, exc)
        return None
    return parse_detail_price(html)


def parse_detail_price(html: str) -> int | None:
    """Extract the authoritative price from a Mercari item detail page HTML."""
    m = _PRICE_META_RE.search(html)
    return int(m.group(1)) if m else None


def _verify_candidates_via_detail_pages(
    candidates: list[dict[str, object]],
    *,
    price_max: int,
) -> list[dict[str, object]]:
    verified: list[dict[str, object]] = []
    for item in candidates:
        item_id = str(item.get("item_id") or "")
        if not item_id:
            continue
        detail_price = fetch_item_detail_price(item_id)
        if detail_price is None:
            logger.warning(
                "Mercari: detail-page verification failed for %s — dropping (search price was ¥%s)",
                item_id,
                item.get("price_jpy"),
            )
            continue
        if detail_price > price_max:
            logger.info(
                "Mercari: %s above price_max after verification (search=¥%s, detail=¥%s, max=¥%s)",
                item_id,
                item.get("price_jpy"),
                detail_price,
                price_max,
            )
            continue
        search_price = int(item.get("price_jpy") or 0)
        if search_price and detail_price != search_price:
            logger.warning(
                "Mercari: search/detail price mismatch for %s (search=¥%s, detail=¥%s) — using detail",
                item_id,
                search_price,
                detail_price,
            )
        verified.append({**item, "price_jpy": detail_price})
    return verified


# ── Relevance filter ──────────────────────────────────────────────────────────

def _chromium_launch_options() -> dict[str, object]:
    options: dict[str, object] = {"headless": True}
    executable_path = _resolve_chromium_executable()
    if executable_path:
        options["executable_path"] = executable_path
    return options


def _resolve_chromium_executable() -> str | None:
    configured = os.getenv("PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH")
    if configured:
        return configured
    for candidate in ("chromium", "chromium-browser", "google-chrome-stable"):
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
    return None


def _normalise(text: str) -> str:
    return (
        text.lower()
        .replace("　", " ")
        .replace("・", " ")
        .replace("【", " ").replace("】", " ")
        .replace("（", " ").replace("）", " ")
        .replace("「", " ").replace("」", " ")
        .removesuffix("のサムネイル")
        .strip()
    )


def _query_tokens(query: str) -> list[str]:
    tokens = _normalise(query).split()
    return [t for t in tokens if t]


def _filter_by_query(items: list[dict[str, object]], query: str) -> list[dict[str, object]]:
    tokens = _query_tokens(query)
    if not tokens:
        return items

    kept: list[dict[str, object]] = []
    for item in items:
        raw_title = str(item.get("title") or "")
        normalised = _normalise(raw_title)
        if all(token in normalised for token in tokens):
            kept.append(item)
        else:
            logger.debug(
                "Mercari filter: dropped item_id=%s title=%r (missing tokens)",
                item.get("item_id"),
                raw_title[:60],
            )
    return kept


def _clean_title(raw: str) -> str:
    return raw.removesuffix("のサムネイル").strip()
