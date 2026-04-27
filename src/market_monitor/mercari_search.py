"""Mercari Japan search client using Playwright for keyword + price-max queries."""

from __future__ import annotations

import logging
import re
from urllib.parse import quote

logger = logging.getLogger(__name__)

MERCARI_SEARCH_BASE = "https://jp.mercari.com/search"


def build_search_url(query: str, *, price_max: int) -> str:
    params = f"keyword={quote(query)}&price_max={price_max}&status=on_sale&sort=updated_time&order=desc"
    return f"{MERCARI_SEARCH_BASE}?{params}"


def search_mercari(
    query: str,
    *,
    price_max: int,
    max_results: int = 30,
    timeout_ms: int = 30000,
) -> list[dict[str, object]]:
    """Search Mercari Japan and return items below price_max that match ALL query tokens.

    Returns a list of dicts with keys:
      item_id, title, price_jpy, url, thumbnail_url
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise RuntimeError(
            "playwright is not installed — run: pip install playwright && playwright install chromium"
        ) from exc

    url = build_search_url(query, price_max=price_max)
    logger.info("Mercari search query=%s price_max=%d url=%s", query, price_max, url)

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        context = browser.new_context(
            locale="ja-JP",
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            viewport={"width": 1280, "height": 900},
        )
        page = context.new_page()
        try:
            page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
            # Wait for item grid to appear
            try:
                page.wait_for_selector(
                    "mer-item-thumbnail, [data-testid='item-cell'], li[data-item-id]",
                    timeout=15000,
                )
            except Exception:
                logger.debug("Mercari: item selector timed out, trying JS scroll")
                page.evaluate("window.scrollTo(0, 300)")
                page.wait_for_timeout(2000)

            raw_items = _extract_items(page, max_results=max_results * 3)
            query_matched = _filter_by_query(raw_items, query)
            items = [item for item in query_matched if int(item.get("price_jpy") or 0) <= price_max]
            dropped_price = len(query_matched) - len(items)
            logger.info(
                "Mercari search raw=%d matched=%d price_filtered=%d query=%s",
                len(raw_items),
                len(items),
                dropped_price,
                query,
            )
            return items[:max_results]
        finally:
            context.close()
            browser.close()


# ── Relevance filter ──────────────────────────────────────────────────────────

def _normalise(text: str) -> str:
    """Lower-case and normalise spaces/full-width chars for substring matching."""
    return (
        text.lower()
        .replace("　", " ")   # full-width space → ASCII space
        .replace("・", " ")   # middle dot → space
        .replace("【", " ").replace("】", " ")
        .replace("（", " ").replace("）", " ")
        .replace("「", " ").replace("」", " ")
        # strip the Mercari auto-appended alt-text suffix
        .removesuffix("のサムネイル")
        .strip()
    )


def _query_tokens(query: str) -> list[str]:
    """Split query into non-empty tokens for AND matching."""
    # Normalise then split on whitespace
    tokens = _normalise(query).split()
    return [t for t in tokens if t]


def _filter_by_query(items: list[dict[str, object]], query: str) -> list[dict[str, object]]:
    """Keep only items whose title contains ALL tokens from query."""
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


def _extract_items(page: object, *, max_results: int) -> list[dict[str, object]]:
    """Try multiple extraction strategies to handle Mercari's evolving DOM."""
    items: list[dict[str, object]] = []

    # Strategy 1: evaluate JS to pull structured data from the page
    try:
        js_items = page.evaluate(_JS_EXTRACT)  # type: ignore[attr-defined]
        if js_items and isinstance(js_items, list):
            for raw in js_items[:max_results]:
                parsed = _parse_js_item(raw)
                if parsed:
                    items.append(parsed)
            if items:
                logger.debug("Mercari JS extraction succeeded items=%d", len(items))
                return items
    except Exception as exc:
        logger.debug("Mercari JS extraction failed: %s", exc)

    # Strategy 2: DOM selector scraping
    try:
        dom_items = _extract_via_dom(page, max_results=max_results)
        if dom_items:
            logger.debug("Mercari DOM extraction succeeded items=%d", len(dom_items))
            return dom_items
    except Exception as exc:
        logger.debug("Mercari DOM extraction failed: %s", exc)

    return items


_JS_EXTRACT = """
(() => {
  const results = [];

  // Try <mer-item-thumbnail> web components (Mercari's custom elements)
  const thumbnails = document.querySelectorAll('mer-item-thumbnail');
  if (thumbnails.length > 0) {
    thumbnails.forEach(el => {
      try {
        const anchor = el.closest('a') || el.querySelector('a');
        const href = anchor ? anchor.getAttribute('href') : null;
        const itemIdMatch = href ? href.match(/\\/item\\/(m[0-9a-zA-Z]+)/) : null;
        const itemId = itemIdMatch ? itemIdMatch[1] : null;
        if (!itemId) return;

        // Price
        const priceEl = el.querySelector('[data-testid="thumbnail-price"]') ||
                        el.querySelector('.merPrice') ||
                        el.closest('li')?.querySelector('[class*="price"]');
        const priceText = priceEl ? priceEl.textContent : '';
        const priceMatch = priceText.replace(/[,，]/g, '').match(/([0-9]+)/);
        const price = priceMatch ? parseInt(priceMatch[1]) : null;
        if (!price) return;

        // Title
        const titleEl = el.querySelector('img');
        const title = titleEl ? (titleEl.getAttribute('alt') || '') : '';

        // Thumbnail
        const imgEl = el.querySelector('img');
        const thumbnail = imgEl ? imgEl.src : null;

        results.push({ item_id: itemId, title, price_jpy: price, href, thumbnail });
      } catch(e) {}
    });
    return results;
  }

  // Try generic li[data-item-id] pattern
  const listItems = document.querySelectorAll('li[data-item-id]');
  listItems.forEach(li => {
    try {
      const itemId = li.getAttribute('data-item-id');
      if (!itemId) return;
      const anchor = li.querySelector('a[href*="/item/"]');
      const href = anchor ? anchor.getAttribute('href') : `/item/${itemId}`;
      const priceEl = li.querySelector('[class*="price"]');
      const priceText = priceEl ? priceEl.textContent : '';
      const priceMatch = priceText.replace(/[,，]/g, '').match(/([0-9]+)/);
      const price = priceMatch ? parseInt(priceMatch[1]) : null;
      if (!price) return;
      const imgEl = li.querySelector('img');
      const title = imgEl ? imgEl.getAttribute('alt') || '' : '';
      const thumbnail = imgEl ? imgEl.src : null;
      results.push({ item_id: itemId, title, price_jpy: price, href, thumbnail });
    } catch(e) {}
  });
  return results;
})()
"""


def _clean_title(raw: str) -> str:
    """Strip Mercari's auto-appended alt-text suffix and trim whitespace."""
    return raw.removesuffix("のサムネイル").strip()


def _parse_js_item(raw: dict[str, object]) -> dict[str, object] | None:
    item_id = raw.get("item_id") or ""
    price_jpy = raw.get("price_jpy")
    if not item_id or not isinstance(price_jpy, int) or price_jpy <= 0:
        return None
    href = str(raw.get("href") or f"/item/{item_id}")
    url = href if href.startswith("http") else f"https://jp.mercari.com{href}"
    return {
        "item_id": str(item_id),
        "title": _clean_title(str(raw.get("title") or "")),
        "price_jpy": price_jpy,
        "url": url,
        "thumbnail_url": raw.get("thumbnail") or None,
    }


def _extract_via_dom(page: object, *, max_results: int) -> list[dict[str, object]]:
    items: list[dict[str, object]] = []

    # Try anchor tags pointing to /item/mXXXX
    anchors = page.query_selector_all("a[href*='/item/m']")  # type: ignore[attr-defined]
    seen_ids: set[str] = set()

    for anchor in anchors[:max_results * 3]:
        try:
            href = anchor.get_attribute("href") or ""
            m = re.search(r"/item/(m[0-9a-zA-Z]+)", href)
            if not m:
                continue
            item_id = m.group(1)
            if item_id in seen_ids:
                continue
            seen_ids.add(item_id)

            # Walk up to find price
            price_jpy: int | None = None
            parent = anchor
            for _ in range(5):
                price_el = parent.query_selector('[class*="price"], [data-testid*="price"]')
                if price_el:
                    price_text = price_el.inner_text() or ""
                    pm = re.search(r"([0-9,，]+)", price_text.replace(",", "").replace("，", ""))
                    if pm:
                        price_jpy = int(pm.group(1))
                    break
                try:
                    parent = parent.evaluate_handle("el => el.parentElement")
                except Exception:
                    break
            if not price_jpy:
                continue

            img = anchor.query_selector("img")
            title = ""
            thumbnail_url = None
            if img:
                title = _clean_title(img.get_attribute("alt") or "")
                thumbnail_url = img.get_attribute("src")

            url = href if href.startswith("http") else f"https://jp.mercari.com{href}"
            items.append({
                "item_id": item_id,
                "title": title,
                "price_jpy": price_jpy,
                "url": url,
                "thumbnail_url": thumbnail_url,
            })
            if len(items) >= max_results:
                break
        except Exception:
            continue

    return items
