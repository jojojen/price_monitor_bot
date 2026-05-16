"""One-off: capture Mercari HTML samples for parser regression tests.

Saves rendered (post-JS) HTML for:
  - search results for "綾波レイ ユニオンアリーナ プロモカード" (price_max=4500)
  - item detail page m18542743389  (real price ¥8,300)
  - item detail page m85537287496  (real price ¥6,555)

Run with the aka_no_claw venv that has playwright installed:
  /Users/jen/ai_work_space/related_to_claw/aka_no_claw/.venv/bin/python \
      /Users/jen/ai_work_space/related_to_claw/price_monitor_bot/scripts/capture_mercari_fixtures.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from urllib.parse import quote

from playwright.sync_api import sync_playwright

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "tests" / "fixtures"

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)

QUERY = "綾波レイ ユニオンアリーナ プロモカード"
PRICE_MAX = 4500
ITEM_IDS = ["m18542743389", "m85537287496"]


def _save(name: str, html: str) -> None:
    out = FIXTURES_DIR / name
    out.write_text(html, encoding="utf-8")
    print(f"  saved {out}  ({len(html):,} bytes)")


def main() -> int:
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)

    search_url = (
        "https://jp.mercari.com/search?"
        f"keyword={quote(QUERY)}&price_max={PRICE_MAX}"
        "&status=on_sale&sort=updated_time&order=desc"
    )

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        context = browser.new_context(
            locale="ja-JP",
            user_agent=USER_AGENT,
            viewport={"width": 1280, "height": 900},
        )
        page = context.new_page()

        print(f"[search] {search_url}")
        page.goto(search_url, wait_until="domcontentloaded", timeout=30000)
        try:
            page.wait_for_selector(
                "mer-item-thumbnail, [data-testid='item-cell'], li[data-item-id]",
                timeout=15000,
            )
        except Exception:
            page.evaluate("window.scrollTo(0, 600)")
            page.wait_for_timeout(2000)
        page.wait_for_timeout(2000)
        _save("mercari_search_eva_rei.html", page.content())

        for item_id in ITEM_IDS:
            url = f"https://jp.mercari.com/item/{item_id}"
            print(f"[item] {url}")
            page.goto(url, wait_until="domcontentloaded", timeout=30000)
            try:
                page.wait_for_selector(
                    "[data-testid='price'], .merPrice, meta[name='product:price:amount']",
                    timeout=15000,
                )
            except Exception:
                page.wait_for_timeout(2000)
            page.wait_for_timeout(1500)
            _save(f"mercari_item_{item_id}.html", page.content())

        context.close()
        browser.close()

    print("done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
