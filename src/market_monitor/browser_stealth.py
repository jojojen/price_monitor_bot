"""Shared human-like browser fingerprint for every scraper in the stack.

Marketplaces (Mercari, Yahoo, yuyu-tei…) increasingly serve bot interstitials to
headless Chromium with a stitched-together identity. This module centralises one
coherent, human macOS-Chrome fingerprint so all page-fetching code presents the
same real-browser face:

  * real Chrome (channel="chrome") when available — far fewer automation signals
    than bundled Chromium — with a safe fallback to bundled Chromium
  * a macOS UA that matches the host, kept consistent with navigator.platform,
    WebGL vendor/renderer, timezone and locale (a mismatch is itself a tell)
  * Asia/Tokyo timezone, ja-JP Accept-Language, a realistic laptop viewport
  * a stealth init script and small randomized "human" scrolls

Both repos import from here (aka_no_claw already depends on market_monitor), so
there is a single source of truth rather than per-scraper copies that drift.
"""
from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

# A current macOS Chrome. Mac UAs freeze the OS token at 10_15_7 by design.
MAC_CHROME_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/149.0.0.0 Safari/537.36"
)

ACCEPT_LANGUAGE = "ja-JP,ja;q=0.9,en-US;q=0.8,en;q=0.7"

DEFAULT_LAUNCH_ARGS = (
    "--disable-blink-features=AutomationControlled",
    "--no-sandbox",
    "--disable-dev-shm-usage",
)

# Patches applied before any page script runs. Real Chrome already satisfies most
# of these; they matter for the bundled-Chromium fallback and are idempotent and
# harmless under real Chrome.
STEALTH_INIT_SCRIPT = """
() => {
  Object.defineProperty(navigator, 'webdriver', {get: () => false});
  Object.defineProperty(navigator, 'languages', {get: () => ['ja-JP', 'ja', 'en-US', 'en']});
  Object.defineProperty(navigator, 'platform', {get: () => 'MacIntel'});
  Object.defineProperty(navigator, 'maxTouchPoints', {get: () => 0});
  Object.defineProperty(navigator, 'hardwareConcurrency', {get: () => 8});
  try { Object.defineProperty(navigator, 'deviceMemory', {get: () => 8}); } catch (e) {}
  Object.defineProperty(navigator, 'plugins', {get: () => [1, 2, 3, 4, 5]});
  window.chrome = window.chrome || { runtime: {} };
  const origQuery = window.navigator.permissions && window.navigator.permissions.query;
  if (origQuery) {
    window.navigator.permissions.query = (p) => (
      p && p.name === 'notifications'
        ? Promise.resolve({ state: Notification.permission })
        : origQuery(p)
    );
  }
  try {
    const getParam = WebGLRenderingContext.prototype.getParameter;
    WebGLRenderingContext.prototype.getParameter = function (p) {
      if (p === 37445) return 'Google Inc. (Apple)';                 // UNMASKED_VENDOR_WEBGL
      if (p === 37446) return 'ANGLE (Apple, Apple M2, OpenGL 4.1)'; // UNMASKED_RENDERER_WEBGL
      return getParam.call(this, p);
    };
  } catch (e) {}
}
"""


def http_headers(extra: dict[str, str] | None = None) -> dict[str, str]:
    """Headers for urllib/requests scrapers so static fetches wear the same human
    Chrome identity as the Playwright ones (a CLI-looking UA invites blocks)."""
    headers = {
        "User-Agent": MAC_CHROME_UA,
        "Accept-Language": ACCEPT_LANGUAGE,
        "Accept": (
            "text/html,application/xhtml+xml,application/xml;q=0.9,"
            "image/avif,image/webp,*/*;q=0.8"
        ),
    }
    if extra:
        headers.update(extra)
    return headers


def stealth_context_kwargs(**overrides: Any) -> dict[str, Any]:
    """new_context()/launch_persistent_context() kwargs pinning a coherent
    JP-Mac-Chrome identity. Callers may override any field (e.g. viewport)."""
    kwargs: dict[str, Any] = {
        "locale": "ja-JP",
        "timezone_id": "Asia/Tokyo",
        "viewport": {"width": 1512, "height": 982},  # MacBook-ish, not a bot canvas
        "device_scale_factor": 2,
        "is_mobile": False,
        "has_touch": False,
        "user_agent": MAC_CHROME_UA,
        "extra_http_headers": {"Accept-Language": ACCEPT_LANGUAGE},
    }
    kwargs.update(overrides)
    return kwargs


def resolve_browser_channel() -> str | None:
    """Prefer the real Chrome binary — far fewer automation signals than bundled
    Chromium. Override with OPENCLAW_BROWSER_CHANNEL (empty string forces bundled
    Chromium)."""
    override = os.getenv("OPENCLAW_BROWSER_CHANNEL")
    if override is not None:
        return override or None
    if Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome").exists():
        return "chrome"
    return None


def _executable_override() -> str | None:
    return os.getenv("PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH") or None


def _merge_args(extra_args: tuple[str, ...] | list[str] | None) -> list[str]:
    args = list(DEFAULT_LAUNCH_ARGS)
    for arg in extra_args or ():
        if arg not in args:
            args.append(arg)
    return args


def launch_stealth_chromium(
    playwright: Any,
    *,
    headless: bool = True,
    extra_args: tuple[str, ...] | list[str] | None = None,
    logger: Any = None,
) -> Any:
    """Launch Chromium as real Chrome when possible, falling back to bundled
    Chromium. An explicit PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH wins over channel."""
    launch_kwargs: dict[str, Any] = {"headless": headless, "args": _merge_args(extra_args)}
    executable = _executable_override()
    if executable:
        launch_kwargs["executable_path"] = executable
        return playwright.chromium.launch(**launch_kwargs)
    channel = resolve_browser_channel()
    if channel:
        try:
            return playwright.chromium.launch(channel=channel, **launch_kwargs)
        except Exception as exc:  # real Chrome momentarily unavailable (auto-update…)
            if logger is not None:
                logger.warning(
                    "browser_stealth: channel=%s launch failed (%s); falling back to bundled chromium",
                    channel, exc,
                )
    return playwright.chromium.launch(**launch_kwargs)


def new_stealth_context(browser: Any, **overrides: Any) -> Any:
    """A new_context() with the stealth fingerprint and init script applied."""
    ctx = browser.new_context(**stealth_context_kwargs(**overrides))
    ctx.add_init_script(STEALTH_INIT_SCRIPT)
    return ctx


def launch_stealth_persistent_context(
    playwright: Any,
    user_data_dir: str,
    *,
    headless: bool = True,
    extra_args: tuple[str, ...] | list[str] | None = None,
    logger: Any = None,
    **overrides: Any,
) -> Any:
    """Persistent-profile variant (used by the Yahoo search session). Same channel
    preference, fingerprint and init script as the throwaway-context path."""
    kwargs = stealth_context_kwargs(**overrides)
    kwargs["headless"] = headless
    kwargs["args"] = _merge_args(extra_args)
    executable = _executable_override()
    ctx = None
    if executable:
        kwargs["executable_path"] = executable
        ctx = playwright.chromium.launch_persistent_context(str(user_data_dir), **kwargs)
    else:
        channel = resolve_browser_channel()
        if channel:
            try:
                ctx = playwright.chromium.launch_persistent_context(
                    str(user_data_dir), channel=channel, **kwargs
                )
            except Exception as exc:
                if logger is not None:
                    logger.warning(
                        "browser_stealth: channel=%s persistent launch failed (%s); "
                        "falling back to bundled chromium",
                        channel, exc,
                    )
        if ctx is None:
            ctx = playwright.chromium.launch_persistent_context(str(user_data_dir), **kwargs)
    ctx.add_init_script(STEALTH_INIT_SCRIPT)
    return ctx


def humanize(page: Any) -> None:
    """A few small, randomized scrolls + pauses so a session reads less like a
    headless bot going straight goto→scrape. Best-effort; never fail a capture
    over a cosmetic gesture."""
    try:
        for _ in range(random.randint(2, 4)):
            page.mouse.wheel(0, random.randint(280, 620))
            page.wait_for_timeout(random.randint(350, 900))
    except Exception:
        pass
