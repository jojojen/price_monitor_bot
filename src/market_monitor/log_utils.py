"""Helpers for logging network failures without flooding the logs.

Background fetchers (tcg_tracker source clients, official-store preorder
crawlers) hit transient/expected upstream failures constantly: 429 rate limits,
404s, DNS hiccups, read timeouts, SSL EOFs. Logging each with a full
``logger.exception`` stack turned ``opportunity_agent.log`` into a 1.3G wall of
near-identical tracebacks (issue #42, ~39K of them).

``log_network_failure`` collapses those expected network errors to a single
WARNING line while still preserving the full stack for genuinely unexpected
exceptions — so a real bug hiding among the noise is not silently downgraded.
"""
from __future__ import annotations

import logging
import socket
import ssl
from urllib.error import HTTPError, URLError

# Plain network/transport failures that are expected when scraping flaky upstream
# stores. These never warrant a stack trace.
_TRANSIENT_NETWORK_EXCEPTIONS: tuple[type[BaseException], ...] = (
    TimeoutError,
    socket.timeout,
    URLError,  # also the base of HTTPError; covers DNS via its wrapped reason
    ssl.SSLError,
    ConnectionError,
)


def is_transient_network_error(exc: BaseException) -> bool:
    """True when ``exc`` is an expected upstream network failure (rate limit,
    404/4xx/5xx, DNS, timeout, SSL EOF, connection reset, open circuit)."""
    if isinstance(exc, _TRANSIENT_NETWORK_EXCEPTIONS):
        return True
    # HostRateLimitedError lives in market_monitor.http, which imports heavy
    # optional deps (truststore); resolve it lazily and by name so this module
    # stays import-light and free of circular imports.
    if type(exc).__name__ == "HostRateLimitedError":
        return True
    return False


def describe_network_error(exc: BaseException) -> str:
    """A compact one-line description suitable for a WARNING message."""
    if isinstance(exc, HTTPError):
        return f"HTTP {exc.code} {exc.reason}"
    if isinstance(exc, URLError):
        return f"{type(exc).__name__}: {exc.reason}"
    return f"{type(exc).__name__}: {exc}"


def log_network_failure(
    logger: logging.Logger,
    exc: BaseException,
    msg: str,
    *args: object,
) -> None:
    """Log ``msg % args`` for a fetch failure.

    Transient/expected network errors are logged as a single WARNING line (no
    stack). Anything else keeps the full ``exc_info`` stack at ERROR so real
    bugs stay visible.
    """
    if is_transient_network_error(exc):
        logger.warning("%s — %s", msg % args if args else msg, describe_network_error(exc))
    else:
        logger.error(msg, *args, exc_info=exc)
