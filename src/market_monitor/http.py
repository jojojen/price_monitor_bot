from __future__ import annotations

import logging
import os
import random
import re
import socket
import shutil
import ssl
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen

import truststore

from . import browser_stealth as bs
from .host_budget import get_host_budget

logger = logging.getLogger(__name__)
_TRANSIENT_HTTP_EXCEPTIONS = (HTTPError, URLError, TimeoutError, socket.timeout)

# Status codes worth retrying; anything else (4xx except 429) goes straight to
# curl fallback without burning time on retries that won't help.
_RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})
_HTTP_MAX_RETRIES = 2
_HTTP_RETRY_BASE_SEC = 1.5
_HTTP_RETRY_AFTER_CAP_SEC = 30.0

# ── Per-host circuit breaker (cross-process) ──────────────────────────────────
# Once a host answers 429 (rate limited), every crawler short-circuits further
# requests to that host for a cooldown window instead of continuing to poke it.
# Hammering a host that is already limiting you only prolongs the IP cooldown;
# backing off lets it clear.
#
# The cooldown is shared TWO ways: in-process (a monotonic deadline per host,
# across all HttpClient instances) AND cross-process (a wall-clock expiry written
# to a tempdir marker file per host). OpenClaw runs several independent processes
# on one machine — the opportunity-agent, the Telegram /research worker, scrape
# subprocesses — each with its own memory. A monotonic deadline can't be shared
# across processes (the clock origins differ), so the file carries a wall-clock
# expiry that any peer process reads before it fires its own request. This is the
# fix for the yuyu-tei rate-limit amplification: a background 429 now makes the
# foreground /research back off instead of discovering the limit the hard way.
_HOST_COOLDOWN_SEC = 300.0
_circuit_lock = threading.Lock()
_host_open_until: dict[str, float] = {}
# Marker files this process has written, so reset_circuit_breaker() can clean up
# after tests without disturbing cooldowns owned by other live processes.
_circuit_files_written: set[str] = set()


class HostRateLimitedError(Exception):
    """Raised instead of issuing a request to a host whose circuit is open, or
    whose shared host-budget (#24) declined the slot (cooldown / concurrency /
    disabled). ``decision``/``reason`` carry the budget's verdict when present."""

    def __init__(
        self,
        host: str,
        remaining_seconds: float,
        *,
        decision: str | None = None,
        reason: str | None = None,
    ) -> None:
        self.host = host
        self.remaining_seconds = remaining_seconds
        self.decision = decision
        self.reason = reason
        detail = reason or decision
        suffix = f" — {detail}" if detail else ""
        super().__init__(
            f"circuit open for host={host} ({remaining_seconds:.0f}s remaining){suffix}"
        )


def _host_of(target: str) -> str:
    return (urlparse(target).hostname or target).lower()


def _circuit_file_path(host: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", host) or "unknown"
    return Path(tempfile.gettempdir()) / f"openclaw_circuit_{safe}"


def _read_file_cooldown(host: str) -> float:
    """Seconds left on this host's cross-process marker (0.0 if none/expired)."""
    path = _circuit_file_path(host)
    try:
        raw = path.read_text().strip()
    except OSError:
        return 0.0
    try:
        expiry = float(raw)
    except ValueError:
        # Legacy/empty marker (older code touched the file): treat mtime as the
        # start of a default-length window so a stale signal still backs us off.
        try:
            expiry = path.stat().st_mtime + _HOST_COOLDOWN_SEC
        except OSError:
            return 0.0
    return max(0.0, expiry - time.time())


def _write_file_cooldown(host: str, cooldown_seconds: float) -> None:
    """Persist a wall-clock expiry so peer processes back off too. Never shortens
    a longer cooldown another process already wrote (bias toward backing off)."""
    path = _circuit_file_path(host)
    if _read_file_cooldown(host) >= cooldown_seconds:
        _circuit_files_written.add(str(path))
        return
    try:
        path.write_text(str(time.time() + cooldown_seconds))
        _circuit_files_written.add(str(path))
    except OSError:
        pass


def _circuit_remaining(host: str) -> float:
    with _circuit_lock:
        until = _host_open_until.get(host)
    in_process = max(0.0, until - time.monotonic()) if until is not None else 0.0
    return max(in_process, _read_file_cooldown(host))


def _trip_circuit(host: str, cooldown_seconds: float) -> None:
    until = time.monotonic() + cooldown_seconds
    with _circuit_lock:
        if until > _host_open_until.get(host, 0.0):
            _host_open_until[host] = until
    _write_file_cooldown(host, cooldown_seconds)
    logger.warning(
        "HTTP circuit OPEN host=%s cooldown=%.0fs (persisted cross-process) — "
        "skipping further requests until it clears",
        host, cooldown_seconds,
    )


def _clear_circuit(host: str) -> None:
    # Only clears the in-process deadline. The cross-process marker is left to
    # expire on its own: a single success here must not wipe a backoff a peer
    # process set from a 429, or we'd start amplifying the rate limit again.
    with _circuit_lock:
        _host_open_until.pop(host, None)


def trip_host_cooldown(url: str, cooldown_seconds: float | None = None) -> None:
    """Open a host's circuit from outside ``HttpClient`` — e.g. a scrape
    subprocess that received a 429 through a different fetch path — persisting it
    cross-process so peer processes (the foreground /research worker) back off."""
    _trip_circuit(_host_of(url), cooldown_seconds or _HOST_COOLDOWN_SEC)


def reset_circuit_breaker() -> None:
    """Clear all open circuits (test/maintenance helper). Removes only marker
    files this process wrote, leaving peer-owned cooldowns intact."""
    with _circuit_lock:
        _host_open_until.clear()
    for marker in list(_circuit_files_written):
        try:
            os.unlink(marker)
        except OSError:
            pass
    _circuit_files_written.clear()


# Public helpers for call sites that fetch with their own ``urlopen`` (Mercari /
# Rakuma / LLM store extractor) instead of ``HttpClient``, so they share the
# same per-host circuit breaker and stop hammering a rate-limited host.
def host_cooldown_remaining(url: str) -> float:
    """Seconds left on this URL's host circuit (0.0 if closed). Check before
    issuing a raw request and skip the fetch when > 0."""
    return _circuit_remaining(_host_of(url))


def note_http_success(url: str) -> None:
    _clear_circuit(_host_of(url))


def note_http_error(url: str, exc: BaseException) -> None:
    """Trip the host circuit when a raw fetch hit HTTP 429."""
    if isinstance(exc, HTTPError) and exc.code == 429:
        _trip_circuit(_host_of(url), _cooldown_for_429(exc))


def _cooldown_for_429(exc: HTTPError) -> float:
    raw = exc.headers.get("Retry-After") if exc.headers else None
    if raw:
        try:
            return max(_HOST_COOLDOWN_SEC, float(raw))
        except ValueError:
            pass
    return _HOST_COOLDOWN_SEC


def _is_retryable(exc: Exception) -> bool:
    if isinstance(exc, HTTPError):
        return exc.code in _RETRYABLE_STATUS_CODES
    return isinstance(exc, (URLError, TimeoutError, socket.timeout))


def _retry_delay_seconds(exc: Exception, attempt: int) -> float:
    """Back-off duration before the next retry. Respects Retry-After for 429."""
    if isinstance(exc, HTTPError) and exc.code == 429:
        raw = exc.headers.get("Retry-After") if exc.headers else None
        if raw:
            try:
                return min(float(raw), _HTTP_RETRY_AFTER_CAP_SEC)
            except ValueError:
                pass
        return _HTTP_RETRY_AFTER_CAP_SEC
    jitter = random.uniform(0.5, 1.5)
    return _HTTP_RETRY_BASE_SEC * jitter * (2.0 ** (attempt - 1))


class HttpClient:
    def __init__(
        self,
        user_agent: str | None = None,
        timeout_seconds: int = 20,
        ssl_context: ssl.SSLContext | None = None,
    ) -> None:
        # Default to the shared human macOS-Chrome UA — a CLI-looking UA invites
        # marketplace blocks/429s (e.g. yuyu-tei). Callers may still override.
        self.user_agent = user_agent or bs.MAC_CHROME_UA
        self.timeout_seconds = timeout_seconds
        self.ssl_context = ssl_context or truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)

    def get_text(
        self,
        url: str,
        *,
        params: dict[str, str | list[str]] | None = None,
        encoding: str | None = "utf-8",
        headers: dict[str, str] | None = None,
        timeout_seconds: int | None = None,
        retries: int | None = None,
        curl_fallback: bool = True,
        requester: str | None = None,
        priority: str | None = None,
    ) -> str:
        """Fetch text. ``retries`` caps urllib attempts (default
        ``_HTTP_MAX_RETRIES``); pass ``retries=1`` + ``curl_fallback=False`` to
        fail fast and avoid amplifying load against a rate-limited host (a 429
        otherwise triggers a 30s back-off retry **and** a curl re-fetch, turning
        one polite request into three and prolonging the IP cooldown)."""
        target = url
        if params:
            query = urlencode(params, doseq=True)
            separator = "&" if "?" in url else "?"
            target = f"{url}{separator}{query}"

        request_headers = bs.http_headers({"Cache-Control": "no-cache"})
        request_headers["User-Agent"] = self.user_agent
        if headers:
            request_headers.update(headers)

        request = Request(
            target,
            headers=request_headers,
        )
        effective_timeout = timeout_seconds if timeout_seconds is not None else self.timeout_seconds
        max_attempts = retries if retries is not None else _HTTP_MAX_RETRIES
        host = _host_of(target)
        remaining = _circuit_remaining(host)
        if remaining > 0:
            logger.warning("HTTP GET short-circuited host=%s cooldown_remaining=%.0fs target=%s", host, remaining, target)
            raise HostRateLimitedError(host, remaining)
        # Shared host budget (#24): ask the cross-process coordinator before the
        # network. Manual priorities may wait briefly for a concurrency slot;
        # background callers fail fast so they can't starve a user request.
        permit = get_host_budget().acquire_fetch_slot(
            url=target, requester=requester, priority=priority,
            timeout_seconds=effective_timeout,
        )
        if not permit.granted:
            logger.warning(
                "HTTP GET budget-declined host=%s decision=%s reason=%s target=%s",
                host, permit.decision, permit.reason, target,
            )
            raise HostRateLimitedError(host, permit.wait_seconds,
                                       decision=permit.decision, reason=permit.reason)
        logger.debug("HTTP GET target=%s timeout_seconds=%s", target, effective_timeout)
        last_exc: Exception | None = None
        try:
            for attempt in range(1, max_attempts + 1):
                try:
                    with urlopen(request, timeout=effective_timeout, context=self.ssl_context) as response:
                        payload = response.read()
                        selected_encoding = encoding or response.headers.get_content_charset() or "utf-8"
                        text = payload.decode(selected_encoding, errors="replace")
                        logger.debug(
                            "HTTP GET completed target=%s status=%s bytes=%s encoding=%s",
                            target,
                            getattr(response, "status", "unknown"),
                            len(payload),
                            selected_encoding,
                        )
                        _clear_circuit(host)
                        return text
                except _TRANSIENT_HTTP_EXCEPTIONS as exc:
                    last_exc = exc
                    if isinstance(exc, HTTPError) and exc.code == 429:
                        _trip_circuit(host, _cooldown_for_429(exc))
                        get_host_budget().record_result(url=target, status=429, requester=requester)
                    if not _is_retryable(exc) or attempt == max_attempts:
                        break
                    delay = _retry_delay_seconds(exc, attempt)
                    logger.warning(
                        "HTTP GET transient error attempt=%d/%d target=%s status=%s; retrying in %.1fs",
                        attempt, max_attempts, target,
                        exc.code if isinstance(exc, HTTPError) else type(exc).__name__,
                        delay,
                    )
                    time.sleep(delay)

            assert last_exc is not None
            if not curl_fallback:
                raise last_exc
            # host cooldown > everything: if this error just opened the host's
            # cooldown (e.g. a live 429), do NOT amplify with a curl re-fetch —
            # one 429 must not spawn another request to a host we've decided is
            # rate-limited (#24/#25 review).
            if _circuit_remaining(host) > 0:
                logger.warning(
                    "HTTP GET host=%s cooldown open after error — skipping curl fallback (no amplification)",
                    host,
                )
                raise last_exc
            if isinstance(last_exc, HTTPError):
                logger.warning("HTTP GET failed target=%s status=%s; trying curl fallback", target, last_exc.code)
            elif isinstance(last_exc, URLError):
                logger.warning("HTTP GET failed target=%s reason=%s; trying curl fallback", target, last_exc.reason)
            else:
                logger.warning("HTTP GET timed out target=%s error=%s; trying curl fallback", target, last_exc)

            curl_text = self._get_text_with_curl(target=target, headers=request_headers, encoding=encoding, timeout=effective_timeout)
            if curl_text is not None:
                return curl_text
            raise last_exc
        finally:
            permit.release()

    def get_bytes(
        self,
        url: str,
        *,
        params: dict[str, str | list[str]] | None = None,
        headers: dict[str, str] | None = None,
        timeout_seconds: int | None = None,
        requester: str | None = None,
        priority: str | None = None,
    ) -> bytes:
        """Binary counterpart to ``get_text``. Uses the same urllib + curl
        fallback path, but never decodes the payload. Needed for image
        downloads (perceptual-hash crawler, etc.)."""
        target = url
        if params:
            query = urlencode(params, doseq=True)
            separator = "&" if "?" in url else "?"
            target = f"{url}{separator}{query}"

        request_headers = bs.http_headers({"Cache-Control": "no-cache"})
        request_headers["User-Agent"] = self.user_agent
        if headers:
            request_headers.update(headers)

        request = Request(target, headers=request_headers)
        effective_timeout = timeout_seconds if timeout_seconds is not None else self.timeout_seconds
        host = _host_of(target)
        remaining = _circuit_remaining(host)
        if remaining > 0:
            logger.warning("HTTP GET (bytes) short-circuited host=%s cooldown_remaining=%.0fs target=%s", host, remaining, target)
            raise HostRateLimitedError(host, remaining)
        permit = get_host_budget().acquire_fetch_slot(
            url=target, requester=requester, priority=priority,
            timeout_seconds=effective_timeout,
        )
        if not permit.granted:
            logger.warning(
                "HTTP GET (bytes) budget-declined host=%s decision=%s reason=%s target=%s",
                host, permit.decision, permit.reason, target,
            )
            raise HostRateLimitedError(host, permit.wait_seconds,
                                       decision=permit.decision, reason=permit.reason)
        logger.debug("HTTP GET (bytes) target=%s timeout_seconds=%s", target, effective_timeout)
        last_exc: Exception | None = None
        try:
            for attempt in range(1, _HTTP_MAX_RETRIES + 1):
                try:
                    with urlopen(request, timeout=effective_timeout, context=self.ssl_context) as response:
                        payload = response.read()
                        logger.debug(
                            "HTTP GET (bytes) completed target=%s status=%s bytes=%s",
                            target,
                            getattr(response, "status", "unknown"),
                            len(payload),
                        )
                        _clear_circuit(host)
                        return payload
                except _TRANSIENT_HTTP_EXCEPTIONS as exc:
                    last_exc = exc
                    if isinstance(exc, HTTPError) and exc.code == 429:
                        _trip_circuit(host, _cooldown_for_429(exc))
                        get_host_budget().record_result(url=target, status=429, requester=requester)
                    if not _is_retryable(exc) or attempt == _HTTP_MAX_RETRIES:
                        break
                    delay = _retry_delay_seconds(exc, attempt)
                    logger.warning(
                        "HTTP GET (bytes) transient error attempt=%d/%d target=%s status=%s; retrying in %.1fs",
                        attempt, _HTTP_MAX_RETRIES, target,
                        exc.code if isinstance(exc, HTTPError) else type(exc).__name__,
                        delay,
                    )
                    time.sleep(delay)

            assert last_exc is not None
            # host cooldown > everything: don't amplify a 429 with a curl re-fetch.
            if _circuit_remaining(host) > 0:
                logger.warning(
                    "HTTP GET (bytes) host=%s cooldown open after error — skipping curl fallback (no amplification)",
                    host,
                )
                raise last_exc
            if isinstance(last_exc, HTTPError):
                logger.warning("HTTP GET (bytes) failed target=%s status=%s; trying curl fallback", target, last_exc.code)
            else:
                logger.warning("HTTP GET (bytes) failed target=%s error=%s; trying curl fallback", target, last_exc)
            curl_bytes = self._get_bytes_with_curl(target=target, headers=request_headers, timeout=effective_timeout)
            if curl_bytes is not None:
                return curl_bytes
            raise last_exc
        finally:
            permit.release()

    def _get_bytes_with_curl(
        self, *, target: str, headers: dict[str, str], timeout: int | None,
    ) -> bytes | None:
        curl_path = shutil.which("curl.exe") or shutil.which("curl")
        if curl_path is None:
            return None
        command = [curl_path, "-L", "-sS", "--compressed", "-f"]
        if self.ssl_context.verify_mode == ssl.CERT_NONE:
            command.append("-k")
        for key, value in headers.items():
            command.extend(["-H", f"{key}: {value}"])
        command.append(target)
        effective_timeout = timeout if timeout is not None else self.timeout_seconds
        try:
            completed = subprocess.run(
                command, capture_output=True, check=True, timeout=effective_timeout,
            )
        except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            logger.warning("curl bytes fallback failed target=%s error=%s", target, exc)
            return None
        return completed.stdout

    def _get_text_with_curl(
        self,
        *,
        target: str,
        headers: dict[str, str],
        encoding: str | None,
        timeout: int | None = None,
    ) -> str | None:
        curl_path = shutil.which("curl.exe") or shutil.which("curl")
        if curl_path is None:
            return None

        command = [curl_path, "-L", "-sS", "--compressed", "-f"]
        if self.ssl_context.verify_mode == ssl.CERT_NONE:
            command.append("-k")
        for key, value in headers.items():
            command.extend(["-H", f"{key}: {value}"])
        command.append(target)

        effective_timeout = timeout if timeout is not None else self.timeout_seconds
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                check=True,
                timeout=effective_timeout,
            )
        except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            logger.warning("curl fallback failed target=%s error=%s", target, exc)
            return None

        selected_encoding = encoding or "utf-8"
        text = completed.stdout.decode(selected_encoding, errors="replace")
        logger.debug(
            "HTTP GET curl fallback completed target=%s bytes=%s encoding=%s",
            target,
            len(completed.stdout),
            selected_encoding,
        )
        return text
