from __future__ import annotations

import logging
import socket
import shutil
import ssl
import subprocess
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import truststore

logger = logging.getLogger(__name__)
_TRANSIENT_HTTP_EXCEPTIONS = (HTTPError, URLError, TimeoutError, socket.timeout)


class HttpClient:
    def __init__(
        self,
        user_agent: str | None = None,
        timeout_seconds: int = 20,
        ssl_context: ssl.SSLContext | None = None,
    ) -> None:
        self.user_agent = user_agent or "OpenClawPriceMonitor/0.1 (+https://local-dev)"
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
    ) -> str:
        target = url
        if params:
            query = urlencode(params, doseq=True)
            separator = "&" if "?" in url else "?"
            target = f"{url}{separator}{query}"

        request_headers = {
            "User-Agent": self.user_agent,
            "Accept-Language": "ja-JP,ja;q=0.9",
            "Cache-Control": "no-cache",
        }
        if headers:
            request_headers.update(headers)

        request = Request(
            target,
            headers=request_headers,
        )
        effective_timeout = timeout_seconds if timeout_seconds is not None else self.timeout_seconds
        logger.debug("HTTP GET target=%s timeout_seconds=%s", target, effective_timeout)
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
                return text
        except _TRANSIENT_HTTP_EXCEPTIONS as exc:
            if isinstance(exc, HTTPError):
                logger.warning("HTTP GET failed target=%s status=%s; trying curl fallback", target, exc.code)
            elif isinstance(exc, URLError):
                logger.warning("HTTP GET failed target=%s reason=%s; trying curl fallback", target, exc.reason)
            else:
                logger.warning("HTTP GET timed out target=%s error=%s; trying curl fallback", target, exc)

            curl_text = self._get_text_with_curl(target=target, headers=request_headers, encoding=encoding, timeout=effective_timeout)
            if curl_text is not None:
                return curl_text
            raise

    def get_bytes(
        self,
        url: str,
        *,
        params: dict[str, str | list[str]] | None = None,
        headers: dict[str, str] | None = None,
        timeout_seconds: int | None = None,
    ) -> bytes:
        """Binary counterpart to ``get_text``. Uses the same urllib + curl
        fallback path, but never decodes the payload. Needed for image
        downloads (perceptual-hash crawler, etc.)."""
        target = url
        if params:
            query = urlencode(params, doseq=True)
            separator = "&" if "?" in url else "?"
            target = f"{url}{separator}{query}"

        request_headers = {
            "User-Agent": self.user_agent,
            "Accept-Language": "ja-JP,ja;q=0.9",
            "Cache-Control": "no-cache",
        }
        if headers:
            request_headers.update(headers)

        request = Request(target, headers=request_headers)
        effective_timeout = timeout_seconds if timeout_seconds is not None else self.timeout_seconds
        logger.debug("HTTP GET (bytes) target=%s timeout_seconds=%s", target, effective_timeout)
        try:
            with urlopen(request, timeout=effective_timeout, context=self.ssl_context) as response:
                payload = response.read()
                logger.debug(
                    "HTTP GET (bytes) completed target=%s status=%s bytes=%s",
                    target,
                    getattr(response, "status", "unknown"),
                    len(payload),
                )
                return payload
        except _TRANSIENT_HTTP_EXCEPTIONS as exc:
            if isinstance(exc, HTTPError):
                logger.warning("HTTP GET (bytes) failed target=%s status=%s; trying curl fallback", target, exc.code)
            else:
                logger.warning("HTTP GET (bytes) failed target=%s error=%s; trying curl fallback", target, exc)
            curl_bytes = self._get_bytes_with_curl(target=target, headers=request_headers, timeout=effective_timeout)
            if curl_bytes is not None:
                return curl_bytes
            raise

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
