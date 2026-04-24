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
        logger.debug("HTTP GET target=%s timeout_seconds=%s", target, self.timeout_seconds)
        try:
            with urlopen(request, timeout=self.timeout_seconds, context=self.ssl_context) as response:
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

            curl_text = self._get_text_with_curl(target=target, headers=request_headers, encoding=encoding)
            if curl_text is not None:
                return curl_text
            raise

    def _get_text_with_curl(
        self,
        *,
        target: str,
        headers: dict[str, str],
        encoding: str | None,
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

        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                check=True,
                timeout=self.timeout_seconds,
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
