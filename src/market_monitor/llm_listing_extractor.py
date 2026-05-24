"""LLM-assisted listing extractor using a local Ollama model.

Replaces brittle CSS-selector crawlers with a prompt-based approach:
given raw HTML + page URL + store name, the LLM returns a JSON array of
pre-order/lottery listings.  No extra dependencies — uses only stdlib
``urllib.request``.
"""

from __future__ import annotations

import json
import logging
import re
import urllib.request
from typing import Any
from urllib.parse import urljoin, urlparse

from .official_store_base import (
    AVAILABLE,
    COMING_SOON,
    LOTTERY_OPEN,
    PREORDER_OPEN,
    STATUS_UNKNOWN,
    OfficialStoreListing,
    item_key_from_url,
)

logger = logging.getLogger(__name__)

_OLLAMA_URL = "http://localhost:11434/api/chat"
_DEFAULT_MODEL = "qwen3:4b"

_VALID_STATUSES = frozenset({
    "lottery_open", "preorder_open", "available", "coming_soon", "unknown",
})

_SYSTEM_PROMPT = (
    "You are a structured data extractor specialising in Japanese TCG "
    "(trading card game) pre-order and lottery listings.  "
    "Given the plain-text content of a Japanese online store page, extract "
    "every product that is a pre-order, lottery entry, or currently on sale.  "
    "Return ONLY a valid JSON object with a single key \"listings\" whose "
    "value is an array.  Each element must have these fields:\n"
    "  title        (string)  — product name in the original language\n"
    "  url          (string)  — absolute or relative URL to the product page, "
    "empty string if unavailable\n"
    "  status       (string)  — one of: lottery_open, preorder_open, available, "
    "coming_soon, unknown\n"
    "  price_jpy    (integer or null) — price in Japanese yen, null if unknown\n"
    "  deadline_iso (string or null)  — ISO 8601 date/datetime of entry deadline, "
    "null if unknown\n"
    "  open_date_iso(string or null)  — ISO 8601 date/datetime when lottery/preorder "
    "opens, null if unknown\n"
    "Output nothing besides the JSON object."
)


# ── HTML → plain-text helper ─────────────────────────────────────────────────

def _html_to_text(html: str) -> str:
    """Convert HTML to plain text suitable for an LLM prompt.

    * Removes ``<script>`` and ``<style>`` blocks entirely.
    * Replaces ``<a href="…">text</a>`` with ``text [href]`` so URLs survive.
    * Strips all remaining tags.
    * Collapses excessive whitespace while preserving line breaks.
    """
    # Remove script / style blocks (including content)
    text = re.sub(r"<script[^>]*>.*?</script>", " ", html, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<style[^>]*>.*?</style>", " ", text, flags=re.DOTALL | re.IGNORECASE)

    # Replace anchor tags: keep both link text and href
    def _replace_anchor(m: re.Match) -> str:
        attrs = m.group(1)
        inner = m.group(2)
        href_m = re.search(r'href=["\']([^"\']+)["\']', attrs)
        href = href_m.group(1) if href_m else ""
        inner_clean = re.sub(r"<[^>]+>", "", inner).strip()
        if href and inner_clean:
            return f"{inner_clean} [{href}]"
        return inner_clean or href

    text = re.sub(
        r"<a([^>]*)>(.*?)</a>",
        _replace_anchor,
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )

    # Strip remaining tags
    text = re.sub(r"<[^>]+>", " ", text)

    # Decode common HTML entities
    text = text.replace("&nbsp;", " ").replace("&amp;", "&")
    text = text.replace("&lt;", "<").replace("&gt;", ">")
    text = text.replace("&quot;", '"').replace("&#39;", "'")

    # Collapse whitespace — preserve single newlines
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    # Remove consecutive blank lines
    result_lines: list[str] = []
    prev_blank = False
    for line in lines:
        if not line:
            if not prev_blank:
                result_lines.append("")
            prev_blank = True
        else:
            result_lines.append(line)
            prev_blank = False

    return "\n".join(result_lines).strip()


# ── Main extractor class ──────────────────────────────────────────────────────

class LlmListingExtractor:
    """Calls a local Ollama model to extract TCG listings from HTML.

    Parameters
    ----------
    model:
        Ollama model tag to use (must be pulled on the local Ollama instance).
    ollama_url:
        Base URL of the Ollama ``/api/chat`` endpoint.
    timeout:
        HTTP timeout in seconds for the Ollama call.
    max_html_chars:
        Truncate cleaned text to this many characters before sending to the LLM
        to avoid exceeding context windows.
    """

    def __init__(
        self,
        *,
        model: str = _DEFAULT_MODEL,
        ollama_url: str = _OLLAMA_URL,
        timeout: int = 30,
        max_html_chars: int = 12_000,
    ) -> None:
        self.model = model
        self.ollama_url = ollama_url
        self.timeout = timeout
        self.max_html_chars = max_html_chars

    # ── Public API ────────────────────────────────────────────────────────────

    def extract(
        self,
        html: str,
        *,
        store_name: str,
        base_url: str,
    ) -> list[OfficialStoreListing]:
        """Extract listings from *html* and return them as ``OfficialStoreListing`` objects.

        Returns an empty list on any error (network, JSON parse, etc.) so the
        calling crawler degrades gracefully.
        """
        try:
            cleaned = _html_to_text(html)
            if self.max_html_chars and len(cleaned) > self.max_html_chars:
                cleaned = cleaned[: self.max_html_chars]

            raw_listings = self._call_ollama(cleaned, base_url=base_url)
            return self._to_official_listings(raw_listings, store_name=store_name, base_url=base_url)
        except Exception:
            logger.exception(
                "LlmListingExtractor.extract failed store=%s url=%s", store_name, base_url
            )
            return []

    # ── Ollama call ───────────────────────────────────────────────────────────

    def _call_ollama(self, text: str, *, base_url: str) -> list[dict[str, Any]]:
        """Send *text* to Ollama and parse the JSON response.

        Returns the raw list of listing dicts from the LLM.
        Raises on HTTP / JSON errors — callers should catch.
        """
        user_prompt = (
            f"Store page URL: {base_url}\n\n"
            "Page content (plain text):\n"
            f"{text}\n\n"
            "Extract all TCG pre-order/lottery/available listings and return "
            "the result as a JSON object with key \"listings\"."
        )

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "format": {"type": "json_object"},
        }

        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.ollama_url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            response_text = resp.read().decode("utf-8")

        response_data = json.loads(response_text)
        # Ollama wraps the assistant reply in message.content
        content = response_data["message"]["content"]
        parsed = json.loads(content)

        # Accept either {"listings": [...]} or a bare array
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict):
            listings = parsed.get("listings")
            if isinstance(listings, list):
                return listings
        logger.warning(
            "LlmListingExtractor: unexpected JSON structure from Ollama: %r", type(parsed)
        )
        return []

    # ── Conversion helpers ────────────────────────────────────────────────────

    def _to_official_listings(
        self,
        raw: list[dict[str, Any]],
        *,
        store_name: str,
        base_url: str,
    ) -> list[OfficialStoreListing]:
        results: list[OfficialStoreListing] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            title = str(item.get("title") or "").strip()
            if not title:
                continue

            url_raw = str(item.get("url") or "").strip()
            url = urljoin(base_url, url_raw) if url_raw else base_url

            status_raw = str(item.get("status") or STATUS_UNKNOWN).strip().lower()
            status = status_raw if status_raw in _VALID_STATUSES else STATUS_UNKNOWN

            price_raw = item.get("price_jpy")
            price_jpy: int | None = None
            if price_raw is not None:
                try:
                    price_jpy = int(price_raw)
                except (TypeError, ValueError):
                    pass

            deadline_iso = item.get("deadline_iso") or None
            open_date_iso = item.get("open_date_iso") or None

            item_key = item_key_from_url(url) if url != base_url else f"{store_name}:{title[:60]}"

            results.append(
                OfficialStoreListing(
                    store_name=store_name,
                    item_key=item_key,
                    title=title,
                    url=url,
                    status=status,
                    price_jpy=price_jpy,
                    deadline_iso=deadline_iso,
                    open_date_iso=open_date_iso,
                )
            )
        return results
