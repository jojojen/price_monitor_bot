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
from dataclasses import dataclass
from typing import Any, Sequence
from urllib.parse import urljoin, urlparse

from .models import ExtractionExample
from .official_store_base import (
    AVAILABLE,
    COMING_SOON,
    LOTTERY_OPEN,
    PREORDER_OPEN,
    STATUS_UNKNOWN,
    OfficialStoreListing,
    item_key_from_url,
)


@dataclass(frozen=True, slots=True)
class SingleProductExtraction:
    """Result of extracting price+title for a single product page (feedback flow)."""

    price_jpy: int | None
    title: str
    raw_response: str
    error: str | None = None

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


# ── Rule-based fast-path extraction for feedback URLs ────────────────────────

# Sanity bounds for TCG sealed-box / card prices in JPY. Drops dust like
# shipping fees, point balances, and absurd outliers.
_RULE_PRICE_MIN_JPY = 100
_RULE_PRICE_MAX_JPY = 10_000_000


def _rule_based_price_extraction(html: str, *, base_url: str) -> "SingleProductExtraction | None":
    """Try to extract (price_jpy, title) from structured HTML markers before
    calling the LLM. Returns None when no plausible price is found — caller
    falls back to qwen.

    Sources tried in order:
      1. <script type="application/ld+json"> Schema.org Product / Offer (most reliable)
      2. <meta property="og:price:amount"> / "product:price:amount"
      3. Visible "¥12,345" in HTML body (cheapest signal, last resort)
    """
    # 1) JSON-LD
    ld_blocks = re.findall(
        r'<script[^>]*type=["\']application/ld\+json["\'][^>]*>(.*?)</script>',
        html,
        flags=re.DOTALL | re.IGNORECASE,
    )
    for block in ld_blocks:
        price, title = _extract_from_jsonld(block)
        if price is not None and _RULE_PRICE_MIN_JPY <= price <= _RULE_PRICE_MAX_JPY:
            return SingleProductExtraction(
                price_jpy=price, title=title or "", raw_response="rule:jsonld", error=None,
            )

    # 2) OG / product meta tags
    meta_amount = re.search(
        r'<meta[^>]+(?:property|name)=["\'](?:og:price:amount|product:price:amount)["\'][^>]*content=["\']([^"\']+)["\']',
        html, flags=re.IGNORECASE,
    )
    if meta_amount:
        price = _coerce_price(meta_amount.group(1))
        if price is not None and _RULE_PRICE_MIN_JPY <= price <= _RULE_PRICE_MAX_JPY:
            title_m = re.search(
                r'<meta[^>]+(?:property|name)=["\']og:title["\'][^>]*content=["\']([^"\']+)["\']',
                html, flags=re.IGNORECASE,
            )
            title = (title_m.group(1).strip() if title_m else "")
            return SingleProductExtraction(
                price_jpy=price, title=title, raw_response="rule:og", error=None,
            )

    # 3) Visible "¥XX,XXX" — pick the first plausible occurrence (skip stripped scripts)
    body = re.sub(r"<script[^>]*>.*?</script>", " ", html, flags=re.DOTALL | re.IGNORECASE)
    body = re.sub(r"<style[^>]*>.*?</style>", " ", body, flags=re.DOTALL | re.IGNORECASE)
    yen_match = re.search(r"¥\s*([\d,]{3,15})", body)
    if yen_match:
        price = _coerce_price(yen_match.group(1))
        if price is not None and _RULE_PRICE_MIN_JPY <= price <= _RULE_PRICE_MAX_JPY:
            title_m = re.search(r"<title>([^<]+)</title>", html, flags=re.IGNORECASE)
            title = (title_m.group(1).strip() if title_m else "")
            return SingleProductExtraction(
                price_jpy=price, title=title, raw_response="rule:visible_yen", error=None,
            )

    return None


def _extract_from_jsonld(block: str) -> "tuple[int | None, str]":
    """Parse one LD-JSON block; return (price_jpy, title) if it carries a
    Product offer. Quietly returns (None, "") on any failure — the caller
    walks through several blocks before giving up."""
    try:
        data = json.loads(block.strip())
    except Exception:
        return None, ""
    candidates: list[dict] = []
    if isinstance(data, list):
        candidates.extend(d for d in data if isinstance(d, dict))
    elif isinstance(data, dict):
        candidates.append(data)
    for obj in candidates:
        offers = obj.get("offers") or obj.get("offer")
        offer_list: list[dict] = []
        if isinstance(offers, list):
            offer_list.extend(o for o in offers if isinstance(o, dict))
        elif isinstance(offers, dict):
            offer_list.append(offers)
        # Some sites put "price" directly on the Product
        if "price" in obj and not offer_list:
            offer_list.append(obj)
        for off in offer_list:
            for key in ("price", "lowPrice", "lowestPrice"):
                raw = off.get(key)
                if raw is None:
                    continue
                price = _coerce_price(raw)
                if price is None:
                    continue
                title = str(obj.get("name") or off.get("name") or "").strip()
                return price, title
    return None, ""


def _coerce_price(raw: object) -> int | None:
    """Robust int conversion: accepts "12,345", "12345.00", 12345, 12345.0."""
    if isinstance(raw, bool):  # bool is int subclass, exclude
        return None
    if isinstance(raw, (int, float)):
        try:
            value = int(raw)
        except (ValueError, OverflowError):
            return None
        return value if value > 0 else None
    if isinstance(raw, str):
        cleaned = raw.strip().replace(",", "").replace("¥", "").replace("円", "")
        cleaned = re.sub(r"\.\d+$", "", cleaned)
        if not cleaned.isdigit():
            return None
        try:
            value = int(cleaned)
        except ValueError:
            return None
        return value if value > 0 else None
    return None


# ── Few-shot prompt builder (single-product extraction) ──────────────────────

def _build_single_product_system_prompt(
    *,
    game: str,
    item_kind: str,
    few_shot_examples: Sequence[ExtractionExample],
) -> str:
    base = (
        "You are a price extractor for Japanese TCG product pages. "
        "Given the plain-text content of one product page, identify the price "
        "of the product the page is selling. Return ONLY a JSON object with "
        "two fields: \"price_jpy\" (integer Japanese yen, or null if no price "
        "is shown) and \"title\" (the product name as shown). Do not output "
        "anything else. If the page lists multiple products, pick the one "
        "that best matches the 'Item we are pricing' hint from the user "
        "message. Reject obvious shipping costs, points balances, and other "
        "non-product prices."
    )
    if not few_shot_examples:
        return base

    lines = [base, "", f"Recent verified prices for {game}/{item_kind}:"]
    for example in few_shot_examples:
        title = (example.title or "").strip().replace("\n", " ")
        # Cap title length to keep prompt compact
        if len(title) > 80:
            title = title[:77] + "..."
        lines.append(f'  - "{title}" from {example.domain}: ¥{example.price_jpy:,}')
    lines.append(
        "Use these as a sanity anchor: extracted prices should be in a "
        "comparable order of magnitude unless the page explicitly indicates "
        "a different product type."
    )
    return "\n".join(lines)


# ── HTML → plain-text helper ─────────────────────────────────────────────────

def _html_to_text(html: str) -> str:
    """Convert HTML to plain text suitable for an LLM prompt.

    * Removes ``<script>`` and ``<style>`` blocks entirely, **except**
      ``<script type="application/ld+json">`` which carries Schema.org
      structured data (price, name, …) on many marketplaces.
    * Replaces ``<a href="…">text</a>`` with ``text [href]`` so URLs survive.
    * Strips all remaining tags.
    * Collapses excessive whitespace while preserving line breaks.
    """
    # Preserve LD-JSON blocks before stripping all <script>s — many sites
    # (snkrdunk, mercari, …) carry the canonical product price inside
    # <script type="application/ld+json">. Inline them as visible JSON so the
    # downstream regex / LLM can find the price.
    ld_blocks = re.findall(
        r'<script[^>]*type=["\']application/ld\+json["\'][^>]*>(.*?)</script>',
        html,
        flags=re.DOTALL | re.IGNORECASE,
    )
    text = re.sub(r"<script[^>]*>.*?</script>", " ", html, flags=re.DOTALL | re.IGNORECASE)
    if ld_blocks:
        text = "[LD-JSON]\n" + "\n".join(ld_blocks) + "\n[/LD-JSON]\n" + text
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

    # ── Single-product price extraction (feedback flow) ──────────────────────

    def extract_price_for_feedback(
        self,
        html: str,
        *,
        base_url: str,
        item_title_hint: str = "",
        game: str = "",
        item_kind: str = "",
        few_shot_examples: Sequence[ExtractionExample] = (),
        temperature: float = 0.7,
        skip_rule_based: bool = False,
    ) -> SingleProductExtraction:
        # Rule-based fast path: JSON-LD / og:price / "¥XX,XXX" regex. Deterministic,
        # ~5ms vs ~30-60s for qwen. Only fall through to the LLM when rules
        # fail to yield a plausible price.
        if not skip_rule_based:
            rule_result = _rule_based_price_extraction(html, base_url=base_url)
            if rule_result is not None:
                return rule_result
        """Extract `(price_jpy, title)` for the single product on the page that
        best matches the given item context. Designed for user-submitted
        reference URLs (feedback loop).

        `few_shot_examples`, when non-empty, are prepended to the system
        prompt as a sanity anchor: "Recent verified prices for game/item_kind:
          - ... ¥..." This is how the system gets gradually more accurate
        without code changes — high-confidence past extractions feed back into
        future calls.

        Returns a SingleProductExtraction. `price_jpy is None` means the LLM
        either couldn't find a price or returned an invalid response; the
        caller decides how to treat this (we still record the feedback event
        so `vote_count` gets bumped, but `consensus_fail_count` ticks up).
        """
        try:
            cleaned = _html_to_text(html)
            if self.max_html_chars and len(cleaned) > self.max_html_chars:
                cleaned = cleaned[: self.max_html_chars]
        except Exception as exc:
            return SingleProductExtraction(
                price_jpy=None, title="", raw_response="", error=f"html_to_text: {exc}"
            )

        system_prompt = _build_single_product_system_prompt(
            game=game, item_kind=item_kind, few_shot_examples=few_shot_examples,
        )
        user_prompt = (
            f"Product page URL: {base_url}\n"
            + (f"Item we are pricing: {item_title_hint}\n" if item_title_hint else "")
            + "\nPage content (plain text):\n"
            + f"{cleaned}\n\n"
            + 'Return ONLY a JSON object: {"price_jpy": <integer or null>, "title": "<best matching product name>"}.'
        )

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "format": {"type": "json_object"},
            "options": {"temperature": float(temperature)},
        }
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.ollama_url, data=body,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                response_text = resp.read().decode("utf-8")
        except Exception as exc:
            return SingleProductExtraction(
                price_jpy=None, title="", raw_response="", error=f"ollama: {exc}"
            )

        try:
            data = json.loads(response_text)
            content = data["message"]["content"]
            parsed = json.loads(content)
        except (json.JSONDecodeError, KeyError) as exc:
            return SingleProductExtraction(
                price_jpy=None, title="", raw_response=response_text, error=f"parse: {exc}"
            )

        price_raw = parsed.get("price_jpy") if isinstance(parsed, dict) else None
        price_jpy: int | None
        if price_raw is None:
            price_jpy = None
        else:
            try:
                price_jpy = int(price_raw)
                if price_jpy <= 0:
                    price_jpy = None
            except (TypeError, ValueError):
                price_jpy = None

        title = str(parsed.get("title") or "").strip() if isinstance(parsed, dict) else ""

        return SingleProductExtraction(
            price_jpy=price_jpy, title=title, raw_response=content,
        )

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
