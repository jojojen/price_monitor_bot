"""Yuyutei (yuyu-tei.jp) sell-listing search client.

Yuyu亭 is Japan's largest TCG card shop. This module implements
``MarketplaceSearchClient`` for the *sell* (ask) side only — yuyutei
publishes fixed sell prices directly on the web without authentication.

Avoids importing ``tcg_tracker`` (which imports ``market_monitor.http``) to
prevent a circular dependency. Only the fields needed by ``MarketplaceListing``
are extracted; card-specific metadata (rarity, card_number) is left to the
``tcg_tracker`` layer.
"""

from __future__ import annotations

import logging
import re
import statistics
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
from urllib.parse import urljoin, urlparse, urlencode

from bs4 import BeautifulSoup

from .host_budget import DEFAULT_PRIORITY, REQUESTER_RESEARCH
from .http import HostRateLimitedError, HttpClient
from .marketplace_search import MarketplaceListing

logger = logging.getLogger(__name__)

YUYUTEI_BASE_URL = "https://yuyu-tei.jp"

# Game codes supported in the sell path: /sell/{code}/s/search
# poc = Pokémon  ygo = 遊戯王  ws = Weiss Schwarz  ua = Union Arena  op = One Piece
DEFAULT_GAME_CODES: tuple[str, ...] = ("ygo", "ws", "ua", "op")
KNOWN_SELL_CODES: frozenset[str] = frozenset({"poc", "ygo", "ws", "ua", "op"})

# Adapter-level routing table: which yuyutei sell path a query belongs to, keyed
# by the game *name* the user typed. This is the client's own URL-routing config
# (like a base URL), NOT open-world content recognition — deciding that a bare
# character name like「ピカチュウ」is Pokémon is left to upstream LLM callers, who
# pass the resolved game via source_options["game_code"]. When neither a game
# word nor an explicit code is present we skip Yuyutei rather than blindly
# fanning out across every category and burning the host's rate limit.
_GAME_WORD_TO_SELL_CODE: dict[str, str] = {
    "pokemon": "poc", "ptcg": "poc", "ポケモン": "poc", "寶可夢": "poc",
    "宝可梦": "poc", "寶可卡": "poc", "宝可卡": "poc",
    "ws": "ws", "weiss": "ws", "ヴァイス": "ws",
    "yugioh": "ygo", "ygo": "ygo", "遊戯王": "ygo", "遊戲王": "ygo", "游戏王": "ygo",
    "ua": "ua", "unionarena": "ua", "ユニオンアリーナ": "ua",
    "op": "op", "onepiece": "op", "optcg": "op", "opcg": "op",
    "ワンピース": "op", "航海王": "op", "海賊王": "op",
}
_TOKEN_SPLIT_RE = re.compile(r"[\s/・,，、_\-]+")
_CJK_GAME_WORDS = tuple(w for w in _GAME_WORD_TO_SELL_CODE if any(ord(c) > 0x3000 for c in w))


def resolve_sell_code(value: str | None) -> str | None:
    """Map a free-text query (or an explicit game token) to a single yuyutei
    sell code, or ``None`` when no game can be identified. Reused for both the
    auto-routing default and the ``source_options['game_code']`` override."""
    if not value:
        return None
    raw = value.strip().lower()
    if raw in KNOWN_SELL_CODES:
        return raw
    for token in _TOKEN_SPLIT_RE.split(raw):
        code = _GAME_WORD_TO_SELL_CODE.get(token)
        if code:
            return code
    for word in _CJK_GAME_WORDS:
        if word in value:
            return _GAME_WORD_TO_SELL_CODE[word]
    return None


_PRICE_DIGITS_RE = re.compile(r"(\d[\d,]*)")
_STOCK_COUNT_RE = re.compile(r"(\d+)\s*点")


def _parse_jpy(text: str) -> int | None:
    if not text:
        return None
    match = _PRICE_DIGITS_RE.search(text)
    if match is None:
        return None
    try:
        return int(match.group(1).replace(",", ""))
    except ValueError:
        return None


def _parse_stock(card) -> tuple[bool, int | None]:
    """Return ``(in_stock, stock_count)`` for a ``div.card-product`` element.

    Yuyutei marks per-card stock in ``label.cart_sell_zaiko`` as either
    「在庫 : ×」(out of stock) or 「在庫 : N 点」(N units in stock). Some pages
    (and the synthetic test fixtures) omit the label entirely for in-stock
    cards. So: label absent → in stock (count unknown); 「×」/「在庫なし」/empty →
    out of stock; 「N 点」→ in stock with count N. ``stock_count`` is ``None``
    when the count can't be read."""
    label = card.select_one("label.cart_sell_zaiko")
    if label is None:
        return True, None
    text = label.get_text(" ", strip=True).replace("在庫 :", "").strip()
    if not text or "×" in text or "在庫なし" in text:
        return False, None
    match = _STOCK_COUNT_RE.search(text)
    return True, (int(match.group(1)) if match else None)


def _parse_card_listings(
    html: str, *, require_in_stock: bool, base_url: str = YUYUTEI_BASE_URL
) -> list[dict]:
    """Parse a yuyutei sell/buy search page into raw dicts
    (item_id, title, price_jpy, url, stock_count). When ``require_in_stock`` is
    set, out-of-stock cards are dropped — used for the *sell* side, where a
    price with no stock is not a purchasable reference."""
    soup = BeautifulSoup(html, "html.parser")
    results: list[dict] = []

    for card in soup.select("div.card-product"):
        in_stock, stock_count = _parse_stock(card)
        if require_in_stock and not in_stock:
            continue

        anchors = [a for a in card.select("a[href]") if "/card/" in a.get("href", "")]
        title_el = card.find("h4")
        price_el = card.find("strong")

        if not anchors or title_el is None or price_el is None:
            continue

        href = urljoin(base_url, anchors[0]["href"])
        path_parts = [p for p in urlparse(href).path.split("/") if p]
        version_code = path_parts[-2] if len(path_parts) >= 2 else ""
        product_id = path_parts[-1] if path_parts else href
        item_id = f"{version_code}:{product_id}"

        title = title_el.get_text(" ", strip=True)
        price_jpy = _parse_jpy(price_el.get_text(" ", strip=True))
        if price_jpy is None or price_jpy <= 0:
            continue

        results.append({
            "item_id": item_id,
            "title": title,
            "price_jpy": price_jpy,
            "url": href,
            "stock_count": stock_count,
        })

    return results


def _parse_sell_listings(html: str, *, game_code: str, base_url: str = YUYUTEI_BASE_URL) -> list[dict]:
    """Parse yuyutei *sell* (販売) search HTML → in-stock listings only.

    Out-of-stock cards (在庫「×」/「在庫なし」) are skipped: a sell price with no
    stock isn't a purchasable upper-bound reference. ``game_code`` is accepted
    for call-site clarity but not needed by the parse itself."""
    return _parse_card_listings(html, require_in_stock=True, base_url=base_url)


def _parse_buy_listings(html: str, *, base_url: str = YUYUTEI_BASE_URL) -> list[dict]:
    """Parse yuyutei *buy* (買取) search HTML. Buy is a standing offer (the shop
    pays this to acquire the card), so there is no per-card stock to gate on."""
    return _parse_card_listings(html, require_in_stock=False, base_url=base_url)


@dataclass(frozen=True)
class YuyuteiReferenceBand:
    """Shop-price reference band for a query: 買取 (what the shop pays — the
    lower / liquidation side) and in-stock 販売 (what the shop charges — the
    upper / acquisition side). Kept separate from C2C active listings so shop
    prices don't get folded into a Mercari/Rakuma median."""

    game_code: str
    buy_prices: tuple[int, ...]
    sell_prices: tuple[int, ...]
    sell_stock_total: int
    sample_urls: tuple[str, ...]
    # Verbatim matched-listing titles (e.g. "リザードンex SAR 200/165"). They carry
    # the card number / rarity / set / box name already fetched on this band, so a
    # caller can record the item's real identity without any extra request.
    sample_titles: tuple[str, ...] = ()

    @property
    def has_data(self) -> bool:
        return bool(self.buy_prices or self.sell_prices)

    @property
    def buy_reference(self) -> int | None:
        return int(statistics.median(self.buy_prices)) if self.buy_prices else None

    @property
    def sell_reference(self) -> int | None:
        return int(statistics.median(self.sell_prices)) if self.sell_prices else None

    @property
    def buy_min(self) -> int | None:
        return min(self.buy_prices) if self.buy_prices else None

    @property
    def buy_max(self) -> int | None:
        return max(self.buy_prices) if self.buy_prices else None

    @property
    def sell_min(self) -> int | None:
        return min(self.sell_prices) if self.sell_prices else None

    @property
    def sell_max(self) -> int | None:
        return max(self.sell_prices) if self.sell_prices else None


class YuyuteiMarketplaceSearchClient:
    """``MarketplaceSearchClient`` for Yuyutei sell listings.

    Iterates over ``game_codes``, fetches the search page for each, then
    deduplicates, filters, sorts, and caps the combined result.

    ``source_options`` key recognised: ``"game_code"`` (str) — when set, only
    that single game code is queried.

    Routing: if ``game_codes`` is left at its default (``None``), each query is
    routed to a *single* sell code inferred from the query text, and Yuyutei is
    skipped entirely when no game can be identified — this avoids the previous
    behaviour of firing one request per category (4–5×) per search, which
    repeatedly tripped yuyu-tei.jp's IP rate limit. Callers that genuinely want
    multi-category fan-out can still pass an explicit ``game_codes`` tuple."""

    source_name: str = "yuyutei"

    def __init__(
        self,
        http_client: HttpClient | None = None,
        game_codes: tuple[str, ...] | None = None,
        *,
        requester: str = REQUESTER_RESEARCH,
        priority: str = DEFAULT_PRIORITY,
    ) -> None:
        self._http = http_client or HttpClient()
        self._game_codes = game_codes
        # Host-budget identity (#24). Defaults to the lowest (background)
        # priority; the /research command constructs this client with the manual
        # priority so user-driven lookups can claim Yuyutei's single slot ahead
        # of background enrichment.
        self._requester = requester
        self._priority = priority
        # Set when the last fetch was declined by the shared host budget BEFORE
        # any network call (cooldown / concurrency). Lets callers distinguish a
        # pre-network skip from a genuine "no data" empty result (#24/#25 D5).
        self.last_budget_skip: HostRateLimitedError | None = None

    def search(
        self,
        query: str,
        *,
        price_max: int,
        max_results: int = 30,
        timeout_ms: int = 30_000,
        source_options: Mapping[str, Any] | None = None,
    ) -> list[MarketplaceListing]:
        self.last_budget_skip = None
        if not query.strip() or price_max <= 0:
            return []

        opts = dict(source_options or {})
        if "game_code" in opts:
            code = resolve_sell_code(str(opts["game_code"])) or str(opts["game_code"]).strip().lower()
            codes: tuple[str, ...] = (code,)
        elif self._game_codes is not None:
            codes = self._game_codes
        else:
            resolved = resolve_sell_code(query)
            if resolved is None:
                logger.info(
                    "Yuyutei: skipping query=%s (no identifiable TCG game; not fanning out to avoid rate limit)",
                    query,
                )
                return []
            codes = (resolved,)

        params = urlencode({"search_word": query, "rare": "", "type": ""})
        seen_ids: set[str] = set()
        raw_hits: list[dict] = []

        for code in codes:
            url = f"{YUYUTEI_BASE_URL}/sell/{code}/s/search?{params}"
            try:
                # Fail fast: no 30s retry, no curl re-fetch — a 429 must not be
                # amplified into extra requests that prolong the IP cooldown.
                html = self._http.get_text(
                    url, timeout_seconds=timeout_ms / 1000.0, retries=1, curl_fallback=False,
                    requester=self._requester, priority=self._priority,
                )
            except HostRateLimitedError as exc:
                # Pre-network budget skip (cooldown / concurrency) — distinct from
                # a live 429, so the cause is legible in logs (#25 D5).
                logger.warning(
                    "YuyuteiMarketplaceSearchClient: budget skip code=%s decision=%s reason=%s",
                    code, exc.decision, exc.reason,
                )
                self.last_budget_skip = exc
                continue
            except Exception:
                logger.warning("YuyuteiMarketplaceSearchClient: HTTP failed code=%s url=%s", code, url)
                continue
            for hit in _parse_sell_listings(html, game_code=code):
                if hit["item_id"] in seen_ids:
                    continue
                if hit["price_jpy"] > price_max:
                    continue
                seen_ids.add(hit["item_id"])
                raw_hits.append(hit)

        raw_hits.sort(key=lambda h: h["price_jpy"])
        return [
            MarketplaceListing(
                source="yuyutei",
                item_id=h["item_id"],
                title=h["title"],
                price_jpy=h["price_jpy"],
                url=h["url"],
                thumbnail_url=None,
                stock_count=h.get("stock_count"),
                listing_kind="fixed_price",
            )
            for h in raw_hits[:max_results]
        ]

    def _resolve_single_code(
        self, query: str, source_options: Mapping[str, Any] | None
    ) -> str | None:
        """Resolve a query to one yuyutei sell code for the reference band — no
        multi-category fan-out (which would re-introduce the rate-limit risk)."""
        opts = dict(source_options or {})
        if "game_code" in opts:
            return resolve_sell_code(str(opts["game_code"])) or str(opts["game_code"]).strip().lower()
        return resolve_sell_code(query)

    def _fetch_band_side(self, url: str, *, require_in_stock: bool, timeout_ms: int) -> list[dict]:
        try:
            # Fail fast (no retry/curl) and rely on the shared per-host circuit
            # breaker — same rate-limit discipline as the sell search.
            html = self._http.get_text(
                url, timeout_seconds=timeout_ms / 1000.0, retries=1, curl_fallback=False,
                requester=self._requester, priority=self._priority,
            )
        except HostRateLimitedError as exc:
            logger.warning(
                "Yuyutei reference band: budget skip decision=%s reason=%s",
                exc.decision, exc.reason,
            )
            self.last_budget_skip = exc
            return []
        except Exception:
            logger.warning("Yuyutei reference band: HTTP failed url=%s", url)
            return []
        return _parse_card_listings(html, require_in_stock=require_in_stock)

    def reference_band(
        self,
        query: str,
        *,
        price_max: int,
        timeout_ms: int = 30_000,
        source_options: Mapping[str, Any] | None = None,
    ) -> YuyuteiReferenceBand | None:
        """Fetch the 買取 (buy) + in-stock 販売 (sell) reference band for a query.

        Costs TWO requests (one /buy/, one /sell/) for a *single* resolved game
        code — never a multi-category fan-out. Returns ``None`` when no TCG game
        can be identified (so non-TCG queries skip Yuyutei entirely) or when
        neither side yields data. Both fetches are fail-fast and protected by
        the shared per-host circuit breaker, so a 429 on the first short-circuits
        the second instead of amplifying the cooldown."""
        self.last_budget_skip = None
        if not query.strip() or price_max <= 0:
            return None
        code = self._resolve_single_code(query, source_options)
        if not code:
            logger.info("Yuyutei reference band: skipping query=%s (no identifiable TCG game)", query)
            return None

        params = urlencode({"search_word": query, "rare": "", "type": ""})
        sell_hits = self._fetch_band_side(
            f"{YUYUTEI_BASE_URL}/sell/{code}/s/search?{params}", require_in_stock=True, timeout_ms=timeout_ms,
        )
        buy_hits = self._fetch_band_side(
            f"{YUYUTEI_BASE_URL}/buy/{code}/s/search?{params}", require_in_stock=False, timeout_ms=timeout_ms,
        )

        sell = [h for h in sell_hits if 0 < h["price_jpy"] <= price_max]
        buy = [h for h in buy_hits if 0 < h["price_jpy"] <= price_max]
        if not sell and not buy:
            return None

        sample_urls = tuple(
            h["url"] for h in (*buy[:1], *sell[:1]) if h.get("url")
        )
        stock_total = sum(h["stock_count"] for h in sell if h.get("stock_count"))
        # Carry a few verbatim matched titles (sell/在庫あり first — the purchasable
        # side — then buy) so a caller can record the item's real identity. Deduped
        # and capped; this is already-parsed data, no extra request.
        seen_titles: set[str] = set()
        sample_titles: list[str] = []
        for h in (*sell, *buy):
            title = (h.get("title") or "").strip()
            if not title or title in seen_titles:
                continue
            seen_titles.add(title)
            sample_titles.append(title)
            if len(sample_titles) >= 8:
                break
        return YuyuteiReferenceBand(
            game_code=code,
            buy_prices=tuple(h["price_jpy"] for h in buy),
            sell_prices=tuple(h["price_jpy"] for h in sell),
            sell_stock_total=stock_total,
            sample_urls=sample_urls,
            sample_titles=tuple(sample_titles),
        )
