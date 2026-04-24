from __future__ import annotations

import json
import logging
import math
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Iterable
from urllib.parse import urlencode, urljoin

from bs4 import BeautifulSoup
from bs4.element import Tag

from market_monitor.http import HttpClient
from market_monitor.models import MarketOffer
from market_monitor.normalize import normalize_card_number, normalize_text

from .catalog import TcgCardSpec
from .matching import minimum_match_score, score_tcg_offer
from .yuyutei import YuyuteiClient

CARDRUSH_POKEMON_RANKING_URL = "https://www.cardrush-pokemon.jp/product-group/22?sort=rank&num=100"
MAGI_WS_RANKING_URL = "https://magi.camp/series/7/products"
MAGI_POKEMON_LIST_URL = "https://magi.camp/brands/3/items"
YUYUTEI_WS_TOP_URL = "https://yuyu-tei.jp/top/ws"
SNKRDUNK_POKEMON_MONTHLY_TRADES_URL = "https://snkrdunk.com/articles/31649/"
SNKRDUNK_POKEMON_UR_TRADES_URL = "https://snkrdunk.com/articles/31962/"
SNKRDUNK_POKEMON_SA_TRADES_URL = "https://snkrdunk.com/articles/31708/"
SNKRDUNK_WS_ANNUAL_SALES_URL = "https://snkrdunk.com/articles/26509/"
SNKRDUNK_WS_SUMMER_POCKETS_INITIAL_MARKET_URL = "https://snkrdunk.com/articles/31956/"
SNKRDUNK_WS_AOBUTA_INITIAL_MARKET_URL = "https://snkrdunk.com/articles/31830/"
YAHOO_REALTIME_SEARCH_URL = "https://search.yahoo.co.jp/realtime/search"
DEFAULT_BOARD_LIMIT = 10
SOCIAL_QUERY_CANDIDATE_LIMIT = 4
SOCIAL_CACHE_TTL_SECONDS = 15 * 60
BUY_SIGNAL_CANDIDATE_LIMIT = 24
BUY_SIGNAL_CACHE_TTL_SECONDS = 30 * 60

CARDRUSH_BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/135.0.0.0 Safari/537.36"
    ),
    "Accept": (
        "text/html,application/xhtml+xml,application/xml;q=0.9,"
        "image/avif,image/webp,image/apng,*/*;q=0.8"
    ),
    "Accept-Language": "ja-JP,ja;q=0.9,en-US;q=0.8,en;q=0.7",
    "Cache-Control": "max-age=0",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
    "Sec-CH-UA": '"Google Chrome";v="135", "Not-A.Brand";v="8", "Chromium";v="135"',
    "Sec-CH-UA-Mobile": "?0",
    "Sec-CH-UA-Platform": '"Windows"',
}
YAHOO_BROWSER_HEADERS = {
    "User-Agent": CARDRUSH_BROWSER_HEADERS["User-Agent"],
    "Accept": CARDRUSH_BROWSER_HEADERS["Accept"],
    "Accept-Language": CARDRUSH_BROWSER_HEADERS["Accept-Language"],
    "Cache-Control": "no-cache",
}

JPY_PRICE_RE = re.compile(r"(?P<price>\d[\d,]*)円")
MAGI_PRICE_RE = re.compile(r"¥\s*(?P<price>\d[\d,]*)")
COUNT_RE = re.compile(r"(?:在庫数|出品数)\s*(?P<count>\d+)")
GRADING_RE = re.compile(r"^(?:【|〖)(?P<label>(?:PSA|BGS|ARS|CGC)[^】〗]*)(?:】|〗)")
CARDRUSH_STATE_RE = re.compile(r"^〔(?P<label>[^〕]+)〕")
CARDRUSH_RARITY_RE = re.compile(r"【(?P<label>[^】]+)】")
CARDRUSH_SET_CODE_RE = re.compile(r"\[\s*(?:\[[^\]]+\]\s*)?(?P<code>[A-Za-z0-9]+)\s*\]")
WS_CODE_RE = re.compile(r"(?P<code>[A-Z0-9]+/[A-Z0-9-]+-[A-Z0-9]+)$")
POKEMON_CODE_RE = re.compile(r"\{(?P<code>[^}]+)\}")
POKEMON_INLINE_CODE_RE = re.compile(r"(?P<code>\d{1,3}/\d{1,3})$")
SOCIAL_COUNT_RE = re.compile(r"(?:reply|retweet|like|quote):(?P<count>\d+)")
SOCIAL_REPLY_RE = re.compile(r"reply:(?P<count>\d+)")
SOCIAL_RETWEET_RE = re.compile(r"retweet:(?P<count>\d+)")
SOCIAL_LIKE_RE = re.compile(r"like:(?P<count>\d+)")
SOCIAL_QUOTE_RE = re.compile(r"quote:(?P<count>\d+)")
SOCIAL_TWEET_SELECTOR = "div#sr div[class*='Tweet_TweetContainer']"
SNKRDUNK_CODE_BLOCK_RE = re.compile(r"\[(?P<code>[^\]]+)\]")
SNKRDUNK_PRICE_RE = re.compile(r"¥\s*(?P<price>\d[\d,]*)")
SNKRDUNK_HEADING_RANK_RE = re.compile(r"^(?:■\s*)?第?(?P<rank>\d+)位[:：]?\s*(?P<title>.+)$")

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class HotCardReference:
    label: str
    url: str


@dataclass(frozen=True, slots=True)
class HotCardSocialSignal:
    query: str
    search_url: str
    matched_post_count: int
    engagement_count: int
    score_ratio: float


@dataclass(frozen=True, slots=True)
class HotCardBuySignal:
    best_ask_jpy: int | None
    best_bid_jpy: int | None
    previous_bid_jpy: int | None
    ask_count: int
    bid_count: int
    bid_ask_ratio: float | None
    buy_support_ratio: float
    momentum_boost_ratio: float
    buy_signal_label: str | None
    references: tuple[HotCardReference, ...]


@dataclass(frozen=True, slots=True)
class HotCardEntry:
    game: str
    rank: int
    title: str
    price_jpy: int | None
    thumbnail_url: str | None
    card_number: str | None
    rarity: str | None
    set_code: str | None
    listing_count: int | None
    best_ask_jpy: int | None
    best_bid_jpy: int | None
    previous_bid_jpy: int | None
    bid_ask_ratio: float | None
    buy_support_score: float
    momentum_boost_score: float
    buy_signal_label: str | None
    hot_score: float
    attention_score: float
    social_post_count: int | None
    social_engagement_count: int | None
    notes: tuple[str, ...]
    is_graded: bool
    references: tuple[HotCardReference, ...]


@dataclass(frozen=True, slots=True)
class HotCardBoard:
    game: str
    label: str
    methodology: str
    generated_at: datetime
    items: tuple[HotCardEntry, ...]


@dataclass(frozen=True, slots=True)
class TcgLookupHint:
    game: str
    title: str
    card_number: str | None
    rarity: str | None
    set_code: str | None
    listing_count: int | None
    confidence: float
    references: tuple[HotCardReference, ...]


@dataclass
class _ParsedHotItem:
    title: str
    price_jpy: int | None
    thumbnail_url: str | None
    card_number: str | None
    rarity: str | None
    set_code: str | None
    listing_count: int | None
    is_graded: bool
    condition: str | None
    detail_url: str
    board_url: str
    note: str
    source_label: str = "Source"
    source_rank: int | None = None
    demand_ratio: float = 0.0
    grade_label: str | None = None


class TcgHotCardService:
    def __init__(self, http_client: HttpClient | None = None) -> None:
        self.http_client = http_client or HttpClient()
        self._yuyutei_client = YuyuteiClient(self.http_client)
        self._social_signal_cache: dict[str, tuple[float, HotCardSocialSignal | None]] = {}
        self._buy_signal_cache: dict[str, tuple[float, HotCardBuySignal | None]] = {}

    def load_boards(self, *, limit: int = DEFAULT_BOARD_LIMIT) -> tuple[HotCardBoard, ...]:
        with ThreadPoolExecutor(max_workers=2) as executor:
            pokemon_future = executor.submit(self.load_pokemon_board, limit=limit)
            ws_future = executor.submit(self.load_ws_board, limit=limit)
            return (pokemon_future.result(), ws_future.result())

    def load_pokemon_board(self, *, limit: int = DEFAULT_BOARD_LIMIT) -> HotCardBoard:
        parsed_items, methodology = self._load_pokemon_board_items()
        items = self._build_ranked_entries(
            game="pokemon",
            parsed_items=parsed_items,
            limit=limit,
        )
        return HotCardBoard(
            game="pokemon",
            label="Pokemon Liquidity Board",
            methodology=methodology,
            generated_at=datetime.now(timezone.utc),
            items=items,
        )

    def load_ws_board(self, *, limit: int = DEFAULT_BOARD_LIMIT) -> HotCardBoard:
        parsed_items, methodology = self._load_ws_board_items()
        items = self._build_ranked_entries(
            game="ws",
            parsed_items=parsed_items,
            limit=limit,
        )
        return HotCardBoard(
            game="ws",
            label="WS Liquidity Board",
            methodology=methodology,
            generated_at=datetime.now(timezone.utc),
            items=items,
        )

    def resolve_lookup_spec(self, spec: TcgCardSpec) -> TcgCardSpec | None:
        if spec.game not in {"pokemon", "ws"}:
            return None
        if spec.card_number:
            return None
        if not any((spec.rarity, spec.set_code, spec.set_name)):
            return None

        hints = self.search_lookup_hints(spec, limit=2)
        if not hints:
            return None

        best_hint = hints[0]
        if best_hint.confidence < 26.0:
            return None

        if len(hints) > 1 and hints[1].confidence >= best_hint.confidence - 6.0:
            return None

        aliases = list(spec.aliases)
        if best_hint.title != spec.title and best_hint.title not in aliases:
            aliases.append(best_hint.title)

        return replace(
            spec,
            title=best_hint.title,
            card_number=best_hint.card_number or spec.card_number,
            rarity=spec.rarity or best_hint.rarity,
            set_code=spec.set_code or best_hint.set_code,
            aliases=tuple(aliases),
        )

    def search_lookup_hints(self, spec: TcgCardSpec, *, limit: int = 5) -> tuple[TcgLookupHint, ...]:
        parsed_items = self._load_source_items(spec.game)
        ranked: list[tuple[float, _ParsedHotItem]] = []
        for item in parsed_items:
            confidence = self._hint_score(spec, item)
            if confidence < 18.0:
                continue
            ranked.append((confidence, item))

        ranked.sort(
            key=lambda value: (
                value[0],
                value[1].listing_count or 0,
                0 if not value[1].is_graded else -1,
                value[1].title,
            ),
            reverse=True,
        )

        return tuple(
            TcgLookupHint(
                game=spec.game,
                title=item.title,
                card_number=item.card_number,
                rarity=item.rarity,
                set_code=item.set_code,
                listing_count=item.listing_count,
                confidence=confidence,
                references=(
                    HotCardReference(label="Ranking Source", url=item.board_url),
                    HotCardReference(label="Item Page", url=item.detail_url),
                ),
            )
            for confidence, item in ranked[:limit]
        )

    def _build_ranked_entries(
        self,
        *,
        game: str,
        parsed_items: Iterable[_ParsedHotItem],
        limit: int,
    ) -> tuple[HotCardEntry, ...]:
        aggregates: dict[str, dict[str, object]] = {}
        for source_rank, item in enumerate(parsed_items, start=1):
            key = self._hot_item_key(game, item)
            entry = aggregates.get(key)
            if entry is None:
                aggregates[key] = {
                    "key": key,
                    "best_rank": source_rank,
                    "best_item": item,
                    "total_count": item.listing_count or 0,
                    "activity_ratio": item.demand_ratio,
                    "activity_sources": set() if item.demand_ratio <= 0 else {item.source_label},
                    "activity_best_ranks": (
                        {}
                        if item.demand_ratio <= 0 or item.source_rank is None
                        else {item.source_label: item.source_rank}
                    ),
                    "activity_references": (
                        []
                        if item.demand_ratio <= 0
                        else [HotCardReference(label=item.source_label, url=item.board_url)]
                    ),
                }
                continue

            entry["best_rank"] = min(int(entry["best_rank"]), source_rank)
            entry["total_count"] = int(entry["total_count"]) + (item.listing_count or 0)
            if item.demand_ratio > 0:
                activity_sources: set[str] = entry["activity_sources"]  # type: ignore[assignment]
                is_new_source = item.source_label not in activity_sources
                entry["activity_ratio"] = self._merge_activity_ratio(
                    existing=float(entry["activity_ratio"]),
                    incoming=item.demand_ratio,
                    is_new_source=is_new_source,
                )
                activity_sources.add(item.source_label)
                if item.source_rank is not None:
                    activity_best_ranks: dict[str, int] = entry["activity_best_ranks"]  # type: ignore[assignment]
                    current_rank = activity_best_ranks.get(item.source_label)
                    if current_rank is None or item.source_rank < current_rank:
                        activity_best_ranks[item.source_label] = item.source_rank
                reference = HotCardReference(label=item.source_label, url=item.board_url)
                activity_references: list[HotCardReference] = entry["activity_references"]  # type: ignore[assignment]
                if reference not in activity_references:
                    activity_references.append(reference)
            if self._prefer_item(item, entry["best_item"]):  # type: ignore[arg-type]
                entry["best_item"] = item

        ranked = sorted(
            aggregates.values(),
            key=lambda value: (
                -self._market_activity_score(
                    activity_ratio=float(value["activity_ratio"]),
                    source_count=len(value["activity_sources"]),  # type: ignore[arg-type]
                ),
                int(value["best_rank"]),
                1 if value["best_item"].is_graded else 0,  # type: ignore[attr-defined]
                normalize_text(value["best_item"].title),  # type: ignore[attr-defined]
            ),
        )
        candidate_limit = min(len(ranked), max(limit + 2, BUY_SIGNAL_CANDIDATE_LIMIT))
        ranked = ranked[:candidate_limit]

        def _fetch_buy_signal(aggregate: dict[str, object]) -> None:
            best_item: _ParsedHotItem = aggregate["best_item"]  # type: ignore[assignment]
            try:
                aggregate["buy_signal"] = self._lookup_buy_signal(
                    self._spec_from_hot_item(game=game, item=best_item),
                    reference_ask_jpy=best_item.price_jpy,
                )
            except Exception as exc:
                logger.warning("Buy signal parallel lookup failed title=%s error=%s", best_item.title, exc)
                aggregate["buy_signal"] = None

        with ThreadPoolExecutor(max_workers=10) as executor:
            list(executor.map(_fetch_buy_signal, ranked))

        ranked = [
            aggregate
            for aggregate in ranked
            if aggregate.get("buy_signal") is not None
        ]
        ranked.sort(
            key=lambda value: self._base_liquidity_sort_key(
                best_item=value["best_item"],  # type: ignore[arg-type]
                buy_signal=value.get("buy_signal"),  # type: ignore[arg-type]
                market_activity_ratio=float(value["activity_ratio"]),
                activity_source_count=len(value["activity_sources"]),  # type: ignore[arg-type]
            )
        )

        social_signals = self._load_social_signals(
            game=game,
            ranked=ranked,
            limit=limit,
        )
        ranked.sort(
            key=lambda value: self._final_liquidity_sort_key(
                best_item=value["best_item"],  # type: ignore[arg-type]
                best_rank=int(value["best_rank"]),
                buy_signal=value.get("buy_signal"),  # type: ignore[arg-type]
                market_activity_ratio=float(value["activity_ratio"]),
                activity_source_count=len(value["activity_sources"]),  # type: ignore[arg-type]
                social_signal=social_signals.get(str(value["key"])),
            )
        )

        items: list[HotCardEntry] = []
        for display_rank, aggregate in enumerate(ranked[:limit], start=1):
            best_item: _ParsedHotItem = aggregate["best_item"]  # type: ignore[assignment]
            best_rank = int(aggregate["best_rank"])
            total_count = int(aggregate["total_count"])
            market_activity_ratio = float(aggregate["activity_ratio"])
            activity_source_count = len(aggregate["activity_sources"])  # type: ignore[arg-type]
            activity_best_ranks: dict[str, int] = aggregate["activity_best_ranks"]  # type: ignore[assignment]
            activity_references: list[HotCardReference] = aggregate["activity_references"]  # type: ignore[assignment]
            buy_signal: HotCardBuySignal | None = aggregate.get("buy_signal")  # type: ignore[assignment]
            social_signal = social_signals.get(str(aggregate["key"]))
            liquidity_score = self._hot_score(
                buy_signal=buy_signal,
                is_graded=best_item.is_graded,
                market_activity_ratio=market_activity_ratio,
                activity_source_count=activity_source_count,
                social_signal=social_signal,
            )
            market_activity_score = self._market_activity_score(
                activity_ratio=market_activity_ratio,
                source_count=activity_source_count,
            )
            buy_support_score = round((buy_signal.buy_support_ratio if buy_signal is not None else 0.0) * 100.0, 2)
            momentum_boost_score = round((buy_signal.momentum_boost_ratio if buy_signal is not None else 0.0) * 100.0, 2)
            attention_score = self._attention_score(social_signal=social_signal)
            notes = [best_item.note]
            if activity_best_ranks:
                ranked_activity_sources = ", ".join(
                    f"{label} #{rank}"
                    for label, rank in sorted(activity_best_ranks.items(), key=lambda item: item[1])[:4]
                )
                notes.append(
                    f"Recent market activity signal: {ranked_activity_sources}."
                )
            else:
                notes.append(
                    "Recent market activity signal: no recent transaction-ranking evidence was found on the current external trend pages, so this entry is relying more heavily on buy-side support and SNS context."
                )
            if buy_signal is None or buy_signal.best_bid_jpy is None:
                notes.append(
                    "Primary liquidity signal: no credible buylist quote was found on 遊々亭, so this entry is treated as a low-confidence fallback."
                )
            elif buy_signal.best_ask_jpy is None or buy_signal.bid_ask_ratio is None:
                notes.append(
                    f"Primary liquidity signal: 遊々亭 buylist bid ¥{buy_signal.best_bid_jpy:,} is available."
                )
            else:
                notes.append(
                    "Primary liquidity signal: "
                    f"遊々亭 buylist bid ¥{buy_signal.best_bid_jpy:,} versus best ask ¥{buy_signal.best_ask_jpy:,} "
                    f"(bid/ask {buy_signal.bid_ask_ratio:.0%})."
                )
            if buy_signal is not None and buy_signal.buy_signal_label == "priceup" and buy_signal.previous_bid_jpy is not None:
                notes.append(
                    "Store-side buy pressure signal: "
                    f"遊々亭 marked this buy quote as raised from ¥{buy_signal.previous_bid_jpy:,} to ¥{buy_signal.best_bid_jpy:,}."
                )
            notes.append(
                f"Composite board score: recent market activity {market_activity_score:.2f}, buy-side support {buy_support_score:.2f}, SNS attention {attention_score:.2f}. Final weighting is market activity 50%, buy-side support 45%, SNS attention 5%, with a small raw-card fungibility bonus."
            )
            if total_count > 0:
                notes.append(
                    f"Source-page depth context only: {total_count} merged listing / stock unit(s) were observed."
                )
            notes.append(
                f"Candidate discovery context: source-page rank #{best_rank}; this is only used as a late tie-breaker."
            )
            references = [
                HotCardReference(label="Ranking Source", url=best_item.board_url),
                HotCardReference(label="Item Page", url=best_item.detail_url),
            ]
            references.extend(reference for reference in activity_references if reference not in references)
            if buy_signal is not None:
                references.extend(reference for reference in buy_signal.references if reference not in references)
            if social_signal is not None:
                notes.append(
                    "SNS attention side channel: "
                    f"{social_signal.matched_post_count} matched post(s), "
                    f"{social_signal.engagement_count} combined engagement via Yahoo!リアルタイム検索."
                )
                references.append(HotCardReference(label="SNS Search", url=social_signal.search_url))
            if best_item.is_graded:
                notes.append("Penalty applied: graded copies are treated as less fungible than raw copies.")

            items.append(
                HotCardEntry(
                    game=game,
                    rank=display_rank,
                    title=best_item.title,
                    price_jpy=best_item.price_jpy,
                    thumbnail_url=best_item.thumbnail_url,
                    card_number=best_item.card_number,
                    rarity=best_item.rarity,
                    set_code=best_item.set_code,
                    listing_count=total_count or None,
                    best_ask_jpy=None if buy_signal is None else buy_signal.best_ask_jpy,
                    best_bid_jpy=None if buy_signal is None else buy_signal.best_bid_jpy,
                    previous_bid_jpy=None if buy_signal is None else buy_signal.previous_bid_jpy,
                    bid_ask_ratio=None if buy_signal is None else buy_signal.bid_ask_ratio,
                    buy_support_score=buy_support_score,
                    momentum_boost_score=momentum_boost_score,
                    buy_signal_label=None if buy_signal is None else buy_signal.buy_signal_label,
                    hot_score=liquidity_score,
                    attention_score=attention_score,
                    social_post_count=None if social_signal is None else social_signal.matched_post_count,
                    social_engagement_count=None if social_signal is None else social_signal.engagement_count,
                    notes=tuple(notes),
                    is_graded=best_item.is_graded,
                    references=tuple(references),
                )
            )
        return tuple(items)

    def _load_source_items(self, game: str) -> list[_ParsedHotItem]:
        if game == "pokemon":
            parsed_items, _ = self._load_pokemon_board_items()
            return parsed_items
        if game == "ws":
            parsed_items, _ = self._load_ws_board_items()
            return parsed_items
        return []

    def _load_pokemon_board_items(self) -> tuple[list[_ParsedHotItem], str]:
        parsed_items: list[_ParsedHotItem] = []
        methodology_parts = [
            "Pokemon 榜單現在先整合近期交易排名、類別交易排名與店家頁候選，再做綜合排序。",
            " 單一店家的在庫 / 出品順序不再主導熱門判斷；近期實際交易活躍度會先決定市場熱度骨架。",
            " 遊々亭的 bid / ask 與 priceup 仍保留，因為它們能補足買方承接力。",
            " Yahoo!リアルタイム検索 只做 SNS 注意力輔助，不會單獨把卡推上前列。",
        ]

        pokemon_trend_sources = (
            (
                SNKRDUNK_POKEMON_MONTHLY_TRADES_URL,
                "SNKRDUNK monthly trades",
                50,
                1.0,
                "Signal source: SNKRDUNK recent monthly transaction ranking.",
            ),
            (
                SNKRDUNK_POKEMON_UR_TRADES_URL,
                "SNKRDUNK UR trades",
                30,
                0.78,
                "Signal source: SNKRDUNK UR-category transaction ranking.",
            ),
            (
                SNKRDUNK_POKEMON_SA_TRADES_URL,
                "SNKRDUNK SA trades",
                50,
                0.72,
                "Signal source: SNKRDUNK SA-category transaction ranking.",
            ),
        )

        def _fetch_snkrdunk(args: tuple[str, str, int, float, str]) -> list[_ParsedHotItem]:
            board_url, source_label, max_rank, source_weight, note = args
            try:
                html = self.http_client.get_text(board_url)
                return self._parse_snkrdunk_ranking_items(
                    html,
                    board_url=board_url,
                    source_label=source_label,
                    max_rank=max_rank,
                    source_weight=source_weight,
                    note=note,
                )
            except Exception as exc:  # pragma: no cover - network-dependent.
                logger.warning("Pokemon trend source failed url=%s error=%s", board_url, exc)
                return []

        def _fetch_cardrush() -> list[_ParsedHotItem]:
            try:
                html = self.http_client.get_text(
                    CARDRUSH_POKEMON_RANKING_URL,
                    headers=CARDRUSH_BROWSER_HEADERS,
                )
                return self._parse_cardrush_pokemon_items(html)
            except Exception as exc:  # pragma: no cover - network-dependent.
                logger.warning("Cardrush Pokemon ranking failed; continuing without it. error=%s", exc)
                return []

        def _fetch_magi_pokemon() -> list[_ParsedHotItem]:
            try:
                html = self.http_client.get_text(MAGI_POKEMON_LIST_URL)
                return self._parse_magi_pokemon_items(html)
            except Exception as exc:  # pragma: no cover - network-dependent.
                logger.warning("Magi Pokemon listing page failed; continuing without it. error=%s", exc)
                return []

        with ThreadPoolExecutor(max_workers=5) as executor:
            snkrdunk_futures = [executor.submit(_fetch_snkrdunk, args) for args in pokemon_trend_sources]
            cardrush_future = executor.submit(_fetch_cardrush)
            magi_future = executor.submit(_fetch_magi_pokemon)
            for future in snkrdunk_futures:
                parsed_items.extend(future.result())
            parsed_items.extend(cardrush_future.result())
            parsed_items.extend(magi_future.result())

        return (parsed_items, "".join(methodology_parts))

    def _load_ws_board_items_legacy(self) -> tuple[list[_ParsedHotItem], str]:
        parsed_items: list[_ParsedHotItem] = []
        methodology_parts = [
            "WS 現在會先把外部市場活動頁面的熱度信號拉進來，再和遊々亭買盤與 Yahoo!リアルタイム検索 一起做綜合排序。",
            " 除了 magi 頁面順序之外，WS 目前也會吃 SNKRDUNK 的年度賣上排行與近期初動相場文章，讓市場活動層不再只靠單一頁面候選。",
            " 這讓 WS 比原本更接近 Pokemon 現在的多來源 activity model，但仍保留買盤與 SNS 只作輔助驗證的保守框架。",
        ]

        ws_trend_sources = (
            (
                SNKRDUNK_WS_ANNUAL_SALES_URL,
                "SNKRDUNK annual sales ranking",
                0.54,
                self._parse_snkrdunk_heading_ranking_items,
                "Signal source: SNKRDUNK annual Weiss Schwarz sales ranking.",
            ),
            (
                SNKRDUNK_WS_SUMMER_POCKETS_INITIAL_MARKET_URL,
                "SNKRDUNK Summer Pockets initial market",
                0.48,
                self._parse_snkrdunk_article_apparel_items,
                "Signal source: SNKRDUNK initial-market article for Summer Pockets.",
            ),
            (
                SNKRDUNK_WS_AOBUTA_INITIAL_MARKET_URL,
                "SNKRDUNK Aobuta initial market",
                0.48,
                self._parse_snkrdunk_article_apparel_items,
                "Signal source: SNKRDUNK initial-market article for Rascal Does Not Dream of Santa Claus.",
            ),
        )

        def _fetch_ws_source(args: tuple[str, str, float, object, str]) -> list[_ParsedHotItem]:
            board_url, source_label, source_weight, parser, note = args
            try:
                html = self.http_client.get_text(board_url)
                return parser(  # type: ignore[operator]
                    html,
                    board_url=board_url,
                    source_label=source_label,
                    source_weight=source_weight,
                    note=note,
                )
            except Exception as exc:  # pragma: no cover - network-dependent.
                logger.warning("WS trend source failed url=%s error=%s", board_url, exc)
                return []

        def _fetch_magi_ws() -> list[_ParsedHotItem]:
            try:
                html = self.http_client.get_text(MAGI_WS_RANKING_URL)
                return self._parse_magi_ws_items(html)
            except Exception as exc:  # pragma: no cover - network-dependent.
                logger.warning("Magi WS listing page failed; continuing without it. error=%s", exc)
                return []

        with ThreadPoolExecutor(max_workers=4) as executor:
            ws_futures = [executor.submit(_fetch_ws_source, args) for args in ws_trend_sources]
            magi_future = executor.submit(_fetch_magi_ws)
            for future in ws_futures:
                parsed_items.extend(future.result())
            parsed_items.extend(magi_future.result())

        return (parsed_items, "".join(methodology_parts))

    def _load_ws_board_items(self) -> tuple[list[_ParsedHotItem], str]:
        parsed_items: list[_ParsedHotItem] = []
        methodology_parts = [
            "WS liquidity board now prioritizes live marketplace/store surfaces: Yuyutei top-page featured singles, Yuyutei latest-release spotlight, and Magi's current Weiss Schwarz listing order.",
            " Static SNKRDUNK article rankings are no longer used as primary WS market-activity inputs, because year-old article pages can distort what is actually active today.",
            " Yuyutei bid / ask and priceup remain in the score because they are still the best buy-side support signal, while Yahoo realtime remains only a light secondary attention check.",
        ]

        def _fetch_yuyutei_ws_top() -> list[_ParsedHotItem]:
            try:
                html = self.http_client.get_text(YUYUTEI_WS_TOP_URL)
            except Exception as exc:  # pragma: no cover - network-dependent.
                logger.warning("Yuyutei WS top page failed; continuing without it. error=%s", exc)
                return []

            return [
                *self._parse_yuyutei_ws_carousel_items(
                    html,
                    board_url=YUYUTEI_WS_TOP_URL,
                    carousel_id="recommendedItemList",
                    source_label="Yuyutei featured singles",
                    source_weight=0.34,
                    note="Signal source: Yuyutei top-page featured Weiss Schwarz singles.",
                    minimum_price_jpy=5000,
                ),
                *self._parse_yuyutei_ws_carousel_items(
                    html,
                    board_url=YUYUTEI_WS_TOP_URL,
                    carousel_id="newestCardList",
                    source_label="Yuyutei latest-release spotlight",
                    source_weight=0.12,
                    note="Signal source: Yuyutei latest-release Weiss Schwarz spotlight.",
                    minimum_price_jpy=1000,
                ),
            ]

        def _fetch_magi_ws() -> list[_ParsedHotItem]:
            try:
                html = self.http_client.get_text(MAGI_WS_RANKING_URL)
                return self._parse_magi_ws_items(html)
            except Exception as exc:  # pragma: no cover - network-dependent.
                logger.warning("Magi WS listing page failed; continuing without it. error=%s", exc)
                return []

        with ThreadPoolExecutor(max_workers=2) as executor:
            yuyutei_future = executor.submit(_fetch_yuyutei_ws_top)
            magi_future = executor.submit(_fetch_magi_ws)
            parsed_items.extend(yuyutei_future.result())
            parsed_items.extend(magi_future.result())

        return (parsed_items, "".join(methodology_parts))

    def _load_social_signals(
        self,
        *,
        game: str,
        ranked: list[dict[str, object]],
        limit: int,
    ) -> dict[str, HotCardSocialSignal]:
        if not ranked:
            return {}

        candidate_limit = min(len(ranked), max(min(limit, 4), SOCIAL_QUERY_CANDIDATE_LIMIT))
        candidates = ranked[:candidate_limit]
        signals: dict[str, HotCardSocialSignal] = {}

        def _fetch_social(aggregate: dict[str, object]) -> tuple[str, HotCardSocialSignal | None]:
            best_item: _ParsedHotItem = aggregate["best_item"]  # type: ignore[assignment]
            return str(aggregate["key"]), self._lookup_social_signal(game=game, item=best_item)

        with ThreadPoolExecutor(max_workers=4) as executor:
            for key, signal in executor.map(_fetch_social, candidates):
                if signal is not None:
                    signals[key] = signal
        return signals

    def _lookup_social_signal(self, *, game: str, item: _ParsedHotItem) -> HotCardSocialSignal | None:
        query = _build_social_query(game=game, item=item)
        if not query:
            return None

        now = time.time()
        cached = self._social_signal_cache.get(query)
        if cached is not None and now - cached[0] < SOCIAL_CACHE_TTL_SECONDS:
            return cached[1]

        search_url = f"{YAHOO_REALTIME_SEARCH_URL}?{urlencode({'p': query})}"
        try:
            html = self.http_client.get_text(search_url, headers=YAHOO_BROWSER_HEADERS)
            signal = _parse_yahoo_realtime_signal(html=html, query=query, search_url=search_url)
        except Exception as exc:  # pragma: no cover - network-dependent.
            logger.warning("Social signal lookup failed query=%s error=%s", query, exc)
            signal = None

        self._social_signal_cache[query] = (now, signal)
        return signal

    def _lookup_buy_signal(self, spec: TcgCardSpec, *, reference_ask_jpy: int | None = None) -> HotCardBuySignal | None:
        cache_key = "|".join(
            [
                spec.game,
                spec.title,
                spec.card_number or "",
                spec.rarity or "",
                spec.set_code or "",
                str(reference_ask_jpy or ""),
            ]
        )
        now = time.time()
        cached = self._buy_signal_cache.get(cache_key)
        if cached is not None and now - cached[0] < BUY_SIGNAL_CACHE_TTL_SECONDS:
            return cached[1]

        minimum_score = minimum_match_score(spec)
        exact_search_word = spec.card_number or spec.title
        exact_matches = self._match_buy_signal_candidates(
            spec=spec,
            search_word=exact_search_word,
            minimum_score=minimum_score,
        )
        if exact_matches:
            signal = self._build_buy_signal(exact_matches, reference_ask_jpy=reference_ask_jpy)
        else:
            fallback_matches = self._yuyutei_client.lookup(spec, minimum_score=minimum_score)
            signal = self._build_buy_signal(fallback_matches or exact_matches, reference_ask_jpy=reference_ask_jpy)

        self._buy_signal_cache[cache_key] = (now, signal)
        return signal

    def _match_buy_signal_candidates(
        self,
        *,
        spec: TcgCardSpec,
        search_word: str,
        minimum_score: float,
    ) -> list[MarketOffer]:
        offers = self._yuyutei_client.search_buy(spec, search_word=search_word)
        matched: list[MarketOffer] = []
        for offer in offers:
            score = score_tcg_offer(spec, offer)
            if score >= minimum_score:
                matched.append(replace(offer, score=score))
        return matched

    @staticmethod
    def _build_buy_signal(
        offers: Iterable[MarketOffer],
        *,
        reference_ask_jpy: int | None = None,
    ) -> HotCardBuySignal | None:
        ask_offers = sorted(
            (offer for offer in offers if offer.price_kind == "ask"),
            key=lambda offer: offer.price_jpy,
        )
        bid_offers = sorted(
            (offer for offer in offers if offer.price_kind == "bid"),
            key=lambda offer: offer.price_jpy,
            reverse=True,
        )
        best_ask = ask_offers[0] if ask_offers else None
        best_bid = bid_offers[0] if bid_offers else None
        effective_ask_jpy = reference_ask_jpy if reference_ask_jpy is not None else None if best_ask is None else best_ask.price_jpy
        if effective_ask_jpy is None and best_bid is None:
            return None

        previous_bid = None
        buy_signal_label = None
        momentum_boost_ratio = 0.0
        if best_bid is not None:
            previous_bid = _parse_optional_int(best_bid.attributes.get("compare_price_jpy"))
            buy_signal_label = _buy_signal_label(best_bid)
            momentum_boost_ratio = TcgHotCardService._buy_momentum_boost(
                current_bid=best_bid.price_jpy,
                previous_bid=previous_bid,
                signal_label=buy_signal_label,
            )

        bid_ask_ratio = None
        if effective_ask_jpy is not None and best_bid is not None and effective_ask_jpy > 0:
            bid_ask_ratio = round(min(1.0, best_bid.price_jpy / effective_ask_jpy), 4)

        buy_support_ratio = 0.0
        if best_bid is not None:
            buy_support_ratio += 0.35
        if bid_ask_ratio is not None:
            buy_support_ratio += bid_ask_ratio * 0.50
        if best_bid is not None and effective_ask_jpy is not None:
            buy_support_ratio += 0.15
        buy_support_ratio += momentum_boost_ratio
        buy_support_ratio = round(min(1.0, buy_support_ratio), 4)

        references: list[HotCardReference] = []
        if best_bid is not None:
            references.append(HotCardReference(label="Yuyutei Buylist", url=best_bid.url))
        if best_ask is not None:
            references.append(HotCardReference(label="Yuyutei Ask", url=best_ask.url))

        return HotCardBuySignal(
            best_ask_jpy=effective_ask_jpy,
            best_bid_jpy=None if best_bid is None else best_bid.price_jpy,
            previous_bid_jpy=previous_bid,
            ask_count=len(ask_offers),
            bid_count=len(bid_offers),
            bid_ask_ratio=bid_ask_ratio,
            buy_support_ratio=buy_support_ratio,
            momentum_boost_ratio=momentum_boost_ratio,
            buy_signal_label=buy_signal_label,
            references=tuple(references),
        )

    @staticmethod
    def _spec_from_hot_item(*, game: str, item: _ParsedHotItem) -> TcgCardSpec:
        return TcgCardSpec(
            game=game,
            title=item.title,
            card_number=item.card_number,
            rarity=item.rarity,
            set_code=item.set_code,
        )

    @staticmethod
    def _prefer_item(candidate: _ParsedHotItem, current: _ParsedHotItem) -> bool:
        candidate_priority = _condition_priority(candidate.condition, candidate.is_graded)
        current_priority = _condition_priority(current.condition, current.is_graded)
        if candidate_priority != current_priority:
            return candidate_priority > current_priority
        if candidate.price_jpy is None:
            return False
        if current.price_jpy is None:
            return True
        return candidate.price_jpy < current.price_jpy

    @staticmethod
    def _base_liquidity_sort_key(
        *,
        best_item: _ParsedHotItem,
        buy_signal: HotCardBuySignal | None,
        market_activity_ratio: float,
        activity_source_count: int,
    ) -> tuple[object, ...]:
        market_activity_score = TcgHotCardService._market_activity_score(
            activity_ratio=market_activity_ratio,
            source_count=activity_source_count,
        )
        buy_support_score = 0.0 if buy_signal is None else buy_signal.buy_support_ratio
        preliminary_score = (market_activity_score * 0.6) + (buy_support_score * 40.0)
        if not best_item.is_graded:
            preliminary_score += 5.0
        return (
            -round(preliminary_score, 2),
            -market_activity_score,
            -buy_support_score,
            1 if best_item.is_graded else 0,
            normalize_text(best_item.title),
        )

    @staticmethod
    def _final_liquidity_sort_key(
        *,
        best_item: _ParsedHotItem,
        best_rank: int,
        buy_signal: HotCardBuySignal | None,
        market_activity_ratio: float,
        activity_source_count: int,
        social_signal: HotCardSocialSignal | None,
    ) -> tuple[object, ...]:
        liquidity_score = TcgHotCardService._hot_score(
            buy_signal=buy_signal,
            is_graded=best_item.is_graded,
            market_activity_ratio=market_activity_ratio,
            activity_source_count=activity_source_count,
            social_signal=social_signal,
        )
        market_activity_score = TcgHotCardService._market_activity_score(
            activity_ratio=market_activity_ratio,
            source_count=activity_source_count,
        )
        buy_support_score = 0.0 if buy_signal is None else buy_signal.buy_support_ratio
        attention_score = TcgHotCardService._attention_score(social_signal=social_signal)
        return (
            -liquidity_score,
            -market_activity_score,
            -buy_support_score,
            -attention_score,
            1 if best_item.is_graded else 0,
            best_rank,
            normalize_text(best_item.title),
        )

    @staticmethod
    def _hot_score(
        *,
        buy_signal: HotCardBuySignal | None,
        is_graded: bool,
        market_activity_ratio: float,
        activity_source_count: int,
        social_signal: HotCardSocialSignal | None,
    ) -> float:
        market_activity_score = TcgHotCardService._market_activity_score(
            activity_ratio=market_activity_ratio,
            source_count=activity_source_count,
        )
        buy_support_score = 0.0 if buy_signal is None else buy_signal.buy_support_ratio * 100.0
        attention_score = TcgHotCardService._attention_score(social_signal=social_signal)
        fungibility_bonus = 0.0 if is_graded else 5.0
        score = (
            (market_activity_score * 0.50)
            + (buy_support_score * 0.45)
            + (attention_score * 0.05)
            + fungibility_bonus
        )
        return round(score, 2)

    @staticmethod
    def _market_activity_score(*, activity_ratio: float, source_count: int) -> float:
        clamped_ratio = min(1.0, max(0.0, activity_ratio))
        diversity_bonus = min(10.0, max(0, source_count - 1) * 3.5)
        return round(min(100.0, (clamped_ratio * 100.0) + diversity_bonus), 2)

    @staticmethod
    def _merge_activity_ratio(*, existing: float, incoming: float, is_new_source: bool) -> float:
        if incoming <= 0:
            return round(max(0.0, existing), 4)
        if existing <= 0:
            return round(min(1.0, incoming), 4)
        multiplier = 0.45 if is_new_source else 0.20
        return round(min(1.0, existing + (incoming * multiplier)), 4)

    @staticmethod
    def _buy_momentum_boost(*, current_bid: int, previous_bid: int | None, signal_label: str | None) -> float:
        if signal_label != "priceup":
            return 0.0
        if previous_bid is None or previous_bid <= 0 or current_bid <= previous_bid:
            return 0.03
        increase_ratio = max(0.0, (current_bid - previous_bid) / previous_bid)
        return round(min(0.06, 0.03 + (increase_ratio * 0.03)), 4)

    @staticmethod
    def _attention_score(
        *,
        social_signal: HotCardSocialSignal | None,
    ) -> float:
        social_ratio = 0.0 if social_signal is None else social_signal.score_ratio
        return round(social_ratio * 100.0, 2)

    @staticmethod
    def _hint_score(spec: TcgCardSpec, item: _ParsedHotItem) -> float:
        query_title = _title_key(spec.game, spec.title)
        item_title = _title_key(spec.game, item.title)
        base_query_title = _title_key(spec.game, spec.title, drop_game_suffixes=True)
        base_item_title = _title_key(spec.game, item.title, drop_game_suffixes=True)

        score = 0.0
        if query_title == item_title:
            score += 34.0
        elif query_title and (query_title in item_title or item_title in query_title):
            score += 24.0

        if base_query_title == base_item_title:
            score += 18.0
        elif base_query_title and (base_query_title in base_item_title or base_item_title in base_query_title):
            score += 10.0

        if spec.card_number and item.card_number:
            if normalize_card_number(spec.card_number) == normalize_card_number(item.card_number):
                score += 40.0
            else:
                score -= 20.0

        if spec.rarity and item.rarity:
            if normalize_text(spec.rarity) == normalize_text(item.rarity):
                score += 16.0
            else:
                score -= 6.0

        if spec.set_code and item.set_code:
            if normalize_text(spec.set_code) == normalize_text(item.set_code):
                score += 10.0
            else:
                score -= 4.0

        score += min(item.listing_count or 0, 250) * 0.03
        if item.is_graded:
            score -= 4.0
        return score

    @staticmethod
    def _hot_item_key(game: str, item: _ParsedHotItem) -> str:
        card_number = normalize_card_number(item.card_number or "")
        rarity = normalize_text(item.rarity or "")
        title = normalize_text(item.title)
        return "|".join([game, title, card_number, rarity])

    def _parse_cardrush_pokemon_items(self, html: str) -> list[_ParsedHotItem]:
        soup = BeautifulSoup(html, "html.parser")
        items: list[_ParsedHotItem] = []
        for source_rank, anchor in enumerate(soup.select("ul.item_list li.list_item_cell div.item_data a[href]"), start=1):
            text = " ".join(anchor.get_text(" ", strip=True).split())
            if not text:
                continue
            parsed = _parse_cardrush_text(
                text,
                detail_url=urljoin("https://www.cardrush-pokemon.jp", anchor["href"]),
                board_url=CARDRUSH_POKEMON_RANKING_URL,
                thumbnail_url=_extract_cardrush_thumbnail_url(anchor),
                source_rank=source_rank,
                source_label="Cardrush category rank",
                demand_ratio=_rank_signal_ratio(rank=source_rank, max_rank=100, source_weight=0.22, floor=0.08),
            )
            if parsed is not None:
                items.append(parsed)
        return items

    def _parse_snkrdunk_article_apparel_items(
        self,
        html: str,
        *,
        board_url: str,
        source_label: str,
        source_weight: float,
        note: str,
    ) -> list[_ParsedHotItem]:
        soup = BeautifulSoup(html, "html.parser")
        article_content = soup.select_one("article-content")
        if article_content is None:
            return []

        raw_apparels = article_content.get(":apparels")
        if not isinstance(raw_apparels, str) or not raw_apparels.strip():
            return []

        try:
            apparel_items = json.loads(raw_apparels)
        except json.JSONDecodeError as exc:
            logger.warning("Failed to decode SNKRDUNK article apparel payload url=%s error=%s", board_url, exc)
            return []

        parsed_candidates: list[_ParsedHotItem] = []
        for apparel in apparel_items:
            if not isinstance(apparel, dict):
                continue

            apparel_id = _parse_optional_int(apparel.get("id"))
            title_text = _first_populated_text(apparel.get("localizedName"), apparel.get("name"))
            if apparel_id is None or title_text is None:
                continue

            parsed = _parse_snkrdunk_text(
                title_text,
                detail_url=f"https://snkrdunk.com/apparels/{apparel_id}?slide=right",
                board_url=board_url,
                thumbnail_url=_extract_snkrdunk_primary_image_url(apparel),
                note=note,
                source_label=source_label,
                source_rank=1,
                demand_ratio=0.0,
            )
            if parsed is None or parsed.card_number is None:
                continue

            parsed_candidates.append(
                replace(
                    parsed,
                    price_jpy=_parse_jpy_price_text(apparel.get("displayPrice")),
                    listing_count=_parse_optional_int(apparel.get("totalListingCount")),
                )
            )

        max_rank = len(parsed_candidates)
        if max_rank <= 0:
            return []

        return [
            replace(
                parsed,
                source_rank=source_rank,
                demand_ratio=_rank_signal_ratio(
                    rank=source_rank,
                    max_rank=max_rank,
                    source_weight=source_weight,
                    floor=0.18,
                ),
            )
            for source_rank, parsed in enumerate(parsed_candidates, start=1)
        ]

    def _parse_snkrdunk_heading_ranking_items(
        self,
        html: str,
        *,
        board_url: str,
        source_label: str,
        source_weight: float,
        note: str,
    ) -> list[_ParsedHotItem]:
        soup = BeautifulSoup(html, "html.parser")
        ranked_candidates: list[tuple[int, _ParsedHotItem]] = []
        for heading in soup.find_all("h3"):
            heading_text = " ".join(heading.get_text(" ", strip=True).split())
            ranked_heading = _parse_snkrdunk_heading_rank(heading_text)
            if ranked_heading is None:
                continue

            source_rank, title_text = ranked_heading
            anchor = _find_next_snkrdunk_apparel_anchor(heading)
            if anchor is None:
                continue

            href = anchor.get("href")
            if not isinstance(href, str) or not href:
                continue

            image = anchor.select_one("img")
            price_text = " ".join(anchor.get_text(" ", strip=True).split())
            parsed = _parse_snkrdunk_text(
                title_text,
                detail_url=urljoin("https://snkrdunk.com", href),
                board_url=board_url,
                thumbnail_url=_pick_image_url(image, attribute_names=("src", "data-src")),
                note=note,
                source_label=source_label,
                source_rank=source_rank,
                demand_ratio=0.0,
            )
            if parsed is None or parsed.card_number is None:
                continue

            ranked_candidates.append(
                (
                    source_rank,
                    replace(parsed, price_jpy=_parse_jpy_price_text(price_text)),
                )
            )

        if not ranked_candidates:
            return []

        max_rank = max(source_rank for source_rank, _ in ranked_candidates)
        return [
            replace(
                parsed,
                source_rank=source_rank,
                demand_ratio=_rank_signal_ratio(
                    rank=source_rank,
                    max_rank=max_rank,
                    source_weight=source_weight,
                    floor=0.18,
                ),
            )
            for source_rank, parsed in ranked_candidates
        ]

    def _parse_magi_ws_items(self, html: str) -> list[_ParsedHotItem]:
        soup = BeautifulSoup(html, "html.parser")
        items: list[_ParsedHotItem] = []
        for source_rank, anchor in enumerate(soup.select("div.product-list__box a[href^='/products/']"), start=1):
            text = " ".join(anchor.get_text(" ", strip=True).split())
            if not text:
                continue
            parsed = _parse_magi_text(
                text,
                detail_url=urljoin("https://magi.camp", anchor["href"]),
                board_url=MAGI_WS_RANKING_URL,
                thumbnail_url=_extract_magi_thumbnail_url(anchor),
                note="Signal source: Magi popular/recommended Weiss Schwarz page order.",
                source_rank=source_rank,
                source_label="Magi recommendation rank",
                demand_ratio=_rank_signal_ratio(rank=source_rank, max_rank=100, source_weight=0.18, floor=0.06),
            )
            if parsed is not None:
                items.append(parsed)
        return items

    def _parse_yuyutei_ws_carousel_items(
        self,
        html: str,
        *,
        board_url: str,
        carousel_id: str,
        source_label: str,
        source_weight: float,
        note: str,
        minimum_price_jpy: int = 0,
    ) -> list[_ParsedHotItem]:
        soup = BeautifulSoup(html, "html.parser")
        carousel = soup.select_one(f"#{carousel_id}")
        if carousel is None:
            return []

        parsed_candidates: list[_ParsedHotItem] = []
        seen_urls: set[str] = set()
        for card_box in carousel.select("div.col-md-4"):
            anchors = [
                anchor
                for anchor in card_box.select("a[href*='/sell/ws/card/']")
                if anchor.select_one("img") is not None
            ]
            if not anchors:
                continue

            anchor = anchors[0]
            href = anchor.get("href")
            if not isinstance(href, str) or not href:
                continue

            detail_url = urljoin("https://yuyu-tei.jp", href)
            if detail_url in seen_urls:
                continue

            image = anchor.select_one("img")
            if image is None:
                continue

            image_alt = " ".join(str(image.get("alt", "")).split())
            alt_parts = image_alt.split()
            if len(alt_parts) < 3:
                continue

            card_number = alt_parts[0].strip()
            rarity = alt_parts[1].strip()
            title = " ".join(alt_parts[2:]).strip()
            price_element = card_box.select_one("strong")
            price_jpy = _parse_jpy_price_text(price_element.get_text(" ", strip=True) if price_element is not None else None)
            if price_jpy is None or price_jpy < minimum_price_jpy:
                continue

            parsed_candidates.append(
                _ParsedHotItem(
                    title=title,
                    price_jpy=price_jpy,
                    thumbnail_url=_pick_image_url(image, attribute_names=("src", "data-src")),
                    card_number=card_number,
                    rarity=rarity,
                    set_code=card_number.split("/", 1)[0].lower() if "/" in card_number else None,
                    listing_count=1,
                    is_graded=False,
                    condition=None,
                    detail_url=detail_url,
                    board_url=board_url,
                    note=note,
                    source_label=source_label,
                )
            )
            seen_urls.add(detail_url)

        max_rank = len(parsed_candidates)
        if max_rank <= 0:
            return []

        return [
            replace(
                parsed,
                source_rank=source_rank,
                demand_ratio=_rank_signal_ratio(
                    rank=source_rank,
                    max_rank=max_rank,
                    source_weight=source_weight,
                    floor=0.10,
                ),
            )
            for source_rank, parsed in enumerate(parsed_candidates, start=1)
        ]

    def _parse_magi_pokemon_items(self, html: str) -> list[_ParsedHotItem]:
        soup = BeautifulSoup(html, "html.parser")
        items: list[_ParsedHotItem] = []
        for source_rank, anchor in enumerate(soup.select("a[href^='/items/'], a[href^='/products/']"), start=1):
            text = " ".join(anchor.get_text(" ", strip=True).split())
            if not text:
                continue
            href = anchor.get("href")
            if not isinstance(href, str) or not href:
                continue
            parsed = _parse_magi_text(
                text,
                detail_url=urljoin("https://magi.camp", href),
                board_url=MAGI_POKEMON_LIST_URL,
                thumbnail_url=_extract_magi_thumbnail_url(anchor),
                note="Signal source: Magi Pokemon card listing page order.",
                source_rank=source_rank,
                source_label="Magi listing order",
                demand_ratio=_rank_signal_ratio(rank=source_rank, max_rank=100, source_weight=0.16, floor=0.06),
            )
            if parsed is not None:
                items.append(parsed)
        return items

    def _parse_snkrdunk_ranking_items(
        self,
        html: str,
        *,
        board_url: str,
        source_label: str,
        max_rank: int,
        source_weight: float,
        note: str,
    ) -> list[_ParsedHotItem]:
        soup = BeautifulSoup(html, "html.parser")
        items: list[_ParsedHotItem] = []
        seen_urls: set[str] = set()
        for anchor in soup.select("a[href*='/apparels/']"):
            href = anchor.get("href")
            if not isinstance(href, str) or not href:
                continue
            detail_url = urljoin("https://snkrdunk.com", href)
            if detail_url in seen_urls:
                continue

            rank_tag = anchor.select_one("span")
            image = anchor.select_one("img[alt]")
            if rank_tag is None or image is None:
                continue

            rank_text = rank_tag.get_text(strip=True)
            if not rank_text.isdigit():
                continue
            source_rank = int(rank_text)
            if source_rank <= 0 or source_rank > max_rank:
                continue

            title_text = image.get("alt")
            if not isinstance(title_text, str) or not title_text.strip():
                continue

            parsed = _parse_snkrdunk_text(
                title_text,
                detail_url=detail_url,
                board_url=board_url,
                thumbnail_url=_pick_image_url(image, attribute_names=("src", "data-src")),
                note=note,
                source_label=source_label,
                source_rank=source_rank,
                demand_ratio=_rank_signal_ratio(
                    rank=source_rank,
                    max_rank=max_rank,
                    source_weight=source_weight,
                    floor=0.18,
                ),
            )
            if parsed is None:
                continue

            price_match = SNKRDUNK_PRICE_RE.search(anchor.get_text(" ", strip=True))
            if price_match is not None:
                parsed = replace(parsed, price_jpy=int(price_match.group("price").replace(",", "")))

            items.append(parsed)
            seen_urls.add(detail_url)
        return items


def _parse_cardrush_text(
    text: str,
    *,
    detail_url: str,
    board_url: str,
    thumbnail_url: str | None = None,
    source_rank: int | None = None,
    source_label: str = "Cardrush category rank",
    demand_ratio: float = 0.0,
) -> _ParsedHotItem | None:
    condition = None
    working = text
    condition_match = CARDRUSH_STATE_RE.match(working)
    if condition_match is not None:
        condition = condition_match.group("label")
        working = working[condition_match.end():].strip()

    rarity_match = CARDRUSH_RARITY_RE.search(working)
    rarity = rarity_match.group("label") if rarity_match is not None else None
    title = working[: rarity_match.start()].strip() if rarity_match is not None else working

    card_number_match = POKEMON_CODE_RE.search(working)
    card_number = card_number_match.group("code").strip() if card_number_match is not None else None
    set_code_match = CARDRUSH_SET_CODE_RE.search(working)
    set_code = set_code_match.group("code").lower() if set_code_match is not None else None

    price_match = JPY_PRICE_RE.search(working)
    price_jpy = int(price_match.group("price").replace(",", "")) if price_match is not None else None

    count_match = COUNT_RE.search(working)
    listing_count = int(count_match.group("count")) if count_match is not None else None

    if not title:
        return None

    return _ParsedHotItem(
        title=title,
        price_jpy=price_jpy,
        thumbnail_url=thumbnail_url,
        card_number=card_number,
        rarity=rarity,
        set_code=set_code,
        listing_count=listing_count,
        is_graded=False,
        condition=condition,
        detail_url=detail_url,
        board_url=board_url,
        note="Signal source: Cardrush best-seller order within the current high-rarity singles category.",
        source_label=source_label,
        source_rank=source_rank,
        demand_ratio=demand_ratio,
    )


def _parse_magi_text(
    text: str,
    *,
    detail_url: str,
    board_url: str,
    thumbnail_url: str | None = None,
    note: str = "Signal source: Magi popular/recommended Weiss Schwarz page order.",
    source_rank: int | None = None,
    source_label: str = "Magi recommendation rank",
    demand_ratio: float = 0.0,
) -> _ParsedHotItem | None:
    grading_match = GRADING_RE.match(text)
    is_graded = grading_match is not None
    grade_label = re.sub(r"\s+", "", grading_match.group("label")).upper() if grading_match is not None else None
    working = text[grading_match.end():].strip() if grading_match is not None else text

    price_match = MAGI_PRICE_RE.search(working)
    price_jpy = int(price_match.group("price").replace(",", "")) if price_match is not None else None

    count_match = COUNT_RE.search(working)
    listing_count = int(count_match.group("count")) if count_match is not None else None

    body_end = price_match.start() if price_match is not None else working.find("- 出品数")
    if body_end == -1:
        body_end = len(working)
    body = working[:body_end].strip()

    card_number = None
    rarity = None
    set_code = None
    ws_code_match = WS_CODE_RE.search(body)
    if ws_code_match is not None:
        card_number = ws_code_match.group("code")
        set_code = card_number.split("/", 1)[0].lower()
        prefix = body[: ws_code_match.start()].strip()
        title, rarity = _split_title_and_rarity(prefix)
    else:
        pokemon_code_match = POKEMON_INLINE_CODE_RE.search(body)
        if pokemon_code_match is not None:
            card_number = pokemon_code_match.group("code").strip()
            prefix = body[: pokemon_code_match.start()].strip()
            title, rarity = _split_title_and_rarity(prefix)
        else:
            title = body

    if not title:
        return None

    return _ParsedHotItem(
        title=title,
        price_jpy=price_jpy,
        thumbnail_url=thumbnail_url,
        card_number=card_number,
        rarity=rarity,
        set_code=set_code,
        listing_count=listing_count,
        is_graded=is_graded,
        condition=None,
        detail_url=detail_url,
        board_url=board_url,
        note=note,
        source_label=source_label,
        source_rank=source_rank,
        demand_ratio=demand_ratio,
        grade_label=grade_label,
    )


def _parse_snkrdunk_text(
    text: str,
    *,
    detail_url: str,
    board_url: str,
    thumbnail_url: str | None = None,
    note: str,
    source_label: str,
    source_rank: int,
    demand_ratio: float,
) -> _ParsedHotItem | None:
    working = " ".join(text.split())
    code_match = SNKRDUNK_CODE_BLOCK_RE.search(working)
    code_block = None if code_match is None else code_match.group("code").strip()
    prefix = working[: code_match.start()].strip() if code_match is not None else working
    title, rarity = _split_title_and_rarity(prefix)

    card_number = None
    set_code = None
    if code_block:
        if " " in code_block:
            left, right = code_block.split(" ", 1)
            left = left.strip()
            right = right.strip()
            if re.fullmatch(r"[A-Za-z0-9]+/[A-Za-z0-9-]+-[A-Za-z0-9]+", left):
                card_number = left
                set_code = left.split("/", 1)[0].lower()
                if rarity is None and _looks_like_rarity(right):
                    rarity = right
            elif re.fullmatch(r"[A-Za-z0-9-]+", left):
                set_code = left.lower()
                card_number = right
            else:
                card_number = code_block
        else:
            card_number = code_block
            ws_code_match = re.fullmatch(r"(?P<code>[A-Za-z0-9]+/[A-Za-z0-9-]+-[A-Za-z0-9]+)", code_block)
            if ws_code_match is not None:
                set_code = ws_code_match.group("code").split("/", 1)[0].lower()

    if not title:
        return None

    return _ParsedHotItem(
        title=title,
        price_jpy=None,
        thumbnail_url=thumbnail_url,
        card_number=card_number,
        rarity=rarity,
        set_code=set_code,
        listing_count=None,
        is_graded=False,
        condition=None,
        detail_url=detail_url,
        board_url=board_url,
        note=note,
        source_label=source_label,
        source_rank=source_rank,
        demand_ratio=demand_ratio,
    )


def _parse_yahoo_realtime_signal(
    *,
    html: str,
    query: str,
    search_url: str,
) -> HotCardSocialSignal | None:
    soup = BeautifulSoup(html, "html.parser")
    tweet_cards = soup.select(SOCIAL_TWEET_SELECTOR)
    if not tweet_cards:
        return None

    matched_post_count = 0
    engagement_count = 0
    normalized_query = normalize_text(query)

    for tweet_card in tweet_cards[:10]:
        body_node = tweet_card.select_one("p[class*='Tweet_body__']")
        if body_node is None:
            continue

        body_text = " ".join(body_node.get_text(" ", strip=True).split())
        if not body_text:
            continue
        if not _social_body_matches_query(normalized_query, body_text):
            continue

        params = ""
        menu_link = tweet_card.select_one("a[data-cl-params*='reply:'][data-cl-params*='retweet:'][data-cl-params*='like:']")
        if isinstance(menu_link, Tag):
            params = str(menu_link.get("data-cl-params", ""))

        matched_post_count += 1
        engagement_count += _social_engagement_from_params(params)

    if matched_post_count <= 0:
        return None

    return HotCardSocialSignal(
        query=query,
        search_url=search_url,
        matched_post_count=matched_post_count,
        engagement_count=engagement_count,
        score_ratio=_social_score_ratio(matched_post_count=matched_post_count, engagement_count=engagement_count),
    )


def _parse_snkrdunk_heading_rank(text: str) -> tuple[int, str] | None:
    match = SNKRDUNK_HEADING_RANK_RE.match(" ".join(text.split()))
    if match is None:
        return None

    rank = int(match.group("rank"))
    title = match.group("title").strip()
    if rank <= 0 or not title:
        return None
    return (rank, title)


def _find_next_snkrdunk_apparel_anchor(node: Tag) -> Tag | None:
    sibling = node.next_sibling
    while sibling is not None:
        if isinstance(sibling, Tag):
            if sibling.name == "h3":
                return None
            if sibling.name == "a" and isinstance(sibling.get("href"), str) and "/apparels/" in sibling["href"]:
                return sibling
            nested_anchor = sibling.select_one("a[href*='/apparels/']")
            if nested_anchor is not None:
                return nested_anchor
        sibling = sibling.next_sibling
    return None


def _extract_snkrdunk_primary_image_url(item: dict[str, object]) -> str | None:
    primary_media = item.get("primaryMedia")
    if not isinstance(primary_media, dict):
        return None
    image_url = primary_media.get("imageUrl")
    if not isinstance(image_url, str) or not image_url:
        return None
    return image_url


def _first_populated_text(*values: object) -> str | None:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _parse_jpy_price_text(value: object) -> int | None:
    if value is None:
        return None
    normalized = str(value).replace("￥", "¥")
    price_match = JPY_PRICE_RE.search(normalized)
    if price_match is None:
        price_match = re.search(r"(?P<price>\d[\d,]*)", normalized)
    if price_match is None:
        return None
    return int(price_match.group("price").replace(",", ""))


def _split_title_and_rarity(prefix: str) -> tuple[str, str | None]:
    parts = prefix.rsplit(" ", 1)
    if len(parts) == 2 and _looks_like_rarity(parts[1]):
        return parts[0].strip(), parts[1].strip()
    return prefix.strip(), None


def _rank_signal_ratio(
    *,
    rank: int,
    max_rank: int,
    source_weight: float,
    floor: float = 0.0,
) -> float:
    if rank <= 0 or max_rank <= 0 or source_weight <= 0:
        return 0.0
    bounded_rank = min(rank, max_rank)
    span = max(max_rank - 1, 1)
    position_ratio = (max_rank - bounded_rank) / span
    scaled_ratio = floor + ((1.0 - floor) * position_ratio)
    return round(min(1.0, max(0.0, scaled_ratio * source_weight)), 4)


def _looks_like_rarity(token: str) -> bool:
    value = token.strip().upper()
    if not value or len(value) > 6:
        return False
    return value.isalnum()


def _condition_priority(condition: str | None, is_graded: bool) -> int:
    if is_graded:
        return 0
    if condition is None:
        return 4
    normalized = normalize_text(condition)
    if "状態a" in normalized:
        return 3
    if "状態b" in normalized:
        return 2
    if "状態c" in normalized:
        return 1
    return 1


def _title_key(game: str, title: str, *, drop_game_suffixes: bool = False) -> str:
    normalized = normalize_text(title)
    if game == "pokemon" and drop_game_suffixes:
        normalized = normalized.removesuffix("ex")
    return normalized


def _build_social_query(*, game: str, item: _ParsedHotItem) -> str:
    query_parts = [item.title]
    if game == "pokemon":
        query_parts.append("ポケカ")
    elif game == "ws":
        query_parts.append("ヴァイス")

    if item.rarity and _looks_like_rarity(item.rarity):
        query_parts.append(item.rarity.upper())

    return " ".join(part.strip() for part in query_parts if part and part.strip())


def _parse_optional_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return None


def _buy_signal_label(offer: MarketOffer) -> str | None:
    direction = normalize_text(offer.attributes.get("price_change_direction", ""))
    if direction == "up":
        return "priceup"
    if direction == "down":
        return "pricedown"
    return None


def _social_body_matches_query(normalized_query: str, body_text: str) -> bool:
    normalized_body = normalize_text(body_text)
    query_tokens = [token for token in normalized_query.split() if token]
    if not query_tokens:
        return bool(normalized_body)

    relevant_tokens = [
        token
        for token in query_tokens
        if token not in {"ポケカ", "ヴァイス"}
    ]
    if not relevant_tokens:
        relevant_tokens = query_tokens

    matched_tokens = sum(1 for token in relevant_tokens if token in normalized_body)
    required_matches = 1 if len(relevant_tokens) <= 2 else 2
    return matched_tokens >= required_matches


def _social_engagement_from_params(params: str) -> int:
    if not params:
        return 0

    reply_count = _extract_social_count(SOCIAL_REPLY_RE, params)
    retweet_count = _extract_social_count(SOCIAL_RETWEET_RE, params)
    like_count = _extract_social_count(SOCIAL_LIKE_RE, params)
    quote_count = _extract_social_count(SOCIAL_QUOTE_RE, params)
    return like_count + retweet_count + reply_count + quote_count


def _extract_social_count(pattern: re.Pattern[str], params: str) -> int:
    match = pattern.search(params)
    if match is None:
        return 0
    return int(match.group("count"))


def _social_score_ratio(*, matched_post_count: int, engagement_count: int) -> float:
    post_ratio = min(1.0, math.log1p(matched_post_count) / math.log1p(8))
    engagement_ratio = min(1.0, math.log1p(max(engagement_count, 0)) / math.log1p(5000))
    return round((post_ratio * 0.45) + (engagement_ratio * 0.55), 4)


def _extract_cardrush_thumbnail_url(anchor: Tag) -> str | None:
    image = anchor.select_one("img")
    if image is None:
        return None
    return _pick_image_url(image, attribute_names=("data-x2", "src"))


def _extract_magi_thumbnail_url(anchor: Tag) -> str | None:
    image = anchor.select_one("img")
    if image is None:
        return None
    url = _pick_image_url(image, attribute_names=("data-src", "src"))
    if not url:
        return None
    return urljoin("https://magi.camp", url)


def _pick_image_url(image: Tag, *, attribute_names: tuple[str, ...]) -> str | None:
    for attribute_name in attribute_names:
        candidate = image.get(attribute_name)
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    return None
