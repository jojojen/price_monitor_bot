from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import logging
import re
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup

from market_monitor.http import HttpClient
from market_monitor.models import MarketOffer
from market_monitor.normalize import normalize_card_number

from .catalog import TcgCardSpec
from .matching import minimum_match_score, score_tcg_offer
from .search_terms import generic_card_number_variants

SURUGAYA_BASE_URL = "https://www.suruga-ya.jp"
SURUGAYA_SEARCH_URL = f"{SURUGAYA_BASE_URL}/search"
logger = logging.getLogger(__name__)

_SUPPORTED_GAMES = {"union_arena"}
_TCG_CODE_RE = re.compile(r"(?P<code>[A-Z0-9]+/[A-Z0-9]+(?:-[A-Z0-9]+)*-\d{1,3})", re.IGNORECASE)
_RARITY_RE = re.compile(r"\[(?P<rarity>[A-Z0-9*★-]+)\]")


class SurugayaClient:
    def __init__(self, http_client: HttpClient | None = None) -> None:
        self.http_client = http_client or HttpClient()

    def lookup(self, spec: TcgCardSpec, *, minimum_score: float | None = None) -> list[MarketOffer]:
        if spec.game not in _SUPPORTED_GAMES or not spec.card_number:
            return []

        resolved_minimum_score = minimum_score if minimum_score is not None else minimum_match_score(spec)
        matched: list[MarketOffer] = []
        seen: set[tuple[str, str]] = set()
        for offer in self._search_candidates(spec):
            score = score_tcg_offer(spec, offer)
            logger.debug("Surugaya candidate scored score=%s offer=%s", score, _offer_summary(offer))
            if score < resolved_minimum_score:
                continue
            key = (offer.url, offer.price_kind)
            if key in seen:
                continue
            seen.add(key)
            matched.append(replace(offer, score=score))
        return matched

    def _search_candidates(self, spec: TcgCardSpec) -> list[MarketOffer]:
        offers: list[MarketOffer] = []
        seen_urls: set[str] = set()
        for query in _build_surugaya_queries(spec):
            logger.debug("Surugaya search query=%s", query)
            html = self.http_client.get_text(SURUGAYA_SEARCH_URL, params={"search_word": query})
            search_offers = self._parse_search_page(html)
            logger.debug(
                "Surugaya raw search candidates query=%s count=%s offers=%s",
                query,
                len(search_offers),
                [_offer_summary(offer) for offer in search_offers[: _log_limit()]],
            )
            for search_offer in search_offers:
                if search_offer.url in seen_urls:
                    continue
                seen_urls.add(search_offer.url)
                detail_offers = self._load_detail_offers(search_offer)
                offers.extend(detail_offers or [search_offer])
        return offers

    def _load_detail_offers(self, search_offer: MarketOffer) -> list[MarketOffer]:
        try:
            html = self.http_client.get_text(search_offer.url)
        except Exception:
            logger.exception("Surugaya detail fetch failed url=%s", search_offer.url)
            return []
        return self._parse_detail_page(html, fallback_offer=search_offer)

    def _parse_search_page(self, html: str) -> list[MarketOffer]:
        soup = BeautifulSoup(html, "html.parser")
        offers: list[MarketOffer] = []
        for item in soup.select("div.item_box"):
            anchor = item.select_one("a[href*='/product/detail/']")
            title_element = item.select_one(".title")
            price_element = item.select_one(".price_teika")
            if anchor is None or title_element is None or price_element is None:
                continue

            detail_url = _canonical_detail_url(urljoin(SURUGAYA_BASE_URL, anchor["href"]))
            title = " ".join(title_element.get_text(" ", strip=True).split())
            price_jpy = parse_jpy(price_element.get_text(" ", strip=True))
            if not title or price_jpy is None:
                continue

            offers.append(_build_offer(url=detail_url, title=title, price_jpy=price_jpy, price_kind="ask"))
        return offers

    def _parse_detail_page(self, html: str, *, fallback_offer: MarketOffer) -> list[MarketOffer]:
        soup = BeautifulSoup(html, "html.parser")
        title_element = soup.select_one("h1")
        title = (
            " ".join(title_element.get_text(" ", strip=True).split())
            if title_element is not None
            else fallback_offer.title
        )

        offers: list[MarketOffer] = []
        ask_element = soup.select_one(".text-price-detail.price-buy")
        ask_price = parse_jpy(ask_element.get_text(" ", strip=True)) if ask_element is not None else None
        if ask_price is not None:
            availability = _detail_availability(soup)
            offers.append(
                _build_offer(
                    url=fallback_offer.url,
                    title=title,
                    price_jpy=ask_price,
                    price_kind="ask",
                    availability=availability,
                )
            )

        bid_element = soup.select_one(".purchase-price")
        bid_price = parse_jpy(bid_element.get_text(" ", strip=True)) if bid_element is not None else None
        if bid_price is not None:
            offers.append(
                _build_offer(
                    url=fallback_offer.url,
                    title=title,
                    price_jpy=bid_price,
                    price_kind="bid",
                    availability="buylist",
                )
            )

        return offers


def parse_jpy(text: str) -> int | None:
    digits = "".join(character for character in text if character.isdigit())
    if not digits:
        return None
    return int(digits)


def _build_surugaya_queries(spec: TcgCardSpec) -> tuple[str, ...]:
    if not spec.card_number:
        return ()
    values = [spec.card_number, *generic_card_number_variants(spec.card_number)]
    deduped: list[str] = []
    for value in values:
        cleaned = value.strip()
        if cleaned and cleaned not in deduped:
            deduped.append(cleaned)
    return tuple(deduped)


def _build_offer(
    *,
    url: str,
    title: str,
    price_jpy: int,
    price_kind: str,
    availability: str | None = None,
) -> MarketOffer:
    card_number = _extract_card_number(title)
    listing_id = urlparse(url).path.strip("/") or url
    set_code = _derive_set_code(card_number)
    attributes = {
        "card_number": card_number,
        "rarity": _extract_rarity(title),
        "version_code": set_code,
        "set_code": set_code,
        "image_alt": title,
    }
    return MarketOffer(
        source="surugaya",
        listing_id=listing_id,
        url=url,
        title=_display_title(title),
        price_jpy=price_jpy,
        price_kind=price_kind,  # type: ignore[arg-type]
        captured_at=datetime.now(timezone.utc),
        source_category="specialty_store",
        availability=availability,
        attributes=attributes,
    )


def _extract_card_number(title: str) -> str:
    match = _TCG_CODE_RE.search(title.upper())
    return "" if match is None else match.group("code").upper()


def _extract_rarity(title: str) -> str:
    match = _RARITY_RE.search(title.upper())
    return "" if match is None else match.group("rarity").upper()


def _derive_set_code(card_number: str) -> str:
    if not card_number:
        return ""
    return card_number.split("/", 1)[0].strip().lower()


def _display_title(title: str) -> str:
    text = re.sub(r"^.*?\]\s*[：:]", "", title).strip()
    text = re.sub(r"（[^）]*）", "", text).strip()
    return text or title


def _canonical_detail_url(url: str) -> str:
    parsed = urlparse(url)
    return f"{parsed.scheme}://{parsed.netloc}{parsed.path}"


def _detail_availability(soup: BeautifulSoup) -> str | None:
    amount_max = soup.select_one("input.amount_max")
    if amount_max is not None:
        amount_value = str(amount_max.get("value") or "").strip()
        if amount_value and amount_value != "0":
            return f"stock {amount_value}"
    if soup.select_one("button.btn_buy, button.cart1") is not None:
        return "available"
    return None


def _offer_summary(offer: MarketOffer) -> dict[str, object]:
    return {
        "source": offer.source,
        "price_kind": offer.price_kind,
        "price_jpy": offer.price_jpy,
        "title": offer.title,
        "url": offer.url,
        "card_number": normalize_card_number(offer.attributes.get("card_number", "")),
        "rarity": offer.attributes.get("rarity", ""),
    }


_LOG_LIMIT = 5


def _log_limit() -> int:
    return _LOG_LIMIT
