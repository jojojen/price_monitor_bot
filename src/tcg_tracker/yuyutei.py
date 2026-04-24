from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import logging
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup, Tag

from market_monitor.http import HttpClient
from market_monitor.models import MarketOffer

from .catalog import TcgCardSpec
from .matching import minimum_match_score, score_tcg_offer
from .search_terms import build_lookup_terms

YUYUTEI_BASE_URL = "https://yuyu-tei.jp"
logger = logging.getLogger(__name__)


def parse_jpy(text: str) -> int | None:
    digits = "".join(character for character in text if character.isdigit())
    if not digits:
        return None
    return int(digits)


class YuyuteiClient:
    def __init__(self, http_client: HttpClient | None = None) -> None:
        self.http_client = http_client or HttpClient()

    def search_sell(self, spec: TcgCardSpec, *, search_word: str | None = None) -> list[MarketOffer]:
        html = self.http_client.get_text(
            f"{YUYUTEI_BASE_URL}/sell/{spec.source_code}/s/search",
            params={"search_word": search_word or spec.title, "rare": "", "type": ""},
        )
        return self._parse_search_page(html, price_kind="ask", source_category=spec.source_code)

    def search_buy(self, spec: TcgCardSpec, *, search_word: str | None = None) -> list[MarketOffer]:
        html = self.http_client.get_text(
            f"{YUYUTEI_BASE_URL}/buy/{spec.source_code}/s/search",
            params={"search_word": search_word or spec.title, "rare": "", "type": ""},
        )
        return self._parse_search_page(html, price_kind="bid", source_category=spec.source_code)

    def lookup(self, spec: TcgCardSpec, *, minimum_score: float | None = None) -> list[MarketOffer]:
        resolved_minimum_score = minimum_score if minimum_score is not None else minimum_match_score(spec)
        search_terms = self._search_terms(spec)
        logger.debug(
            "Yuyutei lookup starting title=%s card_number=%s rarity=%s set_code=%s search_terms=%s minimum_score=%s",
            spec.title,
            spec.card_number,
            spec.rarity,
            spec.set_code,
            list(search_terms),
            resolved_minimum_score,
        )
        offers = self._search_candidates(spec)
        logger.debug(
            "Yuyutei raw candidates count=%s candidates=%s",
            len(offers),
            [_offer_summary(offer) for offer in offers[: _log_limit()]],
        )
        matched: list[MarketOffer] = []
        for offer in offers:
            score = score_tcg_offer(spec, offer)
            logger.debug("Yuyutei candidate scored score=%s offer=%s", score, _offer_summary(offer))
            if score >= resolved_minimum_score:
                matched.append(replace(offer, score=score))

        matched.sort(
            key=lambda offer: (
                offer.score or 0,
                {"ask": 2, "market": 1, "last_sale": 1, "bid": 0}.get(offer.price_kind, 0),
                offer.price_jpy,
            ),
            reverse=True,
        )
        logger.debug(
            "Yuyutei matched offers count=%s matched=%s",
            len(matched),
            [_offer_summary(offer) for offer in matched[: _log_limit()]],
        )
        return matched

    def _search_candidates(self, spec: TcgCardSpec) -> list[MarketOffer]:
        offers: list[MarketOffer] = []
        seen: set[tuple[str, str]] = set()
        for search_word in self._search_terms(spec):
            logger.debug("Yuyutei search term=%s", search_word)
            for offer in [*self.search_sell(spec, search_word=search_word), *self.search_buy(spec, search_word=search_word)]:
                dedupe_key = (offer.url, offer.price_kind)
                if dedupe_key in seen:
                    continue
                seen.add(dedupe_key)
                offers.append(offer)
        return offers

    def _parse_search_page(self, html: str, *, price_kind: str, source_category: str) -> list[MarketOffer]:
        soup = BeautifulSoup(html, "html.parser")
        power = soup.select_one("div#power")
        if power is None:
            return []

        offers: list[MarketOffer] = []
        seen: set[tuple[str, str]] = set()
        search_root = power.parent if power.parent is not None else soup
        sections = []
        for section in search_root.select("div.cards-list"):
            if not section.select_one("div.card-product"):
                continue
            heading = section.find("h3")
            if heading is None:
                continue
            if "Card List" not in heading.get_text(" ", strip=True):
                continue
            sections.append(section)

        for section in sections:
            rarity = self._extract_section_rarity(section)
            for card in section.select("div.card-product"):
                offer = self._parse_card_product(card, default_rarity=rarity, price_kind=price_kind, source_category=source_category)
                if offer is None:
                    continue
                dedupe_key = (offer.url, offer.price_kind)
                if dedupe_key in seen:
                    continue
                seen.add(dedupe_key)
                offers.append(offer)

        return offers

    @staticmethod
    def _extract_section_rarity(section: Tag) -> str | None:
        heading = section.find("h3")
        if heading is None:
            return None
        badge = heading.find("span")
        if badge is None:
            return None
        value = badge.get_text(" ", strip=True).replace("\xa0", "").strip()
        return value or None

    def _parse_card_product(
        self,
        card: Tag,
        *,
        default_rarity: str | None,
        price_kind: str,
        source_category: str,
    ) -> MarketOffer | None:
        anchors = [anchor for anchor in card.select("a[href]") if "/card/" in anchor.get("href", "")]
        title_element = card.find("h4")
        price_element = card.find("strong")
        if not anchors or title_element is None or price_element is None:
            return None

        href = urljoin(YUYUTEI_BASE_URL, anchors[0]["href"])
        parsed_path = [part for part in urlparse(href).path.split("/") if part]
        version_code = parsed_path[-2] if len(parsed_path) >= 2 else ""
        product_id = parsed_path[-1] if parsed_path else href

        title = title_element.get_text(" ", strip=True)
        card_number_element = card.find("span", class_=lambda value: value and "text-center" in value and "border" in value)
        card_number = card_number_element.get_text(" ", strip=True) if card_number_element else ""

        image = card.find("img", class_="card")
        image_alt = image.get("alt", "").strip() if image else ""
        alt_number, alt_rarity = self._parse_alt_metadata(image_alt)
        card_number = card_number or alt_number
        rarity = default_rarity or alt_rarity

        price_jpy = parse_jpy(price_element.get_text(" ", strip=True))
        if price_jpy is None:
            return None

        compare_price = None
        compare_element = card.find("del")
        if compare_element is not None:
            compare_price = parse_jpy(compare_element.get_text(" ", strip=True))
        card_classes = set(card.get("class", []))
        price_change_direction = None
        if "priceup" in card_classes:
            price_change_direction = "up"
        elif "pricedown" in card_classes:
            price_change_direction = "down"

        availability = None
        availability_element = card.select_one("label.cart_sell_zaiko")
        if availability_element is not None:
            availability = availability_element.get_text(" ", strip=True).replace("在庫 :", "").strip()

        attributes = {
            "card_number": card_number,
            "rarity": rarity or "",
            "version_code": version_code,
            "product_id": product_id,
            "image_alt": image_alt,
        }
        if compare_price is not None:
            attributes["compare_price_jpy"] = str(compare_price)
        if price_change_direction is not None:
            attributes["price_change_direction"] = price_change_direction

        return MarketOffer(
            source="yuyutei",
            listing_id=f"{version_code}:{product_id}",
            url=href,
            title=title,
            price_jpy=price_jpy,
            price_kind=price_kind,  # type: ignore[arg-type]
            captured_at=datetime.now(timezone.utc),
            source_category=source_category,
            availability=availability,
            attributes=attributes,
        )

    @staticmethod
    def _parse_alt_metadata(image_alt: str) -> tuple[str, str | None]:
        parts = image_alt.strip().split()
        if len(parts) < 2:
            return "", None
        card_number = parts[0]
        rarity = parts[1]
        return card_number, rarity

    @staticmethod
    def _search_terms(spec: TcgCardSpec) -> tuple[str, ...]:
        return build_lookup_terms(spec)


def _offer_summary(offer: MarketOffer) -> dict[str, object]:
    return {
        "source": offer.source,
        "price_kind": offer.price_kind,
        "price_jpy": offer.price_jpy,
        "title": offer.title,
        "url": offer.url,
        "card_number": offer.attributes.get("card_number", ""),
        "rarity": offer.attributes.get("rarity", ""),
        "set_code": offer.attributes.get("version_code", ""),
    }


_LOG_LIMIT = 5


def _log_limit() -> int:
    return _LOG_LIMIT
