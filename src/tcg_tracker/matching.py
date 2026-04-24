from __future__ import annotations

import re

from market_monitor.models import MarketOffer
from market_monitor.normalize import normalize_card_number, normalize_text

from .catalog import TcgCardSpec


def minimum_match_score(spec: TcgCardSpec) -> float:
    if spec.card_number:
        return 40.0
    if spec.rarity or spec.set_code:
        return 32.0
    return 24.0


def score_tcg_offer(spec: TcgCardSpec, offer: MarketOffer) -> float:
    title_norm = normalize_text(offer.title)
    alt_norm = normalize_text(offer.attributes.get("image_alt", ""))
    score = 0.0

    canonical_title = normalize_text(spec.title)
    if title_norm == canonical_title:
        score += 35
    elif canonical_title in title_norm or title_norm in canonical_title:
        score += 25

    for alias in spec.aliases:
        alias_norm = normalize_text(alias)
        if alias_norm and (alias_norm in title_norm or alias_norm in alt_norm):
            score += 10

    for keyword in spec.extra_keywords:
        keyword_norm = normalize_text(keyword)
        if keyword_norm and (keyword_norm in title_norm or keyword_norm in alt_norm):
            score += 6

    offer_number = normalize_card_number(offer.attributes.get("card_number", ""))
    if spec.card_number:
        if _card_numbers_match(spec.normalized_card_number, offer_number):
            score += 40
        else:
            score -= 30

    offer_rarity = normalize_text(offer.attributes.get("rarity", ""))
    if spec.rarity:
        if offer_rarity == normalize_text(spec.rarity):
            score += 18
        else:
            score -= 8

    offer_set_code = normalize_text(offer.attributes.get("version_code", "") or offer.attributes.get("set_code", ""))
    if spec.set_code:
        if offer_set_code == normalize_text(spec.set_code):
            score += 15
        else:
            score -= 5

    if offer.attributes.get("is_graded") == "1":
        score -= 4

    if spec.set_name:
        set_name_norm = normalize_text(spec.set_name)
        if set_name_norm and set_name_norm in alt_norm:
            score += 8

    return score


def _card_numbers_match(left: str, right: str) -> bool:
    if left == right:
        return True
    return _normalize_simple_pokemon_number(left) == _normalize_simple_pokemon_number(right) != ""


def _normalize_simple_pokemon_number(value: str) -> str:
    match = re.fullmatch(r"(?P<numerator>\d{1,3})/(?P<denominator>\d{1,3})", value)
    if match is None:
        return ""
    numerator = int(match.group("numerator"))
    denominator = int(match.group("denominator"))
    if numerator <= 0 or denominator <= 0:
        return ""
    return f"{numerator}/{denominator}"
