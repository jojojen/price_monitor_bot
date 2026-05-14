from __future__ import annotations

import re

from market_monitor.models import MarketOffer
from market_monitor.normalize import normalize_card_number, normalize_text

from .catalog import TcgCardSpec

_SEALED_BOX_MARKERS = (
    "未開封box",
    "未開封 box",
    "box",
    "booster box",
    "ブースターボックス",
    "ボックス",
)
_SEALED_BOX_TOKEN_HINTS = {
    "charizard": ("リザードン",),
    "dream": ("ドリーム",),
    "premium": ("プレミアム",),
    "special": ("スペシャル",),
    "starter": ("スターター",),
}


def minimum_match_score(spec: TcgCardSpec) -> float:
    if spec.item_kind == "sealed_box":
        return 30.0
    if spec.card_number:
        return 40.0
    if spec.rarity or spec.set_code:
        return 32.0
    return 24.0


def score_tcg_offer(spec: TcgCardSpec, offer: MarketOffer) -> float:
    if spec.item_kind == "sealed_box":
        return _score_sealed_box_offer(spec, offer)

    title_norm = normalize_text(offer.title)
    alt_norm = normalize_text(offer.attributes.get("image_alt", ""))
    score = 0.0

    canonical_title = normalize_text(spec.title)
    if title_norm == canonical_title:
        score += 35
    elif canonical_title in title_norm or title_norm in canonical_title:
        score += 25
    else:
        canonical_compact = _compact_text(canonical_title)
        title_compact = _compact_text(title_norm)
        if len(canonical_compact) >= 2 and (
            canonical_compact in title_compact or title_compact in canonical_compact
        ):
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


def _score_sealed_box_offer(spec: TcgCardSpec, offer: MarketOffer) -> float:
    title_norm = normalize_text(offer.title)
    alt_norm = normalize_text(offer.attributes.get("image_alt", ""))
    score = 0.0

    canonical_title = normalize_text(spec.title)
    title_signal = 0.0
    if title_norm == canonical_title:
        title_signal += 45
    elif canonical_title and (canonical_title in title_norm or title_norm in canonical_title):
        title_signal += 34
    else:
        title_signal += _shared_sealed_box_token_score(spec.title, offer.title)
        title_signal += _shared_sealed_box_token_score(spec.title, offer.attributes.get("image_alt", ""))
        if title_signal <= 0:
            score -= 16
    score += title_signal

    for alias in spec.aliases:
        alias_norm = normalize_text(alias)
        if alias_norm and (alias_norm in title_norm or alias_norm in alt_norm):
            score += 12

    for keyword in spec.extra_keywords:
        keyword_norm = normalize_text(keyword)
        if keyword_norm and (keyword_norm in title_norm or keyword_norm in alt_norm):
            score += 8

    if _offer_looks_like_sealed_box(offer):
        score += 32
    else:
        score -= 24

    if offer.attributes.get("card_number"):
        score -= 16

    offer_set_code = normalize_text(offer.attributes.get("version_code", "") or offer.attributes.get("set_code", ""))
    if spec.set_code:
        if offer_set_code == normalize_text(spec.set_code):
            score += 12
        else:
            score -= 4

    if spec.set_name:
        set_name_norm = normalize_text(spec.set_name)
        if set_name_norm and (set_name_norm in title_norm or set_name_norm in alt_norm):
            score += 8

    if offer.attributes.get("is_graded") == "1":
        score -= 40

    return score


def _offer_looks_like_sealed_box(offer: MarketOffer) -> bool:
    if offer.attributes.get("product_kind") == "sealed_box":
        return True
    combined = normalize_text(" ".join(filter(None, (offer.title, offer.attributes.get("image_alt", "")))))
    return any(marker in combined for marker in _SEALED_BOX_MARKERS)


def _shared_sealed_box_token_score(left: str, right: str) -> float:
    right_text = right or ""
    left_tokens = {
        token.lower()
        for token in re.findall(r"[A-Za-z0-9]+", left or "")
        if token and token.lower() not in {"box"}
    }
    right_tokens = {
        token.lower()
        for token in re.findall(r"[A-Za-z0-9]+", right or "")
        if token and token.lower() not in {"box"}
    }
    score = 0.0
    for token in left_tokens:
        matched = token in right_tokens
        if not matched:
            matched = any(hint in right_text for hint in _SEALED_BOX_TOKEN_HINTS.get(token, ()))
        if not matched:
            continue
        if len(token) >= 8:
            score += 16
        elif len(token) >= 4:
            score += 10
        elif token in {"ex", "gx", "v", "vmax", "vstar"}:
            score += 4
    return score


def _card_numbers_match(left: str, right: str) -> bool:
    if left == right:
        return True
    if _normalize_trailing_card_number(left) == _normalize_trailing_card_number(right) != "":
        return True
    return _normalize_simple_pokemon_number(left) == _normalize_simple_pokemon_number(right) != ""


def _compact_text(value: str) -> str:
    return re.sub(r"[\s・･]+", "", value or "")


def _normalize_trailing_card_number(value: str) -> str:
    match = re.fullmatch(r"(?P<prefix>.+[-/])(?P<number>\d{1,3})", value)
    if match is None:
        return ""
    number = int(match.group("number"))
    if number <= 0:
        return ""
    return f"{match.group('prefix')}{number:03d}"


def _normalize_simple_pokemon_number(value: str) -> str:
    match = re.fullmatch(r"(?P<numerator>\d{1,3})/(?P<denominator>\d{1,3})", value)
    if match is None:
        return ""
    numerator = int(match.group("numerator"))
    denominator = int(match.group("denominator"))
    if numerator <= 0 or denominator <= 0:
        return ""
    return f"{numerator}/{denominator}"
