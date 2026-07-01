# Compatibility re-export shim — implementation has moved to telegram_nl.
from telegram_nl.natural_language import (
    TelegramNaturalLanguageIntent,
    TelegramNaturalLanguageRouter,
    build_telegram_natural_language_router,
    fallback_route_telegram_natural_language,
    fast_route_telegram_natural_language,
    slow_fallback_route_telegram_natural_language,
    _extract_opportunity_target,
    _extract_sns_schedule_minutes,
    _extract_watch_query,
    _looks_like_opportunity_remove_request,
    _looks_like_web_research_question,
    _normalize_intent,
    _normalize_keyword_values,
    _parse_kanji_number,
    _parse_price_threshold,
    _recover_lookup_fields,
    _split_keyword_phrase,
)

__all__ = [
    "TelegramNaturalLanguageIntent",
    "TelegramNaturalLanguageRouter",
    "build_telegram_natural_language_router",
    "fallback_route_telegram_natural_language",
    "fast_route_telegram_natural_language",
    "slow_fallback_route_telegram_natural_language",
]
