from __future__ import annotations

from price_monitor_bot.bot import (
    TelegramCommandProcessor,
    _build_photo_intent_options,
    _caption_requests_image_translation,
)


def _make_processor(*, recognizer=None) -> TelegramCommandProcessor:
    return TelegramCommandProcessor(
        allowed_chat_ids=frozenset({"123"}),
        lookup_renderer=lambda query: query.name,
        board_loader=lambda: (),
        catalog_renderer=lambda: "catalog",
        image_translate_recognizer=recognizer,
    )


def test_processor_uses_embedding_recognizer_when_present() -> None:
    # Recognizer accepts a paraphrase the keyword check would miss.
    processor = _make_processor(recognizer=lambda c: c == "這張圖寫什麼")
    assert processor.caption_requests_image_translation("這張圖寫什麼")
    assert not processor.caption_requests_image_translation("翻譯")  # recognizer is authoritative


def test_processor_falls_back_to_keyword_without_recognizer() -> None:
    processor = _make_processor(recognizer=None)
    assert processor.caption_requests_image_translation("翻譯")
    assert not processor.caption_requests_image_translation("查價")


def test_processor_falls_back_to_keyword_when_recognizer_errors() -> None:
    def boom(_c):
        raise RuntimeError("embed down")

    processor = _make_processor(recognizer=boom)
    assert processor.caption_requests_image_translation("翻譯")  # keyword fallback
    assert not processor.caption_requests_image_translation("查價")


def test_caption_requests_image_translation_keywords() -> None:
    assert _caption_requests_image_translation("翻譯")
    assert _caption_requests_image_translation("翻訳して")
    assert _caption_requests_image_translation("please translate")
    assert _caption_requests_image_translation("OCR 一下")
    assert not _caption_requests_image_translation("查價")
    assert not _caption_requests_image_translation("")
    assert not _caption_requests_image_translation(None)


def test_photo_intent_options_always_offer_ocr_translate() -> None:
    # Card photo: translate option is appended after the card options.
    pokemon = _build_photo_intent_options(parsed_game="pokemon", item_kind="card")
    assert pokemon[-1].action_key == "ocr_translate"
    assert pokemon[-1].synthetic_caption == "翻譯"
    assert pokemon[-1].option_number == len(pokemon)

    # Non-card screenshot (no game detected): translate option still present.
    unknown = _build_photo_intent_options(parsed_game=None, item_kind=None)
    assert any(opt.action_key == "ocr_translate" for opt in unknown)
