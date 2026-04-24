from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from market_monitor.models import FairValueEstimate, MarketOffer, TrackedItem

from tcg_tracker.catalog import TcgCardSpec
from tcg_tracker.image_lookup import (
    TcgVisionSettings,
    ParsedCardImage,
    TcgImagePriceService,
    _merge_local_vision_candidate,
    _repair_pokemon_card_number_with_slab,
    parse_image_caption_hints,
    parse_tcg_ocr_text,
)
from tcg_tracker.hot_cards import TcgLookupHint
from tcg_tracker.local_vision import LocalVisionCardCandidate, LocalVisionTimeoutError, OllamaLocalVisionClient
from tcg_tracker.service import TcgLookupResult
from tests.image_lookup_case_fixtures import get_image_lookup_live_case

CHARIZARD_JP = "\u30ea\u30b6\u30fc\u30c9\u30f3ex"
PIKACHU_EX_JP = "\u30d4\u30ab\u30c1\u30e5\u30a6ex"
RIRIE_JP = "\u30ea\u30fc\u30ea\u30a8\u306e\u30d4\u30c3\u30d4ex"


def test_parse_image_caption_hints_supports_scan_prefix() -> None:
    assert parse_image_caption_hints("/scan pokemon Pikachu ex") == ("pokemon", "Pikachu ex")
    assert parse_image_caption_hints("ws Hatsune Miku") == ("ws", "Hatsune Miku")
    assert parse_image_caption_hints(None) == (None, None)


def test_parse_image_caption_hints_ignores_generic_price_request_text() -> None:
    assert parse_image_caption_hints("查這個box市價") == (None, None)
    assert parse_image_caption_hints("/scan pokemon 查這個box市價") == ("pokemon", None)


def test_parse_tcg_ocr_text_extracts_charizard_reference_fields() -> None:
    raw_text = "\n".join(
        [
            "2025 POKEMON M2a JP",
            "MEGA CHARIZARD X ex",
            "GEM MT 10",
            CHARIZARD_JP,
            "223/193 MA",
        ]
    )

    parsed = parse_tcg_ocr_text(raw_text)

    assert parsed.status == "success"
    assert parsed.game == "pokemon"
    assert parsed.title == CHARIZARD_JP
    assert parsed.card_number == "223/193"
    assert parsed.rarity == "MA"
    assert parsed.set_code == "m2a"
    assert "MEGA CHARIZARD X ex" in parsed.aliases


def test_parse_tcg_ocr_text_preserves_zero_padded_pokemon_card_numbers() -> None:
    raw_text = "\n".join(
        [
            "2025 POKEMON M2 JP",
            "MEGA CHARIZARD X ex",
            "SPECIAL ART RARE",
            "110/080 SAR",
        ]
    )

    parsed = parse_tcg_ocr_text(raw_text)

    assert parsed.card_number == "110/080"
    assert parsed.rarity == "SAR"


def test_parse_tcg_ocr_text_recovers_dense_footer_card_number() -> None:
    raw_text = "\n".join(
        [
            "sv2a 2017,165 SAR",
            "lee. cath Cencto",
        ]
    )

    parsed = parse_tcg_ocr_text(raw_text, game_hint="pokemon")

    assert parsed.card_number == "201/165"
    assert parsed.rarity == "SAR"


def test_repair_pokemon_card_number_with_slab_repairs_wrong_numerator() -> None:
    assert _repair_pokemon_card_number_with_slab("470/080", "110") == "110/080"


def test_parse_tcg_ocr_text_rejects_lowercase_multiline_gibberish_title() -> None:
    raw_text = "\n".join(
        [
            "lee. cath Cero",
            "sv2a 2077,165 SAR",
        ]
    )

    parsed = parse_tcg_ocr_text(raw_text, game_hint="pokemon")

    assert parsed.title is None
    assert parsed.card_number == "207/165"


def test_parse_tcg_ocr_text_rejects_zero_denominator_card_number_noise() -> None:
    raw_text = "\n".join(
        [
            "2025 POKEMON M2 JP",
            "MEGA CHARIZARD X ex",
            "SPECIAL ART RARE",
            "110/0 A",
        ]
    )

    parsed = parse_tcg_ocr_text(raw_text)

    assert parsed.card_number is None
    assert parsed.rarity == "SAR"


def test_parse_tcg_ocr_text_prefers_valid_zero_padded_number_over_slab_noise() -> None:
    raw_text = "\n".join(
        [
            "2025 POKEMON M2 JP",
            "MEGA CHARIZARD X ex",
            "SPECIAL ART RARE",
            "#110",
            "110/0 A",
            "メガリザードンXex",
            "110/080",
        ]
    )

    parsed = parse_tcg_ocr_text(raw_text)

    assert parsed.card_number == "110/080"
    assert parsed.rarity == "SAR"


def test_parse_tcg_ocr_text_recovers_slab_set_and_rarity_from_noisy_header() -> None:
    raw_text = "\n".join(
        [
            "2025 POKEMONMZ JP",
            "#110",
            "MEGA CHARIZARD X ex",
            "SS",
            "SPECIAL AATAARE",
        ]
    )

    parsed = parse_tcg_ocr_text(raw_text, game_hint="pokemon")

    assert parsed.title == "MEGA CHARIZARD X ex"
    assert parsed.card_number is None
    assert parsed.rarity == "SAR"
    assert parsed.set_code == "m2"


def test_parse_tcg_ocr_text_extracts_ririe_reference_fields() -> None:
    raw_text = "\n".join(
        [
            "2025 POKEMON SV9 JP",
            "LILLIE'S CLEFAIRY ex",
            "SPECIAL ART RARE",
            RIRIE_JP,
            "126/100 SAR",
        ]
    )

    parsed = parse_tcg_ocr_text(raw_text)

    assert parsed.status == "success"
    assert parsed.game == "pokemon"
    assert parsed.title == RIRIE_JP
    assert parsed.card_number == "126/100"
    assert parsed.rarity == "SAR"
    assert parsed.set_code == "sv9"


def test_parse_tcg_ocr_text_extracts_pokemon_promo_footer_codes() -> None:
    raw_text = "\n".join(
        [
            "2023 POKEMON SVP EN",
            "Pikachu with Grey Felt Hat",
            "SVP EN 085",
        ]
    )

    parsed = parse_tcg_ocr_text(raw_text, game_hint="pokemon")

    assert parsed.status == "success"
    assert parsed.game == "pokemon"
    assert parsed.title == "Pikachu with Grey Felt Hat"
    assert parsed.card_number == "085/SV-P"
    assert parsed.set_code == "svp"


def test_parse_tcg_ocr_text_detects_pokemon_sealed_box() -> None:
    raw_text = "\n".join(
        [
            "ポケモンカードゲーム スカーレット&バイオレット",
            "強化拡張パック",
            "ポケモンカード151",
            "ランダム7枚入り",
            "SV2a",
        ]
    )

    parsed = parse_tcg_ocr_text(raw_text, game_hint="pokemon")

    assert parsed.status == "success"
    assert parsed.game == "pokemon"
    assert parsed.item_kind == "sealed_box"
    assert parsed.title == "強化拡張パック ポケモンカード151"
    assert parsed.card_number is None
    assert parsed.rarity is None
    assert parsed.set_code == "sv2a"


def test_parse_tcg_ocr_text_rejects_gibberish_sealed_box_title() -> None:
    raw_text = "\n".join(
        [
            "ポケモンカードゲーム スカーレット&バイオレット",
            "強化拡張パック",
            "mw Om",
            "s29n",
            "ランダム7枚入り",
        ]
    )

    parsed = parse_tcg_ocr_text(raw_text, game_hint="pokemon")

    assert parsed.item_kind == "sealed_box"
    assert parsed.title is None
    assert parsed.status == "unresolved"


def test_parse_tcg_ocr_text_accepts_special_collection_number() -> None:
    raw_text = "\n".join(
        [
            "2025 POKEMON MC JP",
            PIKACHU_EX_JP,
            "764/742",
        ]
    )

    parsed = parse_tcg_ocr_text(raw_text, game_hint="pokemon")

    assert parsed.status == "success"
    assert parsed.game == "pokemon"
    assert parsed.title == PIKACHU_EX_JP
    assert parsed.card_number == "764/742"
    assert parsed.set_code == "mc"


def test_parse_tcg_ocr_text_rejects_copyright_noise_as_title() -> None:
    raw_text = "\n".join(
        [
            "Seta Fakerion/nintendo/Cieatures/GAMEPREAK sample",
            "@2023 Pokemon/Nintendo/Creatures/GAMEFREAK",
            "201/16559",
        ]
    )

    parsed = parse_tcg_ocr_text(raw_text, game_hint="pokemon")

    assert parsed.status == "unresolved"
    assert parsed.game == "pokemon"
    assert parsed.title is None
    assert parsed.card_number == "201/165"


def test_parse_tcg_ocr_text_accepts_easyocr_style_japanese_title() -> None:
    raw_text = "\n".join(
        [
            CHARIZARD_JP,
            "201/16559",
            "@2023 Pokemon/Nintendo/Creatures/GAMEFREAK",
        ]
    )

    parsed = parse_tcg_ocr_text(raw_text, game_hint="pokemon")

    assert parsed.status == "success"
    assert parsed.game == "pokemon"
    assert parsed.title == CHARIZARD_JP
    assert parsed.card_number == "201/165"


def test_parse_tcg_ocr_text_rejects_vowel_heavy_gibberish_title() -> None:
    raw_text = "\n".join(
        [
            "oon eee ante ee",
            "9/30",
        ]
    )

    parsed = parse_tcg_ocr_text(raw_text)

    assert parsed.title is None
    assert parsed.card_number == "9/30"


def test_parse_tcg_ocr_text_keeps_legit_english_title() -> None:
    raw_text = "\n".join(
        [
            "Pikachu with Grey Felt Hat",
            "085/SV-P",
        ]
    )

    parsed = parse_tcg_ocr_text(raw_text, game_hint="pokemon")

    assert parsed.title == "Pikachu with Grey Felt Hat"
    assert parsed.card_number == "085/SV-P"


def test_parse_tcg_ocr_text_prefers_ws_when_weiss_signals_and_ws_code_are_present() -> None:
    raw_text = "\n".join(
        [
            "ヴァイスシュヴァルツ",
            "未来の花嫁 タミコ",
            "CS/S114-043",
            "153/165",
        ]
    )

    parsed = parse_tcg_ocr_text(raw_text)

    assert parsed.game == "ws"
    assert parsed.card_number == "CS/S114-043"
    assert parsed.title == "未来の花嫁 タミコ"


def test_local_vision_prompt_requires_exactly_one_identifiable_card() -> None:
    client = OllamaLocalVisionClient(
        endpoint="http://127.0.0.1:11434",
        model="qwen2.5vl:7b",
        timeout_seconds=30,
    )

    prompt = client._build_prompt(game_hint=None, title_hint=None)

    assert "exactly one identifiable trading card" in prompt
    assert "Do not merge multiple cards" in prompt


def test_image_service_reports_unavailable_when_tesseract_is_missing() -> None:
    service = TcgImagePriceService(
        db_path="data/test-image-lookup.sqlite3",
        tesseract_path="C:/missing/tesseract.exe",
    )
    sample_path = get_image_lookup_live_case("pokemon-mega-charizard-x-ex-223-193-ma-m2a").image_path

    outcome = service.lookup_image(sample_path, caption="/scan pokemon")

    assert outcome.status == "unavailable"
    assert outcome.lookup_result is None
    assert any("OPENCLAW_TESSERACT_PATH" in warning for warning in outcome.warnings)


def test_lookup_image_uses_card_number_placeholder_to_recover_title(monkeypatch, tmp_path) -> None:
    calls: list[TcgCardSpec] = []

    class StubService:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def lookup(self, spec: TcgCardSpec, *, persist: bool = True) -> TcgLookupResult:
            calls.append(spec)
            if spec.card_number == "201/165":
                offer = MarketOffer(
                    source="cardrush_pokemon",
                    listing_id="sv2a-201-165",
                    url="https://example.com/charizard",
                    title=CHARIZARD_JP,
                    price_jpy=60800,
                    price_kind="ask",
                    captured_at=datetime.now(timezone.utc),
                    source_category="specialty_store",
                    attributes={"card_number": "201/165", "rarity": "SAR", "version_code": "sv2a"},
                )
                fair_value = FairValueEstimate(
                    item_id="tcg-charizard",
                    amount_jpy=60800,
                    confidence=0.82,
                    sample_count=1,
                    reasoning=("stub",),
                )
                item = TrackedItem(
                    item_id="tcg-charizard",
                    item_type="tcg_card",
                    category="tcg",
                    title=spec.title,
                    attributes={"game": "pokemon", "card_number": "201/165", "rarity": "SAR", "set_code": "sv2a"},
                )
                return TcgLookupResult(
                    spec=spec,
                    item=item,
                    offers=(offer,),
                    fair_value=fair_value,
                    notes=(),
                )

            item = TrackedItem(
                item_id="tcg-empty",
                item_type="tcg_card",
                category="tcg",
                title=spec.title,
                attributes={"game": spec.game},
            )
            return TcgLookupResult(
                spec=spec,
                item=item,
                offers=(),
                fair_value=None,
                notes=("No matching offers were found.",),
            )

    monkeypatch.setattr("tcg_tracker.image_lookup.TcgPriceService", StubService)

    service = TcgImagePriceService(
        db_path=tmp_path / "monitor.sqlite3",
        tesseract_path="C:/missing/tesseract.exe",
    )
    parsed = ParsedCardImage(
        status="unresolved",
        game="pokemon",
        title=None,
        aliases=(),
        card_number="201/165",
        rarity="SAR",
        set_code=None,
        raw_text="201/16559",
        extracted_lines=("201/16559",),
        warnings=(),
    )
    monkeypatch.setattr(service, "parse_image", lambda *args, **kwargs: parsed)

    outcome = service.lookup_image(tmp_path / "telegram-upload-charizard.jpg", persist=False)

    assert outcome.status == "success"
    assert outcome.lookup_result is not None
    assert outcome.lookup_result.spec.title == CHARIZARD_JP
    assert outcome.parsed.title == CHARIZARD_JP
    assert any("Resolved the card title from OCR metadata fallback" in warning for warning in outcome.warnings)
    assert calls[0].title == "201/165"
    assert calls[-1].title == CHARIZARD_JP


def test_lookup_image_supports_sealed_box_products(monkeypatch, tmp_path) -> None:
    class StubService:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def lookup(self, spec: TcgCardSpec, *, persist: bool = True) -> TcgLookupResult:
            offer = MarketOffer(
                source="magi",
                listing_id="box-151",
                url="https://example.com/box-151",
                title="強化拡張パック ポケモンカード151 未開封BOX",
                price_jpy=70000,
                price_kind="market",
                captured_at=datetime.now(timezone.utc),
                source_category="marketplace",
                attributes={"product_kind": "sealed_box", "set_code": "sv2a"},
            )
            item = TrackedItem(
                item_id="tcg-box-151",
                item_type="tcg_sealed_box",
                category="tcg",
                title=spec.title,
                attributes={"game": "pokemon", "item_kind": "sealed_box", "set_code": "sv2a"},
            )
            fair_value = FairValueEstimate(
                item_id="tcg-box-151",
                amount_jpy=70000,
                confidence=0.8,
                sample_count=1,
                reasoning=("stub",),
            )
            return TcgLookupResult(spec=spec, item=item, offers=(offer,), fair_value=fair_value, notes=())

    monkeypatch.setattr("tcg_tracker.image_lookup.TcgPriceService", StubService)

    service = TcgImagePriceService(
        db_path=tmp_path / "monitor.sqlite3",
        tesseract_path="C:/missing/tesseract.exe",
    )
    parsed = ParsedCardImage(
        status="success",
        game="pokemon",
        title="強化拡張パック ポケモンカード151",
        aliases=("ポケモンカード151",),
        card_number=None,
        rarity=None,
        set_code="sv2a",
        raw_text="強化拡張パック\nポケモンカード151\nSV2a",
        extracted_lines=("強化拡張パック", "ポケモンカード151", "SV2a"),
        item_kind="sealed_box",
        warnings=(),
    )
    monkeypatch.setattr(service, "parse_image", lambda *args, **kwargs: parsed)

    outcome = service.lookup_image(tmp_path / "telegram-upload-box.jpg", persist=False)

    assert outcome.status == "success"
    assert outcome.lookup_result is not None
    assert outcome.lookup_result.spec.item_kind == "sealed_box"
    assert outcome.lookup_result.spec.title == "強化拡張パック ポケモンカード151"


def test_lookup_image_trusts_card_number_recovery_when_title_is_unusable(monkeypatch, tmp_path) -> None:
    class StubService:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def lookup(self, spec: TcgCardSpec, *, persist: bool = True) -> TcgLookupResult:
            if spec.title == "201/165":
                offer = MarketOffer(
                    source="magi",
                    listing_id="sv2a-201-165",
                    url="https://example.com/charizard",
                    title=CHARIZARD_JP,
                    price_jpy=60800,
                    price_kind="market",
                    captured_at=datetime.now(timezone.utc),
                    source_category="marketplace",
                    attributes={"card_number": "201/165", "rarity": "SAR", "version_code": "sv2a"},
                )
                item = TrackedItem(
                    item_id="tcg-charizard",
                    item_type="tcg_card",
                    category="tcg",
                    title=spec.title,
                    attributes={"game": "pokemon", "card_number": "201/165", "rarity": "SAR", "set_code": "sv2a"},
                )
                fair_value = FairValueEstimate(
                    item_id="tcg-charizard",
                    amount_jpy=60800,
                    confidence=0.82,
                    sample_count=1,
                    reasoning=("stub",),
                )
                return TcgLookupResult(spec=spec, item=item, offers=(offer,), fair_value=fair_value, notes=())

            item = TrackedItem(
                item_id="tcg-empty",
                item_type="tcg_card",
                category="tcg",
                title=spec.title,
                attributes={"game": spec.game},
            )
            return TcgLookupResult(spec=spec, item=item, offers=(), fair_value=None, notes=("No matching offers were found.",))

    monkeypatch.setattr("tcg_tracker.image_lookup.TcgPriceService", StubService)

    service = TcgImagePriceService(
        db_path=tmp_path / "monitor.sqlite3",
    )
    parsed = ParsedCardImage(
        status="success",
        game="pokemon",
        title="exルー川",
        aliases=(),
        card_number="201/165",
        rarity="SS",
        set_code="s2",
        raw_text="201/165\nexルー川",
        extracted_lines=("201/165", "exルー川"),
        warnings=(),
    )

    resolved_parsed, spec = service._prepare_lookup_spec(parsed)

    assert spec is not None
    assert spec.title == CHARIZARD_JP
    assert spec.card_number == "201/165"
    assert spec.rarity == "SAR"
    assert spec.set_code == "sv2a"
    assert resolved_parsed.title == CHARIZARD_JP


def test_parse_image_uses_local_vision_when_tesseract_is_missing(tmp_path) -> None:
    image_path = tmp_path / "telegram-upload-charizard.jpg"
    image_path.write_bytes(b"fake-image")

    service = TcgImagePriceService(
        db_path=tmp_path / "monitor.sqlite3",
        tesseract_path="C:/missing/tesseract.exe",
    )

    class StubVisionClient:
        descriptor = "ollama:qwen2.5vl:3b"

        def analyze_card_image(
            self,
            image_path: Path,
            *,
            game_hint: str | None = None,
            title_hint: str | None = None,
        ) -> LocalVisionCardCandidate:
            return LocalVisionCardCandidate(
                backend="ollama",
                model="qwen2.5vl:3b",
                game="pokemon",
                title=CHARIZARD_JP,
                aliases=("Charizard ex",),
                card_number="201/165",
                rarity="SAR",
                set_code="sv2a",
                confidence=0.93,
            )

    service._local_vision_clients = (StubVisionClient(),)

    parsed = service.parse_image(image_path)

    assert service.is_available() is True
    assert parsed.status == "success"
    assert parsed.game == "pokemon"
    assert parsed.title == CHARIZARD_JP
    assert parsed.card_number == "201/165"
    assert parsed.rarity == "SAR"
    assert parsed.set_code == "sv2a"
    assert any("Applied local vision fallback via ollama:qwen2.5vl:3b." in warning for warning in parsed.warnings)


def test_lookup_image_uses_footer_focused_local_vision_metadata_to_recover_charizard(monkeypatch, tmp_path) -> None:
    calls: list[TcgCardSpec] = []

    class StubService:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def lookup(self, spec: TcgCardSpec, *, persist: bool = True) -> TcgLookupResult:
            calls.append(spec)
            if spec.card_number == "201/165":
                offer = MarketOffer(
                    source="cardrush_pokemon",
                    listing_id="sv2a-201-165",
                    url="https://example.com/charizard",
                    title=CHARIZARD_JP,
                    price_jpy=60800,
                    price_kind="ask",
                    captured_at=datetime.now(timezone.utc),
                    source_category="specialty_store",
                    attributes={"card_number": "201/165", "rarity": "SAR", "version_code": "sv2a"},
                )
                fair_value = FairValueEstimate(
                    item_id="tcg-charizard",
                    amount_jpy=60800,
                    confidence=0.82,
                    sample_count=1,
                    reasoning=("stub",),
                )
                item = TrackedItem(
                    item_id="tcg-charizard",
                    item_type="tcg_card",
                    category="tcg",
                    title=spec.title,
                    attributes={"game": "pokemon", "card_number": "201/165", "rarity": "SAR", "set_code": "sv2a"},
                )
                return TcgLookupResult(spec=spec, item=item, offers=(offer,), fair_value=fair_value, notes=())

            item = TrackedItem(
                item_id="tcg-empty",
                item_type="tcg_card",
                category="tcg",
                title=spec.title,
                attributes={"game": spec.game},
            )
            return TcgLookupResult(spec=spec, item=item, offers=(), fair_value=None, notes=("No matching offers were found.",))

    monkeypatch.setattr("tcg_tracker.image_lookup.TcgPriceService", StubService)

    service = TcgImagePriceService(
        db_path=tmp_path / "monitor.sqlite3",
        tesseract_path=str(tmp_path / "tesseract.exe"),
    )
    service._tesseract_path = "tesseract"
    monkeypatch.setattr(
        service,
        "_extract_text",
        lambda _image_path: (
            "\n".join(
                [
                    "lee. cath Cero",
                    "sv2a 2077,165 SAR",
                ]
            ),
            (),
        ),
    )

    class StubVisionClient:
        descriptor = "ollama:gemma3:12b"

        def analyze_card_image(
            self,
            image_path: Path,
            *,
            game_hint: str | None = None,
            title_hint: str | None = None,
        ) -> LocalVisionCardCandidate:
            return LocalVisionCardCandidate(
                backend="ollama",
                model="gemma3:12b",
                game="pokemon",
                title="Pikachu",
                aliases=("ピカチュウ",),
                card_number="166/198",
                rarity="ULTRARARE",
                set_code="spl",
                confidence=0.99,
            )

        def analyze_card_image_text_focus(
            self,
            image_path: Path,
            *,
            game_hint: str | None = None,
            title_hint: str | None = None,
        ) -> LocalVisionCardCandidate:
            return LocalVisionCardCandidate(
                backend="ollama",
                model="gemma3:12b",
                game="pokemon",
                title=None,
                aliases=(),
                card_number="201/165",
                rarity="SAR",
                set_code="sv2a",
                confidence=0.93,
            )

    service._local_vision_clients = (StubVisionClient(),)
    monkeypatch.setattr(
        service,
        "_run_footer_metadata_probe",
        lambda *args, **kwargs: (
            LocalVisionCardCandidate(
                backend="ollama",
                model="gemma3:12b",
                game="pokemon",
                title=None,
                aliases=(),
                card_number="201/165",
                rarity="SAR",
                set_code="sv2a",
                confidence=0.93,
            ),
            (),
        ),
    )

    image_path = tmp_path / "telegram-upload-charizard.jpg"
    image_path.write_bytes(b"fake-image")

    outcome = service.lookup_image(image_path, persist=False)

    assert outcome.status == "success"
    assert outcome.lookup_result is not None
    assert outcome.lookup_result.spec.title == CHARIZARD_JP
    assert outcome.lookup_result.spec.card_number == "201/165"
    assert outcome.parsed.title == CHARIZARD_JP
    assert calls[0].title == "201/165"
    assert any("footer-focused local vision metadata" in warning for warning in outcome.warnings)


def test_parse_image_escalates_to_second_local_vision_model_when_first_is_incomplete(tmp_path) -> None:
    image_path = tmp_path / "telegram-upload-charizard.jpg"
    image_path.write_bytes(b"fake-image")

    service = TcgImagePriceService(
        db_path=tmp_path / "monitor.sqlite3",
        tesseract_path="C:/missing/tesseract.exe",
    )

    class FastButIncompleteClient:
        descriptor = "ollama:qwen2.5vl:3b"

        def analyze_card_image(
            self,
            image_path: Path,
            *,
            game_hint: str | None = None,
            title_hint: str | None = None,
        ) -> LocalVisionCardCandidate:
            return LocalVisionCardCandidate(
                backend="ollama",
                model="qwen2.5vl:3b",
                game="pokemon",
                title=CHARIZARD_JP,
                aliases=("Charizard ex",),
                card_number=None,
                rarity=None,
                set_code=None,
                confidence=0.72,
            )

    class SlowerCompleteClient:
        descriptor = "ollama:gemma3:4b"

        def analyze_card_image(
            self,
            image_path: Path,
            *,
            game_hint: str | None = None,
            title_hint: str | None = None,
        ) -> LocalVisionCardCandidate:
            return LocalVisionCardCandidate(
                backend="ollama",
                model="gemma3:4b",
                game="pokemon",
                title=CHARIZARD_JP,
                aliases=("Charizard ex",),
                card_number="201/165",
                rarity="SAR",
                set_code="sv2a",
                confidence=0.91,
            )

    service._local_vision_clients = (FastButIncompleteClient(), SlowerCompleteClient())

    parsed = service.parse_image(image_path)

    assert parsed.status == "success"
    assert parsed.title == CHARIZARD_JP
    assert parsed.card_number == "201/165"
    assert parsed.rarity == "SAR"
    assert parsed.set_code == "sv2a"
    assert any("Applied local vision fallback via ollama:gemma3:4b." in warning for warning in parsed.warnings)


def test_parse_image_uses_second_local_vision_model_after_first_times_out(tmp_path, caplog) -> None:
    image_path = tmp_path / "telegram-upload-charizard.jpg"
    image_path.write_bytes(b"fake-image")

    service = TcgImagePriceService(
        db_path=tmp_path / "monitor.sqlite3",
        tesseract_path="C:/missing/tesseract.exe",
    )

    class TimeoutClient:
        descriptor = "ollama:qwen2.5vl:7b"

        def cooldown_remaining_seconds(self) -> int:
            return 900

        def is_temporarily_disabled(self) -> bool:
            return False

        def mark_timeout_cooldown(self) -> None:
            return None

        def analyze_card_image(
            self,
            image_path: Path,
            *,
            game_hint: str | None = None,
            title_hint: str | None = None,
        ) -> LocalVisionCardCandidate:
            raise LocalVisionTimeoutError(self.descriptor, timeout_seconds=45, detail="timed out")

    class CompleteClient:
        descriptor = "ollama:gemma3:12b"

        def analyze_card_image(
            self,
            image_path: Path,
            *,
            game_hint: str | None = None,
            title_hint: str | None = None,
        ) -> LocalVisionCardCandidate:
            return LocalVisionCardCandidate(
                backend="ollama",
                model="gemma3:12b",
                game="pokemon",
                title=CHARIZARD_JP,
                aliases=("Charizard ex",),
                card_number="201/165",
                rarity="SAR",
                set_code="sv2a",
                confidence=0.91,
            )

    service._local_vision_clients = (TimeoutClient(), CompleteClient())

    with caplog.at_level("WARNING"):
        parsed = service.parse_image(image_path)

    assert parsed.status == "success"
    assert parsed.title == CHARIZARD_JP
    assert parsed.card_number == "201/165"
    assert any("timed out and was put on cooldown" in warning for warning in parsed.warnings)
    assert "Local vision fallback timed out" in caplog.text


def test_parse_image_escalates_when_first_local_vision_candidate_looks_slab_derived(tmp_path) -> None:
    image_path = tmp_path / "telegram-upload-pikachu.jpg"
    image_path.write_bytes(b"fake-image")

    service = TcgImagePriceService(
        db_path=tmp_path / "monitor.sqlite3",
        tesseract_path="C:/missing/tesseract.exe",
    )

    class FastButSlabDerivedClient:
        descriptor = "ollama:qwen2.5vl:7b"

        def analyze_card_image(
            self,
            image_path: Path,
            *,
            game_hint: str | None = None,
            title_hint: str | None = None,
        ) -> LocalVisionCardCandidate:
            return LocalVisionCardCandidate(
                backend="ollama",
                model="qwen2.5vl:7b",
                game="pokemon",
                title="PIKACHU ex",
                aliases=(PIKACHU_EX_JP,),
                card_number="764",
                rarity="GEM MT",
                set_code="MC JP",
                confidence=0.98,
            )

    class SlowerCollectorNumberClient:
        descriptor = "ollama:gemma3:12b"

        def analyze_card_image(
            self,
            image_path: Path,
            *,
            game_hint: str | None = None,
            title_hint: str | None = None,
        ) -> LocalVisionCardCandidate:
            return LocalVisionCardCandidate(
                backend="ollama",
                model="gemma3:12b",
                game="pokemon",
                title=PIKACHU_EX_JP,
                aliases=("PIKACHU ex",),
                card_number="764/742",
                rarity=None,
                set_code="MC",
                confidence=0.93,
            )

    service._local_vision_clients = (FastButSlabDerivedClient(), SlowerCollectorNumberClient())

    parsed = service.parse_image(image_path)

    assert parsed.status == "success"
    assert parsed.title == PIKACHU_EX_JP
    assert parsed.card_number == "764/742"
    assert parsed.rarity is None
    assert parsed.set_code == "mc"
    assert any("Applied local vision fallback via ollama:gemma3:12b." in warning for warning in parsed.warnings)


def test_parse_image_uses_sealed_box_title_probe_when_box_title_is_weak(monkeypatch, tmp_path) -> None:
    image_path = tmp_path / "telegram-upload-box.jpg"
    image_path.write_bytes(b"fake-image")

    service = TcgImagePriceService(
        db_path=tmp_path / "monitor.sqlite3",
        tesseract_path=str(tmp_path / "tesseract.exe"),
    )
    service._tesseract_path = "tesseract"
    monkeypatch.setattr(
        service,
        "_extract_text",
        lambda _image_path: (
            "\n".join(
                [
                    "ポケモンカードゲーム スカーレット&バイオレット",
                    "強化拡張パック",
                    "mw Om",
                    "s29n",
                    "ランダム7枚入り",
                ]
            ),
            (),
        ),
    )

    class StubVisionClient:
        descriptor = "ollama:gemma3:12b"

        def analyze_card_image(
            self,
            image_path: Path,
            *,
            game_hint: str | None = None,
            title_hint: str | None = None,
        ) -> LocalVisionCardCandidate:
            return LocalVisionCardCandidate(
                backend="ollama",
                model="gemma3:12b",
                game="pokemon",
                title="mw Om",
                aliases=(),
                card_number=None,
                rarity=None,
                set_code="s29n",
                item_kind="sealed_box",
                confidence=0.62,
            )

        def analyze_sealed_box_title_focus(
            self,
            image_path: Path,
            *,
            game_hint: str | None = None,
            title_hint: str | None = None,
        ) -> LocalVisionCardCandidate:
            return LocalVisionCardCandidate(
                backend="ollama",
                model="gemma3:12b",
                game="pokemon",
                title="強化拡張パック ポケモンカード151",
                aliases=("ポケモンカード151",),
                card_number=None,
                rarity=None,
                set_code="sv2a",
                item_kind="sealed_box",
                confidence=0.95,
            )

    service._local_vision_clients = (StubVisionClient(),)

    parsed = service.parse_image(image_path, game_hint="pokemon")

    assert parsed.status == "success"
    assert parsed.item_kind == "sealed_box"
    assert parsed.title == "強化拡張パック ポケモンカード151"
    assert parsed.set_code == "sv2a"
    assert any("Ran sealed box title probe via ollama:gemma3:12b." in warning for warning in parsed.warnings)


def test_merge_local_vision_candidate_ignores_conflicting_metadata_when_titles_disagree() -> None:
    parsed = ParsedCardImage(
        status="success",
        game="pokemon",
        title="MEGA CHARIZARD X ex",
        aliases=("paypay charizard psa",),
        card_number=None,
        rarity="SAR",
        set_code="m2",
        raw_text="2025 POKEMONMZ JP\n#110\nMEGA CHARIZARD X ex\nSPECIAL AATAARE",
        extracted_lines=("2025 POKEMONMZ JP", "#110", "MEGA CHARIZARD X ex", "SPECIAL AATAARE"),
        warnings=(),
    )
    candidate = LocalVisionCardCandidate(
        backend="ollama",
        model="gemma3:12b",
        game="pokemon",
        title="リザードン",
        aliases=("Charizard",),
        card_number="165/165",
        rarity="H",
        set_code="sov",
        confidence=0.81,
    )

    merged = _merge_local_vision_candidate(parsed, candidate)

    assert merged.title == "MEGA CHARIZARD X ex"
    assert merged.card_number is None
    assert merged.rarity == "SAR"
    assert merged.set_code == "m2"
    assert any("ignored conflicting card metadata" in warning for warning in merged.warnings)


def test_merge_local_vision_candidate_rejects_weak_sealed_box_title() -> None:
    parsed = ParsedCardImage(
        status="unresolved",
        game="pokemon",
        title=None,
        aliases=(),
        card_number=None,
        rarity=None,
        set_code=None,
        raw_text="強化拡張パック\nmw Om\ns29n",
        extracted_lines=("強化拡張パック", "mw Om", "s29n"),
        item_kind="sealed_box",
        warnings=(),
    )
    candidate = LocalVisionCardCandidate(
        backend="ollama",
        model="gemma3:12b",
        game="pokemon",
        title="mw Om",
        aliases=(),
        card_number=None,
        rarity=None,
        set_code="s29n",
        item_kind="sealed_box",
        confidence=0.71,
    )

    merged = _merge_local_vision_candidate(parsed, candidate)

    assert merged.title is None
    assert merged.status == "unresolved"


def test_prepare_lookup_spec_uses_slab_number_to_recover_full_card_number(monkeypatch, tmp_path) -> None:
    class StubHotCardService:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def search_lookup_hints(self, spec: TcgCardSpec, *, limit: int = 5) -> tuple[TcgLookupHint, ...]:
            return (
                TcgLookupHint(
                    game="pokemon",
                    title="メガリザードンXex",
                    card_number="110/080",
                    rarity="SAR",
                    set_code="m2",
                    listing_count=12,
                    confidence=26.0,
                    references=(),
                ),
                TcgLookupHint(
                    game="pokemon",
                    title="オドリドリex",
                    card_number="111/080",
                    rarity="SAR",
                    set_code="m2",
                    listing_count=7,
                    confidence=26.0,
                    references=(),
                ),
            )

    monkeypatch.setattr("tcg_tracker.image_lookup.TcgHotCardService", StubHotCardService)

    service = TcgImagePriceService(
        db_path=tmp_path / "monitor.sqlite3",
    )
    parsed = ParsedCardImage(
        status="success",
        game="pokemon",
        title="MEGA CHARIZARD X ex",
        aliases=(),
        card_number=None,
        rarity="SAR",
        set_code="m2",
        raw_text="2025 POKEMON M2 JP\n#110\nMEGA CHARIZARD X ex",
        extracted_lines=("2025 POKEMON M2 JP", "#110", "MEGA CHARIZARD X ex"),
        warnings=(),
    )

    resolved_parsed, spec = service._prepare_lookup_spec(parsed)

    assert spec is not None
    assert spec.title == "メガリザードンXex"
    assert spec.card_number == "110/080"
    assert spec.rarity == "SAR"
    assert spec.set_code == "m2"
    assert resolved_parsed.card_number == "110/080"


def test_prepare_lookup_spec_repairs_wrong_card_number_using_slab_number(monkeypatch, tmp_path) -> None:
    class StubPriceService:
        def __init__(self, db_path):
            self.db_path = db_path

        def lookup(self, spec: TcgCardSpec, *, persist: bool = False) -> TcgLookupResult:
            if spec.card_number != "110/080":
                return TcgLookupResult(
                    spec=spec,
                    item=TrackedItem(item_id="x", item_type="card", category="tcg", title=spec.title),
                    offers=(),
                    fair_value=None,
                )
            offer = MarketOffer(
                source="cardrush_pokemon",
                listing_id="1",
                url="https://example.com/1",
                title="メガリザードンXex",
                price_jpy=79800,
                price_kind="ask",
                captured_at=datetime.now(timezone.utc),
                source_category="marketplace",
                attributes={"card_number": "110/080", "rarity": "SAR", "set_code": "m2"},
            )
            return TcgLookupResult(
                spec=spec,
                item=TrackedItem(item_id="x", item_type="card", category="tcg", title=spec.title),
                offers=(offer,),
                fair_value=None,
            )

    monkeypatch.setattr("tcg_tracker.image_lookup.TcgPriceService", StubPriceService)

    service = TcgImagePriceService(
        db_path=tmp_path / "monitor.sqlite3",
    )
    parsed = ParsedCardImage(
        status="success",
        game="pokemon",
        title="MEGA CHARIZARD X ex",
        aliases=(),
        card_number="470/080",
        rarity="AR",
        set_code="m2",
        raw_text="2025 POKEMON M2 JP\n#110\nMEGA CHARIZARD X ex",
        extracted_lines=("2025 POKEMON M2 JP", "#110", "MEGA CHARIZARD X ex"),
        warnings=(),
    )

    resolved_parsed, spec = service._prepare_lookup_spec(parsed)

    assert spec is not None
    assert spec.title == "メガリザードンXex"
    assert spec.card_number == "110/080"
    assert resolved_parsed.card_number == "110/080"

