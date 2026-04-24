from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from market_monitor.models import FairValueEstimate, MarketOffer, TrackedItem
from tcg_tracker.catalog import TcgCardSpec
from tcg_tracker.hot_cards import HotCardBoard, HotCardEntry, HotCardReference
from tcg_tracker.image_lookup import ParsedCardImage, TcgImageLookupOutcome
from tcg_tracker.service import TcgLookupResult
from tests.image_lookup_case_fixtures import get_image_lookup_live_case

from price_monitor_bot.formatters import format_lookup_result_telegram
from price_monitor_bot.natural_language import TelegramNaturalLanguageIntent, fallback_route_telegram_natural_language
from price_monitor_bot.bot import (
    TelegramCommandProcessor,
    TelegramFileAttachment,
    TelegramLookupQuery,
    TelegramReputationQuery,
    TelegramReputationDelivery,
    build_processing_ack,
    format_liquidity_board,
    format_photo_lookup_result,
    handle_telegram_message,
    parse_lookup_command,
    parse_reputation_snapshot_command,
)


def _stub_board() -> HotCardBoard:
    return HotCardBoard(
        game="pokemon",
        label="Pokemon Liquidity Board",
        methodology="stub methodology",
        generated_at=datetime.now(timezone.utc),
        items=(
            HotCardEntry(
                game="pokemon",
                rank=1,
                title="Pikachu ex",
                price_jpy=99800,
                thumbnail_url="https://example.com/pikachu.jpg",
                card_number="132/106",
                rarity="SAR",
                set_code="sv08",
                listing_count=5,
                best_ask_jpy=99800,
                best_bid_jpy=80000,
                previous_bid_jpy=50000,
                bid_ask_ratio=0.8016,
                buy_support_score=90.08,
                momentum_boost_score=6.0,
                buy_signal_label="priceup",
                hot_score=88.2,
                attention_score=41.7,
                social_post_count=3,
                social_engagement_count=120,
                notes=("stub note",),
                is_graded=False,
                references=(HotCardReference(label="Ranking Source", url="https://example.com/rank"),),
            ),
        ),
    )


class FakeTelegramClient:
    def __init__(self, sample_path: Path | None = None) -> None:
        self.sample_path = sample_path
        self.sent_messages: list[str] = []
        self.sent_documents: list[tuple[str, str | None]] = []
        self.sent_photos: list[tuple[str, str | None]] = []

    def send_message(self, *, chat_id: str | int, text: str) -> dict[str, object]:
        self.sent_messages.append(text)
        return {"chat_id": str(chat_id), "text": text}

    def send_document(self, *, chat_id: str | int, document_path: Path, caption: str | None = None) -> dict[str, object]:
        self.sent_documents.append((document_path.name, caption))
        return {"chat_id": str(chat_id), "document": document_path.name, "caption": caption}

    def send_photo(self, *, chat_id: str | int, photo_path: Path, caption: str | None = None) -> dict[str, object]:
        self.sent_photos.append((photo_path.name, caption))
        return {"chat_id": str(chat_id), "photo": photo_path.name, "caption": caption}

    def get_file(self, *, file_id: str) -> dict[str, object]:
        assert self.sample_path is not None
        return {"file_path": self.sample_path.name, "file_id": file_id}

    def download_file(self, *, file_path: str) -> bytes:
        assert self.sample_path is not None
        assert file_path == self.sample_path.name
        return self.sample_path.read_bytes()


class StubNaturalLanguageRouter:
    def __init__(self, intent: TelegramNaturalLanguageIntent | None) -> None:
        self.intent = intent
        self.seen_texts: list[str] = []

    def route(self, text: str) -> TelegramNaturalLanguageIntent | None:
        self.seen_texts.append(text)
        return self.intent


def test_parse_lookup_command_supports_pipe_format() -> None:
    query = parse_lookup_command("pokemon | Pikachu ex | 132/106 | SAR | sv08")

    assert query == TelegramLookupQuery(
        game="pokemon",
        name="Pikachu ex",
        card_number="132/106",
        rarity="SAR",
        set_code="sv08",
    )


def test_parse_lookup_command_supports_simple_format() -> None:
    query = parse_lookup_command("ws Hatsune Miku")

    assert query == TelegramLookupQuery(
        game="ws",
        name="Hatsune Miku",
    )


def test_command_processor_restricts_unconfigured_chat() -> None:
    processor = TelegramCommandProcessor(
        allowed_chat_id="999",
        lookup_renderer=lambda query: query.name,
        board_loader=lambda: (_stub_board(),),
        catalog_renderer=lambda: "catalog",
    )

    assert processor.build_reply(chat_id="123", text="/ping") is None


def test_command_processor_handles_price_and_trend_aliases() -> None:
    processor = TelegramCommandProcessor(
        allowed_chat_id="123",
        lookup_renderer=lambda query: f"{query.game}:{query.name}:{query.card_number}",
        board_loader=lambda: (_stub_board(),),
        catalog_renderer=lambda: "catalog",
    )

    lookup_reply = processor.build_reply(chat_id="123", text="/price pokemon | Pikachu ex | 132/106")
    trend_reply = processor.build_reply(chat_id="123", text="/trend pokemon")
    hot_reply = processor.build_reply(chat_id="123", text="/hot pokemon 1")

    assert lookup_reply == "pokemon:Pikachu ex:132/106"
    assert "Pokemon Liquidity Board" in trend_reply
    assert "bid " in trend_reply and "80,000" in trend_reply
    assert "ask " in trend_reply and "99,800" in trend_reply
    assert "boost 6.00" in trend_reply
    assert "Pokemon Liquidity Board" in hot_reply
    assert "\n1. Pikachu ex\n" in hot_reply


def test_command_processor_help_lists_trend_and_scan_commands() -> None:
    processor = TelegramCommandProcessor(
        allowed_chat_id="123",
        lookup_renderer=lambda query: query.name,
        board_loader=lambda: (_stub_board(),),
        catalog_renderer=lambda: "catalog",
    )

    help_reply = processor.build_reply(chat_id="123", text="/help")

    assert "/trend pokemon" in help_reply
    assert "/price pokemon | Pikachu ex | 132/106 | SAR | sv08" in help_reply
    assert "/snapshot https://jp.mercari.com/item/m123456789" in help_reply
    assert "Send a photo with caption: /scan pokemon" in help_reply


def test_parse_reputation_snapshot_command_requires_url() -> None:
    query = parse_reputation_snapshot_command("https://jp.mercari.com/item/m123456789")

    assert query == TelegramReputationQuery(query_url="https://jp.mercari.com/item/m123456789")


def test_command_processor_handles_snapshot_command() -> None:
    processor = TelegramCommandProcessor(
        allowed_chat_id="123",
        lookup_renderer=lambda query: query.name,
        board_loader=lambda: (_stub_board(),),
        catalog_renderer=lambda: "catalog",
        reputation_renderer=lambda query: f"snapshot:{query.query_url}",
    )

    reply = processor.build_reply(chat_id="123", text="/snapshot https://jp.mercari.com/item/m123456789")

    assert reply == "snapshot:https://jp.mercari.com/item/m123456789"


def test_command_processor_handles_natural_language_lookup_via_router() -> None:
    router = StubNaturalLanguageRouter(
        TelegramNaturalLanguageIntent(
            intent="lookup_card",
            game="pokemon",
            name="Pikachu ex",
            card_number="132/106",
            rarity="SAR",
            set_code="sv08",
            confidence=0.98,
        )
    )
    processor = TelegramCommandProcessor(
        allowed_chat_id="123",
        lookup_renderer=lambda query: f"{query.game}:{query.name}:{query.card_number}:{query.rarity}:{query.set_code}",
        board_loader=lambda: (_stub_board(),),
        catalog_renderer=lambda: "catalog",
        natural_language_router=router,
    )

    reply = processor.build_reply(chat_id="123", text="幫我查 pokemon Pikachu ex 132/106 SAR sv08")

    assert reply == "pokemon:Pikachu ex:132/106:SAR:sv08"
    assert router.seen_texts == ["幫我查 pokemon Pikachu ex 132/106 SAR sv08"]


def test_command_processor_handles_natural_language_trend_via_router() -> None:
    router = StubNaturalLanguageRouter(
        TelegramNaturalLanguageIntent(
            intent="trend_board",
            game="pokemon",
            limit=3,
            confidence=0.91,
        )
    )
    processor = TelegramCommandProcessor(
        allowed_chat_id="123",
        lookup_renderer=lambda query: query.name,
        board_loader=lambda: (_stub_board(),),
        catalog_renderer=lambda: "catalog",
        natural_language_router=router,
    )

    reply = processor.build_reply(chat_id="123", text="pokemon 熱門前 3")

    assert "Pokemon Liquidity Board" in reply
    assert router.seen_texts == ["pokemon 熱門前 3"]


def test_command_processor_builds_ack_for_natural_language_trend() -> None:
    router = StubNaturalLanguageRouter(
        TelegramNaturalLanguageIntent(
            intent="trend_board",
            game="ws",
            limit=5,
            confidence=0.94,
        )
    )
    processor = TelegramCommandProcessor(
        allowed_chat_id="123",
        lookup_renderer=lambda query: query.name,
        board_loader=lambda: (
            HotCardBoard(
                game="ws",
                label="WS Liquidity Board",
                methodology="stub methodology",
                generated_at=datetime.now(timezone.utc),
                items=_stub_board().items,
            ),
        ),
        catalog_renderer=lambda: "catalog",
        natural_language_router=router,
    )

    plan = processor.build_reply_plan(chat_id="123", text="ws 熱門前 5")

    assert plan.ack == "已理解查詢內容，相當於 /trend ws 5，開始整理資料。"
    reply = plan.execute()
    assert reply is not None
    assert "WS Liquidity Board" in reply
    assert router.seen_texts == ["ws 熱門前 5"]


def test_build_processing_ack_for_heavy_actions() -> None:
    assert build_processing_ack(text="/price pokemon Pikachu ex") == "收到查價指令，開始處理。"
    assert build_processing_ack(text="/trend pokemon") == "收到趨勢榜查詢，開始整理資料。"
    assert build_processing_ack(text="/snapshot https://jp.mercari.com/item/m123456789") == (
        "收到信譽快照查詢，先檢查既有 proof，必要時建立新快照。"
    )
    assert build_processing_ack(has_photo=True) == "收到圖片，開始解析與查價。"
    assert build_processing_ack(text="/ping") is None


def test_handle_telegram_message_sends_ack_then_photo_result() -> None:
    sample_path = get_image_lookup_live_case("pokemon-pikachu-partial-s40").image_path
    client = FakeTelegramClient(sample_path=sample_path)
    processor = TelegramCommandProcessor(
        allowed_chat_id="123",
        lookup_renderer=lambda query: query.name,
        board_loader=lambda: (_stub_board(),),
        catalog_renderer=lambda: "catalog",
    )

    replies = handle_telegram_message(
        client=client,
        processor=processor,
        photo_renderer=lambda query: f"photo:{query.game_hint}:{query.title_hint}:{query.image_path.suffix}",
        message={
            "chat": {"id": "123"},
            "photo": [{"file_id": "photo-1", "file_size": 128}],
            "caption": "/scan pokemon Pikachu ex",
        },
    )

    assert replies == (
        "收到圖片，開始解析與查價。",
        "photo:pokemon:Pikachu ex:.jpg",
    )
    assert client.sent_messages == list(replies)


def test_handle_telegram_message_sends_ack_then_text_result() -> None:
    client = FakeTelegramClient()
    processor = TelegramCommandProcessor(
        allowed_chat_id="123",
        lookup_renderer=lambda query: f"{query.game}:{query.name}",
        board_loader=lambda: (_stub_board(),),
        catalog_renderer=lambda: "catalog",
    )

    replies = handle_telegram_message(
        client=client,
        processor=processor,
        photo_renderer=lambda query: "unused",
        message={
            "chat": {"id": "123"},
            "text": "/price pokemon Pikachu ex",
        },
    )

    assert replies == (
        "收到查價指令，開始處理。",
        "pokemon:Pikachu ex",
    )
    assert client.sent_messages == list(replies)


def test_handle_telegram_message_sends_snapshot_ack_then_result(tmp_path: Path) -> None:
    client = FakeTelegramClient()
    pdf_path = tmp_path / "proof_123.pdf"
    png_path = tmp_path / "proof_123.png"
    pdf_path.write_bytes(b"%PDF-1.4 stub")
    png_path.write_bytes(b"\x89PNG\r\n\x1a\nstub")
    processor = TelegramCommandProcessor(
        allowed_chat_id="123",
        lookup_renderer=lambda query: f"{query.game}:{query.name}",
        board_loader=lambda: (_stub_board(),),
        catalog_renderer=lambda: "catalog",
        reputation_renderer=lambda query: TelegramReputationDelivery(
            summary_text=f"snapshot:{query.query_url}",
            attachments=(
                TelegramFileAttachment(kind="document", path=pdf_path, caption="Reputation snapshot PDF"),
                TelegramFileAttachment(kind="photo", path=png_path, caption="Reputation snapshot preview"),
            ),
            cleanup_paths=(pdf_path, png_path),
        ),
    )

    replies = handle_telegram_message(
        client=client,
        processor=processor,
        photo_renderer=lambda query: "unused",
        message={
            "chat": {"id": "123"},
            "text": "/snapshot https://jp.mercari.com/item/m123456789",
        },
    )

    assert replies == (
        "收到信譽快照查詢，先檢查既有 proof，必要時建立新快照。",
        "snapshot:https://jp.mercari.com/item/m123456789",
    )
    assert client.sent_messages == list(replies)
    assert client.sent_documents == [("proof_123.pdf", "Reputation snapshot PDF")]
    assert client.sent_photos == [("proof_123.png", "Reputation snapshot preview")]
    assert not pdf_path.exists()
    assert not png_path.exists()


def test_handle_telegram_message_sends_natural_language_ack_then_result() -> None:
    client = FakeTelegramClient()
    router = StubNaturalLanguageRouter(
        TelegramNaturalLanguageIntent(
            intent="trend_board",
            game="ws",
            limit=5,
            confidence=0.94,
        )
    )
    processor = TelegramCommandProcessor(
        allowed_chat_id="123",
        lookup_renderer=lambda query: f"{query.game}:{query.name}",
        board_loader=lambda: (
            HotCardBoard(
                game="ws",
                label="WS Liquidity Board",
                methodology="stub methodology",
                generated_at=datetime.now(timezone.utc),
                items=_stub_board().items,
            ),
        ),
        catalog_renderer=lambda: "catalog",
        natural_language_router=router,
    )

    replies = handle_telegram_message(
        client=client,
        processor=processor,
        photo_renderer=lambda query: "unused",
        message={
            "chat": {"id": "123"},
            "text": "ws 熱門前 5",
        },
    )

    assert replies[0] == "已理解查詢內容，相當於 /trend ws 5，開始整理資料。"
    assert "WS Liquidity Board" in replies[1]
    assert client.sent_messages == list(replies)


def test_handle_telegram_message_sends_natural_language_ack_before_running_heavy_work() -> None:
    client = FakeTelegramClient()
    router = StubNaturalLanguageRouter(
        TelegramNaturalLanguageIntent(
            intent="trend_board",
            game="ws",
            limit=3,
            confidence=0.94,
        )
    )

    def board_loader() -> tuple[HotCardBoard, ...]:
        assert client.sent_messages == ["已理解查詢內容，相當於 /trend ws 3，開始整理資料。"]
        return (
            HotCardBoard(
                game="ws",
                label="WS Liquidity Board",
                methodology="stub methodology",
                generated_at=datetime.now(timezone.utc),
                items=_stub_board().items,
            ),
        )

    processor = TelegramCommandProcessor(
        allowed_chat_id="123",
        lookup_renderer=lambda query: f"{query.game}:{query.name}",
        board_loader=board_loader,
        catalog_renderer=lambda: "catalog",
        natural_language_router=router,
    )

    replies = handle_telegram_message(
        client=client,
        processor=processor,
        photo_renderer=lambda query: "unused",
        message={
            "chat": {"id": "123"},
            "text": "查ws熱門前3",
        },
    )

    assert replies[0] == "已理解查詢內容，相當於 /trend ws 3，開始整理資料。"
    assert "WS Liquidity Board" in replies[1]


def _mixed_grade_lookup_result() -> TcgLookupResult:
    now = datetime.now(timezone.utc)
    offers = (
        MarketOffer(source="cardrush_pokemon", listing_id="a1", url="https://cardrush.example/a1",
                    title="ピカチュウex 132/106", price_jpy=32800, price_kind="ask",
                    captured_at=now, source_category="marketplace",
                    attributes={"card_number": "132/106", "rarity": "SAR"}),
        MarketOffer(source="cardrush_pokemon", listing_id="a2", url="https://cardrush.example/a2",
                    title="ピカチュウex 132/106", price_jpy=28000, price_kind="bid",
                    captured_at=now, source_category="marketplace",
                    attributes={"card_number": "132/106", "rarity": "SAR"}),
        MarketOffer(source="magi", listing_id="m1", url="https://magi.example/m1",
                    title="ピカチュウex 132/106", price_jpy=30500, price_kind="market",
                    captured_at=now, source_category="marketplace",
                    attributes={"card_number": "132/106", "rarity": "SAR"}),
        MarketOffer(source="cardrush_pokemon", listing_id="p1", url="https://cardrush.example/p1",
                    title="【PSA10】ピカチュウex 132/106", price_jpy=98000, price_kind="ask",
                    captured_at=now, source_category="marketplace", condition="graded",
                    attributes={"card_number": "132/106", "rarity": "SAR", "is_graded": "1", "grade_label": "PSA10"}),
        MarketOffer(source="cardrush_pokemon", listing_id="p2", url="https://cardrush.example/p2",
                    title="【PSA10】ピカチュウex 132/106", price_jpy=88000, price_kind="bid",
                    captured_at=now, source_category="marketplace", condition="graded",
                    attributes={"card_number": "132/106", "rarity": "SAR", "is_graded": "1", "grade_label": "PSA10"}),
        MarketOffer(source="magi", listing_id="p3", url="https://magi.example/p3",
                    title="【PSA10】ピカチュウex 132/106", price_jpy=92000, price_kind="market",
                    captured_at=now, source_category="marketplace", condition="graded",
                    attributes={"card_number": "132/106", "rarity": "SAR", "is_graded": "1", "grade_label": "PSA10"}),
    )
    spec = TcgCardSpec(game="pokemon", title="Pikachu ex", card_number="132/106", rarity="SAR")
    item = TrackedItem(item_id="x", item_type="card", category="tcg", title="Pikachu ex")
    fv = FairValueEstimate(item_id="x", amount_jpy=32500, confidence=0.72, sample_count=3, reasoning=())
    return TcgLookupResult(spec=spec, item=item, offers=offers, fair_value=fv, notes=("sample note",))


def test_format_lookup_result_telegram_shows_raw_and_psa10_sections_without_mixed_total_price() -> None:
    text = format_lookup_result_telegram(_mixed_grade_lookup_result())

    assert "Raw" in text
    assert "PSA 10" in text

    raw_section = text.split("Raw", 1)[1].split("PSA 10", 1)[0]
    psa_section = text.split("PSA 10", 1)[1].split("Sources:", 1)[0]
    header_section = text.split("Raw", 1)[0]

    for section_name, section in (("raw", raw_section), ("psa10", psa_section)):
        assert "Fair Value:" in section, f"{section_name} section missing Fair Value"
        assert "Avg Price:" in section, f"{section_name} section missing Avg Price"
        assert "Best Bid:" in section, f"{section_name} section missing Best Bid"
        assert "Best Ask:" in section, f"{section_name} section missing Best Ask"
        assert "Best Market:" in section, f"{section_name} section missing Best Market"

    assert "Fair Value: ￥32,500" not in header_section
    assert "Fair Value: ￥32,800" in raw_section
    assert "￥31,100" in raw_section or "￥31,200" in raw_section or "￥31,650" in raw_section
    assert "￥28,000" in raw_section
    assert "￥32,800" in raw_section
    assert "￥30,500" in raw_section
    assert "Source URL: https://magi.example/m1" in raw_section
    assert "Fair Value: ￥98,000" in psa_section
    assert "￥95,000" in psa_section
    assert "￥88,000" in psa_section
    assert "￥98,000" in psa_section
    assert "￥92,000" in psa_section
    assert "Source URL: https://magi.example/p3" in psa_section
    assert "Offers:" not in text


def test_format_lookup_result_telegram_has_no_scan_or_note_noise() -> None:
    text = format_lookup_result_telegram(_mixed_grade_lookup_result())

    for forbidden in ("Image scan result", "Detected game", "Detected card", "Detected fields", "Note:", "sample note"):
        assert forbidden not in text, f"Telegram output should not contain {forbidden!r}: {text!r}"


def test_format_photo_lookup_result_is_identical_to_telegram_lookup_result() -> None:
    lookup_result = _mixed_grade_lookup_result()
    parsed = ParsedCardImage(
        status="success",
        game="pokemon",
        title="Pikachu ex",
        aliases=(),
        card_number="132/106",
        rarity="SAR",
        set_code=None,
        raw_text="",
        extracted_lines=(),
    )
    outcome = TcgImageLookupOutcome(
        status="success",
        parsed=parsed,
        lookup_result=lookup_result,
        warnings=("sample warning",),
    )

    text = format_photo_lookup_result(outcome)

    assert text == format_lookup_result_telegram(lookup_result)
    for forbidden in ("Image scan result", "Detected game", "Detected card", "Detected fields", "sample warning", "sample note"):
        assert forbidden not in text


def test_format_lookup_result_telegram_detects_psa10_from_title_with_spacing() -> None:
    now = datetime.now(timezone.utc)
    lookup_result = TcgLookupResult(
        spec=TcgCardSpec(game="pokemon", title="Pikachu ex", card_number="132/106", rarity="SAR"),
        item=TrackedItem(item_id="x", item_type="card", category="tcg", title="Pikachu ex"),
        offers=(
            MarketOffer(
                source="magi",
                listing_id="p1",
                url="https://magi.example/p1",
                title="PSA 10 ピカチュウex 132/106",
                price_jpy=92000,
                price_kind="market",
                captured_at=now,
                source_category="marketplace",
                condition="graded",
            ),
        ),
        fair_value=None,
    )

    text = format_lookup_result_telegram(lookup_result)

    assert "PSA 10" in text
    assert "其他鑑定卡" not in text


def test_format_liquidity_board_includes_reference_url() -> None:
    text = format_liquidity_board(_stub_board(), limit=1)

    assert "https://example.com/rank" in text
    assert "liq 88.20" in text
    assert "attn 41.70" in text
    assert "support 90.08" in text
    assert "buy-up" in text
    assert "stub methodology" not in text


def test_fallback_router_infers_ws_game_surrounded_by_cjk() -> None:
    intent = fallback_route_telegram_natural_language("查熱門ws前三")

    assert intent is not None
    assert intent.intent == "trend_board"
    assert intent.game == "ws"
    assert intent.limit == 3


def test_fallback_router_infers_ws_game_with_arabic_limit() -> None:
    intent = fallback_route_telegram_natural_language("ws熱門前5")

    assert intent is not None
    assert intent.intent == "trend_board"
    assert intent.game == "ws"
    assert intent.limit == 5


def test_fallback_router_infers_chinese_numeral_limit() -> None:
    intent = fallback_route_telegram_natural_language("pokemon 熱門前十")

    assert intent is not None
    assert intent.intent == "trend_board"
    assert intent.game == "pokemon"
    assert intent.limit == 10


def test_fallback_router_defaults_limit_when_no_number_given() -> None:
    intent = fallback_route_telegram_natural_language("ws 熱門")

    assert intent is not None
    assert intent.intent == "trend_board"
    assert intent.game == "ws"
    assert intent.limit == 5
