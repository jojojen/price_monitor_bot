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
from price_monitor_bot.natural_language import (
    TelegramNaturalLanguageIntent,
    _normalize_intent,
    fallback_route_telegram_natural_language,
)
from price_monitor_bot.bot import (
    PendingTelegramPhotoClarification,
    PendingTelegramTextClarification,
    PhotoLookupReply,
    TelegramCommandProcessor,
    TelegramFileAttachment,
    TelegramPhotoIntentAnalysis,
    TelegramPhotoIntentOption,
    TelegramLookupQuery,
    TelegramReputationQuery,
    TelegramReputationDelivery,
    TelegramTextIntentOption,
    build_processing_ack,
    format_liquidity_board,
    format_photo_lookup_result,
    handle_telegram_message,
    parse_lookup_command,
    parse_reputation_snapshot_command,
)

# Every call to `handle_telegram_message` now sends an immediate intake ack
# before kicking off the real processing pipeline; the assertions below
# include these to lock in that behaviour.
PHOTO_INTAKE_ACK = "已收到圖片，開始解讀使用者意圖"
TEXT_INTAKE_ACK = "已收到訊息，開始解讀使用者意圖"


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


def _ambiguous_photo_analysis(
    *,
    parsed_game: str | None = "pokemon",
    parsed_item_kind: str | None = "card",
    parsed_title: str | None = None,
) -> TelegramPhotoIntentAnalysis:
    return TelegramPhotoIntentAnalysis(
        options=(
            TelegramPhotoIntentOption(1, "pokemon_card_price", "要我查這張寶可夢卡市價嗎？", "/scan pokemon"),
            TelegramPhotoIntentOption(2, "yugioh_card_price", "要我查這張遊戲王卡市價嗎？", "/scan yugioh"),
            TelegramPhotoIntentOption(3, "pokemon_box_price", "要我查這個寶可夢卡盒市價嗎？", "/scan pokemon"),
        ),
        parsed_game=parsed_game,
        parsed_item_kind=parsed_item_kind,
        parsed_title=parsed_title,
    )


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


def test_parse_lookup_command_supports_yugioh_and_union_arena_aliases() -> None:
    ygo_query = parse_lookup_command("ygo | 青眼の白龍 | QCCP-JP001 | ウルトラ")
    ua_query = parse_lookup_command("ua 綾波レイ")

    assert ygo_query == TelegramLookupQuery(
        game="yugioh",
        name="青眼の白龍",
        card_number="QCCP-JP001",
        rarity="ウルトラ",
        set_code="qccp",
    )
    assert ua_query == TelegramLookupQuery(game="union_arena", name="綾波レイ")


def test_parse_lookup_command_recovers_card_metadata_from_simple_format() -> None:
    ua_query = parse_lookup_command("union_arena uapr/eva-1-071 綾波レイ")
    ua_card_query = parse_lookup_command("ua card uapr/eva-1-071 綾波レイ")
    pokemon_query = parse_lookup_command("pokemon リザードンex 201/165 SAR")

    assert ua_query == TelegramLookupQuery(
        game="union_arena",
        name="綾波レイ",
        card_number="UAPR/EVA-1-071",
        set_code="uapr",
    )
    assert ua_card_query == ua_query
    assert pokemon_query == TelegramLookupQuery(
        game="pokemon",
        name="リザードンex",
        card_number="201/165",
        rarity="SAR",
    )


def test_command_processor_restricts_unconfigured_chat() -> None:
    processor = TelegramCommandProcessor(
        allowed_chat_ids=frozenset({"999"}),
        lookup_renderer=lambda query: query.name,
        board_loader=lambda: (_stub_board(),),
        catalog_renderer=lambda: "catalog",
    )

    assert processor.build_reply(chat_id="123", text="/ping") is None


def test_command_processor_handles_price_and_trend_aliases() -> None:
    processor = TelegramCommandProcessor(
        allowed_chat_ids=frozenset({"123"}),
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
        allowed_chat_ids=frozenset({"123"}),
        lookup_renderer=lambda query: query.name,
        board_loader=lambda: (_stub_board(),),
        catalog_renderer=lambda: "catalog",
    )

    help_reply = processor.build_reply(chat_id="123", text="/help")

    assert "/trend pokemon" in help_reply
    assert "/price pokemon | Pikachu ex | 132/106 | SAR | sv08" in help_reply
    assert "/snapshot https://jp.mercari.com/item/m123456789" in help_reply
    assert "Send a photo with caption: /scan pokemon" in help_reply
    assert "/hunt status" in help_reply


def test_command_processor_handles_hunt_status() -> None:
    processor = TelegramCommandProcessor(
        allowed_chat_ids=frozenset({"123"}),
        lookup_renderer=lambda query: query.name,
        board_loader=lambda: (_stub_board(),),
        catalog_renderer=lambda: "catalog",
        opportunity_status_renderer=lambda: "targets: Umbreon",
    )

    assert processor.build_reply(chat_id="123", text="/hunt status") == "targets: Umbreon"


def test_parse_reputation_snapshot_command_requires_url() -> None:
    query = parse_reputation_snapshot_command("https://jp.mercari.com/item/m123456789")

    assert query == TelegramReputationQuery(query_url="https://jp.mercari.com/item/m123456789")


def test_command_processor_handles_snapshot_command() -> None:
    processor = TelegramCommandProcessor(
        allowed_chat_ids=frozenset({"123"}),
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
        allowed_chat_ids=frozenset({"123"}),
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
        allowed_chat_ids=frozenset({"123"}),
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
        allowed_chat_ids=frozenset({"123"}),
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
        allowed_chat_ids=frozenset({"123"}),
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
        PHOTO_INTAKE_ACK,
        "收到圖片，開始解析與查價。",
        "photo:pokemon:Pikachu ex:.jpg",
    )
    assert client.sent_messages == list(replies)


def test_handle_telegram_message_sends_ack_then_text_result() -> None:
    client = FakeTelegramClient()
    processor = TelegramCommandProcessor(
        allowed_chat_ids=frozenset({"123"}),
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
        TEXT_INTAKE_ACK,
        "收到查價指令，開始處理。",
        "pokemon:Pikachu ex",
    )
    assert client.sent_messages == list(replies)


def test_handle_telegram_message_ignores_generic_price_caption_as_title_hint() -> None:
    sample_path = get_image_lookup_live_case("pokemon-pikachu-partial-s40").image_path
    client = FakeTelegramClient(sample_path=sample_path)
    processor = TelegramCommandProcessor(
        allowed_chat_ids=frozenset({"123"}),
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
            "caption": "查這個box市價",
        },
    )

    assert replies == (
        PHOTO_INTAKE_ACK,
        build_processing_ack(has_photo=True),
        "photo:None:None:.jpg",
    )


def test_handle_telegram_message_clarifies_image_without_caption() -> None:
    sample_path = get_image_lookup_live_case("pokemon-pikachu-partial-s40").image_path
    client = FakeTelegramClient(sample_path=sample_path)
    processor = TelegramCommandProcessor(
        allowed_chat_ids=frozenset({"123"}),
        lookup_renderer=lambda query: query.name,
        board_loader=lambda: (_stub_board(),),
        catalog_renderer=lambda: "catalog",
        photo_intent_analyzer=lambda query: _ambiguous_photo_analysis(parsed_title="Pikachu ex"),
    )

    replies = handle_telegram_message(
        client=client,
        processor=processor,
        photo_renderer=lambda query: "unused",
        message={
            "chat": {"id": "123"},
            "photo": [{"file_id": "photo-1", "file_size": 128}],
        },
    )

    assert len(replies) == 2
    assert replies[0] == PHOTO_INTAKE_ACK
    assert "請回覆數字" in replies[1]
    assert "1. 要我查這張寶可夢卡市價嗎？" in replies[1]
    assert "4. 都不是，請回答：否，[您的意圖]" in replies[1]
    assert client.sent_messages == list(replies)


def test_handle_telegram_message_uses_explicit_price_caption_without_clarifying() -> None:
    sample_path = get_image_lookup_live_case("pokemon-pikachu-partial-s40").image_path
    client = FakeTelegramClient(sample_path=sample_path)

    def analyzer(query):
        raise AssertionError("photo intent analyzer should not run for explicit price captions")

    processor = TelegramCommandProcessor(
        allowed_chat_ids=frozenset({"123"}),
        lookup_renderer=lambda query: query.name,
        board_loader=lambda: (_stub_board(),),
        catalog_renderer=lambda: "catalog",
        photo_intent_analyzer=analyzer,
    )

    replies = handle_telegram_message(
        client=client,
        processor=processor,
        photo_renderer=lambda query: f"photo:{query.game_hint}:{query.title_hint}",
        message={
            "chat": {"id": "123"},
            "photo": [{"file_id": "photo-1", "file_size": 128}],
            "caption": "查這張寶可夢卡市價",
        },
    )

    assert replies == (
        PHOTO_INTAKE_ACK,
        build_processing_ack(has_photo=True),
        "photo:pokemon:None",
    )


def test_handle_telegram_message_runs_selected_photo_option_after_clarification() -> None:
    sample_path = get_image_lookup_live_case("pokemon-pikachu-partial-s40").image_path
    client = FakeTelegramClient(sample_path=sample_path)
    processor = TelegramCommandProcessor(
        allowed_chat_ids=frozenset({"123"}),
        lookup_renderer=lambda query: query.name,
        board_loader=lambda: (_stub_board(),),
        catalog_renderer=lambda: "catalog",
        photo_intent_analyzer=lambda query: _ambiguous_photo_analysis(),
    )

    handle_telegram_message(
        client=client,
        processor=processor,
        photo_renderer=lambda query: "unused",
        message={
            "chat": {"id": "123"},
            "photo": [{"file_id": "photo-1", "file_size": 128}],
        },
    )
    replies = handle_telegram_message(
        client=client,
        processor=processor,
        photo_renderer=lambda query: f"resolved:{query.caption}:{query.game_hint}",
        message={
            "chat": {"id": "123"},
            "text": "1",
        },
    )

    assert replies == (
        TEXT_INTAKE_ACK,
        "收到，我就照第 1 個方式處理。",
        "resolved:/scan pokemon:pokemon",
    )


def test_handle_telegram_message_supports_photo_override_text() -> None:
    sample_path = get_image_lookup_live_case("pokemon-pikachu-partial-s40").image_path
    client = FakeTelegramClient(sample_path=sample_path)
    processor = TelegramCommandProcessor(
        allowed_chat_ids=frozenset({"123"}),
        lookup_renderer=lambda query: query.name,
        board_loader=lambda: (_stub_board(),),
        catalog_renderer=lambda: "catalog",
        photo_intent_analyzer=lambda query: _ambiguous_photo_analysis(),
    )

    handle_telegram_message(
        client=client,
        processor=processor,
        photo_renderer=lambda query: "unused",
        message={
            "chat": {"id": "123"},
            "photo": [{"file_id": "photo-1", "file_size": 128}],
        },
    )
    replies = handle_telegram_message(
        client=client,
        processor=processor,
        photo_renderer=lambda query: f"resolved:{query.caption}:{query.game_hint}:{query.item_kind_hint}",
        message={
            "chat": {"id": "123"},
            "text": "否，查這張遊戲王卡市價",
        },
    )

    assert replies == (
        TEXT_INTAKE_ACK,
        "收到，我改照你補充的意思處理：查這張遊戲王卡市價",
        "resolved:/scan yugioh:yugioh:card",
    )


def test_handle_telegram_message_card_selection_overrides_box_guess() -> None:
    sample_path = get_image_lookup_live_case("pokemon-pikachu-partial-s40").image_path
    client = FakeTelegramClient(sample_path=sample_path)
    processor = TelegramCommandProcessor(
        allowed_chat_ids=frozenset({"123"}),
        lookup_renderer=lambda query: query.name,
        board_loader=lambda: (_stub_board(),),
        catalog_renderer=lambda: "catalog",
        photo_intent_analyzer=lambda query: _ambiguous_photo_analysis(parsed_item_kind="sealed_box"),
    )

    handle_telegram_message(
        client=client,
        processor=processor,
        photo_renderer=lambda query: "unused",
        message={"chat": {"id": "123"}, "photo": [{"file_id": "photo-1", "file_size": 128}]},
    )
    replies = handle_telegram_message(
        client=client,
        processor=processor,
        photo_renderer=lambda query: f"resolved:{query.caption}:{query.game_hint}:{query.item_kind_hint}",
        message={"chat": {"id": "123"}, "text": "2"},
    )

    assert replies == (
        TEXT_INTAKE_ACK,
        "收到，我就照第 2 個方式處理。",
        "resolved:/scan yugioh:yugioh:card",
    )


def test_handle_telegram_message_supports_photo_identify_override() -> None:
    sample_path = get_image_lookup_live_case("pokemon-pikachu-partial-s40").image_path
    client = FakeTelegramClient(sample_path=sample_path)
    processor = TelegramCommandProcessor(
        allowed_chat_ids=frozenset({"123"}),
        lookup_renderer=lambda query: query.name,
        board_loader=lambda: (_stub_board(),),
        catalog_renderer=lambda: "catalog",
        photo_intent_analyzer=lambda query: _ambiguous_photo_analysis(
            parsed_game="pokemon",
            parsed_item_kind="sealed_box",
            parsed_title="ポケモンカード151",
        ),
    )

    handle_telegram_message(
        client=client,
        processor=processor,
        photo_renderer=lambda query: "unused",
        message={
            "chat": {"id": "123"},
            "photo": [{"file_id": "photo-1", "file_size": 128}],
        },
    )
    replies = handle_telegram_message(
        client=client,
        processor=processor,
        photo_renderer=lambda query: "unused",
        message={
            "chat": {"id": "123"},
            "text": "否，這是什麼",
        },
    )

    assert replies == (TEXT_INTAKE_ACK, "我目前看起來比較像 寶可夢卡盒：ポケモンカード151")
    assert processor.get_pending_photo_clarification("123") is None


def test_handle_telegram_message_reminds_when_photo_clarification_reply_is_unrecognized() -> None:
    sample_path = get_image_lookup_live_case("pokemon-pikachu-partial-s40").image_path
    client = FakeTelegramClient(sample_path=sample_path)
    processor = TelegramCommandProcessor(
        allowed_chat_ids=frozenset({"123"}),
        lookup_renderer=lambda query: query.name,
        board_loader=lambda: (_stub_board(),),
        catalog_renderer=lambda: "catalog",
        photo_intent_analyzer=lambda query: _ambiguous_photo_analysis(),
    )

    handle_telegram_message(
        client=client,
        processor=processor,
        photo_renderer=lambda query: "unused",
        message={
            "chat": {"id": "123"},
            "photo": [{"file_id": "photo-1", "file_size": 128}],
        },
    )
    replies = handle_telegram_message(
        client=client,
        processor=processor,
        photo_renderer=lambda query: "unused",
        message={
            "chat": {"id": "123"},
            "text": "蛤",
        },
    )

    assert len(replies) == 2
    assert replies[0] == TEXT_INTAKE_ACK
    assert "我現在在等你確認這張圖要怎麼處理" in replies[1]
    assert "1. 要我查這張寶可夢卡市價嗎？" in replies[1]


def test_handle_telegram_message_requests_override_when_other_option_is_selected() -> None:
    sample_path = get_image_lookup_live_case("pokemon-pikachu-partial-s40").image_path
    client = FakeTelegramClient(sample_path=sample_path)
    processor = TelegramCommandProcessor(
        allowed_chat_ids=frozenset({"123"}),
        lookup_renderer=lambda query: query.name,
        board_loader=lambda: (_stub_board(),),
        catalog_renderer=lambda: "catalog",
        photo_intent_analyzer=lambda query: _ambiguous_photo_analysis(),
    )

    handle_telegram_message(
        client=client,
        processor=processor,
        photo_renderer=lambda query: "unused",
        message={
            "chat": {"id": "123"},
            "photo": [{"file_id": "photo-1", "file_size": 128}],
        },
    )
    replies = handle_telegram_message(
        client=client,
        processor=processor,
        photo_renderer=lambda query: "unused",
        message={
            "chat": {"id": "123"},
            "text": "4",
        },
    )

    assert replies == (TEXT_INTAKE_ACK, "好，請直接回答：否，[您的意圖]")


def test_pending_photo_clarification_expires(tmp_path: Path) -> None:
    sample_path = tmp_path / "pending.jpg"
    sample_path.write_bytes(b"stub")
    processor = TelegramCommandProcessor(
        allowed_chat_ids=frozenset({"123"}),
        lookup_renderer=lambda query: query.name,
        board_loader=lambda: (_stub_board(),),
        catalog_renderer=lambda: "catalog",
    )
    processor.set_pending_photo_clarification(
        PendingTelegramPhotoClarification(
            chat_id="123",
            image_path=sample_path,
            caption=None,
            file_id="photo-1",
            options=_ambiguous_photo_analysis().options,
            created_at=0.0,
        )
    )
    assert processor.get_pending_photo_clarification("123") is None


def test_handle_telegram_message_sends_snapshot_ack_then_result(tmp_path: Path) -> None:
    client = FakeTelegramClient()
    pdf_path = tmp_path / "proof_123.pdf"
    png_path = tmp_path / "proof_123.png"
    pdf_path.write_bytes(b"%PDF-1.4 stub")
    png_path.write_bytes(b"\x89PNG\r\n\x1a\nstub")
    processor = TelegramCommandProcessor(
        allowed_chat_ids=frozenset({"123"}),
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
        TEXT_INTAKE_ACK,
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
        allowed_chat_ids=frozenset({"123"}),
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

    assert replies[0] == TEXT_INTAKE_ACK
    assert replies[1] == "已理解查詢內容，相當於 /trend ws 5，開始整理資料。"
    assert "WS Liquidity Board" in replies[2]
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
        assert client.sent_messages == [TEXT_INTAKE_ACK, "已理解查詢內容，相當於 /trend ws 3，開始整理資料。"]
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
        allowed_chat_ids=frozenset({"123"}),
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

    assert replies[0] == TEXT_INTAKE_ACK
    assert replies[1] == "已理解查詢內容，相當於 /trend ws 3，開始整理資料。"
    assert "WS Liquidity Board" in replies[2]


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


def test_format_lookup_result_telegram_supports_sealed_box_products() -> None:
    now = datetime.now(timezone.utc)
    lookup_result = TcgLookupResult(
        spec=TcgCardSpec(game="pokemon", title="強化拡張パック ポケモンカード151", item_kind="sealed_box", set_code="sv2a"),
        item=TrackedItem(item_id="box151", item_type="tcg_sealed_box", category="tcg", title="強化拡張パック ポケモンカード151"),
        offers=(
            MarketOffer(
                source="cardrush_pokemon",
                listing_id="c1",
                url="https://cardrush.example/box151",
                title="強化拡張パック ポケモンカード151 未開封BOX",
                price_jpy=70800,
                price_kind="ask",
                captured_at=now,
                source_category="specialty_store",
                attributes={"product_kind": "sealed_box", "set_code": "sv2a"},
            ),
            MarketOffer(
                source="magi",
                listing_id="m1",
                url="https://magi.example/box151",
                title="強化拡張パック ポケモンカード151 未開封BOX",
                price_jpy=70000,
                price_kind="market",
                captured_at=now,
                source_category="marketplace",
                attributes={"product_kind": "sealed_box", "set_code": "sv2a"},
            ),
        ),
        fair_value=FairValueEstimate(item_id="box151", amount_jpy=70400, confidence=0.78, sample_count=2, reasoning=()),
    )

    text = format_lookup_result_telegram(lookup_result)

    assert "[pokemon sealed box] 強化拡張パック ポケモンカード151" in text
    assert "Raw" not in text
    assert "Fair Value:" in text
    assert "Best Ask:" in text
    assert "Best Market:" in text
    assert "Source URL: https://magi.example/box151" in text


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


def _llm_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "intent": "lookup_card",
        "game": None,
        "name": None,
        "card_number": None,
        "rarity": None,
        "set_code": None,
        "limit": None,
        "confidence": 0.6,
        "watch_query": None,
        "watch_price_threshold": None,
        "watch_id": None,
        "query_url": None,
        "sns_handle": None,
        "sns_keyword": None,
        "sns_buzz_query": None,
    }
    payload.update(overrides)
    return payload


def test_normalize_intent_recovers_union_arena_card_number_from_name() -> None:
    intent = _normalize_intent(
        _llm_payload(game="union_arena", name="uapr/eva-1-071 綾波レイ")
    )

    assert intent.intent == "lookup_card"
    assert intent.name == "綾波レイ"
    assert intent.card_number == "UAPR/EVA-1-071"
    assert intent.set_code == "uapr"


def test_normalize_intent_recovers_pokemon_card_number_and_rarity_from_name() -> None:
    intent = _normalize_intent(
        _llm_payload(game="pokemon", name="リザードンex 201/165 SAR")
    )

    assert intent.name == "リザードンex"
    assert intent.card_number == "201/165"
    assert intent.rarity == "SAR"


def test_normalize_intent_leaves_already_structured_fields_alone() -> None:
    intent = _normalize_intent(
        _llm_payload(
            game="union_arena",
            name="綾波レイ",
            card_number="UAPR/EVA-1-71",
            rarity="SR",
            set_code="uapr",
        )
    )

    assert intent.name == "綾波レイ"
    assert intent.card_number == "UAPR/EVA-1-71"
    assert intent.rarity == "SR"
    assert intent.set_code == "uapr"


def test_fallback_router_preserves_union_arena_card_number_in_name() -> None:
    intent = fallback_route_telegram_natural_language("查 UA卡 uapr/eva-1-071 綾波レイ 價格")

    assert intent is not None
    assert intent.intent == "lookup_card"
    assert intent.game == "union_arena"
    assert intent.card_number == "UAPR/EVA-1-071"
    assert intent.name == "綾波レイ"
    # `ua` must no longer be stripped out of the card-number prefix
    assert "pr/eva" not in intent.name.lower()


# ─── Text intent clarification (mirrors the photo flow) ───────────────────────


def _make_clarifying_processor(
    router_intent: TelegramNaturalLanguageIntent | None,
    *,
    research_renderer=lambda q: f"research:{q.query}",
    lookup_renderer=lambda q: f"lookup:{q.game}:{q.name}",
) -> TelegramCommandProcessor:
    return TelegramCommandProcessor(
        allowed_chat_ids=frozenset({"123"}),
        lookup_renderer=lookup_renderer,
        board_loader=lambda: (_stub_board(),),
        catalog_renderer=lambda: "catalog",
        research_renderer=research_renderer,
        natural_language_router=StubNaturalLanguageRouter(router_intent),
    )


def test_handle_telegram_message_clarifies_when_intent_is_unknown() -> None:
    # Stub returns unknown — _route_natural_language filters that out and the
    # fallback rules also return None for this neutral text, so the processor
    # must fall back to the clarification flow rather than the generic
    # "Unknown command" reply.
    client = FakeTelegramClient()
    processor = _make_clarifying_processor(
        TelegramNaturalLanguageIntent(intent="unknown", confidence=0.1),
    )

    replies = handle_telegram_message(
        client=client,
        processor=processor,
        photo_renderer=lambda query: "unused",
        message={"chat": {"id": "123"}, "text": "今天心情很差"},
    )

    assert len(replies) == 2
    assert replies[0] == TEXT_INTAKE_ACK
    assert "請回覆數字" in replies[1]
    # Falls back to /search + /help options when nothing else fits.
    assert "上網搜尋" in replies[1]
    assert "/help" in replies[1]
    assert "都不是，請回答：否，[您的意圖]" in replies[1]
    assert processor.get_pending_text_clarification("123") is not None


def test_handle_telegram_message_clarifies_when_intent_confidence_is_low() -> None:
    # Confidence below the 0.55 threshold should trigger clarification even
    # when the LLM has a guess — exactly the "don't guess silently" rule that
    # already applies on the photo side.
    client = FakeTelegramClient()
    low_conf_intent = TelegramNaturalLanguageIntent(
        intent="web_research",
        research_query="why are pokemon cards popular",
        confidence=0.4,
    )
    processor = _make_clarifying_processor(low_conf_intent)

    replies = handle_telegram_message(
        client=client,
        processor=processor,
        photo_renderer=lambda query: "unused",
        message={"chat": {"id": "123"}, "text": "why are pokemon cards popular"},
    )

    assert len(replies) == 2
    assert replies[0] == TEXT_INTAKE_ACK
    reply = replies[1]
    assert "請回覆數字" in reply
    # The LLM's top guess shows up first.
    assert "1. 上網搜尋" in reply
    # Game keyword "pokemon" added a /price alternative.
    assert "/price" in reply
    pending = processor.get_pending_text_clarification("123")
    assert pending is not None
    assert pending.original_text == "why are pokemon cards popular"
    assert pending.options[0].intent.intent == "web_research"


def test_handle_telegram_message_runs_selected_text_option_after_clarification() -> None:
    client = FakeTelegramClient()
    research_calls: list[str] = []

    def research_renderer(query):
        research_calls.append(query.query)
        return f"web:{query.query}"

    processor = _make_clarifying_processor(
        TelegramNaturalLanguageIntent(intent="unknown", confidence=0.1),
        research_renderer=research_renderer,
    )

    # Step 1: trigger the clarification options.
    handle_telegram_message(
        client=client,
        processor=processor,
        photo_renderer=lambda query: "unused",
        message={"chat": {"id": "123"}, "text": "ありがとう"},
    )
    pending = processor.get_pending_text_clarification("123")
    assert pending is not None
    # Find which option is web_research so we exercise the right number.
    web_option = next(
        opt for opt in pending.options if opt.intent.intent == "web_research"
    )

    # Step 2: reply with that option number.
    replies = handle_telegram_message(
        client=client,
        processor=processor,
        photo_renderer=lambda query: "unused",
        message={"chat": {"id": "123"}, "text": str(web_option.option_number)},
    )

    assert any(
        reply.startswith(f"收到，我就照第 {web_option.option_number} 個方式處理。")
        for reply in replies
    )
    assert research_calls == ["ありがとう"]
    assert "web:ありがとう" in replies[-1]
    # Pending state must be consumed after a successful selection.
    assert processor.get_pending_text_clarification("123") is None


def test_handle_telegram_message_text_clarification_other_option_requests_override() -> None:
    client = FakeTelegramClient()
    processor = _make_clarifying_processor(
        TelegramNaturalLanguageIntent(intent="unknown", confidence=0.1),
    )

    handle_telegram_message(
        client=client,
        processor=processor,
        photo_renderer=lambda query: "unused",
        message={"chat": {"id": "123"}, "text": "在嗎"},
    )
    pending = processor.get_pending_text_clarification("123")
    assert pending is not None
    sentinel = len(pending.options) + 1

    replies = handle_telegram_message(
        client=client,
        processor=processor,
        photo_renderer=lambda query: "unused",
        message={"chat": {"id": "123"}, "text": str(sentinel)},
    )

    assert replies == (TEXT_INTAKE_ACK, "好，請直接回答：否，[您的意圖]")
    # Pending state remains so the follow-up "否，..." can be matched.
    assert processor.get_pending_text_clarification("123") is not None


def test_handle_telegram_message_text_clarification_override_reroutes_through_router() -> None:
    client = FakeTelegramClient()
    captured: list[TelegramLookupQuery] = []

    def lookup_renderer(query):
        captured.append(query)
        return f"lookup:{query.game}:{query.name}"

    # The first message is unknown; the override text "查 pokemon Pikachu ex"
    # then comes back through the router and gets a confident lookup intent.
    router = StubNaturalLanguageRouter(
        TelegramNaturalLanguageIntent(intent="unknown", confidence=0.1)
    )
    processor = TelegramCommandProcessor(
        allowed_chat_ids=frozenset({"123"}),
        lookup_renderer=lookup_renderer,
        board_loader=lambda: (_stub_board(),),
        catalog_renderer=lambda: "catalog",
        natural_language_router=router,
    )

    handle_telegram_message(
        client=client,
        processor=processor,
        photo_renderer=lambda query: "unused",
        message={"chat": {"id": "123"}, "text": "蛤"},
    )
    assert processor.get_pending_text_clarification("123") is not None

    # Re-target the stub so the override path gets a confident answer.
    router.intent = TelegramNaturalLanguageIntent(
        intent="lookup_card",
        game="pokemon",
        name="Pikachu ex",
        confidence=0.92,
    )
    replies = handle_telegram_message(
        client=client,
        processor=processor,
        photo_renderer=lambda query: "unused",
        message={"chat": {"id": "123"}, "text": "否，查 pokemon Pikachu ex"},
    )

    assert any("收到，我改照你補充的意思處理：查 pokemon Pikachu ex" in r for r in replies)
    assert any(r.endswith("lookup:pokemon:Pikachu ex") for r in replies)
    assert captured and captured[0].game == "pokemon" and captured[0].name == "Pikachu ex"
    # Pending state was popped and not re-established because the override was confident.
    assert processor.get_pending_text_clarification("123") is None


# ─── Per-intent fallback router behaviour (locks in the slim from this change) ─


CANONICAL_FALLBACK_PHRASES: tuple[tuple[str, str], ...] = (
    # trend_board — must require an explicit ranking word, never plain "趨勢".
    ("pokemon 熱門前 5", "trend_board"),
    ("ws trending top 3", "trend_board"),
    ("幫我看 pokemon liquidity", "trend_board"),
    ("遊戲王最近熱門排行", "trend_board"),
    # web_research — only canonical question phrasings stay reliable in fallback.
    ("為什麼Pokemon卡這麼貴", "web_research"),
    ("why are pokemon cards popular", "web_research"),
    # watch / list / remove / update — touched as little as possible.
    ("追蹤 初音ミク SSP 5萬以下", "add_watch"),
    ("我的追蹤清單", "list_watches"),
    ("取消追蹤 abc12345", "remove_watch"),
    ("把 abc12345 改成 4萬", "update_watch_price"),
    # reputation, sns — untouched lists, still work.
    ("信用 https://jp.mercari.com/user/profile/123", "reputation_snapshot"),
    ("追蹤 @elonmusk", "sns_add_account"),
    ("snslist", "sns_list"),
    ("取消追蹤 @elonmusk", "sns_delete"),
    # sns_buzz — slimmed; canonical phrasings keep working, generic
    # "什麼熱門" no longer over-eats (asserted separately below).
    ("snsbuzz amd", "sns_buzz"),
    ("整理 amd 最近的熱門討論", "sns_buzz"),
    ("buzz on amd", "sns_buzz"),
    # status / tools — only canonical roots kept (狀態 / health / 工具 / catalog / 功能清單).
    ("目前狀態", "status"),
    ("服務 health", "status"),
    ("tools", "tools"),
    ("catalog", "tools"),
    ("功能清單", "tools"),
    # opportunity_remove — canonical phrasings.
    ("移除機會目標 2", "opportunity_remove"),
    ("remove target 3", "opportunity_remove"),
    ("I am not interested in Umbreon ex SAR anymore", "opportunity_remove"),
)


def test_fallback_router_canonical_phrases_keep_routing() -> None:
    for phrase, expected_intent in CANONICAL_FALLBACK_PHRASES:
        result = fallback_route_telegram_natural_language(phrase)
        assert result is not None, f"phrase={phrase!r} unexpectedly returned None"
        assert result.intent == expected_intent, (
            f"phrase={phrase!r} routed to {result.intent}, expected {expected_intent}"
        )


def test_fallback_router_does_not_route_bare_trend_keyword_to_trend_board() -> None:
    # "趨勢" was removed from _TREND_KEYWORDS because in Chinese it usually
    # means "direction", not "ranking". A bare "最近趨勢" should fall through
    # to the LLM (or to clarification options) rather than fire a confident
    # trend_board fallback.
    assert fallback_route_telegram_natural_language("最近趨勢") is None


def test_fallback_router_no_longer_overeats_with_sns_buzz() -> None:
    # Before the slim "遊戲王最近熱門排行" matched _SNS_BUZZ_KEYWORDS via
    # "什麼熱門"/"最近熱門" and routed to sns_buzz despite the clear ranking
    # intent. After dropping those entries from _SNS_BUZZ_KEYWORDS it goes
    # to trend_board as the user would expect.
    result = fallback_route_telegram_natural_language("遊戲王最近熱門排行")
    assert result is not None
    assert result.intent == "trend_board"
    assert result.game == "yugioh"


def test_handle_photo_message_installs_clarification_when_renderer_returns_unresolved() -> None:
    # End-to-end check that when the photo renderer returns a
    # PhotoLookupReply with a pending_clarification (the
    # unresolved/rejected_sanity path), the dispatcher both replies with the
    # clarification text AND installs the pending photo state so the user's
    # next text message can use the "否，[您的意圖]" override flow.
    sample_path = get_image_lookup_live_case("pokemon-pikachu-partial-s40").image_path
    client = FakeTelegramClient(sample_path=sample_path)
    processor = TelegramCommandProcessor(
        allowed_chat_ids=frozenset({"123"}),
        lookup_renderer=lambda query: query.name,
        board_loader=lambda: (_stub_board(),),
        catalog_renderer=lambda: "catalog",
        photo_intent_analyzer=lambda query: _ambiguous_photo_analysis(),
    )

    def stub_photo_renderer(query):
        pending = PendingTelegramPhotoClarification(
            chat_id=str(query.chat_id),
            image_path=query.image_path,
            caption=query.caption,
            file_id=query.file_id,
            options=(),
            parsed_game="pokemon",
            parsed_item_kind="card",
            parsed_title=None,
        )
        return PhotoLookupReply(
            text="我看不太清楚這張卡。可以直接告訴我卡片名稱嗎？",
            pending_clarification=pending,
        )

    replies = handle_telegram_message(
        client=client,
        processor=processor,
        photo_renderer=stub_photo_renderer,
        message={
            "chat": {"id": "123"},
            "photo": [{"file_id": "photo-1", "file_size": 128}],
            "caption": "/scan pokemon",  # would normally trigger direct lookup
        },
    )

    assert any("我看不太清楚這張卡" in reply for reply in replies)
    # The pending photo state must be installed so a follow-up "否，..."
    # message reaches the existing override flow.
    pending = processor.get_pending_photo_clarification("123")
    assert pending is not None
    assert pending.parsed_game == "pokemon"
