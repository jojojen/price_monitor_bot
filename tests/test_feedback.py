from __future__ import annotations

from pathlib import Path

from market_monitor.llm_listing_extractor import SingleProductExtraction
from market_monitor.storage import MonitorDatabase
from tcg_tracker.catalog import TcgCardSpec
from tcg_tracker.feedback import TcgPriceFeedbackService


class _StubHttp:
    def __init__(self, html: str | None = None, raises: Exception | None = None) -> None:
        self.html = html
        self.raises = raises
        self.calls = 0

    def get_text(self, url: str, **kwargs):
        self.calls += 1
        if self.raises is not None:
            raise self.raises
        return self.html


class _StubExtractor:
    def __init__(self, prices_titles: list[tuple[int | None, str]]) -> None:
        self.prices_titles = prices_titles
        self.calls = 0
        self.last_kwargs: dict | None = None

    def extract_price_for_feedback(self, html, **kwargs):
        self.last_kwargs = kwargs
        idx = min(self.calls, len(self.prices_titles) - 1)
        price, title = self.prices_titles[idx]
        self.calls += 1
        return SingleProductExtraction(
            price_jpy=price, title=title, raw_response="", error=None,
        )


def _make_service(tmp_path: Path, *, http, extractor):
    db = MonitorDatabase(tmp_path / "fb.sqlite3")
    db.bootstrap()
    return db, TcgPriceFeedbackService(database=db, http_client=http, extractor=extractor)


def _make_spec_and_item(db):
    spec = TcgCardSpec(game="pokemon", title="MEGA アビスアイ", item_kind="sealed_box")
    item = spec.to_tracked_item()
    db.upsert_item(item)
    return spec, item


def test_high_confidence_path_bumps_trust_and_saves_example(tmp_path: Path) -> None:
    http = _StubHttp(html="<html>price ¥16,800</html>")
    extractor = _StubExtractor([(16500, "MEGA アビスアイBOX"), (17100, "MEGA アビスアイBOX")])
    db, svc = _make_service(tmp_path, http=http, extractor=extractor)
    spec, item = _make_spec_and_item(db)

    outcome = svc.submit(
        item=item, spec=spec, chat_id=12345,
        original_fair_value_jpy=16800,
        claimed_url="https://yuyu-tei.jp/sealed/abyss",
    )
    assert outcome.confidence == "high"
    assert outcome.status == "analyzed"
    # extracted_avg = (16500 + 17100) / 2 = 16800
    assert outcome.extracted_avg_jpy == 16800
    # |16500-17100| / 16800 ≈ 3.57%
    assert outcome.consistency_pct is not None and outcome.consistency_pct < 10
    # consensus matches exactly
    assert outcome.consensus_pct == 0.0
    # one extraction_example saved
    examples = db.recent_extraction_examples(game="pokemon", item_kind="sealed_box")
    assert len(examples) == 1
    assert examples[0].price_jpy == 16800
    # domain_trust bumped with success
    assert outcome.domain_trust.consensus_success_count == 1
    assert outcome.domain_trust.consensus_fail_count == 0


def test_low_consistency_does_not_promote(tmp_path: Path) -> None:
    """qwen gives wildly different prices: low_consistency status, no example."""
    extractor = _StubExtractor([(1000, "x"), (50000, "y")])  # 98% delta
    http = _StubHttp(html="<html>...</html>")
    db, svc = _make_service(tmp_path, http=http, extractor=extractor)
    spec, item = _make_spec_and_item(db)

    outcome = svc.submit(
        item=item, spec=spec, chat_id=None,
        original_fair_value_jpy=16800,
        claimed_url="https://shady.example/x",
    )
    assert outcome.confidence == "low"
    assert outcome.status == "low_consistency"
    examples = db.recent_extraction_examples(game="pokemon", item_kind="sealed_box")
    assert len(examples) == 0
    # domain_trust still got a vote (human intent recorded), but it counts as a failure
    assert outcome.domain_trust.consensus_success_count == 0
    assert outcome.domain_trust.consensus_fail_count == 1


def test_low_consensus_disagrees_with_fair_value(tmp_path: Path) -> None:
    """qwen extracts consistently but the price is way off from bot's estimate."""
    extractor = _StubExtractor([(2000, "shipping fee"), (2000, "shipping fee")])
    http = _StubHttp(html="<html>...</html>")
    db, svc = _make_service(tmp_path, http=http, extractor=extractor)
    spec, item = _make_spec_and_item(db)

    outcome = svc.submit(
        item=item, spec=spec, chat_id="42",
        original_fair_value_jpy=16800,
        claimed_url="https://bad.example/x",
    )
    # consistency is great (0%); consensus is terrible (~157%)
    assert outcome.confidence == "low"
    assert outcome.status == "low_consensus"
    # no example saved, vote counted as failure
    examples = db.recent_extraction_examples(game="pokemon", item_kind="sealed_box")
    assert len(examples) == 0
    assert outcome.domain_trust.consensus_fail_count == 1


def test_fetch_failure_still_records_event_and_bumps_vote(tmp_path: Path) -> None:
    http = _StubHttp(raises=RuntimeError("timeout"))
    extractor = _StubExtractor([(0, "")])  # never called
    db, svc = _make_service(tmp_path, http=http, extractor=extractor)
    spec, item = _make_spec_and_item(db)

    outcome = svc.submit(
        item=item, spec=spec, chat_id=None,
        original_fair_value_jpy=16800,
        claimed_url="https://timeout.example/x",
    )
    assert outcome.status == "fetch_failed"
    assert outcome.confidence == "low"
    assert outcome.extracted_avg_jpy is None
    # Vote IS still recorded
    assert outcome.domain_trust.vote_count == 1
    assert outcome.domain_trust.consensus_fail_count == 1
    # No extraction calls because we never got HTML
    assert extractor.calls == 0


def test_few_shot_examples_forwarded_to_extractor(tmp_path: Path) -> None:
    """Seed an extraction_example so the next submit() forwards it as few-shot."""
    extractor = _StubExtractor([(15000, "A"), (15500, "A")])
    http = _StubHttp(html="<html>...</html>")
    db, svc = _make_service(tmp_path, http=http, extractor=extractor)
    spec, item = _make_spec_and_item(db)

    # First submission seeds one high-confidence example
    svc.submit(
        item=item, spec=spec, chat_id="1",
        original_fair_value_jpy=15300,
        claimed_url="https://yuyu-tei.jp/a",
    )
    # Second submission should now see the previous example as few-shot
    svc.submit(
        item=item, spec=spec, chat_id="1",
        original_fair_value_jpy=15300,
        claimed_url="https://cardrush.jp/b",
    )
    assert extractor.last_kwargs is not None
    fewshot = extractor.last_kwargs.get("few_shot_examples") or ()
    assert any(e.domain == "yuyu-tei.jp" for e in fewshot)
