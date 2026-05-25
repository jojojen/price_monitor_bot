"""Price-feedback service (Phase 1 of the self-evolving loop).

When the user disagrees with a lookup result they paste a reference URL.
This module:
1. Fetches the URL's HTML.
2. Asks qwen3:4b to extract the product price TWICE (temperature 0.7).
3. Computes self-consistency (|p1-p2| / mean) and cross-source consensus
   (|extracted_avg - original_fair_value| / fair_value).
4. Writes a `price_feedback_events` row.
5. Bumps `domain_trust` for (game, item_kind, domain) — success only when
   the cross-source consensus comes in within ±30 %.
6. On `high` confidence, saves an `extraction_examples` row that future
   `LlmListingExtractor.extract_price_for_feedback` calls will see as a
   few-shot anchor.

Even on fetch / extraction failures we still record the feedback event and
bump the domain (`consensus_fail_count`) — the human's intent matters and
silent drops would hide bad actors / unreachable sites.
"""

from __future__ import annotations

import gzip
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from urllib.parse import urlparse

from market_monitor.http import HttpClient
from market_monitor.llm_listing_extractor import (
    LlmListingExtractor,
    SingleProductExtraction,
)
from market_monitor.models import (
    DomainTrust,
    ExtractionExample,
    PriceFeedbackEvent,
    TrackedItem,
    utc_now,
)
from market_monitor.storage import MonitorDatabase

from .catalog import TcgCardSpec

logger = logging.getLogger(__name__)

# Thresholds for confidence classification (matches plan section "extraction_confidence")
_CONSISTENCY_HIGH = 10.0
_CONSISTENCY_MEDIUM = 30.0
_CONSENSUS_HIGH = 30.0
_CONSENSUS_MEDIUM = 50.0


@dataclass(frozen=True, slots=True)
class FeedbackOutcome:
    feedback_id: str
    domain: str
    extracted_avg_jpy: int | None
    consistency_pct: float | None
    consensus_pct: float | None
    confidence: str       # "high" / "medium" / "low"
    status: str           # "analyzed" / "fetch_failed" / "extraction_failed" / "low_consistency" / "low_consensus"
    domain_trust: DomainTrust
    summary_for_user: str


class TcgPriceFeedbackService:
    def __init__(
        self,
        *,
        database: MonitorDatabase,
        http_client: HttpClient | None = None,
        extractor: LlmListingExtractor | None = None,
        fetch_timeout_seconds: int = 20,
    ) -> None:
        self.db = database
        self.http = http_client or HttpClient()
        self.extractor = extractor or LlmListingExtractor()
        self._fetch_timeout = fetch_timeout_seconds

    def submit(
        self,
        *,
        item: TrackedItem,
        spec: TcgCardSpec,
        chat_id: str | int | None,
        original_fair_value_jpy: int | None,
        claimed_url: str,
    ) -> FeedbackOutcome:
        url = claimed_url.strip()
        domain = urlparse(url).netloc.lower()
        url_hash = MonitorDatabase._url_hash(url)
        chat_key = str(chat_id) if chat_id is not None else None
        created_at = utc_now()
        feedback_id = MonitorDatabase._feedback_id(
            item_id=item.item_id,
            chat_id=chat_key,
            url_hash=url_hash,
            created_at=created_at.isoformat(),
        )

        few_shot = self.db.recent_extraction_examples(
            game=spec.game, item_kind=spec.item_kind, limit=3,
        )

        # 1. Fetch
        html: str | None = None
        fetch_error: str | None = None
        try:
            html = self.http.get_text(url, timeout_seconds=self._fetch_timeout)
        except Exception as exc:
            fetch_error = str(exc)
            logger.warning("Feedback fetch failed url=%s error=%s", url, exc)

        # 2. Extract twice (only if fetch succeeded)
        extractions: list[SingleProductExtraction] = []
        if html is not None:
            for pass_idx in (1, 2):
                try:
                    ex = self.extractor.extract_price_for_feedback(
                        html,
                        base_url=url,
                        item_title_hint=spec.title,
                        game=spec.game,
                        item_kind=spec.item_kind,
                        few_shot_examples=few_shot,
                        temperature=0.7,
                    )
                except Exception as exc:
                    logger.warning("LLM extraction pass %d crashed: %s", pass_idx, exc)
                    ex = SingleProductExtraction(
                        price_jpy=None, title="", raw_response="",
                        error=f"crash: {exc}",
                    )
                extractions.append(ex)

        # 3. Compute consistency / consensus / confidence
        p1 = extractions[0].price_jpy if len(extractions) > 0 else None
        p2 = extractions[1].price_jpy if len(extractions) > 1 else None
        consistency_pct = _percent_delta(p1, p2)
        extracted_avg = _average(p1, p2)
        consensus_pct = _percent_delta(extracted_avg, original_fair_value_jpy)

        confidence: str
        if fetch_error is not None:
            confidence = "low"
            status = "fetch_failed"
        elif p1 is None and p2 is None:
            confidence = "low"
            status = "extraction_failed"
        elif consistency_pct is None or consistency_pct > _CONSISTENCY_MEDIUM:
            confidence = "low"
            status = "low_consistency"
        elif consensus_pct is None or consensus_pct > _CONSENSUS_MEDIUM:
            confidence = "low"
            status = "low_consensus"
        elif consistency_pct <= _CONSISTENCY_HIGH and consensus_pct <= _CONSENSUS_HIGH:
            confidence = "high"
            status = "analyzed"
        else:
            confidence = "medium"
            status = "analyzed"

        # 4. Build LLM notes
        llm_notes_json = json.dumps({
            "fetch_error": fetch_error,
            "pass1": {
                "price_jpy": p1, "title": extractions[0].title if extractions else "",
                "error": extractions[0].error if extractions else "no_fetch",
            },
            "pass2": {
                "price_jpy": p2, "title": extractions[1].title if len(extractions) > 1 else "",
                "error": extractions[1].error if len(extractions) > 1 else None,
            },
        }, ensure_ascii=False)

        # 5. Gzip HTML for future re-extraction
        raw_html_gz = None
        if html is not None:
            try:
                raw_html_gz = gzip.compress(html.encode("utf-8"), compresslevel=6)
            except Exception:
                raw_html_gz = None

        # 6. Persist feedback event
        event = PriceFeedbackEvent(
            feedback_id=feedback_id,
            chat_id=chat_key,
            item_id=item.item_id,
            game=spec.game,
            item_kind=spec.item_kind,
            original_fair_value_jpy=original_fair_value_jpy,
            claimed_url=url,
            claimed_domain=domain,
            url_hash=url_hash,
            extracted_price_jpy_pass1=p1,
            extracted_price_jpy_pass2=p2,
            consistency_pct=consistency_pct,
            consensus_pct=consensus_pct,
            extraction_confidence=confidence,
            raw_html_gzipped=raw_html_gz,
            llm_notes_json=llm_notes_json,
            status=status,
            created_at=created_at,
            updated_at=created_at,
        )
        self.db.save_price_feedback(event)

        # 7. Bump domain_trust (success = confidence != "low")
        trust = self.db.bump_domain_trust(
            game=spec.game,
            item_kind=spec.item_kind,
            domain=domain,
            success=(confidence != "low"),
        )

        # 8. Save high-confidence extraction as a future few-shot example
        if confidence == "high" and extracted_avg is not None:
            title_for_example = (
                extractions[0].title or extractions[1].title or spec.title
            )[:200]
            example = ExtractionExample(
                example_id=MonitorDatabase._example_id(feedback_id),
                game=spec.game,
                item_kind=spec.item_kind,
                domain=domain,
                title=title_for_example,
                price_jpy=int(extracted_avg),
                captured_from_feedback_id=feedback_id,
                captured_at=created_at,
            )
            self.db.save_extraction_example(example)

        summary = _build_user_summary(
            domain=domain,
            extracted_avg=extracted_avg,
            original_fair_value_jpy=original_fair_value_jpy,
            consistency_pct=consistency_pct,
            consensus_pct=consensus_pct,
            confidence=confidence,
            status=status,
            trust=trust,
        )
        return FeedbackOutcome(
            feedback_id=feedback_id,
            domain=domain,
            extracted_avg_jpy=int(extracted_avg) if extracted_avg is not None else None,
            consistency_pct=consistency_pct,
            consensus_pct=consensus_pct,
            confidence=confidence,
            status=status,
            domain_trust=trust,
            summary_for_user=summary,
        )


def _percent_delta(a: int | float | None, b: int | float | None) -> float | None:
    if a is None or b is None:
        return None
    if a == 0 and b == 0:
        return 0.0
    mean = (a + b) / 2.0
    if mean <= 0:
        return None
    return abs(a - b) / mean * 100.0


def _average(a: int | None, b: int | None) -> float | None:
    if a is None and b is None:
        return None
    if a is None:
        return float(b)  # type: ignore[arg-type]
    if b is None:
        return float(a)
    return (a + b) / 2.0


def _fmt_jpy(value: int | float | None) -> str:
    if value is None:
        return "—"
    return f"¥{int(value):,}"


def _build_user_summary(
    *,
    domain: str,
    extracted_avg: float | None,
    original_fair_value_jpy: int | None,
    consistency_pct: float | None,
    consensus_pct: float | None,
    confidence: str,
    status: str,
    trust: DomainTrust,
) -> str:
    confidence_zh = {"high": "高", "medium": "中", "low": "低"}.get(confidence, confidence)
    if status == "fetch_failed":
        head = "⚠️ 抓不到該 URL（網路或網站問題），已記錄你的指名意圖"
    elif status == "extraction_failed":
        head = "⚠️ LLM 在這個頁面找不到價格，已記錄你的指名意圖"
    elif status == "low_consistency":
        head = "⚠️ qwen 兩次抽取結果差太多，無法採信"
    elif status == "low_consensus":
        head = "⚠️ 抽出的價格跟原估差距太大，標為低信心保留審視"
    else:
        head = "✅ 已記錄"

    lines = [head]
    lines.append(f"網站：{domain}")
    if extracted_avg is not None:
        delta = (
            f"（vs 原估 {_fmt_jpy(original_fair_value_jpy)}, 差 {consensus_pct:.0f}%）"
            if consensus_pct is not None
            else ""
        )
        lines.append(f"抽取價：{_fmt_jpy(extracted_avg)} {delta}")
    if consistency_pct is not None:
        lines.append(f"qwen 兩次一致性：差 {consistency_pct:.1f}% 信心：{confidence_zh}")
    lines.append(
        f"🌱 {domain} 在 {trust.game}/{trust.item_kind} 的信任分數："
        f"{trust.bayes_accuracy_score:.2f}（{trust.vote_count} 票）"
    )
    return "\n".join(lines)
