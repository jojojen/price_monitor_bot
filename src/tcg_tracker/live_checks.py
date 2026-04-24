from __future__ import annotations

from pathlib import Path

from .examples import EXAMPLE_CARDS
from .service import TcgPriceService


def run_live_checks(db_path: str | Path = "data/live-checks.sqlite3") -> int:
    service = TcgPriceService(db_path=db_path)
    failures: list[str] = []

    for example in EXAMPLE_CARDS:
        result = service.lookup(example.spec)
        if not result.offers:
            failures.append(f"{example.spec.title}: no offers returned")
            continue

        top_offer = result.offers[0]
        actual_number = top_offer.attributes.get("card_number", "")
        actual_rarity = top_offer.attributes.get("rarity", "")
        if actual_number != example.expected_card_number:
            failures.append(
                f"{example.spec.title}: expected card number {example.expected_card_number}, got {actual_number}"
            )
        if actual_rarity != example.expected_rarity:
            failures.append(f"{example.spec.title}: expected rarity {example.expected_rarity}, got {actual_rarity}")

        ask_matches = [
            offer for offer in result.offers if offer.price_kind == "ask" and offer.attributes.get("card_number") == actual_number
        ]
        bid_matches = [
            offer for offer in result.offers if offer.price_kind == "bid" and offer.attributes.get("card_number") == actual_number
        ]
        if not ask_matches:
            failures.append(f"{example.spec.title}: ask-side match missing")
        if not bid_matches:
            failures.append(f"{example.spec.title}: bid-side match missing")

        summary = result.fair_value.amount_jpy if result.fair_value else "n/a"
        print(
            f"[OK] {example.spec.game} | {example.spec.title} | top={actual_number} {actual_rarity} | fair={summary}"
        )

    if failures:
        for failure in failures:
            print(f"[FAIL] {failure}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(run_live_checks())
