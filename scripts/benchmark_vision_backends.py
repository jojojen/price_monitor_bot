#!/usr/bin/env python3
"""Benchmark local vision LLM backends on the sealed-box title probe.

Usage:
    .venv/bin/python scripts/benchmark_vision_backends.py \
        --models qwen2.5vl:7b,gemma3:12b \
        --endpoint http://127.0.0.1:11434

Reads the fixture set under
``tests/fixtures/vision_backend_benchmark/<slug>/`` (one directory per
sample, each containing ``image.webp`` and ``expected.json``), runs each
configured backend against every sample, and reports a per-backend
PASS / PARTIAL / FAIL summary.

A case PASSes when the backend's returned title contains any token in
``expected.title_contains_any`` AND the title clears the project's
``_sealed_box_title_looks_usable`` filter (so a sealed-box read that
would be dropped downstream is not credited as success). A case is
PARTIAL when the title contains the expected token but the filter
rejects it (e.g. the model returned only "MEGA" — the right family but
unusable as a set identifier). Anything else is FAIL.

Add new samples by dropping a new directory into the fixture root; this
script auto-discovers them.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tcg_tracker.image_lookup import _sealed_box_title_looks_usable  # noqa: E402
from tcg_tracker.local_vision import OllamaLocalVisionClient  # noqa: E402

BENCHMARK_ROOT = REPO_ROOT / "tests" / "fixtures" / "vision_backend_benchmark"


def discover_cases() -> list[dict]:
    cases: list[dict] = []
    for case_dir in sorted(BENCHMARK_ROOT.iterdir()):
        if not case_dir.is_dir():
            continue
        expected_path = case_dir / "expected.json"
        if not expected_path.exists():
            continue
        payload = json.loads(expected_path.read_text(encoding="utf-8"))
        image_candidates = sorted(
            path
            for path in case_dir.iterdir()
            if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
        )
        if len(image_candidates) != 1:
            raise SystemExit(
                f"Expected exactly one image file in {case_dir}; found {len(image_candidates)}."
            )
        cases.append(
            {
                "slug": case_dir.name,
                "image_path": image_candidates[0],
                "tokens": payload.get("title_contains_any") or [],
                "canonical": payload.get("set_name_canonical"),
                "game": payload.get("game") or "pokemon",
            }
        )
    return cases


def score(title: str | None, tokens: list[str]) -> tuple[str, bool, bool]:
    if not title:
        return "FAIL", False, False
    usable = _sealed_box_title_looks_usable(title)
    hit = any(tok in title for tok in tokens) if tokens else False
    if hit and usable:
        verdict = "PASS"
    elif hit:
        verdict = "PARTIAL"
    else:
        verdict = "FAIL"
    return verdict, usable, hit


def run_backend(
    model: str,
    *,
    endpoint: str,
    timeout: int,
    cases: list[dict],
) -> list[dict]:
    client = OllamaLocalVisionClient(endpoint=endpoint, model=model, timeout_seconds=timeout)
    out: list[dict] = []
    for case in cases:
        started = time.monotonic()
        try:
            candidate = client.analyze_sealed_box_title_focus(
                case["image_path"], game_hint=case["game"]
            )
            title = candidate.title if candidate is not None else None
            verdict, usable, hit = score(title, case["tokens"])
            elapsed = time.monotonic() - started
            print(
                f"  [{verdict:7s}] {case['slug']:30s} -> title={title!r} usable={usable} "
                f"expected_any={case['tokens']} elapsed={elapsed:.1f}s",
                flush=True,
            )
            out.append(
                {
                    "slug": case["slug"],
                    "verdict": verdict,
                    "title": title,
                    "usable": usable,
                    "hit": hit,
                    "elapsed_seconds": round(elapsed, 2),
                }
            )
        except Exception as exc:  # noqa: BLE001
            elapsed = time.monotonic() - started
            print(
                f"  [ERROR  ] {case['slug']:30s} -> {type(exc).__name__}: {exc} "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )
            out.append(
                {
                    "slug": case["slug"],
                    "verdict": "ERROR",
                    "error": f"{type(exc).__name__}: {exc}",
                    "elapsed_seconds": round(elapsed, 2),
                }
            )
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        required=True,
        help="Comma-separated Ollama model identifiers (e.g. qwen2.5vl:7b,gemma3:12b).",
    )
    parser.add_argument(
        "--endpoint",
        default="http://127.0.0.1:11434",
        help="Ollama HTTP endpoint (default: http://127.0.0.1:11434).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=180,
        help="Per-request timeout seconds (default: 180).",
    )
    parser.add_argument(
        "--out",
        default="/tmp/vision_backend_benchmark_results.json",
        help="Path to write the full JSON results (default: /tmp/vision_backend_benchmark_results.json).",
    )
    args = parser.parse_args(argv)

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    if not models:
        parser.error("--models must list at least one model.")

    cases = discover_cases()
    if not cases:
        parser.error(f"No benchmark cases found under {BENCHMARK_ROOT}.")

    print(f"Loaded {len(cases)} benchmark samples from {BENCHMARK_ROOT}.", flush=True)
    results: dict[str, list[dict]] = {}
    for model in models:
        print(f"\n##### {model} #####", flush=True)
        results[model] = run_backend(
            model, endpoint=args.endpoint, timeout=args.timeout, cases=cases
        )

    print("\n##### SUMMARY #####", flush=True)
    total = len(cases)
    for model, model_results in results.items():
        counts = {
            "PASS": sum(1 for r in model_results if r["verdict"] == "PASS"),
            "PARTIAL": sum(1 for r in model_results if r["verdict"] == "PARTIAL"),
            "FAIL": sum(1 for r in model_results if r["verdict"] == "FAIL"),
            "ERROR": sum(1 for r in model_results if r["verdict"] == "ERROR"),
        }
        print(
            f"  {model:30s} PASS={counts['PASS']}/{total}  PARTIAL={counts['PARTIAL']}  "
            f"FAIL={counts['FAIL']}  ERROR={counts['ERROR']}",
            flush=True,
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nWrote full results to {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
