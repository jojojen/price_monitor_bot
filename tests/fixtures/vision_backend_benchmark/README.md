# Vision Backend Benchmark — Sealed-Box Title Probe

A reusable set of snkrdunk Pokemon sealed booster-box product thumbnails
for comparing local vision LLM backends on the *sealed-box title probe*
specifically (`OllamaLocalVisionClient.analyze_sealed_box_title_focus`).

This is **not** part of the live regression suite — those fixtures live
under `live_regression_cases/` and assert full end-to-end pipeline state.
This benchmark only scores how well a backend reads the set-name printed
on the box front.

## When to use

- A new vision model is released (`mistral-small3.2:24b`, `llava-next`,
  next `qwen2.5vl` revision, etc.) and you want to know whether it should
  replace `qwen2.5vl:7b` as the default.
- The sealed-box title prompt is materially edited and you want to make
  sure no regression slipped in.

## Layout

```
vision_backend_benchmark/
  <case_slug>/
    image.webp          # snkrdunk bg-removed product thumbnail
    expected.json       # set_name_canonical + title_contains_any tokens
```

Each `expected.json`:

```json
{
  "case_id": "abyss-eye",
  "game": "pokemon",
  "item_kind": "sealed_box",
  "set_name_canonical": "アビスアイ",
  "title_contains_any": ["アビスアイ", "MEGA", "メガ"],
  "source_url": "https://snkrdunk.com/apparels/806644",
  "image_url": "https://cdn.snkrdunk.com/upload_bg_removed/..."
}
```

The scoring is intentionally lenient — a backend PASSes a case if its
returned title contains **any** of the `title_contains_any` tokens. A real
box prints multiple equivalent renderings (`アビスアイ`,
`MEGA アビスアイ`, `ポケモンカードゲームMEGA 拡張パック アビスアイ`)
and any of them is a fair read.

## Running

From the `price_monitor_bot` repo root:

```bash
.venv/bin/python scripts/benchmark_vision_backends.py \
    --models qwen2.5vl:7b,gemma3:12b \
    --endpoint http://127.0.0.1:11434
```

Output is both a human-readable table and a JSON summary written to
`/tmp/vision_backend_benchmark_results.json` (override with `--out`).

## Adding new samples

1. Find a snkrdunk Pokemon sealed-box product page. The bg-removed
   thumbnail URL is in the page's `og:image` meta tag and has the form
   `https://cdn.snkrdunk.com/upload_bg_removed/<id>.webp`.
2. Download the webp to `<case_slug>/image.webp` (lowercase-hyphen slug).
3. Write `<case_slug>/expected.json` following the schema above. Choose
   tokens lenient enough that any reasonable read passes — usually the
   katakana set name, plus 1-2 partial fragments humans would accept as
   "close enough".
4. Re-run the benchmark to make sure your tokens aren't accidentally
   matching unrelated outputs.

## 2026-05-26 baseline

Backends benchmarked on these 15 samples (`analyze_sealed_box_title_focus`):

| Backend         | PASS  | PARTIAL | FAIL  |
|-----------------|-------|---------|-------|
| qwen2.5vl:7b    | 11/15 | 1       | 3     |
| gemma3:12b      | 0/15  | 4       | 11    |

`gemma3:12b` predominantly returned the series fallback
`スカーレット&バイオレット` (7 cases) or fabricated English set names
(`MEGA Cosmic Guardians`, `MEGA ミラクルファイター`). After this run we
removed it from the default `OPENCLAW_LOCAL_VISION_MODEL` list.
