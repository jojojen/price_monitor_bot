# PTCG OCR + Live Pricing Spec (English)

Last updated: March 3, 2026 (JST)

This spec is intentionally implementation-level so the current service can be rebuilt from it.

## 1) Scope
Build a service that:
- Accepts image upload (`photo`)
- OCRs card/slab text at request time
- Builds query keywords from OCR fields
- Fetches live market prices from web sources
- Returns normalized JPY output + provenance + warnings
- Emits full trace logs for every pipeline step

Hardcoded price catalogs are prohibited.

## 2) Runtime Stack
- Frontend: Vue 3 + Vite
- Backend: FastAPI + Pydantic v2
- OCR: pytesseract (required), easyocr (optional fallback)
- Parsing/fetch: httpx + BeautifulSoup
- Matching: rapidfuzz

Environment variables:
- `APP_LOG_LEVEL` (`INFO` default)
- `ENABLE_NETOFF_SEARCH` (`0` default)

## 3) API Definition
### 3.1 POST `/price-check`
Input:
- multipart `photo` (required)

Output:
- `status`: `success | partial | unresolved`
- `warnings: string[]`
- `trace_id: string`
- `card_id`, `card_name`, `locale_name`, `collector_number`, `confidence`
- `search_terms: string[]`
- `ocr_text: string | null`
- `ocr_regions: object`
- `price_points: PricePoint[]`
- `source_targets: string[]`
- `median_price_jpy: number | null`
- `sources_used: string[]`

`PricePoint` fields:
- `source`, `vendor`, `label`, `kind`
- `currency`, `amount`, `amount_jpy`
- `observed_at`, `url`

Behavior:
- empty payload -> `400`
- OCR unresolved -> `200` + `unresolved`
- source found but no usable sale points -> `200` + `partial`

### 3.2 GET `/healthz`
- returns `{ "status": "ok" }`

## 4) OCR Specification (exact current behavior)
### 4.1 Preprocessing
- EXIF transpose
- grayscale
- autocontrast
- unsharp mask
- optional OpenCV adaptive threshold; fallback to PIL-only if OpenCV missing

### 4.2 Region OCR
Regions:
- `slab_top`
- `card_title`
- `card_body`
- `card_footer`

Tesseract configs:
- `slab_top`: `eng`, psm 11 + 7
- `card_title`: `jpn+eng`, psm 7 + 11
- `card_body`: `jpn+eng`, psm 6 + 11
- `card_footer`: `jpn+eng`, psm 11 + 6

EasyOCR usage:
- optional per-region text merge when package is available
- absence of easyocr must not break service

### 4.3 OCR output
Return structured `OcrResult`:
- `raw_full_text`
- `regions`
- `tokens_by_region`
- `parsed_fields`
- `quality`

Parsed fields:
- `slab_set_text`
- `card_name_en`
- `card_name_jp`
- `collector_no`
- `grade`
- `cert_no`

## 5) Data Source Specification (exact current behavior)
### 5.1 Discovery stage (resolver)
1. PriceCharting search:
   - `GET https://www.pricecharting.com/search-products?q=<query>`
   - parse candidate `/offers?product=...`
   - open offer page and extract `/game/...` URL
2. PRICE BASE search:
   - `GET https://price-base.com/useful/wp-json/wp/v2/search`
   - params: `search`, `subtype=post`, `_fields=url,title`, `per_page`
   - fuzzy select best candidate
3. Netoff search (optional):
   - DDG HTML endpoint: `https://html.duckduckgo.com/html/`
   - query pattern: `site:netoff.co.jp moetaku <query>`
   - enabled only when `ENABLE_NETOFF_SEARCH=1`

### 5.2 Price extraction stage (fetchers)
- PriceCharting:
  - summary selector first: `#manual_only_price > span.price`
  - fallback sales table parsing
- PRICE BASE:
  - parse labels: `販売価格（PSA10）`, `販売価格`, `PSA10`
- Netoff:
  - parse buyback and optional sale estimate (configured multiplier)
- FX conversion:
  - `https://open.er-api.com/v6/latest/USD` for USD->JPY

Note:
- `fetch_gamepedia_points` is implemented, but resolver currently prioritizes the 3 sources above.

## 6) Resolver and Pricing Requirements
### 6.1 Resolver (`card_profiles.py`)
Must:
- use structured OCR output only (no alias table dependency)
- generate prioritized query list from OCR fields + filename hints
- generate `CardProfile` with:
  - `sources[]`
  - `status`
  - `warnings`
  - `confidence`

### 6.2 Pricing (`pricing.py`)
Must:
- execute all selected sources
- tolerate per-source failures
- aggregate warnings
- compute median only from sale points
- return final status/result payload

## 7) Logging Contract (mandatory)
All logs must include `trace_id`.

Required event families:
- `step1 image_extract`
- `step2 query_keywords`
- `step2 query_target`
- `step3 query_result`

Also required:
- fetch start/response/parsed count per source
- each normalized `PricePoint`
- final aggregate summary

## 8) Frontend Layout Contract (exact current behavior)
Result page structure:
1. Upload panel
2. Error panel (conditional)
3. Result panel:
   - header: left metadata, right uploaded thumbnail
   - median row
   - warnings block
   - sources block
   - query keywords block
   - table block

Result table columns:
- `Source`
- `Label`
- `Vendor`
- `URL` (clickable)
- `Observed`
- `Amount (original)`
- `Amount (JPY)`

No separate "Source URLs" section.

## 9) Validation / Acceptance
Minimum checks:
- `PICKACHU3.jpg` resolves to a valid collector number and source URL
- response contains query keywords and table URL links
- logs contain full step1/step2/step3 trail
- unresolved requests still return HTTP 200 with structured payload

Sanity ranges (non-binding, live-dependent):
- Van Gogh Pikachu class images: around low 300k JPY
- `#764` sample: around low 300k JPY

## 10) Definition of Done
If all below are explicit, this service is reproducible from docs:
- API payload contract
- OCR behavior and params
- source discovery/extraction flow + URLs
- resolver/pricing responsibilities
- logging event contract
- frontend layout contract
- setup and smoke test commands
