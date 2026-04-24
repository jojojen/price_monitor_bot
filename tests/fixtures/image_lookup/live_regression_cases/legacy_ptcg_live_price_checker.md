# PTCG Live Price Checker

This repository implements a complete Vue 3 + FastAPI service that:
1) OCRs uploaded card/slab images in real time
2) Generates search keywords from OCR output
3) Queries live web sources for price data
4) Returns normalized JPY prices with full step-by-step logs

No static card-price table is used.

## 1) Repository Layout
```text
demo_codex_card_pricing/
|-- backend/
|   |-- app/
|   |   |-- main.py
|   |   |-- schemas.py
|   |   `-- services/
|   |       |-- ocr.py
|   |       |-- card_profiles.py
|   |       |-- fetchers.py
|   |       `-- pricing.py
|   |-- requirements.txt
|   `-- run_price_tests.py
|-- frontend/
|   `-- src/App.vue
|-- project.md
`-- project_ch.md
```

## 2) Setup
### Backend (virtualenv required)
```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload --log-level info
```

### Frontend
```powershell
cd frontend
npm install
npm run dev -- --host
```

### Runtime env
```env
APP_LOG_LEVEL=INFO
ENABLE_NETOFF_SEARCH=0
VITE_API_BASE=http://127.0.0.1:8001
```

- `APP_LOG_LEVEL`: root logger level.
- `ENABLE_NETOFF_SEARCH`: enables DDG-based Netoff discovery (`0` by default for stability).

Tesseract must exist on host machine (`C:\Program Files\Tesseract-OCR\tesseract.exe` on common Windows installs).

## 3) OCR Implementation (current, concrete)
Current OCR stack is not generic; it is implemented as:

1. Image preprocessing:
   - EXIF orientation normalize
   - grayscale + autocontrast + unsharp mask
   - optional OpenCV adaptive threshold (`opencv-python-headless`); fallback to PIL-only if unavailable
2. Region OCR:
   - `slab_top`: tesseract `eng` with psm 11 and 7
   - `card_title`: tesseract `jpn+eng` with psm 7 and 11
   - `card_body`: tesseract `jpn+eng` with psm 6 and 11
   - `card_footer`: tesseract `jpn+eng` with psm 11 and 6
3. EasyOCR:
   - optional fallback per region when package is installed
   - service still works if easyocr is missing
4. Parsed fields:
   - `slab_set_text`, `card_name_en`, `card_name_jp`, `collector_no`, `grade`, `cert_no`

## 4) Query/Data Sources (current, concrete)
Resolver/fetch pipeline currently targets these sources:

1. PriceCharting (primary)
   - search: `https://www.pricecharting.com/search-products?q=<query>`
   - follow: `/offers?product=<id>` -> extract `/game/...` URL
   - price parse: summary selector `#manual_only_price > span.price`, fallback sales tables
2. PRICE BASE (secondary discovery)
   - search API: `https://price-base.com/useful/wp-json/wp/v2/search`
   - params: `search`, `subtype=post`, `_fields=url,title`, `per_page`
   - page parse labels: `販売価格（PSA10）`, `販売価格`, `PSA10`
3. Netoff (optional)
   - discovery via DDG HTML: `https://html.duckduckgo.com/html/`
   - query pattern: `site:netoff.co.jp moetaku <query>`
   - disabled by default unless `ENABLE_NETOFF_SEARCH=1`
4. FX conversion
   - `https://open.er-api.com/v6/latest/USD`
   - used when source currency is USD

`fetch_gamepedia_points` exists for compatible configs but resolver currently does not prioritize Gamepedia discovery.

## 5) Backend Contract
### Endpoints
- `POST /price-check` (multipart field `photo`)
- `GET /healthz`

### `PriceCheckResponse`
- `status`: `success | partial | unresolved`
- `warnings[]`
- `trace_id`
- `card_id`, `card_name`, `locale_name`, `collector_number`, `confidence`
- `search_terms[]`
- `ocr_text`, `ocr_regions`
- `price_points[]` (`source/vendor/label/kind/currency/amount/amount_jpy/observed_at/url`)
- `source_targets[]`
- `median_price_jpy` (nullable)
- `sources_used[]`

Behavior:
- empty file -> HTTP 400
- unresolved OCR -> HTTP 200 + `unresolved`
- no sale points -> HTTP 200 + `partial`

## 6) Frontend Layout Contract (current)
`frontend/src/App.vue` layout order:
1. Upload panel (title, dropzone, submit/reset)
2. Error panel (shown only on request failure)
3. Result panel:
   - header: left = card/status/trace, right = uploaded thumbnail
   - median row
   - warnings block (conditional)
   - sources block
   - query keywords block
   - result table

Result table columns:
- `Source`, `Label`, `Vendor`, `URL`, `Observed`, `Amount (original)`, `Amount (JPY)`

There is intentionally no standalone "Source URLs" section (duplicate of table URL column).

## 7) Logging Contract
Every request is traceable by `trace_id`.

- `step1 image_extract`: OCR text/tokens/parsed fields from image
- `step2 query_keywords`: generated query strings + normalized terms
- `step2 query_target`: selected source targets and URLs
- `step3 query_result`: search hit counts, top candidates, selected URL
- fetch logs: request URL/status/bytes/parsed count
- pricing logs: each normalized `PricePoint` + final aggregate

## 8) Smoke Test
```powershell
cd backend
$env:PYTHONIOENCODING='utf-8'
python run_price_tests.py
```

Covers:
- `charizard.jpg`
- `pikachu.jpg`
- `PICKACHU3.jpg`
- `ririe.jpg`

## 9) Notes
- Prices are fetched live at request time.
- Partial/unresolved responses are expected for weak OCR cases.
- Use frontend `trace_id` to follow exact backend steps in logs.
