# PTCG OCR + 即時查價規格（中文版）

最後更新：2026-03-03（JST）

本規格是「可重建目前服務」等級，會寫到實作細節。

## 1) 範圍
要完成的服務：
- 接收圖片上傳（`photo`）
- 請求當下做 OCR
- 由 OCR 欄位組查詢關鍵字
- 到網路來源抓即時價格
- 回傳 JPY 正規化結果、來源證據、warnings
- 全流程以 `trace_id` 可追蹤

禁止硬編碼價格表。

## 2) 技術與執行
- 前端：Vue 3 + Vite
- 後端：FastAPI + Pydantic v2
- OCR：pytesseract（必備）、easyocr（可選）
- 網頁抓取：httpx + BeautifulSoup
- 關鍵字相似度：rapidfuzz

環境變數：
- `APP_LOG_LEVEL`（預設 `INFO`）
- `ENABLE_NETOFF_SEARCH`（預設 `0`）

## 3) API 定義
### 3.1 `POST /price-check`
輸入：
- multipart 欄位 `photo`（必填）

輸出：
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

`PricePoint` 欄位：
- `source`, `vendor`, `label`, `kind`
- `currency`, `amount`, `amount_jpy`
- `observed_at`, `url`

行為規則：
- 空檔案 -> HTTP 400
- OCR 無法辨識 -> HTTP 200 + `unresolved`
- 有來源但無有效 sale points -> HTTP 200 + `partial`

### 3.2 `GET /healthz`
- 回傳 `{ "status": "ok" }`

## 4) OCR 規格（目前實作）
### 4.1 前處理
- EXIF 方向修正
- 灰階
- autocontrast
- unsharp mask
- 若有 OpenCV，做 adaptive threshold；沒有則用 PIL 流程

### 4.2 區域化 OCR
區域：
- `slab_top`
- `card_title`
- `card_body`
- `card_footer`

Tesseract 設定：
- `slab_top`: `eng`，psm 11 + 7
- `card_title`: `jpn+eng`，psm 7 + 11
- `card_body`: `jpn+eng`，psm 6 + 11
- `card_footer`: `jpn+eng`，psm 11 + 6

EasyOCR：
- 僅在安裝時啟用，作為補充文字來源
- 沒有 easyocr 時服務也必須正常運作

### 4.3 OCR 輸出
輸出 `OcrResult`：
- `raw_full_text`
- `regions`
- `tokens_by_region`
- `parsed_fields`
- `quality`

解析欄位：
- `slab_set_text`
- `card_name_en`
- `card_name_jp`
- `collector_no`
- `grade`
- `cert_no`

## 5) 查詢資料來源規格（目前實作）
### 5.1 Resolver 探索順序
1. PriceCharting（主要）
   - `GET https://www.pricecharting.com/search-products?q=<query>`
   - 解析 `/offers?product=...`
   - 進一步解析 `/game/...` 目標頁
2. PRICE BASE（次要）
   - `GET https://price-base.com/useful/wp-json/wp/v2/search`
   - 參數：`search`, `subtype=post`, `_fields=url,title`, `per_page`
   - 以 fuzzy score 選最佳文章
3. Netoff（可選）
   - 透過 DDG HTML：`https://html.duckduckgo.com/html/`
   - 查詢格式：`site:netoff.co.jp moetaku <query>`
   - 只有 `ENABLE_NETOFF_SEARCH=1` 才啟用

### 5.2 價格抓取
- PriceCharting：
  - 先抓 `#manual_only_price > span.price`
  - 失敗再解析 sales table
- PRICE BASE：
  - 解析標籤：`販売価格（PSA10）`, `販売価格`, `PSA10`
- Netoff：
  - 解析 buyback，必要時加 sale estimate
- 匯率：
  - `https://open.er-api.com/v6/latest/USD`（USD->JPY）

補充：
- `fetch_gamepedia_points` 有實作，但 resolver 目前未優先探索 Gamepedia。

## 6) Resolver / Pricing 職責
### 6.1 Resolver（`card_profiles.py`）
必須：
- 使用結構化 OCR（不依賴 alias 卡表）
- 依 OCR 欄位 + 檔名提示建立 query 優先序
- 產生 `CardProfile`：
  - `sources[]`
  - `status`
  - `warnings`
  - `confidence`

### 6.2 Pricing（`pricing.py`）
必須：
- 執行所有選中的來源
- 單來源失敗不可讓整體請求失敗
- 只用 sale points 算中位數
- 回傳降級後的最終狀態與 warnings

## 7) 日誌規格（強制）
所有 log 都必須帶 `trace_id`。

必要事件名稱：
- `step1 image_extract`
- `step2 query_keywords`
- `step2 query_target`
- `step3 query_result`

另需記錄：
- fetch start / response / parsed count
- 每筆 `PricePoint`
- 最終聚合結果（status / median / warnings）

## 8) 前端 Layout 規格（目前實作）
畫面區塊順序：
1. 上傳區塊
2. 錯誤區塊（條件顯示）
3. 結果區塊：
   - header：左邊卡資訊，右邊上傳縮圖
   - 中位數列
   - warnings
   - sources
   - query keywords
   - 價格表格

表格欄位固定：
- `Source`
- `Label`
- `Vendor`
- `URL`（可點擊）
- `Observed`
- `Amount (original)`
- `Amount (JPY)`

不保留獨立 `Source URLs` 區塊（避免和表格重複）。

## 9) 驗收
最小驗收：
- `PICKACHU3.jpg` 可解析到卡號與有效來源 URL
- 回應含 query keywords 與表格 URL
- log 有完整 step1/step2/step3
- 無法辨識時仍回 HTTP 200 + 結構化 payload

行情 sanity（僅參考，依即時資料浮動）：
- 梵谷皮卡丘類型：30 萬日幣附近
- `#764` 樣本：30 萬日幣附近

## 10) 完成定義（DoD）
若以下都明確，文件即可重建服務：
- API payload 契約
- OCR 參數與行為
- 資料來源網址與查詢順序
- Resolver/Pricing 職責
- 日誌事件規範
- 前端 layout 規範
- 建置與 smoke test 指令
