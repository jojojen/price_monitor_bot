Image lookup regression fixtures that should stay in git.

Current curated samples:
- `telegram-upload-charizard_sv2a_official.jpg`
- `paypay_charizard_psa.jpg`
- `rosetreca_charizard_psa.jpg`

Live regression cases now live under `live_regression_cases/`.
Each case keeps one input image plus an `expected.json` file in the same folder.

Recommended verification command after image-lookup changes:
`$env:OPENCLAW_RUN_LIVE_IMAGE_FIXTURES='1'; .\.venv\Scripts\python.exe -m pytest tests/test_image_lookup_live_regression.py -q`

Keep transient benchmark output and markdown reports out of git.
Store those under `reports/` or `.openclaw_tmp/`.
