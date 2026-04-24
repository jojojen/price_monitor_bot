# TCG Image Lookup Live Regression Cases

This folder replaces the old `fwdspecptcg/` drop folder.

Each regression case now lives in its own subfolder and keeps:
- one input image (`image.jpg` or `image.jpeg`)
- one `expected.json` file with the current reasonable lookup expectation

Current cases:
- `pokemon-mega-charizard-x-ex-223-193-ma-m2a`
- `pokemon-mega-charizard-x-ex-110-080-sar-m2`
- `pokemon-charizard-ex-201-165-sar-sv2a`
- `pokemon-pikachu-020-m-p-promo`
- `pokemon-pikachu-ex-764-742-mc`
- `pokemon-pikachu-partial-s40`
- `pokemon-lillies-clefairy-ex-126-100-sar-sv9`

Live verification commands:
- `.\.venv\Scripts\python.exe scripts\verify_image_lookup_live_fixtures.py`
- `$env:OPENCLAW_RUN_LIVE_IMAGE_FIXTURES='1'; .\.venv\Scripts\python.exe -m pytest tests/test_image_lookup_live_regression.py -q`

The older planning/spec notes from `fwdspecptcg/` were preserved here as:
- `legacy_ptcg_live_price_checker.md`
- `legacy_spec_en.md`
- `legacy_spec_zh.md`
