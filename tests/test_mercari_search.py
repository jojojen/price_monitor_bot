from __future__ import annotations

from market_monitor.mercari_search import _chromium_launch_options


def test_chromium_launch_options_use_configured_system_chromium(monkeypatch) -> None:
    monkeypatch.setenv("PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH", "/usr/bin/chromium")

    assert _chromium_launch_options() == {
        "headless": True,
        "executable_path": "/usr/bin/chromium",
    }


def test_chromium_launch_options_fall_back_to_playwright_bundle(monkeypatch) -> None:
    monkeypatch.delenv("PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH", raising=False)
    monkeypatch.setattr("market_monitor.mercari_search.shutil.which", lambda command: None)

    assert _chromium_launch_options() == {"headless": True}
