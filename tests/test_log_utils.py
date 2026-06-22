import socket
import ssl
from urllib.error import HTTPError, URLError

import pytest

from market_monitor.http import HostRateLimitedError
from market_monitor.log_utils import (
    describe_network_error,
    is_transient_network_error,
    log_network_failure,
)


@pytest.mark.parametrize(
    "exc",
    [
        HTTPError("u", 429, "Too Many Requests", {}, None),
        HTTPError("u", 404, "Not Found", {}, None),
        HTTPError("u", 403, "Forbidden", {}, None),
        URLError("nodename nor servname provided"),
        TimeoutError("read timed out"),
        socket.timeout("timed out"),
        ssl.SSLError("UNEXPECTED_EOF_WHILE_READING"),
        ConnectionResetError("reset"),
        HostRateLimitedError("yuyu-tei.jp", 30.0),
    ],
)
def test_transient_network_errors_classified(exc):
    assert is_transient_network_error(exc) is True


@pytest.mark.parametrize("exc", [ValueError("boom"), KeyError("x"), RuntimeError("nope")])
def test_non_transient_errors_not_classified(exc):
    assert is_transient_network_error(exc) is False


def test_transient_logged_as_single_warning_without_stack(caplog):
    import logging

    with caplog.at_level("DEBUG"):
        log_network_failure(
            logging.getLogger("t"),
            HTTPError("u", 429, "Too Many Requests", {}, None),
            "fetch failed url=%s",
            "http://x",
        )
    assert "fetch failed url=http://x — HTTP 429 Too Many Requests" in caplog.text
    assert "Traceback" not in caplog.text
    assert all(r.levelname == "WARNING" for r in caplog.records)


def test_unexpected_error_keeps_full_stack_at_error(caplog):
    import logging

    with caplog.at_level("DEBUG"):
        try:
            raise ValueError("real bug")
        except ValueError as exc:
            log_network_failure(logging.getLogger("t"), exc, "fetch failed url=%s", "http://y")
    assert any(r.levelname == "ERROR" for r in caplog.records)
    assert "Traceback" in caplog.text
    assert "ValueError: real bug" in caplog.text


def test_describe_network_error_is_compact():
    assert describe_network_error(HTTPError("u", 429, "Too Many", {}, None)) == "HTTP 429 Too Many"
    assert "URLError" in describe_network_error(URLError("dns"))
