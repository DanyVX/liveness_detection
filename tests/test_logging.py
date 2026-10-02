import json
import logging

import pytest

from liveness.logging_utils import JsonFormatter, configure_logging


def test_json_formatter_includes_correlation_id() -> None:
    record = logging.LogRecord("t", logging.INFO, __file__, 1, "hello %s", ("x",), None)
    record.correlation_id = "abc"
    out = json.loads(JsonFormatter().format(record))
    assert out["msg"] == "hello x" and out["correlation_id"] == "abc" and out["level"] == "INFO"


def test_json_formatter_includes_exception() -> None:
    try:
        raise ValueError("boom")
    except ValueError:
        import sys

        record = logging.LogRecord("t", logging.ERROR, __file__, 1, "failed", (), sys.exc_info())
    assert "ValueError: boom" in json.loads(JsonFormatter().format(record))["exc"]


@pytest.mark.parametrize("json_output", [True, False])
def test_configure_logging_sets_level(json_output: bool) -> None:
    configure_logging("WARNING", json_output)
    assert logging.getLogger().level == logging.WARNING
    assert len(logging.getLogger().handlers) == 1
