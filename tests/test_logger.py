"""
Tests for the structured JSON logger.
"""

import io
import json
import logging
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest


def _capture_logger(name: str):
    """Return a StructuredLogger that writes to a StringIO buffer."""
    from utils.logger import StructuredLogger, get_logger, _JsonFormatter
    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    handler.setFormatter(_JsonFormatter())
    log = logging.getLogger(name + "_test_capture")
    log.handlers.clear()
    log.addHandler(handler)
    log.setLevel(logging.DEBUG)
    log.propagate = False
    return log, buf


class TestJsonFormatter:
    def test_output_is_valid_json(self):
        log, buf = _capture_logger("fmt_test")
        log.info("hello world")
        line = buf.getvalue().strip()
        payload = json.loads(line)
        assert payload["message"] == "hello world"
        assert payload["level"] == "INFO"

    def test_timestamp_present(self):
        log, buf = _capture_logger("fmt_ts")
        log.info("ts test")
        payload = json.loads(buf.getvalue().strip())
        assert "timestamp" in payload
        assert "T" in payload["timestamp"]  # ISO-8601

    def test_extra_fields_included(self):
        log, buf = _capture_logger("fmt_extra")
        log.info("event", extra={"event": "handover", "snr_db": 12.5})
        payload = json.loads(buf.getvalue().strip())
        assert payload["event"] == "handover"
        assert abs(payload["snr_db"] - 12.5) < 1e-9

    def test_exception_included(self):
        log, buf = _capture_logger("fmt_exc")
        try:
            raise ValueError("test error")
        except ValueError:
            log.error("error occurred", exc_info=True)
        payload = json.loads(buf.getvalue().strip())
        assert "exception" in payload
        assert "ValueError" in payload["exception"]


class TestStructuredLogger:
    def _make(self, name: str = "test"):
        from utils.logger import StructuredLogger
        return StructuredLogger(f"{name}_struct")

    def test_import(self):
        from utils.logger import StructuredLogger
        assert StructuredLogger is not None

    def test_exported_from_utils(self):
        from utils import StructuredLogger
        assert StructuredLogger is not None

    def test_info_does_not_raise(self):
        logger = self._make("info")
        logger.info("test message", event="test")

    def test_warning_does_not_raise(self):
        logger = self._make("warn")
        logger.warning("test warning")

    def test_error_does_not_raise(self):
        logger = self._make("err")
        logger.error("test error")

    def test_log_decision_does_not_raise(self):
        import numpy as np
        logger = self._make("dec")
        logger.log_decision(
            satellite_id="SAT-1",
            action=np.array([0.1, 0.8, 2, 50]),
            snr_db=14.0,
            rain_mm_h=5.0,
            latency_ms=3.2,
        )

    def test_log_handover_does_not_raise(self):
        logger = self._make("ho")
        logger.log_handover("SAT-1", "SAT-2", reason="policy")

    def test_log_outage_does_not_raise(self):
        logger = self._make("out")
        logger.log_outage("SAT-1", snr_db=2.0, duration_steps=3)

    def test_log_fallback_does_not_raise(self):
        logger = self._make("fb")
        logger.log_fallback("timeout")


class TestGetLogger:
    def test_returns_logger_instance(self):
        from utils.logger import get_logger
        log = get_logger("test_get")
        assert isinstance(log, logging.Logger)

    def test_same_name_returns_same_logger(self):
        from utils.logger import get_logger
        a = get_logger("same_name_x")
        b = get_logger("same_name_x")
        assert a is b
