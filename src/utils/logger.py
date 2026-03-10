"""
Structured JSON logger for the Starlink Amazon DRL beamforming system.

Provides a :class:`StructuredLogger` that wraps Python's standard
``logging`` module and emits every log record as a single-line JSON
object.  This format is directly ingestable by ELK Stack, Loki, and
other centralised log aggregation systems.

Each log record includes:
    - ``timestamp``   – ISO-8601 UTC timestamp.
    - ``level``       – Log level name (INFO, WARNING, ERROR, DEBUG).
    - ``logger``      – Logger name / component identifier.
    - ``message``     – Human-readable event description.
    - ``event``       – Machine-readable event key (e.g. ``"handover"``).
    - ``satellite_id``– Optional satellite identifier.
    - ``extra``       – Any additional key-value context.

Usage::

    logger = get_logger("inference")
    logger.info("Handover executed", event="handover",
                satellite_id="SAT-42", snr_db=12.5, rain_mm_h=30.0)

    # Produces: {"timestamp": "...", "level": "INFO", "logger": "inference",
    #            "message": "Handover executed", "event": "handover",
    #            "satellite_id": "SAT-42", "snr_db": 12.5, "rain_mm_h": 30.0}
"""

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any, Optional


# ---------------------------------------------------------------------------
# JSON formatter
# ---------------------------------------------------------------------------

class _JsonFormatter(logging.Formatter):
    """
    Format a :class:`logging.LogRecord` as a single-line JSON string.

    Standard :attr:`logging.LogRecord` attributes are mapped to fixed JSON
    keys; any extra key-value pairs attached via ``extra=`` in the log call
    are merged at the top level.
    """

    # LogRecord attributes that are handled explicitly (not dumped as-is)
    _SKIP_ATTRS = frozenset({
        "args", "created", "exc_info", "exc_text", "filename",
        "funcName", "levelname", "levelno", "lineno", "message",
        "module", "msecs", "msg", "name", "pathname", "process",
        "processName", "relativeCreated", "stack_info", "thread",
        "threadName",
    })

    def format(self, record: logging.LogRecord) -> str:  # noqa: A003
        # Ensure record.message is populated
        record.message = record.getMessage()

        ts = datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat()
        payload: dict = {
            "timestamp": ts,
            "level": record.levelname,
            "logger": record.name,
            "message": record.message,
        }

        # Merge any extra fields attached to the record
        for key, value in record.__dict__.items():
            if key not in self._SKIP_ATTRS and not key.startswith("_"):
                try:
                    json.dumps(value)  # verify it's JSON-serialisable
                    payload[key] = value
                except (TypeError, ValueError):
                    payload[key] = str(value)

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def get_logger(
    name: str,
    level: int = logging.INFO,
    stream=None,
) -> logging.Logger:
    """
    Return a :class:`logging.Logger` that emits structured JSON records.

    Calling this function multiple times with the same ``name`` returns the
    same underlying logger (standard Python ``logging`` semantics).

    Args:
        name:   Logger name, typically the component (e.g. ``"inference"``,
                ``"channel"``, ``"handover"``).
        level:  Minimum log level (default ``logging.INFO``).
        stream: Output stream (default ``sys.stdout``).

    Returns:
        Configured :class:`logging.Logger` instance.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        # Already configured – return as-is to avoid duplicate handlers
        return logger

    handler = logging.StreamHandler(stream or sys.stdout)
    handler.setFormatter(_JsonFormatter())
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger


class StructuredLogger:
    """
    Thin wrapper around :class:`logging.Logger` with convenience methods
    for the most common beamforming events.

    Args:
        name:  Logger / component name.
        level: Minimum log level.
    """

    def __init__(self, name: str, level: int = logging.INFO) -> None:
        self._log = get_logger(name, level)
        self.name = name

    # ------------------------------------------------------------------
    # Generic log methods
    # ------------------------------------------------------------------

    def debug(self, message: str, **kwargs: Any) -> None:
        self._log.debug(message, extra=kwargs)

    def info(self, message: str, **kwargs: Any) -> None:
        self._log.info(message, extra=kwargs)

    def warning(self, message: str, **kwargs: Any) -> None:
        self._log.warning(message, extra=kwargs)

    def error(self, message: str, **kwargs: Any) -> None:
        self._log.error(message, extra=kwargs)

    # ------------------------------------------------------------------
    # Domain-specific convenience methods
    # ------------------------------------------------------------------

    def log_decision(
        self,
        satellite_id: Any,
        action,
        snr_db: float,
        rain_mm_h: float,
        latency_ms: float,
        **kwargs: Any,
    ) -> None:
        """
        Log a beamforming decision (one per inference step).

        Args:
            satellite_id:  Identifier of the chosen satellite.
            action:        Action array or integer from the DRL agent.
            snr_db:        Current SNR (dB).
            rain_mm_h:     Current rain rate (mm/h).
            latency_ms:    Inference latency (ms).
        """
        self._log.info(
            "Beamforming decision",
            extra={
                "event": "decision",
                "satellite_id": satellite_id,
                "action": action.tolist() if hasattr(action, "tolist") else action,
                "snr_db": snr_db,
                "rain_mm_h": rain_mm_h,
                "latency_ms": latency_ms,
                **kwargs,
            },
        )

    def log_handover(
        self,
        from_sat: Any,
        to_sat: Any,
        reason: str = "policy",
        **kwargs: Any,
    ) -> None:
        """
        Log a satellite handover event.

        Args:
            from_sat: Previous satellite identifier.
            to_sat:   New satellite identifier.
            reason:   Why the handover occurred (``"policy"``, ``"fallback"``,
                      ``"timeout"``).
        """
        self._log.info(
            f"Handover: {from_sat} → {to_sat}",
            extra={
                "event": "handover",
                "from_satellite": from_sat,
                "to_satellite": to_sat,
                "reason": reason,
                **kwargs,
            },
        )

    def log_outage(
        self,
        satellite_id: Any,
        snr_db: float,
        duration_steps: int = 1,
        **kwargs: Any,
    ) -> None:
        """
        Log an outage event.

        Args:
            satellite_id:   Satellite affected.
            snr_db:         SNR at the time of outage (dB).
            duration_steps: Number of consecutive outage steps.
        """
        self._log.warning(
            "Outage detected",
            extra={
                "event": "outage",
                "satellite_id": satellite_id,
                "snr_db": snr_db,
                "duration_steps": duration_steps,
                **kwargs,
            },
        )

    def log_fallback(self, reason: str, **kwargs: Any) -> None:
        """
        Log activation of the deterministic fallback policy.

        Args:
            reason: Why the fallback was triggered (``"timeout"``,
                    ``"agent_error"``, ``"no_visible_satellites"``).
        """
        self._log.warning(
            f"Fallback policy activated: {reason}",
            extra={"event": "fallback", "reason": reason, **kwargs},
        )
