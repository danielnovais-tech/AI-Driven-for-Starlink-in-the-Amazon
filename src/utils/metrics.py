"""
Prometheus-compatible runtime metrics for the beamforming controller.

Provides an in-process metrics registry that collects counters, gauges,
and histograms without requiring a running Prometheus server.  Metrics
can be exported in:
    - Prometheus text format (via :meth:`MetricsRegistry.to_prometheus_text`)
    - Python dictionary (via :meth:`MetricsRegistry.snapshot`)

A singleton :data:`GLOBAL_REGISTRY` is available for convenience.

Metrics tracked:
    - ``decisions_total``         – Counter: total beamforming decisions.
    - ``handovers_total``         – Counter: total handover events.
    - ``outages_total``           – Counter: total outage events.
    - ``fallback_total``          – Counter: fallback policy activations.
    - ``snr_db``                  – Gauge: last observed SNR (dB).
    - ``rain_rate_mmh``           – Gauge: last observed rain rate (mm/h).
    - ``inference_latency_ms``    – Histogram: per-decision inference latency.
    - ``throughput_mbps``         – Histogram: per-decision throughput.

Usage::

    from utils.metrics import GLOBAL_REGISTRY as metrics

    metrics.increment("decisions_total")
    metrics.set_gauge("snr_db", 14.2)
    metrics.observe("inference_latency_ms", 3.5)

    # Export for Prometheus scrape
    print(metrics.to_prometheus_text())
"""

import math
import threading
import time
from typing import Dict, List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Metric primitives
# ---------------------------------------------------------------------------

class Counter:
    """
    Monotonically increasing integer counter.

    Thread-safe via a :class:`threading.Lock`.

    Args:
        name:   Metric name (used in Prometheus output).
        help_:  Human-readable description.
        labels: Static label dict applied to all observations.
    """

    def __init__(self, name: str, help_: str = "", labels: Optional[Dict] = None) -> None:
        self.name = name
        self.help = help_
        self.labels = labels or {}
        self._value: int = 0
        self._lock = threading.Lock()

    def increment(self, amount: int = 1) -> None:
        """Increase the counter by ``amount``."""
        with self._lock:
            self._value += amount

    def get(self) -> int:
        """Return the current counter value."""
        return self._value

    def reset(self) -> None:
        """Reset the counter to zero (useful for testing)."""
        with self._lock:
            self._value = 0


class Gauge:
    """
    A metric that can go up or down (last-value semantics).

    Args:
        name:   Metric name.
        help_:  Human-readable description.
    """

    def __init__(self, name: str, help_: str = "") -> None:
        self.name = name
        self.help = help_
        self._value: float = 0.0
        self._lock = threading.Lock()

    def set(self, value: float) -> None:
        """Set the gauge to ``value``."""
        with self._lock:
            self._value = float(value)

    def get(self) -> float:
        """Return the current gauge value."""
        return self._value


class Histogram:
    """
    Tracks the distribution of observed values using configurable buckets.

    Computes sum, count, and bucketed cumulative counts compatible with
    the Prometheus histogram data model.

    Args:
        name:    Metric name.
        help_:   Human-readable description.
        buckets: Sorted list of upper-bound values (must end with +Inf).
                 Defaults to a latency-friendly set in milliseconds.
    """

    _DEFAULT_BUCKETS = (1.0, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0, float("inf"))

    def __init__(
        self,
        name: str,
        help_: str = "",
        buckets: tuple = (),
    ) -> None:
        self.name = name
        self.help = help_
        self._buckets = buckets if buckets else self._DEFAULT_BUCKETS
        self._counts: List[int] = [0] * len(self._buckets)
        self._sum: float = 0.0
        self._total: int = 0
        self._lock = threading.Lock()

    def observe(self, value: float) -> None:
        """Record a single observation."""
        with self._lock:
            self._sum += value
            self._total += 1
            for i, upper in enumerate(self._buckets):
                if value <= upper:
                    self._counts[i] += 1

    def snapshot(self) -> Dict:
        """Return a dict with ``sum``, ``count``, ``p50``, ``p95``, ``p99``."""
        with self._lock:
            if self._total == 0:
                return {"sum": 0.0, "count": 0, "p50": 0.0, "p95": 0.0, "p99": 0.0}
            # Estimate percentiles from bucket boundaries
            p50 = self._percentile(0.50)
            p95 = self._percentile(0.95)
            p99 = self._percentile(0.99)
            return {
                "sum": self._sum,
                "count": self._total,
                "p50": p50,
                "p95": p95,
                "p99": p99,
                "mean": self._sum / self._total,
            }

    def _percentile(self, q: float) -> float:
        """Linear interpolation within the bucket containing the q-th quantile."""
        target = q * self._total
        cumulative = 0
        for i, upper in enumerate(self._buckets):
            prev_cumulative = cumulative
            cumulative += self._counts[i]
            if cumulative >= target:
                lower = self._buckets[i - 1] if i > 0 else 0.0
                if upper == float("inf"):
                    return lower
                bucket_count = self._counts[i]
                if bucket_count == 0:
                    return lower
                frac = (target - prev_cumulative) / bucket_count
                return lower + frac * (upper - lower)
        return float(self._buckets[-2]) if len(self._buckets) > 1 else 0.0


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class MetricsRegistry:
    """
    Central container for all runtime metrics.

    Registers named :class:`Counter`, :class:`Gauge`, and
    :class:`Histogram` instances.  All public methods are thread-safe.

    Pre-registered metrics match the standard beamforming controller
    dimensions described in the module docstring.
    """

    def __init__(self) -> None:
        self._counters: Dict[str, Counter] = {}
        self._gauges: Dict[str, Gauge] = {}
        self._histograms: Dict[str, Histogram] = {}
        self._start_time: float = time.time()
        self._lock = threading.Lock()
        self._register_defaults()

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register_counter(self, name: str, help_: str = "") -> Counter:
        """Register and return a :class:`Counter`."""
        with self._lock:
            if name not in self._counters:
                self._counters[name] = Counter(name, help_)
            return self._counters[name]

    def register_gauge(self, name: str, help_: str = "") -> Gauge:
        """Register and return a :class:`Gauge`."""
        with self._lock:
            if name not in self._gauges:
                self._gauges[name] = Gauge(name, help_)
            return self._gauges[name]

    def register_histogram(
        self, name: str, help_: str = "", buckets: tuple = ()
    ) -> Histogram:
        """Register and return a :class:`Histogram`."""
        with self._lock:
            if name not in self._histograms:
                self._histograms[name] = Histogram(name, help_, buckets)
            return self._histograms[name]

    # ------------------------------------------------------------------
    # Convenience update methods
    # ------------------------------------------------------------------

    def increment(self, name: str, amount: int = 1) -> None:
        """Increment a registered counter by ``amount``."""
        counter = self._counters.get(name)
        if counter is not None:
            counter.increment(amount)

    def set_gauge(self, name: str, value: float) -> None:
        """Set a registered gauge to ``value``."""
        gauge = self._gauges.get(name)
        if gauge is not None:
            gauge.set(value)

    def observe(self, name: str, value: float) -> None:
        """Record a value in a registered histogram."""
        hist = self._histograms.get(name)
        if hist is not None:
            hist.observe(value)

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def snapshot(self) -> Dict:
        """
        Return a dictionary with all current metric values.

        Returns:
            Dict with keys ``counters``, ``gauges``, ``histograms``,
            and ``uptime_s`` (seconds since the registry was created).
        """
        return {
            "uptime_s": time.time() - self._start_time,
            "counters": {k: v.get() for k, v in self._counters.items()},
            "gauges": {k: v.get() for k, v in self._gauges.items()},
            "histograms": {k: v.snapshot() for k, v in self._histograms.items()},
        }

    def to_prometheus_text(self) -> str:
        """
        Render all metrics in Prometheus exposition format (text/plain v0.0.4).

        Returns:
            Multi-line string suitable for a ``/metrics`` HTTP endpoint.
        """
        lines: List[str] = []

        for name, counter in sorted(self._counters.items()):
            lines.append(f"# HELP {name} {counter.help}")
            lines.append(f"# TYPE {name} counter")
            lines.append(f"{name} {counter.get()}")

        for name, gauge in sorted(self._gauges.items()):
            lines.append(f"# HELP {name} {gauge.help}")
            lines.append(f"# TYPE {name} gauge")
            lines.append(f"{name} {gauge.get()}")

        for name, hist in sorted(self._histograms.items()):
            snap = hist.snapshot()
            lines.append(f"# HELP {name} {hist.help}")
            lines.append(f"# TYPE {name} histogram")
            cumulative = 0
            for i, upper in enumerate(hist._buckets):
                cumulative += hist._counts[i]
                label = "+Inf" if math.isinf(upper) else str(upper)
                lines.append(f'{name}_bucket{{le="{label}"}} {cumulative}')
            lines.append(f"{name}_sum {snap['sum']}")
            lines.append(f"{name}_count {snap['count']}")

        return "\n".join(lines) + "\n"

    def reset_all(self) -> None:
        """Reset all counters, gauges, and histograms (useful for testing)."""
        with self._lock:
            for c in self._counters.values():
                c.reset()
            for g in self._gauges.values():
                g.set(0.0)
            for h in self._histograms.values():
                h._counts = [0] * len(h._buckets)
                h._sum = 0.0
                h._total = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _register_defaults(self) -> None:
        """Pre-register the standard beamforming controller metrics."""
        self.register_counter("decisions_total", "Total beamforming decisions made")
        self.register_counter("handovers_total", "Total satellite handover events")
        self.register_counter("outages_total", "Total outage events detected")
        self.register_counter("fallback_total", "Fallback policy activations")
        self.register_counter("errors_total", "Unhandled errors in the controller")

        self.register_gauge("snr_db", "Last observed SNR in dB")
        self.register_gauge("rain_rate_mmh", "Last observed rain rate in mm/h")
        self.register_gauge("active_satellite_id", "Index of the active satellite")
        self.register_gauge("visible_satellites", "Number of currently visible satellites")

        self.register_histogram(
            "inference_latency_ms",
            "Agent inference latency per decision in milliseconds",
            buckets=(1.0, 2.0, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0, float("inf")),
        )
        self.register_histogram(
            "throughput_mbps",
            "Achieved throughput per decision in Mbps",
            buckets=(0.0, 10.0, 25.0, 50.0, 75.0, 100.0, float("inf")),
        )


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

#: Global metrics registry – import and use anywhere in the codebase.
GLOBAL_REGISTRY: MetricsRegistry = MetricsRegistry()
