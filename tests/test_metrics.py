"""
Tests for the Prometheus-compatible metrics module.
"""

import sys
import os
import threading

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest


def _fresh_registry():
    from utils.metrics import MetricsRegistry
    return MetricsRegistry()


class TestCounter:
    def test_import(self):
        from utils.metrics import Counter
        assert Counter is not None

    def test_initial_value_zero(self):
        from utils.metrics import Counter
        c = Counter("test_c")
        assert c.get() == 0

    def test_increment_by_one(self):
        from utils.metrics import Counter
        c = Counter("test_c2")
        c.increment()
        assert c.get() == 1

    def test_increment_by_amount(self):
        from utils.metrics import Counter
        c = Counter("test_c3")
        c.increment(5)
        assert c.get() == 5

    def test_reset(self):
        from utils.metrics import Counter
        c = Counter("test_c4")
        c.increment(10)
        c.reset()
        assert c.get() == 0

    def test_thread_safe(self):
        from utils.metrics import Counter
        c = Counter("test_c5")
        threads = [threading.Thread(target=lambda: c.increment()) for _ in range(100)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert c.get() == 100


class TestGauge:
    def test_initial_value_zero(self):
        from utils.metrics import Gauge
        g = Gauge("test_g")
        assert g.get() == 0.0

    def test_set_value(self):
        from utils.metrics import Gauge
        g = Gauge("test_g2")
        g.set(42.5)
        assert abs(g.get() - 42.5) < 1e-9

    def test_set_negative(self):
        from utils.metrics import Gauge
        g = Gauge("test_g3")
        g.set(-10.0)
        assert abs(g.get() - (-10.0)) < 1e-9


class TestHistogram:
    def test_initial_count_zero(self):
        from utils.metrics import Histogram
        h = Histogram("test_h")
        snap = h.snapshot()
        assert snap["count"] == 0

    def test_observe_increments_count(self):
        from utils.metrics import Histogram
        h = Histogram("test_h2")
        h.observe(5.0)
        h.observe(50.0)
        snap = h.snapshot()
        assert snap["count"] == 2

    def test_sum_correct(self):
        from utils.metrics import Histogram
        h = Histogram("test_h3")
        h.observe(10.0)
        h.observe(20.0)
        snap = h.snapshot()
        assert abs(snap["sum"] - 30.0) < 1e-9

    def test_mean_correct(self):
        from utils.metrics import Histogram
        h = Histogram("test_h4")
        h.observe(10.0)
        h.observe(30.0)
        snap = h.snapshot()
        assert abs(snap["mean"] - 20.0) < 1e-9

    def test_percentiles_present(self):
        from utils.metrics import Histogram
        h = Histogram("test_h5")
        for v in range(1, 101):
            h.observe(float(v))
        snap = h.snapshot()
        assert "p50" in snap
        assert "p95" in snap
        assert "p99" in snap
        assert snap["p50"] >= 1.0


class TestMetricsRegistry:
    def test_default_counters_present(self):
        reg = _fresh_registry()
        snap = reg.snapshot()
        for name in ["decisions_total", "handovers_total", "outages_total"]:
            assert name in snap["counters"]

    def test_default_gauges_present(self):
        reg = _fresh_registry()
        snap = reg.snapshot()
        assert "snr_db" in snap["gauges"]
        assert "rain_rate_mmh" in snap["gauges"]

    def test_default_histograms_present(self):
        reg = _fresh_registry()
        snap = reg.snapshot()
        assert "inference_latency_ms" in snap["histograms"]
        assert "throughput_mbps" in snap["histograms"]

    def test_increment_counter(self):
        reg = _fresh_registry()
        reg.increment("decisions_total")
        reg.increment("decisions_total", 4)
        assert reg.snapshot()["counters"]["decisions_total"] == 5

    def test_set_gauge(self):
        reg = _fresh_registry()
        reg.set_gauge("snr_db", 14.2)
        assert abs(reg.snapshot()["gauges"]["snr_db"] - 14.2) < 1e-9

    def test_observe_histogram(self):
        reg = _fresh_registry()
        reg.observe("inference_latency_ms", 3.5)
        snap = reg.snapshot()
        assert snap["histograms"]["inference_latency_ms"]["count"] == 1

    def test_prometheus_text_format(self):
        reg = _fresh_registry()
        reg.increment("decisions_total", 3)
        text = reg.to_prometheus_text()
        assert "decisions_total" in text
        assert "# TYPE decisions_total counter" in text
        assert "decisions_total 3" in text

    def test_prometheus_text_histogram(self):
        reg = _fresh_registry()
        reg.observe("inference_latency_ms", 10.0)
        text = reg.to_prometheus_text()
        assert "inference_latency_ms_sum" in text
        assert "inference_latency_ms_count 1" in text

    def test_reset_all(self):
        reg = _fresh_registry()
        reg.increment("decisions_total", 10)
        reg.set_gauge("snr_db", 99.0)
        reg.reset_all()
        snap = reg.snapshot()
        assert snap["counters"]["decisions_total"] == 0
        assert snap["gauges"]["snr_db"] == 0.0

    def test_uptime_positive(self):
        reg = _fresh_registry()
        snap = reg.snapshot()
        assert snap["uptime_s"] >= 0.0

    def test_global_registry_singleton(self):
        from utils.metrics import GLOBAL_REGISTRY
        assert GLOBAL_REGISTRY is not None

    def test_exported_from_utils(self):
        from utils import GLOBAL_REGISTRY, MetricsRegistry
        assert GLOBAL_REGISTRY is not None
        assert MetricsRegistry is not None
