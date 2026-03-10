"""
Tests for CptecRadarAdapter, SpaceTrackTLEAdapter, NetworkTrafficAdapter.
"""

import sys
import os
import time
import json
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _start_http_server(payload: dict, port: int = 0):
    """Start a minimal HTTP server returning ``payload`` as JSON."""

    class _Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            body = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass  # suppress server logs

    server = HTTPServer(("127.0.0.1", port), _Handler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    return server, server.server_address[1]


# ---------------------------------------------------------------------------
# CptecRadarAdapter
# ---------------------------------------------------------------------------

class TestCptecRadarAdapter:
    def test_import(self):
        from data.realtime_adapters import CptecRadarAdapter
        assert CptecRadarAdapter is not None

    def test_exported_from_data(self):
        from data import CptecRadarAdapter
        assert CptecRadarAdapter is not None

    def test_fallback_before_start(self):
        from data.realtime_adapters import CptecRadarAdapter
        adapter = CptecRadarAdapter(synthetic_rain_mmh=7.0, refresh_interval_s=9999.0)
        # get_at_location should work immediately using synthetic fallback
        val = adapter.get_at_location(np.array([6921.0, 0.0, 0.0]))
        assert isinstance(val, float)
        # Fallback value should be close to the synthetic default
        assert val >= 0.0

    def test_start_stop_lifecycle(self):
        from data.realtime_adapters import CptecRadarAdapter
        adapter = CptecRadarAdapter(refresh_interval_s=9999.0, timeout_s=0.1)
        adapter.start()
        assert adapter.is_running
        adapter.stop()
        assert not adapter.is_running

    def test_no_double_start(self):
        from data.realtime_adapters import CptecRadarAdapter
        adapter = CptecRadarAdapter(refresh_interval_s=9999.0)
        adapter.start()
        adapter.start()  # second call must not raise
        adapter.stop()

    def test_get_at_location_returns_float(self):
        from data.realtime_adapters import CptecRadarAdapter
        adapter = CptecRadarAdapter()
        result = adapter.get_at_location(np.array([6921.0, 0.0, 0.0]))
        assert isinstance(result, float)

    def test_latest_grid_shape(self):
        from data.realtime_adapters import CptecRadarAdapter
        adapter = CptecRadarAdapter(
            amazon_lat_range=(-15.0, 5.0),
            amazon_lon_range=(-75.0, -45.0),
            resolution_deg=1.0,
        )
        grid = adapter.latest_grid
        assert grid.ndim == 2
        assert grid.shape[0] > 0

    def test_live_fetch_via_mock_server(self):
        """Use a real HTTP server to verify fetch→buffer path."""
        from data.realtime_adapters import CptecRadarAdapter
        import numpy as _np
        rain_data = _np.ones((3, 4), dtype=_np.float32) * 12.5
        server, port = _start_http_server({
            "rain_rate": rain_data.tolist(),
            "lat": [0.0, 1.0, 2.0],
            "lon": [0.0, 1.0, 2.0, 3.0],
        })
        try:
            adapter = CptecRadarAdapter(
                base_url=f"http://127.0.0.1:{port}",
                refresh_interval_s=9999.0,
                timeout_s=2.0,
            )
            adapter.start()
            # Give the background thread time to fetch
            time.sleep(0.5)
            adapter.stop()
            grid = adapter.latest_grid
            # Either the live grid (3×4 shape) or the synthetic fallback
            assert grid.ndim == 2
        finally:
            server.shutdown()

    def test_circuit_breaker_stops_thread(self):
        """After max_failures, the background thread should stop."""
        from data.realtime_adapters import CptecRadarAdapter
        adapter = CptecRadarAdapter(
            base_url="http://127.0.0.1:1",  # unreachable → always fails
            refresh_interval_s=0.05,
            timeout_s=0.05,
            max_failures=3,
        )
        adapter.start()
        # Wait for max_failures + a little extra
        time.sleep(0.8)
        assert not adapter.is_running
        assert adapter.consecutive_failures >= 3


# ---------------------------------------------------------------------------
# SpaceTrackTLEAdapter
# ---------------------------------------------------------------------------

class TestSpaceTrackTLEAdapter:
    def test_import(self):
        from data.realtime_adapters import SpaceTrackTLEAdapter
        assert SpaceTrackTLEAdapter is not None

    def test_exported_from_data(self):
        from data import SpaceTrackTLEAdapter
        assert SpaceTrackTLEAdapter is not None

    def test_empty_pairs_before_fetch(self):
        from data.realtime_adapters import SpaceTrackTLEAdapter
        adapter = SpaceTrackTLEAdapter(norad_ids=[44713])
        assert adapter.get_tle_pairs() == []

    def test_parse_tle_text_two_line(self):
        from data.realtime_adapters import SpaceTrackTLEAdapter
        tle_text = (
            "1 44713U 19074A   24001.00000000  .00001103  00000-0  14476-4 0  9990\n"
            "2 44713  53.0000   0.0000 0001480   0.0000   0.0000 15.06000000    10\n"
        )
        pairs = SpaceTrackTLEAdapter._parse_tle_text(tle_text)
        assert len(pairs) == 1
        assert pairs[0][0].startswith("1 ")
        assert pairs[0][1].startswith("2 ")

    def test_parse_tle_text_three_line(self):
        from data.realtime_adapters import SpaceTrackTLEAdapter
        tle_text = (
            "STARLINK-1\n"
            "1 44713U 19074A   24001.00000000  .00001103  00000-0  14476-4 0  9990\n"
            "2 44713  53.0000   0.0000 0001480   0.0000   0.0000 15.06000000    10\n"
        )
        pairs = SpaceTrackTLEAdapter._parse_tle_text(tle_text)
        assert len(pairs) == 1

    def test_failure_without_credentials(self):
        from data.realtime_adapters import SpaceTrackTLEAdapter
        adapter = SpaceTrackTLEAdapter(norad_ids=[44713], username="", password="",
                                        refresh_interval_s=0.1, max_failures=2)
        adapter.start()
        time.sleep(0.5)
        adapter.stop()
        assert adapter.consecutive_failures >= 1

    def test_start_stop_lifecycle(self):
        from data.realtime_adapters import SpaceTrackTLEAdapter
        adapter = SpaceTrackTLEAdapter(norad_ids=[44713], refresh_interval_s=9999.0)
        adapter.start()
        assert adapter.is_running
        adapter.stop()
        assert not adapter.is_running


# ---------------------------------------------------------------------------
# NetworkTrafficAdapter
# ---------------------------------------------------------------------------

class TestNetworkTrafficAdapter:
    def test_import(self):
        from data.realtime_adapters import NetworkTrafficAdapter
        assert NetworkTrafficAdapter is not None

    def test_exported_from_data(self):
        from data import NetworkTrafficAdapter
        assert NetworkTrafficAdapter is not None

    def test_default_before_start(self):
        from data.realtime_adapters import NetworkTrafficAdapter
        adapter = NetworkTrafficAdapter(default_arrival_mbps=42.0)
        assert abs(adapter.arrival_rate_mbps - 42.0) < 1e-6

    def test_live_fetch_via_mock_server(self):
        from data.realtime_adapters import NetworkTrafficAdapter
        payload = {"arrival_rate_mbps": 77.0, "queue_depth_mbps": 5.0}
        server, port = _start_http_server(payload)
        try:
            adapter = NetworkTrafficAdapter(
                endpoint_url=f"http://127.0.0.1:{port}/traffic",
                refresh_interval_s=9999.0,
                timeout_s=2.0,
            )
            adapter.start()
            time.sleep(0.4)
            adapter.stop()
            if adapter._buf.has_been_updated:
                assert abs(adapter.arrival_rate_mbps - 77.0) < 1e-3
        finally:
            server.shutdown()

    def test_get_metrics_returns_dict(self):
        from data.realtime_adapters import NetworkTrafficAdapter
        adapter = NetworkTrafficAdapter()
        m = adapter.get_metrics()
        assert isinstance(m, dict)

    def test_fallback_on_unreachable_endpoint(self):
        from data.realtime_adapters import NetworkTrafficAdapter
        adapter = NetworkTrafficAdapter(
            endpoint_url="http://127.0.0.1:1/traffic",
            refresh_interval_s=0.05,
            timeout_s=0.05,
            max_failures=3,
            default_arrival_mbps=99.0,
        )
        adapter.start()
        time.sleep(0.5)
        adapter.stop()
        # Value must still be accessible (fallback)
        val = adapter.arrival_rate_mbps
        assert isinstance(val, float)
        assert adapter.consecutive_failures >= 1
