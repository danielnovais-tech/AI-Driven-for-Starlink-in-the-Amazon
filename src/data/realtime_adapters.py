"""
Real-time data source adapters for live operational deployment.

Provides three adapters that replace the synthetic sensor generators used
during development and simulation:

1. :class:`CptecRadarAdapter`      – fetches rain-rate grids from the CPTEC/INPE
   radar API (Brazilian National Weather Centre).
2. :class:`SpaceTrackTLEAdapter`   – downloads up-to-date TLE sets for Starlink
   satellites from the Space-Track.org REST API.
3. :class:`NetworkTrafficAdapter`  – reads live traffic metrics (arrival rate,
   queue depth) from a configurable HTTP endpoint.

All three share the same design principles:
    - **Thread-safe ring buffer**: a background thread fetches new data
      periodically; the main thread reads the most recent valid value with
      ``get_*()`` calls that never block.
    - **Configurable fallback**: if the live feed is unavailable (network
      error, timeout, authentication failure), the adapter transparently falls
      back to a synthetic/cached value and logs a warning.
    - **Failure counter**: :attr:`consecutive_failures` tracks how many
      consecutive fetches have failed; callers can check this or set
      :attr:`max_failures` to trigger a circuit-breaker.

Usage::

    from data.realtime_adapters import CptecRadarAdapter, SpaceTrackTLEAdapter

    radar = CptecRadarAdapter(refresh_interval_s=60)
    radar.start()                          # starts background polling thread
    rain = radar.get_at_location([x, y, z])  # non-blocking
    radar.stop()

    tle = SpaceTrackTLEAdapter(username="...", password="...", norad_ids=[...])
    tle.start()
    tle_pairs = tle.get_tle_pairs()        # list of (line1, line2) strings
    tle.stop()

Note on credentials:
    ``SpaceTrackTLEAdapter`` requires valid Space-Track.org credentials.
    Pass them via constructor args or set the environment variables
    ``SPACETRACK_USER`` and ``SPACETRACK_PASS``.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared ring buffer
# ---------------------------------------------------------------------------

class _RingBuffer:
    """
    Thread-safe single-slot buffer (keeps only the latest value).

    Args:
        default: Value returned before the first successful fetch.
    """

    def __init__(self, default: Any) -> None:
        self._value = default
        self._lock = threading.Lock()
        self._updated = False

    def put(self, value: Any) -> None:
        with self._lock:
            self._value = value
            self._updated = True

    def get(self) -> Any:
        with self._lock:
            return self._value

    @property
    def has_been_updated(self) -> bool:
        with self._lock:
            return self._updated


# ---------------------------------------------------------------------------
# Base adapter
# ---------------------------------------------------------------------------

class _BaseRealtimeAdapter:
    """
    Shared infrastructure for background-polling real-time adapters.

    Subclasses must implement :meth:`_fetch` which returns fresh data.
    The base class manages the background thread, failure counting, and
    the ring buffer.

    Args:
        refresh_interval_s: Polling interval in seconds.
        max_failures:       Number of consecutive fetch failures before
                            the adapter stops retrying automatically.
                            Set to ``0`` to disable the circuit-breaker.
        timeout_s:          HTTP / network timeout (seconds).
    """

    def __init__(
        self,
        refresh_interval_s: float = 30.0,
        max_failures: int = 10,
        timeout_s: float = 10.0,
    ) -> None:
        self.refresh_interval_s = refresh_interval_s
        self.max_failures = max_failures
        self.timeout_s = timeout_s
        self.consecutive_failures: int = 0
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    # ------------------------------------------------------------------
    # Public lifecycle API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the background polling thread."""
        if self._thread is not None and self._thread.is_alive():
            return  # already running
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._polling_loop, daemon=True, name=f"{type(self).__name__}_bg"
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop the background polling thread and wait for it to exit."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=max(2.0, self.refresh_interval_s))
            self._thread = None

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ------------------------------------------------------------------
    # Subclass hooks
    # ------------------------------------------------------------------

    def _fetch(self) -> Any:
        """Fetch fresh data from the remote source.  Must be overridden."""
        raise NotImplementedError

    def _on_success(self, data: Any) -> None:
        """Called after a successful _fetch(); store data in buffer."""
        raise NotImplementedError

    def _on_failure(self, exc: Exception) -> None:
        """Called after a failed _fetch(); log warning."""
        self.consecutive_failures += 1
        logger.warning(
            "%s fetch failed (%d consecutive): %s",
            type(self).__name__, self.consecutive_failures, exc,
        )
        if self.max_failures > 0 and self.consecutive_failures >= self.max_failures:
            logger.error(
                "%s circuit-breaker opened after %d failures; stopping.",
                type(self).__name__, self.consecutive_failures,
            )
            self._stop_event.set()

    # ------------------------------------------------------------------
    # Background thread
    # ------------------------------------------------------------------

    def _polling_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                data = self._fetch()
                self.consecutive_failures = 0
                self._on_success(data)
            except Exception as exc:  # noqa: BLE001
                self._on_failure(exc)
            self._stop_event.wait(timeout=self.refresh_interval_s)


# ---------------------------------------------------------------------------
# CPTEC / INPE Radar Adapter
# ---------------------------------------------------------------------------

class CptecRadarAdapter(_BaseRealtimeAdapter):
    """
    Adapter for the CPTEC/INPE radar rain-rate API.

    Fetches the latest gridded rain-rate product from the CPTEC REST endpoint
    (``https://climanalise.cptec.inpe.br/``) and stores it in a ring buffer.
    Calls to :meth:`get_at_location` are always non-blocking.

    Fallback behaviour:
        If the API is unreachable, the last successfully fetched grid is
        retained.  Before the first successful fetch (or if the API has never
        responded), a synthetic constant rain-rate grid is returned.

    Args:
        base_url:            CPTEC API base URL.
        product:             Radar product identifier (default ``"MERGE_1H"``).
        amazon_lat_range:    (lat_min, lat_max) bounding box (degrees).
        amazon_lon_range:    (lon_min, lon_max) bounding box (degrees).
        resolution_deg:      Grid cell size in degrees (default 0.1°).
        synthetic_rain_mmh:  Fallback rain rate (mm/h) used before first fetch.
        refresh_interval_s:  How often to poll the API (seconds).
        timeout_s:           HTTP request timeout (seconds).
    """

    _DEFAULT_BASE_URL = "https://climanalise.cptec.inpe.br/~rclimatol/products/radar"

    def __init__(
        self,
        base_url: str = _DEFAULT_BASE_URL,
        product: str = "MERGE_1H",
        amazon_lat_range: Tuple[float, float] = (-15.0, 5.0),
        amazon_lon_range: Tuple[float, float] = (-75.0, -45.0),
        resolution_deg: float = 0.1,
        synthetic_rain_mmh: float = 5.0,
        refresh_interval_s: float = 300.0,  # 5 minutes
        timeout_s: float = 15.0,
        max_failures: int = 10,
    ) -> None:
        super().__init__(
            refresh_interval_s=refresh_interval_s, timeout_s=timeout_s,
            max_failures=max_failures,
        )
        self.base_url = base_url
        self.product = product
        self.amazon_lat_range = amazon_lat_range
        self.amazon_lon_range = amazon_lon_range
        self.resolution_deg = resolution_deg

        # Build synthetic fallback grid
        n_lat = max(1, int((amazon_lat_range[1] - amazon_lat_range[0]) / resolution_deg))
        n_lon = max(1, int((amazon_lon_range[1] - amazon_lon_range[0]) / resolution_deg))
        self._lats = np.linspace(amazon_lat_range[0], amazon_lat_range[1], n_lat)
        self._lons = np.linspace(amazon_lon_range[0], amazon_lon_range[1], n_lon)
        fallback_grid = np.full((n_lat, n_lon), synthetic_rain_mmh, dtype=np.float32)
        self._buf: _RingBuffer = _RingBuffer(fallback_grid)

    # ------------------------------------------------------------------
    # Public data access
    # ------------------------------------------------------------------

    def get_at_location(self, ecef_pos_km: np.ndarray) -> float:
        """
        Return the rain rate (mm/h) at the given ECEF position.

        Converts the ECEF position to lat/lon and performs bilinear
        interpolation on the buffered rain-rate grid.

        Args:
            ecef_pos_km: Satellite or ground position (km), shape (3,).

        Returns:
            Rain rate in mm/h.
        """
        pos = np.asarray(ecef_pos_km, dtype=np.float64)
        r = float(np.linalg.norm(pos))
        if r < 1e-3:
            return 0.0
        lat = float(np.degrees(np.arcsin(np.clip(pos[2] / r, -1.0, 1.0))))
        lon = float(np.degrees(np.arctan2(pos[1], pos[0])))
        return self._interpolate(lat, lon)

    @property
    def latest_grid(self) -> np.ndarray:
        """Return the most recent rain-rate grid (lat × lon, mm/h)."""
        return self._buf.get()

    # ------------------------------------------------------------------
    # _BaseRealtimeAdapter hooks
    # ------------------------------------------------------------------

    def _fetch(self) -> np.ndarray:
        """
        Download the latest MERGE_1H rain product.

        This implementation attempts a JSON API call; on failure it falls
        through to the ring buffer's existing value.

        Returns:
            Grid array (lat × lon) of rain rates in mm/h.

        Raises:
            RuntimeError: On any network or parsing error.
        """
        try:
            import urllib.request
            import json as _json
            url = f"{self.base_url}/{self.product}/latest.json"
            req = urllib.request.Request(url)
            req.add_header("Accept", "application/json")
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                payload = _json.loads(resp.read().decode())
            # Expected payload: {"rain_rate": [[...]], "lat": [...], "lon": [...]}
            rain = np.array(payload["rain_rate"], dtype=np.float32)
            return rain
        except Exception as exc:
            raise RuntimeError(f"CPTEC fetch error: {exc}") from exc

    def _on_success(self, data: np.ndarray) -> None:
        self._buf.put(data)
        # Update lat/lon arrays to match the received grid dimensions so that
        # _interpolate() uses correct index mapping if the API returns a
        # different resolution than the synthetic fallback.
        n_lat, n_lon = data.shape
        if n_lat != len(self._lats) or n_lon != len(self._lons):
            self._lats = np.linspace(
                self.amazon_lat_range[0], self.amazon_lat_range[1], n_lat
            )
            self._lons = np.linspace(
                self.amazon_lon_range[0], self.amazon_lon_range[1], n_lon
            )
        logger.info("CptecRadarAdapter: rain grid updated, shape=%s", data.shape)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _interpolate(self, lat: float, lon: float) -> float:
        """Nearest-neighbour lookup in the buffered rain grid."""
        grid = self._buf.get()
        lat_idx = int(np.argmin(np.abs(self._lats - lat)))
        lon_idx = int(np.argmin(np.abs(self._lons - lon)))
        lat_idx = max(0, min(lat_idx, grid.shape[0] - 1))
        lon_idx = max(0, min(lon_idx, grid.shape[1] - 1))
        return float(grid[lat_idx, lon_idx])


# ---------------------------------------------------------------------------
# Space-Track TLE Adapter
# ---------------------------------------------------------------------------

class SpaceTrackTLEAdapter(_BaseRealtimeAdapter):
    """
    Adapter for the Space-Track.org REST API (TLE downloads).

    Authenticates with Space-Track, downloads the latest TLE set for the
    specified NORAD catalogue IDs, and stores them in a ring buffer.

    Credentials:
        Set ``username`` and ``password`` explicitly, or export
        ``SPACETRACK_USER`` and ``SPACETRACK_PASS`` environment variables.

    Args:
        norad_ids:          List of NORAD IDs to track (e.g. ``[44713, 44714]``
                            for the first Starlink batch).
        username:           Space-Track.org username.
        password:           Space-Track.org password.
        refresh_interval_s: How often to refresh TLEs (seconds).  TLEs are
                            valid for ~hours; 3600 s is a sensible default.
        timeout_s:          HTTP timeout (seconds).
    """

    _BASE_URL = "https://www.space-track.org"
    _AUTH_URL = _BASE_URL + "/ajaxauth/login"
    _TLE_URL = _BASE_URL + "/basicspacedata/query/class/gp/NORAD_CAT_ID/{ids}/format/tle"

    def __init__(
        self,
        norad_ids: Optional[List[int]] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        refresh_interval_s: float = 3600.0,
        timeout_s: float = 30.0,
        max_failures: int = 10,
    ) -> None:
        super().__init__(
            refresh_interval_s=refresh_interval_s, timeout_s=timeout_s,
            max_failures=max_failures,
        )
        self.norad_ids: List[int] = norad_ids or []
        self.username = username or os.environ.get("SPACETRACK_USER", "")
        self.password = password or os.environ.get("SPACETRACK_PASS", "")
        self._buf: _RingBuffer = _RingBuffer([])  # list of (line1, line2)

    # ------------------------------------------------------------------
    # Public data access
    # ------------------------------------------------------------------

    def get_tle_pairs(self) -> List[Tuple[str, str]]:
        """
        Return the latest list of (TLE line 1, TLE line 2) tuples.

        Returns an empty list before the first successful fetch if no
        cached data is available.
        """
        return self._buf.get()

    # ------------------------------------------------------------------
    # _BaseRealtimeAdapter hooks
    # ------------------------------------------------------------------

    def _fetch(self) -> List[Tuple[str, str]]:
        """
        Authenticate and download TLEs from Space-Track.

        Returns:
            List of (line1, line2) TLE string pairs.

        Raises:
            RuntimeError: On authentication or download failure.
        """
        if not self.username or not self.password:
            raise RuntimeError("Space-Track credentials not configured.")
        if not self.norad_ids:
            raise RuntimeError("No NORAD IDs specified.")

        try:
            import urllib.request
            import urllib.parse
            # Authenticate (cookie-based session)
            auth_data = urllib.parse.urlencode({
                "identity": self.username,
                "password": self.password,
            }).encode()
            auth_req = urllib.request.Request(self._AUTH_URL, data=auth_data)
            # Build cookie jar
            import http.cookiejar
            jar = http.cookiejar.CookieJar()
            opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))
            opener.open(auth_req, timeout=self.timeout_s)

            ids_str = ",".join(str(i) for i in self.norad_ids)
            tle_url = self._TLE_URL.format(ids=ids_str)
            with opener.open(tle_url, timeout=self.timeout_s) as resp:
                raw = resp.read().decode("utf-8", errors="replace")

            return self._parse_tle_text(raw)
        except Exception as exc:
            raise RuntimeError(f"Space-Track fetch error: {exc}") from exc

    def _on_success(self, data: List[Tuple[str, str]]) -> None:
        self._buf.put(data)
        logger.info("SpaceTrackTLEAdapter: %d TLE pairs updated.", len(data))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_tle_text(text: str) -> List[Tuple[str, str]]:
        """Parse a multi-TLE text block into (line1, line2) pairs."""
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        pairs: List[Tuple[str, str]] = []
        i = 0
        while i + 1 < len(lines):
            # Accept both 2-line (no name) and 3-line (name + 2 lines) format
            if lines[i].startswith("1 ") and lines[i + 1].startswith("2 "):
                pairs.append((lines[i], lines[i + 1]))
                i += 2
            else:
                # Skip name line
                i += 1
        return pairs


# ---------------------------------------------------------------------------
# Network Traffic Adapter
# ---------------------------------------------------------------------------

class NetworkTrafficAdapter(_BaseRealtimeAdapter):
    """
    Adapter that reads live traffic metrics from a configurable HTTP endpoint.

    The endpoint is expected to return a JSON payload with at least:
        ``arrival_rate_mbps``  – current average packet arrival rate (Mbps).
        ``queue_depth_mbps``   – current queue occupancy (Mbps).

    Fallback behaviour:
        If the endpoint is unreachable, the last valid reading is retained.
        Before the first successful fetch, synthetic default values are used.

    Args:
        endpoint_url:        URL of the traffic metrics endpoint.
        arrival_rate_key:    JSON key for the arrival rate field.
        queue_depth_key:     JSON key for the queue depth field.
        default_arrival_mbps: Fallback arrival rate (Mbps).
        default_queue_mbps:  Fallback queue depth (Mbps).
        refresh_interval_s:  Polling interval (seconds).
        timeout_s:           HTTP timeout (seconds).
    """

    def __init__(
        self,
        endpoint_url: str = "http://localhost:9091/traffic",
        arrival_rate_key: str = "arrival_rate_mbps",
        queue_depth_key: str = "queue_depth_mbps",
        default_arrival_mbps: float = 50.0,
        default_queue_mbps: float = 0.0,
        refresh_interval_s: float = 5.0,
        timeout_s: float = 5.0,
        max_failures: int = 10,
    ) -> None:
        super().__init__(
            refresh_interval_s=refresh_interval_s, timeout_s=timeout_s,
            max_failures=max_failures,
        )
        self.endpoint_url = endpoint_url
        self.arrival_rate_key = arrival_rate_key
        self.queue_depth_key = queue_depth_key
        _default: Dict[str, float] = {
            arrival_rate_key: default_arrival_mbps,
            queue_depth_key: default_queue_mbps,
        }
        self._buf: _RingBuffer = _RingBuffer(_default)

    # ------------------------------------------------------------------
    # Public data access
    # ------------------------------------------------------------------

    @property
    def arrival_rate_mbps(self) -> float:
        """Return the latest arrival rate (Mbps), or the synthetic default."""
        return float(self._buf.get().get(self.arrival_rate_key, 50.0))

    @property
    def queue_depth_mbps(self) -> float:
        """Return the latest queue depth (Mbps), or the synthetic default."""
        return float(self._buf.get().get(self.queue_depth_key, 0.0))

    def get_metrics(self) -> Dict[str, float]:
        """Return the full latest metrics dictionary."""
        return dict(self._buf.get())

    # ------------------------------------------------------------------
    # _BaseRealtimeAdapter hooks
    # ------------------------------------------------------------------

    def _fetch(self) -> Dict[str, float]:
        import urllib.request
        import json as _json
        req = urllib.request.Request(self.endpoint_url)
        req.add_header("Accept", "application/json")
        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
            return _json.loads(resp.read().decode())

    def _on_success(self, data: Dict[str, float]) -> None:
        self._buf.put(data)
        logger.debug(
            "NetworkTrafficAdapter: arrival=%.1f Mbps, queue=%.1f Mbps",
            data.get(self.arrival_rate_key, 0.0),
            data.get(self.queue_depth_key, 0.0),
        )
