"""
Short-term weather nowcasting for the Amazon beamforming system.

Provides:
    - :class:`WeatherForecast` – abstract interface that all forecast
      back-ends must satisfy.
    - :class:`SyntheticWeatherForecast` – fully deterministic, dependency-
      free implementation suitable for unit testing and offline simulation.
      Generates physically plausible spatiotemporal rain-rate fields using
      a superposition of Gaussian rain cells whose intensity and position
      vary with time.
    - :func:`make_forecast` – factory that returns the best available
      back-end (synthetic by default; real API back-end when credentials
      are configured).

The forecast is used by :class:`~envs.multi_satellite_env.MultiSatelliteEnv`
(weather-aware variant) to extend the observation vector with a short-term
rain-rate prediction, enabling the agent to pre-position beams before a
heavy rain event degrades the link.

Rain rates are expressed in mm/h and represent expected values over the
forecast horizon (e.g. 5 minutes).  All spatial coordinates follow the
same ECEF convention used throughout the rest of the codebase (km).

Usage::

    from data.weather_forecast import SyntheticWeatherForecast

    forecast = SyntheticWeatherForecast(seed=42)
    # Get 10-minute rain-rate forecast at a satellite position (km)
    pos = np.array([0.0, 0.0, 6921.0])
    rain_predicted = forecast.get_forecast(pos, t_now=0.0, horizon_s=600.0)
"""

from __future__ import annotations

import abc
from typing import List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------

class WeatherForecast(abc.ABC):
    """
    Abstract interface for short-term weather/rain-rate nowcasting.

    All implementations must provide :meth:`get_forecast`, which returns a
    predicted rain rate (mm/h) at a given location and future time offset.

    Implementations are expected to be cheap to query (< 1 ms per call)
    so that they can be polled inside the real-time inference loop.
    """

    @abc.abstractmethod
    def get_forecast(
        self,
        position: np.ndarray,
        t_now: float,
        horizon_s: float = 300.0,
    ) -> float:
        """
        Return the predicted rain rate at ``position`` at time
        ``t_now + horizon_s``.

        Args:
            position:   ECEF position (x, y, z) in km.
            t_now:      Current simulation time in seconds.
            horizon_s:  Forecast horizon in seconds (default 5 min).

        Returns:
            Predicted rain rate in mm/h (non-negative).
        """

    def get_forecast_vector(
        self,
        positions: List[np.ndarray],
        t_now: float,
        horizon_s: float = 300.0,
    ) -> np.ndarray:
        """
        Vectorised forecast: return predicted rain rate for each position.

        Args:
            positions:  List of ECEF positions in km.
            t_now:      Current simulation time in seconds.
            horizon_s:  Forecast horizon in seconds.

        Returns:
            1-D numpy array of rain rates (mm/h), one per position.
        """
        return np.array(
            [self.get_forecast(p, t_now, horizon_s) for p in positions],
            dtype=np.float32,
        )

    def current_rain_rate(self, position: np.ndarray, t_now: float) -> float:
        """
        Return the *current* rain rate (zero-horizon forecast).

        Alias for :meth:`get_forecast` with ``horizon_s=0``.

        Args:
            position:  ECEF position (km).
            t_now:     Current time (s).

        Returns:
            Current rain rate (mm/h).
        """
        return self.get_forecast(position, t_now, horizon_s=0.0)


# ---------------------------------------------------------------------------
# Synthetic (simulation) implementation
# ---------------------------------------------------------------------------

class SyntheticWeatherForecast(WeatherForecast):
    """
    Deterministic synthetic weather forecast for testing and simulation.

    Models rain as a superposition of ``n_cells`` Gaussian rain cells whose:
        - positions drift slowly with time (convective motion ~5 m/s),
        - intensities oscillate sinusoidally to mimic passing storms,
        - spatial scale is configurable (default 200 km radius).

    The forecast at ``t_now + horizon_s`` is computed by extrapolating each
    cell's position and intensity using simple linear/sinusoidal kinematics.
    No persistence model or data assimilation is performed.

    Args:
        n_cells:     Number of independent rain cells.
        max_rain:    Maximum rain rate at the centre of a cell (mm/h).
        cell_radius_km: Gaussian half-width of each rain cell (km).
        seed:        Random seed for reproducibility.
    """

    _EARTH_RADIUS_KM = 6371.0
    _MOTION_SPEED_KMS = 0.005  # ~5 m/s convective drift

    def __init__(
        self,
        n_cells: int = 5,
        max_rain: float = 50.0,
        cell_radius_km: float = 200.0,
        seed: int = 0,
    ) -> None:
        rng = np.random.default_rng(seed)

        # Cell centres on the Earth's surface (lat, lon in radians)
        lats = rng.uniform(-np.pi / 6, np.pi / 6, n_cells)  # ±30° latitude
        lons = rng.uniform(-np.pi, np.pi, n_cells)
        # ECEF positions (km) on Earth's surface
        self._cell_centres = np.column_stack([
            self._EARTH_RADIUS_KM * np.cos(lats) * np.cos(lons),
            self._EARTH_RADIUS_KM * np.cos(lats) * np.sin(lons),
            self._EARTH_RADIUS_KM * np.sin(lats),
        ])  # shape (n_cells, 3)

        # Cell drift velocities (km/s) – random direction, fixed speed
        directions = rng.uniform(-1.0, 1.0, (n_cells, 3))
        norms = np.linalg.norm(directions, axis=1, keepdims=True).clip(1e-9)
        self._cell_velocities = directions / norms * self._MOTION_SPEED_KMS

        # Peak intensities
        self._peak_rain = rng.uniform(5.0, max_rain, n_cells)

        # Oscillation phases / periods for intensity variation
        self._phase = rng.uniform(0.0, 2 * np.pi, n_cells)
        self._period_s = rng.uniform(600.0, 3600.0, n_cells)  # 10 min – 1 h

        self.cell_radius_km = cell_radius_km
        self.n_cells = n_cells

    # ------------------------------------------------------------------
    # WeatherForecast interface
    # ------------------------------------------------------------------

    def get_forecast(
        self,
        position: np.ndarray,
        t_now: float,
        horizon_s: float = 300.0,
    ) -> float:
        """
        Predict rain rate at ``position`` at time ``t_now + horizon_s``.

        Args:
            position:   ECEF position (km), shape (3,).
            t_now:      Current simulation time (seconds).
            horizon_s:  Forecast horizon (seconds).

        Returns:
            Predicted rain rate (mm/h, ≥ 0).
        """
        t_future = t_now + horizon_s
        pos = np.asarray(position, dtype=np.float64)

        total_rain = 0.0
        for i in range(self.n_cells):
            # Extrapolate cell centre position
            centre = self._cell_centres[i] + self._cell_velocities[i] * t_future

            # Gaussian intensity field
            dist_km = float(np.linalg.norm(pos[:3] - centre))
            spatial = np.exp(-0.5 * (dist_km / self.cell_radius_km) ** 2)

            # Sinusoidal intensity modulation
            phase = self._phase[i] + 2.0 * np.pi * t_future / self._period_s[i]
            intensity = self._peak_rain[i] * max(0.0, np.sin(phase))

            total_rain += intensity * spatial

        return float(np.clip(total_rain, 0.0, 200.0))

    def get_cell_positions(self, t: float) -> np.ndarray:
        """
        Return the positions of all rain cells at time ``t`` (for visualisation).

        Args:
            t: Simulation time (seconds).

        Returns:
            Array of shape ``(n_cells, 3)`` in ECEF km.
        """
        return self._cell_centres + self._cell_velocities * t


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_forecast(
    backend: str = "synthetic",
    seed: int = 0,
    **kwargs,
) -> WeatherForecast:
    """
    Factory function that returns the requested forecast back-end.

    Args:
        backend: ``"synthetic"`` (default) – uses
                 :class:`SyntheticWeatherForecast`.  Additional back-ends
                 (e.g. ``"openweather"``, ``"inmet"``) can be registered
                 here as they are implemented.
        seed:    Random seed (used by synthetic back-end).
        **kwargs: Extra arguments forwarded to the constructor.

    Returns:
        A :class:`WeatherForecast` instance.

    Raises:
        ValueError: If ``backend`` is not recognised.
    """
    if backend == "synthetic":
        return SyntheticWeatherForecast(seed=seed, **kwargs)
    raise ValueError(
        f"Unknown weather forecast backend: '{backend}'. "
        "Available: 'synthetic'."
    )
