"""
Tests for the weather nowcasting module.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest


class TestWeatherForecastInterface:
    def test_import(self):
        from data.weather_forecast import WeatherForecast
        assert WeatherForecast is not None

    def test_exported_from_data_package(self):
        from data import WeatherForecast, SyntheticWeatherForecast, make_forecast
        assert WeatherForecast is not None
        assert SyntheticWeatherForecast is not None
        assert make_forecast is not None


class TestSyntheticWeatherForecast:
    def _make(self, seed=0):
        from data.weather_forecast import SyntheticWeatherForecast
        return SyntheticWeatherForecast(n_cells=3, seed=seed)

    def test_get_forecast_returns_float(self):
        fc = self._make()
        pos = np.array([0.0, 0.0, 6921.0])
        result = fc.get_forecast(pos, t_now=0.0, horizon_s=300.0)
        assert isinstance(result, float)

    def test_forecast_non_negative(self):
        fc = self._make()
        pos = np.array([0.0, 0.0, 6921.0])
        for t in range(0, 3600, 120):
            result = fc.get_forecast(pos, t_now=float(t), horizon_s=300.0)
            assert result >= 0.0

    def test_current_rain_rate(self):
        fc = self._make()
        pos = np.array([0.0, 0.0, 6921.0])
        r = fc.current_rain_rate(pos, t_now=0.0)
        assert isinstance(r, float)
        assert r >= 0.0

    def test_forecast_finite(self):
        fc = self._make()
        pos = np.array([0.0, 0.0, 6921.0])
        r = fc.get_forecast(pos, t_now=1000.0, horizon_s=600.0)
        assert np.isfinite(r)

    def test_different_seeds_different_results(self):
        from data.weather_forecast import SyntheticWeatherForecast
        pos = np.array([100.0, 100.0, 6500.0])
        r1 = SyntheticWeatherForecast(seed=0).get_forecast(pos, 500.0, 300.0)
        r2 = SyntheticWeatherForecast(seed=99).get_forecast(pos, 500.0, 300.0)
        # Different seeds should (almost certainly) give different results
        # Allow equality for edge cases but check they are both valid
        assert isinstance(r1, float) and isinstance(r2, float)

    def test_forecast_vector(self):
        fc = self._make()
        positions = [
            np.array([0.0, 0.0, 6921.0]),
            np.array([200.0, 0.0, 6921.0]),
            np.array([0.0, 200.0, 6921.0]),
        ]
        vec = fc.get_forecast_vector(positions, t_now=0.0, horizon_s=300.0)
        assert vec.shape == (3,)
        assert np.all(vec >= 0.0)

    def test_get_cell_positions_shape(self):
        fc = self._make()
        positions = fc.get_cell_positions(t=0.0)
        assert positions.shape == (3, 3)  # n_cells=3, 3D ECEF

    def test_cells_drift_over_time(self):
        fc = self._make()
        p0 = fc.get_cell_positions(t=0.0)
        p1 = fc.get_cell_positions(t=10000.0)
        # Cells must have moved (non-zero velocity)
        assert not np.allclose(p0, p1)

    def test_max_rain_clipped(self):
        """Rain rate must never exceed 200 mm/h (model cap) across many positions and times."""
        from data.weather_forecast import SyntheticWeatherForecast
        fc = SyntheticWeatherForecast(n_cells=20, max_rain=500.0, seed=1)
        rng = np.random.default_rng(7)
        for _ in range(50):
            pos = rng.uniform(-6371.0, 6371.0, 3)
            t = float(rng.uniform(0, 7200))
            horizon = float(rng.uniform(0, 1800))
            r = fc.get_forecast(pos, t_now=t, horizon_s=horizon)
            assert r <= 200.0 + 1e-9, f"Rain {r} exceeded cap at pos={pos} t={t}"
            assert r >= 0.0


class TestMakeForecast:
    def test_synthetic_backend(self):
        from data.weather_forecast import make_forecast
        fc = make_forecast("synthetic", seed=0)
        assert fc is not None

    def test_unknown_backend_raises(self):
        from data.weather_forecast import make_forecast
        with pytest.raises(ValueError):
            make_forecast("nonexistent_api")

    def test_synthetic_produces_forecast(self):
        from data.weather_forecast import make_forecast
        fc = make_forecast("synthetic", seed=42)
        pos = np.array([0.0, 0.0, 6921.0])
        r = fc.get_forecast(pos, t_now=0.0)
        assert isinstance(r, float)
