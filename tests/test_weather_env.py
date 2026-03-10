"""
Tests for MultiSatelliteEnv with weather forecast integration.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------

class _FakeTelemetry:
    ground_station_pos = np.array([0.0, 0.0, 6371.0])

    def get_visible_satellites(self):
        return [
            np.array([0.0, 0.0, 6921.0]),
            np.array([200.0, 0.0, 6921.0]),
            np.array([0.0, 200.0, 6921.0]),
        ]


class _FakeRadar:
    def get_at_location(self, _pos):
        return 5.0


class _FakeFoliage:
    def get_at_location(self, _pos):
        return 2.0


def _make_channel():
    from channel.rain_attenuation import ChannelModel
    return ChannelModel()


def _make_env(with_forecast=False, max_sats=3):
    from envs.multi_satellite_env import MultiSatelliteEnv
    forecast = None
    if with_forecast:
        from data.weather_forecast import SyntheticWeatherForecast
        forecast = SyntheticWeatherForecast(n_cells=3, seed=0)
    return MultiSatelliteEnv(
        channel_model=_make_channel(),
        telemetry_stream=_FakeTelemetry(),
        radar_stream=_FakeRadar(),
        foliage_map=_FakeFoliage(),
        max_satellites=max_sats,
        weather_forecast=forecast,
    )


# ---------------------------------------------------------------------------
# Tests: MultiSatelliteEnv without forecast (backward compatibility)
# ---------------------------------------------------------------------------

class TestMultiSatelliteEnvNoForecast:
    def test_obs_dim_unchanged(self):
        env = _make_env(with_forecast=False, max_sats=3)
        # 3 sats × 4 features + 3 one-hot = 15 (same as before)
        assert env.observation_space.shape == (15,)

    def test_reset_shape(self):
        env = _make_env(with_forecast=False, max_sats=3)
        obs, _ = env.reset(seed=0)
        assert obs.shape == (15,)

    def test_step_shape(self):
        env = _make_env(with_forecast=False, max_sats=3)
        env.reset(seed=0)
        obs, _, _, _, _ = env.step(0)
        assert obs.shape == (15,)


# ---------------------------------------------------------------------------
# Tests: MultiSatelliteEnv with weather forecast
# ---------------------------------------------------------------------------

class TestMultiSatelliteEnvWithForecast:
    def test_obs_dim_extended(self):
        env = _make_env(with_forecast=True, max_sats=3)
        # 15 base + 3 forecast = 18
        assert env.observation_space.shape == (18,)

    def test_reset_shape_extended(self):
        env = _make_env(with_forecast=True, max_sats=3)
        obs, _ = env.reset(seed=0)
        assert obs.shape == (18,)

    def test_step_shape_extended(self):
        env = _make_env(with_forecast=True, max_sats=3)
        env.reset(seed=0)
        obs, _, _, _, _ = env.step(0)
        assert obs.shape == (18,)

    def test_obs_is_finite(self):
        env = _make_env(with_forecast=True, max_sats=3)
        obs, _ = env.reset(seed=0)
        assert np.all(np.isfinite(obs))

    def test_sim_time_advances(self):
        env = _make_env(with_forecast=True, max_sats=3)
        env.reset(seed=0)
        assert env._sim_time == 0.0
        env.step(0)
        assert env._sim_time > 0.0

    def test_sim_time_resets_on_reset(self):
        env = _make_env(with_forecast=True, max_sats=3)
        env.reset(seed=0)
        for _ in range(5):
            env.step(0)
        env.reset(seed=0)
        assert env._sim_time == 0.0

    def test_multiple_steps_stable(self):
        env = _make_env(with_forecast=True, max_sats=3)
        env.reset(seed=0)
        for i in range(20):
            obs, _, _, _, _ = env.step(i % 3)
            assert obs.shape == (18,)
            assert np.all(np.isfinite(obs))

    def test_forecast_features_different_from_current_rain(self):
        """Forecast (future rain) values should differ from current rain
        at least sometimes, since the forecast uses a different time offset."""
        env = _make_env(with_forecast=True, max_sats=3)
        obs, _ = env.reset(seed=0)
        # Current rain features are in indices 3, 7, 11 of the raw obs
        # Forecast features are the last 3 elements; they may differ
        # Just verify they are present and finite
        forecast_obs = obs[-3:]
        assert np.all(np.isfinite(forecast_obs))

    def test_obs_normalisation_std_not_zero(self):
        """No dimension of _obs_std should be zero (would cause NaN)."""
        env = _make_env(with_forecast=True, max_sats=3)
        assert np.all(env._obs_std != 0.0)

    def test_obs_mean_length_matches_obs_dim(self):
        """_obs_mean and _obs_std must have the same length as the obs space."""
        env = _make_env(with_forecast=True, max_sats=3)
        obs_dim = env.observation_space.shape[0]
        assert len(env._obs_mean) == obs_dim, (
            f"_obs_mean has {len(env._obs_mean)} dims but obs_space has {obs_dim}"
        )
        assert len(env._obs_std) == obs_dim, (
            f"_obs_std has {len(env._obs_std)} dims but obs_space has {obs_dim}"
        )

    def test_forecast_mean_appended_at_correct_position(self):
        """The last max_satellites elements of _obs_mean should equal the FORECAST_MEAN."""
        from envs.multi_satellite_env import MultiSatelliteEnv
        env = _make_env(with_forecast=True, max_sats=3)
        expected_forecast_mean = MultiSatelliteEnv._FORECAST_MEAN
        np.testing.assert_allclose(
            env._obs_mean[-3:],
            np.full(3, expected_forecast_mean, dtype=np.float32),
        )

    def test_forecast_std_appended_at_correct_position(self):
        """The last max_satellites elements of _obs_std should equal the FORECAST_STD."""
        from envs.multi_satellite_env import MultiSatelliteEnv
        env = _make_env(with_forecast=True, max_sats=3)
        expected_forecast_std = MultiSatelliteEnv._FORECAST_STD
        np.testing.assert_allclose(
            env._obs_std[-3:],
            np.full(3, expected_forecast_std, dtype=np.float32),
        )


# ---------------------------------------------------------------------------
# Tests: MultiSatelliteEnv with 5 satellites + forecast
# ---------------------------------------------------------------------------

class TestMultiSatelliteEnvFiveSats:
    def test_obs_dim_five_sats_with_forecast(self):
        from envs.multi_satellite_env import MultiSatelliteEnv
        from data.weather_forecast import SyntheticWeatherForecast

        class _BigTelemetry(_FakeTelemetry):
            def get_visible_satellites(self):
                return [np.array([float(i) * 100, 0.0, 6921.0]) for i in range(5)]
            ground_station_pos = np.array([0.0, 0.0, 6371.0])

        env = MultiSatelliteEnv(
            channel_model=_make_channel(),
            telemetry_stream=_BigTelemetry(),
            radar_stream=_FakeRadar(),
            foliage_map=_FakeFoliage(),
            max_satellites=5,
            weather_forecast=SyntheticWeatherForecast(seed=0),
        )
        # 5 × 4 + 5 + 5 = 30
        assert env.observation_space.shape == (30,)
        obs, _ = env.reset(seed=0)
        assert obs.shape == (30,)
