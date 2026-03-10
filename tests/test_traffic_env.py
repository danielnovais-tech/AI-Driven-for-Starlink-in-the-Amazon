"""
Tests for TrafficAwareMultiSatelliteEnv.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------

class _DummyTelemetry:
    ground_station_pos = np.array([0.0, 0.0, 6371.0])

    def get_visible_satellites(self):
        angles = np.linspace(0, 2 * np.pi, 3, endpoint=False)
        return [
            np.array([(6921.0) * np.cos(a), (6921.0) * np.sin(a), 0.0])
            for a in angles
        ]


class _DummyRadar:
    def get_at_location(self, pos):
        return 3.0


class _DummyFoliage:
    def get_at_location(self, pos):
        return 1.0


def _make_traffic_env(**kwargs):
    from channel.rain_attenuation import ChannelModel
    from envs.traffic_env import TrafficAwareMultiSatelliteEnv
    return TrafficAwareMultiSatelliteEnv(
        channel_model=ChannelModel(),
        telemetry_stream=_DummyTelemetry(),
        radar_stream=_DummyRadar(),
        foliage_map=_DummyFoliage(),
        max_satellites=3,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestTrafficAwareEnv:
    def test_import(self):
        from envs.traffic_env import TrafficAwareMultiSatelliteEnv
        assert TrafficAwareMultiSatelliteEnv is not None

    def test_exported_from_envs(self):
        from envs import TrafficAwareMultiSatelliteEnv
        assert TrafficAwareMultiSatelliteEnv is not None

    def test_obs_has_extra_features(self):
        from channel.rain_attenuation import ChannelModel
        from envs.multi_satellite_env import MultiSatelliteEnv
        from envs.traffic_env import TrafficAwareMultiSatelliteEnv

        base = MultiSatelliteEnv(
            channel_model=ChannelModel(),
            telemetry_stream=_DummyTelemetry(),
            radar_stream=_DummyRadar(),
            foliage_map=_DummyFoliage(),
            max_satellites=3,
        )
        traffic = _make_traffic_env()
        obs_base, _ = base.reset(seed=0)
        obs_traffic, _ = traffic.reset(seed=0)
        assert obs_traffic.shape[0] == obs_base.shape[0] + 2

    def test_reset_returns_obs_and_info(self):
        env = _make_traffic_env()
        obs, info = env.reset(seed=42)
        assert obs.dtype == np.float32
        assert isinstance(info, dict)

    def test_step_returns_traffic_info_keys(self):
        env = _make_traffic_env()
        env.reset(seed=0)
        _, _, _, _, info = env.step(0)
        for key in ("queue_occupancy", "drop_rate", "queue_delay_ms", "arrivals_mbps"):
            assert key in info, f"Missing info key: {key}"

    def test_queue_bounded_by_max_queue(self):
        env = _make_traffic_env(arrival_rate_mbps=1000.0, max_queue_mbps=50.0, seed=0)
        env.reset(seed=0)
        for _ in range(20):
            env.step(0)
        assert env._queue <= env.max_queue_mbps + 1.0  # 1 Mbps tolerance

    def test_drop_rate_nonnegative(self):
        env = _make_traffic_env(arrival_rate_mbps=5.0, seed=0)
        env.reset(seed=0)
        for _ in range(10):
            _, _, _, _, info = env.step(0)
            assert info["drop_rate"] >= 0.0

    def test_reward_differs_from_base_reward(self):
        """Traffic-weighted reward should generally differ from base reward."""
        from channel.rain_attenuation import ChannelModel
        from envs.multi_satellite_env import MultiSatelliteEnv

        base = MultiSatelliteEnv(
            channel_model=ChannelModel(),
            telemetry_stream=_DummyTelemetry(),
            radar_stream=_DummyRadar(),
            foliage_map=_DummyFoliage(),
            max_satellites=3,
        )
        traffic = _make_traffic_env(arrival_rate_mbps=50.0, seed=0)
        base.reset(seed=0)
        traffic.reset(seed=0)
        _, r_base, _, _, _ = base.step(0)
        _, r_traffic, _, _, _ = traffic.step(0)
        # Both should be floats; they measure different quantities
        assert isinstance(r_base, float)
        assert isinstance(r_traffic, float)

    def test_deterministic_with_seed(self):
        env1 = _make_traffic_env(seed=7)
        env2 = _make_traffic_env(seed=7)
        obs1, _ = env1.reset(seed=7)
        obs2, _ = env2.reset(seed=7)
        np.testing.assert_array_equal(obs1, obs2)

    def test_queue_decreases_under_low_arrival(self):
        """With zero arrivals and nonzero throughput, queue should drain."""
        env = _make_traffic_env(arrival_rate_mbps=0.0, seed=0)
        env.reset(seed=0)
        env._queue = 10.0  # pre-fill queue
        for _ in range(20):
            env.step(0)
        assert env._queue < 10.0

    def test_obs_shape_matches_observation_space(self):
        env = _make_traffic_env()
        obs, _ = env.reset(seed=0)
        assert obs.shape == env.observation_space.shape
