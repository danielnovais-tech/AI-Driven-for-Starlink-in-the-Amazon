"""
Tests for MultiSatelliteEnv.
"""

import sys
import os
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------

class _FakeTelemetryMulti:
    """Provides 3 static visible satellites."""

    ground_station_pos = np.array([0.0, 0.0, 6371.0])  # surface of Earth (approx)

    def get_visible_satellites(self):
        return [
            np.array([0.0, 0.0, 6921.0]),    # ~550 km above Earth centre
            np.array([200.0, 0.0, 6921.0]),
            np.array([0.0, 200.0, 6921.0]),
        ]


class _FakeRadarMulti:
    def get_at_location(self, _pos):
        return 5.0


class _FakeFoliageMulti:
    def get_at_location(self, _pos):
        return 2.0


def _make_channel():
    from channel.rain_attenuation import ChannelModel
    return ChannelModel()


def _make_env(max_sats=3):
    from envs.multi_satellite_env import MultiSatelliteEnv
    return MultiSatelliteEnv(
        channel_model=_make_channel(),
        telemetry_stream=_FakeTelemetryMulti(),
        radar_stream=_FakeRadarMulti(),
        foliage_map=_FakeFoliageMulti(),
        max_satellites=max_sats,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestMultiSatelliteEnv:
    def test_import(self):
        from envs.multi_satellite_env import MultiSatelliteEnv
        assert MultiSatelliteEnv is not None

    def test_obs_space_dim(self):
        env = _make_env(max_sats=3)
        # 3 sats * 4 features + 3 one-hot = 15
        assert env.observation_space.shape == (15,)

    def test_action_space(self):
        env = _make_env(max_sats=3)
        assert env.action_space.n == 3

    def test_reset_obs_shape(self):
        env = _make_env(max_sats=3)
        obs, info = env.reset(seed=0)
        assert obs.shape == (15,)
        assert isinstance(info, dict)

    def test_step_returns_correct_types(self):
        env = _make_env(max_sats=3)
        env.reset(seed=0)
        obs, reward, terminated, truncated, info = env.step(0)
        assert obs.shape == (15,)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert "throughput" in info
        assert "outage" in info
        assert "handover" in info

    def test_handover_flag_true_on_switch(self):
        env = _make_env(max_sats=3)
        env.reset(seed=0)
        # Force current satellite to index 0
        env.current_sat_idx = 0
        _, _, _, _, info = env.step(1)  # switch to satellite 1
        assert info["handover"] is True

    def test_handover_flag_false_on_same(self):
        env = _make_env(max_sats=3)
        env.reset(seed=0)
        env.current_sat_idx = 0
        _, _, _, _, info = env.step(0)  # stay on satellite 0
        assert info["handover"] is False

    def test_reward_lower_with_handover(self):
        """Handover penalty should reduce the reward."""
        env = _make_env(max_sats=3)
        env.reset(seed=0)
        env.current_sat_idx = 0
        _, r_same, _, _, _ = env.step(0)

        env.reset(seed=0)
        env.current_sat_idx = 0
        _, r_switch, _, _, _ = env.step(1)

        assert r_same > r_switch

    def test_multiple_steps(self):
        env = _make_env(max_sats=3)
        env.reset(seed=0)
        for i in range(10):
            obs, _, _, _, _ = env.step(i % 3)
            assert obs.shape == (15,)

    def test_env_exported_from_package(self):
        from envs import MultiSatelliteEnv
        assert MultiSatelliteEnv is not None

    def test_max_sats_larger_than_visible(self):
        """max_satellites can exceed actual visible satellites (padding with defaults)."""
        env = _make_env(max_sats=5)  # 3 visible, 5 slots
        obs, _ = env.reset(seed=0)
        assert obs.shape == (5 * 4 + 5,)  # 5 sats × 4 features + 5 one-hot = 25
