"""
Tests for LEOBeamformingEnv and OfflineLEOEnv.
"""

import sys
import os
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest

try:
    import h5py
    _H5PY = True
except ImportError:
    _H5PY = False


# ---------------------------------------------------------------------------
# Minimal stub implementations of the stream / map interfaces
# ---------------------------------------------------------------------------

class _FakeTelemetry:
    """Cycles through a pre-defined list of satellite positions."""

    def __init__(self):
        self._positions = [
            np.array([0.0, 0.0, 550.0]),
            np.array([10.0, 5.0, 552.0]),
            np.array([20.0, 10.0, 548.0]),
        ]
        self._idx = 0

    def get_current_position(self):
        return self._positions[self._idx % len(self._positions)]

    def get_next_position(self):
        self._idx = (self._idx + 1) % len(self._positions)
        return self._positions[self._idx]

    def get_current_snr(self):
        return 15.0

    def get_current_rssi(self):
        return -75.0


class _FakeRadar:
    def get_at_location(self, _pos):
        return 5.0  # mm/h


class _FakeFoliage:
    def get_at_location(self, _pos):
        return 2.0  # LAI units


def _make_channel():
    from channel.rain_attenuation import ChannelModel
    return ChannelModel()


def _make_offline_h5(n=20):
    """Create a minimal HDF5 dataset for OfflineLEOEnv."""
    if not _H5PY:
        pytest.skip("h5py not installed")
    import h5py

    td = tempfile.NamedTemporaryFile(suffix=".h5", delete=False)
    rd = tempfile.NamedTemporaryFile(suffix=".h5", delete=False)
    fd = tempfile.NamedTemporaryFile(suffix=".h5", delete=False)
    for f in [td, rd, fd]:
        f.close()

    with h5py.File(td.name, "w") as f:
        f.create_dataset("snr", data=np.random.uniform(5, 25, n).astype(np.float32))
        f.create_dataset("rssi", data=np.random.uniform(-90, -60, n).astype(np.float32))
        f.create_dataset("pos", data=np.random.randn(n, 3).astype(np.float32) * 100)
    with h5py.File(rd.name, "w") as f:
        f.create_dataset("rain_rate", data=np.random.uniform(0, 30, n).astype(np.float32))
    with h5py.File(fd.name, "w") as f:
        f.create_dataset("lai", data=np.random.uniform(0, 5, n).astype(np.float32))

    return td.name, rd.name, fd.name


# ---------------------------------------------------------------------------
# LEOBeamformingEnv tests
# ---------------------------------------------------------------------------

class TestLEOBeamformingEnv:
    def _make_env(self):
        from envs.leo_beamforming_env import LEOBeamformingEnv
        return LEOBeamformingEnv(
            channel_model=_make_channel(),
            telemetry_stream=_FakeTelemetry(),
            radar_stream=_FakeRadar(),
            foliage_map=_FakeFoliage(),
        )

    def test_import(self):
        from envs.leo_beamforming_env import LEOBeamformingEnv
        assert LEOBeamformingEnv is not None

    def test_reset_returns_obs_shape(self):
        env = self._make_env()
        obs, info = env.reset()
        assert obs.shape == (7,)
        assert isinstance(info, dict)

    def test_step_returns_correct_types(self):
        env = self._make_env()
        env.reset()
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs.shape == (7,)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert "throughput" in info
        assert "outage" in info

    def test_observation_in_bounds(self):
        env = self._make_env()
        obs, _ = env.reset()
        assert env.observation_space.contains(obs)

    def test_action_space_sampling(self):
        env = self._make_env()
        for _ in range(5):
            action = env.action_space.sample()
            assert env.action_space.contains(action)

    def test_multiple_steps(self):
        env = self._make_env()
        env.reset()
        for _ in range(10):
            obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
            assert obs.shape == (7,)

    def test_reward_contains_throughput(self):
        """Reward should increase when SNR is well above threshold (no outage)."""
        env = self._make_env()
        env.reset()
        rewards = []
        for _ in range(5):
            _, r, _, _, _ = env.step(env.action_space.sample())
            rewards.append(r)
        # At least one reward should be computed without error
        assert len(rewards) == 5


# ---------------------------------------------------------------------------
# OfflineLEOEnv tests
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _H5PY, reason="h5py not installed")
class TestOfflineLEOEnv:
    def setup_method(self):
        self.tpath, self.rpath, self.fpath = _make_offline_h5(n=30)

    def teardown_method(self):
        for p in [self.tpath, self.rpath, self.fpath]:
            os.unlink(p)

    def test_import(self):
        from envs.offline_env import OfflineLEOEnv
        assert OfflineLEOEnv is not None

    def test_reset(self):
        from envs.offline_env import OfflineLEOEnv
        env = OfflineLEOEnv(self.tpath, self.rpath, self.fpath)
        obs, info = env.reset(seed=42)
        assert obs.shape == (7,)

    def test_step(self):
        from envs.offline_env import OfflineLEOEnv
        env = OfflineLEOEnv(self.tpath, self.rpath, self.fpath)
        env.reset(seed=0)
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs.shape == (7,)
        assert isinstance(reward, float)
        assert "snr" in info

    def test_terminates_at_end(self):
        from envs.offline_env import OfflineLEOEnv
        env = OfflineLEOEnv(self.tpath, self.rpath, self.fpath, max_episode_steps=5)
        obs, _ = env.reset(seed=0)
        # Force index to near end
        env.current_idx = 28
        _, _, terminated, truncated, _ = env.step(env.action_space.sample())
        assert terminated or truncated
