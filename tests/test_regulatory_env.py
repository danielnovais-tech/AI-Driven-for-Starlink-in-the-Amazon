"""
Tests for RegulatoryEnv wrapper (Phase 5 – compliance).
"""

import sys
import os
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest
import gymnasium as gym


# ---------------------------------------------------------------------------
# Minimal fake continuous environment
# ---------------------------------------------------------------------------

class _FakeContinuousEnv(gym.Env):
    """Trivial continuous env: action = [phase, power, mcs, rb]."""

    observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
    action_space = gym.spaces.Box(
        low=np.array([-math.pi, 0.0, 0.0, 0.0]),
        high=np.array([math.pi, 1.0, 4.0, 100.0]),
        dtype=np.float32,
    )

    def reset(self, *, seed=None, options=None):
        return np.zeros(4, dtype=np.float32), {}

    def step(self, action):
        obs = np.zeros(4, dtype=np.float32)
        reward = 1.0
        return obs, reward, False, False, {"raw_action": action}


def _make_reg_env(**kwargs):
    from envs.regulatory_env import RegulatoryEnv
    return RegulatoryEnv(_FakeContinuousEnv(), **kwargs)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestRegulatoryEnv:
    def test_import(self):
        from envs.regulatory_env import RegulatoryEnv
        assert RegulatoryEnv is not None

    def test_exported_from_envs(self):
        from envs import RegulatoryEnv
        assert RegulatoryEnv is not None

    def test_reset_delegates_to_inner(self):
        env = _make_reg_env()
        obs, info = env.reset()
        assert obs.shape == (4,)

    def test_step_returns_compliance_info(self):
        env = _make_reg_env()
        env.reset()
        action = np.array([0.1, 0.5, 1.0, 50.0], dtype=np.float32)
        _, _, _, _, info = env.step(action)
        assert "compliance_violations" in info
        assert "total_compliance_violations" in info

    def test_valid_action_no_violation(self):
        env = _make_reg_env(max_eirp_dbw=55.0, min_elevation_deg=20.0)
        env.reset()
        # Small phase, low power → should be within constraints
        action = np.array([0.01, 0.1, 1.0, 10.0], dtype=np.float32)
        _, reward, _, _, info = env.step(action)
        assert info["compliance_violations"] == 0

    def test_excessive_power_triggers_violation(self):
        env = _make_reg_env(
            max_eirp_dbw=10.0,  # very low limit
            tx_gain_dbi=35.0,
            max_tx_power_dbw=20.0,
        )
        env.reset()
        # Full power action should violate the EIRP ceiling
        action = np.array([0.0, 1.0, 1.0, 10.0], dtype=np.float32)
        _, _, _, _, info = env.step(action)
        assert info["compliance_violations"] >= 1

    def test_excessive_phase_triggers_violation(self):
        env = _make_reg_env(min_elevation_deg=80.0)  # max steering ~10 deg
        env.reset()
        # Large phase steering should violate minimum elevation
        action = np.array([math.pi / 2.0 - 0.01, 0.5, 1.0, 10.0], dtype=np.float32)
        _, _, _, _, info = env.step(action)
        assert info["compliance_violations"] >= 1

    def test_compliance_penalty_reduces_reward(self):
        env_strict = _make_reg_env(
            max_eirp_dbw=0.0,  # impossible to satisfy → always violated
            compliance_penalty=10.0,
        )
        env_permissive = _make_reg_env(max_eirp_dbw=100.0, compliance_penalty=10.0)

        env_strict.reset()
        env_permissive.reset()

        action = np.array([0.0, 1.0, 1.0, 10.0], dtype=np.float32)
        _, r_strict, _, _, info_strict = env_strict.step(action)
        _, r_permissive, _, _, info_permissive = env_permissive.step(action)

        assert r_strict < r_permissive

    def test_compliance_summary_keys(self):
        env = _make_reg_env()
        summary = env.compliance_summary()
        for key in ["total_violations", "max_eirp_dbw", "min_elevation_deg",
                    "max_power_fraction", "max_steering_rad"]:
            assert key in summary

    def test_total_violations_accumulates(self):
        env = _make_reg_env(max_eirp_dbw=0.0)  # always violated
        env.reset()
        action = np.array([0.0, 1.0, 1.0, 10.0], dtype=np.float32)
        for _ in range(5):
            env.step(action)
        assert env.total_violations >= 5

    def test_discrete_env_passthrough(self):
        """RegulatoryEnv should be transparent for discrete action spaces."""
        from envs.regulatory_env import RegulatoryEnv

        class _DiscreteEnv(gym.Env):
            observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(2,))
            action_space = gym.spaces.Discrete(3)

            def reset(self, *, seed=None, options=None):
                return np.zeros(2, dtype=np.float32), {}

            def step(self, action):
                return np.zeros(2), 1.0, False, False, {}

        env = RegulatoryEnv(_DiscreteEnv())
        env.reset()
        _, reward, _, _, info = env.step(1)
        assert info["compliance_violations"] == 0
        assert reward == 1.0  # no penalty for discrete actions
