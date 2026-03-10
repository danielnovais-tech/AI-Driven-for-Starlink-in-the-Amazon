"""
Tests for the evaluation utility.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Minimal stub environment and agent
# ---------------------------------------------------------------------------

class _DummyEnv:
    """A trivial environment that runs for exactly ``steps`` steps."""

    def __init__(self, steps=5, snr=15.0):
        self.steps = steps
        self.snr = snr
        self._step_count = 0

    @property
    def action_space(self):
        class AS:
            def sample(self_):
                return np.zeros(4, dtype=np.float32)
        return AS()

    def reset(self, **kwargs):
        self._step_count = 0
        return np.zeros(7, dtype=np.float32), {}

    def step(self, action):
        self._step_count += 1
        obs = np.zeros(7, dtype=np.float32)
        reward = 1.0
        terminated = self._step_count >= self.steps
        truncated = False
        info = {
            "throughput": 50.0,
            "latency": 600.0,
            "outage": 0.0 if self.snr >= 5.0 else 1.0,
            "snr": self.snr,
        }
        return obs, reward, terminated, truncated, info


class _DummyAgent:
    """Agent that always takes a zero action."""

    def get_action(self, state, deterministic=True):
        return np.zeros(4, dtype=np.float32)


class _DummyDQNAgent:
    """DQN-style agent returning a tuple of (action, log_prob)."""

    def get_action(self, state, deterministic=True):
        return (np.zeros(4, dtype=np.float32), 0.0)


class TestEvaluate:
    def test_import(self):
        from utils.evaluation import evaluate
        assert evaluate is not None

    def test_basic_run(self):
        from utils.evaluation import evaluate
        env = _DummyEnv(steps=5)
        agent = _DummyAgent()
        results = evaluate(env, agent, episodes=3, max_steps_per_episode=5)
        assert "mean_throughput" in results
        assert "mean_latency" in results
        assert "outage_prob" in results
        assert "reliability" in results
        assert "total_reward" in results
        assert results["n_episodes"] == 3

    def test_reliability_plus_outage_equals_one(self):
        from utils.evaluation import evaluate
        env = _DummyEnv(steps=5, snr=15.0)
        agent = _DummyAgent()
        results = evaluate(env, agent, episodes=5, max_steps_per_episode=5)
        assert abs(results["reliability"] + results["outage_prob"] - 1.0) < 1e-6

    def test_zero_outage_on_high_snr(self):
        from utils.evaluation import evaluate
        env = _DummyEnv(steps=5, snr=25.0)  # well above threshold
        agent = _DummyAgent()
        results = evaluate(env, agent, episodes=5, max_steps_per_episode=5)
        assert results["outage_prob"] == 0.0
        assert results["reliability"] == 1.0

    def test_throughput_value(self):
        from utils.evaluation import evaluate
        env = _DummyEnv(steps=5, snr=25.0)
        agent = _DummyAgent()
        results = evaluate(env, agent, episodes=2, max_steps_per_episode=5)
        assert abs(results["mean_throughput"] - 50.0) < 1e-6

    def test_works_with_dqn_agent_tuple_return(self):
        from utils.evaluation import evaluate
        env = _DummyEnv(steps=3)
        agent = _DummyDQNAgent()
        results = evaluate(env, agent, episodes=2, max_steps_per_episode=3)
        assert isinstance(results["mean_throughput"], float)
