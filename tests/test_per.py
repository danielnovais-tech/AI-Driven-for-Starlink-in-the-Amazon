"""
Tests for PrioritizedReplayBuffer (PER) and PEROnlinePPOAgent.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest


STATE_DIM = 7
ACTION_DIM = 4


# ---------------------------------------------------------------------------
# PrioritizedReplayBuffer
# ---------------------------------------------------------------------------

class TestPrioritizedReplayBuffer:
    def _make(self, capacity=100, **kwargs):
        from agents.per_buffer import PrioritizedReplayBuffer
        return PrioritizedReplayBuffer(capacity=capacity, **kwargs)

    def test_import(self):
        from agents.per_buffer import PrioritizedReplayBuffer
        assert PrioritizedReplayBuffer is not None

    def test_exported_from_agents_package(self):
        from agents import PrioritizedReplayBuffer
        assert PrioritizedReplayBuffer is not None

    def test_initial_size_is_zero(self):
        buf = self._make()
        assert buf.size == 0

    def test_add_increments_size(self):
        buf = self._make()
        s = np.zeros(STATE_DIM, dtype=np.float32)
        a = np.zeros(ACTION_DIM, dtype=np.float32)
        buf.add(s, a, 1.0, s, False)
        assert buf.size == 1

    def test_capacity_respected(self):
        buf = self._make(capacity=5)
        s = np.zeros(STATE_DIM, dtype=np.float32)
        a = np.zeros(ACTION_DIM, dtype=np.float32)
        for _ in range(10):
            buf.add(s, a, 1.0, s, False)
        assert buf.size == 5

    def test_sample_returns_correct_shapes(self):
        buf = self._make(capacity=50)
        s = np.zeros(STATE_DIM, dtype=np.float32)
        a = np.zeros(ACTION_DIM, dtype=np.float32)
        for _ in range(20):
            buf.add(s, a, 1.0, s, False)
        batch, leaf_idxs, is_weights = buf.sample(8)
        assert len(batch) == 8
        assert len(leaf_idxs) == 8
        assert is_weights.shape == (8,)

    def test_is_weights_in_range(self):
        buf = self._make(capacity=50)
        s = np.zeros(STATE_DIM, dtype=np.float32)
        a = np.zeros(ACTION_DIM, dtype=np.float32)
        for _ in range(20):
            buf.add(s, a, 1.0, s, False)
        _, _, is_weights = buf.sample(8)
        assert np.all(is_weights > 0)
        assert np.all(is_weights <= 1.0 + 1e-6)

    def test_update_priorities(self):
        buf = self._make(capacity=50)
        s = np.zeros(STATE_DIM, dtype=np.float32)
        a = np.zeros(ACTION_DIM, dtype=np.float32)
        for _ in range(20):
            buf.add(s, a, 1.0, s, False)
        _, leaf_idxs, _ = buf.sample(8)
        td_errors = np.abs(np.random.randn(8)).astype(np.float32)
        buf.update_priorities(leaf_idxs, td_errors)  # should not raise

    def test_sample_raises_when_too_few(self):
        buf = self._make(capacity=50)
        s = np.zeros(STATE_DIM, dtype=np.float32)
        a = np.zeros(ACTION_DIM, dtype=np.float32)
        buf.add(s, a, 1.0, s, False)
        with pytest.raises(RuntimeError):
            buf.sample(32)

    def test_zero_capacity_raises(self):
        from agents.per_buffer import PrioritizedReplayBuffer
        with pytest.raises(ValueError):
            PrioritizedReplayBuffer(capacity=0)

    def test_high_priority_transition_sampled_more(self):
        """Transitions with higher priority should be sampled more often."""
        from agents.per_buffer import PrioritizedReplayBuffer
        buf = PrioritizedReplayBuffer(capacity=100, alpha=1.0)
        s = np.zeros(STATE_DIM, dtype=np.float32)
        a = np.zeros(ACTION_DIM, dtype=np.float32)
        # Add 90 low-priority transitions
        for _ in range(90):
            buf.add(s, a, 1.0, s, False, td_error=0.01)
        # Add 10 high-priority transitions
        for _ in range(10):
            buf.add(s, a, 1.0, s, False, td_error=100.0)

        # Sample many times and count how often high-priority leaves appear
        HIGH_PRIORITY_THRESHOLD = 1.0
        high_count = 0
        total_samples = 500
        for _ in range(total_samples // 10):
            _, leaf_idxs, _ = buf.sample(10)
            for idx in leaf_idxs:
                if buf.get_leaf_priority(idx) > HIGH_PRIORITY_THRESHOLD:
                    high_count += 1
        # High-priority transitions (10% of buffer) should appear in far more
        # than 10% of samples when alpha=1.0
        assert high_count > total_samples * 0.1


# ---------------------------------------------------------------------------
# PEROnlinePPOAgent
# ---------------------------------------------------------------------------

class TestPEROnlinePPOAgent:
    def test_import(self):
        from agents.online_ppo import PEROnlinePPOAgent
        assert PEROnlinePPOAgent is not None

    def test_exported_from_agents_package(self):
        from agents import PEROnlinePPOAgent
        assert PEROnlinePPOAgent is not None

    def test_store_transition(self):
        from agents.online_ppo import PEROnlinePPOAgent
        agent = PEROnlinePPOAgent(STATE_DIM, ACTION_DIM)
        s = np.zeros(STATE_DIM, dtype=np.float32)
        a = np.zeros(ACTION_DIM, dtype=np.float32)
        agent.store_transition(s, a, 1.0, s, False)
        assert agent.per_buffer.size == 1

    def test_update_online_none_when_buffer_small(self):
        from agents.online_ppo import PEROnlinePPOAgent
        agent = PEROnlinePPOAgent(STATE_DIM, ACTION_DIM, batch_size=64)
        assert agent.update_online() is None

    def test_update_online_returns_loss(self):
        from agents.online_ppo import PEROnlinePPOAgent
        agent = PEROnlinePPOAgent(
            STATE_DIM, ACTION_DIM, batch_size=8, update_freq=10, epochs=2
        )
        s = np.zeros(STATE_DIM, dtype=np.float32)
        a = np.zeros(ACTION_DIM, dtype=np.float32)
        for _ in range(100):
            agent.store_transition(s, a, 1.0, s, False)
        agent._step_counter = 100  # trigger update
        loss = agent.update_online()
        assert isinstance(loss, float)

    def test_get_action_still_works(self):
        from agents.online_ppo import PEROnlinePPOAgent
        agent = PEROnlinePPOAgent(STATE_DIM, ACTION_DIM)
        state = np.zeros(STATE_DIM, dtype=np.float32)
        action, log_prob = agent.get_action(state)
        assert action.shape == (ACTION_DIM,)
