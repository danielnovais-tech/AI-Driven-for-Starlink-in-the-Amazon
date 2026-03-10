"""
Tests for OnlinePPOAgent (continuous fine-tuning PPO).
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest


STATE_DIM = 7
ACTION_DIM = 4


class TestOnlinePPOAgent:
    def test_import(self):
        from agents.online_ppo import OnlinePPOAgent
        assert OnlinePPOAgent is not None

    def test_exported_from_package(self):
        from agents import OnlinePPOAgent
        assert OnlinePPOAgent is not None

    def test_is_subclass_of_ppo(self):
        from agents.online_ppo import OnlinePPOAgent
        from agents.ppo_agent import PPOAgent
        assert issubclass(OnlinePPOAgent, PPOAgent)

    def test_get_action_shape(self):
        from agents.online_ppo import OnlinePPOAgent
        agent = OnlinePPOAgent(STATE_DIM, ACTION_DIM)
        state = np.random.randn(STATE_DIM).astype(np.float32)
        action, log_prob = agent.get_action(state)
        assert action.shape == (ACTION_DIM,)
        assert isinstance(log_prob, float)

    def test_store_transition(self):
        from agents.online_ppo import OnlinePPOAgent
        agent = OnlinePPOAgent(STATE_DIM, ACTION_DIM)
        s = np.zeros(STATE_DIM, dtype=np.float32)
        a = np.zeros(ACTION_DIM, dtype=np.float32)
        agent.store_transition(s, a, 1.0, s, False)
        assert len(agent.buffer) == 1

    def test_update_online_none_when_buffer_small(self):
        from agents.online_ppo import OnlinePPOAgent
        agent = OnlinePPOAgent(STATE_DIM, ACTION_DIM, batch_size=64, update_freq=1)
        result = agent.update_online()
        assert result is None  # buffer too small

    def test_update_online_none_when_freq_not_met(self):
        from agents.online_ppo import OnlinePPOAgent
        agent = OnlinePPOAgent(STATE_DIM, ACTION_DIM, batch_size=4, update_freq=10)
        s = np.zeros(STATE_DIM, dtype=np.float32)
        a = np.zeros(ACTION_DIM, dtype=np.float32)
        for _ in range(20):
            agent.store_transition(s, a, 1.0, s, False)
        # _step_counter is now 20 but update_freq=10 → counter % 10 == 0 → should update
        # Let's check: store sets counter to 20, which % 10 == 0 → update runs
        # Actually we want to test the case where it doesn't run: use update_freq=7, counter=20 → 20%7=6
        agent2 = OnlinePPOAgent(STATE_DIM, ACTION_DIM, batch_size=4, update_freq=7)
        for _ in range(20):
            agent2.store_transition(s, a, 1.0, s, False)
        # 20 % 7 = 6 != 0 → should return None
        result = agent2.update_online()
        assert result is None

    def test_update_online_returns_loss(self):
        from agents.online_ppo import OnlinePPOAgent
        agent = OnlinePPOAgent(
            STATE_DIM, ACTION_DIM, batch_size=8, update_freq=10, epochs=2
        )
        s = np.zeros(STATE_DIM, dtype=np.float32)
        a = np.zeros(ACTION_DIM, dtype=np.float32)
        # Fill buffer and align step counter to a multiple of update_freq
        for _ in range(100):
            agent.store_transition(s, a, 1.0, s, False)
        # Make step counter a multiple of 10
        agent._step_counter = 100
        loss = agent.update_online()
        assert isinstance(loss, float)

    def test_buffer_eviction(self):
        from agents.online_ppo import OnlinePPOAgent
        agent = OnlinePPOAgent(STATE_DIM, ACTION_DIM, buffer_size=5)
        s = np.zeros(STATE_DIM, dtype=np.float32)
        a = np.zeros(ACTION_DIM, dtype=np.float32)
        for _ in range(10):
            agent.store_transition(s, a, 1.0, s, False)
        # Buffer should not exceed maxlen=5
        assert len(agent.buffer) == 5
