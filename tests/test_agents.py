"""
Tests for BeamformingNetwork, DQNNetwork, DQNAgent, PPOAgent, and A3CWorker.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest
import torch


STATE_DIM = 7
ACTION_DIM = 4
N_ACTIONS = 10  # for DQN discrete


class TestBeamformingNetwork:
    def test_import(self):
        from agents.networks import BeamformingNetwork
        assert BeamformingNetwork is not None

    def test_forward_shape(self):
        from agents.networks import BeamformingNetwork
        net = BeamformingNetwork(STATE_DIM, ACTION_DIM)
        x = torch.randn(4, STATE_DIM)
        mean, value = net(x)
        assert mean.shape == (4, ACTION_DIM)
        assert value.shape == (4, 1)

    def test_get_action_stochastic(self):
        from agents.networks import BeamformingNetwork
        net = BeamformingNetwork(STATE_DIM, ACTION_DIM)
        state = torch.randn(1, STATE_DIM)
        action, log_prob, value = net.get_action(state, deterministic=False)
        assert action.shape == (1, ACTION_DIM)
        assert log_prob.shape == torch.Size([1])

    def test_get_action_deterministic(self):
        from agents.networks import BeamformingNetwork
        net = BeamformingNetwork(STATE_DIM, ACTION_DIM)
        state = torch.randn(1, STATE_DIM)
        a1, _, _ = net.get_action(state, deterministic=True)
        a2, _, _ = net.get_action(state, deterministic=True)
        assert torch.allclose(a1, a2)

    def test_log_std_parameter(self):
        from agents.networks import BeamformingNetwork
        net = BeamformingNetwork(STATE_DIM, ACTION_DIM)
        assert net.log_std.shape == (ACTION_DIM,)


class TestDQNNetwork:
    def test_import(self):
        from agents.networks import DQNNetwork
        assert DQNNetwork is not None

    def test_forward_shape(self):
        from agents.networks import DQNNetwork
        net = DQNNetwork(STATE_DIM, N_ACTIONS)
        x = torch.randn(4, STATE_DIM)
        q = net(x)
        assert q.shape == (4, N_ACTIONS)


class TestDQNAgent:
    def test_import(self):
        from agents.dqn_agent import DQNAgent
        assert DQNAgent is not None

    def test_select_action_range(self):
        from agents.dqn_agent import DQNAgent
        agent = DQNAgent(STATE_DIM, N_ACTIONS)
        state = np.random.randn(STATE_DIM).astype(np.float32)
        action = agent.select_action(state)
        assert 0 <= action < N_ACTIONS

    def test_deterministic_action_consistent(self):
        from agents.dqn_agent import DQNAgent
        agent = DQNAgent(STATE_DIM, N_ACTIONS)
        state = np.random.randn(STATE_DIM).astype(np.float32)
        a1 = agent.get_action(state, deterministic=True)
        a2 = agent.get_action(state, deterministic=True)
        assert a1 == a2

    def test_store_transition(self):
        from agents.dqn_agent import DQNAgent
        agent = DQNAgent(STATE_DIM, N_ACTIONS)
        s = np.zeros(STATE_DIM, dtype=np.float32)
        agent.store_transition(s, 0, 1.0, s, False)
        assert len(agent.replay_buffer) == 1

    def test_update_returns_none_when_buffer_small(self):
        from agents.dqn_agent import DQNAgent
        agent = DQNAgent(STATE_DIM, N_ACTIONS, batch_size=64)
        result = agent.update()
        assert result is None

    def test_update_returns_loss_when_buffer_full(self):
        from agents.dqn_agent import DQNAgent
        agent = DQNAgent(STATE_DIM, N_ACTIONS, batch_size=8)
        s = np.zeros(STATE_DIM, dtype=np.float32)
        for _ in range(20):
            agent.store_transition(s, 0, 1.0, s, False)
        loss = agent.update()
        assert isinstance(loss, float)
        assert loss >= 0.0

    def test_epsilon_decay(self):
        from agents.dqn_agent import DQNAgent
        agent = DQNAgent(STATE_DIM, N_ACTIONS, batch_size=4, epsilon_start=1.0, epsilon_decay=0.5)
        s = np.zeros(STATE_DIM, dtype=np.float32)
        for _ in range(10):
            agent.store_transition(s, 0, 1.0, s, False)
        agent.update()
        # Epsilon should have decayed at least once
        assert agent.epsilon < 1.0


class TestPPOAgent:
    def test_import(self):
        from agents.ppo_agent import PPOAgent
        assert PPOAgent is not None

    def test_get_action_shape(self):
        from agents.ppo_agent import PPOAgent
        agent = PPOAgent(STATE_DIM, ACTION_DIM)
        state = np.random.randn(STATE_DIM).astype(np.float32)
        action, log_prob = agent.get_action(state)
        assert action.shape == (ACTION_DIM,)
        assert isinstance(log_prob, float)

    def test_deterministic_action_consistent(self):
        from agents.ppo_agent import PPOAgent
        agent = PPOAgent(STATE_DIM, ACTION_DIM)
        state = np.random.randn(STATE_DIM).astype(np.float32)
        a1, _ = agent.get_action(state, deterministic=True)
        a2, _ = agent.get_action(state, deterministic=True)
        np.testing.assert_array_almost_equal(a1, a2)

    def test_compute_gae(self):
        from agents.ppo_agent import PPOAgent
        agent = PPOAgent(STATE_DIM, ACTION_DIM)
        rewards = [1.0, 1.0, 1.0, 1.0]
        values = [0.5, 0.5, 0.5, 0.5]
        dones = [False, False, False, True]
        returns, advantages = agent.compute_gae(rewards, values, dones)
        assert returns.shape == (4,)
        assert advantages.shape == (4,)

    def test_update_runs(self):
        from agents.ppo_agent import PPOAgent
        agent = PPOAgent(STATE_DIM, ACTION_DIM, epochs=2)
        T = 16
        states = torch.randn(T, STATE_DIM)
        actions = torch.randn(T, ACTION_DIM)
        old_log_probs = torch.randn(T)
        returns = torch.randn(T)
        advantages = torch.randn(T)
        loss = agent.update((states, actions, old_log_probs, returns, advantages))
        assert isinstance(loss, float)


class TestA3CWorker:
    def test_import(self):
        from agents.a3c_agent import A3CWorker, run_a3c
        assert A3CWorker is not None
        assert run_a3c is not None

    def test_worker_starts_and_stops(self):
        from agents.networks import BeamformingNetwork
        from agents.a3c_agent import A3CWorker

        # Minimal fake environment
        class _Env:
            observation_space = type("OS", (), {"shape": (STATE_DIM,)})()
            action_space = type("AS", (), {})()

            def reset(self, **kwargs):
                return np.zeros(STATE_DIM, dtype=np.float32), {}

            def step(self, action):
                obs = np.zeros(STATE_DIM, dtype=np.float32)
                return obs, 1.0, False, False, {}

        global_net = BeamformingNetwork(STATE_DIM, ACTION_DIM)
        global_net.share_memory()
        optimizer = torch.optim.Adam(global_net.parameters(), lr=1e-4)

        worker = A3CWorker(
            worker_id=0,
            global_net=global_net,
            optimizer=optimizer,
            env_factory=_Env,
            state_dim=STATE_DIM,
            action_dim=ACTION_DIM,
            max_steps=5,
        )
        worker.start()
        worker.join(timeout=10)
        # Worker should complete within timeout
        assert not worker.is_alive()
