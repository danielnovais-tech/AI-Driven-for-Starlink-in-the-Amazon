"""
Tests for federated learning: SatelliteAgent and FederatedAggregator.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest


STATE_DIM = 7
ACTION_DIM = 4


def _make_agent(sat_id=0, buffer_size=200, batch_size=32):
    from agents.federated_learner import SatelliteAgent
    return SatelliteAgent(
        satellite_id=sat_id,
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        buffer_size=buffer_size,
        batch_size=batch_size,
        local_epochs=2,
        update_freq=1,
    )


def _fill_buffer(agent, n=100):
    rng = np.random.default_rng(42)
    for _ in range(n):
        s = rng.standard_normal(STATE_DIM).astype(np.float32)
        a = rng.standard_normal(ACTION_DIM).astype(np.float32)
        r = float(rng.uniform(-1.0, 5.0))
        s2 = rng.standard_normal(STATE_DIM).astype(np.float32)
        agent.store_transition(s, a, r, s2, False)


# ---------------------------------------------------------------------------
# SatelliteAgent
# ---------------------------------------------------------------------------

class TestSatelliteAgent:
    def test_import(self):
        from agents.federated_learner import SatelliteAgent
        assert SatelliteAgent is not None

    def test_exported_from_agents_package(self):
        from agents import SatelliteAgent
        assert SatelliteAgent is not None

    def test_has_satellite_id(self):
        agent = _make_agent(sat_id=7)
        assert agent.satellite_id == 7

    def test_get_action_returns_array(self):
        agent = _make_agent()
        obs = np.zeros(STATE_DIM, dtype=np.float32)
        result = agent.get_action(obs, deterministic=True)
        assert result is not None

    def test_get_weights_returns_dict(self):
        agent = _make_agent()
        weights = agent.get_weights()
        assert isinstance(weights, dict)
        assert len(weights) > 0

    def test_set_weights_updates_params(self):
        import torch
        agent1 = _make_agent(sat_id=0)
        agent2 = _make_agent(sat_id=1)
        # Modify agent2's weights manually
        with torch.no_grad():
            for p in agent2.net.parameters():
                p.fill_(0.0)
        # Copy agent1's weights to agent2
        agent2.set_weights(agent1.get_weights())
        # Now they should have the same weights
        w1 = agent1.get_weights()
        w2 = agent2.get_weights()
        for k in w1:
            assert torch.allclose(w1[k], w2[k])

    def test_store_transition_increments_counter(self):
        agent = _make_agent()
        s = np.zeros(STATE_DIM, dtype=np.float32)
        agent.store_transition(s, np.zeros(ACTION_DIM, dtype=np.float32), 1.0, s, False)
        assert agent.samples_since_last_sync == 1

    def test_reset_sync_counter(self):
        agent = _make_agent()
        _fill_buffer(agent, 10)
        agent.reset_sync_counter()
        assert agent.samples_since_last_sync == 0

    def test_local_train_returns_none_small_buffer(self):
        agent = _make_agent(batch_size=200)
        # Buffer too small
        _fill_buffer(agent, 5)
        result = agent.local_train()
        assert result is None

    def test_local_train_returns_loss_when_buffer_ready(self):
        agent = _make_agent(batch_size=32)
        _fill_buffer(agent, 64)
        result = agent.local_train()
        assert result is not None
        assert np.isfinite(result)


# ---------------------------------------------------------------------------
# FederatedAggregator
# ---------------------------------------------------------------------------

class TestFederatedAggregator:
    def test_import(self):
        from agents.federated_learner import FederatedAggregator
        assert FederatedAggregator is not None

    def test_exported_from_agents_package(self):
        from agents import FederatedAggregator
        assert FederatedAggregator is not None

    def test_no_agents_raises(self):
        from agents.federated_learner import FederatedAggregator
        with pytest.raises(ValueError):
            FederatedAggregator([])

    def test_aggregate_returns_dict(self):
        from agents.federated_learner import FederatedAggregator
        agents = [_make_agent(i) for i in range(3)]
        agg = FederatedAggregator(agents)
        weights = agg.aggregate()
        assert isinstance(weights, dict)

    def test_broadcast_before_aggregate_raises(self):
        from agents.federated_learner import FederatedAggregator
        agents = [_make_agent(0)]
        agg = FederatedAggregator(agents)
        with pytest.raises(RuntimeError):
            agg.broadcast()

    def test_round_counter_increments(self):
        from agents.federated_learner import FederatedAggregator
        agents = [_make_agent(i) for i in range(2)]
        agg = FederatedAggregator(agents)
        assert agg.round == 0
        agg.aggregate()
        assert agg.round == 1
        agg.aggregate()
        assert agg.round == 2

    def test_aggregate_produces_average_weights(self):
        """Two agents with identical weights should produce the same averaged weights."""
        import torch
        from agents.federated_learner import FederatedAggregator
        a1 = _make_agent(0)
        a2 = _make_agent(1)
        # Set both to the same weights as a1
        a2.set_weights(a1.get_weights())
        agg = FederatedAggregator([a1, a2])
        # Simulate equal sample counts
        a1.samples_since_last_sync = 50
        a2.samples_since_last_sync = 50
        global_weights = agg.aggregate()
        orig = a1.get_weights()
        for k in orig:
            assert torch.allclose(orig[k].float(), global_weights[k].float(), atol=1e-5)

    def test_broadcast_updates_all_agents(self):
        """After broadcast, all agents should have identical weights."""
        import torch
        from agents.federated_learner import FederatedAggregator
        agents = [_make_agent(i) for i in range(4)]
        agg = FederatedAggregator(agents)
        agg.aggregate()
        agg.broadcast()
        w0 = agents[0].get_weights()
        for agent in agents[1:]:
            wi = agent.get_weights()
            for k in w0:
                assert torch.allclose(w0[k].float(), wi[k].float(), atol=1e-5)

    def test_broadcast_resets_sample_counters(self):
        from agents.federated_learner import FederatedAggregator
        agents = [_make_agent(i) for i in range(3)]
        for i, a in enumerate(agents):
            a.samples_since_last_sync = (i + 1) * 10
        agg = FederatedAggregator(agents)
        agg.aggregate()
        agg.broadcast()
        for a in agents:
            assert a.samples_since_last_sync == 0

    def test_run_round_convenience(self):
        from agents.federated_learner import FederatedAggregator
        agents = [_make_agent(i) for i in range(2)]
        agg = FederatedAggregator(agents)
        weights = agg.run_round()
        assert isinstance(weights, dict)
        assert agg.round == 1

    def test_get_global_weights_none_before_aggregate(self):
        from agents.federated_learner import FederatedAggregator
        agg = FederatedAggregator([_make_agent(0)])
        assert agg.get_global_weights() is None

    def test_get_global_weights_returns_copy(self):
        """Modifying the returned dict should not affect the registry."""
        import torch
        from agents.federated_learner import FederatedAggregator
        agents = [_make_agent(0)]
        agg = FederatedAggregator(agents)
        agg.aggregate()
        w1 = agg.get_global_weights()
        # Record the sum of all entries before corruption
        total_before = sum(v.float().sum().item() for v in w1.values())
        # Corrupt the copy in-place with a very large value
        for k in w1:
            w1[k].fill_(1e9)
        # Original should be unchanged (deep copy)
        w2 = agg.get_global_weights()
        total_after = sum(v.float().sum().item() for v in w2.values())
        assert abs(total_before - total_after) < 1e-3  # copy was not mutated

    def test_weighted_averaging_sample_counts(self):
        """Agent with more samples should dominate the average."""
        import torch
        from agents.federated_learner import FederatedAggregator, SatelliteAgent

        a1 = _make_agent(0)
        a2 = _make_agent(1)
        # Set a1 to all-ones, a2 to all-zeros
        with torch.no_grad():
            for p in a1.net.parameters():
                p.fill_(1.0)
            for p in a2.net.parameters():
                p.fill_(0.0)

        a1.samples_since_last_sync = 90
        a2.samples_since_last_sync = 10

        agg = FederatedAggregator([a1, a2])
        global_weights = agg.aggregate()
        # Expected average ≈ 0.9 × 1 + 0.1 × 0 = 0.9
        first_key = list(global_weights.keys())[0]
        avg_val = global_weights[first_key].float().mean().item()
        assert abs(avg_val - 0.9) < 1e-5

    def test_min_participants_not_met_raises(self):
        from agents.federated_learner import FederatedAggregator
        agents = [_make_agent(0)]
        agg = FederatedAggregator(agents, min_participants=3)
        with pytest.raises(RuntimeError):
            agg.aggregate()
