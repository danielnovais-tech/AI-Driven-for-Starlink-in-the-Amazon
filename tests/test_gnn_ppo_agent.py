"""
Tests for GNNPPOAgent (requires torch_geometric).
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

# Skip entire module if torch_geometric is not installed
try:
    import torch_geometric  # noqa: F401
    _HAS_TORCH_GEO = True
except ImportError:
    _HAS_TORCH_GEO = False

pytestmark = pytest.mark.skipif(
    not _HAS_TORCH_GEO, reason="torch_geometric not installed"
)


def _make_fake_graph(n_sats: int = 3):
    """Build a minimal HeteroData graph for testing."""
    import torch
    from torch_geometric.data import HeteroData

    data = HeteroData()
    data["sat"].x = torch.randn(n_sats, 4)
    data["sat"].num_nodes = n_sats
    data["ground_station"].x = torch.zeros(1, 4)
    data["ground_station"].num_nodes = 1
    src = torch.arange(n_sats, dtype=torch.long)
    dst = torch.zeros(n_sats, dtype=torch.long)
    data["sat", "to", "ground_station"].edge_index = torch.stack([src, dst], dim=0)
    return data


class TestGNNPPOAgentBasic:
    def test_import(self):
        from agents.gnn_ppo_agent import GNNPPOAgent
        assert GNNPPOAgent is not None

    def test_get_action_deterministic(self):
        from agents.gnn_ppo_agent import GNNPPOAgent
        agent = GNNPPOAgent(node_features=4)
        graph = _make_fake_graph(n_sats=3)
        action, log_prob = agent.get_action(graph, deterministic=True)
        assert 0 <= action < 3
        assert isinstance(log_prob, float)

    def test_get_action_stochastic(self):
        from agents.gnn_ppo_agent import GNNPPOAgent
        agent = GNNPPOAgent(node_features=4)
        graph = _make_fake_graph(n_sats=3)
        action, log_prob = agent.get_action(graph, deterministic=False)
        assert 0 <= action < 3

    def test_store_and_update(self):
        from agents.gnn_ppo_agent import GNNPPOAgent
        agent = GNNPPOAgent(node_features=4, batch_size=4, update_freq=4)
        graphs = [_make_fake_graph(3) for _ in range(8)]
        for i, g in enumerate(graphs):
            a, _ = agent.get_action(g)
            agent.store_transition(g, a, float(i * 0.1), graphs[(i + 1) % 8], i == 7)
        # Should now be able to update
        loss = agent.update_online()
        assert loss is None or isinstance(loss, float)

    def test_get_set_weights(self):
        import torch
        from agents.gnn_ppo_agent import GNNPPOAgent
        a1 = GNNPPOAgent(node_features=4)
        a2 = GNNPPOAgent(node_features=4)
        # Set a2 to a1's weights
        a2.set_weights(a1.get_weights())
        for k in a1.get_weights():
            assert torch.allclose(
                a1.get_weights()[k].float(),
                a2.get_weights()[k].float(),
                atol=1e-5,
            )

    def test_update_requires_enough_transitions(self):
        from agents.gnn_ppo_agent import GNNPPOAgent
        agent = GNNPPOAgent(node_features=4, batch_size=32)
        graph = _make_fake_graph(3)
        agent.store_transition(graph, 0, 1.0, graph, False)
        # Buffer too small → None
        agent._step_counter = 1
        result = agent.update_online()
        assert result is None


class TestGNNPPOAgentMissingDep:
    """Tests the ImportError raised when torch_geometric is absent."""

    def test_error_message(self):
        """Verify helpful ImportError message (mocked scenario)."""
        import agents.gnn_ppo_agent as mod
        original = mod._TORCH_GEO_AVAILABLE
        mod._TORCH_GEO_AVAILABLE = False
        try:
            with pytest.raises(ImportError, match="torch_geometric"):
                mod._require_torch_geo()
        finally:
            mod._TORCH_GEO_AVAILABLE = original
