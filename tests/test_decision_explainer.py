"""
Tests for GNN node importance and DecisionExplainer.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest

# Earth's mean radius used as a reference distance in ECEF coordinates (km)
_EARTH_RADIUS_KM = 6371.0
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Stub actor-critic network for flat-state agents
# ---------------------------------------------------------------------------

class _SimplePolicyNet(nn.Module):
    def __init__(self, state_dim: int = 7, action_dim: int = 4) -> None:
        super().__init__()
        self.fc = nn.Linear(state_dim, 32)
        self.actor = nn.Linear(32, action_dim)
        self.critic = nn.Linear(32, 1)

    def forward(self, x):
        h = torch.relu(self.fc(x))
        return self.actor(h), self.critic(h)


class _FlatAgent:
    def __init__(self, state_dim=7):
        self.net = _SimplePolicyNet(state_dim)

    def get_action(self, state, deterministic=True):
        return np.zeros(4, dtype=np.float32), 0.0


# ---------------------------------------------------------------------------
# DecisionExplainer – flat-state
# ---------------------------------------------------------------------------

class TestDecisionExplainerFlat:
    def test_import(self):
        from utils.explainability import DecisionExplainer
        assert DecisionExplainer is not None

    def test_exported_from_utils(self):
        from utils import DecisionExplainer
        assert DecisionExplainer is not None

    def test_explain_returns_dict(self):
        from utils.explainability import DecisionExplainer
        agent = _FlatAgent()
        explainer = DecisionExplainer(agent, method="vanilla")
        state = np.random.randn(7).astype(np.float32)
        result = explainer.explain(state, action=np.array([0.1, 0.5, 1.0, 20.0]))
        assert isinstance(result, dict)

    def test_explain_has_feature_scores(self):
        from utils.explainability import DecisionExplainer
        agent = _FlatAgent()
        explainer = DecisionExplainer(
            agent, feature_names=["snr", "rssi", "x", "y", "z", "rain", "lai"],
            method="vanilla"
        )
        state = np.random.randn(7).astype(np.float32)
        result = explainer.explain(state, action=0)
        assert "feature_scores" in result
        assert "snr" in result["feature_scores"]

    def test_explain_top_features_sorted(self):
        from utils.explainability import DecisionExplainer
        agent = _FlatAgent()
        explainer = DecisionExplainer(agent, method="vanilla")
        state = np.random.randn(7).astype(np.float32)
        result = explainer.explain(state, action=0)
        if "top_features" in result:
            scores = [v for _, v in result["top_features"]]
            assert scores == sorted(scores, reverse=True)

    def test_explain_disabled_returns_empty(self):
        from utils.explainability import DecisionExplainer
        agent = _FlatAgent()
        explainer = DecisionExplainer(agent, enabled=False)
        result = explainer.explain(np.zeros(7), action=0)
        assert result == {}

    def test_explain_smooth_grad(self):
        from utils.explainability import DecisionExplainer
        agent = _FlatAgent()
        explainer = DecisionExplainer(agent, method="smooth_grad")
        state = np.random.randn(7).astype(np.float32)
        result = explainer.explain(state, action=0)
        assert "feature_scores" in result

    def test_explain_ig(self):
        from utils.explainability import DecisionExplainer
        agent = _FlatAgent()
        explainer = DecisionExplainer(agent, method="ig")
        state = np.random.randn(7).astype(np.float32)
        result = explainer.explain(state, action=0)
        assert "feature_scores" in result


# ---------------------------------------------------------------------------
# gnn_node_importance (requires torch_geometric)
# ---------------------------------------------------------------------------

try:
    import torch_geometric  # noqa: F401
    _HAS_TG = True
except ImportError:
    _HAS_TG = False


class TestGNNNodeImportance:
    def test_import(self):
        from utils.explainability import gnn_node_importance
        assert gnn_node_importance is not None

    @pytest.mark.skipif(not _HAS_TG, reason="torch_geometric not installed")
    def test_returns_expected_keys(self):
        from utils.explainability import gnn_node_importance
        from agents.gnn_ppo_agent import GNNPPOAgent
        from torch_geometric.data import HeteroData
        agent = GNNPPOAgent(node_features=4)
        data = HeteroData()
        data["sat"].x = torch.randn(3, 4)
        data["sat"].num_nodes = 3
        data["ground_station"].x = torch.zeros(1, 4)
        data["ground_station"].num_nodes = 1
        src = torch.arange(3, dtype=torch.long)
        dst = torch.zeros(3, dtype=torch.long)
        data["sat", "to", "ground_station"].edge_index = torch.stack([src, dst], dim=0)
        result = gnn_node_importance(agent, data)
        for key in ("node_scores", "feature_scores", "top_node", "top_features"):
            assert key in result

    @pytest.mark.skipif(not _HAS_TG, reason="torch_geometric not installed")
    def test_node_scores_length_matches_nodes(self):
        from utils.explainability import gnn_node_importance
        from agents.gnn_ppo_agent import GNNPPOAgent
        from torch_geometric.data import HeteroData
        n_sats = 5
        agent = GNNPPOAgent(node_features=4)
        data = HeteroData()
        data["sat"].x = torch.randn(n_sats, 4)
        data["sat"].num_nodes = n_sats
        data["ground_station"].x = torch.zeros(1, 4)
        data["ground_station"].num_nodes = 1
        src = torch.arange(n_sats, dtype=torch.long)
        dst = torch.zeros(n_sats, dtype=torch.long)
        data["sat", "to", "ground_station"].edge_index = torch.stack([src, dst], dim=0)
        result = gnn_node_importance(agent, data)
        assert len(result["node_scores"]) == n_sats

    @pytest.mark.skipif(not _HAS_TG, reason="torch_geometric not installed")
    def test_top_node_is_valid_index(self):
        from utils.explainability import gnn_node_importance
        from agents.gnn_ppo_agent import GNNPPOAgent
        from torch_geometric.data import HeteroData
        n_sats = 3
        agent = GNNPPOAgent(node_features=4)
        data = HeteroData()
        data["sat"].x = torch.randn(n_sats, 4)
        data["sat"].num_nodes = n_sats
        data["ground_station"].x = torch.zeros(1, 4)
        data["ground_station"].num_nodes = 1
        src = torch.arange(n_sats, dtype=torch.long)
        dst = torch.zeros(n_sats, dtype=torch.long)
        data["sat", "to", "ground_station"].edge_index = torch.stack([src, dst], dim=0)
        result = gnn_node_importance(agent, data)
        assert 0 <= result["top_node"] < n_sats

    @pytest.mark.skipif(not _HAS_TG, reason="torch_geometric not installed")
    def test_custom_feature_names(self):
        from utils.explainability import gnn_node_importance
        from agents.gnn_ppo_agent import GNNPPOAgent
        from torch_geometric.data import HeteroData
        n_sats = 2
        feat_names = ["snr", "distance", "elevation", "rain_rate"]
        agent = GNNPPOAgent(node_features=4)
        data = HeteroData()
        data["sat"].x = torch.randn(n_sats, 4)
        data["sat"].num_nodes = n_sats
        data["ground_station"].x = torch.zeros(1, 4)
        data["ground_station"].num_nodes = 1
        src = torch.arange(n_sats, dtype=torch.long)
        dst = torch.zeros(n_sats, dtype=torch.long)
        data["sat", "to", "ground_station"].edge_index = torch.stack([src, dst], dim=0)
        result = gnn_node_importance(agent, data, feature_names=feat_names)
        assert "snr" in result["feature_scores"]


# ---------------------------------------------------------------------------
# DecisionExplainer – wired into OnlineBeamController
# ---------------------------------------------------------------------------

class TestControllerExplanationIntegration:
    def _make_controller(self, explainer=None):
        from inference.online_controller import OnlineBeamController

        class _Tel:
            ground_station_pos = np.array([0.0, 0.0, _EARTH_RADIUS_KM])
            def get_current_position(self): return np.array([_EARTH_RADIUS_KM + 550.0, 0.0, 0.0])
            def get_current_snr(self): return 15.0
            def get_current_rssi(self): return -80.0

        class _Radar:
            def get_at_location(self, pos): return 3.0

        class _Foliage:
            def get_at_location(self, pos): return 1.5

        return OnlineBeamController(
            agent=_FlatAgent(),
            telemetry_stream=_Tel(),
            radar_stream=_Radar(),
            foliage_map=_Foliage(),
            explainer=explainer,
        )

    def test_step_has_explanation_key(self):
        ctrl = self._make_controller()
        result = ctrl.step()
        assert "explanation" in result

    def test_step_explanation_empty_by_default(self):
        """Without an explainer, the explanation dict is empty."""
        ctrl = self._make_controller(explainer=None)
        result = ctrl.step()
        assert result["explanation"] == {}

    def test_step_with_explainer_returns_content(self):
        from utils.explainability import DecisionExplainer
        agent = _FlatAgent()
        explainer = DecisionExplainer(agent, method="vanilla", enabled=True)
        ctrl = self._make_controller(explainer=explainer)
        result = ctrl.step()
        # Explainer should populate feature_scores
        assert isinstance(result["explanation"], dict)
        # Should not be empty when the net is available
        assert len(result["explanation"]) > 0

    def test_explainer_error_does_not_crash_step(self):
        """An explainer that always raises must not crash the controller."""
        class _BrokenExplainer:
            def explain(self, state, action):
                raise RuntimeError("boom")

        ctrl = self._make_controller(explainer=_BrokenExplainer())
        result = ctrl.step()  # must not raise
        assert "explanation" in result
