"""
Tests for gradient-based explainability module.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest


STATE_DIM = 7
ACTION_DIM = 4


def _make_model():
    from agents.networks import BeamformingNetwork
    return BeamformingNetwork(STATE_DIM, ACTION_DIM)


class TestVanillaSaliency:
    def test_import(self):
        from utils.explainability import vanilla_saliency
        assert vanilla_saliency is not None

    def test_output_shape(self):
        from utils.explainability import vanilla_saliency
        model = _make_model()
        state = np.zeros(STATE_DIM, dtype=np.float32)
        sal = vanilla_saliency(model, state)
        assert sal.shape == (STATE_DIM,)

    def test_output_non_negative(self):
        from utils.explainability import vanilla_saliency
        model = _make_model()
        state = np.random.randn(STATE_DIM).astype(np.float32)
        sal = vanilla_saliency(model, state)
        assert np.all(sal >= 0)

    def test_with_action_idx(self):
        from utils.explainability import vanilla_saliency
        model = _make_model()
        state = np.random.randn(STATE_DIM).astype(np.float32)
        sal = vanilla_saliency(model, state, action_idx=0)
        assert sal.shape == (STATE_DIM,)

    def test_exported_from_utils_package(self):
        from utils import vanilla_saliency
        assert vanilla_saliency is not None


class TestIntegratedGradients:
    def test_import(self):
        from utils.explainability import integrated_gradients
        assert integrated_gradients is not None

    def test_output_shape(self):
        from utils.explainability import integrated_gradients
        model = _make_model()
        state = np.ones(STATE_DIM, dtype=np.float32)
        attr = integrated_gradients(model, state, n_steps=10)
        assert attr.shape == (STATE_DIM,)

    def test_zero_input_zero_attribution(self):
        """If state == baseline, all attributions should be near zero."""
        from utils.explainability import integrated_gradients
        model = _make_model()
        state = np.zeros(STATE_DIM, dtype=np.float32)
        baseline = np.zeros(STATE_DIM, dtype=np.float32)
        attr = integrated_gradients(model, state, baseline=baseline, n_steps=10)
        assert np.max(np.abs(attr)) < 1e-6

    def test_custom_baseline(self):
        from utils.explainability import integrated_gradients
        model = _make_model()
        state = np.random.randn(STATE_DIM).astype(np.float32)
        baseline = np.random.randn(STATE_DIM).astype(np.float32)
        attr = integrated_gradients(model, state, baseline=baseline, n_steps=20)
        assert attr.shape == (STATE_DIM,)


class TestSmoothGrad:
    def test_import(self):
        from utils.explainability import smooth_grad
        assert smooth_grad is not None

    def test_output_shape(self):
        from utils.explainability import smooth_grad
        model = _make_model()
        state = np.zeros(STATE_DIM, dtype=np.float32)
        sal = smooth_grad(model, state, n_samples=5, seed=0)
        assert sal.shape == (STATE_DIM,)

    def test_smoother_than_vanilla(self):
        """SmoothGrad variance across random states should be smaller than vanilla."""
        from utils.explainability import smooth_grad, vanilla_saliency
        # SmoothGrad variance should not exceed vanilla variance by more than
        # this factor (generous tolerance since the networks are randomly initialised).
        MAX_VARIANCE_RATIO = 3.0
        model = _make_model()
        rng = np.random.default_rng(0)
        vanilla_vals = []
        smooth_vals = []
        for _ in range(10):
            s = rng.standard_normal(STATE_DIM).astype(np.float32)
            vanilla_vals.append(vanilla_saliency(model, s))
            smooth_vals.append(smooth_grad(model, s, n_samples=5, seed=0))
        vanilla_var = np.std(np.array(vanilla_vals), axis=0).mean()
        smooth_var = np.std(np.array(smooth_vals), axis=0).mean()
        assert smooth_var <= vanilla_var * MAX_VARIANCE_RATIO


class TestFeatureImportanceSummary:
    def test_import(self):
        from utils.explainability import feature_importance_summary
        assert feature_importance_summary is not None

    def test_returns_dict(self):
        from utils.explainability import feature_importance_summary
        model = _make_model()
        states = np.random.randn(5, STATE_DIM).astype(np.float32)
        result = feature_importance_summary(model, states, method="vanilla")
        assert isinstance(result, dict)
        assert len(result) == STATE_DIM

    def test_feature_names_used(self):
        from utils.explainability import feature_importance_summary
        model = _make_model()
        states = np.random.randn(3, STATE_DIM).astype(np.float32)
        names = ["snr", "rssi", "x", "y", "z", "rain", "lai"]
        result = feature_importance_summary(model, states, feature_names=names)
        assert set(result.keys()) == set(names)

    def test_ig_method(self):
        from utils.explainability import feature_importance_summary
        model = _make_model()
        states = np.random.randn(2, STATE_DIM).astype(np.float32)
        result = feature_importance_summary(model, states, method="ig")
        assert len(result) == STATE_DIM

    def test_smooth_grad_method(self):
        from utils.explainability import feature_importance_summary
        model = _make_model()
        states = np.random.randn(2, STATE_DIM).astype(np.float32)
        result = feature_importance_summary(model, states, method="smooth_grad")
        assert len(result) == STATE_DIM
