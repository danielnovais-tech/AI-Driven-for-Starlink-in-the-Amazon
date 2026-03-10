"""
Tests for OutageValidator and synthetic dataset generation.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest


class TestMakeSyntheticDataset:
    def test_import(self):
        from utils.outage_validator import make_synthetic_dataset
        assert make_synthetic_dataset is not None

    def test_returns_dict_with_required_keys(self):
        from utils.outage_validator import make_synthetic_dataset
        ds = make_synthetic_dataset(n_steps=100, seed=0)
        for key in ["snr", "rain_rate", "foliage", "outage", "throughput"]:
            assert key in ds

    def test_arrays_same_length(self):
        from utils.outage_validator import make_synthetic_dataset
        ds = make_synthetic_dataset(n_steps=200, seed=0)
        lengths = [len(v) for v in ds.values()]
        assert len(set(lengths)) == 1  # all equal
        assert lengths[0] == 200

    def test_outage_is_binary(self):
        from utils.outage_validator import make_synthetic_dataset
        ds = make_synthetic_dataset(n_steps=100, seed=0)
        assert set(np.unique(ds["outage"])).issubset({0.0, 1.0})

    def test_rain_peak_controls_outage_rate(self):
        """Higher rain peak → more outages expected."""
        from utils.outage_validator import make_synthetic_dataset
        ds_lo = make_synthetic_dataset(n_steps=1000, rain_peak_mmh=5.0, seed=0)
        ds_hi = make_synthetic_dataset(n_steps=1000, rain_peak_mmh=200.0, seed=0)
        assert ds_hi["outage"].mean() >= ds_lo["outage"].mean()


class TestOutageValidator:
    def _make_dataset(self, n=200, peak=60.0):
        from utils.outage_validator import make_synthetic_dataset
        return make_synthetic_dataset(n_steps=n, rain_peak_mmh=peak, seed=42)

    def test_import(self):
        from utils.outage_validator import OutageValidator
        assert OutageValidator is not None

    def test_exported_from_utils_package(self):
        from utils import OutageValidator
        assert OutageValidator is not None

    def test_evaluate_with_fixed_policy(self):
        """Evaluate a fixed zero-action policy."""
        from utils.outage_validator import OutageValidator
        ds = self._make_dataset()
        validator = OutageValidator(snr_threshold_db=5.0)

        def zero_policy(state):
            return np.zeros(4, dtype=np.float32)

        results = validator.evaluate_policy(zero_policy, ds)
        assert results.n_steps == 200

    def test_results_has_all_fields(self):
        from utils.outage_validator import OutageValidator
        ds = self._make_dataset()
        validator = OutageValidator()

        def policy(s):
            return np.zeros(4, dtype=np.float32)

        r = validator.evaluate_policy(policy, ds)
        assert 0.0 <= r.policy_outage_rate <= 1.0
        assert 0.0 <= r.baseline_outage_rate <= 1.0
        assert r.policy_mean_throughput >= 0.0
        assert r.n_steps == 200

    def test_summary_is_string(self):
        from utils.outage_validator import OutageValidator
        ds = self._make_dataset()
        validator = OutageValidator()
        r = validator.evaluate_policy(lambda s: np.zeros(4, dtype=np.float32), ds)
        summary = r.summary()
        assert isinstance(summary, str)
        assert "Outage" in summary

    def test_confusion_matrix_sums_to_n(self):
        from utils.outage_validator import OutageValidator
        ds = self._make_dataset(n=100)
        validator = OutageValidator()
        r = validator.evaluate_policy(lambda s: np.zeros(4, dtype=np.float32), ds)
        cm = r.confusion_matrix
        total = cm["TP"] + cm["FP"] + cm["TN"] + cm["FN"]
        assert total == 100

    def test_policy_accepting_tuple_return(self):
        """Policy returning (action, log_prob) tuple should work."""
        from utils.outage_validator import OutageValidator
        ds = self._make_dataset(n=50)
        validator = OutageValidator()

        def ppo_like_policy(s):
            return np.zeros(4, dtype=np.float32), -1.5  # tuple

        r = validator.evaluate_policy(ppo_like_policy, ds)
        assert r.n_steps == 50
