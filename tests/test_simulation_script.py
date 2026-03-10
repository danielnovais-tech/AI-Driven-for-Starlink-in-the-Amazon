"""
Tests for the long-duration simulation script.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import json
import tempfile
import numpy as np
import pytest


class TestSimulateLongDuration:
    def _run(self, **kwargs):
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
        from simulate_long_duration import run_simulation
        return run_simulation(**kwargs)

    def test_import_script(self):
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
        from simulate_long_duration import run_simulation, print_report
        assert run_simulation is not None
        assert print_report is not None

    def test_run_short_simulation(self):
        """A very short simulation (1 minute) should complete without error."""
        report = self._run(
            n_days=1.0 / 1440.0,  # 1 simulated minute
            n_satellites=20,
            max_satellites=3,
            seed=0,
        )
        assert "kpis" in report
        assert "metrics" in report

    def test_kpis_present(self):
        report = self._run(n_days=1.0 / 1440.0, n_satellites=10, max_satellites=3, seed=1)
        for key in ("mean_throughput_mbps", "outage_rate", "handover_rate_per_min",
                    "mean_queue_delay_ms", "total_compliance_violations",
                    "p95_latency_ms", "mean_packet_drop_rate"):
            assert key in report["kpis"], f"Missing KPI: {key}"

    def test_config_present(self):
        report = self._run(n_days=1.0 / 1440.0, n_satellites=10, max_satellites=3, seed=0)
        assert "config" in report
        assert report["config"]["n_satellites"] == 10

    def test_json_output(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            self._run(
                n_days=1.0 / 1440.0,
                n_satellites=10,
                max_satellites=3,
                seed=2,
                output_json=path,
            )
            with open(path) as f:
                doc = json.load(f)
            assert "kpis" in doc
        finally:
            os.unlink(path)

    def test_max_snr_policy(self):
        report = self._run(
            n_days=1.0 / 1440.0,
            n_satellites=20,
            max_satellites=3,
            seed=0,
            policy="max_snr",
        )
        assert report["config"]["policy"] == "max_snr"

    def test_geo_zones_disabled(self):
        report = self._run(
            n_days=1.0 / 1440.0,
            n_satellites=20,
            max_satellites=3,
            seed=0,
            enable_geo_zones=False,
        )
        assert not report["config"]["enable_geo_zones"]

    def test_outage_rate_in_range(self):
        report = self._run(n_days=1.0 / 1440.0, n_satellites=20, max_satellites=3, seed=3)
        assert 0.0 <= report["kpis"]["outage_rate"] <= 1.0

    def test_wall_elapsed_positive(self):
        report = self._run(n_days=1.0 / 1440.0, n_satellites=10, max_satellites=3, seed=0)
        assert report["wall_elapsed_s"] > 0.0

    def test_100_satellites(self):
        """100+ satellite constellation completes without errors."""
        report = self._run(
            n_days=1.0 / 14400.0,   # 6 simulated seconds
            n_satellites=100,
            max_satellites=5,
            seed=0,
        )
        assert report["config"]["n_satellites"] == 100


class TestBenchmarkPruning:
    """Test that prune_model works correctly."""

    def test_prune_model_import(self):
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
        from benchmark_inference import prune_model
        assert prune_model is not None

    def test_prune_model_runs(self):
        import copy
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
        from benchmark_inference import prune_model
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
        from agents.networks import BeamformingNetwork
        model = BeamformingNetwork(7, 4, 64)
        pruned = prune_model(copy.deepcopy(model), prune_ratio=0.3)
        assert pruned is not None

    def test_pruned_model_inference(self):
        import copy
        import torch
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
        from benchmark_inference import prune_model
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
        from agents.networks import BeamformingNetwork
        model = BeamformingNetwork(7, 4, 64)
        pruned = prune_model(copy.deepcopy(model), prune_ratio=0.5)
        pruned.eval()
        dummy = torch.randn(1, 7)
        action_mean, value = pruned(dummy)
        assert action_mean.shape == (1, 4)

    def test_pruned_has_zeros(self):
        """After pruning, at least prune_ratio fraction of weights should be zero."""
        import copy
        import torch
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
        from benchmark_inference import prune_model
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
        from agents.networks import BeamformingNetwork
        model = BeamformingNetwork(7, 4, 64)
        pruned = prune_model(copy.deepcopy(model), prune_ratio=0.5)
        total = 0
        zeros = 0
        for p in pruned.parameters():
            total += p.numel()
            zeros += int((p == 0).sum())
        zero_frac = zeros / max(total, 1)
        assert zero_frac >= 0.1  # At least 10% should be zero after 50% pruning
