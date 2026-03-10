"""
Tests for the federated export-to-registry pipeline and retrain_job script.
"""

import sys
import os
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import pytest
import numpy as np


# ---------------------------------------------------------------------------
# FederatedAggregator.export_to_registry
# ---------------------------------------------------------------------------

class TestFederatedAggregatorExport:
    def _make_agents(self, n=2, state_dim=7, action_dim=4, batch_size=32):
        from agents.federated_learner import SatelliteAgent
        return [
            SatelliteAgent(
                satellite_id=i,
                state_dim=state_dim,
                action_dim=action_dim,
                buffer_size=200,
                batch_size=batch_size,
                local_epochs=1,
                update_freq=1,
            )
            for i in range(n)
        ]

    def test_export_after_aggregate(self):
        from agents.federated_learner import FederatedAggregator
        from utils.model_registry import ModelRegistry

        agents = self._make_agents(2)
        agg = FederatedAggregator(agents)
        agg.aggregate()

        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ModelRegistry(tmpdir)
            path = agg.export_to_registry(registry, model_name="test_model")
            assert os.path.isdir(path)
            versions = registry.list_versions("test_model")
            assert len(versions) == 1
            assert versions[0] == "v1"

    def test_export_includes_round_metadata(self):
        from agents.federated_learner import FederatedAggregator
        from utils.model_registry import ModelRegistry

        agents = self._make_agents(2)
        agg = FederatedAggregator(agents)
        agg.aggregate()

        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ModelRegistry(tmpdir)
            agg.export_to_registry(registry, model_name="test_model")
            meta = registry.get_metadata("test_model")
            assert meta.get("federated_round") == 1
            assert meta.get("n_participants") == 2

    def test_export_before_aggregate_raises(self):
        from agents.federated_learner import FederatedAggregator
        from utils.model_registry import ModelRegistry

        agents = self._make_agents(1)
        agg = FederatedAggregator(agents)

        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ModelRegistry(tmpdir)
            with pytest.raises(RuntimeError):
                agg.export_to_registry(registry)

    def test_export_multiple_rounds_increments_version(self):
        from agents.federated_learner import FederatedAggregator
        from utils.model_registry import ModelRegistry

        agents = self._make_agents(2)
        agg = FederatedAggregator(agents)

        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ModelRegistry(tmpdir)
            agg.aggregate()
            agg.export_to_registry(registry, model_name="multi")
            agg.aggregate()
            agg.export_to_registry(registry, model_name="multi")
            versions = registry.list_versions("multi")
            assert versions == ["v1", "v2"]

    def test_export_with_extra_metadata(self):
        from agents.federated_learner import FederatedAggregator
        from utils.model_registry import ModelRegistry

        agents = self._make_agents(1)
        agg = FederatedAggregator(agents)
        agg.aggregate()

        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ModelRegistry(tmpdir)
            agg.export_to_registry(
                registry,
                model_name="meta_test",
                extra_metadata={"val_outage_prob": 0.05, "env": "MultiSatelliteEnv"},
            )
            meta = registry.get_metadata("meta_test")
            assert abs(meta["val_outage_prob"] - 0.05) < 1e-9
            assert meta["env"] == "MultiSatelliteEnv"

    def test_loaded_weights_match_exported(self):
        """The registry state-dict should equal the aggregated global weights."""
        import torch
        from agents.federated_learner import FederatedAggregator
        from utils.model_registry import ModelRegistry

        agents = self._make_agents(2)
        agg = FederatedAggregator(agents)
        global_weights = agg.aggregate()

        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ModelRegistry(tmpdir)
            agg.export_to_registry(registry, model_name="weight_check")
            state_dict, _ = registry.load("weight_check")

        for k in global_weights:
            assert torch.allclose(
                global_weights[k].float(),
                state_dict[k].float(),
                atol=1e-5,
            ), f"Mismatch for key {k}"


# ---------------------------------------------------------------------------
# retrain_job.py
# ---------------------------------------------------------------------------

class TestRetrainJob:
    def test_import(self):
        import retrain_job
        assert retrain_job is not None

    def test_run_retrain_synthetic(self):
        import retrain_job
        with tempfile.TemporaryDirectory() as tmpdir:
            summary = retrain_job.run_retrain(
                registry_path=tmpdir,
                model_name="ci_test",
                n_episodes=5,
                rounds=2,
                outage_threshold=1.0,
                seed=0,
            )
        assert "promoted" in summary
        assert "candidate_metrics" in summary
        assert isinstance(summary["elapsed_s"], float)

    def test_run_retrain_promotes_first_time(self):
        """When no incumbent exists, the candidate should always be promoted."""
        import retrain_job
        with tempfile.TemporaryDirectory() as tmpdir:
            summary = retrain_job.run_retrain(
                registry_path=tmpdir,
                model_name="first_run",
                n_episodes=3,
                rounds=1,
                outage_threshold=1.0,
                seed=1,
            )
        assert summary["promoted"] is True

    def test_run_retrain_version_saved(self):
        import retrain_job
        from utils.model_registry import ModelRegistry
        with tempfile.TemporaryDirectory() as tmpdir:
            retrain_job.run_retrain(
                registry_path=tmpdir,
                model_name="version_test",
                n_episodes=3,
                rounds=1,
                outage_threshold=1.0,
                seed=2,
            )
            registry = ModelRegistry(tmpdir)
            versions = registry.list_versions("version_test")
            assert len(versions) >= 1

    def test_retrain_job_cli_exit_zero(self):
        import retrain_job
        with tempfile.TemporaryDirectory() as tmpdir:
            code = retrain_job.main([
                "--registry", tmpdir,
                "--model-name", "cli_test",
                "--n-episodes", "3",
                "--rounds", "1",
            ])
        assert code == 0
