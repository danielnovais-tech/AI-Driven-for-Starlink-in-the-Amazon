"""
Tests for ModelRegistry (model versioning).
"""

import sys
import os
import json
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest


class TestModelRegistryWithoutTorch:
    """Tests that do NOT require torch (metadata and filesystem operations)."""

    def _make_registry(self, tmp_path):
        from utils.model_registry import ModelRegistry
        return ModelRegistry(root_dir=str(tmp_path))

    def test_import(self):
        from utils.model_registry import ModelRegistry
        assert ModelRegistry is not None

    def test_exported_from_utils(self):
        from utils import ModelRegistry
        assert ModelRegistry is not None

    def test_empty_registry_list_models(self, tmp_path):
        reg = self._make_registry(tmp_path)
        assert reg.list_models() == []

    def test_list_versions_missing_model(self, tmp_path):
        reg = self._make_registry(tmp_path)
        assert reg.list_versions("nonexistent") == []

    def test_load_missing_model_raises(self, tmp_path):
        from utils.model_registry import ModelRegistry
        reg = ModelRegistry(root_dir=str(tmp_path))
        with pytest.raises(FileNotFoundError):
            reg.load("no_such_model")

    def test_delete_latest_alias_raises(self, tmp_path):
        reg = self._make_registry(tmp_path)
        # Create model directory manually so it "exists"
        (tmp_path / "my_model").mkdir()
        with pytest.raises(ValueError):
            reg.delete_version("my_model", "latest")


class TestModelRegistryWithTorch:
    """Tests that require torch for save/load."""

    @pytest.fixture
    def registry(self, tmp_path):
        from utils.model_registry import ModelRegistry
        return ModelRegistry(root_dir=str(tmp_path))

    def _make_model(self):
        import torch.nn as nn
        return nn.Linear(7, 4)

    def test_save_creates_version_dir(self, registry, tmp_path):
        import torch
        model = self._make_model()
        path = registry.save("test_model", model, metadata={"lr": 3e-4})
        assert os.path.isdir(path)

    def test_save_creates_model_file(self, registry, tmp_path):
        import torch
        model = self._make_model()
        path = registry.save("test_model2", model)
        assert os.path.isfile(os.path.join(path, "model.pt"))

    def test_save_creates_metadata_file(self, registry, tmp_path):
        model = self._make_model()
        path = registry.save("test_model3", model, metadata={"val_loss": 0.05})
        meta_path = os.path.join(path, "metadata.json")
        assert os.path.isfile(meta_path)
        with open(meta_path) as f:
            meta = json.load(f)
        assert meta["val_loss"] == 0.05
        assert "saved_at" in meta

    def test_versions_increment(self, registry):
        model = self._make_model()
        registry.save("versioned_model", model)
        registry.save("versioned_model", model)
        registry.save("versioned_model", model)
        versions = registry.list_versions("versioned_model")
        assert versions == ["v1", "v2", "v3"]

    def test_list_models_includes_saved(self, registry):
        model = self._make_model()
        registry.save("alpha_model", model)
        registry.save("beta_model", model)
        models = registry.list_models()
        assert "alpha_model" in models
        assert "beta_model" in models

    def test_load_latest_state_dict(self, registry):
        import torch
        model = self._make_model()
        registry.save("load_test", model)
        state_dict, meta = registry.load("load_test")
        assert isinstance(state_dict, dict)
        assert "weight" in str(list(state_dict.keys()))

    def test_load_returns_metadata(self, registry):
        model = self._make_model()
        registry.save("meta_test", model, metadata={"accuracy": 0.99})
        _, meta = registry.load("meta_test")
        assert abs(meta["accuracy"] - 0.99) < 1e-9

    def test_load_specific_version(self, registry):
        import torch
        model = self._make_model()
        registry.save("versioned_load", model)
        registry.save("versioned_load", model)
        _, meta = registry.load("versioned_load", version="v1")
        assert meta["version"] == "v1"

    def test_get_metadata(self, registry):
        model = self._make_model()
        registry.save("meta_get_test", model, metadata={"tag": "production"})
        meta = registry.get_metadata("meta_get_test")
        assert meta["tag"] == "production"

    def test_delete_version(self, registry):
        model = self._make_model()
        registry.save("del_test", model)
        registry.save("del_test", model)
        registry.delete_version("del_test", "v1")
        versions = registry.list_versions("del_test")
        assert "v1" not in versions
        assert "v2" in versions

    def test_delete_updates_latest(self, registry):
        model = self._make_model()
        registry.save("latest_del", model)
        registry.save("latest_del", model)
        registry.save("latest_del", model)
        registry.delete_version("latest_del", "v3")
        # latest should now point to v2
        _, meta = registry.load("latest_del")
        assert meta["version"] == "v2"
