"""
Model versioning registry for the beamforming DRL agents.

Provides :class:`ModelRegistry` which saves, loads, and lists agent
checkpoints together with their training metadata.  It is designed to be
a lightweight local equivalent of MLflow Model Registry – no extra server
required.

Stored artefacts (one directory per version)::

    registry_root/
    └── <model_name>/
        ├── v1/
        │   ├── model.pt          – PyTorch state-dict
        │   └── metadata.json     – hyperparameters, metrics, timestamp
        ├── v2/
        │   └── ...
        └── latest -> v2          – symlink (or text pointer on Windows) to the most recent version

Usage::

    from utils.model_registry import ModelRegistry

    registry = ModelRegistry("/var/lib/beamforming/models")

    # Save after training
    registry.save(
        model_name="ppo_amazon",
        model=agent.net,            # any nn.Module
        metadata={
            "val_outage_prob": 0.023,
            "train_steps": 500_000,
            "env": "MultiSatelliteEnv",
        },
    )

    # Load the latest version in production
    state_dict, meta = registry.load("ppo_amazon")
    agent.net.load_state_dict(state_dict)
"""

from __future__ import annotations

import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# ModelRegistry
# ---------------------------------------------------------------------------

class ModelRegistry:
    """
    Filesystem-based model versioning registry.

    Each model name gets a dedicated sub-directory.  Every call to
    :meth:`save` creates a new versioned sub-directory (``v1``, ``v2``, …)
    and updates the ``latest`` symlink.

    Args:
        root_dir: Path to the root directory of the registry.  It is
                  created automatically if it does not exist.
    """

    _METADATA_FILE = "metadata.json"
    _MODEL_FILE = "model.pt"
    _LATEST_ALIAS = "latest"

    def __init__(self, root_dir: str = "model_registry") -> None:
        self.root = Path(root_dir)
        self.root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save(
        self,
        model_name: str,
        model,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Save a model checkpoint and its metadata as a new version.

        Args:
            model_name: Logical name for the model (e.g. ``"ppo_amazon"``).
            model:      A ``torch.nn.Module`` whose state dict is saved.
            metadata:   Arbitrary JSON-serialisable metadata dict.

        Returns:
            Path to the version directory (e.g. ``".../ppo_amazon/v3"``).

        Raises:
            ImportError: If PyTorch is not available.
        """
        import torch  # optional dependency

        model_dir = self.root / model_name
        model_dir.mkdir(parents=True, exist_ok=True)

        version = self._next_version(model_dir)
        version_dir = model_dir / version
        version_dir.mkdir(parents=True)

        # Save state dict
        torch.save(model.state_dict(), version_dir / self._MODEL_FILE)

        # Save metadata
        full_meta = {
            "version": version,
            "model_name": model_name,
            "saved_at": datetime.now(tz=timezone.utc).isoformat(),
            **(metadata or {}),
        }
        with open(version_dir / self._METADATA_FILE, "w", encoding="utf-8") as f:
            json.dump(full_meta, f, indent=2, default=str)

        # Update 'latest' alias (symlink when possible; pointer file otherwise)
        self._set_latest_alias(model_dir, version)

        return str(version_dir)

    def load(
        self,
        model_name: str,
        version: str = "latest",
    ) -> Tuple[Dict, Dict[str, Any]]:
        """
        Load a model checkpoint by name and optional version tag.

        Args:
            model_name: Logical model name.
            version:    Version string (``"v1"``, ``"v2"``, …) or
                        ``"latest"`` (default).

        Returns:
            Tuple of (state_dict, metadata_dict).

        Raises:
            FileNotFoundError: If the model or version does not exist.
            ImportError:       If PyTorch is not available.
        """
        import torch

        version_dir = self._resolve_version_dir(model_name, version)
        model_path = version_dir / self._MODEL_FILE
        meta_path = version_dir / self._METADATA_FILE

        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        state_dict = torch.load(model_path, map_location="cpu")
        metadata: Dict[str, Any] = {}
        if meta_path.exists():
            with open(meta_path, encoding="utf-8") as f:
                metadata = json.load(f)

        return state_dict, metadata

    def list_versions(self, model_name: str) -> List[str]:
        """
        List all saved versions of ``model_name``, sorted in ascending order.

        Args:
            model_name: Logical model name.

        Returns:
            List of version strings (e.g. ``["v1", "v2", "v3"]``).
            Returns an empty list if the model does not exist.
        """
        model_dir = self.root / model_name
        if not model_dir.is_dir():
            return []
        versions = sorted(
            [
                d.name
                for d in model_dir.iterdir()
                if d.is_dir() and d.name.startswith("v")
            ],
            key=lambda s: int(s[1:]),
        )
        return versions

    def list_models(self) -> List[str]:
        """
        Return a list of all registered model names.

        Returns:
            Sorted list of model name strings.
        """
        return sorted(
            d.name
            for d in self.root.iterdir()
            if d.is_dir()
        )

    def get_metadata(
        self, model_name: str, version: str = "latest"
    ) -> Dict[str, Any]:
        """
        Return the metadata dict for a specific model version.

        Args:
            model_name: Logical model name.
            version:    Version string or ``"latest"``.

        Returns:
            Metadata dictionary.

        Raises:
            FileNotFoundError: If the metadata file is missing.
        """
        version_dir = self._resolve_version_dir(model_name, version)
        meta_path = version_dir / self._METADATA_FILE
        if not meta_path.exists():
            raise FileNotFoundError(f"Metadata not found: {meta_path}")
        with open(meta_path, encoding="utf-8") as f:
            return json.load(f)

    def delete_version(self, model_name: str, version: str) -> None:
        """
        Delete a specific model version.

        If the deleted version was ``latest``, the symlink is updated
        to point at the most recent remaining version (if any).

        Args:
            model_name: Logical model name.
            version:    Version string to delete (must not be ``"latest"``).

        Raises:
            ValueError:        If ``version == "latest"``.
            FileNotFoundError: If the version directory does not exist.
        """
        if version == "latest":
            raise ValueError("Cannot delete the 'latest' alias directly; "
                             "specify an explicit version string.")
        version_dir = self._resolve_version_dir(model_name, version)
        shutil.rmtree(version_dir)

        # Re-point latest
        model_dir = self.root / model_name
        remaining = self.list_versions(model_name)
        if remaining:
            self._set_latest_alias(model_dir, remaining[-1])
        else:
            self._remove_latest_alias(model_dir)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _next_version(self, model_dir: Path) -> str:
        existing = self.list_versions(model_dir.name)
        if not existing:
            return "v1"
        last_n = int(existing[-1][1:])
        return f"v{last_n + 1}"

    def _resolve_version_dir(self, model_name: str, version: str) -> Path:
        model_dir = self.root / model_name
        if not model_dir.is_dir():
            raise FileNotFoundError(f"No model named '{model_name}' in registry")
        if version == "latest":
            latest_alias = model_dir / self._LATEST_ALIAS
            if not latest_alias.exists():
                raise FileNotFoundError(f"No 'latest' version for '{model_name}'")

            # Preferred: symlink/junction
            if latest_alias.is_symlink():
                return latest_alias.resolve()

            # Windows-friendly fallback: plain text file containing the version string (e.g. "v3")
            if latest_alias.is_file():
                target = latest_alias.read_text(encoding="utf-8").strip()
                if not target:
                    raise FileNotFoundError(
                        f"Invalid 'latest' pointer for '{model_name}': empty file"
                    )
                version_dir = model_dir / target
                if not version_dir.is_dir():
                    raise FileNotFoundError(
                        f"Invalid 'latest' pointer for '{model_name}': {target} does not exist"
                    )
                return version_dir

            raise FileNotFoundError(
                f"Invalid 'latest' alias for '{model_name}': not a symlink or file"
            )
        version_dir = model_dir / version
        if not version_dir.is_dir():
            raise FileNotFoundError(
                f"Version '{version}' not found for model '{model_name}'"
            )
        return version_dir

    def _remove_latest_alias(self, model_dir: Path) -> None:
        latest_alias = model_dir / self._LATEST_ALIAS
        if not latest_alias.exists() and not latest_alias.is_symlink():
            return
        try:
            if latest_alias.is_symlink() or latest_alias.is_file():
                latest_alias.unlink()
            elif latest_alias.is_dir():
                shutil.rmtree(latest_alias)
        except OSError:
            # Best-effort cleanup; callers treat missing alias as "no latest".
            return

    def _set_latest_alias(self, model_dir: Path, version: str) -> None:
        self._remove_latest_alias(model_dir)
        latest_alias = model_dir / self._LATEST_ALIAS

        # Try to create a symlink/junction first.
        try:
            latest_alias.symlink_to(version, target_is_directory=True)
            return
        except (OSError, NotImplementedError):
            pass

        # Fallback: write a pointer file.
        latest_alias.write_text(f"{version}\n", encoding="utf-8")
