"""
GNN-based multi-satellite coordination environment.

Wraps ``MultiSatelliteEnv`` to supply a graph-structured observation
compatible with ``CognitiveBeamGNN``.  Each episode step produces a
``torch_geometric.data.HeteroData`` object alongside the flat Gymnasium
observation, enabling joint training of a GNN policy with standard DRL
algorithms.

When ``torch_geometric`` is not installed the environment degrades
gracefully: ``step()`` and ``reset()`` return the standard flat
observation from the parent class and the ``graph_obs`` key in ``info``
is set to ``None``.

Graph structure:
    - Satellite nodes: ``[SNR, distance, elevation, rain_rate]`` per visible sat.
    - Ground-station node: ``[0, 0, 0, 0]`` (placeholder for future extensions).
    - Edges: satellite → ground_station for each visible satellite.

Reference:
    Velickovic et al. (2018) – Graph Attention Networks.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .multi_satellite_env import MultiSatelliteEnv

# ---------------------------------------------------------------------------
# Optional torch_geometric import
# ---------------------------------------------------------------------------
try:
    import torch
    import torch_geometric  # noqa: F401
    from torch_geometric.data import HeteroData
    _TORCH_GEO_AVAILABLE = True
except ImportError:
    _TORCH_GEO_AVAILABLE = False


def _build_hetero_graph(
    sat_features: np.ndarray,  # (N_vis, 4)
    gs_features: Optional[np.ndarray] = None,  # (1, 4)
) -> "HeteroData":  # type: ignore[name-defined]
    """
    Construct a heterogeneous graph for the current environment step.

    Args:
        sat_features:  (N_vis, 4) array with [SNR, dist, elev, rain] per sat.
        gs_features:   (1, 4) array for the single ground station.

    Returns:
        ``HeteroData`` with satellite and ground-station nodes connected
        by directed edges (satellite → ground_station).
    """
    import torch as _torch  # local import to avoid import error at module level
    from torch_geometric.data import HeteroData as _HeteroData

    data = _HeteroData()
    n_sats = sat_features.shape[0]

    data["sat"].x = _torch.FloatTensor(sat_features)
    data["sat"].num_nodes = n_sats

    if gs_features is None:
        gs_features = np.zeros((1, sat_features.shape[1]), dtype=np.float32)
    data["ground_station"].x = _torch.FloatTensor(gs_features)
    data["ground_station"].num_nodes = 1

    # Each satellite is connected to the single ground station
    src = _torch.arange(n_sats, dtype=_torch.long)
    dst = _torch.zeros(n_sats, dtype=_torch.long)
    data["sat", "to", "ground_station"].edge_index = _torch.stack([src, dst], dim=0)

    return data


class GNNBeamformingEnv(MultiSatelliteEnv):
    """
    Multi-satellite environment with graph-structured observations.

    Extends :class:`MultiSatelliteEnv` by building a ``HeteroData`` graph
    at each step and attaching it to the ``info`` dictionary under the key
    ``'graph_obs'``.  The flat Gymnasium observation is unchanged.

    Args:
        All arguments are forwarded to :class:`MultiSatelliteEnv`.
        node_feature_dim: Number of features per satellite node (default 4:
                          SNR, distance, elevation, rain_rate).

    Note:
        If ``torch_geometric`` is not installed ``info['graph_obs']`` is
        ``None`` and the environment still functions as a standard
        ``MultiSatelliteEnv``.
    """

    def __init__(self, *args, node_feature_dim: int = 4, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.node_feature_dim = node_feature_dim
        self._graph_available = _TORCH_GEO_AVAILABLE

    # ------------------------------------------------------------------
    # Override Gymnasium API to attach graph observation
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        obs, info = super().reset(seed=seed, options=options)
        info["graph_obs"] = self._build_graph_obs() if self._graph_available else None
        return obs, info

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        obs, reward, terminated, truncated, info = super().step(action)
        info["graph_obs"] = self._build_graph_obs() if self._graph_available else None
        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_graph_obs(self) -> Optional[Any]:
        """
        Build a ``HeteroData`` object from the current visible satellites.

        Returns ``None`` if no satellites are visible.
        """
        if not self.visible_sats:
            return None

        n_vis = len(self.visible_sats)
        sat_feat = np.zeros((n_vis, self.node_feature_dim), dtype=np.float32)
        gs = self._gs_pos()

        for i, sat in enumerate(self.visible_sats):
            snr = self._compute_snr(sat)
            dist = float(np.linalg.norm(sat - gs))
            elev = self._elevation(sat)
            rain = float(self.radar.get_at_location(sat))
            sat_feat[i] = [snr, dist, elev, rain]

        return _build_hetero_graph(sat_feat)
