"""Gymnasium MDP environments sub-package."""

from .multi_satellite_env import MultiSatelliteEnv
from .regulatory_env import RegulatoryEnv, ExclusionZone, GeoRegulatoryEnv
from .traffic_env import TrafficAwareMultiSatelliteEnv

# h5py / torch dependent modules: imported conditionally so that the rest of
# the package remains usable in environments where those are not installed.
try:
    from .leo_beamforming_env import LEOBeamformingEnv
    _LEO_ENV_AVAILABLE = True
except ImportError:
    _LEO_ENV_AVAILABLE = False

try:
    from .offline_env import OfflineLEOEnv
    _OFFLINE_ENV_AVAILABLE = True
except ImportError:
    _OFFLINE_ENV_AVAILABLE = False

try:
    from .gnn_beamforming_env import GNNBeamformingEnv
    _GNN_ENV_AVAILABLE = True
except ImportError:
    _GNN_ENV_AVAILABLE = False

__all__ = [
    "LEOBeamformingEnv",
    "OfflineLEOEnv",
    "MultiSatelliteEnv",
    "GNNBeamformingEnv",
    "RegulatoryEnv",
    "ExclusionZone",
    "GeoRegulatoryEnv",
    "TrafficAwareMultiSatelliteEnv",
]
