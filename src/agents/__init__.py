"""DRL agents sub-package."""

from .networks import BeamformingNetwork
from .dqn_agent import DQNAgent
from .ppo_agent import PPOAgent
from .a3c_agent import A3CWorker, run_a3c
from .online_ppo import OnlinePPOAgent, PEROnlinePPOAgent
from .per_buffer import PrioritizedReplayBuffer

# torch-dependent federated learning module
try:
    from .federated_learner import SatelliteAgent, FederatedAggregator
    _FEDERATED_AVAILABLE = True
except ImportError:
    _FEDERATED_AVAILABLE = False

# torch-geometric-dependent GNN PPO agent
try:
    from .gnn_ppo_agent import GNNPPOAgent
    _GNN_AVAILABLE = True
except ImportError:
    _GNN_AVAILABLE = False

__all__ = [
    "BeamformingNetwork",
    "DQNAgent",
    "PPOAgent",
    "A3CWorker",
    "run_a3c",
    "OnlinePPOAgent",
    "PEROnlinePPOAgent",
    "PrioritizedReplayBuffer",
    "SatelliteAgent",
    "FederatedAggregator",
    "GNNPPOAgent",
]
