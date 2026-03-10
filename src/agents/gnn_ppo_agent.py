"""
PPO agent that uses ``_GNNActorCritic`` as its policy network.

This module provides the bridge between the graph-structured
multi-satellite environment (:class:`~envs.gnn_beamforming_env.GNNBeamformingEnv`)
and a standard PPO training loop.

``GNNPPOAgent`` wraps :class:`_GNNActorCritic`, a two-layer Graph Attention
Network actor-critic, and exposes the same ``get_action`` / ``store_transition``
/ ``update_online`` interface as the other agents in the package, so that it
can be swapped into the :class:`~inference.online_controller.OnlineBeamController`
and :class:`~agents.federated_learner.FederatedAggregator` without changes.

Architecture:
    - **Actor**: GNN → 4-dim action mean; log-std as learnable parameters.
    - **Critic**: GNN → per-node value, pooled to a scalar via mean.

Training:
    The agent expects graph observations (``HeteroData``) produced by
    :class:`~envs.gnn_beamforming_env.GNNBeamformingEnv`.  Each training
    step stores the graph and the associated flat action / reward tuple in
    a FIFO replay buffer.  ``update_online()`` runs PPO mini-batch updates
    identical to :class:`~agents.online_ppo.OnlinePPOAgent`.

Requirements:
    ``torch_geometric`` must be installed::

        pip install torch-geometric

Usage::

    from agents.gnn_ppo_agent import GNNPPOAgent
    from envs.gnn_beamforming_env import GNNBeamformingEnv

    env = GNNBeamformingEnv(...)
    agent = GNNPPOAgent(node_features=4, action_dim=5)

    obs, info = env.reset()
    graph = info["graph_obs"]
    action = agent.get_action(graph)
    # action is the index of the best satellite (0 … n_sats-1)
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Any, Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional torch_geometric guard
# ---------------------------------------------------------------------------
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import GATConv
    _TORCH_GEO_AVAILABLE = True
except ImportError:
    _TORCH_GEO_AVAILABLE = False


def _require_torch_geo():
    if not _TORCH_GEO_AVAILABLE:
        raise ImportError(
            "GNNPPOAgent requires torch_geometric. "
            "Install with: pip install torch-geometric"
        )


# ---------------------------------------------------------------------------
# Actor-Critic network (GNN backbone)
# ---------------------------------------------------------------------------

class _GNNActorCritic(nn.Module if _TORCH_GEO_AVAILABLE else object):
    """
    GNN-based actor-critic network for multi-satellite beamforming.

    Produces:
        - ``action_logits`` of shape ``(N_sat,)`` – one logit per satellite
          (softmax → discrete satellite selection distribution).
        - ``value`` of shape ``(1,)`` – scalar state value (mean of
          per-satellite value estimates).

    Args:
        node_features: Input features per satellite node.
        hidden:        Hidden dimension for GAT layers.
        gat_heads:     Number of attention heads.
        dropout:       Dropout probability.
    """

    def __init__(
        self,
        node_features: int,
        hidden: int = 64,
        gat_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        _require_torch_geo()
        super().__init__()
        self.dropout = dropout

        self.sat_enc = nn.Linear(node_features, hidden)
        self.gs_enc = nn.Linear(node_features, hidden)

        self.conv1 = GATConv(hidden, hidden, heads=gat_heads, concat=False,
                             dropout=dropout, add_self_loops=False)
        self.conv2 = GATConv(hidden, hidden, heads=gat_heads, concat=False,
                             dropout=dropout, add_self_loops=False)

        self.actor_head = nn.Linear(hidden, 1)   # per-node logit
        self.critic_head = nn.Linear(hidden, 1)  # per-node value

        for layer in [self.sat_enc, self.gs_enc,
                      self.actor_head, self.critic_head]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.constant_(layer.bias, 0.0)

    def forward(self, data) -> Tuple["torch.Tensor", "torch.Tensor"]:
        """
        Forward pass.

        Args:
            data: ``HeteroData`` from :class:`~envs.gnn_beamforming_env.GNNBeamformingEnv`.

        Returns:
            Tuple ``(action_logits, value)`` where:
              - ``action_logits`` shape ``(N_sat,)``
              - ``value``         shape ``(1,)``
        """
        x_sat = F.relu(self.sat_enc(data["sat"].x))
        x_gs = F.relu(self.gs_enc(data["ground_station"].x))
        x = torch.cat([x_sat, x_gs], dim=0)

        edge_index = data["sat", "to", "ground_station"].edge_index

        x = F.dropout(
            F.relu(self.conv1(x, edge_index)), p=self.dropout, training=self.training
        )
        x = F.dropout(
            F.relu(self.conv2(x, edge_index)), p=self.dropout, training=self.training
        )

        n_sat = data["sat"].num_nodes
        x_sat_out = x[:n_sat]

        logits = self.actor_head(x_sat_out).squeeze(-1)  # (N_sat,)
        value = self.critic_head(x_sat_out).mean(dim=0, keepdim=True)  # (1,)
        return logits, value


# ---------------------------------------------------------------------------
# GNNPPOAgent
# ---------------------------------------------------------------------------

class GNNPPOAgent:
    """
    PPO agent backed by a GNN actor-critic for multi-satellite handover.

    Exposes the same interface as :class:`~agents.online_ppo.OnlinePPOAgent`
    so it can be used in :class:`~agents.federated_learner.SatelliteAgent`.

    Actions are discrete satellite indices (0 … N_sat-1).  The agent
    selects the satellite whose ``action_logit`` is highest (deterministic)
    or samples from the softmax distribution (stochastic).

    Args:
        node_features: Number of features per satellite node.
        hidden:        GAT hidden dimension.
        gat_heads:     Number of attention heads.
        dropout:       Dropout probability.
        lr:            Adam learning rate.
        gamma:         Discount factor.
        clip_eps:      PPO clipping epsilon.
        epochs:        Number of gradient steps per mini-batch update.
        value_coef:    Critic loss coefficient.
        entropy_coef:  Entropy bonus coefficient.
        max_grad_norm: Gradient clipping norm.
        device:        Torch device string (``"cpu"`` or ``"cuda"``).
        buffer_size:   Replay buffer capacity.
        batch_size:    Transitions sampled per PPO update.
        update_freq:   Minimum ``store_transition`` calls between updates.
    """

    def __init__(
        self,
        node_features: int = 4,
        hidden: int = 64,
        gat_heads: int = 4,
        dropout: float = 0.1,
        lr: float = 3e-4,
        gamma: float = 0.99,
        clip_eps: float = 0.2,
        epochs: int = 4,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        device: str = "cpu",
        buffer_size: int = 5_000,
        batch_size: int = 32,
        update_freq: int = 64,
    ) -> None:
        _require_torch_geo()

        self.gamma = gamma
        self.clip_eps = clip_eps
        self.epochs = epochs
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.batch_size = batch_size
        self.update_freq = update_freq
        self.device = torch.device(device)

        self.net = _GNNActorCritic(
            node_features=node_features,
            hidden=hidden,
            gat_heads=gat_heads,
            dropout=dropout,
        ).to(self.device)

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)

        # Replay buffer stores (graph, action, reward, next_graph, done) tuples
        self.buffer: deque = deque(maxlen=buffer_size)
        self._step_counter: int = 0

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def get_action(
        self,
        graph,
        deterministic: bool = False,
    ) -> Tuple[int, float]:
        """
        Select a satellite index from the current graph observation.

        Args:
            graph:         ``HeteroData`` graph from the environment.
            deterministic: If ``True``, return the argmax; otherwise
                           sample from the softmax distribution.

        Returns:
            Tuple ``(action_int, log_prob_float)``.
        """
        self.net.eval()
        with torch.no_grad():
            graph = self._to_device(graph)
            logits, _ = self.net(graph)
            dist = torch.distributions.Categorical(logits=logits)
            if deterministic:
                action = int(logits.argmax().item())
            else:
                action = int(dist.sample().item())
            log_prob = float(dist.log_prob(torch.tensor(action)).item())
        return action, log_prob

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def store_transition(
        self,
        graph,
        action: int,
        reward: float,
        next_graph,
        done: bool,
    ) -> None:
        """
        Push a (graph, action, reward, next_graph, done) transition.

        Args:
            graph:      Current graph observation.
            action:     Selected satellite index.
            reward:     Scalar reward.
            next_graph: Next graph observation.
            done:       Terminal flag.
        """
        self.buffer.append((graph, int(action), float(reward), next_graph, float(done)))
        self._step_counter += 1

    def update_online(self) -> Optional[float]:
        """
        Perform one PPO mini-batch update from the replay buffer.

        Returns:
            Mean loss (float) or ``None`` if the update was skipped.
        """
        if len(self.buffer) < self.batch_size:
            return None
        if self._step_counter % self.update_freq != 0:
            return None

        indices = np.random.choice(len(self.buffer), self.batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        graphs, actions, rewards, next_graphs, dones = zip(*batch)

        actions_t = torch.tensor(actions, dtype=torch.long, device=self.device)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones_t = torch.tensor(dones, dtype=torch.float32, device=self.device)

        # Bootstrap value estimates
        with torch.no_grad():
            next_values = torch.stack([
                self.net(self._to_device(g))[1].squeeze()
                for g in next_graphs
            ])
            target_values = rewards_t + self.gamma * (1.0 - dones_t) * next_values

        # Old log-probs under current policy (reference point)
        with torch.no_grad():
            old_log_probs = torch.stack([
                torch.distributions.Categorical(
                    logits=self.net(self._to_device(g))[0]
                ).log_prob(a)
                for g, a in zip(graphs, actions_t)
            ])

        # PPO epochs
        total_loss = 0.0
        for _ in range(self.epochs):
            log_probs = torch.stack([
                torch.distributions.Categorical(
                    logits=self.net(self._to_device(g))[0]
                ).log_prob(a)
                for g, a in zip(graphs, actions_t)
            ])
            values = torch.stack([
                self.net(self._to_device(g))[1].squeeze()
                for g in graphs
            ])
            entropy = torch.stack([
                torch.distributions.Categorical(
                    logits=self.net(self._to_device(g))[0]
                ).entropy()
                for g in graphs
            ]).mean()

            advantages = target_values - values.detach()
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            ratios = torch.exp(log_probs - old_log_probs)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.clip_eps, 1 + self.clip_eps) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = F.mse_loss(values, target_values)
            loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
            self.optimizer.step()
            total_loss += loss.item()

        return total_loss / self.epochs

    # ------------------------------------------------------------------
    # Weight exchange (for FederatedAggregator compatibility)
    # ------------------------------------------------------------------

    def get_weights(self) -> Dict[str, Any]:
        """Return the network state-dict (deep copy)."""
        return {k: v.cpu().clone() for k, v in self.net.state_dict().items()}

    def set_weights(self, weights: Dict[str, Any]) -> None:
        """Replace network parameters with the given state-dict."""
        self.net.load_state_dict(weights)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _to_device(self, graph):
        """Move all tensors in a ``HeteroData`` object to ``self.device``."""
        return graph.to(self.device)
