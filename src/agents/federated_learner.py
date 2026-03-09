"""
Federated learning for multi-satellite beamforming agents.

Implements the Federated Averaging (FedAvg) algorithm adapted for the
LEO beamforming use case:

    - Each satellite maintains a local :class:`OnlinePPOAgent` and trains
      on its own stream of experiences.
    - A central :class:`FederatedAggregator` collects model weights from
      all satellites, computes a weighted average, and distributes the
      global model back.
    - Only model weights (not raw experience tuples) are transmitted,
      minimising inter-satellite communication overhead.

Design goals:
    - Satellites train locally for ``local_epochs`` gradient steps before
      syncing; this amortises the communication cost.
    - The aggregator weights each satellite's contribution by the number
      of transitions it has seen since the last round (sample-count
      weighting, not uniform averaging).
    - No external framework (Flower, PySyft) required; the implementation
      uses pure PyTorch state-dict manipulation.

References:
    McMahan et al. (2017) – "Communication-Efficient Learning of Deep
    Networks from Decentralized Data." (FedAvg)

Usage::

    from agents.federated_learner import SatelliteAgent, FederatedAggregator

    # Create one local agent per satellite
    agents = [SatelliteAgent(sat_id=i, state_dim=7, action_dim=4) for i in range(4)]

    # Federated training loop
    aggregator = FederatedAggregator(agents)

    for round_idx in range(100):
        for agent in agents:
            # Each satellite collects transitions from its own orbit segment
            for step in range(agent.local_steps):
                obs = ...
                action, _ = agent.get_action(obs)
                agent.store_transition(obs, action, reward, next_obs, done)
            agent.local_train()   # local PPO updates

        # Synchronise: weighted FedAvg
        aggregator.aggregate()
        aggregator.broadcast()
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional

import numpy as np

from .online_ppo import OnlinePPOAgent


# ---------------------------------------------------------------------------
# SatelliteAgent
# ---------------------------------------------------------------------------

class SatelliteAgent(OnlinePPOAgent):
    """
    Local DRL agent running on (or for) a single satellite.

    Extends :class:`~agents.online_ppo.OnlinePPOAgent` with:
        - A ``satellite_id`` for identification and logging.
        - ``local_train()`` which runs ``local_epochs`` PPO mini-batch
          updates on the local replay buffer.
        - ``get_weights()`` / ``set_weights()`` helpers for weight exchange
          with the aggregator.
        - A ``samples_since_last_sync`` counter used by the aggregator for
          weighted averaging.

    Args:
        satellite_id:  Integer or string identifier for this satellite.
        local_epochs:  Number of local PPO epochs to run before syncing.
        local_steps:   Transitions to collect locally between syncs
                       (informational; the main loop controls this).
        state_dim:     State-space dimensionality.
        action_dim:    Action-space dimensionality.
        **kwargs:      All remaining keyword arguments are forwarded to
                       :class:`~agents.online_ppo.OnlinePPOAgent`.
    """

    def __init__(
        self,
        satellite_id,
        state_dim: int = 7,
        action_dim: int = 4,
        local_epochs: int = 5,
        local_steps: int = 100,
        **kwargs,
    ) -> None:
        super().__init__(
            state_dim=state_dim,
            action_dim=action_dim,
            **kwargs,
        )
        self.satellite_id = satellite_id
        self.local_epochs = local_epochs
        self.local_steps = local_steps
        self.samples_since_last_sync: int = 0

        # Override parent update_freq so local_train() always triggers
        self.update_freq = 1

    # ------------------------------------------------------------------
    # Weight exchange
    # ------------------------------------------------------------------

    def get_weights(self) -> Dict[str, Any]:
        """
        Return the network state-dict (deep copy, safe to send remotely).

        Returns:
            Dictionary mapping parameter names to tensors.
        """
        import torch
        return {k: v.cpu().clone() for k, v in self.net.state_dict().items()}

    def set_weights(self, weights: Dict[str, Any]) -> None:
        """
        Replace the network parameters with a given state-dict.

        Args:
            weights: State-dict returned by another agent's
                     :meth:`get_weights` or produced by the aggregator.
        """
        import torch
        self.net.load_state_dict(weights)

    # ------------------------------------------------------------------
    # Local training
    # ------------------------------------------------------------------

    def store_transition(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        td_error: Optional[float] = None,  # accepted for API compatibility but not forwarded
    ) -> None:
        """
        Push a transition and track the sample count for FedAvg weighting.

        ``td_error`` is accepted for API compatibility with
        :class:`~agents.online_ppo.PEROnlinePPOAgent`, but is not forwarded
        to :class:`~agents.online_ppo.OnlinePPOAgent` whose FIFO buffer does
        not support priority-based sampling.

        Args: See :class:`~agents.online_ppo.OnlinePPOAgent.store_transition`.
        """
        # OnlinePPOAgent.store_transition uses a plain FIFO deque and does
        # not accept td_error; pass only the five core arguments.
        super().store_transition(state, action, reward, next_state, done)
        self.samples_since_last_sync += 1

    def local_train(self) -> Optional[float]:
        """
        Run ``local_epochs`` PPO mini-batch updates on the local buffer.

        This is the *local computation* step of FedAvg: each satellite
        trains its own copy of the global model using its private data
        before sending weights back to the aggregator.

        Returns:
            Mean loss across all local epochs, or ``None`` if the buffer
            is too small to form a mini-batch.
        """
        if len(self.buffer) < self.batch_size:
            return None

        total_loss = 0.0
        successful = 0
        # Temporarily override update_freq to force updates
        saved_freq = self.update_freq
        saved_counter = self._step_counter
        self.update_freq = 1

        for epoch in range(self.local_epochs):
            # Force the step counter to hit the update gate
            self._step_counter = epoch + 1
            loss = self.update_online()
            if loss is not None:
                total_loss += loss
                successful += 1

        # Restore state
        self.update_freq = saved_freq
        self._step_counter = saved_counter
        return total_loss / max(1, successful)

    def reset_sync_counter(self) -> None:
        """Reset the sample count accumulated since the last synchronisation."""
        self.samples_since_last_sync = 0


# ---------------------------------------------------------------------------
# FederatedAggregator
# ---------------------------------------------------------------------------

class FederatedAggregator:
    """
    Central coordinator that aggregates local model weights via FedAvg.

    Maintains references to all :class:`SatelliteAgent` instances (or
    receives state-dicts from them in a distributed deployment) and
    produces a single global model by computing a sample-count-weighted
    average of the parameter tensors.

    Args:
        agents:          List of :class:`SatelliteAgent` instances that
                         participate in federated training.  In a real
                         deployment these would be replaced by RPC calls.
        min_participants: Minimum number of agents that must report weights
                         before aggregation proceeds (default 1 = aggregation
                         always runs).
    """

    def __init__(
        self,
        agents: List[SatelliteAgent],
        min_participants: int = 1,
    ) -> None:
        if not agents:
            raise ValueError("FederatedAggregator requires at least one agent.")
        self.agents = agents
        self.min_participants = max(1, min_participants)
        self._round: int = 0
        # Store global weights as reference (copy of first agent after init)
        self._global_weights: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------
    # Core FedAvg protocol
    # ------------------------------------------------------------------

    def aggregate(self, participants: Optional[List[SatelliteAgent]] = None) -> Dict[str, Any]:
        """
        Compute the weighted average of participant model weights (FedAvg).

        The weight of each agent's contribution is proportional to the
        number of samples it has collected since the last synchronisation.
        If all agents have zero new samples (e.g. at the very start), uniform
        weighting is used instead.

        Args:
            participants: Subset of agents to aggregate.  Defaults to
                          ``self.agents``.

        Returns:
            Global state-dict (averaged weights as a new dict).

        Raises:
            RuntimeError: If fewer than ``min_participants`` agents are
                          available.
        """
        import torch

        if participants is None:
            participants = self.agents

        if len(participants) < self.min_participants:
            raise RuntimeError(
                f"FedAvg requires at least {self.min_participants} participants; "
                f"only {len(participants)} available."
            )

        # Compute sample counts for weighting
        sample_counts = np.array(
            [max(1, a.samples_since_last_sync) for a in participants],
            dtype=np.float64,
        )
        weights = sample_counts / sample_counts.sum()

        # Collect state-dicts
        state_dicts = [a.get_weights() for a in participants]
        param_names = list(state_dicts[0].keys())

        global_weights: Dict[str, Any] = {}
        for name in param_names:
            # Weighted sum of tensors
            weighted_sum = sum(
                w * state_dicts[i][name].float()
                for i, w in enumerate(weights)
            )
            global_weights[name] = weighted_sum

        self._global_weights = global_weights
        self._round += 1
        return global_weights

    def broadcast(self, participants: Optional[List[SatelliteAgent]] = None) -> None:
        """
        Push the current global weights to all participants and reset their
        sample counters.

        Args:
            participants: Agents to update.  Defaults to ``self.agents``.

        Raises:
            RuntimeError: If :meth:`aggregate` has not been called yet.
        """
        if self._global_weights is None:
            raise RuntimeError("Call aggregate() before broadcast().")

        if participants is None:
            participants = self.agents

        for agent in participants:
            agent.set_weights(copy.deepcopy(self._global_weights))
            agent.reset_sync_counter()

    def get_global_weights(self) -> Optional[Dict[str, Any]]:
        """
        Return the most recent global averaged weights, or ``None`` if
        :meth:`aggregate` has not yet been called.
        """
        return copy.deepcopy(self._global_weights) if self._global_weights else None

    @property
    def round(self) -> int:
        """Number of completed federated aggregation rounds."""
        return self._round

    # ------------------------------------------------------------------
    # Simulation helper
    # ------------------------------------------------------------------

    def run_round(
        self,
        participants: Optional[List[SatelliteAgent]] = None,
    ) -> Dict[str, Any]:
        """
        Convenience method: call :meth:`aggregate` then :meth:`broadcast`.

        Returns:
            The global weights produced by aggregation.
        """
        weights = self.aggregate(participants)
        self.broadcast(participants)
        return weights
