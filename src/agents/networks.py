"""
Shared neural network architecture for all DRL agents.

The ``BeamformingNetwork`` provides a three-layer MLP with shared feature
extraction and two heads:
    - Policy head (mean of a Gaussian policy for PPO/A3C).
    - Value head (state-value estimate V(s)).

For DQN the architecture is sub-classed into ``DQNNetwork`` which replaces
the policy head with a Q-value head.

Reference:
    Mnih et al. (2015) – Human-level control through deep reinforcement
        learning (DQN).
    Schulman et al. (2017) – Proximal Policy Optimization Algorithms.
    Mnih et al. (2016) – Asynchronous Methods for Deep Reinforcement
        Learning (A3C).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class BeamformingNetwork(nn.Module):
    """
    Shared actor-critic network for continuous action spaces (PPO, A3C).

    Args:
        state_dim:  Dimensionality of the (normalised) state vector.
        action_dim: Dimensionality of the action vector.
        hidden:     Number of hidden units per layer.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden: int = 256) -> None:
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, hidden)

        # Policy head: outputs mean of Gaussian distribution
        self.mean_head = nn.Linear(hidden, action_dim)
        # Value head: outputs scalar V(s)
        self.value_head = nn.Linear(hidden, 1)
        # Learnable log std (shared across states, per action dimension)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

        self._init_weights()

    def _init_weights(self) -> None:
        for layer in [self.fc1, self.fc2, self.fc3]:
            nn.init.orthogonal_(layer.weight, gain=1.0)
            nn.init.constant_(layer.bias, 0.0)
        nn.init.orthogonal_(self.mean_head.weight, gain=0.01)
        nn.init.constant_(self.mean_head.bias, 0.0)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.constant_(self.value_head.bias, 0.0)

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x

    def forward(self, x: torch.Tensor):
        """
        Forward pass returning (mean, value).

        Args:
            x: State tensor of shape (batch, state_dim).

        Returns:
            mean:  Action mean tensor of shape (batch, action_dim).
            value: State value tensor of shape (batch, 1).
        """
        feat = self._features(x)
        mean = torch.tanh(self.mean_head(feat)) * 2.0  # squash to (−2, 2)
        value = self.value_head(feat)
        return mean, value

    def get_action(self, state: torch.Tensor, deterministic: bool = False):
        """
        Sample or compute a deterministic action from the current policy.

        Args:
            state:         State tensor of shape (1, state_dim) or (state_dim,).
            deterministic: If True, return the mean action without sampling.

        Returns:
            action:   Action tensor.
            log_prob: Log-probability of the sampled action (scalar tensor).
            value:    State-value estimate (scalar tensor).
        """
        mean, value = self.forward(state)
        std = self.log_std.exp()
        dist = Normal(mean, std)
        if deterministic:
            action = mean
        else:
            action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob, value


class DQNNetwork(nn.Module):
    """
    Q-network for discrete action spaces (DQN).

    Args:
        state_dim:  Dimensionality of the normalised state vector.
        n_actions:  Total number of discrete actions.
        hidden:     Hidden layer width.
    """

    def __init__(self, state_dim: int, n_actions: int, hidden: int = 256) -> None:
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, hidden)
        self.q_head = nn.Linear(hidden, n_actions)

        self._init_weights()

    def _init_weights(self) -> None:
        for layer in [self.fc1, self.fc2, self.fc3]:
            nn.init.orthogonal_(layer.weight, gain=1.0)
            nn.init.constant_(layer.bias, 0.0)
        nn.init.orthogonal_(self.q_head.weight, gain=0.01)
        nn.init.constant_(self.q_head.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning Q-values for all actions.

        Args:
            x: State tensor of shape (batch, state_dim).

        Returns:
            Q-values of shape (batch, n_actions).
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.q_head(x)
