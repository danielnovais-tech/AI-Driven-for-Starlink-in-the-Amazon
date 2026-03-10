"""
Proximal Policy Optimisation (PPO) agent for continuous beamforming control.

Implements the clipped surrogate objective with Generalized Advantage
Estimation (GAE-λ) and value-function loss.

Reference:
    Schulman et al. (2017) – Proximal Policy Optimization Algorithms.
    https://arxiv.org/abs/1707.06347
"""

from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Normal

from .networks import BeamformingNetwork


class PPOAgent:
    """
    PPO agent with clipped surrogate objective and GAE.

    Args:
        state_dim:   Dimensionality of the state vector.
        action_dim:  Dimensionality of the continuous action vector.
        hidden:      Hidden layer width.
        lr:          Learning rate for Adam.
        gamma:       Discount factor.
        gae_lambda:  GAE smoothing parameter λ.
        clip_eps:    PPO clip parameter ε.
        epochs:      Number of optimisation epochs per policy update.
        value_coef:  Weight for the value-function loss term.
        entropy_coef: Weight for the entropy bonus.
        max_grad_norm: Maximum gradient norm for clipping.
        device:      Torch device string.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden: int = 256,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        epochs: int = 10,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        device: str = "cpu",
    ) -> None:
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.epochs = epochs
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.device = torch.device(device)

        self.net = BeamformingNetwork(state_dim, action_dim, hidden).to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_action(
        self, state: np.ndarray, deterministic: bool = False
    ) -> Tuple[np.ndarray, float]:
        """
        Sample or compute a deterministic action from the current policy.

        Args:
            state:         1-D normalised state array.
            deterministic: If True, return the policy mean.

        Returns:
            action:   Numpy action array.
            log_prob: Log-probability of the chosen action.
        """
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            mean, _ = self.net(state_t)
            std = self.net.log_std.exp()
            dist = Normal(mean, std)
            action = mean if deterministic else dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)
        return action.cpu().numpy().flatten(), float(log_prob.item())

    def update(self, rollout: Tuple) -> float:
        """
        Perform multiple epochs of PPO gradient updates on a rollout buffer.

        Args:
            rollout: Tuple of (states, actions, old_log_probs, returns,
                     advantages) where each element is a torch.Tensor
                     already on the correct device.

        Returns:
            Mean total loss over all epochs (float).
        """
        states, actions, old_log_probs, returns, advantages = rollout
        # Normalise advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_loss = 0.0
        for _ in range(self.epochs):
            mean, values = self.net(states)
            std = self.net.log_std.exp()
            dist = Normal(mean, std)
            log_probs = dist.log_prob(actions).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1).mean()

            ratios = torch.exp(log_probs - old_log_probs)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = F.mse_loss(values.squeeze(-1), returns)
            loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
            self.optimizer.step()
            total_loss += loss.item()

        return total_loss / self.epochs

    def compute_gae(
        self,
        rewards: List[float],
        values: List[float],
        dones: List[bool],
        last_value: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute GAE advantages and discounted returns.

        Args:
            rewards:    List of scalar rewards collected in the rollout.
            values:     List of state-value estimates V(s_t).
            dones:      List of episode-termination flags.
            last_value: Bootstrap value V(s_T) for the final state.

        Returns:
            returns:    Tensor of shape (T,).
            advantages: Tensor of shape (T,).
        """
        T = len(rewards)
        returns = np.zeros(T, dtype=np.float32)
        advantages = np.zeros(T, dtype=np.float32)
        gae = 0.0
        next_value = last_value
        for t in reversed(range(T)):
            mask = 1.0 - float(dones[t])
            delta = rewards[t] + self.gamma * next_value * mask - values[t]
            gae = delta + self.gamma * self.gae_lambda * mask * gae
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]
            next_value = values[t]
        return (
            torch.FloatTensor(returns).to(self.device),
            torch.FloatTensor(advantages).to(self.device),
        )
