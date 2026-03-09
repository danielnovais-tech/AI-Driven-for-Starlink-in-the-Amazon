"""
Online PPO agent with continuous (lifelong) fine-tuning capability.

Extends :class:`~agents.ppo_agent.PPOAgent` with:
    - An experience replay buffer for asynchronous batch sampling.
    - A periodic ``update_online()`` call that re-uses the PPO clipped
      objective on recently collected transitions.

This is useful for adapting a pre-trained policy to distribution shifts
observed during live operation (e.g. a new rain season in the Amazon).

Usage::

    agent = OnlinePPOAgent(state_dim=7, action_dim=4)
    # In the inference loop:
    agent.store_transition(state, action, reward, next_state, done)
    agent.update_online()   # no-op until buffer has enough samples

Reference:
    Continual/Online RL: Ring (1997); Thrun & Mitchell (1995).
"""

from collections import deque
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from .ppo_agent import PPOAgent


class OnlinePPOAgent(PPOAgent):
    """
    PPO agent augmented with a replay buffer for online fine-tuning.

    All constructor arguments from :class:`PPOAgent` are supported.
    Additional args:

    Args:
        buffer_size:   Maximum number of transitions stored in the replay
                       buffer (FIFO eviction).
        batch_size:    Number of transitions sampled per update.
        update_freq:   Minimum number of :meth:`store_transition` calls
                       between successive gradient updates.
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
        epochs: int = 4,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        device: str = "cpu",
        buffer_size: int = 10_000,
        batch_size: int = 64,
        update_freq: int = 100,
    ) -> None:
        super().__init__(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden=hidden,
            lr=lr,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_eps=clip_eps,
            epochs=epochs,
            value_coef=value_coef,
            entropy_coef=entropy_coef,
            max_grad_norm=max_grad_norm,
            device=device,
        )
        self.buffer: deque = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.update_freq = update_freq
        self._step_counter: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def store_transition(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """
        Push a (s, a, r, s', done) transition into the replay buffer and
        increment the internal step counter.

        Args:
            state:      Normalised state array at time t.
            action:     Action array taken at time t.
            reward:     Scalar reward received.
            next_state: Normalised state array at time t+1.
            done:       Whether the episode ended after this transition.
        """
        self.buffer.append(
            (
                np.array(state, dtype=np.float32),
                np.array(action, dtype=np.float32),
                float(reward),
                np.array(next_state, dtype=np.float32),
                float(done),
            )
        )
        self._step_counter += 1

    def update_online(self) -> Optional[float]:
        """
        Perform one PPO mini-batch update if the buffer is large enough and
        the step counter has reached the next scheduled update.

        Returns:
            Mean loss over PPO epochs (float) or ``None`` if the update was
            skipped (buffer too small or update not yet due).
        """
        if len(self.buffer) < self.batch_size:
            return None
        if self._step_counter % self.update_freq != 0:
            return None

        # Sample a random mini-batch from the buffer
        indices = np.random.choice(len(self.buffer), self.batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        states, actions, rewards, next_states, dones = zip(*batch)

        states_t = torch.FloatTensor(np.array(states)).to(self.device)
        actions_t = torch.FloatTensor(np.array(actions)).to(self.device)
        rewards_t = torch.FloatTensor(np.array(rewards, dtype=np.float32)).to(self.device)
        next_states_t = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones_t = torch.FloatTensor(np.array(dones, dtype=np.float32)).to(self.device)

        # Bootstrap target values using current value head
        with torch.no_grad():
            _, next_values = self.net(next_states_t)
            target_values = (
                rewards_t + self.gamma * (1.0 - dones_t) * next_values.squeeze(-1)
            )

        # Compute old log-probs under the current policy (treated as a fixed reference)
        with torch.no_grad():
            mean_old, _ = self.net(states_t)
            std_old = self.net.log_std.exp()
            dist_old = torch.distributions.Normal(mean_old, std_old)
            old_log_probs = dist_old.log_prob(actions_t).sum(dim=-1)

        # Run the standard PPO clipped update
        return self._ppo_mini_update(states_t, actions_t, old_log_probs, target_values)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ppo_mini_update(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        returns: torch.Tensor,
    ) -> float:
        """
        Run ``self.epochs`` PPO gradient steps on a fixed mini-batch.

        Args:
            states:        (B, state_dim) state tensor.
            actions:       (B, action_dim) action tensor.
            old_log_probs: (B,) log-probability under the old policy.
            returns:       (B,) bootstrap target values.

        Returns:
            Mean total loss across all epochs.
        """
        total_loss = 0.0
        for _ in range(self.epochs):
            mean, values = self.net(states)
            std = self.net.log_std.exp()
            dist = torch.distributions.Normal(mean, std)
            log_probs = dist.log_prob(actions).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1).mean()

            advantages = returns - values.squeeze(-1).detach()
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            ratios = torch.exp(log_probs - old_log_probs)
            surr1 = ratios * advantages
            surr2 = (
                torch.clamp(ratios, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages
            )
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = F.mse_loss(values.squeeze(-1), returns)
            loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
            self.optimizer.step()
            total_loss += loss.item()

        return total_loss / self.epochs
