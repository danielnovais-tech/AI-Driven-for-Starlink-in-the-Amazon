"""
Deep Q-Network (DQN) agent for discrete beamforming control.

The agent supports:
    - ε-greedy exploration with linear decay.
    - Experience replay buffer.
    - Periodic target-network soft or hard updates.

Reference:
    Mnih et al. (2015) – Human-level control through deep reinforcement
        learning.
"""

import random
from collections import deque
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .networks import DQNNetwork


class DQNAgent:
    """
    DQN agent with experience replay and target network.

    Args:
        state_dim:        Dimensionality of the state vector.
        n_actions:        Number of discrete actions.
        hidden:           Hidden layer width.
        lr:               Learning rate for the Adam optimiser.
        gamma:            Discount factor.
        buffer_size:      Maximum replay buffer capacity.
        batch_size:       Mini-batch size for each gradient update.
        target_update_freq: Number of gradient updates between target-network
                            copies (hard update).
        epsilon_start:    Initial ε for ε-greedy exploration.
        epsilon_end:      Minimum ε after decay.
        epsilon_decay:    Multiplicative decay factor applied per step.
        device:           Torch device string ('cpu' or 'cuda').
    """

    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        hidden: int = 256,
        lr: float = 1e-3,
        gamma: float = 0.99,
        buffer_size: int = 10_000,
        batch_size: int = 64,
        target_update_freq: int = 100,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.995,
        device: str = "cpu",
    ) -> None:
        self.n_actions = n_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.device = torch.device(device)
        self._update_count = 0

        self.q_net = DQNNetwork(state_dim, n_actions, hidden).to(self.device)
        self.target_net = DQNNetwork(state_dim, n_actions, hidden).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        self.replay_buffer: deque = deque(maxlen=buffer_size)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def select_action(self, state: np.ndarray) -> int:
        """
        Select an action using ε-greedy policy.

        Args:
            state: 1-D numpy state array.

        Returns:
            Integer action index.
        """
        if random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_net(s)
        return int(q_values.argmax(dim=1).item())

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Push a (s, a, r, s', done) transition into the replay buffer."""
        self.replay_buffer.append((state, action, reward, next_state, float(done)))

    def update(self) -> Optional[float]:
        """
        Sample a mini-batch and perform one gradient descent step.

        Returns:
            TD loss value (float) or None if the buffer is too small.
        """
        if len(self.replay_buffer) < self.batch_size:
            return None

        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states_t = torch.FloatTensor(np.array(states)).to(self.device)
        actions_t = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards_t = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states_t = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones_t = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # Current Q-values
        q_values = self.q_net(states_t).gather(1, actions_t)

        # Target Q-values (Bellman backup, no gradient)
        with torch.no_grad():
            next_q = self.target_net(next_states_t).max(dim=1, keepdim=True)[0]
            target_q = rewards_t + self.gamma * (1.0 - dones_t) * next_q

        loss = F.mse_loss(q_values, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        self._update_count += 1
        if self._update_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        # Decay exploration rate
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        return float(loss.item())

    def get_action(self, state: np.ndarray, deterministic: bool = False) -> int:
        """
        Alias for :meth:`select_action` with optional deterministic flag.

        When ``deterministic=True`` ε is effectively set to 0.

        Args:
            state:         1-D numpy state array.
            deterministic: If True, always pick the greedy action.

        Returns:
            Integer action index.
        """
        if deterministic:
            with torch.no_grad():
                s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.q_net(s)
            return int(q_values.argmax(dim=1).item())
        return self.select_action(state)
