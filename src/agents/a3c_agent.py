"""
Asynchronous Advantage Actor-Critic (A3C) agent.

Each worker thread runs its own copy of the environment and periodically
pushes gradients to the shared global network.  Workers are implemented as
Python threads using the torch multiprocessing-safe shared memory mechanism.

Reference:
    Mnih et al. (2016) – Asynchronous Methods for Deep Reinforcement
        Learning.  https://arxiv.org/abs/1602.01783
"""

import threading
from typing import Callable, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from .networks import BeamformingNetwork


class A3CWorker(threading.Thread):
    """
    A3C worker thread.

    Runs a local copy of the environment for ``t_max`` steps, accumulates
    gradients, then applies them to the shared global network before
    re-syncing the local network weights.

    Args:
        worker_id:   Integer identifier for logging.
        global_net:  Shared :class:`BeamformingNetwork` stored in shared memory.
        optimizer:   Shared optimiser tied to ``global_net`` parameters.
        env_factory: Zero-argument callable that returns a new Gymnasium env.
        state_dim:   State vector dimensionality.
        action_dim:  Action vector dimensionality.
        hidden:      Hidden layer width (must match global_net).
        gamma:       Discount factor.
        t_max:       Steps between gradient synchronisations.
        max_steps:   Total environment steps before the worker stops.
                     If ``None``, run until :meth:`stop` is called.
        device:      Torch device string.
    """

    def __init__(
        self,
        worker_id: int,
        global_net: BeamformingNetwork,
        optimizer: torch.optim.Optimizer,
        env_factory: Callable,
        state_dim: int,
        action_dim: int,
        hidden: int = 256,
        gamma: float = 0.99,
        t_max: int = 20,
        max_steps: Optional[int] = None,
        device: str = "cpu",
    ) -> None:
        super().__init__(name=f"A3CWorker-{worker_id}", daemon=True)
        self.worker_id = worker_id
        self.global_net = global_net
        self.optimizer = optimizer
        self.env_factory = env_factory
        self.gamma = gamma
        self.t_max = t_max
        self.max_steps = max_steps
        self.device = torch.device(device)
        self._stop_event = threading.Event()
        self.episode_rewards: List[float] = []

        self.local_net = BeamformingNetwork(state_dim, action_dim, hidden).to(self.device)

    def stop(self) -> None:
        """Signal the worker to stop after completing its current update."""
        self._stop_event.set()

    def run(self) -> None:
        """Main worker loop: collect trajectories and update global network."""
        env = self.env_factory()
        obs, _ = env.reset()
        total_steps = 0
        episode_reward = 0.0

        while not self._stop_event.is_set():
            if self.max_steps is not None and total_steps >= self.max_steps:
                break

            # Sync local network with global
            self.local_net.load_state_dict(self.global_net.state_dict())

            log_probs: List[torch.Tensor] = []
            values: List[torch.Tensor] = []
            rewards: List[float] = []
            dones: List[bool] = []

            for _ in range(self.t_max):
                state_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                action, log_prob, value = self.local_net.get_action(state_t)
                action_np = action.detach().cpu().numpy().flatten()

                obs, reward, terminated, truncated, _ = env.step(action_np)
                done = terminated or truncated

                log_probs.append(log_prob)
                values.append(value)
                rewards.append(reward)
                dones.append(done)
                episode_reward += reward
                total_steps += 1

                if done:
                    self.episode_rewards.append(episode_reward)
                    episode_reward = 0.0
                    obs, _ = env.reset()
                    break

            # Bootstrap value for the last state
            if not dones[-1]:
                with torch.no_grad():
                    _, bootstrap_value = self.local_net(
                        torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                    )
                R = float(bootstrap_value.item())
            else:
                R = 0.0

            # Compute returns and advantages
            returns: List[float] = []
            for r in reversed(rewards):
                R = r + self.gamma * R
                returns.insert(0, R)

            returns_t = torch.FloatTensor(returns).to(self.device)
            values_t = torch.cat(values).squeeze(-1)
            log_probs_t = torch.cat(log_probs)

            advantage = returns_t - values_t.detach()
            actor_loss = -(log_probs_t * advantage).mean()
            critic_loss = F.mse_loss(values_t, returns_t)
            loss = actor_loss + 0.5 * critic_loss

            # Accumulate gradients into global network
            self.optimizer.zero_grad()
            loss.backward()
            # Copy local gradients to global parameters
            for local_p, global_p in zip(
                self.local_net.parameters(), self.global_net.parameters()
            ):
                if local_p.grad is not None:
                    if global_p.grad is None:
                        global_p.grad = local_p.grad.clone()
                    else:
                        global_p.grad += local_p.grad
            self.optimizer.step()


def run_a3c(
    global_net: BeamformingNetwork,
    env_factory: Callable,
    state_dim: int,
    action_dim: int,
    n_workers: int = 4,
    lr: float = 1e-4,
    gamma: float = 0.99,
    t_max: int = 20,
    max_steps_per_worker: Optional[int] = 10_000,
    hidden: int = 256,
    device: str = "cpu",
) -> List[A3CWorker]:
    """
    Launch ``n_workers`` A3C threads sharing ``global_net``.

    Args:
        global_net:             Pre-allocated global network (call
                                ``.share_memory()`` before passing in).
        env_factory:            Zero-argument callable returning a new env.
        state_dim:              State dimensionality.
        action_dim:             Action dimensionality.
        n_workers:              Number of parallel worker threads.
        lr:                     Shared optimiser learning rate.
        gamma:                  Discount factor.
        t_max:                  Steps per worker update.
        max_steps_per_worker:   Each worker stops after this many env steps.
        hidden:                 Hidden layer width.
        device:                 Torch device.

    Returns:
        List of started :class:`A3CWorker` threads.
    """
    global_net.share_memory()
    optimizer = torch.optim.Adam(global_net.parameters(), lr=lr)

    workers = [
        A3CWorker(
            worker_id=i,
            global_net=global_net,
            optimizer=optimizer,
            env_factory=env_factory,
            state_dim=state_dim,
            action_dim=action_dim,
            hidden=hidden,
            gamma=gamma,
            t_max=t_max,
            max_steps=max_steps_per_worker,
            device=device,
        )
        for i in range(n_workers)
    ]
    for w in workers:
        w.start()
    return workers
