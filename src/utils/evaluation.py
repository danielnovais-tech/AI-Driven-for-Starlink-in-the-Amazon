"""
Evaluation utilities for comparing DRL agents against a baseline.

Metrics reported:
    - Mean throughput (Mbps)
    - Mean latency (ms)
    - Outage probability
    - Connectivity reliability (fraction of steps with SNR > threshold)

Reference:
    Industry studies on LEO reliability benchmarks (>99.2 % under 50 mm/h).
"""

from typing import Any, Dict, Optional

import numpy as np


def evaluate(
    env,
    agent,
    episodes: int = 100,
    max_steps_per_episode: Optional[int] = None,
    deterministic: bool = True,
) -> Dict[str, Any]:
    """
    Evaluate a trained agent on the given environment for multiple episodes.

    The agent must expose ``get_action(state, deterministic)`` returning either
    a numpy array (continuous) or an integer (discrete).

    Args:
        env:                  A Gymnasium-compatible environment.
        agent:                Trained DRL agent.
        episodes:             Number of evaluation episodes.
        max_steps_per_episode: Maximum steps per episode (``None`` = no limit).
        deterministic:        Whether to use deterministic actions.

    Returns:
        Dictionary with keys:
            ``mean_throughput``  – average throughput over all steps (Mbps).
            ``mean_latency``     – average latency (ms).
            ``outage_prob``      – fraction of steps with an outage event.
            ``reliability``      – 1 − outage_prob.
            ``total_reward``     – mean total reward per episode.
            ``n_episodes``       – number of completed episodes.
    """
    total_throughput = 0.0
    total_latency = 0.0
    total_outage = 0.0
    total_steps = 0
    total_episode_reward = 0.0

    for ep in range(episodes):
        obs, _ = env.reset()
        episode_reward = 0.0
        step = 0
        terminated = truncated = False

        while not (terminated or truncated):
            action = _get_action(agent, obs, deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            step += 1
            total_steps += 1

            total_throughput += float(info.get("throughput", 0.0))
            total_latency += float(info.get("latency", 0.0))
            total_outage += float(info.get("outage", 0.0))

            if max_steps_per_episode is not None and step >= max_steps_per_episode:
                break

        total_episode_reward += episode_reward

    if total_steps == 0:
        total_steps = 1  # avoid division by zero if no steps were taken

    outage_prob = total_outage / total_steps
    return {
        "mean_throughput": total_throughput / total_steps,
        "mean_latency": total_latency / total_steps,
        "outage_prob": outage_prob,
        "reliability": 1.0 - outage_prob,
        "total_reward": total_episode_reward / max(1, episodes),
        "n_episodes": episodes,
    }


def _get_action(agent, obs: np.ndarray, deterministic: bool):
    """
    Helper that calls ``agent.get_action`` and handles both continuous and
    discrete return signatures.

    Args:
        agent:         DRL agent.
        obs:           Current observation array.
        deterministic: Whether to use deterministic inference.

    Returns:
        Action (numpy array or int).
    """
    result = agent.get_action(obs, deterministic=deterministic)
    if isinstance(result, tuple):
        return result[0]
    return result
