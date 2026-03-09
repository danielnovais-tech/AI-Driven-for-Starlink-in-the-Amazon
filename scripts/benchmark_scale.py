#!/usr/bin/env python3
"""
Large-scale benchmark comparing DRL vs. classical baseline policies.

Policies compared:
    1. ``drl``          – trained :class:`~agents.online_ppo.OnlinePPOAgent`
                          (random weights as stand-in for a pre-trained policy).
    2. ``max_snr``      – always selects the satellite with the highest current SNR.
    3. ``max_elevation``– always selects the satellite with the highest elevation.
    4. ``round_robin``  – cycles through satellites sequentially.
    5. ``random``       – chooses uniformly at random.

Metrics collected (per episode, then aggregated):
    - Mean throughput (Mbps)
    - Mean latency (ms)
    - Outage probability
    - Number of handovers
    - Total reward

Usage::

    python scripts/benchmark_scale.py \\
        --n-episodes 500 \\
        --max-steps 200 \\
        --max-satellites 5 \\
        --seed 42

Output:
    Prints a formatted Markdown table to stdout and optionally saves a JSON
    report to the path specified by ``--output-json``.
"""

import argparse
import json
import os
import sys
import time
from typing import Any, Callable, Dict, List

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from channel.rain_attenuation import ChannelModel
from envs.multi_satellite_env import MultiSatelliteEnv


# ---------------------------------------------------------------------------
# Stub sensor classes
# ---------------------------------------------------------------------------

class _CyclicTelemetry:
    """
    Minimal telemetry provider that cycles through a fixed set of satellite
    positions to simulate orbital motion.
    """

    ground_station_pos = np.array([0.0, 0.0, 6371.0])

    def __init__(self, n_satellites: int, rng: np.random.Generator) -> None:
        # Distribute satellites uniformly in azimuth at 550 km altitude
        angles = np.linspace(0, 2 * np.pi, n_satellites, endpoint=False)
        offset = float(rng.uniform(0, 50.0))
        self._positions = [
            np.array([
                (6371.0 + 550.0) * np.cos(a + offset * 0.001),
                (6371.0 + 550.0) * np.sin(a + offset * 0.001),
                float(rng.uniform(-300.0, 300.0)),
            ])
            for a in angles
        ]
        self._step = 0

    def get_visible_satellites(self) -> List[np.ndarray]:
        # Slowly rotate satellite positions to simulate orbital motion
        angle = self._step * 0.002
        self._step += 1
        return [
            np.array([
                p[0] * np.cos(angle) - p[1] * np.sin(angle),
                p[0] * np.sin(angle) + p[1] * np.cos(angle),
                p[2],
            ])
            for p in self._positions
        ]


class _ConstantRadar:
    def __init__(self, rain_mm_h: float = 5.0) -> None:
        self._rain = rain_mm_h

    def get_at_location(self, _pos) -> float:
        return self._rain


class _ConstantFoliage:
    def get_at_location(self, _pos) -> float:
        return 1.5


# ---------------------------------------------------------------------------
# Baseline policies
# ---------------------------------------------------------------------------

class _MaxSNRPolicy:
    """Pick the satellite with the highest current SNR from the raw observation."""

    def __init__(self, max_satellites: int) -> None:
        self.max_satellites = max_satellites

    def get_action(self, obs: np.ndarray, env: MultiSatelliteEnv) -> int:
        """
        Select satellite with highest SNR by querying the environment's
        internal visible_sats and channel model.
        """
        best_idx = 0
        best_snr = -float("inf")
        for i, sat in enumerate(env.visible_sats[: env.max_satellites]):
            rain = float(env.radar.get_at_location(sat))
            foliage = float(env.foliage.get_at_location(sat))
            snr = env.channel.compute_snr(sat, rain, foliage)
            if snr > best_snr:
                best_snr = snr
                best_idx = i
        return best_idx


class _MaxElevationPolicy:
    """Pick the satellite with the highest elevation angle."""

    def get_action(self, obs: np.ndarray, env: MultiSatelliteEnv) -> int:
        best_idx = 0
        best_elev = -float("inf")
        for i, sat in enumerate(env.visible_sats[: env.max_satellites]):
            elev = env._elevation(sat)
            if elev > best_elev:
                best_elev = elev
                best_idx = i
        return best_idx


class _RoundRobinPolicy:
    def __init__(self) -> None:
        self._counter = 0

    def get_action(self, obs: np.ndarray, env: MultiSatelliteEnv) -> int:
        idx = self._counter % env.max_satellites
        self._counter += 1
        return idx


class _RandomPolicy:
    def __init__(self, seed: int = 0) -> None:
        self._rng = np.random.default_rng(seed)

    def get_action(self, obs: np.ndarray, env: MultiSatelliteEnv) -> int:
        return int(self._rng.integers(0, env.max_satellites))


class _DRLPolicy:
    """
    Wraps an :class:`~agents.online_ppo.OnlinePPOAgent`-compatible agent
    with the same interface as the baseline policies.
    """

    def __init__(self, agent) -> None:
        self._agent = agent

    def get_action(self, obs: np.ndarray, env: MultiSatelliteEnv) -> int:
        result = self._agent.get_action(obs, deterministic=True)
        # Discrete action: map to valid satellite index
        if isinstance(result, tuple):
            action = result[0]
        else:
            action = result
        if isinstance(action, np.ndarray):
            action = int(np.clip(round(float(action[0])), 0, env.max_satellites - 1))
        return int(action) % env.max_satellites


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def _run_episodes(
    env: MultiSatelliteEnv,
    policy,
    n_episodes: int,
    max_steps: int,
    seed_offset: int = 0,
) -> Dict[str, Any]:
    """
    Run ``n_episodes`` evaluation episodes with ``policy`` and collect metrics.

    Returns:
        Dictionary with aggregated statistics:
            - ``mean_throughput``
            - ``mean_latency``
            - ``outage_prob``
            - ``mean_handovers_per_episode``
            - ``mean_reward``
            - ``wall_time_s``
    """
    total_throughput = 0.0
    total_latency = 0.0
    total_outage = 0.0
    total_steps = 0
    total_handovers = 0
    total_reward = 0.0

    t_start = time.perf_counter()

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed_offset + ep)
        ep_reward = 0.0
        ep_handovers = 0

        for _ in range(max_steps):
            action = policy.get_action(obs, env)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            total_throughput += float(info.get("throughput", 0.0))
            total_latency += float(info.get("latency", 0.0))
            total_outage += float(info.get("outage", 0.0))
            total_steps += 1
            if info.get("handover", False):
                ep_handovers += 1
            if terminated or truncated:
                break

        total_handovers += ep_handovers
        total_reward += ep_reward

    wall_time = time.perf_counter() - t_start
    steps = max(1, total_steps)

    return {
        "mean_throughput": total_throughput / steps,
        "mean_latency": total_latency / steps,
        "outage_prob": total_outage / steps,
        "mean_handovers_per_episode": total_handovers / max(1, n_episodes),
        "mean_reward": total_reward / max(1, n_episodes),
        "wall_time_s": wall_time,
    }


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

def _fmt_table(results: Dict[str, Dict[str, Any]]) -> str:
    """Render results as a Markdown table."""
    policies = list(results.keys())
    metrics = [
        ("mean_throughput", "Throughput (Mbps)"),
        ("mean_latency", "Latency (ms)"),
        ("outage_prob", "Outage Prob."),
        ("mean_handovers_per_episode", "HO/Episode"),
        ("mean_reward", "Mean Reward"),
        ("wall_time_s", "Wall Time (s)"),
    ]

    col_w = max(max(len(p) for p in policies), 16)
    header = "| " + "Policy".ljust(col_w) + " |"
    separator = "|-" + "-" * col_w + "-|"
    for _, label in metrics:
        header += f" {label:>18} |"
        separator += f"-{'-'*18}-|"

    rows = [header, separator]
    for policy in policies:
        row = "| " + policy.ljust(col_w) + " |"
        for key, _ in metrics:
            val = results[policy].get(key, float("nan"))
            if key == "outage_prob":
                row += f" {val:>17.4f} |"
            elif key in ("mean_handovers_per_episode",):
                row += f" {val:>18.1f} |"
            elif key == "wall_time_s":
                row += f" {val:>17.2f} |"
            else:
                row += f" {val:>18.2f} |"
        rows.append(row)

    return "\n".join(rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Benchmark DRL vs. baseline beamforming policies at scale."
    )
    parser.add_argument("--n-episodes", type=int, default=100,
                        help="Number of evaluation episodes per policy.")
    parser.add_argument("--max-steps", type=int, default=200,
                        help="Maximum steps per episode.")
    parser.add_argument("--max-satellites", type=int, default=5,
                        help="Number of visible satellites in the environment.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Master random seed.")
    parser.add_argument("--output-json", default=None,
                        help="Optional path to save benchmark results as JSON.")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    rng = np.random.default_rng(args.seed)

    channel = ChannelModel()
    telemetry = _CyclicTelemetry(args.max_satellites, rng)
    radar = _ConstantRadar()
    foliage = _ConstantFoliage()

    def _make_env():
        return MultiSatelliteEnv(
            channel_model=channel,
            telemetry_stream=telemetry,
            radar_stream=radar,
            foliage_map=foliage,
            max_satellites=args.max_satellites,
        )

    # -----------
    # DRL policy (random weights – substitute with a real trained agent)
    # -----------
    try:
        from agents.online_ppo import OnlinePPOAgent
        # obs_dim = max_satellites * 5 (4 features + 1 one-hot slot)
        obs_dim = args.max_satellites * 4 + args.max_satellites
        drl_agent = OnlinePPOAgent(state_dim=obs_dim, action_dim=args.max_satellites)
        drl_policy = _DRLPolicy(drl_agent)
        has_drl = True
    except (ImportError, RuntimeError) as exc:
        print(f"[WARN] DRL policy not available: {exc}")
        has_drl = False

    policies = {}
    if has_drl:
        policies["drl"] = drl_policy
    policies["max_snr"] = _MaxSNRPolicy(args.max_satellites)
    policies["max_elevation"] = _MaxElevationPolicy()
    policies["round_robin"] = _RoundRobinPolicy()
    policies["random"] = _RandomPolicy(seed=args.seed)

    print(f"\nBenchmark: {args.n_episodes} episodes × {args.max_steps} steps "
          f"| {args.max_satellites} satellites | seed={args.seed}\n")

    results = {}
    for name, policy in policies.items():
        env = _make_env()
        metrics = _run_episodes(
            env, policy,
            n_episodes=args.n_episodes,
            max_steps=args.max_steps,
            seed_offset=args.seed,
        )
        results[name] = metrics
        print(f"  [{name}] done  ({metrics['wall_time_s']:.1f}s)")

    print("\n" + _fmt_table(results))

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output_json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
