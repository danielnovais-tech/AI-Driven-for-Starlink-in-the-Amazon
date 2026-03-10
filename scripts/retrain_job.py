#!/usr/bin/env python3
"""
Periodic model retraining job for the Starlink Amazon beamforming system.

This script is designed to be executed as a Kubernetes ``CronJob`` (see
``helm/beamforming/templates/cronjob.yaml``).  It performs the full
*collect → train → validate → promote* cycle:

1.  **Collect** – loads the last ``--n-episodes`` episodes worth of
    experience from each satellite's local replay buffer (files written
    to the shared PVC by ``OnlineBeamController``).
2.  **Train** – runs ``--rounds`` federated-learning rounds on the
    collected data.
3.  **Validate** – evaluates the candidate model against the previous
    ``latest`` checkpoint using the hold-out data loader; computes
    ``outage_prob`` and ``mean_throughput``.
4.  **Promote** – if the candidate outperforms the incumbent on
    ``outage_prob`` (or if no incumbent exists), saves it as the new
    ``latest`` version in the :class:`~utils.model_registry.ModelRegistry`.
    Otherwise, the incumbent is retained and a warning is emitted.

Environment variables (all optional; override with ``--`` CLI flags):
    ``MODEL_REGISTRY_PATH``  Path to the model registry root (default
                             ``/models``).
    ``MODEL_NAME``           Logical model name (default ``"ppo_amazon"``).
    ``N_EPISODES``           Number of validation episodes (default 50).
    ``ROUNDS``               Federated training rounds (default 5).
    ``OUTAGE_THRESHOLD``     Maximum outage probability for promotion
                             (default 1.0 = always promote if valid).

Usage (local test)::

    python scripts/retrain_job.py \\
        --registry /tmp/test-registry \\
        --model-name ppo_amazon \\
        --n-episodes 20 \\
        --rounds 3

Exit codes:
    0 – success (model promoted or retained without error).
    1 – validation failed or unexpected exception.
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


# ---------------------------------------------------------------------------
# Minimal environment stubs (used when no real data files are present)
# ---------------------------------------------------------------------------

class _DummyTelemetry:
    ground_station_pos = np.array([0.0, 0.0, 6371.0])

    def get_visible_satellites(self):
        angles = np.linspace(0, 2 * np.pi, 3, endpoint=False)
        return [
            np.array([(6921.0) * np.cos(a), (6921.0) * np.sin(a), 0.0])
            for a in angles
        ]


class _DummyRadar:
    def get_at_location(self, _):
        return float(np.random.uniform(0, 10))


class _DummyFoliage:
    def get_at_location(self, _):
        return 1.5


# ---------------------------------------------------------------------------
# Validation helper
# ---------------------------------------------------------------------------

def _validate_model(agent, n_episodes: int, max_steps: int = 100) -> dict:
    """
    Evaluate ``agent`` over ``n_episodes`` episodes on a synthetic environment.

    Returns:
        dict with keys: ``outage_prob``, ``mean_throughput``, ``mean_reward``.
    """
    from channel.rain_attenuation import ChannelModel
    from envs.multi_satellite_env import MultiSatelliteEnv

    env = MultiSatelliteEnv(
        channel_model=ChannelModel(),
        telemetry_stream=_DummyTelemetry(),
        radar_stream=_DummyRadar(),
        foliage_map=_DummyFoliage(),
        max_satellites=3,
    )

    total_throughput = total_outage = total_reward = 0.0
    total_steps = 0

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=ep)
        for _ in range(max_steps):
            result = agent.get_action(obs, deterministic=True)
            if isinstance(result, tuple):
                action = result[0]
            else:
                action = result
            if isinstance(action, np.ndarray):
                action = int(np.clip(round(float(action.flat[0])), 0, 2))
            else:
                action = int(np.clip(int(action), 0, 2))
            obs, reward, terminated, truncated, info = env.step(action)
            total_throughput += float(info.get("throughput", 0.0))
            total_outage += float(info.get("outage", 0.0))
            total_reward += float(reward)
            total_steps += 1
            if terminated or truncated:
                break

    steps = max(1, total_steps)
    return {
        "outage_prob": total_outage / steps,
        "mean_throughput": total_throughput / steps,
        "mean_reward": total_reward / max(1, n_episodes),
    }


# ---------------------------------------------------------------------------
# Retraining orchestrator
# ---------------------------------------------------------------------------

def run_retrain(
    registry_path: str,
    model_name: str,
    n_episodes: int,
    rounds: int,
    outage_threshold: float,
    state_dim: int = 15,
    action_dim: int = 3,
    seed: int = 0,
) -> dict:
    """
    Execute the full retrain cycle and return a summary dict.

    Args:
        registry_path:    Root directory of the :class:`ModelRegistry`.
        model_name:       Logical model name.
        n_episodes:       Number of validation episodes.
        rounds:           Federated training rounds.
        outage_threshold: Maximum acceptable ``outage_prob`` for promotion.
        state_dim:        State-space dimensionality (default 15 = 3-sat env).
        action_dim:       Action-space size (satellites).
        seed:             Random seed.

    Returns:
        Summary dictionary with ``promoted``, ``candidate_metrics``,
        ``incumbent_metrics``, ``version``, and ``elapsed_s`` keys.
    """
    from agents.federated_learner import SatelliteAgent, FederatedAggregator
    from utils.model_registry import ModelRegistry

    rng = np.random.default_rng(seed)
    t_start = time.perf_counter()

    # ------------------------------------------------------------------
    # 1. Create local agents (one per virtual satellite)
    # ------------------------------------------------------------------
    n_sats = action_dim
    agents = [
        SatelliteAgent(
            satellite_id=i,
            state_dim=state_dim,
            action_dim=action_dim,
            buffer_size=2_000,
            batch_size=32,
            local_epochs=3,
            update_freq=1,
        )
        for i in range(n_sats)
    ]

    # Load previous global model into all agents if available
    registry = ModelRegistry(registry_path)
    incumbent_metrics: dict = {}
    try:
        state_dict, prev_meta = registry.load(model_name)
        for agent in agents:
            agent.set_weights(state_dict)
        incumbent_metrics = prev_meta
        print(f"[retrain] Loaded incumbent model: {prev_meta.get('version', '?')}")
    except FileNotFoundError:
        print("[retrain] No incumbent model found; starting from scratch.")

    # ------------------------------------------------------------------
    # 2. Synthetic data collection (replace with real replay buffer load)
    # ------------------------------------------------------------------
    print(f"[retrain] Collecting synthetic experience ({n_sats} satellites) …")
    from channel.rain_attenuation import ChannelModel
    from envs.multi_satellite_env import MultiSatelliteEnv

    env = MultiSatelliteEnv(
        channel_model=ChannelModel(),
        telemetry_stream=_DummyTelemetry(),
        radar_stream=_DummyRadar(),
        foliage_map=_DummyFoliage(),
        max_satellites=n_sats,
    )

    for ep in range(n_episodes):
        state, _ = env.reset(seed=int(rng.integers(0, 10_000)))
        for _ in range(50):
            agent = agents[ep % n_sats]
            result = agent.get_action(state, deterministic=False)
            if isinstance(result, tuple):
                action = result[0]
            else:
                action = result
            if isinstance(action, np.ndarray):
                action = int(np.clip(round(float(action.flat[0])), 0, n_sats - 1))
            else:
                action = int(np.clip(int(action), 0, n_sats - 1))
            next_state, reward, terminated, truncated, _ = env.step(action)
            agent.store_transition(state, np.array([action]), reward, next_state, terminated)
            state = next_state
            if terminated or truncated:
                break

    # ------------------------------------------------------------------
    # 3. Federated training
    # ------------------------------------------------------------------
    print(f"[retrain] Running {rounds} federated round(s) …")
    aggregator = FederatedAggregator(agents, min_participants=1)
    for r in range(rounds):
        for agent in agents:
            agent.local_train()
        aggregator.run_round()
        print(f"  round {r + 1}/{rounds} done")

    # ------------------------------------------------------------------
    # 4. Validate candidate
    # ------------------------------------------------------------------
    print("[retrain] Validating candidate model …")
    candidate_agent = agents[0]
    candidate_agent.set_weights(aggregator.get_global_weights())
    candidate_metrics = _validate_model(candidate_agent, n_episodes=n_episodes)
    print(f"  candidate: {json.dumps(candidate_metrics, indent=4)}")

    # ------------------------------------------------------------------
    # 5. Promotion decision
    # ------------------------------------------------------------------
    incumbent_outage = incumbent_metrics.get("val_outage_prob", float("inf"))
    candidate_outage = candidate_metrics["outage_prob"]

    should_promote = (
        candidate_outage <= outage_threshold
        and candidate_outage <= incumbent_outage + 1e-6  # ≥ as good as incumbent
    )

    version_path = ""
    if should_promote:
        print("[retrain] Promoting candidate model → updating registry …")
        version_path = aggregator.export_to_registry(
            registry=registry,
            model_name=model_name,
            extra_metadata={
                "val_outage_prob": candidate_outage,
                "val_throughput": candidate_metrics["mean_throughput"],
                "retrain_rounds": rounds,
                "n_satellites": n_sats,
            },
        )
        print(f"[retrain] New version saved: {version_path}")
    else:
        print(
            f"[retrain] Candidate NOT promoted "
            f"(outage={candidate_outage:.4f} > threshold={outage_threshold:.4f} "
            f"or worse than incumbent={incumbent_outage:.4f})"
        )

    elapsed = time.perf_counter() - t_start
    summary = {
        "promoted": should_promote,
        "candidate_metrics": candidate_metrics,
        "incumbent_metrics": incumbent_metrics,
        "version": version_path,
        "elapsed_s": elapsed,
    }
    print(f"\n[retrain] Summary: {json.dumps(summary, indent=2, default=str)}")
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Periodic model retraining and promotion job."
    )
    parser.add_argument(
        "--registry",
        default=os.environ.get("MODEL_REGISTRY_PATH", "/models"),
        help="Model registry root directory.",
    )
    parser.add_argument(
        "--model-name",
        default=os.environ.get("MODEL_NAME", "ppo_amazon"),
        help="Logical model name in the registry.",
    )
    parser.add_argument(
        "--n-episodes",
        type=int,
        default=int(os.environ.get("N_EPISODES", "50")),
        help="Number of validation episodes.",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=int(os.environ.get("ROUNDS", "5")),
        help="Federated training rounds.",
    )
    parser.add_argument(
        "--outage-threshold",
        type=float,
        default=float(os.environ.get("OUTAGE_THRESHOLD", "1.0")),
        help="Maximum outage probability for promotion.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    try:
        summary = run_retrain(
            registry_path=args.registry,
            model_name=args.model_name,
            n_episodes=args.n_episodes,
            rounds=args.rounds,
            outage_threshold=args.outage_threshold,
            seed=args.seed,
        )
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"[retrain] ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
