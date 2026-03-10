#!/usr/bin/env python3
"""
Validate the DRL agent on extreme Amazon weather scenarios.

Generates synthetic rain profiles (convective cells, persistent storms) and
evaluates a PPO agent against a null-action baseline, reporting outage
probability and mean throughput for each scenario.

Usage:
    python scripts/extreme_scenarios.py

The script does not require a trained agent checkpoint; it instantiates a
freshly-initialised PPOAgent so that the infrastructure (environment,
evaluation loop, result display) can be validated independently of training.

Scenarios:
    1. Convective cell (bell-shaped rain event, peak 100 mm/h)
    2. Persistent moderate rain (50 mm/h, 600 steps)
    3. Layered rain + foliage (50 mm/h + LAI = 6)
"""

import sys
import os
import argparse

import numpy as np

# ---------------------------------------------------------------------------
# Make src importable when running from the repository root
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from channel.rain_attenuation import ChannelModel
from envs.leo_beamforming_env import LEOBeamformingEnv
from agents.ppo_agent import PPOAgent


# ---------------------------------------------------------------------------
# Synthetic scenario generators
# ---------------------------------------------------------------------------

def convective_cell_rain(duration: int = 600, peak: float = 100.0, dt: float = 1.0) -> np.ndarray:
    """
    Bell-shaped convective-cell rain profile.

    Args:
        duration: Total duration in seconds.
        peak:     Peak rain rate (mm/h).
        dt:       Time step (s).

    Returns:
        1-D array of rain rates (mm/h) with length ``duration / dt``.
    """
    t = np.arange(0, duration, dt)
    sigma = duration / 6.0
    return peak * np.exp(-((t - duration / 2.0) ** 2) / (2.0 * sigma ** 2))


def persistent_rain(duration: int = 600, rate: float = 50.0) -> np.ndarray:
    """
    Constant rain rate profile.

    Args:
        duration: Total duration in steps.
        rate:     Rain rate (mm/h).

    Returns:
        1-D array of shape (duration,).
    """
    return np.full(duration, rate, dtype=np.float64)


# ---------------------------------------------------------------------------
# Stub stream / map objects for the environment
# ---------------------------------------------------------------------------

class _StaticTelemetry:
    """Telemetry stub returning a fixed satellite position."""

    _POS = np.array([0.0, 0.0, 550.0])

    def get_current_position(self):
        return self._POS.copy()

    def get_next_position(self):
        return self._POS.copy()

    def get_current_snr(self):
        return 15.0

    def get_current_rssi(self):
        return -75.0


class _InjectedRadar:
    """
    Radar stub with an injectable current rain rate.

    Set ``self.current_rate`` before each environment step to simulate
    varying rainfall.
    """

    def __init__(self):
        self.current_rate = 0.0

    def get_at_location(self, _pos):
        return self.current_rate


class _StaticFoliage:
    """Foliage stub returning a configurable constant LAI."""

    def __init__(self, lai: float = 2.0):
        self.lai = lai

    def get_at_location(self, _pos):
        return self.lai


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def _make_env(radar_stub: _InjectedRadar, foliage_stub: _StaticFoliage) -> LEOBeamformingEnv:
    channel = ChannelModel()
    telemetry = _StaticTelemetry()
    return LEOBeamformingEnv(channel, telemetry, radar_stub, foliage_stub)


def evaluate_on_rain(
    env: LEOBeamformingEnv,
    agent,
    rain_profile: np.ndarray,
    radar_stub: _InjectedRadar,
) -> dict:
    """
    Step through the environment using ``rain_profile`` to control the radar.

    Args:
        env:         Gymnasium environment.
        agent:       DRL agent with ``get_action(state, deterministic)`` method.
        rain_profile: Rain rate (mm/h) at each step.
        radar_stub:  The radar stub whose ``current_rate`` will be updated.

    Returns:
        Dictionary with keys ``outages``, ``throughputs``, ``mean_outage``,
        ``mean_throughput``.
    """
    state, _ = env.reset()
    outages = []
    throughputs = []
    for rain_rate in rain_profile:
        radar_stub.current_rate = float(rain_rate)
        result = agent.get_action(state, deterministic=True)
        action = result[0] if isinstance(result, tuple) else result
        state, _reward, terminated, truncated, info = env.step(action)
        outages.append(float(info["outage"]))
        throughputs.append(float(info["throughput"]))
        if terminated or truncated:
            break
    return {
        "outages": outages,
        "throughputs": throughputs,
        "mean_outage": float(np.mean(outages)) if outages else float("nan"),
        "mean_throughput": float(np.mean(throughputs)) if throughputs else float("nan"),
    }


def baseline_policy(
    env: LEOBeamformingEnv,
    rain_profile: np.ndarray,
    radar_stub: _InjectedRadar,
) -> dict:
    """
    Null-action baseline: fixed MCS=2, 50 resource blocks, no beam adjustment.

    Args:
        env:         Gymnasium environment.
        rain_profile: Rain rate profile (mm/h).
        radar_stub:  The radar stub.

    Returns:
        Same dictionary structure as :func:`evaluate_on_rain`.
    """
    state, _ = env.reset()
    outages = []
    throughputs = []
    null_action = np.array([0.0, 0.5, 2.0, 50.0], dtype=np.float32)
    for rain_rate in rain_profile:
        radar_stub.current_rate = float(rain_rate)
        state, _reward, terminated, truncated, info = env.step(null_action)
        outages.append(float(info["outage"]))
        throughputs.append(float(info["throughput"]))
        if terminated or truncated:
            break
    return {
        "outages": outages,
        "throughputs": throughputs,
        "mean_outage": float(np.mean(outages)) if outages else float("nan"),
        "mean_throughput": float(np.mean(throughputs)) if throughputs else float("nan"),
    }


# ---------------------------------------------------------------------------
# Scenario runner
# ---------------------------------------------------------------------------

def run_scenario(
    name: str,
    rain_profile: np.ndarray,
    lai: float = 2.0,
    state_dim: int = 7,
    action_dim: int = 4,
) -> None:
    """
    Run a single extreme scenario and print a results summary.

    Args:
        name:         Human-readable scenario name.
        rain_profile: Rain rate (mm/h) array.
        lai:          Leaf area index for the foliage stub.
        state_dim:    Agent state dimension.
        action_dim:   Agent action dimension.
    """
    radar_stub = _InjectedRadar()
    foliage_stub = _StaticFoliage(lai)
    env = _make_env(radar_stub, foliage_stub)

    # Use a freshly initialised (untrained) agent as a stand-in
    agent = PPOAgent(state_dim=state_dim, action_dim=action_dim)

    drl = evaluate_on_rain(env, agent, rain_profile, radar_stub)
    base = baseline_policy(env, rain_profile, radar_stub)

    outage_reduction = (base["mean_outage"] - drl["mean_outage"]) / max(base["mean_outage"], 1e-9) * 100.0

    print(f"\n{'=' * 60}")
    print(f"Scenario: {name}")
    print(f"{'=' * 60}")
    print(f"  Rain steps      : {len(rain_profile)}")
    print(f"  Peak rain       : {rain_profile.max():.1f} mm/h")
    print(f"  LAI (foliage)   : {lai}")
    print(f"  {'Metric':<25} {'DRL':>10} {'Baseline':>10}")
    print(f"  {'-' * 47}")
    print(f"  {'Outage prob':<25} {drl['mean_outage']:>9.3f} {base['mean_outage']:>10.3f}")
    print(
        f"  {'Mean throughput (Mbps)':<25} "
        f"{drl['mean_throughput']:>9.1f} {base['mean_throughput']:>10.1f}"
    )
    print(f"  {'Outage reduction':<25} {outage_reduction:>9.1f}%")


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Validate agent on extreme Amazon weather scenarios.")
    parser.add_argument("--duration", type=int, default=600, help="Scenario duration (steps).")
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    duration = args.duration

    print("Extreme-scenario validation")
    print(f"Duration: {duration} steps per scenario\n")

    run_scenario(
        name="Convective cell (peak 100 mm/h)",
        rain_profile=convective_cell_rain(duration=duration, peak=100.0),
        lai=2.0,
    )
    run_scenario(
        name="Persistent moderate rain (50 mm/h)",
        rain_profile=persistent_rain(duration=duration, rate=50.0),
        lai=2.0,
    )
    run_scenario(
        name="Layered rain + dense foliage (50 mm/h, LAI=6)",
        rain_profile=persistent_rain(duration=duration, rate=50.0),
        lai=6.0,
    )


if __name__ == "__main__":
    main()
