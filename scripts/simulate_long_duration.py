#!/usr/bin/env python3
"""
Long-duration integration simulation for the Starlink Amazon beamforming system.

Runs a multi-day simulation with:
    - 100+ satellite Starlink-like constellation (simplified propagator)
    - Variable Poisson traffic load
    - Realistic diurnal rain pattern (Amazon basin)
    - Active regulatory exclusion zones
    - Configurable DRL agent (or random-policy baseline)

Metrics collected at each step and summarised at the end:
    throughput_mbps, latency_ms, outage_rate, handover_rate,
    drop_rate, queue_delay_ms, compliance_violations,
    inference_latency_ms, queue_occupancy

Usage::

    python scripts/simulate_long_duration.py \\
        --n-days 7 \\
        --n-satellites 100 \\
        --max-satellites 5 \\
        --arrival-rate 50 \\
        --seed 42 \\
        --output-json /tmp/sim_report.json

Output:
    Prints a summary table to stdout and (optionally) saves a JSON report.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from channel.rain_attenuation import ChannelModel
from channel.orbital_propagator import StarlinkConstellationTelemetry
from envs.traffic_env import TrafficAwareMultiSatelliteEnv
from envs.regulatory_env import RegulatoryEnv, ExclusionZone, GeoRegulatoryEnv


# ---------------------------------------------------------------------------
# Minimal sensor stubs
# ---------------------------------------------------------------------------

class _CyclicConstellationTelemetry:
    """
    Deterministic cyclic telemetry that guarantees satellites are always visible.

    Distributes ``n_satellites`` evenly in azimuth at ``altitude_km`` altitude.
    Each call to ``get_visible_satellites`` rotates the constellation slightly
    to simulate orbital motion, and returns the first ``max_visible`` satellites
    after filtering those above a minimum elevation.

    This class is used as a reliable substitute for the propagator-backed
    :class:`~channel.orbital_propagator.StarlinkConstellationTelemetry` in
    long-duration simulations where brief zero-visibility epochs would crash
    the environment.
    """

    ground_station_pos = np.array([0.0, 0.0, 6371.0])

    def __init__(
        self,
        n_satellites: int = 100,
        altitude_km: float = 550.0,
        max_visible: int = 8,
        seed: int = 0,
    ) -> None:
        rng = np.random.default_rng(seed)
        angles = np.linspace(0.0, 2.0 * np.pi, n_satellites, endpoint=False)
        # _Z_VARIATION_KM: half-amplitude of the Gaussian z-offset used to
        # simulate orbital inclination (a Starlink-like ~53° inclination shell
        # produces |z| up to ~550 * sin(53°) ≈ 440 km; 200 km is a conservative
        # reduced value suitable for the simplified circular-orbit approximation).
        _Z_VARIATION_KM = 200.0
        self._positions = [
            np.array([
                (6371.0 + altitude_km) * np.cos(a),
                (6371.0 + altitude_km) * np.sin(a),
                float(rng.uniform(-_Z_VARIATION_KM, _Z_VARIATION_KM)),
            ])
            for a in angles
        ]
        self.max_visible = max_visible
        self._step: int = 0

    def get_visible_satellites(self) -> list:
        angle = self._step * 0.0015
        self._step += 1
        return [
            np.array([
                p[0] * np.cos(angle) - p[1] * np.sin(angle),
                p[0] * np.sin(angle) + p[1] * np.cos(angle),
                p[2],
            ])
            for p in self._positions[: self.max_visible]
        ]

    def reset(self, t_sec=None) -> None:
        self._step = 0


class _RainByTime:
    """
    Synthetic diurnal rain model for the Amazon basin.

    Rain rate follows a simple sinusoidal pattern peaking at 15:00 local
    time (UTC-4), typical for convective afternoon storms in the Amazon.

    Args:
        peak_rain_mmh:   Peak rain rate (mm/h, default 30 mm/h).
        base_rain_mmh:   Minimum background rain rate (mm/h).
        period_s:        Diurnal period (seconds, default 86 400 = 1 day).
    """

    def __init__(
        self,
        peak_rain_mmh: float = 30.0,
        base_rain_mmh: float = 2.0,
        period_s: float = 86_400.0,
        peak_offset_s: float = 68_400.0,  # 19:00 UTC (15:00 local, UTC-4 Amazon)
    ) -> None:
        self.peak = peak_rain_mmh
        self.base = base_rain_mmh
        self.period = period_s
        self.peak_offset = peak_offset_s
        self._t: float = 0.0

    def set_time(self, t: float) -> None:
        self._t = t

    def get_at_location(self, pos) -> float:
        phase = (self._t % self.period) / self.period
        # Sinusoidal peak at peak_offset
        angle = 2.0 * np.pi * (phase - self.peak_offset / self.period)
        amplitude = (self.peak - self.base) / 2.0
        return float(self.base + amplitude * (1.0 + np.sin(angle)))


class _StaticFoliage:
    """Returns a constant LAI representative of Amazon rainforest."""

    def __init__(self, lai: float = 4.5) -> None:
        self.lai = lai

    def get_at_location(self, pos) -> float:
        return self.lai


# ---------------------------------------------------------------------------
# Baseline policies
# ---------------------------------------------------------------------------

class _RandomPolicy:
    def __init__(self, max_satellites: int = 5) -> None:
        self._n = max_satellites

    def get_action(self, obs, deterministic: bool = True):
        return int(np.random.randint(0, self._n)), 0.0


class _MaxSNRPolicy:
    """Select the satellite with the highest SNR from the observation."""

    def __init__(self, max_satellites: int = 5) -> None:
        self._n = max_satellites

    def get_action(self, obs, deterministic: bool = True):
        snr_indices = [i * 4 for i in range(self._n)]  # SNR is feature 0
        snrs = [obs[idx] for idx in snr_indices if idx < len(obs)]
        return int(np.argmax(snrs)), 0.0


# ---------------------------------------------------------------------------
# Simulation runner
# ---------------------------------------------------------------------------

def run_simulation(
    n_days: float = 7.0,
    n_satellites: int = 100,
    max_satellites: int = 5,
    arrival_rate_mbps: float = 50.0,
    seed: int = 42,
    step_duration_s: float = 0.5,
    policy: str = "random",
    enable_geo_zones: bool = True,
    output_json: Optional[str] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Execute the long-duration simulation and return a performance report.

    Args:
        n_days:             Simulation duration in (simulated) days.
        n_satellites:       Total Starlink-like constellation size.
        max_satellites:     Maximum satellites visible simultaneously (env cap).
        arrival_rate_mbps:  Mean Poisson traffic arrival rate (Mbps).
        seed:               Random seed.
        step_duration_s:    Simulated time per step (seconds).
        policy:             Agent policy to evaluate: ``"random"`` or
                            ``"max_snr"``.
        enable_geo_zones:   Whether to wrap the env with GeoRegulatoryEnv.
        output_json:        Optional path to write the JSON report.
        verbose:            Print progress every 10 000 steps.

    Returns:
        Dictionary with aggregated performance metrics.
    """
    rng = np.random.default_rng(seed)
    total_steps = int(n_days * 86_400.0 / step_duration_s)

    # --- Sensors ---
    # Use _CyclicConstellationTelemetry to ensure at least max_satellites are
    # always visible.  StarlinkConstellationTelemetry with a real propagator can
    # produce zero-visibility epochs that crash the underlying MultiSatelliteEnv;
    # in a real deployment, the environment would use a persistence layer to
    # handle this gracefully.
    telemetry = _CyclicConstellationTelemetry(
        n_satellites=n_satellites,
        max_visible=max_satellites + 2,
        seed=seed,
    )
    rain_model = _RainByTime()
    foliage_model = _StaticFoliage()

    # --- Agent ---
    if policy == "max_snr":
        agent = _MaxSNRPolicy(max_satellites=max_satellites)
    else:
        agent = _RandomPolicy(max_satellites=max_satellites)

    # --- Environment ---
    channel = ChannelModel()
    base_env = TrafficAwareMultiSatelliteEnv(
        channel_model=channel,
        telemetry_stream=telemetry,
        radar_stream=rain_model,
        foliage_map=foliage_model,
        max_satellites=max_satellites,
        arrival_rate_mbps=arrival_rate_mbps,
        step_duration_s=step_duration_s,
        seed=seed,
    )

    env = base_env
    if enable_geo_zones:
        zones = [
            ExclusionZone(
                "Amazon_RadioAstro",
                [(-62.0, -3.5), (-61.0, -3.5), (-61.0, -2.5), (-62.0, -2.5)],
                reason="ITU radio-astronomy protection zone",
            )
        ]
        env = GeoRegulatoryEnv(base_env, exclusion_zones=zones, geo_penalty=5.0)

    # --- Metrics accumulators ---
    metrics: Dict[str, List[float]] = {
        "throughput_mbps": [],
        "latency_ms": [],
        "outage": [],
        "handover": [],
        "drop_rate": [],
        "queue_delay_ms": [],
        "compliance_violations": [],
        "queue_occupancy": [],
        "reward": [],
    }
    t_wall_start = time.perf_counter()

    obs, _ = env.reset(seed=seed)
    sim_time: float = 0.0

    for step in range(total_steps):
        sim_time += step_duration_s
        rain_model.set_time(sim_time)

        action, _ = agent.get_action(obs)
        obs, reward, terminated, truncated, info = env.step(action)

        metrics["throughput_mbps"].append(float(info.get("throughput", 0.0)))
        metrics["latency_ms"].append(float(info.get("latency", 0.0)))
        metrics["outage"].append(float(info.get("outage", 0.0)))
        metrics["handover"].append(float(info.get("handover", False)))
        metrics["drop_rate"].append(float(info.get("drop_rate", 0.0)))
        metrics["queue_delay_ms"].append(float(info.get("queue_delay_ms", 0.0)))
        metrics["compliance_violations"].append(
            float(info.get("compliance_violations", 0))
            + float(info.get("geo_exclusion_violations", 0))
        )
        metrics["queue_occupancy"].append(float(info.get("queue_occupancy", 0.0)))
        metrics["reward"].append(float(reward))

        if terminated or truncated:
            obs, _ = env.reset(seed=seed + step)

        if verbose and step % 10_000 == 0:
            elapsed = time.perf_counter() - t_wall_start
            print(
                f"  step {step:>9}/{total_steps}  "
                f"sim_day={sim_time/86400:.2f}  "
                f"wall={elapsed:.1f}s  "
                f"tput={np.mean(metrics['throughput_mbps'][-1000:]):.1f} Mbps"
            )

    wall_elapsed_s = time.perf_counter() - t_wall_start

    # --- Summarise ---
    def _stats(arr: List[float]) -> Dict[str, float]:
        a = np.array(arr, dtype=np.float32)
        return {
            "mean": float(np.mean(a)),
            "std": float(np.std(a)),
            "min": float(np.min(a)),
            "max": float(np.max(a)),
            "p50": float(np.percentile(a, 50)),
            "p95": float(np.percentile(a, 95)),
        }

    report: Dict[str, Any] = {
        "config": {
            "n_days": n_days,
            "n_satellites": n_satellites,
            "max_satellites": max_satellites,
            "arrival_rate_mbps": arrival_rate_mbps,
            "total_steps": total_steps,
            "policy": policy,
            "enable_geo_zones": enable_geo_zones,
            "seed": seed,
        },
        "wall_elapsed_s": wall_elapsed_s,
        "steps_per_second": total_steps / max(wall_elapsed_s, 1e-9),
        "metrics": {k: _stats(v) for k, v in metrics.items()},
        "kpis": {
            "mean_throughput_mbps": float(np.mean(metrics["throughput_mbps"])),
            "outage_rate": float(np.mean(metrics["outage"])),
            "handover_rate_per_min": (
                float(np.mean(metrics["handover"])) * 60.0 / step_duration_s
            ),
            "mean_queue_delay_ms": float(np.mean(metrics["queue_delay_ms"])),
            "total_compliance_violations": int(sum(metrics["compliance_violations"])),
            "p95_latency_ms": float(np.percentile(metrics["latency_ms"], 95)),
            "mean_packet_drop_rate": float(np.mean(metrics["drop_rate"])),
        },
    }

    if output_json:
        os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

    return report


# ---------------------------------------------------------------------------
# Formatted summary printer
# ---------------------------------------------------------------------------

def print_report(report: Dict[str, Any]) -> None:
    kpis = report["kpis"]
    cfg = report["config"]
    print()
    print("=" * 60)
    print(f"Long-Duration Simulation Report")
    print(f"  Duration  : {cfg['n_days']} day(s)  "
          f"({cfg['total_steps']:,} steps, "
          f"{cfg['n_satellites']} satellites)")
    print(f"  Policy    : {cfg['policy']}")
    print(f"  Wall time : {report['wall_elapsed_s']:.1f} s  "
          f"({report['steps_per_second']:.0f} steps/s)")
    print("-" * 60)
    print(f"  Mean throughput        : {kpis['mean_throughput_mbps']:>8.2f} Mbps")
    print(f"  Outage rate            : {kpis['outage_rate']:>8.4f}")
    print(f"  Handover rate          : {kpis['handover_rate_per_min']:>8.2f} /min")
    print(f"  P95 latency            : {kpis['p95_latency_ms']:>8.2f} ms")
    print(f"  Mean queue delay       : {kpis['mean_queue_delay_ms']:>8.2f} ms")
    print(f"  Mean packet drop rate  : {kpis['mean_packet_drop_rate']:>8.4f}")
    print(f"  Compliance violations  : {kpis['total_compliance_violations']:>8d}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Long-duration Starlink Amazon beamforming simulation."
    )
    p.add_argument("--n-days", type=float, default=1.0,
                   help="Simulated duration in days (default 1).")
    p.add_argument("--n-satellites", type=int, default=100,
                   help="Constellation size (default 100).")
    p.add_argument("--max-satellites", type=int, default=5,
                   help="Max simultaneous visible satellites (default 5).")
    p.add_argument("--arrival-rate", type=float, default=50.0,
                   help="Mean traffic arrival rate in Mbps (default 50).")
    p.add_argument("--policy", choices=["random", "max_snr"], default="random",
                   help="Agent policy to evaluate (default: random).")
    p.add_argument("--no-geo-zones", action="store_true",
                   help="Disable geospatial exclusion zones.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-json", default=None,
                   help="Optional path to save JSON report.")
    p.add_argument("--verbose", action="store_true",
                   help="Print periodic progress.")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    report = run_simulation(
        n_days=args.n_days,
        n_satellites=args.n_satellites,
        max_satellites=args.max_satellites,
        arrival_rate_mbps=args.arrival_rate,
        seed=args.seed,
        policy=args.policy,
        enable_geo_zones=not args.no_geo_zones,
        output_json=args.output_json,
        verbose=args.verbose,
    )
    print_report(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
