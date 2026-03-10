#!/usr/bin/env python3
"""
Pilot Deployment Monitor
========================

Runs a continuous monitoring loop that collects QoS/QoE metrics from the
inference controller and writes them to a rolling JSON report.  Designed
for the 1-week (or longer) pilot deployment scenario.

Metrics collected per interval:
    - mean / P50 / P95 / P99 SNR (dB)
    - outage rate (fraction of steps below SNR threshold)
    - handover rate (state changes per minute)
    - inference latency P50 / P95 (ms)
    - fallback activation rate (fraction of steps using fallback)
    - number of watchdog alerts

The monitor runs until ``--max-steps`` steps have been collected, or until
a ``SIGINT`` (Ctrl-C) or ``SIGTERM`` signal is received.  All accumulated
data is flushed to disk before exit.

Usage::

    # Simulation mode (uses stub telemetry):
    python scripts/pilot_monitor.py \\
        --steps 1000 \\
        --interval-s 0.5 \\
        --output-json /tmp/pilot_report.json \\
        --verbose

    # With calibration applied at startup:
    python scripts/pilot_monitor.py \\
        --calibration /tmp/controller_calibration.json \\
        --steps 5000 \\
        --output-json /tmp/pilot_report.json

Exit codes:
    0 – completed normally or interrupted cleanly.
    1 – unrecoverable error during initialisation.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from typing import Any, Dict, List, Optional

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from hardware.phaser_driver import NullPhasedArrayDriver, LoggingPhasedArrayDriver
from inference.online_controller import HardwareBeamController

# ---------------------------------------------------------------------------
# Minimal sensor stubs (used when no live telemetry is available)
# ---------------------------------------------------------------------------

_EARTH_RADIUS_KM = 6371.0


class _PilotTelemetry:
    """
    Synthetic telemetry that simulates a time-varying SNR profile,
    including diurnal Amazon rain patterns and occasional deep fades.
    """

    ground_station_pos = np.array([0.0, 0.0, _EARTH_RADIUS_KM])

    def __init__(self, seed: int = 0) -> None:
        self._rng = np.random.default_rng(seed)
        self._step = 0

    def get_current_snr(self) -> float:
        t = self._step / 60.0  # minutes
        # Diurnal component: SNR dips in mid-afternoon (Amazon rain peak)
        diurnal = -3.0 * np.sin(2 * np.pi * t / 1440.0)
        base = 18.0 + diurnal + self._rng.normal(0.0, 2.0)
        # Occasional deep fades (-15 dB, 0.5% probability per step)
        if self._rng.random() < 0.005:
            base -= 15.0
        self._step += 1
        return float(np.clip(base, -10.0, 35.0))

    def get_current_rssi(self) -> float:
        return -80.0 + self._rng.normal(0.0, 2.0)

    def get_current_position(self) -> np.ndarray:
        return np.array([_EARTH_RADIUS_KM + 550.0, 0.0, 0.0])


class _PilotRadar:
    def __init__(self, seed: int = 0) -> None:
        self._rng = np.random.default_rng(seed)

    def get_at_location(self, pos) -> float:
        return max(0.0, self._rng.exponential(3.0))


class _PilotFoliage:
    def get_at_location(self, pos) -> float:
        return 2.0  # dense Amazon canopy


class _PilotAgent:
    """Simple max-SNR agent stub for the pilot monitor."""

    def get_action(self, state, deterministic=True):
        snr = float(state[0]) if hasattr(state, "__len__") else float(state)
        power = float(np.clip(1.0 - max(0.0, -snr) / 30.0, 0.1, 1.0))
        mcs = int(np.clip(round(snr / 6.0), 0, 4))
        return np.array([0.0, power, float(mcs), 50.0], dtype=np.float32), 0.0


# ---------------------------------------------------------------------------
# QoS/QoE metric accumulators
# ---------------------------------------------------------------------------

class _Accumulator:
    """Rolling window accumulator for a single metric."""

    def __init__(self) -> None:
        self._values: List[float] = []

    def record(self, value: float) -> None:
        self._values.append(value)

    def summary(self) -> Dict[str, float]:
        if not self._values:
            return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0, "count": 0}
        arr = np.array(self._values, dtype=np.float64)
        return {
            "mean": float(np.mean(arr)),
            "p50": float(np.percentile(arr, 50)),
            "p95": float(np.percentile(arr, 95)),
            "p99": float(np.percentile(arr, 99)),
            "count": len(self._values),
        }

    def clear(self) -> None:
        self._values.clear()


# ---------------------------------------------------------------------------
# Monitor
# ---------------------------------------------------------------------------

def run_pilot_monitor(
    max_steps: int = 1000,
    interval_s: float = 0.5,
    snr_threshold_db: float = 5.0,
    calibration: Optional[Dict[str, Any]] = None,
    output_json: Optional[str] = None,
    verbose: bool = False,
    seed: int = 0,
) -> Dict[str, Any]:
    """
    Execute the pilot monitoring loop.

    Args:
        max_steps:         Maximum number of inference steps to collect.
        interval_s:        Sleep interval between steps (seconds).
        snr_threshold_db:  SNR threshold for outage detection (dB).
        calibration:       Optional calibration dict (from analyze_field_test).
        output_json:       Optional path to save the JSON report.
        verbose:           Print per-step progress.
        seed:              Random seed for the synthetic telemetry.

    Returns:
        Consolidated pilot report dictionary.
    """
    # --- Build controller with stub data sources ---
    driver = NullPhasedArrayDriver()
    tel = _PilotTelemetry(seed=seed)
    agent = _PilotAgent()

    ctrl = HardwareBeamController(
        agent=agent,
        telemetry_stream=tel,
        radar_stream=_PilotRadar(seed=seed),
        foliage_map=_PilotFoliage(),
        hw_driver=driver,
        snr_threshold_db=snr_threshold_db,
    )

    # Apply calibration if provided
    if calibration:
        ctrl.apply_calibration(calibration)

    # --- Metric accumulators ---
    snr_acc = _Accumulator()
    latency_acc = _Accumulator()

    outage_steps = 0
    fallback_steps = 0
    watchdog_alerts = 0
    handovers = 0
    prev_action = None

    # --- Graceful shutdown handler ---
    _interrupted = [False]

    def _handle_signal(signum, frame):  # noqa: ANN001
        _interrupted[0] = True

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    t_start = time.perf_counter()
    completed_steps = 0

    for step in range(max_steps):
        if _interrupted[0]:
            if verbose:
                print(f"  Interrupted at step {step}")
            break

        result = ctrl.step()
        completed_steps += 1

        snr = result.get("snr", 0.0)
        lat = result.get("latency_ms", 0.0)
        fb = result.get("fallback", False)
        action = result.get("action")

        snr_acc.record(snr)
        latency_acc.record(lat)

        if snr < ctrl.snr_threshold:
            outage_steps += 1
        if fb:
            fallback_steps += 1
        if not ctrl.is_healthy:
            watchdog_alerts += 1
        # Detect handover: any element of the action vector changed
        action_key = (
            tuple(float(x) for x in action)
            if hasattr(action, "__len__")
            else (float(action),)
        )
        if prev_action is not None and action_key != prev_action:
            handovers += 1
        prev_action = action_key

        if verbose and step % max(1, max_steps // 20) == 0:
            print(
                f"  Step {step:5d}/{max_steps}  "
                f"SNR={snr:6.2f}dB  lat={lat:.2f}ms  "
                f"outages={outage_steps}  ho={handovers}",
                flush=True,
            )

        time.sleep(interval_s)

    elapsed_s = time.perf_counter() - t_start

    # --- Build report ---
    total = completed_steps if completed_steps > 0 else 1
    outage_rate = outage_steps / total
    fallback_rate = fallback_steps / total
    elapsed_min = elapsed_s / 60.0
    handover_rate_per_min = handovers / max(elapsed_min, 1e-6)

    report: Dict[str, Any] = {
        "completed_steps": completed_steps,
        "elapsed_s": elapsed_s,
        "snr_threshold_db": ctrl.snr_threshold,
        "max_failures": ctrl.max_failures,
        "outage_rate": outage_rate,
        "outage_steps": outage_steps,
        "fallback_rate": fallback_rate,
        "fallback_steps": fallback_steps,
        "watchdog_alerts": watchdog_alerts,
        "handovers": handovers,
        "handover_rate_per_min": handover_rate_per_min,
        "snr_stats": snr_acc.summary(),
        "latency_stats": latency_acc.summary(),
    }

    if output_json:
        os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        if verbose:
            print(f"\nReport saved to {output_json}")

    return report


def _print_summary(report: Dict[str, Any]) -> None:
    """Print a Markdown-formatted pilot report summary."""
    print()
    print("# Pilot Monitor Report")
    print()
    print(f"Steps completed : {report['completed_steps']}")
    print(f"Elapsed time    : {report['elapsed_s']:.1f} s")
    print()
    print("## QoS / QoE Metrics")
    print()
    print("| Metric | Value |")
    print("|---|---|")
    print(f"| Outage rate | {report['outage_rate']:.4f} ({report['outage_steps']} / {report['completed_steps']}) |")
    print(f"| Fallback rate | {report['fallback_rate']:.4f} |")
    print(f"| Watchdog alerts | {report['watchdog_alerts']} |")
    print(f"| Handovers | {report['handovers']} ({report['handover_rate_per_min']:.2f} / min) |")
    print(f"| SNR mean | {report['snr_stats']['mean']:.2f} dB |")
    print(f"| SNR P95 | {report['snr_stats']['p95']:.2f} dB |")
    print(f"| Latency P50 | {report['latency_stats']['p50']:.2f} ms |")
    print(f"| Latency P95 | {report['latency_stats']['p95']:.2f} ms |")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Pilot deployment monitor – collect QoS/QoE metrics."
    )
    p.add_argument("--steps", type=int, default=1000,
                   help="Maximum number of inference steps (default 1000).")
    p.add_argument("--interval-s", type=float, default=0.0,
                   help="Sleep interval between steps in seconds (default 0, no sleep).")
    p.add_argument("--snr-threshold-db", type=float, default=5.0,
                   help="SNR outage threshold (dB).")
    p.add_argument("--calibration", default=None,
                   help="Path to calibration JSON from analyze_field_test.py.")
    p.add_argument("--output-json", default=None,
                   help="Optional path to save pilot report JSON.")
    p.add_argument("--seed", type=int, default=0,
                   help="Random seed for synthetic telemetry.")
    p.add_argument("--verbose", action="store_true",
                   help="Print per-step progress.")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)

    calibration = None
    if args.calibration:
        if not os.path.isfile(args.calibration):
            print(f"ERROR: calibration file not found: {args.calibration}", file=sys.stderr)
            return 1
        with open(args.calibration, encoding="utf-8") as f:
            calibration = json.load(f)

    report = run_pilot_monitor(
        max_steps=args.steps,
        interval_s=args.interval_s,
        snr_threshold_db=args.snr_threshold_db,
        calibration=calibration,
        output_json=args.output_json,
        verbose=args.verbose,
        seed=args.seed,
    )
    _print_summary(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
