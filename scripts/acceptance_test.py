#!/usr/bin/env python3
"""
Production Acceptance Test
==========================

Simulates a 48-hour (configurable) production acceptance test using the
inference controller and synthetic data sources.  The test mirrors a real
production deployment by:

  - Running the full inference loop at the configured step rate.
  - Injecting diurnal SNR variation (Amazon rain peak in early afternoon).
  - Recording all QoS / QoE metrics.
  - Verifying that Prometheus-style alert thresholds are *not* breached.
  - Writing a structured JSON + Markdown acceptance report.

Acceptance criteria (all must pass for the test to succeed):
    - Outage rate ≤ 1 %
    - Fallback rate ≤ 5 %
    - Inference latency P95 ≤ 500 ms
    - Inference latency P99 ≤ 750 ms
    - Zero consecutive-failure watchdog alerts (``max_failures`` exceeded)
    - Handover success: handovers per minute ≤ configured limit

Usage::

    # Fast CI/staging run (1000 steps ≈ 8 min of simulated operation):
    python scripts/acceptance_test.py \\
        --steps 1000 \\
        --output-json /tmp/acceptance_report.json \\
        --output-md   /tmp/acceptance_report.md \\
        --verbose

    # Full 48-hour simulation (steps = 48 * 3600 / interval):
    python scripts/acceptance_test.py \\
        --steps 172800 \\
        --interval-s 1.0 \\
        --output-json /tmp/acceptance_report_full.json

Exit codes:
    0 – all acceptance criteria PASS.
    2 – one or more acceptance criteria FAIL.
    1 – initialisation error.
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

from hardware.phaser_driver import NullPhasedArrayDriver
from inference.online_controller import HardwareBeamController


# ---------------------------------------------------------------------------
# Acceptance criteria
# ---------------------------------------------------------------------------

_ACCEPTANCE_CRITERIA: Dict[str, Any] = {
    "outage_rate_max": 0.01,
    "fallback_rate_max": 0.05,
    "latency_p95_ms_max": 500.0,
    "latency_p99_ms_max": 750.0,
    "watchdog_alerts_max": 0,
    "handover_rate_per_min_max": 10.0,
}


# ---------------------------------------------------------------------------
# Synthetic sensor stubs (same as pilot_monitor but with more detail)
# ---------------------------------------------------------------------------

class _AcceptanceTelemetry:
    """
    Synthetic telemetry simulating a 48-hour Amazon deployment profile:

    - Diurnal SNR variation (~5 dB peak-to-peak).
    - Rain fade events in the afternoon (probability proportional to
      the Amazon monthly rain cycle).
    - Occasional deep fades simulating dense canopy blockage.
    """

    ground_station_pos = np.array([0.0, 0.0, 6371.0])

    def __init__(self, seed: int = 0) -> None:
        self._rng = np.random.default_rng(seed)
        self._step = 0
        self._total_snr_below_threshold = 0

    def get_current_snr(self) -> float:
        # Simulate a day cycle (1440 minutes), step = 1 s
        minutes = (self._step / 60.0) % 1440.0
        # Diurnal dip: deepest around minute 840 (14:00 local)
        diurnal = -4.0 * np.sin(np.pi * minutes / 1440.0)
        # Rain fade: higher probability in the afternoon (min 720–1020)
        rain_prob = 0.002 if 720 <= minutes <= 1020 else 0.0005
        base_snr = 18.0 + diurnal + self._rng.normal(0.0, 1.5)
        if self._rng.random() < rain_prob:
            base_snr -= self._rng.uniform(5.0, 20.0)
        # Canopy blockage (0.1% probability)
        if self._rng.random() < 0.001:
            base_snr -= 25.0
        self._step += 1
        return float(np.clip(base_snr, -15.0, 35.0))

    def get_current_rssi(self) -> float:
        return -80.0 + self._rng.normal(0.0, 2.0)

    def get_current_position(self) -> np.ndarray:
        return np.array([6921.0, 0.0, 0.0])


class _AcceptanceRadar:
    def __init__(self, seed: int = 0) -> None:
        self._rng = np.random.default_rng(seed)

    def get_at_location(self, pos) -> float:
        return max(0.0, self._rng.exponential(2.0))


class _AcceptanceFoliage:
    def get_at_location(self, pos) -> float:
        return 3.0  # dense Amazonian canopy


class _AcceptanceAgent:
    def get_action(self, state, deterministic=True):
        snr = float(state[0]) if hasattr(state, "__len__") else float(state)
        power = float(np.clip(1.0 - max(0.0, -snr) / 30.0, 0.1, 1.0))
        mcs = int(np.clip(round(snr / 6.0), 0, 4))
        return np.array([0.0, power, float(mcs), 50.0], dtype=np.float32), 0.0


# ---------------------------------------------------------------------------
# Acceptance test runner
# ---------------------------------------------------------------------------

class _Accumulator:
    def __init__(self) -> None:
        self._vals: List[float] = []

    def record(self, v: float) -> None:
        self._vals.append(v)

    def summary(self) -> Dict[str, float]:
        if not self._vals:
            return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0, "count": 0}
        arr = np.array(self._vals)
        return {
            "mean": float(np.mean(arr)),
            "p50": float(np.percentile(arr, 50)),
            "p95": float(np.percentile(arr, 95)),
            "p99": float(np.percentile(arr, 99)),
            "count": len(self._vals),
        }


def _check_criterion(
    label: str,
    value: float,
    limit: float,
    high_is_bad: bool = True,
) -> Dict[str, Any]:
    ok = (value <= limit) if high_is_bad else (value >= limit)
    return {
        "label": label,
        "value": value,
        "limit": limit,
        "status": "PASS" if ok else "FAIL",
    }


def run_acceptance_test(
    steps: int = 1000,
    interval_s: float = 0.0,
    snr_threshold_db: float = 5.0,
    calibration: Optional[Dict[str, Any]] = None,
    criteria: Optional[Dict[str, Any]] = None,
    output_json: Optional[str] = None,
    output_md: Optional[str] = None,
    verbose: bool = False,
    seed: int = 0,
) -> Dict[str, Any]:
    """
    Run the acceptance test and return the report.

    Args:
        steps:             Number of inference steps to run.
        interval_s:        Sleep between steps (use 0.0 for CI/fast mode).
        snr_threshold_db:  SNR outage threshold (dB).
        calibration:       Optional calibration dict from analyze_field_test.
        criteria:          Override acceptance criteria.
        output_json:       Optional path to write JSON report.
        output_md:         Optional path to write Markdown report.
        verbose:           Print step progress.
        seed:              Random seed.

    Returns:
        Acceptance test report dictionary.
    """
    crit = dict(_ACCEPTANCE_CRITERIA)
    if criteria:
        crit.update(criteria)

    driver = NullPhasedArrayDriver()
    ctrl = HardwareBeamController(
        agent=_AcceptanceAgent(),
        telemetry_stream=_AcceptanceTelemetry(seed=seed),
        radar_stream=_AcceptanceRadar(seed=seed),
        foliage_map=_AcceptanceFoliage(),
        hw_driver=driver,
        snr_threshold_db=snr_threshold_db,
    )
    if calibration:
        ctrl.apply_calibration(calibration)

    snr_acc = _Accumulator()
    lat_acc = _Accumulator()
    outage_steps = 0
    fallback_steps = 0
    watchdog_alerts = 0
    handovers = 0
    prev_action_key = None

    t_start = time.perf_counter()
    completed = 0

    for step in range(steps):
        result = ctrl.step()
        completed += 1
        snr = result.get("snr", 0.0)
        lat = result.get("latency_ms", 0.0)
        fb = result.get("fallback", False)
        action = result.get("action")

        snr_acc.record(snr)
        lat_acc.record(lat)

        if snr < ctrl.snr_threshold:
            outage_steps += 1
        if fb:
            fallback_steps += 1
        if not ctrl.is_healthy:
            watchdog_alerts += 1

        ak = (
            tuple(float(x) for x in action)
            if hasattr(action, "__len__")
            else (float(action),)
        )
        if prev_action_key is not None and ak != prev_action_key:
            handovers += 1
        prev_action_key = ak

        if verbose and step % max(1, steps // 10) == 0:
            print(
                f"  Step {step:6d}/{steps}  "
                f"SNR={snr:6.2f}dB  lat={lat:.2f}ms  "
                f"outages={outage_steps}  ho={handovers}",
                flush=True,
            )

        if interval_s > 0:
            time.sleep(interval_s)

    elapsed_s = time.perf_counter() - t_start
    total = completed if completed > 0 else 1
    outage_rate = outage_steps / total
    fallback_rate = fallback_steps / total
    elapsed_min = elapsed_s / 60.0
    ho_rate = handovers / max(elapsed_min, 1e-6)

    # Evaluate acceptance criteria
    lat_stats = lat_acc.summary()
    checks: List[Dict[str, Any]] = [
        _check_criterion("Outage rate", outage_rate, crit["outage_rate_max"]),
        _check_criterion("Fallback rate", fallback_rate, crit["fallback_rate_max"]),
        _check_criterion(
            "Latency P95 (ms)", lat_stats["p95"], crit["latency_p95_ms_max"]
        ),
        _check_criterion(
            "Latency P99 (ms)", lat_stats["p99"], crit["latency_p99_ms_max"]
        ),
        _check_criterion(
            "Watchdog alerts", float(watchdog_alerts), float(crit["watchdog_alerts_max"])
        ),
        _check_criterion(
            "Handover rate (/ min)", ho_rate, crit["handover_rate_per_min_max"]
        ),
    ]

    fail_count = sum(1 for c in checks if c["status"] == "FAIL")
    all_passed = fail_count == 0

    report: Dict[str, Any] = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "completed_steps": completed,
        "elapsed_s": elapsed_s,
        "snr_threshold_db": ctrl.snr_threshold,
        "outage_rate": outage_rate,
        "fallback_rate": fallback_rate,
        "watchdog_alerts": watchdog_alerts,
        "handovers": handovers,
        "handover_rate_per_min": ho_rate,
        "snr_stats": snr_acc.summary(),
        "latency_stats": lat_stats,
        "checks": checks,
        "fail_count": fail_count,
        "all_passed": all_passed,
        "acceptance_criteria": crit,
    }

    md = _render_acceptance_md(report)

    if output_json:
        os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

    if output_md:
        os.makedirs(os.path.dirname(output_md) or ".", exist_ok=True)
        with open(output_md, "w", encoding="utf-8") as f:
            f.write(md)

    if verbose:
        print(md)

    return report


# ---------------------------------------------------------------------------
# Markdown renderer
# ---------------------------------------------------------------------------

def _render_acceptance_md(report: Dict[str, Any]) -> str:
    status = "✅ ACCEPTED" if report["all_passed"] else "❌ REJECTED"
    lines = [
        "# Production Acceptance Test Report",
        "",
        f"Generated  : {report['generated_at']}",
        f"Steps run  : {report['completed_steps']}",
        f"Elapsed    : {report['elapsed_s']:.1f} s",
        f"Result     : {status}  "
        f"({len(report['checks']) - report['fail_count']} PASS / "
        f"{report['fail_count']} FAIL)",
        "",
        "## Acceptance Criteria",
        "",
        "| Metric | Value | Limit | Status |",
        "|---|---|---|---|",
    ]
    for c in report["checks"]:
        icon = "✅" if c["status"] == "PASS" else "❌"
        lines.append(
            f"| {c['label']} | {c['value']:.4g} | {c['limit']:.4g} | {icon} |"
        )
    lines += [
        "",
        "## QoS / QoE Summary",
        "",
        f"| SNR mean (dB) | SNR P5 (dB) | Latency P50 (ms) | Latency P95 (ms) |",
        f"|---|---|---|---|",
        f"| {report['snr_stats']['mean']:.2f} "
        f"| {report['snr_stats'].get('p5', '—') if 'p5' in report['snr_stats'] else '—'} "
        f"| {report['latency_stats']['p50']:.2f} "
        f"| {report['latency_stats']['p95']:.2f} |",
    ]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Production acceptance test for the beamforming controller."
    )
    p.add_argument("--steps", type=int, default=1000,
                   help="Number of inference steps (default 1000).")
    p.add_argument("--interval-s", type=float, default=0.0,
                   help="Sleep interval between steps (default 0 = fast).")
    p.add_argument("--snr-threshold-db", type=float, default=5.0,
                   help="SNR outage threshold (dB).")
    p.add_argument("--calibration", default=None,
                   help="Path to calibration JSON from analyze_field_test.py.")
    p.add_argument("--output-json", default=None,
                   help="Output path for acceptance test JSON report.")
    p.add_argument("--output-md", default=None,
                   help="Output path for Markdown acceptance report.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)

    calibration = None
    if args.calibration:
        if not os.path.isfile(args.calibration):
            print(f"ERROR: calibration file not found: {args.calibration}",
                  file=sys.stderr)
            return 1
        with open(args.calibration, encoding="utf-8") as f:
            calibration = json.load(f)

    report = run_acceptance_test(
        steps=args.steps,
        interval_s=args.interval_s,
        snr_threshold_db=args.snr_threshold_db,
        calibration=calibration,
        output_json=args.output_json,
        output_md=args.output_md,
        verbose=args.verbose,
        seed=args.seed,
    )
    return 0 if report["all_passed"] else 2


if __name__ == "__main__":
    sys.exit(main())
