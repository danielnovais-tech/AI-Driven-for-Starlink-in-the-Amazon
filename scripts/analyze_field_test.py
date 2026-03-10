#!/usr/bin/env python3
"""
Field Test Analysis and Controller Calibration
===============================================

Loads the JSON report produced by ``scripts/field_test_hardware.py``,
analyses per-scenario metrics, and emits calibration recommendations for:

- **SNR threshold** (``snr_threshold_db``)  – set to the empirical 5th
  percentile of observed SNR to avoid spurious outage declarations.
- **Fallback window / max_failures** – derived from the measured P95
  end-to-end latency; smaller latency budgets allow tighter watchdog windows.
- **Steering gain** – ratio of the expected phase to the achieved phase,
  indicating a systematic driver offset that should be corrected.

Calibration values are written to a JSON calibration file that can be
consumed by the controller at start-up (or applied via the
``OnlineBeamController.apply_calibration()`` method introduced in this wave).

Usage::

    python scripts/analyze_field_test.py \\
        --report /tmp/field_test_report.json \\
        --output-cal /tmp/controller_calibration.json \\
        --verbose

Exit codes:
    0 – analysis complete (recommendations written).
    1 – input report missing or unreadable.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Default calibration baseline (mirrors controller defaults)
# ---------------------------------------------------------------------------

_DEFAULT_SNR_THRESHOLD_DB = 5.0
_DEFAULT_MAX_FAILURES = 3
_DEFAULT_STEERING_GAIN = 1.0
_LATENCY_BUDGET_MS = 500.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_scenario(report: Dict[str, Any], name: str) -> Optional[Dict[str, Any]]:
    """Return the scenario dict with ``scenario == name``, or ``None``."""
    for s in report.get("scenarios", []):
        if s.get("scenario") == name:
            return s
    return None


def _recommend_snr_threshold(
    rain_scenario: Optional[Dict[str, Any]],
    baseline_snr_db: float = 20.0,
) -> float:
    """
    Recommend a new SNR threshold based on rain-injection steps.

    Sets the threshold to the SNR observed at the highest attenuation
    level that still produced a non-fallback decision, minus a 1 dB
    guard-band.  If no rain scenario is present, returns the default.
    """
    if rain_scenario is None:
        return _DEFAULT_SNR_THRESHOLD_DB

    steps = rain_scenario.get("steps", [])
    if not steps:
        return _DEFAULT_SNR_THRESHOLD_DB

    # Steps are ordered from lowest to highest attenuation
    effective_snrs = [s.get("effective_snr_db", baseline_snr_db) for s in steps]
    # Use the P5 SNR as the lower guard boundary
    p5_snr = float(np.percentile(effective_snrs, 5))
    # Threshold is P5 minus 1 dB guard band, clamped to [1.0, 20.0]
    return float(np.clip(p5_snr - 1.0, 1.0, 20.0))


def _recommend_max_failures(
    latency_scenario: Optional[Dict[str, Any]],
    step_interval_ms: float = 500.0,
) -> int:
    """
    Recommend max_failures watchdog threshold from measured P95 latency.

    ``max_failures`` is the number of *consecutive* agent failures before the
    watchdog emits a health-degraded alert.  When P95 latency is close to the
    step-interval budget, each individual step is likely to time out, so the
    watchdog must fire quickly (after 2 consecutive failures instead of 3) to
    avoid accumulating silent degradation across multiple intervals.
    """
    if latency_scenario is None:
        return _DEFAULT_MAX_FAILURES

    p95 = latency_scenario.get("p95_ms", 0.0)
    if p95 > step_interval_ms * 0.8:
        # High latency → tighten watchdog to catch degradation faster
        return 2
    return _DEFAULT_MAX_FAILURES


def _recommend_steering_gain(
    sweep_scenario: Optional[Dict[str, Any]],
) -> float:
    """
    Recommend a steering-gain correction factor from the azimuth-sweep data.

    If the mean achieved phase differs systematically from the commanded
    phase, the gain is ``mean_commanded / mean_achieved``.  A value > 1
    means the hardware under-steers; < 1 means it over-steers.
    """
    if sweep_scenario is None:
        return _DEFAULT_STEERING_GAIN

    commanded = sweep_scenario.get("angles_commanded_deg", [])
    achieved = sweep_scenario.get("angles_achieved_deg", [])
    if not commanded or not achieved or len(commanded) != len(achieved):
        return _DEFAULT_STEERING_GAIN

    # Exclude near-zero commanded angles (noise-dominated)
    pairs = [(c, a) for c, a in zip(commanded, achieved) if abs(c) > 1.0]
    if not pairs:
        return _DEFAULT_STEERING_GAIN

    gains = [abs(c) / abs(a) for c, a in pairs if abs(a) > 1e-6]
    if not gains:
        return _DEFAULT_STEERING_GAIN

    return float(np.median(gains))


# ---------------------------------------------------------------------------
# Main analysis function
# ---------------------------------------------------------------------------

def analyse_report(
    report: Dict[str, Any],
    step_interval_ms: float = 500.0,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Analyse a field-test report and return calibration recommendations.

    Args:
        report:            Parsed JSON report from ``field_test_hardware.py``.
        step_interval_ms:  Controller inference step interval (ms); used to
                           compute the max_failures recommendation.
        verbose:           Print per-scenario analysis details.

    Returns:
        Calibration dictionary with keys:
            ``snr_threshold_db``  – recommended SNR outage threshold (dB).
            ``max_failures``      – recommended watchdog consecutive-failure limit.
            ``steering_gain``     – multiplicative steering gain correction factor.
            ``latency_p95_ms``    – measured P95 end-to-end latency.
            ``steering_std_deg``  – measured steering standard deviation (°).
            ``all_passed``        – True if all field-test scenarios passed.
            ``recommendations``   – list of human-readable recommendation strings.
    """
    boresight = _find_scenario(report, "boresight_baseline")
    sweep = _find_scenario(report, "azimuth_sweep")
    rain = _find_scenario(report, "rain_attenuation_injection")
    latency = _find_scenario(report, "handover_latency")
    precision = _find_scenario(report, "steering_precision")

    snr_threshold = _recommend_snr_threshold(rain)
    max_failures = _recommend_max_failures(latency, step_interval_ms)
    steering_gain = _recommend_steering_gain(sweep)

    latency_p95 = latency.get("p95_ms", 0.0) if latency else 0.0
    steering_std = precision.get("std_deg", 0.0) if precision else 0.0

    recommendations: List[str] = []

    if snr_threshold != _DEFAULT_SNR_THRESHOLD_DB:
        recommendations.append(
            f"Set snr_threshold_db={snr_threshold:.2f}  "
            f"(field 5th-percentile SNR − 1 dB guard band)"
        )
    else:
        recommendations.append(
            f"Keep default snr_threshold_db={snr_threshold:.2f}"
        )

    if max_failures != _DEFAULT_MAX_FAILURES:
        recommendations.append(
            f"Tighten max_failures={max_failures}  "
            f"(P95 latency {latency_p95:.1f} ms is > 80% of step budget)"
        )

    if abs(steering_gain - 1.0) > 0.02:
        recommendations.append(
            f"Apply steering gain correction: {steering_gain:.4f}  "
            f"(multiply commanded phase by this factor before issuing to driver)"
        )

    if latency_p95 > _LATENCY_BUDGET_MS * 0.9:
        recommendations.append(
            f"WARNING: P95 latency {latency_p95:.1f} ms is within 10% of budget "
            f"({_LATENCY_BUDGET_MS} ms) – investigate hardware or network bottleneck"
        )

    if steering_std > 1.0:
        recommendations.append(
            f"WARNING: Steering std dev {steering_std:.3f}° exceeds 1° – "
            "check phased-array driver for jitter or communication errors"
        )

    calibration = {
        "snr_threshold_db": snr_threshold,
        "max_failures": max_failures,
        "steering_gain": steering_gain,
        "latency_p95_ms": latency_p95,
        "steering_std_deg": steering_std,
        "all_passed": bool(report.get("all_passed", False)),
        "recommendations": recommendations,
    }

    if verbose:
        _print_analysis(report, calibration)

    return calibration


def _print_analysis(
    report: Dict[str, Any],
    calibration: Dict[str, Any],
) -> None:
    """Print a human-readable analysis summary."""
    overall = "✅ PASS" if calibration["all_passed"] else "❌ FAIL"
    print()
    print("# Field Test Analysis Report")
    print()
    print(f"Overall result : {overall}")
    print(f"Driver type    : {report.get('driver_type', 'unknown')}")
    print()
    print("## Calibration Recommendations")
    print()
    for rec in calibration["recommendations"]:
        print(f"  • {rec}")
    print()
    print("## Calibration Values")
    print()
    print(f"  snr_threshold_db  = {calibration['snr_threshold_db']:.2f} dB")
    print(f"  max_failures      = {calibration['max_failures']}")
    print(f"  steering_gain     = {calibration['steering_gain']:.4f}")
    print(f"  latency_p95_ms    = {calibration['latency_p95_ms']:.2f} ms")
    print(f"  steering_std_deg  = {calibration['steering_std_deg']:.3f} °")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Analyse field-test report and emit controller calibration."
    )
    p.add_argument("--report", required=True,
                   help="Path to the JSON report from field_test_hardware.py.")
    p.add_argument("--output-cal", default=None,
                   help="Optional path to write calibration JSON.")
    p.add_argument("--step-interval-ms", type=float, default=500.0,
                   help="Controller step interval (ms); used for latency analysis.")
    p.add_argument("--verbose", action="store_true",
                   help="Print detailed analysis to stdout.")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)

    if not os.path.isfile(args.report):
        print(f"ERROR: report not found: {args.report}", file=sys.stderr)
        return 1

    with open(args.report, encoding="utf-8") as f:
        report = json.load(f)

    calibration = analyse_report(
        report,
        step_interval_ms=args.step_interval_ms,
        verbose=args.verbose,
    )

    if args.output_cal:
        os.makedirs(os.path.dirname(args.output_cal) or ".", exist_ok=True)
        with open(args.output_cal, "w", encoding="utf-8") as f:
            json.dump(calibration, f, indent=2)
        if args.verbose:
            print(f"Calibration saved to {args.output_cal}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
