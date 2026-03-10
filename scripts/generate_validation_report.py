#!/usr/bin/env python3
"""
Technical Validation Report Generator
======================================

Aggregates outputs from the field-test campaign, calibration analysis, and
pilot monitor into a single technical validation report suitable for sign-off
before production deployment.

The report evaluates each metric against the hardware specification and marks
it PASS or FAIL:

=====================  ========================================
Metric                 Specification
=====================  ========================================
Steering std dev       ≤ 1.0 ° (array spec)
Steering mean error    ≤ 2.0 ° (boresight tolerance)
Handover latency P95   ≤ 500 ms (SLA)
Handover latency P99   ≤ 750 ms (hard limit)
Outage rate (pilot)    ≤ 0.01  (1 %)
Fallback rate (pilot)  ≤ 0.05  (5 %)
Steering gain          0.9 – 1.1 (±10 % driver offset)
SNR P5 margin          ≥ 1.0 dB above configured threshold
=====================  ========================================

Input files (all optional – missing files produce partial reports):
    field test report    : JSON from ``field_test_hardware.py``
    calibration file     : JSON from ``analyze_field_test.py``
    pilot monitor report : JSON from ``pilot_monitor.py``

Usage::

    python scripts/generate_validation_report.py \\
        --field-test   /tmp/field_test_report.json \\
        --calibration  /tmp/calibration.json \\
        --pilot        /tmp/pilot_report.json \\
        --output-json  /tmp/validation_report.json \\
        --output-md    /tmp/validation_report.md \\
        --verbose

Exit codes:
    0 – all checks PASS (or no checks performed).
    2 – one or more checks FAIL.
    1 – unrecoverable input error.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Specification limits (adjustable via CLI)
# ---------------------------------------------------------------------------

_SPEC = {
    "steering_std_deg_max": 1.0,
    "steering_mean_error_deg_max": 2.0,
    "latency_p95_ms_max": 500.0,
    "latency_p99_ms_max": 750.0,
    "outage_rate_max": 0.01,
    "fallback_rate_max": 0.05,
    "steering_gain_min": 0.9,
    "steering_gain_max": 1.1,
    "snr_threshold_margin_db": 1.0,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load(path: Optional[str]) -> Optional[Dict[str, Any]]:
    if path and os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return None


def _check(
    label: str,
    value: Optional[float],
    limit: float,
    high_is_bad: bool = True,
) -> Dict[str, Any]:
    """Build a single check result row."""
    if value is None:
        return {"label": label, "value": None, "limit": limit, "status": "N/A"}
    ok = (value <= limit) if high_is_bad else (value >= limit)
    return {
        "label": label,
        "value": value,
        "limit": limit,
        "status": "PASS" if ok else "FAIL",
    }


# ---------------------------------------------------------------------------
# Main report function
# ---------------------------------------------------------------------------

def build_validation_report(
    ft_report: Optional[Dict[str, Any]],
    cal: Optional[Dict[str, Any]],
    pilot: Optional[Dict[str, Any]],
    spec: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build the validation report from pre-loaded data dicts.

    Args:
        ft_report: Parsed field-test JSON report.
        cal:       Parsed calibration JSON.
        pilot:     Parsed pilot-monitor JSON report.
        spec:      Override specification limits (merged with defaults).

    Returns:
        Report dictionary with keys ``checks``, ``all_passed``,
        ``generated_at``, ``field_test_passed``, ``summary``.
    """
    limits = dict(_SPEC)
    if spec:
        limits.update(spec)

    checks: List[Dict[str, Any]] = []

    # ---- Field-test metrics ----
    ft_passed: Optional[bool] = None
    if ft_report:
        ft_passed = bool(ft_report.get("all_passed", False))

        # Steering precision scenario
        prec = _find_scenario(ft_report, "steering_precision")
        if prec:
            checks.append(_check(
                "Steering std dev (°)",
                prec.get("std_deg"),
                limits["steering_std_deg_max"],
            ))
            checks.append(_check(
                "Steering mean error (°)",
                prec.get("mean_error_deg"),
                limits["steering_mean_error_deg_max"],
            ))

        # Boresight
        boresight = _find_scenario(ft_report, "boresight_baseline")
        if boresight:
            checks.append(_check(
                "Boresight phase error (°)",
                boresight.get("phase_error_deg"),
                boresight.get("tolerance_deg", 2.0),
            ))

        # Handover latency
        lat_sc = _find_scenario(ft_report, "handover_latency")
        if lat_sc:
            checks.append(_check(
                "Handover latency P95 (ms)",
                lat_sc.get("p95_ms"),
                limits["latency_p95_ms_max"],
            ))
            checks.append(_check(
                "Handover latency P99 (ms)",
                lat_sc.get("p99_ms"),
                limits["latency_p99_ms_max"],
            ))

    # ---- Calibration metrics ----
    if cal:
        gain = cal.get("steering_gain")
        if gain is not None:
            in_range = limits["steering_gain_min"] <= gain <= limits["steering_gain_max"]
            checks.append({
                "label": "Steering gain (ratio)",
                "value": gain,
                "limit": f"{limits['steering_gain_min']}–{limits['steering_gain_max']}",
                "status": "PASS" if in_range else "FAIL",
            })

        # SNR threshold margin: P5 SNR from field-test vs. configured threshold
        snr_thr = cal.get("snr_threshold_db")
        ft_p5 = _ft_p5_snr(ft_report)
        if snr_thr is not None and ft_p5 is not None:
            margin = ft_p5 - snr_thr
            checks.append(_check(
                "SNR P5 margin above threshold (dB)",
                margin,
                limits["snr_threshold_margin_db"],
                high_is_bad=False,
            ))

    # ---- Pilot metrics ----
    if pilot:
        checks.append(_check(
            "Pilot outage rate",
            pilot.get("outage_rate"),
            limits["outage_rate_max"],
        ))
        checks.append(_check(
            "Pilot fallback rate",
            pilot.get("fallback_rate"),
            limits["fallback_rate_max"],
        ))
        lat_stats = pilot.get("latency_stats", {})
        checks.append(_check(
            "Pilot inference latency P95 (ms)",
            lat_stats.get("p95"),
            limits["latency_p95_ms_max"],
        ))

    # ---- Summary ----
    fail_count = sum(1 for c in checks if c.get("status") == "FAIL")
    pass_count = sum(1 for c in checks if c.get("status") == "PASS")
    all_passed = (fail_count == 0 and pass_count > 0)

    return {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "field_test_passed": ft_passed,
        "checks": checks,
        "pass_count": pass_count,
        "fail_count": fail_count,
        "all_passed": all_passed,
        "specification": limits,
    }


def _find_scenario(
    report: Dict[str, Any], name: str
) -> Optional[Dict[str, Any]]:
    for s in report.get("scenarios", []):
        if s.get("scenario") == name:
            return s
    return None


def _ft_p5_snr(ft_report: Optional[Dict[str, Any]]) -> Optional[float]:
    if not ft_report:
        return None
    rain = _find_scenario(ft_report, "rain_attenuation_injection")
    if not rain:
        return None
    snrs = [s.get("effective_snr_db") for s in rain.get("steps", [])
            if s.get("effective_snr_db") is not None]
    if not snrs:
        return None
    import numpy as np
    return float(np.percentile(snrs, 5))


# ---------------------------------------------------------------------------
# Markdown renderer
# ---------------------------------------------------------------------------

def render_markdown(report: Dict[str, Any]) -> str:
    """Render the validation report as a Markdown document."""
    lines = [
        "# Technical Validation Report",
        "",
        f"Generated : {report['generated_at']}",
        "",
    ]

    ft = report.get("field_test_passed")
    if ft is not None:
        lines.append(f"Field test campaign : {'✅ PASS' if ft else '❌ FAIL'}")
        lines.append("")

    overall = "✅ ALL PASS" if report["all_passed"] else "❌ FAIL(S) DETECTED"
    lines += [
        f"Overall result : {overall}  "
        f"({report['pass_count']} PASS / {report['fail_count']} FAIL)",
        "",
        "## Acceptance Criteria",
        "",
        "| Metric | Value | Limit | Status |",
        "|---|---|---|---|",
    ]

    for c in report["checks"]:
        val = c["value"]
        val_str = f"{val:.4g}" if isinstance(val, float) else str(val)
        status = c["status"]
        icon = {"PASS": "✅", "FAIL": "❌", "N/A": "—"}.get(status, status)
        lines.append(f"| {c['label']} | {val_str} | {c['limit']} | {icon} {status} |")

    lines += [
        "",
        "## Specification",
        "",
    ]
    for k, v in report.get("specification", {}).items():
        lines.append(f"- `{k}` = {v}")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Generate technical validation report from field-test data."
    )
    p.add_argument("--field-test", default=None,
                   help="Path to field_test_hardware.py JSON report.")
    p.add_argument("--calibration", default=None,
                   help="Path to analyze_field_test.py calibration JSON.")
    p.add_argument("--pilot", default=None,
                   help="Path to pilot_monitor.py JSON report.")
    p.add_argument("--output-json", default=None,
                   help="Output path for the validation report JSON.")
    p.add_argument("--output-md", default=None,
                   help="Output path for the Markdown report.")
    p.add_argument("--verbose", action="store_true",
                   help="Print Markdown report to stdout.")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)

    ft = _load(args.field_test)
    cal = _load(args.calibration)
    pilot = _load(args.pilot)

    if ft is None and cal is None and pilot is None:
        print("WARNING: no input files found – generating empty report.", file=sys.stderr)

    report = build_validation_report(ft, cal, pilot)
    md = render_markdown(report)

    if args.verbose:
        print(md)

    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

    if args.output_md:
        os.makedirs(os.path.dirname(args.output_md) or ".", exist_ok=True)
        with open(args.output_md, "w", encoding="utf-8") as f:
            f.write(md)

    return 2 if report["fail_count"] > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
