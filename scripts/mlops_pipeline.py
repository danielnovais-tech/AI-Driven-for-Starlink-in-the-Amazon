#!/usr/bin/env python3
"""
MLOps Pipeline – Automated Collect → Analyse → Retrain → Deploy Cycle
======================================================================

Orchestrates the full end-to-end MLOps workflow:

  1. **Field test** (optional) – run ``field_test_hardware.py`` scenarios in
     simulation mode and save the JSON report.
  2. **Analyse** – run ``analyze_field_test.py`` on the report and emit a
     calibration file.
  3. **Pilot monitor** – execute a short monitoring run and collect QoS/QoE
     metrics.
  4. **Retrain** – run ``retrain_job.py`` (the existing
     :func:`~scripts.retrain_job.run_retrain` function) with the validated
     model-registry path.
  5. **Audit log** – write a consolidated JSON audit log of the entire run,
     recording versions, metrics, and promotion decision.

This script is designed to be called from a CI/CD system (e.g.,
``scripts/mlops_pipeline.py --mode ci``) or from the Kubernetes
``CronJob`` already configured in
``helm/beamforming/templates/cronjob.yaml``.

Environment variables (all have ``--`` CLI flag equivalents):
    ``MODEL_REGISTRY_PATH``    Root path for the model registry.
    ``MODEL_NAME``             Logical model name.
    ``MLOPS_SKIP_FIELD_TEST``  Set to ``1`` to skip the field-test step.
    ``MLOPS_SKIP_PILOT``       Set to ``1`` to skip the pilot-monitor step.
    ``MLOPS_AUDIT_LOG``        Path to write the audit log.

Usage::

    # Full pipeline (all stages):
    python scripts/mlops_pipeline.py \\
        --registry /tmp/models \\
        --model-name ppo_amazon \\
        --output-audit /tmp/mlops_audit.json \\
        --verbose

    # CI mode (skip pilot, use minimal steps):
    python scripts/mlops_pipeline.py \\
        --mode ci \\
        --registry /tmp/models \\
        --output-audit /tmp/mlops_audit.json

Exit codes:
    0 – pipeline completed; model promoted or retained without error.
    1 – pipeline failed at any stage.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))


# ---------------------------------------------------------------------------
# Stage wrappers
# ---------------------------------------------------------------------------

def _stage_field_test(
    output_json: str,
    n_steps: int = 20,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Run field-test scenarios and return the report."""
    from field_test_hardware import run_field_test  # type: ignore[import]
    if verbose:
        print("[mlops] Stage 1: field test …")
    report = run_field_test(
        driver_type="null",
        n_steps=n_steps,
        output_json=output_json,
        verbose=verbose,
    )
    return report


def _stage_analyse(
    report_json: str,
    cal_json: str,
    step_interval_ms: float = 500.0,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Analyse field-test report and return calibration dict."""
    from analyze_field_test import analyse_report  # type: ignore[import]
    if verbose:
        print("[mlops] Stage 2: analyse field test …")
    with open(report_json, encoding="utf-8") as f:
        report = json.load(f)
    cal = analyse_report(report, step_interval_ms=step_interval_ms, verbose=verbose)
    os.makedirs(os.path.dirname(cal_json) or ".", exist_ok=True)
    with open(cal_json, "w", encoding="utf-8") as f:
        json.dump(cal, f, indent=2)
    return cal


def _stage_pilot(
    calibration: Dict[str, Any],
    output_json: str,
    pilot_steps: int = 100,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Run pilot monitor and return QoS/QoE report."""
    from pilot_monitor import run_pilot_monitor  # type: ignore[import]
    if verbose:
        print(f"[mlops] Stage 3: pilot monitor ({pilot_steps} steps) …")
    report = run_pilot_monitor(
        max_steps=pilot_steps,
        interval_s=0.0,
        calibration=calibration,
        output_json=output_json,
        verbose=verbose,
    )
    return report


def _stage_retrain(
    registry_path: str,
    model_name: str,
    n_episodes: int = 20,
    rounds: int = 3,
    outage_threshold: float = 1.0,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Run retrain job and return summary."""
    from retrain_job import run_retrain  # type: ignore[import]
    if verbose:
        print("[mlops] Stage 4: retrain …")
    summary = run_retrain(
        registry_path=registry_path,
        model_name=model_name,
        n_episodes=n_episodes,
        rounds=rounds,
        outage_threshold=outage_threshold,
    )
    return summary


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def run_pipeline(
    registry_path: str = "/tmp/models",
    model_name: str = "ppo_amazon",
    mode: str = "full",
    skip_field_test: bool = False,
    skip_pilot: bool = False,
    output_audit: Optional[str] = None,
    tmp_dir: str = "/tmp/mlops_pipeline",
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Execute the full MLOps pipeline and return the audit record.

    Args:
        registry_path:  Model registry root path.
        model_name:     Logical model name in the registry.
        mode:           ``"full"`` or ``"ci"``; CI mode reduces step counts.
        skip_field_test: Skip the field-test stage.
        skip_pilot:     Skip the pilot-monitor stage.
        output_audit:   Optional path to write the audit JSON.
        tmp_dir:        Scratch directory for intermediate files.
        verbose:        Print stage progress.

    Returns:
        Audit record dictionary.
    """
    os.makedirs(tmp_dir, exist_ok=True)

    # Reduce step counts in CI mode for fast feedback
    is_ci = mode == "ci"
    field_test_steps = 10 if is_ci else 50
    pilot_steps = 30 if is_ci else 200
    n_episodes = 10 if is_ci else 50
    rounds = 2 if is_ci else 5

    audit: Dict[str, Any] = {
        "pipeline_mode": mode,
        "model_name": model_name,
        "registry_path": registry_path,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "stages": {},
        "success": False,
    }

    try:
        # ----- Stage 1: Field test ----------------------------------------
        ft_json = os.path.join(tmp_dir, "field_test_report.json")
        cal_json = os.path.join(tmp_dir, "calibration.json")

        if not skip_field_test:
            ft_report = _stage_field_test(ft_json, n_steps=field_test_steps, verbose=verbose)
            audit["stages"]["field_test"] = {
                "all_passed": bool(ft_report.get("all_passed")),
                "pass_count": ft_report.get("pass_count", 0),
                "fail_count": ft_report.get("fail_count", 0),
            }
        else:
            if verbose:
                print("[mlops] Skipping field test stage.")
            ft_json = None

        # ----- Stage 2: Analyse -------------------------------------------
        calibration: Dict[str, Any] = {}
        if ft_json and os.path.isfile(ft_json):
            calibration = _stage_analyse(ft_json, cal_json, verbose=verbose)
            audit["stages"]["analysis"] = {
                "snr_threshold_db": calibration.get("snr_threshold_db"),
                "max_failures": calibration.get("max_failures"),
                "steering_gain": calibration.get("steering_gain"),
                "all_passed": calibration.get("all_passed"),
            }

        # ----- Stage 3: Pilot monitor -------------------------------------
        pilot_json = os.path.join(tmp_dir, "pilot_report.json")
        if not skip_pilot:
            pilot_report = _stage_pilot(
                calibration, pilot_json, pilot_steps=pilot_steps, verbose=verbose,
            )
            audit["stages"]["pilot_monitor"] = {
                "completed_steps": pilot_report.get("completed_steps"),
                "outage_rate": pilot_report.get("outage_rate"),
                "fallback_rate": pilot_report.get("fallback_rate"),
                "latency_p95_ms": pilot_report.get("latency_stats", {}).get("p95", 0.0),
            }
        else:
            if verbose:
                print("[mlops] Skipping pilot monitor stage.")

        # ----- Stage 4: Retrain -------------------------------------------
        retrain_summary = _stage_retrain(
            registry_path=registry_path,
            model_name=model_name,
            n_episodes=n_episodes,
            rounds=rounds,
            verbose=verbose,
        )
        audit["stages"]["retrain"] = {
            "promoted": bool(retrain_summary.get("promoted", False)),
            "candidate_metrics": retrain_summary.get("candidate_metrics"),
            "incumbent_metrics": retrain_summary.get("incumbent_metrics"),
        }

        audit["success"] = True

    except Exception as exc:  # noqa: BLE001
        audit["error"] = str(exc)
        audit["success"] = False
        print(f"[mlops] Pipeline ERROR: {exc}", file=sys.stderr)

    finally:
        audit["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    # Write audit log
    if output_audit:
        os.makedirs(os.path.dirname(output_audit) or ".", exist_ok=True)
        with open(output_audit, "w", encoding="utf-8") as f:
            json.dump(audit, f, indent=2)
        if verbose:
            print(f"[mlops] Audit log saved to {output_audit}")

    return audit


def _print_audit(audit: Dict[str, Any]) -> None:
    """Print a concise pipeline audit summary."""
    ok = "✅ SUCCESS" if audit.get("success") else "❌ FAILED"
    print()
    print("# MLOps Pipeline Audit")
    print()
    print(f"Status  : {ok}")
    print(f"Model   : {audit.get('model_name')}")
    print(f"Started : {audit.get('started_at')}")
    print(f"Finished: {audit.get('finished_at')}")
    print()
    for stage_name, stage_data in audit.get("stages", {}).items():
        print(f"  [{stage_name}]")
        for k, v in stage_data.items():
            print(f"    {k}: {v}")
    if "error" in audit:
        print(f"\n  ERROR: {audit['error']}")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="MLOps pipeline – field test → analyse → pilot → retrain."
    )
    p.add_argument("--registry",
                   default=os.environ.get("MODEL_REGISTRY_PATH", "/tmp/models"),
                   help="Model registry root path.")
    p.add_argument("--model-name",
                   default=os.environ.get("MODEL_NAME", "ppo_amazon"),
                   help="Logical model name.")
    p.add_argument("--mode", choices=["full", "ci"], default="full",
                   help="Pipeline mode: 'ci' reduces step counts for fast feedback.")
    p.add_argument("--skip-field-test",
                   action="store_true",
                   default=bool(int(os.environ.get("MLOPS_SKIP_FIELD_TEST", "0"))),
                   help="Skip field-test stage.")
    p.add_argument("--skip-pilot",
                   action="store_true",
                   default=bool(int(os.environ.get("MLOPS_SKIP_PILOT", "0"))),
                   help="Skip pilot-monitor stage.")
    p.add_argument("--output-audit",
                   default=os.environ.get("MLOPS_AUDIT_LOG"),
                   help="Path to write audit log JSON.")
    p.add_argument("--tmp-dir", default="/tmp/mlops_pipeline",
                   help="Scratch directory for intermediate files.")
    p.add_argument("--verbose", action="store_true",
                   help="Print stage progress.")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    audit = run_pipeline(
        registry_path=args.registry,
        model_name=args.model_name,
        mode=args.mode,
        skip_field_test=args.skip_field_test,
        skip_pilot=args.skip_pilot,
        output_audit=args.output_audit,
        tmp_dir=args.tmp_dir,
        verbose=args.verbose,
    )
    _print_audit(audit)
    return 0 if audit.get("success") else 1


if __name__ == "__main__":
    sys.exit(main())
