#!/usr/bin/env python3
"""
Final Executive Report Generator
==================================

Consolidates all project deliverables into a single executive report:

1. Field-test validation (from ``generate_validation_report.py``)
2. Regulatory compliance (from ``generate_compliance_report.py``)
3. Pilot deployment performance (from ``pilot_monitor.py``)
4. Production acceptance test (from ``acceptance_test.py``)
5. MLOps pipeline audit (from ``mlops_pipeline.py``)

The final report is written as:
    - A JSON document for machine processing and archival.
    - A Markdown document suitable for executive presentation.

All input files are optional; missing files produce partial reports.

Usage::

    python scripts/generate_final_report.py \\
        --validation   /tmp/validation_report.json \\
        --compliance   /tmp/compliance_report.json \\
        --pilot        /tmp/pilot_report.json \\
        --acceptance   /tmp/acceptance_report.json \\
        --mlops-audit  /tmp/mlops_audit.json \\
        --output-json  /tmp/final_report.json \\
        --output-md    /tmp/final_report.md \\
        --verbose

Exit codes:
    0 – report generated; all available checks passed.
    2 – one or more sections report failures.
    1 – input / output error.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load(path: Optional[str]) -> Optional[Dict[str, Any]]:
    if path and os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return None


def _section_status(data: Optional[Dict[str, Any]], key: str = "all_passed") -> str:
    if data is None:
        return "N/A"
    val = data.get(key)
    if val is True:
        return "PASS"
    if val is False:
        return "FAIL"
    return "N/A"


def _overall_icon(status: str) -> str:
    return {"PASS": "✅", "FAIL": "❌", "N/A": "—"}.get(status, status)


# ---------------------------------------------------------------------------
# Report builder
# ---------------------------------------------------------------------------

def build_final_report(
    validation: Optional[Dict[str, Any]],
    compliance: Optional[Dict[str, Any]],
    pilot: Optional[Dict[str, Any]],
    acceptance: Optional[Dict[str, Any]],
    mlops_audit: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Build the consolidated final executive report.

    Args:
        validation:   Technical validation report dict.
        compliance:   Regulatory compliance report dict.
        pilot:        Pilot monitor report dict.
        acceptance:   Acceptance test report dict.
        mlops_audit:  MLOps pipeline audit dict.

    Returns:
        Final report dictionary.
    """
    # Per-section status
    val_status = _section_status(validation)
    comp_status = "PASS" if (compliance and compliance.get("overall_compliant")) else (
        "FAIL" if compliance else "N/A"
    )
    pilot_status = _section_status(pilot, key="outage_rate")
    if pilot_status == "N/A" and pilot:
        pilot_status = "PASS" if pilot.get("outage_rate", 1.0) <= 0.01 else "FAIL"
    acc_status = _section_status(acceptance)
    mlops_status = "PASS" if (mlops_audit and mlops_audit.get("success")) else (
        "FAIL" if mlops_audit else "N/A"
    )

    # Determine overall status
    statuses = [val_status, comp_status, pilot_status, acc_status, mlops_status]
    has_fail = any(s == "FAIL" for s in statuses)
    has_data = any(s != "N/A" for s in statuses)
    overall = "FAIL" if has_fail else ("PASS" if has_data else "N/A")

    sections: List[Dict[str, Any]] = [
        {
            "title": "Technical Validation",
            "status": val_status,
            "summary": _validation_summary(validation),
        },
        {
            "title": "Regulatory Compliance (Anatel/FCC)",
            "status": comp_status,
            "summary": _compliance_summary(compliance),
        },
        {
            "title": "Pilot Deployment Performance",
            "status": pilot_status,
            "summary": _pilot_summary(pilot),
        },
        {
            "title": "Production Acceptance Test",
            "status": acc_status,
            "summary": _acceptance_summary(acceptance),
        },
        {
            "title": "MLOps Pipeline",
            "status": mlops_status,
            "summary": _mlops_summary(mlops_audit),
        },
    ]

    return {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "project": "AI-Driven Beamforming for Starlink in the Amazon",
        "overall_status": overall,
        "sections": sections,
    }


def _validation_summary(d: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not d:
        return {}
    return {
        "field_test_passed": d.get("field_test_passed"),
        "checks_pass": d.get("pass_count", 0),
        "checks_fail": d.get("fail_count", 0),
    }


def _compliance_summary(d: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not d:
        return {}
    stats = d.get("statistics", {})
    return {
        "overall_compliant": d.get("overall_compliant"),
        "total_violations": stats.get("total_violations", 0),
        "n_constraints": len(d.get("constraints", [])),
        "monte_carlo_enforcement_rate": d.get("monte_carlo_validation", {}).get(
            "enforcement_rate"
        ),
    }


def _pilot_summary(d: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not d:
        return {}
    return {
        "completed_steps": d.get("completed_steps"),
        "outage_rate": d.get("outage_rate"),
        "fallback_rate": d.get("fallback_rate"),
        "latency_p95_ms": d.get("latency_stats", {}).get("p95"),
    }


def _acceptance_summary(d: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not d:
        return {}
    return {
        "all_passed": d.get("all_passed"),
        "checks_fail": d.get("fail_count", 0),
        "outage_rate": d.get("outage_rate"),
        "latency_p95_ms": d.get("latency_stats", {}).get("p95"),
    }


def _mlops_summary(d: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not d:
        return {}
    stages = d.get("stages", {})
    return {
        "success": d.get("success"),
        "model_promoted": stages.get("retrain", {}).get("promoted"),
        "pilot_steps": stages.get("pilot_monitor", {}).get("completed_steps"),
    }


# ---------------------------------------------------------------------------
# Markdown renderer
# ---------------------------------------------------------------------------

def render_final_markdown(report: Dict[str, Any]) -> str:
    overall = report["overall_status"]
    icon = _overall_icon(overall)
    lines = [
        f"# {report['project']}",
        "## Final Executive Report",
        "",
        f"Generated  : {report['generated_at']}",
        f"**Overall  : {icon} {overall}**",
        "",
        "---",
        "",
        "## Section Summary",
        "",
        "| Section | Status | Key Metrics |",
        "|---|---|---|",
    ]

    for sec in report["sections"]:
        st = sec["status"]
        ico = _overall_icon(st)
        metrics = "; ".join(
            f"{k}={v}" for k, v in sec["summary"].items() if v is not None
        )
        lines.append(f"| {sec['title']} | {ico} {st} | {metrics} |")

    lines += ["", "---", ""]
    for sec in report["sections"]:
        st = sec["status"]
        ico = _overall_icon(st)
        lines += [
            f"## {sec['title']}",
            "",
            f"**Status: {ico} {st}**",
            "",
        ]
        if sec["summary"]:
            for k, v in sec["summary"].items():
                lines.append(f"- **{k}**: {v}")
        lines.append("")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Generate consolidated final executive report."
    )
    p.add_argument("--validation", default=None,
                   help="Path to generate_validation_report.py JSON output.")
    p.add_argument("--compliance", default=None,
                   help="Path to generate_compliance_report.py JSON output.")
    p.add_argument("--pilot", default=None,
                   help="Path to pilot_monitor.py JSON output.")
    p.add_argument("--acceptance", default=None,
                   help="Path to acceptance_test.py JSON output.")
    p.add_argument("--mlops-audit", default=None,
                   help="Path to mlops_pipeline.py audit JSON output.")
    p.add_argument("--output-json", default=None,
                   help="Output path for final report JSON.")
    p.add_argument("--output-md", default=None,
                   help="Output path for final report Markdown.")
    p.add_argument("--verbose", action="store_true",
                   help="Print Markdown report to stdout.")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)

    validation = _load(args.validation)
    compliance = _load(args.compliance)
    pilot = _load(args.pilot)
    acceptance = _load(args.acceptance)
    mlops_audit = _load(args.mlops_audit)

    report = build_final_report(validation, compliance, pilot, acceptance, mlops_audit)
    md = render_final_markdown(report)

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

    return 2 if report["overall_status"] == "FAIL" else 0


if __name__ == "__main__":
    sys.exit(main())
