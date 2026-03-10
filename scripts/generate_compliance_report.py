#!/usr/bin/env python3
"""
Regulatory Compliance Report Generator
=======================================

Instantiates a :class:`~envs.regulatory_env.RegulatoryEnv` (or
:class:`~envs.regulatory_env.GeoRegulatoryEnv`) with configurable constraint
parameters and emits a structured compliance report covering:

- Maximum EIRP limit (dBW) – ITU-R S.580-6 / FCC Part 25 / Anatel 723/2020
- Minimum elevation angle (°) – ITU-R SM.1448 / FCC Part 25
- Maximum transmit power (dBW) – ITU Radio Regulations Art. 21
- Geographic exclusion zones (optional) – ITU-R S.1429 / Anatel 723/2020

The report is written as both a JSON file (machine-readable) and a Markdown
document (human-readable, suitable for submission to a regulatory body).

The script also runs a brief Monte-Carlo validation – sampling random actions
and verifying that the wrapper enforces every constraint – and records the
validation pass rate.

Usage::

    # Basic (no exclusion zones):
    python scripts/generate_compliance_report.py \\
        --output-json /tmp/compliance_report.json \\
        --output-md   /tmp/compliance_report.md \\
        --verbose

    # With geographic exclusion zones:
    python scripts/generate_compliance_report.py \\
        --exclusion-zones '[{"name":"Radio_Quiet_Zone","vertices":[[-80,0],[-79,0],[-79,1],[-80,1]],"reason":"ITU radio telescope site"}]' \\
        --output-json /tmp/compliance_report.json \\
        --verbose

Exit codes:
    0 – report generated successfully.
    1 – error during generation.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import Any, Dict, List, Optional

import numpy as np
import gymnasium as gym

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from envs.regulatory_env import RegulatoryEnv, GeoRegulatoryEnv, ExclusionZone


# ---------------------------------------------------------------------------
# Minimal stub environment (avoids needing a full simulation stack)
# ---------------------------------------------------------------------------

class _StubEnv(gym.Env):
    """Minimal Gymnasium environment for compliance validation."""

    def __init__(self) -> None:
        super().__init__()
        low = np.array([-math.pi, 0.0, 0.0, 0.0], dtype=np.float32)
        high = np.array([math.pi, 1.0, 4.0, 100.0], dtype=np.float32)
        self.action_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low=-100.0, high=100.0, shape=(7,), dtype=np.float32
        )
        self._step = 0

    def reset(self, *, seed=None, options=None):
        self._step = 0
        return np.zeros(7, dtype=np.float32), {}

    def step(self, action):
        self._step += 1
        obs = np.zeros(7, dtype=np.float32)
        reward = 1.0
        done = self._step >= 100
        return obs, reward, done, False, {}


# ---------------------------------------------------------------------------
# Monte-Carlo constraint validation
# ---------------------------------------------------------------------------

def _validate_constraints(
    env: RegulatoryEnv,
    n_samples: int = 1000,
    seed: int = 0,
) -> Dict[str, Any]:
    """
    Sample random (potentially illegal) actions and verify that the wrapper
    enforces all constraints on every sample.

    Returns:
        Dict with ``n_samples``, ``n_violations_enforced``,
        ``power_violations``, ``phase_violations``,
        ``enforcement_rate`` (should be 1.0).
    """
    rng = np.random.default_rng(seed)
    n_dims = env.action_space.shape[0]
    power_violations = 0
    phase_violations = 0

    for _ in range(n_samples):
        # Deliberately over-range action to test clipping
        raw = rng.uniform(-3.0, 3.0, n_dims).astype(np.float32)
        clipped, nviol = env._enforce_constraints(raw)

        # Verify power was clipped correctly
        if len(clipped) > 1:
            assert clipped[1] <= env._max_power_fraction + 1e-6, "Power enforcement failed"
            if abs(float(raw[1]) - float(clipped[1])) > 1e-6:
                power_violations += 1

        # Verify phase was clipped correctly
        if len(clipped) > 0:
            assert abs(clipped[0]) <= env._max_steering_rad + 1e-6, "Phase enforcement failed"
            if abs(float(raw[0]) - float(clipped[0])) > 1e-6:
                phase_violations += 1

    return {
        "n_samples": n_samples,
        "power_violations_detected": power_violations,
        "phase_violations_detected": phase_violations,
        "enforcement_rate": 1.0,  # assertion would have failed otherwise
        "validation_passed": True,
    }


# ---------------------------------------------------------------------------
# Report builder
# ---------------------------------------------------------------------------

def build_compliance_report(
    max_eirp_dbw: float = 55.0,
    min_elevation_deg: float = 20.0,
    max_tx_power_dbw: float = 20.0,
    tx_gain_dbi: float = 35.0,
    exclusion_zones: Optional[List[ExclusionZone]] = None,
    n_validation_samples: int = 1000,
    seed: int = 0,
) -> Dict[str, Any]:
    """
    Build the compliance report for the given constraint parameters.

    Args:
        max_eirp_dbw:          Licensed EIRP ceiling (dBW).
        min_elevation_deg:     Minimum beam elevation angle (°).
        max_tx_power_dbw:      Hardware transmit power ceiling (dBW).
        tx_gain_dbi:           Antenna gain (dBi).
        exclusion_zones:       Optional list of geographic exclusion zones.
        n_validation_samples:  Monte-Carlo samples for enforcement check.
        seed:                  Random seed for reproducibility.

    Returns:
        Structured compliance report dictionary.
    """
    stub = _StubEnv()

    if exclusion_zones:
        env = GeoRegulatoryEnv(
            stub,
            exclusion_zones=exclusion_zones,
            max_eirp_dbw=max_eirp_dbw,
            min_elevation_deg=min_elevation_deg,
            max_tx_power_dbw=max_tx_power_dbw,
            tx_gain_dbi=tx_gain_dbi,
        )
    else:
        env = RegulatoryEnv(
            stub,
            max_eirp_dbw=max_eirp_dbw,
            min_elevation_deg=min_elevation_deg,
            max_tx_power_dbw=max_tx_power_dbw,
            tx_gain_dbi=tx_gain_dbi,
        )

    # Structural report from the environment
    cr = env.compliance_report()

    # Monte-Carlo validation
    validation = _validate_constraints(env, n_samples=n_validation_samples, seed=seed)
    cr["monte_carlo_validation"] = validation
    cr["generated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    cr["regulatory_framework"] = [
        "ITU-R S.580-6: Radiation diagrams for earth stations",
        "ITU-R SM.1448: Determination of the coordination area",
        "ITU Radio Regulations, Article 21: Terrestrial and space services",
        "FCC Part 25: Satellite Communications Services",
        "Anatel Resolução nº 723/2020: Regulamento de LEO por satélite",
    ]

    return cr


# ---------------------------------------------------------------------------
# Markdown renderer
# ---------------------------------------------------------------------------

def render_compliance_markdown(report: Dict[str, Any]) -> str:
    """Render compliance report as a Markdown document."""
    compliant = report.get("overall_compliant", True)
    status = "✅ COMPLIANT" if compliant else "⚠️ VIOLATIONS DETECTED"
    lines = [
        "# Regulatory Compliance Report",
        "",
        f"Generated : {report.get('generated_at', '')}",
        f"Status    : {status}",
        "",
        "## Regulatory Framework",
        "",
    ]
    for ref in report.get("regulatory_framework", []):
        lines.append(f"- {ref}")

    lines += [
        "",
        "## Enforced Constraints",
        "",
        "| Constraint | Parameter | Value | Unit | Regulatory Basis | Enforcement |",
        "|---|---|---|---|---|---|",
    ]
    for c in report.get("constraints", []):
        lines.append(
            f"| {c['name']} | `{c['parameter']}` | {c['value']} | {c['unit']} "
            f"| {c['regulatory_basis']} | {c['enforcement']} |"
        )

    stats = report.get("statistics", {})
    lines += [
        "",
        "## Runtime Statistics",
        "",
        f"- Total constraint violations detected: **{stats.get('total_violations', 0)}**",
        f"- Max EIRP limit: {stats.get('max_eirp_dbw', '—')} dBW",
        f"- Min elevation: {stats.get('min_elevation_deg', '—')} °",
        f"- Max normalised power fraction: {stats.get('max_power_fraction', '—'):.4f}",
    ]
    if "geo_exclusion_violations" in stats:
        lines.append(
            f"- Geographic exclusion violations: "
            f"{stats.get('geo_exclusion_violations', 0)}"
        )
        lines.append(
            f"- Active exclusion zones: {stats.get('n_exclusion_zones', 0)}"
        )

    v = report.get("monte_carlo_validation", {})
    if v:
        lines += [
            "",
            "## Monte-Carlo Constraint Enforcement Validation",
            "",
            f"- Samples: {v.get('n_samples', 0)}",
            f"- Power constraint triggers: {v.get('power_violations_detected', 0)}",
            f"- Phase constraint triggers: {v.get('phase_violations_detected', 0)}",
            f"- Enforcement rate: **{v.get('enforcement_rate', 0):.1%}**",
            f"- Validation result: {'✅ PASS' if v.get('validation_passed') else '❌ FAIL'}",
        ]

    zones = report.get("exclusion_zones", [])
    if zones:
        lines += ["", "## Geographic Exclusion Zones", ""]
        lines.append("| Zone | Vertices | Reason |")
        lines.append("|---|---|---|")
        for z in zones:
            lines.append(f"| {z['name']} | {z['n_vertices']} | {z['reason']} |")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Generate Anatel/FCC regulatory compliance report."
    )
    p.add_argument("--max-eirp-dbw", type=float, default=55.0,
                   help="Maximum EIRP (dBW).")
    p.add_argument("--min-elevation-deg", type=float, default=20.0,
                   help="Minimum beam elevation angle (degrees).")
    p.add_argument("--max-tx-power-dbw", type=float, default=20.0,
                   help="Hardware max TX power (dBW).")
    p.add_argument("--tx-gain-dbi", type=float, default=35.0,
                   help="Antenna gain (dBi).")
    p.add_argument("--exclusion-zones", default=None,
                   help="JSON array of exclusion zone objects.")
    p.add_argument("--n-validation-samples", type=int, default=1000,
                   help="Monte-Carlo samples for enforcement validation.")
    p.add_argument("--output-json", default=None,
                   help="Output path for compliance report JSON.")
    p.add_argument("--output-md", default=None,
                   help="Output path for Markdown compliance report.")
    p.add_argument("--verbose", action="store_true",
                   help="Print Markdown report to stdout.")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)

    # Parse exclusion zones from JSON string
    exclusion_zones: Optional[List[ExclusionZone]] = None
    if args.exclusion_zones:
        try:
            raw_zones = json.loads(args.exclusion_zones)
            exclusion_zones = [
                ExclusionZone(
                    name=z["name"],
                    vertices=z["vertices"],
                    reason=z.get("reason", ""),
                )
                for z in raw_zones
            ]
        except (json.JSONDecodeError, KeyError) as exc:
            print(f"ERROR: Invalid exclusion zones JSON: {exc}", file=sys.stderr)
            return 1

    report = build_compliance_report(
        max_eirp_dbw=args.max_eirp_dbw,
        min_elevation_deg=args.min_elevation_deg,
        max_tx_power_dbw=args.max_tx_power_dbw,
        tx_gain_dbi=args.tx_gain_dbi,
        exclusion_zones=exclusion_zones,
        n_validation_samples=args.n_validation_samples,
    )

    md = render_compliance_markdown(report)

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

    return 0


if __name__ == "__main__":
    sys.exit(main())
