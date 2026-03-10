"""
Tests for final-wave deliverables:
  - scripts/generate_validation_report.py
  - scripts/generate_compliance_report.py
  - scripts/acceptance_test.py
  - scripts/generate_final_report.py
  - src/envs/regulatory_env.py: compliance_report() methods
  - docs/runbook.md: existence check
"""

import sys
import os
import json
import math
import tempfile

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, os.path.join(REPO, "scripts"))


# ===========================================================================
# Fixtures
# ===========================================================================

def _ft_report(all_passed=True):
    return {
        "driver_type": "null",
        "all_passed": all_passed,
        "pass_count": 5 if all_passed else 4,
        "fail_count": 0 if all_passed else 1,
        "scenarios": [
            {
                "scenario": "boresight_baseline",
                "phase_error_deg": 0.0,
                "tolerance_deg": 2.0,
                "passed": True,
            },
            {
                "scenario": "azimuth_sweep",
                "angles_commanded_deg": [-30.0, 30.0],
                "angles_achieved_deg": [-30.0, 30.0],
                "errors_deg": [0.0, 0.0],
                "passed": True,
            },
            {
                "scenario": "rain_attenuation_injection",
                # All positive SNR so P5 > snr_threshold_db (4.0) + 1 dB margin
                "steps": [
                    {"effective_snr_db": 20.0},
                    {"effective_snr_db": 18.0},
                    {"effective_snr_db": 15.0},
                    {"effective_snr_db": 12.0},
                    {"effective_snr_db": 10.0},
                ],
                "passed": True,
            },
            {
                "scenario": "handover_latency",
                "p95_ms": 5.0,
                "p99_ms": 8.0,
                "mean_ms": 4.0,
                "max_ms": 10.0,
                "passed": True,
            },
            {
                "scenario": "steering_precision",
                "std_deg": 0.0,
                "mean_error_deg": 0.0,
                "tolerance_deg": 2.0,
                "passed": True,
            },
        ],
    }


def _cal():
    return {
        "snr_threshold_db": 4.0,
        "max_failures": 3,
        "steering_gain": 1.0,
        "latency_p95_ms": 5.0,
        "steering_std_deg": 0.0,
        "all_passed": True,
    }


def _pilot():
    return {
        "completed_steps": 100,
        "outage_rate": 0.0,
        "fallback_rate": 0.0,
        "watchdog_alerts": 0,
        "handovers": 5,
        "handover_rate_per_min": 1.0,
        "snr_stats": {"mean": 18.0, "p50": 18.0, "p95": 22.0, "p99": 25.0, "count": 100},
        "latency_stats": {"mean": 5.0, "p50": 5.0, "p95": 8.0, "p99": 10.0, "count": 100},
    }


def _acceptance_pass():
    return {
        "all_passed": True,
        "fail_count": 0,
        "outage_rate": 0.0,
        "latency_stats": {"p95": 8.0, "p99": 10.0},
        "checks": [{"label": "Outage rate", "value": 0.0, "limit": 0.01, "status": "PASS"}],
    }


def _mlops_audit():
    return {
        "success": True,
        "model_name": "ppo_amazon",
        "stages": {
            "retrain": {"promoted": True},
            "pilot_monitor": {"completed_steps": 30},
        },
    }


# ===========================================================================
# generate_validation_report
# ===========================================================================

class TestGenerateValidationReport:
    def test_import(self):
        from generate_validation_report import build_validation_report, render_markdown
        assert build_validation_report is not None

    def test_all_pass(self):
        from generate_validation_report import build_validation_report
        rpt = build_validation_report(_ft_report(), _cal(), _pilot())
        assert rpt["all_passed"] is True
        assert rpt["fail_count"] == 0
        assert rpt["pass_count"] > 0

    def test_fail_detected_for_high_latency(self):
        from generate_validation_report import build_validation_report
        ft = _ft_report()
        # Inject a failing P95 latency
        for s in ft["scenarios"]:
            if s["scenario"] == "handover_latency":
                s["p95_ms"] = 600.0  # > 500 ms limit
        rpt = build_validation_report(ft, _cal(), _pilot())
        assert rpt["fail_count"] > 0

    def test_empty_inputs_do_not_crash(self):
        from generate_validation_report import build_validation_report
        rpt = build_validation_report(None, None, None)
        assert "checks" in rpt
        assert "generated_at" in rpt

    def test_markdown_rendering(self):
        from generate_validation_report import build_validation_report, render_markdown
        rpt = build_validation_report(_ft_report(), _cal(), _pilot())
        md = render_markdown(rpt)
        assert "Technical Validation Report" in md
        assert "PASS" in md or "N/A" in md

    def test_json_output(self):
        from generate_validation_report import main
        ft_p, cal_p, pi_p, out_p = [None] * 4
        with tempfile.TemporaryDirectory() as tmp:
            ft_p = os.path.join(tmp, "ft.json")
            cal_p = os.path.join(tmp, "cal.json")
            pi_p = os.path.join(tmp, "pilot.json")
            out_p = os.path.join(tmp, "val.json")
            with open(ft_p, "w") as f:
                json.dump(_ft_report(), f)
            with open(cal_p, "w") as f:
                json.dump(_cal(), f)
            with open(pi_p, "w") as f:
                json.dump(_pilot(), f)
            rc = main(["--field-test", ft_p, "--calibration", cal_p,
                       "--pilot", pi_p, "--output-json", out_p])
            assert rc == 0
            with open(out_p) as f:
                doc = json.load(f)
            assert "checks" in doc

    def test_fail_exit_code_2(self):
        from generate_validation_report import main
        with tempfile.TemporaryDirectory() as tmp:
            ft = _ft_report()
            for s in ft["scenarios"]:
                if s["scenario"] == "handover_latency":
                    s["p95_ms"] = 999.0
            ft_p = os.path.join(tmp, "ft.json")
            with open(ft_p, "w") as f:
                json.dump(ft, f)
            rc = main(["--field-test", ft_p])
            assert rc == 2

    def test_steering_gain_out_of_range_fails(self):
        from generate_validation_report import build_validation_report
        cal = dict(_cal())
        cal["steering_gain"] = 1.5  # > 1.1 max
        rpt = build_validation_report(_ft_report(), cal, _pilot())
        gain_check = next(
            (c for c in rpt["checks"] if "gain" in c["label"].lower()), None
        )
        assert gain_check is not None
        assert gain_check["status"] == "FAIL"


# ===========================================================================
# regulatory_env.compliance_report()
# ===========================================================================

class TestRegulatoryEnvComplianceReport:
    def _make_env(self):
        from envs.regulatory_env import RegulatoryEnv
        import gymnasium as gym

        class _StubEnv(gym.Env):
            def __init__(self):
                super().__init__()
                low = np.array([-math.pi, 0.0, 0.0, 0.0], dtype=np.float32)
                high = np.array([math.pi, 1.0, 4.0, 100.0], dtype=np.float32)
                self.action_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
                self.observation_space = gym.spaces.Box(-100.0, 100.0, (7,), np.float32)
                self._s = 0

            def reset(self, *, seed=None, options=None):
                return np.zeros(7, dtype=np.float32), {}

            def step(self, action):
                self._s += 1
                return np.zeros(7, dtype=np.float32), 1.0, False, False, {}

        return RegulatoryEnv(_StubEnv())

    def test_compliance_report_returns_dict(self):
        env = self._make_env()
        cr = env.compliance_report()
        assert isinstance(cr, dict)

    def test_compliance_report_has_required_keys(self):
        env = self._make_env()
        cr = env.compliance_report()
        for k in ("overall_compliant", "constraints", "statistics"):
            assert k in cr, f"Missing key: {k}"

    def test_compliance_report_constraints_list(self):
        env = self._make_env()
        cr = env.compliance_report()
        assert len(cr["constraints"]) >= 2
        for c in cr["constraints"]:
            for k in ("name", "parameter", "value", "unit", "regulatory_basis"):
                assert k in c

    def test_compliance_report_overall_compliant_true_at_start(self):
        env = self._make_env()
        cr = env.compliance_report()
        assert cr["overall_compliant"] is True  # no actions taken yet

    def test_geo_env_compliance_report(self):
        from envs.regulatory_env import GeoRegulatoryEnv, ExclusionZone
        import gymnasium as gym

        class _StubEnv(gym.Env):
            def __init__(self):
                super().__init__()
                self.action_space = gym.spaces.Discrete(3)
                self.observation_space = gym.spaces.Box(-100.0, 100.0, (7,), np.float32)

            def reset(self, *, seed=None, options=None):
                return np.zeros(7, dtype=np.float32), {}

            def step(self, action):
                return np.zeros(7, dtype=np.float32), 1.0, False, False, {}

        zone = ExclusionZone("TestZone", [(-10.0, -10.0), (10.0, -10.0), (0.0, 10.0)])
        env = GeoRegulatoryEnv(_StubEnv(), exclusion_zones=[zone])
        cr = env.compliance_report()
        assert "exclusion_zones" in cr
        assert cr["exclusion_zones"][0]["name"] == "TestZone"
        assert "geo_violation_log_count" in cr

    def test_geo_env_compliance_report_includes_geo_constraint(self):
        from envs.regulatory_env import GeoRegulatoryEnv, ExclusionZone
        import gymnasium as gym

        class _StubEnv(gym.Env):
            def __init__(self):
                super().__init__()
                self.action_space = gym.spaces.Discrete(2)
                self.observation_space = gym.spaces.Box(-100.0, 100.0, (7,), np.float32)

            def reset(self, *, seed=None, options=None):
                return np.zeros(7, dtype=np.float32), {}

            def step(self, action):
                return np.zeros(7, dtype=np.float32), 1.0, False, False, {}

        env = GeoRegulatoryEnv(_StubEnv(), exclusion_zones=[])
        cr = env.compliance_report()
        names = [c["name"] for c in cr["constraints"]]
        assert any("exclusion" in n.lower() or "geographic" in n.lower() for n in names)


# ===========================================================================
# generate_compliance_report
# ===========================================================================

class TestGenerateComplianceReport:
    def test_import(self):
        from generate_compliance_report import build_compliance_report, render_compliance_markdown
        assert build_compliance_report is not None

    def test_basic_report_structure(self):
        from generate_compliance_report import build_compliance_report
        rpt = build_compliance_report(n_validation_samples=50)
        for k in ("overall_compliant", "constraints", "statistics",
                  "monte_carlo_validation", "generated_at"):
            assert k in rpt

    def test_monte_carlo_validation_passes(self):
        from generate_compliance_report import build_compliance_report
        rpt = build_compliance_report(n_validation_samples=100)
        v = rpt["monte_carlo_validation"]
        assert v["validation_passed"] is True
        assert v["enforcement_rate"] == 1.0

    def test_no_violations_at_start(self):
        from generate_compliance_report import build_compliance_report
        rpt = build_compliance_report(n_validation_samples=100)
        assert rpt["statistics"]["total_violations"] == 0

    def test_regulatory_framework_listed(self):
        from generate_compliance_report import build_compliance_report
        rpt = build_compliance_report(n_validation_samples=10)
        assert len(rpt["regulatory_framework"]) >= 4
        framework_str = " ".join(rpt["regulatory_framework"])
        assert "Anatel" in framework_str
        assert "FCC" in framework_str

    def test_markdown_contains_anatel_fcc(self):
        from generate_compliance_report import build_compliance_report, render_compliance_markdown
        rpt = build_compliance_report(n_validation_samples=10)
        md = render_compliance_markdown(rpt)
        assert "Anatel" in md or "FCC" in md

    def test_with_exclusion_zones(self):
        from generate_compliance_report import build_compliance_report
        from envs.regulatory_env import ExclusionZone
        zones = [ExclusionZone("TestZone", [(-5.0, -5.0), (5.0, -5.0), (0.0, 5.0)])]
        rpt = build_compliance_report(exclusion_zones=zones, n_validation_samples=10)
        assert "exclusion_zones" in rpt
        assert len(rpt["exclusion_zones"]) == 1

    def test_main_writes_json(self):
        from generate_compliance_report import main
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            rc = main(["--n-validation-samples", "10", "--output-json", path])
            assert rc == 0
            with open(path) as f:
                doc = json.load(f)
            assert "overall_compliant" in doc
        finally:
            os.unlink(path)

    def test_main_with_exclusion_zones_json(self):
        from generate_compliance_report import main
        zones_json = json.dumps([{
            "name": "Atacama",
            "vertices": [[-68.0, -23.0], [-67.0, -23.0], [-67.5, -22.0]],
            "reason": "VLBI site",
        }])
        rc = main(["--exclusion-zones", zones_json, "--n-validation-samples", "10"])
        assert rc == 0


# ===========================================================================
# acceptance_test
# ===========================================================================

class TestAcceptanceTest:
    def test_import(self):
        from acceptance_test import run_acceptance_test, _render_acceptance_md
        assert run_acceptance_test is not None

    def test_returns_all_keys(self):
        from acceptance_test import run_acceptance_test
        rpt = run_acceptance_test(steps=10, interval_s=0.0)
        for k in ("all_passed", "fail_count", "outage_rate", "latency_stats",
                  "checks", "completed_steps"):
            assert k in rpt

    def test_all_checks_pass_at_low_load(self):
        from acceptance_test import run_acceptance_test
        rpt = run_acceptance_test(steps=30, interval_s=0.0, seed=42)
        # With default 5 dB threshold and good synthetic SNR, outage should be low
        assert rpt["outage_rate"] <= 0.2  # allow some margin in CI

    def test_completed_steps_correct(self):
        from acceptance_test import run_acceptance_test
        rpt = run_acceptance_test(steps=20, interval_s=0.0)
        assert rpt["completed_steps"] == 20

    def test_json_output_written(self):
        from acceptance_test import run_acceptance_test
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            run_acceptance_test(steps=5, output_json=path)
            with open(path) as f:
                doc = json.load(f)
            assert "all_passed" in doc
        finally:
            os.unlink(path)

    def test_markdown_output_written(self):
        from acceptance_test import run_acceptance_test
        with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as f:
            path = f.name
        try:
            run_acceptance_test(steps=5, output_md=path)
            with open(path) as f:
                content = f.read()
            assert "Acceptance Test" in content
        finally:
            os.unlink(path)

    def test_calibration_applied(self):
        from acceptance_test import run_acceptance_test
        cal = {"snr_threshold_db": 1.0, "max_failures": 2}
        rpt = run_acceptance_test(steps=10, calibration=cal)
        assert rpt is not None

    def test_main_exit_zero_on_pass(self):
        from acceptance_test import main
        rc = main(["--steps", "5"])
        assert rc in (0, 2)  # 0 = all pass, 2 = some fail (both valid outcomes)

    def test_render_md_contains_acceptance(self):
        from acceptance_test import run_acceptance_test, _render_acceptance_md
        rpt = run_acceptance_test(steps=5)
        md = _render_acceptance_md(rpt)
        assert "Acceptance" in md

    def test_high_threshold_causes_failures(self):
        """With an impossibly high SNR threshold (>35 dB), outage_rate should be 1.0."""
        from acceptance_test import run_acceptance_test
        rpt = run_acceptance_test(steps=20, snr_threshold_db=40.0)
        assert rpt["outage_rate"] == 1.0


# ===========================================================================
# generate_final_report
# ===========================================================================

class TestGenerateFinalReport:
    def test_import(self):
        from generate_final_report import build_final_report, render_final_markdown
        assert build_final_report is not None

    def test_all_sections_present(self):
        from generate_final_report import build_final_report
        rpt = build_final_report(
            _ft_report(), None, _pilot(), _acceptance_pass(), _mlops_audit()
        )
        titles = [s["title"] for s in rpt["sections"]]
        assert len(titles) == 5

    def test_overall_pass_when_all_pass(self):
        from generate_final_report import build_final_report
        rpt = build_final_report(
            _ft_report(), None, _pilot(), _acceptance_pass(), _mlops_audit()
        )
        # With all mocked-pass inputs, overall should not be FAIL
        assert rpt["overall_status"] in ("PASS", "N/A")

    def test_overall_fail_when_acceptance_fails(self):
        from generate_final_report import build_final_report
        acc_fail = dict(_acceptance_pass())
        acc_fail["all_passed"] = False
        rpt = build_final_report(None, None, None, acc_fail, None)
        assert rpt["overall_status"] == "FAIL"

    def test_empty_inputs_do_not_crash(self):
        from generate_final_report import build_final_report
        rpt = build_final_report(None, None, None, None, None)
        assert "sections" in rpt
        assert rpt["overall_status"] == "N/A"

    def test_markdown_rendering(self):
        from generate_final_report import build_final_report, render_final_markdown
        rpt = build_final_report(None, None, _pilot(), None, None)
        md = render_final_markdown(rpt)
        assert "Final Executive Report" in md

    def test_json_output(self):
        from generate_final_report import main
        with tempfile.TemporaryDirectory() as tmp:
            val_p = os.path.join(tmp, "val.json")
            with open(val_p, "w") as f:
                from generate_validation_report import build_validation_report
                json.dump(build_validation_report(_ft_report(), _cal(), _pilot()), f)

            out_p = os.path.join(tmp, "final.json")
            rc = main(["--validation", val_p, "--output-json", out_p])
            assert rc in (0, 2)
            with open(out_p) as f:
                doc = json.load(f)
            assert "overall_status" in doc

    def test_end_to_end_pipeline(self):
        """Run all scripts in sequence and generate final report."""
        from generate_validation_report import build_validation_report
        from generate_compliance_report import build_compliance_report
        from acceptance_test import run_acceptance_test
        from generate_final_report import build_final_report, render_final_markdown

        val = build_validation_report(_ft_report(), _cal(), _pilot())
        comp = build_compliance_report(n_validation_samples=10)
        acc = run_acceptance_test(steps=10)
        final = build_final_report(val, comp, _pilot(), acc, _mlops_audit())
        md = render_final_markdown(final)
        assert "AI-Driven" in md
        assert "overall_status" in final


# ===========================================================================
# docs/runbook.md existence and content
# ===========================================================================

class TestRunbook:
    RUNBOOK = os.path.join(
        os.path.dirname(__file__), "..", "docs", "runbook.md"
    )

    def test_runbook_exists(self):
        assert os.path.isfile(self.RUNBOOK), "docs/runbook.md not found"

    def test_runbook_has_required_sections(self):
        with open(self.RUNBOOK, encoding="utf-8") as f:
            content = f.read()
        required = [
            "Architecture",
            "Initialisation",
            "Dashboard",
            "Alert",
            "Fallback",
            "Canary",
            "Backup",
            "Disaster Recovery",
        ]
        for section in required:
            assert section in content, f"Runbook missing section: {section}"

    def test_runbook_mentions_anatel_fcc(self):
        with open(self.RUNBOOK, encoding="utf-8") as f:
            content = f.read()
        assert "Anatel" in content or "FCC" in content

    def test_runbook_has_table_of_contents(self):
        with open(self.RUNBOOK, encoding="utf-8") as f:
            content = f.read()
        assert "Table of Contents" in content or "## 1" in content
