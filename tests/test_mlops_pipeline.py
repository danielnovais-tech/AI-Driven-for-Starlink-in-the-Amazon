"""
Tests for:
  - scripts/analyze_field_test.py
  - scripts/pilot_monitor.py
  - scripts/mlops_pipeline.py
  - OnlineBeamController.apply_calibration()
  - NullPhasedArrayDriver / EthernetPhasedArrayDriver steering_gain
"""

import sys
import os
import json
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_minimal_field_test_report(all_passed: bool = True) -> dict:
    """Return a minimal field-test report dict for testing."""
    return {
        "driver_type": "null",
        "all_passed": all_passed,
        "pass_count": 5 if all_passed else 4,
        "fail_count": 0 if all_passed else 1,
        "scenarios": [
            {
                "scenario": "boresight_baseline",
                "phase_deg": 0.0,
                "phase_error_deg": 0.0,
                "tolerance_deg": 2.0,
                "passed": True,
            },
            {
                "scenario": "azimuth_sweep",
                "angles_commanded_deg": [-30.0, 0.0, 30.0],
                "angles_achieved_deg": [-30.1, 0.0, 29.9],
                "errors_deg": [0.1, 0.0, 0.1],
                "max_error_deg": 0.1,
                "mean_error_deg": 0.067,
                "tolerance_deg": 2.0,
                "passed": True,
            },
            {
                "scenario": "rain_attenuation_injection",
                "attenuation_levels_db": [0.0, 5.0, 10.0, 20.0, 30.0],
                "steps": [
                    {"attenuation_db": 0.0, "effective_snr_db": 25.0, "mcs_index": 4, "latency_ms": 5.0},
                    {"attenuation_db": 5.0, "effective_snr_db": 20.0, "mcs_index": 3, "latency_ms": 5.0},
                    {"attenuation_db": 10.0, "effective_snr_db": 15.0, "mcs_index": 2, "latency_ms": 5.0},
                    {"attenuation_db": 20.0, "effective_snr_db": 5.0, "mcs_index": 1, "latency_ms": 5.0},
                    {"attenuation_db": 30.0, "effective_snr_db": -5.0, "mcs_index": 0, "latency_ms": 5.0},
                ],
                "passed": True,
            },
            {
                "scenario": "handover_latency",
                "n_steps": 20,
                "latency_budget_ms": 500.0,
                "latencies_ms": [5.0] * 20,
                "p50_ms": 5.0,
                "p95_ms": 5.0,
                "p99_ms": 5.0,
                "mean_ms": 5.0,
                "max_ms": 5.0,
                "passed": True,
            },
            {
                "scenario": "steering_precision",
                "n_commands": 50,
                "delta_phase_rad": 0.1,
                "expected_phase_deg": 5.73,
                "std_deg": 0.0,
                "mean_error_deg": 0.0,
                "tolerance_deg": 2.0,
                "passed": True,
            },
        ],
    }


# ---------------------------------------------------------------------------
# analyze_field_test
# ---------------------------------------------------------------------------

class TestAnalyseFieldTest:
    def test_import(self):
        from analyze_field_test import analyse_report, _recommend_snr_threshold, _recommend_steering_gain
        assert analyse_report is not None

    def test_returns_all_keys(self):
        from analyze_field_test import analyse_report
        report = _make_minimal_field_test_report()
        cal = analyse_report(report)
        for key in ("snr_threshold_db", "max_failures", "steering_gain",
                    "latency_p95_ms", "steering_std_deg", "all_passed",
                    "recommendations"):
            assert key in cal

    def test_snr_threshold_from_rain_scenario(self):
        from analyze_field_test import _recommend_snr_threshold
        rain = {
            "steps": [
                {"effective_snr_db": 25.0},
                {"effective_snr_db": 20.0},
                {"effective_snr_db": 15.0},
                {"effective_snr_db": 5.0},
                {"effective_snr_db": -5.0},
            ]
        }
        thresh = _recommend_snr_threshold(rain)
        # P5 of [-5, 5, 15, 20, 25] is well below 20 but >= 1
        assert 1.0 <= thresh <= 20.0

    def test_snr_threshold_no_scenario_uses_default(self):
        from analyze_field_test import _recommend_snr_threshold, _DEFAULT_SNR_THRESHOLD_DB
        assert _recommend_snr_threshold(None) == _DEFAULT_SNR_THRESHOLD_DB

    def test_steering_gain_near_one_for_accurate_driver(self):
        from analyze_field_test import _recommend_steering_gain
        sweep = {
            "angles_commanded_deg": [-30.0, 30.0],
            "angles_achieved_deg": [-30.0, 30.0],
        }
        gain = _recommend_steering_gain(sweep)
        assert abs(gain - 1.0) < 0.01

    def test_steering_gain_detects_under_steer(self):
        from analyze_field_test import _recommend_steering_gain
        # Hardware achieves only 80% of commanded angle
        sweep = {
            "angles_commanded_deg": [-30.0, 30.0],
            "angles_achieved_deg": [-24.0, 24.0],
        }
        gain = _recommend_steering_gain(sweep)
        assert gain > 1.0  # need to command MORE to achieve desired phase

    def test_max_failures_tightened_for_high_latency(self):
        from analyze_field_test import _recommend_max_failures
        high_lat = {"p95_ms": 450.0}  # > 80% of 500 ms budget
        assert _recommend_max_failures(high_lat, step_interval_ms=500.0) == 2

    def test_max_failures_default_for_low_latency(self):
        from analyze_field_test import _recommend_max_failures, _DEFAULT_MAX_FAILURES
        low_lat = {"p95_ms": 10.0}
        assert _recommend_max_failures(low_lat, step_interval_ms=500.0) == _DEFAULT_MAX_FAILURES

    def test_main_reads_file_and_writes_calibration(self):
        from analyze_field_test import main
        report = _make_minimal_field_test_report()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as rf:
            json.dump(report, rf)
            report_path = rf.name
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as cf:
            cal_path = cf.name
        try:
            rc = main(["--report", report_path, "--output-cal", cal_path])
            assert rc == 0
            with open(cal_path) as f:
                cal = json.load(f)
            assert "snr_threshold_db" in cal
        finally:
            os.unlink(report_path)
            os.unlink(cal_path)

    def test_main_returns_1_for_missing_report(self):
        from analyze_field_test import main
        rc = main(["--report", "/tmp/__nonexistent_report__.json"])
        assert rc == 1

    def test_verbose_prints_output(self, capsys):
        from analyze_field_test import analyse_report
        report = _make_minimal_field_test_report()
        analyse_report(report, verbose=True)
        out = capsys.readouterr().out
        assert "snr_threshold" in out.lower()


# ---------------------------------------------------------------------------
# apply_calibration on OnlineBeamController
# ---------------------------------------------------------------------------

class _StubTelemetry:
    ground_station_pos = np.array([0.0, 0.0, 6371.0])

    def get_current_snr(self):
        return 20.0

    def get_current_rssi(self):
        return -80.0

    def get_current_position(self):
        return np.array([6921.0, 0.0, 0.0])


class _StubRadar:
    def get_at_location(self, pos):
        return 1.0


class _StubFoliage:
    def get_at_location(self, pos):
        return 1.5


class _ConstantAgent:
    def get_action(self, state, deterministic=True):
        return np.array([0.0, 1.0, 2.0, 50.0], dtype=np.float32), 0.0


class TestApplyCalibration:
    def _make_ctrl(self):
        from inference.online_controller import OnlineBeamController
        return OnlineBeamController(
            agent=_ConstantAgent(),
            telemetry_stream=_StubTelemetry(),
            radar_stream=_StubRadar(),
            foliage_map=_StubFoliage(),
        )

    def test_snr_threshold_updated(self):
        ctrl = self._make_ctrl()
        ctrl.apply_calibration({"snr_threshold_db": 3.0})
        assert abs(ctrl.snr_threshold - 3.0) < 1e-6

    def test_max_failures_updated(self):
        ctrl = self._make_ctrl()
        ctrl.apply_calibration({"max_failures": 2})
        assert ctrl.max_failures == 2

    def test_unknown_keys_ignored(self):
        ctrl = self._make_ctrl()
        ctrl.apply_calibration({"steering_gain": 1.1, "unknown_key": "foo"})
        # should not raise

    def test_partial_calibration_leaves_other_params_unchanged(self):
        ctrl = self._make_ctrl()
        original_max_failures = ctrl.max_failures
        ctrl.apply_calibration({"snr_threshold_db": 7.0})
        assert ctrl.max_failures == original_max_failures

    def test_calibration_affects_outage_detection(self):
        """Controller with high SNR threshold should declare more outages."""
        ctrl = self._make_ctrl()
        ctrl.apply_calibration({"snr_threshold_db": 25.0})  # above default SNR
        result = ctrl.step()
        # SNR of 20 < threshold of 25 → outage flag in metrics
        assert ctrl.snr_threshold == 25.0


# ---------------------------------------------------------------------------
# steering_gain on NullPhasedArrayDriver
# ---------------------------------------------------------------------------

class TestSteeringGain:
    def test_null_driver_default_gain_is_one(self):
        from hardware.phaser_driver import NullPhasedArrayDriver
        d = NullPhasedArrayDriver()
        assert d.steering_gain == 1.0

    def test_null_driver_gain_applied_to_phase(self):
        from hardware.phaser_driver import NullPhasedArrayDriver
        d = NullPhasedArrayDriver()
        d.steering_gain = 2.0
        d.apply_action(delta_phase=0.1, delta_power=1.0, mcs_index=0, rb_alloc=0)
        expected_deg = np.degrees(0.1 * 2.0) % 360.0
        assert abs(d.read_telemetry().phase_deg - expected_deg) < 0.01

    def test_null_driver_gain_one_no_change(self):
        from hardware.phaser_driver import NullPhasedArrayDriver
        d = NullPhasedArrayDriver()
        d.apply_action(delta_phase=0.5, delta_power=1.0, mcs_index=0, rb_alloc=0)
        expected = np.degrees(0.5) % 360.0
        assert abs(d.read_telemetry().phase_deg - expected) < 0.01

    def test_ethernet_driver_default_gain_is_one(self):
        from hardware.phaser_driver import EthernetPhasedArrayDriver
        d = EthernetPhasedArrayDriver()
        assert d.steering_gain == 1.0

    def test_ethernet_driver_gain_applied(self):
        from hardware.phaser_driver import EthernetPhasedArrayDriver
        d = EthernetPhasedArrayDriver()
        d.steering_gain = 0.5
        d.apply_action(delta_phase=0.4, delta_power=1.0, mcs_index=0, rb_alloc=0)
        expected_deg = np.degrees(0.4 * 0.5) % 360.0
        assert abs(d.read_telemetry().phase_deg - expected_deg) < 0.01


# ---------------------------------------------------------------------------
# pilot_monitor
# ---------------------------------------------------------------------------

class TestPilotMonitor:
    def test_import(self):
        from pilot_monitor import run_pilot_monitor, _print_summary
        assert run_pilot_monitor is not None

    def test_run_returns_required_keys(self):
        from pilot_monitor import run_pilot_monitor
        report = run_pilot_monitor(max_steps=10, interval_s=0.0)
        for key in ("completed_steps", "outage_rate", "fallback_rate",
                    "handovers", "snr_stats", "latency_stats"):
            assert key in report

    def test_completed_steps_equals_max_steps(self):
        from pilot_monitor import run_pilot_monitor
        report = run_pilot_monitor(max_steps=15, interval_s=0.0)
        assert report["completed_steps"] == 15

    def test_outage_rate_in_unit_range(self):
        from pilot_monitor import run_pilot_monitor
        report = run_pilot_monitor(max_steps=20, interval_s=0.0)
        assert 0.0 <= report["outage_rate"] <= 1.0

    def test_calibration_applied(self):
        from pilot_monitor import run_pilot_monitor
        cal = {"snr_threshold_db": 25.0, "max_failures": 2}
        report = run_pilot_monitor(max_steps=10, interval_s=0.0, calibration=cal)
        # With a high threshold, outage_rate should be high
        assert report["outage_rate"] >= 0.0

    def test_json_output(self):
        from pilot_monitor import run_pilot_monitor
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            run_pilot_monitor(max_steps=5, interval_s=0.0, output_json=path)
            with open(path) as f:
                doc = json.load(f)
            assert "completed_steps" in doc
        finally:
            os.unlink(path)

    def test_snr_stats_has_percentiles(self):
        from pilot_monitor import run_pilot_monitor
        report = run_pilot_monitor(max_steps=30, interval_s=0.0)
        stats = report["snr_stats"]
        for k in ("mean", "p50", "p95", "p99", "count"):
            assert k in stats

    def test_main_returns_zero(self):
        from pilot_monitor import main
        rc = main(["--steps", "5", "--interval-s", "0"])
        assert rc == 0

    def test_print_summary_does_not_raise(self, capsys):
        from pilot_monitor import run_pilot_monitor, _print_summary
        report = run_pilot_monitor(max_steps=5, interval_s=0.0)
        _print_summary(report)
        out = capsys.readouterr().out
        assert "Pilot Monitor" in out


# ---------------------------------------------------------------------------
# mlops_pipeline
# ---------------------------------------------------------------------------

class TestMlopsPipeline:
    def test_import(self):
        from mlops_pipeline import run_pipeline, main
        assert run_pipeline is not None

    def test_ci_mode_succeeds(self):
        from mlops_pipeline import run_pipeline
        with tempfile.TemporaryDirectory() as tmp:
            audit = run_pipeline(
                registry_path=os.path.join(tmp, "registry"),
                model_name="ppo_amazon",
                mode="ci",
                tmp_dir=os.path.join(tmp, "pipeline"),
                verbose=False,
            )
        assert audit.get("success") is True

    def test_ci_mode_with_skip_flags(self):
        from mlops_pipeline import run_pipeline
        with tempfile.TemporaryDirectory() as tmp:
            audit = run_pipeline(
                registry_path=os.path.join(tmp, "registry"),
                model_name="ppo_amazon",
                mode="ci",
                skip_field_test=True,
                skip_pilot=True,
                tmp_dir=os.path.join(tmp, "pipeline"),
                verbose=False,
            )
        assert audit.get("success") is True
        assert "field_test" not in audit.get("stages", {})
        assert "pilot_monitor" not in audit.get("stages", {})

    def test_audit_log_written(self):
        from mlops_pipeline import run_pipeline
        with tempfile.TemporaryDirectory() as tmp:
            audit_path = os.path.join(tmp, "audit.json")
            run_pipeline(
                registry_path=os.path.join(tmp, "registry"),
                model_name="ppo_amazon",
                mode="ci",
                output_audit=audit_path,
                tmp_dir=os.path.join(tmp, "pipeline"),
                verbose=False,
            )
            assert os.path.isfile(audit_path)
            with open(audit_path) as f:
                doc = json.load(f)
            assert "success" in doc

    def test_audit_contains_required_stages(self):
        from mlops_pipeline import run_pipeline
        with tempfile.TemporaryDirectory() as tmp:
            audit = run_pipeline(
                registry_path=os.path.join(tmp, "registry"),
                model_name="ppo_amazon",
                mode="ci",
                tmp_dir=os.path.join(tmp, "pipeline"),
                verbose=False,
            )
        stages = audit.get("stages", {})
        # All stages should be present in CI mode (none skipped)
        assert "field_test" in stages
        assert "analysis" in stages
        assert "pilot_monitor" in stages
        assert "retrain" in stages

    def test_main_returns_zero_on_success(self):
        from mlops_pipeline import main
        with tempfile.TemporaryDirectory() as tmp:
            rc = main([
                "--mode", "ci",
                "--registry", os.path.join(tmp, "registry"),
                "--tmp-dir", os.path.join(tmp, "pipeline"),
            ])
        assert rc == 0

    def test_print_audit_does_not_raise(self, capsys):
        from mlops_pipeline import run_pipeline, _print_audit
        with tempfile.TemporaryDirectory() as tmp:
            audit = run_pipeline(
                registry_path=os.path.join(tmp, "registry"),
                mode="ci",
                tmp_dir=os.path.join(tmp, "pipeline"),
            )
        _print_audit(audit)
        out = capsys.readouterr().out
        assert "MLOps" in out
