"""
Tests for the field test hardware validation script and HardwareBeamController
enhancements (inject_rain_attenuation_db, last_hw_telemetry, _collect_state).
"""

import sys
import os
import json
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import numpy as np
import pytest

# Earth's mean radius (km) - used for ECEF positions in test stubs
_EARTH_RADIUS_KM = 6371.0


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------

class _StubTelemetry:
    ground_station_pos = np.array([0.0, 0.0, _EARTH_RADIUS_KM])

    def __init__(self, snr_db=20.0):
        self._snr = snr_db

    @property
    def snr_db(self):
        return self._snr

    @snr_db.setter
    def snr_db(self, val):
        self._snr = val

    def get_current_position(self):
        return np.array([_EARTH_RADIUS_KM + 550.0, 0.0, 0.0])

    def get_current_snr(self):
        return self._snr

    def get_current_rssi(self):
        return -80.0


class _StubRadar:
    def get_at_location(self, pos):
        return 3.0


class _StubFoliage:
    def get_at_location(self, pos):
        return 1.5


class _ConstantAgent:
    def __init__(self, action=None):
        self._action = action if action is not None else np.array([0.0, 1.0, 2.0, 50.0])

    def get_action(self, state, deterministic=True):
        return self._action, 0.0


# ---------------------------------------------------------------------------
# HardwareBeamController enhancements
# ---------------------------------------------------------------------------

class TestHardwareBeamControllerEnhancements:
    def _make_ctrl(self, snr_db=20.0, attenuation_db=0.0):
        from hardware.phaser_driver import NullPhasedArrayDriver
        from inference.online_controller import HardwareBeamController
        driver = NullPhasedArrayDriver()
        tel = _StubTelemetry(snr_db=snr_db)
        ctrl = HardwareBeamController(
            agent=_ConstantAgent(),
            telemetry_stream=tel,
            radar_stream=_StubRadar(),
            foliage_map=_StubFoliage(),
            hw_driver=driver,
            inject_rain_attenuation_db=attenuation_db,
        )
        return ctrl, driver, tel

    def test_hardware_controller_has_last_hw_telemetry(self):
        ctrl, _, _ = self._make_ctrl()
        assert hasattr(ctrl, "last_hw_telemetry")

    def test_last_hw_telemetry_populated_after_step(self):
        from hardware.phaser_driver import DriverTelemetry
        ctrl, _, _ = self._make_ctrl()
        ctrl.step()
        assert ctrl.last_hw_telemetry is not None
        assert isinstance(ctrl.last_hw_telemetry, DriverTelemetry)

    def test_inject_rain_attenuation_lowers_effective_snr(self):
        """With 10 dB attenuation, the state's SNR should be 10 dB lower."""
        ctrl_no_att, _, _ = self._make_ctrl(snr_db=20.0, attenuation_db=0.0)
        ctrl_with_att, _, _ = self._make_ctrl(snr_db=20.0, attenuation_db=10.0)
        _, raw_no_att = ctrl_no_att._collect_state()
        _, raw_with_att = ctrl_with_att._collect_state()
        # raw[0] is the SNR
        assert abs(raw_no_att[0] - raw_with_att[0] - 10.0) < 1e-4

    def test_inject_zero_attenuation_no_change(self):
        ctrl, _, _ = self._make_ctrl(snr_db=15.0, attenuation_db=0.0)
        _, raw = ctrl._collect_state()
        assert abs(raw[0] - 15.0) < 1e-4

    def test_step_returns_explanation_key(self):
        ctrl, _, _ = self._make_ctrl()
        result = ctrl.step()
        assert "explanation" in result

    def test_discrete_action_converted_to_vector(self):
        """Integer action should be forwarded as a 4-element vector."""
        ctrl, driver, _ = self._make_ctrl()
        ctrl._total_steps = 0
        ctrl.apply_beam_steering(2)  # integer action
        assert len(driver.command_log) == 1
        cmd = driver.command_log[-1]
        # mcs_index should be 2 % 5 = 2
        assert cmd.mcs_index == 2

    def test_continuous_action_forwarded(self):
        ctrl, driver, _ = self._make_ctrl()
        ctrl.apply_beam_steering(np.array([0.1, 0.8, 3.0, 70.0], dtype=np.float32))
        cmd = driver.command_log[-1]
        assert cmd.mcs_index == 3
        assert cmd.rb_alloc == 70

    def test_inject_attenuation_parameter_default(self):
        from hardware.phaser_driver import NullPhasedArrayDriver
        from inference.online_controller import HardwareBeamController
        ctrl = HardwareBeamController(
            agent=_ConstantAgent(),
            telemetry_stream=_StubTelemetry(),
            radar_stream=_StubRadar(),
            foliage_map=_StubFoliage(),
            hw_driver=NullPhasedArrayDriver(),
        )
        assert ctrl.inject_rain_attenuation_db == 0.0

    def test_attenuation_injectable_at_runtime(self):
        ctrl, _, _ = self._make_ctrl(snr_db=20.0, attenuation_db=0.0)
        _, raw_before = ctrl._collect_state()
        ctrl.inject_rain_attenuation_db = 5.0
        _, raw_after = ctrl._collect_state()
        assert abs(raw_before[0] - raw_after[0] - 5.0) < 1e-4


# ---------------------------------------------------------------------------
# Field test script scenarios (unit tests)
# ---------------------------------------------------------------------------

class TestFieldTestScenarios:
    def test_import_script(self):
        from field_test_hardware import (
            _scenario_boresight,
            _scenario_azimuth_sweep,
            _scenario_handover_latency,
            _scenario_steering_precision,
            run_field_test,
            print_report,
        )

    def test_boresight_passes_with_null_driver(self):
        from hardware.phaser_driver import NullPhasedArrayDriver
        from field_test_hardware import _scenario_boresight
        driver = NullPhasedArrayDriver()
        result = _scenario_boresight(driver)
        assert result["passed"]
        assert result["scenario"] == "boresight_baseline"

    def test_azimuth_sweep_returns_correct_keys(self):
        from hardware.phaser_driver import NullPhasedArrayDriver
        from field_test_hardware import _scenario_azimuth_sweep
        driver = NullPhasedArrayDriver()
        result = _scenario_azimuth_sweep(driver, angles_deg=[-30.0, 0.0, 30.0])
        for key in ("angles_commanded_deg", "angles_achieved_deg", "errors_deg",
                    "max_error_deg", "passed"):
            assert key in result

    def test_azimuth_sweep_length_matches_angles(self):
        from hardware.phaser_driver import NullPhasedArrayDriver
        from field_test_hardware import _scenario_azimuth_sweep
        driver = NullPhasedArrayDriver()
        angles = [-60.0, -30.0, 0.0, 30.0, 60.0]
        result = _scenario_azimuth_sweep(driver, angles_deg=angles)
        assert len(result["angles_commanded_deg"]) == len(angles)
        assert len(result["angles_achieved_deg"]) == len(angles)

    def test_handover_latency_within_budget(self):
        from hardware.phaser_driver import NullPhasedArrayDriver
        from field_test_hardware import _scenario_handover_latency
        driver = NullPhasedArrayDriver()
        result = _scenario_handover_latency(driver, n_steps=10)
        assert result["passed"]
        assert result["p95_ms"] < 500.0

    def test_steering_precision_passes_with_null_driver(self):
        from hardware.phaser_driver import NullPhasedArrayDriver
        from field_test_hardware import _scenario_steering_precision
        driver = NullPhasedArrayDriver()
        result = _scenario_steering_precision(driver, n_commands=20)
        assert result["passed"]
        assert result["std_deg"] < 1.0

    def test_run_field_test_all_pass(self):
        from field_test_hardware import run_field_test
        report = run_field_test(driver_type="null", n_steps=10)
        assert report["all_passed"]
        assert report["fail_count"] == 0
        assert len(report["scenarios"]) == 5

    def test_run_field_test_json_output(self):
        from field_test_hardware import run_field_test
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            run_field_test(driver_type="null", n_steps=5, output_json=path)
            with open(path) as f:
                doc = json.load(f)
            assert "scenarios" in doc
            assert "all_passed" in doc
        finally:
            os.unlink(path)

    def test_print_report_does_not_raise(self, capsys):
        from field_test_hardware import run_field_test, print_report
        report = run_field_test(driver_type="null", n_steps=5)
        print_report(report)
        out = capsys.readouterr().out
        assert "PASS" in out or "FAIL" in out


# ---------------------------------------------------------------------------
# generate_paper_figures
# ---------------------------------------------------------------------------

class TestGeneratePaperFigures:
    def test_import(self):
        from generate_paper_figures import main, _run_all, _aggregate, _to_markdown
        assert main is not None

    def test_run_one_seed_one_day(self):
        from generate_paper_figures import _run_all, _aggregate
        results = _run_all(
            n_days=1.0 / 1440.0,  # 1 simulated minute
            n_satellites=20,
            max_satellites=3,
            seeds=[42],
            verbose=False,
        )
        assert "random" in results
        assert "max_snr" in results
        for policy in ("random", "max_snr"):
            assert len(results[policy]) == 1
            agg = _aggregate(results[policy])
            assert "mean_throughput_mbps" in agg

    def test_markdown_table_contains_policies(self):
        from generate_paper_figures import _run_all, _aggregate, _build_table_rows, _to_markdown
        results = _run_all(
            n_days=1.0 / 1440.0, n_satellites=10, max_satellites=3,
            seeds=[42], verbose=False,
        )
        agg = {p: _aggregate(r) for p, r in results.items()}
        rows = _build_table_rows(agg)
        md = _to_markdown(rows)
        assert "random" in md
        assert "max_snr" in md

    def test_csv_output(self):
        from generate_paper_figures import _run_all, _aggregate, _build_table_rows, _to_csv
        results = _run_all(
            n_days=1.0 / 1440.0, n_satellites=10, max_satellites=3,
            seeds=[42], verbose=False,
        )
        agg = {p: _aggregate(r) for p, r in results.items()}
        rows = _build_table_rows(agg)
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = f.name
        try:
            _to_csv(rows, path)
            import csv as _csv
            with open(path) as f:
                reader = _csv.DictReader(f)
                csv_rows = list(reader)
            assert len(csv_rows) == 2  # random + max_snr
        finally:
            os.unlink(path)

    def test_main_runs_without_error(self):
        from generate_paper_figures import main
        rc = main([
            "--n-days", "0.000694",  # 1 minute
            "--n-satellites", "20",
            "--max-satellites", "3",
            "--seeds", "42",
        ])
        assert rc == 0


# ---------------------------------------------------------------------------
# benchmark_inference TensorRT stub
# ---------------------------------------------------------------------------

class TestTensorRTStub:
    def test_export_tensorrt_stub_import(self):
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
        from benchmark_inference import export_tensorrt_stub
        assert export_tensorrt_stub is not None

    def test_export_tensorrt_stub_returns_false_without_tensorrt(self):
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
        from benchmark_inference import export_tensorrt_stub
        # tensorrt is not installed in the test environment → should return False
        result = export_tensorrt_stub("/tmp/dummy.onnx", "/tmp/dummy.engine")
        assert result is False
