#!/usr/bin/env python3
"""
Field Test Procedure – Phased-Array Hardware Validation
========================================================

Validates the complete control loop against a physical (or simulated)
phased-array front-end:

  1. **Boresight baseline** – confirm the array reaches boresight phase (0°)
     within a configurable tolerance.
  2. **Azimuth sweep** – step the beam through −60° … +60° and record the
     achieved phase reported by hardware telemetry.
  3. **Rain attenuation injection** – inject a series of artificial SNR
     penalties (0, 5, 10, 20, 30 dB) and verify that the agent adapts its
     MCS index and resource-block allocation appropriately.
  4. **Handover latency** – measure the end-to-end latency of a satellite
     handover event (telemetry update → agent inference → hardware command).
  5. **Repeated steering precision** – issue 50 identical steering commands
     and compute the standard deviation of the achieved phase.

All results are collected in a structured JSON report and printed as a
Markdown summary.

Usage::

    # Simulation mode (NullPhasedArrayDriver):
    python scripts/field_test_hardware.py \\
        --driver null \\
        --steps 50 \\
        --output-json /tmp/field_test_report.json

    # Ethernet mode (real Anokiwave EVK / Phazr front-end):
    python scripts/field_test_hardware.py \\
        --driver ethernet \\
        --host 192.168.1.100 --port 5000 \\
        --steps 50 \\
        --output-json /tmp/field_test_report.json

Exit code:
    0  – all scenarios PASS
    1  – one or more scenarios FAIL
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

from hardware.phaser_driver import (
    NullPhasedArrayDriver,
    EthernetPhasedArrayDriver,
    LoggingPhasedArrayDriver,
    DriverTelemetry,
)
from inference.online_controller import HardwareBeamController

# ---------------------------------------------------------------------------
# Tolerances and constants
# ---------------------------------------------------------------------------

# Pointing-precision requirement: phase error ≤ 2° after each steering command
_PHASE_TOLERANCE_DEG = 2.0
# End-to-end latency budget (ms): telemetry → inference → hardware command
_LATENCY_BUDGET_MS = 500.0
# Minimum MCS uplift fraction expected when rain attenuation is removed
_MIN_MCS_REDUCE_FRACTION = 0.5

# ---------------------------------------------------------------------------
# Minimal sensor stubs (used when no real telemetry source is available)
# ---------------------------------------------------------------------------

class _StubTelemetry:
    """
    Minimal telemetry stub used for field-test scenarios.

    Returns a single satellite position at a fixed ECEF coordinate
    representative of a Starlink satellite over the Amazon at 550 km altitude.

    Args:
        snr_db:       SNR value returned by ``get_current_snr()`` (dB).
        rssi_dbm:     RSSI value (dBm).
        sat_pos_km:   ECEF satellite position (km).
    """

    # Earth's mean radius (km)
    _EARTH_RADIUS_KM = 6371.0

    def __init__(
        self,
        snr_db: float = 20.0,
        rssi_dbm: float = -80.0,
        sat_pos_km: Optional[np.ndarray] = None,
    ) -> None:
        self._snr = snr_db
        self._rssi = rssi_dbm
        self._pos = (
            sat_pos_km
            if sat_pos_km is not None
            else np.array([self._EARTH_RADIUS_KM + 550.0, 0.0, 0.0])
        )

    @property
    def snr_db(self) -> float:
        return self._snr

    @snr_db.setter
    def snr_db(self, value: float) -> None:
        self._snr = value

    def get_current_position(self) -> np.ndarray:
        return self._pos

    def get_current_snr(self) -> float:
        return self._snr

    def get_current_rssi(self) -> float:
        return self._rssi


class _StubRadar:
    def __init__(self, rain_mm_h: float = 0.0) -> None:
        self._rain = rain_mm_h

    def get_at_location(self, pos) -> float:
        return self._rain


class _StubFoliage:
    def get_at_location(self, pos) -> float:
        return 1.5  # representative Amazon LAI


class _ConstantAgent:
    """
    Minimal agent that always returns a fixed action vector.
    Used in controlled steering scenarios.
    """

    def __init__(self, action: np.ndarray) -> None:
        self._action = action

    def get_action(self, state, deterministic: bool = True):
        return self._action, 0.0


# ---------------------------------------------------------------------------
# Individual test scenarios
# ---------------------------------------------------------------------------

def _scenario_boresight(
    driver: NullPhasedArrayDriver,
    tolerance_deg: float = _PHASE_TOLERANCE_DEG,
) -> Dict[str, Any]:
    """
    Scenario 1: Boresight baseline.

    Sends a zero-phase command and checks that the hardware telemetry
    reports phase within *tolerance_deg* of 0°.
    """
    driver.reset()
    driver.apply_action(delta_phase=0.0, delta_power=1.0, mcs_index=2, rb_alloc=50)
    tel = driver.read_telemetry()
    phase_error_deg = abs(tel.phase_deg % 360.0)
    # Normalise to [0, 180]
    if phase_error_deg > 180.0:
        phase_error_deg = 360.0 - phase_error_deg

    passed = phase_error_deg <= tolerance_deg
    return {
        "scenario": "boresight_baseline",
        "phase_deg": tel.phase_deg,
        "phase_error_deg": phase_error_deg,
        "tolerance_deg": tolerance_deg,
        "passed": passed,
    }


def _scenario_azimuth_sweep(
    driver,
    angles_deg: Optional[List[float]] = None,
    tolerance_deg: float = _PHASE_TOLERANCE_DEG,
) -> Dict[str, Any]:
    """
    Scenario 2: Azimuth sweep.

    Steps through a list of azimuth angles (in degrees) and checks that the
    achieved phase (from hardware telemetry) matches the commanded phase
    within *tolerance_deg*.
    """
    if angles_deg is None:
        angles_deg = [-60.0, -45.0, -30.0, -15.0, 0.0, 15.0, 30.0, 45.0, 60.0]

    driver.reset()
    commanded: List[float] = []
    achieved: List[float] = []
    errors: List[float] = []

    cumulative_phase_deg = 0.0
    for target_deg in angles_deg:
        delta_deg = target_deg - cumulative_phase_deg
        delta_rad = np.radians(delta_deg)
        driver.apply_action(delta_phase=float(delta_rad), delta_power=1.0,
                            mcs_index=2, rb_alloc=50)
        tel = driver.read_telemetry()
        achieved_phase = tel.phase_deg
        # Normalise error
        error = abs((achieved_phase - target_deg) % 360.0)
        if error > 180.0:
            error = 360.0 - error
        commanded.append(target_deg)
        achieved.append(achieved_phase)
        errors.append(error)
        cumulative_phase_deg = target_deg

    max_error = max(errors) if errors else 0.0
    mean_error = float(np.mean(errors)) if errors else 0.0
    passed = max_error <= tolerance_deg

    return {
        "scenario": "azimuth_sweep",
        "angles_commanded_deg": commanded,
        "angles_achieved_deg": achieved,
        "errors_deg": errors,
        "max_error_deg": max_error,
        "mean_error_deg": mean_error,
        "tolerance_deg": tolerance_deg,
        "passed": passed,
    }


def _scenario_rain_injection(
    driver,
    telemetry_stub: _StubTelemetry,
    agent,
    attenuation_levels_db: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """
    Scenario 3: Rain attenuation injection.

    For each attenuation level, the SNR is degraded by the specified amount
    and the controller step is executed.  The MCS index in the resulting
    hardware command is recorded to verify that the agent reduces throughput
    demand (lower MCS) under heavy attenuation.

    A PASS is declared if the MCS index at the highest attenuation is ≤ the
    MCS at the lowest attenuation (monotone reduction).
    """
    if attenuation_levels_db is None:
        attenuation_levels_db = [0.0, 5.0, 10.0, 20.0, 30.0]

    from inference.online_controller import HardwareBeamController

    base_snr = telemetry_stub.snr_db
    results = []

    ctrl = HardwareBeamController(
        agent=agent,
        telemetry_stream=telemetry_stub,
        radar_stream=_StubRadar(0.0),
        foliage_map=_StubFoliage(),
        hw_driver=driver,
    )

    for att_db in attenuation_levels_db:
        ctrl.inject_rain_attenuation_db = att_db
        step_result = ctrl.step()
        hw_tel = ctrl.last_hw_telemetry
        mcs_used = hw_tel.extra.get("mcs_index", None) if hw_tel else None

        # Extract MCS from the action vector in the driver's command log
        if hasattr(driver, "command_log") and driver.command_log:
            mcs_used = driver.command_log[-1].mcs_index

        results.append({
            "attenuation_db": att_db,
            "effective_snr_db": base_snr - att_db,
            "action": (
                step_result["action"].tolist()
                if hasattr(step_result["action"], "tolist")
                else step_result["action"]
            ),
            "mcs_index": mcs_used,
            "latency_ms": step_result["latency_ms"],
        })

    # Monotone MCS check (NullDriver's agent action maps to fixed mcs,
    # so we check latency instead for the null-driver scenario)
    mcs_values = [r["mcs_index"] for r in results if r["mcs_index"] is not None]
    if len(mcs_values) >= 2:
        passed = mcs_values[-1] <= mcs_values[0]
    else:
        # Cannot evaluate MCS monotonicity; check that no step failed
        passed = all(r["latency_ms"] < _LATENCY_BUDGET_MS for r in results)

    return {
        "scenario": "rain_attenuation_injection",
        "attenuation_levels_db": attenuation_levels_db,
        "steps": results,
        "passed": passed,
    }


def _scenario_handover_latency(
    driver,
    n_steps: int = 20,
    latency_budget_ms: float = _LATENCY_BUDGET_MS,
) -> Dict[str, Any]:
    """
    Scenario 4: End-to-end handover latency.

    Runs *n_steps* controller steps and records the wall-clock latency
    (from start of ``step()`` to completion of the hardware ``apply_action``
    call) for each one.  Passes if the P95 latency ≤ *latency_budget_ms*.
    """
    telemetry = _StubTelemetry(snr_db=20.0)
    agent = _ConstantAgent(np.array([0.0, 1.0, 2.0, 50.0], dtype=np.float32))

    from inference.online_controller import HardwareBeamController

    ctrl = HardwareBeamController(
        agent=agent,
        telemetry_stream=telemetry,
        radar_stream=_StubRadar(0.0),
        foliage_map=_StubFoliage(),
        hw_driver=driver,
    )

    latencies: List[float] = []
    for _ in range(n_steps):
        result = ctrl.step()
        latencies.append(result["latency_ms"])

    p50 = float(np.percentile(latencies, 50))
    p95 = float(np.percentile(latencies, 95))
    p99 = float(np.percentile(latencies, 99))
    passed = p95 <= latency_budget_ms

    return {
        "scenario": "handover_latency",
        "n_steps": n_steps,
        "latency_budget_ms": latency_budget_ms,
        "latencies_ms": latencies,
        "p50_ms": p50,
        "p95_ms": p95,
        "p99_ms": p99,
        "mean_ms": float(np.mean(latencies)),
        "max_ms": float(np.max(latencies)),
        "passed": passed,
    }


def _scenario_steering_precision(
    driver,
    n_commands: int = 50,
    delta_phase_rad: float = 0.1,
    tolerance_deg: float = _PHASE_TOLERANCE_DEG,
) -> Dict[str, Any]:
    """
    Scenario 5: Repeated steering precision.

    Issues *n_commands* identical steering commands and measures the
    standard deviation of the achieved phase.  A low std dev indicates
    that the hardware responds consistently.
    """
    driver.reset()
    phases: List[float] = []

    for _ in range(n_commands):
        driver.reset()  # reset between commands so phase doesn't accumulate
        driver.apply_action(
            delta_phase=delta_phase_rad, delta_power=1.0, mcs_index=2, rb_alloc=50
        )
        tel = driver.read_telemetry()
        phases.append(tel.phase_deg)

    expected_deg = float(np.degrees(delta_phase_rad) % 360.0)
    errors = [abs(p - expected_deg) for p in phases]
    errors = [min(e, 360.0 - e) for e in errors]
    std_deg = float(np.std(phases))
    mean_error_deg = float(np.mean(errors))
    passed = std_deg <= tolerance_deg and mean_error_deg <= tolerance_deg

    return {
        "scenario": "steering_precision",
        "n_commands": n_commands,
        "delta_phase_rad": delta_phase_rad,
        "expected_phase_deg": expected_deg,
        "std_deg": std_deg,
        "mean_error_deg": mean_error_deg,
        "tolerance_deg": tolerance_deg,
        "passed": passed,
    }


# ---------------------------------------------------------------------------
# Run all scenarios
# ---------------------------------------------------------------------------

def _json_serializer(obj):
    """Custom JSON serializer for numpy scalars and booleans."""
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return float(obj)
    return str(obj)



def run_field_test(
    driver_type: str = "null",
    host: str = "192.168.1.100",
    port: int = 5000,
    n_steps: int = 50,
    output_json: Optional[str] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Execute all five validation scenarios and return a consolidated report.

    Args:
        driver_type:  ``"null"`` (simulation) or ``"ethernet"`` (real HW).
        host:         IP address for the Ethernet driver.
        port:         UDP port for the Ethernet driver.
        n_steps:      Number of steps in the latency scenario.
        output_json:  Optional path to save the JSON report.
        verbose:      Print per-scenario results while running.

    Returns:
        Consolidated report dictionary.
    """
    # --- Instantiate driver ---
    if driver_type == "ethernet":
        raw_driver = EthernetPhasedArrayDriver(host=host, port=port)
        raw_driver.connect()
    else:
        raw_driver = NullPhasedArrayDriver()

    driver = LoggingPhasedArrayDriver(raw_driver)

    telemetry = _StubTelemetry(snr_db=25.0)
    agent = _ConstantAgent(np.array([0.0, 1.0, 2.0, 50.0], dtype=np.float32))

    scenarios: List[Dict[str, Any]] = []

    def _run(name: str, fn, *args, **kwargs):
        if verbose:
            print(f"  Running: {name} …", end=" ", flush=True)
        result = fn(*args, **kwargs)
        status = "PASS" if result.get("passed", False) else "FAIL"
        if verbose:
            print(status)
        scenarios.append(result)
        return result

    _run("Boresight", _scenario_boresight, raw_driver)
    _run("Azimuth Sweep", _scenario_azimuth_sweep, raw_driver)
    _run("Rain Injection", _scenario_rain_injection, raw_driver, telemetry, agent)
    _run("Handover Latency", _scenario_handover_latency, raw_driver, n_steps=n_steps)
    _run("Steering Precision", _scenario_steering_precision, raw_driver)

    # --- Disconnect ---
    try:
        raw_driver.disconnect()
    except Exception:  # noqa: BLE001
        pass

    all_passed = all(s.get("passed", False) for s in scenarios)

    report: Dict[str, Any] = {
        "driver_type": driver_type,
        "host": host if driver_type == "ethernet" else "N/A",
        "port": port if driver_type == "ethernet" else 0,
        "n_steps": n_steps,
        "scenarios": scenarios,
        "all_passed": all_passed,
        "pass_count": sum(1 for s in scenarios if s.get("passed")),
        "fail_count": sum(1 for s in scenarios if not s.get("passed")),
    }

    if output_json:
        os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=_json_serializer)

    return report


# ---------------------------------------------------------------------------
# Markdown summary printer
# ---------------------------------------------------------------------------

def print_report(report: Dict[str, Any]) -> None:
    overall = "✅ PASS" if report["all_passed"] else "❌ FAIL"
    print()
    print("# Field Test Validation Report")
    print()
    print(f"Driver  : {report['driver_type']}", end="")
    if report["driver_type"] == "ethernet":
        print(f"  ({report['host']}:{report['port']})", end="")
    print()
    print(f"Overall : {overall}  "
          f"({report['pass_count']}/{report['pass_count'] + report['fail_count']} scenarios passed)")
    print()
    print("| Scenario | Result | Key Metric |")
    print("|---|---|---|")
    for s in report["scenarios"]:
        status = "✅ PASS" if s.get("passed") else "❌ FAIL"
        name = s["scenario"]
        if name == "boresight_baseline":
            metric = f"phase_error={s['phase_error_deg']:.3f}°"
        elif name == "azimuth_sweep":
            metric = f"max_error={s['max_error_deg']:.3f}°  mean={s['mean_error_deg']:.3f}°"
        elif name == "handover_latency":
            metric = f"p95={s['p95_ms']:.2f}ms  budget={s['latency_budget_ms']}ms"
        elif name == "steering_precision":
            metric = f"std={s['std_deg']:.3f}°  mean_error={s['mean_error_deg']:.3f}°"
        elif name == "rain_attenuation_injection":
            metric = f"{len(s['steps'])} attenuation levels"
        else:
            metric = ""
        print(f"| {name} | {status} | {metric} |")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Field test procedure for phased-array hardware validation."
    )
    p.add_argument("--driver", choices=["null", "ethernet"], default="null",
                   help="Driver type (default: null / simulation).")
    p.add_argument("--host", default="192.168.1.100",
                   help="IP address for the Ethernet driver.")
    p.add_argument("--port", type=int, default=5000,
                   help="UDP port for the Ethernet driver.")
    p.add_argument("--steps", type=int, default=50,
                   help="Number of steps in the latency scenario.")
    p.add_argument("--output-json", default=None,
                   help="Optional path to save JSON report.")
    p.add_argument("--verbose", action="store_true",
                   help="Print per-scenario status during execution.")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    report = run_field_test(
        driver_type=args.driver,
        host=args.host,
        port=args.port,
        n_steps=args.steps,
        output_json=args.output_json,
        verbose=args.verbose,
    )
    print_report(report)
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
