"""
Tests for src/hardware/phaser_driver.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest


class TestNullPhasedArrayDriver:
    def _make(self):
        from hardware.phaser_driver import NullPhasedArrayDriver
        return NullPhasedArrayDriver()

    def test_import(self):
        from hardware.phaser_driver import NullPhasedArrayDriver
        assert NullPhasedArrayDriver is not None

    def test_exported_from_hardware_package(self):
        from hardware import NullPhasedArrayDriver
        assert NullPhasedArrayDriver is not None

    def test_apply_action_records_command(self):
        from hardware.phaser_driver import BeamCommand
        d = self._make()
        d.apply_action(0.1, 0.8, 2, 50)
        assert len(d.command_log) == 1
        cmd = d.command_log[0]
        assert abs(cmd.delta_phase - 0.1) < 1e-9
        assert abs(cmd.delta_power - 0.8) < 1e-9
        assert cmd.mcs_index == 2
        assert cmd.rb_alloc == 50

    def test_apply_action_vector(self):
        d = self._make()
        d.apply_action_vector(np.array([0.05, 0.6, 1.9, 40.0]))
        cmd = d.command_log[-1]
        assert cmd.mcs_index == 2
        assert cmd.rb_alloc == 40

    def test_apply_action_vector_too_short_raises(self):
        from hardware.phaser_driver import NullPhasedArrayDriver
        d = NullPhasedArrayDriver()
        with pytest.raises(ValueError):
            d.apply_action_vector(np.array([0.1, 0.5]))

    def test_read_telemetry_returns_telemetry(self):
        from hardware.phaser_driver import DriverTelemetry
        d = self._make()
        t = d.read_telemetry()
        assert isinstance(t, DriverTelemetry)
        assert t.timestamp_s > 0

    def test_power_reflected_in_telemetry(self):
        d = self._make()
        d.apply_action(0.0, 1.0, 0, 0)
        t = d.read_telemetry()
        assert abs(t.tx_power_dbm - 30.0) < 1e-9

    def test_phase_accumulation(self):
        d = self._make()
        d.apply_action(0.1, 0.5, 0, 10)
        d.apply_action(0.2, 0.5, 0, 10)
        t = d.read_telemetry()
        expected = (np.degrees(0.1) + np.degrees(0.2)) % 360.0
        assert abs(t.phase_deg - expected) < 0.01

    def test_reset_clears_log(self):
        d = self._make()
        d.apply_action(0.0, 0.5, 1, 25)
        d.reset()
        assert len(d.command_log) == 0

    def test_measure_rtt_ms_reasonable(self):
        d = self._make()
        rtt = d.measure_rtt_ms(n_samples=5)
        assert 0.0 <= rtt < 100.0  # null driver is very fast


class TestLoggingPhasedArrayDriver:
    def test_wraps_null_driver(self):
        from hardware.phaser_driver import NullPhasedArrayDriver, LoggingPhasedArrayDriver
        inner = NullPhasedArrayDriver()
        outer = LoggingPhasedArrayDriver(inner)
        outer.apply_action(0.0, 0.5, 1, 30)
        assert len(inner.command_log) == 1

    def test_telemetry_passes_through(self):
        from hardware.phaser_driver import NullPhasedArrayDriver, LoggingPhasedArrayDriver, DriverTelemetry
        inner = NullPhasedArrayDriver()
        outer = LoggingPhasedArrayDriver(inner)
        t = outer.read_telemetry()
        assert isinstance(t, DriverTelemetry)

    def test_rtt_recorded_in_telemetry(self):
        from hardware.phaser_driver import NullPhasedArrayDriver, LoggingPhasedArrayDriver
        inner = NullPhasedArrayDriver()
        outer = LoggingPhasedArrayDriver(inner)
        t = outer.read_telemetry()
        assert t.rtt_ms >= 0.0


class TestEthernetPhasedArrayDriver:
    def test_import(self):
        from hardware.phaser_driver import EthernetPhasedArrayDriver
        assert EthernetPhasedArrayDriver is not None

    def test_apply_action_no_socket_does_not_raise(self):
        from hardware.phaser_driver import EthernetPhasedArrayDriver
        d = EthernetPhasedArrayDriver()
        # Socket not connected – should log a warning but not raise
        d.apply_action(0.1, 0.8, 2, 50)

    def test_apply_action_vector_clips(self):
        from hardware.phaser_driver import EthernetPhasedArrayDriver
        d = EthernetPhasedArrayDriver()
        d.apply_action_vector(np.array([0.0, 2.0, 10.0, 200.0]))

    def test_read_telemetry_returns_telemetry(self):
        from hardware.phaser_driver import EthernetPhasedArrayDriver, DriverTelemetry
        d = EthernetPhasedArrayDriver()
        t = d.read_telemetry()
        assert isinstance(t, DriverTelemetry)

    def test_phase_tracked_locally(self):
        from hardware.phaser_driver import EthernetPhasedArrayDriver
        d = EthernetPhasedArrayDriver()
        d.apply_action(0.5, 0.5, 0, 0)
        t = d.read_telemetry()
        expected_phase = np.degrees(0.5) % 360.0
        assert abs(t.phase_deg - expected_phase) < 0.01


class TestCanPhasedArrayDriver:
    def test_import(self):
        from hardware.phaser_driver import CanPhasedArrayDriver
        assert CanPhasedArrayDriver is not None

    def test_apply_action_simulation_mode(self):
        from hardware.phaser_driver import CanPhasedArrayDriver
        d = CanPhasedArrayDriver(can_id=0x100, bus=None)
        d.apply_action(0.1, 0.8, 2, 50)  # no bus → simulation

    def test_read_telemetry_returns_telemetry(self):
        from hardware.phaser_driver import CanPhasedArrayDriver, DriverTelemetry
        d = CanPhasedArrayDriver()
        t = d.read_telemetry()
        assert isinstance(t, DriverTelemetry)

    def test_reset_zeroes_state(self):
        from hardware.phaser_driver import CanPhasedArrayDriver
        d = CanPhasedArrayDriver()
        d.apply_action(0.5, 0.9, 3, 80)
        d.reset()
        t = d.read_telemetry()
        assert t.tx_power_dbm == 0.0
        assert t.phase_deg == 0.0

    def test_apply_action_vector_too_short_raises(self):
        from hardware.phaser_driver import CanPhasedArrayDriver
        d = CanPhasedArrayDriver()
        with pytest.raises(ValueError):
            d.apply_action_vector(np.array([0.1]))
