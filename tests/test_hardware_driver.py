"""
Tests for hardware driver abstraction layer.
"""

import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest


class TestNullHardwareDriver:
    def test_import(self):
        from beamforming.hardware_driver import NullHardwareDriver
        assert NullHardwareDriver is not None

    def test_exported_from_beamforming_package(self):
        from beamforming import NullHardwareDriver
        assert NullHardwareDriver is not None

    def test_apply_action_records_command(self):
        from beamforming.hardware_driver import NullHardwareDriver
        d = NullHardwareDriver()
        d.apply_action(0.1, 0.8, 2, 50)
        assert len(d.command_log) == 1
        cmd = d.command_log[0]
        assert abs(cmd.delta_phase - 0.1) < 1e-9
        assert abs(cmd.delta_power - 0.8) < 1e-9
        assert cmd.mcs_index == 2
        assert cmd.rb_alloc == 50

    def test_apply_action_vector(self):
        from beamforming.hardware_driver import NullHardwareDriver
        d = NullHardwareDriver()
        action = np.array([0.05, 0.6, 1.9, 40.0])
        d.apply_action_vector(action)
        cmd = d.command_log[-1]
        assert cmd.mcs_index == 2  # round(1.9) = 2
        assert cmd.rb_alloc == 40

    def test_read_telemetry_returns_telemetry(self):
        from beamforming.hardware_driver import NullHardwareDriver, Telemetry
        d = NullHardwareDriver()
        t = d.read_telemetry()
        assert isinstance(t, Telemetry)
        assert t.timestamp_s > 0

    def test_power_reflected_in_telemetry(self):
        from beamforming.hardware_driver import NullHardwareDriver
        d = NullHardwareDriver()
        d.apply_action(0.0, 1.0, 0, 0)
        t = d.read_telemetry()
        assert t.tx_power_dbm == 30.0

    def test_reset_clears_log(self):
        from beamforming.hardware_driver import NullHardwareDriver
        d = NullHardwareDriver()
        d.apply_action(0.0, 0.5, 1, 25)
        d.reset()
        assert len(d.command_log) == 0

    def test_multiple_actions_accumulate_phase(self):
        from beamforming.hardware_driver import NullHardwareDriver
        d = NullHardwareDriver()
        d.apply_action(0.1, 0.5, 0, 10)
        d.apply_action(0.2, 0.5, 0, 10)
        t = d.read_telemetry()
        # Phase should have accumulated: degrees(0.1) + degrees(0.2)
        expected = (np.degrees(0.1) + np.degrees(0.2)) % 360.0
        assert abs(t.phase_deg - expected) < 0.01


class TestLoggingHardwareDriver:
    def test_wraps_null_driver(self):
        from beamforming.hardware_driver import NullHardwareDriver, LoggingHardwareDriver
        inner = NullHardwareDriver()
        outer = LoggingHardwareDriver(inner)
        outer.apply_action(0.0, 0.5, 1, 30)
        assert len(inner.command_log) == 1

    def test_telemetry_passes_through(self):
        from beamforming.hardware_driver import NullHardwareDriver, LoggingHardwareDriver, Telemetry
        inner = NullHardwareDriver()
        outer = LoggingHardwareDriver(inner)
        t = outer.read_telemetry()
        assert isinstance(t, Telemetry)


class TestSpiHardwareDriver:
    def test_import(self):
        from beamforming.hardware_driver import SpiHardwareDriver
        assert SpiHardwareDriver is not None

    def test_apply_action_does_not_raise(self):
        from beamforming.hardware_driver import SpiHardwareDriver
        d = SpiHardwareDriver()
        d.apply_action(0.1, 0.8, 2, 50)

    def test_telemetry_returns_telemetry(self):
        from beamforming.hardware_driver import SpiHardwareDriver, Telemetry
        d = SpiHardwareDriver()
        t = d.read_telemetry()
        assert isinstance(t, Telemetry)

    def test_apply_action_vector_clips(self):
        from beamforming.hardware_driver import SpiHardwareDriver
        d = SpiHardwareDriver()
        action = np.array([0.0, 2.0, 10.0, 200.0])  # out-of-range values
        d.apply_action_vector(action)  # should not raise

    def test_apply_action_vector_too_short_raises(self):
        from beamforming.hardware_driver import SpiHardwareDriver
        d = SpiHardwareDriver()
        with pytest.raises(ValueError):
            d.apply_action_vector(np.array([0.1, 0.5]))
