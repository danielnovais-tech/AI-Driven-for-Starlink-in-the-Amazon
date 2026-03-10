"""
Tests for the new GNNBeamController and HardwareBeamController subclasses.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------

class _FakeTelemetryMulti:
    ground_station_pos = np.array([0.0, 0.0, 6371.0])

    def get_visible_satellites(self):
        return [
            np.array([6371.0 + 550.0, 0.0, 0.0]),
            np.array([0.0, 6371.0 + 550.0, 0.0]),
        ]

    def get_current_snr(self):
        return 15.0

    def get_current_rssi(self):
        return -80.0

    def get_current_position(self):
        return np.array([6371.0 + 550.0, 0.0, 0.0])


class _FakeTelemetrySingle:
    ground_station_pos = np.array([0.0, 0.0, 6371.0])

    def get_visible_satellites(self):
        return [np.array([6371.0 + 550.0, 0.0, 0.0])]

    def get_current_snr(self):
        return 12.0

    def get_current_rssi(self):
        return -82.0

    def get_current_position(self):
        return np.array([6371.0 + 550.0, 0.0, 0.0])


class _FakeRadar:
    def get_at_location(self, pos):
        return 3.0


class _FakeFoliage:
    def get_at_location(self, pos):
        return 1.0


class _FakeAgent:
    """Simple flat-state DRL agent stub."""
    def get_action(self, state, deterministic=True):
        return np.array([0.1, 0.5, 1.0, 20.0], dtype=np.float32), 0.0


class _FakeDiscreteAgent:
    """Discrete agent returning satellite index 0."""
    def get_action(self, state, deterministic=True):
        return 0, 0.0


# ---------------------------------------------------------------------------
# HardwareBeamController
# ---------------------------------------------------------------------------

class TestHardwareBeamController:
    def test_import(self):
        from inference.online_controller import HardwareBeamController
        assert HardwareBeamController is not None

    def _make(self):
        from inference.online_controller import HardwareBeamController
        from hardware.phaser_driver import NullPhasedArrayDriver
        driver = NullPhasedArrayDriver()
        return HardwareBeamController(
            agent=_FakeAgent(),
            telemetry_stream=_FakeTelemetrySingle(),
            radar_stream=_FakeRadar(),
            foliage_map=_FakeFoliage(),
            hw_driver=driver,
        ), driver

    def test_step_applies_to_driver(self):
        ctrl, driver = self._make()
        ctrl.step()
        assert len(driver.command_log) >= 1
        # Verify the command has a valid power fraction [0, 1]
        cmd = driver.command_log[0]
        assert 0.0 <= cmd.delta_power <= 1.0
        # MCS index must be in [0, 4]
        assert 0 <= cmd.mcs_index <= 4

    def test_discrete_action_forwarded_to_driver(self):
        from inference.online_controller import HardwareBeamController
        from hardware.phaser_driver import NullPhasedArrayDriver
        driver = NullPhasedArrayDriver()
        ctrl = HardwareBeamController(
            agent=_FakeDiscreteAgent(),
            telemetry_stream=_FakeTelemetrySingle(),
            radar_stream=_FakeRadar(),
            foliage_map=_FakeFoliage(),
            hw_driver=driver,
        )
        ctrl.apply_beam_steering(2)
        assert len(driver.command_log) == 1

    def test_driver_error_does_not_raise(self):
        """A broken driver must not crash the controller."""
        from inference.online_controller import HardwareBeamController

        class _BrokenDriver:
            def apply_action_vector(self, _):
                raise RuntimeError("hardware fault")

        ctrl = HardwareBeamController(
            agent=_FakeAgent(),
            telemetry_stream=_FakeTelemetrySingle(),
            radar_stream=_FakeRadar(),
            foliage_map=_FakeFoliage(),
            hw_driver=_BrokenDriver(),
        )
        ctrl.apply_beam_steering(np.array([0.1, 0.5, 1.0, 20.0]))  # must not raise

    def test_exported_from_inference_package(self):
        from inference import HardwareBeamController
        assert HardwareBeamController is not None


# ---------------------------------------------------------------------------
# GNNBeamController  (skipped if torch_geometric absent)
# ---------------------------------------------------------------------------

try:
    import torch_geometric  # noqa: F401
    _HAS_TG = True
except ImportError:
    _HAS_TG = False

pytestmark_gnn = pytest.mark.skipif(not _HAS_TG, reason="torch_geometric not installed")


class TestGNNBeamController:
    def test_import(self):
        from inference.online_controller import GNNBeamController
        assert GNNBeamController is not None

    @pytest.mark.skipif(not _HAS_TG, reason="torch_geometric not installed")
    def test_step_returns_expected_keys(self):
        from inference.online_controller import GNNBeamController
        from agents.gnn_ppo_agent import GNNPPOAgent

        agent = GNNPPOAgent(node_features=4)
        ctrl = GNNBeamController(
            agent=agent,
            telemetry_stream=_FakeTelemetryMulti(),
            radar_stream=_FakeRadar(),
            foliage_map=_FakeFoliage(),
        )
        result = ctrl.step()
        for key in ("action", "graph_obs", "snr", "rain", "fallback", "latency_ms", "n_visible"):
            assert key in result

    @pytest.mark.skipif(not _HAS_TG, reason="torch_geometric not installed")
    def test_action_is_valid_satellite_index(self):
        from inference.online_controller import GNNBeamController
        from agents.gnn_ppo_agent import GNNPPOAgent

        agent = GNNPPOAgent(node_features=4)
        ctrl = GNNBeamController(
            agent=agent,
            telemetry_stream=_FakeTelemetryMulti(),
            radar_stream=_FakeRadar(),
            foliage_map=_FakeFoliage(),
        )
        result = ctrl.step()
        assert isinstance(result["action"], (int, np.integer))

    @pytest.mark.skipif(not _HAS_TG, reason="torch_geometric not installed")
    def test_fallback_on_agent_error(self):
        from inference.online_controller import GNNBeamController

        class _FailAgent:
            def get_action(self, graph, deterministic=True):
                raise ValueError("forced error")

        ctrl = GNNBeamController(
            agent=_FailAgent(),
            telemetry_stream=_FakeTelemetryMulti(),
            radar_stream=_FakeRadar(),
            foliage_map=_FakeFoliage(),
        )
        result = ctrl.step()
        assert result["fallback"] is True

    @pytest.mark.skipif(not _HAS_TG, reason="torch_geometric not installed")
    def test_latency_within_budget(self):
        from inference.online_controller import GNNBeamController
        from agents.gnn_ppo_agent import GNNPPOAgent

        agent = GNNPPOAgent(node_features=4)
        ctrl = GNNBeamController(
            agent=agent,
            telemetry_stream=_FakeTelemetryMulti(),
            radar_stream=_FakeRadar(),
            foliage_map=_FakeFoliage(),
        )
        result = ctrl.step()
        assert result["latency_ms"] < 500.0

    def test_exported_from_inference_package(self):
        from inference import GNNBeamController
        assert GNNBeamController is not None
