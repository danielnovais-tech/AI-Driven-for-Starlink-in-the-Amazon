"""
Tests for the online beam controller (fallback policy + watchdog + metrics).
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------

class _FakeTelemetry:
    def get_current_position(self):
        return [0.0, 0.0, 6921.0]

    def get_current_snr(self):
        return 15.0

    def get_current_rssi(self):
        return -75.0


class _FakeRadar:
    def get_at_location(self, _pos):
        return 2.0


class _FakeFoliage:
    def get_at_location(self, _pos):
        return 1.5


class _GoodAgent:
    """Agent that returns a fixed action."""
    _ACTION = np.array([0.1, 0.8, 2.0, 50.0], dtype=np.float32)

    def get_action(self, state, deterministic=True):
        return self._ACTION.copy(), -1.5  # (action, log_prob) PPO-style


class _BrokenAgent:
    """Agent that always raises."""
    def get_action(self, state, deterministic=True):
        raise RuntimeError("simulated agent failure")


def _make_controller(agent=None):
    from inference.online_controller import OnlineBeamController
    return OnlineBeamController(
        agent=agent or _GoodAgent(),
        telemetry_stream=_FakeTelemetry(),
        radar_stream=_FakeRadar(),
        foliage_map=_FakeFoliage(),
    )


# ---------------------------------------------------------------------------
# FallbackPolicy
# ---------------------------------------------------------------------------

class TestFallbackPolicy:
    def test_import(self):
        from inference.online_controller import FallbackPolicy
        assert FallbackPolicy is not None

    def test_exported_from_inference(self):
        from inference import FallbackPolicy
        assert FallbackPolicy is not None

    def test_default_action_safe(self):
        from inference.online_controller import FallbackPolicy
        fp = FallbackPolicy()
        action = fp.get_action(np.zeros(7))
        assert action.shape == (4,)

    def test_hold_last_valid(self):
        from inference.online_controller import FallbackPolicy
        fp = FallbackPolicy()
        stored = np.array([0.2, 0.6, 1.0, 30.0], dtype=np.float32)
        fp.update(stored)
        action = fp.get_action(np.zeros(7))
        np.testing.assert_array_almost_equal(action, stored)

    def test_returns_copy_not_reference(self):
        from inference.online_controller import FallbackPolicy
        fp = FallbackPolicy()
        stored = np.array([0.2, 0.6, 1.0, 30.0], dtype=np.float32)
        fp.update(stored)
        a1 = fp.get_action(np.zeros(7))
        a1[0] = 999.0
        a2 = fp.get_action(np.zeros(7))
        assert abs(a2[0] - 0.2) < 1e-6


# ---------------------------------------------------------------------------
# OnlineBeamController – normal operation
# ---------------------------------------------------------------------------

class TestOnlineBeamController:
    def test_import(self):
        from inference.online_controller import OnlineBeamController
        assert OnlineBeamController is not None

    def test_step_returns_dict(self):
        ctrl = _make_controller()
        result = ctrl.step()
        assert isinstance(result, dict)

    def test_step_result_keys(self):
        ctrl = _make_controller()
        result = ctrl.step()
        for key in ["action", "state", "snr", "rain", "fallback", "latency_ms"]:
            assert key in result

    def test_step_fallback_false_when_agent_works(self):
        ctrl = _make_controller()
        result = ctrl.step()
        assert result["fallback"] is False

    def test_latency_ms_positive(self):
        ctrl = _make_controller()
        result = ctrl.step()
        assert result["latency_ms"] >= 0.0

    def test_snr_is_float(self):
        ctrl = _make_controller()
        result = ctrl.step()
        assert isinstance(result["snr"], float)

    def test_history_grows(self):
        ctrl = _make_controller()
        for _ in range(5):
            ctrl.step()
        assert len(ctrl.state_history) == 5
        assert len(ctrl.action_history) == 5

    def test_is_healthy_initially(self):
        ctrl = _make_controller()
        assert ctrl.is_healthy is True


# ---------------------------------------------------------------------------
# Fallback / watchdog behaviour
# ---------------------------------------------------------------------------

class TestControllerFallback:
    def test_fallback_used_when_agent_fails(self):
        ctrl = _make_controller(agent=_BrokenAgent())
        result = ctrl.step()
        assert result["fallback"] is True

    def test_consecutive_failures_increments(self):
        ctrl = _make_controller(agent=_BrokenAgent())
        ctrl.step()
        ctrl.step()
        assert ctrl.consecutive_failures == 2

    def test_failures_reset_after_recovery(self):
        ctrl = _make_controller()
        # Manually inject failures then recover
        ctrl.consecutive_failures = 5
        ctrl.step()  # agent is good → resets failures
        assert ctrl.consecutive_failures == 0

    def test_is_healthy_false_after_max_failures(self):
        ctrl = _make_controller(agent=_BrokenAgent())
        ctrl.max_failures = 2
        ctrl.step()
        ctrl.step()
        assert ctrl.is_healthy is False
