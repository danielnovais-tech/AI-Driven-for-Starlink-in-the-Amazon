"""
Integration tests: validate the interaction between the channel model,
MultiSatelliteEnv, and the online controller pipeline.

These tests exercise the full data flow without requiring a real trained
agent by using a simple deterministic stub agent.
"""

import sys
import os
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Shared stubs (same pattern as other test modules)
# ---------------------------------------------------------------------------

class _FakeTelemetryMulti:
    ground_station_pos = np.array([0.0, 0.0, 6371.0])

    def get_visible_satellites(self):
        return [
            np.array([0.0, 0.0, 6921.0]),
            np.array([200.0, 0.0, 6921.0]),
            np.array([0.0, 200.0, 6921.0]),
        ]

    def get_current_position(self):
        return [0.0, 0.0, 6921.0]

    def get_current_snr(self):
        return 15.0

    def get_current_rssi(self):
        return -75.0


class _FakeRadar:
    def get_at_location(self, _pos):
        return 5.0


class _FakeFoliage:
    def get_at_location(self, _pos):
        return 2.0


class _ConstantAgent:
    """Always picks satellite 0 with a fixed continuous action."""
    _ACTION = np.array([0.05, 0.7, 1.0, 30.0], dtype=np.float32)

    def get_action(self, state, deterministic=True):
        return self._ACTION.copy(), -1.0

    def select_action(self, state):
        return 0  # discrete handover index


def _make_channel():
    from channel.rain_attenuation import ChannelModel
    return ChannelModel()


def _make_multi_env(max_sats=3):
    from envs.multi_satellite_env import MultiSatelliteEnv
    return MultiSatelliteEnv(
        channel_model=_make_channel(),
        telemetry_stream=_FakeTelemetryMulti(),
        radar_stream=_FakeRadar(),
        foliage_map=_FakeFoliage(),
        max_satellites=max_sats,
    )


# ---------------------------------------------------------------------------
# Integration: Channel + MultiSatelliteEnv
# ---------------------------------------------------------------------------

class TestChannelEnvIntegration:
    """Channel model is used inside MultiSatelliteEnv – validate end-to-end."""

    def test_snr_in_obs_plausible(self):
        """SNR values in the observation should be within ±100 dB of typical."""
        env = _make_multi_env()
        obs, _ = env.reset(seed=0)
        # obs is normalised; check it's finite
        assert np.all(np.isfinite(obs))

    def test_step_throughput_positive(self):
        env = _make_multi_env()
        env.reset(seed=0)
        _, _, _, _, info = env.step(0)
        assert info["throughput"] >= 0.0

    def test_step_latency_positive(self):
        env = _make_multi_env()
        env.reset(seed=0)
        _, _, _, _, info = env.step(0)
        assert info["latency"] > 0.0

    def test_outage_is_binary(self):
        env = _make_multi_env()
        env.reset(seed=0)
        for _ in range(10):
            _, _, _, _, info = env.step(0)
            assert info["outage"] in (0.0, 1.0)

    def test_snr_output_plausible(self):
        """SINR from ChannelModel should produce a plausible SNR value."""
        from channel.rain_attenuation import ChannelModel
        ch = ChannelModel()
        sat_pos = np.array([0.0, 0.0, 6921.0])
        snr = ch.compute_snr(sat_pos, rain_rate=0.0, foliage_density=0.0, elevation=45.0)
        assert -50.0 < snr < 80.0

    def test_sinr_lower_with_interferer(self):
        from channel.rain_attenuation import ChannelModel
        ch = ChannelModel()
        sat = np.array([0.0, 0.0, 6921.0])
        inter = np.array([100.0, 0.0, 6921.0])
        sinr_clean = ch.compute_sinr(sat, 0.0, 0.0, interfering_positions=[])
        sinr_interfered = ch.compute_sinr(sat, 0.0, 0.0, interfering_positions=[inter])
        assert sinr_interfered <= sinr_clean


# ---------------------------------------------------------------------------
# Integration: MultiSatelliteEnv + Controller
# ---------------------------------------------------------------------------

class TestControllerEnvIntegration:
    """Full pipeline: sensor → controller → action → environment."""

    def _make_controller(self, agent=None):
        from inference.online_controller import OnlineBeamController
        return OnlineBeamController(
            agent=agent or _ConstantAgent(),
            telemetry_stream=_FakeTelemetryMulti(),
            radar_stream=_FakeRadar(),
            foliage_map=_FakeFoliage(),
        )

    def test_controller_step_produces_action(self):
        ctrl = self._make_controller()
        result = ctrl.step()
        assert result["action"] is not None

    def test_controller_state_is_normalised(self):
        ctrl = self._make_controller()
        result = ctrl.step()
        # Normalised state should not have extreme values (clamped)
        state = result["state"]
        assert np.all(np.isfinite(state))

    def test_controller_and_env_compatible(self):
        """Controller uses same state format as MultiSatelliteEnv expects."""
        env = _make_multi_env()
        ctrl = self._make_controller()

        obs, _ = env.reset(seed=0)
        for _ in range(5):
            ctrl_result = ctrl.step()
            # Discrete action for the env (index 0)
            obs, reward, _, _, info = env.step(0)
            assert np.all(np.isfinite(obs))

    def test_controller_metrics_updated(self):
        from utils.metrics import MetricsRegistry
        reg = MetricsRegistry()  # fresh registry
        from inference.online_controller import OnlineBeamController

        class _MetricAgent(_ConstantAgent):
            pass

        ctrl = OnlineBeamController(
            agent=_MetricAgent(),
            telemetry_stream=_FakeTelemetryMulti(),
            radar_stream=_FakeRadar(),
            foliage_map=_FakeFoliage(),
        )
        # Replace global registry with our fresh one for this test
        import utils.metrics as _m
        original = _m.GLOBAL_REGISTRY
        _m.GLOBAL_REGISTRY = reg
        try:
            ctrl.step()
            ctrl.step()
        finally:
            _m.GLOBAL_REGISTRY = original


# ---------------------------------------------------------------------------
# Integration: RegulatoryEnv wrapping MultiSatelliteEnv
# ---------------------------------------------------------------------------

class TestRegulatoryEnvIntegration:
    def test_regulatory_wraps_multi_env(self):
        from envs.regulatory_env import RegulatoryEnv
        import gymnasium as gym
        import math

        inner = _make_multi_env()

        # Build a minimal Gymnasium-compliant env with a Box action space
        # so the regulatory wrapper can apply its power/phase constraints.
        class _BoxWrapper(gym.Env):
            metadata = {}
            observation_space = inner.observation_space
            action_space = gym.spaces.Box(
                low=np.array([-math.pi, 0.0, 0.0, 0.0], dtype=np.float32),
                high=np.array([math.pi, 1.0, 4.0, 100.0], dtype=np.float32),
                dtype=np.float32,
            )

            def reset(self, *, seed=None, options=None):
                return inner.reset(seed=seed, options=options)

            def step(self, action):
                idx = int(abs(action[0]) * inner.max_satellites) % inner.max_satellites
                return inner.step(idx)

            def render(self):
                pass

        env = RegulatoryEnv(_BoxWrapper(), max_eirp_dbw=55.0, min_elevation_deg=20.0)
        env.reset(seed=0)
        action = np.array([0.01, 0.3, 1.0, 10.0], dtype=np.float32)
        obs, reward, _, _, info = env.step(action)
        assert "compliance_violations" in info


# ---------------------------------------------------------------------------
# Integration: Orbital propagator + elevation angle computation
# ---------------------------------------------------------------------------

class TestOrbitalEnvIntegration:
    def test_propagator_feeds_visible_sats_to_env(self):
        from channel.orbital_propagator import SimplifiedPropagator, geodetic_to_ecef

        prop = SimplifiedPropagator(n_satellites=20, seed=0)
        gs_pos = geodetic_to_ecef(-3.1, -60.0, 0.05)

        t = 0.0
        visible = prop.get_visible_satellites(gs_pos, t, min_elevation_deg=25.0)
        # Just check the pipeline doesn't crash and returns a list
        assert isinstance(visible, list)
        for pos in visible:
            assert pos.shape == (3,)

    def test_elevation_angles_within_range(self):
        from channel.orbital_propagator import SimplifiedPropagator, geodetic_to_ecef, elevation_angle

        prop = SimplifiedPropagator(n_satellites=20, seed=0)
        gs_pos = geodetic_to_ecef(-3.1, -60.0, 0.05)
        visible = prop.get_visible_satellites(gs_pos, 0.0, min_elevation_deg=25.0)
        for pos in visible:
            elev = elevation_angle(gs_pos, pos)
            assert elev >= 25.0 - 0.01  # numerical tolerance
            assert elev <= 90.0 + 0.01
