"""
Tests for ExclusionZone, GeoRegulatoryEnv.
"""

import sys
import os
import math
import tempfile
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest
import gymnasium as gym


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------

class _DiscreteInner(gym.Env):
    observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
    action_space = gym.spaces.Discrete(3)

    def reset(self, *, seed=None, options=None):
        return np.zeros(4, dtype=np.float32), {}

    def step(self, action):
        return np.zeros(4, dtype=np.float32), 1.0, False, False, {}


class _MultiSatInner(gym.Env):
    """Fake discrete env that exposes visible_sats like MultiSatelliteEnv."""
    observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(10,), dtype=np.float32)
    action_space = gym.spaces.Discrete(3)

    class _FakeTelemetry:
        ground_station_pos = np.array([0.0, 0.0, 6371.0])

    telemetry = _FakeTelemetry()
    max_satellites = 3

    def __init__(self):
        super().__init__()
        # Visible satellite positions (ECEF km); SSPs computed as lon/lat
        # sat 0: roughly above lon=0, lat=0 (equator, prime meridian)
        # sat 1: roughly above lon=-60, lat=-3 (Amazon region)
        # sat 2: roughly above lon=10, lat=48 (central Europe)
        r = 6371.0 + 550.0
        self.visible_sats = [
            np.array([r, 0.0, 0.0]),                # lon=0, lat=0
            np.array([-r * 0.5, -r * 0.866, 0.0]), # lon≈-120, lat≈0
            np.array([r * 0.7, 0.0, r * 0.72]),    # lon≈0, lat≈45
        ]

    def reset(self, *, seed=None, options=None):
        return np.zeros(10, dtype=np.float32), {}

    def step(self, action):
        return np.zeros(10, dtype=np.float32), 1.0, False, False, {}


# ---------------------------------------------------------------------------
# ExclusionZone tests
# ---------------------------------------------------------------------------

class TestExclusionZone:
    def test_import(self):
        from envs.regulatory_env import ExclusionZone
        assert ExclusionZone is not None

    def test_exported_from_envs(self):
        from envs import ExclusionZone
        assert ExclusionZone is not None

    def test_point_inside(self):
        from envs.regulatory_env import ExclusionZone
        z = ExclusionZone("test", [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)])
        assert z.contains(0.0, 0.0) is True

    def test_point_outside(self):
        from envs.regulatory_env import ExclusionZone
        z = ExclusionZone("test", [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)])
        assert z.contains(5.0, 5.0) is False

    def test_point_on_boundary(self):
        from envs.regulatory_env import ExclusionZone
        z = ExclusionZone("test", [(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)])
        # Point exactly on an edge – result may vary by algorithm; just confirm no crash
        result = z.contains(1.0, 0.0)
        assert isinstance(result, bool)

    def test_name_and_reason(self):
        from envs.regulatory_env import ExclusionZone
        z = ExclusionZone("Atacama", [(0.0, 0.0), (1.0, 0.0), (0.5, 1.0)],
                          reason="Radio telescope protection")
        assert z.name == "Atacama"
        assert "telescope" in z.reason.lower()


# ---------------------------------------------------------------------------
# GeoRegulatoryEnv tests
# ---------------------------------------------------------------------------

class TestGeoRegulatoryEnv:
    def test_import(self):
        from envs.regulatory_env import GeoRegulatoryEnv
        assert GeoRegulatoryEnv is not None

    def test_exported_from_envs(self):
        from envs import GeoRegulatoryEnv
        assert GeoRegulatoryEnv is not None

    def test_no_zones_passthrough(self):
        from envs.regulatory_env import GeoRegulatoryEnv
        env = GeoRegulatoryEnv(_DiscreteInner(), exclusion_zones=[])
        env.reset()
        _, reward, _, _, info = env.step(0)
        assert info["geo_exclusion_violations"] == 0
        assert reward == 1.0

    def test_zone_violation_triggers_penalty(self):
        from envs.regulatory_env import GeoRegulatoryEnv, ExclusionZone

        # Sat 0 at (lon=0, lat=0): add a zone that covers that point
        inner = _MultiSatInner()
        zone = ExclusionZone("all_equator", [(-5.0, -5.0), (5.0, -5.0),
                                              (5.0, 5.0), (-5.0, 5.0)])
        env = GeoRegulatoryEnv(inner, exclusion_zones=[zone], geo_penalty=20.0)
        env.reset()
        _, reward, _, _, info = env.step(0)
        assert info["geo_exclusion_violations"] >= 1
        assert reward < 1.0  # penalty applied

    def test_no_violation_without_zone_match(self):
        from envs.regulatory_env import GeoRegulatoryEnv, ExclusionZone

        inner = _MultiSatInner()
        # Zone far from all satellite SSPs
        zone = ExclusionZone("remote", [(170.0, 70.0), (175.0, 70.0),
                                         (175.0, 75.0), (170.0, 75.0)])
        env = GeoRegulatoryEnv(inner, exclusion_zones=[zone])
        env.reset()
        _, reward, _, _, info = env.step(0)
        # Should NOT violate (sat 0 is at lon=0, lat=0, far from zone)
        assert info["geo_exclusion_violations"] == 0

    def test_violation_log_grows(self):
        from envs.regulatory_env import GeoRegulatoryEnv, ExclusionZone

        inner = _MultiSatInner()
        zone = ExclusionZone("wide", [(-5.0, -5.0), (5.0, -5.0),
                                       (5.0, 5.0), (-5.0, 5.0)])
        env = GeoRegulatoryEnv(inner, exclusion_zones=[zone])
        env.reset()
        for _ in range(3):
            env.step(0)
        assert len(env.violation_log) >= 1  # at least one blocked

    def test_audit_log_written(self):
        from envs.regulatory_env import GeoRegulatoryEnv, ExclusionZone

        inner = _MultiSatInner()
        zone = ExclusionZone("equatorial", [(-5.0, -5.0), (5.0, -5.0),
                                             (5.0, 5.0), (-5.0, 5.0)])
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = f.name

        try:
            env = GeoRegulatoryEnv(inner, exclusion_zones=[zone], audit_log_path=path)
            env.reset()
            env.step(0)
            if len(env.violation_log) > 0:
                with open(path) as f:
                    lines = f.readlines()
                assert len(lines) >= 1
                rec = json.loads(lines[0])
                assert "zone" in rec
        finally:
            os.unlink(path)

    def test_compliance_summary_has_geo_keys(self):
        from envs.regulatory_env import GeoRegulatoryEnv, ExclusionZone

        inner = _MultiSatInner()
        zone = ExclusionZone("z1", [(0.0, 0.0), (1.0, 0.0), (0.5, 1.0)])
        env = GeoRegulatoryEnv(inner, exclusion_zones=[zone])
        summary = env.compliance_summary()
        assert "geo_exclusion_violations" in summary
        assert "n_exclusion_zones" in summary
        assert summary["n_exclusion_zones"] == 1
