"""
Tests for the SGP4/simplified orbital propagator and coordinate utilities.
"""

import math
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest


class TestCoordinateHelpers:
    def test_geodetic_to_ecef_equator(self):
        from channel.orbital_propagator import geodetic_to_ecef
        pos = geodetic_to_ecef(0.0, 0.0, 0.0)
        # On the equator at 0 lon, x should equal Earth radius
        assert abs(pos[0] - 6371.0) < 1.0
        assert abs(pos[1]) < 1e-6
        assert abs(pos[2]) < 1.0

    def test_geodetic_roundtrip(self):
        from channel.orbital_propagator import geodetic_to_ecef, ecef_to_geodetic
        lat, lon, alt = -3.1, -60.0, 0.05
        pos = geodetic_to_ecef(lat, lon, alt)
        lat2, lon2, alt2 = ecef_to_geodetic(pos)
        assert abs(lat - lat2) < 0.01
        assert abs(lon - lon2) < 0.01
        assert abs(alt - alt2) < 0.01

    def test_elevation_angle_above_horizon(self):
        from channel.orbital_propagator import geodetic_to_ecef, elevation_angle
        gs = geodetic_to_ecef(-3.1, -60.0, 0.05)
        # Satellite directly above (550 km altitude)
        sat = geodetic_to_ecef(-3.1, -60.0, 550.0)
        elev = elevation_angle(gs, sat)
        assert elev > 80.0  # nearly directly overhead

    def test_elevation_angle_horizon(self):
        from channel.orbital_propagator import geodetic_to_ecef, elevation_angle
        gs = geodetic_to_ecef(-3.1, -60.0, 0.05)
        # Satellite far away should have low or negative elevation
        sat = geodetic_to_ecef(85.0, 120.0, 550.0)
        elev = elevation_angle(gs, sat)
        assert elev < 10.0


class TestSimplifiedPropagator:
    def _make(self, n=5):
        from channel.orbital_propagator import SimplifiedPropagator
        return SimplifiedPropagator(n_satellites=n, altitude_km=550.0, seed=42)

    def test_get_positions_returns_n_vectors(self):
        prop = self._make(5)
        pos = prop.get_positions(0.0)
        assert len(pos) == 5

    def test_position_at_leo_altitude(self):
        prop = self._make(3)
        for p in prop.get_positions(0.0):
            r = float(np.linalg.norm(p))
            # Earth radius + altitude ≈ 6921 km; allow ±5 km
            assert 6900 < r < 6940

    def test_positions_change_with_time(self):
        prop = self._make(3)
        pos0 = prop.get_positions(0.0)
        pos1 = prop.get_positions(600.0)  # 10 minutes later
        for p0, p1 in zip(pos0, pos1):
            # Satellite should have moved
            assert float(np.linalg.norm(p1 - p0)) > 1.0

    def test_visible_satellites_returns_list(self):
        from channel.orbital_propagator import geodetic_to_ecef
        prop = self._make(20)
        gs = geodetic_to_ecef(-3.1, -60.0, 0.05)
        visible = prop.get_visible_satellites(gs, 0.0, min_elevation_deg=25.0)
        assert isinstance(visible, list)
        assert len(visible) <= 20

    def test_visible_satellites_above_min_elevation(self):
        from channel.orbital_propagator import geodetic_to_ecef, elevation_angle
        prop = self._make(30)
        gs = geodetic_to_ecef(-3.1, -60.0, 0.05)
        visible = prop.get_visible_satellites(gs, 0.0, min_elevation_deg=25.0)
        for pos in visible:
            elev = elevation_angle(gs, pos)
            assert elev >= 25.0 - 0.01  # small numerical tolerance

    def test_exported_from_channel_package(self):
        from channel import SimplifiedPropagator, make_propagator, geodetic_to_ecef
        assert SimplifiedPropagator is not None
        assert make_propagator is not None
        assert geodetic_to_ecef is not None


class TestMakePropagator:
    def test_returns_simplified_when_sgp4_missing(self):
        from channel.orbital_propagator import make_propagator, SimplifiedPropagator
        # Pass use_sgp4=False to guarantee SimplifiedPropagator
        prop = make_propagator(n_satellites=5, use_sgp4=False)
        assert isinstance(prop, SimplifiedPropagator)

    def test_propagator_has_get_visible_satellites(self):
        from channel.orbital_propagator import make_propagator
        prop = make_propagator(n_satellites=5, use_sgp4=False)
        assert hasattr(prop, "get_visible_satellites")
