"""
Tests for PhasedArray antenna pattern model.
"""

import math
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest


class TestPhasedArray:
    def _make(self, n_elements=64, array_type="planar"):
        from beamforming.array_pattern import PhasedArray
        return PhasedArray(frequency=20e9, element_spacing=0.5,
                           n_elements=n_elements, array_type=array_type)

    def test_import(self):
        from beamforming.array_pattern import PhasedArray
        assert PhasedArray is not None

    def test_exported_from_package(self):
        from beamforming import PhasedArray
        assert PhasedArray is not None

    def test_planar_positions_shape(self):
        arr = self._make(n_elements=64, array_type="planar")
        # 8x8 = 64 elements, each with (x, y, z)
        assert arr.positions.shape == (64, 3)

    def test_linear_positions_shape(self):
        arr = self._make(n_elements=16, array_type="linear")
        assert arr.positions.shape == (16, 3)

    def test_steering_vector_normalised(self):
        arr = self._make()
        sv = arr.steering_vector(0.0, 0.0)
        assert abs(float(np.sum(np.abs(sv) ** 2)) - 1.0) < 1e-6

    def test_boresight_array_factor_is_one(self):
        arr = self._make()
        af = arr.array_factor(0.0, 0.0, 0.0, 0.0)
        assert abs(af - 1.0) < 1e-6

    def test_array_factor_off_boresight_less_than_one(self):
        arr = self._make()
        af = arr.array_factor(math.pi / 4, 0.0, 0.0, 0.0)
        assert af < 1.0

    def test_gain_db_boresight_high(self):
        arr = self._make(n_elements=64)
        gain = arr.gain_db(0.0, 0.0, 0.0, 0.0)
        # Maximum gain should be ~10*log10(64) = ~18 dBi
        assert 17.0 < gain < 20.0

    def test_gain_db_off_axis_less_than_boresight(self):
        arr = self._make()
        gain_bore = arr.gain_db(0.0, 0.0, 0.0, 0.0)
        gain_off = arr.gain_db(math.pi / 6, 0.0, 0.0, 0.0)
        assert gain_bore > gain_off

    def test_steering_matches_observation_gives_max(self):
        """When steering direction equals observation direction, AF should be very close to 1."""
        arr = self._make()
        theta = math.pi / 6
        phi = math.pi / 4
        af = arr.array_factor(theta, phi, theta, phi)
        assert abs(af - 1.0) < 1e-4

    def test_beam_gain_from_angles(self):
        arr = self._make()
        # Same direction → max gain
        g = arr.beam_gain_from_angles(0.0, 0.0, 0.0, 0.0)
        assert g > 15.0

    def test_linear_array(self):
        arr = self._make(n_elements=16, array_type="linear")
        af = arr.array_factor(0.0, 0.0, 0.0, 0.0)
        assert abs(af - 1.0) < 1e-6

    def test_wavelength_computed_correctly(self):
        arr = self._make()
        expected_wavelength = 3e8 / 20e9
        assert abs(arr.wavelength - expected_wavelength) < 1e-12

    def test_non_square_planar_uses_isqrt(self):
        """n_elements=50 → 7x7=49 elements."""
        arr = self._make(n_elements=50, array_type="planar")
        assert arr.n_actual == 49
        assert arr.positions.shape == (49, 3)
