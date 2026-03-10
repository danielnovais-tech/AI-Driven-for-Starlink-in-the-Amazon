"""
Tests for ITU-R P.833-9 vegetation attenuation model.
"""

import math
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
from channel.vegetation_attenuation import (
    vegetation_specific_attenuation,
    vegetation_excess_attenuation,
    _MAX_VEG_ATTENUATION_DB,
)


class TestVegetationSpecificAttenuation:
    def test_tropical_positive(self):
        gamma = vegetation_specific_attenuation(20.0, "tropical")
        assert gamma > 0.0

    def test_temperate_positive(self):
        gamma = vegetation_specific_attenuation(20.0, "temperate")
        assert gamma > 0.0

    def test_light_positive(self):
        gamma = vegetation_specific_attenuation(20.0, "light")
        assert gamma > 0.0

    def test_tropical_denser_than_light(self):
        g_trop = vegetation_specific_attenuation(20.0, "tropical")
        g_light = vegetation_specific_attenuation(20.0, "light")
        assert g_trop > g_light

    def test_higher_frequency_higher_attenuation(self):
        g_low = vegetation_specific_attenuation(10.0, "tropical")
        g_high = vegetation_specific_attenuation(30.0, "tropical")
        assert g_high > g_low

    def test_zero_frequency_returns_zero(self):
        assert vegetation_specific_attenuation(0.0) == 0.0

    def test_negative_frequency_returns_zero(self):
        assert vegetation_specific_attenuation(-5.0) == 0.0

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError):
            vegetation_specific_attenuation(20.0, "unknown_type")

    def test_returns_float(self):
        assert isinstance(vegetation_specific_attenuation(20.0), float)


class TestVegetationExcessAttenuation:
    def test_positive_for_normal_inputs(self):
        A = vegetation_excess_attenuation(f=20, elevation=45, depth=15, vegetation_type="tropical")
        assert A > 0.0

    def test_zero_elevation_returns_zero(self):
        assert vegetation_excess_attenuation(f=20, elevation=0, depth=15) == 0.0

    def test_negative_elevation_returns_zero(self):
        assert vegetation_excess_attenuation(f=20, elevation=-10, depth=15) == 0.0

    def test_zero_depth_returns_zero(self):
        assert vegetation_excess_attenuation(f=20, elevation=45, depth=0) == 0.0

    def test_capped_at_max(self):
        # Very low elevation + thick forest should hit the cap
        A = vegetation_excess_attenuation(f=20, elevation=1, depth=100, vegetation_type="tropical")
        assert A <= _MAX_VEG_ATTENUATION_DB + 1e-9

    def test_lower_elevation_higher_attenuation(self):
        A_high = vegetation_excess_attenuation(f=20, elevation=60, depth=15)
        A_low = vegetation_excess_attenuation(f=20, elevation=20, depth=15)
        assert A_low > A_high

    def test_tropical_greater_than_light(self):
        A_trop = vegetation_excess_attenuation(f=20, elevation=45, depth=15, vegetation_type="tropical")
        A_light = vegetation_excess_attenuation(f=20, elevation=45, depth=15, vegetation_type="light")
        assert A_trop > A_light

    def test_value_range_typical(self):
        """Typical tropical scenario at 30 deg elevation should be between 0 and 30 dB (capped)."""
        A = vegetation_excess_attenuation(f=20, elevation=30, depth=15, vegetation_type="tropical")
        assert 0.0 < A <= 30.0

    def test_channel_init_exports(self):
        """Ensure vegetation functions are importable from the channel package."""
        from channel import vegetation_specific_attenuation as vsa
        from channel import vegetation_excess_attenuation as vea
        assert vsa is not None
        assert vea is not None
