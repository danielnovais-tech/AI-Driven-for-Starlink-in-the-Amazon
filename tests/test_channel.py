"""
Tests for the ITU-R P.838-3 rain attenuation and ChannelModel.
"""

import math
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
from channel.rain_attenuation import (
    rain_specific_attenuation,
    slant_path_attenuation,
    ChannelModel,
)


class TestRainSpecificAttenuation:
    def test_zero_rain_rate(self):
        assert rain_specific_attenuation(0, 20, "V") == 0.0

    def test_negative_rain_rate(self):
        assert rain_specific_attenuation(-5, 20, "V") == 0.0

    def test_positive_attenuation(self):
        gamma = rain_specific_attenuation(50, 20, "V")
        assert gamma > 0.0

    def test_higher_rain_higher_attenuation(self):
        g1 = rain_specific_attenuation(10, 20, "V")
        g2 = rain_specific_attenuation(50, 20, "V")
        assert g2 > g1

    def test_horizontal_vs_vertical(self):
        g_v = rain_specific_attenuation(50, 20, "V")
        g_h = rain_specific_attenuation(50, 20, "H")
        # Both should be positive; H is typically slightly higher than V
        assert g_v > 0.0
        assert g_h > 0.0

    def test_circular_polarisation(self):
        g_c = rain_specific_attenuation(50, 20, "C")
        g_h = rain_specific_attenuation(50, 20, "H")
        g_v = rain_specific_attenuation(50, 20, "V")
        # Circular should be between H and V (average)
        assert min(g_h, g_v) <= g_c <= max(g_h, g_v) + 1e-9

    def test_ka_band_higher_than_ku(self):
        # Ka-band (30 GHz) should have higher attenuation than Ku-band (12 GHz)
        g_ku = rain_specific_attenuation(50, 12, "V")
        g_ka = rain_specific_attenuation(50, 30, "V")
        assert g_ka > g_ku

    def test_returns_float(self):
        result = rain_specific_attenuation(25, 20, "V")
        assert isinstance(result, float)


class TestSlantPathAttenuation:
    def test_zero_rain(self):
        assert slant_path_attenuation(0, 45) == 0.0

    def test_zero_elevation(self):
        # elevation <= 0 should return 0 to avoid division by zero
        assert slant_path_attenuation(50, 0) == 0.0

    def test_positive_attenuation(self):
        a = slant_path_attenuation(50, 45)
        assert a > 0.0

    def test_lower_elevation_higher_attenuation(self):
        a_high = slant_path_attenuation(50, 60)
        a_low = slant_path_attenuation(50, 30)
        assert a_low > a_high

    def test_higher_rain_higher_attenuation(self):
        a1 = slant_path_attenuation(10, 45)
        a2 = slant_path_attenuation(100, 45)
        assert a2 > a1


class TestChannelModel:
    def setup_method(self):
        self.model = ChannelModel(frequency_ghz=20.0, polarisation="V")

    def test_snr_returns_float(self):
        snr = self.model.compute_snr([0, 0, 550], rain_rate=0, foliage_density=0)
        assert isinstance(snr, float)

    def test_snr_decreases_with_rain(self):
        snr_dry = self.model.compute_snr([0, 0, 550], rain_rate=0, foliage_density=0)
        snr_rain = self.model.compute_snr([0, 0, 550], rain_rate=50, foliage_density=0)
        assert snr_dry > snr_rain

    def test_snr_decreases_with_foliage(self):
        snr_bare = self.model.compute_snr([0, 0, 550], rain_rate=0, foliage_density=0)
        snr_forest = self.model.compute_snr([0, 0, 550], rain_rate=0, foliage_density=5)
        assert snr_bare > snr_forest

    def test_rssi_returns_float(self):
        rssi = self.model.compute_rssi([0, 0, 550])
        assert isinstance(rssi, float)

    def test_closer_satellite_higher_snr(self):
        snr_close = self.model.compute_snr([0, 0, 400], rain_rate=0, foliage_density=0)
        snr_far = self.model.compute_snr([0, 0, 700], rain_rate=0, foliage_density=0)
        assert snr_close > snr_far

    def test_zero_distance_defaults_to_leo(self):
        # Should not raise and should return a finite value
        snr = self.model.compute_snr([0, 0, 0], rain_rate=0, foliage_density=0)
        assert math.isfinite(snr)
