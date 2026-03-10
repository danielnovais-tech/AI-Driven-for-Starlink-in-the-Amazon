"""
Tests for SINR interference model in ChannelModel.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest


def _make_channel():
    from channel.rain_attenuation import ChannelModel
    return ChannelModel()


class TestSINR:
    def test_sinr_no_interference_equals_snr(self):
        """With no interferers, SINR should be close to SNR."""
        ch = _make_channel()
        sat_pos = np.array([0.0, 0.0, 6921.0])
        snr = ch.compute_snr(sat_pos, rain_rate=0.0, foliage_density=0.0, elevation=45.0)
        sinr = ch.compute_sinr(sat_pos, rain_rate=0.0, foliage_density=0.0,
                               interfering_positions=[], elevation=45.0)
        assert abs(snr - sinr) < 0.1

    def test_sinr_decreases_with_more_interference(self):
        ch = _make_channel()
        sat_pos = np.array([0.0, 0.0, 6921.0])
        interferer = np.array([100.0, 0.0, 6921.0])

        sinr_0 = ch.compute_sinr(sat_pos, 0.0, 0.0, interfering_positions=[])
        sinr_1 = ch.compute_sinr(sat_pos, 0.0, 0.0, interfering_positions=[interferer])
        sinr_3 = ch.compute_sinr(sat_pos, 0.0, 0.0,
                                  interfering_positions=[interferer, interferer, interferer])
        assert sinr_1 < sinr_0
        assert sinr_3 < sinr_1

    def test_sinr_higher_isolation_higher_sinr(self):
        ch = _make_channel()
        sat_pos = np.array([0.0, 0.0, 6921.0])
        interferer = np.array([50.0, 0.0, 6921.0])

        sinr_lo = ch.compute_sinr(sat_pos, 0.0, 0.0,
                                   interfering_positions=[interferer],
                                   beam_isolation_db=10.0)
        sinr_hi = ch.compute_sinr(sat_pos, 0.0, 0.0,
                                   interfering_positions=[interferer],
                                   beam_isolation_db=30.0)
        assert sinr_hi > sinr_lo

    def test_sinr_returns_float(self):
        ch = _make_channel()
        sat_pos = np.array([0.0, 0.0, 6921.0])
        sinr = ch.compute_sinr(sat_pos, 5.0, 1.0)
        assert isinstance(sinr, float)

    def test_sinr_decreases_with_rain(self):
        ch = _make_channel()
        sat_pos = np.array([0.0, 0.0, 6921.0])
        sinr_clear = ch.compute_sinr(sat_pos, 0.0, 0.0, elevation=45.0)
        sinr_rain = ch.compute_sinr(sat_pos, 50.0, 0.0, elevation=45.0)
        assert sinr_clear > sinr_rain

    def test_sinr_with_per_interferer_rain(self):
        ch = _make_channel()
        sat_pos = np.array([0.0, 0.0, 6921.0])
        interferers = [np.array([100.0, 0.0, 6921.0])]
        sinr = ch.compute_sinr(
            sat_pos, 10.0, 0.0,
            interfering_positions=interferers,
            interferer_rain_rates=[30.0],
            interferer_foliage=[1.0],
        )
        assert isinstance(sinr, float)
