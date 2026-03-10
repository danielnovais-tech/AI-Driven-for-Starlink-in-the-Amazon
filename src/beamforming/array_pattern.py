"""
Phased-array antenna pattern simulation.

Provides an accurate beam-gain model for linear and planar phased arrays,
replacing the simplified closed-form approximation used in the basic
LEOBeamformingEnv.

The array factor is computed from the superposition of element contributions
with phase shifts applied by the beamforming weights (conjugate steering
vectors).

References:
    Balanis (2016) - Antenna Theory: Analysis and Design, 4th ed., Ch. 6.
    Van Veen & Buckley (1988) - Beamforming: A versatile approach to spatial
        filtering.
"""

import math
from typing import Literal

import numpy as np


class PhasedArray:
    """
    Planar or linear phased-array antenna model.

    The array elements are arranged on a regular rectangular grid in the
    x-y plane.  The array factor is computed exactly from the steering
    vectors, enabling precise modelling of grating lobes and sidelobes.

    Args:
        frequency:        Carrier frequency in Hz (default 20 GHz).
        element_spacing:  Element spacing in wavelengths (default 0.5 = half-wavelength).
        n_elements:       Total number of elements.  For a planar array the
                          nearest perfect square is used as the grid dimension.
        array_type:       'linear' (1-D ULA) or 'planar' (2-D URA).
    """

    def __init__(
        self,
        frequency: float = 20e9,
        element_spacing: float = 0.5,
        n_elements: int = 64,
        array_type: str = "planar",
    ) -> None:
        self.freq = frequency
        self.wavelength = 3e8 / frequency
        self.d = element_spacing * self.wavelength
        self.n = n_elements
        self.array_type = array_type

        if array_type == "planar":
            self.nx = int(math.isqrt(n_elements))
            self.ny = self.nx
            # Actual populated elements (largest perfect square <= n_elements)
            self.n_actual = self.nx * self.ny
        else:
            self.nx = n_elements
            self.ny = 1
            self.n_actual = n_elements

        self.positions = self._compute_positions()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def steering_vector(self, theta: float, phi: float) -> np.ndarray:
        """
        Complex steering vector for direction (theta, phi).

        Args:
            theta: Zenith angle from boresight (rad). 0 = boresight.
            phi:   Azimuth angle (rad).

        Returns:
            Complex ndarray of shape (n_actual,), normalised so that the
            sum of squared magnitudes equals 1.
        """
        k = 2.0 * math.pi / self.wavelength
        u = np.array(
            [
                math.sin(theta) * math.cos(phi),
                math.sin(theta) * math.sin(phi),
                math.cos(theta),
            ]
        )
        phase_shifts = np.exp(1j * k * (self.positions @ u))
        return phase_shifts / math.sqrt(self.n_actual)

    def array_factor(
        self, theta: float, phi: float, theta0: float, phi0: float
    ) -> float:
        """
        Normalised array factor (linear power) for look direction (theta, phi)
        when the beam is steered to (theta0, phi0).

        Args:
            theta:  Observation zenith angle (rad).
            phi:    Observation azimuth angle (rad).
            theta0: Steering zenith angle (rad).
            phi0:   Steering azimuth angle (rad).

        Returns:
            Normalised power array factor in [0, 1].
        """
        w = self.steering_vector(theta0, phi0)
        v = self.steering_vector(theta, phi)
        af = float(abs(np.dot(w.conj(), v)) ** 2)
        return min(1.0, max(0.0, af))

    def gain_db(
        self, theta: float, phi: float, theta0: float, phi0: float
    ) -> float:
        """
        Total array gain in dBi for observation direction (theta, phi) steered
        towards (theta0, phi0).

        Assumes isotropic element pattern; the maximum directive gain of an
        N-element array is approximately 10*log10(N) dBi.

        Args:
            theta:  Observation zenith angle (rad).
            phi:    Observation azimuth angle (rad).
            theta0: Steering zenith angle (rad).
            phi0:   Steering azimuth angle (rad).

        Returns:
            Gain in dBi.
        """
        af = self.array_factor(theta, phi, theta0, phi0)
        gain_max_dbi = 10.0 * math.log10(self.n_actual)
        # Element pattern assumed isotropic; use small positive floor
        return gain_max_dbi + 10.0 * math.log10(max(af, 1e-12))

    def beam_gain_from_angles(
        self,
        sat_theta: float,
        sat_phi: float,
        steer_theta: float,
        steer_phi: float,
    ) -> float:
        """
        Convenience wrapper: gain (dBi) when steering towards (steer_theta,
        steer_phi) and the satellite is at (sat_theta, sat_phi).

        This is the method to call from the environment step function.

        Args:
            sat_theta:   Satellite zenith angle (rad) in the array frame.
            sat_phi:     Satellite azimuth angle (rad).
            steer_theta: Current steering zenith angle (rad).
            steer_phi:   Current steering azimuth angle (rad).

        Returns:
            Beam gain in dBi.
        """
        return self.gain_db(sat_theta, sat_phi, steer_theta, steer_phi)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_positions(self) -> np.ndarray:
        """
        Compute the Cartesian positions (x, y, 0) of all array elements.

        Returns:
            ndarray of shape (n_actual, 3).
        """
        pos = []
        for i in range(self.nx):
            for j in range(self.ny):
                x = (i - (self.nx - 1) / 2.0) * self.d
                y = (j - (self.ny - 1) / 2.0) * self.d
                pos.append([x, y, 0.0])
        return np.array(pos, dtype=np.float64)
