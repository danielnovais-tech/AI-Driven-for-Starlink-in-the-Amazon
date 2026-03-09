"""
Rain attenuation and channel model for LEO satellite links over the Amazon.

Implements:
    rain_specific_attenuation – ITU-R P.838-3 specific attenuation (dB/km).
    slant_path_attenuation    – Total rain attenuation along a slant path.
    ChannelModel              – Combined channel model (free-space + rain + foliage).

References:
    ITU-R P.838-3: Specific attenuation model for rain for use in
        prediction methods.
    ITU-R P.618-13: Propagation data and prediction methods required for
        the design of Earth-space telecommunication systems.
    ITU-R P.676-12: Attenuation by atmospheric gases.
"""

import math
from typing import Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# ITU-R P.838-3 coefficient tables (frequency in GHz)
# Each entry: (f_lo, f_hi, k_H, alpha_H, k_V, alpha_V)
# ---------------------------------------------------------------------------
_ITU838_TABLE: Tuple[Tuple, ...] = (
    (1,   2,   0.0000387, 0.912, 0.0000352, 0.880),
    (2,   4,   0.000154,  0.963, 0.000138,  0.923),
    (4,   6,   0.000650,  1.121, 0.000591,  1.075),
    (6,   8,   0.00175,   1.308, 0.00155,   1.265),
    (8,   10,  0.00301,   1.332, 0.00265,   1.312),
    (10,  12,  0.00454,   1.327, 0.00395,   1.310),
    (12,  15,  0.00701,   1.276, 0.00613,   1.266),
    (15,  20,  0.0151,    1.209, 0.0128,    1.200),
    (20,  25,  0.0366,    1.149, 0.0304,    1.128),
    (25,  30,  0.0751,    1.099, 0.0641,    1.065),
    (30,  35,  0.113,     1.061, 0.0980,    1.030),
    (35,  40,  0.167,     1.021, 0.143,     1.000),
    (40,  45,  0.224,     0.979, 0.190,     0.963),
    (45,  50,  0.310,     0.939, 0.264,     0.929),
    (50,  60,  0.454,     0.903, 0.393,     0.897),
    (60,  70,  0.540,     0.873, 0.479,     0.868),
    (70,  80,  0.620,     0.826, 0.553,     0.824),
    (80,  90,  0.708,     0.793, 0.637,     0.793),
    (90,  100, 0.784,     0.769, 0.715,     0.769),
    (100, 120, 0.873,     0.753, 0.806,     0.748),
    (120, 150, 1.050,     0.730, 0.968,     0.728),
    (150, 200, 1.310,     0.689, 1.210,     0.688),
    (200, 300, 1.770,     0.688, 1.630,     0.683),
    (300, 400, 2.430,     0.683, 2.260,     0.677),
)


def _get_itu838_coefficients(f_ghz: float, pol: str) -> Tuple[float, float]:
    """
    Return ITU-R P.838-3 (k, alpha) coefficients for the given frequency
    and polarisation.

    Args:
        f_ghz: Carrier frequency in GHz (1 – 400 GHz).
        pol:   Polarisation: 'H' (horizontal), 'V' (vertical), or
               'C' (circular, uses the average of H and V).

    Returns:
        Tuple (k, alpha) for use in γ = k · R^alpha.
    """
    pol = pol.upper()

    # Find bounding rows for interpolation
    lower = upper = None
    for row in _ITU838_TABLE:
        f_lo, f_hi, k_H, alpha_H, k_V, alpha_V = row
        if f_lo <= f_ghz < f_hi:
            lower = upper = row
            break
        if f_ghz < f_lo and lower is None:
            upper = row
        if f_ghz >= f_lo:
            lower = row

    if lower is None:
        lower = _ITU838_TABLE[0]
    if upper is None:
        upper = _ITU838_TABLE[-1]

    _, _, k_H_lo, alpha_H_lo, k_V_lo, alpha_V_lo = lower
    _, _, k_H_hi, alpha_H_hi, k_V_hi, alpha_V_hi = upper

    # Linear interpolation on frequency
    f_lo = lower[0]
    f_hi = upper[1]
    t = 0.0 if f_lo == f_hi else (f_ghz - f_lo) / (f_hi - f_lo)
    t = max(0.0, min(1.0, t))

    k_H = k_H_lo + t * (k_H_hi - k_H_lo)
    alpha_H = alpha_H_lo + t * (alpha_H_hi - alpha_H_lo)
    k_V = k_V_lo + t * (k_V_hi - k_V_lo)
    alpha_V = alpha_V_lo + t * (alpha_V_hi - alpha_V_lo)

    if pol == "H":
        return k_H, alpha_H
    if pol == "V":
        return k_V, alpha_V
    # Circular polarisation: average H and V
    k_C = (k_H + k_V) / 2.0
    alpha_C = (alpha_H + alpha_V) / 2.0
    return k_C, alpha_C


def rain_specific_attenuation(R: float, f: float, pol: str = "V") -> float:
    """
    Compute specific rain attenuation γ (dB/km) per ITU-R P.838-3.

    Args:
        R:   Rain rate (mm/h).  Must be ≥ 0.
        f:   Carrier frequency (GHz).
        pol: Polarisation: 'H', 'V', or 'C' (circular).

    Returns:
        Specific attenuation γ in dB/km.

    Reference: ITU-R P.838-3, Eq. (1): γ = k · R^α
    """
    if R <= 0:
        return 0.0
    k, alpha = _get_itu838_coefficients(f, pol)
    return k * (R ** alpha)


def slant_path_attenuation(
    R: float,
    elevation: float,
    f: float = 20.0,
    pol: str = "V",
    h_rain: float = 4.0,
    h_station: float = 0.05,
) -> float:
    """
    Compute total rain attenuation (dB) along the slant path to a LEO sat.

    Uses a simplified uniform rain-column model.  The effective path length
    through the rain layer is derived from the geometry of the rain height,
    station altitude and satellite elevation angle (ITU-R P.618-13, §2.2).

    Args:
        R:          Mean rain rate along the path (mm/h).
        elevation:  Satellite elevation angle (degrees, 0–90).
        f:          Carrier frequency (GHz).  Default 20 GHz (Ka-band).
        pol:        Polarisation ('H', 'V', 'C').
        h_rain:     Height of the rain layer (km).  Tropical default = 4 km.
        h_station:  Altitude of ground station (km).  Default 0.05 km.

    Returns:
        Total rain attenuation A_rain in dB (≥ 0).
    """
    if R <= 0 or elevation <= 0:
        return 0.0

    gamma = rain_specific_attenuation(R, f, pol)
    theta = math.radians(max(0.1, elevation))
    delta_h = max(0.0, h_rain - h_station)
    L_s = delta_h / math.sin(theta)
    return gamma * L_s


class ChannelModel:
    """
    Simplified link-budget channel model for a LEO satellite over the Amazon.

    Combines free-space path loss, rain attenuation (ITU-R P.838-3) and a
    foliage attenuation term.  The model is intended for simulation and
    offline training; it is not a replacement for a full ray-tracing tool.

    Args:
        frequency_ghz:   Carrier frequency in GHz (default 20 GHz, Ka-band).
        polarisation:    Polarisation string: 'H', 'V', or 'C'.
        tx_power_dbw:    Satellite transmit power (dBW).
        tx_gain_dbi:     Satellite antenna transmit gain (dBi).
        rx_gain_dbi:     Ground terminal receive gain (dBi).
        noise_temp_k:    System noise temperature (K).
        foliage_loss_db_per_unit: Foliage loss coefficient (dB per LAI unit).
        snr_threshold_db: SNR threshold below which an outage is declared.
    """

    SPEED_OF_LIGHT = 3e8  # m/s
    BOLTZMANN_DB = -228.6  # dBW/(Hz·K)

    def __init__(
        self,
        frequency_ghz: float = 20.0,
        polarisation: str = "V",
        tx_power_dbw: float = 10.0,
        tx_gain_dbi: float = 35.0,
        rx_gain_dbi: float = 32.0,
        noise_temp_k: float = 150.0,
        foliage_loss_db_per_unit: float = 0.5,
        snr_threshold_db: float = 5.0,
    ) -> None:
        self.frequency_ghz = frequency_ghz
        self.polarisation = polarisation
        self.tx_power_dbw = tx_power_dbw
        self.tx_gain_dbi = tx_gain_dbi
        self.rx_gain_dbi = rx_gain_dbi
        self.noise_temp_k = noise_temp_k
        self.foliage_loss_db_per_unit = foliage_loss_db_per_unit
        self.snr_threshold_db = snr_threshold_db

        freq_hz = frequency_ghz * 1e9
        self._wavelength = self.SPEED_OF_LIGHT / freq_hz

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def free_space_path_loss(self, distance_km: float) -> float:
        """
        Free-space path loss (dB) for a given slant-range distance.

        Args:
            distance_km: Slant-range distance in km.

        Returns:
            FSPL in dB.
        """
        d_m = distance_km * 1e3
        if d_m <= 0:
            return 0.0
        fspl = 20 * math.log10(4 * math.pi * d_m / self._wavelength)
        return fspl

    def compute_snr(
        self,
        sat_pos: Sequence[float],
        rain_rate: float,
        foliage_density: float,
        elevation: float = 45.0,
        bandwidth_hz: float = 500e6,
    ) -> float:
        """
        Estimate received SNR (dB) for the given satellite state and
        environmental conditions.

        Args:
            sat_pos:        Satellite position vector (x, y, z) in km from
                            the Earth centre.  The Euclidean norm is used as
                            an approximation of the slant range.
            rain_rate:      Surface rain rate (mm/h).
            foliage_density: Leaf area index (LAI) or normalised foliation
                            measure (unitless ≥ 0).
            elevation:      Satellite elevation angle (degrees).
            bandwidth_hz:   Receiver noise bandwidth (Hz).

        Returns:
            SNR in dB.
        """
        sat_pos_arr = np.asarray(sat_pos, dtype=float)
        distance_km = float(np.linalg.norm(sat_pos_arr))
        if distance_km <= 0:
            distance_km = 550.0  # default LEO altitude

        fspl = self.free_space_path_loss(distance_km)
        a_rain = slant_path_attenuation(
            rain_rate, elevation, self.frequency_ghz, self.polarisation
        )
        a_foliage = self.foliage_loss_db_per_unit * max(0.0, foliage_density)

        # Noise power: N = kTB  (in dBW)
        noise_power_dbw = (
            self.BOLTZMANN_DB
            + 10 * math.log10(self.noise_temp_k)
            + 10 * math.log10(bandwidth_hz)
        )

        received_power_dbw = (
            self.tx_power_dbw
            + self.tx_gain_dbi
            + self.rx_gain_dbi
            - fspl
            - a_rain
            - a_foliage
        )
        snr_db = received_power_dbw - noise_power_dbw
        return snr_db

    def compute_rssi(self, sat_pos: Sequence[float]) -> float:
        """
        Estimate received signal strength indicator (dBm) for a given
        satellite position.

        A simplified model that considers only free-space path loss and
        transmitter EIRP.

        Args:
            sat_pos: Satellite position vector (x, y, z) in km.

        Returns:
            RSSI in dBm.
        """
        sat_pos_arr = np.asarray(sat_pos, dtype=float)
        distance_km = float(np.linalg.norm(sat_pos_arr))
        if distance_km <= 0:
            distance_km = 550.0
        fspl = self.free_space_path_loss(distance_km)
        received_power_dbw = self.tx_power_dbw + self.tx_gain_dbi + self.rx_gain_dbi - fspl
        # Convert dBW → dBm
        return received_power_dbw + 30.0
