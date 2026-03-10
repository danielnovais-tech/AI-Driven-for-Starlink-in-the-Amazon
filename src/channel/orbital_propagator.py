"""
SGP4-based orbital propagator for LEO satellite position estimation.

Provides two back-ends:
    1. ``Sgp4Propagator`` – uses the ``sgp4`` Python library (pip install sgp4)
       with real or synthetic Two-Line Element (TLE) sets.
    2. ``SimplifiedPropagator`` – pure-Python circular-orbit approximation
       requiring no extra dependencies; used automatically when ``sgp4``
       is not installed.

The public interface is the same for both back-ends, making it easy to
swap between them in ``MultiSatelliteEnv``.

Coordinate system:
    All position vectors are returned in the **TEME** (True Equator Mean
    Equinox) frame expressed in **kilometres**, consistent with the rest of
    the codebase.  Ground-station positions are assumed to be in the same
    frame and should be converted from geodetic (lat/lon/alt) using
    :func:`geodetic_to_ecef` before use.

Reference:
    Vallado et al. (2006) – "Revisiting Spacetrack Report #3."
    Hoots & Roehrich (1980) – "Spacetrack Report #3: Models for
        Propagation of NORAD Element Sets."
"""

from __future__ import annotations

import math
import time
from datetime import datetime, timezone
from typing import List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Optional SGP4 import
# ---------------------------------------------------------------------------
try:
    from sgp4.api import Satrec, jday  # type: ignore
    _SGP4_AVAILABLE = True
except ImportError:  # pragma: no cover
    _SGP4_AVAILABLE = False


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------

_EARTH_RADIUS_KM = 6371.0  # mean equatorial radius


def geodetic_to_ecef(lat_deg: float, lon_deg: float, alt_km: float = 0.0) -> np.ndarray:
    """
    Convert geodetic (lat, lon, alt) to ECEF Cartesian coordinates (km).

    Uses a spherical Earth approximation sufficient for link-budget
    calculations.

    Args:
        lat_deg: Geodetic latitude (degrees, −90 … +90).
        lon_deg: Longitude (degrees, −180 … +180).
        alt_km:  Altitude above Earth's surface (km).

    Returns:
        Numpy array [x, y, z] in km.
    """
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    r = _EARTH_RADIUS_KM + alt_km
    x = r * math.cos(lat) * math.cos(lon)
    y = r * math.cos(lat) * math.sin(lon)
    z = r * math.sin(lat)
    return np.array([x, y, z], dtype=np.float64)


def ecef_to_geodetic(pos_km: np.ndarray) -> Tuple[float, float, float]:
    """
    Convert ECEF Cartesian position (km) to (lat_deg, lon_deg, alt_km).

    Uses spherical Earth approximation.

    Args:
        pos_km: Array [x, y, z] in km.

    Returns:
        (latitude_deg, longitude_deg, altitude_km).
    """
    x, y, z = pos_km
    r = float(np.linalg.norm(pos_km))
    lat = math.degrees(math.asin(z / r)) if r > 0 else 0.0
    lon = math.degrees(math.atan2(y, x))
    alt = r - _EARTH_RADIUS_KM
    return lat, lon, alt


def elevation_angle(gs_pos: np.ndarray, sat_pos: np.ndarray) -> float:
    """
    Compute the elevation angle (degrees) of ``sat_pos`` as seen from
    ``gs_pos`` using the dot-product formula.

    Args:
        gs_pos:  Ground-station ECEF position (km).
        sat_pos: Satellite ECEF position (km).

    Returns:
        Elevation angle in degrees (−90 … +90).  Positive values indicate
        the satellite is above the local horizon.
    """
    vec = sat_pos - gs_pos
    dist = float(np.linalg.norm(vec))
    gs_norm = float(np.linalg.norm(gs_pos))
    if dist < 1e-6 or gs_norm < 1e-6:
        return 90.0
    # The angle between the look vector and the local vertical equals the
    # complement of the zenith angle.
    cos_zenith = float(np.dot(vec, gs_pos)) / (dist * gs_norm)
    elev_rad = math.asin(min(1.0, max(-1.0, cos_zenith)))
    return math.degrees(elev_rad)


# ---------------------------------------------------------------------------
# TLE helper: synthetic Starlink-like TLE generator
# ---------------------------------------------------------------------------

# Representative Starlink shell parameters (shell 1, 550 km, 53 deg inclination)
_STARLINK_SHELL = {
    "altitude_km": 550.0,
    "inclination_deg": 53.0,
    "n_planes": 72,
    "sats_per_plane": 22,
}

# Tropical Amazon ground station (Manaus, Brazil)
AMAZON_GS_LAT_DEG = -3.1
AMAZON_GS_LON_DEG = -60.0
AMAZON_GS_ALT_KM = 0.05


def _synthetic_tle(
    sat_id: int,
    altitude_km: float = 550.0,
    inclination_deg: float = 53.0,
    raan_deg: float = 0.0,
    mean_anomaly_deg: float = 0.0,
) -> Tuple[str, str]:
    """
    Generate a synthetic TLE pair for a circular LEO satellite.

    This is used as a fallback when real Starlink TLEs are not available.
    The TLE is formatted according to the standard, but the checksum values
    are not validated (acceptable for simulation).

    Args:
        sat_id:            Unique integer identifier (0 – 9999).
        altitude_km:       Circular orbit altitude (km).
        inclination_deg:   Orbital inclination (degrees).
        raan_deg:          Right ascension of ascending node (degrees).
        mean_anomaly_deg:  Mean anomaly at epoch (degrees).

    Returns:
        Tuple of (TLE line 1, TLE line 2) strings.
    """
    earth_radius_km = 6378.137
    mu_km3_s2 = 398600.4418
    a_km = earth_radius_km + altitude_km
    n_rev_day = 86400.0 / (2.0 * math.pi * math.sqrt(a_km ** 3 / mu_km3_s2))

    epoch = "24001.00000000"  # 2024 day 1
    line1 = (
        f"1 {sat_id:05d}U 24001A   {epoch}  .00001103  00000-0  14476-4 0  9990"
    )
    line2 = (
        f"2 {sat_id:05d} {inclination_deg:8.4f} {raan_deg:8.4f} 0001480 "
        f"  0.0000 {mean_anomaly_deg:8.4f} {n_rev_day:11.8f}    10"
    )
    return line1, line2


# ---------------------------------------------------------------------------
# Propagator implementations
# ---------------------------------------------------------------------------

class SimplifiedPropagator:
    """
    Circular-orbit satellite propagator (no ``sgp4`` dependency required).

    Models a constellation of circular LEO satellites at a fixed altitude
    and inclination.  The satellite positions advance uniformly around their
    orbits at each call to :meth:`get_positions`.

    Args:
        n_satellites:    Number of satellites in the constellation.
        altitude_km:     Orbital altitude (km).
        inclination_deg: Orbital inclination (degrees).
        seed:            Random seed for initial orbital phase distribution.
    """

    def __init__(
        self,
        n_satellites: int = 10,
        altitude_km: float = 550.0,
        inclination_deg: float = 53.0,
        seed: int = 0,
    ) -> None:
        self.n_sats = n_satellites
        self.altitude_km = altitude_km
        self.inclination_deg = inclination_deg

        mu_km3_s2 = 398600.4418
        earth_radius_km = 6378.137
        a_km = earth_radius_km + altitude_km
        # Mean motion (rad/s)
        self._omega = math.sqrt(mu_km3_s2 / a_km ** 3)
        self._a = a_km

        rng = np.random.default_rng(seed)
        # Distribute initial mean anomalies evenly + small random offset
        base = np.linspace(0.0, 2.0 * math.pi, n_satellites, endpoint=False)
        self._mean_anomaly_0 = base + rng.uniform(-0.1, 0.1, n_satellites)
        # Distribute RAAN (right ascension of ascending node) across planes
        n_planes = max(1, n_satellites // 5)
        self._raan = np.array(
            [2.0 * math.pi * (i // (n_satellites // n_planes)) / n_planes
             for i in range(n_satellites)]
        )
        # Reference epoch (seconds since start of simulation)
        self._t0 = 0.0

    def get_positions(self, t_sec: float) -> List[np.ndarray]:
        """
        Compute satellite ECEF positions (km) at time ``t_sec``.

        Args:
            t_sec: Simulation time in seconds.

        Returns:
            List of (3,) numpy arrays [x, y, z] in km.
        """
        dt = t_sec - self._t0
        inc = math.radians(self.inclination_deg)
        positions = []
        for i in range(self.n_sats):
            M = self._mean_anomaly_0[i] + self._omega * dt  # mean anomaly
            # Circular orbit: true anomaly == mean anomaly
            u = M  # argument of latitude
            raan = self._raan[i]
            # Position in orbital plane
            r = self._a
            x_orb = r * math.cos(u)
            y_orb = r * math.sin(u)
            # Rotate to ECEF
            x = (math.cos(raan) * x_orb
                 - math.sin(raan) * math.cos(inc) * y_orb)
            y = (math.sin(raan) * x_orb
                 + math.cos(raan) * math.cos(inc) * y_orb)
            z = math.sin(inc) * y_orb
            positions.append(np.array([x, y, z], dtype=np.float64))
        return positions

    def get_visible_satellites(
        self,
        gs_pos: np.ndarray,
        t_sec: float,
        min_elevation_deg: float = 25.0,
    ) -> List[np.ndarray]:
        """
        Return positions of satellites visible from ``gs_pos`` at ``t_sec``.

        Args:
            gs_pos:            Ground-station ECEF position (km).
            t_sec:             Current simulation time (s).
            min_elevation_deg: Minimum elevation angle threshold (degrees).

        Returns:
            List of satellite position arrays (km) with elevation ≥ threshold.
        """
        all_pos = self.get_positions(t_sec)
        visible = []
        for pos in all_pos:
            elev = elevation_angle(gs_pos, pos)
            if elev >= min_elevation_deg:
                visible.append(pos)
        return visible


class Sgp4Propagator:
    """
    SGP4-based orbital propagator backed by the ``sgp4`` Python library.

    Propagates one or more satellite TLE sets and returns TEME-frame
    positions in km.

    Args:
        tle_pairs:  List of (line1, line2) TLE string pairs.

    Raises:
        ImportError: If the ``sgp4`` package is not installed.
    """

    def __init__(self, tle_pairs: List[Tuple[str, str]]) -> None:
        if not _SGP4_AVAILABLE:  # pragma: no cover
            raise ImportError(
                "sgp4 is required for Sgp4Propagator. "
                "Install it with: pip install sgp4"
            )
        self._satellites = []
        for line1, line2 in tle_pairs:
            sat = Satrec.twoline2rv(line1, line2)
            self._satellites.append(sat)

    def get_positions(self, t_sec: float) -> List[np.ndarray]:
        """
        Propagate all satellites to ``t_sec`` seconds since Unix epoch.

        Args:
            t_sec: Unix timestamp (seconds).

        Returns:
            List of (3,) numpy arrays [x, y, z] in km (TEME frame).
            Failed propagations are replaced with the zero vector.
        """
        dt = datetime.fromtimestamp(t_sec, tz=timezone.utc)
        jd, fr = jday(dt.year, dt.month, dt.day,
                      dt.hour, dt.minute, dt.second + dt.microsecond / 1e6)
        positions = []
        for sat in self._satellites:
            e, pos, _ = sat.sgp4(jd, fr)
            if e == 0:
                positions.append(np.array(pos, dtype=np.float64))
            else:
                positions.append(np.zeros(3, dtype=np.float64))  # propagation error
        return positions

    def get_visible_satellites(
        self,
        gs_pos: np.ndarray,
        t_sec: float,
        min_elevation_deg: float = 25.0,
    ) -> List[np.ndarray]:
        """
        Return visible satellite positions at ``t_sec``.

        Args:
            gs_pos:            Ground-station ECEF position (km).
            t_sec:             Unix timestamp (seconds).
            min_elevation_deg: Minimum elevation threshold (degrees).

        Returns:
            Filtered list of satellite position arrays.
        """
        all_pos = self.get_positions(t_sec)
        visible = []
        for pos in all_pos:
            if float(np.linalg.norm(pos)) < 1e-3:
                continue  # skip failed propagations
            elev = elevation_angle(gs_pos, pos)
            if elev >= min_elevation_deg:
                visible.append(pos)
        return visible


def make_propagator(
    n_satellites: int = 10,
    altitude_km: float = 550.0,
    inclination_deg: float = 53.0,
    use_sgp4: bool = True,
    seed: int = 0,
) -> "SimplifiedPropagator | Sgp4Propagator":
    """
    Factory function that returns the best available propagator.

    If ``use_sgp4`` is True and the ``sgp4`` library is installed, a
    :class:`Sgp4Propagator` is returned with synthetic TLEs.  Otherwise a
    :class:`SimplifiedPropagator` is used.

    Args:
        n_satellites:    Number of satellites to model.
        altitude_km:     Orbital altitude (km).
        inclination_deg: Inclination (degrees).
        use_sgp4:        Whether to prefer the SGP4 back-end.
        seed:            Random seed for initial orbital phases.

    Returns:
        A propagator instance with :meth:`get_visible_satellites` method.
    """
    if use_sgp4 and _SGP4_AVAILABLE:
        rng = np.random.default_rng(seed)
        raans = np.linspace(0.0, 360.0, n_satellites, endpoint=False)
        anomalies = rng.uniform(0.0, 360.0, n_satellites)
        tle_pairs = [
            _synthetic_tle(
                sat_id=i + 1,
                altitude_km=altitude_km,
                inclination_deg=inclination_deg,
                raan_deg=float(raans[i]),
                mean_anomaly_deg=float(anomalies[i]),
            )
            for i in range(n_satellites)
        ]
        return Sgp4Propagator(tle_pairs)
    return SimplifiedPropagator(
        n_satellites=n_satellites,
        altitude_km=altitude_km,
        inclination_deg=inclination_deg,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# High-level telemetry adapter for MultiSatelliteEnv
# ---------------------------------------------------------------------------

class StarlinkConstellationTelemetry:
    """
    High-level telemetry adapter wrapping a propagator for use with
    :class:`~envs.multi_satellite_env.MultiSatelliteEnv`.

    Exposes the ``get_visible_satellites()`` and ``ground_station_pos``
    attributes expected by the environment, advancing the simulated time at
    each call to simulate orbital motion.

    Supports constellations of 5 to 1 584 satellites (matching the first
    Starlink shell of 72 planes × 22 satellites).  For benchmarks requiring
    100+ satellites, pass ``n_satellites=100`` or more.

    Args:
        n_satellites:    Total constellation size (default 72, Starlink-like).
        altitude_km:     Orbital altitude (km).
        inclination_deg: Inclination (degrees).
        gs_lat_deg:      Ground-station latitude (degrees).
        gs_lon_deg:      Ground-station longitude (degrees).
        gs_alt_km:       Ground-station altitude (km above sea level).
        min_elevation_deg: Minimum elevation for visibility (degrees).
        time_step_s:     Simulated time advanced per ``get_visible_satellites``
                         call (seconds, default 0.5 = one control step).
        seed:            Random seed for initial orbital phases.
        use_sgp4:        Whether to use the SGP4 back-end (requires ``sgp4``
                         package; falls back to simplified propagator).
    """

    def __init__(
        self,
        n_satellites: int = 72,
        altitude_km: float = 550.0,
        inclination_deg: float = 53.0,
        gs_lat_deg: float = AMAZON_GS_LAT_DEG,
        gs_lon_deg: float = AMAZON_GS_LON_DEG,
        gs_alt_km: float = AMAZON_GS_ALT_KM,
        min_elevation_deg: float = 25.0,
        time_step_s: float = 0.5,
        seed: int = 0,
        use_sgp4: bool = True,
    ) -> None:
        self.n_satellites = n_satellites
        self.min_elevation_deg = min_elevation_deg
        self.time_step_s = time_step_s

        self.ground_station_pos = geodetic_to_ecef(gs_lat_deg, gs_lon_deg, gs_alt_km)
        self._propagator = make_propagator(
            n_satellites=n_satellites,
            altitude_km=altitude_km,
            inclination_deg=inclination_deg,
            use_sgp4=use_sgp4,
            seed=seed,
        )
        # Initialise to current wall-clock Unix timestamp so that Sgp4Propagator
        # (which interprets t_sec as a Unix timestamp) starts at a realistic epoch.
        # For SimplifiedPropagator, call reset(t_sec=0.0) to start at simulation
        # time zero if wall-clock alignment is not required.
        self._current_time: float = time.time()

    def get_visible_satellites(self) -> List[np.ndarray]:
        """
        Return positions of currently visible satellites and advance the
        simulated time by ``time_step_s``.

        Returns:
            List of (3,) numpy position arrays in km (ECEF/TEME).
        """
        visible = self._propagator.get_visible_satellites(
            gs_pos=self.ground_station_pos,
            t_sec=self._current_time,
            min_elevation_deg=self.min_elevation_deg,
        )
        self._current_time += self.time_step_s
        return visible

    def reset(self, t_sec: Optional[float] = None, seed: Optional[int] = None) -> None:
        """
        Reset the simulation clock.

        Args:
            t_sec: New simulation start time (Unix timestamp).
                   Defaults to the current wall-clock time.
            seed:  Unused (accepted for API compatibility).
        """
        self._current_time = t_sec if t_sec is not None else time.time()

