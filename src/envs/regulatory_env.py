"""
Regulatory compliance wrapper for the LEO beamforming environment.

Enforces power and beam-direction constraints derived from ITU-R and
national regulator (Anatel/FCC) rules.  The wrapper intercepts every
action before it is passed to the underlying environment and:

    1. **Clips transmit power** to the licensed EIRP ceiling.
    2. **Clips the elevation angle** to ensure the beam never points below
       a minimum elevation threshold (prevents interference with adjacent
       ground systems).
    3. **Logs a ``compliance_violation`` event** and applies a compliance
       penalty to the reward whenever the raw agent action would have
       violated a constraint.

This approach keeps regulatory logic decoupled from the agent and the
channel model, making it easy to update rules without re-training.

Usage::

    from envs.regulatory_env import RegulatoryEnv
    from envs.leo_beamforming_env import LEOBeamformingEnv

    inner = LEOBeamformingEnv(channel_model, telemetry, radar, foliage)
    env = RegulatoryEnv(inner, max_eirp_dbw=55.0, min_elevation_deg=20.0)
    obs, info = env.reset()

References:
    ITU-R S.580-6: Radiation diagrams for earth stations.
    ITU-R SM.1448: Determination of the coordination area.
    FCC Part 25: Satellite Communications Services.
    Anatel Resolução 723/2020: LEO satellite regulation for Brazil.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np


# ---------------------------------------------------------------------------
# Geospatial helpers
# ---------------------------------------------------------------------------

@dataclass
class ExclusionZone:
    """
    A geographic polygon within which beam pointing is prohibited.

    The polygon is defined by a list of (longitude_deg, latitude_deg) vertices
    (in degrees, WGS-84).  The polygon is assumed to be convex or simple
    (no self-intersections).  The point-in-polygon test uses the ray-casting
    algorithm.

    Args:
        name:     Human-readable name (e.g. "Atacama_VLBI_Site").
        vertices: Ordered list of (lon_deg, lat_deg) tuples forming the polygon.
        reason:   Optional description (regulatory basis, zone type, etc.).
    """

    name: str
    vertices: List[Tuple[float, float]]  # (lon_deg, lat_deg)
    reason: str = ""

    def contains(self, lon_deg: float, lat_deg: float) -> bool:
        """
        Test whether the point ``(lon_deg, lat_deg)`` lies inside the polygon.

        Uses the standard ray-casting (even-odd rule) algorithm.

        Args:
            lon_deg: Point longitude (degrees).
            lat_deg: Point latitude (degrees).

        Returns:
            ``True`` if the point is inside (or on the boundary of) the polygon.
        """
        n = len(self.vertices)
        inside = False
        j = n - 1
        x, y = lon_deg, lat_deg
        for i in range(n):
            xi, yi = self.vertices[i]
            xj, yj = self.vertices[j]
            # _RAY_CAST_EPSILON prevents division by zero when the ray passes
            # through a horizontal edge of the polygon.
            _RAY_CAST_EPSILON = 1e-12
            if ((yi > y) != (yj > y)) and (
                x < (xj - xi) * (y - yi) / (yj - yi + _RAY_CAST_EPSILON) + xi
            ):
                inside = not inside
            j = i
        return inside


class RegulatoryEnv(gym.Wrapper):
    """
    Gymnasium wrapper that enforces ITU-R / FCC / Anatel regulatory constraints.

    Wraps any :class:`gymnasium.Env` and applies compliance checks to each
    action before forwarding it to the underlying environment.  Actions that
    would violate constraints are clipped to the nearest valid value and a
    penalty is subtracted from the reward.

    The wrapper assumes a **continuous action space** with at least 2 elements:
        ``action[0]`` – delta_phase (radians): beam steering increment.
        ``action[1]`` – delta_power (0–1): normalised transmit power fraction.

    For **discrete action** environments the wrapper is a transparent pass-
    through (no clipping applied).

    Args:
        env:                 The inner :class:`gymnasium.Env` to wrap.
        max_eirp_dbw:        Maximum allowed EIRP in dBW (default 55 dBW).
                             Typical Ka-band LEO limit.
        min_elevation_deg:   Minimum elevation angle for the beam (degrees).
                             Below this angle the beam would illuminate the
                             horizon and risk interference.
        max_phase_rad:       Maximum allowed absolute beam phase (radians).
                             Corresponds to the maximum off-boresight angle
                             supported by the phased array.
        compliance_penalty:  Reward penalty applied per violated constraint
                             (default 5.0).
        tx_gain_dbi:         Satellite transmit antenna gain (dBi), used to
                             derive power from EIRP: P_tx = EIRP - G_tx.
        max_tx_power_dbw:    Maximum hardware transmit power (dBW), used to
                             scale the normalised ``delta_power`` action.
    """

    def __init__(
        self,
        env: gym.Env,
        max_eirp_dbw: float = 55.0,
        min_elevation_deg: float = 20.0,
        max_phase_rad: float = math.pi / 2.0,
        compliance_penalty: float = 5.0,
        tx_gain_dbi: float = 35.0,
        max_tx_power_dbw: float = 20.0,
    ) -> None:
        super().__init__(env)
        self.max_eirp_dbw = max_eirp_dbw
        self.min_elevation_deg = min_elevation_deg
        self.max_phase_rad = max_phase_rad
        self.compliance_penalty = compliance_penalty
        self.tx_gain_dbi = tx_gain_dbi
        self.max_tx_power_dbw = max_tx_power_dbw

        # Derived: max power fraction that keeps EIRP ≤ max_eirp_dbw
        # EIRP (dBW) = P_tx (dBW) + G_tx (dBi)
        # P_tx_max = max_eirp_dbw - tx_gain_dbi (dBW)
        self._max_allowed_power_dbw = max_eirp_dbw - tx_gain_dbi
        # Convert dBW limits to a normalised power fraction [0, 1]:
        #   P_linear (W) = 10 ^ (P_dBW / 10)
        # The fraction is: P_allowed_W / P_hardware_max_W
        self._max_power_fraction = min(
            1.0,
            10.0 ** (self._max_allowed_power_dbw / 10.0)
            / 10.0 ** (max_tx_power_dbw / 10.0),
        )

        # Phase limit: convert minimum elevation angle to phase increment limit
        # Elevation < min_elevation_deg  ↔  zenith angle > 90 - min_elevation_deg
        # For a 1-D phased array: max steering angle = 90 - min_elevation_deg
        self._max_steering_rad = math.radians(90.0 - min_elevation_deg)

        # Track cumulative violations for diagnostics
        self.total_violations: int = 0

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def step(
        self, action
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Apply regulatory clipping to ``action`` then step the inner env.

        Args:
            action: Raw action from the agent.

        Returns:
            Standard Gymnasium ``(obs, reward, terminated, truncated, info)``
            tuple, with ``info["compliance_violations"]`` set to the number
            of constraints that were violated.
        """
        is_continuous = isinstance(self.action_space, gym.spaces.Box)

        violations = 0
        if is_continuous:
            action = np.array(action, dtype=np.float32)
            action, violations = self._enforce_constraints(action)

        obs, reward, terminated, truncated, info = self.env.step(action)

        if violations > 0:
            reward -= self.compliance_penalty * violations
            self.total_violations += violations

        info["compliance_violations"] = violations
        info["total_compliance_violations"] = self.total_violations
        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _enforce_constraints(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, int]:
        """
        Clip the action to the feasible regulatory region.

        Args:
            action: Raw continuous action array.

        Returns:
            Tuple of (clipped_action, number_of_violated_constraints).
        """
        clipped = action.copy()
        violations = 0

        # ---- Constraint 1: transmit power ----
        if len(action) > 1:
            raw_power = float(action[1])
            clipped_power = float(np.clip(raw_power, 0.0, self._max_power_fraction))
            if abs(clipped_power - raw_power) > 1e-6:
                violations += 1
            clipped[1] = clipped_power

        # ---- Constraint 2: beam steering angle ----
        if len(action) > 0:
            raw_phase = float(action[0])
            clipped_phase = float(
                np.clip(raw_phase, -self._max_steering_rad, self._max_steering_rad)
            )
            if abs(clipped_phase - raw_phase) > 1e-6:
                violations += 1
            clipped[0] = clipped_phase

        return clipped, violations

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def compliance_summary(self) -> Dict[str, Any]:
        """
        Return a dictionary summarising regulatory compliance statistics.

        Returns:
            Dict with keys ``total_violations``, ``max_eirp_dbw``,
            ``min_elevation_deg``, ``max_power_fraction``.
        """
        return {
            "total_violations": self.total_violations,
            "max_eirp_dbw": self.max_eirp_dbw,
            "min_elevation_deg": self.min_elevation_deg,
            "max_power_fraction": self._max_power_fraction,
            "max_steering_rad": self._max_steering_rad,
        }


# ---------------------------------------------------------------------------
# Geospatial regulatory wrapper
# ---------------------------------------------------------------------------

class GeoRegulatoryEnv(RegulatoryEnv):
    """
    Regulatory wrapper with geospatial exclusion-zone enforcement.

    Extends :class:`RegulatoryEnv` with a list of geographic polygons
    (``ExclusionZone`` objects) that forbid beam pointing towards any satellite
    whose sub-satellite point (SSP) projects over a protected area.

    When the agent selects a satellite that would point the beam over an
    exclusion zone, the action is blocked and re-routed to the best compliant
    alternative (highest-elevation satellite outside all exclusion zones).
    If no compliant satellite exists, the original action passes through
    (unavoidable coverage gap).

    Every exclusion-zone violation is recorded in an in-memory audit log
    (:attr:`violation_log`) and optionally written to an append-only JSONL
    file for regulatory reporting.

    Args:
        env:              Inner Gymnasium environment.
        exclusion_zones:  List of :class:`ExclusionZone` polygons.
        audit_log_path:   Optional path to a JSONL audit log file.
                          Each line is a JSON object describing one violation.
        geo_penalty:      Additional reward penalty per geo-exclusion violation.
        **kwargs:         Forwarded to :class:`RegulatoryEnv`.
    """

    def __init__(
        self,
        env: gym.Env,
        exclusion_zones: Optional[List[ExclusionZone]] = None,
        audit_log_path: Optional[str] = None,
        geo_penalty: float = 10.0,
        **kwargs,
    ) -> None:
        super().__init__(env, **kwargs)
        self.exclusion_zones: List[ExclusionZone] = exclusion_zones or []
        self.audit_log_path = audit_log_path
        self.geo_penalty = geo_penalty
        self.violation_log: List[Dict[str, Any]] = []

        if audit_log_path:
            os.makedirs(os.path.dirname(audit_log_path) or ".", exist_ok=True)

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def step(
        self, action
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Enforce power/phase constraints (via parent) and geospatial exclusion.

        Returns:
            Standard Gymnasium tuple with additional ``info`` keys:
                ``geo_exclusion_violations``  – count of zone violations.
                ``geo_blocked_action``        – True if action was redirected.
        """
        # Let parent handle EIRP / phase clipping first
        is_continuous = isinstance(self.action_space, gym.spaces.Box)
        geo_violations = 0
        blocked = False

        if is_continuous:
            # Continuous: cannot determine satellite position directly; skip geo check
            pass
        else:
            # Discrete: action = satellite index; check SSP of that satellite
            action, geo_violations, blocked = self._geo_check_action(int(action))

        obs, reward, terminated, truncated, info = super().step(action)

        if geo_violations > 0:
            reward -= self.geo_penalty * geo_violations

        info["geo_exclusion_violations"] = geo_violations
        info["geo_blocked_action"] = blocked
        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_sat_lonlat(self, sat_idx: int) -> Optional[Tuple[float, float]]:
        """
        Return (lon_deg, lat_deg) of satellite ``sat_idx`` from the inner
        environment's visible satellite list.

        Falls back gracefully if the inner env does not expose
        ``visible_sats`` (returns ``None``).
        """
        inner = self.unwrapped
        visible = getattr(inner, "visible_sats", None)
        if visible is None or sat_idx >= len(visible):
            return None
        pos = np.asarray(visible[sat_idx], dtype=np.float64)
        r = float(np.linalg.norm(pos))
        if r < 1e-3:
            return None
        lat_deg = math.degrees(math.asin(pos[2] / r))
        lon_deg = math.degrees(math.atan2(pos[1], pos[0]))
        return lon_deg, lat_deg

    def _is_in_exclusion_zone(
        self, lon_deg: float, lat_deg: float
    ) -> Optional[str]:
        """
        Return the name of the first exclusion zone that contains the point,
        or ``None`` if no zone is matched.
        """
        for zone in self.exclusion_zones:
            if zone.contains(lon_deg, lat_deg):
                return zone.name
        return None

    def _geo_check_action(
        self, action: int
    ) -> Tuple[int, int, bool]:
        """
        Check whether ``action`` (satellite index) points the beam over an
        exclusion zone and, if so, redirect to the best compliant satellite.

        Returns:
            Tuple ``(final_action, n_geo_violations, was_blocked)``.
        """
        lonlat = self._get_sat_lonlat(action)
        if lonlat is None:
            return action, 0, False

        zone_name = self._is_in_exclusion_zone(*lonlat)
        if zone_name is None:
            return action, 0, False

        # Record violation
        event: Dict[str, Any] = {
            "timestamp": time.time(),
            "blocked_action": action,
            "zone": zone_name,
            "lon_deg": lonlat[0],
            "lat_deg": lonlat[1],
        }
        self.violation_log.append(event)
        if self.audit_log_path:
            try:
                with open(self.audit_log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(event) + "\n")
            except OSError:
                pass

        # Find best compliant alternative
        inner = self.unwrapped
        n_sats = getattr(inner, "max_satellites", 1)
        best_idx = action  # keep original if no alternative found
        best_elev = -float("inf")
        found_compliant = False

        for idx in range(n_sats):
            if idx == action:
                continue
            alt_lonlat = self._get_sat_lonlat(idx)
            if alt_lonlat is None:
                continue
            if self._is_in_exclusion_zone(*alt_lonlat) is not None:
                continue
            # Use elevation as a proxy for link quality
            visible = getattr(inner, "visible_sats", [])
            if idx < len(visible):
                pos = np.asarray(visible[idx], dtype=np.float64)
                gs = np.asarray(
                    getattr(inner.telemetry, "ground_station_pos",
                            np.array([0.0, 0.0, 6371.0])),
                    dtype=np.float64,
                )
                # Compute true elevation angle from ground station to satellite
                vec = pos - gs
                dist = float(np.linalg.norm(vec))
                gs_norm = float(np.linalg.norm(gs))
                if dist > 1e-6 and gs_norm > 1e-6:
                    cos_zenith = float(np.dot(vec, gs)) / (dist * gs_norm)
                    elev = math.degrees(math.asin(min(1.0, max(-1.0, cos_zenith))))
                else:
                    elev = 90.0
                if elev > best_elev:
                    best_elev = elev
                    best_idx = idx
                    found_compliant = True

        return best_idx, 1, True

    def compliance_summary(self) -> Dict[str, Any]:
        """
        Return a dictionary with both EIRP/phase and geospatial statistics.
        """
        summary = super().compliance_summary()
        summary["geo_exclusion_violations"] = len(self.violation_log)
        summary["n_exclusion_zones"] = len(self.exclusion_zones)
        return summary
