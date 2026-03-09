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

import math
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np


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
