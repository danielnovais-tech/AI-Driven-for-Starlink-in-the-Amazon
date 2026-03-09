"""
Multi-satellite LEO environment with predictive handover decisions.

The agent observes signal metrics for up to ``max_satellites`` visible
satellites and decides which one to use at each step.  If the chosen
satellite differs from the currently active one, a handover is executed
and a penalty is applied to the reward.

State (obs_dim = max_satellites × 4 + max_satellites):
    For each satellite i in [0, max_satellites):
        [SNR_i (dB), distance_i (km), elevation_i (deg), rain_rate_i (mm/h)]
    Followed by a one-hot vector indicating the currently active satellite.

Action:
    Discrete – index of the target satellite (0 … max_satellites-1).

Reward:
    throughput  −  λ · latency  −  μ · outage  −  ν · handover_penalty

Reference:
    Handover strategies for LEO constellations: Bhattacherjee & Singla (2019).
"""

from typing import Any, Dict, List, Optional, Tuple

import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces


# ---------------------------------------------------------------------------
# Simplified SNR → throughput mapping (Mbps)
# ---------------------------------------------------------------------------
def _snr_to_throughput(snr_db: float) -> float:
    if snr_db < -5.0:
        return 0.0
    if snr_db < 5.0:
        return 10.0
    if snr_db < 15.0:
        return 50.0
    return 100.0


class MultiSatelliteEnv(gym.Env):
    """
    MDP environment for multi-satellite LEO handover optimisation.

    The ``telemetry_stream`` must expose:
        - ``get_visible_satellites()`` → list of position arrays (x, y, z) in km.
        - ``ground_station_pos``       → position of the ground station (x, y, z) km.

    The ``radar_stream`` must expose:
        - ``get_at_location(pos)`` → rain rate (mm/h) at that position.

    The ``foliage_map`` must expose:
        - ``get_at_location(pos)`` → LAI at that position.

    Args:
        channel_model:     Instance of :class:`channel.ChannelModel`.
        telemetry_stream:  Telemetry interface (see above).
        radar_stream:      Radar interface (see above).
        foliage_map:       Foliage interface (see above).
        max_satellites:    Maximum number of simultaneously visible satellites.
        snr_threshold_db:  Outage SNR threshold (dB).
        handover_penalty:  Reward penalty applied when a handover occurs.
        lambda_latency:    Latency penalty weight.
        mu_outage:         Outage penalty weight.
    """

    metadata = {"render_modes": []}

    # Normalisation constants for the observation vector
    _OBS_MEAN_PER_SAT = np.array([15.0, 600.0, 45.0, 5.0], dtype=np.float32)
    _OBS_STD_PER_SAT = np.array([10.0, 200.0, 20.0, 10.0], dtype=np.float32)

    def __init__(
        self,
        channel_model,
        telemetry_stream,
        radar_stream,
        foliage_map,
        max_satellites: int = 5,
        snr_threshold_db: float = 5.0,
        handover_penalty: float = 1.0,
        lambda_latency: float = 0.1,
        mu_outage: float = 10.0,
    ) -> None:
        super().__init__()
        self.channel = channel_model
        self.telemetry = telemetry_stream
        self.radar = radar_stream
        self.foliage = foliage_map
        self.max_satellites = max_satellites
        self.snr_threshold = snr_threshold_db
        self.handover_penalty = handover_penalty
        self.lambda_latency = lambda_latency
        self.mu_outage = mu_outage

        # State: [SNR, dist, elev, rain] × max_sats + one-hot current
        obs_dim = max_satellites * 4 + max_satellites
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(max_satellites)

        # Normalisation constants for the full obs vector
        self._obs_mean = np.tile(self._OBS_MEAN_PER_SAT, max_satellites)
        self._obs_mean = np.concatenate([self._obs_mean, np.zeros(max_satellites, dtype=np.float32)])
        self._obs_std = np.tile(self._OBS_STD_PER_SAT, max_satellites)
        self._obs_std = np.concatenate([self._obs_std, np.ones(max_satellites, dtype=np.float32)])

        # Internal state
        self.visible_sats: List[np.ndarray] = []
        self.current_sat_idx: int = 0

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        self.visible_sats = [
            np.asarray(s, dtype=np.float64)
            for s in self.telemetry.get_visible_satellites()
        ]
        n = len(self.visible_sats)
        rng = np.random.default_rng(seed)
        self.current_sat_idx = int(rng.integers(0, max(1, n)))
        return self._build_obs(), {}

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        target_idx = int(action)

        # Update visible satellites (orbital movement)
        self.visible_sats = [
            np.asarray(s, dtype=np.float64)
            for s in self.telemetry.get_visible_satellites()
        ]
        n_vis = len(self.visible_sats)

        # If target is no longer visible, fall back to nearest
        if target_idx >= n_vis:
            target_idx = self._nearest_satellite_idx()

        handover = target_idx != self.current_sat_idx
        self.current_sat_idx = target_idx

        sat_pos = self.visible_sats[self.current_sat_idx]
        snr = self._compute_snr(sat_pos)
        throughput = _snr_to_throughput(snr)
        latency = self._compute_latency(sat_pos)
        outage = 1.0 if snr < self.snr_threshold else 0.0

        reward = (
            throughput
            - self.lambda_latency * latency
            - self.mu_outage * outage
            - (self.handover_penalty if handover else 0.0)
        )

        obs = self._build_obs()
        info: Dict[str, Any] = {
            "throughput": throughput,
            "latency": latency,
            "outage": outage,
            "handover": handover,
            "snr": snr,
        }
        return obs, float(reward), False, False, info

    def render(self) -> None:  # pragma: no cover
        pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_obs(self) -> np.ndarray:
        raw = np.zeros(self.max_satellites * 4, dtype=np.float32)
        for i in range(self.max_satellites):
            base = i * 4
            if i < len(self.visible_sats):
                sat = self.visible_sats[i]
                raw[base + 0] = self._compute_snr(sat)
                raw[base + 1] = float(np.linalg.norm(sat - self._gs_pos()))
                raw[base + 2] = self._elevation(sat)
                raw[base + 3] = float(self.radar.get_at_location(sat))
            else:
                raw[base + 0] = -10.0   # dummy SNR
                raw[base + 1] = 1000.0  # large distance
                raw[base + 2] = 0.0
                raw[base + 3] = 0.0

        one_hot = np.zeros(self.max_satellites, dtype=np.float32)
        if self.current_sat_idx < len(self.visible_sats):
            one_hot[self.current_sat_idx] = 1.0

        full = np.concatenate([raw, one_hot])
        # Normalise
        return ((full - self._obs_mean) / self._obs_std).astype(np.float32)

    def _gs_pos(self) -> np.ndarray:
        return np.asarray(self.telemetry.ground_station_pos, dtype=np.float64)

    def _compute_snr(self, sat_pos: np.ndarray) -> float:
        rain = float(self.radar.get_at_location(sat_pos))
        foliage = float(self.foliage.get_at_location(sat_pos))
        return self.channel.compute_snr(sat_pos, rain, foliage)

    def _elevation(self, sat_pos: np.ndarray) -> float:
        """
        Approximate elevation angle of ``sat_pos`` as seen from the ground
        station (degrees).
        """
        gs = self._gs_pos()
        vec = sat_pos - gs
        dist = float(np.linalg.norm(vec))
        if dist < 1e-6:
            return 90.0
        # Dot product of the look vector with the local vertical (gs direction)
        gs_norm = float(np.linalg.norm(gs))
        if gs_norm < 1e-6:
            return 90.0
        cos_zenith = float(np.dot(vec, gs)) / (dist * gs_norm)
        elev_rad = math.asin(min(1.0, max(-1.0, cos_zenith)))
        return math.degrees(elev_rad)

    def _compute_latency(self, sat_pos: np.ndarray) -> float:
        """One-way propagation latency (ms)."""
        dist_km = float(np.linalg.norm(sat_pos - self._gs_pos()))
        # speed of light = 3e5 km/s; divide distance by speed → seconds, then × 1e3 → ms
        return dist_km / 3e5 * 1e3

    def _nearest_satellite_idx(self) -> int:
        if not self.visible_sats:
            return 0
        gs = self._gs_pos()
        dists = [float(np.linalg.norm(s - gs)) for s in self.visible_sats]
        return int(np.argmin(dists))
