"""
Gymnasium MDP environment for LEO satellite beamforming optimisation.

State space (7 dimensions, z-score normalised):
    [SNR (dB), RSSI (dBm), sat_x (km), sat_y (km), sat_z (km),
     rain_rate (mm/h), foliage_density (LAI)]

Action space (continuous Box, 4 dimensions):
    [delta_phase (rad), delta_power (normalised 0–1),
     mcs_index (0–4), rb_allocation (0–100)]

Reward:
    throughput  −  λ · latency  −  μ · outage_penalty

Reference: Schulman et al. (2017) PPO; Mnih et al. (2015) DQN.
"""

from typing import Any, Dict, Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces


# ---------------------------------------------------------------------------
# MCS lookup table: (min_snr_db, bits_per_symbol, code_rate)
# Based on 3GPP NR MCS Table 1 (TS 38.214 Table 5.1.3.1-1)
# ---------------------------------------------------------------------------
_MCS_TABLE = [
    (0.0,  2, 0.12),   # QPSK  1/8
    (4.0,  2, 0.30),   # QPSK  1/3
    (6.5,  4, 0.44),   # 16QAM 4/9
    (11.0, 4, 0.60),   # 16QAM 2/3
    (14.5, 6, 0.75),   # 64QAM 3/4
]

_SUBCARRIER_SPACING_HZ = 30e3        # 30 kHz (NR μ=1)
_RB_SUBCARRIERS = 12                 # 12 subcarriers per resource block
_SYMBOLS_PER_SLOT = 14               # OFDM symbols per slot
_SLOTS_PER_SECOND = 2000             # 2000 slots/s for 30 kHz SCS
_RB_BANDWIDTH_HZ = _RB_SUBCARRIERS * _SUBCARRIER_SPACING_HZ


class LEOBeamformingEnv(gym.Env):
    """
    Simulation environment for adaptive beamforming on LEO satellites.

    The environment models the link between a single Starlink-Gen2 satellite
    and a ground station in the Brazilian Amazon, including:
      - Orbital dynamics (simplified circular orbit at ~550 km).
      - Rain attenuation per ITU-R P.838-3 via the supplied channel model.
      - Foliage shadowing.
      - Phased-array beam steering (phase and power adjustment).
      - Adaptive Modulation and Coding (AMC) via an MCS lookup table.
      - Resource block (RB) allocation.

    Args:
        channel_model:   An instance of :class:`channel.ChannelModel`.
        telemetry_stream: Object with ``get_current_position()``,
                          ``get_next_position()``, ``get_current_snr()``,
                          and ``get_current_rssi()`` methods.
        radar_stream:    Object with ``get_at_location(sat_pos)`` method
                         returning rain rate (mm/h).
        foliage_map:     Object with ``get_at_location(sat_pos)`` method
                         returning foliage density (LAI, dimensionless).
        state_mean:      Optional pre-computed state mean for normalisation.
        state_std:       Optional pre-computed state std for normalisation.
        snr_threshold_db: SNR below which an outage penalty is applied.
        lambda_latency:  Latency penalty weight in the reward function.
        mu_outage:       Outage penalty weight in the reward function.
    """

    metadata = {"render_modes": []}

    # Default normalisation constants (approximate operating range)
    _DEFAULT_MEAN = np.array([15.0, -80.0, 0.0, 0.0, 550.0, 5.0, 2.0], dtype=np.float32)
    _DEFAULT_STD = np.array([10.0, 15.0, 100.0, 100.0, 50.0, 10.0, 1.5], dtype=np.float32)

    def __init__(
        self,
        channel_model,
        telemetry_stream,
        radar_stream,
        foliage_map,
        state_mean: Optional[np.ndarray] = None,
        state_std: Optional[np.ndarray] = None,
        snr_threshold_db: float = 5.0,
        lambda_latency: float = 0.1,
        mu_outage: float = 10.0,
    ) -> None:
        super().__init__()
        self.channel = channel_model
        self.telemetry = telemetry_stream
        self.radar = radar_stream
        self.foliage = foliage_map
        self.snr_threshold = snr_threshold_db
        self.lambda_latency = lambda_latency
        self.mu_outage = mu_outage

        self.state_mean = state_mean if state_mean is not None else self._DEFAULT_MEAN
        self.state_std = state_std if state_std is not None else self._DEFAULT_STD
        # Guard against zero std
        self.state_std = np.where(self.state_std == 0, 1.0, self.state_std)

        # Action space: [delta_phase, delta_power, mcs_index, rb_allocation]
        self.action_space = spaces.Box(
            low=np.array([-np.pi / 4, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([np.pi / 4, 1.0, 4.0, 100.0], dtype=np.float32),
            dtype=np.float32,
        )
        # Observation space: 7-dimensional normalised state
        self.observation_space = spaces.Box(
            low=-10.0,
            high=10.0,
            shape=(7,),
            dtype=np.float32,
        )

        # Internal state
        self.sat_pos = np.zeros(3, dtype=np.float64)
        self.snr = 0.0
        self.rssi = 0.0
        self.rain_rate = 0.0
        self.foliage_density = 0.0

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
        self.sat_pos = np.asarray(self.telemetry.get_current_position(), dtype=np.float64)
        self.rain_rate = float(self.radar.get_at_location(self.sat_pos))
        self.foliage_density = float(self.foliage.get_at_location(self.sat_pos))
        self.snr = self.channel.compute_snr(
            self.sat_pos, self.rain_rate, self.foliage_density
        )
        self.rssi = self.channel.compute_rssi(self.sat_pos)
        obs = self._build_obs()
        return obs, {}

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        action = np.asarray(action, dtype=np.float32)
        delta_phase, delta_power, mcs_idx, rb_allocation = action

        # Advance satellite position
        self.sat_pos = np.asarray(
            self.telemetry.get_next_position(), dtype=np.float64
        )
        self.rain_rate = float(self.radar.get_at_location(self.sat_pos))
        self.foliage_density = float(self.foliage.get_at_location(self.sat_pos))

        # Apply beam steering: compute phased-array gain from delta_phase
        beam_gain_db = self._compute_beam_gain(delta_phase, delta_power)
        base_snr = self.channel.compute_snr(
            self.sat_pos, self.rain_rate, self.foliage_density
        )
        self.snr = base_snr + beam_gain_db
        self.rssi = self.channel.compute_rssi(self.sat_pos)

        mcs_idx_int = int(np.clip(round(float(mcs_idx)), 0, len(_MCS_TABLE) - 1))
        rb_count = int(np.clip(round(float(rb_allocation)), 1, 100))

        throughput = self._compute_throughput(self.snr, mcs_idx_int, rb_count)
        latency = self._compute_latency(rb_count)
        outage = 1.0 if self.snr < self.snr_threshold else 0.0

        reward = float(throughput - self.lambda_latency * latency - self.mu_outage * outage)

        obs = self._build_obs()
        terminated = False
        truncated = False
        info: Dict[str, Any] = {
            "throughput": throughput,
            "latency": latency,
            "outage": outage,
            "snr": self.snr,
        }
        return obs, reward, terminated, truncated, info

    def render(self) -> None:  # pragma: no cover
        pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_obs(self) -> np.ndarray:
        raw = np.array(
            [self.snr, self.rssi, *self.sat_pos, self.rain_rate, self.foliage_density],
            dtype=np.float32,
        )
        return self._normalize(raw)

    def _normalize(self, state: np.ndarray) -> np.ndarray:
        return ((state - self.state_mean) / self.state_std).astype(np.float32)

    def _compute_beam_gain(self, delta_phase: float, delta_power: float) -> float:
        """
        Simplified phased-array beam gain (dB).

        The maximum gain from a 64-element array is ~18 dB.  The actual gain
        is reduced by a sinc-squared steering pattern based on the phase error
        and scaled by the relative transmit power.

        Args:
            delta_phase:  Phase adjustment (rad) relative to boresight.
            delta_power:  Relative power level in [0, 1].

        Returns:
            Additional beam gain in dB.
        """
        n_elements = 64
        max_gain_db = 10 * np.log10(n_elements)  # ~18 dB
        # Array factor: |sin(N*psi/2) / (N*sin(psi/2))|^2
        psi = delta_phase
        if abs(psi) < 1e-9:
            array_factor = 1.0
        else:
            array_factor = abs(
                np.sin(n_elements * psi / 2) / (n_elements * np.sin(psi / 2))
            ) ** 2
        power_db = 10 * np.log10(max(1e-9, float(delta_power)))
        return max_gain_db + 10 * np.log10(max(1e-9, array_factor)) + power_db

    def _compute_throughput(self, snr_db: float, mcs_idx: int, rb_count: int) -> float:
        """
        Estimate achievable throughput (Mbps) for given SNR, MCS and RB count.

        Uses the AMC lookup table and a simplified OFDM capacity formula.

        Args:
            snr_db:   Effective SNR in dB.
            mcs_idx:  MCS table index (0–4).
            rb_count: Number of allocated resource blocks.

        Returns:
            Throughput in Mbps.
        """
        min_snr, bps, code_rate = _MCS_TABLE[mcs_idx]
        if snr_db < min_snr:
            # Fallback to lowest MCS
            _, bps, code_rate = _MCS_TABLE[0]

        bits_per_re = bps * code_rate
        re_per_rb_per_slot = _RB_SUBCARRIERS * _SYMBOLS_PER_SLOT
        bits_per_slot = bits_per_re * re_per_rb_per_slot * rb_count
        throughput_bps = bits_per_slot * _SLOTS_PER_SECOND
        return throughput_bps / 1e6  # Mbps

    @staticmethod
    def _compute_latency(rb_count: int) -> float:
        """
        Approximate one-way radio-access latency (ms).

        A larger RB allocation reduces scheduling delay (fewer retransmissions).

        Args:
            rb_count: Number of allocated resource blocks.

        Returns:
            Latency estimate in ms.
        """
        base_latency_ms = 600.0  # ~550 km propagation + processing overhead
        scheduling_ms = max(0.5, 10.0 / max(1, rb_count))
        return base_latency_ms + scheduling_ms
