"""
Offline training environment backed by pre-recorded HDF5 logs.

This environment replays real telemetry and radar recordings to train DRL
agents without live infrastructure.  The "world-model" approach treats the
next recorded state as the outcome of the current action, providing a lower
bound on the quality of a real deployment.

HDF5 layout expected:
    telemetry_h5:
        snr       (N,)   – SNR per sample (dB)
        rssi      (N,)   – RSSI per sample (dBm)
        pos       (N, 3) – Satellite position (km)
    radar_h5:
        rain_rate (N,)   – Rain rate interpolated at satellite position (mm/h)
    foliage_h5:
        lai       (N,)   – Leaf area index at satellite position

Reference: Offline RL / world-model approaches (Levine et al., 2020).
"""

from typing import Any, Dict, Optional, Tuple

import h5py
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class OfflineLEOEnv(gym.Env):
    """
    Offline Gymnasium environment replaying recorded Amazon link logs.

    The episode ends when the last sample is reached or after
    ``max_episode_steps`` steps (whichever comes first).

    Args:
        telemetry_h5:       Path to HDF5 with telemetry data.
        radar_h5:           Path to HDF5 with rain-rate data.
        foliage_h5:         Path to HDF5 with LAI data.
        snr_threshold_db:   Outage SNR threshold (dB).
        max_episode_steps:  Maximum steps per episode (None = unlimited).
    """

    metadata = {"render_modes": []}

    # Default normalisation (override via state_mean / state_std if needed)
    _DEFAULT_MEAN = np.array([15.0, -80.0, 0.0, 0.0, 550.0, 5.0, 2.0], dtype=np.float32)
    _DEFAULT_STD = np.array([10.0, 15.0, 100.0, 100.0, 50.0, 10.0, 1.5], dtype=np.float32)

    def __init__(
        self,
        telemetry_h5: str,
        radar_h5: str,
        foliage_h5: str,
        snr_threshold_db: float = 5.0,
        max_episode_steps: Optional[int] = None,
        state_mean: Optional[np.ndarray] = None,
        state_std: Optional[np.ndarray] = None,
    ) -> None:
        super().__init__()

        # Load datasets into memory for fast indexing
        with h5py.File(telemetry_h5, "r") as f:
            self._snr: np.ndarray = f["snr"][:].astype(np.float32)
            self._rssi: np.ndarray = f["rssi"][:].astype(np.float32)
            self._pos: np.ndarray = f["pos"][:].astype(np.float32)

        with h5py.File(radar_h5, "r") as f:
            self._rain: np.ndarray = f["rain_rate"][:].astype(np.float32)

        with h5py.File(foliage_h5, "r") as f:
            self._lai: np.ndarray = f["lai"][:].astype(np.float32)

        self.num_samples = len(self._snr)
        assert self.num_samples > 1, "Dataset must have at least 2 samples."

        self.snr_threshold = snr_threshold_db
        self.max_episode_steps = max_episode_steps
        self.current_idx: int = 0
        self._episode_steps: int = 0

        self.state_mean = state_mean if state_mean is not None else self._DEFAULT_MEAN
        self.state_std = state_std if state_std is not None else self._DEFAULT_STD
        self.state_std = np.where(self.state_std == 0, 1.0, self.state_std)

        self.action_space = spaces.Box(
            low=np.array([-np.pi / 4, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([np.pi / 4, 1.0, 4.0, 100.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(7,), dtype=np.float32
        )

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
        rng = np.random.default_rng(seed)
        self.current_idx = int(rng.integers(0, self.num_samples - 1))
        self._episode_steps = 0
        return self._get_obs(self.current_idx), {}

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        next_idx = self.current_idx + 1
        if next_idx >= self.num_samples:
            next_idx = self.num_samples - 1

        next_obs = self._get_obs(next_idx)
        snr_next = float(self._snr[next_idx])
        reward = self._compute_reward(snr_next)

        self.current_idx = next_idx
        self._episode_steps += 1

        terminated = next_idx >= self.num_samples - 1
        truncated = (
            self.max_episode_steps is not None
            and self._episode_steps >= self.max_episode_steps
        )
        info: Dict[str, Any] = {
            "snr": snr_next,
            "outage": 1.0 if snr_next < self.snr_threshold else 0.0,
        }
        return next_obs, reward, terminated, truncated, info

    def render(self) -> None:  # pragma: no cover
        pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_obs(self, idx: int) -> np.ndarray:
        raw = np.array(
            [
                self._snr[idx],
                self._rssi[idx],
                self._pos[idx, 0],
                self._pos[idx, 1],
                self._pos[idx, 2],
                self._rain[idx],
                self._lai[idx],
            ],
            dtype=np.float32,
        )
        return ((raw - self.state_mean) / self.state_std).astype(np.float32)

    def _compute_reward(self, snr_db: float) -> float:
        """
        Simple reward proportional to SNR above threshold.

        Args:
            snr_db: SNR of the next recorded state.

        Returns:
            Scalar reward value.
        """
        margin = snr_db - self.snr_threshold
        outage_penalty = 10.0 if snr_db < self.snr_threshold else 0.0
        return float(margin - outage_penalty)
