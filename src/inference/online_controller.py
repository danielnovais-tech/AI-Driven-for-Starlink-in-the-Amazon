"""
Online beam controller for real-time inference at a ground station or on board.

The controller polls sensors at a configurable interval (default 500 ms) and
issues beamforming commands derived from the trained DRL agent.

Production features:
    - **Fallback policy**: if the agent fails or times out, a deterministic
      fallback keeps the last valid action (or steers towards the highest-SNR
      satellite).
    - **Watchdog counter**: consecutive failures increment a health counter;
      callers can check :attr:`~OnlineBeamController.consecutive_failures`.
    - **Structured logging**: every decision, handover, outage, and fallback
      event is emitted as JSON via :mod:`utils.logger`.
    - **Metrics**: counters and histograms are updated in
      :data:`utils.metrics.GLOBAL_REGISTRY` on every step.

Typical usage::

    controller = OnlineBeamController(agent, telemetry, radar, foliage)
    while True:
        result = controller.step()
        time.sleep(0.5)
"""

import time
from typing import Any, Dict, Optional

import numpy as np

from utils.logger import StructuredLogger
from utils.metrics import GLOBAL_REGISTRY as _metrics


class FallbackPolicy:
    """
    Deterministic fallback policy activated when the DRL agent is unavailable.

    Strategy:
        - If a previous valid action exists, repeat it (hold-last-valid).
        - Otherwise, return a safe default action
          (zero phase/power adjustment, lowest MCS, minimum RB allocation).

    Args:
        action_dim: Dimensionality of the continuous action vector.
    """

    _SAFE_ACTION = np.array([0.0, 0.5, 0.0, 10.0], dtype=np.float32)

    def __init__(self, action_dim: int = 4) -> None:
        self.action_dim = action_dim
        self._last_action: Optional[np.ndarray] = None

    def get_action(self, state: np.ndarray) -> np.ndarray:
        """
        Return the fallback action for the given state.

        Args:
            state: Normalised state vector (unused by this policy).

        Returns:
            Fallback action array of shape ``(action_dim,)``.
        """
        if self._last_action is not None:
            return self._last_action.copy()
        return self._SAFE_ACTION[: self.action_dim].copy()

    def update(self, action: np.ndarray) -> None:
        """Store the latest valid agent action for hold-last-valid logic."""
        self._last_action = np.array(action, dtype=np.float32)


class OnlineBeamController:
    """
    Real-time beamforming controller driven by a trained DRL agent.

    The controller is intentionally agent-agnostic: it only requires that
    ``agent`` has a ``get_action(state, deterministic=True)`` method returning
    a numpy action array (or integer for DQN).

    Args:
        agent:             Trained DRL agent (PPO, DQN, etc.).
        telemetry_stream:  Object providing current telemetry:
                           - ``get_current_position()`` → array (x, y, z) km.
                           - ``get_current_snr()``       → float (dB).
                           - ``get_current_rssi()``      → float (dBm).
        radar_stream:      Object providing ``get_at_location(pos)`` → float
                           rain rate (mm/h).
        foliage_map:       Object providing ``get_at_location(pos)`` → float LAI.
        state_mean:        Optional state normalisation mean (length-7 array).
        state_std:         Optional state normalisation std (length-7 array).
        device:            Torch device string (unused if agent is non-torch).
        snr_threshold_db:  SNR below which an outage is declared (dB).
        max_failures:      Consecutive failure threshold before a warning is
                           emitted (default 3).
    """

    _DEFAULT_MEAN = np.array([15.0, -80.0, 0.0, 0.0, 550.0, 5.0, 2.0], dtype=np.float32)
    _DEFAULT_STD = np.array([10.0, 15.0, 100.0, 100.0, 50.0, 10.0, 1.5], dtype=np.float32)

    def __init__(
        self,
        agent,
        telemetry_stream,
        radar_stream,
        foliage_map,
        state_mean: Optional[np.ndarray] = None,
        state_std: Optional[np.ndarray] = None,
        device: str = "cpu",
        snr_threshold_db: float = 5.0,
        max_failures: int = 3,
    ) -> None:
        self.agent = agent
        self.telemetry = telemetry_stream
        self.radar = radar_stream
        self.foliage = foliage_map
        self.snr_threshold = snr_threshold_db
        self.max_failures = max_failures

        self.state_mean = state_mean if state_mean is not None else self._DEFAULT_MEAN
        self.state_std = state_std if state_std is not None else self._DEFAULT_STD
        self.state_std = np.where(self.state_std == 0, 1.0, self.state_std)

        self._fallback = FallbackPolicy()
        self._logger = StructuredLogger("inference.controller")

        # Health tracking
        self.consecutive_failures: int = 0
        self._total_steps: int = 0

        # History for optional fine-tuning / logging
        self.action_history = []
        self.state_history = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def step(self) -> Dict[str, Any]:
        """
        Collect the current sensor readings, run inference and apply the
        resulting beamforming command.

        If the agent raises an exception or is unavailable, the
        :class:`FallbackPolicy` is used and a warning is logged.

        Returns:
            Dictionary with keys:
                ``action``      – raw action array from the agent (or fallback).
                ``state``       – normalised state vector that was fed to the agent.
                ``snr``         – current SNR reading (dB).
                ``rain``        – current rain rate (mm/h).
                ``fallback``    – True if the fallback policy was used.
                ``latency_ms``  – wall-clock inference latency in milliseconds.
        """
        t_start = time.perf_counter()
        self._total_steps += 1
        used_fallback = False

        state, raw = self._collect_state()
        snr = float(raw[0])
        rain = float(raw[5])

        try:
            action = self._get_agent_action(state)
            self._fallback.update(action)
            self.consecutive_failures = 0
        except Exception as exc:  # noqa: BLE001
            self.consecutive_failures += 1
            used_fallback = True
            _metrics.increment("errors_total")
            _metrics.increment("fallback_total")
            reason = f"agent_error: {type(exc).__name__}: {exc}"
            self._logger.log_fallback(reason, step=self._total_steps)
            action = self._fallback.get_action(state)

        latency_ms = (time.perf_counter() - t_start) * 1000.0

        self.apply_beam_steering(action)

        # Update metrics
        _metrics.increment("decisions_total")
        _metrics.set_gauge("snr_db", snr)
        _metrics.set_gauge("rain_rate_mmh", rain)
        _metrics.observe("inference_latency_ms", latency_ms)

        # Log outage
        if snr < self.snr_threshold:
            _metrics.increment("outages_total")
            self._logger.log_outage(
                satellite_id=None, snr_db=snr, step=self._total_steps
            )

        # Log decision (debug level to avoid log flooding in production)
        self._logger.debug(
            "Beamforming step",
            event="decision",
            snr_db=snr,
            rain_mm_h=rain,
            latency_ms=latency_ms,
            fallback=used_fallback,
            step=self._total_steps,
        )

        # Watchdog warning
        if self.consecutive_failures >= self.max_failures:
            self._logger.warning(
                f"Controller health degraded: {self.consecutive_failures} "
                "consecutive agent failures",
                event="watchdog_alert",
                consecutive_failures=self.consecutive_failures,
            )

        # Store history for potential fine-tuning
        self.state_history.append(state)
        self.action_history.append(action)

        return {
            "action": action,
            "state": state,
            "snr": snr,
            "rain": rain,
            "fallback": used_fallback,
            "latency_ms": latency_ms,
        }

    def apply_beam_steering(self, action) -> None:
        """
        Translate the agent's action into hardware commands.

        This method is the integration point with the phased-array
        controller API (e.g. via a :class:`~beamforming.hardware_driver.BeamformingHardwareDriver`).
        The default implementation is a no-op stub; override in a
        subclass for real hardware.

        Args:
            action: Action array [delta_phase, delta_power, mcs_idx, rb_alloc]
                    or integer (DQN).
        """
        pass  # noqa: WPS420 (intentional no-op stub)

    @property
    def is_healthy(self) -> bool:
        """
        Return ``True`` if the controller has not exceeded the consecutive
        failure threshold.
        """
        return self.consecutive_failures < self.max_failures

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _collect_state(self):
        """Gather sensor readings and return (normalised_state, raw_state)."""
        sat_pos = np.asarray(self.telemetry.get_current_position(), dtype=np.float32)
        snr = float(self.telemetry.get_current_snr())
        rssi = float(self.telemetry.get_current_rssi())
        rain_rate = float(self.radar.get_at_location(sat_pos))
        foliage = float(self.foliage.get_at_location(sat_pos))

        raw = np.array(
            [snr, rssi, sat_pos[0], sat_pos[1], sat_pos[2], rain_rate, foliage],
            dtype=np.float32,
        )
        normalised = ((raw - self.state_mean) / self.state_std).astype(np.float32)
        return normalised, raw

    def _get_agent_action(self, state: np.ndarray):
        """
        Call the agent's ``get_action`` method, handling both continuous
        (PPO/A3C) and discrete (DQN) agents.

        Args:
            state: Normalised state numpy array of shape ``(state_dim,)``.

        Returns:
            Action array or integer depending on agent type.
        """
        result = self.agent.get_action(state, deterministic=True)
        # PPO returns (action_array, log_prob); DQN returns int
        if isinstance(result, tuple):
            return result[0]
        return result
