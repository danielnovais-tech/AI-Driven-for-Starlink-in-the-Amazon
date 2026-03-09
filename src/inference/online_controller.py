"""
Online beam controller for real-time inference at a ground station or on board.

The controller polls sensors at a configurable interval (default 500 ms) and
issues beamforming commands derived from the trained DRL agent.

Typical usage::

    controller = OnlineBeamController(agent, telemetry, radar, foliage)
    while True:
        result = controller.step()
        time.sleep(0.5)
"""

from typing import Any, Dict, Optional

import numpy as np
import torch


class OnlineBeamController:
    """
    Real-time beamforming controller driven by a trained DRL agent.

    The controller is intentionally agent-agnostic: it only requires that
    ``agent`` has a ``get_action(state, deterministic=True)`` method returning
    a numpy action array (or integer for DQN).

    Args:
        agent:           Trained DRL agent (PPO, DQN, etc.).
        telemetry_stream: Object providing current telemetry:
                          - ``get_current_position()`` → array (x, y, z) km.
                          - ``get_current_snr()``       → float (dB).
                          - ``get_current_rssi()``      → float (dBm).
        radar_stream:    Object providing ``get_at_location(pos)`` → float
                         rain rate (mm/h).
        foliage_map:     Object providing ``get_at_location(pos)`` → float LAI.
        state_mean:      Optional state normalisation mean (length-7 array).
        state_std:       Optional state normalisation std (length-7 array).
        device:          Torch device string.
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
    ) -> None:
        self.agent = agent
        self.telemetry = telemetry_stream
        self.radar = radar_stream
        self.foliage = foliage_map
        self.device = torch.device(device)

        self.state_mean = state_mean if state_mean is not None else self._DEFAULT_MEAN
        self.state_std = state_std if state_std is not None else self._DEFAULT_STD
        self.state_std = np.where(self.state_std == 0, 1.0, self.state_std)

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

        Returns:
            Dictionary with keys:
                ``action``  – raw action array from the agent.
                ``state``   – normalised state vector that was fed to the agent.
                ``snr``     – current SNR reading (dB).
                ``rain``    – current rain rate (mm/h).
        """
        state, raw = self._collect_state()
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        action = self._get_agent_action(state_t)
        self.apply_beam_steering(action)

        # Log for potential fine-tuning
        self.state_history.append(state)
        self.action_history.append(action)

        return {
            "action": action,
            "state": state,
            "snr": float(raw[0]),
            "rain": float(raw[5]),
        }

    def apply_beam_steering(self, action) -> None:
        """
        Translate the agent's action into hardware commands.

        This method is the integration point with the phased-array
        controller API (e.g. PyPhasedArray).  The default implementation
        prints the command for demonstration purposes; override in a
        subclass for real hardware.

        Args:
            action: Action array [delta_phase, delta_power, mcs_idx, rb_alloc]
                    or integer (DQN).
        """
        # In production, issue commands via the phased-array driver here.
        pass  # noqa: WPS420 (intentional no-op stub)

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

    def _get_agent_action(self, state_t: torch.Tensor):
        """
        Call the agent's ``get_action`` method, handling both continuous
        (PPO/A3C) and discrete (DQN) agents.

        Args:
            state_t: State tensor of shape (1, state_dim).

        Returns:
            Action array or integer depending on agent type.
        """
        result = self.agent.get_action(
            state_t.squeeze(0).cpu().numpy(), deterministic=True
        )
        # PPO returns (action_array, log_prob); DQN returns int
        if isinstance(result, tuple):
            return result[0]
        return result
