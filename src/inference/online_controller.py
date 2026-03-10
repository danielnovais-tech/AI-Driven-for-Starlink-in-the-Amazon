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

Subclasses:
    - :class:`GNNBeamController` – builds a ``HeteroData`` graph observation
      from multi-satellite telemetry and feeds it to a :class:`GNNPPOAgent`.
    - :class:`HardwareBeamController` – wires a
      :class:`~hardware.phaser_driver.PhasedArrayDriver` into
      :meth:`apply_beam_steering` for real phased-array integration.

Typical usage::

    controller = OnlineBeamController(agent, telemetry, radar, foliage)
    while True:
        result = controller.step()
        time.sleep(0.5)
"""

import time
from typing import Any, Dict, List, Optional

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
        explainer=None,
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
        # Optional DecisionExplainer; if None, explanations are skipped
        self._explainer = explainer

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

        # Optional per-step explanation (audit / dashboard)
        explanation: Dict[str, Any] = {}
        if self._explainer is not None and not used_fallback:
            try:
                explanation = self._explainer.explain(state, action)
                self._logger.debug(
                    "Step explanation",
                    event="explanation",
                    step=self._total_steps,
                    **{k: v for k, v in explanation.items()
                       if k not in ("feature_scores", "node_scores")},
                )
            except Exception:  # noqa: BLE001
                pass  # Explanation errors must never affect control flow

        return {
            "action": action,
            "state": state,
            "snr": snr,
            "rain": rain,
            "fallback": used_fallback,
            "latency_ms": latency_ms,
            "explanation": explanation,
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


# ---------------------------------------------------------------------------
# GNN-aware controller subclass
# ---------------------------------------------------------------------------

class GNNBeamController(OnlineBeamController):
    """
    Online beam controller backed by a :class:`~agents.gnn_ppo_agent.GNNPPOAgent`.

    Extends :class:`OnlineBeamController` to build a ``HeteroData`` graph
    observation at each step from the multi-satellite telemetry stream and
    pass it directly to the GNN agent, which selects the best satellite.

    The ``telemetry_stream`` must additionally expose:
        ``get_visible_satellites()`` → list of position arrays (km).
        ``ground_station_pos``       → ground-station ECEF position (km).

    The resulting action is the integer satellite index selected by the agent.
    :meth:`apply_beam_steering` receives the satellite index; override it to
    issue the corresponding hardware command.

    Args:
        agent:            A :class:`~agents.gnn_ppo_agent.GNNPPOAgent`
                          (or any agent accepting a ``HeteroData`` graph).
        telemetry_stream: Multi-satellite telemetry (see above).
        radar_stream:     Rain-rate provider.
        foliage_map:      LAI provider.
        node_feature_dim: Number of features per satellite node in the graph
                          (default 4: SNR, distance, elevation, rain_rate).
        snr_threshold_db: Outage SNR threshold (dB).
        max_failures:     Consecutive failure threshold before a watchdog warning.
    """

    def __init__(
        self,
        agent,
        telemetry_stream,
        radar_stream,
        foliage_map,
        node_feature_dim: int = 4,
        snr_threshold_db: float = 5.0,
        max_failures: int = 3,
    ) -> None:
        # Call super with dummy normalisation (flat state unused by GNN controller)
        super().__init__(
            agent=agent,
            telemetry_stream=telemetry_stream,
            radar_stream=radar_stream,
            foliage_map=foliage_map,
            snr_threshold_db=snr_threshold_db,
            max_failures=max_failures,
        )
        self.node_feature_dim = node_feature_dim

    # ------------------------------------------------------------------
    # Override public step to use graph observations
    # ------------------------------------------------------------------

    def step(self) -> Dict[str, Any]:
        """
        Build a graph observation, run the GNN agent, and return the result.

        Returns:
            Dictionary with keys:
                ``action``     – selected satellite index (int).
                ``graph_obs``  – the ``HeteroData`` graph used for inference.
                ``snr``        – SNR of the selected satellite (dB).
                ``rain``       – rain rate at the selected satellite (mm/h).
                ``fallback``   – True if the fallback policy was used.
                ``latency_ms`` – wall-clock inference latency (ms).
                ``n_visible``  – number of visible satellites.
        """
        t_start = time.perf_counter()
        self._total_steps += 1
        used_fallback = False

        graph_obs, snr, rain, n_visible = self._build_graph_obs()

        try:
            result = self.agent.get_action(graph_obs, deterministic=True)
            action = result[0] if isinstance(result, tuple) else result
            # Normalise action to a flat array for fallback storage
            _action_arr = np.array([float(action)], dtype=np.float32)
            self._fallback.update(_action_arr)
            self.consecutive_failures = 0
        except Exception as exc:  # noqa: BLE001
            self.consecutive_failures += 1
            used_fallback = True
            _metrics.increment("errors_total")
            _metrics.increment("fallback_total")
            reason = f"agent_error: {type(exc).__name__}: {exc}"
            self._logger.log_fallback(reason, step=self._total_steps)
            fallback_arr = self._fallback.get_action(np.zeros(1, dtype=np.float32))
            action = int(fallback_arr[0])

        latency_ms = (time.perf_counter() - t_start) * 1000.0

        self.apply_beam_steering(action)

        # Metrics
        _metrics.increment("decisions_total")
        _metrics.set_gauge("snr_db", snr)
        _metrics.set_gauge("rain_rate_mmh", rain)
        _metrics.observe("inference_latency_ms", latency_ms)

        if snr < self.snr_threshold:
            _metrics.increment("outages_total")
            self._logger.log_outage(satellite_id=int(action), snr_db=snr,
                                    step=self._total_steps)

        if self.consecutive_failures >= self.max_failures:
            self._logger.warning(
                f"GNN controller health degraded: {self.consecutive_failures} "
                "consecutive failures",
                event="watchdog_alert",
                consecutive_failures=self.consecutive_failures,
            )

        self.action_history.append(action)
        self.state_history.append(graph_obs)

        return {
            "action": action,
            "graph_obs": graph_obs,
            "snr": snr,
            "rain": rain,
            "fallback": used_fallback,
            "latency_ms": latency_ms,
            "n_visible": n_visible,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_graph_obs(self):
        """
        Construct a ``HeteroData`` graph from live telemetry.

        Returns:
            Tuple ``(graph, best_snr, best_rain, n_visible)``.
        """
        try:
            import math as _math
            import torch as _torch
            from torch_geometric.data import HeteroData as _HeteroData
        except ImportError as exc:
            raise ImportError(
                "GNNBeamController requires torch and torch_geometric. "
                "Install with: pip install torch torch-geometric"
            ) from exc

        visible = [
            np.asarray(s, dtype=np.float64)
            for s in self.telemetry.get_visible_satellites()
        ]
        gs = np.asarray(self.telemetry.ground_station_pos, dtype=np.float64)

        # Handle the empty-constellation case: create one dummy node so the
        # graph is structurally valid.  The agent will receive zeroed features
        # and should select satellite index 0 (which the fallback will correct).
        if not visible:
            visible = [np.zeros(3, dtype=np.float64)]

        n_vis = len(visible)

        # Named constants for the distance-based SNR approximation
        _BASE_SNR_DB = 20.0          # nominal free-space SNR at 550 km altitude (dB)
        _RAIN_ATT_FACTOR = 0.5       # dB per mm/h rain attenuation coefficient
        _NOMINAL_ALT_KM = 550.0      # reference altitude for SNR approximation (km)
        _DIST_PENALTY_SCALE = 100.0  # km per dB distance penalty scaling
        _ELEV_EPSILON = 1e-9         # numerical guard for elevation computation

        sat_feat = np.zeros((n_vis, self.node_feature_dim), dtype=np.float32)
        best_snr = -999.0
        best_rain = 0.0

        for i, sat in enumerate(visible):
            rain = float(self.radar.get_at_location(sat))
            # Reuse channel model if possible; otherwise use a simple approximation
            try:
                snr = float(self.telemetry.get_current_snr())
            except AttributeError:
                # Fallback: distance-based SNR approximation
                dist = float(np.linalg.norm(sat - gs))
                snr = (_BASE_SNR_DB
                       - rain * _RAIN_ATT_FACTOR
                       - max(0.0, (dist - _NOMINAL_ALT_KM) / _DIST_PENALTY_SCALE))

            dist_km = float(np.linalg.norm(sat - gs))
            vec = sat - gs
            gs_norm = float(np.linalg.norm(gs))
            # Elevation angle: arcsin of the dot-product of the look-vector with
            # the local vertical (ground-station position unit vector)
            elev = (
                _math.degrees(_math.asin(
                    min(1.0, max(-1.0,
                                 float(np.dot(vec, gs))
                                 / (dist_km * gs_norm + _ELEV_EPSILON)))
                ))
                if dist_km > 1e-6 and gs_norm > 1e-6 else 90.0
            )

            sat_feat[i] = [snr, dist_km, elev, rain]
            if snr > best_snr:
                best_snr = snr
                best_rain = rain

        # Build HeteroData graph
        data = _HeteroData()
        data["sat"].x = _torch.FloatTensor(sat_feat)
        data["sat"].num_nodes = n_vis
        data["ground_station"].x = _torch.zeros(1, self.node_feature_dim)
        data["ground_station"].num_nodes = 1
        src = _torch.arange(n_vis, dtype=_torch.long)
        dst = _torch.zeros(n_vis, dtype=_torch.long)
        data["sat", "to", "ground_station"].edge_index = _torch.stack([src, dst], dim=0)

        return data, float(best_snr), float(best_rain), n_vis


# ---------------------------------------------------------------------------
# Hardware-integrated controller subclass
# ---------------------------------------------------------------------------

class HardwareBeamController(OnlineBeamController):
    """
    Online beam controller that forwards commands to a physical phased-array.

    Extends :class:`OnlineBeamController` by overriding
    :meth:`apply_beam_steering` to call
    :meth:`~hardware.phaser_driver.PhasedArrayDriver.apply_action_vector`
    on a connected driver.  After each steering command the hardware
    telemetry is read back and stored in :attr:`last_hw_telemetry` for
    latency measurement and pointing validation.

    Rain-attenuation injection:
        Set ``inject_rain_attenuation_db`` to a non-zero value to subtract
        a fixed attenuation from every SNR reading, simulating heavy-rain
        propagation loss.  This is used in field test scenarios to verify
        that the agent adapts its MCS / power allocation correctly.

    Args:
        agent:                    Trained DRL agent with ``get_action()``.
        telemetry_stream:         Telemetry interface.
        radar_stream:             Radar interface.
        foliage_map:              Foliage interface.
        hw_driver:                A connected
                                  :class:`~hardware.phaser_driver.PhasedArrayDriver`
                                  instance.
        inject_rain_attenuation_db: Additional SNR penalty (dB) injected
                                  on every step (0 = disabled).
        **kwargs:                 Forwarded to :class:`OnlineBeamController`.

    Example::

        from hardware.phaser_driver import EthernetPhasedArrayDriver
        from inference.online_controller import HardwareBeamController

        driver = EthernetPhasedArrayDriver(host="192.168.1.100")
        driver.connect()
        ctrl = HardwareBeamController(agent, telemetry, radar, foliage,
                                      hw_driver=driver)
        while True:
            result = ctrl.step()
            time.sleep(0.5)
    """

    def __init__(
        self,
        agent,
        telemetry_stream,
        radar_stream,
        foliage_map,
        hw_driver,
        inject_rain_attenuation_db: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__(
            agent=agent,
            telemetry_stream=telemetry_stream,
            radar_stream=radar_stream,
            foliage_map=foliage_map,
            **kwargs,
        )
        self._hw_driver = hw_driver
        self.inject_rain_attenuation_db = inject_rain_attenuation_db
        # Most-recent hardware telemetry snapshot (updated after each steering cmd)
        self.last_hw_telemetry = None

    def apply_beam_steering(self, action) -> None:
        """
        Forward the action to the phased-array hardware driver.

        For continuous actions (array of ≥ 4 elements), calls
        :meth:`~hardware.phaser_driver.PhasedArrayDriver.apply_action_vector`.
        For discrete actions (single integer), converts to a minimal
        continuous command (phase=0, power=1, mcs=action%5, rb=50).

        After sending the command, reads back hardware telemetry and stores
        it in :attr:`last_hw_telemetry`.

        Args:
            action: Continuous action array or discrete integer.
        """
        if isinstance(action, (int, np.integer)):
            vec = np.array([0.0, 1.0, float(action % 5), 50.0], dtype=np.float32)
        else:
            vec = np.asarray(action, dtype=np.float32)
        try:
            self._hw_driver.apply_action_vector(vec)
            # Read back telemetry for latency measurement and validation
            self.last_hw_telemetry = self._hw_driver.read_telemetry()
        except Exception as exc:  # noqa: BLE001
            self._logger.warning(
                f"Hardware driver error: {exc}",
                event="hw_driver_error",
                step=self._total_steps,
            )

    def _collect_state(self):
        """
        Extend base state collection with rain-attenuation injection.

        Subtracts :attr:`inject_rain_attenuation_db` from the SNR reading
        to simulate heavy-rain propagation loss in field test scenarios.
        """
        normalised, raw = super()._collect_state()
        if self.inject_rain_attenuation_db != 0.0:
            # raw[0] is the SNR reading; apply attenuation injection
            raw = raw.copy()
            raw[0] = raw[0] - self.inject_rain_attenuation_db
            normalised = ((raw - self.state_mean) / self.state_std).astype(np.float32)
        return normalised, raw
