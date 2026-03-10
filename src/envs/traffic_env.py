"""
Traffic-aware multi-satellite environment with user-demand modelling.

Extends :class:`~envs.multi_satellite_env.MultiSatelliteEnv` by adding a
Poisson packet-arrival traffic model and a user-experience reward that accounts
for queue delay and service fairness, not just raw SNR-based throughput.

Traffic model:
    Each step, ``arrival_rate_mbps`` Mbps worth of packets arrive according to
    a Poisson process.  Arriving traffic joins a finite FIFO queue
    (``max_queue_mbps``).  Packets served during the step are dequeued at the
    rate equal to the link throughput; excess packets are dropped.

Reward:
    ``reward = w_tp · throughput − w_delay · queue_delay_ms
               − w_drop · drop_rate − w_ho · handover − w_outage · outage``

    Where:
    - ``throughput`` (Mbps) is the link throughput of the selected satellite.
    - ``queue_delay_ms`` is estimated as ``queue_occupancy / service_rate`` (ms).
    - ``drop_rate`` is the fraction of arriving traffic that was dropped.
    - ``handover``, ``outage`` are as in the base class.

The traffic state is appended to the observation::

    [... base obs ..., queue_occupancy_norm, arrival_rate_norm]

Usage::

    from envs.traffic_env import TrafficAwareMultiSatelliteEnv
    env = TrafficAwareMultiSatelliteEnv(
        channel_model=..., telemetry_stream=..., radar_stream=..., foliage_map=...,
        arrival_rate_mbps=30.0,  # Poisson mean
    )
    obs, info = env.reset()
    # info keys: throughput, latency, outage, handover, snr,
    #            queue_occupancy, drop_rate, queue_delay_ms
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
from gymnasium import spaces

from .multi_satellite_env import MultiSatelliteEnv, _snr_to_throughput


class TrafficAwareMultiSatelliteEnv(MultiSatelliteEnv):
    """
    Multi-satellite environment with Poisson traffic model and queue-aware reward.

    Appends two additional features to the base observation:
        - Normalised queue occupancy (queue / max_queue).
        - Normalised arrival rate (arrival / max_queue).

    Args:
        channel_model:       :class:`~channel.ChannelModel` instance.
        telemetry_stream:    Telemetry interface.
        radar_stream:        Radar interface.
        foliage_map:         Foliage interface.
        arrival_rate_mbps:   Mean Poisson arrival rate (Mbps).
        max_queue_mbps:      Maximum queue capacity (Mbps per step).
                             Packets arriving beyond this limit are dropped.
        step_duration_s:     Simulated step duration (seconds), used to
                             convert throughput from Mbps to bits served.
        w_throughput:        Throughput weight in the reward.
        w_delay:             Queue-delay penalty weight.
        w_drop:              Packet-drop penalty weight.
        w_outage:            Outage penalty weight.
        w_handover:          Handover penalty weight.
        seed:                Random seed for the Poisson arrival process.
        **kwargs:            Forwarded to :class:`MultiSatelliteEnv`.
    """

    # Normalisation constants for the two new features
    _TRAFFIC_MEAN = np.array([0.5, 0.5], dtype=np.float32)
    _TRAFFIC_STD = np.array([0.3, 0.3], dtype=np.float32)

    def __init__(
        self,
        channel_model,
        telemetry_stream,
        radar_stream,
        foliage_map,
        arrival_rate_mbps: float = 50.0,
        max_queue_mbps: float = 200.0,
        step_duration_s: float = 0.5,
        w_throughput: float = 1.0,
        w_delay: float = 0.01,
        w_drop: float = 5.0,
        w_outage: float = 10.0,
        w_handover: float = 1.0,
        seed: int = 0,
        **kwargs,
    ) -> None:
        super().__init__(
            channel_model=channel_model,
            telemetry_stream=telemetry_stream,
            radar_stream=radar_stream,
            foliage_map=foliage_map,
            **kwargs,
        )
        self.arrival_rate_mbps = arrival_rate_mbps
        self.max_queue_mbps = max_queue_mbps
        self.step_duration_s = step_duration_s
        self.w_throughput = w_throughput
        self.w_delay = w_delay
        self.w_drop = w_drop
        self.w_outage = w_outage
        self.w_handover = w_handover

        # Extend observation space with 2 traffic features
        base_dim = self.observation_space.shape[0]
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(base_dim + 2,), dtype=np.float32
        )
        # NOTE: do NOT extend self._obs_mean / self._obs_std here because the
        # parent's _build_obs() uses those arrays to normalise the base
        # (base_dim) raw observation.  Traffic features are normalised
        # independently in _augment_obs().

        # Internal traffic state
        self._rng = np.random.default_rng(seed)
        self._queue: float = 0.0  # current queue occupancy (Mbps units)

    # ------------------------------------------------------------------
    # Gymnasium API overrides
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._queue = 0.0
        obs, info = super().reset(seed=seed, options=options)
        return self._augment_obs(obs), info

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        base_obs, base_reward, terminated, truncated, info = super().step(action)

        throughput = float(info.get("throughput", 0.0))
        outage = float(info.get("outage", 0.0))
        handover = float(info.get("handover", False))

        # Poisson arrivals (Mbps) during this step
        # lambda_param: expected Mbps-equivalent traffic arriving this step
        lambda_param = self.arrival_rate_mbps * self.step_duration_s
        arrivals = float(self._rng.poisson(lambda_param))

        # Service: drain up to ``throughput * step_duration_s`` Mbps from queue
        service_capacity = throughput * self.step_duration_s
        new_queue_before_service = self._queue + arrivals
        drop = max(0.0, new_queue_before_service - self.max_queue_mbps)
        new_queue_before_service = min(new_queue_before_service, self.max_queue_mbps)
        served = min(new_queue_before_service, service_capacity)
        self._queue = max(0.0, new_queue_before_service - served)

        # Queue delay estimate: M/D/1 approximation → D = Q / µ
        service_rate_mbps = throughput + 1e-9  # avoid division by zero
        queue_delay_ms = (self._queue / service_rate_mbps) * 1000.0  # ms

        # Drop rate (fraction of arriving traffic dropped this step)
        drop_rate = drop / max(1.0, arrivals)

        # Weighted user-experience reward
        reward = (
            self.w_throughput * served
            - self.w_delay * queue_delay_ms
            - self.w_drop * drop_rate
            - self.w_outage * outage
            - self.w_handover * handover
        )

        info["queue_occupancy"] = self._queue
        info["drop_rate"] = drop_rate
        info["queue_delay_ms"] = queue_delay_ms
        info["arrivals_mbps"] = arrivals

        return self._augment_obs(base_obs), float(reward), terminated, truncated, info

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _augment_obs(self, base_obs: np.ndarray) -> np.ndarray:
        """Append normalised traffic features to the base observation."""
        queue_norm = float(np.clip(self._queue / max(self.max_queue_mbps, 1.0), 0.0, 1.0))
        arrival_norm = float(np.clip(
            self.arrival_rate_mbps * self.step_duration_s / max(self.max_queue_mbps, 1.0),
            0.0, 1.0,
        ))
        traffic_raw = np.array([queue_norm, arrival_norm], dtype=np.float32)
        traffic_norm = ((traffic_raw - self._TRAFFIC_MEAN) / self._TRAFFIC_STD).astype(np.float32)
        return np.concatenate([base_obs, traffic_norm]).astype(np.float32)
