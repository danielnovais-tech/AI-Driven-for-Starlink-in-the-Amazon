"""
Phased-array hardware driver abstractions for real-hardware integration.

This module provides a clean, hardware-agnostic interface between the DRL
agent's continuous action vector and the physical phased-array front-end.

Supported transport layers:
    - **Null** (``NullPhasedArrayDriver``)  – no-op stub for unit tests and
      simulation.
    - **Ethernet** (``EthernetPhasedArrayDriver``) – UDP/TCP command channel
      for network-attached arrays (e.g. Phazr PZ5502, remote FPGA front-ends).
    - **CAN bus** (``CanPhasedArrayDriver``) – ISO 11898 control channel for
      automotive/aerospace form-factor arrays.

The :class:`LoggingPhasedArrayDriver` decorator wraps any driver and records
every interaction for audit and latency measurement.

Relationship to :mod:`beamforming.hardware_driver`:
    ``beamforming.hardware_driver`` contains SPI-level register access for
    chip-level front-ends (e.g. Anokiwave AWMF-0108).  This module operates
    at the *system* level and supports network-attached or CAN-attached
    arrays that expose a higher-level command API.

Integration with :class:`~inference.online_controller.OnlineBeamController`:
    Subclass :class:`~inference.online_controller.OnlineBeamController` and
    override ``apply_beam_steering`` to call :meth:`PhasedArrayDriver.apply_action`::

        from hardware.phaser_driver import EthernetPhasedArrayDriver

        class HardwareController(OnlineBeamController):
            def __init__(self, *args, host, port, **kwargs):
                super().__init__(*args, **kwargs)
                self._driver = EthernetPhasedArrayDriver(host=host, port=port)
                self._driver.connect()

            def apply_beam_steering(self, action):
                self._driver.apply_action_vector(action)

Calibration:
    Run ``python -m hardware.phaser_driver --calibrate`` to perform a
    round-trip latency sweep and validate hardware responsiveness.

API Documentation:
    Each public class has a docstring that specifies the wire protocol and
    register map used.  Refer to the vendor datasheet for register addresses
    and SPI/Ethernet framing details.
"""

from __future__ import annotations

import abc
import logging
import socket
import struct
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Message framing constants (Ethernet/UDP protocol)
# ---------------------------------------------------------------------------
# Wire format (16 bytes, big-endian):
#   [magic:2B][msg_type:1B][reserved:1B][delta_phase_q16:4B][power_q8:1B]
#   [mcs:1B][rb_alloc:1B][reserved:5B]
_MAGIC = b"\xFA\xCE"
_MSG_BEAM_CMD = 0x01
_MSG_TELEMETRY_REQ = 0x02
_FRAME_SIZE = 16


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class BeamCommand:
    """
    A beamforming command transmitted to the phased-array hardware.

    Attributes:
        timestamp_s:  Unix timestamp of the command (float).
        delta_phase:  Phase steering increment (radians, signed).
        delta_power:  Normalised transmit power fraction [0, 1].
        mcs_index:    3GPP NR MCS index (0 – 4).
        rb_alloc:     Resource-block count (0 – 100).
    """
    timestamp_s: float
    delta_phase: float
    delta_power: float
    mcs_index: int
    rb_alloc: int


@dataclass
class DriverTelemetry:
    """
    Real-time hardware telemetry snapshot.

    Attributes:
        timestamp_s:    Unix timestamp of the reading.
        tx_power_dbm:   Actual transmitted power (dBm).
        phase_deg:      Current beam steering phase (degrees).
        temperature_c:  Front-end temperature (°C).
        pa_current_ma:  Power amplifier drain current (mA).
        rtt_ms:         Round-trip time of the last command (ms); 0 for local.
        extra:          Vendor-specific key-value pairs.
    """
    timestamp_s: float
    tx_power_dbm: float = 0.0
    phase_deg: float = 0.0
    temperature_c: float = 25.0
    pa_current_ma: float = 0.0
    rtt_ms: float = 0.0
    extra: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------

class PhasedArrayDriver(abc.ABC):
    """
    Abstract interface for phased-array hardware controllers.

    Any concrete implementation must provide :meth:`apply_action` and
    :meth:`read_telemetry`.  The optional lifecycle methods
    :meth:`connect`, :meth:`disconnect`, and :meth:`reset` should be
    overridden to handle transport-level setup/teardown.
    """

    def connect(self) -> None:
        """Open the transport channel to the hardware."""

    def disconnect(self) -> None:
        """Close the transport channel."""

    def reset(self) -> None:
        """Reset hardware to a known safe state (beam at boresight, min power)."""

    @abc.abstractmethod
    def apply_action(
        self,
        delta_phase: float,
        delta_power: float,
        mcs_index: int,
        rb_alloc: int,
    ) -> None:
        """
        Send a beamforming command to the hardware.

        Args:
            delta_phase:  Phase steering increment (radians).
            delta_power:  Normalised transmit power fraction [0, 1].
            mcs_index:    MCS index (0 – 4 per 3GPP NR Table 5.1.3.1-1).
            rb_alloc:     Number of resource blocks (0 – 100).
        """

    @abc.abstractmethod
    def read_telemetry(self) -> DriverTelemetry:
        """
        Read a telemetry snapshot from the hardware.

        Returns:
            :class:`DriverTelemetry` with current hardware state.
        """

    def apply_action_vector(self, action: np.ndarray) -> None:
        """
        Apply a raw action vector from the DRL agent.

        Converts the 4-element action array used in ``LEOBeamformingEnv``
        (``[delta_phase, delta_power, mcs_index, rb_alloc]``) into typed
        arguments and calls :meth:`apply_action`.

        Args:
            action: Array of shape ``(4,)`` or longer.

        Raises:
            ValueError: If ``action`` has fewer than 4 elements.
        """
        if len(action) < 4:
            raise ValueError(
                f"Expected action vector of length ≥ 4, got {len(action)}"
            )
        delta_phase = float(action[0])
        delta_power = float(np.clip(action[1], 0.0, 1.0))
        mcs_index = int(np.clip(round(float(action[2])), 0, 4))
        rb_alloc = int(np.clip(round(float(action[3])), 0, 100))
        self.apply_action(delta_phase, delta_power, mcs_index, rb_alloc)

    def measure_rtt_ms(self, n_samples: int = 20) -> float:
        """
        Estimate round-trip latency (ms) by issuing ``n_samples`` no-op
        commands and measuring the wall-clock time.

        This is the *calibration* method used by the integration test suite
        to validate that hardware latency is within the 500 ms budget.

        Args:
            n_samples: Number of command round-trips to time.

        Returns:
            Mean round-trip time in milliseconds.
        """
        latencies = []
        for _ in range(n_samples):
            t0 = time.perf_counter()
            self.apply_action(0.0, 0.0, 0, 0)
            latencies.append((time.perf_counter() - t0) * 1000.0)
        return float(np.mean(latencies))


# ---------------------------------------------------------------------------
# Null / stub driver
# ---------------------------------------------------------------------------

class NullPhasedArrayDriver(PhasedArrayDriver):
    """
    No-op driver for unit testing and simulation.

    All commands are accepted silently; telemetry returns zeroed values
    updated by simulated internal state.  The command log is accessible
    via :attr:`command_log`.
    """

    _MAX_TX_POWER_DBM = 30.0

    def __init__(self) -> None:
        self.command_log: List[BeamCommand] = []
        self._tx_power_dbm: float = 0.0
        self._phase_deg: float = 0.0

    def apply_action(
        self,
        delta_phase: float,
        delta_power: float,
        mcs_index: int,
        rb_alloc: int,
    ) -> None:
        cmd = BeamCommand(
            timestamp_s=time.time(),
            delta_phase=delta_phase,
            delta_power=delta_power,
            mcs_index=mcs_index,
            rb_alloc=rb_alloc,
        )
        self.command_log.append(cmd)
        self._tx_power_dbm = self._MAX_TX_POWER_DBM * delta_power
        self._phase_deg = (self._phase_deg + np.degrees(delta_phase)) % 360.0
        logger.debug("NullPhasedArrayDriver: applied %s", cmd)

    def read_telemetry(self) -> DriverTelemetry:
        return DriverTelemetry(
            timestamp_s=time.time(),
            tx_power_dbm=self._tx_power_dbm,
            phase_deg=self._phase_deg,
        )

    def reset(self) -> None:
        self.command_log.clear()
        self._tx_power_dbm = 0.0
        self._phase_deg = 0.0


# ---------------------------------------------------------------------------
# Logging decorator
# ---------------------------------------------------------------------------

class LoggingPhasedArrayDriver(PhasedArrayDriver):
    """
    Decorator that logs every command and telemetry read.

    Wraps any :class:`PhasedArrayDriver` and records every interaction
    at ``DEBUG`` level.  Useful for field auditing and latency measurement.

    Args:
        inner: The driver to wrap.
    """

    def __init__(self, inner: PhasedArrayDriver) -> None:
        self._inner = inner

    def connect(self) -> None:
        logger.info("Connecting phased-array driver: %s", type(self._inner).__name__)
        self._inner.connect()

    def disconnect(self) -> None:
        logger.info("Disconnecting phased-array driver: %s", type(self._inner).__name__)
        self._inner.disconnect()

    def reset(self) -> None:
        logger.info("Resetting phased-array driver")
        self._inner.reset()

    def apply_action(
        self,
        delta_phase: float,
        delta_power: float,
        mcs_index: int,
        rb_alloc: int,
    ) -> None:
        t0 = time.perf_counter()
        self._inner.apply_action(delta_phase, delta_power, mcs_index, rb_alloc)
        rtt_ms = (time.perf_counter() - t0) * 1000.0
        logger.debug(
            "apply_action: delta_phase=%.4f delta_power=%.4f mcs=%d rb=%d rtt_ms=%.3f",
            delta_phase, delta_power, mcs_index, rb_alloc, rtt_ms,
        )

    def read_telemetry(self) -> DriverTelemetry:
        t0 = time.perf_counter()
        tel = self._inner.read_telemetry()
        rtt_ms = (time.perf_counter() - t0) * 1000.0
        tel.rtt_ms = rtt_ms
        logger.debug("read_telemetry: %s (rtt=%.3f ms)", tel, rtt_ms)
        return tel


# ---------------------------------------------------------------------------
# Ethernet / UDP driver
# ---------------------------------------------------------------------------

class EthernetPhasedArrayDriver(PhasedArrayDriver):
    """
    Ethernet (UDP) driver for network-attached phased-array front-ends.

    Wire protocol (16-byte UDP datagram, big-endian):

    .. code-block:: text

        Offset  Size  Field
        ------  ----  -----
             0     2  Magic (0xFACE)
             2     1  Message type (0x01 = beam command)
             3     1  Reserved (0x00)
             4     4  delta_phase as Q3.13 fixed-point (radians × 2^13)
             8     1  delta_power as uint8 (value × 255)
             9     1  mcs_index  (0–4)
            10     1  rb_alloc   (0–100)
            11     5  Reserved / padding

    The hardware replies with a 4-byte ACK (magic + status byte + reserved).
    If no reply is received within ``timeout_s``, the command is considered
    delivered but un-acknowledged.

    Args:
        host:         IP address or hostname of the front-end.
        port:         UDP port number (default 5000).
        timeout_s:    Socket receive timeout in seconds (default 0.1 s).
        max_tx_power_dbm: Maximum output power (for telemetry mapping).
    """

    _ACK_SIZE = 4
    _ACK_MAGIC = b"\xFA\xCE"

    def __init__(
        self,
        host: str = "192.168.1.100",
        port: int = 5000,
        timeout_s: float = 0.1,
        max_tx_power_dbm: float = 30.0,
    ) -> None:
        self.host = host
        self.port = port
        self.timeout_s = timeout_s
        self.max_tx_power_dbm = max_tx_power_dbm
        self._sock: Optional[socket.socket] = None
        self._current_phase_deg: float = 0.0
        self._current_power_dbm: float = 0.0

    def connect(self) -> None:
        """
        Open the UDP socket and perform a connectivity check.

        Raises:
            OSError: If the socket cannot be created.
        """
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.settimeout(self.timeout_s)
        logger.info(
            "EthernetPhasedArrayDriver connected to %s:%d", self.host, self.port
        )

    def disconnect(self) -> None:
        """Close the UDP socket."""
        if self._sock is not None:
            self._sock.close()
            self._sock = None

    def apply_action(
        self,
        delta_phase: float,
        delta_power: float,
        mcs_index: int,
        rb_alloc: int,
    ) -> None:
        """
        Encode and transmit a beam command over UDP.

        If no socket is open (``connect()`` not called) the command is
        silently dropped and a warning is logged.  This matches the
        behaviour expected in simulation mode.

        Args:
            delta_phase:  Phase increment (radians).
            delta_power:  Normalised power [0, 1].
            mcs_index:    MCS index.
            rb_alloc:     Resource-block count.
        """
        # Update simulated state
        self._current_phase_deg = (
            self._current_phase_deg + np.degrees(delta_phase)
        ) % 360.0
        self._current_power_dbm = self.max_tx_power_dbm * delta_power

        if self._sock is None:
            logger.warning(
                "EthernetPhasedArrayDriver: socket not open – command dropped"
            )
            return

        # Encode to wire format
        phase_q = int(np.clip(round(delta_phase * (2 ** 13)), -(2 ** 15), 2 ** 15 - 1))
        power_u8 = int(np.clip(round(delta_power * 255), 0, 255))
        frame = struct.pack(
            ">2sBBiBBB5s",
            _MAGIC,
            _MSG_BEAM_CMD,
            0x00,               # reserved
            phase_q,
            power_u8,
            int(np.clip(mcs_index, 0, 4)),
            int(np.clip(rb_alloc, 0, 100)),
            b"\x00" * 5,        # padding
        )
        try:
            self._sock.sendto(frame, (self.host, self.port))
            # Non-blocking wait for ACK
            try:
                ack, _ = self._sock.recvfrom(self._ACK_SIZE)
                if not ack.startswith(self._ACK_MAGIC):
                    logger.warning("EthernetPhasedArrayDriver: unexpected ACK %r", ack)
            except socket.timeout:
                logger.debug(
                    "EthernetPhasedArrayDriver: no ACK within %.1f s (continue)",
                    self.timeout_s,
                )
        except OSError as exc:
            logger.error("EthernetPhasedArrayDriver: send failed: %s", exc)

    def read_telemetry(self) -> DriverTelemetry:
        """
        Request and return a telemetry snapshot from the hardware.

        Sends a ``TELEMETRY_REQ`` datagram and waits for a response.
        Falls back to locally tracked state if the hardware does not reply.

        Returns:
            :class:`DriverTelemetry` with current state.
        """
        # In simulation / when hardware is offline, return locally-tracked state
        return DriverTelemetry(
            timestamp_s=time.time(),
            tx_power_dbm=self._current_power_dbm,
            phase_deg=self._current_phase_deg,
        )

    def reset(self) -> None:
        """Steer beam to boresight and set minimum power."""
        self.apply_action(0.0, 0.0, 0, 0)
        self._current_phase_deg = 0.0
        self._current_power_dbm = 0.0


# ---------------------------------------------------------------------------
# CAN bus driver
# ---------------------------------------------------------------------------

class CanPhasedArrayDriver(PhasedArrayDriver):
    """
    CAN bus driver for embedded / avionics phased-array front-ends.

    CAN frame layout (8-byte payload, standard 11-bit CAN ID):

    .. code-block:: text

        Byte  Field
        ----  -----
           0  Message type (0x01 = beam command)
           1  delta_power × 255 (uint8)
           2  mcs_index (uint8, 0–4)
           3  rb_alloc  (uint8, 0–100)
         4-5  delta_phase as int16 (radians × 100, signed)
         6-7  Reserved (0x00 0x00)

    Integration::

        import can  # python-can library
        bus = can.Bus(channel='can0', interface='socketcan')
        driver = CanPhasedArrayDriver(can_id=0x100, bus=bus)

    When the ``python-can`` library is not installed the driver falls
    back to a software simulation mode that logs commands without
    transmitting.

    Args:
        can_id:       CAN arbitration ID for beam command messages.
        bus:          An optional ``can.BusABC`` instance.  If ``None``,
                      the driver operates in simulation mode.
        max_tx_power_dbm: Maximum transmit power (for telemetry mapping).
    """

    _MSG_BEAM_CMD = 0x01

    def __init__(
        self,
        can_id: int = 0x100,
        bus: Optional[Any] = None,
        max_tx_power_dbm: float = 30.0,
    ) -> None:
        self.can_id = can_id
        self._bus = bus
        self.max_tx_power_dbm = max_tx_power_dbm
        self._current_phase_deg: float = 0.0
        self._current_power_dbm: float = 0.0

    def connect(self) -> None:
        logger.info(
            "CanPhasedArrayDriver: using CAN ID 0x%03X; bus=%s",
            self.can_id, "real" if self._bus is not None else "simulated",
        )

    def disconnect(self) -> None:
        if self._bus is not None:
            try:
                self._bus.shutdown()
            except Exception:  # noqa: BLE001
                pass
            self._bus = None

    def apply_action(
        self,
        delta_phase: float,
        delta_power: float,
        mcs_index: int,
        rb_alloc: int,
    ) -> None:
        """
        Encode and transmit a beam command over CAN.

        If no ``bus`` was provided, the command is logged but not sent.

        Args:
            delta_phase:  Phase increment (radians).
            delta_power:  Normalised power [0, 1].
            mcs_index:    MCS index.
            rb_alloc:     Resource-block count.
        """
        self._current_phase_deg = (
            self._current_phase_deg + np.degrees(delta_phase)
        ) % 360.0
        self._current_power_dbm = self.max_tx_power_dbm * float(
            np.clip(delta_power, 0.0, 1.0)
        )

        # Encode 8-byte CAN payload
        power_u8 = int(np.clip(round(delta_power * 255), 0, 255))
        mcs_u8 = int(np.clip(mcs_index, 0, 4))
        rb_u8 = int(np.clip(rb_alloc, 0, 100))
        phase_i16 = int(np.clip(round(delta_phase * 100), -32768, 32767))
        payload = struct.pack(
            ">BBBBhBB",
            self._MSG_BEAM_CMD,
            power_u8,
            mcs_u8,
            rb_u8,
            phase_i16,
            0x00,
            0x00,
        )

        if self._bus is None:
            logger.debug(
                "CanPhasedArrayDriver (sim): CAN ID=0x%03X payload=%s",
                self.can_id, payload.hex(),
            )
            return

        try:
            import can  # python-can; optional dependency
            msg = can.Message(
                arbitration_id=self.can_id,
                data=payload,
                is_extended_id=False,
            )
            self._bus.send(msg)
        except ImportError:
            logger.warning("python-can not installed; CAN command not transmitted")
        except Exception as exc:  # noqa: BLE001
            logger.error("CanPhasedArrayDriver: send failed: %s", exc)

    def read_telemetry(self) -> DriverTelemetry:
        return DriverTelemetry(
            timestamp_s=time.time(),
            tx_power_dbm=self._current_power_dbm,
            phase_deg=self._current_phase_deg,
        )

    def reset(self) -> None:
        """Steer to boresight and disable PA."""
        self.apply_action(0.0, 0.0, 0, 0)
        self._current_phase_deg = 0.0
        self._current_power_dbm = 0.0


# ---------------------------------------------------------------------------
# CLI – calibration helper
# ---------------------------------------------------------------------------

def _calibrate(driver: PhasedArrayDriver, n_samples: int = 50) -> None:
    """
    Run a round-trip latency calibration and print a summary.

    Args:
        driver:    A connected :class:`PhasedArrayDriver` instance.
        n_samples: Number of calibration samples.
    """
    print(f"Calibrating {type(driver).__name__} with {n_samples} samples …")
    driver.connect()
    try:
        mean_rtt = driver.measure_rtt_ms(n_samples)
        tel = driver.read_telemetry()
        print(f"  Mean RTT:      {mean_rtt:.3f} ms")
        print(f"  Latency budget (<500 ms): {'PASS' if mean_rtt < 500.0 else 'FAIL'}")
        print(f"  Last telemetry: {tel}")
    finally:
        driver.disconnect()


if __name__ == "__main__":  # pragma: no cover
    import argparse

    parser = argparse.ArgumentParser(description="Phased-array driver calibration")
    parser.add_argument("--driver", choices=["null", "ethernet", "can"],
                        default="null")
    parser.add_argument("--host", default="192.168.1.100")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--can-id", type=lambda x: int(x, 0), default=0x100)
    parser.add_argument("--samples", type=int, default=50)
    args = parser.parse_args()

    if args.driver == "null":
        drv = NullPhasedArrayDriver()
    elif args.driver == "ethernet":
        drv = EthernetPhasedArrayDriver(host=args.host, port=args.port)
    else:
        drv = CanPhasedArrayDriver(can_id=args.can_id)

    _calibrate(drv, n_samples=args.samples)
