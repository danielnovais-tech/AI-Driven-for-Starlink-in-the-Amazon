"""
Abstract hardware driver for phased-array beamforming controllers.

Translates the DRL agent's continuous action vector
``[delta_phase (rad), delta_power (0-1), mcs_index (0-4), rb_alloc (0-100)]``
into hardware-specific commands for a physical phased-array front-end.

Design philosophy:
    - ``BeamformingHardwareDriver`` is an abstract base class (ABC) that
      defines the mandatory interface every hardware adapter must implement.
    - ``NullHardwareDriver`` is a no-op stub for unit testing and simulation.
    - ``LoggingHardwareDriver`` wraps any driver and records all commands.
    - ``SpiHardwareDriver`` is a skeleton for real SPI-based front-ends
      (e.g. Anokiwave AWMF-0108) that developers fill in for their platform.

Usage::

    driver = NullHardwareDriver()
    driver.apply_action(delta_phase=0.1, delta_power=0.8, mcs=2, rb_alloc=50)
    telemetry = driver.read_telemetry()
"""

import abc
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class BeamCommand:
    """
    A single beamforming command issued to the hardware.

    Attributes:
        timestamp_s:  Unix timestamp of the command.
        delta_phase:  Phase increment (radians) applied to the steering angle.
        delta_power:  Normalised transmit power (0.0 = min, 1.0 = max).
        mcs_index:    Modulation and coding scheme index (0 – 4).
        rb_alloc:     Number of resource blocks allocated (0 – 100).
    """
    timestamp_s: float
    delta_phase: float
    delta_power: float
    mcs_index: int
    rb_alloc: int


@dataclass
class Telemetry:
    """
    Hardware telemetry snapshot.

    Attributes:
        timestamp_s:   Unix timestamp of the reading.
        tx_power_dbm:  Actual transmitted power in dBm.
        phase_deg:     Current beam steering phase in degrees.
        temperature_c: Front-end temperature in Celsius.
        pa_current_ma: Power amplifier current draw in mA.
        extra:         Vendor-specific key-value pairs.
    """
    timestamp_s: float
    tx_power_dbm: float = 0.0
    phase_deg: float = 0.0
    temperature_c: float = 25.0
    pa_current_ma: float = 0.0
    extra: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------

class BeamformingHardwareDriver(abc.ABC):
    """
    Abstract interface for phased-array hardware controllers.

    Sub-classes must implement :meth:`apply_action` and
    :meth:`read_telemetry`.  They may optionally override :meth:`connect`,
    :meth:`disconnect`, and :meth:`reset`.
    """

    def connect(self) -> None:
        """Open the communication channel to the hardware."""

    def disconnect(self) -> None:
        """Close the communication channel."""

    def reset(self) -> None:
        """Reset the hardware to a known safe state."""

    @abc.abstractmethod
    def apply_action(
        self,
        delta_phase: float,
        delta_power: float,
        mcs_index: int,
        rb_alloc: int,
    ) -> None:
        """
        Apply a beamforming command to the hardware.

        Args:
            delta_phase:  Phase steering increment (radians).
            delta_power:  Normalised transmit power fraction [0, 1].
            mcs_index:    MCS table index (0 – 4 per 3GPP NR Table 5.1.3.1-1).
            rb_alloc:     Number of resource blocks (0 – 100).
        """

    @abc.abstractmethod
    def read_telemetry(self) -> Telemetry:
        """
        Return a telemetry snapshot from the hardware.

        Returns:
            :class:`Telemetry` dataclass with current hardware state.
        """

    def apply_action_vector(self, action: np.ndarray) -> None:
        """
        Apply a raw action vector from the DRL agent.

        Converts the 4-element action array used in ``LEOBeamformingEnv``
        into typed arguments and calls :meth:`apply_action`.

        Args:
            action: Array ``[delta_phase, delta_power, mcs_index, rb_alloc]``.
        """
        if len(action) < 4:
            raise ValueError(f"Expected action vector of length 4, got {len(action)}")
        delta_phase = float(action[0])
        delta_power = float(np.clip(action[1], 0.0, 1.0))
        mcs_index = int(np.clip(round(float(action[2])), 0, 4))
        rb_alloc = int(np.clip(round(float(action[3])), 0, 100))
        self.apply_action(delta_phase, delta_power, mcs_index, rb_alloc)


# ---------------------------------------------------------------------------
# Null / stub driver
# ---------------------------------------------------------------------------

class NullHardwareDriver(BeamformingHardwareDriver):
    """
    No-op hardware driver for testing and simulation.

    All commands are accepted silently and telemetry returns zeroed values.
    The command log is accessible via :attr:`command_log`.
    """

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
        # Update simulated internal state
        max_power_dbm = 30.0
        self._tx_power_dbm = max_power_dbm * delta_power
        self._phase_deg += np.degrees(delta_phase)
        logger.debug("NullHardwareDriver: applied %s", cmd)

    def read_telemetry(self) -> Telemetry:
        return Telemetry(
            timestamp_s=time.time(),
            tx_power_dbm=self._tx_power_dbm,
            phase_deg=self._phase_deg % 360.0,
        )

    def reset(self) -> None:
        self.command_log.clear()
        self._tx_power_dbm = 0.0
        self._phase_deg = 0.0


# ---------------------------------------------------------------------------
# Logging decorator
# ---------------------------------------------------------------------------

class LoggingHardwareDriver(BeamformingHardwareDriver):
    """
    Decorator that logs all commands and telemetry reads to ``logger``.

    Wraps any :class:`BeamformingHardwareDriver` implementation and
    records every interaction at DEBUG level.

    Args:
        inner:  The driver to wrap.
    """

    def __init__(self, inner: BeamformingHardwareDriver) -> None:
        self._inner = inner

    def connect(self) -> None:
        logger.info("Connecting hardware driver %s", type(self._inner).__name__)
        self._inner.connect()

    def disconnect(self) -> None:
        logger.info("Disconnecting hardware driver %s", type(self._inner).__name__)
        self._inner.disconnect()

    def reset(self) -> None:
        logger.info("Resetting hardware driver")
        self._inner.reset()

    def apply_action(
        self,
        delta_phase: float,
        delta_power: float,
        mcs_index: int,
        rb_alloc: int,
    ) -> None:
        logger.debug(
            "apply_action: delta_phase=%.4f  delta_power=%.4f  mcs=%d  rb=%d",
            delta_phase, delta_power, mcs_index, rb_alloc,
        )
        self._inner.apply_action(delta_phase, delta_power, mcs_index, rb_alloc)

    def read_telemetry(self) -> Telemetry:
        telemetry = self._inner.read_telemetry()
        logger.debug("read_telemetry: %s", telemetry)
        return telemetry


# ---------------------------------------------------------------------------
# SPI skeleton (for real hardware integration)
# ---------------------------------------------------------------------------

class SpiHardwareDriver(BeamformingHardwareDriver):
    """
    Skeleton SPI driver for physical phased-array front-ends.

    Intended for platforms such as Anokiwave AWMF-0108 or Phazr PZ5502.
    Developers should fill in the register-map specifics for their hardware.

    Args:
        spi_device:    Path to the SPI device node (e.g. ``'/dev/spidev0.0'``).
        max_tx_power_dbm: Maximum transmit power (dBm).
        phase_bits:    Phase resolution in bits (e.g. 6 for 64 states).
    """

    def __init__(
        self,
        spi_device: str = "/dev/spidev0.0",
        max_tx_power_dbm: float = 30.0,
        phase_bits: int = 6,
    ) -> None:
        self.spi_device = spi_device
        self.max_tx_power_dbm = max_tx_power_dbm
        self.phase_bits = phase_bits
        self._n_phase_states = 2 ** phase_bits
        self._spi: Optional[Any] = None  # SPI bus handle (platform-specific)
        self._current_phase_idx: int = 0
        self._current_power_dbm: float = 0.0

    def connect(self) -> None:
        """
        Open the SPI bus.

        Replace the body with platform-specific SPI initialisation, e.g.::

            import spidev
            self._spi = spidev.SpiDev()
            self._spi.open(0, 0)
            self._spi.max_speed_hz = 1_000_000
        """
        logger.info("SpiHardwareDriver.connect: spi_device=%s", self.spi_device)
        # Placeholder – replace with real SPI open call on target platform.

    def disconnect(self) -> None:
        """Close the SPI bus."""
        if self._spi is not None:
            # self._spi.close()
            self._spi = None

    def apply_action(
        self,
        delta_phase: float,
        delta_power: float,
        mcs_index: int,
        rb_alloc: int,
    ) -> None:
        """
        Translate action into SPI register writes.

        Phase steering: quantise ``delta_phase`` to the nearest of
        ``2**phase_bits`` equally spaced phase states.

        Power control: map ``delta_power`` ∈ [0, 1] to a DAC value or
        attenuator register.

        Args:
            delta_phase:  Phase increment (radians).
            delta_power:  Normalised power [0, 1].
            mcs_index:    MCS index (ignored at the hardware level; handled
                          by the baseband processor).
            rb_alloc:     Resource-block count (baseband-level, ignored here).
        """
        # Quantise phase
        phase_step = 2.0 * np.pi / self._n_phase_states
        phase_delta_idx = round(delta_phase / phase_step)
        self._current_phase_idx = (self._current_phase_idx + phase_delta_idx) % self._n_phase_states

        # Quantise power (linear scale → register value 0 – 255)
        power_reg = int(np.clip(delta_power * 255, 0, 255))
        self._current_power_dbm = self.max_tx_power_dbm * delta_power

        logger.debug(
            "SPI apply: phase_idx=%d  power_reg=%d  mcs=%d  rb=%d",
            self._current_phase_idx, power_reg, mcs_index, rb_alloc,
        )
        # Placeholder register write:
        # self._spi.xfer2([PHASE_REG_ADDR, self._current_phase_idx])
        # self._spi.xfer2([POWER_REG_ADDR, power_reg])

    def read_telemetry(self) -> Telemetry:
        """
        Read telemetry registers from the hardware.

        Returns:
            :class:`Telemetry` with current phase and power values.

        Note:
            Replace the placeholder with real SPI register reads:

            .. code-block:: python

                raw = self._spi.readbytes(4)
                temp_c = raw[0] * 0.5  # example ADC scaling
        """
        return Telemetry(
            timestamp_s=time.time(),
            tx_power_dbm=self._current_power_dbm,
            phase_deg=(self._current_phase_idx / self._n_phase_states) * 360.0,
            temperature_c=25.0,  # placeholder – read from ADC on real hardware
        )
