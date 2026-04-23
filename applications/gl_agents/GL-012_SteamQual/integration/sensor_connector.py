"""
GL-012 STEAMQUAL - Sensor Connector

Industrial sensor data acquisition for steam quality control:
- Pressure/Temperature/Flow sensor connectors
- Chemistry analyzer connectors (pH, conductivity, silica, dissolved O2)
- Separator differential pressure connectors
- Drain valve position connectors
- Signal conditioning (filtering, scaling, status bits)

Signal Conditioning Features:
- Low-pass filtering for noise reduction
- Median filtering for spike rejection
- Moving average for smoothing
- Exponential smoothing for trend tracking
- Range scaling and clamping
- Status bit decoding (maintenance, fault, override)

Playbook Requirements:
- Time synchronization to NTP/GPS time source
- Sample rates: 1s for critical, 5s for standard, 60s for chemistry
- Data quality flagging per IEC 61131 and OPC-UA standards
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum, IntFlag
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import asyncio
import logging
import hashlib
import math
from collections import deque

logger = logging.getLogger(__name__)


class SensorType(Enum):
    """Steam quality sensor types."""
    PRESSURE = "pressure"
    TEMPERATURE = "temperature"
    FLOW = "flow"
    DIFFERENTIAL_PRESSURE = "differential_pressure"
    LEVEL = "level"
    CONDUCTIVITY = "conductivity"
    PH = "ph"
    DISSOLVED_O2 = "dissolved_o2"
    SILICA = "silica"
    VALVE_POSITION = "valve_position"
    DRAIN_DUTY_CYCLE = "drain_duty_cycle"
    SEPARATOR_DP = "separator_dp"
    STEAM_QUALITY = "steam_quality"


class StatusBits(IntFlag):
    """Sensor status bit flags per IEC 61131-3."""
    GOOD = 0x00
    MAINTENANCE = 0x01
    FAULT = 0x02
    OVERRIDE = 0x04
    SIMULATION = 0x08
    OUT_OF_RANGE = 0x10
    STALE = 0x20
    CALIBRATION_DUE = 0x40
    COMMUNICATION_ERROR = 0x80


class FilterType(Enum):
    """Signal conditioning filter types."""
    NONE = "none"
    LOW_PASS = "low_pass"
    MEDIAN = "median"
    MOVING_AVERAGE = "moving_average"
    EXPONENTIAL = "exponential"
    BUTTERWORTH = "butterworth"


class QualityCode(Enum):
    """OPC-UA quality codes."""
    GOOD = 0x00000000
    GOOD_LOCAL_OVERRIDE = 0x00D80000
    UNCERTAIN = 0x40000000
    UNCERTAIN_LAST_USABLE = 0x40480000
    UNCERTAIN_SENSOR_CAL = 0x40890000
    BAD = 0x80000000
    BAD_CONFIG_ERROR = 0x80040000
    BAD_SENSOR_FAILURE = 0x80120000
    BAD_COMM_FAILURE = 0x80130000
    BAD_OUT_OF_RANGE = 0x80350000
    BAD_OUT_OF_SERVICE = 0x808D0000

    def is_good(self) -> bool:
        """Check if quality is good."""
        return (self.value & 0xC0000000) == 0x00000000

    def is_uncertain(self) -> bool:
        """Check if quality is uncertain."""
        return (self.value & 0xC0000000) == 0x40000000

    def is_bad(self) -> bool:
        """Check if quality is bad."""
        return (self.value & 0xC0000000) == 0x80000000


@dataclass
class SensorReading:
    """
    Single sensor reading with metadata.

    Attributes:
        sensor_id: Unique sensor identifier
        sensor_type: Type of sensor
        raw_value: Unconditioned raw value
        value: Conditioned/filtered value
        unit: Engineering unit
        timestamp: Reading timestamp (UTC)
        quality: OPC-UA quality code
        status_bits: IEC 61131 status flags
        source_hash: SHA-256 provenance hash
    """
    sensor_id: str
    sensor_type: SensorType
    raw_value: float
    value: float
    unit: str
    timestamp: datetime
    quality: QualityCode = QualityCode.GOOD
    status_bits: StatusBits = StatusBits.GOOD
    source_hash: str = ""

    # Filtering metadata
    filter_applied: str = "none"
    samples_averaged: int = 1

    def __post_init__(self):
        """Calculate provenance hash."""
        if not self.source_hash:
            hash_data = f"{self.sensor_id}:{self.raw_value}:{self.timestamp.isoformat()}"
            self.source_hash = hashlib.sha256(hash_data.encode()).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "sensor_id": self.sensor_id,
            "sensor_type": self.sensor_type.value,
            "raw_value": self.raw_value,
            "value": self.value,
            "unit": self.unit,
            "timestamp": self.timestamp.isoformat(),
            "quality": self.quality.name,
            "is_good": self.quality.is_good(),
            "status_bits": int(self.status_bits),
            "source_hash": self.source_hash,
        }


@dataclass
class SignalConditionerConfig:
    """Signal conditioning configuration."""
    filter_type: FilterType = FilterType.MOVING_AVERAGE

    # Filter parameters
    window_size: int = 5  # Samples for moving average/median
    cutoff_frequency_hz: float = 1.0  # Low-pass cutoff
    alpha: float = 0.3  # Exponential smoothing factor (0-1)

    # Scaling
    raw_min: float = 0.0
    raw_max: float = 100.0
    scaled_min: float = 0.0
    scaled_max: float = 100.0

    # Clamping
    clamp_enabled: bool = True
    clamp_min: float = 0.0
    clamp_max: float = 100.0

    # Deadband
    deadband_enabled: bool = False
    deadband_value: float = 0.5

    # Rate limiting
    rate_limit_enabled: bool = False
    max_rate_per_second: float = 10.0

    # Status bit mapping (raw bit positions)
    maintenance_bit: int = 0
    fault_bit: int = 1
    override_bit: int = 2


class SignalConditioner:
    """
    Signal conditioning for sensor data.

    Applies filtering, scaling, and quality assessment to raw sensor values.

    Example:
        config = SignalConditionerConfig(
            filter_type=FilterType.MOVING_AVERAGE,
            window_size=5,
            raw_min=4.0, raw_max=20.0,
            scaled_min=0.0, scaled_max=100.0,
        )
        conditioner = SignalConditioner("pressure_001", config)

        reading = conditioner.process(raw_value=12.0, raw_status=0)
    """

    def __init__(self, sensor_id: str, config: SignalConditionerConfig) -> None:
        """
        Initialize signal conditioner.

        Args:
            sensor_id: Sensor identifier
            config: Conditioning configuration
        """
        self.sensor_id = sensor_id
        self.config = config

        # Filter state
        self._buffer: deque = deque(maxlen=config.window_size)
        self._last_output: Optional[float] = None
        self._last_timestamp: Optional[datetime] = None

        # Statistics
        self._stats = {
            "samples_processed": 0,
            "spikes_rejected": 0,
            "rate_limited": 0,
            "clamped": 0,
        }

        logger.debug(f"SignalConditioner initialized: {sensor_id}")

    def process(
        self,
        raw_value: float,
        raw_status: int = 0,
        timestamp: Optional[datetime] = None,
        sensor_type: SensorType = SensorType.PRESSURE,
        unit: str = "",
    ) -> SensorReading:
        """
        Process raw sensor value.

        Args:
            raw_value: Raw sensor value
            raw_status: Raw status word
            timestamp: Value timestamp
            sensor_type: Type of sensor
            unit: Engineering unit

        Returns:
            Conditioned SensorReading
        """
        ts = timestamp or datetime.now(timezone.utc)
        self._stats["samples_processed"] += 1

        # Scale raw value
        scaled = self._scale_value(raw_value)

        # Apply filter
        filtered = self._apply_filter(scaled)

        # Apply rate limiting
        if self.config.rate_limit_enabled:
            filtered = self._apply_rate_limit(filtered, ts)

        # Apply clamping
        if self.config.clamp_enabled:
            filtered = self._clamp_value(filtered)

        # Decode status bits
        status_bits = self._decode_status(raw_status)

        # Determine quality code
        quality = self._assess_quality(filtered, status_bits)

        # Update state
        self._last_output = filtered
        self._last_timestamp = ts

        return SensorReading(
            sensor_id=self.sensor_id,
            sensor_type=sensor_type,
            raw_value=raw_value,
            value=round(filtered, 6),
            unit=unit,
            timestamp=ts,
            quality=quality,
            status_bits=status_bits,
            filter_applied=self.config.filter_type.value,
            samples_averaged=len(self._buffer),
        )

    def _scale_value(self, raw: float) -> float:
        """Scale raw value to engineering units."""
        raw_range = self.config.raw_max - self.config.raw_min
        scaled_range = self.config.scaled_max - self.config.scaled_min

        if raw_range == 0:
            return self.config.scaled_min

        normalized = (raw - self.config.raw_min) / raw_range
        return self.config.scaled_min + normalized * scaled_range

    def _apply_filter(self, value: float) -> float:
        """Apply configured filter."""
        self._buffer.append(value)

        if self.config.filter_type == FilterType.NONE:
            return value

        if self.config.filter_type == FilterType.MOVING_AVERAGE:
            return sum(self._buffer) / len(self._buffer)

        if self.config.filter_type == FilterType.MEDIAN:
            sorted_buf = sorted(self._buffer)
            n = len(sorted_buf)
            if n % 2 == 0:
                return (sorted_buf[n//2 - 1] + sorted_buf[n//2]) / 2
            return sorted_buf[n//2]

        if self.config.filter_type == FilterType.EXPONENTIAL:
            if self._last_output is None:
                return value
            alpha = self.config.alpha
            return alpha * value + (1 - alpha) * self._last_output

        if self.config.filter_type == FilterType.LOW_PASS:
            # Simple first-order low-pass approximation
            if self._last_output is None:
                return value
            # RC time constant based on cutoff frequency
            rc = 1.0 / (2 * math.pi * self.config.cutoff_frequency_hz)
            dt = 1.0  # Assume 1 second sample rate
            alpha = dt / (rc + dt)
            return alpha * value + (1 - alpha) * self._last_output

        return value

    def _apply_rate_limit(self, value: float, timestamp: datetime) -> float:
        """Apply rate limiting."""
        if self._last_output is None or self._last_timestamp is None:
            return value

        dt = (timestamp - self._last_timestamp).total_seconds()
        if dt <= 0:
            return self._last_output

        max_change = self.config.max_rate_per_second * dt
        change = value - self._last_output

        if abs(change) > max_change:
            self._stats["rate_limited"] += 1
            if change > 0:
                return self._last_output + max_change
            return self._last_output - max_change

        return value

    def _clamp_value(self, value: float) -> float:
        """Clamp value to configured range."""
        if value < self.config.clamp_min:
            self._stats["clamped"] += 1
            return self.config.clamp_min
        if value > self.config.clamp_max:
            self._stats["clamped"] += 1
            return self.config.clamp_max
        return value

    def _decode_status(self, raw_status: int) -> StatusBits:
        """Decode raw status word to StatusBits."""
        status = StatusBits.GOOD

        if raw_status & (1 << self.config.maintenance_bit):
            status |= StatusBits.MAINTENANCE
        if raw_status & (1 << self.config.fault_bit):
            status |= StatusBits.FAULT
        if raw_status & (1 << self.config.override_bit):
            status |= StatusBits.OVERRIDE

        return status

    def _assess_quality(self, value: float, status: StatusBits) -> QualityCode:
        """Assess quality code based on value and status."""
        if status & StatusBits.FAULT:
            return QualityCode.BAD_SENSOR_FAILURE
        if status & StatusBits.COMMUNICATION_ERROR:
            return QualityCode.BAD_COMM_FAILURE
        if status & StatusBits.OUT_OF_RANGE:
            return QualityCode.BAD_OUT_OF_RANGE
        if status & StatusBits.STALE:
            return QualityCode.UNCERTAIN_LAST_USABLE
        if status & StatusBits.MAINTENANCE:
            return QualityCode.UNCERTAIN_SENSOR_CAL
        if status & StatusBits.OVERRIDE:
            return QualityCode.GOOD_LOCAL_OVERRIDE
        return QualityCode.GOOD

    def reset(self) -> None:
        """Reset filter state."""
        self._buffer.clear()
        self._last_output = None
        self._last_timestamp = None

    def get_statistics(self) -> Dict[str, Any]:
        """Get conditioning statistics."""
        return {
            **self._stats,
            "buffer_size": len(self._buffer),
            "filter_type": self.config.filter_type.value,
        }


@dataclass
class SensorConnectorConfig:
    """Sensor connector configuration."""
    # Connection settings
    poll_interval_ms: int = 1000
    timeout_ms: int = 5000
    retry_count: int = 3
    retry_delay_ms: int = 1000

    # Time synchronization
    time_source: str = "system"  # system, ntp, gps
    max_clock_drift_ms: int = 100

    # Batch settings
    batch_size: int = 50
    batch_timeout_ms: int = 100

    # Quality thresholds
    stale_timeout_s: float = 10.0
    min_good_quality_percent: float = 90.0


class BaseSensor(ABC):
    """Abstract base class for sensor implementations."""

    def __init__(
        self,
        sensor_id: str,
        sensor_type: SensorType,
        unit: str,
        conditioner_config: Optional[SignalConditionerConfig] = None,
    ) -> None:
        """Initialize base sensor."""
        self.sensor_id = sensor_id
        self.sensor_type = sensor_type
        self.unit = unit
        self.conditioner = SignalConditioner(
            sensor_id,
            conditioner_config or SignalConditionerConfig()
        )
        self._last_reading: Optional[SensorReading] = None
        self._connected = False

    @abstractmethod
    async def read(self) -> SensorReading:
        """Read current sensor value."""
        pass

    @abstractmethod
    async def connect(self) -> bool:
        """Connect to sensor."""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from sensor."""
        pass

    @property
    def last_reading(self) -> Optional[SensorReading]:
        """Get last reading."""
        return self._last_reading

    @property
    def is_connected(self) -> bool:
        """Check connection status."""
        return self._connected


class PressureSensor(BaseSensor):
    """
    Pressure sensor connector.

    Supports:
    - Gauge pressure (psig, barg)
    - Absolute pressure (psia, bara)
    - Differential pressure (psid, bard)
    """

    def __init__(
        self,
        sensor_id: str,
        unit: str = "kPa",
        node_id: Optional[str] = None,
        conditioner_config: Optional[SignalConditionerConfig] = None,
    ) -> None:
        """Initialize pressure sensor."""
        config = conditioner_config or SignalConditionerConfig(
            filter_type=FilterType.MOVING_AVERAGE,
            window_size=5,
            raw_min=4.0,
            raw_max=20.0,
            scaled_min=0.0,
            scaled_max=1000.0,  # 0-1000 kPa
            clamp_min=0.0,
            clamp_max=2000.0,
        )
        super().__init__(sensor_id, SensorType.PRESSURE, unit, config)
        self.node_id = node_id or f"ns=2;s={sensor_id}"

    async def connect(self) -> bool:
        """Connect to pressure sensor."""
        # In production, connect via OPC-UA or Modbus
        self._connected = True
        logger.info(f"Pressure sensor connected: {self.sensor_id}")
        return True

    async def disconnect(self) -> None:
        """Disconnect from pressure sensor."""
        self._connected = False
        logger.info(f"Pressure sensor disconnected: {self.sensor_id}")

    async def read(self) -> SensorReading:
        """Read pressure value."""
        if not self._connected:
            raise ConnectionError(f"Sensor not connected: {self.sensor_id}")

        # In production, read from OPC-UA/Modbus
        # For framework, simulate realistic pressure value
        import random
        base_pressure = 500.0  # kPa
        noise = random.gauss(0, 2)
        raw_value = 12.0 + noise * 0.1  # 4-20mA scaled

        reading = self.conditioner.process(
            raw_value=raw_value,
            raw_status=0,
            sensor_type=self.sensor_type,
            unit=self.unit,
        )
        self._last_reading = reading
        return reading


class TemperatureSensor(BaseSensor):
    """
    Temperature sensor connector.

    Supports:
    - RTD (PT100, PT1000)
    - Thermocouple (J, K, T, E, N, S, R, B)
    - Thermistor
    """

    def __init__(
        self,
        sensor_id: str,
        unit: str = "degC",
        sensor_element: str = "PT100",
        node_id: Optional[str] = None,
        conditioner_config: Optional[SignalConditionerConfig] = None,
    ) -> None:
        """Initialize temperature sensor."""
        config = conditioner_config or SignalConditionerConfig(
            filter_type=FilterType.EXPONENTIAL,
            alpha=0.2,
            raw_min=4.0,
            raw_max=20.0,
            scaled_min=0.0,
            scaled_max=300.0,  # 0-300 degC
            clamp_min=-50.0,
            clamp_max=500.0,
        )
        super().__init__(sensor_id, SensorType.TEMPERATURE, unit, config)
        self.sensor_element = sensor_element
        self.node_id = node_id or f"ns=2;s={sensor_id}"

    async def connect(self) -> bool:
        """Connect to temperature sensor."""
        self._connected = True
        logger.info(f"Temperature sensor connected: {self.sensor_id}")
        return True

    async def disconnect(self) -> None:
        """Disconnect from temperature sensor."""
        self._connected = False

    async def read(self) -> SensorReading:
        """Read temperature value."""
        if not self._connected:
            raise ConnectionError(f"Sensor not connected: {self.sensor_id}")

        import random
        base_temp = 180.0  # degC (saturated steam at ~10 bar)
        noise = random.gauss(0, 0.5)
        raw_value = 12.0 + noise * 0.05

        reading = self.conditioner.process(
            raw_value=raw_value,
            raw_status=0,
            sensor_type=self.sensor_type,
            unit=self.unit,
        )
        self._last_reading = reading
        return reading


class FlowSensor(BaseSensor):
    """
    Flow sensor connector.

    Supports:
    - Vortex flowmeters
    - Orifice plate with DP transmitter
    - Coriolis mass flow
    - Ultrasonic
    """

    def __init__(
        self,
        sensor_id: str,
        unit: str = "kg/s",
        meter_type: str = "vortex",
        k_factor: float = 1.0,
        node_id: Optional[str] = None,
        conditioner_config: Optional[SignalConditionerConfig] = None,
    ) -> None:
        """Initialize flow sensor."""
        config = conditioner_config or SignalConditionerConfig(
            filter_type=FilterType.MOVING_AVERAGE,
            window_size=10,
            raw_min=4.0,
            raw_max=20.0,
            scaled_min=0.0,
            scaled_max=50.0,  # 0-50 kg/s
            clamp_min=0.0,
            clamp_max=100.0,
        )
        super().__init__(sensor_id, SensorType.FLOW, unit, config)
        self.meter_type = meter_type
        self.k_factor = k_factor
        self.node_id = node_id or f"ns=2;s={sensor_id}"

    async def connect(self) -> bool:
        """Connect to flow sensor."""
        self._connected = True
        logger.info(f"Flow sensor connected: {self.sensor_id}")
        return True

    async def disconnect(self) -> None:
        """Disconnect from flow sensor."""
        self._connected = False

    async def read(self) -> SensorReading:
        """Read flow value."""
        if not self._connected:
            raise ConnectionError(f"Sensor not connected: {self.sensor_id}")

        import random
        base_flow = 25.0  # kg/s
        noise = random.gauss(0, 1)
        raw_value = 12.0 + noise * 0.2

        reading = self.conditioner.process(
            raw_value=raw_value,
            raw_status=0,
            sensor_type=self.sensor_type,
            unit=self.unit,
        )
        self._last_reading = reading
        return reading


class ChemistryAnalyzer(BaseSensor):
    """
    Chemistry analyzer connector.

    Supports:
    - pH analyzers
    - Conductivity (specific, cation)
    - Dissolved oxygen
    - Silica analyzers
    """

    def __init__(
        self,
        sensor_id: str,
        measurement_type: str,  # ph, conductivity, dissolved_o2, silica
        unit: str,
        sample_interval_s: float = 60.0,
        node_id: Optional[str] = None,
        conditioner_config: Optional[SignalConditionerConfig] = None,
    ) -> None:
        """Initialize chemistry analyzer."""
        sensor_type_map = {
            "ph": SensorType.PH,
            "conductivity": SensorType.CONDUCTIVITY,
            "dissolved_o2": SensorType.DISSOLVED_O2,
            "silica": SensorType.SILICA,
        }
        sensor_type = sensor_type_map.get(measurement_type, SensorType.CONDUCTIVITY)

        config = conditioner_config or SignalConditionerConfig(
            filter_type=FilterType.MEDIAN,
            window_size=3,
        )
        super().__init__(sensor_id, sensor_type, unit, config)
        self.measurement_type = measurement_type
        self.sample_interval_s = sample_interval_s
        self.node_id = node_id or f"ns=2;s={sensor_id}"

    async def connect(self) -> bool:
        """Connect to chemistry analyzer."""
        self._connected = True
        logger.info(f"Chemistry analyzer connected: {self.sensor_id}")
        return True

    async def disconnect(self) -> None:
        """Disconnect from chemistry analyzer."""
        self._connected = False

    async def read(self) -> SensorReading:
        """Read chemistry value."""
        if not self._connected:
            raise ConnectionError(f"Analyzer not connected: {self.sensor_id}")

        import random
        # Simulate realistic chemistry values
        if self.measurement_type == "ph":
            value = 9.2 + random.gauss(0, 0.1)  # Target pH 9.0-9.5
            raw_value = 12.0
        elif self.measurement_type == "conductivity":
            value = 5.0 + random.gauss(0, 0.5)  # uS/cm
            raw_value = 10.0
        elif self.measurement_type == "dissolved_o2":
            value = 7.0 + random.gauss(0, 0.5)  # ppb
            raw_value = 8.0
        elif self.measurement_type == "silica":
            value = 10.0 + random.gauss(0, 1.0)  # ppb
            raw_value = 9.0
        else:
            value = 0.0
            raw_value = 4.0

        return SensorReading(
            sensor_id=self.sensor_id,
            sensor_type=self.sensor_type,
            raw_value=raw_value,
            value=round(value, 4),
            unit=self.unit,
            timestamp=datetime.now(timezone.utc),
            quality=QualityCode.GOOD,
        )


class SeparatorDPSensor(BaseSensor):
    """
    Separator differential pressure sensor.

    Measures pressure drop across moisture separator for:
    - Separation efficiency estimation
    - Fouling detection
    - Performance monitoring
    """

    def __init__(
        self,
        sensor_id: str,
        unit: str = "kPa",
        design_dp_kpa: float = 10.0,
        node_id: Optional[str] = None,
        conditioner_config: Optional[SignalConditionerConfig] = None,
    ) -> None:
        """Initialize separator DP sensor."""
        config = conditioner_config or SignalConditionerConfig(
            filter_type=FilterType.MOVING_AVERAGE,
            window_size=10,
            raw_min=4.0,
            raw_max=20.0,
            scaled_min=0.0,
            scaled_max=50.0,  # 0-50 kPa DP
            clamp_min=0.0,
            clamp_max=100.0,
        )
        super().__init__(sensor_id, SensorType.SEPARATOR_DP, unit, config)
        self.design_dp_kpa = design_dp_kpa
        self.node_id = node_id or f"ns=2;s={sensor_id}"

    async def connect(self) -> bool:
        """Connect to separator DP sensor."""
        self._connected = True
        logger.info(f"Separator DP sensor connected: {self.sensor_id}")
        return True

    async def disconnect(self) -> None:
        """Disconnect from separator DP sensor."""
        self._connected = False

    async def read(self) -> SensorReading:
        """Read separator DP value."""
        if not self._connected:
            raise ConnectionError(f"Sensor not connected: {self.sensor_id}")

        import random
        # Simulate separator DP around design value
        dp = self.design_dp_kpa * (1.0 + random.gauss(0, 0.05))
        raw_value = 12.0 + random.gauss(0, 0.1)

        reading = self.conditioner.process(
            raw_value=raw_value,
            raw_status=0,
            sensor_type=self.sensor_type,
            unit=self.unit,
        )
        self._last_reading = reading
        return reading


class DrainValveSensor(BaseSensor):
    """
    Drain valve position and duty cycle sensor.

    Monitors:
    - Valve position (0-100%)
    - Drain duty cycle
    - Cycle count
    - Open/close time
    """

    def __init__(
        self,
        sensor_id: str,
        unit: str = "%",
        valve_type: str = "modulating",  # modulating, on_off
        node_id: Optional[str] = None,
        conditioner_config: Optional[SignalConditionerConfig] = None,
    ) -> None:
        """Initialize drain valve sensor."""
        config = conditioner_config or SignalConditionerConfig(
            filter_type=FilterType.MOVING_AVERAGE,
            window_size=3,
            raw_min=4.0,
            raw_max=20.0,
            scaled_min=0.0,
            scaled_max=100.0,
            clamp_min=0.0,
            clamp_max=100.0,
        )
        super().__init__(sensor_id, SensorType.VALVE_POSITION, unit, config)
        self.valve_type = valve_type
        self.node_id = node_id or f"ns=2;s={sensor_id}"

        # Duty cycle tracking
        self._cycle_count = 0
        self._last_state = False
        self._open_time_total_s = 0.0
        self._cycle_start: Optional[datetime] = None

    async def connect(self) -> bool:
        """Connect to drain valve sensor."""
        self._connected = True
        logger.info(f"Drain valve sensor connected: {self.sensor_id}")
        return True

    async def disconnect(self) -> None:
        """Disconnect from drain valve sensor."""
        self._connected = False

    async def read(self) -> SensorReading:
        """Read drain valve position."""
        if not self._connected:
            raise ConnectionError(f"Sensor not connected: {self.sensor_id}")

        import random
        # Simulate valve position
        if self.valve_type == "modulating":
            position = 15.0 + random.gauss(0, 2)  # ~15% open typically
        else:
            # On/off valve - simulate cycling
            position = 100.0 if random.random() < 0.1 else 0.0

        raw_value = 4.0 + (position / 100.0) * 16.0

        reading = self.conditioner.process(
            raw_value=raw_value,
            raw_status=0,
            sensor_type=self.sensor_type,
            unit=self.unit,
        )

        # Track duty cycle for on/off valves
        if self.valve_type == "on_off":
            is_open = reading.value > 50.0
            if is_open and not self._last_state:
                self._cycle_count += 1
                self._cycle_start = reading.timestamp
            elif not is_open and self._last_state and self._cycle_start:
                self._open_time_total_s += (
                    reading.timestamp - self._cycle_start
                ).total_seconds()
            self._last_state = is_open

        self._last_reading = reading
        return reading

    def get_duty_cycle(self, period_s: float = 3600.0) -> float:
        """Get duty cycle as percentage over period."""
        if period_s <= 0:
            return 0.0
        return min(100.0, (self._open_time_total_s / period_s) * 100.0)

    def get_cycle_count(self) -> int:
        """Get total cycle count."""
        return self._cycle_count


@dataclass
class SensorSubscription:
    """Subscription for sensor data callbacks."""
    sensor_id: str
    callback: Callable[[SensorReading], None]
    interval_ms: int = 1000
    filter_quality: Optional[QualityCode] = None

    # State
    is_active: bool = False
    last_callback_time: Optional[datetime] = None


@dataclass
class SensorBatch:
    """Batch of sensor readings."""
    readings: Dict[str, SensorReading]
    timestamp: datetime
    total_count: int
    good_count: int
    uncertain_count: int
    bad_count: int
    quality_score: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "readings": {k: v.to_dict() for k, v in self.readings.items()},
            "total_count": self.total_count,
            "good_count": self.good_count,
            "uncertain_count": self.uncertain_count,
            "bad_count": self.bad_count,
            "quality_score": self.quality_score,
        }


class SensorConnector:
    """
    Unified sensor connector for steam quality control.

    Manages connections to all sensor types and provides:
    - Batch reading for efficiency
    - Subscription-based updates
    - Quality aggregation
    - Time synchronization

    Example:
        connector = SensorConnector(SensorConnectorConfig())

        # Add sensors
        connector.add_sensor(PressureSensor("PT-001", "kPa"))
        connector.add_sensor(TemperatureSensor("TT-001", "degC"))
        connector.add_sensor(FlowSensor("FT-001", "kg/s"))

        # Connect all
        await connector.connect_all()

        # Read batch
        batch = await connector.read_batch()

        # Subscribe to updates
        connector.subscribe("PT-001", on_pressure_change)
    """

    def __init__(self, config: Optional[SensorConnectorConfig] = None) -> None:
        """Initialize sensor connector."""
        self.config = config or SensorConnectorConfig()
        self._sensors: Dict[str, BaseSensor] = {}
        self._subscriptions: Dict[str, List[SensorSubscription]] = {}
        self._poll_task: Optional[asyncio.Task] = None
        self._running = False

        # Statistics
        self._stats = {
            "reads": 0,
            "errors": 0,
            "batches": 0,
            "callbacks": 0,
        }

        logger.info("SensorConnector initialized")

    def add_sensor(self, sensor: BaseSensor) -> None:
        """Add sensor to connector."""
        self._sensors[sensor.sensor_id] = sensor
        self._subscriptions[sensor.sensor_id] = []
        logger.debug(f"Added sensor: {sensor.sensor_id}")

    def remove_sensor(self, sensor_id: str) -> Optional[BaseSensor]:
        """Remove sensor from connector."""
        sensor = self._sensors.pop(sensor_id, None)
        self._subscriptions.pop(sensor_id, None)
        return sensor

    def get_sensor(self, sensor_id: str) -> Optional[BaseSensor]:
        """Get sensor by ID."""
        return self._sensors.get(sensor_id)

    async def connect_all(self) -> Dict[str, bool]:
        """Connect all sensors."""
        results = {}
        for sensor_id, sensor in self._sensors.items():
            try:
                results[sensor_id] = await sensor.connect()
            except Exception as e:
                logger.error(f"Failed to connect sensor {sensor_id}: {e}")
                results[sensor_id] = False
        return results

    async def disconnect_all(self) -> None:
        """Disconnect all sensors."""
        self._running = False
        if self._poll_task:
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass

        for sensor in self._sensors.values():
            try:
                await sensor.disconnect()
            except Exception as e:
                logger.error(f"Error disconnecting sensor {sensor.sensor_id}: {e}")

    async def read_single(self, sensor_id: str) -> Optional[SensorReading]:
        """Read single sensor."""
        sensor = self._sensors.get(sensor_id)
        if not sensor:
            logger.warning(f"Sensor not found: {sensor_id}")
            return None

        try:
            self._stats["reads"] += 1
            return await sensor.read()
        except Exception as e:
            self._stats["errors"] += 1
            logger.error(f"Error reading sensor {sensor_id}: {e}")
            return None

    async def read_batch(
        self,
        sensor_ids: Optional[List[str]] = None,
    ) -> SensorBatch:
        """
        Read batch of sensors.

        Args:
            sensor_ids: Specific sensors to read (None = all)

        Returns:
            SensorBatch with all readings
        """
        timestamp = datetime.now(timezone.utc)
        readings: Dict[str, SensorReading] = {}

        ids_to_read = sensor_ids or list(self._sensors.keys())

        for sensor_id in ids_to_read:
            reading = await self.read_single(sensor_id)
            if reading:
                readings[sensor_id] = reading

        # Calculate quality statistics
        good_count = sum(1 for r in readings.values() if r.quality.is_good())
        uncertain_count = sum(1 for r in readings.values() if r.quality.is_uncertain())
        bad_count = sum(1 for r in readings.values() if r.quality.is_bad())
        total_count = len(readings)

        quality_score = 100.0
        if total_count > 0:
            quality_score = (good_count * 100 + uncertain_count * 50) / total_count

        self._stats["batches"] += 1

        return SensorBatch(
            readings=readings,
            timestamp=timestamp,
            total_count=total_count,
            good_count=good_count,
            uncertain_count=uncertain_count,
            bad_count=bad_count,
            quality_score=round(quality_score, 2),
        )

    def subscribe(
        self,
        sensor_id: str,
        callback: Callable[[SensorReading], None],
        interval_ms: int = 1000,
    ) -> SensorSubscription:
        """Subscribe to sensor updates."""
        if sensor_id not in self._sensors:
            raise ValueError(f"Sensor not found: {sensor_id}")

        subscription = SensorSubscription(
            sensor_id=sensor_id,
            callback=callback,
            interval_ms=interval_ms,
            is_active=True,
        )
        self._subscriptions[sensor_id].append(subscription)
        return subscription

    def unsubscribe(self, subscription: SensorSubscription) -> None:
        """Unsubscribe from sensor updates."""
        subscription.is_active = False
        if subscription.sensor_id in self._subscriptions:
            self._subscriptions[subscription.sensor_id] = [
                s for s in self._subscriptions[subscription.sensor_id]
                if s.is_active
            ]

    async def start_polling(self) -> None:
        """Start background polling for subscriptions."""
        self._running = True
        self._poll_task = asyncio.create_task(self._poll_loop())

    async def _poll_loop(self) -> None:
        """Background polling loop."""
        while self._running:
            try:
                # Group subscriptions by interval
                for sensor_id, subs in self._subscriptions.items():
                    active_subs = [s for s in subs if s.is_active]
                    if not active_subs:
                        continue

                    # Check if any subscription is due
                    now = datetime.now(timezone.utc)
                    for sub in active_subs:
                        if sub.last_callback_time:
                            elapsed_ms = (
                                now - sub.last_callback_time
                            ).total_seconds() * 1000
                            if elapsed_ms < sub.interval_ms:
                                continue

                        # Read and callback
                        reading = await self.read_single(sensor_id)
                        if reading:
                            try:
                                sub.callback(reading)
                                sub.last_callback_time = now
                                self._stats["callbacks"] += 1
                            except Exception as e:
                                logger.error(f"Callback error for {sensor_id}: {e}")

                await asyncio.sleep(self.config.poll_interval_ms / 1000.0)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Poll loop error: {e}")

    def get_statistics(self) -> Dict[str, Any]:
        """Get connector statistics."""
        return {
            **self._stats,
            "sensors": len(self._sensors),
            "active_subscriptions": sum(
                len([s for s in subs if s.is_active])
                for subs in self._subscriptions.values()
            ),
            "connected_sensors": sum(
                1 for s in self._sensors.values() if s.is_connected
            ),
        }


def create_steam_quality_sensors() -> List[BaseSensor]:
    """
    Create standard sensor set for steam quality control.

    Returns:
        List of configured sensors for typical steam quality monitoring
    """
    sensors: List[BaseSensor] = []

    # Pressure sensors
    sensors.append(PressureSensor(
        sensor_id="PT-100",
        unit="kPa",
        conditioner_config=SignalConditionerConfig(
            filter_type=FilterType.MOVING_AVERAGE,
            window_size=5,
            scaled_min=0.0,
            scaled_max=2000.0,
        ),
    ))

    # Temperature sensors
    sensors.append(TemperatureSensor(
        sensor_id="TT-100",
        unit="degC",
        sensor_element="PT100",
        conditioner_config=SignalConditionerConfig(
            filter_type=FilterType.EXPONENTIAL,
            alpha=0.2,
            scaled_min=0.0,
            scaled_max=300.0,
        ),
    ))

    # Flow sensor
    sensors.append(FlowSensor(
        sensor_id="FT-100",
        unit="kg/s",
        meter_type="vortex",
        conditioner_config=SignalConditionerConfig(
            filter_type=FilterType.MOVING_AVERAGE,
            window_size=10,
            scaled_min=0.0,
            scaled_max=100.0,
        ),
    ))

    # Separator DP sensor
    sensors.append(SeparatorDPSensor(
        sensor_id="PDT-101",
        unit="kPa",
        design_dp_kpa=15.0,
    ))

    # Drain valve sensors
    sensors.append(DrainValveSensor(
        sensor_id="ZT-101",
        unit="%",
        valve_type="modulating",
    ))

    # Chemistry analyzers
    sensors.append(ChemistryAnalyzer(
        sensor_id="AIT-101",
        measurement_type="conductivity",
        unit="uS/cm",
        sample_interval_s=60.0,
    ))

    sensors.append(ChemistryAnalyzer(
        sensor_id="AIT-102",
        measurement_type="ph",
        unit="pH",
        sample_interval_s=60.0,
    ))

    sensors.append(ChemistryAnalyzer(
        sensor_id="AIT-103",
        measurement_type="silica",
        unit="ppb",
        sample_interval_s=300.0,
    ))

    return sensors
