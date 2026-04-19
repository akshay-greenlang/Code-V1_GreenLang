"""
GL-020 ECONOPULSE - Sensor Connector Module

Enterprise-grade connectors for economizer instrumentation sensors including:
- RTD Temperature Sensors (PT100/PT1000)
- Thermocouple Connectors (Type J, K, T)
- Flow Meter Connectors (Orifice, Vortex, Ultrasonic)
- Pressure Transducer Connectors (Differential pressure)

Supports protocols: 4-20mA, HART, Modbus RTU
Implements calibration management and bad sensor detection.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import asyncio
import logging
import threading
import struct
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


# =============================================================================
# Enumerations
# =============================================================================

class SensorProtocol(Enum):
    """Supported sensor communication protocols."""
    ANALOG_4_20MA = auto()
    HART = auto()
    MODBUS_RTU = auto()
    MODBUS_TCP = auto()
    OPC_UA = auto()


class SensorStatus(Enum):
    """Sensor operational status."""
    ONLINE = "online"
    OFFLINE = "offline"
    FAULT = "fault"
    CALIBRATING = "calibrating"
    STUCK = "stuck"
    OUT_OF_RANGE = "out_of_range"
    NOISY = "noisy"
    UNKNOWN = "unknown"


class SensorType(Enum):
    """Types of sensors supported."""
    RTD_PT100 = "PT100"
    RTD_PT1000 = "PT1000"
    THERMOCOUPLE_J = "TC_J"
    THERMOCOUPLE_K = "TC_K"
    THERMOCOUPLE_T = "TC_T"
    FLOW_ORIFICE = "FLOW_ORIFICE"
    FLOW_VORTEX = "FLOW_VORTEX"
    FLOW_ULTRASONIC = "FLOW_ULTRASONIC"
    PRESSURE_DIFFERENTIAL = "PRESSURE_DP"
    PRESSURE_GAUGE = "PRESSURE_GAUGE"
    PRESSURE_ABSOLUTE = "PRESSURE_ABS"


class DataQuality(Enum):
    """Data quality indicators."""
    GOOD = "good"
    UNCERTAIN = "uncertain"
    BAD = "bad"
    SUBSTITUTED = "substituted"


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class CalibrationData:
    """Sensor calibration data and offsets."""
    sensor_id: str
    calibration_date: datetime
    next_calibration_due: datetime
    zero_offset: float = 0.0
    span_factor: float = 1.0
    linearization_coefficients: List[float] = field(default_factory=list)
    temperature_compensation: float = 0.0
    calibrated_by: str = ""
    certificate_number: str = ""
    notes: str = ""

    def is_calibration_due(self) -> bool:
        """Check if calibration is due."""
        return datetime.now() >= self.next_calibration_due

    def apply_calibration(self, raw_value: float) -> float:
        """Apply calibration offsets to raw value."""
        # Apply zero offset
        corrected = raw_value - self.zero_offset

        # Apply span factor
        corrected = corrected * self.span_factor

        # Apply linearization if coefficients provided
        if self.linearization_coefficients:
            result = 0.0
            for i, coef in enumerate(self.linearization_coefficients):
                result += coef * (corrected ** i)
            corrected = result

        return corrected


@dataclass
class SensorReading:
    """Sensor reading with metadata and quality information."""
    sensor_id: str
    value: float
    unit: str
    timestamp: datetime
    quality: DataQuality
    status: SensorStatus
    raw_value: Optional[float] = None
    calibrated: bool = True
    confidence: float = 1.0  # 0.0 to 1.0
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert reading to dictionary."""
        return {
            "sensor_id": self.sensor_id,
            "value": self.value,
            "unit": self.unit,
            "timestamp": self.timestamp.isoformat(),
            "quality": self.quality.value,
            "status": self.status.value,
            "raw_value": self.raw_value,
            "calibrated": self.calibrated,
            "confidence": self.confidence,
            "error_message": self.error_message,
        }


@dataclass
class SensorConfig:
    """Sensor configuration parameters."""
    sensor_id: str
    sensor_type: SensorType
    protocol: SensorProtocol
    address: str  # IP, serial port, or device address
    unit_id: int = 1  # Modbus unit ID
    register_address: int = 0  # Modbus register address
    engineering_unit: str = ""
    range_low: float = 0.0
    range_high: float = 100.0
    alarm_low: Optional[float] = None
    alarm_high: Optional[float] = None
    scan_rate_ms: int = 1000
    description: str = ""
    location: str = ""


@dataclass
class BadSensorDetectionConfig:
    """Configuration for bad sensor detection algorithms."""
    stuck_threshold_seconds: float = 60.0  # Time without change to consider stuck
    stuck_tolerance: float = 0.001  # Minimum change threshold
    noise_window_seconds: float = 10.0  # Window for noise calculation
    noise_std_threshold: float = 5.0  # Standard deviation threshold for noise
    rate_of_change_limit: float = 100.0  # Maximum rate of change per second
    out_of_range_margin: float = 0.05  # 5% margin beyond configured range


# =============================================================================
# Circuit Breaker Pattern
# =============================================================================

class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = auto()
    OPEN = auto()
    HALF_OPEN = auto()


class CircuitBreaker:
    """
    Circuit breaker for fault tolerance.

    Prevents cascading failures by failing fast when a service is unavailable.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        expected_exceptions: Tuple[type, ...] = (Exception,),
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exceptions = expected_exceptions

        self._state = CircuitBreakerState.CLOSED
        self._failure_count = 0
        self._last_failure_time: Optional[datetime] = None
        self._lock = threading.Lock()

    @property
    def state(self) -> CircuitBreakerState:
        """Get current circuit breaker state."""
        with self._lock:
            if self._state == CircuitBreakerState.OPEN:
                if self._last_failure_time:
                    elapsed = (datetime.now() - self._last_failure_time).total_seconds()
                    if elapsed >= self.recovery_timeout:
                        self._state = CircuitBreakerState.HALF_OPEN
                        logger.info("Circuit breaker transitioning to HALF_OPEN")
            return self._state

    def record_success(self) -> None:
        """Record a successful call."""
        with self._lock:
            self._failure_count = 0
            if self._state == CircuitBreakerState.HALF_OPEN:
                self._state = CircuitBreakerState.CLOSED
                logger.info("Circuit breaker CLOSED after successful call")

    def record_failure(self) -> None:
        """Record a failed call."""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = datetime.now()

            if self._failure_count >= self.failure_threshold:
                self._state = CircuitBreakerState.OPEN
                logger.warning(
                    f"Circuit breaker OPEN after {self._failure_count} failures"
                )

    def is_available(self) -> bool:
        """Check if calls are allowed."""
        return self.state != CircuitBreakerState.OPEN


# =============================================================================
# Abstract Base Class
# =============================================================================

class SensorConnectorBase(ABC):
    """
    Abstract base class for sensor connectors.

    Provides common functionality for all sensor types including:
    - Connection management with circuit breaker
    - Calibration offset management
    - Bad sensor detection
    - Thread-safe concurrent access
    """

    def __init__(
        self,
        config: SensorConfig,
        calibration: Optional[CalibrationData] = None,
        detection_config: Optional[BadSensorDetectionConfig] = None,
    ):
        self.config = config
        self.calibration = calibration
        self.detection_config = detection_config or BadSensorDetectionConfig()

        self._connected = False
        self._lock = threading.RLock()
        self._reading_history: List[SensorReading] = []
        self._max_history_size = 1000

        # Circuit breaker for fault tolerance
        self._circuit_breaker = CircuitBreaker(
            failure_threshold=5,
            recovery_timeout=30.0,
        )

        # Thread pool for async operations
        self._executor = ThreadPoolExecutor(max_workers=4)

        logger.info(
            f"Initialized {self.__class__.__name__} for sensor {config.sensor_id}"
        )

    @property
    def sensor_id(self) -> str:
        """Get sensor ID."""
        return self.config.sensor_id

    @property
    def is_connected(self) -> bool:
        """Check if sensor is connected."""
        with self._lock:
            return self._connected

    @abstractmethod
    async def connect(self) -> bool:
        """
        Establish connection to sensor.

        Returns:
            True if connection successful, False otherwise.
        """
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from sensor."""
        pass

    @abstractmethod
    async def read_raw(self) -> Tuple[float, datetime]:
        """
        Read raw value from sensor.

        Returns:
            Tuple of (raw_value, timestamp)
        """
        pass

    async def read(self) -> SensorReading:
        """
        Read calibrated value from sensor with quality assessment.

        Returns:
            SensorReading with value, quality, and status information.
        """
        if not self._circuit_breaker.is_available():
            return SensorReading(
                sensor_id=self.sensor_id,
                value=0.0,
                unit=self.config.engineering_unit,
                timestamp=datetime.now(),
                quality=DataQuality.BAD,
                status=SensorStatus.OFFLINE,
                error_message="Circuit breaker open - sensor unavailable",
            )

        try:
            # Read raw value
            raw_value, timestamp = await self.read_raw()

            # Apply calibration
            if self.calibration:
                calibrated_value = self.calibration.apply_calibration(raw_value)
            else:
                calibrated_value = raw_value

            # Assess sensor status
            status = self._assess_sensor_status(calibrated_value, timestamp)

            # Determine data quality
            quality = self._assess_data_quality(calibrated_value, status)

            # Calculate confidence
            confidence = self._calculate_confidence(calibrated_value, status, quality)

            reading = SensorReading(
                sensor_id=self.sensor_id,
                value=calibrated_value,
                unit=self.config.engineering_unit,
                timestamp=timestamp,
                quality=quality,
                status=status,
                raw_value=raw_value,
                calibrated=self.calibration is not None,
                confidence=confidence,
            )

            # Record success
            self._circuit_breaker.record_success()

            # Store in history
            self._add_to_history(reading)

            return reading

        except Exception as e:
            self._circuit_breaker.record_failure()
            logger.error(f"Error reading sensor {self.sensor_id}: {e}")

            return SensorReading(
                sensor_id=self.sensor_id,
                value=0.0,
                unit=self.config.engineering_unit,
                timestamp=datetime.now(),
                quality=DataQuality.BAD,
                status=SensorStatus.FAULT,
                error_message=str(e),
            )

    def _assess_sensor_status(
        self, value: float, timestamp: datetime
    ) -> SensorStatus:
        """
        Assess sensor status based on value and history.

        Detects:
        - Stuck sensors (no change over time)
        - Out-of-range values
        - Noisy sensors (high variance)
        """
        # Check if out of range
        range_margin = (
            self.config.range_high - self.config.range_low
        ) * self.detection_config.out_of_range_margin

        if value < (self.config.range_low - range_margin):
            return SensorStatus.OUT_OF_RANGE
        if value > (self.config.range_high + range_margin):
            return SensorStatus.OUT_OF_RANGE

        # Check for stuck sensor
        if self._is_stuck(value, timestamp):
            return SensorStatus.STUCK

        # Check for noisy sensor
        if self._is_noisy():
            return SensorStatus.NOISY

        return SensorStatus.ONLINE

    def _is_stuck(self, current_value: float, timestamp: datetime) -> bool:
        """Check if sensor is stuck (no change over configured time)."""
        if not self._reading_history:
            return False

        # Get readings within stuck threshold window
        cutoff = timestamp - timedelta(
            seconds=self.detection_config.stuck_threshold_seconds
        )

        recent_readings = [
            r for r in self._reading_history if r.timestamp >= cutoff
        ]

        if len(recent_readings) < 2:
            return False

        # Check if all readings are within tolerance
        for reading in recent_readings:
            if abs(reading.value - current_value) > self.detection_config.stuck_tolerance:
                return False

        return True

    def _is_noisy(self) -> bool:
        """Check if sensor readings are excessively noisy."""
        if len(self._reading_history) < 10:
            return False

        # Calculate standard deviation of recent readings
        recent_values = [r.value for r in self._reading_history[-50:]]

        if len(recent_values) < 2:
            return False

        mean = sum(recent_values) / len(recent_values)
        variance = sum((x - mean) ** 2 for x in recent_values) / len(recent_values)
        std_dev = variance ** 0.5

        return std_dev > self.detection_config.noise_std_threshold

    def _assess_data_quality(
        self, value: float, status: SensorStatus
    ) -> DataQuality:
        """Assess data quality based on value and sensor status."""
        if status in (SensorStatus.FAULT, SensorStatus.OFFLINE):
            return DataQuality.BAD

        if status in (SensorStatus.STUCK, SensorStatus.OUT_OF_RANGE, SensorStatus.NOISY):
            return DataQuality.UNCERTAIN

        # Check calibration status
        if self.calibration and self.calibration.is_calibration_due():
            return DataQuality.UNCERTAIN

        return DataQuality.GOOD

    def _calculate_confidence(
        self, value: float, status: SensorStatus, quality: DataQuality
    ) -> float:
        """Calculate confidence score (0.0 to 1.0)."""
        confidence = 1.0

        # Reduce confidence based on status
        status_penalties = {
            SensorStatus.ONLINE: 0.0,
            SensorStatus.STUCK: 0.4,
            SensorStatus.NOISY: 0.3,
            SensorStatus.OUT_OF_RANGE: 0.5,
            SensorStatus.FAULT: 0.9,
            SensorStatus.OFFLINE: 1.0,
            SensorStatus.UNKNOWN: 0.3,
        }
        confidence -= status_penalties.get(status, 0.2)

        # Reduce confidence if calibration is due
        if self.calibration and self.calibration.is_calibration_due():
            confidence -= 0.2

        return max(0.0, min(1.0, confidence))

    def _add_to_history(self, reading: SensorReading) -> None:
        """Add reading to history buffer."""
        with self._lock:
            self._reading_history.append(reading)

            # Trim history if needed
            if len(self._reading_history) > self._max_history_size:
                self._reading_history = self._reading_history[-self._max_history_size:]

    def get_statistics(self) -> Dict[str, Any]:
        """Get sensor statistics from reading history."""
        with self._lock:
            if not self._reading_history:
                return {}

            values = [r.value for r in self._reading_history if r.quality == DataQuality.GOOD]

            if not values:
                return {"count": len(self._reading_history), "good_readings": 0}

            return {
                "count": len(self._reading_history),
                "good_readings": len(values),
                "min": min(values),
                "max": max(values),
                "mean": sum(values) / len(values),
                "latest": self._reading_history[-1].value,
                "latest_timestamp": self._reading_history[-1].timestamp.isoformat(),
            }

    async def close(self) -> None:
        """Clean up resources."""
        await self.disconnect()
        self._executor.shutdown(wait=True)


# =============================================================================
# RTD Temperature Sensor Connector
# =============================================================================

class RTDTemperatureSensor(SensorConnectorBase):
    """
    RTD Temperature Sensor Connector for PT100/PT1000 sensors.

    Supports:
    - PT100 (100 ohm at 0C)
    - PT1000 (1000 ohm at 0C)
    - 2-wire, 3-wire, and 4-wire configurations
    - IEC 60751 standard
    """

    # Callendar-Van Dusen coefficients for platinum RTDs
    RTD_A = 3.9083e-3
    RTD_B = -5.775e-7
    RTD_C = -4.183e-12  # Only used below 0C

    def __init__(
        self,
        config: SensorConfig,
        calibration: Optional[CalibrationData] = None,
        detection_config: Optional[BadSensorDetectionConfig] = None,
        wire_config: int = 4,  # 2, 3, or 4 wire
    ):
        super().__init__(config, calibration, detection_config)
        self.wire_config = wire_config
        self._base_resistance = 100.0 if config.sensor_type == SensorType.RTD_PT100 else 1000.0
        self._modbus_client = None

    async def connect(self) -> bool:
        """Connect to RTD sensor via configured protocol."""
        try:
            if self.config.protocol == SensorProtocol.MODBUS_RTU:
                # Initialize Modbus RTU client
                from pymodbus.client import AsyncModbusSerialClient

                self._modbus_client = AsyncModbusSerialClient(
                    port=self.config.address,
                    baudrate=9600,
                    parity='N',
                    stopbits=1,
                    bytesize=8,
                    timeout=3,
                )
                await self._modbus_client.connect()
                self._connected = True

            elif self.config.protocol == SensorProtocol.MODBUS_TCP:
                from pymodbus.client import AsyncModbusTcpClient

                host, port = self.config.address.split(':')
                self._modbus_client = AsyncModbusTcpClient(
                    host=host,
                    port=int(port),
                    timeout=3,
                )
                await self._modbus_client.connect()
                self._connected = True

            elif self.config.protocol == SensorProtocol.ANALOG_4_20MA:
                # For 4-20mA, we typically read via a DAQ or PLC
                # Connection is managed by the parent system
                self._connected = True

            logger.info(f"Connected to RTD sensor {self.sensor_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to connect to RTD sensor {self.sensor_id}: {e}")
            self._connected = False
            return False

    async def disconnect(self) -> None:
        """Disconnect from RTD sensor."""
        try:
            if self._modbus_client:
                self._modbus_client.close()
                self._modbus_client = None

            self._connected = False
            logger.info(f"Disconnected from RTD sensor {self.sensor_id}")

        except Exception as e:
            logger.error(f"Error disconnecting RTD sensor {self.sensor_id}: {e}")

    async def read_raw(self) -> Tuple[float, datetime]:
        """Read raw temperature value from RTD sensor."""
        timestamp = datetime.now()

        if self.config.protocol in (SensorProtocol.MODBUS_RTU, SensorProtocol.MODBUS_TCP):
            if not self._modbus_client:
                raise ConnectionError("Modbus client not connected")

            # Read resistance value from Modbus register
            result = await self._modbus_client.read_holding_registers(
                address=self.config.register_address,
                count=2,
                slave=self.config.unit_id,
            )

            if result.isError():
                raise IOError(f"Modbus read error: {result}")

            # Convert registers to float (IEEE 754)
            raw_bytes = struct.pack('>HH', result.registers[0], result.registers[1])
            resistance = struct.unpack('>f', raw_bytes)[0]

            # Convert resistance to temperature
            temperature = self._resistance_to_temperature(resistance)
            return temperature, timestamp

        elif self.config.protocol == SensorProtocol.ANALOG_4_20MA:
            # Simulated read for 4-20mA (actual implementation depends on DAQ)
            # In production, this would interface with an analog input module
            raise NotImplementedError(
                "4-20mA reading requires specific DAQ implementation"
            )

        else:
            raise ValueError(f"Unsupported protocol: {self.config.protocol}")

    def _resistance_to_temperature(self, resistance: float) -> float:
        """
        Convert RTD resistance to temperature using Callendar-Van Dusen equation.

        For T > 0C: R(T) = R0(1 + A*T + B*T^2)
        For T < 0C: R(T) = R0(1 + A*T + B*T^2 + C*(T-100)*T^3)
        """
        r0 = self._base_resistance
        r_ratio = resistance / r0

        # Quadratic solution for T > 0
        a = self.RTD_B
        b = self.RTD_A
        c = 1 - r_ratio

        discriminant = b * b - 4 * a * c
        if discriminant < 0:
            # Handle invalid reading
            logger.warning(f"Invalid RTD resistance: {resistance}")
            return float('nan')

        temperature = (-b + (discriminant ** 0.5)) / (2 * a)

        # If temperature is negative, use iterative solution
        if temperature < 0:
            temperature = self._iterative_rtd_solve(resistance)

        return temperature

    def _iterative_rtd_solve(
        self, resistance: float, tolerance: float = 0.001, max_iter: int = 100
    ) -> float:
        """Iterative solution for RTD equation below 0C."""
        r0 = self._base_resistance
        t_guess = -50.0  # Initial guess

        for _ in range(max_iter):
            # Calculate resistance at guessed temperature
            r_calc = r0 * (
                1
                + self.RTD_A * t_guess
                + self.RTD_B * t_guess ** 2
                + self.RTD_C * (t_guess - 100) * t_guess ** 3
            )

            # Check convergence
            if abs(r_calc - resistance) < tolerance:
                return t_guess

            # Newton-Raphson update
            dr_dt = r0 * (
                self.RTD_A
                + 2 * self.RTD_B * t_guess
                + self.RTD_C * (4 * t_guess ** 3 - 300 * t_guess ** 2)
            )

            t_guess = t_guess - (r_calc - resistance) / dr_dt

        return t_guess


# =============================================================================
# Thermocouple Connector
# =============================================================================

class ThermocoupleConnector(SensorConnectorBase):
    """
    Thermocouple Connector for Type J, K, T thermocouples.

    Supports:
    - Type J (Iron-Constantan): -40C to 750C
    - Type K (Chromel-Alumel): -200C to 1250C
    - Type T (Copper-Constantan): -200C to 350C

    Implements cold junction compensation.
    """

    # Thermocouple voltage ranges (mV) - simplified coefficients
    TC_COEFFICIENTS = {
        SensorType.THERMOCOUPLE_J: {
            "range": (-40, 750),
            "mv_per_c": 0.0517,  # Average coefficient
        },
        SensorType.THERMOCOUPLE_K: {
            "range": (-200, 1250),
            "mv_per_c": 0.0405,
        },
        SensorType.THERMOCOUPLE_T: {
            "range": (-200, 350),
            "mv_per_c": 0.0407,
        },
    }

    def __init__(
        self,
        config: SensorConfig,
        calibration: Optional[CalibrationData] = None,
        detection_config: Optional[BadSensorDetectionConfig] = None,
        cold_junction_source: Optional[str] = None,  # Tag name for CJ temperature
    ):
        super().__init__(config, calibration, detection_config)
        self.cold_junction_source = cold_junction_source
        self._cold_junction_temp = 25.0  # Default CJ temperature
        self._modbus_client = None

    async def connect(self) -> bool:
        """Connect to thermocouple transmitter."""
        try:
            if self.config.protocol == SensorProtocol.MODBUS_RTU:
                from pymodbus.client import AsyncModbusSerialClient

                self._modbus_client = AsyncModbusSerialClient(
                    port=self.config.address,
                    baudrate=9600,
                    parity='N',
                    stopbits=1,
                    bytesize=8,
                    timeout=3,
                )
                await self._modbus_client.connect()
                self._connected = True

            elif self.config.protocol == SensorProtocol.HART:
                # HART protocol implementation
                # In production, this would use a HART modem
                logger.info("HART protocol connection simulated")
                self._connected = True

            logger.info(f"Connected to thermocouple {self.sensor_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to connect to thermocouple {self.sensor_id}: {e}")
            self._connected = False
            return False

    async def disconnect(self) -> None:
        """Disconnect from thermocouple."""
        try:
            if self._modbus_client:
                self._modbus_client.close()
                self._modbus_client = None

            self._connected = False
            logger.info(f"Disconnected from thermocouple {self.sensor_id}")

        except Exception as e:
            logger.error(f"Error disconnecting thermocouple {self.sensor_id}: {e}")

    async def read_raw(self) -> Tuple[float, datetime]:
        """Read raw temperature value from thermocouple."""
        timestamp = datetime.now()

        if self.config.protocol == SensorProtocol.MODBUS_RTU:
            if not self._modbus_client:
                raise ConnectionError("Modbus client not connected")

            # Read millivolt value from Modbus register
            result = await self._modbus_client.read_holding_registers(
                address=self.config.register_address,
                count=2,
                slave=self.config.unit_id,
            )

            if result.isError():
                raise IOError(f"Modbus read error: {result}")

            # Convert registers to float
            raw_bytes = struct.pack('>HH', result.registers[0], result.registers[1])
            millivolts = struct.unpack('>f', raw_bytes)[0]

            # Convert millivolts to temperature with cold junction compensation
            temperature = self._millivolts_to_temperature(millivolts)
            return temperature, timestamp

        elif self.config.protocol == SensorProtocol.HART:
            # HART protocol read (simulated)
            raise NotImplementedError("HART reading requires specific modem implementation")

        else:
            raise ValueError(f"Unsupported protocol: {self.config.protocol}")

    def _millivolts_to_temperature(self, millivolts: float) -> float:
        """
        Convert thermocouple millivolt reading to temperature.

        Applies cold junction compensation.
        """
        tc_type = self.config.sensor_type

        if tc_type not in self.TC_COEFFICIENTS:
            raise ValueError(f"Unknown thermocouple type: {tc_type}")

        coef = self.TC_COEFFICIENTS[tc_type]

        # Calculate cold junction millivolts
        cj_mv = self._cold_junction_temp * coef["mv_per_c"]

        # Total millivolts = measured + cold junction
        total_mv = millivolts + cj_mv

        # Convert to temperature (simplified linear conversion)
        temperature = total_mv / coef["mv_per_c"]

        return temperature

    def set_cold_junction_temperature(self, temperature: float) -> None:
        """Set cold junction reference temperature."""
        self._cold_junction_temp = temperature


# =============================================================================
# Flow Meter Connector
# =============================================================================

class FlowMeterConnector(SensorConnectorBase):
    """
    Flow Meter Connector for various flow measurement technologies.

    Supports:
    - Orifice plate differential pressure flow meters
    - Vortex flow meters
    - Ultrasonic flow meters (transit time and Doppler)

    Provides flow rate in engineering units with temperature/pressure compensation.
    """

    def __init__(
        self,
        config: SensorConfig,
        calibration: Optional[CalibrationData] = None,
        detection_config: Optional[BadSensorDetectionConfig] = None,
        pipe_diameter_mm: float = 100.0,
        orifice_diameter_mm: Optional[float] = None,
        fluid_density_kg_m3: float = 1000.0,
    ):
        super().__init__(config, calibration, detection_config)
        self.pipe_diameter_mm = pipe_diameter_mm
        self.orifice_diameter_mm = orifice_diameter_mm or (pipe_diameter_mm * 0.7)
        self.fluid_density_kg_m3 = fluid_density_kg_m3
        self._modbus_client = None

        # Calculate beta ratio for orifice plates
        self.beta_ratio = self.orifice_diameter_mm / self.pipe_diameter_mm

    async def connect(self) -> bool:
        """Connect to flow meter transmitter."""
        try:
            if self.config.protocol == SensorProtocol.MODBUS_TCP:
                from pymodbus.client import AsyncModbusTcpClient

                host, port = self.config.address.split(':')
                self._modbus_client = AsyncModbusTcpClient(
                    host=host,
                    port=int(port),
                    timeout=3,
                )
                await self._modbus_client.connect()
                self._connected = True

            elif self.config.protocol == SensorProtocol.MODBUS_RTU:
                from pymodbus.client import AsyncModbusSerialClient

                self._modbus_client = AsyncModbusSerialClient(
                    port=self.config.address,
                    baudrate=9600,
                    parity='N',
                    stopbits=1,
                    bytesize=8,
                    timeout=3,
                )
                await self._modbus_client.connect()
                self._connected = True

            logger.info(f"Connected to flow meter {self.sensor_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to connect to flow meter {self.sensor_id}: {e}")
            self._connected = False
            return False

    async def disconnect(self) -> None:
        """Disconnect from flow meter."""
        try:
            if self._modbus_client:
                self._modbus_client.close()
                self._modbus_client = None

            self._connected = False
            logger.info(f"Disconnected from flow meter {self.sensor_id}")

        except Exception as e:
            logger.error(f"Error disconnecting flow meter {self.sensor_id}: {e}")

    async def read_raw(self) -> Tuple[float, datetime]:
        """Read raw flow value from flow meter."""
        timestamp = datetime.now()

        if self.config.protocol in (SensorProtocol.MODBUS_TCP, SensorProtocol.MODBUS_RTU):
            if not self._modbus_client:
                raise ConnectionError("Modbus client not connected")

            # Read flow rate from Modbus register
            result = await self._modbus_client.read_holding_registers(
                address=self.config.register_address,
                count=2,
                slave=self.config.unit_id,
            )

            if result.isError():
                raise IOError(f"Modbus read error: {result}")

            # Convert registers to float
            raw_bytes = struct.pack('>HH', result.registers[0], result.registers[1])
            flow_rate = struct.unpack('>f', raw_bytes)[0]

            return flow_rate, timestamp

        else:
            raise ValueError(f"Unsupported protocol: {self.config.protocol}")

    def calculate_flow_from_dp(
        self,
        differential_pressure_pa: float,
        temperature_c: float = 25.0,
        pressure_kpa: float = 101.325,
    ) -> float:
        """
        Calculate volumetric flow rate from differential pressure (orifice plate).

        Uses ISO 5167 orifice plate equation.

        Args:
            differential_pressure_pa: Differential pressure in Pascals
            temperature_c: Process temperature in Celsius
            pressure_kpa: Process pressure in kPa

        Returns:
            Volumetric flow rate in m3/h
        """
        import math

        # Discharge coefficient (simplified - depends on Reynolds number)
        cd = 0.61

        # Expansibility factor (for liquids = 1.0)
        epsilon = 1.0

        # Orifice area
        d = self.orifice_diameter_mm / 1000  # Convert to meters
        area = math.pi * (d ** 2) / 4

        # Flow rate calculation (ISO 5167)
        if differential_pressure_pa <= 0:
            return 0.0

        beta_term = 1 / math.sqrt(1 - self.beta_ratio ** 4)
        velocity = cd * epsilon * beta_term * math.sqrt(
            2 * differential_pressure_pa / self.fluid_density_kg_m3
        )

        # Volumetric flow rate in m3/s
        flow_m3_s = area * velocity

        # Convert to m3/h
        flow_m3_h = flow_m3_s * 3600

        return flow_m3_h

    async def read_totalizer(self) -> Tuple[float, str]:
        """
        Read flow totalizer value.

        Returns:
            Tuple of (total_volume, unit)
        """
        if not self._modbus_client:
            raise ConnectionError("Modbus client not connected")

        # Totalizer is typically at a different register
        totalizer_register = self.config.register_address + 10

        result = await self._modbus_client.read_holding_registers(
            address=totalizer_register,
            count=4,  # 64-bit value
            slave=self.config.unit_id,
        )

        if result.isError():
            raise IOError(f"Modbus read error: {result}")

        # Convert to double
        raw_bytes = struct.pack(
            '>HHHH',
            result.registers[0],
            result.registers[1],
            result.registers[2],
            result.registers[3],
        )
        total_volume = struct.unpack('>d', raw_bytes)[0]

        return total_volume, "m3"


# =============================================================================
# Pressure Transducer Connector
# =============================================================================

class PressureTransducerConnector(SensorConnectorBase):
    """
    Pressure Transducer Connector for differential, gauge, and absolute pressure.

    Supports:
    - Differential pressure transmitters
    - Gauge pressure transducers
    - Absolute pressure sensors

    Used for economizer tube fouling detection via differential pressure monitoring.
    """

    def __init__(
        self,
        config: SensorConfig,
        calibration: Optional[CalibrationData] = None,
        detection_config: Optional[BadSensorDetectionConfig] = None,
        pressure_range_kpa: Tuple[float, float] = (0.0, 100.0),
    ):
        super().__init__(config, calibration, detection_config)
        self.pressure_range_kpa = pressure_range_kpa
        self._modbus_client = None

    async def connect(self) -> bool:
        """Connect to pressure transducer."""
        try:
            if self.config.protocol == SensorProtocol.MODBUS_TCP:
                from pymodbus.client import AsyncModbusTcpClient

                host, port = self.config.address.split(':')
                self._modbus_client = AsyncModbusTcpClient(
                    host=host,
                    port=int(port),
                    timeout=3,
                )
                await self._modbus_client.connect()
                self._connected = True

            elif self.config.protocol == SensorProtocol.MODBUS_RTU:
                from pymodbus.client import AsyncModbusSerialClient

                self._modbus_client = AsyncModbusSerialClient(
                    port=self.config.address,
                    baudrate=9600,
                    parity='N',
                    stopbits=1,
                    bytesize=8,
                    timeout=3,
                )
                await self._modbus_client.connect()
                self._connected = True

            elif self.config.protocol == SensorProtocol.ANALOG_4_20MA:
                # 4-20mA connection managed by parent system
                self._connected = True

            logger.info(f"Connected to pressure transducer {self.sensor_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to connect to pressure transducer {self.sensor_id}: {e}")
            self._connected = False
            return False

    async def disconnect(self) -> None:
        """Disconnect from pressure transducer."""
        try:
            if self._modbus_client:
                self._modbus_client.close()
                self._modbus_client = None

            self._connected = False
            logger.info(f"Disconnected from pressure transducer {self.sensor_id}")

        except Exception as e:
            logger.error(f"Error disconnecting pressure transducer {self.sensor_id}: {e}")

    async def read_raw(self) -> Tuple[float, datetime]:
        """Read raw pressure value from transducer."""
        timestamp = datetime.now()

        if self.config.protocol in (SensorProtocol.MODBUS_TCP, SensorProtocol.MODBUS_RTU):
            if not self._modbus_client:
                raise ConnectionError("Modbus client not connected")

            # Read pressure from Modbus register
            result = await self._modbus_client.read_holding_registers(
                address=self.config.register_address,
                count=2,
                slave=self.config.unit_id,
            )

            if result.isError():
                raise IOError(f"Modbus read error: {result}")

            # Convert registers to float
            raw_bytes = struct.pack('>HH', result.registers[0], result.registers[1])
            pressure = struct.unpack('>f', raw_bytes)[0]

            return pressure, timestamp

        elif self.config.protocol == SensorProtocol.ANALOG_4_20MA:
            # For 4-20mA, convert mA to pressure
            raise NotImplementedError(
                "4-20mA reading requires specific DAQ implementation"
            )

        else:
            raise ValueError(f"Unsupported protocol: {self.config.protocol}")

    def ma_to_pressure(self, milliamps: float) -> float:
        """
        Convert 4-20mA signal to pressure.

        Args:
            milliamps: Current signal in mA (4-20 range)

        Returns:
            Pressure in kPa
        """
        if milliamps < 4.0 or milliamps > 20.0:
            logger.warning(f"4-20mA signal out of range: {milliamps} mA")

        # Linear scaling
        span = self.pressure_range_kpa[1] - self.pressure_range_kpa[0]
        pressure = self.pressure_range_kpa[0] + (span * (milliamps - 4.0) / 16.0)

        return pressure

    async def check_sensor_health(self) -> Dict[str, Any]:
        """
        Check pressure transducer health status.

        Returns diagnostic information including:
        - Sensor status
        - Signal quality
        - Calibration status
        """
        try:
            reading = await self.read()

            return {
                "sensor_id": self.sensor_id,
                "status": reading.status.value,
                "quality": reading.quality.value,
                "confidence": reading.confidence,
                "current_value": reading.value,
                "unit": reading.unit,
                "calibration_due": (
                    self.calibration.is_calibration_due()
                    if self.calibration
                    else None
                ),
                "circuit_breaker_state": self._circuit_breaker.state.name,
            }

        except Exception as e:
            return {
                "sensor_id": self.sensor_id,
                "status": "error",
                "error_message": str(e),
            }
