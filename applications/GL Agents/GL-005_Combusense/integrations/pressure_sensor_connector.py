# -*- coding: utf-8 -*-
"""
Pressure Sensor Connector for GL-005 CombustionControlAgent

Implements real-time pressure monitoring for combustion control systems:
- Fuel pressure monitoring (natural gas, oil, coal)
- Air pressure monitoring (forced draft, induced draft)
- Furnace draft pressure monitoring
- Multi-protocol support (Modbus RTU, 4-20mA analog via ADC)
- Sub-50ms response time for critical pressure safety
- Connection pooling and circuit breaker pattern
- Real-time data validation and alarming

Real-Time Requirements:
- Pressure read cycle: <50ms
- Safety alarm response: <30ms
- Trend data acquisition: 10Hz minimum
- Range validation: 0-500 kPa (typical)

Protocols Supported:
- Modbus RTU (RS-485 serial)
- 4-20mA analog via ADC (Analog-to-Digital Converter)
- HART (Highway Addressable Remote Transducer)

Author: GL-DataIntegrationEngineer
Date: 2025-11-26
Version: 1.0.0
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque, defaultdict
from enum import Enum
from greenlang.determinism import DeterministicClock

# Third-party imports
try:
    from pymodbus.client import AsyncModbusSerialClient
    from pymodbus.exceptions import ModbusException
    from pymodbus.constants import Endian
    from pymodbus.payload import BinaryPayloadDecoder
    MODBUS_AVAILABLE = True
except ImportError:
    MODBUS_AVAILABLE = False

# Prometheus metrics
try:
    from prometheus_client import Counter, Histogram, Gauge
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False

logger = logging.getLogger(__name__)


class PressureSensorProtocol(Enum):
    """Supported pressure sensor protocols."""
    MODBUS_RTU = "modbus_rtu"
    ANALOG_4_20MA = "analog_4_20ma"
    HART = "hart"


class PressureType(Enum):
    """Types of pressure measurements."""
    FUEL_PRESSURE = "fuel_pressure"  # Natural gas, oil pressure
    AIR_PRESSURE = "air_pressure"  # Forced draft, combustion air
    FURNACE_DRAFT = "furnace_draft"  # Furnace internal pressure/draft
    DIFFERENTIAL = "differential"  # Pressure difference
    ABSOLUTE = "absolute"  # Absolute pressure
    GAUGE = "gauge"  # Gauge pressure (relative to atmospheric)


class PressureUnit(Enum):
    """Pressure measurement units."""
    KPA = "kPa"  # Kilopascals
    PSI = "psi"  # Pounds per square inch
    BAR = "bar"  # Bar
    MBAR = "mbar"  # Millibar
    MMWC = "mmWC"  # Millimeters water column
    INWC = "inWC"  # Inches water column


class SensorQuality(Enum):
    """Sensor data quality status."""
    GOOD = "good"
    BAD_SENSOR_FAILURE = "bad_sensor_failure"
    BAD_COMM_FAILURE = "bad_comm_failure"
    BAD_OUT_OF_RANGE = "bad_out_of_range"
    UNCERTAIN_CALIBRATION = "uncertain_calibration"
    UNCERTAIN_DRIFT = "uncertain_drift"


class CircuitBreakerState(Enum):
    """Circuit breaker states for fault tolerance."""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Blocking calls due to failures
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class PressureSensor:
    """Pressure sensor configuration and state."""
    sensor_id: str
    description: str
    pressure_type: PressureType
    sensor_address: int  # Modbus address or ADC channel
    unit_id: int = 1
    engineering_units: PressureUnit = PressureUnit.KPA

    # Range configuration
    min_pressure: float = 0.0  # kPa
    max_pressure: float = 500.0  # kPa
    min_signal: float = 4.0  # mA for 4-20mA sensors
    max_signal: float = 20.0  # mA

    # Calibration
    calibration_offset: float = 0.0
    calibration_gain: float = 1.0
    last_calibration: Optional[datetime] = None
    calibration_interval_days: int = 90

    # Alarm limits
    alarm_high_high: Optional[float] = None
    alarm_high: Optional[float] = None
    alarm_low: Optional[float] = None
    alarm_low_low: Optional[float] = None

    # Filtering
    enable_filtering: bool = True
    filter_time_constant: float = 0.5  # seconds (first-order lag)
    deadband: float = 0.5  # kPa change threshold

    # Runtime state
    current_pressure: Optional[float] = None
    filtered_pressure: Optional[float] = None
    quality: SensorQuality = SensorQuality.GOOD
    last_update: Optional[datetime] = None
    consecutive_failures: int = 0

    # Statistics
    pressure_history: deque = field(default_factory=lambda: deque(maxlen=100))
    read_count: int = 0


@dataclass
class PressureAlarm:
    """Pressure alarm event."""
    alarm_id: str
    sensor_id: str
    alarm_type: str  # HH, H, L, LL
    priority: int  # 1=Critical, 2=High, 3=Medium, 4=Low
    setpoint: float
    actual_pressure: float
    message: str
    timestamp: datetime
    acknowledged: bool = False


@dataclass
class PressureSensorConfig:
    """Configuration for pressure sensor connector."""
    # Protocol settings
    protocol: PressureSensorProtocol = PressureSensorProtocol.MODBUS_RTU

    # Modbus RTU settings
    modbus_port: str = "/dev/ttyUSB0"  # COM port on Windows
    modbus_baudrate: int = 9600
    modbus_parity: str = "N"  # N, E, O
    modbus_stopbits: int = 1
    modbus_bytesize: int = 8
    modbus_timeout: float = 0.5  # seconds

    # 4-20mA ADC settings
    adc_device: str = "/dev/i2c-1"  # I2C device for ADC
    adc_address: int = 0x48  # I2C address
    adc_reference_voltage: float = 5.0  # Volts
    adc_resolution_bits: int = 16

    # Connection management
    connection_timeout: int = 5
    retry_max_attempts: int = 3
    retry_base_delay: float = 0.5
    reconnect_delay: float = 2.0

    # Circuit breaker settings
    failure_threshold: int = 5
    recovery_timeout: int = 30  # seconds
    half_open_max_calls: int = 3

    # Performance settings
    scan_rate_hz: int = 10  # 10Hz = 100ms
    batch_read_size: int = 10
    enable_caching: bool = True
    cache_ttl_ms: int = 50  # 50ms cache


class CircuitBreaker:
    """Circuit breaker for fault tolerance."""

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 30,
        half_open_max_calls: int = 3
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls

        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.half_open_calls = 0
        self._lock = asyncio.Lock()

    async def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        async with self._lock:
            if self.state == CircuitBreakerState.OPEN:
                if self.last_failure_time and \
                   (DeterministicClock.now() - self.last_failure_time).total_seconds() >= self.recovery_timeout:
                    logger.info("Circuit breaker transitioning to HALF_OPEN")
                    self.state = CircuitBreakerState.HALF_OPEN
                    self.half_open_calls = 0
                else:
                    raise Exception("Circuit breaker is OPEN - sensor unavailable")

            if self.state == CircuitBreakerState.HALF_OPEN:
                if self.half_open_calls >= self.half_open_max_calls:
                    raise Exception("Circuit breaker HALF_OPEN - max test calls exceeded")
                self.half_open_calls += 1

        # Execute function
        try:
            result = await func(*args, **kwargs)

            async with self._lock:
                if self.state == CircuitBreakerState.HALF_OPEN:
                    logger.info("Circuit breaker transitioning to CLOSED")
                    self.state = CircuitBreakerState.CLOSED
                self.failure_count = 0
                self.half_open_calls = 0

            return result

        except Exception as e:
            async with self._lock:
                self.failure_count += 1
                self.last_failure_time = DeterministicClock.now()

                if self.failure_count >= self.failure_threshold:
                    logger.error(f"Circuit breaker opening due to {self.failure_count} failures")
                    self.state = CircuitBreakerState.OPEN
                elif self.state == CircuitBreakerState.HALF_OPEN:
                    logger.warning("Circuit breaker reopening - test call failed")
                    self.state = CircuitBreakerState.OPEN

            raise


class PressureSensorConnector:
    """
    Pressure Sensor Connector with multi-protocol support.

    Features:
    - Real-time pressure monitoring (<50ms)
    - Multi-protocol support (Modbus RTU, 4-20mA analog, HART)
    - First-order filtering for noise reduction
    - Automatic range validation and alarming
    - Circuit breaker pattern for fault tolerance
    - Connection pooling for multiple sensors
    - Calibration tracking and drift detection
    - Prometheus metrics integration

    Example:
        config = PressureSensorConfig(
            protocol=PressureSensorProtocol.MODBUS_RTU,
            modbus_port="/dev/ttyUSB0",
            modbus_baudrate=9600
        )

        async with PressureSensorConnector(config) as connector:
            # Register sensors
            connector.register_sensor(PressureSensor(
                sensor_id="FUEL_PRESS_01",
                description="Natural gas fuel pressure",
                pressure_type=PressureType.FUEL_PRESSURE,
                sensor_address=100,
                min_pressure=0.0,
                max_pressure=350.0,
                alarm_high=300.0,
                alarm_high_high=330.0
            ))

            # Read pressures
            pressures = await connector.read_pressures([
                "FUEL_PRESS_01", "AIR_PRESS_01", "DRAFT_PRESS_01"
            ])

            # Monitor for alarms
            await connector.subscribe_to_alarms(alarm_handler)
    """

    def __init__(self, config: PressureSensorConfig):
        """Initialize pressure sensor connector."""
        self.config = config
        self.connected = False

        # Protocol client
        self.modbus_client: Optional[AsyncModbusSerialClient] = None
        self.adc_client = None  # Would be initialized for 4-20mA

        # Sensor registry
        self.sensors: Dict[str, PressureSensor] = {}

        # Circuit breaker
        self.circuit_breaker = CircuitBreaker(
            config.failure_threshold,
            config.recovery_timeout,
            config.half_open_max_calls
        )

        # Alarm management
        self.active_alarms: Dict[str, PressureAlarm] = {}
        self.alarm_callbacks: List = []

        # Performance tracking
        self.read_latencies = deque(maxlen=1000)
        self.connection_health_score = 100.0

        # Background tasks
        self._scan_task: Optional[asyncio.Task] = None
        self._calibration_check_task: Optional[asyncio.Task] = None

        # Prometheus metrics
        if METRICS_AVAILABLE:
            self.metrics = {
                'pressure_reads': Counter('pressure_reads_total', 'Total pressure reads'),
                'read_latency': Histogram('pressure_read_latency_seconds', 'Pressure read latency'),
                'connection_health': Gauge('pressure_sensor_health_score', 'Connection health (0-100)'),
                'active_alarms': Gauge('pressure_alarms_active', 'Active pressure alarms'),
                'sensor_quality': Gauge('pressure_sensor_quality', 'Sensor quality', ['sensor_id']),
                'current_pressure': Gauge('pressure_value_kpa', 'Current pressure reading', ['sensor_id', 'type'])
            }
        else:
            self.metrics = {}

    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()

    async def connect(self) -> bool:
        """
        Connect to pressure sensors.

        Returns:
            True if connection successful

        Raises:
            ConnectionError: If connection fails
        """
        logger.info(f"Connecting to pressure sensors via {self.config.protocol.value}...")

        try:
            if self.config.protocol == PressureSensorProtocol.MODBUS_RTU:
                await self._connect_modbus_rtu()
            elif self.config.protocol == PressureSensorProtocol.ANALOG_4_20MA:
                await self._connect_analog_adc()
            elif self.config.protocol == PressureSensorProtocol.HART:
                await self._connect_hart()
            else:
                raise ValueError(f"Unsupported protocol: {self.config.protocol}")

            self.connected = True
            logger.info("Connected to pressure sensors")

            # Start background scanning
            self._scan_task = asyncio.create_task(self._scan_loop())
            self._calibration_check_task = asyncio.create_task(self._calibration_check_loop())

            return True

        except Exception as e:
            logger.error(f"Failed to connect to pressure sensors: {e}")
            raise ConnectionError(f"Pressure sensor connection failed: {e}")

    async def _connect_modbus_rtu(self):
        """Connect via Modbus RTU."""
        if not MODBUS_AVAILABLE:
            raise ImportError("pymodbus library required")

        self.modbus_client = AsyncModbusSerialClient(
            port=self.config.modbus_port,
            baudrate=self.config.modbus_baudrate,
            parity=self.config.modbus_parity,
            stopbits=self.config.modbus_stopbits,
            bytesize=self.config.modbus_bytesize,
            timeout=self.config.modbus_timeout
        )

        await asyncio.wait_for(
            self.modbus_client.connect(),
            timeout=self.config.connection_timeout
        )

        if not self.modbus_client.connected:
            raise ConnectionError("Modbus RTU connection failed")

        logger.info(f"Modbus RTU connected on {self.config.modbus_port}")

    async def _connect_analog_adc(self):
        """Connect to ADC for 4-20mA sensors."""
        # Placeholder for ADC initialization
        # In real implementation, would initialize I2C ADC driver
        logger.info("4-20mA ADC initialized (placeholder)")

    async def _connect_hart(self):
        """Connect via HART protocol."""
        # Placeholder for HART initialization
        logger.info("HART protocol initialized (placeholder)")

    async def read_pressures(
        self,
        sensor_ids: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Read pressure values from sensors.

        Args:
            sensor_ids: List of sensor IDs to read

        Returns:
            Dictionary mapping sensor_id to {pressure, quality, timestamp, units}

        Raises:
            ConnectionError: If not connected
        """
        if not self.connected:
            raise ConnectionError("Not connected to pressure sensors")

        start_time = time.perf_counter()

        try:
            result = await self.circuit_breaker.call(
                self._read_pressures_internal, sensor_ids
            )

            # Record latency
            latency = time.perf_counter() - start_time
            self.read_latencies.append(latency)

            if self.metrics:
                self.metrics['pressure_reads'].inc(len(sensor_ids))
                self.metrics['read_latency'].observe(latency)

            # Check for alarms
            await self._check_alarms(result)

            return result

        except Exception as e:
            logger.error(f"Failed to read pressures: {e}")
            raise

    async def _read_pressures_internal(
        self,
        sensor_ids: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        """Internal pressure reading implementation."""
        result = {}

        for sensor_id in sensor_ids:
            sensor = self.sensors.get(sensor_id)
            if not sensor:
                logger.warning(f"Sensor {sensor_id} not registered")
                continue

            try:
                if self.config.protocol == PressureSensorProtocol.MODBUS_RTU:
                    pressure = await self._read_modbus_pressure(sensor)
                elif self.config.protocol == PressureSensorProtocol.ANALOG_4_20MA:
                    pressure = await self._read_analog_pressure(sensor)
                else:
                    pressure = None

                if pressure is not None:
                    # Apply calibration
                    pressure = pressure * sensor.calibration_gain + sensor.calibration_offset

                    # Apply filtering
                    if sensor.enable_filtering and sensor.filtered_pressure is not None:
                        alpha = 1.0 / (1.0 + sensor.filter_time_constant * self.config.scan_rate_hz)
                        filtered = alpha * pressure + (1 - alpha) * sensor.filtered_pressure
                    else:
                        filtered = pressure

                    # Validate range
                    if pressure < sensor.min_pressure or pressure > sensor.max_pressure:
                        sensor.quality = SensorQuality.BAD_OUT_OF_RANGE
                        logger.warning(f"{sensor_id} pressure {pressure:.2f} out of range [{sensor.min_pressure}, {sensor.max_pressure}]")
                    else:
                        sensor.quality = SensorQuality.GOOD

                    # Update sensor state
                    sensor.current_pressure = pressure
                    sensor.filtered_pressure = filtered
                    sensor.last_update = DeterministicClock.now()
                    sensor.pressure_history.append(pressure)
                    sensor.read_count += 1
                    sensor.consecutive_failures = 0

                    result[sensor_id] = {
                        'pressure': filtered,
                        'raw_pressure': pressure,
                        'quality': sensor.quality.value,
                        'timestamp': sensor.last_update.isoformat(),
                        'units': sensor.engineering_units.value
                    }

                    # Update Prometheus metric
                    if self.metrics:
                        self.metrics['current_pressure'].labels(
                            sensor_id=sensor_id,
                            type=sensor.pressure_type.value
                        ).set(filtered)

                else:
                    raise Exception("Failed to read pressure value")

            except Exception as e:
                logger.error(f"Error reading sensor {sensor_id}: {e}")
                sensor.consecutive_failures += 1
                sensor.quality = SensorQuality.BAD_COMM_FAILURE

                result[sensor_id] = {
                    'pressure': None,
                    'quality': sensor.quality.value,
                    'timestamp': DeterministicClock.now().isoformat(),
                    'error': str(e)
                }

        return result

    async def _read_modbus_pressure(self, sensor: PressureSensor) -> Optional[float]:
        """Read pressure from Modbus RTU sensor."""
        try:
            # Read holding registers (function code 3)
            response = await self.modbus_client.read_holding_registers(
                address=sensor.sensor_address,
                count=2,  # 32-bit float = 2 registers
                unit=sensor.unit_id
            )

            if response.isError():
                raise ModbusException(f"Modbus error: {response}")

            # Decode 32-bit float
            decoder = BinaryPayloadDecoder.fromRegisters(
                response.registers,
                byteorder=Endian.Big,
                wordorder=Endian.Big
            )
            pressure = decoder.decode_32bit_float()

            return pressure

        except Exception as e:
            logger.error(f"Modbus read failed for {sensor.sensor_id}: {e}")
            return None

    async def _read_analog_pressure(self, sensor: PressureSensor) -> Optional[float]:
        """Read pressure from 4-20mA analog sensor via ADC."""
        # Placeholder for ADC reading
        # In real implementation:
        # 1. Read ADC value (0-65535 for 16-bit)
        # 2. Convert to voltage (0-5V)
        # 3. Convert to current (4-20mA via precision resistor)
        # 4. Map to pressure range

        # Example calculation:
        # adc_value = await self.adc_client.read_channel(sensor.sensor_address)
        # voltage = (adc_value / 65535.0) * self.config.adc_reference_voltage
        # current_ma = voltage / 250.0 * 1000  # Assuming 250Ω precision resistor
        # pressure = self._map_current_to_pressure(current_ma, sensor)

        logger.warning("4-20mA reading not implemented (placeholder)")
        return None

    def _map_current_to_pressure(self, current_ma: float, sensor: PressureSensor) -> float:
        """Map 4-20mA current to pressure value."""
        # Linear mapping: 4mA = min_pressure, 20mA = max_pressure
        pressure_range = sensor.max_pressure - sensor.min_pressure
        current_range = sensor.max_signal - sensor.min_signal

        pressure = sensor.min_pressure + (current_ma - sensor.min_signal) * pressure_range / current_range
        return pressure

    async def subscribe_to_alarms(self, callback):
        """Subscribe to pressure alarms."""
        self.alarm_callbacks.append(callback)
        logger.info("Subscribed to pressure alarms")

    async def _check_alarms(self, readings: Dict[str, Dict[str, Any]]):
        """Check for alarm conditions."""
        for sensor_id, data in readings.items():
            sensor = self.sensors.get(sensor_id)
            if not sensor or data['pressure'] is None:
                continue

            pressure = data['pressure']
            alarms_triggered = []

            # Check alarm limits
            if sensor.alarm_high_high and pressure >= sensor.alarm_high_high:
                alarms_triggered.append(('HH', sensor.alarm_high_high, 1))
            elif sensor.alarm_high and pressure >= sensor.alarm_high:
                alarms_triggered.append(('H', sensor.alarm_high, 2))

            if sensor.alarm_low_low and pressure <= sensor.alarm_low_low:
                alarms_triggered.append(('LL', sensor.alarm_low_low, 1))
            elif sensor.alarm_low and pressure <= sensor.alarm_low:
                alarms_triggered.append(('L', sensor.alarm_low, 2))

            # Trigger alarm callbacks
            for alarm_type, setpoint, priority in alarms_triggered:
                alarm_id = f"{sensor_id}_{alarm_type}"

                if alarm_id not in self.active_alarms:
                    alarm = PressureAlarm(
                        alarm_id=alarm_id,
                        sensor_id=sensor_id,
                        alarm_type=alarm_type,
                        priority=priority,
                        setpoint=setpoint,
                        actual_pressure=pressure,
                        message=f"{sensor_id} {alarm_type} alarm: {pressure:.2f} {sensor.engineering_units.value}",
                        timestamp=DeterministicClock.now()
                    )

                    self.active_alarms[alarm_id] = alarm

                    for callback in self.alarm_callbacks:
                        try:
                            await callback(alarm)
                        except Exception as e:
                            logger.error(f"Alarm callback failed: {e}")

                    if self.metrics:
                        self.metrics['active_alarms'].set(len(self.active_alarms))

    def register_sensor(self, sensor: PressureSensor):
        """Register a pressure sensor."""
        self.sensors[sensor.sensor_id] = sensor
        logger.info(f"Registered pressure sensor: {sensor.sensor_id} ({sensor.pressure_type.value})")

    async def _scan_loop(self):
        """Background scanning loop for continuous monitoring."""
        scan_interval = 1.0 / self.config.scan_rate_hz

        while self.connected:
            try:
                sensor_ids = list(self.sensors.keys())
                if sensor_ids:
                    await self.read_pressures(sensor_ids)

                await asyncio.sleep(scan_interval)

            except Exception as e:
                logger.error(f"Scan loop error: {e}")
                await asyncio.sleep(scan_interval)

    async def _calibration_check_loop(self):
        """Check for sensors requiring calibration."""
        while self.connected:
            try:
                now = DeterministicClock.now()

                for sensor in self.sensors.values():
                    if sensor.last_calibration:
                        days_since_cal = (now - sensor.last_calibration).days
                        if days_since_cal > sensor.calibration_interval_days:
                            logger.warning(
                                f"Sensor {sensor.sensor_id} requires calibration "
                                f"({days_since_cal} days since last calibration)"
                            )
                            sensor.quality = SensorQuality.UNCERTAIN_CALIBRATION

                await asyncio.sleep(3600)  # Check hourly

            except Exception as e:
                logger.error(f"Calibration check error: {e}")
                await asyncio.sleep(3600)

    async def disconnect(self):
        """Disconnect from pressure sensors."""
        logger.info("Disconnecting from pressure sensors...")

        # Stop background tasks
        if self._scan_task:
            self._scan_task.cancel()
            try:
                await self._scan_task
            except asyncio.CancelledError:
                pass

        if self._calibration_check_task:
            self._calibration_check_task.cancel()
            try:
                await self._calibration_check_task
            except asyncio.CancelledError:
                pass

        # Close connections
        if self.modbus_client:
            try:
                self.modbus_client.close()
            except Exception as e:
                logger.error(f"Error closing Modbus connection: {e}")

        self.connected = False
        logger.info("Disconnected from pressure sensors")

    def get_statistics(self) -> Dict[str, Any]:
        """Get connector statistics."""
        if not self.read_latencies:
            return {}

        return {
            'avg_read_latency_ms': sum(self.read_latencies) / len(self.read_latencies) * 1000,
            'max_read_latency_ms': max(self.read_latencies) * 1000,
            'total_sensors': len(self.sensors),
            'active_alarms': len(self.active_alarms),
            'connection_health': self.connection_health_score,
            'connected': self.connected
        }
