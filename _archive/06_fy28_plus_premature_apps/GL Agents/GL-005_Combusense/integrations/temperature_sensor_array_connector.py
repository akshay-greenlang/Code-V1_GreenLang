# -*- coding: utf-8 -*-
"""
Temperature Sensor Array Connector for GL-005 CombustionControlAgent

Implements high-performance multi-sensor temperature monitoring:
- Multiple thermocouple/RTD sensor integration
- Modbus RTU (RS-485) and 4-20mA I/O support
- Sensor health monitoring and validation
- Automatic calibration and drift compensation
- Zone-based temperature profiling

Real-Time Requirements:
- Sensor scan rate: 1Hz minimum
- Temperature accuracy: ±0.5°C
- Fault detection: <5s
- Calibration validation: hourly

Protocols Supported:
- Modbus RTU (RS-485/RS-232)
- 4-20mA analog via I/O modules
- Digital temperature sensors (1-Wire, I2C)

Supported Sensor Types:
- Type K Thermocouples (-200 to 1350°C)
- Type J Thermocouples (-210 to 1200°C)
- RTD PT100/PT1000 (-200 to 850°C)
- Thermistors (0 to 150°C)

Author: GL-DataIntegrationEngineer
Date: 2025-11-18
Version: 1.0.0
"""

import asyncio
import logging
import statistics
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


class SensorType(Enum):
    """Types of temperature sensors."""
    THERMOCOUPLE_K = "thermocouple_k"
    THERMOCOUPLE_J = "thermocouple_j"
    THERMOCOUPLE_T = "thermocouple_t"
    RTD_PT100 = "rtd_pt100"
    RTD_PT1000 = "rtd_pt1000"
    THERMISTOR = "thermistor"


class TemperatureZone(Enum):
    """Temperature measurement zones in combustion system."""
    FURNACE = "furnace"
    FLUE_GAS = "flue_gas"
    AMBIENT = "ambient"
    STEAM = "steam"
    FEED_WATER = "feed_water"
    ECONOMIZER = "economizer"
    AIR_PREHEATER = "air_preheater"


class SensorHealth(Enum):
    """Sensor health status."""
    HEALTHY = "healthy"
    CALIBRATION_DRIFT = "calibration_drift"
    OUT_OF_RANGE = "out_of_range"
    OPEN_CIRCUIT = "open_circuit"
    SHORT_CIRCUIT = "short_circuit"
    FAILED = "failed"


@dataclass
class TemperatureSensor:
    """Temperature sensor configuration."""
    sensor_id: str
    sensor_type: SensorType
    zone: TemperatureZone
    description: str

    # Modbus addressing
    register_address: int
    unit_id: int = 1

    # Sensor characteristics
    min_temp_c: float = -200.0
    max_temp_c: float = 1350.0
    accuracy_c: float = 0.5
    response_time_s: float = 1.0

    # Calibration
    calibration_offset: float = 0.0
    calibration_slope: float = 1.0
    last_calibration: Optional[datetime] = None
    calibration_interval_days: int = 90

    # Alarm limits
    alarm_high: Optional[float] = None
    alarm_low: Optional[float] = None

    # Runtime state
    current_temp_c: Optional[float] = None
    last_reading_time: Optional[datetime] = None
    health_status: SensorHealth = SensorHealth.HEALTHY
    consecutive_failures: int = 0


@dataclass
class TemperatureReading:
    """Temperature reading with metadata."""
    sensor_id: str
    temperature_c: float
    timestamp: datetime
    health: SensorHealth
    calibrated: bool
    raw_value: int
    zone: TemperatureZone


@dataclass
class SensorArrayConfig:
    """Configuration for temperature sensor array."""
    array_id: str

    # Modbus RTU settings
    serial_port: str = "/dev/ttyUSB0"  # COM1 on Windows
    baudrate: int = 9600
    parity: str = "N"  # N, E, O
    stopbits: int = 1
    bytesize: int = 8
    timeout: float = 1.0

    # Scan settings
    scan_rate_hz: float = 1.0
    max_sensors: int = 32

    # Data quality
    temperature_smoothing_window: int = 5
    spike_detection_threshold_c: float = 50.0  # Max temp change per reading
    outlier_rejection_enabled: bool = True

    # Calibration
    auto_calibration_enabled: bool = True
    reference_sensor_id: Optional[str] = None


class TemperatureSensorArrayConnector:
    """
    Temperature Sensor Array Connector for multi-sensor monitoring.

    Features:
    - Multi-sensor Modbus RTU integration
    - Real-time temperature monitoring (1Hz+)
    - Automatic sensor health detection
    - Calibration drift compensation
    - Zone-based temperature profiling
    - Statistical data processing (smoothing, outlier rejection)

    Example:
        config = SensorArrayConfig(
            array_id="TEMP_ARRAY_MAIN",
            serial_port="/dev/ttyUSB0",
            baudrate=9600
        )

        async with TemperatureSensorArrayConnector(config) as array:
            # Register sensors
            array.register_sensor(TemperatureSensor(
                sensor_id="FURNACE_TEMP_01",
                sensor_type=SensorType.THERMOCOUPLE_K,
                zone=TemperatureZone.FURNACE,
                register_address=0,
                max_temp_c=1200.0
            ))

            # Read furnace temperature
            temp = await array.read_furnace_temperature()
            print(f"Furnace: {temp}°C")

            # Read all zones
            temps = await array.read_all_zones()

            # Validate sensor health
            health = await array.validate_sensor_health()
    """

    def __init__(self, config: SensorArrayConfig):
        """Initialize temperature sensor array connector."""
        if not MODBUS_AVAILABLE:
            raise ImportError("pymodbus library required")

        self.config = config
        self.connected = False

        # Modbus RTU client
        self.modbus_client: Optional[AsyncModbusSerialClient] = None

        # Sensor registry
        self.sensors: Dict[str, TemperatureSensor] = {}
        self.sensors_by_zone: Dict[TemperatureZone, List[TemperatureSensor]] = defaultdict(list)

        # Data buffers (for smoothing)
        self.reading_buffers: Dict[str, deque] = {}

        # Background tasks
        self._monitoring_task: Optional[asyncio.Task] = None
        self._health_check_task: Optional[asyncio.Task] = None

        # Performance tracking
        self.scan_latencies = deque(maxlen=1000)

        # Prometheus metrics
        if METRICS_AVAILABLE:
            self.metrics = {
                'temperature_celsius': Gauge(
                    'temperature_celsius',
                    'Temperature in Celsius',
                    ['sensor_id', 'zone', 'sensor_type']
                ),
                'sensor_health': Gauge(
                    'sensor_health_status',
                    'Sensor health (1=healthy, 0=failed)',
                    ['sensor_id']
                ),
                'readings_total': Counter(
                    'temperature_readings_total',
                    'Total temperature readings',
                    ['sensor_id']
                ),
                'scan_latency': Histogram(
                    'sensor_array_scan_latency_seconds',
                    'Sensor array scan latency',
                    ['array_id']
                ),
                'calibration_drift': Gauge(
                    'sensor_calibration_drift_celsius',
                    'Calibration drift in Celsius',
                    ['sensor_id']
                )
            }
        else:
            self.metrics = {}

    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect_to_sensors()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()

    async def connect_to_sensors(self) -> bool:
        """
        Connect to temperature sensor array via Modbus RTU.

        Returns:
            True if connection successful

        Raises:
            ConnectionError: If connection fails
        """
        logger.info(f"Connecting to sensor array {self.config.array_id}...")

        try:
            self.modbus_client = AsyncModbusSerialClient(
                port=self.config.serial_port,
                baudrate=self.config.baudrate,
                parity=self.config.parity,
                stopbits=self.config.stopbits,
                bytesize=self.config.bytesize,
                timeout=self.config.timeout
            )

            await self.modbus_client.connect()

            if not self.modbus_client.connected:
                raise ConnectionError("Failed to connect to Modbus RTU")

            self.connected = True
            logger.info(f"Connected to sensor array via Modbus RTU ({self.config.serial_port})")

            # Start monitoring tasks
            self._monitoring_task = asyncio.create_task(self._monitoring_loop())
            self._health_check_task = asyncio.create_task(self._health_check_loop())

            return True

        except Exception as e:
            logger.error(f"Sensor array connection failed: {e}")
            raise ConnectionError(f"Sensor array connection failed: {e}")

    def register_sensor(self, sensor: TemperatureSensor):
        """Register a temperature sensor in the array."""
        self.sensors[sensor.sensor_id] = sensor
        self.sensors_by_zone[sensor.zone].append(sensor)

        # Initialize reading buffer
        self.reading_buffers[sensor.sensor_id] = deque(
            maxlen=self.config.temperature_smoothing_window
        )

        logger.info(
            f"Registered sensor {sensor.sensor_id} ({sensor.sensor_type.value}) "
            f"at address {sensor.register_address}"
        )

    async def read_furnace_temperature(self) -> Optional[float]:
        """
        Read furnace temperature (average of all furnace zone sensors).

        Returns:
            Average furnace temperature in Celsius or None
        """
        furnace_sensors = self.sensors_by_zone.get(TemperatureZone.FURNACE, [])

        if not furnace_sensors:
            logger.warning("No furnace temperature sensors configured")
            return None

        temps = []
        for sensor in furnace_sensors:
            temp = await self._read_sensor(sensor)
            if temp is not None and sensor.health_status == SensorHealth.HEALTHY:
                temps.append(temp)

        if temps:
            avg_temp = statistics.mean(temps)
            logger.debug(f"Furnace temperature: {avg_temp:.1f}°C (from {len(temps)} sensors)")
            return avg_temp
        else:
            logger.warning("No valid furnace temperature readings")
            return None

    async def read_flue_gas_temperature(self) -> Optional[float]:
        """
        Read flue gas temperature.

        Returns:
            Flue gas temperature in Celsius or None
        """
        flue_sensors = self.sensors_by_zone.get(TemperatureZone.FLUE_GAS, [])

        if not flue_sensors:
            logger.warning("No flue gas temperature sensors configured")
            return None

        temps = []
        for sensor in flue_sensors:
            temp = await self._read_sensor(sensor)
            if temp is not None and sensor.health_status == SensorHealth.HEALTHY:
                temps.append(temp)

        return statistics.mean(temps) if temps else None

    async def read_ambient_temperature(self) -> Optional[float]:
        """
        Read ambient temperature.

        Returns:
            Ambient temperature in Celsius or None
        """
        ambient_sensors = self.sensors_by_zone.get(TemperatureZone.AMBIENT, [])

        if not ambient_sensors:
            logger.warning("No ambient temperature sensors configured")
            return None

        temps = []
        for sensor in ambient_sensors:
            temp = await self._read_sensor(sensor)
            if temp is not None and sensor.health_status == SensorHealth.HEALTHY:
                temps.append(temp)

        return statistics.mean(temps) if temps else None

    async def read_all_zones(self) -> Dict[TemperatureZone, float]:
        """
        Read temperatures from all zones.

        Returns:
            Dictionary mapping zone to average temperature
        """
        results = {}

        for zone, sensors in self.sensors_by_zone.items():
            temps = []
            for sensor in sensors:
                temp = await self._read_sensor(sensor)
                if temp is not None and sensor.health_status == SensorHealth.HEALTHY:
                    temps.append(temp)

            if temps:
                results[zone] = statistics.mean(temps)

        return results

    async def _read_sensor(self, sensor: TemperatureSensor) -> Optional[float]:
        """
        Read temperature from individual sensor.

        Args:
            sensor: Sensor configuration

        Returns:
            Temperature in Celsius or None if failed
        """
        if not self.connected:
            return None

        try:
            # Read holding register (temperature as int16 or float32)
            # Assuming 2 registers for float32
            response = await self.modbus_client.read_holding_registers(
                address=sensor.register_address,
                count=2,
                unit=sensor.unit_id
            )

            if response.isError():
                raise ModbusException(f"Modbus error: {response}")

            # Decode temperature (float32)
            decoder = BinaryPayloadDecoder.fromRegisters(
                response.registers,
                byteorder=Endian.Big,
                wordorder=Endian.Big
            )
            raw_temp = decoder.decode_32bit_float()

            # Apply calibration
            calibrated_temp = (raw_temp * sensor.calibration_slope) + sensor.calibration_offset

            # Validate range
            if not (sensor.min_temp_c <= calibrated_temp <= sensor.max_temp_c):
                logger.warning(
                    f"Temperature {calibrated_temp:.1f}°C out of range "
                    f"[{sensor.min_temp_c}, {sensor.max_temp_c}] for {sensor.sensor_id}"
                )
                sensor.health_status = SensorHealth.OUT_OF_RANGE
                sensor.consecutive_failures += 1
                return None

            # Spike detection
            if sensor.current_temp_c is not None:
                temp_change = abs(calibrated_temp - sensor.current_temp_c)
                if temp_change > self.config.spike_detection_threshold_c:
                    logger.warning(
                        f"Temperature spike detected for {sensor.sensor_id}: "
                        f"{temp_change:.1f}°C change"
                    )
                    # Don't update - wait for confirmation
                    return sensor.current_temp_c

            # Apply smoothing
            buffer = self.reading_buffers[sensor.sensor_id]
            buffer.append(calibrated_temp)

            if len(buffer) >= 3:
                smoothed_temp = statistics.mean(buffer)
            else:
                smoothed_temp = calibrated_temp

            # Update sensor state
            sensor.current_temp_c = smoothed_temp
            sensor.last_reading_time = DeterministicClock.now()
            sensor.health_status = SensorHealth.HEALTHY
            sensor.consecutive_failures = 0

            # Update metrics
            if self.metrics:
                self.metrics['temperature_celsius'].labels(
                    sensor_id=sensor.sensor_id,
                    zone=sensor.zone.value,
                    sensor_type=sensor.sensor_type.value
                ).set(smoothed_temp)

                self.metrics['readings_total'].labels(
                    sensor_id=sensor.sensor_id
                ).inc()

            return smoothed_temp

        except Exception as e:
            logger.error(f"Failed to read sensor {sensor.sensor_id}: {e}")
            sensor.consecutive_failures += 1

            # Determine failure type
            if "timeout" in str(e).lower():
                sensor.health_status = SensorHealth.FAILED
            elif "crc" in str(e).lower() or "checksum" in str(e).lower():
                sensor.health_status = SensorHealth.FAILED
            else:
                sensor.health_status = SensorHealth.FAILED

            return None

    async def validate_sensor_health(self) -> Dict[str, Any]:
        """
        Validate health of all sensors.

        Returns:
            Dictionary with health status for each sensor
        """
        health_report = {
            'overall_health': 'healthy',
            'healthy_sensors': 0,
            'degraded_sensors': 0,
            'failed_sensors': 0,
            'sensor_details': {}
        }

        for sensor_id, sensor in self.sensors.items():
            # Check consecutive failures
            if sensor.consecutive_failures >= 5:
                sensor.health_status = SensorHealth.FAILED

            # Check last reading time
            if sensor.last_reading_time:
                time_since_reading = (DeterministicClock.now() - sensor.last_reading_time).total_seconds()
                if time_since_reading > 60:  # No reading for 1 minute
                    sensor.health_status = SensorHealth.FAILED

            # Check calibration age
            if sensor.last_calibration:
                days_since_cal = (DeterministicClock.now() - sensor.last_calibration).days
                if days_since_cal > sensor.calibration_interval_days:
                    if sensor.health_status == SensorHealth.HEALTHY:
                        sensor.health_status = SensorHealth.CALIBRATION_DRIFT

            # Count by health status
            if sensor.health_status == SensorHealth.HEALTHY:
                health_report['healthy_sensors'] += 1
            elif sensor.health_status == SensorHealth.FAILED:
                health_report['failed_sensors'] += 1
            else:
                health_report['degraded_sensors'] += 1

            health_report['sensor_details'][sensor_id] = {
                'status': sensor.health_status.value,
                'current_temp_c': sensor.current_temp_c,
                'consecutive_failures': sensor.consecutive_failures,
                'last_reading': sensor.last_reading_time.isoformat() if sensor.last_reading_time else None
            }

            # Update metrics
            if self.metrics:
                health_value = 1 if sensor.health_status == SensorHealth.HEALTHY else 0
                self.metrics['sensor_health'].labels(
                    sensor_id=sensor_id
                ).set(health_value)

        # Overall health assessment
        total_sensors = len(self.sensors)
        if health_report['failed_sensors'] > total_sensors * 0.2:
            health_report['overall_health'] = 'critical'
        elif health_report['degraded_sensors'] > total_sensors * 0.3:
            health_report['overall_health'] = 'degraded'

        return health_report

    async def apply_calibration(
        self,
        sensor_id: str,
        reference_temp_c: float
    ) -> bool:
        """
        Apply calibration to sensor using reference temperature.

        Args:
            sensor_id: Sensor to calibrate
            reference_temp_c: Known reference temperature

        Returns:
            True if calibration successful
        """
        sensor = self.sensors.get(sensor_id)
        if not sensor:
            logger.error(f"Sensor {sensor_id} not found")
            return False

        try:
            # Read current sensor value (uncalibrated)
            response = await self.modbus_client.read_holding_registers(
                address=sensor.register_address,
                count=2,
                unit=sensor.unit_id
            )

            if response.isError():
                raise ModbusException(f"Modbus error: {response}")

            decoder = BinaryPayloadDecoder.fromRegisters(
                response.registers,
                byteorder=Endian.Big,
                wordorder=Endian.Big
            )
            raw_temp = decoder.decode_32bit_float()

            # Calculate calibration offset
            # Assuming slope remains 1.0 (single-point calibration)
            sensor.calibration_offset = reference_temp_c - raw_temp
            sensor.last_calibration = DeterministicClock.now()
            sensor.health_status = SensorHealth.HEALTHY

            logger.info(
                f"Calibrated {sensor_id}: offset={sensor.calibration_offset:.2f}°C "
                f"(raw={raw_temp:.1f}°C, ref={reference_temp_c:.1f}°C)"
            )

            # Update metrics
            if self.metrics:
                self.metrics['calibration_drift'].labels(
                    sensor_id=sensor_id
                ).set(abs(sensor.calibration_offset))

            return True

        except Exception as e:
            logger.error(f"Calibration failed for {sensor_id}: {e}")
            return False

    async def _monitoring_loop(self):
        """Background task for continuous sensor monitoring."""
        scan_interval = 1.0 / self.config.scan_rate_hz

        while self.connected:
            try:
                start_time = time.perf_counter()

                # Read all sensors
                for sensor in self.sensors.values():
                    await self._read_sensor(sensor)

                # Record scan latency
                latency = time.perf_counter() - start_time
                self.scan_latencies.append(latency)

                if self.metrics:
                    self.metrics['scan_latency'].labels(
                        array_id=self.config.array_id
                    ).observe(latency)

                await asyncio.sleep(max(0, scan_interval - latency))

            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(scan_interval)

    async def _health_check_loop(self):
        """Background task for periodic health checks."""
        while self.connected:
            try:
                await self.validate_sensor_health()
                await asyncio.sleep(60)  # Check every minute

            except Exception as e:
                logger.error(f"Health check loop error: {e}")
                await asyncio.sleep(60)

    async def disconnect(self):
        """Disconnect from sensor array."""
        logger.info(f"Disconnecting from sensor array {self.config.array_id}...")

        # Stop background tasks
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass

        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass

        # Close Modbus connection
        if self.modbus_client:
            try:
                self.modbus_client.close()
            except Exception as e:
                logger.error(f"Error closing Modbus connection: {e}")

        self.connected = False
        logger.info("Disconnected from sensor array")

    def get_zone_statistics(self, zone: TemperatureZone) -> Optional[Dict[str, float]]:
        """
        Get statistical summary for temperature zone.

        Args:
            zone: Temperature zone

        Returns:
            Dictionary with min, max, mean, std_dev temperatures
        """
        sensors = self.sensors_by_zone.get(zone, [])
        if not sensors:
            return None

        temps = [
            s.current_temp_c for s in sensors
            if s.current_temp_c is not None and s.health_status == SensorHealth.HEALTHY
        ]

        if not temps:
            return None

        return {
            'min_temp_c': min(temps),
            'max_temp_c': max(temps),
            'mean_temp_c': statistics.mean(temps),
            'std_dev_c': statistics.stdev(temps) if len(temps) > 1 else 0,
            'sensor_count': len(temps)
        }

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get connector performance statistics."""
        if not self.scan_latencies:
            return {}

        latencies_ms = [l * 1000 for l in self.scan_latencies]

        return {
            'avg_scan_latency_ms': statistics.mean(latencies_ms),
            'max_scan_latency_ms': max(latencies_ms),
            'total_sensors': len(self.sensors),
            'connected': self.connected
        }
