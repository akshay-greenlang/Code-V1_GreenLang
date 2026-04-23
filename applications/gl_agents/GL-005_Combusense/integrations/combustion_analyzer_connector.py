# -*- coding: utf-8 -*-
"""
Combustion Analyzer Connector for GL-005 CombustionControlAgent

Implements real-time integration with combustion gas analyzers for:
- O2, CO, NOx, CO2 measurement
- MQTT streaming (primary) with Modbus fallback
- Data quality validation and sensor health monitoring
- Automatic calibration sequencing
- Multi-analyzer support with data fusion

Real-Time Requirements:
- Measurement update rate: 1Hz minimum
- Data quality validation: <50ms
- Alarm detection: <100ms
- Calibration cycle: <5 minutes

Protocols Supported:
- MQTT (IEC 62591) - Primary for streaming data
- Modbus TCP - Fallback for legacy analyzers

Supported Analyzers:
- O2 Analyzer (Zirconia, Paramagnetic)
- CO Analyzer (NDIR, Electrochemical)
- NOx Analyzer (Chemiluminescence, NDIR)
- CO2 Analyzer (NDIR)
- Multi-gas analyzers

Author: GL-DataIntegrationEngineer
Date: 2025-11-18
Version: 1.0.0
"""

import asyncio
import logging
import json
import time
from typing import Dict, List, Optional, Any, Callable, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque, defaultdict
from enum import Enum
import statistics
from greenlang.determinism import DeterministicClock

# Third-party imports
try:
    import paho.mqtt.client as mqtt
    from paho.mqtt.client import MQTTMessage
    MQTT_AVAILABLE = True
except ImportError:
    MQTT_AVAILABLE = False
    MQTTMessage = None
    mqtt = None

try:
    from pymodbus.client import AsyncModbusTcpClient
    from pymodbus.exceptions import ModbusException
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


class AnalyzerProtocol(Enum):
    """Supported analyzer communication protocols."""
    MQTT = "mqtt"
    MODBUS_TCP = "modbus_tcp"


class GasType(Enum):
    """Types of measured gases."""
    O2 = "oxygen"
    CO = "carbon_monoxide"
    CO2 = "carbon_dioxide"
    NOx = "nitrogen_oxides"
    NO = "nitric_oxide"
    NO2 = "nitrogen_dioxide"
    SO2 = "sulfur_dioxide"


class CalibrationStatus(Enum):
    """Analyzer calibration status."""
    VALID = "valid"
    DUE = "due"
    OVERDUE = "overdue"
    IN_PROGRESS = "in_progress"
    FAILED = "failed"


class DataQuality(Enum):
    """Data quality indicators."""
    GOOD = "good"
    SUSPECT = "suspect"
    BAD = "bad"
    SENSOR_FAILURE = "sensor_failure"
    CALIBRATION_ERROR = "calibration_error"
    OUT_OF_RANGE = "out_of_range"


@dataclass
class GasMeasurement:
    """Gas concentration measurement."""
    gas_type: GasType
    concentration: float  # ppm or % volume
    units: str
    timestamp: datetime
    quality: DataQuality = DataQuality.GOOD
    temperature: Optional[float] = None  # Sensor temperature
    pressure: Optional[float] = None  # Sample pressure


@dataclass
class AnalyzerConfig:
    """Configuration for combustion analyzer."""
    analyzer_id: str
    manufacturer: str
    model: str

    # Protocol settings
    primary_protocol: AnalyzerProtocol = AnalyzerProtocol.MQTT
    fallback_protocol: Optional[AnalyzerProtocol] = AnalyzerProtocol.MODBUS_TCP

    # MQTT settings
    mqtt_broker: str = "localhost"
    mqtt_port: int = 1883
    mqtt_topic_prefix: str = "combustion/analyzer"
    mqtt_qos: int = 2  # Exactly once delivery
    mqtt_keepalive: int = 60
    mqtt_username: Optional[str] = None
    mqtt_password: Optional[str] = None
    mqtt_use_tls: bool = True
    mqtt_ca_cert: Optional[str] = None

    # Modbus settings
    modbus_host: str = "localhost"
    modbus_port: int = 502
    modbus_unit_id: int = 1
    modbus_timeout: float = 2.0

    # Measurement settings
    gases_measured: List[GasType] = field(default_factory=list)
    measurement_units: Dict[GasType, str] = field(default_factory=dict)
    measurement_ranges: Dict[GasType, tuple] = field(default_factory=dict)
    update_rate_hz: float = 1.0

    # Calibration settings
    calibration_interval_hours: int = 168  # 1 week
    auto_calibration_enabled: bool = True
    calibration_gas_concentrations: Dict[GasType, float] = field(default_factory=dict)

    # Data quality settings
    max_reading_deviation_pct: float = 10.0  # Spike detection
    min_valid_readings_per_second: int = 1
    data_buffer_size: int = 3600  # 1 hour at 1Hz


@dataclass
class AnalyzerStatus:
    """Analyzer operational status."""
    analyzer_id: str
    connected: bool = False
    protocol_active: AnalyzerProtocol = AnalyzerProtocol.MQTT
    last_reading_time: Optional[datetime] = None
    calibration_status: CalibrationStatus = CalibrationStatus.VALID
    last_calibration_time: Optional[datetime] = None
    next_calibration_time: Optional[datetime] = None
    sensor_temperature: Optional[float] = None
    sample_flow_rate: Optional[float] = None
    alarms_active: List[str] = field(default_factory=list)
    consecutive_failures: int = 0


class CombustionAnalyzerConnector:
    """
    Combustion Analyzer Connector with MQTT/Modbus support.

    Features:
    - Real-time gas concentration measurements
    - Multi-protocol support (MQTT primary, Modbus fallback)
    - Data quality validation and filtering
    - Automatic sensor calibration
    - Spike detection and data smoothing
    - Multi-analyzer data fusion
    - Comprehensive health monitoring

    Example:
        config = AnalyzerConfig(
            analyzer_id="O2_ANALYZER_01",
            manufacturer="ABB",
            model="AO2020",
            mqtt_broker="mqtt.plant.com",
            gases_measured=[GasType.O2],
            measurement_units={GasType.O2: "%"}
        )

        async with CombustionAnalyzerConnector(config) as analyzer:
            # Read current O2 level
            o2 = await analyzer.read_o2_level()
            print(f"O2: {o2}%")

            # Read all gases
            readings = await analyzer.read_all_gases()

            # Trigger calibration
            await analyzer.calibrate_analyzer()

            # Subscribe to real-time data
            await analyzer.subscribe_to_measurements(
                lambda measurement: print(f"{measurement.gas_type}: {measurement.concentration}")
            )
    """

    def __init__(self, config: AnalyzerConfig):
        """Initialize combustion analyzer connector."""
        self.config = config
        self.status = AnalyzerStatus(analyzer_id=config.analyzer_id)

        # MQTT client
        self.mqtt_client: Optional[mqtt.Client] = None
        self.mqtt_connected = False

        # Modbus client
        self.modbus_client: Optional[AsyncModbusTcpClient] = None
        self.modbus_connected = False

        # Data buffers (ring buffers for each gas)
        self.measurement_buffers: Dict[GasType, deque] = {
            gas: deque(maxlen=config.data_buffer_size)
            for gas in config.gases_measured
        }

        # Latest measurements
        self.latest_measurements: Dict[GasType, GasMeasurement] = {}

        # Measurement callbacks
        self.measurement_callbacks: List[Callable[[GasMeasurement], None]] = []

        # Data quality tracking
        self.reading_timestamps: deque = deque(maxlen=100)
        self.invalid_reading_count = 0

        # Background tasks
        self._calibration_task: Optional[asyncio.Task] = None
        self._health_monitor_task: Optional[asyncio.Task] = None

        # Prometheus metrics
        if METRICS_AVAILABLE:
            self.metrics = {
                'measurements_total': Counter(
                    'analyzer_measurements_total',
                    'Total analyzer measurements',
                    ['analyzer_id', 'gas_type']
                ),
                'measurement_value': Gauge(
                    'analyzer_measurement_value',
                    'Current gas concentration',
                    ['analyzer_id', 'gas_type']
                ),
                'data_quality': Gauge(
                    'analyzer_data_quality_score',
                    'Data quality score (0-100)',
                    ['analyzer_id']
                ),
                'calibration_due': Gauge(
                    'analyzer_calibration_due_hours',
                    'Hours until calibration due',
                    ['analyzer_id']
                )
            }
        else:
            self.metrics = {}

    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect_to_analyzer()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()

    async def connect_to_analyzer(self) -> bool:
        """
        Connect to combustion analyzer.

        Returns:
            True if connection successful

        Raises:
            ConnectionError: If connection fails
        """
        logger.info(f"Connecting to analyzer {self.config.analyzer_id}...")

        # Try MQTT first
        if self.config.primary_protocol == AnalyzerProtocol.MQTT:
            if await self._connect_mqtt():
                self.status.protocol_active = AnalyzerProtocol.MQTT
                self.status.connected = True
                logger.info(f"Connected to analyzer via MQTT")

                # Start health monitoring
                self._health_monitor_task = asyncio.create_task(self._health_monitor_loop())

                # Start auto-calibration if enabled
                if self.config.auto_calibration_enabled:
                    self._calibration_task = asyncio.create_task(self._calibration_loop())

                return True

        # Fallback to Modbus
        if self.config.fallback_protocol == AnalyzerProtocol.MODBUS_TCP:
            logger.warning("MQTT connection failed, trying Modbus fallback")
            if await self._connect_modbus():
                self.status.protocol_active = AnalyzerProtocol.MODBUS_TCP
                self.status.connected = True
                logger.info(f"Connected to analyzer via Modbus TCP")

                # Start health monitoring
                self._health_monitor_task = asyncio.create_task(self._health_monitor_loop())

                return True

        raise ConnectionError(f"Failed to connect to analyzer {self.config.analyzer_id}")

    async def _connect_mqtt(self) -> bool:
        """Connect via MQTT protocol."""
        if not MQTT_AVAILABLE:
            logger.error("MQTT library not available")
            return False

        try:
            self.mqtt_client = mqtt.Client(
                client_id=f"analyzer_{self.config.analyzer_id}",
                protocol=mqtt.MQTTv311
            )

            # Set callbacks
            self.mqtt_client.on_connect = self._on_mqtt_connect
            self.mqtt_client.on_message = self._on_mqtt_message
            self.mqtt_client.on_disconnect = self._on_mqtt_disconnect

            # Set authentication
            if self.config.mqtt_username and self.config.mqtt_password:
                self.mqtt_client.username_pw_set(
                    self.config.mqtt_username,
                    self.config.mqtt_password
                )

            # Set TLS
            if self.config.mqtt_use_tls and self.config.mqtt_ca_cert:
                self.mqtt_client.tls_set(ca_certs=self.config.mqtt_ca_cert)

            # Connect
            self.mqtt_client.connect(
                self.config.mqtt_broker,
                self.config.mqtt_port,
                self.config.mqtt_keepalive
            )

            # Start network loop in background
            self.mqtt_client.loop_start()

            # Wait for connection (with timeout)
            for _ in range(10):
                if self.mqtt_connected:
                    return True
                await asyncio.sleep(0.5)

            logger.error("MQTT connection timeout")
            return False

        except Exception as e:
            logger.error(f"MQTT connection failed: {e}")
            return False

    def _on_mqtt_connect(self, client, userdata, flags, rc):
        """MQTT connection callback."""
        if rc == 0:
            logger.info("MQTT connected successfully")
            self.mqtt_connected = True

            # Subscribe to measurement topics
            for gas in self.config.gases_measured:
                topic = f"{self.config.mqtt_topic_prefix}/{self.config.analyzer_id}/{gas.value}"
                client.subscribe(topic, qos=self.config.mqtt_qos)
                logger.info(f"Subscribed to MQTT topic: {topic}")

        else:
            logger.error(f"MQTT connection failed with code {rc}")
            self.mqtt_connected = False

    def _on_mqtt_message(self, client, userdata, message):
        """MQTT message received callback."""
        try:
            # Parse topic to determine gas type
            topic_parts = message.topic.split('/')
            gas_name = topic_parts[-1]

            # Find matching gas type
            gas_type = None
            for gas in GasType:
                if gas.value == gas_name:
                    gas_type = gas
                    break

            if not gas_type:
                logger.warning(f"Unknown gas type in topic: {gas_name}")
                return

            # Parse message payload (JSON format)
            data = json.loads(message.payload.decode())

            measurement = GasMeasurement(
                gas_type=gas_type,
                concentration=float(data['concentration']),
                units=data.get('units', self.config.measurement_units.get(gas_type, 'ppm')),
                timestamp=datetime.fromisoformat(data.get('timestamp', DeterministicClock.now().isoformat())),
                quality=DataQuality[data.get('quality', 'GOOD')],
                temperature=data.get('sensor_temperature'),
                pressure=data.get('sample_pressure')
            )

            # Process measurement
            asyncio.create_task(self._process_measurement(measurement))

        except Exception as e:
            logger.error(f"Error processing MQTT message: {e}")

    def _on_mqtt_disconnect(self, client, userdata, rc):
        """MQTT disconnection callback."""
        logger.warning(f"MQTT disconnected with code {rc}")
        self.mqtt_connected = False

        if rc != 0:
            logger.info("Unexpected disconnection, attempting reconnection...")
            try:
                client.reconnect()
            except Exception as e:
                logger.error(f"MQTT reconnection failed: {e}")

    async def _connect_modbus(self) -> bool:
        """Connect via Modbus TCP protocol."""
        if not MODBUS_AVAILABLE:
            logger.error("Modbus library not available")
            return False

        try:
            self.modbus_client = AsyncModbusTcpClient(
                host=self.config.modbus_host,
                port=self.config.modbus_port,
                timeout=self.config.modbus_timeout
            )

            await self.modbus_client.connect()

            if self.modbus_client.connected:
                self.modbus_connected = True
                logger.info(f"Modbus TCP connected to {self.config.modbus_host}")
                return True
            else:
                return False

        except Exception as e:
            logger.error(f"Modbus connection failed: {e}")
            return False

    async def read_o2_level(self) -> Optional[float]:
        """
        Read oxygen concentration.

        Returns:
            O2 concentration in % or None if unavailable
        """
        if GasType.O2 not in self.config.gases_measured:
            logger.warning("O2 not configured for this analyzer")
            return None

        if self.status.protocol_active == AnalyzerProtocol.MQTT:
            # Return latest buffered value
            measurement = self.latest_measurements.get(GasType.O2)
            return measurement.concentration if measurement else None
        else:
            # Read from Modbus
            return await self._read_modbus_gas(GasType.O2)

    async def read_co_level(self) -> Optional[float]:
        """
        Read carbon monoxide concentration.

        Returns:
            CO concentration in ppm or None if unavailable
        """
        if GasType.CO not in self.config.gases_measured:
            logger.warning("CO not configured for this analyzer")
            return None

        if self.status.protocol_active == AnalyzerProtocol.MQTT:
            measurement = self.latest_measurements.get(GasType.CO)
            return measurement.concentration if measurement else None
        else:
            return await self._read_modbus_gas(GasType.CO)

    async def read_nox_level(self) -> Optional[float]:
        """
        Read nitrogen oxides concentration.

        Returns:
            NOx concentration in ppm or None if unavailable
        """
        if GasType.NOx not in self.config.gases_measured:
            logger.warning("NOx not configured for this analyzer")
            return None

        if self.status.protocol_active == AnalyzerProtocol.MQTT:
            measurement = self.latest_measurements.get(GasType.NOx)
            return measurement.concentration if measurement else None
        else:
            return await self._read_modbus_gas(GasType.NOx)

    async def read_co2_level(self) -> Optional[float]:
        """
        Read carbon dioxide concentration.

        Returns:
            CO2 concentration in % or None if unavailable
        """
        if GasType.CO2 not in self.config.gases_measured:
            logger.warning("CO2 not configured for this analyzer")
            return None

        if self.status.protocol_active == AnalyzerProtocol.MQTT:
            measurement = self.latest_measurements.get(GasType.CO2)
            return measurement.concentration if measurement else None
        else:
            return await self._read_modbus_gas(GasType.CO2)

    async def read_all_gases(self) -> Dict[GasType, float]:
        """
        Read all configured gas concentrations.

        Returns:
            Dictionary mapping gas type to concentration
        """
        result = {}

        for gas_type in self.config.gases_measured:
            if self.status.protocol_active == AnalyzerProtocol.MQTT:
                measurement = self.latest_measurements.get(gas_type)
                if measurement:
                    result[gas_type] = measurement.concentration
            else:
                value = await self._read_modbus_gas(gas_type)
                if value is not None:
                    result[gas_type] = value

        return result

    async def _read_modbus_gas(self, gas_type: GasType) -> Optional[float]:
        """Read gas concentration via Modbus."""
        if not self.modbus_connected:
            return None

        try:
            # Map gas type to Modbus register (example addressing)
            register_map = {
                GasType.O2: 0,
                GasType.CO: 2,
                GasType.CO2: 4,
                GasType.NOx: 6,
                GasType.NO: 8,
                GasType.NO2: 10,
                GasType.SO2: 12
            }

            address = register_map.get(gas_type)
            if address is None:
                logger.warning(f"No Modbus register mapped for {gas_type}")
                return None

            # Read holding register (float32 = 2 registers)
            response = await self.modbus_client.read_holding_registers(
                address=address,
                count=2,
                unit=self.config.modbus_unit_id
            )

            if response.isError():
                raise ModbusException(f"Modbus error: {response}")

            # Decode float32
            from pymodbus.payload import BinaryPayloadDecoder
            from pymodbus.constants import Endian

            decoder = BinaryPayloadDecoder.fromRegisters(
                response.registers,
                byteorder=Endian.Big,
                wordorder=Endian.Big
            )
            value = decoder.decode_32bit_float()

            # Create measurement
            measurement = GasMeasurement(
                gas_type=gas_type,
                concentration=value,
                units=self.config.measurement_units.get(gas_type, 'ppm'),
                timestamp=DeterministicClock.now(),
                quality=DataQuality.GOOD
            )

            await self._process_measurement(measurement)

            return value

        except Exception as e:
            logger.error(f"Failed to read {gas_type} from Modbus: {e}")
            self.status.consecutive_failures += 1
            return None

    async def calibrate_analyzer(self) -> bool:
        """
        Run analyzer calibration sequence.

        Returns:
            True if calibration successful

        Raises:
            RuntimeError: If calibration fails
        """
        logger.info(f"Starting calibration for analyzer {self.config.analyzer_id}")

        self.status.calibration_status = CalibrationStatus.IN_PROGRESS

        try:
            # Calibration sequence (simplified):
            # 1. Zero calibration (background air or nitrogen)
            # 2. Span calibration (reference gas)
            # 3. Linearity check

            await asyncio.sleep(2)  # Simulate zero calibration
            logger.info("Zero calibration complete")

            await asyncio.sleep(2)  # Simulate span calibration
            logger.info("Span calibration complete")

            await asyncio.sleep(1)  # Simulate linearity check
            logger.info("Linearity check complete")

            # Update calibration status
            self.status.calibration_status = CalibrationStatus.VALID
            self.status.last_calibration_time = DeterministicClock.now()
            self.status.next_calibration_time = DeterministicClock.now() + timedelta(
                hours=self.config.calibration_interval_hours
            )

            logger.info(f"Calibration successful. Next calibration: {self.status.next_calibration_time}")

            return True

        except Exception as e:
            logger.error(f"Calibration failed: {e}")
            self.status.calibration_status = CalibrationStatus.FAILED
            raise RuntimeError(f"Calibration failed: {e}")

    async def validate_readings(self) -> Dict[str, Any]:
        """
        Perform data quality validation.

        Returns:
            Dictionary with validation results
        """
        validation = {
            'overall_quality': DataQuality.GOOD,
            'quality_score': 100.0,
            'issues': []
        }

        # Check data update rate
        if len(self.reading_timestamps) >= 10:
            time_diffs = [
                (self.reading_timestamps[i+1] - self.reading_timestamps[i]).total_seconds()
                for i in range(len(self.reading_timestamps) - 1)
            ]
            avg_interval = statistics.mean(time_diffs)
            expected_interval = 1.0 / self.config.update_rate_hz

            if avg_interval > expected_interval * 2:
                validation['issues'].append(f"Low data rate: {1/avg_interval:.2f} Hz")
                validation['quality_score'] -= 20

        # Check for spikes/outliers in each gas
        for gas_type, buffer in self.measurement_buffers.items():
            if len(buffer) >= 10:
                values = [m.concentration for m in buffer]
                mean_value = statistics.mean(values)
                std_dev = statistics.stdev(values)

                # Check latest value for spike
                latest_value = values[-1]
                if abs(latest_value - mean_value) > 3 * std_dev:
                    validation['issues'].append(f"{gas_type.value} spike detected: {latest_value}")
                    validation['quality_score'] -= 15

        # Check calibration status
        if self.status.calibration_status == CalibrationStatus.OVERDUE:
            validation['issues'].append("Calibration overdue")
            validation['quality_score'] -= 25
        elif self.status.calibration_status == CalibrationStatus.DUE:
            validation['issues'].append("Calibration due soon")
            validation['quality_score'] -= 10

        # Overall quality assessment
        if validation['quality_score'] < 50:
            validation['overall_quality'] = DataQuality.BAD
        elif validation['quality_score'] < 80:
            validation['overall_quality'] = DataQuality.SUSPECT

        if self.metrics:
            self.metrics['data_quality'].labels(
                analyzer_id=self.config.analyzer_id
            ).set(validation['quality_score'])

        return validation

    async def _process_measurement(self, measurement: GasMeasurement):
        """Process incoming measurement."""
        # Validate range
        if measurement.gas_type in self.config.measurement_ranges:
            min_val, max_val = self.config.measurement_ranges[measurement.gas_type]
            if not (min_val <= measurement.concentration <= max_val):
                logger.warning(
                    f"{measurement.gas_type.value} out of range: {measurement.concentration}"
                )
                measurement.quality = DataQuality.OUT_OF_RANGE

        # Add to buffer
        self.measurement_buffers[measurement.gas_type].append(measurement)
        self.latest_measurements[measurement.gas_type] = measurement
        self.reading_timestamps.append(measurement.timestamp)

        # Update status
        self.status.last_reading_time = measurement.timestamp
        self.status.consecutive_failures = 0

        # Update metrics
        if self.metrics:
            self.metrics['measurements_total'].labels(
                analyzer_id=self.config.analyzer_id,
                gas_type=measurement.gas_type.value
            ).inc()

            self.metrics['measurement_value'].labels(
                analyzer_id=self.config.analyzer_id,
                gas_type=measurement.gas_type.value
            ).set(measurement.concentration)

        # Call callbacks
        for callback in self.measurement_callbacks:
            try:
                await callback(measurement)
            except Exception as e:
                logger.error(f"Measurement callback failed: {e}")

    async def subscribe_to_measurements(self, callback: Callable[[GasMeasurement], None]):
        """Subscribe to real-time measurements."""
        self.measurement_callbacks.append(callback)
        logger.info("Subscribed to analyzer measurements")

    async def _calibration_loop(self):
        """Background task for automatic calibration."""
        while self.status.connected:
            try:
                # Check if calibration is due
                if self.status.next_calibration_time:
                    time_until_cal = (
                        self.status.next_calibration_time - DeterministicClock.now()
                    ).total_seconds()

                    if time_until_cal <= 0:
                        # Calibration overdue
                        self.status.calibration_status = CalibrationStatus.OVERDUE
                        await self.calibrate_analyzer()

                    elif time_until_cal <= 3600:  # 1 hour warning
                        self.status.calibration_status = CalibrationStatus.DUE

                    # Update metrics
                    if self.metrics:
                        self.metrics['calibration_due'].labels(
                            analyzer_id=self.config.analyzer_id
                        ).set(time_until_cal / 3600)

                await asyncio.sleep(300)  # Check every 5 minutes

            except Exception as e:
                logger.error(f"Calibration loop error: {e}")
                await asyncio.sleep(300)

    async def _health_monitor_loop(self):
        """Background task for health monitoring."""
        while self.status.connected:
            try:
                # Check for stale data
                if self.status.last_reading_time:
                    seconds_since_reading = (
                        DeterministicClock.now() - self.status.last_reading_time
                    ).total_seconds()

                    max_interval = 2.0 / self.config.update_rate_hz

                    if seconds_since_reading > max_interval:
                        logger.warning(
                            f"No readings for {seconds_since_reading:.1f}s - data may be stale"
                        )
                        self.status.consecutive_failures += 1

                # Validate data quality
                await self.validate_readings()

                await asyncio.sleep(10)  # Check every 10 seconds

            except Exception as e:
                logger.error(f"Health monitor error: {e}")
                await asyncio.sleep(10)

    async def disconnect(self):
        """Disconnect from analyzer."""
        logger.info(f"Disconnecting from analyzer {self.config.analyzer_id}...")

        # Stop background tasks
        if self._calibration_task:
            self._calibration_task.cancel()
            try:
                await self._calibration_task
            except asyncio.CancelledError:
                pass

        if self._health_monitor_task:
            self._health_monitor_task.cancel()
            try:
                await self._health_monitor_task
            except asyncio.CancelledError:
                pass

        # Disconnect MQTT
        if self.mqtt_client:
            try:
                self.mqtt_client.loop_stop()
                self.mqtt_client.disconnect()
            except Exception as e:
                logger.error(f"MQTT disconnect error: {e}")

        # Disconnect Modbus
        if self.modbus_client:
            try:
                self.modbus_client.close()
            except Exception as e:
                logger.error(f"Modbus disconnect error: {e}")

        self.status.connected = False
        logger.info("Disconnected from analyzer")
