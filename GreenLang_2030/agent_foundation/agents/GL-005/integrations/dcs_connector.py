"""
DCS (Distributed Control System) Connector for GL-005 CombustionControlAgent

Implements secure, real-time connections to industrial DCS systems with:
- OPC UA (primary protocol) with fallback to Modbus TCP
- Sub-100ms response time for control loop requirements
- Connection pooling and health monitoring
- Circuit breaker pattern for fault tolerance
- Comprehensive error handling and retry logic

Real-Time Requirements:
- Control loop time: <100ms
- Data acquisition rate: 10Hz minimum
- Alarm response: <50ms
- Historical data retrieval: 1-hour windows

Protocols Supported:
- OPC UA (IEC 62541) - Primary
- Modbus TCP (IEC 61158) - Fallback

Author: GL-DataIntegrationEngineer
Date: 2025-11-18
Version: 1.0.0
"""

import asyncio
import logging
import ssl
import time
from typing import Dict, List, Optional, Any, Set, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque, defaultdict
from enum import Enum
import json
import hashlib
from contextlib import asynccontextmanager

# Third-party imports
try:
    from asyncua import Client as OPCUAClient, ua
    from asyncua.common.subscription import DataChangeNotif
    OPCUA_AVAILABLE = True
except ImportError:
    OPCUA_AVAILABLE = False

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


class DCSProtocol(Enum):
    """Supported DCS communication protocols."""
    OPC_UA = "opc_ua"
    MODBUS_TCP = "modbus_tcp"


class DataQuality(Enum):
    """OPC UA data quality indicators."""
    GOOD = "good"
    BAD = "bad"
    UNCERTAIN = "uncertain"
    BAD_COMM_FAILURE = "bad_comm_failure"
    BAD_SENSOR_FAILURE = "bad_sensor_failure"
    BAD_OUT_OF_SERVICE = "bad_out_of_service"
    UNCERTAIN_SENSOR_CAL = "uncertain_sensor_cal"


class CircuitBreakerState(Enum):
    """Circuit breaker states for fault tolerance."""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"      # Blocking calls due to failures
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class ProcessVariable:
    """Process variable (tag) configuration for DCS."""
    tag_name: str
    node_id: str  # OPC UA node ID or Modbus register
    description: str
    data_type: str  # float, int, bool, string
    engineering_units: str
    scan_rate_ms: int = 100  # Milliseconds
    deadband: float = 0.1  # Change threshold for notification
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    alarm_high_high: Optional[float] = None
    alarm_high: Optional[float] = None
    alarm_low: Optional[float] = None
    alarm_low_low: Optional[float] = None
    scaling_factor: float = 1.0
    offset: float = 0.0
    writable: bool = False

    # Runtime state
    current_value: Any = None
    quality: DataQuality = DataQuality.GOOD
    last_update: Optional[datetime] = None
    consecutive_bad_reads: int = 0


@dataclass
class DCSAlarm:
    """DCS alarm event."""
    alarm_id: str
    tag_name: str
    alarm_type: str  # HH, H, L, LL, deviation, roc
    priority: int  # 1=Critical, 2=High, 3=Medium, 4=Low
    setpoint: float
    actual_value: float
    message: str
    timestamp: datetime
    acknowledged: bool = False
    ack_timestamp: Optional[datetime] = None
    ack_user: Optional[str] = None


@dataclass
class DCSConfig:
    """Configuration for DCS connection."""
    # Connection settings
    primary_protocol: DCSProtocol = DCSProtocol.OPC_UA
    fallback_protocol: Optional[DCSProtocol] = DCSProtocol.MODBUS_TCP

    # OPC UA settings
    opcua_endpoint: str = "opc.tcp://localhost:4840"
    opcua_security_policy: str = "Basic256Sha256"
    opcua_security_mode: str = "SignAndEncrypt"
    opcua_username: Optional[str] = None
    opcua_password: Optional[str] = None  # From vault
    opcua_cert_path: Optional[str] = None
    opcua_private_key_path: Optional[str] = None

    # Modbus TCP settings
    modbus_host: str = "localhost"
    modbus_port: int = 502
    modbus_unit_id: int = 1
    modbus_timeout: float = 1.0  # seconds

    # Connection management
    connection_timeout: int = 30
    retry_max_attempts: int = 3
    retry_base_delay: float = 1.0  # exponential backoff base
    retry_max_delay: float = 60.0
    keepalive_interval: int = 10  # seconds

    # Circuit breaker settings
    failure_threshold: int = 5
    recovery_timeout: int = 60  # seconds
    half_open_max_calls: int = 3

    # Performance settings
    max_concurrent_reads: int = 50
    max_concurrent_writes: int = 10
    subscription_queue_size: int = 1000
    historical_data_max_points: int = 10000


class CircuitBreaker:
    """Circuit breaker for fault tolerance."""

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
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

    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection."""
        async with self._lock:
            if self.state == CircuitBreakerState.OPEN:
                # Check if recovery timeout has elapsed
                if self.last_failure_time and \
                   (datetime.now() - self.last_failure_time).total_seconds() >= self.recovery_timeout:
                    logger.info("Circuit breaker transitioning to HALF_OPEN")
                    self.state = CircuitBreakerState.HALF_OPEN
                    self.half_open_calls = 0
                else:
                    raise Exception("Circuit breaker is OPEN - service unavailable")

            if self.state == CircuitBreakerState.HALF_OPEN:
                if self.half_open_calls >= self.half_open_max_calls:
                    raise Exception("Circuit breaker HALF_OPEN - max test calls exceeded")
                self.half_open_calls += 1

        # Execute function
        try:
            result = await func(*args, **kwargs)

            # Success - reset failure count
            async with self._lock:
                if self.state == CircuitBreakerState.HALF_OPEN:
                    logger.info("Circuit breaker transitioning to CLOSED")
                    self.state = CircuitBreakerState.CLOSED
                self.failure_count = 0
                self.half_open_calls = 0

            return result

        except Exception as e:
            # Failure - increment failure count
            async with self._lock:
                self.failure_count += 1
                self.last_failure_time = datetime.now()

                if self.failure_count >= self.failure_threshold:
                    logger.error(f"Circuit breaker opening due to {self.failure_count} failures")
                    self.state = CircuitBreakerState.OPEN
                elif self.state == CircuitBreakerState.HALF_OPEN:
                    logger.warning("Circuit breaker reopening - test call failed")
                    self.state = CircuitBreakerState.OPEN

            raise


class DCSConnector:
    """
    DCS Connector with OPC UA (primary) and Modbus TCP (fallback).

    Features:
    - Async/await for all I/O operations
    - Connection pooling and health monitoring
    - Circuit breaker pattern for fault tolerance
    - Automatic protocol fallback
    - Real-time subscriptions for process variables
    - Historical data retrieval
    - Alarm management
    - Sub-100ms control loop support

    Example:
        config = DCSConfig(
            opcua_endpoint="opc.tcp://dcs.plant.com:4840",
            modbus_host="10.0.1.100"
        )

        async with DCSConnector(config) as connector:
            # Read process variables
            values = await connector.read_process_variables([
                "FurnaceTemp", "SteamPressure", "O2Content"
            ])

            # Write setpoint
            await connector.write_setpoints({
                "FuelFlowSetpoint": 150.5
            })

            # Subscribe to alarms
            await connector.subscribe_to_alarms(alarm_handler)
    """

    def __init__(self, config: DCSConfig):
        """Initialize DCS connector."""
        self.config = config
        self.connected = False
        self.active_protocol: Optional[DCSProtocol] = None

        # OPC UA client
        self.opcua_client: Optional[OPCUAClient] = None
        self.opcua_subscription = None

        # Modbus client
        self.modbus_client: Optional[AsyncModbusTcpClient] = None

        # Process variables registry
        self.process_variables: Dict[str, ProcessVariable] = {}

        # Circuit breakers (per protocol)
        self.circuit_breakers: Dict[DCSProtocol, CircuitBreaker] = {
            DCSProtocol.OPC_UA: CircuitBreaker(
                config.failure_threshold,
                config.recovery_timeout,
                config.half_open_max_calls
            ),
            DCSProtocol.MODBUS_TCP: CircuitBreaker(
                config.failure_threshold,
                config.recovery_timeout,
                config.half_open_max_calls
            )
        }

        # Alarm management
        self.active_alarms: Dict[str, DCSAlarm] = {}
        self.alarm_callbacks: List[Callable] = []

        # Performance metrics
        self.read_latencies = deque(maxlen=1000)
        self.write_latencies = deque(maxlen=1000)
        self.connection_health_score = 100.0

        # Connection health monitoring
        self._health_monitor_task: Optional[asyncio.Task] = None
        self._subscription_task: Optional[asyncio.Task] = None

        # Prometheus metrics
        if METRICS_AVAILABLE:
            self.metrics = {
                'read_count': Counter('dcs_reads_total', 'Total DCS read operations'),
                'write_count': Counter('dcs_writes_total', 'Total DCS write operations'),
                'read_latency': Histogram('dcs_read_latency_seconds', 'DCS read latency'),
                'write_latency': Histogram('dcs_write_latency_seconds', 'DCS write latency'),
                'connection_health': Gauge('dcs_connection_health_score', 'DCS connection health (0-100)'),
                'active_alarms': Gauge('dcs_active_alarms', 'Number of active DCS alarms')
            }
        else:
            self.metrics = {}

    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect_to_dcs()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()

    async def connect_to_dcs(self) -> bool:
        """
        Establish connection to DCS with automatic protocol fallback.

        Returns:
            True if connection successful, False otherwise

        Raises:
            ConnectionError: If all connection attempts fail
        """
        logger.info("Connecting to DCS system...")

        # Try primary protocol
        if self.config.primary_protocol == DCSProtocol.OPC_UA:
            if await self._connect_opcua():
                self.active_protocol = DCSProtocol.OPC_UA
                self.connected = True
                logger.info("Connected to DCS via OPC UA")

                # Start health monitor
                self._health_monitor_task = asyncio.create_task(self._health_monitor_loop())
                return True

        # Fallback to secondary protocol
        if self.config.fallback_protocol == DCSProtocol.MODBUS_TCP:
            logger.warning("Primary protocol failed, attempting Modbus TCP fallback")
            if await self._connect_modbus():
                self.active_protocol = DCSProtocol.MODBUS_TCP
                self.connected = True
                logger.info("Connected to DCS via Modbus TCP (fallback)")

                # Start health monitor
                self._health_monitor_task = asyncio.create_task(self._health_monitor_loop())
                return True

        raise ConnectionError("Failed to connect to DCS via all available protocols")

    async def _connect_opcua(self) -> bool:
        """Connect via OPC UA protocol."""
        if not OPCUA_AVAILABLE:
            logger.error("OPC UA library not available")
            return False

        try:
            self.opcua_client = OPCUAClient(url=self.config.opcua_endpoint)

            # Set security policy
            if self.config.opcua_security_policy != "None":
                await self.opcua_client.set_security_string(
                    f"{self.config.opcua_security_policy},{self.config.opcua_security_mode},"
                    f"{self.config.opcua_cert_path},{self.config.opcua_private_key_path}"
                )

            # Set authentication
            if self.config.opcua_username and self.config.opcua_password:
                self.opcua_client.set_user(self.config.opcua_username)
                self.opcua_client.set_password(self.config.opcua_password)

            # Connect with timeout
            await asyncio.wait_for(
                self.opcua_client.connect(),
                timeout=self.config.connection_timeout
            )

            logger.info(f"OPC UA connected to {self.config.opcua_endpoint}")
            return True

        except asyncio.TimeoutError:
            logger.error(f"OPC UA connection timeout after {self.config.connection_timeout}s")
            return False
        except Exception as e:
            logger.error(f"OPC UA connection failed: {e}")
            return False

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

            # Connect
            await self.modbus_client.connect()

            if self.modbus_client.connected:
                logger.info(f"Modbus TCP connected to {self.config.modbus_host}:{self.config.modbus_port}")
                return True
            else:
                logger.error("Modbus TCP connection failed")
                return False

        except Exception as e:
            logger.error(f"Modbus TCP connection failed: {e}")
            return False

    async def read_process_variables(
        self,
        tag_names: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Read process variables from DCS.

        Args:
            tag_names: List of tag names to read

        Returns:
            Dictionary mapping tag name to {value, quality, timestamp}

        Raises:
            ConnectionError: If not connected
            ValueError: If tag not found
        """
        if not self.connected:
            raise ConnectionError("Not connected to DCS")

        start_time = time.perf_counter()

        try:
            if self.active_protocol == DCSProtocol.OPC_UA:
                result = await self.circuit_breakers[DCSProtocol.OPC_UA].call(
                    self._read_opcua_variables, tag_names
                )
            else:
                result = await self.circuit_breakers[DCSProtocol.MODBUS_TCP].call(
                    self._read_modbus_variables, tag_names
                )

            # Record latency
            latency = time.perf_counter() - start_time
            self.read_latencies.append(latency)

            if self.metrics:
                self.metrics['read_count'].inc(len(tag_names))
                self.metrics['read_latency'].observe(latency)

            # Check for alarms
            await self._check_alarms(result)

            return result

        except Exception as e:
            logger.error(f"Failed to read process variables: {e}")
            raise

    async def _read_opcua_variables(
        self,
        tag_names: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        """Read variables via OPC UA."""
        result = {}

        for tag_name in tag_names:
            pv = self.process_variables.get(tag_name)
            if not pv:
                logger.warning(f"Tag {tag_name} not registered")
                continue

            try:
                node = self.opcua_client.get_node(pv.node_id)
                data_value = await node.read_data_value()

                # Extract value and quality
                value = data_value.Value.Value
                quality_code = data_value.StatusCode.name
                timestamp = data_value.SourceTimestamp

                # Apply scaling
                if pv.data_type in ['float', 'int']:
                    value = value * pv.scaling_factor + pv.offset

                # Update process variable
                pv.current_value = value
                pv.quality = self._map_opcua_quality(quality_code)
                pv.last_update = timestamp
                pv.consecutive_bad_reads = 0

                result[tag_name] = {
                    'value': value,
                    'quality': pv.quality.value,
                    'timestamp': timestamp.isoformat(),
                    'units': pv.engineering_units
                }

            except Exception as e:
                logger.error(f"Error reading OPC UA tag {tag_name}: {e}")
                pv.consecutive_bad_reads += 1
                result[tag_name] = {
                    'value': None,
                    'quality': DataQuality.BAD_COMM_FAILURE.value,
                    'timestamp': datetime.now().isoformat(),
                    'error': str(e)
                }

        return result

    async def _read_modbus_variables(
        self,
        tag_names: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        """Read variables via Modbus TCP."""
        result = {}

        for tag_name in tag_names:
            pv = self.process_variables.get(tag_name)
            if not pv:
                logger.warning(f"Tag {tag_name} not registered")
                continue

            try:
                # Parse Modbus register address
                register_address = int(pv.node_id)

                # Read holding register (function code 3)
                response = await self.modbus_client.read_holding_registers(
                    address=register_address,
                    count=1,
                    unit=self.config.modbus_unit_id
                )

                if response.isError():
                    raise Exception(f"Modbus error: {response}")

                value = response.registers[0]

                # Apply scaling
                value = value * pv.scaling_factor + pv.offset

                # Update process variable
                pv.current_value = value
                pv.quality = DataQuality.GOOD
                pv.last_update = datetime.now()
                pv.consecutive_bad_reads = 0

                result[tag_name] = {
                    'value': value,
                    'quality': DataQuality.GOOD.value,
                    'timestamp': datetime.now().isoformat(),
                    'units': pv.engineering_units
                }

            except Exception as e:
                logger.error(f"Error reading Modbus tag {tag_name}: {e}")
                pv.consecutive_bad_reads += 1
                result[tag_name] = {
                    'value': None,
                    'quality': DataQuality.BAD_COMM_FAILURE.value,
                    'timestamp': datetime.now().isoformat(),
                    'error': str(e)
                }

        return result

    async def write_setpoints(
        self,
        setpoints: Dict[str, float]
    ) -> Dict[str, bool]:
        """
        Write setpoints to DCS.

        Args:
            setpoints: Dictionary mapping tag name to setpoint value

        Returns:
            Dictionary mapping tag name to write success status

        Raises:
            ConnectionError: If not connected
            ValueError: If tag not writable
        """
        if not self.connected:
            raise ConnectionError("Not connected to DCS")

        start_time = time.perf_counter()
        result = {}

        for tag_name, value in setpoints.items():
            pv = self.process_variables.get(tag_name)
            if not pv:
                logger.warning(f"Tag {tag_name} not registered")
                result[tag_name] = False
                continue

            if not pv.writable:
                logger.error(f"Tag {tag_name} is read-only")
                result[tag_name] = False
                continue

            # Validate value range
            if pv.min_value is not None and value < pv.min_value:
                logger.error(f"Value {value} below minimum {pv.min_value} for {tag_name}")
                result[tag_name] = False
                continue

            if pv.max_value is not None and value > pv.max_value:
                logger.error(f"Value {value} above maximum {pv.max_value} for {tag_name}")
                result[tag_name] = False
                continue

            # Write to DCS
            try:
                if self.active_protocol == DCSProtocol.OPC_UA:
                    success = await self._write_opcua_variable(tag_name, value)
                else:
                    success = await self._write_modbus_variable(tag_name, value)

                result[tag_name] = success

            except Exception as e:
                logger.error(f"Failed to write {tag_name}: {e}")
                result[tag_name] = False

        # Record latency
        latency = time.perf_counter() - start_time
        self.write_latencies.append(latency)

        if self.metrics:
            self.metrics['write_count'].inc(len(setpoints))
            self.metrics['write_latency'].observe(latency)

        return result

    async def _write_opcua_variable(self, tag_name: str, value: float) -> bool:
        """Write variable via OPC UA."""
        pv = self.process_variables[tag_name]

        try:
            node = self.opcua_client.get_node(pv.node_id)

            # Apply reverse scaling
            scaled_value = (value - pv.offset) / pv.scaling_factor

            # Write value
            await node.write_value(scaled_value)

            logger.info(f"Wrote {value} to OPC UA tag {tag_name}")
            return True

        except Exception as e:
            logger.error(f"OPC UA write failed for {tag_name}: {e}")
            return False

    async def _write_modbus_variable(self, tag_name: str, value: float) -> bool:
        """Write variable via Modbus TCP."""
        pv = self.process_variables[tag_name]

        try:
            # Parse register address
            register_address = int(pv.node_id)

            # Apply reverse scaling
            scaled_value = int((value - pv.offset) / pv.scaling_factor)

            # Write holding register
            response = await self.modbus_client.write_register(
                address=register_address,
                value=scaled_value,
                unit=self.config.modbus_unit_id
            )

            if response.isError():
                raise Exception(f"Modbus error: {response}")

            logger.info(f"Wrote {value} to Modbus tag {tag_name}")
            return True

        except Exception as e:
            logger.error(f"Modbus write failed for {tag_name}: {e}")
            return False

    async def subscribe_to_alarms(self, callback: Callable[[DCSAlarm], None]):
        """
        Subscribe to DCS alarms with callback.

        Args:
            callback: Function called when alarm triggers
        """
        self.alarm_callbacks.append(callback)
        logger.info("Subscribed to DCS alarms")

    async def get_historical_data(
        self,
        tag_name: str,
        start_time: datetime,
        end_time: datetime,
        max_points: int = 10000
    ) -> List[Dict[str, Any]]:
        """
        Retrieve historical trend data from DCS.

        Args:
            tag_name: Tag name to retrieve
            start_time: Start of time window
            end_time: End of time window
            max_points: Maximum number of data points

        Returns:
            List of {timestamp, value, quality} dictionaries
        """
        if not self.connected:
            raise ConnectionError("Not connected to DCS")

        if self.active_protocol != DCSProtocol.OPC_UA:
            logger.warning("Historical data only available via OPC UA")
            return []

        pv = self.process_variables.get(tag_name)
        if not pv:
            raise ValueError(f"Tag {tag_name} not registered")

        try:
            node = self.opcua_client.get_node(pv.node_id)

            # Read historical data
            history = await node.read_raw_history(
                start_time,
                end_time,
                max_points
            )

            result = []
            for data_value in history:
                value = data_value.Value.Value
                if pv.data_type in ['float', 'int']:
                    value = value * pv.scaling_factor + pv.offset

                result.append({
                    'timestamp': data_value.SourceTimestamp.isoformat(),
                    'value': value,
                    'quality': self._map_opcua_quality(data_value.StatusCode.name).value
                })

            logger.info(f"Retrieved {len(result)} historical points for {tag_name}")
            return result

        except Exception as e:
            logger.error(f"Failed to retrieve historical data: {e}")
            return []

    def register_process_variable(self, pv: ProcessVariable):
        """Register a process variable for monitoring."""
        self.process_variables[pv.tag_name] = pv
        logger.info(f"Registered process variable: {pv.tag_name}")

    async def _check_alarms(self, readings: Dict[str, Dict[str, Any]]):
        """Check for alarm conditions in readings."""
        for tag_name, data in readings.items():
            pv = self.process_variables.get(tag_name)
            if not pv or data['value'] is None:
                continue

            value = data['value']

            # Check alarm limits
            alarms_triggered = []

            if pv.alarm_high_high and value >= pv.alarm_high_high:
                alarms_triggered.append(('HH', pv.alarm_high_high, 1))
            elif pv.alarm_high and value >= pv.alarm_high:
                alarms_triggered.append(('H', pv.alarm_high, 2))

            if pv.alarm_low_low and value <= pv.alarm_low_low:
                alarms_triggered.append(('LL', pv.alarm_low_low, 1))
            elif pv.alarm_low and value <= pv.alarm_low:
                alarms_triggered.append(('L', pv.alarm_low, 2))

            # Trigger alarm callbacks
            for alarm_type, setpoint, priority in alarms_triggered:
                alarm_id = f"{tag_name}_{alarm_type}"

                if alarm_id not in self.active_alarms:
                    alarm = DCSAlarm(
                        alarm_id=alarm_id,
                        tag_name=tag_name,
                        alarm_type=alarm_type,
                        priority=priority,
                        setpoint=setpoint,
                        actual_value=value,
                        message=f"{tag_name} {alarm_type} alarm: {value:.2f} {pv.engineering_units}",
                        timestamp=datetime.now()
                    )

                    self.active_alarms[alarm_id] = alarm

                    # Call alarm callbacks
                    for callback in self.alarm_callbacks:
                        try:
                            await callback(alarm)
                        except Exception as e:
                            logger.error(f"Alarm callback failed: {e}")

                    if self.metrics:
                        self.metrics['active_alarms'].set(len(self.active_alarms))

    async def _health_monitor_loop(self):
        """Background task for connection health monitoring."""
        while self.connected:
            try:
                # Calculate health score based on recent performance
                if self.read_latencies:
                    avg_latency = sum(self.read_latencies) / len(self.read_latencies)
                    max_latency = max(self.read_latencies)

                    # Health score components:
                    # - Latency: 50 points (100ms target)
                    # - Reliability: 30 points (consecutive failures)
                    # - Data quality: 20 points (good quality readings)

                    latency_score = max(0, 50 * (1 - avg_latency / 0.1))  # 100ms = 0 score

                    # Count recent failures
                    failure_count = sum(
                        1 for pv in self.process_variables.values()
                        if pv.consecutive_bad_reads > 0
                    )
                    reliability_score = max(0, 30 * (1 - failure_count / len(self.process_variables)))

                    # Count good quality readings
                    good_quality_count = sum(
                        1 for pv in self.process_variables.values()
                        if pv.quality == DataQuality.GOOD
                    )
                    quality_score = 20 * (good_quality_count / max(len(self.process_variables), 1))

                    self.connection_health_score = latency_score + reliability_score + quality_score

                    if self.metrics:
                        self.metrics['connection_health'].set(self.connection_health_score)

                    logger.debug(f"DCS health score: {self.connection_health_score:.1f}")

                await asyncio.sleep(self.config.keepalive_interval)

            except Exception as e:
                logger.error(f"Health monitor error: {e}")
                await asyncio.sleep(self.config.keepalive_interval)

    def _map_opcua_quality(self, quality_code: str) -> DataQuality:
        """Map OPC UA quality code to DataQuality enum."""
        if 'Good' in quality_code:
            return DataQuality.GOOD
        elif 'Bad' in quality_code:
            if 'Comm' in quality_code:
                return DataQuality.BAD_COMM_FAILURE
            elif 'Sensor' in quality_code:
                return DataQuality.BAD_SENSOR_FAILURE
            else:
                return DataQuality.BAD
        elif 'Uncertain' in quality_code:
            return DataQuality.UNCERTAIN
        else:
            return DataQuality.BAD

    async def disconnect(self):
        """Disconnect from DCS."""
        logger.info("Disconnecting from DCS...")

        # Stop health monitor
        if self._health_monitor_task:
            self._health_monitor_task.cancel()
            try:
                await self._health_monitor_task
            except asyncio.CancelledError:
                pass

        # Disconnect OPC UA
        if self.opcua_client:
            try:
                await self.opcua_client.disconnect()
            except Exception as e:
                logger.error(f"OPC UA disconnect error: {e}")

        # Disconnect Modbus
        if self.modbus_client:
            try:
                self.modbus_client.close()
            except Exception as e:
                logger.error(f"Modbus disconnect error: {e}")

        self.connected = False
        logger.info("Disconnected from DCS")
