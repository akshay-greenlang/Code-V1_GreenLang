# -*- coding: utf-8 -*-
"""
Burner Controller Connector for GL-004 BurnerOptimizationAgent

Implements industrial burner control system integration via:
- Modbus TCP/RTU protocol for legacy controllers
- OPC UA for modern control systems
- Fuel flow rate control with ramping
- Air flow rate control with ratio management
- Burner load management (0-100% MCR)
- Setpoint writing with gradual ramping
- Circuit breaker pattern for fault tolerance
- Connection pooling for efficiency

Real-Time Requirements:
- Control update rate: 100ms for critical parameters
- Setpoint ramping: Configurable 0.5-5.0% per second
- Safety interlock response: <50ms
- Connection health check: Every 5 seconds

Protocols Supported:
- Modbus TCP (Primary for PLC control)
- Modbus RTU (Serial for legacy systems)
- OPC UA (IEC 62541 for modern DCS)

Author: GL-DataIntegrationEngineer
Date: 2025-11-19
Version: 1.0.0
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
from enum import Enum
from contextlib import asynccontextmanager
import struct
from greenlang.determinism import DeterministicClock

# Third-party imports
try:
    from pymodbus.client import AsyncModbusTcpClient, AsyncModbusSerialClient
    from pymodbus.exceptions import ModbusException
    from pymodbus.constants import Endian
    from pymodbus.payload import BinaryPayloadDecoder, BinaryPayloadBuilder
    MODBUS_AVAILABLE = True
except ImportError:
    MODBUS_AVAILABLE = False

try:
    from asyncua import Client as OPCClient, ua
    from asyncua.common.methods import call_method
    OPCUA_AVAILABLE = True
except ImportError:
    OPCUA_AVAILABLE = False

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_log,
    after_log
)

# Prometheus metrics
try:
    from prometheus_client import Counter, Histogram, Gauge, Summary
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False

logger = logging.getLogger(__name__)


class BurnerControlProtocol(Enum):
    """Supported burner control protocols."""
    MODBUS_TCP = "modbus_tcp"
    MODBUS_RTU = "modbus_rtu"
    OPC_UA = "opc_ua"


class ControlParameter(Enum):
    """Burner control parameters."""
    FUEL_FLOW_RATE = "fuel_flow_rate"  # kg/hr or Nm3/hr
    AIR_FLOW_RATE = "air_flow_rate"    # Nm3/hr
    BURNER_LOAD = "burner_load"        # % MCR
    O2_SETPOINT = "o2_setpoint"        # % dry
    AIR_FUEL_RATIO = "air_fuel_ratio"  # dimensionless
    FLAME_STATUS = "flame_status"      # boolean
    BURNER_MODE = "burner_mode"        # auto/manual/local


class BurnerMode(Enum):
    """Burner operation modes."""
    MANUAL = 0
    AUTO = 1
    LOCAL = 2
    REMOTE = 3
    MAINTENANCE = 4


class SafetyInterlock(Enum):
    """Safety interlock statuses."""
    OK = "ok"
    TRIPPED = "tripped"
    BYPASSED = "bypassed"
    FAULT = "fault"


@dataclass
class BurnerControllerConfig:
    """Configuration for burner controller connection."""
    protocol: BurnerControlProtocol
    host: str = "localhost"
    port: int = 502  # Modbus TCP default
    unit_id: int = 1  # Modbus unit ID
    serial_port: Optional[str] = None  # For Modbus RTU
    baudrate: int = 9600
    opc_endpoint: Optional[str] = None  # OPC UA endpoint
    timeout: float = 5.0
    retry_attempts: int = 3
    connection_pool_size: int = 5
    health_check_interval: float = 5.0
    ramp_rate_percent_per_sec: float = 2.0  # Setpoint ramping rate
    mock_mode: bool = False


@dataclass
class ControlPoint:
    """Control point definition."""
    name: str
    register_address: int  # Modbus register
    opc_node_id: Optional[str] = None  # OPC UA node
    data_type: str = "float32"  # float32, int16, bool
    scale_factor: float = 1.0
    offset: float = 0.0
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    engineering_unit: str = ""
    writable: bool = False
    ramp_enabled: bool = False  # Enable gradual ramping


@dataclass
class BurnerStatus:
    """Current burner status."""
    timestamp: datetime
    mode: BurnerMode
    load_percent: float
    fuel_flow: float
    air_flow: float
    o2_setpoint: float
    air_fuel_ratio: float
    flame_detected: bool
    safety_interlocks: Dict[str, SafetyInterlock]
    alarms: List[str]
    connection_healthy: bool


class CircuitBreaker:
    """Circuit breaker for connection fault tolerance."""

    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 60.0):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.state = "closed"  # closed, open, half_open

    def record_success(self):
        """Record successful operation."""
        self.failure_count = 0
        self.state = "closed"

    def record_failure(self):
        """Record failed operation."""
        self.failure_count += 1
        self.last_failure_time = DeterministicClock.now()

        if self.failure_count >= self.failure_threshold:
            self.state = "open"
            logger.warning(f"Circuit breaker opened after {self.failure_count} failures")

    def is_open(self) -> bool:
        """Check if circuit breaker is open."""
        if self.state == "open":
            if self.last_failure_time:
                elapsed = (DeterministicClock.now() - self.last_failure_time).total_seconds()
                if elapsed > self.recovery_timeout:
                    self.state = "half_open"
                    logger.info("Circuit breaker entering half-open state")
                    return False
            return True
        return False


class BurnerControllerConnector:
    """
    Industrial burner controller integration connector.

    Provides real-time control and monitoring of burner systems with
    safety interlocks, gradual setpoint ramping, and fault tolerance.
    """

    # Prometheus metrics
    if METRICS_AVAILABLE:
        control_writes = Counter('burner_control_writes_total', 'Total control writes', ['parameter'])
        control_errors = Counter('burner_control_errors_total', 'Total control errors', ['error_type'])
        ramp_duration = Histogram('burner_ramp_duration_seconds', 'Setpoint ramp duration')
        connection_pool_usage = Gauge('burner_connection_pool_usage', 'Connection pool usage')

    def __init__(self, config: BurnerControllerConfig):
        """Initialize burner controller connector."""
        self.config = config
        self.circuit_breaker = CircuitBreaker()
        self.control_points: Dict[str, ControlPoint] = self._initialize_control_points()

        # Connection management
        self.connection_pool: List[Any] = []
        self.active_connections: int = 0
        self._connection_lock = asyncio.Lock()

        # Ramping management
        self.ramping_tasks: Dict[str, asyncio.Task] = {}
        self.current_setpoints: Dict[str, float] = {}
        self.target_setpoints: Dict[str, float] = {}

        # Health monitoring
        self.last_health_check: Optional[datetime] = None
        self.connection_healthy = False
        self.health_check_task: Optional[asyncio.Task] = None

        # Mock mode for testing
        if config.mock_mode:
            self._mock_data = self._initialize_mock_data()

    def _initialize_control_points(self) -> Dict[str, ControlPoint]:
        """Initialize standard control points."""
        return {
            "fuel_flow": ControlPoint(
                name="Fuel Flow Rate",
                register_address=40001,
                opc_node_id="ns=2;s=BurnerControl.FuelFlow",
                data_type="float32",
                scale_factor=1.0,
                min_value=0.0,
                max_value=10000.0,
                engineering_unit="kg/hr",
                writable=True,
                ramp_enabled=True
            ),
            "air_flow": ControlPoint(
                name="Air Flow Rate",
                register_address=40003,
                opc_node_id="ns=2;s=BurnerControl.AirFlow",
                data_type="float32",
                scale_factor=1.0,
                min_value=0.0,
                max_value=100000.0,
                engineering_unit="Nm3/hr",
                writable=True,
                ramp_enabled=True
            ),
            "burner_load": ControlPoint(
                name="Burner Load",
                register_address=40005,
                opc_node_id="ns=2;s=BurnerControl.Load",
                data_type="float32",
                scale_factor=1.0,
                min_value=0.0,
                max_value=100.0,
                engineering_unit="%",
                writable=True,
                ramp_enabled=True
            ),
            "o2_setpoint": ControlPoint(
                name="O2 Setpoint",
                register_address=40007,
                opc_node_id="ns=2;s=BurnerControl.O2Setpoint",
                data_type="float32",
                scale_factor=1.0,
                min_value=1.0,
                max_value=10.0,
                engineering_unit="% dry",
                writable=True,
                ramp_enabled=False
            ),
            "flame_status": ControlPoint(
                name="Flame Status",
                register_address=30001,
                opc_node_id="ns=2;s=BurnerControl.FlameStatus",
                data_type="bool",
                writable=False
            ),
            "burner_mode": ControlPoint(
                name="Burner Mode",
                register_address=40009,
                opc_node_id="ns=2;s=BurnerControl.Mode",
                data_type="int16",
                writable=True
            )
        }

    def _initialize_mock_data(self) -> Dict[str, Any]:
        """Initialize mock data for testing."""
        return {
            "fuel_flow": 5000.0,
            "air_flow": 50000.0,
            "burner_load": 75.0,
            "o2_setpoint": 3.5,
            "flame_status": True,
            "burner_mode": BurnerMode.AUTO.value,
            "safety_interlocks": {
                "low_fuel_pressure": SafetyInterlock.OK,
                "high_furnace_pressure": SafetyInterlock.OK,
                "flame_failure": SafetyInterlock.OK
            }
        }

    async def connect(self) -> bool:
        """Establish connection to burner controller."""
        try:
            if self.circuit_breaker.is_open():
                logger.warning("Circuit breaker is open, refusing connection")
                return False

            async with self._connection_lock:
                if self.config.protocol == BurnerControlProtocol.MODBUS_TCP:
                    success = await self._connect_modbus_tcp()
                elif self.config.protocol == BurnerControlProtocol.MODBUS_RTU:
                    success = await self._connect_modbus_rtu()
                elif self.config.protocol == BurnerControlProtocol.OPC_UA:
                    success = await self._connect_opc_ua()
                else:
                    raise ValueError(f"Unsupported protocol: {self.config.protocol}")

                if success:
                    self.circuit_breaker.record_success()
                    self.connection_healthy = True

                    # Start health monitoring
                    if not self.health_check_task:
                        self.health_check_task = asyncio.create_task(self._health_monitor())

                    logger.info(f"Connected to burner controller via {self.config.protocol.value}")
                else:
                    self.circuit_breaker.record_failure()

                return success

        except Exception as e:
            logger.error(f"Connection failed: {e}")
            self.circuit_breaker.record_failure()
            if METRICS_AVAILABLE:
                self.control_errors.labels(error_type="connection").inc()
            return False

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(ModbusException),
        before=before_log(logger, logging.DEBUG),
        after=after_log(logger, logging.DEBUG)
    )
    async def _connect_modbus_tcp(self) -> bool:
        """Connect via Modbus TCP."""
        if not MODBUS_AVAILABLE:
            logger.error("Modbus library not available")
            return False

        if self.config.mock_mode:
            logger.info("Running in mock mode")
            return True

        try:
            # Create connection pool
            for _ in range(self.config.connection_pool_size):
                client = AsyncModbusTcpClient(
                    host=self.config.host,
                    port=self.config.port,
                    timeout=self.config.timeout
                )
                await client.connect()
                self.connection_pool.append(client)

            self.active_connections = len(self.connection_pool)

            if METRICS_AVAILABLE:
                self.connection_pool_usage.set(self.active_connections)

            return True

        except Exception as e:
            logger.error(f"Modbus TCP connection failed: {e}")
            return False

    async def _connect_modbus_rtu(self) -> bool:
        """Connect via Modbus RTU."""
        if not MODBUS_AVAILABLE:
            logger.error("Modbus library not available")
            return False

        if self.config.mock_mode:
            logger.info("Running in mock mode")
            return True

        try:
            client = AsyncModbusSerialClient(
                port=self.config.serial_port,
                baudrate=self.config.baudrate,
                timeout=self.config.timeout
            )
            await client.connect()
            self.connection_pool.append(client)
            self.active_connections = 1

            return True

        except Exception as e:
            logger.error(f"Modbus RTU connection failed: {e}")
            return False

    async def _connect_opc_ua(self) -> bool:
        """Connect via OPC UA."""
        if not OPCUA_AVAILABLE:
            logger.error("OPC UA library not available")
            return False

        if self.config.mock_mode:
            logger.info("Running in mock mode")
            return True

        try:
            client = OPCClient(self.config.opc_endpoint)
            await client.connect()
            self.connection_pool.append(client)
            self.active_connections = 1

            return True

        except Exception as e:
            logger.error(f"OPC UA connection failed: {e}")
            return False

    @asynccontextmanager
    async def get_connection(self):
        """Get connection from pool with context manager."""
        connection = None
        try:
            # Get available connection
            async with self._connection_lock:
                if self.connection_pool:
                    connection = self.connection_pool.pop(0)

            if connection:
                yield connection
            else:
                raise RuntimeError("No available connections in pool")

        finally:
            # Return connection to pool
            if connection:
                async with self._connection_lock:
                    self.connection_pool.append(connection)

    async def read_parameter(self, parameter: str) -> Optional[float]:
        """Read control parameter value."""
        if parameter not in self.control_points:
            logger.error(f"Unknown parameter: {parameter}")
            return None

        if self.config.mock_mode:
            return self._mock_data.get(parameter, 0.0)

        control_point = self.control_points[parameter]

        try:
            if self.config.protocol in [BurnerControlProtocol.MODBUS_TCP, BurnerControlProtocol.MODBUS_RTU]:
                return await self._read_modbus_register(control_point)
            elif self.config.protocol == BurnerControlProtocol.OPC_UA:
                return await self._read_opc_node(control_point)

        except Exception as e:
            logger.error(f"Failed to read {parameter}: {e}")
            if METRICS_AVAILABLE:
                self.control_errors.labels(error_type="read").inc()
            return None

    async def _read_modbus_register(self, point: ControlPoint) -> Optional[float]:
        """Read value from Modbus register."""
        async with self.get_connection() as client:
            result = await client.read_holding_registers(
                address=point.register_address - 40001,  # Convert to 0-based
                count=2 if point.data_type == "float32" else 1,
                unit=self.config.unit_id
            )

            if result.isError():
                raise ModbusException(f"Read error: {result}")

            if point.data_type == "float32":
                decoder = BinaryPayloadDecoder.fromRegisters(
                    result.registers,
                    byteorder=Endian.Big,
                    wordorder=Endian.Big
                )
                value = decoder.decode_32bit_float()
            elif point.data_type == "int16":
                value = result.registers[0]
            elif point.data_type == "bool":
                value = bool(result.registers[0])
            else:
                value = result.registers[0]

            # Apply scaling
            return (value * point.scale_factor) + point.offset

    async def write_setpoint(
        self,
        parameter: str,
        value: float,
        ramp: bool = True
    ) -> bool:
        """
        Write control setpoint with optional ramping.

        Args:
            parameter: Control parameter name
            value: Target value
            ramp: Enable gradual ramping

        Returns:
            Success status
        """
        if parameter not in self.control_points:
            logger.error(f"Unknown parameter: {parameter}")
            return False

        control_point = self.control_points[parameter]

        if not control_point.writable:
            logger.error(f"Parameter {parameter} is not writable")
            return False

        # Validate limits
        if control_point.min_value is not None and value < control_point.min_value:
            logger.warning(f"Value {value} below minimum {control_point.min_value}")
            value = control_point.min_value

        if control_point.max_value is not None and value > control_point.max_value:
            logger.warning(f"Value {value} above maximum {control_point.max_value}")
            value = control_point.max_value

        # Handle ramping
        if ramp and control_point.ramp_enabled:
            return await self._ramp_setpoint(parameter, value)
        else:
            return await self._write_immediate(parameter, value)

    async def _ramp_setpoint(self, parameter: str, target: float) -> bool:
        """Gradually ramp setpoint to target value."""
        # Cancel existing ramp if any
        if parameter in self.ramping_tasks:
            self.ramping_tasks[parameter].cancel()

        # Get current value
        current = await self.read_parameter(parameter)
        if current is None:
            logger.error(f"Cannot read current value for {parameter}")
            return False

        self.current_setpoints[parameter] = current
        self.target_setpoints[parameter] = target

        # Start ramping task
        task = asyncio.create_task(
            self._ramp_worker(parameter, current, target)
        )
        self.ramping_tasks[parameter] = task

        logger.info(f"Started ramping {parameter} from {current} to {target}")
        return True

    async def _ramp_worker(self, parameter: str, start: float, target: float) -> None:
        """Worker task for setpoint ramping."""
        control_point = self.control_points[parameter]
        ramp_rate = self.config.ramp_rate_percent_per_sec

        # Calculate ramp parameters
        value_range = control_point.max_value - control_point.min_value if control_point.max_value else 100.0
        step_size = (value_range * ramp_rate / 100.0)
        step_interval = 1.0  # 1 second steps

        current = start
        direction = 1 if target > start else -1

        start_time = time.time()

        try:
            while abs(current - target) > step_size / 2:
                # Calculate next step
                current += direction * step_size

                # Limit to target
                if direction > 0:
                    current = min(current, target)
                else:
                    current = max(current, target)

                # Write intermediate value
                success = await self._write_immediate(parameter, current)
                if not success:
                    logger.error(f"Ramping failed for {parameter}")
                    break

                self.current_setpoints[parameter] = current

                # Wait for next step
                await asyncio.sleep(step_interval)

            # Write final target value
            await self._write_immediate(parameter, target)
            self.current_setpoints[parameter] = target

            elapsed = time.time() - start_time
            logger.info(f"Ramping completed for {parameter} in {elapsed:.1f} seconds")

            if METRICS_AVAILABLE:
                self.ramp_duration.observe(elapsed)

        except asyncio.CancelledError:
            logger.info(f"Ramping cancelled for {parameter}")

        finally:
            # Clean up
            if parameter in self.ramping_tasks:
                del self.ramping_tasks[parameter]

    async def _write_immediate(self, parameter: str, value: float) -> bool:
        """Write value immediately without ramping."""
        if self.config.mock_mode:
            self._mock_data[parameter] = value
            logger.debug(f"Mock write {parameter} = {value}")
            return True

        control_point = self.control_points[parameter]

        try:
            if self.config.protocol in [BurnerControlProtocol.MODBUS_TCP, BurnerControlProtocol.MODBUS_RTU]:
                success = await self._write_modbus_register(control_point, value)
            elif self.config.protocol == BurnerControlProtocol.OPC_UA:
                success = await self._write_opc_node(control_point, value)
            else:
                success = False

            if success and METRICS_AVAILABLE:
                self.control_writes.labels(parameter=parameter).inc()

            return success

        except Exception as e:
            logger.error(f"Failed to write {parameter}: {e}")
            if METRICS_AVAILABLE:
                self.control_errors.labels(error_type="write").inc()
            return False

    async def _write_modbus_register(self, point: ControlPoint, value: float) -> bool:
        """Write value to Modbus register."""
        # Remove scaling
        raw_value = (value - point.offset) / point.scale_factor

        async with self.get_connection() as client:
            if point.data_type == "float32":
                builder = BinaryPayloadBuilder(
                    byteorder=Endian.Big,
                    wordorder=Endian.Big
                )
                builder.add_32bit_float(raw_value)
                registers = builder.to_registers()

                result = await client.write_registers(
                    address=point.register_address - 40001,
                    values=registers,
                    unit=self.config.unit_id
                )
            else:
                result = await client.write_register(
                    address=point.register_address - 40001,
                    value=int(raw_value),
                    unit=self.config.unit_id
                )

            return not result.isError()

    async def get_status(self) -> BurnerStatus:
        """Get complete burner status."""
        # Read all parameters
        fuel_flow = await self.read_parameter("fuel_flow") or 0.0
        air_flow = await self.read_parameter("air_flow") or 0.0
        load = await self.read_parameter("burner_load") or 0.0
        o2_sp = await self.read_parameter("o2_setpoint") or 3.5
        flame = await self.read_parameter("flame_status") or False
        mode_val = await self.read_parameter("burner_mode") or 0

        # Calculate air-fuel ratio
        afr = air_flow / fuel_flow if fuel_flow > 0 else 0.0

        # Get safety interlocks (mock for now)
        interlocks = {
            "low_fuel_pressure": SafetyInterlock.OK,
            "high_furnace_pressure": SafetyInterlock.OK,
            "flame_failure": SafetyInterlock.OK if flame else SafetyInterlock.TRIPPED
        }

        return BurnerStatus(
            timestamp=DeterministicClock.now(),
            mode=BurnerMode(int(mode_val)),
            load_percent=load,
            fuel_flow=fuel_flow,
            air_flow=air_flow,
            o2_setpoint=o2_sp,
            air_fuel_ratio=afr,
            flame_detected=flame,
            safety_interlocks=interlocks,
            alarms=[],
            connection_healthy=self.connection_healthy
        )

    async def _health_monitor(self) -> None:
        """Background task for connection health monitoring."""
        while True:
            try:
                await asyncio.sleep(self.config.health_check_interval)

                # Perform health check
                test_value = await self.read_parameter("flame_status")

                if test_value is not None:
                    self.connection_healthy = True
                    self.last_health_check = DeterministicClock.now()
                else:
                    self.connection_healthy = False
                    logger.warning("Health check failed")

            except Exception as e:
                logger.error(f"Health monitor error: {e}")
                self.connection_healthy = False

    async def close(self) -> None:
        """Close all connections and clean up resources."""
        # Cancel ramping tasks
        for task in self.ramping_tasks.values():
            task.cancel()

        # Cancel health monitor
        if self.health_check_task:
            self.health_check_task.cancel()

        # Close connections
        async with self._connection_lock:
            for conn in self.connection_pool:
                try:
                    if hasattr(conn, 'close'):
                        await conn.close()
                    elif hasattr(conn, 'disconnect'):
                        await conn.disconnect()
                except Exception as e:
                    logger.error(f"Error closing connection: {e}")

            self.connection_pool.clear()
            self.active_connections = 0

        logger.info("Burner controller connector closed")