# -*- coding: utf-8 -*-
"""
Heat Exchanger Connector for GL-006 HeatRecoveryMaximizer

Implements Modbus TCP/RTU and OPC UA protocols for real-time monitoring
of heat exchanger performance, temperature differentials, flow rates,
and fouling detection with 99.9% uptime design.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import struct
import numpy as np
from abc import ABC, abstractmethod
from greenlang.determinism import DeterministicClock

# Third-party imports (would be actual in production)
try:
    from pymodbus.client import AsyncModbusTcpClient, AsyncModbusSerialClient
    from pymodbus.exceptions import ModbusException
    from asyncua import Client as OPCClient, ua
    MODBUS_AVAILABLE = True
    OPCUA_AVAILABLE = True
except ImportError:
    MODBUS_AVAILABLE = False
    OPCUA_AVAILABLE = False

logger = logging.getLogger(__name__)


class ProtocolType(Enum):
    """Supported communication protocols."""
    MODBUS_TCP = "modbus_tcp"
    MODBUS_RTU = "modbus_rtu"
    OPC_UA = "opc_ua"


class HeatExchangerStatus(Enum):
    """Heat exchanger operational status."""
    ONLINE = "online"
    OFFLINE = "offline"
    MAINTENANCE = "maintenance"
    FOULED = "fouled"
    DEGRADED = "degraded"
    FAULT = "fault"


@dataclass
class HeatExchangerMetrics:
    """Real-time heat exchanger performance metrics."""
    timestamp: datetime
    hot_inlet_temp: float  # °C
    hot_outlet_temp: float  # °C
    cold_inlet_temp: float  # °C
    cold_outlet_temp: float  # °C
    hot_flow_rate: float  # kg/s
    cold_flow_rate: float  # kg/s
    effectiveness: float  # 0-1
    heat_transfer_rate: float  # kW
    pressure_drop_hot: float  # kPa
    pressure_drop_cold: float  # kPa
    fouling_factor: float  # m²K/W
    status: HeatExchangerStatus
    warnings: List[str] = field(default_factory=list)


@dataclass
class ConnectionConfig:
    """Heat exchanger connection configuration."""
    protocol: ProtocolType
    host: str
    port: int
    unit_id: int = 1
    timeout: float = 10.0
    retry_count: int = 3
    retry_delay: float = 1.0

    # Modbus RTU specific
    serial_port: Optional[str] = None
    baudrate: int = 9600
    parity: str = 'N'
    stopbits: int = 1
    bytesize: int = 8

    # OPC UA specific
    endpoint_url: Optional[str] = None
    namespace: str = "ns=2"
    username: Optional[str] = None
    password: Optional[str] = None

    # Circuit breaker settings
    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    half_open_requests: int = 3


class CircuitBreaker:
    """Circuit breaker pattern for connection resilience."""

    class State(Enum):
        CLOSED = "closed"  # Normal operation
        OPEN = "open"  # Failing, reject requests
        HALF_OPEN = "half_open"  # Testing recovery

    def __init__(self, config: ConnectionConfig):
        self.config = config
        self.state = self.State.CLOSED
        self.failure_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.success_count = 0
        self._lock = asyncio.Lock()

    async def call(self, func, *args, **kwargs):
        """Execute function through circuit breaker."""
        async with self._lock:
            if self.state == self.State.OPEN:
                if self._should_attempt_reset():
                    self.state = self.State.HALF_OPEN
                    self.success_count = 0
                else:
                    raise Exception("Circuit breaker is OPEN")

            try:
                result = await func(*args, **kwargs)
                await self._on_success()
                return result
            except Exception as e:
                await self._on_failure()
                raise

    async def _on_success(self):
        """Handle successful call."""
        self.failure_count = 0
        if self.state == self.State.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.half_open_requests:
                self.state = self.State.CLOSED
                logger.info("Circuit breaker closed after recovery")

    async def _on_failure(self):
        """Handle failed call."""
        self.failure_count += 1
        self.last_failure_time = DeterministicClock.now()

        if self.failure_count >= self.config.failure_threshold:
            self.state = self.State.OPEN
            logger.warning(f"Circuit breaker opened after {self.failure_count} failures")

        if self.state == self.State.HALF_OPEN:
            self.state = self.State.OPEN
            logger.warning("Circuit breaker reopened during recovery")

    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt reset."""
        if not self.last_failure_time:
            return False

        time_since_failure = (DeterministicClock.now() - self.last_failure_time).total_seconds()
        return time_since_failure >= self.config.recovery_timeout


class BaseProtocolHandler(ABC):
    """Base class for protocol handlers."""

    def __init__(self, config: ConnectionConfig):
        self.config = config
        self.connected = False
        self.circuit_breaker = CircuitBreaker(config)

    @abstractmethod
    async def connect(self) -> bool:
        """Establish connection."""
        pass

    @abstractmethod
    async def disconnect(self):
        """Close connection."""
        pass

    @abstractmethod
    async def read_registers(self, address: int, count: int) -> List[int]:
        """Read holding registers."""
        pass

    @abstractmethod
    async def write_register(self, address: int, value: int) -> bool:
        """Write single register."""
        pass


class ModbusHandler(BaseProtocolHandler):
    """Modbus TCP/RTU protocol handler."""

    def __init__(self, config: ConnectionConfig):
        super().__init__(config)
        self.client = None

    async def connect(self) -> bool:
        """Establish Modbus connection."""
        if not MODBUS_AVAILABLE:
            logger.warning("Modbus libraries not available, using mock mode")
            self.connected = True
            return True

        try:
            if self.config.protocol == ProtocolType.MODBUS_TCP:
                self.client = AsyncModbusTcpClient(
                    self.config.host,
                    port=self.config.port,
                    timeout=self.config.timeout
                )
            else:  # MODBUS_RTU
                self.client = AsyncModbusSerialClient(
                    port=self.config.serial_port,
                    baudrate=self.config.baudrate,
                    parity=self.config.parity,
                    stopbits=self.config.stopbits,
                    bytesize=self.config.bytesize,
                    timeout=self.config.timeout
                )

            await self.client.connect()
            self.connected = True
            logger.info(f"Connected to Modbus {self.config.protocol.value}")
            return True

        except Exception as e:
            logger.error(f"Modbus connection failed: {e}")
            self.connected = False
            return False

    async def disconnect(self):
        """Close Modbus connection."""
        if self.client:
            self.client.close()
            self.connected = False
            logger.info("Modbus connection closed")

    async def read_registers(self, address: int, count: int) -> List[int]:
        """Read holding registers with retry logic."""
        if not self.connected:
            await self.connect()

        for attempt in range(self.config.retry_count):
            try:
                if not MODBUS_AVAILABLE:
                    # Mock data for testing
                    return [3200 + i for i in range(count)]

                result = await self.client.read_holding_registers(
                    address, count, self.config.unit_id
                )

                if not result.isError():
                    return result.registers

            except Exception as e:
                logger.warning(f"Read attempt {attempt + 1} failed: {e}")
                await asyncio.sleep(self.config.retry_delay * (2 ** attempt))

        raise Exception(f"Failed to read registers after {self.config.retry_count} attempts")

    async def write_register(self, address: int, value: int) -> bool:
        """Write single register."""
        if not self.connected:
            await self.connect()

        try:
            if not MODBUS_AVAILABLE:
                return True

            result = await self.client.write_register(
                address, value, self.config.unit_id
            )
            return not result.isError()

        except Exception as e:
            logger.error(f"Write register failed: {e}")
            return False


class OPCUAHandler(BaseProtocolHandler):
    """OPC UA protocol handler."""

    def __init__(self, config: ConnectionConfig):
        super().__init__(config)
        self.client = None
        self.namespace_index = 2

    async def connect(self) -> bool:
        """Establish OPC UA connection."""
        if not OPCUA_AVAILABLE:
            logger.warning("OPC UA libraries not available, using mock mode")
            self.connected = True
            return True

        try:
            self.client = OPCClient(self.config.endpoint_url)

            if self.config.username and self.config.password:
                self.client.set_user(self.config.username)
                self.client.set_password(self.config.password)

            await self.client.connect()
            self.connected = True
            logger.info(f"Connected to OPC UA server at {self.config.endpoint_url}")
            return True

        except Exception as e:
            logger.error(f"OPC UA connection failed: {e}")
            self.connected = False
            return False

    async def disconnect(self):
        """Close OPC UA connection."""
        if self.client:
            await self.client.disconnect()
            self.connected = False
            logger.info("OPC UA connection closed")

    async def read_registers(self, address: int, count: int) -> List[int]:
        """Read OPC UA nodes (mapped as registers)."""
        if not self.connected:
            await self.connect()

        if not OPCUA_AVAILABLE:
            return [3200 + i for i in range(count)]

        values = []
        for i in range(count):
            node_id = f"{self.config.namespace};i={address + i}"
            node = self.client.get_node(node_id)
            value = await node.read_value()
            values.append(int(value))

        return values

    async def write_register(self, address: int, value: int) -> bool:
        """Write OPC UA node."""
        if not self.connected:
            await self.connect()

        if not OPCUA_AVAILABLE:
            return True

        try:
            node_id = f"{self.config.namespace};i={address}"
            node = self.client.get_node(node_id)
            await node.write_value(ua.DataValue(ua.Variant(value, ua.VariantType.Int32)))
            return True
        except Exception as e:
            logger.error(f"OPC UA write failed: {e}")
            return False


class HeatExchangerConnector:
    """Main heat exchanger connector with monitoring and fouling detection."""

    # Modbus register mappings
    REGISTERS = {
        'hot_inlet_temp': 40001,      # 2 registers, float32
        'hot_outlet_temp': 40003,     # 2 registers, float32
        'cold_inlet_temp': 40005,     # 2 registers, float32
        'cold_outlet_temp': 40007,    # 2 registers, float32
        'hot_flow_rate': 40009,       # 2 registers, float32
        'cold_flow_rate': 40011,      # 2 registers, float32
        'pressure_hot_in': 40013,     # 2 registers, float32
        'pressure_hot_out': 40015,    # 2 registers, float32
        'pressure_cold_in': 40017,    # 2 registers, float32
        'pressure_cold_out': 40019,   # 2 registers, float32
        'status': 40021,               # 1 register, uint16
        'alarm_bits': 40022,          # 1 register, uint16
    }

    def __init__(self, config: ConnectionConfig, mock_mode: bool = False):
        """Initialize heat exchanger connector."""
        self.config = config
        self.mock_mode = mock_mode

        # Create protocol handler
        if config.protocol in [ProtocolType.MODBUS_TCP, ProtocolType.MODBUS_RTU]:
            self.handler = ModbusHandler(config)
        else:
            self.handler = OPCUAHandler(config)

        # Performance tracking
        self.baseline_effectiveness: Optional[float] = None
        self.fouling_history: List[Tuple[datetime, float]] = []
        self.performance_history: List[HeatExchangerMetrics] = []

        # Connection pool for concurrent operations
        self.connection_pool = []
        self.pool_size = 5
        self._pool_lock = asyncio.Lock()

    async def initialize(self):
        """Initialize connector and establish baseline."""
        logger.info("Initializing heat exchanger connector")

        # Connect to device
        connected = await self.handler.circuit_breaker.call(self.handler.connect)
        if not connected:
            raise Exception("Failed to initialize heat exchanger connection")

        # Establish baseline performance
        metrics = await self.get_metrics()
        if metrics:
            self.baseline_effectiveness = metrics.effectiveness
            logger.info(f"Baseline effectiveness established: {self.baseline_effectiveness:.2%}")

    async def get_metrics(self) -> Optional[HeatExchangerMetrics]:
        """Get current heat exchanger metrics."""
        try:
            # Read all registers in one batch for efficiency
            data = await self.handler.circuit_breaker.call(
                self.handler.read_registers,
                self.REGISTERS['hot_inlet_temp'] - 40001,
                22  # Total registers to read
            )

            # Parse float values (2 registers each)
            hot_inlet = self._registers_to_float(data[0:2])
            hot_outlet = self._registers_to_float(data[2:4])
            cold_inlet = self._registers_to_float(data[4:6])
            cold_outlet = self._registers_to_float(data[6:8])
            hot_flow = self._registers_to_float(data[8:10])
            cold_flow = self._registers_to_float(data[10:12])
            pressure_hot_in = self._registers_to_float(data[12:14])
            pressure_hot_out = self._registers_to_float(data[14:16])
            pressure_cold_in = self._registers_to_float(data[16:18])
            pressure_cold_out = self._registers_to_float(data[18:20])
            status_code = data[20]
            alarm_bits = data[21]

            # Calculate derived metrics
            effectiveness = self._calculate_effectiveness(
                hot_inlet, hot_outlet, cold_inlet, cold_outlet,
                hot_flow, cold_flow
            )

            heat_transfer = self._calculate_heat_transfer(
                hot_inlet, hot_outlet, hot_flow
            )

            fouling_factor = self._calculate_fouling_factor(
                effectiveness, heat_transfer
            )

            # Determine status
            status = self._determine_status(status_code, fouling_factor, effectiveness)
            warnings = self._check_warnings(alarm_bits, fouling_factor, effectiveness)

            metrics = HeatExchangerMetrics(
                timestamp=DeterministicClock.now(),
                hot_inlet_temp=hot_inlet,
                hot_outlet_temp=hot_outlet,
                cold_inlet_temp=cold_inlet,
                cold_outlet_temp=cold_outlet,
                hot_flow_rate=hot_flow,
                cold_flow_rate=cold_flow,
                effectiveness=effectiveness,
                heat_transfer_rate=heat_transfer,
                pressure_drop_hot=pressure_hot_out - pressure_hot_in,
                pressure_drop_cold=pressure_cold_out - pressure_cold_in,
                fouling_factor=fouling_factor,
                status=status,
                warnings=warnings
            )

            # Update history
            self.performance_history.append(metrics)
            if len(self.performance_history) > 1000:
                self.performance_history.pop(0)

            self.fouling_history.append((DeterministicClock.now(), fouling_factor))
            if len(self.fouling_history) > 1000:
                self.fouling_history.pop(0)

            return metrics

        except Exception as e:
            logger.error(f"Failed to get heat exchanger metrics: {e}")
            return None

    def _registers_to_float(self, registers: List[int]) -> float:
        """Convert two Modbus registers to float32."""
        if self.mock_mode:
            return float(registers[0]) / 100.0

        # Combine registers and convert to float
        combined = (registers[0] << 16) | registers[1]
        bytes_data = struct.pack('>I', combined)
        return struct.unpack('>f', bytes_data)[0]

    def _calculate_effectiveness(self, Th_in: float, Th_out: float,
                                Tc_in: float, Tc_out: float,
                                mh: float, mc: float) -> float:
        """Calculate heat exchanger effectiveness (NTU method)."""
        if Th_in <= Tc_in:
            return 0.0

        # Heat capacities (assuming water, Cp = 4.18 kJ/kg·K)
        Cp = 4.18
        Ch = mh * Cp
        Cc = mc * Cp
        Cmin = min(Ch, Cc)
        Cmax = max(Ch, Cc)

        # Actual heat transfer
        Qh = Ch * (Th_in - Th_out)
        Qc = Cc * (Tc_out - Tc_in)
        Q_actual = (Qh + Qc) / 2  # Average for accuracy

        # Maximum possible heat transfer
        Q_max = Cmin * (Th_in - Tc_in)

        if Q_max == 0:
            return 0.0

        effectiveness = Q_actual / Q_max
        return min(max(effectiveness, 0.0), 1.0)

    def _calculate_heat_transfer(self, T_in: float, T_out: float, flow_rate: float) -> float:
        """Calculate heat transfer rate in kW."""
        Cp = 4.18  # kJ/kg·K for water
        return flow_rate * Cp * abs(T_in - T_out)

    def _calculate_fouling_factor(self, effectiveness: float, heat_transfer: float) -> float:
        """Calculate fouling factor based on performance degradation."""
        if not self.baseline_effectiveness or self.baseline_effectiveness == 0:
            return 0.0

        # Fouling resistance estimation (simplified)
        degradation = 1.0 - (effectiveness / self.baseline_effectiveness)
        fouling_factor = degradation * 0.001  # Convert to m²K/W scale

        return max(fouling_factor, 0.0)

    def _determine_status(self, status_code: int, fouling: float, effectiveness: float) -> HeatExchangerStatus:
        """Determine operational status."""
        if status_code == 0:
            return HeatExchangerStatus.OFFLINE

        if fouling > 0.0005:  # High fouling threshold
            return HeatExchangerStatus.FOULED

        if self.baseline_effectiveness and effectiveness < self.baseline_effectiveness * 0.8:
            return HeatExchangerStatus.DEGRADED

        if status_code > 100:
            return HeatExchangerStatus.FAULT

        return HeatExchangerStatus.ONLINE

    def _check_warnings(self, alarm_bits: int, fouling: float, effectiveness: float) -> List[str]:
        """Check for warning conditions."""
        warnings = []

        if alarm_bits & 0x01:
            warnings.append("High temperature differential")
        if alarm_bits & 0x02:
            warnings.append("Low flow rate detected")
        if alarm_bits & 0x04:
            warnings.append("Pressure drop exceeds limit")

        if fouling > 0.0003:
            warnings.append(f"Fouling detected: {fouling:.5f} m²K/W")

        if self.baseline_effectiveness and effectiveness < self.baseline_effectiveness * 0.9:
            warnings.append(f"Performance degraded: {(1 - effectiveness/self.baseline_effectiveness)*100:.1f}%")

        return warnings

    async def detect_fouling_trend(self) -> Dict[str, Any]:
        """Analyze fouling trend over time."""
        if len(self.fouling_history) < 10:
            return {"trend": "insufficient_data"}

        # Extract time series
        times = [(t - self.fouling_history[0][0]).total_seconds() / 3600
                for t, _ in self.fouling_history]
        values = [f for _, f in self.fouling_history]

        # Calculate trend (simple linear regression)
        if len(times) > 1:
            coefficients = np.polyfit(times, values, 1)
            trend_rate = coefficients[0]  # Fouling rate per hour

            # Predict time to cleaning threshold
            current_fouling = values[-1]
            cleaning_threshold = 0.0008

            if trend_rate > 0:
                hours_to_cleaning = (cleaning_threshold - current_fouling) / trend_rate
            else:
                hours_to_cleaning = float('inf')

            return {
                "trend": "increasing" if trend_rate > 0 else "stable",
                "fouling_rate": trend_rate,
                "current_fouling": current_fouling,
                "hours_to_cleaning": hours_to_cleaning,
                "recommendation": self._get_cleaning_recommendation(hours_to_cleaning)
            }

        return {"trend": "stable"}

    def _get_cleaning_recommendation(self, hours_to_cleaning: float) -> str:
        """Get maintenance recommendation based on fouling trend."""
        if hours_to_cleaning < 24:
            return "URGENT: Schedule cleaning within 24 hours"
        elif hours_to_cleaning < 168:  # 1 week
            return "Schedule cleaning within the week"
        elif hours_to_cleaning < 720:  # 1 month
            return "Plan cleaning in next maintenance window"
        else:
            return "Normal operation, monitor fouling trend"

    async def close(self):
        """Close all connections."""
        await self.handler.disconnect()
        logger.info("Heat exchanger connector closed")


# Example usage and testing
async def main():
    """Example usage of heat exchanger connector."""
    config = ConnectionConfig(
        protocol=ProtocolType.MODBUS_TCP,
        host="192.168.1.100",
        port=502,
        unit_id=1,
        timeout=10.0,
        retry_count=3,
        retry_delay=1.0
    )

    connector = HeatExchangerConnector(config, mock_mode=True)

    try:
        await connector.initialize()

        # Monitor heat exchanger
        for _ in range(5):
            metrics = await connector.get_metrics()
            if metrics:
                logger.info(f"Heat Exchanger Status: {metrics.status.value}")
                logger.info(f"Effectiveness: {metrics.effectiveness:.2%}")
                logger.info(f"Heat Transfer: {metrics.heat_transfer_rate:.2f} kW")
                logger.info(f"Fouling Factor: {metrics.fouling_factor:.5f} m²K/W")

                if metrics.warnings:
                    for warning in metrics.warnings:
                        logger.warning(warning)

            await asyncio.sleep(5)

        # Check fouling trend
        trend = await connector.detect_fouling_trend()
        logger.info(f"Fouling trend: {trend}")

    finally:
        await connector.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())