# -*- coding: utf-8 -*-
"""
Preheater Connector for GL-006 HeatRecoveryMaximizer

Manages air and gas preheater systems with Modbus control,
damper management, leakage detection, and heat recovery optimization.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import struct
from greenlang.determinism import DeterministicClock

try:
    from pymodbus.client import AsyncModbusTcpClient, AsyncModbusSerialClient
    MODBUS_AVAILABLE = True
except ImportError:
    MODBUS_AVAILABLE = False

logger = logging.getLogger(__name__)


class PreheaterType(Enum):
    """Types of preheater systems."""
    ROTARY_REGENERATIVE = "rotary_regenerative"
    TUBULAR = "tubular"
    PLATE = "plate"
    HEAT_PIPE = "heat_pipe"


class DamperPosition(Enum):
    """Damper control positions."""
    CLOSED = 0
    QUARTER = 25
    HALF = 50
    THREE_QUARTER = 75
    FULL = 100


class PreheaterStatus(Enum):
    """Preheater operational status."""
    OPERATING = "operating"
    WARMING_UP = "warming_up"
    STANDBY = "standby"
    LEAKAGE_DETECTED = "leakage_detected"
    MAINTENANCE = "maintenance"
    FAULT = "fault"
    OFFLINE = "offline"


@dataclass
class PreheaterMetrics:
    """Real-time preheater performance metrics."""
    timestamp: datetime
    air_inlet_temp: float  # °C
    air_outlet_temp: float  # °C
    gas_inlet_temp: float  # °C
    gas_outlet_temp: float  # °C
    air_flow_rate: float  # m³/h
    gas_flow_rate: float  # m³/h
    air_pressure_in: float  # Pa
    air_pressure_out: float  # Pa
    gas_pressure_in: float  # Pa
    gas_pressure_out: float  # Pa
    heat_transfer_rate: float  # kW
    effectiveness: float  # %
    leakage_rate: float  # %
    damper_position: DamperPosition
    rotation_speed: float  # RPM (for rotary types)
    status: PreheaterStatus
    warnings: List[str] = field(default_factory=list)


@dataclass
class PreheaterConfig:
    """Preheater connection and control configuration."""
    # Modbus settings
    host: str = "192.168.1.101"
    port: int = 502
    serial_port: Optional[str] = None
    baudrate: int = 9600
    unit_id: int = 1
    timeout: float = 10.0

    # Preheater specs
    preheater_type: PreheaterType = PreheaterType.ROTARY_REGENERATIVE
    design_heat_recovery: float = 500.0  # kW
    max_air_temp: float = 350.0  # °C
    max_gas_temp: float = 400.0  # °C

    # Control parameters
    target_outlet_temp: float = 150.0  # °C
    max_leakage_rate: float = 5.0  # %
    damper_control_enabled: bool = True
    rotation_speed_rpm: float = 3.0  # For rotary types

    # Monitoring
    sample_interval: float = 5.0
    leakage_check_interval: float = 60.0


class LeakageDetector:
    """Detects air-to-gas leakage in preheater."""

    def __init__(self, max_leakage: float = 5.0):
        self.max_leakage = max_leakage
        self.baseline_o2: Optional[float] = None
        self.leakage_history: List[Tuple[datetime, float]] = []

    def calculate_leakage(self, o2_inlet: float, o2_outlet: float,
                         air_flow: float, gas_flow: float) -> float:
        """Calculate leakage rate based on O2 measurements."""
        if o2_outlet <= o2_inlet:
            return 0.0

        # Leakage estimation based on oxygen increase
        o2_increase = o2_outlet - o2_inlet
        leakage_ratio = o2_increase / (21.0 - o2_inlet)  # 21% O2 in air

        # Consider flow rates
        leakage_rate = leakage_ratio * (air_flow / gas_flow) * 100

        # Store in history
        self.leakage_history.append((DeterministicClock.now(), leakage_rate))
        if len(self.leakage_history) > 100:
            self.leakage_history.pop(0)

        return min(leakage_rate, 100.0)

    def is_excessive_leakage(self, current_leakage: float) -> bool:
        """Check if leakage exceeds acceptable limits."""
        return current_leakage > self.max_leakage

    def get_leakage_trend(self) -> str:
        """Analyze leakage trend over time."""
        if len(self.leakage_history) < 10:
            return "insufficient_data"

        recent = [l for _, l in self.leakage_history[-10:]]
        older = [l for _, l in self.leakage_history[-20:-10]]

        if not older:
            return "stable"

        recent_avg = sum(recent) / len(recent)
        older_avg = sum(older) / len(older)

        if recent_avg > older_avg * 1.2:
            return "increasing"
        elif recent_avg < older_avg * 0.8:
            return "decreasing"
        else:
            return "stable"


class DamperController:
    """Controls preheater damper positions."""

    def __init__(self):
        self.current_position = DamperPosition.HALF
        self.target_position = DamperPosition.HALF
        self.last_adjustment = DeterministicClock.now()
        self.adjustment_rate = 5  # % per second

    async def set_position(self, position: DamperPosition) -> bool:
        """Set damper to target position."""
        self.target_position = position
        steps = abs(position.value - self.current_position.value) // self.adjustment_rate

        for _ in range(int(steps)):
            if self.current_position.value < position.value:
                new_value = min(self.current_position.value + self.adjustment_rate, 100)
            else:
                new_value = max(self.current_position.value - self.adjustment_rate, 0)

            self.current_position = DamperPosition(new_value)
            await asyncio.sleep(1.0)  # Gradual adjustment

        self.current_position = position
        self.last_adjustment = DeterministicClock.now()
        logger.info(f"Damper moved to {position.value}% position")
        return True

    def calculate_optimal_position(self, target_temp: float, current_temp: float,
                                  flow_rate: float) -> DamperPosition:
        """Calculate optimal damper position for target temperature."""
        temp_error = target_temp - current_temp

        # PID-like control logic (simplified)
        if abs(temp_error) < 5:
            return self.current_position

        if temp_error > 20:
            return DamperPosition.FULL
        elif temp_error > 10:
            return DamperPosition.THREE_QUARTER
        elif temp_error > 0:
            return DamperPosition.HALF
        elif temp_error > -10:
            return DamperPosition.QUARTER
        else:
            return DamperPosition.CLOSED


class ModbusPreheaterClient:
    """Modbus client for preheater communication."""

    # Modbus register mappings
    REGISTERS = {
        'air_inlet_temp': 40001,       # Float32 (2 registers)
        'air_outlet_temp': 40003,
        'gas_inlet_temp': 40005,
        'gas_outlet_temp': 40007,
        'air_flow': 40009,
        'gas_flow': 40011,
        'air_pressure_in': 40013,
        'air_pressure_out': 40015,
        'gas_pressure_in': 40017,
        'gas_pressure_out': 40019,
        'o2_inlet': 40021,             # For leakage detection
        'o2_outlet': 40023,
        'rotation_speed': 40025,        # For rotary types
        'damper_position': 40027,       # UInt16
        'status': 40028,                # UInt16
        'alarm_bits': 40029,            # UInt16
    }

    # Control registers (writable)
    CONTROL_REGISTERS = {
        'set_damper_position': 40100,   # UInt16 (0-100)
        'set_rotation_speed': 40101,    # Float32 (2 registers)
        'control_mode': 40103,          # UInt16
    }

    def __init__(self, config: PreheaterConfig):
        self.config = config
        self.client = None
        self.connected = False

    async def connect(self) -> bool:
        """Establish Modbus connection."""
        try:
            if not MODBUS_AVAILABLE:
                logger.info("Mock Modbus connection (libraries not available)")
                self.connected = True
                return True

            if self.config.serial_port:
                # Modbus RTU
                self.client = AsyncModbusSerialClient(
                    port=self.config.serial_port,
                    baudrate=self.config.baudrate,
                    timeout=self.config.timeout
                )
            else:
                # Modbus TCP
                self.client = AsyncModbusTcpClient(
                    self.config.host,
                    port=self.config.port,
                    timeout=self.config.timeout
                )

            await self.client.connect()
            self.connected = True
            logger.info(f"Connected to preheater via Modbus")
            return True

        except Exception as e:
            logger.error(f"Modbus connection failed: {e}")
            return False

    async def read_all_data(self) -> Dict[str, Any]:
        """Read all preheater data points."""
        if not self.connected:
            await self.connect()

        if not MODBUS_AVAILABLE:
            return self._get_mock_data()

        try:
            # Read measurement registers (29 registers total)
            response = await self.client.read_holding_registers(
                address=self.REGISTERS['air_inlet_temp'] - 40001,
                count=29,
                slave=self.config.unit_id
            )

            if response.isError():
                raise Exception(f"Modbus read error: {response}")

            # Parse float values
            data = {}
            float_indices = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24]
            float_names = ['air_inlet_temp', 'air_outlet_temp', 'gas_inlet_temp',
                          'gas_outlet_temp', 'air_flow', 'gas_flow',
                          'air_pressure_in', 'air_pressure_out',
                          'gas_pressure_in', 'gas_pressure_out',
                          'o2_inlet', 'o2_outlet', 'rotation_speed']

            for idx, name in zip(float_indices, float_names):
                data[name] = self._registers_to_float(response.registers[idx:idx+2])

            # Parse integer values
            data['damper_position'] = response.registers[26]
            data['status'] = response.registers[27]
            data['alarms'] = response.registers[28]

            return data

        except Exception as e:
            logger.error(f"Failed to read preheater data: {e}")
            return {}

    async def write_damper_position(self, position: int) -> bool:
        """Write damper position setpoint (0-100)."""
        if not MODBUS_AVAILABLE:
            return True

        try:
            response = await self.client.write_register(
                address=self.CONTROL_REGISTERS['set_damper_position'] - 40001,
                value=max(0, min(100, position)),
                slave=self.config.unit_id
            )
            return not response.isError()

        except Exception as e:
            logger.error(f"Failed to write damper position: {e}")
            return False

    def _registers_to_float(self, registers: List[int]) -> float:
        """Convert Modbus registers to float32."""
        combined = (registers[0] << 16) | registers[1]
        bytes_data = struct.pack('>I', combined)
        return struct.unpack('>f', bytes_data)[0]

    def _get_mock_data(self) -> Dict[str, Any]:
        """Generate mock data for testing."""
        import random
        return {
            'air_inlet_temp': 30.0 + random.uniform(-5, 5),
            'air_outlet_temp': 150.0 + random.uniform(-10, 10),
            'gas_inlet_temp': 350.0 + random.uniform(-10, 10),
            'gas_outlet_temp': 180.0 + random.uniform(-5, 5),
            'air_flow': 10000.0 + random.uniform(-500, 500),
            'gas_flow': 8000.0 + random.uniform(-400, 400),
            'air_pressure_in': 1000.0,
            'air_pressure_out': 950.0,
            'gas_pressure_in': 800.0,
            'gas_pressure_out': 750.0,
            'o2_inlet': 3.0 + random.uniform(-0.5, 0.5),
            'o2_outlet': 3.5 + random.uniform(-0.5, 0.5),
            'rotation_speed': 3.0,
            'damper_position': 50,
            'status': 1,
            'alarms': 0
        }

    async def disconnect(self):
        """Close Modbus connection."""
        if self.client:
            self.client.close()
            self.connected = False


class PreheaterConnector:
    """Main preheater connector with control and monitoring."""

    def __init__(self, config: PreheaterConfig):
        self.config = config
        self.modbus_client = ModbusPreheaterClient(config)
        self.damper_controller = DamperController()
        self.leakage_detector = LeakageDetector(config.max_leakage_rate)
        self.metrics_history: List[PreheaterMetrics] = []
        self._running = False

    async def initialize(self):
        """Initialize preheater connector."""
        logger.info("Initializing preheater connector")

        connected = await self.modbus_client.connect()
        if not connected:
            raise Exception("Failed to connect to preheater")

        # Set initial damper position
        if self.config.damper_control_enabled:
            await self.damper_controller.set_position(DamperPosition.HALF)

        self._running = True
        logger.info("Preheater connector initialized")

    async def get_current_metrics(self) -> Optional[PreheaterMetrics]:
        """Get current preheater performance metrics."""
        data = await self.modbus_client.read_all_data()

        if not data:
            return None

        # Calculate performance metrics
        heat_transfer = self._calculate_heat_transfer(data)
        effectiveness = self._calculate_effectiveness(data)
        leakage = self.leakage_detector.calculate_leakage(
            data['o2_inlet'], data['o2_outlet'],
            data['air_flow'], data['gas_flow']
        )

        # Determine status
        status = self._determine_status(data, leakage)
        warnings = self._check_warnings(data, leakage, effectiveness)

        metrics = PreheaterMetrics(
            timestamp=DeterministicClock.now(),
            air_inlet_temp=data['air_inlet_temp'],
            air_outlet_temp=data['air_outlet_temp'],
            gas_inlet_temp=data['gas_inlet_temp'],
            gas_outlet_temp=data['gas_outlet_temp'],
            air_flow_rate=data['air_flow'],
            gas_flow_rate=data['gas_flow'],
            air_pressure_in=data['air_pressure_in'],
            air_pressure_out=data['air_pressure_out'],
            gas_pressure_in=data['gas_pressure_in'],
            gas_pressure_out=data['gas_pressure_out'],
            heat_transfer_rate=heat_transfer,
            effectiveness=effectiveness,
            leakage_rate=leakage,
            damper_position=DamperPosition(data['damper_position']),
            rotation_speed=data['rotation_speed'],
            status=status,
            warnings=warnings
        )

        # Store metrics
        self.metrics_history.append(metrics)
        if len(self.metrics_history) > 1000:
            self.metrics_history.pop(0)

        return metrics

    def _calculate_heat_transfer(self, data: Dict) -> float:
        """Calculate heat transfer rate in kW."""
        # Using air side calculation
        air_density = 1.2  # kg/m³ at standard conditions
        cp_air = 1.005  # kJ/kg·K

        air_mass_flow = data['air_flow'] * air_density / 3600  # kg/s
        delta_t = data['air_outlet_temp'] - data['air_inlet_temp']

        return air_mass_flow * cp_air * delta_t

    def _calculate_effectiveness(self, data: Dict) -> float:
        """Calculate heat exchanger effectiveness."""
        # ε = (T_cold_out - T_cold_in) / (T_hot_in - T_cold_in)
        if data['gas_inlet_temp'] <= data['air_inlet_temp']:
            return 0.0

        actual_heat_gain = data['air_outlet_temp'] - data['air_inlet_temp']
        max_heat_gain = data['gas_inlet_temp'] - data['air_inlet_temp']

        if max_heat_gain == 0:
            return 0.0

        effectiveness = (actual_heat_gain / max_heat_gain) * 100
        return min(max(effectiveness, 0.0), 100.0)

    def _determine_status(self, data: Dict, leakage: float) -> PreheaterStatus:
        """Determine preheater operational status."""
        status_code = data['status']

        if status_code == 0:
            return PreheaterStatus.OFFLINE
        elif leakage > self.config.max_leakage_rate:
            return PreheaterStatus.LEAKAGE_DETECTED
        elif status_code == 1:
            return PreheaterStatus.OPERATING
        elif status_code == 2:
            return PreheaterStatus.WARMING_UP
        elif status_code == 3:
            return PreheaterStatus.STANDBY
        elif status_code == 4:
            return PreheaterStatus.MAINTENANCE
        else:
            return PreheaterStatus.FAULT

    def _check_warnings(self, data: Dict, leakage: float, effectiveness: float) -> List[str]:
        """Check for warning conditions."""
        warnings = []

        if leakage > self.config.max_leakage_rate * 0.8:
            warnings.append(f"High leakage rate: {leakage:.1f}%")

        if effectiveness < 70:
            warnings.append(f"Low effectiveness: {effectiveness:.1f}%")

        if data['air_outlet_temp'] > self.config.max_air_temp * 0.9:
            warnings.append(f"High air outlet temperature: {data['air_outlet_temp']:.1f}°C")

        pressure_drop_air = data['air_pressure_in'] - data['air_pressure_out']
        if pressure_drop_air > 200:
            warnings.append(f"High air-side pressure drop: {pressure_drop_air:.0f} Pa")

        return warnings

    async def control_loop(self):
        """Automatic control loop for optimization."""
        logger.info("Starting preheater control loop")

        while self._running:
            try:
                metrics = await self.get_current_metrics()

                if metrics and self.config.damper_control_enabled:
                    # Calculate optimal damper position
                    optimal_position = self.damper_controller.calculate_optimal_position(
                        self.config.target_outlet_temp,
                        metrics.air_outlet_temp,
                        metrics.air_flow_rate
                    )

                    # Adjust if needed
                    if optimal_position != self.damper_controller.current_position:
                        await self.modbus_client.write_damper_position(optimal_position.value)
                        await self.damper_controller.set_position(optimal_position)

                    # Log performance
                    logger.info(f"Preheater effectiveness: {metrics.effectiveness:.1f}%")
                    logger.info(f"Heat recovery: {metrics.heat_transfer_rate:.1f} kW")
                    logger.info(f"Leakage rate: {metrics.leakage_rate:.1f}%")

                await asyncio.sleep(self.config.sample_interval)

            except Exception as e:
                logger.error(f"Control loop error: {e}")
                await asyncio.sleep(self.config.sample_interval)

    async def shutdown(self):
        """Shutdown preheater connector."""
        logger.info("Shutting down preheater connector")
        self._running = False

        # Move damper to safe position
        if self.config.damper_control_enabled:
            await self.damper_controller.set_position(DamperPosition.HALF)

        await self.modbus_client.disconnect()


# Example usage
async def main():
    """Example preheater control."""
    config = PreheaterConfig(
        host="192.168.1.101",
        port=502,
        preheater_type=PreheaterType.ROTARY_REGENERATIVE
    )

    connector = PreheaterConnector(config)

    try:
        await connector.initialize()
        control_task = asyncio.create_task(connector.control_loop())
        await asyncio.sleep(30)

    finally:
        await connector.shutdown()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())