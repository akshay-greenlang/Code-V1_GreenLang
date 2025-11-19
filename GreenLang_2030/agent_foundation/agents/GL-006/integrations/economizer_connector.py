"""
Economizer Connector for GL-006 HeatRecoveryMaximizer

Monitors boiler economizer performance via Modbus TCP and OPC UA,
tracking flue gas heat recovery, feedwater preheating, and fouling.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import struct
import statistics

# Third-party imports (would be actual in production)
try:
    from pymodbus.client import AsyncModbusTcpClient
    from asyncua import Client as OPCClient, ua
    PROTOCOL_LIBS_AVAILABLE = True
except ImportError:
    PROTOCOL_LIBS_AVAILABLE = False

logger = logging.getLogger(__name__)


class EconomizerType(Enum):
    """Types of economizer systems."""
    CONDENSING = "condensing"
    NON_CONDENSING = "non_condensing"
    FEEDWATER = "feedwater"
    AIR_PREHEATER = "air_preheater"


class EconomizerStatus(Enum):
    """Economizer operational status."""
    RUNNING = "running"
    STANDBY = "standby"
    FOULED = "fouled"
    MAINTENANCE = "maintenance"
    FAULT = "fault"
    OFFLINE = "offline"


@dataclass
class EconomizerMetrics:
    """Real-time economizer performance metrics."""
    timestamp: datetime
    flue_gas_inlet_temp: float  # °C
    flue_gas_outlet_temp: float  # °C
    feedwater_inlet_temp: float  # °C
    feedwater_outlet_temp: float  # °C
    flue_gas_flow: float  # m³/h
    feedwater_flow: float  # kg/h
    stack_temp: float  # °C
    heat_recovered: float  # kW
    efficiency: float  # %
    gas_side_pressure_drop: float  # Pa
    water_side_pressure_drop: float  # kPa
    fouling_index: float  # 0-100
    condensate_flow: float  # L/h (for condensing types)
    sox_concentration: float  # ppm
    nox_concentration: float  # ppm
    status: EconomizerStatus
    alerts: List[str] = field(default_factory=list)


@dataclass
class EconomizerConfig:
    """Economizer connection and operational configuration."""
    # Connection settings
    protocol: str = "modbus_tcp"
    host: str = "192.168.1.100"
    port: int = 502
    unit_id: int = 1
    timeout: float = 10.0

    # OPC UA settings
    opc_endpoint: Optional[str] = None
    opc_namespace: str = "ns=2"

    # Economizer specifications
    economizer_type: EconomizerType = EconomizerType.NON_CONDENSING
    design_capacity: float = 1000.0  # kW
    min_stack_temp: float = 120.0  # °C (to avoid condensation)
    max_pressure_drop: float = 500.0  # Pa

    # Performance thresholds
    fouling_threshold: float = 70.0  # Fouling index
    efficiency_threshold: float = 85.0  # %
    cleaning_interval_hours: int = 2160  # 90 days

    # Monitoring settings
    sample_interval: float = 5.0  # seconds
    history_size: int = 1000
    alert_repeat_interval: int = 3600  # seconds


class PerformanceMonitor:
    """Monitors economizer performance degradation."""

    def __init__(self, baseline_efficiency: float = 90.0):
        self.baseline_efficiency = baseline_efficiency
        self.efficiency_history: List[Tuple[datetime, float]] = []
        self.fouling_history: List[Tuple[datetime, float]] = []
        self.last_cleaning: datetime = datetime.now()

    def update(self, efficiency: float, fouling_index: float):
        """Update performance history."""
        now = datetime.now()
        self.efficiency_history.append((now, efficiency))
        self.fouling_history.append((now, fouling_index))

        # Maintain history size
        if len(self.efficiency_history) > 1000:
            self.efficiency_history.pop(0)
        if len(self.fouling_history) > 1000:
            self.fouling_history.pop(0)

    def get_degradation_rate(self) -> float:
        """Calculate performance degradation rate per day."""
        if len(self.efficiency_history) < 10:
            return 0.0

        # Get data from last 24 hours
        cutoff = datetime.now() - timedelta(hours=24)
        recent_data = [(t, e) for t, e in self.efficiency_history if t > cutoff]

        if len(recent_data) < 2:
            return 0.0

        # Calculate trend
        start_eff = recent_data[0][1]
        end_eff = recent_data[-1][1]
        time_diff = (recent_data[-1][0] - recent_data[0][0]).total_seconds() / 86400

        if time_diff == 0:
            return 0.0

        return (start_eff - end_eff) / time_diff

    def predict_cleaning_needed(self) -> Tuple[bool, int]:
        """Predict when cleaning will be needed."""
        if not self.fouling_history:
            return False, -1

        current_fouling = self.fouling_history[-1][1]

        if current_fouling > 80:
            return True, 0  # Immediate cleaning needed

        # Estimate based on fouling trend
        if len(self.fouling_history) > 10:
            recent_foulings = [f for _, f in self.fouling_history[-10:]]
            fouling_rate = statistics.mean(
                recent_foulings[i] - recent_foulings[i-1]
                for i in range(1, len(recent_foulings))
            )

            if fouling_rate > 0:
                days_to_threshold = (80 - current_fouling) / fouling_rate
                return False, int(days_to_threshold)

        return False, -1


class ModbusEconomizerClient:
    """Modbus client for economizer communication."""

    # Register map for economizer data
    REGISTER_MAP = {
        'flue_gas_inlet_temp': 40001,    # Float32 (2 registers)
        'flue_gas_outlet_temp': 40003,
        'feedwater_inlet_temp': 40005,
        'feedwater_outlet_temp': 40007,
        'flue_gas_flow': 40009,
        'feedwater_flow': 40011,
        'stack_temp': 40013,
        'gas_pressure_in': 40015,
        'gas_pressure_out': 40017,
        'water_pressure_in': 40019,
        'water_pressure_out': 40021,
        'condensate_flow': 40023,
        'sox_ppm': 40025,
        'nox_ppm': 40027,
        'status_register': 40029,        # UInt16
        'alarm_register': 40030,         # UInt16
        'fouling_index': 40031,          # UInt16 (0-100)
    }

    def __init__(self, config: EconomizerConfig):
        self.config = config
        self.client = None
        self.connected = False
        self.retry_count = 0
        self.max_retries = 3

    async def connect(self) -> bool:
        """Establish Modbus connection."""
        try:
            if not PROTOCOL_LIBS_AVAILABLE:
                logger.info("Using mock Modbus connection")
                self.connected = True
                return True

            self.client = AsyncModbusTcpClient(
                self.config.host,
                port=self.config.port,
                timeout=self.config.timeout
            )

            await self.client.connect()
            self.connected = self.client.connected

            if self.connected:
                logger.info(f"Connected to economizer at {self.config.host}:{self.config.port}")
                self.retry_count = 0

            return self.connected

        except Exception as e:
            logger.error(f"Modbus connection failed: {e}")
            self.retry_count += 1
            return False

    async def read_metrics(self) -> Dict[str, Any]:
        """Read all economizer metrics."""
        if not self.connected:
            await self.connect()

        try:
            # Mock data for testing
            if not PROTOCOL_LIBS_AVAILABLE:
                return self._get_mock_data()

            # Read all float registers (28 registers for 14 float values)
            response = await self.client.read_holding_registers(
                address=self.REGISTER_MAP['flue_gas_inlet_temp'] - 40001,
                count=28,
                slave=self.config.unit_id
            )

            if response.isError():
                raise Exception(f"Modbus read error: {response}")

            # Convert register pairs to floats
            values = {}
            for i in range(0, 28, 2):
                float_val = self._registers_to_float(
                    response.registers[i:i+2]
                )
                values[f'value_{i//2}'] = float_val

            # Read status registers
            status_response = await self.client.read_holding_registers(
                address=self.REGISTER_MAP['status_register'] - 40001,
                count=3,
                slave=self.config.unit_id
            )

            return {
                'flue_gas_inlet_temp': values['value_0'],
                'flue_gas_outlet_temp': values['value_1'],
                'feedwater_inlet_temp': values['value_2'],
                'feedwater_outlet_temp': values['value_3'],
                'flue_gas_flow': values['value_4'],
                'feedwater_flow': values['value_5'],
                'stack_temp': values['value_6'],
                'gas_pressure_in': values['value_7'],
                'gas_pressure_out': values['value_8'],
                'water_pressure_in': values['value_9'],
                'water_pressure_out': values['value_10'],
                'condensate_flow': values['value_11'],
                'sox_ppm': values['value_12'],
                'nox_ppm': values['value_13'],
                'status': status_response.registers[0],
                'alarms': status_response.registers[1],
                'fouling_index': status_response.registers[2]
            }

        except Exception as e:
            logger.error(f"Failed to read economizer metrics: {e}")
            return {}

    def _registers_to_float(self, registers: List[int]) -> float:
        """Convert Modbus registers to float32."""
        combined = (registers[0] << 16) | registers[1]
        bytes_data = struct.pack('>I', combined)
        return struct.unpack('>f', bytes_data)[0]

    def _get_mock_data(self) -> Dict[str, Any]:
        """Generate mock data for testing."""
        import random
        return {
            'flue_gas_inlet_temp': 320.0 + random.uniform(-10, 10),
            'flue_gas_outlet_temp': 150.0 + random.uniform(-5, 5),
            'feedwater_inlet_temp': 60.0 + random.uniform(-2, 2),
            'feedwater_outlet_temp': 120.0 + random.uniform(-3, 3),
            'flue_gas_flow': 5000.0 + random.uniform(-100, 100),
            'feedwater_flow': 10000.0 + random.uniform(-200, 200),
            'stack_temp': 145.0 + random.uniform(-5, 5),
            'gas_pressure_in': 100.0,
            'gas_pressure_out': 80.0,
            'water_pressure_in': 300.0,
            'water_pressure_out': 295.0,
            'condensate_flow': 50.0 if random.random() > 0.5 else 0,
            'sox_ppm': 20.0 + random.uniform(-5, 5),
            'nox_ppm': 50.0 + random.uniform(-10, 10),
            'status': 1,
            'alarms': 0,
            'fouling_index': 30 + int(random.uniform(-5, 5))
        }

    async def disconnect(self):
        """Close Modbus connection."""
        if self.client:
            self.client.close()
            self.connected = False
            logger.info("Economizer Modbus connection closed")


class EconomizerConnector:
    """Main economizer connector with monitoring and optimization."""

    def __init__(self, config: EconomizerConfig):
        self.config = config
        self.modbus_client = ModbusEconomizerClient(config)
        self.performance_monitor = PerformanceMonitor()
        self.metrics_history: List[EconomizerMetrics] = []
        self.last_alert_time: Dict[str, datetime] = {}
        self._running = False

    async def initialize(self):
        """Initialize economizer connector."""
        logger.info("Initializing economizer connector")

        # Connect to economizer
        connected = await self.modbus_client.connect()
        if not connected:
            raise Exception("Failed to connect to economizer")

        # Get initial metrics for baseline
        metrics = await self.get_current_metrics()
        if metrics:
            self.performance_monitor.baseline_efficiency = metrics.efficiency
            logger.info(f"Baseline efficiency: {metrics.efficiency:.1f}%")

        self._running = True

    async def get_current_metrics(self) -> Optional[EconomizerMetrics]:
        """Get current economizer performance metrics."""
        data = await self.modbus_client.read_metrics()

        if not data:
            return None

        # Calculate derived metrics
        heat_recovered = self._calculate_heat_recovered(data)
        efficiency = self._calculate_efficiency(data, heat_recovered)
        status = self._determine_status(data)
        alerts = self._check_alerts(data, efficiency)

        metrics = EconomizerMetrics(
            timestamp=datetime.now(),
            flue_gas_inlet_temp=data['flue_gas_inlet_temp'],
            flue_gas_outlet_temp=data['flue_gas_outlet_temp'],
            feedwater_inlet_temp=data['feedwater_inlet_temp'],
            feedwater_outlet_temp=data['feedwater_outlet_temp'],
            flue_gas_flow=data['flue_gas_flow'],
            feedwater_flow=data['feedwater_flow'],
            stack_temp=data['stack_temp'],
            heat_recovered=heat_recovered,
            efficiency=efficiency,
            gas_side_pressure_drop=data['gas_pressure_in'] - data['gas_pressure_out'],
            water_side_pressure_drop=data['water_pressure_in'] - data['water_pressure_out'],
            fouling_index=data['fouling_index'],
            condensate_flow=data['condensate_flow'],
            sox_concentration=data['sox_ppm'],
            nox_concentration=data['nox_ppm'],
            status=status,
            alerts=alerts
        )

        # Update monitoring
        self.performance_monitor.update(efficiency, data['fouling_index'])
        self.metrics_history.append(metrics)

        if len(self.metrics_history) > self.config.history_size:
            self.metrics_history.pop(0)

        return metrics

    def _calculate_heat_recovered(self, data: Dict) -> float:
        """Calculate heat recovered in kW."""
        # Q = m * Cp * ΔT
        feedwater_flow_kg_s = data['feedwater_flow'] / 3600  # Convert kg/h to kg/s
        cp_water = 4.18  # kJ/kg·K

        delta_t = data['feedwater_outlet_temp'] - data['feedwater_inlet_temp']
        heat_recovered = feedwater_flow_kg_s * cp_water * delta_t

        return heat_recovered

    def _calculate_efficiency(self, data: Dict, heat_recovered: float) -> float:
        """Calculate economizer efficiency."""
        # Efficiency based on stack temperature reduction
        flue_gas_delta = data['flue_gas_inlet_temp'] - data['flue_gas_outlet_temp']
        max_delta = data['flue_gas_inlet_temp'] - self.config.min_stack_temp

        if max_delta <= 0:
            return 0.0

        efficiency = (flue_gas_delta / max_delta) * 100

        # Adjust for fouling
        fouling_factor = 1 - (data['fouling_index'] / 200)
        efficiency *= fouling_factor

        return min(max(efficiency, 0.0), 100.0)

    def _determine_status(self, data: Dict) -> EconomizerStatus:
        """Determine economizer operational status."""
        status_code = data['status']

        if status_code == 0:
            return EconomizerStatus.OFFLINE
        elif status_code == 1:
            if data['fouling_index'] > self.config.fouling_threshold:
                return EconomizerStatus.FOULED
            return EconomizerStatus.RUNNING
        elif status_code == 2:
            return EconomizerStatus.STANDBY
        elif status_code == 3:
            return EconomizerStatus.MAINTENANCE
        else:
            return EconomizerStatus.FAULT

    def _check_alerts(self, data: Dict, efficiency: float) -> List[str]:
        """Check for alert conditions."""
        alerts = []
        now = datetime.now()

        # Stack temperature too low (condensation risk)
        if data['stack_temp'] < self.config.min_stack_temp:
            if self._should_alert('low_stack_temp', now):
                alerts.append(f"Stack temperature too low: {data['stack_temp']:.1f}°C")

        # High fouling
        if data['fouling_index'] > self.config.fouling_threshold:
            if self._should_alert('high_fouling', now):
                alerts.append(f"High fouling detected: {data['fouling_index']}/100")

        # Low efficiency
        if efficiency < self.config.efficiency_threshold:
            if self._should_alert('low_efficiency', now):
                alerts.append(f"Low efficiency: {efficiency:.1f}%")

        # High pressure drop
        gas_pressure_drop = data['gas_pressure_in'] - data['gas_pressure_out']
        if gas_pressure_drop > self.config.max_pressure_drop:
            if self._should_alert('high_pressure_drop', now):
                alerts.append(f"High gas-side pressure drop: {gas_pressure_drop:.0f} Pa")

        # Emissions check
        if data['sox_ppm'] > 100:
            alerts.append(f"High SOx emissions: {data['sox_ppm']:.1f} ppm")

        if data['nox_ppm'] > 200:
            alerts.append(f"High NOx emissions: {data['nox_ppm']:.1f} ppm")

        return alerts

    def _should_alert(self, alert_type: str, now: datetime) -> bool:
        """Check if alert should be raised based on repeat interval."""
        if alert_type not in self.last_alert_time:
            self.last_alert_time[alert_type] = now
            return True

        time_since_last = (now - self.last_alert_time[alert_type]).total_seconds()
        if time_since_last > self.config.alert_repeat_interval:
            self.last_alert_time[alert_type] = now
            return True

        return False

    async def monitor_performance(self):
        """Continuous performance monitoring loop."""
        logger.info("Starting economizer performance monitoring")

        while self._running:
            try:
                metrics = await self.get_current_metrics()

                if metrics:
                    # Log critical metrics
                    logger.info(f"Economizer efficiency: {metrics.efficiency:.1f}%")
                    logger.info(f"Heat recovered: {metrics.heat_recovered:.1f} kW")
                    logger.info(f"Fouling index: {metrics.fouling_index}/100")

                    # Check for alerts
                    if metrics.alerts:
                        for alert in metrics.alerts:
                            logger.warning(f"ALERT: {alert}")

                    # Check cleaning prediction
                    needs_cleaning, days = self.performance_monitor.predict_cleaning_needed()
                    if needs_cleaning:
                        logger.warning("Economizer requires immediate cleaning")
                    elif days > 0:
                        logger.info(f"Estimated {days} days until cleaning needed")

                await asyncio.sleep(self.config.sample_interval)

            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                await asyncio.sleep(self.config.sample_interval)

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary statistics."""
        if not self.metrics_history:
            return {}

        recent_metrics = self.metrics_history[-100:]  # Last 100 samples

        efficiencies = [m.efficiency for m in recent_metrics]
        heat_recovered = [m.heat_recovered for m in recent_metrics]
        fouling_indices = [m.fouling_index for m in recent_metrics]

        return {
            'avg_efficiency': statistics.mean(efficiencies),
            'min_efficiency': min(efficiencies),
            'max_efficiency': max(efficiencies),
            'avg_heat_recovered': statistics.mean(heat_recovered),
            'total_heat_recovered': sum(heat_recovered),
            'avg_fouling_index': statistics.mean(fouling_indices),
            'degradation_rate': self.performance_monitor.get_degradation_rate(),
            'samples_analyzed': len(recent_metrics)
        }

    async def shutdown(self):
        """Shutdown economizer connector."""
        logger.info("Shutting down economizer connector")
        self._running = False
        await self.modbus_client.disconnect()


# Example usage
async def main():
    """Example economizer monitoring."""
    config = EconomizerConfig(
        protocol="modbus_tcp",
        host="192.168.1.100",
        port=502,
        economizer_type=EconomizerType.FEEDWATER,
        design_capacity=1500.0
    )

    connector = EconomizerConnector(config)

    try:
        await connector.initialize()

        # Start monitoring
        monitor_task = asyncio.create_task(connector.monitor_performance())

        # Let it run for demonstration
        await asyncio.sleep(30)

        # Get summary
        summary = connector.get_performance_summary()
        logger.info(f"Performance summary: {summary}")

    finally:
        await connector.shutdown()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())