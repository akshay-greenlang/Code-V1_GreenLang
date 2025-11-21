# -*- coding: utf-8 -*-
"""
Process Stream Monitor for GL-006 HeatRecoveryMaximizer

Multi-point temperature and flow monitoring across facility streams
with distributed sensor support via Modbus RTU and 4-20mA signals.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import statistics
import numpy as np
from collections import defaultdict
from greenlang.determinism import DeterministicClock

try:
    from pymodbus.client import AsyncModbusSerialClient, AsyncModbusTcpClient
    MODBUS_AVAILABLE = True
except ImportError:
    MODBUS_AVAILABLE = False

logger = logging.getLogger(__name__)


class StreamType(Enum):
    """Types of process streams."""
    HOT_EXHAUST = "hot_exhaust"
    HOT_PROCESS = "hot_process"
    HOT_COOLING = "hot_cooling"
    COLD_FEED = "cold_feed"
    COLD_PROCESS = "cold_process"
    COLD_UTILITY = "cold_utility"
    STEAM = "steam"
    CONDENSATE = "condensate"


class SensorType(Enum):
    """Types of sensors in the monitoring network."""
    TEMPERATURE = "temperature"
    FLOW = "flow"
    PRESSURE = "pressure"
    COMPOSITION = "composition"
    ANALOG_4_20MA = "4_20ma"


@dataclass
class StreamProperties:
    """Properties of a process stream."""
    stream_id: str
    stream_type: StreamType
    temperature: float  # °C
    flow_rate: float  # kg/h or m³/h
    pressure: float  # bar
    specific_heat: float  # kJ/kg·K
    density: float  # kg/m³
    composition: Dict[str, float] = field(default_factory=dict)  # Component fractions
    heat_content: float = 0.0  # kW
    quality_score: float = 0.0  # Data quality 0-100


@dataclass
class SensorPoint:
    """Individual sensor point configuration."""
    sensor_id: str
    sensor_type: SensorType
    stream_id: str
    modbus_address: int
    register_count: int = 2  # For float values
    scaling_factor: float = 1.0
    offset: float = 0.0
    unit: str = ""
    min_value: float = -1000.0
    max_value: float = 1000.0
    last_value: Optional[float] = None
    last_update: Optional[datetime] = None
    error_count: int = 0


@dataclass
class MonitorConfig:
    """Process stream monitor configuration."""
    # Modbus RTU settings for distributed sensors
    serial_ports: List[str] = field(default_factory=lambda: ["/dev/ttyUSB0"])
    baudrate: int = 19200
    parity: str = 'E'
    stopbits: int = 1
    bytesize: int = 8

    # Modbus TCP gateways
    tcp_gateways: List[Tuple[str, int]] = field(default_factory=list)

    # 4-20mA analog input settings
    analog_input_base_address: int = 30001
    analog_channels: int = 16
    analog_scaling: Tuple[float, float] = (4.0, 20.0)  # mA range

    # Monitoring parameters
    scan_interval: float = 2.0  # seconds
    data_retention_hours: int = 24
    anomaly_threshold: float = 3.0  # Standard deviations
    min_data_quality: float = 80.0  # %

    # Heat recovery thresholds
    min_temp_difference: float = 20.0  # °C
    min_heat_potential: float = 50.0  # kW


class DataAggregator:
    """Aggregates and analyzes stream data."""

    def __init__(self, retention_hours: int = 24):
        self.retention_hours = retention_hours
        self.stream_history: Dict[str, List[Tuple[datetime, StreamProperties]]] = defaultdict(list)
        self.heat_map: Dict[str, float] = {}
        self.recovery_opportunities: List[Dict] = []

    def add_stream_data(self, stream: StreamProperties):
        """Add stream data point to history."""
        now = DeterministicClock.now()
        self.stream_history[stream.stream_id].append((now, stream))

        # Calculate heat content
        stream.heat_content = self._calculate_heat_content(stream)

        # Clean old data
        cutoff = now - timedelta(hours=self.retention_hours)
        self.stream_history[stream.stream_id] = [
            (t, s) for t, s in self.stream_history[stream.stream_id]
            if t > cutoff
        ]

    def _calculate_heat_content(self, stream: StreamProperties) -> float:
        """Calculate heat content of stream in kW."""
        if stream.flow_rate <= 0:
            return 0.0

        # Convert flow to kg/s
        mass_flow = stream.flow_rate / 3600  # kg/h to kg/s

        # Reference temperature (ambient)
        t_ref = 25.0  # °C

        # Heat content relative to reference
        heat_content = mass_flow * stream.specific_heat * (stream.temperature - t_ref)

        return abs(heat_content)

    def identify_recovery_opportunities(self) -> List[Dict]:
        """Identify heat recovery opportunities between streams."""
        opportunities = []

        # Get latest data for all streams
        latest_streams = {}
        for stream_id, history in self.stream_history.items():
            if history:
                latest_streams[stream_id] = history[-1][1]

        # Find hot-cold stream pairs
        hot_streams = [s for s in latest_streams.values()
                      if s.stream_type in [StreamType.HOT_EXHAUST, StreamType.HOT_PROCESS, StreamType.HOT_COOLING]]
        cold_streams = [s for s in latest_streams.values()
                       if s.stream_type in [StreamType.COLD_FEED, StreamType.COLD_PROCESS, StreamType.COLD_UTILITY]]

        for hot in hot_streams:
            for cold in cold_streams:
                if hot.temperature > cold.temperature + 20:  # Minimum driving force
                    # Calculate potential heat transfer
                    hot_capacity = hot.flow_rate * hot.specific_heat / 3600
                    cold_capacity = cold.flow_rate * cold.specific_heat / 3600
                    min_capacity = min(hot_capacity, cold_capacity)

                    max_heat_transfer = min_capacity * (hot.temperature - cold.temperature)

                    if max_heat_transfer > 50:  # Minimum 50 kW potential
                        opportunities.append({
                            'hot_stream': hot.stream_id,
                            'cold_stream': cold.stream_id,
                            'hot_temp': hot.temperature,
                            'cold_temp': cold.temperature,
                            'potential_heat_recovery': max_heat_transfer,
                            'hot_heat_available': hot.heat_content,
                            'cold_heat_required': cold.heat_content,
                            'feasibility_score': self._calculate_feasibility(hot, cold, max_heat_transfer)
                        })

        # Sort by potential
        opportunities.sort(key=lambda x: x['potential_heat_recovery'], reverse=True)
        self.recovery_opportunities = opportunities[:10]  # Top 10 opportunities

        return self.recovery_opportunities

    def _calculate_feasibility(self, hot: StreamProperties, cold: StreamProperties,
                              heat_transfer: float) -> float:
        """Calculate feasibility score for heat recovery (0-100)."""
        score = 100.0

        # Temperature difference factor
        delta_t = hot.temperature - cold.temperature
        if delta_t < 30:
            score -= 20  # Small driving force
        elif delta_t > 100:
            score -= 10  # May need special materials

        # Flow stability (check variability in history)
        hot_variability = self._get_flow_variability(hot.stream_id)
        cold_variability = self._get_flow_variability(cold.stream_id)

        score -= min(hot_variability + cold_variability, 30)

        # Data quality
        score -= (100 - hot.quality_score) * 0.3
        score -= (100 - cold.quality_score) * 0.3

        # Heat transfer magnitude
        if heat_transfer < 100:
            score -= 15  # Small recovery
        elif heat_transfer > 1000:
            score += 10  # Large recovery

        return max(0, min(100, score))

    def _get_flow_variability(self, stream_id: str) -> float:
        """Calculate flow rate variability (coefficient of variation)."""
        if stream_id not in self.stream_history:
            return 50.0  # Unknown

        history = self.stream_history[stream_id]
        if len(history) < 10:
            return 50.0

        flows = [s.flow_rate for _, s in history[-50:]]
        if not flows or statistics.mean(flows) == 0:
            return 50.0

        cv = (statistics.stdev(flows) / statistics.mean(flows)) * 100
        return min(cv, 50.0)

    def get_stream_statistics(self, stream_id: str) -> Dict[str, Any]:
        """Get statistics for a stream."""
        if stream_id not in self.stream_history:
            return {}

        history = self.stream_history[stream_id]
        if len(history) < 2:
            return {}

        temps = [s.temperature for _, s in history]
        flows = [s.flow_rate for _, s in history]
        heats = [s.heat_content for _, s in history]

        return {
            'avg_temperature': statistics.mean(temps),
            'std_temperature': statistics.stdev(temps) if len(temps) > 1 else 0,
            'avg_flow': statistics.mean(flows),
            'std_flow': statistics.stdev(flows) if len(flows) > 1 else 0,
            'avg_heat_content': statistics.mean(heats),
            'total_heat_available': sum(heats) * (self.retention_hours / len(heats)),
            'data_points': len(history)
        }


class ModbusSensorNetwork:
    """Manages distributed Modbus sensor network."""

    def __init__(self, config: MonitorConfig):
        self.config = config
        self.rtu_clients: List[AsyncModbusSerialClient] = []
        self.tcp_clients: List[AsyncModbusTcpClient] = []
        self.sensor_points: Dict[str, SensorPoint] = {}
        self.connected = False

    async def initialize(self):
        """Initialize all Modbus connections."""
        logger.info("Initializing Modbus sensor network")

        if not MODBUS_AVAILABLE:
            logger.warning("Modbus libraries not available, using mock mode")
            self.connected = True
            return

        # Initialize RTU connections
        for port in self.config.serial_ports:
            try:
                client = AsyncModbusSerialClient(
                    port=port,
                    baudrate=self.config.baudrate,
                    parity=self.config.parity,
                    stopbits=self.config.stopbits,
                    bytesize=self.config.bytesize,
                    timeout=3.0
                )
                await client.connect()
                self.rtu_clients.append(client)
                logger.info(f"Connected to Modbus RTU on {port}")
            except Exception as e:
                logger.error(f"Failed to connect to {port}: {e}")

        # Initialize TCP connections
        for host, port in self.config.tcp_gateways:
            try:
                client = AsyncModbusTcpClient(host, port=port, timeout=5.0)
                await client.connect()
                self.tcp_clients.append(client)
                logger.info(f"Connected to Modbus TCP at {host}:{port}")
            except Exception as e:
                logger.error(f"Failed to connect to {host}:{port}: {e}")

        self.connected = len(self.rtu_clients) > 0 or len(self.tcp_clients) > 0

    def register_sensor(self, sensor: SensorPoint):
        """Register a sensor point."""
        self.sensor_points[sensor.sensor_id] = sensor
        logger.info(f"Registered sensor {sensor.sensor_id} for stream {sensor.stream_id}")

    async def read_sensor(self, sensor: SensorPoint, client_index: int = 0) -> Optional[float]:
        """Read value from a sensor."""
        try:
            if not MODBUS_AVAILABLE:
                # Mock data
                import random
                base_value = 50.0 if sensor.sensor_type == SensorType.TEMPERATURE else 100.0
                return base_value + random.uniform(-10, 10)

            # Select client (RTU or TCP)
            if client_index < len(self.rtu_clients):
                client = self.rtu_clients[client_index]
            elif client_index - len(self.rtu_clients) < len(self.tcp_clients):
                client = self.tcp_clients[client_index - len(self.rtu_clients)]
            else:
                logger.error(f"Invalid client index {client_index}")
                return None

            # Read registers
            if sensor.sensor_type == SensorType.ANALOG_4_20MA:
                # Read analog input register
                response = await client.read_input_registers(
                    sensor.modbus_address,
                    1,
                    slave=1
                )
            else:
                # Read holding registers
                response = await client.read_holding_registers(
                    sensor.modbus_address,
                    sensor.register_count,
                    slave=1
                )

            if response.isError():
                sensor.error_count += 1
                logger.warning(f"Modbus read error for {sensor.sensor_id}: {response}")
                return None

            # Convert to value
            if sensor.sensor_type == SensorType.ANALOG_4_20MA:
                # Convert 4-20mA to engineering units
                raw_value = response.registers[0]
                mA = 4.0 + (raw_value / 65535.0) * 16.0  # Convert to mA
                value = sensor.offset + (mA - 4.0) / 16.0 * sensor.scaling_factor
            else:
                # Standard float conversion
                if sensor.register_count == 2:
                    # Float32
                    import struct
                    combined = (response.registers[0] << 16) | response.registers[1]
                    bytes_data = struct.pack('>I', combined)
                    value = struct.unpack('>f', bytes_data)[0]
                else:
                    # Single register integer
                    value = response.registers[0] * sensor.scaling_factor + sensor.offset

            # Validate range
            if sensor.min_value <= value <= sensor.max_value:
                sensor.last_value = value
                sensor.last_update = DeterministicClock.now()
                sensor.error_count = 0
                return value
            else:
                logger.warning(f"Sensor {sensor.sensor_id} value {value} out of range")
                sensor.error_count += 1
                return None

        except Exception as e:
            logger.error(f"Error reading sensor {sensor.sensor_id}: {e}")
            sensor.error_count += 1
            return None

    async def close(self):
        """Close all Modbus connections."""
        for client in self.rtu_clients:
            client.close()
        for client in self.tcp_clients:
            client.close()
        self.connected = False


class ProcessStreamMonitor:
    """Main process stream monitoring system."""

    def __init__(self, config: MonitorConfig):
        self.config = config
        self.sensor_network = ModbusSensorNetwork(config)
        self.data_aggregator = DataAggregator(config.data_retention_hours)
        self.streams: Dict[str, StreamProperties] = {}
        self._running = False

        # Default stream configurations
        self._setup_default_streams()

    def _setup_default_streams(self):
        """Setup default process stream configurations."""
        # Example stream definitions
        default_streams = [
            ("EXHAUST_001", StreamType.HOT_EXHAUST, 350.0, 5000.0, 1.01, 1.2),
            ("EXHAUST_002", StreamType.HOT_EXHAUST, 280.0, 3000.0, 1.01, 1.2),
            ("COOLING_001", StreamType.HOT_COOLING, 90.0, 10000.0, 4.18, 1000.0),
            ("FEED_001", StreamType.COLD_FEED, 25.0, 8000.0, 4.18, 1000.0),
            ("FEED_002", StreamType.COLD_PROCESS, 40.0, 6000.0, 2.5, 800.0),
        ]

        for stream_id, stream_type, temp, flow, cp, density in default_streams:
            self.streams[stream_id] = StreamProperties(
                stream_id=stream_id,
                stream_type=stream_type,
                temperature=temp,
                flow_rate=flow,
                pressure=1.0,
                specific_heat=cp,
                density=density,
                quality_score=95.0
            )

            # Register sensors for each stream
            # Temperature sensor
            self.sensor_network.register_sensor(SensorPoint(
                sensor_id=f"{stream_id}_TEMP",
                sensor_type=SensorType.TEMPERATURE,
                stream_id=stream_id,
                modbus_address=40001 + len(self.sensor_network.sensor_points) * 2,
                register_count=2,
                scaling_factor=1.0,
                offset=0.0,
                unit="°C",
                min_value=0.0,
                max_value=500.0
            ))

            # Flow sensor
            self.sensor_network.register_sensor(SensorPoint(
                sensor_id=f"{stream_id}_FLOW",
                sensor_type=SensorType.FLOW,
                stream_id=stream_id,
                modbus_address=40021 + len(self.sensor_network.sensor_points) * 2,
                register_count=2,
                scaling_factor=1.0,
                offset=0.0,
                unit="m³/h",
                min_value=0.0,
                max_value=20000.0
            ))

    async def initialize(self):
        """Initialize stream monitor."""
        logger.info("Initializing process stream monitor")

        await self.sensor_network.initialize()

        if not self.sensor_network.connected:
            logger.warning("No sensor connections established, using mock data")

        self._running = True
        logger.info("Process stream monitor initialized")

    async def scan_all_sensors(self) -> Dict[str, StreamProperties]:
        """Scan all registered sensors and update stream properties."""
        updated_streams = {}

        for sensor_id, sensor in self.sensor_network.sensor_points.items():
            value = await self.sensor_network.read_sensor(sensor)

            if value is not None and sensor.stream_id in self.streams:
                stream = self.streams[sensor.stream_id]

                # Update stream property based on sensor type
                if sensor.sensor_type == SensorType.TEMPERATURE:
                    stream.temperature = value
                elif sensor.sensor_type == SensorType.FLOW:
                    stream.flow_rate = value
                elif sensor.sensor_type == SensorType.PRESSURE:
                    stream.pressure = value

                # Calculate data quality
                if sensor.error_count == 0:
                    stream.quality_score = 100.0
                else:
                    stream.quality_score = max(0, 100 - sensor.error_count * 10)

                updated_streams[sensor.stream_id] = stream

        return updated_streams

    async def monitoring_loop(self):
        """Main monitoring loop."""
        logger.info("Starting process stream monitoring loop")

        while self._running:
            try:
                # Scan all sensors
                updated_streams = await self.scan_all_sensors()

                # Add to aggregator
                for stream in updated_streams.values():
                    self.data_aggregator.add_stream_data(stream)

                # Identify recovery opportunities
                opportunities = self.data_aggregator.identify_recovery_opportunities()

                if opportunities:
                    logger.info(f"Found {len(opportunities)} heat recovery opportunities")
                    for opp in opportunities[:3]:  # Log top 3
                        logger.info(
                            f"Opportunity: {opp['hot_stream']} -> {opp['cold_stream']}: "
                            f"{opp['potential_heat_recovery']:.1f} kW potential"
                        )

                # Log stream status
                for stream_id, stream in updated_streams.items():
                    logger.debug(
                        f"Stream {stream_id}: {stream.temperature:.1f}°C, "
                        f"{stream.flow_rate:.1f} m³/h, {stream.heat_content:.1f} kW"
                    )

                await asyncio.sleep(self.config.scan_interval)

            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(self.config.scan_interval)

    def get_current_status(self) -> Dict[str, Any]:
        """Get current monitoring status."""
        total_hot_heat = sum(
            s.heat_content for s in self.streams.values()
            if s.stream_type in [StreamType.HOT_EXHAUST, StreamType.HOT_PROCESS]
        )

        total_cold_demand = sum(
            s.heat_content for s in self.streams.values()
            if s.stream_type in [StreamType.COLD_FEED, StreamType.COLD_PROCESS]
        )

        return {
            'total_streams': len(self.streams),
            'active_sensors': len([s for s in self.sensor_network.sensor_points.values()
                                  if s.last_update and
                                  (DeterministicClock.now() - s.last_update).seconds < 60]),
            'total_hot_heat_available': total_hot_heat,
            'total_cold_heat_demand': total_cold_demand,
            'recovery_opportunities': len(self.data_aggregator.recovery_opportunities),
            'top_opportunity': self.data_aggregator.recovery_opportunities[0]
                              if self.data_aggregator.recovery_opportunities else None
        }

    async def shutdown(self):
        """Shutdown stream monitor."""
        logger.info("Shutting down process stream monitor")
        self._running = False
        await self.sensor_network.close()


# Example usage
async def main():
    """Example stream monitoring."""
    config = MonitorConfig(
        serial_ports=["/dev/ttyUSB0"],
        tcp_gateways=[("192.168.1.100", 502)],
        scan_interval=5.0
    )

    monitor = ProcessStreamMonitor(config)

    try:
        await monitor.initialize()
        monitoring_task = asyncio.create_task(monitor.monitoring_loop())

        # Let it run
        await asyncio.sleep(30)

        # Get status
        status = monitor.get_current_status()
        logger.info(f"Monitor status: {status}")

    finally:
        await monitor.shutdown()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())