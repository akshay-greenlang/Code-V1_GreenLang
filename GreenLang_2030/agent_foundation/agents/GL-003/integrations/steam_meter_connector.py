"""
Steam Meter Integration Connector for GL-003 SteamSystemAnalyzer

Supports multiple steam meter protocols and provides comprehensive flow measurement:
- Modbus RTU/TCP for digital steam meters
- HART protocol for smart transmitters
- 4-20mA analog input for traditional meters
- Mass flow and volumetric flow measurement
- Totalizer readings and accumulation
- Data quality validation
- Thread-safe data collection

Features:
- Multi-protocol support (Modbus, HART, 4-20mA)
- Flow rate calculation (mass and volumetric)
- Steam quality compensation
- Totalizer management
- Calibration factor application
- Trend analysis and anomaly detection
"""

import asyncio
import logging
import struct
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import deque
import random  # For simulation

from .base_connector import (
    BaseConnector,
    ConnectionConfig,
    ConnectionState
)

logger = logging.getLogger(__name__)


class MeterProtocol(Enum):
    """Supported steam meter protocols."""
    MODBUS_RTU = "modbus_rtu"
    MODBUS_TCP = "modbus_tcp"
    HART = "hart"
    ANALOG_4_20MA = "4-20ma"
    OPC_UA = "opc_ua"


class FlowMeasurementType(Enum):
    """Flow measurement types."""
    MASS_FLOW = "mass_flow"  # kg/hr or t/hr
    VOLUMETRIC_FLOW = "volumetric_flow"  # m3/hr
    VELOCITY = "velocity"  # m/s
    ENERGY_FLOW = "energy_flow"  # MW or MMBtu/hr


@dataclass
class SteamMeterConfig(ConnectionConfig):
    """Steam meter specific configuration."""
    protocol: MeterProtocol = MeterProtocol.MODBUS_TCP
    meter_id: str = "steam_meter_1"
    register_map: Dict[str, int] = field(default_factory=dict)

    # Modbus specific
    slave_address: int = 1
    baud_rate: int = 9600
    parity: str = "N"  # N, E, O
    stop_bits: int = 1

    # HART specific
    hart_address: int = 0
    poll_address: int = 0

    # 4-20mA specific
    current_min_ma: float = 4.0
    current_max_ma: float = 20.0
    flow_min: float = 0.0
    flow_max: float = 100.0  # t/hr

    # Calibration
    calibration_factor: float = 1.0
    calibration_offset: float = 0.0
    k_factor: Optional[float] = None  # For turbine meters

    # Measurement settings
    measurement_type: FlowMeasurementType = FlowMeasurementType.MASS_FLOW
    units: str = "t/hr"
    sampling_rate_hz: float = 1.0
    averaging_window_seconds: int = 60

    # Totalizer settings
    enable_totalizer: bool = True
    totalizer_rollover: float = 9999999.0

    # Quality settings
    min_valid_flow: float = 0.0
    max_valid_flow: float = 500.0
    max_flow_change_rate: float = 50.0  # t/hr per minute


@dataclass
class FlowReading:
    """Flow measurement reading."""
    timestamp: datetime
    flow_rate: float  # Current flow rate
    totalizer: float  # Accumulated total
    velocity: Optional[float] = None
    density: Optional[float] = None
    temperature: Optional[float] = None
    pressure: Optional[float] = None
    quality_score: float = 100.0
    raw_value: Optional[float] = None
    unit: str = "t/hr"
    meter_id: str = ""
    protocol: str = ""


@dataclass
class TotalizerState:
    """Totalizer state tracking."""
    current_total: float = 0.0
    previous_total: float = 0.0
    last_update: Optional[datetime] = None
    rollover_count: int = 0
    lifetime_total: float = 0.0


class SteamMeterConnector(BaseConnector):
    """
    Steam meter integration connector.

    Provides unified interface for multiple steam meter protocols.
    """

    def __init__(self, config: SteamMeterConfig):
        """
        Initialize steam meter connector.

        Args:
            config: Steam meter configuration
        """
        super().__init__(config)
        self.config: SteamMeterConfig = config

        self.current_reading: Optional[FlowReading] = None
        self.reading_history = deque(maxlen=1000)
        self.totalizer_state = TotalizerState()

        self._sampling_task = None
        self._trend_analyzer_task = None

        # Protocol-specific handlers
        self.protocol_handlers = {
            MeterProtocol.MODBUS_TCP: self._read_modbus_tcp,
            MeterProtocol.MODBUS_RTU: self._read_modbus_rtu,
            MeterProtocol.HART: self._read_hart,
            MeterProtocol.ANALOG_4_20MA: self._read_4_20ma,
            MeterProtocol.OPC_UA: self._read_opc_ua
        }

        # Initialize register map if not provided
        if not config.register_map:
            self._init_default_register_map()

    def _init_default_register_map(self):
        """Initialize default Modbus register map."""
        self.config.register_map = {
            'flow_rate': 0,          # Holding register for flow rate
            'totalizer': 2,          # Holding register for totalizer (2 registers)
            'velocity': 4,           # Velocity
            'density': 5,            # Steam density
            'temperature': 6,        # Steam temperature
            'pressure': 7,           # Steam pressure
            'status': 8,             # Meter status
            'alarm': 9               # Alarm register
        }

    async def _connect_impl(self) -> bool:
        """Implement steam meter connection."""
        try:
            if self.config.protocol == MeterProtocol.MODBUS_TCP:
                # Simulate Modbus TCP connection
                self.connection = {
                    'type': 'modbus_tcp',
                    'host': self.config.host,
                    'port': self.config.port,
                    'connected': True
                }
                logger.info(f"Connected to Modbus TCP meter at {self.config.host}:{self.config.port}")

            elif self.config.protocol == MeterProtocol.MODBUS_RTU:
                # Simulate Modbus RTU connection
                self.connection = {
                    'type': 'modbus_rtu',
                    'port': self.config.host,  # Serial port
                    'baud': self.config.baud_rate,
                    'connected': True
                }
                logger.info(f"Connected to Modbus RTU meter on {self.config.host}")

            elif self.config.protocol == MeterProtocol.HART:
                # Simulate HART connection
                self.connection = {
                    'type': 'hart',
                    'address': self.config.hart_address,
                    'connected': True
                }
                logger.info(f"Connected to HART meter address {self.config.hart_address}")

            elif self.config.protocol == MeterProtocol.ANALOG_4_20MA:
                # Simulate analog input
                self.connection = {
                    'type': '4-20ma',
                    'channel': self.config.port,
                    'connected': True
                }
                logger.info(f"Connected to 4-20mA meter on channel {self.config.port}")

            # Start sampling task
            self._sampling_task = asyncio.create_task(self._sampling_loop())

            # Start trend analyzer
            self._trend_analyzer_task = asyncio.create_task(self._trend_analyzer_loop())

            return True

        except Exception as e:
            logger.error(f"Failed to connect to steam meter: {e}")
            return False

    async def _disconnect_impl(self):
        """Implement steam meter disconnection."""
        # Cancel sampling task
        if self._sampling_task:
            self._sampling_task.cancel()
            self._sampling_task = None

        # Cancel trend analyzer
        if self._trend_analyzer_task:
            self._trend_analyzer_task.cancel()
            self._trend_analyzer_task = None

        self.connection = None
        logger.info("Disconnected from steam meter")

    async def _health_check_impl(self) -> bool:
        """Implement health check for steam meter."""
        try:
            # Try to read current flow
            reading = await self.read_flow()
            return reading is not None
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

    async def _sampling_loop(self):
        """Continuous sampling loop."""
        interval = 1.0 / self.config.sampling_rate_hz

        while self.state == ConnectionState.CONNECTED:
            try:
                await self.read_flow()
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Sampling error: {e}")
                await asyncio.sleep(interval)

    async def read_flow(self) -> Optional[FlowReading]:
        """
        Read current flow rate from meter.

        Returns:
            Flow reading or None if failed
        """
        try:
            # Call protocol-specific handler
            handler = self.protocol_handlers.get(self.config.protocol)
            if not handler:
                logger.error(f"No handler for protocol: {self.config.protocol}")
                return None

            reading = await self.execute_with_retry(handler)

            if reading:
                # Apply calibration
                reading.flow_rate = (reading.flow_rate * self.config.calibration_factor) + \
                                  self.config.calibration_offset

                # Validate reading
                reading.quality_score = self._validate_reading(reading)

                # Update totalizer
                if self.config.enable_totalizer:
                    self._update_totalizer(reading)

                # Store reading
                self.current_reading = reading
                self.reading_history.append(reading)

                logger.debug(f"Flow reading: {reading.flow_rate:.2f} {reading.unit}")

            return reading

        except Exception as e:
            logger.error(f"Failed to read flow: {e}")
            return None

    async def _read_modbus_tcp(self) -> FlowReading:
        """Read flow via Modbus TCP."""
        # Simulate Modbus TCP read
        # In production, would use pymodbus library

        # Simulate realistic steam flow (80-120 t/hr with some variation)
        base_flow = 100.0
        flow_rate = base_flow + random.uniform(-20, 20)

        # Simulate totalizer increment
        if self.totalizer_state.last_update:
            time_delta = (datetime.utcnow() - self.totalizer_state.last_update).total_seconds() / 3600
            increment = flow_rate * time_delta
            totalizer = self.totalizer_state.current_total + increment
        else:
            totalizer = random.uniform(1000000, 2000000)

        return FlowReading(
            timestamp=datetime.utcnow(),
            flow_rate=flow_rate,
            totalizer=totalizer,
            velocity=random.uniform(20, 30),  # m/s
            density=random.uniform(15, 20),   # kg/m3
            temperature=random.uniform(180, 200),  # °C
            pressure=random.uniform(10, 12),  # bar
            raw_value=flow_rate,
            unit=self.config.units,
            meter_id=self.config.meter_id,
            protocol="modbus_tcp"
        )

    async def _read_modbus_rtu(self) -> FlowReading:
        """Read flow via Modbus RTU."""
        # Similar to Modbus TCP but over serial
        return await self._read_modbus_tcp()

    async def _read_hart(self) -> FlowReading:
        """Read flow via HART protocol."""
        # Simulate HART read
        # HART provides primary variable (PV), secondary (SV), tertiary (TV), quaternary (QV)

        flow_rate = 100.0 + random.uniform(-15, 15)

        if self.totalizer_state.last_update:
            time_delta = (datetime.utcnow() - self.totalizer_state.last_update).total_seconds() / 3600
            increment = flow_rate * time_delta
            totalizer = self.totalizer_state.current_total + increment
        else:
            totalizer = random.uniform(1000000, 2000000)

        return FlowReading(
            timestamp=datetime.utcnow(),
            flow_rate=flow_rate,
            totalizer=totalizer,
            temperature=random.uniform(180, 200),  # From HART SV
            pressure=random.uniform(10, 12),  # From HART TV
            raw_value=flow_rate,
            unit=self.config.units,
            meter_id=self.config.meter_id,
            protocol="hart"
        )

    async def _read_4_20ma(self) -> FlowReading:
        """Read flow from 4-20mA analog signal."""
        # Simulate analog input reading
        current_ma = random.uniform(10, 18)  # 4-20mA signal

        # Convert current to flow rate
        current_range = self.config.current_max_ma - self.config.current_min_ma
        flow_range = self.config.flow_max - self.config.flow_min

        flow_rate = self.config.flow_min + \
                   ((current_ma - self.config.current_min_ma) / current_range) * flow_range

        # 4-20mA typically doesn't provide totalizer, calculate from flow
        if self.totalizer_state.last_update:
            time_delta = (datetime.utcnow() - self.totalizer_state.last_update).total_seconds() / 3600
            increment = flow_rate * time_delta
            totalizer = self.totalizer_state.current_total + increment
        else:
            totalizer = 0.0

        return FlowReading(
            timestamp=datetime.utcnow(),
            flow_rate=flow_rate,
            totalizer=totalizer,
            raw_value=current_ma,
            unit=self.config.units,
            meter_id=self.config.meter_id,
            protocol="4-20ma"
        )

    async def _read_opc_ua(self) -> FlowReading:
        """Read flow via OPC UA."""
        # Similar to Modbus but over OPC UA
        return await self._read_modbus_tcp()

    def _validate_reading(self, reading: FlowReading) -> float:
        """
        Validate flow reading and calculate quality score.

        Args:
            reading: Flow reading to validate

        Returns:
            Quality score (0-100)
        """
        quality_score = 100.0

        # Range check
        if reading.flow_rate < self.config.min_valid_flow or \
           reading.flow_rate > self.config.max_valid_flow:
            quality_score -= 50
            logger.warning(f"Flow rate out of range: {reading.flow_rate}")

        # Rate of change check
        if len(self.reading_history) > 0:
            last_reading = self.reading_history[-1]
            time_delta = (reading.timestamp - last_reading.timestamp).total_seconds() / 60

            if time_delta > 0:
                flow_change = abs(reading.flow_rate - last_reading.flow_rate)
                rate_of_change = flow_change / time_delta

                if rate_of_change > self.config.max_flow_change_rate:
                    quality_score -= 30
                    logger.warning(f"Excessive flow rate change: {rate_of_change:.1f} t/hr/min")

        # Negative flow check
        if reading.flow_rate < 0:
            quality_score -= 40
            logger.warning("Negative flow rate detected")

        # Flatline detection
        if len(self.reading_history) >= 10:
            recent_flows = [r.flow_rate for r in list(self.reading_history)[-10:]]
            if all(abs(f - recent_flows[0]) < 0.01 for f in recent_flows):
                quality_score -= 20
                logger.warning("Flatline detected in flow measurements")

        return max(0, quality_score)

    def _update_totalizer(self, reading: FlowReading):
        """
        Update totalizer state and detect rollovers.

        Args:
            reading: Current reading with totalizer value
        """
        current_total = reading.totalizer

        # Detect rollover
        if self.totalizer_state.previous_total > 0 and \
           current_total < self.totalizer_state.previous_total:
            if abs(current_total - self.totalizer_state.previous_total) > \
               self.config.totalizer_rollover * 0.5:
                # Likely rollover
                self.totalizer_state.rollover_count += 1
                logger.info(f"Totalizer rollover detected (count: {self.totalizer_state.rollover_count})")

        # Update state
        self.totalizer_state.previous_total = self.totalizer_state.current_total
        self.totalizer_state.current_total = current_total
        self.totalizer_state.last_update = reading.timestamp

        # Calculate lifetime total
        self.totalizer_state.lifetime_total = \
            (self.totalizer_state.rollover_count * self.config.totalizer_rollover) + current_total

        # Update reading with lifetime total
        reading.totalizer = self.totalizer_state.lifetime_total

    async def _trend_analyzer_loop(self):
        """Analyze flow trends and detect anomalies."""
        while self.state == ConnectionState.CONNECTED:
            try:
                await asyncio.sleep(self.config.averaging_window_seconds)

                if len(self.reading_history) >= 10:
                    await self._analyze_trends()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Trend analyzer error: {e}")

    async def _analyze_trends(self):
        """Analyze flow trends."""
        # Get recent readings
        recent_flows = [r.flow_rate for r in list(self.reading_history)[-60:]]

        if not recent_flows:
            return

        # Calculate statistics
        avg_flow = sum(recent_flows) / len(recent_flows)
        min_flow = min(recent_flows)
        max_flow = max(recent_flows)

        # Calculate standard deviation
        variance = sum((f - avg_flow) ** 2 for f in recent_flows) / len(recent_flows)
        std_dev = variance ** 0.5

        # Detect anomalies
        if self.current_reading:
            z_score = abs(self.current_reading.flow_rate - avg_flow) / std_dev if std_dev > 0 else 0

            if z_score > 3:
                logger.warning(f"Flow anomaly detected: Z-score = {z_score:.2f}")

        logger.debug(f"Flow trend: avg={avg_flow:.1f}, min={min_flow:.1f}, max={max_flow:.1f}, std={std_dev:.1f}")

    def get_averaged_flow(self, window_seconds: int = 60) -> Optional[float]:
        """
        Get averaged flow rate over time window.

        Args:
            window_seconds: Averaging window in seconds

        Returns:
            Averaged flow rate or None
        """
        if not self.reading_history:
            return None

        cutoff_time = datetime.utcnow() - timedelta(seconds=window_seconds)
        recent_readings = [
            r for r in self.reading_history
            if r.timestamp > cutoff_time
        ]

        if not recent_readings:
            return None

        return sum(r.flow_rate for r in recent_readings) / len(recent_readings)

    def get_totalizer(self) -> Dict[str, Any]:
        """
        Get totalizer information.

        Returns:
            Totalizer state dictionary
        """
        return {
            'current_total': self.totalizer_state.current_total,
            'lifetime_total': self.totalizer_state.lifetime_total,
            'rollover_count': self.totalizer_state.rollover_count,
            'last_update': self.totalizer_state.last_update.isoformat() if \
                          self.totalizer_state.last_update else None,
            'unit': self.config.units
        }

    def reset_totalizer(self):
        """Reset totalizer (for maintenance/calibration)."""
        self.totalizer_state = TotalizerState()
        logger.info("Totalizer reset")

    def get_flow_statistics(self, window_minutes: int = 60) -> Dict[str, Any]:
        """
        Get flow statistics over time window.

        Args:
            window_minutes: Analysis window in minutes

        Returns:
            Statistics dictionary
        """
        cutoff_time = datetime.utcnow() - timedelta(minutes=window_minutes)
        recent_readings = [
            r for r in self.reading_history
            if r.timestamp > cutoff_time
        ]

        if not recent_readings:
            return {'error': 'No data available'}

        flows = [r.flow_rate for r in recent_readings]
        qualities = [r.quality_score for r in recent_readings]

        return {
            'count': len(flows),
            'min': min(flows),
            'max': max(flows),
            'avg': sum(flows) / len(flows),
            'current': recent_readings[-1].flow_rate if recent_readings else None,
            'avg_quality': sum(qualities) / len(qualities),
            'window_minutes': window_minutes,
            'unit': self.config.units
        }


# Example usage
async def main():
    """Example usage of steam meter connector."""

    # Configure steam meter
    config = SteamMeterConfig(
        host="192.168.1.100",
        port=502,
        protocol=MeterProtocol.MODBUS_TCP,
        meter_id="steam_header_meter",
        slave_address=1,
        sampling_rate_hz=1.0,
        units="t/hr",
        min_valid_flow=0.0,
        max_valid_flow=200.0,
        enable_totalizer=True
    )

    # Initialize connector
    connector = SteamMeterConnector(config)

    # Connect
    if await connector.connect():
        print(f"Connected to steam meter: {config.meter_id}")

        # Let it collect data
        await asyncio.sleep(10)

        # Get current reading
        reading = connector.current_reading
        if reading:
            print(f"\nCurrent Flow:")
            print(f"  Rate: {reading.flow_rate:.2f} {reading.unit}")
            print(f"  Totalizer: {reading.totalizer:.0f} {reading.unit}")
            print(f"  Quality: {reading.quality_score:.0f}%")
            print(f"  Temperature: {reading.temperature:.1f}°C")
            print(f"  Pressure: {reading.pressure:.1f} bar")

        # Get averaged flow
        avg_flow = connector.get_averaged_flow(window_seconds=60)
        print(f"\n1-Minute Average: {avg_flow:.2f} {config.units}")

        # Get statistics
        stats = connector.get_flow_statistics(window_minutes=1)
        print(f"\nFlow Statistics:")
        print(f"  Count: {stats['count']}")
        print(f"  Min: {stats['min']:.2f} {stats['unit']}")
        print(f"  Max: {stats['max']:.2f} {stats['unit']}")
        print(f"  Avg: {stats['avg']:.2f} {stats['unit']}")
        print(f"  Avg Quality: {stats['avg_quality']:.1f}%")

        # Get totalizer info
        totalizer = connector.get_totalizer()
        print(f"\nTotalizer:")
        print(f"  Lifetime Total: {totalizer['lifetime_total']:.0f} {totalizer['unit']}")
        print(f"  Rollovers: {totalizer['rollover_count']}")

        # Get metrics
        metrics = connector.get_metrics()
        print(f"\nConnector Metrics:")
        print(f"  State: {metrics['state']}")
        print(f"  Total Calls: {metrics['total_calls']}")
        print(f"  Success Rate: {metrics['successful_calls'] / max(metrics['total_calls'], 1) * 100:.1f}%")
        print(f"  Avg Response: {metrics['avg_response_time_ms']:.1f}ms")

        # Disconnect
        await connector.disconnect()
        print("\nDisconnected")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
