# -*- coding: utf-8 -*-
"""
Pressure Sensor Integration Connector for GL-003 SteamSystemAnalyzer

Multi-point pressure monitoring with high-frequency sampling:
- Absolute, gauge, and differential pressure measurement
- Multiple sensor types (strain gauge, piezoelectric, capacitive)
- High-frequency sampling (1Hz - 10Hz)
- Pressure transducer support
- Calibration drift detection
- Sensor health monitoring
- Multi-zone pressure profiling
"""

import asyncio
import logging
import random
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import deque

from .base_connector import BaseConnector, ConnectionConfig, ConnectionState
from greenlang.determinism import DeterministicClock

logger = logging.getLogger(__name__)


class PressureType(Enum):
    """Pressure measurement types."""
    ABSOLUTE = "absolute"  # Absolute pressure
    GAUGE = "gauge"        # Gauge pressure (relative to atmospheric)
    DIFFERENTIAL = "differential"  # Pressure difference


class SensorType(Enum):
    """Pressure sensor types."""
    STRAIN_GAUGE = "strain_gauge"
    PIEZOELECTRIC = "piezoelectric"
    CAPACITIVE = "capacitive"
    RESONANT = "resonant"


@dataclass
class PressureSensorConfig(ConnectionConfig):
    """Pressure sensor configuration."""
    sensor_id: str = "pressure_sensor_1"
    sensor_type: SensorType = SensorType.STRAIN_GAUGE
    pressure_type: PressureType = PressureType.GAUGE

    # Range and units
    min_pressure: float = 0.0
    max_pressure: float = 20.0  # bar
    unit: str = "bar"

    # Sampling
    sampling_rate_hz: float = 1.0
    burst_mode_hz: Optional[float] = None  # For high-speed capture

    # Calibration
    calibration_factor: float = 1.0
    calibration_offset: float = 0.0
    zero_pressure_offset: float = 0.0

    # Validation
    max_pressure_change_rate: float = 5.0  # bar/s
    drift_detection_threshold: float = 0.1  # bar

    # Protocol
    modbus_address: int = 1
    register_address: int = 0


@dataclass
class PressureReading:
    """Pressure measurement reading."""
    timestamp: datetime
    pressure: float
    unit: str
    sensor_id: str
    pressure_type: str
    temperature: Optional[float] = None  # Sensor temp for compensation
    quality_score: float = 100.0
    raw_value: Optional[float] = None


class PressureSensorConnector(BaseConnector):
    """Multi-point pressure sensor connector."""

    def __init__(self, configs: List[PressureSensorConfig]):
        """
        Initialize pressure sensor connector.

        Args:
            configs: List of sensor configurations for multi-point monitoring
        """
        # Use first config for base connection
        super().__init__(configs[0] if configs else PressureSensorConfig(host="localhost", port=502))

        self.sensor_configs = {cfg.sensor_id: cfg for cfg in configs}
        self.current_readings: Dict[str, PressureReading] = {}
        self.reading_history: Dict[str, deque] = {
            sid: deque(maxlen=1000) for sid in self.sensor_configs.keys()
        }

        self._sampling_tasks: Dict[str, asyncio.Task] = {}
        self._drift_baselines: Dict[str, float] = {}

    async def _connect_impl(self) -> bool:
        """Connect to all pressure sensors."""
        try:
            # Simulate connection to pressure monitoring system
            self.connection = {
                'type': 'pressure_monitoring',
                'sensor_count': len(self.sensor_configs),
                'connected': True
            }

            # Start sampling for each sensor
            for sensor_id, config in self.sensor_configs.items():
                task = asyncio.create_task(
                    self._sensor_sampling_loop(sensor_id, config)
                )
                self._sampling_tasks[sensor_id] = task

                # Initialize drift baseline
                self._drift_baselines[sensor_id] = 0.0

            logger.info(f"Connected to {len(self.sensor_configs)} pressure sensors")
            return True

        except Exception as e:
            logger.error(f"Failed to connect pressure sensors: {e}")
            return False

    async def _disconnect_impl(self):
        """Disconnect from pressure sensors."""
        # Cancel all sampling tasks
        for task in self._sampling_tasks.values():
            task.cancel()
        self._sampling_tasks.clear()

        self.connection = None
        logger.info("Disconnected from pressure sensors")

    async def _health_check_impl(self) -> bool:
        """Health check for all sensors."""
        try:
            # Check if we have recent readings from all sensors
            cutoff = DeterministicClock.utcnow() - timedelta(seconds=30)

            for sensor_id in self.sensor_configs.keys():
                if sensor_id not in self.current_readings:
                    return False

                reading = self.current_readings[sensor_id]
                if reading.timestamp < cutoff:
                    return False

            return True

        except Exception:
            return False

    async def _sensor_sampling_loop(self, sensor_id: str, config: PressureSensorConfig):
        """Continuous sampling loop for a sensor."""
        interval = 1.0 / config.sampling_rate_hz

        while self.state == ConnectionState.CONNECTED:
            try:
                reading = await self._read_sensor(sensor_id, config)

                if reading:
                    # Validate and store
                    reading.quality_score = self._validate_reading(sensor_id, reading, config)

                    self.current_readings[sensor_id] = reading
                    self.reading_history[sensor_id].append(reading)

                    # Check for drift
                    await self._check_drift(sensor_id, reading, config)

                await asyncio.sleep(interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Sampling error for {sensor_id}: {e}")
                await asyncio.sleep(interval)

    async def _read_sensor(
        self,
        sensor_id: str,
        config: PressureSensorConfig
    ) -> PressureReading:
        """Read pressure from sensor."""
        # Simulate pressure reading based on sensor type
        # In production, would read from actual Modbus/OPC UA

        base_pressure = (config.min_pressure + config.max_pressure) / 2
        variation = (config.max_pressure - config.min_pressure) * 0.1

        raw_pressure = base_pressure + random.uniform(-variation, variation)

        # Apply calibration
        calibrated = (raw_pressure * config.calibration_factor) + \
                    config.calibration_offset + config.zero_pressure_offset

        return PressureReading(
            timestamp=DeterministicClock.utcnow(),
            pressure=calibrated,
            unit=config.unit,
            sensor_id=sensor_id,
            pressure_type=config.pressure_type.value,
            temperature=random.uniform(20, 25),  # Sensor temperature
            raw_value=raw_pressure
        )

    def _validate_reading(
        self,
        sensor_id: str,
        reading: PressureReading,
        config: PressureSensorConfig
    ) -> float:
        """Validate pressure reading."""
        quality = 100.0

        # Range check
        if reading.pressure < config.min_pressure or \
           reading.pressure > config.max_pressure:
            quality -= 50
            logger.warning(f"{sensor_id}: Pressure out of range: {reading.pressure}")

        # Rate of change check
        history = self.reading_history.get(sensor_id, [])
        if history:
            last_reading = history[-1]
            time_delta = (reading.timestamp - last_reading.timestamp).total_seconds()

            if time_delta > 0:
                pressure_change = abs(reading.pressure - last_reading.pressure)
                rate = pressure_change / time_delta

                if rate > config.max_pressure_change_rate:
                    quality -= 30
                    logger.warning(f"{sensor_id}: Excessive pressure change: {rate:.2f} bar/s")

        return max(0, quality)

    async def _check_drift(
        self,
        sensor_id: str,
        reading: PressureReading,
        config: PressureSensorConfig
    ):
        """Check for calibration drift."""
        history = self.reading_history.get(sensor_id, [])

        if len(history) < 100:
            return

        # Calculate baseline if not set
        if self._drift_baselines[sensor_id] == 0.0:
            baseline_readings = list(history)[-100:-50]
            self._drift_baselines[sensor_id] = sum(r.pressure for r in baseline_readings) / len(baseline_readings)

        # Check recent average against baseline
        recent = list(history)[-50:]
        recent_avg = sum(r.pressure for r in recent) / len(recent)

        drift = abs(recent_avg - self._drift_baselines[sensor_id])

        if drift > config.drift_detection_threshold:
            logger.warning(f"{sensor_id}: Drift detected: {drift:.3f} {config.unit}")

    def get_pressure(self, sensor_id: str) -> Optional[float]:
        """Get current pressure from sensor."""
        reading = self.current_readings.get(sensor_id)
        return reading.pressure if reading else None

    def get_all_pressures(self) -> Dict[str, float]:
        """Get pressures from all sensors."""
        return {
            sid: reading.pressure
            for sid, reading in self.current_readings.items()
        }

    def get_pressure_profile(self) -> List[Dict[str, Any]]:
        """Get pressure profile across all zones."""
        return [
            {
                'sensor_id': sid,
                'pressure': reading.pressure,
                'unit': reading.unit,
                'type': reading.pressure_type,
                'quality': reading.quality_score,
                'timestamp': reading.timestamp.isoformat()
            }
            for sid, reading in self.current_readings.items()
        ]

    def get_differential_pressure(self, sensor_id_high: str, sensor_id_low: str) -> Optional[float]:
        """Calculate differential pressure between two sensors."""
        high = self.current_readings.get(sensor_id_high)
        low = self.current_readings.get(sensor_id_low)

        if high and low:
            return high.pressure - low.pressure
        return None


async def main():
    """Example usage."""

    # Configure multi-point pressure monitoring
    configs = [
        PressureSensorConfig(
            host="192.168.1.100",
            port=502,
            sensor_id="header_pressure",
            min_pressure=0.0,
            max_pressure=20.0,
            sampling_rate_hz=1.0
        ),
        PressureSensorConfig(
            host="192.168.1.100",
            port=502,
            sensor_id="distribution_pressure",
            min_pressure=0.0,
            max_pressure=15.0,
            sampling_rate_hz=1.0
        )
    ]

    connector = PressureSensorConnector(configs)

    if await connector.connect():
        print("Connected to pressure sensors")

        await asyncio.sleep(5)

        # Get all pressures
        pressures = connector.get_all_pressures()
        print(f"\nCurrent Pressures:")
        for sid, pressure in pressures.items():
            print(f"  {sid}: {pressure:.2f} bar")

        # Get pressure profile
        profile = connector.get_pressure_profile()
        print(f"\nPressure Profile:")
        for sensor in profile:
            print(f"  {sensor['sensor_id']}: {sensor['pressure']:.2f} {sensor['unit']} (Q: {sensor['quality']:.0f}%)")

        await connector.disconnect()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
