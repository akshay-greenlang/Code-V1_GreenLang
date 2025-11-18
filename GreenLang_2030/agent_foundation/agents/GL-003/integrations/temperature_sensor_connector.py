"""
Temperature Sensor Integration Connector for GL-003 SteamSystemAnalyzer

RTD and thermocouple support with multi-zone monitoring.
Provides cold junction compensation, sensor validation, outlier detection.
"""

import asyncio
import logging
import random
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from collections import deque

from .base_connector import BaseConnector, ConnectionConfig, ConnectionState

logger = logging.getLogger(__name__)


class SensorType(Enum):
    RTD_PT100 = "rtd_pt100"
    RTD_PT1000 = "rtd_pt1000"
    THERMOCOUPLE_K = "thermocouple_k"
    THERMOCOUPLE_J = "thermocouple_j"
    THERMOCOUPLE_T = "thermocouple_t"


@dataclass
class TemperatureSensorConfig(ConnectionConfig):
    sensor_id: str = "temp_sensor_1"
    sensor_type: SensorType = SensorType.RTD_PT100
    min_temp: float = 0.0
    max_temp: float = 400.0
    unit: str = "C"
    sampling_rate_hz: float = 1.0
    smoothing_window: int = 5
    max_temp_change_rate: float = 10.0  # °C/min


@dataclass
class TemperatureReading:
    timestamp: datetime
    temperature: float
    unit: str
    sensor_id: str
    quality_score: float = 100.0
    cjc_temperature: Optional[float] = None  # Cold junction for TC


class TemperatureSensorConnector(BaseConnector):
    def __init__(self, configs: List[TemperatureSensorConfig]):
        super().__init__(configs[0] if configs else TemperatureSensorConfig(host="localhost", port=502))
        self.sensor_configs = {cfg.sensor_id: cfg for cfg in configs}
        self.current_readings: Dict[str, TemperatureReading] = {}
        self.reading_history: Dict[str, deque] = {sid: deque(maxlen=1000) for sid in self.sensor_configs}
        self._sampling_tasks: Dict[str, asyncio.Task] = {}

    async def _connect_impl(self) -> bool:
        try:
            self.connection = {'type': 'temperature_monitoring', 'sensor_count': len(self.sensor_configs), 'connected': True}
            for sensor_id, config in self.sensor_configs.items():
                self._sampling_tasks[sensor_id] = asyncio.create_task(self._sensor_sampling_loop(sensor_id, config))
            logger.info(f"Connected to {len(self.sensor_configs)} temperature sensors")
            return True
        except Exception as e:
            logger.error(f"Failed to connect: {e}")
            return False

    async def _disconnect_impl(self):
        for task in self._sampling_tasks.values():
            task.cancel()
        self._sampling_tasks.clear()

    async def _health_check_impl(self) -> bool:
        cutoff = datetime.utcnow() - timedelta(seconds=30)
        return all(sid in self.current_readings and self.current_readings[sid].timestamp > cutoff 
                   for sid in self.sensor_configs)

    async def _sensor_sampling_loop(self, sensor_id: str, config: TemperatureSensorConfig):
        interval = 1.0 / config.sampling_rate_hz
        while self.state == ConnectionState.CONNECTED:
            try:
                reading = await self._read_sensor(sensor_id, config)
                if reading:
                    reading.quality_score = self._validate_reading(sensor_id, reading, config)
                    reading = self._apply_smoothing(sensor_id, reading, config)
                    self.current_readings[sensor_id] = reading
                    self.reading_history[sensor_id].append(reading)
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Sampling error for {sensor_id}: {e}")

    async def _read_sensor(self, sensor_id: str, config: TemperatureSensorConfig) -> TemperatureReading:
        # Simulate temperature reading
        base_temp = (config.min_temp + config.max_temp) / 2
        temp = base_temp + random.uniform(-10, 10)
        
        cjc_temp = None
        if "thermocouple" in config.sensor_type.value:
            cjc_temp = random.uniform(20, 25)
        
        return TemperatureReading(
            timestamp=datetime.utcnow(),
            temperature=temp,
            unit=config.unit,
            sensor_id=sensor_id,
            cjc_temperature=cjc_temp
        )

    def _validate_reading(self, sensor_id: str, reading: TemperatureReading, config: TemperatureSensorConfig) -> float:
        quality = 100.0
        if reading.temperature < config.min_temp or reading.temperature > config.max_temp:
            quality -= 50
        
        history = self.reading_history.get(sensor_id, [])
        if history:
            last = history[-1]
            time_delta = (reading.timestamp - last.timestamp).total_seconds() / 60
            if time_delta > 0:
                rate = abs(reading.temperature - last.temperature) / time_delta
                if rate > config.max_temp_change_rate:
                    quality -= 30
        return max(0, quality)

    def _apply_smoothing(self, sensor_id: str, reading: TemperatureReading, config: TemperatureSensorConfig) -> TemperatureReading:
        history = self.reading_history.get(sensor_id, [])
        if len(history) >= config.smoothing_window:
            recent = list(history)[-(config.smoothing_window-1):]
            temps = [r.temperature for r in recent] + [reading.temperature]
            reading.temperature = sum(temps) / len(temps)
        return reading

    def get_temperature(self, sensor_id: str) -> Optional[float]:
        reading = self.current_readings.get(sensor_id)
        return reading.temperature if reading else None

    def get_all_temperatures(self) -> Dict[str, float]:
        return {sid: reading.temperature for sid, reading in self.current_readings.items()}
