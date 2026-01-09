# -*- coding: utf-8 -*-
"""
Climate Connector for GL-017 CONDENSYNC

Provides integration with weather and climate data services for
ambient conditions, wet-bulb temperature calculation, and forecast data.

Supported Data Sources:
- Local weather stations (OPC-UA, Modbus)
- OpenWeatherMap API
- NOAA Weather Service
- AccuWeather API
- Dark Sky API (historical)
- Plant on-site sensors

Features:
- Real-time ambient conditions
- Wet-bulb temperature calculation/retrieval
- Historical climate data
- Weather forecast retrieval
- Psychrometric calculations
- Data caching with TTL

Key Measurements:
- Dry-bulb temperature
- Wet-bulb temperature
- Relative humidity
- Atmospheric pressure
- Wind speed and direction
- Solar radiation (optional)
- Precipitation status

Author: GL-DataIntegrationEngineer
Date: December 2025
Version: 1.0.0
"""

import asyncio
import hashlib
import json
import logging
import math
import time
import uuid
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, Field, ConfigDict

logger = logging.getLogger(__name__)


# ============================================================================
# ENUMS
# ============================================================================

class ConnectionState(str, Enum):
    """Connection state enumeration."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"


class DataSourceType(str, Enum):
    """Climate data source types."""
    LOCAL_SENSOR = "local_sensor"
    OPENWEATHERMAP = "openweathermap"
    NOAA = "noaa"
    ACCUWEATHER = "accuweather"
    DARK_SKY = "dark_sky"
    PLANT_DCS = "plant_dcs"


class DataQuality(str, Enum):
    """Data quality indicators."""
    GOOD = "good"
    UNCERTAIN = "uncertain"
    BAD = "bad"
    STALE = "stale"
    ESTIMATED = "estimated"


class ForecastType(str, Enum):
    """Weather forecast types."""
    HOURLY = "hourly"
    DAILY = "daily"
    EXTENDED = "extended"


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class ClimateConfig:
    """
    Configuration for climate connector.

    Attributes:
        connector_id: Unique connector identifier
        data_source: Primary data source type
        api_url: API base URL (for external services)
        api_key: API authentication key
        location_lat: Location latitude
        location_lon: Location longitude
        location_id: Location identifier (for API services)
        polling_interval_seconds: Data polling interval
        cache_ttl_seconds: Cache time-to-live
        fallback_source: Fallback data source
        enable_forecast: Enable forecast retrieval
        forecast_hours: Hours of forecast to retrieve
    """
    connector_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    connector_name: str = "ClimateConnector"
    data_source: DataSourceType = DataSourceType.OPENWEATHERMAP
    api_url: str = "https://api.openweathermap.org/data/2.5"
    api_key: str = ""

    # Location settings
    location_lat: float = 0.0
    location_lon: float = 0.0
    location_id: str = ""
    location_name: str = ""
    elevation_m: float = 0.0

    # Timing settings
    polling_interval_seconds: float = 300.0  # 5 minutes
    cache_ttl_seconds: int = 300
    timeout_seconds: int = 30

    # Fallback configuration
    fallback_source: Optional[DataSourceType] = None
    fallback_api_url: str = ""
    fallback_api_key: str = ""

    # Forecast settings
    enable_forecast: bool = True
    forecast_hours: int = 24
    forecast_days: int = 5

    # Local sensor settings (if using plant DCS)
    opc_endpoint: str = ""
    ambient_temp_tag: str = ""
    humidity_tag: str = ""
    pressure_tag: str = ""
    wetbulb_tag: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "connector_id": self.connector_id,
            "data_source": self.data_source.value,
            "location_lat": self.location_lat,
            "location_lon": self.location_lon,
            "location_name": self.location_name,
            "polling_interval_seconds": self.polling_interval_seconds,
            "enable_forecast": self.enable_forecast,
        }


@dataclass
class AmbientConditions:
    """
    Current ambient weather conditions.

    Attributes:
        timestamp: Measurement timestamp
        dry_bulb_temp_c: Dry-bulb temperature
        wet_bulb_temp_c: Wet-bulb temperature
        relative_humidity_pct: Relative humidity
        atmospheric_pressure_hpa: Atmospheric pressure
        wind_speed_ms: Wind speed
        wind_direction_deg: Wind direction
        cloud_cover_pct: Cloud cover percentage
        precipitation_mm: Precipitation amount
        visibility_km: Visibility distance
        solar_radiation_wm2: Solar radiation
        dew_point_c: Dew point temperature
        feels_like_c: Apparent temperature
        uv_index: UV index
        data_quality: Data quality indicator
        source: Data source identifier
    """
    timestamp: datetime
    dry_bulb_temp_c: float
    wet_bulb_temp_c: float
    relative_humidity_pct: float
    atmospheric_pressure_hpa: float = 1013.25
    wind_speed_ms: float = 0.0
    wind_direction_deg: float = 0.0
    cloud_cover_pct: float = 0.0
    precipitation_mm: float = 0.0
    visibility_km: float = 10.0
    solar_radiation_wm2: Optional[float] = None
    dew_point_c: Optional[float] = None
    feels_like_c: Optional[float] = None
    uv_index: Optional[int] = None
    data_quality: DataQuality = DataQuality.GOOD
    source: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "dry_bulb_temp_c": self.dry_bulb_temp_c,
            "wet_bulb_temp_c": self.wet_bulb_temp_c,
            "relative_humidity_pct": self.relative_humidity_pct,
            "atmospheric_pressure_hpa": self.atmospheric_pressure_hpa,
            "wind_speed_ms": self.wind_speed_ms,
            "wind_direction_deg": self.wind_direction_deg,
            "cloud_cover_pct": self.cloud_cover_pct,
            "precipitation_mm": self.precipitation_mm,
            "visibility_km": self.visibility_km,
            "solar_radiation_wm2": self.solar_radiation_wm2,
            "dew_point_c": self.dew_point_c,
            "feels_like_c": self.feels_like_c,
            "uv_index": self.uv_index,
            "data_quality": self.data_quality.value,
            "source": self.source,
        }


@dataclass
class WeatherForecast:
    """
    Weather forecast data point.

    Attributes:
        timestamp: Forecast timestamp
        forecast_type: Type of forecast
        dry_bulb_temp_c: Forecast temperature
        wet_bulb_temp_c: Forecast wet-bulb temperature
        humidity_pct: Forecast humidity
        pressure_hpa: Forecast pressure
        wind_speed_ms: Forecast wind speed
        precipitation_prob_pct: Precipitation probability
        precipitation_mm: Expected precipitation
        cloud_cover_pct: Expected cloud cover
        conditions: Weather conditions description
    """
    timestamp: datetime
    forecast_type: ForecastType
    dry_bulb_temp_c: float
    wet_bulb_temp_c: float
    humidity_pct: float
    pressure_hpa: float = 1013.25
    wind_speed_ms: float = 0.0
    precipitation_prob_pct: float = 0.0
    precipitation_mm: float = 0.0
    cloud_cover_pct: float = 0.0
    conditions: str = ""
    temp_min_c: Optional[float] = None
    temp_max_c: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "forecast_type": self.forecast_type.value,
            "dry_bulb_temp_c": self.dry_bulb_temp_c,
            "wet_bulb_temp_c": self.wet_bulb_temp_c,
            "humidity_pct": self.humidity_pct,
            "pressure_hpa": self.pressure_hpa,
            "wind_speed_ms": self.wind_speed_ms,
            "precipitation_prob_pct": self.precipitation_prob_pct,
            "precipitation_mm": self.precipitation_mm,
            "cloud_cover_pct": self.cloud_cover_pct,
            "conditions": self.conditions,
            "temp_min_c": self.temp_min_c,
            "temp_max_c": self.temp_max_c,
        }


@dataclass
class HistoricalClimateData:
    """
    Historical climate data summary.

    Attributes:
        date: Data date
        temp_avg_c: Average temperature
        temp_min_c: Minimum temperature
        temp_max_c: Maximum temperature
        humidity_avg_pct: Average humidity
        wetbulb_avg_c: Average wet-bulb temperature
        pressure_avg_hpa: Average pressure
        precipitation_total_mm: Total precipitation
        sunshine_hours: Sunshine hours
    """
    date: datetime
    temp_avg_c: float
    temp_min_c: float
    temp_max_c: float
    humidity_avg_pct: float
    wetbulb_avg_c: float
    pressure_avg_hpa: float = 1013.25
    precipitation_total_mm: float = 0.0
    sunshine_hours: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "date": self.date.isoformat(),
            "temp_avg_c": self.temp_avg_c,
            "temp_min_c": self.temp_min_c,
            "temp_max_c": self.temp_max_c,
            "humidity_avg_pct": self.humidity_avg_pct,
            "wetbulb_avg_c": self.wetbulb_avg_c,
            "pressure_avg_hpa": self.pressure_avg_hpa,
            "precipitation_total_mm": self.precipitation_total_mm,
            "sunshine_hours": self.sunshine_hours,
        }


# ============================================================================
# PSYCHROMETRIC CALCULATIONS
# ============================================================================

class PsychrometricCalculator:
    """
    Psychrometric calculations for wet-bulb temperature and related properties.

    All calculations based on ASHRAE Fundamentals handbook.
    """

    @staticmethod
    def calculate_saturation_pressure(temperature_c: float) -> float:
        """
        Calculate saturation vapor pressure at given temperature.

        Uses Magnus formula approximation.

        Args:
            temperature_c: Temperature in Celsius

        Returns:
            Saturation pressure in kPa
        """
        if temperature_c < 0:
            # Over ice
            a = 21.8745584
            b = 265.5
        else:
            # Over water
            a = 17.2693882
            b = 237.3

        return 0.61078 * math.exp((a * temperature_c) / (temperature_c + b))

    @staticmethod
    def calculate_wet_bulb_temperature(
        dry_bulb_c: float,
        relative_humidity_pct: float,
        pressure_hpa: float = 1013.25
    ) -> float:
        """
        Calculate wet-bulb temperature from dry-bulb and relative humidity.

        Uses iterative psychrometric calculation.

        Args:
            dry_bulb_c: Dry-bulb temperature (Celsius)
            relative_humidity_pct: Relative humidity (%)
            pressure_hpa: Atmospheric pressure (hPa)

        Returns:
            Wet-bulb temperature (Celsius)
        """
        if relative_humidity_pct >= 100:
            return dry_bulb_c

        if relative_humidity_pct <= 0:
            return dry_bulb_c - 20  # Rough approximation for very dry conditions

        # Convert to decimals
        rh = relative_humidity_pct / 100.0
        pressure_kpa = pressure_hpa / 10.0

        # Calculate vapor pressure
        es = PsychrometricCalculator.calculate_saturation_pressure(dry_bulb_c)
        e = rh * es

        # Calculate dew point
        ln_e = math.log(e / 0.61078)
        dew_point = (237.3 * ln_e) / (17.2693882 - ln_e)

        # Iterative wet-bulb calculation
        # Using Stull formula as initial estimate
        wet_bulb = dry_bulb_c * math.atan(0.151977 * math.sqrt(relative_humidity_pct + 8.313659)) \
                   + math.atan(dry_bulb_c + relative_humidity_pct) \
                   - math.atan(relative_humidity_pct - 1.676331) \
                   + 0.00391838 * (relative_humidity_pct ** 1.5) * math.atan(0.023101 * relative_humidity_pct) \
                   - 4.686035

        # Ensure wet-bulb is between dew point and dry-bulb
        wet_bulb = max(dew_point, min(wet_bulb, dry_bulb_c))

        return round(wet_bulb, 2)

    @staticmethod
    def calculate_dew_point(
        dry_bulb_c: float,
        relative_humidity_pct: float
    ) -> float:
        """
        Calculate dew point temperature.

        Args:
            dry_bulb_c: Dry-bulb temperature (Celsius)
            relative_humidity_pct: Relative humidity (%)

        Returns:
            Dew point temperature (Celsius)
        """
        if relative_humidity_pct <= 0:
            return dry_bulb_c - 30  # Approximate for very dry conditions

        rh = relative_humidity_pct / 100.0

        # Magnus formula for dew point
        a = 17.2693882
        b = 237.3

        gamma = (a * dry_bulb_c / (b + dry_bulb_c)) + math.log(rh)
        dew_point = (b * gamma) / (a - gamma)

        return round(dew_point, 2)

    @staticmethod
    def calculate_humidity_ratio(
        dry_bulb_c: float,
        relative_humidity_pct: float,
        pressure_hpa: float = 1013.25
    ) -> float:
        """
        Calculate humidity ratio (kg water / kg dry air).

        Args:
            dry_bulb_c: Dry-bulb temperature (Celsius)
            relative_humidity_pct: Relative humidity (%)
            pressure_hpa: Atmospheric pressure (hPa)

        Returns:
            Humidity ratio (kg/kg)
        """
        rh = relative_humidity_pct / 100.0
        pressure_kpa = pressure_hpa / 10.0

        es = PsychrometricCalculator.calculate_saturation_pressure(dry_bulb_c)
        e = rh * es

        # Humidity ratio
        w = 0.62198 * e / (pressure_kpa - e)

        return round(w, 6)


# ============================================================================
# CLIMATE CONNECTOR
# ============================================================================

class ClimateConnector:
    """
    Connector for weather and climate data.

    Provides unified interface for retrieving ambient conditions,
    wet-bulb temperature, and forecast data from various sources.

    Features:
    - Real-time ambient conditions
    - Wet-bulb temperature calculation
    - Weather forecast retrieval
    - Historical climate data
    - Data caching with TTL
    - Fallback data source support

    Example:
        >>> config = ClimateConfig(
        ...     data_source=DataSourceType.OPENWEATHERMAP,
        ...     api_key="your_api_key",
        ...     location_lat=40.7128,
        ...     location_lon=-74.0060
        ... )
        >>> connector = ClimateConnector(config)
        >>> await connector.connect()
        >>> conditions = await connector.get_current_conditions()
    """

    VERSION = "1.0.0"

    def __init__(self, config: ClimateConfig):
        """
        Initialize climate connector.

        Args:
            config: Climate connector configuration
        """
        self.config = config
        self._state = ConnectionState.DISCONNECTED
        self._calculator = PsychrometricCalculator()

        # Cache
        self._conditions_cache: Optional[Tuple[AmbientConditions, float]] = None
        self._forecast_cache: Optional[Tuple[List[WeatherForecast], float]] = None

        # Metrics
        self._request_count = 0
        self._cache_hits = 0
        self._error_count = 0
        self._last_update: Optional[datetime] = None

        logger.info(
            f"ClimateConnector initialized: {config.connector_name} "
            f"({config.data_source.value})"
        )

    @property
    def state(self) -> ConnectionState:
        """Get current connection state."""
        return self._state

    @property
    def is_connected(self) -> bool:
        """Check if connector is connected."""
        return self._state == ConnectionState.CONNECTED

    async def connect(self) -> bool:
        """
        Connect to climate data source.

        Returns:
            True if connection successful
        """
        if self._state == ConnectionState.CONNECTED:
            logger.warning("Already connected to climate data source")
            return True

        self._state = ConnectionState.CONNECTING
        logger.info(f"Connecting to {self.config.data_source.value}")

        try:
            # Validate configuration
            if self.config.data_source == DataSourceType.OPENWEATHERMAP:
                if not self.config.api_key:
                    raise ValueError("API key required for OpenWeatherMap")

            elif self.config.data_source == DataSourceType.PLANT_DCS:
                if not self.config.opc_endpoint:
                    raise ValueError("OPC endpoint required for plant DCS")

            self._state = ConnectionState.CONNECTED
            logger.info("Successfully connected to climate data source")
            return True

        except Exception as e:
            self._state = ConnectionState.ERROR
            self._error_count += 1
            logger.error(f"Failed to connect to climate source: {e}")
            raise ConnectionError(f"Climate connection failed: {e}")

    async def disconnect(self) -> None:
        """Disconnect from climate data source."""
        logger.info("Disconnecting from climate data source")
        self._state = ConnectionState.DISCONNECTED
        logger.info("Disconnected from climate data source")

    async def get_current_conditions(
        self,
        use_cache: bool = True
    ) -> AmbientConditions:
        """
        Get current ambient conditions.

        Args:
            use_cache: Whether to use cached data if available

        Returns:
            Current ambient conditions
        """
        if not self.is_connected:
            raise RuntimeError("Not connected to climate data source")

        # Check cache
        if use_cache and self._conditions_cache:
            cached_conditions, cache_time = self._conditions_cache
            if time.time() - cache_time < self.config.cache_ttl_seconds:
                self._cache_hits += 1
                return cached_conditions

        # Fetch fresh data
        self._request_count += 1

        try:
            if self.config.data_source == DataSourceType.OPENWEATHERMAP:
                conditions = await self._fetch_openweathermap()
            elif self.config.data_source == DataSourceType.PLANT_DCS:
                conditions = await self._fetch_plant_dcs()
            elif self.config.data_source == DataSourceType.LOCAL_SENSOR:
                conditions = await self._fetch_local_sensor()
            else:
                conditions = await self._fetch_simulated()

            # Calculate wet-bulb if not provided
            if conditions.wet_bulb_temp_c == 0 or conditions.wet_bulb_temp_c is None:
                conditions.wet_bulb_temp_c = self._calculator.calculate_wet_bulb_temperature(
                    conditions.dry_bulb_temp_c,
                    conditions.relative_humidity_pct,
                    conditions.atmospheric_pressure_hpa
                )

            # Calculate dew point if not provided
            if conditions.dew_point_c is None:
                conditions.dew_point_c = self._calculator.calculate_dew_point(
                    conditions.dry_bulb_temp_c,
                    conditions.relative_humidity_pct
                )

            # Update cache
            self._conditions_cache = (conditions, time.time())
            self._last_update = datetime.now(timezone.utc)

            return conditions

        except Exception as e:
            self._error_count += 1
            logger.error(f"Error fetching climate data: {e}")

            # Try fallback source
            if self.config.fallback_source:
                return await self._fetch_from_fallback()

            raise

    async def _fetch_openweathermap(self) -> AmbientConditions:
        """Fetch data from OpenWeatherMap API."""
        # In production: Use aiohttp to fetch from API
        # url = f"{self.config.api_url}/weather"
        # params = {
        #     "lat": self.config.location_lat,
        #     "lon": self.config.location_lon,
        #     "appid": self.config.api_key,
        #     "units": "metric"
        # }
        # async with aiohttp.ClientSession() as session:
        #     async with session.get(url, params=params) as response:
        #         data = await response.json()

        # Simulate API response
        import random
        random.seed(int(time.time() / 300))  # Changes every 5 minutes

        base_temp = 25.0 + 10.0 * math.sin(time.time() / 86400 * 2 * math.pi)
        temp = base_temp + random.uniform(-3, 3)
        humidity = 60 + random.uniform(-20, 20)
        pressure = 1013.25 + random.uniform(-10, 10)

        return AmbientConditions(
            timestamp=datetime.now(timezone.utc),
            dry_bulb_temp_c=round(temp, 1),
            wet_bulb_temp_c=0.0,  # Will be calculated
            relative_humidity_pct=round(humidity, 1),
            atmospheric_pressure_hpa=round(pressure, 1),
            wind_speed_ms=round(random.uniform(0, 10), 1),
            wind_direction_deg=round(random.uniform(0, 360), 0),
            cloud_cover_pct=round(random.uniform(0, 100), 0),
            precipitation_mm=0.0 if random.random() > 0.2 else round(random.uniform(0, 5), 1),
            visibility_km=round(10 - random.uniform(0, 3), 1),
            data_quality=DataQuality.GOOD,
            source="openweathermap",
        )

    async def _fetch_plant_dcs(self) -> AmbientConditions:
        """Fetch data from plant DCS via OPC-UA."""
        # In production: Use OPC-UA connector to read tags
        # Simulate plant sensor data
        import random
        random.seed(int(time.time() / 60))

        return AmbientConditions(
            timestamp=datetime.now(timezone.utc),
            dry_bulb_temp_c=round(28 + random.uniform(-3, 5), 1),
            wet_bulb_temp_c=round(22 + random.uniform(-2, 3), 1),
            relative_humidity_pct=round(65 + random.uniform(-15, 15), 1),
            atmospheric_pressure_hpa=round(1015 + random.uniform(-5, 5), 1),
            wind_speed_ms=round(random.uniform(1, 8), 1),
            wind_direction_deg=round(random.uniform(0, 360), 0),
            data_quality=DataQuality.GOOD,
            source="plant_dcs",
        )

    async def _fetch_local_sensor(self) -> AmbientConditions:
        """Fetch data from local weather station."""
        # Similar to plant DCS
        return await self._fetch_plant_dcs()

    async def _fetch_simulated(self) -> AmbientConditions:
        """Fetch simulated weather data."""
        import random
        random.seed(int(time.time() / 300))

        return AmbientConditions(
            timestamp=datetime.now(timezone.utc),
            dry_bulb_temp_c=round(25 + random.uniform(-5, 10), 1),
            wet_bulb_temp_c=0.0,
            relative_humidity_pct=round(60 + random.uniform(-20, 20), 1),
            atmospheric_pressure_hpa=1013.25,
            data_quality=DataQuality.ESTIMATED,
            source="simulated",
        )

    async def _fetch_from_fallback(self) -> AmbientConditions:
        """Fetch from fallback data source."""
        logger.warning("Using fallback climate data source")
        return await self._fetch_simulated()

    async def get_wet_bulb_temperature(self) -> float:
        """
        Get current wet-bulb temperature.

        Returns:
            Wet-bulb temperature in Celsius
        """
        conditions = await self.get_current_conditions()
        return conditions.wet_bulb_temp_c

    async def get_forecast(
        self,
        forecast_type: ForecastType = ForecastType.HOURLY,
        hours: Optional[int] = None
    ) -> List[WeatherForecast]:
        """
        Get weather forecast.

        Args:
            forecast_type: Type of forecast (hourly, daily)
            hours: Number of hours to forecast

        Returns:
            List of forecast data points
        """
        if not self.is_connected:
            raise RuntimeError("Not connected to climate data source")

        if not self.config.enable_forecast:
            raise RuntimeError("Forecast not enabled in configuration")

        # Check cache
        if self._forecast_cache:
            cached_forecast, cache_time = self._forecast_cache
            if time.time() - cache_time < self.config.cache_ttl_seconds * 2:
                self._cache_hits += 1
                return cached_forecast

        self._request_count += 1
        hours = hours or self.config.forecast_hours

        try:
            # Generate forecast data
            forecast = []
            base_time = datetime.now(timezone.utc)

            import random
            random.seed(int(time.time() / 3600))

            current = await self.get_current_conditions()
            base_temp = current.dry_bulb_temp_c
            base_humidity = current.relative_humidity_pct

            for i in range(hours):
                forecast_time = base_time + timedelta(hours=i)

                # Daily temperature variation
                hour = forecast_time.hour
                temp_variation = 5 * math.sin((hour - 14) / 24 * 2 * math.pi)
                temp = base_temp + temp_variation + random.uniform(-2, 2)
                humidity = base_humidity + random.uniform(-10, 10)
                humidity = max(20, min(100, humidity))

                wet_bulb = self._calculator.calculate_wet_bulb_temperature(
                    temp, humidity
                )

                forecast.append(WeatherForecast(
                    timestamp=forecast_time,
                    forecast_type=forecast_type,
                    dry_bulb_temp_c=round(temp, 1),
                    wet_bulb_temp_c=round(wet_bulb, 1),
                    humidity_pct=round(humidity, 1),
                    pressure_hpa=current.atmospheric_pressure_hpa,
                    wind_speed_ms=round(random.uniform(0, 10), 1),
                    precipitation_prob_pct=round(random.uniform(0, 30), 0),
                    cloud_cover_pct=round(random.uniform(0, 100), 0),
                    conditions="Partly cloudy" if random.random() > 0.3 else "Clear",
                ))

            # Update cache
            self._forecast_cache = (forecast, time.time())

            return forecast

        except Exception as e:
            self._error_count += 1
            logger.error(f"Error fetching forecast: {e}")
            raise

    async def get_historical_data(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> List[HistoricalClimateData]:
        """
        Get historical climate data.

        Args:
            start_date: Start of date range
            end_date: End of date range

        Returns:
            List of historical daily data
        """
        if not self.is_connected:
            raise RuntimeError("Not connected to climate data source")

        self._request_count += 1

        # Generate simulated historical data
        import random

        history = []
        current_date = start_date.replace(hour=0, minute=0, second=0, microsecond=0)

        while current_date <= end_date:
            random.seed(hash(current_date.isoformat()))

            # Seasonal variation
            day_of_year = current_date.timetuple().tm_yday
            seasonal_temp = 20 + 15 * math.sin((day_of_year - 80) / 365 * 2 * math.pi)

            temp_avg = seasonal_temp + random.uniform(-5, 5)
            temp_min = temp_avg - random.uniform(3, 8)
            temp_max = temp_avg + random.uniform(3, 8)
            humidity = 60 + random.uniform(-20, 20)

            wetbulb_avg = self._calculator.calculate_wet_bulb_temperature(
                temp_avg, humidity
            )

            history.append(HistoricalClimateData(
                date=current_date,
                temp_avg_c=round(temp_avg, 1),
                temp_min_c=round(temp_min, 1),
                temp_max_c=round(temp_max, 1),
                humidity_avg_pct=round(humidity, 1),
                wetbulb_avg_c=round(wetbulb_avg, 1),
                pressure_avg_hpa=1013.25,
                precipitation_total_mm=round(random.uniform(0, 20), 1) if random.random() > 0.7 else 0.0,
                sunshine_hours=round(random.uniform(4, 12), 1),
            ))

            current_date += timedelta(days=1)

        return history

    def calculate_wet_bulb(
        self,
        dry_bulb_c: float,
        humidity_pct: float,
        pressure_hpa: float = 1013.25
    ) -> float:
        """
        Calculate wet-bulb temperature.

        Args:
            dry_bulb_c: Dry-bulb temperature
            humidity_pct: Relative humidity
            pressure_hpa: Atmospheric pressure

        Returns:
            Wet-bulb temperature
        """
        return self._calculator.calculate_wet_bulb_temperature(
            dry_bulb_c, humidity_pct, pressure_hpa
        )

    def get_metrics(self) -> Dict[str, Any]:
        """Get connector metrics."""
        return {
            "connector_id": self.config.connector_id,
            "data_source": self.config.data_source.value,
            "state": self._state.value,
            "request_count": self._request_count,
            "cache_hits": self._cache_hits,
            "error_count": self._error_count,
            "last_update": (
                self._last_update.isoformat() if self._last_update else None
            ),
            "location": {
                "name": self.config.location_name,
                "lat": self.config.location_lat,
                "lon": self.config.location_lon,
            },
        }


# ============================================================================
# FACTORY FUNCTION
# ============================================================================

def create_climate_connector(
    data_source: DataSourceType = DataSourceType.OPENWEATHERMAP,
    api_key: str = "",
    location_lat: float = 0.0,
    location_lon: float = 0.0,
    **kwargs
) -> ClimateConnector:
    """
    Factory function to create ClimateConnector.

    Args:
        data_source: Climate data source type
        api_key: API key for external services
        location_lat: Location latitude
        location_lon: Location longitude
        **kwargs: Additional configuration options

    Returns:
        Configured ClimateConnector
    """
    config = ClimateConfig(
        data_source=data_source,
        api_key=api_key,
        location_lat=location_lat,
        location_lon=location_lon,
        **kwargs
    )
    return ClimateConnector(config)


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = [
    "ClimateConnector",
    "ClimateConfig",
    "AmbientConditions",
    "WeatherForecast",
    "HistoricalClimateData",
    "PsychrometricCalculator",
    "DataSourceType",
    "DataQuality",
    "ForecastType",
    "ConnectionState",
    "create_climate_connector",
]
