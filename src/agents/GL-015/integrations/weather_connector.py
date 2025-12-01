"""
Weather Connector Module for GL-015 INSULSCAN (Insulation Inspection Agent).

Provides integration with weather data services for insulation inspection planning:
- OpenWeatherMap API - Current conditions, forecasts
- NOAA Weather Service - US weather data
- Historical weather data for analysis
- Forecast data for inspection scheduling
- Wind speed, temperature, solar radiation data

Weather data is critical for thermal imaging accuracy and inspection planning.

Author: GL-DataIntegrationEngineer
Version: 1.0.0
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Tuple,
)
import asyncio
import logging
import uuid

import aiohttp
from pydantic import BaseModel, Field, ConfigDict, field_validator

from .base_connector import (
    BaseConnector,
    BaseConnectorConfig,
    ConfigurationError,
    ConnectionError,
    ConnectionState,
    ConnectorError,
    ConnectorType,
    HealthCheckResult,
    HealthStatus,
    with_retry,
)

# Configure module logger
logger = logging.getLogger(__name__)


# =============================================================================
# Enumerations
# =============================================================================


class WeatherProvider(str, Enum):
    """Supported weather data providers."""

    OPENWEATHERMAP = "openweathermap"  # OpenWeatherMap API
    NOAA = "noaa"  # NOAA National Weather Service
    WEATHER_API = "weather_api"  # WeatherAPI.com
    VISUAL_CROSSING = "visual_crossing"  # Visual Crossing
    METEOSTAT = "meteostat"  # Meteostat historical data


class WeatherCondition(str, Enum):
    """Weather condition classifications."""

    CLEAR = "clear"
    PARTLY_CLOUDY = "partly_cloudy"
    CLOUDY = "cloudy"
    OVERCAST = "overcast"
    FOG = "fog"
    MIST = "mist"
    LIGHT_RAIN = "light_rain"
    RAIN = "rain"
    HEAVY_RAIN = "heavy_rain"
    THUNDERSTORM = "thunderstorm"
    SNOW = "snow"
    SLEET = "sleet"
    HAIL = "hail"
    WINDY = "windy"
    UNKNOWN = "unknown"


class InspectionSuitability(str, Enum):
    """Weather suitability for thermal inspection."""

    EXCELLENT = "excellent"  # Ideal conditions
    GOOD = "good"  # Suitable with minor considerations
    MARGINAL = "marginal"  # May affect accuracy
    POOR = "poor"  # Not recommended
    UNSUITABLE = "unsuitable"  # Do not inspect


class WindDirection(str, Enum):
    """Wind direction."""

    N = "N"
    NNE = "NNE"
    NE = "NE"
    ENE = "ENE"
    E = "E"
    ESE = "ESE"
    SE = "SE"
    SSE = "SSE"
    S = "S"
    SSW = "SSW"
    SW = "SW"
    WSW = "WSW"
    W = "W"
    WNW = "WNW"
    NW = "NW"
    NNW = "NNW"


# =============================================================================
# Pydantic Models - Configuration
# =============================================================================


class OpenWeatherMapConfig(BaseModel):
    """OpenWeatherMap API configuration."""

    model_config = ConfigDict(extra="forbid")

    api_key: str = Field(..., description="OpenWeatherMap API key")
    api_version: str = Field(default="2.5", description="API version")
    base_url: str = Field(
        default="https://api.openweathermap.org/data",
        description="API base URL"
    )
    units: str = Field(
        default="metric",
        pattern="^(metric|imperial|standard)$",
        description="Units (metric, imperial, standard)"
    )
    language: str = Field(default="en", description="Response language")


class NOAAConfig(BaseModel):
    """NOAA Weather Service configuration."""

    model_config = ConfigDict(extra="forbid")

    base_url: str = Field(
        default="https://api.weather.gov",
        description="NOAA API base URL"
    )
    user_agent: str = Field(
        default="GL-015-INSULSCAN/1.0",
        description="User agent for NOAA API"
    )
    default_office: Optional[str] = Field(
        default=None,
        description="Default forecast office"
    )


class LocationConfig(BaseModel):
    """Location configuration for weather data."""

    model_config = ConfigDict(extra="forbid")

    latitude: float = Field(
        ...,
        ge=-90,
        le=90,
        description="Latitude"
    )
    longitude: float = Field(
        ...,
        ge=-180,
        le=180,
        description="Longitude"
    )
    location_name: Optional[str] = Field(
        default=None,
        description="Location name"
    )
    timezone: str = Field(
        default="UTC",
        description="Timezone"
    )
    elevation_m: Optional[float] = Field(
        default=None,
        description="Elevation in meters"
    )


class WeatherConnectorConfig(BaseConnectorConfig):
    """Configuration for weather connector."""

    model_config = ConfigDict(extra="forbid")

    # Provider settings
    provider: WeatherProvider = Field(
        default=WeatherProvider.OPENWEATHERMAP,
        description="Weather data provider"
    )

    # Provider-specific configurations
    openweathermap_config: Optional[OpenWeatherMapConfig] = Field(
        default=None,
        description="OpenWeatherMap configuration"
    )
    noaa_config: Optional[NOAAConfig] = Field(
        default=None,
        description="NOAA configuration"
    )

    # Default location
    default_location: Optional[LocationConfig] = Field(
        default=None,
        description="Default location for weather data"
    )

    # Forecast settings
    forecast_days: int = Field(
        default=7,
        ge=1,
        le=16,
        description="Number of forecast days"
    )
    forecast_interval_hours: int = Field(
        default=3,
        ge=1,
        le=24,
        description="Forecast interval in hours"
    )

    # Historical data settings
    historical_data_enabled: bool = Field(
        default=True,
        description="Enable historical data retrieval"
    )
    historical_days_limit: int = Field(
        default=30,
        ge=1,
        le=365,
        description="Maximum historical days to retrieve"
    )

    # Inspection suitability thresholds
    max_wind_speed_m_s: float = Field(
        default=10.0,
        ge=0,
        description="Max wind speed for inspection (m/s)"
    )
    min_delta_t_c: float = Field(
        default=10.0,
        ge=0,
        description="Min temperature difference for thermal imaging"
    )
    max_humidity_percent: float = Field(
        default=90.0,
        ge=0,
        le=100,
        description="Max humidity for inspection"
    )
    rain_threshold_mm: float = Field(
        default=0.5,
        ge=0,
        description="Rain threshold (mm/h)"
    )

    def __init__(self, **data):
        """Initialize with connector type set."""
        data['connector_type'] = ConnectorType.WEATHER_SERVICE
        super().__init__(**data)


# =============================================================================
# Data Models - Weather Data
# =============================================================================


class CurrentWeather(BaseModel):
    """Current weather conditions."""

    model_config = ConfigDict(frozen=True)

    # Location
    location_name: Optional[str] = Field(default=None)
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    elevation_m: Optional[float] = Field(default=None)

    # Timestamp
    observation_time: datetime = Field(
        default_factory=datetime.utcnow,
        description="Observation timestamp"
    )
    timezone: str = Field(default="UTC")

    # Temperature
    temperature_c: float = Field(..., description="Current temperature")
    feels_like_c: float = Field(..., description="Feels like temperature")
    temperature_min_c: Optional[float] = Field(
        default=None,
        description="Today's minimum"
    )
    temperature_max_c: Optional[float] = Field(
        default=None,
        description="Today's maximum"
    )
    dew_point_c: Optional[float] = Field(
        default=None,
        description="Dew point temperature"
    )

    # Humidity and pressure
    humidity_percent: float = Field(
        ...,
        ge=0,
        le=100,
        description="Relative humidity"
    )
    pressure_hpa: float = Field(..., ge=0, description="Atmospheric pressure")

    # Wind
    wind_speed_m_s: float = Field(
        ...,
        ge=0,
        description="Wind speed in m/s"
    )
    wind_gust_m_s: Optional[float] = Field(
        default=None,
        ge=0,
        description="Wind gust speed"
    )
    wind_direction_deg: Optional[float] = Field(
        default=None,
        ge=0,
        le=360,
        description="Wind direction in degrees"
    )
    wind_direction: Optional[WindDirection] = Field(
        default=None,
        description="Wind direction cardinal"
    )

    # Conditions
    condition: WeatherCondition = Field(
        default=WeatherCondition.UNKNOWN,
        description="Weather condition"
    )
    condition_description: Optional[str] = Field(
        default=None,
        description="Condition description"
    )
    cloud_cover_percent: Optional[float] = Field(
        default=None,
        ge=0,
        le=100,
        description="Cloud cover percentage"
    )
    visibility_m: Optional[float] = Field(
        default=None,
        ge=0,
        description="Visibility in meters"
    )

    # Precipitation
    precipitation_mm: float = Field(
        default=0,
        ge=0,
        description="Precipitation last hour"
    )
    snow_mm: float = Field(
        default=0,
        ge=0,
        description="Snowfall last hour"
    )

    # Solar
    sunrise: Optional[datetime] = Field(default=None)
    sunset: Optional[datetime] = Field(default=None)
    solar_radiation_w_m2: Optional[float] = Field(
        default=None,
        ge=0,
        description="Solar radiation"
    )
    uv_index: Optional[float] = Field(
        default=None,
        ge=0,
        description="UV index"
    )


class HourlyForecast(BaseModel):
    """Hourly weather forecast."""

    model_config = ConfigDict(frozen=True)

    forecast_time: datetime = Field(..., description="Forecast timestamp")

    # Temperature
    temperature_c: float = Field(..., description="Temperature")
    feels_like_c: float = Field(..., description="Feels like")

    # Humidity and pressure
    humidity_percent: float = Field(..., ge=0, le=100)
    pressure_hpa: float = Field(..., ge=0)

    # Wind
    wind_speed_m_s: float = Field(..., ge=0)
    wind_gust_m_s: Optional[float] = Field(default=None, ge=0)
    wind_direction_deg: Optional[float] = Field(default=None, ge=0, le=360)

    # Conditions
    condition: WeatherCondition = Field(default=WeatherCondition.UNKNOWN)
    condition_description: Optional[str] = Field(default=None)
    cloud_cover_percent: Optional[float] = Field(default=None, ge=0, le=100)

    # Precipitation
    precipitation_probability_percent: float = Field(
        default=0,
        ge=0,
        le=100,
        description="Precipitation probability"
    )
    precipitation_mm: float = Field(default=0, ge=0)

    # Visibility
    visibility_m: Optional[float] = Field(default=None, ge=0)


class DailyForecast(BaseModel):
    """Daily weather forecast."""

    model_config = ConfigDict(frozen=True)

    date: datetime = Field(..., description="Forecast date")

    # Temperature
    temperature_min_c: float = Field(..., description="Minimum temperature")
    temperature_max_c: float = Field(..., description="Maximum temperature")
    temperature_morning_c: Optional[float] = Field(default=None)
    temperature_day_c: Optional[float] = Field(default=None)
    temperature_evening_c: Optional[float] = Field(default=None)
    temperature_night_c: Optional[float] = Field(default=None)

    # Humidity
    humidity_percent: float = Field(..., ge=0, le=100)

    # Wind
    wind_speed_m_s: float = Field(..., ge=0)
    wind_gust_m_s: Optional[float] = Field(default=None, ge=0)
    wind_direction_deg: Optional[float] = Field(default=None, ge=0, le=360)

    # Conditions
    condition: WeatherCondition = Field(default=WeatherCondition.UNKNOWN)
    condition_description: Optional[str] = Field(default=None)
    cloud_cover_percent: Optional[float] = Field(default=None, ge=0, le=100)

    # Precipitation
    precipitation_probability_percent: float = Field(default=0, ge=0, le=100)
    precipitation_mm: float = Field(default=0, ge=0)
    snow_mm: float = Field(default=0, ge=0)

    # Sun
    sunrise: Optional[datetime] = Field(default=None)
    sunset: Optional[datetime] = Field(default=None)
    uv_index: Optional[float] = Field(default=None, ge=0)


class WeatherForecast(BaseModel):
    """Complete weather forecast."""

    model_config = ConfigDict(frozen=False)

    location_name: Optional[str] = Field(default=None)
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    timezone: str = Field(default="UTC")

    generated_at: datetime = Field(default_factory=datetime.utcnow)

    hourly_forecasts: List[HourlyForecast] = Field(
        default_factory=list,
        description="Hourly forecasts"
    )
    daily_forecasts: List[DailyForecast] = Field(
        default_factory=list,
        description="Daily forecasts"
    )


class HistoricalWeather(BaseModel):
    """Historical weather data."""

    model_config = ConfigDict(frozen=True)

    date: datetime = Field(..., description="Date")
    location_name: Optional[str] = Field(default=None)
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)

    # Temperature
    temperature_avg_c: float = Field(..., description="Average temperature")
    temperature_min_c: float = Field(..., description="Minimum temperature")
    temperature_max_c: float = Field(..., description="Maximum temperature")

    # Humidity and pressure
    humidity_avg_percent: float = Field(..., ge=0, le=100)
    pressure_avg_hpa: Optional[float] = Field(default=None, ge=0)

    # Wind
    wind_speed_avg_m_s: float = Field(..., ge=0)
    wind_speed_max_m_s: Optional[float] = Field(default=None, ge=0)
    wind_direction_dominant_deg: Optional[float] = Field(
        default=None,
        ge=0,
        le=360
    )

    # Precipitation
    precipitation_total_mm: float = Field(default=0, ge=0)
    snow_total_mm: float = Field(default=0, ge=0)

    # Other
    cloud_cover_avg_percent: Optional[float] = Field(
        default=None,
        ge=0,
        le=100
    )
    sunshine_hours: Optional[float] = Field(default=None, ge=0, le=24)


# =============================================================================
# Data Models - Inspection Suitability
# =============================================================================


class InspectionWindow(BaseModel):
    """Suitable time window for thermal inspection."""

    model_config = ConfigDict(frozen=True)

    start_time: datetime = Field(..., description="Window start")
    end_time: datetime = Field(..., description="Window end")
    duration_hours: float = Field(..., ge=0, description="Duration in hours")

    suitability: InspectionSuitability = Field(
        ...,
        description="Overall suitability"
    )
    suitability_score: float = Field(
        ...,
        ge=0,
        le=100,
        description="Suitability score 0-100"
    )

    # Average conditions during window
    avg_temperature_c: float = Field(..., description="Average temperature")
    avg_wind_speed_m_s: float = Field(..., ge=0, description="Average wind")
    avg_humidity_percent: float = Field(..., ge=0, le=100)
    precipitation_probability_percent: float = Field(
        default=0,
        ge=0,
        le=100
    )
    cloud_cover_percent: Optional[float] = Field(default=None, ge=0, le=100)

    # Issues
    issues: List[str] = Field(
        default_factory=list,
        description="Potential issues"
    )
    recommendations: List[str] = Field(
        default_factory=list,
        description="Recommendations"
    )


class InspectionPlanningReport(BaseModel):
    """Weather-based inspection planning report."""

    model_config = ConfigDict(frozen=False)

    report_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Report ID"
    )
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    location_name: Optional[str] = Field(default=None)
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)

    # Current conditions
    current_weather: Optional[CurrentWeather] = Field(default=None)
    current_suitability: InspectionSuitability = Field(
        default=InspectionSuitability.UNSUITABLE
    )
    current_suitability_score: float = Field(default=0, ge=0, le=100)

    # Suitable windows in next 7 days
    suitable_windows: List[InspectionWindow] = Field(
        default_factory=list,
        description="Suitable inspection windows"
    )
    best_window: Optional[InspectionWindow] = Field(
        default=None,
        description="Best inspection window"
    )

    # Summary
    total_suitable_hours: float = Field(
        default=0,
        ge=0,
        description="Total suitable hours"
    )
    earliest_suitable_time: Optional[datetime] = Field(default=None)
    overall_recommendation: str = Field(
        default="",
        description="Overall recommendation"
    )


# =============================================================================
# Weather Connector
# =============================================================================


class WeatherConnector(BaseConnector):
    """
    Weather Connector for GL-015 INSULSCAN.

    Provides weather data for thermal inspection planning including
    current conditions, forecasts, and historical data.

    Features:
    - Current weather conditions
    - Multi-day forecasts (hourly and daily)
    - Historical weather data
    - Inspection suitability assessment
    - Optimal inspection window identification
    """

    def __init__(self, config: WeatherConnectorConfig) -> None:
        """
        Initialize weather connector.

        Args:
            config: Weather connector configuration
        """
        super().__init__(config)
        self._weather_config = config

        # HTTP session
        self._http_session: Optional[aiohttp.ClientSession] = None

    # =========================================================================
    # Connection Lifecycle
    # =========================================================================

    async def connect(self) -> None:
        """Establish connection to weather service."""
        self._logger.info(
            f"Connecting to weather service: {self._weather_config.provider.value}"
        )

        try:
            self._state = ConnectionState.CONNECTING

            # Get HTTP session from pool
            self._http_session = await self._pool.get_session()

            # Verify API key/connection
            await self._verify_connection()

            self._state = ConnectionState.CONNECTED
            self._logger.info("Connected to weather service")

        except Exception as e:
            self._state = ConnectionState.ERROR
            self._logger.error(f"Failed to connect to weather service: {e}")
            raise ConnectionError(f"Weather service connection failed: {e}")

    async def disconnect(self) -> None:
        """Disconnect from weather service."""
        self._logger.info("Disconnecting from weather service")
        self._state = ConnectionState.DISCONNECTED

    async def health_check(self) -> HealthCheckResult:
        """Perform health check on weather service connection."""
        import time
        start_time = time.time()

        try:
            if self._state != ConnectionState.CONNECTED:
                return HealthCheckResult(
                    status=HealthStatus.UNHEALTHY,
                    message=f"Not connected: {self._state.value}"
                )

            # Test API
            await self._verify_connection()

            latency_ms = (time.time() - start_time) * 1000

            return HealthCheckResult(
                status=HealthStatus.HEALTHY,
                latency_ms=latency_ms,
                message="Weather service healthy",
                details={
                    "provider": self._weather_config.provider.value,
                }
            )

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                latency_ms=latency_ms,
                message=f"Health check failed: {e}"
            )

    async def validate_configuration(self) -> bool:
        """Validate connector configuration."""
        provider = self._weather_config.provider

        if provider == WeatherProvider.OPENWEATHERMAP:
            if not self._weather_config.openweathermap_config:
                raise ConfigurationError(
                    "OpenWeatherMap config required for OpenWeatherMap provider"
                )

        elif provider == WeatherProvider.NOAA:
            if not self._weather_config.noaa_config:
                raise ConfigurationError(
                    "NOAA config required for NOAA provider"
                )

        return True

    async def _verify_connection(self) -> None:
        """Verify weather service connection."""
        provider = self._weather_config.provider

        if provider == WeatherProvider.OPENWEATHERMAP:
            await self._verify_openweathermap()
        elif provider == WeatherProvider.NOAA:
            await self._verify_noaa()

    async def _verify_openweathermap(self) -> None:
        """Verify OpenWeatherMap API connection."""
        config = self._weather_config.openweathermap_config
        # Use a test location (London)
        url = f"{config.base_url}/{config.api_version}/weather"
        params = {
            "lat": "51.5074",
            "lon": "-0.1278",
            "appid": config.api_key,
            "units": config.units,
        }

        async with self._http_session.get(
            url,
            params=params,
            timeout=aiohttp.ClientTimeout(total=10)
        ) as response:
            if response.status == 401:
                raise ConnectionError("Invalid OpenWeatherMap API key")
            if response.status != 200:
                raise ConnectionError(
                    f"OpenWeatherMap API error: {response.status}"
                )

    async def _verify_noaa(self) -> None:
        """Verify NOAA API connection."""
        config = self._weather_config.noaa_config
        url = f"{config.base_url}/points/39.7456,-104.9897"
        headers = {"User-Agent": config.user_agent}

        async with self._http_session.get(
            url,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=10)
        ) as response:
            if response.status not in [200, 404]:
                raise ConnectionError(f"NOAA API error: {response.status}")

    # =========================================================================
    # Current Weather
    # =========================================================================

    @with_retry(max_retries=3, base_delay=1.0)
    async def get_current_weather(
        self,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        location: Optional[LocationConfig] = None
    ) -> CurrentWeather:
        """
        Get current weather conditions.

        Args:
            latitude: Location latitude
            longitude: Location longitude
            location: Location configuration

        Returns:
            Current weather conditions
        """
        # Resolve location
        if location:
            lat, lon = location.latitude, location.longitude
        elif latitude is not None and longitude is not None:
            lat, lon = latitude, longitude
        elif self._weather_config.default_location:
            lat = self._weather_config.default_location.latitude
            lon = self._weather_config.default_location.longitude
        else:
            raise ConfigurationError("No location specified")

        async def _fetch():
            provider = self._weather_config.provider

            if provider == WeatherProvider.OPENWEATHERMAP:
                return await self._get_current_openweathermap(lat, lon)
            elif provider == WeatherProvider.NOAA:
                return await self._get_current_noaa(lat, lon)
            else:
                raise ConfigurationError(f"Unsupported provider: {provider}")

        return await self.execute_with_protection(
            operation=_fetch,
            operation_name="get_current_weather",
            use_cache=True,
            cache_key=f"current_{lat:.4f}_{lon:.4f}"
        )

    async def _get_current_openweathermap(
        self,
        lat: float,
        lon: float
    ) -> CurrentWeather:
        """Get current weather from OpenWeatherMap."""
        config = self._weather_config.openweathermap_config
        url = f"{config.base_url}/{config.api_version}/weather"
        params = {
            "lat": str(lat),
            "lon": str(lon),
            "appid": config.api_key,
            "units": config.units,
            "lang": config.language,
        }

        async with self._http_session.get(
            url,
            params=params,
            timeout=aiohttp.ClientTimeout(total=30)
        ) as response:
            if response.status != 200:
                raise ConnectorError(
                    f"OpenWeatherMap API error: {response.status}"
                )

            data = await response.json()

            # Parse response
            main = data.get("main", {})
            wind = data.get("wind", {})
            clouds = data.get("clouds", {})
            weather = data.get("weather", [{}])[0]
            sys = data.get("sys", {})
            rain = data.get("rain", {})
            snow = data.get("snow", {})

            # Map condition
            condition = self._map_owm_condition(
                weather.get("id", 0),
                weather.get("main", "")
            )

            # Parse wind direction
            wind_deg = wind.get("deg")
            wind_dir = self._degrees_to_cardinal(wind_deg) if wind_deg else None

            return CurrentWeather(
                location_name=data.get("name"),
                latitude=lat,
                longitude=lon,
                observation_time=datetime.utcfromtimestamp(data.get("dt", 0)),
                temperature_c=main.get("temp", 0),
                feels_like_c=main.get("feels_like", 0),
                temperature_min_c=main.get("temp_min"),
                temperature_max_c=main.get("temp_max"),
                humidity_percent=main.get("humidity", 0),
                pressure_hpa=main.get("pressure", 0),
                wind_speed_m_s=wind.get("speed", 0),
                wind_gust_m_s=wind.get("gust"),
                wind_direction_deg=wind_deg,
                wind_direction=wind_dir,
                condition=condition,
                condition_description=weather.get("description"),
                cloud_cover_percent=clouds.get("all"),
                visibility_m=data.get("visibility"),
                precipitation_mm=rain.get("1h", 0),
                snow_mm=snow.get("1h", 0),
                sunrise=datetime.utcfromtimestamp(sys.get("sunrise", 0)) if sys.get("sunrise") else None,
                sunset=datetime.utcfromtimestamp(sys.get("sunset", 0)) if sys.get("sunset") else None,
            )

    async def _get_current_noaa(
        self,
        lat: float,
        lon: float
    ) -> CurrentWeather:
        """Get current weather from NOAA."""
        config = self._weather_config.noaa_config
        headers = {"User-Agent": config.user_agent}

        # First get the grid point
        points_url = f"{config.base_url}/points/{lat},{lon}"

        async with self._http_session.get(
            points_url,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=30)
        ) as response:
            if response.status != 200:
                raise ConnectorError(f"NOAA points API error: {response.status}")
            points_data = await response.json()

        # Get observation stations
        stations_url = points_data.get("properties", {}).get("observationStations")
        if not stations_url:
            raise ConnectorError("No observation stations found")

        async with self._http_session.get(
            stations_url,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=30)
        ) as response:
            if response.status != 200:
                raise ConnectorError(f"NOAA stations API error: {response.status}")
            stations_data = await response.json()

        # Get latest observation from first station
        features = stations_data.get("features", [])
        if not features:
            raise ConnectorError("No stations available")

        station_id = features[0].get("properties", {}).get("stationIdentifier")
        obs_url = f"{config.base_url}/stations/{station_id}/observations/latest"

        async with self._http_session.get(
            obs_url,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=30)
        ) as response:
            if response.status != 200:
                raise ConnectorError(f"NOAA observations API error: {response.status}")
            obs_data = await response.json()

        props = obs_data.get("properties", {})

        # Parse NOAA response
        temp = props.get("temperature", {}).get("value")
        if temp is not None:
            temp_c = temp  # NOAA returns Celsius
        else:
            temp_c = 0

        humidity = props.get("relativeHumidity", {}).get("value", 0)
        wind_speed = props.get("windSpeed", {}).get("value", 0)
        if wind_speed:
            wind_speed = wind_speed / 3.6  # km/h to m/s

        return CurrentWeather(
            latitude=lat,
            longitude=lon,
            observation_time=datetime.fromisoformat(
                props.get("timestamp", datetime.utcnow().isoformat()).replace("Z", "+00:00")
            ),
            temperature_c=temp_c or 0,
            feels_like_c=temp_c or 0,
            humidity_percent=humidity or 0,
            pressure_hpa=props.get("barometricPressure", {}).get("value", 0) / 100 if props.get("barometricPressure", {}).get("value") else 1013,
            wind_speed_m_s=wind_speed or 0,
            wind_direction_deg=props.get("windDirection", {}).get("value"),
            condition=WeatherCondition.UNKNOWN,
            condition_description=props.get("textDescription"),
            visibility_m=props.get("visibility", {}).get("value"),
        )

    # =========================================================================
    # Weather Forecast
    # =========================================================================

    @with_retry(max_retries=3, base_delay=1.0)
    async def get_forecast(
        self,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        days: Optional[int] = None
    ) -> WeatherForecast:
        """
        Get weather forecast.

        Args:
            latitude: Location latitude
            longitude: Location longitude
            days: Number of forecast days

        Returns:
            Weather forecast
        """
        # Resolve location
        if latitude is not None and longitude is not None:
            lat, lon = latitude, longitude
        elif self._weather_config.default_location:
            lat = self._weather_config.default_location.latitude
            lon = self._weather_config.default_location.longitude
        else:
            raise ConfigurationError("No location specified")

        days = days or self._weather_config.forecast_days

        async def _fetch():
            provider = self._weather_config.provider

            if provider == WeatherProvider.OPENWEATHERMAP:
                return await self._get_forecast_openweathermap(lat, lon, days)
            elif provider == WeatherProvider.NOAA:
                return await self._get_forecast_noaa(lat, lon)
            else:
                raise ConfigurationError(f"Unsupported provider: {provider}")

        return await self.execute_with_protection(
            operation=_fetch,
            operation_name="get_forecast",
            use_cache=True,
            cache_key=f"forecast_{lat:.4f}_{lon:.4f}_{days}"
        )

    async def _get_forecast_openweathermap(
        self,
        lat: float,
        lon: float,
        days: int
    ) -> WeatherForecast:
        """Get forecast from OpenWeatherMap."""
        config = self._weather_config.openweathermap_config

        # Get 5-day/3-hour forecast
        url = f"{config.base_url}/{config.api_version}/forecast"
        params = {
            "lat": str(lat),
            "lon": str(lon),
            "appid": config.api_key,
            "units": config.units,
            "cnt": str(min(days * 8, 40)),  # 8 forecasts per day, max 40
        }

        async with self._http_session.get(
            url,
            params=params,
            timeout=aiohttp.ClientTimeout(total=30)
        ) as response:
            if response.status != 200:
                raise ConnectorError(
                    f"OpenWeatherMap forecast API error: {response.status}"
                )

            data = await response.json()

        # Parse hourly forecasts
        hourly_forecasts = []
        for item in data.get("list", []):
            weather = item.get("weather", [{}])[0]
            main = item.get("main", {})
            wind = item.get("wind", {})
            clouds = item.get("clouds", {})
            pop = item.get("pop", 0)
            rain = item.get("rain", {})

            condition = self._map_owm_condition(
                weather.get("id", 0),
                weather.get("main", "")
            )

            hourly_forecasts.append(HourlyForecast(
                forecast_time=datetime.utcfromtimestamp(item.get("dt", 0)),
                temperature_c=main.get("temp", 0),
                feels_like_c=main.get("feels_like", 0),
                humidity_percent=main.get("humidity", 0),
                pressure_hpa=main.get("pressure", 0),
                wind_speed_m_s=wind.get("speed", 0),
                wind_gust_m_s=wind.get("gust"),
                wind_direction_deg=wind.get("deg"),
                condition=condition,
                condition_description=weather.get("description"),
                cloud_cover_percent=clouds.get("all"),
                precipitation_probability_percent=pop * 100,
                precipitation_mm=rain.get("3h", 0) / 3,  # Convert 3h to 1h
            ))

        # Aggregate to daily forecasts
        daily_forecasts = self._aggregate_to_daily(hourly_forecasts)

        return WeatherForecast(
            location_name=data.get("city", {}).get("name"),
            latitude=lat,
            longitude=lon,
            hourly_forecasts=hourly_forecasts,
            daily_forecasts=daily_forecasts,
        )

    async def _get_forecast_noaa(
        self,
        lat: float,
        lon: float
    ) -> WeatherForecast:
        """Get forecast from NOAA."""
        config = self._weather_config.noaa_config
        headers = {"User-Agent": config.user_agent}

        # Get grid point
        points_url = f"{config.base_url}/points/{lat},{lon}"

        async with self._http_session.get(
            points_url,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=30)
        ) as response:
            if response.status != 200:
                raise ConnectorError(f"NOAA points API error: {response.status}")
            points_data = await response.json()

        # Get forecast URL
        forecast_url = points_data.get("properties", {}).get("forecastHourly")
        if not forecast_url:
            raise ConnectorError("No forecast URL found")

        async with self._http_session.get(
            forecast_url,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=30)
        ) as response:
            if response.status != 200:
                raise ConnectorError(f"NOAA forecast API error: {response.status}")
            forecast_data = await response.json()

        # Parse hourly forecasts
        hourly_forecasts = []
        for period in forecast_data.get("properties", {}).get("periods", [])[:48]:
            temp_c = (period.get("temperature", 0) - 32) * 5 / 9  # F to C
            wind_speed_str = period.get("windSpeed", "0 mph")
            wind_speed = float(wind_speed_str.split()[0]) * 0.447  # mph to m/s

            hourly_forecasts.append(HourlyForecast(
                forecast_time=datetime.fromisoformat(
                    period.get("startTime", datetime.utcnow().isoformat()).replace("Z", "+00:00")
                ),
                temperature_c=temp_c,
                feels_like_c=temp_c,
                humidity_percent=period.get("relativeHumidity", {}).get("value", 50),
                pressure_hpa=1013,  # NOAA doesn't provide pressure in hourly
                wind_speed_m_s=wind_speed,
                wind_direction_deg=None,
                condition=WeatherCondition.UNKNOWN,
                condition_description=period.get("shortForecast"),
                precipitation_probability_percent=period.get("probabilityOfPrecipitation", {}).get("value", 0) or 0,
            ))

        daily_forecasts = self._aggregate_to_daily(hourly_forecasts)

        return WeatherForecast(
            latitude=lat,
            longitude=lon,
            hourly_forecasts=hourly_forecasts,
            daily_forecasts=daily_forecasts,
        )

    def _aggregate_to_daily(
        self,
        hourly: List[HourlyForecast]
    ) -> List[DailyForecast]:
        """Aggregate hourly forecasts to daily."""
        daily_data: Dict[str, List[HourlyForecast]] = {}

        for h in hourly:
            date_key = h.forecast_time.strftime("%Y-%m-%d")
            if date_key not in daily_data:
                daily_data[date_key] = []
            daily_data[date_key].append(h)

        daily_forecasts = []
        for date_str, hours in sorted(daily_data.items()):
            temps = [h.temperature_c for h in hours]
            winds = [h.wind_speed_m_s for h in hours]
            humidities = [h.humidity_percent for h in hours]
            pops = [h.precipitation_probability_percent for h in hours]
            precips = [h.precipitation_mm for h in hours]

            daily_forecasts.append(DailyForecast(
                date=datetime.strptime(date_str, "%Y-%m-%d"),
                temperature_min_c=min(temps),
                temperature_max_c=max(temps),
                humidity_percent=sum(humidities) / len(humidities),
                wind_speed_m_s=sum(winds) / len(winds),
                wind_gust_m_s=max(h.wind_gust_m_s or 0 for h in hours) or None,
                condition=hours[len(hours) // 2].condition,  # Midday condition
                condition_description=hours[len(hours) // 2].condition_description,
                precipitation_probability_percent=max(pops),
                precipitation_mm=sum(precips),
            ))

        return daily_forecasts

    # =========================================================================
    # Historical Weather
    # =========================================================================

    @with_retry(max_retries=3, base_delay=1.0)
    async def get_historical_weather(
        self,
        latitude: float,
        longitude: float,
        date: datetime
    ) -> Optional[HistoricalWeather]:
        """
        Get historical weather data for a specific date.

        Args:
            latitude: Location latitude
            longitude: Location longitude
            date: Date to retrieve

        Returns:
            Historical weather data or None
        """
        if not self._weather_config.historical_data_enabled:
            self._logger.warning("Historical data is disabled")
            return None

        # Check date limit
        days_ago = (datetime.utcnow() - date).days
        if days_ago > self._weather_config.historical_days_limit:
            raise ConfigurationError(
                f"Date exceeds historical limit of {self._weather_config.historical_days_limit} days"
            )

        # OpenWeatherMap One Call API (requires paid subscription for historical)
        # For now, return None as historical data requires premium APIs
        self._logger.warning(
            "Historical weather data requires premium API subscription"
        )
        return None

    # =========================================================================
    # Inspection Planning
    # =========================================================================

    async def assess_inspection_suitability(
        self,
        weather: CurrentWeather
    ) -> Tuple[InspectionSuitability, float, List[str]]:
        """
        Assess current weather suitability for thermal inspection.

        Args:
            weather: Current weather conditions

        Returns:
            Tuple of (suitability, score 0-100, list of issues)
        """
        config = self._weather_config
        issues = []
        score = 100.0

        # Check wind speed
        if weather.wind_speed_m_s > config.max_wind_speed_m_s:
            issues.append(
                f"Wind speed ({weather.wind_speed_m_s:.1f} m/s) exceeds limit "
                f"({config.max_wind_speed_m_s} m/s)"
            )
            score -= 30
        elif weather.wind_speed_m_s > config.max_wind_speed_m_s * 0.7:
            issues.append(
                f"Wind speed ({weather.wind_speed_m_s:.1f} m/s) is high"
            )
            score -= 15

        # Check humidity
        if weather.humidity_percent > config.max_humidity_percent:
            issues.append(
                f"Humidity ({weather.humidity_percent:.0f}%) exceeds limit "
                f"({config.max_humidity_percent}%)"
            )
            score -= 20

        # Check precipitation
        if weather.precipitation_mm > config.rain_threshold_mm:
            issues.append(
                f"Active precipitation ({weather.precipitation_mm:.1f} mm/h)"
            )
            score -= 40

        # Check conditions
        if weather.condition in [
            WeatherCondition.RAIN,
            WeatherCondition.HEAVY_RAIN,
            WeatherCondition.THUNDERSTORM,
            WeatherCondition.SNOW,
            WeatherCondition.SLEET,
        ]:
            issues.append(f"Weather condition: {weather.condition.value}")
            score -= 50

        elif weather.condition in [
            WeatherCondition.FOG,
            WeatherCondition.MIST,
        ]:
            issues.append(f"Reduced visibility: {weather.condition.value}")
            score -= 20

        # Check for solar radiation interference (if available)
        if weather.solar_radiation_w_m2 and weather.solar_radiation_w_m2 > 800:
            issues.append(
                f"High solar radiation ({weather.solar_radiation_w_m2:.0f} W/m2) "
                "may affect readings"
            )
            score -= 10

        # Determine suitability
        score = max(0, min(100, score))

        if score >= 90:
            suitability = InspectionSuitability.EXCELLENT
        elif score >= 70:
            suitability = InspectionSuitability.GOOD
        elif score >= 50:
            suitability = InspectionSuitability.MARGINAL
        elif score >= 30:
            suitability = InspectionSuitability.POOR
        else:
            suitability = InspectionSuitability.UNSUITABLE

        return suitability, score, issues

    async def find_inspection_windows(
        self,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        min_duration_hours: float = 2.0,
        days_ahead: int = 7
    ) -> InspectionPlanningReport:
        """
        Find suitable windows for thermal inspection.

        Args:
            latitude: Location latitude
            longitude: Location longitude
            min_duration_hours: Minimum window duration
            days_ahead: Days to search ahead

        Returns:
            Inspection planning report
        """
        # Resolve location
        if latitude is not None and longitude is not None:
            lat, lon = latitude, longitude
        elif self._weather_config.default_location:
            lat = self._weather_config.default_location.latitude
            lon = self._weather_config.default_location.longitude
        else:
            raise ConfigurationError("No location specified")

        # Get current weather
        current = await self.get_current_weather(lat, lon)
        current_suitability, current_score, current_issues = \
            await self.assess_inspection_suitability(current)

        # Get forecast
        forecast = await self.get_forecast(lat, lon, days_ahead)

        # Find suitable windows
        suitable_windows: List[InspectionWindow] = []
        window_start: Optional[datetime] = None
        window_forecasts: List[HourlyForecast] = []

        for hourly in forecast.hourly_forecasts:
            # Create a mock CurrentWeather for assessment
            mock_weather = CurrentWeather(
                latitude=lat,
                longitude=lon,
                temperature_c=hourly.temperature_c,
                feels_like_c=hourly.feels_like_c,
                humidity_percent=hourly.humidity_percent,
                pressure_hpa=hourly.pressure_hpa,
                wind_speed_m_s=hourly.wind_speed_m_s,
                condition=hourly.condition,
                precipitation_mm=hourly.precipitation_mm,
            )

            suitability, score, issues = await self.assess_inspection_suitability(
                mock_weather
            )

            is_suitable = suitability in [
                InspectionSuitability.EXCELLENT,
                InspectionSuitability.GOOD,
                InspectionSuitability.MARGINAL,
            ]

            if is_suitable:
                if window_start is None:
                    window_start = hourly.forecast_time
                window_forecasts.append(hourly)
            else:
                # End of window
                if window_start and len(window_forecasts) > 0:
                    duration = (window_forecasts[-1].forecast_time - window_start).total_seconds() / 3600
                    if duration >= min_duration_hours:
                        window = self._create_inspection_window(
                            window_start,
                            window_forecasts
                        )
                        suitable_windows.append(window)

                window_start = None
                window_forecasts = []

        # Handle last window
        if window_start and len(window_forecasts) > 0:
            duration = (window_forecasts[-1].forecast_time - window_start).total_seconds() / 3600
            if duration >= min_duration_hours:
                window = self._create_inspection_window(
                    window_start,
                    window_forecasts
                )
                suitable_windows.append(window)

        # Sort by suitability score
        suitable_windows.sort(
            key=lambda w: w.suitability_score,
            reverse=True
        )

        # Create report
        best_window = suitable_windows[0] if suitable_windows else None
        total_hours = sum(w.duration_hours for w in suitable_windows)
        earliest = min((w.start_time for w in suitable_windows), default=None)

        # Generate recommendation
        if not suitable_windows:
            recommendation = (
                "No suitable inspection windows found in the next "
                f"{days_ahead} days. Consider rescheduling or using "
                "alternative inspection methods."
            )
        elif current_suitability in [
            InspectionSuitability.EXCELLENT,
            InspectionSuitability.GOOD
        ]:
            recommendation = (
                "Current conditions are suitable for thermal inspection. "
                f"Suitability score: {current_score:.0f}/100."
            )
        elif best_window:
            recommendation = (
                f"Best inspection window: {best_window.start_time.strftime('%Y-%m-%d %H:%M')} "
                f"for {best_window.duration_hours:.1f} hours "
                f"(score: {best_window.suitability_score:.0f}/100)."
            )
        else:
            recommendation = "Weather conditions require careful planning."

        return InspectionPlanningReport(
            latitude=lat,
            longitude=lon,
            current_weather=current,
            current_suitability=current_suitability,
            current_suitability_score=current_score,
            suitable_windows=suitable_windows,
            best_window=best_window,
            total_suitable_hours=total_hours,
            earliest_suitable_time=earliest,
            overall_recommendation=recommendation,
        )

    def _create_inspection_window(
        self,
        start_time: datetime,
        forecasts: List[HourlyForecast]
    ) -> InspectionWindow:
        """Create an inspection window from forecasts."""
        end_time = forecasts[-1].forecast_time + timedelta(hours=1)
        duration = (end_time - start_time).total_seconds() / 3600

        temps = [f.temperature_c for f in forecasts]
        winds = [f.wind_speed_m_s for f in forecasts]
        humidities = [f.humidity_percent for f in forecasts]
        pops = [f.precipitation_probability_percent for f in forecasts]
        clouds = [f.cloud_cover_percent for f in forecasts if f.cloud_cover_percent is not None]

        avg_temp = sum(temps) / len(temps)
        avg_wind = sum(winds) / len(winds)
        avg_humidity = sum(humidities) / len(humidities)
        max_pop = max(pops)
        avg_clouds = sum(clouds) / len(clouds) if clouds else None

        # Calculate suitability score
        score = 100.0
        issues = []
        recommendations = []

        if avg_wind > self._weather_config.max_wind_speed_m_s * 0.7:
            score -= 15
            issues.append("Elevated wind speeds")
            recommendations.append("Position equipment to minimize wind effects")

        if avg_humidity > 80:
            score -= 10
            issues.append("High humidity")

        if max_pop > 30:
            score -= 15
            issues.append(f"Precipitation probability: {max_pop:.0f}%")
            recommendations.append("Have rain contingency plan")

        score = max(0, min(100, score))

        if score >= 90:
            suitability = InspectionSuitability.EXCELLENT
        elif score >= 70:
            suitability = InspectionSuitability.GOOD
        else:
            suitability = InspectionSuitability.MARGINAL

        return InspectionWindow(
            start_time=start_time,
            end_time=end_time,
            duration_hours=duration,
            suitability=suitability,
            suitability_score=score,
            avg_temperature_c=avg_temp,
            avg_wind_speed_m_s=avg_wind,
            avg_humidity_percent=avg_humidity,
            precipitation_probability_percent=max_pop,
            cloud_cover_percent=avg_clouds,
            issues=issues,
            recommendations=recommendations,
        )

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _map_owm_condition(
        self,
        condition_id: int,
        main: str
    ) -> WeatherCondition:
        """Map OpenWeatherMap condition code to enum."""
        # Thunderstorm group (2xx)
        if 200 <= condition_id < 300:
            return WeatherCondition.THUNDERSTORM

        # Drizzle group (3xx)
        if 300 <= condition_id < 400:
            return WeatherCondition.LIGHT_RAIN

        # Rain group (5xx)
        if 500 <= condition_id < 510:
            return WeatherCondition.RAIN
        if 510 <= condition_id < 520:
            return WeatherCondition.HEAVY_RAIN
        if 520 <= condition_id < 600:
            return WeatherCondition.RAIN

        # Snow group (6xx)
        if 600 <= condition_id < 700:
            if condition_id in [611, 612, 613, 615, 616]:
                return WeatherCondition.SLEET
            return WeatherCondition.SNOW

        # Atmosphere group (7xx)
        if condition_id == 701:
            return WeatherCondition.MIST
        if condition_id == 741:
            return WeatherCondition.FOG

        # Clear (800)
        if condition_id == 800:
            return WeatherCondition.CLEAR

        # Clouds (80x)
        if condition_id == 801:
            return WeatherCondition.PARTLY_CLOUDY
        if condition_id == 802:
            return WeatherCondition.CLOUDY
        if condition_id in [803, 804]:
            return WeatherCondition.OVERCAST

        return WeatherCondition.UNKNOWN

    def _degrees_to_cardinal(self, degrees: float) -> WindDirection:
        """Convert wind direction degrees to cardinal direction."""
        directions = [
            WindDirection.N, WindDirection.NNE, WindDirection.NE, WindDirection.ENE,
            WindDirection.E, WindDirection.ESE, WindDirection.SE, WindDirection.SSE,
            WindDirection.S, WindDirection.SSW, WindDirection.SW, WindDirection.WSW,
            WindDirection.W, WindDirection.WNW, WindDirection.NW, WindDirection.NNW,
        ]
        idx = int((degrees + 11.25) / 22.5) % 16
        return directions[idx]


# =============================================================================
# Factory Function
# =============================================================================


def create_weather_connector(
    provider: WeatherProvider = WeatherProvider.OPENWEATHERMAP,
    connector_name: str = "Weather",
    **kwargs
) -> WeatherConnector:
    """
    Factory function to create weather connector.

    Args:
        provider: Weather data provider
        connector_name: Connector name
        **kwargs: Additional configuration options

    Returns:
        Configured WeatherConnector instance
    """
    config = WeatherConnectorConfig(
        connector_name=connector_name,
        provider=provider,
        **kwargs
    )
    return WeatherConnector(config)
