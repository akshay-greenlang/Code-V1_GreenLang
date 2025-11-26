"""
Weather and Meteorological Data Connector for GL-010 EMISSIONWATCH.

Provides integration with on-site meteorological stations and NWS API
for weather data required for air dispersion modeling and emissions
calculations per EPA modeling guidelines.

Author: GL-DataIntegrationEngineer
Version: 1.0.0
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta, date
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)
import asyncio
import logging
import math
import time
import uuid

from pydantic import BaseModel, Field, ConfigDict, field_validator
import httpx

from .base_connector import (
    BaseConnector,
    BaseConnectorConfig,
    ConnectionState,
    ConnectorType,
    HealthCheckResult,
    HealthStatus,
    ConnectorError,
    ConnectionError,
    ConfigurationError,
    ValidationError,
    with_retry,
)

# Configure module logger
logger = logging.getLogger(__name__)


# =============================================================================
# Enumerations
# =============================================================================


class DataSource(str, Enum):
    """Meteorological data sources."""

    ONSITE = "onsite"  # On-site met station
    NWS = "nws"  # National Weather Service
    ASOS = "asos"  # Automated Surface Observing System
    AWOS = "awos"  # Automated Weather Observing System
    METAR = "metar"  # Aviation routine weather report
    UPPER_AIR = "upper_air"  # Upper air sounding
    SATELLITE = "satellite"
    FORECAST = "forecast"


class StabilityClass(str, Enum):
    """Pasquill-Gifford atmospheric stability classes."""

    A = "A"  # Very unstable (strong convection)
    B = "B"  # Unstable (moderate convection)
    C = "C"  # Slightly unstable (weak convection)
    D = "D"  # Neutral (overcast)
    E = "E"  # Slightly stable (light wind, clear night)
    F = "F"  # Stable (calm, clear night)
    G = "G"  # Very stable (very calm, clear night)


class CloudCoverCategory(str, Enum):
    """Cloud cover categories."""

    CLEAR = "clear"  # 0-10% coverage
    SCATTERED = "scattered"  # 10-50% coverage
    BROKEN = "broken"  # 50-90% coverage
    OVERCAST = "overcast"  # 90-100% coverage


class PrecipitationType(str, Enum):
    """Precipitation types."""

    NONE = "none"
    RAIN = "rain"
    DRIZZLE = "drizzle"
    SNOW = "snow"
    SLEET = "sleet"
    HAIL = "hail"
    FREEZING_RAIN = "freezing_rain"
    FOG = "fog"


class WindDirection(str, Enum):
    """Cardinal and intercardinal wind directions."""

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
    CALM = "CALM"
    VARIABLE = "VRB"


class MetStationVendor(str, Enum):
    """Meteorological station vendors."""

    CAMPBELL_SCIENTIFIC = "campbell_scientific"
    VAISALA = "vaisala"
    DAVIS = "davis"
    LUFFT = "lufft"
    RM_YOUNG = "rm_young"
    ALL_WEATHER = "all_weather"
    MET_ONE = "met_one"
    GENERIC = "generic"


# =============================================================================
# Pydantic Models
# =============================================================================


class WindData(BaseModel):
    """Wind measurement data."""

    model_config = ConfigDict(frozen=True)

    timestamp: datetime = Field(default_factory=datetime.utcnow)

    # Speed
    speed_mph: float = Field(..., ge=0, description="Wind speed (mph)")
    speed_mps: Optional[float] = Field(default=None, ge=0, description="Speed (m/s)")
    gust_mph: Optional[float] = Field(default=None, ge=0, description="Gust speed")

    # Direction
    direction_degrees: float = Field(
        ...,
        ge=0,
        le=360,
        description="Direction (degrees from north)"
    )
    direction_cardinal: WindDirection = Field(..., description="Cardinal direction")
    direction_sigma: Optional[float] = Field(
        default=None,
        ge=0,
        le=180,
        description="Direction std dev (sigma theta)"
    )

    # Measurement height
    measurement_height_m: float = Field(
        default=10.0,
        ge=0,
        description="Measurement height (m)"
    )

    # Quality
    is_valid: bool = Field(default=True)
    quality_code: str = Field(default="OK")


class TemperatureData(BaseModel):
    """Temperature measurement data."""

    model_config = ConfigDict(frozen=True)

    timestamp: datetime = Field(default_factory=datetime.utcnow)

    # Temperature
    temperature_f: float = Field(..., description="Temperature (F)")
    temperature_c: Optional[float] = Field(default=None, description="Temperature (C)")
    dew_point_f: Optional[float] = Field(default=None, description="Dew point (F)")
    wet_bulb_f: Optional[float] = Field(default=None, description="Wet bulb (F)")

    # Humidity
    relative_humidity_percent: Optional[float] = Field(
        default=None,
        ge=0,
        le=100,
        description="Relative humidity %"
    )

    # Vertical temperature profile
    delta_t: Optional[float] = Field(
        default=None,
        description="Temperature difference (upper-lower)"
    )
    lapse_rate: Optional[float] = Field(
        default=None,
        description="Temperature lapse rate (C/100m)"
    )

    # Measurement height
    measurement_height_m: float = Field(default=2.0, ge=0)

    is_valid: bool = Field(default=True)
    quality_code: str = Field(default="OK")


class PressureData(BaseModel):
    """Atmospheric pressure data."""

    model_config = ConfigDict(frozen=True)

    timestamp: datetime = Field(default_factory=datetime.utcnow)

    # Pressure
    station_pressure_inhg: float = Field(
        ...,
        ge=25,
        le=35,
        description="Station pressure (inHg)"
    )
    sea_level_pressure_inhg: Optional[float] = Field(
        default=None,
        description="Sea level pressure"
    )
    station_pressure_mb: Optional[float] = Field(default=None, description="Pressure (mb)")

    # Altitude
    station_elevation_m: Optional[float] = Field(default=None, ge=0)
    pressure_altitude_m: Optional[float] = Field(default=None)

    is_valid: bool = Field(default=True)
    quality_code: str = Field(default="OK")


class SolarRadiationData(BaseModel):
    """Solar radiation data."""

    model_config = ConfigDict(frozen=True)

    timestamp: datetime = Field(default_factory=datetime.utcnow)

    # Radiation
    global_horizontal_wm2: Optional[float] = Field(
        default=None,
        ge=0,
        description="Global horizontal radiation (W/m2)"
    )
    direct_normal_wm2: Optional[float] = Field(
        default=None,
        ge=0,
        description="Direct normal radiation"
    )
    diffuse_horizontal_wm2: Optional[float] = Field(
        default=None,
        ge=0,
        description="Diffuse horizontal radiation"
    )
    net_radiation_wm2: Optional[float] = Field(default=None, description="Net radiation")

    # Solar position (calculated)
    solar_elevation_deg: Optional[float] = Field(
        default=None,
        ge=-90,
        le=90,
        description="Solar elevation angle"
    )
    solar_azimuth_deg: Optional[float] = Field(
        default=None,
        ge=0,
        le=360,
        description="Solar azimuth"
    )

    is_valid: bool = Field(default=True)


class PrecipitationData(BaseModel):
    """Precipitation data."""

    model_config = ConfigDict(frozen=True)

    timestamp: datetime = Field(default_factory=datetime.utcnow)

    precipitation_type: PrecipitationType = Field(..., description="Type")
    precipitation_rate_in_hr: Optional[float] = Field(
        default=None,
        ge=0,
        description="Rate (in/hr)"
    )
    accumulation_in: Optional[float] = Field(
        default=None,
        ge=0,
        description="Accumulation (in)"
    )
    accumulation_period_hours: Optional[float] = Field(default=None, ge=0)

    is_valid: bool = Field(default=True)


class CloudData(BaseModel):
    """Cloud cover and ceiling data."""

    model_config = ConfigDict(frozen=True)

    timestamp: datetime = Field(default_factory=datetime.utcnow)

    cloud_cover_category: CloudCoverCategory = Field(..., description="Category")
    cloud_cover_percent: Optional[float] = Field(
        default=None,
        ge=0,
        le=100,
        description="Coverage %"
    )
    ceiling_ft: Optional[int] = Field(default=None, ge=0, description="Ceiling (ft AGL)")
    ceiling_m: Optional[int] = Field(default=None, ge=0, description="Ceiling (m AGL)")

    # Layer information
    cloud_layers: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Cloud layer details"
    )

    is_valid: bool = Field(default=True)


class AtmosphericStability(BaseModel):
    """Atmospheric stability classification."""

    model_config = ConfigDict(frozen=True)

    timestamp: datetime = Field(default_factory=datetime.utcnow)

    stability_class: StabilityClass = Field(
        ...,
        description="Pasquill-Gifford stability class"
    )
    classification_method: str = Field(
        ...,
        description="Method used for classification"
    )

    # Supporting parameters
    wind_speed_mps: Optional[float] = Field(default=None, ge=0)
    solar_radiation_wm2: Optional[float] = Field(default=None, ge=0)
    cloud_cover_fraction: Optional[float] = Field(default=None, ge=0, le=1)
    is_daytime: bool = Field(..., description="Daytime flag")

    # Turbulence parameters
    sigma_theta: Optional[float] = Field(
        default=None,
        ge=0,
        le=180,
        description="Sigma theta (degrees)"
    )
    sigma_w: Optional[float] = Field(
        default=None,
        ge=0,
        description="Sigma w (m/s)"
    )

    # Boundary layer
    mixing_height_m: Optional[float] = Field(
        default=None,
        ge=0,
        description="Mixing height (m)"
    )
    friction_velocity_mps: Optional[float] = Field(
        default=None,
        ge=0,
        description="Friction velocity"
    )
    monin_obukhov_length_m: Optional[float] = Field(
        default=None,
        description="Monin-Obukhov length"
    )


class MeteorologicalObservation(BaseModel):
    """Complete meteorological observation."""

    model_config = ConfigDict(frozen=True)

    observation_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Observation identifier"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Observation timestamp"
    )
    data_source: DataSource = Field(..., description="Data source")
    station_id: str = Field(..., description="Station identifier")

    # Components
    wind: Optional[WindData] = Field(default=None, description="Wind data")
    temperature: Optional[TemperatureData] = Field(
        default=None,
        description="Temperature data"
    )
    pressure: Optional[PressureData] = Field(default=None, description="Pressure data")
    solar: Optional[SolarRadiationData] = Field(default=None, description="Solar data")
    precipitation: Optional[PrecipitationData] = Field(
        default=None,
        description="Precipitation"
    )
    clouds: Optional[CloudData] = Field(default=None, description="Cloud data")

    # Derived
    stability: Optional[AtmosphericStability] = Field(
        default=None,
        description="Stability class"
    )

    # Quality
    overall_quality: str = Field(default="OK", description="Overall quality")
    missing_parameters: List[str] = Field(
        default_factory=list,
        description="Missing parameters"
    )


class MixingHeightEstimate(BaseModel):
    """Mixing height (boundary layer depth) estimate."""

    model_config = ConfigDict(frozen=True)

    timestamp: datetime = Field(default_factory=datetime.utcnow)

    mixing_height_m: float = Field(..., ge=0, description="Mixing height (m)")
    estimation_method: str = Field(..., description="Estimation method")
    confidence: str = Field(default="medium", description="Confidence level")

    # Supporting data
    surface_temperature_c: Optional[float] = Field(default=None)
    max_temperature_c: Optional[float] = Field(default=None)
    stability_class: Optional[StabilityClass] = Field(default=None)
    wind_speed_mps: Optional[float] = Field(default=None, ge=0)


class WeatherConnectorConfig(BaseConnectorConfig):
    """Configuration for weather connector."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    connector_type: ConnectorType = Field(
        default=ConnectorType.WEATHER,
        description="Connector type"
    )

    # Station identification
    station_id: str = Field(..., description="Station identifier")
    station_name: str = Field(..., description="Station name")

    # Location
    latitude: float = Field(..., ge=-90, le=90, description="Latitude")
    longitude: float = Field(..., ge=-180, le=180, description="Longitude")
    elevation_m: float = Field(default=0, ge=0, description="Station elevation (m)")
    timezone: str = Field(default="UTC", description="Timezone")

    # On-site met station settings
    onsite_enabled: bool = Field(default=False, description="On-site station enabled")
    onsite_vendor: MetStationVendor = Field(
        default=MetStationVendor.GENERIC,
        description="Met station vendor"
    )
    onsite_protocol: str = Field(default="modbus_tcp", description="Protocol")
    onsite_host: Optional[str] = Field(default=None, description="IP address")
    onsite_port: int = Field(default=502, ge=1, le=65535)

    # Sensor heights
    wind_height_m: float = Field(default=10.0, ge=0, description="Wind sensor height")
    temperature_height_m: float = Field(
        default=2.0,
        ge=0,
        description="Temperature sensor height"
    )
    upper_temp_height_m: Optional[float] = Field(
        default=None,
        description="Upper temperature height"
    )

    # NWS API settings
    nws_enabled: bool = Field(default=True, description="NWS API enabled")
    nws_api_base_url: str = Field(
        default="https://api.weather.gov",
        description="NWS API base URL"
    )
    nws_station_id: Optional[str] = Field(
        default=None,
        description="NWS observation station ID"
    )
    nws_grid_point: Optional[str] = Field(
        default=None,
        description="NWS grid point (office/gridX,gridY)"
    )

    # Data collection settings
    polling_interval_seconds: float = Field(
        default=60.0,
        ge=10.0,
        le=3600.0,
        description="Polling interval"
    )
    averaging_period_minutes: int = Field(
        default=60,
        ge=1,
        le=1440,
        description="Averaging period"
    )

    # Quality control
    wind_speed_max_mph: float = Field(default=100.0, ge=0)
    temp_min_f: float = Field(default=-50.0)
    temp_max_f: float = Field(default=130.0)


# =============================================================================
# Stability Classification
# =============================================================================


class StabilityClassifier:
    """
    Atmospheric stability classifier using Pasquill-Gifford method.

    Implements multiple classification methods:
    - Turner method (solar radiation + wind speed)
    - Sigma theta method (wind direction variability)
    - Delta T method (temperature difference)
    """

    # Pasquill stability class lookup (Turner method)
    # Key: (wind_speed_class, radiation_class)
    TURNER_LOOKUP = {
        # Wind < 2 m/s
        (1, "strong"): StabilityClass.A,
        (1, "moderate"): StabilityClass.A,
        (1, "slight"): StabilityClass.B,
        (1, "cloudy"): StabilityClass.D,
        (1, "night_clear"): StabilityClass.F,
        (1, "night_cloudy"): StabilityClass.E,
        # Wind 2-3 m/s
        (2, "strong"): StabilityClass.A,
        (2, "moderate"): StabilityClass.B,
        (2, "slight"): StabilityClass.C,
        (2, "cloudy"): StabilityClass.D,
        (2, "night_clear"): StabilityClass.E,
        (2, "night_cloudy"): StabilityClass.D,
        # Wind 3-5 m/s
        (3, "strong"): StabilityClass.B,
        (3, "moderate"): StabilityClass.B,
        (3, "slight"): StabilityClass.C,
        (3, "cloudy"): StabilityClass.D,
        (3, "night_clear"): StabilityClass.D,
        (3, "night_cloudy"): StabilityClass.D,
        # Wind 5-6 m/s
        (4, "strong"): StabilityClass.C,
        (4, "moderate"): StabilityClass.C,
        (4, "slight"): StabilityClass.D,
        (4, "cloudy"): StabilityClass.D,
        (4, "night_clear"): StabilityClass.D,
        (4, "night_cloudy"): StabilityClass.D,
        # Wind > 6 m/s
        (5, "strong"): StabilityClass.C,
        (5, "moderate"): StabilityClass.D,
        (5, "slight"): StabilityClass.D,
        (5, "cloudy"): StabilityClass.D,
        (5, "night_clear"): StabilityClass.D,
        (5, "night_cloudy"): StabilityClass.D,
    }

    # Sigma theta ranges for stability classification
    SIGMA_THETA_RANGES = {
        StabilityClass.A: (22.5, 180.0),
        StabilityClass.B: (17.5, 22.5),
        StabilityClass.C: (12.5, 17.5),
        StabilityClass.D: (7.5, 12.5),
        StabilityClass.E: (3.75, 7.5),
        StabilityClass.F: (0.0, 3.75),
    }

    def __init__(self, latitude: float, longitude: float) -> None:
        """
        Initialize classifier.

        Args:
            latitude: Site latitude
            longitude: Site longitude
        """
        self._latitude = latitude
        self._longitude = longitude
        self._logger = logging.getLogger("weather.stability")

    def classify_turner(
        self,
        wind_speed_mps: float,
        solar_radiation_wm2: Optional[float],
        cloud_cover_fraction: float,
        is_daytime: bool,
    ) -> StabilityClass:
        """
        Classify stability using Turner method.

        Args:
            wind_speed_mps: Wind speed (m/s)
            solar_radiation_wm2: Solar radiation (W/m2)
            cloud_cover_fraction: Cloud cover (0-1)
            is_daytime: Daytime flag

        Returns:
            Stability class
        """
        # Determine wind speed class
        if wind_speed_mps < 2:
            wind_class = 1
        elif wind_speed_mps < 3:
            wind_class = 2
        elif wind_speed_mps < 5:
            wind_class = 3
        elif wind_speed_mps < 6:
            wind_class = 4
        else:
            wind_class = 5

        # Determine radiation class
        if is_daytime:
            if solar_radiation_wm2 is not None:
                if solar_radiation_wm2 > 700:
                    rad_class = "strong"
                elif solar_radiation_wm2 > 350:
                    rad_class = "moderate"
                else:
                    rad_class = "slight"
            else:
                # Use cloud cover if no radiation data
                if cloud_cover_fraction < 0.4:
                    rad_class = "strong"
                elif cloud_cover_fraction < 0.7:
                    rad_class = "moderate"
                else:
                    rad_class = "cloudy"
        else:
            if cloud_cover_fraction < 0.5:
                rad_class = "night_clear"
            else:
                rad_class = "night_cloudy"

        # Look up stability
        stability = self.TURNER_LOOKUP.get(
            (wind_class, rad_class),
            StabilityClass.D  # Default to neutral
        )

        self._logger.debug(
            f"Turner classification: wind={wind_speed_mps:.1f} m/s, "
            f"rad_class={rad_class} -> {stability.value}"
        )

        return stability

    def classify_sigma_theta(self, sigma_theta: float) -> StabilityClass:
        """
        Classify stability using sigma theta method.

        Args:
            sigma_theta: Wind direction standard deviation (degrees)

        Returns:
            Stability class
        """
        for stability, (lower, upper) in self.SIGMA_THETA_RANGES.items():
            if lower <= sigma_theta < upper:
                return stability

        return StabilityClass.D

    def estimate_mixing_height(
        self,
        stability: StabilityClass,
        wind_speed_mps: float,
        temperature_c: float,
        is_daytime: bool,
    ) -> MixingHeightEstimate:
        """
        Estimate mixing height based on stability.

        Args:
            stability: Stability class
            wind_speed_mps: Wind speed (m/s)
            temperature_c: Surface temperature (C)
            is_daytime: Daytime flag

        Returns:
            Mixing height estimate
        """
        # Simple parameterization based on stability
        # More sophisticated methods would use actual sounding data
        mixing_heights = {
            StabilityClass.A: 2000,
            StabilityClass.B: 1500,
            StabilityClass.C: 1000,
            StabilityClass.D: 800,
            StabilityClass.E: 300,
            StabilityClass.F: 100,
            StabilityClass.G: 50,
        }

        base_height = mixing_heights.get(stability, 800)

        # Adjust for wind speed (mechanical mixing)
        wind_factor = 1.0 + (wind_speed_mps - 5) * 0.05

        # Adjust for time of day
        if not is_daytime:
            base_height *= 0.5  # Lower at night

        mixing_height = max(50, base_height * wind_factor)

        return MixingHeightEstimate(
            mixing_height_m=mixing_height,
            estimation_method="parameterized",
            confidence="medium",
            surface_temperature_c=temperature_c,
            stability_class=stability,
            wind_speed_mps=wind_speed_mps,
        )

    def is_daytime(self, timestamp: datetime) -> bool:
        """
        Determine if timestamp is during daytime.

        Uses simple solar angle calculation.

        Args:
            timestamp: Timestamp to check

        Returns:
            True if daytime
        """
        # Calculate solar elevation
        day_of_year = timestamp.timetuple().tm_yday
        hour_angle = (timestamp.hour + timestamp.minute / 60 - 12) * 15

        declination = 23.45 * math.sin(math.radians((360 / 365) * (day_of_year - 81)))

        sin_elevation = (
            math.sin(math.radians(self._latitude)) *
            math.sin(math.radians(declination)) +
            math.cos(math.radians(self._latitude)) *
            math.cos(math.radians(declination)) *
            math.cos(math.radians(hour_angle))
        )

        solar_elevation = math.degrees(math.asin(sin_elevation))

        return solar_elevation > 0


# =============================================================================
# NWS API Client
# =============================================================================


class NWSAPIClient:
    """
    National Weather Service API client.

    Retrieves current observations and forecasts from NWS API.
    """

    def __init__(
        self,
        base_url: str = "https://api.weather.gov",
        station_id: Optional[str] = None,
    ) -> None:
        """
        Initialize NWS client.

        Args:
            base_url: API base URL
            station_id: Observation station ID
        """
        self._base_url = base_url
        self._station_id = station_id
        self._client: Optional[httpx.AsyncClient] = None
        self._logger = logging.getLogger("weather.nws")

    async def initialize(self) -> None:
        """Initialize HTTP client."""
        self._client = httpx.AsyncClient(
            timeout=30.0,
            headers={
                "User-Agent": "(GreenLang EMISSIONWATCH, contact@greenlang.io)",
                "Accept": "application/geo+json",
            },
        )

    async def close(self) -> None:
        """Close HTTP client."""
        if self._client:
            await self._client.aclose()

    async def get_current_observation(
        self,
        station_id: Optional[str] = None,
    ) -> MeteorologicalObservation:
        """
        Get current observation from NWS station.

        Args:
            station_id: Station ID (uses configured if not provided)

        Returns:
            Meteorological observation
        """
        if not self._client:
            await self.initialize()

        sid = station_id or self._station_id
        if not sid:
            raise ConfigurationError("NWS station ID not configured")

        try:
            url = f"{self._base_url}/stations/{sid}/observations/latest"
            response = await self._client.get(url)
            response.raise_for_status()

            data = response.json()
            props = data.get("properties", {})

            # Parse wind data
            wind = None
            if props.get("windSpeed") and props.get("windDirection"):
                wind_speed_mps = props["windSpeed"].get("value", 0) or 0
                wind_dir = props["windDirection"].get("value", 0) or 0

                wind = WindData(
                    speed_mph=wind_speed_mps * 2.237,  # m/s to mph
                    speed_mps=wind_speed_mps,
                    direction_degrees=wind_dir,
                    direction_cardinal=self._degrees_to_cardinal(wind_dir),
                )

            # Parse temperature
            temperature = None
            if props.get("temperature"):
                temp_c = props["temperature"].get("value")
                if temp_c is not None:
                    temperature = TemperatureData(
                        temperature_f=temp_c * 9 / 5 + 32,
                        temperature_c=temp_c,
                        relative_humidity_percent=props.get("relativeHumidity", {}).get("value"),
                    )

            # Parse pressure
            pressure = None
            if props.get("barometricPressure"):
                pressure_pa = props["barometricPressure"].get("value")
                if pressure_pa is not None:
                    pressure = PressureData(
                        station_pressure_inhg=pressure_pa * 0.0002953,
                        station_pressure_mb=pressure_pa / 100,
                    )

            return MeteorologicalObservation(
                timestamp=datetime.utcnow(),
                data_source=DataSource.NWS,
                station_id=sid,
                wind=wind,
                temperature=temperature,
                pressure=pressure,
            )

        except httpx.HTTPError as e:
            self._logger.error(f"NWS API error: {e}")
            raise ConnectionError(f"NWS API request failed: {e}")

    async def get_forecast(
        self,
        latitude: float,
        longitude: float,
    ) -> Dict[str, Any]:
        """
        Get forecast for a location.

        Args:
            latitude: Latitude
            longitude: Longitude

        Returns:
            Forecast data
        """
        if not self._client:
            await self.initialize()

        try:
            # Get grid point
            url = f"{self._base_url}/points/{latitude},{longitude}"
            response = await self._client.get(url)
            response.raise_for_status()

            point_data = response.json()
            forecast_url = point_data.get("properties", {}).get("forecast")

            if not forecast_url:
                raise ValidationError("Could not determine forecast endpoint")

            # Get forecast
            response = await self._client.get(forecast_url)
            response.raise_for_status()

            return response.json()

        except httpx.HTTPError as e:
            self._logger.error(f"NWS forecast error: {e}")
            raise ConnectionError(f"NWS forecast request failed: {e}")

    def _degrees_to_cardinal(self, degrees: float) -> WindDirection:
        """Convert degrees to cardinal direction."""
        if degrees < 0:
            return WindDirection.CALM

        directions = [
            WindDirection.N, WindDirection.NNE, WindDirection.NE, WindDirection.ENE,
            WindDirection.E, WindDirection.ESE, WindDirection.SE, WindDirection.SSE,
            WindDirection.S, WindDirection.SSW, WindDirection.SW, WindDirection.WSW,
            WindDirection.W, WindDirection.WNW, WindDirection.NW, WindDirection.NNW,
        ]

        index = int((degrees + 11.25) / 22.5) % 16
        return directions[index]


# =============================================================================
# Weather Connector
# =============================================================================


class WeatherConnector(BaseConnector):
    """
    Weather and Meteorological Data Connector.

    Provides integration for meteorological data required for:
    - Air dispersion modeling (AERMOD, CALPUFF)
    - Emissions calculations
    - Regulatory reporting

    Features:
    - On-site met station integration
    - NWS API integration
    - Atmospheric stability classification
    - Mixing height estimation
    - Data quality validation
    """

    def __init__(self, config: WeatherConnectorConfig) -> None:
        """
        Initialize weather connector.

        Args:
            config: Connector configuration
        """
        super().__init__(config)
        self._weather_config = config

        # Initialize NWS client if enabled
        self._nws_client: Optional[NWSAPIClient] = None
        if config.nws_enabled:
            self._nws_client = NWSAPIClient(
                base_url=config.nws_api_base_url,
                station_id=config.nws_station_id,
            )

        # Initialize stability classifier
        self._stability_classifier = StabilityClassifier(
            config.latitude,
            config.longitude,
        )

        # Current data
        self._current_observation: Optional[MeteorologicalObservation] = None
        self._observation_history: List[MeteorologicalObservation] = []

        # Polling
        self._polling_task: Optional[asyncio.Task] = None
        self._polling_active = False

        # Callbacks
        self._data_callbacks: List[Callable[[MeteorologicalObservation], None]] = []

        self._logger = logging.getLogger(f"weather.connector.{config.station_id}")

    # -------------------------------------------------------------------------
    # Abstract Method Implementations
    # -------------------------------------------------------------------------

    async def connect(self) -> None:
        """
        Establish connections to weather data sources.

        Raises:
            ConnectionError: If connection fails
        """
        self._state = ConnectionState.CONNECTING
        self._logger.info("Connecting to weather data sources")

        try:
            # Initialize NWS client
            if self._nws_client:
                await self._nws_client.initialize()

            # TODO: Connect to on-site station if configured

            self._state = ConnectionState.CONNECTED
            self._logger.info("Weather connector connected")

            await self._audit_logger.log_operation(
                operation="connect",
                status="success",
            )

        except Exception as e:
            self._state = ConnectionState.ERROR
            raise ConnectionError(f"Failed to connect to weather sources: {e}")

    async def disconnect(self) -> None:
        """Disconnect from weather data sources."""
        self._logger.info("Disconnecting from weather sources")

        await self.stop_polling()

        if self._nws_client:
            await self._nws_client.close()

        self._state = ConnectionState.DISCONNECTED

        await self._audit_logger.log_operation(
            operation="disconnect",
            status="success",
        )

    async def health_check(self) -> HealthCheckResult:
        """
        Perform health check on weather data sources.

        Returns:
            Health check result
        """
        start_time = time.time()

        try:
            # Test NWS connection
            if self._nws_client and self._weather_config.nws_station_id:
                await self._nws_client.get_current_observation()

            latency_ms = (time.time() - start_time) * 1000

            return HealthCheckResult(
                status=HealthStatus.HEALTHY,
                latency_ms=latency_ms,
                message="Weather data sources healthy",
                details={
                    "nws_enabled": self._weather_config.nws_enabled,
                    "onsite_enabled": self._weather_config.onsite_enabled,
                },
            )

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                latency_ms=latency_ms,
                message=f"Health check failed: {e}",
            )

    async def validate_configuration(self) -> bool:
        """
        Validate weather connector configuration.

        Returns:
            True if configuration is valid
        """
        issues: List[str] = []

        if not self._weather_config.nws_enabled and not self._weather_config.onsite_enabled:
            issues.append("At least one data source must be enabled")

        if self._weather_config.nws_enabled and not self._weather_config.nws_station_id:
            issues.append("NWS station ID required when NWS is enabled")

        if self._weather_config.onsite_enabled and not self._weather_config.onsite_host:
            issues.append("On-site host required when on-site is enabled")

        if issues:
            raise ConfigurationError(
                f"Invalid weather configuration: {issues}",
                connector_id=self._config.connector_id,
            )

        return True

    # -------------------------------------------------------------------------
    # Weather-Specific Methods
    # -------------------------------------------------------------------------

    async def get_current_observation(self) -> MeteorologicalObservation:
        """
        Get current meteorological observation.

        Returns:
            Current observation
        """
        start_time = time.time()

        try:
            # Get data from available sources
            observation = None

            if self._nws_client:
                observation = await self._nws_client.get_current_observation()

            if observation is None:
                raise ConnectionError("No weather data available")

            # Add stability classification
            if observation.wind and observation.temperature:
                is_day = self._stability_classifier.is_daytime(observation.timestamp)

                cloud_fraction = 0.5  # Default
                if observation.clouds:
                    cloud_fraction = (observation.clouds.cloud_cover_percent or 50) / 100

                stability = self._stability_classifier.classify_turner(
                    wind_speed_mps=observation.wind.speed_mps or observation.wind.speed_mph / 2.237,
                    solar_radiation_wm2=observation.solar.global_horizontal_wm2 if observation.solar else None,
                    cloud_cover_fraction=cloud_fraction,
                    is_daytime=is_day,
                )

                stability_data = AtmosphericStability(
                    stability_class=stability,
                    classification_method="turner",
                    wind_speed_mps=observation.wind.speed_mps,
                    cloud_cover_fraction=cloud_fraction,
                    is_daytime=is_day,
                )

                # Create new observation with stability
                observation = MeteorologicalObservation(
                    observation_id=observation.observation_id,
                    timestamp=observation.timestamp,
                    data_source=observation.data_source,
                    station_id=observation.station_id,
                    wind=observation.wind,
                    temperature=observation.temperature,
                    pressure=observation.pressure,
                    solar=observation.solar,
                    precipitation=observation.precipitation,
                    clouds=observation.clouds,
                    stability=stability_data,
                )

            self._current_observation = observation
            self._observation_history.append(observation)

            # Trim history
            max_history = 1440  # 24 hours at 1-minute intervals
            if len(self._observation_history) > max_history:
                self._observation_history = self._observation_history[-max_history:]

            duration_ms = (time.time() - start_time) * 1000
            await self._metrics.record_request(
                success=True,
                latency_ms=duration_ms,
            )

            return observation

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            await self._metrics.record_request(
                success=False,
                latency_ms=duration_ms,
                error=str(e),
            )
            raise

    async def get_stability_class(self) -> AtmosphericStability:
        """
        Get current atmospheric stability classification.

        Returns:
            Stability classification
        """
        if self._current_observation and self._current_observation.stability:
            return self._current_observation.stability

        observation = await self.get_current_observation()
        if observation.stability:
            return observation.stability

        raise DataQualityError("Could not determine stability class")

    async def get_mixing_height(self) -> MixingHeightEstimate:
        """
        Get current mixing height estimate.

        Returns:
            Mixing height estimate
        """
        observation = await self.get_current_observation()

        if not observation.wind or not observation.temperature:
            raise DataQualityError("Insufficient data for mixing height")

        stability = observation.stability
        if not stability:
            raise DataQualityError("Stability class not available")

        mixing_height = self._stability_classifier.estimate_mixing_height(
            stability=stability.stability_class,
            wind_speed_mps=observation.wind.speed_mps or 0,
            temperature_c=observation.temperature.temperature_c or 20,
            is_daytime=stability.is_daytime,
        )

        await self._audit_logger.log_operation(
            operation="get_mixing_height",
            status="success",
            response_summary=f"Mixing height: {mixing_height.mixing_height_m:.0f} m",
        )

        return mixing_height

    async def get_forecast(self) -> Dict[str, Any]:
        """
        Get weather forecast.

        Returns:
            Forecast data
        """
        if not self._nws_client:
            raise ConfigurationError("NWS not configured for forecasts")

        return await self._nws_client.get_forecast(
            self._weather_config.latitude,
            self._weather_config.longitude,
        )

    def get_observation_history(
        self,
        hours: int = 24,
    ) -> List[MeteorologicalObservation]:
        """
        Get observation history.

        Args:
            hours: Number of hours of history

        Returns:
            List of observations
        """
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        return [
            obs for obs in self._observation_history
            if obs.timestamp >= cutoff
        ]

    async def start_polling(
        self,
        callback: Optional[Callable[[MeteorologicalObservation], None]] = None,
    ) -> None:
        """
        Start continuous data polling.

        Args:
            callback: Optional callback for new data
        """
        if self._polling_active:
            return

        if callback:
            self._data_callbacks.append(callback)

        self._polling_active = True
        self._polling_task = asyncio.create_task(self._polling_loop())

        self._logger.info("Started weather polling")

    async def stop_polling(self) -> None:
        """Stop data polling."""
        self._polling_active = False

        if self._polling_task:
            self._polling_task.cancel()
            try:
                await self._polling_task
            except asyncio.CancelledError:
                pass

        self._logger.info("Stopped weather polling")

    async def _polling_loop(self) -> None:
        """Background polling loop."""
        while self._polling_active:
            try:
                observation = await self.get_current_observation()

                for callback in self._data_callbacks:
                    try:
                        if asyncio.iscoroutinefunction(callback):
                            await callback(observation)
                        else:
                            callback(observation)
                    except Exception as e:
                        self._logger.error(f"Callback error: {e}")

                await asyncio.sleep(self._weather_config.polling_interval_seconds)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self._logger.error(f"Polling error: {e}")
                await asyncio.sleep(self._weather_config.polling_interval_seconds)


# =============================================================================
# Factory Function
# =============================================================================


def create_weather_connector(
    station_id: str,
    latitude: float,
    longitude: float,
    nws_station_id: Optional[str] = None,
    **kwargs: Any,
) -> WeatherConnector:
    """
    Factory function to create weather connector.

    Args:
        station_id: Station identifier
        latitude: Site latitude
        longitude: Site longitude
        nws_station_id: NWS observation station ID
        **kwargs: Additional configuration

    Returns:
        Configured weather connector
    """
    config = WeatherConnectorConfig(
        connector_name=f"Weather_{station_id}",
        station_id=station_id,
        station_name=station_id,
        latitude=latitude,
        longitude=longitude,
        nws_enabled=nws_station_id is not None,
        nws_station_id=nws_station_id,
        **kwargs,
    )

    return WeatherConnector(config)
