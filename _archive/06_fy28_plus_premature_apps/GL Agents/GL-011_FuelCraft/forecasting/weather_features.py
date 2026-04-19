# -*- coding: utf-8 -*-
"""
Weather Features Module for GL-011 FuelCraft

Provides weather-based features for fuel price forecasting including
HDD/CDD calculations, storm risk signals, and shipping disruption indicators.

Features:
- Temperature-based features (HDD/CDD)
- Wind speed and precipitation features
- Storm risk assessment
- Shipping disruption signals
- Region mapping with stable identifiers

Zero-Hallucination Architecture:
- All calculations use deterministic formulas
- Standard meteorological definitions
- No LLM-based weather interpretation
- Complete provenance tracking

Author: GreenLang AI Team
Version: 1.0.0
"""

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

# Constants
DEFAULT_HDD_BASE_TEMP_F = 65.0
DEFAULT_CDD_BASE_TEMP_F = 65.0
PRECISION = 2

# Storm thresholds
WIND_THRESHOLD_WARNING = 25.0  # mph
WIND_THRESHOLD_SEVERE = 50.0  # mph
PRECIP_THRESHOLD_WARNING = 1.0  # inches
PRECIP_THRESHOLD_SEVERE = 3.0  # inches


class WeatherRegion(str, Enum):
    """Standard weather regions for fuel markets."""
    NORTHEAST = "northeast"
    SOUTHEAST = "southeast"
    MIDWEST = "midwest"
    SOUTHWEST = "southwest"
    WEST_COAST = "west_coast"
    GULF_COAST = "gulf_coast"
    MOUNTAIN = "mountain"
    PACIFIC_NORTHWEST = "pacific_northwest"


class StormSeverity(str, Enum):
    """Storm severity levels."""
    NONE = "none"
    WATCH = "watch"
    WARNING = "warning"
    SEVERE = "severe"
    EXTREME = "extreme"


class DisruptionType(str, Enum):
    """Types of shipping/supply disruptions."""
    NONE = "none"
    PIPELINE = "pipeline"
    PORT = "port"
    RAIL = "rail"
    ROAD = "road"
    MULTIPLE = "multiple"


class WeatherObservation(BaseModel):
    """
    Single weather observation with all relevant fields.
    """

    observation_id: str = Field(..., description="Unique observation ID")
    station_id: str = Field(..., description="Weather station identifier")
    region: WeatherRegion = Field(..., description="Weather region")
    observation_time: datetime = Field(..., description="Observation timestamp")

    # Temperature
    temperature_f: float = Field(..., description="Temperature in Fahrenheit")
    temperature_min_f: Optional[float] = Field(None, description="Daily minimum")
    temperature_max_f: Optional[float] = Field(None, description="Daily maximum")

    # Wind
    wind_speed_mph: float = Field(0.0, description="Wind speed in mph")
    wind_gust_mph: Optional[float] = Field(None, description="Wind gust in mph")
    wind_direction_deg: Optional[float] = Field(None, description="Wind direction degrees")

    # Precipitation
    precipitation_in: float = Field(0.0, description="Precipitation in inches")
    snow_in: Optional[float] = Field(None, description="Snowfall in inches")

    # Other
    humidity_pct: Optional[float] = Field(None, description="Relative humidity %")
    pressure_mb: Optional[float] = Field(None, description="Barometric pressure")
    visibility_mi: Optional[float] = Field(None, description="Visibility in miles")

    # Quality
    quality_flag: str = Field("good", description="Data quality flag")

    @field_validator("temperature_f")
    @classmethod
    def validate_temperature(cls, v: float) -> float:
        """Validate temperature is within reasonable range."""
        if v < -100 or v > 150:
            raise ValueError(f"Temperature {v}F outside reasonable range")
        return v


class HDDCDDResult(BaseModel):
    """
    Heating and Cooling Degree Days calculation result.
    """

    calculation_date: datetime = Field(..., description="Date of calculation")
    region: WeatherRegion = Field(..., description="Region")

    # HDD/CDD values
    hdd: float = Field(..., ge=0.0, description="Heating Degree Days")
    cdd: float = Field(..., ge=0.0, description="Cooling Degree Days")

    # Base temperatures
    hdd_base_temp_f: float = Field(DEFAULT_HDD_BASE_TEMP_F)
    cdd_base_temp_f: float = Field(DEFAULT_CDD_BASE_TEMP_F)

    # Calculation inputs
    mean_temperature_f: float = Field(..., description="Mean temperature used")
    observation_count: int = Field(1, description="Number of observations")

    # Cumulative values
    hdd_cumulative_month: Optional[float] = Field(None, description="Month-to-date HDD")
    cdd_cumulative_month: Optional[float] = Field(None, description="Month-to-date CDD")
    hdd_cumulative_year: Optional[float] = Field(None, description="Year-to-date HDD")
    cdd_cumulative_year: Optional[float] = Field(None, description="Year-to-date CDD")

    # Normals comparison
    hdd_normal: Optional[float] = Field(None, description="Normal HDD for this date")
    cdd_normal: Optional[float] = Field(None, description="Normal CDD for this date")
    hdd_deviation: Optional[float] = Field(None, description="Deviation from normal")
    cdd_deviation: Optional[float] = Field(None, description="Deviation from normal")

    # Provenance
    provenance_hash: str = Field("", description="SHA-256 hash")

    def model_post_init(self, __context: Any) -> None:
        """Compute provenance hash."""
        if not self.provenance_hash:
            self.provenance_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        """Compute SHA-256 provenance hash."""
        data = {
            "calculation_date": self.calculation_date.isoformat(),
            "region": self.region.value,
            "hdd": self.hdd,
            "cdd": self.cdd,
            "mean_temperature_f": self.mean_temperature_f,
        }
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


class StormRiskSignal(BaseModel):
    """
    Storm risk assessment for a region.
    """

    signal_id: str = Field(..., description="Unique signal ID")
    region: WeatherRegion = Field(..., description="Affected region")
    assessment_time: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # Risk levels
    severity: StormSeverity = Field(..., description="Storm severity")
    risk_score: float = Field(..., ge=0.0, le=1.0, description="Risk score 0-1")

    # Contributing factors
    wind_risk: float = Field(0.0, ge=0.0, le=1.0, description="Wind-based risk")
    precipitation_risk: float = Field(0.0, ge=0.0, le=1.0, description="Precipitation risk")
    temperature_risk: float = Field(0.0, ge=0.0, le=1.0, description="Temperature risk")

    # Metrics
    max_wind_speed_mph: Optional[float] = Field(None)
    expected_precipitation_in: Optional[float] = Field(None)
    duration_hours: Optional[float] = Field(None, description="Expected duration")

    # Confidence
    forecast_confidence: float = Field(0.8, ge=0.0, le=1.0)

    # Message
    summary: str = Field("", description="Human-readable summary")


class ShippingDisruptionSignal(BaseModel):
    """
    Shipping and supply chain disruption signal.
    """

    signal_id: str = Field(..., description="Unique signal ID")
    region: WeatherRegion = Field(..., description="Affected region")
    assessment_time: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # Disruption info
    disruption_type: DisruptionType = Field(..., description="Type of disruption")
    severity: StormSeverity = Field(..., description="Disruption severity")
    disruption_score: float = Field(..., ge=0.0, le=1.0, description="Disruption score 0-1")

    # Affected infrastructure
    affected_pipelines: List[str] = Field(default_factory=list)
    affected_ports: List[str] = Field(default_factory=list)
    affected_terminals: List[str] = Field(default_factory=list)

    # Timing
    expected_start: Optional[datetime] = Field(None)
    expected_duration_hours: Optional[float] = Field(None)
    recovery_probability: float = Field(0.5, ge=0.0, le=1.0)

    # Market impact
    estimated_supply_impact_pct: Optional[float] = Field(None)
    estimated_price_impact_pct: Optional[float] = Field(None)

    # Confidence
    forecast_confidence: float = Field(0.8, ge=0.0, le=1.0)


class RegionMapping(BaseModel):
    """
    Mapping between weather regions and market hubs.
    """

    region_id: str = Field(..., description="Stable region identifier")
    region: WeatherRegion = Field(..., description="Weather region")
    display_name: str = Field(..., description="Human-readable name")

    # Geographic info
    states: List[str] = Field(default_factory=list, description="US states in region")
    primary_city: str = Field(..., description="Primary city for reference")
    latitude: float = Field(..., description="Center latitude")
    longitude: float = Field(..., description="Center longitude")

    # Market associations
    natural_gas_hubs: List[str] = Field(default_factory=list)
    electricity_markets: List[str] = Field(default_factory=list)
    coal_basins: List[str] = Field(default_factory=list)

    # Weather stations
    primary_weather_station: str = Field(..., description="Primary weather station ID")
    backup_stations: List[str] = Field(default_factory=list)


@dataclass
class WeatherFeatureConfig:
    """Configuration for weather feature extraction."""

    hdd_base_temp_f: float = DEFAULT_HDD_BASE_TEMP_F
    cdd_base_temp_f: float = DEFAULT_CDD_BASE_TEMP_F
    precision: int = PRECISION
    wind_warning_threshold: float = WIND_THRESHOLD_WARNING
    wind_severe_threshold: float = WIND_THRESHOLD_SEVERE
    precip_warning_threshold: float = PRECIP_THRESHOLD_WARNING
    precip_severe_threshold: float = PRECIP_THRESHOLD_SEVERE
    cache_enabled: bool = True
    cache_ttl_seconds: int = 3600


class WeatherFeatureExtractor:
    """
    Weather feature extractor for fuel price forecasting.

    Provides deterministic calculation of weather-based features
    including HDD/CDD, storm risk, and disruption signals.

    Zero-Hallucination Guarantees:
    - Standard meteorological formulas
    - Deterministic calculations
    - No LLM-based interpretation
    """

    def __init__(self, config: Optional[WeatherFeatureConfig] = None):
        """
        Initialize weather feature extractor.

        Args:
            config: Feature extraction configuration
        """
        self.config = config or WeatherFeatureConfig()

        # Region mappings
        self._region_mappings: Dict[WeatherRegion, RegionMapping] = {}
        self._initialize_region_mappings()

        # Normals data (historical averages)
        self._hdd_normals: Dict[str, Dict[int, float]] = {}
        self._cdd_normals: Dict[str, Dict[int, float]] = {}

        # Cache
        self._cache: Dict[str, Tuple[datetime, Any]] = {}

        logger.info(
            f"WeatherFeatureExtractor initialized: "
            f"HDD base={self.config.hdd_base_temp_f}F, "
            f"CDD base={self.config.cdd_base_temp_f}F"
        )

    def calculate_hdd_cdd(
        self,
        observation: WeatherObservation,
        use_mean: bool = True
    ) -> HDDCDDResult:
        """
        Calculate Heating and Cooling Degree Days.

        HDD = max(0, base_temp - mean_temp)
        CDD = max(0, mean_temp - base_temp)

        Args:
            observation: Weather observation
            use_mean: Use mean of min/max if available

        Returns:
            HDDCDDResult with calculated values
        """
        # Calculate mean temperature
        if use_mean and observation.temperature_min_f and observation.temperature_max_f:
            mean_temp = (observation.temperature_min_f + observation.temperature_max_f) / 2
        else:
            mean_temp = observation.temperature_f

        # Calculate HDD and CDD
        hdd = max(0.0, self.config.hdd_base_temp_f - mean_temp)
        cdd = max(0.0, mean_temp - self.config.cdd_base_temp_f)

        # Round to precision
        hdd = self._round_value(hdd)
        cdd = self._round_value(cdd)

        # Get normals if available
        day_of_year = observation.observation_time.timetuple().tm_yday
        hdd_normal = self._get_hdd_normal(observation.region, day_of_year)
        cdd_normal = self._get_cdd_normal(observation.region, day_of_year)

        hdd_deviation = None
        cdd_deviation = None

        if hdd_normal is not None:
            hdd_deviation = hdd - hdd_normal

        if cdd_normal is not None:
            cdd_deviation = cdd - cdd_normal

        return HDDCDDResult(
            calculation_date=observation.observation_time,
            region=observation.region,
            hdd=hdd,
            cdd=cdd,
            hdd_base_temp_f=self.config.hdd_base_temp_f,
            cdd_base_temp_f=self.config.cdd_base_temp_f,
            mean_temperature_f=self._round_value(mean_temp),
            observation_count=1,
            hdd_normal=hdd_normal,
            cdd_normal=cdd_normal,
            hdd_deviation=hdd_deviation,
            cdd_deviation=cdd_deviation,
        )

    def calculate_hdd_cdd_batch(
        self,
        observations: List[WeatherObservation],
        aggregate: bool = True
    ) -> List[HDDCDDResult]:
        """
        Calculate HDD/CDD for multiple observations.

        Args:
            observations: List of weather observations
            aggregate: Aggregate by date if True

        Returns:
            List of HDDCDDResult objects
        """
        results = [self.calculate_hdd_cdd(obs) for obs in observations]

        if not aggregate:
            return results

        # Aggregate by date and region
        aggregated: Dict[str, HDDCDDResult] = {}

        for result in results:
            key = f"{result.calculation_date.date()}_{result.region.value}"

            if key not in aggregated:
                aggregated[key] = result
            else:
                # Average the values
                existing = aggregated[key]
                existing.hdd = self._round_value((existing.hdd + result.hdd) / 2)
                existing.cdd = self._round_value((existing.cdd + result.cdd) / 2)
                existing.observation_count += 1

        return list(aggregated.values())

    def assess_storm_risk(
        self,
        observations: List[WeatherObservation],
        region: WeatherRegion
    ) -> StormRiskSignal:
        """
        Assess storm risk for a region.

        Args:
            observations: Recent weather observations
            region: Region to assess

        Returns:
            StormRiskSignal with risk assessment
        """
        import uuid

        # Filter observations for region
        region_obs = [o for o in observations if o.region == region]

        if not region_obs:
            return StormRiskSignal(
                signal_id=str(uuid.uuid4()),
                region=region,
                severity=StormSeverity.NONE,
                risk_score=0.0,
                summary="No observations available for region",
            )

        # Calculate risk components
        max_wind = max(o.wind_speed_mph for o in region_obs)
        max_gust = max((o.wind_gust_mph or o.wind_speed_mph) for o in region_obs)
        total_precip = sum(o.precipitation_in for o in region_obs)

        # Wind risk
        wind_risk = 0.0
        if max_wind >= self.config.wind_severe_threshold:
            wind_risk = 1.0
        elif max_wind >= self.config.wind_warning_threshold:
            wind_risk = 0.5 + 0.5 * (max_wind - self.config.wind_warning_threshold) / (
                self.config.wind_severe_threshold - self.config.wind_warning_threshold
            )
        elif max_wind > 15:
            wind_risk = max_wind / self.config.wind_warning_threshold * 0.5

        # Precipitation risk
        precip_risk = 0.0
        if total_precip >= self.config.precip_severe_threshold:
            precip_risk = 1.0
        elif total_precip >= self.config.precip_warning_threshold:
            precip_risk = 0.5 + 0.5 * (total_precip - self.config.precip_warning_threshold) / (
                self.config.precip_severe_threshold - self.config.precip_warning_threshold
            )
        elif total_precip > 0.25:
            precip_risk = total_precip / self.config.precip_warning_threshold * 0.5

        # Combined risk score
        risk_score = max(wind_risk, precip_risk) * 0.7 + min(wind_risk, precip_risk) * 0.3
        risk_score = min(1.0, risk_score)

        # Determine severity
        if risk_score >= 0.8:
            severity = StormSeverity.EXTREME
        elif risk_score >= 0.6:
            severity = StormSeverity.SEVERE
        elif risk_score >= 0.4:
            severity = StormSeverity.WARNING
        elif risk_score >= 0.2:
            severity = StormSeverity.WATCH
        else:
            severity = StormSeverity.NONE

        # Generate summary
        summary = self._generate_storm_summary(severity, max_wind, total_precip)

        return StormRiskSignal(
            signal_id=str(uuid.uuid4()),
            region=region,
            severity=severity,
            risk_score=self._round_value(risk_score),
            wind_risk=self._round_value(wind_risk),
            precipitation_risk=self._round_value(precip_risk),
            max_wind_speed_mph=max_wind,
            expected_precipitation_in=total_precip,
            summary=summary,
        )

    def assess_shipping_disruption(
        self,
        storm_signal: StormRiskSignal
    ) -> ShippingDisruptionSignal:
        """
        Assess shipping/supply disruption based on storm risk.

        Args:
            storm_signal: Storm risk signal

        Returns:
            ShippingDisruptionSignal with disruption assessment
        """
        import uuid

        # Map storm severity to disruption
        disruption_mapping = {
            StormSeverity.NONE: (DisruptionType.NONE, 0.0),
            StormSeverity.WATCH: (DisruptionType.NONE, 0.1),
            StormSeverity.WARNING: (DisruptionType.PIPELINE, 0.3),
            StormSeverity.SEVERE: (DisruptionType.MULTIPLE, 0.6),
            StormSeverity.EXTREME: (DisruptionType.MULTIPLE, 0.9),
        }

        disruption_type, base_score = disruption_mapping[storm_signal.severity]

        # Adjust based on region
        region_multiplier = self._get_region_disruption_multiplier(storm_signal.region)
        disruption_score = min(1.0, base_score * region_multiplier)

        # Estimate impacts
        supply_impact = disruption_score * 0.1  # Up to 10% supply impact
        price_impact = disruption_score * 0.15  # Up to 15% price impact

        # Get affected infrastructure
        affected_pipelines, affected_ports = self._get_affected_infrastructure(
            storm_signal.region
        )

        return ShippingDisruptionSignal(
            signal_id=str(uuid.uuid4()),
            region=storm_signal.region,
            disruption_type=disruption_type,
            severity=storm_signal.severity,
            disruption_score=self._round_value(disruption_score),
            affected_pipelines=affected_pipelines if disruption_score > 0.3 else [],
            affected_ports=affected_ports if disruption_score > 0.5 else [],
            estimated_supply_impact_pct=self._round_value(supply_impact * 100),
            estimated_price_impact_pct=self._round_value(price_impact * 100),
            forecast_confidence=storm_signal.forecast_confidence,
        )

    def extract_weather_features(
        self,
        observations: List[WeatherObservation],
        region: WeatherRegion
    ) -> Dict[str, float]:
        """
        Extract all weather features for a region.

        Args:
            observations: Weather observations
            region: Target region

        Returns:
            Dictionary of feature name to value
        """
        # Filter for region
        region_obs = [o for o in observations if o.region == region]

        if not region_obs:
            return self._get_default_features()

        # Calculate HDD/CDD
        hdd_cdd_results = self.calculate_hdd_cdd_batch(region_obs)
        total_hdd = sum(r.hdd for r in hdd_cdd_results)
        total_cdd = sum(r.cdd for r in hdd_cdd_results)

        # Calculate temperature stats
        temps = [o.temperature_f for o in region_obs]
        temp_mean = float(np.mean(temps))
        temp_std = float(np.std(temps)) if len(temps) > 1 else 0.0
        temp_min = float(np.min(temps))
        temp_max = float(np.max(temps))

        # Calculate wind stats
        winds = [o.wind_speed_mph for o in region_obs]
        wind_mean = float(np.mean(winds))
        wind_max = float(np.max(winds))

        # Calculate precipitation
        precip_total = sum(o.precipitation_in for o in region_obs)

        # Assess storm risk
        storm_signal = self.assess_storm_risk(region_obs, region)

        return {
            "heating_degree_days": self._round_value(total_hdd),
            "cooling_degree_days": self._round_value(total_cdd),
            "temperature_mean_f": self._round_value(temp_mean),
            "temperature_std_f": self._round_value(temp_std),
            "temperature_min_f": self._round_value(temp_min),
            "temperature_max_f": self._round_value(temp_max),
            "wind_speed_mean_mph": self._round_value(wind_mean),
            "wind_speed_max_mph": self._round_value(wind_max),
            "precipitation_total_in": self._round_value(precip_total),
            "storm_risk_score": storm_signal.risk_score,
            "observation_count": len(region_obs),
        }

    def get_region_mapping(self, region: WeatherRegion) -> Optional[RegionMapping]:
        """Get region mapping."""
        return self._region_mappings.get(region)

    def _initialize_region_mappings(self) -> None:
        """Initialize standard region mappings."""
        mappings = [
            RegionMapping(
                region_id="NE-001",
                region=WeatherRegion.NORTHEAST,
                display_name="Northeast",
                states=["NY", "NJ", "PA", "CT", "MA", "RI", "NH", "VT", "ME"],
                primary_city="New York",
                latitude=40.7128,
                longitude=-74.0060,
                natural_gas_hubs=["dominion_south", "transco_z6"],
                electricity_markets=["nyiso", "iso_ne", "pjm"],
                primary_weather_station="KNYC",
            ),
            RegionMapping(
                region_id="GC-001",
                region=WeatherRegion.GULF_COAST,
                display_name="Gulf Coast",
                states=["TX", "LA", "MS", "AL"],
                primary_city="Houston",
                latitude=29.7604,
                longitude=-95.3698,
                natural_gas_hubs=["henry_hub", "houston_ship_channel"],
                electricity_markets=["ercot"],
                primary_weather_station="KHOU",
            ),
            RegionMapping(
                region_id="MW-001",
                region=WeatherRegion.MIDWEST,
                display_name="Midwest",
                states=["IL", "IN", "OH", "MI", "WI", "MN", "IA", "MO"],
                primary_city="Chicago",
                latitude=41.8781,
                longitude=-87.6298,
                natural_gas_hubs=["chicago_citygate"],
                electricity_markets=["miso", "pjm"],
                coal_basins=["illinois_basin"],
                primary_weather_station="KORD",
            ),
            RegionMapping(
                region_id="WC-001",
                region=WeatherRegion.WEST_COAST,
                display_name="West Coast",
                states=["CA", "OR", "WA"],
                primary_city="Los Angeles",
                latitude=34.0522,
                longitude=-118.2437,
                natural_gas_hubs=["socal_citygate", "pge_citygate"],
                electricity_markets=["caiso"],
                primary_weather_station="KLAX",
            ),
        ]

        for mapping in mappings:
            self._region_mappings[mapping.region] = mapping

    def _get_hdd_normal(
        self,
        region: WeatherRegion,
        day_of_year: int
    ) -> Optional[float]:
        """Get normal HDD for region and day of year."""
        # Simplified normal calculation
        # In production, would use historical data
        if region in [WeatherRegion.NORTHEAST, WeatherRegion.MIDWEST]:
            if day_of_year < 90 or day_of_year > 300:  # Winter
                return 25.0
            elif day_of_year < 150 or day_of_year > 250:  # Spring/Fall
                return 10.0
            else:  # Summer
                return 0.0
        elif region == WeatherRegion.GULF_COAST:
            if day_of_year < 60 or day_of_year > 320:
                return 5.0
            else:
                return 0.0
        return None

    def _get_cdd_normal(
        self,
        region: WeatherRegion,
        day_of_year: int
    ) -> Optional[float]:
        """Get normal CDD for region and day of year."""
        if region in [WeatherRegion.GULF_COAST, WeatherRegion.SOUTHWEST]:
            if 120 < day_of_year < 280:  # Summer
                return 15.0
            elif 90 < day_of_year < 300:
                return 5.0
            else:
                return 0.0
        elif region == WeatherRegion.WEST_COAST:
            if 150 < day_of_year < 250:
                return 10.0
            else:
                return 2.0
        return None

    def _generate_storm_summary(
        self,
        severity: StormSeverity,
        max_wind: float,
        total_precip: float
    ) -> str:
        """Generate human-readable storm summary."""
        if severity == StormSeverity.NONE:
            return "No significant weather impacts expected."

        parts = []

        if severity == StormSeverity.WATCH:
            parts.append("Monitoring conditions")
        elif severity == StormSeverity.WARNING:
            parts.append("Weather warning in effect")
        elif severity == StormSeverity.SEVERE:
            parts.append("Severe weather expected")
        elif severity == StormSeverity.EXTREME:
            parts.append("Extreme weather conditions")

        if max_wind >= self.config.wind_warning_threshold:
            parts.append(f"winds up to {max_wind:.0f} mph")

        if total_precip >= self.config.precip_warning_threshold:
            parts.append(f"precipitation {total_precip:.1f} inches")

        return "; ".join(parts) + "."

    def _get_region_disruption_multiplier(self, region: WeatherRegion) -> float:
        """Get disruption sensitivity multiplier for region."""
        # Gulf Coast more sensitive due to infrastructure concentration
        multipliers = {
            WeatherRegion.GULF_COAST: 1.5,
            WeatherRegion.NORTHEAST: 1.2,
            WeatherRegion.MIDWEST: 1.0,
            WeatherRegion.WEST_COAST: 0.9,
            WeatherRegion.SOUTHEAST: 1.1,
            WeatherRegion.SOUTHWEST: 0.8,
            WeatherRegion.MOUNTAIN: 0.7,
            WeatherRegion.PACIFIC_NORTHWEST: 0.8,
        }
        return multipliers.get(region, 1.0)

    def _get_affected_infrastructure(
        self,
        region: WeatherRegion
    ) -> Tuple[List[str], List[str]]:
        """Get potentially affected infrastructure for region."""
        pipelines: Dict[WeatherRegion, List[str]] = {
            WeatherRegion.GULF_COAST: ["Gulf_Coast_Express", "Permian_Highway", "Valley_Crossing"],
            WeatherRegion.NORTHEAST: ["Transco", "Texas_Eastern", "Tennessee_Gas"],
            WeatherRegion.MIDWEST: ["Rockies_Express", "Northern_Border"],
        }

        ports: Dict[WeatherRegion, List[str]] = {
            WeatherRegion.GULF_COAST: ["Houston", "Corpus_Christi", "New_Orleans"],
            WeatherRegion.NORTHEAST: ["New_York", "Philadelphia", "Boston"],
            WeatherRegion.WEST_COAST: ["Los_Angeles", "Long_Beach", "Seattle"],
        }

        return pipelines.get(region, []), ports.get(region, [])

    def _get_default_features(self) -> Dict[str, float]:
        """Get default feature values when no data available."""
        return {
            "heating_degree_days": 0.0,
            "cooling_degree_days": 0.0,
            "temperature_mean_f": 65.0,
            "temperature_std_f": 0.0,
            "temperature_min_f": 65.0,
            "temperature_max_f": 65.0,
            "wind_speed_mean_mph": 5.0,
            "wind_speed_max_mph": 10.0,
            "precipitation_total_in": 0.0,
            "storm_risk_score": 0.0,
            "observation_count": 0,
        }

    def _round_value(self, value: float) -> float:
        """Round value to configured precision."""
        decimal_value = Decimal(str(value))
        rounded = decimal_value.quantize(
            Decimal(10) ** -self.config.precision,
            rounding=ROUND_HALF_UP
        )
        return float(rounded)


# Utility functions

def calculate_hdd(
    temperature_f: float,
    base_temp_f: float = DEFAULT_HDD_BASE_TEMP_F
) -> float:
    """
    Calculate Heating Degree Days.

    HDD = max(0, base_temp - actual_temp)

    Args:
        temperature_f: Actual temperature in Fahrenheit
        base_temp_f: Base temperature (default 65F)

    Returns:
        Heating degree days value
    """
    return max(0.0, base_temp_f - temperature_f)


def calculate_cdd(
    temperature_f: float,
    base_temp_f: float = DEFAULT_CDD_BASE_TEMP_F
) -> float:
    """
    Calculate Cooling Degree Days.

    CDD = max(0, actual_temp - base_temp)

    Args:
        temperature_f: Actual temperature in Fahrenheit
        base_temp_f: Base temperature (default 65F)

    Returns:
        Cooling degree days value
    """
    return max(0.0, temperature_f - base_temp_f)


def assess_storm_risk(
    wind_speed_mph: float,
    precipitation_in: float,
    wind_gust_mph: Optional[float] = None
) -> Tuple[StormSeverity, float]:
    """
    Quick storm risk assessment.

    Args:
        wind_speed_mph: Wind speed in mph
        precipitation_in: Precipitation in inches
        wind_gust_mph: Optional wind gust

    Returns:
        Tuple of (severity, risk_score)
    """
    max_wind = wind_gust_mph if wind_gust_mph else wind_speed_mph

    # Calculate risk score
    wind_risk = min(1.0, max_wind / WIND_THRESHOLD_SEVERE)
    precip_risk = min(1.0, precipitation_in / PRECIP_THRESHOLD_SEVERE)

    risk_score = max(wind_risk, precip_risk)

    # Determine severity
    if risk_score >= 0.8:
        severity = StormSeverity.EXTREME
    elif risk_score >= 0.6:
        severity = StormSeverity.SEVERE
    elif risk_score >= 0.4:
        severity = StormSeverity.WARNING
    elif risk_score >= 0.2:
        severity = StormSeverity.WATCH
    else:
        severity = StormSeverity.NONE

    return severity, risk_score
