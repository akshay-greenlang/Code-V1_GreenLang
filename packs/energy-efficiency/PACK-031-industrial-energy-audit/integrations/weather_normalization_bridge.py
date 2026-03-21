# -*- coding: utf-8 -*-
"""
WeatherNormalizationBridge - Weather Data Integration for Baseline Adjustment
===============================================================================

This module provides weather data integration for normalizing energy consumption
baselines. It calculates Heating Degree Days (HDD) and Cooling Degree Days (CDD),
manages weather data sources, supports Typical Meteorological Year (TMY) data,
determines climate zones, and computes seasonal adjustment factors.

Key Formulas (deterministic, zero-hallucination):
    HDD = max(0, base_temperature - daily_mean_temperature)
    CDD = max(0, daily_mean_temperature - base_temperature)
    Normalized_kWh = Actual_kWh * (TMY_DD / Actual_DD)

Climate Zone Classification (simplified Koppen-Geiger):
    Tropical (A), Dry (B), Temperate (C), Continental (D), Polar (E)

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-031 Industrial Energy Audit
Status: Production Ready
"""

import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, date, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"


def _utcnow() -> datetime:
    """Return current UTC datetime."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _new_uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())


def _compute_hash(data: Any) -> str:
    """Compute SHA-256 hash for provenance tracking."""
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ClimateZone(str, Enum):
    """Simplified Koppen-Geiger climate zones."""

    TROPICAL = "tropical"
    DRY = "dry"
    TEMPERATE_OCEANIC = "temperate_oceanic"
    TEMPERATE_CONTINENTAL = "temperate_continental"
    CONTINENTAL = "continental"
    POLAR = "polar"
    MEDITERRANEAN = "mediterranean"


class DegreeDayMethod(str, Enum):
    """Degree-day calculation methods."""

    DAILY_MEAN = "daily_mean"
    HOURLY_INTEGRATION = "hourly_integration"
    MIN_MAX_AVERAGE = "min_max_average"


class WeatherSource(str, Enum):
    """Weather data source types."""

    WEATHER_STATION = "weather_station"
    TMY_FILE = "tmy_file"
    API_SERVICE = "api_service"
    MANUAL_ENTRY = "manual_entry"


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


class WeatherStationConfig(BaseModel):
    """Configuration for a weather data source."""

    station_id: str = Field(default_factory=_new_uuid)
    name: str = Field(default="")
    source_type: WeatherSource = Field(default=WeatherSource.WEATHER_STATION)
    latitude: float = Field(default=0.0, ge=-90, le=90)
    longitude: float = Field(default=0.0, ge=-180, le=180)
    altitude_m: float = Field(default=0.0)
    country_code: str = Field(default="")
    wmo_id: Optional[str] = Field(None, description="WMO station identifier")
    distance_to_facility_km: float = Field(default=0.0, ge=0)


class DailyWeatherRecord(BaseModel):
    """Daily weather observation record."""

    date: str = Field(default="", description="ISO date string")
    mean_temperature_c: float = Field(default=0.0)
    max_temperature_c: float = Field(default=0.0)
    min_temperature_c: float = Field(default=0.0)
    relative_humidity_pct: float = Field(default=0.0, ge=0, le=100)
    wind_speed_ms: float = Field(default=0.0, ge=0)
    solar_radiation_wm2: float = Field(default=0.0, ge=0)
    precipitation_mm: float = Field(default=0.0, ge=0)


class DegreeDayResult(BaseModel):
    """Result of degree-day calculation for a period."""

    result_id: str = Field(default_factory=_new_uuid)
    period_start: str = Field(default="")
    period_end: str = Field(default="")
    hdd: float = Field(default=0.0, ge=0, description="Heating Degree Days")
    cdd: float = Field(default=0.0, ge=0, description="Cooling Degree Days")
    base_temperature_heating_c: float = Field(default=18.0)
    base_temperature_cooling_c: float = Field(default=22.0)
    method: DegreeDayMethod = Field(default=DegreeDayMethod.DAILY_MEAN)
    days_count: int = Field(default=0, ge=0)
    data_completeness_pct: float = Field(default=0.0, ge=0, le=100)
    provenance_hash: str = Field(default="")


class WeatherNormalizationResult(BaseModel):
    """Result of weather-normalizing energy consumption."""

    normalization_id: str = Field(default_factory=_new_uuid)
    actual_kwh: float = Field(default=0.0)
    normalized_kwh: float = Field(default=0.0)
    actual_dd: float = Field(default=0.0, description="Actual degree days")
    reference_dd: float = Field(default=0.0, description="TMY/reference degree days")
    adjustment_factor: float = Field(default=1.0)
    dd_type: str = Field(default="hdd", description="hdd or cdd")
    base_load_kwh: float = Field(default=0.0, description="Weather-independent base load")
    weather_dependent_kwh: float = Field(default=0.0)
    provenance_hash: str = Field(default="")


class SeasonalAdjustmentFactor(BaseModel):
    """Monthly or seasonal adjustment factors."""

    month: int = Field(default=1, ge=1, le=12)
    month_name: str = Field(default="")
    hdd_factor: float = Field(default=1.0)
    cdd_factor: float = Field(default=1.0)
    production_factor: float = Field(default=1.0)
    combined_factor: float = Field(default=1.0)


class WeatherNormalizationBridgeConfig(BaseModel):
    """Configuration for the Weather Normalization Bridge."""

    pack_id: str = Field(default="PACK-031")
    enable_provenance: bool = Field(default=True)
    base_temperature_heating_c: float = Field(default=18.0, description="HDD base temp")
    base_temperature_cooling_c: float = Field(default=22.0, description="CDD base temp")
    default_method: DegreeDayMethod = Field(default=DegreeDayMethod.DAILY_MEAN)
    base_load_estimation_method: str = Field(
        default="summer_minimum", description="summer_minimum|regression|manual"
    )


# ---------------------------------------------------------------------------
# Default TMY Degree Days (selected European cities)
# ---------------------------------------------------------------------------

TMY_DEGREE_DAYS: Dict[str, Dict[str, float]] = {
    "DE_BERLIN": {"hdd": 3155, "cdd": 105},
    "DE_MUNICH": {"hdd": 3510, "cdd": 78},
    "DE_HAMBURG": {"hdd": 3200, "cdd": 65},
    "FR_PARIS": {"hdd": 2500, "cdd": 145},
    "NL_AMSTERDAM": {"hdd": 2800, "cdd": 55},
    "IT_MILAN": {"hdd": 2400, "cdd": 250},
    "IT_ROME": {"hdd": 1415, "cdd": 410},
    "ES_MADRID": {"hdd": 2000, "cdd": 480},
    "SE_STOCKHOLM": {"hdd": 3980, "cdd": 25},
    "PL_WARSAW": {"hdd": 3400, "cdd": 80},
    "AT_VIENNA": {"hdd": 3100, "cdd": 120},
    "GB_LONDON": {"hdd": 2600, "cdd": 45},
    "FI_HELSINKI": {"hdd": 4600, "cdd": 15},
    "DK_COPENHAGEN": {"hdd": 3200, "cdd": 35},
}

# Climate zone by country (simplified)
COUNTRY_CLIMATE_ZONES: Dict[str, ClimateZone] = {
    "DE": ClimateZone.TEMPERATE_CONTINENTAL,
    "FR": ClimateZone.TEMPERATE_OCEANIC,
    "NL": ClimateZone.TEMPERATE_OCEANIC,
    "IT": ClimateZone.MEDITERRANEAN,
    "ES": ClimateZone.MEDITERRANEAN,
    "SE": ClimateZone.CONTINENTAL,
    "PL": ClimateZone.TEMPERATE_CONTINENTAL,
    "AT": ClimateZone.TEMPERATE_CONTINENTAL,
    "GB": ClimateZone.TEMPERATE_OCEANIC,
    "FI": ClimateZone.CONTINENTAL,
    "DK": ClimateZone.TEMPERATE_OCEANIC,
    "BE": ClimateZone.TEMPERATE_OCEANIC,
    "CZ": ClimateZone.TEMPERATE_CONTINENTAL,
    "RO": ClimateZone.TEMPERATE_CONTINENTAL,
    "PT": ClimateZone.MEDITERRANEAN,
    "IE": ClimateZone.TEMPERATE_OCEANIC,
}

MONTH_NAMES = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]


# ---------------------------------------------------------------------------
# WeatherNormalizationBridge
# ---------------------------------------------------------------------------


class WeatherNormalizationBridge:
    """Weather data integration for baseline energy normalization.

    Calculates HDD/CDD, manages weather data sources, provides TMY data,
    determines climate zones, and computes weather-normalized energy baselines.

    Attributes:
        config: Bridge configuration.
        _stations: Registered weather stations.
        _weather_data: Stored daily weather records by station.

    Example:
        >>> bridge = WeatherNormalizationBridge()
        >>> dd = bridge.calculate_degree_days(daily_records)
        >>> normalized = bridge.normalize_consumption(10000, dd.hdd, 3155)
    """

    def __init__(self, config: Optional[WeatherNormalizationBridgeConfig] = None) -> None:
        """Initialize the Weather Normalization Bridge."""
        self.config = config or WeatherNormalizationBridgeConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        self._stations: Dict[str, WeatherStationConfig] = {}
        self._weather_data: Dict[str, List[DailyWeatherRecord]] = {}
        self.logger.info(
            "WeatherNormalizationBridge initialized: HDD base=%.1fC, CDD base=%.1fC",
            self.config.base_temperature_heating_c,
            self.config.base_temperature_cooling_c,
        )

    # -------------------------------------------------------------------------
    # Weather Station Management
    # -------------------------------------------------------------------------

    def register_station(self, station: WeatherStationConfig) -> WeatherStationConfig:
        """Register a weather data source.

        Args:
            station: Weather station configuration.

        Returns:
            Registered WeatherStationConfig.
        """
        self._stations[station.station_id] = station
        self._weather_data[station.station_id] = []
        self.logger.info("Weather station registered: %s (%s)", station.name, station.source_type.value)
        return station

    def load_weather_data(
        self, station_id: str, records: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Load daily weather records for a station.

        Args:
            station_id: Weather station identifier.
            records: List of daily weather data dicts.

        Returns:
            Dict with load summary.
        """
        if station_id not in self._stations:
            return {"station_id": station_id, "success": False, "message": "Station not found"}

        loaded = 0
        for r in records:
            try:
                record = DailyWeatherRecord(**r)
                self._weather_data[station_id].append(record)
                loaded += 1
            except Exception as exc:
                self.logger.warning("Weather record parse failed: %s", exc)

        return {"station_id": station_id, "success": True, "records_loaded": loaded}

    # -------------------------------------------------------------------------
    # Degree-Day Calculation
    # -------------------------------------------------------------------------

    def calculate_degree_days(
        self,
        daily_records: List[DailyWeatherRecord],
        base_temp_heating: Optional[float] = None,
        base_temp_cooling: Optional[float] = None,
    ) -> DegreeDayResult:
        """Calculate HDD and CDD from daily weather records.

        Deterministic formulas:
            HDD_day = max(0, base_heating - daily_mean)
            CDD_day = max(0, daily_mean - base_cooling)
            HDD_total = sum(HDD_day for each day)
            CDD_total = sum(CDD_day for each day)

        Args:
            daily_records: List of daily weather observations.
            base_temp_heating: Override HDD base temperature.
            base_temp_cooling: Override CDD base temperature.

        Returns:
            DegreeDayResult with HDD and CDD totals.
        """
        base_h = base_temp_heating or self.config.base_temperature_heating_c
        base_c = base_temp_cooling or self.config.base_temperature_cooling_c

        total_hdd = 0.0
        total_cdd = 0.0

        for record in daily_records:
            mean_t = record.mean_temperature_c
            # Deterministic HDD/CDD calculation
            hdd_day = max(0.0, base_h - mean_t)
            cdd_day = max(0.0, mean_t - base_c)
            total_hdd += hdd_day
            total_cdd += cdd_day

        days_count = len(daily_records)
        completeness = (days_count / 365.0 * 100.0) if days_count > 0 else 0.0

        dates = [r.date for r in daily_records if r.date]
        period_start = min(dates) if dates else ""
        period_end = max(dates) if dates else ""

        result = DegreeDayResult(
            period_start=period_start,
            period_end=period_end,
            hdd=round(total_hdd, 1),
            cdd=round(total_cdd, 1),
            base_temperature_heating_c=base_h,
            base_temperature_cooling_c=base_c,
            method=self.config.default_method,
            days_count=days_count,
            data_completeness_pct=round(min(completeness, 100.0), 1),
        )
        if self.config.enable_provenance:
            result.provenance_hash = _compute_hash(result)

        self.logger.info(
            "Degree days calculated: HDD=%.1f, CDD=%.1f (%d days)",
            total_hdd, total_cdd, days_count,
        )
        return result

    # -------------------------------------------------------------------------
    # Weather Normalization
    # -------------------------------------------------------------------------

    def normalize_consumption(
        self,
        actual_kwh: float,
        actual_dd: float,
        reference_dd: float,
        base_load_kwh: float = 0.0,
        dd_type: str = "hdd",
    ) -> WeatherNormalizationResult:
        """Weather-normalize energy consumption.

        Deterministic formula:
            weather_dependent = actual_kwh - base_load
            adjustment = reference_dd / actual_dd
            normalized = base_load + (weather_dependent * adjustment)

        Args:
            actual_kwh: Actual energy consumption.
            actual_dd: Actual degree days for the period.
            reference_dd: TMY or reference degree days.
            base_load_kwh: Weather-independent base load.
            dd_type: Type of degree days ('hdd' or 'cdd').

        Returns:
            WeatherNormalizationResult with normalized consumption.
        """
        weather_dependent = actual_kwh - base_load_kwh

        if actual_dd > 0:
            adjustment = reference_dd / actual_dd
        else:
            adjustment = 1.0

        normalized = base_load_kwh + (weather_dependent * adjustment)

        result = WeatherNormalizationResult(
            actual_kwh=actual_kwh,
            normalized_kwh=round(normalized, 2),
            actual_dd=actual_dd,
            reference_dd=reference_dd,
            adjustment_factor=round(adjustment, 4),
            dd_type=dd_type,
            base_load_kwh=base_load_kwh,
            weather_dependent_kwh=round(weather_dependent, 2),
        )
        if self.config.enable_provenance:
            result.provenance_hash = _compute_hash(result)

        return result

    # -------------------------------------------------------------------------
    # TMY Data
    # -------------------------------------------------------------------------

    def get_tmy_degree_days(self, location_key: str) -> Optional[Dict[str, float]]:
        """Get TMY degree days for a location.

        Args:
            location_key: Location key (e.g., 'DE_BERLIN').

        Returns:
            Dict with 'hdd' and 'cdd', or None if not found.
        """
        return TMY_DEGREE_DAYS.get(location_key)

    def list_tmy_locations(self) -> List[str]:
        """List available TMY location keys.

        Returns:
            Sorted list of location keys.
        """
        return sorted(TMY_DEGREE_DAYS.keys())

    # -------------------------------------------------------------------------
    # Climate Zone
    # -------------------------------------------------------------------------

    def get_climate_zone(self, country_code: str) -> Optional[str]:
        """Determine climate zone for a country.

        Args:
            country_code: ISO 3166-1 alpha-2 country code.

        Returns:
            Climate zone string, or None if not found.
        """
        zone = COUNTRY_CLIMATE_ZONES.get(country_code.upper())
        return zone.value if zone else None

    # -------------------------------------------------------------------------
    # Seasonal Adjustment
    # -------------------------------------------------------------------------

    def calculate_seasonal_factors(
        self,
        monthly_dd: List[float],
        monthly_production: Optional[List[float]] = None,
    ) -> List[SeasonalAdjustmentFactor]:
        """Calculate monthly seasonal adjustment factors.

        Deterministic calculation:
            factor = month_dd / average_dd

        Args:
            monthly_dd: List of 12 monthly degree-day values.
            monthly_production: Optional list of 12 monthly production values.

        Returns:
            List of 12 SeasonalAdjustmentFactor.
        """
        if len(monthly_dd) != 12:
            self.logger.warning("Expected 12 monthly values, got %d", len(monthly_dd))
            return []

        avg_dd = sum(monthly_dd) / 12.0 if sum(monthly_dd) > 0 else 1.0
        avg_prod = (
            sum(monthly_production) / 12.0
            if monthly_production and sum(monthly_production) > 0
            else 1.0
        )

        factors: List[SeasonalAdjustmentFactor] = []
        for i in range(12):
            dd_factor = monthly_dd[i] / avg_dd if avg_dd > 0 else 1.0
            prod_factor = (
                monthly_production[i] / avg_prod
                if monthly_production and avg_prod > 0
                else 1.0
            )
            combined = dd_factor * prod_factor

            factors.append(SeasonalAdjustmentFactor(
                month=i + 1,
                month_name=MONTH_NAMES[i],
                hdd_factor=round(dd_factor, 4),
                cdd_factor=round(dd_factor, 4),
                production_factor=round(prod_factor, 4),
                combined_factor=round(combined, 4),
            ))

        return factors

    # -------------------------------------------------------------------------
    # Health Check
    # -------------------------------------------------------------------------

    def check_health(self) -> Dict[str, Any]:
        """Check weather data health.

        Returns:
            Dict with health metrics.
        """
        total_records = sum(len(r) for r in self._weather_data.values())
        return {
            "stations_registered": len(self._stations),
            "total_weather_records": total_records,
            "tmy_locations_available": len(TMY_DEGREE_DAYS),
            "climate_zones_covered": len(COUNTRY_CLIMATE_ZONES),
            "status": "healthy",
        }
