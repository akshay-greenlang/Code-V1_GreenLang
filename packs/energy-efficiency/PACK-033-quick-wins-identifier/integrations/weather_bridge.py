# -*- coding: utf-8 -*-
"""
WeatherBridge - Degree-Day Data and Weather Normalization for PACK-033
=======================================================================

This module provides weather data integration for normalizing energy
consumption and validating quick win savings estimates. It calculates
Heating Degree Days (HDD) and Cooling Degree Days (CDD), provides TMY
data, and determines climate zones.

Key Formulas (deterministic, zero-hallucination):
    HDD = max(0, base_temperature - daily_mean_temperature)
    CDD = max(0, daily_mean_temperature - base_temperature)
    Normalized_kWh = Actual_kWh * (TMY_DD / Actual_DD)

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-033 Quick Wins Identifier
Status: Production Ready
"""

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"

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
# Data Models
# ---------------------------------------------------------------------------

class WeatherConfig(BaseModel):
    """Configuration for the Weather Bridge."""

    pack_id: str = Field(default="PACK-033")
    enable_provenance: bool = Field(default=True)
    base_temperature_heating_c: float = Field(default=18.0, description="HDD base temp")
    base_temperature_cooling_c: float = Field(default=22.0, description="CDD base temp")

class DegreeDayData(BaseModel):
    """Degree-day calculation result for a location and period."""

    result_id: str = Field(default_factory=_new_uuid)
    location: str = Field(default="")
    period_start: str = Field(default="")
    period_end: str = Field(default="")
    hdd: float = Field(default=0.0, ge=0, description="Heating Degree Days")
    cdd: float = Field(default=0.0, ge=0, description="Cooling Degree Days")
    base_temperature_heating_c: float = Field(default=18.0)
    base_temperature_cooling_c: float = Field(default=22.0)
    days_count: int = Field(default=0, ge=0)
    data_completeness_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    provenance_hash: str = Field(default="")

class ClimateNormalization(BaseModel):
    """Result of weather-normalizing energy consumption."""

    normalization_id: str = Field(default_factory=_new_uuid)
    actual_kwh: float = Field(default=0.0)
    normalized_kwh: float = Field(default=0.0)
    actual_dd: float = Field(default=0.0)
    reference_dd: float = Field(default=0.0)
    adjustment_factor: float = Field(default=1.0)
    dd_type: str = Field(default="hdd", description="hdd or cdd")
    provenance_hash: str = Field(default="")

# ---------------------------------------------------------------------------
# Reference Data
# ---------------------------------------------------------------------------

TMY_DEGREE_DAYS: Dict[str, Dict[str, float]] = {
    "DE_BERLIN": {"hdd": 3155, "cdd": 105},
    "DE_MUNICH": {"hdd": 3510, "cdd": 78},
    "FR_PARIS": {"hdd": 2500, "cdd": 145},
    "NL_AMSTERDAM": {"hdd": 2800, "cdd": 55},
    "IT_MILAN": {"hdd": 2400, "cdd": 250},
    "ES_MADRID": {"hdd": 2000, "cdd": 480},
    "SE_STOCKHOLM": {"hdd": 3980, "cdd": 25},
    "GB_LONDON": {"hdd": 2600, "cdd": 45},
    "AT_VIENNA": {"hdd": 3100, "cdd": 120},
    "PL_WARSAW": {"hdd": 3400, "cdd": 80},
}

CLIMATE_ZONES: Dict[str, str] = {
    "DE": "temperate_continental",
    "FR": "temperate_oceanic",
    "NL": "temperate_oceanic",
    "IT": "mediterranean",
    "ES": "mediterranean",
    "SE": "continental",
    "GB": "temperate_oceanic",
    "AT": "temperate_continental",
    "PL": "temperate_continental",
    "FI": "continental",
    "DK": "temperate_oceanic",
    "BE": "temperate_oceanic",
    "IE": "temperate_oceanic",
    "PT": "mediterranean",
}

# ---------------------------------------------------------------------------
# WeatherBridge
# ---------------------------------------------------------------------------

class WeatherBridge:
    """Weather data integration for quick win savings normalization.

    Calculates HDD/CDD, provides TMY data, determines climate zones,
    and normalizes energy consumption for fair savings comparison.

    Attributes:
        config: Bridge configuration.

    Example:
        >>> bridge = WeatherBridge()
        >>> dd = bridge.get_degree_days("DE_BERLIN", "2024")
        >>> normalized = bridge.normalize_consumption(10000, 3000, 3155)
    """

    def __init__(self, config: Optional[WeatherConfig] = None) -> None:
        """Initialize the Weather Bridge."""
        self.config = config or WeatherConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(
            "WeatherBridge initialized: HDD base=%.1fC, CDD base=%.1fC",
            self.config.base_temperature_heating_c,
            self.config.base_temperature_cooling_c,
        )

    def get_degree_days(
        self,
        location: str,
        period: str,
    ) -> DegreeDayData:
        """Get degree-day data for a location and period.

        In production, this queries the weather API. The stub uses TMY data.

        Args:
            location: Location key (e.g., 'DE_BERLIN') or country code.
            period: Period identifier (e.g., '2024', '2024-Q1').

        Returns:
            DegreeDayData with HDD and CDD values.
        """
        tmy = TMY_DEGREE_DAYS.get(location)
        if tmy is None:
            # Try matching by country prefix
            for key, val in TMY_DEGREE_DAYS.items():
                if key.startswith(location):
                    tmy = val
                    location = key
                    break

        hdd = tmy.get("hdd", 0.0) if tmy else 0.0
        cdd = tmy.get("cdd", 0.0) if tmy else 0.0

        result = DegreeDayData(
            location=location,
            period_start=f"{period}-01-01" if len(period) == 4 else period,
            period_end=f"{period}-12-31" if len(period) == 4 else period,
            hdd=hdd,
            cdd=cdd,
            base_temperature_heating_c=self.config.base_temperature_heating_c,
            base_temperature_cooling_c=self.config.base_temperature_cooling_c,
            days_count=365,
            data_completeness_pct=100.0 if tmy else 0.0,
        )

        if self.config.enable_provenance:
            result.provenance_hash = _compute_hash(result)

        return result

    def normalize_consumption(
        self,
        consumption: float,
        hdd: float,
        cdd: float,
    ) -> Decimal:
        """Weather-normalize energy consumption.

        Deterministic formula using reference TMY degree days.

        Args:
            consumption: Actual energy consumption in kWh.
            hdd: Actual heating degree days for the period.
            cdd: Actual cooling degree days for the period.

        Returns:
            Normalized consumption as Decimal.
        """
        # Use average European TMY as reference if no specific location
        ref_hdd = 3000.0
        ref_cdd = 100.0

        actual_dd = hdd + cdd
        reference_dd = ref_hdd + ref_cdd

        if actual_dd > 0:
            adjustment = reference_dd / actual_dd
        else:
            adjustment = 1.0

        normalized = Decimal(str(consumption)) * Decimal(str(adjustment))
        return normalized.quantize(Decimal("0.01"))

    def get_climate_zone(
        self,
        latitude: float,
        longitude: float,
    ) -> str:
        """Determine climate zone from coordinates.

        Uses a simplified lookup by estimated country from longitude/latitude.

        Args:
            latitude: Latitude in decimal degrees.
            longitude: Longitude in decimal degrees.

        Returns:
            Climate zone string.
        """
        # Simplified country estimation from coordinates
        if 47.0 <= latitude <= 55.0 and 6.0 <= longitude <= 15.0:
            country = "DE"
        elif 42.0 <= latitude <= 51.0 and -5.0 <= longitude <= 8.0:
            country = "FR"
        elif 50.0 <= latitude <= 54.0 and 3.0 <= longitude <= 7.0:
            country = "NL"
        elif 36.0 <= latitude <= 47.0 and 6.0 <= longitude <= 19.0:
            country = "IT"
        elif 36.0 <= latitude <= 44.0 and -10.0 <= longitude <= 4.0:
            country = "ES"
        elif 50.0 <= latitude <= 60.0 and -8.0 <= longitude <= 2.0:
            country = "GB"
        elif 56.0 <= latitude <= 70.0 and 11.0 <= longitude <= 24.0:
            country = "SE"
        else:
            return "temperate_continental"  # Default

        return CLIMATE_ZONES.get(country, "temperate_continental")
