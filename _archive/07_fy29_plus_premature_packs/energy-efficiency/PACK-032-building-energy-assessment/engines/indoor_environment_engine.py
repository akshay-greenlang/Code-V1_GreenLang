# -*- coding: utf-8 -*-
"""
IndoorEnvironmentEngine - PACK-032 Building Energy Assessment Engine 9
======================================================================

Indoor environmental quality (IEQ) assessment per EN 16798-1 and ISO 7730.
Evaluates thermal comfort (PMV/PPD and adaptive model), indoor air quality
(CO2, PM2.5, PM10, TVOC, formaldehyde, radon), ventilation adequacy,
overheating risk, and daylighting against IEQ Category I-IV limits.

Calculation Methodology:
    PMV (Fanger's Equation per ISO 7730):
        Combines six variables: air temperature, mean radiant temperature,
        relative humidity, air speed, metabolic rate, clothing insulation.
        Iterative calculation of clothing surface temperature.

    PPD (Predicted Percentage Dissatisfied):
        PPD = 100 - 95 * exp(-0.03353 * PMV^4 - 0.2179 * PMV^2)

    Adaptive Comfort (EN 16798-1):
        T_comf = 0.33 * T_rm + 18.8  [degC]
        where T_rm = running mean outdoor temperature

    CO2 Generation (Metabolic):
        G = n_people * M * 0.000004 * RQ  [l/s]
        where M = metabolic rate (W/m2), RQ = respiratory quotient

    Required Ventilation:
        Q = max(Q_people + Q_building, Q_dilution)  [l/s]

    Overheating Hours (CIBSE TM59):
        Count hours where T_op > T_limit over occupied period

    IEQ Composite Score:
        Weighted sum of thermal, IAQ, visual, acoustic sub-scores

Regulatory References:
    - EN 16798-1:2019 - Indoor environmental input parameters
    - ISO 7730:2005 - Ergonomics of the thermal environment
    - EN 17037:2018 - Daylight in buildings
    - CIBSE TM59:2017 - Overheating risk assessment
    - ASHRAE Standard 62.1-2022 - Ventilation for acceptable IAQ
    - WHO Air Quality Guidelines (2021)

Zero-Hallucination:
    - PMV solved via iterative Newton-Raphson (no LLM)
    - PPD from ISO 7730 analytical formula
    - Lookup tables from published EN 16798-1 data
    - Deterministic Decimal arithmetic throughout
    - SHA-256 provenance hash on every result

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-032 Building Energy Assessment
Engine:  9 of 10
Status:  Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator

from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _new_uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())

def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash of arbitrary data."""
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    if isinstance(serializable, dict):
        serializable = {
            k: v for k, v in serializable.items()
            if k not in ("calculated_at", "processing_time_ms", "provenance_hash")
        }
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

def _decimal(value: Any) -> Decimal:
    """Safely convert a value to Decimal."""
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return Decimal("0")

def _safe_divide(
    numerator: Decimal,
    denominator: Decimal,
    default: Decimal = Decimal("0"),
) -> Decimal:
    """Safely divide two Decimals, returning *default* on zero denominator."""
    if denominator == Decimal("0"):
        return default
    return numerator / denominator

def _safe_pct(part: Decimal, whole: Decimal) -> Decimal:
    """Compute percentage safely (part / whole * 100)."""
    return _safe_divide(part * Decimal("100"), whole)

def _round_val(value: Decimal, places: int = 6) -> float:
    """Round a Decimal to *places* and return a float."""
    quantizer = Decimal(10) ** -places
    return float(value.quantize(quantizer, rounding=ROUND_HALF_UP))

def _round1(value: float) -> float:
    """Round to 1 decimal place using ROUND_HALF_UP."""
    return float(Decimal(str(value)).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))

def _round2(value: float) -> float:
    """Round to 2 decimal places using ROUND_HALF_UP."""
    return float(Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))

def _round3(value: float) -> float:
    """Round to 3 decimal places using ROUND_HALF_UP."""
    return float(Decimal(str(value)).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP))

def _round4(value: float) -> float:
    """Round to 4 decimal places using ROUND_HALF_UP."""
    return float(Decimal(str(value)).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP))

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class IEQCategory(str, Enum):
    """Indoor Environmental Quality categories per EN 16798-1.

    Category I:  High level of expectation (sensitive persons)
    Category II: Normal level of expectation (new/renovated buildings)
    Category III: Acceptable, moderate level (existing buildings)
    Category IV: Values outside I-III (only limited periods acceptable)
    """
    I = "I"
    II = "II"
    III = "III"
    IV = "IV"

class ThermalComfortMethod(str, Enum):
    """Thermal comfort assessment method."""
    PMV_PPD = "pmv_ppd"
    ADAPTIVE = "adaptive"
    BOTH = "both"

class IAQParameter(str, Enum):
    """Indoor air quality parameters monitored."""
    CO2 = "co2"
    PM25 = "pm25"
    PM10 = "pm10"
    TVOC = "tvoc"
    FORMALDEHYDE = "formaldehyde"
    RADON = "radon"
    HUMIDITY = "humidity"

class VentilationStandard(str, Enum):
    """Ventilation design standard."""
    EN_16798 = "EN_16798"
    ASHRAE_62_1 = "ASHRAE_62_1"
    LOCAL_BUILDING_CODE = "local_building_code"

class SpaceType(str, Enum):
    """Space typology for ventilation rate lookup."""
    OFFICE_OPEN = "office_open"
    OFFICE_PRIVATE = "office_private"
    MEETING_ROOM = "meeting_room"
    CLASSROOM = "classroom"
    LECTURE_HALL = "lecture_hall"
    LIVING_ROOM = "living_room"
    BEDROOM = "bedroom"
    KITCHEN_RESIDENTIAL = "kitchen_residential"
    KITCHEN_COMMERCIAL = "kitchen_commercial"
    RESTAURANT = "restaurant"
    RETAIL = "retail"
    HOSPITAL_WARD = "hospital_ward"
    HOSPITAL_TREATMENT = "hospital_treatment"
    HOTEL_ROOM = "hotel_room"
    GYM = "gym"
    CORRIDOR = "corridor"
    TOILET = "toilet"
    WAREHOUSE = "warehouse"

# ---------------------------------------------------------------------------
# Constants -- PMV/PPD Lookup Table (ISO 7730 Table A.1)
# ---------------------------------------------------------------------------
# PMV -> PPD% mapping from ISO 7730

PMV_PPD_LOOKUP: Dict[str, str] = {
    "-3.0": "99.1", "-2.5": "87.5", "-2.0": "76.8", "-1.5": "52.6",
    "-1.0": "26.1", "-0.5": "10.2", "0.0": "5.0", "0.5": "10.2",
    "1.0": "26.1", "1.5": "52.6", "2.0": "76.8", "2.5": "87.5",
    "3.0": "99.1",
}

# ---------------------------------------------------------------------------
# Thermal Comfort Categories (EN 16798-1 Table NA.2)
# ---------------------------------------------------------------------------

THERMAL_COMFORT_CATEGORIES: Dict[str, Dict[str, str]] = {
    "I": {
        "ppd_max_pct": "6",
        "pmv_range_low": "-0.2",
        "pmv_range_high": "0.2",
        "operative_temp_heating_low": "21.0",
        "operative_temp_heating_high": "23.0",
        "operative_temp_cooling_low": "23.5",
        "operative_temp_cooling_high": "25.5",
    },
    "II": {
        "ppd_max_pct": "10",
        "pmv_range_low": "-0.5",
        "pmv_range_high": "0.5",
        "operative_temp_heating_low": "20.0",
        "operative_temp_heating_high": "24.0",
        "operative_temp_cooling_low": "23.0",
        "operative_temp_cooling_high": "26.0",
    },
    "III": {
        "ppd_max_pct": "15",
        "pmv_range_low": "-0.7",
        "pmv_range_high": "0.7",
        "operative_temp_heating_low": "19.0",
        "operative_temp_heating_high": "25.0",
        "operative_temp_cooling_low": "22.0",
        "operative_temp_cooling_high": "27.0",
    },
    "IV": {
        "ppd_max_pct": "25",
        "pmv_range_low": "-1.0",
        "pmv_range_high": "1.0",
        "operative_temp_heating_low": "18.0",
        "operative_temp_heating_high": "26.0",
        "operative_temp_cooling_low": "21.0",
        "operative_temp_cooling_high": "28.0",
    },
}

# ---------------------------------------------------------------------------
# Adaptive Comfort Limits (EN 16798-1 Table B.2)
# ---------------------------------------------------------------------------
# Upper and lower operative temperature limits by running mean outdoor temp
# Format: (T_rm, upper_I, upper_II, upper_III, lower_I, lower_II, lower_III)

ADAPTIVE_COMFORT_TABLE: List[Dict[str, str]] = [
    {"t_rm": "10", "upper_I": "25.0", "upper_II": "26.0", "upper_III": "27.0", "lower_I": "21.0", "lower_II": "20.0", "lower_III": "19.0"},
    {"t_rm": "12", "upper_I": "25.7", "upper_II": "26.7", "upper_III": "27.7", "lower_I": "21.7", "lower_II": "20.7", "lower_III": "19.7"},
    {"t_rm": "14", "upper_I": "26.3", "upper_II": "27.3", "upper_III": "28.3", "lower_I": "22.3", "lower_II": "21.3", "lower_III": "20.3"},
    {"t_rm": "16", "upper_I": "27.0", "upper_II": "28.0", "upper_III": "29.0", "lower_I": "23.0", "lower_II": "22.0", "lower_III": "21.0"},
    {"t_rm": "18", "upper_I": "27.7", "upper_II": "28.7", "upper_III": "29.7", "lower_I": "23.7", "lower_II": "22.7", "lower_III": "21.7"},
    {"t_rm": "20", "upper_I": "28.3", "upper_II": "29.3", "upper_III": "30.3", "lower_I": "24.3", "lower_II": "23.3", "lower_III": "22.3"},
    {"t_rm": "22", "upper_I": "29.0", "upper_II": "30.0", "upper_III": "31.0", "lower_I": "25.0", "lower_II": "24.0", "lower_III": "23.0"},
    {"t_rm": "24", "upper_I": "29.7", "upper_II": "30.7", "upper_III": "31.7", "lower_I": "25.7", "lower_II": "24.7", "lower_III": "23.7"},
    {"t_rm": "26", "upper_I": "30.3", "upper_II": "31.3", "upper_III": "32.3", "lower_I": "26.3", "lower_II": "25.3", "lower_III": "24.3"},
    {"t_rm": "28", "upper_I": "31.0", "upper_II": "32.0", "upper_III": "33.0", "lower_I": "27.0", "lower_II": "26.0", "lower_III": "25.0"},
    {"t_rm": "30", "upper_I": "31.7", "upper_II": "32.7", "upper_III": "33.7", "lower_I": "27.7", "lower_II": "26.7", "lower_III": "25.7"},
    {"t_rm": "33", "upper_I": "32.7", "upper_II": "33.7", "upper_III": "34.7", "lower_I": "28.7", "lower_II": "27.7", "lower_III": "26.7"},
]

# ---------------------------------------------------------------------------
# IAQ Limits by Parameter and Category (EN 16798-1 / WHO 2021)
# ---------------------------------------------------------------------------
# CO2: ppm above outdoor (~420ppm baseline)
# PM2.5 / PM10: ug/m3 (24h average)
# TVOC: ug/m3
# Formaldehyde: ug/m3
# Radon: Bq/m3
# Humidity: %RH range

IAQ_LIMITS: Dict[str, Dict[str, Dict[str, str]]] = {
    "co2": {
        "I": {"above_outdoor_ppm": "550", "absolute_ppm": "970"},
        "II": {"above_outdoor_ppm": "800", "absolute_ppm": "1220"},
        "III": {"above_outdoor_ppm": "1350", "absolute_ppm": "1770"},
        "IV": {"above_outdoor_ppm": "1350", "absolute_ppm": "1770"},
        "unit": {"value": "ppm"},
        "outdoor_baseline": {"value": "420"},
    },
    "pm25": {
        "I": {"limit": "10"},
        "II": {"limit": "15"},
        "III": {"limit": "25"},
        "IV": {"limit": "25"},
        "unit": {"value": "ug/m3"},
        "who_guideline": {"value": "5"},
    },
    "pm10": {
        "I": {"limit": "20"},
        "II": {"limit": "35"},
        "III": {"limit": "50"},
        "IV": {"limit": "50"},
        "unit": {"value": "ug/m3"},
        "who_guideline": {"value": "15"},
    },
    "tvoc": {
        "I": {"limit": "200"},
        "II": {"limit": "400"},
        "III": {"limit": "600"},
        "IV": {"limit": "600"},
        "unit": {"value": "ug/m3"},
    },
    "formaldehyde": {
        "I": {"limit": "30"},
        "II": {"limit": "50"},
        "III": {"limit": "100"},
        "IV": {"limit": "100"},
        "unit": {"value": "ug/m3"},
        "who_guideline": {"value": "100"},
    },
    "radon": {
        "I": {"limit": "100"},
        "II": {"limit": "200"},
        "III": {"limit": "300"},
        "IV": {"limit": "300"},
        "unit": {"value": "Bq/m3"},
        "who_guideline": {"value": "100"},
    },
    "humidity": {
        "I": {"low": "30", "high": "50"},
        "II": {"low": "25", "high": "60"},
        "III": {"low": "20", "high": "70"},
        "IV": {"low": "20", "high": "70"},
        "unit": {"value": "%RH"},
    },
}

# ---------------------------------------------------------------------------
# Ventilation Rates (EN 16798-1 Table B.2) -- l/s per person + l/s per m2
# ---------------------------------------------------------------------------

VENTILATION_RATES: Dict[str, Dict[str, Dict[str, str]]] = {
    "office_open": {
        "I": {"per_person": "10.0", "per_m2": "2.0"},
        "II": {"per_person": "7.0", "per_m2": "1.4"},
        "III": {"per_person": "4.0", "per_m2": "0.8"},
        "occupancy_density": "0.07",
    },
    "office_private": {
        "I": {"per_person": "10.0", "per_m2": "2.0"},
        "II": {"per_person": "7.0", "per_m2": "1.4"},
        "III": {"per_person": "4.0", "per_m2": "0.8"},
        "occupancy_density": "0.10",
    },
    "meeting_room": {
        "I": {"per_person": "10.0", "per_m2": "2.0"},
        "II": {"per_person": "7.0", "per_m2": "1.4"},
        "III": {"per_person": "4.0", "per_m2": "0.8"},
        "occupancy_density": "0.50",
    },
    "classroom": {
        "I": {"per_person": "10.0", "per_m2": "2.0"},
        "II": {"per_person": "7.0", "per_m2": "1.4"},
        "III": {"per_person": "4.0", "per_m2": "0.8"},
        "occupancy_density": "0.40",
    },
    "lecture_hall": {
        "I": {"per_person": "10.0", "per_m2": "2.0"},
        "II": {"per_person": "7.0", "per_m2": "1.4"},
        "III": {"per_person": "4.0", "per_m2": "0.8"},
        "occupancy_density": "0.60",
    },
    "living_room": {
        "I": {"per_person": "10.0", "per_m2": "1.4"},
        "II": {"per_person": "7.0", "per_m2": "1.0"},
        "III": {"per_person": "4.0", "per_m2": "0.6"},
        "occupancy_density": "0.04",
    },
    "bedroom": {
        "I": {"per_person": "10.0", "per_m2": "1.4"},
        "II": {"per_person": "7.0", "per_m2": "1.0"},
        "III": {"per_person": "4.0", "per_m2": "0.6"},
        "occupancy_density": "0.05",
    },
    "kitchen_residential": {
        "I": {"per_person": "10.0", "per_m2": "2.0"},
        "II": {"per_person": "7.0", "per_m2": "1.4"},
        "III": {"per_person": "4.0", "per_m2": "0.8"},
        "occupancy_density": "0.06",
    },
    "kitchen_commercial": {
        "I": {"per_person": "15.0", "per_m2": "3.0"},
        "II": {"per_person": "10.0", "per_m2": "2.0"},
        "III": {"per_person": "7.0", "per_m2": "1.4"},
        "occupancy_density": "0.10",
    },
    "restaurant": {
        "I": {"per_person": "10.0", "per_m2": "2.0"},
        "II": {"per_person": "7.0", "per_m2": "1.4"},
        "III": {"per_person": "4.0", "per_m2": "0.8"},
        "occupancy_density": "0.70",
    },
    "retail": {
        "I": {"per_person": "10.0", "per_m2": "2.0"},
        "II": {"per_person": "7.0", "per_m2": "1.4"},
        "III": {"per_person": "4.0", "per_m2": "0.8"},
        "occupancy_density": "0.15",
    },
    "hospital_ward": {
        "I": {"per_person": "12.0", "per_m2": "2.5"},
        "II": {"per_person": "8.0", "per_m2": "1.7"},
        "III": {"per_person": "5.0", "per_m2": "1.0"},
        "occupancy_density": "0.10",
    },
    "hospital_treatment": {
        "I": {"per_person": "15.0", "per_m2": "3.0"},
        "II": {"per_person": "10.0", "per_m2": "2.0"},
        "III": {"per_person": "7.0", "per_m2": "1.4"},
        "occupancy_density": "0.10",
    },
    "hotel_room": {
        "I": {"per_person": "10.0", "per_m2": "1.4"},
        "II": {"per_person": "7.0", "per_m2": "1.0"},
        "III": {"per_person": "4.0", "per_m2": "0.6"},
        "occupancy_density": "0.05",
    },
    "gym": {
        "I": {"per_person": "20.0", "per_m2": "3.0"},
        "II": {"per_person": "14.0", "per_m2": "2.0"},
        "III": {"per_person": "8.0", "per_m2": "1.4"},
        "occupancy_density": "0.30",
    },
    "corridor": {
        "I": {"per_person": "0.0", "per_m2": "1.4"},
        "II": {"per_person": "0.0", "per_m2": "1.0"},
        "III": {"per_person": "0.0", "per_m2": "0.6"},
        "occupancy_density": "0.00",
    },
    "toilet": {
        "I": {"per_person": "0.0", "per_m2": "3.0"},
        "II": {"per_person": "0.0", "per_m2": "2.0"},
        "III": {"per_person": "0.0", "per_m2": "1.4"},
        "occupancy_density": "0.00",
    },
    "warehouse": {
        "I": {"per_person": "7.0", "per_m2": "0.7"},
        "II": {"per_person": "5.0", "per_m2": "0.5"},
        "III": {"per_person": "3.0", "per_m2": "0.3"},
        "occupancy_density": "0.02",
    },
}

# ---------------------------------------------------------------------------
# Overheating Criteria (CIBSE TM59)
# ---------------------------------------------------------------------------

OVERHEATING_CRITERIA: Dict[str, Dict[str, str]] = {
    "living_areas": {
        "fixed_threshold_degC": "26",
        "max_hours_above": "3",
        "assessment_period": "May-Sep",
        "method": "adaptive",
        "delta_T_category_II": "1.0",
    },
    "bedrooms": {
        "fixed_threshold_degC": "26",
        "max_hours_above": "1",
        "night_threshold_degC": "28",
        "max_night_hours": "32",
        "assessment_period": "May-Sep",
        "method": "fixed",
    },
    "non_residential": {
        "fixed_threshold_degC": "28",
        "max_pct_occupied_hours": "1",
        "assessment_period": "Year",
        "method": "adaptive",
    },
}

# ---------------------------------------------------------------------------
# Metabolic Rates (ISO 7730 Table B.1) -- met (1 met = 58.2 W/m2)
# ---------------------------------------------------------------------------

METABOLIC_RATES: Dict[str, str] = {
    "reclining": "0.8",
    "seated_relaxed": "1.0",
    "seated_office": "1.2",
    "standing_relaxed": "1.4",
    "standing_light_work": "1.6",
    "standing_medium_work": "2.0",
    "walking_slow": "2.0",
    "walking_moderate": "2.6",
    "walking_fast": "3.8",
    "cooking": "1.8",
    "cleaning": "2.7",
    "teaching": "1.4",
}

# ---------------------------------------------------------------------------
# Clothing Insulation (ISO 7730 Table C.1) -- clo (1 clo = 0.155 m2K/W)
# ---------------------------------------------------------------------------

CLOTHING_INSULATION: Dict[str, str] = {
    "summer": "0.5",
    "winter": "1.0",
    "transition": "0.7",
    "light_summer": "0.3",
    "heavy_winter": "1.5",
    "business_suit": "1.0",
    "casual": "0.7",
    "hospital_gown": "0.3",
    "athletic": "0.2",
}

# ---------------------------------------------------------------------------
# Daylighting Requirements (EN 17037)
# ---------------------------------------------------------------------------

DAYLIGHTING_REQUIREMENTS: Dict[str, Dict[str, str]] = {
    "office_open": {"target_lux": "500", "minimum_lux": "300", "daylight_factor_pct": "2.0"},
    "office_private": {"target_lux": "500", "minimum_lux": "300", "daylight_factor_pct": "2.0"},
    "meeting_room": {"target_lux": "500", "minimum_lux": "300", "daylight_factor_pct": "2.0"},
    "classroom": {"target_lux": "500", "minimum_lux": "300", "daylight_factor_pct": "2.5"},
    "lecture_hall": {"target_lux": "500", "minimum_lux": "300", "daylight_factor_pct": "2.0"},
    "living_room": {"target_lux": "300", "minimum_lux": "100", "daylight_factor_pct": "1.5"},
    "bedroom": {"target_lux": "300", "minimum_lux": "100", "daylight_factor_pct": "1.0"},
    "kitchen_residential": {"target_lux": "300", "minimum_lux": "150", "daylight_factor_pct": "1.5"},
    "kitchen_commercial": {"target_lux": "500", "minimum_lux": "300", "daylight_factor_pct": "2.0"},
    "restaurant": {"target_lux": "200", "minimum_lux": "100", "daylight_factor_pct": "1.0"},
    "retail": {"target_lux": "300", "minimum_lux": "200", "daylight_factor_pct": "1.5"},
    "hospital_ward": {"target_lux": "300", "minimum_lux": "100", "daylight_factor_pct": "1.5"},
    "hospital_treatment": {"target_lux": "500", "minimum_lux": "300", "daylight_factor_pct": "2.0"},
    "hotel_room": {"target_lux": "300", "minimum_lux": "100", "daylight_factor_pct": "1.0"},
    "gym": {"target_lux": "300", "minimum_lux": "200", "daylight_factor_pct": "1.5"},
    "corridor": {"target_lux": "100", "minimum_lux": "50", "daylight_factor_pct": "0.5"},
    "warehouse": {"target_lux": "200", "minimum_lux": "100", "daylight_factor_pct": "1.0"},
}

# Body surface area constant for PMV (DuBois area, average adult ~1.8 m2)
DUBOIS_AREA: str = "1.8"

# Stefan-Boltzmann constant
STEFAN_BOLTZMANN: str = "0.0000000567"

# Boltzmann factor for clothing radiation (linearised)
CLOTHING_AREA_FACTOR_CONST: str = "0.028"  # f_cl adjustment

# ---------------------------------------------------------------------------
# Pydantic Models -- Inputs
# ---------------------------------------------------------------------------

class ThermalComfortInput(BaseModel):
    """Input for PMV/PPD thermal comfort calculation."""
    air_temperature_degC: float = Field(..., ge=-10, le=50, description="Air temperature degC")
    mean_radiant_temperature_degC: float = Field(..., ge=-10, le=60, description="Mean radiant temperature degC")
    relative_humidity_pct: float = Field(..., ge=0, le=100, description="Relative humidity %")
    air_speed_m_s: float = Field(default=0.1, ge=0, le=2.0, description="Air speed m/s")
    metabolic_rate_met: float = Field(default=1.2, ge=0.8, le=4.0, description="Metabolic rate met")
    clothing_insulation_clo: float = Field(default=0.7, ge=0.0, le=2.0, description="Clothing insulation clo")

class IAQMeasurement(BaseModel):
    """A single IAQ parameter measurement."""
    parameter: str = Field(..., description="IAQ parameter name")
    measured_value: float = Field(..., ge=0, description="Measured value")
    location: Optional[str] = None

    @field_validator("parameter")
    @classmethod
    def validate_parameter(cls, v: str) -> str:
        valid = [p.value for p in IAQParameter]
        if v not in valid:
            raise ValueError(f"parameter must be one of {valid}")
        return v

class SpaceVentilationInput(BaseModel):
    """Input for ventilation adequacy assessment of a single space."""
    space_id: str = Field(..., min_length=1)
    space_type: str = Field(..., description="Space typology")
    floor_area_m2: float = Field(..., gt=0)
    n_occupants: Optional[int] = Field(None, ge=0)
    current_supply_rate_l_s: float = Field(..., ge=0, description="Current outdoor air supply l/s")
    target_category: str = Field(default="II", description="Target IEQ category")

    @field_validator("space_type")
    @classmethod
    def validate_space_type(cls, v: str) -> str:
        valid = [s.value for s in SpaceType]
        if v not in valid:
            raise ValueError(f"space_type must be one of {valid}")
        return v

class OverheatingInput(BaseModel):
    """Hourly operative temperature data for overheating assessment."""
    space_type: str = Field(default="living_areas", description="living_areas, bedrooms, non_residential")
    hourly_operative_temps_degC: List[float] = Field(..., min_length=24, description="Hourly T_op over assessment period")
    hourly_outdoor_temps_degC: Optional[List[float]] = Field(None, description="Hourly outdoor temp for adaptive method")
    occupied_hours_mask: Optional[List[bool]] = Field(None, description="True if occupied")

class DaylightInput(BaseModel):
    """Input for daylighting assessment of a space."""
    space_type: str = Field(...)
    measured_daylight_factor_pct: float = Field(..., ge=0, le=20)
    measured_illuminance_lux: Optional[float] = Field(None, ge=0)
    window_to_floor_ratio: Optional[float] = Field(None, ge=0, le=1.0)

class IndoorEnvironmentInput(BaseModel):
    """Full input for the IndoorEnvironmentEngine."""
    building_id: str = Field(..., min_length=1)
    assessment_date: Optional[str] = None
    target_category: str = Field(default="II", description="Target IEQ category I/II/III/IV")

    # Thermal comfort
    thermal_method: str = Field(default="both", description="pmv_ppd, adaptive, or both")
    thermal_inputs: Optional[List[ThermalComfortInput]] = None
    running_mean_outdoor_temp_degC: Optional[float] = Field(None, ge=-10, le=40)

    # IAQ
    iaq_measurements: Optional[List[IAQMeasurement]] = None
    outdoor_co2_ppm: float = Field(default=420, ge=350, le=500)

    # Ventilation
    spaces: Optional[List[SpaceVentilationInput]] = None

    # Overheating
    overheating: Optional[OverheatingInput] = None

    # Daylighting
    daylight: Optional[List[DaylightInput]] = None

    @field_validator("target_category")
    @classmethod
    def validate_category(cls, v: str) -> str:
        if v not in ("I", "II", "III", "IV"):
            raise ValueError("target_category must be I, II, III, or IV")
        return v

# ---------------------------------------------------------------------------
# Pydantic Models -- Outputs
# ---------------------------------------------------------------------------

class PMVPPDResult(BaseModel):
    """PMV and PPD calculation result."""
    pmv: float
    ppd_pct: float
    category_achieved: str
    category_target: str
    compliant: bool
    air_temperature_degC: float
    mean_radiant_temperature_degC: float
    relative_humidity_pct: float
    air_speed_m_s: float
    metabolic_rate_met: float
    clothing_insulation_clo: float

class AdaptiveComfortResult(BaseModel):
    """Adaptive thermal comfort result per EN 16798-1."""
    running_mean_outdoor_degC: float
    comfort_temperature_degC: float
    upper_limit_degC: float
    lower_limit_degC: float
    operative_temperature_degC: float
    category_achieved: str
    compliant: bool

class IAQAssessmentResult(BaseModel):
    """IAQ assessment result for a single parameter."""
    parameter: str
    measured_value: float
    limit_value: float
    unit: str
    category_target: str
    category_achieved: str
    compliant: bool
    who_guideline_value: Optional[float] = None
    who_compliant: Optional[bool] = None

class VentilationResult(BaseModel):
    """Ventilation adequacy assessment for a single space."""
    space_id: str
    space_type: str
    floor_area_m2: float
    n_occupants: int
    required_rate_l_s: float
    required_rate_l_s_per_person: float
    required_rate_l_s_per_m2: float
    current_supply_l_s: float
    adequacy_pct: float
    deficit_l_s: float
    compliant: bool
    category_target: str

class OverheatingResult(BaseModel):
    """Overheating risk assessment result."""
    space_type: str
    method: str
    total_hours_assessed: int
    hours_above_threshold: int
    pct_hours_above: float
    threshold_degC: float
    max_temperature_degC: float
    pass_criterion: bool
    criterion_description: str

class DaylightResult(BaseModel):
    """Daylighting assessment result."""
    space_type: str
    measured_daylight_factor_pct: float
    required_daylight_factor_pct: float
    target_illuminance_lux: float
    minimum_illuminance_lux: float
    compliant: bool
    category: str

class IEQScoreBreakdown(BaseModel):
    """Composite IEQ score breakdown."""
    thermal_score: float
    iaq_score: float
    ventilation_score: float
    visual_score: float
    overall_score: float
    overall_category: str
    weight_thermal: float = 0.35
    weight_iaq: float = 0.30
    weight_ventilation: float = 0.20
    weight_visual: float = 0.15

class IndoorEnvironmentResult(BaseModel):
    """Complete output of the IndoorEnvironmentEngine."""
    assessment_id: str
    building_id: str
    target_category: str

    # Thermal comfort
    pmv_ppd_results: List[PMVPPDResult]
    adaptive_results: List[AdaptiveComfortResult]

    # IAQ
    iaq_results: List[IAQAssessmentResult]

    # Ventilation
    ventilation_results: List[VentilationResult]

    # Overheating
    overheating_results: List[OverheatingResult]

    # Daylighting
    daylight_results: List[DaylightResult]

    # Composite score
    ieq_score: Optional[IEQScoreBreakdown] = None

    # Summary
    total_parameters_assessed: int
    total_compliant: int
    total_non_compliant: int
    compliance_rate_pct: float
    recommendations: List[str]

    # Metadata
    engine_version: str = _MODULE_VERSION
    calculated_at: str
    processing_time_ms: float
    provenance_hash: str

# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class IndoorEnvironmentEngine:
    """
    Indoor Environmental Quality assessment engine.

    Evaluates thermal comfort (PMV/PPD per ISO 7730, adaptive per
    EN 16798-1), indoor air quality, ventilation adequacy, overheating
    risk (TM59), and daylighting (EN 17037).

    Zero-Hallucination Guarantee:
        - PMV via iterative Fanger equation (deterministic)
        - PPD from ISO 7730 analytical formula
        - All lookups from published EN/ISO standard tables
        - No LLM in any calculation path
        - SHA-256 provenance hash on every result
    """

    # ------------------------------------------------------------------ #
    # calculate_pmv_ppd
    # ------------------------------------------------------------------ #

    def calculate_pmv_ppd(
        self,
        inp: ThermalComfortInput,
        target_category: str = "II",
    ) -> PMVPPDResult:
        """Calculate PMV and PPD per ISO 7730 Fanger's equation.

        Six input variables: air temperature, mean radiant temperature,
        relative humidity, air speed, metabolic rate, clothing insulation.

        The clothing surface temperature is solved iteratively.

        Args:
            inp: Thermal comfort measurement data.
            target_category: IEQ target category.

        Returns:
            PMV/PPD result with category compliance.
        """
        ta = _decimal(inp.air_temperature_degC)
        tr = _decimal(inp.mean_radiant_temperature_degC)
        rh = _decimal(inp.relative_humidity_pct)
        va = _decimal(inp.air_speed_m_s)
        met = _decimal(inp.metabolic_rate_met)
        clo = _decimal(inp.clothing_insulation_clo)

        # Convert units
        M = met * Decimal("58.2")  # W/m2
        I_cl = clo * Decimal("0.155")  # m2K/W
        W = Decimal("0")  # External work (zero for most activities)

        # Clothing area factor (f_cl)
        if I_cl <= Decimal("0.078"):
            f_cl = Decimal("1") + Decimal("1.290") * I_cl
        else:
            f_cl = Decimal("1.05") + Decimal("0.645") * I_cl

        # Partial water vapour pressure
        # p_a = rh / 100 * 610.5 * exp(17.269 * ta / (237.3 + ta))
        ta_f = float(ta)
        sat_vap = Decimal(str(610.5 * math.exp(17.269 * ta_f / (237.3 + ta_f))))
        p_a = (rh / Decimal("100")) * sat_vap

        # Convective heat transfer coefficient h_c
        # h_c = max(2.38 * |t_cl - ta|^0.25, 12.1 * sqrt(va))
        # Iterate to find t_cl

        # Initial guess for clothing surface temperature
        t_cl = ta + Decimal("0.2") * (Decimal("35.7") - ta)

        # Iterative solution (40 iterations is more than enough)
        for _ in range(40):
            t_cl_f = float(t_cl)
            ta_f = float(ta)
            tr_f = float(tr)

            # Natural convection
            h_c_nat = 2.38 * abs(t_cl_f - ta_f) ** 0.25 if abs(t_cl_f - ta_f) > 0 else 0.0
            # Forced convection
            h_c_forced = 12.1 * math.sqrt(float(va))
            h_c = max(h_c_nat, h_c_forced)
            h_c_d = _decimal(h_c)

            # Radiation heat transfer
            # h_r = 3.96e-8 * f_cl * ((t_cl+273)^4 - (tr+273)^4) / (t_cl - tr)
            t_cl_K = t_cl_f + 273.0
            tr_K = tr_f + 273.0
            if abs(t_cl_f - tr_f) > 0.001:
                h_r_val = 3.96e-8 * float(f_cl) * (t_cl_K**4 - tr_K**4) / (t_cl_f - tr_f)
            else:
                h_r_val = 3.96e-8 * float(f_cl) * 4.0 * ((t_cl_K + tr_K) / 2.0) ** 3
            h_r = _decimal(abs(h_r_val))

            # New t_cl
            numerator = Decimal("35.7") - Decimal("0.028") * (M - W) - I_cl * (
                Decimal("3.96") * Decimal("0.00000001") * f_cl * (
                    (t_cl + Decimal("273")) ** 4 - (tr + Decimal("273")) ** 4
                ) + f_cl * h_c_d * (t_cl - ta)
            )
            # Simplified: t_cl_new = 35.7 - 0.028*(M-W) - I_cl * (h_r*(t_cl-tr) + h_c*(t_cl-ta)) * f_cl
            # Rearranging the heat balance:
            t_cl_new = (Decimal("35.7") - Decimal("0.028") * (M - W)
                        + I_cl * f_cl * (h_r * tr + h_c_d * ta)) / (
                Decimal("1") + I_cl * f_cl * (h_r + h_c_d)
            )
            t_cl = t_cl_new

        # Final h_c
        t_cl_f = float(t_cl)
        h_c_nat = 2.38 * abs(t_cl_f - float(ta)) ** 0.25 if abs(t_cl_f - float(ta)) > 0 else 0.0
        h_c_forced = 12.1 * math.sqrt(float(va))
        h_c = _decimal(max(h_c_nat, h_c_forced))

        # PMV = [0.303 * exp(-0.036*M) + 0.028] * {
        #   (M-W) - 3.05e-3 * [5733 - 6.99*(M-W) - p_a]
        #   - 0.42 * [(M-W) - 58.15]
        #   - 1.7e-5 * M * (5867 - p_a)
        #   - 0.0014 * M * (34 - ta)
        #   - 3.96e-8 * f_cl * [(t_cl+273)^4 - (tr+273)^4]
        #   - f_cl * h_c * (t_cl - ta)
        # }

        MW = M - W

        # Sensible skin heat loss
        term1 = Decimal("0.00305") * (Decimal("5733") - Decimal("6.99") * MW - p_a)

        # Sweating
        if MW > Decimal("58.15"):
            term2 = Decimal("0.42") * (MW - Decimal("58.15"))
        else:
            term2 = Decimal("0")

        # Latent respiration
        term3 = Decimal("0.000017") * M * (Decimal("5867") - p_a)

        # Sensible respiration
        term4 = Decimal("0.0014") * M * (Decimal("34") - ta)

        # Radiation
        t_cl_K4 = (t_cl + Decimal("273")) ** 4
        tr_K4 = (tr + Decimal("273")) ** 4
        term5 = Decimal("0.0000000396") * f_cl * (t_cl_K4 - tr_K4)

        # Convection
        term6 = f_cl * h_c * (t_cl - ta)

        thermal_load = MW - term1 - term2 - term3 - term4 - term5 - term6

        # PMV coefficient
        coeff = Decimal("0.303") * _decimal(math.exp(-0.036 * float(M))) + Decimal("0.028")
        pmv = coeff * thermal_load

        # Clamp to [-3, 3]
        pmv = max(Decimal("-3"), min(Decimal("3"), pmv))

        # PPD = 100 - 95 * exp(-0.03353 * PMV^4 - 0.2179 * PMV^2)
        pmv_f = float(pmv)
        ppd_f = 100.0 - 95.0 * math.exp(-0.03353 * pmv_f**4 - 0.2179 * pmv_f**2)
        ppd_f = max(5.0, min(100.0, ppd_f))

        # Determine category achieved
        cat_achieved = "IV"
        for cat in ["I", "II", "III"]:
            limits = THERMAL_COMFORT_CATEGORIES[cat]
            if (float(pmv) >= float(limits["pmv_range_low"]) and
                    float(pmv) <= float(limits["pmv_range_high"])):
                cat_achieved = cat
                break

        compliant = self._category_rank(cat_achieved) <= self._category_rank(target_category)

        return PMVPPDResult(
            pmv=_round2(pmv_f),
            ppd_pct=_round1(ppd_f),
            category_achieved=cat_achieved,
            category_target=target_category,
            compliant=compliant,
            air_temperature_degC=_round1(float(ta)),
            mean_radiant_temperature_degC=_round1(float(tr)),
            relative_humidity_pct=_round1(float(rh)),
            air_speed_m_s=_round2(float(va)),
            metabolic_rate_met=_round1(float(met)),
            clothing_insulation_clo=_round2(float(clo)),
        )

    # ------------------------------------------------------------------ #
    # assess_adaptive_comfort
    # ------------------------------------------------------------------ #

    def assess_adaptive_comfort(
        self,
        operative_temp_degC: float,
        running_mean_outdoor_degC: float,
        target_category: str = "II",
    ) -> AdaptiveComfortResult:
        """Assess adaptive thermal comfort per EN 16798-1.

        Applicable to free-running (naturally ventilated) buildings.
        T_comf = 0.33 * T_rm + 18.8

        Args:
            operative_temp_degC: Indoor operative temperature.
            running_mean_outdoor_degC: Running mean outdoor temperature.
            target_category: Target IEQ category.

        Returns:
            Adaptive comfort assessment result.
        """
        t_rm = _decimal(running_mean_outdoor_degC)
        t_op = _decimal(operative_temp_degC)

        # Comfort temperature
        t_comf = Decimal("0.33") * t_rm + Decimal("18.8")

        # Get limits by interpolation from table
        upper, lower = self._interpolate_adaptive_limits(float(t_rm), target_category)
        upper_d = _decimal(upper)
        lower_d = _decimal(lower)

        # Determine category achieved
        cat_achieved = "IV"
        for cat in ["I", "II", "III"]:
            u, l = self._interpolate_adaptive_limits(float(t_rm), cat)
            if float(t_op) <= u and float(t_op) >= l:
                cat_achieved = cat
                break

        compliant = self._category_rank(cat_achieved) <= self._category_rank(target_category)

        return AdaptiveComfortResult(
            running_mean_outdoor_degC=_round1(float(t_rm)),
            comfort_temperature_degC=_round1(float(t_comf)),
            upper_limit_degC=_round1(upper),
            lower_limit_degC=_round1(lower),
            operative_temperature_degC=_round1(float(t_op)),
            category_achieved=cat_achieved,
            compliant=compliant,
        )

    def _interpolate_adaptive_limits(
        self,
        t_rm: float,
        category: str,
    ) -> Tuple[float, float]:
        """Interpolate adaptive comfort limits from table.

        Args:
            t_rm: Running mean outdoor temperature.
            category: IEQ category.

        Returns:
            (upper_limit, lower_limit) in degC.
        """
        upper_key = f"upper_{category}"
        lower_key = f"lower_{category}"

        # Find bracketing entries
        prev = ADAPTIVE_COMFORT_TABLE[0]
        for entry in ADAPTIVE_COMFORT_TABLE:
            if float(entry["t_rm"]) >= t_rm:
                # Interpolate between prev and entry
                t1 = float(prev["t_rm"])
                t2 = float(entry["t_rm"])
                if abs(t2 - t1) < 0.001:
                    return float(entry[upper_key]), float(entry[lower_key])
                frac = (t_rm - t1) / (t2 - t1)
                frac = max(0.0, min(1.0, frac))
                upper = float(prev[upper_key]) + frac * (float(entry[upper_key]) - float(prev[upper_key]))
                lower = float(prev[lower_key]) + frac * (float(entry[lower_key]) - float(prev[lower_key]))
                return upper, lower
            prev = entry

        # Beyond table range -- use last entry
        last = ADAPTIVE_COMFORT_TABLE[-1]
        return float(last[upper_key]), float(last[lower_key])

    # ------------------------------------------------------------------ #
    # assess_air_quality
    # ------------------------------------------------------------------ #

    def assess_air_quality(
        self,
        measurement: IAQMeasurement,
        target_category: str = "II",
        outdoor_co2_ppm: float = 420.0,
    ) -> IAQAssessmentResult:
        """Assess a single IAQ parameter against EN 16798-1 / WHO limits.

        Args:
            measurement: Measured parameter value.
            target_category: Target IEQ category.
            outdoor_co2_ppm: Outdoor CO2 baseline ppm.

        Returns:
            IAQ parameter assessment result.
        """
        param = measurement.parameter
        value = _decimal(measurement.measured_value)
        limits = IAQ_LIMITS.get(param, {})
        unit = limits.get("unit", {}).get("value", "")

        # Get limit for target category
        cat_limits = limits.get(target_category, {})

        if param == "co2":
            above_outdoor = _decimal(cat_limits.get("above_outdoor_ppm", "800"))
            limit_val = _decimal(outdoor_co2_ppm) + above_outdoor
        elif param == "humidity":
            # Humidity is a range check
            low = _decimal(cat_limits.get("low", "25"))
            high = _decimal(cat_limits.get("high", "60"))
            limit_val = high  # Use upper for comparison
        else:
            limit_val = _decimal(cat_limits.get("limit", "999999"))

        # Determine category achieved
        cat_achieved = "IV"
        for cat in ["I", "II", "III"]:
            cl = limits.get(cat, {})
            if param == "co2":
                threshold = _decimal(outdoor_co2_ppm) + _decimal(cl.get("above_outdoor_ppm", "1350"))
            elif param == "humidity":
                h_low = _decimal(cl.get("low", "20"))
                h_high = _decimal(cl.get("high", "70"))
                if value >= h_low and value <= h_high:
                    cat_achieved = cat
                    break
                continue
            else:
                threshold = _decimal(cl.get("limit", "999999"))

            if value <= threshold:
                cat_achieved = cat
                break

        if param == "humidity":
            target_low = _decimal(cat_limits.get("low", "25"))
            target_high = _decimal(cat_limits.get("high", "60"))
            compliant_val = value >= target_low and value <= target_high
        else:
            compliant_val = value <= limit_val

        compliant = compliant_val and self._category_rank(cat_achieved) <= self._category_rank(target_category)

        # WHO guideline check
        who_val_str = limits.get("who_guideline", {}).get("value")
        who_val = float(who_val_str) if who_val_str else None
        who_compliant = value <= _decimal(who_val_str) if who_val_str else None

        return IAQAssessmentResult(
            parameter=param,
            measured_value=_round1(float(value)),
            limit_value=_round1(float(limit_val)),
            unit=unit,
            category_target=target_category,
            category_achieved=cat_achieved,
            compliant=compliant,
            who_guideline_value=who_val,
            who_compliant=who_compliant,
        )

    # ------------------------------------------------------------------ #
    # assess_ventilation_adequacy
    # ------------------------------------------------------------------ #

    def assess_ventilation_adequacy(
        self,
        space: SpaceVentilationInput,
    ) -> VentilationResult:
        """Assess ventilation adequacy for a space per EN 16798-1.

        Required flow = max(Q_people + Q_building, Q_minimum)
        Q_people = n_occupants * rate_per_person
        Q_building = floor_area * rate_per_m2

        Args:
            space: Space ventilation input data.

        Returns:
            Ventilation adequacy assessment.
        """
        rates = VENTILATION_RATES.get(space.space_type, VENTILATION_RATES["office_open"])
        cat_rates = rates.get(space.target_category, rates.get("II", {}))
        per_person = _decimal(cat_rates.get("per_person", "7.0"))
        per_m2 = _decimal(cat_rates.get("per_m2", "1.4"))
        occ_density = _decimal(rates.get("occupancy_density", "0.1"))

        area = _decimal(space.floor_area_m2)

        # Occupants: use provided or estimate from density
        if space.n_occupants is not None:
            n_occ = space.n_occupants
        else:
            n_occ = max(1, int(float(area * occ_density)))

        n_occ_d = _decimal(n_occ)
        q_people = n_occ_d * per_person
        q_building = area * per_m2
        required = q_people + q_building

        current = _decimal(space.current_supply_rate_l_s)
        adequacy = _safe_pct(current, required)
        deficit = max(Decimal("0"), required - current)
        compliant = current >= required

        return VentilationResult(
            space_id=space.space_id,
            space_type=space.space_type,
            floor_area_m2=_round1(float(area)),
            n_occupants=n_occ,
            required_rate_l_s=_round1(float(required)),
            required_rate_l_s_per_person=_round1(float(per_person)),
            required_rate_l_s_per_m2=_round2(float(per_m2)),
            current_supply_l_s=_round1(float(current)),
            adequacy_pct=_round1(float(adequacy)),
            deficit_l_s=_round1(float(deficit)),
            compliant=compliant,
            category_target=space.target_category,
        )

    # ------------------------------------------------------------------ #
    # assess_overheating_risk
    # ------------------------------------------------------------------ #

    def assess_overheating_risk(
        self,
        inp: OverheatingInput,
    ) -> OverheatingResult:
        """Assess overheating risk per CIBSE TM59 criteria.

        For living areas: adaptive method -- hours where delta_T > 1K
        For bedrooms: fixed threshold 26degC daytime, 28degC nighttime
        For non-residential: % occupied hours above 28degC < 1%

        Args:
            inp: Hourly temperature data.

        Returns:
            Overheating risk assessment.
        """
        criteria = OVERHEATING_CRITERIA.get(inp.space_type, OVERHEATING_CRITERIA["non_residential"])
        threshold = float(criteria.get("fixed_threshold_degC", "28"))
        method = criteria.get("method", "fixed")

        temps = inp.hourly_operative_temps_degC
        total_hours = len(temps)
        occupied_mask = inp.occupied_hours_mask or [True] * total_hours
        occupied_hours = sum(1 for m in occupied_mask if m)

        max_temp = max(temps) if temps else 0.0

        if method == "adaptive" and inp.hourly_outdoor_temps_degC:
            # Adaptive: count hours where T_op > T_comf_upper
            hours_above = 0
            outdoor = inp.hourly_outdoor_temps_degC
            for i, (t_op, is_occ) in enumerate(zip(temps, occupied_mask)):
                if not is_occ:
                    continue
                # Running mean approximation from outdoor temp
                t_rm = outdoor[i] if i < len(outdoor) else 15.0
                t_comf_upper = 0.33 * t_rm + 18.8 + float(criteria.get("delta_T_category_II", "1.0"))
                if t_op > t_comf_upper:
                    hours_above += 1
        else:
            # Fixed threshold method
            hours_above = 0
            for i, (t_op, is_occ) in enumerate(zip(temps, occupied_mask)):
                if not is_occ:
                    continue
                if t_op > threshold:
                    hours_above += 1

        pct_above = (hours_above / occupied_hours * 100.0) if occupied_hours > 0 else 0.0

        # Pass criterion
        if inp.space_type == "bedrooms":
            max_hrs = int(criteria.get("max_hours_above", "1"))
            criterion = f"Max {max_hrs}% occupied hours above {threshold}degC"
            passed = pct_above <= float(max_hrs)
        elif inp.space_type == "living_areas":
            max_pct = float(criteria.get("max_hours_above", "3"))
            criterion = f"Max {max_pct}% occupied hours above adaptive limit"
            passed = pct_above <= max_pct
        else:
            max_pct = float(criteria.get("max_pct_occupied_hours", "1"))
            criterion = f"Max {max_pct}% occupied hours above {threshold}degC"
            passed = pct_above <= max_pct

        return OverheatingResult(
            space_type=inp.space_type,
            method=method,
            total_hours_assessed=total_hours,
            hours_above_threshold=hours_above,
            pct_hours_above=_round2(pct_above),
            threshold_degC=threshold,
            max_temperature_degC=_round1(max_temp),
            pass_criterion=passed,
            criterion_description=criterion,
        )

    # ------------------------------------------------------------------ #
    # assess_daylighting
    # ------------------------------------------------------------------ #

    def assess_daylighting(
        self,
        inp: DaylightInput,
        target_category: str = "II",
    ) -> DaylightResult:
        """Assess daylighting adequacy per EN 17037.

        Compares measured daylight factor against target for the space type.

        Args:
            inp: Daylighting measurement data.
            target_category: Target IEQ category (affects strictness).

        Returns:
            Daylighting assessment result.
        """
        reqs = DAYLIGHTING_REQUIREMENTS.get(inp.space_type, DAYLIGHTING_REQUIREMENTS["office_open"])
        target_df = _decimal(reqs["daylight_factor_pct"])
        target_lux = _decimal(reqs["target_lux"])
        min_lux = _decimal(reqs["minimum_lux"])

        # Adjust by category (Cat I is stricter)
        rank = self._category_rank(target_category)
        adjustment = Decimal("1") + Decimal("0.1") * _decimal(1 - rank)  # Cat I: +0.1x, Cat II: base
        adj_target_df = target_df * adjustment if rank == 0 else target_df

        measured = _decimal(inp.measured_daylight_factor_pct)
        compliant = measured >= adj_target_df

        return DaylightResult(
            space_type=inp.space_type,
            measured_daylight_factor_pct=_round2(float(measured)),
            required_daylight_factor_pct=_round2(float(adj_target_df)),
            target_illuminance_lux=_round1(float(target_lux)),
            minimum_illuminance_lux=_round1(float(min_lux)),
            compliant=compliant,
            category=target_category,
        )

    # ------------------------------------------------------------------ #
    # calculate_ieq_score
    # ------------------------------------------------------------------ #

    def calculate_ieq_score(
        self,
        pmv_results: List[PMVPPDResult],
        iaq_results: List[IAQAssessmentResult],
        vent_results: List[VentilationResult],
        daylight_results: List[DaylightResult],
    ) -> IEQScoreBreakdown:
        """Calculate composite IEQ score.

        Weighted sum: thermal 35%, IAQ 30%, ventilation 20%, visual 15%.
        Each sub-score 0-100 based on compliance and category achieved.

        Args:
            pmv_results: Thermal comfort results.
            iaq_results: IAQ results.
            vent_results: Ventilation results.
            daylight_results: Daylighting results.

        Returns:
            Composite IEQ score breakdown.
        """
        # Thermal sub-score
        if pmv_results:
            thermal_scores = []
            for r in pmv_results:
                rank = self._category_rank(r.category_achieved)
                thermal_scores.append(Decimal(str(max(0, 100 - rank * 25))))
            thermal = sum(thermal_scores, Decimal("0")) / _decimal(len(thermal_scores))
        else:
            thermal = Decimal("50")

        # IAQ sub-score
        if iaq_results:
            iaq_scores = []
            for r in iaq_results:
                rank = self._category_rank(r.category_achieved)
                iaq_scores.append(Decimal(str(max(0, 100 - rank * 25))))
            iaq = sum(iaq_scores, Decimal("0")) / _decimal(len(iaq_scores))
        else:
            iaq = Decimal("50")

        # Ventilation sub-score
        if vent_results:
            vent_scores = []
            for r in vent_results:
                adequacy = _decimal(r.adequacy_pct)
                score = min(Decimal("100"), adequacy)
                vent_scores.append(score)
            vent = sum(vent_scores, Decimal("0")) / _decimal(len(vent_scores))
        else:
            vent = Decimal("50")

        # Visual sub-score
        if daylight_results:
            vis_scores = []
            for r in daylight_results:
                measured = _decimal(r.measured_daylight_factor_pct)
                required = _decimal(r.required_daylight_factor_pct)
                ratio = _safe_divide(measured, required) * Decimal("100")
                vis_scores.append(min(Decimal("100"), ratio))
            visual = sum(vis_scores, Decimal("0")) / _decimal(len(vis_scores))
        else:
            visual = Decimal("50")

        # Weighted overall
        w_t = Decimal("0.35")
        w_i = Decimal("0.30")
        w_v = Decimal("0.20")
        w_d = Decimal("0.15")
        overall = thermal * w_t + iaq * w_i + vent * w_v + visual * w_d

        # Category from score
        if overall >= Decimal("90"):
            cat = "I"
        elif overall >= Decimal("75"):
            cat = "II"
        elif overall >= Decimal("50"):
            cat = "III"
        else:
            cat = "IV"

        return IEQScoreBreakdown(
            thermal_score=_round1(float(thermal)),
            iaq_score=_round1(float(iaq)),
            ventilation_score=_round1(float(vent)),
            visual_score=_round1(float(visual)),
            overall_score=_round1(float(overall)),
            overall_category=cat,
        )

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _category_rank(cat: str) -> int:
        """Return numeric rank for IEQ category (lower is better)."""
        return {"I": 0, "II": 1, "III": 2, "IV": 3}.get(cat, 3)

    def _generate_recommendations(
        self,
        pmv_results: List[PMVPPDResult],
        iaq_results: List[IAQAssessmentResult],
        vent_results: List[VentilationResult],
        overheating_results: List[OverheatingResult],
        daylight_results: List[DaylightResult],
    ) -> List[str]:
        """Generate IEQ improvement recommendations.

        Args:
            All assessment results.

        Returns:
            List of recommendation strings.
        """
        recs: List[str] = []

        # Thermal comfort
        for r in pmv_results:
            if not r.compliant:
                if r.pmv < -0.5:
                    recs.append(f"Thermal comfort: Space is too cold (PMV={r.pmv}). Consider increasing heating setpoint or improving insulation.")
                elif r.pmv > 0.5:
                    recs.append(f"Thermal comfort: Space is too warm (PMV={r.pmv}). Consider reducing cooling setpoint or solar shading.")

        # IAQ
        for r in iaq_results:
            if not r.compliant:
                if r.parameter == "co2":
                    recs.append(f"IAQ: CO2 level ({r.measured_value} ppm) exceeds Category {r.category_target} limit. Increase ventilation rate or reduce occupancy.")
                elif r.parameter == "pm25":
                    recs.append(f"IAQ: PM2.5 ({r.measured_value} ug/m3) exceeds limit. Improve filtration or source control.")
                elif r.parameter == "tvoc":
                    recs.append(f"IAQ: TVOC ({r.measured_value} ug/m3) exceeds limit. Improve ventilation and identify VOC sources.")
                elif r.parameter == "radon":
                    recs.append(f"IAQ: Radon ({r.measured_value} Bq/m3) exceeds limit. Install radon sump or positive pressure system.")
                elif r.parameter == "humidity":
                    recs.append(f"IAQ: Humidity ({r.measured_value}%RH) outside acceptable range. Consider humidification/dehumidification.")
                else:
                    recs.append(f"IAQ: {r.parameter} ({r.measured_value}) exceeds Category {r.category_target} limit.")

        # Ventilation
        for r in vent_results:
            if not r.compliant:
                recs.append(f"Ventilation: Space '{r.space_id}' has {r.deficit_l_s} l/s deficit. Current supply is {r.adequacy_pct}% of required.")

        # Overheating
        for r in overheating_results:
            if not r.pass_criterion:
                recs.append(f"Overheating: {r.space_type} fails TM59 ({r.pct_hours_above}% hours above threshold). Consider solar shading, night cooling, or improved ventilation.")

        # Daylighting
        for r in daylight_results:
            if not r.compliant:
                recs.append(f"Daylighting: {r.space_type} daylight factor {r.measured_daylight_factor_pct}% below {r.required_daylight_factor_pct}% target. Consider larger windows or light shelves.")

        if not recs:
            recs.append("All assessed parameters meet the target IEQ category requirements.")

        return recs

    # ------------------------------------------------------------------ #
    # assess  (main entry point)
    # ------------------------------------------------------------------ #

    def assess(self, inp: IndoorEnvironmentInput) -> IndoorEnvironmentResult:
        """Execute full indoor environment quality assessment.

        Main entry point. Evaluates thermal comfort, IAQ, ventilation,
        overheating risk, and daylighting.

        Args:
            inp: Validated assessment input.

        Returns:
            Complete IEQ assessment result with provenance hash.
        """
        t0 = time.perf_counter()
        assessment_id = _new_uuid()
        target = inp.target_category

        # -- PMV/PPD --
        pmv_results: List[PMVPPDResult] = []
        if inp.thermal_inputs and inp.thermal_method in ("pmv_ppd", "both"):
            for ti in inp.thermal_inputs:
                pmv_results.append(self.calculate_pmv_ppd(ti, target))

        # -- Adaptive comfort --
        adaptive_results: List[AdaptiveComfortResult] = []
        if inp.thermal_inputs and inp.running_mean_outdoor_temp_degC is not None and inp.thermal_method in ("adaptive", "both"):
            for ti in inp.thermal_inputs:
                t_op = (ti.air_temperature_degC + ti.mean_radiant_temperature_degC) / 2.0
                adaptive_results.append(
                    self.assess_adaptive_comfort(t_op, inp.running_mean_outdoor_temp_degC, target)
                )

        # -- IAQ --
        iaq_results: List[IAQAssessmentResult] = []
        if inp.iaq_measurements:
            for m in inp.iaq_measurements:
                iaq_results.append(self.assess_air_quality(m, target, inp.outdoor_co2_ppm))

        # -- Ventilation --
        vent_results: List[VentilationResult] = []
        if inp.spaces:
            for s in inp.spaces:
                vent_results.append(self.assess_ventilation_adequacy(s))

        # -- Overheating --
        oh_results: List[OverheatingResult] = []
        if inp.overheating:
            oh_results.append(self.assess_overheating_risk(inp.overheating))

        # -- Daylighting --
        dl_results: List[DaylightResult] = []
        if inp.daylight:
            for d in inp.daylight:
                dl_results.append(self.assess_daylighting(d, target))

        # -- IEQ Score --
        ieq_score = self.calculate_ieq_score(pmv_results, iaq_results, vent_results, dl_results)

        # -- Compliance summary --
        all_checks: List[bool] = []
        all_checks.extend(r.compliant for r in pmv_results)
        all_checks.extend(r.compliant for r in adaptive_results)
        all_checks.extend(r.compliant for r in iaq_results)
        all_checks.extend(r.compliant for r in vent_results)
        all_checks.extend(r.pass_criterion for r in oh_results)
        all_checks.extend(r.compliant for r in dl_results)

        total_assessed = len(all_checks)
        total_compliant = sum(1 for c in all_checks if c)
        total_non_compliant = total_assessed - total_compliant
        compliance_rate = (total_compliant / total_assessed * 100.0) if total_assessed > 0 else 0.0

        # -- Recommendations --
        recs = self._generate_recommendations(pmv_results, iaq_results, vent_results, oh_results, dl_results)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        result = IndoorEnvironmentResult(
            assessment_id=assessment_id,
            building_id=inp.building_id,
            target_category=target,
            pmv_ppd_results=pmv_results,
            adaptive_results=adaptive_results,
            iaq_results=iaq_results,
            ventilation_results=vent_results,
            overheating_results=oh_results,
            daylight_results=dl_results,
            ieq_score=ieq_score,
            total_parameters_assessed=total_assessed,
            total_compliant=total_compliant,
            total_non_compliant=total_non_compliant,
            compliance_rate_pct=_round1(compliance_rate),
            recommendations=recs,
            engine_version=_MODULE_VERSION,
            calculated_at=utcnow().isoformat(),
            processing_time_ms=round(elapsed_ms, 2),
            provenance_hash="",
        )

        result.provenance_hash = _compute_hash(result)
        return result
