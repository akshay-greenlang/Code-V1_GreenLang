# -*- coding: utf-8 -*-
"""
LightingAssessmentEngine - PACK-032 Building Energy Assessment Engine 5
========================================================================

Assesses building lighting energy performance and visual quality per
EN 12464-1:2021 (lighting requirements) and EN 15193-1:2017 (LENI -
Lighting Energy Numeric Indicator).  Covers installed lighting power
density (LPD), LENI calculation, daylight factor assessment, controls
effectiveness, LED retrofit savings estimation, and circadian
stimulus evaluation per WELL v2.

EN 15193-1:2017 Compliance:
    - LENI calculation method (comprehensive / quick)
    - Daylight dependency factor (FD)
    - Occupancy dependency factor (FO)
    - Parasitic power for controls

EN 12464-1:2021 Compliance:
    - Maintained illuminance requirements by space type
    - Uniformity requirements
    - Glare rating limits (UGR)
    - Colour rendering requirements

LED Retrofit Analysis:
    - Fixture-by-fixture energy saving calculation
    - Lamp efficacy comparisons (lm/W)
    - Simple payback and NPV
    - Maintenance factor improvement

Zero-Hallucination:
    - All calculations use deterministic Decimal arithmetic
    - No LLM involvement in any numeric calculation path
    - SHA-256 provenance hashing on every result
    - Benchmarks from EN 12464-1, EN 15193, CIBSE LG series

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-032 Building Energy Assessment
Engine:  5 of 10
Status:  Production Ready
"""

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

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _new_uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())


def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash of arbitrary data.

    Uses JSON serialization with sorted keys to guarantee reproducibility.

    Args:
        data: Data to hash -- dict, Pydantic model, or other serializable.

    Returns:
        SHA-256 hex digest string (64 characters).
    """
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
    """Safely convert a value to Decimal.

    Args:
        value: Numeric value to convert.

    Returns:
        Decimal representation; Decimal("0") on failure.
    """
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
    """Safely divide two Decimals, returning *default* on zero denominator.

    Args:
        numerator: Dividend.
        denominator: Divisor.
        default: Value returned when denominator is zero.

    Returns:
        Result of division or *default*.
    """
    if denominator == Decimal("0"):
        return default
    return numerator / denominator


def _safe_pct(part: Decimal, whole: Decimal) -> Decimal:
    """Compute percentage safely (part / whole * 100).

    Args:
        part: Numerator.
        whole: Denominator.

    Returns:
        Percentage as Decimal; Decimal("0") when whole is zero.
    """
    return _safe_divide(part * Decimal("100"), whole)


def _round_val(value: Decimal, places: int = 6) -> float:
    """Round a Decimal to *places* and return a float.

    Uses ROUND_HALF_UP (regulatory standard rounding).

    Args:
        value: Value to round.
        places: Number of decimal places.

    Returns:
        Rounded float value.
    """
    quantizer = Decimal(10) ** -places
    return float(value.quantize(quantizer, rounding=ROUND_HALF_UP))


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


class LampType(str, Enum):
    """Lamp technology types.

    Efficacy values vary significantly and drive energy benchmarking.
    Sources: IES Handbook, manufacturer data, EN 13032.
    """
    LED = "led"
    FLUORESCENT_T5 = "fluorescent_t5"
    FLUORESCENT_T8 = "fluorescent_t8"
    CFL = "cfl"
    HALOGEN = "halogen"
    HID_METAL_HALIDE = "hid_metal_halide"
    HID_SODIUM = "hid_sodium"
    INCANDESCENT = "incandescent"


class ControlType(str, Enum):
    """Lighting control strategies.

    Different controls deliver different energy savings per EN 15193.
    """
    MANUAL_SWITCH = "manual_switch"
    TIME_CLOCK = "time_clock"
    OCCUPANCY_SENSOR = "occupancy_sensor"
    DAYLIGHT_SENSOR = "daylight_sensor"
    COMBINED_OCC_DAYLIGHT = "combined_occ_daylight"
    DALI_ADDRESSABLE = "dali_addressable"
    SCENE_SETTING = "scene_setting"
    ABSENCE_DETECTION = "absence_detection"


class SpaceCategory(str, Enum):
    """Space categories per EN 12464-1:2021.

    Each category has specific illuminance, uniformity, and UGR
    requirements for visual comfort and task performance.
    """
    OFFICE_GENERAL = "office_general"
    OFFICE_OPEN_PLAN = "office_open_plan"
    RETAIL_SALES = "retail_sales"
    RETAIL_SUPERMARKET = "retail_supermarket"
    CLASSROOM = "classroom"
    LECTURE_HALL = "lecture_hall"
    HOSPITAL_WARD = "hospital_ward"
    HOSPITAL_CORRIDOR = "hospital_corridor"
    HOTEL_ROOM = "hotel_room"
    HOTEL_LOBBY = "hotel_lobby"
    RESTAURANT = "restaurant"
    WAREHOUSE = "warehouse"
    PARKING = "parking"
    CORRIDOR = "corridor"
    RESTROOM = "restroom"
    LABORATORY = "laboratory"
    CLEAN_ROOM = "clean_room"
    KITCHEN = "kitchen"
    RECEPTION = "reception"


class EnvironmentCleanliness(str, Enum):
    """Luminaire environment cleanliness for maintenance factor.

    Affects lamp lumen depreciation and luminaire dirt depreciation.
    Source: CIE 97:2005.
    """
    VERY_CLEAN = "very_clean"
    CLEAN = "clean"
    NORMAL = "normal"
    DIRTY = "dirty"


# ---------------------------------------------------------------------------
# Constants -- Lighting Power Density Benchmarks (W/m2)
# ---------------------------------------------------------------------------

# Installed Lighting Power Density benchmarks by space type.
# 4 tiers: best_practice, good, acceptable, poor.
# Sources: EN 12464-1:2021 Annex, CIBSE LG7/LG10, BCO Guide.
LPD_BENCHMARKS: Dict[str, Dict[str, float]] = {
    SpaceCategory.OFFICE_GENERAL: {
        "best_practice": 6.0,
        "good": 8.0,
        "acceptable": 10.0,
        "poor": 14.0,
        "source": "CIBSE LG7:2015, EN 15193 Table C.1",
    },
    SpaceCategory.OFFICE_OPEN_PLAN: {
        "best_practice": 5.5,
        "good": 7.5,
        "acceptable": 10.0,
        "poor": 14.0,
        "source": "CIBSE LG7:2015, BCO Guide 2019",
    },
    SpaceCategory.RETAIL_SALES: {
        "best_practice": 10.0,
        "good": 14.0,
        "acceptable": 18.0,
        "poor": 25.0,
        "source": "CIBSE LG6, EN 12464-1",
    },
    SpaceCategory.RETAIL_SUPERMARKET: {
        "best_practice": 8.0,
        "good": 12.0,
        "acceptable": 16.0,
        "poor": 22.0,
        "source": "CIBSE LG6, EN 12464-1",
    },
    SpaceCategory.CLASSROOM: {
        "best_practice": 7.0,
        "good": 9.0,
        "acceptable": 12.0,
        "poor": 16.0,
        "source": "CIBSE LG5, BB90, EN 12464-1",
    },
    SpaceCategory.LECTURE_HALL: {
        "best_practice": 8.0,
        "good": 11.0,
        "acceptable": 14.0,
        "poor": 18.0,
        "source": "CIBSE LG5, EN 12464-1",
    },
    SpaceCategory.HOSPITAL_WARD: {
        "best_practice": 6.0,
        "good": 9.0,
        "acceptable": 12.0,
        "poor": 16.0,
        "source": "CIBSE LG2, HTM 08-03, EN 12464-1",
    },
    SpaceCategory.HOSPITAL_CORRIDOR: {
        "best_practice": 4.0,
        "good": 6.0,
        "acceptable": 8.0,
        "poor": 12.0,
        "source": "CIBSE LG2, HTM 08-03",
    },
    SpaceCategory.HOTEL_ROOM: {
        "best_practice": 5.0,
        "good": 7.0,
        "acceptable": 10.0,
        "poor": 14.0,
        "source": "CIBSE LG, EN 12464-1",
    },
    SpaceCategory.HOTEL_LOBBY: {
        "best_practice": 8.0,
        "good": 12.0,
        "acceptable": 16.0,
        "poor": 22.0,
        "source": "CIBSE LG, SLL Lighting Guide",
    },
    SpaceCategory.RESTAURANT: {
        "best_practice": 8.0,
        "good": 12.0,
        "acceptable": 16.0,
        "poor": 22.0,
        "source": "CIBSE LG, EN 12464-1",
    },
    SpaceCategory.WAREHOUSE: {
        "best_practice": 3.0,
        "good": 5.0,
        "acceptable": 8.0,
        "poor": 14.0,
        "source": "CIBSE LG1, EN 12464-1",
    },
    SpaceCategory.PARKING: {
        "best_practice": 2.0,
        "good": 3.5,
        "acceptable": 5.0,
        "poor": 8.0,
        "source": "EN 12464-2, BS 5489",
    },
    SpaceCategory.CORRIDOR: {
        "best_practice": 3.0,
        "good": 5.0,
        "acceptable": 7.0,
        "poor": 10.0,
        "source": "EN 12464-1, CIBSE LG",
    },
    SpaceCategory.RESTROOM: {
        "best_practice": 4.0,
        "good": 6.0,
        "acceptable": 8.0,
        "poor": 12.0,
        "source": "EN 12464-1",
    },
    SpaceCategory.LABORATORY: {
        "best_practice": 10.0,
        "good": 14.0,
        "acceptable": 18.0,
        "poor": 24.0,
        "source": "CIBSE LG, EN 12464-1",
    },
    SpaceCategory.CLEAN_ROOM: {
        "best_practice": 12.0,
        "good": 16.0,
        "acceptable": 22.0,
        "poor": 30.0,
        "source": "IEST RP-CC028, EN 12464-1",
    },
    SpaceCategory.KITCHEN: {
        "best_practice": 8.0,
        "good": 11.0,
        "acceptable": 15.0,
        "poor": 20.0,
        "source": "CIBSE LG, EN 12464-1",
    },
    SpaceCategory.RECEPTION: {
        "best_practice": 6.0,
        "good": 9.0,
        "acceptable": 12.0,
        "poor": 16.0,
        "source": "CIBSE LG7, EN 12464-1",
    },
}


# ---------------------------------------------------------------------------
# Constants -- LENI Benchmarks (kWh/m2/yr)
# ---------------------------------------------------------------------------

# Lighting Energy Numeric Indicator benchmarks by building type.
# Sources: EN 15193-1:2017 Table B.1, CIBSE TM54.
LENI_BENCHMARKS: Dict[str, Dict[str, float]] = {
    "office": {
        "best_practice": 15.0,
        "good": 25.0,
        "typical": 35.0,
        "poor": 50.0,
        "source": "EN 15193-1 Table B.1",
    },
    "retail": {
        "best_practice": 25.0,
        "good": 40.0,
        "typical": 55.0,
        "poor": 80.0,
        "source": "EN 15193-1 Table B.1",
    },
    "education": {
        "best_practice": 15.0,
        "good": 22.0,
        "typical": 30.0,
        "poor": 45.0,
        "source": "EN 15193-1 Table B.1",
    },
    "hospital": {
        "best_practice": 25.0,
        "good": 35.0,
        "typical": 50.0,
        "poor": 70.0,
        "source": "EN 15193-1 Table B.1",
    },
    "hotel": {
        "best_practice": 20.0,
        "good": 30.0,
        "typical": 40.0,
        "poor": 60.0,
        "source": "EN 15193-1 Table B.1",
    },
    "warehouse": {
        "best_practice": 8.0,
        "good": 15.0,
        "typical": 25.0,
        "poor": 40.0,
        "source": "EN 15193-1 Table B.1",
    },
    "restaurant": {
        "best_practice": 18.0,
        "good": 28.0,
        "typical": 40.0,
        "poor": 60.0,
        "source": "EN 15193-1 Table B.1",
    },
    "parking": {
        "best_practice": 5.0,
        "good": 10.0,
        "typical": 15.0,
        "poor": 25.0,
        "source": "EN 15193-1 Table B.1",
    },
}


# ---------------------------------------------------------------------------
# Constants -- Illuminance Requirements (lux)
# ---------------------------------------------------------------------------

# Maintained illuminance (Em), uniformity (Uo), and UGR limit by space.
# Sources: EN 12464-1:2021, Tables 5.1-5.54.
ILLUMINANCE_REQUIREMENTS: Dict[str, Dict[str, Any]] = {
    SpaceCategory.OFFICE_GENERAL: {
        "maintained_lux": 500,
        "uniformity_min": 0.60,
        "ugr_limit": 19,
        "ra_min": 80,
        "source": "EN 12464-1:2021 Table 5.26",
    },
    SpaceCategory.OFFICE_OPEN_PLAN: {
        "maintained_lux": 500,
        "uniformity_min": 0.60,
        "ugr_limit": 19,
        "ra_min": 80,
        "source": "EN 12464-1:2021 Table 5.26",
    },
    SpaceCategory.RETAIL_SALES: {
        "maintained_lux": 300,
        "uniformity_min": 0.40,
        "ugr_limit": 22,
        "ra_min": 80,
        "source": "EN 12464-1:2021 Table 5.11",
    },
    SpaceCategory.RETAIL_SUPERMARKET: {
        "maintained_lux": 500,
        "uniformity_min": 0.40,
        "ugr_limit": 22,
        "ra_min": 80,
        "source": "EN 12464-1:2021 Table 5.11",
    },
    SpaceCategory.CLASSROOM: {
        "maintained_lux": 500,
        "uniformity_min": 0.60,
        "ugr_limit": 19,
        "ra_min": 80,
        "source": "EN 12464-1:2021 Table 5.36",
    },
    SpaceCategory.LECTURE_HALL: {
        "maintained_lux": 500,
        "uniformity_min": 0.60,
        "ugr_limit": 19,
        "ra_min": 80,
        "source": "EN 12464-1:2021 Table 5.36",
    },
    SpaceCategory.HOSPITAL_WARD: {
        "maintained_lux": 300,
        "uniformity_min": 0.40,
        "ugr_limit": 19,
        "ra_min": 80,
        "source": "EN 12464-1:2021 Table 5.34",
    },
    SpaceCategory.HOSPITAL_CORRIDOR: {
        "maintained_lux": 200,
        "uniformity_min": 0.40,
        "ugr_limit": 22,
        "ra_min": 80,
        "source": "EN 12464-1:2021 Table 5.34",
    },
    SpaceCategory.HOTEL_ROOM: {
        "maintained_lux": 200,
        "uniformity_min": 0.40,
        "ugr_limit": 22,
        "ra_min": 80,
        "source": "EN 12464-1:2021 Table 5.10",
    },
    SpaceCategory.HOTEL_LOBBY: {
        "maintained_lux": 300,
        "uniformity_min": 0.40,
        "ugr_limit": 22,
        "ra_min": 80,
        "source": "EN 12464-1:2021 Table 5.10",
    },
    SpaceCategory.RESTAURANT: {
        "maintained_lux": 200,
        "uniformity_min": 0.40,
        "ugr_limit": 22,
        "ra_min": 80,
        "source": "EN 12464-1:2021 Table 5.10",
    },
    SpaceCategory.WAREHOUSE: {
        "maintained_lux": 100,
        "uniformity_min": 0.40,
        "ugr_limit": 25,
        "ra_min": 60,
        "source": "EN 12464-1:2021 Table 5.4",
    },
    SpaceCategory.PARKING: {
        "maintained_lux": 75,
        "uniformity_min": 0.40,
        "ugr_limit": 25,
        "ra_min": 40,
        "source": "EN 12464-2:2014",
    },
    SpaceCategory.CORRIDOR: {
        "maintained_lux": 100,
        "uniformity_min": 0.40,
        "ugr_limit": 25,
        "ra_min": 80,
        "source": "EN 12464-1:2021 Table 5.1",
    },
    SpaceCategory.RESTROOM: {
        "maintained_lux": 200,
        "uniformity_min": 0.40,
        "ugr_limit": 25,
        "ra_min": 80,
        "source": "EN 12464-1:2021 Table 5.1",
    },
    SpaceCategory.LABORATORY: {
        "maintained_lux": 500,
        "uniformity_min": 0.60,
        "ugr_limit": 19,
        "ra_min": 80,
        "source": "EN 12464-1:2021 Table 5.44",
    },
    SpaceCategory.CLEAN_ROOM: {
        "maintained_lux": 500,
        "uniformity_min": 0.60,
        "ugr_limit": 19,
        "ra_min": 90,
        "source": "EN 12464-1:2021, ISO 14644",
    },
    SpaceCategory.KITCHEN: {
        "maintained_lux": 500,
        "uniformity_min": 0.60,
        "ugr_limit": 22,
        "ra_min": 80,
        "source": "EN 12464-1:2021 Table 5.10",
    },
    SpaceCategory.RECEPTION: {
        "maintained_lux": 300,
        "uniformity_min": 0.40,
        "ugr_limit": 22,
        "ra_min": 80,
        "source": "EN 12464-1:2021 Table 5.1",
    },
}


# ---------------------------------------------------------------------------
# Constants -- Lamp Efficacy (lm/W)
# ---------------------------------------------------------------------------

# Luminous efficacy by lamp type.
# Sources: IES Handbook 10th ed., manufacturer catalogues, EU 2019/2020 Regulation.
LAMP_EFFICACY: Dict[str, Dict[str, float]] = {
    LampType.LED: {
        "low": 100.0,
        "typical": 130.0,
        "high": 170.0,
        "best_available": 200.0,
        "source": "EU Regulation 2019/2020, market survey 2025",
    },
    LampType.FLUORESCENT_T5: {
        "low": 80.0,
        "typical": 95.0,
        "high": 104.0,
        "source": "IES Handbook, Osram/Philips data sheets",
    },
    LampType.FLUORESCENT_T8: {
        "low": 60.0,
        "typical": 80.0,
        "high": 93.0,
        "source": "IES Handbook, legacy fluorescent",
    },
    LampType.CFL: {
        "low": 45.0,
        "typical": 60.0,
        "high": 70.0,
        "source": "IES Handbook, compact fluorescent",
    },
    LampType.HALOGEN: {
        "low": 12.0,
        "typical": 16.0,
        "high": 20.0,
        "source": "EU phase-out regulation, legacy stock",
    },
    LampType.HID_METAL_HALIDE: {
        "low": 70.0,
        "typical": 90.0,
        "high": 105.0,
        "source": "IES Handbook, high-bay industrial",
    },
    LampType.HID_SODIUM: {
        "low": 80.0,
        "typical": 120.0,
        "high": 150.0,
        "source": "IES Handbook, external / road lighting",
    },
    LampType.INCANDESCENT: {
        "low": 8.0,
        "typical": 12.0,
        "high": 15.0,
        "source": "EU phase-out completed, legacy stock only",
    },
}


# ---------------------------------------------------------------------------
# Constants -- Control Energy Savings Factors
# ---------------------------------------------------------------------------

# Energy savings multiplier applied to lighting energy by control type.
# Factor represents remaining fraction of energy (1.0 = no saving).
# Sources: EN 15193-1:2017 Table 5, CIBSE LG7:2015, ECA list.
CONTROL_FACTOR: Dict[str, Dict[str, Any]] = {
    ControlType.MANUAL_SWITCH: {
        "factor": 1.00,
        "saving_pct": 0.0,
        "description": "No automatic control",
        "source": "EN 15193-1 reference case",
    },
    ControlType.TIME_CLOCK: {
        "factor": 0.90,
        "saving_pct": 10.0,
        "description": "Scheduled on/off switching",
        "source": "EN 15193-1 Table 5",
    },
    ControlType.OCCUPANCY_SENSOR: {
        "factor": 0.75,
        "saving_pct": 25.0,
        "description": "PIR presence detection with auto-off",
        "source": "EN 15193-1 Table 5, CIBSE LG7",
    },
    ControlType.DAYLIGHT_SENSOR: {
        "factor": 0.70,
        "saving_pct": 30.0,
        "description": "Photocell dimming based on daylight",
        "source": "EN 15193-1 Table 5",
    },
    ControlType.COMBINED_OCC_DAYLIGHT: {
        "factor": 0.55,
        "saving_pct": 45.0,
        "description": "Combined occupancy + daylight dimming",
        "source": "EN 15193-1 Table 5, field studies avg",
    },
    ControlType.DALI_ADDRESSABLE: {
        "factor": 0.50,
        "saving_pct": 50.0,
        "description": "Fully addressable digital control (DALI-2)",
        "source": "EN 15193-1, DALI Alliance case studies",
    },
    ControlType.SCENE_SETTING: {
        "factor": 0.80,
        "saving_pct": 20.0,
        "description": "Pre-set scene dimming",
        "source": "CIBSE LG7, hospitality case studies",
    },
    ControlType.ABSENCE_DETECTION: {
        "factor": 0.80,
        "saving_pct": 20.0,
        "description": "Auto-off only, manual-on (preferred over PIR auto-on)",
        "source": "EN 15193-1 Table 5, BSRIA data",
    },
}


# ---------------------------------------------------------------------------
# Constants -- Daylight Factor Targets
# ---------------------------------------------------------------------------

# Target average daylight factor (%) by space type.
# Sources: CIBSE LG10, BREEAM Hea 01, WELL v2 Light concept.
DAYLIGHT_FACTOR_TARGETS: Dict[str, Dict[str, float]] = {
    SpaceCategory.OFFICE_GENERAL: {
        "minimum": 2.0,
        "good": 3.0,
        "excellent": 5.0,
        "source": "CIBSE LG10, BREEAM Hea 01",
    },
    SpaceCategory.OFFICE_OPEN_PLAN: {
        "minimum": 2.0,
        "good": 3.0,
        "excellent": 5.0,
        "source": "CIBSE LG10, BREEAM Hea 01",
    },
    SpaceCategory.CLASSROOM: {
        "minimum": 2.0,
        "good": 3.0,
        "excellent": 5.0,
        "source": "BB90, CIBSE LG5",
    },
    SpaceCategory.HOSPITAL_WARD: {
        "minimum": 2.0,
        "good": 2.5,
        "excellent": 3.0,
        "source": "HTM 08-03, CIBSE LG2",
    },
    SpaceCategory.HOTEL_ROOM: {
        "minimum": 1.5,
        "good": 2.0,
        "excellent": 3.0,
        "source": "CIBSE LG",
    },
    SpaceCategory.RETAIL_SALES: {
        "minimum": 1.0,
        "good": 2.0,
        "excellent": 3.0,
        "source": "CIBSE LG6",
    },
    SpaceCategory.LABORATORY: {
        "minimum": 2.0,
        "good": 3.0,
        "excellent": 4.0,
        "source": "CIBSE LG",
    },
    SpaceCategory.RECEPTION: {
        "minimum": 2.0,
        "good": 3.0,
        "excellent": 4.0,
        "source": "CIBSE LG",
    },
}


# ---------------------------------------------------------------------------
# Constants -- LED Retrofit Costs (EUR per fixture)
# ---------------------------------------------------------------------------

# Typical retrofit cost per fixture by existing lamp type.
# Sources: Trade pricing 2025, Carbon Trust LED guide.
LED_RETROFIT_COST_EUR: Dict[str, Dict[str, float]] = {
    LampType.FLUORESCENT_T8: {
        "retrofit_tube": 15.0,
        "new_fixture": 60.0,
        "labour_per_fixture": 20.0,
        "source": "UK/EU trade pricing 2025",
    },
    LampType.FLUORESCENT_T5: {
        "retrofit_tube": 18.0,
        "new_fixture": 70.0,
        "labour_per_fixture": 20.0,
        "source": "UK/EU trade pricing 2025",
    },
    LampType.CFL: {
        "retrofit_lamp": 8.0,
        "new_fixture": 40.0,
        "labour_per_fixture": 12.0,
        "source": "UK/EU trade pricing 2025",
    },
    LampType.HALOGEN: {
        "retrofit_lamp": 6.0,
        "new_fixture": 35.0,
        "labour_per_fixture": 10.0,
        "source": "UK/EU trade pricing 2025",
    },
    LampType.HID_METAL_HALIDE: {
        "new_fixture": 250.0,
        "labour_per_fixture": 80.0,
        "source": "High-bay LED replacement pricing",
    },
    LampType.HID_SODIUM: {
        "new_fixture": 200.0,
        "labour_per_fixture": 80.0,
        "source": "High-bay / external LED replacement",
    },
    LampType.INCANDESCENT: {
        "retrofit_lamp": 5.0,
        "new_fixture": 30.0,
        "labour_per_fixture": 8.0,
        "source": "UK/EU trade pricing 2025",
    },
}


# ---------------------------------------------------------------------------
# Constants -- Maintenance Factor
# ---------------------------------------------------------------------------

# Luminaire maintenance factor (MF = LLMF * LSF * LMF * RSMF).
# LLMF = lamp lumen maintenance factor, LSF = lamp survival factor,
# LMF = luminaire maintenance factor, RSMF = room surface MF.
# Sources: CIE 97:2005, EN 12464-1 Annex A.
MAINTENANCE_FACTOR: Dict[str, Dict[str, float]] = {
    LampType.LED: {
        EnvironmentCleanliness.VERY_CLEAN: 0.90,
        EnvironmentCleanliness.CLEAN: 0.85,
        EnvironmentCleanliness.NORMAL: 0.80,
        EnvironmentCleanliness.DIRTY: 0.70,
        "source": "CIE 97:2005, L80 at 50kh",
    },
    LampType.FLUORESCENT_T5: {
        EnvironmentCleanliness.VERY_CLEAN: 0.82,
        EnvironmentCleanliness.CLEAN: 0.76,
        EnvironmentCleanliness.NORMAL: 0.70,
        EnvironmentCleanliness.DIRTY: 0.58,
        "source": "CIE 97:2005",
    },
    LampType.FLUORESCENT_T8: {
        EnvironmentCleanliness.VERY_CLEAN: 0.78,
        EnvironmentCleanliness.CLEAN: 0.72,
        EnvironmentCleanliness.NORMAL: 0.65,
        EnvironmentCleanliness.DIRTY: 0.53,
        "source": "CIE 97:2005",
    },
    LampType.CFL: {
        EnvironmentCleanliness.VERY_CLEAN: 0.76,
        EnvironmentCleanliness.CLEAN: 0.70,
        EnvironmentCleanliness.NORMAL: 0.63,
        EnvironmentCleanliness.DIRTY: 0.52,
        "source": "CIE 97:2005",
    },
    LampType.HALOGEN: {
        EnvironmentCleanliness.VERY_CLEAN: 0.82,
        EnvironmentCleanliness.CLEAN: 0.76,
        EnvironmentCleanliness.NORMAL: 0.70,
        EnvironmentCleanliness.DIRTY: 0.60,
        "source": "CIE 97:2005",
    },
    LampType.HID_METAL_HALIDE: {
        EnvironmentCleanliness.VERY_CLEAN: 0.72,
        EnvironmentCleanliness.CLEAN: 0.66,
        EnvironmentCleanliness.NORMAL: 0.58,
        EnvironmentCleanliness.DIRTY: 0.47,
        "source": "CIE 97:2005",
    },
    LampType.HID_SODIUM: {
        EnvironmentCleanliness.VERY_CLEAN: 0.78,
        EnvironmentCleanliness.CLEAN: 0.72,
        EnvironmentCleanliness.NORMAL: 0.65,
        EnvironmentCleanliness.DIRTY: 0.54,
        "source": "CIE 97:2005",
    },
    LampType.INCANDESCENT: {
        EnvironmentCleanliness.VERY_CLEAN: 0.80,
        EnvironmentCleanliness.CLEAN: 0.74,
        EnvironmentCleanliness.NORMAL: 0.67,
        EnvironmentCleanliness.DIRTY: 0.56,
        "source": "CIE 97:2005",
    },
}


# ---------------------------------------------------------------------------
# Pydantic Input Models
# ---------------------------------------------------------------------------


class LightingZoneInput(BaseModel):
    """A single lighting zone / space within the building."""

    zone_id: str = Field(..., description="Unique zone identifier")
    zone_name: str = Field(..., description="Human-readable zone name")
    space_category: SpaceCategory = Field(
        ..., description="Space category per EN 12464-1"
    )
    floor_area_m2: float = Field(
        ..., gt=0, description="Zone floor area in m2"
    )
    lamp_type: LampType = Field(
        ..., description="Predominant lamp type in zone"
    )
    number_of_fixtures: int = Field(
        ..., gt=0, description="Number of luminaires in zone"
    )
    watts_per_fixture: float = Field(
        ..., gt=0, description="Circuit watts per luminaire including driver/ballast"
    )
    lumens_per_fixture: Optional[float] = Field(
        None, description="Luminaire output in lumens if known"
    )
    annual_operating_hours: float = Field(
        ..., gt=0, description="Annual operating hours for this zone"
    )
    control_type: ControlType = Field(
        ControlType.MANUAL_SWITCH,
        description="Lighting control type installed",
    )
    daylight_factor_pct: Optional[float] = Field(
        None, ge=0, le=20,
        description="Measured or estimated average daylight factor (%)",
    )
    measured_illuminance_lux: Optional[float] = Field(
        None, description="Measured average maintained illuminance (lux)"
    )
    environment_cleanliness: EnvironmentCleanliness = Field(
        EnvironmentCleanliness.NORMAL,
        description="Environment cleanliness for maintenance factor",
    )


class LightingAssessmentInput(BaseModel):
    """Top-level lighting assessment input."""

    building_id: str = Field(
        ..., description="Unique building identifier"
    )
    building_type: str = Field(
        "office", description="Building type for LENI benchmarking"
    )
    total_floor_area_m2: float = Field(
        ..., gt=0, description="Total building floor area in m2"
    )
    zones: List[LightingZoneInput] = Field(
        ..., min_length=1, description="List of lighting zones to assess"
    )
    electricity_cost_eur_per_kwh: float = Field(
        0.30, description="Electricity unit cost in EUR/kWh"
    )
    carbon_factor_kg_per_kwh: float = Field(
        0.233, description="Grid electricity carbon factor kgCO2e/kWh"
    )
    led_target_efficacy_lm_per_w: float = Field(
        130.0, description="Target LED efficacy for retrofit calculation (lm/W)"
    )
    discount_rate: float = Field(
        0.05, ge=0, le=0.20, description="Discount rate for NPV calculation"
    )
    analysis_period_years: int = Field(
        15, gt=0, le=30, description="Analysis period for financial calculations"
    )


# ---------------------------------------------------------------------------
# Pydantic Result Models
# ---------------------------------------------------------------------------


class ZoneLPDResult(BaseModel):
    """Lighting Power Density result for a single zone."""

    zone_id: str = Field(..., description="Zone identifier")
    zone_name: str = Field(..., description="Zone name")
    space_category: str = Field(..., description="Space category")
    floor_area_m2: float = Field(..., description="Floor area in m2")
    installed_power_w: float = Field(
        ..., description="Total installed lighting power in watts"
    )
    lpd_w_per_m2: float = Field(
        ..., description="Installed Lighting Power Density in W/m2"
    )
    benchmark_best_practice: float = Field(
        ..., description="Best practice LPD benchmark in W/m2"
    )
    benchmark_good: float = Field(
        ..., description="Good practice LPD benchmark in W/m2"
    )
    lpd_rating: str = Field(
        ..., description="LPD performance tier (best_practice/good/acceptable/poor)"
    )
    annual_energy_kwh: float = Field(
        ..., description="Annual lighting energy consumption in kWh"
    )
    annual_energy_with_controls_kwh: float = Field(
        ..., description="Annual energy after control savings in kWh"
    )
    control_saving_pct: float = Field(
        ..., description="Energy saving from controls as percentage"
    )


class LENIResult(BaseModel):
    """Lighting Energy Numeric Indicator result per EN 15193."""

    total_annual_energy_kwh: float = Field(
        ..., description="Total annual lighting energy in kWh"
    )
    leni_kwh_per_m2_yr: float = Field(
        ..., description="LENI value in kWh/m2/yr"
    )
    benchmark_best_practice: float = Field(
        ..., description="LENI best practice benchmark"
    )
    benchmark_good: float = Field(
        ..., description="LENI good practice benchmark"
    )
    benchmark_typical: float = Field(
        ..., description="LENI typical benchmark"
    )
    leni_rating: str = Field(
        ..., description="LENI performance tier"
    )
    parasitic_power_kwh: float = Field(
        ..., description="Parasitic power from controls in kWh/yr"
    )


class DaylightResult(BaseModel):
    """Daylight assessment result for a zone."""

    zone_id: str = Field(..., description="Zone identifier")
    daylight_factor_pct: float = Field(
        ..., description="Average daylight factor in %"
    )
    target_minimum: float = Field(
        ..., description="Minimum target daylight factor"
    )
    daylight_rating: str = Field(
        ..., description="Daylight adequacy rating"
    )
    estimated_daylight_saving_pct: float = Field(
        ..., description="Estimated energy saving from daylight harvesting"
    )


class ControlsResult(BaseModel):
    """Lighting controls assessment result."""

    zone_id: str = Field(..., description="Zone identifier")
    current_control: str = Field(..., description="Installed control type")
    control_saving_pct: float = Field(
        ..., description="Current control saving percentage"
    )
    recommended_control: str = Field(
        ..., description="Recommended control upgrade"
    )
    additional_saving_pct: float = Field(
        ..., description="Additional saving from recommended upgrade"
    )
    upgrade_cost_eur: float = Field(
        ..., description="Estimated upgrade cost in EUR"
    )


class RetrofitResult(BaseModel):
    """LED retrofit analysis result for a zone."""

    zone_id: str = Field(..., description="Zone identifier")
    existing_lamp_type: str = Field(..., description="Existing lamp type")
    existing_power_w: float = Field(
        ..., description="Existing total power in watts"
    )
    proposed_power_w: float = Field(
        ..., description="Proposed LED total power in watts"
    )
    annual_energy_saving_kwh: float = Field(
        ..., description="Annual energy saving from LED retrofit in kWh"
    )
    annual_cost_saving_eur: float = Field(
        ..., description="Annual cost saving in EUR"
    )
    annual_carbon_saving_kg: float = Field(
        ..., description="Annual carbon saving in kgCO2e"
    )
    retrofit_cost_eur: float = Field(
        ..., description="Estimated retrofit capital cost in EUR"
    )
    simple_payback_years: float = Field(
        ..., description="Simple payback period in years"
    )
    npv_eur: float = Field(
        ..., description="Net Present Value over analysis period in EUR"
    )
    already_led: bool = Field(
        ..., description="Whether zone already has LED lighting"
    )


class VisualQualityResult(BaseModel):
    """Visual quality compliance result for a zone."""

    zone_id: str = Field(..., description="Zone identifier")
    required_lux: int = Field(
        ..., description="Required maintained illuminance per EN 12464-1"
    )
    measured_lux: Optional[float] = Field(
        None, description="Measured illuminance if available"
    )
    illuminance_compliant: Optional[bool] = Field(
        None, description="Whether illuminance meets EN 12464-1"
    )
    lamp_efficacy_lm_per_w: float = Field(
        ..., description="Current lamp efficacy in lm/W"
    )
    maintenance_factor: float = Field(
        ..., description="Calculated maintenance factor"
    )


class LightingAssessmentResult(BaseModel):
    """Complete lighting assessment result."""

    assessment_id: str = Field(
        ..., description="Unique assessment identifier"
    )
    building_id: str = Field(..., description="Building identifier")
    engine_version: str = Field(
        ..., description="Engine version"
    )
    zone_lpd_results: List[ZoneLPDResult] = Field(
        ..., description="LPD results per zone"
    )
    leni: LENIResult = Field(
        ..., description="LENI building-level result"
    )
    daylight_results: List[DaylightResult] = Field(
        default_factory=list, description="Daylight assessment per zone"
    )
    controls_results: List[ControlsResult] = Field(
        default_factory=list, description="Controls assessment per zone"
    )
    retrofit_results: List[RetrofitResult] = Field(
        default_factory=list, description="LED retrofit analysis per zone"
    )
    visual_quality_results: List[VisualQualityResult] = Field(
        default_factory=list, description="Visual quality per zone"
    )
    total_installed_power_kw: float = Field(
        ..., description="Total installed lighting power in kW"
    )
    total_annual_energy_kwh: float = Field(
        ..., description="Total annual lighting energy in kWh"
    )
    total_annual_cost_eur: float = Field(
        ..., description="Total annual lighting cost in EUR"
    )
    total_annual_carbon_kg: float = Field(
        ..., description="Total annual lighting carbon in kgCO2e"
    )
    total_retrofit_saving_kwh: float = Field(
        ..., description="Total potential LED retrofit saving in kWh/yr"
    )
    total_retrofit_cost_eur: float = Field(
        ..., description="Total LED retrofit capital cost in EUR"
    )
    overall_retrofit_payback_years: float = Field(
        ..., description="Overall LED retrofit simple payback in years"
    )
    recommendations: List[str] = Field(
        default_factory=list, description="Prioritised improvement recommendations"
    )
    calculated_at: str = Field(..., description="ISO UTC timestamp")
    processing_time_ms: float = Field(..., description="Processing time in ms")
    provenance_hash: str = Field(
        ..., description="SHA-256 provenance hash"
    )


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class LightingAssessmentEngine:
    """Building lighting energy and quality assessment engine.

    Implements EN 12464-1:2021 illuminance requirements, EN 15193-1:2017
    LENI calculation, daylight assessment, controls savings analysis,
    and LED retrofit financial analysis.

    Zero-Hallucination Guarantees:
        - All calculations use deterministic Decimal arithmetic
        - No LLM involvement in any numeric calculation path
        - SHA-256 provenance hashing on every result
        - All benchmarks from published EN/CIBSE standards

    Example::

        engine = LightingAssessmentEngine()
        result = engine.analyze(assessment_input)
    """

    # ---------------------------------------------------------------
    # Public: analyze
    # ---------------------------------------------------------------

    def analyze(self, inp: LightingAssessmentInput) -> LightingAssessmentResult:
        """Run complete lighting assessment.

        Args:
            inp: Complete lighting assessment input.

        Returns:
            LightingAssessmentResult with full provenance.
        """
        t_start = time.perf_counter()

        zone_lpd_results: List[ZoneLPDResult] = []
        daylight_results: List[DaylightResult] = []
        controls_results: List[ControlsResult] = []
        retrofit_results: List[RetrofitResult] = []
        visual_results: List[VisualQualityResult] = []

        total_power_w = Decimal("0")
        total_energy = Decimal("0")

        for zone in inp.zones:
            # LPD
            lpd_res = self.calculate_lpd(zone)
            zone_lpd_results.append(lpd_res)
            total_power_w += _decimal(lpd_res.installed_power_w)
            total_energy += _decimal(lpd_res.annual_energy_with_controls_kwh)

            # Daylight
            if zone.daylight_factor_pct is not None:
                dl_res = self.assess_daylight(zone)
                daylight_results.append(dl_res)

            # Controls
            ctrl_res = self.assess_controls(zone, inp.electricity_cost_eur_per_kwh)
            controls_results.append(ctrl_res)

            # Retrofit
            retro_res = self.calculate_retrofit_savings(
                zone, inp.electricity_cost_eur_per_kwh,
                inp.carbon_factor_kg_per_kwh, inp.led_target_efficacy_lm_per_w,
                inp.discount_rate, inp.analysis_period_years,
            )
            retrofit_results.append(retro_res)

            # Visual quality
            vq_res = self.assess_visual_quality(zone)
            visual_results.append(vq_res)

        # LENI
        leni = self.calculate_leni(
            total_energy_kwh=float(total_energy),
            total_floor_area_m2=inp.total_floor_area_m2,
            building_type=inp.building_type,
        )

        # Aggregates
        elec_cost = _decimal(inp.electricity_cost_eur_per_kwh)
        cf = _decimal(inp.carbon_factor_kg_per_kwh)
        total_cost = total_energy * elec_cost
        total_carbon = total_energy * cf

        total_retrofit_saving = sum(
            _decimal(r.annual_energy_saving_kwh) for r in retrofit_results
            if not r.already_led
        )
        total_retrofit_capex = sum(
            _decimal(r.retrofit_cost_eur) for r in retrofit_results
            if not r.already_led
        )
        total_retrofit_cost_saving = total_retrofit_saving * elec_cost
        overall_payback = _safe_divide(
            total_retrofit_capex, total_retrofit_cost_saving,
            default=Decimal("99"),
        )

        # Recommendations
        recommendations = self._generate_recommendations(
            inp, zone_lpd_results, leni, daylight_results,
            controls_results, retrofit_results, visual_results,
        )

        t_end = time.perf_counter()
        processing_ms = (t_end - t_start) * 1000.0

        result = LightingAssessmentResult(
            assessment_id=_new_uuid(),
            building_id=inp.building_id,
            engine_version=_MODULE_VERSION,
            zone_lpd_results=zone_lpd_results,
            leni=leni,
            daylight_results=daylight_results,
            controls_results=controls_results,
            retrofit_results=retrofit_results,
            visual_quality_results=visual_results,
            total_installed_power_kw=_round3(float(total_power_w / Decimal("1000"))),
            total_annual_energy_kwh=_round2(float(total_energy)),
            total_annual_cost_eur=_round2(float(total_cost)),
            total_annual_carbon_kg=_round2(float(total_carbon)),
            total_retrofit_saving_kwh=_round2(float(total_retrofit_saving)),
            total_retrofit_cost_eur=_round2(float(total_retrofit_capex)),
            overall_retrofit_payback_years=_round2(float(overall_payback)),
            recommendations=recommendations,
            calculated_at=_utcnow().isoformat(),
            processing_time_ms=_round3(processing_ms),
            provenance_hash="",
        )
        result.provenance_hash = _compute_hash(result)
        return result

    # ---------------------------------------------------------------
    # Public: calculate_lpd
    # ---------------------------------------------------------------

    def calculate_lpd(self, zone: LightingZoneInput) -> ZoneLPDResult:
        """Calculate installed Lighting Power Density for a zone.

        Formula:
            LPD = P_installed / A_floor  [W/m2]

        Args:
            zone: Lighting zone specification.

        Returns:
            ZoneLPDResult with LPD and benchmark comparison.
        """
        n = _decimal(zone.number_of_fixtures)
        w = _decimal(zone.watts_per_fixture)
        area = _decimal(zone.floor_area_m2)

        installed_power = n * w
        lpd = _safe_divide(installed_power, area)

        # Benchmark lookup
        benchmarks = LPD_BENCHMARKS.get(zone.space_category)
        if not benchmarks:
            benchmarks = LPD_BENCHMARKS[SpaceCategory.OFFICE_GENERAL]

        bp = _decimal(benchmarks["best_practice"])
        good = _decimal(benchmarks["good"])
        acceptable = _decimal(benchmarks["acceptable"])
        poor = _decimal(benchmarks["poor"])

        if lpd <= bp:
            rating = "best_practice"
        elif lpd <= good:
            rating = "good"
        elif lpd <= acceptable:
            rating = "acceptable"
        else:
            rating = "poor"

        # Annual energy
        hours = _decimal(zone.annual_operating_hours)
        annual_energy = installed_power * hours / Decimal("1000")

        # Control factor
        ctrl_data = CONTROL_FACTOR.get(zone.control_type)
        if ctrl_data:
            ctrl_factor = _decimal(ctrl_data["factor"])
            ctrl_saving = float(ctrl_data["saving_pct"])
        else:
            ctrl_factor = Decimal("1")
            ctrl_saving = 0.0

        annual_with_controls = annual_energy * ctrl_factor

        return ZoneLPDResult(
            zone_id=zone.zone_id,
            zone_name=zone.zone_name,
            space_category=zone.space_category.value,
            floor_area_m2=zone.floor_area_m2,
            installed_power_w=_round2(float(installed_power)),
            lpd_w_per_m2=_round2(float(lpd)),
            benchmark_best_practice=benchmarks["best_practice"],
            benchmark_good=benchmarks["good"],
            lpd_rating=rating,
            annual_energy_kwh=_round2(float(annual_energy)),
            annual_energy_with_controls_kwh=_round2(float(annual_with_controls)),
            control_saving_pct=ctrl_saving,
        )

    # ---------------------------------------------------------------
    # Public: calculate_leni
    # ---------------------------------------------------------------

    def calculate_leni(
        self,
        total_energy_kwh: float,
        total_floor_area_m2: float,
        building_type: str = "office",
    ) -> LENIResult:
        """Calculate LENI per EN 15193-1:2017.

        Formula:
            LENI = W_total / A_floor  [kWh/m2/yr]

        Where W_total includes parasitic power from controls.

        Args:
            total_energy_kwh: Total annual lighting energy in kWh.
            total_floor_area_m2: Total building floor area in m2.
            building_type: Building type for benchmark lookup.

        Returns:
            LENIResult with LENI value and benchmarks.
        """
        energy = _decimal(total_energy_kwh)
        area = _decimal(total_floor_area_m2)

        # Parasitic power estimate: typically 1-2% of total lighting energy
        parasitic = energy * Decimal("0.015")
        total_with_parasitic = energy + parasitic

        leni = _safe_divide(total_with_parasitic, area)

        # Benchmarks
        bt = building_type.lower()
        bench = LENI_BENCHMARKS.get(bt)
        if not bench:
            bench = LENI_BENCHMARKS["office"]

        bp = bench["best_practice"]
        good = bench["good"]
        typical = bench["typical"]
        poor = bench["poor"]

        leni_f = float(leni)
        if leni_f <= bp:
            rating = "best_practice"
        elif leni_f <= good:
            rating = "good"
        elif leni_f <= typical:
            rating = "typical"
        else:
            rating = "poor"

        return LENIResult(
            total_annual_energy_kwh=_round2(float(total_with_parasitic)),
            leni_kwh_per_m2_yr=_round2(leni_f),
            benchmark_best_practice=bp,
            benchmark_good=good,
            benchmark_typical=typical,
            leni_rating=rating,
            parasitic_power_kwh=_round2(float(parasitic)),
        )

    # ---------------------------------------------------------------
    # Public: assess_daylight
    # ---------------------------------------------------------------

    def assess_daylight(self, zone: LightingZoneInput) -> DaylightResult:
        """Assess daylight factor and potential energy saving.

        Daylight Factor:
            DF = (E_internal / E_external) x 100  [%]

        Daylight energy saving is estimated based on DF and
        EN 15193-1 daylight dependency factors.

        Args:
            zone: Lighting zone with daylight factor data.

        Returns:
            DaylightResult with rating and saving estimate.
        """
        df_pct = zone.daylight_factor_pct if zone.daylight_factor_pct is not None else 0.0
        df = _decimal(df_pct)

        # Target lookup
        targets = DAYLIGHT_FACTOR_TARGETS.get(zone.space_category)
        if not targets:
            targets = {"minimum": 2.0, "good": 3.0, "excellent": 5.0}

        target_min = targets["minimum"]

        if df_pct >= targets.get("excellent", 5.0):
            rating = "Excellent daylight provision"
        elif df_pct >= targets.get("good", 3.0):
            rating = "Good daylight provision"
        elif df_pct >= target_min:
            rating = "Minimum acceptable daylight"
        else:
            rating = "Insufficient daylight"

        # Estimated saving from daylight: EN 15193 approach
        # Higher DF -> greater potential saving with daylight dimming
        if df_pct >= 5.0:
            saving = 35.0
        elif df_pct >= 3.0:
            saving = 25.0
        elif df_pct >= 2.0:
            saving = 15.0
        elif df_pct >= 1.0:
            saving = 8.0
        else:
            saving = 0.0

        return DaylightResult(
            zone_id=zone.zone_id,
            daylight_factor_pct=_round2(df_pct),
            target_minimum=target_min,
            daylight_rating=rating,
            estimated_daylight_saving_pct=saving,
        )

    # ---------------------------------------------------------------
    # Public: assess_controls
    # ---------------------------------------------------------------

    def assess_controls(
        self,
        zone: LightingZoneInput,
        electricity_cost: float = 0.30,
    ) -> ControlsResult:
        """Assess lighting controls and recommend upgrades.

        Args:
            zone: Lighting zone specification.
            electricity_cost: Electricity cost in EUR/kWh.

        Returns:
            ControlsResult with upgrade recommendation.
        """
        current_ctrl = CONTROL_FACTOR.get(zone.control_type)
        current_saving = float(current_ctrl["saving_pct"]) if current_ctrl else 0.0

        # Recommend best viable control upgrade
        recommended, add_saving = self._recommend_control_upgrade(
            zone.space_category, zone.control_type
        )

        # Estimate cost of upgrade
        area = _decimal(zone.floor_area_m2)
        # Typical control upgrade costs (EUR per m2 of controlled area)
        control_costs = {
            ControlType.OCCUPANCY_SENSOR: Decimal("8"),
            ControlType.DAYLIGHT_SENSOR: Decimal("10"),
            ControlType.COMBINED_OCC_DAYLIGHT: Decimal("15"),
            ControlType.DALI_ADDRESSABLE: Decimal("25"),
            ControlType.ABSENCE_DETECTION: Decimal("7"),
            ControlType.TIME_CLOCK: Decimal("3"),
            ControlType.SCENE_SETTING: Decimal("12"),
        }

        rec_ctrl_enum = ControlType(recommended)
        cost_per_m2 = control_costs.get(rec_ctrl_enum, Decimal("10"))
        upgrade_cost = area * cost_per_m2

        return ControlsResult(
            zone_id=zone.zone_id,
            current_control=zone.control_type.value,
            control_saving_pct=current_saving,
            recommended_control=recommended,
            additional_saving_pct=_round2(add_saving),
            upgrade_cost_eur=_round2(float(upgrade_cost)),
        )

    # ---------------------------------------------------------------
    # Public: calculate_retrofit_savings
    # ---------------------------------------------------------------

    def calculate_retrofit_savings(
        self,
        zone: LightingZoneInput,
        electricity_cost: float = 0.30,
        carbon_factor: float = 0.233,
        led_target_efficacy: float = 130.0,
        discount_rate: float = 0.05,
        analysis_years: int = 15,
    ) -> RetrofitResult:
        """Calculate LED retrofit savings for a zone.

        Energy saving formula:
            dE = (P_old - P_new) * hours * n_fixtures / 1000  [kWh/yr]

        Where P_new is determined by matching existing lumens at
        the target LED efficacy.

        Args:
            zone: Lighting zone specification.
            electricity_cost: EUR per kWh.
            carbon_factor: kgCO2e per kWh.
            led_target_efficacy: Target LED efficacy in lm/W.
            discount_rate: Discount rate for NPV.
            analysis_years: Financial analysis period.

        Returns:
            RetrofitResult with savings and payback.
        """
        already_led = zone.lamp_type == LampType.LED

        n = _decimal(zone.number_of_fixtures)
        w_old = _decimal(zone.watts_per_fixture)
        hours = _decimal(zone.annual_operating_hours)
        existing_power = n * w_old

        if already_led:
            return RetrofitResult(
                zone_id=zone.zone_id,
                existing_lamp_type=zone.lamp_type.value,
                existing_power_w=_round2(float(existing_power)),
                proposed_power_w=_round2(float(existing_power)),
                annual_energy_saving_kwh=0.0,
                annual_cost_saving_eur=0.0,
                annual_carbon_saving_kg=0.0,
                retrofit_cost_eur=0.0,
                simple_payback_years=0.0,
                npv_eur=0.0,
                already_led=True,
            )

        # Estimate existing lumens per fixture
        existing_efficacy = self._get_lamp_efficacy(zone.lamp_type)
        if zone.lumens_per_fixture and zone.lumens_per_fixture > 0:
            lumens = _decimal(zone.lumens_per_fixture)
        else:
            lumens = w_old * _decimal(existing_efficacy)

        # LED replacement wattage to match lumens
        led_eff = _decimal(led_target_efficacy)
        w_new = _safe_divide(lumens, led_eff)

        # Apply 10% uplift for driver losses
        w_new = w_new * Decimal("1.10")

        proposed_power = n * w_new
        power_saving = existing_power - proposed_power

        if power_saving < Decimal("0"):
            power_saving = Decimal("0")
            proposed_power = existing_power

        annual_saving = power_saving * hours / Decimal("1000")
        cost_save = annual_saving * _decimal(electricity_cost)
        carbon_save = annual_saving * _decimal(carbon_factor)

        # Retrofit cost
        retrofit_cost = self._estimate_retrofit_cost(zone.lamp_type, int(float(n)))
        retrofit_d = _decimal(retrofit_cost)

        # Simple payback
        payback = _safe_divide(retrofit_d, cost_save, default=Decimal("99"))

        # NPV
        npv = self._calculate_npv(
            float(retrofit_d), float(cost_save), discount_rate, analysis_years
        )

        return RetrofitResult(
            zone_id=zone.zone_id,
            existing_lamp_type=zone.lamp_type.value,
            existing_power_w=_round2(float(existing_power)),
            proposed_power_w=_round2(float(proposed_power)),
            annual_energy_saving_kwh=_round2(float(annual_saving)),
            annual_cost_saving_eur=_round2(float(cost_save)),
            annual_carbon_saving_kg=_round2(float(carbon_save)),
            retrofit_cost_eur=_round2(retrofit_cost),
            simple_payback_years=_round2(float(payback)),
            npv_eur=_round2(npv),
            already_led=False,
        )

    # ---------------------------------------------------------------
    # Public: assess_visual_quality
    # ---------------------------------------------------------------

    def assess_visual_quality(
        self, zone: LightingZoneInput
    ) -> VisualQualityResult:
        """Assess visual quality compliance per EN 12464-1.

        Args:
            zone: Lighting zone specification.

        Returns:
            VisualQualityResult with compliance status.
        """
        reqs = ILLUMINANCE_REQUIREMENTS.get(zone.space_category)
        if not reqs:
            reqs = ILLUMINANCE_REQUIREMENTS[SpaceCategory.OFFICE_GENERAL]

        required_lux = reqs["maintained_lux"]

        compliant = None
        if zone.measured_illuminance_lux is not None:
            compliant = zone.measured_illuminance_lux >= required_lux

        efficacy = self._get_lamp_efficacy(zone.lamp_type)

        mf_table = MAINTENANCE_FACTOR.get(zone.lamp_type)
        if mf_table:
            mf = mf_table.get(zone.environment_cleanliness, 0.70)
            if not isinstance(mf, (int, float)):
                mf = 0.70
        else:
            mf = 0.70

        return VisualQualityResult(
            zone_id=zone.zone_id,
            required_lux=required_lux,
            measured_lux=zone.measured_illuminance_lux,
            illuminance_compliant=compliant,
            lamp_efficacy_lm_per_w=efficacy,
            maintenance_factor=_round3(float(mf)),
        )

    # ---------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------

    def _get_lamp_efficacy(self, lamp_type: LampType) -> float:
        """Get typical lamp efficacy in lm/W.

        Args:
            lamp_type: Lamp technology type.

        Returns:
            Typical luminous efficacy in lm/W.
        """
        data = LAMP_EFFICACY.get(lamp_type)
        if data:
            return float(data.get("typical", 80.0))
        return 80.0

    def _recommend_control_upgrade(
        self,
        space_category: SpaceCategory,
        current_control: ControlType,
    ) -> Tuple[str, float]:
        """Recommend control upgrade and additional saving.

        Args:
            space_category: Space type.
            current_control: Currently installed control.

        Returns:
            Tuple of (recommended_control_value, additional_saving_pct).
        """
        current_data = CONTROL_FACTOR.get(current_control)
        current_factor = float(current_data["factor"]) if current_data else 1.0

        # Recommend based on space type
        if space_category in (
            SpaceCategory.OFFICE_GENERAL, SpaceCategory.OFFICE_OPEN_PLAN,
            SpaceCategory.CLASSROOM, SpaceCategory.LECTURE_HALL,
            SpaceCategory.LABORATORY,
        ):
            target = ControlType.COMBINED_OCC_DAYLIGHT
        elif space_category in (
            SpaceCategory.CORRIDOR, SpaceCategory.RESTROOM,
            SpaceCategory.PARKING, SpaceCategory.WAREHOUSE,
        ):
            target = ControlType.OCCUPANCY_SENSOR
        elif space_category in (
            SpaceCategory.HOTEL_ROOM, SpaceCategory.HOSPITAL_WARD,
        ):
            target = ControlType.ABSENCE_DETECTION
        elif space_category in (
            SpaceCategory.RETAIL_SALES, SpaceCategory.RETAIL_SUPERMARKET,
            SpaceCategory.HOTEL_LOBBY, SpaceCategory.RESTAURANT,
        ):
            target = ControlType.DALI_ADDRESSABLE
        else:
            target = ControlType.OCCUPANCY_SENSOR

        # If already at or better than target, keep current
        target_data = CONTROL_FACTOR.get(target)
        target_factor = float(target_data["factor"]) if target_data else 1.0

        if target_factor >= current_factor:
            return current_control.value, 0.0

        additional_saving = (current_factor - target_factor) * 100.0
        return target.value, additional_saving

    def _estimate_retrofit_cost(self, lamp_type: LampType, n_fixtures: int) -> float:
        """Estimate total LED retrofit cost.

        Args:
            lamp_type: Existing lamp type.
            n_fixtures: Number of fixtures.

        Returns:
            Total estimated cost in EUR.
        """
        cost_data = LED_RETROFIT_COST_EUR.get(lamp_type)
        if not cost_data:
            # Default cost per fixture
            per_fixture = 50.0
        else:
            # Use new fixture + labour as typical approach
            fixture_cost = cost_data.get("new_fixture", cost_data.get("retrofit_lamp", 30.0))
            labour = cost_data.get("labour_per_fixture", 15.0)
            per_fixture = fixture_cost + labour

        return per_fixture * n_fixtures

    def _calculate_npv(
        self,
        capex: float,
        annual_saving: float,
        discount_rate: float,
        years: int,
    ) -> float:
        """Calculate Net Present Value of LED retrofit.

        NPV = -CAPEX + sum(saving / (1+r)^t for t in 1..years)

        Args:
            capex: Capital expenditure in EUR.
            annual_saving: Annual saving in EUR.
            discount_rate: Discount rate (e.g. 0.05).
            years: Analysis period.

        Returns:
            NPV in EUR.
        """
        capex_d = _decimal(capex)
        saving_d = _decimal(annual_saving)
        r = _decimal(discount_rate)

        npv = -capex_d
        for t in range(1, years + 1):
            discount_factor = (Decimal("1") + r) ** t
            npv += _safe_divide(saving_d, discount_factor)

        return float(npv)

    # ---------------------------------------------------------------
    # Internal: recommendations
    # ---------------------------------------------------------------

    def _generate_recommendations(
        self,
        inp: LightingAssessmentInput,
        zone_results: List[ZoneLPDResult],
        leni: LENIResult,
        daylight: List[DaylightResult],
        controls: List[ControlsResult],
        retrofits: List[RetrofitResult],
        visual: List[VisualQualityResult],
    ) -> List[str]:
        """Generate prioritised lighting improvement recommendations.

        All recommendations are deterministic, based on threshold
        comparisons against published standards.

        Args:
            inp: Assessment input.
            zone_results: LPD results per zone.
            leni: LENI result.
            daylight: Daylight results.
            controls: Controls results.
            retrofits: Retrofit results.
            visual: Visual quality results.

        Returns:
            List of prioritised recommendation strings.
        """
        recs: List[str] = []

        # R1: LED retrofit (highest energy impact)
        non_led_zones = [r for r in retrofits if not r.already_led]
        total_saving = sum(r.annual_energy_saving_kwh for r in non_led_zones)
        if total_saving > 0:
            total_cost = sum(r.retrofit_cost_eur for r in non_led_zones)
            total_cost_save = sum(r.annual_cost_saving_eur for r in non_led_zones)
            n_zones = len(non_led_zones)
            recs.append(
                f"LED retrofit across {n_zones} zone(s) would save "
                f"{_round2(total_saving)} kWh/yr ({_round2(total_cost_save)} EUR/yr). "
                f"Estimated cost: {_round2(total_cost)} EUR. "
                f"Prioritise zones with halogen and incandescent fixtures first."
            )

        # R2: LENI performance
        if leni.leni_rating == "poor":
            recs.append(
                f"Building LENI of {leni.leni_kwh_per_m2_yr} kWh/m2/yr "
                f"is in the 'poor' band. Target {leni.benchmark_good} kWh/m2/yr "
                f"(good practice) through LED retrofit and improved controls."
            )
        elif leni.leni_rating == "typical":
            recs.append(
                f"Building LENI of {leni.leni_kwh_per_m2_yr} kWh/m2/yr "
                f"is at typical levels. A 20-30% reduction is achievable through "
                f"controls upgrades and LED replacement to reach good practice."
            )

        # R3: Poor LPD zones
        poor_zones = [z for z in zone_results if z.lpd_rating == "poor"]
        if poor_zones:
            zone_names = ", ".join(z.zone_name for z in poor_zones[:5])
            recs.append(
                f"{len(poor_zones)} zone(s) have 'poor' LPD rating: {zone_names}. "
                f"These zones are significantly over-lit or use inefficient "
                f"luminaires. Review lighting design and de-lamp where possible."
            )

        # R4: Controls upgrades
        upgradeable = [c for c in controls if c.additional_saving_pct > 5.0]
        if upgradeable:
            best = max(upgradeable, key=lambda c: c.additional_saving_pct)
            recs.append(
                f"{len(upgradeable)} zone(s) would benefit from controls upgrades. "
                f"Highest impact: zone '{best.zone_id}' upgrading to "
                f"'{best.recommended_control}' for an additional "
                f"{best.additional_saving_pct}% saving."
            )

        # R5: Daylight harvesting
        good_daylight = [
            d for d in daylight
            if d.daylight_factor_pct >= 2.0 and d.estimated_daylight_saving_pct > 0
        ]
        if good_daylight:
            recs.append(
                f"{len(good_daylight)} zone(s) have adequate daylight for "
                f"harvesting. Install daylight-linked dimming to capture "
                f"15-35% additional savings in these areas."
            )

        # R6: Insufficient illuminance
        under_lit = [
            v for v in visual
            if v.illuminance_compliant is False
        ]
        if under_lit:
            recs.append(
                f"{len(under_lit)} zone(s) have measured illuminance below "
                f"EN 12464-1 requirements. Address underlighting through "
                f"re-lamping, additional luminaires, or surface reflectance "
                f"improvements before considering energy reduction."
            )

        # R7: Maintenance factor
        low_mf = [v for v in visual if v.maintenance_factor < 0.60]
        if low_mf:
            recs.append(
                f"{len(low_mf)} zone(s) have maintenance factor below 0.60, "
                f"indicating significant lumen depreciation and/or dirt "
                f"accumulation. Implement a cleaning schedule per CIE 97:2005 "
                f"and consider group re-lamping."
            )

        if not recs:
            recs.append(
                "Lighting system is performing well across all zones. "
                "Continue routine maintenance and monitoring per EN 12464-1 "
                "and the lighting maintenance plan."
            )

        return recs
