# -*- coding: utf-8 -*-
"""
BuildingBenchmarkEngine - PACK-032 Building Energy Assessment Engine 7
=======================================================================

Benchmarks building energy performance against standards, regulatory
thresholds, and peer databases.  Implements Energy Use Intensity (EUI)
calculation with weather normalisation, Display Energy Certificate
(DEC) operational rating, CRREM 1.5/2.0 C pathway compliance,
Energy Star score estimation, and peer comparison analytics.

Benchmarking Standards:
    - CIBSE TM46 (UK building energy benchmarks)
    - EN 15603 / EN ISO 52000 (EU energy performance)
    - ASHRAE BEAP (Building Energy Assessment Professional)
    - ENERGY STAR Portfolio Manager methodology
    - CRREM (Carbon Risk Real Estate Monitor) pathways
    - DEC (Display Energy Certificate) operational rating
    - NABERS (National Australian Built Environment Rating System)
    - BREEAM / LEED energy credits

Weather Normalisation:
    - Heating Degree Days (HDD) correction
    - Cooling Degree Days (CDD) correction
    - Base temperature by climate zone
    - Reference year standardisation

CRREM Compliance:
    - 1.5 C and 2.0 C decarbonisation pathways
    - Building type specific CO2 intensity targets
    - Stranding risk year identification
    - Gap-to-pathway quantification

Zero-Hallucination:
    - All calculations use deterministic Decimal arithmetic
    - No LLM involvement in any numeric calculation path
    - SHA-256 provenance hashing on every result
    - Benchmarks from CIBSE TM46, CRREM, ENERGY STAR

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-032 Building Energy Assessment
Engine:  7 of 10
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

class BuildingType(str, Enum):
    """Building types for benchmarking.

    Each type has specific EUI benchmarks, CRREM pathways,
    and end-use breakdown profiles.
    """
    OFFICE = "office"
    RETAIL = "retail"
    HOTEL = "hotel"
    HOSPITAL = "hospital"
    EDUCATION_PRIMARY = "education_primary"
    EDUCATION_SECONDARY = "education_secondary"
    EDUCATION_UNIVERSITY = "education_university"
    WAREHOUSE = "warehouse"
    DATA_CENTER = "data_center"
    RESTAURANT = "restaurant"
    LEISURE_CENTRE = "leisure_centre"
    LIBRARY = "library"
    MUSEUM = "museum"
    COURT = "court"
    PRISON = "prison"
    RESIDENTIAL_APARTMENT = "residential_apartment"

class BenchmarkStandard(str, Enum):
    """Benchmarking standards and rating systems."""
    ENERGY_STAR = "energy_star"
    CIBSE_TM46 = "cibse_tm46"
    DEC = "dec"
    CRREM = "crrem"
    NABERS = "nabers"
    LEED = "leed"
    BREEAM = "breeam"
    ASHRAE_BEAP = "ashrae_beap"

class ClimateZone(str, Enum):
    """Climate zones for weather normalisation."""
    NORTHERN_EUROPE = "northern_europe"
    CENTRAL_EUROPE = "central_europe"
    SOUTHERN_EUROPE = "southern_europe"
    MEDITERRANEAN = "mediterranean"
    OCEANIC = "oceanic"
    CONTINENTAL = "continental"

class DECRating(str, Enum):
    """Display Energy Certificate rating bands A-G."""
    A = "A"
    B = "B"
    C = "C"
    D = "D"
    E = "E"
    F = "F"
    G = "G"

class CRREMScenario(str, Enum):
    """CRREM pathway scenarios."""
    SCENARIO_1_5C = "1.5C"
    SCENARIO_2_0C = "2.0C"

# ---------------------------------------------------------------------------
# Constants -- EUI Benchmarks (kWh/m2/yr)
# ---------------------------------------------------------------------------

# Energy Use Intensity benchmarks by building type, climate zone, and tier.
# Sources: CIBSE TM46:2008, CIBSE Guide F:2012, EN 15603, ASHRAE BEAP.
EUI_BENCHMARKS: Dict[str, Dict[str, Dict[str, float]]] = {
    BuildingType.OFFICE: {
        ClimateZone.NORTHERN_EUROPE: {
            "best_practice": 85, "good": 120, "typical": 170, "poor": 250,
        },
        ClimateZone.CENTRAL_EUROPE: {
            "best_practice": 90, "good": 130, "typical": 180, "poor": 260,
        },
        ClimateZone.SOUTHERN_EUROPE: {
            "best_practice": 100, "good": 140, "typical": 200, "poor": 300,
        },
        ClimateZone.MEDITERRANEAN: {
            "best_practice": 110, "good": 155, "typical": 220, "poor": 320,
        },
        ClimateZone.OCEANIC: {
            "best_practice": 85, "good": 125, "typical": 175, "poor": 255,
        },
        ClimateZone.CONTINENTAL: {
            "best_practice": 95, "good": 135, "typical": 190, "poor": 280,
        },
    },
    BuildingType.RETAIL: {
        ClimateZone.NORTHERN_EUROPE: {
            "best_practice": 120, "good": 180, "typical": 260, "poor": 400,
        },
        ClimateZone.CENTRAL_EUROPE: {
            "best_practice": 130, "good": 195, "typical": 280, "poor": 420,
        },
        ClimateZone.SOUTHERN_EUROPE: {
            "best_practice": 150, "good": 220, "typical": 320, "poor": 480,
        },
        ClimateZone.MEDITERRANEAN: {
            "best_practice": 160, "good": 240, "typical": 350, "poor": 520,
        },
        ClimateZone.OCEANIC: {
            "best_practice": 125, "good": 185, "typical": 270, "poor": 410,
        },
        ClimateZone.CONTINENTAL: {
            "best_practice": 135, "good": 200, "typical": 290, "poor": 430,
        },
    },
    BuildingType.HOTEL: {
        ClimateZone.NORTHERN_EUROPE: {
            "best_practice": 150, "good": 220, "typical": 320, "poor": 480,
        },
        ClimateZone.CENTRAL_EUROPE: {
            "best_practice": 160, "good": 240, "typical": 340, "poor": 500,
        },
        ClimateZone.SOUTHERN_EUROPE: {
            "best_practice": 180, "good": 270, "typical": 380, "poor": 560,
        },
        ClimateZone.MEDITERRANEAN: {
            "best_practice": 200, "good": 290, "typical": 410, "poor": 600,
        },
        ClimateZone.OCEANIC: {
            "best_practice": 155, "good": 230, "typical": 330, "poor": 490,
        },
        ClimateZone.CONTINENTAL: {
            "best_practice": 165, "good": 250, "typical": 350, "poor": 510,
        },
    },
    BuildingType.HOSPITAL: {
        ClimateZone.NORTHERN_EUROPE: {
            "best_practice": 250, "good": 340, "typical": 450, "poor": 650,
        },
        ClimateZone.CENTRAL_EUROPE: {
            "best_practice": 270, "good": 360, "typical": 480, "poor": 680,
        },
        ClimateZone.SOUTHERN_EUROPE: {
            "best_practice": 290, "good": 400, "typical": 530, "poor": 750,
        },
        ClimateZone.MEDITERRANEAN: {
            "best_practice": 310, "good": 420, "typical": 560, "poor": 800,
        },
        ClimateZone.OCEANIC: {
            "best_practice": 260, "good": 350, "typical": 460, "poor": 660,
        },
        ClimateZone.CONTINENTAL: {
            "best_practice": 275, "good": 370, "typical": 490, "poor": 700,
        },
    },
    BuildingType.EDUCATION_PRIMARY: {
        ClimateZone.NORTHERN_EUROPE: {
            "best_practice": 70, "good": 100, "typical": 145, "poor": 210,
        },
        ClimateZone.CENTRAL_EUROPE: {
            "best_practice": 75, "good": 110, "typical": 155, "poor": 225,
        },
        ClimateZone.SOUTHERN_EUROPE: {
            "best_practice": 80, "good": 115, "typical": 165, "poor": 245,
        },
        ClimateZone.MEDITERRANEAN: {
            "best_practice": 85, "good": 120, "typical": 175, "poor": 260,
        },
        ClimateZone.OCEANIC: {
            "best_practice": 72, "good": 105, "typical": 150, "poor": 215,
        },
        ClimateZone.CONTINENTAL: {
            "best_practice": 78, "good": 112, "typical": 160, "poor": 235,
        },
    },
    BuildingType.EDUCATION_SECONDARY: {
        ClimateZone.NORTHERN_EUROPE: {
            "best_practice": 80, "good": 115, "typical": 165, "poor": 240,
        },
        ClimateZone.CENTRAL_EUROPE: {
            "best_practice": 85, "good": 125, "typical": 175, "poor": 255,
        },
        ClimateZone.SOUTHERN_EUROPE: {
            "best_practice": 95, "good": 135, "typical": 195, "poor": 285,
        },
        ClimateZone.MEDITERRANEAN: {
            "best_practice": 100, "good": 145, "typical": 210, "poor": 310,
        },
        ClimateZone.OCEANIC: {
            "best_practice": 82, "good": 120, "typical": 170, "poor": 248,
        },
        ClimateZone.CONTINENTAL: {
            "best_practice": 88, "good": 128, "typical": 182, "poor": 265,
        },
    },
    BuildingType.EDUCATION_UNIVERSITY: {
        ClimateZone.NORTHERN_EUROPE: {
            "best_practice": 110, "good": 160, "typical": 230, "poor": 340,
        },
        ClimateZone.CENTRAL_EUROPE: {
            "best_practice": 120, "good": 175, "typical": 250, "poor": 370,
        },
        ClimateZone.SOUTHERN_EUROPE: {
            "best_practice": 130, "good": 190, "typical": 270, "poor": 400,
        },
        ClimateZone.MEDITERRANEAN: {
            "best_practice": 140, "good": 200, "typical": 290, "poor": 430,
        },
        ClimateZone.OCEANIC: {
            "best_practice": 115, "good": 168, "typical": 240, "poor": 355,
        },
        ClimateZone.CONTINENTAL: {
            "best_practice": 125, "good": 180, "typical": 260, "poor": 385,
        },
    },
    BuildingType.WAREHOUSE: {
        ClimateZone.NORTHERN_EUROPE: {
            "best_practice": 40, "good": 65, "typical": 100, "poor": 160,
        },
        ClimateZone.CENTRAL_EUROPE: {
            "best_practice": 45, "good": 70, "typical": 110, "poor": 175,
        },
        ClimateZone.SOUTHERN_EUROPE: {
            "best_practice": 50, "good": 80, "typical": 120, "poor": 190,
        },
        ClimateZone.MEDITERRANEAN: {
            "best_practice": 55, "good": 85, "typical": 130, "poor": 210,
        },
        ClimateZone.OCEANIC: {
            "best_practice": 42, "good": 68, "typical": 105, "poor": 168,
        },
        ClimateZone.CONTINENTAL: {
            "best_practice": 48, "good": 75, "typical": 115, "poor": 180,
        },
    },
    BuildingType.DATA_CENTER: {
        ClimateZone.NORTHERN_EUROPE: {
            "best_practice": 600, "good": 900, "typical": 1300, "poor": 2000,
        },
        ClimateZone.CENTRAL_EUROPE: {
            "best_practice": 650, "good": 950, "typical": 1400, "poor": 2100,
        },
        ClimateZone.SOUTHERN_EUROPE: {
            "best_practice": 700, "good": 1050, "typical": 1500, "poor": 2300,
        },
        ClimateZone.MEDITERRANEAN: {
            "best_practice": 750, "good": 1100, "typical": 1600, "poor": 2500,
        },
        ClimateZone.OCEANIC: {
            "best_practice": 620, "good": 920, "typical": 1350, "poor": 2050,
        },
        ClimateZone.CONTINENTAL: {
            "best_practice": 670, "good": 980, "typical": 1420, "poor": 2150,
        },
    },
    BuildingType.RESTAURANT: {
        ClimateZone.NORTHERN_EUROPE: {
            "best_practice": 200, "good": 300, "typical": 430, "poor": 650,
        },
        ClimateZone.CENTRAL_EUROPE: {
            "best_practice": 220, "good": 320, "typical": 460, "poor": 690,
        },
        ClimateZone.SOUTHERN_EUROPE: {
            "best_practice": 240, "good": 350, "typical": 500, "poor": 750,
        },
        ClimateZone.MEDITERRANEAN: {
            "best_practice": 260, "good": 380, "typical": 540, "poor": 810,
        },
        ClimateZone.OCEANIC: {
            "best_practice": 210, "good": 310, "typical": 445, "poor": 670,
        },
        ClimateZone.CONTINENTAL: {
            "best_practice": 230, "good": 340, "typical": 480, "poor": 720,
        },
    },
    BuildingType.LEISURE_CENTRE: {
        ClimateZone.NORTHERN_EUROPE: {
            "best_practice": 180, "good": 270, "typical": 400, "poor": 600,
        },
        ClimateZone.CENTRAL_EUROPE: {
            "best_practice": 195, "good": 290, "typical": 420, "poor": 630,
        },
        ClimateZone.SOUTHERN_EUROPE: {
            "best_practice": 210, "good": 315, "typical": 460, "poor": 690,
        },
        ClimateZone.MEDITERRANEAN: {
            "best_practice": 230, "good": 340, "typical": 490, "poor": 740,
        },
        ClimateZone.OCEANIC: {
            "best_practice": 185, "good": 280, "typical": 410, "poor": 615,
        },
        ClimateZone.CONTINENTAL: {
            "best_practice": 200, "good": 300, "typical": 440, "poor": 660,
        },
    },
    BuildingType.RESIDENTIAL_APARTMENT: {
        ClimateZone.NORTHERN_EUROPE: {
            "best_practice": 60, "good": 90, "typical": 140, "poor": 220,
        },
        ClimateZone.CENTRAL_EUROPE: {
            "best_practice": 65, "good": 100, "typical": 150, "poor": 240,
        },
        ClimateZone.SOUTHERN_EUROPE: {
            "best_practice": 55, "good": 85, "typical": 130, "poor": 210,
        },
        ClimateZone.MEDITERRANEAN: {
            "best_practice": 50, "good": 80, "typical": 120, "poor": 200,
        },
        ClimateZone.OCEANIC: {
            "best_practice": 62, "good": 95, "typical": 145, "poor": 230,
        },
        ClimateZone.CONTINENTAL: {
            "best_practice": 70, "good": 105, "typical": 160, "poor": 255,
        },
    },
}

# ---------------------------------------------------------------------------
# Constants -- DEC Rating Thresholds
# ---------------------------------------------------------------------------

# DEC operational rating bands.
# OR = (actual energy / benchmark) * 100.
# Sources: DCLG, EPBD (recast) 2024.
DEC_RATING_THRESHOLDS: Dict[str, Dict[str, int]] = {
    DECRating.A: {"lower": 0, "upper": 25},
    DECRating.B: {"lower": 26, "upper": 50},
    DECRating.C: {"lower": 51, "upper": 75},
    DECRating.D: {"lower": 76, "upper": 100},
    DECRating.E: {"lower": 101, "upper": 125},
    DECRating.F: {"lower": 126, "upper": 150},
    DECRating.G: {"lower": 151, "upper": 9999},
}

# ---------------------------------------------------------------------------
# Constants -- CRREM Pathways (kgCO2/m2/yr)
# ---------------------------------------------------------------------------

# Decarbonisation pathway targets by building type for 1.5C and 2.0C.
# Key years: 2020, 2025, 2030, 2035, 2040, 2045, 2050.
# Sources: CRREM Global Pathways 2022, CRREM EU Downscaling 2023.
CRREM_PATHWAYS: Dict[str, Dict[str, Dict[int, float]]] = {
    BuildingType.OFFICE: {
        CRREMScenario.SCENARIO_1_5C: {
            2020: 42.0, 2025: 32.0, 2030: 22.0, 2035: 14.0,
            2040: 8.0, 2045: 4.0, 2050: 0.0,
        },
        CRREMScenario.SCENARIO_2_0C: {
            2020: 42.0, 2025: 35.0, 2030: 27.0, 2035: 20.0,
            2040: 13.0, 2045: 7.0, 2050: 2.0,
        },
    },
    BuildingType.RETAIL: {
        CRREMScenario.SCENARIO_1_5C: {
            2020: 55.0, 2025: 42.0, 2030: 30.0, 2035: 19.0,
            2040: 10.0, 2045: 5.0, 2050: 0.0,
        },
        CRREMScenario.SCENARIO_2_0C: {
            2020: 55.0, 2025: 46.0, 2030: 36.0, 2035: 27.0,
            2040: 18.0, 2045: 10.0, 2050: 3.0,
        },
    },
    BuildingType.HOTEL: {
        CRREMScenario.SCENARIO_1_5C: {
            2020: 65.0, 2025: 50.0, 2030: 35.0, 2035: 22.0,
            2040: 12.0, 2045: 5.0, 2050: 0.0,
        },
        CRREMScenario.SCENARIO_2_0C: {
            2020: 65.0, 2025: 54.0, 2030: 42.0, 2035: 31.0,
            2040: 21.0, 2045: 12.0, 2050: 4.0,
        },
    },
    BuildingType.HOSPITAL: {
        CRREMScenario.SCENARIO_1_5C: {
            2020: 90.0, 2025: 70.0, 2030: 50.0, 2035: 32.0,
            2040: 18.0, 2045: 8.0, 2050: 0.0,
        },
        CRREMScenario.SCENARIO_2_0C: {
            2020: 90.0, 2025: 76.0, 2030: 60.0, 2035: 44.0,
            2040: 30.0, 2045: 16.0, 2050: 5.0,
        },
    },
    BuildingType.RESIDENTIAL_APARTMENT: {
        CRREMScenario.SCENARIO_1_5C: {
            2020: 35.0, 2025: 27.0, 2030: 18.0, 2035: 11.0,
            2040: 6.0, 2045: 3.0, 2050: 0.0,
        },
        CRREMScenario.SCENARIO_2_0C: {
            2020: 35.0, 2025: 29.0, 2030: 23.0, 2035: 16.0,
            2040: 10.0, 2045: 5.0, 2050: 1.5,
        },
    },
    BuildingType.WAREHOUSE: {
        CRREMScenario.SCENARIO_1_5C: {
            2020: 20.0, 2025: 15.0, 2030: 10.0, 2035: 6.0,
            2040: 3.0, 2045: 1.5, 2050: 0.0,
        },
        CRREMScenario.SCENARIO_2_0C: {
            2020: 20.0, 2025: 17.0, 2030: 13.0, 2035: 9.0,
            2040: 6.0, 2045: 3.0, 2050: 1.0,
        },
    },
    BuildingType.EDUCATION_PRIMARY: {
        CRREMScenario.SCENARIO_1_5C: {
            2020: 30.0, 2025: 23.0, 2030: 16.0, 2035: 10.0,
            2040: 5.0, 2045: 2.5, 2050: 0.0,
        },
        CRREMScenario.SCENARIO_2_0C: {
            2020: 30.0, 2025: 25.0, 2030: 20.0, 2035: 14.0,
            2040: 9.0, 2045: 5.0, 2050: 1.5,
        },
    },
}

# ---------------------------------------------------------------------------
# Constants -- Typical End Use Split (%)
# ---------------------------------------------------------------------------

# Typical end-use energy breakdown by building type.
# Sources: CIBSE TM46, BEIS NEED/BEES data, Carbon Trust benchmarks.
TYPICAL_END_USE_SPLIT: Dict[str, Dict[str, float]] = {
    BuildingType.OFFICE: {
        "heating_pct": 40.0, "cooling_pct": 10.0, "lighting_pct": 20.0,
        "plug_loads_pct": 18.0, "dhw_pct": 5.0, "fans_pumps_pct": 7.0,
        "source": "CIBSE TM46, typical UK office",
    },
    BuildingType.RETAIL: {
        "heating_pct": 30.0, "cooling_pct": 15.0, "lighting_pct": 25.0,
        "plug_loads_pct": 15.0, "dhw_pct": 3.0, "fans_pumps_pct": 12.0,
        "source": "CIBSE TM46, typical retail",
    },
    BuildingType.HOTEL: {
        "heating_pct": 35.0, "cooling_pct": 10.0, "lighting_pct": 15.0,
        "plug_loads_pct": 10.0, "dhw_pct": 20.0, "fans_pumps_pct": 10.0,
        "source": "CIBSE TM46, hotel",
    },
    BuildingType.HOSPITAL: {
        "heating_pct": 38.0, "cooling_pct": 8.0, "lighting_pct": 15.0,
        "plug_loads_pct": 20.0, "dhw_pct": 12.0, "fans_pumps_pct": 7.0,
        "source": "HTM 07-02, CIBSE TM46",
    },
    BuildingType.EDUCATION_PRIMARY: {
        "heating_pct": 55.0, "cooling_pct": 2.0, "lighting_pct": 15.0,
        "plug_loads_pct": 12.0, "dhw_pct": 10.0, "fans_pumps_pct": 6.0,
        "source": "BB87/BB90, CIBSE TM46",
    },
    BuildingType.WAREHOUSE: {
        "heating_pct": 50.0, "cooling_pct": 5.0, "lighting_pct": 20.0,
        "plug_loads_pct": 10.0, "dhw_pct": 3.0, "fans_pumps_pct": 12.0,
        "source": "CIBSE TM46",
    },
    BuildingType.DATA_CENTER: {
        "heating_pct": 1.0, "cooling_pct": 38.0, "lighting_pct": 3.0,
        "plug_loads_pct": 50.0, "dhw_pct": 0.5, "fans_pumps_pct": 7.5,
        "source": "Uptime Institute, EU Code of Conduct",
    },
    BuildingType.RESTAURANT: {
        "heating_pct": 25.0, "cooling_pct": 10.0, "lighting_pct": 12.0,
        "plug_loads_pct": 8.0, "dhw_pct": 15.0, "fans_pumps_pct": 30.0,
        "source": "CIBSE TM46, includes kitchen extract",
    },
    BuildingType.RESIDENTIAL_APARTMENT: {
        "heating_pct": 55.0, "cooling_pct": 5.0, "lighting_pct": 10.0,
        "plug_loads_pct": 15.0, "dhw_pct": 12.0, "fans_pumps_pct": 3.0,
        "source": "BEIS NEED, SAP methodology",
    },
}

# ---------------------------------------------------------------------------
# Constants -- Weather Correction (Reference HDD)
# ---------------------------------------------------------------------------

# Reference Heating Degree Days by climate zone and base temperature.
# Sources: ASHRAE Fundamentals, CIBSE Guide A, Eurostat degree-day data.
WEATHER_REFERENCE_HDD: Dict[str, Dict[str, float]] = {
    ClimateZone.NORTHERN_EUROPE: {
        "hdd_ref": 4200, "cdd_ref": 80, "base_heating_c": 15.5,
        "base_cooling_c": 18.0, "source": "Eurostat, Nordic avg"
    },
    ClimateZone.CENTRAL_EUROPE: {
        "hdd_ref": 3200, "cdd_ref": 200, "base_heating_c": 15.5,
        "base_cooling_c": 18.0, "source": "Eurostat, DACH avg"
    },
    ClimateZone.SOUTHERN_EUROPE: {
        "hdd_ref": 1800, "cdd_ref": 600, "base_heating_c": 15.0,
        "base_cooling_c": 21.0, "source": "Eurostat, IT/FR south"
    },
    ClimateZone.MEDITERRANEAN: {
        "hdd_ref": 1200, "cdd_ref": 900, "base_heating_c": 15.0,
        "base_cooling_c": 22.0, "source": "Eurostat, ES/GR"
    },
    ClimateZone.OCEANIC: {
        "hdd_ref": 2800, "cdd_ref": 100, "base_heating_c": 15.5,
        "base_cooling_c": 18.0, "source": "Eurostat, UK/IE"
    },
    ClimateZone.CONTINENTAL: {
        "hdd_ref": 3600, "cdd_ref": 250, "base_heating_c": 15.0,
        "base_cooling_c": 18.0, "source": "Eurostat, PL/CZ/HU"
    },
}

# ---------------------------------------------------------------------------
# Constants -- Occupancy Correction Factors
# ---------------------------------------------------------------------------

# Occupancy normalisation factors for non-standard occupancy levels.
# Factor = 1.0 at standard occupancy; >1 means higher consumption expected.
# Sources: ASHRAE BEAP, CIBSE Guide F.
OCCUPANCY_CORRECTION: Dict[str, Dict[str, float]] = {
    BuildingType.OFFICE: {
        "low_occupancy_factor": 0.80,  # < 8 m2/person
        "standard_occupancy_factor": 1.00,  # 8-12 m2/person
        "high_occupancy_factor": 1.15,  # > 12 m2/person
        "standard_density_m2_per_person": 10.0,
    },
    BuildingType.RETAIL: {
        "low_occupancy_factor": 0.85,
        "standard_occupancy_factor": 1.00,
        "high_occupancy_factor": 1.10,
        "standard_density_m2_per_person": 5.0,
    },
    BuildingType.HOTEL: {
        "low_occupancy_factor": 0.75,
        "standard_occupancy_factor": 1.00,
        "high_occupancy_factor": 1.20,
        "standard_density_m2_per_person": 35.0,
    },
    BuildingType.HOSPITAL: {
        "low_occupancy_factor": 0.90,
        "standard_occupancy_factor": 1.00,
        "high_occupancy_factor": 1.10,
        "standard_density_m2_per_person": 20.0,
    },
    BuildingType.EDUCATION_PRIMARY: {
        "low_occupancy_factor": 0.80,
        "standard_occupancy_factor": 1.00,
        "high_occupancy_factor": 1.15,
        "standard_density_m2_per_person": 5.0,
    },
}

# ---------------------------------------------------------------------------
# Constants -- Carbon Factors for DEC / CRREM
# ---------------------------------------------------------------------------

CARBON_FACTORS_KG_PER_KWH: Dict[str, float] = {
    "electricity_uk": 0.207,
    "electricity_de": 0.366,
    "electricity_fr": 0.052,
    "electricity_eu_avg": 0.233,
    "natural_gas": 0.203,
    "oil": 0.274,
    "district_heating": 0.150,
    "biomass": 0.015,
    "lpg": 0.214,
}

# ---------------------------------------------------------------------------
# Pydantic Input Models
# ---------------------------------------------------------------------------

class EnergyConsumptionInput(BaseModel):
    """Annual energy consumption breakdown."""

    electricity_kwh: float = Field(
        ..., ge=0, description="Annual electricity consumption in kWh"
    )
    gas_kwh: float = Field(
        0.0, ge=0, description="Annual gas consumption in kWh"
    )
    oil_kwh: float = Field(
        0.0, ge=0, description="Annual oil consumption in kWh"
    )
    district_heating_kwh: float = Field(
        0.0, ge=0, description="Annual district heating in kWh"
    )
    district_cooling_kwh: float = Field(
        0.0, ge=0, description="Annual district cooling in kWh"
    )
    biomass_kwh: float = Field(
        0.0, ge=0, description="Annual biomass consumption in kWh"
    )
    lpg_kwh: float = Field(
        0.0, ge=0, description="Annual LPG consumption in kWh"
    )
    on_site_renewable_kwh: float = Field(
        0.0, ge=0, description="On-site renewable generation in kWh"
    )

class WeatherDataInput(BaseModel):
    """Actual weather data for the assessment period."""

    actual_hdd: float = Field(
        ..., gt=0, description="Actual Heating Degree Days for the period"
    )
    actual_cdd: float = Field(
        0.0, ge=0, description="Actual Cooling Degree Days for the period"
    )

class PeerBuildingInput(BaseModel):
    """Peer building data for comparison."""

    building_name: str = Field(..., description="Peer building name")
    eui_kwh_per_m2: float = Field(
        ..., gt=0, description="Peer building EUI in kWh/m2/yr"
    )
    carbon_intensity_kg_per_m2: Optional[float] = Field(
        None, description="Peer building carbon intensity kgCO2/m2/yr"
    )

class BenchmarkInput(BaseModel):
    """Top-level benchmarking assessment input."""

    building_id: str = Field(..., description="Building identifier")
    building_name: str = Field("", description="Building name")
    building_type: BuildingType = Field(
        ..., description="Building type for benchmark selection"
    )
    climate_zone: ClimateZone = Field(
        ClimateZone.CENTRAL_EUROPE, description="Climate zone"
    )
    gross_internal_area_m2: float = Field(
        ..., gt=0, description="Gross Internal Area in m2"
    )
    net_lettable_area_m2: Optional[float] = Field(
        None, description="Net Lettable Area in m2 (for NABERS/ENERGY STAR)"
    )
    occupant_count: Optional[int] = Field(
        None, gt=0, description="Number of occupants"
    )
    operating_hours_per_week: float = Field(
        45.0, gt=0, description="Weekly operating hours"
    )
    assessment_year: int = Field(
        2025, description="Year of assessment for CRREM pathway"
    )
    energy: EnergyConsumptionInput = Field(
        ..., description="Annual energy consumption"
    )
    weather: Optional[WeatherDataInput] = Field(
        None, description="Actual weather data for normalisation"
    )
    peers: List[PeerBuildingInput] = Field(
        default_factory=list, description="Peer buildings for comparison"
    )
    carbon_factor_electricity: float = Field(
        0.233, description="Electricity carbon factor kgCO2e/kWh"
    )
    carbon_factor_gas: float = Field(
        0.203, description="Gas carbon factor kgCO2e/kWh"
    )
    electricity_cost_eur_per_kwh: float = Field(
        0.30, description="Electricity cost EUR/kWh"
    )
    gas_cost_eur_per_kwh: float = Field(
        0.08, description="Gas cost EUR/kWh"
    )

# ---------------------------------------------------------------------------
# Pydantic Result Models
# ---------------------------------------------------------------------------

class EUIResult(BaseModel):
    """Energy Use Intensity calculation result."""

    total_energy_kwh: float = Field(
        ..., description="Total annual energy consumption in kWh"
    )
    eui_kwh_per_m2: float = Field(
        ..., description="Energy Use Intensity in kWh/m2/yr"
    )
    eui_normalised_kwh_per_m2: Optional[float] = Field(
        None, description="Weather-normalised EUI in kWh/m2/yr"
    )
    electricity_eui: float = Field(
        ..., description="Electricity EUI in kWh/m2/yr"
    )
    thermal_eui: float = Field(
        ..., description="Thermal (heating) EUI in kWh/m2/yr"
    )
    benchmark_best_practice: float = Field(
        ..., description="Best practice EUI benchmark"
    )
    benchmark_good: float = Field(
        ..., description="Good practice EUI benchmark"
    )
    benchmark_typical: float = Field(
        ..., description="Typical EUI benchmark"
    )
    performance_tier: str = Field(
        ..., description="Performance tier vs benchmarks"
    )

class DECResult(BaseModel):
    """Display Energy Certificate rating result."""

    operational_rating: float = Field(
        ..., description="DEC operational rating (OR)"
    )
    dec_band: str = Field(..., description="DEC band (A-G)")
    benchmark_eui: float = Field(
        ..., description="DEC benchmark EUI used"
    )
    improvement_to_next_band: Optional[float] = Field(
        None, description="EUI reduction needed to reach next band (kWh/m2)"
    )

class CRREMResult(BaseModel):
    """CRREM pathway compliance result."""

    scenario: str = Field(..., description="CRREM scenario (1.5C or 2.0C)")
    current_carbon_intensity_kg_m2: float = Field(
        ..., description="Current carbon intensity in kgCO2/m2/yr"
    )
    pathway_target_kg_m2: float = Field(
        ..., description="CRREM target for assessment year in kgCO2/m2/yr"
    )
    gap_kg_m2: float = Field(
        ..., description="Gap to CRREM target (positive = exceeds)"
    )
    compliant: bool = Field(
        ..., description="Whether building meets CRREM pathway"
    )
    estimated_stranding_year: Optional[int] = Field(
        None, description="Estimated year building becomes stranded"
    )
    reduction_needed_pct: float = Field(
        ..., description="Percentage reduction needed to meet target"
    )

class EnergyStarResult(BaseModel):
    """Estimated ENERGY STAR score."""

    estimated_score: int = Field(
        ..., ge=1, le=100, description="Estimated ENERGY STAR score (1-100)"
    )
    percentile_rank: str = Field(
        ..., description="Percentile description"
    )
    certification_eligible: bool = Field(
        ..., description="Whether score >= 75 (certification threshold)"
    )

class PeerComparisonResult(BaseModel):
    """Peer comparison result."""

    peer_name: str = Field(..., description="Peer building name")
    peer_eui: float = Field(..., description="Peer EUI")
    difference_kwh_m2: float = Field(
        ..., description="EUI difference (positive = assessed is higher)"
    )
    difference_pct: float = Field(
        ..., description="Percentage difference"
    )

class EndUseSplitResult(BaseModel):
    """End use energy breakdown."""

    heating_kwh: float = Field(0, description="Estimated heating energy")
    cooling_kwh: float = Field(0, description="Estimated cooling energy")
    lighting_kwh: float = Field(0, description="Estimated lighting energy")
    plug_loads_kwh: float = Field(0, description="Estimated plug loads")
    dhw_kwh: float = Field(0, description="Estimated DHW energy")
    fans_pumps_kwh: float = Field(0, description="Estimated fans/pumps energy")

class GapAnalysisResult(BaseModel):
    """Gap to best practice analysis."""

    gap_to_best_practice_kwh_m2: float = Field(
        ..., description="Gap to best practice in kWh/m2/yr"
    )
    gap_to_good_kwh_m2: float = Field(
        ..., description="Gap to good practice in kWh/m2/yr"
    )
    potential_saving_kwh: float = Field(
        ..., description="Total potential saving to best practice in kWh/yr"
    )
    potential_saving_eur: float = Field(
        ..., description="Total potential cost saving in EUR/yr"
    )
    potential_carbon_saving_kg: float = Field(
        ..., description="Total potential carbon saving in kgCO2e/yr"
    )

class BuildingBenchmarkResult(BaseModel):
    """Complete benchmarking assessment result."""

    assessment_id: str = Field(..., description="Assessment ID")
    building_id: str = Field(..., description="Building ID")
    building_name: str = Field(..., description="Building name")
    engine_version: str = Field(..., description="Engine version")
    eui: EUIResult = Field(..., description="EUI calculation")
    dec: DECResult = Field(..., description="DEC rating")
    crrem_1_5c: Optional[CRREMResult] = Field(
        None, description="CRREM 1.5C compliance"
    )
    crrem_2_0c: Optional[CRREMResult] = Field(
        None, description="CRREM 2.0C compliance"
    )
    energy_star: Optional[EnergyStarResult] = Field(
        None, description="Energy Star score estimate"
    )
    peer_comparisons: List[PeerComparisonResult] = Field(
        default_factory=list, description="Peer comparisons"
    )
    end_use_split: EndUseSplitResult = Field(
        ..., description="End use breakdown"
    )
    gap_analysis: GapAnalysisResult = Field(
        ..., description="Gap to best practice"
    )
    total_annual_energy_kwh: float = Field(
        ..., description="Total annual energy"
    )
    total_annual_cost_eur: float = Field(
        ..., description="Total annual energy cost"
    )
    total_annual_carbon_kg: float = Field(
        ..., description="Total annual carbon"
    )
    carbon_intensity_kg_per_m2: float = Field(
        ..., description="Carbon intensity kgCO2/m2/yr"
    )
    recommendations: List[str] = Field(
        default_factory=list, description="Recommendations"
    )
    calculated_at: str = Field(..., description="ISO UTC timestamp")
    processing_time_ms: float = Field(..., description="Processing time ms")
    provenance_hash: str = Field(..., description="SHA-256 hash")

# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class BuildingBenchmarkEngine:
    """Building energy benchmarking engine.

    Benchmarks building energy performance against standards (CIBSE TM46,
    CRREM, DEC, ENERGY STAR), applies weather normalisation, and
    identifies gaps to best practice.

    Zero-Hallucination Guarantees:
        - All calculations use deterministic Decimal arithmetic
        - No LLM involvement in any numeric calculation path
        - SHA-256 provenance hashing on every result
        - Benchmarks from CIBSE TM46, CRREM, ENERGY STAR

    Example::

        engine = BuildingBenchmarkEngine()
        result = engine.benchmark(benchmark_input)
    """

    # ---------------------------------------------------------------
    # Public: benchmark (main entry point)
    # ---------------------------------------------------------------

    def benchmark(self, inp: BenchmarkInput) -> BuildingBenchmarkResult:
        """Run complete benchmarking assessment.

        Args:
            inp: Benchmarking assessment input.

        Returns:
            BuildingBenchmarkResult with full provenance.
        """
        t_start = time.perf_counter()

        # Step 1: Calculate EUI
        eui = self.calculate_eui(inp)

        # Step 2: Weather normalisation
        if inp.weather:
            eui_norm = self.normalize_weather(
                eui_kwh_m2=eui.eui_kwh_per_m2,
                actual_hdd=inp.weather.actual_hdd,
                actual_cdd=inp.weather.actual_cdd,
                climate_zone=inp.climate_zone,
                building_type=inp.building_type,
            )
            eui.eui_normalised_kwh_per_m2 = eui_norm

        # Step 3: DEC rating
        dec = self.calculate_dec_rating(
            eui_kwh_m2=eui.eui_normalised_kwh_per_m2 or eui.eui_kwh_per_m2,
            building_type=inp.building_type,
            climate_zone=inp.climate_zone,
        )

        # Step 4: CRREM compliance
        crrem_15 = self.check_crrem_compliance(
            inp, CRREMScenario.SCENARIO_1_5C
        )
        crrem_20 = self.check_crrem_compliance(
            inp, CRREMScenario.SCENARIO_2_0C
        )

        # Step 5: Energy Star estimate
        es = self.estimate_energy_star_score(
            eui_kwh_m2=eui.eui_normalised_kwh_per_m2 or eui.eui_kwh_per_m2,
            building_type=inp.building_type,
            climate_zone=inp.climate_zone,
        )

        # Step 6: Peer comparison
        peer_results = self.compare_peers(
            eui_kwh_m2=eui.eui_kwh_per_m2,
            peers=inp.peers,
        )

        # Step 7: End use split
        end_use = self._estimate_end_use_split(
            total_energy_kwh=eui.total_energy_kwh,
            building_type=inp.building_type,
        )

        # Step 8: Gap analysis
        gap = self.identify_gaps(
            eui_kwh_m2=eui.eui_normalised_kwh_per_m2 or eui.eui_kwh_per_m2,
            building_type=inp.building_type,
            climate_zone=inp.climate_zone,
            floor_area_m2=inp.gross_internal_area_m2,
            electricity_cost=inp.electricity_cost_eur_per_kwh,
            gas_cost=inp.gas_cost_eur_per_kwh,
            carbon_factor_elec=inp.carbon_factor_electricity,
            carbon_factor_gas=inp.carbon_factor_gas,
        )

        # Totals
        total_energy = _decimal(eui.total_energy_kwh)
        elec = _decimal(inp.energy.electricity_kwh)
        thermal = total_energy - elec

        elec_cost = elec * _decimal(inp.electricity_cost_eur_per_kwh)
        thermal_cost = thermal * _decimal(inp.gas_cost_eur_per_kwh)
        total_cost = elec_cost + thermal_cost

        elec_carbon = elec * _decimal(inp.carbon_factor_electricity)
        thermal_carbon = thermal * _decimal(inp.carbon_factor_gas)
        total_carbon = elec_carbon + thermal_carbon

        area = _decimal(inp.gross_internal_area_m2)
        carbon_intensity = _safe_divide(total_carbon, area)

        # Recommendations
        recommendations = self._generate_recommendations(
            inp, eui, dec, crrem_15, crrem_20, es, gap
        )

        t_end = time.perf_counter()
        processing_ms = (t_end - t_start) * 1000.0

        result = BuildingBenchmarkResult(
            assessment_id=_new_uuid(),
            building_id=inp.building_id,
            building_name=inp.building_name,
            engine_version=_MODULE_VERSION,
            eui=eui,
            dec=dec,
            crrem_1_5c=crrem_15,
            crrem_2_0c=crrem_20,
            energy_star=es,
            peer_comparisons=peer_results,
            end_use_split=end_use,
            gap_analysis=gap,
            total_annual_energy_kwh=_round2(float(total_energy)),
            total_annual_cost_eur=_round2(float(total_cost)),
            total_annual_carbon_kg=_round2(float(total_carbon)),
            carbon_intensity_kg_per_m2=_round2(float(carbon_intensity)),
            recommendations=recommendations,
            calculated_at=utcnow().isoformat(),
            processing_time_ms=_round3(processing_ms),
            provenance_hash="",
        )
        result.provenance_hash = _compute_hash(result)
        return result

    # ---------------------------------------------------------------
    # Public: calculate_eui
    # ---------------------------------------------------------------

    def calculate_eui(self, inp: BenchmarkInput) -> EUIResult:
        """Calculate Energy Use Intensity.

        EUI = E_total / A_floor  [kWh/m2/yr]

        Args:
            inp: Benchmark input.

        Returns:
            EUIResult with EUI and benchmark comparison.
        """
        e = inp.energy
        elec = _decimal(e.electricity_kwh)
        gas = _decimal(e.gas_kwh)
        oil = _decimal(e.oil_kwh)
        dh = _decimal(e.district_heating_kwh)
        dc = _decimal(e.district_cooling_kwh)
        bio = _decimal(e.biomass_kwh)
        lpg = _decimal(e.lpg_kwh)
        renew = _decimal(e.on_site_renewable_kwh)

        total = elec + gas + oil + dh + dc + bio + lpg
        # On-site renewables offset electricity
        net_total = total - renew
        if net_total < Decimal("0"):
            net_total = Decimal("0")

        area = _decimal(inp.gross_internal_area_m2)
        eui = _safe_divide(net_total, area)

        elec_eui = _safe_divide(elec, area)
        thermal = gas + oil + dh + bio + lpg
        thermal_eui = _safe_divide(thermal, area)

        # Benchmarks
        benchmarks = self._get_eui_benchmarks(inp.building_type, inp.climate_zone)
        bp = benchmarks["best_practice"]
        good = benchmarks["good"]
        typical = benchmarks["typical"]
        poor = benchmarks["poor"]

        eui_f = float(eui)
        if eui_f <= bp:
            tier = "best_practice"
        elif eui_f <= good:
            tier = "good"
        elif eui_f <= typical:
            tier = "typical"
        else:
            tier = "poor"

        return EUIResult(
            total_energy_kwh=_round2(float(net_total)),
            eui_kwh_per_m2=_round2(eui_f),
            eui_normalised_kwh_per_m2=None,
            electricity_eui=_round2(float(elec_eui)),
            thermal_eui=_round2(float(thermal_eui)),
            benchmark_best_practice=bp,
            benchmark_good=good,
            benchmark_typical=typical,
            performance_tier=tier,
        )

    # ---------------------------------------------------------------
    # Public: normalize_weather
    # ---------------------------------------------------------------

    def normalize_weather(
        self,
        eui_kwh_m2: float,
        actual_hdd: float,
        actual_cdd: float,
        climate_zone: ClimateZone,
        building_type: BuildingType,
    ) -> float:
        """Weather-normalise EUI using HDD/CDD correction.

        Formula:
            EUI_norm = EUI_elec + EUI_heat * (HDD_ref / HDD_actual)
                     + EUI_cool * (CDD_ref / CDD_actual)

        Simplified approach using heating fraction from end-use split.

        Args:
            eui_kwh_m2: Actual EUI.
            actual_hdd: Actual HDD.
            actual_cdd: Actual CDD.
            climate_zone: Climate zone for reference HDD.
            building_type: Building type for heating fraction.

        Returns:
            Weather-normalised EUI in kWh/m2/yr.
        """
        ref_data = WEATHER_REFERENCE_HDD.get(climate_zone)
        if not ref_data:
            ref_data = WEATHER_REFERENCE_HDD[ClimateZone.CENTRAL_EUROPE]

        hdd_ref = _decimal(ref_data["hdd_ref"])
        cdd_ref = _decimal(ref_data["cdd_ref"])
        hdd_act = _decimal(actual_hdd)
        cdd_act = _decimal(actual_cdd)

        # End use split
        split = TYPICAL_END_USE_SPLIT.get(building_type)
        if not split:
            split = TYPICAL_END_USE_SPLIT[BuildingType.OFFICE]

        heat_frac = _decimal(split["heating_pct"]) / Decimal("100")
        cool_frac = _decimal(split["cooling_pct"]) / Decimal("100")
        base_frac = Decimal("1") - heat_frac - cool_frac

        eui_d = _decimal(eui_kwh_m2)

        # Split EUI by end use
        heat_eui = eui_d * heat_frac
        cool_eui = eui_d * cool_frac
        base_eui = eui_d * base_frac

        # Normalise heating and cooling
        heat_norm = heat_eui * _safe_divide(hdd_ref, hdd_act, Decimal("1"))
        cool_norm = cool_eui * _safe_divide(cdd_ref, cdd_act, Decimal("1"))

        eui_norm = base_eui + heat_norm + cool_norm
        return _round2(float(eui_norm))

    # ---------------------------------------------------------------
    # Public: calculate_dec_rating
    # ---------------------------------------------------------------

    def calculate_dec_rating(
        self,
        eui_kwh_m2: float,
        building_type: BuildingType,
        climate_zone: ClimateZone,
    ) -> DECResult:
        """Calculate Display Energy Certificate operational rating.

        OR = (E_actual / E_benchmark) * 100

        Args:
            eui_kwh_m2: Building EUI.
            building_type: Building type.
            climate_zone: Climate zone.

        Returns:
            DECResult with rating band.
        """
        benchmarks = self._get_eui_benchmarks(building_type, climate_zone)
        bench_typical = _decimal(benchmarks["typical"])
        eui = _decimal(eui_kwh_m2)

        # Operational Rating
        or_val = _safe_divide(eui * Decimal("100"), bench_typical, Decimal("100"))

        or_f = float(or_val)
        or_int = int(or_f)

        # Determine band
        band = DECRating.G
        for rating, thresholds in DEC_RATING_THRESHOLDS.items():
            if thresholds["lower"] <= or_int <= thresholds["upper"]:
                band = rating
                break

        # Improvement to next band
        improvement = None
        band_order = [
            DECRating.A, DECRating.B, DECRating.C, DECRating.D,
            DECRating.E, DECRating.F, DECRating.G,
        ]
        band_idx = band_order.index(band)
        if band_idx > 0:
            target_band = band_order[band_idx - 1]
            target_upper = DEC_RATING_THRESHOLDS[target_band]["upper"]
            target_eui = bench_typical * _decimal(target_upper) / Decimal("100")
            improvement = float(eui - target_eui)
            if improvement < 0:
                improvement = 0.0

        return DECResult(
            operational_rating=_round2(or_f),
            dec_band=band.value,
            benchmark_eui=_round2(float(bench_typical)),
            improvement_to_next_band=_round2(improvement) if improvement is not None else None,
        )

    # ---------------------------------------------------------------
    # Public: check_crrem_compliance
    # ---------------------------------------------------------------

    def check_crrem_compliance(
        self,
        inp: BenchmarkInput,
        scenario: CRREMScenario,
    ) -> Optional[CRREMResult]:
        """Check CRREM pathway compliance.

        Compares building carbon intensity against CRREM pathway target
        for the assessment year.

        Args:
            inp: Benchmark input.
            scenario: CRREM scenario (1.5C or 2.0C).

        Returns:
            CRREMResult or None if no pathway data.
        """
        pathways = CRREM_PATHWAYS.get(inp.building_type)
        if not pathways:
            return None

        scenario_pathway = pathways.get(scenario)
        if not scenario_pathway:
            return None

        # Calculate current carbon intensity
        e = inp.energy
        elec = _decimal(e.electricity_kwh)
        gas = _decimal(e.gas_kwh)
        oil = _decimal(e.oil_kwh)
        dh = _decimal(e.district_heating_kwh)
        bio = _decimal(e.biomass_kwh)
        lpg = _decimal(e.lpg_kwh)
        renew = _decimal(e.on_site_renewable_kwh)

        cf_elec = _decimal(inp.carbon_factor_electricity)
        cf_gas = _decimal(inp.carbon_factor_gas)
        cf_oil = _decimal(CARBON_FACTORS_KG_PER_KWH["oil"])
        cf_dh = _decimal(CARBON_FACTORS_KG_PER_KWH["district_heating"])
        cf_bio = _decimal(CARBON_FACTORS_KG_PER_KWH["biomass"])
        cf_lpg = _decimal(CARBON_FACTORS_KG_PER_KWH["lpg"])

        total_carbon = (
            elec * cf_elec + gas * cf_gas + oil * cf_oil
            + dh * cf_dh + bio * cf_bio + lpg * cf_lpg
            - renew * cf_elec
        )
        if total_carbon < Decimal("0"):
            total_carbon = Decimal("0")

        area = _decimal(inp.gross_internal_area_m2)
        carbon_intensity = _safe_divide(total_carbon, area)

        # Interpolate pathway target for assessment year
        target = self._interpolate_crrem(scenario_pathway, inp.assessment_year)
        target_d = _decimal(target)

        gap = carbon_intensity - target_d
        compliant = gap <= Decimal("0")

        reduction_needed = Decimal("0")
        if gap > Decimal("0"):
            reduction_needed = _safe_pct(gap, carbon_intensity)

        # Estimate stranding year
        stranding_year = self._estimate_stranding_year(
            scenario_pathway, float(carbon_intensity)
        )

        return CRREMResult(
            scenario=scenario.value,
            current_carbon_intensity_kg_m2=_round2(float(carbon_intensity)),
            pathway_target_kg_m2=_round2(target),
            gap_kg_m2=_round2(float(gap)),
            compliant=compliant,
            estimated_stranding_year=stranding_year,
            reduction_needed_pct=_round2(float(reduction_needed)),
        )

    # ---------------------------------------------------------------
    # Public: estimate_energy_star_score
    # ---------------------------------------------------------------

    def estimate_energy_star_score(
        self,
        eui_kwh_m2: float,
        building_type: BuildingType,
        climate_zone: ClimateZone,
    ) -> EnergyStarResult:
        """Estimate ENERGY STAR score from EUI percentile.

        Uses a simplified percentile approach based on EUI benchmarks.
        Actual ENERGY STAR uses regression with multiple variables.

        Args:
            eui_kwh_m2: Building EUI.
            building_type: Building type.
            climate_zone: Climate zone.

        Returns:
            EnergyStarResult with estimated score.
        """
        benchmarks = self._get_eui_benchmarks(building_type, climate_zone)
        bp = _decimal(benchmarks["best_practice"])
        good = _decimal(benchmarks["good"])
        typical = _decimal(benchmarks["typical"])
        poor = _decimal(benchmarks["poor"])

        eui = _decimal(eui_kwh_m2)

        # Map EUI to score (lower EUI = higher score)
        if eui <= bp:
            score = 95
        elif eui <= good:
            # Linear interpolation: bp->95, good->75
            frac = _safe_divide(eui - bp, good - bp, Decimal("0.5"))
            score = int(95 - float(frac) * 20)
        elif eui <= typical:
            frac = _safe_divide(eui - good, typical - good, Decimal("0.5"))
            score = int(75 - float(frac) * 25)
        elif eui <= poor:
            frac = _safe_divide(eui - typical, poor - typical, Decimal("0.5"))
            score = int(50 - float(frac) * 30)
        else:
            # Beyond poor
            score = max(1, int(20 - float(_safe_divide(eui - poor, poor, Decimal("0"))) * 10))

        score = max(1, min(100, score))

        if score >= 90:
            percentile = "Top 10% of comparable buildings"
        elif score >= 75:
            percentile = "Top 25% of comparable buildings"
        elif score >= 50:
            percentile = "Top 50% of comparable buildings"
        else:
            percentile = "Below median performance"

        return EnergyStarResult(
            estimated_score=score,
            percentile_rank=percentile,
            certification_eligible=score >= 75,
        )

    # ---------------------------------------------------------------
    # Public: compare_peers
    # ---------------------------------------------------------------

    def compare_peers(
        self,
        eui_kwh_m2: float,
        peers: List[PeerBuildingInput],
    ) -> List[PeerComparisonResult]:
        """Compare building EUI against peers.

        Args:
            eui_kwh_m2: Assessed building EUI.
            peers: List of peer buildings.

        Returns:
            List of PeerComparisonResult.
        """
        results: List[PeerComparisonResult] = []
        eui = _decimal(eui_kwh_m2)

        for peer in peers:
            peer_eui = _decimal(peer.eui_kwh_per_m2)
            diff = eui - peer_eui
            pct_diff = _safe_pct(diff, peer_eui)

            results.append(PeerComparisonResult(
                peer_name=peer.building_name,
                peer_eui=peer.eui_kwh_per_m2,
                difference_kwh_m2=_round2(float(diff)),
                difference_pct=_round2(float(pct_diff)),
            ))

        return results

    # ---------------------------------------------------------------
    # Public: identify_gaps
    # ---------------------------------------------------------------

    def identify_gaps(
        self,
        eui_kwh_m2: float,
        building_type: BuildingType,
        climate_zone: ClimateZone,
        floor_area_m2: float,
        electricity_cost: float = 0.30,
        gas_cost: float = 0.08,
        carbon_factor_elec: float = 0.233,
        carbon_factor_gas: float = 0.203,
    ) -> GapAnalysisResult:
        """Identify gap to best practice and potential savings.

        Gap = EUI_actual - EUI_best_practice  [kWh/m2/yr]
        Saving = Gap * A_floor  [kWh/yr]

        Args:
            eui_kwh_m2: Building EUI.
            building_type: Building type.
            climate_zone: Climate zone.
            floor_area_m2: Floor area.
            electricity_cost: EUR/kWh electricity.
            gas_cost: EUR/kWh gas.
            carbon_factor_elec: kgCO2e/kWh electricity.
            carbon_factor_gas: kgCO2e/kWh gas.

        Returns:
            GapAnalysisResult with savings potential.
        """
        benchmarks = self._get_eui_benchmarks(building_type, climate_zone)
        bp = _decimal(benchmarks["best_practice"])
        good = _decimal(benchmarks["good"])

        eui = _decimal(eui_kwh_m2)
        area = _decimal(floor_area_m2)

        gap_bp = eui - bp
        gap_good = eui - good

        if gap_bp < Decimal("0"):
            gap_bp = Decimal("0")
        if gap_good < Decimal("0"):
            gap_good = Decimal("0")

        saving_kwh = gap_bp * area

        # Blended cost and carbon factor (assume 50/50 elec/thermal split)
        blended_cost = (_decimal(electricity_cost) + _decimal(gas_cost)) / Decimal("2")
        blended_carbon = (_decimal(carbon_factor_elec) + _decimal(carbon_factor_gas)) / Decimal("2")

        saving_eur = saving_kwh * blended_cost
        saving_carbon = saving_kwh * blended_carbon

        return GapAnalysisResult(
            gap_to_best_practice_kwh_m2=_round2(float(gap_bp)),
            gap_to_good_kwh_m2=_round2(float(gap_good)),
            potential_saving_kwh=_round2(float(saving_kwh)),
            potential_saving_eur=_round2(float(saving_eur)),
            potential_carbon_saving_kg=_round2(float(saving_carbon)),
        )

    # ---------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------

    def _get_eui_benchmarks(
        self, building_type: BuildingType, climate_zone: ClimateZone
    ) -> Dict[str, float]:
        """Get EUI benchmarks for building type and climate zone.

        Args:
            building_type: Building type.
            climate_zone: Climate zone.

        Returns:
            Dict with best_practice, good, typical, poor values.
        """
        bt_data = EUI_BENCHMARKS.get(building_type)
        if not bt_data:
            bt_data = EUI_BENCHMARKS[BuildingType.OFFICE]

        cz_data = bt_data.get(climate_zone)
        if not cz_data:
            cz_data = bt_data.get(ClimateZone.CENTRAL_EUROPE, {
                "best_practice": 100, "good": 150, "typical": 200, "poor": 300
            })

        return cz_data

    def _interpolate_crrem(
        self, pathway: Dict[int, float], year: int
    ) -> float:
        """Linearly interpolate CRREM pathway target for a year.

        Args:
            pathway: CRREM pathway {year: target_kgCO2/m2}.
            year: Assessment year.

        Returns:
            Interpolated target in kgCO2/m2/yr.
        """
        years_sorted = sorted(pathway.keys())

        if year <= years_sorted[0]:
            return pathway[years_sorted[0]]
        if year >= years_sorted[-1]:
            return pathway[years_sorted[-1]]

        # Find bracketing years
        for i in range(len(years_sorted) - 1):
            y1 = years_sorted[i]
            y2 = years_sorted[i + 1]
            if y1 <= year <= y2:
                v1 = _decimal(pathway[y1])
                v2 = _decimal(pathway[y2])
                frac = _safe_divide(
                    _decimal(year - y1), _decimal(y2 - y1), Decimal("0")
                )
                target = v1 + frac * (v2 - v1)
                return _round2(float(target))

        return pathway[years_sorted[-1]]

    def _estimate_stranding_year(
        self, pathway: Dict[int, float], carbon_intensity: float
    ) -> Optional[int]:
        """Estimate year building becomes stranded on CRREM pathway.

        Building is stranded when its intensity exceeds the pathway.

        Args:
            pathway: CRREM pathway.
            carbon_intensity: Current carbon intensity.

        Returns:
            Estimated stranding year or None if already stranded / never.
        """
        ci = carbon_intensity
        years_sorted = sorted(pathway.keys())

        for yr in years_sorted:
            if ci > pathway[yr]:
                return yr

        return None

    def _estimate_end_use_split(
        self,
        total_energy_kwh: float,
        building_type: BuildingType,
    ) -> EndUseSplitResult:
        """Estimate end use breakdown from typical profiles.

        Args:
            total_energy_kwh: Total energy.
            building_type: Building type.

        Returns:
            EndUseSplitResult.
        """
        split = TYPICAL_END_USE_SPLIT.get(building_type)
        if not split:
            split = TYPICAL_END_USE_SPLIT[BuildingType.OFFICE]

        total = _decimal(total_energy_kwh)

        return EndUseSplitResult(
            heating_kwh=_round2(float(total * _decimal(split["heating_pct"]) / Decimal("100"))),
            cooling_kwh=_round2(float(total * _decimal(split["cooling_pct"]) / Decimal("100"))),
            lighting_kwh=_round2(float(total * _decimal(split["lighting_pct"]) / Decimal("100"))),
            plug_loads_kwh=_round2(float(total * _decimal(split["plug_loads_pct"]) / Decimal("100"))),
            dhw_kwh=_round2(float(total * _decimal(split["dhw_pct"]) / Decimal("100"))),
            fans_pumps_kwh=_round2(float(total * _decimal(split["fans_pumps_pct"]) / Decimal("100"))),
        )

    # ---------------------------------------------------------------
    # Internal: recommendations
    # ---------------------------------------------------------------

    def _generate_recommendations(
        self,
        inp: BenchmarkInput,
        eui: EUIResult,
        dec: DECResult,
        crrem_15: Optional[CRREMResult],
        crrem_20: Optional[CRREMResult],
        energy_star: Optional[EnergyStarResult],
        gap: GapAnalysisResult,
    ) -> List[str]:
        """Generate benchmarking recommendations.

        Args:
            inp: Benchmark input.
            eui: EUI result.
            dec: DEC result.
            crrem_15: CRREM 1.5C result.
            crrem_20: CRREM 2.0C result.
            energy_star: Energy Star result.
            gap: Gap analysis result.

        Returns:
            List of recommendation strings.
        """
        recs: List[str] = []

        # R1: EUI performance
        if eui.performance_tier == "poor":
            recs.append(
                f"Building EUI of {eui.eui_kwh_per_m2} kWh/m2/yr is in the "
                f"'poor' performance band for {inp.building_type.value} buildings. "
                f"A comprehensive energy audit (EN 16247-2) is recommended "
                f"to identify major inefficiencies. Target: {eui.benchmark_good} "
                f"kWh/m2/yr (good practice)."
            )
        elif eui.performance_tier == "typical":
            recs.append(
                f"Building EUI of {eui.eui_kwh_per_m2} kWh/m2/yr is at "
                f"typical performance for {inp.building_type.value} buildings. "
                f"Systematic improvements to HVAC, lighting, and controls "
                f"can achieve the good practice target of {eui.benchmark_good} "
                f"kWh/m2/yr."
            )

        # R2: DEC rating
        if dec.dec_band in ("F", "G"):
            recs.append(
                f"DEC operational rating of {dec.dec_band} (OR={dec.operational_rating:.0f}) "
                f"indicates severe energy inefficiency. Under the EPBD recast, "
                f"buildings rated F or G may face mandatory improvement "
                f"requirements. Prioritise fabric and systems improvements."
            )
        elif dec.dec_band in ("D", "E"):
            recs.append(
                f"DEC rating of {dec.dec_band} (OR={dec.operational_rating:.0f}). "
                f"Improvement of {dec.improvement_to_next_band:.0f} kWh/m2 "
                f"would achieve the next rating band."
                if dec.improvement_to_next_band
                else f"DEC rating of {dec.dec_band}. Review major end-use systems."
            )

        # R3: CRREM stranding risk
        if crrem_15 and not crrem_15.compliant:
            recs.append(
                f"CRREM 1.5 C pathway: Building exceeds target by "
                f"{crrem_15.gap_kg_m2} kgCO2/m2/yr. "
                f"{'Estimated stranding year: ' + str(crrem_15.estimated_stranding_year) + '. ' if crrem_15.estimated_stranding_year else ''}"
                f"A {crrem_15.reduction_needed_pct:.0f}% carbon reduction is "
                f"needed. This represents significant stranding risk for "
                f"investors and lenders."
            )

        if crrem_20 and not crrem_20.compliant:
            recs.append(
                f"CRREM 2.0 C pathway: Building exceeds target by "
                f"{crrem_20.gap_kg_m2} kgCO2/m2/yr. "
                f"Carbon reduction of {crrem_20.reduction_needed_pct:.0f}% needed."
            )

        # R4: Energy Star
        if energy_star and energy_star.estimated_score < 50:
            recs.append(
                f"Estimated ENERGY STAR score of {energy_star.estimated_score} "
                f"(below median). Improving EUI by 20-30% would bring the "
                f"building above the 50th percentile."
            )
        elif energy_star and energy_star.certification_eligible:
            recs.append(
                f"Estimated ENERGY STAR score of {energy_star.estimated_score} "
                f"qualifies for ENERGY STAR certification (>= 75). "
                f"Consider applying for certification to demonstrate energy leadership."
            )

        # R5: Gap savings
        if gap.gap_to_best_practice_kwh_m2 > 10:
            recs.append(
                f"Gap to best practice is {gap.gap_to_best_practice_kwh_m2} kWh/m2/yr, "
                f"equivalent to {gap.potential_saving_kwh:.0f} kWh/yr and "
                f"{gap.potential_saving_eur:.0f} EUR/yr. Focus on the largest "
                f"end-use categories: review heating, cooling, and lighting systems."
            )

        # R6: Weather normalisation observation
        if eui.eui_normalised_kwh_per_m2 is not None:
            diff = abs(eui.eui_kwh_per_m2 - eui.eui_normalised_kwh_per_m2)
            if diff > 10:
                recs.append(
                    f"Weather normalisation changed EUI by "
                    f"{diff:.1f} kWh/m2/yr, indicating the assessment year "
                    f"had atypical weather. Use normalised EUI for fair "
                    f"year-on-year comparison."
                )

        if not recs:
            recs.append(
                "Building energy performance benchmarks well against "
                "comparable buildings. Continue monitoring and maintaining "
                "systems per ISO 50001 energy management best practice."
            )

        return recs
