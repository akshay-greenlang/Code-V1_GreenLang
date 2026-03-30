# -*- coding: utf-8 -*-
"""
EntericFermentationEngine - Livestock Enteric CH4 Emissions (Engine 2 of 7)

AGENT-MRV-008: Agricultural Emissions Agent

Core calculation engine implementing IPCC 2006 Guidelines Vol 4 Ch 10.3
methodology for methane (CH4) emissions from livestock enteric fermentation.
Supports both Tier 1 (emission factor) and Tier 2 (feed intake / gross energy)
approaches across 20 animal types.

    1. **Tier 1 (IPCC Eq 10.19)**:
       CH4_enteric = EF_T * N_T / 1000  (tonnes CH4/yr per animal type)
       Total = SUM over all animal types
       Emission factors from IPCC Table 10.11 (kg CH4/head/yr).

    2. **Tier 2 (IPCC Eq 10.21 / 10.16-10.18)**:
       EF = (GE * Ym/100 * 365) / 55.65  (kg CH4/head/yr)

       Gross Energy intake:
       GE = [(NE_m + NE_a + NE_l + NE_work + NE_p) / REM
             + NE_g / REG] / (DE/100)

       Net Energy components (MJ/day):
       NE_m   = Cfi * BW^0.75               (maintenance)
       NE_a   = Ca * NE_m                    (activity)
       NE_l   = Milk * (1.47 + 0.40 * Fat)  (lactation)
       NE_p   = 0.10 * NE_m                 (pregnancy)
       NE_g   = 22.02 * (BW / (C * BW_mature))^0.75 * WG^1.097  (growth)
       NE_work = 0.10 * NE_m * hours_work   (work, for draught animals)

       Ratios of NE available in diet for maintenance / growth:
       REM = 1.123 - 4.092e-3*DE + 1.126e-5*DE^2 - 25.4/DE
       REG = 1.164 - 5.160e-3*DE + 1.308e-5*DE^2 - 37.4/DE

All calculations use Python Decimal arithmetic with 8+ decimal places for
zero-hallucination determinism.  Every calculation result includes a per-gas
breakdown, GWP-adjusted CO2e, full calculation trace, processing time, and
SHA-256 provenance hash.

Zero-Hallucination Guarantees:
    - All numeric calculations use Python Decimal.
    - No LLM calls in any calculation path.
    - Every calculation step is logged and traceable via TraceStep.
    - SHA-256 provenance hash for every result.
    - Identical inputs always produce identical outputs.

Thread Safety:
    Stateless per-calculation.  Mutable counters protected by reentrant lock.

Example:
    >>> from greenlang.agents.mrv.agricultural_emissions.enteric_fermentation import (
    ...     EntericFermentationEngine,
    ... )
    >>> engine = EntericFermentationEngine()
    >>> result = engine.calculate_tier1(
    ...     animal_type="DAIRY_CATTLE",
    ...     head_count=500,
    ...     region="DEVELOPED",
    ... )
    >>> assert result["status"] == "SUCCESS"
    >>> assert result["ch4_tonnes"]

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-008 Agricultural Emissions (GL-MRV-SCOPE1-008)
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4
from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level exports
# ---------------------------------------------------------------------------

__all__ = ["EntericFermentationEngine"]

# ---------------------------------------------------------------------------
# Conditional imports
# ---------------------------------------------------------------------------

try:
    from greenlang.agents.mrv.agricultural_emissions.config import get_config as _get_config
    _CONFIG_AVAILABLE = True
except ImportError:
    _CONFIG_AVAILABLE = False
    _get_config = None  # type: ignore[assignment]

try:
    from greenlang.agents.mrv.agricultural_emissions.provenance import (
        get_provenance_tracker as _get_provenance_tracker,
    )
    _PROVENANCE_AVAILABLE = True
except ImportError:
    _PROVENANCE_AVAILABLE = False
    _get_provenance_tracker = None  # type: ignore[assignment]

try:
    from greenlang.agents.mrv.agricultural_emissions.metrics import (
        record_calculation as _record_calculation,
        observe_calculation_duration as _observe_calculation_duration,
        record_emissions as _record_emissions,
        record_enteric_calculation as _record_enteric_calculation,
        record_calculation_error as _record_calculation_error,
    )
    _METRICS_AVAILABLE = True
except ImportError:
    _METRICS_AVAILABLE = False
    _record_calculation = None  # type: ignore[assignment]
    _observe_calculation_duration = None  # type: ignore[assignment]
    _record_emissions = None  # type: ignore[assignment]
    _record_enteric_calculation = None  # type: ignore[assignment]
    _record_calculation_error = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# UTC helper
# ---------------------------------------------------------------------------

def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash of arbitrary data.

    Args:
        data: Data to hash (dict, list, str, or Pydantic model).

    Returns:
        SHA-256 hex digest string.
    """
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    else:
        serializable = data
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()

# ---------------------------------------------------------------------------
# Decimal precision constants
# ---------------------------------------------------------------------------

_PRECISION = Decimal("0.00000001")  # 8 decimal places
_ZERO = Decimal("0")
_ONE = Decimal("1")
_TEN = Decimal("10")
_HUNDRED = Decimal("100")
_THOUSAND = Decimal("1000")
_KG_TO_TONNES = Decimal("0.001")
_TONNES_TO_KG = Decimal("1000")

#: Energy content of methane in MJ/kg CH4 (IPCC 2006 Vol 4 Ch 10)
_CH4_ENERGY_MJ_PER_KG = Decimal("55.65")

#: Days per year for annualization
_DAYS_PER_YEAR = Decimal("365")

def _D(value: Any) -> Decimal:
    """Convert a value to Decimal with controlled precision.

    Args:
        value: Numeric value (int, float, str, or Decimal).

    Returns:
        Decimal representation.

    Raises:
        ValueError: If value cannot be converted to Decimal.
    """
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"Cannot convert {value!r} to Decimal") from exc

def _safe_decimal(value: Any, default: Decimal = _ZERO) -> Decimal:
    """Safely convert a value to Decimal, returning default on failure.

    Args:
        value: Value to convert.
        default: Fallback Decimal value.

    Returns:
        Converted Decimal or default.
    """
    if value is None:
        return default
    try:
        return _D(value)
    except (InvalidOperation, ValueError, TypeError):
        return default

def _quantize(value: Decimal) -> Decimal:
    """Quantize a Decimal to the standard 8-decimal-place precision.

    Args:
        value: Decimal value to quantize.

    Returns:
        Quantized Decimal.
    """
    try:
        return value.quantize(_PRECISION, rounding=ROUND_HALF_UP)
    except InvalidOperation:
        logger.warning("Failed to quantize value: %s", value)
        return value

# ===========================================================================
# Enumerations
# ===========================================================================

class AnimalType(str, Enum):
    """IPCC Table 10.11 livestock categories for enteric fermentation."""

    DAIRY_CATTLE = "DAIRY_CATTLE"
    NON_DAIRY_CATTLE = "NON_DAIRY_CATTLE"
    BUFFALO = "BUFFALO"
    SHEEP = "SHEEP"
    GOATS = "GOATS"
    CAMELS = "CAMELS"
    HORSES = "HORSES"
    MULES_ASSES = "MULES_ASSES"
    SWINE = "SWINE"
    POULTRY = "POULTRY"
    LLAMAS = "LLAMAS"
    ALPACAS = "ALPACAS"
    DEER = "DEER"
    REINDEER = "REINDEER"
    RABBITS = "RABBITS"
    FUR_BEARING_ANIMALS = "FUR_BEARING_ANIMALS"
    OSTRICH = "OSTRICH"
    DAIRY_YOUNG = "DAIRY_YOUNG"
    BEEF_FEEDLOT = "BEEF_FEEDLOT"
    BEEF_PASTURE = "BEEF_PASTURE"

class Region(str, Enum):
    """Regional classification for default emission factor selection."""

    DEVELOPED = "DEVELOPED"
    DEVELOPING = "DEVELOPING"
    LATIN_AMERICA = "LATIN_AMERICA"
    AFRICA = "AFRICA"
    MIDDLE_EAST = "MIDDLE_EAST"
    EASTERN_EUROPE = "EASTERN_EUROPE"
    INDIAN_SUBCONTINENT = "INDIAN_SUBCONTINENT"
    ASIA = "ASIA"
    OCEANIA = "OCEANIA"

class FeedQuality(str, Enum):
    """Feed quality levels for default Ym and DE selection."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"

class ActivityLevel(str, Enum):
    """Livestock activity levels for Ca coefficient."""

    STALL = "STALL"
    HOUSED = "HOUSED"
    PASTURE = "PASTURE"
    GRAZING_LARGE = "GRAZING_LARGE"

class CalculationStatus(str, Enum):
    """Result status codes."""

    SUCCESS = "SUCCESS"
    PARTIAL = "PARTIAL"
    ERROR = "ERROR"

# ===========================================================================
# Trace Step Dataclass
# ===========================================================================

@dataclass
class TraceStep:
    """Single step in a calculation trace for audit trail.

    Attributes:
        step_number: Sequential step number.
        description: Human-readable description.
        formula: Mathematical formula applied.
        inputs: Input values for this step.
        output: Output value from this step.
        unit: Unit of the output value.
    """

    step_number: int
    description: str
    formula: str
    inputs: Dict[str, str]
    output: str
    unit: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "step_number": self.step_number,
            "description": self.description,
            "formula": self.formula,
            "inputs": self.inputs,
            "output": self.output,
            "unit": self.unit,
        }

# ===========================================================================
# GWP Lookup Tables (built-in fallback)
# ===========================================================================

_GWP_TABLE: Dict[str, Dict[str, Decimal]] = {
    "AR4": {
        "CO2": Decimal("1"),
        "CH4": Decimal("25"),
        "N2O": Decimal("298"),
    },
    "AR5": {
        "CO2": Decimal("1"),
        "CH4": Decimal("28"),
        "N2O": Decimal("265"),
    },
    "AR6": {
        "CO2": Decimal("1"),
        "CH4": Decimal("27.9"),
        "N2O": Decimal("273"),
    },
    "AR6_20YR": {
        "CO2": Decimal("1"),
        "CH4": Decimal("82.5"),
        "N2O": Decimal("273"),
    },
}

# ===========================================================================
# IPCC Table 10.11 - Enteric Fermentation Emission Factors
# (kg CH4 / head / yr)
# ===========================================================================

#: Tier 1 emission factors by animal type and region (kg CH4/head/yr)
#: Sources: IPCC 2006 Vol 4, Table 10.11; IPCC 2019 Refined, Table 10.11
_ENTERIC_EF_TIER1: Dict[str, Dict[str, Decimal]] = {
    "DAIRY_CATTLE": {
        "DEVELOPED": Decimal("128"),
        "DEVELOPING": Decimal("68"),
        "LATIN_AMERICA": Decimal("72"),
        "AFRICA": Decimal("46"),
        "MIDDLE_EAST": Decimal("63"),
        "EASTERN_EUROPE": Decimal("99"),
        "INDIAN_SUBCONTINENT": Decimal("58"),
        "ASIA": Decimal("68"),
        "OCEANIA": Decimal("100"),
    },
    "NON_DAIRY_CATTLE": {
        "DEVELOPED": Decimal("66"),
        "DEVELOPING": Decimal("47"),
        "LATIN_AMERICA": Decimal("56"),
        "AFRICA": Decimal("31"),
        "MIDDLE_EAST": Decimal("37"),
        "EASTERN_EUROPE": Decimal("52"),
        "INDIAN_SUBCONTINENT": Decimal("27"),
        "ASIA": Decimal("47"),
        "OCEANIA": Decimal("60"),
    },
    "BUFFALO": {
        "DEVELOPED": Decimal("55"),
        "DEVELOPING": Decimal("55"),
        "LATIN_AMERICA": Decimal("55"),
        "AFRICA": Decimal("55"),
        "MIDDLE_EAST": Decimal("55"),
        "EASTERN_EUROPE": Decimal("55"),
        "INDIAN_SUBCONTINENT": Decimal("55"),
        "ASIA": Decimal("55"),
        "OCEANIA": Decimal("55"),
    },
    "SHEEP": {
        "DEVELOPED": Decimal("8"),
        "DEVELOPING": Decimal("5"),
        "LATIN_AMERICA": Decimal("5"),
        "AFRICA": Decimal("5"),
        "MIDDLE_EAST": Decimal("5"),
        "EASTERN_EUROPE": Decimal("8"),
        "INDIAN_SUBCONTINENT": Decimal("5"),
        "ASIA": Decimal("5"),
        "OCEANIA": Decimal("8"),
    },
    "GOATS": {
        "DEVELOPED": Decimal("5"),
        "DEVELOPING": Decimal("5"),
        "LATIN_AMERICA": Decimal("5"),
        "AFRICA": Decimal("5"),
        "MIDDLE_EAST": Decimal("5"),
        "EASTERN_EUROPE": Decimal("5"),
        "INDIAN_SUBCONTINENT": Decimal("5"),
        "ASIA": Decimal("5"),
        "OCEANIA": Decimal("5"),
    },
    "CAMELS": {
        "DEVELOPED": Decimal("46"),
        "DEVELOPING": Decimal("46"),
        "LATIN_AMERICA": Decimal("46"),
        "AFRICA": Decimal("46"),
        "MIDDLE_EAST": Decimal("46"),
        "EASTERN_EUROPE": Decimal("46"),
        "INDIAN_SUBCONTINENT": Decimal("46"),
        "ASIA": Decimal("46"),
        "OCEANIA": Decimal("46"),
    },
    "HORSES": {
        "DEVELOPED": Decimal("18"),
        "DEVELOPING": Decimal("18"),
        "LATIN_AMERICA": Decimal("18"),
        "AFRICA": Decimal("18"),
        "MIDDLE_EAST": Decimal("18"),
        "EASTERN_EUROPE": Decimal("18"),
        "INDIAN_SUBCONTINENT": Decimal("18"),
        "ASIA": Decimal("18"),
        "OCEANIA": Decimal("18"),
    },
    "MULES_ASSES": {
        "DEVELOPED": Decimal("10"),
        "DEVELOPING": Decimal("10"),
        "LATIN_AMERICA": Decimal("10"),
        "AFRICA": Decimal("10"),
        "MIDDLE_EAST": Decimal("10"),
        "EASTERN_EUROPE": Decimal("10"),
        "INDIAN_SUBCONTINENT": Decimal("10"),
        "ASIA": Decimal("10"),
        "OCEANIA": Decimal("10"),
    },
    "SWINE": {
        "DEVELOPED": Decimal("1.5"),
        "DEVELOPING": Decimal("1"),
        "LATIN_AMERICA": Decimal("1"),
        "AFRICA": Decimal("1"),
        "MIDDLE_EAST": Decimal("1"),
        "EASTERN_EUROPE": Decimal("1.5"),
        "INDIAN_SUBCONTINENT": Decimal("1"),
        "ASIA": Decimal("1"),
        "OCEANIA": Decimal("1.5"),
    },
    "POULTRY": {
        "DEVELOPED": Decimal("0"),
        "DEVELOPING": Decimal("0"),
        "LATIN_AMERICA": Decimal("0"),
        "AFRICA": Decimal("0"),
        "MIDDLE_EAST": Decimal("0"),
        "EASTERN_EUROPE": Decimal("0"),
        "INDIAN_SUBCONTINENT": Decimal("0"),
        "ASIA": Decimal("0"),
        "OCEANIA": Decimal("0"),
    },
    "LLAMAS": {
        "DEVELOPED": Decimal("8"),
        "DEVELOPING": Decimal("8"),
        "LATIN_AMERICA": Decimal("8"),
        "AFRICA": Decimal("8"),
        "MIDDLE_EAST": Decimal("8"),
        "EASTERN_EUROPE": Decimal("8"),
        "INDIAN_SUBCONTINENT": Decimal("8"),
        "ASIA": Decimal("8"),
        "OCEANIA": Decimal("8"),
    },
    "ALPACAS": {
        "DEVELOPED": Decimal("8"),
        "DEVELOPING": Decimal("8"),
        "LATIN_AMERICA": Decimal("8"),
        "AFRICA": Decimal("8"),
        "MIDDLE_EAST": Decimal("8"),
        "EASTERN_EUROPE": Decimal("8"),
        "INDIAN_SUBCONTINENT": Decimal("8"),
        "ASIA": Decimal("8"),
        "OCEANIA": Decimal("8"),
    },
    "DEER": {
        "DEVELOPED": Decimal("20"),
        "DEVELOPING": Decimal("20"),
        "LATIN_AMERICA": Decimal("20"),
        "AFRICA": Decimal("20"),
        "MIDDLE_EAST": Decimal("20"),
        "EASTERN_EUROPE": Decimal("20"),
        "INDIAN_SUBCONTINENT": Decimal("20"),
        "ASIA": Decimal("20"),
        "OCEANIA": Decimal("20"),
    },
    "REINDEER": {
        "DEVELOPED": Decimal("20"),
        "DEVELOPING": Decimal("20"),
        "LATIN_AMERICA": Decimal("20"),
        "AFRICA": Decimal("20"),
        "MIDDLE_EAST": Decimal("20"),
        "EASTERN_EUROPE": Decimal("20"),
        "INDIAN_SUBCONTINENT": Decimal("20"),
        "ASIA": Decimal("20"),
        "OCEANIA": Decimal("20"),
    },
    "RABBITS": {
        "DEVELOPED": Decimal("0"),
        "DEVELOPING": Decimal("0"),
        "LATIN_AMERICA": Decimal("0"),
        "AFRICA": Decimal("0"),
        "MIDDLE_EAST": Decimal("0"),
        "EASTERN_EUROPE": Decimal("0"),
        "INDIAN_SUBCONTINENT": Decimal("0"),
        "ASIA": Decimal("0"),
        "OCEANIA": Decimal("0"),
    },
    "FUR_BEARING_ANIMALS": {
        "DEVELOPED": Decimal("0"),
        "DEVELOPING": Decimal("0"),
        "LATIN_AMERICA": Decimal("0"),
        "AFRICA": Decimal("0"),
        "MIDDLE_EAST": Decimal("0"),
        "EASTERN_EUROPE": Decimal("0"),
        "INDIAN_SUBCONTINENT": Decimal("0"),
        "ASIA": Decimal("0"),
        "OCEANIA": Decimal("0"),
    },
    "OSTRICH": {
        "DEVELOPED": Decimal("0"),
        "DEVELOPING": Decimal("0"),
        "LATIN_AMERICA": Decimal("0"),
        "AFRICA": Decimal("0"),
        "MIDDLE_EAST": Decimal("0"),
        "EASTERN_EUROPE": Decimal("0"),
        "INDIAN_SUBCONTINENT": Decimal("0"),
        "ASIA": Decimal("0"),
        "OCEANIA": Decimal("0"),
    },
    "DAIRY_YOUNG": {
        "DEVELOPED": Decimal("66"),
        "DEVELOPING": Decimal("47"),
        "LATIN_AMERICA": Decimal("47"),
        "AFRICA": Decimal("31"),
        "MIDDLE_EAST": Decimal("37"),
        "EASTERN_EUROPE": Decimal("52"),
        "INDIAN_SUBCONTINENT": Decimal("27"),
        "ASIA": Decimal("47"),
        "OCEANIA": Decimal("60"),
    },
    "BEEF_FEEDLOT": {
        "DEVELOPED": Decimal("53"),
        "DEVELOPING": Decimal("47"),
        "LATIN_AMERICA": Decimal("56"),
        "AFRICA": Decimal("31"),
        "MIDDLE_EAST": Decimal("37"),
        "EASTERN_EUROPE": Decimal("52"),
        "INDIAN_SUBCONTINENT": Decimal("27"),
        "ASIA": Decimal("47"),
        "OCEANIA": Decimal("53"),
    },
    "BEEF_PASTURE": {
        "DEVELOPED": Decimal("70"),
        "DEVELOPING": Decimal("56"),
        "LATIN_AMERICA": Decimal("63"),
        "AFRICA": Decimal("44"),
        "MIDDLE_EAST": Decimal("44"),
        "EASTERN_EUROPE": Decimal("58"),
        "INDIAN_SUBCONTINENT": Decimal("40"),
        "ASIA": Decimal("56"),
        "OCEANIA": Decimal("68"),
    },
}

# ===========================================================================
# Maintenance Coefficients (Cfi) - IPCC 2006 Vol 4, Table 10.4
# (MJ/day/kg BW^0.75)
# ===========================================================================

_MAINTENANCE_COEFFICIENTS: Dict[str, Decimal] = {
    "DAIRY_CATTLE": Decimal("0.386"),
    "NON_DAIRY_CATTLE": Decimal("0.322"),
    "BUFFALO": Decimal("0.322"),
    "SHEEP": Decimal("0.217"),
    "GOATS": Decimal("0.217"),
    "CAMELS": Decimal("0.322"),
    "HORSES": Decimal("0.322"),
    "MULES_ASSES": Decimal("0.322"),
    "SWINE": Decimal("0.322"),
    "LLAMAS": Decimal("0.246"),
    "ALPACAS": Decimal("0.246"),
    "DEER": Decimal("0.322"),
    "REINDEER": Decimal("0.322"),
    "DAIRY_YOUNG": Decimal("0.322"),
    "BEEF_FEEDLOT": Decimal("0.322"),
    "BEEF_PASTURE": Decimal("0.322"),
}

# ===========================================================================
# Default Body Weights by Animal Type (kg)
# IPCC 2006 Vol 4, Table 10.A-1 / region-averaged representative weights
# ===========================================================================

_DEFAULT_BODY_WEIGHTS: Dict[str, Decimal] = {
    "DAIRY_CATTLE": Decimal("600"),
    "NON_DAIRY_CATTLE": Decimal("420"),
    "BUFFALO": Decimal("380"),
    "SHEEP": Decimal("48"),
    "GOATS": Decimal("40"),
    "CAMELS": Decimal("450"),
    "HORSES": Decimal("380"),
    "MULES_ASSES": Decimal("200"),
    "SWINE": Decimal("60"),
    "POULTRY": Decimal("2"),
    "LLAMAS": Decimal("120"),
    "ALPACAS": Decimal("65"),
    "DEER": Decimal("120"),
    "REINDEER": Decimal("100"),
    "RABBITS": Decimal("3"),
    "FUR_BEARING_ANIMALS": Decimal("5"),
    "OSTRICH": Decimal("100"),
    "DAIRY_YOUNG": Decimal("250"),
    "BEEF_FEEDLOT": Decimal("400"),
    "BEEF_PASTURE": Decimal("400"),
}

# ===========================================================================
# Default Mature Body Weights by Animal Type (kg)
# Used for NE_g calculation (growth energy)
# ===========================================================================

_DEFAULT_MATURE_WEIGHTS: Dict[str, Decimal] = {
    "DAIRY_CATTLE": Decimal("650"),
    "NON_DAIRY_CATTLE": Decimal("550"),
    "BUFFALO": Decimal("500"),
    "SHEEP": Decimal("60"),
    "GOATS": Decimal("50"),
    "CAMELS": Decimal("550"),
    "HORSES": Decimal("500"),
    "MULES_ASSES": Decimal("300"),
    "SWINE": Decimal("120"),
    "LLAMAS": Decimal("140"),
    "ALPACAS": Decimal("80"),
    "DEER": Decimal("160"),
    "REINDEER": Decimal("130"),
    "DAIRY_YOUNG": Decimal("650"),
    "BEEF_FEEDLOT": Decimal("550"),
    "BEEF_PASTURE": Decimal("550"),
}

# ===========================================================================
# Mature Body Weight Fraction C Values (dimensionless)
# IPCC 2006 Vol 4, Eq 10.6: C = 0.8 for females, 1.0 for intact males,
# 1.2 for castrated males.  Default: 0.8 (female majority in herds)
# ===========================================================================

_DEFAULT_C_VALUES: Dict[str, Decimal] = {
    "DAIRY_CATTLE": Decimal("0.8"),
    "NON_DAIRY_CATTLE": Decimal("0.8"),
    "BUFFALO": Decimal("0.8"),
    "SHEEP": Decimal("0.8"),
    "GOATS": Decimal("0.8"),
    "CAMELS": Decimal("0.8"),
    "HORSES": Decimal("0.8"),
    "MULES_ASSES": Decimal("1.0"),
    "SWINE": Decimal("0.8"),
    "LLAMAS": Decimal("0.8"),
    "ALPACAS": Decimal("0.8"),
    "DEER": Decimal("0.8"),
    "REINDEER": Decimal("0.8"),
    "DAIRY_YOUNG": Decimal("0.8"),
    "BEEF_FEEDLOT": Decimal("1.2"),
    "BEEF_PASTURE": Decimal("0.8"),
}

# ===========================================================================
# Default Ym Values (methane conversion factor) by Animal Type & Feed Quality
# Ym = fraction of GE converted to CH4 (% of GE).
# IPCC 2006 Vol 4, Table 10.12 / 10.13
# ===========================================================================

_DEFAULT_YM: Dict[str, Dict[str, Decimal]] = {
    "DAIRY_CATTLE": {
        "LOW": Decimal("7.0"),
        "MEDIUM": Decimal("6.5"),
        "HIGH": Decimal("5.5"),
    },
    "NON_DAIRY_CATTLE": {
        "LOW": Decimal("7.5"),
        "MEDIUM": Decimal("6.5"),
        "HIGH": Decimal("5.0"),
    },
    "BUFFALO": {
        "LOW": Decimal("7.5"),
        "MEDIUM": Decimal("6.5"),
        "HIGH": Decimal("5.5"),
    },
    "SHEEP": {
        "LOW": Decimal("7.0"),
        "MEDIUM": Decimal("6.5"),
        "HIGH": Decimal("5.5"),
    },
    "GOATS": {
        "LOW": Decimal("7.0"),
        "MEDIUM": Decimal("6.5"),
        "HIGH": Decimal("5.5"),
    },
    "CAMELS": {
        "LOW": Decimal("7.5"),
        "MEDIUM": Decimal("7.0"),
        "HIGH": Decimal("6.0"),
    },
    "HORSES": {
        "LOW": Decimal("4.5"),
        "MEDIUM": Decimal("4.0"),
        "HIGH": Decimal("3.5"),
    },
    "MULES_ASSES": {
        "LOW": Decimal("4.5"),
        "MEDIUM": Decimal("4.0"),
        "HIGH": Decimal("3.5"),
    },
    "SWINE": {
        "LOW": Decimal("1.5"),
        "MEDIUM": Decimal("1.0"),
        "HIGH": Decimal("0.7"),
    },
    "LLAMAS": {
        "LOW": Decimal("7.0"),
        "MEDIUM": Decimal("6.5"),
        "HIGH": Decimal("5.5"),
    },
    "ALPACAS": {
        "LOW": Decimal("7.0"),
        "MEDIUM": Decimal("6.5"),
        "HIGH": Decimal("5.5"),
    },
    "DEER": {
        "LOW": Decimal("7.0"),
        "MEDIUM": Decimal("6.5"),
        "HIGH": Decimal("5.5"),
    },
    "REINDEER": {
        "LOW": Decimal("7.0"),
        "MEDIUM": Decimal("6.5"),
        "HIGH": Decimal("5.5"),
    },
    "DAIRY_YOUNG": {
        "LOW": Decimal("7.5"),
        "MEDIUM": Decimal("6.5"),
        "HIGH": Decimal("5.0"),
    },
    "BEEF_FEEDLOT": {
        "LOW": Decimal("5.0"),
        "MEDIUM": Decimal("4.0"),
        "HIGH": Decimal("3.0"),
    },
    "BEEF_PASTURE": {
        "LOW": Decimal("7.5"),
        "MEDIUM": Decimal("6.5"),
        "HIGH": Decimal("5.0"),
    },
}

# ===========================================================================
# Activity Coefficients (Ca) - IPCC 2006 Vol 4, Table 10.5
# Fraction of NE_m additional energy for activity
# ===========================================================================

_ACTIVITY_COEFFICIENTS: Dict[str, Decimal] = {
    "STALL": Decimal("0.00"),
    "HOUSED": Decimal("0.00"),
    "PASTURE": Decimal("0.17"),
    "GRAZING_LARGE": Decimal("0.36"),
}

# ===========================================================================
# Default Feed Digestibility (DE%) by Animal Type and Feed Quality
# IPCC 2006 Vol 4, Table 10.2
# ===========================================================================

_DEFAULT_FEED_DIGESTIBILITY: Dict[str, Dict[str, Decimal]] = {
    "DAIRY_CATTLE": {
        "LOW": Decimal("55"),
        "MEDIUM": Decimal("65"),
        "HIGH": Decimal("75"),
    },
    "NON_DAIRY_CATTLE": {
        "LOW": Decimal("50"),
        "MEDIUM": Decimal("60"),
        "HIGH": Decimal("70"),
    },
    "BUFFALO": {
        "LOW": Decimal("50"),
        "MEDIUM": Decimal("60"),
        "HIGH": Decimal("70"),
    },
    "SHEEP": {
        "LOW": Decimal("55"),
        "MEDIUM": Decimal("63"),
        "HIGH": Decimal("72"),
    },
    "GOATS": {
        "LOW": Decimal("55"),
        "MEDIUM": Decimal("63"),
        "HIGH": Decimal("72"),
    },
    "CAMELS": {
        "LOW": Decimal("48"),
        "MEDIUM": Decimal("55"),
        "HIGH": Decimal("65"),
    },
    "HORSES": {
        "LOW": Decimal("50"),
        "MEDIUM": Decimal("58"),
        "HIGH": Decimal("68"),
    },
    "MULES_ASSES": {
        "LOW": Decimal("50"),
        "MEDIUM": Decimal("58"),
        "HIGH": Decimal("68"),
    },
    "SWINE": {
        "LOW": Decimal("68"),
        "MEDIUM": Decimal("75"),
        "HIGH": Decimal("85"),
    },
    "LLAMAS": {
        "LOW": Decimal("55"),
        "MEDIUM": Decimal("63"),
        "HIGH": Decimal("72"),
    },
    "ALPACAS": {
        "LOW": Decimal("55"),
        "MEDIUM": Decimal("63"),
        "HIGH": Decimal("72"),
    },
    "DEER": {
        "LOW": Decimal("50"),
        "MEDIUM": Decimal("60"),
        "HIGH": Decimal("70"),
    },
    "REINDEER": {
        "LOW": Decimal("50"),
        "MEDIUM": Decimal("60"),
        "HIGH": Decimal("70"),
    },
    "DAIRY_YOUNG": {
        "LOW": Decimal("55"),
        "MEDIUM": Decimal("65"),
        "HIGH": Decimal("75"),
    },
    "BEEF_FEEDLOT": {
        "LOW": Decimal("65"),
        "MEDIUM": Decimal("75"),
        "HIGH": Decimal("82"),
    },
    "BEEF_PASTURE": {
        "LOW": Decimal("50"),
        "MEDIUM": Decimal("60"),
        "HIGH": Decimal("70"),
    },
}

# ===========================================================================
# Default Milk Fat Content by Region (%)
# ===========================================================================

_DEFAULT_MILK_FAT: Dict[str, Decimal] = {
    "DEVELOPED": Decimal("4.0"),
    "DEVELOPING": Decimal("5.0"),
    "LATIN_AMERICA": Decimal("4.5"),
    "AFRICA": Decimal("5.0"),
    "MIDDLE_EAST": Decimal("4.5"),
    "EASTERN_EUROPE": Decimal("4.0"),
    "INDIAN_SUBCONTINENT": Decimal("6.0"),
    "ASIA": Decimal("4.5"),
    "OCEANIA": Decimal("4.5"),
}

# ===========================================================================
# Default Milk Yields by Region (kg/day)
# ===========================================================================

_DEFAULT_MILK_YIELD: Dict[str, Decimal] = {
    "DEVELOPED": Decimal("25"),
    "DEVELOPING": Decimal("6"),
    "LATIN_AMERICA": Decimal("8"),
    "AFRICA": Decimal("3"),
    "MIDDLE_EAST": Decimal("7"),
    "EASTERN_EUROPE": Decimal("12"),
    "INDIAN_SUBCONTINENT": Decimal("4"),
    "ASIA": Decimal("6"),
    "OCEANIA": Decimal("18"),
}

# ===========================================================================
# EntericFermentationEngine
# ===========================================================================

class EntericFermentationEngine:
    """Core calculation engine for CH4 emissions from livestock enteric
    fermentation implementing IPCC 2006 Guidelines Vol 4 Ch 10.3 Tier 1
    and Tier 2 methodologies.

    Uses deterministic Decimal arithmetic throughout.  All numeric lookups
    are performed from built-in IPCC default tables.  An external livestock
    database may be injected to override defaults when site-specific or
    country-specific data is available.

    Thread Safety:
        Per-calculation state is created fresh for each method call.
        Shared counters use a reentrant lock.

    Attributes:
        _config: Optional configuration dictionary.
        _lock: Reentrant lock protecting mutable counters.
        _total_calculations: Counter of total calculations performed.
        _total_batches: Counter of total batch operations.
        _total_errors: Counter of total errors encountered.
        _default_gwp_source: Default GWP source for CO2e conversion.
        _created_at: Engine initialization timestamp.

    Example:
        >>> engine = EntericFermentationEngine()
        >>> result = engine.calculate_tier1(
        ...     animal_type="DAIRY_CATTLE",
        ...     head_count=1000,
        ...     region="DEVELOPED",
        ... )
        >>> assert result["status"] == "SUCCESS"
    """

    def __init__(
        self,
        livestock_database: Optional[Any] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize the EntericFermentationEngine.

        Args:
            livestock_database: Optional livestock database engine for
                site-specific emission factor / animal parameter lookups.
                If None, built-in IPCC default tables are used.
            config: Optional configuration dictionary.  Supports:
                - default_gwp_source (str): Default GWP report (AR4/AR5/AR6).
                - enable_provenance (bool): Enable provenance tracking.
                - decimal_precision (int): Decimal places (default 8).
        """
        self._livestock_db = livestock_database
        self._config = config or {}
        self._lock = threading.RLock()

        # Configuration
        self._default_gwp_source: str = self._config.get(
            "default_gwp_source", "AR6",
        )
        self._enable_provenance: bool = self._config.get(
            "enable_provenance", True,
        )

        # Provenance tracker
        if self._enable_provenance and _PROVENANCE_AVAILABLE:
            self._provenance = _get_provenance_tracker()
        else:
            self._provenance = None

        # Statistics
        self._total_calculations: int = 0
        self._total_batches: int = 0
        self._total_errors: int = 0
        self._created_at: datetime = utcnow()

        logger.info(
            "EntericFermentationEngine initialized: default_gwp=%s, "
            "provenance=%s, livestock_db=%s",
            self._default_gwp_source,
            "enabled" if self._provenance else "disabled",
            "connected" if self._livestock_db else "built-in",
        )

    # ------------------------------------------------------------------
    # Thread-safe counter helpers
    # ------------------------------------------------------------------

    def _increment_calculations(self) -> None:
        """Thread-safe increment of the calculation counter."""
        with self._lock:
            self._total_calculations += 1

    def _increment_batches(self) -> None:
        """Thread-safe increment of the batch counter."""
        with self._lock:
            self._total_batches += 1

    def _increment_errors(self) -> None:
        """Thread-safe increment of the error counter."""
        with self._lock:
            self._total_errors += 1

    # ==================================================================
    # PUBLIC API: Tier 1 Calculation
    # ==================================================================

    def calculate_tier1(
        self,
        animal_type: str,
        head_count: Any,
        region: str = "DEVELOPED",
        gwp_source: Optional[str] = None,
        ef_override: Optional[Any] = None,
        calculation_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Calculate CH4 emissions from enteric fermentation using IPCC Tier 1.

        Implements IPCC 2006 Vol 4 Ch 10, Equation 10.19:
            CH4_enteric = EF_T * N_T / 1000  (tonnes CH4/yr)
        where:
            EF_T = emission factor for animal type T (kg CH4/head/yr)
            N_T  = number of head of animal type T

        This method looks up EF from IPCC Table 10.11 based on animal type
        and regional classification, unless overridden by the caller.

        Args:
            animal_type: IPCC livestock category (e.g. ``"DAIRY_CATTLE"``).
                Must be a valid AnimalType value.
            head_count: Number of head (positive integer or Decimal).
            region: Regional classification for EF selection.  Must be a
                valid Region value.  Default ``"DEVELOPED"``.
            gwp_source: GWP report edition override (AR4/AR5/AR6/AR6_20YR).
                Default uses engine configuration.
            ef_override: Override emission factor in kg CH4/head/yr.
                If provided, bypasses table lookup.
            calculation_id: Optional external calculation identifier.

        Returns:
            Dictionary with keys:
                - calculation_id (str)
                - status (str): "SUCCESS" or "ERROR"
                - method (str): "TIER_1"
                - animal_type (str)
                - region (str)
                - head_count (str)
                - ef_used_kg_ch4_per_head_yr (str)
                - ef_source (str): "IPCC_TABLE_10.11" or "USER_OVERRIDE"
                - ch4_kg (str)
                - ch4_tonnes (str)
                - co2e_tonnes (str)
                - co2e_kg (str)
                - gwp_source (str)
                - gwp_ch4 (str)
                - calculation_trace (List[Dict])
                - provenance_hash (str)
                - processing_time_ms (float)
                - calculated_at (str)

        Example:
            >>> result = engine.calculate_tier1(
            ...     animal_type="DAIRY_CATTLE",
            ...     head_count=500,
            ...     region="DEVELOPED",
            ... )
            >>> result["ch4_tonnes"]
            '64.00000000'
        """
        self._increment_calculations()
        start_time = time.monotonic()
        calc_id = calculation_id or f"ag_ef_t1_{uuid4().hex[:12]}"
        trace_steps: List[TraceStep] = []
        step_num = 0

        try:
            # -- Parse and validate inputs ------------------------------------
            n_head = _D(head_count)
            animal_key = animal_type.upper()
            region_key = region.upper()
            gwp = (gwp_source or self._default_gwp_source).upper()

            errors = self._validate_tier1_inputs(
                n_head, animal_key, region_key,
            )
            if errors:
                raise ValueError(
                    f"Tier 1 input validation failed: {'; '.join(errors)}"
                )

            step_num += 1
            trace_steps.append(TraceStep(
                step_number=step_num,
                description="Input parameters validated",
                formula="N/A",
                inputs={
                    "animal_type": animal_key,
                    "head_count": str(n_head),
                    "region": region_key,
                    "gwp_source": gwp,
                },
                output="VALID",
                unit="N/A",
            ))

            # -- Resolve emission factor --------------------------------------
            if ef_override is not None:
                ef = _D(ef_override)
                ef_source = "USER_OVERRIDE"
            else:
                ef = self._lookup_enteric_ef(animal_key, region_key)
                ef_source = "IPCC_TABLE_10.11"

            step_num += 1
            trace_steps.append(TraceStep(
                step_number=step_num,
                description="Enteric fermentation emission factor resolved",
                formula="EF lookup from IPCC 2006 Table 10.11 or user override",
                inputs={
                    "animal_type": animal_key,
                    "region": region_key,
                    "ef_override": str(ef_override),
                },
                output=str(ef),
                unit="kg CH4/head/yr",
            ))

            # -- Calculate CH4 emissions (Eq 10.19) ---------------------------
            # CH4_kg = EF * N_head
            ch4_kg = _quantize(ef * n_head)

            step_num += 1
            trace_steps.append(TraceStep(
                step_number=step_num,
                description="CH4 mass from enteric fermentation (Eq 10.19)",
                formula="CH4_kg = EF * N_head",
                inputs={
                    "EF_kg_CH4_per_head_yr": str(ef),
                    "N_head": str(n_head),
                },
                output=str(ch4_kg),
                unit="kg CH4/yr",
            ))

            # -- Convert to tonnes --------------------------------------------
            ch4_tonnes = _quantize(ch4_kg * _KG_TO_TONNES)

            step_num += 1
            trace_steps.append(TraceStep(
                step_number=step_num,
                description="CH4 emissions converted to tonnes",
                formula="CH4_tonnes = CH4_kg / 1000",
                inputs={
                    "CH4_kg": str(ch4_kg),
                },
                output=str(ch4_tonnes),
                unit="tonnes CH4/yr",
            ))

            # -- GWP conversion -----------------------------------------------
            gwp_ch4 = self._resolve_gwp("CH4", gwp)
            co2e_tonnes = _quantize(ch4_tonnes * gwp_ch4)
            co2e_kg = _quantize(co2e_tonnes * _TONNES_TO_KG)

            step_num += 1
            trace_steps.append(TraceStep(
                step_number=step_num,
                description="GWP conversion to CO2e",
                formula="CO2e_tonnes = CH4_tonnes * GWP_CH4",
                inputs={
                    "CH4_tonnes": str(ch4_tonnes),
                    "GWP_CH4": str(gwp_ch4),
                    "gwp_source": gwp,
                },
                output=str(co2e_tonnes),
                unit="tonnes CO2e/yr",
            ))

            # -- Build result -------------------------------------------------
            elapsed_ms = (time.monotonic() - start_time) * 1000.0

            result: Dict[str, Any] = {
                "calculation_id": calc_id,
                "status": CalculationStatus.SUCCESS.value,
                "method": "TIER_1",
                "animal_type": animal_key,
                "region": region_key,
                "head_count": str(n_head),
                "ef_used_kg_ch4_per_head_yr": str(ef),
                "ef_source": ef_source,
                "ch4_kg": str(ch4_kg),
                "ch4_tonnes": str(ch4_tonnes),
                "co2e_tonnes": str(co2e_tonnes),
                "co2e_kg": str(co2e_kg),
                "gwp_source": gwp,
                "gwp_ch4": str(gwp_ch4),
                "calculation_trace": [ts.to_dict() for ts in trace_steps],
                "processing_time_ms": round(elapsed_ms, 3),
                "calculated_at": utcnow().isoformat(),
            }
            result["provenance_hash"] = _compute_hash(result)

            # -- Metrics ------------------------------------------------------
            self._record_metrics_tier1(
                animal_key, co2e_tonnes, ch4_tonnes, elapsed_ms,
            )

            # -- Provenance ---------------------------------------------------
            self._record_provenance(
                "calculate_tier1", calc_id,
                {
                    "animal_type": animal_key,
                    "head_count": str(n_head),
                    "ch4_tonnes": str(ch4_tonnes),
                    "co2e_tonnes": str(co2e_tonnes),
                    "hash": result["provenance_hash"],
                },
            )

            logger.info(
                "Tier1 enteric %s: %s head -> %s t CH4 / %s t CO2e "
                "(EF=%s) in %.1fms [%s]",
                animal_key, n_head, ch4_tonnes, co2e_tonnes,
                ef, elapsed_ms, calc_id,
            )
            return result

        except Exception as exc:
            self._increment_errors()
            elapsed_ms = (time.monotonic() - start_time) * 1000.0

            if _METRICS_AVAILABLE and _record_calculation_error is not None:
                _record_calculation_error("calculation_error")

            error_result: Dict[str, Any] = {
                "calculation_id": calc_id,
                "status": CalculationStatus.ERROR.value,
                "method": "TIER_1",
                "error": str(exc),
                "error_type": type(exc).__name__,
                "processing_time_ms": round(elapsed_ms, 3),
                "calculated_at": utcnow().isoformat(),
            }
            error_result["provenance_hash"] = _compute_hash(error_result)

            logger.error(
                "Tier1 enteric calculation failed [%s]: %s in %.1fms",
                calc_id, exc, elapsed_ms, exc_info=True,
            )
            return error_result

    # ==================================================================
    # PUBLIC API: Tier 2 Calculation
    # ==================================================================

    def calculate_tier2(
        self,
        animal_type: str,
        head_count: Any,
        body_weight_kg: Optional[Any] = None,
        milk_yield_kg_day: Any = 0,
        fat_pct: Any = 4.0,
        feed_digestibility_pct: Optional[Any] = None,
        ym_pct: Optional[Any] = None,
        activity_coefficient: Optional[Any] = None,
        activity_level: Optional[str] = None,
        is_pregnant: bool = False,
        weight_gain_kg_day: Any = 0,
        mature_body_weight_kg: Optional[Any] = None,
        c_value: Optional[Any] = None,
        hours_work: Any = 0,
        feed_quality: str = "MEDIUM",
        gwp_source: Optional[str] = None,
        calculation_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Calculate CH4 emissions from enteric fermentation using IPCC Tier 2.

        Implements IPCC 2006 Vol 4 Ch 10, Equations 10.3-10.21:
            EF = (GE * Ym/100 * 365) / 55.65  (kg CH4/head/yr)

        Where Gross Energy (GE) is derived from individual Net Energy
        components: maintenance (NE_m), activity (NE_a), lactation (NE_l),
        pregnancy (NE_p), growth (NE_g), and work (NE_work).

        REM and REG are the ratios of net energy available in the diet for
        maintenance and growth respectively.

        Args:
            animal_type: IPCC livestock category.
            head_count: Number of head.
            body_weight_kg: Average body weight in kg.  If None, default
                for animal type is used.
            milk_yield_kg_day: Milk production in kg/day.  Default 0.
            fat_pct: Milk fat content in percent.  Default 4.0%.
            feed_digestibility_pct: Digestible energy as % of GE (DE%).
                If None, default for animal type and feed quality.
            ym_pct: Methane conversion factor (% of GE).  If None, default
                for animal type and feed quality.
            activity_coefficient: Activity coefficient (Ca).  If None,
                derived from activity_level or defaults to 0.
            activity_level: Activity level (STALL, HOUSED, PASTURE,
                GRAZING_LARGE).  Used if activity_coefficient is None.
            is_pregnant: Whether the animal is pregnant (adds NE_p).
            weight_gain_kg_day: Average daily weight gain in kg/day.
                Default 0 (no growth).
            mature_body_weight_kg: Mature body weight in kg.  If None,
                default for animal type.
            c_value: Mature body weight fraction coefficient.  If None,
                default for animal type.
            hours_work: Average hours of draught work per day.  Default 0.
            feed_quality: Feed quality level for default DE% and Ym.
                Default ``"MEDIUM"``.
            gwp_source: GWP report edition override.
            calculation_id: Optional external calculation identifier.

        Returns:
            Dictionary with keys:
                - calculation_id, status, method ("TIER_2")
                - animal_type, head_count, body_weight_kg
                - net_energy_components (Dict): NE_m, NE_a, NE_l, NE_p,
                  NE_g, NE_work in MJ/day
                - REM, REG, DE_pct, GE_mj_day
                - ym_pct, ef_kg_ch4_per_head_yr
                - ch4_kg, ch4_tonnes, co2e_tonnes, co2e_kg
                - gwp_source, gwp_ch4
                - calculation_trace, provenance_hash, processing_time_ms

        Example:
            >>> result = engine.calculate_tier2(
            ...     animal_type="DAIRY_CATTLE",
            ...     head_count=200,
            ...     body_weight_kg=600,
            ...     milk_yield_kg_day=25,
            ...     fat_pct=4.0,
            ...     feed_digestibility_pct=65,
            ... )
            >>> result["ch4_tonnes"]
        """
        self._increment_calculations()
        start_time = time.monotonic()
        calc_id = calculation_id or f"ag_ef_t2_{uuid4().hex[:12]}"
        trace_steps: List[TraceStep] = []
        step_num = 0

        try:
            # -- Parse and validate inputs ------------------------------------
            n_head = _D(head_count)
            animal_key = animal_type.upper()
            fq_key = feed_quality.upper()
            gwp = (gwp_source or self._default_gwp_source).upper()

            errors = self._validate_tier2_inputs(
                n_head, animal_key, fq_key,
            )
            if errors:
                raise ValueError(
                    f"Tier 2 input validation failed: {'; '.join(errors)}"
                )

            # -- Resolve body weight ------------------------------------------
            bw = self._resolve_param(
                body_weight_kg,
                _DEFAULT_BODY_WEIGHTS.get(animal_key, Decimal("400")),
            )

            # -- Resolve other parameters -------------------------------------
            milk = _D(milk_yield_kg_day)
            fat = _D(fat_pct)
            wg = _D(weight_gain_kg_day)
            hw = _D(hours_work)

            de_pct = self._resolve_param(
                feed_digestibility_pct,
                self._get_default_de(animal_key, fq_key),
            )
            ym = self._resolve_param(
                ym_pct,
                self._get_default_ym(animal_key, fq_key),
            )
            bw_mature = self._resolve_param(
                mature_body_weight_kg,
                _DEFAULT_MATURE_WEIGHTS.get(animal_key, Decimal("550")),
            )
            c_val = self._resolve_param(
                c_value,
                _DEFAULT_C_VALUES.get(animal_key, Decimal("0.8")),
            )

            # Resolve activity coefficient
            ca = self._resolve_activity_coefficient(
                activity_coefficient, activity_level, animal_key,
            )

            step_num += 1
            trace_steps.append(TraceStep(
                step_number=step_num,
                description="Tier 2 input parameters validated and resolved",
                formula="Parameter lookup / user override",
                inputs={
                    "animal_type": animal_key,
                    "head_count": str(n_head),
                    "body_weight_kg": str(bw),
                    "milk_yield_kg_day": str(milk),
                    "fat_pct": str(fat),
                    "DE_pct": str(de_pct),
                    "Ym_pct": str(ym),
                    "Ca": str(ca),
                    "is_pregnant": str(is_pregnant),
                    "weight_gain_kg_day": str(wg),
                    "BW_mature_kg": str(bw_mature),
                    "C_value": str(c_val),
                    "hours_work": str(hw),
                    "feed_quality": fq_key,
                },
                output="PARAMETERS_RESOLVED",
                unit="N/A",
            ))

            # -- Step 1: NE_m (maintenance energy) ----------------------------
            # NE_m = Cfi * BW^0.75  (MJ/day)
            cfi = _MAINTENANCE_COEFFICIENTS.get(animal_key, Decimal("0.322"))
            bw_075 = self._decimal_power(bw, Decimal("0.75"))
            ne_m = _quantize(cfi * bw_075)

            step_num += 1
            trace_steps.append(TraceStep(
                step_number=step_num,
                description="Net energy for maintenance (NE_m)",
                formula="NE_m = Cfi * BW^0.75",
                inputs={
                    "Cfi": str(cfi),
                    "BW_kg": str(bw),
                    "BW^0.75": str(_quantize(bw_075)),
                },
                output=str(ne_m),
                unit="MJ/day",
            ))

            # -- Step 2: NE_a (activity energy) -------------------------------
            # NE_a = Ca * NE_m
            ne_a = _quantize(ca * ne_m)

            step_num += 1
            trace_steps.append(TraceStep(
                step_number=step_num,
                description="Net energy for activity (NE_a)",
                formula="NE_a = Ca * NE_m",
                inputs={
                    "Ca": str(ca),
                    "NE_m": str(ne_m),
                },
                output=str(ne_a),
                unit="MJ/day",
            ))

            # -- Step 3: NE_l (lactation energy) ------------------------------
            # NE_l = Milk * (1.47 + 0.40 * Fat)  (MJ/day)
            if milk > _ZERO:
                ne_l = _quantize(
                    milk * (Decimal("1.47") + Decimal("0.40") * fat)
                )
            else:
                ne_l = _ZERO

            step_num += 1
            trace_steps.append(TraceStep(
                step_number=step_num,
                description="Net energy for lactation (NE_l)",
                formula="NE_l = Milk * (1.47 + 0.40 * Fat)",
                inputs={
                    "Milk_kg_day": str(milk),
                    "Fat_pct": str(fat),
                },
                output=str(ne_l),
                unit="MJ/day",
            ))

            # -- Step 4: NE_work (work energy for draught animals) ------------
            # NE_work = 0.10 * NE_m * hours_work  (MJ/day)
            if hw > _ZERO:
                ne_work = _quantize(Decimal("0.10") * ne_m * hw)
            else:
                ne_work = _ZERO

            step_num += 1
            trace_steps.append(TraceStep(
                step_number=step_num,
                description="Net energy for draught work (NE_work)",
                formula="NE_work = 0.10 * NE_m * hours_work",
                inputs={
                    "NE_m": str(ne_m),
                    "hours_work": str(hw),
                },
                output=str(ne_work),
                unit="MJ/day",
            ))

            # -- Step 5: NE_p (pregnancy energy) ------------------------------
            # NE_p = 0.10 * NE_m  (for pregnant females)
            if is_pregnant:
                ne_p = _quantize(Decimal("0.10") * ne_m)
            else:
                ne_p = _ZERO

            step_num += 1
            trace_steps.append(TraceStep(
                step_number=step_num,
                description="Net energy for pregnancy (NE_p)",
                formula="NE_p = 0.10 * NE_m (if pregnant)",
                inputs={
                    "NE_m": str(ne_m),
                    "is_pregnant": str(is_pregnant),
                },
                output=str(ne_p),
                unit="MJ/day",
            ))

            # -- Step 6: NE_g (growth energy) ---------------------------------
            # NE_g = 22.02 * (BW / (C * BW_mature))^0.75 * WG^1.097
            if wg > _ZERO and bw_mature > _ZERO and c_val > _ZERO:
                bw_ratio = bw / (c_val * bw_mature)
                bw_ratio_075 = self._decimal_power(bw_ratio, Decimal("0.75"))
                wg_exp = self._decimal_power(wg, Decimal("1.097"))
                ne_g = _quantize(
                    Decimal("22.02") * bw_ratio_075 * wg_exp
                )
            else:
                ne_g = _ZERO

            step_num += 1
            trace_steps.append(TraceStep(
                step_number=step_num,
                description="Net energy for growth (NE_g)",
                formula="NE_g = 22.02 * (BW / (C * BW_mature))^0.75 * WG^1.097",
                inputs={
                    "BW_kg": str(bw),
                    "C": str(c_val),
                    "BW_mature_kg": str(bw_mature),
                    "WG_kg_day": str(wg),
                },
                output=str(ne_g),
                unit="MJ/day",
            ))

            # -- Step 7: REM and REG ------------------------------------------
            # REM = 1.123 - 4.092e-3*DE + 1.126e-5*DE^2 - 25.4/DE
            # REG = 1.164 - 5.160e-3*DE + 1.308e-5*DE^2 - 37.4/DE
            rem = self._calculate_rem(de_pct)
            reg = self._calculate_reg(de_pct)

            step_num += 1
            trace_steps.append(TraceStep(
                step_number=step_num,
                description="REM and REG ratios from digestibility",
                formula="REM = 1.123 - 4.092e-3*DE + 1.126e-5*DE^2 - 25.4/DE; "
                        "REG = 1.164 - 5.160e-3*DE + 1.308e-5*DE^2 - 37.4/DE",
                inputs={
                    "DE_pct": str(de_pct),
                },
                output=f"REM={rem}, REG={reg}",
                unit="dimensionless",
            ))

            # -- Step 8: Gross Energy intake (GE) -----------------------------
            # GE = [(NE_m + NE_a + NE_l + NE_work + NE_p) / REM
            #       + NE_g / REG] / (DE/100)
            ne_sum_maintenance = ne_m + ne_a + ne_l + ne_work + ne_p

            if rem > _ZERO and de_pct > _ZERO:
                ge_maint_part = ne_sum_maintenance / rem
            else:
                ge_maint_part = _ZERO

            if reg > _ZERO:
                ge_growth_part = ne_g / reg
            else:
                ge_growth_part = _ZERO

            de_fraction = de_pct / _HUNDRED

            if de_fraction > _ZERO:
                ge_mj_day = _quantize(
                    (ge_maint_part + ge_growth_part) / de_fraction
                )
            else:
                ge_mj_day = _ZERO

            step_num += 1
            trace_steps.append(TraceStep(
                step_number=step_num,
                description="Gross energy intake (GE)",
                formula="GE = [(NE_m + NE_a + NE_l + NE_work + NE_p) / REM "
                        "+ NE_g / REG] / (DE/100)",
                inputs={
                    "NE_m": str(ne_m),
                    "NE_a": str(ne_a),
                    "NE_l": str(ne_l),
                    "NE_work": str(ne_work),
                    "NE_p": str(ne_p),
                    "NE_g": str(ne_g),
                    "REM": str(rem),
                    "REG": str(reg),
                    "DE_pct": str(de_pct),
                },
                output=str(ge_mj_day),
                unit="MJ/day",
            ))

            # -- Step 9: Emission Factor (EF, Eq 10.21) -----------------------
            # EF = (GE * Ym/100 * 365) / 55.65  (kg CH4/head/yr)
            ym_fraction = ym / _HUNDRED
            ef_kg = _quantize(
                (ge_mj_day * ym_fraction * _DAYS_PER_YEAR)
                / _CH4_ENERGY_MJ_PER_KG
            )

            step_num += 1
            trace_steps.append(TraceStep(
                step_number=step_num,
                description="Enteric CH4 emission factor (IPCC Eq 10.21)",
                formula="EF = (GE * Ym/100 * 365) / 55.65",
                inputs={
                    "GE_mj_day": str(ge_mj_day),
                    "Ym_pct": str(ym),
                    "days_per_year": str(_DAYS_PER_YEAR),
                    "CH4_energy_mj_per_kg": str(_CH4_ENERGY_MJ_PER_KG),
                },
                output=str(ef_kg),
                unit="kg CH4/head/yr",
            ))

            # -- Step 10: Total CH4 emissions ---------------------------------
            ch4_kg = _quantize(ef_kg * n_head)
            ch4_tonnes = _quantize(ch4_kg * _KG_TO_TONNES)

            step_num += 1
            trace_steps.append(TraceStep(
                step_number=step_num,
                description="Total CH4 emissions from herd",
                formula="CH4_kg = EF * N_head; CH4_tonnes = CH4_kg / 1000",
                inputs={
                    "EF_kg": str(ef_kg),
                    "N_head": str(n_head),
                },
                output=f"CH4_kg={ch4_kg}, CH4_tonnes={ch4_tonnes}",
                unit="tonnes CH4/yr",
            ))

            # -- Step 11: GWP conversion --------------------------------------
            gwp_ch4 = self._resolve_gwp("CH4", gwp)
            co2e_tonnes = _quantize(ch4_tonnes * gwp_ch4)
            co2e_kg = _quantize(co2e_tonnes * _TONNES_TO_KG)

            step_num += 1
            trace_steps.append(TraceStep(
                step_number=step_num,
                description="GWP conversion to CO2e",
                formula="CO2e_tonnes = CH4_tonnes * GWP_CH4",
                inputs={
                    "CH4_tonnes": str(ch4_tonnes),
                    "GWP_CH4": str(gwp_ch4),
                },
                output=str(co2e_tonnes),
                unit="tonnes CO2e/yr",
            ))

            # -- Build result -------------------------------------------------
            elapsed_ms = (time.monotonic() - start_time) * 1000.0

            result: Dict[str, Any] = {
                "calculation_id": calc_id,
                "status": CalculationStatus.SUCCESS.value,
                "method": "TIER_2",
                "animal_type": animal_key,
                "head_count": str(n_head),
                "body_weight_kg": str(bw),
                "feed_quality": fq_key,
                "net_energy_components": {
                    "NE_m_mj_day": str(ne_m),
                    "NE_a_mj_day": str(ne_a),
                    "NE_l_mj_day": str(ne_l),
                    "NE_work_mj_day": str(ne_work),
                    "NE_p_mj_day": str(ne_p),
                    "NE_g_mj_day": str(ne_g),
                },
                "REM": str(rem),
                "REG": str(reg),
                "DE_pct": str(de_pct),
                "GE_mj_day": str(ge_mj_day),
                "ym_pct": str(ym),
                "ef_kg_ch4_per_head_yr": str(ef_kg),
                "ch4_kg": str(ch4_kg),
                "ch4_tonnes": str(ch4_tonnes),
                "co2e_tonnes": str(co2e_tonnes),
                "co2e_kg": str(co2e_kg),
                "gwp_source": gwp,
                "gwp_ch4": str(gwp_ch4),
                "calculation_trace": [ts.to_dict() for ts in trace_steps],
                "processing_time_ms": round(elapsed_ms, 3),
                "calculated_at": utcnow().isoformat(),
            }
            result["provenance_hash"] = _compute_hash(result)

            # -- Metrics ------------------------------------------------------
            self._record_metrics_tier2(
                animal_key, co2e_tonnes, ch4_tonnes, elapsed_ms,
            )

            # -- Provenance ---------------------------------------------------
            self._record_provenance(
                "calculate_tier2", calc_id,
                {
                    "animal_type": animal_key,
                    "head_count": str(n_head),
                    "ch4_tonnes": str(ch4_tonnes),
                    "co2e_tonnes": str(co2e_tonnes),
                    "ef_kg": str(ef_kg),
                    "hash": result["provenance_hash"],
                },
            )

            logger.info(
                "Tier2 enteric %s: %s head, BW=%s kg -> GE=%s MJ/d, "
                "EF=%s kg CH4/hd/yr, %s t CH4 / %s t CO2e in %.1fms [%s]",
                animal_key, n_head, bw, ge_mj_day,
                ef_kg, ch4_tonnes, co2e_tonnes, elapsed_ms, calc_id,
            )
            return result

        except Exception as exc:
            self._increment_errors()
            elapsed_ms = (time.monotonic() - start_time) * 1000.0

            if _METRICS_AVAILABLE and _record_calculation_error is not None:
                _record_calculation_error("calculation_error")

            error_result: Dict[str, Any] = {
                "calculation_id": calc_id,
                "status": CalculationStatus.ERROR.value,
                "method": "TIER_2",
                "error": str(exc),
                "error_type": type(exc).__name__,
                "processing_time_ms": round(elapsed_ms, 3),
                "calculated_at": utcnow().isoformat(),
            }
            error_result["provenance_hash"] = _compute_hash(error_result)

            logger.error(
                "Tier2 enteric calculation failed [%s]: %s in %.1fms",
                calc_id, exc, elapsed_ms, exc_info=True,
            )
            return error_result

    # ==================================================================
    # PUBLIC API: Herd Calculation (Multiple Animal Types)
    # ==================================================================

    def calculate_herd(
        self,
        herd_data: List[Dict[str, Any]],
        gwp_source: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Calculate total enteric CH4 emissions for a multi-type herd.

        Accepts a list of animal group definitions, each specifying the
        calculation method (``tier1`` or ``tier2``), animal type, head count,
        and any method-specific parameters.

        Args:
            herd_data: List of animal group parameter dictionaries.  Each
                must include:
                - ``animal_type`` (str): IPCC animal category.
                - ``head_count`` (int/float): Number of head.
                - ``method`` (str): ``"tier1"`` or ``"tier2"``.
                - (For tier2) body_weight_kg, milk_yield_kg_day, fat_pct,
                  feed_digestibility_pct, ym_pct, activity_coefficient,
                  is_pregnant, weight_gain_kg_day, etc.
                - (For tier1) region, ef_override.
            gwp_source: Optional GWP override applied to all groups.

        Returns:
            Dictionary with keys:
                - herd_id (str)
                - results (List[Dict]): Per-group calculation results
                - total_ch4_kg (str)
                - total_ch4_tonnes (str)
                - total_co2e_kg (str)
                - total_co2e_tonnes (str)
                - success_count (int)
                - failure_count (int)
                - total_count (int)
                - summary_by_animal_type (Dict): Aggregation by type
                - processing_time_ms (float)

        Example:
            >>> results = engine.calculate_herd([
            ...     {
            ...         "animal_type": "DAIRY_CATTLE",
            ...         "head_count": 200,
            ...         "method": "tier2",
            ...         "body_weight_kg": 600,
            ...         "milk_yield_kg_day": 25,
            ...     },
            ...     {
            ...         "animal_type": "SHEEP",
            ...         "head_count": 500,
            ...         "method": "tier1",
            ...         "region": "DEVELOPED",
            ...     },
            ... ])
            >>> results["success_count"]
            2
        """
        self._increment_batches()
        start_time = time.monotonic()
        herd_id = f"ag_ef_herd_{uuid4().hex[:12]}"

        results: List[Dict[str, Any]] = []
        total_ch4_kg = _ZERO
        total_ch4_tonnes = _ZERO
        total_co2e_kg = _ZERO
        total_co2e_tonnes = _ZERO
        success_count = 0
        failure_count = 0

        # Aggregation by animal type
        summary: Dict[str, Dict[str, Decimal]] = {}

        for idx, group in enumerate(herd_data):
            method = str(group.get("method", "tier1")).lower()

            # Apply GWP override
            if gwp_source and "gwp_source" not in group:
                group["gwp_source"] = gwp_source

            if method == "tier1":
                result = self._dispatch_tier1(group)
            elif method == "tier2":
                result = self._dispatch_tier2(group)
            else:
                result = {
                    "calculation_id": f"ag_ef_herd_{herd_id}_{idx}",
                    "status": CalculationStatus.ERROR.value,
                    "method": method.upper(),
                    "error": (
                        f"Unknown method: '{method}'. "
                        "Valid: tier1, tier2"
                    ),
                    "error_type": "ValueError",
                    "processing_time_ms": 0.0,
                    "calculated_at": utcnow().isoformat(),
                }
                result["provenance_hash"] = _compute_hash(result)

            results.append(result)

            if result.get("status") == CalculationStatus.SUCCESS.value:
                success_count += 1
                r_ch4_kg = _safe_decimal(result.get("ch4_kg"))
                r_ch4_t = _safe_decimal(result.get("ch4_tonnes"))
                r_co2e_kg = _safe_decimal(result.get("co2e_kg"))
                r_co2e_t = _safe_decimal(result.get("co2e_tonnes"))

                total_ch4_kg += r_ch4_kg
                total_ch4_tonnes += r_ch4_t
                total_co2e_kg += r_co2e_kg
                total_co2e_tonnes += r_co2e_t

                # Update summary by animal type
                atype = result.get("animal_type", "UNKNOWN")
                if atype not in summary:
                    summary[atype] = {
                        "count": _ZERO,
                        "head_count": _ZERO,
                        "ch4_tonnes": _ZERO,
                        "co2e_tonnes": _ZERO,
                    }
                summary[atype]["count"] += _ONE
                summary[atype]["head_count"] += _safe_decimal(
                    result.get("head_count")
                )
                summary[atype]["ch4_tonnes"] += r_ch4_t
                summary[atype]["co2e_tonnes"] += r_co2e_t
            else:
                failure_count += 1

        elapsed_ms = (time.monotonic() - start_time) * 1000.0

        # Serialize summary
        summary_serialized: Dict[str, Dict[str, str]] = {}
        for atype, data in summary.items():
            summary_serialized[atype] = {
                "count": str(data["count"]),
                "head_count": str(_quantize(data["head_count"])),
                "ch4_tonnes": str(_quantize(data["ch4_tonnes"])),
                "co2e_tonnes": str(_quantize(data["co2e_tonnes"])),
            }

        herd_result: Dict[str, Any] = {
            "herd_id": herd_id,
            "results": results,
            "total_ch4_kg": str(_quantize(total_ch4_kg)),
            "total_ch4_tonnes": str(_quantize(total_ch4_tonnes)),
            "total_co2e_kg": str(_quantize(total_co2e_kg)),
            "total_co2e_tonnes": str(_quantize(total_co2e_tonnes)),
            "success_count": success_count,
            "failure_count": failure_count,
            "total_count": len(herd_data),
            "summary_by_animal_type": summary_serialized,
            "processing_time_ms": round(elapsed_ms, 3),
        }

        logger.info(
            "Herd %s: %d/%d succeeded, total=%s t CH4 / %s t CO2e in %.1fms",
            herd_id, success_count, len(herd_data),
            _quantize(total_ch4_tonnes), _quantize(total_co2e_tonnes),
            elapsed_ms,
        )
        return herd_result

    # ==================================================================
    # PUBLIC API: Batch Processing (Multiple Farms/Herds)
    # ==================================================================

    def calculate_enteric_batch(
        self,
        requests: List[Dict[str, Any]],
        gwp_source: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Process multiple enteric fermentation calculations in batch.

        Each element in ``requests`` is either a single animal group
        (with ``animal_type``, ``head_count``, ``method``) or a herd
        (with ``herd_data`` list).  The dispatcher detects the format
        automatically.

        Args:
            requests: List of calculation parameter dictionaries.  Each
                must have either:
                - ``herd_data`` (List): For herd-level calculation.
                - ``animal_type`` + ``head_count`` + ``method``: For
                  single-group calculation.
            gwp_source: Optional GWP override applied to all calculations.

        Returns:
            Dictionary with keys:
                - batch_id (str)
                - results (List[Dict]): Individual calculation results
                - total_ch4_kg (str)
                - total_ch4_tonnes (str)
                - total_co2e_kg (str)
                - total_co2e_tonnes (str)
                - successful (int)
                - failed (int)
                - total_count (int)
                - processing_time_ms (float)

        Example:
            >>> batch = engine.calculate_enteric_batch([
            ...     {
            ...         "animal_type": "DAIRY_CATTLE",
            ...         "head_count": 500,
            ...         "method": "tier1",
            ...         "region": "DEVELOPED",
            ...     },
            ...     {
            ...         "herd_data": [
            ...             {"animal_type": "SHEEP", "head_count": 200,
            ...              "method": "tier1", "region": "DEVELOPED"},
            ...             {"animal_type": "GOATS", "head_count": 100,
            ...              "method": "tier1"},
            ...         ],
            ...     },
            ... ])
            >>> batch["successful"]
            2
        """
        self._increment_batches()
        start_time = time.monotonic()
        batch_id = f"ag_ef_batch_{uuid4().hex[:12]}"

        results: List[Dict[str, Any]] = []
        total_ch4_kg = _ZERO
        total_ch4_tonnes = _ZERO
        total_co2e_kg = _ZERO
        total_co2e_tonnes = _ZERO
        success_count = 0
        failure_count = 0

        for idx, req in enumerate(requests):
            # Apply GWP override
            if gwp_source and "gwp_source" not in req:
                req["gwp_source"] = gwp_source

            # Detect request type
            if "herd_data" in req:
                # Herd-level calculation
                result = self.calculate_herd(
                    herd_data=req["herd_data"],
                    gwp_source=req.get("gwp_source", gwp_source),
                )
                # Extract totals from herd result
                is_success = result.get("success_count", 0) > 0
                ch4_t = _safe_decimal(result.get("total_ch4_tonnes"))
                ch4_k = _safe_decimal(result.get("total_ch4_kg"))
                co2e_t = _safe_decimal(result.get("total_co2e_tonnes"))
                co2e_k = _safe_decimal(result.get("total_co2e_kg"))
            else:
                # Single-group calculation
                method = str(req.get("method", "tier1")).lower()
                if method == "tier1":
                    result = self._dispatch_tier1(req)
                elif method == "tier2":
                    result = self._dispatch_tier2(req)
                else:
                    result = {
                        "calculation_id": f"ag_ef_batch_{batch_id}_{idx}",
                        "status": CalculationStatus.ERROR.value,
                        "method": method.upper(),
                        "error": (
                            f"Unknown method: '{method}'. "
                            "Valid: tier1, tier2"
                        ),
                        "error_type": "ValueError",
                        "processing_time_ms": 0.0,
                        "calculated_at": utcnow().isoformat(),
                    }
                    result["provenance_hash"] = _compute_hash(result)

                is_success = (
                    result.get("status") == CalculationStatus.SUCCESS.value
                )
                ch4_t = _safe_decimal(result.get("ch4_tonnes"))
                ch4_k = _safe_decimal(result.get("ch4_kg"))
                co2e_t = _safe_decimal(result.get("co2e_tonnes"))
                co2e_k = _safe_decimal(result.get("co2e_kg"))

            results.append(result)

            if is_success:
                success_count += 1
                total_ch4_kg += ch4_k
                total_ch4_tonnes += ch4_t
                total_co2e_kg += co2e_k
                total_co2e_tonnes += co2e_t
            else:
                failure_count += 1

        elapsed_ms = (time.monotonic() - start_time) * 1000.0

        batch_result: Dict[str, Any] = {
            "batch_id": batch_id,
            "results": results,
            "total_ch4_kg": str(_quantize(total_ch4_kg)),
            "total_ch4_tonnes": str(_quantize(total_ch4_tonnes)),
            "total_co2e_kg": str(_quantize(total_co2e_kg)),
            "total_co2e_tonnes": str(_quantize(total_co2e_tonnes)),
            "successful": success_count,
            "failed": failure_count,
            "total_count": len(requests),
            "processing_time_ms": round(elapsed_ms, 3),
        }

        logger.info(
            "Enteric batch %s: %d/%d succeeded, total=%s t CH4 / "
            "%s t CO2e in %.1fms",
            batch_id, success_count, len(requests),
            _quantize(total_ch4_tonnes), _quantize(total_co2e_tonnes),
            elapsed_ms,
        )
        return batch_result

    # ==================================================================
    # PUBLIC API: Gross Energy Estimator (Standalone)
    # ==================================================================

    def estimate_gross_energy(
        self,
        animal_type: str,
        body_weight_kg: Optional[Any] = None,
        milk_yield_kg_day: Any = 0,
        fat_pct: Any = 4.0,
        feed_digestibility_pct: Optional[Any] = None,
        activity_coefficient: Optional[Any] = None,
        activity_level: Optional[str] = None,
        is_pregnant: bool = False,
        weight_gain_kg_day: Any = 0,
        mature_body_weight_kg: Optional[Any] = None,
        c_value: Optional[Any] = None,
        hours_work: Any = 0,
        feed_quality: str = "MEDIUM",
    ) -> Dict[str, Any]:
        """Estimate gross energy (GE) intake and all NE components.

        This is a standalone utility to calculate the Tier 2 energy
        components without performing the full CH4 emission calculation.
        Useful for feed planning and diet assessment.

        Args:
            animal_type: IPCC livestock category.
            body_weight_kg: Average body weight (kg).
            milk_yield_kg_day: Daily milk production (kg/day).
            fat_pct: Milk fat content (%).
            feed_digestibility_pct: Digestible energy (% of GE).
            activity_coefficient: Activity coefficient (Ca).
            activity_level: Activity level string.
            is_pregnant: Whether the animal is pregnant.
            weight_gain_kg_day: Daily weight gain (kg/day).
            mature_body_weight_kg: Mature body weight (kg).
            c_value: Mature body weight fraction coefficient.
            hours_work: Hours of draught work per day.
            feed_quality: Feed quality level for defaults.

        Returns:
            Dictionary with keys:
                - animal_type, body_weight_kg, feed_quality
                - NE_m, NE_a, NE_l, NE_work, NE_p, NE_g (all MJ/day)
                - NE_total_mj_day (sum of all NE components)
                - REM, REG, DE_pct
                - GE_mj_day (gross energy intake)
                - GE_mj_yr (annualized)
                - Cfi (maintenance coefficient)
                - Ca (activity coefficient)
                - provenance_hash, processing_time_ms

        Example:
            >>> ge = engine.estimate_gross_energy(
            ...     animal_type="DAIRY_CATTLE",
            ...     body_weight_kg=600,
            ...     milk_yield_kg_day=25,
            ... )
            >>> ge["GE_mj_day"]
        """
        self._increment_calculations()
        start_time = time.monotonic()

        try:
            animal_key = animal_type.upper()
            fq_key = feed_quality.upper()

            # Resolve parameters
            bw = self._resolve_param(
                body_weight_kg,
                _DEFAULT_BODY_WEIGHTS.get(animal_key, Decimal("400")),
            )
            milk = _D(milk_yield_kg_day)
            fat = _D(fat_pct)
            wg = _D(weight_gain_kg_day)
            hw = _D(hours_work)

            de_pct = self._resolve_param(
                feed_digestibility_pct,
                self._get_default_de(animal_key, fq_key),
            )
            bw_mature = self._resolve_param(
                mature_body_weight_kg,
                _DEFAULT_MATURE_WEIGHTS.get(animal_key, Decimal("550")),
            )
            c_val = self._resolve_param(
                c_value,
                _DEFAULT_C_VALUES.get(animal_key, Decimal("0.8")),
            )
            ca = self._resolve_activity_coefficient(
                activity_coefficient, activity_level, animal_key,
            )
            cfi = _MAINTENANCE_COEFFICIENTS.get(animal_key, Decimal("0.322"))

            # NE_m
            bw_075 = self._decimal_power(bw, Decimal("0.75"))
            ne_m = _quantize(cfi * bw_075)

            # NE_a
            ne_a = _quantize(ca * ne_m)

            # NE_l
            if milk > _ZERO:
                ne_l = _quantize(
                    milk * (Decimal("1.47") + Decimal("0.40") * fat)
                )
            else:
                ne_l = _ZERO

            # NE_work
            if hw > _ZERO:
                ne_work = _quantize(Decimal("0.10") * ne_m * hw)
            else:
                ne_work = _ZERO

            # NE_p
            ne_p = _quantize(Decimal("0.10") * ne_m) if is_pregnant else _ZERO

            # NE_g
            if wg > _ZERO and bw_mature > _ZERO and c_val > _ZERO:
                bw_ratio = bw / (c_val * bw_mature)
                bw_ratio_075 = self._decimal_power(bw_ratio, Decimal("0.75"))
                wg_exp = self._decimal_power(wg, Decimal("1.097"))
                ne_g = _quantize(Decimal("22.02") * bw_ratio_075 * wg_exp)
            else:
                ne_g = _ZERO

            ne_total = _quantize(ne_m + ne_a + ne_l + ne_work + ne_p + ne_g)

            # REM and REG
            rem = self._calculate_rem(de_pct)
            reg = self._calculate_reg(de_pct)

            # GE
            ne_sum_maint = ne_m + ne_a + ne_l + ne_work + ne_p
            ge_maint_part = ne_sum_maint / rem if rem > _ZERO else _ZERO
            ge_growth_part = ne_g / reg if reg > _ZERO else _ZERO
            de_frac = de_pct / _HUNDRED

            if de_frac > _ZERO:
                ge_mj_day = _quantize(
                    (ge_maint_part + ge_growth_part) / de_frac
                )
            else:
                ge_mj_day = _ZERO

            ge_mj_yr = _quantize(ge_mj_day * _DAYS_PER_YEAR)

            elapsed_ms = (time.monotonic() - start_time) * 1000.0

            result: Dict[str, Any] = {
                "animal_type": animal_key,
                "body_weight_kg": str(bw),
                "feed_quality": fq_key,
                "Cfi": str(cfi),
                "Ca": str(ca),
                "NE_m_mj_day": str(ne_m),
                "NE_a_mj_day": str(ne_a),
                "NE_l_mj_day": str(ne_l),
                "NE_work_mj_day": str(ne_work),
                "NE_p_mj_day": str(ne_p),
                "NE_g_mj_day": str(ne_g),
                "NE_total_mj_day": str(ne_total),
                "REM": str(rem),
                "REG": str(reg),
                "DE_pct": str(de_pct),
                "GE_mj_day": str(ge_mj_day),
                "GE_mj_yr": str(ge_mj_yr),
                "processing_time_ms": round(elapsed_ms, 3),
            }
            result["provenance_hash"] = _compute_hash(result)

            logger.info(
                "GE estimate %s: BW=%s kg -> GE=%s MJ/day, "
                "NE_total=%s MJ/day in %.1fms",
                animal_key, bw, ge_mj_day, ne_total, elapsed_ms,
            )
            return result

        except Exception as exc:
            self._increment_errors()
            elapsed_ms = (time.monotonic() - start_time) * 1000.0

            error_result: Dict[str, Any] = {
                "status": CalculationStatus.ERROR.value,
                "error": str(exc),
                "error_type": type(exc).__name__,
                "processing_time_ms": round(elapsed_ms, 3),
            }
            error_result["provenance_hash"] = _compute_hash(error_result)

            logger.error(
                "GE estimate failed: %s in %.1fms",
                exc, elapsed_ms, exc_info=True,
            )
            return error_result

    # ==================================================================
    # PUBLIC API: Default Ym Lookup
    # ==================================================================

    def get_ym_default(
        self,
        animal_type: str,
        feed_quality: str = "MEDIUM",
    ) -> Decimal:
        """Return the default methane conversion factor (Ym) for an animal type.

        Ym represents the fraction of gross energy intake converted to
        methane, expressed as a percentage.

        Args:
            animal_type: IPCC livestock category (e.g. ``"DAIRY_CATTLE"``).
            feed_quality: Feed quality level (``"LOW"``, ``"MEDIUM"``,
                ``"HIGH"``).  Default ``"MEDIUM"``.

        Returns:
            Ym as Decimal (percent of GE).

        Raises:
            KeyError: If animal_type or feed_quality is not recognized.

        Example:
            >>> engine.get_ym_default("DAIRY_CATTLE", "HIGH")
            Decimal('5.5')
        """
        key = animal_type.upper()
        fq = feed_quality.upper()

        ym_table = _DEFAULT_YM.get(key)
        if ym_table is None:
            raise KeyError(
                f"No default Ym for animal type '{animal_type}'. "
                f"Available: {sorted(_DEFAULT_YM.keys())}"
            )

        ym_val = ym_table.get(fq)
        if ym_val is None:
            raise KeyError(
                f"No Ym for feed quality '{feed_quality}' in animal type "
                f"'{animal_type}'. Available: {sorted(ym_table.keys())}"
            )

        return ym_val

    # ==================================================================
    # PUBLIC API: Direct EF Lookup
    # ==================================================================

    def get_enteric_ef(
        self,
        animal_type: str,
        region: str = "DEVELOPED",
    ) -> Decimal:
        """Return the Tier 1 enteric fermentation emission factor.

        Looks up the EF from IPCC Table 10.11 for the given animal type
        and regional classification.

        Args:
            animal_type: IPCC livestock category (e.g. ``"DAIRY_CATTLE"``).
            region: Regional classification (e.g. ``"DEVELOPED"``).
                Default ``"DEVELOPED"``.

        Returns:
            Emission factor in kg CH4/head/yr as Decimal.

        Raises:
            KeyError: If animal_type or region is not recognized.

        Example:
            >>> engine.get_enteric_ef("DAIRY_CATTLE", "DEVELOPED")
            Decimal('128')
        """
        return self._lookup_enteric_ef(
            animal_type.upper(), region.upper(),
        )

    # ==================================================================
    # PUBLIC API: Default Body Weight Lookup
    # ==================================================================

    def get_default_body_weight(self, animal_type: str) -> Decimal:
        """Return the default body weight for an animal type.

        Args:
            animal_type: IPCC livestock category.

        Returns:
            Default body weight in kg as Decimal.

        Raises:
            KeyError: If animal_type is not recognized.

        Example:
            >>> engine.get_default_body_weight("DAIRY_CATTLE")
            Decimal('600')
        """
        key = animal_type.upper()
        if key in _DEFAULT_BODY_WEIGHTS:
            return _DEFAULT_BODY_WEIGHTS[key]
        raise KeyError(
            f"No default body weight for animal type '{animal_type}'. "
            f"Available: {sorted(_DEFAULT_BODY_WEIGHTS.keys())}"
        )

    # ==================================================================
    # PUBLIC API: Default Feed Digestibility Lookup
    # ==================================================================

    def get_default_feed_digestibility(
        self,
        animal_type: str,
        feed_quality: str = "MEDIUM",
    ) -> Decimal:
        """Return the default feed digestibility (DE%) for an animal type.

        Args:
            animal_type: IPCC livestock category.
            feed_quality: Feed quality level.  Default ``"MEDIUM"``.

        Returns:
            Digestible energy as % of gross energy (Decimal).

        Raises:
            KeyError: If animal_type or feed_quality is not recognized.

        Example:
            >>> engine.get_default_feed_digestibility("DAIRY_CATTLE", "HIGH")
            Decimal('75')
        """
        key = animal_type.upper()
        fq = feed_quality.upper()

        de_table = _DEFAULT_FEED_DIGESTIBILITY.get(key)
        if de_table is None:
            raise KeyError(
                f"No default DE% for animal type '{animal_type}'. "
                f"Available: {sorted(_DEFAULT_FEED_DIGESTIBILITY.keys())}"
            )

        de_val = de_table.get(fq)
        if de_val is None:
            raise KeyError(
                f"No DE% for feed quality '{feed_quality}' in animal type "
                f"'{animal_type}'. Available: {sorted(de_table.keys())}"
            )

        return de_val

    # ==================================================================
    # PUBLIC API: Activity Coefficient Lookup
    # ==================================================================

    def get_activity_coefficient(self, activity_level: str) -> Decimal:
        """Return the activity coefficient (Ca) for a given activity level.

        Args:
            activity_level: Activity level (STALL, HOUSED, PASTURE,
                GRAZING_LARGE).

        Returns:
            Activity coefficient as Decimal (fraction of NE_m).

        Raises:
            KeyError: If activity_level is not recognized.

        Example:
            >>> engine.get_activity_coefficient("PASTURE")
            Decimal('0.17')
        """
        key = activity_level.upper()
        if key in _ACTIVITY_COEFFICIENTS:
            return _ACTIVITY_COEFFICIENTS[key]
        raise KeyError(
            f"No activity coefficient for level '{activity_level}'. "
            f"Available: {sorted(_ACTIVITY_COEFFICIENTS.keys())}"
        )

    # ==================================================================
    # PUBLIC API: Maintenance Coefficient Lookup
    # ==================================================================

    def get_maintenance_coefficient(self, animal_type: str) -> Decimal:
        """Return the maintenance energy coefficient (Cfi) for an animal type.

        Args:
            animal_type: IPCC livestock category.

        Returns:
            Cfi in MJ/day/kg BW^0.75 as Decimal.

        Raises:
            KeyError: If animal_type is not recognized.

        Example:
            >>> engine.get_maintenance_coefficient("DAIRY_CATTLE")
            Decimal('0.386')
        """
        key = animal_type.upper()
        if key in _MAINTENANCE_COEFFICIENTS:
            return _MAINTENANCE_COEFFICIENTS[key]
        raise KeyError(
            f"No maintenance coefficient for animal type '{animal_type}'. "
            f"Available: {sorted(_MAINTENANCE_COEFFICIENTS.keys())}"
        )

    # ==================================================================
    # PUBLIC API: Statistics
    # ==================================================================

    def get_statistics(self) -> Dict[str, Any]:
        """Return engine runtime statistics.

        Returns:
            Dictionary with total_calculations, total_batches,
            total_errors, created_at, uptime_seconds, and
            lists of available animal types, regions, and GWP sources.
        """
        with self._lock:
            now = utcnow()
            uptime = (now - self._created_at).total_seconds()
            return {
                "total_calculations": self._total_calculations,
                "total_batches": self._total_batches,
                "total_errors": self._total_errors,
                "created_at": self._created_at.isoformat(),
                "uptime_seconds": round(uptime, 1),
                "default_gwp_source": self._default_gwp_source,
                "available_animal_types": [
                    t.value for t in AnimalType
                ],
                "available_regions": [r.value for r in Region],
                "available_feed_qualities": [
                    q.value for q in FeedQuality
                ],
                "available_activity_levels": [
                    a.value for a in ActivityLevel
                ],
                "available_gwp_sources": sorted(_GWP_TABLE.keys()),
            }

    # ==================================================================
    # PRIVATE: Validation helpers
    # ==================================================================

    def _validate_tier1_inputs(
        self,
        head_count: Decimal,
        animal_type: str,
        region: str,
    ) -> List[str]:
        """Validate Tier 1 calculation inputs.

        Args:
            head_count: Number of head.
            animal_type: Animal type key.
            region: Region key.

        Returns:
            List of validation error messages (empty if valid).
        """
        errors: List[str] = []

        if head_count <= _ZERO:
            errors.append("head_count must be > 0")

        valid_types = {t.value for t in AnimalType}
        if animal_type not in valid_types:
            errors.append(
                f"Invalid animal_type '{animal_type}'. "
                f"Valid: {sorted(valid_types)}"
            )

        valid_regions = {r.value for r in Region}
        if region not in valid_regions:
            errors.append(
                f"Invalid region '{region}'. "
                f"Valid: {sorted(valid_regions)}"
            )

        return errors

    def _validate_tier2_inputs(
        self,
        head_count: Decimal,
        animal_type: str,
        feed_quality: str,
    ) -> List[str]:
        """Validate Tier 2 calculation inputs.

        Args:
            head_count: Number of head.
            animal_type: Animal type key.
            feed_quality: Feed quality key.

        Returns:
            List of validation error messages (empty if valid).
        """
        errors: List[str] = []

        if head_count <= _ZERO:
            errors.append("head_count must be > 0")

        valid_types = {t.value for t in AnimalType}
        if animal_type not in valid_types:
            errors.append(
                f"Invalid animal_type '{animal_type}'. "
                f"Valid: {sorted(valid_types)}"
            )

        valid_qualities = {q.value for q in FeedQuality}
        if feed_quality not in valid_qualities:
            errors.append(
                f"Invalid feed_quality '{feed_quality}'. "
                f"Valid: {sorted(valid_qualities)}"
            )

        return errors

    # ==================================================================
    # PRIVATE: Parameter resolution helpers
    # ==================================================================

    def _resolve_param(
        self,
        user_value: Optional[Any],
        default: Decimal,
    ) -> Decimal:
        """Resolve a parameter, using user override or default.

        Args:
            user_value: User-supplied value (may be None).
            default: Default Decimal value.

        Returns:
            Resolved Decimal value.
        """
        if user_value is not None:
            return _D(user_value)
        return default

    def _lookup_enteric_ef(
        self,
        animal_type: str,
        region: str,
    ) -> Decimal:
        """Look up Tier 1 enteric fermentation emission factor.

        Checks the external livestock database first, then falls back to
        the built-in IPCC Table 10.11.

        Args:
            animal_type: Animal type key.
            region: Region key.

        Returns:
            Emission factor in kg CH4/head/yr.

        Raises:
            KeyError: If animal_type or region is not found.
        """
        # Try external database first
        if self._livestock_db is not None:
            try:
                return _D(
                    self._livestock_db.get_enteric_ef(animal_type, region)
                )
            except (AttributeError, KeyError):
                pass

        # Built-in fallback
        ef_table = _ENTERIC_EF_TIER1.get(animal_type)
        if ef_table is None:
            raise KeyError(
                f"No enteric EF for animal type '{animal_type}'. "
                f"Available: {sorted(_ENTERIC_EF_TIER1.keys())}"
            )

        ef_val = ef_table.get(region)
        if ef_val is None:
            raise KeyError(
                f"No enteric EF for region '{region}' in animal type "
                f"'{animal_type}'. Available: {sorted(ef_table.keys())}"
            )

        return ef_val

    def _resolve_activity_coefficient(
        self,
        user_coefficient: Optional[Any],
        activity_level: Optional[str],
        animal_type: str,
    ) -> Decimal:
        """Resolve the activity coefficient (Ca).

        Priority: user_coefficient > activity_level lookup > default.

        Args:
            user_coefficient: User-supplied Ca value.
            activity_level: Activity level string.
            animal_type: Animal type (for default selection).

        Returns:
            Activity coefficient as Decimal.
        """
        if user_coefficient is not None:
            return _D(user_coefficient)

        if activity_level is not None:
            al_key = activity_level.upper()
            return _ACTIVITY_COEFFICIENTS.get(al_key, _ZERO)

        # Default: stall-fed for most, pasture for beef pasture
        if animal_type in ("BEEF_PASTURE", "SHEEP", "GOATS",
                           "LLAMAS", "ALPACAS", "DEER", "REINDEER"):
            return Decimal("0.17")

        return _ZERO

    def _get_default_de(
        self,
        animal_type: str,
        feed_quality: str,
    ) -> Decimal:
        """Get default feed digestibility (DE%) for animal and quality.

        Args:
            animal_type: Animal type key.
            feed_quality: Feed quality key.

        Returns:
            DE% as Decimal.
        """
        de_table = _DEFAULT_FEED_DIGESTIBILITY.get(animal_type, {})
        return de_table.get(feed_quality, Decimal("60"))

    def _get_default_ym(
        self,
        animal_type: str,
        feed_quality: str,
    ) -> Decimal:
        """Get default Ym (methane conversion factor) for animal and quality.

        Args:
            animal_type: Animal type key.
            feed_quality: Feed quality key.

        Returns:
            Ym as Decimal (percent of GE).
        """
        ym_table = _DEFAULT_YM.get(animal_type, {})
        return ym_table.get(feed_quality, Decimal("6.5"))

    # ==================================================================
    # PRIVATE: IPCC Formula helpers
    # ==================================================================

    def _calculate_rem(self, de_pct: Decimal) -> Decimal:
        """Calculate the ratio of NE available for maintenance (REM).

        IPCC 2006 Vol 4, Equation 10.14:
            REM = 1.123 - 4.092e-3*DE + 1.126e-5*DE^2 - 25.4/DE

        Args:
            de_pct: Digestible energy as percent of gross energy.

        Returns:
            REM as Decimal (dimensionless).
        """
        if de_pct <= _ZERO:
            return _ZERO

        de = de_pct
        de_sq = de * de

        rem = (
            Decimal("1.123")
            - Decimal("0.004092") * de
            + Decimal("0.00001126") * de_sq
            - Decimal("25.4") / de
        )
        return _quantize(rem)

    def _calculate_reg(self, de_pct: Decimal) -> Decimal:
        """Calculate the ratio of NE available for growth (REG).

        IPCC 2006 Vol 4, Equation 10.15:
            REG = 1.164 - 5.160e-3*DE + 1.308e-5*DE^2 - 37.4/DE

        Args:
            de_pct: Digestible energy as percent of gross energy.

        Returns:
            REG as Decimal (dimensionless).
        """
        if de_pct <= _ZERO:
            return _ZERO

        de = de_pct
        de_sq = de * de

        reg = (
            Decimal("1.164")
            - Decimal("0.005160") * de
            + Decimal("0.00001308") * de_sq
            - Decimal("37.4") / de
        )
        return _quantize(reg)

    def _decimal_power(self, base: Decimal, exponent: Decimal) -> Decimal:
        """Compute base^exponent using float for fractional exponents.

        Decimal does not natively support fractional powers, so we convert
        through float and back.  The result is quantized to 8 decimal places
        to maintain determinism.

        Args:
            base: Base value (must be positive).
            exponent: Exponent value.

        Returns:
            Quantized result of base^exponent.

        Raises:
            ValueError: If base is negative.
        """
        if base < _ZERO:
            raise ValueError(
                f"Cannot raise negative base {base} to power {exponent}"
            )
        if base == _ZERO:
            return _ZERO

        # Use Python float for fractional power
        result_float = float(base) ** float(exponent)
        return _quantize(_D(result_float))

    def _resolve_gwp(self, gas: str, gwp_source: str) -> Decimal:
        """Resolve GWP value from built-in table or database.

        Args:
            gas: Gas identifier (CH4, N2O, CO2).
            gwp_source: GWP report edition (AR4, AR5, AR6, AR6_20YR).

        Returns:
            GWP value as Decimal.

        Raises:
            KeyError: If GWP not found.
        """
        gas_key = gas.upper()
        source_key = gwp_source.upper()

        # Try external database first
        if self._livestock_db is not None:
            try:
                return self._livestock_db.get_gwp(gas_key, source_key)
            except (AttributeError, KeyError):
                pass

        # Built-in fallback
        gwp_table = _GWP_TABLE.get(source_key)
        if gwp_table is None:
            raise KeyError(
                f"Unknown GWP source '{gwp_source}'. "
                f"Available: {sorted(_GWP_TABLE.keys())}"
            )

        gwp_value = gwp_table.get(gas_key)
        if gwp_value is None:
            raise KeyError(
                f"No GWP for gas '{gas}' in source '{gwp_source}'. "
                f"Available gases: {sorted(gwp_table.keys())}"
            )

        return gwp_value

    # ==================================================================
    # PRIVATE: Batch dispatch helpers
    # ==================================================================

    def _dispatch_tier1(
        self, params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Extract parameters and call calculate_tier1.

        Args:
            params: Raw parameter dictionary from batch/herd.

        Returns:
            Tier 1 calculation result.
        """
        return self.calculate_tier1(
            animal_type=params.get("animal_type", "NON_DAIRY_CATTLE"),
            head_count=params.get("head_count", 0),
            region=params.get("region", "DEVELOPED"),
            gwp_source=params.get("gwp_source"),
            ef_override=params.get("ef_override"),
            calculation_id=params.get("calculation_id"),
        )

    def _dispatch_tier2(
        self, params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Extract parameters and call calculate_tier2.

        Args:
            params: Raw parameter dictionary from batch/herd.

        Returns:
            Tier 2 calculation result.
        """
        return self.calculate_tier2(
            animal_type=params.get("animal_type", "DAIRY_CATTLE"),
            head_count=params.get("head_count", 0),
            body_weight_kg=params.get("body_weight_kg"),
            milk_yield_kg_day=params.get("milk_yield_kg_day", 0),
            fat_pct=params.get("fat_pct", 4.0),
            feed_digestibility_pct=params.get("feed_digestibility_pct"),
            ym_pct=params.get("ym_pct"),
            activity_coefficient=params.get("activity_coefficient"),
            activity_level=params.get("activity_level"),
            is_pregnant=params.get("is_pregnant", False),
            weight_gain_kg_day=params.get("weight_gain_kg_day", 0),
            mature_body_weight_kg=params.get("mature_body_weight_kg"),
            c_value=params.get("c_value"),
            hours_work=params.get("hours_work", 0),
            feed_quality=params.get("feed_quality", "MEDIUM"),
            gwp_source=params.get("gwp_source"),
            calculation_id=params.get("calculation_id"),
        )

    # ==================================================================
    # PRIVATE: Metrics recording helpers
    # ==================================================================

    def _record_metrics_tier1(
        self,
        animal_type: str,
        co2e_tonnes: Decimal,
        ch4_tonnes: Decimal,
        elapsed_ms: float,
    ) -> None:
        """Record Prometheus metrics for a Tier 1 calculation.

        Args:
            animal_type: Animal type calculated.
            co2e_tonnes: Total CO2e in tonnes.
            ch4_tonnes: CH4 emitted in tonnes.
            elapsed_ms: Processing time in milliseconds.
        """
        if not _METRICS_AVAILABLE:
            return

        at_lower = animal_type.lower()
        if _record_calculation is not None:
            _record_calculation(
                "enteric_fermentation", "ipcc_tier1", at_lower,
            )
        if _observe_calculation_duration is not None:
            _observe_calculation_duration(
                "enteric_fermentation", "ipcc_tier1",
                elapsed_ms / 1000.0,
            )
        if _record_emissions is not None:
            _record_emissions(
                "CH4", "enteric_fermentation", float(co2e_tonnes),
            )
        if _record_enteric_calculation is not None:
            _record_enteric_calculation(at_lower, "ipcc_tier1")

    def _record_metrics_tier2(
        self,
        animal_type: str,
        co2e_tonnes: Decimal,
        ch4_tonnes: Decimal,
        elapsed_ms: float,
    ) -> None:
        """Record Prometheus metrics for a Tier 2 calculation.

        Args:
            animal_type: Animal type calculated.
            co2e_tonnes: Total CO2e in tonnes.
            ch4_tonnes: CH4 emitted in tonnes.
            elapsed_ms: Processing time in milliseconds.
        """
        if not _METRICS_AVAILABLE:
            return

        at_lower = animal_type.lower()
        if _record_calculation is not None:
            _record_calculation(
                "enteric_fermentation", "ipcc_tier2", at_lower,
            )
        if _observe_calculation_duration is not None:
            _observe_calculation_duration(
                "enteric_fermentation", "ipcc_tier2",
                elapsed_ms / 1000.0,
            )
        if _record_emissions is not None:
            _record_emissions(
                "CH4", "enteric_fermentation", float(co2e_tonnes),
            )
        if _record_enteric_calculation is not None:
            _record_enteric_calculation(at_lower, "ipcc_tier2")

    # ==================================================================
    # PRIVATE: Provenance recording
    # ==================================================================

    def _record_provenance(
        self,
        action: str,
        entity_id: str,
        data: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record an operation in the provenance tracker if available.

        Args:
            action: Action name (e.g. ``"calculate_tier1"``).
            entity_id: Entity identifier (calculation ID).
            data: Optional data dictionary.
        """
        if self._provenance is not None:
            try:
                self._provenance.record(
                    entity_type="enteric_fermentation",
                    action=action,
                    entity_id=entity_id,
                    data=data or {},
                )
            except Exception as exc:
                logger.debug(
                    "Provenance recording failed (non-critical): %s", exc,
                )
