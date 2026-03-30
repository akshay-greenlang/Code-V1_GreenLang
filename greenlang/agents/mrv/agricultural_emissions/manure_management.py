# -*- coding: utf-8 -*-
"""
ManureManagementEngine - CH4 and N2O from Animal Manure Management (Engine 3 of 7)

AGENT-MRV-008: Agricultural Emissions Agent

Core calculation engine implementing IPCC 2006 Vol 4 Chapter 10 (Sections 10.4
and 10.5) methods for estimating methane (CH4) and nitrous oxide (N2O) emissions
from animal manure management systems (AWMS).

Animal Waste Management Systems (AWMS) Supported:
    1.  Pasture/Range/Paddock        (PRP)
    2.  Daily Spread                 (DS)
    3.  Solid Storage                (SS)
    4.  Dry Lot                      (DL)
    5.  Liquid/Slurry (with crust)   (LS_CRUST)
    6.  Liquid/Slurry (without crust)(LS_NO_CRUST)
    7.  Uncovered Anaerobic Lagoon   (UAL)
    8.  Pit Storage (<1 month)       (PIT_LT1)
    9.  Pit Storage (>1 month)       (PIT_GT1)
    10. Anaerobic Digester           (AD)
    11. Burned for Fuel              (BF)
    12. Deep Bedding (no mixing)     (DB_NO_MIX)
    13. Deep Bedding (active mixing) (DB_ACTIVE)
    14. Composting (in-vessel)       (COMP_IV)
    15. Composting (static pile)     (COMP_SP)

Key Formulas (IPCC 2006 Vol 4 Ch 10):

    Manure CH4 (Eq 10.22-10.23):
        CH4_manure = SUM_T[ EF_T * N_T ] * 10^-6   (Gg CH4/yr)

        EF_T = VS_T * 365 * Bo_T * 0.67 * SUM_S[ MCF_S * MS_T,S ]
               (kg CH4/head/yr)

        Where:
            VS_T    = volatile solids excreted (kg VS/head/day) [Table 10A-4]
            Bo_T    = max CH4 producing capacity (m3 CH4/kg VS) [Table 10A-7]
            0.67    = density of CH4 at STP (kg/m3)
            MCF_S   = methane correction factor for AWMS S [Table 10.17]
            MS_T,S  = fraction of manure handled by AWMS S for animal T

    Manure N2O Direct (Eq 10.25):
        N2O_D = [ SUM_S( SUM_T( N_T * Nex_T * MS_T,S ) * EF3_S ) ] * 44/28

        Where:
            N_T     = population of animal type T
            Nex_T   = annual N excretion (kg N/head/yr) [Table 10.19]
            MS_T,S  = fraction of manure in AWMS S
            EF3_S   = N2O emission factor for AWMS S [Table 10.21]
            44/28   = molecular weight conversion N2O-N to N2O

    Indirect N2O from Manure (Eq 10.26-10.27):
        N2O_G = (SUM_T,S[N_T * Nex_T * MS_T,S * Frac_gas_S]) * EF4 * 44/28
        N2O_L = (SUM_T,S[N_T * Nex_T * MS_T,S * Frac_leach_S]) * EF5 * 44/28

All calculations use Python Decimal arithmetic with 8+ decimal places for
zero-hallucination determinism. Every calculation result includes a per-gas
breakdown, GWP-adjusted CO2e, processing time, and SHA-256 provenance hash.

Zero-Hallucination Guarantees:
    - All numeric calculations use Python Decimal.
    - No LLM calls in any calculation path.
    - Every calculation step is logged and traceable.
    - SHA-256 provenance hash for every result.
    - Identical inputs always produce identical outputs.

Thread Safety:
    Stateless per-calculation. Mutable counters protected by reentrant lock.

Example:
    >>> from greenlang.agents.mrv.agricultural_emissions.manure_management import (
    ...     ManureManagementEngine,
    ... )
    >>> engine = ManureManagementEngine()
    >>> result = engine.calculate_manure_ch4(
    ...     animal_type="DAIRY_CATTLE",
    ...     head_count=500,
    ...     mean_temp_c=20.0,
    ... )
    >>> assert result["status"] == "SUCCESS"

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

__all__ = ["ManureManagementEngine"]

# ---------------------------------------------------------------------------
# Conditional imports (graceful fallback)
# ---------------------------------------------------------------------------

try:
    from greenlang.agents.mrv.agricultural_emissions.config import (
        get_config as _get_config,
    )
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
        record_calculation as _record_calc_operation,
        record_emissions as _record_emissions,
        record_calculation_error as _record_calc_error,
        observe_calculation_duration as _observe_duration,
        record_manure_calculation as _record_manure_calc,
    )
    _METRICS_AVAILABLE = True
except ImportError:
    _METRICS_AVAILABLE = False
    _record_calc_operation = None  # type: ignore[assignment]
    _record_emissions = None  # type: ignore[assignment]
    _record_calc_error = None  # type: ignore[assignment]
    _observe_duration = None  # type: ignore[assignment]
    _record_manure_calc = None  # type: ignore[assignment]

try:
    from greenlang.agents.mrv.agricultural_emissions.database_engine import (
        DatabaseEngine as _DatabaseEngine,
    )
    _DB_AVAILABLE = True
except ImportError:
    _DB_AVAILABLE = False
    _DatabaseEngine = None  # type: ignore[assignment]

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
_THOUSAND = Decimal("1000")

#: N2O molecular weight ratio: 44/28 (N2O / 2N)
N2O_MOLECULAR_RATIO = Decimal("1.57142857")

#: CH4 density at STP (kg/m3)
CH4_DENSITY_STP = Decimal("0.67")

#: Days per year
DAYS_PER_YEAR = Decimal("365")

#: Conversion factor: kg to tonnes (10^-3)
KG_TO_TONNES = Decimal("0.001")

#: Conversion factor: Gg to tonnes (10^3)
GG_TO_TONNES = Decimal("1000")

#: Default indirect N2O emission factor for volatilization (EF4)
#: IPCC 2006 Vol 4 Ch 10, Table 10.22 default = 0.01 kg N2O-N/kg NH3-N+NOx-N
EF4_DEFAULT = Decimal("0.01")

#: Default indirect N2O emission factor for leaching/runoff (EF5)
#: IPCC 2006 Vol 4 Ch 10, Table 10.22 default = 0.0075 kg N2O-N/kg N leached
EF5_DEFAULT = Decimal("0.0075")

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
        default: Default if conversion fails.

    Returns:
        Decimal value.
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
    return value.quantize(_PRECISION, rounding=ROUND_HALF_UP)

# ===========================================================================
# GWP Values (fallback when database module is not available)
# ===========================================================================

GWP_VALUES: Dict[str, Dict[str, Decimal]] = {
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
        "CH4": Decimal("29.8"),
        "N2O": Decimal("273"),
    },
    "AR6_20YR": {
        "CO2": Decimal("1"),
        "CH4": Decimal("82.5"),
        "N2O": Decimal("273"),
    },
}

# ===========================================================================
# Enumerations
# ===========================================================================

class AnimalType(str, Enum):
    """Livestock categories with IPCC manure parameters."""

    DAIRY_CATTLE = "DAIRY_CATTLE"
    NON_DAIRY_CATTLE = "NON_DAIRY_CATTLE"
    BUFFALO = "BUFFALO"
    MARKET_SWINE = "MARKET_SWINE"
    BREEDING_SWINE = "BREEDING_SWINE"
    SHEEP = "SHEEP"
    GOATS = "GOATS"
    HORSES = "HORSES"
    MULES_ASSES = "MULES_ASSES"
    CAMELS = "CAMELS"
    LLAMAS = "LLAMAS"
    POULTRY_LAYERS = "POULTRY_LAYERS"
    POULTRY_BROILERS = "POULTRY_BROILERS"
    TURKEYS = "TURKEYS"
    DUCKS = "DUCKS"
    DEER = "DEER"
    RABBITS = "RABBITS"
    FUR_ANIMALS = "FUR_ANIMALS"
    OSTRICH = "OSTRICH"
    SWINE_NURSERY = "SWINE_NURSERY"

class AWMSType(str, Enum):
    """Animal Waste Management System types with IPCC MCF and EF3 values."""

    PASTURE_RANGE = "PASTURE_RANGE"
    DAILY_SPREAD = "DAILY_SPREAD"
    SOLID_STORAGE = "SOLID_STORAGE"
    DRY_LOT = "DRY_LOT"
    LIQUID_SLURRY_CRUST = "LIQUID_SLURRY_CRUST"
    LIQUID_SLURRY_NO_CRUST = "LIQUID_SLURRY_NO_CRUST"
    UNCOVERED_ANAEROBIC_LAGOON = "UNCOVERED_ANAEROBIC_LAGOON"
    PIT_STORAGE_LT1 = "PIT_STORAGE_LT1"
    PIT_STORAGE_GT1 = "PIT_STORAGE_GT1"
    ANAEROBIC_DIGESTER = "ANAEROBIC_DIGESTER"
    BURNED_FOR_FUEL = "BURNED_FOR_FUEL"
    DEEP_BEDDING_NO_MIX = "DEEP_BEDDING_NO_MIX"
    DEEP_BEDDING_ACTIVE = "DEEP_BEDDING_ACTIVE"
    COMPOSTING_IN_VESSEL = "COMPOSTING_IN_VESSEL"
    COMPOSTING_STATIC_PILE = "COMPOSTING_STATIC_PILE"

class TemperatureRange(str, Enum):
    """Temperature classification for MCF lookup."""

    COOL = "COOL"       # < 15 C annual mean
    TEMPERATE = "TEMPERATE"  # 15-25 C annual mean
    WARM = "WARM"       # > 25 C annual mean

class CalculationStatus(str, Enum):
    """Result status codes."""

    SUCCESS = "SUCCESS"
    PARTIAL = "PARTIAL"
    ERROR = "ERROR"
    VALIDATION_ERROR = "VALIDATION_ERROR"

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
# IPCC Default Parameters -- Volatile Solids (VS) by Animal Type
# Table 10A-4: Default values for VS daily excretion (kg VS/head/day)
# ===========================================================================

VS_DEFAULTS: Dict[str, Decimal] = {
    AnimalType.DAIRY_CATTLE.value: Decimal("5.40"),
    AnimalType.NON_DAIRY_CATTLE.value: Decimal("3.64"),
    AnimalType.BUFFALO.value: Decimal("3.90"),
    AnimalType.MARKET_SWINE.value: Decimal("0.50"),
    AnimalType.BREEDING_SWINE.value: Decimal("0.53"),
    AnimalType.SHEEP.value: Decimal("0.32"),
    AnimalType.GOATS.value: Decimal("0.35"),
    AnimalType.HORSES.value: Decimal("2.08"),
    AnimalType.MULES_ASSES.value: Decimal("1.10"),
    AnimalType.CAMELS.value: Decimal("1.86"),
    AnimalType.LLAMAS.value: Decimal("1.19"),
    AnimalType.POULTRY_LAYERS.value: Decimal("0.02"),
    AnimalType.POULTRY_BROILERS.value: Decimal("0.01"),
    AnimalType.TURKEYS.value: Decimal("0.04"),
    AnimalType.DUCKS.value: Decimal("0.01"),
    AnimalType.DEER.value: Decimal("0.88"),
    AnimalType.RABBITS.value: Decimal("0.06"),
    AnimalType.FUR_ANIMALS.value: Decimal("0.05"),
    AnimalType.OSTRICH.value: Decimal("0.56"),
    AnimalType.SWINE_NURSERY.value: Decimal("0.22"),
}

# ===========================================================================
# IPCC Default Parameters -- Bo (Maximum CH4 Producing Capacity)
# Table 10A-7: Default values for Bo (m3 CH4/kg VS)
# ===========================================================================

BO_DEFAULTS: Dict[str, Decimal] = {
    AnimalType.DAIRY_CATTLE.value: Decimal("0.24"),
    AnimalType.NON_DAIRY_CATTLE.value: Decimal("0.19"),
    AnimalType.BUFFALO.value: Decimal("0.10"),
    AnimalType.MARKET_SWINE.value: Decimal("0.48"),
    AnimalType.BREEDING_SWINE.value: Decimal("0.48"),
    AnimalType.SHEEP.value: Decimal("0.19"),
    AnimalType.GOATS.value: Decimal("0.18"),
    AnimalType.HORSES.value: Decimal("0.33"),
    AnimalType.MULES_ASSES.value: Decimal("0.33"),
    AnimalType.CAMELS.value: Decimal("0.19"),
    AnimalType.LLAMAS.value: Decimal("0.19"),
    AnimalType.POULTRY_LAYERS.value: Decimal("0.39"),
    AnimalType.POULTRY_BROILERS.value: Decimal("0.36"),
    AnimalType.TURKEYS.value: Decimal("0.36"),
    AnimalType.DUCKS.value: Decimal("0.36"),
    AnimalType.DEER.value: Decimal("0.19"),
    AnimalType.RABBITS.value: Decimal("0.32"),
    AnimalType.FUR_ANIMALS.value: Decimal("0.32"),
    AnimalType.OSTRICH.value: Decimal("0.36"),
    AnimalType.SWINE_NURSERY.value: Decimal("0.48"),
}

# ===========================================================================
# IPCC Default Parameters -- Methane Correction Factors (MCF)
# Table 10.17: MCF by AWMS type and temperature range
# Format: {AWMS: {COOL: x, TEMPERATE: y, WARM: z}}
# ===========================================================================

MCF_BY_AWMS: Dict[str, Dict[str, Decimal]] = {
    AWMSType.PASTURE_RANGE.value: {
        TemperatureRange.COOL.value: Decimal("0.01"),
        TemperatureRange.TEMPERATE.value: Decimal("0.015"),
        TemperatureRange.WARM.value: Decimal("0.02"),
    },
    AWMSType.DAILY_SPREAD.value: {
        TemperatureRange.COOL.value: Decimal("0.001"),
        TemperatureRange.TEMPERATE.value: Decimal("0.005"),
        TemperatureRange.WARM.value: Decimal("0.01"),
    },
    AWMSType.SOLID_STORAGE.value: {
        TemperatureRange.COOL.value: Decimal("0.02"),
        TemperatureRange.TEMPERATE.value: Decimal("0.04"),
        TemperatureRange.WARM.value: Decimal("0.05"),
    },
    AWMSType.DRY_LOT.value: {
        TemperatureRange.COOL.value: Decimal("0.01"),
        TemperatureRange.TEMPERATE.value: Decimal("0.015"),
        TemperatureRange.WARM.value: Decimal("0.02"),
    },
    AWMSType.LIQUID_SLURRY_CRUST.value: {
        TemperatureRange.COOL.value: Decimal("0.10"),
        TemperatureRange.TEMPERATE.value: Decimal("0.17"),
        TemperatureRange.WARM.value: Decimal("0.35"),
    },
    AWMSType.LIQUID_SLURRY_NO_CRUST.value: {
        TemperatureRange.COOL.value: Decimal("0.17"),
        TemperatureRange.TEMPERATE.value: Decimal("0.35"),
        TemperatureRange.WARM.value: Decimal("0.65"),
    },
    AWMSType.UNCOVERED_ANAEROBIC_LAGOON.value: {
        TemperatureRange.COOL.value: Decimal("0.66"),
        TemperatureRange.TEMPERATE.value: Decimal("0.74"),
        TemperatureRange.WARM.value: Decimal("0.80"),
    },
    AWMSType.PIT_STORAGE_LT1.value: {
        TemperatureRange.COOL.value: Decimal("0.03"),
        TemperatureRange.TEMPERATE.value: Decimal("0.03"),
        TemperatureRange.WARM.value: Decimal("0.05"),
    },
    AWMSType.PIT_STORAGE_GT1.value: {
        TemperatureRange.COOL.value: Decimal("0.17"),
        TemperatureRange.TEMPERATE.value: Decimal("0.35"),
        TemperatureRange.WARM.value: Decimal("0.65"),
    },
    AWMSType.ANAEROBIC_DIGESTER.value: {
        TemperatureRange.COOL.value: Decimal("0.10"),
        TemperatureRange.TEMPERATE.value: Decimal("0.10"),
        TemperatureRange.WARM.value: Decimal("0.10"),
    },
    AWMSType.BURNED_FOR_FUEL.value: {
        TemperatureRange.COOL.value: Decimal("0.10"),
        TemperatureRange.TEMPERATE.value: Decimal("0.10"),
        TemperatureRange.WARM.value: Decimal("0.10"),
    },
    AWMSType.DEEP_BEDDING_NO_MIX.value: {
        TemperatureRange.COOL.value: Decimal("0.17"),
        TemperatureRange.TEMPERATE.value: Decimal("0.17"),
        TemperatureRange.WARM.value: Decimal("0.44"),
    },
    AWMSType.DEEP_BEDDING_ACTIVE.value: {
        TemperatureRange.COOL.value: Decimal("0.44"),
        TemperatureRange.TEMPERATE.value: Decimal("0.44"),
        TemperatureRange.WARM.value: Decimal("0.44"),
    },
    AWMSType.COMPOSTING_IN_VESSEL.value: {
        TemperatureRange.COOL.value: Decimal("0.005"),
        TemperatureRange.TEMPERATE.value: Decimal("0.005"),
        TemperatureRange.WARM.value: Decimal("0.005"),
    },
    AWMSType.COMPOSTING_STATIC_PILE.value: {
        TemperatureRange.COOL.value: Decimal("0.005"),
        TemperatureRange.TEMPERATE.value: Decimal("0.005"),
        TemperatureRange.WARM.value: Decimal("0.005"),
    },
}

# ===========================================================================
# IPCC Default Parameters -- N Excretion Rates (Nex)
# Table 10.19: Default N excretion rates (kg N/head/yr)
# ===========================================================================

NEX_DEFAULTS: Dict[str, Decimal] = {
    AnimalType.DAIRY_CATTLE.value: Decimal("100.0"),
    AnimalType.NON_DAIRY_CATTLE.value: Decimal("60.0"),
    AnimalType.BUFFALO.value: Decimal("60.0"),
    AnimalType.MARKET_SWINE.value: Decimal("13.5"),
    AnimalType.BREEDING_SWINE.value: Decimal("20.0"),
    AnimalType.SHEEP.value: Decimal("12.0"),
    AnimalType.GOATS.value: Decimal("12.5"),
    AnimalType.HORSES.value: Decimal("40.0"),
    AnimalType.MULES_ASSES.value: Decimal("26.0"),
    AnimalType.CAMELS.value: Decimal("46.0"),
    AnimalType.LLAMAS.value: Decimal("28.0"),
    AnimalType.POULTRY_LAYERS.value: Decimal("0.82"),
    AnimalType.POULTRY_BROILERS.value: Decimal("0.47"),
    AnimalType.TURKEYS.value: Decimal("1.10"),
    AnimalType.DUCKS.value: Decimal("0.83"),
    AnimalType.DEER.value: Decimal("25.0"),
    AnimalType.RABBITS.value: Decimal("5.70"),
    AnimalType.FUR_ANIMALS.value: Decimal("5.70"),
    AnimalType.OSTRICH.value: Decimal("1.70"),
    AnimalType.SWINE_NURSERY.value: Decimal("7.0"),
}

# ===========================================================================
# IPCC Default Parameters -- Direct N2O Emission Factors (EF3)
# Table 10.21: EF3 by AWMS type (kg N2O-N/kg N)
# ===========================================================================

EF3_BY_AWMS: Dict[str, Decimal] = {
    AWMSType.PASTURE_RANGE.value: Decimal("0.02"),
    AWMSType.DAILY_SPREAD.value: Decimal("0.0"),
    AWMSType.SOLID_STORAGE.value: Decimal("0.005"),
    AWMSType.DRY_LOT.value: Decimal("0.02"),
    AWMSType.LIQUID_SLURRY_CRUST.value: Decimal("0.005"),
    AWMSType.LIQUID_SLURRY_NO_CRUST.value: Decimal("0.0"),
    AWMSType.UNCOVERED_ANAEROBIC_LAGOON.value: Decimal("0.0"),
    AWMSType.PIT_STORAGE_LT1.value: Decimal("0.002"),
    AWMSType.PIT_STORAGE_GT1.value: Decimal("0.002"),
    AWMSType.ANAEROBIC_DIGESTER.value: Decimal("0.0"),
    AWMSType.BURNED_FOR_FUEL.value: Decimal("0.0"),
    AWMSType.DEEP_BEDDING_NO_MIX.value: Decimal("0.01"),
    AWMSType.DEEP_BEDDING_ACTIVE.value: Decimal("0.07"),
    AWMSType.COMPOSTING_IN_VESSEL.value: Decimal("0.006"),
    AWMSType.COMPOSTING_STATIC_PILE.value: Decimal("0.006"),
}

# ===========================================================================
# IPCC Default Parameters -- Volatilization Fractions (Frac_MMS_gas)
# Fraction of managed manure N volatilised as NH3 and NOx
# ===========================================================================

FRAC_GAS_BY_AWMS: Dict[str, Decimal] = {
    AWMSType.PASTURE_RANGE.value: Decimal("0.0"),
    AWMSType.DAILY_SPREAD.value: Decimal("0.30"),
    AWMSType.SOLID_STORAGE.value: Decimal("0.30"),
    AWMSType.DRY_LOT.value: Decimal("0.20"),
    AWMSType.LIQUID_SLURRY_CRUST.value: Decimal("0.28"),
    AWMSType.LIQUID_SLURRY_NO_CRUST.value: Decimal("0.40"),
    AWMSType.UNCOVERED_ANAEROBIC_LAGOON.value: Decimal("0.35"),
    AWMSType.PIT_STORAGE_LT1.value: Decimal("0.25"),
    AWMSType.PIT_STORAGE_GT1.value: Decimal("0.28"),
    AWMSType.ANAEROBIC_DIGESTER.value: Decimal("0.15"),
    AWMSType.BURNED_FOR_FUEL.value: Decimal("0.0"),
    AWMSType.DEEP_BEDDING_NO_MIX.value: Decimal("0.25"),
    AWMSType.DEEP_BEDDING_ACTIVE.value: Decimal("0.30"),
    AWMSType.COMPOSTING_IN_VESSEL.value: Decimal("0.40"),
    AWMSType.COMPOSTING_STATIC_PILE.value: Decimal("0.45"),
}

# ===========================================================================
# IPCC Default Parameters -- Leaching Fractions (Frac_MMS_leach)
# Fraction of managed manure N lost through leaching/runoff
# ===========================================================================

FRAC_LEACH_BY_AWMS: Dict[str, Decimal] = {
    AWMSType.PASTURE_RANGE.value: Decimal("0.0"),
    AWMSType.DAILY_SPREAD.value: Decimal("0.0"),
    AWMSType.SOLID_STORAGE.value: Decimal("0.02"),
    AWMSType.DRY_LOT.value: Decimal("0.02"),
    AWMSType.LIQUID_SLURRY_CRUST.value: Decimal("0.0"),
    AWMSType.LIQUID_SLURRY_NO_CRUST.value: Decimal("0.0"),
    AWMSType.UNCOVERED_ANAEROBIC_LAGOON.value: Decimal("0.0"),
    AWMSType.PIT_STORAGE_LT1.value: Decimal("0.0"),
    AWMSType.PIT_STORAGE_GT1.value: Decimal("0.0"),
    AWMSType.ANAEROBIC_DIGESTER.value: Decimal("0.0"),
    AWMSType.BURNED_FOR_FUEL.value: Decimal("0.0"),
    AWMSType.DEEP_BEDDING_NO_MIX.value: Decimal("0.02"),
    AWMSType.DEEP_BEDDING_ACTIVE.value: Decimal("0.02"),
    AWMSType.COMPOSTING_IN_VESSEL.value: Decimal("0.0"),
    AWMSType.COMPOSTING_STATIC_PILE.value: Decimal("0.02"),
}

# ===========================================================================
# Default AWMS Allocations by Animal Type
# Fraction of manure handled by each AWMS (must sum to 1.0)
# ===========================================================================

DEFAULT_AWMS_ALLOCATIONS: Dict[str, Dict[str, Decimal]] = {
    AnimalType.DAIRY_CATTLE.value: {
        AWMSType.PASTURE_RANGE.value: Decimal("0.25"),
        AWMSType.LIQUID_SLURRY_NO_CRUST.value: Decimal("0.30"),
        AWMSType.SOLID_STORAGE.value: Decimal("0.25"),
        AWMSType.DAILY_SPREAD.value: Decimal("0.20"),
    },
    AnimalType.NON_DAIRY_CATTLE.value: {
        AWMSType.PASTURE_RANGE.value: Decimal("0.65"),
        AWMSType.DRY_LOT.value: Decimal("0.15"),
        AWMSType.SOLID_STORAGE.value: Decimal("0.10"),
        AWMSType.DAILY_SPREAD.value: Decimal("0.10"),
    },
    AnimalType.BUFFALO.value: {
        AWMSType.PASTURE_RANGE.value: Decimal("0.70"),
        AWMSType.DAILY_SPREAD.value: Decimal("0.15"),
        AWMSType.SOLID_STORAGE.value: Decimal("0.15"),
    },
    AnimalType.MARKET_SWINE.value: {
        AWMSType.LIQUID_SLURRY_NO_CRUST.value: Decimal("0.50"),
        AWMSType.PIT_STORAGE_GT1.value: Decimal("0.25"),
        AWMSType.SOLID_STORAGE.value: Decimal("0.15"),
        AWMSType.DEEP_BEDDING_NO_MIX.value: Decimal("0.10"),
    },
    AnimalType.BREEDING_SWINE.value: {
        AWMSType.LIQUID_SLURRY_NO_CRUST.value: Decimal("0.50"),
        AWMSType.PIT_STORAGE_GT1.value: Decimal("0.25"),
        AWMSType.SOLID_STORAGE.value: Decimal("0.15"),
        AWMSType.DEEP_BEDDING_NO_MIX.value: Decimal("0.10"),
    },
    AnimalType.SHEEP.value: {
        AWMSType.PASTURE_RANGE.value: Decimal("0.90"),
        AWMSType.SOLID_STORAGE.value: Decimal("0.10"),
    },
    AnimalType.GOATS.value: {
        AWMSType.PASTURE_RANGE.value: Decimal("0.90"),
        AWMSType.SOLID_STORAGE.value: Decimal("0.10"),
    },
    AnimalType.HORSES.value: {
        AWMSType.PASTURE_RANGE.value: Decimal("0.70"),
        AWMSType.SOLID_STORAGE.value: Decimal("0.20"),
        AWMSType.DRY_LOT.value: Decimal("0.10"),
    },
    AnimalType.MULES_ASSES.value: {
        AWMSType.PASTURE_RANGE.value: Decimal("0.80"),
        AWMSType.SOLID_STORAGE.value: Decimal("0.10"),
        AWMSType.DRY_LOT.value: Decimal("0.10"),
    },
    AnimalType.CAMELS.value: {
        AWMSType.PASTURE_RANGE.value: Decimal("0.85"),
        AWMSType.DRY_LOT.value: Decimal("0.15"),
    },
    AnimalType.LLAMAS.value: {
        AWMSType.PASTURE_RANGE.value: Decimal("0.85"),
        AWMSType.DRY_LOT.value: Decimal("0.15"),
    },
    AnimalType.POULTRY_LAYERS.value: {
        AWMSType.SOLID_STORAGE.value: Decimal("0.40"),
        AWMSType.DEEP_BEDDING_NO_MIX.value: Decimal("0.40"),
        AWMSType.DAILY_SPREAD.value: Decimal("0.20"),
    },
    AnimalType.POULTRY_BROILERS.value: {
        AWMSType.DEEP_BEDDING_NO_MIX.value: Decimal("0.80"),
        AWMSType.SOLID_STORAGE.value: Decimal("0.20"),
    },
    AnimalType.TURKEYS.value: {
        AWMSType.DEEP_BEDDING_NO_MIX.value: Decimal("0.80"),
        AWMSType.SOLID_STORAGE.value: Decimal("0.20"),
    },
    AnimalType.DUCKS.value: {
        AWMSType.PASTURE_RANGE.value: Decimal("0.50"),
        AWMSType.DEEP_BEDDING_NO_MIX.value: Decimal("0.30"),
        AWMSType.SOLID_STORAGE.value: Decimal("0.20"),
    },
    AnimalType.DEER.value: {
        AWMSType.PASTURE_RANGE.value: Decimal("0.90"),
        AWMSType.SOLID_STORAGE.value: Decimal("0.10"),
    },
    AnimalType.RABBITS.value: {
        AWMSType.SOLID_STORAGE.value: Decimal("0.60"),
        AWMSType.DEEP_BEDDING_NO_MIX.value: Decimal("0.30"),
        AWMSType.DAILY_SPREAD.value: Decimal("0.10"),
    },
    AnimalType.FUR_ANIMALS.value: {
        AWMSType.SOLID_STORAGE.value: Decimal("0.60"),
        AWMSType.DEEP_BEDDING_NO_MIX.value: Decimal("0.30"),
        AWMSType.DAILY_SPREAD.value: Decimal("0.10"),
    },
    AnimalType.OSTRICH.value: {
        AWMSType.PASTURE_RANGE.value: Decimal("0.60"),
        AWMSType.SOLID_STORAGE.value: Decimal("0.20"),
        AWMSType.DEEP_BEDDING_NO_MIX.value: Decimal("0.20"),
    },
    AnimalType.SWINE_NURSERY.value: {
        AWMSType.LIQUID_SLURRY_NO_CRUST.value: Decimal("0.60"),
        AWMSType.PIT_STORAGE_LT1.value: Decimal("0.20"),
        AWMSType.DEEP_BEDDING_NO_MIX.value: Decimal("0.20"),
    },
}

# ===========================================================================
# ManureManagementEngine
# ===========================================================================

class ManureManagementEngine:
    """Core calculation engine for CH4 and N2O emissions from animal manure
    management systems implementing IPCC 2006 Vol 4 Ch 10 methods.

    This engine performs deterministic Decimal arithmetic for all manure
    emission calculations including CH4 from volatile solids decomposition,
    direct N2O from manure nitrogen, and indirect N2O from volatilization
    and leaching pathways.

    Thread Safety:
        Per-calculation state is created fresh for each method call.
        Shared counters use a reentrant lock.

    Attributes:
        _config: Optional configuration dictionary.
        _lock: Reentrant lock protecting mutable counters.
        _total_calculations: Counter of total calculations performed.
        _total_errors: Counter of total calculation errors.
        _default_gwp_source: Default GWP assessment report for CO2e.

    Example:
        >>> engine = ManureManagementEngine()
        >>> result = engine.calculate_manure_ch4(
        ...     animal_type="DAIRY_CATTLE",
        ...     head_count=500,
        ...     mean_temp_c=20.0,
        ... )
        >>> assert result["status"] == "SUCCESS"
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize the ManureManagementEngine.

        Args:
            config: Optional configuration dictionary. Supports:
                - default_gwp_source (str): Default GWP report (AR4/AR5/AR6).
                - decimal_precision (int): Decimal places (default 8).
        """
        self._config = config or {}
        self._lock = threading.RLock()
        self._total_calculations: int = 0
        self._total_errors: int = 0
        self._created_at = utcnow()

        self._default_gwp_source: str = self._config.get(
            "default_gwp_source", "AR6",
        ).upper()

        logger.info(
            "ManureManagementEngine initialized: "
            "default_gwp=%s",
            self._default_gwp_source,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _increment_calculations(self) -> None:
        """Thread-safe increment of the calculation counter."""
        with self._lock:
            self._total_calculations += 1

    def _increment_errors(self) -> None:
        """Thread-safe increment of the error counter."""
        with self._lock:
            self._total_errors += 1

    def _resolve_gwp(
        self,
        gas: str,
        gwp_source: Optional[str] = None,
    ) -> Decimal:
        """Look up GWP value for a gas species.

        Args:
            gas: Gas species (CH4, CO2, N2O).
            gwp_source: IPCC assessment report. Defaults to engine default.

        Returns:
            GWP value as Decimal. Returns 1 for unknown gases.
        """
        source = (gwp_source or self._default_gwp_source).upper()
        if source in GWP_VALUES and gas.upper() in GWP_VALUES[source]:
            return GWP_VALUES[source][gas.upper()]
        logger.warning(
            "GWP lookup failed for %s/%s, using 1.0", gas, source,
        )
        return _ONE

    def _build_gas_result(
        self,
        gas: str,
        emission_kg: Decimal,
        gwp_source: Optional[str] = None,
    ) -> Dict[str, str]:
        """Build a per-gas emission result dictionary.

        Args:
            gas: Gas species identifier.
            emission_kg: Emission mass in kilograms.
            gwp_source: GWP assessment report for CO2e conversion.

        Returns:
            Dictionary with emission_kg, emission_tonnes, gwp, co2e values.
        """
        gwp = self._resolve_gwp(gas, gwp_source)
        emission_tonnes = _quantize(emission_kg * KG_TO_TONNES)
        co2e_kg = _quantize(emission_kg * gwp)
        co2e_tonnes = _quantize(co2e_kg * KG_TO_TONNES)

        return {
            "gas": gas,
            "emission_kg": str(_quantize(emission_kg)),
            "emission_tonnes": str(emission_tonnes),
            "gwp_value": str(gwp),
            "gwp_source": (gwp_source or self._default_gwp_source).upper(),
            "co2e_kg": str(co2e_kg),
            "co2e_tonnes": str(co2e_tonnes),
        }

    def _aggregate_gas_results(
        self,
        gas_results: List[Dict[str, str]],
    ) -> Tuple[Decimal, Decimal]:
        """Aggregate total CO2e across gas results.

        Args:
            gas_results: List of per-gas result dictionaries.

        Returns:
            Tuple of (total_co2e_kg, total_co2e_tonnes).
        """
        total_co2e_kg = _ZERO
        for gr in gas_results:
            total_co2e_kg += _D(gr["co2e_kg"])
        total_co2e_tonnes = _quantize(total_co2e_kg * KG_TO_TONNES)
        return _quantize(total_co2e_kg), total_co2e_tonnes

    def _record_metrics(
        self,
        emission_source: str,
        calculation_method: str,
        animal_type: str,
        elapsed_seconds: float,
    ) -> None:
        """Record Prometheus metrics if available.

        Args:
            emission_source: Emission source label.
            calculation_method: Calculation method label.
            animal_type: Animal type label.
            elapsed_seconds: Calculation duration in seconds.
        """
        if _METRICS_AVAILABLE and _record_calc_operation is not None:
            try:
                _record_calc_operation(
                    emission_source, calculation_method, animal_type,
                )
            except Exception:
                pass
        if _METRICS_AVAILABLE and _observe_duration is not None:
            try:
                _observe_duration(
                    emission_source, calculation_method, elapsed_seconds,
                )
            except Exception:
                pass

    def _classify_temperature(self, temp_c: Decimal) -> str:
        """Classify a mean annual temperature into IPCC range.

        Args:
            temp_c: Mean annual temperature in degrees Celsius.

        Returns:
            Temperature range string (COOL, TEMPERATE, WARM).
        """
        if temp_c < Decimal("15"):
            return TemperatureRange.COOL.value
        elif temp_c <= Decimal("25"):
            return TemperatureRange.TEMPERATE.value
        else:
            return TemperatureRange.WARM.value

    def _resolve_awms_allocation(
        self,
        animal_type: str,
        awms_allocation: Optional[Dict[str, Any]],
    ) -> Dict[str, Decimal]:
        """Resolve AWMS allocation fractions, using defaults if not provided.

        Args:
            animal_type: Animal type key for default lookup.
            awms_allocation: Optional user-supplied allocation.

        Returns:
            Dictionary mapping AWMS type to Decimal fraction.

        Raises:
            ValueError: If animal_type has no default and no allocation given.
        """
        if awms_allocation is not None:
            return {k: _D(v) for k, v in awms_allocation.items()}

        key = animal_type.upper()
        if key in DEFAULT_AWMS_ALLOCATIONS:
            return dict(DEFAULT_AWMS_ALLOCATIONS[key])

        raise ValueError(
            f"No default AWMS allocation for animal_type={animal_type}. "
            f"Please provide awms_allocation explicitly."
        )

    def _resolve_vs(
        self,
        animal_type: str,
        vs_kg_day: Optional[Any],
    ) -> Decimal:
        """Resolve volatile solids excreted per head per day.

        Args:
            animal_type: Animal type key.
            vs_kg_day: User-supplied VS, or None for IPCC default.

        Returns:
            VS value as Decimal (kg VS/head/day).
        """
        if vs_kg_day is not None:
            return _D(vs_kg_day)
        key = animal_type.upper()
        if key in VS_DEFAULTS:
            return VS_DEFAULTS[key]
        raise ValueError(
            f"No default VS for animal_type={animal_type}. "
            f"Please provide vs_kg_day explicitly."
        )

    def _resolve_bo(
        self,
        animal_type: str,
        bo: Optional[Any],
    ) -> Decimal:
        """Resolve maximum CH4 producing capacity (Bo).

        Args:
            animal_type: Animal type key.
            bo: User-supplied Bo (m3 CH4/kg VS), or None for default.

        Returns:
            Bo value as Decimal.
        """
        if bo is not None:
            return _D(bo)
        key = animal_type.upper()
        if key in BO_DEFAULTS:
            return BO_DEFAULTS[key]
        raise ValueError(
            f"No default Bo for animal_type={animal_type}. "
            f"Please provide bo explicitly."
        )

    def _resolve_nex(
        self,
        animal_type: str,
        nex_kg_yr: Optional[Any],
    ) -> Decimal:
        """Resolve annual nitrogen excretion rate.

        Args:
            animal_type: Animal type key.
            nex_kg_yr: User-supplied Nex (kg N/head/yr), or None for default.

        Returns:
            Nex value as Decimal.
        """
        if nex_kg_yr is not None:
            return _D(nex_kg_yr)
        key = animal_type.upper()
        if key in NEX_DEFAULTS:
            return NEX_DEFAULTS[key]
        raise ValueError(
            f"No default Nex for animal_type={animal_type}. "
            f"Please provide nex_kg_yr explicitly."
        )

    # ------------------------------------------------------------------
    # Static Data Lookups
    # ------------------------------------------------------------------

    def get_mcf_for_temperature(
        self,
        awms_type: str,
        temperature_c: Any,
    ) -> Decimal:
        """Get the temperature-adjusted MCF for an AWMS type.

        Performs linear interpolation between IPCC Table 10.17 temperature
        breakpoints (cool < 15 C, temperate 15-25 C, warm > 25 C) when the
        temperature falls within a range boundary zone (+/- 2 C).

        Args:
            awms_type: AWMS type string (see AWMSType enum).
            temperature_c: Mean annual temperature in degrees Celsius.

        Returns:
            MCF value as Decimal, interpolated for the given temperature.

        Raises:
            ValueError: If awms_type is not recognized.
        """
        key = awms_type.upper()
        if key not in MCF_BY_AWMS:
            valid = [a.value for a in AWMSType]
            raise ValueError(
                f"Unknown awms_type: {awms_type}. Valid types: {valid}"
            )

        temp = _D(temperature_c)
        mcf_map = MCF_BY_AWMS[key]
        cool_mcf = mcf_map[TemperatureRange.COOL.value]
        temp_mcf = mcf_map[TemperatureRange.TEMPERATE.value]
        warm_mcf = mcf_map[TemperatureRange.WARM.value]

        # Boundary interpolation zones: 13-17 C and 23-27 C
        lower_bound = Decimal("13")
        lower_mid = Decimal("15")
        upper_mid = Decimal("25")
        upper_bound = Decimal("27")

        if temp <= lower_bound:
            return cool_mcf
        elif temp < lower_mid:
            # Interpolate between cool and temperate
            frac = _quantize((temp - lower_bound) / (lower_mid - lower_bound))
            return _quantize(cool_mcf + frac * (temp_mcf - cool_mcf))
        elif temp <= upper_mid:
            # Within temperate range, interpolate linearly
            if temp_mcf == warm_mcf:
                return temp_mcf
            frac = _quantize(
                (temp - lower_mid) / (upper_mid - lower_mid)
            )
            return _quantize(temp_mcf + frac * (warm_mcf - temp_mcf))
        elif temp < upper_bound:
            # Interpolate between temperate and warm
            frac = _quantize(
                (temp - upper_mid) / (upper_bound - upper_mid)
            )
            return _quantize(temp_mcf + frac * (warm_mcf - temp_mcf))
        else:
            return warm_mcf

    def validate_awms_allocation(
        self,
        allocation: Dict[str, Any],
    ) -> bool:
        """Validate that AWMS allocation fractions sum to 1.0.

        Args:
            allocation: Dictionary mapping AWMS type names to fractions.

        Returns:
            True if all fractions are non-negative and sum to 1.0 (within
            tolerance of 0.001), False otherwise.
        """
        if not allocation:
            return False

        total = _ZERO
        for awms_name, fraction in allocation.items():
            frac = _safe_decimal(fraction, Decimal("-1"))
            if frac < _ZERO:
                logger.warning(
                    "AWMS allocation for %s has negative fraction: %s",
                    awms_name, fraction,
                )
                return False
            total += frac

        tolerance = Decimal("0.001")
        if abs(total - _ONE) > tolerance:
            logger.warning(
                "AWMS allocation sums to %s (expected 1.0, tolerance %s)",
                total, tolerance,
            )
            return False

        return True

    def get_vs_default(self, animal_type: str) -> Decimal:
        """Get IPCC default volatile solids excreted for an animal type.

        Args:
            animal_type: Animal type (see AnimalType enum). Case-insensitive.

        Returns:
            VS value in kg VS/head/day.

        Raises:
            ValueError: If animal_type is not recognized.
        """
        key = animal_type.upper()
        if key in VS_DEFAULTS:
            return VS_DEFAULTS[key]
        valid = [a.value for a in AnimalType]
        raise ValueError(
            f"Unknown animal_type: {animal_type}. Valid types: {valid}"
        )

    def get_bo_default(self, animal_type: str) -> Decimal:
        """Get IPCC default maximum CH4 producing capacity for an animal type.

        Args:
            animal_type: Animal type (see AnimalType enum). Case-insensitive.

        Returns:
            Bo value in m3 CH4/kg VS.

        Raises:
            ValueError: If animal_type is not recognized.
        """
        key = animal_type.upper()
        if key in BO_DEFAULTS:
            return BO_DEFAULTS[key]
        valid = [a.value for a in AnimalType]
        raise ValueError(
            f"Unknown animal_type: {animal_type}. Valid types: {valid}"
        )

    def get_nex_default(self, animal_type: str) -> Decimal:
        """Get IPCC default N excretion rate for an animal type.

        Args:
            animal_type: Animal type (see AnimalType enum). Case-insensitive.

        Returns:
            Nex value in kg N/head/yr.

        Raises:
            ValueError: If animal_type is not recognized.
        """
        key = animal_type.upper()
        if key in NEX_DEFAULTS:
            return NEX_DEFAULTS[key]
        valid = [a.value for a in AnimalType]
        raise ValueError(
            f"Unknown animal_type: {animal_type}. Valid types: {valid}"
        )

    def get_ef3_default(self, awms_type: str) -> Decimal:
        """Get IPCC default N2O emission factor (EF3) for an AWMS type.

        Args:
            awms_type: AWMS type (see AWMSType enum). Case-insensitive.

        Returns:
            EF3 value in kg N2O-N/kg N.

        Raises:
            ValueError: If awms_type is not recognized.
        """
        key = awms_type.upper()
        if key in EF3_BY_AWMS:
            return EF3_BY_AWMS[key]
        valid = [a.value for a in AWMSType]
        raise ValueError(
            f"Unknown awms_type: {awms_type}. Valid types: {valid}"
        )

    def get_default_awms_allocation(self, animal_type: str) -> Dict[str, Decimal]:
        """Get IPCC default AWMS allocation for an animal type.

        Args:
            animal_type: Animal type (see AnimalType enum). Case-insensitive.

        Returns:
            Dictionary mapping AWMS type to allocation fraction.

        Raises:
            ValueError: If animal_type has no default allocation.
        """
        key = animal_type.upper()
        if key in DEFAULT_AWMS_ALLOCATIONS:
            return dict(DEFAULT_AWMS_ALLOCATIONS[key])
        valid = list(DEFAULT_AWMS_ALLOCATIONS.keys())
        raise ValueError(
            f"No default AWMS allocation for animal_type: {animal_type}. "
            f"Available: {valid}"
        )

    # ==================================================================
    # Core Calculation: CH4 from Manure (Eq 10.22-10.23)
    # ==================================================================

    def calculate_manure_ch4(
        self,
        animal_type: str,
        head_count: Any,
        vs_kg_day: Optional[Any] = None,
        bo: Optional[Any] = None,
        awms_allocation: Optional[Dict[str, Any]] = None,
        mean_temp_c: Any = 15.0,
        gwp_source: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Calculate CH4 emissions from manure management.

        Implements IPCC 2006 Vol 4 Ch 10 Equations 10.22-10.23:
            EF_T = VS_T * 365 * Bo_T * 0.67 * SUM_S[MCF_S * MS_T,S]
            CH4 = EF_T * N_T  (kg CH4/yr)

        Args:
            animal_type: Animal type (see AnimalType enum). Case-insensitive.
            head_count: Number of animals (head).
            vs_kg_day: Volatile solids excreted (kg VS/head/day). Defaults
                to IPCC Table 10A-4 value for animal_type.
            bo: Maximum CH4 producing capacity (m3 CH4/kg VS). Defaults
                to IPCC Table 10A-7 value for animal_type.
            awms_allocation: Dictionary mapping AWMS type to fraction of
                manure handled. Must sum to 1.0. Defaults to IPCC defaults.
            mean_temp_c: Mean annual temperature in degrees Celsius for
                MCF temperature adjustment. Default 15.0.
            gwp_source: GWP assessment report (AR4/AR5/AR6/AR6_20YR).

        Returns:
            Calculation result dictionary with:
                - calculation_id, status, emissions_by_gas, total_co2e_kg,
                  total_co2e_tonnes, per_awms_breakdown, calculation_details,
                  trace_steps, processing_time_ms, provenance_hash.
        """
        self._increment_calculations()
        t0 = time.monotonic()
        trace_steps: List[TraceStep] = []
        step_num = 0
        calc_id = f"mm_ch4_{uuid4().hex[:12]}"

        try:
            # -- Parse and resolve inputs --------------------------------------
            at_key = animal_type.upper()
            n_head = _D(head_count)
            vs = self._resolve_vs(at_key, vs_kg_day)
            bo_val = self._resolve_bo(at_key, bo)
            allocation = self._resolve_awms_allocation(at_key, awms_allocation)
            temp = _D(mean_temp_c)

            # -- Validate inputs -----------------------------------------------
            errors: List[str] = []
            if n_head < _ZERO:
                errors.append("head_count must be >= 0")
            if vs < _ZERO:
                errors.append("vs_kg_day must be >= 0")
            if bo_val < _ZERO:
                errors.append("bo must be >= 0")
            if not self.validate_awms_allocation(allocation):
                errors.append(
                    "awms_allocation fractions must be non-negative and sum "
                    "to 1.0 (tolerance 0.001)"
                )
            if errors:
                return self._error_result(calc_id, errors, t0)

            # -- Step 1: Calculate annual VS per head --------------------------
            vs_annual = _quantize(vs * DAYS_PER_YEAR)

            step_num += 1
            trace_steps.append(TraceStep(
                step_number=step_num,
                description="Calculate annual volatile solids per head",
                formula="VS_annual = VS_day * 365",
                inputs={
                    "VS_day_kg": str(vs),
                    "days_per_year": str(DAYS_PER_YEAR),
                },
                output=str(vs_annual),
                unit="kg VS/head/yr",
            ))

            # -- Step 2: Calculate weighted MCF across AWMS --------------------
            per_awms_ch4: List[Dict[str, str]] = []
            weighted_mcf_sum = _ZERO
            temp_range = self._classify_temperature(temp)

            for awms_name, frac in allocation.items():
                awms_key = awms_name.upper()
                if awms_key not in MCF_BY_AWMS:
                    errors.append(f"Unknown AWMS type: {awms_name}")
                    continue
                mcf_val = self.get_mcf_for_temperature(awms_key, temp)
                product = _quantize(mcf_val * frac)
                weighted_mcf_sum += product

                per_awms_ch4.append({
                    "awms_type": awms_key,
                    "fraction": str(frac),
                    "mcf": str(mcf_val),
                    "mcf_x_fraction": str(product),
                })

            if errors:
                return self._error_result(calc_id, errors, t0)

            weighted_mcf_sum = _quantize(weighted_mcf_sum)

            step_num += 1
            trace_steps.append(TraceStep(
                step_number=step_num,
                description="Calculate weighted MCF across AWMS allocations",
                formula="SUM_S[MCF_S * MS_T,S]",
                inputs={
                    "temperature_c": str(temp),
                    "temperature_range": temp_range,
                    "awms_count": str(len(allocation)),
                },
                output=str(weighted_mcf_sum),
                unit="dimensionless",
            ))

            # -- Step 3: Calculate emission factor EF_T (kg CH4/head/yr) ------
            ef_t = _quantize(
                vs_annual * bo_val * CH4_DENSITY_STP * weighted_mcf_sum
            )

            step_num += 1
            trace_steps.append(TraceStep(
                step_number=step_num,
                description="Calculate CH4 emission factor per head",
                formula="EF_T = VS_annual * Bo * 0.67 * SUM[MCF*MS]",
                inputs={
                    "VS_annual_kg": str(vs_annual),
                    "Bo_m3_per_kg": str(bo_val),
                    "CH4_density": str(CH4_DENSITY_STP),
                    "weighted_mcf": str(weighted_mcf_sum),
                },
                output=str(ef_t),
                unit="kg CH4/head/yr",
            ))

            # -- Step 4: Calculate total CH4 from herd -------------------------
            ch4_total_kg = _quantize(ef_t * n_head)

            step_num += 1
            trace_steps.append(TraceStep(
                step_number=step_num,
                description="Calculate total CH4 emissions from herd",
                formula="CH4_total = EF_T * N_T",
                inputs={
                    "EF_T_kg_per_head": str(ef_t),
                    "head_count": str(n_head),
                },
                output=str(ch4_total_kg),
                unit="kg CH4/yr",
            ))

            # -- Step 5: Per-AWMS CH4 breakdown --------------------------------
            for awms_entry in per_awms_ch4:
                awms_frac = _D(awms_entry["fraction"])
                awms_ch4_kg = _quantize(ch4_total_kg * awms_frac)
                awms_entry["ch4_kg"] = str(awms_ch4_kg)
                awms_entry["ch4_tonnes"] = str(
                    _quantize(awms_ch4_kg * KG_TO_TONNES)
                )

            step_num += 1
            trace_steps.append(TraceStep(
                step_number=step_num,
                description="Allocate CH4 across AWMS for breakdown",
                formula="CH4_S = CH4_total * MS_T,S",
                inputs={
                    "CH4_total_kg": str(ch4_total_kg),
                    "awms_count": str(len(per_awms_ch4)),
                },
                output=str(ch4_total_kg),
                unit="kg CH4/yr",
            ))

            # -- Step 6: GWP conversion ----------------------------------------
            gas_result = self._build_gas_result("CH4", ch4_total_kg, gwp_source)
            gas_results = [gas_result]
            total_co2e_kg, total_co2e_tonnes = self._aggregate_gas_results(
                gas_results,
            )

            step_num += 1
            trace_steps.append(TraceStep(
                step_number=step_num,
                description="Apply GWP to convert CH4 to CO2e",
                formula="CO2e = CH4_kg * GWP_CH4",
                inputs={
                    "CH4_kg": str(ch4_total_kg),
                    "GWP_CH4": gas_result["gwp_value"],
                },
                output=gas_result["co2e_kg"],
                unit="kg CO2e",
            ))

            # -- Assemble result -----------------------------------------------
            elapsed_ms = (time.monotonic() - t0) * 1000.0
            ch4_total_tonnes = _quantize(ch4_total_kg * KG_TO_TONNES)

            result: Dict[str, Any] = {
                "calculation_id": calc_id,
                "status": CalculationStatus.SUCCESS.value,
                "method": "IPCC_MANURE_CH4",
                "ipcc_reference": "Vol 4, Ch 10, Eq 10.22-10.23",
                "animal_type": at_key,
                "head_count": str(n_head),
                "emissions_by_gas": gas_results,
                "total_ch4_kg": str(ch4_total_kg),
                "total_ch4_tonnes": str(ch4_total_tonnes),
                "total_co2e_kg": str(total_co2e_kg),
                "total_co2e_tonnes": str(total_co2e_tonnes),
                "per_awms_breakdown": per_awms_ch4,
                "calculation_details": {
                    "vs_kg_day": str(vs),
                    "vs_annual_kg": str(vs_annual),
                    "bo_m3_per_kg_vs": str(bo_val),
                    "ch4_density_stp": str(CH4_DENSITY_STP),
                    "mean_temp_c": str(temp),
                    "temperature_range": temp_range,
                    "weighted_mcf": str(weighted_mcf_sum),
                    "ef_t_kg_ch4_per_head_yr": str(ef_t),
                    "formula": (
                        "EF_T = VS*365*Bo*0.67*SUM[MCF_S*MS_T,S]; "
                        "CH4 = EF_T * N_T"
                    ),
                },
                "trace_steps": [s.to_dict() for s in trace_steps],
                "processing_time_ms": round(elapsed_ms, 3),
                "calculated_at": utcnow().isoformat(),
            }
            result["provenance_hash"] = _compute_hash(result)

            self._record_metrics(
                "manure_management", "ipcc_tier1",
                at_key.lower(), elapsed_ms / 1000.0,
            )

            if _METRICS_AVAILABLE and _record_manure_calc is not None:
                try:
                    primary_awms = max(
                        allocation, key=lambda k: allocation[k]
                    )
                    _record_manure_calc(at_key.lower(), primary_awms.lower())
                except Exception:
                    pass

            logger.info(
                "Manure CH4 calculation %s: %s head %s -> "
                "%s tonnes CH4, %s tonnes CO2e in %.1fms",
                calc_id, n_head, at_key,
                ch4_total_tonnes, total_co2e_tonnes, elapsed_ms,
            )
            return result

        except Exception as exc:
            self._increment_errors()
            return self._exception_result(calc_id, exc, t0)

    # ==================================================================
    # Core Calculation: Direct N2O from Manure (Eq 10.25)
    # ==================================================================

    def calculate_manure_n2o(
        self,
        animal_type: str,
        head_count: Any,
        nex_kg_yr: Optional[Any] = None,
        awms_allocation: Optional[Dict[str, Any]] = None,
        gwp_source: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Calculate direct N2O emissions from manure management.

        Implements IPCC 2006 Vol 4 Ch 10 Equation 10.25:
            N2O_D = [SUM_S(SUM_T(N_T * Nex_T * MS_T,S) * EF3_S)] * 44/28

        Args:
            animal_type: Animal type (see AnimalType enum). Case-insensitive.
            head_count: Number of animals (head).
            nex_kg_yr: Annual N excretion (kg N/head/yr). Defaults to IPCC
                Table 10.19 value for animal_type.
            awms_allocation: Dictionary mapping AWMS type to fraction of
                manure handled. Must sum to 1.0. Defaults to IPCC defaults.
            gwp_source: GWP assessment report (AR4/AR5/AR6/AR6_20YR).

        Returns:
            Calculation result dictionary with per-AWMS N2O breakdown,
            total_n2o_tonnes, total_co2e_tonnes, trace_steps, provenance hash.
        """
        self._increment_calculations()
        t0 = time.monotonic()
        trace_steps: List[TraceStep] = []
        step_num = 0
        calc_id = f"mm_n2o_{uuid4().hex[:12]}"

        try:
            # -- Parse and resolve inputs --------------------------------------
            at_key = animal_type.upper()
            n_head = _D(head_count)
            nex = self._resolve_nex(at_key, nex_kg_yr)
            allocation = self._resolve_awms_allocation(at_key, awms_allocation)

            # -- Validate inputs -----------------------------------------------
            errors: List[str] = []
            if n_head < _ZERO:
                errors.append("head_count must be >= 0")
            if nex < _ZERO:
                errors.append("nex_kg_yr must be >= 0")
            if not self.validate_awms_allocation(allocation):
                errors.append(
                    "awms_allocation fractions must be non-negative and sum "
                    "to 1.0 (tolerance 0.001)"
                )
            if errors:
                return self._error_result(calc_id, errors, t0)

            # -- Step 1: Total N excreted by herd ------------------------------
            total_n_excreted = _quantize(n_head * nex)

            step_num += 1
            trace_steps.append(TraceStep(
                step_number=step_num,
                description="Calculate total N excreted by herd",
                formula="N_total = N_T * Nex_T",
                inputs={
                    "head_count": str(n_head),
                    "Nex_kg_N_per_head_yr": str(nex),
                },
                output=str(total_n_excreted),
                unit="kg N/yr",
            ))

            # -- Step 2: N2O-N per AWMS ----------------------------------------
            per_awms_n2o: List[Dict[str, str]] = []
            total_n2o_n_kg = _ZERO

            for awms_name, frac in allocation.items():
                awms_key = awms_name.upper()
                if awms_key not in EF3_BY_AWMS:
                    errors.append(f"Unknown AWMS type for EF3: {awms_name}")
                    continue

                ef3 = EF3_BY_AWMS[awms_key]
                n_in_awms = _quantize(total_n_excreted * frac)
                n2o_n_kg = _quantize(n_in_awms * ef3)
                total_n2o_n_kg += n2o_n_kg

                per_awms_n2o.append({
                    "awms_type": awms_key,
                    "fraction": str(frac),
                    "ef3_kg_n2o_n_per_kg_n": str(ef3),
                    "n_in_awms_kg": str(n_in_awms),
                    "n2o_n_kg": str(n2o_n_kg),
                })

            if errors:
                return self._error_result(calc_id, errors, t0)

            total_n2o_n_kg = _quantize(total_n2o_n_kg)

            step_num += 1
            trace_steps.append(TraceStep(
                step_number=step_num,
                description="Calculate N2O-N emissions per AWMS using EF3",
                formula="N2O-N_S = N_total * MS_T,S * EF3_S",
                inputs={
                    "total_n_excreted_kg": str(total_n_excreted),
                    "awms_count": str(len(allocation)),
                },
                output=str(total_n2o_n_kg),
                unit="kg N2O-N/yr",
            ))

            # -- Step 3: Convert N2O-N to N2O (44/28 ratio) -------------------
            total_n2o_kg = _quantize(total_n2o_n_kg * N2O_MOLECULAR_RATIO)

            step_num += 1
            trace_steps.append(TraceStep(
                step_number=step_num,
                description="Convert N2O-N to N2O using molecular weight ratio",
                formula="N2O = N2O-N * 44/28",
                inputs={
                    "total_n2o_n_kg": str(total_n2o_n_kg),
                    "ratio_44_28": str(N2O_MOLECULAR_RATIO),
                },
                output=str(total_n2o_kg),
                unit="kg N2O/yr",
            ))

            # Populate per-AWMS N2O values
            for entry in per_awms_n2o:
                n2o_n = _D(entry["n2o_n_kg"])
                entry["n2o_kg"] = str(_quantize(n2o_n * N2O_MOLECULAR_RATIO))
                entry["n2o_tonnes"] = str(
                    _quantize(
                        _D(entry["n2o_kg"]) * KG_TO_TONNES
                    )
                )

            # -- Step 4: GWP conversion ----------------------------------------
            gas_result = self._build_gas_result("N2O", total_n2o_kg, gwp_source)
            gas_results = [gas_result]
            total_co2e_kg, total_co2e_tonnes = self._aggregate_gas_results(
                gas_results,
            )

            step_num += 1
            trace_steps.append(TraceStep(
                step_number=step_num,
                description="Apply GWP to convert N2O to CO2e",
                formula="CO2e = N2O_kg * GWP_N2O",
                inputs={
                    "N2O_kg": str(total_n2o_kg),
                    "GWP_N2O": gas_result["gwp_value"],
                },
                output=gas_result["co2e_kg"],
                unit="kg CO2e",
            ))

            # -- Assemble result -----------------------------------------------
            elapsed_ms = (time.monotonic() - t0) * 1000.0
            n2o_total_tonnes = _quantize(total_n2o_kg * KG_TO_TONNES)

            result: Dict[str, Any] = {
                "calculation_id": calc_id,
                "status": CalculationStatus.SUCCESS.value,
                "method": "IPCC_MANURE_N2O_DIRECT",
                "ipcc_reference": "Vol 4, Ch 10, Eq 10.25",
                "animal_type": at_key,
                "head_count": str(n_head),
                "emissions_by_gas": gas_results,
                "total_n2o_n_kg": str(total_n2o_n_kg),
                "total_n2o_kg": str(total_n2o_kg),
                "total_n2o_tonnes": str(n2o_total_tonnes),
                "total_co2e_kg": str(total_co2e_kg),
                "total_co2e_tonnes": str(total_co2e_tonnes),
                "per_awms_breakdown": per_awms_n2o,
                "calculation_details": {
                    "nex_kg_n_per_head_yr": str(nex),
                    "total_n_excreted_kg": str(total_n_excreted),
                    "n2o_molecular_ratio": str(N2O_MOLECULAR_RATIO),
                    "formula": (
                        "N2O_D = [SUM_S(N_T * Nex_T * MS_T,S * EF3_S)] "
                        "* 44/28"
                    ),
                },
                "trace_steps": [s.to_dict() for s in trace_steps],
                "processing_time_ms": round(elapsed_ms, 3),
                "calculated_at": utcnow().isoformat(),
            }
            result["provenance_hash"] = _compute_hash(result)

            self._record_metrics(
                "manure_management", "ipcc_tier1",
                at_key.lower(), elapsed_ms / 1000.0,
            )

            logger.info(
                "Manure N2O (direct) calculation %s: %s head %s -> "
                "%s tonnes N2O, %s tonnes CO2e in %.1fms",
                calc_id, n_head, at_key,
                n2o_total_tonnes, total_co2e_tonnes, elapsed_ms,
            )
            return result

        except Exception as exc:
            self._increment_errors()
            return self._exception_result(calc_id, exc, t0)

    # ==================================================================
    # Indirect N2O from Manure (Eq 10.26-10.27)
    # ==================================================================

    def calculate_indirect_n2o(
        self,
        animal_type: str,
        head_count: Any,
        nex_kg_yr: Optional[Any] = None,
        awms_allocation: Optional[Dict[str, Any]] = None,
        ef4: Optional[Any] = None,
        ef5: Optional[Any] = None,
        gwp_source: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Calculate indirect N2O emissions from manure (volatilization + leaching).

        Implements IPCC 2006 Vol 4 Ch 10 Equations 10.26-10.27:
            N2O_G = SUM_T,S[N_T * Nex_T * MS_T,S * Frac_gas_S] * EF4 * 44/28
            N2O_L = SUM_T,S[N_T * Nex_T * MS_T,S * Frac_leach_S] * EF5 * 44/28

        Args:
            animal_type: Animal type (see AnimalType enum). Case-insensitive.
            head_count: Number of animals (head).
            nex_kg_yr: Annual N excretion (kg N/head/yr). Defaults to IPCC
                Table 10.19 value for animal_type.
            awms_allocation: Dictionary mapping AWMS type to fraction of
                manure handled. Must sum to 1.0.
            ef4: Indirect N2O EF for volatilization (kg N2O-N/kg NH3-N+NOx-N).
                Defaults to 0.01 (IPCC 2006).
            ef5: Indirect N2O EF for leaching (kg N2O-N/kg N leached).
                Defaults to 0.0075 (IPCC 2006).
            gwp_source: GWP assessment report (AR4/AR5/AR6/AR6_20YR).

        Returns:
            Calculation result dictionary with n2o_volatilization,
            n2o_leaching, total_n2o_indirect, trace_steps, provenance hash.
        """
        self._increment_calculations()
        t0 = time.monotonic()
        trace_steps: List[TraceStep] = []
        step_num = 0
        calc_id = f"mm_n2o_ind_{uuid4().hex[:12]}"

        try:
            # -- Parse and resolve inputs --------------------------------------
            at_key = animal_type.upper()
            n_head = _D(head_count)
            nex = self._resolve_nex(at_key, nex_kg_yr)
            allocation = self._resolve_awms_allocation(at_key, awms_allocation)
            ef4_val = _D(ef4) if ef4 is not None else EF4_DEFAULT
            ef5_val = _D(ef5) if ef5 is not None else EF5_DEFAULT

            # -- Validate inputs -----------------------------------------------
            errors: List[str] = []
            if n_head < _ZERO:
                errors.append("head_count must be >= 0")
            if nex < _ZERO:
                errors.append("nex_kg_yr must be >= 0")
            if not self.validate_awms_allocation(allocation):
                errors.append(
                    "awms_allocation fractions must be non-negative and sum "
                    "to 1.0 (tolerance 0.001)"
                )
            if ef4_val < _ZERO:
                errors.append("ef4 must be >= 0")
            if ef5_val < _ZERO:
                errors.append("ef5 must be >= 0")
            if errors:
                return self._error_result(calc_id, errors, t0)

            # -- Step 1: Total N excreted by herd ------------------------------
            total_n_excreted = _quantize(n_head * nex)

            step_num += 1
            trace_steps.append(TraceStep(
                step_number=step_num,
                description="Calculate total N excreted by herd",
                formula="N_total = N_T * Nex_T",
                inputs={
                    "head_count": str(n_head),
                    "Nex_kg_N_per_head_yr": str(nex),
                },
                output=str(total_n_excreted),
                unit="kg N/yr",
            ))

            # -- Step 2: Volatilization pathway (Eq 10.26) ---------------------
            per_awms_vol: List[Dict[str, str]] = []
            total_n_volatilized = _ZERO

            for awms_name, frac in allocation.items():
                awms_key = awms_name.upper()
                if awms_key not in FRAC_GAS_BY_AWMS:
                    errors.append(
                        f"Unknown AWMS type for Frac_gas: {awms_name}"
                    )
                    continue

                frac_gas = FRAC_GAS_BY_AWMS[awms_key]
                n_in_awms = _quantize(total_n_excreted * frac)
                n_vol = _quantize(n_in_awms * frac_gas)
                total_n_volatilized += n_vol

                per_awms_vol.append({
                    "awms_type": awms_key,
                    "fraction": str(frac),
                    "frac_gas": str(frac_gas),
                    "n_in_awms_kg": str(n_in_awms),
                    "n_volatilized_kg": str(n_vol),
                })

            if errors:
                return self._error_result(calc_id, errors, t0)

            total_n_volatilized = _quantize(total_n_volatilized)

            step_num += 1
            trace_steps.append(TraceStep(
                step_number=step_num,
                description="Calculate total N volatilized from manure",
                formula="N_vol = SUM[N_total * MS_T,S * Frac_gas_S]",
                inputs={
                    "total_n_excreted_kg": str(total_n_excreted),
                    "awms_count": str(len(allocation)),
                },
                output=str(total_n_volatilized),
                unit="kg NH3-N+NOx-N/yr",
            ))

            # N2O from volatilization
            n2o_n_vol = _quantize(total_n_volatilized * ef4_val)
            n2o_vol_kg = _quantize(n2o_n_vol * N2O_MOLECULAR_RATIO)

            step_num += 1
            trace_steps.append(TraceStep(
                step_number=step_num,
                description="Calculate indirect N2O from volatilization",
                formula="N2O_G = N_vol * EF4 * 44/28",
                inputs={
                    "N_volatilized_kg": str(total_n_volatilized),
                    "EF4": str(ef4_val),
                    "ratio_44_28": str(N2O_MOLECULAR_RATIO),
                },
                output=str(n2o_vol_kg),
                unit="kg N2O/yr",
            ))

            # -- Step 3: Leaching pathway (Eq 10.27) ---------------------------
            per_awms_leach: List[Dict[str, str]] = []
            total_n_leached = _ZERO

            for awms_name, frac in allocation.items():
                awms_key = awms_name.upper()
                if awms_key not in FRAC_LEACH_BY_AWMS:
                    continue  # Already validated above

                frac_leach = FRAC_LEACH_BY_AWMS[awms_key]
                n_in_awms = _quantize(total_n_excreted * frac)
                n_leach = _quantize(n_in_awms * frac_leach)
                total_n_leached += n_leach

                per_awms_leach.append({
                    "awms_type": awms_key,
                    "fraction": str(frac),
                    "frac_leach": str(frac_leach),
                    "n_in_awms_kg": str(n_in_awms),
                    "n_leached_kg": str(n_leach),
                })

            total_n_leached = _quantize(total_n_leached)

            step_num += 1
            trace_steps.append(TraceStep(
                step_number=step_num,
                description="Calculate total N leached from manure",
                formula="N_leach = SUM[N_total * MS_T,S * Frac_leach_S]",
                inputs={
                    "total_n_excreted_kg": str(total_n_excreted),
                    "awms_count": str(len(allocation)),
                },
                output=str(total_n_leached),
                unit="kg N leached/yr",
            ))

            # N2O from leaching
            n2o_n_leach = _quantize(total_n_leached * ef5_val)
            n2o_leach_kg = _quantize(n2o_n_leach * N2O_MOLECULAR_RATIO)

            step_num += 1
            trace_steps.append(TraceStep(
                step_number=step_num,
                description="Calculate indirect N2O from leaching/runoff",
                formula="N2O_L = N_leach * EF5 * 44/28",
                inputs={
                    "N_leached_kg": str(total_n_leached),
                    "EF5": str(ef5_val),
                    "ratio_44_28": str(N2O_MOLECULAR_RATIO),
                },
                output=str(n2o_leach_kg),
                unit="kg N2O/yr",
            ))

            # -- Step 4: Total indirect N2O ------------------------------------
            total_indirect_n2o_kg = _quantize(n2o_vol_kg + n2o_leach_kg)

            step_num += 1
            trace_steps.append(TraceStep(
                step_number=step_num,
                description="Sum indirect N2O from all pathways",
                formula="N2O_indirect = N2O_G + N2O_L",
                inputs={
                    "N2O_volatilization_kg": str(n2o_vol_kg),
                    "N2O_leaching_kg": str(n2o_leach_kg),
                },
                output=str(total_indirect_n2o_kg),
                unit="kg N2O/yr",
            ))

            # -- Step 5: GWP conversion ----------------------------------------
            gas_result = self._build_gas_result(
                "N2O", total_indirect_n2o_kg, gwp_source,
            )
            gas_results = [gas_result]
            total_co2e_kg, total_co2e_tonnes = self._aggregate_gas_results(
                gas_results,
            )

            step_num += 1
            trace_steps.append(TraceStep(
                step_number=step_num,
                description="Apply GWP to convert indirect N2O to CO2e",
                formula="CO2e = N2O_kg * GWP_N2O",
                inputs={
                    "N2O_indirect_kg": str(total_indirect_n2o_kg),
                    "GWP_N2O": gas_result["gwp_value"],
                },
                output=gas_result["co2e_kg"],
                unit="kg CO2e",
            ))

            # -- Assemble result -----------------------------------------------
            elapsed_ms = (time.monotonic() - t0) * 1000.0
            n2o_vol_tonnes = _quantize(n2o_vol_kg * KG_TO_TONNES)
            n2o_leach_tonnes = _quantize(n2o_leach_kg * KG_TO_TONNES)
            n2o_indirect_tonnes = _quantize(
                total_indirect_n2o_kg * KG_TO_TONNES
            )

            result: Dict[str, Any] = {
                "calculation_id": calc_id,
                "status": CalculationStatus.SUCCESS.value,
                "method": "IPCC_MANURE_N2O_INDIRECT",
                "ipcc_reference": "Vol 4, Ch 10, Eq 10.26-10.27",
                "animal_type": at_key,
                "head_count": str(n_head),
                "emissions_by_gas": gas_results,
                "n2o_volatilization": {
                    "n_volatilized_kg": str(total_n_volatilized),
                    "ef4": str(ef4_val),
                    "n2o_n_kg": str(n2o_n_vol),
                    "n2o_kg": str(n2o_vol_kg),
                    "n2o_tonnes": str(n2o_vol_tonnes),
                    "per_awms": per_awms_vol,
                },
                "n2o_leaching": {
                    "n_leached_kg": str(total_n_leached),
                    "ef5": str(ef5_val),
                    "n2o_n_kg": str(n2o_n_leach),
                    "n2o_kg": str(n2o_leach_kg),
                    "n2o_tonnes": str(n2o_leach_tonnes),
                    "per_awms": per_awms_leach,
                },
                "total_n2o_indirect_kg": str(total_indirect_n2o_kg),
                "total_n2o_indirect_tonnes": str(n2o_indirect_tonnes),
                "total_co2e_kg": str(total_co2e_kg),
                "total_co2e_tonnes": str(total_co2e_tonnes),
                "calculation_details": {
                    "nex_kg_n_per_head_yr": str(nex),
                    "total_n_excreted_kg": str(total_n_excreted),
                    "ef4_volatilization": str(ef4_val),
                    "ef5_leaching": str(ef5_val),
                    "n2o_molecular_ratio": str(N2O_MOLECULAR_RATIO),
                    "formula_vol": (
                        "N2O_G = SUM[N_T*Nex_T*MS_T,S*Frac_gas_S] "
                        "* EF4 * 44/28"
                    ),
                    "formula_leach": (
                        "N2O_L = SUM[N_T*Nex_T*MS_T,S*Frac_leach_S] "
                        "* EF5 * 44/28"
                    ),
                },
                "trace_steps": [s.to_dict() for s in trace_steps],
                "processing_time_ms": round(elapsed_ms, 3),
                "calculated_at": utcnow().isoformat(),
            }
            result["provenance_hash"] = _compute_hash(result)

            self._record_metrics(
                "manure_management", "ipcc_tier1",
                at_key.lower(), elapsed_ms / 1000.0,
            )

            logger.info(
                "Manure indirect N2O calculation %s: %s head %s -> "
                "vol=%s tonnes, leach=%s tonnes, total=%s tonnes N2O, "
                "%s tonnes CO2e in %.1fms",
                calc_id, n_head, at_key,
                n2o_vol_tonnes, n2o_leach_tonnes,
                n2o_indirect_tonnes, total_co2e_tonnes, elapsed_ms,
            )
            return result

        except Exception as exc:
            self._increment_errors()
            return self._exception_result(calc_id, exc, t0)

    # ==================================================================
    # Combined: CH4 + Direct N2O + Indirect N2O
    # ==================================================================

    def calculate_manure_total(
        self,
        animal_type: str,
        head_count: Any,
        vs_kg_day: Optional[Any] = None,
        bo: Optional[Any] = None,
        nex_kg_yr: Optional[Any] = None,
        awms_allocation: Optional[Dict[str, Any]] = None,
        mean_temp_c: Any = 15.0,
        gwp_source: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Calculate combined CH4 + direct N2O + indirect N2O from manure.

        Calls calculate_manure_ch4, calculate_manure_n2o, and
        calculate_indirect_n2o with the same parameters and aggregates
        the results.

        Args:
            animal_type: Animal type (see AnimalType enum). Case-insensitive.
            head_count: Number of animals (head).
            vs_kg_day: Volatile solids excreted (kg VS/head/day).
            bo: Maximum CH4 producing capacity (m3 CH4/kg VS).
            nex_kg_yr: Annual N excretion (kg N/head/yr).
            awms_allocation: Dictionary mapping AWMS type to fraction.
            mean_temp_c: Mean annual temperature (degrees C).
            gwp_source: GWP assessment report (AR4/AR5/AR6/AR6_20YR).

        Returns:
            Combined result dictionary with ch4_tonnes, n2o_direct_tonnes,
            n2o_indirect_tonnes, total_co2e, sub-results, provenance hash.
        """
        self._increment_calculations()
        t0 = time.monotonic()
        calc_id = f"mm_total_{uuid4().hex[:12]}"

        try:
            # -- Run sub-calculations ------------------------------------------
            ch4_result = self.calculate_manure_ch4(
                animal_type=animal_type,
                head_count=head_count,
                vs_kg_day=vs_kg_day,
                bo=bo,
                awms_allocation=awms_allocation,
                mean_temp_c=mean_temp_c,
                gwp_source=gwp_source,
            )

            n2o_direct_result = self.calculate_manure_n2o(
                animal_type=animal_type,
                head_count=head_count,
                nex_kg_yr=nex_kg_yr,
                awms_allocation=awms_allocation,
                gwp_source=gwp_source,
            )

            n2o_indirect_result = self.calculate_indirect_n2o(
                animal_type=animal_type,
                head_count=head_count,
                nex_kg_yr=nex_kg_yr,
                awms_allocation=awms_allocation,
                gwp_source=gwp_source,
            )

            # -- Check for sub-calculation errors ------------------------------
            sub_errors: List[str] = []
            if ch4_result.get("status") not in (
                CalculationStatus.SUCCESS.value,
            ):
                sub_errors.append(
                    f"CH4 calculation failed: "
                    f"{ch4_result.get('errors', ch4_result.get('error', 'unknown'))}"
                )
            if n2o_direct_result.get("status") not in (
                CalculationStatus.SUCCESS.value,
            ):
                sub_errors.append(
                    f"Direct N2O calculation failed: "
                    f"{n2o_direct_result.get('errors', n2o_direct_result.get('error', 'unknown'))}"
                )
            if n2o_indirect_result.get("status") not in (
                CalculationStatus.SUCCESS.value,
            ):
                sub_errors.append(
                    f"Indirect N2O calculation failed: "
                    f"{n2o_indirect_result.get('errors', n2o_indirect_result.get('error', 'unknown'))}"
                )

            if sub_errors:
                return self._error_result(calc_id, sub_errors, t0)

            # -- Aggregate results ---------------------------------------------
            ch4_tonnes = _D(ch4_result["total_ch4_tonnes"])
            ch4_co2e_tonnes = _D(ch4_result["total_co2e_tonnes"])

            n2o_direct_tonnes = _D(n2o_direct_result["total_n2o_tonnes"])
            n2o_direct_co2e = _D(n2o_direct_result["total_co2e_tonnes"])

            n2o_indirect_tonnes = _D(
                n2o_indirect_result["total_n2o_indirect_tonnes"]
            )
            n2o_indirect_co2e = _D(n2o_indirect_result["total_co2e_tonnes"])

            total_n2o_tonnes = _quantize(
                n2o_direct_tonnes + n2o_indirect_tonnes
            )
            total_co2e_tonnes = _quantize(
                ch4_co2e_tonnes + n2o_direct_co2e + n2o_indirect_co2e
            )

            # -- Build gas results list ----------------------------------------
            gas_results: List[Dict[str, str]] = []
            if ch4_result.get("emissions_by_gas"):
                gas_results.extend(ch4_result["emissions_by_gas"])
            # Combine N2O entries
            combined_n2o_kg = _quantize(
                _D(n2o_direct_result.get("total_n2o_kg", "0"))
                + _D(n2o_indirect_result.get("total_n2o_indirect_kg", "0"))
            )
            n2o_gas = self._build_gas_result(
                "N2O", combined_n2o_kg, gwp_source,
            )
            gas_results.append(n2o_gas)

            # -- Assemble result -----------------------------------------------
            elapsed_ms = (time.monotonic() - t0) * 1000.0

            result: Dict[str, Any] = {
                "calculation_id": calc_id,
                "status": CalculationStatus.SUCCESS.value,
                "method": "IPCC_MANURE_TOTAL",
                "ipcc_reference": "Vol 4, Ch 10, Eq 10.22-10.27",
                "animal_type": animal_type.upper(),
                "head_count": str(_D(head_count)),
                "emissions_by_gas": gas_results,
                "ch4_tonnes": str(ch4_tonnes),
                "ch4_co2e_tonnes": str(ch4_co2e_tonnes),
                "n2o_direct_tonnes": str(n2o_direct_tonnes),
                "n2o_direct_co2e_tonnes": str(n2o_direct_co2e),
                "n2o_indirect_tonnes": str(n2o_indirect_tonnes),
                "n2o_indirect_co2e_tonnes": str(n2o_indirect_co2e),
                "total_n2o_tonnes": str(total_n2o_tonnes),
                "total_co2e_tonnes": str(total_co2e_tonnes),
                "sub_results": {
                    "ch4": ch4_result,
                    "n2o_direct": n2o_direct_result,
                    "n2o_indirect": n2o_indirect_result,
                },
                "processing_time_ms": round(elapsed_ms, 3),
                "calculated_at": utcnow().isoformat(),
            }
            result["provenance_hash"] = _compute_hash(result)

            logger.info(
                "Manure total calculation %s: %s %s -> "
                "CH4=%s t, N2O_d=%s t, N2O_i=%s t, CO2e=%s t in %.1fms",
                calc_id, head_count, animal_type.upper(),
                ch4_tonnes, n2o_direct_tonnes,
                n2o_indirect_tonnes, total_co2e_tonnes, elapsed_ms,
            )
            return result

        except Exception as exc:
            self._increment_errors()
            return self._exception_result(calc_id, exc, t0)

    # ==================================================================
    # Herd-Level Calculation (Multiple Animal Types)
    # ==================================================================

    def calculate_herd_manure(
        self,
        herd_data: List[Dict[str, Any]],
        gwp_source: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Calculate manure emissions for a mixed herd of multiple animal types.

        Each entry in herd_data should contain at minimum:
            - animal_type (str): Animal type key.
            - head_count (int/float): Number of animals.

        Optional per-entry keys:
            - vs_kg_day, bo, nex_kg_yr, awms_allocation, mean_temp_c

        Args:
            herd_data: List of dictionaries, one per animal type in the herd.
            gwp_source: GWP assessment report override for all calculations.

        Returns:
            Dictionary with per-type results, herd totals by gas, and
            overall total CO2e.
        """
        self._increment_calculations()
        t0 = time.monotonic()
        calc_id = f"mm_herd_{uuid4().hex[:12]}"

        try:
            if not herd_data:
                return self._error_result(
                    calc_id, ["herd_data must not be empty"], t0,
                )

            per_type_results: List[Dict[str, Any]] = []
            total_ch4_tonnes = _ZERO
            total_n2o_direct_tonnes = _ZERO
            total_n2o_indirect_tonnes = _ZERO
            total_co2e_tonnes = _ZERO
            errors_collected: List[str] = []

            for idx, entry in enumerate(herd_data):
                animal_type = entry.get("animal_type")
                head_count = entry.get("head_count")

                if not animal_type:
                    errors_collected.append(
                        f"herd_data[{idx}]: missing 'animal_type'"
                    )
                    continue
                if head_count is None:
                    errors_collected.append(
                        f"herd_data[{idx}]: missing 'head_count'"
                    )
                    continue

                sub_result = self.calculate_manure_total(
                    animal_type=animal_type,
                    head_count=head_count,
                    vs_kg_day=entry.get("vs_kg_day"),
                    bo=entry.get("bo"),
                    nex_kg_yr=entry.get("nex_kg_yr"),
                    awms_allocation=entry.get("awms_allocation"),
                    mean_temp_c=entry.get("mean_temp_c", 15.0),
                    gwp_source=gwp_source,
                )

                per_type_results.append({
                    "animal_type": animal_type.upper(),
                    "head_count": str(_D(head_count)),
                    "result": sub_result,
                })

                if sub_result.get("status") == CalculationStatus.SUCCESS.value:
                    total_ch4_tonnes += _D(sub_result["ch4_tonnes"])
                    total_n2o_direct_tonnes += _D(
                        sub_result["n2o_direct_tonnes"]
                    )
                    total_n2o_indirect_tonnes += _D(
                        sub_result["n2o_indirect_tonnes"]
                    )
                    total_co2e_tonnes += _D(sub_result["total_co2e_tonnes"])
                else:
                    errors_collected.append(
                        f"herd_data[{idx}] ({animal_type}): "
                        f"{sub_result.get('status')}"
                    )

            # Quantize aggregated totals
            total_ch4_tonnes = _quantize(total_ch4_tonnes)
            total_n2o_direct_tonnes = _quantize(total_n2o_direct_tonnes)
            total_n2o_indirect_tonnes = _quantize(total_n2o_indirect_tonnes)
            total_n2o_tonnes = _quantize(
                total_n2o_direct_tonnes + total_n2o_indirect_tonnes
            )
            total_co2e_tonnes = _quantize(total_co2e_tonnes)

            # Determine overall status
            successful = sum(
                1 for r in per_type_results
                if r["result"].get("status") == CalculationStatus.SUCCESS.value
            )
            if successful == len(herd_data):
                status = CalculationStatus.SUCCESS.value
            elif successful > 0:
                status = CalculationStatus.PARTIAL.value
            else:
                status = CalculationStatus.ERROR.value

            elapsed_ms = (time.monotonic() - t0) * 1000.0

            result: Dict[str, Any] = {
                "calculation_id": calc_id,
                "status": status,
                "method": "IPCC_MANURE_HERD",
                "animal_types_count": len(herd_data),
                "successful_calculations": successful,
                "per_type_results": per_type_results,
                "herd_totals": {
                    "total_ch4_tonnes": str(total_ch4_tonnes),
                    "total_n2o_direct_tonnes": str(total_n2o_direct_tonnes),
                    "total_n2o_indirect_tonnes": str(
                        total_n2o_indirect_tonnes
                    ),
                    "total_n2o_tonnes": str(total_n2o_tonnes),
                    "total_co2e_tonnes": str(total_co2e_tonnes),
                },
                "per_gas_totals": {
                    "CH4": {
                        "tonnes": str(total_ch4_tonnes),
                    },
                    "N2O": {
                        "direct_tonnes": str(total_n2o_direct_tonnes),
                        "indirect_tonnes": str(total_n2o_indirect_tonnes),
                        "total_tonnes": str(total_n2o_tonnes),
                    },
                },
                "processing_time_ms": round(elapsed_ms, 3),
                "calculated_at": utcnow().isoformat(),
            }

            if errors_collected:
                result["warnings"] = errors_collected

            result["provenance_hash"] = _compute_hash(result)

            logger.info(
                "Herd manure calculation %s: %d types, "
                "CH4=%s t, N2O=%s t, CO2e=%s t in %.1fms",
                calc_id, len(herd_data),
                total_ch4_tonnes, total_n2o_tonnes,
                total_co2e_tonnes, elapsed_ms,
            )
            return result

        except Exception as exc:
            self._increment_errors()
            return self._exception_result(calc_id, exc, t0)

    # ==================================================================
    # Batch Processing (Multiple Farms)
    # ==================================================================

    def calculate_manure_batch(
        self,
        requests: List[Dict[str, Any]],
        gwp_source: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Batch process manure emission calculations for multiple farms.

        Each request dictionary should contain one of:
            - "herd_data": List[Dict] for herd-level calculation
            - "animal_type" + "head_count": for single-type calculation

        Optional keys per request: vs_kg_day, bo, nex_kg_yr,
        awms_allocation, mean_temp_c, farm_id.

        Args:
            requests: List of calculation request dictionaries.
            gwp_source: GWP assessment report override for all calculations.

        Returns:
            Batch result dictionary with batch_id, per-request results,
            total_co2e, and summary statistics.
        """
        self._increment_calculations()
        t0 = time.monotonic()
        batch_id = f"mm_batch_{uuid4().hex[:12]}"

        try:
            if not requests:
                return self._error_result(
                    batch_id, ["requests must not be empty"], t0,
                )

            results: List[Dict[str, Any]] = []
            total_co2e_tonnes = _ZERO
            total_ch4_tonnes = _ZERO
            total_n2o_tonnes = _ZERO
            successful = 0
            failed = 0

            for idx, req in enumerate(requests):
                farm_id = req.get("farm_id", f"farm_{idx}")

                try:
                    if "herd_data" in req:
                        sub_result = self.calculate_herd_manure(
                            herd_data=req["herd_data"],
                            gwp_source=gwp_source or req.get("gwp_source"),
                        )
                    elif "animal_type" in req and "head_count" in req:
                        sub_result = self.calculate_manure_total(
                            animal_type=req["animal_type"],
                            head_count=req["head_count"],
                            vs_kg_day=req.get("vs_kg_day"),
                            bo=req.get("bo"),
                            nex_kg_yr=req.get("nex_kg_yr"),
                            awms_allocation=req.get("awms_allocation"),
                            mean_temp_c=req.get("mean_temp_c", 15.0),
                            gwp_source=gwp_source or req.get("gwp_source"),
                        )
                    else:
                        sub_result = {
                            "status": CalculationStatus.VALIDATION_ERROR.value,
                            "errors": [
                                f"Request {idx}: must contain 'herd_data' "
                                f"or 'animal_type'+'head_count'"
                            ],
                        }

                    entry: Dict[str, Any] = {
                        "farm_id": farm_id,
                        "request_index": idx,
                        "result": sub_result,
                    }
                    results.append(entry)

                    if sub_result.get("status") in (
                        CalculationStatus.SUCCESS.value,
                        CalculationStatus.PARTIAL.value,
                    ):
                        successful += 1
                        # Aggregate from herd or total result
                        if "herd_totals" in sub_result:
                            total_co2e_tonnes += _D(
                                sub_result["herd_totals"]["total_co2e_tonnes"]
                            )
                            total_ch4_tonnes += _D(
                                sub_result["herd_totals"]["total_ch4_tonnes"]
                            )
                            total_n2o_tonnes += _D(
                                sub_result["herd_totals"]["total_n2o_tonnes"]
                            )
                        elif "total_co2e_tonnes" in sub_result:
                            total_co2e_tonnes += _D(
                                sub_result["total_co2e_tonnes"]
                            )
                            total_ch4_tonnes += _D(
                                sub_result.get("ch4_tonnes", "0")
                            )
                            total_n2o_tonnes += _D(
                                sub_result.get("total_n2o_tonnes", "0")
                            )
                    else:
                        failed += 1

                except Exception as req_exc:
                    failed += 1
                    results.append({
                        "farm_id": farm_id,
                        "request_index": idx,
                        "result": {
                            "status": CalculationStatus.ERROR.value,
                            "error": str(req_exc),
                            "error_type": type(req_exc).__name__,
                        },
                    })

            total_co2e_tonnes = _quantize(total_co2e_tonnes)
            total_ch4_tonnes = _quantize(total_ch4_tonnes)
            total_n2o_tonnes = _quantize(total_n2o_tonnes)

            elapsed_ms = (time.monotonic() - t0) * 1000.0

            batch_result: Dict[str, Any] = {
                "batch_id": batch_id,
                "status": (
                    CalculationStatus.SUCCESS.value if failed == 0
                    else CalculationStatus.PARTIAL.value if successful > 0
                    else CalculationStatus.ERROR.value
                ),
                "total_requests": len(requests),
                "successful": successful,
                "failed": failed,
                "results": results,
                "batch_totals": {
                    "total_ch4_tonnes": str(total_ch4_tonnes),
                    "total_n2o_tonnes": str(total_n2o_tonnes),
                    "total_co2e_tonnes": str(total_co2e_tonnes),
                },
                "processing_time_ms": round(elapsed_ms, 3),
                "calculated_at": utcnow().isoformat(),
            }
            batch_result["provenance_hash"] = _compute_hash(batch_result)

            logger.info(
                "Manure batch %s: %d requests (%d ok, %d fail), "
                "CO2e=%s t in %.1fms",
                batch_id, len(requests), successful, failed,
                total_co2e_tonnes, elapsed_ms,
            )
            return batch_result

        except Exception as exc:
            self._increment_errors()
            return self._exception_result(batch_id, exc, t0)

    # ==================================================================
    # Engine Statistics & Lifecycle
    # ==================================================================

    def get_statistics(self) -> Dict[str, Any]:
        """Return engine usage statistics.

        Returns:
            Dictionary with calculation counts and metadata.
        """
        with self._lock:
            return {
                "engine": "ManureManagementEngine",
                "version": "1.0.0",
                "created_at": self._created_at.isoformat(),
                "total_calculations": self._total_calculations,
                "total_errors": self._total_errors,
                "default_gwp_source": self._default_gwp_source,
                "animal_types_supported": len(AnimalType),
                "awms_types_supported": len(AWMSType),
            }

    def reset(self) -> None:
        """Reset engine counters. Intended for testing teardown."""
        with self._lock:
            self._total_calculations = 0
            self._total_errors = 0
        logger.info("ManureManagementEngine reset")

    def get_supported_animal_types(self) -> List[str]:
        """Return list of supported animal type keys.

        Returns:
            List of AnimalType enum values.
        """
        return [a.value for a in AnimalType]

    def get_supported_awms_types(self) -> List[str]:
        """Return list of supported AWMS type keys.

        Returns:
            List of AWMSType enum values.
        """
        return [a.value for a in AWMSType]

    def get_all_vs_defaults(self) -> Dict[str, str]:
        """Return all IPCC default volatile solids values.

        Returns:
            Dictionary mapping animal type to VS (kg VS/head/day) as string.
        """
        return {k: str(v) for k, v in VS_DEFAULTS.items()}

    def get_all_bo_defaults(self) -> Dict[str, str]:
        """Return all IPCC default Bo values.

        Returns:
            Dictionary mapping animal type to Bo (m3 CH4/kg VS) as string.
        """
        return {k: str(v) for k, v in BO_DEFAULTS.items()}

    def get_all_nex_defaults(self) -> Dict[str, str]:
        """Return all IPCC default N excretion rates.

        Returns:
            Dictionary mapping animal type to Nex (kg N/head/yr) as string.
        """
        return {k: str(v) for k, v in NEX_DEFAULTS.items()}

    def get_all_ef3_defaults(self) -> Dict[str, str]:
        """Return all IPCC default EF3 values.

        Returns:
            Dictionary mapping AWMS type to EF3 (kg N2O-N/kg N) as string.
        """
        return {k: str(v) for k, v in EF3_BY_AWMS.items()}

    def get_all_mcf_values(self) -> Dict[str, Dict[str, str]]:
        """Return all IPCC MCF values by AWMS and temperature range.

        Returns:
            Nested dictionary: {awms_type: {temp_range: mcf_str}}.
        """
        return {
            awms: {tr: str(mcf) for tr, mcf in temps.items()}
            for awms, temps in MCF_BY_AWMS.items()
        }

    # ==================================================================
    # Error response helpers
    # ==================================================================

    def _error_result(
        self,
        calc_id: str,
        errors: List[str],
        t0: float,
    ) -> Dict[str, Any]:
        """Build a validation error result.

        Args:
            calc_id: Calculation identifier.
            errors: List of validation error messages.
            t0: Monotonic start time for duration calculation.

        Returns:
            Error result dictionary.
        """
        self._increment_errors()
        elapsed_ms = (time.monotonic() - t0) * 1000.0
        result: Dict[str, Any] = {
            "calculation_id": calc_id,
            "status": CalculationStatus.VALIDATION_ERROR.value,
            "errors": errors,
            "processing_time_ms": round(elapsed_ms, 3),
            "calculated_at": utcnow().isoformat(),
        }
        result["provenance_hash"] = _compute_hash(result)

        if _METRICS_AVAILABLE and _record_calc_error is not None:
            try:
                _record_calc_error("validation_error")
            except Exception:
                pass

        logger.warning(
            "Validation error in %s: %s", calc_id, errors,
        )
        return result

    def _exception_result(
        self,
        calc_id: str,
        exc: Exception,
        t0: float,
    ) -> Dict[str, Any]:
        """Build an exception error result.

        Args:
            calc_id: Calculation identifier.
            exc: The exception that occurred.
            t0: Monotonic start time for duration calculation.

        Returns:
            Error result dictionary.
        """
        elapsed_ms = (time.monotonic() - t0) * 1000.0
        result: Dict[str, Any] = {
            "calculation_id": calc_id,
            "status": CalculationStatus.ERROR.value,
            "error": str(exc),
            "error_type": type(exc).__name__,
            "processing_time_ms": round(elapsed_ms, 3),
            "calculated_at": utcnow().isoformat(),
        }
        result["provenance_hash"] = _compute_hash(result)

        if _METRICS_AVAILABLE and _record_calc_error is not None:
            try:
                _record_calc_error("calculation_error")
            except Exception:
                pass

        logger.error(
            "Calculation %s failed: %s in %.1fms",
            calc_id, exc, elapsed_ms, exc_info=True,
        )
        return result
