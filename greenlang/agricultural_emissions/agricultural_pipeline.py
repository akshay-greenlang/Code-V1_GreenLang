# -*- coding: utf-8 -*-
"""
AgriculturalPipelineEngine - 8-Stage Orchestration Pipeline (Engine 7 of 7)

AGENT-MRV-008: Agricultural Emissions Agent

End-to-end orchestration pipeline for IPCC Volume 4 (Agriculture, Forestry
and Other Land Use - AFOLU) emission calculations covering enteric
fermentation, manure management, agricultural soils, rice cultivation,
liming, urea application, and field burning of agricultural residues.
Coordinates five upstream engines through a deterministic, eight-stage
pipeline:

    1. VALIDATE_INPUT       - Validate farm_id, emission sources, animal/crop types
    2. CLASSIFY_SOURCES     - Route to ENTERIC/MANURE/SOIL/RICE/LIMING/UREA/BURNING
    3. LOOKUP_FACTORS       - Get EFs from database engine, fall back to IPCC defaults
    4. CALCULATE_LIVESTOCK  - Enteric CH4 + Manure CH4/N2O via upstream or fallback
    5. CALCULATE_CROPLAND   - Soil N2O + rice CH4 + liming/urea CO2 + burning
    6. APPLY_GWP            - Convert CH4/N2O to CO2e using selected AR source
    7. CHECK_COMPLIANCE     - Run compliance checks via ComplianceCheckerEngine
    8. ASSEMBLE_RESULTS     - Combine all results, per-gas and per-source breakdown

Each stage is checkpointed so that failures produce partial results with
complete provenance.

Batch Processing:
    ``execute_batch()`` processes multiple calculation requests,
    accumulating results and producing an aggregate batch summary with
    per-source and per-animal/crop breakdowns.

Zero-Hallucination Guarantees:
    - All emission calculations use deterministic Python Decimal arithmetic
    - No LLM calls in the calculation path
    - SHA-256 provenance hash at every pipeline stage
    - Full audit trail for regulatory traceability (IPCC/GHG Protocol/CSRD)

Thread Safety:
    All mutable state is protected by a ``threading.Lock``.  Concurrent
    ``execute`` invocations from different threads are safe.

Gas Separation:
    The pipeline tracks CO2, CH4 (biogenic), and N2O separately.
    CH4 from enteric fermentation and manure management is biogenic.
    CO2 from liming and urea application is fossil-origin.
    Per-source CO2e breakdowns are provided for flexible regulatory
    reporting across GHG Protocol, IPCC, CSRD, and SBTi FLAG.

Example:
    >>> from greenlang.agricultural_emissions.agricultural_pipeline import (
    ...     AgriculturalPipelineEngine,
    ... )
    >>> pipeline = AgriculturalPipelineEngine()
    >>> result = pipeline.execute({
    ...     "tenant_id": "tenant_001",
    ...     "farm_id": "farm_abc",
    ...     "emission_sources": [
    ...         {
    ...             "source_id": "src_01",
    ...             "source_type": "ENTERIC",
    ...             "animal_type": "DAIRY_CATTLE",
    ...             "head_count": 200,
    ...             "calculation_method": "IPCC_TIER_1",
    ...         },
    ...         {
    ...             "source_id": "src_02",
    ...             "source_type": "RICE",
    ...             "crop_type": "RICE",
    ...             "area_ha": 50,
    ...             "cultivation_period_days": 120,
    ...             "water_regime": "CONTINUOUSLY_FLOODED",
    ...         },
    ...     ],
    ...     "gwp_source": "AR6",
    ...     "frameworks": ["GHG_PROTOCOL", "IPCC_2006"],
    ... })
    >>> assert result["status"] in ("SUCCESS", "PARTIAL")

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
from collections import defaultdict
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level exports
# ---------------------------------------------------------------------------

__all__ = ["AgriculturalPipelineEngine"]

# ---------------------------------------------------------------------------
# Optional upstream-engine imports (graceful degradation)
# ---------------------------------------------------------------------------

try:
    from greenlang.agricultural_emissions.config import (
        get_config,
    )
except ImportError:
    def get_config() -> Any:  # type: ignore[misc]
        """Stub returning None when config module is unavailable."""
        return None

try:
    from greenlang.agricultural_emissions.agricultural_database import (
        AgriculturalDatabaseEngine,
    )
    _DB_ENGINE_AVAILABLE = True
except ImportError:
    AgriculturalDatabaseEngine = None  # type: ignore[assignment, misc]
    _DB_ENGINE_AVAILABLE = False

try:
    from greenlang.agricultural_emissions.enteric_fermentation import (
        EntericFermentationEngine,
    )
    _ENTERIC_ENGINE_AVAILABLE = True
except ImportError:
    EntericFermentationEngine = None  # type: ignore[assignment, misc]
    _ENTERIC_ENGINE_AVAILABLE = False

try:
    from greenlang.agricultural_emissions.manure_management import (
        ManureManagementEngine,
    )
    _MANURE_ENGINE_AVAILABLE = True
except ImportError:
    ManureManagementEngine = None  # type: ignore[assignment, misc]
    _MANURE_ENGINE_AVAILABLE = False

try:
    from greenlang.agricultural_emissions.cropland_emissions import (
        CroplandEmissionsEngine,
    )
    _CROPLAND_ENGINE_AVAILABLE = True
except ImportError:
    CroplandEmissionsEngine = None  # type: ignore[assignment, misc]
    _CROPLAND_ENGINE_AVAILABLE = False

try:
    from greenlang.agricultural_emissions.compliance_checker import (
        ComplianceCheckerEngine,
    )
    _COMPLIANCE_ENGINE_AVAILABLE = True
except ImportError:
    ComplianceCheckerEngine = None  # type: ignore[assignment, misc]
    _COMPLIANCE_ENGINE_AVAILABLE = False

try:
    from greenlang.agricultural_emissions.provenance import (
        ProvenanceTracker,
    )
except ImportError:
    ProvenanceTracker = None  # type: ignore[assignment, misc]


# ---------------------------------------------------------------------------
# UTC helpers
# ---------------------------------------------------------------------------

def _utcnow() -> datetime:
    """Return the current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _utcnow_iso() -> str:
    """Return current UTC datetime as an ISO-8601 string."""
    return _utcnow().isoformat()


def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash of arbitrary data.

    Args:
        data: Any JSON-serializable object or Pydantic model.

    Returns:
        Hex-encoded SHA-256 hash string.
    """
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    else:
        serializable = data
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Decimal helpers
# ---------------------------------------------------------------------------

_PRECISION = Decimal("0.00000001")
_ZERO = Decimal("0")
_ONE = Decimal("1")
_THOUSAND = Decimal("1000")
_MILLION = Decimal("1000000")
_CO2_C_RATIO = Decimal("3.66667")    # 44/12 molecular weight ratio CO2/C
_N2O_N_RATIO = Decimal("1.571429")   # 44/28 molecular weight ratio N2O-N to N2O
_CH4_C_RATIO = Decimal("1.333333")   # 16/12 molecular weight ratio CH4/C
_DAYS_PER_YEAR = Decimal("365")
_KG_PER_TONNE = Decimal("1000")


def _D(value: Any) -> Decimal:
    """Convert a value to Decimal.

    Args:
        value: Numeric value (int, float, str, Decimal).

    Returns:
        Decimal representation of the value.
    """
    if isinstance(value, Decimal):
        return value
    return Decimal(str(value))


def _safe_decimal(value: Any, default: Decimal = _ZERO) -> Decimal:
    """Safely convert to Decimal, returning a default on failure.

    Args:
        value: Value to convert.
        default: Fallback value if conversion fails.

    Returns:
        Decimal representation or the default.
    """
    if value is None:
        return default
    try:
        return _D(value)
    except Exception:
        return default


def _quantize(value: Decimal) -> Decimal:
    """Quantize a Decimal to pipeline precision.

    Args:
        value: Decimal value to quantize.

    Returns:
        Quantized Decimal value.
    """
    return value.quantize(_PRECISION, rounding=ROUND_HALF_UP)


# ---------------------------------------------------------------------------
# GWP values (IPCC AR4 / AR5 / AR6 100-year)
# ---------------------------------------------------------------------------

GWP_VALUES: Dict[str, Dict[str, Decimal]] = {
    "AR6": {
        "CO2": _ONE,
        "CH4": Decimal("27.0"),
        "CH4_FOSSIL": Decimal("29.8"),
        "CH4_BIOGENIC": Decimal("27.0"),
        "N2O": Decimal("273"),
    },
    "AR5": {
        "CO2": _ONE,
        "CH4": Decimal("28"),
        "CH4_FOSSIL": Decimal("28"),
        "CH4_BIOGENIC": Decimal("28"),
        "N2O": Decimal("265"),
    },
    "AR4": {
        "CO2": _ONE,
        "CH4": Decimal("25"),
        "CH4_FOSSIL": Decimal("25"),
        "CH4_BIOGENIC": Decimal("25"),
        "N2O": Decimal("298"),
    },
}


# ===========================================================================
# Pipeline Stages
# ===========================================================================


class PipelineStage(str, Enum):
    """Enumeration of the 8 pipeline stages for agricultural emissions."""

    VALIDATE_INPUT = "VALIDATE_INPUT"
    CLASSIFY_SOURCES = "CLASSIFY_SOURCES"
    LOOKUP_FACTORS = "LOOKUP_FACTORS"
    CALCULATE_LIVESTOCK = "CALCULATE_LIVESTOCK"
    CALCULATE_CROPLAND = "CALCULATE_CROPLAND"
    APPLY_GWP = "APPLY_GWP"
    CHECK_COMPLIANCE = "CHECK_COMPLIANCE"
    ASSEMBLE_RESULTS = "ASSEMBLE_RESULTS"


# ---------------------------------------------------------------------------
# Valid enumerations
# ---------------------------------------------------------------------------

#: Valid emission source types per IPCC 2006 Vol 4.
VALID_SOURCE_TYPES: List[str] = [
    "ENTERIC", "MANURE", "SOIL", "RICE",
    "LIMING", "UREA", "BURNING",
]

#: Source types classified as livestock.
LIVESTOCK_SOURCE_TYPES: frozenset = frozenset({
    "ENTERIC", "MANURE",
})

#: Source types classified as cropland.
CROPLAND_SOURCE_TYPES: frozenset = frozenset({
    "SOIL", "RICE", "LIMING", "UREA", "BURNING",
})

#: Valid animal types per IPCC 2006 Vol 4 Ch 10.
VALID_ANIMAL_TYPES: List[str] = [
    "DAIRY_CATTLE", "NON_DAIRY_CATTLE", "BUFFALO",
    "SHEEP", "GOATS", "CAMELS", "HORSES",
    "MULES_ASSES", "SWINE", "POULTRY",
    "DEER", "ALPACAS", "LLAMAS",
    "FUR_BEARING_ANIMALS", "RABBITS",
    "TURKEYS", "DUCKS", "GEESE",
    "OSTRICH", "OTHER_LIVESTOCK",
]

#: Valid crop types for field burning.
VALID_CROP_TYPES: List[str] = [
    "RICE", "WHEAT", "MAIZE", "SUGARCANE", "COTTON",
    "BARLEY", "OATS", "SORGHUM", "MILLET",
    "SOYBEANS", "OTHER",
]

#: Valid water regimes for rice cultivation per IPCC Ch 5.
VALID_WATER_REGIMES: List[str] = [
    "CONTINUOUSLY_FLOODED", "INTERMITTENT_SINGLE",
    "INTERMITTENT_MULTIPLE", "RAINFED_REGULAR",
    "RAINFED_DROUGHT", "DEEP_WATER", "UPLAND",
]

#: Valid manure management systems per IPCC Ch 10.
VALID_AWMS_TYPES: List[str] = [
    "PASTURE_RANGE", "DAILY_SPREAD", "SOLID_STORAGE",
    "DRY_LOT", "LIQUID_SLURRY", "UNCOVERED_ANAEROBIC_LAGOON",
    "PIT_STORAGE", "ANAEROBIC_DIGESTER", "BURNED_FOR_FUEL",
    "DEEP_BEDDING", "COMPOSTING_STATIC", "COMPOSTING_INTENSIVE",
    "COMPOSTING_WINDROW", "AEROBIC_TREATMENT", "POULTRY_WITH_LITTER",
]

#: Valid calculation methods.
VALID_CALCULATION_METHODS: List[str] = [
    "IPCC_TIER_1", "IPCC_TIER_2", "IPCC_TIER_3",
    "COUNTRY_SPECIFIC", "DIRECT_MEASUREMENT",
    "EMISSION_FACTOR", "MASS_BALANCE", "SPEND_BASED",
]

#: Valid GWP assessment report sources.
VALID_GWP_SOURCES: List[str] = ["AR6", "AR5", "AR4"]


# ---------------------------------------------------------------------------
# IPCC Tier 1 Enteric Fermentation Default EFs (kg CH4/head/year)
# IPCC 2006 Vol 4 Table 10.11
# ---------------------------------------------------------------------------

ENTERIC_DEFAULT_EFS: Dict[str, Decimal] = {
    "DAIRY_CATTLE": Decimal("128"),
    "NON_DAIRY_CATTLE": Decimal("53"),
    "BUFFALO": Decimal("55"),
    "SHEEP": Decimal("8"),
    "GOATS": Decimal("5"),
    "CAMELS": Decimal("46"),
    "HORSES": Decimal("18"),
    "MULES_ASSES": Decimal("10"),
    "SWINE": Decimal("1.5"),
    "POULTRY": _ZERO,
    "DEER": Decimal("20"),
    "ALPACAS": Decimal("8"),
    "LLAMAS": Decimal("8"),
    "FUR_BEARING_ANIMALS": _ZERO,
    "RABBITS": _ZERO,
    "TURKEYS": _ZERO,
    "DUCKS": _ZERO,
    "GEESE": _ZERO,
    "OSTRICH": _ZERO,
    "OTHER_LIVESTOCK": Decimal("10"),
}


# ---------------------------------------------------------------------------
# Manure Management Default MCF (Methane Conversion Factor) by AWMS
# IPCC 2006 Vol 4 Table 10.17 (temperate climate ~15 deg C)
# ---------------------------------------------------------------------------

MANURE_DEFAULT_MCF: Dict[str, Decimal] = {
    "PASTURE_RANGE": Decimal("0.01"),
    "DAILY_SPREAD": Decimal("0.001"),
    "SOLID_STORAGE": Decimal("0.02"),
    "DRY_LOT": Decimal("0.01"),
    "LIQUID_SLURRY": Decimal("0.35"),
    "UNCOVERED_ANAEROBIC_LAGOON": Decimal("0.66"),
    "PIT_STORAGE": Decimal("0.35"),
    "ANAEROBIC_DIGESTER": Decimal("0.05"),
    "BURNED_FOR_FUEL": Decimal("0.01"),
    "DEEP_BEDDING": Decimal("0.17"),
    "COMPOSTING_STATIC": Decimal("0.005"),
    "COMPOSTING_INTENSIVE": Decimal("0.005"),
    "COMPOSTING_WINDROW": Decimal("0.005"),
    "AEROBIC_TREATMENT": Decimal("0.0"),
    "POULTRY_WITH_LITTER": Decimal("0.015"),
}


# ---------------------------------------------------------------------------
# Manure Management Default Bo (max CH4 producing capacity, m3 CH4/kg VS)
# IPCC 2006 Vol 4 Table 10A-4 through 10A-9
# ---------------------------------------------------------------------------

MANURE_DEFAULT_BO: Dict[str, Decimal] = {
    "DAIRY_CATTLE": Decimal("0.24"),
    "NON_DAIRY_CATTLE": Decimal("0.18"),
    "BUFFALO": Decimal("0.10"),
    "SHEEP": Decimal("0.13"),
    "GOATS": Decimal("0.13"),
    "CAMELS": Decimal("0.13"),
    "HORSES": Decimal("0.26"),
    "MULES_ASSES": Decimal("0.26"),
    "SWINE": Decimal("0.45"),
    "POULTRY": Decimal("0.36"),
    "DEER": Decimal("0.13"),
    "ALPACAS": Decimal("0.13"),
    "LLAMAS": Decimal("0.13"),
    "FUR_BEARING_ANIMALS": Decimal("0.36"),
    "RABBITS": Decimal("0.36"),
    "TURKEYS": Decimal("0.36"),
    "DUCKS": Decimal("0.36"),
    "GEESE": Decimal("0.36"),
    "OSTRICH": Decimal("0.36"),
    "OTHER_LIVESTOCK": Decimal("0.17"),
}

#: Default volatile solids excretion rates (kg VS/head/day)
#: IPCC 2006 Vol 4 Tables 10A-4 through 10A-9
MANURE_DEFAULT_VS: Dict[str, Decimal] = {
    "DAIRY_CATTLE": Decimal("5.4"),
    "NON_DAIRY_CATTLE": Decimal("3.7"),
    "BUFFALO": Decimal("3.9"),
    "SHEEP": Decimal("0.32"),
    "GOATS": Decimal("0.35"),
    "CAMELS": Decimal("2.4"),
    "HORSES": Decimal("2.1"),
    "MULES_ASSES": Decimal("1.5"),
    "SWINE": Decimal("0.50"),
    "POULTRY": Decimal("0.02"),
    "DEER": Decimal("0.90"),
    "ALPACAS": Decimal("0.90"),
    "LLAMAS": Decimal("0.90"),
    "FUR_BEARING_ANIMALS": Decimal("0.02"),
    "RABBITS": Decimal("0.02"),
    "TURKEYS": Decimal("0.02"),
    "DUCKS": Decimal("0.02"),
    "GEESE": Decimal("0.02"),
    "OSTRICH": Decimal("0.10"),
    "OTHER_LIVESTOCK": Decimal("1.0"),
}

#: Manure N excretion rates (kg N/head/year) - IPCC Table 10.19
MANURE_DEFAULT_NEX: Dict[str, Decimal] = {
    "DAIRY_CATTLE": Decimal("100"),
    "NON_DAIRY_CATTLE": Decimal("60"),
    "BUFFALO": Decimal("60"),
    "SHEEP": Decimal("12"),
    "GOATS": Decimal("12"),
    "CAMELS": Decimal("46"),
    "HORSES": Decimal("50"),
    "MULES_ASSES": Decimal("50"),
    "SWINE": Decimal("16"),
    "POULTRY": Decimal("0.6"),
    "DEER": Decimal("20"),
    "ALPACAS": Decimal("12"),
    "LLAMAS": Decimal("12"),
    "FUR_BEARING_ANIMALS": Decimal("0.6"),
    "RABBITS": Decimal("0.6"),
    "TURKEYS": Decimal("0.6"),
    "DUCKS": Decimal("0.6"),
    "GEESE": Decimal("0.6"),
    "OSTRICH": Decimal("5.0"),
    "OTHER_LIVESTOCK": Decimal("20"),
}

#: Manure N2O direct emission factors by AWMS (kg N2O-N/kg N)
#: IPCC 2006 Vol 4 Table 10.21
MANURE_N2O_EF3: Dict[str, Decimal] = {
    "PASTURE_RANGE": Decimal("0.02"),
    "DAILY_SPREAD": Decimal("0.0"),
    "SOLID_STORAGE": Decimal("0.005"),
    "DRY_LOT": Decimal("0.02"),
    "LIQUID_SLURRY": Decimal("0.005"),
    "UNCOVERED_ANAEROBIC_LAGOON": Decimal("0.0"),
    "PIT_STORAGE": Decimal("0.002"),
    "ANAEROBIC_DIGESTER": Decimal("0.0"),
    "BURNED_FOR_FUEL": Decimal("0.0"),
    "DEEP_BEDDING": Decimal("0.01"),
    "COMPOSTING_STATIC": Decimal("0.006"),
    "COMPOSTING_INTENSIVE": Decimal("0.006"),
    "COMPOSTING_WINDROW": Decimal("0.006"),
    "AEROBIC_TREATMENT": Decimal("0.005"),
    "POULTRY_WITH_LITTER": Decimal("0.001"),
}


# ---------------------------------------------------------------------------
# Agricultural Soils N2O Default Parameters
# IPCC 2006 Vol 4 Ch 11
# ---------------------------------------------------------------------------

SOIL_N2O_DEFAULTS: Dict[str, Decimal] = {
    # Direct emissions EF1: kg N2O-N / kg N input
    "EF1": Decimal("0.01"),
    # Organic soils EF2 (cropland): kg N2O-N / ha / yr
    "EF2_CROPLAND": Decimal("8.0"),
    # Organic soils EF2 (grassland): kg N2O-N / ha / yr
    "EF2_GRASSLAND": Decimal("8.0"),
    # Pasture/range/paddock EF3 (cattle): kg N2O-N / kg N
    "EF3_PRP_CATTLE": Decimal("0.02"),
    # Pasture/range/paddock EF3 (other): kg N2O-N / kg N
    "EF3_PRP_OTHER": Decimal("0.01"),
    # Indirect emission from atmospheric deposition EF4
    "EF4": Decimal("0.01"),
    # Indirect emission from leaching/runoff EF5
    "EF5": Decimal("0.0075"),
    # Fraction of synthetic N that volatilises as NH3 + NOx
    "FRAC_GASF": Decimal("0.10"),
    # Fraction of organic N that volatilises as NH3 + NOx
    "FRAC_GASM": Decimal("0.20"),
    # Fraction of N that is leached/runoff
    "FRAC_LEACH": Decimal("0.30"),
    # Crop residue N content fraction
    "FRAC_RESIDUE_N": Decimal("0.015"),
    # Fraction of residue returned to soil
    "FRAC_RENEW": Decimal("0.50"),
}


# ---------------------------------------------------------------------------
# Rice Cultivation Default Parameters
# IPCC 2006 Vol 4 Ch 5 Table 5.11 / 5.12 / 5.13
# ---------------------------------------------------------------------------

RICE_DEFAULTS: Dict[str, Any] = {
    # Baseline EF for continuously flooded fields (kg CH4/ha/day)
    "BASELINE_EF": Decimal("1.30"),
    # Scaling factors for water regime (SFw)
    "WATER_REGIME_SF": {
        "CONTINUOUSLY_FLOODED": Decimal("1.0"),
        "INTERMITTENT_SINGLE": Decimal("0.60"),
        "INTERMITTENT_MULTIPLE": Decimal("0.52"),
        "RAINFED_REGULAR": Decimal("0.28"),
        "RAINFED_DROUGHT": Decimal("0.25"),
        "DEEP_WATER": Decimal("0.31"),
        "UPLAND": _ZERO,
    },
    # Scaling factor for organic amendments (SFo) base
    "ORGANIC_AMENDMENT_SF_BASE": Decimal("1.0"),
    # Scaling factor per tonne/ha straw incorporated (SFo increment)
    "ORGANIC_AMENDMENT_SF_INCREMENT": Decimal("0.29"),
    # Scaling factor for soil type (SFs) default
    "SOIL_TYPE_SF": Decimal("1.0"),
    # Default cultivation period (days)
    "DEFAULT_CULTIVATION_DAYS": Decimal("120"),
}


# ---------------------------------------------------------------------------
# Liming and Urea Application Default EFs
# IPCC 2006 Vol 4 Ch 11.3
# ---------------------------------------------------------------------------

LIMING_UREA_DEFAULTS: Dict[str, Decimal] = {
    # Limestone (CaCO3): EF = 0.12 t CO2 / t limestone
    "LIMESTONE_EF": Decimal("0.12"),
    # Dolomite (CaMg(CO3)2): EF = 0.13 t CO2 / t dolomite
    "DOLOMITE_EF": Decimal("0.13"),
    # Urea: EF = 0.20 t CO2 / t urea (= 44/60 * 12/44)
    "UREA_EF": Decimal("0.7333"),
    # Urea molecular weight ratio CO2/urea = 44/60
    "UREA_CO2_RATIO": Decimal("0.7333"),
}


# ---------------------------------------------------------------------------
# Field Burning Default EFs
# IPCC 2006 Vol 4 Ch 2 Table 2.5 and Table 2.6
# ---------------------------------------------------------------------------

FIELD_BURNING_DEFAULTS: Dict[str, Dict[str, Decimal]] = {
    # {crop: {residue_to_crop_ratio, dry_matter_fraction,
    #         combustion_factor, burn_fraction,
    #         ef_ch4_g_per_kg_dm, ef_n2o_g_per_kg_dm,
    #         ef_co_g_per_kg_dm, n_content_fraction}}
    "RICE": {
        "residue_to_crop_ratio": Decimal("1.4"),
        "dry_matter_fraction": Decimal("0.85"),
        "combustion_factor": Decimal("0.80"),
        "burn_fraction": Decimal("0.25"),
        "ef_ch4_g_per_kg_dm": Decimal("2.7"),
        "ef_n2o_g_per_kg_dm": Decimal("0.07"),
        "ef_co_g_per_kg_dm": Decimal("92"),
        "n_content_fraction": Decimal("0.007"),
    },
    "WHEAT": {
        "residue_to_crop_ratio": Decimal("1.3"),
        "dry_matter_fraction": Decimal("0.88"),
        "combustion_factor": Decimal("0.90"),
        "burn_fraction": Decimal("0.20"),
        "ef_ch4_g_per_kg_dm": Decimal("2.7"),
        "ef_n2o_g_per_kg_dm": Decimal("0.07"),
        "ef_co_g_per_kg_dm": Decimal("92"),
        "n_content_fraction": Decimal("0.006"),
    },
    "MAIZE": {
        "residue_to_crop_ratio": Decimal("1.0"),
        "dry_matter_fraction": Decimal("0.89"),
        "combustion_factor": Decimal("0.80"),
        "burn_fraction": Decimal("0.15"),
        "ef_ch4_g_per_kg_dm": Decimal("2.7"),
        "ef_n2o_g_per_kg_dm": Decimal("0.07"),
        "ef_co_g_per_kg_dm": Decimal("92"),
        "n_content_fraction": Decimal("0.006"),
    },
    "SUGARCANE": {
        "residue_to_crop_ratio": Decimal("0.30"),
        "dry_matter_fraction": Decimal("0.84"),
        "combustion_factor": Decimal("0.80"),
        "burn_fraction": Decimal("0.80"),
        "ef_ch4_g_per_kg_dm": Decimal("2.7"),
        "ef_n2o_g_per_kg_dm": Decimal("0.07"),
        "ef_co_g_per_kg_dm": Decimal("92"),
        "n_content_fraction": Decimal("0.004"),
    },
    "COTTON": {
        "residue_to_crop_ratio": Decimal("2.1"),
        "dry_matter_fraction": Decimal("0.91"),
        "combustion_factor": Decimal("0.90"),
        "burn_fraction": Decimal("0.10"),
        "ef_ch4_g_per_kg_dm": Decimal("2.7"),
        "ef_n2o_g_per_kg_dm": Decimal("0.07"),
        "ef_co_g_per_kg_dm": Decimal("92"),
        "n_content_fraction": Decimal("0.009"),
    },
    "BARLEY": {
        "residue_to_crop_ratio": Decimal("1.2"),
        "dry_matter_fraction": Decimal("0.88"),
        "combustion_factor": Decimal("0.90"),
        "burn_fraction": Decimal("0.15"),
        "ef_ch4_g_per_kg_dm": Decimal("2.7"),
        "ef_n2o_g_per_kg_dm": Decimal("0.07"),
        "ef_co_g_per_kg_dm": Decimal("92"),
        "n_content_fraction": Decimal("0.007"),
    },
    "OATS": {
        "residue_to_crop_ratio": Decimal("1.3"),
        "dry_matter_fraction": Decimal("0.88"),
        "combustion_factor": Decimal("0.90"),
        "burn_fraction": Decimal("0.15"),
        "ef_ch4_g_per_kg_dm": Decimal("2.7"),
        "ef_n2o_g_per_kg_dm": Decimal("0.07"),
        "ef_co_g_per_kg_dm": Decimal("92"),
        "n_content_fraction": Decimal("0.007"),
    },
    "SORGHUM": {
        "residue_to_crop_ratio": Decimal("1.4"),
        "dry_matter_fraction": Decimal("0.88"),
        "combustion_factor": Decimal("0.80"),
        "burn_fraction": Decimal("0.15"),
        "ef_ch4_g_per_kg_dm": Decimal("2.7"),
        "ef_n2o_g_per_kg_dm": Decimal("0.07"),
        "ef_co_g_per_kg_dm": Decimal("92"),
        "n_content_fraction": Decimal("0.007"),
    },
    "MILLET": {
        "residue_to_crop_ratio": Decimal("1.4"),
        "dry_matter_fraction": Decimal("0.88"),
        "combustion_factor": Decimal("0.80"),
        "burn_fraction": Decimal("0.15"),
        "ef_ch4_g_per_kg_dm": Decimal("2.7"),
        "ef_n2o_g_per_kg_dm": Decimal("0.07"),
        "ef_co_g_per_kg_dm": Decimal("92"),
        "n_content_fraction": Decimal("0.007"),
    },
    "SOYBEANS": {
        "residue_to_crop_ratio": Decimal("1.5"),
        "dry_matter_fraction": Decimal("0.87"),
        "combustion_factor": Decimal("0.80"),
        "burn_fraction": Decimal("0.10"),
        "ef_ch4_g_per_kg_dm": Decimal("2.7"),
        "ef_n2o_g_per_kg_dm": Decimal("0.07"),
        "ef_co_g_per_kg_dm": Decimal("92"),
        "n_content_fraction": Decimal("0.008"),
    },
    "OTHER": {
        "residue_to_crop_ratio": Decimal("1.3"),
        "dry_matter_fraction": Decimal("0.88"),
        "combustion_factor": Decimal("0.80"),
        "burn_fraction": Decimal("0.15"),
        "ef_ch4_g_per_kg_dm": Decimal("2.7"),
        "ef_n2o_g_per_kg_dm": Decimal("0.07"),
        "ef_co_g_per_kg_dm": Decimal("92"),
        "n_content_fraction": Decimal("0.007"),
    },
}

#: CH4 density at STP (kg/m3) for manure Bo conversion
_CH4_DENSITY = Decimal("0.6682")


# ===========================================================================
# AgriculturalPipelineEngine
# ===========================================================================


class AgriculturalPipelineEngine:
    """End-to-end orchestration pipeline for agricultural emissions.

    Coordinates AgriculturalDatabaseEngine, EntericFermentationEngine,
    ManureManagementEngine, CroplandEmissionsEngine, and
    ComplianceCheckerEngine through an 8-stage deterministic pipeline.

    The pipeline handles multi-source agricultural facilities where a
    single farm may have both livestock and cropland emission sources.
    Each source is calculated independently and results are aggregated
    in the ASSEMBLE_RESULTS stage.

    Thread Safety:
        All mutable state is protected by a ``threading.Lock``.

    Attributes:
        _db_engine: AgriculturalDatabaseEngine instance.
        _enteric_engine: EntericFermentationEngine instance.
        _manure_engine: ManureManagementEngine instance.
        _cropland_engine: CroplandEmissionsEngine instance.
        _compliance_engine: ComplianceCheckerEngine instance.
        _provenance_tracker: ProvenanceTracker instance.
        _lock: Thread lock for mutable state.
        _total_executions: Total pipeline executions counter.
        _stage_timings: Accumulated per-stage timing data.

    Example:
        >>> pipeline = AgriculturalPipelineEngine()
        >>> result = pipeline.execute(request)
    """

    def __init__(
        self,
        config: Optional[Any] = None,
        database_engine: Optional[Any] = None,
        enteric_engine: Optional[Any] = None,
        manure_engine: Optional[Any] = None,
        cropland_engine: Optional[Any] = None,
        compliance_engine: Optional[Any] = None,
    ) -> None:
        """Initialize the AgriculturalPipelineEngine.

        Creates default engine instances if not provided.  Engines that
        fail to import are set to None and their stages use built-in
        IPCC fallback calculations.

        Args:
            config: Optional configuration object.
            database_engine: Optional AgriculturalDatabaseEngine.
            enteric_engine: Optional EntericFermentationEngine.
            manure_engine: Optional ManureManagementEngine.
            cropland_engine: Optional CroplandEmissionsEngine.
            compliance_engine: Optional ComplianceCheckerEngine.
        """
        self._config = config

        # Initialize database engine
        self._db_engine = database_engine
        if self._db_engine is None and AgriculturalDatabaseEngine is not None:
            try:
                self._db_engine = AgriculturalDatabaseEngine()
            except Exception:
                self._db_engine = None

        # Initialize enteric fermentation engine
        self._enteric_engine = enteric_engine
        if self._enteric_engine is None and EntericFermentationEngine is not None:
            try:
                self._enteric_engine = EntericFermentationEngine()
            except Exception:
                self._enteric_engine = None

        # Initialize manure management engine
        self._manure_engine = manure_engine
        if self._manure_engine is None and ManureManagementEngine is not None:
            try:
                self._manure_engine = ManureManagementEngine()
            except Exception:
                self._manure_engine = None

        # Initialize cropland emissions engine
        self._cropland_engine = cropland_engine
        if self._cropland_engine is None and CroplandEmissionsEngine is not None:
            try:
                self._cropland_engine = CroplandEmissionsEngine()
            except Exception:
                self._cropland_engine = None

        # Initialize compliance engine
        self._compliance_engine = compliance_engine
        if self._compliance_engine is None and ComplianceCheckerEngine is not None:
            try:
                self._compliance_engine = ComplianceCheckerEngine()
            except Exception:
                self._compliance_engine = None

        # Initialize provenance tracker
        self._provenance_tracker: Optional[Any] = None
        if ProvenanceTracker is not None:
            try:
                self._provenance_tracker = ProvenanceTracker()
            except Exception:
                self._provenance_tracker = None

        # Thread-safe counters and timing accumulators
        self._lock = threading.Lock()
        self._total_executions: int = 0
        self._total_batches: int = 0
        self._stage_timings: Dict[str, List[float]] = {
            stage.value: [] for stage in PipelineStage
        }
        self._created_at = _utcnow()

        engine_status = {
            "db": self._db_engine is not None,
            "enteric": self._enteric_engine is not None,
            "manure": self._manure_engine is not None,
            "cropland": self._cropland_engine is not None,
            "compliance": self._compliance_engine is not None,
            "provenance": self._provenance_tracker is not None,
        }

        logger.info(
            "AgriculturalPipelineEngine initialized: stages=%d, engines=%s",
            len(PipelineStage), engine_status,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _record_stage_timing(self, stage: str, elapsed_ms: float) -> None:
        """Record timing for a pipeline stage.

        Args:
            stage: Pipeline stage name.
            elapsed_ms: Elapsed time in milliseconds.
        """
        with self._lock:
            self._stage_timings[stage].append(elapsed_ms)

    def _run_stage(
        self,
        stage: PipelineStage,
        context: Dict[str, Any],
        stage_func: Any,
    ) -> Tuple[Dict[str, Any], Optional[str]]:
        """Execute a single pipeline stage with timing and error handling.

        Each stage is wrapped in timing, error capture, and provenance
        hash generation.  On success the stage name is appended to
        ``stages_completed``; on failure it goes to ``stages_failed``
        and the error message is recorded.

        Args:
            stage: Pipeline stage enum.
            context: Pipeline context dictionary (mutated in place).
            stage_func: Callable that performs the stage work.

        Returns:
            Tuple of (updated context, error message or None).
        """
        stage_start = time.monotonic()
        error: Optional[str] = None

        try:
            stage_func(context)
            context["stages_completed"].append(stage.value)
        except Exception as e:
            error = f"Stage {stage.value} failed: {str(e)}"
            context["errors"].append(error)
            context["stages_failed"].append(stage.value)
            logger.error(
                "Pipeline stage %s failed: %s",
                stage.value, str(e), exc_info=True,
            )

        elapsed_ms = (time.monotonic() - stage_start) * 1000
        context["stage_timings"][stage.value] = round(elapsed_ms, 3)
        self._record_stage_timing(stage.value, elapsed_ms)

        # Provenance per stage
        stage_data = {
            "stage": stage.value,
            "elapsed_ms": elapsed_ms,
            "error": error,
        }
        context["provenance_chain"].append(_compute_hash(stage_data))

        return context, error

    def _get_gwp(
        self,
        gas: str,
        gwp_source: str,
    ) -> Decimal:
        """Look up GWP value for a gas and assessment report.

        Args:
            gas: Gas identifier (CO2, CH4, N2O, CH4_FOSSIL, CH4_BIOGENIC).
            gwp_source: Assessment report (AR6, AR5, AR4).

        Returns:
            GWP-100 value as Decimal.
        """
        source_gwps = GWP_VALUES.get(gwp_source, GWP_VALUES["AR6"])
        return source_gwps.get(gas, _ONE)

    def _classify_source_type(self, source_type: str) -> str:
        """Classify an emission source type into a category.

        Args:
            source_type: Source type string (uppercase).

        Returns:
            One of LIVESTOCK, CROPLAND, or OTHER.
        """
        if source_type in LIVESTOCK_SOURCE_TYPES:
            return "LIVESTOCK"
        if source_type in CROPLAND_SOURCE_TYPES:
            return "CROPLAND"
        return "OTHER"

    # ------------------------------------------------------------------
    # Stage 1: Validate Input
    # ------------------------------------------------------------------

    def _stage_validate_input(self, ctx: Dict[str, Any]) -> None:
        """Stage 1: Validate input request.

        Checks required fields, validates enums, normalises values,
        and verifies that animal/crop parameters are consistent.

        Raises:
            ValueError: If validation fails with accumulated errors.
        """
        request = ctx["request"]
        errors: List[str] = []

        # Tenant ID
        tenant_id = str(request.get("tenant_id", "")).strip()
        if not tenant_id:
            errors.append("tenant_id is required")
        ctx["tenant_id"] = tenant_id

        # Farm ID
        farm_id = str(request.get("farm_id", "")).strip()
        if not farm_id:
            errors.append("farm_id is required")
        ctx["farm_id"] = farm_id

        # GWP source
        gwp_source = str(request.get("gwp_source", "AR6")).upper()
        if gwp_source not in VALID_GWP_SOURCES:
            errors.append(
                f"Invalid gwp_source: {gwp_source}. "
                f"Valid: {VALID_GWP_SOURCES}"
            )
        ctx["gwp_source"] = gwp_source

        # Calculation method (top-level default)
        calc_method = str(
            request.get("calculation_method", "IPCC_TIER_1")
        ).upper()
        if calc_method not in VALID_CALCULATION_METHODS:
            errors.append(
                f"Invalid calculation_method: {calc_method}. "
                f"Valid: {VALID_CALCULATION_METHODS}"
            )
        ctx["calculation_method"] = calc_method

        # Frameworks
        ctx["frameworks"] = request.get("frameworks", [])

        # Reporting year
        ctx["reporting_year"] = request.get("reporting_year", _utcnow().year)

        # Climate zone (optional, used for MCF temperature adjustment)
        ctx["climate_zone"] = str(
            request.get("climate_zone", "TEMPERATE")
        ).upper()

        # Annual average temperature (optional, for MCF)
        ctx["annual_avg_temp_c"] = _safe_decimal(
            request.get("annual_avg_temp_c"), default=Decimal("15")
        )

        # Emission sources validation
        sources = request.get("emission_sources", [])
        if not sources:
            errors.append(
                "emission_sources is required and must not be empty"
            )

        validated_sources: List[Dict[str, Any]] = []
        for i, source in enumerate(sources):
            source_errors = self._validate_source(source, i)
            errors.extend(source_errors)
            if not source_errors:
                validated_sources.append(self._normalize_source(source, ctx))

        ctx["emission_sources"] = validated_sources
        ctx["source_count"] = len(validated_sources)

        if errors:
            ctx["validation_errors"] = errors
            raise ValueError(f"Validation failed: {errors}")

        ctx["validation_status"] = "PASSED"
        logger.debug(
            "Validation passed: %d sources, tenant=%s, farm=%s",
            len(validated_sources), tenant_id, farm_id,
        )

    def _validate_source(
        self, source: Dict[str, Any], index: int
    ) -> List[str]:
        """Validate a single emission source.

        Args:
            source: Source dictionary from the request.
            index: Zero-based index of the source in the array.

        Returns:
            List of error messages (empty if valid).
        """
        errors: List[str] = []
        prefix = f"emission_sources[{index}]"

        # Source ID
        source_id = str(source.get("source_id", "")).strip()
        if not source_id:
            errors.append(f"{prefix}.source_id is required")

        # Source type
        source_type = str(source.get("source_type", "")).upper()
        if not source_type:
            errors.append(f"{prefix}.source_type is required")
        elif source_type not in VALID_SOURCE_TYPES:
            errors.append(
                f"{prefix}.source_type '{source_type}' is invalid. "
                f"Valid: {VALID_SOURCE_TYPES}"
            )

        # Animal type validation (for ENTERIC and MANURE)
        if source_type in ("ENTERIC", "MANURE"):
            animal_type = str(source.get("animal_type", "")).upper()
            if not animal_type:
                errors.append(
                    f"{prefix}.animal_type is required for "
                    f"source_type={source_type}"
                )
            elif animal_type not in VALID_ANIMAL_TYPES:
                errors.append(
                    f"{prefix}.animal_type '{animal_type}' is invalid. "
                    f"Valid: {VALID_ANIMAL_TYPES}"
                )
            head_count = _safe_decimal(source.get("head_count"))
            if head_count <= _ZERO:
                errors.append(
                    f"{prefix}.head_count must be > 0 for "
                    f"source_type={source_type}"
                )

        # AWMS validation (for MANURE)
        if source_type == "MANURE":
            awms = str(source.get("awms_type", "")).upper()
            if awms and awms not in VALID_AWMS_TYPES:
                errors.append(
                    f"{prefix}.awms_type '{awms}' is invalid. "
                    f"Valid: {VALID_AWMS_TYPES}"
                )

        # Rice-specific validation
        if source_type == "RICE":
            area = _safe_decimal(source.get("area_ha"))
            if area <= _ZERO:
                errors.append(
                    f"{prefix}.area_ha must be > 0 for source_type=RICE"
                )
            water_regime = str(source.get("water_regime", "")).upper()
            if water_regime and water_regime not in VALID_WATER_REGIMES:
                errors.append(
                    f"{prefix}.water_regime '{water_regime}' is invalid. "
                    f"Valid: {VALID_WATER_REGIMES}"
                )

        # Soil N2O validation
        if source_type == "SOIL":
            n_input = _safe_decimal(source.get("n_input_kg"))
            if n_input < _ZERO:
                errors.append(
                    f"{prefix}.n_input_kg must be >= 0 for source_type=SOIL"
                )

        # Liming validation
        if source_type == "LIMING":
            amount = _safe_decimal(source.get("amount_tonnes"))
            if amount <= _ZERO:
                errors.append(
                    f"{prefix}.amount_tonnes must be > 0 for "
                    f"source_type=LIMING"
                )

        # Urea validation
        if source_type == "UREA":
            amount = _safe_decimal(source.get("amount_tonnes"))
            if amount <= _ZERO:
                errors.append(
                    f"{prefix}.amount_tonnes must be > 0 for "
                    f"source_type=UREA"
                )

        # Burning validation
        if source_type == "BURNING":
            crop_type = str(source.get("crop_type", "")).upper()
            if crop_type and crop_type not in VALID_CROP_TYPES:
                errors.append(
                    f"{prefix}.crop_type '{crop_type}' is invalid. "
                    f"Valid: {VALID_CROP_TYPES}"
                )
            crop_production = _safe_decimal(
                source.get("crop_production_tonnes")
            )
            if crop_production <= _ZERO:
                errors.append(
                    f"{prefix}.crop_production_tonnes must be > 0 for "
                    f"source_type=BURNING"
                )

        return errors

    def _normalize_source(
        self, source: Dict[str, Any], ctx: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Normalize a validated emission source into canonical form.

        Args:
            source: Raw source dictionary from the request.
            ctx: Pipeline context for defaults.

        Returns:
            Normalized source dictionary with uppercase enums and
            Decimal-converted numeric fields.
        """
        source_type = str(source.get("source_type", "")).upper()
        calc_method = str(
            source.get("calculation_method", ctx.get("calculation_method", "IPCC_TIER_1"))
        ).upper()

        normalized: Dict[str, Any] = {
            "source_id": str(source.get("source_id", "")),
            "source_type": source_type,
            "source_class": self._classify_source_type(source_type),
            "calculation_method": calc_method,
        }

        # Livestock-specific fields
        if source_type in ("ENTERIC", "MANURE"):
            normalized["animal_type"] = str(
                source.get("animal_type", "")
            ).upper()
            normalized["head_count"] = _safe_decimal(
                source.get("head_count")
            )
            normalized["weight_kg"] = _safe_decimal(
                source.get("weight_kg"),
                default=_ZERO,
            )
            normalized["milk_production_kg_yr"] = _safe_decimal(
                source.get("milk_production_kg_yr"),
                default=_ZERO,
            )
            normalized["feed_digestibility_pct"] = _safe_decimal(
                source.get("feed_digestibility_pct"),
                default=_ZERO,
            )
            normalized["gross_energy_mj_per_day"] = _safe_decimal(
                source.get("gross_energy_mj_per_day"),
                default=_ZERO,
            )
            normalized["ym_pct"] = _safe_decimal(
                source.get("ym_pct"),
                default=_ZERO,
            )
            normalized["days_on_farm"] = _safe_decimal(
                source.get("days_on_farm"),
                default=_DAYS_PER_YEAR,
            )

        # Manure-specific fields
        if source_type == "MANURE":
            normalized["awms_type"] = str(
                source.get("awms_type", "SOLID_STORAGE")
            ).upper()
            normalized["vs_rate_kg_per_day"] = _safe_decimal(
                source.get("vs_rate_kg_per_day"),
                default=_ZERO,
            )
            normalized["nex_rate_kg_per_yr"] = _safe_decimal(
                source.get("nex_rate_kg_per_yr"),
                default=_ZERO,
            )
            normalized["ms_fraction"] = _safe_decimal(
                source.get("ms_fraction"),
                default=_ONE,
            )
            normalized["biogas_recovery_fraction"] = _safe_decimal(
                source.get("biogas_recovery_fraction"),
                default=_ZERO,
            )

        # Soil N2O fields
        if source_type == "SOIL":
            normalized["n_input_kg"] = _safe_decimal(
                source.get("n_input_kg")
            )
            normalized["n_source"] = str(
                source.get("n_source", "SYNTHETIC")
            ).upper()
            normalized["organic_n_input_kg"] = _safe_decimal(
                source.get("organic_n_input_kg"),
                default=_ZERO,
            )
            normalized["crop_residue_n_kg"] = _safe_decimal(
                source.get("crop_residue_n_kg"),
                default=_ZERO,
            )
            normalized["organic_soil_area_ha"] = _safe_decimal(
                source.get("organic_soil_area_ha"),
                default=_ZERO,
            )
            normalized["organic_soil_type"] = str(
                source.get("organic_soil_type", "CROPLAND")
            ).upper()
            normalized["prp_n_kg"] = _safe_decimal(
                source.get("prp_n_kg"),
                default=_ZERO,
            )
            normalized["prp_animal_type"] = str(
                source.get("prp_animal_type", "OTHER")
            ).upper()

        # Rice cultivation fields
        if source_type == "RICE":
            normalized["area_ha"] = _safe_decimal(
                source.get("area_ha")
            )
            normalized["cultivation_period_days"] = _safe_decimal(
                source.get("cultivation_period_days"),
                default=RICE_DEFAULTS["DEFAULT_CULTIVATION_DAYS"],
            )
            normalized["water_regime"] = str(
                source.get("water_regime", "CONTINUOUSLY_FLOODED")
            ).upper()
            normalized["organic_amendment_tonnes_per_ha"] = _safe_decimal(
                source.get("organic_amendment_tonnes_per_ha"),
                default=_ZERO,
            )
            normalized["soil_type_sf"] = _safe_decimal(
                source.get("soil_type_sf"),
                default=RICE_DEFAULTS["SOIL_TYPE_SF"],
            )
            normalized["pre_season_water_regime"] = str(
                source.get("pre_season_water_regime", "NON_FLOODED_LONG")
            ).upper()

        # Liming fields
        if source_type == "LIMING":
            normalized["amount_tonnes"] = _safe_decimal(
                source.get("amount_tonnes")
            )
            normalized["limestone_type"] = str(
                source.get("limestone_type", "LIMESTONE")
            ).upper()

        # Urea fields
        if source_type == "UREA":
            normalized["amount_tonnes"] = _safe_decimal(
                source.get("amount_tonnes")
            )

        # Field burning fields
        if source_type == "BURNING":
            normalized["crop_type"] = str(
                source.get("crop_type", "OTHER")
            ).upper()
            normalized["crop_production_tonnes"] = _safe_decimal(
                source.get("crop_production_tonnes")
            )
            normalized["area_burned_ha"] = _safe_decimal(
                source.get("area_burned_ha"),
                default=_ZERO,
            )
            normalized["burn_fraction_override"] = _safe_decimal(
                source.get("burn_fraction_override"),
                default=_ZERO,
            )

        return normalized

    # ------------------------------------------------------------------
    # Stage 2: Classify Sources
    # ------------------------------------------------------------------

    def _stage_classify_sources(self, ctx: Dict[str, Any]) -> None:
        """Stage 2: Classify each emission source by type.

        Routes each source to the appropriate calculation engine and
        identifies the farm profile based on the mix of emission sources.
        """
        sources = ctx["emission_sources"]
        enteric_sources: List[Dict[str, Any]] = []
        manure_sources: List[Dict[str, Any]] = []
        soil_sources: List[Dict[str, Any]] = []
        rice_sources: List[Dict[str, Any]] = []
        liming_sources: List[Dict[str, Any]] = []
        urea_sources: List[Dict[str, Any]] = []
        burning_sources: List[Dict[str, Any]] = []

        for source in sources:
            st = source["source_type"]
            if st == "ENTERIC":
                enteric_sources.append(source)
            elif st == "MANURE":
                manure_sources.append(source)
            elif st == "SOIL":
                soil_sources.append(source)
            elif st == "RICE":
                rice_sources.append(source)
            elif st == "LIMING":
                liming_sources.append(source)
            elif st == "UREA":
                urea_sources.append(source)
            elif st == "BURNING":
                burning_sources.append(source)

        ctx["enteric_sources"] = enteric_sources
        ctx["manure_sources"] = manure_sources
        ctx["soil_sources"] = soil_sources
        ctx["rice_sources"] = rice_sources
        ctx["liming_sources"] = liming_sources
        ctx["urea_sources"] = urea_sources
        ctx["burning_sources"] = burning_sources

        # Livestock aggregation
        ctx["livestock_sources"] = enteric_sources + manure_sources
        ctx["cropland_sources"] = (
            soil_sources + rice_sources + liming_sources
            + urea_sources + burning_sources
        )

        # Farm profile
        farm_types = set()
        if enteric_sources or manure_sources:
            farm_types.add("LIVESTOCK_OPERATION")
        if rice_sources:
            farm_types.add("RICE_PADDY")
        if soil_sources or liming_sources or urea_sources:
            farm_types.add("CROPLAND")
        if burning_sources:
            farm_types.add("RESIDUE_BURNING")
        if not farm_types:
            farm_types.add("GENERAL_FARM")
        ctx["farm_types"] = sorted(farm_types)

        # Animal types present
        animal_types_present = sorted(set(
            s["animal_type"] for s in sources
            if s["source_type"] in ("ENTERIC", "MANURE")
            and "animal_type" in s
        ))
        ctx["animal_types_present"] = animal_types_present

        # Crop types present
        crop_types_present = sorted(set(
            s.get("crop_type", "UNKNOWN")
            for s in sources
            if s["source_type"] == "BURNING"
        ))
        ctx["crop_types_present"] = crop_types_present

        # Total head count
        total_head_count = sum(
            s.get("head_count", _ZERO)
            for s in sources
            if s["source_type"] in ("ENTERIC", "MANURE")
        )
        ctx["total_head_count"] = total_head_count

        # Auto-classify using database engine if available
        if self._db_engine is not None:
            for source in sources:
                try:
                    db_classification = self._db_engine.classify_source(
                        source["source_type"],
                        source,
                    )
                    source["db_classification"] = db_classification
                except Exception as e:
                    source["db_classification"] = {
                        "status": "ERROR", "error": str(e)
                    }

        ctx["classification_status"] = "COMPLETE"
        logger.debug(
            "Classification: enteric=%d, manure=%d, soil=%d, rice=%d, "
            "liming=%d, urea=%d, burning=%d",
            len(enteric_sources), len(manure_sources),
            len(soil_sources), len(rice_sources),
            len(liming_sources), len(urea_sources),
            len(burning_sources),
        )

    # ------------------------------------------------------------------
    # Stage 3: Lookup Factors
    # ------------------------------------------------------------------

    def _stage_lookup_factors(self, ctx: Dict[str, Any]) -> None:
        """Stage 3: Look up emission factors from database engine.

        Fetches per-source emission factors.  Falls back to built-in
        IPCC defaults when the database engine is unavailable.
        """
        sources = ctx["emission_sources"]
        factors_by_source: Dict[str, Dict[str, Any]] = {}

        for source in sources:
            source_id = source["source_id"]
            source_type = source["source_type"]

            if self._db_engine is not None:
                try:
                    factors = self._db_engine.get_emission_factors(
                        source_type=source_type,
                        source_params=source,
                    )
                    factors_by_source[source_id] = factors
                    continue
                except Exception as e:
                    logger.warning(
                        "DB factor lookup failed for source %s: %s",
                        source_id, str(e),
                    )

            # Fallback to built-in defaults
            factors_by_source[source_id] = self._get_default_factors(
                source
            )

        ctx["factors_by_source"] = factors_by_source
        ctx["factors_status"] = "COMPLETE"
        logger.debug(
            "Factors retrieved for %d sources",
            len(factors_by_source),
        )

    def _get_default_factors(
        self, source: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Get built-in IPCC default factors for a source.

        Args:
            source: Normalized source dictionary.

        Returns:
            Dictionary of emission factors for the source.
        """
        source_type = source["source_type"]
        factors: Dict[str, Any] = {
            "source": "IPCC_2006_DEFAULTS",
            "source_type": source_type,
        }

        if source_type == "ENTERIC":
            animal_type = source.get("animal_type", "OTHER_LIVESTOCK")
            ef = ENTERIC_DEFAULT_EFS.get(
                animal_type,
                ENTERIC_DEFAULT_EFS["OTHER_LIVESTOCK"],
            )
            factors["ef_kg_ch4_per_head_yr"] = str(ef)

        elif source_type == "MANURE":
            animal_type = source.get("animal_type", "OTHER_LIVESTOCK")
            awms = source.get("awms_type", "SOLID_STORAGE")
            mcf = MANURE_DEFAULT_MCF.get(awms, Decimal("0.02"))
            bo = MANURE_DEFAULT_BO.get(animal_type, Decimal("0.17"))
            vs_rate = MANURE_DEFAULT_VS.get(animal_type, Decimal("1.0"))
            nex = MANURE_DEFAULT_NEX.get(animal_type, Decimal("20"))
            n2o_ef3 = MANURE_N2O_EF3.get(awms, Decimal("0.005"))
            factors.update({
                "mcf": str(mcf),
                "bo_m3_ch4_per_kg_vs": str(bo),
                "vs_rate_kg_per_day": str(vs_rate),
                "nex_kg_n_per_yr": str(nex),
                "n2o_ef3": str(n2o_ef3),
            })

        elif source_type == "SOIL":
            factors.update({
                "ef1": str(SOIL_N2O_DEFAULTS["EF1"]),
                "ef4": str(SOIL_N2O_DEFAULTS["EF4"]),
                "ef5": str(SOIL_N2O_DEFAULTS["EF5"]),
                "frac_gasf": str(SOIL_N2O_DEFAULTS["FRAC_GASF"]),
                "frac_gasm": str(SOIL_N2O_DEFAULTS["FRAC_GASM"]),
                "frac_leach": str(SOIL_N2O_DEFAULTS["FRAC_LEACH"]),
            })

        elif source_type == "RICE":
            water_regime = source.get("water_regime", "CONTINUOUSLY_FLOODED")
            sf_w = RICE_DEFAULTS["WATER_REGIME_SF"].get(
                water_regime, _ONE
            )
            factors.update({
                "baseline_ef_kg_ch4_per_ha_day": str(
                    RICE_DEFAULTS["BASELINE_EF"]
                ),
                "water_regime_sf": str(sf_w),
            })

        elif source_type == "LIMING":
            lime_type = source.get("limestone_type", "LIMESTONE")
            if lime_type == "DOLOMITE":
                ef = LIMING_UREA_DEFAULTS["DOLOMITE_EF"]
            else:
                ef = LIMING_UREA_DEFAULTS["LIMESTONE_EF"]
            factors["ef_t_co2_per_t_limestone"] = str(ef)

        elif source_type == "UREA":
            factors["ef_t_co2_per_t_urea"] = str(
                LIMING_UREA_DEFAULTS["UREA_EF"]
            )

        elif source_type == "BURNING":
            crop_type = source.get("crop_type", "OTHER")
            burning_ef = FIELD_BURNING_DEFAULTS.get(
                crop_type,
                FIELD_BURNING_DEFAULTS["OTHER"],
            )
            factors.update({
                k: str(v) for k, v in burning_ef.items()
            })

        return factors

    # ------------------------------------------------------------------
    # Stage 4: Calculate Livestock
    # ------------------------------------------------------------------

    def _stage_calculate_livestock(self, ctx: Dict[str, Any]) -> None:
        """Stage 4: Calculate livestock emissions.

        Handles enteric fermentation CH4 and manure management CH4/N2O.
        Uses upstream EntericFermentationEngine and ManureManagementEngine
        if available, otherwise falls back to IPCC Tier 1 defaults.
        """
        enteric_sources = ctx.get("enteric_sources", [])
        manure_sources = ctx.get("manure_sources", [])

        if not enteric_sources and not manure_sources:
            ctx["livestock_results"] = {
                "status": "SKIPPED",
                "reason": "No livestock emission sources",
            }
            return

        # Calculate enteric fermentation
        enteric_results = self._calculate_enteric_all(ctx, enteric_sources)
        ctx["enteric_results"] = enteric_results

        # Calculate manure management
        manure_results = self._calculate_manure_all(ctx, manure_sources)
        ctx["manure_results"] = manure_results

        # Aggregate livestock totals
        total_ch4 = (
            _safe_decimal(enteric_results.get("total_ch4_tonnes", "0"))
            + _safe_decimal(manure_results.get("total_ch4_tonnes", "0"))
        )
        total_n2o = _safe_decimal(
            manure_results.get("total_n2o_tonnes", "0")
        )

        ctx["livestock_results"] = {
            "status": "COMPLETE",
            "enteric": enteric_results,
            "manure": manure_results,
            "total_ch4_tonnes": str(_quantize(total_ch4)),
            "total_n2o_tonnes": str(_quantize(total_n2o)),
        }
        logger.debug(
            "Livestock calc: enteric=%d sources, manure=%d sources, "
            "total_ch4=%s t, total_n2o=%s t",
            len(enteric_sources), len(manure_sources),
            total_ch4, total_n2o,
        )

    def _calculate_enteric_all(
        self,
        ctx: Dict[str, Any],
        sources: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Calculate enteric fermentation for all sources.

        Args:
            ctx: Pipeline context.
            sources: List of enteric emission sources.

        Returns:
            Aggregated enteric fermentation results.
        """
        if not sources:
            return {"status": "SKIPPED", "reason": "No enteric sources"}

        stream_results: List[Dict[str, Any]] = []
        total_ch4 = _ZERO

        for source in sources:
            sid = source["source_id"]
            factors = ctx.get("factors_by_source", {}).get(sid, {})

            if self._enteric_engine is not None:
                try:
                    result = self._enteric_engine.calculate(source, factors)
                    stream_results.append(result)
                    total_ch4 += _safe_decimal(result.get("ch4_tonnes"))
                    continue
                except Exception as e:
                    logger.warning(
                        "Enteric engine failed for source %s: %s, "
                        "falling back to defaults",
                        sid, str(e),
                    )

            # Fallback IPCC Tier 1 calculation
            result = self._calculate_enteric_default(source, factors)
            stream_results.append(result)
            total_ch4 += _safe_decimal(result.get("ch4_tonnes"))

        return {
            "status": "COMPLETE",
            "source_count": len(stream_results),
            "sources": stream_results,
            "total_ch4_tonnes": str(_quantize(total_ch4)),
        }

    def _calculate_enteric_default(
        self,
        source: Dict[str, Any],
        factors: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Calculate enteric fermentation using IPCC Tier 1 defaults.

        IPCC Tier 1: CH4 = EF * N * days/365
        where EF = kg CH4/head/year, N = number of head

        For Tier 2 (when GE is provided):
        CH4 = GE * (Ym/100) * days/365 / 55.65

        Args:
            source: Normalized source dictionary.
            factors: Emission factors for this source.

        Returns:
            Dictionary of enteric fermentation emission results.
        """
        animal_type = source.get("animal_type", "OTHER_LIVESTOCK")
        head_count = source.get("head_count", _ZERO)
        days_on_farm = source.get("days_on_farm", _DAYS_PER_YEAR)
        calc_method = source.get("calculation_method", "IPCC_TIER_1")
        ge_mj = source.get("gross_energy_mj_per_day", _ZERO)
        ym_pct = source.get("ym_pct", _ZERO)

        method_used = "IPCC_TIER_1"

        if calc_method in ("IPCC_TIER_2", "IPCC_TIER_3") and ge_mj > _ZERO and ym_pct > _ZERO:
            # Tier 2: CH4 (kg/head/yr) = GE * (Ym/100) * 365 / 55.65
            ef_kg_per_head_yr = _quantize(
                ge_mj * (ym_pct / Decimal("100"))
                * _DAYS_PER_YEAR / Decimal("55.65")
            )
            method_used = "IPCC_TIER_2"
        else:
            # Tier 1: use default EF
            ef_kg_per_head_yr = _safe_decimal(
                factors.get("ef_kg_ch4_per_head_yr"),
                default=ENTERIC_DEFAULT_EFS.get(
                    animal_type,
                    ENTERIC_DEFAULT_EFS["OTHER_LIVESTOCK"],
                ),
            )

        # CH4 (tonnes) = EF * head_count * days/365 / 1000
        ch4_kg = _quantize(
            ef_kg_per_head_yr * head_count * days_on_farm / _DAYS_PER_YEAR
        )
        ch4_tonnes = _quantize(ch4_kg / _KG_PER_TONNE)

        return {
            "source_id": source["source_id"],
            "source_type": "ENTERIC",
            "animal_type": animal_type,
            "head_count": str(head_count),
            "days_on_farm": str(days_on_farm),
            "calculation_method": method_used,
            "ef_kg_ch4_per_head_yr": str(ef_kg_per_head_yr),
            "ch4_kg": str(ch4_kg),
            "ch4_tonnes": str(ch4_tonnes),
            "n2o_tonnes": "0",
            "co2_tonnes": "0",
            "provenance_hash": _compute_hash({
                "source_id": source["source_id"],
                "animal_type": animal_type,
                "head_count": str(head_count),
                "ch4_tonnes": str(ch4_tonnes),
            }),
        }

    def _calculate_manure_all(
        self,
        ctx: Dict[str, Any],
        sources: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Calculate manure management for all sources.

        Args:
            ctx: Pipeline context.
            sources: List of manure emission sources.

        Returns:
            Aggregated manure management results.
        """
        if not sources:
            return {"status": "SKIPPED", "reason": "No manure sources"}

        stream_results: List[Dict[str, Any]] = []
        total_ch4 = _ZERO
        total_n2o = _ZERO

        for source in sources:
            sid = source["source_id"]
            factors = ctx.get("factors_by_source", {}).get(sid, {})

            if self._manure_engine is not None:
                try:
                    result = self._manure_engine.calculate(source, factors)
                    stream_results.append(result)
                    total_ch4 += _safe_decimal(result.get("ch4_tonnes"))
                    total_n2o += _safe_decimal(result.get("n2o_tonnes"))
                    continue
                except Exception as e:
                    logger.warning(
                        "Manure engine failed for source %s: %s, "
                        "falling back to defaults",
                        sid, str(e),
                    )

            # Fallback IPCC Tier 1/2 calculation
            result = self._calculate_manure_default(source, factors, ctx)
            stream_results.append(result)
            total_ch4 += _safe_decimal(result.get("ch4_tonnes"))
            total_n2o += _safe_decimal(result.get("n2o_tonnes"))

        return {
            "status": "COMPLETE",
            "source_count": len(stream_results),
            "sources": stream_results,
            "total_ch4_tonnes": str(_quantize(total_ch4)),
            "total_n2o_tonnes": str(_quantize(total_n2o)),
        }

    def _calculate_manure_default(
        self,
        source: Dict[str, Any],
        factors: Dict[str, Any],
        ctx: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Calculate manure management emissions using IPCC Tier 2 defaults.

        CH4 manure = VS * days * Bo * MCF * 0.67 * MS * N
        where VS = volatile solids (kg/head/day),
              Bo = max CH4 capacity (m3 CH4/kg VS),
              MCF = methane conversion factor,
              0.67 = CH4 density (kg/m3),
              MS = fraction of manure in this AWMS,
              N = number of head

        N2O manure = Nex * MS * EF3 * 44/28
        where Nex = N excretion (kg N/head/yr),
              EF3 = emission factor (kg N2O-N / kg N)

        Args:
            source: Normalized source dictionary.
            factors: Emission factors for this source.
            ctx: Pipeline context for temperature/climate adjustments.

        Returns:
            Dictionary of manure management emission results.
        """
        animal_type = source.get("animal_type", "OTHER_LIVESTOCK")
        head_count = source.get("head_count", _ZERO)
        awms_type = source.get("awms_type", "SOLID_STORAGE")
        days_on_farm = source.get("days_on_farm", _DAYS_PER_YEAR)
        ms_fraction = source.get("ms_fraction", _ONE)

        # VS rate (kg VS / head / day)
        vs_rate = source.get("vs_rate_kg_per_day", _ZERO)
        if vs_rate <= _ZERO:
            vs_rate = _safe_decimal(
                factors.get("vs_rate_kg_per_day"),
                default=MANURE_DEFAULT_VS.get(animal_type, Decimal("1.0")),
            )

        # Bo (m3 CH4 / kg VS)
        bo = _safe_decimal(
            factors.get("bo_m3_ch4_per_kg_vs"),
            default=MANURE_DEFAULT_BO.get(animal_type, Decimal("0.17")),
        )

        # MCF
        mcf = _safe_decimal(
            factors.get("mcf"),
            default=MANURE_DEFAULT_MCF.get(awms_type, Decimal("0.02")),
        )

        # CH4 (kg) = VS * days * Bo * 0.67 * MCF * MS * N
        ch4_kg_total = _quantize(
            vs_rate * days_on_farm * bo * _CH4_DENSITY
            * mcf * ms_fraction * head_count
        )

        # Biogas recovery
        biogas_recovery = source.get("biogas_recovery_fraction", _ZERO)
        ch4_recovered_kg = _quantize(ch4_kg_total * biogas_recovery)
        ch4_emitted_kg = _quantize(ch4_kg_total - ch4_recovered_kg)
        ch4_emitted_kg = max(ch4_emitted_kg, _ZERO)
        ch4_tonnes = _quantize(ch4_emitted_kg / _KG_PER_TONNE)

        # N2O from manure management
        nex_rate = source.get("nex_rate_kg_per_yr", _ZERO)
        if nex_rate <= _ZERO:
            nex_rate = _safe_decimal(
                factors.get("nex_kg_n_per_yr"),
                default=MANURE_DEFAULT_NEX.get(animal_type, Decimal("20")),
            )

        n2o_ef3 = _safe_decimal(
            factors.get("n2o_ef3"),
            default=MANURE_N2O_EF3.get(awms_type, Decimal("0.005")),
        )

        # N2O (kg) = Nex * N * MS * EF3 * 44/28
        # Adjusted for days on farm: Nex * (days/365) * N * MS * EF3 * 44/28
        n2o_kg = _quantize(
            nex_rate * (days_on_farm / _DAYS_PER_YEAR)
            * head_count * ms_fraction * n2o_ef3 * _N2O_N_RATIO
        )
        n2o_tonnes = _quantize(n2o_kg / _KG_PER_TONNE)

        return {
            "source_id": source["source_id"],
            "source_type": "MANURE",
            "animal_type": animal_type,
            "awms_type": awms_type,
            "head_count": str(head_count),
            "days_on_farm": str(days_on_farm),
            "ms_fraction": str(ms_fraction),
            "calculation_method": "IPCC_TIER_2_DEFAULT",
            "vs_rate_kg_per_day": str(vs_rate),
            "bo_m3_ch4_per_kg_vs": str(bo),
            "mcf": str(mcf),
            "ch4_generated_kg": str(ch4_kg_total),
            "ch4_recovered_kg": str(ch4_recovered_kg),
            "ch4_emitted_kg": str(ch4_emitted_kg),
            "ch4_tonnes": str(ch4_tonnes),
            "nex_kg_n_per_yr": str(nex_rate),
            "n2o_ef3": str(n2o_ef3),
            "n2o_kg": str(n2o_kg),
            "n2o_tonnes": str(n2o_tonnes),
            "co2_tonnes": "0",
            "provenance_hash": _compute_hash({
                "source_id": source["source_id"],
                "animal_type": animal_type,
                "awms_type": awms_type,
                "head_count": str(head_count),
                "ch4_tonnes": str(ch4_tonnes),
                "n2o_tonnes": str(n2o_tonnes),
            }),
        }

    # ------------------------------------------------------------------
    # Stage 5: Calculate Cropland
    # ------------------------------------------------------------------

    def _stage_calculate_cropland(self, ctx: Dict[str, Any]) -> None:
        """Stage 5: Calculate cropland emissions.

        Handles agricultural soils N2O, rice cultivation CH4,
        liming CO2, urea application CO2, and field burning CH4/N2O.
        Uses upstream CroplandEmissionsEngine if available, otherwise
        falls back to IPCC Tier 1 defaults.
        """
        soil_sources = ctx.get("soil_sources", [])
        rice_sources = ctx.get("rice_sources", [])
        liming_sources = ctx.get("liming_sources", [])
        urea_sources = ctx.get("urea_sources", [])
        burning_sources = ctx.get("burning_sources", [])

        has_cropland = (
            soil_sources or rice_sources or liming_sources
            or urea_sources or burning_sources
        )
        if not has_cropland:
            ctx["cropland_results"] = {
                "status": "SKIPPED",
                "reason": "No cropland emission sources",
            }
            return

        # Calculate each sub-category
        soil_results = self._calculate_soil_all(ctx, soil_sources)
        rice_results = self._calculate_rice_all(ctx, rice_sources)
        liming_results = self._calculate_liming_all(ctx, liming_sources)
        urea_results = self._calculate_urea_all(ctx, urea_sources)
        burning_results = self._calculate_burning_all(ctx, burning_sources)

        # Aggregate cropland totals
        total_co2 = (
            _safe_decimal(liming_results.get("total_co2_tonnes", "0"))
            + _safe_decimal(urea_results.get("total_co2_tonnes", "0"))
        )
        total_ch4 = (
            _safe_decimal(rice_results.get("total_ch4_tonnes", "0"))
            + _safe_decimal(burning_results.get("total_ch4_tonnes", "0"))
        )
        total_n2o = (
            _safe_decimal(soil_results.get("total_n2o_tonnes", "0"))
            + _safe_decimal(burning_results.get("total_n2o_tonnes", "0"))
        )

        ctx["cropland_results"] = {
            "status": "COMPLETE",
            "soil": soil_results,
            "rice": rice_results,
            "liming": liming_results,
            "urea": urea_results,
            "burning": burning_results,
            "total_co2_tonnes": str(_quantize(total_co2)),
            "total_ch4_tonnes": str(_quantize(total_ch4)),
            "total_n2o_tonnes": str(_quantize(total_n2o)),
        }
        logger.debug(
            "Cropland calc: soil=%d, rice=%d, liming=%d, urea=%d, "
            "burning=%d, co2=%s t, ch4=%s t, n2o=%s t",
            len(soil_sources), len(rice_sources),
            len(liming_sources), len(urea_sources),
            len(burning_sources), total_co2, total_ch4, total_n2o,
        )

    # -- Soil N2O helpers --

    def _calculate_soil_all(
        self,
        ctx: Dict[str, Any],
        sources: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Calculate agricultural soils N2O for all sources.

        Args:
            ctx: Pipeline context.
            sources: List of soil N2O emission sources.

        Returns:
            Aggregated soil N2O results.
        """
        if not sources:
            return {"status": "SKIPPED", "reason": "No soil sources"}

        stream_results: List[Dict[str, Any]] = []
        total_n2o = _ZERO

        for source in sources:
            sid = source["source_id"]
            factors = ctx.get("factors_by_source", {}).get(sid, {})

            if self._cropland_engine is not None:
                try:
                    result = self._cropland_engine.calculate_soil(
                        source, factors
                    )
                    stream_results.append(result)
                    total_n2o += _safe_decimal(result.get("n2o_tonnes"))
                    continue
                except Exception as e:
                    logger.warning(
                        "Cropland engine (soil) failed for source %s: %s",
                        sid, str(e),
                    )

            result = self._calculate_soil_default(source, factors)
            stream_results.append(result)
            total_n2o += _safe_decimal(result.get("n2o_tonnes"))

        return {
            "status": "COMPLETE",
            "source_count": len(stream_results),
            "sources": stream_results,
            "total_n2o_tonnes": str(_quantize(total_n2o)),
        }

    def _calculate_soil_default(
        self,
        source: Dict[str, Any],
        factors: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Calculate agricultural soils N2O using IPCC Tier 1 defaults.

        Direct N2O = (FSN + FON + FCR + FSOM) * EF1 * 44/28
        Indirect N2O (volatilisation) = (FSN*FRAC_GASF + FON*FRAC_GASM) * EF4 * 44/28
        Indirect N2O (leaching) = (FSN + FON + FCR + FSOM) * FRAC_LEACH * EF5 * 44/28

        Args:
            source: Normalized source dictionary.
            factors: Emission factors for this source.

        Returns:
            Dictionary of soil N2O emission results.
        """
        # N inputs (kg N)
        fsn = source.get("n_input_kg", _ZERO)  # Synthetic N
        fon = source.get("organic_n_input_kg", _ZERO)  # Organic N
        fcr = source.get("crop_residue_n_kg", _ZERO)  # Crop residue N
        fsom = _ZERO  # Mineralization from organic soils (simplified)

        # EFs
        ef1 = _safe_decimal(
            factors.get("ef1", SOIL_N2O_DEFAULTS["EF1"])
        )
        ef4 = _safe_decimal(
            factors.get("ef4", SOIL_N2O_DEFAULTS["EF4"])
        )
        ef5 = _safe_decimal(
            factors.get("ef5", SOIL_N2O_DEFAULTS["EF5"])
        )
        frac_gasf = _safe_decimal(
            factors.get("frac_gasf", SOIL_N2O_DEFAULTS["FRAC_GASF"])
        )
        frac_gasm = _safe_decimal(
            factors.get("frac_gasm", SOIL_N2O_DEFAULTS["FRAC_GASM"])
        )
        frac_leach = _safe_decimal(
            factors.get("frac_leach", SOIL_N2O_DEFAULTS["FRAC_LEACH"])
        )

        # PRP (pasture/range/paddock) N2O
        prp_n = source.get("prp_n_kg", _ZERO)
        prp_animal = source.get("prp_animal_type", "OTHER")
        ef3_prp = SOIL_N2O_DEFAULTS.get(
            f"EF3_PRP_{prp_animal.upper()}",
            SOIL_N2O_DEFAULTS["EF3_PRP_OTHER"],
        )

        # Organic soils N2O
        organic_area = source.get("organic_soil_area_ha", _ZERO)
        organic_type = source.get("organic_soil_type", "CROPLAND")
        ef2 = SOIL_N2O_DEFAULTS.get(
            f"EF2_{organic_type}",
            SOIL_N2O_DEFAULTS["EF2_CROPLAND"],
        )

        total_n_input = fsn + fon + fcr + fsom

        # Direct N2O-N (kg) = total_n_input * EF1
        direct_n2o_n = _quantize(total_n_input * ef1)

        # PRP N2O-N
        prp_n2o_n = _quantize(prp_n * ef3_prp)

        # Organic soils N2O-N
        organic_n2o_n = _quantize(organic_area * ef2)

        # Indirect N2O-N from volatilisation
        indirect_vol_n2o_n = _quantize(
            (fsn * frac_gasf + fon * frac_gasm) * ef4
        )

        # Indirect N2O-N from leaching
        indirect_leach_n2o_n = _quantize(
            total_n_input * frac_leach * ef5
        )

        # Total N2O-N (kg)
        total_n2o_n_kg = _quantize(
            direct_n2o_n + prp_n2o_n + organic_n2o_n
            + indirect_vol_n2o_n + indirect_leach_n2o_n
        )

        # Convert N2O-N to N2O (kg) and then to tonnes
        total_n2o_kg = _quantize(total_n2o_n_kg * _N2O_N_RATIO)
        n2o_tonnes = _quantize(total_n2o_kg / _KG_PER_TONNE)

        return {
            "source_id": source["source_id"],
            "source_type": "SOIL",
            "calculation_method": "IPCC_TIER_1_DEFAULT",
            "fsn_kg_n": str(fsn),
            "fon_kg_n": str(fon),
            "fcr_kg_n": str(fcr),
            "prp_kg_n": str(prp_n),
            "organic_soil_area_ha": str(organic_area),
            "direct_n2o_n_kg": str(direct_n2o_n),
            "prp_n2o_n_kg": str(prp_n2o_n),
            "organic_n2o_n_kg": str(organic_n2o_n),
            "indirect_vol_n2o_n_kg": str(indirect_vol_n2o_n),
            "indirect_leach_n2o_n_kg": str(indirect_leach_n2o_n),
            "total_n2o_n_kg": str(total_n2o_n_kg),
            "n2o_kg": str(total_n2o_kg),
            "n2o_tonnes": str(n2o_tonnes),
            "ch4_tonnes": "0",
            "co2_tonnes": "0",
            "provenance_hash": _compute_hash({
                "source_id": source["source_id"],
                "fsn": str(fsn),
                "fon": str(fon),
                "n2o_tonnes": str(n2o_tonnes),
            }),
        }

    # -- Rice CH4 helpers --

    def _calculate_rice_all(
        self,
        ctx: Dict[str, Any],
        sources: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Calculate rice cultivation CH4 for all sources.

        Args:
            ctx: Pipeline context.
            sources: List of rice emission sources.

        Returns:
            Aggregated rice cultivation results.
        """
        if not sources:
            return {"status": "SKIPPED", "reason": "No rice sources"}

        stream_results: List[Dict[str, Any]] = []
        total_ch4 = _ZERO

        for source in sources:
            sid = source["source_id"]
            factors = ctx.get("factors_by_source", {}).get(sid, {})

            if self._cropland_engine is not None:
                try:
                    result = self._cropland_engine.calculate_rice(
                        source, factors
                    )
                    stream_results.append(result)
                    total_ch4 += _safe_decimal(result.get("ch4_tonnes"))
                    continue
                except Exception as e:
                    logger.warning(
                        "Cropland engine (rice) failed for source %s: %s",
                        sid, str(e),
                    )

            result = self._calculate_rice_default(source, factors)
            stream_results.append(result)
            total_ch4 += _safe_decimal(result.get("ch4_tonnes"))

        return {
            "status": "COMPLETE",
            "source_count": len(stream_results),
            "sources": stream_results,
            "total_ch4_tonnes": str(_quantize(total_ch4)),
        }

    def _calculate_rice_default(
        self,
        source: Dict[str, Any],
        factors: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Calculate rice cultivation CH4 using IPCC Tier 1 defaults.

        CH4_rice = EF_baseline * SFw * SFp * SFo * SFs * A * days * 1e-3

        Args:
            source: Normalized source dictionary.
            factors: Emission factors for this source.

        Returns:
            Dictionary of rice cultivation emission results.
        """
        area_ha = source.get("area_ha", _ZERO)
        days = source.get(
            "cultivation_period_days",
            RICE_DEFAULTS["DEFAULT_CULTIVATION_DAYS"],
        )
        water_regime = source.get("water_regime", "CONTINUOUSLY_FLOODED")

        # Baseline EF (kg CH4 / ha / day)
        ef_base = _safe_decimal(
            factors.get("baseline_ef_kg_ch4_per_ha_day"),
            default=RICE_DEFAULTS["BASELINE_EF"],
        )

        # Water regime scaling factor
        sf_w = _safe_decimal(
            factors.get("water_regime_sf"),
            default=RICE_DEFAULTS["WATER_REGIME_SF"].get(
                water_regime, _ONE
            ),
        )

        # Pre-season water regime factor (SFp) - simplified
        sf_p = _ONE

        # Organic amendment factor
        organic_amt = source.get("organic_amendment_tonnes_per_ha", _ZERO)
        sf_o_base = RICE_DEFAULTS["ORGANIC_AMENDMENT_SF_BASE"]
        sf_o_incr = RICE_DEFAULTS["ORGANIC_AMENDMENT_SF_INCREMENT"]
        sf_o = sf_o_base + (organic_amt * sf_o_incr)

        # Soil type factor
        sf_s = source.get("soil_type_sf", RICE_DEFAULTS["SOIL_TYPE_SF"])

        # CH4 (kg) = EF * SFw * SFp * SFo * SFs * Area * days
        ch4_kg = _quantize(
            ef_base * sf_w * sf_p * sf_o * sf_s * area_ha * days
        )
        ch4_tonnes = _quantize(ch4_kg / _KG_PER_TONNE)

        return {
            "source_id": source["source_id"],
            "source_type": "RICE",
            "water_regime": water_regime,
            "area_ha": str(area_ha),
            "cultivation_period_days": str(days),
            "calculation_method": "IPCC_TIER_1_DEFAULT",
            "ef_baseline_kg_ch4_per_ha_day": str(ef_base),
            "sf_water_regime": str(sf_w),
            "sf_pre_season": str(sf_p),
            "sf_organic_amendment": str(sf_o),
            "sf_soil_type": str(sf_s),
            "ch4_kg": str(ch4_kg),
            "ch4_tonnes": str(ch4_tonnes),
            "n2o_tonnes": "0",
            "co2_tonnes": "0",
            "provenance_hash": _compute_hash({
                "source_id": source["source_id"],
                "area_ha": str(area_ha),
                "water_regime": water_regime,
                "ch4_tonnes": str(ch4_tonnes),
            }),
        }

    # -- Liming CO2 helpers --

    def _calculate_liming_all(
        self,
        ctx: Dict[str, Any],
        sources: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Calculate liming CO2 for all sources.

        Args:
            ctx: Pipeline context.
            sources: List of liming emission sources.

        Returns:
            Aggregated liming results.
        """
        if not sources:
            return {"status": "SKIPPED", "reason": "No liming sources"}

        stream_results: List[Dict[str, Any]] = []
        total_co2 = _ZERO

        for source in sources:
            sid = source["source_id"]
            factors = ctx.get("factors_by_source", {}).get(sid, {})

            if self._cropland_engine is not None:
                try:
                    result = self._cropland_engine.calculate_liming(
                        source, factors
                    )
                    stream_results.append(result)
                    total_co2 += _safe_decimal(result.get("co2_tonnes"))
                    continue
                except Exception as e:
                    logger.warning(
                        "Cropland engine (liming) failed for source %s: %s",
                        sid, str(e),
                    )

            result = self._calculate_liming_default(source, factors)
            stream_results.append(result)
            total_co2 += _safe_decimal(result.get("co2_tonnes"))

        return {
            "status": "COMPLETE",
            "source_count": len(stream_results),
            "sources": stream_results,
            "total_co2_tonnes": str(_quantize(total_co2)),
        }

    def _calculate_liming_default(
        self,
        source: Dict[str, Any],
        factors: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Calculate liming CO2 using IPCC defaults.

        CO2 = M_limestone * EF_limestone + M_dolomite * EF_dolomite

        Args:
            source: Normalized source dictionary.
            factors: Emission factors for this source.

        Returns:
            Dictionary of liming emission results.
        """
        amount = source.get("amount_tonnes", _ZERO)
        lime_type = source.get("limestone_type", "LIMESTONE")

        if lime_type == "DOLOMITE":
            ef = _safe_decimal(
                factors.get("ef_t_co2_per_t_limestone"),
                default=LIMING_UREA_DEFAULTS["DOLOMITE_EF"],
            )
        else:
            ef = _safe_decimal(
                factors.get("ef_t_co2_per_t_limestone"),
                default=LIMING_UREA_DEFAULTS["LIMESTONE_EF"],
            )

        co2_tonnes = _quantize(amount * ef)

        return {
            "source_id": source["source_id"],
            "source_type": "LIMING",
            "limestone_type": lime_type,
            "amount_tonnes": str(amount),
            "calculation_method": "IPCC_DEFAULT",
            "ef_t_co2_per_t": str(ef),
            "co2_tonnes": str(co2_tonnes),
            "ch4_tonnes": "0",
            "n2o_tonnes": "0",
            "provenance_hash": _compute_hash({
                "source_id": source["source_id"],
                "amount": str(amount),
                "co2_tonnes": str(co2_tonnes),
            }),
        }

    # -- Urea CO2 helpers --

    def _calculate_urea_all(
        self,
        ctx: Dict[str, Any],
        sources: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Calculate urea application CO2 for all sources.

        Args:
            ctx: Pipeline context.
            sources: List of urea emission sources.

        Returns:
            Aggregated urea results.
        """
        if not sources:
            return {"status": "SKIPPED", "reason": "No urea sources"}

        stream_results: List[Dict[str, Any]] = []
        total_co2 = _ZERO

        for source in sources:
            sid = source["source_id"]
            factors = ctx.get("factors_by_source", {}).get(sid, {})

            if self._cropland_engine is not None:
                try:
                    result = self._cropland_engine.calculate_urea(
                        source, factors
                    )
                    stream_results.append(result)
                    total_co2 += _safe_decimal(result.get("co2_tonnes"))
                    continue
                except Exception as e:
                    logger.warning(
                        "Cropland engine (urea) failed for source %s: %s",
                        sid, str(e),
                    )

            result = self._calculate_urea_default(source, factors)
            stream_results.append(result)
            total_co2 += _safe_decimal(result.get("co2_tonnes"))

        return {
            "status": "COMPLETE",
            "source_count": len(stream_results),
            "sources": stream_results,
            "total_co2_tonnes": str(_quantize(total_co2)),
        }

    def _calculate_urea_default(
        self,
        source: Dict[str, Any],
        factors: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Calculate urea application CO2 using IPCC defaults.

        CO2 = M_urea * EF_urea (where EF = 0.7333 = 44/60)

        Args:
            source: Normalized source dictionary.
            factors: Emission factors for this source.

        Returns:
            Dictionary of urea emission results.
        """
        amount = source.get("amount_tonnes", _ZERO)
        ef = _safe_decimal(
            factors.get("ef_t_co2_per_t_urea"),
            default=LIMING_UREA_DEFAULTS["UREA_EF"],
        )

        co2_tonnes = _quantize(amount * ef)

        return {
            "source_id": source["source_id"],
            "source_type": "UREA",
            "amount_tonnes": str(amount),
            "calculation_method": "IPCC_DEFAULT",
            "ef_t_co2_per_t_urea": str(ef),
            "co2_tonnes": str(co2_tonnes),
            "ch4_tonnes": "0",
            "n2o_tonnes": "0",
            "provenance_hash": _compute_hash({
                "source_id": source["source_id"],
                "amount": str(amount),
                "co2_tonnes": str(co2_tonnes),
            }),
        }

    # -- Field Burning helpers --

    def _calculate_burning_all(
        self,
        ctx: Dict[str, Any],
        sources: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Calculate field burning emissions for all sources.

        Args:
            ctx: Pipeline context.
            sources: List of burning emission sources.

        Returns:
            Aggregated field burning results.
        """
        if not sources:
            return {"status": "SKIPPED", "reason": "No burning sources"}

        stream_results: List[Dict[str, Any]] = []
        total_ch4 = _ZERO
        total_n2o = _ZERO

        for source in sources:
            sid = source["source_id"]
            factors = ctx.get("factors_by_source", {}).get(sid, {})

            if self._cropland_engine is not None:
                try:
                    result = self._cropland_engine.calculate_burning(
                        source, factors
                    )
                    stream_results.append(result)
                    total_ch4 += _safe_decimal(result.get("ch4_tonnes"))
                    total_n2o += _safe_decimal(result.get("n2o_tonnes"))
                    continue
                except Exception as e:
                    logger.warning(
                        "Cropland engine (burning) failed for source %s: %s",
                        sid, str(e),
                    )

            result = self._calculate_burning_default(source, factors)
            stream_results.append(result)
            total_ch4 += _safe_decimal(result.get("ch4_tonnes"))
            total_n2o += _safe_decimal(result.get("n2o_tonnes"))

        return {
            "status": "COMPLETE",
            "source_count": len(stream_results),
            "sources": stream_results,
            "total_ch4_tonnes": str(_quantize(total_ch4)),
            "total_n2o_tonnes": str(_quantize(total_n2o)),
        }

    def _calculate_burning_default(
        self,
        source: Dict[str, Any],
        factors: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Calculate field burning emissions using IPCC Tier 1 defaults.

        Biomass burned = crop_production * residue_to_crop_ratio *
                         dry_matter_fraction * burn_fraction *
                         combustion_factor
        CH4 = biomass_burned * EF_CH4 / 1e6  (g->tonnes)
        N2O = biomass_burned * EF_N2O / 1e6  (g->tonnes)

        Args:
            source: Normalized source dictionary.
            factors: Emission factors for this source.

        Returns:
            Dictionary of field burning emission results.
        """
        crop_type = source.get("crop_type", "OTHER")
        crop_production = source.get("crop_production_tonnes", _ZERO)

        defaults = FIELD_BURNING_DEFAULTS.get(
            crop_type, FIELD_BURNING_DEFAULTS["OTHER"]
        )

        rcr = _safe_decimal(
            factors.get("residue_to_crop_ratio"),
            default=defaults["residue_to_crop_ratio"],
        )
        dm_frac = _safe_decimal(
            factors.get("dry_matter_fraction"),
            default=defaults["dry_matter_fraction"],
        )
        cf = _safe_decimal(
            factors.get("combustion_factor"),
            default=defaults["combustion_factor"],
        )

        # Allow burn fraction override
        bf_override = source.get("burn_fraction_override", _ZERO)
        if bf_override > _ZERO:
            bf = bf_override
        else:
            bf = _safe_decimal(
                factors.get("burn_fraction"),
                default=defaults["burn_fraction"],
            )

        ef_ch4 = _safe_decimal(
            factors.get("ef_ch4_g_per_kg_dm"),
            default=defaults["ef_ch4_g_per_kg_dm"],
        )
        ef_n2o = _safe_decimal(
            factors.get("ef_n2o_g_per_kg_dm"),
            default=defaults["ef_n2o_g_per_kg_dm"],
        )

        # Biomass burned (tonnes dry matter)
        residue_mass = crop_production * rcr
        dm_mass = residue_mass * dm_frac
        burned_mass = dm_mass * bf
        combusted_mass = burned_mass * cf

        # Convert to kg for EF application
        combusted_mass_kg = combusted_mass * _KG_PER_TONNE

        # CH4 (tonnes) = combusted_mass_kg * EF_CH4_g_per_kg / 1e6
        ch4_tonnes = _quantize(
            combusted_mass_kg * ef_ch4 / _MILLION
        )

        # N2O (tonnes) = combusted_mass_kg * EF_N2O_g_per_kg / 1e6
        n2o_tonnes = _quantize(
            combusted_mass_kg * ef_n2o / _MILLION
        )

        return {
            "source_id": source["source_id"],
            "source_type": "BURNING",
            "crop_type": crop_type,
            "crop_production_tonnes": str(crop_production),
            "calculation_method": "IPCC_TIER_1_DEFAULT",
            "residue_to_crop_ratio": str(rcr),
            "dry_matter_fraction": str(dm_frac),
            "burn_fraction": str(bf),
            "combustion_factor": str(cf),
            "combusted_mass_tonnes_dm": str(_quantize(combusted_mass)),
            "ef_ch4_g_per_kg_dm": str(ef_ch4),
            "ef_n2o_g_per_kg_dm": str(ef_n2o),
            "ch4_tonnes": str(ch4_tonnes),
            "n2o_tonnes": str(n2o_tonnes),
            "co2_tonnes": "0",
            "provenance_hash": _compute_hash({
                "source_id": source["source_id"],
                "crop_type": crop_type,
                "crop_production": str(crop_production),
                "ch4_tonnes": str(ch4_tonnes),
                "n2o_tonnes": str(n2o_tonnes),
            }),
        }

    # ------------------------------------------------------------------
    # Stage 6: Apply GWP
    # ------------------------------------------------------------------

    def _stage_apply_gwp(self, ctx: Dict[str, Any]) -> None:
        """Stage 6: Convert per-gas emissions to CO2e using GWP values.

        Applies GWP-100 conversion factors from the selected assessment
        report (AR4, AR5, or AR6) to CH4 and N2O totals from both
        livestock and cropland stages.
        """
        gwp_source = ctx["gwp_source"]
        ch4_gwp = self._get_gwp("CH4_BIOGENIC", gwp_source)
        n2o_gwp = self._get_gwp("N2O", gwp_source)
        co2_gwp = self._get_gwp("CO2", gwp_source)

        livestock = ctx.get("livestock_results", {})
        cropland = ctx.get("cropland_results", {})

        # --- Livestock CO2e ---
        livestock_ch4 = _safe_decimal(
            livestock.get("total_ch4_tonnes", "0")
        )
        livestock_n2o = _safe_decimal(
            livestock.get("total_n2o_tonnes", "0")
        )
        livestock_ch4_co2e = _quantize(livestock_ch4 * ch4_gwp)
        livestock_n2o_co2e = _quantize(livestock_n2o * n2o_gwp)
        livestock_total_co2e = _quantize(
            livestock_ch4_co2e + livestock_n2o_co2e
        )

        # --- Cropland CO2e ---
        cropland_co2 = _safe_decimal(
            cropland.get("total_co2_tonnes", "0")
        )
        cropland_ch4 = _safe_decimal(
            cropland.get("total_ch4_tonnes", "0")
        )
        cropland_n2o = _safe_decimal(
            cropland.get("total_n2o_tonnes", "0")
        )
        cropland_co2_co2e = _quantize(cropland_co2 * co2_gwp)
        cropland_ch4_co2e = _quantize(cropland_ch4 * ch4_gwp)
        cropland_n2o_co2e = _quantize(cropland_n2o * n2o_gwp)
        cropland_total_co2e = _quantize(
            cropland_co2_co2e + cropland_ch4_co2e + cropland_n2o_co2e
        )

        # --- Grand totals ---
        total_co2 = cropland_co2  # Only liming/urea produce CO2
        total_ch4 = livestock_ch4 + cropland_ch4
        total_n2o = livestock_n2o + cropland_n2o

        total_co2_co2e = _quantize(total_co2 * co2_gwp)
        total_ch4_co2e = _quantize(total_ch4 * ch4_gwp)
        total_n2o_co2e = _quantize(total_n2o * n2o_gwp)
        total_co2e = _quantize(
            total_co2_co2e + total_ch4_co2e + total_n2o_co2e
        )

        # --- Per-source CO2e breakdown ---
        per_source_co2e: Dict[str, str] = {}

        # Enteric
        enteric = livestock.get("enteric", {})
        enteric_ch4 = _safe_decimal(enteric.get("total_ch4_tonnes", "0"))
        per_source_co2e["enteric"] = str(_quantize(enteric_ch4 * ch4_gwp))

        # Manure
        manure = livestock.get("manure", {})
        manure_ch4 = _safe_decimal(manure.get("total_ch4_tonnes", "0"))
        manure_n2o = _safe_decimal(manure.get("total_n2o_tonnes", "0"))
        per_source_co2e["manure"] = str(_quantize(
            manure_ch4 * ch4_gwp + manure_n2o * n2o_gwp
        ))

        # Soil
        soil = cropland.get("soil", {})
        soil_n2o = _safe_decimal(soil.get("total_n2o_tonnes", "0"))
        per_source_co2e["soils"] = str(_quantize(soil_n2o * n2o_gwp))

        # Rice
        rice = cropland.get("rice", {})
        rice_ch4 = _safe_decimal(rice.get("total_ch4_tonnes", "0"))
        per_source_co2e["rice"] = str(_quantize(rice_ch4 * ch4_gwp))

        # Liming
        liming = cropland.get("liming", {})
        liming_co2 = _safe_decimal(liming.get("total_co2_tonnes", "0"))
        per_source_co2e["liming"] = str(_quantize(liming_co2 * co2_gwp))

        # Urea
        urea = cropland.get("urea", {})
        urea_co2 = _safe_decimal(urea.get("total_co2_tonnes", "0"))
        per_source_co2e["urea"] = str(_quantize(urea_co2 * co2_gwp))

        # Burning
        burning = cropland.get("burning", {})
        burning_ch4 = _safe_decimal(burning.get("total_ch4_tonnes", "0"))
        burning_n2o = _safe_decimal(burning.get("total_n2o_tonnes", "0"))
        per_source_co2e["burning"] = str(_quantize(
            burning_ch4 * ch4_gwp + burning_n2o * n2o_gwp
        ))

        ctx["gwp_results"] = {
            "status": "COMPLETE",
            "gwp_source": gwp_source,
            "gwp_co2": str(co2_gwp),
            "gwp_ch4": str(ch4_gwp),
            "gwp_n2o": str(n2o_gwp),
            # Grand totals (tonnes of each gas)
            "total_co2_tonnes": str(_quantize(total_co2)),
            "total_ch4_tonnes": str(_quantize(total_ch4)),
            "total_n2o_tonnes": str(_quantize(total_n2o)),
            # CO2e by gas
            "co2_co2e_tonnes": str(total_co2_co2e),
            "ch4_co2e_tonnes": str(total_ch4_co2e),
            "n2o_co2e_tonnes": str(total_n2o_co2e),
            "total_co2e_tonnes": str(total_co2e),
            # Livestock subtotal
            "livestock_ch4_co2e": str(livestock_ch4_co2e),
            "livestock_n2o_co2e": str(livestock_n2o_co2e),
            "livestock_total_co2e": str(livestock_total_co2e),
            # Cropland subtotal
            "cropland_co2_co2e": str(cropland_co2_co2e),
            "cropland_ch4_co2e": str(cropland_ch4_co2e),
            "cropland_n2o_co2e": str(cropland_n2o_co2e),
            "cropland_total_co2e": str(cropland_total_co2e),
            # Per-source breakdown
            "per_source_co2e": per_source_co2e,
        }

        logger.debug(
            "GWP applied (%s): co2=%s, ch4=%s, n2o=%s -> total=%s tCO2e",
            gwp_source, total_co2_co2e, total_ch4_co2e,
            total_n2o_co2e, total_co2e,
        )

    # ------------------------------------------------------------------
    # Stage 7: Check Compliance
    # ------------------------------------------------------------------

    def _stage_check_compliance(self, ctx: Dict[str, Any]) -> None:
        """Stage 7: Run compliance checks against selected frameworks.

        Supports GHG_PROTOCOL, IPCC_2006, IPCC_2019, CSRD_ESRS_E1,
        SBTi_FLAG, EPA_GHGRP, and UNFCCC.
        """
        if self._compliance_engine is None:
            ctx["compliance_result"] = {
                "status": "COMPLIANCE_ENGINE_UNAVAILABLE",
            }
            return

        frameworks = ctx.get("frameworks", [])
        if not frameworks:
            ctx["compliance_result"] = {
                "status": "SKIPPED",
                "reason": "No frameworks specified",
            }
            return

        gwp_results = ctx.get("gwp_results", {})
        livestock = ctx.get("livestock_results", {})
        cropland = ctx.get("cropland_results", {})

        compliance_data: Dict[str, Any] = {
            "tenant_id": ctx["tenant_id"],
            "farm_id": ctx.get("farm_id", ""),
            "reporting_year": ctx.get("reporting_year"),
            "gwp_source": ctx["gwp_source"],
            "calculation_method": ctx["calculation_method"],
            "source_count": ctx.get("source_count", 0),
            "source_types": sorted(set(
                s["source_type"]
                for s in ctx.get("emission_sources", [])
            )),
            "animal_types": ctx.get("animal_types_present", []),
            "crop_types": ctx.get("crop_types_present", []),
            "farm_types": ctx.get("farm_types", []),
            # Totals
            "total_co2e_tonnes": gwp_results.get(
                "total_co2e_tonnes", "0"
            ),
            "total_co2_tonnes": gwp_results.get("total_co2_tonnes", "0"),
            "total_ch4_tonnes": gwp_results.get("total_ch4_tonnes", "0"),
            "total_n2o_tonnes": gwp_results.get("total_n2o_tonnes", "0"),
            # Livestock
            "livestock_status": livestock.get("status", "SKIPPED"),
            "enteric_ch4_tonnes": livestock.get(
                "enteric", {}
            ).get("total_ch4_tonnes", "0"),
            "manure_ch4_tonnes": livestock.get(
                "manure", {}
            ).get("total_ch4_tonnes", "0"),
            "manure_n2o_tonnes": livestock.get(
                "manure", {}
            ).get("total_n2o_tonnes", "0"),
            # Cropland
            "cropland_status": cropland.get("status", "SKIPPED"),
            "soil_n2o_tonnes": cropland.get(
                "soil", {}
            ).get("total_n2o_tonnes", "0"),
            "rice_ch4_tonnes": cropland.get(
                "rice", {}
            ).get("total_ch4_tonnes", "0"),
            "liming_co2_tonnes": cropland.get(
                "liming", {}
            ).get("total_co2_tonnes", "0"),
            "urea_co2_tonnes": cropland.get(
                "urea", {}
            ).get("total_co2_tonnes", "0"),
            # Per-source CO2e
            "per_source_co2e": gwp_results.get("per_source_co2e", {}),
            # Separation flags
            "has_livestock": livestock.get("status") == "COMPLETE",
            "has_cropland": cropland.get("status") == "COMPLETE",
            "has_rice_cultivation": cropland.get(
                "rice", {}
            ).get("status") == "COMPLETE",
        }

        try:
            compliance_result = self._compliance_engine.check_compliance(
                compliance_data, frameworks
            )
            ctx["compliance_result"] = compliance_result
        except Exception as e:
            ctx["compliance_result"] = {
                "status": "ERROR",
                "error": str(e),
            }
            logger.error(
                "Compliance check failed: %s", str(e), exc_info=True
            )

        logger.debug(
            "Compliance: %s",
            ctx["compliance_result"].get(
                "overall", {}
            ).get("compliance_status", "UNKNOWN"),
        )

    # ------------------------------------------------------------------
    # Stage 8: Assemble Results
    # ------------------------------------------------------------------

    def _stage_assemble_results(self, ctx: Dict[str, Any]) -> None:
        """Stage 8: Assemble all results into final output.

        Combines livestock and cropland results into unified totals.
        Provides per-gas CO2e breakdown (CO2, CH4, N2O) and
        per-source CO2e breakdown (enteric, manure, soils, rice,
        liming, urea, burning).
        """
        gwp_results = ctx.get("gwp_results", {})
        livestock = ctx.get("livestock_results", {})
        cropland = ctx.get("cropland_results", {})

        # Grand totals from GWP stage
        total_co2e = gwp_results.get("total_co2e_tonnes", "0")
        total_co2 = gwp_results.get("total_co2_tonnes", "0")
        total_ch4 = gwp_results.get("total_ch4_tonnes", "0")
        total_n2o = gwp_results.get("total_n2o_tonnes", "0")

        # Per-gas CO2e
        per_gas: Dict[str, str] = {
            "co2": gwp_results.get("co2_co2e_tonnes", "0"),
            "ch4": gwp_results.get("ch4_co2e_tonnes", "0"),
            "n2o": gwp_results.get("n2o_co2e_tonnes", "0"),
        }

        # Per-source CO2e
        per_source = gwp_results.get("per_source_co2e", {
            "enteric": "0",
            "manure": "0",
            "soils": "0",
            "rice": "0",
            "liming": "0",
            "urea": "0",
            "burning": "0",
        })

        # Collect all per-source results for detailed breakdown
        all_source_results: List[Dict[str, Any]] = []

        # Enteric sources
        enteric_data = livestock.get("enteric", {})
        if enteric_data.get("status") == "COMPLETE":
            for sr in enteric_data.get("sources", []):
                sr_copy = dict(sr)
                sr_copy["emission_category"] = "ENTERIC"
                all_source_results.append(sr_copy)

        # Manure sources
        manure_data = livestock.get("manure", {})
        if manure_data.get("status") == "COMPLETE":
            for sr in manure_data.get("sources", []):
                sr_copy = dict(sr)
                sr_copy["emission_category"] = "MANURE"
                all_source_results.append(sr_copy)

        # Cropland sub-sources
        for sub_key in ("soil", "rice", "liming", "urea", "burning"):
            sub_data = cropland.get(sub_key, {})
            if sub_data.get("status") == "COMPLETE":
                for sr in sub_data.get("sources", []):
                    sr_copy = dict(sr)
                    sr_copy["emission_category"] = sub_key.upper()
                    all_source_results.append(sr_copy)

        # Calculation steps for audit trail
        calculation_steps: List[Dict[str, str]] = []
        calculation_steps.append({
            "step": "1_ENTERIC_CH4",
            "description": "Enteric fermentation CH4",
            "value_tonnes": livestock.get(
                "enteric", {}
            ).get("total_ch4_tonnes", "0"),
            "value_co2e": per_source.get("enteric", "0"),
        })
        calculation_steps.append({
            "step": "2_MANURE_CH4_N2O",
            "description": "Manure management CH4 and N2O",
            "value_ch4_tonnes": livestock.get(
                "manure", {}
            ).get("total_ch4_tonnes", "0"),
            "value_n2o_tonnes": livestock.get(
                "manure", {}
            ).get("total_n2o_tonnes", "0"),
            "value_co2e": per_source.get("manure", "0"),
        })
        calculation_steps.append({
            "step": "3_SOIL_N2O",
            "description": "Agricultural soils direct and indirect N2O",
            "value_tonnes": cropland.get(
                "soil", {}
            ).get("total_n2o_tonnes", "0"),
            "value_co2e": per_source.get("soils", "0"),
        })
        calculation_steps.append({
            "step": "4_RICE_CH4",
            "description": "Rice cultivation CH4",
            "value_tonnes": cropland.get(
                "rice", {}
            ).get("total_ch4_tonnes", "0"),
            "value_co2e": per_source.get("rice", "0"),
        })
        calculation_steps.append({
            "step": "5_LIMING_CO2",
            "description": "Liming CO2 (limestone/dolomite)",
            "value_tonnes": cropland.get(
                "liming", {}
            ).get("total_co2_tonnes", "0"),
            "value_co2e": per_source.get("liming", "0"),
        })
        calculation_steps.append({
            "step": "6_UREA_CO2",
            "description": "Urea application CO2",
            "value_tonnes": cropland.get(
                "urea", {}
            ).get("total_co2_tonnes", "0"),
            "value_co2e": per_source.get("urea", "0"),
        })
        calculation_steps.append({
            "step": "7_BURNING_CH4_N2O",
            "description": "Field burning CH4 and N2O",
            "value_ch4_tonnes": cropland.get(
                "burning", {}
            ).get("total_ch4_tonnes", "0"),
            "value_n2o_tonnes": cropland.get(
                "burning", {}
            ).get("total_n2o_tonnes", "0"),
            "value_co2e": per_source.get("burning", "0"),
        })
        calculation_steps.append({
            "step": "8_TOTAL",
            "description": (
                "Total agricultural emissions "
                "(CO2 + CH4 CO2e + N2O CO2e)"
            ),
            "value_tco2e": total_co2e,
        })

        ctx["assembled"] = {
            "total_co2e_tonnes": total_co2e,
            "total_co2_tonnes": total_co2,
            "total_ch4_tonnes": total_ch4,
            "total_n2o_tonnes": total_n2o,
            "per_gas": per_gas,
            "per_source": per_source,
            "source_results": all_source_results,
            "calculation_steps": calculation_steps,
            "gwp_source": gwp_results.get("gwp_source", "AR6"),
            "gwp_co2": gwp_results.get("gwp_co2", "1"),
            "gwp_ch4": gwp_results.get("gwp_ch4", "27.0"),
            "gwp_n2o": gwp_results.get("gwp_n2o", "273"),
            "livestock_total_co2e": gwp_results.get(
                "livestock_total_co2e", "0"
            ),
            "cropland_total_co2e": gwp_results.get(
                "cropland_total_co2e", "0"
            ),
        }

        ctx["assembly_status"] = "COMPLETE"
        logger.debug(
            "Assembly complete: total=%s tCO2e, per_gas=%s, per_source=%s",
            total_co2e, per_gas, per_source,
        )

    # ------------------------------------------------------------------
    # Main Execute
    # ------------------------------------------------------------------

    def execute(
        self,
        request: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute the full 8-stage pipeline for a single calculation.

        Runs all stages sequentially.  Stage 1 (VALIDATE_INPUT) failure
        is fatal and aborts remaining stages.  Stages 4-5 (calculation
        engines) can fail independently without aborting the pipeline;
        partial results are still assembled.

        Args:
            request: Calculation request dictionary containing
                tenant_id, farm_id, emission_sources, and optional
                parameters.

        Returns:
            Complete calculation result dictionary with all stage
            outputs, per-gas totals, per-source CO2e breakdown,
            compliance results, and SHA-256 provenance hash.
        """
        pipeline_start = time.monotonic()
        calculation_id = str(uuid4())

        with self._lock:
            self._total_executions += 1

        # Initialize context
        ctx: Dict[str, Any] = {
            "calculation_id": calculation_id,
            "request": request,
            "stages_completed": [],
            "stages_failed": [],
            "errors": [],
            "stage_timings": {},
            "provenance_chain": [],
        }

        # Stage definitions
        stages = [
            (PipelineStage.VALIDATE_INPUT, self._stage_validate_input),
            (PipelineStage.CLASSIFY_SOURCES, self._stage_classify_sources),
            (PipelineStage.LOOKUP_FACTORS, self._stage_lookup_factors),
            (PipelineStage.CALCULATE_LIVESTOCK, self._stage_calculate_livestock),
            (PipelineStage.CALCULATE_CROPLAND, self._stage_calculate_cropland),
            (PipelineStage.APPLY_GWP, self._stage_apply_gwp),
            (PipelineStage.CHECK_COMPLIANCE, self._stage_check_compliance),
            (PipelineStage.ASSEMBLE_RESULTS, self._stage_assemble_results),
        ]

        # Execute stages sequentially
        abort = False
        for stage, func in stages:
            if abort:
                ctx["stages_failed"].append(stage.value)
                continue

            _, error = self._run_stage(stage, ctx, func)

            # Abort on validation errors (Stage 1 failure is fatal)
            if error and stage == PipelineStage.VALIDATE_INPUT:
                abort = True

        # Build final result
        pipeline_time = round((time.monotonic() - pipeline_start) * 1000, 3)
        is_success = len(ctx["stages_failed"]) == 0
        assembled = ctx.get("assembled", {})

        result: Dict[str, Any] = {
            "calculation_id": calculation_id,
            "tenant_id": ctx.get("tenant_id", ""),
            "farm_id": ctx.get("farm_id", ""),
            "status": (
                "SUCCESS" if is_success
                else "PARTIAL" if ctx["stages_completed"]
                else "FAILED"
            ),
            "stages_completed": ctx["stages_completed"],
            "stages_failed": ctx["stages_failed"],

            # Totals
            "total_co2e_tonnes": assembled.get(
                "total_co2e_tonnes", "0"
            ),
            "total_co2_tonnes": assembled.get("total_co2_tonnes", "0"),
            "total_ch4_tonnes": assembled.get("total_ch4_tonnes", "0"),
            "total_n2o_tonnes": assembled.get("total_n2o_tonnes", "0"),

            # Per-gas CO2e breakdown
            "per_gas": assembled.get("per_gas", {
                "co2": "0", "ch4": "0", "n2o": "0",
            }),

            # Per-source CO2e breakdown
            "per_source": assembled.get("per_source", {
                "enteric": "0", "manure": "0", "soils": "0",
                "rice": "0", "liming": "0", "urea": "0",
                "burning": "0",
            }),

            # Livestock vs cropland subtotals
            "livestock_total_co2e": assembled.get(
                "livestock_total_co2e", "0"
            ),
            "cropland_total_co2e": assembled.get(
                "cropland_total_co2e", "0"
            ),

            # Source-level results (detailed)
            "source_results": assembled.get("source_results", []),

            # Engine results (detailed)
            "results": {
                "livestock": ctx.get("livestock_results", {}),
                "cropland": ctx.get("cropland_results", {}),
                "gwp": ctx.get("gwp_results", {}),
            },

            # Compliance
            "compliance": ctx.get("compliance_result", {}),

            # Context
            "gwp_source": ctx.get("gwp_source", "AR6"),
            "calculation_method": ctx.get("calculation_method", ""),
            "reporting_year": ctx.get("reporting_year"),
            "farm_types": ctx.get("farm_types", []),
            "animal_types_present": ctx.get("animal_types_present", []),
            "crop_types_present": ctx.get("crop_types_present", []),
            "source_count": ctx.get("source_count", 0),
            "total_head_count": str(
                ctx.get("total_head_count", _ZERO)
            ),

            # Audit trail
            "calculation_steps": assembled.get("calculation_steps", []),
            "errors": ctx["errors"],
            "stage_timings": ctx["stage_timings"],
            "provenance_chain": ctx["provenance_chain"],
            "processing_time_ms": pipeline_time,
            "calculated_at": _utcnow_iso(),
        }

        # Final provenance hash over the entire result
        result["provenance_hash"] = f"sha256:{_compute_hash(result)}"

        # Record provenance if tracker available
        if self._provenance_tracker is not None:
            try:
                self._provenance_tracker.record(
                    entity_type="CALCULATION",
                    entity_id=calculation_id,
                    action="CALCULATE",
                    data=result,
                    metadata={
                        "tenant_id": ctx.get("tenant_id", ""),
                        "farm_id": ctx.get("farm_id", ""),
                        "status": result["status"],
                        "total_tco2e": result["total_co2e_tonnes"],
                    },
                )
            except Exception as e:
                logger.warning(
                    "Provenance recording failed: %s", str(e)
                )

        logger.info(
            "Pipeline execute: id=%s, status=%s, "
            "stages=%d/%d, total_tco2e=%s, time=%.3fms",
            calculation_id, result["status"],
            len(ctx["stages_completed"]), len(stages),
            result["total_co2e_tonnes"], pipeline_time,
        )
        return result

    # ------------------------------------------------------------------
    # Batch Execute
    # ------------------------------------------------------------------

    def execute_batch(
        self,
        requests: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Execute the pipeline for a batch of calculation requests.

        Processes each request sequentially through the full 8-stage
        pipeline.  Failures in individual requests do not abort the
        batch.  Produces aggregate summaries by emission source and
        by animal/crop type.

        Args:
            requests: List of calculation request dictionaries.

        Returns:
            Batch results with individual results, aggregate totals
            by source type and animal/crop type, and batch-level
            provenance hash.
        """
        batch_start = time.monotonic()
        batch_id = str(uuid4())

        with self._lock:
            self._total_batches += 1

        results: List[Dict[str, Any]] = []
        total_emissions = _ZERO
        total_co2 = _ZERO
        total_ch4 = _ZERO
        total_n2o = _ZERO
        success_count = 0
        failure_count = 0

        for i, request in enumerate(requests):
            try:
                result = self.execute(request)
                results.append(result)

                if result["status"] in ("SUCCESS", "PARTIAL"):
                    success_count += 1
                    total_emissions += _safe_decimal(
                        result.get("total_co2e_tonnes", "0")
                    )
                    total_co2 += _safe_decimal(
                        result.get("total_co2_tonnes", "0")
                    )
                    total_ch4 += _safe_decimal(
                        result.get("total_ch4_tonnes", "0")
                    )
                    total_n2o += _safe_decimal(
                        result.get("total_n2o_tonnes", "0")
                    )
                else:
                    failure_count += 1
            except Exception as e:
                failure_count += 1
                results.append({
                    "calculation_id": str(uuid4()),
                    "status": "FAILED",
                    "errors": [f"Batch item {i} failed: {str(e)}"],
                    "request_index": i,
                })
                logger.error(
                    "Batch item %d failed: %s", i, str(e), exc_info=True
                )

        batch_time = round((time.monotonic() - batch_start) * 1000, 3)

        # Aggregate by emission source
        by_source: Dict[str, Decimal] = defaultdict(lambda: _ZERO)
        # Aggregate by animal type
        by_animal: Dict[str, Decimal] = defaultdict(lambda: _ZERO)
        # Aggregate by crop type
        by_crop: Dict[str, Decimal] = defaultdict(lambda: _ZERO)

        for r in results:
            if r.get("status") in ("SUCCESS", "PARTIAL"):
                per_source = r.get("per_source", {})
                for source_key, co2e_val in per_source.items():
                    by_source[source_key] += _safe_decimal(co2e_val)

                for sr in r.get("source_results", []):
                    sr_co2e = _ZERO
                    sr_ch4 = _safe_decimal(sr.get("ch4_tonnes", "0"))
                    sr_n2o = _safe_decimal(sr.get("n2o_tonnes", "0"))
                    sr_co2 = _safe_decimal(sr.get("co2_tonnes", "0"))
                    # Approximate CO2e using AR6 defaults
                    sr_co2e = sr_co2 + sr_ch4 * Decimal("27") + sr_n2o * Decimal("273")

                    animal_type = sr.get("animal_type")
                    if animal_type:
                        by_animal[animal_type] += sr_co2e

                    crop_type = sr.get("crop_type")
                    if crop_type:
                        by_crop[crop_type] += sr_co2e

        batch_result: Dict[str, Any] = {
            "batch_id": batch_id,
            "status": (
                "SUCCESS" if failure_count == 0
                else "PARTIAL" if success_count > 0
                else "FAILED"
            ),
            "total_requests": len(requests),
            "success_count": success_count,
            "failure_count": failure_count,
            "total_co2e_tonnes": str(_quantize(total_emissions)),
            "total_co2_tonnes": str(_quantize(total_co2)),
            "total_ch4_tonnes": str(_quantize(total_ch4)),
            "total_n2o_tonnes": str(_quantize(total_n2o)),
            "by_emission_source": {
                k: str(_quantize(v))
                for k, v in sorted(by_source.items())
            },
            "by_animal_type": {
                k: str(_quantize(v))
                for k, v in sorted(by_animal.items())
            },
            "by_crop_type": {
                k: str(_quantize(v))
                for k, v in sorted(by_crop.items())
            },
            "results": results,
            "processing_time_ms": batch_time,
            "calculated_at": _utcnow_iso(),
        }
        batch_result["provenance_hash"] = (
            f"sha256:{_compute_hash(batch_result)}"
        )

        # Record batch provenance
        if self._provenance_tracker is not None:
            try:
                self._provenance_tracker.record(
                    entity_type="BATCH",
                    entity_id=batch_id,
                    action="CALCULATE",
                    data={
                        "total_requests": len(requests),
                        "success_count": success_count,
                        "failure_count": failure_count,
                        "total_tco2e": str(_quantize(total_emissions)),
                    },
                    metadata={"batch_id": batch_id},
                )
            except Exception as e:
                logger.warning(
                    "Batch provenance recording failed: %s", str(e)
                )

        logger.info(
            "Batch execute: id=%s, total=%d, success=%d, "
            "failed=%d, tco2e=%s, time=%.3fms",
            batch_id, len(requests), success_count,
            failure_count, total_emissions, batch_time,
        )
        return batch_result

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_statistics(self) -> Dict[str, Any]:
        """Return pipeline engine statistics.

        Returns:
            Dictionary containing engine version, uptime, execution
            counts, average stage timings, and engine availability.
        """
        with self._lock:
            avg_timings: Dict[str, Optional[float]] = {}
            for stage, times in self._stage_timings.items():
                if times:
                    avg_timings[stage] = round(
                        sum(times) / len(times), 3
                    )
                else:
                    avg_timings[stage] = None

            return {
                "engine": "AgriculturalPipelineEngine",
                "version": "1.0.0",
                "created_at": self._created_at.isoformat(),
                "total_executions": self._total_executions,
                "total_batches": self._total_batches,
                "stages": [s.value for s in PipelineStage],
                "stage_count": len(PipelineStage),
                "avg_stage_timings_ms": avg_timings,
                "engines": {
                    "db": self._db_engine is not None,
                    "enteric": self._enteric_engine is not None,
                    "manure": self._manure_engine is not None,
                    "cropland": self._cropland_engine is not None,
                    "compliance": self._compliance_engine is not None,
                    "provenance": self._provenance_tracker is not None,
                },
                "available_imports": {
                    "db": _DB_ENGINE_AVAILABLE,
                    "enteric": _ENTERIC_ENGINE_AVAILABLE,
                    "manure": _MANURE_ENGINE_AVAILABLE,
                    "cropland": _CROPLAND_ENGINE_AVAILABLE,
                    "compliance": _COMPLIANCE_ENGINE_AVAILABLE,
                },
            }

    def get_engine_health(self) -> Dict[str, Any]:
        """Return health status of the pipeline and upstream engines.

        Returns:
            Dictionary with overall health, per-engine status, and
            stage availability assessment.
        """
        engines_available = {
            "db": self._db_engine is not None,
            "enteric": self._enteric_engine is not None,
            "manure": self._manure_engine is not None,
            "cropland": self._cropland_engine is not None,
            "compliance": self._compliance_engine is not None,
            "provenance": self._provenance_tracker is not None,
        }

        # Stages that require specific engines
        stage_availability = {
            PipelineStage.VALIDATE_INPUT.value: True,
            PipelineStage.CLASSIFY_SOURCES.value: True,
            PipelineStage.LOOKUP_FACTORS.value: True,  # has fallback
            PipelineStage.CALCULATE_LIVESTOCK.value: True,  # has fallback
            PipelineStage.CALCULATE_CROPLAND.value: True,  # has fallback
            PipelineStage.APPLY_GWP.value: True,
            PipelineStage.CHECK_COMPLIANCE.value: engines_available["compliance"],
            PipelineStage.ASSEMBLE_RESULTS.value: True,
        }

        all_critical_available = all(stage_availability.values())

        return {
            "healthy": True,  # Pipeline always works with fallbacks
            "all_engines_available": all(engines_available.values()),
            "all_stages_available": all_critical_available,
            "engines": engines_available,
            "stage_availability": stage_availability,
            "total_executions": self._total_executions,
            "total_batches": self._total_batches,
        }

    def reset(self) -> None:
        """Reset pipeline state. Intended for testing teardown."""
        with self._lock:
            self._total_executions = 0
            self._total_batches = 0
            self._stage_timings = {
                stage.value: [] for stage in PipelineStage
            }

        # Reset upstream engines
        for engine in [
            self._db_engine,
            self._enteric_engine,
            self._manure_engine,
            self._cropland_engine,
            self._compliance_engine,
        ]:
            if engine is not None and hasattr(engine, "reset"):
                engine.reset()

        # Clear provenance
        if self._provenance_tracker is not None:
            if hasattr(self._provenance_tracker, "clear"):
                self._provenance_tracker.clear()
            elif hasattr(self._provenance_tracker, "clear_trail"):
                self._provenance_tracker.clear_trail()

        logger.info(
            "AgriculturalPipelineEngine and all upstream engines reset"
        )
