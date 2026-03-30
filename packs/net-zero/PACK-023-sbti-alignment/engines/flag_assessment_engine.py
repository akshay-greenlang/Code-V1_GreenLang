# -*- coding: utf-8 -*-
"""
FLAGAssessmentEngine - PACK-023 SBTi Alignment Engine 5
==========================================================

Forest, Land and Agriculture (FLAG) emissions assessment engine for
SBTi target-setting.  Evaluates FLAG emissions across 11 commodity
categories, determines whether a separate FLAG target is required
(20% trigger threshold), calculates commodity-specific emission
allocations, computes FLAG reduction pathways at 3.03%/yr linear
reduction, and validates no-deforestation commitments against the
SBTi FLAG Guidance V1.1.

FLAG Commodities (11):
    cattle, soy, palm_oil, timber, cocoa, coffee, rubber,
    rice, sugarcane, maize, wheat

FLAG Trigger Assessment:
    Companies whose FLAG emissions (including land-use change,
    agricultural processes, and deforestation-linked supply chain
    emissions) constitute >= 20% of total Scope 1+2+3 emissions
    are required to set separate FLAG targets per SBTi FLAG
    Guidance V1.1, Section 3.

FLAG Pathway Calculation:
    The FLAG pathway uses a 3.03%/yr linear reduction from the
    base year.  This rate is derived from the global FLAG sector
    1.5C-aligned pathway published in SBTi FLAG Guidance V1.1,
    Section 5, Table 5.1.

    E(t) = E(base) * max(0, 1 - 0.0303 * (t - base_year))

    This linear formulation means FLAG emissions reach zero
    approximately 33 years after the base year.

No-Deforestation Commitment:
    SBTi FLAG Guidance V1.1 Section 6 requires companies with
    FLAG targets to commit to zero deforestation and zero
    conversion of natural ecosystems by 2025 at the latest,
    covering all commodities in the company's value chain.

Land Use Change (LUC) Emissions:
    LUC emissions are quantified per commodity using IPCC AR6
    emission factors for land conversion (forest to cropland,
    forest to pasture, peatland drainage).  These are included
    in the FLAG emissions boundary.

FLAG Target Types:
    - ABSOLUTE: Absolute reduction in FLAG emissions (tCO2e)
    - INTENSITY: Intensity reduction per unit of commodity output
      (tCO2e per tonne of commodity)

Per-Commodity Emission Categories:
    Each commodity's emissions are broken into:
    - land_use_change_tco2e: Deforestation and land conversion
    - agricultural_process_tco2e: Enteric fermentation, manure,
      rice cultivation, soil N2O, etc.
    - input_production_tco2e: Fertilizer manufacturing, feed
    - on_farm_energy_tco2e: Machinery, irrigation energy
    - post_harvest_tco2e: Drying, storage, initial processing

Regulatory References:
    - SBTi FLAG Guidance V1.1 (2022) - Primary standard
    - SBTi Corporate Manual V5.3 (2024) - Integration with
      corporate targets
    - SBTi Corporate Net-Zero Standard V1.3 (2024) - Long-term
      FLAG targets
    - IPCC AR6 WG3 (2022) - AFOLU emission factors
    - GHG Protocol Agricultural Guidance (2014)
    - GHG Protocol Land Sector and Removals Guidance (2022)
    - Accountability Framework initiative (AFi) - No-deforestation
    - CDP Forests Questionnaire (2024) - Commodity disclosure

Zero-Hallucination:
    - FLAG rate (3.03%/yr) from SBTi FLAG Guidance V1.1 Table 5.1
    - 20% trigger threshold from SBTi FLAG Guidance V1.1 Section 3
    - All reductions computed with deterministic Decimal arithmetic
    - Emission factors from published IPCC AR6 tables only
    - No LLM involvement in any calculation path
    - SHA-256 provenance hash on every result

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-023 SBTi Alignment
Engine:  5 of 10
Status:  Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator, model_validator

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

    Excludes volatile fields (calculated_at, processing_time_ms,
    provenance_hash) from the hash computation so that two
    semantically identical results produce the same hash.
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
            if k not in (
                "calculated_at", "processing_time_ms", "provenance_hash"
            )
        }
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

def _decimal(value: Any) -> Decimal:
    """Safely convert a value to Decimal.

    Args:
        value: Value to convert (int, float, str, Decimal).

    Returns:
        Decimal representation; Decimal("0") on conversion failure.
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
        default: Value to return when denominator is zero.

    Returns:
        Result of division or *default*.
    """
    if denominator == Decimal("0"):
        return default
    return numerator / denominator

def _safe_pct(part: Decimal, whole: Decimal) -> Decimal:
    """Compute percentage safely (part / whole * 100).

    Args:
        part: Numerator value.
        whole: Denominator value.

    Returns:
        Percentage as Decimal; 0 if *whole* is zero.
    """
    return _safe_divide(part * Decimal("100"), whole)

def _round_val(value: Decimal, places: int = 6) -> Decimal:
    """Round a Decimal to *places* using ROUND_HALF_UP.

    Args:
        value: Value to round.
        places: Number of decimal places.

    Returns:
        Rounded Decimal.
    """
    quantize_str = "0." + "0" * places
    return value.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP)

def _round3(value: float) -> float:
    """Round to 3 decimal places using ROUND_HALF_UP.

    Args:
        value: Float to round.

    Returns:
        Rounded float.
    """
    return float(
        Decimal(str(value)).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)
    )

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class FLAGCommodity(str, Enum):
    """The 11 FLAG commodity categories per SBTi FLAG Guidance V1.1.

    Each commodity has its own emission profile covering land-use
    change, agricultural processes, input production, on-farm
    energy, and post-harvest activities.
    """
    CATTLE = "cattle"
    SOY = "soy"
    PALM_OIL = "palm_oil"
    TIMBER = "timber"
    COCOA = "cocoa"
    COFFEE = "coffee"
    RUBBER = "rubber"
    RICE = "rice"
    SUGARCANE = "sugarcane"
    MAIZE = "maize"
    WHEAT = "wheat"

class FLAGTargetType(str, Enum):
    """FLAG target type classification.

    ABSOLUTE: Total FLAG emissions reduction in tCO2e.
    INTENSITY: Emissions intensity per unit of commodity output.
    """
    ABSOLUTE = "absolute"
    INTENSITY = "intensity"

class FLAGTriggerStatus(str, Enum):
    """Whether a separate FLAG target is required.

    REQUIRED: FLAG emissions >= 20% of total -- target mandatory.
    NOT_REQUIRED: FLAG emissions < 20% of total.
    BORDERLINE: FLAG emissions between 15% and 20% (review advised).
    UNDETERMINED: Insufficient data to assess.
    """
    REQUIRED = "required"
    NOT_REQUIRED = "not_required"
    BORDERLINE = "borderline"
    UNDETERMINED = "undetermined"

class DeforestationCommitmentStatus(str, Enum):
    """Status of no-deforestation commitment.

    COMMITTED: Zero-deforestation commitment made with target date.
    PARTIAL: Commitment covers some but not all commodities.
    NOT_COMMITTED: No deforestation commitment in place.
    EXPIRED: Commitment date has passed without verification.
    """
    COMMITTED = "committed"
    PARTIAL = "partial"
    NOT_COMMITTED = "not_committed"
    EXPIRED = "expired"

class LandUseChangeType(str, Enum):
    """Types of land use change contributing to FLAG emissions.

    FOREST_TO_CROPLAND: Conversion of forest to crop production.
    FOREST_TO_PASTURE: Conversion of forest to grazing land.
    PEATLAND_DRAINAGE: Draining of peatlands for agriculture.
    WETLAND_CONVERSION: Conversion of wetlands.
    GRASSLAND_TO_CROPLAND: Conversion of natural grassland.
    OTHER_CONVERSION: Other natural ecosystem conversions.
    """
    FOREST_TO_CROPLAND = "forest_to_cropland"
    FOREST_TO_PASTURE = "forest_to_pasture"
    PEATLAND_DRAINAGE = "peatland_drainage"
    WETLAND_CONVERSION = "wetland_conversion"
    GRASSLAND_TO_CROPLAND = "grassland_to_cropland"
    OTHER_CONVERSION = "other_conversion"

class EmissionCategory(str, Enum):
    """FLAG emission source categories per commodity.

    LAND_USE_CHANGE: Deforestation and land conversion.
    AGRICULTURAL_PROCESS: Enteric fermentation, manure management,
        rice cultivation, soil N2O emissions.
    INPUT_PRODUCTION: Fertiliser manufacturing, animal feed.
    ON_FARM_ENERGY: Machinery fuel, irrigation pumping.
    POST_HARVEST: Drying, storage, initial processing.
    """
    LAND_USE_CHANGE = "land_use_change"
    AGRICULTURAL_PROCESS = "agricultural_process"
    INPUT_PRODUCTION = "input_production"
    ON_FARM_ENERGY = "on_farm_energy"
    POST_HARVEST = "post_harvest"

class AssessmentConfidence(str, Enum):
    """Data quality / confidence level for FLAG assessment.

    HIGH: Primary data, measured and verified.
    MEDIUM: Secondary data or modelled estimates.
    LOW: Proxy data or spend-based estimates.
    VERY_LOW: Default factors with high uncertainty.
    """
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    VERY_LOW = "very_low"

class PathwayStatus(str, Enum):
    """Status of FLAG pathway compliance.

    ON_TRACK: Current emissions are at or below pathway.
    MINOR_DEVIATION: Within 5% above pathway.
    MAJOR_DEVIATION: Between 5% and 15% above pathway.
    CRITICAL: More than 15% above pathway.
    NOT_STARTED: No tracking data available yet.
    """
    ON_TRACK = "on_track"
    MINOR_DEVIATION = "minor_deviation"
    MAJOR_DEVIATION = "major_deviation"
    CRITICAL = "critical"
    NOT_STARTED = "not_started"

# ---------------------------------------------------------------------------
# Constants -- SBTi FLAG Guidance V1.1
# ---------------------------------------------------------------------------

# FLAG linear annual reduction rate: 3.03%/yr.
# Source: SBTi FLAG Guidance V1.1, Section 5, Table 5.1.
FLAG_ANNUAL_RATE: Decimal = Decimal("0.0303")

# FLAG trigger threshold: 20% of total emissions.
# Source: SBTi FLAG Guidance V1.1, Section 3.
FLAG_TRIGGER_THRESHOLD: Decimal = Decimal("0.20")

# Borderline range for FLAG trigger (15%-20%).
FLAG_BORDERLINE_LOWER: Decimal = Decimal("0.15")

# No-deforestation commitment deadline year.
# Source: SBTi FLAG Guidance V1.1, Section 6.
NO_DEFORESTATION_DEADLINE: int = 2025

# Minimum FLAG target coverage of FLAG emissions.
FLAG_COVERAGE_MIN: Decimal = Decimal("0.95")

# Maximum years for near-term FLAG target.
FLAG_NEAR_TERM_MAX_YEARS: int = 10

# Minimum years for near-term FLAG target.
FLAG_NEAR_TERM_MIN_YEARS: int = 5

# Long-term FLAG target year ceiling.
FLAG_LONG_TERM_MAX_YEAR: int = 2050

# Base year minimum.
FLAG_BASE_YEAR_MIN: int = 2015

# Minimum long-term FLAG reduction by 2050 (%).
FLAG_LONG_TERM_MIN_REDUCTION_PCT: Decimal = Decimal("72.0")

# Net-zero FLAG residual max (% of base year).
FLAG_NET_ZERO_MAX_RESIDUAL_PCT: Decimal = Decimal("10.0")

# Number of FLAG commodity categories.
FLAG_COMMODITY_COUNT: int = 11

# ---------------------------------------------------------------------------
# Commodity Default Emission Factors
# ---------------------------------------------------------------------------
# These default emission factors (tCO2e per tonne of commodity) are
# representative averages from IPCC AR6 WG3 Chapter 7 and FAO datasets.
# They serve as fallbacks when primary data is unavailable.
# Companies should use site-specific data where possible.

COMMODITY_DEFAULT_EMISSION_FACTORS: Dict[str, Dict[str, Decimal]] = {
    FLAGCommodity.CATTLE.value: {
        EmissionCategory.LAND_USE_CHANGE.value: Decimal("5.200"),
        EmissionCategory.AGRICULTURAL_PROCESS.value: Decimal("22.800"),
        EmissionCategory.INPUT_PRODUCTION.value: Decimal("2.100"),
        EmissionCategory.ON_FARM_ENERGY.value: Decimal("0.450"),
        EmissionCategory.POST_HARVEST.value: Decimal("0.350"),
    },
    FLAGCommodity.SOY.value: {
        EmissionCategory.LAND_USE_CHANGE.value: Decimal("1.850"),
        EmissionCategory.AGRICULTURAL_PROCESS.value: Decimal("0.320"),
        EmissionCategory.INPUT_PRODUCTION.value: Decimal("0.180"),
        EmissionCategory.ON_FARM_ENERGY.value: Decimal("0.095"),
        EmissionCategory.POST_HARVEST.value: Decimal("0.065"),
    },
    FLAGCommodity.PALM_OIL.value: {
        EmissionCategory.LAND_USE_CHANGE.value: Decimal("3.600"),
        EmissionCategory.AGRICULTURAL_PROCESS.value: Decimal("1.250"),
        EmissionCategory.INPUT_PRODUCTION.value: Decimal("0.280"),
        EmissionCategory.ON_FARM_ENERGY.value: Decimal("0.120"),
        EmissionCategory.POST_HARVEST.value: Decimal("0.190"),
    },
    FLAGCommodity.TIMBER.value: {
        EmissionCategory.LAND_USE_CHANGE.value: Decimal("2.400"),
        EmissionCategory.AGRICULTURAL_PROCESS.value: Decimal("0.150"),
        EmissionCategory.INPUT_PRODUCTION.value: Decimal("0.085"),
        EmissionCategory.ON_FARM_ENERGY.value: Decimal("0.310"),
        EmissionCategory.POST_HARVEST.value: Decimal("0.220"),
    },
    FLAGCommodity.COCOA.value: {
        EmissionCategory.LAND_USE_CHANGE.value: Decimal("2.100"),
        EmissionCategory.AGRICULTURAL_PROCESS.value: Decimal("0.480"),
        EmissionCategory.INPUT_PRODUCTION.value: Decimal("0.130"),
        EmissionCategory.ON_FARM_ENERGY.value: Decimal("0.065"),
        EmissionCategory.POST_HARVEST.value: Decimal("0.110"),
    },
    FLAGCommodity.COFFEE.value: {
        EmissionCategory.LAND_USE_CHANGE.value: Decimal("1.750"),
        EmissionCategory.AGRICULTURAL_PROCESS.value: Decimal("0.620"),
        EmissionCategory.INPUT_PRODUCTION.value: Decimal("0.210"),
        EmissionCategory.ON_FARM_ENERGY.value: Decimal("0.085"),
        EmissionCategory.POST_HARVEST.value: Decimal("0.155"),
    },
    FLAGCommodity.RUBBER.value: {
        EmissionCategory.LAND_USE_CHANGE.value: Decimal("2.800"),
        EmissionCategory.AGRICULTURAL_PROCESS.value: Decimal("0.350"),
        EmissionCategory.INPUT_PRODUCTION.value: Decimal("0.165"),
        EmissionCategory.ON_FARM_ENERGY.value: Decimal("0.095"),
        EmissionCategory.POST_HARVEST.value: Decimal("0.130"),
    },
    FLAGCommodity.RICE.value: {
        EmissionCategory.LAND_USE_CHANGE.value: Decimal("0.450"),
        EmissionCategory.AGRICULTURAL_PROCESS.value: Decimal("1.480"),
        EmissionCategory.INPUT_PRODUCTION.value: Decimal("0.240"),
        EmissionCategory.ON_FARM_ENERGY.value: Decimal("0.180"),
        EmissionCategory.POST_HARVEST.value: Decimal("0.095"),
    },
    FLAGCommodity.SUGARCANE.value: {
        EmissionCategory.LAND_USE_CHANGE.value: Decimal("0.680"),
        EmissionCategory.AGRICULTURAL_PROCESS.value: Decimal("0.420"),
        EmissionCategory.INPUT_PRODUCTION.value: Decimal("0.155"),
        EmissionCategory.ON_FARM_ENERGY.value: Decimal("0.110"),
        EmissionCategory.POST_HARVEST.value: Decimal("0.085"),
    },
    FLAGCommodity.MAIZE.value: {
        EmissionCategory.LAND_USE_CHANGE.value: Decimal("0.520"),
        EmissionCategory.AGRICULTURAL_PROCESS.value: Decimal("0.380"),
        EmissionCategory.INPUT_PRODUCTION.value: Decimal("0.290"),
        EmissionCategory.ON_FARM_ENERGY.value: Decimal("0.135"),
        EmissionCategory.POST_HARVEST.value: Decimal("0.075"),
    },
    FLAGCommodity.WHEAT.value: {
        EmissionCategory.LAND_USE_CHANGE.value: Decimal("0.350"),
        EmissionCategory.AGRICULTURAL_PROCESS.value: Decimal("0.310"),
        EmissionCategory.INPUT_PRODUCTION.value: Decimal("0.260"),
        EmissionCategory.ON_FARM_ENERGY.value: Decimal("0.120"),
        EmissionCategory.POST_HARVEST.value: Decimal("0.065"),
    },
}

# ---------------------------------------------------------------------------
# Land Use Change Emission Factors (tCO2e per hectare)
# Source: IPCC AR6 WG3 Chapter 7, Table 7.1
# ---------------------------------------------------------------------------

LUC_EMISSION_FACTORS: Dict[str, Decimal] = {
    LandUseChangeType.FOREST_TO_CROPLAND.value: Decimal("450.0"),
    LandUseChangeType.FOREST_TO_PASTURE.value: Decimal("380.0"),
    LandUseChangeType.PEATLAND_DRAINAGE.value: Decimal("620.0"),
    LandUseChangeType.WETLAND_CONVERSION.value: Decimal("350.0"),
    LandUseChangeType.GRASSLAND_TO_CROPLAND.value: Decimal("120.0"),
    LandUseChangeType.OTHER_CONVERSION.value: Decimal("200.0"),
}

# ---------------------------------------------------------------------------
# Commodity Deforestation Risk Ratings
# ---------------------------------------------------------------------------
# Risk classification for deforestation linkage per commodity.
# Source: Accountability Framework initiative (AFi) risk assessments,
# Global Forest Watch commodity risk data (2023).

COMMODITY_DEFORESTATION_RISK: Dict[str, str] = {
    FLAGCommodity.CATTLE.value: "very_high",
    FLAGCommodity.SOY.value: "very_high",
    FLAGCommodity.PALM_OIL.value: "very_high",
    FLAGCommodity.TIMBER.value: "high",
    FLAGCommodity.COCOA.value: "high",
    FLAGCommodity.COFFEE.value: "medium",
    FLAGCommodity.RUBBER.value: "high",
    FLAGCommodity.RICE.value: "low",
    FLAGCommodity.SUGARCANE.value: "medium",
    FLAGCommodity.MAIZE.value: "low",
    FLAGCommodity.WHEAT.value: "low",
}

# Commodity intensity units (output denominator).
COMMODITY_INTENSITY_UNITS: Dict[str, str] = {
    FLAGCommodity.CATTLE.value: "tCO2e/t_liveweight",
    FLAGCommodity.SOY.value: "tCO2e/t_soy",
    FLAGCommodity.PALM_OIL.value: "tCO2e/t_cpo",
    FLAGCommodity.TIMBER.value: "tCO2e/m3_roundwood",
    FLAGCommodity.COCOA.value: "tCO2e/t_cocoa_beans",
    FLAGCommodity.COFFEE.value: "tCO2e/t_green_coffee",
    FLAGCommodity.RUBBER.value: "tCO2e/t_dry_rubber",
    FLAGCommodity.RICE.value: "tCO2e/t_paddy",
    FLAGCommodity.SUGARCANE.value: "tCO2e/t_sugarcane",
    FLAGCommodity.MAIZE.value: "tCO2e/t_maize",
    FLAGCommodity.WHEAT.value: "tCO2e/t_wheat",
}

# Valid FLAG commodities set for quick lookup.
VALID_COMMODITIES: set = {c.value for c in FLAGCommodity}

# ---------------------------------------------------------------------------
# Pydantic Models -- Input
# ---------------------------------------------------------------------------

class LandUseChangeEntry(BaseModel):
    """A single land use change event contributing to FLAG emissions.

    Attributes:
        luc_type: Type of land conversion.
        area_hectares: Area affected in hectares.
        emission_factor_tco2e_per_ha: Custom emission factor (optional).
        year: Year of conversion.
        commodity: Associated commodity (if known).
        region: Geographic region of conversion.
        calculated_emissions_tco2e: Pre-calculated emissions (if provided).
    """
    luc_type: str = Field(
        default=LandUseChangeType.FOREST_TO_CROPLAND.value,
        description="Type of land conversion",
    )
    area_hectares: Decimal = Field(
        default=Decimal("0"), ge=0,
        description="Area affected in hectares",
    )
    emission_factor_tco2e_per_ha: Optional[Decimal] = Field(
        default=None, ge=0,
        description="Custom emission factor (tCO2e/ha)",
    )
    year: int = Field(
        default=0, ge=0,
        description="Year of conversion event",
    )
    commodity: str = Field(
        default="", max_length=50,
        description="Associated commodity",
    )
    region: str = Field(
        default="", max_length=200,
        description="Geographic region",
    )
    calculated_emissions_tco2e: Optional[Decimal] = Field(
        default=None, ge=0,
        description="Pre-calculated emissions (tCO2e)",
    )

class CommodityEmissionBreakdown(BaseModel):
    """Per-category emission breakdown for a single FLAG commodity.

    Attributes:
        land_use_change_tco2e: LUC emissions (deforestation/conversion).
        agricultural_process_tco2e: Agricultural process emissions.
        input_production_tco2e: Input production emissions.
        on_farm_energy_tco2e: On-farm energy emissions.
        post_harvest_tco2e: Post-harvest emissions.
    """
    land_use_change_tco2e: Decimal = Field(
        default=Decimal("0"), ge=0,
        description="LUC emissions (tCO2e)",
    )
    agricultural_process_tco2e: Decimal = Field(
        default=Decimal("0"), ge=0,
        description="Agricultural process emissions (tCO2e)",
    )
    input_production_tco2e: Decimal = Field(
        default=Decimal("0"), ge=0,
        description="Input production emissions (tCO2e)",
    )
    on_farm_energy_tco2e: Decimal = Field(
        default=Decimal("0"), ge=0,
        description="On-farm energy emissions (tCO2e)",
    )
    post_harvest_tco2e: Decimal = Field(
        default=Decimal("0"), ge=0,
        description="Post-harvest emissions (tCO2e)",
    )

    @property
    def total_tco2e(self) -> Decimal:
        """Sum of all emission categories."""
        return (
            self.land_use_change_tco2e
            + self.agricultural_process_tco2e
            + self.input_production_tco2e
            + self.on_farm_energy_tco2e
            + self.post_harvest_tco2e
        )

class CommodityInput(BaseModel):
    """Input data for a single commodity in the FLAG assessment.

    Attributes:
        commodity: One of the 11 FLAG commodities.
        production_volume: Annual production volume (commodity units).
        production_unit: Unit of production measurement.
        emission_breakdown: Per-category emission breakdown.
        total_emissions_tco2e: Total commodity emissions (if pre-calculated).
        sourcing_regions: Key sourcing regions.
        has_certification: Whether commodity has sustainability certification.
        certification_standard: Certification standard name.
        traceability_pct: Percentage of supply traceable to origin.
        data_quality: Data quality assessment.
    """
    commodity: str = Field(
        ..., description="FLAG commodity name",
    )
    production_volume: Decimal = Field(
        default=Decimal("0"), ge=0,
        description="Annual production/procurement volume",
    )
    production_unit: str = Field(
        default="tonnes",
        description="Unit of production measurement",
    )
    emission_breakdown: Optional[CommodityEmissionBreakdown] = Field(
        default=None,
        description="Per-category emission breakdown",
    )
    total_emissions_tco2e: Decimal = Field(
        default=Decimal("0"), ge=0,
        description="Total commodity emissions (tCO2e)",
    )
    sourcing_regions: List[str] = Field(
        default_factory=list,
        description="Key sourcing regions",
    )
    has_certification: bool = Field(
        default=False,
        description="Has sustainability certification",
    )
    certification_standard: str = Field(
        default="",
        description="Certification standard (e.g. RSPO, FSC, Rainforest Alliance)",
    )
    traceability_pct: Decimal = Field(
        default=Decimal("0"), ge=0, le=Decimal("100"),
        description="Supply chain traceability percentage",
    )
    data_quality: str = Field(
        default=AssessmentConfidence.MEDIUM.value,
        description="Data quality level",
    )

    @field_validator("commodity")
    @classmethod
    def validate_commodity(cls, v: str) -> str:
        """Validate that commodity is one of the 11 FLAG commodities."""
        v_lower = v.lower().strip()
        if v_lower not in VALID_COMMODITIES:
            raise ValueError(
                f"Invalid commodity '{v}'. Must be one of: "
                f"{', '.join(sorted(VALID_COMMODITIES))}"
            )
        return v_lower

class DeforestationCommitment(BaseModel):
    """No-deforestation and no-conversion commitment details.

    Attributes:
        has_commitment: Whether commitment exists.
        commitment_date: Date commitment was made.
        target_date: Date for achieving zero deforestation.
        covers_all_commodities: Whether all FLAG commodities are covered.
        covered_commodities: List of commodities covered.
        includes_no_conversion: Whether natural ecosystem conversion included.
        verification_mechanism: How commitment is verified.
        is_public: Whether commitment is publicly disclosed.
        aligned_with_afi: Whether aligned with Accountability Framework.
        progress_pct: Estimated progress toward commitment (%).
    """
    has_commitment: bool = Field(
        default=False,
        description="Whether deforestation commitment exists",
    )
    commitment_date: Optional[str] = Field(
        default=None,
        description="Date commitment was made (YYYY-MM-DD)",
    )
    target_date: Optional[str] = Field(
        default=None,
        description="Target date for zero deforestation (YYYY-MM-DD)",
    )
    covers_all_commodities: bool = Field(
        default=False,
        description="Whether all FLAG commodities are covered",
    )
    covered_commodities: List[str] = Field(
        default_factory=list,
        description="Commodities covered by commitment",
    )
    includes_no_conversion: bool = Field(
        default=False,
        description="Whether commitment includes zero conversion of natural ecosystems",
    )
    verification_mechanism: str = Field(
        default="",
        description="Verification mechanism (e.g. satellite monitoring, third-party audit)",
    )
    is_public: bool = Field(
        default=False,
        description="Whether commitment is publicly disclosed",
    )
    aligned_with_afi: bool = Field(
        default=False,
        description="Whether aligned with Accountability Framework initiative",
    )
    progress_pct: Decimal = Field(
        default=Decimal("0"), ge=0, le=Decimal("100"),
        description="Progress toward achieving commitment (%)",
    )

class FLAGAssessmentInput(BaseModel):
    """Complete input for FLAG emissions assessment.

    Attributes:
        entity_name: Reporting entity name.
        base_year: Emissions base year.
        target_year_near_term: Near-term FLAG target year.
        target_year_long_term: Long-term FLAG target year.
        commodities: Per-commodity emission data.
        total_scope1_tco2e: Total Scope 1 emissions (non-FLAG).
        total_scope2_tco2e: Total Scope 2 emissions.
        total_scope3_tco2e: Total Scope 3 emissions (non-FLAG).
        total_flag_tco2e: Total FLAG emissions (if pre-aggregated).
        land_use_changes: Land use change events.
        deforestation_commitment: No-deforestation commitment details.
        preferred_target_type: Preferred FLAG target type.
        current_year: Current reporting year (for progress tracking).
        current_flag_emissions_tco2e: Current year FLAG emissions.
        sector: Company sector for context.
        include_removals: Whether to include carbon removals.
        removals_tco2e: Carbon removals from land sector (tCO2e).
    """
    entity_name: str = Field(
        ..., min_length=1, max_length=300,
        description="Reporting entity name",
    )
    base_year: int = Field(
        ..., ge=2015, le=2030,
        description="Emissions base year",
    )
    target_year_near_term: int = Field(
        default=0, ge=0, le=2040,
        description="Near-term FLAG target year (0 = auto-calculate)",
    )
    target_year_long_term: int = Field(
        default=2050, ge=2035, le=2060,
        description="Long-term FLAG target year",
    )
    commodities: List[CommodityInput] = Field(
        default_factory=list,
        description="Per-commodity emission data",
    )
    total_scope1_tco2e: Decimal = Field(
        default=Decimal("0"), ge=0,
        description="Total Scope 1 emissions (non-FLAG) (tCO2e)",
    )
    total_scope2_tco2e: Decimal = Field(
        default=Decimal("0"), ge=0,
        description="Total Scope 2 emissions (tCO2e)",
    )
    total_scope3_tco2e: Decimal = Field(
        default=Decimal("0"), ge=0,
        description="Total Scope 3 emissions (non-FLAG) (tCO2e)",
    )
    total_flag_tco2e: Decimal = Field(
        default=Decimal("0"), ge=0,
        description="Total FLAG emissions, pre-aggregated (tCO2e)",
    )
    land_use_changes: List[LandUseChangeEntry] = Field(
        default_factory=list,
        description="Land use change events",
    )
    deforestation_commitment: DeforestationCommitment = Field(
        default_factory=DeforestationCommitment,
        description="No-deforestation commitment details",
    )
    preferred_target_type: str = Field(
        default=FLAGTargetType.ABSOLUTE.value,
        description="Preferred FLAG target type (absolute/intensity)",
    )
    current_year: int = Field(
        default=0, ge=0,
        description="Current reporting year (0 = auto)",
    )
    current_flag_emissions_tco2e: Decimal = Field(
        default=Decimal("0"), ge=0,
        description="Current year FLAG emissions (tCO2e)",
    )
    sector: str = Field(
        default="", max_length=200,
        description="Company sector",
    )
    include_removals: bool = Field(
        default=False,
        description="Whether to include carbon removals",
    )
    removals_tco2e: Decimal = Field(
        default=Decimal("0"), ge=0,
        description="Carbon removals from land sector (tCO2e)",
    )

    @field_validator("target_year_near_term")
    @classmethod
    def validate_near_term_year(cls, v: int, info: Any) -> int:
        """Auto-calculate near-term year if zero."""
        if v == 0:
            base = info.data.get("base_year", 2023)
            return base + 7
        return v

    @field_validator("current_year")
    @classmethod
    def validate_current_year(cls, v: int) -> int:
        """Auto-set current year if zero."""
        if v == 0:
            return utcnow().year
        return v

# ---------------------------------------------------------------------------
# Pydantic Models -- Output
# ---------------------------------------------------------------------------

class CommodityAssessment(BaseModel):
    """Assessment result for a single FLAG commodity.

    Attributes:
        commodity: Commodity name.
        total_emissions_tco2e: Total emissions for this commodity.
        emission_breakdown: Per-category breakdown.
        pct_of_flag_total: Percentage of total FLAG emissions.
        pct_of_entity_total: Percentage of total entity emissions.
        intensity_value: Emissions intensity per unit output.
        intensity_unit: Unit of intensity measurement.
        base_year_emissions_tco2e: Emissions in base year.
        target_year_emissions_tco2e: Projected target year emissions.
        required_reduction_pct: Required reduction percentage.
        deforestation_risk: Deforestation risk rating.
        has_certification: Whether commodity is certified.
        data_quality: Data quality level.
        traceability_pct: Supply chain traceability percentage.
        pathway_milestones: Annual pathway milestones.
        confidence: Assessment confidence level.
    """
    commodity: str = Field(default="")
    total_emissions_tco2e: Decimal = Field(default=Decimal("0"))
    emission_breakdown: Optional[CommodityEmissionBreakdown] = Field(None)
    pct_of_flag_total: Decimal = Field(default=Decimal("0"))
    pct_of_entity_total: Decimal = Field(default=Decimal("0"))
    intensity_value: Decimal = Field(default=Decimal("0"))
    intensity_unit: str = Field(default="")
    base_year_emissions_tco2e: Decimal = Field(default=Decimal("0"))
    target_year_emissions_tco2e: Decimal = Field(default=Decimal("0"))
    required_reduction_pct: Decimal = Field(default=Decimal("0"))
    deforestation_risk: str = Field(default="")
    has_certification: bool = Field(default=False)
    data_quality: str = Field(default="")
    traceability_pct: Decimal = Field(default=Decimal("0"))
    pathway_milestones: List[Dict[str, Any]] = Field(default_factory=list)
    confidence: str = Field(default=AssessmentConfidence.MEDIUM.value)

class FLAGTriggerAssessment(BaseModel):
    """Result of the FLAG 20% trigger threshold evaluation.

    Attributes:
        flag_emissions_tco2e: Total FLAG emissions.
        total_entity_emissions_tco2e: Total entity emissions (all scopes).
        flag_pct_of_total: FLAG as percentage of total emissions.
        trigger_threshold_pct: The 20% threshold.
        status: Whether FLAG target is required.
        margin_to_threshold_pct: Distance to threshold (positive = above).
        is_borderline: Whether in the 15-20% borderline zone.
        message: Human-readable assessment.
    """
    flag_emissions_tco2e: Decimal = Field(default=Decimal("0"))
    total_entity_emissions_tco2e: Decimal = Field(default=Decimal("0"))
    flag_pct_of_total: Decimal = Field(default=Decimal("0"))
    trigger_threshold_pct: Decimal = Field(
        default=Decimal("20.0")
    )
    status: str = Field(default=FLAGTriggerStatus.UNDETERMINED.value)
    margin_to_threshold_pct: Decimal = Field(default=Decimal("0"))
    is_borderline: bool = Field(default=False)
    message: str = Field(default="")

class FLAGPathwayMilestone(BaseModel):
    """Annual milestone on the FLAG reduction pathway.

    Attributes:
        year: Calendar year.
        target_emissions_tco2e: Target FLAG emissions at this year.
        reduction_from_base_pct: Cumulative reduction from base (%).
        annual_reduction_rate_pct: Implied annual rate (%).
        on_track_threshold_tco2e: Max emissions to be on-track.
        cumulative_budget_tco2e: Cumulative carbon budget consumed.
    """
    year: int = Field(default=0)
    target_emissions_tco2e: Decimal = Field(default=Decimal("0"))
    reduction_from_base_pct: Decimal = Field(default=Decimal("0"))
    annual_reduction_rate_pct: Decimal = Field(default=Decimal("0"))
    on_track_threshold_tco2e: Decimal = Field(default=Decimal("0"))
    cumulative_budget_tco2e: Decimal = Field(default=Decimal("0"))

class FLAGTargetDefinition(BaseModel):
    """Complete FLAG target definition.

    Attributes:
        target_id: Unique target identifier.
        target_type: Absolute or intensity.
        entity_name: Reporting entity.
        base_year: Target base year.
        target_year: Target achievement year.
        base_emissions_tco2e: Base year FLAG emissions.
        target_emissions_tco2e: Target year FLAG emissions.
        reduction_pct: Total reduction percentage.
        annual_reduction_rate_pct: Annual linear reduction rate (%).
        commodities_covered: Number of commodities in target boundary.
        coverage_pct: Percentage of FLAG emissions covered.
        meets_coverage_requirement: Whether coverage >= 95%.
        milestones: Annual pathway milestones.
        is_near_term: Whether this is a near-term target.
        is_long_term: Whether this is a long-term target.
    """
    target_id: str = Field(default_factory=_new_uuid)
    target_type: str = Field(default=FLAGTargetType.ABSOLUTE.value)
    entity_name: str = Field(default="")
    base_year: int = Field(default=0)
    target_year: int = Field(default=0)
    base_emissions_tco2e: Decimal = Field(default=Decimal("0"))
    target_emissions_tco2e: Decimal = Field(default=Decimal("0"))
    reduction_pct: Decimal = Field(default=Decimal("0"))
    annual_reduction_rate_pct: Decimal = Field(default=Decimal("0"))
    commodities_covered: int = Field(default=0)
    coverage_pct: Decimal = Field(default=Decimal("0"))
    meets_coverage_requirement: bool = Field(default=False)
    milestones: List[FLAGPathwayMilestone] = Field(default_factory=list)
    is_near_term: bool = Field(default=True)
    is_long_term: bool = Field(default=False)

class DeforestationValidation(BaseModel):
    """Validation result for no-deforestation commitment.

    Attributes:
        has_commitment: Whether commitment exists.
        covers_all_commodities: Whether all commodities are covered.
        uncovered_commodities: List of commodities not covered.
        meets_deadline: Whether target date meets 2025 deadline.
        includes_no_conversion: Whether natural ecosystem conversion included.
        is_public: Whether commitment is publicly disclosed.
        aligned_with_afi: Whether aligned with AFi.
        has_verification: Whether verification mechanism exists.
        overall_status: Overall commitment status.
        gaps: Identified gaps in commitment.
        recommendations: Specific recommendations.
        score_pct: Commitment compliance score (%).
    """
    has_commitment: bool = Field(default=False)
    covers_all_commodities: bool = Field(default=False)
    uncovered_commodities: List[str] = Field(default_factory=list)
    meets_deadline: bool = Field(default=False)
    includes_no_conversion: bool = Field(default=False)
    is_public: bool = Field(default=False)
    aligned_with_afi: bool = Field(default=False)
    has_verification: bool = Field(default=False)
    overall_status: str = Field(
        default=DeforestationCommitmentStatus.NOT_COMMITTED.value
    )
    gaps: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    score_pct: Decimal = Field(default=Decimal("0"))

class LandUseChangeQuantification(BaseModel):
    """Quantification of land use change emissions.

    Attributes:
        total_luc_emissions_tco2e: Total LUC emissions.
        by_type: LUC emissions broken down by conversion type.
        by_commodity: LUC emissions attributed to each commodity.
        total_area_hectares: Total area of land conversion.
        pct_of_flag_total: LUC as percentage of total FLAG emissions.
        data_sources: Data sources used for quantification.
        confidence: Assessment confidence level.
    """
    total_luc_emissions_tco2e: Decimal = Field(default=Decimal("0"))
    by_type: Dict[str, Decimal] = Field(default_factory=dict)
    by_commodity: Dict[str, Decimal] = Field(default_factory=dict)
    total_area_hectares: Decimal = Field(default=Decimal("0"))
    pct_of_flag_total: Decimal = Field(default=Decimal("0"))
    data_sources: List[str] = Field(default_factory=list)
    confidence: str = Field(default=AssessmentConfidence.MEDIUM.value)

class FLAGProgressTracking(BaseModel):
    """Progress tracking for FLAG targets.

    Attributes:
        current_year: Current reporting year.
        current_emissions_tco2e: Current year FLAG emissions.
        pathway_target_tco2e: Pathway target for current year.
        deviation_tco2e: Deviation from pathway (positive = above).
        deviation_pct: Deviation as percentage of pathway target.
        status: Progress status (on_track / deviation / critical).
        cumulative_reduction_pct: Cumulative reduction from base year.
        required_reduction_pct: Required cumulative reduction.
        gap_pct: Gap between required and actual (percentage points).
        years_remaining: Years until target year.
        required_annual_rate_pct: Required rate from now to target.
        trajectory_target_year_tco2e: Projected emissions at target year.
        on_track_for_target: Whether trajectory meets target.
    """
    current_year: int = Field(default=0)
    current_emissions_tco2e: Decimal = Field(default=Decimal("0"))
    pathway_target_tco2e: Decimal = Field(default=Decimal("0"))
    deviation_tco2e: Decimal = Field(default=Decimal("0"))
    deviation_pct: Decimal = Field(default=Decimal("0"))
    status: str = Field(default=PathwayStatus.NOT_STARTED.value)
    cumulative_reduction_pct: Decimal = Field(default=Decimal("0"))
    required_reduction_pct: Decimal = Field(default=Decimal("0"))
    gap_pct: Decimal = Field(default=Decimal("0"))
    years_remaining: int = Field(default=0)
    required_annual_rate_pct: Decimal = Field(default=Decimal("0"))
    trajectory_target_year_tco2e: Decimal = Field(default=Decimal("0"))
    on_track_for_target: bool = Field(default=False)

class FLAGAssessmentResult(BaseModel):
    """Complete FLAG assessment result.

    Attributes:
        result_id: Unique result identifier.
        engine_version: Engine version string.
        calculated_at: Timestamp of calculation.
        entity_name: Reporting entity name.
        trigger_assessment: FLAG 20% trigger evaluation.
        commodity_assessments: Per-commodity assessments.
        flag_targets: Defined FLAG targets (near-term and/or long-term).
        deforestation_validation: No-deforestation commitment validation.
        luc_quantification: Land use change emissions quantification.
        progress_tracking: Progress tracking (if current data available).
        total_flag_emissions_tco2e: Total FLAG emissions (base year).
        total_entity_emissions_tco2e: Total entity emissions.
        commodity_count: Number of commodities assessed.
        commodities_with_data: Number with emission data.
        removals_tco2e: Carbon removals from land sector.
        net_flag_emissions_tco2e: FLAG emissions net of removals.
        warnings: Non-blocking warnings.
        errors: Blocking errors.
        recommendations: Strategic recommendations.
        processing_time_ms: Processing duration (ms).
        provenance_hash: SHA-256 audit hash.
    """
    result_id: str = Field(default_factory=_new_uuid)
    engine_version: str = Field(default=_MODULE_VERSION)
    calculated_at: datetime = Field(default_factory=utcnow)
    entity_name: str = Field(default="")
    trigger_assessment: Optional[FLAGTriggerAssessment] = Field(None)
    commodity_assessments: List[CommodityAssessment] = Field(
        default_factory=list
    )
    flag_targets: List[FLAGTargetDefinition] = Field(default_factory=list)
    deforestation_validation: Optional[DeforestationValidation] = Field(None)
    luc_quantification: Optional[LandUseChangeQuantification] = Field(None)
    progress_tracking: Optional[FLAGProgressTracking] = Field(None)
    total_flag_emissions_tco2e: Decimal = Field(default=Decimal("0"))
    total_entity_emissions_tco2e: Decimal = Field(default=Decimal("0"))
    commodity_count: int = Field(default=0)
    commodities_with_data: int = Field(default=0)
    removals_tco2e: Decimal = Field(default=Decimal("0"))
    net_flag_emissions_tco2e: Decimal = Field(default=Decimal("0"))
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    processing_time_ms: float = Field(default=0.0)
    provenance_hash: str = Field(default="")

# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class FLAGAssessmentEngine:
    """SBTi FLAG Assessment Engine.

    Evaluates FLAG (Forest, Land and Agriculture) emissions across
    11 commodity categories, determines whether a separate FLAG target
    is required (20% trigger threshold), calculates commodity-specific
    emission allocations and FLAG reduction pathways at 3.03%/yr
    linear reduction, and validates no-deforestation commitments.

    All calculations use deterministic Decimal arithmetic.  No LLM
    involvement in any calculation path.  SHA-256 provenance hash on
    every result.

    Usage::

        engine = FLAGAssessmentEngine()
        result = engine.assess(input_data)
        print(f"FLAG trigger: {result.trigger_assessment.status}")
        for ca in result.commodity_assessments:
            print(f"  {ca.commodity}: {ca.total_emissions_tco2e} tCO2e")
    """

    engine_version: str = _MODULE_VERSION

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialise FLAGAssessmentEngine.

        Args:
            config: Optional configuration overrides.  Supported keys:
                - flag_rate_override: Override FLAG annual rate (Decimal).
                - trigger_threshold_override: Override 20% trigger (Decimal).
                - include_removals_default: Default for removals inclusion.
        """
        self.config = config or {}
        self._flag_rate = _decimal(
            self.config.get("flag_rate_override", FLAG_ANNUAL_RATE)
        )
        self._trigger_threshold = _decimal(
            self.config.get("trigger_threshold_override", FLAG_TRIGGER_THRESHOLD)
        )
        logger.info(
            "FLAGAssessmentEngine v%s initialised (rate=%.4f, trigger=%.2f)",
            self.engine_version, float(self._flag_rate),
            float(self._trigger_threshold),
        )

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #

    def assess(self, data: FLAGAssessmentInput) -> FLAGAssessmentResult:
        """Run complete FLAG assessment pipeline.

        Orchestrates the full FLAG assessment: aggregates commodity
        emissions, evaluates the 20% trigger, builds per-commodity
        assessments, defines FLAG targets, validates deforestation
        commitments, quantifies LUC emissions, and optionally tracks
        progress against existing targets.

        Args:
            data: Validated FLAG assessment input.

        Returns:
            FLAGAssessmentResult with trigger, targets, and validation.
        """
        t0 = time.perf_counter()
        logger.info(
            "FLAG assessment: entity=%s, base=%d, commodities=%d",
            data.entity_name, data.base_year, len(data.commodities),
        )

        warnings: List[str] = []
        errors: List[str] = []
        recommendations: List[str] = []

        # Step 1: Aggregate FLAG emissions
        total_flag = self._aggregate_flag_emissions(data)

        # Step 2: Compute total entity emissions
        total_entity = (
            data.total_scope1_tco2e
            + data.total_scope2_tco2e
            + data.total_scope3_tco2e
            + total_flag
        )

        # Step 3: Evaluate 20% trigger
        trigger = self.evaluate_trigger(total_flag, total_entity)

        if trigger.status == FLAGTriggerStatus.REQUIRED.value:
            warnings.append(
                f"FLAG emissions ({_round_val(trigger.flag_pct_of_total, 1)}% "
                f"of total) exceed 20% trigger. Separate FLAG target required "
                f"per SBTi FLAG Guidance V1.1 Section 3."
            )
        elif trigger.status == FLAGTriggerStatus.BORDERLINE.value:
            warnings.append(
                f"FLAG emissions ({_round_val(trigger.flag_pct_of_total, 1)}% "
                f"of total) are in the borderline zone (15-20%). "
                f"Review recommended; FLAG target may be required."
            )

        # Step 4: Assess each commodity
        commodity_assessments = self._assess_commodities(
            data, total_flag, total_entity,
        )

        # Step 5: Quantify LUC emissions
        luc = self._quantify_luc(data, total_flag)

        # Step 6: Validate deforestation commitment
        deforestation = self.validate_deforestation_commitment(
            data.deforestation_commitment,
            [c.commodity for c in data.commodities],
        )

        if deforestation.overall_status != DeforestationCommitmentStatus.COMMITTED.value:
            recommendations.extend(deforestation.recommendations)
            if not deforestation.has_commitment:
                warnings.append(
                    "No no-deforestation commitment in place. "
                    "SBTi FLAG Guidance V1.1 Section 6 requires commitment "
                    "to zero deforestation by 2025."
                )

        # Step 7: Build FLAG targets
        flag_targets: List[FLAGTargetDefinition] = []
        if trigger.status in (
            FLAGTriggerStatus.REQUIRED.value,
            FLAGTriggerStatus.BORDERLINE.value,
        ) or len(data.commodities) > 0:
            # Near-term target
            nt_target = self._build_near_term_target(data, total_flag)
            flag_targets.append(nt_target)

            # Long-term target
            lt_target = self._build_long_term_target(data, total_flag)
            flag_targets.append(lt_target)

        # Step 8: Progress tracking (if current data available)
        progress: Optional[FLAGProgressTracking] = None
        if (
            data.current_flag_emissions_tco2e > Decimal("0")
            and flag_targets
        ):
            progress = self._track_progress(
                data, total_flag, flag_targets[0],
            )

        # Step 9: Generate recommendations
        recommendations.extend(
            self._generate_recommendations(
                data, trigger, commodity_assessments, deforestation,
            )
        )

        # Step 10: Validate inputs
        if not data.commodities:
            errors.append(
                "No commodity data provided. At least one FLAG commodity "
                "must be specified for assessment."
            )
        if total_flag <= Decimal("0") and not data.commodities:
            errors.append(
                "Total FLAG emissions are zero and no commodities provided."
            )

        # Removals
        removals = data.removals_tco2e if data.include_removals else Decimal("0")
        net_flag = max(Decimal("0"), total_flag - removals)

        commodities_with_data = sum(
            1 for ca in commodity_assessments
            if ca.total_emissions_tco2e > Decimal("0")
        )

        elapsed_ms = _round3((time.perf_counter() - t0) * 1000.0)

        result = FLAGAssessmentResult(
            entity_name=data.entity_name,
            trigger_assessment=trigger,
            commodity_assessments=commodity_assessments,
            flag_targets=flag_targets,
            deforestation_validation=deforestation,
            luc_quantification=luc,
            progress_tracking=progress,
            total_flag_emissions_tco2e=_round_val(total_flag),
            total_entity_emissions_tco2e=_round_val(total_entity),
            commodity_count=len(data.commodities),
            commodities_with_data=commodities_with_data,
            removals_tco2e=_round_val(removals),
            net_flag_emissions_tco2e=_round_val(net_flag),
            warnings=warnings,
            errors=errors,
            recommendations=recommendations,
            processing_time_ms=elapsed_ms,
        )
        result.provenance_hash = _compute_hash(result)

        logger.info(
            "FLAG assessment complete: trigger=%s, commodities=%d, "
            "targets=%d, flag_total=%.2f, hash=%s",
            trigger.status, len(commodity_assessments),
            len(flag_targets), float(total_flag),
            result.provenance_hash[:16],
        )
        return result

    def evaluate_trigger(
        self,
        flag_emissions_tco2e: Decimal,
        total_emissions_tco2e: Decimal,
    ) -> FLAGTriggerAssessment:
        """Evaluate the 20% FLAG trigger threshold.

        Determines whether FLAG emissions constitute >= 20% of total
        entity emissions, requiring a separate FLAG target.

        Args:
            flag_emissions_tco2e: Total FLAG emissions.
            total_emissions_tco2e: Total entity emissions (all scopes + FLAG).

        Returns:
            FLAGTriggerAssessment with status and details.
        """
        if total_emissions_tco2e <= Decimal("0"):
            return FLAGTriggerAssessment(
                flag_emissions_tco2e=flag_emissions_tco2e,
                total_entity_emissions_tco2e=total_emissions_tco2e,
                status=FLAGTriggerStatus.UNDETERMINED.value,
                message="Total emissions are zero; cannot assess FLAG trigger.",
            )

        flag_fraction = _safe_divide(
            flag_emissions_tco2e, total_emissions_tco2e,
        )
        flag_pct = _round_val(flag_fraction * Decimal("100"), 2)
        threshold_pct = _round_val(
            self._trigger_threshold * Decimal("100"), 2
        )
        margin = flag_pct - threshold_pct

        # Determine status
        if flag_fraction >= self._trigger_threshold:
            status = FLAGTriggerStatus.REQUIRED.value
            message = (
                f"FLAG emissions are {flag_pct}% of total, exceeding the "
                f"{threshold_pct}% trigger threshold. A separate FLAG "
                f"target is required per SBTi FLAG Guidance V1.1."
            )
        elif flag_fraction >= FLAG_BORDERLINE_LOWER:
            status = FLAGTriggerStatus.BORDERLINE.value
            message = (
                f"FLAG emissions are {flag_pct}% of total, in the "
                f"borderline zone ({_round_val(FLAG_BORDERLINE_LOWER * Decimal('100'), 0)}-"
                f"{threshold_pct}%). Detailed review recommended."
            )
        else:
            status = FLAGTriggerStatus.NOT_REQUIRED.value
            message = (
                f"FLAG emissions are {flag_pct}% of total, below the "
                f"{threshold_pct}% trigger threshold. Separate FLAG "
                f"target is not required."
            )

        return FLAGTriggerAssessment(
            flag_emissions_tco2e=_round_val(flag_emissions_tco2e),
            total_entity_emissions_tco2e=_round_val(total_emissions_tco2e),
            flag_pct_of_total=flag_pct,
            trigger_threshold_pct=threshold_pct,
            status=status,
            margin_to_threshold_pct=_round_val(margin, 2),
            is_borderline=(
                status == FLAGTriggerStatus.BORDERLINE.value
            ),
            message=message,
        )

    def calculate_flag_pathway(
        self,
        base_emissions: Decimal,
        base_year: int,
        target_year: int,
    ) -> List[FLAGPathwayMilestone]:
        """Calculate annual FLAG pathway milestones.

        Uses the 3.03%/yr linear reduction rate from SBTi FLAG
        Guidance V1.1.  The pathway formula is:

            E(t) = E(base) * max(0, 1 - 0.0303 * (t - base_year))

        Args:
            base_emissions: Base year FLAG emissions (tCO2e).
            base_year: Base calendar year.
            target_year: Target calendar year.

        Returns:
            List of FLAGPathwayMilestone from base to target year.
        """
        milestones: List[FLAGPathwayMilestone] = []
        cumulative_budget = Decimal("0")
        rate_pct = _round_val(self._flag_rate * Decimal("100"), 2)

        for year in range(base_year, target_year + 1):
            elapsed = year - base_year
            target_em = self._project_flag_emissions(
                base_emissions, elapsed,
            )
            reduction_pct = _safe_pct(
                base_emissions - target_em, base_emissions,
            )
            cumulative_budget += target_em

            milestones.append(FLAGPathwayMilestone(
                year=year,
                target_emissions_tco2e=_round_val(target_em),
                reduction_from_base_pct=_round_val(reduction_pct, 2),
                annual_reduction_rate_pct=rate_pct,
                on_track_threshold_tco2e=_round_val(
                    target_em * Decimal("1.05")
                ),
                cumulative_budget_tco2e=_round_val(cumulative_budget),
            ))

        return milestones

    def validate_deforestation_commitment(
        self,
        commitment: DeforestationCommitment,
        assessed_commodities: List[str],
    ) -> DeforestationValidation:
        """Validate no-deforestation commitment against SBTi requirements.

        SBTi FLAG Guidance V1.1 Section 6 requires:
        - Zero deforestation commitment by 2025
        - Covers all FLAG commodities in value chain
        - Includes no-conversion of natural ecosystems
        - Publicly disclosed with verification mechanism

        Args:
            commitment: Deforestation commitment details.
            assessed_commodities: Commodities being assessed.

        Returns:
            DeforestationValidation with gaps and recommendations.
        """
        gaps: List[str] = []
        recommendations: List[str] = []
        score_items: List[bool] = []

        # Check 1: Commitment exists
        has_commitment = commitment.has_commitment
        score_items.append(has_commitment)
        if not has_commitment:
            gaps.append(
                "No no-deforestation commitment in place."
            )
            recommendations.append(
                "Establish a corporate no-deforestation and no-conversion "
                "commitment covering all FLAG commodities in line with "
                "SBTi FLAG Guidance V1.1 Section 6."
            )

        # Check 2: Covers all commodities
        covered = set(
            c.lower().strip() for c in commitment.covered_commodities
        )
        assessed = set(c.lower().strip() for c in assessed_commodities)
        uncovered = sorted(assessed - covered)
        covers_all = commitment.covers_all_commodities or len(uncovered) == 0
        score_items.append(covers_all)
        if not covers_all and uncovered:
            gaps.append(
                f"Commitment does not cover: {', '.join(uncovered)}."
            )
            recommendations.append(
                f"Extend no-deforestation commitment to cover: "
                f"{', '.join(uncovered)}."
            )

        # Check 3: Meets deadline (2025)
        meets_deadline = False
        if commitment.target_date:
            try:
                target_year = int(commitment.target_date[:4])
                meets_deadline = target_year <= NO_DEFORESTATION_DEADLINE
            except (ValueError, IndexError):
                pass
        score_items.append(meets_deadline)
        if not meets_deadline and has_commitment:
            gaps.append(
                f"Target date does not meet the {NO_DEFORESTATION_DEADLINE} "
                f"deadline required by SBTi FLAG Guidance V1.1."
            )
            recommendations.append(
                f"Advance no-deforestation target date to "
                f"{NO_DEFORESTATION_DEADLINE} or earlier."
            )

        # Check 4: Includes no-conversion
        score_items.append(commitment.includes_no_conversion)
        if not commitment.includes_no_conversion and has_commitment:
            gaps.append(
                "Commitment does not include zero conversion of "
                "natural ecosystems."
            )
            recommendations.append(
                "Extend commitment to include zero conversion of all "
                "natural ecosystems (forests, peatlands, wetlands)."
            )

        # Check 5: Public disclosure
        score_items.append(commitment.is_public)
        if not commitment.is_public and has_commitment:
            gaps.append("Commitment is not publicly disclosed.")
            recommendations.append(
                "Publicly disclose no-deforestation commitment in "
                "sustainability report and corporate website."
            )

        # Check 6: Verification mechanism
        has_verification = bool(commitment.verification_mechanism)
        score_items.append(has_verification)
        if not has_verification and has_commitment:
            gaps.append("No verification mechanism defined.")
            recommendations.append(
                "Establish verification mechanism (satellite monitoring, "
                "third-party audit, or certification scheme)."
            )

        # Check 7: AFi alignment
        score_items.append(commitment.aligned_with_afi)
        if not commitment.aligned_with_afi and has_commitment:
            gaps.append(
                "Commitment is not aligned with the Accountability "
                "Framework initiative (AFi)."
            )
            recommendations.append(
                "Align commitment with AFi Core Principles for "
                "no-deforestation supply chains."
            )

        # Determine overall status
        if has_commitment and covers_all and meets_deadline:
            if commitment.includes_no_conversion:
                overall_status = DeforestationCommitmentStatus.COMMITTED.value
            else:
                overall_status = DeforestationCommitmentStatus.PARTIAL.value
        elif has_commitment:
            overall_status = DeforestationCommitmentStatus.PARTIAL.value
        else:
            overall_status = DeforestationCommitmentStatus.NOT_COMMITTED.value

        # Check for expired deadline
        current_year = utcnow().year
        if (
            has_commitment
            and commitment.target_date
            and meets_deadline
            and current_year > NO_DEFORESTATION_DEADLINE
        ):
            # Deadline has passed -- check if verified
            if not has_verification:
                overall_status = DeforestationCommitmentStatus.EXPIRED.value
                gaps.append(
                    "No-deforestation deadline has passed without "
                    "verification of achievement."
                )

        # Calculate score
        total_checks = len(score_items)
        passed_checks = sum(1 for s in score_items if s)
        score_pct = _safe_pct(
            _decimal(passed_checks), _decimal(total_checks),
        )

        return DeforestationValidation(
            has_commitment=has_commitment,
            covers_all_commodities=covers_all,
            uncovered_commodities=uncovered,
            meets_deadline=meets_deadline,
            includes_no_conversion=commitment.includes_no_conversion,
            is_public=commitment.is_public,
            aligned_with_afi=commitment.aligned_with_afi,
            has_verification=has_verification,
            overall_status=overall_status,
            gaps=gaps,
            recommendations=recommendations,
            score_pct=_round_val(score_pct, 1),
        )

    def calculate_commodity_emissions(
        self,
        commodity: str,
        production_volume: Decimal,
        breakdown: Optional[CommodityEmissionBreakdown] = None,
    ) -> Tuple[Decimal, CommodityEmissionBreakdown]:
        """Calculate emissions for a single commodity.

        If a breakdown is provided, uses the provided values.
        Otherwise, applies default emission factors from
        COMMODITY_DEFAULT_EMISSION_FACTORS.

        Args:
            commodity: FLAG commodity name.
            production_volume: Annual production volume (tonnes).
            breakdown: Optional pre-calculated emission breakdown.

        Returns:
            Tuple of (total_emissions, emission_breakdown).
        """
        commodity_key = commodity.lower().strip()

        if breakdown is not None and breakdown.total_tco2e > Decimal("0"):
            return breakdown.total_tco2e, breakdown

        # Apply default factors
        factors = COMMODITY_DEFAULT_EMISSION_FACTORS.get(
            commodity_key, {}
        )
        if not factors or production_volume <= Decimal("0"):
            empty_breakdown = CommodityEmissionBreakdown()
            return Decimal("0"), empty_breakdown

        luc = production_volume * factors.get(
            EmissionCategory.LAND_USE_CHANGE.value, Decimal("0")
        )
        ag = production_volume * factors.get(
            EmissionCategory.AGRICULTURAL_PROCESS.value, Decimal("0")
        )
        inp = production_volume * factors.get(
            EmissionCategory.INPUT_PRODUCTION.value, Decimal("0")
        )
        energy = production_volume * factors.get(
            EmissionCategory.ON_FARM_ENERGY.value, Decimal("0")
        )
        post = production_volume * factors.get(
            EmissionCategory.POST_HARVEST.value, Decimal("0")
        )

        calc_breakdown = CommodityEmissionBreakdown(
            land_use_change_tco2e=_round_val(luc),
            agricultural_process_tco2e=_round_val(ag),
            input_production_tco2e=_round_val(inp),
            on_farm_energy_tco2e=_round_val(energy),
            post_harvest_tco2e=_round_val(post),
        )

        return _round_val(calc_breakdown.total_tco2e), calc_breakdown

    def calculate_commodity_intensity(
        self,
        total_emissions: Decimal,
        production_volume: Decimal,
        commodity: str,
    ) -> Tuple[Decimal, str]:
        """Calculate emission intensity for a commodity.

        Args:
            total_emissions: Total commodity emissions (tCO2e).
            production_volume: Production volume.
            commodity: Commodity name.

        Returns:
            Tuple of (intensity_value, intensity_unit).
        """
        intensity_unit = COMMODITY_INTENSITY_UNITS.get(
            commodity.lower().strip(), "tCO2e/t"
        )
        intensity = _safe_divide(total_emissions, production_volume)
        return _round_val(intensity, 4), intensity_unit

    def get_commodity_risk(self, commodity: str) -> str:
        """Get deforestation risk rating for a commodity.

        Args:
            commodity: FLAG commodity name.

        Returns:
            Risk rating string (very_high/high/medium/low).
        """
        return COMMODITY_DEFORESTATION_RISK.get(
            commodity.lower().strip(), "unknown"
        )

    def project_emissions(
        self,
        base_emissions: Decimal,
        years_elapsed: int,
    ) -> Decimal:
        """Project FLAG emissions forward using 3.03%/yr linear reduction.

        This is a public wrapper around the internal projection method.

        Args:
            base_emissions: Base year FLAG emissions (tCO2e).
            years_elapsed: Number of years from base.

        Returns:
            Projected emissions (non-negative Decimal).
        """
        return self._project_flag_emissions(base_emissions, years_elapsed)

    # ------------------------------------------------------------------ #
    # Target Builders                                                     #
    # ------------------------------------------------------------------ #

    def _build_near_term_target(
        self,
        data: FLAGAssessmentInput,
        total_flag: Decimal,
    ) -> FLAGTargetDefinition:
        """Build near-term FLAG target (5-10 year horizon).

        Uses 3.03%/yr linear reduction per SBTi FLAG Guidance V1.1.

        Args:
            data: Assessment input.
            total_flag: Total FLAG emissions.

        Returns:
            FLAGTargetDefinition for near-term.
        """
        base_year = data.base_year
        target_year = data.target_year_near_term
        elapsed = target_year - base_year

        # Validate timeframe
        if elapsed < FLAG_NEAR_TERM_MIN_YEARS:
            target_year = base_year + FLAG_NEAR_TERM_MIN_YEARS
            elapsed = FLAG_NEAR_TERM_MIN_YEARS
        elif elapsed > FLAG_NEAR_TERM_MAX_YEARS:
            target_year = base_year + FLAG_NEAR_TERM_MAX_YEARS
            elapsed = FLAG_NEAR_TERM_MAX_YEARS

        target_em = self._project_flag_emissions(total_flag, elapsed)
        reduction_pct = _safe_pct(total_flag - target_em, total_flag)

        milestones = self.calculate_flag_pathway(
            total_flag, base_year, target_year,
        )

        coverage = self._calculate_coverage(data, total_flag)

        return FLAGTargetDefinition(
            target_type=data.preferred_target_type,
            entity_name=data.entity_name,
            base_year=base_year,
            target_year=target_year,
            base_emissions_tco2e=_round_val(total_flag),
            target_emissions_tco2e=_round_val(target_em),
            reduction_pct=_round_val(reduction_pct, 2),
            annual_reduction_rate_pct=_round_val(
                self._flag_rate * Decimal("100"), 2
            ),
            commodities_covered=len(data.commodities),
            coverage_pct=_round_val(coverage, 2),
            meets_coverage_requirement=(
                coverage >= FLAG_COVERAGE_MIN * Decimal("100")
            ),
            milestones=milestones,
            is_near_term=True,
            is_long_term=False,
        )

    def _build_long_term_target(
        self,
        data: FLAGAssessmentInput,
        total_flag: Decimal,
    ) -> FLAGTargetDefinition:
        """Build long-term FLAG target (by 2050).

        Long-term FLAG targets require >= 72% reduction from base year
        by 2050, per SBTi FLAG Guidance V1.1 and SBTi Corporate
        Net-Zero Standard V1.3.

        Args:
            data: Assessment input.
            total_flag: Total FLAG emissions.

        Returns:
            FLAGTargetDefinition for long-term.
        """
        base_year = data.base_year
        target_year = min(data.target_year_long_term, FLAG_LONG_TERM_MAX_YEAR)
        elapsed = target_year - base_year

        # Calculate FLAG pathway emissions at target year
        pathway_target = self._project_flag_emissions(total_flag, elapsed)

        # Ensure at least 72% reduction for long-term
        max_residual = total_flag * (
            Decimal("1") - FLAG_LONG_TERM_MIN_REDUCTION_PCT / Decimal("100")
        )
        target_em = min(pathway_target, max_residual)
        reduction_pct = _safe_pct(total_flag - target_em, total_flag)

        # Calculate effective annual rate for the long-term target
        effective_rate = Decimal("0")
        if elapsed > 0 and total_flag > Decimal("0"):
            effective_rate = _safe_divide(
                total_flag - target_em,
                total_flag * _decimal(elapsed),
            )

        milestones = self.calculate_flag_pathway(
            total_flag, base_year, target_year,
        )

        coverage = self._calculate_coverage(data, total_flag)

        return FLAGTargetDefinition(
            target_type=data.preferred_target_type,
            entity_name=data.entity_name,
            base_year=base_year,
            target_year=target_year,
            base_emissions_tco2e=_round_val(total_flag),
            target_emissions_tco2e=_round_val(target_em),
            reduction_pct=_round_val(reduction_pct, 2),
            annual_reduction_rate_pct=_round_val(
                effective_rate * Decimal("100"), 2
            ),
            commodities_covered=len(data.commodities),
            coverage_pct=_round_val(coverage, 2),
            meets_coverage_requirement=(
                coverage >= FLAG_COVERAGE_MIN * Decimal("100")
            ),
            milestones=milestones,
            is_near_term=False,
            is_long_term=True,
        )

    # ------------------------------------------------------------------ #
    # Commodity Assessment                                                #
    # ------------------------------------------------------------------ #

    def _assess_commodities(
        self,
        data: FLAGAssessmentInput,
        total_flag: Decimal,
        total_entity: Decimal,
    ) -> List[CommodityAssessment]:
        """Assess each commodity's emissions and pathway.

        Args:
            data: Assessment input.
            total_flag: Total FLAG emissions.
            total_entity: Total entity emissions.

        Returns:
            List of CommodityAssessment objects.
        """
        assessments: List[CommodityAssessment] = []

        for ci in data.commodities:
            # Calculate or use provided emissions
            if ci.total_emissions_tco2e > Decimal("0"):
                total_em = ci.total_emissions_tco2e
                breakdown = ci.emission_breakdown
            elif ci.emission_breakdown and ci.emission_breakdown.total_tco2e > Decimal("0"):
                total_em = ci.emission_breakdown.total_tco2e
                breakdown = ci.emission_breakdown
            else:
                total_em, breakdown = self.calculate_commodity_emissions(
                    ci.commodity, ci.production_volume, ci.emission_breakdown,
                )

            # Percentages
            pct_flag = _safe_pct(total_em, total_flag)
            pct_entity = _safe_pct(total_em, total_entity)

            # Intensity
            intensity_val, intensity_unit = self.calculate_commodity_intensity(
                total_em, ci.production_volume, ci.commodity,
            )

            # Per-commodity pathway
            elapsed = data.target_year_near_term - data.base_year
            if elapsed < FLAG_NEAR_TERM_MIN_YEARS:
                elapsed = FLAG_NEAR_TERM_MIN_YEARS
            target_em = self._project_flag_emissions(total_em, elapsed)
            reduction_pct = _safe_pct(total_em - target_em, total_em)

            # Generate per-commodity milestones
            target_year = data.base_year + elapsed
            commodity_milestones = []
            for ms in self.calculate_flag_pathway(
                total_em, data.base_year, target_year,
            ):
                commodity_milestones.append({
                    "year": ms.year,
                    "target_tco2e": str(ms.target_emissions_tco2e),
                    "reduction_pct": str(ms.reduction_from_base_pct),
                })

            # Determine confidence
            confidence = self._assess_data_confidence(ci)

            assessments.append(CommodityAssessment(
                commodity=ci.commodity,
                total_emissions_tco2e=_round_val(total_em),
                emission_breakdown=breakdown,
                pct_of_flag_total=_round_val(pct_flag, 2),
                pct_of_entity_total=_round_val(pct_entity, 2),
                intensity_value=intensity_val,
                intensity_unit=intensity_unit,
                base_year_emissions_tco2e=_round_val(total_em),
                target_year_emissions_tco2e=_round_val(target_em),
                required_reduction_pct=_round_val(reduction_pct, 2),
                deforestation_risk=self.get_commodity_risk(ci.commodity),
                has_certification=ci.has_certification,
                data_quality=ci.data_quality,
                traceability_pct=ci.traceability_pct,
                pathway_milestones=commodity_milestones,
                confidence=confidence,
            ))

        # Sort by emissions (highest first)
        assessments.sort(
            key=lambda a: a.total_emissions_tco2e, reverse=True,
        )

        return assessments

    def _assess_data_confidence(
        self, ci: CommodityInput,
    ) -> str:
        """Assess data quality confidence for a commodity.

        Args:
            ci: Commodity input data.

        Returns:
            Confidence level string.
        """
        score = Decimal("0")

        # Primary data available
        if ci.emission_breakdown and ci.emission_breakdown.total_tco2e > Decimal("0"):
            score += Decimal("30")
        elif ci.total_emissions_tco2e > Decimal("0"):
            score += Decimal("20")

        # Production volume provided
        if ci.production_volume > Decimal("0"):
            score += Decimal("20")

        # Traceability
        if ci.traceability_pct >= Decimal("80"):
            score += Decimal("20")
        elif ci.traceability_pct >= Decimal("50"):
            score += Decimal("10")

        # Certification
        if ci.has_certification:
            score += Decimal("15")

        # Sourcing regions specified
        if ci.sourcing_regions:
            score += Decimal("15")

        if score >= Decimal("80"):
            return AssessmentConfidence.HIGH.value
        if score >= Decimal("50"):
            return AssessmentConfidence.MEDIUM.value
        if score >= Decimal("25"):
            return AssessmentConfidence.LOW.value
        return AssessmentConfidence.VERY_LOW.value

    # ------------------------------------------------------------------ #
    # Land Use Change Quantification                                      #
    # ------------------------------------------------------------------ #

    def _quantify_luc(
        self,
        data: FLAGAssessmentInput,
        total_flag: Decimal,
    ) -> LandUseChangeQuantification:
        """Quantify land use change emissions.

        Uses IPCC AR6 emission factors per conversion type and
        attributing to commodities where linkage is specified.

        Args:
            data: Assessment input.
            total_flag: Total FLAG emissions.

        Returns:
            LandUseChangeQuantification with totals and breakdowns.
        """
        total_luc = Decimal("0")
        by_type: Dict[str, Decimal] = {}
        by_commodity: Dict[str, Decimal] = {}
        total_area = Decimal("0")
        data_sources: List[str] = []

        for entry in data.land_use_changes:
            # Calculate emissions for this entry
            if (
                entry.calculated_emissions_tco2e is not None
                and entry.calculated_emissions_tco2e > Decimal("0")
            ):
                em = entry.calculated_emissions_tco2e
            else:
                ef = entry.emission_factor_tco2e_per_ha
                if ef is None:
                    ef = LUC_EMISSION_FACTORS.get(
                        entry.luc_type, Decimal("200.0")
                    )
                em = entry.area_hectares * ef

            total_luc += em
            total_area += entry.area_hectares

            # Aggregate by type
            luc_type = entry.luc_type
            by_type[luc_type] = by_type.get(
                luc_type, Decimal("0")
            ) + em

            # Aggregate by commodity
            if entry.commodity:
                comm = entry.commodity.lower().strip()
                by_commodity[comm] = by_commodity.get(
                    comm, Decimal("0")
                ) + em

        # Also aggregate LUC from commodity breakdowns
        for ci in data.commodities:
            if ci.emission_breakdown:
                luc_em = ci.emission_breakdown.land_use_change_tco2e
                if luc_em > Decimal("0"):
                    comm = ci.commodity.lower().strip()
                    by_commodity[comm] = by_commodity.get(
                        comm, Decimal("0")
                    ) + luc_em
                    total_luc += luc_em

        pct_of_flag = _safe_pct(total_luc, total_flag)

        # Round all values
        by_type_rounded = {
            k: _round_val(v) for k, v in by_type.items()
        }
        by_commodity_rounded = {
            k: _round_val(v) for k, v in by_commodity.items()
        }

        if data.land_use_changes:
            data_sources.append("Land use change entries (direct input)")
        if any(
            ci.emission_breakdown and ci.emission_breakdown.land_use_change_tco2e > Decimal("0")
            for ci in data.commodities
        ):
            data_sources.append("Commodity emission breakdowns (LUC component)")

        if not data_sources:
            data_sources.append("No direct LUC data provided")

        # Determine confidence
        if data.land_use_changes and any(
            e.calculated_emissions_tco2e is not None
            for e in data.land_use_changes
        ):
            confidence = AssessmentConfidence.HIGH.value
        elif data.land_use_changes:
            confidence = AssessmentConfidence.MEDIUM.value
        else:
            confidence = AssessmentConfidence.LOW.value

        return LandUseChangeQuantification(
            total_luc_emissions_tco2e=_round_val(total_luc),
            by_type=by_type_rounded,
            by_commodity=by_commodity_rounded,
            total_area_hectares=_round_val(total_area, 2),
            pct_of_flag_total=_round_val(pct_of_flag, 2),
            data_sources=data_sources,
            confidence=confidence,
        )

    # ------------------------------------------------------------------ #
    # Progress Tracking                                                   #
    # ------------------------------------------------------------------ #

    def _track_progress(
        self,
        data: FLAGAssessmentInput,
        base_flag: Decimal,
        near_term_target: FLAGTargetDefinition,
    ) -> FLAGProgressTracking:
        """Track progress against FLAG pathway.

        Compares current FLAG emissions against the expected pathway
        position and projects trajectory to target year.

        Args:
            data: Assessment input with current year emissions.
            base_flag: Base year total FLAG emissions.
            near_term_target: Near-term FLAG target definition.

        Returns:
            FLAGProgressTracking with deviation and trajectory.
        """
        current_year = data.current_year
        current_em = data.current_flag_emissions_tco2e
        base_year = data.base_year
        target_year = near_term_target.target_year

        # Calculate pathway target for current year
        elapsed_from_base = current_year - base_year
        pathway_target = self._project_flag_emissions(
            base_flag, elapsed_from_base,
        )

        # Deviation
        deviation = current_em - pathway_target
        deviation_pct = _safe_pct(deviation, pathway_target)

        # Determine status
        abs_deviation_pct = abs(deviation_pct)
        if deviation <= Decimal("0"):
            status = PathwayStatus.ON_TRACK.value
        elif abs_deviation_pct <= Decimal("5"):
            status = PathwayStatus.MINOR_DEVIATION.value
        elif abs_deviation_pct <= Decimal("15"):
            status = PathwayStatus.MAJOR_DEVIATION.value
        else:
            status = PathwayStatus.CRITICAL.value

        # Cumulative reduction achieved
        cumulative_reduction = _safe_pct(
            base_flag - current_em, base_flag,
        )
        required_reduction = _safe_pct(
            base_flag - pathway_target, base_flag,
        )
        gap = required_reduction - cumulative_reduction

        # Years remaining
        years_remaining = max(0, target_year - current_year)

        # Required annual rate from now
        if years_remaining > 0 and current_em > Decimal("0"):
            remaining_reduction = (
                current_em - near_term_target.target_emissions_tco2e
            )
            required_annual = _safe_divide(
                remaining_reduction,
                current_em * _decimal(years_remaining),
            )
            required_annual_pct = required_annual * Decimal("100")
        else:
            required_annual_pct = Decimal("0")

        # Trajectory projection (linear from base to current)
        if elapsed_from_base > 0 and current_em > Decimal("0"):
            annual_actual_reduction = _safe_divide(
                base_flag - current_em,
                _decimal(elapsed_from_base),
            )
            total_elapsed_to_target = target_year - base_year
            trajectory_em = max(
                Decimal("0"),
                base_flag - annual_actual_reduction * _decimal(
                    total_elapsed_to_target
                ),
            )
        else:
            trajectory_em = current_em

        on_track = trajectory_em <= near_term_target.target_emissions_tco2e

        return FLAGProgressTracking(
            current_year=current_year,
            current_emissions_tco2e=_round_val(current_em),
            pathway_target_tco2e=_round_val(pathway_target),
            deviation_tco2e=_round_val(deviation),
            deviation_pct=_round_val(deviation_pct, 2),
            status=status,
            cumulative_reduction_pct=_round_val(cumulative_reduction, 2),
            required_reduction_pct=_round_val(required_reduction, 2),
            gap_pct=_round_val(gap, 2),
            years_remaining=years_remaining,
            required_annual_rate_pct=_round_val(required_annual_pct, 2),
            trajectory_target_year_tco2e=_round_val(trajectory_em),
            on_track_for_target=on_track,
        )

    # ------------------------------------------------------------------ #
    # Emission Projection                                                 #
    # ------------------------------------------------------------------ #

    def _project_flag_emissions(
        self,
        base_emissions: Decimal,
        elapsed_years: int,
    ) -> Decimal:
        """Project FLAG emissions using 3.03%/yr linear reduction.

        Formula: E(t) = E(base) * max(0, 1 - 0.0303 * elapsed)

        This is the linear reduction pathway specified in SBTi FLAG
        Guidance V1.1, Section 5, Table 5.1.

        Args:
            base_emissions: Base year emissions (tCO2e).
            elapsed_years: Years elapsed from base year.

        Returns:
            Projected emissions (non-negative Decimal).
        """
        if elapsed_years <= 0:
            return base_emissions

        factor = max(
            Decimal("0"),
            Decimal("1") - self._flag_rate * _decimal(elapsed_years),
        )
        return base_emissions * factor

    # ------------------------------------------------------------------ #
    # Emission Aggregation                                                #
    # ------------------------------------------------------------------ #

    def _aggregate_flag_emissions(
        self,
        data: FLAGAssessmentInput,
    ) -> Decimal:
        """Aggregate total FLAG emissions from all sources.

        Priority:
        1. If total_flag_tco2e is provided and > 0, use it.
        2. Otherwise, sum commodity-level emissions.
        3. Add any LUC events not already captured in commodities.

        Args:
            data: Assessment input.

        Returns:
            Total FLAG emissions (Decimal).
        """
        if data.total_flag_tco2e > Decimal("0"):
            return data.total_flag_tco2e

        total = Decimal("0")

        for ci in data.commodities:
            if ci.total_emissions_tco2e > Decimal("0"):
                total += ci.total_emissions_tco2e
            elif (
                ci.emission_breakdown
                and ci.emission_breakdown.total_tco2e > Decimal("0")
            ):
                total += ci.emission_breakdown.total_tco2e
            elif ci.production_volume > Decimal("0"):
                em, _ = self.calculate_commodity_emissions(
                    ci.commodity, ci.production_volume,
                )
                total += em

        # Add standalone LUC events (not already attributed to commodities)
        for entry in data.land_use_changes:
            if not entry.commodity:
                if (
                    entry.calculated_emissions_tco2e is not None
                    and entry.calculated_emissions_tco2e > Decimal("0")
                ):
                    total += entry.calculated_emissions_tco2e
                elif entry.area_hectares > Decimal("0"):
                    ef = entry.emission_factor_tco2e_per_ha
                    if ef is None:
                        ef = LUC_EMISSION_FACTORS.get(
                            entry.luc_type, Decimal("200.0")
                        )
                    total += entry.area_hectares * ef

        return total

    # ------------------------------------------------------------------ #
    # Coverage Calculation                                                #
    # ------------------------------------------------------------------ #

    def _calculate_coverage(
        self,
        data: FLAGAssessmentInput,
        total_flag: Decimal,
    ) -> Decimal:
        """Calculate FLAG target coverage percentage.

        Coverage is the percentage of total FLAG emissions covered
        by the commodities included in the target boundary.

        Args:
            data: Assessment input.
            total_flag: Total FLAG emissions.

        Returns:
            Coverage as percentage (0-100).
        """
        if total_flag <= Decimal("0"):
            return Decimal("100")

        covered = Decimal("0")
        for ci in data.commodities:
            if ci.total_emissions_tco2e > Decimal("0"):
                covered += ci.total_emissions_tco2e
            elif (
                ci.emission_breakdown
                and ci.emission_breakdown.total_tco2e > Decimal("0")
            ):
                covered += ci.emission_breakdown.total_tco2e
            elif ci.production_volume > Decimal("0"):
                em, _ = self.calculate_commodity_emissions(
                    ci.commodity, ci.production_volume,
                )
                covered += em

        return _safe_pct(covered, total_flag)

    # ------------------------------------------------------------------ #
    # Recommendations                                                     #
    # ------------------------------------------------------------------ #

    def _generate_recommendations(
        self,
        data: FLAGAssessmentInput,
        trigger: FLAGTriggerAssessment,
        commodities: List[CommodityAssessment],
        deforestation: DeforestationValidation,
    ) -> List[str]:
        """Generate strategic recommendations based on assessment.

        Args:
            data: Assessment input.
            trigger: Trigger assessment result.
            commodities: Per-commodity assessments.
            deforestation: Deforestation validation result.

        Returns:
            List of recommendation strings.
        """
        recs: List[str] = []

        # High-risk commodities without certification
        for ca in commodities:
            if (
                ca.deforestation_risk in ("very_high", "high")
                and not ca.has_certification
            ):
                recs.append(
                    f"Prioritise sustainability certification for "
                    f"{ca.commodity} (deforestation risk: {ca.deforestation_risk})."
                )

        # Low traceability commodities
        for ca in commodities:
            if ca.traceability_pct < Decimal("50"):
                recs.append(
                    f"Improve supply chain traceability for {ca.commodity} "
                    f"(currently {ca.traceability_pct}%). Target >= 80% "
                    f"for reliable FLAG reporting."
                )

        # Dominant commodity (>50% of FLAG)
        for ca in commodities:
            if ca.pct_of_flag_total > Decimal("50"):
                recs.append(
                    f"{ca.commodity.replace('_', ' ').title()} represents "
                    f"{ca.pct_of_flag_total}% of FLAG emissions. Prioritise "
                    f"reduction interventions for this commodity."
                )

        # Data quality improvements
        low_quality = [
            ca.commodity for ca in commodities
            if ca.confidence in (
                AssessmentConfidence.LOW.value,
                AssessmentConfidence.VERY_LOW.value,
            )
        ]
        if low_quality:
            recs.append(
                f"Improve data quality for: {', '.join(low_quality)}. "
                f"Transition from proxy/spend data to primary measured data."
            )

        # Carbon removals opportunity
        if not data.include_removals and data.removals_tco2e > Decimal("0"):
            recs.append(
                "Consider including land-sector carbon removals in "
                "FLAG assessment to demonstrate full land-sector profile."
            )

        # If borderline, recommend proactive target setting
        if trigger.status == FLAGTriggerStatus.BORDERLINE.value:
            recs.append(
                "FLAG emissions are in the borderline zone (15-20%). "
                "Consider proactively setting a FLAG target to demonstrate "
                "leadership and prepare for potential threshold exceedance."
            )

        # LUC-specific recommendations
        if any(
            ca.emission_breakdown
            and ca.emission_breakdown.land_use_change_tco2e > Decimal("0")
            for ca in commodities
            if ca.emission_breakdown
        ):
            recs.append(
                "Implement satellite-based monitoring for land use change "
                "in high-risk sourcing regions to improve LUC emission "
                "quantification accuracy."
            )

        return recs

    # ------------------------------------------------------------------ #
    # Utility Methods                                                     #
    # ------------------------------------------------------------------ #

    def get_supported_commodities(self) -> List[Dict[str, str]]:
        """Return list of supported FLAG commodities with metadata.

        Returns:
            List of dicts with commodity, risk, and intensity unit.
        """
        return [
            {
                "commodity": c.value,
                "deforestation_risk": COMMODITY_DEFORESTATION_RISK.get(
                    c.value, "unknown"
                ),
                "intensity_unit": COMMODITY_INTENSITY_UNITS.get(
                    c.value, "tCO2e/t"
                ),
            }
            for c in FLAGCommodity
        ]

    def get_emission_factors(
        self, commodity: str,
    ) -> Dict[str, str]:
        """Return default emission factors for a commodity.

        Args:
            commodity: FLAG commodity name.

        Returns:
            Dict mapping emission category to factor value.
        """
        factors = COMMODITY_DEFAULT_EMISSION_FACTORS.get(
            commodity.lower().strip(), {}
        )
        return {k: str(v) for k, v in factors.items()}

    def get_luc_factors(self) -> Dict[str, str]:
        """Return land use change emission factors.

        Returns:
            Dict mapping LUC type to tCO2e/ha factor.
        """
        return {k: str(v) for k, v in LUC_EMISSION_FACTORS.items()}

    def get_flag_rate(self) -> Dict[str, str]:
        """Return the FLAG annual reduction rate and source.

        Returns:
            Dict with rate, percentage, and source reference.
        """
        return {
            "annual_rate_fraction": str(self._flag_rate),
            "annual_rate_pct": str(
                _round_val(self._flag_rate * Decimal("100"), 2)
            ),
            "source": "SBTi FLAG Guidance V1.1, Section 5, Table 5.1",
            "pathway_type": "linear",
            "formula": "E(t) = E(base) * max(0, 1 - 0.0303 * (t - base_year))",
        }

    def get_trigger_threshold(self) -> Dict[str, str]:
        """Return the FLAG trigger threshold and source.

        Returns:
            Dict with threshold fraction, percentage, and source.
        """
        return {
            "threshold_fraction": str(self._trigger_threshold),
            "threshold_pct": str(
                _round_val(
                    self._trigger_threshold * Decimal("100"), 1
                )
            ),
            "borderline_lower_pct": str(
                _round_val(
                    FLAG_BORDERLINE_LOWER * Decimal("100"), 1
                )
            ),
            "source": "SBTi FLAG Guidance V1.1, Section 3",
        }

    def get_summary(
        self, result: FLAGAssessmentResult,
    ) -> Dict[str, Any]:
        """Generate concise summary from FLAG assessment result.

        Args:
            result: Complete FLAG assessment result.

        Returns:
            Dict with key metrics and provenance hash.
        """
        summary: Dict[str, Any] = {
            "entity_name": result.entity_name,
            "flag_trigger_status": (
                result.trigger_assessment.status
                if result.trigger_assessment else "unknown"
            ),
            "flag_pct_of_total": str(
                result.trigger_assessment.flag_pct_of_total
                if result.trigger_assessment else "0"
            ),
            "total_flag_emissions_tco2e": str(
                result.total_flag_emissions_tco2e
            ),
            "commodity_count": result.commodity_count,
            "commodities_with_data": result.commodities_with_data,
            "targets_defined": len(result.flag_targets),
            "deforestation_status": (
                result.deforestation_validation.overall_status
                if result.deforestation_validation else "unknown"
            ),
            "deforestation_score_pct": str(
                result.deforestation_validation.score_pct
                if result.deforestation_validation else "0"
            ),
            "top_commodities": [
                {
                    "commodity": ca.commodity,
                    "emissions_tco2e": str(ca.total_emissions_tco2e),
                    "pct_of_flag": str(ca.pct_of_flag_total),
                    "risk": ca.deforestation_risk,
                }
                for ca in (result.commodity_assessments or [])[:5]
            ],
            "progress_status": (
                result.progress_tracking.status
                if result.progress_tracking else "not_tracked"
            ),
            "warnings_count": len(result.warnings),
            "errors_count": len(result.errors),
            "recommendations_count": len(result.recommendations),
        }
        summary["provenance_hash"] = _compute_hash(summary)
        return summary

    def get_commodity_comparison(
        self, result: FLAGAssessmentResult,
    ) -> List[Dict[str, Any]]:
        """Generate a commodity comparison table from result.

        Args:
            result: Complete FLAG assessment result.

        Returns:
            List of dicts with comparable metrics per commodity.
        """
        comparison: List[Dict[str, Any]] = []
        for ca in result.commodity_assessments:
            luc_em = Decimal("0")
            if ca.emission_breakdown:
                luc_em = ca.emission_breakdown.land_use_change_tco2e

            comparison.append({
                "commodity": ca.commodity,
                "total_emissions_tco2e": str(ca.total_emissions_tco2e),
                "pct_of_flag_total": str(ca.pct_of_flag_total),
                "pct_of_entity_total": str(ca.pct_of_entity_total),
                "luc_emissions_tco2e": str(luc_em),
                "intensity": str(ca.intensity_value),
                "intensity_unit": ca.intensity_unit,
                "deforestation_risk": ca.deforestation_risk,
                "certified": ca.has_certification,
                "traceability_pct": str(ca.traceability_pct),
                "data_quality": ca.data_quality,
                "confidence": ca.confidence,
                "target_reduction_pct": str(ca.required_reduction_pct),
            })
        return comparison

    def validate_base_year(
        self, base_year: int,
    ) -> Tuple[bool, List[str]]:
        """Validate FLAG base year against SBTi requirements.

        Args:
            base_year: Proposed base year.

        Returns:
            Tuple of (is_valid, list_of_issues).
        """
        issues: List[str] = []
        current_year = utcnow().year

        if base_year < FLAG_BASE_YEAR_MIN:
            issues.append(
                f"Base year {base_year} is before minimum "
                f"{FLAG_BASE_YEAR_MIN}."
            )
        if (current_year - base_year) > 5:
            issues.append(
                f"Base year {base_year} is more than 5 years old. "
                f"SBTi requires recent base years for new submissions."
            )

        return len(issues) == 0, issues

    def validate_target_timeframe(
        self,
        base_year: int,
        target_year: int,
        is_long_term: bool = False,
    ) -> Tuple[bool, List[str]]:
        """Validate FLAG target timeframe.

        Args:
            base_year: Base year.
            target_year: Target year.
            is_long_term: Whether this is a long-term target.

        Returns:
            Tuple of (is_valid, list_of_issues).
        """
        issues: List[str] = []
        elapsed = target_year - base_year

        if is_long_term:
            if target_year > FLAG_LONG_TERM_MAX_YEAR:
                issues.append(
                    f"Long-term target year {target_year} exceeds "
                    f"maximum {FLAG_LONG_TERM_MAX_YEAR}."
                )
        else:
            if elapsed < FLAG_NEAR_TERM_MIN_YEARS:
                issues.append(
                    f"Near-term timeframe ({elapsed} years) is less than "
                    f"minimum {FLAG_NEAR_TERM_MIN_YEARS} years."
                )
            if elapsed > FLAG_NEAR_TERM_MAX_YEARS:
                issues.append(
                    f"Near-term timeframe ({elapsed} years) exceeds "
                    f"maximum {FLAG_NEAR_TERM_MAX_YEARS} years."
                )

        return len(issues) == 0, issues

    def estimate_reduction_potential(
        self,
        commodity: str,
        current_emissions_tco2e: Decimal,
        interventions: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Estimate emission reduction potential for a commodity.

        Provides indicative reduction estimates based on common
        intervention types.  These are approximate ranges from
        published literature and should not replace detailed
        feasibility studies.

        Args:
            commodity: FLAG commodity name.
            current_emissions_tco2e: Current emissions for commodity.
            interventions: List of planned interventions.

        Returns:
            Dict with reduction estimates per intervention type.
        """
        commodity_key = commodity.lower().strip()
        interventions = interventions or []

        # Intervention reduction potential ranges (fraction of total)
        # Based on IPCC AR6 WG3 Chapter 7 mitigation options.
        intervention_potentials: Dict[str, Dict[str, Decimal]] = {
            "zero_deforestation": {
                "min_reduction_pct": Decimal("20"),
                "max_reduction_pct": Decimal("60"),
                "applicability": "All commodities with LUC emissions",
            },
            "sustainable_intensification": {
                "min_reduction_pct": Decimal("10"),
                "max_reduction_pct": Decimal("30"),
                "applicability": "Crop and livestock commodities",
            },
            "improved_feed_management": {
                "min_reduction_pct": Decimal("5"),
                "max_reduction_pct": Decimal("20"),
                "applicability": "Cattle and livestock",
            },
            "agroforestry": {
                "min_reduction_pct": Decimal("15"),
                "max_reduction_pct": Decimal("40"),
                "applicability": "Cocoa, coffee, rubber, palm oil",
            },
            "regenerative_agriculture": {
                "min_reduction_pct": Decimal("10"),
                "max_reduction_pct": Decimal("25"),
                "applicability": "All crop commodities",
            },
            "precision_agriculture": {
                "min_reduction_pct": Decimal("5"),
                "max_reduction_pct": Decimal("15"),
                "applicability": "Soy, maize, wheat, rice, sugarcane",
            },
            "alternate_wetting_drying": {
                "min_reduction_pct": Decimal("30"),
                "max_reduction_pct": Decimal("50"),
                "applicability": "Rice only",
            },
            "peatland_restoration": {
                "min_reduction_pct": Decimal("40"),
                "max_reduction_pct": Decimal("80"),
                "applicability": "Palm oil on peatlands",
            },
            "renewable_energy_transition": {
                "min_reduction_pct": Decimal("5"),
                "max_reduction_pct": Decimal("15"),
                "applicability": "All commodities (on-farm energy)",
            },
            "certification_and_standards": {
                "min_reduction_pct": Decimal("10"),
                "max_reduction_pct": Decimal("25"),
                "applicability": "All commodities",
            },
        }

        result_interventions: List[Dict[str, Any]] = []
        total_min = Decimal("0")
        total_max = Decimal("0")

        effective_interventions = interventions or list(
            intervention_potentials.keys()
        )

        for intervention in effective_interventions:
            potential = intervention_potentials.get(intervention)
            if not potential:
                continue

            min_pct = potential["min_reduction_pct"]
            max_pct = potential["max_reduction_pct"]
            min_tco2e = current_emissions_tco2e * min_pct / Decimal("100")
            max_tco2e = current_emissions_tco2e * max_pct / Decimal("100")

            result_interventions.append({
                "intervention": intervention,
                "min_reduction_pct": str(min_pct),
                "max_reduction_pct": str(max_pct),
                "min_reduction_tco2e": str(_round_val(min_tco2e)),
                "max_reduction_tco2e": str(_round_val(max_tco2e)),
                "applicability": str(potential["applicability"]),
            })

            total_min += min_pct
            total_max += max_pct

        return {
            "commodity": commodity_key,
            "current_emissions_tco2e": str(current_emissions_tco2e),
            "interventions": result_interventions,
            "combined_min_reduction_pct": str(
                min(total_min, Decimal("100"))
            ),
            "combined_max_reduction_pct": str(
                min(total_max, Decimal("100"))
            ),
            "note": (
                "Reduction potentials are indicative ranges from "
                "published literature (IPCC AR6 WG3 Chapter 7). "
                "Actual reductions depend on site-specific conditions. "
                "Combined reductions are not additive due to overlaps."
            ),
            "provenance_hash": _compute_hash({
                "commodity": commodity_key,
                "emissions": str(current_emissions_tco2e),
            }),
        }

    def get_sbti_flag_reference(self) -> Dict[str, Any]:
        """Return SBTi FLAG Guidance V1.1 key parameters.

        Provides a reference dict of all key parameters from the
        SBTi FLAG Guidance for documentation and audit purposes.

        Returns:
            Dict with all FLAG guidance key parameters.
        """
        return {
            "guidance_version": "V1.1",
            "guidance_year": 2022,
            "publisher": "Science Based Targets initiative (SBTi)",
            "annual_reduction_rate": {
                "value": str(FLAG_ANNUAL_RATE),
                "percentage": str(
                    _round_val(FLAG_ANNUAL_RATE * Decimal("100"), 2)
                ),
                "pathway_type": "linear",
                "section": "Section 5, Table 5.1",
            },
            "trigger_threshold": {
                "value": str(FLAG_TRIGGER_THRESHOLD),
                "percentage": str(
                    _round_val(
                        FLAG_TRIGGER_THRESHOLD * Decimal("100"), 1
                    )
                ),
                "section": "Section 3",
            },
            "no_deforestation": {
                "deadline_year": NO_DEFORESTATION_DEADLINE,
                "scope": "All FLAG commodities in value chain",
                "includes_no_conversion": True,
                "section": "Section 6",
            },
            "commodities": {
                "count": FLAG_COMMODITY_COUNT,
                "list": [c.value for c in FLAGCommodity],
            },
            "coverage_requirement": {
                "minimum_pct": str(
                    _round_val(FLAG_COVERAGE_MIN * Decimal("100"), 0)
                ),
            },
            "target_timeframe": {
                "near_term_min_years": FLAG_NEAR_TERM_MIN_YEARS,
                "near_term_max_years": FLAG_NEAR_TERM_MAX_YEARS,
                "long_term_max_year": FLAG_LONG_TERM_MAX_YEAR,
            },
            "long_term_reduction": {
                "minimum_pct": str(FLAG_LONG_TERM_MIN_REDUCTION_PCT),
            },
            "supporting_standards": [
                "IPCC AR6 WG3 (2022) - AFOLU emission factors",
                "GHG Protocol Agricultural Guidance (2014)",
                "GHG Protocol Land Sector and Removals Guidance (2022)",
                "Accountability Framework initiative (AFi)",
                "CDP Forests Questionnaire (2024)",
            ],
        }
