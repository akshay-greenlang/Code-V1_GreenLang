# -*- coding: utf-8 -*-
"""
SDASectorEngine - PACK-023 SBTi Alignment Engine 8
====================================================

Sectoral Decarbonisation Approach (SDA) intensity convergence engine
for 12 homogeneous sectors with 2050 benchmarks derived from IEA Net
Zero Emissions by 2050 Scenario (NZE 2023).  Implements the SBTi SDA
convergence formula, generates annual intensity milestones, validates
targets against SBTi SDA Tool V3.0 cross-checks, and produces
sector-specific reduction pathways.

The SDA methodology is used for Scope 1 + 2 targets in homogeneous
sectors where physical intensity metrics are appropriate (e.g.
tCO2e/MWh for power, tCO2e/tonne for cement).

Calculation Methodology:
    SDA Intensity Convergence Formula:
        I(t) = I_sector(t) + (I_company(base) - I_sector(base))
               * ((I_sector(target) - I_sector(t))
                  / (I_sector(target) - I_sector(base)))

    Where:
        I(t)              = Company intensity at year t
        I_sector(t)       = Sector pathway intensity at year t
        I_company(base)   = Company intensity at base year
        I_sector(base)    = Sector pathway intensity at base year
        I_sector(target)  = Sector pathway intensity at target year

    Sector Pathway Interpolation:
        I_sector(t) = I_sector(base) + (I_sector(target) - I_sector(base))
                      * ((t - base_year) / (target_year - base_year))

    Annual Reduction Rate:
        ARR = 1 - (I(t+1) / I(t))

    Cumulative Reduction:
        cumulative_pct = (1 - I(target) / I(base)) * 100

    Cross-Validation (SDA Tool V3.0):
        intensity_deviation = |I_calc(t) - I_tool(t)| / I_tool(t)
        PASS if deviation < 0.02 (2% tolerance)

Regulatory References:
    - SBTi Corporate Manual V5.3 (2024) - SDA methodology (C5, C6)
    - SBTi Sectoral Decarbonisation Approach (SDA) V2.1 (2024)
    - SBTi SDA Tool V3.0 (2024) - Cross-validation benchmarks
    - IEA Net Zero Emissions by 2050 Scenario (NZE 2023)
    - IEA Energy Technology Perspectives 2023
    - SBTi Target Validation Protocol V3.0 (2024)
    - ISO 14064-1:2018 - GHG quantification
    - GHG Protocol Corporate Standard (2004, Rev. 2015)
    - IPCC AR6 WG3 (2022) - Sector mitigation pathways

Zero-Hallucination:
    - All sector benchmarks from IEA NZE 2023 published data
    - Convergence formula from SBTi SDA V2.1 specification
    - No LLM involvement in any calculation path
    - Deterministic Decimal arithmetic throughout
    - SHA-256 provenance hash on every result

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-023 SBTi Alignment
Engine:  8 of 10
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

from pydantic import BaseModel, Field, field_validator

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

def _round_val(value: Decimal, places: int = 6) -> Decimal:
    """Round a Decimal to *places* using ROUND_HALF_UP."""
    quantize_str = "0." + "0" * places
    return value.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP)

def _round3(value: float) -> float:
    """Round to 3 decimal places using ROUND_HALF_UP."""
    return float(
        Decimal(str(value)).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)
    )

def _interpolate_linear(
    y_start: Decimal,
    y_end: Decimal,
    t: int,
    t_start: int,
    t_end: int,
) -> Decimal:
    """Linear interpolation between two points.

    Args:
        y_start: Value at t_start.
        y_end: Value at t_end.
        t: Year to interpolate for.
        t_start: Start year.
        t_end: End year.

    Returns:
        Interpolated value at year t.
    """
    if t_end == t_start:
        return y_start
    fraction = _decimal(t - t_start) / _decimal(t_end - t_start)
    return y_start + (y_end - y_start) * fraction

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class SdaSector(str, Enum):
    """SDA-eligible homogeneous sectors per SBTi SDA V2.1.

    Each sector has a well-defined physical intensity metric and
    IEA NZE 2023 benchmark pathway.
    """
    POWER_GENERATION = "power_generation"
    CEMENT = "cement"
    IRON_STEEL = "iron_steel"
    ALUMINIUM = "aluminium"
    PULP_PAPER = "pulp_paper"
    COMMERCIAL_BUILDINGS = "commercial_buildings"
    RESIDENTIAL_BUILDINGS = "residential_buildings"
    PASSENGER_TRANSPORT = "passenger_transport"
    FREIGHT_TRANSPORT = "freight_transport"
    CHEMICALS = "chemicals"
    AGRICULTURE = "agriculture"
    AVIATION = "aviation"

class IntensityUnit(str, Enum):
    """Physical intensity metric units per sector.

    Each SDA sector uses a specific physical intensity denominator
    reflecting its primary output.
    """
    TCO2E_PER_MWH = "tCO2e/MWh"
    TCO2E_PER_TONNE_CEMENT = "tCO2e/t cement"
    TCO2E_PER_TONNE_STEEL = "tCO2e/t steel"
    TCO2E_PER_TONNE_ALUMINIUM = "tCO2e/t aluminium"
    TCO2E_PER_TONNE_PULP = "tCO2e/t pulp"
    TCO2E_PER_SQM = "tCO2e/m2"
    TCO2E_PER_PKM = "tCO2e/pkm"
    TCO2E_PER_TKM = "tCO2e/tkm"
    TCO2E_PER_TONNE_CHEMICAL = "tCO2e/t chemical"
    TCO2E_PER_TONNE_CROP = "tCO2e/t crop"
    TCO2E_PER_RPK = "tCO2e/RPK"

class ConvergenceStatus(str, Enum):
    """Status of convergence to the sector pathway.

    CONVERGING:     Company pathway converges to sector benchmark.
    ALIGNED:        Company is at or below sector pathway.
    ABOVE_PATHWAY:  Company is above sector pathway at target year.
    INSUFFICIENT:   Reduction rate is below SBTi minimum.
    NOT_ASSESSED:   Convergence not yet calculated.
    """
    CONVERGING = "converging"
    ALIGNED = "aligned"
    ABOVE_PATHWAY = "above_pathway"
    INSUFFICIENT = "insufficient"
    NOT_ASSESSED = "not_assessed"

class ValidationStatus(str, Enum):
    """Cross-validation result against SBTi SDA Tool V3.0.

    PASS:           Deviation within 2% tolerance.
    MARGINAL:       Deviation between 2% and 5%.
    FAIL:           Deviation exceeds 5%.
    NOT_VALIDATED:  Cross-validation not performed.
    """
    PASS_VALID = "pass"
    MARGINAL = "marginal"
    FAIL = "fail"
    NOT_VALIDATED = "not_validated"

class AmbitionLevel(str, Enum):
    """Target ambition level per SBTi classification.

    WELL_BELOW_2C:  Well-below 2 degrees C aligned.
    C_1_5:          1.5 degrees C aligned.
    NET_ZERO:       Net-zero aligned (SBTi NZ Standard).
    BELOW_MINIMUM:  Below SBTi minimum ambition.
    """
    WELL_BELOW_2C = "well_below_2c"
    C_1_5 = "1.5c"
    NET_ZERO = "net_zero"
    BELOW_MINIMUM = "below_minimum"

class ScopeInclusion(str, Enum):
    """Which scopes are included in the SDA target.

    SCOPE_1:    Scope 1 only.
    SCOPE_1_2:  Scope 1 and 2 combined.
    SCOPE_2:    Scope 2 only (rare, location-based).
    """
    SCOPE_1 = "scope_1"
    SCOPE_1_2 = "scope_1_2"
    SCOPE_2 = "scope_2"

# ---------------------------------------------------------------------------
# Constants -- IEA NZE 2023 Sector Benchmarks
# ---------------------------------------------------------------------------

# Sector benchmark data: base year 2020 intensities and 2050 targets.
# Source: IEA Net Zero Emissions by 2050 Scenario (NZE 2023 Update).
# All values are physical intensity metrics in the sector's native unit.
# Intermediate milestone years (2025, 2030, 2035, 2040, 2045) interpolated
# from published IEA NZE trajectory data.
#
# Structure per sector:
#   "unit": intensity metric unit
#   "description": sector description
#   "base_year": reference year for pathway (2020)
#   "milestones": {year: intensity_value}
#   "min_annual_reduction_rate": SBTi minimum ARR for this sector
#   "wb2c_annual_rate": well-below 2C minimum rate
#   "c15_annual_rate": 1.5C-aligned minimum rate

SECTOR_BENCHMARKS: Dict[str, Dict[str, Any]] = {
    SdaSector.POWER_GENERATION.value: {
        "unit": IntensityUnit.TCO2E_PER_MWH.value,
        "description": "Electricity and heat generation",
        "base_year": 2020,
        "milestones": {
            2020: "0.4500",
            2025: "0.3380",
            2030: "0.1380",
            2035: "0.0550",
            2040: "0.0200",
            2045: "0.0050",
            2050: "0.0000",
        },
        "min_annual_reduction_rate": "0.042",
        "wb2c_annual_rate": "0.042",
        "c15_annual_rate": "0.070",
    },
    SdaSector.CEMENT.value: {
        "unit": IntensityUnit.TCO2E_PER_TONNE_CEMENT.value,
        "description": "Cement and clinker production",
        "base_year": 2020,
        "milestones": {
            2020: "0.6100",
            2025: "0.5600",
            2030: "0.4300",
            2035: "0.3200",
            2040: "0.2100",
            2045: "0.1200",
            2050: "0.0600",
        },
        "min_annual_reduction_rate": "0.030",
        "wb2c_annual_rate": "0.030",
        "c15_annual_rate": "0.050",
    },
    SdaSector.IRON_STEEL.value: {
        "unit": IntensityUnit.TCO2E_PER_TONNE_STEEL.value,
        "description": "Iron and steel production",
        "base_year": 2020,
        "milestones": {
            2020: "1.4000",
            2025: "1.2600",
            2030: "0.9800",
            2035: "0.6700",
            2040: "0.4000",
            2045: "0.2000",
            2050: "0.0500",
        },
        "min_annual_reduction_rate": "0.030",
        "wb2c_annual_rate": "0.030",
        "c15_annual_rate": "0.050",
    },
    SdaSector.ALUMINIUM.value: {
        "unit": IntensityUnit.TCO2E_PER_TONNE_ALUMINIUM.value,
        "description": "Primary aluminium smelting",
        "base_year": 2020,
        "milestones": {
            2020: "8.5000",
            2025: "7.5000",
            2030: "5.5000",
            2035: "3.8000",
            2040: "2.5000",
            2045: "1.2000",
            2050: "0.3000",
        },
        "min_annual_reduction_rate": "0.030",
        "wb2c_annual_rate": "0.030",
        "c15_annual_rate": "0.048",
    },
    SdaSector.PULP_PAPER.value: {
        "unit": IntensityUnit.TCO2E_PER_TONNE_PULP.value,
        "description": "Pulp and paper production",
        "base_year": 2020,
        "milestones": {
            2020: "0.4500",
            2025: "0.3900",
            2030: "0.3000",
            2035: "0.2100",
            2040: "0.1400",
            2045: "0.0700",
            2050: "0.0200",
        },
        "min_annual_reduction_rate": "0.025",
        "wb2c_annual_rate": "0.025",
        "c15_annual_rate": "0.042",
    },
    SdaSector.COMMERCIAL_BUILDINGS.value: {
        "unit": IntensityUnit.TCO2E_PER_SQM.value,
        "description": "Commercial and institutional buildings",
        "base_year": 2020,
        "milestones": {
            2020: "0.0700",
            2025: "0.0560",
            2030: "0.0380",
            2035: "0.0240",
            2040: "0.0130",
            2045: "0.0050",
            2050: "0.0010",
        },
        "min_annual_reduction_rate": "0.042",
        "wb2c_annual_rate": "0.042",
        "c15_annual_rate": "0.070",
    },
    SdaSector.RESIDENTIAL_BUILDINGS.value: {
        "unit": IntensityUnit.TCO2E_PER_SQM.value,
        "description": "Residential buildings",
        "base_year": 2020,
        "milestones": {
            2020: "0.0550",
            2025: "0.0440",
            2030: "0.0300",
            2035: "0.0190",
            2040: "0.0100",
            2045: "0.0040",
            2050: "0.0008",
        },
        "min_annual_reduction_rate": "0.042",
        "wb2c_annual_rate": "0.042",
        "c15_annual_rate": "0.070",
    },
    SdaSector.PASSENGER_TRANSPORT.value: {
        "unit": IntensityUnit.TCO2E_PER_PKM.value,
        "description": "Passenger road and rail transport",
        "base_year": 2020,
        "milestones": {
            2020: "0.0001050",
            2025: "0.0000850",
            2030: "0.0000600",
            2035: "0.0000380",
            2040: "0.0000200",
            2045: "0.0000080",
            2050: "0.0000020",
        },
        "min_annual_reduction_rate": "0.042",
        "wb2c_annual_rate": "0.042",
        "c15_annual_rate": "0.070",
    },
    SdaSector.FREIGHT_TRANSPORT.value: {
        "unit": IntensityUnit.TCO2E_PER_TKM.value,
        "description": "Freight road, rail, and shipping",
        "base_year": 2020,
        "milestones": {
            2020: "0.0000800",
            2025: "0.0000680",
            2030: "0.0000500",
            2035: "0.0000340",
            2040: "0.0000200",
            2045: "0.0000100",
            2050: "0.0000030",
        },
        "min_annual_reduction_rate": "0.030",
        "wb2c_annual_rate": "0.030",
        "c15_annual_rate": "0.050",
    },
    SdaSector.CHEMICALS.value: {
        "unit": IntensityUnit.TCO2E_PER_TONNE_CHEMICAL.value,
        "description": "Primary chemicals production",
        "base_year": 2020,
        "milestones": {
            2020: "0.8500",
            2025: "0.7600",
            2030: "0.5800",
            2035: "0.4100",
            2040: "0.2600",
            2045: "0.1300",
            2050: "0.0400",
        },
        "min_annual_reduction_rate": "0.025",
        "wb2c_annual_rate": "0.025",
        "c15_annual_rate": "0.042",
    },
    SdaSector.AGRICULTURE.value: {
        "unit": IntensityUnit.TCO2E_PER_TONNE_CROP.value,
        "description": "Agricultural production",
        "base_year": 2020,
        "milestones": {
            2020: "0.3200",
            2025: "0.2900",
            2030: "0.2400",
            2035: "0.1900",
            2040: "0.1500",
            2045: "0.1100",
            2050: "0.0800",
        },
        "min_annual_reduction_rate": "0.020",
        "wb2c_annual_rate": "0.020",
        "c15_annual_rate": "0.035",
    },
    SdaSector.AVIATION.value: {
        "unit": IntensityUnit.TCO2E_PER_RPK.value,
        "description": "Commercial aviation (passenger)",
        "base_year": 2020,
        "milestones": {
            2020: "0.0001000",
            2025: "0.0000880",
            2030: "0.0000680",
            2035: "0.0000480",
            2040: "0.0000310",
            2045: "0.0000170",
            2050: "0.0000060",
        },
        "min_annual_reduction_rate": "0.025",
        "wb2c_annual_rate": "0.025",
        "c15_annual_rate": "0.042",
    },
}

# Cross-validation tolerance: maximum acceptable deviation from
# SBTi SDA Tool V3.0 output as a fraction.
# Source: SBTi Target Validation Protocol V3.0, Section 4.2.
CROSS_VALIDATION_TOLERANCE: Decimal = Decimal("0.02")

# Marginal tolerance threshold (warning but not failure).
CROSS_VALIDATION_MARGINAL: Decimal = Decimal("0.05")

# Minimum target duration in years per SBTi V5.3, Section 4.3.
MIN_TARGET_DURATION_YEARS: int = 5

# Maximum target duration for near-term targets.
MAX_NEAR_TERM_DURATION_YEARS: int = 10

# Maximum target year for long-term SDA targets.
MAX_LONG_TERM_TARGET_YEAR: int = 2050

# Minimum base year per SBTi V5.3.
MIN_BASE_YEAR: int = 2015

# Total number of SDA-eligible sectors.
TOTAL_SDA_SECTORS: int = 12

# ---------------------------------------------------------------------------
# Pydantic Models -- Input
# ---------------------------------------------------------------------------

class CompanyIntensityInput(BaseModel):
    """Company-specific intensity data for SDA convergence.

    Attributes:
        company_name: Reporting entity name.
        sector: SDA sector identifier.
        base_year: Emissions base year.
        target_year: Target year for convergence.
        base_year_intensity: Company intensity at base year.
        current_year: Most recent reporting year.
        current_intensity: Company intensity at current year.
        base_year_production: Production volume at base year.
        current_production: Production volume at current year.
        projected_production_growth_pct: Annual production growth (%).
        scope_inclusion: Which scopes are included (S1, S1+S2, S2).
        absolute_emissions_base_tco2e: Total absolute emissions at base year.
        absolute_emissions_current_tco2e: Total absolute emissions current.
        intensity_unit_override: Override the default intensity unit.
        notes: Additional context or notes.
    """
    company_name: str = Field(
        ..., min_length=1, max_length=300,
        description="Reporting entity name"
    )
    sector: str = Field(
        ..., description="SDA sector identifier"
    )
    base_year: int = Field(
        ..., ge=2015, le=2030,
        description="Emissions base year"
    )
    target_year: int = Field(
        ..., ge=2025, le=2055,
        description="Target year for convergence"
    )
    base_year_intensity: Decimal = Field(
        ..., ge=0,
        description="Company intensity at base year"
    )
    current_year: int = Field(
        default=0, ge=0, le=2030,
        description="Most recent reporting year (0 = same as base)"
    )
    current_intensity: Decimal = Field(
        default=Decimal("0"), ge=0,
        description="Company intensity at current year"
    )
    base_year_production: Decimal = Field(
        default=Decimal("0"), ge=0,
        description="Production volume at base year"
    )
    current_production: Decimal = Field(
        default=Decimal("0"), ge=0,
        description="Production volume at current year"
    )
    projected_production_growth_pct: Decimal = Field(
        default=Decimal("0"),
        description="Projected annual production growth rate (%)"
    )
    scope_inclusion: str = Field(
        default=ScopeInclusion.SCOPE_1_2.value,
        description="Which scopes are included"
    )
    absolute_emissions_base_tco2e: Decimal = Field(
        default=Decimal("0"), ge=0,
        description="Absolute emissions at base year (tCO2e)"
    )
    absolute_emissions_current_tco2e: Decimal = Field(
        default=Decimal("0"), ge=0,
        description="Absolute emissions at current year (tCO2e)"
    )
    intensity_unit_override: str = Field(
        default="",
        description="Override default intensity unit"
    )
    notes: str = Field(
        default="",
        description="Additional context or notes"
    )

    @field_validator("sector")
    @classmethod
    def validate_sector(cls, v: str) -> str:
        """Validate sector is a known SDA sector."""
        valid = {s.value for s in SdaSector}
        if v not in valid:
            raise ValueError(
                f"Unknown SDA sector '{v}'. "
                f"Must be one of: {sorted(valid)}"
            )
        return v

    @field_validator("scope_inclusion")
    @classmethod
    def validate_scope_inclusion(cls, v: str) -> str:
        """Validate scope inclusion."""
        valid = {s.value for s in ScopeInclusion}
        if v not in valid:
            raise ValueError(
                f"Unknown scope inclusion '{v}'. "
                f"Must be one of: {sorted(valid)}"
            )
        return v

    @field_validator("current_year")
    @classmethod
    def validate_current_year(cls, v: int, info: Any) -> int:
        """Default current year to base year if zero."""
        if v == 0:
            base = info.data.get("base_year", 2023)
            return base
        return v

class CrossValidationInput(BaseModel):
    """Cross-validation reference data from SBTi SDA Tool V3.0.

    Attributes:
        year: Reference year for the data point.
        sda_tool_intensity: Intensity value from SBTi SDA Tool.
        sda_tool_absolute: Absolute emissions from SBTi SDA Tool.
    """
    year: int = Field(
        ..., ge=2015, le=2060,
        description="Reference year"
    )
    sda_tool_intensity: Decimal = Field(
        ..., ge=0,
        description="Intensity from SBTi SDA Tool V3.0"
    )
    sda_tool_absolute: Decimal = Field(
        default=Decimal("0"), ge=0,
        description="Absolute emissions from SBTi SDA Tool"
    )

class SdaInput(BaseModel):
    """Complete SDA analysis input.

    Attributes:
        company: Company-specific intensity data.
        cross_validation_points: Reference data from SDA Tool V3.0.
        include_annual_milestones: Generate year-by-year milestones.
        include_ambition_assessment: Assess ambition level.
        include_cross_validation: Perform SDA Tool cross-validation.
        include_absolute_pathway: Generate absolute emissions pathway.
        include_production_forecast: Include production projections.
    """
    company: CompanyIntensityInput = Field(
        ..., description="Company intensity data"
    )
    cross_validation_points: List[CrossValidationInput] = Field(
        default_factory=list,
        description="SDA Tool V3.0 cross-validation data"
    )
    include_annual_milestones: bool = Field(
        default=True,
        description="Generate annual intensity milestones"
    )
    include_ambition_assessment: bool = Field(
        default=True,
        description="Assess ambition level against thresholds"
    )
    include_cross_validation: bool = Field(
        default=True,
        description="Perform SDA Tool V3.0 cross-validation"
    )
    include_absolute_pathway: bool = Field(
        default=True,
        description="Generate absolute emissions pathway"
    )
    include_production_forecast: bool = Field(
        default=True,
        description="Include production volume projections"
    )

# ---------------------------------------------------------------------------
# Pydantic Models -- Output
# ---------------------------------------------------------------------------

class AnnualMilestone(BaseModel):
    """A single year's intensity milestone on the SDA pathway.

    Attributes:
        year: Calendar year.
        sector_intensity: Sector benchmark intensity at this year.
        company_intensity: Company converged intensity at this year.
        intensity_reduction_from_base_pct: Cumulative reduction from base (%).
        annual_reduction_rate_pct: Year-over-year reduction rate (%).
        is_on_track: Whether company is at or below sector pathway.
        absolute_emissions_tco2e: Estimated absolute emissions.
        production_volume: Estimated production volume.
    """
    year: int = Field(default=0)
    sector_intensity: Decimal = Field(default=Decimal("0"))
    company_intensity: Decimal = Field(default=Decimal("0"))
    intensity_reduction_from_base_pct: Decimal = Field(default=Decimal("0"))
    annual_reduction_rate_pct: Decimal = Field(default=Decimal("0"))
    is_on_track: bool = Field(default=True)
    absolute_emissions_tco2e: Decimal = Field(default=Decimal("0"))
    production_volume: Decimal = Field(default=Decimal("0"))

class ConvergenceAssessment(BaseModel):
    """SDA convergence assessment result.

    Attributes:
        sector: SDA sector.
        sector_description: Human-readable sector description.
        intensity_unit: Physical intensity metric unit.
        base_year: Base year.
        target_year: Target year.
        company_base_intensity: Company intensity at base year.
        company_target_intensity: Company converged intensity at target.
        sector_base_intensity: Sector benchmark at base year.
        sector_target_intensity: Sector benchmark at target year.
        total_reduction_pct: Total intensity reduction base to target (%).
        avg_annual_reduction_rate_pct: Average annual reduction rate (%).
        convergence_status: Whether company converges to sector pathway.
        years_to_target: Number of years from base to target.
        intensity_gap_at_base: Company vs sector gap at base year.
        intensity_gap_at_target: Company vs sector gap at target year.
        convergence_year: Year when company meets sector (if applicable).
        message: Human-readable convergence assessment.
    """
    sector: str = Field(default="")
    sector_description: str = Field(default="")
    intensity_unit: str = Field(default="")
    base_year: int = Field(default=0)
    target_year: int = Field(default=0)
    company_base_intensity: Decimal = Field(default=Decimal("0"))
    company_target_intensity: Decimal = Field(default=Decimal("0"))
    sector_base_intensity: Decimal = Field(default=Decimal("0"))
    sector_target_intensity: Decimal = Field(default=Decimal("0"))
    total_reduction_pct: Decimal = Field(default=Decimal("0"))
    avg_annual_reduction_rate_pct: Decimal = Field(default=Decimal("0"))
    convergence_status: str = Field(default=ConvergenceStatus.NOT_ASSESSED.value)
    years_to_target: int = Field(default=0)
    intensity_gap_at_base: Decimal = Field(default=Decimal("0"))
    intensity_gap_at_target: Decimal = Field(default=Decimal("0"))
    convergence_year: int = Field(default=0)
    message: str = Field(default="")

class AmbitionAssessment(BaseModel):
    """Ambition level assessment per SBTi classification.

    Attributes:
        ambition_level: SBTi ambition classification.
        company_annual_rate_pct: Company's average annual reduction (%).
        wb2c_threshold_pct: Well-below 2C threshold (%).
        c15_threshold_pct: 1.5C threshold (%).
        sector_minimum_pct: Sector-specific SBTi minimum (%).
        exceeds_wb2c: Whether target exceeds WB2C threshold.
        exceeds_c15: Whether target exceeds 1.5C threshold.
        exceeds_minimum: Whether target meets SBTi minimum.
        gap_to_minimum_pct: Gap to minimum threshold (%).
        gap_to_wb2c_pct: Gap to WB2C threshold (%).
        gap_to_c15_pct: Gap to 1.5C threshold (%).
        message: Human-readable ambition assessment.
    """
    ambition_level: str = Field(default=AmbitionLevel.BELOW_MINIMUM.value)
    company_annual_rate_pct: Decimal = Field(default=Decimal("0"))
    wb2c_threshold_pct: Decimal = Field(default=Decimal("0"))
    c15_threshold_pct: Decimal = Field(default=Decimal("0"))
    sector_minimum_pct: Decimal = Field(default=Decimal("0"))
    exceeds_wb2c: bool = Field(default=False)
    exceeds_c15: bool = Field(default=False)
    exceeds_minimum: bool = Field(default=False)
    gap_to_minimum_pct: Decimal = Field(default=Decimal("0"))
    gap_to_wb2c_pct: Decimal = Field(default=Decimal("0"))
    gap_to_c15_pct: Decimal = Field(default=Decimal("0"))
    message: str = Field(default="")

class CrossValidationResult(BaseModel):
    """Cross-validation result against SBTi SDA Tool V3.0.

    Attributes:
        year: Reference year.
        calculated_intensity: Engine-calculated intensity.
        sda_tool_intensity: SBTi SDA Tool intensity.
        absolute_deviation: Absolute difference.
        relative_deviation_pct: Relative deviation (%).
        validation_status: PASS/MARGINAL/FAIL.
        within_tolerance: Whether within 2% tolerance.
    """
    year: int = Field(default=0)
    calculated_intensity: Decimal = Field(default=Decimal("0"))
    sda_tool_intensity: Decimal = Field(default=Decimal("0"))
    absolute_deviation: Decimal = Field(default=Decimal("0"))
    relative_deviation_pct: Decimal = Field(default=Decimal("0"))
    validation_status: str = Field(default=ValidationStatus.NOT_VALIDATED.value)
    within_tolerance: bool = Field(default=True)

class CrossValidationSummary(BaseModel):
    """Summary of SDA Tool cross-validation.

    Attributes:
        total_points: Number of validation points.
        points_passed: Points within 2% tolerance.
        points_marginal: Points between 2% and 5%.
        points_failed: Points exceeding 5% deviation.
        max_deviation_pct: Maximum deviation across all points.
        avg_deviation_pct: Average deviation across all points.
        overall_status: Overall cross-validation status.
        details: Per-year validation details.
        message: Human-readable summary.
    """
    total_points: int = Field(default=0)
    points_passed: int = Field(default=0)
    points_marginal: int = Field(default=0)
    points_failed: int = Field(default=0)
    max_deviation_pct: Decimal = Field(default=Decimal("0"))
    avg_deviation_pct: Decimal = Field(default=Decimal("0"))
    overall_status: str = Field(default=ValidationStatus.NOT_VALIDATED.value)
    details: List[CrossValidationResult] = Field(default_factory=list)
    message: str = Field(default="")

class AbsolutePathwayPoint(BaseModel):
    """A single year's absolute emissions on the SDA pathway.

    Attributes:
        year: Calendar year.
        intensity: Company intensity at this year.
        production_volume: Estimated production volume.
        absolute_emissions_tco2e: Intensity * production volume.
        reduction_from_base_pct: Cumulative absolute reduction (%).
    """
    year: int = Field(default=0)
    intensity: Decimal = Field(default=Decimal("0"))
    production_volume: Decimal = Field(default=Decimal("0"))
    absolute_emissions_tco2e: Decimal = Field(default=Decimal("0"))
    reduction_from_base_pct: Decimal = Field(default=Decimal("0"))

class ProductionForecast(BaseModel):
    """Production volume forecast for absolute pathway.

    Attributes:
        base_year: Base year production.
        base_production: Production at base year.
        annual_growth_pct: Annual growth rate (%).
        projected_years: List of projected (year, volume) pairs.
    """
    base_year: int = Field(default=0)
    base_production: Decimal = Field(default=Decimal("0"))
    annual_growth_pct: Decimal = Field(default=Decimal("0"))
    projected_years: List[Dict[str, Any]] = Field(default_factory=list)

class SectorBenchmarkInfo(BaseModel):
    """Sector benchmark reference information.

    Attributes:
        sector: Sector identifier.
        description: Sector description.
        unit: Intensity unit.
        base_year_intensity: IEA NZE base intensity.
        target_year_intensity: IEA NZE 2050 intensity.
        milestones: Key milestone intensities.
        min_annual_rate_pct: SBTi minimum annual rate (%).
        wb2c_annual_rate_pct: WB2C annual rate (%).
        c15_annual_rate_pct: 1.5C annual rate (%).
        source: Data source reference.
    """
    sector: str = Field(default="")
    description: str = Field(default="")
    unit: str = Field(default="")
    base_year_intensity: Decimal = Field(default=Decimal("0"))
    target_year_intensity: Decimal = Field(default=Decimal("0"))
    milestones: Dict[str, Decimal] = Field(default_factory=dict)
    min_annual_rate_pct: Decimal = Field(default=Decimal("0"))
    wb2c_annual_rate_pct: Decimal = Field(default=Decimal("0"))
    c15_annual_rate_pct: Decimal = Field(default=Decimal("0"))
    source: str = Field(default="IEA Net Zero Emissions by 2050 (NZE 2023)")

class SdaRecommendation(BaseModel):
    """A single SDA-specific recommendation.

    Attributes:
        recommendation_id: Unique recommendation identifier.
        priority: Priority level (immediate/short/medium/long).
        category: Recommendation category.
        action: Description of recommended action.
        rationale: Why this action is recommended.
        estimated_impact: Expected impact description.
        timeline_months: Estimated implementation time.
    """
    recommendation_id: str = Field(default_factory=_new_uuid)
    priority: str = Field(default="medium_term")
    category: str = Field(default="general")
    action: str = Field(default="")
    rationale: str = Field(default="")
    estimated_impact: str = Field(default="")
    timeline_months: int = Field(default=12)

class SdaResult(BaseModel):
    """Complete SDA analysis result.

    Attributes:
        result_id: Unique result identifier.
        engine_version: Engine version.
        calculated_at: Timestamp.
        company_name: Entity name.
        sector: SDA sector.
        convergence: Convergence assessment.
        ambition: Ambition level assessment.
        annual_milestones: Year-by-year intensity milestones.
        cross_validation: SDA Tool cross-validation.
        absolute_pathway: Absolute emissions pathway.
        production_forecast: Production volume forecast.
        sector_benchmark: Sector benchmark reference.
        recommendations: SDA-specific recommendations.
        warnings: Non-blocking warnings.
        errors: Blocking errors.
        processing_time_ms: Processing duration (ms).
        provenance_hash: SHA-256 audit hash.
    """
    result_id: str = Field(default_factory=_new_uuid)
    engine_version: str = Field(default=_MODULE_VERSION)
    calculated_at: datetime = Field(default_factory=utcnow)
    company_name: str = Field(default="")
    sector: str = Field(default="")
    convergence: Optional[ConvergenceAssessment] = Field(None)
    ambition: Optional[AmbitionAssessment] = Field(None)
    annual_milestones: List[AnnualMilestone] = Field(default_factory=list)
    cross_validation: Optional[CrossValidationSummary] = Field(None)
    absolute_pathway: List[AbsolutePathwayPoint] = Field(default_factory=list)
    production_forecast: Optional[ProductionForecast] = Field(None)
    sector_benchmark: Optional[SectorBenchmarkInfo] = Field(None)
    recommendations: List[SdaRecommendation] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    processing_time_ms: float = Field(default=0.0)
    provenance_hash: str = Field(default="")

# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class SDASectorEngine:
    """SBTi Sectoral Decarbonisation Approach (SDA) engine.

    Performs SDA intensity convergence analysis for 12 homogeneous
    sectors including:
      - SDA convergence formula calculation
      - Annual intensity milestone generation
      - Ambition level assessment (WB2C / 1.5C / Net-Zero)
      - Cross-validation against SBTi SDA Tool V3.0
      - Absolute emissions pathway with production forecasts
      - Sector-specific reduction recommendations

    All calculations use deterministic Decimal arithmetic with SHA-256
    provenance hashing.  No LLM involvement in any calculation path.

    Usage::

        engine = SDASectorEngine()
        result = engine.analyse(input_data)
        print(f"Convergence: {result.convergence.convergence_status}")
        for ms in result.annual_milestones:
            print(f"  {ms.year}: {ms.company_intensity} {ms.sector_intensity}")
    """

    engine_version: str = _MODULE_VERSION

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialise SDASectorEngine.

        Args:
            config: Optional configuration overrides.  Supported keys:
                - cross_validation_tolerance (Decimal)
                - cross_validation_marginal (Decimal)
                - include_all_years (bool)
                - custom_benchmarks (Dict)
        """
        self.config = config or {}
        self._cv_tolerance = _decimal(
            self.config.get(
                "cross_validation_tolerance", CROSS_VALIDATION_TOLERANCE
            )
        )
        self._cv_marginal = _decimal(
            self.config.get(
                "cross_validation_marginal", CROSS_VALIDATION_MARGINAL
            )
        )
        self._include_all_years = bool(
            self.config.get("include_all_years", True)
        )
        self._custom_benchmarks: Dict[str, Dict[str, Any]] = self.config.get(
            "custom_benchmarks", {}
        )
        logger.info(
            "SDASectorEngine v%s initialised", self.engine_version
        )

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #

    def analyse(self, data: SdaInput) -> SdaResult:
        """Perform complete SDA convergence analysis.

        Orchestrates the full SDA pipeline: calculates converged
        intensity pathway, generates annual milestones, assesses
        ambition level, cross-validates against SDA Tool V3.0,
        generates absolute pathway, and produces recommendations.

        Args:
            data: Validated SDA input.

        Returns:
            SdaResult with all assessments and milestones.
        """
        t0 = time.perf_counter()
        company = data.company
        logger.info(
            "SDA analysis: company=%s, sector=%s, base=%d, target=%d",
            company.company_name, company.sector,
            company.base_year, company.target_year,
        )

        warnings: List[str] = []
        errors: List[str] = []

        # Step 1: Retrieve sector benchmark
        benchmark = self._get_sector_benchmark(company.sector)
        if benchmark is None:
            errors.append(
                f"Unknown sector '{company.sector}'. "
                f"Cannot perform SDA analysis."
            )
            elapsed_ms = _round3((time.perf_counter() - t0) * 1000.0)
            result = SdaResult(
                company_name=company.company_name,
                sector=company.sector,
                errors=errors,
                processing_time_ms=elapsed_ms,
            )
            result.provenance_hash = _compute_hash(result)
            return result

        # Step 2: Validate inputs
        input_warnings = self._validate_inputs(company, benchmark)
        warnings.extend(input_warnings)

        # Step 3: Get sector benchmark info
        sector_info = self._build_sector_benchmark_info(
            company.sector, benchmark
        )

        # Step 4: Calculate convergence
        convergence = self._calculate_convergence(company, benchmark)

        # Step 5: Generate annual milestones
        milestones: List[AnnualMilestone] = []
        if data.include_annual_milestones:
            milestones = self._generate_annual_milestones(
                company, benchmark, convergence
            )

        # Step 6: Ambition assessment
        ambition: Optional[AmbitionAssessment] = None
        if data.include_ambition_assessment:
            ambition = self._assess_ambition(
                convergence, benchmark, company
            )

        # Step 7: Cross-validation
        cross_val: Optional[CrossValidationSummary] = None
        if data.include_cross_validation and data.cross_validation_points:
            cross_val = self._cross_validate(
                company, benchmark, data.cross_validation_points
            )
        elif data.include_cross_validation and not data.cross_validation_points:
            warnings.append(
                "Cross-validation requested but no SDA Tool V3.0 "
                "reference data provided. Skipping cross-validation."
            )

        # Step 8: Production forecast
        prod_forecast: Optional[ProductionForecast] = None
        if data.include_production_forecast:
            prod_forecast = self._generate_production_forecast(company)

        # Step 9: Absolute pathway
        abs_pathway: List[AbsolutePathwayPoint] = []
        if data.include_absolute_pathway:
            abs_pathway = self._generate_absolute_pathway(
                company, milestones, prod_forecast
            )

        # Step 10: Recommendations
        recommendations = self._generate_recommendations(
            convergence, ambition, cross_val, company, benchmark
        )

        # Step 11: Additional warnings
        if convergence.convergence_status == ConvergenceStatus.ABOVE_PATHWAY.value:
            warnings.append(
                f"Company intensity at target year "
                f"({convergence.company_target_intensity}) exceeds "
                f"sector benchmark ({convergence.sector_target_intensity}). "
                f"Target may not meet SBTi requirements."
            )
        if convergence.convergence_status == ConvergenceStatus.INSUFFICIENT.value:
            warnings.append(
                "Annual reduction rate is below the SBTi minimum "
                "for this sector. Target ambition is insufficient."
            )

        elapsed_ms = _round3((time.perf_counter() - t0) * 1000.0)

        result = SdaResult(
            company_name=company.company_name,
            sector=company.sector,
            convergence=convergence,
            ambition=ambition,
            annual_milestones=milestones,
            cross_validation=cross_val,
            absolute_pathway=abs_pathway,
            production_forecast=prod_forecast,
            sector_benchmark=sector_info,
            recommendations=recommendations,
            warnings=warnings,
            errors=errors,
            processing_time_ms=elapsed_ms,
        )
        result.provenance_hash = _compute_hash(result)

        logger.info(
            "SDA analysis complete: sector=%s, convergence=%s, "
            "milestones=%d, ambition=%s, hash=%s",
            company.sector,
            convergence.convergence_status,
            len(milestones),
            ambition.ambition_level if ambition else "n/a",
            result.provenance_hash[:16],
        )
        return result

    def calculate_converged_intensity(
        self,
        company_base_intensity: Decimal,
        sector_base_intensity: Decimal,
        sector_target_intensity: Decimal,
        sector_intensity_at_t: Decimal,
    ) -> Decimal:
        """Calculate company converged intensity at year t using SDA formula.

        SDA Convergence Formula:
            I(t) = I_sector(t) + (I_company(base) - I_sector(base))
                   * ((I_sector(target) - I_sector(t))
                      / (I_sector(target) - I_sector(base)))

        This is the core SDA calculation from SBTi SDA V2.1.
        When the company starts above the sector pathway, it converges
        towards the sector benchmark at the target year.  When the company
        starts below, it maintains its advantage proportionally.

        Args:
            company_base_intensity: Company intensity at base year.
            sector_base_intensity: Sector benchmark at base year.
            sector_target_intensity: Sector benchmark at target year.
            sector_intensity_at_t: Sector benchmark at year t.

        Returns:
            Company converged intensity at year t.
        """
        delta_company_sector = company_base_intensity - sector_base_intensity
        denominator = sector_target_intensity - sector_base_intensity

        if denominator == Decimal("0"):
            # Sector pathway is flat; company simply follows sector
            return sector_intensity_at_t

        convergence_factor = (
            (sector_target_intensity - sector_intensity_at_t) / denominator
        )

        result = sector_intensity_at_t + (
            delta_company_sector * convergence_factor
        )

        # Ensure non-negative intensity
        return max(result, Decimal("0"))

    def get_sector_intensity_at_year(
        self,
        sector: str,
        year: int,
    ) -> Decimal:
        """Get the sector benchmark intensity at a specific year.

        Interpolates linearly between published milestone years
        from IEA NZE 2023 data.

        Args:
            sector: SDA sector identifier.
            year: Calendar year.

        Returns:
            Interpolated sector intensity at the given year.
        """
        benchmark = self._get_sector_benchmark(sector)
        if benchmark is None:
            return Decimal("0")
        return self._interpolate_sector_intensity(benchmark, year)

    def get_available_sectors(self) -> List[Dict[str, str]]:
        """Return list of all SDA-eligible sectors with descriptions.

        Returns:
            List of dicts with sector, description, unit, and source.
        """
        result: List[Dict[str, str]] = []
        for sector_enum in SdaSector:
            bm = SECTOR_BENCHMARKS.get(sector_enum.value, {})
            result.append({
                "sector": sector_enum.value,
                "description": bm.get("description", ""),
                "unit": bm.get("unit", ""),
                "source": "IEA Net Zero Emissions by 2050 (NZE 2023)",
            })
        return result

    def get_sector_pathway(
        self,
        sector: str,
        start_year: int = 2020,
        end_year: int = 2050,
    ) -> List[Dict[str, Any]]:
        """Get the full sector benchmark pathway for a given sector.

        Args:
            sector: SDA sector identifier.
            start_year: Start year for pathway.
            end_year: End year for pathway.

        Returns:
            List of dicts with year and sector intensity.
        """
        benchmark = self._get_sector_benchmark(sector)
        if benchmark is None:
            return []

        pathway: List[Dict[str, Any]] = []
        for yr in range(start_year, end_year + 1):
            intensity = self._interpolate_sector_intensity(benchmark, yr)
            pathway.append({
                "year": yr,
                "sector_intensity": str(_round_val(intensity, 8)),
                "unit": benchmark.get("unit", ""),
            })
        return pathway

    def compare_company_to_sector(
        self,
        company_intensity: Decimal,
        sector: str,
        year: int,
    ) -> Dict[str, Any]:
        """Compare company intensity to sector benchmark at a given year.

        Args:
            company_intensity: Company's current intensity.
            sector: SDA sector identifier.
            year: Reference year.

        Returns:
            Dict with comparison metrics.
        """
        sector_intensity = self.get_sector_intensity_at_year(sector, year)
        gap = company_intensity - sector_intensity
        gap_pct = _safe_pct(gap, sector_intensity) if sector_intensity > Decimal("0") else Decimal("0")

        return {
            "company_intensity": str(_round_val(company_intensity, 8)),
            "sector_intensity": str(_round_val(sector_intensity, 8)),
            "gap": str(_round_val(gap, 8)),
            "gap_pct": str(_round_val(gap_pct, 2)),
            "is_above_sector": gap > Decimal("0"),
            "is_aligned": gap <= Decimal("0"),
            "year": year,
            "sector": sector,
        }

    def estimate_target_year_for_alignment(
        self,
        company_base_intensity: Decimal,
        sector: str,
        base_year: int,
        max_year: int = 2050,
    ) -> int:
        """Estimate the earliest year the company aligns with sector pathway.

        Uses the SDA convergence formula to find the first year where
        the company intensity meets or falls below the sector pathway.

        Args:
            company_base_intensity: Company intensity at base year.
            sector: SDA sector identifier.
            base_year: Base year.
            max_year: Maximum year to search through.

        Returns:
            Year of alignment, or 0 if not achieved by max_year.
        """
        benchmark = self._get_sector_benchmark(sector)
        if benchmark is None:
            return 0

        sector_base = self._interpolate_sector_intensity(benchmark, base_year)
        sector_target = self._interpolate_sector_intensity(benchmark, max_year)

        for yr in range(base_year + 1, max_year + 1):
            sector_at_t = self._interpolate_sector_intensity(benchmark, yr)
            company_at_t = self.calculate_converged_intensity(
                company_base_intensity, sector_base,
                sector_target, sector_at_t
            )
            if company_at_t <= sector_at_t:
                return yr

        return 0

    # ------------------------------------------------------------------ #
    # Internal: Benchmark Retrieval                                       #
    # ------------------------------------------------------------------ #

    def _get_sector_benchmark(
        self, sector: str,
    ) -> Optional[Dict[str, Any]]:
        """Retrieve sector benchmark data.

        Checks custom benchmarks first, then falls back to IEA NZE data.

        Args:
            sector: SDA sector identifier.

        Returns:
            Benchmark dict or None if not found.
        """
        if sector in self._custom_benchmarks:
            return self._custom_benchmarks[sector]
        return SECTOR_BENCHMARKS.get(sector)

    def _interpolate_sector_intensity(
        self,
        benchmark: Dict[str, Any],
        year: int,
    ) -> Decimal:
        """Interpolate sector intensity at a given year.

        Uses linear interpolation between published milestone years.

        Args:
            benchmark: Sector benchmark data.
            year: Year to interpolate for.

        Returns:
            Sector intensity at the given year.
        """
        milestones = benchmark.get("milestones", {})
        if not milestones:
            return Decimal("0")

        # Convert milestone keys to ints and values to Decimal
        ms_years = sorted(int(y) for y in milestones.keys())
        ms_vals = {int(y): _decimal(v) for y, v in milestones.items()}

        if year <= ms_years[0]:
            return ms_vals[ms_years[0]]
        if year >= ms_years[-1]:
            return ms_vals[ms_years[-1]]

        # Find bracketing years
        for i in range(len(ms_years) - 1):
            y_lo = ms_years[i]
            y_hi = ms_years[i + 1]
            if y_lo <= year <= y_hi:
                return _round_val(
                    _interpolate_linear(
                        ms_vals[y_lo], ms_vals[y_hi],
                        year, y_lo, y_hi
                    ),
                    10,
                )

        return Decimal("0")

    # ------------------------------------------------------------------ #
    # Internal: Input Validation                                          #
    # ------------------------------------------------------------------ #

    def _validate_inputs(
        self,
        company: CompanyIntensityInput,
        benchmark: Dict[str, Any],
    ) -> List[str]:
        """Validate company inputs against sector benchmark.

        Args:
            company: Company input data.
            benchmark: Sector benchmark data.

        Returns:
            List of warning messages.
        """
        warnings: List[str] = []

        # Check target duration
        duration = company.target_year - company.base_year
        if duration < MIN_TARGET_DURATION_YEARS:
            warnings.append(
                f"Target duration of {duration} years is below the "
                f"SBTi minimum of {MIN_TARGET_DURATION_YEARS} years."
            )
        if duration > MAX_NEAR_TERM_DURATION_YEARS and company.target_year <= 2035:
            warnings.append(
                f"Near-term target duration of {duration} years exceeds "
                f"the recommended maximum of {MAX_NEAR_TERM_DURATION_YEARS} years."
            )

        # Check base year validity
        if company.base_year < MIN_BASE_YEAR:
            warnings.append(
                f"Base year {company.base_year} is before the SBTi "
                f"minimum of {MIN_BASE_YEAR}."
            )

        # Check intensity is reasonable vs sector
        sector_base = self._interpolate_sector_intensity(
            benchmark, company.base_year
        )
        if sector_base > Decimal("0"):
            ratio = company.base_year_intensity / sector_base
            if ratio > Decimal("5"):
                warnings.append(
                    f"Company base intensity is {_round_val(ratio, 1)}x "
                    f"the sector average. Verify data accuracy."
                )
            if ratio < Decimal("0.1"):
                warnings.append(
                    f"Company base intensity is only "
                    f"{_round_val(ratio * Decimal('100'), 1)}% of "
                    f"the sector average. Verify data accuracy."
                )

        # Check current intensity progression
        if (
            company.current_intensity > Decimal("0")
            and company.current_year > company.base_year
            and company.current_intensity > company.base_year_intensity
        ):
            warnings.append(
                "Current intensity is higher than base year intensity. "
                "Emissions have increased since the base year."
            )

        return warnings

    # ------------------------------------------------------------------ #
    # Internal: Convergence Calculation                                    #
    # ------------------------------------------------------------------ #

    def _calculate_convergence(
        self,
        company: CompanyIntensityInput,
        benchmark: Dict[str, Any],
    ) -> ConvergenceAssessment:
        """Calculate SDA convergence assessment.

        Args:
            company: Company input data.
            benchmark: Sector benchmark data.

        Returns:
            ConvergenceAssessment with full convergence analysis.
        """
        sector_base = self._interpolate_sector_intensity(
            benchmark, company.base_year
        )
        sector_target = self._interpolate_sector_intensity(
            benchmark, company.target_year
        )

        # Calculate company target intensity using SDA formula
        company_target = self.calculate_converged_intensity(
            company.base_year_intensity,
            sector_base,
            sector_target,
            sector_target,  # At target year, sector_at_t = sector_target
        )

        # Total reduction percentage
        total_reduction_pct = Decimal("0")
        if company.base_year_intensity > Decimal("0"):
            total_reduction_pct = _round_val(
                (Decimal("1") - _safe_divide(
                    company_target, company.base_year_intensity
                )) * Decimal("100"),
                2,
            )

        # Average annual reduction rate
        years = company.target_year - company.base_year
        avg_annual_rate = Decimal("0")
        if years > 0 and company.base_year_intensity > Decimal("0"):
            # Compound annual reduction rate
            ratio = _safe_divide(
                company_target, company.base_year_intensity,
                default=Decimal("1"),
            )
            if ratio > Decimal("0"):
                # ARR = 1 - (I_target / I_base)^(1/years)
                # Using logarithmic approach for precision
                import math
                ratio_float = float(ratio)
                if ratio_float > 0:
                    annual_factor = ratio_float ** (1.0 / years)
                    avg_annual_rate = _round_val(
                        (Decimal("1") - _decimal(annual_factor)) * Decimal("100"),
                        4,
                    )

        # Determine convergence status
        min_rate = _decimal(benchmark.get("min_annual_reduction_rate", "0.025"))
        min_rate_pct = min_rate * Decimal("100")

        if company_target <= sector_target:
            if company.base_year_intensity <= sector_base:
                status = ConvergenceStatus.ALIGNED.value
            else:
                status = ConvergenceStatus.CONVERGING.value
        elif avg_annual_rate < min_rate_pct:
            status = ConvergenceStatus.INSUFFICIENT.value
        else:
            status = ConvergenceStatus.ABOVE_PATHWAY.value

        # Intensity gaps
        gap_base = _round_val(
            company.base_year_intensity - sector_base, 8
        )
        gap_target = _round_val(company_target - sector_target, 8)

        # Convergence year
        conv_year = 0
        if gap_base > Decimal("0"):
            conv_year = self.estimate_target_year_for_alignment(
                company.base_year_intensity, company.sector,
                company.base_year, company.target_year
            )

        # Build message
        unit = benchmark.get("unit", "")
        if status == ConvergenceStatus.ALIGNED.value:
            msg = (
                f"Company is already at or below the sector pathway. "
                f"Base intensity: {_round_val(company.base_year_intensity, 6)} "
                f"{unit} vs sector {_round_val(sector_base, 6)} {unit}. "
                f"Target intensity: {_round_val(company_target, 6)} {unit}."
            )
        elif status == ConvergenceStatus.CONVERGING.value:
            msg = (
                f"Company pathway converges to sector benchmark by "
                f"{company.target_year}. Reduction of "
                f"{total_reduction_pct}% ({avg_annual_rate}% annually) "
                f"from {_round_val(company.base_year_intensity, 6)} to "
                f"{_round_val(company_target, 6)} {unit}."
            )
        elif status == ConvergenceStatus.INSUFFICIENT.value:
            msg = (
                f"Annual reduction rate of {avg_annual_rate}% is below "
                f"the sector minimum of {min_rate_pct}%. "
                f"Target ambition is insufficient for SBTi validation."
            )
        else:
            msg = (
                f"Company target intensity ({_round_val(company_target, 6)} "
                f"{unit}) exceeds sector benchmark "
                f"({_round_val(sector_target, 6)} {unit}) at "
                f"{company.target_year}. Convergence not achieved."
            )

        return ConvergenceAssessment(
            sector=company.sector,
            sector_description=benchmark.get("description", ""),
            intensity_unit=unit,
            base_year=company.base_year,
            target_year=company.target_year,
            company_base_intensity=_round_val(company.base_year_intensity, 8),
            company_target_intensity=_round_val(company_target, 8),
            sector_base_intensity=_round_val(sector_base, 8),
            sector_target_intensity=_round_val(sector_target, 8),
            total_reduction_pct=total_reduction_pct,
            avg_annual_reduction_rate_pct=avg_annual_rate,
            convergence_status=status,
            years_to_target=years,
            intensity_gap_at_base=gap_base,
            intensity_gap_at_target=gap_target,
            convergence_year=conv_year,
            message=msg,
        )

    # ------------------------------------------------------------------ #
    # Internal: Annual Milestones                                          #
    # ------------------------------------------------------------------ #

    def _generate_annual_milestones(
        self,
        company: CompanyIntensityInput,
        benchmark: Dict[str, Any],
        convergence: ConvergenceAssessment,
    ) -> List[AnnualMilestone]:
        """Generate year-by-year intensity milestones.

        For each year from base to target, calculates the sector
        benchmark intensity and company converged intensity using
        the SDA formula.

        Args:
            company: Company input data.
            benchmark: Sector benchmark data.
            convergence: Convergence assessment.

        Returns:
            List of AnnualMilestone objects.
        """
        milestones: List[AnnualMilestone] = []
        sector_base = convergence.sector_base_intensity
        sector_target = convergence.sector_target_intensity
        prev_company = company.base_year_intensity

        for yr in range(company.base_year, company.target_year + 1):
            sector_at_t = self._interpolate_sector_intensity(benchmark, yr)

            company_at_t = self.calculate_converged_intensity(
                company.base_year_intensity,
                sector_base,
                sector_target,
                sector_at_t,
            )

            # Cumulative reduction from base
            cum_reduction = Decimal("0")
            if company.base_year_intensity > Decimal("0"):
                cum_reduction = _round_val(
                    (Decimal("1") - _safe_divide(
                        company_at_t, company.base_year_intensity
                    )) * Decimal("100"),
                    2,
                )

            # Annual reduction rate
            arr = Decimal("0")
            if prev_company > Decimal("0") and yr > company.base_year:
                arr = _round_val(
                    (Decimal("1") - _safe_divide(
                        company_at_t, prev_company
                    )) * Decimal("100"),
                    4,
                )

            # Production volume estimate
            prod_vol = Decimal("0")
            if company.base_year_production > Decimal("0"):
                years_from_base = yr - company.base_year
                growth_rate = Decimal("1") + (
                    company.projected_production_growth_pct / Decimal("100")
                )
                prod_vol = _round_val(
                    company.base_year_production * (growth_rate ** years_from_base),
                    2,
                )

            # Absolute emissions
            abs_emissions = Decimal("0")
            if prod_vol > Decimal("0"):
                abs_emissions = _round_val(company_at_t * prod_vol, 2)

            milestones.append(AnnualMilestone(
                year=yr,
                sector_intensity=_round_val(sector_at_t, 8),
                company_intensity=_round_val(company_at_t, 8),
                intensity_reduction_from_base_pct=cum_reduction,
                annual_reduction_rate_pct=arr,
                is_on_track=company_at_t <= sector_at_t,
                absolute_emissions_tco2e=abs_emissions,
                production_volume=prod_vol,
            ))

            prev_company = company_at_t

        return milestones

    # ------------------------------------------------------------------ #
    # Internal: Ambition Assessment                                        #
    # ------------------------------------------------------------------ #

    def _assess_ambition(
        self,
        convergence: ConvergenceAssessment,
        benchmark: Dict[str, Any],
        company: CompanyIntensityInput,
    ) -> AmbitionAssessment:
        """Assess target ambition level.

        Compares the company's average annual reduction rate against
        SBTi thresholds for well-below 2C and 1.5C alignment.

        Args:
            convergence: Convergence assessment result.
            benchmark: Sector benchmark data.
            company: Company input data.

        Returns:
            AmbitionAssessment with classification and gaps.
        """
        company_rate = convergence.avg_annual_reduction_rate_pct

        min_rate = _decimal(benchmark.get("min_annual_reduction_rate", "0.025"))
        wb2c_rate = _decimal(benchmark.get("wb2c_annual_rate", "0.042"))
        c15_rate = _decimal(benchmark.get("c15_annual_rate", "0.070"))

        min_rate_pct = _round_val(min_rate * Decimal("100"), 2)
        wb2c_pct = _round_val(wb2c_rate * Decimal("100"), 2)
        c15_pct = _round_val(c15_rate * Decimal("100"), 2)

        exceeds_c15 = company_rate >= c15_pct
        exceeds_wb2c = company_rate >= wb2c_pct
        exceeds_min = company_rate >= min_rate_pct

        # Determine ambition level
        if exceeds_c15:
            level = AmbitionLevel.NET_ZERO.value
        elif exceeds_wb2c:
            level = AmbitionLevel.C_1_5.value
        elif exceeds_min:
            level = AmbitionLevel.WELL_BELOW_2C.value
        else:
            level = AmbitionLevel.BELOW_MINIMUM.value

        gap_min = _round_val(
            max(Decimal("0"), min_rate_pct - company_rate), 2
        )
        gap_wb2c = _round_val(
            max(Decimal("0"), wb2c_pct - company_rate), 2
        )
        gap_c15 = _round_val(
            max(Decimal("0"), c15_pct - company_rate), 2
        )

        if level == AmbitionLevel.NET_ZERO.value:
            msg = (
                f"Target is net-zero aligned with {company_rate}% "
                f"annual reduction, exceeding the 1.5C threshold "
                f"of {c15_pct}% for {company.sector}."
            )
        elif level == AmbitionLevel.C_1_5.value:
            msg = (
                f"Target is 1.5C-aligned with {company_rate}% "
                f"annual reduction, meeting the WB2C threshold "
                f"of {wb2c_pct}% for {company.sector}. "
                f"Gap to 1.5C: {gap_c15}%."
            )
        elif level == AmbitionLevel.WELL_BELOW_2C.value:
            msg = (
                f"Target meets SBTi minimum ({min_rate_pct}%) with "
                f"{company_rate}% annual reduction for {company.sector}. "
                f"Gap to WB2C: {gap_wb2c}%."
            )
        else:
            msg = (
                f"Target is below SBTi minimum of {min_rate_pct}% "
                f"annual reduction for {company.sector}. "
                f"Current rate: {company_rate}%. Gap: {gap_min}%."
            )

        return AmbitionAssessment(
            ambition_level=level,
            company_annual_rate_pct=company_rate,
            wb2c_threshold_pct=wb2c_pct,
            c15_threshold_pct=c15_pct,
            sector_minimum_pct=min_rate_pct,
            exceeds_wb2c=exceeds_wb2c,
            exceeds_c15=exceeds_c15,
            exceeds_minimum=exceeds_min,
            gap_to_minimum_pct=gap_min,
            gap_to_wb2c_pct=gap_wb2c,
            gap_to_c15_pct=gap_c15,
            message=msg,
        )

    # ------------------------------------------------------------------ #
    # Internal: Cross-Validation                                           #
    # ------------------------------------------------------------------ #

    def _cross_validate(
        self,
        company: CompanyIntensityInput,
        benchmark: Dict[str, Any],
        cv_points: List[CrossValidationInput],
    ) -> CrossValidationSummary:
        """Cross-validate against SBTi SDA Tool V3.0 reference data.

        For each provided reference point, calculates the engine's
        intensity and compares against the SDA Tool value.  A point
        passes if the relative deviation is within 2% tolerance.

        Args:
            company: Company input data.
            benchmark: Sector benchmark data.
            cv_points: SDA Tool reference data.

        Returns:
            CrossValidationSummary with per-year details.
        """
        sector_base = self._interpolate_sector_intensity(
            benchmark, company.base_year
        )
        sector_target = self._interpolate_sector_intensity(
            benchmark, company.target_year
        )

        details: List[CrossValidationResult] = []
        total_deviation = Decimal("0")
        max_dev = Decimal("0")
        passed = 0
        marginal = 0
        failed = 0

        for cvp in cv_points:
            sector_at_t = self._interpolate_sector_intensity(
                benchmark, cvp.year
            )
            calc_intensity = self.calculate_converged_intensity(
                company.base_year_intensity,
                sector_base,
                sector_target,
                sector_at_t,
            )

            abs_dev = abs(calc_intensity - cvp.sda_tool_intensity)
            rel_dev = Decimal("0")
            if cvp.sda_tool_intensity > Decimal("0"):
                rel_dev = _round_val(
                    abs_dev / cvp.sda_tool_intensity * Decimal("100"), 4
                )

            rel_frac = _safe_divide(abs_dev, cvp.sda_tool_intensity)

            if rel_frac <= self._cv_tolerance:
                status = ValidationStatus.PASS_VALID.value
                passed += 1
            elif rel_frac <= self._cv_marginal:
                status = ValidationStatus.MARGINAL.value
                marginal += 1
            else:
                status = ValidationStatus.FAIL.value
                failed += 1

            total_deviation += rel_dev
            if rel_dev > max_dev:
                max_dev = rel_dev

            details.append(CrossValidationResult(
                year=cvp.year,
                calculated_intensity=_round_val(calc_intensity, 8),
                sda_tool_intensity=_round_val(cvp.sda_tool_intensity, 8),
                absolute_deviation=_round_val(abs_dev, 8),
                relative_deviation_pct=rel_dev,
                validation_status=status,
                within_tolerance=rel_frac <= self._cv_tolerance,
            ))

        total = len(cv_points)
        avg_dev = _safe_divide(total_deviation, _decimal(total)) if total > 0 else Decimal("0")

        # Overall status
        if failed > 0:
            overall = ValidationStatus.FAIL.value
        elif marginal > 0:
            overall = ValidationStatus.MARGINAL.value
        elif total > 0:
            overall = ValidationStatus.PASS_VALID.value
        else:
            overall = ValidationStatus.NOT_VALIDATED.value

        if overall == ValidationStatus.PASS_VALID.value:
            msg = (
                f"Cross-validation PASSED: {passed}/{total} points "
                f"within {self._cv_tolerance * 100}% tolerance. "
                f"Max deviation: {_round_val(max_dev, 2)}%."
            )
        elif overall == ValidationStatus.MARGINAL.value:
            msg = (
                f"Cross-validation MARGINAL: {marginal}/{total} points "
                f"between {self._cv_tolerance * 100}% and "
                f"{self._cv_marginal * 100}% deviation. "
                f"Max deviation: {_round_val(max_dev, 2)}%."
            )
        elif overall == ValidationStatus.FAIL.value:
            msg = (
                f"Cross-validation FAILED: {failed}/{total} points "
                f"exceed {self._cv_marginal * 100}% tolerance. "
                f"Max deviation: {_round_val(max_dev, 2)}%. "
                f"Review calculation methodology."
            )
        else:
            msg = "No cross-validation data provided."

        return CrossValidationSummary(
            total_points=total,
            points_passed=passed,
            points_marginal=marginal,
            points_failed=failed,
            max_deviation_pct=_round_val(max_dev, 4),
            avg_deviation_pct=_round_val(avg_dev, 4),
            overall_status=overall,
            details=details,
            message=msg,
        )

    # ------------------------------------------------------------------ #
    # Internal: Absolute Pathway                                           #
    # ------------------------------------------------------------------ #

    def _generate_absolute_pathway(
        self,
        company: CompanyIntensityInput,
        milestones: List[AnnualMilestone],
        prod_forecast: Optional[ProductionForecast],
    ) -> List[AbsolutePathwayPoint]:
        """Generate absolute emissions pathway from intensity milestones.

        Combines intensity milestones with production forecast to
        produce absolute emissions at each year.

        Args:
            company: Company input data.
            milestones: Annual intensity milestones.
            prod_forecast: Production volume forecast.

        Returns:
            List of AbsolutePathwayPoint objects.
        """
        if not milestones:
            return []

        # Build production lookup from forecast
        prod_lookup: Dict[int, Decimal] = {}
        if prod_forecast and prod_forecast.projected_years:
            for entry in prod_forecast.projected_years:
                yr = entry.get("year", 0)
                vol = _decimal(entry.get("production_volume", "0"))
                prod_lookup[yr] = vol

        # Fall back to milestone production volumes
        for ms in milestones:
            if ms.year not in prod_lookup and ms.production_volume > Decimal("0"):
                prod_lookup[ms.year] = ms.production_volume

        base_abs = company.absolute_emissions_base_tco2e
        pathway: List[AbsolutePathwayPoint] = []

        for ms in milestones:
            prod_vol = prod_lookup.get(ms.year, ms.production_volume)
            abs_em = Decimal("0")
            if prod_vol > Decimal("0"):
                abs_em = _round_val(ms.company_intensity * prod_vol, 2)

            reduction_pct = Decimal("0")
            if base_abs > Decimal("0"):
                reduction_pct = _round_val(
                    (Decimal("1") - _safe_divide(abs_em, base_abs))
                    * Decimal("100"),
                    2,
                )

            pathway.append(AbsolutePathwayPoint(
                year=ms.year,
                intensity=ms.company_intensity,
                production_volume=prod_vol,
                absolute_emissions_tco2e=abs_em,
                reduction_from_base_pct=reduction_pct,
            ))

        return pathway

    # ------------------------------------------------------------------ #
    # Internal: Production Forecast                                        #
    # ------------------------------------------------------------------ #

    def _generate_production_forecast(
        self,
        company: CompanyIntensityInput,
    ) -> ProductionForecast:
        """Generate production volume forecast.

        Projects production using compound annual growth rate
        from base year to target year.

from greenlang.schemas import utcnow

        Args:
            company: Company input data.

        Returns:
            ProductionForecast with year-by-year projections.
        """
        if company.base_year_production <= Decimal("0"):
            return ProductionForecast(
                base_year=company.base_year,
                base_production=Decimal("0"),
                annual_growth_pct=company.projected_production_growth_pct,
                projected_years=[],
            )

        growth_rate = Decimal("1") + (
            company.projected_production_growth_pct / Decimal("100")
        )

        projections: List[Dict[str, Any]] = []
        for yr in range(company.base_year, company.target_year + 1):
            years_from_base = yr - company.base_year
            volume = _round_val(
                company.base_year_production * (growth_rate ** years_from_base),
                2,
            )
            projections.append({
                "year": yr,
                "production_volume": str(volume),
                "years_from_base": years_from_base,
            })

        return ProductionForecast(
            base_year=company.base_year,
            base_production=company.base_year_production,
            annual_growth_pct=company.projected_production_growth_pct,
            projected_years=projections,
        )

    # ------------------------------------------------------------------ #
    # Internal: Sector Benchmark Info                                      #
    # ------------------------------------------------------------------ #

    def _build_sector_benchmark_info(
        self,
        sector: str,
        benchmark: Dict[str, Any],
    ) -> SectorBenchmarkInfo:
        """Build sector benchmark reference information.

        Args:
            sector: Sector identifier.
            benchmark: Raw benchmark data.

        Returns:
            SectorBenchmarkInfo model.
        """
        milestones_raw = benchmark.get("milestones", {})
        milestones_out: Dict[str, Decimal] = {}
        for yr, val in milestones_raw.items():
            milestones_out[str(yr)] = _round_val(_decimal(val), 8)

        base_intensity = _decimal(
            milestones_raw.get(
                benchmark.get("base_year", 2020), "0"
            )
        )
        target_intensity = _decimal(milestones_raw.get(2050, "0"))

        return SectorBenchmarkInfo(
            sector=sector,
            description=benchmark.get("description", ""),
            unit=benchmark.get("unit", ""),
            base_year_intensity=_round_val(base_intensity, 8),
            target_year_intensity=_round_val(target_intensity, 8),
            milestones=milestones_out,
            min_annual_rate_pct=_round_val(
                _decimal(benchmark.get("min_annual_reduction_rate", "0")) * Decimal("100"), 2
            ),
            wb2c_annual_rate_pct=_round_val(
                _decimal(benchmark.get("wb2c_annual_rate", "0")) * Decimal("100"), 2
            ),
            c15_annual_rate_pct=_round_val(
                _decimal(benchmark.get("c15_annual_rate", "0")) * Decimal("100"), 2
            ),
            source="IEA Net Zero Emissions by 2050 (NZE 2023)",
        )

    # ------------------------------------------------------------------ #
    # Internal: Recommendations                                            #
    # ------------------------------------------------------------------ #

    def _generate_recommendations(
        self,
        convergence: ConvergenceAssessment,
        ambition: Optional[AmbitionAssessment],
        cross_val: Optional[CrossValidationSummary],
        company: CompanyIntensityInput,
        benchmark: Dict[str, Any],
    ) -> List[SdaRecommendation]:
        """Generate SDA-specific recommendations.

        Produces recommendations based on convergence status, ambition
        level, cross-validation results, and sector characteristics.

        Args:
            convergence: Convergence assessment.
            ambition: Ambition assessment (optional).
            cross_val: Cross-validation summary (optional).
            company: Company input data.
            benchmark: Sector benchmark data.

        Returns:
            List of SdaRecommendation sorted by priority.
        """
        recs: List[SdaRecommendation] = []

        # R1: Convergence not achieved
        if convergence.convergence_status == ConvergenceStatus.ABOVE_PATHWAY.value:
            recs.append(SdaRecommendation(
                priority="immediate",
                category="convergence",
                action=(
                    f"Increase target ambition to achieve convergence "
                    f"with the {convergence.sector_description} sector "
                    f"benchmark by {convergence.target_year}. "
                    f"Current gap at target: "
                    f"{convergence.intensity_gap_at_target} "
                    f"{convergence.intensity_unit}."
                ),
                rationale=(
                    "SBTi SDA methodology requires company intensity "
                    "to converge to the sector benchmark at the target year."
                ),
                estimated_impact=(
                    f"Close {convergence.intensity_gap_at_target} "
                    f"{convergence.intensity_unit} gap"
                ),
                timeline_months=6,
            ))

        # R2: Insufficient ambition
        if convergence.convergence_status == ConvergenceStatus.INSUFFICIENT.value:
            min_rate = _decimal(benchmark.get("min_annual_reduction_rate", "0.025"))
            recs.append(SdaRecommendation(
                priority="immediate",
                category="ambition",
                action=(
                    f"Increase annual reduction rate from "
                    f"{convergence.avg_annual_reduction_rate_pct}% to at "
                    f"least {_round_val(min_rate * Decimal('100'), 2)}% "
                    f"to meet SBTi sector minimum for "
                    f"{convergence.sector_description}."
                ),
                rationale=(
                    f"SBTi requires a minimum annual reduction rate of "
                    f"{_round_val(min_rate * Decimal('100'), 2)}% for the "
                    f"{convergence.sector_description} sector."
                ),
                estimated_impact="Meets SBTi minimum ambition threshold",
                timeline_months=3,
            ))

        # R3: Upgrade to 1.5C alignment
        if ambition and ambition.ambition_level == AmbitionLevel.WELL_BELOW_2C.value:
            recs.append(SdaRecommendation(
                priority="short_term",
                category="ambition",
                action=(
                    f"Consider increasing target ambition to 1.5C alignment. "
                    f"Current rate: {ambition.company_annual_rate_pct}%, "
                    f"1.5C threshold: {ambition.c15_threshold_pct}%. "
                    f"Gap: {ambition.gap_to_c15_pct}%."
                ),
                rationale=(
                    "1.5C-aligned targets demonstrate higher climate "
                    "ambition and are increasingly expected by investors "
                    "and regulators."
                ),
                estimated_impact="1.5C-aligned SBTi target classification",
                timeline_months=6,
            ))

        # R4: Below minimum ambition
        if ambition and ambition.ambition_level == AmbitionLevel.BELOW_MINIMUM.value:
            recs.append(SdaRecommendation(
                priority="immediate",
                category="ambition",
                action=(
                    f"Increase target ambition to meet SBTi minimum. "
                    f"Current rate: {ambition.company_annual_rate_pct}%, "
                    f"minimum: {ambition.sector_minimum_pct}%. "
                    f"Gap: {ambition.gap_to_minimum_pct}%."
                ),
                rationale=(
                    "Target does not meet SBTi minimum requirements "
                    "and will not be validated."
                ),
                estimated_impact="Meets SBTi validation requirements",
                timeline_months=3,
            ))

        # R5: Cross-validation issues
        if cross_val and cross_val.overall_status == ValidationStatus.FAIL.value:
            recs.append(SdaRecommendation(
                priority="immediate",
                category="validation",
                action=(
                    f"Resolve cross-validation discrepancies with SBTi "
                    f"SDA Tool V3.0. {cross_val.points_failed} of "
                    f"{cross_val.total_points} reference points exceed "
                    f"tolerance. Max deviation: "
                    f"{cross_val.max_deviation_pct}%."
                ),
                rationale=(
                    "SBTi target validation requires alignment with "
                    "the SDA Tool V3.0 output within 2% tolerance."
                ),
                estimated_impact="Consistent SDA Tool alignment",
                timeline_months=1,
            ))

        if cross_val and cross_val.overall_status == ValidationStatus.MARGINAL.value:
            recs.append(SdaRecommendation(
                priority="short_term",
                category="validation",
                action=(
                    f"Review marginal cross-validation results. "
                    f"{cross_val.points_marginal} points are between "
                    f"2% and 5% deviation. Verify input data precision."
                ),
                rationale=(
                    "Marginal deviations may indicate input data rounding "
                    "differences. Verify alignment before submission."
                ),
                estimated_impact="Improved data consistency",
                timeline_months=1,
            ))

        # R6: Data quality recommendations
        if company.base_year_production <= Decimal("0"):
            recs.append(SdaRecommendation(
                priority="short_term",
                category="data_quality",
                action=(
                    "Provide base year production volume data to enable "
                    "absolute emissions pathway calculation."
                ),
                rationale=(
                    "Production data is needed to translate intensity "
                    "targets into absolute emissions pathways for "
                    "comprehensive SBTi reporting."
                ),
                estimated_impact="Enables absolute pathway analysis",
                timeline_months=3,
            ))

        if company.current_intensity <= Decimal("0"):
            recs.append(SdaRecommendation(
                priority="short_term",
                category="data_quality",
                action=(
                    "Provide current year intensity data to enable "
                    "progress tracking against the SDA pathway."
                ),
                rationale=(
                    "Current intensity data enables assessment of "
                    "year-over-year progress towards the target."
                ),
                estimated_impact="Enables progress tracking",
                timeline_months=1,
            ))

        # R7: Target duration
        duration = company.target_year - company.base_year
        if duration < MIN_TARGET_DURATION_YEARS:
            recs.append(SdaRecommendation(
                priority="immediate",
                category="target_design",
                action=(
                    f"Extend target duration from {duration} years to "
                    f"at least {MIN_TARGET_DURATION_YEARS} years "
                    f"(SBTi minimum requirement)."
                ),
                rationale=(
                    "SBTi Corporate Manual V5.3 requires a minimum "
                    f"target duration of {MIN_TARGET_DURATION_YEARS} years."
                ),
                estimated_impact="Meets SBTi duration requirement",
                timeline_months=1,
            ))

        # R8: Consider net-zero pathway
        if (
            ambition
            and ambition.ambition_level in (
                AmbitionLevel.WELL_BELOW_2C.value,
                AmbitionLevel.C_1_5.value,
            )
            and company.target_year < 2050
        ):
            recs.append(SdaRecommendation(
                priority="medium_term",
                category="target_design",
                action=(
                    "Develop a long-term net-zero target to 2050 to "
                    "complement the near-term SDA target."
                ),
                rationale=(
                    "SBTi Net-Zero Standard encourages complementing "
                    "near-term SDA targets with long-term net-zero "
                    "commitments covering all material scopes."
                ),
                estimated_impact="Complete SBTi net-zero alignment",
                timeline_months=12,
            ))

        # R9: Regular review
        recs.append(SdaRecommendation(
            priority="medium_term",
            category="governance",
            action=(
                "Establish annual review of SDA pathway progress, "
                "intensity metrics, and production volume data."
            ),
            rationale=(
                "SBTi requires regular progress reporting and target "
                "recalculation triggers (e.g. acquisitions, methodology "
                "changes)."
            ),
            estimated_impact="Ongoing SBTi compliance and reporting",
            timeline_months=3,
        ))

        # Sort by priority
        priority_order = {
            "immediate": 0,
            "short_term": 1,
            "medium_term": 2,
            "long_term": 3,
        }
        recs.sort(key=lambda r: priority_order.get(r.priority, 4))

        return recs

    # ------------------------------------------------------------------ #
    # Utility Methods                                                      #
    # ------------------------------------------------------------------ #

    def get_summary(self, result: SdaResult) -> Dict[str, Any]:
        """Generate concise summary from SDA result.

        Args:
            result: SDA analysis result to summarise.

        Returns:
            Dict with key metrics and provenance hash.
        """
        summary: Dict[str, Any] = {
            "company_name": result.company_name,
            "sector": result.sector,
            "engine_version": result.engine_version,
        }

        if result.convergence:
            summary["convergence_status"] = result.convergence.convergence_status
            summary["total_reduction_pct"] = str(
                result.convergence.total_reduction_pct
            )
            summary["avg_annual_rate_pct"] = str(
                result.convergence.avg_annual_reduction_rate_pct
            )
            summary["base_year"] = result.convergence.base_year
            summary["target_year"] = result.convergence.target_year
            summary["company_base_intensity"] = str(
                result.convergence.company_base_intensity
            )
            summary["company_target_intensity"] = str(
                result.convergence.company_target_intensity
            )

        if result.ambition:
            summary["ambition_level"] = result.ambition.ambition_level
            summary["exceeds_minimum"] = result.ambition.exceeds_minimum

        if result.cross_validation:
            summary["cross_validation_status"] = (
                result.cross_validation.overall_status
            )

        summary["milestones_count"] = len(result.annual_milestones)
        summary["recommendations_count"] = len(result.recommendations)
        summary["warnings_count"] = len(result.warnings)
        summary["provenance_hash"] = _compute_hash(summary)
        return summary

    def get_sector_comparison_table(self) -> List[Dict[str, Any]]:
        """Return comparison table of all 12 SDA sectors.

        Returns:
            List of dicts with sector metrics for comparison.
        """
        table: List[Dict[str, Any]] = []
        for sector_enum in SdaSector:
            bm = SECTOR_BENCHMARKS.get(sector_enum.value, {})
            milestones = bm.get("milestones", {})
            base_val = _decimal(milestones.get(2020, "0"))
            target_val = _decimal(milestones.get(2050, "0"))

            total_reduction = Decimal("0")
            if base_val > Decimal("0"):
                total_reduction = _round_val(
                    (Decimal("1") - _safe_divide(target_val, base_val))
                    * Decimal("100"),
                    1,
                )

            table.append({
                "sector": sector_enum.value,
                "description": bm.get("description", ""),
                "unit": bm.get("unit", ""),
                "base_2020_intensity": str(_round_val(base_val, 6)),
                "target_2050_intensity": str(_round_val(target_val, 6)),
                "total_reduction_pct": str(total_reduction),
                "min_annual_rate_pct": str(
                    _round_val(
                        _decimal(bm.get("min_annual_reduction_rate", "0"))
                        * Decimal("100"), 2
                    )
                ),
                "c15_annual_rate_pct": str(
                    _round_val(
                        _decimal(bm.get("c15_annual_rate", "0"))
                        * Decimal("100"), 2
                    )
                ),
                "source": "IEA NZE 2023",
            })

        return table

    def calculate_required_reduction_for_alignment(
        self,
        company_base_intensity: Decimal,
        sector: str,
        base_year: int,
        target_year: int,
    ) -> Dict[str, Any]:
        """Calculate the total and annual reduction needed for alignment.

        Args:
            company_base_intensity: Company intensity at base year.
            sector: SDA sector identifier.
            base_year: Base year.
            target_year: Target year.

        Returns:
            Dict with required reductions for WB2C, 1.5C, and net-zero.
        """
        benchmark = self._get_sector_benchmark(sector)
        if benchmark is None:
            return {"error": f"Unknown sector: {sector}"}

        sector_base = self._interpolate_sector_intensity(benchmark, base_year)
        sector_target = self._interpolate_sector_intensity(benchmark, target_year)

        company_target = self.calculate_converged_intensity(
            company_base_intensity, sector_base,
            sector_target, sector_target,
        )

        total_red = Decimal("0")
        if company_base_intensity > Decimal("0"):
            total_red = _round_val(
                (Decimal("1") - _safe_divide(
                    company_target, company_base_intensity
                )) * Decimal("100"),
                2,
            )

        years = target_year - base_year
        annual_red = Decimal("0")
        if years > 0:
            annual_red = _round_val(total_red / _decimal(years), 2)

        return {
            "sector": sector,
            "base_year": base_year,
            "target_year": target_year,
            "years": years,
            "company_base_intensity": str(_round_val(company_base_intensity, 8)),
            "company_sda_target_intensity": str(_round_val(company_target, 8)),
            "sector_base_intensity": str(_round_val(sector_base, 8)),
            "sector_target_intensity": str(_round_val(sector_target, 8)),
            "total_reduction_pct": str(total_red),
            "annual_reduction_pct": str(annual_red),
            "unit": benchmark.get("unit", ""),
        }
