# -*- coding: utf-8 -*-
"""
DenominatorRegistryEngine - PACK-046 Intensity Metrics Engine 1
====================================================================

Manages a comprehensive registry of 25+ standard intensity denominators
with metadata, validation rules, unit conversions, and recommendation
scoring for sector- and framework-appropriate denominator selection.

Calculation Methodology:
    Denominator Recommendation Scoring:
        For each candidate denominator D and organisation context C:
            sector_score(D, C) = 1.0 if D.sectors contains C.sector else 0.0
            framework_score(D, C) = count(D.frameworks INTERSECT C.frameworks) / count(C.frameworks)
            data_availability(D, C) = 1.0 if data available for D else 0.0
            relevance(D, C) = sector_score * 0.3 + framework_score * 0.5 + data_availability * 0.2

    Denominator Validation:
        - Value must be positive (> 0)
        - Value must be non-zero
        - Year-over-year change must not exceed max_yoy_change_pct
        - Unit must match denominator definition
        - Data quality score must be in range [1, 5]

    Unit Conversion:
        All conversions use exact Decimal factors stored in the registry.
        Example: revenue_usd -> revenue_eur = value * Decimal("0.92")
        No floating-point arithmetic in any conversion path.

Regulatory References:
    - GHG Protocol Corporate Standard (2004, revised 2015), Chapter 6
    - GRI 305-4: GHG emissions intensity
    - ESRS E1-6: Gross Scopes 1, 2, 3 and Total GHG emissions (intensity)
    - CDP Climate Change C6.10: Emissions intensities
    - SEC Climate Disclosure Rule (2024), Item 1504(c)(1)
    - SBTi Corporate Manual (2023), Section 5 (intensity targets)
    - ISO 14064-1:2018 Clause 5.3.4 (intensity metrics)
    - TCFD Metrics and Targets (b): Scope 1, 2, 3 intensity

Zero-Hallucination:
    - All calculations use deterministic Decimal arithmetic
    - Denominator definitions are statically configured, not generated
    - No LLM involvement in any calculation or recommendation path
    - SHA-256 provenance hash on every result

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-046 Intensity Metrics
Engine:  1 of 10
Status:  Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import time
import uuid
from datetime import datetime, date, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator, model_validator

from greenlang.schemas import utcnow
from greenlang.schemas.enums import ValidationSeverity

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
    provenance_hash) so that identical logical inputs always produce
    the same hash.
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

def _round2(value: Any) -> float:
    """Round to 2 decimal places using ROUND_HALF_UP."""
    return float(Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))

def _round3(value: Any) -> float:
    """Round to 3 decimal places."""
    return float(Decimal(str(value)).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP))

def _round4(value: Any) -> float:
    """Round to 4 decimal places."""
    return float(Decimal(str(value)).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP))

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class DenominatorCategory(str, Enum):
    """Category of intensity denominator.

    ECONOMIC:    Revenue, EBITDA, value added, production cost.
    PHYSICAL:    Tonnes produced, MWh generated, km travelled.
    HEADCOUNT:   FTE employees, total headcount.
    AREA:        Floor area (m2, sq ft), land area (hectares).
    CAPACITY:    Installed capacity, bed count, room count.
    ACTIVITY:    Passenger-km, tonne-km, transactions.
    """
    ECONOMIC = "economic"
    PHYSICAL = "physical"
    HEADCOUNT = "headcount"
    AREA = "area"
    CAPACITY = "capacity"
    ACTIVITY = "activity"

class DenominatorUnit(str, Enum):
    """Standard units for intensity denominators."""
    # Economic
    USD_MILLION = "USD_million"
    EUR_MILLION = "EUR_million"
    GBP_MILLION = "GBP_million"
    LOCAL_CURRENCY_MILLION = "local_currency_million"
    # Physical
    TONNE = "tonne"
    KG = "kg"
    MWH = "MWh"
    GWH = "GWh"
    GJ = "GJ"
    TJ = "TJ"
    LITRE = "litre"
    BARREL = "barrel"
    # Headcount
    FTE = "FTE"
    HEADCOUNT = "headcount"
    # Area
    M2 = "m2"
    SQ_FT = "sq_ft"
    HECTARE = "hectare"
    # Capacity / Activity
    MW = "MW"
    BED = "bed"
    ROOM = "room"
    PASSENGER_KM = "passenger_km"
    TONNE_KM = "tonne_km"
    VEHICLE_KM = "vehicle_km"
    TRANSACTION = "transaction"
    UNIT = "unit"

# ---------------------------------------------------------------------------
# Constants -- Built-in Denominator Definitions
# ---------------------------------------------------------------------------

# Conversion factors between compatible units (exact Decimal factors).
UNIT_CONVERSION_FACTORS: Dict[Tuple[str, str], Decimal] = {
    # Area
    (DenominatorUnit.M2.value, DenominatorUnit.SQ_FT.value): Decimal("10.7639"),
    (DenominatorUnit.SQ_FT.value, DenominatorUnit.M2.value): Decimal("0.092903"),
    (DenominatorUnit.HECTARE.value, DenominatorUnit.M2.value): Decimal("10000"),
    (DenominatorUnit.M2.value, DenominatorUnit.HECTARE.value): Decimal("0.0001"),
    # Mass
    (DenominatorUnit.TONNE.value, DenominatorUnit.KG.value): Decimal("1000"),
    (DenominatorUnit.KG.value, DenominatorUnit.TONNE.value): Decimal("0.001"),
    # Energy
    (DenominatorUnit.MWH.value, DenominatorUnit.GJ.value): Decimal("3.6"),
    (DenominatorUnit.GJ.value, DenominatorUnit.MWH.value): Decimal("0.277778"),
    (DenominatorUnit.GWH.value, DenominatorUnit.MWH.value): Decimal("1000"),
    (DenominatorUnit.MWH.value, DenominatorUnit.GWH.value): Decimal("0.001"),
    (DenominatorUnit.TJ.value, DenominatorUnit.GJ.value): Decimal("1000"),
    (DenominatorUnit.GJ.value, DenominatorUnit.TJ.value): Decimal("0.001"),
    # Volume / Barrel
    (DenominatorUnit.BARREL.value, DenominatorUnit.LITRE.value): Decimal("158.987"),
    (DenominatorUnit.LITRE.value, DenominatorUnit.BARREL.value): Decimal("0.00629"),
    # Currency (indicative; real rates should be injected)
    (DenominatorUnit.USD_MILLION.value, DenominatorUnit.EUR_MILLION.value): Decimal("0.92"),
    (DenominatorUnit.EUR_MILLION.value, DenominatorUnit.USD_MILLION.value): Decimal("1.087"),
    (DenominatorUnit.USD_MILLION.value, DenominatorUnit.GBP_MILLION.value): Decimal("0.79"),
    (DenominatorUnit.GBP_MILLION.value, DenominatorUnit.USD_MILLION.value): Decimal("1.266"),
    (DenominatorUnit.EUR_MILLION.value, DenominatorUnit.GBP_MILLION.value): Decimal("0.859"),
    (DenominatorUnit.GBP_MILLION.value, DenominatorUnit.EUR_MILLION.value): Decimal("1.164"),
}

# Maximum year-over-year change thresholds by category (percent).
MAX_YOY_CHANGE_DEFAULTS: Dict[str, Decimal] = {
    DenominatorCategory.ECONOMIC.value: Decimal("50"),
    DenominatorCategory.PHYSICAL.value: Decimal("30"),
    DenominatorCategory.HEADCOUNT.value: Decimal("40"),
    DenominatorCategory.AREA.value: Decimal("25"),
    DenominatorCategory.CAPACITY.value: Decimal("25"),
    DenominatorCategory.ACTIVITY.value: Decimal("40"),
}

# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------

class DenominatorDefinition(BaseModel):
    """Definition of a standard intensity denominator.

    Attributes:
        denominator_id:     Unique identifier (e.g. 'revenue_usd').
        name:               Human-readable name.
        unit:               Standard unit of measurement.
        category:           Denominator category.
        description:        Detailed description.
        sectors:            Applicable sectors (GICS or custom).
        frameworks:         Regulatory frameworks that require/recommend this.
        validation_rules:   Custom validation rules.
        conversion_factors: Available unit conversions.
        max_yoy_change_pct: Maximum acceptable year-over-year change (%).
        is_universal:       Whether applicable to all sectors.
    """
    denominator_id: str = Field(..., min_length=1, max_length=100, description="Unique identifier")
    name: str = Field(..., min_length=1, max_length=200, description="Human-readable name")
    unit: str = Field(..., description="Standard unit of measurement")
    category: DenominatorCategory = Field(..., description="Denominator category")
    description: str = Field(default="", description="Detailed description")
    sectors: List[str] = Field(default_factory=list, description="Applicable sectors")
    frameworks: List[str] = Field(default_factory=list, description="Applicable frameworks")
    validation_rules: Dict[str, Any] = Field(
        default_factory=dict, description="Custom validation rules"
    )
    conversion_factors: Dict[str, str] = Field(
        default_factory=dict, description="Unit conversion targets"
    )
    max_yoy_change_pct: Decimal = Field(
        default=Decimal("50"), ge=0,
        description="Max acceptable YoY change (%)"
    )
    is_universal: bool = Field(default=False, description="Applicable to all sectors")

    @field_validator("max_yoy_change_pct", mode="before")
    @classmethod
    def coerce_max_yoy(cls, v: Any) -> Decimal:
        """Coerce to Decimal."""
        return _decimal(v)

class DenominatorValue(BaseModel):
    """A single period value for a denominator.

    Attributes:
        denominator_id:     Which denominator this value represents.
        period:             Reporting period (e.g. '2024', '2024-Q1').
        value:              Numeric value as Decimal.
        unit:               Unit of the reported value.
        data_quality_score: Data quality (1-5 per GHG Protocol).
        source:             Data source description.
        notes:              Additional notes.
    """
    denominator_id: str = Field(..., description="Denominator ID")
    period: str = Field(..., min_length=1, max_length=20, description="Reporting period")
    value: Decimal = Field(..., description="Denominator value")
    unit: str = Field(..., description="Unit of measurement")
    data_quality_score: int = Field(default=3, ge=1, le=5, description="Data quality (1-5)")
    source: str = Field(default="", description="Data source")
    notes: str = Field(default="", description="Additional notes")

    @field_validator("value", mode="before")
    @classmethod
    def coerce_value(cls, v: Any) -> Decimal:
        """Coerce value to Decimal."""
        return _decimal(v)

class DenominatorRecommendation(BaseModel):
    """Recommendation for a denominator with relevance scoring.

    Attributes:
        denominator_id:       Recommended denominator ID.
        denominator_name:     Human-readable name.
        relevance_score:      Composite relevance score (0-1).
        sector_score:         Sector match score (0 or 1).
        framework_score:      Framework coverage score (0-1).
        data_availability:    Data availability score (0 or 1).
        rationale:            Explanation of the recommendation.
        rank:                 Rank among recommendations (1 = best).
    """
    denominator_id: str = Field(..., description="Denominator ID")
    denominator_name: str = Field(default="", description="Denominator name")
    relevance_score: Decimal = Field(default=Decimal("0"), description="Relevance score (0-1)")
    sector_score: Decimal = Field(default=Decimal("0"), description="Sector score (0 or 1)")
    framework_score: Decimal = Field(default=Decimal("0"), description="Framework score (0-1)")
    data_availability: Decimal = Field(default=Decimal("0"), description="Data availability (0 or 1)")
    rationale: str = Field(default="", description="Recommendation rationale")
    rank: int = Field(default=0, ge=0, description="Rank (1 = best)")

class ValidationFinding(BaseModel):
    """A single validation finding for a denominator value.

    Attributes:
        denominator_id: Denominator being validated.
        period:         Reporting period.
        severity:       Severity level (error, warning, info).
        code:           Machine-readable finding code.
        message:        Human-readable description.
    """
    denominator_id: str = Field(..., description="Denominator ID")
    period: str = Field(default="", description="Period")
    severity: ValidationSeverity = Field(..., description="Severity")
    code: str = Field(default="", description="Finding code")
    message: str = Field(..., description="Finding message")

class RegistryInput(BaseModel):
    """Input for denominator registry operations.

    Attributes:
        organisation_id:       Organisation identifier.
        sector:                Organisation sector.
        target_frameworks:     Frameworks the organisation reports to.
        denominator_values:    Reported denominator values.
        available_data_ids:    IDs of denominators with available data.
        custom_denominators:   Additional custom denominator definitions.
        reporting_currency:    Reporting currency for economic denominators.
    """
    organisation_id: str = Field(default="", description="Organisation ID")
    sector: str = Field(default="other", description="Organisation sector")
    target_frameworks: List[str] = Field(
        default_factory=list, description="Target reporting frameworks"
    )
    denominator_values: List[DenominatorValue] = Field(
        default_factory=list, description="Reported denominator values"
    )
    available_data_ids: List[str] = Field(
        default_factory=list, description="Denominator IDs with available data"
    )
    custom_denominators: List[DenominatorDefinition] = Field(
        default_factory=list, description="Custom denominator definitions"
    )
    reporting_currency: str = Field(
        default=DenominatorUnit.USD_MILLION.value,
        description="Reporting currency"
    )

class RegistryResult(BaseModel):
    """Result from denominator registry operations.

    Attributes:
        result_id:             Unique result identifier.
        recommendations:       Ranked denominator recommendations.
        validation_findings:   Validation findings for provided values.
        available_denominators: Count of available denominators.
        converted_values:      Unit-converted denominator values.
        summary:               Summary statistics.
        warnings:              Warnings and advisory notes.
        calculated_at:         Timestamp of calculation.
        processing_time_ms:    Processing time in milliseconds.
        provenance_hash:       SHA-256 provenance hash.
    """
    result_id: str = Field(default_factory=_new_uuid, description="Result ID")
    recommendations: List[DenominatorRecommendation] = Field(
        default_factory=list, description="Ranked recommendations"
    )
    validation_findings: List[ValidationFinding] = Field(
        default_factory=list, description="Validation findings"
    )
    available_denominators: int = Field(default=0, description="Available denominator count")
    converted_values: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict, description="Converted values"
    )
    summary: Dict[str, Any] = Field(default_factory=dict, description="Summary statistics")
    warnings: List[str] = Field(default_factory=list, description="Warnings")
    calculated_at: str = Field(default="", description="Calculation timestamp (ISO 8601)")
    processing_time_ms: float = Field(default=0.0, description="Processing time (ms)")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

# ---------------------------------------------------------------------------
# Built-in Denominators
# ---------------------------------------------------------------------------

BUILT_IN_DENOMINATORS: List[DenominatorDefinition] = [
    # -- Economic (1-6) --
    DenominatorDefinition(
        denominator_id="revenue_usd",
        name="Revenue (USD millions)",
        unit=DenominatorUnit.USD_MILLION.value,
        category=DenominatorCategory.ECONOMIC,
        description="Annual revenue in USD millions. Most universal economic intensity metric.",
        sectors=["all"],
        frameworks=["GRI_305_4", "CDP_C6_10", "ESRS_E1_6", "SEC_1504", "TCFD", "SBTi"],
        max_yoy_change_pct=Decimal("50"),
        is_universal=True,
    ),
    DenominatorDefinition(
        denominator_id="revenue_eur",
        name="Revenue (EUR millions)",
        unit=DenominatorUnit.EUR_MILLION.value,
        category=DenominatorCategory.ECONOMIC,
        description="Annual revenue in EUR millions for EU-reporting entities.",
        sectors=["all"],
        frameworks=["ESRS_E1_6", "GRI_305_4", "CDP_C6_10"],
        max_yoy_change_pct=Decimal("50"),
        is_universal=True,
    ),
    DenominatorDefinition(
        denominator_id="revenue_gbp",
        name="Revenue (GBP millions)",
        unit=DenominatorUnit.GBP_MILLION.value,
        category=DenominatorCategory.ECONOMIC,
        description="Annual revenue in GBP millions for UK-reporting entities.",
        sectors=["all"],
        frameworks=["GRI_305_4", "CDP_C6_10", "TCFD"],
        max_yoy_change_pct=Decimal("50"),
        is_universal=True,
    ),
    DenominatorDefinition(
        denominator_id="ebitda_usd",
        name="EBITDA (USD millions)",
        unit=DenominatorUnit.USD_MILLION.value,
        category=DenominatorCategory.ECONOMIC,
        description="Earnings before interest, taxes, depreciation and amortisation.",
        sectors=["all"],
        frameworks=["CDP_C6_10", "TCFD"],
        max_yoy_change_pct=Decimal("60"),
        is_universal=True,
    ),
    DenominatorDefinition(
        denominator_id="value_added_usd",
        name="Gross Value Added (USD millions)",
        unit=DenominatorUnit.USD_MILLION.value,
        category=DenominatorCategory.ECONOMIC,
        description="Gross value added (revenue minus cost of goods). ESRS preferred for some sectors.",
        sectors=["manufacturing", "services"],
        frameworks=["ESRS_E1_6", "GRI_305_4"],
        max_yoy_change_pct=Decimal("50"),
        is_universal=False,
    ),
    DenominatorDefinition(
        denominator_id="production_cost_usd",
        name="Total Production Cost (USD millions)",
        unit=DenominatorUnit.USD_MILLION.value,
        category=DenominatorCategory.ECONOMIC,
        description="Total cost of production for manufacturing intensity.",
        sectors=["manufacturing", "mining"],
        frameworks=["CDP_C6_10"],
        max_yoy_change_pct=Decimal("40"),
        is_universal=False,
    ),
    # -- Physical (7-14) --
    DenominatorDefinition(
        denominator_id="production_tonnes",
        name="Production Volume (tonnes)",
        unit=DenominatorUnit.TONNE.value,
        category=DenominatorCategory.PHYSICAL,
        description="Total production output in metric tonnes.",
        sectors=["manufacturing", "mining", "cement", "steel", "chemicals", "aluminium"],
        frameworks=["SBTi_SDA", "CDP_C6_10", "ESRS_E1_6", "GRI_305_4"],
        max_yoy_change_pct=Decimal("30"),
        is_universal=False,
    ),
    DenominatorDefinition(
        denominator_id="clinker_tonnes",
        name="Clinker Production (tonnes)",
        unit=DenominatorUnit.TONNE.value,
        category=DenominatorCategory.PHYSICAL,
        description="Clinker production for cement sector SBTi SDA pathway.",
        sectors=["cement"],
        frameworks=["SBTi_SDA", "CDP_C6_10"],
        max_yoy_change_pct=Decimal("25"),
        is_universal=False,
    ),
    DenominatorDefinition(
        denominator_id="steel_tonnes",
        name="Crude Steel Production (tonnes)",
        unit=DenominatorUnit.TONNE.value,
        category=DenominatorCategory.PHYSICAL,
        description="Crude steel production for steel sector SBTi SDA pathway.",
        sectors=["steel"],
        frameworks=["SBTi_SDA", "CDP_C6_10"],
        max_yoy_change_pct=Decimal("25"),
        is_universal=False,
    ),
    DenominatorDefinition(
        denominator_id="aluminium_tonnes",
        name="Primary Aluminium Production (tonnes)",
        unit=DenominatorUnit.TONNE.value,
        category=DenominatorCategory.PHYSICAL,
        description="Primary aluminium production for aluminium sector SBTi SDA pathway.",
        sectors=["aluminium"],
        frameworks=["SBTi_SDA", "CDP_C6_10"],
        max_yoy_change_pct=Decimal("25"),
        is_universal=False,
    ),
    DenominatorDefinition(
        denominator_id="electricity_mwh",
        name="Electricity Generated (MWh)",
        unit=DenominatorUnit.MWH.value,
        category=DenominatorCategory.PHYSICAL,
        description="Net electricity generated for power sector.",
        sectors=["power", "energy", "utilities"],
        frameworks=["SBTi_SDA", "CDP_C6_10", "ESRS_E1_6"],
        max_yoy_change_pct=Decimal("30"),
        is_universal=False,
    ),
    DenominatorDefinition(
        denominator_id="electricity_gwh",
        name="Electricity Generated (GWh)",
        unit=DenominatorUnit.GWH.value,
        category=DenominatorCategory.PHYSICAL,
        description="Net electricity generated (GWh) for large utilities.",
        sectors=["power", "energy", "utilities"],
        frameworks=["SBTi_SDA", "CDP_C6_10"],
        max_yoy_change_pct=Decimal("30"),
        is_universal=False,
    ),
    DenominatorDefinition(
        denominator_id="energy_gj",
        name="Energy Consumption (GJ)",
        unit=DenominatorUnit.GJ.value,
        category=DenominatorCategory.PHYSICAL,
        description="Total energy consumption in gigajoules.",
        sectors=["all"],
        frameworks=["ESRS_E1_6", "GRI_305_4"],
        max_yoy_change_pct=Decimal("30"),
        is_universal=True,
    ),
    DenominatorDefinition(
        denominator_id="barrels_oil",
        name="Oil Production (barrels)",
        unit=DenominatorUnit.BARREL.value,
        category=DenominatorCategory.PHYSICAL,
        description="Barrels of oil equivalent produced.",
        sectors=["oil_gas", "energy"],
        frameworks=["CDP_C6_10", "SBTi_SDA"],
        max_yoy_change_pct=Decimal("30"),
        is_universal=False,
    ),
    # -- Headcount (15-16) --
    DenominatorDefinition(
        denominator_id="fte_employees",
        name="Full-Time Equivalent Employees",
        unit=DenominatorUnit.FTE.value,
        category=DenominatorCategory.HEADCOUNT,
        description="Average number of FTE employees during the reporting period.",
        sectors=["all"],
        frameworks=["GRI_305_4", "CDP_C6_10", "ESRS_E1_6", "SEC_1504"],
        max_yoy_change_pct=Decimal("40"),
        is_universal=True,
    ),
    DenominatorDefinition(
        denominator_id="total_headcount",
        name="Total Headcount",
        unit=DenominatorUnit.HEADCOUNT.value,
        category=DenominatorCategory.HEADCOUNT,
        description="Total employee headcount including part-time and contractors.",
        sectors=["all"],
        frameworks=["GRI_305_4", "CDP_C6_10"],
        max_yoy_change_pct=Decimal("40"),
        is_universal=True,
    ),
    # -- Area (17-19) --
    DenominatorDefinition(
        denominator_id="floor_area_m2",
        name="Gross Floor Area (m2)",
        unit=DenominatorUnit.M2.value,
        category=DenominatorCategory.AREA,
        description="Total gross floor area of buildings in square metres.",
        sectors=["real_estate", "retail", "hospitality", "services"],
        frameworks=["SBTi_SDA", "CDP_C6_10", "CRREM", "GRESB", "ESRS_E1_6"],
        max_yoy_change_pct=Decimal("25"),
        is_universal=False,
    ),
    DenominatorDefinition(
        denominator_id="floor_area_sqft",
        name="Gross Floor Area (sq ft)",
        unit=DenominatorUnit.SQ_FT.value,
        category=DenominatorCategory.AREA,
        description="Total gross floor area in square feet (US/UK reporting).",
        sectors=["real_estate", "retail", "hospitality", "services"],
        frameworks=["CDP_C6_10", "SEC_1504"],
        max_yoy_change_pct=Decimal("25"),
        is_universal=False,
    ),
    DenominatorDefinition(
        denominator_id="land_area_hectare",
        name="Land Area (hectares)",
        unit=DenominatorUnit.HECTARE.value,
        category=DenominatorCategory.AREA,
        description="Total managed land area in hectares.",
        sectors=["agriculture", "forestry", "mining"],
        frameworks=["SBTi_FLAG", "CDP_C6_10"],
        max_yoy_change_pct=Decimal("20"),
        is_universal=False,
    ),
    # -- Capacity (20-22) --
    DenominatorDefinition(
        denominator_id="installed_capacity_mw",
        name="Installed Capacity (MW)",
        unit=DenominatorUnit.MW.value,
        category=DenominatorCategory.CAPACITY,
        description="Total installed generation capacity in megawatts.",
        sectors=["power", "energy", "utilities"],
        frameworks=["CDP_C6_10", "SBTi_SDA"],
        max_yoy_change_pct=Decimal("25"),
        is_universal=False,
    ),
    DenominatorDefinition(
        denominator_id="hotel_rooms",
        name="Hotel Room Count",
        unit=DenominatorUnit.ROOM.value,
        category=DenominatorCategory.CAPACITY,
        description="Total available hotel rooms.",
        sectors=["hospitality"],
        frameworks=["CDP_C6_10", "GRESB"],
        max_yoy_change_pct=Decimal("20"),
        is_universal=False,
    ),
    DenominatorDefinition(
        denominator_id="hospital_beds",
        name="Hospital Bed Count",
        unit=DenominatorUnit.BED.value,
        category=DenominatorCategory.CAPACITY,
        description="Total hospital beds for healthcare sector.",
        sectors=["healthcare"],
        frameworks=["CDP_C6_10"],
        max_yoy_change_pct=Decimal("15"),
        is_universal=False,
    ),
    # -- Activity (23-27) --
    DenominatorDefinition(
        denominator_id="passenger_km",
        name="Passenger Kilometres",
        unit=DenominatorUnit.PASSENGER_KM.value,
        category=DenominatorCategory.ACTIVITY,
        description="Total passenger-kilometres for transport sector.",
        sectors=["transport", "aviation", "rail"],
        frameworks=["SBTi_SDA", "CDP_C6_10"],
        max_yoy_change_pct=Decimal("40"),
        is_universal=False,
    ),
    DenominatorDefinition(
        denominator_id="tonne_km",
        name="Tonne Kilometres",
        unit=DenominatorUnit.TONNE_KM.value,
        category=DenominatorCategory.ACTIVITY,
        description="Total tonne-kilometres for freight transport sector.",
        sectors=["transport", "logistics", "shipping"],
        frameworks=["SBTi_SDA", "CDP_C6_10"],
        max_yoy_change_pct=Decimal("40"),
        is_universal=False,
    ),
    DenominatorDefinition(
        denominator_id="vehicle_km",
        name="Vehicle Kilometres",
        unit=DenominatorUnit.VEHICLE_KM.value,
        category=DenominatorCategory.ACTIVITY,
        description="Total vehicle-kilometres driven for fleet-based transport.",
        sectors=["transport", "logistics"],
        frameworks=["CDP_C6_10"],
        max_yoy_change_pct=Decimal("35"),
        is_universal=False,
    ),
    DenominatorDefinition(
        denominator_id="transactions_count",
        name="Transaction Count",
        unit=DenominatorUnit.TRANSACTION.value,
        category=DenominatorCategory.ACTIVITY,
        description="Total number of transactions for financial/tech sectors.",
        sectors=["financial", "technology", "retail"],
        frameworks=["CDP_C6_10"],
        max_yoy_change_pct=Decimal("50"),
        is_universal=False,
    ),
    DenominatorDefinition(
        denominator_id="units_produced",
        name="Units Produced",
        unit=DenominatorUnit.UNIT.value,
        category=DenominatorCategory.ACTIVITY,
        description="Total discrete units produced for manufacturing.",
        sectors=["manufacturing", "automotive", "electronics"],
        frameworks=["CDP_C6_10", "GRI_305_4"],
        max_yoy_change_pct=Decimal("35"),
        is_universal=False,
    ),
]

# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class DenominatorRegistryEngine:
    """Manages denominator definitions, validation, and recommendations.

    Provides a registry of 27 built-in intensity denominators plus
    support for custom denominator definitions.  Implements relevance
    scoring to recommend the most appropriate denominators for a given
    organisation's sector and target reporting frameworks.

    Guarantees:
        - Deterministic: Same inputs always produce the same output.
        - Reproducible: Full provenance tracking with SHA-256 hashes.
        - Auditable: Every recommendation scored with documented rationale.
        - Zero-Hallucination: No LLM in any calculation path.

    Usage:
        engine = DenominatorRegistryEngine()
        input_data = RegistryInput(
            sector="manufacturing",
            target_frameworks=["SBTi_SDA", "ESRS_E1_6"],
            available_data_ids=["revenue_usd", "production_tonnes"],
        )
        result = engine.calculate(input_data)
        print(result.recommendations[0].denominator_id)
        print(result.provenance_hash)
    """

    def __init__(self) -> None:
        """Initialise the DenominatorRegistryEngine with built-in denominators."""
        self._version = _MODULE_VERSION
        self._registry: Dict[str, DenominatorDefinition] = {}
        for d in BUILT_IN_DENOMINATORS:
            self._registry[d.denominator_id] = d
        logger.info(
            "DenominatorRegistryEngine v%s initialised with %d built-in denominators",
            self._version,
            len(self._registry),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def calculate(self, input_data: RegistryInput) -> RegistryResult:
        """Execute denominator registry operations.

        Main entry point.  Registers any custom denominators, validates
        provided values, scores and ranks denominator recommendations,
        and performs unit conversions where applicable.

        Args:
            input_data: Registry input with sector, frameworks, and values.

        Returns:
            RegistryResult with recommendations, validations, and provenance.
        """
        t0 = time.perf_counter()
        warnings: List[str] = []

        # Register custom denominators
        for custom in input_data.custom_denominators:
            self.register_denominator(custom)

        # Validate provided values
        findings = self._validate_values(input_data.denominator_values)

        # Score and rank recommendations
        recommendations = self._score_recommendations(
            sector=input_data.sector,
            frameworks=input_data.target_frameworks,
            available_ids=input_data.available_data_ids,
        )

        # Perform unit conversions
        converted_values = self._convert_values(
            values=input_data.denominator_values,
            target_currency=input_data.reporting_currency,
        )

        # Build summary
        summary = self._build_summary(input_data, recommendations, findings)

        # Warnings
        error_count = sum(1 for f in findings if f.severity == ValidationSeverity.ERROR)
        if error_count > 0:
            warnings.append(f"{error_count} validation error(s) found in denominator values.")

        no_data = [
            r.denominator_id for r in recommendations[:5]
            if r.data_availability == Decimal("0")
        ]
        if no_data:
            warnings.append(
                f"Top-ranked denominators without available data: {no_data}. "
                f"Consider collecting data for these metrics."
            )

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        result = RegistryResult(
            recommendations=recommendations,
            validation_findings=findings,
            available_denominators=len(self._registry),
            converted_values=converted_values,
            summary=summary,
            warnings=warnings,
            calculated_at=utcnow().isoformat(),
            processing_time_ms=round(elapsed_ms, 3),
        )
        result.provenance_hash = _compute_hash(result)
        return result

    def register_denominator(self, definition: DenominatorDefinition) -> None:
        """Register a custom denominator definition.

        Args:
            definition: Denominator definition to register.

        Raises:
            ValueError: If denominator_id already exists as a built-in.
        """
        builtin_ids = {d.denominator_id for d in BUILT_IN_DENOMINATORS}
        if definition.denominator_id in builtin_ids:
            raise ValueError(
                f"Cannot override built-in denominator: {definition.denominator_id}"
            )
        self._registry[definition.denominator_id] = definition
        logger.info("Registered custom denominator: %s", definition.denominator_id)

    def get_denominator(self, denominator_id: str) -> Optional[DenominatorDefinition]:
        """Retrieve a denominator definition by ID.

        Args:
            denominator_id: Unique identifier.

        Returns:
            DenominatorDefinition or None if not found.
        """
        return self._registry.get(denominator_id)

    def list_denominators(
        self,
        category: Optional[DenominatorCategory] = None,
        sector: Optional[str] = None,
        framework: Optional[str] = None,
    ) -> List[DenominatorDefinition]:
        """List denominators with optional filtering.

        Args:
            category:  Filter by denominator category.
            sector:    Filter by applicable sector.
            framework: Filter by applicable framework.

        Returns:
            List of matching DenominatorDefinition objects.
        """
        results: List[DenominatorDefinition] = []
        for d in self._registry.values():
            if category is not None and d.category != category:
                continue
            if sector is not None:
                if not d.is_universal and sector not in d.sectors and "all" not in d.sectors:
                    continue
            if framework is not None:
                if framework not in d.frameworks:
                    continue
            results.append(d)
        return results

    def convert_value(
        self,
        value: Decimal,
        from_unit: str,
        to_unit: str,
    ) -> Optional[Decimal]:
        """Convert a denominator value between compatible units.

        Uses exact Decimal conversion factors.  Returns None if
        conversion is not available.

        Args:
            value:     Value to convert.
            from_unit: Source unit.
            to_unit:   Target unit.

        Returns:
            Converted Decimal value, or None if no conversion factor exists.
        """
        if from_unit == to_unit:
            return value

        factor = UNIT_CONVERSION_FACTORS.get((from_unit, to_unit))
        if factor is None:
            return None

        return (value * factor).quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP)

    def validate_value(
        self,
        denom_value: DenominatorValue,
        previous_value: Optional[DenominatorValue] = None,
    ) -> List[ValidationFinding]:
        """Validate a single denominator value.

        Checks:
            - Value is positive (> 0)
            - Value is not zero
            - Unit matches denominator definition
            - Data quality score is in range [1, 5]
            - YoY change within limits (if previous value provided)

        Args:
            denom_value:    Value to validate.
            previous_value: Previous period value for YoY check.

        Returns:
            List of ValidationFinding objects.
        """
        findings: List[ValidationFinding] = []
        defn = self._registry.get(denom_value.denominator_id)

        # Check denominator exists
        if defn is None:
            findings.append(ValidationFinding(
                denominator_id=denom_value.denominator_id,
                period=denom_value.period,
                severity=ValidationSeverity.ERROR,
                code="DENOM_NOT_FOUND",
                message=f"Denominator '{denom_value.denominator_id}' not found in registry.",
            ))
            return findings

        # Check positive
        if denom_value.value <= Decimal("0"):
            findings.append(ValidationFinding(
                denominator_id=denom_value.denominator_id,
                period=denom_value.period,
                severity=ValidationSeverity.ERROR,
                code="VALUE_NON_POSITIVE",
                message=f"Denominator value must be positive (got {denom_value.value}).",
            ))

        # Check unit consistency
        if denom_value.unit != defn.unit:
            # Check if conversion is available
            convertible = (denom_value.unit, defn.unit) in UNIT_CONVERSION_FACTORS
            if convertible:
                findings.append(ValidationFinding(
                    denominator_id=denom_value.denominator_id,
                    period=denom_value.period,
                    severity=ValidationSeverity.WARNING,
                    code="UNIT_MISMATCH_CONVERTIBLE",
                    message=(
                        f"Unit '{denom_value.unit}' differs from standard "
                        f"'{defn.unit}' but conversion is available."
                    ),
                ))
            else:
                findings.append(ValidationFinding(
                    denominator_id=denom_value.denominator_id,
                    period=denom_value.period,
                    severity=ValidationSeverity.ERROR,
                    code="UNIT_MISMATCH",
                    message=(
                        f"Unit '{denom_value.unit}' does not match expected "
                        f"'{defn.unit}' and no conversion is available."
                    ),
                ))

        # Check data quality score
        if not (1 <= denom_value.data_quality_score <= 5):
            findings.append(ValidationFinding(
                denominator_id=denom_value.denominator_id,
                period=denom_value.period,
                severity=ValidationSeverity.ERROR,
                code="DQ_OUT_OF_RANGE",
                message=f"Data quality score must be 1-5 (got {denom_value.data_quality_score}).",
            ))

        # Check YoY change
        if previous_value is not None and previous_value.value > Decimal("0"):
            change_pct = abs(
                (denom_value.value - previous_value.value)
                / previous_value.value * Decimal("100")
            )
            max_change = defn.max_yoy_change_pct
            if change_pct > max_change:
                findings.append(ValidationFinding(
                    denominator_id=denom_value.denominator_id,
                    period=denom_value.period,
                    severity=ValidationSeverity.WARNING,
                    code="YOY_CHANGE_EXCEEDED",
                    message=(
                        f"Year-over-year change of {_round2(change_pct)}% exceeds "
                        f"maximum {_round2(max_change)}% for this denominator."
                    ),
                ))

        return findings

    def recommend_denominators(
        self,
        sector: str,
        frameworks: List[str],
        available_ids: Optional[List[str]] = None,
        top_n: int = 10,
    ) -> List[DenominatorRecommendation]:
        """Recommend denominators for an organisation.

        Scoring formula:
            relevance = sector_score * 0.3 + framework_score * 0.5 + data_availability * 0.2

        Where:
            sector_score = 1.0 if denominator.sectors contains sector (or is_universal), else 0.0
            framework_score = count(denominator.frameworks INTERSECT target_frameworks) / count(target_frameworks)
            data_availability = 1.0 if denominator_id in available_ids, else 0.0

        Args:
            sector:        Organisation sector.
            frameworks:    Target reporting frameworks.
            available_ids: Denominator IDs with available data.
            top_n:         Number of recommendations to return.

        Returns:
            Ranked list of DenominatorRecommendation objects.
        """
        return self._score_recommendations(sector, frameworks, available_ids or [], top_n)

    # ------------------------------------------------------------------
    # Internal Methods
    # ------------------------------------------------------------------

    def _score_recommendations(
        self,
        sector: str,
        frameworks: List[str],
        available_ids: List[str],
        top_n: int = 10,
    ) -> List[DenominatorRecommendation]:
        """Score and rank all denominators for relevance.

        Args:
            sector:        Organisation sector.
            frameworks:    Target reporting frameworks.
            available_ids: IDs with available data.
            top_n:         Number of results to return.

        Returns:
            Sorted list of recommendations.
        """
        recommendations: List[DenominatorRecommendation] = []
        framework_count = Decimal(str(max(len(frameworks), 1)))

        for defn in self._registry.values():
            # Sector score
            sector_match = (
                defn.is_universal
                or "all" in defn.sectors
                or sector in defn.sectors
            )
            sector_score = Decimal("1") if sector_match else Decimal("0")

            # Framework score
            if frameworks:
                matching_frameworks = sum(
                    1 for fw in frameworks if fw in defn.frameworks
                )
                framework_score = _safe_divide(
                    Decimal(str(matching_frameworks)), framework_count
                )
            else:
                framework_score = Decimal("0.5")

            # Data availability
            data_avail = Decimal("1") if defn.denominator_id in available_ids else Decimal("0")

            # Composite score: sector*0.3 + framework*0.5 + data*0.2
            relevance = (
                sector_score * Decimal("0.3")
                + framework_score * Decimal("0.5")
                + data_avail * Decimal("0.2")
            ).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)

            # Build rationale
            rationale_parts: List[str] = []
            if sector_match:
                rationale_parts.append(f"Applicable to '{sector}' sector")
            else:
                rationale_parts.append(f"Not sector-specific for '{sector}'")
            if frameworks:
                matched = [fw for fw in frameworks if fw in defn.frameworks]
                if matched:
                    rationale_parts.append(f"Covers frameworks: {', '.join(matched)}")
                else:
                    rationale_parts.append("No framework overlap")
            if defn.denominator_id in available_ids:
                rationale_parts.append("Data available")
            else:
                rationale_parts.append("Data not yet available")

            recommendations.append(DenominatorRecommendation(
                denominator_id=defn.denominator_id,
                denominator_name=defn.name,
                relevance_score=relevance,
                sector_score=sector_score,
                framework_score=framework_score.quantize(
                    Decimal("0.0001"), rounding=ROUND_HALF_UP
                ),
                data_availability=data_avail,
                rationale="; ".join(rationale_parts),
            ))

        # Sort by relevance descending, then alphabetically for stability
        recommendations.sort(
            key=lambda r: (-r.relevance_score, r.denominator_id)
        )

        # Assign ranks
        for i, rec in enumerate(recommendations):
            rec.rank = i + 1

        return recommendations[:top_n]

    def _validate_values(
        self,
        values: List[DenominatorValue],
    ) -> List[ValidationFinding]:
        """Validate all provided denominator values.

        Groups values by denominator_id and period to enable YoY checks.

        Args:
            values: List of denominator values to validate.

        Returns:
            List of all validation findings.
        """
        all_findings: List[ValidationFinding] = []

        # Group by denominator_id and sort by period
        grouped: Dict[str, List[DenominatorValue]] = {}
        for v in values:
            grouped.setdefault(v.denominator_id, []).append(v)

        for denom_id, denom_values in grouped.items():
            sorted_vals = sorted(denom_values, key=lambda x: x.period)
            prev: Optional[DenominatorValue] = None
            for val in sorted_vals:
                findings = self.validate_value(val, prev)
                all_findings.extend(findings)
                prev = val

        return all_findings

    def _convert_values(
        self,
        values: List[DenominatorValue],
        target_currency: str,
    ) -> Dict[str, Dict[str, Any]]:
        """Convert denominator values to standard/target units.

        Args:
            values:          Denominator values to convert.
            target_currency: Target currency for economic denominators.

        Returns:
            Dictionary of converted values keyed by denominator_id.
        """
        converted: Dict[str, Dict[str, Any]] = {}

        for val in values:
            defn = self._registry.get(val.denominator_id)
            if defn is None:
                continue

            entry: Dict[str, Any] = {
                "original_value": str(val.value),
                "original_unit": val.unit,
                "period": val.period,
                "conversions": {},
            }

            # If unit differs from standard, convert to standard
            if val.unit != defn.unit:
                standard_val = self.convert_value(val.value, val.unit, defn.unit)
                if standard_val is not None:
                    entry["conversions"][defn.unit] = str(standard_val)

            # For economic denominators, convert to target currency
            if defn.category == DenominatorCategory.ECONOMIC and val.unit != target_currency:
                currency_val = self.convert_value(val.value, val.unit, target_currency)
                if currency_val is not None:
                    entry["conversions"][target_currency] = str(currency_val)

            key = f"{val.denominator_id}_{val.period}"
            converted[key] = entry

        return converted

    def _build_summary(
        self,
        input_data: RegistryInput,
        recommendations: List[DenominatorRecommendation],
        findings: List[ValidationFinding],
    ) -> Dict[str, Any]:
        """Build summary statistics for the result.

        Args:
            input_data:      Original input.
            recommendations: Scored recommendations.
            findings:        Validation findings.

        Returns:
            Dictionary of summary statistics.
        """
        error_count = sum(1 for f in findings if f.severity == ValidationSeverity.ERROR)
        warning_count = sum(1 for f in findings if f.severity == ValidationSeverity.WARNING)
        info_count = sum(1 for f in findings if f.severity == ValidationSeverity.INFO)

        top_recs = [
            {"id": r.denominator_id, "name": r.denominator_name, "score": str(r.relevance_score)}
            for r in recommendations[:5]
        ]

        categories_available = set()
        for v in input_data.denominator_values:
            defn = self._registry.get(v.denominator_id)
            if defn:
                categories_available.add(defn.category.value)

        return {
            "total_denominators_in_registry": len(self._registry),
            "custom_denominators_registered": len(input_data.custom_denominators),
            "values_provided": len(input_data.denominator_values),
            "validation_errors": error_count,
            "validation_warnings": warning_count,
            "validation_info": info_count,
            "top_5_recommendations": top_recs,
            "categories_with_data": sorted(categories_available),
            "sector": input_data.sector,
            "target_frameworks": input_data.target_frameworks,
        }

    # ------------------------------------------------------------------
    # Utility Methods
    # ------------------------------------------------------------------

    def get_version(self) -> str:
        """Return engine version string."""
        return self._version

    def get_registry_size(self) -> int:
        """Return total number of registered denominators."""
        return len(self._registry)

    def get_categories(self) -> List[str]:
        """Return list of available denominator categories."""
        return sorted(set(d.category.value for d in self._registry.values()))

    def get_frameworks(self) -> List[str]:
        """Return list of all referenced frameworks."""
        frameworks: set[str] = set()
        for d in self._registry.values():
            frameworks.update(d.frameworks)
        return sorted(frameworks)

    def get_sectors(self) -> List[str]:
        """Return list of all referenced sectors."""
        sectors: set[str] = set()
        for d in self._registry.values():
            sectors.update(d.sectors)
        sectors.discard("all")
        return sorted(sectors)

    def get_conversion_pairs(self) -> List[Tuple[str, str]]:
        """Return list of available unit conversion pairs."""
        return sorted(UNIT_CONVERSION_FACTORS.keys())

# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------

def get_built_in_denominators() -> List[DenominatorDefinition]:
    """Return the built-in denominator definitions.

    Returns:
        List of all 27 built-in DenominatorDefinition objects.
    """
    return list(BUILT_IN_DENOMINATORS)

def recommend_denominators(
    sector: str,
    frameworks: List[str],
    available_ids: Optional[List[str]] = None,
) -> List[DenominatorRecommendation]:
    """Module-level convenience for denominator recommendation.

    Args:
        sector:        Organisation sector.
        frameworks:    Target reporting frameworks.
        available_ids: Denominator IDs with available data.

    Returns:
        Ranked list of DenominatorRecommendation objects.
    """
    engine = DenominatorRegistryEngine()
    return engine.recommend_denominators(sector, frameworks, available_ids)

# ---------------------------------------------------------------------------
# __all__
# ---------------------------------------------------------------------------

__all__ = [
    # Enums
    "DenominatorCategory",
    "DenominatorUnit",
    "ValidationSeverity",
    # Models
    "DenominatorDefinition",
    "DenominatorValue",
    "DenominatorRecommendation",
    "ValidationFinding",
    "RegistryInput",
    "RegistryResult",
    # Engine
    "DenominatorRegistryEngine",
    # Convenience functions
    "get_built_in_denominators",
    "recommend_denominators",
    # Constants
    "BUILT_IN_DENOMINATORS",
    "UNIT_CONVERSION_FACTORS",
    "MAX_YOY_CHANGE_DEFAULTS",
]
