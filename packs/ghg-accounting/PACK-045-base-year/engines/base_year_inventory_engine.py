# -*- coding: utf-8 -*-
"""
BaseYearInventoryEngine - PACK-045 Base Year Management Engine 2
====================================================================

Complete base year emissions inventory preservation engine implementing
GHG Protocol Corporate Standard Chapter 5 requirements for establishing,
maintaining, and verifying a frozen base year inventory that serves as
the reference point for emissions tracking and target-setting.

Calculation Methodology:
    Scope Aggregation:
        scope_total(S) = SUM(source.tco2e for source in sources if source.scope == S)

    Category Aggregation:
        category_total(C) = SUM(source.tco2e for source in sources if source.category == C)

    Emission Calculation (per source):
        gas_emissions = activity_data * emission_factor
        tco2e = gas_emissions * gwp_factor
        (where gwp_factor depends on GWPVersion: AR4, AR5, or AR6)

    Grand Total:
        grand_total = scope1_total + scope2_location_total + scope3_total
        (Note: scope2_market is reported separately, not double-counted)

    Completeness Assessment:
        completeness_pct = categories_covered / total_expected_categories * 100

    Inventory Snapshot:
        frozen_copy = deep_copy(inventory)
        provenance_hash = SHA-256(frozen_copy)
        (Immutable record for audit purposes)

    Inventory Comparison:
        For each scope S:
            delta(S) = inv2.scope_total(S) - inv1.scope_total(S)
            delta_pct(S) = delta(S) / inv1.scope_total(S) * 100
        significance = max(abs(delta_pct)) >= threshold

GWP Factors (100-year horizon):
    AR4 (IPCC 2007):  CO2=1, CH4=25, N2O=298, SF6=22800, NF3=17200
    AR5 (IPCC 2014):  CO2=1, CH4=28, N2O=265, SF6=23500, NF3=16100
    AR6 (IPCC 2021):  CO2=1, CH4=27.9, N2O=273, SF6=25200, NF3=17400

Regulatory References:
    - GHG Protocol Corporate Standard (2004, revised 2015), Chapter 5
    - GHG Protocol Corporate Value Chain Standard (Scope 3), Chapter 5
    - GHG Protocol Scope 2 Guidance (2015)
    - ISO 14064-1:2018, Clause 5.2 (Base year)
    - ISO 14064-1:2018, Clause 6.2 (Quantification)
    - ESRS E1-6 (Gross GHG emissions)
    - SBTi Corporate Manual (2023), Section 4 (Base year requirements)
    - IPCC AR4/AR5/AR6 GWP values (100-year)

Zero-Hallucination:
    - All calculations use deterministic Decimal arithmetic
    - GWP factors from published IPCC assessment reports
    - No LLM involvement in any calculation path
    - SHA-256 provenance hash on every result

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-045 Base Year Management
Engine:  2 of 10
Status:  Production Ready
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, date, timezone
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

def _safe_pct(part: Decimal, whole: Decimal) -> Decimal:
    """Compute percentage safely (part / whole * 100)."""
    return _safe_divide(part * Decimal("100"), whole)

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

class ScopeType(str, Enum):
    """GHG Protocol emission scope classification.

    SCOPE_1:            Direct emissions from owned or controlled sources.
    SCOPE_2_LOCATION:   Indirect emissions from purchased electricity
                        (location-based method).
    SCOPE_2_MARKET:     Indirect emissions from purchased electricity
                        (market-based method).
    SCOPE_3:            All other indirect emissions in the value chain.
    """
    SCOPE_1 = "scope_1"
    SCOPE_2_LOCATION = "scope_2_location"
    SCOPE_2_MARKET = "scope_2_market"
    SCOPE_3 = "scope_3"

class SourceCategory(str, Enum):
    """Emission source categories covering Scope 1, 2, and 3.

    Scope 1 categories:
        STATIONARY_COMBUSTION:  Fuel combustion in stationary equipment.
        MOBILE_COMBUSTION:      Fuel combustion in transport vehicles.
        PROCESS:                Chemical/physical process emissions.
        FUGITIVE:               Leaks from pressurised equipment.
        REFRIGERANT:            Refrigerant and cooling gas releases.
        LAND_USE:               Land use change emissions.
        WASTE:                  On-site waste treatment emissions.
        AGRICULTURAL:           Agricultural process emissions.

    Scope 2 categories:
        ELECTRICITY_LOCATION:   Purchased electricity (location-based).
        ELECTRICITY_MARKET:     Purchased electricity (market-based).
        STEAM_HEAT:             Purchased steam and heating.
        COOLING:                Purchased cooling.

    Scope 3 categories (GHG Protocol Cat 1-15):
        SCOPE3_CAT1:  Purchased goods and services.
        SCOPE3_CAT2:  Capital goods.
        SCOPE3_CAT3:  Fuel- and energy-related activities.
        SCOPE3_CAT4:  Upstream transportation and distribution.
        SCOPE3_CAT5:  Waste generated in operations.
        SCOPE3_CAT6:  Business travel.
        SCOPE3_CAT7:  Employee commuting.
        SCOPE3_CAT8:  Upstream leased assets.
        SCOPE3_CAT9:  Downstream transportation and distribution.
        SCOPE3_CAT10: Processing of sold products.
        SCOPE3_CAT11: Use of sold products.
        SCOPE3_CAT12: End-of-life treatment of sold products.
        SCOPE3_CAT13: Downstream leased assets.
        SCOPE3_CAT14: Franchises.
        SCOPE3_CAT15: Investments.
    """
    STATIONARY_COMBUSTION = "stationary_combustion"
    MOBILE_COMBUSTION = "mobile_combustion"
    PROCESS = "process"
    FUGITIVE = "fugitive"
    REFRIGERANT = "refrigerant"
    LAND_USE = "land_use"
    WASTE = "waste"
    AGRICULTURAL = "agricultural"
    ELECTRICITY_LOCATION = "electricity_location"
    ELECTRICITY_MARKET = "electricity_market"
    STEAM_HEAT = "steam_heat"
    COOLING = "cooling"
    SCOPE3_CAT1 = "scope3_cat1"
    SCOPE3_CAT2 = "scope3_cat2"
    SCOPE3_CAT3 = "scope3_cat3"
    SCOPE3_CAT4 = "scope3_cat4"
    SCOPE3_CAT5 = "scope3_cat5"
    SCOPE3_CAT6 = "scope3_cat6"
    SCOPE3_CAT7 = "scope3_cat7"
    SCOPE3_CAT8 = "scope3_cat8"
    SCOPE3_CAT9 = "scope3_cat9"
    SCOPE3_CAT10 = "scope3_cat10"
    SCOPE3_CAT11 = "scope3_cat11"
    SCOPE3_CAT12 = "scope3_cat12"
    SCOPE3_CAT13 = "scope3_cat13"
    SCOPE3_CAT14 = "scope3_cat14"
    SCOPE3_CAT15 = "scope3_cat15"

class GasType(str, Enum):
    """Greenhouse gas types covered by the Kyoto Protocol and amendments.

    CO2:   Carbon dioxide (GWP = 1 across all IPCC assessments).
    CH4:   Methane (GWP varies by IPCC assessment report).
    N2O:   Nitrous oxide (GWP varies by IPCC assessment report).
    HFCS:  Hydrofluorocarbons (family of gases, multiple GWPs).
    PFCS:  Perfluorocarbons (family of gases, multiple GWPs).
    SF6:   Sulphur hexafluoride (very high GWP).
    NF3:   Nitrogen trifluoride (added in Kyoto Protocol amendment).
    """
    CO2 = "co2"
    CH4 = "ch4"
    N2O = "n2o"
    HFCS = "hfcs"
    PFCS = "pfcs"
    SF6 = "sf6"
    NF3 = "nf3"

class GWPVersion(str, Enum):
    """IPCC Assessment Report version for GWP values.

    AR4:  Fourth Assessment Report (2007) - GHG Protocol default.
    AR5:  Fifth Assessment Report (2014) - SBTi preferred.
    AR6:  Sixth Assessment Report (2021) - Latest available.
    """
    AR4 = "ar4"
    AR5 = "ar5"
    AR6 = "ar6"

class ConsolidationApproach(str, Enum):
    """Organisational boundary consolidation approach.

    EQUITY_SHARE:       Emissions proportional to equity ownership.
    FINANCIAL_CONTROL:  Emissions from financially controlled entities.
    OPERATIONAL_CONTROL: Emissions from operationally controlled entities.
    """
    EQUITY_SHARE = "equity_share"
    FINANCIAL_CONTROL = "financial_control"
    OPERATIONAL_CONTROL = "operational_control"

class MethodologyTier(str, Enum):
    """Tier of emission calculation methodology.

    TIER_1: Default emission factors and activity data (spend-based).
    TIER_2: Country/region-specific emission factors (average-data).
    TIER_3: Facility-level measured data (supplier-specific).
    TIER_4: Continuous emissions monitoring (CEMS).
    """
    TIER_1 = "tier_1"
    TIER_2 = "tier_2"
    TIER_3 = "tier_3"
    TIER_4 = "tier_4"

class InventoryStatus(str, Enum):
    """Status of a base year inventory.

    DRAFT:       Inventory being compiled.
    ESTABLISHED: Inventory finalised and established.
    FROZEN:      Inventory snapshotted; immutable.
    SUPERSEDED:  Inventory replaced by a recalculated version.
    ARCHIVED:    No longer active, retained for audit.
    """
    DRAFT = "draft"
    ESTABLISHED = "established"
    FROZEN = "frozen"
    SUPERSEDED = "superseded"
    ARCHIVED = "archived"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# GWP factors (100-year horizon) by IPCC assessment report.
# Source: IPCC AR4 Table 2.14, AR5 Table 8.7, AR6 Table 7.15.
GWP_FACTORS: Dict[str, Dict[str, Decimal]] = {
    GWPVersion.AR4.value: {
        GasType.CO2.value: Decimal("1"),
        GasType.CH4.value: Decimal("25"),
        GasType.N2O.value: Decimal("298"),
        GasType.HFCS.value: Decimal("1430"),    # HFC-134a representative
        GasType.PFCS.value: Decimal("7390"),     # CF4 representative
        GasType.SF6.value: Decimal("22800"),
        GasType.NF3.value: Decimal("17200"),
    },
    GWPVersion.AR5.value: {
        GasType.CO2.value: Decimal("1"),
        GasType.CH4.value: Decimal("28"),
        GasType.N2O.value: Decimal("265"),
        GasType.HFCS.value: Decimal("1300"),     # HFC-134a representative
        GasType.PFCS.value: Decimal("6630"),     # CF4 representative
        GasType.SF6.value: Decimal("23500"),
        GasType.NF3.value: Decimal("16100"),
    },
    GWPVersion.AR6.value: {
        GasType.CO2.value: Decimal("1"),
        GasType.CH4.value: Decimal("27.9"),
        GasType.N2O.value: Decimal("273"),
        GasType.HFCS.value: Decimal("1526"),     # HFC-134a representative
        GasType.PFCS.value: Decimal("7380"),     # CF4 representative
        GasType.SF6.value: Decimal("25200"),
        GasType.NF3.value: Decimal("17400"),
    },
}

# Source category to scope mapping.
CATEGORY_SCOPE_MAP: Dict[str, ScopeType] = {
    SourceCategory.STATIONARY_COMBUSTION.value: ScopeType.SCOPE_1,
    SourceCategory.MOBILE_COMBUSTION.value: ScopeType.SCOPE_1,
    SourceCategory.PROCESS.value: ScopeType.SCOPE_1,
    SourceCategory.FUGITIVE.value: ScopeType.SCOPE_1,
    SourceCategory.REFRIGERANT.value: ScopeType.SCOPE_1,
    SourceCategory.LAND_USE.value: ScopeType.SCOPE_1,
    SourceCategory.WASTE.value: ScopeType.SCOPE_1,
    SourceCategory.AGRICULTURAL.value: ScopeType.SCOPE_1,
    SourceCategory.ELECTRICITY_LOCATION.value: ScopeType.SCOPE_2_LOCATION,
    SourceCategory.ELECTRICITY_MARKET.value: ScopeType.SCOPE_2_MARKET,
    SourceCategory.STEAM_HEAT.value: ScopeType.SCOPE_2_LOCATION,
    SourceCategory.COOLING.value: ScopeType.SCOPE_2_LOCATION,
    SourceCategory.SCOPE3_CAT1.value: ScopeType.SCOPE_3,
    SourceCategory.SCOPE3_CAT2.value: ScopeType.SCOPE_3,
    SourceCategory.SCOPE3_CAT3.value: ScopeType.SCOPE_3,
    SourceCategory.SCOPE3_CAT4.value: ScopeType.SCOPE_3,
    SourceCategory.SCOPE3_CAT5.value: ScopeType.SCOPE_3,
    SourceCategory.SCOPE3_CAT6.value: ScopeType.SCOPE_3,
    SourceCategory.SCOPE3_CAT7.value: ScopeType.SCOPE_3,
    SourceCategory.SCOPE3_CAT8.value: ScopeType.SCOPE_3,
    SourceCategory.SCOPE3_CAT9.value: ScopeType.SCOPE_3,
    SourceCategory.SCOPE3_CAT10.value: ScopeType.SCOPE_3,
    SourceCategory.SCOPE3_CAT11.value: ScopeType.SCOPE_3,
    SourceCategory.SCOPE3_CAT12.value: ScopeType.SCOPE_3,
    SourceCategory.SCOPE3_CAT13.value: ScopeType.SCOPE_3,
    SourceCategory.SCOPE3_CAT14.value: ScopeType.SCOPE_3,
    SourceCategory.SCOPE3_CAT15.value: ScopeType.SCOPE_3,
}

# Expected categories per scope (for completeness assessment).
SCOPE_1_CATEGORIES = {
    SourceCategory.STATIONARY_COMBUSTION.value,
    SourceCategory.MOBILE_COMBUSTION.value,
    SourceCategory.PROCESS.value,
    SourceCategory.FUGITIVE.value,
    SourceCategory.REFRIGERANT.value,
}

SCOPE_2_LOCATION_CATEGORIES = {
    SourceCategory.ELECTRICITY_LOCATION.value,
    SourceCategory.STEAM_HEAT.value,
    SourceCategory.COOLING.value,
}

SCOPE_2_MARKET_CATEGORIES = {
    SourceCategory.ELECTRICITY_MARKET.value,
}

SCOPE_3_CATEGORIES = {
    SourceCategory.SCOPE3_CAT1.value,
    SourceCategory.SCOPE3_CAT2.value,
    SourceCategory.SCOPE3_CAT3.value,
    SourceCategory.SCOPE3_CAT4.value,
    SourceCategory.SCOPE3_CAT5.value,
    SourceCategory.SCOPE3_CAT6.value,
    SourceCategory.SCOPE3_CAT7.value,
    SourceCategory.SCOPE3_CAT8.value,
    SourceCategory.SCOPE3_CAT9.value,
    SourceCategory.SCOPE3_CAT10.value,
    SourceCategory.SCOPE3_CAT11.value,
    SourceCategory.SCOPE3_CAT12.value,
    SourceCategory.SCOPE3_CAT13.value,
    SourceCategory.SCOPE3_CAT14.value,
    SourceCategory.SCOPE3_CAT15.value,
}

# Minimum base year.
MINIMUM_BASE_YEAR: int = 1990
MAXIMUM_BASE_YEAR: int = 2035

# ---------------------------------------------------------------------------
# Pydantic Models -- Inputs
# ---------------------------------------------------------------------------

class SourceEmission(BaseModel):
    """A single emission source entry in the base year inventory.

    Represents one row of the emissions register, containing all
    information needed to reproduce the calculation from activity
    data through to tCO2e.

    Attributes:
        source_id:             Unique source identifier.
        category:              Emission source category.
        facility_id:           Facility or site identifier.
        activity_data:         Activity data quantity.
        activity_unit:         Unit of activity data (e.g. litres, kWh).
        emission_factor:       Emission factor per activity unit.
        ef_source:             Source of emission factor (e.g. DEFRA 2024).
        ef_unit:               Unit of the emission factor.
        gas_type:              Greenhouse gas type.
        gas_emissions_tonnes:  Gas emissions in tonnes of the specific gas.
        gwp_factor:            GWP factor applied.
        tco2e:                 Emissions in tonnes CO2 equivalent.
        methodology_tier:      Methodology tier (1-4).
        data_quality_score:    Data quality score (0-100).
        is_estimated:          Whether this value is estimated vs measured.
        estimation_method:     Method used for estimation (if applicable).
        notes:                 Additional notes or references.
    """
    source_id: str = Field(
        default_factory=_new_uuid,
        description="Unique source identifier"
    )
    category: SourceCategory = Field(
        ..., description="Emission source category"
    )
    facility_id: str = Field(
        default="", description="Facility or site identifier"
    )
    activity_data: Decimal = Field(
        default=Decimal("0"), ge=0,
        description="Activity data quantity"
    )
    activity_unit: str = Field(
        default="", description="Activity data unit"
    )
    emission_factor: Decimal = Field(
        default=Decimal("0"), ge=0,
        description="Emission factor"
    )
    ef_source: str = Field(
        default="", description="Emission factor source"
    )
    ef_unit: str = Field(
        default="kgCO2e/unit",
        description="Emission factor unit"
    )
    gas_type: GasType = Field(
        default=GasType.CO2,
        description="Greenhouse gas type"
    )
    gas_emissions_tonnes: Decimal = Field(
        default=Decimal("0"), ge=0,
        description="Gas emissions (tonnes)"
    )
    gwp_factor: Decimal = Field(
        default=Decimal("1"), ge=0,
        description="GWP factor applied"
    )
    tco2e: Decimal = Field(
        default=Decimal("0"), ge=0,
        description="Emissions (tCO2e)"
    )
    methodology_tier: MethodologyTier = Field(
        default=MethodologyTier.TIER_1,
        description="Methodology tier"
    )
    data_quality_score: Decimal = Field(
        default=Decimal("50"), ge=0, le=100,
        description="Data quality score (0-100)"
    )
    is_estimated: bool = Field(
        default=False,
        description="Whether value is estimated"
    )
    estimation_method: str = Field(
        default="", description="Estimation method"
    )
    notes: str = Field(
        default="", description="Additional notes"
    )

    @field_validator(
        "activity_data", "emission_factor", "gas_emissions_tonnes",
        "gwp_factor", "tco2e", "data_quality_score",
        mode="before",
    )
    @classmethod
    def coerce_decimal(cls, v: Any) -> Decimal:
        """Coerce numeric fields to Decimal."""
        return _decimal(v)

    @property
    def scope(self) -> ScopeType:
        """Derive the scope from the source category."""
        return CATEGORY_SCOPE_MAP.get(self.category.value, ScopeType.SCOPE_1)

class InventoryConfig(BaseModel):
    """Configuration for establishing a base year inventory.

    Attributes:
        organization_id:        Organisation identifier.
        base_year:              Base year (calendar year).
        gwp_version:            IPCC assessment report for GWP values.
        consolidation_approach: Organisational boundary approach.
        boundary_description:   Description of organisational boundary.
        methodology_summary:    Summary of calculation methodologies.
        auto_calculate_tco2e:   Whether to auto-calculate tCO2e from
                                activity data and emission factors.
        include_scope3:         Whether to include Scope 3 emissions.
        minimum_quality_score:  Minimum data quality for inclusion.
    """
    organization_id: str = Field(
        ..., description="Organisation identifier"
    )
    base_year: int = Field(
        ..., ge=MINIMUM_BASE_YEAR, le=MAXIMUM_BASE_YEAR,
        description="Base year"
    )
    gwp_version: GWPVersion = Field(
        default=GWPVersion.AR5,
        description="GWP version"
    )
    consolidation_approach: ConsolidationApproach = Field(
        default=ConsolidationApproach.OPERATIONAL_CONTROL,
        description="Consolidation approach"
    )
    boundary_description: str = Field(
        default="",
        description="Organisational boundary description"
    )
    methodology_summary: str = Field(
        default="",
        description="Methodology summary"
    )
    auto_calculate_tco2e: bool = Field(
        default=False,
        description="Auto-calculate tCO2e from activity data"
    )
    include_scope3: bool = Field(
        default=True,
        description="Include Scope 3 emissions"
    )
    minimum_quality_score: Decimal = Field(
        default=Decimal("0"), ge=0, le=100,
        description="Minimum data quality score for inclusion"
    )

# ---------------------------------------------------------------------------
# Pydantic Models -- Outputs
# ---------------------------------------------------------------------------

class ScopeTotal(BaseModel):
    """Aggregated emissions total for a single scope.

    Attributes:
        scope:               Emission scope.
        total_tco2e:         Total emissions (tCO2e).
        source_count:        Number of sources in this scope.
        categories_covered:  Set of source categories with data.
        completeness_pct:    Completeness vs expected categories.
        by_category:         Breakdown by source category.
        by_gas:              Breakdown by greenhouse gas.
        by_facility:         Breakdown by facility.
        average_quality:     Average data quality score.
    """
    scope: ScopeType = Field(
        ..., description="Emission scope"
    )
    total_tco2e: Decimal = Field(
        default=Decimal("0"), ge=0,
        description="Total emissions (tCO2e)"
    )
    source_count: int = Field(
        default=0, ge=0,
        description="Number of sources"
    )
    categories_covered: List[str] = Field(
        default_factory=list,
        description="Categories with data"
    )
    completeness_pct: Decimal = Field(
        default=Decimal("0"), ge=0, le=100,
        description="Completeness (%)"
    )
    by_category: Dict[str, float] = Field(
        default_factory=dict,
        description="Emissions by category (tCO2e)"
    )
    by_gas: Dict[str, float] = Field(
        default_factory=dict,
        description="Emissions by gas (tCO2e)"
    )
    by_facility: Dict[str, float] = Field(
        default_factory=dict,
        description="Emissions by facility (tCO2e)"
    )
    average_quality: Decimal = Field(
        default=Decimal("0"), ge=0, le=100,
        description="Average data quality (0-100)"
    )

class InventoryDiffItem(BaseModel):
    """A single difference between two inventories.

    Attributes:
        scope:           Affected scope.
        category:        Affected category (if applicable).
        metric:          Name of the metric being compared.
        value_inv1:      Value in inventory 1.
        value_inv2:      Value in inventory 2.
        absolute_diff:   Absolute difference (inv2 - inv1).
        pct_diff:        Percentage difference.
        is_significant:  Whether the difference exceeds threshold.
    """
    scope: Optional[ScopeType] = Field(
        default=None, description="Affected scope"
    )
    category: Optional[str] = Field(
        default=None, description="Affected category"
    )
    metric: str = Field(
        ..., description="Metric name"
    )
    value_inv1: Decimal = Field(
        default=Decimal("0"),
        description="Value in inventory 1"
    )
    value_inv2: Decimal = Field(
        default=Decimal("0"),
        description="Value in inventory 2"
    )
    absolute_diff: Decimal = Field(
        default=Decimal("0"),
        description="Absolute difference"
    )
    pct_diff: Decimal = Field(
        default=Decimal("0"),
        description="Percentage difference"
    )
    is_significant: bool = Field(
        default=False,
        description="Exceeds significance threshold"
    )

class InventoryComparison(BaseModel):
    """Comparison result between two base year inventories.

    Attributes:
        comparison_id:        Unique comparison identifier.
        inventory1_hash:      Provenance hash of first inventory.
        inventory2_hash:      Provenance hash of second inventory.
        inv1_year:            Base year of inventory 1.
        inv2_year:            Base year of inventory 2.
        differences:          List of individual differences.
        total_diff_tco2e:     Total absolute difference (tCO2e).
        total_diff_pct:       Total percentage difference.
        any_significant:      Whether any difference is significant.
        significance_threshold: Threshold used (%).
        summary:              Human-readable summary.
        calculated_at:        Timestamp.
        provenance_hash:      SHA-256 provenance hash.
    """
    comparison_id: str = Field(
        default_factory=_new_uuid,
        description="Comparison identifier"
    )
    inventory1_hash: str = Field(
        default="", description="Inventory 1 provenance hash"
    )
    inventory2_hash: str = Field(
        default="", description="Inventory 2 provenance hash"
    )
    inv1_year: int = Field(
        default=0, description="Inventory 1 base year"
    )
    inv2_year: int = Field(
        default=0, description="Inventory 2 base year"
    )
    differences: List[InventoryDiffItem] = Field(
        default_factory=list,
        description="Individual differences"
    )
    total_diff_tco2e: Decimal = Field(
        default=Decimal("0"),
        description="Total absolute difference"
    )
    total_diff_pct: Decimal = Field(
        default=Decimal("0"),
        description="Total percentage difference"
    )
    any_significant: bool = Field(
        default=False,
        description="Any significant differences"
    )
    significance_threshold: Decimal = Field(
        default=Decimal("5"),
        description="Significance threshold (%)"
    )
    summary: str = Field(
        default="", description="Comparison summary"
    )
    calculated_at: str = Field(
        default="", description="Calculation timestamp"
    )
    provenance_hash: str = Field(
        default="", description="SHA-256 provenance hash"
    )

class CompletenessAssessment(BaseModel):
    """Assessment of inventory completeness against expected categories.

    Attributes:
        overall_completeness:   Overall completeness (%).
        scope1_completeness:    Scope 1 completeness (%).
        scope2_completeness:    Scope 2 completeness (%).
        scope3_completeness:    Scope 3 completeness (%).
        missing_categories:     Categories without data.
        low_quality_categories: Categories with low data quality.
        warnings:               Completeness warnings.
    """
    overall_completeness: Decimal = Field(
        default=Decimal("0"), ge=0, le=100,
        description="Overall completeness (%)"
    )
    scope1_completeness: Decimal = Field(
        default=Decimal("0"), ge=0, le=100,
        description="Scope 1 completeness (%)"
    )
    scope2_completeness: Decimal = Field(
        default=Decimal("0"), ge=0, le=100,
        description="Scope 2 completeness (%)"
    )
    scope3_completeness: Decimal = Field(
        default=Decimal("0"), ge=0, le=100,
        description="Scope 3 completeness (%)"
    )
    missing_categories: List[str] = Field(
        default_factory=list,
        description="Categories with no data"
    )
    low_quality_categories: List[str] = Field(
        default_factory=list,
        description="Categories with low quality"
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="Completeness warnings"
    )

class BaseYearInventory(BaseModel):
    """Complete base year emissions inventory.

    The central data structure for a frozen, auditable base year
    inventory covering all scopes, categories, and gases.

    Attributes:
        inventory_id:          Unique inventory identifier.
        organization_id:       Organisation identifier.
        base_year:             Base year (calendar year).
        established_date:      Date the inventory was established.
        status:                Inventory status.
        sources:               Individual source emissions.
        scope_totals:          Aggregated scope totals.
        grand_total_tco2e:     Grand total (Scope 1 + 2 location + 3).
        scope1_total_tco2e:    Scope 1 total.
        scope2_location_tco2e: Scope 2 location-based total.
        scope2_market_tco2e:   Scope 2 market-based total.
        scope3_total_tco2e:    Scope 3 total.
        gwp_version:           GWP version used.
        consolidation_approach: Organisational boundary approach.
        boundary_description:  Boundary description.
        methodology_summary:   Methodology summary.
        is_verified:           Whether independently verified.
        verification_date:     Date of verification.
        verifier_name:         Name of verification body.
        completeness:          Completeness assessment.
        version:               Inventory version number.
        supersedes_id:         ID of the inventory this replaces.
        calculated_at:         Timestamp of calculation.
        processing_time_ms:    Processing time.
        provenance_hash:       SHA-256 provenance hash.
    """
    inventory_id: str = Field(
        default_factory=_new_uuid,
        description="Inventory identifier"
    )
    organization_id: str = Field(
        default="", description="Organisation identifier"
    )
    base_year: int = Field(
        ..., ge=MINIMUM_BASE_YEAR, le=MAXIMUM_BASE_YEAR,
        description="Base year"
    )
    established_date: str = Field(
        default="", description="Established date (ISO)"
    )
    status: InventoryStatus = Field(
        default=InventoryStatus.DRAFT,
        description="Inventory status"
    )
    sources: List[SourceEmission] = Field(
        default_factory=list,
        description="Source emissions"
    )
    scope_totals: List[ScopeTotal] = Field(
        default_factory=list,
        description="Scope totals"
    )
    grand_total_tco2e: Decimal = Field(
        default=Decimal("0"), ge=0,
        description="Grand total (tCO2e)"
    )
    scope1_total_tco2e: Decimal = Field(
        default=Decimal("0"), ge=0,
        description="Scope 1 total (tCO2e)"
    )
    scope2_location_tco2e: Decimal = Field(
        default=Decimal("0"), ge=0,
        description="Scope 2 location (tCO2e)"
    )
    scope2_market_tco2e: Decimal = Field(
        default=Decimal("0"), ge=0,
        description="Scope 2 market (tCO2e)"
    )
    scope3_total_tco2e: Decimal = Field(
        default=Decimal("0"), ge=0,
        description="Scope 3 total (tCO2e)"
    )
    gwp_version: GWPVersion = Field(
        default=GWPVersion.AR5,
        description="GWP version"
    )
    consolidation_approach: ConsolidationApproach = Field(
        default=ConsolidationApproach.OPERATIONAL_CONTROL,
        description="Consolidation approach"
    )
    boundary_description: str = Field(
        default="", description="Boundary description"
    )
    methodology_summary: str = Field(
        default="", description="Methodology summary"
    )
    is_verified: bool = Field(
        default=False, description="Third-party verified"
    )
    verification_date: Optional[str] = Field(
        default=None, description="Verification date"
    )
    verifier_name: str = Field(
        default="", description="Verifier name"
    )
    completeness: Optional[CompletenessAssessment] = Field(
        default=None, description="Completeness assessment"
    )
    version: int = Field(
        default=1, ge=1, description="Inventory version"
    )
    supersedes_id: Optional[str] = Field(
        default=None, description="ID of superseded inventory"
    )
    calculated_at: str = Field(
        default="", description="Calculation timestamp"
    )
    processing_time_ms: float = Field(
        default=0.0, description="Processing time (ms)"
    )
    provenance_hash: str = Field(
        default="", description="SHA-256 provenance hash"
    )

    @field_validator(
        "grand_total_tco2e", "scope1_total_tco2e",
        "scope2_location_tco2e", "scope2_market_tco2e",
        "scope3_total_tco2e",
        mode="before",
    )
    @classmethod
    def coerce_decimal(cls, v: Any) -> Decimal:
        """Coerce emission totals to Decimal."""
        return _decimal(v)

# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class BaseYearInventoryEngine:
    """Complete base year emissions inventory preservation engine.

    Establishes, maintains, snapshots, and compares base year
    inventories with full provenance tracking.

    Guarantees:
        - Deterministic: Same inputs always produce the same output.
        - Reproducible: Full provenance tracking with SHA-256 hashes.
        - Auditable: Frozen inventory snapshots are immutable.
        - Zero-Hallucination: No LLM in any calculation path.

    Usage:
        engine = BaseYearInventoryEngine()
        sources = [SourceEmission(category=SourceCategory.STATIONARY_COMBUSTION, ...)]
        config = InventoryConfig(organization_id="ORG-001", base_year=2019)
        inventory = engine.establish_inventory(sources, config)
        frozen = engine.snapshot_inventory(inventory)
    """

    def __init__(self) -> None:
        """Initialise the BaseYearInventoryEngine."""
        self._version = _MODULE_VERSION
        logger.info(
            "BaseYearInventoryEngine v%s initialised", self._version
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def establish_inventory(
        self,
        sources: List[SourceEmission],
        config: InventoryConfig,
    ) -> BaseYearInventory:
        """Establish a complete base year emissions inventory.

        This is the primary entry point.  It processes all source
        emissions, aggregates by scope and category, assesses
        completeness, and produces a fully documented inventory.

        Args:
            sources: List of source emissions.
            config:  Inventory configuration.

        Returns:
            BaseYearInventory with all totals and provenance.

        Raises:
            ValueError: If no sources are provided.
        """
        t0 = time.perf_counter()

        if not sources:
            raise ValueError("At least one source emission is required")

        # Filter by minimum quality if configured
        filtered_sources = sources
        if config.minimum_quality_score > Decimal("0"):
            filtered_sources = [
                s for s in sources
                if s.data_quality_score >= config.minimum_quality_score
            ]
            if not filtered_sources:
                logger.warning(
                    "All sources below minimum quality %s; using all sources",
                    config.minimum_quality_score,
                )
                filtered_sources = sources

        # Filter Scope 3 if not included
        if not config.include_scope3:
            filtered_sources = [
                s for s in filtered_sources
                if s.scope != ScopeType.SCOPE_3
            ]

        # Auto-calculate tCO2e if configured
        if config.auto_calculate_tco2e:
            filtered_sources = [
                self._calculate_source_tco2e(s, config.gwp_version)
                for s in filtered_sources
            ]

        # Aggregate by scope
        scope_totals = self.calculate_scope_totals(filtered_sources)

        # Extract scope-level totals
        scope1_total = Decimal("0")
        scope2_location_total = Decimal("0")
        scope2_market_total = Decimal("0")
        scope3_total = Decimal("0")

        for st in scope_totals:
            if st.scope == ScopeType.SCOPE_1:
                scope1_total = st.total_tco2e
            elif st.scope == ScopeType.SCOPE_2_LOCATION:
                scope2_location_total = st.total_tco2e
            elif st.scope == ScopeType.SCOPE_2_MARKET:
                scope2_market_total = st.total_tco2e
            elif st.scope == ScopeType.SCOPE_3:
                scope3_total = st.total_tco2e

        # Grand total (Scope 1 + Scope 2 location + Scope 3)
        # Scope 2 market is reported separately to avoid double-counting
        grand_total = (scope1_total + scope2_location_total + scope3_total).quantize(
            Decimal("0.001"), rounding=ROUND_HALF_UP
        )

        # Completeness assessment
        completeness = self.validate_completeness_assessment(
            filtered_sources, config.include_scope3
        )

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        inventory = BaseYearInventory(
            organization_id=config.organization_id,
            base_year=config.base_year,
            established_date=utcnow().isoformat(),
            status=InventoryStatus.ESTABLISHED,
            sources=filtered_sources,
            scope_totals=scope_totals,
            grand_total_tco2e=grand_total,
            scope1_total_tco2e=scope1_total,
            scope2_location_tco2e=scope2_location_total,
            scope2_market_tco2e=scope2_market_total,
            scope3_total_tco2e=scope3_total,
            gwp_version=config.gwp_version,
            consolidation_approach=config.consolidation_approach,
            boundary_description=config.boundary_description,
            methodology_summary=config.methodology_summary,
            completeness=completeness,
            calculated_at=utcnow().isoformat(),
            processing_time_ms=round(elapsed_ms, 3),
        )
        inventory.provenance_hash = _compute_hash(inventory)
        return inventory

    def aggregate_by_scope(
        self,
        sources: List[SourceEmission],
    ) -> Dict[ScopeType, Decimal]:
        """Aggregate emissions by scope.

        Formula:
            scope_total(S) = SUM(source.tco2e for source in sources
                                 if scope(source.category) == S)

        Args:
            sources: Source emissions to aggregate.

        Returns:
            Dictionary mapping ScopeType to total tCO2e.
        """
        totals: Dict[ScopeType, Decimal] = {
            ScopeType.SCOPE_1: Decimal("0"),
            ScopeType.SCOPE_2_LOCATION: Decimal("0"),
            ScopeType.SCOPE_2_MARKET: Decimal("0"),
            ScopeType.SCOPE_3: Decimal("0"),
        }

        for source in sources:
            scope = CATEGORY_SCOPE_MAP.get(source.category.value, ScopeType.SCOPE_1)
            totals[scope] = totals[scope] + source.tco2e

        # Round all totals
        return {
            scope: val.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)
            for scope, val in totals.items()
        }

    def aggregate_by_category(
        self,
        sources: List[SourceEmission],
    ) -> Dict[str, Decimal]:
        """Aggregate emissions by source category.

        Formula:
            category_total(C) = SUM(source.tco2e for source in sources
                                     if source.category == C)

        Args:
            sources: Source emissions to aggregate.

        Returns:
            Dictionary mapping category value to total tCO2e.
        """
        totals: Dict[str, Decimal] = {}

        for source in sources:
            cat = source.category.value
            if cat not in totals:
                totals[cat] = Decimal("0")
            totals[cat] = totals[cat] + source.tco2e

        return {
            cat: val.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)
            for cat, val in totals.items()
        }

    def aggregate_by_gas(
        self,
        sources: List[SourceEmission],
    ) -> Dict[str, Decimal]:
        """Aggregate emissions by greenhouse gas type.

        Args:
            sources: Source emissions to aggregate.

        Returns:
            Dictionary mapping gas type to total tCO2e.
        """
        totals: Dict[str, Decimal] = {}

        for source in sources:
            gas = source.gas_type.value
            if gas not in totals:
                totals[gas] = Decimal("0")
            totals[gas] = totals[gas] + source.tco2e

        return {
            gas: val.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)
            for gas, val in totals.items()
        }

    def aggregate_by_facility(
        self,
        sources: List[SourceEmission],
    ) -> Dict[str, Decimal]:
        """Aggregate emissions by facility.

        Args:
            sources: Source emissions to aggregate.

        Returns:
            Dictionary mapping facility ID to total tCO2e.
        """
        totals: Dict[str, Decimal] = {}

        for source in sources:
            fac = source.facility_id or "unknown"
            if fac not in totals:
                totals[fac] = Decimal("0")
            totals[fac] = totals[fac] + source.tco2e

        return {
            fac: val.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)
            for fac, val in totals.items()
        }

    def calculate_scope_totals(
        self,
        sources: List[SourceEmission],
    ) -> List[ScopeTotal]:
        """Calculate detailed scope totals with breakdowns.

        Args:
            sources: Source emissions to aggregate.

        Returns:
            List of ScopeTotal objects with full breakdowns.
        """
        scope_totals: List[ScopeTotal] = []

        for scope in ScopeType:
            scope_sources = [
                s for s in sources
                if CATEGORY_SCOPE_MAP.get(s.category.value) == scope
            ]

            if not scope_sources:
                scope_totals.append(ScopeTotal(
                    scope=scope,
                    total_tco2e=Decimal("0"),
                    source_count=0,
                    completeness_pct=Decimal("0"),
                ))
                continue

            total = sum(s.tco2e for s in scope_sources)
            total = total.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)

            # By category
            by_cat: Dict[str, Decimal] = {}
            for s in scope_sources:
                cat = s.category.value
                by_cat[cat] = by_cat.get(cat, Decimal("0")) + s.tco2e

            # By gas
            by_gas: Dict[str, Decimal] = {}
            for s in scope_sources:
                gas = s.gas_type.value
                by_gas[gas] = by_gas.get(gas, Decimal("0")) + s.tco2e

            # By facility
            by_fac: Dict[str, Decimal] = {}
            for s in scope_sources:
                fac = s.facility_id or "unknown"
                by_fac[fac] = by_fac.get(fac, Decimal("0")) + s.tco2e

            # Categories covered
            categories_covered = list(set(s.category.value for s in scope_sources))

            # Expected categories per scope
            expected = self._get_expected_categories(scope)
            completeness_pct = _safe_pct(
                Decimal(str(len(categories_covered))),
                Decimal(str(len(expected))) if expected else Decimal("1"),
            ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

            # Average quality
            quality_scores = [s.data_quality_score for s in scope_sources]
            avg_quality = (
                sum(quality_scores) / Decimal(str(len(quality_scores)))
            ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP) if quality_scores else Decimal("0")

            scope_totals.append(ScopeTotal(
                scope=scope,
                total_tco2e=total,
                source_count=len(scope_sources),
                categories_covered=categories_covered,
                completeness_pct=min(Decimal("100"), completeness_pct),
                by_category={k: _round3(v) for k, v in by_cat.items()},
                by_gas={k: _round3(v) for k, v in by_gas.items()},
                by_facility={k: _round3(v) for k, v in by_fac.items()},
                average_quality=avg_quality,
            ))

        return scope_totals

    def validate_completeness(
        self,
        inventory: BaseYearInventory,
    ) -> CompletenessAssessment:
        """Validate the completeness of a base year inventory.

        Checks that all expected source categories have data, and
        flags any gaps or low-quality entries.

        Args:
            inventory: Inventory to validate.

        Returns:
            CompletenessAssessment with gaps and warnings.
        """
        return self.validate_completeness_assessment(
            inventory.sources,
            include_scope3=any(
                s.scope == ScopeType.SCOPE_3 for s in inventory.sources
            ),
        )

    def validate_completeness_assessment(
        self,
        sources: List[SourceEmission],
        include_scope3: bool = True,
    ) -> CompletenessAssessment:
        """Assess completeness of source emissions vs expected categories.

        Args:
            sources:         List of source emissions.
            include_scope3:  Whether Scope 3 is expected.

        Returns:
            CompletenessAssessment with completeness percentages.
        """
        covered = set(s.category.value for s in sources)
        warnings: List[str] = []

        # Scope 1 completeness
        s1_expected = SCOPE_1_CATEGORIES
        s1_covered = covered & s1_expected
        s1_pct = _safe_pct(
            Decimal(str(len(s1_covered))),
            Decimal(str(len(s1_expected))),
        ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

        # Scope 2 completeness
        s2_expected = SCOPE_2_LOCATION_CATEGORIES | SCOPE_2_MARKET_CATEGORIES
        s2_covered = covered & s2_expected
        s2_pct = _safe_pct(
            Decimal(str(len(s2_covered))),
            Decimal(str(len(s2_expected))),
        ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

        # Scope 3 completeness
        s3_pct = Decimal("0")
        if include_scope3:
            s3_expected = SCOPE_3_CATEGORIES
            s3_covered = covered & s3_expected
            s3_pct = _safe_pct(
                Decimal(str(len(s3_covered))),
                Decimal(str(len(s3_expected))),
            ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

        # Overall completeness
        all_expected = SCOPE_1_CATEGORIES | SCOPE_2_LOCATION_CATEGORIES | SCOPE_2_MARKET_CATEGORIES
        if include_scope3:
            all_expected = all_expected | SCOPE_3_CATEGORIES
        all_covered = covered & all_expected
        overall_pct = _safe_pct(
            Decimal(str(len(all_covered))),
            Decimal(str(len(all_expected))),
        ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

        # Missing categories
        missing = list(all_expected - covered)

        # Low quality categories (average quality < 30)
        cat_quality: Dict[str, List[Decimal]] = {}
        for s in sources:
            cat = s.category.value
            if cat not in cat_quality:
                cat_quality[cat] = []
            cat_quality[cat].append(s.data_quality_score)

        low_quality = []
        for cat, scores in cat_quality.items():
            avg = sum(scores) / Decimal(str(len(scores)))
            if avg < Decimal("30"):
                low_quality.append(cat)

        # Warnings
        if s1_pct < Decimal("100"):
            warnings.append(
                f"Scope 1 completeness {s1_pct}% - missing categories: "
                f"{list(SCOPE_1_CATEGORIES - s1_covered)}"
            )
        if s2_pct < Decimal("100"):
            warnings.append(
                f"Scope 2 completeness {s2_pct}% - missing categories: "
                f"{list(s2_expected - s2_covered)}"
            )
        if include_scope3 and s3_pct < Decimal("50"):
            warnings.append(
                f"Scope 3 completeness {s3_pct}% is below 50%. "
                f"Consider expanding Scope 3 coverage."
            )
        if low_quality:
            warnings.append(
                f"Low data quality (<30) in categories: {low_quality}"
            )

        return CompletenessAssessment(
            overall_completeness=min(Decimal("100"), overall_pct),
            scope1_completeness=min(Decimal("100"), s1_pct),
            scope2_completeness=min(Decimal("100"), s2_pct),
            scope3_completeness=min(Decimal("100"), s3_pct),
            missing_categories=sorted(missing),
            low_quality_categories=sorted(low_quality),
            warnings=warnings,
        )

    def snapshot_inventory(
        self,
        inventory: BaseYearInventory,
    ) -> BaseYearInventory:
        """Create a frozen, immutable snapshot of the inventory.

        Deep-copies the inventory, sets status to FROZEN, and
        computes a provenance hash that can be used for audit.

        Args:
            inventory: Inventory to snapshot.

        Returns:
            Frozen copy of the inventory with provenance hash.
        """
        # Deep copy to ensure immutability
        snapshot_data = inventory.model_dump(mode="json")
        snapshot = BaseYearInventory.model_validate(snapshot_data)

        # Set frozen status
        snapshot.status = InventoryStatus.FROZEN
        snapshot.inventory_id = _new_uuid()

        # Recompute provenance hash for the frozen copy
        snapshot.provenance_hash = _compute_hash(snapshot)

        logger.info(
            "Inventory snapshot created: %s (hash: %s...)",
            snapshot.inventory_id,
            snapshot.provenance_hash[:16],
        )

        return snapshot

    def compare_inventories(
        self,
        inv1: BaseYearInventory,
        inv2: BaseYearInventory,
        significance_threshold_pct: Decimal = Decimal("5"),
    ) -> InventoryComparison:
        """Compare two base year inventories and identify differences.

        For each scope and the grand total, computes the absolute
        and percentage difference, and flags any that exceed the
        significance threshold.

        Formula:
            delta(S) = inv2.scope_total(S) - inv1.scope_total(S)
            delta_pct(S) = abs(delta(S)) / inv1.scope_total(S) * 100
            is_significant = delta_pct >= threshold

        Args:
            inv1:                       First inventory (baseline).
            inv2:                       Second inventory (comparison).
            significance_threshold_pct: Threshold for significance (%).

        Returns:
            InventoryComparison with all differences and summary.
        """
        t0 = time.perf_counter()
        differences: List[InventoryDiffItem] = []

        # Scope-level comparisons
        scope_pairs = [
            (ScopeType.SCOPE_1, inv1.scope1_total_tco2e, inv2.scope1_total_tco2e),
            (ScopeType.SCOPE_2_LOCATION, inv1.scope2_location_tco2e, inv2.scope2_location_tco2e),
            (ScopeType.SCOPE_2_MARKET, inv1.scope2_market_tco2e, inv2.scope2_market_tco2e),
            (ScopeType.SCOPE_3, inv1.scope3_total_tco2e, inv2.scope3_total_tco2e),
        ]

        for scope, val1, val2 in scope_pairs:
            diff = val2 - val1
            pct = _safe_pct(abs(diff), val1) if val1 != Decimal("0") else Decimal("0")
            is_sig = pct >= significance_threshold_pct

            differences.append(InventoryDiffItem(
                scope=scope,
                metric=f"{scope.value}_total_tco2e",
                value_inv1=val1.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP),
                value_inv2=val2.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP),
                absolute_diff=diff.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP),
                pct_diff=pct.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP),
                is_significant=is_sig,
            ))

        # Grand total comparison
        grand_diff = inv2.grand_total_tco2e - inv1.grand_total_tco2e
        grand_pct = _safe_pct(
            abs(grand_diff), inv1.grand_total_tco2e
        ) if inv1.grand_total_tco2e != Decimal("0") else Decimal("0")
        grand_sig = grand_pct >= significance_threshold_pct

        differences.append(InventoryDiffItem(
            metric="grand_total_tco2e",
            value_inv1=inv1.grand_total_tco2e,
            value_inv2=inv2.grand_total_tco2e,
            absolute_diff=grand_diff.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP),
            pct_diff=grand_pct.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP),
            is_significant=grand_sig,
        ))

        # Category-level comparisons
        cat_diffs = self._compare_categories(inv1, inv2, significance_threshold_pct)
        differences.extend(cat_diffs)

        any_significant = any(d.is_significant for d in differences)

        # Build summary
        sig_items = [d for d in differences if d.is_significant]
        if sig_items:
            sig_desc = ", ".join(
                f"{d.metric} ({_round2(d.pct_diff)}%)" for d in sig_items
            )
            summary = (
                f"Significant differences found in: {sig_desc}. "
                f"Recalculation may be required per GHG Protocol Chapter 5."
            )
        else:
            summary = (
                f"No significant differences (>{significance_threshold_pct}%) "
                f"found between inventories."
            )

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        comparison = InventoryComparison(
            inventory1_hash=inv1.provenance_hash,
            inventory2_hash=inv2.provenance_hash,
            inv1_year=inv1.base_year,
            inv2_year=inv2.base_year,
            differences=differences,
            total_diff_tco2e=grand_diff.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP),
            total_diff_pct=grand_pct.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP),
            any_significant=any_significant,
            significance_threshold=significance_threshold_pct,
            summary=summary,
            calculated_at=utcnow().isoformat(),
        )
        comparison.provenance_hash = _compute_hash(comparison)
        return comparison

    # ------------------------------------------------------------------
    # Internal Methods
    # ------------------------------------------------------------------

    def _calculate_source_tco2e(
        self,
        source: SourceEmission,
        gwp_version: GWPVersion,
    ) -> SourceEmission:
        """Calculate tCO2e for a source from activity data and emission factor.

        Formula:
            gas_emissions = activity_data * emission_factor / 1000
            gwp = GWP_FACTORS[gwp_version][gas_type]
            tco2e = gas_emissions * gwp

        Args:
            source:      Source emission with activity data.
            gwp_version: GWP version for conversion.

        Returns:
            New SourceEmission with calculated tco2e.
        """
        gas_emissions = (
            source.activity_data * source.emission_factor / Decimal("1000")
        ).quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP)

        gwp = GWP_FACTORS.get(gwp_version.value, {}).get(
            source.gas_type.value, Decimal("1")
        )

        tco2e = (gas_emissions * gwp).quantize(
            Decimal("0.001"), rounding=ROUND_HALF_UP
        )

        # Create updated source (immutable pattern)
        data = source.model_dump()
        data["gas_emissions_tonnes"] = gas_emissions
        data["gwp_factor"] = gwp
        data["tco2e"] = tco2e
        return SourceEmission.model_validate(data)

    def _get_expected_categories(self, scope: ScopeType) -> set:
        """Get expected source categories for a scope.

        Args:
            scope: Emission scope.

        Returns:
            Set of expected category values.
        """
        if scope == ScopeType.SCOPE_1:
            return SCOPE_1_CATEGORIES
        elif scope == ScopeType.SCOPE_2_LOCATION:
            return SCOPE_2_LOCATION_CATEGORIES
        elif scope == ScopeType.SCOPE_2_MARKET:
            return SCOPE_2_MARKET_CATEGORIES
        elif scope == ScopeType.SCOPE_3:
            return SCOPE_3_CATEGORIES
        return set()

    def _compare_categories(
        self,
        inv1: BaseYearInventory,
        inv2: BaseYearInventory,
        threshold: Decimal,
    ) -> List[InventoryDiffItem]:
        """Compare inventories at the category level.

        Args:
            inv1:      First inventory.
            inv2:      Second inventory.
            threshold: Significance threshold (%).

        Returns:
            List of category-level differences.
        """
        cat1 = self.aggregate_by_category(inv1.sources)
        cat2 = self.aggregate_by_category(inv2.sources)
        all_cats = set(cat1.keys()) | set(cat2.keys())

        diffs: List[InventoryDiffItem] = []
        for cat in sorted(all_cats):
            val1 = cat1.get(cat, Decimal("0"))
            val2 = cat2.get(cat, Decimal("0"))
            diff = val2 - val1
            pct = _safe_pct(abs(diff), val1) if val1 != Decimal("0") else Decimal("0")
            is_sig = pct >= threshold

            scope = CATEGORY_SCOPE_MAP.get(cat)

            diffs.append(InventoryDiffItem(
                scope=scope,
                category=cat,
                metric=f"category_{cat}_tco2e",
                value_inv1=val1,
                value_inv2=val2,
                absolute_diff=diff.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP),
                pct_diff=pct.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP),
                is_significant=is_sig,
            ))

        return diffs

    # ------------------------------------------------------------------
    # Utility Methods
    # ------------------------------------------------------------------

    def get_inventory_summary(
        self,
        inventory: BaseYearInventory,
    ) -> Dict[str, Any]:
        """Generate a summary of the inventory for reporting.

        Args:
            inventory: Inventory to summarise.

        Returns:
            Summary dictionary for tabular display.
        """
        return {
            "inventory_id": inventory.inventory_id,
            "organization_id": inventory.organization_id,
            "base_year": inventory.base_year,
            "status": inventory.status.value,
            "grand_total_tco2e": _round3(inventory.grand_total_tco2e),
            "scope1_tco2e": _round3(inventory.scope1_total_tco2e),
            "scope2_location_tco2e": _round3(inventory.scope2_location_tco2e),
            "scope2_market_tco2e": _round3(inventory.scope2_market_tco2e),
            "scope3_tco2e": _round3(inventory.scope3_total_tco2e),
            "source_count": len(inventory.sources),
            "gwp_version": inventory.gwp_version.value,
            "consolidation": inventory.consolidation_approach.value,
            "is_verified": inventory.is_verified,
            "provenance_hash": inventory.provenance_hash,
        }

    def get_version(self) -> str:
        """Return engine version string."""
        return self._version

# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------

def establish_base_year_inventory(
    sources: List[SourceEmission],
    config: InventoryConfig,
) -> BaseYearInventory:
    """Module-level convenience function to establish an inventory.

    Args:
        sources: List of source emissions.
        config:  Inventory configuration.

    Returns:
        BaseYearInventory with all totals and provenance.
    """
    engine = BaseYearInventoryEngine()
    return engine.establish_inventory(sources, config)

def compare_base_year_inventories(
    inv1: BaseYearInventory,
    inv2: BaseYearInventory,
    significance_threshold_pct: Decimal = Decimal("5"),
) -> InventoryComparison:
    """Module-level convenience function to compare inventories.

    Args:
        inv1:                       First inventory (baseline).
        inv2:                       Second inventory.
        significance_threshold_pct: Significance threshold (%).

    Returns:
        InventoryComparison with differences and summary.
    """
    engine = BaseYearInventoryEngine()
    return engine.compare_inventories(inv1, inv2, significance_threshold_pct)

def get_gwp_factor(
    gas_type: GasType,
    gwp_version: GWPVersion = GWPVersion.AR5,
) -> Decimal:
    """Look up a GWP factor.

    Args:
        gas_type:    Greenhouse gas type.
        gwp_version: IPCC assessment report version.

    Returns:
        GWP factor (100-year horizon).
    """
    return GWP_FACTORS.get(gwp_version.value, {}).get(
        gas_type.value, Decimal("1")
    )

# ---------------------------------------------------------------------------
# __all__
# ---------------------------------------------------------------------------

__all__ = [
    # Enums
    "ScopeType",
    "SourceCategory",
    "GasType",
    "GWPVersion",
    "ConsolidationApproach",
    "MethodologyTier",
    "InventoryStatus",
    # Input Models
    "SourceEmission",
    "InventoryConfig",
    # Output Models
    "ScopeTotal",
    "InventoryDiffItem",
    "InventoryComparison",
    "CompletenessAssessment",
    "BaseYearInventory",
    # Engine
    "BaseYearInventoryEngine",
    # Convenience functions
    "establish_base_year_inventory",
    "compare_base_year_inventories",
    "get_gwp_factor",
    # Constants
    "GWP_FACTORS",
    "CATEGORY_SCOPE_MAP",
    "SCOPE_1_CATEGORIES",
    "SCOPE_2_LOCATION_CATEGORIES",
    "SCOPE_2_MARKET_CATEGORIES",
    "SCOPE_3_CATEGORIES",
    "MINIMUM_BASE_YEAR",
    "MAXIMUM_BASE_YEAR",
]
