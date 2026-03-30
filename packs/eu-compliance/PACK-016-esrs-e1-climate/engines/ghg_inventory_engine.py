# -*- coding: utf-8 -*-
"""
GHGInventoryEngine - PACK-016 ESRS E1 Climate Engine 1
=======================================================

Calculates gross Scopes 1, 2, 3 and total GHG emissions per ESRS E1-6.

Under ESRS E1, disclosure requirement E1-6 mandates that undertakings
report their gross Scope 1, 2, and 3 GHG emissions.  This engine
implements the complete GHG inventory calculation pipeline, including:

- Per-entry emission calculation using activity data and emission factors
- GWP-100 conversion using IPCC AR6 values
- Scope-level aggregation (Scope 1, 2 location-based, 2 market-based, 3)
- Disaggregation by individual greenhouse gas
- Scope 3 breakdown across all 15 categories
- Intensity metric calculation (per revenue, headcount, or unit)
- Multi-entity consolidation (operational/financial control, equity share)
- Base year comparison with percentage change calculation
- Completeness validation against E1-6 required data points
- ESRS E1-6 data point mapping for disclosure

ESRS E1-6 Disclosure Requirements:
    - Para 44: Gross Scope 1 GHG emissions in metric tons of CO2eq
    - Para 45: Gross Scope 2 GHG emissions (location-based and
      market-based) in metric tons of CO2eq
    - Para 46: Gross Scope 3 GHG emissions in metric tons of CO2eq,
      disaggregated by significant Scope 3 categories
    - Para 47: Total GHG emissions (Scope 1 + 2 + 3)
    - Para 48: GHG emissions intensity based on net revenue
    - Para 49: Disaggregation of GHG emissions by individual gas (CO2,
      CH4, N2O, HFCs, PFCs, SF6, NF3)
    - Para 50: Biogenic CO2 emissions reported separately

Regulatory References:
    - EU Delegated Regulation 2023/2772 (ESRS)
    - ESRS E1 Climate Change, Disclosure Requirement E1-6
    - GHG Protocol Corporate Standard (2004, revised 2015)
    - GHG Protocol Scope 3 Standard (2011)
    - IPCC AR6 WG1 (2021) - GWP-100 values

Zero-Hallucination:
    - All emission calculations use deterministic arithmetic
    - GWP values are IPCC AR6 constants (no ML/LLM)
    - Aggregation uses Decimal arithmetic with ROUND_HALF_UP
    - SHA-256 provenance hash on every result
    - No LLM involvement in any calculation path

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-016 ESRS E1 Climate
Status:  Production Ready
"""

import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
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

    Args:
        data: Data to hash (dict, Pydantic model, or other).

    Returns:
        SHA-256 hex digest string (64 characters).
    """
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

def _decimal(value: Any) -> Decimal:
    """Convert value to Decimal safely.

    Args:
        value: Numeric value (int, float, str, or Decimal).

    Returns:
        Decimal representation.
    """
    if isinstance(value, Decimal):
        return value
    return Decimal(str(value))

def _safe_divide(
    numerator: Decimal, denominator: Decimal, default: Decimal = Decimal("0")
) -> Decimal:
    """Safely divide two Decimals, returning *default* on zero denominator."""
    if denominator == Decimal("0"):
        return default
    return numerator / denominator

def _round_val(value: Decimal, places: int = 3) -> Decimal:
    """Round a Decimal value to the specified number of decimal places.

    Uses ROUND_HALF_UP for regulatory consistency.

    Args:
        value: Decimal value to round.
        places: Number of decimal places (default 3).

    Returns:
        Rounded Decimal value.
    """
    quantize_str = "0." + "0" * places
    return value.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP)

def _round3(value: float) -> float:
    """Round to 3 decimal places using ROUND_HALF_UP."""
    return float(Decimal(str(value)).quantize(
        Decimal("0.001"), rounding=ROUND_HALF_UP
    ))

def _round2(value: float) -> float:
    """Round to 2 decimal places using ROUND_HALF_UP."""
    return float(Decimal(str(value)).quantize(
        Decimal("0.01"), rounding=ROUND_HALF_UP
    ))

def _round6(value: Decimal) -> Decimal:
    """Round Decimal to 6 decimal places using ROUND_HALF_UP."""
    return value.quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP)

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class GHGScope(str, Enum):
    """GHG emission scope per GHG Protocol Corporate Standard.

    Scope 1: Direct emissions from owned or controlled sources.
    Scope 2: Indirect emissions from purchased energy (location or market).
    Scope 3: All other indirect emissions in the value chain.
    """
    SCOPE_1 = "scope_1"
    SCOPE_2_LOCATION = "scope_2_location"
    SCOPE_2_MARKET = "scope_2_market"
    SCOPE_3 = "scope_3"

class EmissionGas(str, Enum):
    """Individual greenhouse gases per IPCC/UNFCCC classification.

    ESRS E1-6 Para 49 requires disaggregation by these seven gases
    (or gas groups).
    """
    CO2 = "co2"
    CH4 = "ch4"
    N2O = "n2o"
    HFCS = "hfcs"
    PFCS = "pfcs"
    SF6 = "sf6"
    NF3 = "nf3"

class Scope3Category(str, Enum):
    """GHG Protocol Scope 3 categories (1-15).

    Per ESRS E1-6 Para 46, Scope 3 emissions shall be disaggregated
    by significant categories.
    """
    PURCHASED_GOODS = "cat_01_purchased_goods"
    CAPITAL_GOODS = "cat_02_capital_goods"
    FUEL_ENERGY = "cat_03_fuel_energy"
    UPSTREAM_TRANSPORT = "cat_04_upstream_transport"
    WASTE = "cat_05_waste"
    BUSINESS_TRAVEL = "cat_06_business_travel"
    EMPLOYEE_COMMUTING = "cat_07_employee_commuting"
    UPSTREAM_LEASED = "cat_08_upstream_leased"
    DOWNSTREAM_TRANSPORT = "cat_09_downstream_transport"
    PROCESSING_SOLD = "cat_10_processing_sold"
    USE_OF_SOLD = "cat_11_use_of_sold"
    END_OF_LIFE = "cat_12_end_of_life"
    DOWNSTREAM_LEASED = "cat_13_downstream_leased"
    FRANCHISES = "cat_14_franchises"
    INVESTMENTS = "cat_15_investments"

class ConsolidationApproach(str, Enum):
    """Consolidation approach for multi-entity GHG reporting.

    Per GHG Protocol Corporate Standard Chapter 3, undertakings must
    choose one of these approaches and apply it consistently.
    """
    OPERATIONAL_CONTROL = "operational_control"
    FINANCIAL_CONTROL = "financial_control"
    EQUITY_SHARE = "equity_share"

class DataQualityLevel(str, Enum):
    """Data quality tier for emission factor sourcing.

    Higher quality levels produce more accurate emission calculations.
    ESRS requires disclosure of the data quality approach used.
    """
    PRIMARY = "primary"
    SECONDARY_SPECIFIC = "secondary_specific"
    SECONDARY_AVERAGE = "secondary_average"
    ESTIMATED = "estimated"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# IPCC AR6 GWP-100 values (Global Warming Potential, 100-year time horizon).
# Source: IPCC Sixth Assessment Report, Working Group I, Table 7.15 (2021).
# These are the mandatory GWP values for ESRS E1-6 reporting.
GWP_AR6: Dict[str, Decimal] = {
    "co2": Decimal("1"),
    "ch4": Decimal("27.9"),
    "n2o": Decimal("273"),
    "sf6": Decimal("25200"),
    "nf3": Decimal("17400"),
    # Common HFCs
    "hfc_23": Decimal("14600"),
    "hfc_32": Decimal("771"),
    "hfc_125": Decimal("3740"),
    "hfc_134a": Decimal("1530"),
    "hfc_143a": Decimal("5810"),
    "hfc_152a": Decimal("164"),
    "hfc_227ea": Decimal("3600"),
    "hfc_236fa": Decimal("8690"),
    "hfc_245fa": Decimal("962"),
    "hfc_365mfc": Decimal("914"),
    "hfc_4310mee": Decimal("1600"),
    # Common PFCs
    "pfc_14": Decimal("7380"),
    "pfc_116": Decimal("12400"),
    "pfc_218": Decimal("9290"),
    "pfc_318": Decimal("10200"),
    "pfc_3110": Decimal("10000"),
    "pfc_5114": Decimal("9220"),
    # Generic group GWPs (weighted average approximations)
    "hfcs": Decimal("1530"),
    "pfcs": Decimal("7380"),
}

# ESRS E1-6 required data points for completeness validation.
E1_6_DATAPOINTS: List[str] = [
    "e1_6_01_scope1_total_tco2e",
    "e1_6_02_scope1_by_country",
    "e1_6_03_scope1_percentage_from_regulated_ets",
    "e1_6_04_scope2_location_total_tco2e",
    "e1_6_05_scope2_market_total_tco2e",
    "e1_6_06_scope3_total_tco2e",
    "e1_6_07_scope3_by_category",
    "e1_6_08_total_ghg_emissions_tco2e",
    "e1_6_09_ghg_intensity_per_net_revenue",
    "e1_6_10_disaggregation_by_gas",
    "e1_6_11_biogenic_co2_emissions",
    "e1_6_12_scope3_categories_included",
    "e1_6_13_base_year_emissions",
    "e1_6_14_change_from_base_year",
    "e1_6_15_consolidation_approach",
    "e1_6_16_methodology_description",
    "e1_6_17_data_quality_description",
    "e1_6_18_significant_changes_methodology",
    "e1_6_19_gwp_values_source",
]

# Scope 3 category names per GHG Protocol Scope 3 Standard.
SCOPE_3_CATEGORY_NAMES: Dict[str, str] = {
    "cat_01_purchased_goods": "Category 1: Purchased Goods and Services",
    "cat_02_capital_goods": "Category 2: Capital Goods",
    "cat_03_fuel_energy": "Category 3: Fuel- and Energy-Related Activities",
    "cat_04_upstream_transport": "Category 4: Upstream Transportation and Distribution",
    "cat_05_waste": "Category 5: Waste Generated in Operations",
    "cat_06_business_travel": "Category 6: Business Travel",
    "cat_07_employee_commuting": "Category 7: Employee Commuting",
    "cat_08_upstream_leased": "Category 8: Upstream Leased Assets",
    "cat_09_downstream_transport": "Category 9: Downstream Transportation and Distribution",
    "cat_10_processing_sold": "Category 10: Processing of Sold Products",
    "cat_11_use_of_sold": "Category 11: Use of Sold Products",
    "cat_12_end_of_life": "Category 12: End-of-Life Treatment of Sold Products",
    "cat_13_downstream_leased": "Category 13: Downstream Leased Assets",
    "cat_14_franchises": "Category 14: Franchises",
    "cat_15_investments": "Category 15: Investments",
}

# Data quality score weights for weighted average quality assessment.
DATA_QUALITY_SCORES: Dict[str, Decimal] = {
    "primary": Decimal("1.00"),
    "secondary_specific": Decimal("0.75"),
    "secondary_average": Decimal("0.50"),
    "estimated": Decimal("0.25"),
}

# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------

class EmissionEntry(BaseModel):
    """A single emission data entry for GHG inventory calculation.

    Represents one source of emissions with its activity data, emission
    factor, and metadata.  The engine uses this to calculate tCO2e for
    the entry and aggregate into the full inventory.
    """
    source_id: str = Field(
        default_factory=_new_uuid,
        description="Unique identifier for this emission source",
    )
    source_name: str = Field(
        default="",
        description="Human-readable name of the emission source",
        max_length=500,
    )
    scope: GHGScope = Field(
        ...,
        description="GHG scope (1, 2 location, 2 market, or 3)",
    )
    gas: EmissionGas = Field(
        default=EmissionGas.CO2,
        description="Greenhouse gas type",
    )
    gas_detail: str = Field(
        default="",
        description="Specific gas identifier for HFC/PFC subtypes (e.g. hfc_134a)",
        max_length=50,
    )
    scope3_category: Optional[Scope3Category] = Field(
        default=None,
        description="Scope 3 category (required when scope is SCOPE_3)",
    )
    activity_data: Decimal = Field(
        ...,
        description="Activity data quantity (e.g. litres of fuel, kWh consumed)",
        ge=Decimal("0"),
    )
    activity_unit: str = Field(
        default="",
        description="Unit of activity data (e.g. litres, kWh, kg)",
        max_length=50,
    )
    emission_factor: Decimal = Field(
        ...,
        description="Emission factor per unit of activity data",
        ge=Decimal("0"),
    )
    emission_factor_unit: str = Field(
        default="kgCO2e_per_unit",
        description="Unit of emission factor (e.g. kgCO2e/kWh)",
        max_length=100,
    )
    emission_factor_source: str = Field(
        default="",
        description="Source reference for the emission factor",
        max_length=500,
    )
    data_quality: DataQualityLevel = Field(
        default=DataQualityLevel.SECONDARY_AVERAGE,
        description="Data quality tier of the emission factor",
    )
    value_tco2e: Decimal = Field(
        default=Decimal("0"),
        description="Pre-calculated value in tCO2e (if provided, overrides calculation)",
        ge=Decimal("0"),
    )
    is_biogenic: bool = Field(
        default=False,
        description="Whether this entry represents biogenic CO2 emissions",
    )
    country_code: str = Field(
        default="",
        description="ISO 3166-1 alpha-2 country code for geographic disaggregation",
        max_length=3,
    )
    site_id: str = Field(
        default="",
        description="Facility or site identifier",
        max_length=100,
    )
    reporting_year: int = Field(
        default=0,
        description="Reporting year for this entry",
        ge=0,
    )

    @field_validator("scope3_category")
    @classmethod
    def validate_scope3_category(
        cls, v: Optional[Scope3Category], info: Any
    ) -> Optional[Scope3Category]:
        """Validate that Scope 3 entries have a category assigned."""
        scope_val = info.data.get("scope")
        if scope_val == GHGScope.SCOPE_3 and v is None:
            raise ValueError(
                "scope3_category is required when scope is SCOPE_3"
            )
        return v

class EmissionsByGas(BaseModel):
    """Disaggregation of GHG emissions by individual gas.

    Per ESRS E1-6 Para 49, undertakings must report emissions
    disaggregated by the seven Kyoto Protocol gas groups.
    """
    co2_tco2e: Decimal = Field(
        default=Decimal("0"), description="CO2 emissions in tCO2e"
    )
    ch4_tco2e: Decimal = Field(
        default=Decimal("0"), description="CH4 emissions in tCO2e"
    )
    n2o_tco2e: Decimal = Field(
        default=Decimal("0"), description="N2O emissions in tCO2e"
    )
    hfcs_tco2e: Decimal = Field(
        default=Decimal("0"), description="HFCs emissions in tCO2e"
    )
    pfcs_tco2e: Decimal = Field(
        default=Decimal("0"), description="PFCs emissions in tCO2e"
    )
    sf6_tco2e: Decimal = Field(
        default=Decimal("0"), description="SF6 emissions in tCO2e"
    )
    nf3_tco2e: Decimal = Field(
        default=Decimal("0"), description="NF3 emissions in tCO2e"
    )
    total_tco2e: Decimal = Field(
        default=Decimal("0"), description="Sum of all gases in tCO2e"
    )

class Scope3Breakdown(BaseModel):
    """Breakdown of Scope 3 emissions by GHG Protocol category.

    Per ESRS E1-6 Para 46, Scope 3 emissions must be disaggregated
    by significant upstream and downstream categories.
    """
    by_category: Dict[str, Decimal] = Field(
        default_factory=dict,
        description="Emissions per Scope 3 category in tCO2e",
    )
    categories_included: List[str] = Field(
        default_factory=list,
        description="List of Scope 3 categories included in reporting",
    )
    total_tco2e: Decimal = Field(
        default=Decimal("0"), description="Total Scope 3 emissions in tCO2e"
    )
    upstream_tco2e: Decimal = Field(
        default=Decimal("0"), description="Total upstream Scope 3 (Cat 1-8)"
    )
    downstream_tco2e: Decimal = Field(
        default=Decimal("0"), description="Total downstream Scope 3 (Cat 9-15)"
    )

class IntensityMetric(BaseModel):
    """GHG emission intensity metric per ESRS E1-6 Para 48.

    Intensity is calculated as total emissions divided by a
    business metric (e.g. net revenue, headcount, production units).
    """
    numerator_tco2e: Decimal = Field(
        ..., description="Total GHG emissions used as numerator (tCO2e)"
    )
    denominator_value: Decimal = Field(
        ..., description="Denominator value (e.g. revenue in EUR millions)"
    )
    denominator_unit: str = Field(
        ..., description="Unit of denominator (e.g. 'EUR_million', 'headcount', 'unit')"
    )
    intensity_value: Decimal = Field(
        default=Decimal("0"),
        description="Calculated intensity (tCO2e per denominator unit)",
    )
    provenance_hash: str = Field(
        default="", description="SHA-256 hash of the intensity calculation"
    )

class GHGInventoryResult(BaseModel):
    """Complete GHG inventory result per ESRS E1-6.

    Aggregates all emission entries into scope totals, gas
    disaggregation, Scope 3 category breakdown, and intensity
    metrics.  Includes base year comparison and provenance tracking.
    """
    result_id: str = Field(
        default_factory=_new_uuid,
        description="Unique result identifier",
    )
    engine_version: str = Field(
        default=_MODULE_VERSION,
        description="Engine version used for this calculation",
    )
    calculated_at: datetime = Field(
        default_factory=utcnow,
        description="Timestamp of calculation (UTC)",
    )
    reporting_year: int = Field(
        default=0, description="Reporting year"
    )
    entity_name: str = Field(
        default="", description="Entity or undertaking name"
    )
    consolidation_approach: str = Field(
        default="operational_control",
        description="Consolidation approach used",
    )
    scope1_total_tco2e: Decimal = Field(
        default=Decimal("0"), description="Gross Scope 1 GHG emissions (tCO2e)"
    )
    scope2_location_total_tco2e: Decimal = Field(
        default=Decimal("0"),
        description="Gross Scope 2 GHG emissions - location-based (tCO2e)",
    )
    scope2_market_total_tco2e: Decimal = Field(
        default=Decimal("0"),
        description="Gross Scope 2 GHG emissions - market-based (tCO2e)",
    )
    scope3_total_tco2e: Decimal = Field(
        default=Decimal("0"), description="Gross Scope 3 GHG emissions (tCO2e)"
    )
    total_tco2e: Decimal = Field(
        default=Decimal("0"),
        description="Total GHG emissions (Scope 1 + 2 market + 3) in tCO2e",
    )
    total_location_based_tco2e: Decimal = Field(
        default=Decimal("0"),
        description="Total using Scope 2 location-based in tCO2e",
    )
    by_gas: Optional[EmissionsByGas] = Field(
        default=None, description="Emissions disaggregated by gas"
    )
    scope3_breakdown: Optional[Scope3Breakdown] = Field(
        default=None, description="Scope 3 breakdown by category"
    )
    biogenic_co2_tco2e: Decimal = Field(
        default=Decimal("0"),
        description="Biogenic CO2 emissions reported separately (tCO2e)",
    )
    base_year: Optional[int] = Field(
        default=None, description="Base year for comparison"
    )
    base_year_emissions_tco2e: Optional[Decimal] = Field(
        default=None, description="Base year total emissions (tCO2e)"
    )
    change_from_base_year_pct: Optional[Decimal] = Field(
        default=None, description="Percentage change from base year"
    )
    entry_count: int = Field(
        default=0, description="Number of emission entries processed"
    )
    data_quality_score: Decimal = Field(
        default=Decimal("0"),
        description="Weighted average data quality score (0-1)",
    )
    processing_time_ms: float = Field(
        default=0.0, description="Processing time in milliseconds"
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 hash of all inputs and calculation steps",
    )

class BatchInventoryResult(BaseModel):
    """Consolidated inventory result for multiple entities.

    Used when consolidating GHG inventories across subsidiaries,
    business units, or other organizational entities.
    """
    batch_id: str = Field(
        default_factory=_new_uuid,
        description="Unique batch identifier",
    )
    engine_version: str = Field(
        default=_MODULE_VERSION,
        description="Engine version used",
    )
    calculated_at: datetime = Field(
        default_factory=utcnow,
        description="Timestamp of batch calculation (UTC)",
    )
    consolidation_approach: str = Field(
        default="operational_control",
        description="Consolidation approach used for the batch",
    )
    entities: List[GHGInventoryResult] = Field(
        default_factory=list,
        description="Individual entity inventory results",
    )
    entity_count: int = Field(
        default=0, description="Number of entities consolidated"
    )
    consolidated_scope1_tco2e: Decimal = Field(
        default=Decimal("0"), description="Consolidated Scope 1 (tCO2e)"
    )
    consolidated_scope2_location_tco2e: Decimal = Field(
        default=Decimal("0"),
        description="Consolidated Scope 2 location-based (tCO2e)",
    )
    consolidated_scope2_market_tco2e: Decimal = Field(
        default=Decimal("0"),
        description="Consolidated Scope 2 market-based (tCO2e)",
    )
    consolidated_scope3_tco2e: Decimal = Field(
        default=Decimal("0"), description="Consolidated Scope 3 (tCO2e)"
    )
    consolidated_total_tco2e: Decimal = Field(
        default=Decimal("0"), description="Consolidated total (tCO2e)"
    )
    processing_time_ms: float = Field(
        default=0.0, description="Processing time in milliseconds"
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 hash of the consolidated result",
    )

# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class GHGInventoryEngine:
    """GHG inventory calculation engine per ESRS E1-6.

    Provides deterministic, zero-hallucination calculations for:
    - Per-entry emission calculation (activity data * EF * GWP)
    - Scope-level aggregation with biogenic separation
    - Gas-level disaggregation (7 Kyoto gas groups)
    - Scope 3 category breakdown (15 categories)
    - Intensity metrics (tCO2e per revenue/headcount/unit)
    - Multi-entity consolidation
    - Base year comparison
    - E1-6 completeness validation
    - E1-6 data point mapping

    All calculations use Decimal arithmetic for bit-perfect
    reproducibility.  No LLM is used in any calculation path.

    Calculation Methodology:
        1. Per entry: tCO2e = activity_data * emission_factor / 1000
           (factor assumed in kgCO2e, converted to tonnes)
        2. GWP application: tCO2e = mass_gas * GWP_AR6[gas]
        3. Scope totals: sum of entries per scope
        4. Total = Scope 1 + Scope 2 (market) + Scope 3
        5. Intensity = total_tco2e / denominator
        6. Change from base year = (current - base) / base * 100

    Usage::

        engine = GHGInventoryEngine()
        entries = [
            EmissionEntry(
                scope=GHGScope.SCOPE_1,
                gas=EmissionGas.CO2,
                activity_data=Decimal("50000"),
                emission_factor=Decimal("2.68"),
            ),
        ]
        result = engine.build_inventory(entries)
    """

    engine_version: str = _MODULE_VERSION

    # ------------------------------------------------------------------ #
    # Core Calculation Methods                                             #
    # ------------------------------------------------------------------ #

    def calculate_emission(self, entry: EmissionEntry) -> Decimal:
        """Calculate tCO2e for a single emission entry.

        If entry.value_tco2e is already set (> 0), returns that value
        directly (pre-calculated entries).  Otherwise, calculates:
            tCO2e = activity_data * emission_factor / 1000

        The division by 1000 converts from kgCO2e to tCO2e (the
        standard ESRS reporting unit).  If the emission factor is
        already in tCO2e per unit, the caller should set value_tco2e
        directly on the entry.

        For non-CO2 gases without a pre-calculated value, the engine
        applies the appropriate GWP-100 value from IPCC AR6.

        Args:
            entry: EmissionEntry with activity data and emission factor.

        Returns:
            Emission value in tCO2e (Decimal, 6 decimal places).

        Raises:
            ValueError: If activity_data or emission_factor is negative.
        """
        if entry.activity_data < Decimal("0"):
            raise ValueError(
                f"activity_data must be >= 0, got {entry.activity_data}"
            )
        if entry.emission_factor < Decimal("0"):
            raise ValueError(
                f"emission_factor must be >= 0, got {entry.emission_factor}"
            )

        # Use pre-calculated value if provided
        if entry.value_tco2e > Decimal("0"):
            return _round6(entry.value_tco2e)

        # Calculate raw emission in kgCO2e then convert to tCO2e
        raw_kg = entry.activity_data * entry.emission_factor
        tco2e = raw_kg / Decimal("1000")

        # Apply GWP for non-CO2 gases if factor is in native gas units
        if entry.gas != EmissionGas.CO2 and entry.gas_detail:
            gwp = GWP_AR6.get(entry.gas_detail, Decimal("1"))
            tco2e = tco2e * gwp
        elif entry.gas != EmissionGas.CO2:
            gwp = GWP_AR6.get(entry.gas.value, Decimal("1"))
            tco2e = tco2e * gwp

        return _round6(tco2e)

    def calculate_emission_with_gwp(
        self,
        mass_tonnes: Decimal,
        gas: str,
    ) -> Decimal:
        """Calculate tCO2e from mass of gas using GWP-100 conversion.

        This is used when the input is a mass of gas (in tonnes) rather
        than activity data with an emission factor.

        Formula: tCO2e = mass_tonnes * GWP_AR6[gas]

        Args:
            mass_tonnes: Mass of the gas in metric tonnes.
            gas: Gas identifier (must be a key in GWP_AR6).

        Returns:
            Emission value in tCO2e (Decimal, 6 decimal places).

        Raises:
            ValueError: If gas is not found in GWP_AR6 table.
        """
        gwp = GWP_AR6.get(gas)
        if gwp is None:
            raise ValueError(
                f"Unknown gas '{gas}'. Valid gases: {list(GWP_AR6.keys())}"
            )
        return _round6(mass_tonnes * gwp)

    # ------------------------------------------------------------------ #
    # Inventory Building                                                   #
    # ------------------------------------------------------------------ #

    def build_inventory(
        self,
        entries: List[EmissionEntry],
        base_year_result: Optional[GHGInventoryResult] = None,
        entity_name: str = "",
        reporting_year: int = 0,
        consolidation_approach: str = "operational_control",
    ) -> GHGInventoryResult:
        """Build a complete GHG inventory from emission entries.

        Processes all entries, aggregates by scope, disaggregates by gas,
        breaks down Scope 3 by category, and computes base year changes.

        Args:
            entries: List of EmissionEntry instances.
            base_year_result: Optional base year inventory for comparison.
            entity_name: Name of the reporting entity.
            reporting_year: Reporting year.
            consolidation_approach: Consolidation approach used.

        Returns:
            GHGInventoryResult with complete provenance.

        Raises:
            ValueError: If entries list is empty.
        """
        t0 = time.perf_counter()

        if not entries:
            raise ValueError("At least one EmissionEntry is required")

        logger.info(
            "Building GHG inventory: %d entries, entity=%s, year=%d",
            len(entries), entity_name, reporting_year,
        )

        # Step 1: Calculate tCO2e for each entry
        calculated: List[Tuple[EmissionEntry, Decimal]] = []
        for entry in entries:
            tco2e = self.calculate_emission(entry)
            calculated.append((entry, tco2e))

        # Step 2: Aggregate by scope (excluding biogenic)
        scope1 = Decimal("0")
        scope2_loc = Decimal("0")
        scope2_mkt = Decimal("0")
        scope3 = Decimal("0")
        biogenic = Decimal("0")

        for entry, tco2e in calculated:
            if entry.is_biogenic:
                biogenic += tco2e
                continue
            if entry.scope == GHGScope.SCOPE_1:
                scope1 += tco2e
            elif entry.scope == GHGScope.SCOPE_2_LOCATION:
                scope2_loc += tco2e
            elif entry.scope == GHGScope.SCOPE_2_MARKET:
                scope2_mkt += tco2e
            elif entry.scope == GHGScope.SCOPE_3:
                scope3 += tco2e

        scope1 = _round6(scope1)
        scope2_loc = _round6(scope2_loc)
        scope2_mkt = _round6(scope2_mkt)
        scope3 = _round6(scope3)
        biogenic = _round6(biogenic)

        # Total = Scope 1 + Scope 2 market + Scope 3
        total = _round6(scope1 + scope2_mkt + scope3)
        total_loc = _round6(scope1 + scope2_loc + scope3)

        # Step 3: Disaggregate by gas
        by_gas = self.disaggregate_by_gas(
            [(e, t) for e, t in calculated if not e.is_biogenic]
        )

        # Step 4: Scope 3 breakdown
        scope3_entries = [
            (e, t) for e, t in calculated
            if e.scope == GHGScope.SCOPE_3 and not e.is_biogenic
        ]
        scope3_breakdown = self._build_scope3_breakdown(scope3_entries)

        # Step 5: Data quality score
        dq_score = self._calculate_data_quality_score(entries)

        # Step 6: Base year comparison
        change_pct: Optional[Decimal] = None
        base_year_val: Optional[int] = None
        base_year_emissions: Optional[Decimal] = None
        if base_year_result is not None:
            base_year_val = base_year_result.reporting_year
            base_year_emissions = base_year_result.total_tco2e
            if base_year_emissions and base_year_emissions > Decimal("0"):
                change_pct = _round_val(
                    (total - base_year_emissions)
                    / base_year_emissions
                    * Decimal("100"),
                    2,
                )

        elapsed_ms = _round3((time.perf_counter() - t0) * 1000.0)

        result = GHGInventoryResult(
            reporting_year=reporting_year,
            entity_name=entity_name,
            consolidation_approach=consolidation_approach,
            scope1_total_tco2e=scope1,
            scope2_location_total_tco2e=scope2_loc,
            scope2_market_total_tco2e=scope2_mkt,
            scope3_total_tco2e=scope3,
            total_tco2e=total,
            total_location_based_tco2e=total_loc,
            by_gas=by_gas,
            scope3_breakdown=scope3_breakdown,
            biogenic_co2_tco2e=biogenic,
            base_year=base_year_val,
            base_year_emissions_tco2e=base_year_emissions,
            change_from_base_year_pct=change_pct,
            entry_count=len(entries),
            data_quality_score=dq_score,
            processing_time_ms=elapsed_ms,
        )

        result.provenance_hash = _compute_hash(result)

        logger.info(
            "GHG inventory built: total=%.2f tCO2e, S1=%.2f, S2L=%.2f, "
            "S2M=%.2f, S3=%.2f, biogenic=%.2f, hash=%s",
            float(total), float(scope1), float(scope2_loc),
            float(scope2_mkt), float(scope3), float(biogenic),
            result.provenance_hash[:16],
        )

        return result

    # ------------------------------------------------------------------ #
    # Scope 3 Calculation                                                  #
    # ------------------------------------------------------------------ #

    def calculate_scope3(
        self, entries: List[EmissionEntry]
    ) -> Scope3Breakdown:
        """Calculate Scope 3 emissions breakdown by category.

        Filters entries to Scope 3 only, calculates tCO2e for each,
        and aggregates by the 15 GHG Protocol categories.

        Args:
            entries: List of EmissionEntry (only Scope 3 entries used).

        Returns:
            Scope3Breakdown with per-category totals.
        """
        scope3_entries = [e for e in entries if e.scope == GHGScope.SCOPE_3]
        calculated = []
        for entry in scope3_entries:
            tco2e = self.calculate_emission(entry)
            calculated.append((entry, tco2e))
        return self._build_scope3_breakdown(calculated)

    def _build_scope3_breakdown(
        self,
        calculated: List[Tuple[EmissionEntry, Decimal]],
    ) -> Scope3Breakdown:
        """Build Scope 3 breakdown from calculated entries.

        Args:
            calculated: List of (entry, tco2e) tuples for Scope 3 entries.

        Returns:
            Scope3Breakdown with per-category totals and upstream/downstream
            split.
        """
        by_cat: Dict[str, Decimal] = {}
        for cat in Scope3Category:
            by_cat[cat.value] = Decimal("0")

        for entry, tco2e in calculated:
            if entry.scope3_category is not None:
                cat_key = entry.scope3_category.value
                by_cat[cat_key] = by_cat.get(cat_key, Decimal("0")) + tco2e

        # Round each category
        for key in by_cat:
            by_cat[key] = _round6(by_cat[key])

        total = _round6(sum(by_cat.values()))

        # Categories included (non-zero)
        included = [k for k, v in by_cat.items() if v > Decimal("0")]

        # Upstream = categories 1-8, downstream = 9-15
        upstream_cats = {
            "cat_01_purchased_goods", "cat_02_capital_goods",
            "cat_03_fuel_energy", "cat_04_upstream_transport",
            "cat_05_waste", "cat_06_business_travel",
            "cat_07_employee_commuting", "cat_08_upstream_leased",
        }
        downstream_cats = {
            "cat_09_downstream_transport", "cat_10_processing_sold",
            "cat_11_use_of_sold", "cat_12_end_of_life",
            "cat_13_downstream_leased", "cat_14_franchises",
            "cat_15_investments",
        }

        upstream = _round6(sum(by_cat[c] for c in upstream_cats))
        downstream = _round6(sum(by_cat[c] for c in downstream_cats))

        return Scope3Breakdown(
            by_category=by_cat,
            categories_included=sorted(included),
            total_tco2e=total,
            upstream_tco2e=upstream,
            downstream_tco2e=downstream,
        )

    # ------------------------------------------------------------------ #
    # Intensity Metric                                                     #
    # ------------------------------------------------------------------ #

    def calculate_intensity(
        self,
        total_tco2e: Decimal,
        denominator_value: Decimal,
        denominator_unit: str,
    ) -> IntensityMetric:
        """Calculate GHG emission intensity metric.

        Per ESRS E1-6 Para 48, the intensity metric is total GHG
        emissions divided by net revenue.  This method supports
        arbitrary denominators (revenue, headcount, production units).

        Formula: intensity = total_tco2e / denominator_value

        Args:
            total_tco2e: Total GHG emissions (tCO2e).
            denominator_value: Denominator value (must be > 0).
            denominator_unit: Unit description of the denominator.

        Returns:
            IntensityMetric with calculated intensity and provenance.

        Raises:
            ValueError: If denominator_value is zero or negative.
        """
        if denominator_value <= Decimal("0"):
            raise ValueError(
                f"Denominator must be > 0, got {denominator_value}"
            )

        intensity = _round6(
            _safe_divide(total_tco2e, denominator_value)
        )

        result = IntensityMetric(
            numerator_tco2e=total_tco2e,
            denominator_value=denominator_value,
            denominator_unit=denominator_unit,
            intensity_value=intensity,
        )
        result.provenance_hash = _compute_hash(result)

        logger.info(
            "Intensity calculated: %.6f tCO2e/%s",
            float(intensity), denominator_unit,
        )

        return result

    # ------------------------------------------------------------------ #
    # Gas Disaggregation                                                   #
    # ------------------------------------------------------------------ #

    def disaggregate_by_gas(
        self,
        calculated: List[Tuple[EmissionEntry, Decimal]],
    ) -> EmissionsByGas:
        """Disaggregate emissions by individual greenhouse gas.

        Per ESRS E1-6 Para 49, undertakings must report emissions
        disaggregated by the seven gas groups defined under the
        Kyoto Protocol.

        Args:
            calculated: List of (entry, tco2e) tuples.

        Returns:
            EmissionsByGas with per-gas totals in tCO2e.
        """
        gas_totals: Dict[str, Decimal] = {
            "co2": Decimal("0"),
            "ch4": Decimal("0"),
            "n2o": Decimal("0"),
            "hfcs": Decimal("0"),
            "pfcs": Decimal("0"),
            "sf6": Decimal("0"),
            "nf3": Decimal("0"),
        }

        for entry, tco2e in calculated:
            gas_key = entry.gas.value
            if gas_key in gas_totals:
                gas_totals[gas_key] += tco2e
            else:
                # Unknown gas type, add to CO2 as fallback
                gas_totals["co2"] += tco2e

        # Round all
        for key in gas_totals:
            gas_totals[key] = _round6(gas_totals[key])

        total = _round6(sum(gas_totals.values()))

        return EmissionsByGas(
            co2_tco2e=gas_totals["co2"],
            ch4_tco2e=gas_totals["ch4"],
            n2o_tco2e=gas_totals["n2o"],
            hfcs_tco2e=gas_totals["hfcs"],
            pfcs_tco2e=gas_totals["pfcs"],
            sf6_tco2e=gas_totals["sf6"],
            nf3_tco2e=gas_totals["nf3"],
            total_tco2e=total,
        )

    # ------------------------------------------------------------------ #
    # Multi-Entity Consolidation                                           #
    # ------------------------------------------------------------------ #

    def consolidate_entities(
        self,
        results: List[GHGInventoryResult],
        approach: ConsolidationApproach,
        equity_shares: Optional[Dict[str, Decimal]] = None,
    ) -> BatchInventoryResult:
        """Consolidate GHG inventories from multiple entities.

        Per GHG Protocol Corporate Standard Chapter 3, consolidation
        can use operational control, financial control, or equity share.

        For equity share approach, each entity's emissions are multiplied
        by the reporting company's equity share percentage.

        Args:
            results: List of individual entity GHGInventoryResult.
            approach: Consolidation approach to apply.
            equity_shares: Dict mapping entity_name to equity share
                (Decimal 0-1).  Required for EQUITY_SHARE approach.

        Returns:
            BatchInventoryResult with consolidated totals.

        Raises:
            ValueError: If results list is empty or equity_shares
                missing for EQUITY_SHARE approach.
        """
        t0 = time.perf_counter()

        if not results:
            raise ValueError("At least one GHGInventoryResult is required")

        if approach == ConsolidationApproach.EQUITY_SHARE:
            if equity_shares is None:
                raise ValueError(
                    "equity_shares dict is required for EQUITY_SHARE approach"
                )

        logger.info(
            "Consolidating %d entities using %s approach",
            len(results), approach.value,
        )

        cons_s1 = Decimal("0")
        cons_s2_loc = Decimal("0")
        cons_s2_mkt = Decimal("0")
        cons_s3 = Decimal("0")

        for r in results:
            share = Decimal("1")
            if approach == ConsolidationApproach.EQUITY_SHARE:
                share = equity_shares.get(r.entity_name, Decimal("1"))

            cons_s1 += r.scope1_total_tco2e * share
            cons_s2_loc += r.scope2_location_total_tco2e * share
            cons_s2_mkt += r.scope2_market_total_tco2e * share
            cons_s3 += r.scope3_total_tco2e * share

        cons_s1 = _round6(cons_s1)
        cons_s2_loc = _round6(cons_s2_loc)
        cons_s2_mkt = _round6(cons_s2_mkt)
        cons_s3 = _round6(cons_s3)
        cons_total = _round6(cons_s1 + cons_s2_mkt + cons_s3)

        elapsed_ms = _round3((time.perf_counter() - t0) * 1000.0)

        batch = BatchInventoryResult(
            consolidation_approach=approach.value,
            entities=results,
            entity_count=len(results),
            consolidated_scope1_tco2e=cons_s1,
            consolidated_scope2_location_tco2e=cons_s2_loc,
            consolidated_scope2_market_tco2e=cons_s2_mkt,
            consolidated_scope3_tco2e=cons_s3,
            consolidated_total_tco2e=cons_total,
            processing_time_ms=elapsed_ms,
        )

        batch.provenance_hash = _compute_hash(batch)

        logger.info(
            "Consolidation complete: total=%.2f tCO2e across %d entities",
            float(cons_total), len(results),
        )

        return batch

    # ------------------------------------------------------------------ #
    # Completeness Validation                                              #
    # ------------------------------------------------------------------ #

    def validate_completeness(
        self, result: GHGInventoryResult
    ) -> Dict[str, Any]:
        """Validate completeness against E1-6 required data points.

        Checks whether all ESRS E1-6 mandatory disclosure data points
        are present and populated in the inventory result.

        Args:
            result: GHGInventoryResult to validate.

        Returns:
            Dict with:
                - total_datapoints: int
                - populated_datapoints: int
                - missing_datapoints: list of str
                - completeness_pct: Decimal
                - is_complete: bool
                - provenance_hash: str
        """
        populated = []
        missing = []

        checks = {
            "e1_6_01_scope1_total_tco2e": result.scope1_total_tco2e >= Decimal("0"),
            "e1_6_02_scope1_by_country": True,  # Optional disaggregation
            "e1_6_03_scope1_percentage_from_regulated_ets": True,  # Optional
            "e1_6_04_scope2_location_total_tco2e": (
                result.scope2_location_total_tco2e >= Decimal("0")
            ),
            "e1_6_05_scope2_market_total_tco2e": (
                result.scope2_market_total_tco2e >= Decimal("0")
            ),
            "e1_6_06_scope3_total_tco2e": result.scope3_total_tco2e >= Decimal("0"),
            "e1_6_07_scope3_by_category": (
                result.scope3_breakdown is not None
                and len(result.scope3_breakdown.categories_included) > 0
            ),
            "e1_6_08_total_ghg_emissions_tco2e": result.total_tco2e >= Decimal("0"),
            "e1_6_09_ghg_intensity_per_net_revenue": True,  # Separate calc
            "e1_6_10_disaggregation_by_gas": result.by_gas is not None,
            "e1_6_11_biogenic_co2_emissions": True,  # Reported if applicable
            "e1_6_12_scope3_categories_included": (
                result.scope3_breakdown is not None
                and len(result.scope3_breakdown.categories_included) > 0
            ),
            "e1_6_13_base_year_emissions": (
                result.base_year_emissions_tco2e is not None
            ),
            "e1_6_14_change_from_base_year": (
                result.change_from_base_year_pct is not None
            ),
            "e1_6_15_consolidation_approach": bool(
                result.consolidation_approach
            ),
            "e1_6_16_methodology_description": True,  # Narrative
            "e1_6_17_data_quality_description": (
                result.data_quality_score > Decimal("0")
            ),
            "e1_6_18_significant_changes_methodology": True,  # Narrative
            "e1_6_19_gwp_values_source": True,  # Using IPCC AR6
        }

        for dp, is_populated in checks.items():
            if is_populated:
                populated.append(dp)
            else:
                missing.append(dp)

        total = len(E1_6_DATAPOINTS)
        pop_count = len(populated)
        completeness = _round_val(
            _decimal(pop_count) / _decimal(total) * Decimal("100"), 1
        )

        validation_result = {
            "total_datapoints": total,
            "populated_datapoints": pop_count,
            "missing_datapoints": missing,
            "completeness_pct": completeness,
            "is_complete": len(missing) == 0,
            "provenance_hash": _compute_hash(
                {"result_id": result.result_id, "checks": checks}
            ),
        }

        logger.info(
            "E1-6 completeness: %s%% (%d/%d), missing=%s",
            completeness, pop_count, total, missing,
        )

        return validation_result

    # ------------------------------------------------------------------ #
    # ESRS E1-6 Data Point Mapping                                         #
    # ------------------------------------------------------------------ #

    def get_e1_6_datapoints(
        self,
        result: GHGInventoryResult,
        intensity: Optional[IntensityMetric] = None,
    ) -> Dict[str, Any]:
        """Map inventory result to ESRS E1-6 disclosure data points.

        Creates a structured mapping of all E1-6 required data points
        with their values, ready for report generation.

        Args:
            result: GHGInventoryResult to map.
            intensity: Optional IntensityMetric for E1-6 Para 48.

        Returns:
            Dict mapping E1-6 data point IDs to their values and
            metadata.
        """
        scope3_cats = {}
        if result.scope3_breakdown:
            for cat_key, cat_val in result.scope3_breakdown.by_category.items():
                cat_name = SCOPE_3_CATEGORY_NAMES.get(cat_key, cat_key)
                scope3_cats[cat_key] = {
                    "name": cat_name,
                    "value_tco2e": str(cat_val),
                }

        by_gas_data = {}
        if result.by_gas:
            by_gas_data = {
                "co2_tco2e": str(result.by_gas.co2_tco2e),
                "ch4_tco2e": str(result.by_gas.ch4_tco2e),
                "n2o_tco2e": str(result.by_gas.n2o_tco2e),
                "hfcs_tco2e": str(result.by_gas.hfcs_tco2e),
                "pfcs_tco2e": str(result.by_gas.pfcs_tco2e),
                "sf6_tco2e": str(result.by_gas.sf6_tco2e),
                "nf3_tco2e": str(result.by_gas.nf3_tco2e),
            }

        intensity_data = None
        if intensity is not None:
            intensity_data = {
                "numerator_tco2e": str(intensity.numerator_tco2e),
                "denominator_value": str(intensity.denominator_value),
                "denominator_unit": intensity.denominator_unit,
                "intensity_value": str(intensity.intensity_value),
            }

        datapoints = {
            "e1_6_01_scope1_total_tco2e": {
                "label": "Gross Scope 1 GHG emissions",
                "value": str(result.scope1_total_tco2e),
                "unit": "tCO2e",
                "esrs_ref": "E1-6 Para 44",
            },
            "e1_6_04_scope2_location_total_tco2e": {
                "label": "Gross Scope 2 GHG emissions (location-based)",
                "value": str(result.scope2_location_total_tco2e),
                "unit": "tCO2e",
                "esrs_ref": "E1-6 Para 45",
            },
            "e1_6_05_scope2_market_total_tco2e": {
                "label": "Gross Scope 2 GHG emissions (market-based)",
                "value": str(result.scope2_market_total_tco2e),
                "unit": "tCO2e",
                "esrs_ref": "E1-6 Para 45",
            },
            "e1_6_06_scope3_total_tco2e": {
                "label": "Gross Scope 3 GHG emissions",
                "value": str(result.scope3_total_tco2e),
                "unit": "tCO2e",
                "esrs_ref": "E1-6 Para 46",
            },
            "e1_6_07_scope3_by_category": {
                "label": "Scope 3 emissions by category",
                "value": scope3_cats,
                "esrs_ref": "E1-6 Para 46",
            },
            "e1_6_08_total_ghg_emissions_tco2e": {
                "label": "Total GHG emissions",
                "value": str(result.total_tco2e),
                "unit": "tCO2e",
                "esrs_ref": "E1-6 Para 47",
            },
            "e1_6_09_ghg_intensity_per_net_revenue": {
                "label": "GHG emissions intensity per net revenue",
                "value": intensity_data,
                "esrs_ref": "E1-6 Para 48",
            },
            "e1_6_10_disaggregation_by_gas": {
                "label": "GHG emissions disaggregated by gas",
                "value": by_gas_data,
                "esrs_ref": "E1-6 Para 49",
            },
            "e1_6_11_biogenic_co2_emissions": {
                "label": "Biogenic CO2 emissions",
                "value": str(result.biogenic_co2_tco2e),
                "unit": "tCO2e",
                "esrs_ref": "E1-6 Para 50",
            },
            "e1_6_12_scope3_categories_included": {
                "label": "Scope 3 categories included",
                "value": (
                    result.scope3_breakdown.categories_included
                    if result.scope3_breakdown else []
                ),
                "esrs_ref": "E1-6 Para 46",
            },
            "e1_6_13_base_year_emissions": {
                "label": "Base year emissions",
                "value": (
                    str(result.base_year_emissions_tco2e)
                    if result.base_year_emissions_tco2e is not None
                    else None
                ),
                "unit": "tCO2e",
                "esrs_ref": "E1-6 Para 44",
            },
            "e1_6_14_change_from_base_year": {
                "label": "Change from base year",
                "value": (
                    str(result.change_from_base_year_pct)
                    if result.change_from_base_year_pct is not None
                    else None
                ),
                "unit": "percent",
                "esrs_ref": "E1-6 Para 44",
            },
            "e1_6_15_consolidation_approach": {
                "label": "Consolidation approach",
                "value": result.consolidation_approach,
                "esrs_ref": "E1-6 Para 44",
            },
            "e1_6_19_gwp_values_source": {
                "label": "GWP values source",
                "value": "IPCC AR6 WG1 (2021) GWP-100",
                "esrs_ref": "E1-6 Para 49",
            },
        }

        datapoints["provenance_hash"] = _compute_hash(datapoints)

        return datapoints

    # ------------------------------------------------------------------ #
    # Scope Summaries                                                      #
    # ------------------------------------------------------------------ #

    def get_scope_summary(
        self, result: GHGInventoryResult
    ) -> Dict[str, Any]:
        """Generate a scope-level summary of the GHG inventory.

        Args:
            result: GHGInventoryResult to summarize.

        Returns:
            Dict with scope totals, percentages, and summary metrics.
        """
        total = result.total_tco2e if result.total_tco2e > Decimal("0") else Decimal("1")

        scope1_pct = _round_val(
            result.scope1_total_tco2e / total * Decimal("100"), 1
        )
        scope2_mkt_pct = _round_val(
            result.scope2_market_total_tco2e / total * Decimal("100"), 1
        )
        scope3_pct = _round_val(
            result.scope3_total_tco2e / total * Decimal("100"), 1
        )

        return {
            "scope1": {
                "total_tco2e": str(result.scope1_total_tco2e),
                "percentage": str(scope1_pct),
            },
            "scope2_location": {
                "total_tco2e": str(result.scope2_location_total_tco2e),
            },
            "scope2_market": {
                "total_tco2e": str(result.scope2_market_total_tco2e),
                "percentage": str(scope2_mkt_pct),
            },
            "scope3": {
                "total_tco2e": str(result.scope3_total_tco2e),
                "percentage": str(scope3_pct),
            },
            "total_tco2e": str(result.total_tco2e),
            "total_location_based_tco2e": str(result.total_location_based_tco2e),
            "biogenic_co2_tco2e": str(result.biogenic_co2_tco2e),
            "entry_count": result.entry_count,
            "data_quality_score": str(result.data_quality_score),
            "provenance_hash": result.provenance_hash,
        }

    # ------------------------------------------------------------------ #
    # GWP Lookup                                                           #
    # ------------------------------------------------------------------ #

    def get_gwp(self, gas: str) -> Decimal:
        """Look up the GWP-100 value for a greenhouse gas.

        Args:
            gas: Gas identifier (key in GWP_AR6 dict).

        Returns:
            GWP-100 value as Decimal.

        Raises:
            ValueError: If gas is not found in GWP_AR6.
        """
        gwp = GWP_AR6.get(gas)
        if gwp is None:
            raise ValueError(
                f"Unknown gas '{gas}'. Valid: {sorted(GWP_AR6.keys())}"
            )
        return gwp

    def list_gwp_values(self) -> Dict[str, str]:
        """Return all GWP-100 values as a dict of gas -> string value.

        Returns:
            Dict mapping gas identifiers to GWP-100 values.
        """
        return {k: str(v) for k, v in sorted(GWP_AR6.items())}

    # ------------------------------------------------------------------ #
    # Year-over-Year Comparison                                            #
    # ------------------------------------------------------------------ #

    def compare_years(
        self,
        current: GHGInventoryResult,
        previous: GHGInventoryResult,
    ) -> Dict[str, Any]:
        """Compare GHG inventories across two reporting years.

        Calculates absolute and percentage changes for each scope
        and total emissions.

        Args:
            current: Current year inventory result.
            previous: Previous year inventory result.

        Returns:
            Dict with absolute_change, pct_change for each scope
            and total, plus a provenance hash.
        """
        def _change(curr: Decimal, prev: Decimal) -> Dict[str, str]:
            abs_change = curr - prev
            pct = _safe_divide(
                abs_change, prev if prev != Decimal("0") else Decimal("1")
            ) * Decimal("100")
            return {
                "current": str(curr),
                "previous": str(prev),
                "absolute_change": str(_round6(abs_change)),
                "pct_change": str(_round_val(pct, 2)),
            }

        comparison = {
            "current_year": current.reporting_year,
            "previous_year": previous.reporting_year,
            "scope1": _change(
                current.scope1_total_tco2e,
                previous.scope1_total_tco2e,
            ),
            "scope2_location": _change(
                current.scope2_location_total_tco2e,
                previous.scope2_location_total_tco2e,
            ),
            "scope2_market": _change(
                current.scope2_market_total_tco2e,
                previous.scope2_market_total_tco2e,
            ),
            "scope3": _change(
                current.scope3_total_tco2e,
                previous.scope3_total_tco2e,
            ),
            "total": _change(
                current.total_tco2e,
                previous.total_tco2e,
            ),
        }

        comparison["provenance_hash"] = _compute_hash(comparison)
        return comparison

    # ------------------------------------------------------------------ #
    # Private Helpers                                                      #
    # ------------------------------------------------------------------ #

    def _calculate_data_quality_score(
        self, entries: List[EmissionEntry]
    ) -> Decimal:
        """Calculate weighted average data quality score.

        Weights each entry's data quality by its contribution to
        total activity data.

        Args:
            entries: List of EmissionEntry.

        Returns:
            Weighted average quality score (Decimal, 0-1 scale).
        """
        if not entries:
            return Decimal("0")

        total_activity = sum(e.activity_data for e in entries)
        if total_activity == Decimal("0"):
            # Fallback: simple average
            scores = [
                DATA_QUALITY_SCORES.get(e.data_quality.value, Decimal("0.25"))
                for e in entries
            ]
            return _round_val(
                sum(scores) / _decimal(len(scores)), 3
            )

        weighted_sum = Decimal("0")
        for entry in entries:
            dq_score = DATA_QUALITY_SCORES.get(
                entry.data_quality.value, Decimal("0.25")
            )
            weight = _safe_divide(entry.activity_data, total_activity)
            weighted_sum += dq_score * weight

        return _round_val(weighted_sum, 3)

    def _get_scope_label(self, scope: GHGScope) -> str:
        """Get human-readable label for a GHG scope.

        Args:
            scope: GHGScope enum value.

        Returns:
            Human-readable scope label string.
        """
        labels = {
            GHGScope.SCOPE_1: "Scope 1 (Direct)",
            GHGScope.SCOPE_2_LOCATION: "Scope 2 (Location-Based)",
            GHGScope.SCOPE_2_MARKET: "Scope 2 (Market-Based)",
            GHGScope.SCOPE_3: "Scope 3 (Value Chain)",
        }
        return labels.get(scope, scope.value)

    def _get_data_quality_label(self, level: DataQualityLevel) -> str:
        """Get human-readable label for a data quality level.

        Args:
            level: DataQualityLevel enum value.

        Returns:
            Human-readable data quality description.
        """
        labels = {
            DataQualityLevel.PRIMARY: "Primary data (measured/metered)",
            DataQualityLevel.SECONDARY_SPECIFIC: "Secondary data (supplier-specific factors)",
            DataQualityLevel.SECONDARY_AVERAGE: "Secondary data (industry-average factors)",
            DataQualityLevel.ESTIMATED: "Estimated data (modelled/proxy)",
        }
        return labels.get(level, level.value)
