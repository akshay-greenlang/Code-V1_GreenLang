# -*- coding: utf-8 -*-
"""
NetZeroBaselineEngine - PACK-021 Net Zero Starter Engine 1
============================================================

Unified GHG baseline assessment combining Scope 1 + 2 + 3 with base year
selection, data quality scoring, and organizational boundary handling.

This engine produces the foundational emissions inventory that all other
Net Zero Starter engines depend on.  It calculates Scope 1 (stationary
combustion, mobile combustion, process, fugitive, and refrigerant
emissions), Scope 2 (location-based and market-based), and Scope 3 (all
15 categories using simplified spend-based methodology for the starter
tier).  It also performs base year selection and validation per GHG
Protocol Corporate Standard Chapters 5 and 6.

Calculation Methodology:
    Scope 1:
        tCO2e = fuel_quantity * fuel_emission_factor  (per fuel type)
        refrigerant_tCO2e = leakage_kg * GWP / 1000
        mobile_tCO2e = fuel_litres * fuel_factor_per_litre
        process_tCO2e = activity_data * process_factor
        fugitive_tCO2e = leak_mass * GWP

    Scope 2:
        location_tCO2e = electricity_mwh * grid_factor
        market_tCO2e = electricity_mwh * residual_mix_factor  (or PPA/REC)

    Scope 3 (spend-based):
        tCO2e = spend_amount * spend_based_factor

    Total = Scope 1 + Scope 2 (market-based) + Scope 3

Regulatory References:
    - GHG Protocol Corporate Standard (2004, revised 2015) - Chapters 3-8
    - GHG Protocol Scope 2 Guidance (2015)
    - GHG Protocol Scope 3 Standard (2011) - Technical Guidance
    - IPCC AR6 WG1 (2021) - GWP-100 values, Table 7.15
    - SBTi Corporate Net-Zero Standard v1.2 (2023) - Boundary requirements

Zero-Hallucination:
    - All emission calculations use deterministic Decimal arithmetic
    - Emission factors are hard-coded constants from authoritative sources
    - GWP values sourced from IPCC AR6 (no ML/LLM)
    - SHA-256 provenance hash on every result
    - No LLM involvement in any calculation path

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-021 Net Zero Starter
Status:  Production Ready
"""

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
    """Safely divide two Decimals, returning *default* on zero denominator."""
    if denominator == Decimal("0"):
        return default
    return numerator / denominator

def _safe_pct(part: Decimal, whole: Decimal) -> Decimal:
    """Compute percentage safely (part / whole * 100)."""
    return _safe_divide(part * Decimal("100"), whole)

def _round_val(value: Decimal, places: int = 6) -> Decimal:
    """Round a Decimal to *places* using ROUND_HALF_UP.

    Args:
        value: Value to round.
        places: Number of decimal places.

    Returns:
        Rounded Decimal value.
    """
    quantize_str = "0." + "0" * places
    return value.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP)

def _round3(value: float) -> float:
    """Round to 3 decimal places using ROUND_HALF_UP."""
    return float(
        Decimal(str(value)).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)
    )

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class BoundaryMethod(str, Enum):
    """Organizational boundary method per GHG Protocol Ch 3.

    Determines how the reporting entity accounts for emissions from
    operations where it does not own 100%.
    """
    EQUITY_SHARE = "equity_share"
    FINANCIAL_CONTROL = "financial_control"
    OPERATIONAL_CONTROL = "operational_control"

class DataQualityScore(str, Enum):
    """Data quality scoring on a 1-5 scale per GHG Protocol guidance.

    Score 1 = highest quality (direct measurement).
    Score 5 = lowest quality (rough estimate / proxy).
    """
    SCORE_1 = "1"
    SCORE_2 = "2"
    SCORE_3 = "3"
    SCORE_4 = "4"
    SCORE_5 = "5"

class Scope(str, Enum):
    """GHG emission scope per GHG Protocol Corporate Standard."""
    SCOPE_1 = "scope_1"
    SCOPE_2_LOCATION = "scope_2_location"
    SCOPE_2_MARKET = "scope_2_market"
    SCOPE_3 = "scope_3"

class GasType(str, Enum):
    """Greenhouse gas type per IPCC / UNFCCC classification."""
    CO2 = "co2"
    CH4 = "ch4"
    N2O = "n2o"
    HFCS = "hfcs"
    PFCS = "pfcs"
    SF6 = "sf6"
    NF3 = "nf3"

class FuelType(str, Enum):
    """Fuel types with associated emission factors."""
    NATURAL_GAS = "natural_gas"
    DIESEL = "diesel"
    GASOLINE = "gasoline"
    LPG = "lpg"
    COAL_ANTHRACITE = "coal_anthracite"
    COAL_BITUMINOUS = "coal_bituminous"
    FUEL_OIL = "fuel_oil"
    KEROSENE = "kerosene"
    PROPANE = "propane"
    BIOMASS_WOOD = "biomass_wood"
    BIOGAS = "biogas"
    JET_FUEL = "jet_fuel"
    CNG = "cng"
    ETHANOL = "ethanol"
    BIODIESEL = "biodiesel"

class Scope3Category(str, Enum):
    """GHG Protocol Scope 3 categories (1-15)."""
    CAT_01_PURCHASED_GOODS = "cat_01_purchased_goods"
    CAT_02_CAPITAL_GOODS = "cat_02_capital_goods"
    CAT_03_FUEL_ENERGY = "cat_03_fuel_energy"
    CAT_04_UPSTREAM_TRANSPORT = "cat_04_upstream_transport"
    CAT_05_WASTE = "cat_05_waste"
    CAT_06_BUSINESS_TRAVEL = "cat_06_business_travel"
    CAT_07_EMPLOYEE_COMMUTING = "cat_07_employee_commuting"
    CAT_08_UPSTREAM_LEASED = "cat_08_upstream_leased"
    CAT_09_DOWNSTREAM_TRANSPORT = "cat_09_downstream_transport"
    CAT_10_PROCESSING_SOLD = "cat_10_processing_sold"
    CAT_11_USE_OF_SOLD = "cat_11_use_of_sold"
    CAT_12_END_OF_LIFE = "cat_12_end_of_life"
    CAT_13_DOWNSTREAM_LEASED = "cat_13_downstream_leased"
    CAT_14_FRANCHISES = "cat_14_franchises"
    CAT_15_INVESTMENTS = "cat_15_investments"

class Scope1SourceType(str, Enum):
    """Scope 1 emission source sub-categories."""
    STATIONARY_COMBUSTION = "stationary_combustion"
    MOBILE_COMBUSTION = "mobile_combustion"
    PROCESS_EMISSIONS = "process_emissions"
    FUGITIVE_EMISSIONS = "fugitive_emissions"
    REFRIGERANTS = "refrigerants"

# ---------------------------------------------------------------------------
# Constants -- Reference Data
# ---------------------------------------------------------------------------

# IPCC AR6 GWP-100 values (100-year time horizon).
# Source: IPCC Sixth Assessment Report, Working Group I, Table 7.15 (2021).
GWP_AR6: Dict[str, Decimal] = {
    "co2": Decimal("1"),
    "ch4": Decimal("27.9"),
    "n2o": Decimal("273"),
    "sf6": Decimal("25200"),
    "nf3": Decimal("17400"),
    "hfc_134a": Decimal("1530"),
    "hfc_32": Decimal("771"),
    "hfc_125": Decimal("3740"),
    "hfc_143a": Decimal("5810"),
    "hfc_404a": Decimal("4728"),
    "hfc_410a": Decimal("2256"),
    "hfc_23": Decimal("14600"),
    "r404a": Decimal("3922"),
    "r410a": Decimal("2088"),
    "r134a": Decimal("1430"),
    "r32": Decimal("675"),
    "r290": Decimal("3"),
    "r744": Decimal("1"),
    "hfcs": Decimal("1530"),
    "pfcs": Decimal("7380"),
    "pfc_14": Decimal("7380"),
    "pfc_116": Decimal("12400"),
}

# Fuel emission factors (kgCO2e per unit).
# Source: DEFRA/BEIS 2024 UK Government GHG Conversion Factors & US EPA.
# Units: kgCO2e per litre (liquids), kgCO2e per m3 (gases),
#         kgCO2e per kg (solids).
FUEL_EMISSION_FACTORS: Dict[str, Dict[str, Decimal]] = {
    FuelType.NATURAL_GAS: {
        "factor": Decimal("2.02"),
        "unit": "kgCO2e_per_m3",
    },
    FuelType.DIESEL: {
        "factor": Decimal("2.676"),
        "unit": "kgCO2e_per_litre",
    },
    FuelType.GASOLINE: {
        "factor": Decimal("2.315"),
        "unit": "kgCO2e_per_litre",
    },
    FuelType.LPG: {
        "factor": Decimal("1.557"),
        "unit": "kgCO2e_per_litre",
    },
    FuelType.COAL_ANTHRACITE: {
        "factor": Decimal("2.886"),
        "unit": "kgCO2e_per_kg",
    },
    FuelType.COAL_BITUMINOUS: {
        "factor": Decimal("2.453"),
        "unit": "kgCO2e_per_kg",
    },
    FuelType.FUEL_OIL: {
        "factor": Decimal("3.179"),
        "unit": "kgCO2e_per_litre",
    },
    FuelType.KEROSENE: {
        "factor": Decimal("2.540"),
        "unit": "kgCO2e_per_litre",
    },
    FuelType.PROPANE: {
        "factor": Decimal("1.544"),
        "unit": "kgCO2e_per_litre",
    },
    FuelType.BIOMASS_WOOD: {
        "factor": Decimal("0.015"),
        "unit": "kgCO2e_per_kg",
    },
    FuelType.BIOGAS: {
        "factor": Decimal("0.001"),
        "unit": "kgCO2e_per_m3",
    },
    FuelType.JET_FUEL: {
        "factor": Decimal("2.544"),
        "unit": "kgCO2e_per_litre",
    },
    FuelType.CNG: {
        "factor": Decimal("2.540"),
        "unit": "kgCO2e_per_kg",
    },
    FuelType.ETHANOL: {
        "factor": Decimal("1.610"),
        "unit": "kgCO2e_per_litre",
    },
    FuelType.BIODIESEL: {
        "factor": Decimal("0.170"),
        "unit": "kgCO2e_per_litre",
    },
}

# Grid emission factors by region (tCO2e per MWh).
# Source: IEA Emission Factors 2024, EEA 2024, US EPA eGRID 2024.
GRID_EMISSION_FACTORS: Dict[str, Decimal] = {
    "EU_AVG": Decimal("0.230"),
    "US_AVG": Decimal("0.386"),
    "UK": Decimal("0.207"),
    "DE": Decimal("0.338"),
    "FR": Decimal("0.055"),
    "ES": Decimal("0.150"),
    "IT": Decimal("0.256"),
    "NL": Decimal("0.328"),
    "PL": Decimal("0.635"),
    "SE": Decimal("0.012"),
    "NO": Decimal("0.008"),
    "AT": Decimal("0.091"),
    "BE": Decimal("0.155"),
    "DK": Decimal("0.112"),
    "FI": Decimal("0.068"),
    "IE": Decimal("0.296"),
    "PT": Decimal("0.161"),
    "CH": Decimal("0.015"),
    "JP": Decimal("0.456"),
    "CN": Decimal("0.555"),
    "IN": Decimal("0.708"),
    "AU": Decimal("0.656"),
    "CA": Decimal("0.120"),
    "BR": Decimal("0.075"),
    "KR": Decimal("0.415"),
    "ZA": Decimal("0.928"),
    "GLOBAL_AVG": Decimal("0.436"),
}

# Residual mix factors for market-based Scope 2 (tCO2e per MWh).
RESIDUAL_MIX_FACTORS: Dict[str, Decimal] = {
    "EU_AVG": Decimal("0.376"),
    "US_AVG": Decimal("0.420"),
    "UK": Decimal("0.324"),
    "DE": Decimal("0.493"),
    "FR": Decimal("0.069"),
    "GLOBAL_AVG": Decimal("0.450"),
}

# Spend-based emission factors (tCO2e per $1000 USD spend).
# Source: GHG Protocol Scope 3 Evaluator, EEIO models (US EPA/EXIOBASE).
SPEND_BASED_FACTORS: Dict[str, Decimal] = {
    Scope3Category.CAT_01_PURCHASED_GOODS: Decimal("0.430"),
    Scope3Category.CAT_02_CAPITAL_GOODS: Decimal("0.350"),
    Scope3Category.CAT_03_FUEL_ENERGY: Decimal("0.280"),
    Scope3Category.CAT_04_UPSTREAM_TRANSPORT: Decimal("0.520"),
    Scope3Category.CAT_05_WASTE: Decimal("0.210"),
    Scope3Category.CAT_06_BUSINESS_TRAVEL: Decimal("0.310"),
    Scope3Category.CAT_07_EMPLOYEE_COMMUTING: Decimal("0.180"),
    Scope3Category.CAT_08_UPSTREAM_LEASED: Decimal("0.240"),
    Scope3Category.CAT_09_DOWNSTREAM_TRANSPORT: Decimal("0.480"),
    Scope3Category.CAT_10_PROCESSING_SOLD: Decimal("0.390"),
    Scope3Category.CAT_11_USE_OF_SOLD: Decimal("0.250"),
    Scope3Category.CAT_12_END_OF_LIFE: Decimal("0.190"),
    Scope3Category.CAT_13_DOWNSTREAM_LEASED: Decimal("0.220"),
    Scope3Category.CAT_14_FRANCHISES: Decimal("0.270"),
    Scope3Category.CAT_15_INVESTMENTS: Decimal("0.150"),
}

# Data quality numeric scores for weighted averaging.
DQ_NUMERIC: Dict[str, Decimal] = {
    "1": Decimal("1.00"),
    "2": Decimal("0.80"),
    "3": Decimal("0.60"),
    "4": Decimal("0.40"),
    "5": Decimal("0.20"),
}

# Scope 3 category names for display.
SCOPE_3_NAMES: Dict[str, str] = {
    "cat_01_purchased_goods": "Cat 1: Purchased Goods & Services",
    "cat_02_capital_goods": "Cat 2: Capital Goods",
    "cat_03_fuel_energy": "Cat 3: Fuel- & Energy-Related Activities",
    "cat_04_upstream_transport": "Cat 4: Upstream Transportation",
    "cat_05_waste": "Cat 5: Waste Generated in Operations",
    "cat_06_business_travel": "Cat 6: Business Travel",
    "cat_07_employee_commuting": "Cat 7: Employee Commuting",
    "cat_08_upstream_leased": "Cat 8: Upstream Leased Assets",
    "cat_09_downstream_transport": "Cat 9: Downstream Transportation",
    "cat_10_processing_sold": "Cat 10: Processing of Sold Products",
    "cat_11_use_of_sold": "Cat 11: Use of Sold Products",
    "cat_12_end_of_life": "Cat 12: End-of-Life Treatment",
    "cat_13_downstream_leased": "Cat 13: Downstream Leased Assets",
    "cat_14_franchises": "Cat 14: Franchises",
    "cat_15_investments": "Cat 15: Investments",
}

# ---------------------------------------------------------------------------
# Pydantic Models -- Inputs
# ---------------------------------------------------------------------------

class FuelConsumptionEntry(BaseModel):
    """A single fuel consumption record for Scope 1 calculation.

    Attributes:
        fuel_type: Type of fuel consumed.
        quantity: Quantity consumed in the unit matching the fuel factor.
        source_type: Scope 1 sub-category (stationary, mobile, etc.).
        facility_id: Facility or site identifier.
        data_quality: Quality score for this data point (1-5).
        notes: Optional notes.
    """
    fuel_type: FuelType = Field(..., description="Fuel type")
    quantity: Decimal = Field(
        ..., ge=Decimal("0"), description="Quantity consumed"
    )
    source_type: Scope1SourceType = Field(
        default=Scope1SourceType.STATIONARY_COMBUSTION,
        description="Scope 1 source category",
    )
    facility_id: str = Field(default="", description="Facility identifier")
    data_quality: DataQualityScore = Field(
        default=DataQualityScore.SCORE_3, description="Data quality (1-5)"
    )
    notes: str = Field(default="", max_length=500)

class RefrigerantEntry(BaseModel):
    """Refrigerant leakage data for Scope 1 calculation.

    Attributes:
        refrigerant_id: Refrigerant gas identifier (e.g. 'r134a').
        charge_kg: Total system charge in kilograms.
        leakage_kg: Actual leakage in kg (if measured).
        leakage_rate_pct: Estimated leakage rate (% of charge per year).
        facility_id: Facility identifier.
        data_quality: Quality score (1-5).
    """
    refrigerant_id: str = Field(
        ..., description="Refrigerant gas key (matching GWP_AR6)"
    )
    charge_kg: Decimal = Field(
        ..., ge=Decimal("0"), description="Total charge (kg)"
    )
    leakage_kg: Optional[Decimal] = Field(
        None, ge=Decimal("0"), description="Measured leakage (kg)"
    )
    leakage_rate_pct: Decimal = Field(
        default=Decimal("15.0"), ge=Decimal("0"), le=Decimal("100"),
        description="Annual leakage rate (%)",
    )
    facility_id: str = Field(default="", description="Facility identifier")
    data_quality: DataQualityScore = Field(
        default=DataQualityScore.SCORE_3,
    )

class ElectricityEntry(BaseModel):
    """Electricity consumption record for Scope 2 calculation.

    Attributes:
        quantity_mwh: Electricity consumed in MWh.
        region: Grid region code for emission factor lookup.
        has_ppa: Whether covered by Power Purchase Agreement.
        ppa_factor: PPA contractual emission factor (tCO2e/MWh).
        rec_mwh: MWh covered by Renewable Energy Certificates.
        facility_id: Facility identifier.
        data_quality: Quality score (1-5).
    """
    quantity_mwh: Decimal = Field(
        ..., ge=Decimal("0"), description="Electricity consumed (MWh)"
    )
    region: str = Field(
        default="GLOBAL_AVG", description="Grid region code"
    )
    has_ppa: bool = Field(False, description="Has PPA")
    ppa_factor: Optional[Decimal] = Field(
        None, ge=Decimal("0"),
        description="PPA contractual EF (tCO2e/MWh)",
    )
    rec_mwh: Decimal = Field(
        default=Decimal("0"), ge=Decimal("0"),
        description="RECs purchased (MWh)",
    )
    facility_id: str = Field(default="", description="Facility identifier")
    data_quality: DataQualityScore = Field(
        default=DataQualityScore.SCORE_3,
    )

class Scope3SpendEntry(BaseModel):
    """Spend-based Scope 3 data for a single category.

    Attributes:
        category: Scope 3 category (1-15).
        spend_usd_thousands: Total spend in thousands of USD.
        custom_factor: Override the default spend-based factor.
        data_quality: Quality score (1-5).
        notes: Optional notes.
    """
    category: Scope3Category = Field(
        ..., description="Scope 3 category"
    )
    spend_usd_thousands: Decimal = Field(
        ..., ge=Decimal("0"), description="Spend ($1000 USD)"
    )
    custom_factor: Optional[Decimal] = Field(
        None, ge=Decimal("0"),
        description="Custom factor (tCO2e per $1000)",
    )
    data_quality: DataQualityScore = Field(
        default=DataQualityScore.SCORE_4,
    )
    notes: str = Field(default="", max_length=500)

class BaselineInput(BaseModel):
    """Complete input data for baseline GHG assessment.

    Attributes:
        entity_name: Reporting entity name.
        reporting_year: Year of the data.
        base_year: Proposed base year (validated against GHG Protocol Ch 5).
        boundary_method: Organizational boundary method.
        fuel_entries: Scope 1 fuel consumption data.
        refrigerant_entries: Scope 1 refrigerant data.
        electricity_entries: Scope 2 electricity data.
        scope3_entries: Scope 3 spend-based data.
        revenue_usd_millions: Revenue for intensity calculation.
        headcount: Employee headcount for intensity calculation.
    """
    entity_name: str = Field(
        ..., min_length=1, max_length=300, description="Entity name"
    )
    reporting_year: int = Field(
        ..., ge=1990, le=2100, description="Reporting year"
    )
    base_year: int = Field(
        ..., ge=1990, le=2100, description="Base year"
    )
    boundary_method: BoundaryMethod = Field(
        default=BoundaryMethod.OPERATIONAL_CONTROL,
        description="Organizational boundary method",
    )
    fuel_entries: List[FuelConsumptionEntry] = Field(
        default_factory=list, description="Scope 1 fuel data"
    )
    refrigerant_entries: List[RefrigerantEntry] = Field(
        default_factory=list, description="Scope 1 refrigerant data"
    )
    electricity_entries: List[ElectricityEntry] = Field(
        default_factory=list, description="Scope 2 electricity data"
    )
    scope3_entries: List[Scope3SpendEntry] = Field(
        default_factory=list, description="Scope 3 spend-based data"
    )
    revenue_usd_millions: Optional[Decimal] = Field(
        None, ge=Decimal("0"), description="Revenue ($M USD)"
    )
    headcount: Optional[int] = Field(
        None, ge=0, description="Employee headcount"
    )

    @field_validator("base_year")
    @classmethod
    def validate_base_year(cls, v: int, info: Any) -> int:
        """Validate base year is not after reporting year."""
        reporting = info.data.get("reporting_year", 2100)
        if v > reporting:
            raise ValueError(
                f"base_year ({v}) cannot be after reporting_year ({reporting})"
            )
        return v

# ---------------------------------------------------------------------------
# Pydantic Models -- Outputs
# ---------------------------------------------------------------------------

class Scope1Detail(BaseModel):
    """Scope 1 emission breakdown by source type.

    Attributes:
        stationary_combustion_tco2e: Stationary combustion emissions.
        mobile_combustion_tco2e: Mobile combustion emissions.
        process_emissions_tco2e: Process emissions.
        fugitive_emissions_tco2e: Fugitive emissions.
        refrigerants_tco2e: Refrigerant leakage emissions.
        total_tco2e: Total Scope 1 emissions.
    """
    stationary_combustion_tco2e: Decimal = Field(default=Decimal("0"))
    mobile_combustion_tco2e: Decimal = Field(default=Decimal("0"))
    process_emissions_tco2e: Decimal = Field(default=Decimal("0"))
    fugitive_emissions_tco2e: Decimal = Field(default=Decimal("0"))
    refrigerants_tco2e: Decimal = Field(default=Decimal("0"))
    total_tco2e: Decimal = Field(default=Decimal("0"))

class Scope2Detail(BaseModel):
    """Scope 2 emission breakdown.

    Attributes:
        location_based_tco2e: Location-based total.
        market_based_tco2e: Market-based total.
        total_electricity_mwh: Total electricity consumed.
    """
    location_based_tco2e: Decimal = Field(default=Decimal("0"))
    market_based_tco2e: Decimal = Field(default=Decimal("0"))
    total_electricity_mwh: Decimal = Field(default=Decimal("0"))

class Scope3Detail(BaseModel):
    """Scope 3 emission breakdown by category.

    Attributes:
        by_category: Emissions per Scope 3 category (tCO2e).
        categories_included: List of included categories.
        total_tco2e: Total Scope 3 emissions.
        upstream_tco2e: Total upstream (Cat 1-8).
        downstream_tco2e: Total downstream (Cat 9-15).
        methodology: Calculation methodology used.
    """
    by_category: Dict[str, Decimal] = Field(default_factory=dict)
    categories_included: List[str] = Field(default_factory=list)
    total_tco2e: Decimal = Field(default=Decimal("0"))
    upstream_tco2e: Decimal = Field(default=Decimal("0"))
    downstream_tco2e: Decimal = Field(default=Decimal("0"))
    methodology: str = Field(
        default="spend_based",
        description="Calculation methodology (spend_based for starter tier)",
    )

class EmissionsByGas(BaseModel):
    """Disaggregation of emissions by greenhouse gas.

    Attributes:
        co2_tco2e: CO2 emissions.
        ch4_tco2e: CH4 emissions.
        n2o_tco2e: N2O emissions.
        hfcs_tco2e: HFC emissions.
        pfcs_tco2e: PFC emissions.
        sf6_tco2e: SF6 emissions.
        nf3_tco2e: NF3 emissions.
        total_tco2e: Total across all gases.
    """
    co2_tco2e: Decimal = Field(default=Decimal("0"))
    ch4_tco2e: Decimal = Field(default=Decimal("0"))
    n2o_tco2e: Decimal = Field(default=Decimal("0"))
    hfcs_tco2e: Decimal = Field(default=Decimal("0"))
    pfcs_tco2e: Decimal = Field(default=Decimal("0"))
    sf6_tco2e: Decimal = Field(default=Decimal("0"))
    nf3_tco2e: Decimal = Field(default=Decimal("0"))
    total_tco2e: Decimal = Field(default=Decimal("0"))

class IntensityMetrics(BaseModel):
    """Emission intensity metrics.

    Attributes:
        per_revenue: tCO2e per $M USD revenue.
        per_headcount: tCO2e per employee.
    """
    per_revenue: Optional[Decimal] = Field(
        None, description="tCO2e per $M USD revenue"
    )
    per_headcount: Optional[Decimal] = Field(
        None, description="tCO2e per employee"
    )

class BaseYearAssessment(BaseModel):
    """Base year selection and validation result.

    Attributes:
        base_year: Selected base year.
        base_year_emissions_tco2e: Emissions in the base year.
        is_valid: Whether the base year passes all validation checks.
        validation_notes: Validation messages.
        years_since_base: Number of years from base year to reporting year.
    """
    base_year: int = Field(default=0)
    base_year_emissions_tco2e: Decimal = Field(default=Decimal("0"))
    is_valid: bool = Field(default=True)
    validation_notes: List[str] = Field(default_factory=list)
    years_since_base: int = Field(default=0)

class DataQualityAssessment(BaseModel):
    """Data quality assessment across all inputs.

    Attributes:
        overall_score: Weighted average quality score (0-1 scale).
        score_by_scope: Quality score per scope.
        entry_count: Total entries assessed.
        primary_data_pct: Percentage of entries with score 1 or 2.
        recommendations: Quality improvement recommendations.
    """
    overall_score: Decimal = Field(default=Decimal("0"))
    score_by_scope: Dict[str, Decimal] = Field(default_factory=dict)
    entry_count: int = Field(default=0)
    primary_data_pct: Decimal = Field(default=Decimal("0"))
    recommendations: List[str] = Field(default_factory=list)

class BaselineResult(BaseModel):
    """Complete GHG baseline assessment result.

    Attributes:
        result_id: Unique result identifier.
        engine_version: Engine version.
        calculated_at: Timestamp (UTC).
        entity_name: Reporting entity.
        reporting_year: Reporting year.
        boundary_method: Organizational boundary method used.
        scope1: Scope 1 emissions detail.
        scope2: Scope 2 emissions detail.
        scope3: Scope 3 emissions detail.
        total_tco2e: Grand total (S1 + S2 market + S3).
        total_location_based_tco2e: Total using location-based S2.
        by_gas: Emissions disaggregated by gas.
        intensity: Emission intensity metrics.
        base_year_assessment: Base year validation result.
        data_quality: Data quality assessment.
        scope1_pct: Scope 1 as % of total.
        scope2_pct: Scope 2 (market) as % of total.
        scope3_pct: Scope 3 as % of total.
        processing_time_ms: Calculation time (ms).
        provenance_hash: SHA-256 audit hash.
    """
    result_id: str = Field(default_factory=_new_uuid)
    engine_version: str = Field(default=_MODULE_VERSION)
    calculated_at: datetime = Field(default_factory=utcnow)
    entity_name: str = Field(default="")
    reporting_year: int = Field(default=0)
    boundary_method: str = Field(default="operational_control")
    scope1: Scope1Detail = Field(default_factory=Scope1Detail)
    scope2: Scope2Detail = Field(default_factory=Scope2Detail)
    scope3: Scope3Detail = Field(default_factory=Scope3Detail)
    total_tco2e: Decimal = Field(default=Decimal("0"))
    total_location_based_tco2e: Decimal = Field(default=Decimal("0"))
    by_gas: EmissionsByGas = Field(default_factory=EmissionsByGas)
    intensity: IntensityMetrics = Field(default_factory=IntensityMetrics)
    base_year_assessment: BaseYearAssessment = Field(
        default_factory=BaseYearAssessment
    )
    data_quality: DataQualityAssessment = Field(
        default_factory=DataQualityAssessment
    )
    scope1_pct: Decimal = Field(default=Decimal("0"))
    scope2_pct: Decimal = Field(default=Decimal("0"))
    scope3_pct: Decimal = Field(default=Decimal("0"))
    processing_time_ms: float = Field(default=0.0)
    provenance_hash: str = Field(default="")

# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class NetZeroBaselineEngine:
    """Unified GHG baseline assessment engine for Net Zero Starter Pack.

    Produces a complete emissions inventory covering Scope 1, 2, and 3
    with base year validation, data quality scoring, and intensity metrics.

    All calculations use Decimal arithmetic for bit-perfect reproducibility.
    No LLM is used in any calculation path.

    Usage::

        engine = NetZeroBaselineEngine()
        result = engine.calculate(baseline_input)
        assert result.provenance_hash  # non-empty SHA-256 hash
    """

    engine_version: str = _MODULE_VERSION

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #

    def calculate(self, data: BaselineInput) -> BaselineResult:
        """Run the complete baseline GHG assessment.

        Args:
            data: Validated baseline input data.

        Returns:
            BaselineResult with full emissions breakdown and provenance.
        """
        t0 = time.perf_counter()
        logger.info(
            "Calculating baseline: entity=%s, year=%d, base_year=%d",
            data.entity_name, data.reporting_year, data.base_year,
        )

        # Step 1: Calculate Scope 1
        scope1 = self._calculate_scope1(
            data.fuel_entries, data.refrigerant_entries
        )

        # Step 2: Calculate Scope 2
        scope2 = self._calculate_scope2(data.electricity_entries)

        # Step 3: Calculate Scope 3
        scope3 = self._calculate_scope3(data.scope3_entries)

        # Step 4: Totals
        total_market = _round_val(
            scope1.total_tco2e + scope2.market_based_tco2e + scope3.total_tco2e
        )
        total_location = _round_val(
            scope1.total_tco2e + scope2.location_based_tco2e + scope3.total_tco2e
        )

        # Step 5: Scope percentages
        scope1_pct = _safe_pct(scope1.total_tco2e, total_market)
        scope2_pct = _safe_pct(scope2.market_based_tco2e, total_market)
        scope3_pct = _safe_pct(scope3.total_tco2e, total_market)

        # Step 6: Gas disaggregation
        by_gas = self._disaggregate_by_gas(
            data.fuel_entries, data.refrigerant_entries, scope2, scope3
        )

        # Step 7: Intensity metrics
        intensity = self._calculate_intensity(
            total_market, data.revenue_usd_millions, data.headcount
        )

        # Step 8: Base year assessment
        base_year_assessment = self._assess_base_year(
            data.base_year, data.reporting_year, total_market
        )

        # Step 9: Data quality assessment
        dq = self._assess_data_quality(data)

        elapsed_ms = _round3((time.perf_counter() - t0) * 1000.0)

        result = BaselineResult(
            entity_name=data.entity_name,
            reporting_year=data.reporting_year,
            boundary_method=data.boundary_method.value,
            scope1=scope1,
            scope2=scope2,
            scope3=scope3,
            total_tco2e=total_market,
            total_location_based_tco2e=total_location,
            by_gas=by_gas,
            intensity=intensity,
            base_year_assessment=base_year_assessment,
            data_quality=dq,
            scope1_pct=_round_val(scope1_pct, 2),
            scope2_pct=_round_val(scope2_pct, 2),
            scope3_pct=_round_val(scope3_pct, 2),
            processing_time_ms=elapsed_ms,
        )
        result.provenance_hash = _compute_hash(result)

        logger.info(
            "Baseline complete: total=%.2f tCO2e (S1=%.2f, S2M=%.2f, "
            "S3=%.2f), hash=%s",
            float(total_market), float(scope1.total_tco2e),
            float(scope2.market_based_tco2e), float(scope3.total_tco2e),
            result.provenance_hash[:16],
        )
        return result

    def get_emission_factor(self, fuel_type: FuelType) -> Dict[str, str]:
        """Look up the emission factor for a fuel type.

        Args:
            fuel_type: Fuel type to query.

        Returns:
            Dict with 'factor' and 'unit' strings.

        Raises:
            ValueError: If fuel type is not found.
        """
        ef_data = FUEL_EMISSION_FACTORS.get(fuel_type)
        if ef_data is None:
            raise ValueError(f"Unknown fuel type: {fuel_type}")
        return {"factor": str(ef_data["factor"]), "unit": ef_data["unit"]}

    def get_grid_factor(self, region: str) -> Decimal:
        """Look up grid emission factor for a region.

        Args:
            region: Region code (e.g. 'US_AVG', 'DE', 'UK').

        Returns:
            Grid emission factor in tCO2e/MWh.

        Raises:
            ValueError: If region is not found.
        """
        factor = GRID_EMISSION_FACTORS.get(region)
        if factor is None:
            raise ValueError(
                f"Unknown region '{region}'. "
                f"Valid: {sorted(GRID_EMISSION_FACTORS.keys())}"
            )
        return factor

    def get_gwp(self, gas: str) -> Decimal:
        """Look up GWP-100 value for a greenhouse gas.

        Args:
            gas: Gas key (must match GWP_AR6).

        Returns:
            GWP-100 value as Decimal.

        Raises:
            ValueError: If gas is not found.
        """
        gwp = GWP_AR6.get(gas.lower())
        if gwp is None:
            raise ValueError(
                f"Unknown gas '{gas}'. Valid: {sorted(GWP_AR6.keys())}"
            )
        return gwp

    # ------------------------------------------------------------------ #
    # Scope 1 Calculation                                                 #
    # ------------------------------------------------------------------ #

    def _calculate_scope1(
        self,
        fuel_entries: List[FuelConsumptionEntry],
        refrigerant_entries: List[RefrigerantEntry],
    ) -> Scope1Detail:
        """Calculate total Scope 1 emissions by source type.

        Args:
            fuel_entries: Fuel consumption records.
            refrigerant_entries: Refrigerant leakage records.

        Returns:
            Scope1Detail with breakdown by source type.
        """
        stationary = Decimal("0")
        mobile = Decimal("0")
        process = Decimal("0")
        fugitive = Decimal("0")

        for entry in fuel_entries:
            tco2e = self._calculate_fuel_emission(entry)
            if entry.source_type == Scope1SourceType.STATIONARY_COMBUSTION:
                stationary += tco2e
            elif entry.source_type == Scope1SourceType.MOBILE_COMBUSTION:
                mobile += tco2e
            elif entry.source_type == Scope1SourceType.PROCESS_EMISSIONS:
                process += tco2e
            elif entry.source_type == Scope1SourceType.FUGITIVE_EMISSIONS:
                fugitive += tco2e
            else:
                stationary += tco2e

        refrigerant_total = Decimal("0")
        for ref_entry in refrigerant_entries:
            refrigerant_total += self._calculate_refrigerant_emission(ref_entry)

        stationary = _round_val(stationary)
        mobile = _round_val(mobile)
        process = _round_val(process)
        fugitive = _round_val(fugitive)
        refrigerant_total = _round_val(refrigerant_total)

        total = _round_val(
            stationary + mobile + process + fugitive + refrigerant_total
        )

        return Scope1Detail(
            stationary_combustion_tco2e=stationary,
            mobile_combustion_tco2e=mobile,
            process_emissions_tco2e=process,
            fugitive_emissions_tco2e=fugitive,
            refrigerants_tco2e=refrigerant_total,
            total_tco2e=total,
        )

    def _calculate_fuel_emission(
        self, entry: FuelConsumptionEntry
    ) -> Decimal:
        """Calculate tCO2e from a fuel consumption entry.

        Formula: tCO2e = quantity * factor / 1000

        Args:
            entry: Fuel consumption data.

        Returns:
            Emission in tCO2e.
        """
        ef_data = FUEL_EMISSION_FACTORS.get(entry.fuel_type)
        if ef_data is None:
            logger.warning(
                "No emission factor for fuel type: %s", entry.fuel_type
            )
            return Decimal("0")

        factor = ef_data["factor"]
        # Factor is in kgCO2e/unit; divide by 1000 to get tCO2e
        tco2e = entry.quantity * factor / Decimal("1000")
        return tco2e

    def _calculate_refrigerant_emission(
        self, entry: RefrigerantEntry
    ) -> Decimal:
        """Calculate tCO2e from refrigerant leakage.

        Formula: tCO2e = leakage_kg * GWP / 1000

        Args:
            entry: Refrigerant data.

        Returns:
            Emission in tCO2e.
        """
        gwp = GWP_AR6.get(entry.refrigerant_id.lower(), Decimal("0"))
        if gwp == Decimal("0"):
            logger.warning(
                "No GWP value for refrigerant: %s", entry.refrigerant_id
            )

        if entry.leakage_kg is not None and entry.leakage_kg > Decimal("0"):
            leakage = entry.leakage_kg
        else:
            leakage = entry.charge_kg * entry.leakage_rate_pct / Decimal("100")

        return leakage * gwp / Decimal("1000")

    # ------------------------------------------------------------------ #
    # Scope 2 Calculation                                                 #
    # ------------------------------------------------------------------ #

    def _calculate_scope2(
        self, electricity_entries: List[ElectricityEntry]
    ) -> Scope2Detail:
        """Calculate Scope 2 emissions (location-based and market-based).

        Args:
            electricity_entries: Electricity consumption records.

        Returns:
            Scope2Detail with location and market-based totals.
        """
        location_total = Decimal("0")
        market_total = Decimal("0")
        total_mwh = Decimal("0")

        for entry in electricity_entries:
            total_mwh += entry.quantity_mwh

            # Location-based: grid factor
            grid_factor = GRID_EMISSION_FACTORS.get(
                entry.region, GRID_EMISSION_FACTORS["GLOBAL_AVG"]
            )
            location_total += entry.quantity_mwh * grid_factor

            # Market-based: PPA > REC > residual mix
            market_total += self._calculate_market_based(entry)

        return Scope2Detail(
            location_based_tco2e=_round_val(location_total),
            market_based_tco2e=_round_val(market_total),
            total_electricity_mwh=_round_val(total_mwh),
        )

    def _calculate_market_based(self, entry: ElectricityEntry) -> Decimal:
        """Calculate market-based Scope 2 for a single entry.

        Hierarchy per GHG Protocol Scope 2 Guidance:
        1. PPA / contractual instruments
        2. RECs
        3. Residual mix

        Args:
            entry: Electricity entry.

        Returns:
            Market-based emission in tCO2e.
        """
        qty = entry.quantity_mwh

        if entry.has_ppa and entry.ppa_factor is not None:
            return qty * entry.ppa_factor

        if entry.rec_mwh > Decimal("0"):
            covered = min(entry.rec_mwh, qty)
            uncovered = qty - covered
            residual = RESIDUAL_MIX_FACTORS.get(
                entry.region, RESIDUAL_MIX_FACTORS.get("GLOBAL_AVG", Decimal("0.450"))
            )
            return uncovered * residual

        residual = RESIDUAL_MIX_FACTORS.get(
            entry.region, RESIDUAL_MIX_FACTORS.get("GLOBAL_AVG", Decimal("0.450"))
        )
        return qty * residual

    # ------------------------------------------------------------------ #
    # Scope 3 Calculation                                                 #
    # ------------------------------------------------------------------ #

    def _calculate_scope3(
        self, scope3_entries: List[Scope3SpendEntry]
    ) -> Scope3Detail:
        """Calculate Scope 3 emissions using spend-based methodology.

        Args:
            scope3_entries: Spend data per Scope 3 category.

        Returns:
            Scope3Detail with per-category breakdown.
        """
        by_category: Dict[str, Decimal] = {}
        for cat in Scope3Category:
            by_category[cat.value] = Decimal("0")

        for entry in scope3_entries:
            factor = entry.custom_factor
            if factor is None:
                factor = SPEND_BASED_FACTORS.get(
                    entry.category, Decimal("0.300")
                )
            tco2e = entry.spend_usd_thousands * factor
            by_category[entry.category.value] += tco2e

        # Round all
        for key in by_category:
            by_category[key] = _round_val(by_category[key])

        total = _round_val(sum(by_category.values()))
        included = [k for k, v in by_category.items() if v > Decimal("0")]

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

        upstream = _round_val(sum(by_category.get(c, Decimal("0")) for c in upstream_cats))
        downstream = _round_val(sum(by_category.get(c, Decimal("0")) for c in downstream_cats))

        return Scope3Detail(
            by_category=by_category,
            categories_included=sorted(included),
            total_tco2e=total,
            upstream_tco2e=upstream,
            downstream_tco2e=downstream,
            methodology="spend_based",
        )

    # ------------------------------------------------------------------ #
    # Gas Disaggregation                                                  #
    # ------------------------------------------------------------------ #

    def _disaggregate_by_gas(
        self,
        fuel_entries: List[FuelConsumptionEntry],
        refrigerant_entries: List[RefrigerantEntry],
        scope2: Scope2Detail,
        scope3: Scope3Detail,
    ) -> EmissionsByGas:
        """Approximate gas-level disaggregation.

        For the starter tier, fuel combustion is assumed predominantly CO2
        with small CH4 and N2O fractions.  Refrigerants map to HFCs.
        Scope 2 and 3 are reported as CO2.

        Args:
            fuel_entries: Scope 1 fuel entries.
            refrigerant_entries: Scope 1 refrigerant entries.
            scope2: Scope 2 detail.
            scope3: Scope 3 detail.

        Returns:
            EmissionsByGas with approximate disaggregation.
        """
        co2_total = Decimal("0")
        ch4_total = Decimal("0")
        n2o_total = Decimal("0")
        hfcs_total = Decimal("0")

        # Fuel combustion: ~95% CO2, ~3% CH4, ~2% N2O (typical split)
        for entry in fuel_entries:
            tco2e = self._calculate_fuel_emission(entry)
            co2_total += tco2e * Decimal("0.95")
            ch4_total += tco2e * Decimal("0.03")
            n2o_total += tco2e * Decimal("0.02")

        # Refrigerants -> HFCs
        for ref_entry in refrigerant_entries:
            hfcs_total += self._calculate_refrigerant_emission(ref_entry)

        # Scope 2 and 3 -> CO2 (simplified for starter tier)
        co2_total += scope2.market_based_tco2e
        co2_total += scope3.total_tco2e

        co2_total = _round_val(co2_total)
        ch4_total = _round_val(ch4_total)
        n2o_total = _round_val(n2o_total)
        hfcs_total = _round_val(hfcs_total)
        total = _round_val(co2_total + ch4_total + n2o_total + hfcs_total)

        return EmissionsByGas(
            co2_tco2e=co2_total,
            ch4_tco2e=ch4_total,
            n2o_tco2e=n2o_total,
            hfcs_tco2e=hfcs_total,
            total_tco2e=total,
        )

    # ------------------------------------------------------------------ #
    # Intensity Metrics                                                   #
    # ------------------------------------------------------------------ #

    def _calculate_intensity(
        self,
        total_tco2e: Decimal,
        revenue_usd_millions: Optional[Decimal],
        headcount: Optional[int],
    ) -> IntensityMetrics:
        """Calculate emission intensity metrics.

        Args:
            total_tco2e: Total emissions.
            revenue_usd_millions: Revenue in $M USD.
            headcount: Employee count.

        Returns:
            IntensityMetrics with per-revenue and per-headcount values.
        """
        per_revenue = None
        per_headcount = None

        if revenue_usd_millions is not None and revenue_usd_millions > Decimal("0"):
            per_revenue = _round_val(
                _safe_divide(total_tco2e, revenue_usd_millions), 3
            )

        if headcount is not None and headcount > 0:
            per_headcount = _round_val(
                _safe_divide(total_tco2e, _decimal(headcount)), 3
            )

        return IntensityMetrics(
            per_revenue=per_revenue,
            per_headcount=per_headcount,
        )

    # ------------------------------------------------------------------ #
    # Base Year Assessment                                                #
    # ------------------------------------------------------------------ #

    def _assess_base_year(
        self,
        base_year: int,
        reporting_year: int,
        total_tco2e: Decimal,
    ) -> BaseYearAssessment:
        """Assess and validate base year selection per GHG Protocol Ch 5-6.

        Validation rules:
        - Base year must not be in the future
        - GHG Protocol recommends most recent year with complete data
        - SBTi requires base year within 5 years of target submission
        - Warn if base year is more than 10 years old

        Args:
            base_year: Proposed base year.
            reporting_year: Current reporting year.
            total_tco2e: Total emissions for the reporting year.

        Returns:
            BaseYearAssessment with validity and notes.
        """
        years_since = reporting_year - base_year
        notes: List[str] = []
        is_valid = True

        if base_year > reporting_year:
            is_valid = False
            notes.append("Base year cannot be in the future.")

        if years_since > 10:
            notes.append(
                f"Base year is {years_since} years old. "
                "Consider updating to a more recent base year for "
                "SBTi alignment (max 5 years recommended)."
            )

        if years_since > 5:
            notes.append(
                "SBTi requires base year within 5 years of target "
                "submission. Current base year may not qualify."
            )

        if years_since == 0:
            notes.append(
                "Base year equals reporting year. This is the initial "
                "baseline establishment."
            )
        elif years_since <= 2:
            notes.append("Base year is recent and well-suited for SBTi targets.")

        if total_tco2e <= Decimal("0"):
            notes.append(
                "Total emissions are zero. Verify data completeness before "
                "finalizing base year."
            )
            is_valid = False

        return BaseYearAssessment(
            base_year=base_year,
            base_year_emissions_tco2e=total_tco2e,
            is_valid=is_valid,
            validation_notes=notes,
            years_since_base=years_since,
        )

    # ------------------------------------------------------------------ #
    # Data Quality Assessment                                             #
    # ------------------------------------------------------------------ #

    def _assess_data_quality(
        self, data: BaselineInput
    ) -> DataQualityAssessment:
        """Assess data quality across all input entries.

        Calculates weighted average quality score and provides
        improvement recommendations.

        Args:
            data: Full baseline input.

        Returns:
            DataQualityAssessment with scores and recommendations.
        """
        scores: List[Tuple[str, Decimal]] = []

        for entry in data.fuel_entries:
            scores.append(
                ("scope_1", DQ_NUMERIC.get(entry.data_quality.value, Decimal("0.40")))
            )
        for entry in data.refrigerant_entries:
            scores.append(
                ("scope_1", DQ_NUMERIC.get(entry.data_quality.value, Decimal("0.40")))
            )
        for entry in data.electricity_entries:
            scores.append(
                ("scope_2", DQ_NUMERIC.get(entry.data_quality.value, Decimal("0.40")))
            )
        for entry in data.scope3_entries:
            scores.append(
                ("scope_3", DQ_NUMERIC.get(entry.data_quality.value, Decimal("0.40")))
            )

        if not scores:
            return DataQualityAssessment(
                recommendations=["No data entries provided for quality assessment."]
            )

        # Overall average
        total_score = sum(s for _, s in scores)
        overall = _round_val(
            _safe_divide(total_score, _decimal(len(scores))), 3
        )

        # By scope
        scope_scores: Dict[str, List[Decimal]] = {}
        for scope_key, score in scores:
            scope_scores.setdefault(scope_key, []).append(score)

        score_by_scope: Dict[str, Decimal] = {}
        for scope_key, scope_list in scope_scores.items():
            avg = _safe_divide(sum(scope_list), _decimal(len(scope_list)))
            score_by_scope[scope_key] = _round_val(avg, 3)

        # Primary data percentage (score 1 or 2)
        primary_count = sum(1 for _, s in scores if s >= Decimal("0.80"))
        primary_pct = _round_val(
            _safe_pct(_decimal(primary_count), _decimal(len(scores))), 1
        )

        # Recommendations
        recommendations: List[str] = []
        if overall < Decimal("0.60"):
            recommendations.append(
                "Overall data quality is low. Prioritize primary data "
                "collection for Scope 1 and 2 sources."
            )
        if score_by_scope.get("scope_3", Decimal("0")) < Decimal("0.50"):
            recommendations.append(
                "Scope 3 data quality is low (typical for spend-based). "
                "Consider upgrading to supplier-specific data for top "
                "categories."
            )
        if primary_pct < Decimal("50"):
            recommendations.append(
                f"Only {primary_pct}% of data is primary/high quality. "
                "Target 50%+ primary data for next reporting cycle."
            )

        return DataQualityAssessment(
            overall_score=overall,
            score_by_scope=score_by_scope,
            entry_count=len(scores),
            primary_data_pct=primary_pct,
            recommendations=recommendations,
        )

    # ------------------------------------------------------------------ #
    # Summary Helpers                                                     #
    # ------------------------------------------------------------------ #

    def get_scope_summary(self, result: BaselineResult) -> Dict[str, Any]:
        """Generate a scope-level summary from a baseline result.

        Args:
            result: BaselineResult to summarize.

        Returns:
            Dict with scope totals, percentages, and metadata.
        """
        summary = {
            "entity_name": result.entity_name,
            "reporting_year": result.reporting_year,
            "scope1_tco2e": str(result.scope1.total_tco2e),
            "scope1_pct": str(result.scope1_pct),
            "scope2_location_tco2e": str(result.scope2.location_based_tco2e),
            "scope2_market_tco2e": str(result.scope2.market_based_tco2e),
            "scope2_pct": str(result.scope2_pct),
            "scope3_tco2e": str(result.scope3.total_tco2e),
            "scope3_pct": str(result.scope3_pct),
            "total_tco2e": str(result.total_tco2e),
            "data_quality_score": str(result.data_quality.overall_score),
            "base_year": result.base_year_assessment.base_year,
            "base_year_valid": result.base_year_assessment.is_valid,
        }
        summary["provenance_hash"] = _compute_hash(summary)
        return summary
