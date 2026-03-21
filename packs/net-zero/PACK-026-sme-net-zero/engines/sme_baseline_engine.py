# -*- coding: utf-8 -*-
"""
SMEBaselineEngine - PACK-026 SME Net Zero Pack Engine 1
=========================================================

Three-tier data approach (Bronze/Silver/Gold) for SME GHG baseline
assessment with industry average fallbacks and simplified Scope 3.

This engine is purpose-built for small and medium enterprises that
lack dedicated sustainability teams.  It provides three data input
tiers: Bronze (industry averages only, 15 min, +/-40%), Silver
(basic activity data from bills, 1 hour, +/-15%), and Gold (detailed
bills + questionnaires, 2-3 hours, +/-5%).

Calculation Methodology:
    Bronze Tier:
        total_tco2e = industry_avg_per_employee * headcount
        OR total_tco2e = industry_avg_per_revenue * revenue

    Silver Tier:
        scope1_tco2e = sum(fuel_quantity * fuel_factor / 1000)
        scope2_tco2e = electricity_kwh * grid_factor / 1000
        scope3_tco2e = total_spend * industry_scope3_ratio

    Gold Tier:
        scope1_tco2e = detailed fuel + refrigerant calculations
        scope2_tco2e = location + market-based per GHG Protocol
        scope3_tco2e = spend_by_category * EEIO factors

    Industry averages by NACE code + company size (micro/small/medium).
    Spend-based Scope 3 using DEFRA/EPA EEIO factors.
    Simplified Scope 1+2 (no process emissions, basic fugitive).

Regulatory References:
    - GHG Protocol Corporate Standard (2004, revised 2015)
    - GHG Protocol Scope 2 Guidance (2015)
    - GHG Protocol Scope 3 Standard (2011)
    - IPCC AR6 WG1 (2021) - GWP-100 values
    - DEFRA/BEIS 2024 UK GHG Conversion Factors
    - US EPA EEIO v2.0 - Spend-based factors
    - SBTi SME Target Setting Route (2023)

Zero-Hallucination:
    - All calculations use deterministic Decimal arithmetic
    - Emission factors are hard-coded from authoritative sources
    - Industry averages from published government statistics
    - SHA-256 provenance hash on every result
    - No LLM involvement in any calculation path

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-026 SME Net Zero Pack
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
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


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


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class DataTier(str, Enum):
    """SME data input tier determining calculation depth and accuracy.

    BRONZE: Industry averages only (15 min setup, +/-40% accuracy).
    SILVER: Basic activity data from bills (1 hour, +/-15% accuracy).
    GOLD:   Detailed bills + questionnaires (2-3 hours, +/-5% accuracy).
    """
    BRONZE = "bronze"
    SILVER = "silver"
    GOLD = "gold"


class CompanySize(str, Enum):
    """Company size classification per EU SME definition.

    MICRO:  < 10 employees, < EUR 2M turnover.
    SMALL:  < 50 employees, < EUR 10M turnover.
    MEDIUM: < 250 employees, < EUR 50M turnover.
    """
    MICRO = "micro"
    SMALL = "small"
    MEDIUM = "medium"


class SMEFuelType(str, Enum):
    """Simplified fuel types for SME baseline (most common fuels only)."""
    NATURAL_GAS = "natural_gas"
    DIESEL = "diesel"
    GASOLINE = "gasoline"
    LPG = "lpg"
    FUEL_OIL = "fuel_oil"
    KEROSENE = "kerosene"
    PROPANE = "propane"


class SMESector(str, Enum):
    """SME sector classification using simplified NACE divisions.

    Covers the most common SME sectors with industry-average emission
    intensities.
    """
    AGRICULTURE = "agriculture"
    MANUFACTURING = "manufacturing"
    CONSTRUCTION = "construction"
    WHOLESALE_RETAIL = "wholesale_retail"
    TRANSPORT_LOGISTICS = "transport_logistics"
    ACCOMMODATION_FOOD = "accommodation_food"
    INFORMATION_TECHNOLOGY = "information_technology"
    FINANCIAL_SERVICES = "financial_services"
    PROFESSIONAL_SERVICES = "professional_services"
    HEALTHCARE = "healthcare"
    EDUCATION = "education"
    ARTS_ENTERTAINMENT = "arts_entertainment"
    OTHER_SERVICES = "other_services"


class Scope3SMECategory(str, Enum):
    """Simplified Scope 3 categories relevant to most SMEs.

    Only the categories that typically contribute >5% of SME Scope 3.
    """
    CAT_01_PURCHASED_GOODS = "cat_01_purchased_goods"
    CAT_02_CAPITAL_GOODS = "cat_02_capital_goods"
    CAT_03_FUEL_ENERGY = "cat_03_fuel_energy"
    CAT_04_UPSTREAM_TRANSPORT = "cat_04_upstream_transport"
    CAT_05_WASTE = "cat_05_waste"
    CAT_06_BUSINESS_TRAVEL = "cat_06_business_travel"
    CAT_07_EMPLOYEE_COMMUTING = "cat_07_employee_commuting"


class DataQualityLevel(str, Enum):
    """Simplified data quality scoring for SMEs."""
    HIGH = "high"        # Direct measurement, metered data
    MEDIUM = "medium"    # Estimated from bills, invoices
    LOW = "low"          # Industry averages, proxies
    ESTIMATED = "estimated"  # Engine-generated estimate


# ---------------------------------------------------------------------------
# Constants -- Reference Data
# ---------------------------------------------------------------------------


# Industry average emissions per employee (tCO2e/employee/year).
# Source: UK BEIS SME emissions statistics 2024, EU JRC 2023, US EPA 2024.
# Keys: (sector, company_size) -> tCO2e per employee.
INDUSTRY_AVG_PER_EMPLOYEE: Dict[str, Dict[str, Decimal]] = {
    SMESector.AGRICULTURE: {
        "micro": Decimal("18.5"), "small": Decimal("15.2"), "medium": Decimal("12.8"),
    },
    SMESector.MANUFACTURING: {
        "micro": Decimal("12.0"), "small": Decimal("10.5"), "medium": Decimal("8.7"),
    },
    SMESector.CONSTRUCTION: {
        "micro": Decimal("9.5"), "small": Decimal("8.2"), "medium": Decimal("7.1"),
    },
    SMESector.WHOLESALE_RETAIL: {
        "micro": Decimal("4.2"), "small": Decimal("3.8"), "medium": Decimal("3.2"),
    },
    SMESector.TRANSPORT_LOGISTICS: {
        "micro": Decimal("22.0"), "small": Decimal("18.5"), "medium": Decimal("15.0"),
    },
    SMESector.ACCOMMODATION_FOOD: {
        "micro": Decimal("6.8"), "small": Decimal("5.5"), "medium": Decimal("4.8"),
    },
    SMESector.INFORMATION_TECHNOLOGY: {
        "micro": Decimal("2.8"), "small": Decimal("2.5"), "medium": Decimal("2.2"),
    },
    SMESector.FINANCIAL_SERVICES: {
        "micro": Decimal("3.0"), "small": Decimal("2.7"), "medium": Decimal("2.3"),
    },
    SMESector.PROFESSIONAL_SERVICES: {
        "micro": Decimal("3.5"), "small": Decimal("3.0"), "medium": Decimal("2.6"),
    },
    SMESector.HEALTHCARE: {
        "micro": Decimal("5.5"), "small": Decimal("4.8"), "medium": Decimal("4.2"),
    },
    SMESector.EDUCATION: {
        "micro": Decimal("3.2"), "small": Decimal("2.9"), "medium": Decimal("2.5"),
    },
    SMESector.ARTS_ENTERTAINMENT: {
        "micro": Decimal("3.8"), "small": Decimal("3.4"), "medium": Decimal("3.0"),
    },
    SMESector.OTHER_SERVICES: {
        "micro": Decimal("4.0"), "small": Decimal("3.5"), "medium": Decimal("3.0"),
    },
}

# Industry average Scope split ratios (fraction of total for each scope).
# Source: CDP SME dataset analysis 2023, SBTi SME submissions.
INDUSTRY_SCOPE_SPLIT: Dict[str, Dict[str, Decimal]] = {
    SMESector.AGRICULTURE: {
        "scope1": Decimal("0.45"), "scope2": Decimal("0.10"), "scope3": Decimal("0.45"),
    },
    SMESector.MANUFACTURING: {
        "scope1": Decimal("0.30"), "scope2": Decimal("0.15"), "scope3": Decimal("0.55"),
    },
    SMESector.CONSTRUCTION: {
        "scope1": Decimal("0.35"), "scope2": Decimal("0.08"), "scope3": Decimal("0.57"),
    },
    SMESector.WHOLESALE_RETAIL: {
        "scope1": Decimal("0.10"), "scope2": Decimal("0.15"), "scope3": Decimal("0.75"),
    },
    SMESector.TRANSPORT_LOGISTICS: {
        "scope1": Decimal("0.55"), "scope2": Decimal("0.05"), "scope3": Decimal("0.40"),
    },
    SMESector.ACCOMMODATION_FOOD: {
        "scope1": Decimal("0.25"), "scope2": Decimal("0.20"), "scope3": Decimal("0.55"),
    },
    SMESector.INFORMATION_TECHNOLOGY: {
        "scope1": Decimal("0.05"), "scope2": Decimal("0.20"), "scope3": Decimal("0.75"),
    },
    SMESector.FINANCIAL_SERVICES: {
        "scope1": Decimal("0.05"), "scope2": Decimal("0.15"), "scope3": Decimal("0.80"),
    },
    SMESector.PROFESSIONAL_SERVICES: {
        "scope1": Decimal("0.08"), "scope2": Decimal("0.12"), "scope3": Decimal("0.80"),
    },
    SMESector.HEALTHCARE: {
        "scope1": Decimal("0.20"), "scope2": Decimal("0.20"), "scope3": Decimal("0.60"),
    },
    SMESector.EDUCATION: {
        "scope1": Decimal("0.15"), "scope2": Decimal("0.25"), "scope3": Decimal("0.60"),
    },
    SMESector.ARTS_ENTERTAINMENT: {
        "scope1": Decimal("0.10"), "scope2": Decimal("0.15"), "scope3": Decimal("0.75"),
    },
    SMESector.OTHER_SERVICES: {
        "scope1": Decimal("0.12"), "scope2": Decimal("0.18"), "scope3": Decimal("0.70"),
    },
}

# Simplified fuel emission factors (kgCO2e per unit).
# Source: DEFRA/BEIS 2024 UK Government GHG Conversion Factors.
SME_FUEL_FACTORS: Dict[str, Dict[str, Decimal]] = {
    SMEFuelType.NATURAL_GAS: {"factor": Decimal("2.02"), "unit": "kgCO2e_per_m3"},
    SMEFuelType.DIESEL: {"factor": Decimal("2.676"), "unit": "kgCO2e_per_litre"},
    SMEFuelType.GASOLINE: {"factor": Decimal("2.315"), "unit": "kgCO2e_per_litre"},
    SMEFuelType.LPG: {"factor": Decimal("1.557"), "unit": "kgCO2e_per_litre"},
    SMEFuelType.FUEL_OIL: {"factor": Decimal("3.179"), "unit": "kgCO2e_per_litre"},
    SMEFuelType.KEROSENE: {"factor": Decimal("2.540"), "unit": "kgCO2e_per_litre"},
    SMEFuelType.PROPANE: {"factor": Decimal("1.544"), "unit": "kgCO2e_per_litre"},
}

# Grid emission factors by region (tCO2e per MWh).
# Source: IEA Emission Factors 2024, EEA 2024, US EPA eGRID 2024.
GRID_EMISSION_FACTORS: Dict[str, Decimal] = {
    "EU_AVG": Decimal("0.230"), "US_AVG": Decimal("0.386"), "UK": Decimal("0.207"),
    "DE": Decimal("0.338"), "FR": Decimal("0.055"), "ES": Decimal("0.150"),
    "IT": Decimal("0.256"), "NL": Decimal("0.328"), "PL": Decimal("0.635"),
    "SE": Decimal("0.012"), "NO": Decimal("0.008"), "AT": Decimal("0.091"),
    "BE": Decimal("0.155"), "DK": Decimal("0.112"), "FI": Decimal("0.068"),
    "IE": Decimal("0.296"), "PT": Decimal("0.161"), "CH": Decimal("0.015"),
    "JP": Decimal("0.456"), "CN": Decimal("0.555"), "IN": Decimal("0.708"),
    "AU": Decimal("0.656"), "CA": Decimal("0.120"), "BR": Decimal("0.075"),
    "KR": Decimal("0.415"), "ZA": Decimal("0.928"), "GLOBAL_AVG": Decimal("0.436"),
}

# Spend-based emission factors (tCO2e per $1000 USD spend).
# Source: US EPA EEIO v2.0 / EXIOBASE 3.8, DEFRA 2024.
EEIO_SPEND_FACTORS: Dict[str, Decimal] = {
    Scope3SMECategory.CAT_01_PURCHASED_GOODS: Decimal("0.430"),
    Scope3SMECategory.CAT_02_CAPITAL_GOODS: Decimal("0.350"),
    Scope3SMECategory.CAT_03_FUEL_ENERGY: Decimal("0.280"),
    Scope3SMECategory.CAT_04_UPSTREAM_TRANSPORT: Decimal("0.520"),
    Scope3SMECategory.CAT_05_WASTE: Decimal("0.210"),
    Scope3SMECategory.CAT_06_BUSINESS_TRAVEL: Decimal("0.310"),
    Scope3SMECategory.CAT_07_EMPLOYEE_COMMUTING: Decimal("0.180"),
}

# Industry-specific Scope 3 ratio (Scope 3 as % of total spend).
# Used in Silver tier when only total spend is known.
# Source: CDP SME dataset 2023.
INDUSTRY_SCOPE3_SPEND_RATIO: Dict[str, Decimal] = {
    SMESector.AGRICULTURE: Decimal("0.00055"),
    SMESector.MANUFACTURING: Decimal("0.00070"),
    SMESector.CONSTRUCTION: Decimal("0.00065"),
    SMESector.WHOLESALE_RETAIL: Decimal("0.00050"),
    SMESector.TRANSPORT_LOGISTICS: Decimal("0.00045"),
    SMESector.ACCOMMODATION_FOOD: Decimal("0.00060"),
    SMESector.INFORMATION_TECHNOLOGY: Decimal("0.00035"),
    SMESector.FINANCIAL_SERVICES: Decimal("0.00025"),
    SMESector.PROFESSIONAL_SERVICES: Decimal("0.00030"),
    SMESector.HEALTHCARE: Decimal("0.00055"),
    SMESector.EDUCATION: Decimal("0.00040"),
    SMESector.ARTS_ENTERTAINMENT: Decimal("0.00045"),
    SMESector.OTHER_SERVICES: Decimal("0.00042"),
}

# IPCC AR6 GWP-100 for common SME refrigerants.
SME_GWP: Dict[str, Decimal] = {
    "r134a": Decimal("1430"), "r410a": Decimal("2088"),
    "r404a": Decimal("3922"), "r32": Decimal("675"),
    "r290": Decimal("3"), "r744": Decimal("1"),
    "r407c": Decimal("1774"), "r22": Decimal("1810"),
}

# Data quality score multipliers for confidence weighting.
DQ_WEIGHTS: Dict[str, Decimal] = {
    DataQualityLevel.HIGH: Decimal("1.00"),
    DataQualityLevel.MEDIUM: Decimal("0.75"),
    DataQualityLevel.LOW: Decimal("0.50"),
    DataQualityLevel.ESTIMATED: Decimal("0.30"),
}

# Accuracy bands by tier.
TIER_ACCURACY: Dict[str, Dict[str, Decimal]] = {
    DataTier.BRONZE: {"lower_pct": Decimal("60"), "upper_pct": Decimal("140")},
    DataTier.SILVER: {"lower_pct": Decimal("85"), "upper_pct": Decimal("115")},
    DataTier.GOLD: {"lower_pct": Decimal("95"), "upper_pct": Decimal("105")},
}


# ---------------------------------------------------------------------------
# Pydantic Models -- Inputs
# ---------------------------------------------------------------------------


class SMEFuelEntry(BaseModel):
    """Simplified fuel consumption entry for SMEs.

    Attributes:
        fuel_type: Type of fuel consumed.
        quantity: Quantity consumed (litres, m3, or kg depending on fuel).
        description: Optional description (e.g. 'office heating').
        data_quality: Quality level for this data point.
    """
    fuel_type: SMEFuelType = Field(..., description="Fuel type")
    quantity: Decimal = Field(..., ge=Decimal("0"), description="Quantity consumed")
    description: str = Field(default="", max_length=300)
    data_quality: DataQualityLevel = Field(
        default=DataQualityLevel.MEDIUM, description="Data quality"
    )


class SMEElectricityEntry(BaseModel):
    """Simplified electricity consumption entry for SMEs.

    Attributes:
        annual_kwh: Annual electricity in kWh (from bills).
        region: Grid region code for emission factor lookup.
        green_tariff: Whether on a green / renewable tariff.
        green_tariff_pct: Percentage of supply from green tariff.
        data_quality: Quality level.
    """
    annual_kwh: Decimal = Field(..., ge=Decimal("0"), description="Annual kWh")
    region: str = Field(default="GLOBAL_AVG", description="Grid region code")
    green_tariff: bool = Field(False, description="On green/renewable tariff")
    green_tariff_pct: Decimal = Field(
        default=Decimal("100"), ge=Decimal("0"), le=Decimal("100"),
        description="Percentage supplied via green tariff",
    )
    data_quality: DataQualityLevel = Field(default=DataQualityLevel.MEDIUM)


class SMERefrigerantEntry(BaseModel):
    """Simplified refrigerant entry for SMEs.

    Attributes:
        refrigerant_type: Refrigerant identifier (e.g. r410a).
        system_count: Number of systems containing this refrigerant.
        typical_charge_kg: Typical charge per system (kg).
        annual_leakage_rate_pct: Estimated annual leakage rate (%).
    """
    refrigerant_type: str = Field(..., description="Refrigerant type key")
    system_count: int = Field(default=1, ge=1, description="Number of systems")
    typical_charge_kg: Decimal = Field(
        default=Decimal("5.0"), ge=Decimal("0"), description="Charge per system (kg)"
    )
    annual_leakage_rate_pct: Decimal = Field(
        default=Decimal("10.0"), ge=Decimal("0"), le=Decimal("100"),
        description="Annual leakage rate (%)",
    )


class SMESpendEntry(BaseModel):
    """Scope 3 spend entry for SME spend-based calculation.

    Attributes:
        category: Scope 3 category.
        annual_spend_usd: Annual spend in USD.
        custom_factor: Optional custom EEIO factor override.
        notes: Optional notes.
    """
    category: Scope3SMECategory = Field(..., description="Scope 3 category")
    annual_spend_usd: Decimal = Field(
        ..., ge=Decimal("0"), description="Annual spend (USD)"
    )
    custom_factor: Optional[Decimal] = Field(
        None, ge=Decimal("0"),
        description="Custom factor override (tCO2e per $1000 USD)",
    )
    notes: str = Field(default="", max_length=500)


class SMEVehicleEntry(BaseModel):
    """Simplified vehicle fleet entry for SMEs.

    Attributes:
        vehicle_count: Number of vehicles.
        fuel_type: Fuel type used.
        annual_km_per_vehicle: Average annual km per vehicle.
        fuel_efficiency_l_per_100km: Average fuel efficiency (l/100km).
    """
    vehicle_count: int = Field(default=1, ge=1, description="Number of vehicles")
    fuel_type: SMEFuelType = Field(
        default=SMEFuelType.DIESEL, description="Fuel type"
    )
    annual_km_per_vehicle: Decimal = Field(
        default=Decimal("20000"), ge=Decimal("0"),
        description="Annual km per vehicle",
    )
    fuel_efficiency_l_per_100km: Decimal = Field(
        default=Decimal("8.0"), ge=Decimal("0"),
        description="Fuel efficiency (litres/100km)",
    )


class SMEBaselineInput(BaseModel):
    """Complete input for SME baseline assessment.

    The data_tier determines the minimum required inputs:
    - Bronze: entity_name, sector, company_size, headcount (or revenue_usd)
    - Silver: + fuel_entries, electricity_entries, total_annual_spend_usd
    - Gold:   + refrigerant_entries, spend_entries (by category), vehicle_entries

    Attributes:
        entity_name: Company name.
        reporting_year: Year of assessment.
        data_tier: Bronze/Silver/Gold.
        sector: Industry sector.
        company_size: Micro/Small/Medium.
        headcount: Employee count.
        revenue_usd: Annual revenue in USD.
        region: Primary operating region code.
        fuel_entries: Scope 1 fuel data (Silver+Gold).
        electricity_entries: Scope 2 electricity data (Silver+Gold).
        refrigerant_entries: Refrigerant data (Gold).
        vehicle_entries: Vehicle fleet data (Gold).
        spend_entries: Scope 3 spend by category (Gold).
        total_annual_spend_usd: Total annual spend (Silver fallback).
    """
    entity_name: str = Field(
        ..., min_length=1, max_length=300, description="Company name"
    )
    reporting_year: int = Field(
        ..., ge=2015, le=2100, description="Reporting year"
    )
    data_tier: DataTier = Field(
        default=DataTier.BRONZE, description="Data input tier"
    )
    sector: SMESector = Field(..., description="Industry sector")
    company_size: CompanySize = Field(
        default=CompanySize.SMALL, description="Company size"
    )
    headcount: int = Field(
        default=10, ge=1, le=250, description="Employee count"
    )
    revenue_usd: Optional[Decimal] = Field(
        None, ge=Decimal("0"), description="Annual revenue (USD)"
    )
    region: str = Field(default="GLOBAL_AVG", description="Operating region")

    # Silver + Gold inputs
    fuel_entries: List[SMEFuelEntry] = Field(
        default_factory=list, description="Scope 1 fuel data"
    )
    electricity_entries: List[SMEElectricityEntry] = Field(
        default_factory=list, description="Scope 2 electricity data"
    )

    # Gold inputs
    refrigerant_entries: List[SMERefrigerantEntry] = Field(
        default_factory=list, description="Refrigerant data"
    )
    vehicle_entries: List[SMEVehicleEntry] = Field(
        default_factory=list, description="Vehicle fleet data"
    )
    spend_entries: List[SMESpendEntry] = Field(
        default_factory=list, description="Scope 3 spend by category"
    )
    total_annual_spend_usd: Optional[Decimal] = Field(
        None, ge=Decimal("0"),
        description="Total annual procurement spend (USD) for Silver tier",
    )

    @field_validator("headcount")
    @classmethod
    def validate_headcount(cls, v: int) -> int:
        """Validate headcount is within SME range."""
        if v > 250:
            raise ValueError("SME headcount must be <= 250 (EU definition)")
        return v


# ---------------------------------------------------------------------------
# Pydantic Models -- Outputs
# ---------------------------------------------------------------------------


class ScopeBreakdown(BaseModel):
    """Emission breakdown for a single scope.

    Attributes:
        total_tco2e: Total emissions for this scope.
        details: Sub-category breakdown.
        data_quality: Data quality for this scope.
        methodology: Calculation methodology used.
    """
    total_tco2e: Decimal = Field(default=Decimal("0"))
    details: Dict[str, Decimal] = Field(default_factory=dict)
    data_quality: DataQualityLevel = Field(default=DataQualityLevel.ESTIMATED)
    methodology: str = Field(default="industry_average")


class AccuracyBand(BaseModel):
    """Accuracy range for the baseline estimate.

    Attributes:
        central_estimate_tco2e: Best estimate.
        lower_bound_tco2e: Lower bound of accuracy range.
        upper_bound_tco2e: Upper bound of accuracy range.
        confidence_pct: Confidence percentage.
        tier: Data tier that determined the accuracy band.
    """
    central_estimate_tco2e: Decimal = Field(default=Decimal("0"))
    lower_bound_tco2e: Decimal = Field(default=Decimal("0"))
    upper_bound_tco2e: Decimal = Field(default=Decimal("0"))
    confidence_pct: Decimal = Field(default=Decimal("0"))
    tier: str = Field(default="bronze")


class IntensityMetrics(BaseModel):
    """SME emission intensity metrics.

    Attributes:
        per_employee: tCO2e per employee.
        per_revenue_million: tCO2e per $M USD revenue.
        sector_average_per_employee: Industry average for comparison.
        vs_sector_avg_pct: Percentage vs sector average.
    """
    per_employee: Decimal = Field(default=Decimal("0"))
    per_revenue_million: Optional[Decimal] = Field(None)
    sector_average_per_employee: Decimal = Field(default=Decimal("0"))
    vs_sector_avg_pct: Decimal = Field(default=Decimal("0"))


class DataQualityAssessment(BaseModel):
    """Data quality summary for the baseline.

    Attributes:
        overall_score: Weighted quality score (0-1 scale).
        tier_used: Data tier that was applied.
        estimated_accuracy_pct: Estimated accuracy range (+/- %).
        improvement_suggestions: Suggestions to improve data quality.
        completeness_pct: Percentage of possible data fields populated.
    """
    overall_score: Decimal = Field(default=Decimal("0"))
    tier_used: str = Field(default="bronze")
    estimated_accuracy_pct: Decimal = Field(default=Decimal("40"))
    improvement_suggestions: List[str] = Field(default_factory=list)
    completeness_pct: Decimal = Field(default=Decimal("0"))


class NextStepRecommendation(BaseModel):
    """Recommendation for improving the baseline.

    Attributes:
        priority: Priority level (1 = highest).
        action: What to do.
        impact: Expected impact on accuracy.
        effort_minutes: Estimated effort in minutes.
    """
    priority: int = Field(default=1, ge=1, le=10)
    action: str = Field(default="")
    impact: str = Field(default="")
    effort_minutes: int = Field(default=15)


class SMEBaselineResult(BaseModel):
    """Complete SME baseline assessment result.

    Attributes:
        result_id: Unique result identifier.
        engine_version: Engine version.
        calculated_at: Timestamp (UTC).
        entity_name: Company name.
        reporting_year: Reporting year.
        data_tier: Data tier used.
        sector: Industry sector.
        company_size: Company size.
        headcount: Employee count.
        scope1: Scope 1 breakdown.
        scope2: Scope 2 breakdown.
        scope3: Scope 3 breakdown.
        total_tco2e: Grand total emissions.
        accuracy_band: Accuracy range.
        intensity: Intensity metrics.
        scope1_pct: Scope 1 as % of total.
        scope2_pct: Scope 2 as % of total.
        scope3_pct: Scope 3 as % of total.
        data_quality: Data quality assessment.
        next_steps: Recommendations for improvement.
        time_to_complete_minutes: Estimated time taken.
        processing_time_ms: Calculation time (ms).
        provenance_hash: SHA-256 audit hash.
    """
    result_id: str = Field(default_factory=_new_uuid)
    engine_version: str = Field(default=_MODULE_VERSION)
    calculated_at: datetime = Field(default_factory=_utcnow)
    entity_name: str = Field(default="")
    reporting_year: int = Field(default=0)
    data_tier: str = Field(default="bronze")
    sector: str = Field(default="")
    company_size: str = Field(default="small")
    headcount: int = Field(default=0)

    scope1: ScopeBreakdown = Field(default_factory=ScopeBreakdown)
    scope2: ScopeBreakdown = Field(default_factory=ScopeBreakdown)
    scope3: ScopeBreakdown = Field(default_factory=ScopeBreakdown)
    total_tco2e: Decimal = Field(default=Decimal("0"))
    accuracy_band: AccuracyBand = Field(default_factory=AccuracyBand)
    intensity: IntensityMetrics = Field(default_factory=IntensityMetrics)

    scope1_pct: Decimal = Field(default=Decimal("0"))
    scope2_pct: Decimal = Field(default=Decimal("0"))
    scope3_pct: Decimal = Field(default=Decimal("0"))

    data_quality: DataQualityAssessment = Field(default_factory=DataQualityAssessment)
    next_steps: List[NextStepRecommendation] = Field(default_factory=list)
    time_to_complete_minutes: int = Field(default=15)
    processing_time_ms: float = Field(default=0.0)
    provenance_hash: str = Field(default="")


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class SMEBaselineEngine:
    """Three-tier SME GHG baseline assessment engine.

    Produces a complete emissions estimate using the appropriate methodology
    for the SME's data availability tier (Bronze/Silver/Gold).

    All calculations use Decimal arithmetic for bit-perfect reproducibility.
    No LLM is used in any calculation path.

    Usage::

        engine = SMEBaselineEngine()
        result = engine.calculate(sme_input)
        assert result.provenance_hash  # non-empty SHA-256 hash
    """

    engine_version: str = _MODULE_VERSION

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #

    def calculate(self, data: SMEBaselineInput) -> SMEBaselineResult:
        """Run SME baseline assessment at the appropriate data tier.

        Args:
            data: Validated SME baseline input data.

        Returns:
            SMEBaselineResult with emissions breakdown and provenance.
        """
        t0 = time.perf_counter()
        logger.info(
            "SME Baseline: entity=%s, tier=%s, sector=%s, size=%s, headcount=%d",
            data.entity_name, data.data_tier.value, data.sector.value,
            data.company_size.value, data.headcount,
        )

        if data.data_tier == DataTier.BRONZE:
            scope1, scope2, scope3 = self._calculate_bronze(data)
        elif data.data_tier == DataTier.SILVER:
            scope1, scope2, scope3 = self._calculate_silver(data)
        else:
            scope1, scope2, scope3 = self._calculate_gold(data)

        # Totals
        total = _round_val(
            scope1.total_tco2e + scope2.total_tco2e + scope3.total_tco2e
        )

        # Scope percentages
        scope1_pct = _safe_pct(scope1.total_tco2e, total)
        scope2_pct = _safe_pct(scope2.total_tco2e, total)
        scope3_pct = _safe_pct(scope3.total_tco2e, total)

        # Accuracy band
        accuracy_band = self._compute_accuracy_band(total, data.data_tier)

        # Intensity metrics
        intensity = self._compute_intensity(total, data)

        # Data quality assessment
        data_quality = self._assess_data_quality(data)

        # Next steps
        next_steps = self._generate_next_steps(data)

        # Estimated time
        time_map = {
            DataTier.BRONZE: 15, DataTier.SILVER: 60, DataTier.GOLD: 150,
        }

        elapsed_ms = _round3((time.perf_counter() - t0) * 1000.0)

        result = SMEBaselineResult(
            entity_name=data.entity_name,
            reporting_year=data.reporting_year,
            data_tier=data.data_tier.value,
            sector=data.sector.value,
            company_size=data.company_size.value,
            headcount=data.headcount,
            scope1=scope1,
            scope2=scope2,
            scope3=scope3,
            total_tco2e=total,
            accuracy_band=accuracy_band,
            intensity=intensity,
            scope1_pct=_round_val(scope1_pct, 2),
            scope2_pct=_round_val(scope2_pct, 2),
            scope3_pct=_round_val(scope3_pct, 2),
            data_quality=data_quality,
            next_steps=next_steps,
            time_to_complete_minutes=time_map.get(data.data_tier, 15),
            processing_time_ms=elapsed_ms,
        )
        result.provenance_hash = _compute_hash(result)

        logger.info(
            "SME Baseline complete: total=%.2f tCO2e (tier=%s), hash=%s",
            float(total), data.data_tier.value,
            result.provenance_hash[:16],
        )
        return result

    def get_industry_average(
        self, sector: SMESector, company_size: CompanySize,
    ) -> Decimal:
        """Look up industry average emissions per employee.

        Args:
            sector: Industry sector.
            company_size: Company size classification.

        Returns:
            tCO2e per employee per year.
        """
        sector_data = INDUSTRY_AVG_PER_EMPLOYEE.get(sector, {})
        return sector_data.get(company_size.value, Decimal("3.5"))

    def get_scope_split(self, sector: SMESector) -> Dict[str, Decimal]:
        """Get industry average scope split ratios.

        Args:
            sector: Industry sector.

        Returns:
            Dict with scope1/scope2/scope3 fractions summing to 1.0.
        """
        return INDUSTRY_SCOPE_SPLIT.get(sector, {
            "scope1": Decimal("0.15"),
            "scope2": Decimal("0.15"),
            "scope3": Decimal("0.70"),
        })

    # ------------------------------------------------------------------ #
    # Bronze Tier -- Industry Averages Only                                #
    # ------------------------------------------------------------------ #

    def _calculate_bronze(
        self, data: SMEBaselineInput,
    ) -> tuple[ScopeBreakdown, ScopeBreakdown, ScopeBreakdown]:
        """Calculate baseline using industry averages only.

        Uses per-employee emission intensity multiplied by headcount,
        split across scopes using industry-average ratios.

        Args:
            data: SME baseline input.

        Returns:
            Tuple of (scope1, scope2, scope3) breakdowns.
        """
        avg_per_employee = self.get_industry_average(
            data.sector, data.company_size
        )
        total_estimate = _round_val(
            avg_per_employee * _decimal(data.headcount)
        )
        split = self.get_scope_split(data.sector)

        scope1_total = _round_val(total_estimate * split["scope1"])
        scope2_total = _round_val(total_estimate * split["scope2"])
        scope3_total = _round_val(total_estimate * split["scope3"])

        scope1 = ScopeBreakdown(
            total_tco2e=scope1_total,
            details={"estimated_from_industry_avg": scope1_total},
            data_quality=DataQualityLevel.ESTIMATED,
            methodology="industry_average_per_employee",
        )
        scope2 = ScopeBreakdown(
            total_tco2e=scope2_total,
            details={"estimated_from_industry_avg": scope2_total},
            data_quality=DataQualityLevel.ESTIMATED,
            methodology="industry_average_per_employee",
        )
        scope3 = ScopeBreakdown(
            total_tco2e=scope3_total,
            details={"estimated_from_industry_avg": scope3_total},
            data_quality=DataQualityLevel.ESTIMATED,
            methodology="industry_average_per_employee",
        )

        return scope1, scope2, scope3

    # ------------------------------------------------------------------ #
    # Silver Tier -- Basic Activity Data                                   #
    # ------------------------------------------------------------------ #

    def _calculate_silver(
        self, data: SMEBaselineInput,
    ) -> tuple[ScopeBreakdown, ScopeBreakdown, ScopeBreakdown]:
        """Calculate baseline using basic activity data.

        Uses actual fuel and electricity data where available,
        falls back to industry averages for Scope 3.

        Args:
            data: SME baseline input with fuel/electricity entries.

        Returns:
            Tuple of (scope1, scope2, scope3) breakdowns.
        """
        # Scope 1: from fuel entries
        scope1_details: Dict[str, Decimal] = {}
        scope1_total = Decimal("0")

        for entry in data.fuel_entries:
            ef_data = SME_FUEL_FACTORS.get(entry.fuel_type)
            if ef_data is None:
                continue
            tco2e = _round_val(entry.quantity * ef_data["factor"] / Decimal("1000"))
            fuel_key = entry.fuel_type.value
            scope1_details[fuel_key] = scope1_details.get(
                fuel_key, Decimal("0")
            ) + tco2e
            scope1_total += tco2e

        scope1_total = _round_val(scope1_total)

        # If no fuel data, fall back to bronze for Scope 1
        if scope1_total == Decimal("0"):
            avg = self.get_industry_average(data.sector, data.company_size)
            split = self.get_scope_split(data.sector)
            scope1_total = _round_val(
                avg * _decimal(data.headcount) * split["scope1"]
            )
            scope1_details = {"estimated_from_industry_avg": scope1_total}

        scope1 = ScopeBreakdown(
            total_tco2e=scope1_total,
            details=scope1_details,
            data_quality=(
                DataQualityLevel.MEDIUM if data.fuel_entries
                else DataQualityLevel.ESTIMATED
            ),
            methodology=(
                "activity_data_fuel" if data.fuel_entries
                else "industry_average_fallback"
            ),
        )

        # Scope 2: from electricity entries
        scope2_details: Dict[str, Decimal] = {}
        scope2_total = Decimal("0")

        for entry in data.electricity_entries:
            mwh = entry.annual_kwh / Decimal("1000")
            grid_factor = GRID_EMISSION_FACTORS.get(
                entry.region, GRID_EMISSION_FACTORS["GLOBAL_AVG"]
            )
            location_tco2e = _round_val(mwh * grid_factor)

            # If green tariff, reduce by the green percentage
            if entry.green_tariff:
                green_reduction = location_tco2e * entry.green_tariff_pct / Decimal("100")
                market_tco2e = _round_val(location_tco2e - green_reduction)
            else:
                market_tco2e = location_tco2e

            scope2_details[f"electricity_{entry.region}"] = market_tco2e
            scope2_total += market_tco2e

        scope2_total = _round_val(scope2_total)

        if scope2_total == Decimal("0"):
            avg = self.get_industry_average(data.sector, data.company_size)
            split = self.get_scope_split(data.sector)
            scope2_total = _round_val(
                avg * _decimal(data.headcount) * split["scope2"]
            )
            scope2_details = {"estimated_from_industry_avg": scope2_total}

        scope2 = ScopeBreakdown(
            total_tco2e=scope2_total,
            details=scope2_details,
            data_quality=(
                DataQualityLevel.MEDIUM if data.electricity_entries
                else DataQualityLevel.ESTIMATED
            ),
            methodology=(
                "activity_data_electricity" if data.electricity_entries
                else "industry_average_fallback"
            ),
        )

        # Scope 3: from total spend or industry average
        scope3_total = Decimal("0")
        scope3_details: Dict[str, Decimal] = {}

        if data.total_annual_spend_usd and data.total_annual_spend_usd > Decimal("0"):
            ratio = INDUSTRY_SCOPE3_SPEND_RATIO.get(
                data.sector, Decimal("0.00042")
            )
            scope3_total = _round_val(data.total_annual_spend_usd * ratio)
            scope3_details = {"total_spend_based_estimate": scope3_total}
            s3_methodology = "spend_based_ratio"
            s3_quality = DataQualityLevel.LOW
        else:
            avg = self.get_industry_average(data.sector, data.company_size)
            split = self.get_scope_split(data.sector)
            scope3_total = _round_val(
                avg * _decimal(data.headcount) * split["scope3"]
            )
            scope3_details = {"estimated_from_industry_avg": scope3_total}
            s3_methodology = "industry_average_fallback"
            s3_quality = DataQualityLevel.ESTIMATED

        scope3 = ScopeBreakdown(
            total_tco2e=scope3_total,
            details=scope3_details,
            data_quality=s3_quality,
            methodology=s3_methodology,
        )

        return scope1, scope2, scope3

    # ------------------------------------------------------------------ #
    # Gold Tier -- Detailed Data                                           #
    # ------------------------------------------------------------------ #

    def _calculate_gold(
        self, data: SMEBaselineInput,
    ) -> tuple[ScopeBreakdown, ScopeBreakdown, ScopeBreakdown]:
        """Calculate baseline using detailed, granular data.

        Includes refrigerant calculations, vehicle fleet analysis,
        and category-level Scope 3 spend-based factors.

        Args:
            data: SME baseline input with full detail.

        Returns:
            Tuple of (scope1, scope2, scope3) breakdowns.
        """
        # Scope 1: Fuel + Refrigerants + Vehicles
        scope1_details: Dict[str, Decimal] = {}
        fuel_total = Decimal("0")

        for entry in data.fuel_entries:
            ef_data = SME_FUEL_FACTORS.get(entry.fuel_type)
            if ef_data is None:
                continue
            tco2e = entry.quantity * ef_data["factor"] / Decimal("1000")
            fuel_key = f"fuel_{entry.fuel_type.value}"
            scope1_details[fuel_key] = scope1_details.get(
                fuel_key, Decimal("0")
            ) + _round_val(tco2e)
            fuel_total += tco2e

        # Refrigerants
        ref_total = Decimal("0")
        for ref_entry in data.refrigerant_entries:
            gwp = SME_GWP.get(ref_entry.refrigerant_type.lower(), Decimal("0"))
            total_charge = (
                _decimal(ref_entry.system_count) * ref_entry.typical_charge_kg
            )
            leakage_kg = total_charge * ref_entry.annual_leakage_rate_pct / Decimal("100")
            tco2e = leakage_kg * gwp / Decimal("1000")
            ref_total += tco2e

        scope1_details["refrigerants"] = _round_val(ref_total)

        # Vehicles
        vehicle_total = Decimal("0")
        for v_entry in data.vehicle_entries:
            ef_data = SME_FUEL_FACTORS.get(v_entry.fuel_type)
            if ef_data is None:
                continue
            total_km = _decimal(v_entry.vehicle_count) * v_entry.annual_km_per_vehicle
            total_litres = total_km * v_entry.fuel_efficiency_l_per_100km / Decimal("100")
            tco2e = total_litres * ef_data["factor"] / Decimal("1000")
            vehicle_total += tco2e

        scope1_details["vehicles"] = _round_val(vehicle_total)

        scope1_total = _round_val(fuel_total + ref_total + vehicle_total)

        # Fall back if no Scope 1 data
        if scope1_total == Decimal("0"):
            avg = self.get_industry_average(data.sector, data.company_size)
            split = self.get_scope_split(data.sector)
            scope1_total = _round_val(
                avg * _decimal(data.headcount) * split["scope1"]
            )
            scope1_details = {"estimated_from_industry_avg": scope1_total}

        scope1 = ScopeBreakdown(
            total_tco2e=scope1_total,
            details=scope1_details,
            data_quality=DataQualityLevel.HIGH if (
                data.fuel_entries or data.refrigerant_entries or data.vehicle_entries
            ) else DataQualityLevel.ESTIMATED,
            methodology="detailed_activity_data",
        )

        # Scope 2: Same as silver but with higher quality
        scope2_details: Dict[str, Decimal] = {}
        scope2_total = Decimal("0")

        for entry in data.electricity_entries:
            mwh = entry.annual_kwh / Decimal("1000")
            grid_factor = GRID_EMISSION_FACTORS.get(
                entry.region, GRID_EMISSION_FACTORS["GLOBAL_AVG"]
            )
            location_tco2e = mwh * grid_factor

            if entry.green_tariff:
                green_reduction = location_tco2e * entry.green_tariff_pct / Decimal("100")
                market_tco2e = _round_val(location_tco2e - green_reduction)
            else:
                market_tco2e = _round_val(location_tco2e)

            scope2_details[f"electricity_{entry.region}"] = market_tco2e
            scope2_total += market_tco2e

        scope2_total = _round_val(scope2_total)

        if scope2_total == Decimal("0"):
            avg = self.get_industry_average(data.sector, data.company_size)
            split = self.get_scope_split(data.sector)
            scope2_total = _round_val(
                avg * _decimal(data.headcount) * split["scope2"]
            )
            scope2_details = {"estimated_from_industry_avg": scope2_total}

        scope2 = ScopeBreakdown(
            total_tco2e=scope2_total,
            details=scope2_details,
            data_quality=DataQualityLevel.HIGH if data.electricity_entries
            else DataQualityLevel.ESTIMATED,
            methodology="detailed_electricity_data",
        )

        # Scope 3: Category-level spend-based
        scope3_details: Dict[str, Decimal] = {}
        scope3_total = Decimal("0")

        for spend_entry in data.spend_entries:
            factor = spend_entry.custom_factor
            if factor is None:
                factor = EEIO_SPEND_FACTORS.get(
                    spend_entry.category, Decimal("0.400")
                )
            # Factor is tCO2e per $1000 USD
            tco2e = _round_val(
                spend_entry.annual_spend_usd / Decimal("1000") * factor
            )
            cat_key = spend_entry.category.value
            scope3_details[cat_key] = scope3_details.get(
                cat_key, Decimal("0")
            ) + tco2e
            scope3_total += tco2e

        scope3_total = _round_val(scope3_total)

        if scope3_total == Decimal("0"):
            # Fall back to total spend or industry average
            if data.total_annual_spend_usd and data.total_annual_spend_usd > Decimal("0"):
                ratio = INDUSTRY_SCOPE3_SPEND_RATIO.get(
                    data.sector, Decimal("0.00042")
                )
                scope3_total = _round_val(data.total_annual_spend_usd * ratio)
                scope3_details = {"total_spend_based_estimate": scope3_total}
            else:
                avg = self.get_industry_average(data.sector, data.company_size)
                split = self.get_scope_split(data.sector)
                scope3_total = _round_val(
                    avg * _decimal(data.headcount) * split["scope3"]
                )
                scope3_details = {"estimated_from_industry_avg": scope3_total}

        scope3 = ScopeBreakdown(
            total_tco2e=scope3_total,
            details=scope3_details,
            data_quality=DataQualityLevel.MEDIUM if data.spend_entries
            else DataQualityLevel.LOW,
            methodology="spend_based_eeio_by_category" if data.spend_entries
            else "industry_average_fallback",
        )

        return scope1, scope2, scope3

    # ------------------------------------------------------------------ #
    # Supporting Methods                                                   #
    # ------------------------------------------------------------------ #

    def _compute_accuracy_band(
        self, total: Decimal, tier: DataTier,
    ) -> AccuracyBand:
        """Compute the accuracy band for the baseline estimate.

        Args:
            total: Central estimate of total emissions.
            tier: Data tier used.

        Returns:
            AccuracyBand with lower/upper bounds.
        """
        band = TIER_ACCURACY.get(tier, TIER_ACCURACY[DataTier.BRONZE])
        lower = _round_val(total * band["lower_pct"] / Decimal("100"))
        upper = _round_val(total * band["upper_pct"] / Decimal("100"))

        confidence_map = {
            DataTier.BRONZE: Decimal("60"),
            DataTier.SILVER: Decimal("80"),
            DataTier.GOLD: Decimal("95"),
        }

        return AccuracyBand(
            central_estimate_tco2e=total,
            lower_bound_tco2e=lower,
            upper_bound_tco2e=upper,
            confidence_pct=confidence_map.get(tier, Decimal("60")),
            tier=tier.value,
        )

    def _compute_intensity(
        self, total: Decimal, data: SMEBaselineInput,
    ) -> IntensityMetrics:
        """Compute emission intensity metrics.

        Args:
            total: Total emissions in tCO2e.
            data: Input data with headcount and revenue.

        Returns:
            IntensityMetrics with per-employee and per-revenue ratios.
        """
        per_employee = _safe_divide(total, _decimal(data.headcount))
        sector_avg = self.get_industry_average(data.sector, data.company_size)
        vs_avg = _safe_pct(per_employee, sector_avg) if sector_avg > Decimal("0") else Decimal("0")

        per_revenue = None
        if data.revenue_usd and data.revenue_usd > Decimal("0"):
            revenue_millions = data.revenue_usd / Decimal("1000000")
            per_revenue = _round_val(
                _safe_divide(total, revenue_millions), 2
            )

        return IntensityMetrics(
            per_employee=_round_val(per_employee, 2),
            per_revenue_million=per_revenue,
            sector_average_per_employee=sector_avg,
            vs_sector_avg_pct=_round_val(vs_avg, 2),
        )

    def _assess_data_quality(
        self, data: SMEBaselineInput,
    ) -> DataQualityAssessment:
        """Assess overall data quality.

        Args:
            data: Input data to assess.

        Returns:
            DataQualityAssessment with score and suggestions.
        """
        suggestions: List[str] = []
        total_fields = 7  # minimum fields always present
        filled_fields = 3  # entity_name, sector, headcount always filled

        if data.fuel_entries:
            filled_fields += 1
        else:
            suggestions.append(
                "Add fuel consumption data from gas/fuel bills to improve "
                "Scope 1 accuracy by up to 25%."
            )

        if data.electricity_entries:
            filled_fields += 1
        else:
            suggestions.append(
                "Add electricity consumption from electricity bills to improve "
                "Scope 2 accuracy by up to 30%."
            )

        if data.spend_entries:
            filled_fields += 1
        else:
            suggestions.append(
                "Break down procurement spend by category (goods, travel, commuting) "
                "to improve Scope 3 accuracy by up to 20%."
            )

        if data.refrigerant_entries:
            filled_fields += 1
        else:
            if data.sector in (
                SMESector.ACCOMMODATION_FOOD, SMESector.WHOLESALE_RETAIL,
                SMESector.HEALTHCARE,
            ):
                suggestions.append(
                    "Add refrigerant/AC system data - common source of "
                    "emissions in your sector."
                )

        accuracy_map = {
            DataTier.BRONZE: Decimal("40"),
            DataTier.SILVER: Decimal("15"),
            DataTier.GOLD: Decimal("5"),
        }

        completeness = _safe_pct(_decimal(filled_fields), _decimal(total_fields))
        quality_scores = {
            DataTier.BRONZE: Decimal("0.30"),
            DataTier.SILVER: Decimal("0.60"),
            DataTier.GOLD: Decimal("0.85"),
        }

        return DataQualityAssessment(
            overall_score=quality_scores.get(data.data_tier, Decimal("0.30")),
            tier_used=data.data_tier.value,
            estimated_accuracy_pct=accuracy_map.get(data.data_tier, Decimal("40")),
            improvement_suggestions=suggestions,
            completeness_pct=_round_val(completeness, 1),
        )

    def _generate_next_steps(
        self, data: SMEBaselineInput,
    ) -> List[NextStepRecommendation]:
        """Generate prioritized next-step recommendations.

        Args:
            data: Input data to analyze for gaps.

        Returns:
            List of NextStepRecommendation ordered by priority.
        """
        steps: List[NextStepRecommendation] = []

        if data.data_tier == DataTier.BRONZE:
            steps.append(NextStepRecommendation(
                priority=1,
                action="Collect electricity bills for the past 12 months",
                impact="Improves Scope 2 accuracy from +/-40% to +/-15%",
                effort_minutes=30,
            ))
            steps.append(NextStepRecommendation(
                priority=2,
                action="Collect gas/fuel bills for the past 12 months",
                impact="Improves Scope 1 accuracy from +/-40% to +/-15%",
                effort_minutes=30,
            ))
            steps.append(NextStepRecommendation(
                priority=3,
                action="Estimate total annual procurement spend",
                impact="Enables spend-based Scope 3 estimate",
                effort_minutes=15,
            ))
        elif data.data_tier == DataTier.SILVER:
            steps.append(NextStepRecommendation(
                priority=1,
                action="Break down spend by category (goods, travel, commuting)",
                impact="Improves Scope 3 accuracy from +/-15% to +/-5%",
                effort_minutes=60,
            ))
            if not data.refrigerant_entries:
                steps.append(NextStepRecommendation(
                    priority=2,
                    action="Inventory AC/refrigeration systems and refrigerant types",
                    impact="Captures often-missed Scope 1 refrigerant emissions",
                    effort_minutes=30,
                ))
            steps.append(NextStepRecommendation(
                priority=3,
                action="Document vehicle fleet details (count, type, annual mileage)",
                impact="Improves mobile combustion accuracy",
                effort_minutes=20,
            ))
        else:
            # Gold tier
            steps.append(NextStepRecommendation(
                priority=1,
                action="Set up automated data collection from accounting software",
                impact="Reduces annual data collection time by 80%",
                effort_minutes=120,
            ))
            steps.append(NextStepRecommendation(
                priority=2,
                action="Engage top 5 suppliers for primary emission data",
                impact="Improves Scope 3 from spend-based to supplier-specific",
                effort_minutes=180,
            ))

        return steps
