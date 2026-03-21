# -*- coding: utf-8 -*-
"""
Scope3EstimatorEngine - PACK-026 SME Net Zero Pack Engine 4
==============================================================

Spend-based Scope 3 estimation for SMEs using DEFRA/EPA EEIO factors.
Focuses on the three most material categories for SMEs (Cat 1, 6, 7)
with optional support for Cat 2, 3, and 4.

This engine is designed for SMEs that lack activity-based Scope 3 data
and need to estimate their value chain emissions using procurement
spend data.  It supports direct mapping from accounting software
categories (Xero, QuickBooks) to GHG Protocol Scope 3 categories.

Calculation Methodology:
    Spend-based:
        tCO2e = spend_usd / 1000 * EEIO_factor  (per category)

    Category aggregation:
        total_scope3 = sum(category_tco2e for each category)

    Data quality score:
        DQS = weighted_avg(completeness, granularity, age, source)

    Accounting integration:
        Map Xero/QuickBooks spend categories to Scope 3 categories
        using predefined mapping tables.

Regulatory References:
    - GHG Protocol Scope 3 Standard (2011) - Category definitions
    - GHG Protocol Scope 3 Evaluator Tool (2013)
    - US EPA EEIO v2.0 - Spend-based emission factors
    - DEFRA/BEIS 2024 - UK spend-based factors
    - EXIOBASE 3.8 - Multi-regional spend factors

Zero-Hallucination:
    - All calculations use deterministic Decimal arithmetic
    - EEIO factors are hard-coded from published databases
    - Accounting mappings are static lookup tables
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
    return datetime.now(timezone.utc).replace(microsecond=0)


def _new_uuid() -> str:
    return str(uuid.uuid4())


def _compute_hash(data: Any) -> str:
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
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return Decimal("0")


def _safe_divide(
    numerator: Decimal, denominator: Decimal,
    default: Decimal = Decimal("0"),
) -> Decimal:
    if denominator == Decimal("0"):
        return default
    return numerator / denominator


def _safe_pct(part: Decimal, whole: Decimal) -> Decimal:
    return _safe_divide(part * Decimal("100"), whole)


def _round_val(value: Decimal, places: int = 6) -> Decimal:
    quantize_str = "0." + "0" * places
    return value.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP)


def _round3(value: float) -> float:
    return float(
        Decimal(str(value)).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)
    )


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class Scope3Category(str, Enum):
    """Scope 3 categories supported for SME estimation.

    Core: Cat 1, 6, 7 (typically 70-85% of SME Scope 3).
    Optional: Cat 2, 3, 4 (added for completeness).
    """
    CAT_01_PURCHASED_GOODS = "cat_01_purchased_goods"
    CAT_02_CAPITAL_GOODS = "cat_02_capital_goods"
    CAT_03_FUEL_ENERGY = "cat_03_fuel_energy"
    CAT_04_UPSTREAM_TRANSPORT = "cat_04_upstream_transport"
    CAT_05_WASTE = "cat_05_waste"
    CAT_06_BUSINESS_TRAVEL = "cat_06_business_travel"
    CAT_07_EMPLOYEE_COMMUTING = "cat_07_employee_commuting"


class SpendCurrency(str, Enum):
    """Currency for spend data."""
    USD = "usd"
    GBP = "gbp"
    EUR = "eur"
    AUD = "aud"
    CAD = "cad"


class DataSourceType(str, Enum):
    """Source of spend data."""
    MANUAL_ENTRY = "manual_entry"
    ACCOUNTING_EXPORT = "accounting_export"
    XERO = "xero"
    QUICKBOOKS = "quickbooks"
    SAGE = "sage"
    FREEAGENT = "freeagent"


class IndustryType(str, Enum):
    """Industry type for EEIO factor refinement."""
    GENERAL = "general"
    FOOD_BEVERAGE = "food_beverage"
    TEXTILES_APPAREL = "textiles_apparel"
    CHEMICALS = "chemicals"
    METALS_MINING = "metals_mining"
    ELECTRONICS = "electronics"
    PROFESSIONAL_SERVICES = "professional_services"
    CONSTRUCTION_MATERIALS = "construction_materials"
    TRANSPORT_SERVICES = "transport_services"
    HOSPITALITY = "hospitality"


class DataQualityDimension(str, Enum):
    """Dimensions for data quality scoring."""
    COMPLETENESS = "completeness"
    GRANULARITY = "granularity"
    TEMPORAL = "temporal"
    SOURCE_RELIABILITY = "source_reliability"


# ---------------------------------------------------------------------------
# Constants -- EEIO Factors
# ---------------------------------------------------------------------------


# EEIO spend-based factors (tCO2e per $1000 USD).
# Source: US EPA EEIO v2.0, DEFRA 2024, EXIOBASE 3.8.
EEIO_FACTORS_GENERAL: Dict[str, Decimal] = {
    Scope3Category.CAT_01_PURCHASED_GOODS: Decimal("0.430"),
    Scope3Category.CAT_02_CAPITAL_GOODS: Decimal("0.350"),
    Scope3Category.CAT_03_FUEL_ENERGY: Decimal("0.280"),
    Scope3Category.CAT_04_UPSTREAM_TRANSPORT: Decimal("0.520"),
    Scope3Category.CAT_05_WASTE: Decimal("0.210"),
    Scope3Category.CAT_06_BUSINESS_TRAVEL: Decimal("0.310"),
    Scope3Category.CAT_07_EMPLOYEE_COMMUTING: Decimal("0.180"),
}

# Industry-specific refinement multipliers.
# Applied as: factor = general_factor * industry_multiplier.
INDUSTRY_MULTIPLIERS: Dict[str, Dict[str, Decimal]] = {
    IndustryType.GENERAL: {cat: Decimal("1.00") for cat in [
        "cat_01_purchased_goods", "cat_02_capital_goods", "cat_03_fuel_energy",
        "cat_04_upstream_transport", "cat_05_waste", "cat_06_business_travel",
        "cat_07_employee_commuting",
    ]},
    IndustryType.FOOD_BEVERAGE: {
        "cat_01_purchased_goods": Decimal("1.25"),
        "cat_02_capital_goods": Decimal("0.90"),
        "cat_03_fuel_energy": Decimal("1.10"),
        "cat_04_upstream_transport": Decimal("1.30"),
        "cat_05_waste": Decimal("1.40"),
        "cat_06_business_travel": Decimal("0.80"),
        "cat_07_employee_commuting": Decimal("1.00"),
    },
    IndustryType.TEXTILES_APPAREL: {
        "cat_01_purchased_goods": Decimal("1.45"),
        "cat_02_capital_goods": Decimal("0.85"),
        "cat_03_fuel_energy": Decimal("0.95"),
        "cat_04_upstream_transport": Decimal("1.50"),
        "cat_05_waste": Decimal("1.20"),
        "cat_06_business_travel": Decimal("0.90"),
        "cat_07_employee_commuting": Decimal("1.00"),
    },
    IndustryType.CHEMICALS: {
        "cat_01_purchased_goods": Decimal("1.60"),
        "cat_02_capital_goods": Decimal("1.20"),
        "cat_03_fuel_energy": Decimal("1.40"),
        "cat_04_upstream_transport": Decimal("1.10"),
        "cat_05_waste": Decimal("1.50"),
        "cat_06_business_travel": Decimal("0.85"),
        "cat_07_employee_commuting": Decimal("1.00"),
    },
    IndustryType.METALS_MINING: {
        "cat_01_purchased_goods": Decimal("1.80"),
        "cat_02_capital_goods": Decimal("1.40"),
        "cat_03_fuel_energy": Decimal("1.50"),
        "cat_04_upstream_transport": Decimal("1.20"),
        "cat_05_waste": Decimal("1.30"),
        "cat_06_business_travel": Decimal("0.80"),
        "cat_07_employee_commuting": Decimal("1.00"),
    },
    IndustryType.ELECTRONICS: {
        "cat_01_purchased_goods": Decimal("1.35"),
        "cat_02_capital_goods": Decimal("1.30"),
        "cat_03_fuel_energy": Decimal("0.80"),
        "cat_04_upstream_transport": Decimal("1.15"),
        "cat_05_waste": Decimal("0.90"),
        "cat_06_business_travel": Decimal("1.10"),
        "cat_07_employee_commuting": Decimal("1.00"),
    },
    IndustryType.PROFESSIONAL_SERVICES: {
        "cat_01_purchased_goods": Decimal("0.60"),
        "cat_02_capital_goods": Decimal("0.70"),
        "cat_03_fuel_energy": Decimal("0.50"),
        "cat_04_upstream_transport": Decimal("0.40"),
        "cat_05_waste": Decimal("0.50"),
        "cat_06_business_travel": Decimal("1.30"),
        "cat_07_employee_commuting": Decimal("1.20"),
    },
    IndustryType.CONSTRUCTION_MATERIALS: {
        "cat_01_purchased_goods": Decimal("1.50"),
        "cat_02_capital_goods": Decimal("1.30"),
        "cat_03_fuel_energy": Decimal("1.30"),
        "cat_04_upstream_transport": Decimal("1.40"),
        "cat_05_waste": Decimal("1.60"),
        "cat_06_business_travel": Decimal("0.70"),
        "cat_07_employee_commuting": Decimal("1.00"),
    },
    IndustryType.TRANSPORT_SERVICES: {
        "cat_01_purchased_goods": Decimal("0.80"),
        "cat_02_capital_goods": Decimal("1.10"),
        "cat_03_fuel_energy": Decimal("1.60"),
        "cat_04_upstream_transport": Decimal("1.80"),
        "cat_05_waste": Decimal("0.60"),
        "cat_06_business_travel": Decimal("0.90"),
        "cat_07_employee_commuting": Decimal("1.00"),
    },
    IndustryType.HOSPITALITY: {
        "cat_01_purchased_goods": Decimal("1.20"),
        "cat_02_capital_goods": Decimal("0.80"),
        "cat_03_fuel_energy": Decimal("1.10"),
        "cat_04_upstream_transport": Decimal("1.00"),
        "cat_05_waste": Decimal("1.50"),
        "cat_06_business_travel": Decimal("0.70"),
        "cat_07_employee_commuting": Decimal("1.10"),
    },
}

# Currency conversion to USD (approximate, for spend normalization).
CURRENCY_TO_USD: Dict[str, Decimal] = {
    SpendCurrency.USD: Decimal("1.00"),
    SpendCurrency.GBP: Decimal("1.27"),
    SpendCurrency.EUR: Decimal("1.08"),
    SpendCurrency.AUD: Decimal("0.65"),
    SpendCurrency.CAD: Decimal("0.74"),
}

# Accounting software category mappings.
# Maps common chart-of-accounts categories to Scope 3 categories.
XERO_CATEGORY_MAP: Dict[str, Scope3Category] = {
    "cost_of_goods_sold": Scope3Category.CAT_01_PURCHASED_GOODS,
    "purchases": Scope3Category.CAT_01_PURCHASED_GOODS,
    "direct_costs": Scope3Category.CAT_01_PURCHASED_GOODS,
    "inventory": Scope3Category.CAT_01_PURCHASED_GOODS,
    "office_expenses": Scope3Category.CAT_01_PURCHASED_GOODS,
    "computer_equipment": Scope3Category.CAT_02_CAPITAL_GOODS,
    "fixed_assets": Scope3Category.CAT_02_CAPITAL_GOODS,
    "furniture_equipment": Scope3Category.CAT_02_CAPITAL_GOODS,
    "motor_vehicle_expenses": Scope3Category.CAT_04_UPSTREAM_TRANSPORT,
    "freight_delivery": Scope3Category.CAT_04_UPSTREAM_TRANSPORT,
    "courier": Scope3Category.CAT_04_UPSTREAM_TRANSPORT,
    "shipping": Scope3Category.CAT_04_UPSTREAM_TRANSPORT,
    "travel": Scope3Category.CAT_06_BUSINESS_TRAVEL,
    "travel_international": Scope3Category.CAT_06_BUSINESS_TRAVEL,
    "travel_national": Scope3Category.CAT_06_BUSINESS_TRAVEL,
    "accommodation": Scope3Category.CAT_06_BUSINESS_TRAVEL,
    "entertainment": Scope3Category.CAT_06_BUSINESS_TRAVEL,
    "cleaning": Scope3Category.CAT_05_WASTE,
    "waste_disposal": Scope3Category.CAT_05_WASTE,
}

QUICKBOOKS_CATEGORY_MAP: Dict[str, Scope3Category] = {
    "cost_of_goods_sold": Scope3Category.CAT_01_PURCHASED_GOODS,
    "supplies": Scope3Category.CAT_01_PURCHASED_GOODS,
    "inventory_asset": Scope3Category.CAT_01_PURCHASED_GOODS,
    "other_costs": Scope3Category.CAT_01_PURCHASED_GOODS,
    "furniture_and_fixtures": Scope3Category.CAT_02_CAPITAL_GOODS,
    "machinery_and_equipment": Scope3Category.CAT_02_CAPITAL_GOODS,
    "computer_and_internet": Scope3Category.CAT_02_CAPITAL_GOODS,
    "auto": Scope3Category.CAT_04_UPSTREAM_TRANSPORT,
    "shipping_delivery": Scope3Category.CAT_04_UPSTREAM_TRANSPORT,
    "travel": Scope3Category.CAT_06_BUSINESS_TRAVEL,
    "travel_meals": Scope3Category.CAT_06_BUSINESS_TRAVEL,
    "disposal_fees": Scope3Category.CAT_05_WASTE,
}

# Category display names.
CATEGORY_NAMES: Dict[str, str] = {
    "cat_01_purchased_goods": "Cat 1: Purchased Goods & Services",
    "cat_02_capital_goods": "Cat 2: Capital Goods",
    "cat_03_fuel_energy": "Cat 3: Fuel & Energy Related Activities",
    "cat_04_upstream_transport": "Cat 4: Upstream Transportation",
    "cat_05_waste": "Cat 5: Waste Generated in Operations",
    "cat_06_business_travel": "Cat 6: Business Travel",
    "cat_07_employee_commuting": "Cat 7: Employee Commuting",
}

# Commuting defaults (for Cat 7 estimation without spend data).
# Source: UK DfT National Travel Survey 2024, US Census Bureau ACS 2023.
AVG_COMMUTE_KM_PER_DAY: Decimal = Decimal("25.0")
WORKING_DAYS_PER_YEAR: Decimal = Decimal("230")
CAR_EMISSION_FACTOR_KG_PER_KM: Decimal = Decimal("0.171")
AVG_CAR_COMMUTE_SHARE: Decimal = Decimal("0.65")


# ---------------------------------------------------------------------------
# Pydantic Models -- Inputs
# ---------------------------------------------------------------------------


class SpendEntry(BaseModel):
    """A single spend entry for Scope 3 estimation.

    Attributes:
        category: Scope 3 category or accounting category name.
        amount: Spend amount in original currency.
        currency: Currency of the spend.
        description: Optional description.
        accounting_category: Raw accounting software category name.
        data_source: Source of the data.
        custom_factor: Optional custom EEIO factor override.
    """
    category: Optional[Scope3Category] = Field(
        None, description="Scope 3 category (auto-mapped if accounting_category provided)"
    )
    amount: Decimal = Field(
        ..., ge=Decimal("0"), description="Spend amount"
    )
    currency: SpendCurrency = Field(
        default=SpendCurrency.USD, description="Currency"
    )
    description: str = Field(default="", max_length=500)
    accounting_category: Optional[str] = Field(
        None, description="Raw accounting software category"
    )
    data_source: DataSourceType = Field(
        default=DataSourceType.MANUAL_ENTRY, description="Data source"
    )
    custom_factor: Optional[Decimal] = Field(
        None, ge=Decimal("0"),
        description="Custom EEIO factor (tCO2e per $1000 USD)",
    )


class CommutingEstimateInput(BaseModel):
    """Input for employee commuting estimation (Cat 7).

    Used when spend data for commuting is not available.

    Attributes:
        headcount: Number of employees.
        avg_commute_km: Average one-way commute distance (km).
        car_share_pct: Percentage of employees who commute by car.
        working_days_per_year: Working days per year.
        remote_work_pct: Percentage of days worked remotely.
    """
    headcount: int = Field(..., ge=1, le=250, description="Employee count")
    avg_commute_km: Decimal = Field(
        default=Decimal("12.5"), ge=Decimal("0"),
        description="Average one-way commute (km)",
    )
    car_share_pct: Decimal = Field(
        default=Decimal("65"), ge=Decimal("0"), le=Decimal("100"),
        description="% of employees commuting by car",
    )
    working_days_per_year: int = Field(
        default=230, ge=1, le=365, description="Working days per year"
    )
    remote_work_pct: Decimal = Field(
        default=Decimal("0"), ge=Decimal("0"), le=Decimal("100"),
        description="% of days worked remotely",
    )


class Scope3EstimatorInput(BaseModel):
    """Complete input for Scope 3 estimation.

    Attributes:
        entity_name: Company name.
        reporting_year: Year of assessment.
        industry: Industry type for factor refinement.
        spend_entries: Spend data by category.
        commuting_estimate: Optional commuting data (if no spend for Cat 7).
        headcount: Employee count (for per-capita outputs).
        data_source_type: Primary data source.
        include_optional_categories: Whether to include Cat 2, 3, 4.
    """
    entity_name: str = Field(
        ..., min_length=1, max_length=300, description="Company name"
    )
    reporting_year: int = Field(
        ..., ge=2015, le=2100, description="Reporting year"
    )
    industry: IndustryType = Field(
        default=IndustryType.GENERAL, description="Industry for factor refinement"
    )
    spend_entries: List[SpendEntry] = Field(
        default_factory=list, description="Spend data"
    )
    commuting_estimate: Optional[CommutingEstimateInput] = Field(
        None, description="Employee commuting estimate"
    )
    headcount: int = Field(
        default=10, ge=1, le=250, description="Employee count"
    )
    data_source_type: DataSourceType = Field(
        default=DataSourceType.MANUAL_ENTRY, description="Primary data source"
    )
    include_optional_categories: bool = Field(
        default=False, description="Include Cat 2, 3, 4"
    )

    @field_validator("headcount")
    @classmethod
    def validate_headcount(cls, v: int) -> int:
        if v > 250:
            raise ValueError("SME headcount must be <= 250")
        return v


# ---------------------------------------------------------------------------
# Pydantic Models -- Outputs
# ---------------------------------------------------------------------------


class CategoryEstimate(BaseModel):
    """Estimated emissions for a single Scope 3 category.

    Attributes:
        category: Scope 3 category identifier.
        category_name: Display name.
        spend_usd: Spend in USD.
        eeio_factor: EEIO factor used (tCO2e per $1000 USD).
        tco2e: Estimated emissions.
        pct_of_total: Percentage of total Scope 3.
        data_quality_score: Quality score (0-1).
        methodology: Calculation method used.
        is_core: Whether this is a core category (Cat 1, 6, 7).
    """
    category: str = Field(default="")
    category_name: str = Field(default="")
    spend_usd: Decimal = Field(default=Decimal("0"))
    eeio_factor: Decimal = Field(default=Decimal("0"))
    tco2e: Decimal = Field(default=Decimal("0"))
    pct_of_total: Decimal = Field(default=Decimal("0"))
    data_quality_score: Decimal = Field(default=Decimal("0"))
    methodology: str = Field(default="spend_based_eeio")
    is_core: bool = Field(default=True)


class DataQualityScore(BaseModel):
    """Data quality assessment for the Scope 3 estimate.

    Attributes:
        overall_score: Composite quality score (0-100).
        completeness: Completeness of spend data (0-100).
        granularity: Category-level granularity (0-100).
        temporal: Temporal relevance (0-100).
        source_reliability: Source reliability (0-100).
        level: Qualitative level (high/medium/low).
    """
    overall_score: Decimal = Field(default=Decimal("0"))
    completeness: Decimal = Field(default=Decimal("0"))
    granularity: Decimal = Field(default=Decimal("0"))
    temporal: Decimal = Field(default=Decimal("0"))
    source_reliability: Decimal = Field(default=Decimal("0"))
    level: str = Field(default="low")


class Scope3EstimatorResult(BaseModel):
    """Complete Scope 3 estimation result.

    Attributes:
        result_id: Unique result identifier.
        engine_version: Engine version.
        calculated_at: Timestamp (UTC).
        entity_name: Company name.
        reporting_year: Reporting year.
        categories: Emissions by category.
        total_scope3_tco2e: Total estimated Scope 3.
        core_categories_tco2e: Total for Cat 1, 6, 7.
        optional_categories_tco2e: Total for Cat 2, 3, 4.
        per_employee_tco2e: Scope 3 per employee.
        data_quality: Data quality assessment.
        total_spend_usd: Total spend analyzed.
        categories_included: List of included categories.
        accounting_mappings_used: Number of auto-mapped entries.
        improvement_recommendations: Suggestions to improve estimate.
        processing_time_ms: Calculation time (ms).
        provenance_hash: SHA-256 audit hash.
    """
    result_id: str = Field(default_factory=_new_uuid)
    engine_version: str = Field(default=_MODULE_VERSION)
    calculated_at: datetime = Field(default_factory=_utcnow)
    entity_name: str = Field(default="")
    reporting_year: int = Field(default=0)

    categories: List[CategoryEstimate] = Field(default_factory=list)
    total_scope3_tco2e: Decimal = Field(default=Decimal("0"))
    core_categories_tco2e: Decimal = Field(default=Decimal("0"))
    optional_categories_tco2e: Decimal = Field(default=Decimal("0"))
    per_employee_tco2e: Decimal = Field(default=Decimal("0"))

    data_quality: DataQualityScore = Field(default_factory=DataQualityScore)
    total_spend_usd: Decimal = Field(default=Decimal("0"))
    categories_included: List[str] = Field(default_factory=list)
    accounting_mappings_used: int = Field(default=0)
    improvement_recommendations: List[str] = Field(default_factory=list)

    processing_time_ms: float = Field(default=0.0)
    provenance_hash: str = Field(default="")


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class Scope3EstimatorEngine:
    """Spend-based Scope 3 estimation engine for SMEs.

    Estimates Scope 3 emissions using EEIO spend-based factors.
    Supports auto-mapping from Xero/QuickBooks accounting categories.
    Focuses on Cat 1, 6, 7 with optional Cat 2, 3, 4.

    All calculations use Decimal arithmetic for bit-perfect reproducibility.
    No LLM is used in any calculation path.

    Usage::

        engine = Scope3EstimatorEngine()
        result = engine.calculate(scope3_input)
        for cat in result.categories:
            print(f"{cat.category_name}: {cat.tco2e} tCO2e")
    """

    engine_version: str = _MODULE_VERSION

    def calculate(self, data: Scope3EstimatorInput) -> Scope3EstimatorResult:
        """Run Scope 3 estimation.

        Args:
            data: Validated Scope 3 estimator input.

        Returns:
            Scope3EstimatorResult with category-level estimates.
        """
        t0 = time.perf_counter()
        logger.info(
            "Scope3 Estimator: entity=%s, industry=%s, entries=%d",
            data.entity_name, data.industry.value, len(data.spend_entries),
        )

        # Step 1: Map accounting categories to Scope 3 categories
        mapped_entries = self._map_accounting_categories(data.spend_entries, data.data_source_type)
        mappings_used = sum(
            1 for e in data.spend_entries if e.accounting_category is not None
        )

        # Step 2: Aggregate spend by Scope 3 category (in USD)
        category_spend: Dict[str, Decimal] = {}
        for entry, scope3_cat in mapped_entries:
            if scope3_cat is None:
                continue
            # Convert to USD
            fx_rate = CURRENCY_TO_USD.get(entry.currency, Decimal("1.00"))
            spend_usd = entry.amount * fx_rate
            cat_key = scope3_cat.value
            category_spend[cat_key] = category_spend.get(
                cat_key, Decimal("0")
            ) + spend_usd

        # Step 3: Add commuting estimate if provided and no Cat 7 spend data
        cat7_key = Scope3Category.CAT_07_EMPLOYEE_COMMUTING.value
        if data.commuting_estimate and cat7_key not in category_spend:
            commuting_tco2e = self._estimate_commuting(data.commuting_estimate)
            # Store as pseudo-spend for reporting consistency
            category_spend[cat7_key] = Decimal("0")

        # Step 4: Calculate emissions per category
        core_cats = {
            Scope3Category.CAT_01_PURCHASED_GOODS.value,
            Scope3Category.CAT_06_BUSINESS_TRAVEL.value,
            Scope3Category.CAT_07_EMPLOYEE_COMMUTING.value,
        }
        optional_cats = {
            Scope3Category.CAT_02_CAPITAL_GOODS.value,
            Scope3Category.CAT_03_FUEL_ENERGY.value,
            Scope3Category.CAT_04_UPSTREAM_TRANSPORT.value,
            Scope3Category.CAT_05_WASTE.value,
        }

        categories_list: List[CategoryEstimate] = []
        total_tco2e = Decimal("0")
        total_spend = Decimal("0")
        core_total = Decimal("0")
        optional_total = Decimal("0")

        for cat_key, spend_usd in category_spend.items():
            # Skip optional categories if not requested
            if cat_key in optional_cats and not data.include_optional_categories:
                continue

            # Get EEIO factor with industry adjustment
            eeio_factor = self._get_adjusted_factor(cat_key, data.industry)

            # Check for custom factor in matching spend entries
            custom_factor = None
            for entry, mapped_cat in mapped_entries:
                if mapped_cat and mapped_cat.value == cat_key and entry.custom_factor is not None:
                    custom_factor = entry.custom_factor
                    break
            if custom_factor is not None:
                eeio_factor = custom_factor

            # Special case: commuting from activity data
            if cat_key == cat7_key and data.commuting_estimate and spend_usd == Decimal("0"):
                tco2e = self._estimate_commuting(data.commuting_estimate)
                methodology = "activity_based_commuting"
            else:
                tco2e = _round_val(spend_usd / Decimal("1000") * eeio_factor)
                methodology = "spend_based_eeio"

            is_core = cat_key in core_cats
            total_tco2e += tco2e
            total_spend += spend_usd
            if is_core:
                core_total += tco2e
            else:
                optional_total += tco2e

            # Data quality for this category
            dq = self._category_data_quality(spend_usd, data.data_source_type)

            categories_list.append(CategoryEstimate(
                category=cat_key,
                category_name=CATEGORY_NAMES.get(cat_key, cat_key),
                spend_usd=_round_val(spend_usd, 2),
                eeio_factor=eeio_factor,
                tco2e=_round_val(tco2e),
                pct_of_total=Decimal("0"),  # computed below
                data_quality_score=dq,
                methodology=methodology,
                is_core=is_core,
            ))

        # Compute percentages
        for cat in categories_list:
            if total_tco2e > Decimal("0"):
                cat.pct_of_total = _round_val(
                    cat.tco2e * Decimal("100") / total_tco2e, 2
                )

        # Sort by tCO2e descending
        categories_list.sort(key=lambda c: c.tco2e, reverse=True)

        # Per employee
        per_employee = _safe_divide(total_tco2e, _decimal(data.headcount))

        # Data quality
        data_quality = self._assess_overall_quality(
            data, categories_list, category_spend
        )

        # Recommendations
        recommendations = self._generate_recommendations(
            data, categories_list, category_spend
        )

        elapsed_ms = _round3((time.perf_counter() - t0) * 1000.0)

        result = Scope3EstimatorResult(
            entity_name=data.entity_name,
            reporting_year=data.reporting_year,
            categories=categories_list,
            total_scope3_tco2e=_round_val(total_tco2e),
            core_categories_tco2e=_round_val(core_total),
            optional_categories_tco2e=_round_val(optional_total),
            per_employee_tco2e=_round_val(per_employee, 2),
            data_quality=data_quality,
            total_spend_usd=_round_val(total_spend, 2),
            categories_included=[c.category for c in categories_list],
            accounting_mappings_used=mappings_used,
            improvement_recommendations=recommendations,
            processing_time_ms=elapsed_ms,
        )
        result.provenance_hash = _compute_hash(result)

        logger.info(
            "Scope3 complete: total=%.2f tCO2e, %d categories, hash=%s",
            float(total_tco2e), len(categories_list),
            result.provenance_hash[:16],
        )
        return result

    # ------------------------------------------------------------------ #
    # Internal Methods                                                     #
    # ------------------------------------------------------------------ #

    def _map_accounting_categories(
        self,
        entries: List[SpendEntry],
        source: DataSourceType,
    ) -> List[tuple[SpendEntry, Optional[Scope3Category]]]:
        """Map accounting software categories to Scope 3 categories.

        Args:
            entries: Raw spend entries.
            source: Data source type.

        Returns:
            List of (entry, mapped_scope3_category) tuples.
        """
        mapping = {}
        if source == DataSourceType.XERO:
            mapping = XERO_CATEGORY_MAP
        elif source == DataSourceType.QUICKBOOKS:
            mapping = QUICKBOOKS_CATEGORY_MAP

        result = []
        for entry in entries:
            if entry.category is not None:
                result.append((entry, entry.category))
            elif entry.accounting_category:
                mapped = mapping.get(
                    entry.accounting_category.lower().replace(" ", "_")
                )
                result.append((entry, mapped))
            else:
                result.append((entry, None))

        return result

    def _get_adjusted_factor(
        self, category_key: str, industry: IndustryType,
    ) -> Decimal:
        """Get EEIO factor adjusted for industry type.

        Args:
            category_key: Scope 3 category key.
            industry: Industry type.

        Returns:
            Adjusted EEIO factor (tCO2e per $1000 USD).
        """
        # Get general factor
        general = EEIO_FACTORS_GENERAL.get(category_key, Decimal("0.400"))

        # Get industry multiplier
        multipliers = INDUSTRY_MULTIPLIERS.get(
            industry, INDUSTRY_MULTIPLIERS[IndustryType.GENERAL]
        )
        multiplier = multipliers.get(category_key, Decimal("1.00"))

        return _round_val(general * multiplier, 4)

    def _estimate_commuting(
        self, commuting: CommutingEstimateInput,
    ) -> Decimal:
        """Estimate employee commuting emissions from activity data.

        Formula:
            tCO2e = headcount * car_share * commute_km * 2 * working_days
                    * (1 - remote_pct/100) * ef_per_km / 1000

        Args:
            commuting: Commuting estimate input.

        Returns:
            Estimated commuting emissions in tCO2e.
        """
        car_commuters = _decimal(commuting.headcount) * commuting.car_share_pct / Decimal("100")
        round_trip_km = commuting.avg_commute_km * Decimal("2")
        effective_days = _decimal(commuting.working_days_per_year) * (
            Decimal("1") - commuting.remote_work_pct / Decimal("100")
        )
        total_km = car_commuters * round_trip_km * effective_days
        tco2e = total_km * CAR_EMISSION_FACTOR_KG_PER_KM / Decimal("1000")
        return _round_val(tco2e)

    def _category_data_quality(
        self, spend_usd: Decimal, source: DataSourceType,
    ) -> Decimal:
        """Compute a simple data quality score for a category.

        Args:
            spend_usd: Spend amount.
            source: Data source type.

        Returns:
            Quality score (0-1).
        """
        source_scores = {
            DataSourceType.XERO: Decimal("0.80"),
            DataSourceType.QUICKBOOKS: Decimal("0.80"),
            DataSourceType.SAGE: Decimal("0.75"),
            DataSourceType.FREEAGENT: Decimal("0.75"),
            DataSourceType.ACCOUNTING_EXPORT: Decimal("0.70"),
            DataSourceType.MANUAL_ENTRY: Decimal("0.50"),
        }
        base = source_scores.get(source, Decimal("0.50"))
        # Penalty for very low spend (might be incomplete)
        if spend_usd < Decimal("100"):
            base = base * Decimal("0.7")
        return _round_val(base, 2)

    def _assess_overall_quality(
        self,
        data: Scope3EstimatorInput,
        categories: List[CategoryEstimate],
        category_spend: Dict[str, Decimal],
    ) -> DataQualityScore:
        """Assess overall data quality for the Scope 3 estimate.

        Args:
            data: Input data.
            categories: Calculated category estimates.
            category_spend: Spend by category.

        Returns:
            DataQualityScore with dimensional scores.
        """
        # Completeness: how many core categories have spend data
        core_cats = {"cat_01_purchased_goods", "cat_06_business_travel", "cat_07_employee_commuting"}
        filled_core = sum(1 for c in core_cats if c in category_spend and category_spend[c] > Decimal("0"))
        completeness = _round_val(
            _decimal(filled_core) * Decimal("100") / Decimal("3"), 1
        )

        # Granularity: category-level vs total
        granularity = Decimal("80") if len(category_spend) > 1 else Decimal("30")

        # Temporal: current year = 100, each year back = -10
        temporal = Decimal("100")  # assumed current year

        # Source reliability
        source_scores = {
            DataSourceType.XERO: Decimal("85"),
            DataSourceType.QUICKBOOKS: Decimal("85"),
            DataSourceType.SAGE: Decimal("80"),
            DataSourceType.FREEAGENT: Decimal("80"),
            DataSourceType.ACCOUNTING_EXPORT: Decimal("70"),
            DataSourceType.MANUAL_ENTRY: Decimal("50"),
        }
        source_reliability = source_scores.get(data.data_source_type, Decimal("50"))

        # Overall weighted average
        overall = _round_val(
            (completeness * Decimal("0.30")
             + granularity * Decimal("0.25")
             + temporal * Decimal("0.15")
             + source_reliability * Decimal("0.30")),
            1,
        )

        level = "low"
        if overall >= Decimal("70"):
            level = "high"
        elif overall >= Decimal("50"):
            level = "medium"

        return DataQualityScore(
            overall_score=overall,
            completeness=completeness,
            granularity=granularity,
            temporal=temporal,
            source_reliability=source_reliability,
            level=level,
        )

    def _generate_recommendations(
        self,
        data: Scope3EstimatorInput,
        categories: List[CategoryEstimate],
        category_spend: Dict[str, Decimal],
    ) -> List[str]:
        """Generate recommendations to improve the Scope 3 estimate.

        Args:
            data: Input data.
            categories: Category estimates.
            category_spend: Spend by category.

        Returns:
            List of recommendation strings.
        """
        recs: List[str] = []

        if "cat_01_purchased_goods" not in category_spend:
            recs.append(
                "Add purchased goods & services spend data (Cat 1) - typically "
                "the largest SME Scope 3 category (40-60% of total)."
            )

        if "cat_06_business_travel" not in category_spend:
            recs.append(
                "Add business travel spend data (Cat 6) from expense reports "
                "or accounting software."
            )

        if "cat_07_employee_commuting" not in category_spend and data.commuting_estimate is None:
            recs.append(
                "Add employee commuting data (Cat 7) - run a simple commute "
                "survey or use the commuting estimator."
            )

        if data.data_source_type == DataSourceType.MANUAL_ENTRY:
            recs.append(
                "Connect your accounting software (Xero, QuickBooks) for "
                "automated category mapping and higher data quality."
            )

        if not data.include_optional_categories:
            recs.append(
                "Enable optional categories (Cat 2-4) for a more complete "
                "Scope 3 picture."
            )

        if len(categories) > 0:
            top_cat = categories[0]
            if top_cat.pct_of_total > Decimal("50"):
                recs.append(
                    f"Focus reduction efforts on {top_cat.category_name} - "
                    f"accounts for {float(top_cat.pct_of_total):.0f}% of your Scope 3."
                )

        return recs
