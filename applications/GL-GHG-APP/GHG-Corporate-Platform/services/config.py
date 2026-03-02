"""
GHG Corporate Platform Configuration

This module defines all configuration settings, enumerations, and constants
for the GL-GHG-APP v1.0 platform implementing the GHG Protocol Corporate
Accounting and Reporting Standard.

All settings use the GHG_APP_ prefix for environment variable overrides.

Example:
    >>> config = GHGAppConfig()
    >>> config.default_consolidation_approach
    <ConsolidationApproach.OPERATIONAL_CONTROL: 'operational_control'>
"""

from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Set

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class ConsolidationApproach(str, Enum):
    """GHG Protocol Ch 3 - Organizational boundary approaches."""

    OPERATIONAL_CONTROL = "operational_control"
    FINANCIAL_CONTROL = "financial_control"
    EQUITY_SHARE = "equity_share"


class ReportingPeriod(str, Enum):
    """Reporting period types for GHG inventories."""

    CALENDAR_YEAR = "calendar_year"
    FISCAL_YEAR = "fiscal_year"
    CUSTOM = "custom"


class GHGGas(str, Enum):
    """Seven GHGs covered by the GHG Protocol / Kyoto Protocol."""

    CO2 = "CO2"
    CH4 = "CH4"
    N2O = "N2O"
    HFCS = "HFCs"
    PFCS = "PFCs"
    SF6 = "SF6"
    NF3 = "NF3"


class Scope(str, Enum):
    """GHG Protocol emission scopes."""

    SCOPE_1 = "scope_1"
    SCOPE_2_LOCATION = "scope_2_location"
    SCOPE_2_MARKET = "scope_2_market"
    SCOPE_3 = "scope_3"


class Scope1Category(str, Enum):
    """Scope 1 direct emission categories mapped to MRV agents."""

    STATIONARY_COMBUSTION = "stationary_combustion"
    MOBILE_COMBUSTION = "mobile_combustion"
    PROCESS_EMISSIONS = "process_emissions"
    FUGITIVE_EMISSIONS = "fugitive_emissions"
    REFRIGERANTS = "refrigerants"
    LAND_USE = "land_use"
    WASTE_TREATMENT = "waste_treatment"
    AGRICULTURAL = "agricultural"


class Scope3Category(str, Enum):
    """Scope 3 indirect emission categories (15 categories)."""

    CAT1_PURCHASED_GOODS = "cat1_purchased_goods"
    CAT2_CAPITAL_GOODS = "cat2_capital_goods"
    CAT3_FUEL_ENERGY = "cat3_fuel_energy"
    CAT4_UPSTREAM_TRANSPORT = "cat4_upstream_transport"
    CAT5_WASTE_GENERATED = "cat5_waste_generated"
    CAT6_BUSINESS_TRAVEL = "cat6_business_travel"
    CAT7_EMPLOYEE_COMMUTING = "cat7_employee_commuting"
    CAT8_UPSTREAM_LEASED = "cat8_upstream_leased"
    CAT9_DOWNSTREAM_TRANSPORT = "cat9_downstream_transport"
    CAT10_PROCESSING_SOLD = "cat10_processing_sold"
    CAT11_USE_OF_SOLD = "cat11_use_of_sold"
    CAT12_END_OF_LIFE = "cat12_end_of_life"
    CAT13_DOWNSTREAM_LEASED = "cat13_downstream_leased"
    CAT14_FRANCHISES = "cat14_franchises"
    CAT15_INVESTMENTS = "cat15_investments"


class VerificationLevel(str, Enum):
    """Assurance / verification levels per GHG Protocol guidance."""

    NONE = "none"
    INTERNAL_REVIEW = "internal_review"
    LIMITED_ASSURANCE = "limited_assurance"
    REASONABLE_ASSURANCE = "reasonable_assurance"


class IntensityDenominator(str, Enum):
    """Denominator types for intensity metrics (GHG Protocol Ch 12)."""

    REVENUE = "revenue"
    EMPLOYEES = "employees"
    PRODUCTION_UNITS = "production_units"
    FLOOR_AREA = "floor_area"
    CUSTOM = "custom"


class TargetType(str, Enum):
    """Emission reduction target types."""

    ABSOLUTE = "absolute"
    INTENSITY = "intensity"


class DataQualityTier(str, Enum):
    """Data quality tiers following GHG Protocol guidance."""

    TIER_1 = "tier_1"  # Estimated / industry averages
    TIER_2 = "tier_2"  # Calculated from activity data + published EFs
    TIER_3 = "tier_3"  # Direct measurement / supplier-specific


class EntityType(str, Enum):
    """Types of organizational entities."""

    SUBSIDIARY = "subsidiary"
    FACILITY = "facility"
    OPERATION = "operation"


class VerificationStatus(str, Enum):
    """Verification workflow states."""

    DRAFT = "draft"
    IN_REVIEW = "in_review"
    SUBMITTED = "submitted"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXTERNALLY_VERIFIED = "externally_verified"


class FindingType(str, Enum):
    """Types of verification findings."""

    OBSERVATION = "observation"
    MINOR_NONCONFORMITY = "minor_nonconformity"
    MAJOR_NONCONFORMITY = "major_nonconformity"
    OPPORTUNITY_FOR_IMPROVEMENT = "opportunity_for_improvement"


class ReportFormat(str, Enum):
    """Supported report output formats."""

    JSON = "json"
    CSV = "csv"
    EXCEL = "excel"
    PDF = "pdf"


class FindingSeverity(str, Enum):
    """Severity classification for verification findings."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# ---------------------------------------------------------------------------
# GWP Values -- IPCC AR5 (100-year time horizon)
# ---------------------------------------------------------------------------

GWP_AR5: Dict[GHGGas, int] = {
    GHGGas.CO2: 1,
    GHGGas.CH4: 28,
    GHGGas.N2O: 265,
    GHGGas.HFCS: 1430,   # Representative HFC-134a
    GHGGas.PFCS: 6630,   # Representative PFC-14 (CF4)
    GHGGas.SF6: 23500,
    GHGGas.NF3: 16100,
}


# ---------------------------------------------------------------------------
# Sector Benchmarks -- tCO2e per unit (revenue in million USD)
# ---------------------------------------------------------------------------

SECTOR_BENCHMARKS: Dict[str, Dict[str, Decimal]] = {
    "energy": {
        "revenue": Decimal("850.0"),
        "employees": Decimal("45.0"),
    },
    "manufacturing": {
        "revenue": Decimal("420.0"),
        "employees": Decimal("22.0"),
    },
    "technology": {
        "revenue": Decimal("28.0"),
        "employees": Decimal("3.5"),
    },
    "finance": {
        "revenue": Decimal("12.0"),
        "employees": Decimal("2.8"),
    },
    "retail": {
        "revenue": Decimal("65.0"),
        "employees": Decimal("5.2"),
    },
    "transport": {
        "revenue": Decimal("520.0"),
        "employees": Decimal("35.0"),
    },
    "healthcare": {
        "revenue": Decimal("95.0"),
        "employees": Decimal("8.5"),
    },
    "agriculture": {
        "revenue": Decimal("380.0"),
        "employees": Decimal("28.0"),
    },
    "real_estate": {
        "revenue": Decimal("40.0"),
        "employees": Decimal("6.0"),
    },
    "mining": {
        "revenue": Decimal("720.0"),
        "employees": Decimal("55.0"),
    },
}


# ---------------------------------------------------------------------------
# Default Uncertainty Ranges (% coefficient of variation by tier)
# ---------------------------------------------------------------------------

UNCERTAINTY_CV_BY_TIER: Dict[DataQualityTier, Decimal] = {
    DataQualityTier.TIER_1: Decimal("50.0"),   # +/- 50% CV
    DataQualityTier.TIER_2: Decimal("20.0"),   # +/- 20% CV
    DataQualityTier.TIER_3: Decimal("5.0"),    # +/- 5% CV
}


# ---------------------------------------------------------------------------
# Main Configuration Class
# ---------------------------------------------------------------------------

class GHGAppConfig(BaseSettings):
    """
    GL-GHG-APP v1.0 platform configuration.

    All settings can be overridden via environment variables prefixed
    with ``GHG_APP_``.  For example ``GHG_APP_DEFAULT_CONSOLIDATION_APPROACH``
    maps to ``default_consolidation_approach``.
    """

    model_config = {"env_prefix": "GHG_APP_"}

    # -- Application Metadata -----------------------------------------------
    app_name: str = Field(
        default="GL-GHG-APP",
        description="Application display name",
    )
    app_version: str = Field(
        default="1.0.0",
        description="Semantic version of the application",
    )

    # -- Boundary Defaults --------------------------------------------------
    default_consolidation_approach: ConsolidationApproach = Field(
        default=ConsolidationApproach.OPERATIONAL_CONTROL,
        description="Default organizational boundary approach",
    )
    default_reporting_period: ReportingPeriod = Field(
        default=ReportingPeriod.CALENDAR_YEAR,
        description="Default reporting period type",
    )

    # -- Scopes Enabled -----------------------------------------------------
    scope1_enabled: bool = Field(default=True, description="Enable Scope 1")
    scope2_enabled: bool = Field(default=True, description="Enable Scope 2")
    scope3_enabled: bool = Field(default=True, description="Enable Scope 3")

    # -- Base Year ----------------------------------------------------------
    base_year_significance_threshold: Decimal = Field(
        default=Decimal("5.0"),
        description="Percentage threshold triggering base year recalculation",
    )
    base_year_auto_recalculate: bool = Field(
        default=False,
        description="Automatically recalculate base year on structural changes",
    )

    # -- Uncertainty Engine -------------------------------------------------
    monte_carlo_iterations: int = Field(
        default=10_000,
        ge=1_000,
        le=1_000_000,
        description="Number of Monte Carlo iterations",
    )
    confidence_level: Decimal = Field(
        default=Decimal("95.0"),
        description="Confidence level for uncertainty ranges (%)",
    )

    # -- Data Quality -------------------------------------------------------
    default_data_quality_tier: DataQualityTier = Field(
        default=DataQualityTier.TIER_1,
        description="Default data quality tier when not specified",
    )
    minimum_completeness_pct: Decimal = Field(
        default=Decimal("95.0"),
        description="Minimum completeness percentage for inventory approval",
    )

    # -- Report Generation --------------------------------------------------
    default_report_format: ReportFormat = Field(
        default=ReportFormat.JSON,
        description="Default report export format",
    )
    report_storage_path: str = Field(
        default="reports/",
        description="Path prefix for generated reports",
    )

    # -- Verification -------------------------------------------------------
    default_verification_level: VerificationLevel = Field(
        default=VerificationLevel.INTERNAL_REVIEW,
        description="Default verification level for new inventories",
    )

    # -- Targets / SBTi -----------------------------------------------------
    sbti_near_term_years: int = Field(
        default=5,
        description="SBTi near-term target horizon (years)",
    )
    sbti_long_term_years: int = Field(
        default=15,
        description="SBTi long-term target horizon (years)",
    )
    sbti_1_5c_annual_reduction: Decimal = Field(
        default=Decimal("4.2"),
        description="Required annual reduction (%) for 1.5C pathway",
    )
    sbti_2c_annual_reduction: Decimal = Field(
        default=Decimal("2.5"),
        description="Required annual reduction (%) for well-below 2C pathway",
    )

    # -- MRV Agent Integration ----------------------------------------------
    mrv_agent_timeout_seconds: int = Field(
        default=30,
        ge=5,
        le=300,
        description="Timeout for individual MRV agent calls (seconds)",
    )
    mrv_agent_retry_count: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Number of retries for failed MRV agent calls",
    )

    # -- Logging / Debug ----------------------------------------------------
    log_level: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    )
    debug_mode: bool = Field(
        default=False,
        description="Enable debug mode with verbose logging",
    )
