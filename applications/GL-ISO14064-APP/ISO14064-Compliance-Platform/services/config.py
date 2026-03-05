"""
ISO 14064-1:2018 Compliance Platform Configuration

This module defines all configuration settings, enumerations, GWP tables, and
constants for the GL-ISO14064-APP v1.0 platform implementing the ISO 14064-1:2018
standard for organizational-level quantification and reporting of GHG emissions
and removals.

ISO 14064-1:2018 uses six emission/removal categories instead of the GHG Protocol
scope model.  This configuration maps those categories to the GreenLang MRV agent
layer so that deterministic calculation engines feed into the ISO reporting pipeline.

All settings use the ISO14064_APP_ prefix for environment variable overrides.

Example:
    >>> config = ISO14064AppConfig()
    >>> config.default_consolidation_approach
    <ConsolidationApproach.OPERATIONAL_CONTROL: 'operational_control'>
    >>> config.default_gwp_source
    <GWPSource.AR5: 'ar5'>
"""

from decimal import Decimal
from enum import Enum
from typing import Dict, List

from pydantic import Field
from pydantic_settings import BaseSettings


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class ConsolidationApproach(str, Enum):
    """ISO 14064-1:2018 Clause 5.1 -- Organizational boundary approaches."""

    OPERATIONAL_CONTROL = "operational_control"
    FINANCIAL_CONTROL = "financial_control"
    EQUITY_SHARE = "equity_share"


class ReportingPeriod(str, Enum):
    """Reporting period types for ISO 14064-1 inventories."""

    CALENDAR_YEAR = "calendar_year"
    FISCAL_YEAR = "fiscal_year"
    CUSTOM = "custom"


class GHGGas(str, Enum):
    """Seven GHGs specified by ISO 14064-1:2018 Clause 5.2.4."""

    CO2 = "CO2"
    CH4 = "CH4"
    N2O = "N2O"
    HFCS = "HFCs"
    PFCS = "PFCs"
    SF6 = "SF6"
    NF3 = "NF3"


class ISOCategory(str, Enum):
    """
    ISO 14064-1:2018 Clause 5.2.2 -- Six emission/removal categories.

    Category 1: Direct GHG emissions and removals
    Category 2: Indirect GHG emissions from imported energy
    Category 3: Indirect GHG emissions from transportation
    Category 4: Indirect GHG emissions from products used by the organization
    Category 5: Indirect GHG emissions associated with the use of products
                 from the organization
    Category 6: Indirect GHG emissions from other sources
    """

    CATEGORY_1_DIRECT = "category_1_direct"
    CATEGORY_2_ENERGY = "category_2_energy"
    CATEGORY_3_TRANSPORT = "category_3_transport"
    CATEGORY_4_PRODUCTS_USED = "category_4_products_used"
    CATEGORY_5_PRODUCTS_FROM_ORG = "category_5_products_from_org"
    CATEGORY_6_OTHER = "category_6_other"


class QuantificationMethod(str, Enum):
    """ISO 14064-1:2018 Clause 5.2.4 -- Quantification methods."""

    CALCULATION_BASED = "calculation_based"
    DIRECT_MEASUREMENT = "direct_measurement"
    MASS_BALANCE = "mass_balance"


class RemovalType(str, Enum):
    """Types of GHG removals per ISO 14064-1:2018 Clause 5.2.3."""

    FORESTRY = "forestry"
    SOIL_CARBON = "soil_carbon"
    CCS = "ccs"
    DIRECT_AIR_CAPTURE = "direct_air_capture"
    BECCS = "beccs"
    WETLAND_RESTORATION = "wetland_restoration"
    OCEAN_BASED = "ocean_based"
    OTHER = "other"


class PermanenceLevel(str, Enum):
    """Permanence classification for GHG removals."""

    PERMANENT = "permanent"          # > 1000 years (geological storage)
    LONG_TERM = "long_term"          # 100-1000 years
    MEDIUM_TERM = "medium_term"      # 25-100 years
    SHORT_TERM = "short_term"        # 5-25 years
    REVERSIBLE = "reversible"        # < 5 years


class SignificanceLevel(str, Enum):
    """ISO 14064-1:2018 Clause 5.2.2 significance assessment outcomes."""

    SIGNIFICANT = "significant"
    NOT_SIGNIFICANT = "not_significant"
    UNDER_REVIEW = "under_review"


class DataQualityTier(str, Enum):
    """Data quality tiers following ISO 14064-1:2018 Clause 6.3 guidance."""

    TIER_1 = "tier_1"   # Estimated / industry averages
    TIER_2 = "tier_2"   # Calculated from activity data + published EFs
    TIER_3 = "tier_3"   # Supplier-specific or site-level data
    TIER_4 = "tier_4"   # Direct measurement / continuous monitoring


class ReportFormat(str, Enum):
    """Supported report output formats."""

    PDF = "pdf"
    EXCEL = "excel"
    JSON = "json"
    CSV = "csv"


class VerificationLevel(str, Enum):
    """ISO 14064-3:2019 assurance / verification levels."""

    LIMITED = "limited"
    REASONABLE = "reasonable"
    NOT_VERIFIED = "not_verified"


class VerificationStage(str, Enum):
    """Verification workflow stages per ISO 14064-3:2019."""

    DRAFT = "draft"
    INTERNAL_REVIEW = "internal_review"
    APPROVED = "approved"
    EXTERNAL_VERIFICATION = "external_verification"
    VERIFIED = "verified"


class FindingSeverity(str, Enum):
    """Severity classification for verification findings."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class FindingStatus(str, Enum):
    """Status of verification findings."""

    OPEN = "open"
    IN_PROGRESS = "in_progress"
    RESOLVED = "resolved"
    ACCEPTED = "accepted"


class ActionStatus(str, Enum):
    """Status of improvement actions per ISO 14064-1:2018 Clause 9."""

    PLANNED = "planned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    DEFERRED = "deferred"
    CANCELLED = "cancelled"


class ActionCategory(str, Enum):
    """Categories for improvement and management actions."""

    EMISSION_REDUCTION = "emission_reduction"
    REMOVAL_ENHANCEMENT = "removal_enhancement"
    DATA_IMPROVEMENT = "data_improvement"
    PROCESS_IMPROVEMENT = "process_improvement"


class GWPSource(str, Enum):
    """Source of Global Warming Potential values."""

    AR5 = "ar5"
    AR6 = "ar6"
    CUSTOM = "custom"


class InventoryStatus(str, Enum):
    """Lifecycle status of an ISO 14064-1 GHG inventory."""

    DRAFT = "draft"
    IN_REVIEW = "in_review"
    APPROVED = "approved"
    VERIFIED = "verified"
    PUBLISHED = "published"


class ReportStatus(str, Enum):
    """Report lifecycle status."""

    DRAFT = "draft"
    REVIEW = "review"
    FINAL = "final"
    PUBLISHED = "published"


# ---------------------------------------------------------------------------
# GWP Values -- IPCC AR5 (100-year time horizon, GTP excluded)
# ---------------------------------------------------------------------------

GWP_AR5: Dict[GHGGas, float] = {
    GHGGas.CO2: 1,
    GHGGas.CH4: 28,
    GHGGas.N2O: 265,
    GHGGas.HFCS: 1430,    # Representative HFC-134a
    GHGGas.PFCS: 6630,    # Representative PFC-14 (CF4)
    GHGGas.SF6: 23500,
    GHGGas.NF3: 16100,
}


# ---------------------------------------------------------------------------
# GWP Values -- IPCC AR6 (100-year time horizon)
# ---------------------------------------------------------------------------

GWP_AR6: Dict[GHGGas, float] = {
    GHGGas.CO2: 1,
    GHGGas.CH4: 27.9,      # AR6 WG1 Table 7.15
    GHGGas.N2O: 273,       # AR6 WG1 Table 7.15
    GHGGas.HFCS: 1530,     # Representative HFC-134a (AR6)
    GHGGas.PFCS: 7380,     # Representative PFC-14 CF4 (AR6)
    GHGGas.SF6: 25200,     # AR6 WG1 Table 7.15
    GHGGas.NF3: 17400,     # AR6 WG1 Table 7.15
}


# ---------------------------------------------------------------------------
# Combined GWP Lookup by Source
# ---------------------------------------------------------------------------

GWP_TABLES: Dict[GWPSource, Dict[GHGGas, float]] = {
    GWPSource.AR5: GWP_AR5,
    GWPSource.AR6: GWP_AR6,
}


# ---------------------------------------------------------------------------
# Default Uncertainty Ranges (% coefficient of variation by tier)
# ---------------------------------------------------------------------------

UNCERTAINTY_CV_BY_TIER: Dict[DataQualityTier, Decimal] = {
    DataQualityTier.TIER_1: Decimal("50.0"),   # +/- 50% CV
    DataQualityTier.TIER_2: Decimal("20.0"),   # +/- 20% CV
    DataQualityTier.TIER_3: Decimal("5.0"),    # +/- 5% CV
    DataQualityTier.TIER_4: Decimal("2.0"),    # +/- 2% CV (direct measurement)
}


# ---------------------------------------------------------------------------
# ISO Category Display Names
# ---------------------------------------------------------------------------

ISO_CATEGORY_NAMES: Dict[ISOCategory, str] = {
    ISOCategory.CATEGORY_1_DIRECT: "Category 1 - Direct GHG emissions and removals",
    ISOCategory.CATEGORY_2_ENERGY: "Category 2 - Indirect GHG emissions from imported energy",
    ISOCategory.CATEGORY_3_TRANSPORT: "Category 3 - Indirect GHG emissions from transportation",
    ISOCategory.CATEGORY_4_PRODUCTS_USED: (
        "Category 4 - Indirect GHG emissions from products used by the organization"
    ),
    ISOCategory.CATEGORY_5_PRODUCTS_FROM_ORG: (
        "Category 5 - Indirect GHG emissions associated with the use of products "
        "from the organization"
    ),
    ISOCategory.CATEGORY_6_OTHER: "Category 6 - Indirect GHG emissions from other sources",
}


# ---------------------------------------------------------------------------
# MRV Agent -> ISO Category Mapping
# ---------------------------------------------------------------------------

MRV_AGENT_TO_ISO_CATEGORY: Dict[str, ISOCategory] = {
    # Category 1 - Direct (Scope 1 equivalent)
    "stationary_combustion": ISOCategory.CATEGORY_1_DIRECT,
    "mobile_combustion": ISOCategory.CATEGORY_1_DIRECT,
    "process_emissions": ISOCategory.CATEGORY_1_DIRECT,
    "fugitive_emissions": ISOCategory.CATEGORY_1_DIRECT,
    "refrigerants": ISOCategory.CATEGORY_1_DIRECT,
    "land_use": ISOCategory.CATEGORY_1_DIRECT,
    "waste_treatment": ISOCategory.CATEGORY_1_DIRECT,
    "agricultural": ISOCategory.CATEGORY_1_DIRECT,
    # Category 2 - Energy indirect (Scope 2 equivalent)
    "scope2_location": ISOCategory.CATEGORY_2_ENERGY,
    "scope2_market": ISOCategory.CATEGORY_2_ENERGY,
    "steam_heat_purchase": ISOCategory.CATEGORY_2_ENERGY,
    "cooling_purchase": ISOCategory.CATEGORY_2_ENERGY,
    # Category 3 - Transportation indirect
    "upstream_transportation": ISOCategory.CATEGORY_3_TRANSPORT,
    "downstream_transportation": ISOCategory.CATEGORY_3_TRANSPORT,
    "business_travel": ISOCategory.CATEGORY_3_TRANSPORT,
    "employee_commuting": ISOCategory.CATEGORY_3_TRANSPORT,
    # Category 4 - Products used by the organization
    "purchased_goods_services": ISOCategory.CATEGORY_4_PRODUCTS_USED,
    "capital_goods": ISOCategory.CATEGORY_4_PRODUCTS_USED,
    "fuel_energy_activities": ISOCategory.CATEGORY_4_PRODUCTS_USED,
    "waste_generated": ISOCategory.CATEGORY_4_PRODUCTS_USED,
    "upstream_leased_assets": ISOCategory.CATEGORY_4_PRODUCTS_USED,
    # Category 5 - Products from the organization
    "processing_sold_products": ISOCategory.CATEGORY_5_PRODUCTS_FROM_ORG,
    "use_of_sold_products": ISOCategory.CATEGORY_5_PRODUCTS_FROM_ORG,
    "end_of_life_treatment": ISOCategory.CATEGORY_5_PRODUCTS_FROM_ORG,
    "downstream_leased_assets": ISOCategory.CATEGORY_5_PRODUCTS_FROM_ORG,
    "franchises": ISOCategory.CATEGORY_5_PRODUCTS_FROM_ORG,
    # Category 6 - Other sources
    "investments": ISOCategory.CATEGORY_6_OTHER,
}


# ---------------------------------------------------------------------------
# Sector Benchmarks -- tCO2e per unit (revenue in million USD)
# ---------------------------------------------------------------------------

SECTOR_BENCHMARKS: Dict[str, Dict[str, Decimal]] = {
    "energy": {
        "revenue_intensity": Decimal("850.0"),
        "employee_intensity": Decimal("45.0"),
    },
    "manufacturing": {
        "revenue_intensity": Decimal("420.0"),
        "employee_intensity": Decimal("22.0"),
    },
    "technology": {
        "revenue_intensity": Decimal("28.0"),
        "employee_intensity": Decimal("3.5"),
    },
    "finance": {
        "revenue_intensity": Decimal("12.0"),
        "employee_intensity": Decimal("2.8"),
    },
    "retail": {
        "revenue_intensity": Decimal("65.0"),
        "employee_intensity": Decimal("5.2"),
    },
    "transport": {
        "revenue_intensity": Decimal("520.0"),
        "employee_intensity": Decimal("35.0"),
    },
    "healthcare": {
        "revenue_intensity": Decimal("95.0"),
        "employee_intensity": Decimal("8.5"),
    },
    "agriculture": {
        "revenue_intensity": Decimal("380.0"),
        "employee_intensity": Decimal("28.0"),
    },
    "real_estate": {
        "revenue_intensity": Decimal("40.0"),
        "employee_intensity": Decimal("6.0"),
    },
    "mining": {
        "revenue_intensity": Decimal("720.0"),
        "employee_intensity": Decimal("55.0"),
    },
    "chemicals": {
        "revenue_intensity": Decimal("560.0"),
        "employee_intensity": Decimal("38.0"),
    },
    "construction": {
        "revenue_intensity": Decimal("180.0"),
        "employee_intensity": Decimal("14.0"),
    },
}


# ---------------------------------------------------------------------------
# ISO 14064-1 Mandatory Reporting Element IDs (Clause 9)
# ---------------------------------------------------------------------------

MANDATORY_REPORTING_ELEMENTS: List[str] = [
    "MRE-01",   # Reporting organization description
    "MRE-02",   # Responsible person
    "MRE-03",   # Reporting period
    "MRE-04",   # Organizational boundary and consolidation approach
    "MRE-05",   # Direct GHG emissions (Category 1)
    "MRE-06",   # Indirect GHG emissions from imported energy (Category 2)
    "MRE-07",   # Quantification methodology description
    "MRE-08",   # GHG emissions and removals by gas type
    "MRE-09",   # Emission factors and GWP values used
    "MRE-10",   # Biogenic CO2 emissions reported separately
    "MRE-11",   # Base year and recalculation policy
    "MRE-12",   # Significance assessment for indirect categories (3-6)
    "MRE-13",   # Exclusions with justification
    "MRE-14",   # Uncertainty assessment
]


# ---------------------------------------------------------------------------
# Main Configuration Class
# ---------------------------------------------------------------------------

class ISO14064AppConfig(BaseSettings):
    """
    GL-ISO14064-APP v1.0 platform configuration.

    All settings can be overridden via environment variables prefixed
    with ``ISO14064_APP_``.  For example ``ISO14064_APP_DEFAULT_CONSOLIDATION_APPROACH``
    maps to ``default_consolidation_approach``.

    Example:
        >>> config = ISO14064AppConfig()
        >>> config.app_name
        'GL-ISO14064-APP'
        >>> config.significance_threshold_percent
        Decimal('1.0')
    """

    model_config = {"env_prefix": "ISO14064_APP_"}

    # -- Application Metadata -----------------------------------------------
    app_name: str = Field(
        default="GL-ISO14064-APP",
        description="Application display name",
    )
    version: str = Field(
        default="1.0.0",
        description="Semantic version of the application",
    )
    debug: bool = Field(
        default=False,
        description="Enable debug mode with verbose logging",
    )

    # -- Boundary Defaults --------------------------------------------------
    default_consolidation_approach: ConsolidationApproach = Field(
        default=ConsolidationApproach.OPERATIONAL_CONTROL,
        description="Default organizational boundary approach (ISO 14064-1 Clause 5.1)",
    )
    default_reporting_period: ReportingPeriod = Field(
        default=ReportingPeriod.CALENDAR_YEAR,
        description="Default reporting period type",
    )

    # -- GWP Source ---------------------------------------------------------
    default_gwp_source: GWPSource = Field(
        default=GWPSource.AR5,
        description="Default GWP source (AR5 or AR6)",
    )

    # -- Significance Thresholds (ISO 14064-1 Clause 5.2.2) ----------------
    significance_threshold_percent: Decimal = Field(
        default=Decimal("1.0"),
        ge=Decimal("0"),
        le=Decimal("100"),
        description=(
            "Percentage of total emissions below which an indirect category "
            "may be assessed as not significant"
        ),
    )

    # -- Base Year / Recalculation (ISO 14064-1 Clause 5.3) ----------------
    recalculation_threshold_percent: Decimal = Field(
        default=Decimal("5.0"),
        ge=Decimal("0"),
        le=Decimal("100"),
        description=(
            "Percentage threshold triggering base year recalculation on "
            "structural or methodological changes"
        ),
    )
    base_year_auto_recalculate: bool = Field(
        default=False,
        description="Automatically recalculate base year on structural changes",
    )

    # -- Uncertainty Engine (ISO 14064-1 Clause 6.3) -----------------------
    monte_carlo_iterations: int = Field(
        default=10_000,
        ge=1_000,
        le=1_000_000,
        description="Number of Monte Carlo simulation iterations",
    )
    confidence_levels: List[int] = Field(
        default=[90, 95, 99],
        description="Confidence levels (%) for uncertainty interval reporting",
    )

    # -- Reporting Year -----------------------------------------------------
    reporting_year: int = Field(
        default=2025,
        ge=1990,
        le=2100,
        description="Current reporting year",
    )

    # -- Data Quality -------------------------------------------------------
    default_data_quality_tier: DataQualityTier = Field(
        default=DataQualityTier.TIER_1,
        description="Default data quality tier when not specified",
    )
    minimum_completeness_pct: Decimal = Field(
        default=Decimal("95.0"),
        ge=Decimal("0"),
        le=Decimal("100"),
        description="Minimum completeness percentage for inventory approval",
    )

    # -- Report Generation --------------------------------------------------
    default_report_format: ReportFormat = Field(
        default=ReportFormat.JSON,
        description="Default report export format",
    )
    report_storage_path: str = Field(
        default="reports/iso14064/",
        description="Path prefix for generated reports",
    )

    # -- Verification (ISO 14064-3:2019) ------------------------------------
    default_verification_level: VerificationLevel = Field(
        default=VerificationLevel.NOT_VERIFIED,
        description="Default verification level for new inventories",
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

    # -- Logging ------------------------------------------------------------
    log_level: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    )
