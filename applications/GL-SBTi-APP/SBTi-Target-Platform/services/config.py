"""
GL-SBTi-APP v1.0 -- Science Based Targets Platform Configuration

Enumerations, SBTi criteria constants, sector decarbonization pathways,
PCAF data quality tiers, and application settings for implementing the
SBTi Corporate Net-Zero Standard v1.2 and FI Net-Zero Standard v1.1.

The SBTi framework requires companies to set emission reduction targets
consistent with the level of decarbonization needed to keep global
temperature increase to 1.5C above pre-industrial levels, covering
near-term (5-10 years), long-term (by 2050), and net-zero targets.

All settings use the SBTI_APP_ prefix for environment variable overrides.

Reference:
    - SBTi Corporate Net-Zero Standard v1.2 (April 2023)
    - SBTi Criteria and Recommendations v5.1 (April 2023)
    - SBTi Financial Institutions Net-Zero Standard v1.1 (April 2024)
    - GHG Protocol Corporate Standard (2004, revised 2015)
    - PCAF Global GHG Accounting & Reporting Standard (2022)
    - SBTi Sector Guidance (multiple sectors, 2020-2024)

Example:
    >>> config = SBTiAppConfig()
    >>> config.app_name
    'GL-SBTi-APP'
    >>> config.recalculation_threshold_pct
    Decimal('5.0')
    >>> SECTOR_PATHWAYS["power"]["annual_reduction_1_5c"]
    Decimal('7.0')
"""

from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List

from pydantic import Field
from pydantic_settings import BaseSettings


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class TargetType(str, Enum):
    """SBTi target classification types."""

    NEAR_TERM = "near_term"
    LONG_TERM = "long_term"
    NET_ZERO = "net_zero"


class TargetScope(str, Enum):
    """GHG Protocol emission scopes for SBTi target boundary."""

    SCOPE_1 = "scope_1"
    SCOPE_2 = "scope_2"
    SCOPE_3 = "scope_3"
    SCOPE_1_2 = "scope_1_2"
    SCOPE_1_2_3 = "scope_1_2_3"


class TargetMethod(str, Enum):
    """SBTi target-setting methodological approaches."""

    ABSOLUTE_CONTRACTION = "absolute_contraction"
    SECTORAL_DECARBONIZATION = "sectoral_decarbonization"
    PHYSICAL_INTENSITY = "physical_intensity"
    ECONOMIC_INTENSITY = "economic_intensity"


class AmbitionLevel(str, Enum):
    """Temperature ambition levels for SBTi alignment."""

    ONE_POINT_FIVE_C = "1.5c"
    WELL_BELOW_2C = "well_below_2c"
    TWO_C = "2c"
    NOT_ALIGNED = "not_aligned"


class ValidationStatus(str, Enum):
    """SBTi target validation lifecycle states."""

    COMMITMENT_LETTER = "commitment_letter"
    TARGET_SUBMITTED = "target_submitted"
    VALIDATION_IN_PROGRESS = "validation_in_progress"
    VALIDATED = "validated"
    REVALIDATION_REQUIRED = "revalidation_required"
    EXPIRED = "expired"
    REMOVED = "removed"


class RecalculationTrigger(str, Enum):
    """Triggers for base year emissions recalculation per SBTi criteria."""

    STRUCTURAL_CHANGE = "structural_change"
    METHODOLOGY_CHANGE = "methodology_change"
    DATA_ERROR = "data_error"
    MERGERS_ACQUISITIONS = "mergers_acquisitions"
    DIVESTITURE = "divestiture"
    OUTSOURCING_INSOURCING = "outsourcing_insourcing"
    ORGANIC_GROWTH = "organic_growth"


class ReviewOutcome(str, Enum):
    """Five-year review outcomes per SBTi requirements."""

    RENEWED = "renewed"
    UPDATED = "updated"
    EXPIRED = "expired"
    PENDING = "pending"


class FIAssetClass(str, Enum):
    """PCAF asset classes for financial institution portfolio accounting."""

    LISTED_EQUITY = "listed_equity"
    CORPORATE_BOND = "corporate_bond"
    BUSINESS_LOAN = "business_loan"
    PROJECT_FINANCE = "project_finance"
    COMMERCIAL_REAL_ESTATE = "commercial_real_estate"
    MORTGAGE = "mortgage"
    MOTOR_VEHICLE_LOAN = "motor_vehicle_loan"
    SOVEREIGN_BOND = "sovereign_bond"


class FITargetType(str, Enum):
    """SBTi FI target types for portfolio decarbonization."""

    PORTFOLIO_COVERAGE = "portfolio_coverage"
    SECTORAL_DECARBONIZATION = "sectoral_decarbonization"
    PORTFOLIO_TEMPERATURE = "portfolio_temperature"
    ENGAGEMENT = "engagement"


class PCAFDataQuality(str, Enum):
    """PCAF data quality scores (1=best, 5=worst)."""

    DQ_1 = "dq_1"
    DQ_2 = "dq_2"
    DQ_3 = "dq_3"
    DQ_4 = "dq_4"
    DQ_5 = "dq_5"


class SBTiSector(str, Enum):
    """SBTi sector classification for sector-specific pathways."""

    POWER = "power"
    OIL_GAS = "oil_gas"
    TRANSPORT = "transport"
    BUILDINGS = "buildings"
    CEMENT = "cement"
    STEEL = "steel"
    ALUMINIUM = "aluminium"
    CHEMICALS = "chemicals"
    PULP_PAPER = "pulp_paper"
    AGRICULTURE = "agriculture"
    APPAREL_FOOTWEAR = "apparel_footwear"
    AVIATION = "aviation"
    MARITIME = "maritime"
    ICT = "ict"
    FINANCIAL_INSTITUTIONS = "financial_institutions"
    GENERAL = "general"


class ReportFormat(str, Enum):
    """Supported report output formats."""

    PDF = "pdf"
    EXCEL = "excel"
    JSON = "json"
    XML = "xml"


class GapSeverity(str, Enum):
    """Severity levels for gap analysis findings."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class GapCategory(str, Enum):
    """Categories of gaps identified in SBTi readiness assessment."""

    DATA = "data"
    AMBITION = "ambition"
    PROCESS = "process"
    GOVERNANCE = "governance"
    COVERAGE = "coverage"
    METHODOLOGY = "methodology"


class NotificationType(str, Enum):
    """Types of automated notifications for target lifecycle events."""

    REVIEW_12_MONTHS = "review_12_months"
    REVIEW_6_MONTHS = "review_6_months"
    REVIEW_3_MONTHS = "review_3_months"
    REVIEW_1_MONTH = "review_1_month"
    RECALCULATION_TRIGGER = "recalculation_trigger"
    VALIDATION_EXPIRY = "validation_expiry"
    PROGRESS_ALERT = "progress_alert"


class TargetStatus(str, Enum):
    """SBTi company-level target commitment status."""

    COMMITTED = "committed"
    TARGETS_SET = "targets_set"
    VALIDATED = "validated"
    REMOVED = "removed"


class PathwayAlignment(str, Enum):
    """Temperature pathway alignment for SBTi targets."""

    ONE_POINT_FIVE_C = "1.5c"
    WELL_BELOW_2C = "well_below_2c"
    TWO_C = "2c"
    NOT_ALIGNED = "not_aligned"


class CriterionResult(str, Enum):
    """Result of evaluating a single SBTi validation criterion."""

    PASS = "pass"
    FAIL = "fail"
    NOT_APPLICABLE = "not_applicable"
    INSUFFICIENT_DATA = "insufficient_data"


class DataQualityTier(str, Enum):
    """Data quality tiers for emissions data inputs."""

    MEASURED = "measured"
    CALCULATED = "calculated"
    ESTIMATED = "estimated"
    PROXY = "proxy"
    DEFAULT = "default"


class EffortLevel(str, Enum):
    """Effort levels for gap remediation estimates."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


class TimeHorizon(str, Enum):
    """SBTi time horizons for target classification."""

    SHORT_TERM = "short_term"
    MEDIUM_TERM = "medium_term"
    LONG_TERM = "long_term"


class IntensityMetric(str, Enum):
    """Intensity metrics for SDA sector-specific pathways."""

    TCO2E_PER_MWH = "tco2e_per_mwh"
    TCO2E_PER_TJ = "tco2e_per_tj"
    GCO2E_PER_PKM = "gco2e_per_pkm"
    KGCO2E_PER_M2 = "kgco2e_per_m2"
    TCO2E_PER_T_CEMENT = "tco2e_per_t_cement"
    TCO2E_PER_T_STEEL = "tco2e_per_t_steel"
    TCO2E_PER_T_ALUMINIUM = "tco2e_per_t_aluminium"
    TCO2E_PER_T_PRODUCT = "tco2e_per_t_product"
    GCO2E_PER_RPK = "gco2e_per_rpk"
    TCO2E_PER_M_REVENUE = "tco2e_per_m_revenue"
    CUSTOM = "custom"


class Scope3Category(str, Enum):
    """GHG Protocol Scope 3 categories (1-15)."""

    CAT_1 = "cat_1"
    CAT_2 = "cat_2"
    CAT_3 = "cat_3"
    CAT_4 = "cat_4"
    CAT_5 = "cat_5"
    CAT_6 = "cat_6"
    CAT_7 = "cat_7"
    CAT_8 = "cat_8"
    CAT_9 = "cat_9"
    CAT_10 = "cat_10"
    CAT_11 = "cat_11"
    CAT_12 = "cat_12"
    CAT_13 = "cat_13"
    CAT_14 = "cat_14"
    CAT_15 = "cat_15"


class FLAGCommodity(str, Enum):
    """FLAG commodity categories per SBTi FLAG Guidance."""

    CATTLE = "cattle"
    SOY = "soy"
    PALM_OIL = "palm_oil"
    TIMBER = "timber"
    COCOA = "cocoa"
    COFFEE = "coffee"
    RUBBER = "rubber"
    RICE = "rice"
    MAIZE = "maize"
    WHEAT = "wheat"
    SUGAR_CANE = "sugar_cane"


class FLAGPathwayType(str, Enum):
    """FLAG pathway types for deforestation and land-use targets."""

    COMMODITY = "commodity"
    SECTOR = "sector"
    CUSTOM = "custom"


class FrameworkType(str, Enum):
    """External regulatory/disclosure frameworks for cross-mapping."""

    CDP = "cdp"
    TCFD = "tcfd"
    CSRD = "csrd"
    GHG_PROTOCOL = "ghg_protocol"
    ISO14064 = "iso14064"
    SB253 = "sb253"


class ReviewStatus(str, Enum):
    """Five-year review lifecycle status."""

    UPCOMING = "upcoming"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    OVERDUE = "overdue"


class VerificationAssurance(str, Enum):
    """Third-party verification assurance levels."""

    NOT_VERIFIED = "not_verified"
    LIMITED = "limited"
    REASONABLE = "reasonable"


class ValidationCriterion(str, Enum):
    """SBTi validation criterion identifiers."""

    C1 = "c1"
    C2 = "c2"
    C3 = "c3"
    C4 = "c4"
    C5 = "c5"
    C6 = "c6"
    C7 = "c7"
    C8 = "c8"


class NetZeroCriterion(str, Enum):
    """SBTi Net-Zero criterion identifiers."""

    NZ_C1 = "nz_c1"
    NZ_C2 = "nz_c2"
    NZ_C3 = "nz_c3"
    NZ_C4 = "nz_c4"
    NZ_C5 = "nz_c5"


# ---------------------------------------------------------------------------
# SBTi Target Thresholds
# ---------------------------------------------------------------------------

ACA_ANNUAL_RATES: Dict[str, Decimal] = {
    "1.5c": Decimal("4.2"),
    "well_below_2c": Decimal("2.5"),
    "2c": Decimal("1.23"),
}

BASE_YEAR_MINIMUM: int = 2015
FLAG_TRIGGER_THRESHOLD: Decimal = Decimal("20.0")
SCOPE1_2_COVERAGE_THRESHOLD: Decimal = Decimal("95.0")
SCOPE3_NEAR_TERM_COVERAGE: Decimal = Decimal("67.0")
SCOPE3_TRIGGER_THRESHOLD: Decimal = Decimal("40.0")


# ---------------------------------------------------------------------------
# SBTi Minimum Ambition Thresholds
# ---------------------------------------------------------------------------

SBTI_MINIMUM_AMBITION: Dict[str, Dict[str, Any]] = {
    "near_term": {
        "scope_1_2_annual_reduction": Decimal("4.2"),
        "scope_3_annual_reduction": Decimal("2.5"),
        "minimum_coverage_scope_1_2": Decimal("95.0"),
        "minimum_coverage_scope_3": Decimal("67.0"),
        "timeframe_min_years": 5,
        "timeframe_max_years": 10,
        "description": (
            "Near-term targets must reduce Scope 1+2 at minimum 4.2% per year "
            "(1.5C-aligned) and Scope 3 at minimum 2.5% per year."
        ),
    },
    "long_term": {
        "scope_1_2_reduction_by_2050": Decimal("90.0"),
        "scope_3_reduction_by_2050": Decimal("90.0"),
        "minimum_coverage_scope_1_2": Decimal("95.0"),
        "minimum_coverage_scope_3": Decimal("90.0"),
        "target_year": 2050,
        "description": (
            "Long-term targets must reduce overall emissions by at least 90% "
            "from base year by no later than 2050."
        ),
    },
    "net_zero": {
        "residual_emissions_max_pct": Decimal("10.0"),
        "neutralization_required": True,
        "target_year": 2050,
        "description": (
            "Net-zero targets require at least 90% absolute reduction by 2050, "
            "with neutralization of residual emissions."
        ),
    },
}


# ---------------------------------------------------------------------------
# Sector Decarbonization Pathways
# ---------------------------------------------------------------------------

SECTOR_PATHWAYS: Dict[str, Dict[str, Any]] = {
    "power": {
        "annual_reduction_1_5c": Decimal("7.0"),
        "annual_reduction_wb2c": Decimal("4.0"),
        "intensity_metric": "tCO2e/MWh",
        "base_year_intensity_2020": Decimal("0.45"),
        "target_intensity_2030": Decimal("0.14"),
        "target_intensity_2050": Decimal("0.0"),
        "sda_available": True,
    },
    "oil_gas": {
        "annual_reduction_1_5c": Decimal("6.0"),
        "annual_reduction_wb2c": Decimal("3.5"),
        "intensity_metric": "tCO2e/TJ",
        "base_year_intensity_2020": Decimal("58.0"),
        "target_intensity_2030": Decimal("38.0"),
        "target_intensity_2050": Decimal("5.0"),
        "sda_available": True,
    },
    "transport": {
        "annual_reduction_1_5c": Decimal("5.0"),
        "annual_reduction_wb2c": Decimal("3.0"),
        "intensity_metric": "gCO2e/pkm",
        "base_year_intensity_2020": Decimal("90.0"),
        "target_intensity_2030": Decimal("60.0"),
        "target_intensity_2050": Decimal("5.0"),
        "sda_available": True,
    },
    "buildings": {
        "annual_reduction_1_5c": Decimal("4.5"),
        "annual_reduction_wb2c": Decimal("2.5"),
        "intensity_metric": "kgCO2e/m2",
        "base_year_intensity_2020": Decimal("28.0"),
        "target_intensity_2030": Decimal("18.0"),
        "target_intensity_2050": Decimal("2.0"),
        "sda_available": True,
    },
    "cement": {
        "annual_reduction_1_5c": Decimal("3.5"),
        "annual_reduction_wb2c": Decimal("2.0"),
        "intensity_metric": "tCO2e/t_cement",
        "base_year_intensity_2020": Decimal("0.60"),
        "target_intensity_2030": Decimal("0.47"),
        "target_intensity_2050": Decimal("0.12"),
        "sda_available": True,
    },
    "steel": {
        "annual_reduction_1_5c": Decimal("3.8"),
        "annual_reduction_wb2c": Decimal("2.2"),
        "intensity_metric": "tCO2e/t_steel",
        "base_year_intensity_2020": Decimal("1.80"),
        "target_intensity_2030": Decimal("1.35"),
        "target_intensity_2050": Decimal("0.22"),
        "sda_available": True,
    },
    "aluminium": {
        "annual_reduction_1_5c": Decimal("3.5"),
        "annual_reduction_wb2c": Decimal("2.0"),
        "intensity_metric": "tCO2e/t_aluminium",
        "base_year_intensity_2020": Decimal("8.60"),
        "target_intensity_2030": Decimal("6.50"),
        "target_intensity_2050": Decimal("1.20"),
        "sda_available": True,
    },
    "chemicals": {
        "annual_reduction_1_5c": Decimal("4.0"),
        "annual_reduction_wb2c": Decimal("2.5"),
        "intensity_metric": "tCO2e/t_product",
        "base_year_intensity_2020": Decimal("1.20"),
        "target_intensity_2030": Decimal("0.85"),
        "target_intensity_2050": Decimal("0.10"),
        "sda_available": True,
    },
    "aviation": {
        "annual_reduction_1_5c": Decimal("3.0"),
        "annual_reduction_wb2c": Decimal("1.8"),
        "intensity_metric": "gCO2e/RPK",
        "base_year_intensity_2020": Decimal("88.0"),
        "target_intensity_2030": Decimal("68.0"),
        "target_intensity_2050": Decimal("12.0"),
        "sda_available": True,
    },
    "pulp_paper": {
        "annual_reduction_1_5c": Decimal("3.2"),
        "annual_reduction_wb2c": Decimal("1.9"),
        "intensity_metric": "tCO2e/t_product",
        "base_year_intensity_2020": Decimal("0.52"),
        "target_intensity_2030": Decimal("0.38"),
        "target_intensity_2050": Decimal("0.07"),
        "sda_available": True,
    },
    "maritime": {
        "annual_reduction_1_5c": Decimal("3.5"),
        "annual_reduction_wb2c": Decimal("2.0"),
        "intensity_metric": "gCO2e/t_nm",
        "base_year_intensity_2020": Decimal("11.0"),
        "target_intensity_2030": Decimal("8.0"),
        "target_intensity_2050": Decimal("1.5"),
        "sda_available": True,
    },
    "general": {
        "annual_reduction_1_5c": Decimal("4.2"),
        "annual_reduction_wb2c": Decimal("2.5"),
        "intensity_metric": "tCO2e/M_revenue",
        "base_year_intensity_2020": Decimal("0.0"),
        "target_intensity_2030": Decimal("0.0"),
        "target_intensity_2050": Decimal("0.0"),
        "sda_available": False,
    },
}


# ---------------------------------------------------------------------------
# PCAF Data Quality Scores (numeric)
# ---------------------------------------------------------------------------

PCAF_DQ_SCORES: Dict[PCAFDataQuality, int] = {
    PCAFDataQuality.DQ_1: 1,
    PCAFDataQuality.DQ_2: 2,
    PCAFDataQuality.DQ_3: 3,
    PCAFDataQuality.DQ_4: 4,
    PCAFDataQuality.DQ_5: 5,
}

PCAF_DQ_DESCRIPTIONS: Dict[PCAFDataQuality, str] = {
    PCAFDataQuality.DQ_1: "Verified emissions from investee (audited, reported)",
    PCAFDataQuality.DQ_2: "Unverified emissions from investee (reported, unaudited)",
    PCAFDataQuality.DQ_3: "Estimated using physical activity data (energy use, production)",
    PCAFDataQuality.DQ_4: "Estimated using economic activity data (revenue-based EEIO)",
    PCAFDataQuality.DQ_5: "Estimated using sector-average data or asset-class proxies",
}


# ---------------------------------------------------------------------------
# FI Portfolio Coverage 2040 Pathway
# ---------------------------------------------------------------------------

FI_COVERAGE_PATHWAY: Dict[int, Decimal] = {
    2025: Decimal("25.0"),
    2026: Decimal("30.0"),
    2027: Decimal("35.0"),
    2028: Decimal("40.0"),
    2029: Decimal("45.0"),
    2030: Decimal("50.0"),
    2031: Decimal("53.3"),
    2032: Decimal("56.7"),
    2033: Decimal("60.0"),
    2034: Decimal("63.3"),
    2035: Decimal("66.7"),
    2036: Decimal("70.0"),
    2037: Decimal("76.7"),
    2038: Decimal("83.3"),
    2039: Decimal("90.0"),
    2040: Decimal("100.0"),
}


# ---------------------------------------------------------------------------
# Attribution Method Hierarchy
# ---------------------------------------------------------------------------

ATTRIBUTION_METHODS: Dict[str, Dict[str, Any]] = {
    "evic": {
        "name": "Enterprise Value Including Cash",
        "formula": "financed_emissions = (outstanding / EVIC) * investee_emissions",
        "preferred_for": ["listed_equity", "corporate_bond"],
        "priority": 1,
    },
    "revenue": {
        "name": "Revenue Attribution",
        "formula": "financed_emissions = (outstanding / revenue) * investee_emissions",
        "preferred_for": ["business_loan", "project_finance"],
        "priority": 2,
    },
    "balance_sheet": {
        "name": "Balance Sheet Total Assets",
        "formula": "financed_emissions = (outstanding / total_assets) * investee_emissions",
        "preferred_for": ["business_loan"],
        "priority": 3,
    },
    "floor_area": {
        "name": "Floor Area Attribution",
        "formula": "financed_emissions = (LTV * property_value / property_area) * ef * area",
        "preferred_for": ["commercial_real_estate", "mortgage"],
        "priority": 4,
    },
}


# ---------------------------------------------------------------------------
# Cross-Framework Mapping References
# ---------------------------------------------------------------------------

FRAMEWORK_MAPPING_REFS: Dict[str, Dict[str, str]] = {
    "cdp": {
        "section": "C4 - Targets and Performance",
        "questions": "C4.1, C4.1a, C4.1b, C4.2, C4.2a, C4.2b",
        "description": "CDP Climate Change questionnaire target-setting questions",
    },
    "tcfd": {
        "section": "Metrics & Targets (MT-c)",
        "questions": "MT-c recommended disclosure",
        "description": "TCFD targets used to manage climate risks and performance",
    },
    "csrd": {
        "section": "ESRS E1 - Climate Change",
        "questions": "E1-4 (Targets), E1-6 (GHG reduction targets), E1-7 (Removals/credits)",
        "description": "CSRD/ESRS E1 climate transition plan targets",
    },
    "ghg_protocol": {
        "section": "Corporate Standard + Scope 3 Standard",
        "questions": "Chapters 3-8 (Scope 1/2), GHG Protocol Scope 3 (15 categories)",
        "description": "GHG Protocol base inventory for SBTi target boundary",
    },
    "iso14064": {
        "section": "ISO 14064-1:2018 and ISO 14064-3:2019",
        "questions": "Clause 9 (Verification)",
        "description": "ISO verification linkage for SBTi validation",
    },
    "sb253": {
        "section": "California SB 253 Climate Corporate Data Accountability Act",
        "questions": "Scope 1, 2, and 3 reporting for entities > $1B revenue",
        "description": "SB 253 mandatory GHG reporting alignment",
    },
}


# ---------------------------------------------------------------------------
# SBTi Readiness Dimensions (for gap analysis)
# ---------------------------------------------------------------------------

READINESS_DIMENSIONS: Dict[str, List[Dict[str, str]]] = {
    "data": [
        {"id": "data_01", "name": "Scope 1 Inventory", "description": "Complete Scope 1 emissions inventory"},
        {"id": "data_02", "name": "Scope 2 Inventory", "description": "Scope 2 location-based and market-based"},
        {"id": "data_03", "name": "Scope 3 Screening", "description": "Scope 3 screening of all 15 categories"},
        {"id": "data_04", "name": "Scope 3 Quantification", "description": "Material Scope 3 categories quantified"},
        {"id": "data_05", "name": "Base Year Data", "description": "Verified base year emissions data"},
        {"id": "data_06", "name": "Activity Data Quality", "description": "Primary vs secondary data coverage"},
        {"id": "data_07", "name": "Emission Factor Quality", "description": "Region/sector-specific emission factors"},
        {"id": "data_08", "name": "Temporal Consistency", "description": "Consistent multi-year data tracking"},
    ],
    "ambition": [
        {"id": "amb_01", "name": "Near-Term Rate", "description": "Annual reduction rate vs SBTi minimum 4.2%"},
        {"id": "amb_02", "name": "S1+S2 Coverage", "description": "Scope 1+2 coverage >= 95%"},
        {"id": "amb_03", "name": "S3 Coverage", "description": "Scope 3 coverage >= 67% of total S3"},
        {"id": "amb_04", "name": "Long-Term Target", "description": "90% reduction by 2050 or sooner"},
        {"id": "amb_05", "name": "Net-Zero Commitment", "description": "Net-zero by 2050 with neutralization plan"},
        {"id": "amb_06", "name": "Sector Alignment", "description": "Intensity aligned with sector pathway"},
    ],
    "process": [
        {"id": "proc_01", "name": "Board Governance", "description": "Board oversight of climate targets"},
        {"id": "proc_02", "name": "Internal Processes", "description": "Formal target tracking and review processes"},
        {"id": "proc_03", "name": "Verification", "description": "Third-party verification of emissions inventory"},
        {"id": "proc_04", "name": "Recalculation Policy", "description": "Documented base year recalculation policy"},
        {"id": "proc_05", "name": "Progress Reporting", "description": "Annual target progress reporting"},
        {"id": "proc_06", "name": "Transition Plan", "description": "Documented climate transition plan"},
        {"id": "proc_07", "name": "Capital Alignment", "description": "Capex aligned with emission reduction"},
        {"id": "proc_08", "name": "Supplier Engagement", "description": "Scope 3 supplier engagement program"},
    ],
}


# ---------------------------------------------------------------------------
# Sector Peer Readiness Benchmarks (score 0-100%)
# ---------------------------------------------------------------------------

SECTOR_BENCHMARKS: Dict[str, Dict[str, Any]] = {
    "power": {
        "data": 78.0, "ambition": 72.0, "process": 70.0,
        "overall": 73.0, "sample_size": 120,
        "top_quartile": 88.0, "median": 72.0, "bottom_quartile": 55.0,
    },
    "oil_gas": {
        "data": 75.0, "ambition": 55.0, "process": 62.0,
        "overall": 64.0, "sample_size": 80,
        "top_quartile": 80.0, "median": 62.0, "bottom_quartile": 45.0,
    },
    "transport": {
        "data": 65.0, "ambition": 58.0, "process": 55.0,
        "overall": 59.0, "sample_size": 90,
        "top_quartile": 75.0, "median": 58.0, "bottom_quartile": 42.0,
    },
    "buildings": {
        "data": 62.0, "ambition": 55.0, "process": 50.0,
        "overall": 56.0, "sample_size": 70,
        "top_quartile": 72.0, "median": 55.0, "bottom_quartile": 40.0,
    },
    "cement": {
        "data": 72.0, "ambition": 60.0, "process": 58.0,
        "overall": 63.0, "sample_size": 35,
        "top_quartile": 78.0, "median": 62.0, "bottom_quartile": 48.0,
    },
    "steel": {
        "data": 70.0, "ambition": 58.0, "process": 55.0,
        "overall": 61.0, "sample_size": 40,
        "top_quartile": 76.0, "median": 60.0, "bottom_quartile": 45.0,
    },
    "financial_institutions": {
        "data": 68.0, "ambition": 62.0, "process": 65.0,
        "overall": 65.0, "sample_size": 110,
        "top_quartile": 82.0, "median": 64.0, "bottom_quartile": 48.0,
    },
    "general": {
        "data": 55.0, "ambition": 48.0, "process": 45.0,
        "overall": 49.0, "sample_size": 500,
        "top_quartile": 68.0, "median": 48.0, "bottom_quartile": 32.0,
    },
}


# ---------------------------------------------------------------------------
# SBTi Near-Term Validation Criteria (C1-C8)
# ---------------------------------------------------------------------------

NEAR_TERM_CRITERIA: Dict[str, Dict[str, Any]] = {
    "C1": {
        "name": "Boundary Coverage",
        "description": (
            "Target boundary must cover 95% of company-wide Scope 1 and Scope 2 "
            "emissions, using either location-based or market-based method."
        ),
        "scope": ["scope_1", "scope_2"],
        "threshold": {"coverage_pct": Decimal("95.0")},
        "required": True,
    },
    "C2": {
        "name": "Timeframe",
        "description": (
            "Near-term target year must be a minimum of 5 and a maximum of 10 "
            "years from the date of target submission."
        ),
        "scope": ["scope_1", "scope_2", "scope_3"],
        "threshold": {"min_years": 5, "max_years": 10},
        "required": True,
    },
    "C3": {
        "name": "Scope 1+2 Ambition Level",
        "description": (
            "Scope 1+2 near-term targets must be aligned with a minimum 1.5C "
            "pathway, requiring at least 4.2% linear annual reduction."
        ),
        "scope": ["scope_1_2"],
        "threshold": {"min_annual_reduction_pct": Decimal("4.2")},
        "required": True,
    },
    "C4": {
        "name": "Scope 3 Screening",
        "description": (
            "Companies whose Scope 3 emissions represent 40% or more of total "
            "Scope 1+2+3 emissions must set a Scope 3 near-term target."
        ),
        "scope": ["scope_3"],
        "threshold": {"trigger_pct_of_total": Decimal("40.0")},
        "required": True,
    },
    "C5": {
        "name": "Scope 3 Coverage",
        "description": (
            "Near-term Scope 3 targets must cover at least 67% of total Scope 3 "
            "emissions (collectively across covered categories)."
        ),
        "scope": ["scope_3"],
        "threshold": {"coverage_pct": Decimal("67.0")},
        "required": True,
    },
    "C6": {
        "name": "Scope 3 Ambition Level",
        "description": (
            "Scope 3 near-term targets must be at minimum well-below 2C aligned, "
            "requiring at least 2.5% linear annual reduction."
        ),
        "scope": ["scope_3"],
        "threshold": {"min_annual_reduction_pct": Decimal("2.5")},
        "required": True,
    },
    "C7": {
        "name": "Base Year Recency",
        "description": (
            "The base year must be 2015 or more recent. Companies should use the "
            "most recent year for which credible data is available."
        ),
        "scope": ["scope_1", "scope_2", "scope_3"],
        "threshold": {"min_base_year": 2015},
        "required": True,
    },
    "C8": {
        "name": "Recalculation Policy",
        "description": (
            "Companies must have a documented base-year recalculation policy. "
            "Recalculation is triggered when structural changes exceed 5% of "
            "base year emissions."
        ),
        "scope": ["scope_1", "scope_2", "scope_3"],
        "threshold": {"recalculation_threshold_pct": Decimal("5.0")},
        "required": True,
    },
}


# ---------------------------------------------------------------------------
# SBTi Net-Zero Validation Criteria (NZ-C1 through NZ-C5)
# ---------------------------------------------------------------------------

NET_ZERO_CRITERIA: Dict[str, Dict[str, Any]] = {
    "NZ-C1": {
        "name": "Near-Term Target Prerequisite",
        "description": (
            "Companies must have a validated near-term SBTi target as a "
            "prerequisite for net-zero commitment. The near-term target must "
            "be 1.5C-aligned for Scope 1+2."
        ),
        "prerequisite": True,
        "required": True,
    },
    "NZ-C2": {
        "name": "Long-Term Reduction Target",
        "description": (
            "Long-term targets must reduce Scope 1, 2, and 3 emissions by at "
            "least 90% from the base year by no later than 2050. This must "
            "cover at least 95% of Scope 1+2 and 90% of Scope 3."
        ),
        "threshold": {
            "min_reduction_pct": Decimal("90.0"),
            "scope_1_2_coverage_pct": Decimal("95.0"),
            "scope_3_coverage_pct": Decimal("90.0"),
            "latest_target_year": 2050,
        },
        "required": True,
    },
    "NZ-C3": {
        "name": "Residual Emissions Neutralization",
        "description": (
            "Residual emissions (up to 10% of base year) must be neutralized "
            "through permanent carbon removal and storage. Offsets do not count "
            "toward net-zero; only high-quality removals are accepted."
        ),
        "threshold": {
            "max_residual_pct": Decimal("10.0"),
            "neutralization_required": True,
        },
        "required": True,
    },
    "NZ-C4": {
        "name": "Beyond Value Chain Mitigation",
        "description": (
            "Companies should invest in beyond value chain mitigation (BVCM) "
            "to achieve climate finance contributions during the transition. "
            "This does not replace in-value-chain abatement."
        ),
        "threshold": {},
        "required": False,
    },
    "NZ-C5": {
        "name": "No Use of Offsets for Target Achievement",
        "description": (
            "Companies must not use carbon credits or offsets to count toward "
            "their near-term or long-term target achievement. Carbon removal "
            "credits may only be used for neutralization of residual emissions "
            "at the net-zero target year."
        ),
        "threshold": {},
        "required": True,
    },
}


# ---------------------------------------------------------------------------
# Scope 3 Category Definitions (GHG Protocol 15 Categories)
# ---------------------------------------------------------------------------

SCOPE3_CATEGORY_DEFINITIONS: Dict[int, Dict[str, Any]] = {
    1: {
        "name": "Purchased Goods and Services",
        "description": (
            "Extraction, production, and transportation of goods and services "
            "purchased or acquired by the reporting company in the reporting year."
        ),
        "mrv_agent": "MRV-014",
        "upstream": True,
        "typically_material": True,
    },
    2: {
        "name": "Capital Goods",
        "description": (
            "Extraction, production, and transportation of capital goods purchased "
            "or acquired by the reporting company in the reporting year."
        ),
        "mrv_agent": "MRV-015",
        "upstream": True,
        "typically_material": False,
    },
    3: {
        "name": "Fuel- and Energy-Related Activities",
        "description": (
            "Extraction, production, and transportation of fuels and energy "
            "consumed by the reporting company in the reporting year, not already "
            "accounted for in Scope 1 or Scope 2."
        ),
        "mrv_agent": "MRV-016",
        "upstream": True,
        "typically_material": True,
    },
    4: {
        "name": "Upstream Transportation and Distribution",
        "description": (
            "Transportation and distribution of products purchased by the reporting "
            "company in the reporting year (in vehicles not owned by the company)."
        ),
        "mrv_agent": "MRV-017",
        "upstream": True,
        "typically_material": True,
    },
    5: {
        "name": "Waste Generated in Operations",
        "description": (
            "Disposal and treatment of waste generated in the reporting company's "
            "operations in the reporting year (in facilities not owned by the company)."
        ),
        "mrv_agent": "MRV-018",
        "upstream": True,
        "typically_material": False,
    },
    6: {
        "name": "Business Travel",
        "description": (
            "Transportation of employees for business-related activities in "
            "vehicles not owned by the reporting company."
        ),
        "mrv_agent": "MRV-019",
        "upstream": True,
        "typically_material": False,
    },
    7: {
        "name": "Employee Commuting",
        "description": (
            "Transportation of employees between their homes and worksites, "
            "including telework emissions."
        ),
        "mrv_agent": "MRV-020",
        "upstream": True,
        "typically_material": False,
    },
    8: {
        "name": "Upstream Leased Assets",
        "description": (
            "Operation of assets leased by the reporting company (lessee) "
            "not included in Scope 1 and Scope 2."
        ),
        "mrv_agent": "MRV-021",
        "upstream": True,
        "typically_material": False,
    },
    9: {
        "name": "Downstream Transportation and Distribution",
        "description": (
            "Transportation and distribution of products sold by the reporting "
            "company in the reporting year between the company's operations and "
            "the end consumer."
        ),
        "mrv_agent": "MRV-022",
        "upstream": False,
        "typically_material": True,
    },
    10: {
        "name": "Processing of Sold Products",
        "description": (
            "Processing of intermediate products sold by the reporting company "
            "in the reporting year by third parties."
        ),
        "mrv_agent": "MRV-023",
        "upstream": False,
        "typically_material": False,
    },
    11: {
        "name": "Use of Sold Products",
        "description": (
            "End use of goods and services sold by the reporting company "
            "in the reporting year."
        ),
        "mrv_agent": "MRV-024",
        "upstream": False,
        "typically_material": True,
    },
    12: {
        "name": "End-of-Life Treatment of Sold Products",
        "description": (
            "Waste disposal and treatment of products sold by the reporting "
            "company at the end of their life."
        ),
        "mrv_agent": "MRV-025",
        "upstream": False,
        "typically_material": False,
    },
    13: {
        "name": "Downstream Leased Assets",
        "description": (
            "Operation of assets owned by the reporting company (lessor) "
            "and leased to other entities."
        ),
        "mrv_agent": "MRV-026",
        "upstream": False,
        "typically_material": False,
    },
    14: {
        "name": "Franchises",
        "description": (
            "Operation of franchises not included in Scope 1 and Scope 2, "
            "reported by the franchisor."
        ),
        "mrv_agent": "MRV-027",
        "upstream": False,
        "typically_material": False,
    },
    15: {
        "name": "Investments",
        "description": (
            "Operation of investments not included in Scope 1 and Scope 2, "
            "including equity and debt investments, project finance, etc."
        ),
        "mrv_agent": "MRV-028",
        "upstream": False,
        "typically_material": True,
    },
}


# ---------------------------------------------------------------------------
# FLAG Commodity Pathways (SBTi FLAG Guidance v1.0)
# ---------------------------------------------------------------------------

FLAG_COMMODITY_PATHWAYS: Dict[str, Dict[str, Any]] = {
    "cattle": {
        "base_intensity_2020": Decimal("14.5"),
        "target_intensity_2030": Decimal("11.1"),
        "target_intensity_2050": Decimal("5.5"),
        "annual_reduction_rate": Decimal("3.03"),
        "intensity_unit": "tCO2e/t_protein",
        "deforestation_commitment_required": True,
        "description": "Cattle (beef and dairy) commodity pathway",
    },
    "soy": {
        "base_intensity_2020": Decimal("2.2"),
        "target_intensity_2030": Decimal("1.6"),
        "target_intensity_2050": Decimal("0.8"),
        "annual_reduction_rate": Decimal("3.03"),
        "intensity_unit": "tCO2e/t_commodity",
        "deforestation_commitment_required": True,
        "description": "Soy commodity pathway (includes LUC)",
    },
    "palm_oil": {
        "base_intensity_2020": Decimal("4.8"),
        "target_intensity_2030": Decimal("3.5"),
        "target_intensity_2050": Decimal("1.8"),
        "annual_reduction_rate": Decimal("3.03"),
        "intensity_unit": "tCO2e/t_commodity",
        "deforestation_commitment_required": True,
        "description": "Palm oil commodity pathway (includes peat emissions)",
    },
    "timber": {
        "base_intensity_2020": Decimal("0.6"),
        "target_intensity_2030": Decimal("0.4"),
        "target_intensity_2050": Decimal("0.2"),
        "annual_reduction_rate": Decimal("3.03"),
        "intensity_unit": "tCO2e/m3_roundwood",
        "deforestation_commitment_required": True,
        "description": "Timber and forestry products pathway",
    },
    "cocoa": {
        "base_intensity_2020": Decimal("3.8"),
        "target_intensity_2030": Decimal("2.8"),
        "target_intensity_2050": Decimal("1.4"),
        "annual_reduction_rate": Decimal("3.03"),
        "intensity_unit": "tCO2e/t_commodity",
        "deforestation_commitment_required": True,
        "description": "Cocoa commodity pathway",
    },
    "coffee": {
        "base_intensity_2020": Decimal("5.2"),
        "target_intensity_2030": Decimal("3.8"),
        "target_intensity_2050": Decimal("1.9"),
        "annual_reduction_rate": Decimal("3.03"),
        "intensity_unit": "tCO2e/t_commodity",
        "deforestation_commitment_required": True,
        "description": "Coffee commodity pathway",
    },
    "rubber": {
        "base_intensity_2020": Decimal("2.5"),
        "target_intensity_2030": Decimal("1.8"),
        "target_intensity_2050": Decimal("0.9"),
        "annual_reduction_rate": Decimal("3.03"),
        "intensity_unit": "tCO2e/t_commodity",
        "deforestation_commitment_required": True,
        "description": "Natural rubber commodity pathway",
    },
    "rice": {
        "base_intensity_2020": Decimal("3.4"),
        "target_intensity_2030": Decimal("2.5"),
        "target_intensity_2050": Decimal("1.3"),
        "annual_reduction_rate": Decimal("3.03"),
        "intensity_unit": "tCO2e/t_commodity",
        "deforestation_commitment_required": False,
        "description": "Rice paddy (includes CH4 from flooded paddies)",
    },
    "maize": {
        "base_intensity_2020": Decimal("0.9"),
        "target_intensity_2030": Decimal("0.7"),
        "target_intensity_2050": Decimal("0.3"),
        "annual_reduction_rate": Decimal("3.03"),
        "intensity_unit": "tCO2e/t_commodity",
        "deforestation_commitment_required": False,
        "description": "Maize / corn commodity pathway",
    },
    "wheat": {
        "base_intensity_2020": Decimal("0.7"),
        "target_intensity_2030": Decimal("0.5"),
        "target_intensity_2050": Decimal("0.25"),
        "annual_reduction_rate": Decimal("3.03"),
        "intensity_unit": "tCO2e/t_commodity",
        "deforestation_commitment_required": False,
        "description": "Wheat commodity pathway",
    },
    "sugar_cane": {
        "base_intensity_2020": Decimal("0.4"),
        "target_intensity_2030": Decimal("0.3"),
        "target_intensity_2050": Decimal("0.15"),
        "annual_reduction_rate": Decimal("3.03"),
        "intensity_unit": "tCO2e/t_commodity",
        "deforestation_commitment_required": False,
        "description": "Sugar cane commodity pathway",
    },
}


# ---------------------------------------------------------------------------
# MRV Agent to SBTi Scope Mapping
# ---------------------------------------------------------------------------

MRV_AGENT_TO_SBTI_SCOPE: Dict[str, Dict[str, Any]] = {
    # Scope 1 agents
    "MRV-001": {
        "scope": "scope_1", "name": "Stationary Combustion",
        "sbti_relevance": "Direct S1 emissions from fuel combustion",
    },
    "MRV-002": {
        "scope": "scope_1", "name": "Refrigerants & F-Gas",
        "sbti_relevance": "Direct S1 emissions from refrigerant leaks",
    },
    "MRV-003": {
        "scope": "scope_1", "name": "Mobile Combustion",
        "sbti_relevance": "Direct S1 fleet/vehicle emissions",
    },
    "MRV-004": {
        "scope": "scope_1", "name": "Process Emissions",
        "sbti_relevance": "Direct S1 industrial process emissions",
    },
    "MRV-005": {
        "scope": "scope_1", "name": "Fugitive Emissions",
        "sbti_relevance": "Direct S1 fugitive emissions (LDAR)",
    },
    "MRV-006": {
        "scope": "scope_1", "name": "Land Use Emissions",
        "sbti_relevance": "Direct S1 land use change (FLAG)",
        "flag_relevant": True,
    },
    "MRV-007": {
        "scope": "scope_1", "name": "Waste Treatment Emissions",
        "sbti_relevance": "Direct S1 on-site waste treatment",
    },
    "MRV-008": {
        "scope": "scope_1", "name": "Agricultural Emissions",
        "sbti_relevance": "Direct S1 agricultural (enteric, manure, soil, rice)",
        "flag_relevant": True,
    },
    # Scope 2 agents
    "MRV-009": {
        "scope": "scope_2", "name": "Scope 2 Location-Based",
        "sbti_relevance": "Grid-average purchased electricity emissions",
    },
    "MRV-010": {
        "scope": "scope_2", "name": "Scope 2 Market-Based",
        "sbti_relevance": "Contractual instrument-based electricity emissions",
    },
    "MRV-011": {
        "scope": "scope_2", "name": "Steam/Heat Purchase",
        "sbti_relevance": "Purchased steam and district heating",
    },
    "MRV-012": {
        "scope": "scope_2", "name": "Cooling Purchase",
        "sbti_relevance": "Purchased cooling and district cooling",
    },
    "MRV-013": {
        "scope": "scope_2", "name": "Dual Reporting Reconciliation",
        "sbti_relevance": "Location vs market-based reconciliation",
    },
    # Scope 3 agents (Cat 1-15)
    "MRV-014": {
        "scope": "scope_3", "name": "Purchased Goods & Services",
        "sbti_relevance": "Category 1 upstream purchased goods",
        "category": 1,
    },
    "MRV-015": {
        "scope": "scope_3", "name": "Capital Goods",
        "sbti_relevance": "Category 2 capital goods (year of acquisition)",
        "category": 2,
    },
    "MRV-016": {
        "scope": "scope_3", "name": "Fuel & Energy Activities",
        "sbti_relevance": "Category 3 WTT fuel, upstream electricity, T&D losses",
        "category": 3,
    },
    "MRV-017": {
        "scope": "scope_3", "name": "Upstream Transportation",
        "sbti_relevance": "Category 4 inbound logistics transportation",
        "category": 4,
    },
    "MRV-018": {
        "scope": "scope_3", "name": "Waste Generated",
        "sbti_relevance": "Category 5 operational waste disposal",
        "category": 5,
    },
    "MRV-019": {
        "scope": "scope_3", "name": "Business Travel",
        "sbti_relevance": "Category 6 employee business travel",
        "category": 6,
    },
    "MRV-020": {
        "scope": "scope_3", "name": "Employee Commuting",
        "sbti_relevance": "Category 7 employee commuting and telework",
        "category": 7,
    },
    "MRV-021": {
        "scope": "scope_3", "name": "Upstream Leased Assets",
        "sbti_relevance": "Category 8 upstream leased assets (lessee)",
        "category": 8,
    },
    "MRV-022": {
        "scope": "scope_3", "name": "Downstream Transportation",
        "sbti_relevance": "Category 9 downstream outbound logistics",
        "category": 9,
    },
    "MRV-023": {
        "scope": "scope_3", "name": "Processing of Sold Products",
        "sbti_relevance": "Category 10 third-party processing",
        "category": 10,
    },
    "MRV-024": {
        "scope": "scope_3", "name": "Use of Sold Products",
        "sbti_relevance": "Category 11 direct and indirect use-phase",
        "category": 11,
    },
    "MRV-025": {
        "scope": "scope_3", "name": "End-of-Life Treatment",
        "sbti_relevance": "Category 12 end-of-life disposal and recycling",
        "category": 12,
    },
    "MRV-026": {
        "scope": "scope_3", "name": "Downstream Leased Assets",
        "sbti_relevance": "Category 13 downstream leased assets (lessor)",
        "category": 13,
    },
    "MRV-027": {
        "scope": "scope_3", "name": "Franchises",
        "sbti_relevance": "Category 14 franchise operations (franchisor)",
        "category": 14,
    },
    "MRV-028": {
        "scope": "scope_3", "name": "Investments",
        "sbti_relevance": "Category 15 financed emissions (PCAF)",
        "category": 15,
    },
    # Cross-cutting
    "MRV-029": {
        "scope": "cross_cutting", "name": "Scope 3 Category Mapper",
        "sbti_relevance": "Maps data to correct S3 categories",
    },
    "MRV-030": {
        "scope": "cross_cutting", "name": "Audit Trail & Lineage",
        "sbti_relevance": "Provenance and audit trail for all calculations",
    },
}


# ---------------------------------------------------------------------------
# Temperature Scoring Default Values
# ---------------------------------------------------------------------------

TEMPERATURE_SCORING_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "default_score_no_target": Decimal("3.2"),
    "score_mapping": {
        "1.5c_aligned_near_term": Decimal("1.5"),
        "well_below_2c_near_term": Decimal("1.75"),
        "2c_aligned_near_term": Decimal("2.0"),
        "committed_no_target": Decimal("2.9"),
        "no_commitment": Decimal("3.2"),
    },
    "weighting": {
        "short_term_weight": Decimal("0.4"),
        "mid_term_weight": Decimal("0.2"),
        "long_term_weight": Decimal("0.4"),
    },
    "scope_weights": {
        "scope_1_2_weight": Decimal("0.6"),
        "scope_3_weight": Decimal("0.4"),
    },
    "methodology_version": "v2.0",
    "description": (
        "SBTi Temperature Rating methodology assigns implied temperature "
        "scores based on target ambition. Default score for companies with "
        "no validated target is 3.2C (aligned with current policies)."
    ),
}


# ---------------------------------------------------------------------------
# Cross-Framework Alignment Detail (SBTi requirements to framework sections)
# ---------------------------------------------------------------------------

CROSS_FRAMEWORK_ALIGNMENT: Dict[str, Dict[str, Any]] = {
    "cdp": {
        "framework_name": "CDP Climate Change Questionnaire",
        "alignment_items": [
            {
                "sbti_ref": "Near-term target (C3)",
                "framework_ref": "C4.1a - Absolute emissions target",
                "status": "fully_aligned",
            },
            {
                "sbti_ref": "Scope 3 screening (C4)",
                "framework_ref": "C6.5 - Scope 3 emissions by category",
                "status": "fully_aligned",
            },
            {
                "sbti_ref": "Emissions inventory",
                "framework_ref": "C6.1 (S1), C6.3 (S2), C6.5 (S3)",
                "status": "fully_aligned",
            },
            {
                "sbti_ref": "Progress tracking",
                "framework_ref": "C4.2 - Progress against targets",
                "status": "fully_aligned",
            },
            {
                "sbti_ref": "SBTi validation status",
                "framework_ref": "C4.2b - Validated SBTi target details",
                "status": "fully_aligned",
            },
        ],
    },
    "tcfd": {
        "framework_name": "TCFD Recommendations",
        "alignment_items": [
            {
                "sbti_ref": "Near-term/long-term targets",
                "framework_ref": "Metrics & Targets (c) - Target disclosure",
                "status": "fully_aligned",
            },
            {
                "sbti_ref": "Emissions inventory",
                "framework_ref": "Metrics & Targets (b) - GHG emissions",
                "status": "fully_aligned",
            },
            {
                "sbti_ref": "Transition plan / pathway",
                "framework_ref": "Strategy (b) - Business strategy impact",
                "status": "partially_aligned",
            },
            {
                "sbti_ref": "Scenario alignment",
                "framework_ref": "Strategy (c) - Scenario analysis resilience",
                "status": "partially_aligned",
            },
        ],
    },
    "csrd": {
        "framework_name": "CSRD / ESRS E1 Climate Change",
        "alignment_items": [
            {
                "sbti_ref": "Near-term/long-term targets",
                "framework_ref": "ESRS E1-4 - Climate targets",
                "status": "fully_aligned",
            },
            {
                "sbti_ref": "Emissions inventory",
                "framework_ref": "ESRS E1-6 - GHG reduction targets / E1-5 Energy",
                "status": "fully_aligned",
            },
            {
                "sbti_ref": "Net-zero / residual neutralization",
                "framework_ref": "ESRS E1-7 - Removals and carbon credits",
                "status": "fully_aligned",
            },
            {
                "sbti_ref": "Transition plan",
                "framework_ref": "ESRS E1-1 - Transition plan for climate change",
                "status": "fully_aligned",
            },
            {
                "sbti_ref": "Progress tracking",
                "framework_ref": "ESRS E1-6 - Performance against targets",
                "status": "fully_aligned",
            },
        ],
    },
    "ghg_protocol": {
        "framework_name": "GHG Protocol Corporate Standard",
        "alignment_items": [
            {
                "sbti_ref": "Emissions boundary (C1)",
                "framework_ref": "Chapters 3-4 - Organizational boundaries",
                "status": "fully_aligned",
            },
            {
                "sbti_ref": "Scope 1/2 inventory",
                "framework_ref": "Chapters 5-8 - Scope 1 and Scope 2 quantification",
                "status": "fully_aligned",
            },
            {
                "sbti_ref": "Scope 3 inventory",
                "framework_ref": "GHG Protocol Scope 3 Standard - 15 categories",
                "status": "fully_aligned",
            },
            {
                "sbti_ref": "Base year recalculation (C8)",
                "framework_ref": "Chapter 5 - Base year recalculation policy",
                "status": "fully_aligned",
            },
        ],
    },
    "iso14064": {
        "framework_name": "ISO 14064-1:2018",
        "alignment_items": [
            {
                "sbti_ref": "Emissions inventory",
                "framework_ref": "Clause 5-7 - Quantification of GHG emissions",
                "status": "fully_aligned",
            },
            {
                "sbti_ref": "Verification (optional for SBTi)",
                "framework_ref": "ISO 14064-3 - Verification and validation",
                "status": "partially_aligned",
            },
        ],
    },
    "sb253": {
        "framework_name": "California SB 253",
        "alignment_items": [
            {
                "sbti_ref": "Scope 1/2/3 inventory",
                "framework_ref": "SB 253 - Mandatory GHG reporting (>$1B revenue)",
                "status": "fully_aligned",
            },
            {
                "sbti_ref": "Third-party assurance",
                "framework_ref": "SB 253 - Required assurance (limited then reasonable)",
                "status": "partially_aligned",
            },
        ],
    },
    "nzba": {
        "framework_name": "Net-Zero Banking Alliance",
        "alignment_items": [
            {
                "sbti_ref": "FI portfolio targets",
                "framework_ref": "NZBA - Intermediate 2030 targets for priority sectors",
                "status": "partially_aligned",
            },
            {
                "sbti_ref": "Financed emissions (PCAF)",
                "framework_ref": "NZBA - Portfolio alignment and decarbonization",
                "status": "fully_aligned",
            },
        ],
    },
    "gfanz": {
        "framework_name": "Glasgow Financial Alliance for Net Zero",
        "alignment_items": [
            {
                "sbti_ref": "FI portfolio coverage",
                "framework_ref": "GFANZ - Transition plan framework",
                "status": "partially_aligned",
            },
            {
                "sbti_ref": "Net-zero by 2050",
                "framework_ref": "GFANZ - 2050 net-zero commitment",
                "status": "fully_aligned",
            },
        ],
    },
}


# ---------------------------------------------------------------------------
# Data Quality Tier Numeric Mapping (1 = best, 5 = worst)
# ---------------------------------------------------------------------------

DATA_QUALITY_SCORES: Dict[DataQualityTier, int] = {
    DataQualityTier.MEASURED: 1,
    DataQualityTier.CALCULATED: 2,
    DataQualityTier.ESTIMATED: 3,
    DataQualityTier.PROXY: 4,
    DataQualityTier.DEFAULT: 5,
}


# ---------------------------------------------------------------------------
# Verification Assurance Numeric Mapping
# ---------------------------------------------------------------------------

VERIFICATION_ASSURANCE_SCORES: Dict[VerificationAssurance, int] = {
    VerificationAssurance.NOT_VERIFIED: 0,
    VerificationAssurance.LIMITED: 1,
    VerificationAssurance.REASONABLE: 2,
}


# ---------------------------------------------------------------------------
# Main Configuration Class
# ---------------------------------------------------------------------------

class SBTiAppConfig(BaseSettings):
    """
    GL-SBTi-APP v1.0 platform configuration.

    All settings can be overridden via environment variables prefixed
    with ``SBTI_APP_``.  For example ``SBTI_APP_RECALCULATION_THRESHOLD_PCT``
    maps to ``recalculation_threshold_pct``.

    Example:
        >>> config = SBTiAppConfig()
        >>> config.app_name
        'GL-SBTi-APP'
        >>> config.recalculation_threshold_pct
        Decimal('5.0')
    """

    model_config = {"env_prefix": "SBTI_APP_"}

    # -- Application Metadata -----------------------------------------------
    app_name: str = Field(
        default="GL-SBTi-APP",
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

    # -- Recalculation Settings ---------------------------------------------
    recalculation_threshold_pct: Decimal = Field(
        default=Decimal("5.0"),
        ge=Decimal("1.0"),
        le=Decimal("50.0"),
        description="Percentage change threshold triggering base year recalculation",
    )
    recalculation_auto_apply: bool = Field(
        default=False,
        description="Automatically apply recalculations to affected targets",
    )

    # -- Five-Year Review Settings ------------------------------------------
    review_window_months: int = Field(
        default=12,
        ge=6,
        le=24,
        description="Months allowed to complete five-year review after trigger",
    )
    notification_lead_months: List[int] = Field(
        default=[12, 6, 3, 1],
        description="Months before review deadline to send notifications",
    )

    # -- Target Defaults ----------------------------------------------------
    default_ambition: AmbitionLevel = Field(
        default=AmbitionLevel.ONE_POINT_FIVE_C,
        description="Default target ambition level",
    )
    default_base_year: int = Field(
        default=2020,
        ge=2015,
        le=2025,
        description="Default base year for new targets",
    )
    default_near_term_years: int = Field(
        default=10,
        ge=5,
        le=10,
        description="Default near-term target timeframe in years",
    )

    # -- FI Settings --------------------------------------------------------
    fi_coverage_target_year: int = Field(
        default=2040,
        ge=2030,
        le=2050,
        description="Year by which FI portfolio coverage must reach 100%",
    )
    fi_engagement_threshold_pct: Decimal = Field(
        default=Decimal("50.0"),
        ge=Decimal("10.0"),
        le=Decimal("100.0"),
        description="Minimum engagement threshold for portfolio companies",
    )

    # -- Reporting Settings -------------------------------------------------
    default_report_format: ReportFormat = Field(
        default=ReportFormat.JSON,
        description="Default report export format",
    )
    report_storage_path: str = Field(
        default="reports/sbti/",
        description="Path prefix for generated reports",
    )
    reporting_year: int = Field(
        default=2025,
        ge=1990,
        le=2100,
        description="Current reporting year",
    )

    # -- MRV Agent Integration ----------------------------------------------
    mrv_agent_base_url: str = Field(
        default="http://localhost:8000/api/v1/mrv",
        description="Base URL for MRV agent API endpoints",
    )
    mrv_agent_timeout_seconds: int = Field(
        default=30,
        ge=5,
        le=300,
        description="Timeout for individual MRV agent calls (seconds)",
    )

    # -- Logging ------------------------------------------------------------
    log_level: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    )
