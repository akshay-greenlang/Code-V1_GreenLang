"""
CDP Climate Change Disclosure Platform Configuration

This module defines all configuration settings, enumerations, scoring weights,
thresholds, module definitions, question types, and constants for the
GL-CDP-APP v1.0 platform implementing the CDP Climate Change Questionnaire
(2025/2026 Integrated Format).

CDP scoring uses 17 categories with dual weightings (management and leadership),
8 scoring bands (D- through A), and 5 mandatory A-level requirements.  This
configuration maps modules and categories to the GreenLang MRV agent layer
for auto-population of Scope 1/2/3 emissions data.

All settings use the CDP_APP_ prefix for environment variable overrides.

Example:
    >>> config = CDPAppConfig()
    >>> config.app_name
    'GL-CDP-APP'
    >>> config.scoring_categories["SC09"]["weight_management"]
    10.0
"""

from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import Field
from pydantic_settings import BaseSettings


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class CDPModule(str, Enum):
    """CDP Climate Change Questionnaire modules (M0 through M13)."""

    M0_INTRODUCTION = "M0"
    M1_GOVERNANCE = "M1"
    M2_POLICIES = "M2"
    M3_RISKS = "M3"
    M4_STRATEGY = "M4"
    M5_TRANSITION = "M5"
    M6_IMPLEMENTATION = "M6"
    M7_CLIMATE_PERFORMANCE = "M7"
    M8_FORESTS = "M8"
    M9_WATER = "M9"
    M10_SUPPLY_CHAIN = "M10"
    M11_ADDITIONAL = "M11"
    M12_FINANCIAL_SERVICES = "M12"
    M13_SIGN_OFF = "M13"


class ScoringLevel(str, Enum):
    """CDP scoring levels from lowest (D-) to highest (A)."""

    D_MINUS = "D-"
    D = "D"
    C_MINUS = "C-"
    C = "C"
    B_MINUS = "B-"
    B = "B"
    A_MINUS = "A-"
    A = "A"


class ScoringBand(str, Enum):
    """CDP scoring bands grouping scoring levels."""

    DISCLOSURE = "Disclosure"
    AWARENESS = "Awareness"
    MANAGEMENT = "Management"
    LEADERSHIP = "Leadership"


class ScoringCategory(str, Enum):
    """17 CDP Climate Change scoring categories."""

    SC01_GOVERNANCE = "SC01"
    SC02_RISK_MANAGEMENT = "SC02"
    SC03_RISK_DISCLOSURE = "SC03"
    SC04_OPPORTUNITY_DISCLOSURE = "SC04"
    SC05_BUSINESS_STRATEGY = "SC05"
    SC06_SCENARIO_ANALYSIS = "SC06"
    SC07_TARGETS = "SC07"
    SC08_EMISSIONS_REDUCTION = "SC08"
    SC09_SCOPE12_EMISSIONS = "SC09"
    SC10_SCOPE3_EMISSIONS = "SC10"
    SC11_ENERGY = "SC11"
    SC12_CARBON_PRICING = "SC12"
    SC13_VALUE_CHAIN = "SC13"
    SC14_PUBLIC_POLICY = "SC14"
    SC15_TRANSITION_PLAN = "SC15"
    SC16_PORTFOLIO_FS = "SC16"
    SC17_FINANCIAL_IMPACT = "SC17"


class QuestionType(str, Enum):
    """Supported CDP question response types."""

    TEXT = "text"
    NUMERIC = "numeric"
    PERCENTAGE = "percentage"
    TABLE = "table"
    MULTI_SELECT = "multi_select"
    SINGLE_SELECT = "single_select"
    YES_NO = "yes_no"
    DATE = "date"
    CURRENCY = "currency"
    FILE_UPLOAD = "file_upload"


class ResponseStatus(str, Enum):
    """Response lifecycle status."""

    NOT_STARTED = "not_started"
    DRAFT = "draft"
    IN_REVIEW = "in_review"
    APPROVED = "approved"
    SUBMITTED = "submitted"
    RETURNED = "returned"


class GapSeverity(str, Enum):
    """Gap severity levels for gap analysis."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class GapLevel(str, Enum):
    """Gap level indicating which scoring threshold is missed."""

    DISCLOSURE = "disclosure"
    AWARENESS = "awareness"
    MANAGEMENT = "management"
    LEADERSHIP = "leadership"


class EffortLevel(str, Enum):
    """Effort estimation for closing a gap."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class VerificationAssurance(str, Enum):
    """Third-party assurance levels."""

    NOT_VERIFIED = "not_verified"
    LIMITED = "limited"
    REASONABLE = "reasonable"


class TransitionTimeframe(str, Enum):
    """Transition plan milestone timeframes."""

    SHORT_TERM = "short_term"     # 1-3 years
    MEDIUM_TERM = "medium_term"   # 3-10 years
    LONG_TERM = "long_term"       # 10-30 years


class SupplierStatus(str, Enum):
    """Supplier engagement status."""

    NOT_INVITED = "not_invited"
    INVITED = "invited"
    IN_PROGRESS = "in_progress"
    SUBMITTED = "submitted"
    SCORED = "scored"
    DECLINED = "declined"


class ReportFormat(str, Enum):
    """Supported report output formats."""

    PDF = "pdf"
    EXCEL = "excel"
    XML = "xml"
    JSON = "json"


class GICSector(str, Enum):
    """GICS sector classification codes."""

    ENERGY = "10"
    MATERIALS = "15"
    INDUSTRIALS = "20"
    CONSUMER_DISCRETIONARY = "25"
    CONSUMER_STAPLES = "30"
    HEALTH_CARE = "35"
    FINANCIALS = "40"
    INFORMATION_TECHNOLOGY = "45"
    COMMUNICATION_SERVICES = "50"
    UTILITIES = "55"
    REAL_ESTATE = "60"


# ---------------------------------------------------------------------------
# Scoring Level Thresholds
# ---------------------------------------------------------------------------

SCORING_LEVEL_THRESHOLDS: Dict[ScoringLevel, Tuple[float, float]] = {
    ScoringLevel.A: (80.0, 100.0),
    ScoringLevel.A_MINUS: (70.0, 79.99),
    ScoringLevel.B: (60.0, 69.99),
    ScoringLevel.B_MINUS: (50.0, 59.99),
    ScoringLevel.C: (40.0, 49.99),
    ScoringLevel.C_MINUS: (30.0, 39.99),
    ScoringLevel.D: (20.0, 29.99),
    ScoringLevel.D_MINUS: (0.0, 19.99),
}

SCORING_LEVEL_BANDS: Dict[ScoringLevel, ScoringBand] = {
    ScoringLevel.A: ScoringBand.LEADERSHIP,
    ScoringLevel.A_MINUS: ScoringBand.LEADERSHIP,
    ScoringLevel.B: ScoringBand.MANAGEMENT,
    ScoringLevel.B_MINUS: ScoringBand.MANAGEMENT,
    ScoringLevel.C: ScoringBand.AWARENESS,
    ScoringLevel.C_MINUS: ScoringBand.AWARENESS,
    ScoringLevel.D: ScoringBand.DISCLOSURE,
    ScoringLevel.D_MINUS: ScoringBand.DISCLOSURE,
}


# ---------------------------------------------------------------------------
# 17 Scoring Category Weights (management and leadership)
# ---------------------------------------------------------------------------

SCORING_CATEGORY_WEIGHTS: Dict[str, Dict[str, Any]] = {
    "SC01": {
        "name": "Governance",
        "weight_management": 7.0,
        "weight_leadership": 7.0,
        "modules": ["M1"],
    },
    "SC02": {
        "name": "Risk management processes",
        "weight_management": 6.0,
        "weight_leadership": 5.0,
        "modules": ["M3"],
    },
    "SC03": {
        "name": "Risk disclosure",
        "weight_management": 5.0,
        "weight_leadership": 4.0,
        "modules": ["M3"],
    },
    "SC04": {
        "name": "Opportunity disclosure",
        "weight_management": 5.0,
        "weight_leadership": 4.0,
        "modules": ["M3"],
    },
    "SC05": {
        "name": "Business strategy",
        "weight_management": 6.0,
        "weight_leadership": 5.0,
        "modules": ["M4"],
    },
    "SC06": {
        "name": "Scenario analysis",
        "weight_management": 5.0,
        "weight_leadership": 5.0,
        "modules": ["M4"],
    },
    "SC07": {
        "name": "Targets",
        "weight_management": 8.0,
        "weight_leadership": 8.0,
        "modules": ["M5", "M6"],
    },
    "SC08": {
        "name": "Emissions reduction initiatives",
        "weight_management": 7.0,
        "weight_leadership": 7.0,
        "modules": ["M6"],
    },
    "SC09": {
        "name": "Scope 1 & 2 emissions (incl. verification)",
        "weight_management": 10.0,
        "weight_leadership": 10.0,
        "modules": ["M7"],
    },
    "SC10": {
        "name": "Scope 3 emissions (incl. verification)",
        "weight_management": 8.0,
        "weight_leadership": 8.0,
        "modules": ["M7"],
    },
    "SC11": {
        "name": "Energy",
        "weight_management": 6.0,
        "weight_leadership": 6.0,
        "modules": ["M7", "M11"],
    },
    "SC12": {
        "name": "Carbon pricing",
        "weight_management": 4.0,
        "weight_leadership": 4.0,
        "modules": ["M6"],
    },
    "SC13": {
        "name": "Value chain engagement",
        "weight_management": 6.0,
        "weight_leadership": 6.0,
        "modules": ["M10"],
    },
    "SC14": {
        "name": "Public policy engagement",
        "weight_management": 3.0,
        "weight_leadership": 3.0,
        "modules": ["M1", "M4"],
    },
    "SC15": {
        "name": "Transition plan",
        "weight_management": 6.0,
        "weight_leadership": 8.0,
        "modules": ["M5"],
    },
    "SC16": {
        "name": "Portfolio climate performance (FS only)",
        "weight_management": 5.0,
        "weight_leadership": 7.0,
        "modules": ["M12"],
        "sector_specific": True,
        "applicable_sectors": ["40"],
    },
    "SC17": {
        "name": "Financial impact assessment",
        "weight_management": 3.0,
        "weight_leadership": 3.0,
        "modules": ["M3", "M4"],
    },
}


# ---------------------------------------------------------------------------
# Module Definitions
# ---------------------------------------------------------------------------

MODULE_DEFINITIONS: Dict[str, Dict[str, Any]] = {
    "M0": {
        "name": "Introduction",
        "description": "Organization profile, reporting boundary, base year",
        "question_count": 15,
        "required": True,
        "order": 0,
    },
    "M1": {
        "name": "Governance",
        "description": "Board oversight, management responsibility, incentives",
        "question_count": 20,
        "required": True,
        "order": 1,
    },
    "M2": {
        "name": "Policies & Commitments",
        "description": "Climate policies, commitments, deforestation-free",
        "question_count": 15,
        "required": True,
        "order": 2,
    },
    "M3": {
        "name": "Risks & Opportunities",
        "description": "Climate risk assessment, physical and transition risks",
        "question_count": 25,
        "required": True,
        "order": 3,
    },
    "M4": {
        "name": "Strategy",
        "description": "Business strategy alignment, scenario analysis",
        "question_count": 20,
        "required": True,
        "order": 4,
    },
    "M5": {
        "name": "Transition Plans",
        "description": "1.5C pathway, decarbonization roadmap, milestones",
        "question_count": 20,
        "required": True,
        "order": 5,
    },
    "M6": {
        "name": "Implementation",
        "description": "Emissions reduction initiatives, investments, R&D",
        "question_count": 20,
        "required": True,
        "order": 6,
    },
    "M7": {
        "name": "Environmental Performance - Climate Change",
        "description": "Scope 1/2/3 emissions, methodology, verification",
        "question_count": 35,
        "required": True,
        "order": 7,
    },
    "M8": {
        "name": "Environmental Performance - Forests",
        "description": "Commodity-driven deforestation (if applicable)",
        "question_count": 15,
        "required": False,
        "order": 8,
        "sector_specific": True,
    },
    "M9": {
        "name": "Environmental Performance - Water Security",
        "description": "Water dependencies (if applicable)",
        "question_count": 15,
        "required": False,
        "order": 9,
        "sector_specific": True,
    },
    "M10": {
        "name": "Supply Chain",
        "description": "Supplier engagement, Scope 3 collaboration",
        "question_count": 15,
        "required": True,
        "order": 10,
    },
    "M11": {
        "name": "Additional Metrics",
        "description": "Sector-specific metrics, energy mix",
        "question_count": 10,
        "required": True,
        "order": 11,
    },
    "M12": {
        "name": "Financial Services",
        "description": "Portfolio emissions, financed emissions (if FS)",
        "question_count": 20,
        "required": False,
        "order": 12,
        "sector_specific": True,
        "applicable_sectors": ["40"],
    },
    "M13": {
        "name": "Sign Off",
        "description": "Authorization, verification statement",
        "question_count": 5,
        "required": True,
        "order": 13,
    },
}


# ---------------------------------------------------------------------------
# MRV Agent to CDP Scope Mapping
# ---------------------------------------------------------------------------

MRV_AGENT_TO_CDP_SCOPE: Dict[str, Dict[str, Any]] = {
    # Scope 1 agents
    "MRV-001": {"scope": "scope_1", "name": "Stationary Combustion", "cdp_table": "C6.1"},
    "MRV-002": {"scope": "scope_1", "name": "Refrigerants & F-Gas", "cdp_table": "C6.1"},
    "MRV-003": {"scope": "scope_1", "name": "Mobile Combustion", "cdp_table": "C6.1"},
    "MRV-004": {"scope": "scope_1", "name": "Process Emissions", "cdp_table": "C6.1"},
    "MRV-005": {"scope": "scope_1", "name": "Fugitive Emissions", "cdp_table": "C6.1"},
    "MRV-006": {"scope": "scope_1", "name": "Land Use Emissions", "cdp_table": "C6.1"},
    "MRV-007": {"scope": "scope_1", "name": "Waste Treatment Emissions", "cdp_table": "C6.1"},
    "MRV-008": {"scope": "scope_1", "name": "Agricultural Emissions", "cdp_table": "C6.1"},
    # Scope 2 agents
    "MRV-009": {"scope": "scope_2", "name": "Scope 2 Location-Based", "cdp_table": "C6.3"},
    "MRV-010": {"scope": "scope_2", "name": "Scope 2 Market-Based", "cdp_table": "C6.3"},
    "MRV-011": {"scope": "scope_2", "name": "Steam/Heat Purchase", "cdp_table": "C6.3"},
    "MRV-012": {"scope": "scope_2", "name": "Cooling Purchase", "cdp_table": "C6.3"},
    "MRV-013": {"scope": "scope_2", "name": "Dual Reporting Reconciliation", "cdp_table": "C6.3"},
    # Scope 3 agents (Cat 1-15)
    "MRV-014": {"scope": "scope_3", "name": "Purchased Goods & Services", "cdp_table": "C6.5", "category": 1},
    "MRV-015": {"scope": "scope_3", "name": "Capital Goods", "cdp_table": "C6.5", "category": 2},
    "MRV-016": {"scope": "scope_3", "name": "Fuel & Energy Activities", "cdp_table": "C6.5", "category": 3},
    "MRV-017": {"scope": "scope_3", "name": "Upstream Transportation", "cdp_table": "C6.5", "category": 4},
    "MRV-018": {"scope": "scope_3", "name": "Waste Generated", "cdp_table": "C6.5", "category": 5},
    "MRV-019": {"scope": "scope_3", "name": "Business Travel", "cdp_table": "C6.5", "category": 6},
    "MRV-020": {"scope": "scope_3", "name": "Employee Commuting", "cdp_table": "C6.5", "category": 7},
    "MRV-021": {"scope": "scope_3", "name": "Upstream Leased Assets", "cdp_table": "C6.5", "category": 8},
    "MRV-022": {"scope": "scope_3", "name": "Downstream Transportation", "cdp_table": "C6.5", "category": 9},
    "MRV-023": {"scope": "scope_3", "name": "Processing of Sold Products", "cdp_table": "C6.5", "category": 10},
    "MRV-024": {"scope": "scope_3", "name": "Use of Sold Products", "cdp_table": "C6.5", "category": 11},
    "MRV-025": {"scope": "scope_3", "name": "End-of-Life Treatment", "cdp_table": "C6.5", "category": 12},
    "MRV-026": {"scope": "scope_3", "name": "Downstream Leased Assets", "cdp_table": "C6.5", "category": 13},
    "MRV-027": {"scope": "scope_3", "name": "Franchises", "cdp_table": "C6.5", "category": 14},
    "MRV-028": {"scope": "scope_3", "name": "Investments", "cdp_table": "C6.5", "category": 15},
    # Cross-cutting
    "MRV-029": {"scope": "cross_cutting", "name": "Scope 3 Category Mapper"},
    "MRV-030": {"scope": "cross_cutting", "name": "Audit Trail & Lineage"},
}


# ---------------------------------------------------------------------------
# Sector Benchmark Data (CDP 2024/2025 averages)
# ---------------------------------------------------------------------------

SECTOR_BENCHMARK_SCORES: Dict[str, Dict[str, Any]] = {
    "10": {"name": "Energy", "avg_score": 42.5, "a_list_pct": 3.2, "median": "C"},
    "15": {"name": "Materials", "avg_score": 45.0, "a_list_pct": 4.1, "median": "C"},
    "20": {"name": "Industrials", "avg_score": 48.0, "a_list_pct": 5.5, "median": "C"},
    "25": {"name": "Consumer Disc.", "avg_score": 50.5, "a_list_pct": 7.2, "median": "B-"},
    "30": {"name": "Consumer Staples", "avg_score": 55.0, "a_list_pct": 10.1, "median": "B-"},
    "35": {"name": "Health Care", "avg_score": 47.0, "a_list_pct": 5.0, "median": "C"},
    "40": {"name": "Financials", "avg_score": 52.0, "a_list_pct": 8.3, "median": "B-"},
    "45": {"name": "Info Technology", "avg_score": 56.0, "a_list_pct": 11.5, "median": "B-"},
    "50": {"name": "Comm. Services", "avg_score": 49.0, "a_list_pct": 6.0, "median": "C"},
    "55": {"name": "Utilities", "avg_score": 54.0, "a_list_pct": 9.5, "median": "B-"},
    "60": {"name": "Real Estate", "avg_score": 53.0, "a_list_pct": 9.0, "median": "B-"},
}


# ---------------------------------------------------------------------------
# A-Level Requirement IDs
# ---------------------------------------------------------------------------

A_LEVEL_REQUIREMENTS: List[Dict[str, str]] = [
    {
        "id": "AREQ01",
        "name": "Public transition plan",
        "description": "Publicly available 1.5C-aligned transition plan",
    },
    {
        "id": "AREQ02",
        "name": "Complete emissions inventory",
        "description": "Complete emissions inventory with no material exclusions",
    },
    {
        "id": "AREQ03",
        "name": "Scope 1+2 verification",
        "description": "Third-party verification of 100% Scope 1 and Scope 2 emissions",
    },
    {
        "id": "AREQ04",
        "name": "Scope 3 verification",
        "description": "Third-party verification of >= 70% of at least one Scope 3 category",
    },
    {
        "id": "AREQ05",
        "name": "Science-based target",
        "description": "SBTi-validated or 1.5C-aligned target (>= 4.2% annual absolute reduction)",
    },
]


# ---------------------------------------------------------------------------
# GICS Sector Display Names
# ---------------------------------------------------------------------------

GICS_SECTOR_NAMES: Dict[str, str] = {
    "10": "Energy",
    "15": "Materials",
    "20": "Industrials",
    "25": "Consumer Discretionary",
    "30": "Consumer Staples",
    "35": "Health Care",
    "40": "Financials",
    "45": "Information Technology",
    "50": "Communication Services",
    "55": "Utilities",
    "60": "Real Estate",
}


# ---------------------------------------------------------------------------
# Main Configuration Class
# ---------------------------------------------------------------------------

class CDPAppConfig(BaseSettings):
    """
    GL-CDP-APP v1.0 platform configuration.

    All settings can be overridden via environment variables prefixed
    with ``CDP_APP_``.  For example ``CDP_APP_DEBUG`` maps to ``debug``.

    Example:
        >>> config = CDPAppConfig()
        >>> config.app_name
        'GL-CDP-APP'
        >>> config.default_questionnaire_year
        2026
    """

    model_config = {"env_prefix": "CDP_APP_"}

    # -- Application Metadata -----------------------------------------------
    app_name: str = Field(
        default="GL-CDP-APP",
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

    # -- Questionnaire Configuration ----------------------------------------
    default_questionnaire_year: int = Field(
        default=2026,
        ge=2024,
        le=2030,
        description="Default CDP questionnaire year",
    )
    supported_questionnaire_years: List[int] = Field(
        default=[2024, 2025, 2026],
        description="Supported questionnaire versions",
    )

    # -- Scoring Configuration ----------------------------------------------
    score_confidence_min_completion: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Minimum completion ratio for score confidence",
    )
    scoring_weight_mode: str = Field(
        default="management",
        description="Weight mode for scoring: management or leadership",
    )

    # -- Submission Configuration -------------------------------------------
    submission_deadline_month: int = Field(
        default=7,
        ge=1,
        le=12,
        description="Submission deadline month",
    )
    submission_deadline_day: int = Field(
        default=31,
        ge=1,
        le=31,
        description="Submission deadline day",
    )

    # -- Response Configuration ---------------------------------------------
    response_max_length: int = Field(
        default=5000,
        ge=100,
        le=50000,
        description="Maximum character length for text responses",
    )
    table_max_rows: int = Field(
        default=100,
        ge=1,
        le=1000,
        description="Maximum rows in table responses",
    )
    auto_save_interval_seconds: int = Field(
        default=30,
        ge=5,
        le=300,
        description="Auto-save interval in seconds",
    )

    # -- Evidence Configuration ---------------------------------------------
    evidence_max_size_mb: int = Field(
        default=25,
        ge=1,
        le=100,
        description="Maximum evidence attachment size in MB",
    )
    evidence_allowed_types: List[str] = Field(
        default=["pdf", "xlsx", "csv", "png", "jpg", "docx"],
        description="Allowed evidence file types",
    )

    # -- Benchmarking Configuration -----------------------------------------
    benchmark_min_peer_count: int = Field(
        default=5,
        ge=1,
        le=100,
        description="Minimum peer count for meaningful benchmarking",
    )

    # -- Supply Chain Configuration -----------------------------------------
    supplier_invitation_expiry_days: int = Field(
        default=90,
        ge=7,
        le=365,
        description="Days before supplier invitation expires",
    )
    max_reviewers_per_question: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum reviewers assignable per question",
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

    # -- Report Generation --------------------------------------------------
    default_report_format: ReportFormat = Field(
        default=ReportFormat.JSON,
        description="Default report export format",
    )
    report_storage_path: str = Field(
        default="reports/cdp/",
        description="Path prefix for generated reports",
    )

    # -- Logging ------------------------------------------------------------
    log_level: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    )
