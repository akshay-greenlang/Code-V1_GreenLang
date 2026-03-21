"""
PACK-022 Net Zero Acceleration Pack - Configuration Manager

This module implements the NetZeroAccelerationConfig and PackConfig classes that
load, merge, and validate all configuration for the Net Zero Acceleration Pack.
It provides comprehensive Pydantic v2 models for every aspect of advanced net-zero
planning: multi-scenario Monte Carlo analysis, SBTi Sectoral Decarbonization
Approach (SDA) pathways, tiered supplier engagement, activity-based Scope 3
calculations, climate transition finance, SBTi Temperature Rating, emissions
variance decomposition, multi-entity consolidation, VCMI Claims Code validation,
and assurance workpaper generation.

Scenario Types:
    - BAU: Business-as-usual, no additional reduction actions
    - MODERATE: Moderate ambition, cost-effective actions only
    - AMBITIOUS: Aggressive reduction, all feasible actions
    - CUSTOM: User-defined scenario parameters

Pathway Methodologies:
    - ACA: Absolute Contraction Approach (cross-sector, all sizes)
    - SDA: Sectoral Decarbonization Approach (homogeneous sectors)
    - FLAG: Forest, Land and Agriculture (land-intensive sectors)

Supplier Tiers:
    - INFORM: Awareness communications and basic training
    - ENGAGE: Data collection, target sharing, and capacity building
    - REQUIRE: Contractual emission reduction commitments
    - COLLABORATE: Joint decarbonization projects and innovation

Finance Instruments:
    - GREEN_BOND: ICMA Green Bond Principles-aligned issuance
    - SUSTAINABILITY_LINKED_BOND: KPI-linked coupon adjustment
    - GREEN_LOAN: Green Loan Principles-aligned facility
    - INTERNAL_CARBON_PRICE: Shadow carbon price for decision-making
    - CAPEX_ALLOCATION: Direct capital expenditure for decarbonization
    - OPEX_ALLOCATION: Operational expenditure for green procurement

Temperature Targets:
    - CELSIUS_1_5: 1.5C alignment (Paris-aligned, SBTi net-zero)
    - WELL_BELOW_2: Well-below 2C (SBTi minimum ambition)
    - CELSIUS_2: 2C alignment (minimum Paris requirement)

Decomposition Methods:
    - LMDI: Logarithmic Mean Divisia Index (recommended)
    - SDA_DECOMP: Structural Decomposition Analysis
    - IDA: Index Decomposition Analysis

VCMI Tiers:
    - SILVER: Meeting near-term targets, basic credit quality
    - GOLD: Exceeding targets, high credit quality with removals
    - PLATINUM: Leading reduction, highest credit quality, full removals

Assurance Levels:
    - LIMITED: Limited assurance per ISAE 3000 (negative assurance)
    - REASONABLE: Reasonable assurance per ISAE 3410 (positive assurance)

Configuration Merge Order (later overrides earlier):
    1. Base pack.yaml manifest
    2. PACK-021 baseline config (if linked)
    3. Preset YAML (8 sector presets)
    4. Environment overrides (NET_ZERO_ACCEL_* environment variables)
    5. Explicit runtime overrides

Regulatory Context:
    - SBTi Net-Zero Standard v1.2 (2024)
    - SBTi Temperature Rating Methodology v2.0 (2024)
    - SBTi Sectoral Decarbonization Approach (2015, updated 2024)
    - GHG Protocol Corporate Standard (revised)
    - GHG Protocol Scope 3 Standard
    - IPCC AR6 WG3 (Mitigation, 2022)
    - VCMI Claims Code of Practice (2023)
    - EU Taxonomy Climate Delegated Act (2021/2139)
    - PCAF Global Standard (2022)
    - IEA Net Zero Roadmap (2023)
    - CRREM v2.0 (2024)

Example:
    >>> config = PackConfig.from_preset("heavy_industry")
    >>> print(config.pack.scenario.scenario_types)
    [ScenarioType.BAU, ScenarioType.MODERATE, ScenarioType.AMBITIOUS]
    >>> print(config.pack.pathway.methodology)
    PathwayMethodology.SDA
"""

import hashlib
import logging
import os
import uuid
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

logger = logging.getLogger(__name__)

# Base directory for all pack configuration files
PACK_BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = Path(__file__).resolve().parent

# =============================================================================
# Constants
# =============================================================================

DEFAULT_BASE_YEAR: int = 2021
DEFAULT_NEAR_TERM_YEAR: int = 2030
DEFAULT_LONG_TERM_YEAR: int = 2050
DEFAULT_MONTE_CARLO_RUNS: int = 1000
DEFAULT_MAX_SUPPLIERS: int = 50000
DEFAULT_MAX_ENTITIES: int = 50
DEFAULT_SCOPE3_CATEGORIES: List[int] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

SUPPORTED_PRESETS: Dict[str, str] = {
    "heavy_industry": "Steel, cement, aluminium, chemicals with SDA mandatory",
    "power_utilities": "Power generation and utilities with grid decarbonization",
    "manufacturing": "General manufacturing with mixed SDA/ACA pathway",
    "transport_logistics": "Transport, aviation, shipping with fleet electrification",
    "financial_services": "Banks, insurance, asset managers with PCAF financed emissions",
    "real_estate": "Real estate and construction with CRREM pathway alignment",
    "consumer_goods": "FMCG and retail with supply chain engagement focus",
    "technology": "Technology, software, data centers with ACA and RE100",
}

# SBTi SDA supported sectors (12 sectors)
SDA_SECTORS: Dict[str, str] = {
    "POWER": "Power Generation",
    "CEMENT": "Cement",
    "STEEL": "Iron and Steel",
    "ALUMINIUM": "Aluminium",
    "PULP_PAPER": "Pulp and Paper",
    "TRANSPORT_ROAD": "Road Transport",
    "TRANSPORT_RAIL": "Rail Transport",
    "BUILDINGS_RESIDENTIAL": "Residential Buildings",
    "BUILDINGS_COMMERCIAL": "Commercial Buildings",
    "CHEMICALS": "Chemicals",
    "AVIATION": "Aviation",
    "SHIPPING": "Shipping",
}

# SBTi SDA sector intensity metrics (tCO2e per unit)
SDA_INTENSITY_METRICS: Dict[str, str] = {
    "POWER": "tCO2e/MWh",
    "CEMENT": "tCO2e/tonne clinker",
    "STEEL": "tCO2e/tonne crude steel",
    "ALUMINIUM": "tCO2e/tonne aluminium",
    "PULP_PAPER": "tCO2e/tonne product",
    "TRANSPORT_ROAD": "gCO2e/tonne-km",
    "TRANSPORT_RAIL": "gCO2e/passenger-km",
    "BUILDINGS_RESIDENTIAL": "kgCO2e/m2/year",
    "BUILDINGS_COMMERCIAL": "kgCO2e/m2/year",
    "CHEMICALS": "tCO2e/tonne product",
    "AVIATION": "gCO2e/revenue-tonne-km",
    "SHIPPING": "gCO2e/tonne-nautical-mile",
}

# SBTi Temperature Rating regression coefficients (simplified v2.0)
TEMPERATURE_REGRESSION: Dict[str, Dict[str, float]] = {
    "scope_1_2": {
        "intercept": 3.2,
        "slope_near_term": -0.035,
        "slope_long_term": -0.018,
    },
    "scope_3": {
        "intercept": 3.5,
        "slope_near_term": -0.028,
        "slope_long_term": -0.014,
    },
}

# VCMI Claims Code tier thresholds (minimum percentage scores)
VCMI_TIER_THRESHOLDS: Dict[str, Dict[str, float]] = {
    "SILVER": {
        "foundation_min": 60.0,
        "credibility_min": 50.0,
        "credit_quality_min": 50.0,
        "transparency_min": 50.0,
        "overall_min": 55.0,
    },
    "GOLD": {
        "foundation_min": 80.0,
        "credibility_min": 70.0,
        "credit_quality_min": 70.0,
        "transparency_min": 70.0,
        "overall_min": 72.0,
    },
    "PLATINUM": {
        "foundation_min": 90.0,
        "credibility_min": 85.0,
        "credit_quality_min": 85.0,
        "transparency_min": 85.0,
        "overall_min": 86.0,
    },
}

# IPCC AR6 GWP100 values for common GHGs
IPCC_AR6_GWP100: Dict[str, int] = {
    "CO2": 1,
    "CH4": 27,
    "N2O": 273,
    "HFC_134A": 1430,
    "HFC_32": 675,
    "R404A": 3922,
    "R410A": 2088,
    "SF6": 25200,
    "NF3": 17400,
    "R290": 3,
    "R744": 1,
}

# SBTi minimum annual reduction rates by ambition level
SBTI_REDUCTION_RATES: Dict[str, Dict[str, float]] = {
    "CELSIUS_1_5": {
        "scope_1_2_linear_annual": 4.2,
        "scope_3_linear_annual": 2.5,
        "long_term_reduction_pct": 90.0,
    },
    "WELL_BELOW_2": {
        "scope_1_2_linear_annual": 2.5,
        "scope_3_linear_annual": 1.8,
        "long_term_reduction_pct": 90.0,
    },
    "CELSIUS_2": {
        "scope_1_2_linear_annual": 1.5,
        "scope_3_linear_annual": 1.2,
        "long_term_reduction_pct": 80.0,
    },
}

# SBTi coverage thresholds
SBTI_COVERAGE_THRESHOLDS: Dict[str, float] = {
    "scope_1_near_term_pct": 95.0,
    "scope_2_near_term_pct": 95.0,
    "scope_3_near_term_pct": 67.0,
    "scope_1_long_term_pct": 95.0,
    "scope_2_long_term_pct": 95.0,
    "scope_3_long_term_pct": 90.0,
}

# Sector-specific Scope 3 category priorities
SECTOR_SCOPE3_PRIORITY: Dict[str, Dict[int, str]] = {
    "HEAVY_INDUSTRY": {
        1: "CRITICAL", 2: "MEDIUM", 3: "CRITICAL", 4: "HIGH", 5: "HIGH",
        6: "LOW", 7: "LOW", 8: "LOW", 9: "MEDIUM", 10: "MEDIUM",
        11: "HIGH", 12: "MEDIUM", 13: "LOW", 14: "LOW", 15: "LOW",
    },
    "POWER_UTILITIES": {
        1: "HIGH", 2: "MEDIUM", 3: "CRITICAL", 4: "MEDIUM", 5: "LOW",
        6: "LOW", 7: "LOW", 8: "MEDIUM", 9: "MEDIUM", 10: "LOW",
        11: "CRITICAL", 12: "LOW", 13: "LOW", 14: "LOW", 15: "HIGH",
    },
    "MANUFACTURING": {
        1: "CRITICAL", 2: "MEDIUM", 3: "HIGH", 4: "HIGH", 5: "HIGH",
        6: "LOW", 7: "LOW", 8: "MEDIUM", 9: "MEDIUM", 10: "MEDIUM",
        11: "MEDIUM", 12: "LOW", 13: "LOW", 14: "LOW", 15: "LOW",
    },
    "TRANSPORT": {
        1: "HIGH", 2: "MEDIUM", 3: "CRITICAL", 4: "CRITICAL", 5: "LOW",
        6: "LOW", 7: "LOW", 8: "MEDIUM", 9: "CRITICAL", 10: "LOW",
        11: "LOW", 12: "LOW", 13: "LOW", 14: "LOW", 15: "LOW",
    },
    "FINANCIAL_SERVICES": {
        1: "MEDIUM", 2: "LOW", 3: "LOW", 4: "LOW", 5: "LOW",
        6: "HIGH", 7: "MEDIUM", 8: "HIGH", 9: "LOW", 10: "LOW",
        11: "LOW", 12: "LOW", 13: "MEDIUM", 14: "LOW", 15: "CRITICAL",
    },
    "REAL_ESTATE": {
        1: "HIGH", 2: "HIGH", 3: "MEDIUM", 4: "HIGH", 5: "HIGH",
        6: "LOW", 7: "LOW", 8: "MEDIUM", 9: "LOW", 10: "LOW",
        11: "MEDIUM", 12: "MEDIUM", 13: "CRITICAL", 14: "LOW", 15: "MEDIUM",
    },
    "CONSUMER_GOODS": {
        1: "CRITICAL", 2: "LOW", 3: "MEDIUM", 4: "HIGH", 5: "HIGH",
        6: "LOW", 7: "LOW", 8: "MEDIUM", 9: "HIGH", 10: "LOW",
        11: "MEDIUM", 12: "HIGH", 13: "LOW", 14: "MEDIUM", 15: "LOW",
    },
    "TECHNOLOGY": {
        1: "CRITICAL", 2: "HIGH", 3: "MEDIUM", 4: "MEDIUM", 5: "LOW",
        6: "MEDIUM", 7: "MEDIUM", 8: "HIGH", 9: "LOW", 10: "LOW",
        11: "CRITICAL", 12: "MEDIUM", 13: "LOW", 14: "LOW", 15: "LOW",
    },
}

# Priority Scope 3 categories by sector
PRIORITY_SCOPE3_BY_SECTOR: Dict[str, List[int]] = {
    "HEAVY_INDUSTRY": [1, 3, 4, 5, 11],
    "POWER_UTILITIES": [3, 11, 15],
    "MANUFACTURING": [1, 3, 4, 5, 11],
    "TRANSPORT": [1, 3, 4, 9],
    "FINANCIAL_SERVICES": [1, 6, 7, 15],
    "REAL_ESTATE": [1, 2, 4, 5, 13],
    "CONSUMER_GOODS": [1, 4, 9, 12],
    "TECHNOLOGY": [1, 2, 8, 11],
}

# Sector display names and SBTi pathway recommendations
SECTOR_INFO: Dict[str, Dict[str, Any]] = {
    "HEAVY_INDUSTRY": {
        "name": "Heavy Industry (Cement, Steel, Chemicals)",
        "recommended_pathway": "SDA",
        "sda_sectors": ["CEMENT", "STEEL", "ALUMINIUM", "CHEMICALS"],
        "typical_scope_split": "Scope 1: 50-80%, Scope 2: 10-20%, Scope 3: 15-35%",
        "key_levers": ["process optimization", "CCS", "green hydrogen", "fuel switching"],
    },
    "POWER_UTILITIES": {
        "name": "Power Generation & Utilities",
        "recommended_pathway": "SDA",
        "sda_sectors": ["POWER"],
        "typical_scope_split": "Scope 1: 60-85%, Scope 2: 2-5%, Scope 3: 10-30%",
        "key_levers": ["renewable capacity", "coal phase-out", "grid storage", "CCS"],
    },
    "MANUFACTURING": {
        "name": "General Manufacturing",
        "recommended_pathway": "SDA",
        "sda_sectors": ["PULP_PAPER", "CHEMICALS"],
        "typical_scope_split": "Scope 1: 30-60%, Scope 2: 10-25%, Scope 3: 30-50%",
        "key_levers": ["energy efficiency", "fuel switching", "process optimization"],
    },
    "TRANSPORT": {
        "name": "Transport, Aviation & Shipping",
        "recommended_pathway": "SDA",
        "sda_sectors": ["TRANSPORT_ROAD", "TRANSPORT_RAIL", "AVIATION", "SHIPPING"],
        "typical_scope_split": "Scope 1: 50-80%, Scope 2: 5-15%, Scope 3: 15-40%",
        "key_levers": ["fleet electrification", "SAF", "modal shift", "route optimization"],
    },
    "FINANCIAL_SERVICES": {
        "name": "Financial Services (Banks, Insurance, Asset Management)",
        "recommended_pathway": "ACA",
        "sda_sectors": [],
        "typical_scope_split": "Scope 1: 3-8%, Scope 2: 10-20%, Scope 3: 75-90% (financed)",
        "key_levers": ["portfolio decarbonization", "green finance", "engagement"],
    },
    "REAL_ESTATE": {
        "name": "Real Estate, Property & Construction",
        "recommended_pathway": "SDA",
        "sda_sectors": ["BUILDINGS_COMMERCIAL", "BUILDINGS_RESIDENTIAL"],
        "typical_scope_split": "Scope 1: 10-25%, Scope 2: 15-30%, Scope 3: 50-70%",
        "key_levers": ["building retrofit", "CRREM alignment", "heat pumps", "renewables"],
    },
    "CONSUMER_GOODS": {
        "name": "Consumer Goods, FMCG & Retail",
        "recommended_pathway": "ACA",
        "sda_sectors": [],
        "typical_scope_split": "Scope 1: 5-15%, Scope 2: 10-20%, Scope 3: 70-85%",
        "key_levers": ["supplier engagement", "logistics optimization", "packaging"],
    },
    "TECHNOLOGY": {
        "name": "Technology, Software & Data Centers",
        "recommended_pathway": "ACA",
        "sda_sectors": [],
        "typical_scope_split": "Scope 1: 2-8%, Scope 2: 15-35%, Scope 3: 60-80%",
        "key_levers": ["RE100", "data center PUE", "cloud efficiency", "hardware lifecycle"],
    },
}


# =============================================================================
# Helper Functions
# =============================================================================


def _utcnow() -> datetime:
    """Return the current UTC datetime."""
    return datetime.now(timezone.utc)


def _new_uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())


def _compute_hash(data: str) -> str:
    """Compute SHA-256 hash of a string.

    Args:
        data: Input string to hash.

    Returns:
        Hex-encoded SHA-256 hash.
    """
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _decimal(value: float, places: int = 6) -> Decimal:
    """Convert a float to a Decimal with specified precision.

    Args:
        value: Float value to convert.
        places: Number of decimal places.

    Returns:
        Decimal with specified precision.
    """
    quantize_str = "0." + "0" * places
    return Decimal(str(value)).quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP)


def _safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers, returning default on zero division.

    Args:
        numerator: Dividend.
        denominator: Divisor.
        default: Value to return if denominator is zero.

    Returns:
        Result of division or default.
    """
    if denominator == 0.0:
        return default
    return numerator / denominator


def _safe_pct(part: float, whole: float, default: float = 0.0) -> float:
    """Calculate percentage safely.

    Args:
        part: Numerator (part of whole).
        whole: Denominator (total).
        default: Value to return if whole is zero.

    Returns:
        Percentage value (0-100) or default.
    """
    return _safe_divide(part * 100.0, whole, default)


def _round_val(value: float, decimals: int = 2) -> float:
    """Round a value to the specified number of decimal places.

    Args:
        value: Value to round.
        decimals: Number of decimal places.

    Returns:
        Rounded float.
    """
    return round(value, decimals)


# =============================================================================
# Enums - Net Zero Acceleration specific enumeration types (11 enums)
# =============================================================================


class ScenarioType(str, Enum):
    """Scenario type for Monte Carlo pathway analysis."""

    BAU = "BAU"
    MODERATE = "MODERATE"
    AMBITIOUS = "AMBITIOUS"
    CUSTOM = "CUSTOM"


class PathwayMethodology(str, Enum):
    """SBTi target-setting pathway methodology."""

    ACA = "ACA"
    SDA = "SDA"
    FLAG = "FLAG"


class SupplierTier(str, Enum):
    """Supplier engagement tier level."""

    INFORM = "INFORM"
    ENGAGE = "ENGAGE"
    REQUIRE = "REQUIRE"
    COLLABORATE = "COLLABORATE"


class ScopeCategory(str, Enum):
    """Scope category for emissions classification."""

    SCOPE_1 = "SCOPE_1"
    SCOPE_2_LOCATION = "SCOPE_2_LOCATION"
    SCOPE_2_MARKET = "SCOPE_2_MARKET"
    SCOPE_3 = "SCOPE_3"


class FinanceInstrument(str, Enum):
    """Climate finance instrument types."""

    GREEN_BOND = "GREEN_BOND"
    SUSTAINABILITY_LINKED_BOND = "SUSTAINABILITY_LINKED_BOND"
    GREEN_LOAN = "GREEN_LOAN"
    INTERNAL_CARBON_PRICE = "INTERNAL_CARBON_PRICE"
    CAPEX_ALLOCATION = "CAPEX_ALLOCATION"
    OPEX_ALLOCATION = "OPEX_ALLOCATION"


class TemperatureTarget(str, Enum):
    """Temperature alignment target level."""

    CELSIUS_1_5 = "CELSIUS_1_5"
    WELL_BELOW_2 = "WELL_BELOW_2"
    CELSIUS_2 = "CELSIUS_2"


class DecompositionMethod(str, Enum):
    """Emissions variance decomposition method."""

    LMDI = "LMDI"
    SDA_DECOMP = "SDA_DECOMP"
    IDA = "IDA"


class EntityScope(str, Enum):
    """Multi-entity consolidation scope method."""

    EQUITY_SHARE = "EQUITY_SHARE"
    FINANCIAL_CONTROL = "FINANCIAL_CONTROL"
    OPERATIONAL_CONTROL = "OPERATIONAL_CONTROL"


class VCMITier(str, Enum):
    """VCMI Claims Code claim tier."""

    SILVER = "SILVER"
    GOLD = "GOLD"
    PLATINUM = "PLATINUM"


class AssuranceLevel(str, Enum):
    """Assurance engagement level."""

    LIMITED = "LIMITED"
    REASONABLE = "REASONABLE"


class SectorClassification(str, Enum):
    """Organization sector classification for PACK-022."""

    HEAVY_INDUSTRY = "HEAVY_INDUSTRY"
    POWER_UTILITIES = "POWER_UTILITIES"
    MANUFACTURING = "MANUFACTURING"
    TRANSPORT = "TRANSPORT"
    FINANCIAL_SERVICES = "FINANCIAL_SERVICES"
    REAL_ESTATE = "REAL_ESTATE"
    CONSUMER_GOODS = "CONSUMER_GOODS"
    TECHNOLOGY = "TECHNOLOGY"


# =============================================================================
# Pydantic Sub-Config Models (10 models)
# =============================================================================


class ScenarioConfig(BaseModel):
    """Configuration for multi-scenario Monte Carlo pathway analysis.

    Defines scenario types, simulation parameters, and uncertainty ranges
    for decarbonization pathway modeling.
    """

    scenario_types: List[ScenarioType] = Field(
        default_factory=lambda: [
            ScenarioType.BAU,
            ScenarioType.MODERATE,
            ScenarioType.AMBITIOUS,
        ],
        description="Scenario types to model",
    )
    monte_carlo_runs: int = Field(
        DEFAULT_MONTE_CARLO_RUNS,
        ge=100,
        le=10000,
        description="Number of Monte Carlo simulation runs per scenario",
    )
    random_seed: int = Field(
        42,
        ge=0,
        description="Random seed for reproducible Monte Carlo sampling",
    )
    confidence_interval_pct: float = Field(
        95.0,
        ge=80.0,
        le=99.9,
        description="Confidence interval for probability distribution outputs",
    )
    sensitivity_top_n: int = Field(
        10,
        ge=3,
        le=30,
        description="Number of top variables in tornado sensitivity analysis",
    )
    uncertainty_emission_factor_pct: float = Field(
        15.0,
        ge=0.0,
        le=50.0,
        description="Uncertainty range (+/-%) for emission factor inputs",
    )
    uncertainty_activity_data_pct: float = Field(
        10.0,
        ge=0.0,
        le=50.0,
        description="Uncertainty range (+/-%) for activity data inputs",
    )
    uncertainty_cost_pct: float = Field(
        20.0,
        ge=0.0,
        le=50.0,
        description="Uncertainty range (+/-%) for action cost inputs",
    )
    parallel_scenarios: bool = Field(
        True,
        description="Execute scenarios in parallel for performance",
    )
    max_custom_scenarios: int = Field(
        5,
        ge=1,
        le=10,
        description="Maximum number of custom user-defined scenarios",
    )


class PathwayConfig(BaseModel):
    """Configuration for SBTi Sectoral Decarbonization Approach.

    Defines sector-specific pathway parameters, intensity convergence
    targets, and benchmark sources for SDA compliance.
    """

    methodology: PathwayMethodology = Field(
        PathwayMethodology.ACA,
        description="Primary SBTi pathway methodology",
    )
    sda_enabled: bool = Field(
        False,
        description="Enable SDA sector-specific intensity pathways",
    )
    sda_sectors: List[str] = Field(
        default_factory=list,
        description="SDA sectors to calculate (from SDA_SECTORS keys)",
    )
    sda_base_year: int = Field(
        DEFAULT_BASE_YEAR,
        ge=2015,
        le=2030,
        description="Base year for SDA intensity calculation",
    )
    sda_target_year: int = Field(
        DEFAULT_LONG_TERM_YEAR,
        ge=2030,
        le=2060,
        description="Convergence target year for SDA pathway",
    )
    ambition_level: TemperatureTarget = Field(
        TemperatureTarget.CELSIUS_1_5,
        description="Temperature alignment ambition for pathway",
    )
    near_term_target_year: int = Field(
        DEFAULT_NEAR_TERM_YEAR,
        ge=2025,
        le=2035,
        description="Near-term target year",
    )
    long_term_target_year: int = Field(
        DEFAULT_LONG_TERM_YEAR,
        ge=2040,
        le=2055,
        description="Long-term / net-zero target year",
    )
    coverage_scope1_pct: float = Field(
        95.0,
        ge=0.0,
        le=100.0,
        description="Scope 1 emissions coverage percentage",
    )
    coverage_scope2_pct: float = Field(
        95.0,
        ge=0.0,
        le=100.0,
        description="Scope 2 emissions coverage percentage",
    )
    coverage_scope3_pct: float = Field(
        67.0,
        ge=0.0,
        le=100.0,
        description="Scope 3 emissions coverage percentage",
    )
    flag_pathway_enabled: bool = Field(
        False,
        description="Enable FLAG pathway for land-use sectors",
    )
    iea_benchmark_source: str = Field(
        "IEA_NZE_2023",
        description="IEA benchmark scenario for sector pathways",
    )
    sbti_submission_planned: bool = Field(
        True,
        description="Whether SBTi target submission is planned",
    )

    @field_validator("sda_sectors")
    @classmethod
    def validate_sda_sectors(cls, v: List[str]) -> List[str]:
        """Validate SDA sector codes are recognized."""
        invalid = [s for s in v if s not in SDA_SECTORS]
        if invalid:
            raise ValueError(
                f"Invalid SDA sectors: {invalid}. "
                f"Valid sectors: {sorted(SDA_SECTORS.keys())}"
            )
        return v

    @model_validator(mode="after")
    def validate_sda_requires_sectors(self) -> "PathwayConfig":
        """Ensure SDA-enabled config has at least one sector."""
        if self.sda_enabled and not self.sda_sectors:
            logger.warning(
                "SDA is enabled but no sectors specified. "
                "Add sda_sectors for sector-specific pathways."
            )
        return self

    @model_validator(mode="after")
    def validate_target_years(self) -> "PathwayConfig":
        """Ensure long-term target year is after near-term."""
        if self.long_term_target_year <= self.near_term_target_year:
            raise ValueError(
                f"long_term_target_year ({self.long_term_target_year}) must be "
                f"after near_term_target_year ({self.near_term_target_year})"
            )
        return self


class SupplierConfig(BaseModel):
    """Configuration for tiered supplier engagement program.

    Defines supplier tiering thresholds, engagement parameters,
    data collection settings, and progress tracking criteria.
    """

    enabled: bool = Field(
        True,
        description="Enable supplier engagement engine",
    )
    max_suppliers: int = Field(
        DEFAULT_MAX_SUPPLIERS,
        ge=100,
        le=100000,
        description="Maximum number of suppliers to manage",
    )
    tier_thresholds: Dict[str, float] = Field(
        default_factory=lambda: {
            "collaborate": 5.0,
            "require": 15.0,
            "engage": 40.0,
            "inform": 100.0,
        },
        description="Cumulative emission % thresholds for each tier",
    )
    top_supplier_count: int = Field(
        50,
        ge=10,
        le=500,
        description="Number of top suppliers for detailed engagement",
    )
    data_collection_frequency: str = Field(
        "annual",
        description="Frequency of supplier data collection (annual, semi-annual, quarterly)",
    )
    sbti_target_tracking: bool = Field(
        True,
        description="Track supplier SBTi target status",
    )
    cdp_supply_chain_enabled: bool = Field(
        True,
        description="Enable CDP Supply Chain questionnaire integration",
    )
    engagement_scoring_dimensions: List[str] = Field(
        default_factory=lambda: [
            "data_quality",
            "target_ambition",
            "reduction_progress",
            "responsiveness",
            "innovation",
        ],
        description="Dimensions for supplier engagement maturity scoring",
    )
    batch_size: int = Field(
        2000,
        ge=100,
        le=10000,
        description="Batch size for supplier processing",
    )

    @field_validator("data_collection_frequency")
    @classmethod
    def validate_frequency(cls, v: str) -> str:
        """Validate data collection frequency."""
        valid = {"annual", "semi-annual", "quarterly"}
        if v.lower() not in valid:
            raise ValueError(
                f"Invalid frequency: {v}. Must be one of: {sorted(valid)}"
            )
        return v.lower()


class Scope3Config(BaseModel):
    """Configuration for activity-based Scope 3 calculations.

    Defines which categories use activity-based vs. spend-based methods,
    data quality improvement targets, and materiality thresholds.
    """

    categories: List[int] = Field(
        default_factory=lambda: DEFAULT_SCOPE3_CATEGORIES.copy(),
        description="Scope 3 categories (1-15) included in calculations",
    )
    activity_based_categories: List[int] = Field(
        default_factory=lambda: [1, 3, 4, 5],
        description="Categories using activity-based calculation (high priority)",
    )
    supplier_specific_categories: List[int] = Field(
        default_factory=list,
        description="Categories using supplier-specific emission factors",
    )
    spend_based_categories: List[int] = Field(
        default_factory=lambda: [2, 6, 7, 8, 9, 10, 12, 13, 14],
        description="Categories using spend-based estimation as fallback",
    )
    materiality_threshold_pct: float = Field(
        1.0,
        ge=0.0,
        le=10.0,
        description="Threshold (%) below which a category is considered immaterial",
    )
    target_dqis_score: int = Field(
        2,
        ge=1,
        le=5,
        description="Target Data Quality Indicator Score (1=best, 5=worst)",
    )
    pcaf_enabled: bool = Field(
        False,
        description="Enable PCAF methodology for Category 15 (investments)",
    )
    pcaf_asset_classes: List[str] = Field(
        default_factory=list,
        description="PCAF asset classes to include (listed_equity, corporate_bonds, etc.)",
    )

    @field_validator("categories")
    @classmethod
    def validate_categories(cls, v: List[int]) -> List[int]:
        """Validate Scope 3 category numbers are 1-15."""
        invalid = [c for c in v if c < 1 or c > 15]
        if invalid:
            raise ValueError(
                f"Invalid Scope 3 categories: {invalid}. Must be 1-15."
            )
        return sorted(set(v))


class FinanceConfig(BaseModel):
    """Configuration for climate transition finance integration.

    Defines financial parameters for CapEx/OpEx planning, green bond
    screening, EU Taxonomy alignment, and internal carbon pricing.
    """

    enabled: bool = Field(
        True,
        description="Enable climate finance engine",
    )
    discount_rate_pct: float = Field(
        8.0,
        ge=0.0,
        le=30.0,
        description="Discount rate (%) for NPV calculations",
    )
    carbon_price_eur_per_tco2e: float = Field(
        85.0,
        ge=0.0,
        le=500.0,
        description="Internal carbon price (EUR/tCO2e) for action valuation",
    )
    carbon_price_escalation_pct: float = Field(
        5.0,
        ge=0.0,
        le=20.0,
        description="Annual carbon price escalation rate (%)",
    )
    reporting_currency: str = Field(
        "EUR",
        description="Reporting currency for financial metrics (ISO 4217)",
    )
    budget_constraint_eur: Optional[float] = Field(
        None,
        ge=0,
        description="Maximum total investment budget (EUR) for reduction actions",
    )
    planning_horizon_years: int = Field(
        10,
        ge=3,
        le=30,
        description="Financial planning horizon in years",
    )
    green_bond_screening: bool = Field(
        True,
        description="Screen actions for ICMA Green Bond Principles eligibility",
    )
    taxonomy_alignment: bool = Field(
        True,
        description="Calculate EU Taxonomy climate mitigation alignment ratios",
    )
    finance_instruments: List[FinanceInstrument] = Field(
        default_factory=lambda: [
            FinanceInstrument.CAPEX_ALLOCATION,
            FinanceInstrument.OPEX_ALLOCATION,
            FinanceInstrument.INTERNAL_CARBON_PRICE,
        ],
        description="Finance instruments to evaluate",
    )
    wacc_pct: float = Field(
        10.0,
        ge=0.0,
        le=30.0,
        description="Weighted average cost of capital for hurdle rate comparison",
    )

    @field_validator("reporting_currency")
    @classmethod
    def validate_currency(cls, v: str) -> str:
        """Validate reporting currency is a 3-letter ISO code."""
        if len(v) != 3 or not v.isalpha():
            raise ValueError(
                f"reporting_currency must be a 3-letter ISO 4217 code, got: {v}"
            )
        return v.upper()


class TemperatureConfig(BaseModel):
    """Configuration for SBTi Temperature Rating v2.0.

    Defines aggregation methods, scoring parameters, and portfolio
    settings for implied temperature rise calculation.
    """

    enabled: bool = Field(
        True,
        description="Enable temperature scoring engine",
    )
    aggregation_methods: List[str] = Field(
        default_factory=lambda: ["WATS", "TETS", "MOTS"],
        description="Temperature score aggregation methods to compute",
    )
    default_score_celsius: float = Field(
        3.2,
        ge=1.0,
        le=6.0,
        description="Default temperature score for companies without targets",
    )
    include_scope3_in_score: bool = Field(
        True,
        description="Include Scope 3 targets in temperature score calculation",
    )
    portfolio_scoring: bool = Field(
        False,
        description="Enable portfolio-level temperature scoring (for investors)",
    )
    portfolio_weighting: str = Field(
        "revenue",
        description="Portfolio weighting method: revenue, enterprise_value, market_cap, equal",
    )
    time_frames: List[str] = Field(
        default_factory=lambda: ["short", "mid", "long"],
        description="Time frames for temperature scoring (short=2030, mid=2035, long=2050)",
    )

    @field_validator("aggregation_methods")
    @classmethod
    def validate_aggregation_methods(cls, v: List[str]) -> List[str]:
        """Validate temperature aggregation methods."""
        valid = {"WATS", "TETS", "MOTS", "EOTS", "ECOTS", "AOTS"}
        invalid = [m for m in v if m not in valid]
        if invalid:
            raise ValueError(
                f"Invalid aggregation methods: {invalid}. "
                f"Valid methods: {sorted(valid)}"
            )
        return v

    @field_validator("portfolio_weighting")
    @classmethod
    def validate_weighting(cls, v: str) -> str:
        """Validate portfolio weighting method."""
        valid = {"revenue", "enterprise_value", "market_cap", "equal"}
        if v not in valid:
            raise ValueError(
                f"Invalid weighting: {v}. Must be one of: {sorted(valid)}"
            )
        return v


class DecompositionConfig(BaseModel):
    """Configuration for emissions variance decomposition.

    Defines decomposition methodology, forecasting parameters,
    and alert thresholds for off-track detection.
    """

    enabled: bool = Field(
        True,
        description="Enable variance decomposition engine",
    )
    method: DecompositionMethod = Field(
        DecompositionMethod.LMDI,
        description="Decomposition method (LMDI recommended)",
    )
    decomposition_effects: List[str] = Field(
        default_factory=lambda: ["structural", "activity", "intensity"],
        description="Effects to decompose emissions changes into",
    )
    forecast_periods: int = Field(
        4,
        ge=1,
        le=12,
        description="Number of forward periods for rolling forecast",
    )
    forecast_confidence_pct: float = Field(
        90.0,
        ge=80.0,
        le=99.0,
        description="Confidence interval for forecast projections",
    )
    alert_threshold_pct: float = Field(
        10.0,
        ge=1.0,
        le=50.0,
        description="Threshold (%) variance from target that triggers alert",
    )
    top_drivers_count: int = Field(
        10,
        ge=3,
        le=30,
        description="Number of top drivers to identify in attribution analysis",
    )
    historical_periods: int = Field(
        12,
        ge=4,
        le=40,
        description="Number of historical periods for trend analysis",
    )


class MultiEntityConfig(BaseModel):
    """Configuration for multi-entity consolidation.

    Defines consolidation method, entity structure, and reporting
    hierarchies for corporate group emissions.
    """

    enabled: bool = Field(
        False,
        description="Enable multi-entity consolidation engine",
    )
    max_entities: int = Field(
        DEFAULT_MAX_ENTITIES,
        ge=2,
        le=200,
        description="Maximum number of entities for consolidation",
    )
    consolidation_method: EntityScope = Field(
        EntityScope.OPERATIONAL_CONTROL,
        description="GHG Protocol consolidation method",
    )
    intercompany_elimination: bool = Field(
        True,
        description="Eliminate intercompany transactions (captive power, shared services)",
    )
    reporting_hierarchies: List[str] = Field(
        default_factory=lambda: ["geographic", "divisional", "legal_entity"],
        description="Reporting hierarchies for drill-down analysis",
    )
    entity_list: List[str] = Field(
        default_factory=list,
        description="List of entity names or IDs for consolidation",
    )
    ownership_threshold_pct: float = Field(
        20.0,
        ge=0.0,
        le=100.0,
        description="Minimum ownership percentage for equity share inclusion",
    )


class VCMIConfig(BaseModel):
    """Configuration for VCMI Claims Code validation.

    Defines target claim tier, criteria weights, credit quality
    parameters, and evidence requirements.
    """

    enabled: bool = Field(
        True,
        description="Enable VCMI validation engine",
    )
    target_tier: VCMITier = Field(
        VCMITier.GOLD,
        description="Target VCMI claim tier (Silver, Gold, Platinum)",
    )
    criteria_count: int = Field(
        15,
        ge=15,
        le=15,
        description="Number of VCMI criteria (fixed at 15)",
    )
    credit_quality_min_score: int = Field(
        70,
        ge=0,
        le=100,
        description="Minimum credit quality score for portfolio inclusion",
    )
    max_nature_based_pct: float = Field(
        50.0,
        ge=0.0,
        le=100.0,
        description="Maximum percentage of nature-based credits",
    )
    shift_to_removals_by_year: int = Field(
        2040,
        ge=2025,
        le=2055,
        description="Target year for majority-removal portfolio (Oxford Principles)",
    )
    preferred_registries: List[str] = Field(
        default_factory=lambda: [
            "verra",
            "gold_standard",
            "american_carbon_registry",
            "climate_action_reserve",
        ],
        description="Preferred carbon credit registries",
    )
    track_co_benefits: bool = Field(
        True,
        description="Track SDG co-benefits of offset projects",
    )
    evidence_auto_collection: bool = Field(
        True,
        description="Automatically collect evidence for criteria validation",
    )


class AssuranceConfig(BaseModel):
    """Configuration for assurance workpaper generation.

    Defines assurance level, workpaper format, methodology
    documentation preferences, and audit trail requirements.
    """

    enabled: bool = Field(
        True,
        description="Enable assurance workpaper engine",
    )
    assurance_level: AssuranceLevel = Field(
        AssuranceLevel.LIMITED,
        description="Target assurance engagement level",
    )
    target_assurance_level: AssuranceLevel = Field(
        AssuranceLevel.REASONABLE,
        description="Target assurance level to progress toward",
    )
    progression_year: int = Field(
        2028,
        ge=2025,
        le=2035,
        description="Year to achieve target assurance level",
    )
    workpaper_format: str = Field(
        "ISAE_3000",
        description="Workpaper format standard (ISAE_3000, ISAE_3410)",
    )
    methodology_documentation: bool = Field(
        True,
        description="Generate detailed methodology documentation for each calculation",
    )
    calculation_trace: bool = Field(
        True,
        description="Generate step-by-step calculation trace files",
    )
    data_lineage_maps: bool = Field(
        True,
        description="Generate visual data lineage maps",
    )
    control_evidence: bool = Field(
        True,
        description="Document review and approval control evidence",
    )
    sha256_provenance: bool = Field(
        True,
        description="Generate SHA-256 provenance hashes for all outputs",
    )
    assumption_register: bool = Field(
        True,
        description="Maintain complete assumption register for all calculations",
    )
    retention_years: int = Field(
        10,
        ge=3,
        le=15,
        description="Workpaper retention period in years",
    )

    @field_validator("workpaper_format")
    @classmethod
    def validate_workpaper_format(cls, v: str) -> str:
        """Validate workpaper format standard."""
        valid = {"ISAE_3000", "ISAE_3410"}
        if v not in valid:
            raise ValueError(
                f"Invalid workpaper format: {v}. Must be one of: {sorted(valid)}"
            )
        return v


# =============================================================================
# Main Configuration Models
# =============================================================================


class NetZeroAccelerationConfig(BaseModel):
    """Main configuration for PACK-022 Net Zero Acceleration Pack.

    This is the root configuration model that contains all sub-configurations
    for advanced net-zero planning. The sector field drives SDA pathway
    selection, Scope 3 prioritization, and supplier engagement tiering.
    """

    # Identity
    organization_name: str = Field(
        "",
        description="Legal entity name of the organization",
    )
    sector: SectorClassification = Field(
        SectorClassification.MANUFACTURING,
        description="Organization sector classification (drives engine configuration)",
    )
    region: str = Field(
        "EU",
        description="Primary operating region",
    )
    country: str = Field(
        "DE",
        description="Headquarters country (ISO 3166-1 alpha-2)",
    )
    nace_code: Optional[str] = Field(
        None,
        description="NACE industry classification code for SDA sector mapping",
    )
    gics_code: Optional[str] = Field(
        None,
        description="GICS sector classification code for benchmarking",
    )

    # Temporal
    reporting_year: int = Field(
        2025,
        ge=2020,
        le=2035,
        description="Current reporting year",
    )
    base_year: int = Field(
        DEFAULT_BASE_YEAR,
        ge=2015,
        le=2030,
        description="Base year for emissions baseline",
    )
    pack_version: str = Field(
        "1.0.0",
        description="Pack configuration version",
    )

    # Sub-configurations (10 engine configs)
    scenario: ScenarioConfig = Field(
        default_factory=ScenarioConfig,
        description="Multi-scenario Monte Carlo configuration",
    )
    pathway: PathwayConfig = Field(
        default_factory=PathwayConfig,
        description="SDA/ACA/FLAG pathway configuration",
    )
    supplier: SupplierConfig = Field(
        default_factory=SupplierConfig,
        description="Supplier engagement program configuration",
    )
    scope3: Scope3Config = Field(
        default_factory=Scope3Config,
        description="Activity-based Scope 3 configuration",
    )
    finance: FinanceConfig = Field(
        default_factory=FinanceConfig,
        description="Climate transition finance configuration",
    )
    temperature: TemperatureConfig = Field(
        default_factory=TemperatureConfig,
        description="SBTi Temperature Rating configuration",
    )
    decomposition: DecompositionConfig = Field(
        default_factory=DecompositionConfig,
        description="Variance decomposition configuration",
    )
    multi_entity: MultiEntityConfig = Field(
        default_factory=MultiEntityConfig,
        description="Multi-entity consolidation configuration",
    )
    vcmi: VCMIConfig = Field(
        default_factory=VCMIConfig,
        description="VCMI Claims Code validation configuration",
    )
    assurance: AssuranceConfig = Field(
        default_factory=AssuranceConfig,
        description="Assurance workpaper generation configuration",
    )

    @model_validator(mode="after")
    def validate_base_year_before_reporting(self) -> "NetZeroAccelerationConfig":
        """Ensure base year is not after reporting year."""
        if self.base_year > self.reporting_year:
            raise ValueError(
                f"base_year ({self.base_year}) must not be after "
                f"reporting_year ({self.reporting_year})"
            )
        return self

    @model_validator(mode="after")
    def validate_sda_sector_alignment(self) -> "NetZeroAccelerationConfig":
        """Warn if SDA pathway is enabled for non-SDA sectors."""
        sda_recommended = {
            SectorClassification.HEAVY_INDUSTRY,
            SectorClassification.POWER_UTILITIES,
            SectorClassification.MANUFACTURING,
            SectorClassification.TRANSPORT,
            SectorClassification.REAL_ESTATE,
        }
        if self.pathway.sda_enabled and self.sector not in sda_recommended:
            logger.warning(
                "SDA pathway enabled for %s sector which may not have "
                "SDA sector benchmarks. Consider ACA pathway instead.",
                self.sector.value,
            )
        return self

    @model_validator(mode="after")
    def validate_heavy_industry_requires_sda(self) -> "NetZeroAccelerationConfig":
        """Warn if heavy industry is not using SDA."""
        sda_mandatory = {
            SectorClassification.HEAVY_INDUSTRY,
            SectorClassification.POWER_UTILITIES,
        }
        if self.sector in sda_mandatory:
            if self.pathway.methodology != PathwayMethodology.SDA:
                logger.warning(
                    "SDA pathway is strongly recommended for %s sector. "
                    "Current methodology: %s",
                    self.sector.value,
                    self.pathway.methodology.value,
                )
        return self

    @model_validator(mode="after")
    def validate_financial_services_pcaf(self) -> "NetZeroAccelerationConfig":
        """Warn if financial services sector does not enable PCAF."""
        if self.sector == SectorClassification.FINANCIAL_SERVICES:
            if not self.scope3.pcaf_enabled:
                logger.warning(
                    "PCAF methodology is recommended for financial services "
                    "Scope 3 Category 15 (investments). Enable scope3.pcaf_enabled."
                )
        return self

    def get_enabled_engines(self) -> List[str]:
        """Return list of engine identifiers that are enabled.

        Returns:
            Sorted list of enabled engine identifier strings.
        """
        engines = ["scenario_modeling", "scope3_activity"]

        if self.pathway.sda_enabled:
            engines.append("sda_pathway")

        if self.supplier.enabled:
            engines.append("supplier_engagement")

        if self.finance.enabled:
            engines.append("climate_finance")

        if self.temperature.enabled:
            engines.append("temperature_scoring")

        if self.decomposition.enabled:
            engines.append("variance_decomposition")

        if self.multi_entity.enabled:
            engines.append("multi_entity")

        if self.vcmi.enabled:
            engines.append("vcmi_validation")

        if self.assurance.enabled:
            engines.append("assurance_workpaper")

        return sorted(set(engines))


# =============================================================================
# Pack Configuration Wrapper
# =============================================================================


class PackConfig(BaseModel):
    """Top-level pack configuration wrapper for PACK-022.

    Handles preset loading, environment variable overrides, and
    configuration merging. Provides SHA-256 config hashing for
    provenance tracking.

    Example:
        >>> config = PackConfig.from_preset("heavy_industry")
        >>> print(config.pack.sector)
        SectorClassification.HEAVY_INDUSTRY
        >>> print(config.pack.pathway.methodology)
        PathwayMethodology.SDA
    """

    pack: NetZeroAccelerationConfig = Field(
        default_factory=NetZeroAccelerationConfig,
        description="Main Net Zero Acceleration configuration",
    )
    preset_name: Optional[str] = Field(
        None,
        description="Name of the loaded preset",
    )
    config_version: str = Field(
        "1.0.0",
        description="Configuration schema version",
    )
    pack_id: str = Field(
        "PACK-022-net-zero-acceleration",
        description="Pack identifier",
    )

    @classmethod
    def from_preset(
        cls,
        preset_name: str,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> "PackConfig":
        """Load configuration from a named preset.

        Args:
            preset_name: Name of the preset (heavy_industry, power_utilities,
                manufacturing, transport_logistics, financial_services,
                real_estate, consumer_goods, technology).
            overrides: Optional dictionary of configuration overrides.

        Returns:
            PackConfig instance with preset values applied.

        Raises:
            FileNotFoundError: If preset YAML file does not exist.
            ValueError: If preset_name is not in SUPPORTED_PRESETS.
        """
        if preset_name not in SUPPORTED_PRESETS:
            raise ValueError(
                f"Unknown preset: {preset_name}. "
                f"Available presets: {sorted(SUPPORTED_PRESETS.keys())}"
            )

        preset_path = CONFIG_DIR / "presets" / f"{preset_name}.yaml"
        if not preset_path.exists():
            raise FileNotFoundError(
                f"Preset file not found: {preset_path}. "
                f"Run setup wizard to generate presets."
            )

        with open(preset_path, "r", encoding="utf-8") as f:
            preset_data = yaml.safe_load(f) or {}

        # Apply environment variable overrides
        env_overrides = _get_env_overrides("NET_ZERO_ACCEL_")
        if env_overrides:
            preset_data = _merge_config(preset_data, env_overrides)

        # Apply explicit overrides
        if overrides:
            preset_data = _merge_config(preset_data, overrides)

        pack_config = NetZeroAccelerationConfig(**preset_data)
        return cls(pack=pack_config, preset_name=preset_name)

    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> "PackConfig":
        """Load configuration from a YAML file.

        Args:
            yaml_path: Path to YAML configuration file.

        Returns:
            PackConfig instance with YAML values applied.

        Raises:
            FileNotFoundError: If YAML file does not exist.
        """
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {yaml_path}")

        with open(yaml_path, "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f) or {}

        pack_config = NetZeroAccelerationConfig(**config_data)
        return cls(pack=pack_config)

    def get_config_hash(self) -> str:
        """Generate SHA-256 hash of the current configuration for provenance.

        Returns:
            Hex-encoded SHA-256 hash string.
        """
        config_json = self.model_dump_json(indent=None)
        return _compute_hash(config_json)

    def validate_config(self) -> List[str]:
        """Cross-field validation returning warnings.

        Returns:
            List of warning messages (empty if fully valid).
        """
        return validate_config(self.pack)


# =============================================================================
# Utility Functions
# =============================================================================


def load_config(yaml_path: Union[str, Path]) -> PackConfig:
    """Load configuration from a YAML file.

    Convenience wrapper around PackConfig.from_yaml().

    Args:
        yaml_path: Path to YAML configuration file.

    Returns:
        PackConfig instance.
    """
    return PackConfig.from_yaml(yaml_path)


def load_preset(
    preset_name: str,
    overrides: Optional[Dict[str, Any]] = None,
) -> PackConfig:
    """Load a named preset configuration.

    Convenience wrapper around PackConfig.from_preset().

    Args:
        preset_name: Name of the preset to load.
        overrides: Optional configuration overrides.

    Returns:
        PackConfig instance with preset applied.
    """
    return PackConfig.from_preset(preset_name, overrides)


def get_sector_defaults(
    sector: Union[str, SectorClassification],
) -> NetZeroAccelerationConfig:
    """Get default configuration for a given sector.

    Creates a NetZeroAccelerationConfig with sector-appropriate defaults
    for pathway methodology, SDA sectors, Scope 3 priorities, and
    supplier engagement parameters.

    Args:
        sector: Sector classification enum or string value.

    Returns:
        NetZeroAccelerationConfig with sector defaults applied.
    """
    if isinstance(sector, str):
        sector = SectorClassification(sector)

    sector_info = SECTOR_INFO.get(sector.value, {})
    recommended_pathway = sector_info.get("recommended_pathway", "ACA")
    sda_sectors = sector_info.get("sda_sectors", [])
    priority_cats = PRIORITY_SCOPE3_BY_SECTOR.get(sector.value, [1, 4, 5])

    pathway_method = (
        PathwayMethodology(recommended_pathway)
        if recommended_pathway in PathwayMethodology.__members__
        else PathwayMethodology.ACA
    )
    sda_enabled = pathway_method == PathwayMethodology.SDA and len(sda_sectors) > 0

    # Financial services: enable PCAF
    pcaf_enabled = sector == SectorClassification.FINANCIAL_SERVICES
    pcaf_asset_classes = (
        ["listed_equity", "corporate_bonds", "business_loans"]
        if pcaf_enabled
        else []
    )

    # Multi-entity default for financial services and heavy industry
    multi_entity_enabled = sector in {
        SectorClassification.FINANCIAL_SERVICES,
        SectorClassification.HEAVY_INDUSTRY,
        SectorClassification.POWER_UTILITIES,
        SectorClassification.REAL_ESTATE,
    }

    return NetZeroAccelerationConfig(
        sector=sector,
        pathway=PathwayConfig(
            methodology=pathway_method,
            sda_enabled=sda_enabled,
            sda_sectors=sda_sectors,
        ),
        scope3=Scope3Config(
            activity_based_categories=priority_cats,
            pcaf_enabled=pcaf_enabled,
            pcaf_asset_classes=pcaf_asset_classes,
        ),
        multi_entity=MultiEntityConfig(
            enabled=multi_entity_enabled,
        ),
    )


def _merge_config(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge two dictionaries, with override taking precedence.

    Args:
        base: Base configuration dictionary.
        override: Override dictionary (values take precedence).

    Returns:
        Merged dictionary.
    """
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _merge_config(result[key], value)
        else:
            result[key] = value
    return result


def merge_config(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Public deep merge two dictionaries, with override taking precedence.

    Args:
        base: Base configuration dictionary.
        override: Override dictionary (values take precedence).

    Returns:
        Merged dictionary.
    """
    return _merge_config(base, override)


def _get_env_overrides(prefix: str) -> Dict[str, Any]:
    """Load configuration overrides from environment variables.

    Environment variables prefixed with the given prefix are loaded and
    mapped to configuration keys. Nested keys use double underscore.

    Example:
        NET_ZERO_ACCEL_REPORTING_YEAR=2026
        NET_ZERO_ACCEL_PATHWAY__METHODOLOGY=SDA
        NET_ZERO_ACCEL_SCENARIO__MONTE_CARLO_RUNS=2000

    Args:
        prefix: Environment variable prefix to search for.

    Returns:
        Dictionary of parsed overrides.
    """
    overrides: Dict[str, Any] = {}
    for key, value in os.environ.items():
        if key.startswith(prefix):
            config_key = key[len(prefix):].lower()
            parts = config_key.split("__")
            current = overrides
            for part in parts[:-1]:
                current = current.setdefault(part, {})
            # Parse value types
            if value.lower() in ("true", "yes", "1"):
                current[parts[-1]] = True
            elif value.lower() in ("false", "no", "0"):
                current[parts[-1]] = False
            else:
                try:
                    current[parts[-1]] = int(value)
                except ValueError:
                    try:
                        current[parts[-1]] = float(value)
                    except ValueError:
                        current[parts[-1]] = value
    return overrides


def get_env_overrides(prefix: str) -> Dict[str, Any]:
    """Public wrapper for loading environment variable overrides.

    Args:
        prefix: Environment variable prefix to search for.

    Returns:
        Dictionary of parsed overrides.
    """
    return _get_env_overrides(prefix)


def validate_config(config: NetZeroAccelerationConfig) -> List[str]:
    """Validate an acceleration configuration and return any warnings.

    Performs cross-field validation beyond what Pydantic validators cover.
    Returns advisory warnings, not hard errors.

    Args:
        config: NetZeroAccelerationConfig instance to validate.

    Returns:
        List of warning messages (empty if fully valid).
    """
    warnings: List[str] = []

    # Check organization name is set
    if not config.organization_name:
        warnings.append(
            "Organization name is empty. Set organization_name for meaningful reports."
        )

    # Check SDA configuration for SDA-mandatory sectors
    sda_mandatory = {
        SectorClassification.HEAVY_INDUSTRY,
        SectorClassification.POWER_UTILITIES,
    }
    if config.sector in sda_mandatory and not config.pathway.sda_enabled:
        warnings.append(
            f"SDA pathway is strongly recommended for {config.sector.value}. "
            f"Enable pathway.sda_enabled with appropriate sda_sectors."
        )

    # Check SDA sectors are specified when SDA is enabled
    if config.pathway.sda_enabled and not config.pathway.sda_sectors:
        warnings.append(
            "SDA pathway is enabled but no sda_sectors specified. "
            "Add sector codes (e.g., CEMENT, STEEL, POWER) to pathway.sda_sectors."
        )

    # Check Scope 3 coverage
    if len(config.scope3.categories) < 5:
        warnings.append(
            "Fewer than 5 Scope 3 categories included. Acceleration-grade "
            "programs typically cover all material categories. Consider expanding."
        )

    # Check activity-based Scope 3 categories
    if not config.scope3.activity_based_categories:
        warnings.append(
            "No activity-based Scope 3 categories specified. "
            "Professional tier should upgrade top categories from spend-based."
        )

    # Check supplier engagement
    if config.supplier.enabled and config.supplier.top_supplier_count < 20:
        warnings.append(
            f"Top supplier count ({config.supplier.top_supplier_count}) is low "
            f"for acceleration-grade programs. Consider engaging at least 50 suppliers."
        )

    # Check financial services PCAF
    if config.sector == SectorClassification.FINANCIAL_SERVICES:
        if not config.scope3.pcaf_enabled:
            warnings.append(
                "PCAF methodology should be enabled for financial services. "
                "Set scope3.pcaf_enabled=true for Category 15 (investments)."
            )
        if not config.temperature.portfolio_scoring:
            warnings.append(
                "Portfolio temperature scoring recommended for financial services. "
                "Enable temperature.portfolio_scoring for investor disclosure."
            )

    # Check VCMI credit quality
    if config.vcmi.enabled and config.vcmi.credit_quality_min_score < 50:
        warnings.append(
            "VCMI credit quality minimum score is very low. "
            "Gold/Platinum tiers require high-quality credits (score >= 70)."
        )

    # Check assurance progression
    if config.assurance.enabled:
        if (config.assurance.assurance_level == AssuranceLevel.LIMITED
                and config.assurance.target_assurance_level == AssuranceLevel.LIMITED):
            warnings.append(
                "No progression to reasonable assurance configured. "
                "CSRD and ESRS will require reasonable assurance by 2028+."
            )

    # Check Monte Carlo reproducibility
    if config.scenario.random_seed == 0:
        warnings.append(
            "Monte Carlo random seed is 0. Set a non-zero seed for "
            "reproducible simulation results."
        )

    # Check near-term target year
    if config.pathway.near_term_target_year > 2035:
        warnings.append(
            "SBTi requires near-term targets within 5-10 years of submission. "
            f"Near-term year {config.pathway.near_term_target_year} may be too far."
        )

    # Check long-term target year
    if config.pathway.long_term_target_year > 2050:
        warnings.append(
            "SBTi Net-Zero Standard requires net-zero by no later than 2050. "
            f"Long-term year {config.pathway.long_term_target_year} exceeds this."
        )

    return warnings


def get_sector_info(sector: Union[str, SectorClassification]) -> Dict[str, Any]:
    """Get detailed information about an organization sector.

    Args:
        sector: Sector classification enum or string value.

    Returns:
        Dictionary with name, recommended pathway, SDA sectors,
        typical scope split, and key decarbonization levers.
    """
    key = sector.value if isinstance(sector, SectorClassification) else sector
    return SECTOR_INFO.get(key, {
        "name": key,
        "recommended_pathway": "ACA",
        "sda_sectors": [],
        "typical_scope_split": "Varies by sub-sector",
        "key_levers": ["energy efficiency", "renewable procurement"],
    })


def get_sda_intensity_metric(sector: str) -> str:
    """Get the SDA intensity metric for a given sector.

    Args:
        sector: SDA sector code (e.g., POWER, CEMENT, STEEL).

    Returns:
        Intensity metric string (e.g., tCO2e/MWh).
    """
    return SDA_INTENSITY_METRICS.get(sector, "tCO2e/unit")


def get_vcmi_tier_thresholds(tier: Union[str, VCMITier]) -> Dict[str, float]:
    """Get VCMI Claims Code tier threshold scores.

    Args:
        tier: VCMI tier enum or string value.

    Returns:
        Dictionary with minimum scores for each pillar.
    """
    key = tier.value if isinstance(tier, VCMITier) else tier
    return VCMI_TIER_THRESHOLDS.get(key, VCMI_TIER_THRESHOLDS["SILVER"])


def get_temperature_regression(scope: str) -> Dict[str, float]:
    """Get SBTi temperature regression coefficients for a scope.

    Args:
        scope: Scope identifier (scope_1_2 or scope_3).

    Returns:
        Dictionary with intercept and slope coefficients.
    """
    return TEMPERATURE_REGRESSION.get(scope, TEMPERATURE_REGRESSION["scope_1_2"])


def get_sbti_reduction_rate(ambition: Union[str, TemperatureTarget]) -> Dict[str, float]:
    """Get SBTi minimum annual reduction rates for an ambition level.

    Args:
        ambition: Temperature target enum or string value.

    Returns:
        Dictionary with scope_1_2_linear_annual, scope_3_linear_annual,
        and long_term_reduction_pct.
    """
    key = ambition.value if isinstance(ambition, TemperatureTarget) else ambition
    return SBTI_REDUCTION_RATES.get(key, SBTI_REDUCTION_RATES["CELSIUS_1_5"])


def get_gwp100(gas: str) -> int:
    """Get IPCC AR6 GWP100 value for a greenhouse gas.

    Args:
        gas: Greenhouse gas identifier (CO2, CH4, N2O, etc.).

    Returns:
        GWP100 value (dimensionless, relative to CO2).
    """
    return IPCC_AR6_GWP100.get(gas.upper(), 0)


def list_available_presets() -> Dict[str, str]:
    """List all available configuration presets.

    Returns:
        Dictionary mapping preset names to descriptions.
    """
    return SUPPORTED_PRESETS.copy()


def list_sda_sectors() -> Dict[str, str]:
    """List all supported SDA sectors.

    Returns:
        Dictionary mapping sector codes to display names.
    """
    return SDA_SECTORS.copy()
