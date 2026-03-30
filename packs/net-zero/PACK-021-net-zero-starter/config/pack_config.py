"""
PACK-021 Net Zero Starter Pack - Configuration Manager

This module implements the NetZeroStarterConfig and PackConfig classes that load,
merge, and validate all configuration for the Net Zero Starter Pack. It provides
comprehensive Pydantic v2 models for every aspect of net-zero planning:
GHG baseline calculation, SBTi target setting, decarbonization pathway modeling,
MACC analysis, gap-to-target tracking, carbon credit portfolio management,
residual emissions estimation, and multi-framework reporting.

Organization Sectors:
    - MANUFACTURING: Industrial production, process heat, heavy machinery
    - SERVICES: Professional services, consulting, financial services offices
    - RETAIL: Consumer goods retail, supply chain focused
    - ENERGY: Power generation, oil & gas, utilities
    - TECHNOLOGY: Software, hardware, data centers, cloud services
    - AGRICULTURE: Farming, food production, land use
    - TRANSPORT: Logistics, freight, passenger transport
    - CONSTRUCTION: Building, infrastructure, real estate development
    - MINING: Extraction, minerals, quarrying
    - FINANCIAL_SERVICES: Banking, insurance, asset management
    - HEALTHCARE: Hospitals, pharmaceuticals, medical devices
    - OTHER: Other sectors

Boundary Methods:
    - OPERATIONAL_CONTROL: GHG Protocol operational control approach
    - FINANCIAL_CONTROL: GHG Protocol financial control approach
    - EQUITY_SHARE: GHG Protocol equity share approach

SBTi Pathways:
    - ACA: Absolute Contraction Approach (cross-sector)
    - SDA: Sectoral Decarbonization Approach (homogeneous sectors)
    - FLAG: Forest, Land and Agriculture (land-intensive sectors)

Configuration Merge Order (later overrides earlier):
    1. Base pack.yaml manifest
    2. Preset YAML (manufacturing / services / retail / energy / technology / sme_general)
    3. Environment overrides (NET_ZERO_STARTER_* environment variables)
    4. Explicit runtime overrides

Regulatory Context:
    - SBTi Net-Zero Standard v1.2 (2024)
    - GHG Protocol Corporate Standard (revised)
    - GHG Protocol Scope 3 Standard
    - IPCC AR6 WG3 (Mitigation, 2022)
    - Paris Agreement Article 4
    - ISO 14064-1:2018
    - ISO 14068-1:2023
    - VCMI Claims Code v1.0 (2023)
    - Oxford Principles for Net Zero Aligned Carbon Offsetting (2024)

Example:
    >>> config = PackConfig.from_preset("manufacturing")
    >>> print(config.pack.organization.sector)
    OrganizationSector.MANUFACTURING
    >>> print(config.pack.target.ambition_level)
    AmbitionLevel.CELSIUS_1_5
"""

import hashlib
import logging
import os
from datetime import date
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator
from greenlang.schemas.enums import ReportFormat

logger = logging.getLogger(__name__)

# Base directory for all pack configuration files
PACK_BASE_DIR = Path(__file__).parent.parent
CONFIG_DIR = Path(__file__).parent

# =============================================================================
# Constants
# =============================================================================

DEFAULT_BASE_YEAR: int = 2024
DEFAULT_TARGET_YEAR_NEAR: int = 2030
DEFAULT_TARGET_YEAR_LONG: int = 2050
DEFAULT_SCOPE3_CATEGORIES: List[int] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12]

SUPPORTED_PRESETS: Dict[str, str] = {
    "manufacturing": "Manufacturing sector with high Scope 1 and process emissions",
    "services": "Professional services with low Scope 1, high Scope 3 travel/procurement",
    "retail": "Retail sector with supply chain focus and dominant Scope 3 Cat 1",
    "energy": "Energy sector with very high Scope 1, SDA pathway mandatory",
    "technology": "Technology sector with low Scope 1+2, high Scope 3 (cloud, hardware)",
    "sme_general": "Simplified SME configuration with minimal engine set",
}


# =============================================================================
# Enums - Net-zero specific enumeration types (12 enums)
# =============================================================================


class OrganizationSector(str, Enum):
    """Organization sector classification."""

    MANUFACTURING = "MANUFACTURING"
    SERVICES = "SERVICES"
    RETAIL = "RETAIL"
    ENERGY = "ENERGY"
    TECHNOLOGY = "TECHNOLOGY"
    AGRICULTURE = "AGRICULTURE"
    TRANSPORT = "TRANSPORT"
    CONSTRUCTION = "CONSTRUCTION"
    MINING = "MINING"
    FINANCIAL_SERVICES = "FINANCIAL_SERVICES"
    HEALTHCARE = "HEALTHCARE"
    OTHER = "OTHER"


class BoundaryMethod(str, Enum):
    """GHG Protocol organizational boundary method."""

    OPERATIONAL_CONTROL = "OPERATIONAL_CONTROL"
    FINANCIAL_CONTROL = "FINANCIAL_CONTROL"
    EQUITY_SHARE = "EQUITY_SHARE"


class AmbitionLevel(str, Enum):
    """SBTi target ambition level."""

    CELSIUS_1_5 = "CELSIUS_1_5"
    WELL_BELOW_2 = "WELL_BELOW_2"
    CELSIUS_2 = "CELSIUS_2"


class PathwayType(str, Enum):
    """SBTi target-setting pathway type."""

    ACA = "ACA"
    SDA = "SDA"
    FLAG = "FLAG"


class TargetTimeframe(str, Enum):
    """Target timeframe classification."""

    NEAR_TERM = "NEAR_TERM"
    LONG_TERM = "LONG_TERM"
    NET_ZERO = "NET_ZERO"


class DataSourceType(str, Enum):
    """Data source type for emissions data."""

    MANUAL = "MANUAL"
    ERP_SAP = "ERP_SAP"
    ERP_ORACLE = "ERP_ORACLE"
    ERP_WORKDAY = "ERP_WORKDAY"
    ERP_DYNAMICS = "ERP_DYNAMICS"
    API = "API"
    FILE_UPLOAD = "FILE_UPLOAD"


class Scope3Method(str, Enum):
    """Scope 3 calculation methodology."""

    SPEND_BASED = "SPEND_BASED"
    ACTIVITY_BASED = "ACTIVITY_BASED"
    HYBRID = "HYBRID"


class OffsetStrategy(str, Enum):
    """Carbon offset strategy type."""

    COMPENSATION_ONLY = "COMPENSATION_ONLY"
    NEUTRALIZATION_ONLY = "NEUTRALIZATION_ONLY"
    BOTH = "BOTH"


class MaturityAssessment(str, Enum):
    """Net-zero maturity assessment mode."""

    FULL = "FULL"
    QUICK = "QUICK"
    SKIP = "SKIP"


class ERPType(str, Enum):
    """ERP system type."""

    SAP = "SAP"
    ORACLE = "ORACLE"
    WORKDAY = "WORKDAY"
    DYNAMICS_365 = "DYNAMICS_365"
    NETSUITE = "NETSUITE"
    NONE = "NONE"


class OrganizationSize(str, Enum):
    """Organization size classification."""

    MICRO = "MICRO"
    SMALL = "SMALL"
    MEDIUM = "MEDIUM"
    LARGE = "LARGE"
    ENTERPRISE = "ENTERPRISE"


# =============================================================================
# Reference Data Constants
# =============================================================================

# Sector-specific Scope 3 priorities (category -> priority: CRITICAL/HIGH/MEDIUM/LOW)
SECTOR_SCOPE3_PRIORITY: Dict[str, Dict[int, str]] = {
    "MANUFACTURING": {
        1: "CRITICAL", 2: "MEDIUM", 3: "HIGH", 4: "HIGH", 5: "HIGH",
        6: "LOW", 7: "LOW", 8: "MEDIUM", 9: "MEDIUM", 10: "MEDIUM",
        11: "MEDIUM", 12: "LOW", 13: "LOW", 14: "LOW", 15: "LOW",
    },
    "SERVICES": {
        1: "HIGH", 2: "LOW", 3: "MEDIUM", 4: "LOW", 5: "LOW",
        6: "CRITICAL", 7: "HIGH", 8: "MEDIUM", 9: "LOW", 10: "LOW",
        11: "LOW", 12: "LOW", 13: "LOW", 14: "LOW", 15: "LOW",
    },
    "RETAIL": {
        1: "CRITICAL", 2: "LOW", 3: "MEDIUM", 4: "HIGH", 5: "HIGH",
        6: "LOW", 7: "LOW", 8: "MEDIUM", 9: "HIGH", 10: "LOW",
        11: "MEDIUM", 12: "HIGH", 13: "LOW", 14: "MEDIUM", 15: "LOW",
    },
    "ENERGY": {
        1: "HIGH", 2: "MEDIUM", 3: "CRITICAL", 4: "MEDIUM", 5: "LOW",
        6: "LOW", 7: "LOW", 8: "MEDIUM", 9: "MEDIUM", 10: "LOW",
        11: "CRITICAL", 12: "LOW", 13: "LOW", 14: "LOW", 15: "MEDIUM",
    },
    "TECHNOLOGY": {
        1: "CRITICAL", 2: "HIGH", 3: "MEDIUM", 4: "MEDIUM", 5: "LOW",
        6: "MEDIUM", 7: "MEDIUM", 8: "HIGH", 9: "LOW", 10: "LOW",
        11: "CRITICAL", 12: "MEDIUM", 13: "LOW", 14: "LOW", 15: "LOW",
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

# SBTi minimum annual reduction rates by ambition level (% per year)
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

# Priority Scope 3 categories by sector
PRIORITY_SCOPE3_BY_SECTOR: Dict[str, List[int]] = {
    "MANUFACTURING": [1, 3, 4, 5, 11],
    "SERVICES": [1, 6, 7],
    "RETAIL": [1, 4, 9, 12],
    "ENERGY": [3, 11, 15],
    "TECHNOLOGY": [1, 2, 8, 11],
    "AGRICULTURE": [1, 4, 5],
    "TRANSPORT": [1, 3, 4, 9],
    "CONSTRUCTION": [1, 2, 4, 5],
    "MINING": [1, 3, 4, 5],
    "FINANCIAL_SERVICES": [1, 6, 7, 15],
    "HEALTHCARE": [1, 2, 4, 5],
    "OTHER": [1, 4, 5],
}

# Sector display names and SBTi pathway recommendations
SECTOR_INFO: Dict[str, Dict[str, Any]] = {
    "MANUFACTURING": {
        "name": "Manufacturing",
        "recommended_pathway": "SDA",
        "typical_scope_split": "Scope 1: 30-60%, Scope 2: 10-25%, Scope 3: 30-50%",
        "key_levers": ["process optimization", "fuel switching", "energy efficiency"],
    },
    "SERVICES": {
        "name": "Professional Services",
        "recommended_pathway": "ACA",
        "typical_scope_split": "Scope 1: 5-15%, Scope 2: 10-25%, Scope 3: 60-85%",
        "key_levers": ["renewable procurement", "travel reduction", "remote work"],
    },
    "RETAIL": {
        "name": "Retail & Consumer Goods",
        "recommended_pathway": "ACA",
        "typical_scope_split": "Scope 1: 5-15%, Scope 2: 10-20%, Scope 3: 70-85%",
        "key_levers": ["supplier engagement", "logistics optimization", "refrigerant transition"],
    },
    "ENERGY": {
        "name": "Energy & Utilities",
        "recommended_pathway": "SDA",
        "typical_scope_split": "Scope 1: 60-80%, Scope 2: 2-5%, Scope 3: 15-35%",
        "key_levers": ["renewable capacity", "CCS", "grid decarbonization"],
    },
    "TECHNOLOGY": {
        "name": "Technology & Software",
        "recommended_pathway": "ACA",
        "typical_scope_split": "Scope 1: 2-8%, Scope 2: 15-35%, Scope 3: 60-80%",
        "key_levers": ["renewable data centers", "hardware lifecycle", "cloud efficiency"],
    },
    "AGRICULTURE": {
        "name": "Agriculture & Food Production",
        "recommended_pathway": "FLAG",
        "typical_scope_split": "Scope 1: 40-70%, Scope 2: 5-15%, Scope 3: 20-45%",
        "key_levers": ["land management", "fertilizer optimization", "livestock management"],
    },
    "TRANSPORT": {
        "name": "Transport & Logistics",
        "recommended_pathway": "SDA",
        "typical_scope_split": "Scope 1: 50-80%, Scope 2: 5-15%, Scope 3: 15-40%",
        "key_levers": ["fleet electrification", "route optimization", "modal shift"],
    },
    "CONSTRUCTION": {
        "name": "Construction & Real Estate",
        "recommended_pathway": "SDA",
        "typical_scope_split": "Scope 1: 15-30%, Scope 2: 10-20%, Scope 3: 55-75%",
        "key_levers": ["low-carbon materials", "energy efficient buildings", "circular design"],
    },
    "MINING": {
        "name": "Mining & Extraction",
        "recommended_pathway": "SDA",
        "typical_scope_split": "Scope 1: 40-65%, Scope 2: 15-30%, Scope 3: 15-35%",
        "key_levers": ["fleet electrification", "renewable power", "process optimization"],
    },
    "FINANCIAL_SERVICES": {
        "name": "Financial Services",
        "recommended_pathway": "ACA",
        "typical_scope_split": "Scope 1: 3-8%, Scope 2: 10-20%, Scope 3: 75-90% (financed)",
        "key_levers": ["portfolio decarbonization", "green finance", "office efficiency"],
    },
    "HEALTHCARE": {
        "name": "Healthcare & Pharmaceuticals",
        "recommended_pathway": "ACA",
        "typical_scope_split": "Scope 1: 10-25%, Scope 2: 15-30%, Scope 3: 50-70%",
        "key_levers": ["energy efficiency", "anesthetic gas management", "sustainable procurement"],
    },
    "OTHER": {
        "name": "Other",
        "recommended_pathway": "ACA",
        "typical_scope_split": "Varies by sub-sector",
        "key_levers": ["energy efficiency", "renewable procurement", "supplier engagement"],
    },
}


# =============================================================================
# Pydantic Sub-Config Models (8 models)
# =============================================================================


class OrganizationConfig(BaseModel):
    """Configuration for the organization profile.

    Defines the company identity, sector, size, and operational characteristics
    that drive engine configuration and pathway selection.
    """

    name: str = Field(
        "",
        description="Legal entity name of the organization",
    )
    sector: OrganizationSector = Field(
        OrganizationSector.MANUFACTURING,
        description="Primary business sector (drives pathway and priority selection)",
    )
    size: OrganizationSize = Field(
        OrganizationSize.LARGE,
        description="Organization size classification",
    )
    region: str = Field(
        "EU",
        description="Primary operating region (ISO 3166 or continent code)",
    )
    country: str = Field(
        "DE",
        description="Headquarters country (ISO 3166-1 alpha-2)",
    )
    revenue_eur: Optional[float] = Field(
        None,
        ge=0,
        description="Annual revenue in EUR for intensity metric calculation",
    )
    employee_count: Optional[int] = Field(
        None,
        ge=0,
        description="Total number of employees for intensity metrics",
    )
    fiscal_year_end: str = Field(
        "12-31",
        description="Fiscal year end date in MM-DD format",
    )
    nace_code: Optional[str] = Field(
        None,
        description="NACE industry classification code for SDA pathway matching",
    )
    gics_code: Optional[str] = Field(
        None,
        description="GICS sector classification code for benchmarking",
    )
    erp_system: ERPType = Field(
        ERPType.NONE,
        description="ERP system in use for data integration",
    )

    @field_validator("fiscal_year_end")
    @classmethod
    def validate_fiscal_year_end(cls, v: str) -> str:
        """Validate fiscal year end is in MM-DD format."""
        parts = v.split("-")
        if len(parts) != 2:
            raise ValueError(
                f"fiscal_year_end must be in MM-DD format, got: {v}"
            )
        month, day = int(parts[0]), int(parts[1])
        if month < 1 or month > 12:
            raise ValueError(
                f"Invalid month in fiscal_year_end: {month}. Must be 1-12."
            )
        if day < 1 or day > 31:
            raise ValueError(
                f"Invalid day in fiscal_year_end: {day}. Must be 1-31."
            )
        return v


class BoundaryConfig(BaseModel):
    """Configuration for GHG Protocol organizational boundary.

    Defines the consolidation approach and entities included in the
    GHG inventory boundary per GHG Protocol Corporate Standard.
    """

    method: BoundaryMethod = Field(
        BoundaryMethod.OPERATIONAL_CONTROL,
        description="GHG Protocol boundary approach",
    )
    entities_included: List[str] = Field(
        default_factory=list,
        description="List of legal entities or facilities included in boundary",
    )
    reporting_currency: str = Field(
        "EUR",
        description="Reporting currency for financial metrics (ISO 4217)",
    )
    include_joint_ventures: bool = Field(
        False,
        description="Include joint ventures in boundary (relevant for equity share)",
    )
    base_year_recalculation_threshold_pct: float = Field(
        5.0,
        ge=0.0,
        le=100.0,
        description="Threshold (%) for structural change triggering base year recalculation",
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


class ScopeConfig(BaseModel):
    """Configuration for Scope 1, 2, and 3 emissions calculation.

    Defines which scopes and Scope 3 categories are included, and the
    methodology for each.
    """

    include_scope1: bool = Field(
        True,
        description="Include Scope 1 direct emissions",
    )
    include_scope2: bool = Field(
        True,
        description="Include Scope 2 indirect emissions (purchased energy)",
    )
    include_scope3: bool = Field(
        True,
        description="Include Scope 3 value chain emissions",
    )
    scope2_methods: List[str] = Field(
        default_factory=lambda: ["location_based", "market_based"],
        description="Scope 2 calculation methods to apply",
    )
    scope3_categories: List[int] = Field(
        default_factory=lambda: DEFAULT_SCOPE3_CATEGORIES.copy(),
        description="Scope 3 categories (1-15) to include in inventory",
    )
    scope3_method: Scope3Method = Field(
        Scope3Method.HYBRID,
        description="Default Scope 3 calculation methodology",
    )
    scope3_category_methods: Dict[str, str] = Field(
        default_factory=lambda: {
            "default": "spend_based",
        },
        description="Override calculation method per Scope 3 category (e.g., cat_1: activity_based)",
    )
    scope3_materiality_threshold_pct: float = Field(
        1.0,
        ge=0.0,
        le=100.0,
        description="Threshold (%) below which a Scope 3 category is considered immaterial",
    )

    @field_validator("scope3_categories")
    @classmethod
    def validate_scope3_categories(cls, v: List[int]) -> List[int]:
        """Validate Scope 3 category numbers are 1-15."""
        invalid = [c for c in v if c < 1 or c > 15]
        if invalid:
            raise ValueError(
                f"Invalid Scope 3 categories: {invalid}. Must be 1-15."
            )
        return sorted(set(v))

    @field_validator("scope2_methods")
    @classmethod
    def validate_scope2_methods(cls, v: List[str]) -> List[str]:
        """Validate Scope 2 methods are recognized."""
        valid = {"location_based", "market_based"}
        invalid = [m for m in v if m not in valid]
        if invalid:
            raise ValueError(
                f"Invalid Scope 2 methods: {invalid}. Must be: {sorted(valid)}"
            )
        return v


class TargetConfig(BaseModel):
    """Configuration for science-based target setting.

    Defines ambition level, pathway type, timeframes, and coverage
    thresholds aligned with SBTi Net-Zero Standard v1.2.
    """

    ambition_level: AmbitionLevel = Field(
        AmbitionLevel.CELSIUS_1_5,
        description="SBTi target ambition level",
    )
    pathway_type: PathwayType = Field(
        PathwayType.ACA,
        description="SBTi target-setting pathway (ACA, SDA, or FLAG)",
    )
    near_term_target_year: int = Field(
        DEFAULT_TARGET_YEAR_NEAR,
        ge=2025,
        le=2035,
        description="Near-term target year (5-10 years from submission)",
    )
    long_term_target_year: int = Field(
        DEFAULT_TARGET_YEAR_LONG,
        ge=2040,
        le=2055,
        description="Long-term / net-zero target year (no later than 2050 per SBTi)",
    )
    coverage_scope1_pct: float = Field(
        95.0,
        ge=0.0,
        le=100.0,
        description="Percentage of Scope 1 emissions covered by target",
    )
    coverage_scope2_pct: float = Field(
        95.0,
        ge=0.0,
        le=100.0,
        description="Percentage of Scope 2 emissions covered by target",
    )
    coverage_scope3_pct: float = Field(
        67.0,
        ge=0.0,
        le=100.0,
        description="Percentage of Scope 3 emissions covered by near-term target",
    )
    flag_pathway_enabled: bool = Field(
        False,
        description="Enable FLAG pathway for land-use related emissions",
    )
    sbti_submission_planned: bool = Field(
        True,
        description="Whether SBTi target submission is planned",
    )

    @model_validator(mode="after")
    def validate_target_years(self) -> "TargetConfig":
        """Ensure long-term target year is after near-term target year."""
        if self.long_term_target_year <= self.near_term_target_year:
            raise ValueError(
                f"long_term_target_year ({self.long_term_target_year}) must be "
                f"after near_term_target_year ({self.near_term_target_year})"
            )
        return self

    @model_validator(mode="after")
    def validate_sbti_coverage(self) -> "TargetConfig":
        """Warn if coverage thresholds are below SBTi minimums."""
        if self.sbti_submission_planned:
            if self.coverage_scope1_pct < 95.0:
                logger.warning(
                    "SBTi requires at least 95%% Scope 1 coverage for near-term targets. "
                    "Current: %.1f%%", self.coverage_scope1_pct
                )
            if self.coverage_scope2_pct < 95.0:
                logger.warning(
                    "SBTi requires at least 95%% Scope 2 coverage for near-term targets. "
                    "Current: %.1f%%", self.coverage_scope2_pct
                )
            if self.coverage_scope3_pct < 67.0:
                logger.warning(
                    "SBTi requires at least 67%% Scope 3 coverage for near-term targets. "
                    "Current: %.1f%%", self.coverage_scope3_pct
                )
        return self


class ReductionConfig(BaseModel):
    """Configuration for decarbonization pathway and MACC analysis.

    Defines constraints and parameters for reduction action evaluation
    and pathway modeling.
    """

    budget_constraint_eur: Optional[float] = Field(
        None,
        ge=0,
        description="Maximum investment budget (EUR) for reduction actions",
    )
    max_actions: int = Field(
        100,
        ge=1,
        le=500,
        description="Maximum number of reduction actions to evaluate",
    )
    planning_horizon_years: int = Field(
        10,
        ge=3,
        le=30,
        description="Planning horizon in years for pathway modeling",
    )
    include_renewable_procurement: bool = Field(
        True,
        description="Include renewable electricity procurement (PPAs, GOs) as a lever",
    )
    include_supplier_engagement: bool = Field(
        True,
        description="Include Scope 3 supplier engagement as a reduction lever",
    )
    include_fleet_electrification: bool = Field(
        True,
        description="Include fleet electrification as a reduction lever",
    )
    include_energy_efficiency: bool = Field(
        True,
        description="Include energy efficiency measures (LED, HVAC, insulation)",
    )
    include_fuel_switching: bool = Field(
        True,
        description="Include fuel switching scenarios (gas-to-electric, etc.)",
    )
    include_process_optimization: bool = Field(
        False,
        description="Include industrial process optimization (manufacturing only)",
    )
    include_ccs: bool = Field(
        False,
        description="Include carbon capture and storage scenarios",
    )
    discount_rate_pct: float = Field(
        8.0,
        ge=0.0,
        le=30.0,
        description="Discount rate (%) for NPV calculations in MACC analysis",
    )
    carbon_price_eur_per_tco2e: float = Field(
        80.0,
        ge=0.0,
        description="Internal carbon price (EUR/tCO2e) for action valuation",
    )
    scenario_count: int = Field(
        3,
        ge=1,
        le=10,
        description="Number of decarbonization scenarios to model",
    )


class OffsetConfig(BaseModel):
    """Configuration for carbon credit and offset portfolio management.

    Manages residual emissions neutralization through high-quality
    carbon credits per VCMI and Oxford Principles.
    """

    strategy: OffsetStrategy = Field(
        OffsetStrategy.BOTH,
        description="Offset strategy: compensation (avoidance), neutralization (removal), or both",
    )
    quality_minimum_score: int = Field(
        60,
        ge=0,
        le=100,
        description="Minimum credit quality score (0-100) for portfolio inclusion",
    )
    max_nature_based_pct: float = Field(
        50.0,
        ge=0.0,
        le=100.0,
        description="Maximum percentage of nature-based credits in portfolio",
    )
    vcmi_target_claim: str = Field(
        "SILVER",
        description="Target VCMI claim tier: SILVER, GOLD, or PLATINUM",
    )
    max_offset_budget_eur: Optional[float] = Field(
        None,
        ge=0,
        description="Maximum annual budget (EUR) for carbon credit procurement",
    )
    preferred_credit_types: List[str] = Field(
        default_factory=lambda: [
            "verified_carbon_standard",
            "gold_standard",
            "american_carbon_registry",
        ],
        description="Preferred credit registries and standards",
    )
    shift_to_removals_by_year: int = Field(
        2040,
        ge=2025,
        le=2055,
        description="Target year for majority-removal portfolio per Oxford Principles",
    )
    track_co_benefits: bool = Field(
        True,
        description="Track SDG co-benefits of offset projects",
    )

    @field_validator("vcmi_target_claim")
    @classmethod
    def validate_vcmi_claim(cls, v: str) -> str:
        """Validate VCMI claim tier."""
        valid = {"SILVER", "GOLD", "PLATINUM"}
        if v.upper() not in valid:
            raise ValueError(
                f"Invalid VCMI claim tier: {v}. Must be one of: {sorted(valid)}"
            )
        return v.upper()


class ReportingConfig(BaseModel):
    """Configuration for multi-framework reporting.

    Defines output formats, framework mappings, and disclosure preferences
    for CDP, TCFD, SBTi, and ESRS E1 reporting.
    """

    formats: List[ReportFormat] = Field(
        default_factory=lambda: [ReportFormat.PDF, ReportFormat.HTML],
        description="Output formats for generated reports",
    )
    include_cdp_mapping: bool = Field(
        True,
        description="Map outputs to CDP Climate Change questionnaire",
    )
    include_tcfd_mapping: bool = Field(
        True,
        description="Map outputs to TCFD disclosure recommendations",
    )
    include_esrs_mapping: bool = Field(
        True,
        description="Map outputs to ESRS E1 Climate Change disclosures",
    )
    include_sbti_mapping: bool = Field(
        True,
        description="Map outputs to SBTi progress reporting template",
    )
    language: str = Field(
        "en",
        description="Primary report language (ISO 639-1)",
    )
    multi_language_support: bool = Field(
        False,
        description="Enable multi-language report generation",
    )
    supported_languages: List[str] = Field(
        default_factory=lambda: ["en"],
        description="Supported report languages",
    )
    review_workflow_enabled: bool = Field(
        True,
        description="Enable review and approval workflow for reports",
    )
    watermark_draft: bool = Field(
        True,
        description="Apply DRAFT watermark to unapproved documents",
    )


class PerformanceConfig(BaseModel):
    """Configuration for runtime performance tuning.

    Defines caching, concurrency, and timeout settings for the
    net-zero calculation pipeline.
    """

    cache_enabled: bool = Field(
        True,
        description="Enable Redis-based caching for emission factors and calculations",
    )
    cache_ttl_seconds: int = Field(
        3600,
        ge=60,
        le=86400,
        description="Cache time-to-live in seconds",
    )
    max_concurrent_calcs: int = Field(
        4,
        ge=1,
        le=32,
        description="Maximum concurrent calculation threads",
    )
    timeout_seconds: int = Field(
        300,
        ge=30,
        le=3600,
        description="Maximum timeout for a single engine calculation",
    )
    batch_size: int = Field(
        1000,
        ge=100,
        le=10000,
        description="Batch size for bulk data processing",
    )
    memory_limit_mb: int = Field(
        4096,
        ge=512,
        le=32768,
        description="Memory limit in MB for the calculation pipeline",
    )


class AuditTrailConfig(BaseModel):
    """Configuration for audit trail and provenance tracking."""

    enabled: bool = Field(
        True,
        description="Enable audit trail for all calculations",
    )
    sha256_provenance: bool = Field(
        True,
        description="Generate SHA-256 provenance hashes for all outputs",
    )
    calculation_logging: bool = Field(
        True,
        description="Log all intermediate calculation steps",
    )
    assumption_tracking: bool = Field(
        True,
        description="Track all assumptions used in calculations",
    )
    data_lineage_enabled: bool = Field(
        True,
        description="Track full data lineage from source to output",
    )
    retention_years: int = Field(
        7,
        ge=1,
        le=15,
        description="Audit trail retention period in years",
    )
    external_audit_export: bool = Field(
        True,
        description="Enable export format for external auditors and SBTi validators",
    )


class ScorecardConfig(BaseModel):
    """Configuration for net-zero maturity scorecard."""

    enabled: bool = Field(
        True,
        description="Enable net-zero maturity scorecard",
    )
    assessment_mode: MaturityAssessment = Field(
        MaturityAssessment.FULL,
        description="Assessment mode: FULL (10 dimensions), QUICK (5 dimensions), SKIP",
    )
    dimensions: List[str] = Field(
        default_factory=lambda: [
            "governance",
            "strategy",
            "baseline_completeness",
            "target_ambition",
            "reduction_progress",
            "scope3_coverage",
            "supplier_engagement",
            "offset_quality",
            "disclosure_readiness",
            "transition_plan_credibility",
        ],
        description="Scorecard dimensions for maturity assessment",
    )
    benchmark_enabled: bool = Field(
        True,
        description="Enable sector peer benchmarking",
    )
    peer_group: str = Field(
        "SECTOR",
        description="Peer group for benchmarking: SECTOR, GEOGRAPHY, REVENUE_BAND, SBTI_STATUS",
    )
    kpi_set: List[str] = Field(
        default_factory=lambda: [
            "absolute_emissions_trajectory",
            "emission_intensity_revenue",
            "emission_intensity_employee",
            "scope3_coverage_pct",
            "yoy_reduction_rate",
            "cagr_reduction_rate",
            "renewable_electricity_pct",
            "supplier_engagement_rate",
            "sbti_target_status",
            "cdp_score",
            "offset_portfolio_quality",
        ],
        description="KPI set for scorecard and benchmarking",
    )
    target_years: List[int] = Field(
        default_factory=lambda: [2025, 2030, 2035, 2040, 2050],
        description="Milestone years for trajectory tracking",
    )


# =============================================================================
# Main Configuration Models
# =============================================================================


class NetZeroStarterConfig(BaseModel):
    """Main configuration for PACK-021 Net Zero Starter Pack.

    This is the root configuration model that contains all sub-configurations
    for net-zero planning. The organization.sector field drives pathway
    selection and Scope 3 prioritization.
    """

    # Temporal settings
    reporting_year: int = Field(
        2025,
        ge=2020,
        le=2035,
        description="Current reporting year for GHG inventory",
    )
    base_year: int = Field(
        DEFAULT_BASE_YEAR,
        ge=2015,
        le=2030,
        description="Base year for emissions baseline and target tracking",
    )
    pack_version: str = Field(
        "1.0.0",
        description="Pack configuration version",
    )

    # Sub-configurations
    organization: OrganizationConfig = Field(
        default_factory=OrganizationConfig,
        description="Organization profile configuration",
    )
    boundary: BoundaryConfig = Field(
        default_factory=BoundaryConfig,
        description="GHG Protocol boundary configuration",
    )
    scope: ScopeConfig = Field(
        default_factory=ScopeConfig,
        description="Scope 1/2/3 emissions configuration",
    )
    target: TargetConfig = Field(
        default_factory=TargetConfig,
        description="Science-based target setting configuration",
    )
    reduction: ReductionConfig = Field(
        default_factory=ReductionConfig,
        description="Decarbonization pathway and MACC configuration",
    )
    offset: OffsetConfig = Field(
        default_factory=OffsetConfig,
        description="Carbon credit and offset portfolio configuration",
    )
    reporting: ReportingConfig = Field(
        default_factory=ReportingConfig,
        description="Multi-framework reporting configuration",
    )
    performance: PerformanceConfig = Field(
        default_factory=PerformanceConfig,
        description="Runtime performance tuning configuration",
    )
    audit_trail: AuditTrailConfig = Field(
        default_factory=AuditTrailConfig,
        description="Audit trail and provenance configuration",
    )
    scorecard: ScorecardConfig = Field(
        default_factory=ScorecardConfig,
        description="Net-zero maturity scorecard configuration",
    )

    @model_validator(mode="after")
    def validate_base_year_before_reporting(self) -> "NetZeroStarterConfig":
        """Ensure base year is not after reporting year."""
        if self.base_year > self.reporting_year:
            raise ValueError(
                f"base_year ({self.base_year}) must not be after "
                f"reporting_year ({self.reporting_year})"
            )
        return self

    @model_validator(mode="after")
    def validate_energy_sector_requires_sda(self) -> "NetZeroStarterConfig":
        """Warn if energy sector is not using SDA pathway."""
        energy_sectors = {
            OrganizationSector.ENERGY,
            OrganizationSector.TRANSPORT,
        }
        if self.organization.sector in energy_sectors:
            if self.target.pathway_type != PathwayType.SDA:
                logger.warning(
                    "SDA pathway is recommended for %s sector. "
                    "Current pathway: %s",
                    self.organization.sector.value,
                    self.target.pathway_type.value,
                )
        return self

    @model_validator(mode="after")
    def validate_agriculture_requires_flag(self) -> "NetZeroStarterConfig":
        """Warn if agriculture sector does not enable FLAG pathway."""
        if self.organization.sector == OrganizationSector.AGRICULTURE:
            if not self.target.flag_pathway_enabled:
                logger.warning(
                    "FLAG pathway is recommended for agriculture sector. "
                    "Enabling flag_pathway_enabled."
                )
                object.__setattr__(self.target, "flag_pathway_enabled", True)
        return self

    @model_validator(mode="after")
    def validate_scope3_requires_categories(self) -> "NetZeroStarterConfig":
        """Warn if Scope 3 is enabled but no categories are selected."""
        if self.scope.include_scope3 and not self.scope.scope3_categories:
            logger.warning(
                "Scope 3 is enabled but no categories are selected. "
                "At least categories 1, 4, and 5 are recommended."
            )
        return self

    def get_enabled_engines(self) -> List[str]:
        """Return list of engine names that should be enabled based on config.

        Returns:
            List of engine identifier strings.
        """
        engines = [
            "baseline_inventory",
            "target_setting",
            "gap_analysis",
            "net_zero_scorecard",
            "sector_benchmark",
        ]

        # Always add pathway and MACC if reduction config has actions
        if self.reduction.max_actions > 0:
            engines.append("decarbonization_pathway")
            engines.append("macc_analysis")

        # Add offset engine if strategy is configured
        if self.offset.strategy != OffsetStrategy.COMPENSATION_ONLY:
            engines.append("offset_portfolio")
        elif self.offset.max_offset_budget_eur and self.offset.max_offset_budget_eur > 0:
            engines.append("offset_portfolio")

        return sorted(set(engines))


# =============================================================================
# Pack Configuration Wrapper
# =============================================================================


class PackConfig(BaseModel):
    """Top-level pack configuration wrapper.

    Handles preset loading, environment variable overrides, and
    configuration merging.

    Example:
        >>> config = PackConfig.from_preset("manufacturing")
        >>> print(config.pack.organization.sector)
        OrganizationSector.MANUFACTURING
        >>> config = PackConfig.from_preset("sme_general", overrides={"reporting_year": 2026})
    """

    pack: NetZeroStarterConfig = Field(
        default_factory=NetZeroStarterConfig,
        description="Main Net Zero Starter configuration",
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
        "PACK-021-net-zero-starter",
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
            preset_name: Name of the preset (manufacturing, services, retail,
                energy, technology, sme_general).
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
        env_overrides = _get_env_overrides("NET_ZERO_STARTER_")
        if env_overrides:
            preset_data = _merge_config(preset_data, env_overrides)

        # Apply explicit overrides
        if overrides:
            preset_data = _merge_config(preset_data, overrides)

        pack_config = NetZeroStarterConfig(**preset_data)
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

        pack_config = NetZeroStarterConfig(**config_data)
        return cls(pack=pack_config)

    def get_config_hash(self) -> str:
        """Generate SHA-256 hash of the current configuration for provenance.

        Returns:
            Hex-encoded SHA-256 hash string.
        """
        config_json = self.model_dump_json(indent=None)
        return hashlib.sha256(config_json.encode("utf-8")).hexdigest()

    def validate_config(self) -> List[str]:
        """Cross-field validation returning warnings.

        Returns:
            List of warning messages (empty if fully valid).
        """
        return validate_config(self.pack)


# =============================================================================
# Utility Functions
# =============================================================================


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
        NET_ZERO_STARTER_REPORTING_YEAR=2026
        NET_ZERO_STARTER_SCOPE__INCLUDE_SCOPE3=true
        NET_ZERO_STARTER_TARGET__AMBITION_LEVEL=CELSIUS_1_5

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


def validate_config(config: NetZeroStarterConfig) -> List[str]:
    """Validate a net-zero configuration and return any warnings.

    Performs cross-field validation beyond what Pydantic validators cover.
    Returns advisory warnings, not hard errors.

    Args:
        config: NetZeroStarterConfig instance to validate.

    Returns:
        List of warning messages (empty if fully valid).
    """
    warnings: List[str] = []

    # Check organization name is set
    if not config.organization.name:
        warnings.append(
            "Organization name is empty. Set organization.name for meaningful reports."
        )

    # Check base year is reasonable
    if config.base_year > config.reporting_year:
        warnings.append(
            f"Base year ({config.base_year}) is after reporting year "
            f"({config.reporting_year}). Base year should precede reporting year."
        )

    # Check Scope 3 coverage
    if config.scope.include_scope3 and len(config.scope.scope3_categories) < 3:
        warnings.append(
            "Fewer than 3 Scope 3 categories selected. SBTi requires covering "
            "at least 67% of Scope 3 emissions. Consider adding more categories."
        )

    # Check SDA pathway for eligible sectors
    sda_sectors = {
        OrganizationSector.ENERGY,
        OrganizationSector.TRANSPORT,
        OrganizationSector.MANUFACTURING,
    }
    if config.organization.sector in sda_sectors:
        if config.target.pathway_type == PathwayType.ACA:
            warnings.append(
                f"ACA pathway selected for {config.organization.sector.value} sector. "
                f"SDA pathway may be more appropriate. Check SBTi sector guidance."
            )

    # Check FLAG for agriculture
    if config.organization.sector == OrganizationSector.AGRICULTURE:
        if not config.target.flag_pathway_enabled:
            warnings.append(
                "FLAG pathway is strongly recommended for agriculture sector. "
                "Enable target.flag_pathway_enabled for SBTi compliance."
            )

    # Check near-term target year is within SBTi window
    if config.target.near_term_target_year > 2035:
        warnings.append(
            "SBTi requires near-term targets within 5-10 years of submission. "
            f"Near-term year {config.target.near_term_target_year} may be too far."
        )

    # Check long-term target year
    if config.target.long_term_target_year > 2050:
        warnings.append(
            "SBTi Net-Zero Standard requires net-zero by no later than 2050. "
            f"Long-term year {config.target.long_term_target_year} exceeds this."
        )

    # Check offset quality
    if config.offset.quality_minimum_score < 40:
        warnings.append(
            "Offset quality minimum score is very low. VCMI and Oxford Principles "
            "require high-quality credits. Consider raising quality_minimum_score."
        )

    # Check reporting frameworks
    if not any([
        config.reporting.include_cdp_mapping,
        config.reporting.include_tcfd_mapping,
        config.reporting.include_esrs_mapping,
        config.reporting.include_sbti_mapping,
    ]):
        warnings.append(
            "No reporting framework mappings enabled. Enable at least one "
            "framework (CDP, TCFD, ESRS, SBTi) for disclosure readiness."
        )

    return warnings


def get_sector_info(sector: Union[str, OrganizationSector]) -> Dict[str, Any]:
    """Get detailed information about an organization sector.

    Args:
        sector: Sector enum or string value.

    Returns:
        Dictionary with name, recommended pathway, typical scope split,
        and key decarbonization levers.
    """
    key = sector.value if isinstance(sector, OrganizationSector) else sector
    return SECTOR_INFO.get(key, {
        "name": key,
        "recommended_pathway": "ACA",
        "typical_scope_split": "Varies",
        "key_levers": ["energy efficiency", "renewable procurement"],
    })


def get_sbti_reduction_rate(ambition: Union[str, AmbitionLevel]) -> Dict[str, float]:
    """Get SBTi minimum annual reduction rates for an ambition level.

    Args:
        ambition: Ambition level enum or string value.

    Returns:
        Dictionary with scope_1_2_linear_annual, scope_3_linear_annual,
        and long_term_reduction_pct.
    """
    key = ambition.value if isinstance(ambition, AmbitionLevel) else ambition
    return SBTI_REDUCTION_RATES.get(key, SBTI_REDUCTION_RATES["CELSIUS_1_5"])


def get_gwp100(gas: str) -> int:
    """Get IPCC AR6 GWP100 value for a greenhouse gas.

    Args:
        gas: Greenhouse gas identifier (CO2, CH4, N2O, etc.).

    Returns:
        GWP100 value (dimensionless, relative to CO2).
    """
    return IPCC_AR6_GWP100.get(gas.upper(), 0)


def get_default_config(
    sector: OrganizationSector = OrganizationSector.MANUFACTURING,
) -> NetZeroStarterConfig:
    """Get default configuration for a given organization sector.

    Args:
        sector: Organization sector to configure for.

    Returns:
        NetZeroStarterConfig instance with sector-appropriate defaults.
    """
    sector_info = SECTOR_INFO.get(sector.value, {})
    pathway_str = sector_info.get("recommended_pathway", "ACA")
    pathway = PathwayType(pathway_str) if pathway_str in PathwayType.__members__ else PathwayType.ACA

    return NetZeroStarterConfig(
        organization=OrganizationConfig(sector=sector),
        target=TargetConfig(pathway_type=pathway),
        scope=ScopeConfig(
            scope3_categories=PRIORITY_SCOPE3_BY_SECTOR.get(
                sector.value, DEFAULT_SCOPE3_CATEGORIES
            ),
        ),
    )


def list_available_presets() -> Dict[str, str]:
    """List all available configuration presets.

    Returns:
        Dictionary mapping preset names to descriptions.
    """
    return SUPPORTED_PRESETS.copy()
