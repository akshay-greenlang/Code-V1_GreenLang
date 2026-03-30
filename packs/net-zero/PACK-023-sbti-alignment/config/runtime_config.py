"""
PACK-023 SBTi Alignment Pack - Runtime Configuration Manager

This module implements comprehensive Pydantic v2 configuration management for
the SBTi Alignment Pack, covering the complete lifecycle from near-term/long-term/
net-zero target setting through validated submission and ongoing progress tracking.

Configuration Hierarchy:
    1. Base pack.yaml manifest
    2. Sector-specific presets (power, heavy_industry, manufacturing, transport,
       financial_services, food_agriculture, real_estate, technology)
    3. Environment variable overrides (SBTI_ALIGNMENT_* prefix)
    4. Runtime explicit overrides

Key Subsystems:
    - Organization Profile (sector, size, boundary method)
    - Baseline Inventory (Scope 1/2/3 emissions data sources)
    - Target Setting (near-term, long-term, net-zero with ACA/SDA/FLAG)
    - Scope 3 Screening (15 categories with 40% trigger + 67%/90% coverage)
    - SDA Pathway (12 sectors with IEA NZE benchmarks)
    - FLAG Assessment (11 commodities with deforestation commitment)
    - Temperature Rating (v2.0 with 6 aggregation methods: WATS/TETS/MOTS/EOTS/ECOTS/AOTS)
    - Progress Tracking (annual on-track/off-track assessment)
    - Base Year Recalculation (M&A, divestitures, structural changes)
    - FINZ Portfolio (8 asset classes, PCAF data quality scoring)
    - Submission Readiness (42-criterion automated validation)
    - Multi-Framework Reporting (CDP C4, TCFD, CSRD ESRS E1, GHG Protocol)
    - Performance & Caching
    - Audit Trail & Provenance

Regulatory Framework:
    - SBTi Corporate Manual V5.3 (2024)
    - SBTi Corporate Net-Zero Standard V1.3 (2024)
    - SBTi SDA Tool V3.0 (2024) with 12 sectors
    - SBTi FLAG Guidance V1.1 (2022) with 11 commodities
    - SBTi FINZ V1.0 (2024) with 8 asset classes
    - SBTi Temperature Rating V2.0 (2024) with 6 methods
    - GHG Protocol Corporate Standard (revised)
    - GHG Protocol Scope 3 Standard
    - IPCC AR6 WG1/WG3 (2021/2022)
    - Paris Agreement (UNFCCC, 2015)
    - ISO 14064-1:2018
    - CDP Climate Change (2024)
    - TCFD Recommendations (2017)
    - CSRD ESRS E1 (EU, 2023)
    - IEA Net Zero Roadmap (2023)
    - PCAF Global Standard (2022)

Example:
    >>> config = SBTiAlignmentConfig.from_preset("manufacturing")
    >>> print(config.pack.organization.sector)
    OrganizationSector.MANUFACTURING
    >>> config = SBTiAlignmentConfig.from_yaml("path/to/config.yaml")
    >>> hash_val = config.get_config_hash()
"""

import hashlib
import logging
import os
from datetime import date, datetime
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
DEFAULT_REPORTING_YEAR: int = 2025
DEFAULT_NEAR_TERM_YEAR: int = 2030
DEFAULT_LONG_TERM_YEAR: int = 2050
DEFAULT_SCOPE3_CATEGORIES: List[int] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12]

SUPPORTED_PRESETS: Dict[str, str] = {
    "power": "Power generation with Scope 1 dominant, SDA pathway recommended",
    "heavy_industry": "Heavy industry (cement, steel, chemicals) with high process emissions, SDA pathway",
    "manufacturing": "General manufacturing with mixed Scope 1/2/3, ACA/SDA selection",
    "transport": "Transport & logistics with fleet-dominant emissions, SDA pathway",
    "financial_services": "Financial institutions with portfolio-level targets, FINZ framework",
    "food_agriculture": "Food production and agriculture with FLAG pathway recommended",
    "real_estate": "Real estate and construction with energy-focused emissions",
    "technology": "Technology and software with low Scope 1, high Scope 3 (cloud, hardware)",
}

# SBTi 42-criterion validation (C1-C28 near-term, NZ-C1 to NZ-C14 net-zero)
SBTI_NEAR_TERM_CRITERIA: Dict[str, str] = {
    "C1": "GHG Protocol Scope 1 emissions covered",
    "C2": "GHG Protocol Scope 2 emissions covered",
    "C3": "GHG Protocol Scope 3 emissions covered",
    "C4": "Scope 3 categories materiality assessment (40% trigger)",
    "C5": "Scope 3 coverage 67% near-term threshold",
    "C6": "Base year selection and documentation",
    "C7": "Near-term target year within 5-10 years",
    "C8": "Scope 1+2 reduction ambition (linear pathway)",
    "C9": "Scope 3 reduction ambition (linear pathway)",
    "C10": "Absolute or intensity-based reduction",
    "C11": "Well-to-wheel methodology (transport)",
    "C12": "Boundary method documented (operational/financial control)",
    "C13": "Recalculation policy (structural changes >= 5%)",
    "C14": "Conversion factors and emission factors documented",
    "C15": "SDA pathway validation (12 sectors, IEA NZE benchmarks)",
    "C16": "FLAG pathway validation (11 commodities, no-deforestation)",
    "C17": "Climate science alignment (1.5C, 2C pathways)",
    "C18": "SBTi temperature rating calculation",
    "C19": "Decarbonization actions identified (energy efficiency, renewable, etc.)",
    "C20": "Supplier engagement strategy for Scope 3 reduction",
    "C21": "Near-term submission checklist completeness (90%+)",
    "C22": "Data quality and uncertainty documentation",
    "C23": "Interim targets for tracking progress",
    "C24": "Regular progress reporting (annual or more frequent)",
    "C25": "Science-based target governance and accountability",
    "C26": "Alignment with Paris Agreement (well-below 2C, 1.5C effort)",
    "C27": "Benchmark comparison against peer group (sector, geography, size)",
    "C28": "Executive sign-off and board approval",
}

SBTI_NET_ZERO_CRITERIA: Dict[str, str] = {
    "NZ_C1": "Net-zero target year no later than 2050",
    "NZ_C2": "Scope 1+2 reduction >= 90% by 2050",
    "NZ_C3": "Scope 3 reduction >= 90% by 2050 (if material)",
    "NZ_C4": "Residual emissions quantified and documented",
    "NZ_C5": "Carbon removal strategies for residual emissions",
    "NZ_C6": "High-quality carbon credit portfolio (VCMI Silver+)",
    "NZ_C7": "Offset credit longevity >= 100 years post-certification",
    "NZ_C8": "No more than 10% of 2050 emissions offset",
    "NZ_C9": "Interim milestones (e.g., 2030, 2040) documented",
    "NZ_C10": "Technology transition roadmap (clean energy, efficiency gains)",
    "NZ_C11": "Just transition considerations (worker displacement, regional impacts)",
    "NZ_C12": "Net-zero governance and accountability structure",
    "NZ_C13": "Long-term target submission and SBTi validation",
    "NZ_C14": "Annual progress reporting with trajectory tracking",
}

# SBTi Temperature Rating V2.0 - 6 aggregation methods
TEMPERATURE_RATING_METHODS: Dict[str, str] = {
    "WATS": "Weighted Average Temperature Score (portfolio weighted by emissions)",
    "TETS": "Total Emissions Weighted Temperature Score (all emissions weighted)",
    "MOTS": "Market Owners Trajectory Score (carbon intensity trajectory)",
    "EOTS": "Emissions Owned Trajectory Score (absolute emissions trajectory)",
    "ECOTS": "Emissions Coverage Trajectory Score (coverage-adjusted trajectory)",
    "AOTS": "Absolute Owned Trajectory Score (pure absolute reduction rate)",
}

# SDA Pathway - 12 Sectors with IEA NZE Benchmarks
SDA_SECTORS: Dict[str, Dict[str, Any]] = {
    "power": {"name": "Power Generation", "metric": "gCO2/kWh", "iea_2030": 50, "iea_2050": 0},
    "cement": {"name": "Cement Production", "metric": "kgCO2/t", "iea_2030": 0.45, "iea_2050": 0.1},
    "steel": {"name": "Steel Production", "metric": "tCO2/t", "iea_2030": 1.3, "iea_2050": 0.2},
    "chemicals": {"name": "Chemicals", "metric": "tCO2/t output", "iea_2030": 0.08, "iea_2050": 0.02},
    "aluminum": {"name": "Aluminum Production", "metric": "tCO2/t", "iea_2030": 7.5, "iea_2050": 1.5},
    "aviation": {"name": "Aviation", "metric": "gCO2/passenger-km", "iea_2030": 80, "iea_2050": 20},
    "shipping": {"name": "Shipping", "metric": "gCO2/tonne-km", "iea_2030": 50, "iea_2050": 15},
    "road_freight": {"name": "Road Freight", "metric": "gCO2/tonne-km", "iea_2030": 65, "iea_2050": 20},
    "passenger_cars": {"name": "Passenger Cars", "metric": "gCO2/km", "iea_2030": 100, "iea_2050": 0},
    "rail": {"name": "Rail Transport", "metric": "gCO2/passenger-km", "iea_2030": 30, "iea_2050": 10},
    "oil_refining": {"name": "Oil Refining", "metric": "tCO2/barrel", "iea_2030": 0.06, "iea_2050": 0.01},
    "natural_gas": {"name": "Natural Gas Production", "metric": "kgCO2/kWh", "iea_2030": 0.35, "iea_2050": 0.15},
}

# FLAG Pathway - 11 Commodities with Deforestation Commitments
FLAG_COMMODITIES: Dict[str, Dict[str, Any]] = {
    "beef": {"name": "Beef Production", "trigger_pct": 20, "deforestation_commitment": True},
    "palm": {"name": "Palm Oil", "trigger_pct": 20, "deforestation_commitment": True},
    "soy": {"name": "Soy Production", "trigger_pct": 20, "deforestation_commitment": True},
    "cocoa": {"name": "Cocoa", "trigger_pct": 20, "deforestation_commitment": True},
    "coffee": {"name": "Coffee", "trigger_pct": 20, "deforestation_commitment": True},
    "timber": {"name": "Timber & Paper", "trigger_pct": 20, "deforestation_commitment": True},
    "rubber": {"name": "Rubber", "trigger_pct": 20, "deforestation_commitment": True},
    "wool": {"name": "Wool", "trigger_pct": 20, "deforestation_commitment": True},
    "dairy": {"name": "Dairy", "trigger_pct": 20, "deforestation_commitment": True},
    "maize": {"name": "Maize/Corn", "trigger_pct": 20, "deforestation_commitment": True},
    "sugar": {"name": "Sugar", "trigger_pct": 20, "deforestation_commitment": True},
}

# FINZ V1.0 - 8 Asset Classes with PCAF Data Quality Scoring
FINZ_ASSET_CLASSES: Dict[str, Dict[str, Any]] = {
    "listed_equity": {"name": "Listed Equity", "pcaf_quality_default": 3, "coverage_critical": True},
    "corporate_bonds": {"name": "Corporate Bonds", "pcaf_quality_default": 3, "coverage_critical": True},
    "mortgages": {"name": "Mortgages", "pcaf_quality_default": 4, "coverage_critical": False},
    "commercial_real_estate": {"name": "Commercial Real Estate", "pcaf_quality_default": 4, "coverage_critical": False},
    "project_finance": {"name": "Project Finance", "pcaf_quality_default": 3, "coverage_critical": True},
    "infrastructure": {"name": "Infrastructure Finance", "pcaf_quality_default": 3, "coverage_critical": False},
    "private_equity": {"name": "Private Equity", "pcaf_quality_default": 4, "coverage_critical": False},
    "investment_funds": {"name": "Investment Funds/Funds of Funds", "pcaf_quality_default": 4, "coverage_critical": False},
}

# Scope 3 Materiality Screening - 15 Categories with Priority
SCOPE3_CATEGORIES_INFO: Dict[int, Dict[str, Any]] = {
    1: {"name": "Purchased goods & services", "typical_importance": "HIGH"},
    2: {"name": "Capital goods", "typical_importance": "MEDIUM"},
    3: {"name": "Fuel & energy activities", "typical_importance": "HIGH"},
    4: {"name": "Upstream transportation", "typical_importance": "HIGH"},
    5: {"name": "Waste generated", "typical_importance": "MEDIUM"},
    6: {"name": "Business travel", "typical_importance": "MEDIUM"},
    7: {"name": "Employee commuting", "typical_importance": "MEDIUM"},
    8: {"name": "Upstream leased assets", "typical_importance": "LOW"},
    9: {"name": "Downstream transportation", "typical_importance": "MEDIUM"},
    10: {"name": "Processing of sold products", "typical_importance": "MEDIUM"},
    11: {"name": "Use of sold products", "typical_importance": "CRITICAL"},
    12: {"name": "End-of-life treatment", "typical_importance": "MEDIUM"},
    13: {"name": "Downstream leased assets", "typical_importance": "LOW"},
    14: {"name": "Franchises", "typical_importance": "LOW"},
    15: {"name": "Investments", "typical_importance": "MEDIUM"},
}


# =============================================================================
# Enums - SBTi-specific enumeration types (18 enums)
# =============================================================================


class OrganizationSector(str, Enum):
    """Organization sector classification."""
    POWER = "POWER"
    HEAVY_INDUSTRY = "HEAVY_INDUSTRY"
    MANUFACTURING = "MANUFACTURING"
    TRANSPORT = "TRANSPORT"
    FINANCIAL_SERVICES = "FINANCIAL_SERVICES"
    FOOD_AGRICULTURE = "FOOD_AGRICULTURE"
    REAL_ESTATE = "REAL_ESTATE"
    TECHNOLOGY = "TECHNOLOGY"
    CONSTRUCTION = "CONSTRUCTION"
    HEALTHCARE = "HEALTHCARE"
    RETAIL = "RETAIL"
    UTILITIES = "UTILITIES"
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
    ACA = "ACA"  # Absolute Contraction Approach
    SDA = "SDA"  # Sectoral Decarbonization Approach
    FLAG = "FLAG"  # Forest, Land and Agriculture


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
    DATABASE = "DATABASE"


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
    """SBTi maturity assessment mode."""
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


class ProgressStatus(str, Enum):
    """Annual progress tracking status."""
    ON_TRACK = "ON_TRACK"
    OFF_TRACK = "OFF_TRACK"
    ACCELERATING = "ACCELERATING"
    LAGGING = "LAGGING"
    NO_DATA = "NO_DATA"


class RecalculationType(str, Enum):
    """Base year recalculation type."""
    STRUCTURAL_CHANGE = "STRUCTURAL_CHANGE"
    M_AND_A = "M_AND_A"
    DIVESTITURE = "DIVESTITURE"
    ACCOUNTING_CHANGE = "ACCOUNTING_CHANGE"
    METHODOLOGY_IMPROVEMENT = "METHODOLOGY_IMPROVEMENT"


class VCMIClaimTier(str, Enum):
    """VCMI offset claim tier."""
    SILVER = "SILVER"
    GOLD = "GOLD"
    PLATINUM = "PLATINUM"


class TemperatureRatingMethod(str, Enum):
    """SBTi Temperature Rating v2.0 aggregation method."""
    WATS = "WATS"  # Weighted Average Temperature Score
    TETS = "TETS"  # Total Emissions Weighted Temperature Score
    MOTS = "MOTS"  # Market Owners Trajectory Score
    EOTS = "EOTS"  # Emissions Owned Trajectory Score
    ECOTS = "ECOTS"  # Emissions Coverage Trajectory Score
    AOTS = "AOTS"  # Absolute Owned Trajectory Score


class PCOFDataQuality(str, Enum):
    """PCAF data quality score (1-5, lower is better)."""
    COMPANY_LEVEL = "1"  # Company-specific data
    ACTIVITY_LEVEL = "2"  # Activity/product-level data
    ESTIMATE_PRODUCT_LEVEL = "3"  # Estimated product-level data
    ESTIMATED_SECTOR_AVERAGE = "4"  # Sector average estimate
    SCREENING_DATA_FALLBACK = "5"  # Screening data fallback


class SubmissionReadinessLevel(str, Enum):
    """Submission readiness assessment level."""
    READY = "READY"  # >= 90% criteria met, submission recommended
    ALMOST_READY = "ALMOST_READY"  # 70-89% criteria met, minor gaps
    NEEDS_WORK = "NEEDS_WORK"  # 50-69% criteria met, significant gaps
    NOT_READY = "NOT_READY"  # < 50% criteria met, major work required


# =============================================================================
# Pydantic Sub-Config Models
# =============================================================================


class OrganizationConfig(BaseModel):
    """Configuration for the organization profile."""
    name: str = Field("", description="Legal entity name")
    sector: OrganizationSector = Field(
        OrganizationSector.MANUFACTURING,
        description="Primary business sector"
    )
    size: OrganizationSize = Field(
        OrganizationSize.LARGE,
        description="Organization size classification"
    )
    region: str = Field("EU", description="Primary operating region")
    country: str = Field("DE", description="Headquarters country (ISO 3166-1 alpha-2)")
    revenue_eur: Optional[float] = Field(None, ge=0, description="Annual revenue in EUR")
    employee_count: Optional[int] = Field(None, ge=0, description="Total employees")
    fiscal_year_end: str = Field("12-31", description="Fiscal year end (MM-DD format)")
    nace_code: Optional[str] = Field(None, description="NACE industry classification")
    gics_code: Optional[str] = Field(None, description="GICS sector code")
    erp_system: ERPType = Field(ERPType.NONE, description="ERP system in use")
    public_company: bool = Field(False, description="Is publicly traded company")
    listed_exchange: Optional[str] = Field(None, description="Exchange listing (if public)")

    @field_validator("fiscal_year_end")
    @classmethod
    def validate_fiscal_year_end(cls, v: str) -> str:
        """Validate fiscal year end is in MM-DD format."""
        parts = v.split("-")
        if len(parts) != 2:
            raise ValueError(f"fiscal_year_end must be in MM-DD format, got: {v}")
        month, day = int(parts[0]), int(parts[1])
        if month < 1 or month > 12 or day < 1 or day > 31:
            raise ValueError(f"Invalid date in fiscal_year_end: {v}")
        return v


class BoundaryConfig(BaseModel):
    """Configuration for GHG Protocol organizational boundary."""
    method: BoundaryMethod = Field(
        BoundaryMethod.OPERATIONAL_CONTROL,
        description="GHG Protocol boundary approach"
    )
    entities_included: List[str] = Field(
        default_factory=list,
        description="Legal entities included in boundary"
    )
    reporting_currency: str = Field("EUR", description="Reporting currency (ISO 4217)")
    include_joint_ventures: bool = Field(False, description="Include JVs in boundary")
    base_year_recalculation_threshold_pct: float = Field(
        5.0,
        ge=0.0,
        le=100.0,
        description="Structural change threshold (%) for base year recalculation"
    )

    @field_validator("reporting_currency")
    @classmethod
    def validate_currency(cls, v: str) -> str:
        """Validate currency is 3-letter ISO code."""
        if len(v) != 3 or not v.isalpha():
            raise ValueError(f"reporting_currency must be 3-letter ISO code, got: {v}")
        return v.upper()


class BaselineInventoryConfig(BaseModel):
    """Configuration for baseline inventory data sources and calculation."""
    scope1_data_source: DataSourceType = Field(
        DataSourceType.MANUAL,
        description="Data source for Scope 1 emissions"
    )
    scope2_data_source: DataSourceType = Field(
        DataSourceType.MANUAL,
        description="Data source for Scope 2 emissions"
    )
    scope3_data_source: DataSourceType = Field(
        DataSourceType.MANUAL,
        description="Data source for Scope 3 emissions"
    )
    baseline_year: int = Field(
        DEFAULT_BASE_YEAR,
        ge=2015,
        le=2030,
        description="Baseline inventory year"
    )
    scope2_methods: List[str] = Field(
        default_factory=lambda: ["location_based", "market_based"],
        description="Scope 2 calculation methods"
    )
    emission_factors_source: str = Field(
        "IPCC_AR6_2022",
        description="Emission factors dataset (IPCC AR6, EPA, IVL, etc.)"
    )
    include_biogenic_co2: bool = Field(
        False,
        description="Include biogenic CO2 in inventory"
    )


class Scope3ScreeningConfig(BaseModel):
    """Configuration for Scope 3 materiality screening (15 categories, 40% trigger)."""
    include_scope3: bool = Field(True, description="Include Scope 3 in inventory")
    scope3_categories: List[int] = Field(
        default_factory=lambda: DEFAULT_SCOPE3_CATEGORIES.copy(),
        description="Scope 3 categories (1-15) to include"
    )
    materiality_trigger_pct: float = Field(
        40.0,
        ge=0.0,
        le=100.0,
        description="Materiality threshold: if category <= X% of total, mark as immaterial"
    )
    scope3_method: Scope3Method = Field(
        Scope3Method.HYBRID,
        description="Default Scope 3 calculation methodology"
    )
    scope3_category_methods: Dict[int, str] = Field(
        default_factory=dict,
        description="Override methodology per Scope 3 category"
    )
    coverage_near_term_pct: float = Field(
        67.0,
        ge=0.0,
        le=100.0,
        description="Scope 3 coverage target for near-term (SBTi minimum: 67%)"
    )
    coverage_long_term_pct: float = Field(
        90.0,
        ge=0.0,
        le=100.0,
        description="Scope 3 coverage target for long-term (SBTi minimum: 90%)"
    )

    @field_validator("scope3_categories")
    @classmethod
    def validate_scope3_categories(cls, v: List[int]) -> List[int]:
        """Validate Scope 3 category numbers are 1-15."""
        invalid = [c for c in v if c < 1 or c > 15]
        if invalid:
            raise ValueError(f"Invalid Scope 3 categories: {invalid}. Must be 1-15.")
        return sorted(set(v))


class TargetSettingConfig(BaseModel):
    """Configuration for near-term, long-term, and net-zero target setting."""
    ambition_level: AmbitionLevel = Field(
        AmbitionLevel.CELSIUS_1_5,
        description="SBTi ambition level (1.5C, 2C, 2C+)"
    )
    pathway_type: PathwayType = Field(
        PathwayType.ACA,
        description="Target-setting pathway (ACA, SDA, FLAG)"
    )
    near_term_target_year: int = Field(
        DEFAULT_NEAR_TERM_YEAR,
        ge=2025,
        le=2035,
        description="Near-term target year (5-10 years from submission)"
    )
    long_term_target_year: int = Field(
        DEFAULT_LONG_TERM_YEAR,
        ge=2040,
        le=2055,
        description="Long-term target year (no later than 2050 per SBTi)"
    )
    net_zero_year: int = Field(
        DEFAULT_LONG_TERM_YEAR,
        ge=2040,
        le=2060,
        description="Net-zero achievement year"
    )
    coverage_scope1_pct: float = Field(
        95.0,
        ge=0.0,
        le=100.0,
        description="Scope 1 emissions coverage by target (%)"
    )
    coverage_scope2_pct: float = Field(
        95.0,
        ge=0.0,
        le=100.0,
        description="Scope 2 emissions coverage by target (%)"
    )
    coverage_scope3_pct: float = Field(
        67.0,
        ge=0.0,
        le=100.0,
        description="Scope 3 coverage for near-term (SBTi minimum: 67%)"
    )
    flag_pathway_enabled: bool = Field(
        False,
        description="Enable FLAG pathway for land-use emissions"
    )
    sbti_submission_planned: bool = Field(
        True,
        description="Whether SBTi target submission is planned"
    )
    absolute_vs_intensity: str = Field(
        "ABSOLUTE",
        description="Target metric (ABSOLUTE, INTENSITY, or HYBRID)"
    )

    @model_validator(mode="after")
    def validate_target_years(self) -> "TargetSettingConfig":
        """Ensure long-term target year is after near-term target year."""
        if self.long_term_target_year <= self.near_term_target_year:
            raise ValueError(
                f"long_term_target_year ({self.long_term_target_year}) must be "
                f"after near_term_target_year ({self.near_term_target_year})"
            )
        return self


class SDAPathwayConfig(BaseModel):
    """Configuration for Sectoral Decarbonization Approach (SDA) pathway."""
    enabled: bool = Field(True, description="Enable SDA pathway calculations")
    methodology: str = Field("SBTI_SDA_V3", description="SDA methodology version")
    benchmark_year: int = Field(2030, description="SDA benchmark target year")
    sector: Optional[str] = Field(None, description="SDA sector key (power, cement, steel, etc.)")
    intensity_metric: Optional[str] = Field(None, description="Intensity metric for sector")
    initial_intensity: Optional[float] = Field(None, description="Baseline intensity value")
    target_intensity: Optional[float] = Field(None, description="Target intensity value")
    use_iea_nze_benchmark: bool = Field(
        True,
        description="Use IEA Net Zero Emissions benchmark for SDA"
    )


class FLAGPathwayConfig(BaseModel):
    """Configuration for Forest, Land and Agriculture (FLAG) pathway."""
    enabled: bool = Field(False, description="Enable FLAG pathway")
    methodology: str = Field("SBTI_FLAG_V1.1", description="FLAG methodology version")
    commodities: List[str] = Field(
        default_factory=list,
        description="Commodities subject to FLAG (beef, palm, soy, etc.)"
    )
    no_deforestation_commitment: bool = Field(
        False,
        description="Company has no-deforestation commitment in scope"
    )
    deforestation_trigger_pct: float = Field(
        20.0,
        ge=0.0,
        le=100.0,
        description="Trigger threshold for FLAG pathway inclusion"
    )


class TemperatureRatingConfig(BaseModel):
    """Configuration for SBTi Temperature Rating v2.0 calculations."""
    enabled: bool = Field(True, description="Enable temperature rating calculations")
    version: str = Field("2.0", description="Temperature rating methodology version")
    aggregation_method: TemperatureRatingMethod = Field(
        TemperatureRatingMethod.WATS,
        description="Portfolio aggregation method"
    )
    scenario_year: int = Field(2050, description="Scenario/projection year for temperature calc")
    include_financed_emissions: bool = Field(
        False,
        description="Include financed emissions (for financial institutions)"
    )
    warming_threshold_celsius: float = Field(
        1.5,
        ge=0.5,
        le=3.0,
        description="Climate scenario threshold (1.5C, 2.0C, etc.)"
    )


class ProgressTrackingConfig(BaseModel):
    """Configuration for annual progress tracking and on-track assessment."""
    enabled: bool = Field(True, description="Enable progress tracking")
    tracking_years: List[int] = Field(
        default_factory=list,
        description="Years for which progress data is available"
    )
    on_track_threshold_pct: float = Field(
        90.0,
        ge=50.0,
        le=100.0,
        description="% of required reduction trajectory to be considered on-track"
    )
    update_frequency: str = Field("ANNUAL", description="Progress update frequency")
    use_moving_average: bool = Field(
        True,
        description="Use 3-year moving average for trend smoothing"
    )


class BaseYearRecalculationConfig(BaseModel):
    """Configuration for base year recalculation due to M&A, divestitures, etc."""
    enabled: bool = Field(True, description="Enable base year recalculation capability")
    recalculation_threshold_pct: float = Field(
        5.0,
        ge=0.0,
        le=50.0,
        description="Structural change threshold (%) triggering recalculation"
    )
    last_recalculation_type: Optional[RecalculationType] = Field(
        None,
        description="Last recalculation reason"
    )
    last_recalculation_date: Optional[datetime] = Field(
        None,
        description="Date of last recalculation"
    )


class FINZPortfolioConfig(BaseModel):
    """Configuration for FINZ V1.0 portfolio-level targets (financial institutions)."""
    enabled: bool = Field(False, description="Enable FINZ portfolio targets")
    version: str = Field("1.0", description="FINZ framework version")
    asset_classes: List[str] = Field(
        default_factory=list,
        description="Asset classes in portfolio (listed_equity, corporate_bonds, mortgages, etc.)"
    )
    data_quality_minimum: PCOFDataQuality = Field(
        PCOFDataQuality.ESTIMATED_SECTOR_AVERAGE,
        description="Minimum acceptable PCAF data quality score"
    )
    coverage_target_pct: float = Field(
        90.0,
        ge=0.0,
        le=100.0,
        description="Portfolio coverage target (%)"
    )
    target_setting_year: int = Field(
        2025,
        description="Year in which portfolio targets were set"
    )


class SubmissionReadinessConfig(BaseModel):
    """Configuration for SBTi submission readiness assessment."""
    enabled: bool = Field(True, description="Enable submission readiness checks")
    validation_criteria_count: int = Field(
        42,
        description="Total number of SBTi validation criteria (28 near-term + 14 net-zero)"
    )
    near_term_criteria_enabled: bool = Field(
        True,
        description="Validate near-term criteria (C1-C28)"
    )
    net_zero_criteria_enabled: bool = Field(
        False,
        description="Validate net-zero criteria (NZ-C1 to NZ-C14)"
    )
    target_readiness_pct: float = Field(
        90.0,
        ge=50.0,
        le=100.0,
        description="% criteria met required for READY status"
    )
    generate_action_plan: bool = Field(
        True,
        description="Generate remediation action plan for failing criteria"
    )


class OffsetPortfolioConfig(BaseModel):
    """Configuration for carbon credit and offset portfolio management."""
    strategy: OffsetStrategy = Field(
        OffsetStrategy.BOTH,
        description="Offset strategy type"
    )
    quality_minimum_score: int = Field(
        60,
        ge=0,
        le=100,
        description="Minimum credit quality score"
    )
    max_nature_based_pct: float = Field(
        50.0,
        ge=0.0,
        le=100.0,
        description="Maximum % of nature-based credits"
    )
    vcmi_target_claim: VCMIClaimTier = Field(
        VCMIClaimTier.SILVER,
        description="Target VCMI claim tier"
    )
    preferred_credit_types: List[str] = Field(
        default_factory=lambda: [
            "verified_carbon_standard",
            "gold_standard",
            "american_carbon_registry",
        ],
        description="Preferred credit registries"
    )
    shift_to_removals_by_year: int = Field(
        2040,
        ge=2025,
        le=2055,
        description="Target year for majority-removal portfolio"
    )


class ReportingConfig(BaseModel):
    """Configuration for multi-framework reporting."""
    formats: List[ReportFormat] = Field(
        default_factory=lambda: [ReportFormat.PDF, ReportFormat.HTML],
        description="Output formats"
    )
    include_cdp_mapping: bool = Field(True, description="Map to CDP C4 responses")
    include_tcfd_mapping: bool = Field(True, description="Map to TCFD disclosures")
    include_esrs_mapping: bool = Field(True, description="Map to ESRS E1 disclosures")
    include_sbti_mapping: bool = Field(True, description="Map to SBTi reporting template")
    include_ghg_protocol_mapping: bool = Field(True, description="Map to GHG Protocol")
    language: str = Field("en", description="Primary report language (ISO 639-1)")
    multi_language_support: bool = Field(False, description="Enable multi-language support")
    supported_languages: List[str] = Field(
        default_factory=lambda: ["en"],
        description="Supported languages"
    )
    review_workflow_enabled: bool = Field(True, description="Enable review/approval workflow")
    watermark_draft: bool = Field(True, description="Watermark draft documents")


class PerformanceConfig(BaseModel):
    """Configuration for runtime performance tuning."""
    cache_enabled: bool = Field(True, description="Enable caching")
    cache_ttl_seconds: int = Field(3600, ge=60, le=86400, description="Cache TTL")
    max_concurrent_calcs: int = Field(4, ge=1, le=32, description="Max concurrent calculations")
    timeout_seconds: int = Field(300, ge=30, le=3600, description="Calculation timeout")
    batch_size: int = Field(1000, ge=100, le=10000, description="Batch size for processing")
    memory_limit_mb: int = Field(4096, ge=512, le=32768, description="Memory limit (MB)")


class AuditTrailConfig(BaseModel):
    """Configuration for audit trail and provenance tracking."""
    enabled: bool = Field(True, description="Enable audit trail")
    sha256_provenance: bool = Field(True, description="Generate SHA-256 hashes")
    calculation_logging: bool = Field(True, description="Log calculation steps")
    assumption_tracking: bool = Field(True, description="Track assumptions")
    data_lineage_enabled: bool = Field(True, description="Track data lineage")
    retention_years: int = Field(7, ge=1, le=15, description="Retention period")
    external_audit_export: bool = Field(
        True,
        description="Enable export for external auditors/SBTi validators"
    )


# =============================================================================
# Main Configuration Model
# =============================================================================


class SBTiAlignmentConfig(BaseModel):
    """Main configuration for PACK-023 SBTi Alignment Pack.

    Root configuration containing all sub-configurations for comprehensive
    SBTi lifecycle management: target setting, validation, tracking, and
    submission readiness assessment.
    """

    # Temporal settings
    reporting_year: int = Field(
        DEFAULT_REPORTING_YEAR,
        ge=2020,
        le=2035,
        description="Current reporting year"
    )
    base_year: int = Field(
        DEFAULT_BASE_YEAR,
        ge=2015,
        le=2030,
        description="Base year for emissions baseline"
    )
    pack_version: str = Field("1.0.0", description="Pack configuration version")
    config_created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Configuration creation timestamp"
    )

    # Sub-configurations
    organization: OrganizationConfig = Field(
        default_factory=OrganizationConfig,
        description="Organization profile"
    )
    boundary: BoundaryConfig = Field(
        default_factory=BoundaryConfig,
        description="GHG boundary configuration"
    )
    baseline_inventory: BaselineInventoryConfig = Field(
        default_factory=BaselineInventoryConfig,
        description="Baseline inventory configuration"
    )
    scope3_screening: Scope3ScreeningConfig = Field(
        default_factory=Scope3ScreeningConfig,
        description="Scope 3 materiality screening"
    )
    target_setting: TargetSettingConfig = Field(
        default_factory=TargetSettingConfig,
        description="Target setting configuration"
    )
    sda_pathway: SDAPathwayConfig = Field(
        default_factory=SDAPathwayConfig,
        description="SDA pathway configuration"
    )
    flag_pathway: FLAGPathwayConfig = Field(
        default_factory=FLAGPathwayConfig,
        description="FLAG pathway configuration"
    )
    temperature_rating: TemperatureRatingConfig = Field(
        default_factory=TemperatureRatingConfig,
        description="Temperature rating configuration"
    )
    progress_tracking: ProgressTrackingConfig = Field(
        default_factory=ProgressTrackingConfig,
        description="Progress tracking configuration"
    )
    base_year_recalculation: BaseYearRecalculationConfig = Field(
        default_factory=BaseYearRecalculationConfig,
        description="Base year recalculation configuration"
    )
    finz_portfolio: FINZPortfolioConfig = Field(
        default_factory=FINZPortfolioConfig,
        description="FINZ portfolio target configuration"
    )
    submission_readiness: SubmissionReadinessConfig = Field(
        default_factory=SubmissionReadinessConfig,
        description="Submission readiness assessment"
    )
    offset_portfolio: OffsetPortfolioConfig = Field(
        default_factory=OffsetPortfolioConfig,
        description="Offset portfolio configuration"
    )
    reporting: ReportingConfig = Field(
        default_factory=ReportingConfig,
        description="Multi-framework reporting"
    )
    performance: PerformanceConfig = Field(
        default_factory=PerformanceConfig,
        description="Performance tuning"
    )
    audit_trail: AuditTrailConfig = Field(
        default_factory=AuditTrailConfig,
        description="Audit trail configuration"
    )

    @model_validator(mode="after")
    def validate_base_year_before_reporting(self) -> "SBTiAlignmentConfig":
        """Ensure base year is not after reporting year."""
        if self.base_year > self.reporting_year:
            raise ValueError(
                f"base_year ({self.base_year}) must not be after "
                f"reporting_year ({self.reporting_year})"
            )
        return self

    def get_enabled_engines(self) -> List[str]:
        """Return list of engines that should be enabled based on config.

        Returns:
            List of engine identifier strings.
        """
        engines = [
            "target_setting",
            "criteria_validation",
            "scope3_screening",
            "temperature_rating",
            "progress_tracking",
            "submission_readiness",
        ]

        if self.sda_pathway.enabled and self.target_setting.pathway_type == PathwayType.SDA:
            engines.append("sda_sector")

        if self.flag_pathway.enabled and self.target_setting.pathway_type == PathwayType.FLAG:
            engines.append("flag_assessment")

        if self.finz_portfolio.enabled:
            engines.append("fi_portfolio")

        if self.base_year_recalculation.enabled:
            engines.append("recalculation")

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
        >>> config = PackConfig.from_yaml("path/to/config.yaml")
    """

    pack: SBTiAlignmentConfig = Field(
        default_factory=SBTiAlignmentConfig,
        description="Main SBTi Alignment configuration"
    )
    preset_name: Optional[str] = Field(None, description="Name of loaded preset")
    config_version: str = Field("1.0.0", description="Configuration schema version")
    pack_id: str = Field("PACK-023-sbti-alignment", description="Pack identifier")

    @classmethod
    def from_preset(
        cls,
        preset_name: str,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> "PackConfig":
        """Load configuration from a named preset.

        Args:
            preset_name: Name of the preset (power, heavy_industry, manufacturing, etc.)
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
        env_overrides = _get_env_overrides("SBTI_ALIGNMENT_")
        if env_overrides:
            preset_data = _merge_config(preset_data, env_overrides)

        # Apply explicit overrides
        if overrides:
            preset_data = _merge_config(preset_data, overrides)

        pack_config = SBTiAlignmentConfig(**preset_data)
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

        pack_config = SBTiAlignmentConfig(**config_data)
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
        SBTI_ALIGNMENT_REPORTING_YEAR=2026
        SBTI_ALIGNMENT_TARGET_SETTING__AMBITION_LEVEL=CELSIUS_1_5

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


def validate_config(config: SBTiAlignmentConfig) -> List[str]:
    """Validate an SBTi configuration and return any warnings.

    Performs cross-field validation beyond what Pydantic validators cover.
    Returns advisory warnings, not hard errors.

    Args:
        config: SBTiAlignmentConfig instance to validate.

    Returns:
        List of warning messages (empty if fully valid).
    """
    warnings: List[str] = []

    # Check organization name is set
    if not config.organization.name:
        warnings.append(
            "Organization name is empty. Set organization.name for meaningful reports."
        )

    # Check Scope 3 coverage
    if config.scope3_screening.include_scope3:
        if len(config.scope3_screening.scope3_categories) < 3:
            warnings.append(
                "Fewer than 3 Scope 3 categories selected. SBTi requires covering "
                "at least 67% of Scope 3 emissions."
            )

        if config.scope3_screening.coverage_near_term_pct < 67.0:
            warnings.append(
                f"Near-term Scope 3 coverage ({config.scope3_screening.coverage_near_term_pct}%) "
                f"is below SBTi minimum of 67%."
            )

    # Check target ambition
    if config.target_setting.sbti_submission_planned:
        if config.target_setting.near_term_target_year > 2035:
            warnings.append(
                "SBTi requires near-term targets within 5-10 years of submission. "
                f"Near-term year {config.target_setting.near_term_target_year} may be too far."
            )

        if config.target_setting.long_term_target_year > 2050:
            warnings.append(
                "SBTi Net-Zero Standard requires net-zero by no later than 2050. "
                f"Long-term year {config.target_setting.long_term_target_year} exceeds this."
            )

    # Check SDA configuration
    if config.target_setting.pathway_type == PathwayType.SDA:
        if not config.sda_pathway.enabled:
            warnings.append(
                "SDA pathway selected but sda_pathway.enabled is False. "
                "Enable SDA calculations for target-setting pathway."
            )
        if not config.sda_pathway.sector:
            warnings.append(
                "SDA pathway selected but sector not configured. "
                "Set sda_pathway.sector to one of: power, cement, steel, chemicals, etc."
            )

    # Check FLAG configuration
    if config.target_setting.pathway_type == PathwayType.FLAG:
        if not config.flag_pathway.enabled:
            warnings.append(
                "FLAG pathway selected but flag_pathway.enabled is False. "
                "Enable FLAG calculations for target-setting pathway."
            )
        if not config.flag_pathway.commodities:
            warnings.append(
                "FLAG pathway selected but no commodities configured. "
                "Specify flag_pathway.commodities (beef, palm, soy, etc.)"
            )

    # Check FINZ configuration
    if config.finz_portfolio.enabled and config.organization.sector != OrganizationSector.FINANCIAL_SERVICES:
        warnings.append(
            "FINZ portfolio targets enabled for non-financial organization. "
            "FINZ is designed for financial institutions. Consider disabling."
        )

    # Check offset quality
    if config.offset_portfolio.quality_minimum_score < 40:
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


def list_available_presets() -> Dict[str, str]:
    """List all available configuration presets.

    Returns:
        Dictionary mapping preset names to descriptions.
    """
    return SUPPORTED_PRESETS.copy()
