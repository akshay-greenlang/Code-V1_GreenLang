"""
PACK-026 SME Net Zero Pack - Configuration Manager

This module implements the SMENetZeroConfig and PackConfig classes that load,
merge, and validate all configuration for the SME Net Zero Pack. It provides
comprehensive Pydantic v2 models designed specifically for small and medium
enterprises (SMEs) with constrained budgets, limited data availability, and
simplified compliance requirements.

SME Size Classifications:
    - MICRO: 1-9 employees, <EUR 2M revenue
    - SMALL: 10-49 employees, EUR 2-10M revenue
    - MEDIUM: 50-249 employees, EUR 10-50M revenue

Data Quality Tiers:
    - BRONZE: Industry averages and estimation only
    - SILVER: Basic activity data (utility bills, fuel receipts)
    - GOLD: Detailed bills, supplier questionnaires, metered data

SME Target Pathways:
    - ACA pathway only (Absolute Contraction Approach)
    - Hard-coded 1.5C ambition level
    - SBTi SME Target Route (simplified submission)

Certification Pathways:
    - SME Climate Hub (self-declaration, lowest barrier)
    - B Corp Climate Collective (third-party verified)
    - ISO 14001 (full EMS certification)
    - Carbon Trust Standard (carbon footprint verified)

Configuration Merge Order (later overrides earlier):
    1. Base pack.yaml manifest
    2. Preset YAML (micro_business / small_business / medium_business /
       service_sme / manufacturing_sme / retail_sme)
    3. Environment overrides (SME_NET_ZERO_* environment variables)
    4. Explicit runtime overrides

Regulatory Context:
    - SBTi SME Target Route v1.0 (2024)
    - GHG Protocol SME Guidance (Corporate Standard simplified)
    - GHG Protocol Scope 3 Evaluator Tool
    - SME Climate Hub Commitment Framework
    - EU Energy Efficiency Directive (recast 2023)
    - EU CSRD scope expansion considerations (Omnibus Directive)
    - ISO 14064-1:2018 (simplified boundaries)
    - ISO 14001:2015 (optional EMS pathway)
    - IPCC AR6 WG3 (Mitigation, 2022)

Grant and Funding Context:
    - EU Innovation Fund (small-scale projects)
    - National energy efficiency programmes
    - Sector-specific decarbonization grants
    - SME-focused green transition funds
    - Local/municipal climate action grants

Example:
    >>> config = PackConfig.from_preset("small_business")
    >>> print(config.pack.organization.sme_size)
    SMESize.SMALL
    >>> print(config.pack.target.near_term_reduction_pct)
    42.0
    >>> print(config.pack.data_quality.tier)
    SMEDataQualityTier.SILVER
"""

import copy
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
DEFAULT_SCOPE3_SME_CATEGORIES: List[int] = [1, 6, 7]
DEFAULT_SCOPE3_MEDIUM_CATEGORIES: List[int] = [1, 2, 3, 4, 5, 6, 7]

SUPPORTED_PRESETS: Dict[str, str] = {
    "micro_business": "Micro business (1-9 employees, <EUR 2M), BRONZE data tier, minimal scope",
    "small_business": "Small business (10-49 employees, EUR 2-10M), SILVER data tier, basic scope",
    "medium_business": "Medium business (50-249 employees, EUR 10-50M), GOLD data tier, full scope",
    "service_sme": "Service-sector SME (tech, consulting, professional services), S3-dominant",
    "manufacturing_sme": "Manufacturing SME (small production, workshops), S1+S2-heavy",
    "retail_sme": "Retail SME (shops, small chains), S3-dominant with packaging focus",
}

# SME size thresholds (EU definition, Commission Recommendation 2003/361/EC)
SME_SIZE_THRESHOLDS: Dict[str, Dict[str, Any]] = {
    "MICRO": {
        "min_employees": 1,
        "max_employees": 9,
        "max_revenue_eur": 2_000_000,
        "description": "Micro enterprise (1-9 employees, <EUR 2M revenue)",
    },
    "SMALL": {
        "min_employees": 10,
        "max_employees": 49,
        "max_revenue_eur": 10_000_000,
        "description": "Small enterprise (10-49 employees, EUR 2-10M revenue)",
    },
    "MEDIUM": {
        "min_employees": 50,
        "max_employees": 249,
        "max_revenue_eur": 50_000_000,
        "description": "Medium enterprise (50-249 employees, EUR 10-50M revenue)",
    },
}

# =============================================================================
# Enums - SME-specific enumeration types (16 enums)
# =============================================================================


class SMESize(str, Enum):
    """SME size classification per EU Commission Recommendation 2003/361/EC."""

    MICRO = "MICRO"
    SMALL = "SMALL"
    MEDIUM = "MEDIUM"


class SMESector(str, Enum):
    """SME sector classification (simplified from NACE)."""

    SERVICES = "SERVICES"
    MANUFACTURING = "MANUFACTURING"
    RETAIL = "RETAIL"
    HOSPITALITY = "HOSPITALITY"
    CONSTRUCTION = "CONSTRUCTION"
    AGRICULTURE = "AGRICULTURE"
    TRANSPORT = "TRANSPORT"
    TECHNOLOGY = "TECHNOLOGY"
    HEALTHCARE = "HEALTHCARE"
    OTHER = "OTHER"


class SMEDataQualityTier(str, Enum):
    """SME data quality tier for emissions estimation.

    BRONZE: Industry averages, estimation-only, no primary data
    SILVER: Basic activity data (utility bills, fuel receipts, basic records)
    GOLD: Detailed metered data, supplier questionnaires, verified bills
    """

    BRONZE = "BRONZE"
    SILVER = "SILVER"
    GOLD = "GOLD"


class BoundaryMethod(str, Enum):
    """GHG Protocol organizational boundary method (SME: operational control only)."""

    OPERATIONAL_CONTROL = "OPERATIONAL_CONTROL"


class PathwayType(str, Enum):
    """SBTi target-setting pathway type (SME: ACA only)."""

    ACA = "ACA"


class Scope3Method(str, Enum):
    """Scope 3 calculation methodology for SMEs."""

    SPEND_BASED = "SPEND_BASED"
    ACTIVITY_BASED = "ACTIVITY_BASED"
    HYBRID = "HYBRID"


class OffsetStrategy(str, Enum):
    """Carbon offset strategy type for SMEs."""

    NONE = "NONE"
    COMPENSATION_ONLY = "COMPENSATION_ONLY"


class CertificationPathway(str, Enum):
    """Available certification pathways for SMEs."""

    SME_CLIMATE_HUB = "SME_CLIMATE_HUB"
    B_CORP = "B_CORP"
    ISO_14001 = "ISO_14001"
    CARBON_TRUST = "CARBON_TRUST"
    NONE = "NONE"


class VerificationLevel(str, Enum):
    """Verification/assurance level for SME emissions data."""

    SELF_DECLARATION = "SELF_DECLARATION"
    THIRD_PARTY_LIGHT = "THIRD_PARTY_LIGHT"
    THIRD_PARTY_FULL = "THIRD_PARTY_FULL"


class AccountingSoftware(str, Enum):
    """Accounting software integration options for SMEs."""

    XERO = "XERO"
    QUICKBOOKS = "QUICKBOOKS"
    SAGE = "SAGE"
    FRESHBOOKS = "FRESHBOOKS"
    WAVE = "WAVE"
    SAP_BUSINESS_ONE = "SAP_BUSINESS_ONE"
    DYNAMICS_365_BC = "DYNAMICS_365_BC"
    MANUAL = "MANUAL"
    NONE = "NONE"


class MaturityAssessment(str, Enum):
    """Net-zero maturity assessment mode for SMEs."""

    QUICK = "QUICK"
    SKIP = "SKIP"


class GrantCategory(str, Enum):
    """Grant/funding category for SME decarbonization."""

    MICRO_GRANT = "MICRO_GRANT"
    LOCAL_PROGRAM = "LOCAL_PROGRAM"
    NATIONAL_PROGRAM = "NATIONAL_PROGRAM"
    EU_PROGRAM = "EU_PROGRAM"
    SECTOR_SPECIFIC = "SECTOR_SPECIFIC"
    ENERGY_EFFICIENCY = "ENERGY_EFFICIENCY"
    INNOVATION = "INNOVATION"
    CIRCULAR_ECONOMY = "CIRCULAR_ECONOMY"
    INDUSTRIAL_DECARBONIZATION = "INDUSTRIAL_DECARBONIZATION"
    RETAIL_ENERGY = "RETAIL_ENERGY"
    TECHNOLOGY_ADOPTION = "TECHNOLOGY_ADOPTION"


class QuickWinCategory(str, Enum):
    """Quick-win action categories for SMEs."""

    LED_LIGHTING = "LED_LIGHTING"
    SMART_THERMOSTAT = "SMART_THERMOSTAT"
    RENEWABLE_TARIFF = "RENEWABLE_TARIFF"
    RENEWABLE_PPA = "RENEWABLE_PPA"
    REMOTE_WORK_POLICY = "REMOTE_WORK_POLICY"
    TRAVEL_REDUCTION = "TRAVEL_REDUCTION"
    WASTE_REDUCTION = "WASTE_REDUCTION"
    INSULATION = "INSULATION"
    HVAC_UPGRADE = "HVAC_UPGRADE"
    PROCESS_EFFICIENCY = "PROCESS_EFFICIENCY"
    REFRIGERATION_UPGRADE = "REFRIGERATION_UPGRADE"
    PACKAGING_REDUCTION = "PACKAGING_REDUCTION"
    SUPPLIER_ENGAGEMENT = "SUPPLIER_ENGAGEMENT"
    EV_FLEET = "EV_FLEET"
    HEAT_RECOVERY = "HEAT_RECOVERY"
    MATERIAL_SUBSTITUTION = "MATERIAL_SUBSTITUTION"
    STORE_ENERGY_OPTIMIZATION = "STORE_ENERGY_OPTIMIZATION"
    TRANSPORT_OPTIMIZATION = "TRANSPORT_OPTIMIZATION"
    GREEN_PROCUREMENT = "GREEN_PROCUREMENT"
    BEHAVIORAL_CHANGE = "BEHAVIORAL_CHANGE"


class EmissionsProfileType(str, Enum):
    """Typical emissions profile types for SME sectors."""

    SCOPE3_DOMINANT = "SCOPE3_DOMINANT"
    SCOPE1_HEAVY = "SCOPE1_HEAVY"
    BALANCED = "BALANCED"


# =============================================================================
# Reference Data Constants
# =============================================================================

# Sector-specific default emissions profile split (S1%, S2%, S3%)
SECTOR_EMISSIONS_PROFILE: Dict[str, Dict[str, float]] = {
    "SERVICES": {
        "scope1_pct": 5.0,
        "scope2_pct": 10.0,
        "scope3_pct": 85.0,
        "profile_type": "SCOPE3_DOMINANT",
    },
    "MANUFACTURING": {
        "scope1_pct": 40.0,
        "scope2_pct": 35.0,
        "scope3_pct": 25.0,
        "profile_type": "SCOPE1_HEAVY",
    },
    "RETAIL": {
        "scope1_pct": 8.0,
        "scope2_pct": 12.0,
        "scope3_pct": 80.0,
        "profile_type": "SCOPE3_DOMINANT",
    },
    "HOSPITALITY": {
        "scope1_pct": 25.0,
        "scope2_pct": 30.0,
        "scope3_pct": 45.0,
        "profile_type": "BALANCED",
    },
    "CONSTRUCTION": {
        "scope1_pct": 35.0,
        "scope2_pct": 15.0,
        "scope3_pct": 50.0,
        "profile_type": "BALANCED",
    },
    "AGRICULTURE": {
        "scope1_pct": 55.0,
        "scope2_pct": 10.0,
        "scope3_pct": 35.0,
        "profile_type": "SCOPE1_HEAVY",
    },
    "TRANSPORT": {
        "scope1_pct": 60.0,
        "scope2_pct": 10.0,
        "scope3_pct": 30.0,
        "profile_type": "SCOPE1_HEAVY",
    },
    "TECHNOLOGY": {
        "scope1_pct": 3.0,
        "scope2_pct": 12.0,
        "scope3_pct": 85.0,
        "profile_type": "SCOPE3_DOMINANT",
    },
    "HEALTHCARE": {
        "scope1_pct": 20.0,
        "scope2_pct": 25.0,
        "scope3_pct": 55.0,
        "profile_type": "BALANCED",
    },
    "OTHER": {
        "scope1_pct": 20.0,
        "scope2_pct": 20.0,
        "scope3_pct": 60.0,
        "profile_type": "BALANCED",
    },
}

# SME Scope 3 category priorities by sector
SME_SCOPE3_PRIORITIES: Dict[str, List[int]] = {
    "SERVICES": [1, 6, 7],
    "MANUFACTURING": [1, 3, 4, 5],
    "RETAIL": [1, 4, 9],
    "HOSPITALITY": [1, 3, 5],
    "CONSTRUCTION": [1, 2, 4],
    "AGRICULTURE": [1, 3, 4],
    "TRANSPORT": [1, 3],
    "TECHNOLOGY": [1, 6, 7],
    "HEALTHCARE": [1, 2, 4],
    "OTHER": [1, 6, 7],
}

# SME quick wins by sector (top recommended actions)
SME_QUICK_WINS: Dict[str, List[str]] = {
    "SERVICES": [
        "RENEWABLE_TARIFF",
        "REMOTE_WORK_POLICY",
        "TRAVEL_REDUCTION",
        "LED_LIGHTING",
        "SMART_THERMOSTAT",
    ],
    "MANUFACTURING": [
        "PROCESS_EFFICIENCY",
        "HVAC_UPGRADE",
        "RENEWABLE_TARIFF",
        "LED_LIGHTING",
        "HEAT_RECOVERY",
    ],
    "RETAIL": [
        "LED_LIGHTING",
        "REFRIGERATION_UPGRADE",
        "PACKAGING_REDUCTION",
        "RENEWABLE_TARIFF",
        "STORE_ENERGY_OPTIMIZATION",
    ],
    "HOSPITALITY": [
        "LED_LIGHTING",
        "SMART_THERMOSTAT",
        "WASTE_REDUCTION",
        "RENEWABLE_TARIFF",
        "INSULATION",
    ],
    "CONSTRUCTION": [
        "PROCESS_EFFICIENCY",
        "EV_FLEET",
        "MATERIAL_SUBSTITUTION",
        "RENEWABLE_TARIFF",
        "WASTE_REDUCTION",
    ],
    "AGRICULTURE": [
        "RENEWABLE_TARIFF",
        "PROCESS_EFFICIENCY",
        "WASTE_REDUCTION",
        "INSULATION",
        "SMART_THERMOSTAT",
    ],
    "TRANSPORT": [
        "EV_FLEET",
        "TRANSPORT_OPTIMIZATION",
        "RENEWABLE_TARIFF",
        "BEHAVIORAL_CHANGE",
        "SMART_THERMOSTAT",
    ],
    "TECHNOLOGY": [
        "RENEWABLE_PPA",
        "REMOTE_WORK_POLICY",
        "TRAVEL_REDUCTION",
        "GREEN_PROCUREMENT",
        "BEHAVIORAL_CHANGE",
    ],
    "HEALTHCARE": [
        "LED_LIGHTING",
        "HVAC_UPGRADE",
        "RENEWABLE_TARIFF",
        "WASTE_REDUCTION",
        "GREEN_PROCUREMENT",
    ],
    "OTHER": [
        "RENEWABLE_TARIFF",
        "LED_LIGHTING",
        "SMART_THERMOSTAT",
        "WASTE_REDUCTION",
        "BEHAVIORAL_CHANGE",
    ],
}

# IPCC AR6 GWP100 values for common GHGs (subset for SME use)
IPCC_AR6_GWP100: Dict[str, int] = {
    "CO2": 1,
    "CH4": 27,
    "N2O": 273,
    "HFC_134A": 1430,
    "R404A": 3922,
    "R410A": 2088,
    "SF6": 25200,
}

# SBTi SME target route parameters (simplified)
SBTI_SME_PARAMETERS: Dict[str, Any] = {
    "minimum_reduction_pct_by_2030": 42.0,
    "recommended_reduction_pct_by_2030": 50.0,
    "ambition_level": "CELSIUS_1_5",
    "pathway_type": "ACA",
    "annual_linear_reduction_pct": 4.2,
    "scope1_2_coverage_pct": 95.0,
    "scope3_mandatory": False,
    "scope3_recommended_threshold_pct": 40.0,
    "net_zero_year": 2050,
    "long_term_reduction_pct": 90.0,
    "residual_emissions_max_pct": 10.0,
}

# SME certification pathways with requirements
CERTIFICATION_INFO: Dict[str, Dict[str, Any]] = {
    "SME_CLIMATE_HUB": {
        "name": "SME Climate Hub",
        "barrier": "LOW",
        "cost_eur_range": "0-500",
        "verification": "SELF_DECLARATION",
        "time_to_certify_months": 1,
        "requirements": [
            "Commit to halve emissions by 2030",
            "Achieve net-zero by 2050",
            "Measure and report annually",
        ],
        "suitable_sizes": ["MICRO", "SMALL", "MEDIUM"],
    },
    "B_CORP": {
        "name": "B Corp Climate Collective",
        "barrier": "MEDIUM",
        "cost_eur_range": "1000-5000",
        "verification": "THIRD_PARTY_LIGHT",
        "time_to_certify_months": 6,
        "requirements": [
            "B Impact Assessment score >= 80",
            "Climate action plan published",
            "Annual emissions reporting",
            "Science-based target commitment",
        ],
        "suitable_sizes": ["SMALL", "MEDIUM"],
    },
    "ISO_14001": {
        "name": "ISO 14001 Environmental Management System",
        "barrier": "HIGH",
        "cost_eur_range": "5000-25000",
        "verification": "THIRD_PARTY_FULL",
        "time_to_certify_months": 12,
        "requirements": [
            "Full EMS implementation",
            "Internal audit program",
            "Management review process",
            "Continuous improvement evidence",
            "Legal compliance register",
        ],
        "suitable_sizes": ["MEDIUM"],
    },
    "CARBON_TRUST": {
        "name": "Carbon Trust Standard",
        "barrier": "MEDIUM_HIGH",
        "cost_eur_range": "3000-15000",
        "verification": "THIRD_PARTY_FULL",
        "time_to_certify_months": 6,
        "requirements": [
            "Verified carbon footprint (S1+2 minimum)",
            "Demonstrated year-on-year reduction",
            "Carbon management plan",
            "Board-level commitment",
        ],
        "suitable_sizes": ["SMALL", "MEDIUM"],
    },
}

# SME budget ranges by size
SME_BUDGET_RANGES: Dict[str, Dict[str, float]] = {
    "MICRO": {
        "min_eur": 0,
        "max_eur": 5_000,
        "typical_eur": 2_000,
    },
    "SMALL": {
        "min_eur": 5_000,
        "max_eur": 20_000,
        "typical_eur": 10_000,
    },
    "MEDIUM": {
        "min_eur": 20_000,
        "max_eur": 100_000,
        "typical_eur": 50_000,
    },
}

# SME report templates (3 simplified templates)
SME_REPORT_TEMPLATES: Dict[str, Dict[str, Any]] = {
    "net_zero_summary": {
        "name": "Net Zero Summary Report",
        "description": "One-page executive summary of net-zero status and next steps",
        "pages": 2,
        "formats": ["PDF", "EXCEL"],
        "sections": [
            "emissions_overview",
            "target_progress",
            "top_actions",
            "next_steps",
        ],
    },
    "action_plan": {
        "name": "SME Action Plan",
        "description": "Prioritized list of reduction actions with costs and savings",
        "pages": 5,
        "formats": ["PDF", "EXCEL"],
        "sections": [
            "current_footprint",
            "quick_wins",
            "medium_term_actions",
            "implementation_timeline",
            "budget_summary",
            "grant_opportunities",
        ],
    },
    "annual_progress": {
        "name": "Annual Progress Report",
        "description": "Year-over-year progress tracking for stakeholder communication",
        "pages": 4,
        "formats": ["PDF", "EXCEL"],
        "sections": [
            "year_in_review",
            "emissions_comparison",
            "actions_completed",
            "actions_planned",
            "certification_status",
        ],
    },
}

# Sector display names and SME-specific guidance
SME_SECTOR_INFO: Dict[str, Dict[str, Any]] = {
    "SERVICES": {
        "name": "Professional Services",
        "typical_scope_split": "S1: 5%, S2: 10%, S3: 85%",
        "profile_type": "SCOPE3_DOMINANT",
        "key_levers": [
            "renewable electricity PPA",
            "remote work policies",
            "business travel reduction",
            "supplier engagement",
        ],
        "data_sources": [
            "electricity bills",
            "gas bills",
            "travel expense reports",
            "procurement spend data",
        ],
    },
    "MANUFACTURING": {
        "name": "Small Manufacturing",
        "typical_scope_split": "S1: 40%, S2: 35%, S3: 25%",
        "profile_type": "SCOPE1_HEAVY",
        "key_levers": [
            "process efficiency improvement",
            "HVAC upgrade",
            "renewable energy",
            "waste heat recovery",
            "material substitution",
        ],
        "data_sources": [
            "fuel purchase records",
            "electricity bills",
            "production logs",
            "material purchase records",
        ],
    },
    "RETAIL": {
        "name": "Retail & Shops",
        "typical_scope_split": "S1: 8%, S2: 12%, S3: 80%",
        "profile_type": "SCOPE3_DOMINANT",
        "key_levers": [
            "LED lighting rollout",
            "refrigeration efficiency",
            "packaging reduction",
            "supplier engagement",
            "transport optimization",
        ],
        "data_sources": [
            "electricity bills",
            "refrigerant logs",
            "product purchase records",
            "delivery records",
        ],
    },
    "HOSPITALITY": {
        "name": "Hospitality & Food Service",
        "typical_scope_split": "S1: 25%, S2: 30%, S3: 45%",
        "profile_type": "BALANCED",
        "key_levers": [
            "energy-efficient equipment",
            "food waste reduction",
            "renewable energy",
            "insulation improvements",
        ],
        "data_sources": [
            "utility bills",
            "gas/fuel records",
            "food purchase records",
            "waste disposal records",
        ],
    },
    "CONSTRUCTION": {
        "name": "Construction & Trades",
        "typical_scope_split": "S1: 35%, S2: 15%, S3: 50%",
        "profile_type": "BALANCED",
        "key_levers": [
            "fleet electrification",
            "low-carbon materials",
            "waste reduction on site",
            "renewable energy",
        ],
        "data_sources": [
            "fuel purchase records",
            "electricity bills",
            "material purchase records",
            "vehicle fuel logs",
        ],
    },
    "AGRICULTURE": {
        "name": "Agriculture & Farming",
        "typical_scope_split": "S1: 55%, S2: 10%, S3: 35%",
        "profile_type": "SCOPE1_HEAVY",
        "key_levers": [
            "renewable energy on farm",
            "fertilizer optimization",
            "livestock management",
            "machinery efficiency",
        ],
        "data_sources": [
            "fuel records",
            "fertilizer purchase records",
            "electricity bills",
            "livestock counts",
        ],
    },
    "TRANSPORT": {
        "name": "Transport & Delivery",
        "typical_scope_split": "S1: 60%, S2: 10%, S3: 30%",
        "profile_type": "SCOPE1_HEAVY",
        "key_levers": [
            "fleet electrification",
            "route optimization",
            "driver training (eco-driving)",
            "vehicle right-sizing",
        ],
        "data_sources": [
            "vehicle fuel records",
            "fleet management data",
            "electricity bills",
            "route/distance logs",
        ],
    },
    "TECHNOLOGY": {
        "name": "Technology & Software",
        "typical_scope_split": "S1: 3%, S2: 12%, S3: 85%",
        "profile_type": "SCOPE3_DOMINANT",
        "key_levers": [
            "renewable energy PPA",
            "remote-first policies",
            "cloud efficiency optimization",
            "green procurement",
        ],
        "data_sources": [
            "cloud provider reports",
            "electricity bills",
            "travel expense data",
            "procurement spend data",
        ],
    },
    "HEALTHCARE": {
        "name": "Healthcare & Wellbeing",
        "typical_scope_split": "S1: 20%, S2: 25%, S3: 55%",
        "profile_type": "BALANCED",
        "key_levers": [
            "energy-efficient HVAC",
            "LED lighting",
            "sustainable procurement",
            "waste management",
        ],
        "data_sources": [
            "utility bills",
            "gas/fuel records",
            "medical supply records",
            "waste disposal records",
        ],
    },
    "OTHER": {
        "name": "Other",
        "typical_scope_split": "S1: 20%, S2: 20%, S3: 60%",
        "profile_type": "BALANCED",
        "key_levers": [
            "energy efficiency",
            "renewable energy",
            "waste reduction",
            "green procurement",
        ],
        "data_sources": [
            "utility bills",
            "fuel records",
            "procurement records",
            "waste records",
        ],
    },
}


# =============================================================================
# Pydantic Sub-Config Models (12 models)
# =============================================================================


class SMEOrganizationConfig(BaseModel):
    """Configuration for the SME organization profile.

    Defines the company identity, SME size classification, sector, and
    operational characteristics that drive engine configuration.
    """

    name: str = Field(
        "",
        description="Legal entity name of the organization",
    )
    sme_size: SMESize = Field(
        SMESize.SMALL,
        description="SME size classification (MICRO: 1-9, SMALL: 10-49, MEDIUM: 50-249)",
    )
    sector: SMESector = Field(
        SMESector.OTHER,
        description="Primary business sector",
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
        description="Annual revenue in EUR (used for intensity metrics and size validation)",
    )
    employee_count: Optional[int] = Field(
        None,
        ge=1,
        le=249,
        description="Total number of employees (1-249 for SME)",
    )
    fiscal_year_end: str = Field(
        "12-31",
        description="Fiscal year end date in MM-DD format",
    )
    nace_code: Optional[str] = Field(
        None,
        description="NACE industry classification code (optional for benchmarking)",
    )
    accounting_software: AccountingSoftware = Field(
        AccountingSoftware.NONE,
        description="Accounting software in use for spend-data extraction",
    )
    number_of_sites: int = Field(
        1,
        ge=1,
        le=50,
        description="Number of physical sites/locations (SMEs typically 1-10)",
    )
    has_fleet: bool = Field(
        False,
        description="Whether the organization has a vehicle fleet",
    )
    has_production_facility: bool = Field(
        False,
        description="Whether the organization has manufacturing/production facilities",
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

    @model_validator(mode="after")
    def validate_employee_count_matches_size(self) -> "SMEOrganizationConfig":
        """Warn if employee count does not match declared SME size."""
        if self.employee_count is not None:
            thresholds = SME_SIZE_THRESHOLDS.get(self.sme_size.value, {})
            min_emp = thresholds.get("min_employees", 1)
            max_emp = thresholds.get("max_employees", 249)
            if self.employee_count < min_emp or self.employee_count > max_emp:
                logger.warning(
                    "Employee count %d does not match declared SME size %s "
                    "(expected %d-%d). Consider adjusting sme_size.",
                    self.employee_count,
                    self.sme_size.value,
                    min_emp,
                    max_emp,
                )
        return self


class SMEDataQualityConfig(BaseModel):
    """Configuration for SME data quality tier.

    Controls the level of data granularity expected and the estimation
    methods used when primary data is unavailable.
    """

    tier: SMEDataQualityTier = Field(
        SMEDataQualityTier.SILVER,
        description="Data quality tier: BRONZE (estimates), SILVER (bills), GOLD (metered)",
    )
    allow_industry_averages: bool = Field(
        True,
        description="Allow industry-average emission factors when primary data unavailable",
    )
    allow_spend_based_estimation: bool = Field(
        True,
        description="Allow spend-based estimation for Scope 3 categories",
    )
    require_utility_bills: bool = Field(
        False,
        description="Require actual utility bills for Scope 1+2 (SILVER/GOLD tiers)",
    )
    require_supplier_data: bool = Field(
        False,
        description="Require supplier-specific emission data (GOLD tier only)",
    )
    data_completeness_target_pct: float = Field(
        60.0,
        ge=0.0,
        le=100.0,
        description="Target data completeness percentage before gap-filling",
    )
    emission_factor_source: str = Field(
        "DEFRA_2024",
        description="Default emission factor database (DEFRA, EPA, ADEME, ecoinvent)",
    )
    data_improvement_plan: bool = Field(
        True,
        description="Generate a data improvement plan with prioritized next steps",
    )

    @model_validator(mode="after")
    def validate_tier_consistency(self) -> "SMEDataQualityConfig":
        """Ensure tier settings are consistent."""
        if self.tier == SMEDataQualityTier.BRONZE:
            if self.require_utility_bills:
                logger.warning(
                    "BRONZE tier typically uses industry averages. "
                    "require_utility_bills is set but may not be achievable."
                )
        if self.tier == SMEDataQualityTier.GOLD:
            if not self.require_utility_bills:
                object.__setattr__(self, "require_utility_bills", True)
        return self


class SMEBoundaryConfig(BaseModel):
    """Configuration for GHG Protocol organizational boundary (SME-simplified).

    SMEs use operational control approach only, with simplified boundary
    typically covering a single legal entity and 1-10 sites.
    """

    method: BoundaryMethod = Field(
        BoundaryMethod.OPERATIONAL_CONTROL,
        description="GHG Protocol boundary approach (always operational control for SME)",
    )
    reporting_currency: str = Field(
        "EUR",
        description="Reporting currency for financial metrics (ISO 4217)",
    )
    base_year_recalculation_threshold_pct: float = Field(
        10.0,
        ge=0.0,
        le=100.0,
        description="Threshold (%) for structural change triggering base year recalculation",
    )
    include_all_sites: bool = Field(
        True,
        description="Include all operational sites in boundary (recommended)",
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


class SMEScopeConfig(BaseModel):
    """Configuration for Scope 1, 2, and 3 emissions calculation (SME-simplified).

    SMEs measure Scope 1+2 as mandatory and use spend-based Scope 3
    for a simplified subset of categories relevant to their sector.
    """

    include_scope1: bool = Field(
        True,
        description="Include Scope 1 direct emissions (always true for SME)",
    )
    include_scope2: bool = Field(
        True,
        description="Include Scope 2 indirect emissions (always true for SME)",
    )
    include_scope3: bool = Field(
        True,
        description="Include Scope 3 value chain emissions (recommended but optional for micro)",
    )
    scope1_estimation_method: str = Field(
        "activity_based",
        description="Scope 1 estimation method (activity_based or estimation)",
    )
    scope2_methods: List[str] = Field(
        default_factory=lambda: ["location_based"],
        description="Scope 2 calculation methods to apply",
    )
    scope3_categories: List[int] = Field(
        default_factory=lambda: DEFAULT_SCOPE3_SME_CATEGORIES.copy(),
        description="Scope 3 categories (1-15) included (SME: simplified subset)",
    )
    scope3_method: Scope3Method = Field(
        Scope3Method.SPEND_BASED,
        description="Default Scope 3 calculation methodology (spend-based for SME)",
    )
    scope3_category_methods: Dict[str, str] = Field(
        default_factory=lambda: {
            "default": "spend_based",
        },
        description="Override calculation method per Scope 3 category",
    )
    scope3_materiality_threshold_pct: float = Field(
        5.0,
        ge=0.0,
        le=100.0,
        description="Threshold (%) below which a Scope 3 category is immaterial",
    )
    scope3_optional: bool = Field(
        True,
        description="Whether Scope 3 is optional for this SME size (micro: optional)",
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


class SMETargetConfig(BaseModel):
    """Configuration for SME science-based target setting.

    Hard-coded to ACA pathway and 1.5C ambition per SBTi SME Target Route.
    Pre-set reduction targets by SME size: 30-50% by 2030.
    """

    ambition_level: str = Field(
        "CELSIUS_1_5",
        description="SBTi target ambition level (always 1.5C for SME route)",
    )
    pathway_type: PathwayType = Field(
        PathwayType.ACA,
        description="SBTi target-setting pathway (always ACA for SME)",
    )
    near_term_target_year: int = Field(
        DEFAULT_TARGET_YEAR_NEAR,
        ge=2025,
        le=2035,
        description="Near-term target year (2030 standard for SBTi SME route)",
    )
    long_term_target_year: int = Field(
        DEFAULT_TARGET_YEAR_LONG,
        ge=2040,
        le=2055,
        description="Long-term / net-zero target year (2050 standard)",
    )
    near_term_reduction_pct: float = Field(
        42.0,
        ge=20.0,
        le=70.0,
        description="Near-term reduction target (% from base year). SBTi SME min: 42%",
    )
    long_term_reduction_pct: float = Field(
        90.0,
        ge=80.0,
        le=100.0,
        description="Long-term reduction target (% from base year). SBTi: 90% minimum",
    )
    annual_linear_reduction_pct: float = Field(
        4.2,
        ge=1.0,
        le=10.0,
        description="Annual linear reduction rate to stay on track (SBTi SME: 4.2%/yr)",
    )
    sbti_sme_route: bool = Field(
        True,
        description="Use SBTi SME simplified target route (recommended for <250 employees)",
    )
    sbti_submission_planned: bool = Field(
        False,
        description="Whether formal SBTi target submission is planned",
    )
    sme_climate_hub_commitment: bool = Field(
        True,
        description="Whether SME Climate Hub commitment is the primary pledge mechanism",
    )

    @model_validator(mode="after")
    def validate_target_years(self) -> "SMETargetConfig":
        """Ensure long-term target year is after near-term target year."""
        if self.long_term_target_year <= self.near_term_target_year:
            raise ValueError(
                f"long_term_target_year ({self.long_term_target_year}) must be "
                f"after near_term_target_year ({self.near_term_target_year})"
            )
        return self

    @model_validator(mode="after")
    def validate_sbti_reduction_minimum(self) -> "SMETargetConfig":
        """Warn if near-term reduction is below SBTi SME minimum."""
        if self.sbti_sme_route and self.near_term_reduction_pct < 42.0:
            logger.warning(
                "SBTi SME route requires at least 42%% reduction by %d. "
                "Current target: %.1f%%. This may be acceptable for micro "
                "enterprises with practical constraints.",
                self.near_term_target_year,
                self.near_term_reduction_pct,
            )
        return self


class SMEReductionConfig(BaseModel):
    """Configuration for SME decarbonization actions.

    Limited to max 10 actions and a 3-year roadmap for practical SME
    implementation. Focus on quick wins with clear ROI.
    """

    max_actions: int = Field(
        10,
        ge=1,
        le=25,
        description="Maximum number of reduction actions to evaluate (SME: max 10)",
    )
    planning_horizon_years: int = Field(
        3,
        ge=1,
        le=10,
        description="Planning horizon in years for roadmap (SME: 3-year default)",
    )
    budget_constraint_eur: Optional[float] = Field(
        None,
        ge=0,
        description="Maximum total investment budget (EUR) for reduction actions",
    )
    budget_range_min_eur: float = Field(
        0.0,
        ge=0,
        description="Minimum expected budget range (EUR)",
    )
    budget_range_max_eur: float = Field(
        20_000.0,
        ge=0,
        description="Maximum expected budget range (EUR)",
    )
    include_quick_wins: bool = Field(
        True,
        description="Include quick-win actions with <12 month payback",
    )
    quick_win_max_count: int = Field(
        5,
        ge=1,
        le=10,
        description="Maximum number of quick wins to recommend",
    )
    include_renewable_procurement: bool = Field(
        True,
        description="Include renewable electricity procurement as a lever",
    )
    include_energy_efficiency: bool = Field(
        True,
        description="Include energy efficiency measures (LED, HVAC, insulation)",
    )
    include_supplier_engagement: bool = Field(
        False,
        description="Include Scope 3 supplier engagement (MEDIUM tier and above)",
    )
    include_fleet_electrification: bool = Field(
        False,
        description="Include fleet electrification as a reduction lever",
    )
    include_process_optimization: bool = Field(
        False,
        description="Include process optimization (manufacturing SMEs only)",
    )
    include_behavioral_change: bool = Field(
        True,
        description="Include zero-cost behavioral change actions",
    )
    include_macc_analysis: bool = Field(
        False,
        description="Include MACC (Marginal Abatement Cost Curve) analysis",
    )
    carbon_price_eur_per_tco2e: float = Field(
        50.0,
        ge=0.0,
        description="Internal carbon price (EUR/tCO2e) for action valuation",
    )
    discount_rate_pct: float = Field(
        8.0,
        ge=0.0,
        le=30.0,
        description="Discount rate (%) for payback calculations",
    )
    scenario_count: int = Field(
        2,
        ge=1,
        le=5,
        description="Number of reduction scenarios to model (SME: 2 max)",
    )


class SMEGrantConfig(BaseModel):
    """Configuration for SME grant and funding preferences.

    Identifies relevant grant programs and funding mechanisms to
    support the SME decarbonization investment.
    """

    enabled: bool = Field(
        True,
        description="Enable grant opportunity matching",
    )
    preferred_categories: List[GrantCategory] = Field(
        default_factory=lambda: [
            GrantCategory.ENERGY_EFFICIENCY,
            GrantCategory.LOCAL_PROGRAM,
        ],
        description="Preferred grant categories to search for",
    )
    max_application_complexity: str = Field(
        "LOW",
        description="Maximum grant application complexity (LOW/MEDIUM/HIGH)",
    )
    region_focus: str = Field(
        "NATIONAL",
        description="Geographic focus for grant search (LOCAL/NATIONAL/EU)",
    )
    sector_specific: bool = Field(
        True,
        description="Include sector-specific grant programs",
    )
    include_tax_incentives: bool = Field(
        True,
        description="Include tax incentives and reliefs in recommendations",
    )
    include_green_loans: bool = Field(
        True,
        description="Include green loan and finance products",
    )
    budget_match_required: bool = Field(
        False,
        description="Only show grants within the declared budget range",
    )


class SMECertificationConfig(BaseModel):
    """Configuration for SME certification pathway selection.

    Recommends the most appropriate certification based on SME size,
    budget, and readiness level.
    """

    primary_pathway: CertificationPathway = Field(
        CertificationPathway.SME_CLIMATE_HUB,
        description="Primary certification pathway",
    )
    secondary_pathway: Optional[CertificationPathway] = Field(
        None,
        description="Secondary/aspirational certification pathway",
    )
    auto_recommend: bool = Field(
        True,
        description="Automatically recommend certification based on size and readiness",
    )
    readiness_assessment: bool = Field(
        True,
        description="Run certification readiness assessment",
    )
    max_certification_budget_eur: float = Field(
        1000.0,
        ge=0,
        description="Maximum budget for certification costs (EUR)",
    )
    max_time_months: int = Field(
        6,
        ge=1,
        le=24,
        description="Maximum acceptable time to achieve certification (months)",
    )


class SMEReportingConfig(BaseModel):
    """Configuration for SME reporting (simplified: PDF + Excel only, 3 templates).

    SMEs generate lightweight reports focused on practical action items
    rather than comprehensive disclosure frameworks.
    """

    formats: List[ReportFormat] = Field(
        default_factory=lambda: [ReportFormat.PDF, ReportFormat.EXCEL],
        description="Output formats for generated reports (PDF and Excel only for SME)",
    )
    templates: List[str] = Field(
        default_factory=lambda: [
            "net_zero_summary",
            "action_plan",
            "annual_progress",
        ],
        description="Report templates to generate (3 SME templates available)",
    )
    include_cdp_mapping: bool = Field(
        False,
        description="Map outputs to CDP (typically disabled for SMEs)",
    )
    include_sbti_mapping: bool = Field(
        True,
        description="Map outputs to SBTi SME target route template",
    )
    include_sme_climate_hub_mapping: bool = Field(
        True,
        description="Map outputs to SME Climate Hub reporting format",
    )
    language: str = Field(
        "en",
        description="Primary report language (ISO 639-1)",
    )
    review_workflow_enabled: bool = Field(
        False,
        description="Enable review and approval workflow (typically disabled for SME)",
    )
    watermark_draft: bool = Field(
        True,
        description="Apply DRAFT watermark to unapproved documents",
    )
    include_grant_recommendations: bool = Field(
        True,
        description="Include grant/funding recommendations in action plan report",
    )
    include_certification_roadmap: bool = Field(
        True,
        description="Include certification pathway roadmap in reports",
    )


class SMEPerformanceConfig(BaseModel):
    """Configuration for runtime performance tuning (SME-constrained).

    Optimized for low-resource environments: 512MB-2GB memory,
    <30 second processing time target.
    """

    cache_enabled: bool = Field(
        True,
        description="Enable caching for emission factors and calculations",
    )
    cache_ttl_seconds: int = Field(
        1800,
        ge=60,
        le=86400,
        description="Cache time-to-live in seconds",
    )
    max_concurrent_calcs: int = Field(
        2,
        ge=1,
        le=4,
        description="Maximum concurrent calculation threads (SME: 2 max)",
    )
    timeout_seconds: int = Field(
        30,
        ge=10,
        le=120,
        description="Maximum timeout for a single engine calculation (SME: <30s target)",
    )
    batch_size: int = Field(
        250,
        ge=50,
        le=1000,
        description="Batch size for bulk data processing",
    )
    memory_limit_mb: int = Field(
        1024,
        ge=512,
        le=2048,
        description="Memory limit in MB (SME: 512MB-2GB range)",
    )
    lightweight_mode: bool = Field(
        True,
        description="Enable lightweight processing mode (reduced precision, faster execution)",
    )


class SMEAuditTrailConfig(BaseModel):
    """Configuration for audit trail (SME: 3-year retention)."""

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
        description="Log calculation steps (simplified for SME)",
    )
    assumption_tracking: bool = Field(
        True,
        description="Track assumptions used in calculations",
    )
    data_lineage_enabled: bool = Field(
        False,
        description="Track full data lineage (disabled for SME to reduce complexity)",
    )
    retention_years: int = Field(
        3,
        ge=1,
        le=7,
        description="Audit trail retention period in years (SME: 3 years default)",
    )
    external_audit_export: bool = Field(
        False,
        description="Enable export format for external auditors (off for micro/small SME)",
    )


class SMEVerificationConfig(BaseModel):
    """Configuration for SME emissions verification level."""

    level: VerificationLevel = Field(
        VerificationLevel.SELF_DECLARATION,
        description="Verification level for emissions data",
    )
    third_party_provider: Optional[str] = Field(
        None,
        description="Name of third-party verification provider (if applicable)",
    )
    frequency: str = Field(
        "ANNUAL",
        description="Verification frequency (ANNUAL or BIENNIAL)",
    )
    scope_of_verification: List[str] = Field(
        default_factory=lambda: ["scope_1", "scope_2"],
        description="Which scopes are covered by verification",
    )
    include_scope3_in_verification: bool = Field(
        False,
        description="Include Scope 3 in formal verification (typically not for SME)",
    )


class SMEScorecardConfig(BaseModel):
    """Configuration for SME net-zero maturity scorecard (simplified)."""

    enabled: bool = Field(
        True,
        description="Enable net-zero maturity scorecard",
    )
    assessment_mode: MaturityAssessment = Field(
        MaturityAssessment.QUICK,
        description="Assessment mode: QUICK (5 dimensions), SKIP",
    )
    dimensions: List[str] = Field(
        default_factory=lambda: [
            "baseline_completeness",
            "target_ambition",
            "reduction_progress",
            "data_quality",
            "certification_readiness",
        ],
        description="Scorecard dimensions for SME maturity assessment",
    )
    benchmark_enabled: bool = Field(
        True,
        description="Enable SME peer benchmarking",
    )
    peer_group: str = Field(
        "SME_SIZE",
        description="Peer group for benchmarking: SME_SIZE, SECTOR, REGION",
    )
    kpi_set: List[str] = Field(
        default_factory=lambda: [
            "absolute_emissions_trajectory",
            "emission_intensity_revenue",
            "yoy_reduction_rate",
            "renewable_electricity_pct",
            "data_quality_score",
        ],
        description="KPI set for scorecard",
    )
    target_years: List[int] = Field(
        default_factory=lambda: [2025, 2028, 2030, 2050],
        description="Milestone years for trajectory tracking",
    )


# =============================================================================
# Main Configuration Models
# =============================================================================


class SMENetZeroConfig(BaseModel):
    """Main configuration for PACK-026 SME Net Zero Pack.

    This is the root configuration model that contains all sub-configurations
    for SME net-zero planning. The organization.sme_size field drives
    data quality tier selection and budget constraints.
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
    organization: SMEOrganizationConfig = Field(
        default_factory=SMEOrganizationConfig,
        description="SME organization profile configuration",
    )
    data_quality: SMEDataQualityConfig = Field(
        default_factory=SMEDataQualityConfig,
        description="Data quality tier and estimation configuration",
    )
    boundary: SMEBoundaryConfig = Field(
        default_factory=SMEBoundaryConfig,
        description="GHG Protocol boundary configuration (operational control)",
    )
    scope: SMEScopeConfig = Field(
        default_factory=SMEScopeConfig,
        description="Scope 1/2/3 emissions configuration (simplified for SME)",
    )
    target: SMETargetConfig = Field(
        default_factory=SMETargetConfig,
        description="SBTi SME target setting configuration (ACA, 1.5C)",
    )
    reduction: SMEReductionConfig = Field(
        default_factory=SMEReductionConfig,
        description="Decarbonization actions and roadmap configuration",
    )
    grant: SMEGrantConfig = Field(
        default_factory=SMEGrantConfig,
        description="Grant and funding preference configuration",
    )
    certification: SMECertificationConfig = Field(
        default_factory=SMECertificationConfig,
        description="Certification pathway configuration",
    )
    reporting: SMEReportingConfig = Field(
        default_factory=SMEReportingConfig,
        description="Reporting and template configuration",
    )
    performance: SMEPerformanceConfig = Field(
        default_factory=SMEPerformanceConfig,
        description="Runtime performance tuning configuration",
    )
    audit_trail: SMEAuditTrailConfig = Field(
        default_factory=SMEAuditTrailConfig,
        description="Audit trail and provenance configuration",
    )
    verification: SMEVerificationConfig = Field(
        default_factory=SMEVerificationConfig,
        description="Emissions verification configuration",
    )
    scorecard: SMEScorecardConfig = Field(
        default_factory=SMEScorecardConfig,
        description="Net-zero maturity scorecard configuration",
    )

    @model_validator(mode="after")
    def validate_base_year_before_reporting(self) -> "SMENetZeroConfig":
        """Ensure base year is not after reporting year."""
        if self.base_year > self.reporting_year:
            raise ValueError(
                f"base_year ({self.base_year}) must not be after "
                f"reporting_year ({self.reporting_year})"
            )
        return self

    @model_validator(mode="after")
    def validate_scope3_for_micro(self) -> "SMENetZeroConfig":
        """Set Scope 3 as optional for micro enterprises."""
        if self.organization.sme_size == SMESize.MICRO:
            if self.scope.include_scope3 and not self.scope.scope3_optional:
                logger.warning(
                    "Scope 3 is included but not marked optional for MICRO enterprise. "
                    "Setting scope3_optional=True. Micro enterprises are not expected "
                    "to measure full Scope 3."
                )
                object.__setattr__(self.scope, "scope3_optional", True)
        return self

    @model_validator(mode="after")
    def validate_data_quality_tier_for_size(self) -> "SMENetZeroConfig":
        """Warn if data quality tier seems inappropriate for SME size."""
        if (
            self.organization.sme_size == SMESize.MICRO
            and self.data_quality.tier == SMEDataQualityTier.GOLD
        ):
            logger.warning(
                "GOLD data quality tier selected for MICRO enterprise. "
                "This requires detailed metered data and supplier questionnaires. "
                "Consider BRONZE or SILVER tier for practical implementation."
            )
        return self

    @model_validator(mode="after")
    def validate_budget_for_size(self) -> "SMENetZeroConfig":
        """Warn if budget range does not match SME size."""
        budget_info = SME_BUDGET_RANGES.get(self.organization.sme_size.value, {})
        max_eur = budget_info.get("max_eur", 100_000)
        if (
            self.reduction.budget_constraint_eur is not None
            and self.reduction.budget_constraint_eur > max_eur * 2
        ):
            logger.warning(
                "Budget constraint EUR %.0f exceeds typical range for %s enterprises "
                "(max typical: EUR %.0f). Verify budget or SME size classification.",
                self.reduction.budget_constraint_eur,
                self.organization.sme_size.value,
                max_eur,
            )
        return self

    def get_enabled_engines(self) -> List[str]:
        """Return list of engine names that should be enabled based on config.

        Returns:
            List of engine identifier strings.
        """
        engines = [
            "sme_baseline_inventory",
            "sme_target_setting",
            "sme_gap_analysis",
            "sme_scorecard",
        ]

        # Add reduction engine if actions configured
        if self.reduction.max_actions > 0:
            engines.append("sme_reduction_planner")

        # Add quick wins engine
        if self.reduction.include_quick_wins:
            engines.append("sme_quick_wins")

        # Add grant matcher if enabled
        if self.grant.enabled:
            engines.append("sme_grant_matcher")

        # Add certification engine if pathway selected
        if self.certification.primary_pathway != CertificationPathway.NONE:
            engines.append("sme_certification_assessor")

        return sorted(set(engines))

    def get_sector_info(self) -> Dict[str, Any]:
        """Get SME sector-specific guidance information.

        Returns:
            Dictionary with sector name, typical scope split, key levers,
            and data sources.
        """
        return SME_SECTOR_INFO.get(
            self.organization.sector.value,
            SME_SECTOR_INFO["OTHER"],
        )

    def get_quick_wins(self) -> List[str]:
        """Get recommended quick wins for the configured sector.

        Returns:
            List of quick win action identifiers.
        """
        wins = SME_QUICK_WINS.get(
            self.organization.sector.value,
            SME_QUICK_WINS["OTHER"],
        )
        return wins[: self.reduction.quick_win_max_count]

    def get_scope3_priorities(self) -> List[int]:
        """Get sector-specific Scope 3 category priorities.

        Returns:
            Sorted list of priority Scope 3 category numbers.
        """
        return SME_SCOPE3_PRIORITIES.get(
            self.organization.sector.value,
            SME_SCOPE3_PRIORITIES["OTHER"],
        )


# =============================================================================
# Pack Configuration Wrapper
# =============================================================================


class PackConfig(BaseModel):
    """Top-level pack configuration wrapper for PACK-026 SME Net Zero Pack.

    Handles preset loading, environment variable overrides, and
    configuration merging.

    Example:
        >>> config = PackConfig.from_preset("small_business")
        >>> print(config.pack.organization.sme_size)
        SMESize.SMALL
        >>> config = PackConfig.from_preset("micro_business", overrides={"reporting_year": 2026})
    """

    pack: SMENetZeroConfig = Field(
        default_factory=SMENetZeroConfig,
        description="Main SME Net Zero configuration",
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
        "PACK-026-sme-net-zero",
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
            preset_name: Name of the preset (micro_business, small_business,
                medium_business, service_sme, manufacturing_sme, retail_sme).
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
        env_overrides = _get_env_overrides("SME_NET_ZERO_")
        if env_overrides:
            preset_data = _merge_config(preset_data, env_overrides)

        # Apply explicit overrides
        if overrides:
            preset_data = _merge_config(preset_data, overrides)

        pack_config = SMENetZeroConfig(**preset_data)
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

        pack_config = SMENetZeroConfig(**config_data)
        return cls(pack=pack_config)

    @classmethod
    def from_sme_size(
        cls,
        sme_size: SMESize,
        sector: Optional[SMESector] = None,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> "PackConfig":
        """Auto-select preset based on SME size and optional sector.

        Args:
            sme_size: SME size classification.
            sector: Optional sector for sector-specific preset.
            overrides: Optional configuration overrides.

        Returns:
            PackConfig instance with appropriate preset.
        """
        # Map SME size to default preset
        size_preset_map = {
            SMESize.MICRO: "micro_business",
            SMESize.SMALL: "small_business",
            SMESize.MEDIUM: "medium_business",
        }

        # Override with sector-specific preset if available
        sector_preset_map = {
            SMESector.SERVICES: "service_sme",
            SMESector.TECHNOLOGY: "service_sme",
            SMESector.MANUFACTURING: "manufacturing_sme",
            SMESector.RETAIL: "retail_sme",
        }

        if sector and sector in sector_preset_map:
            preset_name = sector_preset_map[sector]
        else:
            preset_name = size_preset_map.get(sme_size, "small_business")

        return cls.from_preset(preset_name, overrides=overrides)

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

    def get_recommended_certification(self) -> Dict[str, Any]:
        """Get the recommended certification pathway based on current config.

        Returns:
            Dictionary with certification details and readiness indicators.
        """
        size = self.pack.organization.sme_size.value
        pathway = self.pack.certification.primary_pathway.value
        info = CERTIFICATION_INFO.get(pathway, {})

        return {
            "pathway": pathway,
            "info": info,
            "suitable_for_size": size in info.get("suitable_sizes", []),
            "within_budget": (
                self.pack.certification.max_certification_budget_eur
                >= float(info.get("cost_eur_range", "0-0").split("-")[0])
            ),
        }


# =============================================================================
# Utility Functions
# =============================================================================


def load_preset(
    preset_name: str,
    overrides: Optional[Dict[str, Any]] = None,
) -> "PackConfig":
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
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _merge_config(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
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
        SME_NET_ZERO_REPORTING_YEAR=2026
        SME_NET_ZERO_SCOPE__INCLUDE_SCOPE3=true
        SME_NET_ZERO_TARGET__NEAR_TERM_REDUCTION_PCT=50

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


def validate_config(config: SMENetZeroConfig) -> List[str]:
    """Validate an SME net-zero configuration and return any warnings.

    Performs cross-field validation beyond what Pydantic validators cover.
    Returns advisory warnings, not hard errors.

    Args:
        config: SMENetZeroConfig instance to validate.

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

    # Check data quality tier appropriateness
    if (
        config.organization.sme_size == SMESize.MICRO
        and config.data_quality.tier == SMEDataQualityTier.GOLD
    ):
        warnings.append(
            "GOLD data quality tier is ambitious for a MICRO enterprise. "
            "Consider starting with BRONZE tier and improving over time."
        )

    # Check Scope 3 configuration for micro enterprises
    if config.organization.sme_size == SMESize.MICRO and config.scope.include_scope3:
        if len(config.scope.scope3_categories) > 3:
            warnings.append(
                "MICRO enterprises typically only need 1-2 Scope 3 categories. "
                f"Currently configured: {len(config.scope.scope3_categories)} categories."
            )

    # Check budget range matches SME size
    budget_info = SME_BUDGET_RANGES.get(config.organization.sme_size.value, {})
    if config.reduction.budget_range_max_eur > budget_info.get("max_eur", 100_000) * 2:
        warnings.append(
            f"Budget range max EUR {config.reduction.budget_range_max_eur:.0f} "
            f"exceeds typical range for {config.organization.sme_size.value} enterprises."
        )

    # Check near-term target
    if config.target.near_term_reduction_pct < 42.0 and config.target.sbti_sme_route:
        warnings.append(
            f"Near-term target ({config.target.near_term_reduction_pct}%) is below "
            f"SBTi SME minimum of 42%. Acceptable for micro enterprises, "
            f"but review if formal SBTi submission is planned."
        )

    # Check certification pathway suitability
    pathway = config.certification.primary_pathway.value
    cert_info = CERTIFICATION_INFO.get(pathway, {})
    if cert_info:
        suitable_sizes = cert_info.get("suitable_sizes", [])
        if config.organization.sme_size.value not in suitable_sizes:
            warnings.append(
                f"{pathway} certification may not be suitable for "
                f"{config.organization.sme_size.value} enterprises. "
                f"Suitable for: {suitable_sizes}"
            )

    # Check reporting is not over-configured for SME
    if config.reporting.include_cdp_mapping and config.organization.sme_size == SMESize.MICRO:
        warnings.append(
            "CDP mapping is enabled for a MICRO enterprise. "
            "This is unusual and adds complexity. Consider disabling."
        )

    # Check processing constraints
    if config.performance.memory_limit_mb > 2048:
        warnings.append(
            f"Memory limit {config.performance.memory_limit_mb}MB exceeds SME maximum "
            f"of 2048MB. The SME pack is optimized for 512MB-2GB."
        )

    # Check planning horizon
    if config.reduction.planning_horizon_years > 5:
        warnings.append(
            f"Planning horizon of {config.reduction.planning_horizon_years} years "
            f"is long for an SME. Consider 3-year horizon for practical implementation."
        )

    return warnings


def get_sector_info(sector: Union[str, SMESector]) -> Dict[str, Any]:
    """Get detailed information about an SME sector.

    Args:
        sector: Sector enum or string value.

    Returns:
        Dictionary with name, typical scope split, key levers,
        and data sources.
    """
    key = sector.value if isinstance(sector, SMESector) else sector
    return SME_SECTOR_INFO.get(key, SME_SECTOR_INFO["OTHER"])


def get_certification_info(
    pathway: Union[str, CertificationPathway],
) -> Dict[str, Any]:
    """Get detailed information about a certification pathway.

    Args:
        pathway: Certification pathway enum or string value.

    Returns:
        Dictionary with name, barrier level, cost range, requirements,
        and suitable sizes.
    """
    key = pathway.value if isinstance(pathway, CertificationPathway) else pathway
    return CERTIFICATION_INFO.get(key, {})


def get_sme_budget_range(size: Union[str, SMESize]) -> Dict[str, float]:
    """Get typical budget range for an SME size.

    Args:
        size: SME size enum or string value.

    Returns:
        Dictionary with min_eur, max_eur, and typical_eur.
    """
    key = size.value if isinstance(size, SMESize) else size
    return SME_BUDGET_RANGES.get(key, SME_BUDGET_RANGES["SMALL"])


def get_sbti_sme_parameters() -> Dict[str, Any]:
    """Get SBTi SME target route parameters.

    Returns:
        Dictionary with minimum reduction, annual rate, coverage thresholds.
    """
    return SBTI_SME_PARAMETERS.copy()


def get_gwp100(gas: str) -> int:
    """Get IPCC AR6 GWP100 value for a greenhouse gas.

    Args:
        gas: Greenhouse gas identifier (CO2, CH4, N2O, etc.).

    Returns:
        GWP100 value (dimensionless, relative to CO2).
    """
    return IPCC_AR6_GWP100.get(gas.upper(), 0)


def get_default_config(
    sme_size: SMESize = SMESize.SMALL,
    sector: SMESector = SMESector.OTHER,
) -> SMENetZeroConfig:
    """Get default configuration for a given SME size and sector.

    Args:
        sme_size: SME size classification.
        sector: SME sector.

    Returns:
        SMENetZeroConfig instance with appropriate defaults.
    """
    scope3_categories = SME_SCOPE3_PRIORITIES.get(
        sector.value, DEFAULT_SCOPE3_SME_CATEGORIES
    )

    budget_range = SME_BUDGET_RANGES.get(sme_size.value, SME_BUDGET_RANGES["SMALL"])

    # Determine data quality tier by size
    tier_map = {
        SMESize.MICRO: SMEDataQualityTier.BRONZE,
        SMESize.SMALL: SMEDataQualityTier.SILVER,
        SMESize.MEDIUM: SMEDataQualityTier.GOLD,
    }

    # Determine target reduction by size
    target_map = {
        SMESize.MICRO: 30.0,
        SMESize.SMALL: 42.0,
        SMESize.MEDIUM: 50.0,
    }

    return SMENetZeroConfig(
        organization=SMEOrganizationConfig(
            sme_size=sme_size,
            sector=sector,
        ),
        data_quality=SMEDataQualityConfig(
            tier=tier_map.get(sme_size, SMEDataQualityTier.SILVER),
        ),
        scope=SMEScopeConfig(
            scope3_categories=scope3_categories,
        ),
        target=SMETargetConfig(
            near_term_reduction_pct=target_map.get(sme_size, 42.0),
        ),
        reduction=SMEReductionConfig(
            budget_range_min_eur=budget_range.get("min_eur", 0),
            budget_range_max_eur=budget_range.get("max_eur", 20_000),
        ),
    )


def list_available_presets() -> Dict[str, str]:
    """List all available configuration presets.

    Returns:
        Dictionary mapping preset names to descriptions.
    """
    return SUPPORTED_PRESETS.copy()


def get_emissions_profile(sector: Union[str, SMESector]) -> Dict[str, float]:
    """Get typical emissions profile split for an SME sector.

    Args:
        sector: Sector enum or string value.

    Returns:
        Dictionary with scope1_pct, scope2_pct, scope3_pct, profile_type.
    """
    key = sector.value if isinstance(sector, SMESector) else sector
    return SECTOR_EMISSIONS_PROFILE.get(key, SECTOR_EMISSIONS_PROFILE["OTHER"])
