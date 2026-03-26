"""
PACK-049 GHG Multi-Site Management Pack - Configuration Manager

Pydantic v2 configuration for multi-site GHG inventory management including
facility registry, decentralised data collection, organisational boundary
definition, regional emission factor assignment, site-level consolidation,
shared-services allocation, internal benchmarking, completeness tracking,
quality scoring, and multi-site reporting.

Configuration Merge Order (later overrides earlier):
    1. Base pack.yaml manifest
    2. Preset YAML (industry-specific defaults)
    3. Environment overrides (MULTISITE_PACK_* environment variables)
    4. Explicit runtime overrides

Regulatory Context:
    GHG Protocol Corporate Standard (2004, revised 2015) - Chapter 3 & 4
    GHG Protocol Corporate Value Chain (Scope 3) Standard - Chapter 3
    ISO 14064-1:2018 Clause 5 - Organisational boundaries
    EU CSRD (2022/2464) - ESRS E1 multi-site disclosure requirements
    US SEC Climate Disclosure Rules (2024) - Registrant boundary guidance
    PCAF Global GHG Accounting Standard v3 (2024) - Portfolio aggregation
    ISO 50001:2018 - Energy management across multiple sites
    GRI 305 (2016) - Emissions disclosure for multi-site organisations

Author: GreenLang Platform Team
Date: March 2026
Version: 1.0.0
"""
from __future__ import annotations

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
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

logger = logging.getLogger(__name__)

PACK_BASE_DIR = Path(__file__).parent.parent
CONFIG_DIR = Path(__file__).parent


# =============================================================================
# Helper Functions
# =============================================================================


def _utcnow() -> datetime:
    """Return current UTC datetime (mockable for testing)."""
    return datetime.now(timezone.utc)


def _new_uuid() -> str:
    """Return new UUID4 string (mockable for testing)."""
    return str(uuid.uuid4())


def _compute_hash(data: str) -> str:
    """Compute SHA-256 hash of a string for provenance tracking."""
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


# =============================================================================
# Enums (18 total)
# =============================================================================


class FacilityType(str, Enum):
    """Classification of facility type within the organisational portfolio."""
    MANUFACTURING = "MANUFACTURING"
    OFFICE = "OFFICE"
    WAREHOUSE = "WAREHOUSE"
    RETAIL = "RETAIL"
    DATA_CENTER = "DATA_CENTER"
    LABORATORY = "LABORATORY"
    HOSPITAL = "HOSPITAL"
    HOTEL = "HOTEL"
    RESTAURANT = "RESTAURANT"
    SCHOOL = "SCHOOL"
    UNIVERSITY = "UNIVERSITY"
    GOVERNMENT = "GOVERNMENT"
    MILITARY = "MILITARY"
    AIRPORT = "AIRPORT"
    PORT = "PORT"
    MINE = "MINE"
    REFINERY = "REFINERY"
    POWER_PLANT = "POWER_PLANT"
    FARM = "FARM"
    MIXED_USE = "MIXED_USE"
    DISTRIBUTION_CENTER = "DISTRIBUTION_CENTER"
    OTHER = "OTHER"


class FacilityLifecycle(str, Enum):
    """Lifecycle stage of a facility within the portfolio."""
    PLANNED = "PLANNED"
    COMMISSIONING = "COMMISSIONING"
    OPERATIONAL = "OPERATIONAL"
    UNDER_RENOVATION = "UNDER_RENOVATION"
    TEMPORARILY_CLOSED = "TEMPORARILY_CLOSED"
    DECOMMISSIONING = "DECOMMISSIONING"
    DECOMMISSIONED = "DECOMMISSIONED"


class ConsolidationApproach(str, Enum):
    """GHG Protocol organisational boundary consolidation approach."""
    EQUITY_SHARE = "EQUITY_SHARE"
    OPERATIONAL_CONTROL = "OPERATIONAL_CONTROL"
    FINANCIAL_CONTROL = "FINANCIAL_CONTROL"


class OwnershipType(str, Enum):
    """Ownership or control relationship for a facility."""
    WHOLLY_OWNED = "WHOLLY_OWNED"
    MAJORITY_OWNED = "MAJORITY_OWNED"
    JOINT_VENTURE = "JOINT_VENTURE"
    ASSOCIATE = "ASSOCIATE"
    FRANCHISE = "FRANCHISE"
    LEASED_FINANCE = "LEASED_FINANCE"
    LEASED_OPERATING = "LEASED_OPERATING"
    MINORITY_INTEREST = "MINORITY_INTEREST"


class CollectionPeriodType(str, Enum):
    """Frequency of data collection from facilities."""
    MONTHLY = "MONTHLY"
    QUARTERLY = "QUARTERLY"
    SEMI_ANNUAL = "SEMI_ANNUAL"
    ANNUAL = "ANNUAL"


class SubmissionStatus(str, Enum):
    """Status of a facility's data submission for a reporting period."""
    NOT_STARTED = "NOT_STARTED"
    DRAFT = "DRAFT"
    SUBMITTED = "SUBMITTED"
    UNDER_REVIEW = "UNDER_REVIEW"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    RESUBMITTED = "RESUBMITTED"
    OVERDUE = "OVERDUE"


class DataEntryMode(str, Enum):
    """Method of data entry for facility-level emission data."""
    MANUAL = "MANUAL"
    SPREADSHEET_UPLOAD = "SPREADSHEET_UPLOAD"
    API_PUSH = "API_PUSH"
    ERP_CONNECTOR = "ERP_CONNECTOR"
    IOT_FEED = "IOT_FEED"


class AllocationMethod(str, Enum):
    """Method for allocating shared emissions across tenants or units."""
    FLOOR_AREA = "FLOOR_AREA"
    HEADCOUNT = "HEADCOUNT"
    REVENUE = "REVENUE"
    PRODUCTION_OUTPUT = "PRODUCTION_OUTPUT"
    ENERGY_CONSUMPTION = "ENERGY_CONSUMPTION"
    OPERATING_HOURS = "OPERATING_HOURS"
    CUSTOM_FORMULA = "CUSTOM_FORMULA"


class LandlordTenantSplit(str, Enum):
    """Approach for splitting emissions between landlord and tenant."""
    WHOLE_BUILDING = "WHOLE_BUILDING"
    TENANT_ONLY = "TENANT_ONLY"
    COMMON_AREA_PROPORTIONAL = "COMMON_AREA_PROPORTIONAL"
    SUB_METERED = "SUB_METERED"


class CogenerationType(str, Enum):
    """Method for allocating cogeneration (CHP) emissions."""
    EFFICIENCY_METHOD = "EFFICIENCY_METHOD"
    ENERGY_CONTENT_METHOD = "ENERGY_CONTENT_METHOD"
    RESIDUAL_METHOD = "RESIDUAL_METHOD"


class FactorTier(str, Enum):
    """Emission factor data quality tier (higher = more site-specific)."""
    TIER_3_FACILITY = "TIER_3_FACILITY"
    TIER_2_NATIONAL = "TIER_2_NATIONAL"
    TIER_1_REGIONAL = "TIER_1_REGIONAL"
    TIER_0_IPCC_DEFAULT = "TIER_0_IPCC_DEFAULT"


class FactorSource(str, Enum):
    """Source database or publication for emission factors."""
    IPCC_2019 = "IPCC_2019"
    DEFRA = "DEFRA"
    EPA_EGRID = "EPA_EGRID"
    IEA = "IEA"
    UBA = "UBA"
    ADEME = "ADEME"
    ISPRA = "ISPRA"
    ECOINVENT = "ECOINVENT"
    SUPPLIER_SPECIFIC = "SUPPLIER_SPECIFIC"
    CUSTOM = "CUSTOM"


class QualityDimension(str, Enum):
    """Dimension of data quality assessment for site-level data."""
    ACCURACY = "ACCURACY"
    COMPLETENESS = "COMPLETENESS"
    CONSISTENCY = "CONSISTENCY"
    TIMELINESS = "TIMELINESS"
    METHODOLOGY = "METHODOLOGY"
    DOCUMENTATION = "DOCUMENTATION"


class QualityScore(str, Enum):
    """Quality score rating for site-level data (1=best, 5=worst)."""
    SCORE_1_VERIFIED = "SCORE_1_VERIFIED"
    SCORE_2_CALCULATED = "SCORE_2_CALCULATED"
    SCORE_3_ESTIMATED = "SCORE_3_ESTIMATED"
    SCORE_4_EXTRAPOLATED = "SCORE_4_EXTRAPOLATED"
    SCORE_5_PROXY = "SCORE_5_PROXY"


class ComparisonKPI(str, Enum):
    """Key performance indicator for cross-site benchmarking."""
    EMISSIONS_PER_M2 = "EMISSIONS_PER_M2"
    EMISSIONS_PER_FTE = "EMISSIONS_PER_FTE"
    EMISSIONS_PER_UNIT = "EMISSIONS_PER_UNIT"
    EMISSIONS_PER_REVENUE = "EMISSIONS_PER_REVENUE"
    ENERGY_PER_M2 = "ENERGY_PER_M2"
    ENERGY_PER_FTE = "ENERGY_PER_FTE"
    WASTE_PER_FTE = "WASTE_PER_FTE"
    WATER_PER_M2 = "WATER_PER_M2"


class ReportType(str, Enum):
    """Type of multi-site report output."""
    PORTFOLIO_DASHBOARD = "PORTFOLIO_DASHBOARD"
    SITE_DETAIL = "SITE_DETAIL"
    CONSOLIDATION = "CONSOLIDATION"
    BOUNDARY_DEFINITION = "BOUNDARY_DEFINITION"
    FACTOR_ASSIGNMENT = "FACTOR_ASSIGNMENT"
    ALLOCATION = "ALLOCATION"
    COMPARISON = "COMPARISON"
    COLLECTION_STATUS = "COLLECTION_STATUS"
    QUALITY_HEATMAP = "QUALITY_HEATMAP"
    TREND = "TREND"


class ExportFormat(str, Enum):
    """Supported report output formats."""
    MARKDOWN = "MARKDOWN"
    HTML = "HTML"
    JSON = "JSON"
    CSV = "CSV"
    XBRL = "XBRL"


class AlertType(str, Enum):
    """Type of multi-site management alert."""
    DEADLINE_APPROACHING = "DEADLINE_APPROACHING"
    SUBMISSION_OVERDUE = "SUBMISSION_OVERDUE"
    QUALITY_BELOW_THRESHOLD = "QUALITY_BELOW_THRESHOLD"
    BOUNDARY_CHANGE = "BOUNDARY_CHANGE"
    ALLOCATION_VARIANCE = "ALLOCATION_VARIANCE"
    COMPLETENESS_GAP = "COMPLETENESS_GAP"


# =============================================================================
# Reference Data Constants
# =============================================================================


DEFAULT_FACILITY_TYPES: Dict[str, Dict[str, Any]] = {
    "MANUFACTURING": {
        "avg_floor_area_m2": 10000,
        "avg_headcount": 200,
        "emission_profile": "energy_intensive",
        "typical_scopes": ["SCOPE_1", "SCOPE_2"],
        "description": "Industrial manufacturing and production facilities",
    },
    "OFFICE": {
        "avg_floor_area_m2": 3000,
        "avg_headcount": 150,
        "emission_profile": "low_intensity",
        "typical_scopes": ["SCOPE_2"],
        "description": "Corporate offices and administrative buildings",
    },
    "WAREHOUSE": {
        "avg_floor_area_m2": 8000,
        "avg_headcount": 50,
        "emission_profile": "moderate_intensity",
        "typical_scopes": ["SCOPE_1", "SCOPE_2"],
        "description": "Storage and distribution warehouses",
    },
    "RETAIL": {
        "avg_floor_area_m2": 1500,
        "avg_headcount": 25,
        "emission_profile": "low_intensity",
        "typical_scopes": ["SCOPE_2"],
        "description": "Retail stores and customer-facing outlets",
    },
    "DATA_CENTER": {
        "avg_floor_area_m2": 5000,
        "avg_headcount": 30,
        "emission_profile": "energy_intensive",
        "typical_scopes": ["SCOPE_2"],
        "description": "Data centres and server farms",
    },
    "LABORATORY": {
        "avg_floor_area_m2": 2000,
        "avg_headcount": 60,
        "emission_profile": "moderate_intensity",
        "typical_scopes": ["SCOPE_1", "SCOPE_2"],
        "description": "Research and testing laboratories",
    },
    "HOSPITAL": {
        "avg_floor_area_m2": 20000,
        "avg_headcount": 500,
        "emission_profile": "energy_intensive",
        "typical_scopes": ["SCOPE_1", "SCOPE_2"],
        "description": "Hospitals and healthcare facilities",
    },
    "HOTEL": {
        "avg_floor_area_m2": 6000,
        "avg_headcount": 80,
        "emission_profile": "moderate_intensity",
        "typical_scopes": ["SCOPE_1", "SCOPE_2"],
        "description": "Hotels and accommodation properties",
    },
    "RESTAURANT": {
        "avg_floor_area_m2": 300,
        "avg_headcount": 20,
        "emission_profile": "moderate_intensity",
        "typical_scopes": ["SCOPE_1", "SCOPE_2"],
        "description": "Restaurants and food service outlets",
    },
    "SCHOOL": {
        "avg_floor_area_m2": 4000,
        "avg_headcount": 100,
        "emission_profile": "low_intensity",
        "typical_scopes": ["SCOPE_1", "SCOPE_2"],
        "description": "Primary and secondary educational institutions",
    },
    "UNIVERSITY": {
        "avg_floor_area_m2": 50000,
        "avg_headcount": 2000,
        "emission_profile": "moderate_intensity",
        "typical_scopes": ["SCOPE_1", "SCOPE_2"],
        "description": "Universities and higher education campuses",
    },
    "GOVERNMENT": {
        "avg_floor_area_m2": 5000,
        "avg_headcount": 200,
        "emission_profile": "low_intensity",
        "typical_scopes": ["SCOPE_2"],
        "description": "Government and public administration buildings",
    },
    "AIRPORT": {
        "avg_floor_area_m2": 100000,
        "avg_headcount": 1000,
        "emission_profile": "energy_intensive",
        "typical_scopes": ["SCOPE_1", "SCOPE_2"],
        "description": "Airport terminal and ground operations",
    },
    "PORT": {
        "avg_floor_area_m2": 50000,
        "avg_headcount": 300,
        "emission_profile": "energy_intensive",
        "typical_scopes": ["SCOPE_1", "SCOPE_2"],
        "description": "Maritime and inland port facilities",
    },
    "MINE": {
        "avg_floor_area_m2": 25000,
        "avg_headcount": 400,
        "emission_profile": "energy_intensive",
        "typical_scopes": ["SCOPE_1", "SCOPE_2"],
        "description": "Mining and extraction sites",
    },
    "REFINERY": {
        "avg_floor_area_m2": 30000,
        "avg_headcount": 250,
        "emission_profile": "energy_intensive",
        "typical_scopes": ["SCOPE_1", "SCOPE_2"],
        "description": "Oil, gas, and chemical refineries",
    },
    "POWER_PLANT": {
        "avg_floor_area_m2": 15000,
        "avg_headcount": 100,
        "emission_profile": "energy_intensive",
        "typical_scopes": ["SCOPE_1"],
        "description": "Electricity generation stations",
    },
    "FARM": {
        "avg_floor_area_m2": 5000,
        "avg_headcount": 30,
        "emission_profile": "moderate_intensity",
        "typical_scopes": ["SCOPE_1"],
        "description": "Agricultural and livestock operations",
    },
    "DISTRIBUTION_CENTER": {
        "avg_floor_area_m2": 12000,
        "avg_headcount": 100,
        "emission_profile": "moderate_intensity",
        "typical_scopes": ["SCOPE_1", "SCOPE_2"],
        "description": "Distribution and logistics hubs",
    },
    "MIXED_USE": {
        "avg_floor_area_m2": 8000,
        "avg_headcount": 150,
        "emission_profile": "moderate_intensity",
        "typical_scopes": ["SCOPE_1", "SCOPE_2"],
        "description": "Multi-purpose facilities combining office, retail, or industrial",
    },
    "OTHER": {
        "avg_floor_area_m2": 3000,
        "avg_headcount": 50,
        "emission_profile": "low_intensity",
        "typical_scopes": ["SCOPE_1", "SCOPE_2"],
        "description": "Facilities not covered by other categories",
    },
}


DEFAULT_ALLOCATION_PRIORITIES: List[str] = [
    AllocationMethod.FLOOR_AREA.value,
    AllocationMethod.HEADCOUNT.value,
    AllocationMethod.ENERGY_CONSUMPTION.value,
    AllocationMethod.REVENUE.value,
    AllocationMethod.PRODUCTION_OUTPUT.value,
    AllocationMethod.OPERATING_HOURS.value,
    AllocationMethod.CUSTOM_FORMULA.value,
]


DEFAULT_QUALITY_WEIGHTS: Dict[str, Decimal] = {
    QualityDimension.ACCURACY.value: Decimal("0.25"),
    QualityDimension.COMPLETENESS.value: Decimal("0.25"),
    QualityDimension.CONSISTENCY.value: Decimal("0.15"),
    QualityDimension.TIMELINESS.value: Decimal("0.15"),
    QualityDimension.METHODOLOGY.value: Decimal("0.10"),
    QualityDimension.DOCUMENTATION.value: Decimal("0.10"),
}


DEFAULT_DEADLINE_REMINDERS_DAYS: List[int] = [14, 7, 3, 1]

DEFAULT_MATERIALITY_THRESHOLD: Decimal = Decimal("0.05")

DEFAULT_DE_MINIMIS_THRESHOLD: Decimal = Decimal("0.01")

DEFAULT_COMPLETENESS_TARGET: Decimal = Decimal("0.95")


CONSOLIDATION_APPROACH_GUIDANCE: Dict[str, Dict[str, Any]] = {
    "EQUITY_SHARE": {
        "name": "Equity Share",
        "description": (
            "Account for GHG emissions from operations according to the "
            "company's share of equity in the operation. Reflects economic "
            "interest regardless of operational involvement."
        ),
        "applicable_ownership": [
            OwnershipType.WHOLLY_OWNED.value,
            OwnershipType.MAJORITY_OWNED.value,
            OwnershipType.JOINT_VENTURE.value,
            OwnershipType.ASSOCIATE.value,
            OwnershipType.MINORITY_INTEREST.value,
        ],
        "standard_reference": "GHG Protocol Corporate Standard, Chapter 3",
    },
    "OPERATIONAL_CONTROL": {
        "name": "Operational Control",
        "description": (
            "Account for 100% of GHG emissions from operations over which "
            "the company has operational control (i.e. full authority to "
            "introduce and implement operating policies)."
        ),
        "applicable_ownership": [
            OwnershipType.WHOLLY_OWNED.value,
            OwnershipType.MAJORITY_OWNED.value,
            OwnershipType.LEASED_FINANCE.value,
            OwnershipType.FRANCHISE.value,
        ],
        "standard_reference": "GHG Protocol Corporate Standard, Chapter 3",
    },
    "FINANCIAL_CONTROL": {
        "name": "Financial Control",
        "description": (
            "Account for 100% of GHG emissions from operations over which "
            "the company has financial control (i.e. ability to direct "
            "financial and operating policies to gain economic benefits)."
        ),
        "applicable_ownership": [
            OwnershipType.WHOLLY_OWNED.value,
            OwnershipType.MAJORITY_OWNED.value,
            OwnershipType.LEASED_FINANCE.value,
        ],
        "standard_reference": "GHG Protocol Corporate Standard, Chapter 3",
    },
}


REGIONAL_FACTOR_DATABASES: Dict[str, Dict[str, Any]] = {
    "IPCC_2019": {
        "name": "IPCC 2006/2019 Refinement",
        "tier": FactorTier.TIER_0_IPCC_DEFAULT.value,
        "coverage": "Global",
        "update_frequency": "Irregular (assessment reports)",
        "url": "https://www.ipcc-nggip.iges.or.jp/",
    },
    "DEFRA": {
        "name": "UK DEFRA/DESNZ Conversion Factors",
        "tier": FactorTier.TIER_2_NATIONAL.value,
        "coverage": "United Kingdom",
        "update_frequency": "Annual (June/July)",
        "url": "https://www.gov.uk/government/collections/government-conversion-factors-for-company-reporting",
    },
    "EPA_EGRID": {
        "name": "US EPA eGRID",
        "tier": FactorTier.TIER_1_REGIONAL.value,
        "coverage": "United States (subregional)",
        "update_frequency": "Annual (January)",
        "url": "https://www.epa.gov/egrid",
    },
    "IEA": {
        "name": "IEA Emission Factors",
        "tier": FactorTier.TIER_2_NATIONAL.value,
        "coverage": "Global (country-level)",
        "update_frequency": "Annual",
        "url": "https://www.iea.org/data-and-statistics",
    },
    "UBA": {
        "name": "German UBA Emission Factors",
        "tier": FactorTier.TIER_2_NATIONAL.value,
        "coverage": "Germany",
        "update_frequency": "Annual",
        "url": "https://www.umweltbundesamt.de/",
    },
    "ADEME": {
        "name": "French ADEME Base Carbone",
        "tier": FactorTier.TIER_2_NATIONAL.value,
        "coverage": "France",
        "update_frequency": "Continuous updates",
        "url": "https://base-empreinte.ademe.fr/",
    },
    "ISPRA": {
        "name": "Italian ISPRA Emission Factors",
        "tier": FactorTier.TIER_2_NATIONAL.value,
        "coverage": "Italy",
        "update_frequency": "Annual",
        "url": "https://www.isprambiente.gov.it/",
    },
    "ECOINVENT": {
        "name": "ecoinvent LCA Database",
        "tier": FactorTier.TIER_1_REGIONAL.value,
        "coverage": "Global (process-level)",
        "update_frequency": "Biennial",
        "url": "https://ecoinvent.org/",
    },
}


AVAILABLE_PRESETS: Dict[str, str] = {
    "corporate_general": (
        "General multi-site corporate portfolio with operational control, "
        "monthly data collection, and standard consolidation for CSRD compliance"
    ),
    "manufacturing": (
        "Manufacturing portfolio with energy-intensive facilities, production "
        "output allocation, process emission tracking, and ISO 50001 alignment"
    ),
    "retail_chain": (
        "Retail chain with many small outlets, landlord-tenant splits, "
        "floor-area allocation, and streamlined data collection templates"
    ),
    "real_estate": (
        "Real estate portfolio with CRREM alignment, landlord-tenant metering, "
        "common area allocation, and building energy benchmarking"
    ),
    "financial_services": (
        "Financial services multi-site with office-heavy portfolio, financed "
        "emissions overlay, PCAF quality scoring, and headcount allocation"
    ),
    "logistics": (
        "Logistics and transport hub portfolio with warehouse and distribution "
        "centres, fleet allocation, and hub-level emission tracking"
    ),
    "healthcare": (
        "Healthcare portfolio with hospitals, clinics, and laboratories, "
        "24/7 operational profiles, medical gas tracking, and energy intensity"
    ),
    "public_sector": (
        "Public sector multi-site with government buildings, schools, and "
        "community facilities, budget-aligned reporting, and public disclosure"
    ),
}


# =============================================================================
# Sub-Config Models (15 Pydantic v2 models)
# =============================================================================


class SiteRegistryConfig(BaseModel):
    """Configuration for the site registry and facility management."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    max_sites: int = Field(
        500, ge=1, le=50000,
        description="Maximum number of sites in the portfolio",
    )
    facility_types_enabled: List[FacilityType] = Field(
        default_factory=lambda: [
            FacilityType.MANUFACTURING,
            FacilityType.OFFICE,
            FacilityType.WAREHOUSE,
            FacilityType.RETAIL,
            FacilityType.DATA_CENTER,
            FacilityType.OTHER,
        ],
        description="Facility types enabled for registration",
    )
    lifecycle_tracking: bool = Field(
        True,
        description="Track facility lifecycle stages (planned through decommissioned)",
    )
    grouping_enabled: bool = Field(
        True,
        description="Enable hierarchical grouping of sites (region, business unit, etc.)",
    )
    require_geo_coordinates: bool = Field(
        False,
        description="Require latitude/longitude for each facility",
    )
    custom_attributes_enabled: bool = Field(
        True,
        description="Allow custom metadata attributes on facility records",
    )
    auto_assign_region: bool = Field(
        True,
        description="Auto-assign region based on country code for factor selection",
    )

    @field_validator("max_sites")
    @classmethod
    def validate_max_sites(cls, v: int) -> int:
        """Validate max sites is within reasonable bounds."""
        if v < 1:
            raise ValueError("max_sites must be at least 1")
        return v


class DataCollectionConfig(BaseModel):
    """Configuration for decentralised data collection from facilities."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    collection_period: CollectionPeriodType = Field(
        CollectionPeriodType.MONTHLY,
        description="Frequency of data collection from facilities",
    )
    data_entry_modes: List[DataEntryMode] = Field(
        default_factory=lambda: [
            DataEntryMode.MANUAL,
            DataEntryMode.SPREADSHEET_UPLOAD,
            DataEntryMode.API_PUSH,
        ],
        description="Allowed data entry methods",
    )
    validation_strictness: str = Field(
        "STANDARD",
        description="Level of input validation (RELAXED, STANDARD, STRICT)",
    )
    estimation_allowed: bool = Field(
        True,
        description="Allow estimation when actual data is unavailable",
    )
    reminder_days: List[int] = Field(
        default_factory=lambda: list(DEFAULT_DEADLINE_REMINDERS_DAYS),
        description="Days before deadline to send collection reminders",
    )
    auto_escalation: bool = Field(
        True,
        description="Auto-escalate overdue submissions to site managers",
    )
    submission_lock_after_approval: bool = Field(
        True,
        description="Lock submissions after approval to prevent tampering",
    )
    require_evidence_attachments: bool = Field(
        False,
        description="Require supporting documents (invoices, meter reads) on submission",
    )

    @field_validator("validation_strictness")
    @classmethod
    def validate_strictness(cls, v: str) -> str:
        """Validate strictness level."""
        allowed = {"RELAXED", "STANDARD", "STRICT"}
        if v.upper() not in allowed:
            raise ValueError(f"validation_strictness must be one of {allowed}, got '{v}'")
        return v.upper()


class BoundaryConfig(BaseModel):
    """Configuration for organisational boundary definition."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    consolidation_approach: ConsolidationApproach = Field(
        ConsolidationApproach.OPERATIONAL_CONTROL,
        description="GHG Protocol consolidation approach for the organisation",
    )
    materiality_threshold: Decimal = Field(
        DEFAULT_MATERIALITY_THRESHOLD,
        ge=Decimal("0.001"), le=Decimal("0.20"),
        description="Materiality threshold for boundary inclusion (0.05 = 5%)",
    )
    de_minimis_threshold: Decimal = Field(
        DEFAULT_DE_MINIMIS_THRESHOLD,
        ge=Decimal("0.001"), le=Decimal("0.10"),
        description="De minimis threshold below which sources may be excluded (0.01 = 1%)",
    )
    annual_boundary_lock: bool = Field(
        True,
        description="Lock boundary at start of reporting year to prevent mid-year changes",
    )
    track_boundary_changes: bool = Field(
        True,
        description="Track and log all boundary changes with justification",
    )
    require_boundary_approval: bool = Field(
        True,
        description="Require formal approval for boundary changes",
    )
    default_equity_share_pct: Decimal = Field(
        Decimal("100.0"),
        ge=Decimal("0"), le=Decimal("100"),
        description="Default equity share for wholly-owned subsidiaries",
    )


class RegionalFactorConfig(BaseModel):
    """Configuration for regional emission factor assignment."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    preferred_tier: FactorTier = Field(
        FactorTier.TIER_2_NATIONAL,
        description="Preferred tier for emission factor selection (fallback to lower tiers)",
    )
    default_source: FactorSource = Field(
        FactorSource.DEFRA,
        description="Default emission factor source when no region-specific source exists",
    )
    grid_factor_update_frequency: str = Field(
        "ANNUAL",
        description="How often to update grid electricity emission factors (ANNUAL, QUARTERLY)",
    )
    climate_zone_enabled: bool = Field(
        False,
        description="Use climate zone for heating/cooling factor adjustments",
    )
    allow_supplier_specific: bool = Field(
        True,
        description="Allow supplier-specific emission factors (Tier 3)",
    )
    factor_lock_with_boundary: bool = Field(
        True,
        description="Lock emission factors when boundary is locked for the year",
    )
    fallback_chain: List[FactorTier] = Field(
        default_factory=lambda: [
            FactorTier.TIER_3_FACILITY,
            FactorTier.TIER_2_NATIONAL,
            FactorTier.TIER_1_REGIONAL,
            FactorTier.TIER_0_IPCC_DEFAULT,
        ],
        description="Tier fallback chain when preferred tier is unavailable",
    )

    @field_validator("grid_factor_update_frequency")
    @classmethod
    def validate_grid_frequency(cls, v: str) -> str:
        """Validate grid factor update frequency."""
        allowed = {"ANNUAL", "QUARTERLY", "MONTHLY"}
        if v.upper() not in allowed:
            raise ValueError(
                f"grid_factor_update_frequency must be one of {allowed}, got '{v}'"
            )
        return v.upper()


class ConsolidationConfig(BaseModel):
    """Configuration for multi-site inventory consolidation."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    elimination_enabled: bool = Field(
        True,
        description="Enable inter-company emission elimination during consolidation",
    )
    equity_adjustment_enabled: bool = Field(
        True,
        description="Apply equity share adjustments during consolidation",
    )
    completeness_threshold: Decimal = Field(
        DEFAULT_COMPLETENESS_TARGET,
        ge=Decimal("0.50"), le=Decimal("1.00"),
        description="Minimum portfolio completeness to proceed with consolidation (0.95 = 95%)",
    )
    reconciliation_tolerance: Decimal = Field(
        Decimal("0.01"),
        ge=Decimal("0.001"), le=Decimal("0.10"),
        description="Tolerance for consolidation reconciliation (0.01 = 1%)",
    )
    auto_estimate_missing: bool = Field(
        True,
        description="Auto-estimate missing site data based on historical patterns",
    )
    estimation_method: str = Field(
        "PRIOR_YEAR_ADJUSTED",
        description="Method for estimating missing data (PRIOR_YEAR_ADJUSTED, PEER_AVERAGE, ZERO)",
    )
    require_sign_off: bool = Field(
        True,
        description="Require management sign-off on consolidated totals",
    )

    @field_validator("estimation_method")
    @classmethod
    def validate_estimation_method(cls, v: str) -> str:
        """Validate estimation method."""
        allowed = {"PRIOR_YEAR_ADJUSTED", "PEER_AVERAGE", "ZERO"}
        if v.upper() not in allowed:
            raise ValueError(f"estimation_method must be one of {allowed}, got '{v}'")
        return v.upper()


class AllocationConfig(BaseModel):
    """Configuration for shared-services and tenant emission allocation."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    default_method: AllocationMethod = Field(
        AllocationMethod.FLOOR_AREA,
        description="Default allocation method for shared emissions",
    )
    shared_services_enabled: bool = Field(
        True,
        description="Enable allocation of shared-services emissions to business units",
    )
    landlord_tenant_enabled: bool = Field(
        False,
        description="Enable landlord-tenant emission splits",
    )
    cogeneration_enabled: bool = Field(
        False,
        description="Enable cogeneration (CHP) emission allocation",
    )
    cogeneration_method: CogenerationType = Field(
        CogenerationType.EFFICIENCY_METHOD,
        description="Cogeneration allocation method when enabled",
    )
    landlord_tenant_approach: LandlordTenantSplit = Field(
        LandlordTenantSplit.COMMON_AREA_PROPORTIONAL,
        description="Default landlord-tenant split approach",
    )
    allocation_review_frequency: str = Field(
        "ANNUAL",
        description="How often to review allocation keys (ANNUAL, QUARTERLY)",
    )
    variance_alert_threshold_pct: Decimal = Field(
        Decimal("10.0"),
        ge=Decimal("1.0"), le=Decimal("50.0"),
        description="Alert threshold for allocation key variance vs prior period (%)",
    )

    @field_validator("allocation_review_frequency")
    @classmethod
    def validate_review_frequency(cls, v: str) -> str:
        """Validate allocation review frequency."""
        allowed = {"ANNUAL", "QUARTERLY", "MONTHLY"}
        if v.upper() not in allowed:
            raise ValueError(
                f"allocation_review_frequency must be one of {allowed}, got '{v}'"
            )
        return v.upper()


class ComparisonConfig(BaseModel):
    """Configuration for cross-site benchmarking and comparison."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    default_kpi: ComparisonKPI = Field(
        ComparisonKPI.EMISSIONS_PER_M2,
        description="Default KPI for cross-site comparison",
    )
    peer_group_min_size: int = Field(
        3, ge=2, le=50,
        description="Minimum number of facilities in a peer group for valid comparison",
    )
    percentile_bands: List[int] = Field(
        default_factory=lambda: [10, 25, 50, 75, 90],
        description="Percentile bands for performance distribution",
    )
    trend_years: int = Field(
        3, ge=1, le=10,
        description="Number of years to include in trend analysis",
    )
    normalisation_enabled: bool = Field(
        True,
        description="Enable normalisation of KPIs for like-for-like comparison",
    )
    climate_adjustment_enabled: bool = Field(
        False,
        description="Apply heating/cooling degree-day adjustments for fair comparison",
    )
    outlier_detection_enabled: bool = Field(
        True,
        description="Flag statistical outliers in cross-site comparisons",
    )
    outlier_std_dev_threshold: Decimal = Field(
        Decimal("2.0"),
        ge=Decimal("1.0"), le=Decimal("5.0"),
        description="Standard deviation threshold for outlier detection",
    )


class CompletionConfig(BaseModel):
    """Configuration for portfolio completeness tracking."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    completeness_target: Decimal = Field(
        DEFAULT_COMPLETENESS_TARGET,
        ge=Decimal("0.50"), le=Decimal("1.00"),
        description="Target portfolio completeness (0.95 = 95%)",
    )
    deadline_reminders_days: List[int] = Field(
        default_factory=lambda: list(DEFAULT_DEADLINE_REMINDERS_DAYS),
        description="Days before deadline to send reminders",
    )
    escalation_enabled: bool = Field(
        True,
        description="Enable escalation for overdue facility submissions",
    )
    auto_estimation_enabled: bool = Field(
        True,
        description="Auto-estimate data for non-responding sites at period end",
    )
    track_submission_history: bool = Field(
        True,
        description="Track historical submission rates per facility",
    )
    gap_report_frequency: str = Field(
        "WEEKLY",
        description="How often to generate completeness gap reports (DAILY, WEEKLY, MONTHLY)",
    )

    @field_validator("gap_report_frequency")
    @classmethod
    def validate_gap_frequency(cls, v: str) -> str:
        """Validate gap report frequency."""
        allowed = {"DAILY", "WEEKLY", "MONTHLY"}
        if v.upper() not in allowed:
            raise ValueError(f"gap_report_frequency must be one of {allowed}, got '{v}'")
        return v.upper()


class QualityConfig(BaseModel):
    """Configuration for multi-dimension data quality scoring."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    quality_weights: Dict[str, Decimal] = Field(
        default_factory=lambda: dict(DEFAULT_QUALITY_WEIGHTS),
        description="Weights for each quality dimension (must sum to 1.0)",
    )
    minimum_quality_score: int = Field(
        3, ge=1, le=5,
        description="Minimum acceptable quality score (1=best, 5=worst)",
    )
    improvement_tracking_enabled: bool = Field(
        True,
        description="Track quality score improvements over time per facility",
    )
    auto_downgrade_on_estimation: bool = Field(
        True,
        description="Automatically downgrade quality score when data is estimated",
    )
    quality_heatmap_enabled: bool = Field(
        True,
        description="Generate portfolio quality heatmap in reports",
    )
    target_verified_pct: Decimal = Field(
        Decimal("0.80"),
        ge=Decimal("0.0"), le=Decimal("1.0"),
        description="Target percentage of sites with SCORE_1_VERIFIED quality",
    )

    @model_validator(mode="after")
    def validate_weights_sum(self) -> QualityConfig:
        """Ensure quality weights sum to approximately 1.0."""
        total = sum(self.quality_weights.values())
        if abs(total - Decimal("1.0")) > Decimal("0.01"):
            raise ValueError(
                f"quality_weights must sum to 1.0, got {total}"
            )
        return self


class ReportingConfig(BaseModel):
    """Configuration for multi-site report generation."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    default_format: ExportFormat = Field(
        ExportFormat.HTML,
        description="Default output format for generated reports",
    )
    drill_down_levels: int = Field(
        3, ge=1, le=5,
        description="Number of drill-down levels (e.g. Group > Region > Site)",
    )
    multi_year_enabled: bool = Field(
        True,
        description="Include multi-year trend data in reports",
    )
    max_report_sites: int = Field(
        500, ge=10, le=50000,
        description="Maximum number of sites to include in a single report",
    )
    include_charts: bool = Field(True, description="Include charts and visualisations")
    include_data_tables: bool = Field(True, description="Include detailed data tables")
    include_appendices: bool = Field(True, description="Include technical appendices")
    language: str = Field("en", description="Report language (ISO 639-1)")
    decimal_places_display: int = Field(
        2, ge=0, le=6,
        description="Decimal places for display in reports",
    )
    branding: Dict[str, str] = Field(
        default_factory=lambda: {
            "logo_url": "",
            "primary_colour": "#1B5E20",
            "company_name": "",
        },
        description="Report branding configuration",
    )


class SecurityConfig(BaseModel):
    """Configuration for access control and data protection."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    permissions: List[str] = Field(
        default_factory=lambda: [
            "site_admin", "site_manager", "data_collector",
            "reviewer", "approver", "consolidation_admin",
            "report_viewer", "portfolio_admin", "viewer", "admin",
        ],
        description="Available RBAC roles for multi-site management",
    )
    rls_enabled: bool = Field(
        True,
        description="Enable row-level security for site-specific data access",
    )
    audit_enabled: bool = Field(
        True,
        description="Enable audit trail for all multi-site operations",
    )
    site_level_access_control: bool = Field(
        True,
        description="Restrict users to their assigned sites only",
    )


class PerformanceConfig(BaseModel):
    """Configuration for computational performance tuning."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    max_concurrent_sites: int = Field(
        50, ge=5, le=500,
        description="Maximum number of sites to process concurrently",
    )
    batch_size: int = Field(
        100, ge=10, le=5000,
        description="Batch size for bulk site data processing",
    )
    cache_ttl_seconds: int = Field(
        3600, ge=60, le=86400,
        description="Cache TTL in seconds for emission factor lookups",
    )
    lazy_load_site_data: bool = Field(
        True,
        description="Lazy-load site data only when accessed",
    )
    parallel_consolidation: bool = Field(
        True,
        description="Run consolidation across sites in parallel",
    )


class IntegrationConfig(BaseModel):
    """Configuration for integration with other GreenLang components."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    mrv_agents_count: int = Field(
        30, ge=0,
        description="Number of MRV agents to route to for calculations",
    )
    data_agents_count: int = Field(
        20, ge=0,
        description="Number of DATA agents to route to for ingestion",
    )
    foundation_agents: List[str] = Field(
        default_factory=lambda: [
            "FOUND-001-orchestrator",
            "FOUND-002-schema",
            "FOUND-003-normalizer",
            "FOUND-004-assumptions",
            "FOUND-005-citations",
        ],
        description="Foundation agents used by this pack",
    )
    pack_dependencies: List[str] = Field(
        default_factory=lambda: [
            "PACK-041",
            "PACK-042",
            "PACK-044",
            "PACK-045",
            "PACK-046",
        ],
        description="Pack dependencies for multi-site management",
    )


class AlertConfig(BaseModel):
    """Configuration for multi-site management alerting."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    alert_types_enabled: List[AlertType] = Field(
        default_factory=lambda: [
            AlertType.DEADLINE_APPROACHING,
            AlertType.SUBMISSION_OVERDUE,
            AlertType.QUALITY_BELOW_THRESHOLD,
            AlertType.COMPLETENESS_GAP,
        ],
        description="Types of alerts to enable",
    )
    notification_channels: List[str] = Field(
        default_factory=lambda: ["EMAIL"],
        description="Notification delivery channels (EMAIL, SLACK, TEAMS, WEBHOOK)",
    )
    escalation_levels: int = Field(
        3, ge=1, le=5,
        description="Number of escalation levels for overdue items",
    )
    daily_digest: bool = Field(
        False,
        description="Send daily digest of all portfolio alerts",
    )
    quiet_hours_enabled: bool = Field(
        False,
        description="Suppress non-critical alerts outside business hours",
    )


class MigrationConfig(BaseModel):
    """Configuration for database schema migration."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    schema_name: str = Field(
        "ghg_multisite",
        description="Database schema name for multi-site tables",
    )
    table_prefix: str = Field(
        "ms_",
        description="Table name prefix for multi-site tables",
    )
    migration_start: str = Field(
        "V376",
        description="First migration version for PACK-049",
    )
    migration_end: str = Field(
        "V385",
        description="Last migration version for PACK-049",
    )


# =============================================================================
# Main Configuration Model
# =============================================================================


class MultiSitePackConfig(BaseModel):
    """
    Top-level configuration for PACK-049 GHG Multi-Site Management.

    Combines all sub-configurations required for facility registry,
    decentralised data collection, organisational boundary, regional
    factor assignment, consolidation, allocation, comparison,
    completeness, quality, and multi-site reporting.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    company_name: str = Field("", description="Reporting company legal name")
    consolidation_approach: ConsolidationApproach = Field(
        ConsolidationApproach.OPERATIONAL_CONTROL,
        description="Primary consolidation approach for the organisation",
    )
    reporting_year: int = Field(
        2026, ge=2020, le=2035,
        description="Current reporting year",
    )
    base_year: int = Field(
        2020, ge=2015, le=2030,
        description="Base year for GHG inventory",
    )
    country: str = Field(
        "DE",
        description="Primary country of the reporting entity (ISO 3166-1 alpha-2)",
    )
    total_sites: Optional[int] = Field(
        None, ge=1,
        description="Total number of sites in the portfolio (auto-detected if None)",
    )
    scopes_in_scope: List[str] = Field(
        default_factory=lambda: ["SCOPE_1", "SCOPE_2"],
        description="Scopes included in multi-site management",
    )

    site_registry: SiteRegistryConfig = Field(default_factory=SiteRegistryConfig)
    data_collection: DataCollectionConfig = Field(default_factory=DataCollectionConfig)
    boundary: BoundaryConfig = Field(default_factory=BoundaryConfig)
    regional_factor: RegionalFactorConfig = Field(default_factory=RegionalFactorConfig)
    consolidation: ConsolidationConfig = Field(default_factory=ConsolidationConfig)
    allocation: AllocationConfig = Field(default_factory=AllocationConfig)
    comparison: ComparisonConfig = Field(default_factory=ComparisonConfig)
    completion: CompletionConfig = Field(default_factory=CompletionConfig)
    quality: QualityConfig = Field(default_factory=QualityConfig)
    reporting: ReportingConfig = Field(default_factory=ReportingConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)
    integration: IntegrationConfig = Field(default_factory=IntegrationConfig)
    alerts: AlertConfig = Field(default_factory=AlertConfig)
    migration: MigrationConfig = Field(default_factory=MigrationConfig)

    @model_validator(mode="after")
    def validate_base_year_consistency(self) -> MultiSitePackConfig:
        """Ensure base year is before reporting year."""
        if self.base_year > self.reporting_year:
            raise ValueError(
                f"base_year ({self.base_year}) cannot be after "
                f"reporting_year ({self.reporting_year})"
            )
        return self

    @model_validator(mode="after")
    def validate_consolidation_alignment(self) -> MultiSitePackConfig:
        """Ensure consolidation approach is consistent across config."""
        if self.boundary.consolidation_approach != self.consolidation_approach:
            logger.warning(
                "boundary.consolidation_approach (%s) differs from "
                "top-level consolidation_approach (%s). Using top-level value.",
                self.boundary.consolidation_approach.value,
                self.consolidation_approach.value,
            )
        return self

    @model_validator(mode="after")
    def validate_completeness_thresholds(self) -> MultiSitePackConfig:
        """Ensure completeness thresholds are consistent."""
        if self.completion.completeness_target != self.consolidation.completeness_threshold:
            logger.info(
                "completion.completeness_target (%s) differs from "
                "consolidation.completeness_threshold (%s). Both values will be used "
                "in their respective contexts.",
                self.completion.completeness_target,
                self.consolidation.completeness_threshold,
            )
        return self

    @model_validator(mode="after")
    def validate_site_count_vs_max(self) -> MultiSitePackConfig:
        """Warn if total sites exceeds max_sites limit."""
        if self.total_sites is not None:
            if self.total_sites > self.site_registry.max_sites:
                raise ValueError(
                    f"total_sites ({self.total_sites}) exceeds "
                    f"site_registry.max_sites ({self.site_registry.max_sites})"
                )
        return self

    @model_validator(mode="after")
    def validate_allocation_dependencies(self) -> MultiSitePackConfig:
        """Warn about allocation config dependencies."""
        if self.allocation.landlord_tenant_enabled:
            if self.allocation.landlord_tenant_approach == LandlordTenantSplit.SUB_METERED:
                logger.info(
                    "Sub-metered landlord-tenant split requires metering infrastructure. "
                    "Ensure IOT_FEED or API_PUSH is in data_entry_modes."
                )
        return self


# =============================================================================
# Pack Configuration Wrapper
# =============================================================================


class PackConfig(BaseModel):
    """
    Top-level wrapper for PACK-049 configuration.

    Provides factory methods for loading from presets, YAML files,
    environment overrides, and runtime merges. Includes SHA-256
    config hashing for provenance tracking.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    pack: MultiSitePackConfig = Field(default_factory=MultiSitePackConfig)
    preset_name: Optional[str] = Field(None, description="Name of the loaded preset")
    config_version: str = Field("1.0.0", description="Configuration schema version")
    pack_id: str = Field("PACK-049-multi-site", description="Unique pack identifier")

    @classmethod
    def from_preset(cls, preset_name: str, overrides: Optional[Dict[str, Any]] = None) -> PackConfig:
        """
        Load configuration from a named industry preset.

        Args:
            preset_name: Key from AVAILABLE_PRESETS (e.g., 'corporate_general').
            overrides: Optional dict of overrides applied after preset load.

        Returns:
            Fully initialised PackConfig.

        Raises:
            ValueError: If preset_name is not recognised.
            FileNotFoundError: If preset YAML file is missing.
        """
        if preset_name not in AVAILABLE_PRESETS:
            raise ValueError(
                f"Unknown preset: {preset_name}. Available: {sorted(AVAILABLE_PRESETS.keys())}"
            )
        preset_path = PACK_BASE_DIR / "presets" / f"{preset_name}.yaml"
        if not preset_path.exists():
            raise FileNotFoundError(f"Preset file not found: {preset_path}")
        with open(preset_path, "r", encoding="utf-8") as f:
            preset_data = yaml.safe_load(f) or {}
        env_overrides = cls._load_env_overrides()
        if env_overrides:
            preset_data = cls._deep_merge(preset_data, env_overrides)
        if overrides:
            preset_data = cls._deep_merge(preset_data, overrides)
        pack_config = MultiSitePackConfig(**preset_data)
        return cls(pack=pack_config, preset_name=preset_name)

    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> PackConfig:
        """
        Load configuration from an arbitrary YAML file.

        Args:
            yaml_path: Path to the YAML configuration file.

        Returns:
            Fully initialised PackConfig.

        Raises:
            FileNotFoundError: If the YAML file does not exist.
        """
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {yaml_path}")
        with open(yaml_path, "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f) or {}
        pack_config = MultiSitePackConfig(**config_data)
        return cls(pack=pack_config)

    @classmethod
    def merge(cls, base: PackConfig, overrides: Dict[str, Any]) -> PackConfig:
        """
        Create a new PackConfig by merging overrides into a base config.

        Args:
            base: Existing PackConfig to use as the base.
            overrides: Dict of overrides (supports nested keys).

        Returns:
            New PackConfig with merged values.
        """
        base_dict = base.pack.model_dump()
        merged = cls._deep_merge(base_dict, overrides)
        pack_config = MultiSitePackConfig(**merged)
        return cls(
            pack=pack_config,
            preset_name=base.preset_name,
            config_version=base.config_version,
        )

    @staticmethod
    def _load_env_overrides() -> Dict[str, Any]:
        """
        Load configuration overrides from environment variables.

        Environment variables prefixed with MULTISITE_PACK_ are parsed.
        Double underscores denote nested keys.
        Example: MULTISITE_PACK_BOUNDARY__CONSOLIDATION_APPROACH=EQUITY_SHARE
        """
        overrides: Dict[str, Any] = {}
        prefix = "MULTISITE_PACK_"
        for key, value in os.environ.items():
            if key.startswith(prefix):
                config_key = key[len(prefix):].lower()
                parts = config_key.split("__")
                current = overrides
                for part in parts[:-1]:
                    current = current.setdefault(part, {})
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

    @staticmethod
    def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively merge override dict into base dict."""
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = PackConfig._deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    def get_config_hash(self) -> str:
        """
        Compute SHA-256 hash of the full configuration.

        Returns:
            Hex-encoded SHA-256 hash string for provenance tracking.
        """
        config_json = self.model_dump_json(indent=None)
        return _compute_hash(config_json)

    def validate_completeness(self) -> List[str]:
        """
        Run domain-specific validation checks on the configuration.

        Returns:
            List of warning messages (empty list means no issues).
        """
        return validate_config(self.pack)

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialise the full configuration to a plain dictionary.

        Returns:
            Dict representation of the entire PackConfig.
        """
        return self.model_dump()

    def get_active_scopes(self) -> List[str]:
        """
        Return the list of active emission scopes.

        Returns:
            List of scope strings (e.g., ['SCOPE_1', 'SCOPE_2']).
        """
        return list(self.pack.scopes_in_scope)


# =============================================================================
# Utility Functions
# =============================================================================


def load_preset(preset_name: str, overrides: Optional[Dict[str, Any]] = None) -> PackConfig:
    """
    Convenience function to load a preset configuration.

    Args:
        preset_name: Key from AVAILABLE_PRESETS.
        overrides: Optional dict of overrides.

    Returns:
        Initialised PackConfig from the named preset.
    """
    return PackConfig.from_preset(preset_name, overrides)


def validate_config(config: MultiSitePackConfig) -> List[str]:
    """
    Validate configuration for domain-specific consistency.

    Args:
        config: The multi-site pack configuration to validate.

    Returns:
        List of warning strings. Empty list indicates no issues found.
    """
    warnings: List[str] = []

    # Company name check
    if not config.company_name:
        warnings.append("No company_name configured.")

    # Site count check
    if config.total_sites is None:
        warnings.append(
            "total_sites not set. Portfolio size will be auto-detected "
            "from registered facilities."
        )

    # Consolidation approach guidance
    if config.consolidation_approach == ConsolidationApproach.EQUITY_SHARE:
        warnings.append(
            "Equity share approach requires equity percentages for all "
            "facilities. Ensure ownership data is complete."
        )

    # Boundary materiality check
    if config.boundary.materiality_threshold > Decimal("0.10"):
        warnings.append(
            f"Materiality threshold ({config.boundary.materiality_threshold}) "
            "is above 10%. This may exclude significant emission sources."
        )

    # Completeness target vs estimation
    if config.completion.completeness_target > Decimal("0.98"):
        if not config.completion.auto_estimation_enabled:
            warnings.append(
                "Completeness target >98% but auto-estimation is disabled. "
                "This target may be difficult to achieve without estimation."
            )

    # Allocation method consistency
    if config.allocation.landlord_tenant_enabled:
        if not config.allocation.shared_services_enabled:
            warnings.append(
                "Landlord-tenant allocation enabled but shared-services disabled. "
                "Consider enabling both for complete allocation coverage."
            )

    # Factor tier alignment
    if config.regional_factor.preferred_tier == FactorTier.TIER_3_FACILITY:
        warnings.append(
            "Facility-specific factors (Tier 3) are preferred. Ensure "
            "supplier-specific data collection processes are in place."
        )

    # Data collection mode coverage
    if len(config.data_collection.data_entry_modes) < 2:
        warnings.append(
            "Only one data entry mode configured. Consider enabling "
            "multiple modes for flexibility across diverse facilities."
        )

    # Quality weights validation
    total_weight = sum(config.quality.quality_weights.values())
    if abs(total_weight - Decimal("1.0")) > Decimal("0.01"):
        warnings.append(
            f"Quality weights sum to {total_weight}, expected 1.0. "
            "Scores may not be properly normalised."
        )

    # Security configuration
    if config.security.rls_enabled and not config.security.audit_enabled:
        warnings.append(
            "Row-level security enabled but audit trail disabled. "
            "Consider enabling audit for access tracking."
        )

    # Performance check for large portfolios
    if config.total_sites is not None and config.total_sites > 1000:
        if config.performance.max_concurrent_sites < 100:
            warnings.append(
                f"Large portfolio ({config.total_sites} sites) with low "
                f"concurrency ({config.performance.max_concurrent_sites}). "
                "Consider increasing max_concurrent_sites for performance."
            )

    # Comparison peer group
    if config.comparison.peer_group_min_size > 10:
        warnings.append(
            f"Peer group minimum size ({config.comparison.peer_group_min_size}) "
            "is large. Some facility types may not form valid peer groups."
        )

    # Report site limit
    if config.total_sites is not None:
        if config.total_sites > config.reporting.max_report_sites:
            warnings.append(
                f"total_sites ({config.total_sites}) exceeds "
                f"reporting.max_report_sites ({config.reporting.max_report_sites}). "
                "Reports may be truncated."
            )

    return warnings


def get_default_config(
    approach: ConsolidationApproach = ConsolidationApproach.OPERATIONAL_CONTROL,
) -> MultiSitePackConfig:
    """
    Create a default configuration for the given consolidation approach.

    Args:
        approach: GHG Protocol consolidation approach.

    Returns:
        Default MultiSitePackConfig for the approach.
    """
    return MultiSitePackConfig(consolidation_approach=approach)


def list_available_presets() -> Dict[str, str]:
    """
    Return a copy of all available preset names and descriptions.

    Returns:
        Dict mapping preset name to human-readable description.
    """
    return AVAILABLE_PRESETS.copy()


def get_facility_type_defaults(facility_type: str) -> Optional[Dict[str, Any]]:
    """
    Return default characteristics for a facility type.

    Args:
        facility_type: FacilityType value string.

    Returns:
        Dict of facility characteristics, or None if not found.
    """
    return DEFAULT_FACILITY_TYPES.get(facility_type)


def get_consolidation_guidance(approach: str) -> Optional[Dict[str, Any]]:
    """
    Return consolidation approach guidance.

    Args:
        approach: ConsolidationApproach value string.

    Returns:
        Dict of guidance data, or None if not found.
    """
    return CONSOLIDATION_APPROACH_GUIDANCE.get(approach)


def get_regional_factor_database(source: str) -> Optional[Dict[str, Any]]:
    """
    Return metadata for a regional emission factor database.

    Args:
        source: FactorSource value string.

    Returns:
        Dict of database metadata, or None if not found.
    """
    return REGIONAL_FACTOR_DATABASES.get(source)


def get_quality_weights() -> Dict[str, Decimal]:
    """
    Return default quality dimension weights.

    Returns:
        Dict mapping quality dimension to weight.
    """
    return dict(DEFAULT_QUALITY_WEIGHTS)


def get_allocation_priorities() -> List[str]:
    """
    Return default allocation method priority order.

    Returns:
        List of allocation method strings in priority order.
    """
    return list(DEFAULT_ALLOCATION_PRIORITIES)
