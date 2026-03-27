"""
PACK-050 GHG Consolidation Pack - Configuration Manager

Pydantic v2 configuration for multi-entity corporate GHG consolidation
including entity registry, ownership chain management, organisational
boundary definition per GHG Protocol Corporate Standard Chapter 3,
equity share / operational control / financial control consolidation
approaches, inter-company elimination, M&A event handling, adjustment
and restatement processing, group reporting with multi-framework output,
and comprehensive audit trail generation.

Configuration Merge Order (later overrides earlier):
    1. Base pack.yaml manifest
    2. Preset YAML (organisation-type-specific defaults)
    3. Environment overrides (CONSOLIDATION_PACK_* environment variables)
    4. Explicit runtime overrides

Regulatory Context:
    GHG Protocol Corporate Standard (2004, revised 2015) - Chapter 3
    GHG Protocol Scope 2 Guidance (2015) - Dual reporting
    ISO 14064-1:2018 Clause 5 - Organisational boundaries
    EU CSRD (2022/2464) - ESRS E1 consolidated disclosure
    US SEC Climate Disclosure Rules (2024) - Registrant boundary
    IFRS S2 - Climate-related financial disclosures
    GRI 305 (2016) - Consolidated GHG emissions
    CDP Climate Change Questionnaire - C6/C7 consolidation
    SBTi Corporate Net-Zero Standard - Boundary requirements
    PCAF Global Standard v3 (2024) - Financed emissions consolidation

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


class EntityType(str, Enum):
    """Classification of legal entity type within the corporate group."""
    SUBSIDIARY = "SUBSIDIARY"
    JOINT_VENTURE = "JOINT_VENTURE"
    ASSOCIATE = "ASSOCIATE"
    DIVISION = "DIVISION"
    BRANCH = "BRANCH"
    SPV = "SPV"
    FRANCHISE = "FRANCHISE"
    PARTNERSHIP = "PARTNERSHIP"


class EntityLifecycle(str, Enum):
    """Lifecycle stage of a legal entity within the group structure."""
    ACTIVE = "ACTIVE"
    DORMANT = "DORMANT"
    ACQUIRED = "ACQUIRED"
    DIVESTED = "DIVESTED"
    MERGED = "MERGED"
    LIQUIDATED = "LIQUIDATED"


class ConsolidationApproach(str, Enum):
    """GHG Protocol organisational boundary consolidation approach (Chapter 3)."""
    EQUITY_SHARE = "EQUITY_SHARE"
    OPERATIONAL_CONTROL = "OPERATIONAL_CONTROL"
    FINANCIAL_CONTROL = "FINANCIAL_CONTROL"


class ControlType(str, Enum):
    """Type of control the parent has over an entity."""
    OPERATIONAL = "OPERATIONAL"
    FINANCIAL = "FINANCIAL"
    NO_CONTROL = "NO_CONTROL"


class OwnershipType(str, Enum):
    """Ownership relationship between parent and entity."""
    WHOLLY_OWNED = "WHOLLY_OWNED"
    MAJORITY = "MAJORITY"
    MINORITY = "MINORITY"
    EQUAL_JV = "EQUAL_JV"
    ASSOCIATE = "ASSOCIATE"


class EliminationType(str, Enum):
    """Type of inter-company emission transfer to eliminate."""
    ENERGY_TRANSFER = "ENERGY_TRANSFER"
    WASTE_TRANSFER = "WASTE_TRANSFER"
    PRODUCT_TRANSFER = "PRODUCT_TRANSFER"
    SERVICE_TRANSFER = "SERVICE_TRANSFER"


class AdjustmentType(str, Enum):
    """Type of adjustment or restatement applied to consolidated inventory."""
    METHODOLOGY_CHANGE = "METHODOLOGY_CHANGE"
    ERROR_CORRECTION = "ERROR_CORRECTION"
    SCOPE_RECLASSIFICATION = "SCOPE_RECLASSIFICATION"
    TIMING = "TIMING"
    LATE_SUBMISSION = "LATE_SUBMISSION"


class MnAEventType(str, Enum):
    """Type of merger, acquisition, or structural change event."""
    ACQUISITION = "ACQUISITION"
    DIVESTITURE = "DIVESTITURE"
    MERGER = "MERGER"
    DEMERGER = "DEMERGER"
    JV_FORMATION = "JV_FORMATION"
    JV_DISSOLUTION = "JV_DISSOLUTION"


class ReportingFramework(str, Enum):
    """Target regulatory or voluntary reporting framework."""
    CSRD_ESRS_E1 = "CSRD_ESRS_E1"
    CDP = "CDP"
    GRI_305 = "GRI_305"
    TCFD = "TCFD"
    SEC_CLIMATE = "SEC_CLIMATE"
    SBTI = "SBTI"
    IFRS_S2 = "IFRS_S2"
    UK_SECR = "UK_SECR"
    NGER = "NGER"


class DataQualityTier(str, Enum):
    """Data quality tier for entity-level emission data."""
    VERIFIED = "VERIFIED"
    AUDITED = "AUDITED"
    REPORTED = "REPORTED"
    ESTIMATED = "ESTIMATED"
    EXTRAPOLATED = "EXTRAPOLATED"


class CompletionStatus(str, Enum):
    """Completion status of entity-level data submission."""
    COMPLETE = "COMPLETE"
    PARTIAL = "PARTIAL"
    MISSING = "MISSING"
    OVERDUE = "OVERDUE"


class ApprovalStatus(str, Enum):
    """Approval status for consolidated results."""
    DRAFT = "DRAFT"
    SUBMITTED = "SUBMITTED"
    UNDER_REVIEW = "UNDER_REVIEW"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"


class ReportType(str, Enum):
    """Type of consolidation report output."""
    CONSOLIDATED_GHG = "CONSOLIDATED_GHG"
    ENTITY_BREAKDOWN = "ENTITY_BREAKDOWN"
    OWNERSHIP = "OWNERSHIP"
    EQUITY_SHARE = "EQUITY_SHARE"
    ELIMINATION_LOG = "ELIMINATION_LOG"
    MNA_IMPACT = "MNA_IMPACT"
    SCOPE_BREAKDOWN = "SCOPE_BREAKDOWN"
    TREND = "TREND"
    REGULATORY = "REGULATORY"
    DASHBOARD = "DASHBOARD"


class ExportFormat(str, Enum):
    """Supported report output formats."""
    MARKDOWN = "MARKDOWN"
    HTML = "HTML"
    JSON = "JSON"
    CSV = "CSV"
    PDF = "PDF"
    XLSX = "XLSX"


class AlertType(str, Enum):
    """Type of consolidation management alert."""
    DEADLINE = "DEADLINE"
    COMPLETENESS = "COMPLETENESS"
    VARIANCE = "VARIANCE"
    BOUNDARY_CHANGE = "BOUNDARY_CHANGE"
    MNA_EVENT = "MNA_EVENT"
    APPROVAL = "APPROVAL"


class ScopeCategory(str, Enum):
    """GHG emission scope category for consolidation."""
    SCOPE_1 = "SCOPE_1"
    SCOPE_2_LOCATION = "SCOPE_2_LOCATION"
    SCOPE_2_MARKET = "SCOPE_2_MARKET"
    SCOPE_3 = "SCOPE_3"


class MaterialityThreshold(str, Enum):
    """Materiality threshold for boundary inclusion."""
    NONE = "NONE"
    ONE_PCT = "ONE_PCT"
    FIVE_PCT = "FIVE_PCT"
    TEN_PCT = "TEN_PCT"


class ProRataMethod(str, Enum):
    """Method for pro-rata apportionment in M&A events."""
    CALENDAR_DAYS = "CALENDAR_DAYS"
    REPORTING_MONTHS = "REPORTING_MONTHS"
    FINANCIAL_QUARTERS = "FINANCIAL_QUARTERS"


# =============================================================================
# Reference Data Constants
# =============================================================================


DEFAULT_ENTITY_TYPES: Dict[str, Dict[str, Any]] = {
    "SUBSIDIARY": {
        "typical_ownership_pct": 100.0,
        "typical_control": "FINANCIAL",
        "inclusion_approach": "All three approaches",
        "description": "Wholly or majority-owned subsidiary of the parent",
    },
    "JOINT_VENTURE": {
        "typical_ownership_pct": 50.0,
        "typical_control": "NO_CONTROL",
        "inclusion_approach": "Equity share (proportional); Control approaches (depends on JV agreement)",
        "description": "Joint venture with shared ownership and governance",
    },
    "ASSOCIATE": {
        "typical_ownership_pct": 30.0,
        "typical_control": "NO_CONTROL",
        "inclusion_approach": "Equity share only (20-50% ownership, significant influence)",
        "description": "Entity with significant influence but not control",
    },
    "DIVISION": {
        "typical_ownership_pct": 100.0,
        "typical_control": "OPERATIONAL",
        "inclusion_approach": "All three approaches (internal division of parent)",
        "description": "Internal business division of the parent entity",
    },
    "BRANCH": {
        "typical_ownership_pct": 100.0,
        "typical_control": "OPERATIONAL",
        "inclusion_approach": "All three approaches (branch of parent)",
        "description": "Branch office or regional operating unit",
    },
    "SPV": {
        "typical_ownership_pct": 100.0,
        "typical_control": "FINANCIAL",
        "inclusion_approach": "Financial control (structured entity)",
        "description": "Special purpose vehicle for specific assets or projects",
    },
    "FRANCHISE": {
        "typical_ownership_pct": 0.0,
        "typical_control": "OPERATIONAL",
        "inclusion_approach": "Operational control (if franchisor sets operating policies)",
        "description": "Franchise operation under brand licensing agreement",
    },
    "PARTNERSHIP": {
        "typical_ownership_pct": 50.0,
        "typical_control": "NO_CONTROL",
        "inclusion_approach": "Equity share (proportional to partnership interest)",
        "description": "General or limited partnership arrangement",
    },
}


DEFAULT_OWNERSHIP_THRESHOLDS: Dict[str, Dict[str, Any]] = {
    "WHOLLY_OWNED": {
        "min_pct": Decimal("100.0"),
        "max_pct": Decimal("100.0"),
        "consolidation_factor": Decimal("1.0"),
        "description": "100% ownership - full consolidation under all approaches",
    },
    "MAJORITY": {
        "min_pct": Decimal("50.01"),
        "max_pct": Decimal("99.99"),
        "consolidation_factor_equity": "ownership_pct / 100",
        "consolidation_factor_control": Decimal("1.0"),
        "description": "Majority ownership - full consolidation under control, proportional under equity share",
    },
    "MINORITY": {
        "min_pct": Decimal("0.01"),
        "max_pct": Decimal("49.99"),
        "consolidation_factor_equity": "ownership_pct / 100",
        "consolidation_factor_control": Decimal("0.0"),
        "description": "Minority ownership - proportional under equity share only, excluded under control",
    },
    "EQUAL_JV": {
        "min_pct": Decimal("50.0"),
        "max_pct": Decimal("50.0"),
        "consolidation_factor_equity": Decimal("0.50"),
        "consolidation_factor_control": "depends on JV agreement",
        "description": "50/50 JV - proportional under equity share, control depends on agreement",
    },
    "ASSOCIATE": {
        "min_pct": Decimal("20.0"),
        "max_pct": Decimal("50.0"),
        "consolidation_factor_equity": "ownership_pct / 100",
        "consolidation_factor_control": Decimal("0.0"),
        "description": "Associate with significant influence - equity share approach only",
    },
}


DEFAULT_ELIMINATION_RULES: Dict[str, Dict[str, Any]] = {
    "ENERGY_TRANSFER": {
        "description": "Intra-group electricity, steam, heat, or cooling transfers",
        "scope_impact": "Scope 2 (buyer) and Scope 1 (generator)",
        "elimination_method": "Net to zero within consolidated boundary",
        "documentation_required": "Transfer agreements, meter data, allocation keys",
    },
    "WASTE_TRANSFER": {
        "description": "Intra-group waste processing or treatment",
        "scope_impact": "Scope 3 Category 5 (sender) and Scope 1 (processor)",
        "elimination_method": "Remove from both entities within boundary",
        "documentation_required": "Waste transfer notes, processing records",
    },
    "PRODUCT_TRANSFER": {
        "description": "Intra-group product or raw material transfers",
        "scope_impact": "Scope 3 upstream (buyer) and Scope 3 downstream (seller)",
        "elimination_method": "Eliminate upstream/downstream double count",
        "documentation_required": "Internal invoices, bill of materials",
    },
    "SERVICE_TRANSFER": {
        "description": "Intra-group service provision (e.g., shared IT, transport)",
        "scope_impact": "Scope 3 Category 1 (buyer) and revenue of provider",
        "elimination_method": "Remove purchased services within boundary",
        "documentation_required": "Service level agreements, cost allocations",
    },
}


DEFAULT_MNA_RULES: Dict[str, Dict[str, Any]] = {
    "ACQUISITION": {
        "boundary_impact": "Add acquired entity to consolidated boundary",
        "pro_rata": "Emissions from completion date to period end",
        "base_year_impact": "Recalculate base year to include acquired entity",
        "ghg_protocol_ref": "Chapter 5: Tracking Emissions Over Time",
    },
    "DIVESTITURE": {
        "boundary_impact": "Remove divested entity from consolidated boundary",
        "pro_rata": "Emissions from period start to completion date",
        "base_year_impact": "Recalculate base year to exclude divested entity",
        "ghg_protocol_ref": "Chapter 5: Tracking Emissions Over Time",
    },
    "MERGER": {
        "boundary_impact": "Merge entities into combined boundary",
        "pro_rata": "Combined emissions from merger date",
        "base_year_impact": "Recalculate base year for merged entity",
        "ghg_protocol_ref": "Chapter 5: Tracking Emissions Over Time",
    },
    "DEMERGER": {
        "boundary_impact": "Split entity boundary into separate entities",
        "pro_rata": "Allocated emissions based on demerger allocation",
        "base_year_impact": "Recalculate base year for each resulting entity",
        "ghg_protocol_ref": "Chapter 5: Tracking Emissions Over Time",
    },
    "JV_FORMATION": {
        "boundary_impact": "Add JV entity with equity share or control status",
        "pro_rata": "Proportional emissions from formation date",
        "base_year_impact": "Include JV in base year if material",
        "ghg_protocol_ref": "Chapter 3: Setting Organisational Boundaries",
    },
    "JV_DISSOLUTION": {
        "boundary_impact": "Remove JV entity from boundary",
        "pro_rata": "Proportional emissions up to dissolution date",
        "base_year_impact": "Exclude JV from base year going forward",
        "ghg_protocol_ref": "Chapter 3: Setting Organisational Boundaries",
    },
}


DEFAULT_FRAMEWORK_REQUIREMENTS: Dict[str, Dict[str, Any]] = {
    "CSRD_ESRS_E1": {
        "name": "EU CSRD / ESRS E1",
        "scopes_required": ["SCOPE_1", "SCOPE_2_LOCATION", "SCOPE_2_MARKET", "SCOPE_3"],
        "consolidation_approach": "Financial control (aligned with IFRS consolidation)",
        "assurance_level": "Limited (moving to reasonable)",
        "base_year_required": True,
        "intensity_metrics_required": True,
    },
    "CDP": {
        "name": "CDP Climate Change",
        "scopes_required": ["SCOPE_1", "SCOPE_2_LOCATION", "SCOPE_2_MARKET", "SCOPE_3"],
        "consolidation_approach": "Any (must disclose which approach)",
        "assurance_level": "Recommended",
        "base_year_required": True,
        "intensity_metrics_required": True,
    },
    "GRI_305": {
        "name": "GRI 305 Emissions",
        "scopes_required": ["SCOPE_1", "SCOPE_2_LOCATION", "SCOPE_2_MARKET"],
        "consolidation_approach": "Aligned with financial reporting",
        "assurance_level": "Recommended",
        "base_year_required": True,
        "intensity_metrics_required": True,
    },
    "SEC_CLIMATE": {
        "name": "US SEC Climate Disclosure",
        "scopes_required": ["SCOPE_1", "SCOPE_2"],
        "consolidation_approach": "Registrant boundary (financial reporting entity)",
        "assurance_level": "Limited (Scope 1+2), phased-in",
        "base_year_required": False,
        "intensity_metrics_required": True,
    },
    "SBTI": {
        "name": "Science Based Targets initiative",
        "scopes_required": ["SCOPE_1", "SCOPE_2", "SCOPE_3"],
        "consolidation_approach": "Consistent with chosen approach",
        "assurance_level": "Recommended",
        "base_year_required": True,
        "intensity_metrics_required": True,
    },
    "IFRS_S2": {
        "name": "IFRS S2 Climate-related Disclosures",
        "scopes_required": ["SCOPE_1", "SCOPE_2", "SCOPE_3"],
        "consolidation_approach": "Aligned with financial reporting entity",
        "assurance_level": "Reasonable (phased-in)",
        "base_year_required": False,
        "intensity_metrics_required": True,
    },
    "UK_SECR": {
        "name": "UK Streamlined Energy and Carbon Reporting",
        "scopes_required": ["SCOPE_1", "SCOPE_2"],
        "consolidation_approach": "Financial control (Companies Act entity)",
        "assurance_level": "Not required",
        "base_year_required": False,
        "intensity_metrics_required": True,
    },
    "NGER": {
        "name": "Australia National Greenhouse and Energy Reporting",
        "scopes_required": ["SCOPE_1", "SCOPE_2"],
        "consolidation_approach": "Operational control",
        "assurance_level": "Required above threshold",
        "base_year_required": False,
        "intensity_metrics_required": False,
    },
}


AVAILABLE_PRESETS: Dict[str, str] = {
    "corporate_conglomerate": (
        "Large diversified conglomerate with 100+ entities, multi-tier ownership, "
        "complex inter-company eliminations, and all three consolidation approaches"
    ),
    "financial_holding": (
        "Financial holding company with PCAF integration, financed emissions "
        "consolidation, bank subsidiaries, and financial control approach"
    ),
    "jv_partnership": (
        "JV-heavy corporate structure with equity share focus, partner "
        "reconciliation, proportional allocation, and JV governance tracking"
    ),
    "multinational": (
        "Multi-country multinational with regional consolidation tiers, "
        "multi-currency revenue normalisation, and jurisdiction-specific reporting"
    ),
    "private_equity": (
        "Private equity firm with portfolio company consolidation, vintage "
        "tracking, fund-level aggregation, and investor reporting"
    ),
    "real_estate_fund": (
        "Real estate investment trust or fund with property-level roll-up, "
        "tenant allocation, CRREM alignment, and GRESB reporting"
    ),
    "public_company": (
        "Listed public company with SEC/CSRD compliance, assurance-ready "
        "consolidation, investor-grade reporting, and board sign-off workflow"
    ),
    "sme_group": (
        "Small-to-medium enterprise group with fewer entities, simplified "
        "workflows, streamlined consolidation, and reduced complexity"
    ),
}


# =============================================================================
# Sub-Config Models (16 Pydantic v2 models)
# =============================================================================


class EntityRegistryConfig(BaseModel):
    """Configuration for the entity registry and corporate structure management."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    max_entities: int = Field(
        500, ge=1, le=50000,
        description="Maximum number of entities in the corporate group",
    )
    entity_types_enabled: List[EntityType] = Field(
        default_factory=lambda: [
            EntityType.SUBSIDIARY,
            EntityType.JOINT_VENTURE,
            EntityType.ASSOCIATE,
            EntityType.DIVISION,
            EntityType.BRANCH,
        ],
        description="Entity types enabled for registration",
    )
    lifecycle_tracking: bool = Field(
        True,
        description="Track entity lifecycle stages (active through liquidated)",
    )
    hierarchy_depth_limit: int = Field(
        10, ge=1, le=20,
        description="Maximum depth of the ownership hierarchy tree",
    )
    require_legal_entity_id: bool = Field(
        False,
        description="Require a legal entity identifier (LEI) for each entity",
    )
    require_country_code: bool = Field(
        True,
        description="Require ISO 3166-1 country code for each entity",
    )
    custom_attributes_enabled: bool = Field(
        True,
        description="Allow custom metadata attributes on entity records",
    )
    auto_classify_ownership: bool = Field(
        True,
        description="Auto-classify ownership type based on equity percentage",
    )

    @field_validator("max_entities")
    @classmethod
    def validate_max_entities(cls, v: int) -> int:
        """Validate max entities is within reasonable bounds."""
        if v < 1:
            raise ValueError("max_entities must be at least 1")
        return v


class OwnershipConfig(BaseModel):
    """Configuration for ownership chain management and resolution."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    multi_tier_enabled: bool = Field(
        True,
        description="Enable multi-tier indirect ownership chain resolution",
    )
    max_chain_depth: int = Field(
        10, ge=1, le=20,
        description="Maximum depth for indirect ownership chain resolution",
    )
    effective_equity_method: str = Field(
        "MULTIPLICATIVE",
        description="Method for computing effective equity through chain (MULTIPLICATIVE, ADDITIVE)",
    )
    circular_ownership_detection: bool = Field(
        True,
        description="Detect and prevent circular ownership references",
    )
    minority_interest_threshold_pct: Decimal = Field(
        Decimal("20.0"),
        ge=Decimal("0.0"), le=Decimal("50.0"),
        description="Threshold below which entity is classified as minority interest",
    )
    associate_threshold_pct: Decimal = Field(
        Decimal("20.0"),
        ge=Decimal("10.0"), le=Decimal("50.0"),
        description="Minimum equity for associate classification (significant influence)",
    )
    require_evidence_for_control: bool = Field(
        True,
        description="Require documentary evidence for control classification",
    )
    track_ownership_changes: bool = Field(
        True,
        description="Track historical ownership changes with effective dates",
    )

    @field_validator("effective_equity_method")
    @classmethod
    def validate_equity_method(cls, v: str) -> str:
        """Validate effective equity computation method."""
        allowed = {"MULTIPLICATIVE", "ADDITIVE"}
        if v.upper() not in allowed:
            raise ValueError(
                f"effective_equity_method must be one of {allowed}, got '{v}'"
            )
        return v.upper()


class BoundaryConfig(BaseModel):
    """Configuration for organisational boundary determination."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    consolidation_approach: ConsolidationApproach = Field(
        ConsolidationApproach.OPERATIONAL_CONTROL,
        description="GHG Protocol consolidation approach for the organisation",
    )
    materiality_threshold: MaterialityThreshold = Field(
        MaterialityThreshold.FIVE_PCT,
        description="Materiality threshold for boundary inclusion screening",
    )
    materiality_threshold_pct: Decimal = Field(
        Decimal("0.05"),
        ge=Decimal("0.0"), le=Decimal("0.20"),
        description="Numeric materiality threshold (0.05 = 5%)",
    )
    de_minimis_threshold_pct: Decimal = Field(
        Decimal("0.01"),
        ge=Decimal("0.001"), le=Decimal("0.10"),
        description="De minimis threshold below which sources may be excluded (0.01 = 1%)",
    )
    annual_boundary_lock: bool = Field(
        True,
        description="Lock boundary at start of reporting year",
    )
    track_boundary_changes: bool = Field(
        True,
        description="Track and log all boundary changes with justification",
    )
    require_boundary_approval: bool = Field(
        True,
        description="Require formal approval for boundary changes",
    )
    dual_approach_enabled: bool = Field(
        False,
        description="Run consolidation under two approaches simultaneously for comparison",
    )
    secondary_approach: Optional[ConsolidationApproach] = Field(
        None,
        description="Secondary consolidation approach when dual_approach_enabled is True",
    )
    scopes_in_boundary: List[ScopeCategory] = Field(
        default_factory=lambda: [
            ScopeCategory.SCOPE_1,
            ScopeCategory.SCOPE_2_LOCATION,
            ScopeCategory.SCOPE_2_MARKET,
        ],
        description="Emission scopes included in the consolidation boundary",
    )


class EquityShareConfig(BaseModel):
    """Configuration for equity share consolidation approach."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    default_equity_pct: Decimal = Field(
        Decimal("100.0"),
        ge=Decimal("0.0"), le=Decimal("100.0"),
        description="Default equity share for wholly-owned subsidiaries",
    )
    round_equity_to_dp: int = Field(
        2, ge=0, le=6,
        description="Decimal places for equity share rounding",
    )
    include_associates: bool = Field(
        True,
        description="Include associates (20-50% equity) in equity share consolidation",
    )
    include_minority_interests: bool = Field(
        True,
        description="Include minority interests (<20% equity) in equity share consolidation",
    )
    waterfall_reconciliation: bool = Field(
        True,
        description="Generate equity waterfall showing parent to subsidiary attribution",
    )
    proportional_scope3: bool = Field(
        False,
        description="Apply equity share proportionality to Scope 3 as well",
    )


class ControlApproachConfig(BaseModel):
    """Configuration for operational and financial control consolidation."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    control_test_method: str = Field(
        "POLICY_AUTHORITY",
        description="Method for determining operational control (POLICY_AUTHORITY, MAJORITY_BOARD, COMBINED)",
    )
    franchise_inclusion: bool = Field(
        True,
        description="Include franchises under operational control approach",
    )
    leased_asset_treatment: str = Field(
        "FINANCE_LEASE_INCLUDE",
        description="Treatment of leased assets (FINANCE_LEASE_INCLUDE, ALL_EXCLUDE, CASE_BY_CASE)",
    )
    spv_consolidation: bool = Field(
        True,
        description="Consolidate SPVs under financial control approach",
    )
    jv_control_override: bool = Field(
        False,
        description="Allow manual override of JV control classification",
    )
    document_control_basis: bool = Field(
        True,
        description="Require documentation of the basis for control determination",
    )

    @field_validator("control_test_method")
    @classmethod
    def validate_control_test(cls, v: str) -> str:
        """Validate control test method."""
        allowed = {"POLICY_AUTHORITY", "MAJORITY_BOARD", "COMBINED"}
        if v.upper() not in allowed:
            raise ValueError(f"control_test_method must be one of {allowed}, got '{v}'")
        return v.upper()

    @field_validator("leased_asset_treatment")
    @classmethod
    def validate_lease_treatment(cls, v: str) -> str:
        """Validate leased asset treatment."""
        allowed = {"FINANCE_LEASE_INCLUDE", "ALL_EXCLUDE", "CASE_BY_CASE"}
        if v.upper() not in allowed:
            raise ValueError(
                f"leased_asset_treatment must be one of {allowed}, got '{v}'"
            )
        return v.upper()


class EliminationConfig(BaseModel):
    """Configuration for inter-company emission elimination."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    elimination_enabled: bool = Field(
        True,
        description="Enable inter-company emission elimination during consolidation",
    )
    elimination_types: List[EliminationType] = Field(
        default_factory=lambda: [
            EliminationType.ENERGY_TRANSFER,
            EliminationType.WASTE_TRANSFER,
            EliminationType.PRODUCT_TRANSFER,
            EliminationType.SERVICE_TRANSFER,
        ],
        description="Types of inter-company transfers to eliminate",
    )
    require_matching_entries: bool = Field(
        True,
        description="Require matching elimination entries from both counterparties",
    )
    tolerance_pct: Decimal = Field(
        Decimal("5.0"),
        ge=Decimal("0.0"), le=Decimal("20.0"),
        description="Tolerance for matching elimination entries between counterparties (%)",
    )
    auto_detect_transfers: bool = Field(
        False,
        description="Auto-detect potential inter-company transfers from transaction data",
    )
    elimination_log_enabled: bool = Field(
        True,
        description="Generate detailed elimination log for audit trail",
    )
    net_to_zero_validation: bool = Field(
        True,
        description="Validate that eliminations net to zero within consolidated boundary",
    )


class MnAConfig(BaseModel):
    """Configuration for merger, acquisition, and structural change handling."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    mna_tracking_enabled: bool = Field(
        True,
        description="Enable M&A event tracking and processing",
    )
    pro_rata_method: ProRataMethod = Field(
        ProRataMethod.CALENDAR_DAYS,
        description="Default method for pro-rata apportionment of M&A emissions",
    )
    auto_base_year_recalculation: bool = Field(
        True,
        description="Automatically trigger base year recalculation on structural changes",
    )
    base_year_materiality_trigger_pct: Decimal = Field(
        Decimal("5.0"),
        ge=Decimal("1.0"), le=Decimal("20.0"),
        description="M&A impact threshold (%) that triggers base year recalculation",
    )
    require_mna_approval: bool = Field(
        True,
        description="Require management approval for M&A boundary adjustments",
    )
    track_mna_history: bool = Field(
        True,
        description="Maintain full history of M&A events and boundary changes",
    )
    event_types_enabled: List[MnAEventType] = Field(
        default_factory=lambda: [
            MnAEventType.ACQUISITION,
            MnAEventType.DIVESTITURE,
            MnAEventType.MERGER,
            MnAEventType.DEMERGER,
            MnAEventType.JV_FORMATION,
            MnAEventType.JV_DISSOLUTION,
        ],
        description="M&A event types to process",
    )
    lookback_years: int = Field(
        5, ge=1, le=10,
        description="Number of years to look back for historical M&A impact analysis",
    )


class AdjustmentConfig(BaseModel):
    """Configuration for adjustment and restatement processing."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    adjustment_types_enabled: List[AdjustmentType] = Field(
        default_factory=lambda: [
            AdjustmentType.METHODOLOGY_CHANGE,
            AdjustmentType.ERROR_CORRECTION,
            AdjustmentType.SCOPE_RECLASSIFICATION,
            AdjustmentType.TIMING,
            AdjustmentType.LATE_SUBMISSION,
        ],
        description="Types of adjustments allowed",
    )
    require_justification: bool = Field(
        True,
        description="Require written justification for all adjustments",
    )
    require_approval: bool = Field(
        True,
        description="Require management approval for adjustments",
    )
    materiality_threshold_pct: Decimal = Field(
        Decimal("1.0"),
        ge=Decimal("0.1"), le=Decimal("10.0"),
        description="Threshold (%) above which adjustments require escalated approval",
    )
    restatement_window_years: int = Field(
        3, ge=1, le=10,
        description="Number of prior years that can be restated",
    )
    late_submission_grace_days: int = Field(
        30, ge=0, le=180,
        description="Grace period (days) for late submissions before estimation kicks in",
    )
    track_adjustment_history: bool = Field(
        True,
        description="Maintain full history of all adjustments and restatements",
    )


class GroupReportingConfig(BaseModel):
    """Configuration for consolidated group reporting."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    reporting_frameworks: List[ReportingFramework] = Field(
        default_factory=lambda: [
            ReportingFramework.CSRD_ESRS_E1,
            ReportingFramework.CDP,
            ReportingFramework.GRI_305,
        ],
        description="Target reporting frameworks for consolidated output",
    )
    default_format: ExportFormat = Field(
        ExportFormat.HTML,
        description="Default output format for consolidated reports",
    )
    include_entity_breakdown: bool = Field(
        True,
        description="Include entity-level breakdown in consolidated reports",
    )
    include_scope_breakdown: bool = Field(
        True,
        description="Include scope-level breakdown in consolidated reports",
    )
    include_trend_analysis: bool = Field(
        True,
        description="Include multi-year trend analysis",
    )
    trend_years: int = Field(
        3, ge=1, le=10,
        description="Number of years to include in trend analysis",
    )
    include_elimination_log: bool = Field(
        True,
        description="Include elimination details in reports",
    )
    include_mna_impact: bool = Field(
        True,
        description="Include M&A impact analysis in reports",
    )
    decimal_places_display: int = Field(
        2, ge=0, le=6,
        description="Decimal places for display in reports",
    )
    language: str = Field("en", description="Report language (ISO 639-1)")
    branding: Dict[str, str] = Field(
        default_factory=lambda: {
            "logo_url": "",
            "primary_colour": "#1B5E20",
            "company_name": "",
        },
        description="Report branding configuration",
    )


class AuditConfig(BaseModel):
    """Configuration for consolidation audit trail and assurance support."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    audit_trail_enabled: bool = Field(
        True,
        description="Enable comprehensive audit trail for all consolidation operations",
    )
    reconciliation_checks_enabled: bool = Field(
        True,
        description="Run automated reconciliation checks on consolidated totals",
    )
    variance_threshold_pct: Decimal = Field(
        Decimal("5.0"),
        ge=Decimal("1.0"), le=Decimal("25.0"),
        description="Variance threshold (%) for flagging year-over-year changes",
    )
    require_sign_off: bool = Field(
        True,
        description="Require management sign-off on consolidated results",
    )
    sign_off_levels: int = Field(
        2, ge=1, le=5,
        description="Number of approval levels for sign-off",
    )
    evidence_retention_years: int = Field(
        7, ge=3, le=15,
        description="Number of years to retain audit evidence",
    )
    assurance_ready: bool = Field(
        True,
        description="Generate assurance-ready documentation packages",
    )
    data_quality_scoring: bool = Field(
        True,
        description="Score data quality per entity for consolidated assessment",
    )


class SecurityConfig(BaseModel):
    """Configuration for access control and data protection."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    permissions: List[str] = Field(
        default_factory=lambda: [
            "consolidation_admin", "entity_admin", "boundary_manager",
            "data_submitter", "reviewer", "approver",
            "report_viewer", "auditor", "viewer", "admin",
        ],
        description="Available RBAC roles for consolidation management",
    )
    rls_enabled: bool = Field(
        True,
        description="Enable row-level security for entity-specific data access",
    )
    audit_enabled: bool = Field(
        True,
        description="Enable audit trail for all consolidation operations",
    )
    entity_level_access_control: bool = Field(
        True,
        description="Restrict users to their assigned entities only",
    )
    cross_entity_data_sharing: bool = Field(
        False,
        description="Allow data visibility across entity boundaries (for group admins only)",
    )


class PerformanceConfig(BaseModel):
    """Configuration for computational performance tuning."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    max_concurrent_entities: int = Field(
        50, ge=5, le=500,
        description="Maximum number of entities to process concurrently",
    )
    batch_size: int = Field(
        100, ge=10, le=5000,
        description="Batch size for bulk entity data processing",
    )
    cache_ttl_seconds: int = Field(
        3600, ge=60, le=86400,
        description="Cache TTL in seconds for ownership chain lookups",
    )
    lazy_load_entity_data: bool = Field(
        True,
        description="Lazy-load entity data only when accessed",
    )
    parallel_consolidation: bool = Field(
        True,
        description="Run consolidation across entities in parallel",
    )
    ownership_chain_cache: bool = Field(
        True,
        description="Cache resolved ownership chains to avoid recomputation",
    )


class IntegrationConfig(BaseModel):
    """Configuration for integration with other GreenLang components."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    mrv_agents_count: int = Field(
        30, ge=0,
        description="Number of MRV agents to route to for per-entity calculations",
    )
    data_agents_count: int = Field(
        20, ge=0,
        description="Number of DATA agents to route to for entity data ingestion",
    )
    foundation_agents: List[str] = Field(
        default_factory=lambda: [
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
            "PACK-043",
            "PACK-044",
            "PACK-045",
            "PACK-046",
            "PACK-047",
            "PACK-048",
            "PACK-049",
        ],
        description="Pack dependencies for consolidation",
    )
    erp_connector_enabled: bool = Field(
        False,
        description="Enable direct ERP connector for entity financial data",
    )


class AlertConfig(BaseModel):
    """Configuration for consolidation management alerting."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    alert_types_enabled: List[AlertType] = Field(
        default_factory=lambda: [
            AlertType.DEADLINE,
            AlertType.COMPLETENESS,
            AlertType.VARIANCE,
            AlertType.BOUNDARY_CHANGE,
            AlertType.MNA_EVENT,
            AlertType.APPROVAL,
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
        description="Send daily digest of all consolidation alerts",
    )
    quiet_hours_enabled: bool = Field(
        False,
        description="Suppress non-critical alerts outside business hours",
    )
    mna_event_immediate: bool = Field(
        True,
        description="Send immediate alert for M&A events regardless of quiet hours",
    )


class MigrationConfig(BaseModel):
    """Configuration for database schema migration."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    schema_name: str = Field(
        "ghg_consolidation",
        description="Database schema name for consolidation tables",
    )
    table_prefix: str = Field(
        "gl_cons_",
        description="Table name prefix for consolidation tables",
    )
    migration_start: str = Field(
        "V416",
        description="First migration version for PACK-050",
    )
    migration_end: str = Field(
        "V425",
        description="Last migration version for PACK-050",
    )


# =============================================================================
# Main Configuration Model
# =============================================================================


class ConsolidationPackConfig(BaseModel):
    """
    Top-level configuration for PACK-050 GHG Consolidation.

    Combines all sub-configurations required for entity registry,
    ownership chain management, boundary determination, equity share
    consolidation, control approach consolidation, inter-company
    elimination, M&A event handling, adjustment processing, group
    reporting, and audit trail generation.
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
        description="Primary country of the parent entity (ISO 3166-1 alpha-2)",
    )
    currency: str = Field(
        "EUR",
        description="Reporting currency (ISO 4217) for revenue-based metrics",
    )
    total_entities: Optional[int] = Field(
        None, ge=1,
        description="Total number of entities in the group (auto-detected if None)",
    )
    scopes_in_scope: List[ScopeCategory] = Field(
        default_factory=lambda: [
            ScopeCategory.SCOPE_1,
            ScopeCategory.SCOPE_2_LOCATION,
            ScopeCategory.SCOPE_2_MARKET,
        ],
        description="Emission scopes included in consolidation",
    )

    entity_registry: EntityRegistryConfig = Field(default_factory=EntityRegistryConfig)
    ownership: OwnershipConfig = Field(default_factory=OwnershipConfig)
    boundary: BoundaryConfig = Field(default_factory=BoundaryConfig)
    equity_share: EquityShareConfig = Field(default_factory=EquityShareConfig)
    control_approach: ControlApproachConfig = Field(default_factory=ControlApproachConfig)
    elimination: EliminationConfig = Field(default_factory=EliminationConfig)
    mna: MnAConfig = Field(default_factory=MnAConfig)
    adjustment: AdjustmentConfig = Field(default_factory=AdjustmentConfig)
    reporting: GroupReportingConfig = Field(default_factory=GroupReportingConfig)
    audit: AuditConfig = Field(default_factory=AuditConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)
    integration: IntegrationConfig = Field(default_factory=IntegrationConfig)
    alerts: AlertConfig = Field(default_factory=AlertConfig)
    migration: MigrationConfig = Field(default_factory=MigrationConfig)

    @model_validator(mode="after")
    def validate_base_year_consistency(self) -> ConsolidationPackConfig:
        """Ensure base year is before reporting year."""
        if self.base_year > self.reporting_year:
            raise ValueError(
                f"base_year ({self.base_year}) cannot be after "
                f"reporting_year ({self.reporting_year})"
            )
        return self

    @model_validator(mode="after")
    def validate_consolidation_alignment(self) -> ConsolidationPackConfig:
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
    def validate_dual_approach(self) -> ConsolidationPackConfig:
        """Validate dual approach configuration if enabled."""
        if self.boundary.dual_approach_enabled:
            if self.boundary.secondary_approach is None:
                raise ValueError(
                    "dual_approach_enabled is True but secondary_approach is not set"
                )
            if self.boundary.secondary_approach == self.consolidation_approach:
                raise ValueError(
                    "secondary_approach must differ from primary consolidation_approach"
                )
        return self

    @model_validator(mode="after")
    def validate_entity_count_vs_max(self) -> ConsolidationPackConfig:
        """Warn if total entities exceeds max limit."""
        if self.total_entities is not None:
            if self.total_entities > self.entity_registry.max_entities:
                raise ValueError(
                    f"total_entities ({self.total_entities}) exceeds "
                    f"entity_registry.max_entities ({self.entity_registry.max_entities})"
                )
        return self

    @model_validator(mode="after")
    def validate_mna_base_year_alignment(self) -> ConsolidationPackConfig:
        """Ensure M&A base year recalculation is consistent with adjustment config."""
        if self.mna.auto_base_year_recalculation:
            if self.adjustment.restatement_window_years < 1:
                logger.warning(
                    "auto_base_year_recalculation enabled but restatement_window_years "
                    "is less than 1. Base year restatements may not be possible."
                )
        return self


# =============================================================================
# Pack Configuration Wrapper
# =============================================================================


class PackConfig(BaseModel):
    """
    Top-level wrapper for PACK-050 configuration.

    Provides factory methods for loading from presets, YAML files,
    environment overrides, and runtime merges. Includes SHA-256
    config hashing for provenance tracking.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    pack: ConsolidationPackConfig = Field(default_factory=ConsolidationPackConfig)
    preset_name: Optional[str] = Field(None, description="Name of the loaded preset")
    config_version: str = Field("1.0.0", description="Configuration schema version")
    pack_id: str = Field("PACK-050-consolidation", description="Unique pack identifier")

    @classmethod
    def from_preset(cls, preset_name: str, overrides: Optional[Dict[str, Any]] = None) -> PackConfig:
        """
        Load configuration from a named organisation-type preset.

        Args:
            preset_name: Key from AVAILABLE_PRESETS (e.g., 'corporate_conglomerate').
            overrides: Optional dict of overrides applied after preset load.

        Returns:
            Fully initialised PackConfig.

        Raises:
            ValueError: If preset_name is not recognised.
            FileNotFoundError: If preset YAML file is missing.
        """
        if preset_name not in AVAILABLE_PRESETS:
            raise ValueError(
                f"Unknown preset: {preset_name}. "
                f"Available: {sorted(AVAILABLE_PRESETS.keys())}"
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
        pack_config = ConsolidationPackConfig(**preset_data)
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
        pack_config = ConsolidationPackConfig(**config_data)
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
        pack_config = ConsolidationPackConfig(**merged)
        return cls(
            pack=pack_config,
            preset_name=base.preset_name,
            config_version=base.config_version,
        )

    @staticmethod
    def _load_env_overrides() -> Dict[str, Any]:
        """
        Load configuration overrides from environment variables.

        Environment variables prefixed with CONSOLIDATION_PACK_ are parsed.
        Double underscores denote nested keys.
        Example: CONSOLIDATION_PACK_BOUNDARY__CONSOLIDATION_APPROACH=EQUITY_SHARE
        """
        overrides: Dict[str, Any] = {}
        prefix = "CONSOLIDATION_PACK_"
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
            List of scope strings (e.g., ['SCOPE_1', 'SCOPE_2_LOCATION']).
        """
        return [s.value for s in self.pack.scopes_in_scope]

    def get_active_frameworks(self) -> List[str]:
        """
        Return the list of active reporting frameworks.

        Returns:
            List of framework strings (e.g., ['CSRD_ESRS_E1', 'CDP']).
        """
        return [f.value for f in self.pack.reporting.reporting_frameworks]


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


def validate_config(config: ConsolidationPackConfig) -> List[str]:
    """
    Validate configuration for domain-specific consistency.

    Args:
        config: The consolidation pack configuration to validate.

    Returns:
        List of warning strings. Empty list indicates no issues found.
    """
    warnings: List[str] = []

    # Company name check
    if not config.company_name:
        warnings.append("No company_name configured.")

    # Entity count check
    if config.total_entities is None:
        warnings.append(
            "total_entities not set. Group size will be auto-detected "
            "from registered entities."
        )

    # Consolidation approach guidance
    if config.consolidation_approach == ConsolidationApproach.EQUITY_SHARE:
        warnings.append(
            "Equity share approach requires equity percentages for all "
            "entities. Ensure ownership data is complete."
        )

    # Equity share approach-specific checks
    if config.consolidation_approach == ConsolidationApproach.EQUITY_SHARE:
        if not config.equity_share.include_associates:
            warnings.append(
                "Equity share approach selected but associates are excluded. "
                "Consider enabling include_associates for completeness."
            )

    # Control approach-specific checks
    if config.consolidation_approach == ConsolidationApproach.OPERATIONAL_CONTROL:
        if not config.control_approach.franchise_inclusion:
            warnings.append(
                "Operational control approach selected but franchise inclusion "
                "is disabled. Review whether franchises should be in boundary."
            )

    # Boundary materiality check
    if config.boundary.materiality_threshold_pct > Decimal("0.10"):
        warnings.append(
            f"Materiality threshold ({config.boundary.materiality_threshold_pct}) "
            "is above 10%. This may exclude significant emission sources."
        )

    # Dual approach consistency
    if config.boundary.dual_approach_enabled and config.boundary.secondary_approach is None:
        warnings.append(
            "dual_approach_enabled is True but no secondary_approach specified."
        )

    # Elimination configuration
    if config.elimination.elimination_enabled:
        if not config.elimination.require_matching_entries:
            warnings.append(
                "Elimination enabled without requiring matching entries. "
                "One-sided eliminations may reduce audit quality."
            )

    # M&A configuration
    if config.mna.mna_tracking_enabled:
        if not config.mna.auto_base_year_recalculation:
            warnings.append(
                "M&A tracking enabled but auto base year recalculation disabled. "
                "Structural changes may not be reflected in base year."
            )

    # Adjustment restatement window
    if config.adjustment.restatement_window_years < 2:
        warnings.append(
            f"Restatement window ({config.adjustment.restatement_window_years} years) "
            "is short. Consider extending to cover typical audit cycles."
        )

    # Audit configuration
    if config.audit.require_sign_off and config.audit.sign_off_levels < 2:
        warnings.append(
            "Sign-off required with only 1 level. Consider at least 2 levels "
            "for adequate segregation of duties."
        )

    # Reporting framework alignment
    frameworks = [f.value for f in config.reporting.reporting_frameworks]
    if "CSRD_ESRS_E1" in frameworks:
        if ScopeCategory.SCOPE_3 not in config.scopes_in_scope:
            warnings.append(
                "CSRD/ESRS E1 framework selected but Scope 3 not in scope. "
                "ESRS E1 requires Scope 3 disclosure."
            )

    if "SEC_CLIMATE" in frameworks:
        if config.consolidation_approach != ConsolidationApproach.FINANCIAL_CONTROL:
            warnings.append(
                "SEC Climate framework selected but consolidation approach is not "
                "financial control. SEC rules align with financial reporting entity."
            )

    # Security configuration
    if config.security.rls_enabled and not config.security.audit_enabled:
        warnings.append(
            "Row-level security enabled but audit trail disabled. "
            "Consider enabling audit for access tracking."
        )

    # Performance check for large groups
    if config.total_entities is not None and config.total_entities > 500:
        if config.performance.max_concurrent_entities < 50:
            warnings.append(
                f"Large group ({config.total_entities} entities) with low "
                f"concurrency ({config.performance.max_concurrent_entities}). "
                "Consider increasing max_concurrent_entities for performance."
            )

    # Ownership chain depth vs hierarchy depth
    if config.ownership.max_chain_depth > config.entity_registry.hierarchy_depth_limit:
        warnings.append(
            f"ownership.max_chain_depth ({config.ownership.max_chain_depth}) exceeds "
            f"entity_registry.hierarchy_depth_limit ({config.entity_registry.hierarchy_depth_limit}). "
            "Chain resolution may exceed hierarchy limits."
        )

    return warnings


def get_default_config(
    approach: ConsolidationApproach = ConsolidationApproach.OPERATIONAL_CONTROL,
) -> ConsolidationPackConfig:
    """
    Create a default configuration for the given consolidation approach.

    Args:
        approach: GHG Protocol consolidation approach.

    Returns:
        Default ConsolidationPackConfig for the approach.
    """
    return ConsolidationPackConfig(consolidation_approach=approach)


def list_available_presets() -> Dict[str, str]:
    """
    Return a copy of all available preset names and descriptions.

    Returns:
        Dict mapping preset name to human-readable description.
    """
    return AVAILABLE_PRESETS.copy()


def get_entity_type_defaults(entity_type: str) -> Optional[Dict[str, Any]]:
    """
    Return default characteristics for an entity type.

    Args:
        entity_type: EntityType value string.

    Returns:
        Dict of entity characteristics, or None if not found.
    """
    return DEFAULT_ENTITY_TYPES.get(entity_type)


def get_ownership_threshold(ownership_type: str) -> Optional[Dict[str, Any]]:
    """
    Return ownership classification thresholds.

    Args:
        ownership_type: OwnershipType value string.

    Returns:
        Dict of threshold data, or None if not found.
    """
    return DEFAULT_OWNERSHIP_THRESHOLDS.get(ownership_type)


def get_elimination_rules(elimination_type: str) -> Optional[Dict[str, Any]]:
    """
    Return elimination rules for an inter-company transfer type.

    Args:
        elimination_type: EliminationType value string.

    Returns:
        Dict of elimination rules, or None if not found.
    """
    return DEFAULT_ELIMINATION_RULES.get(elimination_type)


def get_mna_rules(event_type: str) -> Optional[Dict[str, Any]]:
    """
    Return M&A processing rules for an event type.

    Args:
        event_type: MnAEventType value string.

    Returns:
        Dict of M&A rules, or None if not found.
    """
    return DEFAULT_MNA_RULES.get(event_type)


def get_framework_requirements(framework: str) -> Optional[Dict[str, Any]]:
    """
    Return regulatory requirements for a reporting framework.

    Args:
        framework: ReportingFramework value string.

    Returns:
        Dict of framework requirements, or None if not found.
    """
    return DEFAULT_FRAMEWORK_REQUIREMENTS.get(framework)
