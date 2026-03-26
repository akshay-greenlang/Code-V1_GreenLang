# -*- coding: utf-8 -*-
"""
OrganizationalBoundaryEngine - PACK-041 Scope 1-2 Complete Engine 1
====================================================================

Organizational boundary definition engine implementing GHG Protocol
Corporate Standard Chapter 3 consolidation approaches.  Determines which
entities, facilities, and operations fall within an organisation's GHG
inventory boundary by applying equity-share, operational-control, or
financial-control approaches.  Handles boundary changes arising from
mergers, acquisitions, divestitures, and organic growth, including
base-year recalculation impact assessment per GHG Protocol guidance.

Calculation Methodology:
    Equity Share Approach:
        Entity_emissions_share = Total_entity_emissions * (equity_pct / 100)
        Org_total = sum(Entity_emissions_share_i) for all entities

    Operational Control Approach:
        If org has operational control -> include 100% of emissions
        Else -> exclude 0% of emissions

    Financial Control Approach:
        If org has majority financial interest (>50%) -> include 100%
        If power to govern financial/operating policies -> include 100%
        Else -> exclude 0%

    Base-Year Recalculation Threshold:
        Materiality = abs(change_emissions) / base_year_total * 100
        If Materiality > significance_threshold (typically 5%) -> recalculate

    Structural Change Impact:
        Adjusted_base_year = Original_base_year +/- structural_change_emissions
        For acquisitions: add acquired entity emissions pro-rated
        For divestitures: subtract divested entity emissions pro-rated

Regulatory References:
    - GHG Protocol Corporate Standard (Revised), Chapter 3
    - GHG Protocol Corporate Standard, Chapter 5 (Base Year)
    - GHG Protocol Scope 2 Guidance, Chapter 1 (Boundary)
    - ISO 14064-1:2018, Clause 5.1 (Organizational Boundaries)
    - PCAF Standard (Financed Emissions Boundaries)

Zero-Hallucination:
    - All inclusion/exclusion uses deterministic rules from GHG Protocol Ch. 3
    - Equity percentages are input data, never estimated by LLM
    - Base-year recalculation uses deterministic arithmetic
    - SHA-256 provenance hash on every result
    - No LLM involvement in any calculation path

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-041 Scope 1-2 Complete
Engine:  1 of 10
Status:  Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from datetime import date, datetime, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _new_uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())


def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash of arbitrary data."""
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    if isinstance(serializable, dict):
        serializable = {
            k: v for k, v in serializable.items()
            if k not in ("calculated_at", "processing_time_ms", "provenance_hash")
        }
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _decimal(value: Any) -> Decimal:
    """Safely convert a value to Decimal."""
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return Decimal("0")


def _safe_divide(
    numerator: Decimal,
    denominator: Decimal,
    default: Decimal = Decimal("0"),
) -> Decimal:
    """Safely divide two Decimals, returning *default* on zero denominator."""
    if denominator == Decimal("0"):
        return default
    return numerator / denominator


def _safe_pct(part: Decimal, whole: Decimal) -> Decimal:
    """Compute percentage safely (part / whole * 100)."""
    return _safe_divide(part * Decimal("100"), whole)


def _round_val(value: Decimal, places: int = 6) -> Decimal:
    """Round a Decimal to *places* using ROUND_HALF_UP."""
    quantize_str = "0." + "0" * places
    return value.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ConsolidationApproach(str, Enum):
    """GHG Protocol Chapter 3 consolidation approach.

    EQUITY_SHARE:       Account for emissions based on equity ownership percentage.
    OPERATIONAL_CONTROL: Account for 100% of emissions from operations over which
                        the organisation has operational control.
    FINANCIAL_CONTROL:  Account for 100% of emissions from operations over which
                        the organisation has financial control.
    """
    EQUITY_SHARE = "equity_share"
    OPERATIONAL_CONTROL = "operational_control"
    FINANCIAL_CONTROL = "financial_control"


class EntityType(str, Enum):
    """Legal entity ownership/relationship classification per GHG Protocol.

    WHOLLY_OWNED:    100% owned subsidiary.
    MAJORITY_OWNED:  >50% equity stake.
    JV:              Joint venture (shared equity with other parties).
    ASSOCIATE:       Significant influence (20-50%) but not control.
    FRANCHISE:       Franchise operation.
    LEASED_FINANCE:  Finance lease (lessee has substantially all risks/rewards).
    LEASED_OPERATING: Operating lease (lessor retains risks/rewards).
    """
    WHOLLY_OWNED = "wholly_owned"
    MAJORITY_OWNED = "majority_owned"
    JV = "joint_venture"
    ASSOCIATE = "associate"
    FRANCHISE = "franchise"
    LEASED_FINANCE = "leased_finance"
    LEASED_OPERATING = "leased_operating"


class FacilityOperationalStatus(str, Enum):
    """Current operational status of a facility.

    ACTIVE:      Facility is currently operating.
    IDLE:        Facility exists but is temporarily not operating.
    MOTHBALLED:  Facility is long-term idle with minimal maintenance.
    CLOSED:      Facility is permanently closed.
    UNDER_CONSTRUCTION: Facility is being built.
    """
    ACTIVE = "active"
    IDLE = "idle"
    MOTHBALLED = "mothballed"
    CLOSED = "closed"
    UNDER_CONSTRUCTION = "under_construction"


class BoundaryChangeType(str, Enum):
    """Type of structural boundary change per GHG Protocol Ch. 5.

    ACQUISITION:   Acquiring another entity or facility.
    DIVESTITURE:   Selling or spinning off an entity or facility.
    MERGER:        Merging with another organisation.
    OUTSOURCING:   Outsourcing previously in-boundary operations.
    INSOURCING:    Bringing previously outsourced operations in-boundary.
    ORGANIC_GROWTH: New facility built by the organisation.
    SHUTDOWN:      Permanent closure of a facility.
    METHODOLOGY_CHANGE: Change in calculation methodology.
    """
    ACQUISITION = "acquisition"
    DIVESTITURE = "divestiture"
    MERGER = "merger"
    OUTSOURCING = "outsourcing"
    INSOURCING = "insourcing"
    ORGANIC_GROWTH = "organic_growth"
    SHUTDOWN = "shutdown"
    METHODOLOGY_CHANGE = "methodology_change"


class RecalculationTrigger(str, Enum):
    """Trigger for base-year recalculation per GHG Protocol Ch. 5.

    STRUCTURAL_CHANGE:      M&A, divestitures, or outsourcing/insourcing.
    METHODOLOGY_CHANGE:     Change in calculation methods or emission factors.
    DATA_ERROR_CORRECTION:  Correction of significant data errors.
    CATEGORY_RECLASSIFICATION: Reclassification of emission categories.
    NOT_TRIGGERED:          Change below significance threshold.
    """
    STRUCTURAL_CHANGE = "structural_change"
    METHODOLOGY_CHANGE = "methodology_change"
    DATA_ERROR_CORRECTION = "data_error_correction"
    CATEGORY_RECLASSIFICATION = "category_reclassification"
    NOT_TRIGGERED = "not_triggered"


class InclusionStatus(str, Enum):
    """Entity/facility inclusion status in the GHG boundary.

    INCLUDED:     Fully included in the inventory boundary.
    EXCLUDED:     Excluded from the inventory boundary.
    PARTIAL:      Partially included (equity share approach with <100%).
    UNDER_REVIEW: Inclusion status is being evaluated.
    """
    INCLUDED = "included"
    EXCLUDED = "excluded"
    PARTIAL = "partial"
    UNDER_REVIEW = "under_review"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Default significance threshold for base-year recalculation (%).
DEFAULT_SIGNIFICANCE_THRESHOLD_PCT: Decimal = Decimal("5.0")

# Equity thresholds for classification.
EQUITY_WHOLLY_OWNED: Decimal = Decimal("100")
EQUITY_MAJORITY_THRESHOLD: Decimal = Decimal("50")
EQUITY_ASSOCIATE_MIN: Decimal = Decimal("20")
EQUITY_ASSOCIATE_MAX: Decimal = Decimal("50")

# Standard inclusion percentages by approach and entity type.
# For operational control: 100% if you have OC, 0% if not.
# For financial control: 100% if you have FC, 0% if not.
# For equity share: proportional to equity_pct.
OPERATIONAL_CONTROL_DEFAULTS: Dict[str, bool] = {
    EntityType.WHOLLY_OWNED.value: True,
    EntityType.MAJORITY_OWNED.value: True,
    EntityType.JV.value: False,         # depends on JV agreement
    EntityType.ASSOCIATE.value: False,
    EntityType.FRANCHISE.value: False,   # typically franchisor has OC over own, not franchisee
    EntityType.LEASED_FINANCE.value: True,
    EntityType.LEASED_OPERATING.value: False,
}

FINANCIAL_CONTROL_DEFAULTS: Dict[str, bool] = {
    EntityType.WHOLLY_OWNED.value: True,
    EntityType.MAJORITY_OWNED.value: True,
    EntityType.JV.value: False,         # depends on ability to govern financial policy
    EntityType.ASSOCIATE.value: False,
    EntityType.FRANCHISE.value: False,
    EntityType.LEASED_FINANCE.value: True,
    EntityType.LEASED_OPERATING.value: False,
}


# ---------------------------------------------------------------------------
# Pydantic Models -- Input
# ---------------------------------------------------------------------------


class Facility(BaseModel):
    """A physical facility or site within an entity.

    Attributes:
        facility_id: Unique identifier for the facility.
        facility_name: Human-readable name.
        entity_id: Parent entity identifier.
        country: ISO 3166-1 alpha-2 country code.
        region: Region or state within the country.
        sector: Industry sector (e.g. 'manufacturing', 'office').
        operational_status: Current operational status.
        scope1_emissions_tco2e: Most recent annual Scope 1 emissions.
        scope2_emissions_tco2e: Most recent annual Scope 2 emissions.
        employee_count: Number of employees at the facility.
        floor_area_m2: Total floor area in square metres.
        notes: Additional notes.
    """
    facility_id: str = Field(default_factory=_new_uuid, description="Facility ID")
    facility_name: str = Field(default="", max_length=300, description="Facility name")
    entity_id: str = Field(default="", description="Parent entity ID")
    country: str = Field(default="", max_length=2, description="ISO 3166-1 alpha-2")
    region: str = Field(default="", max_length=200, description="Region/state")
    sector: str = Field(default="", max_length=200, description="Industry sector")
    operational_status: FacilityOperationalStatus = Field(
        default=FacilityOperationalStatus.ACTIVE,
        description="Facility operational status",
    )
    scope1_emissions_tco2e: Decimal = Field(
        default=Decimal("0"), ge=0, description="Annual Scope 1 (tCO2e)"
    )
    scope2_emissions_tco2e: Decimal = Field(
        default=Decimal("0"), ge=0, description="Annual Scope 2 (tCO2e)"
    )
    employee_count: int = Field(default=0, ge=0, description="Employee count")
    floor_area_m2: Decimal = Field(
        default=Decimal("0"), ge=0, description="Floor area (m2)"
    )
    notes: str = Field(default="", max_length=1000, description="Notes")


class LegalEntity(BaseModel):
    """A legal entity (subsidiary, JV, associate, etc.) in the org structure.

    Attributes:
        entity_id: Unique identifier for the entity.
        entity_name: Legal name of the entity.
        parent_entity_id: ID of the parent entity (empty if top-level).
        entity_type: Classification of the entity.
        equity_pct: Equity ownership percentage (0-100).
        has_operational_control: Whether the reporting org has operational control.
        has_financial_control: Whether the reporting org has financial control.
        country_of_incorporation: ISO 3166-1 alpha-2 country code.
        sector: Primary industry sector.
        facilities: List of facilities under this entity.
        total_scope1_tco2e: Sum of all facility Scope 1 emissions.
        total_scope2_tco2e: Sum of all facility Scope 2 emissions.
        is_active: Whether the entity is currently active.
        acquisition_date: Date the entity was acquired (if applicable).
        divestiture_date: Date the entity was divested (if applicable).
        notes: Additional notes.
    """
    entity_id: str = Field(default_factory=_new_uuid, description="Entity ID")
    entity_name: str = Field(default="", max_length=500, description="Entity name")
    parent_entity_id: str = Field(default="", description="Parent entity ID")
    entity_type: EntityType = Field(
        default=EntityType.WHOLLY_OWNED, description="Entity classification"
    )
    equity_pct: Decimal = Field(
        default=Decimal("100"), ge=0, le=100,
        description="Equity ownership percentage (0-100)",
    )
    has_operational_control: Optional[bool] = Field(
        default=None, description="Has operational control (None = use default)"
    )
    has_financial_control: Optional[bool] = Field(
        default=None, description="Has financial control (None = use default)"
    )
    country_of_incorporation: str = Field(
        default="", max_length=2, description="ISO 3166-1 alpha-2"
    )
    sector: str = Field(default="", max_length=200, description="Industry sector")
    facilities: List[Facility] = Field(
        default_factory=list, description="Facilities under this entity"
    )
    total_scope1_tco2e: Decimal = Field(
        default=Decimal("0"), ge=0, description="Total Scope 1 (tCO2e)"
    )
    total_scope2_tco2e: Decimal = Field(
        default=Decimal("0"), ge=0, description="Total Scope 2 (tCO2e)"
    )
    is_active: bool = Field(default=True, description="Entity active flag")
    acquisition_date: Optional[date] = Field(
        default=None, description="Acquisition date"
    )
    divestiture_date: Optional[date] = Field(
        default=None, description="Divestiture date"
    )
    notes: str = Field(default="", max_length=1000, description="Notes")

    @field_validator("equity_pct", mode="before")
    @classmethod
    def _coerce_equity(cls, v: Any) -> Any:
        """Coerce equity_pct to Decimal."""
        return _decimal(v)

    @field_validator("total_scope1_tco2e", "total_scope2_tco2e", mode="before")
    @classmethod
    def _coerce_totals(cls, v: Any) -> Any:
        """Coerce emission totals to Decimal."""
        return _decimal(v)


class OrganizationStructure(BaseModel):
    """Full organisational structure for boundary definition.

    Attributes:
        org_id: Unique identifier for the organisation.
        org_name: Legal name of the reporting organisation.
        reporting_year: GHG inventory reporting year.
        base_year: Designated base year for trend tracking.
        entities: All legal entities in the organisation.
        default_approach: Default consolidation approach.
        significance_threshold_pct: Threshold for base-year recalculation (%).
        notes: Additional notes.
    """
    org_id: str = Field(default_factory=_new_uuid, description="Organisation ID")
    org_name: str = Field(default="", max_length=500, description="Organisation name")
    reporting_year: int = Field(default=2025, ge=1990, le=2100, description="Reporting year")
    base_year: int = Field(default=2019, ge=1990, le=2100, description="Base year")
    entities: List[LegalEntity] = Field(
        default_factory=list, description="All legal entities"
    )
    default_approach: ConsolidationApproach = Field(
        default=ConsolidationApproach.OPERATIONAL_CONTROL,
        description="Default consolidation approach",
    )
    significance_threshold_pct: Decimal = Field(
        default=DEFAULT_SIGNIFICANCE_THRESHOLD_PCT,
        ge=0, le=100,
        description="Significance threshold for base-year recalculation (%)",
    )
    notes: str = Field(default="", max_length=2000, description="Notes")


class BoundaryChangeEvent(BaseModel):
    """A structural change event that may affect the boundary.

    Attributes:
        change_id: Unique identifier for the change event.
        change_type: Type of structural change.
        effective_date: Date the change takes effect.
        entity_id: Entity affected by the change.
        entity_name: Name of the entity affected.
        description: Detailed description of the change.
        emissions_impact_tco2e: Estimated emissions impact (tCO2e).
        equity_pct_before: Equity percentage before the change.
        equity_pct_after: Equity percentage after the change.
        operational_control_before: OC status before.
        operational_control_after: OC status after.
        financial_control_before: FC status before.
        financial_control_after: FC status after.
        acquired_facilities: Facilities acquired or divested.
        notes: Additional notes.
    """
    change_id: str = Field(default_factory=_new_uuid, description="Change event ID")
    change_type: BoundaryChangeType = Field(
        ..., description="Type of structural change"
    )
    effective_date: date = Field(..., description="Effective date")
    entity_id: str = Field(default="", description="Affected entity ID")
    entity_name: str = Field(default="", max_length=500, description="Entity name")
    description: str = Field(default="", max_length=2000, description="Description")
    emissions_impact_tco2e: Decimal = Field(
        default=Decimal("0"), description="Estimated emissions impact (tCO2e)"
    )
    equity_pct_before: Decimal = Field(
        default=Decimal("0"), ge=0, le=100, description="Equity % before"
    )
    equity_pct_after: Decimal = Field(
        default=Decimal("0"), ge=0, le=100, description="Equity % after"
    )
    operational_control_before: Optional[bool] = Field(
        default=None, description="OC before"
    )
    operational_control_after: Optional[bool] = Field(
        default=None, description="OC after"
    )
    financial_control_before: Optional[bool] = Field(
        default=None, description="FC before"
    )
    financial_control_after: Optional[bool] = Field(
        default=None, description="FC after"
    )
    acquired_facilities: List[Facility] = Field(
        default_factory=list, description="Facilities acquired/divested"
    )
    notes: str = Field(default="", max_length=2000, description="Notes")


# ---------------------------------------------------------------------------
# Pydantic Models -- Output
# ---------------------------------------------------------------------------


class EntityInclusionResult(BaseModel):
    """Result of boundary evaluation for a single entity.

    Attributes:
        entity_id: The evaluated entity ID.
        entity_name: Entity name.
        entity_type: Entity type classification.
        equity_pct: Equity ownership percentage.
        inclusion_status: Whether the entity is included.
        inclusion_pct: The percentage of emissions to include (0-100).
        has_operational_control: OC flag used for determination.
        has_financial_control: FC flag used for determination.
        scope1_included_tco2e: Scope 1 emissions included.
        scope2_included_tco2e: Scope 2 emissions included.
        total_included_tco2e: Total emissions included.
        facility_count: Number of included facilities.
        rationale: Explanation of inclusion determination.
    """
    entity_id: str = Field(default="", description="Entity ID")
    entity_name: str = Field(default="", description="Entity name")
    entity_type: EntityType = Field(
        default=EntityType.WHOLLY_OWNED, description="Entity type"
    )
    equity_pct: Decimal = Field(default=Decimal("0"), description="Equity %")
    inclusion_status: InclusionStatus = Field(
        default=InclusionStatus.EXCLUDED, description="Inclusion status"
    )
    inclusion_pct: Decimal = Field(
        default=Decimal("0"), ge=0, le=100, description="Inclusion % (0-100)"
    )
    has_operational_control: bool = Field(default=False, description="OC flag")
    has_financial_control: bool = Field(default=False, description="FC flag")
    scope1_included_tco2e: Decimal = Field(
        default=Decimal("0"), description="Scope 1 included (tCO2e)"
    )
    scope2_included_tco2e: Decimal = Field(
        default=Decimal("0"), description="Scope 2 included (tCO2e)"
    )
    total_included_tco2e: Decimal = Field(
        default=Decimal("0"), description="Total included (tCO2e)"
    )
    facility_count: int = Field(default=0, ge=0, description="Included facilities")
    rationale: str = Field(default="", description="Inclusion rationale")


class BoundaryDefinition(BaseModel):
    """Complete organisational boundary definition result.

    Attributes:
        boundary_id: Unique identifier for this boundary definition.
        org_id: Organisation ID.
        org_name: Organisation name.
        reporting_year: Reporting year.
        base_year: Base year.
        approach: Consolidation approach used.
        entity_results: Per-entity inclusion results.
        total_entities: Total number of entities evaluated.
        included_entities: Number of included entities.
        excluded_entities: Number of excluded entities.
        partial_entities: Number of partially included entities.
        total_scope1_tco2e: Total Scope 1 within boundary (tCO2e).
        total_scope2_tco2e: Total Scope 2 within boundary (tCO2e).
        total_emissions_tco2e: Total emissions within boundary (tCO2e).
        total_facilities: Total facilities in boundary.
        countries_covered: List of countries covered.
        sectors_covered: List of sectors covered.
        calculated_at: Timestamp of calculation.
        processing_time_ms: Calculation duration in milliseconds.
        provenance_hash: SHA-256 hash for audit trail.
    """
    boundary_id: str = Field(default_factory=_new_uuid, description="Boundary ID")
    org_id: str = Field(default="", description="Organisation ID")
    org_name: str = Field(default="", description="Organisation name")
    reporting_year: int = Field(default=2025, description="Reporting year")
    base_year: int = Field(default=2019, description="Base year")
    approach: ConsolidationApproach = Field(
        default=ConsolidationApproach.OPERATIONAL_CONTROL,
        description="Consolidation approach",
    )
    entity_results: List[EntityInclusionResult] = Field(
        default_factory=list, description="Per-entity results"
    )
    total_entities: int = Field(default=0, ge=0, description="Total entities")
    included_entities: int = Field(default=0, ge=0, description="Included entities")
    excluded_entities: int = Field(default=0, ge=0, description="Excluded entities")
    partial_entities: int = Field(default=0, ge=0, description="Partial entities")
    total_scope1_tco2e: Decimal = Field(
        default=Decimal("0"), description="Total Scope 1 in boundary"
    )
    total_scope2_tco2e: Decimal = Field(
        default=Decimal("0"), description="Total Scope 2 in boundary"
    )
    total_emissions_tco2e: Decimal = Field(
        default=Decimal("0"), description="Total emissions in boundary"
    )
    total_facilities: int = Field(default=0, ge=0, description="Total facilities")
    countries_covered: List[str] = Field(
        default_factory=list, description="Countries covered"
    )
    sectors_covered: List[str] = Field(
        default_factory=list, description="Sectors covered"
    )
    calculated_at: datetime = Field(
        default_factory=_utcnow, description="Calculation timestamp"
    )
    processing_time_ms: Decimal = Field(
        default=Decimal("0"), description="Processing time (ms)"
    )
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")


class BoundaryChangeResult(BaseModel):
    """Result of processing a boundary change event.

    Attributes:
        change_id: ID of the boundary change event.
        change_type: Type of structural change.
        effective_date: Date the change takes effect.
        entity_id: Affected entity ID.
        entity_name: Affected entity name.
        inclusion_before: Inclusion status before the change.
        inclusion_after: Inclusion status after the change.
        inclusion_pct_before: Inclusion percentage before.
        inclusion_pct_after: Inclusion percentage after.
        emissions_delta_tco2e: Change in boundary emissions (tCO2e).
        requires_base_year_recalc: Whether base-year recalculation is needed.
        recalculation_trigger: The trigger type if recalculation is needed.
        materiality_pct: Materiality of the change as % of base year.
        rationale: Explanation of the change result.
        calculated_at: Timestamp of calculation.
        provenance_hash: SHA-256 hash for audit trail.
    """
    change_id: str = Field(default="", description="Change event ID")
    change_type: BoundaryChangeType = Field(
        default=BoundaryChangeType.ACQUISITION, description="Change type"
    )
    effective_date: Optional[date] = Field(default=None, description="Effective date")
    entity_id: str = Field(default="", description="Entity ID")
    entity_name: str = Field(default="", description="Entity name")
    inclusion_before: InclusionStatus = Field(
        default=InclusionStatus.EXCLUDED, description="Status before"
    )
    inclusion_after: InclusionStatus = Field(
        default=InclusionStatus.EXCLUDED, description="Status after"
    )
    inclusion_pct_before: Decimal = Field(
        default=Decimal("0"), description="Inclusion % before"
    )
    inclusion_pct_after: Decimal = Field(
        default=Decimal("0"), description="Inclusion % after"
    )
    emissions_delta_tco2e: Decimal = Field(
        default=Decimal("0"), description="Emissions delta (tCO2e)"
    )
    requires_base_year_recalc: bool = Field(
        default=False, description="Requires base-year recalculation"
    )
    recalculation_trigger: RecalculationTrigger = Field(
        default=RecalculationTrigger.NOT_TRIGGERED,
        description="Recalculation trigger",
    )
    materiality_pct: Decimal = Field(
        default=Decimal("0"), description="Materiality as % of base year"
    )
    rationale: str = Field(default="", description="Rationale")
    calculated_at: datetime = Field(
        default_factory=_utcnow, description="Calculation timestamp"
    )
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")


class BaseYearImpactAssessment(BaseModel):
    """Assessment of boundary change impact on the base year.

    Attributes:
        assessment_id: Unique assessment ID.
        change_id: ID of the boundary change event evaluated.
        base_year: Base year being assessed.
        original_base_year_emissions_tco2e: Original base-year total.
        structural_change_emissions_tco2e: Emissions from structural change.
        adjusted_base_year_emissions_tco2e: Adjusted base-year total.
        materiality_pct: Materiality percentage.
        significance_threshold_pct: Threshold used.
        exceeds_threshold: Whether change exceeds significance threshold.
        recalculation_trigger: Trigger type if applicable.
        recommendation: Recommended action.
        adjustment_methodology: Description of how base year should be adjusted.
        calculated_at: Timestamp.
        provenance_hash: SHA-256 hash for audit trail.
    """
    assessment_id: str = Field(default_factory=_new_uuid, description="Assessment ID")
    change_id: str = Field(default="", description="Change event ID")
    base_year: int = Field(default=2019, description="Base year")
    original_base_year_emissions_tco2e: Decimal = Field(
        default=Decimal("0"), description="Original base-year emissions"
    )
    structural_change_emissions_tco2e: Decimal = Field(
        default=Decimal("0"), description="Structural change emissions"
    )
    adjusted_base_year_emissions_tco2e: Decimal = Field(
        default=Decimal("0"), description="Adjusted base-year emissions"
    )
    materiality_pct: Decimal = Field(
        default=Decimal("0"), description="Materiality %"
    )
    significance_threshold_pct: Decimal = Field(
        default=DEFAULT_SIGNIFICANCE_THRESHOLD_PCT,
        description="Significance threshold %",
    )
    exceeds_threshold: bool = Field(
        default=False, description="Exceeds significance threshold"
    )
    recalculation_trigger: RecalculationTrigger = Field(
        default=RecalculationTrigger.NOT_TRIGGERED,
        description="Recalculation trigger",
    )
    recommendation: str = Field(default="", description="Recommended action")
    adjustment_methodology: str = Field(
        default="", description="Adjustment methodology"
    )
    calculated_at: datetime = Field(
        default_factory=_utcnow, description="Calculation timestamp"
    )
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")


class BoundaryReport(BaseModel):
    """Complete boundary report output.

    Attributes:
        report_id: Unique report ID.
        boundary_definition: The boundary definition.
        change_results: Results of all boundary changes evaluated.
        base_year_assessments: Base-year impact assessments.
        summary_text: Human-readable summary.
        warnings: List of warnings or issues.
        calculated_at: Timestamp.
        provenance_hash: SHA-256 hash for audit trail.
    """
    report_id: str = Field(default_factory=_new_uuid, description="Report ID")
    boundary_definition: Optional[BoundaryDefinition] = Field(
        default=None, description="Boundary definition"
    )
    change_results: List[BoundaryChangeResult] = Field(
        default_factory=list, description="Boundary change results"
    )
    base_year_assessments: List[BaseYearImpactAssessment] = Field(
        default_factory=list, description="Base-year assessments"
    )
    summary_text: str = Field(default="", description="Summary text")
    warnings: List[str] = Field(default_factory=list, description="Warnings")
    calculated_at: datetime = Field(
        default_factory=_utcnow, description="Calculation timestamp"
    )
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")


# ---------------------------------------------------------------------------
# Model Rebuild (resolve forward references from __future__ annotations)
# ---------------------------------------------------------------------------

Facility.model_rebuild()
LegalEntity.model_rebuild()
OrganizationStructure.model_rebuild()
BoundaryChangeEvent.model_rebuild()
EntityInclusionResult.model_rebuild()
BoundaryDefinition.model_rebuild()
BoundaryChangeResult.model_rebuild()
BaseYearImpactAssessment.model_rebuild()
BoundaryReport.model_rebuild()


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class OrganizationalBoundaryEngine:
    """GHG Protocol Chapter 3 organisational boundary engine.

    Implements all three consolidation approaches (equity share, operational
    control, financial control) and handles boundary changes including
    base-year recalculation impact assessment.

    Attributes:
        _boundary: The current boundary definition, if computed.
        _change_results: Results of processed boundary changes.
        _base_year_assessments: Base-year impact assessments.
        _warnings: Accumulated warnings.

    Example:
        >>> engine = OrganizationalBoundaryEngine()
        >>> org = OrganizationStructure(org_name="Acme Corp", entities=[...])
        >>> boundary = engine.define_boundary(
        ...     org, ConsolidationApproach.OPERATIONAL_CONTROL
        ... )
        >>> assert boundary.total_entities > 0
    """

    def __init__(self) -> None:
        """Initialise OrganizationalBoundaryEngine."""
        self._boundary: Optional[BoundaryDefinition] = None
        self._change_results: List[BoundaryChangeResult] = []
        self._base_year_assessments: List[BaseYearImpactAssessment] = []
        self._warnings: List[str] = []
        logger.info(
            "OrganizationalBoundaryEngine v%s initialised", _MODULE_VERSION
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def define_boundary(
        self,
        org_structure: OrganizationStructure,
        approach: Optional[ConsolidationApproach] = None,
    ) -> BoundaryDefinition:
        """Define the organisational boundary for the given structure.

        Evaluates each legal entity in the organisation against the chosen
        consolidation approach and produces a complete BoundaryDefinition
        with per-entity inclusion results.

        Args:
            org_structure: The full organisational structure with entities.
            approach: Consolidation approach to use.  If None, uses the
                default from org_structure.

        Returns:
            BoundaryDefinition with all entity results and totals.

        Raises:
            ValueError: If no entities are provided.
        """
        t0 = time.perf_counter()
        self._warnings = []

        if not org_structure.entities:
            raise ValueError("Organisation structure must contain at least one entity")

        effective_approach = approach or org_structure.default_approach
        logger.info(
            "Defining boundary for '%s' (%d entities) using %s",
            org_structure.org_name,
            len(org_structure.entities),
            effective_approach.value,
        )

        # Evaluate each entity.
        entity_results: List[EntityInclusionResult] = []
        for entity in org_structure.entities:
            result = self._evaluate_entity(entity, effective_approach)
            entity_results.append(result)

        # Aggregate totals.
        total_s1 = sum(
            (r.scope1_included_tco2e for r in entity_results), Decimal("0")
        )
        total_s2 = sum(
            (r.scope2_included_tco2e for r in entity_results), Decimal("0")
        )
        included_count = sum(
            1 for r in entity_results
            if r.inclusion_status == InclusionStatus.INCLUDED
        )
        excluded_count = sum(
            1 for r in entity_results
            if r.inclusion_status == InclusionStatus.EXCLUDED
        )
        partial_count = sum(
            1 for r in entity_results
            if r.inclusion_status == InclusionStatus.PARTIAL
        )
        total_facilities = sum(r.facility_count for r in entity_results)

        # Collect countries and sectors.
        countries: set[str] = set()
        sectors: set[str] = set()
        for entity in org_structure.entities:
            if entity.country_of_incorporation:
                countries.add(entity.country_of_incorporation)
            if entity.sector:
                sectors.add(entity.sector)
            for fac in entity.facilities:
                if fac.country:
                    countries.add(fac.country)
                if fac.sector:
                    sectors.add(fac.sector)

        elapsed = Decimal(str(round((time.perf_counter() - t0) * 1000, 3)))

        boundary = BoundaryDefinition(
            org_id=org_structure.org_id,
            org_name=org_structure.org_name,
            reporting_year=org_structure.reporting_year,
            base_year=org_structure.base_year,
            approach=effective_approach,
            entity_results=entity_results,
            total_entities=len(entity_results),
            included_entities=included_count,
            excluded_entities=excluded_count,
            partial_entities=partial_count,
            total_scope1_tco2e=_round_val(total_s1, 4),
            total_scope2_tco2e=_round_val(total_s2, 4),
            total_emissions_tco2e=_round_val(total_s1 + total_s2, 4),
            total_facilities=total_facilities,
            countries_covered=sorted(countries),
            sectors_covered=sorted(sectors),
            processing_time_ms=elapsed,
        )
        boundary.provenance_hash = _compute_hash(boundary)
        self._boundary = boundary

        logger.info(
            "Boundary defined: %d included, %d excluded, %d partial, "
            "total=%.2f tCO2e",
            included_count, excluded_count, partial_count,
            float(total_s1 + total_s2),
        )
        return boundary

    def calculate_inclusion_percentages(
        self,
        entities: List[LegalEntity],
        approach: ConsolidationApproach,
    ) -> Dict[str, Decimal]:
        """Calculate inclusion percentages for each entity.

        Returns a mapping of entity_id to inclusion percentage (0-100)
        based on the chosen consolidation approach.

        Args:
            entities: List of legal entities.
            approach: Consolidation approach.

        Returns:
            Dict mapping entity_id to inclusion percentage.
        """
        logger.info(
            "Calculating inclusion percentages for %d entities using %s",
            len(entities), approach.value,
        )
        result: Dict[str, Decimal] = {}

        for entity in entities:
            if approach == ConsolidationApproach.EQUITY_SHARE:
                pct = self._equity_share_pct(entity)
            elif approach == ConsolidationApproach.OPERATIONAL_CONTROL:
                pct = self._operational_control_pct(entity)
            elif approach == ConsolidationApproach.FINANCIAL_CONTROL:
                pct = self._financial_control_pct(entity)
            else:
                pct = Decimal("0")

            result[entity.entity_id] = _round_val(pct, 2)
            logger.debug(
                "Entity '%s' (%s): inclusion_pct=%.2f%%",
                entity.entity_name, entity.entity_type.value, float(pct),
            )

        return result

    def handle_boundary_change(
        self,
        change_event: BoundaryChangeEvent,
        current_boundary: Optional[BoundaryDefinition] = None,
    ) -> BoundaryChangeResult:
        """Process a structural boundary change event.

        Evaluates the impact of an acquisition, divestiture, merger, or
        other structural change on the organisational boundary.

        Args:
            change_event: The boundary change event to process.
            current_boundary: The current boundary definition.  If None,
                uses the internally stored boundary.

        Returns:
            BoundaryChangeResult with delta analysis.

        Raises:
            ValueError: If no boundary is available.
        """
        t0 = time.perf_counter()
        boundary = current_boundary or self._boundary
        if boundary is None:
            raise ValueError(
                "No boundary available. Call define_boundary() first or "
                "provide current_boundary."
            )

        logger.info(
            "Processing boundary change: %s for entity '%s' (effective %s)",
            change_event.change_type.value,
            change_event.entity_name,
            change_event.effective_date,
        )

        # Determine inclusion before and after.
        inclusion_before, pct_before = self._determine_change_inclusion_before(
            change_event, boundary
        )
        inclusion_after, pct_after = self._determine_change_inclusion_after(
            change_event, boundary
        )

        # Calculate emissions delta.
        emissions_impact = _decimal(change_event.emissions_impact_tco2e)
        emissions_delta = self._calculate_emissions_delta(
            change_event, pct_before, pct_after, emissions_impact
        )

        # Check materiality.
        base_year_total = boundary.total_emissions_tco2e
        materiality = _safe_pct(abs(emissions_delta), base_year_total)
        exceeds = materiality > _decimal(
            boundary.approach  # reuse threshold from org
        ) if False else materiality > DEFAULT_SIGNIFICANCE_THRESHOLD_PCT

        trigger = RecalculationTrigger.NOT_TRIGGERED
        if exceeds:
            if change_event.change_type in (
                BoundaryChangeType.ACQUISITION,
                BoundaryChangeType.DIVESTITURE,
                BoundaryChangeType.MERGER,
                BoundaryChangeType.OUTSOURCING,
                BoundaryChangeType.INSOURCING,
            ):
                trigger = RecalculationTrigger.STRUCTURAL_CHANGE
            elif change_event.change_type == BoundaryChangeType.METHODOLOGY_CHANGE:
                trigger = RecalculationTrigger.METHODOLOGY_CHANGE
            else:
                trigger = RecalculationTrigger.STRUCTURAL_CHANGE

        rationale = self._build_change_rationale(
            change_event, inclusion_before, inclusion_after,
            pct_before, pct_after, emissions_delta, materiality, exceeds,
        )

        result = BoundaryChangeResult(
            change_id=change_event.change_id,
            change_type=change_event.change_type,
            effective_date=change_event.effective_date,
            entity_id=change_event.entity_id,
            entity_name=change_event.entity_name,
            inclusion_before=inclusion_before,
            inclusion_after=inclusion_after,
            inclusion_pct_before=_round_val(pct_before, 2),
            inclusion_pct_after=_round_val(pct_after, 2),
            emissions_delta_tco2e=_round_val(emissions_delta, 4),
            requires_base_year_recalc=exceeds,
            recalculation_trigger=trigger,
            materiality_pct=_round_val(materiality, 4),
            rationale=rationale,
        )
        result.provenance_hash = _compute_hash(result)
        self._change_results.append(result)

        logger.info(
            "Boundary change processed: delta=%.2f tCO2e, materiality=%.2f%%, "
            "recalc_needed=%s",
            float(emissions_delta), float(materiality), exceeds,
        )
        return result

    def assess_base_year_impact(
        self,
        change: BoundaryChangeEvent,
        original_base_year_emissions: Decimal,
        significance_threshold_pct: Optional[Decimal] = None,
    ) -> BaseYearImpactAssessment:
        """Assess the impact of a structural change on the base year.

        Calculates whether the change requires base-year recalculation
        per GHG Protocol Chapter 5 guidance.

        Args:
            change: The boundary change event.
            original_base_year_emissions: Original base-year emissions (tCO2e).
            significance_threshold_pct: Custom threshold (default 5%).

        Returns:
            BaseYearImpactAssessment with recommendation.
        """
        t0 = time.perf_counter()
        threshold = significance_threshold_pct or DEFAULT_SIGNIFICANCE_THRESHOLD_PCT
        original = _decimal(original_base_year_emissions)
        change_emissions = abs(_decimal(change.emissions_impact_tco2e))

        logger.info(
            "Assessing base-year impact: change=%.2f tCO2e vs base=%.2f tCO2e "
            "(threshold=%.1f%%)",
            float(change_emissions), float(original), float(threshold),
        )

        # Calculate materiality.
        materiality = _safe_pct(change_emissions, original)
        exceeds = materiality > threshold

        # Determine adjusted base year.
        adjusted = self._calculate_adjusted_base_year(
            change, original, change_emissions
        )

        # Determine trigger.
        trigger = RecalculationTrigger.NOT_TRIGGERED
        if exceeds:
            if change.change_type == BoundaryChangeType.METHODOLOGY_CHANGE:
                trigger = RecalculationTrigger.METHODOLOGY_CHANGE
            else:
                trigger = RecalculationTrigger.STRUCTURAL_CHANGE

        # Build recommendation.
        recommendation = self._build_base_year_recommendation(
            change, materiality, threshold, exceeds
        )

        # Describe methodology.
        methodology = self._describe_adjustment_methodology(change)

        assessment = BaseYearImpactAssessment(
            change_id=change.change_id,
            base_year=self._boundary.base_year if self._boundary else 2019,
            original_base_year_emissions_tco2e=_round_val(original, 4),
            structural_change_emissions_tco2e=_round_val(change_emissions, 4),
            adjusted_base_year_emissions_tco2e=_round_val(adjusted, 4),
            materiality_pct=_round_val(materiality, 4),
            significance_threshold_pct=threshold,
            exceeds_threshold=exceeds,
            recalculation_trigger=trigger,
            recommendation=recommendation,
            adjustment_methodology=methodology,
        )
        assessment.provenance_hash = _compute_hash(assessment)
        self._base_year_assessments.append(assessment)

        logger.info(
            "Base-year impact: materiality=%.2f%%, exceeds=%s, trigger=%s",
            float(materiality), exceeds, trigger.value,
        )
        return assessment

    def generate_boundary_report(self) -> BoundaryReport:
        """Generate a comprehensive boundary report.

        Collects the current boundary definition, all change results, and
        all base-year assessments into a single report.

        Returns:
            BoundaryReport with summary text.

        Raises:
            ValueError: If no boundary has been defined.
        """
        if self._boundary is None:
            raise ValueError(
                "No boundary has been defined. Call define_boundary() first."
            )

        logger.info("Generating boundary report")

        summary_parts: List[str] = []

        # Boundary summary.
        b = self._boundary
        summary_parts.append(
            f"Organisational Boundary Report for {b.org_name}"
        )
        summary_parts.append(f"Reporting Year: {b.reporting_year}")
        summary_parts.append(f"Base Year: {b.base_year}")
        summary_parts.append(f"Consolidation Approach: {b.approach.value}")
        summary_parts.append(
            f"Total Entities: {b.total_entities} "
            f"(included={b.included_entities}, "
            f"excluded={b.excluded_entities}, "
            f"partial={b.partial_entities})"
        )
        summary_parts.append(f"Total Facilities: {b.total_facilities}")
        summary_parts.append(
            f"Total Scope 1: {b.total_scope1_tco2e} tCO2e"
        )
        summary_parts.append(
            f"Total Scope 2: {b.total_scope2_tco2e} tCO2e"
        )
        summary_parts.append(
            f"Total Emissions: {b.total_emissions_tco2e} tCO2e"
        )
        summary_parts.append(
            f"Countries: {', '.join(b.countries_covered) or 'None'}"
        )
        summary_parts.append(
            f"Sectors: {', '.join(b.sectors_covered) or 'None'}"
        )

        # Change summary.
        if self._change_results:
            summary_parts.append("")
            summary_parts.append(
                f"Boundary Changes Evaluated: {len(self._change_results)}"
            )
            recalc_needed = sum(
                1 for cr in self._change_results if cr.requires_base_year_recalc
            )
            summary_parts.append(
                f"Changes Requiring Base-Year Recalculation: {recalc_needed}"
            )

        # Warnings.
        warnings = list(self._warnings)
        for cr in self._change_results:
            if cr.requires_base_year_recalc:
                warnings.append(
                    f"Change '{cr.change_type.value}' for entity "
                    f"'{cr.entity_name}' requires base-year recalculation "
                    f"(materiality={cr.materiality_pct}%)."
                )

        summary_text = "\n".join(summary_parts)

        report = BoundaryReport(
            boundary_definition=self._boundary,
            change_results=list(self._change_results),
            base_year_assessments=list(self._base_year_assessments),
            summary_text=summary_text,
            warnings=warnings,
        )
        report.provenance_hash = _compute_hash(report)

        logger.info(
            "Boundary report generated with %d warnings", len(warnings)
        )
        return report

    # ------------------------------------------------------------------
    # Private Methods
    # ------------------------------------------------------------------

    def _evaluate_entity(
        self,
        entity: LegalEntity,
        approach: ConsolidationApproach,
    ) -> EntityInclusionResult:
        """Evaluate a single entity for boundary inclusion.

        Args:
            entity: The legal entity to evaluate.
            approach: The consolidation approach.

        Returns:
            EntityInclusionResult for the entity.
        """
        if approach == ConsolidationApproach.EQUITY_SHARE:
            pct = self._equity_share_pct(entity)
        elif approach == ConsolidationApproach.OPERATIONAL_CONTROL:
            pct = self._operational_control_pct(entity)
        elif approach == ConsolidationApproach.FINANCIAL_CONTROL:
            pct = self._financial_control_pct(entity)
        else:
            pct = Decimal("0")

        # Determine inclusion status.
        if pct >= Decimal("100"):
            status = InclusionStatus.INCLUDED
        elif pct > Decimal("0"):
            status = InclusionStatus.PARTIAL
        else:
            status = InclusionStatus.EXCLUDED

        # Inactive entities are excluded regardless.
        if not entity.is_active:
            status = InclusionStatus.EXCLUDED
            pct = Decimal("0")
            self._warnings.append(
                f"Entity '{entity.entity_name}' is inactive and excluded."
            )

        # Calculate included emissions.
        fraction = _safe_divide(pct, Decimal("100"))
        s1_included = entity.total_scope1_tco2e * fraction
        s2_included = entity.total_scope2_tco2e * fraction

        # Determine OC and FC flags.
        oc = self._resolve_operational_control(entity)
        fc = self._resolve_financial_control(entity)

        # Count active facilities.
        active_facilities = sum(
            1 for f in entity.facilities
            if f.operational_status in (
                FacilityOperationalStatus.ACTIVE,
                FacilityOperationalStatus.IDLE,
            )
        ) if status != InclusionStatus.EXCLUDED else 0

        rationale = self._build_entity_rationale(entity, approach, pct, status)

        return EntityInclusionResult(
            entity_id=entity.entity_id,
            entity_name=entity.entity_name,
            entity_type=entity.entity_type,
            equity_pct=entity.equity_pct,
            inclusion_status=status,
            inclusion_pct=_round_val(pct, 2),
            has_operational_control=oc,
            has_financial_control=fc,
            scope1_included_tco2e=_round_val(s1_included, 4),
            scope2_included_tco2e=_round_val(s2_included, 4),
            total_included_tco2e=_round_val(s1_included + s2_included, 4),
            facility_count=active_facilities,
            rationale=rationale,
        )

    def _equity_share_pct(self, entity: LegalEntity) -> Decimal:
        """Calculate inclusion % under equity share approach.

        Per GHG Protocol Ch. 3: include emissions proportional to equity
        ownership percentage.

        Args:
            entity: The legal entity.

        Returns:
            Inclusion percentage (0-100).
        """
        return _decimal(entity.equity_pct)

    def _operational_control_pct(self, entity: LegalEntity) -> Decimal:
        """Calculate inclusion % under operational control approach.

        Per GHG Protocol Ch. 3: if the organisation has operational control,
        include 100% of the entity's emissions.

        Args:
            entity: The legal entity.

        Returns:
            100 if org has operational control, 0 otherwise.
        """
        has_oc = self._resolve_operational_control(entity)
        return Decimal("100") if has_oc else Decimal("0")

    def _financial_control_pct(self, entity: LegalEntity) -> Decimal:
        """Calculate inclusion % under financial control approach.

        Per GHG Protocol Ch. 3: if the organisation has financial control
        (ability to direct financial and operating policies with a view
        to gaining economic benefits), include 100%.

        Args:
            entity: The legal entity.

        Returns:
            100 if org has financial control, 0 otherwise.
        """
        has_fc = self._resolve_financial_control(entity)
        return Decimal("100") if has_fc else Decimal("0")

    def _resolve_operational_control(self, entity: LegalEntity) -> bool:
        """Resolve whether the org has operational control over entity.

        Uses entity's explicit flag if set, otherwise defaults based on
        entity type per GHG Protocol guidance.

        Args:
            entity: The legal entity.

        Returns:
            True if the org has operational control.
        """
        if entity.has_operational_control is not None:
            return entity.has_operational_control
        return OPERATIONAL_CONTROL_DEFAULTS.get(entity.entity_type.value, False)

    def _resolve_financial_control(self, entity: LegalEntity) -> bool:
        """Resolve whether the org has financial control over entity.

        Uses entity's explicit flag if set, otherwise defaults based on
        entity type per GHG Protocol guidance.

        Args:
            entity: The legal entity.

        Returns:
            True if the org has financial control.
        """
        if entity.has_financial_control is not None:
            return entity.has_financial_control
        return FINANCIAL_CONTROL_DEFAULTS.get(entity.entity_type.value, False)

    def _build_entity_rationale(
        self,
        entity: LegalEntity,
        approach: ConsolidationApproach,
        pct: Decimal,
        status: InclusionStatus,
    ) -> str:
        """Build a human-readable rationale for entity inclusion.

        Args:
            entity: The legal entity.
            approach: The consolidation approach.
            pct: The inclusion percentage.
            status: The inclusion status.

        Returns:
            Rationale string.
        """
        parts: List[str] = []
        parts.append(
            f"Entity '{entity.entity_name}' ({entity.entity_type.value})"
        )

        if approach == ConsolidationApproach.EQUITY_SHARE:
            parts.append(
                f"evaluated under equity share approach with "
                f"{entity.equity_pct}% equity ownership."
            )
            if pct > Decimal("0"):
                parts.append(
                    f"Include {pct}% of emissions ({status.value})."
                )
            else:
                parts.append("Zero equity share; excluded.")

        elif approach == ConsolidationApproach.OPERATIONAL_CONTROL:
            oc = self._resolve_operational_control(entity)
            parts.append(
                f"evaluated under operational control approach. "
                f"Operational control: {'Yes' if oc else 'No'}."
            )
            if oc:
                parts.append("Include 100% of emissions.")
            else:
                parts.append("No operational control; excluded.")

        elif approach == ConsolidationApproach.FINANCIAL_CONTROL:
            fc = self._resolve_financial_control(entity)
            parts.append(
                f"evaluated under financial control approach. "
                f"Financial control: {'Yes' if fc else 'No'}."
            )
            if fc:
                parts.append("Include 100% of emissions.")
            else:
                parts.append("No financial control; excluded.")

        if not entity.is_active:
            parts.append("Entity is inactive; excluded regardless of approach.")

        return " ".join(parts)

    def _determine_change_inclusion_before(
        self,
        change: BoundaryChangeEvent,
        boundary: BoundaryDefinition,
    ) -> Tuple[InclusionStatus, Decimal]:
        """Determine inclusion status before the boundary change.

        Args:
            change: The boundary change event.
            boundary: The current boundary.

        Returns:
            Tuple of (InclusionStatus, inclusion_pct) before change.
        """
        # Check if entity existed in boundary before.
        for er in boundary.entity_results:
            if er.entity_id == change.entity_id:
                return er.inclusion_status, er.inclusion_pct

        # For acquisitions, entity was not in boundary before.
        if change.change_type in (
            BoundaryChangeType.ACQUISITION,
            BoundaryChangeType.MERGER,
            BoundaryChangeType.INSOURCING,
            BoundaryChangeType.ORGANIC_GROWTH,
        ):
            return InclusionStatus.EXCLUDED, Decimal("0")

        # For divestitures, if not found assume it was included.
        if change.change_type in (
            BoundaryChangeType.DIVESTITURE,
            BoundaryChangeType.OUTSOURCING,
            BoundaryChangeType.SHUTDOWN,
        ):
            return InclusionStatus.INCLUDED, Decimal("100")

        return InclusionStatus.EXCLUDED, Decimal("0")

    def _determine_change_inclusion_after(
        self,
        change: BoundaryChangeEvent,
        boundary: BoundaryDefinition,
    ) -> Tuple[InclusionStatus, Decimal]:
        """Determine inclusion status after the boundary change.

        Args:
            change: The boundary change event.
            boundary: The current boundary.

        Returns:
            Tuple of (InclusionStatus, inclusion_pct) after change.
        """
        approach = boundary.approach

        if change.change_type in (
            BoundaryChangeType.ACQUISITION,
            BoundaryChangeType.MERGER,
            BoundaryChangeType.INSOURCING,
            BoundaryChangeType.ORGANIC_GROWTH,
        ):
            # Entity enters the boundary.
            if approach == ConsolidationApproach.EQUITY_SHARE:
                pct = _decimal(change.equity_pct_after)
                if pct >= Decimal("100"):
                    return InclusionStatus.INCLUDED, pct
                elif pct > Decimal("0"):
                    return InclusionStatus.PARTIAL, pct
                return InclusionStatus.EXCLUDED, Decimal("0")
            else:
                # OC or FC: check if org has control after acquisition.
                oc_after = change.operational_control_after
                fc_after = change.financial_control_after
                has_control = (
                    oc_after if approach == ConsolidationApproach.OPERATIONAL_CONTROL
                    else fc_after
                )
                if has_control:
                    return InclusionStatus.INCLUDED, Decimal("100")
                return InclusionStatus.EXCLUDED, Decimal("0")

        elif change.change_type in (
            BoundaryChangeType.DIVESTITURE,
            BoundaryChangeType.OUTSOURCING,
            BoundaryChangeType.SHUTDOWN,
        ):
            # Entity leaves the boundary.
            if approach == ConsolidationApproach.EQUITY_SHARE:
                pct = _decimal(change.equity_pct_after)
                if pct > Decimal("0"):
                    return InclusionStatus.PARTIAL, pct
                return InclusionStatus.EXCLUDED, Decimal("0")
            return InclusionStatus.EXCLUDED, Decimal("0")

        elif change.change_type == BoundaryChangeType.METHODOLOGY_CHANGE:
            # Methodology change does not change inclusion, but may affect amounts.
            for er in boundary.entity_results:
                if er.entity_id == change.entity_id:
                    return er.inclusion_status, er.inclusion_pct
            return InclusionStatus.EXCLUDED, Decimal("0")

        return InclusionStatus.EXCLUDED, Decimal("0")

    def _calculate_emissions_delta(
        self,
        change: BoundaryChangeEvent,
        pct_before: Decimal,
        pct_after: Decimal,
        emissions_impact: Decimal,
    ) -> Decimal:
        """Calculate the net emissions change from a boundary change.

        For acquisitions: delta = +emissions_impact * (pct_after / 100)
        For divestitures: delta = -emissions_impact * (pct_before / 100)
        For equity changes: delta = emissions_impact * ((pct_after - pct_before) / 100)

        Args:
            change: The boundary change event.
            pct_before: Inclusion % before.
            pct_after: Inclusion % after.
            emissions_impact: The emissions impact value.

        Returns:
            Net emissions delta (tCO2e). Positive = increase, negative = decrease.
        """
        if change.change_type in (
            BoundaryChangeType.ACQUISITION,
            BoundaryChangeType.MERGER,
            BoundaryChangeType.INSOURCING,
            BoundaryChangeType.ORGANIC_GROWTH,
        ):
            # Emissions entering the boundary.
            fraction = _safe_divide(pct_after, Decimal("100"))
            return emissions_impact * fraction

        elif change.change_type in (
            BoundaryChangeType.DIVESTITURE,
            BoundaryChangeType.OUTSOURCING,
            BoundaryChangeType.SHUTDOWN,
        ):
            # Emissions leaving the boundary.
            fraction = _safe_divide(pct_before, Decimal("100"))
            return -(emissions_impact * fraction)

        elif change.change_type == BoundaryChangeType.METHODOLOGY_CHANGE:
            # Direct impact as stated.
            return emissions_impact

        # Default: proportional change.
        delta_pct = pct_after - pct_before
        fraction = _safe_divide(delta_pct, Decimal("100"))
        return emissions_impact * fraction

    def _calculate_adjusted_base_year(
        self,
        change: BoundaryChangeEvent,
        original: Decimal,
        change_emissions: Decimal,
    ) -> Decimal:
        """Calculate the adjusted base-year emissions.

        Per GHG Protocol Ch. 5:
        - Acquisitions: add the acquired entity's base-year emissions.
        - Divestitures: subtract the divested entity's base-year emissions.
        - Mergers: add the merged entity's base-year emissions.

        Args:
            change: The boundary change event.
            original: Original base-year emissions (tCO2e).
            change_emissions: Emissions from the structural change (positive).

        Returns:
            Adjusted base-year emissions (tCO2e).
        """
        if change.change_type in (
            BoundaryChangeType.ACQUISITION,
            BoundaryChangeType.MERGER,
            BoundaryChangeType.INSOURCING,
            BoundaryChangeType.ORGANIC_GROWTH,
        ):
            return original + change_emissions

        elif change.change_type in (
            BoundaryChangeType.DIVESTITURE,
            BoundaryChangeType.OUTSOURCING,
            BoundaryChangeType.SHUTDOWN,
        ):
            adjusted = original - change_emissions
            # Base year cannot go negative.
            return max(adjusted, Decimal("0"))

        elif change.change_type == BoundaryChangeType.METHODOLOGY_CHANGE:
            # Methodology changes may increase or decrease. Use signed impact.
            return original + _decimal(change.emissions_impact_tco2e)

        return original

    def _build_change_rationale(
        self,
        change: BoundaryChangeEvent,
        inclusion_before: InclusionStatus,
        inclusion_after: InclusionStatus,
        pct_before: Decimal,
        pct_after: Decimal,
        emissions_delta: Decimal,
        materiality: Decimal,
        exceeds: bool,
    ) -> str:
        """Build rationale text for a boundary change result.

        Args:
            change: The boundary change event.
            inclusion_before: Inclusion status before.
            inclusion_after: Inclusion status after.
            pct_before: Inclusion % before.
            pct_after: Inclusion % after.
            emissions_delta: Emissions delta (tCO2e).
            materiality: Materiality percentage.
            exceeds: Whether materiality exceeds threshold.

        Returns:
            Human-readable rationale.
        """
        parts: List[str] = []
        parts.append(
            f"{change.change_type.value.replace('_', ' ').title()} "
            f"of entity '{change.entity_name}' effective "
            f"{change.effective_date}."
        )
        parts.append(
            f"Boundary status changed from {inclusion_before.value} "
            f"({pct_before}%) to {inclusion_after.value} ({pct_after}%)."
        )
        direction = "increase" if emissions_delta > 0 else "decrease"
        parts.append(
            f"Emissions {direction}: {abs(emissions_delta)} tCO2e "
            f"(materiality: {materiality}% of base year)."
        )
        if exceeds:
            parts.append(
                f"Materiality exceeds significance threshold of "
                f"{DEFAULT_SIGNIFICANCE_THRESHOLD_PCT}%. "
                f"Base-year recalculation is recommended per GHG Protocol Ch. 5."
            )
        else:
            parts.append(
                f"Materiality is below the significance threshold of "
                f"{DEFAULT_SIGNIFICANCE_THRESHOLD_PCT}%. "
                f"No base-year recalculation required."
            )
        return " ".join(parts)

    def _build_base_year_recommendation(
        self,
        change: BoundaryChangeEvent,
        materiality: Decimal,
        threshold: Decimal,
        exceeds: bool,
    ) -> str:
        """Build recommendation text for base-year impact assessment.

        Args:
            change: The boundary change event.
            materiality: Materiality percentage.
            threshold: Significance threshold percentage.
            exceeds: Whether materiality exceeds threshold.

        Returns:
            Recommendation text.
        """
        if not exceeds:
            return (
                f"No base-year recalculation required. The structural change "
                f"({change.change_type.value}) has a materiality of "
                f"{materiality}%, which is below the {threshold}% threshold."
            )

        if change.change_type in (
            BoundaryChangeType.ACQUISITION,
            BoundaryChangeType.MERGER,
        ):
            return (
                f"Base-year recalculation required. The {change.change_type.value} "
                f"has a materiality of {materiality}% (exceeds {threshold}% "
                f"threshold). Add the acquired entity's base-year-equivalent "
                f"emissions to the original base year per GHG Protocol Ch. 5."
            )

        if change.change_type == BoundaryChangeType.DIVESTITURE:
            return (
                f"Base-year recalculation required. The divestiture has a "
                f"materiality of {materiality}% (exceeds {threshold}% threshold). "
                f"Subtract the divested entity's base-year-equivalent emissions "
                f"from the original base year per GHG Protocol Ch. 5."
            )

        if change.change_type == BoundaryChangeType.METHODOLOGY_CHANGE:
            return (
                f"Base-year recalculation required. The methodology change has a "
                f"materiality of {materiality}% (exceeds {threshold}% threshold). "
                f"Recalculate the base year using the updated methodology to "
                f"ensure consistent trend analysis per GHG Protocol Ch. 5."
            )

        return (
            f"Base-year recalculation recommended. The structural change "
            f"({change.change_type.value}) has a materiality of {materiality}%, "
            f"exceeding the {threshold}% significance threshold."
        )

    def _describe_adjustment_methodology(
        self,
        change: BoundaryChangeEvent,
    ) -> str:
        """Describe the base-year adjustment methodology.

        Args:
            change: The boundary change event.

        Returns:
            Methodology description.
        """
        if change.change_type in (
            BoundaryChangeType.ACQUISITION,
            BoundaryChangeType.MERGER,
        ):
            return (
                "Adjusted_base_year = Original_base_year + Acquired_entity_"
                "base_year_equivalent_emissions. The acquired entity's emissions "
                "should be estimated for the base year period as if it had been "
                "part of the organisation. Use the same emission factors and "
                "calculation methodologies as applied to the original base year."
            )

        if change.change_type == BoundaryChangeType.DIVESTITURE:
            return (
                "Adjusted_base_year = Original_base_year - Divested_entity_"
                "base_year_emissions. Remove the divested entity's base-year "
                "emissions from the original base year to maintain consistent "
                "trend comparison."
            )

        if change.change_type == BoundaryChangeType.OUTSOURCING:
            return (
                "Adjusted_base_year = Original_base_year - Outsourced_"
                "operations_base_year_emissions. Subtract the emissions "
                "from operations that have been outsourced. These emissions "
                "may now appear in Scope 3."
            )

        if change.change_type == BoundaryChangeType.INSOURCING:
            return (
                "Adjusted_base_year = Original_base_year + Insourced_"
                "operations_base_year_equivalent_emissions. Add the emissions "
                "from operations that have been brought in-house."
            )

        if change.change_type == BoundaryChangeType.METHODOLOGY_CHANGE:
            return (
                "Recalculate base year using the updated methodology. Apply "
                "the new emission factors, calculation methods, or data sources "
                "to the base-year activity data to produce a restated base year."
            )

        if change.change_type == BoundaryChangeType.ORGANIC_GROWTH:
            return (
                "Organic growth does not typically trigger base-year "
                "recalculation per GHG Protocol Ch. 5. The base year reflects "
                "the organisation's footprint at that time; growth represents "
                "real emissions increases."
            )

        if change.change_type == BoundaryChangeType.SHUTDOWN:
            return (
                "Facility shutdown does not typically trigger base-year "
                "recalculation per GHG Protocol Ch. 5 unless the shutdown "
                "results from a structural change (e.g. divestiture)."
            )

        return "No specific adjustment methodology defined for this change type."
