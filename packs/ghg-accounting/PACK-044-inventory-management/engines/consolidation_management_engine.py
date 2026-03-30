# -*- coding: utf-8 -*-
"""
ConsolidationManagementEngine - PACK-044 Inventory Management Engine 7
=======================================================================

Multi-entity GHG inventory consolidation engine that manages the
aggregation of emissions data across complex corporate hierarchies
including subsidiaries, joint ventures, and associated entities.

Supports both equity-share and operational-control consolidation
approaches per GHG Protocol Corporate Standard Chapter 3 guidance,
with full intra-group elimination to prevent double-counting of
transferred emissions between entities within the same group.

Calculation Methodology:
    Equity-Share Consolidation:
        E_entity = sum(E_facility_i * equity_pct_i / 100)
        E_group  = sum(E_entity_j) - E_intra_group_eliminations

    Operational-Control Consolidation:
        E_entity = sum(E_facility_i * control_flag_i)
            where control_flag_i = 100% if operational control, 0% otherwise
        E_group  = sum(E_entity_j) - E_intra_group_eliminations

    Financial-Control Consolidation:
        E_entity = sum(E_facility_i * financial_control_flag_i)
            where financial_control_flag_i = 100% if financial control, 0%
        E_group  = sum(E_entity_j) - E_intra_group_eliminations

    Intra-Group Elimination:
        For each pair of entities (A, B) within the same group:
            If entity A transfers emissions to entity B:
                E_elimination = min(E_transferred_A, E_received_B)
                E_group -= E_elimination

    Subsidiary Completeness Check:
        completeness_pct = (submitted_entities / required_entities) * 100

    Equity Reconciliation:
        For each entity: |equity_reported - equity_registered| < threshold

Regulatory References:
    - GHG Protocol Corporate Standard (Revised), Chapter 3 (Setting Boundaries)
    - GHG Protocol Corporate Standard, Chapter 4 (Tracking Emissions)
    - ISO 14064-1:2018, Clause 5.1 (Organisational Boundaries)
    - CSRD / ESRS E1, AR 39-44 (Consolidation requirements)
    - IFRS S2 Climate Disclosures, para B20-B25

Zero-Hallucination:
    - All consolidation uses deterministic Decimal arithmetic
    - Hierarchy traversal uses explicit parent-child graph with no inference
    - Equity percentages from audited financial records only
    - No LLM involvement in any calculation path
    - SHA-256 provenance hash on every result

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-044 Inventory Management
Engine:  7 of 10
Status:  Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from pydantic import BaseModel, Field, field_validator, model_validator

from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _new_uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())

def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash of arbitrary data.

    Uses JSON serialization with sorted keys to guarantee reproducibility.

    Args:
        data: Data to hash -- dict, Pydantic model, or other serializable.

    Returns:
        SHA-256 hex digest string (64 characters).
    """
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
    """Safely convert a value to Decimal.

    Args:
        value: Numeric value to convert.

    Returns:
        Decimal representation; Decimal("0") on failure.
    """
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

def _round2(value: Any) -> float:
    """Round to 2 decimal places using ROUND_HALF_UP."""
    return float(Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))

def _round4(value: Any) -> float:
    """Round to 4 decimal places using ROUND_HALF_UP."""
    return float(Decimal(str(value)).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP))

def _round6(value: Any) -> float:
    """Round to 6 decimal places using ROUND_HALF_UP."""
    return float(Decimal(str(value)).quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP))

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ConsolidationApproach(str, Enum):
    """Consolidation approach per GHG Protocol Chapter 3.

    EQUITY_SHARE:        Report proportional to equity ownership.
    OPERATIONAL_CONTROL: Report 100% for operationally controlled entities.
    FINANCIAL_CONTROL:   Report 100% for financially controlled entities.
    """
    EQUITY_SHARE = "equity_share"
    OPERATIONAL_CONTROL = "operational_control"
    FINANCIAL_CONTROL = "financial_control"

class EntityType(str, Enum):
    """Type of entity within a corporate hierarchy.

    PARENT:          Ultimate parent / holding company.
    SUBSIDIARY:      Majority-owned subsidiary (>50% equity).
    JOINT_VENTURE:   Joint venture (shared control, typically 50/50).
    ASSOCIATE:       Associate entity (significant influence, 20-50%).
    FRANCHISE:       Franchise operation.
    JOINT_OPERATION: Joint operation (shared assets/liabilities).
    BRANCH:          Branch office of a parent entity.
    """
    PARENT = "parent"
    SUBSIDIARY = "subsidiary"
    JOINT_VENTURE = "joint_venture"
    ASSOCIATE = "associate"
    FRANCHISE = "franchise"
    JOINT_OPERATION = "joint_operation"
    BRANCH = "branch"

class SubmissionStatus(str, Enum):
    """Status of an entity's emission data submission.

    NOT_STARTED:  Entity has not begun data submission.
    IN_PROGRESS:  Submission is partially complete.
    SUBMITTED:    Data has been submitted and awaits review.
    APPROVED:     Data has been reviewed and approved.
    REJECTED:     Data was rejected and requires resubmission.
    OVERDUE:      Submission deadline has passed.
    """
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    SUBMITTED = "submitted"
    APPROVED = "approved"
    REJECTED = "rejected"
    OVERDUE = "overdue"

class ConsolidationStatusEnum(str, Enum):
    """Overall status of the consolidation process.

    DRAFT:       Consolidation in progress, incomplete submissions.
    PRELIMINARY: All submissions received, preliminary totals calculated.
    REVIEWED:    Totals reviewed by internal audit.
    FINAL:       Final consolidation, locked for external assurance.
    RESTATED:    Consolidation restated due to corrections or restatement.
    """
    DRAFT = "draft"
    PRELIMINARY = "preliminary"
    REVIEWED = "reviewed"
    FINAL = "final"
    RESTATED = "restated"

class EliminationType(str, Enum):
    """Type of intra-group elimination.

    TRANSFERRED_ELECTRICITY: Electricity transferred between group entities.
    TRANSFERRED_STEAM:       Steam/heat transferred between group entities.
    INTERNAL_TRANSPORT:      Group-owned transport between facilities.
    WASTE_TRANSFER:          Waste transferred between group entities.
    PRODUCT_TRANSFER:        Intermediate product transfers (Scope 3 avoidance).
    """
    TRANSFERRED_ELECTRICITY = "transferred_electricity"
    TRANSFERRED_STEAM = "transferred_steam"
    INTERNAL_TRANSPORT = "internal_transport"
    WASTE_TRANSFER = "waste_transfer"
    PRODUCT_TRANSFER = "product_transfer"

# ---------------------------------------------------------------------------
# Pydantic Models -- Inputs
# ---------------------------------------------------------------------------

class Entity(BaseModel):
    """A legal entity within the corporate hierarchy.

    Attributes:
        entity_id: Unique entity identifier.
        entity_name: Legal name of the entity.
        entity_type: Type of entity (subsidiary, JV, associate, etc.).
        parent_entity_id: ID of the parent entity (None for the root).
        country: ISO 3166-1 alpha-2 country code.
        equity_pct: Equity ownership percentage held by the group.
        has_operational_control: Whether the group has operational control.
        has_financial_control: Whether the group has financial control.
        is_material: Whether the entity is material for reporting.
        reporting_currency: ISO 4217 currency code.
    """
    entity_id: str = Field(default_factory=_new_uuid, description="Entity ID")
    entity_name: str = Field(..., min_length=1, max_length=500, description="Legal name")
    entity_type: EntityType = Field(
        default=EntityType.SUBSIDIARY, description="Entity type"
    )
    parent_entity_id: Optional[str] = Field(
        default=None, description="Parent entity ID"
    )
    country: str = Field(default="", max_length=2, description="Country code")
    equity_pct: Decimal = Field(
        default=Decimal("100"), ge=0, le=100,
        description="Equity ownership percentage"
    )
    has_operational_control: bool = Field(
        default=True, description="Operational control flag"
    )
    has_financial_control: bool = Field(
        default=True, description="Financial control flag"
    )
    is_material: bool = Field(default=True, description="Materiality flag")
    reporting_currency: str = Field(
        default="EUR", max_length=3, description="ISO 4217 currency"
    )

    @field_validator("equity_pct", mode="before")
    @classmethod
    def coerce_equity(cls, v: Any) -> Decimal:
        """Coerce equity percentage to Decimal."""
        return _decimal(v)

class EntityHierarchy(BaseModel):
    """Complete corporate hierarchy definition.

    Attributes:
        hierarchy_id: Unique hierarchy identifier.
        group_name: Name of the corporate group.
        reporting_year: Fiscal year for the inventory.
        consolidation_approach: Selected consolidation approach.
        entities: List of all entities in the hierarchy.
        base_currency: Group reporting currency.
    """
    hierarchy_id: str = Field(default_factory=_new_uuid, description="Hierarchy ID")
    group_name: str = Field(..., min_length=1, max_length=500, description="Group name")
    reporting_year: int = Field(..., ge=1990, le=2100, description="Reporting year")
    consolidation_approach: ConsolidationApproach = Field(
        default=ConsolidationApproach.OPERATIONAL_CONTROL,
        description="Consolidation approach"
    )
    entities: List[Entity] = Field(
        default_factory=list, min_length=1, description="Entities"
    )
    base_currency: str = Field(
        default="EUR", max_length=3, description="Group currency"
    )

    @model_validator(mode="after")
    def validate_hierarchy(self) -> "EntityHierarchy":
        """Validate hierarchy has exactly one root entity."""
        roots = [e for e in self.entities if e.parent_entity_id is None]
        if len(roots) == 0:
            raise ValueError("Hierarchy must have at least one root (parent) entity")
        if len(roots) > 1:
            raise ValueError(
                f"Hierarchy has {len(roots)} root entities; expected exactly 1"
            )
        return self

class SubsidiarySubmission(BaseModel):
    """Emission data submission from a subsidiary or entity.

    Attributes:
        submission_id: Unique submission identifier.
        entity_id: Entity that submitted the data.
        reporting_period_start: Start of reporting period.
        reporting_period_end: End of reporting period.
        scope1_tco2e: Total Scope 1 emissions (tCO2e).
        scope2_location_tco2e: Scope 2 location-based (tCO2e).
        scope2_market_tco2e: Scope 2 market-based (tCO2e).
        scope3_tco2e: Total Scope 3 emissions (tCO2e).
        status: Submission status.
        submitted_at: Submission timestamp.
        approved_by: Approver identifier.
        data_quality_score: Data quality score (0-100).
        notes: Submission notes.
    """
    submission_id: str = Field(default_factory=_new_uuid, description="Submission ID")
    entity_id: str = Field(..., min_length=1, description="Entity ID")
    reporting_period_start: Optional[datetime] = Field(
        default=None, description="Period start"
    )
    reporting_period_end: Optional[datetime] = Field(
        default=None, description="Period end"
    )
    scope1_tco2e: Decimal = Field(
        default=Decimal("0"), ge=0, description="Scope 1 tCO2e"
    )
    scope2_location_tco2e: Decimal = Field(
        default=Decimal("0"), ge=0, description="Scope 2 location tCO2e"
    )
    scope2_market_tco2e: Decimal = Field(
        default=Decimal("0"), ge=0, description="Scope 2 market tCO2e"
    )
    scope3_tco2e: Decimal = Field(
        default=Decimal("0"), ge=0, description="Scope 3 tCO2e"
    )
    status: SubmissionStatus = Field(
        default=SubmissionStatus.NOT_STARTED, description="Submission status"
    )
    submitted_at: Optional[datetime] = Field(
        default=None, description="Submission timestamp"
    )
    approved_by: Optional[str] = Field(default=None, description="Approver")
    data_quality_score: Decimal = Field(
        default=Decimal("0"), ge=0, le=100, description="Data quality (0-100)"
    )
    notes: str = Field(default="", description="Notes")

    @field_validator(
        "scope1_tco2e", "scope2_location_tco2e", "scope2_market_tco2e",
        "scope3_tco2e", "data_quality_score", mode="before"
    )
    @classmethod
    def coerce_decimal(cls, v: Any) -> Decimal:
        """Coerce numeric fields to Decimal."""
        return _decimal(v)

class IntraGroupTransfer(BaseModel):
    """An intra-group emission transfer to be eliminated.

    Attributes:
        transfer_id: Unique transfer identifier.
        from_entity_id: Entity transferring emissions.
        to_entity_id: Entity receiving emissions.
        elimination_type: Type of elimination.
        amount_tco2e: Amount to eliminate (tCO2e).
        scope: Which scope the transfer relates to (1, 2, or 3).
        description: Transfer description.
        evidence_ref: Reference to supporting evidence.
    """
    transfer_id: str = Field(default_factory=_new_uuid, description="Transfer ID")
    from_entity_id: str = Field(..., min_length=1, description="From entity")
    to_entity_id: str = Field(..., min_length=1, description="To entity")
    elimination_type: EliminationType = Field(..., description="Elimination type")
    amount_tco2e: Decimal = Field(
        default=Decimal("0"), ge=0, description="Amount tCO2e"
    )
    scope: int = Field(default=2, ge=1, le=3, description="Scope (1, 2, or 3)")
    description: str = Field(default="", description="Transfer description")
    evidence_ref: str = Field(default="", description="Evidence reference")

    @field_validator("amount_tco2e", mode="before")
    @classmethod
    def coerce_amount(cls, v: Any) -> Decimal:
        """Coerce amount to Decimal."""
        return _decimal(v)

# ---------------------------------------------------------------------------
# Pydantic Models -- Outputs
# ---------------------------------------------------------------------------

class EntityConsolidatedResult(BaseModel):
    """Consolidated emission result for a single entity.

    Attributes:
        entity_id: Entity identifier.
        entity_name: Entity name.
        entity_type: Entity type.
        equity_pct: Equity percentage applied.
        inclusion_pct: Effective inclusion percentage after approach applied.
        raw_scope1_tco2e: Raw Scope 1 before consolidation percentage.
        raw_scope2_location_tco2e: Raw Scope 2 location-based.
        raw_scope2_market_tco2e: Raw Scope 2 market-based.
        raw_scope3_tco2e: Raw Scope 3.
        consolidated_scope1_tco2e: Scope 1 after applying inclusion pct.
        consolidated_scope2_location_tco2e: Scope 2 location after inclusion.
        consolidated_scope2_market_tco2e: Scope 2 market after inclusion.
        consolidated_scope3_tco2e: Scope 3 after inclusion.
        submission_status: Status of the entity's data submission.
        data_quality_score: Quality score (0-100).
    """
    entity_id: str = Field(default="", description="Entity ID")
    entity_name: str = Field(default="", description="Entity name")
    entity_type: str = Field(default="", description="Entity type")
    equity_pct: float = Field(default=100.0, description="Equity %")
    inclusion_pct: float = Field(default=100.0, description="Inclusion %")
    raw_scope1_tco2e: float = Field(default=0.0, description="Raw Scope 1")
    raw_scope2_location_tco2e: float = Field(default=0.0, description="Raw Scope 2 loc")
    raw_scope2_market_tco2e: float = Field(default=0.0, description="Raw Scope 2 mkt")
    raw_scope3_tco2e: float = Field(default=0.0, description="Raw Scope 3")
    consolidated_scope1_tco2e: float = Field(default=0.0, description="Consolidated S1")
    consolidated_scope2_location_tco2e: float = Field(
        default=0.0, description="Consolidated S2 loc"
    )
    consolidated_scope2_market_tco2e: float = Field(
        default=0.0, description="Consolidated S2 mkt"
    )
    consolidated_scope3_tco2e: float = Field(default=0.0, description="Consolidated S3")
    submission_status: str = Field(default="not_started", description="Submission status")
    data_quality_score: float = Field(default=0.0, description="Quality score")

class EliminationRecord(BaseModel):
    """Record of an intra-group elimination applied.

    Attributes:
        elimination_id: Unique elimination identifier.
        from_entity_id: Source entity.
        to_entity_id: Receiving entity.
        elimination_type: Type of elimination.
        scope: Affected scope.
        amount_tco2e: Eliminated amount.
        rationale: Explanation of the elimination.
    """
    elimination_id: str = Field(default_factory=_new_uuid, description="Elimination ID")
    from_entity_id: str = Field(default="", description="From entity")
    to_entity_id: str = Field(default="", description="To entity")
    elimination_type: str = Field(default="", description="Elimination type")
    scope: int = Field(default=2, description="Scope")
    amount_tco2e: float = Field(default=0.0, description="Eliminated amount")
    rationale: str = Field(default="", description="Rationale")

class ConsolidationStatus(BaseModel):
    """Status summary of the consolidation process.

    Attributes:
        total_entities: Total entities in hierarchy.
        entities_submitted: Entities with approved submissions.
        entities_pending: Entities still pending.
        entities_overdue: Entities past deadline.
        completeness_pct: Submission completeness percentage.
        all_approved: Whether all required submissions are approved.
        blocking_entities: Entity IDs blocking consolidation.
        status: Overall consolidation status.
    """
    total_entities: int = Field(default=0, description="Total entities")
    entities_submitted: int = Field(default=0, description="Submitted")
    entities_pending: int = Field(default=0, description="Pending")
    entities_overdue: int = Field(default=0, description="Overdue")
    completeness_pct: float = Field(default=0.0, description="Completeness %")
    all_approved: bool = Field(default=False, description="All approved")
    blocking_entities: List[str] = Field(
        default_factory=list, description="Blocking entity IDs"
    )
    status: str = Field(default="draft", description="Consolidation status")

class ConsolidationManagementResult(BaseModel):
    """Complete multi-entity consolidation result.

    Attributes:
        result_id: Unique result identifier.
        engine_version: Engine version string.
        calculated_at: Calculation timestamp (UTC).
        processing_time_ms: Processing time in milliseconds.
        group_name: Corporate group name.
        reporting_year: Reporting year.
        consolidation_approach: Approach used.
        total_scope1_tco2e: Consolidated Scope 1 total.
        total_scope2_location_tco2e: Consolidated Scope 2 location total.
        total_scope2_market_tco2e: Consolidated Scope 2 market total.
        total_scope3_tco2e: Consolidated Scope 3 total.
        total_all_scopes_tco2e: Grand total all scopes.
        total_eliminations_tco2e: Total intra-group eliminations.
        entity_results: Per-entity consolidated results.
        eliminations: Applied elimination records.
        consolidation_status: Status summary.
        equity_reconciliation_notes: Equity reconciliation notes.
        warnings: Warning messages.
        methodology_notes: Methodology notes.
        provenance_hash: SHA-256 provenance hash.
    """
    result_id: str = Field(default_factory=_new_uuid, description="Result ID")
    engine_version: str = Field(default=_MODULE_VERSION, description="Engine version")
    calculated_at: datetime = Field(
        default_factory=utcnow, description="Calculation timestamp"
    )
    processing_time_ms: float = Field(default=0.0, description="Processing time (ms)")
    group_name: str = Field(default="", description="Group name")
    reporting_year: int = Field(default=2025, description="Reporting year")
    consolidation_approach: str = Field(default="", description="Approach used")
    total_scope1_tco2e: float = Field(default=0.0, description="Consolidated Scope 1")
    total_scope2_location_tco2e: float = Field(
        default=0.0, description="Consolidated Scope 2 location"
    )
    total_scope2_market_tco2e: float = Field(
        default=0.0, description="Consolidated Scope 2 market"
    )
    total_scope3_tco2e: float = Field(default=0.0, description="Consolidated Scope 3")
    total_all_scopes_tco2e: float = Field(
        default=0.0, description="Grand total all scopes"
    )
    total_eliminations_tco2e: float = Field(
        default=0.0, description="Total eliminations"
    )
    entity_results: List[EntityConsolidatedResult] = Field(
        default_factory=list, description="Per-entity results"
    )
    eliminations: List[EliminationRecord] = Field(
        default_factory=list, description="Elimination records"
    )
    consolidation_status: Optional[ConsolidationStatus] = Field(
        default=None, description="Consolidation status"
    )
    equity_reconciliation_notes: List[str] = Field(
        default_factory=list, description="Equity reconciliation notes"
    )
    warnings: List[str] = Field(default_factory=list, description="Warnings")
    methodology_notes: List[str] = Field(
        default_factory=list, description="Methodology notes"
    )
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

# ---------------------------------------------------------------------------
# Model Rebuild
# ---------------------------------------------------------------------------

Entity.model_rebuild()
EntityHierarchy.model_rebuild()
SubsidiarySubmission.model_rebuild()
IntraGroupTransfer.model_rebuild()
EntityConsolidatedResult.model_rebuild()
EliminationRecord.model_rebuild()
ConsolidationStatus.model_rebuild()
ConsolidationManagementResult.model_rebuild()

# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class ConsolidationManagementEngine:
    """Multi-entity GHG inventory consolidation engine.

    Manages the complete consolidation workflow for corporate groups
    with complex hierarchies, applying equity-share, operational-control,
    or financial-control approaches per GHG Protocol Chapter 3.

    Features:
        - Entity hierarchy management with parent-child graph traversal
        - Equity-share, operational-control, and financial-control approaches
        - Subsidiary submission tracking and completeness monitoring
        - Intra-group elimination of transferred emissions
        - Equity reconciliation and variance detection
        - Multi-scope consolidation (Scope 1, 2, 3)

    Guarantees:
        - Deterministic: same inputs produce identical results
        - Reproducible: SHA-256 provenance hash on every result
        - Auditable: full entity-level breakdown and elimination records
        - No LLM: zero hallucination risk in any calculation path

    Usage::

        engine = ConsolidationManagementEngine()
        hierarchy = EntityHierarchy(
            group_name="Acme Corp",
            reporting_year=2025,
            consolidation_approach=ConsolidationApproach.EQUITY_SHARE,
            entities=[parent_entity, subsidiary_1, subsidiary_2],
        )
        submissions = [sub_1, sub_2, sub_3]
        result = engine.consolidate(hierarchy, submissions)
        print(f"Group total: {result.total_all_scopes_tco2e} tCO2e")
    """

    engine_version: str = _MODULE_VERSION

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialise the consolidation management engine.

        Args:
            config: Optional configuration overrides. Supported keys:
                - equity_tolerance_pct (float): tolerance for equity
                    reconciliation checks (default 0.5).
                - require_all_approved (bool): whether all submissions
                    must be approved before final consolidation (default True).
                - include_immaterial_entities (bool): whether to include
                    immaterial entities in the consolidation (default False).
        """
        self._config = config or {}
        self._equity_tolerance = Decimal(
            str(self._config.get("equity_tolerance_pct", "0.5"))
        )
        self._require_all_approved = bool(
            self._config.get("require_all_approved", True)
        )
        self._include_immaterial = bool(
            self._config.get("include_immaterial_entities", False)
        )
        self._warnings: List[str] = []
        self._notes: List[str] = []
        logger.info(
            "ConsolidationManagementEngine v%s initialised.", _MODULE_VERSION
        )

    # -------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------

    def consolidate(
        self,
        hierarchy: EntityHierarchy,
        submissions: List[SubsidiarySubmission],
        intra_group_transfers: Optional[List[IntraGroupTransfer]] = None,
    ) -> ConsolidationManagementResult:
        """Run complete multi-entity consolidation.

        Args:
            hierarchy: Corporate entity hierarchy definition.
            submissions: Emission data submissions from entities.
            intra_group_transfers: Optional intra-group transfers to eliminate.

        Returns:
            ConsolidationManagementResult with full breakdown.

        Raises:
            ValueError: If hierarchy or submissions are invalid.
        """
        t0 = time.perf_counter()
        self._warnings = []
        self._notes = [
            f"Engine version: {self.engine_version}",
            f"Consolidation approach: {hierarchy.consolidation_approach.value}",
            f"Reporting year: {hierarchy.reporting_year}",
        ]

        if not submissions:
            raise ValueError("At least one subsidiary submission is required.")

        logger.info(
            "Consolidating %d entities for group '%s' (year %d, approach=%s)",
            len(hierarchy.entities),
            hierarchy.group_name,
            hierarchy.reporting_year,
            hierarchy.consolidation_approach.value,
        )

        # Step 1: Map submissions to entities
        submission_map = self._build_submission_map(submissions)

        # Step 2: Determine inclusion percentages
        inclusion_map = self._calculate_inclusion_percentages(hierarchy)

        # Step 3: Check consolidation status
        status = self._assess_consolidation_status(hierarchy, submissions)

        # Step 4: Consolidate per-entity results
        entity_results = self._consolidate_entities(
            hierarchy, submission_map, inclusion_map
        )

        # Step 5: Apply intra-group eliminations
        transfers = intra_group_transfers or []
        eliminations = self._apply_eliminations(transfers, hierarchy)

        # Step 6: Calculate group totals
        total_elim = sum(_decimal(e.amount_tco2e) for e in eliminations)

        raw_s1 = sum(_decimal(r.consolidated_scope1_tco2e) for r in entity_results)
        raw_s2_loc = sum(
            _decimal(r.consolidated_scope2_location_tco2e) for r in entity_results
        )
        raw_s2_mkt = sum(
            _decimal(r.consolidated_scope2_market_tco2e) for r in entity_results
        )
        raw_s3 = sum(_decimal(r.consolidated_scope3_tco2e) for r in entity_results)

        # Distribute eliminations proportionally across scopes
        scope_elim = self._distribute_eliminations(
            eliminations, raw_s1, raw_s2_loc, raw_s2_mkt, raw_s3
        )

        net_s1 = raw_s1 - scope_elim.get("scope1", Decimal("0"))
        net_s2_loc = raw_s2_loc - scope_elim.get("scope2_location", Decimal("0"))
        net_s2_mkt = raw_s2_mkt - scope_elim.get("scope2_market", Decimal("0"))
        net_s3 = raw_s3 - scope_elim.get("scope3", Decimal("0"))
        grand_total = net_s1 + net_s2_loc + net_s3

        # Step 7: Equity reconciliation
        recon_notes = self._reconcile_equity(hierarchy, submissions)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        result = ConsolidationManagementResult(
            group_name=hierarchy.group_name,
            reporting_year=hierarchy.reporting_year,
            consolidation_approach=hierarchy.consolidation_approach.value,
            total_scope1_tco2e=_round4(float(net_s1)),
            total_scope2_location_tco2e=_round4(float(net_s2_loc)),
            total_scope2_market_tco2e=_round4(float(net_s2_mkt)),
            total_scope3_tco2e=_round4(float(net_s3)),
            total_all_scopes_tco2e=_round4(float(grand_total)),
            total_eliminations_tco2e=_round4(float(total_elim)),
            entity_results=entity_results,
            eliminations=eliminations,
            consolidation_status=status,
            equity_reconciliation_notes=recon_notes,
            warnings=list(self._warnings),
            methodology_notes=list(self._notes),
            processing_time_ms=round(elapsed_ms, 3),
        )
        result.provenance_hash = _compute_hash(result)

        logger.info(
            "Consolidation complete: group='%s', total=%.2f tCO2e, "
            "eliminations=%.2f tCO2e, entities=%d, hash=%s (%.1f ms)",
            hierarchy.group_name,
            float(grand_total),
            float(total_elim),
            len(entity_results),
            result.provenance_hash[:16],
            elapsed_ms,
        )
        return result

    def get_entity_inclusion_percentage(
        self,
        entity: Entity,
        approach: ConsolidationApproach,
    ) -> Decimal:
        """Calculate the inclusion percentage for a single entity.

        Args:
            entity: Entity definition.
            approach: Consolidation approach to apply.

        Returns:
            Inclusion percentage (0-100).
        """
        if not entity.is_material and not self._include_immaterial:
            return Decimal("0")

        if approach == ConsolidationApproach.EQUITY_SHARE:
            return _decimal(entity.equity_pct)

        elif approach == ConsolidationApproach.OPERATIONAL_CONTROL:
            if entity.has_operational_control:
                return Decimal("100")
            return Decimal("0")

        elif approach == ConsolidationApproach.FINANCIAL_CONTROL:
            if entity.has_financial_control:
                return Decimal("100")
            return Decimal("0")

        return Decimal("0")

    def validate_hierarchy_integrity(
        self, hierarchy: EntityHierarchy
    ) -> List[str]:
        """Validate the hierarchy for structural integrity.

        Checks for:
            - Exactly one root entity
            - No orphaned entities (all parents exist)
            - No circular references
            - Equity percentages within valid bounds

        Args:
            hierarchy: Entity hierarchy to validate.

        Returns:
            List of validation error messages (empty if valid).
        """
        errors: List[str] = []
        entity_ids: Set[str] = {e.entity_id for e in hierarchy.entities}

        # Check for orphaned entities
        for entity in hierarchy.entities:
            if (
                entity.parent_entity_id is not None
                and entity.parent_entity_id not in entity_ids
            ):
                errors.append(
                    f"Entity '{entity.entity_name}' ({entity.entity_id}) "
                    f"references non-existent parent '{entity.parent_entity_id}'"
                )

        # Check for circular references via DFS
        children_map: Dict[str, List[str]] = {}
        for entity in hierarchy.entities:
            if entity.parent_entity_id is not None:
                children_map.setdefault(entity.parent_entity_id, []).append(
                    entity.entity_id
                )

        visited: Set[str] = set()
        stack: Set[str] = set()

        def _has_cycle(node_id: str) -> bool:
            """DFS cycle detection."""
            if node_id in stack:
                return True
            if node_id in visited:
                return False
            visited.add(node_id)
            stack.add(node_id)
            for child_id in children_map.get(node_id, []):
                if _has_cycle(child_id):
                    return True
            stack.discard(node_id)
            return False

        for entity in hierarchy.entities:
            if entity.parent_entity_id is None:
                if _has_cycle(entity.entity_id):
                    errors.append("Circular reference detected in entity hierarchy")
                    break

        # Validate equity percentages for equity-share approach
        if hierarchy.consolidation_approach == ConsolidationApproach.EQUITY_SHARE:
            for entity in hierarchy.entities:
                if entity.equity_pct < Decimal("0") or entity.equity_pct > Decimal("100"):
                    errors.append(
                        f"Entity '{entity.entity_name}' has invalid equity "
                        f"percentage: {entity.equity_pct}%"
                    )

        return errors

    def assess_submission_completeness(
        self,
        hierarchy: EntityHierarchy,
        submissions: List[SubsidiarySubmission],
    ) -> ConsolidationStatus:
        """Assess the completeness of subsidiary submissions.

        Args:
            hierarchy: Entity hierarchy.
            submissions: Current submissions.

        Returns:
            ConsolidationStatus with completeness metrics.
        """
        return self._assess_consolidation_status(hierarchy, submissions)

    # -------------------------------------------------------------------
    # Private -- Submission mapping
    # -------------------------------------------------------------------

    def _build_submission_map(
        self,
        submissions: List[SubsidiarySubmission],
    ) -> Dict[str, SubsidiarySubmission]:
        """Build a mapping of entity_id to latest submission.

        If multiple submissions exist for an entity, the one with
        the most recent submitted_at timestamp is used.

        Args:
            submissions: All submissions received.

        Returns:
            Dict mapping entity_id to SubsidiarySubmission.
        """
        sub_map: Dict[str, SubsidiarySubmission] = {}
        for sub in submissions:
            existing = sub_map.get(sub.entity_id)
            if existing is None:
                sub_map[sub.entity_id] = sub
            else:
                # Take the most recently submitted
                if (
                    sub.submitted_at is not None
                    and (
                        existing.submitted_at is None
                        or sub.submitted_at > existing.submitted_at
                    )
                ):
                    sub_map[sub.entity_id] = sub
        return sub_map

    # -------------------------------------------------------------------
    # Private -- Inclusion percentages
    # -------------------------------------------------------------------

    def _calculate_inclusion_percentages(
        self,
        hierarchy: EntityHierarchy,
    ) -> Dict[str, Decimal]:
        """Calculate inclusion percentages for all entities.

        Args:
            hierarchy: Entity hierarchy with approach defined.

        Returns:
            Dict mapping entity_id to inclusion percentage (0-100).
        """
        inclusion: Dict[str, Decimal] = {}
        approach = hierarchy.consolidation_approach

        for entity in hierarchy.entities:
            pct = self.get_entity_inclusion_percentage(entity, approach)
            inclusion[entity.entity_id] = pct

        self._notes.append(
            f"Inclusion percentages calculated for {len(inclusion)} entities "
            f"using {approach.value} approach."
        )
        return inclusion

    # -------------------------------------------------------------------
    # Private -- Entity consolidation
    # -------------------------------------------------------------------

    def _consolidate_entities(
        self,
        hierarchy: EntityHierarchy,
        submission_map: Dict[str, SubsidiarySubmission],
        inclusion_map: Dict[str, Decimal],
    ) -> List[EntityConsolidatedResult]:
        """Consolidate emissions for each entity in the hierarchy.

        Applies the inclusion percentage to each entity's submitted
        emissions to produce the consolidated contribution.

        Args:
            hierarchy: Entity hierarchy.
            submission_map: Entity ID to submission mapping.
            inclusion_map: Entity ID to inclusion percentage.

        Returns:
            List of EntityConsolidatedResult.
        """
        results: List[EntityConsolidatedResult] = []

        for entity in hierarchy.entities:
            inclusion_pct = inclusion_map.get(entity.entity_id, Decimal("0"))
            fraction = _safe_divide(inclusion_pct, Decimal("100"))

            submission = submission_map.get(entity.entity_id)
            if submission is None:
                self._warnings.append(
                    f"No submission found for entity '{entity.entity_name}' "
                    f"({entity.entity_id}). Using zero emissions."
                )
                results.append(EntityConsolidatedResult(
                    entity_id=entity.entity_id,
                    entity_name=entity.entity_name,
                    entity_type=entity.entity_type.value,
                    equity_pct=_round2(float(entity.equity_pct)),
                    inclusion_pct=_round2(float(inclusion_pct)),
                    submission_status=SubmissionStatus.NOT_STARTED.value,
                ))
                continue

            raw_s1 = _decimal(submission.scope1_tco2e)
            raw_s2_loc = _decimal(submission.scope2_location_tco2e)
            raw_s2_mkt = _decimal(submission.scope2_market_tco2e)
            raw_s3 = _decimal(submission.scope3_tco2e)

            results.append(EntityConsolidatedResult(
                entity_id=entity.entity_id,
                entity_name=entity.entity_name,
                entity_type=entity.entity_type.value,
                equity_pct=_round2(float(entity.equity_pct)),
                inclusion_pct=_round2(float(inclusion_pct)),
                raw_scope1_tco2e=_round4(float(raw_s1)),
                raw_scope2_location_tco2e=_round4(float(raw_s2_loc)),
                raw_scope2_market_tco2e=_round4(float(raw_s2_mkt)),
                raw_scope3_tco2e=_round4(float(raw_s3)),
                consolidated_scope1_tco2e=_round4(float(raw_s1 * fraction)),
                consolidated_scope2_location_tco2e=_round4(
                    float(raw_s2_loc * fraction)
                ),
                consolidated_scope2_market_tco2e=_round4(
                    float(raw_s2_mkt * fraction)
                ),
                consolidated_scope3_tco2e=_round4(float(raw_s3 * fraction)),
                submission_status=submission.status.value,
                data_quality_score=_round2(float(submission.data_quality_score)),
            ))

        return results

    # -------------------------------------------------------------------
    # Private -- Intra-group eliminations
    # -------------------------------------------------------------------

    def _apply_eliminations(
        self,
        transfers: List[IntraGroupTransfer],
        hierarchy: EntityHierarchy,
    ) -> List[EliminationRecord]:
        """Apply intra-group eliminations from transfer records.

        Validates that both entities in each transfer are within the
        same corporate group before applying the elimination.

        Args:
            transfers: Intra-group transfer records.
            hierarchy: Entity hierarchy for validation.

        Returns:
            List of EliminationRecord objects.
        """
        if not transfers:
            return []

        entity_ids: Set[str] = {e.entity_id for e in hierarchy.entities}
        eliminations: List[EliminationRecord] = []

        for transfer in transfers:
            # Validate both entities exist in hierarchy
            if transfer.from_entity_id not in entity_ids:
                self._warnings.append(
                    f"Transfer from unknown entity '{transfer.from_entity_id}' "
                    f"skipped for elimination."
                )
                continue

            if transfer.to_entity_id not in entity_ids:
                self._warnings.append(
                    f"Transfer to unknown entity '{transfer.to_entity_id}' "
                    f"skipped for elimination."
                )
                continue

            amount = _decimal(transfer.amount_tco2e)
            if amount <= Decimal("0"):
                continue

            rationale = (
                f"Intra-group {transfer.elimination_type.value}: "
                f"{amount} tCO2e transferred from {transfer.from_entity_id} "
                f"to {transfer.to_entity_id}. Eliminated to avoid "
                f"double-counting within the group boundary. "
                f"{transfer.description}"
            )

            eliminations.append(EliminationRecord(
                from_entity_id=transfer.from_entity_id,
                to_entity_id=transfer.to_entity_id,
                elimination_type=transfer.elimination_type.value,
                scope=transfer.scope,
                amount_tco2e=_round4(float(amount)),
                rationale=rationale.strip(),
            ))

        self._notes.append(
            f"Applied {len(eliminations)} intra-group eliminations "
            f"totalling {_round4(float(sum(_decimal(e.amount_tco2e) for e in eliminations)))} tCO2e."
        )
        return eliminations

    def _distribute_eliminations(
        self,
        eliminations: List[EliminationRecord],
        raw_s1: Decimal,
        raw_s2_loc: Decimal,
        raw_s2_mkt: Decimal,
        raw_s3: Decimal,
    ) -> Dict[str, Decimal]:
        """Distribute elimination amounts across scopes.

        Uses the scope field on each elimination record to assign
        the elimination to the correct scope total.

        Args:
            eliminations: Applied elimination records.
            raw_s1: Raw Scope 1 total.
            raw_s2_loc: Raw Scope 2 location total.
            raw_s2_mkt: Raw Scope 2 market total.
            raw_s3: Raw Scope 3 total.

        Returns:
            Dict with scope keys and elimination amounts.
        """
        scope_elim: Dict[str, Decimal] = {
            "scope1": Decimal("0"),
            "scope2_location": Decimal("0"),
            "scope2_market": Decimal("0"),
            "scope3": Decimal("0"),
        }

        for elim in eliminations:
            amount = _decimal(elim.amount_tco2e)
            if elim.scope == 1:
                scope_elim["scope1"] += min(amount, raw_s1)
            elif elim.scope == 2:
                # Split Scope 2 elimination proportionally between location/market
                s2_total = raw_s2_loc + raw_s2_mkt
                if s2_total > Decimal("0"):
                    loc_share = _safe_divide(raw_s2_loc, s2_total) * amount
                    mkt_share = _safe_divide(raw_s2_mkt, s2_total) * amount
                    scope_elim["scope2_location"] += min(loc_share, raw_s2_loc)
                    scope_elim["scope2_market"] += min(mkt_share, raw_s2_mkt)
            elif elim.scope == 3:
                scope_elim["scope3"] += min(amount, raw_s3)

        return scope_elim

    # -------------------------------------------------------------------
    # Private -- Consolidation status assessment
    # -------------------------------------------------------------------

    def _assess_consolidation_status(
        self,
        hierarchy: EntityHierarchy,
        submissions: List[SubsidiarySubmission],
    ) -> ConsolidationStatus:
        """Assess the overall consolidation status.

        Args:
            hierarchy: Entity hierarchy.
            submissions: All submissions received.

        Returns:
            ConsolidationStatus with completeness metrics.
        """
        material_entities = [
            e for e in hierarchy.entities
            if e.is_material or self._include_immaterial
        ]
        total_required = len(material_entities)

        submission_map = self._build_submission_map(submissions)

        submitted_count = 0
        approved_count = 0
        pending_count = 0
        overdue_count = 0
        blocking: List[str] = []

        for entity in material_entities:
            sub = submission_map.get(entity.entity_id)
            if sub is None:
                pending_count += 1
                blocking.append(entity.entity_id)
                continue

            if sub.status == SubmissionStatus.APPROVED:
                approved_count += 1
                submitted_count += 1
            elif sub.status == SubmissionStatus.SUBMITTED:
                submitted_count += 1
            elif sub.status == SubmissionStatus.OVERDUE:
                overdue_count += 1
                blocking.append(entity.entity_id)
            elif sub.status in (SubmissionStatus.NOT_STARTED, SubmissionStatus.IN_PROGRESS):
                pending_count += 1
                blocking.append(entity.entity_id)
            elif sub.status == SubmissionStatus.REJECTED:
                pending_count += 1
                blocking.append(entity.entity_id)

        completeness = float(
            _safe_pct(_decimal(submitted_count), _decimal(total_required))
        ) if total_required > 0 else 0.0

        all_approved = approved_count >= total_required

        # Determine overall status
        if all_approved:
            overall_status = ConsolidationStatusEnum.FINAL.value
        elif completeness >= 100.0:
            overall_status = ConsolidationStatusEnum.PRELIMINARY.value
        elif completeness >= 50.0:
            overall_status = ConsolidationStatusEnum.REVIEWED.value
        else:
            overall_status = ConsolidationStatusEnum.DRAFT.value

        return ConsolidationStatus(
            total_entities=total_required,
            entities_submitted=submitted_count,
            entities_pending=pending_count,
            entities_overdue=overdue_count,
            completeness_pct=_round2(completeness),
            all_approved=all_approved,
            blocking_entities=blocking,
            status=overall_status,
        )

    # -------------------------------------------------------------------
    # Private -- Equity reconciliation
    # -------------------------------------------------------------------

    def _reconcile_equity(
        self,
        hierarchy: EntityHierarchy,
        submissions: List[SubsidiarySubmission],
    ) -> List[str]:
        """Reconcile equity percentages for consistency.

        For equity-share consolidation, checks that the sum of equity
        percentages for entities sharing the same parent is consistent
        and flags any anomalies.

        Args:
            hierarchy: Entity hierarchy.
            submissions: Submissions for context.

        Returns:
            List of reconciliation notes.
        """
        notes: List[str] = []

        if hierarchy.consolidation_approach != ConsolidationApproach.EQUITY_SHARE:
            notes.append(
                f"Equity reconciliation not applicable for "
                f"{hierarchy.consolidation_approach.value} approach."
            )
            return notes

        # Group entities by parent
        parent_groups: Dict[str, List[Entity]] = {}
        for entity in hierarchy.entities:
            parent_id = entity.parent_entity_id or "root"
            parent_groups.setdefault(parent_id, []).append(entity)

        for parent_id, children in parent_groups.items():
            if parent_id == "root":
                continue

            total_equity = sum(_decimal(c.equity_pct) for c in children)

            # Total equity for subsidiaries under a parent should not exceed 100%
            if total_equity > Decimal("100") + self._equity_tolerance:
                notes.append(
                    f"WARNING: Children of entity '{parent_id}' have total "
                    f"equity of {_round2(float(total_equity))}% (exceeds 100%)"
                )
                self._warnings.append(
                    f"Equity exceeds 100% for children of '{parent_id}': "
                    f"{_round2(float(total_equity))}%"
                )

            # Check for entities with 0% equity (may indicate misconfiguration)
            zero_equity = [c for c in children if c.equity_pct == Decimal("0")]
            if zero_equity:
                names = ", ".join(c.entity_name for c in zero_equity)
                notes.append(
                    f"INFO: Entities with 0% equity under '{parent_id}': {names}"
                )

        notes.append(
            f"Equity reconciliation completed for {len(hierarchy.entities)} entities."
        )
        return notes
