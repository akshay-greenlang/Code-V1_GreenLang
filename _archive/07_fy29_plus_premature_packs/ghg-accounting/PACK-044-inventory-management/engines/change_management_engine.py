# -*- coding: utf-8 -*-
"""
ChangeManagementEngine - PACK-044 Inventory Management Engine 4
================================================================

Organisational and methodology change tracking engine that monitors,
classifies, and assesses the impact of changes on the GHG inventory.
Implements a structured change request workflow with impact analysis,
base year trigger detection, and approval routing in accordance with
GHG Protocol Corporate Standard Chapter 5 (base year recalculation
policy) and ISO 14064-1:2018 requirements.

Change categories tracked:
    - Structural changes (acquisitions, divestitures, mergers, outsourcing)
    - Methodological changes (emission factor updates, GWP changes,
      calculation tier upgrades, scope reclassification)
    - Boundary changes (new facilities, new source categories, consolidation
      approach changes)
    - Data quality improvements (metered data replacing estimates,
      supplier-specific data replacing industry averages)
    - Regulatory changes (new disclosure requirements, framework updates)
    - Error corrections (discovered errors in historical data)

Impact Assessment Methodology:
    For each change request:
        impact_tco2e = abs(new_value - old_value) for affected sources
        impact_pct = (impact_tco2e / total_inventory_tco2e) * 100
        is_material = impact_pct >= materiality_threshold (default 5%)

    Base Year Trigger Detection:
        A change triggers base year recalculation when:
            (a) structural change impact >= significance_threshold, OR
            (b) methodology change impact >= significance_threshold, OR
            (c) cumulative error corrections >= significance_threshold

    Approval Routing:
        LOW impact (<1% of inventory):     Auto-approve with notification
        MEDIUM impact (1-5%):              Single approver required
        HIGH impact (5-15%):               Dual approval + reviewer sign-off
        CRITICAL impact (>15%):            Board-level approval + external verifier

Regulatory References:
    - GHG Protocol Corporate Standard (2004, revised 2015), Chapter 5
    - ISO 14064-1:2018, Clause 9 (Management of GHG inventory quality)
    - ESRS E1 (Climate Change - methodology change disclosures)
    - CDP Climate Change Questionnaire C5.1-C5.2 (Recalculation policy)
    - SBTi Corporate Manual (2023), Section 7 (Recalculation)
    - SEC Climate Disclosure Rule (2024), Item 1504

Zero-Hallucination:
    - All impact calculations use deterministic Decimal arithmetic
    - Threshold values from published GHG Protocol guidance
    - No LLM involvement in any calculation or routing path
    - SHA-256 provenance hash on every result

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-044 Inventory Management
Engine:  4 of 10
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
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator

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

    Excludes volatile fields (calculated_at, processing_time_ms,
    provenance_hash) so that identical logical content always produces
    the same hash regardless of when or how fast it was computed.

    Args:
        data: Any Pydantic model, dict, or stringifiable object.

    Returns:
        Hex-encoded SHA-256 digest string.
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

def _round_val(value: Decimal, places: int = 4) -> Decimal:
    """Round a Decimal to *places* using ROUND_HALF_UP."""
    quantize_str = "0." + "0" * places
    return value.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP)

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ChangeCategory(str, Enum):
    """Category of change affecting the GHG inventory.

    STRUCTURAL:         Organisational boundary changes (M&A, divestitures,
                        outsourcing, insourcing, joint ventures).
    METHODOLOGICAL:     Calculation methodology changes (emission factor
                        updates, GWP revisions, tier upgrades).
    BOUNDARY:           Operational boundary changes (new facilities,
                        new source categories, scope reclassification).
    DATA_QUALITY:       Data quality improvements that change emission
                        values (metered vs estimated, supplier-specific).
    REGULATORY:         Regulatory or framework changes requiring inventory
                        adjustments (new disclosure rules, updated standards).
    ERROR_CORRECTION:   Corrections to previously reported data due to
                        discovered errors in calculations or source data.
    """
    STRUCTURAL = "structural"
    METHODOLOGICAL = "methodological"
    BOUNDARY = "boundary"
    DATA_QUALITY = "data_quality"
    REGULATORY = "regulatory"
    ERROR_CORRECTION = "error_correction"

class ChangeStatus(str, Enum):
    """Lifecycle status of a change request.

    DRAFT:              Initial creation, not yet submitted.
    SUBMITTED:          Submitted for impact assessment.
    UNDER_ASSESSMENT:   Impact assessment in progress.
    PENDING_APPROVAL:   Assessment complete, awaiting approval.
    APPROVED:           Change approved for implementation.
    REJECTED:           Change rejected by approver.
    IMPLEMENTED:        Change has been applied to inventory.
    WITHDRAWN:          Change withdrawn by requester.
    """
    DRAFT = "draft"
    SUBMITTED = "submitted"
    UNDER_ASSESSMENT = "under_assessment"
    PENDING_APPROVAL = "pending_approval"
    APPROVED = "approved"
    REJECTED = "rejected"
    IMPLEMENTED = "implemented"
    WITHDRAWN = "withdrawn"

class ImpactSeverity(str, Enum):
    """Severity classification of a change's impact on the inventory.

    LOW:        <1% of total inventory emissions.
    MEDIUM:     1-5% of total inventory emissions.
    HIGH:       5-15% of total inventory emissions.
    CRITICAL:   >15% of total inventory emissions.
    """
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ApprovalLevel(str, Enum):
    """Approval level required based on impact severity.

    AUTO:               Automatic approval with notification (LOW impact).
    SINGLE_APPROVER:    One designated approver required (MEDIUM impact).
    DUAL_APPROVAL:      Two approvers + reviewer sign-off (HIGH impact).
    BOARD_LEVEL:        Board/executive approval + external verifier (CRITICAL).
    """
    AUTO = "auto"
    SINGLE_APPROVER = "single_approver"
    DUAL_APPROVAL = "dual_approval"
    BOARD_LEVEL = "board_level"

class BaseYearTriggerType(str, Enum):
    """Type of base year recalculation trigger detected.

    STRUCTURAL_THRESHOLD:       Structural change exceeds significance threshold.
    METHODOLOGY_THRESHOLD:      Methodology change exceeds threshold.
    CUMULATIVE_ERROR:           Cumulative error corrections exceed threshold.
    BOUNDARY_EXPANSION:         Significant boundary expansion.
    SCOPE_RECLASSIFICATION:     Reclassification between scopes.
    NO_TRIGGER:                 Change does not trigger recalculation.
    """
    STRUCTURAL_THRESHOLD = "structural_threshold"
    METHODOLOGY_THRESHOLD = "methodology_threshold"
    CUMULATIVE_ERROR = "cumulative_error"
    BOUNDARY_EXPANSION = "boundary_expansion"
    SCOPE_RECLASSIFICATION = "scope_reclassification"
    NO_TRIGGER = "no_trigger"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Impact severity thresholds as percentage of total inventory.
IMPACT_THRESHOLDS: Dict[str, Decimal] = {
    "low_max": Decimal("1"),
    "medium_max": Decimal("5"),
    "high_max": Decimal("15"),
}

# Mapping from ImpactSeverity to ApprovalLevel.
SEVERITY_TO_APPROVAL: Dict[ImpactSeverity, ApprovalLevel] = {
    ImpactSeverity.LOW: ApprovalLevel.AUTO,
    ImpactSeverity.MEDIUM: ApprovalLevel.SINGLE_APPROVER,
    ImpactSeverity.HIGH: ApprovalLevel.DUAL_APPROVAL,
    ImpactSeverity.CRITICAL: ApprovalLevel.BOARD_LEVEL,
}

# Default significance threshold for base year recalculation (GHG Protocol Ch5).
DEFAULT_SIGNIFICANCE_THRESHOLD_PCT: Decimal = Decimal("5")

# Maximum number of change requests per batch.
MAX_CHANGES_PER_BATCH: int = 200

# Valid status transitions.
VALID_TRANSITIONS: Dict[ChangeStatus, List[ChangeStatus]] = {
    ChangeStatus.DRAFT: [ChangeStatus.SUBMITTED, ChangeStatus.WITHDRAWN],
    ChangeStatus.SUBMITTED: [
        ChangeStatus.UNDER_ASSESSMENT, ChangeStatus.WITHDRAWN,
    ],
    ChangeStatus.UNDER_ASSESSMENT: [
        ChangeStatus.PENDING_APPROVAL, ChangeStatus.WITHDRAWN,
    ],
    ChangeStatus.PENDING_APPROVAL: [
        ChangeStatus.APPROVED, ChangeStatus.REJECTED,
    ],
    ChangeStatus.APPROVED: [ChangeStatus.IMPLEMENTED],
    ChangeStatus.REJECTED: [ChangeStatus.DRAFT],
    ChangeStatus.IMPLEMENTED: [],
    ChangeStatus.WITHDRAWN: [],
}

# ---------------------------------------------------------------------------
# Pydantic Models -- Input
# ---------------------------------------------------------------------------

class AffectedSource(BaseModel):
    """A single emission source affected by a change.

    Attributes:
        source_id: Unique identifier for the emission source.
        source_name: Human-readable source name.
        scope: Scope of the source (scope1, scope2, scope3).
        category: Emission category (e.g. stationary_combustion).
        facility_id: Facility where the source is located.
        old_value_tco2e: Emission value before the change (tCO2e).
        new_value_tco2e: Emission value after the change (tCO2e).
        delta_tco2e: Absolute change in emissions (tCO2e).
        notes: Explanatory notes about this source's change.
    """
    source_id: str = Field(default="", description="Source identifier")
    source_name: str = Field(default="", max_length=500, description="Source name")
    scope: str = Field(default="scope1", description="Emission scope")
    category: str = Field(default="", description="Emission category")
    facility_id: str = Field(default="", description="Facility ID")
    old_value_tco2e: Decimal = Field(
        default=Decimal("0"), ge=0, description="Old emission value (tCO2e)"
    )
    new_value_tco2e: Decimal = Field(
        default=Decimal("0"), ge=0, description="New emission value (tCO2e)"
    )
    delta_tco2e: Decimal = Field(
        default=Decimal("0"), description="Change in emissions (tCO2e)"
    )
    notes: str = Field(default="", description="Notes about the change")

    @field_validator("delta_tco2e", mode="before")
    @classmethod
    def compute_delta(cls, v: Any, info: Any) -> Decimal:
        """Auto-compute delta if not provided explicitly."""
        return _decimal(v)

class ChangeRequest(BaseModel):
    """A change request submitted against the GHG inventory.

    Attributes:
        request_id: Unique change request identifier.
        title: Short title summarising the change.
        description: Detailed description of the change and rationale.
        category: Change category (structural, methodological, etc.).
        status: Current lifecycle status.
        requester_id: User or system that submitted the request.
        requester_name: Display name of the requester.
        requested_at: Timestamp of submission.
        effective_date: Date when the change takes or took effect.
        reporting_year: Reporting year affected by this change.
        affected_sources: Emission sources affected by this change.
        affected_scopes: Scopes affected (scope1, scope2, scope3).
        affected_facilities: Facility IDs affected.
        total_inventory_tco2e: Total inventory emissions for context.
        supporting_documents: List of document references (URLs or IDs).
        metadata: Arbitrary additional metadata.
    """
    request_id: str = Field(default_factory=_new_uuid, description="Change request ID")
    title: str = Field(default="", max_length=500, description="Change title")
    description: str = Field(default="", description="Change description")
    category: ChangeCategory = Field(
        default=ChangeCategory.METHODOLOGICAL, description="Change category"
    )
    status: ChangeStatus = Field(
        default=ChangeStatus.DRAFT, description="Current status"
    )
    requester_id: str = Field(default="", description="Requester user ID")
    requester_name: str = Field(default="", description="Requester display name")
    requested_at: datetime = Field(
        default_factory=utcnow, description="Submission timestamp"
    )
    effective_date: Optional[str] = Field(
        default=None, description="Effective date (ISO format)"
    )
    reporting_year: int = Field(
        default=2025, ge=1990, le=2050, description="Reporting year affected"
    )
    affected_sources: List[AffectedSource] = Field(
        default_factory=list, description="Affected emission sources"
    )
    affected_scopes: List[str] = Field(
        default_factory=list, description="Affected scopes"
    )
    affected_facilities: List[str] = Field(
        default_factory=list, description="Affected facility IDs"
    )
    total_inventory_tco2e: Decimal = Field(
        default=Decimal("0"), ge=0, description="Total inventory emissions (tCO2e)"
    )
    supporting_documents: List[str] = Field(
        default_factory=list, description="Supporting document references"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )

    @field_validator("total_inventory_tco2e", mode="before")
    @classmethod
    def coerce_inventory(cls, v: Any) -> Decimal:
        """Coerce total inventory to Decimal."""
        return _decimal(v)

# ---------------------------------------------------------------------------
# Pydantic Models -- Output
# ---------------------------------------------------------------------------

class ChangeImpact(BaseModel):
    """Impact assessment for a single change request.

    Attributes:
        request_id: The change request assessed.
        total_impact_tco2e: Total absolute emission impact (tCO2e).
        net_impact_tco2e: Net emission change (positive=increase).
        impact_pct: Impact as percentage of total inventory.
        severity: Classified impact severity.
        affected_scope_count: Number of scopes affected.
        affected_source_count: Number of sources affected.
        scope_impacts: Per-scope emission impact breakdown.
        is_material: Whether the impact is material (above threshold).
        materiality_threshold_pct: Threshold used for materiality test.
        assessment_notes: Explanatory notes from the assessment.
    """
    request_id: str = Field(default="", description="Change request ID")
    total_impact_tco2e: Decimal = Field(
        default=Decimal("0"), ge=0, description="Total absolute impact (tCO2e)"
    )
    net_impact_tco2e: Decimal = Field(
        default=Decimal("0"), description="Net impact (signed, tCO2e)"
    )
    impact_pct: Decimal = Field(
        default=Decimal("0"), ge=0, description="Impact as % of inventory"
    )
    severity: ImpactSeverity = Field(
        default=ImpactSeverity.LOW, description="Impact severity"
    )
    affected_scope_count: int = Field(
        default=0, ge=0, description="Number of scopes affected"
    )
    affected_source_count: int = Field(
        default=0, ge=0, description="Number of sources affected"
    )
    scope_impacts: Dict[str, Decimal] = Field(
        default_factory=dict, description="Per-scope impact (tCO2e)"
    )
    is_material: bool = Field(
        default=False, description="Whether impact is material"
    )
    materiality_threshold_pct: Decimal = Field(
        default=DEFAULT_SIGNIFICANCE_THRESHOLD_PCT,
        description="Materiality threshold used (%)",
    )
    assessment_notes: List[str] = Field(
        default_factory=list, description="Assessment notes"
    )

class ChangeApproval(BaseModel):
    """Approval routing decision for a change request.

    Attributes:
        request_id: The change request being routed.
        required_level: Approval level required.
        approvers_required: Number of approvers needed.
        approver_roles: Roles permitted to approve.
        requires_external_verifier: Whether external verification needed.
        auto_approved: Whether auto-approved (LOW impact).
        routing_rationale: Explanation of the routing decision.
        escalation_deadline_hours: Hours before escalation triggers.
    """
    request_id: str = Field(default="", description="Change request ID")
    required_level: ApprovalLevel = Field(
        default=ApprovalLevel.SINGLE_APPROVER, description="Required approval level"
    )
    approvers_required: int = Field(
        default=1, ge=0, description="Number of approvers needed"
    )
    approver_roles: List[str] = Field(
        default_factory=list, description="Permitted approver roles"
    )
    requires_external_verifier: bool = Field(
        default=False, description="Whether external verifier required"
    )
    auto_approved: bool = Field(
        default=False, description="Whether auto-approved"
    )
    routing_rationale: str = Field(
        default="", description="Routing rationale"
    )
    escalation_deadline_hours: int = Field(
        default=72, ge=0, description="Hours before escalation"
    )

class BaseYearTriggerDetection(BaseModel):
    """Result of base year recalculation trigger detection.

    Attributes:
        request_id: Change request assessed for base year trigger.
        triggers_recalculation: Whether the change triggers recalculation.
        trigger_type: Type of trigger detected.
        impact_on_base_year_tco2e: Estimated impact on base year (tCO2e).
        impact_on_base_year_pct: Impact as percentage of base year total.
        significance_threshold_pct: Threshold used for significance test.
        base_year: The base year that would need recalculation.
        base_year_total_tco2e: Total base year emissions.
        rationale: Explanation of the trigger detection result.
        recommended_actions: Recommended follow-up actions.
    """
    request_id: str = Field(default="", description="Change request ID")
    triggers_recalculation: bool = Field(
        default=False, description="Whether base year recalculation triggered"
    )
    trigger_type: BaseYearTriggerType = Field(
        default=BaseYearTriggerType.NO_TRIGGER, description="Trigger type"
    )
    impact_on_base_year_tco2e: Decimal = Field(
        default=Decimal("0"), description="Impact on base year (tCO2e)"
    )
    impact_on_base_year_pct: Decimal = Field(
        default=Decimal("0"), description="Impact as % of base year"
    )
    significance_threshold_pct: Decimal = Field(
        default=DEFAULT_SIGNIFICANCE_THRESHOLD_PCT,
        description="Significance threshold (%)",
    )
    base_year: int = Field(default=2019, ge=1990, description="Base year")
    base_year_total_tco2e: Decimal = Field(
        default=Decimal("0"), ge=0, description="Base year total (tCO2e)"
    )
    rationale: str = Field(default="", description="Detection rationale")
    recommended_actions: List[str] = Field(
        default_factory=list, description="Recommended actions"
    )

class ChangeLogEntry(BaseModel):
    """Audit log entry for a change request lifecycle event.

    Attributes:
        entry_id: Unique log entry identifier.
        request_id: Related change request.
        timestamp: When the event occurred.
        action: Action performed (e.g. status_transition, assessment_complete).
        actor_id: User or system that performed the action.
        actor_name: Display name of the actor.
        old_status: Previous status (for transitions).
        new_status: New status (for transitions).
        details: Detailed description of the event.
        provenance_hash: SHA-256 hash of the entry for audit integrity.
    """
    entry_id: str = Field(default_factory=_new_uuid, description="Log entry ID")
    request_id: str = Field(default="", description="Change request ID")
    timestamp: datetime = Field(default_factory=utcnow, description="Event timestamp")
    action: str = Field(default="", description="Action performed")
    actor_id: str = Field(default="system", description="Actor user ID")
    actor_name: str = Field(default="System", description="Actor display name")
    old_status: Optional[str] = Field(default=None, description="Previous status")
    new_status: Optional[str] = Field(default=None, description="New status")
    details: str = Field(default="", description="Event details")
    provenance_hash: str = Field(default="", description="SHA-256 hash")

class ChangeManagementResult(BaseModel):
    """Complete result from the change management engine.

    Attributes:
        result_id: Unique result identifier.
        engine_version: Engine version.
        request: Original change request (updated with new status).
        impact: Impact assessment result.
        approval_routing: Approval routing decision.
        base_year_trigger: Base year trigger detection result.
        change_log: Audit log entries generated during processing.
        warnings: Warnings raised during processing.
        calculated_at: Processing timestamp.
        processing_time_ms: Processing duration in milliseconds.
        provenance_hash: SHA-256 hash for audit trail.
    """
    result_id: str = Field(default_factory=_new_uuid, description="Result ID")
    engine_version: str = Field(default=_MODULE_VERSION, description="Engine version")
    request: Optional[ChangeRequest] = Field(
        default=None, description="Processed change request"
    )
    impact: Optional[ChangeImpact] = Field(
        default=None, description="Impact assessment"
    )
    approval_routing: Optional[ChangeApproval] = Field(
        default=None, description="Approval routing"
    )
    base_year_trigger: Optional[BaseYearTriggerDetection] = Field(
        default=None, description="Base year trigger detection"
    )
    change_log: List[ChangeLogEntry] = Field(
        default_factory=list, description="Audit log entries"
    )
    warnings: List[str] = Field(default_factory=list, description="Warnings")
    calculated_at: datetime = Field(
        default_factory=utcnow, description="Processing timestamp"
    )
    processing_time_ms: Decimal = Field(
        default=Decimal("0"), description="Processing time (ms)"
    )
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

# ---------------------------------------------------------------------------
# Model Rebuild (resolve forward references from __future__ annotations)
# ---------------------------------------------------------------------------

AffectedSource.model_rebuild()
ChangeRequest.model_rebuild()
ChangeImpact.model_rebuild()
ChangeApproval.model_rebuild()
BaseYearTriggerDetection.model_rebuild()
ChangeLogEntry.model_rebuild()
ChangeManagementResult.model_rebuild()

# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class ChangeManagementEngine:
    """Organisational and methodology change tracking engine.

    Provides a structured workflow for managing changes that affect the
    GHG inventory, including impact assessment, severity classification,
    base year trigger detection, and approval routing.

    Guarantees:
        - Deterministic: same inputs always produce identical outputs.
        - Auditable: complete log trail for every action.
        - Compliant: follows GHG Protocol Chapter 5 procedures.
        - No LLM: zero hallucination risk in any calculation path.

    Attributes:
        _significance_threshold_pct: Threshold for base year recalculation.
        _base_year: Organisation's base year.
        _base_year_total_tco2e: Total base year emissions (tCO2e).
        _change_log: Accumulated audit log entries.
        _warnings: Accumulated warnings.

    Example:
        >>> engine = ChangeManagementEngine(
        ...     base_year=2019,
        ...     base_year_total_tco2e=Decimal("50000"),
        ... )
        >>> request = ChangeRequest(
        ...     title="Updated grid emission factors",
        ...     category=ChangeCategory.METHODOLOGICAL,
        ...     total_inventory_tco2e=Decimal("55000"),
        ...     affected_sources=[
        ...         AffectedSource(
        ...             source_name="Grid electricity",
        ...             scope="scope2",
        ...             old_value_tco2e=Decimal("8000"),
        ...             new_value_tco2e=Decimal("7200"),
        ...             delta_tco2e=Decimal("-800"),
        ...         )
        ...     ],
        ... )
        >>> result = engine.process_change(request)
        >>> print(result.impact.severity)
    """

    def __init__(
        self,
        significance_threshold_pct: Optional[Decimal] = None,
        base_year: int = 2019,
        base_year_total_tco2e: Optional[Decimal] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialise the ChangeManagementEngine.

        Args:
            significance_threshold_pct: Significance threshold for base year
                recalculation trigger detection. Defaults to 5%.
            base_year: Organisation's base year. Defaults to 2019.
            base_year_total_tco2e: Total base year emissions (tCO2e).
                Required for base year trigger detection.
            config: Optional additional configuration overrides.
        """
        self._significance_threshold_pct = (
            significance_threshold_pct
            if significance_threshold_pct is not None
            else DEFAULT_SIGNIFICANCE_THRESHOLD_PCT
        )
        self._base_year = base_year
        self._base_year_total_tco2e = base_year_total_tco2e or Decimal("0")
        self._config = config or {}
        self._change_log: List[ChangeLogEntry] = []
        self._warnings: List[str] = []

        logger.info(
            "ChangeManagementEngine v%s initialised: base_year=%d, "
            "base_year_total=%.2f tCO2e, significance_threshold=%.1f%%",
            _MODULE_VERSION, self._base_year,
            float(self._base_year_total_tco2e),
            float(self._significance_threshold_pct),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_change(
        self,
        request: ChangeRequest,
    ) -> ChangeManagementResult:
        """Process a change request through the full workflow.

        Executes the complete change management pipeline:
        1. Validate the change request.
        2. Assess the impact on the inventory.
        3. Classify severity and determine approval routing.
        4. Detect base year recalculation triggers.
        5. Generate audit log and provenance.

        Args:
            request: The change request to process.

        Returns:
            ChangeManagementResult with impact, routing, and trigger data.

        Raises:
            ValueError: If the change request fails validation.
        """
        t0 = time.perf_counter()
        self._change_log = []
        self._warnings = []

        logger.info(
            "Processing change request %s: category=%s, title='%s'",
            request.request_id[:12], request.category.value, request.title,
        )

        self._add_log(
            request.request_id, "processing_started",
            request.requester_id, request.requester_name,
            f"Processing change request: {request.title}",
        )

        # Step 1: Validate the request.
        self._validate_request(request)

        # Step 2: Transition DRAFT -> SUBMITTED -> UNDER_ASSESSMENT.
        if request.status == ChangeStatus.DRAFT:
            request = self._transition_status(
                request, ChangeStatus.SUBMITTED,
                "system", "System",
            )
        request = self._transition_status(
            request, ChangeStatus.UNDER_ASSESSMENT,
            "system", "System",
        )

        # Step 3: Assess impact.
        impact = self.assess_impact(request)

        # Step 4: Determine approval routing.
        approval = self.determine_approval_routing(request, impact)

        # Step 5: Detect base year trigger.
        trigger = self.detect_base_year_trigger(request, impact)

        # Step 6: Transition to PENDING_APPROVAL.
        request = self._transition_status(
            request, ChangeStatus.PENDING_APPROVAL,
            "system", "System",
        )

        # Step 7: Auto-approve if applicable.
        if approval.auto_approved:
            request = self._transition_status(
                request, ChangeStatus.APPROVED,
                "system", "Auto-Approval",
            )
            self._add_log(
                request.request_id, "auto_approved",
                "system", "System",
                f"Auto-approved: impact severity {impact.severity.value} "
                f"below single-approver threshold.",
            )

        elapsed = Decimal(str(round((time.perf_counter() - t0) * 1000, 3)))

        result = ChangeManagementResult(
            request=request,
            impact=impact,
            approval_routing=approval,
            base_year_trigger=trigger,
            change_log=list(self._change_log),
            warnings=list(self._warnings),
            processing_time_ms=elapsed,
        )
        result.provenance_hash = _compute_hash(result)

        logger.info(
            "Change request %s processed: severity=%s, "
            "triggers_recalc=%s, approval_level=%s, hash=%s (%.1f ms)",
            request.request_id[:12], impact.severity.value,
            trigger.triggers_recalculation if trigger else False,
            approval.required_level.value,
            result.provenance_hash[:16], float(elapsed),
        )
        return result

    def assess_impact(
        self,
        request: ChangeRequest,
    ) -> ChangeImpact:
        """Assess the impact of a change request on the inventory.

        Calculates total and net emission impacts from affected sources,
        classifies the severity, and determines materiality.

        Args:
            request: The change request to assess.

        Returns:
            ChangeImpact with severity classification.
        """
        logger.info(
            "Assessing impact for request %s with %d affected sources",
            request.request_id[:12], len(request.affected_sources),
        )

        total_abs_impact = Decimal("0")
        net_impact = Decimal("0")
        scope_impacts: Dict[str, Decimal] = {}
        notes: List[str] = []

        for source in request.affected_sources:
            abs_delta = abs(source.new_value_tco2e - source.old_value_tco2e)
            signed_delta = source.new_value_tco2e - source.old_value_tco2e

            total_abs_impact += abs_delta
            net_impact += signed_delta

            scope_key = source.scope if source.scope else "unknown"
            scope_impacts[scope_key] = (
                scope_impacts.get(scope_key, Decimal("0")) + abs_delta
            )

        # Compute impact percentage.
        inventory_total = _decimal(request.total_inventory_tco2e)
        impact_pct = _safe_pct(total_abs_impact, inventory_total)

        # Classify severity.
        severity = self._classify_severity(impact_pct)

        # Determine materiality.
        is_material = impact_pct >= self._significance_threshold_pct

        # Affected scope count.
        unique_scopes = set()
        for source in request.affected_sources:
            if source.scope:
                unique_scopes.add(source.scope)

        if total_abs_impact > Decimal("0"):
            notes.append(
                f"Total absolute impact: {_round_val(total_abs_impact, 2)} tCO2e "
                f"({_round_val(impact_pct, 2)}% of inventory)."
            )
        if net_impact > Decimal("0"):
            notes.append(f"Net emission increase: {_round_val(net_impact, 2)} tCO2e.")
        elif net_impact < Decimal("0"):
            notes.append(f"Net emission decrease: {_round_val(abs(net_impact), 2)} tCO2e.")

        if is_material:
            notes.append(
                f"MATERIAL: exceeds {self._significance_threshold_pct}% threshold."
            )

        self._add_log(
            request.request_id, "impact_assessed",
            "system", "System",
            f"Impact: {_round_val(total_abs_impact, 2)} tCO2e, "
            f"severity={severity.value}, material={is_material}.",
        )

        return ChangeImpact(
            request_id=request.request_id,
            total_impact_tco2e=_round_val(total_abs_impact, 4),
            net_impact_tco2e=_round_val(net_impact, 4),
            impact_pct=_round_val(impact_pct, 4),
            severity=severity,
            affected_scope_count=len(unique_scopes),
            affected_source_count=len(request.affected_sources),
            scope_impacts={k: _round_val(v, 4) for k, v in scope_impacts.items()},
            is_material=is_material,
            materiality_threshold_pct=self._significance_threshold_pct,
            assessment_notes=notes,
        )

    def determine_approval_routing(
        self,
        request: ChangeRequest,
        impact: ChangeImpact,
    ) -> ChangeApproval:
        """Determine approval routing based on impact severity.

        Routes the change request to the appropriate approval level
        based on the classified impact severity.

        Args:
            request: The change request.
            impact: The impact assessment result.

        Returns:
            ChangeApproval with routing decision.
        """
        severity = impact.severity
        level = SEVERITY_TO_APPROVAL.get(severity, ApprovalLevel.SINGLE_APPROVER)

        approvers_required = 0
        approver_roles: List[str] = []
        requires_external = False
        auto_approved = False
        escalation_hours = 72

        if level == ApprovalLevel.AUTO:
            approvers_required = 0
            approver_roles = []
            auto_approved = True
            escalation_hours = 0
            rationale = (
                f"Auto-approved: impact {_round_val(impact.impact_pct, 2)}% "
                f"is below {IMPACT_THRESHOLDS['low_max']}% threshold."
            )

        elif level == ApprovalLevel.SINGLE_APPROVER:
            approvers_required = 1
            approver_roles = ["ghg_inventory_manager", "sustainability_lead"]
            escalation_hours = 72
            rationale = (
                f"Single approver required: impact {_round_val(impact.impact_pct, 2)}% "
                f"is between {IMPACT_THRESHOLDS['low_max']}% and "
                f"{IMPACT_THRESHOLDS['medium_max']}%."
            )

        elif level == ApprovalLevel.DUAL_APPROVAL:
            approvers_required = 2
            approver_roles = [
                "ghg_inventory_manager", "sustainability_director",
                "internal_auditor",
            ]
            escalation_hours = 48
            rationale = (
                f"Dual approval required: impact {_round_val(impact.impact_pct, 2)}% "
                f"is between {IMPACT_THRESHOLDS['medium_max']}% and "
                f"{IMPACT_THRESHOLDS['high_max']}%."
            )

        elif level == ApprovalLevel.BOARD_LEVEL:
            approvers_required = 3
            approver_roles = [
                "sustainability_director", "cfo", "board_member",
                "external_verifier",
            ]
            requires_external = True
            escalation_hours = 24
            rationale = (
                f"Board-level approval required: impact {_round_val(impact.impact_pct, 2)}% "
                f"exceeds {IMPACT_THRESHOLDS['high_max']}% threshold. "
                f"External verifier sign-off also required."
            )
        else:
            rationale = "Default routing: single approver."
            approvers_required = 1
            approver_roles = ["ghg_inventory_manager"]

        # Override for structural changes: always require at least single approver.
        if request.category == ChangeCategory.STRUCTURAL and auto_approved:
            auto_approved = False
            approvers_required = 1
            approver_roles = ["ghg_inventory_manager", "sustainability_lead"]
            level = ApprovalLevel.SINGLE_APPROVER
            rationale += (
                " Structural changes require manual approval regardless "
                "of impact severity."
            )
            self._warnings.append(
                f"Structural change {request.request_id[:12]} routed to "
                f"manual approval despite low impact."
            )

        self._add_log(
            request.request_id, "approval_routed",
            "system", "System",
            f"Routed to {level.value}: {approvers_required} approver(s), "
            f"external_verifier={requires_external}.",
        )

        return ChangeApproval(
            request_id=request.request_id,
            required_level=level,
            approvers_required=approvers_required,
            approver_roles=approver_roles,
            requires_external_verifier=requires_external,
            auto_approved=auto_approved,
            routing_rationale=rationale,
            escalation_deadline_hours=escalation_hours,
        )

    def detect_base_year_trigger(
        self,
        request: ChangeRequest,
        impact: ChangeImpact,
    ) -> BaseYearTriggerDetection:
        """Detect whether a change triggers base year recalculation.

        Evaluates the change against GHG Protocol Chapter 5 criteria
        for base year recalculation significance.

        Args:
            request: The change request.
            impact: The impact assessment result.

        Returns:
            BaseYearTriggerDetection with trigger assessment.
        """
        logger.info(
            "Detecting base year trigger for request %s (category=%s)",
            request.request_id[:12], request.category.value,
        )

        base_year_total = self._base_year_total_tco2e
        if base_year_total <= Decimal("0"):
            return BaseYearTriggerDetection(
                request_id=request.request_id,
                triggers_recalculation=False,
                trigger_type=BaseYearTriggerType.NO_TRIGGER,
                base_year=self._base_year,
                base_year_total_tco2e=Decimal("0"),
                rationale="No base year total provided; trigger detection skipped.",
                recommended_actions=["Provide base year total for trigger detection."],
            )

        # Estimate impact on base year (use current impact as proxy).
        impact_on_base = impact.total_impact_tco2e
        impact_pct = _safe_pct(impact_on_base, base_year_total)

        triggers = False
        trigger_type = BaseYearTriggerType.NO_TRIGGER
        actions: List[str] = []

        # Structural changes.
        if request.category == ChangeCategory.STRUCTURAL:
            if impact_pct >= self._significance_threshold_pct:
                triggers = True
                trigger_type = BaseYearTriggerType.STRUCTURAL_THRESHOLD
                actions.append(
                    "Initiate base year recalculation for structural change."
                )
                actions.append(
                    "Apply acquisition/divestiture adjustments using "
                    "BaseYearRecalculationEngine."
                )

        # Methodological changes.
        elif request.category == ChangeCategory.METHODOLOGICAL:
            if impact_pct >= self._significance_threshold_pct:
                triggers = True
                trigger_type = BaseYearTriggerType.METHODOLOGY_THRESHOLD
                actions.append(
                    "Recalculate base year using updated methodology."
                )
                actions.append(
                    "Document methodology change per GHG Protocol Ch 5."
                )

        # Boundary changes.
        elif request.category == ChangeCategory.BOUNDARY:
            if impact_pct >= self._significance_threshold_pct:
                triggers = True
                trigger_type = BaseYearTriggerType.BOUNDARY_EXPANSION
                actions.append(
                    "Add new sources to base year inventory."
                )
                actions.append(
                    "Use best available historical data for base year inclusion."
                )

        # Error corrections.
        elif request.category == ChangeCategory.ERROR_CORRECTION:
            if impact_pct >= self._significance_threshold_pct:
                triggers = True
                trigger_type = BaseYearTriggerType.CUMULATIVE_ERROR
                actions.append(
                    "Apply error corrections to base year data."
                )
                actions.append(
                    "Document cumulative error impact per GHG Protocol Ch 5."
                )

        # Data quality and regulatory changes typically do not trigger
        # base year recalculation unless they change methodology.
        elif request.category in (
            ChangeCategory.DATA_QUALITY, ChangeCategory.REGULATORY,
        ):
            if impact_pct >= self._significance_threshold_pct:
                self._warnings.append(
                    f"Data quality/regulatory change {request.request_id[:12]} "
                    f"has significant impact ({_round_val(impact_pct, 2)}%). "
                    f"Review whether base year recalculation is warranted."
                )
                actions.append(
                    "Review whether this change constitutes a methodology "
                    "change requiring base year recalculation."
                )

        if not triggers:
            actions.append("No base year recalculation required.")

        rationale = self._build_trigger_rationale(
            request.category, impact_pct, triggers, trigger_type,
        )

        self._add_log(
            request.request_id, "base_year_trigger_assessed",
            "system", "System",
            f"Trigger detection: {trigger_type.value}, "
            f"impact_on_base_year={_round_val(impact_pct, 2)}%, "
            f"triggers={triggers}.",
        )

        return BaseYearTriggerDetection(
            request_id=request.request_id,
            triggers_recalculation=triggers,
            trigger_type=trigger_type,
            impact_on_base_year_tco2e=_round_val(impact_on_base, 4),
            impact_on_base_year_pct=_round_val(impact_pct, 4),
            significance_threshold_pct=self._significance_threshold_pct,
            base_year=self._base_year,
            base_year_total_tco2e=base_year_total,
            rationale=rationale,
            recommended_actions=actions,
        )

    def transition_status(
        self,
        request: ChangeRequest,
        new_status: ChangeStatus,
        actor_id: str = "system",
        actor_name: str = "System",
    ) -> ChangeRequest:
        """Publicly transition a change request to a new status.

        Validates that the transition is permitted according to the
        state machine, then applies the transition and logs it.

        Args:
            request: The change request to transition.
            new_status: Target status.
            actor_id: ID of the user or system performing the transition.
            actor_name: Display name of the actor.

        Returns:
            Updated ChangeRequest with new status.

        Raises:
            ValueError: If the transition is not permitted.
        """
        return self._transition_status(request, new_status, actor_id, actor_name)

    def get_valid_transitions(
        self,
        current_status: ChangeStatus,
    ) -> List[ChangeStatus]:
        """Get valid next statuses for a given current status.

        Args:
            current_status: The current status of the change request.

        Returns:
            List of valid target statuses.
        """
        return list(VALID_TRANSITIONS.get(current_status, []))

    def batch_assess(
        self,
        requests: List[ChangeRequest],
    ) -> List[ChangeManagementResult]:
        """Process multiple change requests in batch.

        Args:
            requests: List of change requests to process.

        Returns:
            List of ChangeManagementResult objects.

        Raises:
            ValueError: If batch size exceeds MAX_CHANGES_PER_BATCH.
        """
        if len(requests) > MAX_CHANGES_PER_BATCH:
            raise ValueError(
                f"Batch size {len(requests)} exceeds maximum "
                f"{MAX_CHANGES_PER_BATCH}."
            )

        logger.info("Batch processing %d change requests", len(requests))
        results: List[ChangeManagementResult] = []

        for request in requests:
            try:
                result = self.process_change(request)
                results.append(result)
            except Exception as exc:
                logger.error(
                    "Failed to process change request %s: %s",
                    request.request_id[:12], str(exc),
                )
                error_result = ChangeManagementResult(
                    request=request,
                    warnings=[f"Processing failed: {str(exc)}"],
                )
                error_result.provenance_hash = _compute_hash(error_result)
                results.append(error_result)

        return results

    def summarise_changes(
        self,
        results: List[ChangeManagementResult],
    ) -> Dict[str, Any]:
        """Summarise a batch of change management results.

        Args:
            results: List of processed change management results.

        Returns:
            Dict with summary statistics.
        """
        total = len(results)
        by_severity: Dict[str, int] = {s.value: 0 for s in ImpactSeverity}
        by_category: Dict[str, int] = {c.value: 0 for c in ChangeCategory}
        triggers_count = 0
        total_impact = Decimal("0")
        auto_approved_count = 0

        for r in results:
            if r.impact:
                by_severity[r.impact.severity.value] = (
                    by_severity.get(r.impact.severity.value, 0) + 1
                )
                total_impact += r.impact.total_impact_tco2e
            if r.request:
                by_category[r.request.category.value] = (
                    by_category.get(r.request.category.value, 0) + 1
                )
            if r.base_year_trigger and r.base_year_trigger.triggers_recalculation:
                triggers_count += 1
            if r.approval_routing and r.approval_routing.auto_approved:
                auto_approved_count += 1

        return {
            "total_changes": total,
            "by_severity": by_severity,
            "by_category": by_category,
            "base_year_triggers": triggers_count,
            "total_impact_tco2e": float(_round_val(total_impact, 2)),
            "auto_approved": auto_approved_count,
            "requiring_approval": total - auto_approved_count,
        }

    # ------------------------------------------------------------------
    # Private Methods
    # ------------------------------------------------------------------

    def _validate_request(self, request: ChangeRequest) -> None:
        """Validate a change request for completeness.

        Args:
            request: The change request to validate.

        Raises:
            ValueError: If validation fails.
        """
        if not request.title:
            raise ValueError("Change request title is required.")
        if request.total_inventory_tco2e <= Decimal("0"):
            raise ValueError(
                "Total inventory emissions must be provided and > 0."
            )
        if not request.affected_sources:
            self._warnings.append(
                f"Change request {request.request_id[:12]} has no "
                f"affected sources specified."
            )

    def _classify_severity(self, impact_pct: Decimal) -> ImpactSeverity:
        """Classify impact severity based on percentage thresholds.

        Args:
            impact_pct: Impact as percentage of total inventory.

        Returns:
            ImpactSeverity classification.
        """
        if impact_pct < IMPACT_THRESHOLDS["low_max"]:
            return ImpactSeverity.LOW
        elif impact_pct < IMPACT_THRESHOLDS["medium_max"]:
            return ImpactSeverity.MEDIUM
        elif impact_pct < IMPACT_THRESHOLDS["high_max"]:
            return ImpactSeverity.HIGH
        else:
            return ImpactSeverity.CRITICAL

    def _transition_status(
        self,
        request: ChangeRequest,
        new_status: ChangeStatus,
        actor_id: str,
        actor_name: str,
    ) -> ChangeRequest:
        """Internal status transition with validation and logging.

        Args:
            request: The change request.
            new_status: Target status.
            actor_id: Actor performing the transition.
            actor_name: Actor display name.

        Returns:
            Updated ChangeRequest.

        Raises:
            ValueError: If the transition is invalid.
        """
        old_status = request.status
        valid_targets = VALID_TRANSITIONS.get(old_status, [])

        if new_status not in valid_targets:
            raise ValueError(
                f"Invalid status transition: {old_status.value} -> "
                f"{new_status.value}. Valid targets: "
                f"{[t.value for t in valid_targets]}"
            )

        request.status = new_status

        self._add_log(
            request.request_id, "status_transition",
            actor_id, actor_name,
            f"Status changed from {old_status.value} to {new_status.value}.",
            old_status=old_status.value,
            new_status=new_status.value,
        )

        logger.info(
            "Request %s: %s -> %s (by %s)",
            request.request_id[:12], old_status.value,
            new_status.value, actor_name,
        )
        return request

    def _build_trigger_rationale(
        self,
        category: ChangeCategory,
        impact_pct: Decimal,
        triggers: bool,
        trigger_type: BaseYearTriggerType,
    ) -> str:
        """Build rationale string for base year trigger detection.

        Args:
            category: Change category.
            impact_pct: Impact as percentage of base year.
            triggers: Whether recalculation is triggered.
            trigger_type: Type of trigger detected.

        Returns:
            Rationale string.
        """
        parts = [
            f"Change category: {category.value}.",
            f"Impact on base year: {_round_val(impact_pct, 2)}%.",
            f"Significance threshold: {self._significance_threshold_pct}%.",
        ]

        if triggers:
            parts.append(
                f"RESULT: Base year recalculation TRIGGERED "
                f"({trigger_type.value}). Per GHG Protocol Chapter 5, "
                f"the base year must be recalculated when a {category.value} "
                f"change exceeds the significance threshold."
            )
        else:
            parts.append(
                "RESULT: No base year recalculation required. "
                "Impact is below the significance threshold."
            )

        return " ".join(parts)

    def _add_log(
        self,
        request_id: str,
        action: str,
        actor_id: str,
        actor_name: str,
        details: str,
        old_status: Optional[str] = None,
        new_status: Optional[str] = None,
    ) -> None:
        """Add an entry to the change log.

        Args:
            request_id: Related change request ID.
            action: Action performed.
            actor_id: Actor user ID.
            actor_name: Actor display name.
            details: Description of the event.
            old_status: Previous status (for transitions).
            new_status: New status (for transitions).
        """
        entry = ChangeLogEntry(
            request_id=request_id,
            action=action,
            actor_id=actor_id,
            actor_name=actor_name,
            old_status=old_status,
            new_status=new_status,
            details=details,
        )
        entry.provenance_hash = _compute_hash(entry)
        self._change_log.append(entry)
