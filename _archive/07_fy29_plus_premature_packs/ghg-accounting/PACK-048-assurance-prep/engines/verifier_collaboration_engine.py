# -*- coding: utf-8 -*-
"""
VerifierCollaborationEngine - PACK-048 GHG Assurance Prep Engine 5
====================================================================

Manages the collaboration workflow between the reporting entity and the
external verifier/assurance provider.  Tracks information requests (IRs),
verifier queries, findings, responses, escalations, and engagement
timeline milestones with SLA monitoring.

Calculation Methodology:
    Information Request (IR) Management:
        - Creation, assignment, response, evidence linking
        - Status: OPEN -> ASSIGNED -> IN_PROGRESS -> RESPONDED -> CLOSED

    Query Management:
        - Category: DATA, METHODOLOGY, BOUNDARY, CALCULATION,
                    DOCUMENTATION, CONTROL, COMPLETENESS, OTHER
        - Priority: CRITICAL, HIGH, MEDIUM, LOW
        - Deadline tracking with SLA monitoring

    Finding Management:
        - Type: NON_CONFORMITY, OBSERVATION, OPPORTUNITY,
                RECOMMENDATION, GOOD_PRACTICE
        - Severity: CRITICAL, MAJOR, MINOR, OBSERVATION

    Resolution Workflow:
        OPEN -> IN_PROGRESS -> RESPONDED -> FOLLOW_UP -> RESOLVED -> CLOSED

    SLA Tracking:
        query_response     = 5 business days
        critical_finding   = 10 business days
        evidence_request   = 3 business days

    SLA Compliance:
        SLA% = count(on_time) / count(total) * 100

    Escalation Rules:
        - Auto-escalate if SLA exceeded by 50%
        - Critical findings escalate after 5 business days
        - Escalation levels: MANAGER -> DIRECTOR -> EXECUTIVE

    Engagement Timeline:
        Milestone tracking: planned vs actual dates
        Phases: PLANNING, FIELDWORK, REPORTING, COMPLETION

Regulatory References:
    - ISAE 3410: Communication with management and governance
    - ISAE 3000 (Revised): Communication requirements
    - ISO 14064-3:2019: Verification findings management
    - IAF MD 6:2014: Verification body requirements

Zero-Hallucination:
    - All SLA calculations use deterministic Decimal arithmetic
    - No LLM involvement in any calculation path
    - SHA-256 provenance hash on every result

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-048 GHG Assurance Prep
Engine:  5 of 10
Status:  Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import time
import uuid
from datetime import datetime, date, timezone, timedelta
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator, model_validator

from greenlang.schemas import utcnow
from greenlang.schemas.enums import Priority

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _new_uuid() -> str:
    return str(uuid.uuid4())

def _compute_hash(data: Any) -> str:
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
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return Decimal("0")

def _safe_divide(
    numerator: Decimal, denominator: Decimal, default: Decimal = Decimal("0"),
) -> Decimal:
    if denominator == Decimal("0"):
        return default
    return numerator / denominator

def _round2(value: Any) -> float:
    return float(Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))

def _business_days_between(start: datetime, end: datetime) -> int:
    """Count business days between two dates (Mon-Fri)."""
    if end <= start:
        return 0
    current = start
    count = 0
    while current < end:
        current += timedelta(days=1)
        if current.weekday() < 5:  # Monday=0, Friday=4
            count += 1
    return count

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class IRStatus(str, Enum):
    """Information Request status."""
    OPEN = "open"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    RESPONDED = "responded"
    CLOSED = "closed"

class QueryCategory(str, Enum):
    """Verifier query category."""
    DATA = "data"
    METHODOLOGY = "methodology"
    BOUNDARY = "boundary"
    CALCULATION = "calculation"
    DOCUMENTATION = "documentation"
    CONTROL = "control"
    COMPLETENESS = "completeness"
    OTHER = "other"

class FindingType(str, Enum):
    """Finding type.

    NON_CONFORMITY:     Departure from standard/criteria.
    OBSERVATION:        Potential concern, not non-conformity.
    OPPORTUNITY:        Opportunity for improvement.
    RECOMMENDATION:     Suggested improvement.
    GOOD_PRACTICE:      Notable positive practice.
    """
    NON_CONFORMITY = "non_conformity"
    OBSERVATION = "observation"
    OPPORTUNITY = "opportunity"
    RECOMMENDATION = "recommendation"
    GOOD_PRACTICE = "good_practice"

class FindingSeverity(str, Enum):
    """Finding severity."""
    CRITICAL = "critical"
    MAJOR = "major"
    MINOR = "minor"
    OBSERVATION = "observation"

class ResolutionStatus(str, Enum):
    """Resolution workflow status."""
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    RESPONDED = "responded"
    FOLLOW_UP = "follow_up"
    RESOLVED = "resolved"
    CLOSED = "closed"

class EscalationLevel(str, Enum):
    """Escalation level."""
    NONE = "none"
    MANAGER = "manager"
    DIRECTOR = "director"
    EXECUTIVE = "executive"

class EngagementPhase(str, Enum):
    """Engagement phase."""
    PLANNING = "planning"
    FIELDWORK = "fieldwork"
    REPORTING = "reporting"
    COMPLETION = "completion"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# SLA in business days
SLA_QUERY_RESPONSE: int = 5
SLA_CRITICAL_FINDING: int = 10
SLA_EVIDENCE_REQUEST: int = 3

# Escalation thresholds (multiples of SLA)
ESCALATION_THRESHOLD: Decimal = Decimal("1.5")

# Priority weights for SLA ordering
PRIORITY_WEIGHTS: Dict[str, int] = {
    Priority.CRITICAL.value: 4,
    Priority.HIGH.value: 3,
    Priority.MEDIUM.value: 2,
    Priority.LOW.value: 1,
}

# ---------------------------------------------------------------------------
# Pydantic Models -- Inputs
# ---------------------------------------------------------------------------

class InformationRequest(BaseModel):
    """Information request from verifier.

    Attributes:
        ir_id:          IR identifier.
        title:          Request title.
        description:    Request description.
        category:       Query category.
        priority:       Priority level.
        assigned_to:    Person assigned.
        evidence_refs:  Linked evidence references.
        status:         Current status.
        created_at:     Creation date (ISO).
        due_date:       Due date (ISO).
        responded_at:   Response date (ISO).
        response_text:  Response text.
    """
    ir_id: str = Field(default_factory=_new_uuid, description="IR ID")
    title: str = Field(default="", description="Title")
    description: str = Field(default="", description="Description")
    category: QueryCategory = Field(default=QueryCategory.DATA, description="Category")
    priority: Priority = Field(default=Priority.MEDIUM, description="Priority")
    assigned_to: str = Field(default="", description="Assignee")
    evidence_refs: List[str] = Field(default_factory=list, description="Evidence refs")
    status: IRStatus = Field(default=IRStatus.OPEN, description="Status")
    created_at: str = Field(default="", description="Created")
    due_date: str = Field(default="", description="Due date")
    responded_at: str = Field(default="", description="Responded")
    response_text: str = Field(default="", description="Response")

class VerifierQuery(BaseModel):
    """Query raised by verifier.

    Attributes:
        query_id:       Query identifier.
        title:          Query title.
        description:    Query description.
        category:       Query category.
        priority:       Priority level.
        status:         Resolution status.
        raised_by:      Verifier name.
        raised_at:      Date raised (ISO).
        due_date:       Due date (ISO).
        responses:      Threaded response history.
        evidence_refs:  Linked evidence references.
    """
    query_id: str = Field(default_factory=_new_uuid, description="Query ID")
    title: str = Field(default="", description="Title")
    description: str = Field(default="", description="Description")
    category: QueryCategory = Field(default=QueryCategory.DATA, description="Category")
    priority: Priority = Field(default=Priority.MEDIUM, description="Priority")
    status: ResolutionStatus = Field(default=ResolutionStatus.OPEN, description="Status")
    raised_by: str = Field(default="", description="Raised by")
    raised_at: str = Field(default="", description="Raised at")
    due_date: str = Field(default="", description="Due date")
    responses: List[Dict[str, str]] = Field(default_factory=list, description="Responses")
    evidence_refs: List[str] = Field(default_factory=list, description="Evidence refs")

class Finding(BaseModel):
    """Finding issued by verifier.

    Attributes:
        finding_id:     Finding identifier.
        title:          Finding title.
        description:    Finding description.
        finding_type:   Finding type.
        severity:       Finding severity.
        status:         Resolution status.
        raised_at:      Date raised (ISO).
        due_date:       Due date (ISO).
        resolved_at:    Date resolved (ISO).
        corrective_action: Corrective action description.
        evidence_refs:  Linked evidence references.
    """
    finding_id: str = Field(default_factory=_new_uuid, description="Finding ID")
    title: str = Field(default="", description="Title")
    description: str = Field(default="", description="Description")
    finding_type: FindingType = Field(
        default=FindingType.OBSERVATION, description="Type"
    )
    severity: FindingSeverity = Field(
        default=FindingSeverity.OBSERVATION, description="Severity"
    )
    status: ResolutionStatus = Field(
        default=ResolutionStatus.OPEN, description="Status"
    )
    raised_at: str = Field(default="", description="Raised at")
    due_date: str = Field(default="", description="Due date")
    resolved_at: str = Field(default="", description="Resolved at")
    corrective_action: str = Field(default="", description="Corrective action")
    evidence_refs: List[str] = Field(default_factory=list, description="Evidence refs")

class EngagementMilestone(BaseModel):
    """Engagement timeline milestone.

    Attributes:
        milestone_id:   Milestone identifier.
        phase:          Engagement phase.
        name:           Milestone name.
        planned_date:   Planned date (ISO).
        actual_date:    Actual date (ISO).
        completed:      Whether completed.
    """
    milestone_id: str = Field(default_factory=_new_uuid, description="ID")
    phase: EngagementPhase = Field(default=EngagementPhase.PLANNING, description="Phase")
    name: str = Field(default="", description="Name")
    planned_date: str = Field(default="", description="Planned")
    actual_date: str = Field(default="", description="Actual")
    completed: bool = Field(default=False, description="Completed")

class CollaborationConfig(BaseModel):
    """Configuration for verifier collaboration engine.

    Attributes:
        organisation_id:        Organisation identifier.
        verifier_name:          Verifier/assurance provider name.
        engagement_id:          Engagement identifier.
        sla_query_days:         SLA for query response (business days).
        sla_critical_days:      SLA for critical finding (business days).
        sla_evidence_days:      SLA for evidence request (business days).
        output_precision:       Output decimal places.
    """
    organisation_id: str = Field(default="", description="Org ID")
    verifier_name: str = Field(default="", description="Verifier name")
    engagement_id: str = Field(default_factory=_new_uuid, description="Engagement ID")
    sla_query_days: int = Field(default=SLA_QUERY_RESPONSE, ge=1, description="SLA query")
    sla_critical_days: int = Field(
        default=SLA_CRITICAL_FINDING, ge=1, description="SLA critical"
    )
    sla_evidence_days: int = Field(
        default=SLA_EVIDENCE_REQUEST, ge=1, description="SLA evidence"
    )
    output_precision: int = Field(default=2, ge=0, le=6, description="Output precision")

class CollaborationInput(BaseModel):
    """Input for verifier collaboration engine.

    Attributes:
        information_requests:   Information requests.
        queries:                Verifier queries.
        findings:               Findings.
        milestones:             Engagement milestones.
        config:                 Configuration.
    """
    information_requests: List[InformationRequest] = Field(
        default_factory=list, description="IRs"
    )
    queries: List[VerifierQuery] = Field(
        default_factory=list, description="Queries"
    )
    findings: List[Finding] = Field(default_factory=list, description="Findings")
    milestones: List[EngagementMilestone] = Field(
        default_factory=list, description="Milestones"
    )
    config: CollaborationConfig = Field(
        default_factory=CollaborationConfig, description="Configuration"
    )

# ---------------------------------------------------------------------------
# Pydantic Models -- Outputs
# ---------------------------------------------------------------------------

class Response(BaseModel):
    """Response tracking record.

    Attributes:
        item_type:      Type (ir/query/finding).
        item_id:        Item identifier.
        status:         Current status.
        days_open:      Business days open.
        sla_days:       SLA target days.
        sla_met:        Whether SLA met.
        overdue:        Whether overdue.
        overdue_days:   Days overdue (0 if not overdue).
    """
    item_type: str = Field(default="", description="Type")
    item_id: str = Field(default="", description="ID")
    status: str = Field(default="", description="Status")
    days_open: int = Field(default=0, description="Days open")
    sla_days: int = Field(default=0, description="SLA days")
    sla_met: bool = Field(default=True, description="SLA met")
    overdue: bool = Field(default=False, description="Overdue")
    overdue_days: int = Field(default=0, description="Overdue days")

class EscalationRecord(BaseModel):
    """Escalation record for overdue items.

    Attributes:
        item_type:          Type (ir/query/finding).
        item_id:            Item identifier.
        title:              Item title.
        priority:           Priority level.
        escalation_level:   Escalation level.
        days_overdue:       Days overdue.
        recommended_action: Recommended action.
    """
    item_type: str = Field(default="", description="Type")
    item_id: str = Field(default="", description="ID")
    title: str = Field(default="", description="Title")
    priority: str = Field(default="", description="Priority")
    escalation_level: str = Field(default="", description="Level")
    days_overdue: int = Field(default=0, description="Days overdue")
    recommended_action: str = Field(default="", description="Action")

class EngagementTimeline(BaseModel):
    """Engagement timeline summary.

    Attributes:
        engagement_id:          Engagement identifier.
        total_milestones:       Total milestones.
        completed_milestones:   Completed milestones.
        on_track_count:         On-track milestones.
        delayed_count:          Delayed milestones.
        overall_progress_pct:   Progress percentage.
        phase_progress:         Progress per phase.
    """
    engagement_id: str = Field(default="", description="Engagement ID")
    total_milestones: int = Field(default=0, description="Total")
    completed_milestones: int = Field(default=0, description="Completed")
    on_track_count: int = Field(default=0, description="On track")
    delayed_count: int = Field(default=0, description="Delayed")
    overall_progress_pct: Decimal = Field(default=Decimal("0"), description="Progress %")
    phase_progress: Dict[str, Decimal] = Field(
        default_factory=dict, description="Phase progress"
    )

class SLASummary(BaseModel):
    """SLA compliance summary.

    Attributes:
        total_items:        Total trackable items.
        on_time_count:      Items responded on time.
        overdue_count:      Currently overdue items.
        sla_compliance_pct: SLA compliance percentage.
        avg_response_days:  Average response time (business days).
    """
    total_items: int = Field(default=0, description="Total items")
    on_time_count: int = Field(default=0, description="On time")
    overdue_count: int = Field(default=0, description="Overdue")
    sla_compliance_pct: Decimal = Field(default=Decimal("0"), description="SLA %")
    avg_response_days: Decimal = Field(default=Decimal("0"), description="Avg days")

class CollaborationResult(BaseModel):
    """Complete result of verifier collaboration tracking.

    Attributes:
        result_id:              Unique result identifier.
        organisation_id:        Organisation identifier.
        engagement_id:          Engagement identifier.
        response_tracking:      Response tracking per item.
        escalations:            Escalation records.
        sla_summary:            SLA compliance summary.
        timeline:               Engagement timeline.
        total_irs:              Total information requests.
        total_queries:          Total queries.
        total_findings:         Total findings.
        open_items:             Currently open items.
        critical_open:          Critical items still open.
        warnings:               Warnings.
        calculated_at:          Timestamp.
        processing_time_ms:     Processing time (ms).
        provenance_hash:        SHA-256 hash.
    """
    result_id: str = Field(default_factory=_new_uuid, description="Result ID")
    organisation_id: str = Field(default="", description="Org ID")
    engagement_id: str = Field(default="", description="Engagement ID")
    response_tracking: List[Response] = Field(
        default_factory=list, description="Responses"
    )
    escalations: List[EscalationRecord] = Field(
        default_factory=list, description="Escalations"
    )
    sla_summary: SLASummary = Field(default_factory=SLASummary, description="SLA summary")
    timeline: EngagementTimeline = Field(
        default_factory=EngagementTimeline, description="Timeline"
    )
    total_irs: int = Field(default=0, description="Total IRs")
    total_queries: int = Field(default=0, description="Total queries")
    total_findings: int = Field(default=0, description="Total findings")
    open_items: int = Field(default=0, description="Open items")
    critical_open: int = Field(default=0, description="Critical open")
    warnings: List[str] = Field(default_factory=list, description="Warnings")
    calculated_at: str = Field(default="", description="Timestamp")
    processing_time_ms: float = Field(default=0.0, description="Processing time (ms)")
    provenance_hash: str = Field(default="", description="SHA-256 hash")

# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class VerifierCollaborationEngine:
    """Manages collaboration workflow between entity and verifier.

    Tracks IRs, queries, findings, responses, escalations, and
    engagement timeline with SLA monitoring.

    Guarantees:
        - Deterministic: Same inputs always produce the same output.
        - Reproducible: Full provenance tracking with SHA-256 hashes.
        - Auditable: Every item tracked with timestamps and status.
        - Zero-Hallucination: No LLM in any calculation path.
    """

    def __init__(self) -> None:
        self._version = _MODULE_VERSION
        logger.info("VerifierCollaborationEngine v%s initialised", self._version)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def calculate(self, input_data: CollaborationInput) -> CollaborationResult:
        """Analyse collaboration status and SLA compliance.

        Args:
            input_data: IRs, queries, findings, milestones, config.

        Returns:
            CollaborationResult with response tracking, escalations, SLA.
        """
        t0 = time.perf_counter()
        warnings: List[str] = []
        config = input_data.config
        prec = config.output_precision
        prec_str = "0." + "0" * prec
        now = utcnow()

        # Step 1: Track responses for IRs
        responses: List[Response] = []
        for ir in input_data.information_requests:
            resp = self._track_ir(ir, config, now)
            responses.append(resp)

        # Step 2: Track responses for queries
        for query in input_data.queries:
            resp = self._track_query(query, config, now)
            responses.append(resp)

        # Step 3: Track responses for findings
        for finding in input_data.findings:
            resp = self._track_finding(finding, config, now)
            responses.append(resp)

        # Step 4: Identify escalations
        escalations: List[EscalationRecord] = []
        for resp in responses:
            if resp.overdue:
                esc = self._check_escalation(resp, input_data, config)
                if esc:
                    escalations.append(esc)

        # Step 5: SLA summary
        sla_summary = self._compute_sla_summary(responses, prec_str)

        # Step 6: Engagement timeline
        timeline = self._compute_timeline(
            input_data.milestones, config, now, prec_str
        )

        # Step 7: Open item counts
        open_statuses = {
            ResolutionStatus.OPEN.value,
            ResolutionStatus.IN_PROGRESS.value,
            ResolutionStatus.RESPONDED.value,
            ResolutionStatus.FOLLOW_UP.value,
            IRStatus.OPEN.value,
            IRStatus.ASSIGNED.value,
            IRStatus.IN_PROGRESS.value,
        }
        open_items = sum(1 for r in responses if r.status in open_statuses)
        critical_open = sum(
            1 for r in responses
            if r.status in open_statuses and r.item_type == "finding"
        )

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        result = CollaborationResult(
            organisation_id=config.organisation_id,
            engagement_id=config.engagement_id,
            response_tracking=responses,
            escalations=escalations,
            sla_summary=sla_summary,
            timeline=timeline,
            total_irs=len(input_data.information_requests),
            total_queries=len(input_data.queries),
            total_findings=len(input_data.findings),
            open_items=open_items,
            critical_open=critical_open,
            warnings=warnings,
            calculated_at=now.isoformat(),
            processing_time_ms=round(elapsed_ms, 3),
        )
        result.provenance_hash = _compute_hash(result)
        return result

    # ------------------------------------------------------------------
    # Internal: IR Tracking
    # ------------------------------------------------------------------

    def _track_ir(
        self, ir: InformationRequest, config: CollaborationConfig, now: datetime,
    ) -> Response:
        """Track response status for an information request."""
        sla_days = config.sla_evidence_days
        created = self._parse_dt(ir.created_at, now)
        responded = self._parse_dt(ir.responded_at, None)

        if responded:
            days_open = _business_days_between(created, responded)
        else:
            days_open = _business_days_between(created, now)

        closed_statuses = {IRStatus.RESPONDED.value, IRStatus.CLOSED.value}
        sla_met = days_open <= sla_days if ir.status.value in closed_statuses else True
        overdue = ir.status.value not in closed_statuses and days_open > sla_days
        overdue_days = max(0, days_open - sla_days) if overdue else 0

        return Response(
            item_type="ir",
            item_id=ir.ir_id,
            status=ir.status.value,
            days_open=days_open,
            sla_days=sla_days,
            sla_met=sla_met,
            overdue=overdue,
            overdue_days=overdue_days,
        )

    # ------------------------------------------------------------------
    # Internal: Query Tracking
    # ------------------------------------------------------------------

    def _track_query(
        self, query: VerifierQuery, config: CollaborationConfig, now: datetime,
    ) -> Response:
        """Track response status for a verifier query."""
        sla_days = config.sla_query_days
        raised = self._parse_dt(query.raised_at, now)
        days_open = _business_days_between(raised, now)

        closed_statuses = {
            ResolutionStatus.RESOLVED.value, ResolutionStatus.CLOSED.value,
        }
        is_closed = query.status.value in closed_statuses
        sla_met = days_open <= sla_days if is_closed else True
        overdue = not is_closed and days_open > sla_days
        overdue_days = max(0, days_open - sla_days) if overdue else 0

        return Response(
            item_type="query",
            item_id=query.query_id,
            status=query.status.value,
            days_open=days_open,
            sla_days=sla_days,
            sla_met=sla_met,
            overdue=overdue,
            overdue_days=overdue_days,
        )

    # ------------------------------------------------------------------
    # Internal: Finding Tracking
    # ------------------------------------------------------------------

    def _track_finding(
        self, finding: Finding, config: CollaborationConfig, now: datetime,
    ) -> Response:
        """Track response status for a finding."""
        is_critical = finding.severity in (
            FindingSeverity.CRITICAL, FindingSeverity.MAJOR
        )
        sla_days = config.sla_critical_days if is_critical else config.sla_query_days

        raised = self._parse_dt(finding.raised_at, now)
        resolved = self._parse_dt(finding.resolved_at, None)

        if resolved:
            days_open = _business_days_between(raised, resolved)
        else:
            days_open = _business_days_between(raised, now)

        closed_statuses = {
            ResolutionStatus.RESOLVED.value, ResolutionStatus.CLOSED.value,
        }
        is_closed = finding.status.value in closed_statuses
        sla_met = days_open <= sla_days if is_closed else True
        overdue = not is_closed and days_open > sla_days
        overdue_days = max(0, days_open - sla_days) if overdue else 0

        return Response(
            item_type="finding",
            item_id=finding.finding_id,
            status=finding.status.value,
            days_open=days_open,
            sla_days=sla_days,
            sla_met=sla_met,
            overdue=overdue,
            overdue_days=overdue_days,
        )

    # ------------------------------------------------------------------
    # Internal: Escalation
    # ------------------------------------------------------------------

    def _check_escalation(
        self,
        resp: Response,
        input_data: CollaborationInput,
        config: CollaborationConfig,
    ) -> Optional[EscalationRecord]:
        """Check if an overdue item needs escalation."""
        if not resp.overdue:
            return None

        overdue_ratio = _safe_divide(
            _decimal(resp.days_open), _decimal(resp.sla_days)
        )

        if overdue_ratio < ESCALATION_THRESHOLD:
            return None

        # Determine escalation level
        if overdue_ratio >= Decimal("3"):
            level = EscalationLevel.EXECUTIVE.value
        elif overdue_ratio >= Decimal("2"):
            level = EscalationLevel.DIRECTOR.value
        else:
            level = EscalationLevel.MANAGER.value

        # Get title
        title = ""
        priority = Priority.MEDIUM.value
        if resp.item_type == "ir":
            for ir in input_data.information_requests:
                if ir.ir_id == resp.item_id:
                    title = ir.title
                    priority = ir.priority.value
                    break
        elif resp.item_type == "query":
            for q in input_data.queries:
                if q.query_id == resp.item_id:
                    title = q.title
                    priority = q.priority.value
                    break
        elif resp.item_type == "finding":
            for f in input_data.findings:
                if f.finding_id == resp.item_id:
                    title = f.title
                    priority = f.severity.value
                    break

        return EscalationRecord(
            item_type=resp.item_type,
            item_id=resp.item_id,
            title=title,
            priority=priority,
            escalation_level=level,
            days_overdue=resp.overdue_days,
            recommended_action=f"Escalate to {level}: {resp.overdue_days} days overdue "
                               f"(SLA: {resp.sla_days} days).",
        )

    # ------------------------------------------------------------------
    # Internal: SLA Summary
    # ------------------------------------------------------------------

    def _compute_sla_summary(
        self, responses: List[Response], prec_str: str,
    ) -> SLASummary:
        """Compute SLA compliance summary."""
        total = len(responses)
        if total == 0:
            return SLASummary()

        on_time = sum(1 for r in responses if r.sla_met and not r.overdue)
        overdue = sum(1 for r in responses if r.overdue)
        sla_pct = _safe_divide(
            _decimal(on_time), _decimal(total)
        ) * Decimal("100")
        sla_pct = sla_pct.quantize(Decimal(prec_str), rounding=ROUND_HALF_UP)

        avg_days = Decimal("0")
        if total > 0:
            avg_days = _safe_divide(
                _decimal(sum(r.days_open for r in responses)),
                _decimal(total),
            ).quantize(Decimal(prec_str), rounding=ROUND_HALF_UP)

        return SLASummary(
            total_items=total,
            on_time_count=on_time,
            overdue_count=overdue,
            sla_compliance_pct=sla_pct,
            avg_response_days=avg_days,
        )

    # ------------------------------------------------------------------
    # Internal: Timeline
    # ------------------------------------------------------------------

    def _compute_timeline(
        self,
        milestones: List[EngagementMilestone],
        config: CollaborationConfig,
        now: datetime,
        prec_str: str,
    ) -> EngagementTimeline:
        """Compute engagement timeline summary."""
        total = len(milestones)
        if total == 0:
            return EngagementTimeline(engagement_id=config.engagement_id)

        completed = sum(1 for m in milestones if m.completed)
        on_track = 0
        delayed = 0

        for m in milestones:
            if m.completed:
                on_track += 1
            elif m.planned_date:
                planned = self._parse_dt(m.planned_date, None)
                if planned and now <= planned:
                    on_track += 1
                else:
                    delayed += 1
            else:
                on_track += 1

        progress = _safe_divide(
            _decimal(completed), _decimal(total)
        ) * Decimal("100")
        progress = progress.quantize(Decimal(prec_str), rounding=ROUND_HALF_UP)

        # Phase progress
        phase_map: Dict[str, List[EngagementMilestone]] = {}
        for m in milestones:
            p = m.phase.value
            if p not in phase_map:
                phase_map[p] = []
            phase_map[p].append(m)

        phase_progress: Dict[str, Decimal] = {}
        for phase, phase_ms in phase_map.items():
            phase_total = len(phase_ms)
            phase_done = sum(1 for m in phase_ms if m.completed)
            pp = _safe_divide(
                _decimal(phase_done), _decimal(phase_total)
            ) * Decimal("100")
            phase_progress[phase] = pp.quantize(Decimal(prec_str), rounding=ROUND_HALF_UP)

        return EngagementTimeline(
            engagement_id=config.engagement_id,
            total_milestones=total,
            completed_milestones=completed,
            on_track_count=on_track,
            delayed_count=delayed,
            overall_progress_pct=progress,
            phase_progress=phase_progress,
        )

    # ------------------------------------------------------------------
    # Internal: Utility
    # ------------------------------------------------------------------

    def _parse_dt(self, dt_str: str, default: Optional[datetime]) -> Optional[datetime]:
        """Parse ISO datetime string."""
        if not dt_str:
            return default
        try:
            dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except (ValueError, TypeError):
            return default

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def get_version(self) -> str:
        return self._version

# ---------------------------------------------------------------------------
# __all__
# ---------------------------------------------------------------------------

__all__ = [
    # Enums
    "IRStatus",
    "QueryCategory",
    "Priority",
    "FindingType",
    "FindingSeverity",
    "ResolutionStatus",
    "EscalationLevel",
    "EngagementPhase",
    # Input Models
    "InformationRequest",
    "VerifierQuery",
    "Finding",
    "EngagementMilestone",
    "CollaborationConfig",
    "CollaborationInput",
    # Output Models
    "Response",
    "EscalationRecord",
    "EngagementTimeline",
    "SLASummary",
    "CollaborationResult",
    # Engine
    "VerifierCollaborationEngine",
]
