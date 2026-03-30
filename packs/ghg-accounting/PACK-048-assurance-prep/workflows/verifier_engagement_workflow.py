# -*- coding: utf-8 -*-
"""
Verifier Engagement Workflow
====================================

5-phase workflow for GHG assurance verifier engagement management covering
engagement scoping, verifier onboarding, query management, finding tracking,
and engagement closeout within PACK-048 GHG Assurance Prep Pack.

Phases:
    1. EngagementScoping           -- Define the engagement scope, boundaries,
                                      assurance level, reporting period, and
                                      agreed-upon criteria between the
                                      organisation and the assurance provider.
    2. VerifierOnboarding          -- Onboard the verifier with access controls,
                                      evidence access, data room setup, key
                                      contacts, and communication protocols.
    3. QueryManagement             -- Manage the information request (IR) and
                                      query lifecycle from issuance through
                                      response, review, and resolution.
    4. FindingTracking             -- Track assurance findings from initial
                                      draft through management response,
                                      remediation action, and closure.
    5. EngagementCloseout          -- Close the engagement with opinion
                                      documentation, final report acceptance,
                                      lessons learned, and archive.

The workflow follows GreenLang zero-hallucination principles: every numeric
result is derived from deterministic formulas and validated reference data.
SHA-256 provenance hashes guarantee auditability.

Regulatory Basis:
    ISAE 3410 (2012) - Assurance engagement procedures
    ISO 14064-3:2019 - Verification engagement process
    ISQM 1 (2022) - Quality management for assurance engagements
    AA1000AS v3 (2020) - Engagement and assurance process
    ESRS E1 (2024) - Limited/reasonable assurance requirements
    CSRD (2022/2464) - Mandatory assurance engagement procedures

Schedule: Per assurance engagement cycle (annually)
Estimated duration: 6-12 weeks for full engagement cycle

Author: GreenLang Team
Version: 48.0.0
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import time
import uuid
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator
from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"

# =============================================================================
# HELPERS
# =============================================================================

def _new_uuid() -> str:
    """Return a new UUID4 hex string."""
    return uuid.uuid4().hex

def _compute_hash(data: Any) -> str:
    """Compute SHA-256 hash of JSON-serialisable data."""
    serialised = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(serialised.encode("utf-8")).hexdigest()

# =============================================================================
# ENUMS
# =============================================================================

class PhaseStatus(str, Enum):
    """Status of a workflow phase."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"

class WorkflowStatus(str, Enum):
    """Overall workflow execution status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"

class EngagementPhase(str, Enum):
    """Verifier engagement workflow phases."""

    ENGAGEMENT_SCOPING = "engagement_scoping"
    VERIFIER_ONBOARDING = "verifier_onboarding"
    QUERY_MANAGEMENT = "query_management"
    FINDING_TRACKING = "finding_tracking"
    ENGAGEMENT_CLOSEOUT = "engagement_closeout"

class EngagementStatus(str, Enum):
    """Overall engagement status."""

    SCOPING = "scoping"
    ONBOARDING = "onboarding"
    FIELDWORK = "fieldwork"
    REPORTING = "reporting"
    CLOSED = "closed"

class AssuranceLevel(str, Enum):
    """Assurance engagement level."""

    LIMITED = "limited"
    REASONABLE = "reasonable"

class QueryStatus(str, Enum):
    """Status of an information request / query."""

    ISSUED = "issued"
    IN_PROGRESS = "in_progress"
    RESPONDED = "responded"
    UNDER_REVIEW = "under_review"
    RESOLVED = "resolved"
    ESCALATED = "escalated"
    WITHDRAWN = "withdrawn"

class QueryPriority(str, Enum):
    """Priority of a query."""

    CRITICAL = "critical"
    HIGH = "high"
    STANDARD = "standard"
    LOW = "low"

class FindingSeverity(str, Enum):
    """Severity of an assurance finding."""

    QUALIFICATION = "qualification"
    EMPHASIS_OF_MATTER = "emphasis_of_matter"
    RECOMMENDATION = "recommendation"
    OBSERVATION = "observation"

class FindingStatus(str, Enum):
    """Status of an assurance finding."""

    DRAFT = "draft"
    MANAGEMENT_RESPONSE = "management_response"
    REMEDIATION = "remediation"
    CLOSED = "closed"
    ACCEPTED = "accepted"

class OpinionType(str, Enum):
    """Type of assurance opinion."""

    UNMODIFIED = "unmodified"
    MODIFIED_QUALIFIED = "modified_qualified"
    MODIFIED_ADVERSE = "modified_adverse"
    DISCLAIMER = "disclaimer"

class AccessLevel(str, Enum):
    """Data room access level."""

    FULL = "full"
    READ_ONLY = "read_only"
    RESTRICTED = "restricted"

# =============================================================================
# DATA MODELS
# =============================================================================

class PhaseResult(BaseModel):
    """Result from a single workflow phase."""

    phase_name: str = Field(..., description="Phase identifier")
    phase_number: int = Field(default=0, description="Phase sequence number")
    status: PhaseStatus = Field(..., description="Phase completion status")
    duration_seconds: float = Field(default=0.0, description="Phase duration")
    outputs: Dict[str, Any] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="", description="SHA-256 of phase output")

class EngagementScope(BaseModel):
    """Agreed engagement scope definition."""

    scope_id: str = Field(default_factory=lambda: f"scope-{_new_uuid()[:8]}")
    organization_id: str = Field(default="")
    verifier_name: str = Field(default="")
    assurance_level: AssuranceLevel = Field(default=AssuranceLevel.LIMITED)
    reporting_period: str = Field(default="2025")
    scope_boundaries: List[str] = Field(default_factory=lambda: ["scope_1", "scope_2"])
    materiality_threshold_pct: float = Field(default=5.0, ge=0.0, le=25.0)
    agreed_criteria: str = Field(default="GHG Protocol Corporate Standard")
    engagement_start_date: str = Field(default="")
    engagement_end_date: str = Field(default="")
    key_milestones: List[Dict[str, str]] = Field(default_factory=list)
    provenance_hash: str = Field(default="")

class VerifierAccess(BaseModel):
    """Verifier access configuration."""

    verifier_id: str = Field(default_factory=lambda: f"ver-{_new_uuid()[:8]}")
    verifier_name: str = Field(default="")
    lead_verifier: str = Field(default="")
    team_members: List[str] = Field(default_factory=list)
    access_level: AccessLevel = Field(default=AccessLevel.READ_ONLY)
    data_room_url: str = Field(default="")
    nda_signed: bool = Field(default=False)
    key_contacts: List[Dict[str, str]] = Field(default_factory=list)
    communication_protocol: str = Field(default="email")
    provenance_hash: str = Field(default="")

class QueryRecord(BaseModel):
    """Record of an information request / query."""

    query_id: str = Field(default_factory=lambda: f"qry-{_new_uuid()[:8]}")
    query_number: int = Field(default=0)
    issued_by: str = Field(default="verifier")
    issued_date: str = Field(default="")
    subject: str = Field(default="")
    description: str = Field(default="")
    priority: QueryPriority = Field(default=QueryPriority.STANDARD)
    status: QueryStatus = Field(default=QueryStatus.ISSUED)
    response_due_date: str = Field(default="")
    response_date: str = Field(default="")
    response_summary: str = Field(default="")
    resolution_date: str = Field(default="")
    days_outstanding: int = Field(default=0, ge=0)
    provenance_hash: str = Field(default="")

class FindingRecord(BaseModel):
    """Record of an assurance finding."""

    finding_id: str = Field(default_factory=lambda: f"fnd-{_new_uuid()[:8]}")
    finding_number: int = Field(default=0)
    severity: FindingSeverity = Field(default=FindingSeverity.OBSERVATION)
    status: FindingStatus = Field(default=FindingStatus.DRAFT)
    title: str = Field(default="")
    description: str = Field(default="")
    management_response: str = Field(default="")
    remediation_action: str = Field(default="")
    remediation_owner: str = Field(default="")
    remediation_deadline: str = Field(default="")
    closure_date: str = Field(default="")
    closure_evidence: str = Field(default="")
    provenance_hash: str = Field(default="")

class EngagementCloseoutRecord(BaseModel):
    """Engagement closeout record."""

    opinion_type: OpinionType = Field(default=OpinionType.UNMODIFIED)
    opinion_date: str = Field(default="")
    report_title: str = Field(default="")
    report_accepted: bool = Field(default=False)
    lessons_learned: List[str] = Field(default_factory=list)
    recommendations_for_next: List[str] = Field(default_factory=list)
    archive_reference: str = Field(default="")
    provenance_hash: str = Field(default="")

# =============================================================================
# INPUT / OUTPUT
# =============================================================================

class VerifierEngagementInput(BaseModel):
    """Input data model for VerifierEngagementWorkflow."""

    organization_id: str = Field(..., min_length=1, description="Organisation identifier")
    organization_name: str = Field(default="", description="Organisation display name")
    verifier_name: str = Field(default="", description="Assurance provider name")
    lead_verifier: str = Field(default="", description="Lead verifier name")
    team_members: List[str] = Field(default_factory=list)
    assurance_level: AssuranceLevel = Field(default=AssuranceLevel.LIMITED)
    reporting_period: str = Field(default="2025")
    scope_boundaries: List[str] = Field(
        default_factory=lambda: ["scope_1", "scope_2"],
    )
    materiality_threshold_pct: float = Field(default=5.0, ge=0.0, le=25.0)
    queries: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Queries/IRs from verifier with status and responses",
    )
    findings: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Verifier findings with severity and management responses",
    )
    engagement_start_date: str = Field(default="")
    engagement_end_date: str = Field(default="")
    opinion_type: str = Field(default="unmodified")
    tenant_id: str = Field(default="")
    config: Dict[str, Any] = Field(default_factory=dict)

class VerifierEngagementResult(BaseModel):
    """Complete result from verifier engagement workflow."""

    workflow_id: str = Field(..., description="Unique execution ID")
    workflow_name: str = Field(default="verifier_engagement")
    status: WorkflowStatus = Field(...)
    phases: List[PhaseResult] = Field(default_factory=list)
    total_duration_seconds: float = Field(default=0.0)
    organization_id: str = Field(default="")
    engagement_scope: Optional[EngagementScope] = Field(default=None)
    verifier_access: Optional[VerifierAccess] = Field(default=None)
    queries: List[QueryRecord] = Field(default_factory=list)
    findings: List[FindingRecord] = Field(default_factory=list)
    closeout: Optional[EngagementCloseoutRecord] = Field(default=None)
    total_queries: int = Field(default=0)
    resolved_queries: int = Field(default=0)
    total_findings: int = Field(default=0)
    closed_findings: int = Field(default=0)
    engagement_status: EngagementStatus = Field(default=EngagementStatus.SCOPING)
    provenance_hash: str = Field(default="")

# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================

class VerifierEngagementWorkflow:
    """
    5-phase workflow for verifier engagement management.

    Defines engagement scope and boundaries, onboards the verifier with
    access controls, manages query lifecycle, tracks findings through
    remediation, and closes the engagement with opinion documentation.

    Zero-hallucination: all metrics are deterministic counts and date
    calculations; no LLM calls in numeric paths; SHA-256 provenance
    on every output.

    Attributes:
        workflow_id: Unique execution identifier.
        _phase_results: Ordered phase outputs.
        _scope: Engagement scope definition.
        _access: Verifier access configuration.
        _queries: Query records.
        _findings: Finding records.
        _closeout: Engagement closeout record.

    Example:
        >>> wf = VerifierEngagementWorkflow()
        >>> inp = VerifierEngagementInput(
        ...     organization_id="org-001",
        ...     verifier_name="Big4 Auditor",
        ... )
        >>> result = await wf.execute(inp)
        >>> assert result.status == WorkflowStatus.COMPLETED
    """

    PHASE_SEQUENCE: List[EngagementPhase] = [
        EngagementPhase.ENGAGEMENT_SCOPING,
        EngagementPhase.VERIFIER_ONBOARDING,
        EngagementPhase.QUERY_MANAGEMENT,
        EngagementPhase.FINDING_TRACKING,
        EngagementPhase.ENGAGEMENT_CLOSEOUT,
    ]

    MAX_RETRIES: int = 3
    BASE_RETRY_DELAY_S: float = 1.0

    def __init__(self, config: Optional[Any] = None) -> None:
        """Initialize VerifierEngagementWorkflow."""
        self.workflow_id: str = str(uuid.uuid4())
        self.config = config
        self._phase_results: List[PhaseResult] = []
        self._scope: Optional[EngagementScope] = None
        self._access: Optional[VerifierAccess] = None
        self._queries: List[QueryRecord] = []
        self._findings: List[FindingRecord] = []
        self._closeout: Optional[EngagementCloseoutRecord] = None
        self._engagement_status: EngagementStatus = EngagementStatus.SCOPING
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    async def execute(
        self, input_data: VerifierEngagementInput,
    ) -> VerifierEngagementResult:
        """
        Execute the 5-phase verifier engagement workflow.

        Args:
            input_data: Organisation, verifier details, queries, and findings.

        Returns:
            VerifierEngagementResult with engagement lifecycle outcomes.
        """
        started_at = datetime.utcnow()
        self.logger.info(
            "Starting verifier engagement %s org=%s verifier=%s",
            self.workflow_id, input_data.organization_id,
            input_data.verifier_name,
        )

        self._reset_state()
        overall_status = WorkflowStatus.RUNNING

        phase_methods = [
            self._phase_1_engagement_scoping,
            self._phase_2_verifier_onboarding,
            self._phase_3_query_management,
            self._phase_4_finding_tracking,
            self._phase_5_engagement_closeout,
        ]

        try:
            for idx, phase_fn in enumerate(phase_methods, start=1):
                phase_result = await self._run_phase(phase_fn, input_data, idx)
                self._phase_results.append(phase_result)
                if phase_result.status == PhaseStatus.FAILED:
                    raise RuntimeError(f"Phase {idx} failed: {phase_result.errors}")

            overall_status = WorkflowStatus.COMPLETED

        except Exception as exc:
            self.logger.error("Verifier engagement failed: %s", exc, exc_info=True)
            overall_status = WorkflowStatus.FAILED
            self._phase_results.append(PhaseResult(
                phase_name="error", phase_number=0,
                status=PhaseStatus.FAILED, errors=[str(exc)],
            ))

        elapsed = (datetime.utcnow() - started_at).total_seconds()

        resolved_queries = sum(
            1 for q in self._queries if q.status == QueryStatus.RESOLVED
        )
        closed_findings = sum(
            1 for f in self._findings if f.status == FindingStatus.CLOSED
        )

        result = VerifierEngagementResult(
            workflow_id=self.workflow_id,
            status=overall_status,
            phases=self._phase_results,
            total_duration_seconds=elapsed,
            organization_id=input_data.organization_id,
            engagement_scope=self._scope,
            verifier_access=self._access,
            queries=self._queries,
            findings=self._findings,
            closeout=self._closeout,
            total_queries=len(self._queries),
            resolved_queries=resolved_queries,
            total_findings=len(self._findings),
            closed_findings=closed_findings,
            engagement_status=self._engagement_status,
        )
        result.provenance_hash = self._compute_provenance(result)

        self.logger.info(
            "Verifier engagement %s completed in %.2fs status=%s queries=%d findings=%d",
            self.workflow_id, elapsed, overall_status.value,
            len(self._queries), len(self._findings),
        )
        return result

    # -------------------------------------------------------------------------
    # Phase 1: Engagement Scoping
    # -------------------------------------------------------------------------

    async def _phase_1_engagement_scoping(
        self, input_data: VerifierEngagementInput,
    ) -> PhaseResult:
        """Define engagement scope, boundaries, and agreed criteria."""
        started = time.monotonic()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        milestones = [
            {"name": "Planning meeting", "target_week": "1"},
            {"name": "Data room setup", "target_week": "2"},
            {"name": "Fieldwork start", "target_week": "3"},
            {"name": "Fieldwork completion", "target_week": "8"},
            {"name": "Draft report", "target_week": "10"},
            {"name": "Final report", "target_week": "12"},
        ]

        scope_data = {
            "org": input_data.organization_id,
            "verifier": input_data.verifier_name,
            "level": input_data.assurance_level.value,
            "boundaries": input_data.scope_boundaries,
        }

        self._scope = EngagementScope(
            organization_id=input_data.organization_id,
            verifier_name=input_data.verifier_name,
            assurance_level=input_data.assurance_level,
            reporting_period=input_data.reporting_period,
            scope_boundaries=input_data.scope_boundaries,
            materiality_threshold_pct=input_data.materiality_threshold_pct,
            agreed_criteria="GHG Protocol Corporate Standard / ISO 14064-1:2018",
            engagement_start_date=input_data.engagement_start_date or utcnow(),
            engagement_end_date=input_data.engagement_end_date or "",
            key_milestones=milestones,
            provenance_hash=_compute_hash(scope_data),
        )

        if not input_data.verifier_name:
            warnings.append("No verifier name specified; engagement scope incomplete")

        if "scope_3" in input_data.scope_boundaries:
            warnings.append(
                "Scope 3 included in assurance boundaries; "
                "consider increased materiality threshold"
            )

        outputs["assurance_level"] = input_data.assurance_level.value
        outputs["scope_boundaries"] = input_data.scope_boundaries
        outputs["materiality_threshold_pct"] = input_data.materiality_threshold_pct
        outputs["milestones_count"] = len(milestones)

        self._engagement_status = EngagementStatus.SCOPING

        elapsed = time.monotonic() - started
        self.logger.info(
            "Phase 1 EngagementScoping: level=%s boundaries=%s materiality=%.1f%%",
            input_data.assurance_level.value,
            input_data.scope_boundaries,
            input_data.materiality_threshold_pct,
        )
        return PhaseResult(
            phase_name="engagement_scoping", phase_number=1,
            status=PhaseStatus.COMPLETED, duration_seconds=elapsed,
            outputs=outputs, warnings=warnings,
            provenance_hash=_compute_hash(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 2: Verifier Onboarding
    # -------------------------------------------------------------------------

    async def _phase_2_verifier_onboarding(
        self, input_data: VerifierEngagementInput,
    ) -> PhaseResult:
        """Onboard verifier with access controls and evidence access."""
        started = time.monotonic()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        key_contacts = [
            {"role": "Sustainability Director", "name": "TBD"},
            {"role": "Data Management Lead", "name": "TBD"},
            {"role": "Internal Audit Manager", "name": "TBD"},
            {"role": "Finance Controller", "name": "TBD"},
        ]

        access_data = {
            "verifier": input_data.verifier_name,
            "lead": input_data.lead_verifier,
            "team": input_data.team_members,
        }

        self._access = VerifierAccess(
            verifier_name=input_data.verifier_name,
            lead_verifier=input_data.lead_verifier,
            team_members=input_data.team_members,
            access_level=AccessLevel.READ_ONLY,
            data_room_url=f"https://dataroom.greenlang.io/{input_data.organization_id}/assurance",
            nda_signed=True,
            key_contacts=key_contacts,
            communication_protocol="Secure portal with email notifications",
            provenance_hash=_compute_hash(access_data),
        )

        if not input_data.lead_verifier:
            warnings.append("No lead verifier specified; onboarding incomplete")

        outputs["verifier_name"] = input_data.verifier_name
        outputs["team_size"] = len(input_data.team_members) + 1
        outputs["access_level"] = AccessLevel.READ_ONLY.value
        outputs["nda_signed"] = True
        outputs["key_contacts"] = len(key_contacts)

        self._engagement_status = EngagementStatus.ONBOARDING

        elapsed = time.monotonic() - started
        self.logger.info(
            "Phase 2 VerifierOnboarding: %s team=%d",
            input_data.verifier_name, len(input_data.team_members) + 1,
        )
        return PhaseResult(
            phase_name="verifier_onboarding", phase_number=2,
            status=PhaseStatus.COMPLETED, duration_seconds=elapsed,
            outputs=outputs, warnings=warnings,
            provenance_hash=_compute_hash(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 3: Query Management
    # -------------------------------------------------------------------------

    async def _phase_3_query_management(
        self, input_data: VerifierEngagementInput,
    ) -> PhaseResult:
        """Manage information request / query lifecycle."""
        started = time.monotonic()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        self._queries = []
        for idx, q_data in enumerate(input_data.queries, start=1):
            try:
                priority = QueryPriority(q_data.get("priority", "standard"))
            except ValueError:
                priority = QueryPriority.STANDARD

            try:
                q_status = QueryStatus(q_data.get("status", "issued"))
            except ValueError:
                q_status = QueryStatus.ISSUED

            query = QueryRecord(
                query_number=idx,
                issued_by=q_data.get("issued_by", "verifier"),
                issued_date=q_data.get("issued_date", utcnow()),
                subject=q_data.get("subject", f"Query {idx}"),
                description=q_data.get("description", ""),
                priority=priority,
                status=q_status,
                response_due_date=q_data.get("response_due_date", ""),
                response_date=q_data.get("response_date", ""),
                response_summary=q_data.get("response_summary", ""),
                resolution_date=q_data.get("resolution_date", ""),
                days_outstanding=int(q_data.get("days_outstanding", 0)),
            )
            q_hash_data = {
                "number": idx, "subject": query.subject,
                "status": q_status.value,
            }
            query.provenance_hash = _compute_hash(q_hash_data)
            self._queries.append(query)

        resolved = sum(1 for q in self._queries if q.status == QueryStatus.RESOLVED)
        escalated = sum(1 for q in self._queries if q.status == QueryStatus.ESCALATED)
        overdue = sum(1 for q in self._queries if q.days_outstanding > 14)

        outputs["total_queries"] = len(self._queries)
        outputs["resolved"] = resolved
        outputs["escalated"] = escalated
        outputs["overdue"] = overdue
        outputs["resolution_rate_pct"] = round(
            resolved / max(len(self._queries), 1) * 100.0, 1,
        )

        if overdue > 0:
            warnings.append(
                f"{overdue} queries overdue (>14 days outstanding)"
            )
        if escalated > 0:
            warnings.append(f"{escalated} queries escalated to management")

        self._engagement_status = EngagementStatus.FIELDWORK

        elapsed = time.monotonic() - started
        self.logger.info(
            "Phase 3 QueryManagement: %d queries (resolved=%d escalated=%d overdue=%d)",
            len(self._queries), resolved, escalated, overdue,
        )
        return PhaseResult(
            phase_name="query_management", phase_number=3,
            status=PhaseStatus.COMPLETED, duration_seconds=elapsed,
            outputs=outputs, warnings=warnings,
            provenance_hash=_compute_hash(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 4: Finding Tracking
    # -------------------------------------------------------------------------

    async def _phase_4_finding_tracking(
        self, input_data: VerifierEngagementInput,
    ) -> PhaseResult:
        """Track assurance findings through remediation and closure."""
        started = time.monotonic()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        self._findings = []
        for idx, f_data in enumerate(input_data.findings, start=1):
            try:
                severity = FindingSeverity(f_data.get("severity", "observation"))
            except ValueError:
                severity = FindingSeverity.OBSERVATION

            try:
                f_status = FindingStatus(f_data.get("status", "draft"))
            except ValueError:
                f_status = FindingStatus.DRAFT

            finding = FindingRecord(
                finding_number=idx,
                severity=severity,
                status=f_status,
                title=f_data.get("title", f"Finding {idx}"),
                description=f_data.get("description", ""),
                management_response=f_data.get("management_response", ""),
                remediation_action=f_data.get("remediation_action", ""),
                remediation_owner=f_data.get("remediation_owner", ""),
                remediation_deadline=f_data.get("remediation_deadline", ""),
                closure_date=f_data.get("closure_date", ""),
                closure_evidence=f_data.get("closure_evidence", ""),
            )
            f_hash_data = {
                "number": idx, "severity": severity.value,
                "status": f_status.value,
            }
            finding.provenance_hash = _compute_hash(f_hash_data)
            self._findings.append(finding)

        closed = sum(1 for f in self._findings if f.status == FindingStatus.CLOSED)
        qualifications = sum(
            1 for f in self._findings
            if f.severity == FindingSeverity.QUALIFICATION
        )
        emphasis = sum(
            1 for f in self._findings
            if f.severity == FindingSeverity.EMPHASIS_OF_MATTER
        )

        outputs["total_findings"] = len(self._findings)
        outputs["closed"] = closed
        outputs["qualifications"] = qualifications
        outputs["emphasis_of_matter"] = emphasis
        outputs["closure_rate_pct"] = round(
            closed / max(len(self._findings), 1) * 100.0, 1,
        )

        if qualifications > 0:
            warnings.append(
                f"{qualifications} qualification finding(s); "
                "opinion may be modified"
            )

        self._engagement_status = EngagementStatus.REPORTING

        elapsed = time.monotonic() - started
        self.logger.info(
            "Phase 4 FindingTracking: %d findings (closed=%d qualifications=%d)",
            len(self._findings), closed, qualifications,
        )
        return PhaseResult(
            phase_name="finding_tracking", phase_number=4,
            status=PhaseStatus.COMPLETED, duration_seconds=elapsed,
            outputs=outputs, warnings=warnings,
            provenance_hash=_compute_hash(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 5: Engagement Closeout
    # -------------------------------------------------------------------------

    async def _phase_5_engagement_closeout(
        self, input_data: VerifierEngagementInput,
    ) -> PhaseResult:
        """Close engagement with opinion documentation."""
        started = time.monotonic()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        # Determine opinion type
        try:
            opinion = OpinionType(input_data.opinion_type)
        except ValueError:
            opinion = OpinionType.UNMODIFIED

        # Check if qualification findings exist and opinion is unmodified
        qualifications = sum(
            1 for f in self._findings
            if f.severity == FindingSeverity.QUALIFICATION
            and f.status != FindingStatus.CLOSED
        )
        if qualifications > 0 and opinion == OpinionType.UNMODIFIED:
            warnings.append(
                f"Open qualification findings ({qualifications}) may impact "
                "unmodified opinion"
            )

        open_queries = sum(
            1 for q in self._queries if q.status not in (
                QueryStatus.RESOLVED, QueryStatus.WITHDRAWN,
            )
        )
        if open_queries > 0:
            warnings.append(f"{open_queries} queries still open at closeout")

        lessons = [
            "Ensure data room is fully populated before fieldwork",
            "Assign dedicated query responders per subject area",
            "Start evidence collection 8 weeks before engagement",
            "Schedule management review of findings before final report",
        ]

        recommendations = [
            "Implement automated data validation controls",
            "Establish quarterly internal assurance readiness reviews",
            "Develop evidence package template for annual reuse",
        ]

        closeout_data = {
            "opinion": opinion.value, "queries": len(self._queries),
            "findings": len(self._findings),
        }
        self._closeout = EngagementCloseoutRecord(
            opinion_type=opinion,
            opinion_date=utcnow(),
            report_title=(
                f"Assurance Report - {input_data.organization_name or input_data.organization_id} "
                f"- {input_data.reporting_period}"
            ),
            report_accepted=True,
            lessons_learned=lessons,
            recommendations_for_next=recommendations,
            archive_reference=f"archive/{input_data.organization_id}/{input_data.reporting_period}",
            provenance_hash=_compute_hash(closeout_data),
        )

        self._engagement_status = EngagementStatus.CLOSED

        outputs["opinion_type"] = opinion.value
        outputs["report_accepted"] = True
        outputs["lessons_learned"] = len(lessons)
        outputs["recommendations"] = len(recommendations)
        outputs["open_queries_at_close"] = open_queries
        outputs["open_findings_at_close"] = sum(
            1 for f in self._findings if f.status != FindingStatus.CLOSED
        )

        elapsed = time.monotonic() - started
        self.logger.info(
            "Phase 5 EngagementCloseout: opinion=%s accepted=%s",
            opinion.value, True,
        )
        return PhaseResult(
            phase_name="engagement_closeout", phase_number=5,
            status=PhaseStatus.COMPLETED, duration_seconds=elapsed,
            outputs=outputs, warnings=warnings,
            provenance_hash=_compute_hash(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase Execution Wrapper
    # -------------------------------------------------------------------------

    async def _run_phase(
        self, phase_fn: Any, input_data: VerifierEngagementInput,
        phase_number: int,
    ) -> PhaseResult:
        """Execute a phase with exponential backoff retry."""
        last_error: Optional[Exception] = None
        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                return await phase_fn(input_data)
            except Exception as exc:
                last_error = exc
                if attempt < self.MAX_RETRIES:
                    delay = self.BASE_RETRY_DELAY_S * (2 ** (attempt - 1))
                    self.logger.warning(
                        "Phase %d attempt %d/%d failed: %s. Retrying in %.1fs",
                        phase_number, attempt, self.MAX_RETRIES, exc, delay,
                    )
                    import asyncio

                    await asyncio.sleep(delay)
        return PhaseResult(
            phase_name=f"phase_{phase_number}_failed",
            phase_number=phase_number,
            status=PhaseStatus.FAILED,
            errors=[f"All {self.MAX_RETRIES} attempts failed: {last_error}"],
        )

    # -------------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------------

    def _reset_state(self) -> None:
        """Reset all internal state for a fresh execution."""
        self._phase_results = []
        self._scope = None
        self._access = None
        self._queries = []
        self._findings = []
        self._closeout = None
        self._engagement_status = EngagementStatus.SCOPING

    def _compute_provenance(self, result: VerifierEngagementResult) -> str:
        """Compute SHA-256 provenance hash from all phase hashes."""
        chain = "|".join(
            p.provenance_hash for p in result.phases if p.provenance_hash
        )
        chain += (
            f"|{result.workflow_id}|{result.organization_id}"
            f"|{result.total_queries}|{result.total_findings}"
        )
        return hashlib.sha256(chain.encode("utf-8")).hexdigest()
