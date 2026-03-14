# -*- coding: utf-8 -*-
"""
Auditor Collaboration Workflow
=================================

5-phase auditor engagement workflow for CSRD Enterprise Pack. Manages the
complete external audit lifecycle from portal setup through opinion issuance,
with evidence packaging per ISAE 3000 and ISAE 3410 standards.

Phases:
    1. Portal Setup: Create auditor portal, assign access, configure permissions
    2. Evidence Preparation: Package evidence per ISAE 3000/3410 requirements
    3. Review Cycles: Manage comment threads, track review iterations, enforce deadlines
    4. Finding Management: Categorize findings, track remediation, update evidence
    5. Opinion Issuance: Record assurance opinion, generate final audit report

Author: GreenLang Team
Version: 3.0.0
"""

import hashlib
import json
import logging
import uuid
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


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


class AssuranceLevel(str, Enum):
    """Assurance engagement levels per ISAE standards."""

    LIMITED_ASSURANCE = "limited_assurance"
    REASONABLE_ASSURANCE = "reasonable_assurance"


class FindingCategory(str, Enum):
    """Audit finding categories."""

    MATERIAL_MISSTATEMENT = "material_misstatement"
    SCOPE_LIMITATION = "scope_limitation"
    EMPHASIS_OF_MATTER = "emphasis_of_matter"
    OTHER_MATTER = "other_matter"


class FindingSeverity(str, Enum):
    """Finding severity classification."""

    CRITICAL = "critical"
    MAJOR = "major"
    MINOR = "minor"
    OBSERVATION = "observation"


class ReviewStatus(str, Enum):
    """Review cycle status."""

    OPEN = "open"
    IN_REVIEW = "in_review"
    COMMENTS_PENDING = "comments_pending"
    RESOLVED = "resolved"
    CLOSED = "closed"


class OpinionType(str, Enum):
    """Types of auditor opinion."""

    UNMODIFIED = "unmodified"
    MODIFIED_QUALIFIED = "modified_qualified"
    MODIFIED_ADVERSE = "modified_adverse"
    DISCLAIMER = "disclaimer"


# =============================================================================
# DATA MODELS
# =============================================================================


class PhaseResult(BaseModel):
    """Result from a single workflow phase."""

    phase_name: str = Field(..., description="Phase identifier")
    status: PhaseStatus = Field(..., description="Phase completion status")
    duration_seconds: float = Field(default=0.0, description="Phase duration in seconds")
    outputs: Dict[str, Any] = Field(default_factory=dict, description="Phase output data")
    warnings: List[str] = Field(default_factory=list, description="Warnings raised")
    errors: List[str] = Field(default_factory=list, description="Errors encountered")
    provenance_hash: str = Field(default="", description="SHA-256 of phase output")


class AuditorProfile(BaseModel):
    """External auditor profile."""

    auditor_id: str = Field(default_factory=lambda: f"aud-{uuid.uuid4().hex[:8]}")
    firm_name: str = Field(..., description="Audit firm name")
    lead_auditor: str = Field(..., description="Lead auditor name")
    email: str = Field(..., description="Lead auditor email")
    team_members: List[str] = Field(default_factory=list, description="Audit team emails")
    specializations: List[str] = Field(
        default_factory=list, description="Audit specializations (GHG, sustainability)"
    )
    isae_3000_certified: bool = Field(default=True, description="ISAE 3000 certification")
    isae_3410_certified: bool = Field(default=True, description="ISAE 3410 certification")


class EvidenceItem(BaseModel):
    """A single piece of audit evidence."""

    evidence_id: str = Field(default_factory=lambda: f"ev-{uuid.uuid4().hex[:8]}")
    title: str = Field(..., description="Evidence title")
    category: str = Field(..., description="Evidence category (data, calculation, policy, control)")
    standard: str = Field(default="ISAE_3000", description="Applicable standard")
    file_reference: str = Field(default="", description="File path or URL to evidence")
    description: str = Field(default="", description="Evidence description")
    esrs_reference: str = Field(default="", description="ESRS disclosure reference")
    provided_at: datetime = Field(default_factory=datetime.utcnow)
    provenance_hash: str = Field(default="", description="SHA-256 hash of evidence content")


class ReviewComment(BaseModel):
    """A comment in a review thread."""

    comment_id: str = Field(default_factory=lambda: f"cmt-{uuid.uuid4().hex[:8]}")
    author: str = Field(..., description="Comment author")
    role: str = Field(default="auditor", description="Author role (auditor, preparer)")
    content: str = Field(..., description="Comment text")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    parent_comment_id: Optional[str] = Field(None, description="Parent comment for threading")
    attachments: List[str] = Field(default_factory=list, description="Attached file references")


class ReviewCycle(BaseModel):
    """A single review iteration."""

    cycle_id: str = Field(default_factory=lambda: f"rc-{uuid.uuid4().hex[:8]}")
    cycle_number: int = Field(..., ge=1, description="Review cycle number")
    status: ReviewStatus = Field(default=ReviewStatus.OPEN)
    started_at: datetime = Field(default_factory=datetime.utcnow)
    deadline: Optional[datetime] = Field(None, description="Review cycle deadline")
    comments: List[ReviewComment] = Field(default_factory=list)
    evidence_requested: List[str] = Field(
        default_factory=list, description="Additional evidence requested"
    )
    evidence_provided: List[str] = Field(
        default_factory=list, description="Evidence provided in response"
    )


class AuditFinding(BaseModel):
    """An audit finding or issue."""

    finding_id: str = Field(default_factory=lambda: f"find-{uuid.uuid4().hex[:8]}")
    category: FindingCategory = Field(..., description="Finding category")
    severity: FindingSeverity = Field(..., description="Finding severity")
    title: str = Field(..., description="Finding title")
    description: str = Field(..., description="Detailed finding description")
    esrs_reference: str = Field(default="", description="Affected ESRS disclosure")
    impact_on_opinion: bool = Field(
        default=False, description="Whether this impacts the audit opinion"
    )
    remediation_status: str = Field(default="open", description="open, in_progress, resolved")
    remediation_deadline: Optional[datetime] = Field(None)
    remediation_notes: str = Field(default="", description="Remediation notes")
    reported_at: datetime = Field(default_factory=datetime.utcnow)


class AuditEngagement(BaseModel):
    """Input configuration for an audit engagement."""

    engagement_id: str = Field(
        default_factory=lambda: f"eng-{uuid.uuid4().hex[:8]}"
    )
    organization_id: str = Field(..., description="Organization being audited")
    tenant_id: str = Field(default="", description="Tenant isolation ID")
    reporting_year: int = Field(..., ge=2024, le=2050, description="Reporting year")
    assurance_level: AssuranceLevel = Field(
        default=AssuranceLevel.LIMITED_ASSURANCE, description="Target assurance level"
    )
    auditor: AuditorProfile = Field(..., description="External auditor profile")
    scope_esrs: List[str] = Field(
        default_factory=lambda: [
            "ESRS_E1", "ESRS_E2", "ESRS_S1", "ESRS_G1",
        ],
        description="ESRS standards in scope",
    )
    engagement_start: str = Field(..., description="Engagement start date (YYYY-MM-DD)")
    engagement_deadline: str = Field(..., description="Engagement deadline (YYYY-MM-DD)")
    max_review_cycles: int = Field(default=3, ge=1, le=10, description="Max review cycles")
    report_id: str = Field(default="", description="Report ID to audit")
    estimated_hours: int = Field(default=200, ge=1, description="Estimated auditor hours")

    @field_validator("engagement_start", "engagement_deadline")
    @classmethod
    def validate_iso_date(cls, v: str) -> str:
        """Validate ISO date format."""
        try:
            datetime.strptime(v, "%Y-%m-%d")
        except ValueError:
            raise ValueError(f"Date must be YYYY-MM-DD format, got: {v}")
        return v


class AuditCollaborationResult(BaseModel):
    """Complete result from the auditor collaboration workflow."""

    workflow_id: str = Field(..., description="Unique workflow execution ID")
    workflow_name: str = Field(default="auditor_collaboration")
    status: WorkflowStatus = Field(..., description="Overall workflow status")
    engagement_id: str = Field(default="", description="Audit engagement ID")
    phases: List[PhaseResult] = Field(default_factory=list, description="Per-phase results")
    total_duration_seconds: float = Field(default=0.0)
    evidence_items_provided: int = Field(default=0)
    review_cycles_completed: int = Field(default=0)
    findings_count: int = Field(default=0)
    findings_by_category: Dict[str, int] = Field(default_factory=dict)
    opinion_type: Optional[OpinionType] = Field(None, description="Final audit opinion")
    assurance_level: str = Field(default="")
    auditor_hours_tracked: float = Field(default=0.0)
    portal_url: str = Field(default="", description="Auditor portal URL")
    provenance_hash: str = Field(default="", description="SHA-256 of complete output")


# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================


class AuditorCollaborationWorkflow:
    """
    5-phase auditor engagement workflow.

    Manages the complete external audit lifecycle for CSRD reporting,
    from portal setup through opinion issuance. Packages evidence per
    ISAE 3000 (sustainability assurance) and ISAE 3410 (GHG assurance),
    manages iterative review cycles, tracks findings and remediation,
    and records the final assurance opinion.

    Attributes:
        workflow_id: Unique execution identifier.
        config: Optional EnterprisePackConfig.
        _evidence_items: Collected evidence items.
        _review_cycles: Review cycle records.
        _findings: Audit findings.
        _hours_tracked: Auditor hours tracked.

    Example:
        >>> workflow = AuditorCollaborationWorkflow()
        >>> engagement = AuditEngagement(
        ...     organization_id="org-001", reporting_year=2025,
        ...     auditor=AuditorProfile(firm_name="Big4 LLP",
        ...                            lead_auditor="J. Smith",
        ...                            email="jsmith@big4.com"),
        ...     engagement_start="2025-06-01", engagement_deadline="2025-09-30",
        ... )
        >>> result = await workflow.execute(engagement)
        >>> assert result.opinion_type is not None
    """

    def __init__(self, config: Optional[Any] = None) -> None:
        """
        Initialize the auditor collaboration workflow.

        Args:
            config: Optional EnterprisePackConfig.
        """
        self.workflow_id: str = str(uuid.uuid4())
        self.config = config
        self._evidence_items: List[EvidenceItem] = []
        self._review_cycles: List[ReviewCycle] = []
        self._findings: List[AuditFinding] = []
        self._hours_tracked: float = 0.0
        self._context: Dict[str, Any] = {}
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    async def execute(
        self, audit_engagement: AuditEngagement
    ) -> AuditCollaborationResult:
        """
        Execute the 5-phase auditor collaboration workflow.

        Args:
            audit_engagement: Validated audit engagement configuration.

        Returns:
            AuditCollaborationResult with evidence, findings, review cycles,
            and audit opinion.
        """
        started_at = datetime.utcnow()
        self.logger.info(
            "Starting auditor collaboration workflow %s for org=%s year=%d "
            "auditor=%s level=%s",
            self.workflow_id, audit_engagement.organization_id,
            audit_engagement.reporting_year, audit_engagement.auditor.firm_name,
            audit_engagement.assurance_level.value,
        )

        phase_results: List[PhaseResult] = []
        overall_status = WorkflowStatus.RUNNING
        opinion_type: Optional[OpinionType] = None

        try:
            # Phase 1: Portal Setup
            p1 = await self._phase_1_portal_setup(audit_engagement)
            phase_results.append(p1)
            if p1.status == PhaseStatus.FAILED:
                overall_status = WorkflowStatus.FAILED
                raise RuntimeError("Portal setup failed")

            # Phase 2: Evidence Preparation
            p2 = await self._phase_2_evidence_preparation(audit_engagement)
            phase_results.append(p2)

            # Phase 3: Review Cycles
            p3 = await self._phase_3_review_cycles(audit_engagement)
            phase_results.append(p3)

            # Phase 4: Finding Management
            p4 = await self._phase_4_finding_management(audit_engagement)
            phase_results.append(p4)

            # Phase 5: Opinion Issuance
            p5 = await self._phase_5_opinion_issuance(audit_engagement)
            phase_results.append(p5)
            opinion_type = OpinionType(p5.outputs.get("opinion_type", "unmodified"))

            overall_status = WorkflowStatus.COMPLETED

        except RuntimeError:
            if overall_status != WorkflowStatus.FAILED:
                overall_status = WorkflowStatus.FAILED
        except Exception as exc:
            self.logger.critical(
                "Auditor collaboration workflow %s failed: %s",
                self.workflow_id, str(exc), exc_info=True,
            )
            overall_status = WorkflowStatus.FAILED
            phase_results.append(PhaseResult(
                phase_name="workflow_error",
                status=PhaseStatus.FAILED,
                errors=[str(exc)],
                provenance_hash=self._hash_data({"error": str(exc)}),
            ))

        completed_at = datetime.utcnow()
        total_duration = (completed_at - started_at).total_seconds()

        findings_by_cat: Dict[str, int] = {}
        for f in self._findings:
            cat = f.category.value
            findings_by_cat[cat] = findings_by_cat.get(cat, 0) + 1

        provenance = self._hash_data({
            "workflow_id": self.workflow_id,
            "phases": [p.provenance_hash for p in phase_results],
        })

        self.logger.info(
            "Auditor collaboration workflow %s finished status=%s opinion=%s "
            "evidence=%d findings=%d cycles=%d in %.1fs",
            self.workflow_id, overall_status.value,
            opinion_type.value if opinion_type else "N/A",
            len(self._evidence_items), len(self._findings),
            len(self._review_cycles), total_duration,
        )

        return AuditCollaborationResult(
            workflow_id=self.workflow_id,
            status=overall_status,
            engagement_id=audit_engagement.engagement_id,
            phases=phase_results,
            total_duration_seconds=total_duration,
            evidence_items_provided=len(self._evidence_items),
            review_cycles_completed=len(self._review_cycles),
            findings_count=len(self._findings),
            findings_by_category=findings_by_cat,
            opinion_type=opinion_type,
            assurance_level=audit_engagement.assurance_level.value,
            auditor_hours_tracked=self._hours_tracked,
            portal_url=self._context.get("portal_url", ""),
            provenance_hash=provenance,
        )

    # -------------------------------------------------------------------------
    # Phase 1: Portal Setup
    # -------------------------------------------------------------------------

    async def _phase_1_portal_setup(
        self, engagement: AuditEngagement
    ) -> PhaseResult:
        """
        Create auditor collaboration portal with scoped access.

        Sets up a secure, time-limited portal for the external auditor team
        with role-based access to evidence, comments, and findings.

        Steps:
            1. Create portal instance with engagement metadata
            2. Generate auditor user accounts with SSO/invite
            3. Configure access permissions (read evidence, write comments)
            4. Set engagement timeline and milestone notifications
        """
        phase_name = "portal_setup"
        started_at = datetime.utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        # Step 1: Create portal
        portal = await self._create_auditor_portal(engagement)
        outputs["portal_id"] = portal.get("portal_id", "")
        outputs["portal_url"] = portal.get("url", "")
        self._context["portal_url"] = outputs["portal_url"]

        # Step 2: Create user accounts
        team_emails = [engagement.auditor.email] + engagement.auditor.team_members
        users_created = await self._create_auditor_accounts(
            portal.get("portal_id", ""), team_emails, engagement.auditor.firm_name
        )
        outputs["users_created"] = users_created.get("count", 0)
        outputs["invitations_sent"] = users_created.get("invitations_sent", 0)

        # Step 3: Configure permissions
        permissions = await self._configure_auditor_permissions(
            portal.get("portal_id", ""), engagement
        )
        outputs["permissions_configured"] = permissions.get("configured", False)
        outputs["access_scope"] = permissions.get("scope", [])

        # Step 4: Timeline setup
        timeline = await self._setup_engagement_timeline(engagement)
        outputs["milestones"] = timeline.get("milestones", [])
        outputs["notifications_configured"] = timeline.get("notifications", 0)

        self._hours_tracked += 2.0  # Portal setup effort

        status = PhaseStatus.COMPLETED if not errors else PhaseStatus.FAILED
        duration = (datetime.utcnow() - started_at).total_seconds()

        return PhaseResult(
            phase_name=phase_name,
            status=status,
            duration_seconds=duration,
            outputs=outputs,
            warnings=warnings,
            errors=errors,
            provenance_hash=self._hash_data(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 2: Evidence Preparation
    # -------------------------------------------------------------------------

    async def _phase_2_evidence_preparation(
        self, engagement: AuditEngagement
    ) -> PhaseResult:
        """
        Package audit evidence per ISAE 3000 and ISAE 3410 requirements.

        Collects all required evidence across ESRS disclosures, emission
        calculations, data sources, internal controls, and policies.
        Packages evidence with provenance hashes for integrity verification.

        ISAE 3000 evidence categories:
            - Governance and oversight documentation
            - Process and internal control descriptions
            - Data collection methodology
            - Stakeholder engagement records
            - Materiality assessment documentation

        ISAE 3410 evidence categories (GHG specific):
            - Emission factor sources and validation
            - Activity data completeness checks
            - Scope 1/2/3 calculation workpapers
            - Uncertainty analysis
            - Base year recalculation records

        Steps:
            1. Identify required evidence per ESRS scope
            2. Package ISAE 3000 evidence (sustainability)
            3. Package ISAE 3410 evidence (GHG-specific)
            4. Generate evidence index with cross-references
            5. Upload evidence to auditor portal
        """
        phase_name = "evidence_preparation"
        started_at = datetime.utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        # Step 1: Identify required evidence
        required = await self._identify_required_evidence(engagement)
        outputs["required_evidence_count"] = required.get("count", 0)
        outputs["by_standard"] = required.get("by_standard", {})

        # Step 2: ISAE 3000 evidence
        isae_3000_items = await self._package_isae_3000_evidence(engagement)
        self._evidence_items.extend(isae_3000_items)
        outputs["isae_3000_items"] = len(isae_3000_items)

        # Step 3: ISAE 3410 evidence
        isae_3410_items = await self._package_isae_3410_evidence(engagement)
        self._evidence_items.extend(isae_3410_items)
        outputs["isae_3410_items"] = len(isae_3410_items)

        # Step 4: Evidence index
        index = await self._generate_evidence_index(self._evidence_items, engagement)
        outputs["evidence_index_id"] = index.get("index_id", "")
        outputs["cross_references"] = index.get("cross_ref_count", 0)

        # Step 5: Upload to portal
        upload = await self._upload_evidence_to_portal(
            self._evidence_items, self._context.get("portal_url", "")
        )
        outputs["items_uploaded"] = upload.get("uploaded", 0)
        outputs["total_size_mb"] = upload.get("total_size_mb", 0.0)

        if upload.get("uploaded", 0) < len(self._evidence_items):
            warnings.append("Not all evidence items were uploaded successfully")

        self._hours_tracked += 8.0

        status = PhaseStatus.COMPLETED if not errors else PhaseStatus.FAILED
        duration = (datetime.utcnow() - started_at).total_seconds()

        return PhaseResult(
            phase_name=phase_name,
            status=status,
            duration_seconds=duration,
            outputs=outputs,
            warnings=warnings,
            errors=errors,
            provenance_hash=self._hash_data(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 3: Review Cycles
    # -------------------------------------------------------------------------

    async def _phase_3_review_cycles(
        self, engagement: AuditEngagement
    ) -> PhaseResult:
        """
        Manage iterative review cycles with comment threads and deadlines.

        Each review cycle allows the auditor to review evidence, post comments,
        request additional evidence, and the preparer to respond. Cycles continue
        until the auditor marks the review as resolved or the max cycle count
        is reached.

        Steps:
            1. Initiate first review cycle
            2. For each cycle:
               a. Auditor reviews evidence and posts comments
               b. Preparer responds and provides additional evidence
               c. Auditor resolves or requests another cycle
            3. Track cycle deadlines and send reminders
        """
        phase_name = "review_cycles"
        started_at = datetime.utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        total_comments = 0
        additional_evidence_provided = 0

        for cycle_num in range(1, engagement.max_review_cycles + 1):
            cycle_start = datetime.utcnow()
            deadline = cycle_start + timedelta(days=14)

            cycle = ReviewCycle(
                cycle_number=cycle_num,
                status=ReviewStatus.OPEN,
                started_at=cycle_start,
                deadline=deadline,
            )

            # Auditor review phase
            auditor_comments = await self._simulate_auditor_review(
                cycle_num, engagement, self._evidence_items
            )
            for comment_data in auditor_comments:
                comment = ReviewComment(
                    author=engagement.auditor.lead_auditor,
                    role="auditor",
                    content=comment_data.get("content", ""),
                )
                cycle.comments.append(comment)
                total_comments += 1

            cycle.evidence_requested = [
                c.get("evidence_requested", "")
                for c in auditor_comments
                if c.get("evidence_requested")
            ]

            # Preparer response phase
            if cycle.evidence_requested:
                preparer_response = await self._prepare_response(
                    cycle.evidence_requested, engagement
                )
                for resp in preparer_response.get("responses", []):
                    response_comment = ReviewComment(
                        author="preparer",
                        role="preparer",
                        content=resp.get("response", ""),
                        attachments=resp.get("attachments", []),
                    )
                    cycle.comments.append(response_comment)
                    total_comments += 1

                additional_evidence_provided += preparer_response.get(
                    "additional_evidence", 0
                )
                cycle.evidence_provided = preparer_response.get("evidence_ids", [])

            # Check if resolved
            resolved = await self._check_review_resolution(cycle_num, engagement)
            cycle.status = ReviewStatus.RESOLVED if resolved else ReviewStatus.COMMENTS_PENDING

            self._review_cycles.append(cycle)
            self._hours_tracked += 16.0  # Estimated hours per cycle

            if resolved:
                break

        outputs["cycles_completed"] = len(self._review_cycles)
        outputs["total_comments"] = total_comments
        outputs["additional_evidence_provided"] = additional_evidence_provided
        outputs["final_cycle_status"] = (
            self._review_cycles[-1].status.value if self._review_cycles else "none"
        )

        if len(self._review_cycles) >= engagement.max_review_cycles:
            if self._review_cycles[-1].status != ReviewStatus.RESOLVED:
                warnings.append(
                    f"Max review cycles ({engagement.max_review_cycles}) reached "
                    f"without full resolution"
                )

        status = PhaseStatus.COMPLETED if not errors else PhaseStatus.FAILED
        duration = (datetime.utcnow() - started_at).total_seconds()

        return PhaseResult(
            phase_name=phase_name,
            status=status,
            duration_seconds=duration,
            outputs=outputs,
            warnings=warnings,
            errors=errors,
            provenance_hash=self._hash_data(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 4: Finding Management
    # -------------------------------------------------------------------------

    async def _phase_4_finding_management(
        self, engagement: AuditEngagement
    ) -> PhaseResult:
        """
        Categorize and track audit findings through remediation.

        Collects findings from review cycles, categorizes them per ISAE
        standards, tracks remediation status, and determines impact on
        the final audit opinion.

        Finding categories:
            - material_misstatement: Material error in reported data
            - scope_limitation: Unable to obtain sufficient evidence
            - emphasis_of_matter: Important disclosure needing emphasis
            - other_matter: Other matters for report users

        Steps:
            1. Collect and categorize findings from review cycles
            2. Assess impact of each finding on audit opinion
            3. Track remediation for remediable findings
            4. Generate finding summary report
        """
        phase_name = "finding_management"
        started_at = datetime.utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        # Step 1: Collect and categorize findings
        raw_findings = await self._collect_findings(
            self._review_cycles, engagement
        )
        for finding_data in raw_findings:
            finding = AuditFinding(
                category=FindingCategory(finding_data.get("category", "other_matter")),
                severity=FindingSeverity(finding_data.get("severity", "observation")),
                title=finding_data.get("title", ""),
                description=finding_data.get("description", ""),
                esrs_reference=finding_data.get("esrs_reference", ""),
                impact_on_opinion=finding_data.get("impact_on_opinion", False),
            )
            self._findings.append(finding)

        outputs["findings_total"] = len(self._findings)
        outputs["findings_by_category"] = {}
        outputs["findings_by_severity"] = {}
        for f in self._findings:
            cat = f.category.value
            sev = f.severity.value
            outputs["findings_by_category"][cat] = outputs["findings_by_category"].get(cat, 0) + 1
            outputs["findings_by_severity"][sev] = outputs["findings_by_severity"].get(sev, 0) + 1

        # Step 2: Assess opinion impact
        opinion_impacting = [f for f in self._findings if f.impact_on_opinion]
        outputs["opinion_impacting_findings"] = len(opinion_impacting)
        outputs["material_misstatements"] = sum(
            1 for f in self._findings if f.category == FindingCategory.MATERIAL_MISSTATEMENT
        )

        # Step 3: Track remediation
        remediation_results = await self._track_remediation(self._findings, engagement)
        outputs["findings_remediated"] = remediation_results.get("remediated", 0)
        outputs["findings_open"] = remediation_results.get("open", 0)
        outputs["findings_in_progress"] = remediation_results.get("in_progress", 0)

        for f in self._findings:
            remediation = remediation_results.get("details", {}).get(f.finding_id, {})
            f.remediation_status = remediation.get("status", "open")
            if remediation.get("notes"):
                f.remediation_notes = remediation["notes"]

        # Step 4: Finding summary
        summary = await self._generate_finding_summary(self._findings, engagement)
        outputs["summary_report_id"] = summary.get("report_id", "")

        if outputs["material_misstatements"] > 0:
            warnings.append(
                f"{outputs['material_misstatements']} material misstatement(s) identified"
            )

        self._hours_tracked += 12.0

        status = PhaseStatus.COMPLETED if not errors else PhaseStatus.FAILED
        duration = (datetime.utcnow() - started_at).total_seconds()

        return PhaseResult(
            phase_name=phase_name,
            status=status,
            duration_seconds=duration,
            outputs=outputs,
            warnings=warnings,
            errors=errors,
            provenance_hash=self._hash_data(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 5: Opinion Issuance
    # -------------------------------------------------------------------------

    async def _phase_5_opinion_issuance(
        self, engagement: AuditEngagement
    ) -> PhaseResult:
        """
        Record the final assurance opinion and generate audit report.

        Based on findings, remediation status, and evidence sufficiency,
        determines the appropriate opinion type and generates the formal
        audit report.

        Opinion types:
            - unmodified: No material misstatements, sufficient evidence
            - modified_qualified: Material misstatement or scope limitation
            - modified_adverse: Pervasive material misstatement
            - disclaimer: Unable to obtain sufficient evidence

        Steps:
            1. Determine opinion type based on findings
            2. Draft assurance statement
            3. Generate formal audit report
            4. Record final engagement metrics
        """
        phase_name = "opinion_issuance"
        started_at = datetime.utcnow()
        errors: List[str] = []
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        # Step 1: Determine opinion type
        opinion = await self._determine_opinion(self._findings, engagement)
        opinion_type = OpinionType(opinion.get("type", "unmodified"))
        outputs["opinion_type"] = opinion_type.value
        outputs["opinion_basis"] = opinion.get("basis", "")
        outputs["opinion_date"] = datetime.utcnow().isoformat()

        # Step 2: Draft assurance statement
        statement = await self._draft_assurance_statement(
            opinion_type, engagement, self._findings
        )
        outputs["statement_id"] = statement.get("statement_id", "")
        outputs["statement_word_count"] = statement.get("word_count", 0)

        # Step 3: Generate formal report
        report = await self._generate_audit_report(
            opinion_type, engagement, self._findings,
            self._evidence_items, self._review_cycles,
        )
        outputs["report_id"] = report.get("report_id", "")
        outputs["report_pages"] = report.get("pages", 0)

        # Step 4: Final metrics
        outputs["total_auditor_hours"] = self._hours_tracked
        outputs["engagement_duration_days"] = (
            datetime.strptime(engagement.engagement_deadline, "%Y-%m-%d")
            - datetime.strptime(engagement.engagement_start, "%Y-%m-%d")
        ).days
        outputs["evidence_items_reviewed"] = len(self._evidence_items)
        outputs["review_cycles_completed"] = len(self._review_cycles)
        outputs["assurance_level"] = engagement.assurance_level.value

        if opinion_type != OpinionType.UNMODIFIED:
            warnings.append(f"Audit opinion is {opinion_type.value} (not unmodified)")

        self._hours_tracked += 8.0

        status = PhaseStatus.COMPLETED if not errors else PhaseStatus.FAILED
        duration = (datetime.utcnow() - started_at).total_seconds()

        return PhaseResult(
            phase_name=phase_name,
            status=status,
            duration_seconds=duration,
            outputs=outputs,
            warnings=warnings,
            errors=errors,
            provenance_hash=self._hash_data(outputs),
        )

    # -------------------------------------------------------------------------
    # Agent Simulation Stubs
    # -------------------------------------------------------------------------

    async def _create_auditor_portal(
        self, engagement: AuditEngagement
    ) -> Dict[str, Any]:
        """Create auditor collaboration portal."""
        portal_id = f"portal-{uuid.uuid4().hex[:8]}"
        return {
            "portal_id": portal_id,
            "url": f"https://audit.greenlang.io/{portal_id}",
        }

    async def _create_auditor_accounts(
        self, portal_id: str, emails: List[str], firm: str
    ) -> Dict[str, Any]:
        """Create user accounts for auditor team."""
        return {"count": len(emails), "invitations_sent": len(emails)}

    async def _configure_auditor_permissions(
        self, portal_id: str, engagement: AuditEngagement
    ) -> Dict[str, Any]:
        """Configure role-based access for auditor portal."""
        return {
            "configured": True,
            "scope": engagement.scope_esrs,
        }

    async def _setup_engagement_timeline(
        self, engagement: AuditEngagement
    ) -> Dict[str, Any]:
        """Set up engagement milestones and notifications."""
        return {
            "milestones": [
                "evidence_submission", "first_review", "finding_resolution",
                "draft_opinion", "final_opinion",
            ],
            "notifications": 5,
        }

    async def _identify_required_evidence(
        self, engagement: AuditEngagement
    ) -> Dict[str, Any]:
        """Identify required evidence per ESRS scope and assurance level."""
        base_count = len(engagement.scope_esrs) * 15
        return {
            "count": base_count,
            "by_standard": {"ISAE_3000": base_count // 2, "ISAE_3410": base_count // 2},
        }

    async def _package_isae_3000_evidence(
        self, engagement: AuditEngagement
    ) -> List[EvidenceItem]:
        """Package evidence per ISAE 3000 (sustainability assurance)."""
        categories = [
            "governance_documentation", "process_controls", "data_methodology",
            "stakeholder_records", "materiality_assessment", "policy_documents",
        ]
        items = []
        for cat in categories:
            for esrs in engagement.scope_esrs:
                item = EvidenceItem(
                    title=f"{cat} - {esrs}",
                    category=cat,
                    standard="ISAE_3000",
                    esrs_reference=esrs,
                    provenance_hash=self._hash_data({"cat": cat, "esrs": esrs}),
                )
                items.append(item)
        return items

    async def _package_isae_3410_evidence(
        self, engagement: AuditEngagement
    ) -> List[EvidenceItem]:
        """Package evidence per ISAE 3410 (GHG assurance)."""
        categories = [
            "emission_factors", "activity_data", "calculation_workpapers",
            "uncertainty_analysis", "base_year_records",
        ]
        items = []
        for cat in categories:
            item = EvidenceItem(
                title=f"GHG {cat}",
                category=cat,
                standard="ISAE_3410",
                provenance_hash=self._hash_data({"cat": cat}),
            )
            items.append(item)
        return items

    async def _generate_evidence_index(
        self, items: List[EvidenceItem], engagement: AuditEngagement
    ) -> Dict[str, Any]:
        """Generate cross-referenced evidence index."""
        return {
            "index_id": f"idx-{uuid.uuid4().hex[:8]}",
            "cross_ref_count": len(items) * 2,
        }

    async def _upload_evidence_to_portal(
        self, items: List[EvidenceItem], portal_url: str
    ) -> Dict[str, Any]:
        """Upload evidence items to auditor portal."""
        return {"uploaded": len(items), "total_size_mb": len(items) * 0.5}

    async def _simulate_auditor_review(
        self, cycle_num: int, engagement: AuditEngagement,
        evidence: List[EvidenceItem],
    ) -> List[Dict[str, Any]]:
        """Simulate auditor review comments."""
        comments = []
        if cycle_num == 1:
            comments = [
                {"content": "Please provide additional detail on Scope 3 methodology",
                 "evidence_requested": "scope3_methodology_detail"},
                {"content": "Emission factor sources need verification",
                 "evidence_requested": "ef_source_verification"},
            ]
        elif cycle_num == 2:
            comments = [
                {"content": "Responses satisfactory. Minor clarification needed on base year.",
                 "evidence_requested": ""},
            ]
        return comments

    async def _prepare_response(
        self, requests: List[str], engagement: AuditEngagement
    ) -> Dict[str, Any]:
        """Prepare responses to auditor evidence requests."""
        responses = [
            {"response": f"Provided: {req}", "attachments": [f"att_{req}"], "evidence_id": req}
            for req in requests if req
        ]
        return {
            "responses": responses,
            "additional_evidence": len(responses),
            "evidence_ids": [r["evidence_id"] for r in responses],
        }

    async def _check_review_resolution(
        self, cycle_num: int, engagement: AuditEngagement
    ) -> bool:
        """Check if the review cycle is resolved."""
        return cycle_num >= 2

    async def _collect_findings(
        self, cycles: List[ReviewCycle], engagement: AuditEngagement
    ) -> List[Dict[str, Any]]:
        """Collect findings from review cycles."""
        return [
            {
                "category": "emphasis_of_matter",
                "severity": "minor",
                "title": "Scope 3 estimation uncertainty",
                "description": "Scope 3 Cat 1 uses industry-average emission factors",
                "esrs_reference": "ESRS_E1",
                "impact_on_opinion": False,
            },
        ]

    async def _track_remediation(
        self, findings: List[AuditFinding], engagement: AuditEngagement
    ) -> Dict[str, Any]:
        """Track remediation status of findings."""
        return {
            "remediated": 0,
            "open": len(findings),
            "in_progress": 0,
            "details": {},
        }

    async def _generate_finding_summary(
        self, findings: List[AuditFinding], engagement: AuditEngagement
    ) -> Dict[str, Any]:
        """Generate finding summary report."""
        return {"report_id": f"fsr-{uuid.uuid4().hex[:8]}"}

    async def _determine_opinion(
        self, findings: List[AuditFinding], engagement: AuditEngagement
    ) -> Dict[str, Any]:
        """Determine audit opinion based on findings."""
        material_count = sum(
            1 for f in findings if f.category == FindingCategory.MATERIAL_MISSTATEMENT
        )
        scope_limit_count = sum(
            1 for f in findings if f.category == FindingCategory.SCOPE_LIMITATION
        )

        if material_count > 2:
            return {"type": "modified_adverse", "basis": "Pervasive material misstatements"}
        elif material_count > 0 or scope_limit_count > 0:
            return {"type": "modified_qualified", "basis": "Material misstatement or scope limitation"}
        else:
            return {"type": "unmodified", "basis": "No material misstatements identified"}

    async def _draft_assurance_statement(
        self, opinion: OpinionType, engagement: AuditEngagement,
        findings: List[AuditFinding],
    ) -> Dict[str, Any]:
        """Draft the formal assurance statement."""
        return {
            "statement_id": f"stmt-{uuid.uuid4().hex[:8]}",
            "word_count": 1500,
        }

    async def _generate_audit_report(
        self, opinion: OpinionType, engagement: AuditEngagement,
        findings: List[AuditFinding], evidence: List[EvidenceItem],
        cycles: List[ReviewCycle],
    ) -> Dict[str, Any]:
        """Generate the formal audit report."""
        return {
            "report_id": f"audrpt-{uuid.uuid4().hex[:8]}",
            "pages": 25,
        }

    # -------------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------------

    def _hash_data(self, data: Any) -> str:
        """Calculate SHA-256 hash for provenance tracking."""
        serialized = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
