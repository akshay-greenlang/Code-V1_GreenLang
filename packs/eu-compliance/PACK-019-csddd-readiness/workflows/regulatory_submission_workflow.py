# -*- coding: utf-8 -*-
"""
CSDDD Regulatory Submission Workflow
===============================================

4-phase workflow for assembling and validating regulatory submissions under
the EU Corporate Sustainability Due Diligence Directive (CSDDD / CS3D).
Covers documentation assembly, supervisory readiness, submission packaging,
and ongoing compliance tracking.

Phases:
    1. DocumentationAssembly     -- Assemble all DD documentation per articles
    2. SupervisoryReadinessCheck -- Verify readiness for supervisory authority review
    3. SubmissionPackage         -- Build compliant submission package
    4. ComplianceTracking        -- Track ongoing compliance obligations

Regulatory References:
    - Directive (EU) 2024/1760 (CSDDD / CS3D)
    - Art. 5-11: Due diligence obligations
    - Art. 13: Communicating (annual public reporting)
    - Art. 14: Reporting on CSRD
    - Art. 18-20: Supervisory authorities and enforcement
    - Art. 22: Civil liability
    - Art. 25: Administrative sanctions

Author: GreenLang Team
Version: 19.0.0
"""

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

_MODULE_VERSION = "1.0.0"


# =============================================================================
# HELPERS
# =============================================================================


def _utcnow() -> datetime:
    """Return current UTC time."""
    return datetime.now(timezone.utc)


def _new_uuid() -> str:
    """Generate a new UUID4 hex string."""
    return uuid.uuid4().hex


def _compute_hash(data: str) -> str:
    """Compute SHA-256 hex digest of *data*."""
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


# =============================================================================
# ENUMS
# =============================================================================


class WorkflowPhase(str, Enum):
    """Phases of the regulatory submission workflow."""
    DOCUMENTATION_ASSEMBLY = "documentation_assembly"
    SUPERVISORY_READINESS_CHECK = "supervisory_readiness_check"
    SUBMISSION_PACKAGE = "submission_package"
    COMPLIANCE_TRACKING = "compliance_tracking"


class WorkflowStatus(str, Enum):
    """Overall workflow execution status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class PhaseStatus(str, Enum):
    """Status of a single phase."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class DocumentStatus(str, Enum):
    """Status of a required document."""
    COMPLETE = "complete"
    PARTIAL = "partial"
    MISSING = "missing"
    OUTDATED = "outdated"
    UNDER_REVIEW = "under_review"


class ComplianceObligation(str, Enum):
    """Ongoing compliance obligation types."""
    ANNUAL_REPORTING = "annual_reporting"
    PERIODIC_ASSESSMENT = "periodic_assessment"
    STAKEHOLDER_ENGAGEMENT = "stakeholder_engagement"
    GRIEVANCE_MECHANISM = "grievance_mechanism"
    CLIMATE_PLAN_UPDATE = "climate_plan_update"
    SUPERVISORY_RESPONSE = "supervisory_response"
    CSRD_DISCLOSURE = "csrd_disclosure"


class RiskExposure(str, Enum):
    """Regulatory risk exposure level."""
    CRITICAL = "critical"
    HIGH = "high"
    MODERATE = "moderate"
    LOW = "low"


# =============================================================================
# DATA MODELS
# =============================================================================


class PhaseResult(BaseModel):
    """Result from a single workflow phase."""
    phase_name: str = Field(..., description="Phase identifier")
    status: PhaseStatus = Field(..., description="Phase completion status")
    duration_seconds: float = Field(default=0.0)
    outputs: Dict[str, Any] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")


class DDWorkflowResult(BaseModel):
    """Summary result from an upstream DD workflow."""
    workflow_name: str = Field(default="", description="Name of DD workflow")
    completed: bool = Field(default=False)
    score: float = Field(default=0.0, ge=0.0, le=100.0)
    key_findings: List[str] = Field(default_factory=list)
    gaps_count: int = Field(default=0, ge=0)
    last_updated: str = Field(default="", description="ISO date")


class CompanySubmissionProfile(BaseModel):
    """Company profile for submission context."""
    company_id: str = Field(default="")
    company_name: str = Field(default="")
    headquarters_country: str = Field(default="")
    supervisory_authority: str = Field(default="", description="Designated supervisory authority")
    reporting_year: int = Field(default=2026, ge=2024, le=2050)
    compliance_deadline: str = Field(default="", description="ISO date of compliance deadline")
    company_tier: str = Field(default="", description="group_1, group_2, etc.")
    csrd_reporting_entity: bool = Field(default=True, description="Subject to CSRD")


class RequiredDocument(BaseModel):
    """Required document for regulatory submission."""
    document_id: str = Field(default_factory=lambda: f"doc-{_new_uuid()[:8]}")
    document_name: str = Field(default="")
    csddd_article: str = Field(default="", description="Related CSDDD article")
    document_status: DocumentStatus = Field(default=DocumentStatus.MISSING)
    last_updated: str = Field(default="")
    owner: str = Field(default="", description="Responsible person/department")
    notes: str = Field(default="")


class RegulatorySubmissionInput(BaseModel):
    """Input data model for RegulatorySubmissionWorkflow."""
    company_profile: CompanySubmissionProfile = Field(
        default_factory=CompanySubmissionProfile
    )
    dd_results: List[DDWorkflowResult] = Field(
        default_factory=list, description="Results from all DD workflows"
    )
    existing_documents: List[RequiredDocument] = Field(
        default_factory=list, description="Current documentation inventory"
    )
    config: Dict[str, Any] = Field(default_factory=dict)


class ComplianceItem(BaseModel):
    """Ongoing compliance tracking item."""
    item_id: str = Field(default_factory=lambda: f"ci-{_new_uuid()[:8]}")
    obligation_type: ComplianceObligation = Field(default=ComplianceObligation.ANNUAL_REPORTING)
    description: str = Field(default="")
    frequency: str = Field(default="annual")
    next_due_date: str = Field(default="")
    responsible: str = Field(default="")
    status: str = Field(default="pending", description="pending/in_progress/completed")
    risk_if_missed: RiskExposure = Field(default=RiskExposure.HIGH)


class RegulatorySubmissionResult(BaseModel):
    """Complete result from regulatory submission workflow."""
    workflow_id: str = Field(..., description="Unique execution ID")
    workflow_name: str = Field(default="regulatory_submission")
    status: WorkflowStatus = Field(...)
    phases: List[PhaseResult] = Field(default_factory=list)
    phases_completed: int = Field(default=0, ge=0)
    duration_ms: float = Field(default=0.0)
    total_duration_seconds: float = Field(default=0.0)
    # Documentation
    documents_required: int = Field(default=0, ge=0)
    documents_complete: int = Field(default=0, ge=0)
    documents_partial: int = Field(default=0, ge=0)
    documents_missing: int = Field(default=0, ge=0)
    documentation_completeness_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    # Readiness
    supervisory_readiness_score: float = Field(default=0.0, ge=0.0, le=100.0)
    submission_ready: bool = Field(default=False)
    risk_exposure: str = Field(default="high")
    # Submission package
    package_articles_covered: int = Field(default=0, ge=0)
    package_total_articles: int = Field(default=0, ge=0)
    # Compliance tracking
    compliance_items: List[ComplianceItem] = Field(default_factory=list)
    upcoming_deadlines: List[Dict[str, Any]] = Field(default_factory=list)
    overall_compliance_score: float = Field(default=0.0, ge=0.0, le=100.0)
    reporting_year: int = Field(default=2026)
    executed_at: str = Field(default="")
    provenance_hash: str = Field(default="")


# =============================================================================
# REQUIRED DOCUMENTATION BY ARTICLE
# =============================================================================


REQUIRED_DOCS: List[Dict[str, str]] = [
    {"article": "art_5", "doc_name": "Due diligence policy", "description": "Board-approved DD policy"},
    {"article": "art_6", "doc_name": "Impact identification report", "description": "Mapping of adverse impacts"},
    {"article": "art_7", "doc_name": "Prevention action plan", "description": "Prevention/mitigation measures"},
    {"article": "art_8", "doc_name": "Remediation procedures", "description": "Corrective action procedures"},
    {"article": "art_9", "doc_name": "Remediation framework", "description": "Remediation processes and records"},
    {"article": "art_10", "doc_name": "Stakeholder engagement records", "description": "Engagement logs and outcomes"},
    {"article": "art_11", "doc_name": "Complaints procedure documentation", "description": "Grievance mechanism design"},
    {"article": "art_12", "doc_name": "Monitoring and review report", "description": "KPIs and periodic assessments"},
    {"article": "art_13", "doc_name": "Annual public statement", "description": "Public communication on DD"},
    {"article": "art_15", "doc_name": "Climate transition plan", "description": "Paris-aligned transition plan"},
]


# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================


class RegulatorySubmissionWorkflow:
    """
    4-phase CSDDD regulatory submission workflow.

    Assembles documentation, checks supervisory readiness, builds the
    submission package, and establishes ongoing compliance tracking.

    Zero-hallucination: all scoring uses deterministic formulas.
    No LLM in numeric calculation paths.

    Example:
        >>> wf = RegulatorySubmissionWorkflow()
        >>> inp = RegulatorySubmissionInput(
        ...     company_profile=CompanySubmissionProfile(company_name="Test Corp"),
        ...     dd_results=[...],
        ... )
        >>> result = await wf.execute(inp)
        >>> assert result.supervisory_readiness_score >= 0
    """

    def __init__(self, config: Optional[Any] = None) -> None:
        """Initialize RegulatorySubmissionWorkflow."""
        self.workflow_id: str = str(uuid.uuid4())
        self.config = config
        self._doc_inventory: List[RequiredDocument] = []
        self._compliance_items: List[ComplianceItem] = []
        self._readiness_score: float = 0.0
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def get_phases(self) -> List[Dict[str, str]]:
        """Return phase definitions for this workflow."""
        return [
            {"name": WorkflowPhase.DOCUMENTATION_ASSEMBLY.value, "description": "Assemble all DD documentation"},
            {"name": WorkflowPhase.SUPERVISORY_READINESS_CHECK.value, "description": "Check supervisory readiness"},
            {"name": WorkflowPhase.SUBMISSION_PACKAGE.value, "description": "Build submission package"},
            {"name": WorkflowPhase.COMPLIANCE_TRACKING.value, "description": "Set up compliance tracking"},
        ]

    def validate_inputs(self, input_data: RegulatorySubmissionInput) -> List[str]:
        """Validate workflow inputs and return list of issues."""
        issues: List[str] = []
        if not input_data.company_profile.company_name:
            issues.append("Company name is required")
        if not input_data.dd_results:
            issues.append("No DD workflow results provided")
        return issues

    async def execute(
        self,
        input_data: Optional[RegulatorySubmissionInput] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> RegulatorySubmissionResult:
        """
        Execute the 4-phase regulatory submission workflow.

        Args:
            input_data: Full input model.
            config: Configuration overrides.

        Returns:
            RegulatorySubmissionResult with readiness and compliance tracking.
        """
        if input_data is None:
            input_data = RegulatorySubmissionInput(config=config or {})

        started_at = _utcnow()
        self.logger.info("Starting regulatory submission workflow %s", self.workflow_id)
        phase_results: List[PhaseResult] = []
        overall_status = WorkflowStatus.IN_PROGRESS

        try:
            phase_results.append(await self._phase_documentation_assembly(input_data))
            phase_results.append(await self._phase_supervisory_readiness(input_data))
            phase_results.append(await self._phase_submission_package(input_data))
            phase_results.append(await self._phase_compliance_tracking(input_data))
            overall_status = WorkflowStatus.COMPLETED
        except Exception as exc:
            self.logger.error("Regulatory submission failed: %s", exc, exc_info=True)
            overall_status = WorkflowStatus.FAILED
            phase_results.append(PhaseResult(
                phase_name="error", status=PhaseStatus.FAILED, errors=[str(exc)],
            ))

        elapsed = (_utcnow() - started_at).total_seconds()
        completed_count = sum(1 for p in phase_results if p.status == PhaseStatus.COMPLETED)

        # Document stats
        complete_docs = sum(1 for d in self._doc_inventory if d.document_status == DocumentStatus.COMPLETE)
        partial_docs = sum(1 for d in self._doc_inventory if d.document_status == DocumentStatus.PARTIAL)
        missing_docs = sum(1 for d in self._doc_inventory if d.document_status == DocumentStatus.MISSING)

        completeness = round(
            (complete_docs / len(self._doc_inventory)) * 100, 1
        ) if self._doc_inventory else 0.0

        # Submission readiness
        submission_ready = self._readiness_score >= 75 and missing_docs == 0

        # Risk exposure
        if missing_docs > 3 or self._readiness_score < 30:
            risk = RiskExposure.CRITICAL.value
        elif missing_docs > 0 or self._readiness_score < 60:
            risk = RiskExposure.HIGH.value
        elif partial_docs > 2 or self._readiness_score < 80:
            risk = RiskExposure.MODERATE.value
        else:
            risk = RiskExposure.LOW.value

        # Overall compliance score
        dd_scores = [r.score for r in input_data.dd_results if r.completed]
        avg_dd_score = sum(dd_scores) / len(dd_scores) if dd_scores else 0.0
        overall_compliance = round(
            0.40 * completeness + 0.30 * self._readiness_score + 0.30 * avg_dd_score, 1
        )

        # Upcoming deadlines
        deadlines = [
            {"item_id": ci.item_id, "description": ci.description, "due_date": ci.next_due_date}
            for ci in self._compliance_items
            if ci.next_due_date
        ]
        deadlines.sort(key=lambda d: d["due_date"])

        articles_covered = sum(
            1 for d in self._doc_inventory if d.document_status in (DocumentStatus.COMPLETE, DocumentStatus.PARTIAL)
        )

        result = RegulatorySubmissionResult(
            workflow_id=self.workflow_id,
            status=overall_status,
            phases=phase_results,
            phases_completed=completed_count,
            duration_ms=round(elapsed * 1000, 2),
            total_duration_seconds=elapsed,
            documents_required=len(self._doc_inventory),
            documents_complete=complete_docs,
            documents_partial=partial_docs,
            documents_missing=missing_docs,
            documentation_completeness_pct=completeness,
            supervisory_readiness_score=self._readiness_score,
            submission_ready=submission_ready,
            risk_exposure=risk,
            package_articles_covered=articles_covered,
            package_total_articles=len(REQUIRED_DOCS),
            compliance_items=self._compliance_items,
            upcoming_deadlines=deadlines[:10],
            overall_compliance_score=overall_compliance,
            reporting_year=input_data.company_profile.reporting_year,
            executed_at=_utcnow().isoformat(),
        )
        result.provenance_hash = self._compute_provenance(result)
        self.logger.info(
            "Regulatory submission %s completed in %.2fs: readiness=%.1f%%, compliance=%.1f%%",
            self.workflow_id, elapsed, self._readiness_score, overall_compliance,
        )
        return result

    # -------------------------------------------------------------------------
    # Phase 1: Documentation Assembly
    # -------------------------------------------------------------------------

    async def _phase_documentation_assembly(
        self, input_data: RegulatorySubmissionInput,
    ) -> PhaseResult:
        """Assemble and validate all required DD documentation."""
        started = _utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        existing_map: Dict[str, RequiredDocument] = {}
        for doc in input_data.existing_documents:
            existing_map[doc.csddd_article] = doc

        self._doc_inventory = []
        for req in REQUIRED_DOCS:
            article = req["article"]
            if article in existing_map:
                doc = existing_map[article]
                self._doc_inventory.append(doc)
            else:
                # Check if DD workflow result covers this article
                workflow_mapping: Dict[str, str] = {
                    "art_5": "due_diligence_assessment",
                    "art_6": "impact_assessment",
                    "art_7": "prevention_planning",
                    "art_8": "prevention_planning",
                    "art_9": "prevention_planning",
                    "art_10": "impact_assessment",
                    "art_11": "grievance_management",
                    "art_12": "monitoring_review",
                    "art_13": "regulatory_submission",
                    "art_15": "climate_transition_planning",
                }
                related_wf = workflow_mapping.get(article, "")
                has_wf_result = any(
                    r.workflow_name == related_wf and r.completed
                    for r in input_data.dd_results
                )

                status = DocumentStatus.PARTIAL if has_wf_result else DocumentStatus.MISSING

                self._doc_inventory.append(RequiredDocument(
                    document_name=req["doc_name"],
                    csddd_article=article,
                    document_status=status,
                    notes=f"Auto-generated from {related_wf}" if has_wf_result else "Not yet prepared",
                ))

        complete = sum(1 for d in self._doc_inventory if d.document_status == DocumentStatus.COMPLETE)
        partial = sum(1 for d in self._doc_inventory if d.document_status == DocumentStatus.PARTIAL)
        missing = sum(1 for d in self._doc_inventory if d.document_status == DocumentStatus.MISSING)
        outdated = sum(1 for d in self._doc_inventory if d.document_status == DocumentStatus.OUTDATED)

        outputs["documents_required"] = len(self._doc_inventory)
        outputs["documents_complete"] = complete
        outputs["documents_partial"] = partial
        outputs["documents_missing"] = missing
        outputs["documents_outdated"] = outdated
        outputs["completeness_pct"] = round(
            (complete / len(self._doc_inventory)) * 100, 1
        ) if self._doc_inventory else 0.0
        outputs["document_inventory"] = [
            {
                "article": d.csddd_article,
                "name": d.document_name,
                "status": d.document_status.value,
            }
            for d in self._doc_inventory
        ]

        if missing > 0:
            missing_arts = [d.csddd_article for d in self._doc_inventory if d.document_status == DocumentStatus.MISSING]
            warnings.append(f"{missing} required documents missing: {', '.join(missing_arts)}")
        if outdated > 0:
            warnings.append(f"{outdated} documents are outdated and need refresh")

        elapsed = (_utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 1 DocumentationAssembly: %d required, %d complete, %d missing",
            len(self._doc_inventory), complete, missing,
        )
        return PhaseResult(
            phase_name=WorkflowPhase.DOCUMENTATION_ASSEMBLY.value,
            status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 2: Supervisory Readiness Check
    # -------------------------------------------------------------------------

    async def _phase_supervisory_readiness(
        self, input_data: RegulatorySubmissionInput,
    ) -> PhaseResult:
        """Verify readiness for supervisory authority review."""
        started = _utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        dd_results = input_data.dd_results
        cp = input_data.company_profile

        # Readiness components
        # 1. Documentation completeness (30%)
        doc_complete = sum(1 for d in self._doc_inventory if d.document_status == DocumentStatus.COMPLETE)
        doc_score = round(
            (doc_complete / len(self._doc_inventory)) * 100, 1
        ) if self._doc_inventory else 0.0

        # 2. DD workflow completion (30%)
        expected_workflows = [
            "due_diligence_assessment", "value_chain_mapping", "impact_assessment",
            "prevention_planning", "grievance_management", "monitoring_review",
            "climate_transition_planning",
        ]
        completed_workflows = sum(
            1 for wf_name in expected_workflows
            if any(r.workflow_name == wf_name and r.completed for r in dd_results)
        )
        workflow_score = round(
            (completed_workflows / len(expected_workflows)) * 100, 1
        ) if expected_workflows else 0.0

        # 3. DD quality (average scores) (20%)
        dd_scores = [r.score for r in dd_results if r.completed]
        quality_score = round(sum(dd_scores) / len(dd_scores), 1) if dd_scores else 0.0

        # 4. Governance readiness (20%)
        governance_checks: Dict[str, bool] = {
            "supervisory_authority_identified": bool(cp.supervisory_authority),
            "compliance_deadline_known": bool(cp.compliance_deadline),
            "csrd_aligned": cp.csrd_reporting_entity,
            "dd_policy_approved": any(
                d.document_status == DocumentStatus.COMPLETE
                and d.csddd_article == "art_5"
                for d in self._doc_inventory
            ),
        }
        governance_score = round(
            (sum(1 for v in governance_checks.values() if v) / len(governance_checks)) * 100, 1
        )

        self._readiness_score = round(
            0.30 * doc_score
            + 0.30 * workflow_score
            + 0.20 * quality_score
            + 0.20 * governance_score,
            1,
        )

        outputs["documentation_score"] = doc_score
        outputs["workflow_completion_score"] = workflow_score
        outputs["quality_score"] = quality_score
        outputs["governance_score"] = governance_score
        outputs["governance_checks"] = governance_checks
        outputs["supervisory_readiness_score"] = self._readiness_score
        outputs["completed_workflows"] = completed_workflows
        outputs["expected_workflows"] = len(expected_workflows)
        outputs["weight_breakdown"] = {
            "documentation": 0.30,
            "workflow_completion": 0.30,
            "quality": 0.20,
            "governance": 0.20,
        }

        # Identify gaps vs enforcement risk
        missing_critical = [
            d for d in self._doc_inventory
            if d.document_status == DocumentStatus.MISSING
            and d.csddd_article in ("art_5", "art_6", "art_7", "art_11", "art_15")
        ]
        if missing_critical:
            missing_arts = [d.csddd_article for d in missing_critical]
            warnings.append(
                f"Critical articles missing documentation: {', '.join(missing_arts)} -- Art. 25 sanction risk"
            )

        if self._readiness_score < 50:
            warnings.append(
                f"Supervisory readiness at {self._readiness_score}% -- significant preparation needed"
            )
        if not cp.supervisory_authority:
            warnings.append("Designated supervisory authority not identified")

        elapsed = (_utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 2 SupervisoryReadiness: score=%.1f%%, workflows=%d/%d",
            self._readiness_score, completed_workflows, len(expected_workflows),
        )
        return PhaseResult(
            phase_name=WorkflowPhase.SUPERVISORY_READINESS_CHECK.value,
            status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 3: Submission Package
    # -------------------------------------------------------------------------

    async def _phase_submission_package(
        self, input_data: RegulatorySubmissionInput,
    ) -> PhaseResult:
        """Build compliant submission package."""
        started = _utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        cp = input_data.company_profile

        # Package structure
        package_sections: List[Dict[str, Any]] = []
        for req in REQUIRED_DOCS:
            article = req["article"]
            doc = next(
                (d for d in self._doc_inventory if d.csddd_article == article),
                None,
            )
            doc_status = doc.document_status.value if doc else "missing"

            # Find related DD workflow result
            workflow_mapping: Dict[str, str] = {
                "art_5": "due_diligence_assessment",
                "art_6": "impact_assessment",
                "art_7": "prevention_planning",
                "art_8": "prevention_planning",
                "art_9": "prevention_planning",
                "art_10": "impact_assessment",
                "art_11": "grievance_management",
                "art_12": "monitoring_review",
                "art_13": "regulatory_submission",
                "art_15": "climate_transition_planning",
            }
            related_wf_name = workflow_mapping.get(article, "")
            related_result = next(
                (r for r in input_data.dd_results if r.workflow_name == related_wf_name),
                None,
            )

            package_sections.append({
                "article": article,
                "document_name": req["doc_name"],
                "description": req["description"],
                "document_status": doc_status,
                "workflow_completed": related_result.completed if related_result else False,
                "workflow_score": related_result.score if related_result else 0.0,
                "included_in_package": doc_status in ("complete", "partial"),
            })

        included_count = sum(1 for s in package_sections if s["included_in_package"])
        articles_with_workflow = sum(1 for s in package_sections if s["workflow_completed"])

        # Package metadata
        outputs["package_id"] = f"pkg-{_new_uuid()[:12]}"
        outputs["company_name"] = cp.company_name
        outputs["reporting_year"] = cp.reporting_year
        outputs["compliance_deadline"] = cp.compliance_deadline
        outputs["package_sections"] = package_sections
        outputs["sections_included"] = included_count
        outputs["sections_total"] = len(package_sections)
        outputs["coverage_pct"] = round(
            (included_count / len(package_sections)) * 100, 1
        ) if package_sections else 0.0
        outputs["articles_with_workflow_data"] = articles_with_workflow
        outputs["submission_ready"] = included_count == len(package_sections)

        excluded = [s for s in package_sections if not s["included_in_package"]]
        if excluded:
            excluded_arts = [s["article"] for s in excluded]
            warnings.append(f"Package incomplete -- missing sections: {', '.join(excluded_arts)}")

        elapsed = (_utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 3 SubmissionPackage: %d/%d sections included, coverage=%.1f%%",
            included_count, len(package_sections), outputs["coverage_pct"],
        )
        return PhaseResult(
            phase_name=WorkflowPhase.SUBMISSION_PACKAGE.value,
            status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 4: Compliance Tracking
    # -------------------------------------------------------------------------

    async def _phase_compliance_tracking(
        self, input_data: RegulatorySubmissionInput,
    ) -> PhaseResult:
        """Set up ongoing compliance tracking obligations."""
        started = _utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        cp = input_data.company_profile
        year = cp.reporting_year
        self._compliance_items = []

        # Define standard CSDDD compliance obligations
        standard_obligations = [
            {
                "type": ComplianceObligation.ANNUAL_REPORTING,
                "desc": "Annual public DD statement per Art. 13",
                "freq": "annual",
                "due": f"{year + 1}-03-31",
                "risk": RiskExposure.HIGH,
                "responsible": "sustainability",
            },
            {
                "type": ComplianceObligation.PERIODIC_ASSESSMENT,
                "desc": "Periodic impact assessment per Art. 12(1)",
                "freq": "annual",
                "due": f"{year}-12-31",
                "risk": RiskExposure.HIGH,
                "responsible": "compliance",
            },
            {
                "type": ComplianceObligation.STAKEHOLDER_ENGAGEMENT,
                "desc": "Stakeholder engagement per Art. 10",
                "freq": "ongoing",
                "due": f"{year}-12-31",
                "risk": RiskExposure.MODERATE,
                "responsible": "stakeholder_relations",
            },
            {
                "type": ComplianceObligation.GRIEVANCE_MECHANISM,
                "desc": "Maintain complaints procedure per Art. 11",
                "freq": "ongoing",
                "due": f"{year}-12-31",
                "risk": RiskExposure.CRITICAL,
                "responsible": "legal",
            },
            {
                "type": ComplianceObligation.CLIMATE_PLAN_UPDATE,
                "desc": "Update climate transition plan per Art. 15",
                "freq": "annual",
                "due": f"{year + 1}-06-30",
                "risk": RiskExposure.HIGH,
                "responsible": "sustainability",
            },
            {
                "type": ComplianceObligation.SUPERVISORY_RESPONSE,
                "desc": "Respond to supervisory authority inquiries per Art. 18",
                "freq": "as_needed",
                "due": "",
                "risk": RiskExposure.CRITICAL,
                "responsible": "legal",
            },
        ]

        if cp.csrd_reporting_entity:
            standard_obligations.append({
                "type": ComplianceObligation.CSRD_DISCLOSURE,
                "desc": "CSRD-aligned disclosure per Art. 14",
                "freq": "annual",
                "due": f"{year + 1}-04-30",
                "risk": RiskExposure.HIGH,
                "responsible": "finance",
            })

        for ob in standard_obligations:
            self._compliance_items.append(ComplianceItem(
                obligation_type=ob["type"],
                description=ob["desc"],
                frequency=ob["freq"],
                next_due_date=ob["due"],
                responsible=ob["responsible"],
                status="pending",
                risk_if_missed=ob["risk"],
            ))

        # Risk summary
        critical_items = sum(1 for ci in self._compliance_items if ci.risk_if_missed == RiskExposure.CRITICAL)
        high_items = sum(1 for ci in self._compliance_items if ci.risk_if_missed == RiskExposure.HIGH)

        outputs["compliance_items_count"] = len(self._compliance_items)
        outputs["critical_risk_items"] = critical_items
        outputs["high_risk_items"] = high_items
        outputs["by_frequency"] = {
            freq: sum(1 for ci in self._compliance_items if ci.frequency == freq)
            for freq in set(ci.frequency for ci in self._compliance_items)
        }
        outputs["by_responsible"] = {
            resp: sum(1 for ci in self._compliance_items if ci.responsible == resp)
            for resp in set(ci.responsible for ci in self._compliance_items)
        }
        outputs["next_deadline"] = min(
            (ci.next_due_date for ci in self._compliance_items if ci.next_due_date),
            default="",
        )

        if critical_items > 0:
            warnings.append(
                f"{critical_items} compliance items carry critical enforcement risk if missed"
            )

        elapsed = (_utcnow() - started).total_seconds()
        self.logger.info(
            "Phase 4 ComplianceTracking: %d items, %d critical",
            len(self._compliance_items), critical_items,
        )
        return PhaseResult(
            phase_name=WorkflowPhase.COMPLIANCE_TRACKING.value,
            status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _compute_provenance(self, result: RegulatorySubmissionResult) -> str:
        """Compute SHA-256 provenance hash."""
        payload = result.model_dump_json(exclude={"provenance_hash"})
        return _compute_hash(payload)

    def _hash_dict(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 of a dict."""
        raw = json.dumps(data, sort_keys=True, default=str)
        return _compute_hash(raw)
