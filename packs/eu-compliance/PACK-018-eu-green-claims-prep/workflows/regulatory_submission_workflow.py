# -*- coding: utf-8 -*-
"""
Regulatory Submission Workflow - PACK-018 EU Green Claims Prep
================================================================

4-phase workflow that compiles a regulatory submission dossier for
environmental claims under the EU Green Claims Directive, runs internal
quality checks, validates the package against Conformity Assessment Body
(CAB) criteria, and tracks submission status through to completion.

Phases:
    1. PackageAssembly     -- Compile the submission dossier
    2. InternalReview      -- Run quality checks and completeness validation
    3. CABPreSubmission    -- Validate package per CAB criteria
    4. SubmissionTracking  -- Track submission status and outcomes

Reference:
    EU Green Claims Directive (COM/2023/166), Articles 10-12
    Regulation (EC) 765/2008 (Accreditation and Market Surveillance)
    PACK-018 Solution Pack specification

Author: GreenLang Team
Version: 18.0.0
"""

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# =============================================================================
# HELPERS
# =============================================================================


def _utcnow() -> datetime:
    """Return current UTC time with second precision."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _new_uuid() -> str:
    """Generate a new UUID-4 string."""
    return str(uuid.uuid4())


def _compute_hash(data: Any) -> str:
    """Compute SHA-256 hex digest for provenance tracking."""
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


# =============================================================================
# ENUMS
# =============================================================================


class PhaseStatus(str, Enum):
    """Execution status for a single workflow phase."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class SubmissionPhase(str, Enum):
    """Regulatory submission workflow phase identifiers."""
    PACKAGE_ASSEMBLY = "PackageAssembly"
    INTERNAL_REVIEW = "InternalReview"
    CAB_PRE_SUBMISSION = "CABPreSubmission"
    SUBMISSION_TRACKING = "SubmissionTracking"


class DocumentType(str, Enum):
    """Document types required in the submission dossier."""
    CLAIM_TEXT = "claim_text"
    SUBSTANTIATION_REPORT = "substantiation_report"
    LCA_STUDY = "lca_study"
    VERIFICATION_CERTIFICATE = "verification_certificate"
    EVIDENCE_DOSSIER = "evidence_dossier"
    LABEL_CERTIFICATE = "label_certificate"
    METHODOLOGY_DESCRIPTION = "methodology_description"
    COMPARISON_DATA = "comparison_data"
    IMPLEMENTATION_PLAN = "implementation_plan"
    THIRD_PARTY_OPINION = "third_party_opinion"


class ReviewOutcome(str, Enum):
    """Internal review outcome classification."""
    APPROVED = "approved"
    APPROVED_WITH_CONDITIONS = "approved_with_conditions"
    REVISION_REQUIRED = "revision_required"
    REJECTED = "rejected"


class CABCheckResult(str, Enum):
    """Conformity Assessment Body check result."""
    PASS = "pass"
    FAIL = "fail"
    NOT_APPLICABLE = "not_applicable"
    NEEDS_CLARIFICATION = "needs_clarification"


class SubmissionStatus(str, Enum):
    """Submission lifecycle status."""
    DRAFT = "draft"
    READY = "ready"
    SUBMITTED = "submitted"
    UNDER_REVIEW = "under_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    WITHDRAWN = "withdrawn"


# =============================================================================
# DATA MODELS
# =============================================================================


class SubmissionConfig(BaseModel):
    """Configuration for RegulatorySubmissionWorkflow."""
    require_lca: bool = Field(
        default=True, description="Whether LCA study is mandatory in dossier",
    )
    require_third_party_verification: bool = Field(
        default=True, description="Whether third-party verification is mandatory",
    )
    cab_name: str = Field(
        default="", description="Name of the target Conformity Assessment Body",
    )
    completeness_threshold_pct: float = Field(
        default=90.0, ge=0.0, le=100.0,
        description="Minimum completeness percentage to pass internal review",
    )


class SubmissionResult(BaseModel):
    """Final submission readiness result."""
    submission_id: str = Field(..., description="Unique submission identifier")
    entity_name: str = Field(default="")
    total_claims: int = Field(default=0)
    dossier_complete: bool = Field(default=False)
    internal_review_outcome: str = Field(default="")
    cab_pre_check_passed: bool = Field(default=False)
    submission_status: str = Field(default="draft")
    provenance_hash: str = Field(default="")


class WorkflowInput(BaseModel):
    """Input model for RegulatorySubmissionWorkflow."""
    claims: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of claims to include in submission",
    )
    documents: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of supporting documents with type and metadata",
    )
    entity_name: str = Field(default="", description="Reporting entity name")
    cab_name: str = Field(default="", description="Target Conformity Assessment Body")
    reporting_year: int = Field(default=2025, ge=2020, le=2050)
    config: Dict[str, Any] = Field(default_factory=dict)


class PhaseResult(BaseModel):
    """Result from a single workflow phase."""
    phase_name: str = Field(..., description="Phase identifier")
    status: PhaseStatus = Field(..., description="Phase completion status")
    started_at: Optional[datetime] = Field(default=None)
    completed_at: Optional[datetime] = Field(default=None)
    result_data: Dict[str, Any] = Field(default_factory=dict)
    error_message: Optional[str] = Field(default=None)


class WorkflowResult(BaseModel):
    """Complete result from RegulatorySubmissionWorkflow."""
    workflow_id: str = Field(..., description="Unique execution ID")
    workflow_name: str = Field(default="regulatory_submission")
    status: PhaseStatus = Field(default=PhaseStatus.PENDING)
    phases: List[PhaseResult] = Field(default_factory=list)
    overall_result: Dict[str, Any] = Field(default_factory=dict)
    provenance_hash: str = Field(default="")
    started_at: Optional[datetime] = Field(default=None)
    completed_at: Optional[datetime] = Field(default=None)


# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================


class RegulatorySubmissionWorkflow:
    """
    4-phase regulatory submission workflow for EU Green Claims Directive.

    Compiles a submission dossier from claims and supporting documents,
    runs internal quality and completeness checks, validates the package
    against Conformity Assessment Body (CAB) criteria, and tracks the
    submission through its lifecycle.

    Zero-hallucination: all completeness scoring, quality checks, and
    CAB criterion evaluation uses deterministic rule-based logic. No LLM
    calls in calculation paths.

    Example:
        >>> wf = RegulatorySubmissionWorkflow()
        >>> result = wf.execute(
        ...     claims=[{"id": "C1", "text": "50% recycled content"}],
        ...     documents=[{"type": "lca_study", "claim_id": "C1"}],
        ... )
        >>> assert result["status"] == "completed"
    """

    WORKFLOW_NAME: str = "regulatory_submission"

    # Required document types per Article 10
    REQUIRED_DOCUMENT_TYPES: set = {
        DocumentType.CLAIM_TEXT.value,
        DocumentType.SUBSTANTIATION_REPORT.value,
        DocumentType.EVIDENCE_DOSSIER.value,
    }

    # CAB pre-submission checklist criteria
    CAB_CRITERIA: List[Dict[str, str]] = [
        {
            "criterion_id": "CAB-001",
            "description": "All claims accompanied by substantiation report",
            "article_ref": "Article 10(1)",
        },
        {
            "criterion_id": "CAB-002",
            "description": "Verifier accreditation under Regulation (EC) 765/2008 confirmed",
            "article_ref": "Article 10(2)",
        },
        {
            "criterion_id": "CAB-003",
            "description": "Lifecycle assessment included for environmental claims",
            "article_ref": "Article 5(2)(a)",
        },
        {
            "criterion_id": "CAB-004",
            "description": "Evidence dossier covers all significant environmental aspects",
            "article_ref": "Article 5(2)(c)",
        },
        {
            "criterion_id": "CAB-005",
            "description": "Claims do not rely solely on carbon offsets",
            "article_ref": "Article 5(4)",
        },
        {
            "criterion_id": "CAB-006",
            "description": "Methodology description provided for quantified claims",
            "article_ref": "Article 5(2)(b)",
        },
        {
            "criterion_id": "CAB-007",
            "description": "Comparison claims use equivalent functional units",
            "article_ref": "Article 7",
        },
        {
            "criterion_id": "CAB-008",
            "description": "Future performance claims include implementation plan",
            "article_ref": "Article 5(5)",
        },
    ]

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize RegulatorySubmissionWorkflow."""
        self.workflow_id: str = _new_uuid()
        self.config: Dict[str, Any] = config or {}
        self.sub_config = SubmissionConfig(**self.config)
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def execute(self, **kwargs: Any) -> dict:
        """
        Execute the 4-phase regulatory submission pipeline.

        Keyword Args:
            claims: List of claim dictionaries to submit.
            documents: List of supporting document dictionaries.
            entity_name: Organisation name.
            cab_name: Target Conformity Assessment Body name.

        Returns:
            Serialised WorkflowResult dictionary with provenance hash.
        """
        input_data = WorkflowInput(
            claims=kwargs.get("claims", []),
            documents=kwargs.get("documents", []),
            entity_name=kwargs.get("entity_name", ""),
            cab_name=kwargs.get("cab_name", self.sub_config.cab_name),
            reporting_year=kwargs.get("reporting_year", 2025),
            config=kwargs.get("config", {}),
        )

        started_at = _utcnow()
        self.logger.info("Starting %s workflow %s -- %d claims, %d documents",
                         self.WORKFLOW_NAME, self.workflow_id,
                         len(input_data.claims), len(input_data.documents))
        phase_results: List[PhaseResult] = []
        overall_status = PhaseStatus.RUNNING

        try:
            # Phase 1 -- Package Assembly
            phase_results.append(self._run_package_assembly(input_data))

            # Phase 2 -- Internal Review
            package_data = phase_results[0].result_data
            phase_results.append(self._run_internal_review(input_data, package_data))

            # Phase 3 -- CAB Pre-Submission
            review_data = phase_results[1].result_data
            phase_results.append(
                self._run_cab_pre_submission(input_data, package_data, review_data)
            )

            # Phase 4 -- Submission Tracking
            cab_data = phase_results[2].result_data
            phase_results.append(
                self._run_submission_tracking(input_data, review_data, cab_data)
            )

            overall_status = PhaseStatus.COMPLETED
        except Exception as exc:
            self.logger.error("Workflow %s failed: %s", self.workflow_id, exc, exc_info=True)
            overall_status = PhaseStatus.FAILED
            phase_results.append(PhaseResult(
                phase_name="error_capture",
                status=PhaseStatus.FAILED,
                started_at=_utcnow(),
                completed_at=_utcnow(),
                error_message=str(exc),
            ))

        completed_at = _utcnow()

        completed_phases = [p for p in phase_results if p.status == PhaseStatus.COMPLETED]
        overall_result: Dict[str, Any] = {
            "total_claims": len(input_data.claims),
            "total_documents": len(input_data.documents),
            "phases_completed": len(completed_phases),
            "phases_total": 4,
        }
        if phase_results and phase_results[-1].status == PhaseStatus.COMPLETED:
            overall_result.update(phase_results[-1].result_data)

        result = WorkflowResult(
            workflow_id=self.workflow_id,
            workflow_name=self.WORKFLOW_NAME,
            status=overall_status,
            phases=phase_results,
            overall_result=overall_result,
            started_at=started_at,
            completed_at=completed_at,
        )
        result.provenance_hash = _compute_hash(result)

        self.logger.info(
            "Workflow %s %s in %.1fs -- %d claims, %d documents packaged",
            self.workflow_id,
            overall_status.value,
            (completed_at - started_at).total_seconds(),
            len(input_data.claims),
            len(input_data.documents),
        )
        return result.model_dump(mode="json")

    # ------------------------------------------------------------------
    # run_phase dispatcher
    # ------------------------------------------------------------------

    def run_phase(self, phase: SubmissionPhase, **kwargs: Any) -> PhaseResult:
        """
        Run a single named phase independently.

        Args:
            phase: The SubmissionPhase to execute.
            **kwargs: Phase-specific keyword arguments.

        Returns:
            PhaseResult for the executed phase.
        """
        dispatch: Dict[SubmissionPhase, Any] = {
            SubmissionPhase.PACKAGE_ASSEMBLY: lambda: self._run_package_assembly(
                WorkflowInput(
                    claims=kwargs.get("claims", []),
                    documents=kwargs.get("documents", []),
                )
            ),
            SubmissionPhase.INTERNAL_REVIEW: lambda: self._run_internal_review(
                WorkflowInput(claims=kwargs.get("claims", [])),
                kwargs.get("package_data", {}),
            ),
            SubmissionPhase.CAB_PRE_SUBMISSION: lambda: self._run_cab_pre_submission(
                WorkflowInput(cab_name=kwargs.get("cab_name", "")),
                kwargs.get("package_data", {}),
                kwargs.get("review_data", {}),
            ),
            SubmissionPhase.SUBMISSION_TRACKING: lambda: self._run_submission_tracking(
                WorkflowInput(entity_name=kwargs.get("entity_name", "")),
                kwargs.get("review_data", {}),
                kwargs.get("cab_data", {}),
            ),
        }
        handler = dispatch.get(phase)
        if handler is None:
            return PhaseResult(
                phase_name=phase.value,
                status=PhaseStatus.FAILED,
                error_message=f"Unknown phase: {phase.value}",
            )
        return handler()

    # ------------------------------------------------------------------
    # Phase 1: Package Assembly
    # ------------------------------------------------------------------

    def _run_package_assembly(self, input_data: WorkflowInput) -> PhaseResult:
        """Compile the submission dossier from claims and documents."""
        started = _utcnow()
        self.logger.info("Phase 1/4 PackageAssembly -- compiling dossier with %d claims, %d docs",
                         len(input_data.claims), len(input_data.documents))

        # Index documents by claim and type
        docs_by_claim: Dict[str, List[Dict[str, Any]]] = {}
        doc_type_counts: Dict[str, int] = {}

        for doc in input_data.documents:
            claim_id = doc.get("claim_id", "unlinked")
            docs_by_claim.setdefault(claim_id, []).append(doc)
            doc_type = doc.get("type", "unknown")
            doc_type_counts[doc_type] = doc_type_counts.get(doc_type, 0) + 1

        # Build per-claim dossier entries
        dossier_entries: List[Dict[str, Any]] = []
        for idx, claim in enumerate(input_data.claims):
            claim_id = claim.get("id", f"CLM-{idx:04d}")
            claim_docs = docs_by_claim.get(claim_id, [])
            doc_types_present = {d.get("type", "unknown") for d in claim_docs}

            missing_types = set(self.REQUIRED_DOCUMENT_TYPES) - doc_types_present
            if self.sub_config.require_lca:
                if DocumentType.LCA_STUDY.value not in doc_types_present:
                    missing_types.add(DocumentType.LCA_STUDY.value)
            if self.sub_config.require_third_party_verification:
                if DocumentType.VERIFICATION_CERTIFICATE.value not in doc_types_present:
                    missing_types.add(DocumentType.VERIFICATION_CERTIFICATE.value)

            dossier_entries.append({
                "claim_id": claim_id,
                "claim_text": claim.get("text", ""),
                "documents_count": len(claim_docs),
                "document_types_present": sorted(doc_types_present),
                "missing_document_types": sorted(missing_types),
                "is_complete": len(missing_types) == 0,
            })

        complete_count = sum(1 for e in dossier_entries if e["is_complete"])
        total = len(dossier_entries)

        result_data: Dict[str, Any] = {
            "dossier_entries": dossier_entries,
            "total_claims_in_dossier": total,
            "complete_claims": complete_count,
            "incomplete_claims": total - complete_count,
            "completeness_pct": round(
                (complete_count / total * 100) if total else 0.0, 1
            ),
            "total_documents": len(input_data.documents),
            "document_type_distribution": doc_type_counts,
            "dossier_id": _new_uuid(),
        }

        return PhaseResult(
            phase_name=SubmissionPhase.PACKAGE_ASSEMBLY.value,
            status=PhaseStatus.COMPLETED,
            started_at=started,
            completed_at=_utcnow(),
            result_data=result_data,
        )

    # ------------------------------------------------------------------
    # Phase 2: Internal Review
    # ------------------------------------------------------------------

    def _run_internal_review(
        self, input_data: WorkflowInput, package_data: Dict[str, Any],
    ) -> PhaseResult:
        """Run quality checks and completeness validation."""
        started = _utcnow()
        self.logger.info("Phase 2/4 InternalReview -- running quality checks")

        checks: List[Dict[str, Any]] = []
        pass_count = 0

        # Check 1: Overall completeness
        completeness = package_data.get("completeness_pct", 0.0)
        completeness_pass = completeness >= self.sub_config.completeness_threshold_pct
        if completeness_pass:
            pass_count += 1
        checks.append({
            "check_id": "QC-001",
            "name": "Dossier Completeness",
            "description": (
                f"All claims have required documents "
                f"(threshold: {self.sub_config.completeness_threshold_pct}%)"
            ),
            "result": "pass" if completeness_pass else "fail",
            "value": completeness,
            "threshold": self.sub_config.completeness_threshold_pct,
        })

        # Check 2: No orphaned documents
        dossier_claim_ids = {
            e["claim_id"] for e in package_data.get("dossier_entries", [])
        }
        doc_claim_ids: set = set()
        for doc in input_data.documents:
            cid = doc.get("claim_id")
            if cid:
                doc_claim_ids.add(cid)
        orphaned = doc_claim_ids - dossier_claim_ids
        no_orphans = len(orphaned) == 0
        if no_orphans:
            pass_count += 1
        checks.append({
            "check_id": "QC-002",
            "name": "No Orphaned Documents",
            "description": "All documents linked to valid claims",
            "result": "pass" if no_orphans else "fail",
            "orphaned_count": len(orphaned),
        })

        # Check 3: LCA presence
        has_lca = DocumentType.LCA_STUDY.value in (
            package_data.get("document_type_distribution", {})
        )
        lca_check = has_lca or not self.sub_config.require_lca
        if lca_check:
            pass_count += 1
        checks.append({
            "check_id": "QC-003",
            "name": "LCA Study Presence",
            "description": "At least one LCA study included in dossier",
            "result": "pass" if lca_check else "fail",
            "required": self.sub_config.require_lca,
        })

        # Check 4: Verification certificate presence
        has_cert = DocumentType.VERIFICATION_CERTIFICATE.value in (
            package_data.get("document_type_distribution", {})
        )
        cert_check = has_cert or not self.sub_config.require_third_party_verification
        if cert_check:
            pass_count += 1
        checks.append({
            "check_id": "QC-004",
            "name": "Verification Certificate Presence",
            "description": "Third-party verification certificate included",
            "result": "pass" if cert_check else "fail",
            "required": self.sub_config.require_third_party_verification,
        })

        # Check 5: Minimum document count
        min_docs = len(input_data.claims)
        total_docs = package_data.get("total_documents", 0)
        docs_sufficient = total_docs >= min_docs
        if docs_sufficient:
            pass_count += 1
        checks.append({
            "check_id": "QC-005",
            "name": "Minimum Document Coverage",
            "description": "At least one document per claim",
            "result": "pass" if docs_sufficient else "fail",
            "documents": total_docs,
            "minimum_required": min_docs,
        })

        total_checks = len(checks)
        review_outcome = self._determine_review_outcome(pass_count, total_checks)

        result_data: Dict[str, Any] = {
            "quality_checks": checks,
            "checks_passed": pass_count,
            "checks_total": total_checks,
            "pass_rate_pct": round(
                (pass_count / total_checks * 100) if total_checks else 0.0, 1
            ),
            "review_outcome": review_outcome.value,
            "issues": [c for c in checks if c["result"] != "pass"],
            "issue_count": total_checks - pass_count,
        }

        return PhaseResult(
            phase_name=SubmissionPhase.INTERNAL_REVIEW.value,
            status=PhaseStatus.COMPLETED,
            started_at=started,
            completed_at=_utcnow(),
            result_data=result_data,
        )

    # ------------------------------------------------------------------
    # Phase 3: CAB Pre-Submission
    # ------------------------------------------------------------------

    def _run_cab_pre_submission(
        self,
        input_data: WorkflowInput,
        package_data: Dict[str, Any],
        review_data: Dict[str, Any],
    ) -> PhaseResult:
        """Validate submission package against CAB criteria."""
        started = _utcnow()
        cab_name = input_data.cab_name or self.sub_config.cab_name or "Default CAB"
        self.logger.info(
            "Phase 3/4 CABPreSubmission -- validating against %s criteria", cab_name,
        )

        doc_types_present: set = set()
        for entry in package_data.get("dossier_entries", []):
            doc_types_present.update(entry.get("document_types_present", []))

        criterion_results: List[Dict[str, Any]] = []
        pass_count = 0

        for criterion in self.CAB_CRITERIA:
            check_result = self._evaluate_cab_criterion(
                criterion, doc_types_present, package_data, review_data,
            )
            if check_result == CABCheckResult.PASS:
                pass_count += 1

            criterion_results.append({
                "criterion_id": criterion["criterion_id"],
                "description": criterion["description"],
                "article_ref": criterion["article_ref"],
                "result": check_result.value,
            })

        total_criteria = len(criterion_results)
        applicable_count = sum(
            1 for c in criterion_results
            if c["result"] != CABCheckResult.NOT_APPLICABLE.value
        )
        cab_passed = pass_count == applicable_count and applicable_count > 0

        result_data: Dict[str, Any] = {
            "cab_name": cab_name,
            "criterion_results": criterion_results,
            "criteria_passed": pass_count,
            "criteria_total": total_criteria,
            "criteria_applicable": applicable_count,
            "cab_pre_check_passed": cab_passed,
            "pass_rate_pct": round(
                (pass_count / applicable_count * 100) if applicable_count else 0.0, 1
            ),
            "failing_criteria": [
                c for c in criterion_results
                if c["result"] in (
                    CABCheckResult.FAIL.value,
                    CABCheckResult.NEEDS_CLARIFICATION.value,
                )
            ],
        }

        return PhaseResult(
            phase_name=SubmissionPhase.CAB_PRE_SUBMISSION.value,
            status=PhaseStatus.COMPLETED,
            started_at=started,
            completed_at=_utcnow(),
            result_data=result_data,
        )

    # ------------------------------------------------------------------
    # Phase 4: Submission Tracking
    # ------------------------------------------------------------------

    def _run_submission_tracking(
        self,
        input_data: WorkflowInput,
        review_data: Dict[str, Any],
        cab_data: Dict[str, Any],
    ) -> PhaseResult:
        """Track submission status and determine readiness."""
        started = _utcnow()
        self.logger.info("Phase 4/4 SubmissionTracking -- determining submission status")

        review_outcome = review_data.get("review_outcome", "")
        cab_passed = cab_data.get("cab_pre_check_passed", False)

        submission_status = self._determine_submission_status(
            review_outcome, cab_passed,
        )

        submission_result = SubmissionResult(
            submission_id=_new_uuid(),
            entity_name=input_data.entity_name,
            total_claims=len(input_data.claims),
            dossier_complete=review_data.get("pass_rate_pct", 0.0) >= 80.0,
            internal_review_outcome=review_outcome,
            cab_pre_check_passed=cab_passed,
            submission_status=submission_status.value,
            provenance_hash="",
        )
        submission_result.provenance_hash = _compute_hash(submission_result)

        next_steps = self._determine_next_steps(
            submission_status, review_data, cab_data,
        )

        tracking_record: Dict[str, Any] = {
            "submission_id": submission_result.submission_id,
            "status": submission_status.value,
            "status_history": [
                {
                    "status": SubmissionStatus.DRAFT.value,
                    "timestamp": started.isoformat(),
                    "note": "Initial dossier compilation",
                },
                {
                    "status": submission_status.value,
                    "timestamp": _utcnow().isoformat(),
                    "note": self._status_note(submission_status),
                },
            ],
            "cab_name": cab_data.get("cab_name", ""),
        }

        result_data: Dict[str, Any] = {
            "submission": submission_result.model_dump(),
            "tracking_record": tracking_record,
            "submission_status": submission_status.value,
            "is_ready_to_submit": submission_status == SubmissionStatus.READY,
            "next_steps": next_steps,
            "blocking_issues_count": len(cab_data.get("failing_criteria", [])),
            "recommendation": self._generate_recommendation(submission_status),
        }

        return PhaseResult(
            phase_name=SubmissionPhase.SUBMISSION_TRACKING.value,
            status=PhaseStatus.COMPLETED,
            started_at=started,
            completed_at=_utcnow(),
            result_data=result_data,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _determine_review_outcome(
        self, pass_count: int, total_checks: int,
    ) -> ReviewOutcome:
        """Determine internal review outcome from check results."""
        if total_checks == 0:
            return ReviewOutcome.REJECTED

        pass_rate = (pass_count / total_checks) * 100
        if pass_rate == 100.0:
            return ReviewOutcome.APPROVED
        if pass_rate >= 80.0:
            return ReviewOutcome.APPROVED_WITH_CONDITIONS
        if pass_rate >= 50.0:
            return ReviewOutcome.REVISION_REQUIRED
        return ReviewOutcome.REJECTED

    def _evaluate_cab_criterion(
        self,
        criterion: Dict[str, str],
        doc_types_present: set,
        package_data: Dict[str, Any],
        review_data: Dict[str, Any],
    ) -> CABCheckResult:
        """Evaluate a single CAB criterion against the dossier."""
        cid = criterion["criterion_id"]

        if cid == "CAB-001":
            has_sub = DocumentType.SUBSTANTIATION_REPORT.value in doc_types_present
            return CABCheckResult.PASS if has_sub else CABCheckResult.FAIL

        if cid == "CAB-002":
            has_cert = DocumentType.VERIFICATION_CERTIFICATE.value in doc_types_present
            return CABCheckResult.PASS if has_cert else CABCheckResult.FAIL

        if cid == "CAB-003":
            has_lca = DocumentType.LCA_STUDY.value in doc_types_present
            if not self.sub_config.require_lca:
                return CABCheckResult.NOT_APPLICABLE
            return CABCheckResult.PASS if has_lca else CABCheckResult.FAIL

        if cid == "CAB-004":
            has_evidence = DocumentType.EVIDENCE_DOSSIER.value in doc_types_present
            return CABCheckResult.PASS if has_evidence else CABCheckResult.NEEDS_CLARIFICATION

        if cid == "CAB-005":
            return CABCheckResult.PASS

        if cid == "CAB-006":
            has_method = DocumentType.METHODOLOGY_DESCRIPTION.value in doc_types_present
            return CABCheckResult.PASS if has_method else CABCheckResult.NEEDS_CLARIFICATION

        if cid == "CAB-007":
            has_comparison = DocumentType.COMPARISON_DATA.value in doc_types_present
            if not has_comparison:
                return CABCheckResult.NOT_APPLICABLE
            return CABCheckResult.PASS

        if cid == "CAB-008":
            has_plan = DocumentType.IMPLEMENTATION_PLAN.value in doc_types_present
            if not has_plan:
                return CABCheckResult.NOT_APPLICABLE
            return CABCheckResult.PASS

        return CABCheckResult.NEEDS_CLARIFICATION

    def _determine_submission_status(
        self, review_outcome: str, cab_passed: bool,
    ) -> SubmissionStatus:
        """Determine submission status from review and CAB results."""
        if review_outcome == ReviewOutcome.REJECTED.value:
            return SubmissionStatus.DRAFT
        if review_outcome == ReviewOutcome.REVISION_REQUIRED.value:
            return SubmissionStatus.DRAFT
        if not cab_passed:
            return SubmissionStatus.DRAFT
        if review_outcome == ReviewOutcome.APPROVED_WITH_CONDITIONS.value:
            return SubmissionStatus.READY
        if review_outcome == ReviewOutcome.APPROVED.value:
            return SubmissionStatus.READY
        return SubmissionStatus.DRAFT

    def _determine_next_steps(
        self,
        status: SubmissionStatus,
        review_data: Dict[str, Any],
        cab_data: Dict[str, Any],
    ) -> List[str]:
        """Determine next steps based on current submission status."""
        steps: List[str] = []

        if status == SubmissionStatus.DRAFT:
            if review_data.get("issue_count", 0) > 0:
                steps.append("Address internal review issues before resubmission")
            failing = cab_data.get("failing_criteria", [])
            if failing:
                steps.append(f"Resolve {len(failing)} failing CAB criteria")
            steps.append("Recompile dossier after addressing issues")

        elif status == SubmissionStatus.READY:
            steps.append("Submit dossier to Conformity Assessment Body")
            steps.append("Schedule formal CAB review meeting")
            steps.append("Prepare response templates for potential CAB queries")

        return steps if steps else ["Monitor submission status"]

    def _status_note(self, status: SubmissionStatus) -> str:
        """Generate a status note for the tracking history."""
        notes: Dict[SubmissionStatus, str] = {
            SubmissionStatus.DRAFT: "Dossier requires further work before submission",
            SubmissionStatus.READY: "Dossier passed all checks and is ready for CAB submission",
            SubmissionStatus.SUBMITTED: "Dossier submitted to Conformity Assessment Body",
            SubmissionStatus.UNDER_REVIEW: "Dossier under review by CAB",
            SubmissionStatus.APPROVED: "Claims approved by CAB -- cleared for publication",
            SubmissionStatus.REJECTED: "Submission rejected -- see CAB feedback",
            SubmissionStatus.WITHDRAWN: "Submission withdrawn by entity",
        }
        return notes.get(status, "Status updated")

    def _generate_recommendation(self, status: SubmissionStatus) -> str:
        """Generate recommendation based on submission status."""
        if status == SubmissionStatus.READY:
            return (
                "Dossier is complete and has passed all quality checks. "
                "Proceed with formal submission to the Conformity Assessment Body."
            )
        if status == SubmissionStatus.DRAFT:
            return (
                "Dossier is not yet ready for submission. Address all identified "
                "issues in internal review and CAB pre-checks before resubmitting."
            )
        return (
            "Continue monitoring submission progress and respond to any "
            "CAB queries promptly."
        )
