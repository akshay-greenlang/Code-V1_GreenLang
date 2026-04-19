# -*- coding: utf-8 -*-
"""
Audit & Certification Workflow - Internal Audit + Certification Readiness
===================================

4-phase workflow for ISO 50001 internal auditing and certification
readiness assessment within PACK-034 ISO 50001 Energy Management
System Pack.

Phases:
    1. GapAnalysis           -- Run ISO 50001 Clauses 4-10 gap analysis
    2. InternalAudit         -- Conduct internal audit simulation, identify NCs/OFIs
    3. CorrectiveActions     -- Generate corrective action plans for nonconformities
    4. CertificationReadiness -- Assess Stage 1/Stage 2 readiness

The workflow follows GreenLang zero-hallucination principles: clause
assessments use deterministic scoring rubrics against documented evidence,
NC classification follows ISO 19011 guidelines, and readiness scores
are calculated from weighted compliance matrices. SHA-256 provenance
hashes guarantee auditability.

Schedule: semi-annual / pre-certification
Estimated duration: 60 minutes

Author: GreenLang Team
Version: 34.0.0
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from datetime import datetime
from decimal import Decimal
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


class AuditPhase(str, Enum):
    """Phases of the audit certification workflow."""

    GAP_ANALYSIS = "gap_analysis"
    INTERNAL_AUDIT = "internal_audit"
    CORRECTIVE_ACTIONS = "corrective_actions"
    CERTIFICATION_READINESS = "certification_readiness"


class ComplianceLevel(str, Enum):
    """Compliance level for a clause."""

    FULLY_COMPLIANT = "fully_compliant"
    SUBSTANTIALLY_COMPLIANT = "substantially_compliant"
    PARTIALLY_COMPLIANT = "partially_compliant"
    NON_COMPLIANT = "non_compliant"
    NOT_ASSESSED = "not_assessed"


class FindingType(str, Enum):
    """Type of audit finding."""

    MAJOR_NC = "major_nonconformity"
    MINOR_NC = "minor_nonconformity"
    OFI = "opportunity_for_improvement"
    POSITIVE = "positive_practice"


class CertificationStage(str, Enum):
    """ISO certification stage."""

    STAGE_1 = "stage_1"  # Documentation review
    STAGE_2 = "stage_2"  # Implementation audit


# =============================================================================
# ISO 50001:2018 CLAUSE STRUCTURE (Zero-Hallucination Reference)
# =============================================================================

ISO_50001_CLAUSES: Dict[str, Dict[str, Any]] = {
    "4": {
        "title": "Context of the organization",
        "weight": 0.10,
        "subclauses": {
            "4.1": "Understanding the organization and its context",
            "4.2": "Understanding the needs and expectations of interested parties",
            "4.3": "Determining the scope of the EnMS",
            "4.4": "Energy management system",
        },
    },
    "5": {
        "title": "Leadership",
        "weight": 0.15,
        "subclauses": {
            "5.1": "Leadership and commitment",
            "5.2": "Energy policy",
            "5.3": "Organizational roles, responsibilities and authorities",
        },
    },
    "6": {
        "title": "Planning",
        "weight": 0.20,
        "subclauses": {
            "6.1": "Actions to address risks and opportunities",
            "6.2": "Objectives, energy targets, and planning to achieve them",
            "6.3": "Energy review",
            "6.4": "Energy performance indicators",
            "6.5": "Energy baseline",
            "6.6": "Planning for collection of energy data",
        },
    },
    "7": {
        "title": "Support",
        "weight": 0.10,
        "subclauses": {
            "7.1": "Resources",
            "7.2": "Competence",
            "7.3": "Awareness",
            "7.4": "Communication",
            "7.5": "Documented information",
        },
    },
    "8": {
        "title": "Operation",
        "weight": 0.20,
        "subclauses": {
            "8.1": "Operational planning and control",
            "8.2": "Design",
            "8.3": "Procurement",
        },
    },
    "9": {
        "title": "Performance evaluation",
        "weight": 0.15,
        "subclauses": {
            "9.1": "Monitoring, measurement, analysis and evaluation of EnP",
            "9.2": "Internal audit",
            "9.3": "Management review",
        },
    },
    "10": {
        "title": "Improvement",
        "weight": 0.10,
        "subclauses": {
            "10.1": "Nonconformity and corrective action",
            "10.2": "Continual improvement",
        },
    },
}

# Compliance scoring rubric
COMPLIANCE_SCORES: Dict[str, int] = {
    "fully_compliant": 100,
    "substantially_compliant": 75,
    "partially_compliant": 50,
    "non_compliant": 0,
    "not_assessed": 0,
}

# Certification readiness thresholds
STAGE_1_THRESHOLD: float = 70.0  # Documentation readiness
STAGE_2_THRESHOLD: float = 80.0  # Implementation readiness
CERTIFICATION_THRESHOLD: float = 85.0  # Overall certification readiness


# =============================================================================
# DATA MODELS
# =============================================================================


class PhaseResult(BaseModel):
    """Result from a single workflow phase."""

    phase_name: str = Field(..., description="Phase identifier")
    phase_number: int = Field(default=0, description="Phase sequence number")
    status: PhaseStatus = Field(..., description="Phase completion status")
    duration_ms: float = Field(default=0.0, description="Phase duration in milliseconds")
    outputs: Dict[str, Any] = Field(default_factory=dict, description="Phase output data")
    warnings: List[str] = Field(default_factory=list, description="Warnings raised")
    errors: List[str] = Field(default_factory=list, description="Errors encountered")
    provenance_hash: str = Field(default="", description="SHA-256 of phase output")


class ClauseAssessment(BaseModel):
    """Assessment result for a single ISO 50001 clause."""

    clause_number: str = Field(default="", description="Clause number (e.g., '6.3')")
    clause_title: str = Field(default="", description="Clause title")
    compliance_level: ComplianceLevel = Field(default=ComplianceLevel.NOT_ASSESSED)
    score: int = Field(default=0, ge=0, le=100, description="Compliance score 0-100")
    evidence_provided: List[str] = Field(default_factory=list, description="Evidence documents")
    gaps_identified: List[str] = Field(default_factory=list, description="Identified gaps")
    notes: str = Field(default="")


class AuditFinding(BaseModel):
    """An individual audit finding (NC or OFI)."""

    finding_id: str = Field(default_factory=lambda: f"find-{uuid.uuid4().hex[:8]}")
    clause_number: str = Field(default="", description="Related clause")
    finding_type: FindingType = Field(default=FindingType.OFI)
    title: str = Field(default="", description="Finding title")
    description: str = Field(default="", description="Detailed description")
    evidence: str = Field(default="", description="Objective evidence")
    risk_level: str = Field(default="medium", description="high|medium|low")


class CorrectiveAction(BaseModel):
    """Corrective action plan for a nonconformity."""

    action_id: str = Field(default_factory=lambda: f"ca-{uuid.uuid4().hex[:8]}")
    finding_id: str = Field(default="", description="Related finding ID")
    clause_number: str = Field(default="", description="Related clause")
    description: str = Field(default="", description="Action description")
    root_cause: str = Field(default="", description="Identified root cause")
    responsible_party: str = Field(default="", description="Assigned responsible party")
    target_date: str = Field(default="", description="Target completion date")
    priority: str = Field(default="medium", description="high|medium|low")
    verification_method: str = Field(default="", description="How completion will be verified")


class CertificationReadiness(BaseModel):
    """Overall certification readiness assessment."""

    stage_1_score: Decimal = Field(default=Decimal("0"), ge=0, le=100)
    stage_1_ready: bool = Field(default=False)
    stage_2_score: Decimal = Field(default=Decimal("0"), ge=0, le=100)
    stage_2_ready: bool = Field(default=False)
    overall_score: Decimal = Field(default=Decimal("0"), ge=0, le=100)
    certification_ready: bool = Field(default=False)
    blocking_issues: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)


class DocumentChecklistItem(BaseModel):
    """A document checklist item for certification."""

    document_name: str = Field(default="", description="Document name")
    clause_reference: str = Field(default="", description="Related clause(s)")
    status: str = Field(default="missing", description="present|partial|missing")
    notes: str = Field(default="")


class AuditCertificationInput(BaseModel):
    """Input data model for AuditCertificationWorkflow."""

    enms_id: str = Field(default="", description="EnMS program identifier")
    evidence: Dict[str, Any] = Field(
        default_factory=dict,
        description="Evidence by clause: {clause_number: {documents: [], records: []}}",
    )
    assessment_type: str = Field(
        default="full",
        description="full|stage_1|stage_2|surveillance",
    )
    assessor: str = Field(default="", description="Assessor name/ID")
    previous_findings: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Previous audit findings for tracking",
    )
    entity_id: str = Field(default="")
    tenant_id: str = Field(default="")


class AuditCertificationResult(BaseModel):
    """Complete result from audit certification workflow."""

    audit_id: str = Field(..., description="Unique audit ID")
    enms_id: str = Field(default="", description="EnMS program identifier")
    assessment_type: str = Field(default="full")
    assessor: str = Field(default="")
    compliance_scores: Dict[str, int] = Field(
        default_factory=dict,
        description="Compliance score by main clause",
    )
    clause_assessments: List[ClauseAssessment] = Field(default_factory=list)
    nonconformities: List[AuditFinding] = Field(default_factory=list)
    corrective_actions: List[CorrectiveAction] = Field(default_factory=list)
    certification_readiness: CertificationReadiness = Field(
        default_factory=CertificationReadiness,
    )
    document_checklist: List[DocumentChecklistItem] = Field(default_factory=list)
    improvement_plan: List[str] = Field(default_factory=list)
    total_findings: int = Field(default=0, ge=0)
    major_ncs: int = Field(default=0, ge=0)
    minor_ncs: int = Field(default=0, ge=0)
    ofis: int = Field(default=0, ge=0)
    positive_practices: int = Field(default=0, ge=0)
    phases_completed: List[str] = Field(default_factory=list)
    execution_time_ms: float = Field(default=0.0)
    calculated_at: str = Field(default="", description="ISO 8601 timestamp")
    provenance_hash: str = Field(default="", description="SHA-256 of complete result")


# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================


class AuditCertificationWorkflow:
    """
    4-phase audit and certification readiness workflow per ISO 50001.

    Performs clause-by-clause gap analysis, internal audit simulation
    with NC/OFI identification, corrective action plan generation,
    and certification readiness assessment for Stage 1 and Stage 2.

    Zero-hallucination: compliance scoring uses deterministic rubrics,
    NC classification follows ISO 19011 rules, and readiness scores
    use weighted compliance matrices. No LLM calls in the numeric
    computation path.

    Attributes:
        audit_id: Unique audit execution identifier.
        _assessments: Clause assessment results.
        _findings: Audit findings (NCs and OFIs).
        _corrective_actions: Corrective action plans.
        _document_checklist: Document checklist items.
        _phase_results: Ordered phase outputs.

    Example:
        >>> wf = AuditCertificationWorkflow()
        >>> inp = AuditCertificationInput(
        ...     enms_id="enms-001",
        ...     evidence={"6.3": {"documents": ["energy_review_report.pdf"]}},
        ...     assessor="internal_auditor",
        ... )
        >>> result = wf.execute(inp)
        >>> assert result.total_findings >= 0
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize AuditCertificationWorkflow."""
        self.audit_id: str = str(uuid.uuid4())
        self.config: Dict[str, Any] = config or {}
        self._assessments: List[ClauseAssessment] = []
        self._findings: List[AuditFinding] = []
        self._corrective_actions: List[CorrectiveAction] = []
        self._document_checklist: List[DocumentChecklistItem] = []
        self._phase_results: List[PhaseResult] = []
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def execute(self, input_data: AuditCertificationInput) -> AuditCertificationResult:
        """
        Execute the 4-phase audit certification workflow.

        Args:
            input_data: Validated audit certification input.

        Returns:
            AuditCertificationResult with findings, CAs, and readiness.
        """
        t_start = time.perf_counter()
        started_at = datetime.utcnow()
        self.logger.info(
            "Starting audit certification workflow %s enms=%s type=%s assessor=%s",
            self.audit_id, input_data.enms_id,
            input_data.assessment_type, input_data.assessor,
        )

        self._phase_results = []
        self._assessments = []
        self._findings = []
        self._corrective_actions = []
        self._document_checklist = []

        try:
            phase1 = self._phase_gap_analysis(input_data)
            self._phase_results.append(phase1)

            phase2 = self._phase_internal_audit(input_data)
            self._phase_results.append(phase2)

            phase3 = self._phase_corrective_actions(input_data)
            self._phase_results.append(phase3)

            phase4 = self._phase_certification_readiness(input_data)
            self._phase_results.append(phase4)

        except Exception as exc:
            self.logger.error(
                "Audit certification workflow failed: %s", exc, exc_info=True,
            )
            self._phase_results.append(PhaseResult(
                phase_name="error", phase_number=0,
                status=PhaseStatus.FAILED, errors=[str(exc)],
            ))

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0

        # Compile scores
        compliance_scores: Dict[str, int] = {}
        for assessment in self._assessments:
            main_clause = assessment.clause_number.split(".")[0]
            if main_clause not in compliance_scores:
                compliance_scores[main_clause] = assessment.score
            else:
                # Average for sub-clauses
                same_clause = [
                    a.score for a in self._assessments
                    if a.clause_number.startswith(main_clause)
                ]
                compliance_scores[main_clause] = round(sum(same_clause) / max(len(same_clause), 1))

        # Count findings
        major_ncs = sum(1 for f in self._findings if f.finding_type == FindingType.MAJOR_NC)
        minor_ncs = sum(1 for f in self._findings if f.finding_type == FindingType.MINOR_NC)
        ofis = sum(1 for f in self._findings if f.finding_type == FindingType.OFI)
        positive = sum(1 for f in self._findings if f.finding_type == FindingType.POSITIVE)

        # Build certification readiness
        readiness = self._build_certification_readiness(compliance_scores, major_ncs, minor_ncs)

        # Build improvement plan
        improvement_plan = self._build_improvement_plan()

        completed_phases = [
            p.phase_name for p in self._phase_results
            if p.status == PhaseStatus.COMPLETED
        ]

        result = AuditCertificationResult(
            audit_id=self.audit_id,
            enms_id=input_data.enms_id,
            assessment_type=input_data.assessment_type,
            assessor=input_data.assessor,
            compliance_scores=compliance_scores,
            clause_assessments=self._assessments,
            nonconformities=[f for f in self._findings if f.finding_type in (FindingType.MAJOR_NC, FindingType.MINOR_NC)],
            corrective_actions=self._corrective_actions,
            certification_readiness=readiness,
            document_checklist=self._document_checklist,
            improvement_plan=improvement_plan,
            total_findings=len(self._findings),
            major_ncs=major_ncs,
            minor_ncs=minor_ncs,
            ofis=ofis,
            positive_practices=positive,
            phases_completed=completed_phases,
            execution_time_ms=round(elapsed_ms, 2),
            calculated_at=started_at.isoformat() + "Z",
        )
        result.provenance_hash = self._compute_provenance(result)

        self.logger.info(
            "Audit certification workflow %s completed in %.0fms "
            "findings=%d major=%d minor=%d OFIs=%d ready=%s",
            self.audit_id, elapsed_ms, len(self._findings),
            major_ncs, minor_ncs, ofis, readiness.certification_ready,
        )
        return result

    # -------------------------------------------------------------------------
    # Phase 1: Gap Analysis
    # -------------------------------------------------------------------------

    def _phase_gap_analysis(
        self, input_data: AuditCertificationInput
    ) -> PhaseResult:
        """Run ISO 50001 Clauses 4-10 gap analysis, score each clause."""
        t_start = time.perf_counter()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        for main_clause, clause_info in ISO_50001_CLAUSES.items():
            for sub_clause, sub_title in clause_info["subclauses"].items():
                # Check evidence for this subclause
                evidence_data = input_data.evidence.get(sub_clause, {})
                documents = evidence_data.get("documents", [])
                records = evidence_data.get("records", [])

                # Score based on evidence completeness
                score, compliance_level, gaps = self._score_clause(
                    sub_clause, sub_title, documents, records,
                )

                assessment = ClauseAssessment(
                    clause_number=sub_clause,
                    clause_title=sub_title,
                    compliance_level=compliance_level,
                    score=score,
                    evidence_provided=documents + records,
                    gaps_identified=gaps,
                )
                self._assessments.append(assessment)

                # Add to document checklist
                doc_status = "present" if documents else ("partial" if records else "missing")
                self._document_checklist.append(DocumentChecklistItem(
                    document_name=f"{sub_clause} - {sub_title}",
                    clause_reference=sub_clause,
                    status=doc_status,
                ))

        # Calculate overall scores
        total_score = 0.0
        total_weight = 0.0
        for main_clause, clause_info in ISO_50001_CLAUSES.items():
            clause_scores = [
                a.score for a in self._assessments
                if a.clause_number.startswith(main_clause + ".")
            ]
            if clause_scores:
                avg_score = sum(clause_scores) / len(clause_scores)
                total_score += avg_score * clause_info["weight"]
                total_weight += clause_info["weight"]

        overall = total_score / max(total_weight, 0.01)

        outputs["clauses_assessed"] = len(self._assessments)
        outputs["overall_score"] = round(overall, 1)
        outputs["fully_compliant"] = sum(
            1 for a in self._assessments
            if a.compliance_level == ComplianceLevel.FULLY_COMPLIANT
        )
        outputs["non_compliant"] = sum(
            1 for a in self._assessments
            if a.compliance_level == ComplianceLevel.NON_COMPLIANT
        )
        outputs["total_gaps"] = sum(len(a.gaps_identified) for a in self._assessments)

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        self.logger.info(
            "Phase 1 GapAnalysis: %d clauses, overall=%.1f, gaps=%d",
            len(self._assessments), overall, outputs["total_gaps"],
        )
        return PhaseResult(
            phase_name=AuditPhase.GAP_ANALYSIS.value, phase_number=1,
            status=PhaseStatus.COMPLETED, duration_ms=elapsed_ms,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    def _score_clause(
        self,
        clause_number: str,
        clause_title: str,
        documents: List[str],
        records: List[str],
    ) -> tuple:
        """Score a clause based on evidence provided."""
        gaps: List[str] = []
        total_evidence = len(documents) + len(records)

        if total_evidence >= 3:
            return 100, ComplianceLevel.FULLY_COMPLIANT, gaps
        elif total_evidence == 2:
            return 75, ComplianceLevel.SUBSTANTIALLY_COMPLIANT, [
                f"{clause_number}: Additional evidence recommended for full compliance"
            ]
        elif total_evidence == 1:
            gaps.append(f"{clause_number}: Insufficient documented evidence for '{clause_title}'")
            return 50, ComplianceLevel.PARTIALLY_COMPLIANT, gaps
        else:
            gaps.append(f"{clause_number}: No evidence provided for '{clause_title}'")
            return 0, ComplianceLevel.NON_COMPLIANT, gaps

    # -------------------------------------------------------------------------
    # Phase 2: Internal Audit
    # -------------------------------------------------------------------------

    def _phase_internal_audit(
        self, input_data: AuditCertificationInput
    ) -> PhaseResult:
        """Conduct internal audit simulation, identify NCs and OFIs."""
        t_start = time.perf_counter()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        for assessment in self._assessments:
            if assessment.compliance_level == ComplianceLevel.NON_COMPLIANT:
                # Major NC: no evidence at all
                finding = AuditFinding(
                    clause_number=assessment.clause_number,
                    finding_type=FindingType.MAJOR_NC,
                    title=f"Major NC: {assessment.clause_title}",
                    description=(
                        f"No objective evidence of implementation for "
                        f"Clause {assessment.clause_number}: {assessment.clause_title}. "
                        f"This is a systematic failure to meet the requirement."
                    ),
                    evidence="No evidence provided",
                    risk_level="high",
                )
                self._findings.append(finding)

            elif assessment.compliance_level == ComplianceLevel.PARTIALLY_COMPLIANT:
                # Minor NC: partial evidence
                finding = AuditFinding(
                    clause_number=assessment.clause_number,
                    finding_type=FindingType.MINOR_NC,
                    title=f"Minor NC: {assessment.clause_title}",
                    description=(
                        f"Partial evidence of implementation for "
                        f"Clause {assessment.clause_number}: {assessment.clause_title}. "
                        f"Gaps identified: {'; '.join(assessment.gaps_identified)}"
                    ),
                    evidence=", ".join(assessment.evidence_provided) or "Limited evidence",
                    risk_level="medium",
                )
                self._findings.append(finding)

            elif assessment.compliance_level == ComplianceLevel.SUBSTANTIALLY_COMPLIANT:
                # OFI: room for improvement
                finding = AuditFinding(
                    clause_number=assessment.clause_number,
                    finding_type=FindingType.OFI,
                    title=f"OFI: {assessment.clause_title}",
                    description=(
                        f"Substantially compliant with "
                        f"Clause {assessment.clause_number}: {assessment.clause_title}. "
                        f"Additional documentation would strengthen compliance."
                    ),
                    evidence=", ".join(assessment.evidence_provided),
                    risk_level="low",
                )
                self._findings.append(finding)

            elif assessment.compliance_level == ComplianceLevel.FULLY_COMPLIANT:
                # Positive practice
                finding = AuditFinding(
                    clause_number=assessment.clause_number,
                    finding_type=FindingType.POSITIVE,
                    title=f"Positive: {assessment.clause_title}",
                    description=(
                        f"Full compliance with "
                        f"Clause {assessment.clause_number}: {assessment.clause_title}."
                    ),
                    evidence=", ".join(assessment.evidence_provided),
                    risk_level="low",
                )
                self._findings.append(finding)

        # Check previous findings for closure
        for prev in input_data.previous_findings:
            prev_id = prev.get("finding_id", "")
            prev_status = prev.get("status", "open")
            if prev_status == "open":
                warnings.append(f"Previous finding {prev_id} still open")

        outputs["total_findings"] = len(self._findings)
        outputs["major_ncs"] = sum(1 for f in self._findings if f.finding_type == FindingType.MAJOR_NC)
        outputs["minor_ncs"] = sum(1 for f in self._findings if f.finding_type == FindingType.MINOR_NC)
        outputs["ofis"] = sum(1 for f in self._findings if f.finding_type == FindingType.OFI)
        outputs["positive_practices"] = sum(1 for f in self._findings if f.finding_type == FindingType.POSITIVE)
        outputs["open_previous_findings"] = sum(
            1 for p in input_data.previous_findings if p.get("status") == "open"
        )

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        self.logger.info(
            "Phase 2 InternalAudit: %d findings (Major=%d, Minor=%d, OFI=%d, Positive=%d)",
            len(self._findings), outputs["major_ncs"], outputs["minor_ncs"],
            outputs["ofis"], outputs["positive_practices"],
        )
        return PhaseResult(
            phase_name=AuditPhase.INTERNAL_AUDIT.value, phase_number=2,
            status=PhaseStatus.COMPLETED, duration_ms=elapsed_ms,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 3: Corrective Actions
    # -------------------------------------------------------------------------

    def _phase_corrective_actions(
        self, input_data: AuditCertificationInput
    ) -> PhaseResult:
        """Generate corrective action plans for nonconformities."""
        t_start = time.perf_counter()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        nc_findings = [
            f for f in self._findings
            if f.finding_type in (FindingType.MAJOR_NC, FindingType.MINOR_NC)
        ]

        for finding in nc_findings:
            # Determine priority and timeline based on NC type
            if finding.finding_type == FindingType.MAJOR_NC:
                priority = "high"
                target_days = 30
                root_cause = (
                    f"Systematic gap in EnMS implementation for "
                    f"Clause {finding.clause_number}. Likely causes: "
                    f"insufficient resources, lack of awareness, or "
                    f"incomplete implementation planning."
                )
            else:
                priority = "medium"
                target_days = 60
                root_cause = (
                    f"Partial implementation gap for "
                    f"Clause {finding.clause_number}. Likely causes: "
                    f"documentation gaps or incomplete records."
                )

            ca = CorrectiveAction(
                finding_id=finding.finding_id,
                clause_number=finding.clause_number,
                description=(
                    f"Address {finding.finding_type.value} for "
                    f"Clause {finding.clause_number}: {finding.title}. "
                    f"Implement required evidence and processes."
                ),
                root_cause=root_cause,
                responsible_party=input_data.assessor or "energy_manager",
                target_date=f"+{target_days} days",
                priority=priority,
                verification_method=(
                    f"Review updated documentation and records for "
                    f"Clause {finding.clause_number}. Verify implementation "
                    f"through interviews and observation."
                ),
            )
            self._corrective_actions.append(ca)

        outputs["corrective_actions_created"] = len(self._corrective_actions)
        outputs["high_priority"] = sum(1 for ca in self._corrective_actions if ca.priority == "high")
        outputs["medium_priority"] = sum(1 for ca in self._corrective_actions if ca.priority == "medium")

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        self.logger.info(
            "Phase 3 CorrectiveActions: %d actions (high=%d, medium=%d)",
            len(self._corrective_actions),
            outputs["high_priority"], outputs["medium_priority"],
        )
        return PhaseResult(
            phase_name=AuditPhase.CORRECTIVE_ACTIONS.value, phase_number=3,
            status=PhaseStatus.COMPLETED, duration_ms=elapsed_ms,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 4: Certification Readiness
    # -------------------------------------------------------------------------

    def _phase_certification_readiness(
        self, input_data: AuditCertificationInput
    ) -> PhaseResult:
        """Assess overall certification readiness (Stage 1/Stage 2)."""
        t_start = time.perf_counter()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        # Stage 1: Documentation readiness (Clauses 4, 5, 7.5, 6)
        doc_clauses = ["4", "5", "6", "7"]
        doc_scores = []
        for a in self._assessments:
            main = a.clause_number.split(".")[0]
            if main in doc_clauses:
                doc_scores.append(a.score)

        stage_1_score = sum(doc_scores) / max(len(doc_scores), 1)
        stage_1_ready = stage_1_score >= STAGE_1_THRESHOLD

        # Stage 2: Implementation readiness (all clauses)
        all_scores = [a.score for a in self._assessments]
        stage_2_score = sum(all_scores) / max(len(all_scores), 1)
        stage_2_ready = stage_2_score >= STAGE_2_THRESHOLD

        # Overall: weighted average considering NCs
        major_nc_count = sum(1 for f in self._findings if f.finding_type == FindingType.MAJOR_NC)
        minor_nc_count = sum(1 for f in self._findings if f.finding_type == FindingType.MINOR_NC)

        # Major NCs block certification
        nc_penalty = major_nc_count * 10.0 + minor_nc_count * 3.0
        overall_score = max(0.0, stage_2_score - nc_penalty)
        certification_ready = (
            overall_score >= CERTIFICATION_THRESHOLD
            and major_nc_count == 0
        )

        # Blocking issues
        blocking: List[str] = []
        if major_nc_count > 0:
            blocking.append(f"{major_nc_count} major nonconformit{'y' if major_nc_count == 1 else 'ies'} must be resolved")
        if not stage_1_ready:
            blocking.append("Stage 1 documentation readiness not achieved")

        # Recommendations
        recommendations: List[str] = []
        if not certification_ready:
            recommendations.append("Resolve all major nonconformities before scheduling certification audit")
        if minor_nc_count > 0:
            recommendations.append(f"Address {minor_nc_count} minor nonconformities to strengthen compliance")

        # Check clause-specific gaps
        for a in self._assessments:
            if a.compliance_level == ComplianceLevel.NON_COMPLIANT:
                recommendations.append(
                    f"Priority: Implement Clause {a.clause_number} ({a.clause_title})"
                )

        if certification_ready:
            recommendations.append(
                "Organization is ready for certification audit. "
                "Schedule Stage 1 and Stage 2 with certification body."
            )

        outputs["stage_1_score"] = round(stage_1_score, 1)
        outputs["stage_1_ready"] = stage_1_ready
        outputs["stage_2_score"] = round(stage_2_score, 1)
        outputs["stage_2_ready"] = stage_2_ready
        outputs["overall_score"] = round(overall_score, 1)
        outputs["certification_ready"] = certification_ready
        outputs["blocking_issues"] = len(blocking)
        outputs["recommendations_count"] = len(recommendations)

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        self.logger.info(
            "Phase 4 CertificationReadiness: S1=%.1f(%s) S2=%.1f(%s) "
            "overall=%.1f ready=%s blocking=%d",
            stage_1_score, stage_1_ready, stage_2_score, stage_2_ready,
            overall_score, certification_ready, len(blocking),
        )

        # Store readiness for result building
        self._certification_readiness = CertificationReadiness(
            stage_1_score=Decimal(str(round(stage_1_score, 1))),
            stage_1_ready=stage_1_ready,
            stage_2_score=Decimal(str(round(stage_2_score, 1))),
            stage_2_ready=stage_2_ready,
            overall_score=Decimal(str(round(overall_score, 1))),
            certification_ready=certification_ready,
            blocking_issues=blocking,
            recommendations=recommendations,
        )

        return PhaseResult(
            phase_name=AuditPhase.CERTIFICATION_READINESS.value, phase_number=4,
            status=PhaseStatus.COMPLETED, duration_ms=elapsed_ms,
            outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Supporting Methods
    # -------------------------------------------------------------------------

    def _build_certification_readiness(
        self,
        compliance_scores: Dict[str, int],
        major_ncs: int,
        minor_ncs: int,
    ) -> CertificationReadiness:
        """Build certification readiness from phase 4 results."""
        if hasattr(self, '_certification_readiness'):
            return self._certification_readiness
        return CertificationReadiness()

    def _build_improvement_plan(self) -> List[str]:
        """Build prioritized improvement plan from findings."""
        plan: List[str] = []

        # Priority 1: Major NCs
        for f in self._findings:
            if f.finding_type == FindingType.MAJOR_NC:
                plan.append(
                    f"[CRITICAL] Resolve major NC for Clause {f.clause_number}: "
                    f"{f.title}"
                )

        # Priority 2: Minor NCs
        for f in self._findings:
            if f.finding_type == FindingType.MINOR_NC:
                plan.append(
                    f"[HIGH] Address minor NC for Clause {f.clause_number}: "
                    f"{f.title}"
                )

        # Priority 3: OFIs
        for f in self._findings:
            if f.finding_type == FindingType.OFI:
                plan.append(
                    f"[MEDIUM] Consider OFI for Clause {f.clause_number}: "
                    f"{f.title}"
                )

        return plan

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _compute_provenance(self, result: AuditCertificationResult) -> str:
        """Compute SHA-256 provenance hash for the complete result."""
        payload = result.model_dump_json(exclude={"provenance_hash"})
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _hash_dict(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 of a dict."""
        raw = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()
