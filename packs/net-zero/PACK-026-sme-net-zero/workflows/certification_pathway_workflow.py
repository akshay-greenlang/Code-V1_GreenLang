# -*- coding: utf-8 -*-
"""
Certification Pathway Workflow
===================================

6-phase workflow for guiding SMEs through sustainability certification
within PACK-026 SME Net Zero Pack.  Supports SME Climate Hub, B Corp,
ISO 14001, and Carbon Trust Standard certifications.

Phases:
    1. PathwaySelection       -- Select certification (SME Climate Hub / B Corp / ISO 14001 / Carbon Trust)
    2. ReadinessAssessment    -- Assess current readiness against criteria
    3. GapClosure             -- Address gaps (data quality, governance, targets)
    4. Documentation          -- Prepare submission documents
    5. Submission             -- Submit to certification body
    6. VerificationTracking   -- Await and track verification result

Uses: certification_readiness_engine, simplified_target_engine.

Zero-hallucination: all certification criteria from official frameworks.
SHA-256 provenance hashes for auditability.

Author: GreenLang Team
Version: 26.0.0
Pack: PACK-026 SME Net Zero Pack
"""

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

_MODULE_VERSION = "26.0.0"
_PACK_ID = "PACK-026"

# =============================================================================
# HELPERS
# =============================================================================

def _new_uuid() -> str:
    return uuid.uuid4().hex

def _compute_hash(data: str) -> str:
    return hashlib.sha256(data.encode("utf-8")).hexdigest()

# =============================================================================
# ENUMS
# =============================================================================

class PhaseStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"

class WorkflowStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"

class CertificationType(str, Enum):
    SME_CLIMATE_HUB = "sme_climate_hub"
    B_CORP = "b_corp"
    ISO_14001 = "iso_14001"
    CARBON_TRUST = "carbon_trust"

class ReadinessLevel(str, Enum):
    READY = "ready"
    NEARLY_READY = "nearly_ready"
    SIGNIFICANT_GAPS = "significant_gaps"
    NOT_READY = "not_ready"

class GapSeverity(str, Enum):
    CRITICAL = "critical"
    MAJOR = "major"
    MINOR = "minor"
    INFORMATIONAL = "informational"

class DocumentStatus(str, Enum):
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    READY = "ready"
    SUBMITTED = "submitted"
    APPROVED = "approved"
    REJECTED = "rejected"

class CertificationStatus(str, Enum):
    NOT_STARTED = "not_started"
    PREPARING = "preparing"
    SUBMITTED = "submitted"
    UNDER_REVIEW = "under_review"
    VERIFIED = "verified"
    CERTIFIED = "certified"
    REJECTED = "rejected"
    RENEWAL_DUE = "renewal_due"

# =============================================================================
# CERTIFICATION CRITERIA DATABASE
# =============================================================================

CERTIFICATION_CRITERIA: Dict[str, Dict[str, Any]] = {
    "sme_climate_hub": {
        "name": "SME Climate Hub",
        "provider": "We Mean Business Coalition / ICC",
        "description": "Free commitment platform for SMEs to take climate action",
        "cost_gbp": 0,
        "difficulty": "easy",
        "timeline_months": 1,
        "renewal_years": 1,
        "url": "https://smeclimatehub.org",
        "requirements": [
            {
                "id": "sch_01",
                "category": "commitment",
                "title": "Public commitment to halve emissions by 2030",
                "description": "Sign the SME Climate Commitment pledge to halve GHG emissions before 2030 and reach net zero before 2050",
                "weight": 1.0,
                "auto_checkable": True,
            },
            {
                "id": "sch_02",
                "category": "measurement",
                "title": "Measure GHG emissions",
                "description": "Measure Scope 1 and 2 emissions (Scope 3 encouraged but not required)",
                "weight": 1.0,
                "auto_checkable": True,
            },
            {
                "id": "sch_03",
                "category": "target",
                "title": "Set a net-zero target",
                "description": "Commit to net zero by 2050 at the latest",
                "weight": 1.0,
                "auto_checkable": True,
            },
            {
                "id": "sch_04",
                "category": "reporting",
                "title": "Report progress annually",
                "description": "Submit annual progress update through the Climate Hub platform",
                "weight": 0.5,
                "auto_checkable": False,
            },
        ],
    },
    "b_corp": {
        "name": "B Corp Certification",
        "provider": "B Lab",
        "description": "Comprehensive sustainability certification for businesses",
        "cost_gbp": 1500,
        "difficulty": "hard",
        "timeline_months": 12,
        "renewal_years": 3,
        "url": "https://bcorporation.net",
        "requirements": [
            {
                "id": "bc_01",
                "category": "governance",
                "title": "Legal structure alignment",
                "description": "Amend governing documents to consider stakeholder impact",
                "weight": 1.0,
                "auto_checkable": False,
            },
            {
                "id": "bc_02",
                "category": "environment",
                "title": "B Impact Assessment: Environment score >= 20",
                "description": "Score at least 20 points in the environmental category of the B Impact Assessment",
                "weight": 1.5,
                "auto_checkable": False,
            },
            {
                "id": "bc_03",
                "category": "environment",
                "title": "GHG emissions measurement",
                "description": "Measure and disclose Scope 1 and 2 GHG emissions",
                "weight": 1.0,
                "auto_checkable": True,
            },
            {
                "id": "bc_04",
                "category": "environment",
                "title": "Emission reduction targets",
                "description": "Set science-based or equivalent emission reduction targets",
                "weight": 1.0,
                "auto_checkable": True,
            },
            {
                "id": "bc_05",
                "category": "total",
                "title": "Total B Impact Score >= 80",
                "description": "Achieve an overall B Impact Assessment score of at least 80 out of 200",
                "weight": 2.0,
                "auto_checkable": False,
            },
            {
                "id": "bc_06",
                "category": "workers",
                "title": "Worker wellbeing and engagement",
                "description": "Demonstrate fair wages, benefits, and employee engagement practices",
                "weight": 1.0,
                "auto_checkable": False,
            },
        ],
    },
    "iso_14001": {
        "name": "ISO 14001 Environmental Management System",
        "provider": "ISO (via accredited certification bodies)",
        "description": "International standard for environmental management systems",
        "cost_gbp": 5000,
        "difficulty": "hard",
        "timeline_months": 12,
        "renewal_years": 3,
        "url": "https://www.iso.org/iso-14001-environmental-management.html",
        "requirements": [
            {
                "id": "iso_01",
                "category": "policy",
                "title": "Environmental policy",
                "description": "Documented environmental policy approved by top management",
                "weight": 1.0,
                "auto_checkable": False,
            },
            {
                "id": "iso_02",
                "category": "planning",
                "title": "Environmental aspects and impacts",
                "description": "Identify and evaluate significant environmental aspects",
                "weight": 1.0,
                "auto_checkable": False,
            },
            {
                "id": "iso_03",
                "category": "planning",
                "title": "Legal and compliance register",
                "description": "Maintain register of applicable environmental legislation",
                "weight": 1.0,
                "auto_checkable": False,
            },
            {
                "id": "iso_04",
                "category": "implementation",
                "title": "Documented EMS procedures",
                "description": "Operational control procedures for significant aspects",
                "weight": 1.5,
                "auto_checkable": False,
            },
            {
                "id": "iso_05",
                "category": "measurement",
                "title": "Monitoring and measurement",
                "description": "Regular monitoring of key environmental performance indicators",
                "weight": 1.0,
                "auto_checkable": True,
            },
            {
                "id": "iso_06",
                "category": "review",
                "title": "Management review",
                "description": "Regular management review of EMS performance",
                "weight": 1.0,
                "auto_checkable": False,
            },
            {
                "id": "iso_07",
                "category": "improvement",
                "title": "Continual improvement",
                "description": "Evidence of continual improvement in environmental performance",
                "weight": 1.0,
                "auto_checkable": True,
            },
        ],
    },
    "carbon_trust": {
        "name": "Carbon Trust Standard",
        "provider": "Carbon Trust",
        "description": "Certification for organisations reducing carbon emissions year-on-year",
        "cost_gbp": 3000,
        "difficulty": "medium",
        "timeline_months": 6,
        "renewal_years": 2,
        "url": "https://www.carbontrust.com/what-we-do/assurance-and-labelling/the-carbon-trust-standard",
        "requirements": [
            {
                "id": "ct_01",
                "category": "measurement",
                "title": "Carbon footprint measurement",
                "description": "Measure organisational carbon footprint (Scope 1 + 2 minimum)",
                "weight": 1.5,
                "auto_checkable": True,
            },
            {
                "id": "ct_02",
                "category": "reduction",
                "title": "Year-on-year absolute reduction",
                "description": "Demonstrate absolute emission reduction compared to previous year",
                "weight": 2.0,
                "auto_checkable": True,
            },
            {
                "id": "ct_03",
                "category": "management",
                "title": "Carbon management plan",
                "description": "Documented carbon management plan with reduction targets",
                "weight": 1.0,
                "auto_checkable": False,
            },
            {
                "id": "ct_04",
                "category": "governance",
                "title": "Board-level responsibility",
                "description": "Named board member or senior manager responsible for carbon management",
                "weight": 1.0,
                "auto_checkable": False,
            },
            {
                "id": "ct_05",
                "category": "reporting",
                "title": "Third-party verification",
                "description": "Data verified by independent third party",
                "weight": 1.0,
                "auto_checkable": False,
            },
        ],
    },
}

# =============================================================================
# DATA MODELS
# =============================================================================

class PhaseResult(BaseModel):
    """Result from a single workflow phase."""

    phase_name: str = Field(..., description="Phase identifier")
    phase_number: int = Field(default=0, ge=0)
    status: PhaseStatus = Field(...)
    duration_seconds: float = Field(default=0.0)
    completion_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    outputs: Dict[str, Any] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")
    mobile_summary: str = Field(default="")

class CriterionAssessment(BaseModel):
    """Assessment of a single certification criterion."""

    criterion_id: str = Field(default="")
    title: str = Field(default="")
    category: str = Field(default="")
    met: bool = Field(default=False)
    evidence: str = Field(default="", description="Evidence or reason")
    gap_severity: Optional[str] = Field(None, description="critical|major|minor|None")
    remediation_action: str = Field(default="")
    estimated_effort_hours: float = Field(default=0.0, ge=0.0)
    estimated_cost_gbp: float = Field(default=0.0, ge=0.0)

class ReadinessScorecard(BaseModel):
    """Overall readiness assessment scorecard."""

    certification_type: str = Field(default="")
    certification_name: str = Field(default="")
    readiness_level: str = Field(default="not_ready")
    overall_score: float = Field(default=0.0, ge=0.0, le=100.0)
    criteria_met: int = Field(default=0, ge=0)
    criteria_total: int = Field(default=0, ge=0)
    critical_gaps: int = Field(default=0, ge=0)
    major_gaps: int = Field(default=0, ge=0)
    minor_gaps: int = Field(default=0, ge=0)
    estimated_time_to_ready_months: int = Field(default=0, ge=0)
    estimated_cost_to_ready_gbp: float = Field(default=0.0, ge=0.0)
    criteria_assessments: List[CriterionAssessment] = Field(default_factory=list)

class GapClosureAction(BaseModel):
    """Action required to close a certification gap."""

    action_id: str = Field(default="")
    criterion_id: str = Field(default="")
    title: str = Field(default="")
    description: str = Field(default="")
    severity: str = Field(default="minor")
    priority: int = Field(default=0, ge=0)
    estimated_hours: float = Field(default=0.0, ge=0.0)
    estimated_cost_gbp: float = Field(default=0.0, ge=0.0)
    status: str = Field(default="not_started")
    deadline: str = Field(default="")

class CertificationDocument(BaseModel):
    """Certification submission document."""

    document_id: str = Field(default="")
    title: str = Field(default="")
    document_type: str = Field(default="")
    status: str = Field(default="not_started")
    description: str = Field(default="")
    auto_generated: bool = Field(default=False)
    content_summary: str = Field(default="")

class SubmissionRecord(BaseModel):
    """Certification submission record."""

    submission_id: str = Field(default="")
    certification_type: str = Field(default="")
    certification_body: str = Field(default="")
    submission_date: str = Field(default="")
    documents_submitted: List[str] = Field(default_factory=list)
    status: str = Field(default="not_started")
    expected_response_weeks: int = Field(default=4, ge=1)
    notes: List[str] = Field(default_factory=list)

class VerificationTracking(BaseModel):
    """Verification/audit tracking record."""

    tracking_id: str = Field(default="")
    certification_type: str = Field(default="")
    status: str = Field(default="not_started")
    audit_date: str = Field(default="")
    auditor: str = Field(default="")
    findings: List[str] = Field(default_factory=list)
    corrective_actions: List[str] = Field(default_factory=list)
    result: str = Field(default="pending")
    certificate_expiry: str = Field(default="")

class CertificationPathwayConfig(BaseModel):
    """Configuration for certification pathway workflow."""

    selected_certification: str = Field(
        default="sme_climate_hub",
        description="sme_climate_hub|b_corp|iso_14001|carbon_trust",
    )
    has_baseline: bool = Field(default=False)
    baseline_tco2e: float = Field(default=0.0, ge=0.0)
    has_targets: bool = Field(default=False)
    target_reduction_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    has_reduction_plan: bool = Field(default=False)
    has_board_commitment: bool = Field(default=False)
    has_environmental_policy: bool = Field(default=False)
    has_previous_year_data: bool = Field(default=False)
    previous_year_tco2e: float = Field(default=0.0, ge=0.0)
    employee_count: int = Field(default=1, ge=1)
    organization_name: str = Field(default="")
    entity_id: str = Field(default="")
    tenant_id: str = Field(default="")

class CertificationPathwayInput(BaseModel):
    """Complete input for certification pathway workflow."""

    config: CertificationPathwayConfig = Field(
        default_factory=CertificationPathwayConfig,
    )

class CertificationPathwayResult(BaseModel):
    """Complete result from certification pathway workflow."""

    workflow_id: str = Field(..., description="Unique execution ID")
    workflow_name: str = Field(default="sme_certification_pathway")
    pack_id: str = Field(default="PACK-026")
    status: WorkflowStatus = Field(...)
    phases: List[PhaseResult] = Field(default_factory=list)
    total_duration_seconds: float = Field(default=0.0)
    selected_certification: str = Field(default="")
    readiness: ReadinessScorecard = Field(default_factory=ReadinessScorecard)
    gap_closure_actions: List[GapClosureAction] = Field(default_factory=list)
    documents: List[CertificationDocument] = Field(default_factory=list)
    submission: Optional[SubmissionRecord] = Field(None)
    verification: Optional[VerificationTracking] = Field(None)
    next_steps: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")

# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================

class CertificationPathwayWorkflow:
    """
    6-phase workflow for SME sustainability certification.

    Phase 1: Pathway Selection - Choose certification
    Phase 2: Readiness Assessment - Score against criteria
    Phase 3: Gap Closure - Address identified gaps
    Phase 4: Documentation - Prepare submission documents
    Phase 5: Submission - Submit to certification body
    Phase 6: Verification Tracking - Track audit/verification

    Example:
        >>> wf = CertificationPathwayWorkflow()
        >>> inp = CertificationPathwayInput(
        ...     config=CertificationPathwayConfig(
        ...         selected_certification="sme_climate_hub",
        ...         has_baseline=True,
        ...         baseline_tco2e=100.0,
        ...         has_targets=True,
        ...     ),
        ... )
        >>> result = await wf.execute(inp)
        >>> assert result.readiness.readiness_level in ["ready", "nearly_ready"]
    """

    def __init__(self) -> None:
        self.workflow_id: str = _new_uuid()
        self._phase_results: List[PhaseResult] = []
        self._readiness: ReadinessScorecard = ReadinessScorecard()
        self._gap_actions: List[GapClosureAction] = []
        self._documents: List[CertificationDocument] = []
        self._submission: Optional[SubmissionRecord] = None
        self._verification: Optional[VerificationTracking] = None
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    async def execute(self, input_data: CertificationPathwayInput) -> CertificationPathwayResult:
        """Execute the 6-phase certification pathway workflow."""
        started_at = utcnow()
        config = input_data.config
        self.logger.info(
            "Starting certification pathway %s for %s (%s)",
            self.workflow_id, config.organization_name, config.selected_certification,
        )
        self._phase_results = []
        overall_status = WorkflowStatus.RUNNING

        try:
            phase1 = await self._phase_pathway_selection(config)
            self._phase_results.append(phase1)
            if phase1.status == PhaseStatus.FAILED:
                raise ValueError(f"PathwaySelection failed: {phase1.errors}")

            phase2 = await self._phase_readiness_assessment(config)
            self._phase_results.append(phase2)

            phase3 = await self._phase_gap_closure(config)
            self._phase_results.append(phase3)

            phase4 = await self._phase_documentation(config)
            self._phase_results.append(phase4)

            phase5 = await self._phase_submission(config)
            self._phase_results.append(phase5)

            phase6 = await self._phase_verification_tracking(config)
            self._phase_results.append(phase6)

            failed = [p for p in self._phase_results if p.status == PhaseStatus.FAILED]
            overall_status = WorkflowStatus.COMPLETED if not failed else WorkflowStatus.PARTIAL

        except Exception as exc:
            self.logger.error("Certification pathway failed: %s", exc, exc_info=True)
            overall_status = WorkflowStatus.FAILED
            self._phase_results.append(PhaseResult(
                phase_name="error", phase_number=99,
                status=PhaseStatus.FAILED, errors=[str(exc)],
                mobile_summary="Certification workflow failed.",
            ))

        elapsed = (utcnow() - started_at).total_seconds()
        next_steps = self._generate_next_steps(config)

        result = CertificationPathwayResult(
            workflow_id=self.workflow_id,
            status=overall_status,
            phases=self._phase_results,
            total_duration_seconds=round(elapsed, 4),
            selected_certification=config.selected_certification,
            readiness=self._readiness,
            gap_closure_actions=self._gap_actions,
            documents=self._documents,
            submission=self._submission,
            verification=self._verification,
            next_steps=next_steps,
        )
        result.provenance_hash = _compute_hash(
            result.model_dump_json(exclude={"provenance_hash"})
        )
        return result

    # -------------------------------------------------------------------------
    # Phase 1: Pathway Selection
    # -------------------------------------------------------------------------

    async def _phase_pathway_selection(self, config: CertificationPathwayConfig) -> PhaseResult:
        """Validate and confirm certification pathway selection."""
        started = utcnow()
        warnings: List[str] = []
        errors: List[str] = []
        outputs: Dict[str, Any] = {}

        cert_type = config.selected_certification
        if cert_type not in CERTIFICATION_CRITERIA:
            errors.append(f"Unknown certification: {cert_type}")
            return PhaseResult(
                phase_name="pathway_selection", phase_number=1,
                status=PhaseStatus.FAILED, errors=errors,
                mobile_summary="Invalid certification selected.",
            )

        cert_info = CERTIFICATION_CRITERIA[cert_type]
        outputs["certification"] = cert_info["name"]
        outputs["provider"] = cert_info["provider"]
        outputs["cost_gbp"] = cert_info["cost_gbp"]
        outputs["difficulty"] = cert_info["difficulty"]
        outputs["timeline_months"] = cert_info["timeline_months"]
        outputs["requirements_count"] = len(cert_info["requirements"])

        # Difficulty warnings
        if cert_info["difficulty"] == "hard" and config.employee_count < 50:
            warnings.append(
                f"{cert_info['name']} is challenging for small organisations. "
                "Consider SME Climate Hub as a starting point."
            )

        if cert_info["cost_gbp"] > 0:
            warnings.append(
                f"Certification cost: GBP {cert_info['cost_gbp']:,} "
                f"(plus implementation costs)"
            )

        elapsed = (utcnow() - started).total_seconds()
        return PhaseResult(
            phase_name="pathway_selection", phase_number=1,
            status=PhaseStatus.COMPLETED,
            duration_seconds=round(elapsed, 4),
            completion_pct=100.0,
            outputs=outputs, warnings=warnings,
            provenance_hash=_compute_hash(json.dumps(outputs, sort_keys=True, default=str)),
            mobile_summary=f"Selected: {cert_info['name']} ({cert_info['difficulty']}, ~{cert_info['timeline_months']}m)",
        )

    # -------------------------------------------------------------------------
    # Phase 2: Readiness Assessment
    # -------------------------------------------------------------------------

    async def _phase_readiness_assessment(self, config: CertificationPathwayConfig) -> PhaseResult:
        """Assess readiness against certification criteria."""
        started = utcnow()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        cert_info = CERTIFICATION_CRITERIA[config.selected_certification]
        requirements = cert_info["requirements"]

        assessments: List[CriterionAssessment] = []
        total_weight = 0.0
        met_weight = 0.0

        for req in requirements:
            met, evidence, gap_sev, remediation, hours, cost = self._assess_criterion(req, config)

            assessment = CriterionAssessment(
                criterion_id=req["id"],
                title=req["title"],
                category=req.get("category", ""),
                met=met,
                evidence=evidence,
                gap_severity=gap_sev,
                remediation_action=remediation,
                estimated_effort_hours=hours,
                estimated_cost_gbp=cost,
            )
            assessments.append(assessment)

            weight = req.get("weight", 1.0)
            total_weight += weight
            if met:
                met_weight += weight

        overall_score = (met_weight / max(total_weight, 0.01)) * 100
        criteria_met = sum(1 for a in assessments if a.met)
        critical_gaps = sum(1 for a in assessments if a.gap_severity == GapSeverity.CRITICAL.value)
        major_gaps = sum(1 for a in assessments if a.gap_severity == GapSeverity.MAJOR.value)
        minor_gaps = sum(1 for a in assessments if a.gap_severity == GapSeverity.MINOR.value)

        # Determine readiness level
        if overall_score >= 90 and critical_gaps == 0:
            readiness = ReadinessLevel.READY.value
        elif overall_score >= 70 and critical_gaps == 0:
            readiness = ReadinessLevel.NEARLY_READY.value
        elif overall_score >= 40:
            readiness = ReadinessLevel.SIGNIFICANT_GAPS.value
        else:
            readiness = ReadinessLevel.NOT_READY.value

        # Estimate time to ready
        total_hours = sum(a.estimated_effort_hours for a in assessments if not a.met)
        total_cost = sum(a.estimated_cost_gbp for a in assessments if not a.met)
        months_to_ready = max(int(total_hours / 40) + 1, 1) if total_hours > 0 else 0

        self._readiness = ReadinessScorecard(
            certification_type=config.selected_certification,
            certification_name=cert_info["name"],
            readiness_level=readiness,
            overall_score=round(overall_score, 1),
            criteria_met=criteria_met,
            criteria_total=len(requirements),
            critical_gaps=critical_gaps,
            major_gaps=major_gaps,
            minor_gaps=minor_gaps,
            estimated_time_to_ready_months=months_to_ready,
            estimated_cost_to_ready_gbp=round(total_cost, 2),
            criteria_assessments=assessments,
        )

        outputs["readiness_level"] = readiness
        outputs["score"] = round(overall_score, 1)
        outputs["criteria_met"] = criteria_met
        outputs["criteria_total"] = len(requirements)
        outputs["critical_gaps"] = critical_gaps
        outputs["months_to_ready"] = months_to_ready
        outputs["cost_to_ready_gbp"] = round(total_cost, 2)

        elapsed = (utcnow() - started).total_seconds()
        self.logger.info(
            "Readiness: %s (%.1f%%), %d/%d criteria met, %d critical gaps",
            readiness, overall_score, criteria_met, len(requirements), critical_gaps,
        )
        return PhaseResult(
            phase_name="readiness_assessment", phase_number=2,
            status=PhaseStatus.COMPLETED,
            duration_seconds=round(elapsed, 4),
            completion_pct=100.0,
            outputs=outputs, warnings=warnings,
            provenance_hash=_compute_hash(json.dumps(outputs, sort_keys=True, default=str)),
            mobile_summary=f"Readiness: {readiness} ({overall_score:.0f}%, {criteria_met}/{len(requirements)} criteria)",
        )

    def _assess_criterion(
        self, req: Dict[str, Any], config: CertificationPathwayConfig,
    ) -> tuple:
        """Assess a single criterion. Returns (met, evidence, gap_severity, remediation, hours, cost)."""
        req_id = req["id"]
        category = req.get("category", "")
        auto = req.get("auto_checkable", False)

        # Auto-checkable criteria
        if auto and category == "measurement":
            if config.has_baseline and config.baseline_tco2e > 0:
                return (True, f"Baseline measured: {config.baseline_tco2e:.1f} tCO2e", None, "", 0, 0)
            return (
                False, "No GHG baseline established",
                GapSeverity.CRITICAL.value,
                "Complete SME Express Onboarding to establish baseline",
                4.0, 0,
            )

        if auto and category == "target":
            if config.has_targets and config.target_reduction_pct >= 50:
                return (True, f"Target set: {config.target_reduction_pct:.0f}% reduction", None, "", 0, 0)
            return (
                False, "No reduction target set",
                GapSeverity.MAJOR.value,
                "Set SBTi-aligned target (50% by 2030) using Standard Setup workflow",
                2.0, 0,
            )

        if auto and category == "reduction":
            if config.has_previous_year_data and config.previous_year_tco2e > config.baseline_tco2e:
                return (True, "Year-on-year reduction demonstrated", None, "", 0, 0)
            if not config.has_previous_year_data:
                return (
                    False, "No previous year data for comparison",
                    GapSeverity.MAJOR.value,
                    "Collect at least 2 years of emissions data",
                    8.0, 0,
                )
            return (
                False, "No year-on-year reduction demonstrated",
                GapSeverity.CRITICAL.value,
                "Implement quick wins to achieve measurable reduction",
                20.0, 500,
            )

        if auto and category == "improvement":
            if config.has_reduction_plan:
                return (True, "Reduction plan in place", None, "", 0, 0)
            return (
                False, "No documented reduction plan",
                GapSeverity.MAJOR.value,
                "Develop reduction roadmap using Standard Setup workflow",
                8.0, 0,
            )

        if auto and category == "commitment":
            if config.has_targets:
                return (True, "Net-zero commitment target established", None, "", 0, 0)
            return (
                False, "No public commitment made",
                GapSeverity.MAJOR.value,
                "Sign SME Climate Hub pledge",
                1.0, 0,
            )

        # Non-auto criteria - mark as uncertain
        if category == "governance" and config.has_board_commitment:
            return (True, "Board-level commitment confirmed", None, "", 0, 0)
        if category == "policy" and config.has_environmental_policy:
            return (True, "Environmental policy in place", None, "", 0, 0)

        # Default: uncertain / not met
        return (
            False,
            f"Requires manual verification: {req['title']}",
            GapSeverity.MINOR.value,
            f"Review and address: {req['description']}",
            8.0, 500,
        )

    # -------------------------------------------------------------------------
    # Phase 3: Gap Closure
    # -------------------------------------------------------------------------

    async def _phase_gap_closure(self, config: CertificationPathwayConfig) -> PhaseResult:
        """Generate gap closure action plan."""
        started = utcnow()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        self._gap_actions = []
        priority = 0

        for assessment in self._readiness.criteria_assessments:
            if assessment.met:
                continue

            priority += 1
            action = GapClosureAction(
                action_id=f"gap_{assessment.criterion_id}",
                criterion_id=assessment.criterion_id,
                title=f"Close gap: {assessment.title}",
                description=assessment.remediation_action,
                severity=assessment.gap_severity or GapSeverity.MINOR.value,
                priority=priority,
                estimated_hours=assessment.estimated_effort_hours,
                estimated_cost_gbp=assessment.estimated_cost_gbp,
                status="not_started",
            )
            self._gap_actions.append(action)

        # Sort by severity then priority
        severity_order = {
            GapSeverity.CRITICAL.value: 0,
            GapSeverity.MAJOR.value: 1,
            GapSeverity.MINOR.value: 2,
            GapSeverity.INFORMATIONAL.value: 3,
        }
        self._gap_actions.sort(
            key=lambda a: (severity_order.get(a.severity, 9), a.priority)
        )
        for i, action in enumerate(self._gap_actions, 1):
            action.priority = i

        outputs["total_gaps"] = len(self._gap_actions)
        outputs["critical_gaps"] = sum(1 for a in self._gap_actions if a.severity == GapSeverity.CRITICAL.value)
        outputs["total_effort_hours"] = sum(a.estimated_hours for a in self._gap_actions)
        outputs["total_cost_gbp"] = round(sum(a.estimated_cost_gbp for a in self._gap_actions), 2)

        elapsed = (utcnow() - started).total_seconds()
        return PhaseResult(
            phase_name="gap_closure", phase_number=3,
            status=PhaseStatus.COMPLETED,
            duration_seconds=round(elapsed, 4),
            completion_pct=100.0,
            outputs=outputs, warnings=warnings,
            provenance_hash=_compute_hash(json.dumps(outputs, sort_keys=True, default=str)),
            mobile_summary=f"{len(self._gap_actions)} gaps to close ({outputs['total_effort_hours']:.0f} hours est.)",
        )

    # -------------------------------------------------------------------------
    # Phase 4: Documentation
    # -------------------------------------------------------------------------

    async def _phase_documentation(self, config: CertificationPathwayConfig) -> PhaseResult:
        """Prepare certification submission documents."""
        started = utcnow()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        cert_info = CERTIFICATION_CRITERIA[config.selected_certification]
        cert_name = cert_info["name"]

        # Standard document templates
        doc_templates: List[Dict[str, Any]] = [
            {
                "id": "org_profile",
                "title": "Organization Profile",
                "type": "company_profile",
                "auto": True,
                "summary": f"Company details for {config.organization_name}",
            },
            {
                "id": "baseline_report",
                "title": "GHG Emissions Baseline Report",
                "type": "baseline_report",
                "auto": config.has_baseline,
                "summary": f"Baseline: {config.baseline_tco2e:.1f} tCO2e" if config.has_baseline else "Pending baseline",
            },
            {
                "id": "target_statement",
                "title": "Emission Reduction Target Statement",
                "type": "target_declaration",
                "auto": config.has_targets,
                "summary": f"Target: {config.target_reduction_pct:.0f}% reduction" if config.has_targets else "Pending target",
            },
            {
                "id": "action_plan",
                "title": "Carbon Reduction Action Plan",
                "type": "action_plan",
                "auto": config.has_reduction_plan,
                "summary": "Documented reduction roadmap" if config.has_reduction_plan else "Pending plan",
            },
        ]

        # Add certification-specific documents
        if config.selected_certification == "iso_14001":
            doc_templates.extend([
                {"id": "env_policy", "title": "Environmental Policy Statement", "type": "policy",
                 "auto": config.has_environmental_policy, "summary": "Board-approved environmental policy"},
                {"id": "aspects_register", "title": "Environmental Aspects Register", "type": "register",
                 "auto": False, "summary": "Register of significant environmental aspects"},
                {"id": "legal_register", "title": "Legal Compliance Register", "type": "register",
                 "auto": False, "summary": "Register of applicable environmental legislation"},
            ])

        if config.selected_certification == "carbon_trust":
            doc_templates.append(
                {"id": "mgmt_plan", "title": "Carbon Management Plan", "type": "management_plan",
                 "auto": False, "summary": "Documented carbon management plan with KPIs"},
            )

        self._documents = []
        for tmpl in doc_templates:
            status = DocumentStatus.READY.value if tmpl["auto"] else DocumentStatus.NOT_STARTED.value
            doc = CertificationDocument(
                document_id=tmpl["id"],
                title=tmpl["title"],
                document_type=tmpl.get("type", ""),
                status=status,
                description=f"Required for {cert_name} certification",
                auto_generated=tmpl["auto"],
                content_summary=tmpl.get("summary", ""),
            )
            self._documents.append(doc)

        ready_count = sum(1 for d in self._documents if d.status == DocumentStatus.READY.value)
        outputs["documents_total"] = len(self._documents)
        outputs["documents_ready"] = ready_count
        outputs["documents_pending"] = len(self._documents) - ready_count

        elapsed = (utcnow() - started).total_seconds()
        return PhaseResult(
            phase_name="documentation", phase_number=4,
            status=PhaseStatus.COMPLETED,
            duration_seconds=round(elapsed, 4),
            completion_pct=100.0,
            outputs=outputs, warnings=warnings,
            provenance_hash=_compute_hash(json.dumps(outputs, sort_keys=True, default=str)),
            mobile_summary=f"Documents: {ready_count}/{len(self._documents)} ready",
        )

    # -------------------------------------------------------------------------
    # Phase 5: Submission
    # -------------------------------------------------------------------------

    async def _phase_submission(self, config: CertificationPathwayConfig) -> PhaseResult:
        """Prepare submission to certification body."""
        started = utcnow()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        cert_info = CERTIFICATION_CRITERIA[config.selected_certification]
        all_ready = all(d.status == DocumentStatus.READY.value for d in self._documents)
        gaps_closed = len(self._gap_actions) == 0 or self._readiness.readiness_level in [
            ReadinessLevel.READY.value, ReadinessLevel.NEARLY_READY.value,
        ]

        notes: List[str] = []
        if not all_ready:
            pending = [d.title for d in self._documents if d.status != DocumentStatus.READY.value]
            notes.append(f"Pending documents: {', '.join(pending[:3])}")
            warnings.append("Not all documents are ready; complete pending items before submission")

        if not gaps_closed:
            notes.append("Critical or major gaps remain; address before submission")

        submission_status = (
            CertificationStatus.PREPARING.value
            if (all_ready and gaps_closed)
            else CertificationStatus.NOT_STARTED.value
        )

        self._submission = SubmissionRecord(
            submission_id=_new_uuid(),
            certification_type=config.selected_certification,
            certification_body=cert_info["provider"],
            submission_date="" if not all_ready else utcnow().strftime("%Y-%m-%d"),
            documents_submitted=[d.document_id for d in self._documents if d.status == DocumentStatus.READY.value],
            status=submission_status,
            expected_response_weeks=cert_info.get("timeline_months", 3) * 4,
            notes=notes,
        )

        outputs["submission_ready"] = all_ready and gaps_closed
        outputs["status"] = submission_status
        outputs["documents_submitted"] = len(self._submission.documents_submitted)

        elapsed = (utcnow() - started).total_seconds()
        return PhaseResult(
            phase_name="submission", phase_number=5,
            status=PhaseStatus.COMPLETED,
            duration_seconds=round(elapsed, 4),
            completion_pct=100.0,
            outputs=outputs, warnings=warnings,
            provenance_hash=_compute_hash(json.dumps(outputs, sort_keys=True, default=str)),
            mobile_summary=f"Submission: {'Ready' if all_ready and gaps_closed else 'Not yet ready'}",
        )

    # -------------------------------------------------------------------------
    # Phase 6: Verification Tracking
    # -------------------------------------------------------------------------

    async def _phase_verification_tracking(self, config: CertificationPathwayConfig) -> PhaseResult:
        """Set up verification/audit tracking framework."""
        started = utcnow()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        cert_info = CERTIFICATION_CRITERIA[config.selected_certification]

        findings: List[str] = []
        if self._readiness.readiness_level == ReadinessLevel.NOT_READY.value:
            findings.append("Not yet submitted; tracking framework set up for future use")

        self._verification = VerificationTracking(
            tracking_id=_new_uuid(),
            certification_type=config.selected_certification,
            status=CertificationStatus.NOT_STARTED.value,
            audit_date="",
            auditor=cert_info["provider"],
            findings=findings,
            corrective_actions=[a.title for a in self._gap_actions if a.severity == GapSeverity.CRITICAL.value],
            result="pending",
            certificate_expiry="",
        )

        outputs["verification_set_up"] = True
        outputs["renewal_years"] = cert_info.get("renewal_years", 3)
        outputs["pending_corrective_actions"] = len(self._verification.corrective_actions)

        elapsed = (utcnow() - started).total_seconds()
        return PhaseResult(
            phase_name="verification_tracking", phase_number=6,
            status=PhaseStatus.COMPLETED,
            duration_seconds=round(elapsed, 4),
            completion_pct=100.0,
            outputs=outputs, warnings=warnings,
            provenance_hash=_compute_hash(json.dumps(outputs, sort_keys=True, default=str)),
            mobile_summary=f"Verification tracking set up (renewal every {cert_info.get('renewal_years', 3)} yrs)",
        )

    # -------------------------------------------------------------------------
    # Next Steps
    # -------------------------------------------------------------------------

    def _generate_next_steps(self, config: CertificationPathwayConfig) -> List[str]:
        steps: List[str] = []
        readiness = self._readiness

        if readiness.readiness_level == ReadinessLevel.READY.value:
            steps.append(
                f"Submit your {readiness.certification_name} application."
            )
        elif readiness.readiness_level == ReadinessLevel.NEARLY_READY.value:
            steps.append(
                f"Close {readiness.minor_gaps} minor gap(s) to achieve full readiness."
            )
        else:
            critical = [a for a in self._gap_actions if a.severity == GapSeverity.CRITICAL.value]
            if critical:
                steps.append(
                    f"Address {len(critical)} critical gap(s): {critical[0].title}."
                )
            if not config.has_baseline:
                steps.append("Complete Express Onboarding to establish your GHG baseline.")
            if not config.has_targets:
                steps.append("Set emission reduction targets using the Standard Setup workflow.")

        pending_docs = [d for d in self._documents if d.status != DocumentStatus.READY.value]
        if pending_docs:
            steps.append(f"Complete {len(pending_docs)} pending document(s).")

        cert_info = CERTIFICATION_CRITERIA.get(config.selected_certification, {})
        if config.selected_certification != "sme_climate_hub" and readiness.readiness_level in [
            ReadinessLevel.NOT_READY.value, ReadinessLevel.SIGNIFICANT_GAPS.value,
        ]:
            steps.append(
                "Consider starting with SME Climate Hub (free, no audit) while "
                f"working towards {cert_info.get('name', 'certification')}."
            )

        steps.append("Set a calendar reminder for certification renewal.")

        return steps
