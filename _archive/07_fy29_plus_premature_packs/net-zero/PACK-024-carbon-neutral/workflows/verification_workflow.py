# -*- coding: utf-8 -*-
"""
Verification Workflow
==========================

4-phase workflow for third-party verification of carbon neutrality
claims within PACK-024 Carbon Neutral Pack.  Prepares the verification
package, manages the verification body engagement, tracks findings
resolution, and issues the verification opinion.

Phases:
    1. PackagePreparation    -- Compile verification package documentation
    2. BodyEngagement        -- Manage verification body selection and scope
    3. FindingsResolution    -- Track and resolve verification findings
    4. OpinionIssuance       -- Issue verification opinion and statement

Regulatory references:
    - ISO 14064-3:2019 (Verification of GHG assertions)
    - PAS 2060:2014 (Independent validation)
    - ISO 14065:2020 (Verification body requirements)
    - ISAE 3410 (Assurance on GHG statements)
    - AA1000AS v3 (Sustainability assurance)

Author: GreenLang Team
Version: 24.0.0
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

_MODULE_VERSION = "24.0.0"

def _new_uuid() -> str:
    return uuid.uuid4().hex

def _compute_hash(data: Any) -> str:
    if isinstance(data, dict):
        data = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(str(data).encode("utf-8")).hexdigest()

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

class VerificationPhase(str, Enum):
    PACKAGE_PREPARATION = "package_preparation"
    BODY_ENGAGEMENT = "body_engagement"
    FINDINGS_RESOLUTION = "findings_resolution"
    OPINION_ISSUANCE = "opinion_issuance"

class AssuranceLevel(str, Enum):
    LIMITED = "limited"
    REASONABLE = "reasonable"

class VerificationStandard(str, Enum):
    ISO_14064_3 = "iso_14064_3"
    ISAE_3410 = "isae_3410"
    AA1000AS = "aa1000as"
    PAS_2060 = "pas_2060"

class FindingCategory(str, Enum):
    MATERIAL_MISSTATEMENT = "material_misstatement"
    NON_CONFORMITY = "non_conformity"
    OBSERVATION = "observation"
    OPPORTUNITY_FOR_IMPROVEMENT = "opportunity_for_improvement"
    GOOD_PRACTICE = "good_practice"

class FindingSeverity(str, Enum):
    CRITICAL = "critical"
    MAJOR = "major"
    MINOR = "minor"
    INFORMATIONAL = "informational"

class OpinionType(str, Enum):
    UNMODIFIED = "unmodified"
    QUALIFIED = "qualified"
    ADVERSE = "adverse"
    DISCLAIMER = "disclaimer"

# =============================================================================
# REFERENCE DATA
# =============================================================================

# Verification package checklist
PACKAGE_CHECKLIST: List[Dict[str, Any]] = [
    {"item": "GHG Inventory Report", "category": "emissions", "required": True},
    {"item": "Scope 1 Calculation Workbooks", "category": "emissions", "required": True},
    {"item": "Scope 2 Calculation Workbooks", "category": "emissions", "required": True},
    {"item": "Scope 3 Calculation Workbooks", "category": "emissions", "required": True},
    {"item": "Emission Factor Sources", "category": "methodology", "required": True},
    {"item": "Activity Data Records", "category": "data", "required": True},
    {"item": "Carbon Management Plan", "category": "strategy", "required": True},
    {"item": "Credit Retirement Certificates", "category": "offsetting", "required": True},
    {"item": "Registry Account Screenshots", "category": "offsetting", "required": True},
    {"item": "Neutralization Balance Sheet", "category": "balance", "required": True},
    {"item": "Qualifying Explanatory Statement", "category": "claims", "required": True},
    {"item": "Previous Verification Reports", "category": "history", "required": False},
    {"item": "Organizational Boundary Documentation", "category": "boundary", "required": True},
    {"item": "Data Quality Assessment", "category": "quality", "required": True},
    {"item": "Uncertainty Analysis", "category": "quality", "required": True},
]

# Accredited verification bodies (example)
VERIFICATION_BODIES: List[Dict[str, Any]] = [
    {"name": "Bureau Veritas", "accreditations": ["ISO 14065", "ISAE 3410"], "tier": "tier_1"},
    {"name": "SGS", "accreditations": ["ISO 14065", "ISAE 3410", "AA1000AS"], "tier": "tier_1"},
    {"name": "DNV", "accreditations": ["ISO 14065", "ISAE 3410"], "tier": "tier_1"},
    {"name": "LRQA", "accreditations": ["ISO 14065", "AA1000AS"], "tier": "tier_1"},
    {"name": "EY Climate Change", "accreditations": ["ISAE 3410"], "tier": "tier_2"},
    {"name": "Deloitte Sustainability", "accreditations": ["ISAE 3410"], "tier": "tier_2"},
    {"name": "PwC Sustainability", "accreditations": ["ISAE 3410"], "tier": "tier_2"},
    {"name": "KPMG Climate", "accreditations": ["ISAE 3410"], "tier": "tier_2"},
]

# =============================================================================
# DATA MODELS
# =============================================================================

class PhaseResult(BaseModel):
    phase_name: str = Field(...)
    status: PhaseStatus = Field(...)
    duration_seconds: float = Field(default=0.0)
    outputs: Dict[str, Any] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")

class VerificationFinding(BaseModel):
    finding_id: str = Field(default="")
    category: FindingCategory = Field(default=FindingCategory.OBSERVATION)
    severity: FindingSeverity = Field(default=FindingSeverity.INFORMATIONAL)
    title: str = Field(default="")
    description: str = Field(default="")
    affected_scope: str = Field(default="")
    root_cause: str = Field(default="")
    recommendation: str = Field(default="")
    response: str = Field(default="")
    resolution_status: str = Field(default="open")
    resolution_date: Optional[datetime] = Field(None)
    evidence_reference: str = Field(default="")

class VerificationOpinion(BaseModel):
    opinion_id: str = Field(default="")
    opinion_type: OpinionType = Field(default=OpinionType.UNMODIFIED)
    assurance_level: AssuranceLevel = Field(default=AssuranceLevel.LIMITED)
    standard: VerificationStandard = Field(default=VerificationStandard.ISO_14064_3)
    verification_body: str = Field(default="")
    lead_verifier: str = Field(default="")
    opinion_date: Optional[datetime] = Field(None)
    opinion_text: str = Field(default="")
    scope_of_verification: str = Field(default="")
    materiality_threshold_pct: float = Field(default=5.0)
    total_findings: int = Field(default=0)
    critical_findings: int = Field(default=0)
    resolved_findings: int = Field(default=0)
    certificate_number: str = Field(default="")
    valid_until: Optional[datetime] = Field(None)

class VerificationConfig(BaseModel):
    org_name: str = Field(default="")
    reporting_year: int = Field(default=2025, ge=2015, le=2050)
    assurance_level: AssuranceLevel = Field(default=AssuranceLevel.LIMITED)
    standard: VerificationStandard = Field(default=VerificationStandard.ISO_14064_3)
    total_emissions_tco2e: float = Field(default=0.0, ge=0.0)
    materiality_threshold_pct: float = Field(default=5.0, ge=0.5, le=10.0)
    preferred_body_tier: str = Field(default="tier_1")
    pas2060_compliance: bool = Field(default=True)
    entity_id: str = Field(default="")
    tenant_id: str = Field(default="")

class VerificationResult(BaseModel):
    workflow_id: str = Field(...)
    workflow_name: str = Field(default="verification")
    status: WorkflowStatus = Field(...)
    phases: List[PhaseResult] = Field(default_factory=list)
    total_duration_seconds: float = Field(default=0.0)
    package_completeness_pct: float = Field(default=0.0)
    findings: List[VerificationFinding] = Field(default_factory=list)
    opinion: Optional[VerificationOpinion] = Field(None)
    is_verified: bool = Field(default=False)
    assurance_level: AssuranceLevel = Field(default=AssuranceLevel.LIMITED)
    provenance_hash: str = Field(default="")

# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================

class VerificationWorkflow:
    """
    4-phase verification workflow for PACK-024.

    Manages the third-party verification process for carbon neutrality
    claims, from package preparation through verification body engagement,
    findings resolution, and opinion issuance.

    Attributes:
        workflow_id: Unique execution identifier.
    """

    def __init__(self) -> None:
        self.workflow_id: str = _new_uuid()
        self._phase_results: List[PhaseResult] = []
        self._findings: List[VerificationFinding] = []
        self._opinion: Optional[VerificationOpinion] = None
        self._package_completeness: float = 0.0
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    async def execute(self, config: VerificationConfig) -> VerificationResult:
        """Execute the 4-phase verification workflow."""
        started_at = utcnow()
        self.logger.info("Starting verification workflow %s", self.workflow_id)
        self._phase_results = []
        overall_status = WorkflowStatus.RUNNING

        try:
            phase1 = await self._phase_package_preparation(config)
            self._phase_results.append(phase1)

            phase2 = await self._phase_body_engagement(config)
            self._phase_results.append(phase2)

            phase3 = await self._phase_findings_resolution(config)
            self._phase_results.append(phase3)

            phase4 = await self._phase_opinion_issuance(config)
            self._phase_results.append(phase4)

            failed = [p for p in self._phase_results if p.status == PhaseStatus.FAILED]
            overall_status = WorkflowStatus.COMPLETED if not failed else WorkflowStatus.PARTIAL

        except Exception as exc:
            self.logger.error("Verification workflow failed: %s", exc, exc_info=True)
            overall_status = WorkflowStatus.FAILED
            self._phase_results.append(PhaseResult(
                phase_name="error", status=PhaseStatus.FAILED, errors=[str(exc)],
            ))

        elapsed = (utcnow() - started_at).total_seconds()
        is_verified = (
            self._opinion is not None
            and self._opinion.opinion_type in (OpinionType.UNMODIFIED, OpinionType.QUALIFIED)
            and self._opinion.critical_findings == 0
        )

        result = VerificationResult(
            workflow_id=self.workflow_id,
            status=overall_status,
            phases=self._phase_results,
            total_duration_seconds=round(elapsed, 4),
            package_completeness_pct=round(self._package_completeness, 1),
            findings=self._findings,
            opinion=self._opinion,
            is_verified=is_verified,
            assurance_level=config.assurance_level,
        )
        result.provenance_hash = _compute_hash(
            result.model_dump_json(exclude={"provenance_hash"})
        )
        return result

    async def _phase_package_preparation(self, config: VerificationConfig) -> PhaseResult:
        """Compile verification package documentation."""
        started = utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []
        errors: List[str] = []

        available_count = 0
        required_count = 0
        for item in PACKAGE_CHECKLIST:
            if item["required"]:
                required_count += 1
                available_count += 1  # Assume available after pack execution

        total_items = len(PACKAGE_CHECKLIST)
        completeness = (available_count / max(required_count, 1)) * 100.0
        self._package_completeness = completeness

        outputs["total_items"] = total_items
        outputs["required_items"] = required_count
        outputs["available_items"] = available_count
        outputs["completeness_pct"] = round(completeness, 1)
        outputs["categories"] = list(set(item["category"] for item in PACKAGE_CHECKLIST))

        if completeness < 100.0:
            warnings.append(f"Package completeness {completeness:.0f}% -- some items missing")

        status = PhaseStatus.COMPLETED
        elapsed = (utcnow() - started).total_seconds()
        return PhaseResult(
            phase_name=VerificationPhase.PACKAGE_PREPARATION.value,
            status=status, duration_seconds=round(elapsed, 4),
            outputs=outputs, warnings=warnings, errors=errors,
            provenance_hash=_compute_hash(outputs),
        )

    async def _phase_body_engagement(self, config: VerificationConfig) -> PhaseResult:
        """Select and engage verification body."""
        started = utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []
        errors: List[str] = []

        eligible = [
            vb for vb in VERIFICATION_BODIES
            if vb["tier"] == config.preferred_body_tier
        ]
        if not eligible:
            eligible = VERIFICATION_BODIES[:3]

        outputs["eligible_bodies"] = len(eligible)
        outputs["preferred_tier"] = config.preferred_body_tier
        outputs["recommended_body"] = eligible[0]["name"] if eligible else "N/A"
        outputs["accreditations"] = eligible[0]["accreditations"] if eligible else []
        outputs["assurance_level"] = config.assurance_level.value
        outputs["standard"] = config.standard.value

        status = PhaseStatus.COMPLETED
        elapsed = (utcnow() - started).total_seconds()
        return PhaseResult(
            phase_name=VerificationPhase.BODY_ENGAGEMENT.value,
            status=status, duration_seconds=round(elapsed, 4),
            outputs=outputs, warnings=warnings, errors=errors,
            provenance_hash=_compute_hash(outputs),
        )

    async def _phase_findings_resolution(self, config: VerificationConfig) -> PhaseResult:
        """Track and resolve verification findings."""
        started = utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []
        errors: List[str] = []

        # Generate sample findings (in production, these come from verifier)
        findings = [
            VerificationFinding(
                finding_id=_new_uuid(),
                category=FindingCategory.OBSERVATION,
                severity=FindingSeverity.MINOR,
                title="Scope 3 Category 1 data quality",
                description="Spend-based estimates could be improved with supplier-specific factors",
                affected_scope="scope_3",
                recommendation="Request supplier-specific emission data for top 10 suppliers",
                resolution_status="resolved",
            ),
            VerificationFinding(
                finding_id=_new_uuid(),
                category=FindingCategory.OPPORTUNITY_FOR_IMPROVEMENT,
                severity=FindingSeverity.INFORMATIONAL,
                title="Credit portfolio documentation",
                description="Consider adding project-level impact reports to disclosure",
                recommendation="Include project site visit reports where available",
                resolution_status="acknowledged",
            ),
        ]

        self._findings = findings
        resolved = sum(1 for f in findings if f.resolution_status in ("resolved", "acknowledged"))

        outputs["total_findings"] = len(findings)
        outputs["critical_findings"] = sum(1 for f in findings if f.severity == FindingSeverity.CRITICAL)
        outputs["major_findings"] = sum(1 for f in findings if f.severity == FindingSeverity.MAJOR)
        outputs["minor_findings"] = sum(1 for f in findings if f.severity == FindingSeverity.MINOR)
        outputs["resolved_findings"] = resolved

        status = PhaseStatus.COMPLETED
        elapsed = (utcnow() - started).total_seconds()
        return PhaseResult(
            phase_name=VerificationPhase.FINDINGS_RESOLUTION.value,
            status=status, duration_seconds=round(elapsed, 4),
            outputs=outputs, warnings=warnings, errors=errors,
            provenance_hash=_compute_hash(outputs),
        )

    async def _phase_opinion_issuance(self, config: VerificationConfig) -> PhaseResult:
        """Issue verification opinion and statement."""
        started = utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []
        errors: List[str] = []

        critical = sum(1 for f in self._findings if f.severity == FindingSeverity.CRITICAL)
        major = sum(1 for f in self._findings if f.severity == FindingSeverity.MAJOR)

        if critical > 0:
            opinion_type = OpinionType.ADVERSE
        elif major > 0:
            opinion_type = OpinionType.QUALIFIED
        else:
            opinion_type = OpinionType.UNMODIFIED

        self._opinion = VerificationOpinion(
            opinion_id=_new_uuid(),
            opinion_type=opinion_type,
            assurance_level=config.assurance_level,
            standard=config.standard,
            verification_body=VERIFICATION_BODIES[0]["name"],
            opinion_date=utcnow(),
            opinion_text=(
                f"Based on our {config.assurance_level.value} assurance engagement "
                f"conducted in accordance with {config.standard.value}, "
                f"{'nothing has come to our attention' if config.assurance_level == AssuranceLevel.LIMITED else 'we are satisfied'} "
                f"that the carbon neutrality claim of {config.org_name} for the "
                f"year {config.reporting_year} is {'fairly stated' if opinion_type == OpinionType.UNMODIFIED else 'stated with qualifications'}."
            ),
            scope_of_verification=f"Carbon neutrality claim for {config.org_name}, {config.reporting_year}",
            materiality_threshold_pct=config.materiality_threshold_pct,
            total_findings=len(self._findings),
            critical_findings=critical,
            resolved_findings=sum(1 for f in self._findings if f.resolution_status == "resolved"),
            certificate_number=f"CN-{config.reporting_year}-{_new_uuid()[:8].upper()}",
        )

        outputs["opinion_type"] = opinion_type.value
        outputs["assurance_level"] = config.assurance_level.value
        outputs["certificate_number"] = self._opinion.certificate_number
        outputs["is_positive_opinion"] = opinion_type in (OpinionType.UNMODIFIED, OpinionType.QUALIFIED)

        status = PhaseStatus.COMPLETED
        elapsed = (utcnow() - started).total_seconds()
        return PhaseResult(
            phase_name=VerificationPhase.OPINION_ISSUANCE.value,
            status=status, duration_seconds=round(elapsed, 4),
            outputs=outputs, warnings=warnings, errors=errors,
            provenance_hash=_compute_hash(outputs),
        )
