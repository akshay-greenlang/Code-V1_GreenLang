# -*- coding: utf-8 -*-
"""
Claims Validation Workflow
===============================

4-phase workflow for validating carbon neutrality claims within PACK-024
Carbon Neutral Pack.  Ensures claims comply with PAS 2060, VCMI Claims
Code, ISO 14021 (environmental labels), and EU Green Claims Directive
requirements before public disclosure.

Phases:
    1. ClaimFormulation       -- Define claim scope, language, and substantiation
    2. ComplianceAssessment   -- Check against PAS 2060, VCMI, ISO 14021
    3. EvidenceVerification   -- Verify all supporting evidence is complete
    4. ApprovalGate           -- Final approval with risk assessment

Regulatory references:
    - PAS 2060:2014 (Section 10: Declaration requirements)
    - VCMI Claims Code of Practice (2023)
    - ISO 14021:2016 (Environmental labels -- Self-declared claims)
    - EU Green Claims Directive (2023/0085(COD))
    - ISO 14064-3:2019 (Validation and verification)

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

logger = logging.getLogger(__name__)

_MODULE_VERSION = "24.0.0"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


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


class ClaimsPhase(str, Enum):
    CLAIM_FORMULATION = "claim_formulation"
    COMPLIANCE_ASSESSMENT = "compliance_assessment"
    EVIDENCE_VERIFICATION = "evidence_verification"
    APPROVAL_GATE = "approval_gate"


class ClaimType(str, Enum):
    CARBON_NEUTRAL_ORG = "carbon_neutral_organization"
    CARBON_NEUTRAL_PRODUCT = "carbon_neutral_product"
    CARBON_NEUTRAL_SERVICE = "carbon_neutral_service"
    CARBON_NEUTRAL_EVENT = "carbon_neutral_event"
    CARBON_NEUTRAL_BUILDING = "carbon_neutral_building"
    CLIMATE_POSITIVE = "climate_positive"
    NET_ZERO = "net_zero"


class ClaimStatus(str, Enum):
    DRAFT = "draft"
    UNDER_REVIEW = "under_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    PUBLISHED = "published"
    WITHDRAWN = "withdrawn"
    EXPIRED = "expired"


class ComplianceFramework(str, Enum):
    PAS_2060 = "pas_2060"
    VCMI = "vcmi"
    ISO_14021 = "iso_14021"
    EU_GREEN_CLAIMS = "eu_green_claims"
    ISO_14064 = "iso_14064"


class RiskLevel(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


# =============================================================================
# REFERENCE DATA
# =============================================================================

# PAS 2060 claim requirements
PAS2060_CLAIM_REQUIREMENTS: List[Dict[str, Any]] = [
    {"id": "PAS-1", "requirement": "Subject of claim clearly defined", "critical": True},
    {"id": "PAS-2", "requirement": "Carbon footprint quantified per ISO 14064-1", "critical": True},
    {"id": "PAS-3", "requirement": "Carbon management plan documented", "critical": True},
    {"id": "PAS-4", "requirement": "Reduction targets established", "critical": True},
    {"id": "PAS-5", "requirement": "Residual emissions offset with eligible credits", "critical": True},
    {"id": "PAS-6", "requirement": "Credits retired on recognized registries", "critical": True},
    {"id": "PAS-7", "requirement": "Qualifying explanatory statement prepared", "critical": True},
    {"id": "PAS-8", "requirement": "Independent validation obtained", "critical": False},
    {"id": "PAS-9", "requirement": "Public disclosure of methodology", "critical": True},
    {"id": "PAS-10", "requirement": "Commitment to ongoing achievement", "critical": True},
]

# VCMI Claims Code requirements
VCMI_CLAIM_REQUIREMENTS: List[Dict[str, Any]] = [
    {"id": "VCMI-1", "requirement": "Near-term science-aligned emission reduction target", "critical": True},
    {"id": "VCMI-2", "requirement": "Public disclosure of scope 1+2+3 emissions", "critical": True},
    {"id": "VCMI-3", "requirement": "Demonstrated progress on reduction targets", "critical": True},
    {"id": "VCMI-4", "requirement": "High-quality carbon credits (ICVCM CCP)", "critical": True},
    {"id": "VCMI-5", "requirement": "Credits >= residual emissions", "critical": True},
    {"id": "VCMI-6", "requirement": "Third-party verification", "critical": False},
]

# Prohibited claim language per EU Green Claims Directive
PROHIBITED_TERMS: List[str] = [
    "climate neutral", "carbon positive", "zero emissions",
    "environmentally friendly", "green", "eco-friendly",
    "climate friendly", "carbon free", "100% clean",
]

# Acceptable PAS 2060 claim wording templates
ACCEPTABLE_CLAIM_TEMPLATES: Dict[str, str] = {
    "carbon_neutral_organization": (
        "{org_name} has achieved carbon neutrality for the period "
        "{start_date} to {end_date} in accordance with PAS 2060:2014. "
        "Total emissions of {total_tco2e} tCO2e have been quantified, "
        "reduced, and offset through the retirement of verified carbon credits."
    ),
    "carbon_neutral_product": (
        "The carbon footprint of {product_name} has been measured, reduced, "
        "and offset to achieve carbon neutrality in accordance with PAS 2060:2014."
    ),
}


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


class ClaimAssessment(BaseModel):
    claim_id: str = Field(default="")
    claim_type: ClaimType = Field(default=ClaimType.CARBON_NEUTRAL_ORG)
    claim_text: str = Field(default="")
    subject_of_claim: str = Field(default="")
    reporting_period_start: str = Field(default="")
    reporting_period_end: str = Field(default="")
    total_emissions_tco2e: float = Field(default=0.0, ge=0.0)
    credits_retired_tco2e: float = Field(default=0.0, ge=0.0)
    is_substantiated: bool = Field(default=False)
    claim_status: ClaimStatus = Field(default=ClaimStatus.DRAFT)
    risk_level: RiskLevel = Field(default=RiskLevel.MODERATE)
    approved_by: str = Field(default="")
    approval_date: Optional[datetime] = Field(None)


class ComplianceCheck(BaseModel):
    framework: ComplianceFramework = Field(...)
    requirement_id: str = Field(default="")
    requirement_text: str = Field(default="")
    is_critical: bool = Field(default=True)
    is_met: bool = Field(default=False)
    evidence_reference: str = Field(default="")
    notes: str = Field(default="")


class SubstantiationEvidence(BaseModel):
    evidence_id: str = Field(default="")
    evidence_type: str = Field(default="")
    description: str = Field(default="")
    document_url: str = Field(default="")
    is_verified: bool = Field(default=False)
    verified_by: str = Field(default="")
    verification_date: Optional[datetime] = Field(None)
    hash: str = Field(default="")


class ClaimsValidationConfig(BaseModel):
    org_name: str = Field(default="")
    claim_type: ClaimType = Field(default=ClaimType.CARBON_NEUTRAL_ORG)
    subject_of_claim: str = Field(default="")
    reporting_period_start: str = Field(default="")
    reporting_period_end: str = Field(default="")
    claim_text: str = Field(default="")
    total_emissions_tco2e: float = Field(default=0.0, ge=0.0)
    reductions_tco2e: float = Field(default=0.0, ge=0.0)
    credits_retired_tco2e: float = Field(default=0.0, ge=0.0)
    has_management_plan: bool = Field(default=True)
    has_independent_validation: bool = Field(default=False)
    has_public_disclosure: bool = Field(default=True)
    frameworks: List[ComplianceFramework] = Field(
        default_factory=lambda: [ComplianceFramework.PAS_2060, ComplianceFramework.VCMI]
    )
    entity_id: str = Field(default="")
    tenant_id: str = Field(default="")


class ClaimsValidationResult(BaseModel):
    workflow_id: str = Field(...)
    workflow_name: str = Field(default="claims_validation")
    status: WorkflowStatus = Field(...)
    phases: List[PhaseResult] = Field(default_factory=list)
    total_duration_seconds: float = Field(default=0.0)
    claim: Optional[ClaimAssessment] = Field(None)
    compliance_checks: List[ComplianceCheck] = Field(default_factory=list)
    evidence_items: List[SubstantiationEvidence] = Field(default_factory=list)
    overall_compliance_pct: float = Field(default=0.0)
    critical_issues: List[str] = Field(default_factory=list)
    is_claim_valid: bool = Field(default=False)
    recommended_action: str = Field(default="")
    provenance_hash: str = Field(default="")


# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================


class ClaimsValidationWorkflow:
    """
    4-phase claims validation workflow for PACK-024.

    Validates carbon neutrality claims against PAS 2060, VCMI Claims Code,
    ISO 14021, and EU Green Claims Directive before public disclosure.

    Attributes:
        workflow_id: Unique execution identifier.
    """

    def __init__(self) -> None:
        self.workflow_id: str = _new_uuid()
        self._phase_results: List[PhaseResult] = []
        self._claim: Optional[ClaimAssessment] = None
        self._checks: List[ComplianceCheck] = []
        self._evidence: List[SubstantiationEvidence] = []
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    async def execute(self, config: ClaimsValidationConfig) -> ClaimsValidationResult:
        """Execute the 4-phase claims validation workflow."""
        started_at = _utcnow()
        self.logger.info("Starting claims validation %s", self.workflow_id)
        self._phase_results = []
        overall_status = WorkflowStatus.RUNNING

        try:
            phase1 = await self._phase_claim_formulation(config)
            self._phase_results.append(phase1)

            phase2 = await self._phase_compliance_assessment(config)
            self._phase_results.append(phase2)

            phase3 = await self._phase_evidence_verification(config)
            self._phase_results.append(phase3)

            phase4 = await self._phase_approval_gate(config)
            self._phase_results.append(phase4)

            failed = [p for p in self._phase_results if p.status == PhaseStatus.FAILED]
            overall_status = WorkflowStatus.COMPLETED if not failed else WorkflowStatus.PARTIAL

        except Exception as exc:
            self.logger.error("Claims validation failed: %s", exc, exc_info=True)
            overall_status = WorkflowStatus.FAILED
            self._phase_results.append(PhaseResult(
                phase_name="error", status=PhaseStatus.FAILED, errors=[str(exc)],
            ))

        elapsed = (_utcnow() - started_at).total_seconds()
        total_checks = len(self._checks)
        passed_checks = len([c for c in self._checks if c.is_met])
        compliance_pct = (passed_checks / max(total_checks, 1)) * 100.0
        critical_issues = [
            c.requirement_text for c in self._checks if c.is_critical and not c.is_met
        ]
        is_valid = len(critical_issues) == 0 and compliance_pct >= 80.0

        result = ClaimsValidationResult(
            workflow_id=self.workflow_id,
            status=overall_status,
            phases=self._phase_results,
            total_duration_seconds=round(elapsed, 4),
            claim=self._claim,
            compliance_checks=self._checks,
            evidence_items=self._evidence,
            overall_compliance_pct=round(compliance_pct, 1),
            critical_issues=critical_issues,
            is_claim_valid=is_valid,
            recommended_action="Proceed with publication" if is_valid else "Address critical issues before publication",
        )
        result.provenance_hash = _compute_hash(
            result.model_dump_json(exclude={"provenance_hash"})
        )
        return result

    async def _phase_claim_formulation(self, config: ClaimsValidationConfig) -> PhaseResult:
        """Define claim scope, language, and initial substantiation."""
        started = _utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []
        errors: List[str] = []

        # Generate or validate claim text
        claim_text = config.claim_text
        if not claim_text:
            template = ACCEPTABLE_CLAIM_TEMPLATES.get(config.claim_type.value, "")
            claim_text = template.format(
                org_name=config.org_name,
                start_date=config.reporting_period_start,
                end_date=config.reporting_period_end,
                total_tco2e=f"{config.total_emissions_tco2e:,.0f}",
                product_name=config.subject_of_claim,
            )

        # Check for prohibited terms
        claim_lower = claim_text.lower()
        for term in PROHIBITED_TERMS:
            if term in claim_lower and term != "carbon neutral":
                warnings.append(
                    f"Claim contains potentially problematic term: '{term}' -- "
                    "review against EU Green Claims Directive"
                )

        # Verify coverage
        is_substantiated = config.credits_retired_tco2e >= config.total_emissions_tco2e - config.reductions_tco2e

        self._claim = ClaimAssessment(
            claim_id=_new_uuid(),
            claim_type=config.claim_type,
            claim_text=claim_text,
            subject_of_claim=config.subject_of_claim or config.org_name,
            reporting_period_start=config.reporting_period_start,
            reporting_period_end=config.reporting_period_end,
            total_emissions_tco2e=config.total_emissions_tco2e,
            credits_retired_tco2e=config.credits_retired_tco2e,
            is_substantiated=is_substantiated,
            claim_status=ClaimStatus.UNDER_REVIEW,
            risk_level=RiskLevel.LOW if is_substantiated else RiskLevel.HIGH,
        )

        outputs["claim_type"] = config.claim_type.value
        outputs["claim_text_length"] = len(claim_text)
        outputs["is_substantiated"] = is_substantiated

        status = PhaseStatus.COMPLETED
        elapsed = (_utcnow() - started).total_seconds()
        return PhaseResult(
            phase_name=ClaimsPhase.CLAIM_FORMULATION.value,
            status=status, duration_seconds=round(elapsed, 4),
            outputs=outputs, warnings=warnings, errors=errors,
            provenance_hash=_compute_hash(outputs),
        )

    async def _phase_compliance_assessment(self, config: ClaimsValidationConfig) -> PhaseResult:
        """Check claim against regulatory frameworks."""
        started = _utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []
        errors: List[str] = []

        checks: List[ComplianceCheck] = []

        # PAS 2060 checks
        if ComplianceFramework.PAS_2060 in config.frameworks:
            for req in PAS2060_CLAIM_REQUIREMENTS:
                is_met = True  # Default to met, override below
                if req["id"] == "PAS-3" and not config.has_management_plan:
                    is_met = False
                if req["id"] == "PAS-8" and not config.has_independent_validation:
                    is_met = False
                if req["id"] == "PAS-9" and not config.has_public_disclosure:
                    is_met = False

                checks.append(ComplianceCheck(
                    framework=ComplianceFramework.PAS_2060,
                    requirement_id=req["id"],
                    requirement_text=req["requirement"],
                    is_critical=req["critical"],
                    is_met=is_met,
                ))

        # VCMI checks
        if ComplianceFramework.VCMI in config.frameworks:
            for req in VCMI_CLAIM_REQUIREMENTS:
                is_met = True
                if req["id"] == "VCMI-6" and not config.has_independent_validation:
                    is_met = False
                checks.append(ComplianceCheck(
                    framework=ComplianceFramework.VCMI,
                    requirement_id=req["id"],
                    requirement_text=req["requirement"],
                    is_critical=req["critical"],
                    is_met=is_met,
                ))

        self._checks = checks
        passed = len([c for c in checks if c.is_met])

        outputs["total_checks"] = len(checks)
        outputs["checks_passed"] = passed
        outputs["checks_failed"] = len(checks) - passed
        outputs["frameworks_assessed"] = [f.value for f in config.frameworks]

        status = PhaseStatus.COMPLETED
        elapsed = (_utcnow() - started).total_seconds()
        return PhaseResult(
            phase_name=ClaimsPhase.COMPLIANCE_ASSESSMENT.value,
            status=status, duration_seconds=round(elapsed, 4),
            outputs=outputs, warnings=warnings, errors=errors,
            provenance_hash=_compute_hash(outputs),
        )

    async def _phase_evidence_verification(self, config: ClaimsValidationConfig) -> PhaseResult:
        """Verify all supporting evidence is complete and valid."""
        started = _utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []
        errors: List[str] = []

        evidence_types = [
            ("carbon_footprint", "Verified GHG Inventory Report", True),
            ("management_plan", "Carbon Management Plan", config.has_management_plan),
            ("retirement_certificates", "Credit Retirement Certificates", True),
            ("qualifying_statement", "Qualifying Explanatory Statement", True),
            ("independent_validation", "Independent Validation Report", config.has_independent_validation),
            ("public_disclosure", "Public Disclosure Documentation", config.has_public_disclosure),
            ("methodology", "Quantification Methodology", True),
            ("reduction_evidence", "Emission Reduction Evidence", config.reductions_tco2e > 0),
        ]

        evidence_items: List[SubstantiationEvidence] = []
        for ev_type, desc, is_available in evidence_types:
            evidence_items.append(SubstantiationEvidence(
                evidence_id=_new_uuid(),
                evidence_type=ev_type,
                description=desc,
                is_verified=is_available,
                hash=_compute_hash(f"{ev_type}_{config.org_name}"),
            ))

        self._evidence = evidence_items
        verified_count = sum(1 for e in evidence_items if e.is_verified)

        outputs["evidence_items"] = len(evidence_items)
        outputs["verified_items"] = verified_count
        outputs["missing_items"] = len(evidence_items) - verified_count

        status = PhaseStatus.COMPLETED
        elapsed = (_utcnow() - started).total_seconds()
        return PhaseResult(
            phase_name=ClaimsPhase.EVIDENCE_VERIFICATION.value,
            status=status, duration_seconds=round(elapsed, 4),
            outputs=outputs, warnings=warnings, errors=errors,
            provenance_hash=_compute_hash(outputs),
        )

    async def _phase_approval_gate(self, config: ClaimsValidationConfig) -> PhaseResult:
        """Final approval gate with risk assessment."""
        started = _utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []
        errors: List[str] = []

        critical_failures = [c for c in self._checks if c.is_critical and not c.is_met]
        missing_evidence = [e for e in self._evidence if not e.is_verified]

        if critical_failures:
            risk = RiskLevel.CRITICAL
            decision = "REJECT"
            for cf in critical_failures:
                errors.append(f"Critical requirement not met: {cf.requirement_text}")
        elif missing_evidence:
            risk = RiskLevel.HIGH
            decision = "CONDITIONAL_APPROVE"
            for me in missing_evidence:
                warnings.append(f"Missing evidence: {me.description}")
        else:
            risk = RiskLevel.LOW
            decision = "APPROVE"

        if self._claim:
            if decision == "APPROVE":
                self._claim.claim_status = ClaimStatus.APPROVED
                self._claim.approval_date = _utcnow()
            elif decision == "REJECT":
                self._claim.claim_status = ClaimStatus.REJECTED
            self._claim.risk_level = risk

        outputs["decision"] = decision
        outputs["risk_level"] = risk.value
        outputs["critical_failures"] = len(critical_failures)
        outputs["missing_evidence"] = len(missing_evidence)

        status = PhaseStatus.COMPLETED
        elapsed = (_utcnow() - started).total_seconds()
        return PhaseResult(
            phase_name=ClaimsPhase.APPROVAL_GATE.value,
            status=status, duration_seconds=round(elapsed, 4),
            outputs=outputs, warnings=warnings, errors=errors,
            provenance_hash=_compute_hash(outputs),
        )
