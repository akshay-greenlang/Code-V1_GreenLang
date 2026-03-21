# -*- coding: utf-8 -*-
"""
VCMI Certification Workflow
==================================

4-phase workflow for evaluating eligibility for VCMI (Voluntary Carbon
Markets Integrity Initiative) Claims within PACK-022 Net-Zero
Acceleration Pack.  The workflow collects evidence for foundational
criteria, evaluates each criterion, determines the eligible claim
tier, and generates a certification report.

Phases:
    1. EvidenceCollection    -- Collect evidence for 4 foundational criteria
                                 (targets, progress, disclosure, credit quality)
    2. CriteriaCheck         -- Evaluate each criterion with scoring
    3. ClaimValidation       -- Determine eligible claim tier
                                 (Silver/Gold/Platinum), check greenwashing risks
    4. CertificationReport   -- Generate certification report with evidence
                                 summary, gaps, recommendations

Regulatory references:
    - VCMI Claims Code of Practice (June 2023)
    - VCMI Monitoring, Reporting and Assurance Framework
    - SBTi Net-Zero Standard v1.2 (for target criterion)
    - IC-VCM Core Carbon Principles (for credit quality)
    - Oxford Principles for Net Zero Aligned Carbon Offsetting

Author: GreenLang Team
Version: 22.0.0
"""

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

_MODULE_VERSION = "22.0.0"


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


class PhaseStatus(str, Enum):
    """Status of a single workflow phase."""

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


class ClaimTier(str, Enum):
    """VCMI claim tiers."""

    SILVER = "silver"
    GOLD = "gold"
    PLATINUM = "platinum"
    NOT_ELIGIBLE = "not_eligible"


class CriterionStatus(str, Enum):
    """Status of a single foundational criterion."""

    PASS = "pass"
    PARTIAL = "partial"
    FAIL = "fail"
    NOT_ASSESSED = "not_assessed"


class GreenwashingRisk(str, Enum):
    """Greenwashing risk levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


# =============================================================================
# VCMI REFERENCE DATA (Zero-Hallucination, from VCMI Claims Code)
# =============================================================================

# VCMI foundational criteria
VCMI_CRITERIA: Dict[str, Dict[str, Any]] = {
    "FC1_targets": {
        "name": "Science-Based Targets",
        "description": "Company has set or committed to science-based emission reduction targets",
        "requirements": [
            "Near-term SBTi-aligned targets covering Scope 1+2 (and Scope 3 if >40% of total)",
            "Long-term net-zero target with SBTi validation or commitment",
            "Targets publicly disclosed",
        ],
        "weight": 30,
    },
    "FC2_progress": {
        "name": "Emission Reduction Progress",
        "description": "Company is making demonstrable progress toward targets",
        "requirements": [
            "On track with near-term target pathway",
            "Year-over-year emissions reduction demonstrated",
            "Progress publicly reported with third-party verification",
        ],
        "weight": 30,
    },
    "FC3_disclosure": {
        "name": "Public Disclosure",
        "description": "Full transparent disclosure of emissions and climate strategy",
        "requirements": [
            "Annual GHG inventory (Scope 1+2+3) publicly disclosed",
            "TCFD-aligned climate disclosure",
            "CDP submission (score B or higher preferred)",
        ],
        "weight": 20,
    },
    "FC4_credit_quality": {
        "name": "Carbon Credit Quality",
        "description": "Credits meet high-integrity quality standards",
        "requirements": [
            "Credits meet IC-VCM Core Carbon Principles",
            "No double counting (corresponding adjustments where applicable)",
            "Vintage within 5 years",
            "Increasing share of removal credits over time",
        ],
        "weight": 20,
    },
}

# Claim tier requirements (from VCMI Claims Code)
CLAIM_TIER_REQUIREMENTS: Dict[str, Dict[str, Any]] = {
    "silver": {
        "min_score": 60,
        "all_criteria_pass": False,
        "min_credit_coverage_pct": 20,
        "max_credit_coverage_pct": 100,
        "description": "Silver: Company is on track and using credits for 20-100% of residual",
    },
    "gold": {
        "min_score": 75,
        "all_criteria_pass": True,
        "min_credit_coverage_pct": 60,
        "max_credit_coverage_pct": 100,
        "description": "Gold: Company meets all criteria and covers 60-100% with credits",
    },
    "platinum": {
        "min_score": 90,
        "all_criteria_pass": True,
        "min_credit_coverage_pct": 100,
        "max_credit_coverage_pct": 100,
        "description": "Platinum: Company exceeds all criteria and neutralises 100% of residual",
    },
}


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


class EvidenceItem(BaseModel):
    """A single piece of evidence for VCMI assessment."""

    evidence_id: str = Field(default="")
    criterion: str = Field(default="", description="FC1, FC2, FC3, or FC4")
    description: str = Field(default="")
    source: str = Field(default="", description="Source document or system")
    verified: bool = Field(default=False)
    verification_body: str = Field(default="")
    score_contribution: float = Field(default=0.0, ge=0.0, le=100.0)


class CreditPortfolioItem(BaseModel):
    """A carbon credit in the portfolio."""

    credit_id: str = Field(default="")
    standard: str = Field(default="", description="e.g., Verra VCS, Gold Standard")
    project_type: str = Field(default="", description="avoidance or removal")
    volume_tco2e: float = Field(default=0.0, ge=0.0)
    vintage_year: int = Field(default=2024, ge=2015, le=2035)
    price_per_tco2e_usd: float = Field(default=0.0, ge=0.0)
    icvcm_compliant: bool = Field(default=False)
    corresponding_adjustment: bool = Field(default=False)
    is_removal: bool = Field(default=False)


class CriterionAssessment(BaseModel):
    """Assessment result for a single VCMI criterion."""

    criterion_id: str = Field(default="")
    criterion_name: str = Field(default="")
    status: CriterionStatus = Field(default=CriterionStatus.NOT_ASSESSED)
    score: float = Field(default=0.0, ge=0.0, le=100.0)
    max_score: float = Field(default=0.0)
    evidence_items: List[EvidenceItem] = Field(default_factory=list)
    findings: List[str] = Field(default_factory=list)
    gaps: List[str] = Field(default_factory=list)


class ClaimDetermination(BaseModel):
    """VCMI claim tier determination."""

    eligible_tier: ClaimTier = Field(default=ClaimTier.NOT_ELIGIBLE)
    target_tier: ClaimTier = Field(default=ClaimTier.SILVER)
    total_score: float = Field(default=0.0, ge=0.0, le=100.0)
    all_criteria_pass: bool = Field(default=False)
    credit_coverage_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    greenwashing_risk: GreenwashingRisk = Field(default=GreenwashingRisk.MEDIUM)
    greenwashing_flags: List[str] = Field(default_factory=list)
    tier_gap_analysis: str = Field(default="")


class VCMICertificationConfig(BaseModel):
    """Configuration for the VCMI certification workflow."""

    # Targets evidence
    has_sbti_target: bool = Field(default=False)
    sbti_validated: bool = Field(default=False)
    near_term_target_year: int = Field(default=2030)
    near_term_reduction_pct: float = Field(default=0.0)
    long_term_target_year: int = Field(default=2050)
    long_term_reduction_pct: float = Field(default=0.0)

    # Progress evidence
    reduction_progress_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    on_track_with_pathway: bool = Field(default=False)
    years_of_reduction: int = Field(default=0, ge=0)
    third_party_verified: bool = Field(default=False)

    # Disclosure evidence
    ghg_inventory_public: bool = Field(default=False)
    tcfd_disclosure: bool = Field(default=False)
    cdp_submission: bool = Field(default=False)
    cdp_score: str = Field(default="", description="A, A-, B, B-, C, C-, D, D-")

    # Credit portfolio
    credit_portfolio: List[CreditPortfolioItem] = Field(default_factory=list)
    residual_emissions_tco2e: float = Field(default=0.0, ge=0.0)

    # Target tier
    target_tier: str = Field(default="silver")

    entity_id: str = Field(default="")
    tenant_id: str = Field(default="")

    @field_validator("target_tier")
    @classmethod
    def _validate_tier(cls, v: str) -> str:
        if v not in {"silver", "gold", "platinum"}:
            raise ValueError("target_tier must be silver, gold, or platinum")
        return v


class VCMICertificationResult(BaseModel):
    """Complete result from the VCMI certification workflow."""

    workflow_id: str = Field(..., description="Unique execution ID")
    workflow_name: str = Field(default="vcmi_certification")
    status: WorkflowStatus = Field(...)
    phases: List[PhaseResult] = Field(default_factory=list)
    total_duration_seconds: float = Field(default=0.0)
    criterion_assessments: List[CriterionAssessment] = Field(default_factory=list)
    claim_determination: ClaimDetermination = Field(default_factory=ClaimDetermination)
    recommendations: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")


# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================


class VCMICertificationWorkflow:
    """
    4-phase VCMI certification workflow.

    Collects evidence for foundational criteria, evaluates each criterion,
    determines eligible claim tier, and generates a certification report
    with gap analysis and recommendations.

    Zero-hallucination: all scoring thresholds, tier requirements, and
    criterion definitions come from the VCMI Claims Code deterministic
    reference data.  No LLM calls in the numeric computation path.

    Attributes:
        workflow_id: Unique execution identifier.

    Example:
        >>> wf = VCMICertificationWorkflow()
        >>> config = VCMICertificationConfig(has_sbti_target=True, ...)
        >>> result = await wf.execute(config)
        >>> print(result.claim_determination.eligible_tier)
    """

    def __init__(self) -> None:
        """Initialise VCMICertificationWorkflow."""
        self.workflow_id: str = _new_uuid()
        self._phase_results: List[PhaseResult] = []
        self._assessments: List[CriterionAssessment] = []
        self._determination: ClaimDetermination = ClaimDetermination()
        self._recommendations: List[str] = []
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    async def execute(self, config: VCMICertificationConfig) -> VCMICertificationResult:
        """
        Execute the 4-phase VCMI certification workflow.

        Args:
            config: VCMI certification configuration with evidence for
                targets, progress, disclosure, and credit portfolio.

        Returns:
            VCMICertificationResult with criterion assessments, claim
            determination, and recommendations.
        """
        started_at = _utcnow()
        self.logger.info(
            "Starting VCMI certification workflow %s, target_tier=%s",
            self.workflow_id, config.target_tier,
        )
        self._phase_results = []
        overall_status = WorkflowStatus.RUNNING

        try:
            phase1 = await self._phase_evidence_collection(config)
            self._phase_results.append(phase1)

            phase2 = await self._phase_criteria_check(config)
            self._phase_results.append(phase2)

            phase3 = await self._phase_claim_validation(config)
            self._phase_results.append(phase3)

            phase4 = await self._phase_certification_report(config)
            self._phase_results.append(phase4)

            failed = [p for p in self._phase_results if p.status == PhaseStatus.FAILED]
            overall_status = WorkflowStatus.COMPLETED if not failed else WorkflowStatus.PARTIAL

        except Exception as exc:
            self.logger.error("VCMI certification workflow failed: %s", exc, exc_info=True)
            overall_status = WorkflowStatus.FAILED
            self._phase_results.append(PhaseResult(
                phase_name="error", status=PhaseStatus.FAILED, errors=[str(exc)],
            ))

        elapsed = (_utcnow() - started_at).total_seconds()
        result = VCMICertificationResult(
            workflow_id=self.workflow_id,
            status=overall_status,
            phases=self._phase_results,
            total_duration_seconds=round(elapsed, 4),
            criterion_assessments=self._assessments,
            claim_determination=self._determination,
            recommendations=self._recommendations,
        )
        result.provenance_hash = _compute_hash(
            result.model_dump_json(exclude={"provenance_hash"})
        )
        self.logger.info(
            "VCMI certification workflow %s completed in %.2fs, tier=%s",
            self.workflow_id, elapsed,
            self._determination.eligible_tier.value,
        )
        return result

    # -------------------------------------------------------------------------
    # Phase 1: Evidence Collection
    # -------------------------------------------------------------------------

    async def _phase_evidence_collection(self, config: VCMICertificationConfig) -> PhaseResult:
        """Collect evidence for 4 foundational criteria."""
        started = _utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        evidence_map: Dict[str, List[EvidenceItem]] = {
            "FC1_targets": [],
            "FC2_progress": [],
            "FC3_disclosure": [],
            "FC4_credit_quality": [],
        }

        # FC1: Targets evidence
        if config.has_sbti_target:
            evidence_map["FC1_targets"].append(EvidenceItem(
                evidence_id="FC1-E1", criterion="FC1_targets",
                description=f"SBTi target set: near-term {config.near_term_reduction_pct:.1f}% by {config.near_term_target_year}",
                source="SBTi target registry",
                verified=config.sbti_validated,
                score_contribution=20.0 if config.sbti_validated else 12.0,
            ))
        if config.long_term_reduction_pct >= 90.0:
            evidence_map["FC1_targets"].append(EvidenceItem(
                evidence_id="FC1-E2", criterion="FC1_targets",
                description=f"Long-term net-zero target: {config.long_term_reduction_pct:.1f}% by {config.long_term_target_year}",
                source="Corporate disclosure",
                verified=config.sbti_validated,
                score_contribution=10.0,
            ))

        # FC2: Progress evidence
        if config.reduction_progress_pct > 0:
            evidence_map["FC2_progress"].append(EvidenceItem(
                evidence_id="FC2-E1", criterion="FC2_progress",
                description=f"Emission reduction progress: {config.reduction_progress_pct:.1f}%",
                source="Annual sustainability report",
                verified=config.third_party_verified,
                score_contribution=min(config.reduction_progress_pct / 2, 15.0),
            ))
        if config.on_track_with_pathway:
            evidence_map["FC2_progress"].append(EvidenceItem(
                evidence_id="FC2-E2", criterion="FC2_progress",
                description="On track with target pathway",
                source="Progress review",
                verified=config.third_party_verified,
                score_contribution=10.0,
            ))
        if config.third_party_verified:
            evidence_map["FC2_progress"].append(EvidenceItem(
                evidence_id="FC2-E3", criterion="FC2_progress",
                description="Third-party verification of emissions data",
                source="Verification statement",
                verified=True,
                score_contribution=5.0,
            ))

        # FC3: Disclosure evidence
        if config.ghg_inventory_public:
            evidence_map["FC3_disclosure"].append(EvidenceItem(
                evidence_id="FC3-E1", criterion="FC3_disclosure",
                description="Public GHG inventory (Scope 1+2+3)",
                source="Annual report / website",
                verified=True,
                score_contribution=8.0,
            ))
        if config.tcfd_disclosure:
            evidence_map["FC3_disclosure"].append(EvidenceItem(
                evidence_id="FC3-E2", criterion="FC3_disclosure",
                description="TCFD-aligned climate disclosure",
                source="TCFD report",
                verified=True,
                score_contribution=6.0,
            ))
        if config.cdp_submission:
            cdp_score_val = self._cdp_score_to_value(config.cdp_score)
            evidence_map["FC3_disclosure"].append(EvidenceItem(
                evidence_id="FC3-E3", criterion="FC3_disclosure",
                description=f"CDP submission (score: {config.cdp_score or 'submitted'})",
                source="CDP",
                verified=True,
                score_contribution=cdp_score_val,
            ))

        # FC4: Credit quality evidence
        if config.credit_portfolio:
            icvcm_count = sum(1 for c in config.credit_portfolio if c.icvcm_compliant)
            total_credits = len(config.credit_portfolio)
            icvcm_pct = (icvcm_count / total_credits * 100.0) if total_credits > 0 else 0.0
            evidence_map["FC4_credit_quality"].append(EvidenceItem(
                evidence_id="FC4-E1", criterion="FC4_credit_quality",
                description=f"IC-VCM compliant credits: {icvcm_pct:.0f}% ({icvcm_count}/{total_credits})",
                source="Credit registry",
                verified=True,
                score_contribution=min(icvcm_pct / 5, 10.0),
            ))

            removal_vol = sum(c.volume_tco2e for c in config.credit_portfolio if c.is_removal)
            total_vol = sum(c.volume_tco2e for c in config.credit_portfolio)
            removal_pct = (removal_vol / total_vol * 100.0) if total_vol > 0 else 0.0
            evidence_map["FC4_credit_quality"].append(EvidenceItem(
                evidence_id="FC4-E2", criterion="FC4_credit_quality",
                description=f"Removal credits share: {removal_pct:.0f}%",
                source="Credit portfolio analysis",
                verified=True,
                score_contribution=min(removal_pct / 10, 5.0),
            ))

            # Vintage check
            current_year = _utcnow().year
            recent_count = sum(1 for c in config.credit_portfolio if current_year - c.vintage_year <= 5)
            vintage_pct = (recent_count / total_credits * 100.0) if total_credits > 0 else 0.0
            evidence_map["FC4_credit_quality"].append(EvidenceItem(
                evidence_id="FC4-E3", criterion="FC4_credit_quality",
                description=f"Credits with vintage within 5 years: {vintage_pct:.0f}%",
                source="Credit registry",
                verified=True,
                score_contribution=min(vintage_pct / 20, 5.0),
            ))

        total_evidence = sum(len(items) for items in evidence_map.values())
        outputs["total_evidence_items"] = total_evidence
        for criterion, items in evidence_map.items():
            outputs[f"{criterion}_evidence_count"] = len(items)

        # Store for later phases
        self._evidence_map = evidence_map

        elapsed = (_utcnow() - started).total_seconds()
        self.logger.info("Evidence collection: %d items collected", total_evidence)
        return PhaseResult(
            phase_name="evidence_collection",
            status=PhaseStatus.COMPLETED,
            duration_seconds=round(elapsed, 4),
            outputs=outputs,
            warnings=warnings,
            provenance_hash=_compute_hash(json.dumps(outputs, sort_keys=True, default=str)),
        )

    def _cdp_score_to_value(self, cdp_score: str) -> float:
        """Convert CDP letter score to a numeric evidence contribution."""
        score_map = {
            "A": 6.0, "A-": 5.5, "B": 5.0, "B-": 4.0,
            "C": 3.0, "C-": 2.0, "D": 1.0, "D-": 0.5,
        }
        return score_map.get(cdp_score.upper().strip(), 2.0) if cdp_score else 2.0

    # -------------------------------------------------------------------------
    # Phase 2: Criteria Check
    # -------------------------------------------------------------------------

    async def _phase_criteria_check(self, config: VCMICertificationConfig) -> PhaseResult:
        """Evaluate each criterion with scoring."""
        started = _utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        self._assessments = []
        evidence_map = getattr(self, "_evidence_map", {})

        for criterion_id, criterion_info in VCMI_CRITERIA.items():
            evidence = evidence_map.get(criterion_id, [])
            score = sum(e.score_contribution for e in evidence)
            max_score = criterion_info["weight"]
            score = min(score, max_score)

            # Determine status
            score_pct = (score / max_score * 100.0) if max_score > 0 else 0.0
            if score_pct >= 80:
                status = CriterionStatus.PASS
            elif score_pct >= 50:
                status = CriterionStatus.PARTIAL
            elif score_pct > 0:
                status = CriterionStatus.FAIL
            else:
                status = CriterionStatus.NOT_ASSESSED

            # Identify gaps
            gaps = self._identify_criterion_gaps(criterion_id, config, evidence)
            findings = [
                f"Score: {score:.1f}/{max_score} ({score_pct:.0f}%)",
                f"Evidence items: {len(evidence)}",
                f"Status: {status.value}",
            ]

            self._assessments.append(CriterionAssessment(
                criterion_id=criterion_id,
                criterion_name=criterion_info["name"],
                status=status,
                score=round(score, 2),
                max_score=max_score,
                evidence_items=evidence,
                findings=findings,
                gaps=gaps,
            ))

        for assessment in self._assessments:
            outputs[f"{assessment.criterion_id}_status"] = assessment.status.value
            outputs[f"{assessment.criterion_id}_score"] = assessment.score

        elapsed = (_utcnow() - started).total_seconds()
        self.logger.info("Criteria check: %d criteria evaluated", len(self._assessments))
        return PhaseResult(
            phase_name="criteria_check",
            status=PhaseStatus.COMPLETED,
            duration_seconds=round(elapsed, 4),
            outputs=outputs,
            warnings=warnings,
            provenance_hash=_compute_hash(json.dumps(outputs, sort_keys=True, default=str)),
        )

    def _identify_criterion_gaps(
        self, criterion_id: str, config: VCMICertificationConfig,
        evidence: List[EvidenceItem]
    ) -> List[str]:
        """Identify gaps for a specific criterion."""
        gaps: List[str] = []

        if criterion_id == "FC1_targets":
            if not config.has_sbti_target:
                gaps.append("No science-based target set. Commit to SBTi and set targets.")
            elif not config.sbti_validated:
                gaps.append("SBTi target not yet validated. Submit for validation.")
            if config.long_term_reduction_pct < 90.0:
                gaps.append(f"Long-term reduction ({config.long_term_reduction_pct:.1f}%) below 90% net-zero threshold.")

        elif criterion_id == "FC2_progress":
            if config.reduction_progress_pct < 5.0:
                gaps.append("Minimal emission reduction progress demonstrated.")
            if not config.on_track_with_pathway:
                gaps.append("Not on track with target pathway. Accelerate reduction actions.")
            if not config.third_party_verified:
                gaps.append("Emissions data not third-party verified. Engage verification body.")

        elif criterion_id == "FC3_disclosure":
            if not config.ghg_inventory_public:
                gaps.append("GHG inventory not publicly disclosed.")
            if not config.tcfd_disclosure:
                gaps.append("No TCFD-aligned climate disclosure. Implement TCFD framework.")
            if not config.cdp_submission:
                gaps.append("No CDP submission. Submit to CDP for credibility.")

        elif criterion_id == "FC4_credit_quality":
            if not config.credit_portfolio:
                gaps.append("No carbon credits in portfolio. Source high-quality credits.")
            else:
                icvcm_count = sum(1 for c in config.credit_portfolio if c.icvcm_compliant)
                if icvcm_count < len(config.credit_portfolio):
                    gaps.append(
                        f"Only {icvcm_count}/{len(config.credit_portfolio)} credits are IC-VCM compliant. "
                        "Transition to CCP-eligible credits."
                    )
                removal_count = sum(1 for c in config.credit_portfolio if c.is_removal)
                if removal_count == 0:
                    gaps.append("No removal credits. Increase share of carbon removal credits per Oxford Principles.")

        return gaps

    # -------------------------------------------------------------------------
    # Phase 3: Claim Validation
    # -------------------------------------------------------------------------

    async def _phase_claim_validation(self, config: VCMICertificationConfig) -> PhaseResult:
        """Determine eligible claim tier and check greenwashing risks."""
        started = _utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        total_score = sum(a.score for a in self._assessments)
        all_pass = all(a.status == CriterionStatus.PASS for a in self._assessments)

        # Calculate credit coverage
        total_credit_vol = sum(c.volume_tco2e for c in config.credit_portfolio)
        residual = config.residual_emissions_tco2e
        credit_coverage_pct = (total_credit_vol / residual * 100.0) if residual > 0 else 0.0
        credit_coverage_pct = min(credit_coverage_pct, 100.0)

        # Determine eligible tier
        eligible_tier = ClaimTier.NOT_ELIGIBLE

        if (total_score >= CLAIM_TIER_REQUIREMENTS["platinum"]["min_score"]
                and all_pass
                and credit_coverage_pct >= CLAIM_TIER_REQUIREMENTS["platinum"]["min_credit_coverage_pct"]):
            eligible_tier = ClaimTier.PLATINUM
        elif (total_score >= CLAIM_TIER_REQUIREMENTS["gold"]["min_score"]
              and all_pass
              and credit_coverage_pct >= CLAIM_TIER_REQUIREMENTS["gold"]["min_credit_coverage_pct"]):
            eligible_tier = ClaimTier.GOLD
        elif (total_score >= CLAIM_TIER_REQUIREMENTS["silver"]["min_score"]
              and credit_coverage_pct >= CLAIM_TIER_REQUIREMENTS["silver"]["min_credit_coverage_pct"]):
            eligible_tier = ClaimTier.SILVER

        # Greenwashing risk assessment
        gw_risk, gw_flags = self._assess_greenwashing_risk(config, total_score, credit_coverage_pct)

        # Gap analysis for target tier
        target_tier_enum = ClaimTier(config.target_tier)
        gap_analysis = self._tier_gap_analysis(
            eligible_tier, target_tier_enum, total_score, all_pass, credit_coverage_pct
        )

        self._determination = ClaimDetermination(
            eligible_tier=eligible_tier,
            target_tier=target_tier_enum,
            total_score=round(total_score, 2),
            all_criteria_pass=all_pass,
            credit_coverage_pct=round(credit_coverage_pct, 2),
            greenwashing_risk=gw_risk,
            greenwashing_flags=gw_flags,
            tier_gap_analysis=gap_analysis,
        )

        outputs["eligible_tier"] = eligible_tier.value
        outputs["target_tier"] = config.target_tier
        outputs["total_score"] = round(total_score, 2)
        outputs["all_criteria_pass"] = all_pass
        outputs["credit_coverage_pct"] = round(credit_coverage_pct, 2)
        outputs["greenwashing_risk"] = gw_risk.value
        outputs["greenwashing_flags"] = len(gw_flags)

        elapsed = (_utcnow() - started).total_seconds()
        self.logger.info("Claim validation: eligible=%s, score=%.1f, coverage=%.1f%%",
                         eligible_tier.value, total_score, credit_coverage_pct)
        return PhaseResult(
            phase_name="claim_validation",
            status=PhaseStatus.COMPLETED,
            duration_seconds=round(elapsed, 4),
            outputs=outputs,
            warnings=warnings,
            provenance_hash=_compute_hash(json.dumps(outputs, sort_keys=True, default=str)),
        )

    def _assess_greenwashing_risk(
        self, config: VCMICertificationConfig, score: float, coverage_pct: float
    ) -> tuple:
        """Assess greenwashing risk based on evidence and practices."""
        flags: List[str] = []

        # No targets but buying credits
        if not config.has_sbti_target and config.credit_portfolio:
            flags.append("Purchasing credits without science-based reduction targets")

        # Emissions increasing but claiming credits
        if config.reduction_progress_pct < 0:
            flags.append("Emissions are increasing while using carbon credits")

        # Only avoidance credits
        if config.credit_portfolio:
            all_avoidance = all(not c.is_removal for c in config.credit_portfolio)
            if all_avoidance:
                flags.append("Portfolio relies entirely on avoidance credits; consider removals")

        # No verification
        if not config.third_party_verified:
            flags.append("Emissions not independently verified")

        # Old vintages
        current_year = _utcnow().year
        if config.credit_portfolio:
            old_credits = [c for c in config.credit_portfolio if current_year - c.vintage_year > 5]
            if old_credits:
                flags.append(f"{len(old_credits)} credit(s) with vintage older than 5 years")

        # Determine risk level
        if len(flags) >= 3:
            risk = GreenwashingRisk.HIGH
        elif len(flags) >= 1:
            risk = GreenwashingRisk.MEDIUM
        else:
            risk = GreenwashingRisk.LOW

        return risk, flags

    def _tier_gap_analysis(
        self,
        eligible: ClaimTier,
        target: ClaimTier,
        score: float,
        all_pass: bool,
        coverage: float,
    ) -> str:
        """Generate gap analysis between eligible and target tier."""
        if eligible == target or (
            eligible == ClaimTier.PLATINUM or
            (eligible == ClaimTier.GOLD and target == ClaimTier.SILVER)
        ):
            return f"Target tier '{target.value}' is achievable. Currently eligible for '{eligible.value}'."

        target_reqs = CLAIM_TIER_REQUIREMENTS.get(target.value, {})
        gaps: List[str] = []

        min_score = target_reqs.get("min_score", 0)
        if score < min_score:
            gaps.append(f"Score gap: {score:.1f} vs {min_score} required (need +{min_score - score:.1f})")

        if target_reqs.get("all_criteria_pass") and not all_pass:
            gaps.append("Not all criteria pass; address failing criteria")

        min_coverage = target_reqs.get("min_credit_coverage_pct", 0)
        if coverage < min_coverage:
            gaps.append(f"Credit coverage gap: {coverage:.1f}% vs {min_coverage}% required")

        if gaps:
            return f"Gaps to '{target.value}' tier: " + "; ".join(gaps)
        return f"Close to '{target.value}' tier. Review detailed criterion scores."

    # -------------------------------------------------------------------------
    # Phase 4: Certification Report
    # -------------------------------------------------------------------------

    async def _phase_certification_report(self, config: VCMICertificationConfig) -> PhaseResult:
        """Generate certification report with recommendations."""
        started = _utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        self._recommendations = []

        # Recommendations based on gaps
        for assessment in self._assessments:
            for gap in assessment.gaps:
                self._recommendations.append(f"[{assessment.criterion_id}] {gap}")

        # Tier-specific recommendations
        if self._determination.eligible_tier == ClaimTier.NOT_ELIGIBLE:
            self._recommendations.insert(0,
                "Priority: Address foundational criteria gaps to reach Silver tier eligibility."
            )
        elif self._determination.eligible_tier == ClaimTier.SILVER:
            self._recommendations.insert(0,
                "To progress to Gold: Ensure all 4 criteria pass and increase credit coverage to 60%+."
            )
        elif self._determination.eligible_tier == ClaimTier.GOLD:
            self._recommendations.insert(0,
                "To progress to Platinum: Achieve 100% credit coverage with high-quality removal credits."
            )

        # Greenwashing mitigation
        if self._determination.greenwashing_risk in (GreenwashingRisk.MEDIUM, GreenwashingRisk.HIGH):
            for flag in self._determination.greenwashing_flags:
                self._recommendations.append(f"[Greenwashing risk] Mitigate: {flag}")

        outputs["recommendation_count"] = len(self._recommendations)
        outputs["eligible_tier"] = self._determination.eligible_tier.value
        outputs["greenwashing_risk"] = self._determination.greenwashing_risk.value
        outputs["total_gaps"] = sum(len(a.gaps) for a in self._assessments)

        elapsed = (_utcnow() - started).total_seconds()
        self.logger.info("Certification report: %d recommendations", len(self._recommendations))
        return PhaseResult(
            phase_name="certification_report",
            status=PhaseStatus.COMPLETED,
            duration_seconds=round(elapsed, 4),
            outputs=outputs,
            warnings=warnings,
            provenance_hash=_compute_hash(json.dumps(outputs, sort_keys=True, default=str)),
        )
