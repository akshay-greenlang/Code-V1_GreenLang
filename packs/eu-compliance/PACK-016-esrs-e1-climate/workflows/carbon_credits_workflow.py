# -*- coding: utf-8 -*-
"""
Carbon Credits Workflow
==============================

5-phase workflow for carbon credit/offset management per ESRS E1-7.
Implements credit registration, quality assessment, portfolio analysis,
SBTi compliance check, and report generation.

Phases:
    1. CreditRegistration     -- Register carbon credits and offsets
    2. QualityAssessment      -- Assess credit quality and standards
    3. PortfolioAnalysis      -- Analyze credit portfolio composition
    4. SBTiCheck              -- Verify SBTi treatment of credits
    5. ReportGeneration       -- Produce E1-7 disclosure data

Author: GreenLang Team
Version: 16.0.0
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
    """Phases of the carbon credits workflow."""
    CREDIT_REGISTRATION = "credit_registration"
    QUALITY_ASSESSMENT = "quality_assessment"
    PORTFOLIO_ANALYSIS = "portfolio_analysis"
    SBTI_CHECK = "sbti_check"
    REPORT_GENERATION = "report_generation"


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


class CreditType(str, Enum):
    """Carbon credit type."""
    AVOIDANCE = "avoidance"
    REMOVAL = "removal"
    HYBRID = "hybrid"


class CreditStandard(str, Enum):
    """Carbon credit certification standard."""
    VCS = "vcs"
    GOLD_STANDARD = "gold_standard"
    CDM = "cdm"
    ACR = "acr"
    CAR = "car"
    PURO_EARTH = "puro_earth"
    ISOMETRIC = "isometric"
    EU_ETS = "eu_ets"
    OTHER = "other"


class CreditStatus(str, Enum):
    """Credit lifecycle status."""
    ACTIVE = "active"
    RETIRED = "retired"
    CANCELLED = "cancelled"
    PENDING = "pending"
    EXPIRED = "expired"


class QualityTier(str, Enum):
    """Credit quality tier."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNASSESSED = "unassessed"


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


class CarbonCredit(BaseModel):
    """A carbon credit or offset record."""
    credit_id: str = Field(default_factory=lambda: f"cc-{_new_uuid()[:8]}")
    project_name: str = Field(..., description="Credit project name")
    credit_type: CreditType = Field(..., description="Avoidance or removal")
    standard: CreditStandard = Field(default=CreditStandard.VCS)
    vintage_year: int = Field(default=2024, ge=2000, le=2050)
    quantity_tco2e: float = Field(default=0.0, ge=0.0, description="Credits in tCO2e")
    price_per_tco2e_eur: float = Field(default=0.0, ge=0.0)
    status: CreditStatus = Field(default=CreditStatus.ACTIVE)
    registry_serial: str = Field(default="", description="Registry serial number")
    country: str = Field(default="", description="Project country")
    project_type: str = Field(default="", description="e.g., reforestation, dac, cookstoves")
    permanence_years: int = Field(default=0, ge=0, description="Permanence period")
    additionality_verified: bool = Field(default=False)
    co_benefits: List[str] = Field(default_factory=list)


class CreditQualityScore(BaseModel):
    """Quality assessment score for a carbon credit."""
    credit_id: str = Field(default="")
    quality_tier: QualityTier = Field(default=QualityTier.UNASSESSED)
    additionality_score: float = Field(default=0.0, ge=0.0, le=5.0)
    permanence_score: float = Field(default=0.0, ge=0.0, le=5.0)
    measurement_score: float = Field(default=0.0, ge=0.0, le=5.0)
    leakage_score: float = Field(default=0.0, ge=0.0, le=5.0)
    composite_score: float = Field(default=0.0, ge=0.0, le=5.0)
    is_removal: bool = Field(default=False)
    sbti_eligible: bool = Field(default=False)


class CarbonCreditsInput(BaseModel):
    """Input data model for CarbonCreditsWorkflow."""
    credits: List[CarbonCredit] = Field(
        default_factory=list, description="Carbon credit records"
    )
    total_emissions_tco2e: float = Field(
        default=0.0, ge=0.0, description="Total GHG emissions for offset ratio"
    )
    reporting_year: int = Field(default=2025, ge=2020, le=2050)
    entity_id: str = Field(default="")
    entity_name: str = Field(default="")
    quality_threshold: float = Field(
        default=3.0, ge=0.0, le=5.0, description="Minimum quality score"
    )
    config: Dict[str, Any] = Field(default_factory=dict)


class CarbonCreditsResult(BaseModel):
    """Complete result from carbon credits workflow."""
    workflow_id: str = Field(..., description="Unique execution ID")
    workflow_name: str = Field(default="carbon_credits")
    status: WorkflowStatus = Field(...)
    phases: List[PhaseResult] = Field(default_factory=list)
    phases_completed: int = Field(default=0, description="Number of phases completed")
    duration_ms: float = Field(default=0.0, description="Total duration in milliseconds")
    total_duration_seconds: float = Field(default=0.0)
    credits: List[CarbonCredit] = Field(default_factory=list)
    quality_scores: List[CreditQualityScore] = Field(default_factory=list)
    total_credits_tco2e: float = Field(default=0.0)
    removal_credits_tco2e: float = Field(default=0.0)
    avoidance_credits_tco2e: float = Field(default=0.0)
    retired_credits_tco2e: float = Field(default=0.0)
    total_investment_eur: float = Field(default=0.0)
    offset_ratio_pct: float = Field(default=0.0)
    high_quality_pct: float = Field(default=0.0)
    sbti_compliant: bool = Field(default=False)
    reporting_year: int = Field(default=2025)
    provenance_hash: str = Field(default="")


# =============================================================================
# QUALITY SCORING WEIGHTS
# =============================================================================

QUALITY_WEIGHTS: Dict[str, float] = {
    "additionality": 0.30,
    "permanence": 0.25,
    "measurement": 0.25,
    "leakage": 0.20,
}

# Standard quality baseline scores
STANDARD_BASELINE: Dict[str, float] = {
    "gold_standard": 4.0,
    "vcs": 3.5,
    "cdm": 3.0,
    "acr": 3.5,
    "car": 3.5,
    "puro_earth": 4.2,
    "isometric": 4.5,
    "eu_ets": 4.0,
    "other": 2.0,
}


# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================


class CarbonCreditsWorkflow:
    """
    5-phase carbon credit/offset management workflow for ESRS E1-7.

    Implements carbon credit registration, quality assessment with
    additionality/permanence/measurement/leakage scoring, portfolio
    analysis, SBTi compliance checking, and disclosure-ready output.

    Zero-hallucination: all quality scores use deterministic weighted
    averages with documented standard baselines.

    Example:
        >>> wf = CarbonCreditsWorkflow()
        >>> inp = CarbonCreditsInput(credits=[...])
        >>> result = await wf.execute(inp)
        >>> assert result.total_credits_tco2e >= 0
    """

    def __init__(self, config: Optional[Any] = None) -> None:
        """Initialize CarbonCreditsWorkflow."""
        self.workflow_id: str = str(uuid.uuid4())
        self.config = config
        self._credits: List[CarbonCredit] = []
        self._quality_scores: List[CreditQualityScore] = []
        self._sbti_compliant: bool = False
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def get_phases(self) -> List[Dict[str, str]]:
        """Return phase definitions for this workflow."""
        return [
            {"name": WorkflowPhase.CREDIT_REGISTRATION.value, "description": "Register carbon credits and offsets"},
            {"name": WorkflowPhase.QUALITY_ASSESSMENT.value, "description": "Assess credit quality and standards"},
            {"name": WorkflowPhase.PORTFOLIO_ANALYSIS.value, "description": "Analyze credit portfolio composition"},
            {"name": WorkflowPhase.SBTI_CHECK.value, "description": "Verify SBTi treatment of credits"},
            {"name": WorkflowPhase.REPORT_GENERATION.value, "description": "Produce E1-7 disclosure data"},
        ]

    def validate_inputs(self, input_data: CarbonCreditsInput) -> List[str]:
        """Validate workflow inputs and return list of issues."""
        issues: List[str] = []
        if not input_data.credits:
            issues.append("No carbon credits provided")
        for c in input_data.credits:
            if c.quantity_tco2e <= 0:
                issues.append(f"Credit {c.credit_id}: zero or negative quantity")
        return issues

    async def execute(
        self,
        input_data: Optional[CarbonCreditsInput] = None,
        credits: Optional[List[CarbonCredit]] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> CarbonCreditsResult:
        """
        Execute the 5-phase carbon credits workflow.

        Args:
            input_data: Full input model (preferred).
            credits: Carbon credits (fallback).
            config: Configuration overrides.

        Returns:
            CarbonCreditsResult with quality scores, portfolio, and SBTi status.
        """
        if input_data is None:
            input_data = CarbonCreditsInput(
                credits=credits or [],
                config=config or {},
            )

        started_at = _utcnow()
        self.logger.info("Starting carbon credits workflow %s", self.workflow_id)
        phase_results: List[PhaseResult] = []
        overall_status = WorkflowStatus.IN_PROGRESS

        try:
            phase_results.append(await self._phase_credit_registration(input_data))
            phase_results.append(await self._phase_quality_assessment(input_data))
            phase_results.append(await self._phase_portfolio_analysis(input_data))
            phase_results.append(await self._phase_sbti_check(input_data))
            phase_results.append(await self._phase_report_generation(input_data))
            overall_status = WorkflowStatus.COMPLETED
        except Exception as exc:
            self.logger.error("Carbon credits workflow failed: %s", exc, exc_info=True)
            overall_status = WorkflowStatus.FAILED
            phase_results.append(PhaseResult(
                phase_name="error", status=PhaseStatus.FAILED, errors=[str(exc)],
            ))

        elapsed = (_utcnow() - started_at).total_seconds()
        completed_count = sum(1 for p in phase_results if p.status == PhaseStatus.COMPLETED)
        total_tco2e = sum(c.quantity_tco2e for c in self._credits)
        removals = sum(c.quantity_tco2e for c in self._credits if c.credit_type == CreditType.REMOVAL)
        avoidance = sum(c.quantity_tco2e for c in self._credits if c.credit_type == CreditType.AVOIDANCE)
        retired = sum(c.quantity_tco2e for c in self._credits if c.status == CreditStatus.RETIRED)
        total_investment = sum(c.quantity_tco2e * c.price_per_tco2e_eur for c in self._credits)
        offset_ratio = round(
            (total_tco2e / input_data.total_emissions_tco2e * 100)
            if input_data.total_emissions_tco2e > 0 else 0.0, 2
        )
        high_quality = sum(1 for q in self._quality_scores if q.quality_tier == QualityTier.HIGH)
        high_quality_pct = round(
            (high_quality / len(self._quality_scores) * 100) if self._quality_scores else 0.0, 1
        )

        result = CarbonCreditsResult(
            workflow_id=self.workflow_id,
            status=overall_status,
            phases=phase_results,
            phases_completed=completed_count,
            duration_ms=round(elapsed * 1000, 2),
            total_duration_seconds=elapsed,
            credits=self._credits,
            quality_scores=self._quality_scores,
            total_credits_tco2e=round(total_tco2e, 2),
            removal_credits_tco2e=round(removals, 2),
            avoidance_credits_tco2e=round(avoidance, 2),
            retired_credits_tco2e=round(retired, 2),
            total_investment_eur=round(total_investment, 2),
            offset_ratio_pct=offset_ratio,
            high_quality_pct=high_quality_pct,
            sbti_compliant=self._sbti_compliant,
            reporting_year=input_data.reporting_year,
        )
        result.provenance_hash = self._compute_provenance(result)
        self.logger.info(
            "Carbon credits %s completed in %.2fs: %.0f tCO2e total, %.0f removal",
            self.workflow_id, elapsed, total_tco2e, removals,
        )
        return result

    # -------------------------------------------------------------------------
    # Phase 1: Credit Registration
    # -------------------------------------------------------------------------

    async def _phase_credit_registration(
        self, input_data: CarbonCreditsInput,
    ) -> PhaseResult:
        """Register all carbon credits."""
        started = _utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        self._credits = list(input_data.credits)

        type_counts: Dict[str, int] = {}
        standard_counts: Dict[str, int] = {}
        for c in self._credits:
            type_counts[c.credit_type.value] = type_counts.get(c.credit_type.value, 0) + 1
            standard_counts[c.standard.value] = standard_counts.get(c.standard.value, 0) + 1

        outputs["credits_registered"] = len(self._credits)
        outputs["type_distribution"] = type_counts
        outputs["standard_distribution"] = standard_counts
        outputs["total_quantity_tco2e"] = round(
            sum(c.quantity_tco2e for c in self._credits), 2
        )

        if not self._credits:
            warnings.append("No carbon credits registered")

        elapsed = (_utcnow() - started).total_seconds()
        self.logger.info("Phase 1 CreditRegistration: %d credits registered", len(self._credits))
        return PhaseResult(
            phase_name=WorkflowPhase.CREDIT_REGISTRATION.value, status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 2: Quality Assessment
    # -------------------------------------------------------------------------

    async def _phase_quality_assessment(
        self, input_data: CarbonCreditsInput,
    ) -> PhaseResult:
        """Assess quality of each carbon credit."""
        started = _utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []
        self._quality_scores = []

        for credit in self._credits:
            quality = self._assess_credit_quality(credit)
            self._quality_scores.append(quality)

        tier_counts: Dict[str, int] = {}
        for q in self._quality_scores:
            tier_counts[q.quality_tier.value] = tier_counts.get(q.quality_tier.value, 0) + 1

        outputs["credits_assessed"] = len(self._quality_scores)
        outputs["tier_distribution"] = tier_counts
        outputs["avg_composite_score"] = round(
            sum(q.composite_score for q in self._quality_scores) / len(self._quality_scores)
            if self._quality_scores else 0.0, 2
        )

        low_quality = sum(
            1 for q in self._quality_scores
            if q.composite_score < input_data.quality_threshold
        )
        if low_quality > 0:
            warnings.append(f"{low_quality} credits below quality threshold ({input_data.quality_threshold})")

        elapsed = (_utcnow() - started).total_seconds()
        self.logger.info("Phase 2 QualityAssessment: %s", tier_counts)
        return PhaseResult(
            phase_name=WorkflowPhase.QUALITY_ASSESSMENT.value, status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    def _assess_credit_quality(self, credit: CarbonCredit) -> CreditQualityScore:
        """Deterministic quality assessment for a carbon credit."""
        baseline = STANDARD_BASELINE.get(credit.standard.value, 2.0)

        additionality = min(5.0, baseline + (0.5 if credit.additionality_verified else -0.5))
        permanence = min(5.0, 2.0 + min(credit.permanence_years / 25.0, 1.0) * 3.0)
        measurement = min(5.0, baseline + (0.3 if credit.credit_type == CreditType.REMOVAL else 0.0))
        leakage = min(5.0, baseline - 0.2)

        composite = (
            additionality * QUALITY_WEIGHTS["additionality"]
            + permanence * QUALITY_WEIGHTS["permanence"]
            + measurement * QUALITY_WEIGHTS["measurement"]
            + leakage * QUALITY_WEIGHTS["leakage"]
        )

        if composite >= 4.0:
            tier = QualityTier.HIGH
        elif composite >= 3.0:
            tier = QualityTier.MEDIUM
        else:
            tier = QualityTier.LOW

        return CreditQualityScore(
            credit_id=credit.credit_id,
            quality_tier=tier,
            additionality_score=round(additionality, 2),
            permanence_score=round(permanence, 2),
            measurement_score=round(measurement, 2),
            leakage_score=round(leakage, 2),
            composite_score=round(composite, 2),
            is_removal=credit.credit_type == CreditType.REMOVAL,
            sbti_eligible=credit.credit_type == CreditType.REMOVAL and composite >= 3.5,
        )

    # -------------------------------------------------------------------------
    # Phase 3: Portfolio Analysis
    # -------------------------------------------------------------------------

    async def _phase_portfolio_analysis(
        self, input_data: CarbonCreditsInput,
    ) -> PhaseResult:
        """Analyze the carbon credit portfolio composition."""
        started = _utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        total = sum(c.quantity_tco2e for c in self._credits)
        removals = sum(c.quantity_tco2e for c in self._credits if c.credit_type == CreditType.REMOVAL)
        avoidance = sum(c.quantity_tco2e for c in self._credits if c.credit_type == CreditType.AVOIDANCE)

        # Country diversification
        country_breakdown: Dict[str, float] = {}
        for c in self._credits:
            country_breakdown[c.country or "unknown"] = (
                country_breakdown.get(c.country or "unknown", 0.0) + c.quantity_tco2e
            )

        # Vintage distribution
        vintage_breakdown: Dict[int, float] = {}
        for c in self._credits:
            vintage_breakdown[c.vintage_year] = vintage_breakdown.get(c.vintage_year, 0.0) + c.quantity_tco2e

        outputs["total_portfolio_tco2e"] = round(total, 2)
        outputs["removal_share_pct"] = round((removals / total * 100) if total > 0 else 0.0, 1)
        outputs["avoidance_share_pct"] = round((avoidance / total * 100) if total > 0 else 0.0, 1)
        outputs["country_count"] = len(country_breakdown)
        outputs["vintage_years"] = sorted(vintage_breakdown.keys())
        outputs["avg_price_eur"] = round(
            sum(c.price_per_tco2e_eur * c.quantity_tco2e for c in self._credits)
            / total if total > 0 else 0.0, 2
        )

        elapsed = (_utcnow() - started).total_seconds()
        self.logger.info("Phase 3 PortfolioAnalysis: %.0f tCO2e across %d countries", total, len(country_breakdown))
        return PhaseResult(
            phase_name=WorkflowPhase.PORTFOLIO_ANALYSIS.value, status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 4: SBTi Check
    # -------------------------------------------------------------------------

    async def _phase_sbti_check(
        self, input_data: CarbonCreditsInput,
    ) -> PhaseResult:
        """Check SBTi compliance for carbon credit usage."""
        started = _utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        # SBTi requires: credits do NOT count toward Scope 1/2 targets
        # Only removals allowed for beyond value chain mitigation (BVCM)
        sbti_eligible = [q for q in self._quality_scores if q.sbti_eligible]
        sbti_eligible_tco2e = sum(
            c.quantity_tco2e for c in self._credits
            if c.credit_id in {q.credit_id for q in sbti_eligible}
        )

        total_tco2e = sum(c.quantity_tco2e for c in self._credits)
        offset_ratio = (total_tco2e / input_data.total_emissions_tco2e * 100) if input_data.total_emissions_tco2e > 0 else 0.0

        # SBTi: offsets should not substitute for required reductions
        self._sbti_compliant = offset_ratio <= 10.0 and len(sbti_eligible) > 0

        outputs["sbti_eligible_credits"] = len(sbti_eligible)
        outputs["sbti_eligible_tco2e"] = round(sbti_eligible_tco2e, 2)
        outputs["offset_ratio_pct"] = round(offset_ratio, 2)
        outputs["sbti_compliant"] = self._sbti_compliant

        if offset_ratio > 10.0:
            warnings.append(
                f"Offset ratio ({offset_ratio:.1f}%) exceeds SBTi guidance (max ~10%)"
            )
        if not sbti_eligible:
            warnings.append("No SBTi-eligible removal credits in portfolio")

        elapsed = (_utcnow() - started).total_seconds()
        self.logger.info("Phase 4 SBTiCheck: compliant=%s, ratio=%.1f%%", self._sbti_compliant, offset_ratio)
        return PhaseResult(
            phase_name=WorkflowPhase.SBTI_CHECK.value, status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 5: Report Generation
    # -------------------------------------------------------------------------

    async def _phase_report_generation(
        self, input_data: CarbonCreditsInput,
    ) -> PhaseResult:
        """Generate E1-7 disclosure-ready output."""
        started = _utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        total = sum(c.quantity_tco2e for c in self._credits)
        removals = sum(c.quantity_tco2e for c in self._credits if c.credit_type == CreditType.REMOVAL)

        outputs["e1_7_disclosure"] = {
            "total_carbon_credits_tco2e": round(total, 2),
            "removal_credits_tco2e": round(removals, 2),
            "avoidance_credits_tco2e": round(total - removals, 2),
            "retired_credits_tco2e": round(
                sum(c.quantity_tco2e for c in self._credits if c.status == CreditStatus.RETIRED), 2
            ),
            "total_investment_eur": round(
                sum(c.quantity_tco2e * c.price_per_tco2e_eur for c in self._credits), 2
            ),
            "standards_used": sorted(set(c.standard.value for c in self._credits)),
            "sbti_compliant": self._sbti_compliant,
            "reporting_year": input_data.reporting_year,
        }

        outputs["report_ready"] = True

        elapsed = (_utcnow() - started).total_seconds()
        self.logger.info("Phase 5 ReportGeneration: E1-7 disclosure ready")
        return PhaseResult(
            phase_name=WorkflowPhase.REPORT_GENERATION.value, status=PhaseStatus.COMPLETED,
            duration_seconds=elapsed, outputs=outputs, warnings=warnings,
            provenance_hash=self._hash_dict(outputs),
        )

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _compute_provenance(self, result: CarbonCreditsResult) -> str:
        """Compute SHA-256 provenance hash."""
        payload = result.model_dump_json(exclude={"provenance_hash"})
        return _compute_hash(payload)

    def _hash_dict(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 of a dict."""
        raw = json.dumps(data, sort_keys=True, default=str)
        return _compute_hash(raw)
