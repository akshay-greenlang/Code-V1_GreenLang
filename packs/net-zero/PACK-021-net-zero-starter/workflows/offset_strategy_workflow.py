# -*- coding: utf-8 -*-
"""
Offset Strategy Workflow
============================

4-phase workflow for designing a carbon credit offset strategy for
residual emissions within PACK-021 Net-Zero Starter Pack.  The workflow
calculates residual emissions after maximum reduction, screens available
credit types against quality criteria, designs an optimal portfolio,
and validates compliance with SBTi, VCMI, and Oxford Principles.

Phases:
    1. ResidualCalc      -- Calculate residual emissions after abatement
    2. CreditScreening   -- Screen credit types against quality criteria
    3. PortfolioDesign   -- Design optimal credit portfolio
    4. ComplianceCheck   -- Validate against SBTi / VCMI / Oxford Principles

Regulatory references:
    - SBTi Net-Zero Standard v1.2 (Beyond Value Chain Mitigation)
    - VCMI Claims Code of Practice (2023)
    - Oxford Principles for Net Zero Aligned Carbon Offsetting (2024)
    - ICVCM Core Carbon Principles (2023)
    - Verra VCS / Gold Standard certification

Author: GreenLang Team
Version: 21.0.0
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

_MODULE_VERSION = "21.0.0"


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


class CreditType(str, Enum):
    """Carbon credit types."""

    NATURE_BASED_AVOIDANCE = "nature_based_avoidance"
    NATURE_BASED_REMOVAL = "nature_based_removal"
    TECH_BASED_REMOVAL = "tech_based_removal"
    RENEWABLE_ENERGY = "renewable_energy"
    ENERGY_EFFICIENCY = "energy_efficiency"
    COOKSTOVE = "cookstove"
    METHANE_AVOIDANCE = "methane_avoidance"
    DIRECT_AIR_CAPTURE = "direct_air_capture"
    BIOCHAR = "biochar"
    ENHANCED_WEATHERING = "enhanced_weathering"
    OCEAN_BASED = "ocean_based"


class CreditStandard(str, Enum):
    """Carbon credit certification standards."""

    VCS = "vcs"
    GOLD_STANDARD = "gold_standard"
    ACR = "acr"
    CAR = "car"
    PURO = "puro"
    ISOMETRIC = "isometric"
    SOCRATES = "socrates"


class VCMIClaimLevel(str, Enum):
    """VCMI Claims Code claim levels."""

    SILVER = "silver"
    GOLD = "gold"
    PLATINUM = "platinum"


class ComplianceStatus(str, Enum):
    """Compliance check result."""

    COMPLIANT = "compliant"
    PARTIALLY_COMPLIANT = "partially_compliant"
    NON_COMPLIANT = "non_compliant"


# =============================================================================
# CREDIT QUALITY REFERENCE DATA (Zero-Hallucination)
# =============================================================================

# Quality scores for credit types (0-100, based on Oxford Principles / ICVCM)
CREDIT_QUALITY_SCORES: Dict[str, Dict[str, Any]] = {
    "direct_air_capture": {
        "quality_score": 95,
        "permanence_years": 10000,
        "additionality_confidence": "very_high",
        "is_removal": True,
        "is_nature_based": False,
        "typical_price_usd": 600.0,
        "price_range_min": 250.0,
        "price_range_max": 1000.0,
    },
    "biochar": {
        "quality_score": 85,
        "permanence_years": 1000,
        "additionality_confidence": "high",
        "is_removal": True,
        "is_nature_based": False,
        "typical_price_usd": 150.0,
        "price_range_min": 80.0,
        "price_range_max": 300.0,
    },
    "enhanced_weathering": {
        "quality_score": 80,
        "permanence_years": 10000,
        "additionality_confidence": "high",
        "is_removal": True,
        "is_nature_based": False,
        "typical_price_usd": 120.0,
        "price_range_min": 60.0,
        "price_range_max": 250.0,
    },
    "nature_based_removal": {
        "quality_score": 70,
        "permanence_years": 40,
        "additionality_confidence": "medium",
        "is_removal": True,
        "is_nature_based": True,
        "typical_price_usd": 25.0,
        "price_range_min": 10.0,
        "price_range_max": 50.0,
    },
    "nature_based_avoidance": {
        "quality_score": 55,
        "permanence_years": 30,
        "additionality_confidence": "medium",
        "is_removal": False,
        "is_nature_based": True,
        "typical_price_usd": 12.0,
        "price_range_min": 5.0,
        "price_range_max": 30.0,
    },
    "tech_based_removal": {
        "quality_score": 90,
        "permanence_years": 5000,
        "additionality_confidence": "very_high",
        "is_removal": True,
        "is_nature_based": False,
        "typical_price_usd": 400.0,
        "price_range_min": 150.0,
        "price_range_max": 800.0,
    },
    "methane_avoidance": {
        "quality_score": 65,
        "permanence_years": 0,
        "additionality_confidence": "high",
        "is_removal": False,
        "is_nature_based": False,
        "typical_price_usd": 15.0,
        "price_range_min": 8.0,
        "price_range_max": 25.0,
    },
    "renewable_energy": {
        "quality_score": 40,
        "permanence_years": 0,
        "additionality_confidence": "low",
        "is_removal": False,
        "is_nature_based": False,
        "typical_price_usd": 5.0,
        "price_range_min": 2.0,
        "price_range_max": 10.0,
    },
    "energy_efficiency": {
        "quality_score": 45,
        "permanence_years": 0,
        "additionality_confidence": "medium",
        "is_removal": False,
        "is_nature_based": False,
        "typical_price_usd": 8.0,
        "price_range_min": 3.0,
        "price_range_max": 15.0,
    },
    "cookstove": {
        "quality_score": 50,
        "permanence_years": 0,
        "additionality_confidence": "medium",
        "is_removal": False,
        "is_nature_based": False,
        "typical_price_usd": 10.0,
        "price_range_min": 5.0,
        "price_range_max": 20.0,
    },
    "ocean_based": {
        "quality_score": 75,
        "permanence_years": 500,
        "additionality_confidence": "medium",
        "is_removal": True,
        "is_nature_based": True,
        "typical_price_usd": 80.0,
        "price_range_min": 30.0,
        "price_range_max": 200.0,
    },
}

# Oxford Principles: shift from avoidance to removal over time
OXFORD_REMOVAL_TARGETS_BY_YEAR: Dict[int, float] = {
    2025: 20.0,   # minimum % removals
    2030: 40.0,
    2035: 60.0,
    2040: 80.0,
    2050: 100.0,
}

# VCMI Claims Code requirements
VCMI_REQUIREMENTS: Dict[str, Dict[str, Any]] = {
    "silver": {
        "min_nt_reduction_pct": 0.0,    # On track with near-term target
        "min_credit_quality": 50,
        "credit_volume": "cover_residual_or_more",
    },
    "gold": {
        "min_nt_reduction_pct": 50.0,
        "min_credit_quality": 60,
        "credit_volume": "cover_residual_or_more",
    },
    "platinum": {
        "min_nt_reduction_pct": 80.0,
        "min_credit_quality": 70,
        "credit_volume": "cover_residual_or_more",
    },
}


# =============================================================================
# DATA MODELS
# =============================================================================


class PhaseResult(BaseModel):
    """Result from a single workflow phase."""

    phase_name: str = Field(..., description="Phase identifier")
    status: PhaseStatus = Field(...)
    duration_seconds: float = Field(default=0.0)
    outputs: Dict[str, Any] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")


class ResidualBudget(BaseModel):
    """Residual emissions budget after maximum reduction."""

    baseline_tco2e: float = Field(default=0.0, ge=0.0)
    reduction_achieved_tco2e: float = Field(default=0.0, ge=0.0)
    reduction_achieved_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    residual_tco2e: float = Field(default=0.0, ge=0.0)
    residual_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    target_year: int = Field(default=2050)
    annual_residual_tco2e: float = Field(default=0.0, ge=0.0)
    years_to_neutralise: int = Field(default=25, ge=1)


class CreditScreeningResult(BaseModel):
    """Screening result for a single credit type."""

    credit_type: str = Field(default="")
    quality_score: int = Field(default=0, ge=0, le=100)
    passes_minimum: bool = Field(default=False)
    is_removal: bool = Field(default=False)
    is_nature_based: bool = Field(default=False)
    permanence_years: int = Field(default=0, ge=0)
    additionality_confidence: str = Field(default="medium")
    typical_price_usd: float = Field(default=0.0, ge=0.0)
    price_range_min: float = Field(default=0.0, ge=0.0)
    price_range_max: float = Field(default=0.0, ge=0.0)
    screening_notes: List[str] = Field(default_factory=list)


class PortfolioAllocation(BaseModel):
    """Allocation of credits within the portfolio."""

    credit_type: str = Field(default="")
    allocation_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    volume_tco2e: float = Field(default=0.0, ge=0.0)
    estimated_cost_usd: float = Field(default=0.0, ge=0.0)
    quality_score: int = Field(default=0)
    is_removal: bool = Field(default=False)


class PortfolioDesign(BaseModel):
    """Designed credit portfolio."""

    allocations: List[PortfolioAllocation] = Field(default_factory=list)
    total_volume_tco2e: float = Field(default=0.0, ge=0.0)
    total_estimated_cost_usd: float = Field(default=0.0, ge=0.0)
    average_quality_score: float = Field(default=0.0, ge=0.0, le=100.0)
    removal_share_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    nature_based_share_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    weighted_cost_per_tco2e: float = Field(default=0.0, ge=0.0)
    diversification_count: int = Field(default=0)


class ComplianceFinding(BaseModel):
    """A single compliance finding."""

    framework: str = Field(default="")
    criterion: str = Field(default="")
    status: ComplianceStatus = Field(default=ComplianceStatus.NON_COMPLIANT)
    detail: str = Field(default="")


class ComplianceReport(BaseModel):
    """Full compliance check report."""

    sbti_compliant: bool = Field(default=False)
    vcmi_claim_eligible: Optional[str] = Field(None, description="Eligible VCMI claim level")
    oxford_compliant: bool = Field(default=False)
    findings: List[ComplianceFinding] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)


class OffsetStrategyConfig(BaseModel):
    """Configuration for the offset strategy workflow."""

    baseline_total_tco2e: float = Field(default=0.0, ge=0.0)
    reduction_achieved_tco2e: float = Field(default=0.0, ge=0.0)
    reduction_achieved_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    long_term_target_year: int = Field(default=2050, ge=2030, le=2060)
    quality_minimum_score: int = Field(default=50, ge=0, le=100)
    max_nature_based_pct: float = Field(default=60.0, ge=0.0, le=100.0)
    vcmi_target_claim: str = Field(default="silver")
    near_term_reduction_on_track_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    budget_per_year_usd: Optional[float] = Field(None, ge=0.0)
    entity_id: str = Field(default="")
    tenant_id: str = Field(default="")

    @field_validator("vcmi_target_claim")
    @classmethod
    def _validate_vcmi(cls, v: str) -> str:
        if v not in {"silver", "gold", "platinum"}:
            raise ValueError("vcmi_target_claim must be silver, gold, or platinum")
        return v


class OffsetStrategyResult(BaseModel):
    """Complete result from the offset strategy workflow."""

    workflow_id: str = Field(..., description="Unique execution ID")
    workflow_name: str = Field(default="offset_strategy")
    status: WorkflowStatus = Field(...)
    phases: List[PhaseResult] = Field(default_factory=list)
    total_duration_seconds: float = Field(default=0.0)
    residual_budget: ResidualBudget = Field(default_factory=ResidualBudget)
    screened_credits: List[CreditScreeningResult] = Field(default_factory=list)
    portfolio_design: PortfolioDesign = Field(default_factory=PortfolioDesign)
    compliance_status: ComplianceReport = Field(default_factory=ComplianceReport)
    provenance_hash: str = Field(default="")


# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================


class OffsetStrategyWorkflow:
    """
    4-phase offset strategy workflow for residual emissions.

    Calculates residual emissions, screens carbon credits, designs an
    optimal portfolio, and validates compliance with SBTi, VCMI, and
    Oxford Principles.

    Zero-hallucination: all quality scores, pricing, and compliance
    criteria come from deterministic reference data.  No LLM calls
    in the numeric path.

    Attributes:
        workflow_id: Unique execution identifier.

    Example:
        >>> wf = OffsetStrategyWorkflow()
        >>> cfg = OffsetStrategyConfig(baseline_total_tco2e=10000, ...)
        >>> result = await wf.execute(cfg)
        >>> assert result.compliance_status.sbti_compliant is True
    """

    def __init__(self) -> None:
        """Initialise OffsetStrategyWorkflow."""
        self.workflow_id: str = _new_uuid()
        self._phase_results: List[PhaseResult] = []
        self._residual: ResidualBudget = ResidualBudget()
        self._screened: List[CreditScreeningResult] = []
        self._portfolio: PortfolioDesign = PortfolioDesign()
        self._compliance: ComplianceReport = ComplianceReport()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    async def execute(self, config: OffsetStrategyConfig) -> OffsetStrategyResult:
        """
        Execute the 4-phase offset strategy workflow.

        Args:
            config: Offset strategy configuration with baseline, reduction
                achieved, quality preferences, and VCMI target.

        Returns:
            OffsetStrategyResult with portfolio design and compliance status.
        """
        started_at = _utcnow()
        self.logger.info(
            "Starting offset strategy workflow %s, baseline=%.2f, reduction=%.2f%%",
            self.workflow_id, config.baseline_total_tco2e, config.reduction_achieved_pct,
        )
        self._phase_results = []
        overall_status = WorkflowStatus.RUNNING

        try:
            phase1 = await self._phase_residual_calc(config)
            self._phase_results.append(phase1)

            phase2 = await self._phase_credit_screening(config)
            self._phase_results.append(phase2)

            phase3 = await self._phase_portfolio_design(config)
            self._phase_results.append(phase3)

            phase4 = await self._phase_compliance_check(config)
            self._phase_results.append(phase4)

            failed = [p for p in self._phase_results if p.status == PhaseStatus.FAILED]
            overall_status = WorkflowStatus.COMPLETED if not failed else WorkflowStatus.PARTIAL

        except Exception as exc:
            self.logger.error("Offset strategy workflow failed: %s", exc, exc_info=True)
            overall_status = WorkflowStatus.FAILED
            self._phase_results.append(PhaseResult(
                phase_name="error", status=PhaseStatus.FAILED, errors=[str(exc)],
            ))

        elapsed = (_utcnow() - started_at).total_seconds()
        result = OffsetStrategyResult(
            workflow_id=self.workflow_id,
            status=overall_status,
            phases=self._phase_results,
            total_duration_seconds=round(elapsed, 4),
            residual_budget=self._residual,
            screened_credits=self._screened,
            portfolio_design=self._portfolio,
            compliance_status=self._compliance,
        )
        result.provenance_hash = _compute_hash(
            result.model_dump_json(exclude={"provenance_hash"})
        )
        self.logger.info(
            "Offset strategy %s completed in %.2fs, portfolio=%.0f tCO2e, cost=%.0f USD",
            self.workflow_id, elapsed,
            self._portfolio.total_volume_tco2e,
            self._portfolio.total_estimated_cost_usd,
        )
        return result

    # -------------------------------------------------------------------------
    # Phase 1: Residual Calculation
    # -------------------------------------------------------------------------

    async def _phase_residual_calc(self, config: OffsetStrategyConfig) -> PhaseResult:
        """Calculate residual emissions after maximum reduction."""
        started = _utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        baseline = config.baseline_total_tco2e
        reduction = config.reduction_achieved_tco2e
        reduction_pct = config.reduction_achieved_pct

        # If reduction_tco2e not directly provided, compute from pct
        if reduction <= 0 and reduction_pct > 0 and baseline > 0:
            reduction = baseline * (reduction_pct / 100.0)
        if reduction_pct <= 0 and reduction > 0 and baseline > 0:
            reduction_pct = (reduction / baseline) * 100.0

        residual = max(baseline - reduction, 0.0)
        residual_pct = (residual / baseline * 100.0) if baseline > 0 else 0.0

        # SBTi requires at least 90% reduction for net-zero; warn if not met
        if reduction_pct < 90.0:
            warnings.append(
                f"Reduction of {reduction_pct:.1f}% is below SBTi 90% minimum for net-zero. "
                "Residual emissions can only be 'neutralised' if abatement reaches 90%+."
            )

        years_to_target = max(config.long_term_target_year - _utcnow().year, 1)
        annual_residual = residual / years_to_target if years_to_target > 0 else residual

        self._residual = ResidualBudget(
            baseline_tco2e=round(baseline, 4),
            reduction_achieved_tco2e=round(reduction, 4),
            reduction_achieved_pct=round(reduction_pct, 2),
            residual_tco2e=round(residual, 4),
            residual_pct=round(residual_pct, 2),
            target_year=config.long_term_target_year,
            annual_residual_tco2e=round(annual_residual, 4),
            years_to_neutralise=years_to_target,
        )

        outputs["baseline_tco2e"] = self._residual.baseline_tco2e
        outputs["reduction_achieved_tco2e"] = self._residual.reduction_achieved_tco2e
        outputs["residual_tco2e"] = self._residual.residual_tco2e
        outputs["annual_residual_tco2e"] = self._residual.annual_residual_tco2e

        elapsed = (_utcnow() - started).total_seconds()
        self.logger.info("Residual: %.2f tCO2e (%.1f%% of baseline)", residual, residual_pct)
        return PhaseResult(
            phase_name="residual_calc",
            status=PhaseStatus.COMPLETED,
            duration_seconds=round(elapsed, 4),
            outputs=outputs,
            warnings=warnings,
            provenance_hash=_compute_hash(json.dumps(outputs, sort_keys=True, default=str)),
        )

    # -------------------------------------------------------------------------
    # Phase 2: Credit Screening
    # -------------------------------------------------------------------------

    async def _phase_credit_screening(self, config: OffsetStrategyConfig) -> PhaseResult:
        """Screen available credit types against quality criteria."""
        started = _utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []
        screened: List[CreditScreeningResult] = []

        min_quality = config.quality_minimum_score

        for credit_type, info in CREDIT_QUALITY_SCORES.items():
            passes = info["quality_score"] >= min_quality
            notes: List[str] = []

            if not passes:
                notes.append(f"Quality score {info['quality_score']} below minimum {min_quality}")
            if info["permanence_years"] < 100:
                notes.append(f"Limited permanence: {info['permanence_years']} years")
            if info["additionality_confidence"] == "low":
                notes.append("Low additionality confidence")

            screened.append(CreditScreeningResult(
                credit_type=credit_type,
                quality_score=info["quality_score"],
                passes_minimum=passes,
                is_removal=info["is_removal"],
                is_nature_based=info["is_nature_based"],
                permanence_years=info["permanence_years"],
                additionality_confidence=info["additionality_confidence"],
                typical_price_usd=info["typical_price_usd"],
                price_range_min=info["price_range_min"],
                price_range_max=info["price_range_max"],
                screening_notes=notes,
            ))

        # Sort by quality score descending
        screened.sort(key=lambda c: c.quality_score, reverse=True)
        self._screened = screened

        passing = [c for c in screened if c.passes_minimum]
        outputs["total_credit_types_screened"] = len(screened)
        outputs["passing_credit_types"] = len(passing)
        outputs["removal_types_available"] = sum(1 for c in passing if c.is_removal)

        if not passing:
            warnings.append("No credit types pass the minimum quality threshold")

        elapsed = (_utcnow() - started).total_seconds()
        self.logger.info("Credit screening: %d/%d pass quality threshold", len(passing), len(screened))
        return PhaseResult(
            phase_name="credit_screening",
            status=PhaseStatus.COMPLETED,
            duration_seconds=round(elapsed, 4),
            outputs=outputs,
            warnings=warnings,
            provenance_hash=_compute_hash(json.dumps(outputs, sort_keys=True, default=str)),
        )

    # -------------------------------------------------------------------------
    # Phase 3: Portfolio Design
    # -------------------------------------------------------------------------

    async def _phase_portfolio_design(self, config: OffsetStrategyConfig) -> PhaseResult:
        """Design optimal credit portfolio with diversification."""
        started = _utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        eligible = [c for c in self._screened if c.passes_minimum]
        residual = self._residual.residual_tco2e

        if not eligible or residual <= 0:
            self._portfolio = PortfolioDesign()
            outputs["portfolio_empty"] = True
            if residual <= 0:
                warnings.append("No residual emissions; no offset portfolio needed")
            elapsed = (_utcnow() - started).total_seconds()
            return PhaseResult(
                phase_name="portfolio_design",
                status=PhaseStatus.COMPLETED,
                duration_seconds=round(elapsed, 4),
                outputs=outputs,
                warnings=warnings,
                provenance_hash=_compute_hash(json.dumps(outputs, sort_keys=True, default=str)),
            )

        allocations = self._design_portfolio(eligible, residual, config)
        total_volume = sum(a.volume_tco2e for a in allocations)
        total_cost = sum(a.estimated_cost_usd for a in allocations)
        removal_volume = sum(a.volume_tco2e for a in allocations if a.is_removal)
        nature_volume = sum(
            a.volume_tco2e for a in allocations
            if CREDIT_QUALITY_SCORES.get(a.credit_type, {}).get("is_nature_based", False)
        )

        avg_quality = 0.0
        if total_volume > 0:
            avg_quality = sum(a.quality_score * a.volume_tco2e for a in allocations) / total_volume

        self._portfolio = PortfolioDesign(
            allocations=allocations,
            total_volume_tco2e=round(total_volume, 4),
            total_estimated_cost_usd=round(total_cost, 2),
            average_quality_score=round(avg_quality, 1),
            removal_share_pct=round((removal_volume / total_volume * 100) if total_volume > 0 else 0, 2),
            nature_based_share_pct=round((nature_volume / total_volume * 100) if total_volume > 0 else 0, 2),
            weighted_cost_per_tco2e=round((total_cost / total_volume) if total_volume > 0 else 0, 2),
            diversification_count=len(allocations),
        )

        # Validate nature-based share
        if self._portfolio.nature_based_share_pct > config.max_nature_based_pct:
            warnings.append(
                f"Nature-based share ({self._portfolio.nature_based_share_pct:.1f}%) "
                f"exceeds maximum ({config.max_nature_based_pct:.1f}%)"
            )

        outputs["allocation_count"] = len(allocations)
        outputs["total_volume_tco2e"] = self._portfolio.total_volume_tco2e
        outputs["total_cost_usd"] = self._portfolio.total_estimated_cost_usd
        outputs["removal_share_pct"] = self._portfolio.removal_share_pct
        outputs["nature_based_share_pct"] = self._portfolio.nature_based_share_pct
        outputs["average_quality_score"] = self._portfolio.average_quality_score

        elapsed = (_utcnow() - started).total_seconds()
        self.logger.info(
            "Portfolio designed: %d types, %.0f tCO2e, $%.0f, avg quality %.1f",
            len(allocations), total_volume, total_cost, avg_quality,
        )
        return PhaseResult(
            phase_name="portfolio_design",
            status=PhaseStatus.COMPLETED,
            duration_seconds=round(elapsed, 4),
            outputs=outputs,
            warnings=warnings,
            provenance_hash=_compute_hash(json.dumps(outputs, sort_keys=True, default=str)),
        )

    def _design_portfolio(
        self,
        eligible: List[CreditScreeningResult],
        residual: float,
        config: OffsetStrategyConfig,
    ) -> List[PortfolioAllocation]:
        """Design portfolio allocations balancing quality, cost, and diversification."""
        allocations: List[PortfolioAllocation] = []

        # Strategy: prioritise removals (Oxford Principles shift)
        current_year = _utcnow().year
        target_removal_pct = 20.0
        for yr, pct in sorted(OXFORD_REMOVAL_TARGETS_BY_YEAR.items()):
            if current_year >= yr:
                target_removal_pct = pct

        removals = [c for c in eligible if c.is_removal]
        avoidances = [c for c in eligible if not c.is_removal]

        # Allocate removals first
        removal_budget = residual * (target_removal_pct / 100.0)
        removal_allocated = self._allocate_credits(removals, removal_budget, config)
        allocations.extend(removal_allocated)

        # Allocate remainder to avoidance/other
        allocated_so_far = sum(a.volume_tco2e for a in allocations)
        remaining = max(residual - allocated_so_far, 0.0)
        if remaining > 0 and avoidances:
            avoidance_allocated = self._allocate_credits(avoidances, remaining, config)
            allocations.extend(avoidance_allocated)

        return allocations

    def _allocate_credits(
        self,
        credits: List[CreditScreeningResult],
        target_volume: float,
        config: OffsetStrategyConfig,
    ) -> List[PortfolioAllocation]:
        """Allocate credits proportionally by quality score."""
        if not credits or target_volume <= 0:
            return []

        total_quality = sum(c.quality_score for c in credits) or 1
        allocations: List[PortfolioAllocation] = []
        nature_cap = config.max_nature_based_pct / 100.0 * target_volume

        nature_used = 0.0
        for credit in credits:
            share = credit.quality_score / total_quality
            volume = target_volume * share

            # Cap nature-based
            is_nature = CREDIT_QUALITY_SCORES.get(credit.credit_type, {}).get("is_nature_based", False)
            if is_nature:
                available = max(nature_cap - nature_used, 0.0)
                volume = min(volume, available)
                nature_used += volume

            if volume <= 0:
                continue

            cost = volume * credit.typical_price_usd
            pct = (volume / target_volume * 100) if target_volume > 0 else 0

            allocations.append(PortfolioAllocation(
                credit_type=credit.credit_type,
                allocation_pct=round(pct, 2),
                volume_tco2e=round(volume, 4),
                estimated_cost_usd=round(cost, 2),
                quality_score=credit.quality_score,
                is_removal=credit.is_removal,
            ))

        return allocations

    # -------------------------------------------------------------------------
    # Phase 4: Compliance Check
    # -------------------------------------------------------------------------

    async def _phase_compliance_check(self, config: OffsetStrategyConfig) -> PhaseResult:
        """Validate against SBTi, VCMI Claims Code, and Oxford Principles."""
        started = _utcnow()
        outputs: Dict[str, Any] = {}
        findings: List[ComplianceFinding] = []
        recommendations: List[str] = []

        # SBTi Net-Zero Standard checks
        sbti_findings = self._check_sbti_compliance(config)
        findings.extend(sbti_findings)
        sbti_ok = all(f.status == ComplianceStatus.COMPLIANT for f in sbti_findings)

        # VCMI Claims Code checks
        vcmi_findings, vcmi_claim = self._check_vcmi_compliance(config)
        findings.extend(vcmi_findings)

        # Oxford Principles checks
        oxford_findings = self._check_oxford_compliance(config)
        findings.extend(oxford_findings)
        oxford_ok = all(f.status == ComplianceStatus.COMPLIANT for f in oxford_findings)

        # Generate recommendations
        recommendations = self._generate_compliance_recommendations(findings, config)

        self._compliance = ComplianceReport(
            sbti_compliant=sbti_ok,
            vcmi_claim_eligible=vcmi_claim,
            oxford_compliant=oxford_ok,
            findings=findings,
            recommendations=recommendations,
        )

        outputs["sbti_compliant"] = sbti_ok
        outputs["vcmi_claim_eligible"] = vcmi_claim
        outputs["oxford_compliant"] = oxford_ok
        outputs["finding_count"] = len(findings)

        elapsed = (_utcnow() - started).total_seconds()
        self.logger.info(
            "Compliance: SBTi=%s, VCMI=%s, Oxford=%s",
            sbti_ok, vcmi_claim, oxford_ok,
        )
        return PhaseResult(
            phase_name="compliance_check",
            status=PhaseStatus.COMPLETED,
            duration_seconds=round(elapsed, 4),
            outputs=outputs,
            provenance_hash=_compute_hash(json.dumps(outputs, sort_keys=True, default=str)),
        )

    def _check_sbti_compliance(self, config: OffsetStrategyConfig) -> List[ComplianceFinding]:
        """Check SBTi Net-Zero Standard compliance."""
        findings: List[ComplianceFinding] = []

        # SBTi: Credits are BVCM (beyond value chain mitigation), not substitutes
        if self._residual.reduction_achieved_pct >= 90.0:
            findings.append(ComplianceFinding(
                framework="SBTi",
                criterion="BVCM-1",
                status=ComplianceStatus.COMPLIANT,
                detail="Reduction exceeds 90%; credits qualify as neutralisation",
            ))
        elif self._residual.reduction_achieved_pct >= 80.0:
            findings.append(ComplianceFinding(
                framework="SBTi",
                criterion="BVCM-1",
                status=ComplianceStatus.PARTIALLY_COMPLIANT,
                detail="Reduction at 80-90%; credits qualify as compensation only (not neutralisation)",
            ))
        else:
            findings.append(ComplianceFinding(
                framework="SBTi",
                criterion="BVCM-1",
                status=ComplianceStatus.NON_COMPLIANT,
                detail=f"Reduction at {self._residual.reduction_achieved_pct:.1f}%; "
                       "below SBTi minimum for net-zero claim",
            ))

        # SBTi: No substitution of abatement with offsets
        findings.append(ComplianceFinding(
            framework="SBTi",
            criterion="BVCM-2",
            status=ComplianceStatus.COMPLIANT,
            detail="Credits applied to residual only (no abatement substitution)",
        ))

        return findings

    def _check_vcmi_compliance(self, config: OffsetStrategyConfig) -> tuple:
        """Check VCMI Claims Code compliance.  Returns (findings, eligible_claim)."""
        findings: List[ComplianceFinding] = []
        eligible_claim: Optional[str] = None

        target_claim = config.vcmi_target_claim
        reqs = VCMI_REQUIREMENTS.get(target_claim, VCMI_REQUIREMENTS["silver"])

        # Check near-term reduction on-track
        nt_on_track = config.near_term_reduction_on_track_pct >= reqs["min_nt_reduction_pct"]
        if nt_on_track:
            findings.append(ComplianceFinding(
                framework="VCMI",
                criterion=f"{target_claim.upper()}-NT",
                status=ComplianceStatus.COMPLIANT,
                detail=f"Near-term progress ({config.near_term_reduction_on_track_pct:.1f}%) "
                       f"meets {target_claim} requirement",
            ))
        else:
            findings.append(ComplianceFinding(
                framework="VCMI",
                criterion=f"{target_claim.upper()}-NT",
                status=ComplianceStatus.NON_COMPLIANT,
                detail=f"Near-term progress ({config.near_term_reduction_on_track_pct:.1f}%) "
                       f"below {target_claim} requirement ({reqs['min_nt_reduction_pct']}%)",
            ))

        # Check credit quality
        quality_ok = self._portfolio.average_quality_score >= reqs["min_credit_quality"]
        if quality_ok:
            findings.append(ComplianceFinding(
                framework="VCMI",
                criterion=f"{target_claim.upper()}-QUALITY",
                status=ComplianceStatus.COMPLIANT,
                detail=f"Average quality {self._portfolio.average_quality_score:.0f} "
                       f"meets minimum {reqs['min_credit_quality']}",
            ))
        else:
            findings.append(ComplianceFinding(
                framework="VCMI",
                criterion=f"{target_claim.upper()}-QUALITY",
                status=ComplianceStatus.NON_COMPLIANT,
                detail=f"Average quality {self._portfolio.average_quality_score:.0f} "
                       f"below minimum {reqs['min_credit_quality']}",
            ))

        # Determine eligible claim
        if nt_on_track and quality_ok:
            eligible_claim = target_claim
        else:
            # Check if lower claims are achievable
            for level in ["silver", "gold", "platinum"]:
                lvl_reqs = VCMI_REQUIREMENTS[level]
                lvl_nt = config.near_term_reduction_on_track_pct >= lvl_reqs["min_nt_reduction_pct"]
                lvl_q = self._portfolio.average_quality_score >= lvl_reqs["min_credit_quality"]
                if lvl_nt and lvl_q:
                    eligible_claim = level

        return findings, eligible_claim

    def _check_oxford_compliance(self, config: OffsetStrategyConfig) -> List[ComplianceFinding]:
        """Check Oxford Principles for Net Zero Aligned Carbon Offsetting."""
        findings: List[ComplianceFinding] = []
        current_year = _utcnow().year

        # Oxford Principle 1: Shift from avoidance to removal
        target_removal = 20.0
        for yr, pct in sorted(OXFORD_REMOVAL_TARGETS_BY_YEAR.items()):
            if current_year >= yr:
                target_removal = pct

        if self._portfolio.removal_share_pct >= target_removal:
            findings.append(ComplianceFinding(
                framework="Oxford",
                criterion="OP-1",
                status=ComplianceStatus.COMPLIANT,
                detail=f"Removal share ({self._portfolio.removal_share_pct:.1f}%) "
                       f"meets {current_year} target ({target_removal:.0f}%)",
            ))
        else:
            findings.append(ComplianceFinding(
                framework="Oxford",
                criterion="OP-1",
                status=ComplianceStatus.NON_COMPLIANT,
                detail=f"Removal share ({self._portfolio.removal_share_pct:.1f}%) "
                       f"below {current_year} target ({target_removal:.0f}%)",
            ))

        # Oxford Principle 2: Shift to long-lived storage
        long_lived = sum(
            a.volume_tco2e for a in self._portfolio.allocations
            if CREDIT_QUALITY_SCORES.get(a.credit_type, {}).get("permanence_years", 0) >= 100
        )
        long_lived_pct = (long_lived / self._portfolio.total_volume_tco2e * 100) if self._portfolio.total_volume_tco2e > 0 else 0

        if long_lived_pct >= 30:
            findings.append(ComplianceFinding(
                framework="Oxford",
                criterion="OP-2",
                status=ComplianceStatus.COMPLIANT,
                detail=f"Long-lived storage share: {long_lived_pct:.1f}%",
            ))
        else:
            findings.append(ComplianceFinding(
                framework="Oxford",
                criterion="OP-2",
                status=ComplianceStatus.PARTIALLY_COMPLIANT,
                detail=f"Long-lived storage share ({long_lived_pct:.1f}%) could be higher",
            ))

        # Oxford Principle 3: Support development of net-zero solutions
        if self._portfolio.diversification_count >= 3:
            findings.append(ComplianceFinding(
                framework="Oxford",
                criterion="OP-3",
                status=ComplianceStatus.COMPLIANT,
                detail=f"Portfolio diversified across {self._portfolio.diversification_count} credit types",
            ))
        else:
            findings.append(ComplianceFinding(
                framework="Oxford",
                criterion="OP-3",
                status=ComplianceStatus.PARTIALLY_COMPLIANT,
                detail="Limited diversification; consider adding more credit types",
            ))

        return findings

    def _generate_compliance_recommendations(
        self,
        findings: List[ComplianceFinding],
        config: OffsetStrategyConfig,
    ) -> List[str]:
        """Generate recommendations based on compliance findings."""
        recs: List[str] = []
        non_compliant = [f for f in findings if f.status == ComplianceStatus.NON_COMPLIANT]
        partial = [f for f in findings if f.status == ComplianceStatus.PARTIALLY_COMPLIANT]

        for f in non_compliant:
            if "BVCM-1" in f.criterion:
                recs.append(
                    "Increase abatement to reach 90%+ reduction before relying on credits "
                    "for SBTi net-zero neutralisation."
                )
            elif "NT" in f.criterion:
                recs.append(
                    "Accelerate near-term emission reductions to qualify for VCMI claim."
                )
            elif "QUALITY" in f.criterion:
                recs.append(
                    "Shift portfolio towards higher-quality credit types (removals, tech-based)."
                )
            elif "OP-1" in f.criterion:
                recs.append(
                    "Increase share of carbon removal credits to align with Oxford Principles "
                    "trajectory."
                )

        for f in partial:
            if "OP-2" in f.criterion:
                recs.append(
                    "Consider allocating more budget to tech-based removals (DAC, biochar, "
                    "enhanced weathering) for durable carbon storage."
                )
            if "OP-3" in f.criterion:
                recs.append(
                    "Diversify credit portfolio across more project types and geographies."
                )

        if not recs:
            recs.append(
                "Portfolio meets all compliance criteria. Review annually as standards evolve."
            )

        return recs
