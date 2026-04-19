# -*- coding: utf-8 -*-
"""
Full Net-Zero Assessment Workflow
=====================================

6-phase master workflow that chains all sub-workflows into a unified
net-zero strategy assessment within PACK-021 Net-Zero Starter Pack.
This orchestrator establishes a GHG baseline, sets SBTi-aligned
targets, builds a reduction roadmap, designs an offset portfolio,
scores organisational maturity, and compiles a unified strategy
document.

Phases:
    1. Baseline      -- Run onboarding workflow (GHG baseline)
    2. Targets       -- Run target setting workflow (SBTi targets)
    3. Reduction     -- Run reduction planning workflow (MACC + roadmap)
    4. Offsets       -- Run offset strategy workflow (residual credits)
    5. Scorecard     -- Calculate net-zero maturity scorecard
    6. Strategy      -- Compile unified net-zero strategy document

Zero-hallucination: all numeric results are derived from deterministic
sub-workflow calculations.  SHA-256 provenance hashes across the full
chain guarantee end-to-end auditability.

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

from pydantic import BaseModel, Field

from .net_zero_onboarding_workflow import (
    NetZeroOnboardingWorkflow,
    OnboardingConfig,
    OnboardingInput,
    OnboardingResult,
    ScopeBreakdown,
)
from .target_setting_workflow import (
    BaselineEmissions,
    TargetDefinition,
    TargetSettingConfig,
    TargetSettingResult,
    TargetSettingWorkflow,
)
from .reduction_planning_workflow import (
    EmissionsProfile,
    ReductionPlanningConfig,
    ReductionPlanningResult,
    ReductionPlanningWorkflow,
)
from .offset_strategy_workflow import (
    OffsetStrategyConfig,
    OffsetStrategyResult,
    OffsetStrategyWorkflow,
)

logger = logging.getLogger(__name__)

_MODULE_VERSION = "21.0.0"

# =============================================================================
# HELPERS
# =============================================================================

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

class MaturityLevel(str, Enum):
    """Net-zero maturity levels."""

    NASCENT = "nascent"           # Score 0-20
    DEVELOPING = "developing"     # Score 21-40
    ESTABLISHED = "established"   # Score 41-60
    ADVANCED = "advanced"         # Score 61-80
    LEADING = "leading"           # Score 81-100

# =============================================================================
# MATURITY SCORING CRITERIA (Zero-Hallucination)
# =============================================================================

# Scoring dimensions and weights (total = 100)
MATURITY_DIMENSIONS: Dict[str, Dict[str, Any]] = {
    "baseline_quality": {
        "weight": 15,
        "description": "Quality and completeness of GHG emissions baseline",
        "scoring": {
            "complete_s1_s2_s3": 15,
            "complete_s1_s2": 10,
            "partial": 5,
            "none": 0,
        },
    },
    "target_ambition": {
        "weight": 20,
        "description": "Ambition and validity of emission reduction targets",
        "scoring": {
            "sbti_nz_validated": 20,
            "sbti_nt_validated": 15,
            "internal_15c": 10,
            "internal_other": 5,
            "none": 0,
        },
    },
    "reduction_plan": {
        "weight": 20,
        "description": "Comprehensiveness of reduction roadmap",
        "scoring": {
            "detailed_macc_phased": 20,
            "identified_actions": 12,
            "general_plan": 6,
            "none": 0,
        },
    },
    "scope3_engagement": {
        "weight": 15,
        "description": "Scope 3 supplier engagement and data quality",
        "scoring": {
            "primary_data_top_suppliers": 15,
            "spend_based_all_cats": 10,
            "partial_scope3": 5,
            "none": 0,
        },
    },
    "governance": {
        "weight": 10,
        "description": "Board-level oversight and accountability",
        "scoring": {
            "board_oversight_kpis": 10,
            "management_oversight": 6,
            "informal": 3,
            "none": 0,
        },
    },
    "offset_strategy": {
        "weight": 10,
        "description": "Quality and compliance of offset/BVCM strategy",
        "scoring": {
            "high_quality_removals": 10,
            "mixed_portfolio": 7,
            "avoidance_only": 4,
            "none": 0,
        },
    },
    "reporting_transparency": {
        "weight": 10,
        "description": "Public disclosure and reporting transparency",
        "scoring": {
            "cdp_sbti_public": 10,
            "annual_report": 6,
            "internal_only": 3,
            "none": 0,
        },
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

class ScorecardDimension(BaseModel):
    """Single dimension of the maturity scorecard."""

    dimension: str = Field(default="")
    description: str = Field(default="")
    score: float = Field(default=0.0, ge=0.0)
    max_score: float = Field(default=0.0, ge=0.0)
    assessment: str = Field(default="")
    recommendations: List[str] = Field(default_factory=list)

class NetZeroScorecard(BaseModel):
    """Net-zero maturity scorecard."""

    total_score: float = Field(default=0.0, ge=0.0, le=100.0)
    maturity_level: MaturityLevel = Field(default=MaturityLevel.NASCENT)
    dimensions: List[ScorecardDimension] = Field(default_factory=list)
    strengths: List[str] = Field(default_factory=list)
    improvement_areas: List[str] = Field(default_factory=list)

class StrategySummary(BaseModel):
    """Unified net-zero strategy document summary."""

    organisation_name: str = Field(default="")
    assessment_date: str = Field(default="")
    base_year: int = Field(default=2024)
    baseline_total_tco2e: float = Field(default=0.0, ge=0.0)
    scope1_tco2e: float = Field(default=0.0, ge=0.0)
    scope2_tco2e: float = Field(default=0.0, ge=0.0)
    scope3_tco2e: float = Field(default=0.0, ge=0.0)
    near_term_target_year: int = Field(default=2030)
    near_term_reduction_pct: float = Field(default=0.0)
    long_term_target_year: int = Field(default=2050)
    long_term_reduction_pct: float = Field(default=0.0)
    pathway: str = Field(default="")
    ambition_level: str = Field(default="")
    prioritized_action_count: int = Field(default=0)
    total_abatement_tco2e: float = Field(default=0.0, ge=0.0)
    total_abatement_capex_usd: float = Field(default=0.0, ge=0.0)
    residual_tco2e: float = Field(default=0.0, ge=0.0)
    offset_portfolio_cost_usd: float = Field(default=0.0, ge=0.0)
    offset_removal_share_pct: float = Field(default=0.0)
    maturity_score: float = Field(default=0.0, ge=0.0, le=100.0)
    maturity_level: str = Field(default="")
    sbti_compliant: bool = Field(default=False)
    vcmi_claim: Optional[str] = Field(None)
    key_actions: List[str] = Field(default_factory=list)
    key_risks: List[str] = Field(default_factory=list)

class FullAssessmentConfig(BaseModel):
    """Configuration combining all sub-workflow configs."""

    # Onboarding
    onboarding_input: OnboardingInput = Field(default_factory=OnboardingInput)

    # Target setting overrides
    sector: str = Field(default="other")
    sub_sector: str = Field(default="")
    preferred_pathway: Optional[str] = Field(None)
    ambition_level: str = Field(default="1.5C")
    near_term_target_year: Optional[int] = Field(None)
    long_term_target_year: int = Field(default=2050)

    # Reduction planning
    budget_constraint_usd: Optional[float] = Field(None, ge=0.0)
    max_actions: int = Field(default=20, ge=1, le=50)
    planning_horizon_years: int = Field(default=10)
    include_scope3_actions: bool = Field(default=True)

    # Offset strategy
    quality_minimum_score: int = Field(default=50)
    max_nature_based_pct: float = Field(default=60.0)
    vcmi_target_claim: str = Field(default="silver")

    # Governance context (for scorecard)
    has_board_oversight: bool = Field(default=False)
    has_public_reporting: bool = Field(default=False)
    has_cdp_disclosure: bool = Field(default=False)
    primary_supplier_data: bool = Field(default=False)

    entity_id: str = Field(default="")
    tenant_id: str = Field(default="")

class FullAssessmentResult(BaseModel):
    """Complete result from the full net-zero assessment workflow."""

    workflow_id: str = Field(..., description="Unique execution ID")
    workflow_name: str = Field(default="full_net_zero_assessment")
    status: WorkflowStatus = Field(...)
    phases: List[PhaseResult] = Field(default_factory=list)
    total_duration_seconds: float = Field(default=0.0)
    onboarding_result: Optional[OnboardingResult] = Field(None)
    target_setting_result: Optional[TargetSettingResult] = Field(None)
    reduction_planning_result: Optional[ReductionPlanningResult] = Field(None)
    offset_strategy_result: Optional[OffsetStrategyResult] = Field(None)
    scorecard: NetZeroScorecard = Field(default_factory=NetZeroScorecard)
    strategy_summary: StrategySummary = Field(default_factory=StrategySummary)
    provenance_hash: str = Field(default="")

# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================

class FullNetZeroAssessmentWorkflow:
    """
    6-phase master workflow chaining all net-zero sub-workflows.

    Establishes baseline, sets targets, plans reductions, designs
    offset portfolio, scores maturity, and compiles a unified
    net-zero strategy document.

    This orchestrator delegates to sub-workflow classes and aggregates
    their outputs.  Zero-hallucination: all numeric values propagate
    from deterministic sub-workflow calculations.

from greenlang.schemas import utcnow

    Attributes:
        workflow_id: Unique execution identifier.

    Example:
        >>> wf = FullNetZeroAssessmentWorkflow()
        >>> cfg = FullAssessmentConfig(onboarding_input=inp, sector="manufacturing")
        >>> result = await wf.execute(cfg)
        >>> assert result.status == WorkflowStatus.COMPLETED
        >>> print(result.scorecard.maturity_level)
    """

    def __init__(self) -> None:
        """Initialise FullNetZeroAssessmentWorkflow."""
        self.workflow_id: str = _new_uuid()
        self._phase_results: List[PhaseResult] = []
        self._onboarding_result: Optional[OnboardingResult] = None
        self._target_result: Optional[TargetSettingResult] = None
        self._reduction_result: Optional[ReductionPlanningResult] = None
        self._offset_result: Optional[OffsetStrategyResult] = None
        self._scorecard: NetZeroScorecard = NetZeroScorecard()
        self._strategy: StrategySummary = StrategySummary()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    async def execute(self, config: FullAssessmentConfig) -> FullAssessmentResult:
        """
        Execute the 6-phase full net-zero assessment workflow.

        Args:
            config: Full assessment configuration combining onboarding
                input, target preferences, budget constraints, and
                offset preferences.

        Returns:
            FullAssessmentResult with all sub-results and unified strategy.
        """
        started_at = utcnow()
        self.logger.info(
            "Starting full net-zero assessment workflow %s", self.workflow_id,
        )
        self._phase_results = []
        overall_status = WorkflowStatus.RUNNING

        try:
            # Phase 1: Baseline
            phase1 = await self._phase_baseline(config)
            self._phase_results.append(phase1)
            if phase1.status == PhaseStatus.FAILED:
                raise ValueError("Baseline phase failed; cannot proceed")

            # Phase 2: Targets
            phase2 = await self._phase_targets(config)
            self._phase_results.append(phase2)

            # Phase 3: Reduction
            phase3 = await self._phase_reduction(config)
            self._phase_results.append(phase3)

            # Phase 4: Offsets
            phase4 = await self._phase_offsets(config)
            self._phase_results.append(phase4)

            # Phase 5: Scorecard
            phase5 = await self._phase_scorecard(config)
            self._phase_results.append(phase5)

            # Phase 6: Strategy
            phase6 = await self._phase_strategy(config)
            self._phase_results.append(phase6)

            failed = [p for p in self._phase_results if p.status == PhaseStatus.FAILED]
            overall_status = WorkflowStatus.COMPLETED if not failed else WorkflowStatus.PARTIAL

        except Exception as exc:
            self.logger.error("Full assessment workflow failed: %s", exc, exc_info=True)
            overall_status = WorkflowStatus.FAILED
            self._phase_results.append(PhaseResult(
                phase_name="error", status=PhaseStatus.FAILED, errors=[str(exc)],
            ))

        elapsed = (utcnow() - started_at).total_seconds()
        result = FullAssessmentResult(
            workflow_id=self.workflow_id,
            status=overall_status,
            phases=self._phase_results,
            total_duration_seconds=round(elapsed, 4),
            onboarding_result=self._onboarding_result,
            target_setting_result=self._target_result,
            reduction_planning_result=self._reduction_result,
            offset_strategy_result=self._offset_result,
            scorecard=self._scorecard,
            strategy_summary=self._strategy,
        )
        result.provenance_hash = _compute_hash(
            result.model_dump_json(exclude={"provenance_hash"})
        )
        self.logger.info(
            "Full assessment %s completed in %.2fs, maturity=%s (%s)",
            self.workflow_id, elapsed,
            self._scorecard.total_score,
            self._scorecard.maturity_level.value,
        )
        return result

    # -------------------------------------------------------------------------
    # Phase 1: Baseline
    # -------------------------------------------------------------------------

    async def _phase_baseline(self, config: FullAssessmentConfig) -> PhaseResult:
        """Run onboarding workflow to establish GHG baseline."""
        started = utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        try:
            wf = NetZeroOnboardingWorkflow()
            self._onboarding_result = await wf.execute(config.onboarding_input)

            outputs["baseline_status"] = self._onboarding_result.status.value
            outputs["total_tco2e"] = self._onboarding_result.baseline.total_tco2e
            outputs["scope1_tco2e"] = self._onboarding_result.baseline.scope1_total_tco2e
            outputs["scope2_tco2e"] = self._onboarding_result.baseline.scope2_location_tco2e
            outputs["scope3_tco2e"] = self._onboarding_result.baseline.scope3_total_tco2e
            outputs["data_quality_score"] = self._onboarding_result.data_quality_report.overall_score

            if self._onboarding_result.baseline.total_tco2e <= 0:
                warnings.append("Baseline total is zero; downstream phases may produce limited results")

            status = PhaseStatus.COMPLETED
        except Exception as exc:
            self.logger.error("Baseline phase failed: %s", exc, exc_info=True)
            outputs["error"] = str(exc)
            warnings.append(str(exc))
            status = PhaseStatus.FAILED

        elapsed = (utcnow() - started).total_seconds()
        return PhaseResult(
            phase_name="baseline",
            status=status,
            duration_seconds=round(elapsed, 4),
            outputs=outputs,
            warnings=warnings,
            provenance_hash=_compute_hash(json.dumps(outputs, sort_keys=True, default=str)),
        )

    # -------------------------------------------------------------------------
    # Phase 2: Targets
    # -------------------------------------------------------------------------

    async def _phase_targets(self, config: FullAssessmentConfig) -> PhaseResult:
        """Run target setting workflow to define SBTi-aligned targets."""
        started = utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        try:
            baseline = self._onboarding_result.baseline if self._onboarding_result else ScopeBreakdown()
            total = baseline.total_tco2e
            scope3_pct = (baseline.scope3_total_tco2e / total * 100) if total > 0 else 0

            base_year = config.onboarding_input.config.base_year

            ts_config = TargetSettingConfig(
                base_year=base_year,
                baseline_emissions=BaselineEmissions(
                    base_year=base_year,
                    scope1_tco2e=baseline.scope1_total_tco2e,
                    scope2_tco2e=baseline.scope2_location_tco2e,
                    scope3_tco2e=baseline.scope3_total_tco2e,
                    total_tco2e=total,
                    scope3_pct_of_total=scope3_pct,
                ),
                sector=config.sector,
                sub_sector=config.sub_sector,
                preferred_pathway=config.preferred_pathway,
                ambition_level=config.ambition_level,
                near_term_target_year=config.near_term_target_year,
                long_term_target_year=config.long_term_target_year,
                entity_id=config.entity_id,
                tenant_id=config.tenant_id,
            )

            wf = TargetSettingWorkflow()
            self._target_result = await wf.execute(ts_config)

            outputs["target_status"] = self._target_result.status.value
            outputs["validation_valid"] = self._target_result.validation_results.overall_valid
            if self._target_result.near_term_target:
                outputs["near_term_year"] = self._target_result.near_term_target.target_year
                outputs["near_term_reduction_pct"] = self._target_result.near_term_target.reduction_pct
            if self._target_result.long_term_target:
                outputs["long_term_year"] = self._target_result.long_term_target.target_year
                outputs["long_term_reduction_pct"] = self._target_result.long_term_target.reduction_pct
            outputs["pathway"] = self._target_result.pathway.pathway

            status = PhaseStatus.COMPLETED
        except Exception as exc:
            self.logger.error("Targets phase failed: %s", exc, exc_info=True)
            outputs["error"] = str(exc)
            warnings.append(str(exc))
            status = PhaseStatus.FAILED

        elapsed = (utcnow() - started).total_seconds()
        return PhaseResult(
            phase_name="targets",
            status=status,
            duration_seconds=round(elapsed, 4),
            outputs=outputs,
            warnings=warnings,
            provenance_hash=_compute_hash(json.dumps(outputs, sort_keys=True, default=str)),
        )

    # -------------------------------------------------------------------------
    # Phase 3: Reduction
    # -------------------------------------------------------------------------

    async def _phase_reduction(self, config: FullAssessmentConfig) -> PhaseResult:
        """Run reduction planning workflow to build MACC and roadmap."""
        started = utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        try:
            baseline = self._onboarding_result.baseline if self._onboarding_result else ScopeBreakdown()
            base_year = config.onboarding_input.config.base_year

            profile = EmissionsProfile(
                scope1_stationary_tco2e=baseline.scope1_stationary_tco2e,
                scope1_mobile_tco2e=baseline.scope1_mobile_tco2e,
                scope1_total_tco2e=baseline.scope1_total_tco2e,
                scope2_location_tco2e=baseline.scope2_location_tco2e,
                scope3_by_category=baseline.scope3_by_category,
                scope3_total_tco2e=baseline.scope3_total_tco2e,
                total_tco2e=baseline.total_tco2e,
            )

            rp_config = ReductionPlanningConfig(
                emissions_profile=profile,
                budget_constraint_usd=config.budget_constraint_usd,
                max_actions=config.max_actions,
                planning_horizon_years=config.planning_horizon_years,
                base_year=base_year,
                include_scope3_actions=config.include_scope3_actions,
                entity_id=config.entity_id,
                tenant_id=config.tenant_id,
            )

            wf = ReductionPlanningWorkflow()
            self._reduction_result = await wf.execute(rp_config)

            outputs["reduction_status"] = self._reduction_result.status.value
            outputs["action_count"] = len(self._reduction_result.prioritized_actions)
            outputs["total_reduction_tco2e"] = self._reduction_result.total_reduction_tco2e
            outputs["total_reduction_pct"] = self._reduction_result.total_reduction_pct
            outputs["total_capex_usd"] = self._reduction_result.total_capex_usd

            status = PhaseStatus.COMPLETED
        except Exception as exc:
            self.logger.error("Reduction phase failed: %s", exc, exc_info=True)
            outputs["error"] = str(exc)
            warnings.append(str(exc))
            status = PhaseStatus.FAILED

        elapsed = (utcnow() - started).total_seconds()
        return PhaseResult(
            phase_name="reduction",
            status=status,
            duration_seconds=round(elapsed, 4),
            outputs=outputs,
            warnings=warnings,
            provenance_hash=_compute_hash(json.dumps(outputs, sort_keys=True, default=str)),
        )

    # -------------------------------------------------------------------------
    # Phase 4: Offsets
    # -------------------------------------------------------------------------

    async def _phase_offsets(self, config: FullAssessmentConfig) -> PhaseResult:
        """Run offset strategy workflow for residual emissions."""
        started = utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        try:
            baseline = self._onboarding_result.baseline if self._onboarding_result else ScopeBreakdown()
            reduction_tco2e = self._reduction_result.total_reduction_tco2e if self._reduction_result else 0.0
            reduction_pct = self._reduction_result.total_reduction_pct if self._reduction_result else 0.0

            os_config = OffsetStrategyConfig(
                baseline_total_tco2e=baseline.total_tco2e,
                reduction_achieved_tco2e=reduction_tco2e,
                reduction_achieved_pct=reduction_pct,
                long_term_target_year=config.long_term_target_year,
                quality_minimum_score=config.quality_minimum_score,
                max_nature_based_pct=config.max_nature_based_pct,
                vcmi_target_claim=config.vcmi_target_claim,
                entity_id=config.entity_id,
                tenant_id=config.tenant_id,
            )

            wf = OffsetStrategyWorkflow()
            self._offset_result = await wf.execute(os_config)

            outputs["offset_status"] = self._offset_result.status.value
            outputs["residual_tco2e"] = self._offset_result.residual_budget.residual_tco2e
            outputs["portfolio_volume_tco2e"] = self._offset_result.portfolio_design.total_volume_tco2e
            outputs["portfolio_cost_usd"] = self._offset_result.portfolio_design.total_estimated_cost_usd
            outputs["sbti_compliant"] = self._offset_result.compliance_status.sbti_compliant
            outputs["vcmi_claim"] = self._offset_result.compliance_status.vcmi_claim_eligible

            status = PhaseStatus.COMPLETED
        except Exception as exc:
            self.logger.error("Offsets phase failed: %s", exc, exc_info=True)
            outputs["error"] = str(exc)
            warnings.append(str(exc))
            status = PhaseStatus.FAILED

        elapsed = (utcnow() - started).total_seconds()
        return PhaseResult(
            phase_name="offsets",
            status=status,
            duration_seconds=round(elapsed, 4),
            outputs=outputs,
            warnings=warnings,
            provenance_hash=_compute_hash(json.dumps(outputs, sort_keys=True, default=str)),
        )

    # -------------------------------------------------------------------------
    # Phase 5: Scorecard
    # -------------------------------------------------------------------------

    async def _phase_scorecard(self, config: FullAssessmentConfig) -> PhaseResult:
        """Calculate net-zero maturity scorecard."""
        started = utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []
        dimensions: List[ScorecardDimension] = []

        # Dimension 1: Baseline Quality
        d1 = self._score_baseline_quality(config)
        dimensions.append(d1)

        # Dimension 2: Target Ambition
        d2 = self._score_target_ambition(config)
        dimensions.append(d2)

        # Dimension 3: Reduction Plan
        d3 = self._score_reduction_plan()
        dimensions.append(d3)

        # Dimension 4: Scope 3 Engagement
        d4 = self._score_scope3_engagement(config)
        dimensions.append(d4)

        # Dimension 5: Governance
        d5 = self._score_governance(config)
        dimensions.append(d5)

        # Dimension 6: Offset Strategy
        d6 = self._score_offset_strategy()
        dimensions.append(d6)

        # Dimension 7: Reporting Transparency
        d7 = self._score_reporting_transparency(config)
        dimensions.append(d7)

        total_score = sum(d.score for d in dimensions)
        maturity = self._determine_maturity_level(total_score)

        strengths = [d.dimension for d in dimensions if d.score >= d.max_score * 0.7]
        improvements = [d.dimension for d in dimensions if d.score < d.max_score * 0.5]

        self._scorecard = NetZeroScorecard(
            total_score=round(total_score, 1),
            maturity_level=maturity,
            dimensions=dimensions,
            strengths=strengths,
            improvement_areas=improvements,
        )

        outputs["total_score"] = self._scorecard.total_score
        outputs["maturity_level"] = maturity.value
        outputs["strengths"] = len(strengths)
        outputs["improvements"] = len(improvements)

        elapsed = (utcnow() - started).total_seconds()
        self.logger.info("Scorecard: %.1f/100 (%s)", total_score, maturity.value)
        return PhaseResult(
            phase_name="scorecard",
            status=PhaseStatus.COMPLETED,
            duration_seconds=round(elapsed, 4),
            outputs=outputs,
            warnings=warnings,
            provenance_hash=_compute_hash(json.dumps(outputs, sort_keys=True, default=str)),
        )

    def _score_baseline_quality(self, config: FullAssessmentConfig) -> ScorecardDimension:
        """Score baseline completeness and quality."""
        dim_info = MATURITY_DIMENSIONS["baseline_quality"]
        max_score = dim_info["weight"]
        score = 0.0
        recs: List[str] = []

        if self._onboarding_result:
            b = self._onboarding_result.baseline
            dq = self._onboarding_result.data_quality_report
            has_s1 = b.scope1_total_tco2e > 0
            has_s2 = b.scope2_location_tco2e > 0
            has_s3 = b.scope3_total_tco2e > 0

            if has_s1 and has_s2 and has_s3:
                score = max_score
                assessment = "complete_s1_s2_s3"
            elif has_s1 and has_s2:
                score = 10
                assessment = "complete_s1_s2"
                recs.append("Include Scope 3 emissions for a complete baseline")
            else:
                score = 5
                assessment = "partial"
                recs.append("Complete Scope 1 and Scope 2 baseline data")

            # Quality adjustment
            if dq.overall_score > 3.5:
                score = max(score - 3, 0)
                recs.append(f"Improve data quality (score: {dq.overall_score:.1f}/5)")
        else:
            assessment = "none"
            recs.append("Establish a GHG emissions baseline")

        return ScorecardDimension(
            dimension="baseline_quality",
            description=dim_info["description"],
            score=round(score, 1),
            max_score=max_score,
            assessment=assessment,
            recommendations=recs,
        )

    def _score_target_ambition(self, config: FullAssessmentConfig) -> ScorecardDimension:
        """Score target ambition and validity."""
        dim_info = MATURITY_DIMENSIONS["target_ambition"]
        max_score = dim_info["weight"]
        score = 0.0
        recs: List[str] = []

        if self._target_result:
            valid = self._target_result.validation_results.overall_valid
            ambition = self._target_result.pathway.ambition_level

            if valid and ambition == "1.5C":
                score = max_score
                assessment = "sbti_nz_validated"
            elif valid:
                score = 15
                assessment = "sbti_nt_validated"
            elif ambition == "1.5C":
                score = 10
                assessment = "internal_15c"
                recs.append("Submit targets for SBTi validation")
            else:
                score = 5
                assessment = "internal_other"
                recs.append("Increase ambition to 1.5C alignment")
        else:
            assessment = "none"
            recs.append("Set science-based emission reduction targets")

        return ScorecardDimension(
            dimension="target_ambition",
            description=dim_info["description"],
            score=round(score, 1),
            max_score=max_score,
            assessment=assessment,
            recommendations=recs,
        )

    def _score_reduction_plan(self) -> ScorecardDimension:
        """Score comprehensiveness of reduction roadmap."""
        dim_info = MATURITY_DIMENSIONS["reduction_plan"]
        max_score = dim_info["weight"]
        score = 0.0
        recs: List[str] = []

        if self._reduction_result:
            actions = len(self._reduction_result.prioritized_actions)
            has_macc = len(self._reduction_result.macc_data) > 0
            has_roadmap = len(self._reduction_result.roadmap) > 0

            if actions >= 5 and has_macc and has_roadmap:
                score = max_score
                assessment = "detailed_macc_phased"
            elif actions >= 3:
                score = 12
                assessment = "identified_actions"
                recs.append("Expand abatement actions and add financial analysis")
            else:
                score = 6
                assessment = "general_plan"
                recs.append("Identify and prioritise specific reduction actions")
        else:
            assessment = "none"
            recs.append("Develop a comprehensive reduction roadmap with cost analysis")

        return ScorecardDimension(
            dimension="reduction_plan",
            description=dim_info["description"],
            score=round(score, 1),
            max_score=max_score,
            assessment=assessment,
            recommendations=recs,
        )

    def _score_scope3_engagement(self, config: FullAssessmentConfig) -> ScorecardDimension:
        """Score Scope 3 data quality and supplier engagement."""
        dim_info = MATURITY_DIMENSIONS["scope3_engagement"]
        max_score = dim_info["weight"]
        score = 0.0
        recs: List[str] = []

        if self._onboarding_result and self._onboarding_result.baseline.scope3_total_tco2e > 0:
            if config.primary_supplier_data:
                score = max_score
                assessment = "primary_data_top_suppliers"
            elif len(self._onboarding_result.baseline.scope3_by_category) >= 5:
                score = 10
                assessment = "spend_based_all_cats"
                recs.append("Collect primary data from top suppliers to improve accuracy")
            else:
                score = 5
                assessment = "partial_scope3"
                recs.append("Expand Scope 3 coverage to all relevant categories")
        else:
            assessment = "none"
            recs.append("Establish Scope 3 emissions inventory and begin supplier engagement")

        return ScorecardDimension(
            dimension="scope3_engagement",
            description=dim_info["description"],
            score=round(score, 1),
            max_score=max_score,
            assessment=assessment,
            recommendations=recs,
        )

    def _score_governance(self, config: FullAssessmentConfig) -> ScorecardDimension:
        """Score governance and accountability."""
        dim_info = MATURITY_DIMENSIONS["governance"]
        max_score = dim_info["weight"]
        score = 0.0
        recs: List[str] = []

        if config.has_board_oversight:
            score = max_score
            assessment = "board_oversight_kpis"
        elif config.has_public_reporting:
            score = 6
            assessment = "management_oversight"
            recs.append("Elevate climate governance to board level with KPI accountability")
        else:
            score = 3
            assessment = "informal"
            recs.append("Formalise climate governance with board oversight and management KPIs")

        return ScorecardDimension(
            dimension="governance",
            description=dim_info["description"],
            score=round(score, 1),
            max_score=max_score,
            assessment=assessment,
            recommendations=recs,
        )

    def _score_offset_strategy(self) -> ScorecardDimension:
        """Score offset/BVCM strategy quality."""
        dim_info = MATURITY_DIMENSIONS["offset_strategy"]
        max_score = dim_info["weight"]
        score = 0.0
        recs: List[str] = []

        if self._offset_result:
            portfolio = self._offset_result.portfolio_design
            if portfolio.removal_share_pct >= 50:
                score = max_score
                assessment = "high_quality_removals"
            elif portfolio.total_volume_tco2e > 0 and portfolio.diversification_count >= 2:
                score = 7
                assessment = "mixed_portfolio"
                recs.append("Increase share of carbon removal credits for long-term alignment")
            elif portfolio.total_volume_tco2e > 0:
                score = 4
                assessment = "avoidance_only"
                recs.append("Shift from avoidance to removal-based credits per Oxford Principles")
            else:
                assessment = "none"
                recs.append("Develop a carbon credit strategy for residual emissions")
        else:
            assessment = "none"
            recs.append("Develop a beyond-value-chain mitigation strategy")

        return ScorecardDimension(
            dimension="offset_strategy",
            description=dim_info["description"],
            score=round(score, 1),
            max_score=max_score,
            assessment=assessment,
            recommendations=recs,
        )

    def _score_reporting_transparency(self, config: FullAssessmentConfig) -> ScorecardDimension:
        """Score reporting and public disclosure."""
        dim_info = MATURITY_DIMENSIONS["reporting_transparency"]
        max_score = dim_info["weight"]
        score = 0.0
        recs: List[str] = []

        if config.has_cdp_disclosure:
            score = max_score
            assessment = "cdp_sbti_public"
        elif config.has_public_reporting:
            score = 6
            assessment = "annual_report"
            recs.append("Disclose through CDP and commit to SBTi publicly")
        else:
            score = 3
            assessment = "internal_only"
            recs.append("Publish annual sustainability report with GHG emissions data")

        return ScorecardDimension(
            dimension="reporting_transparency",
            description=dim_info["description"],
            score=round(score, 1),
            max_score=max_score,
            assessment=assessment,
            recommendations=recs,
        )

    def _determine_maturity_level(self, score: float) -> MaturityLevel:
        """Map total score to maturity level."""
        if score >= 81:
            return MaturityLevel.LEADING
        elif score >= 61:
            return MaturityLevel.ADVANCED
        elif score >= 41:
            return MaturityLevel.ESTABLISHED
        elif score >= 21:
            return MaturityLevel.DEVELOPING
        else:
            return MaturityLevel.NASCENT

    # -------------------------------------------------------------------------
    # Phase 6: Strategy
    # -------------------------------------------------------------------------

    async def _phase_strategy(self, config: FullAssessmentConfig) -> PhaseResult:
        """Compile all outputs into a unified net-zero strategy document."""
        started = utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []

        baseline = self._onboarding_result.baseline if self._onboarding_result else ScopeBreakdown()
        base_year = config.onboarding_input.config.base_year

        # Near-term / long-term target details
        nt_year = 0
        nt_reduction = 0.0
        lt_year = config.long_term_target_year
        lt_reduction = 0.0
        pathway_name = ""
        ambition = ""

        if self._target_result:
            if self._target_result.near_term_target:
                nt_year = self._target_result.near_term_target.target_year
                nt_reduction = self._target_result.near_term_target.reduction_pct
            if self._target_result.long_term_target:
                lt_year = self._target_result.long_term_target.target_year
                lt_reduction = self._target_result.long_term_target.reduction_pct
            pathway_name = self._target_result.pathway.pathway
            ambition = self._target_result.pathway.ambition_level

        # Reduction details
        action_count = 0
        total_abatement = 0.0
        total_capex = 0.0
        if self._reduction_result:
            action_count = len(self._reduction_result.prioritized_actions)
            total_abatement = self._reduction_result.total_reduction_tco2e
            total_capex = self._reduction_result.total_capex_usd

        # Offset details
        residual = 0.0
        offset_cost = 0.0
        removal_share = 0.0
        sbti_ok = False
        vcmi_claim: Optional[str] = None
        if self._offset_result:
            residual = self._offset_result.residual_budget.residual_tco2e
            offset_cost = self._offset_result.portfolio_design.total_estimated_cost_usd
            removal_share = self._offset_result.portfolio_design.removal_share_pct
            sbti_ok = self._offset_result.compliance_status.sbti_compliant
            vcmi_claim = self._offset_result.compliance_status.vcmi_claim_eligible

        # Key actions (top 5 reduction actions)
        key_actions: List[str] = []
        if self._reduction_result:
            for act in self._reduction_result.prioritized_actions[:5]:
                key_actions.append(
                    f"[{act.priority_rank}] {act.name} - "
                    f"{act.reduction_tco2e:.0f} tCO2e reduction, "
                    f"${act.total_capex_usd:,.0f} capex"
                )

        # Key risks
        key_risks = self._identify_key_risks(config)

        self._strategy = StrategySummary(
            assessment_date=utcnow().strftime("%Y-%m-%d"),
            base_year=base_year,
            baseline_total_tco2e=round(baseline.total_tco2e, 4),
            scope1_tco2e=round(baseline.scope1_total_tco2e, 4),
            scope2_tco2e=round(baseline.scope2_location_tco2e, 4),
            scope3_tco2e=round(baseline.scope3_total_tco2e, 4),
            near_term_target_year=nt_year,
            near_term_reduction_pct=round(nt_reduction, 2),
            long_term_target_year=lt_year,
            long_term_reduction_pct=round(lt_reduction, 2),
            pathway=pathway_name,
            ambition_level=ambition,
            prioritized_action_count=action_count,
            total_abatement_tco2e=round(total_abatement, 4),
            total_abatement_capex_usd=round(total_capex, 2),
            residual_tco2e=round(residual, 4),
            offset_portfolio_cost_usd=round(offset_cost, 2),
            offset_removal_share_pct=round(removal_share, 2),
            maturity_score=self._scorecard.total_score,
            maturity_level=self._scorecard.maturity_level.value,
            sbti_compliant=sbti_ok,
            vcmi_claim=vcmi_claim,
            key_actions=key_actions,
            key_risks=key_risks,
        )

        outputs["baseline_tco2e"] = self._strategy.baseline_total_tco2e
        outputs["near_term_reduction_pct"] = self._strategy.near_term_reduction_pct
        outputs["long_term_reduction_pct"] = self._strategy.long_term_reduction_pct
        outputs["maturity_score"] = self._strategy.maturity_score
        outputs["maturity_level"] = self._strategy.maturity_level
        outputs["key_action_count"] = len(key_actions)
        outputs["key_risk_count"] = len(key_risks)

        elapsed = (utcnow() - started).total_seconds()
        self.logger.info("Strategy document compiled")
        return PhaseResult(
            phase_name="strategy",
            status=PhaseStatus.COMPLETED,
            duration_seconds=round(elapsed, 4),
            outputs=outputs,
            warnings=warnings,
            provenance_hash=_compute_hash(json.dumps(outputs, sort_keys=True, default=str)),
        )

    def _identify_key_risks(self, config: FullAssessmentConfig) -> List[str]:
        """Identify key strategic risks from the assessment."""
        risks: List[str] = []

        # Data quality risk
        if self._onboarding_result:
            dq = self._onboarding_result.data_quality_report.overall_score
            if dq > 3.0:
                risks.append(
                    f"Data quality risk: overall score {dq:.1f}/5 may undermine "
                    "baseline accuracy and progress tracking."
                )

        # Target validation risk
        if self._target_result:
            if not self._target_result.validation_results.overall_valid:
                risks.append(
                    "Target risk: targets do not fully meet SBTi Net-Zero Standard criteria. "
                    "Review validation findings."
                )

        # Cost risk
        if self._reduction_result and self._reduction_result.total_capex_usd > 0:
            if config.budget_constraint_usd and self._reduction_result.total_capex_usd > config.budget_constraint_usd:
                risks.append(
                    "Budget risk: reduction roadmap capex exceeds budget constraint."
                )

        # Scope 3 coverage risk
        baseline = self._onboarding_result.baseline if self._onboarding_result else ScopeBreakdown()
        total = baseline.total_tco2e or 1.0
        if baseline.scope3_total_tco2e / total > 0.5 and not config.primary_supplier_data:
            risks.append(
                "Scope 3 risk: Scope 3 dominates emissions but relies on spend-based estimates. "
                "Supplier-specific data needed for credibility."
            )

        # Offset compliance risk
        if self._offset_result:
            if not self._offset_result.compliance_status.sbti_compliant:
                risks.append(
                    "Offset risk: current strategy does not meet SBTi BVCM requirements."
                )

        if not risks:
            risks.append("No critical risks identified. Continue monitoring annually.")

        return risks
