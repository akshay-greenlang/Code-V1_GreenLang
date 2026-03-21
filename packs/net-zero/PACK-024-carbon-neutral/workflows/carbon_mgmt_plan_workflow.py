# -*- coding: utf-8 -*-
"""
Carbon Management Plan Workflow
====================================

5-phase workflow for developing a carbon management plan within PACK-024
Carbon Neutral Pack.  Creates the reduction-first strategy required by
PAS 2060 before any carbon credits can be applied to the neutralization
balance.

Phases:
    1. BaselineAnalysis     -- Analyze current emissions profile and trends
    2. ReductionTargeting   -- Set reduction targets aligned with science
    3. AbatementPlanning    -- Identify and prioritize reduction actions
    4. ResidualForecasting  -- Forecast residual emissions after reductions
    5. PlanCompilation      -- Compile management plan with timelines

Regulatory references:
    - PAS 2060:2014 Carbon Neutrality (Section 7: Carbon Management Plan)
    - ISO 14064-1:2018 (Planning requirements)
    - GHG Protocol Mitigation Goal Standard (2014)
    - VCMI Claims Code of Practice (2023)

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


class MgmtPlanPhase(str, Enum):
    BASELINE_ANALYSIS = "baseline_analysis"
    REDUCTION_TARGETING = "reduction_targeting"
    ABATEMENT_PLANNING = "abatement_planning"
    RESIDUAL_FORECASTING = "residual_forecasting"
    PLAN_COMPILATION = "plan_compilation"


class ReductionStrategy(str, Enum):
    ENERGY_EFFICIENCY = "energy_efficiency"
    FUEL_SWITCHING = "fuel_switching"
    RENEWABLE_ENERGY = "renewable_energy"
    PROCESS_OPTIMIZATION = "process_optimization"
    SUPPLY_CHAIN = "supply_chain"
    BEHAVIORAL_CHANGE = "behavioral_change"
    TECHNOLOGY_UPGRADE = "technology_upgrade"
    CIRCULAR_ECONOMY = "circular_economy"
    ELECTRIFICATION = "electrification"
    GREEN_PROCUREMENT = "green_procurement"


class AbatementPriority(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    OPTIONAL = "optional"


class TimeHorizon(str, Enum):
    IMMEDIATE = "immediate"  # 0-1 years
    SHORT_TERM = "short_term"  # 1-3 years
    MEDIUM_TERM = "medium_term"  # 3-5 years
    LONG_TERM = "long_term"  # 5-10 years


# =============================================================================
# REFERENCE DATA
# =============================================================================

# PAS 2060 minimum annual reduction rate
PAS2060_MIN_ANNUAL_REDUCTION_PCT = 2.0

# VCMI silver/gold claim thresholds
VCMI_SILVER_REDUCTION_PCT = 20.0
VCMI_GOLD_REDUCTION_PCT = 50.0

# Typical abatement cost ranges by strategy (USD/tCO2e)
ABATEMENT_COST_RANGES: Dict[str, Dict[str, float]] = {
    "energy_efficiency": {"min": -50.0, "max": 20.0, "typical": 0.0},
    "fuel_switching": {"min": 10.0, "max": 80.0, "typical": 40.0},
    "renewable_energy": {"min": 5.0, "max": 60.0, "typical": 25.0},
    "process_optimization": {"min": -30.0, "max": 50.0, "typical": 15.0},
    "supply_chain": {"min": 20.0, "max": 150.0, "typical": 60.0},
    "behavioral_change": {"min": 0.0, "max": 10.0, "typical": 5.0},
    "technology_upgrade": {"min": 30.0, "max": 200.0, "typical": 80.0},
    "circular_economy": {"min": -20.0, "max": 100.0, "typical": 30.0},
    "electrification": {"min": 15.0, "max": 120.0, "typical": 50.0},
    "green_procurement": {"min": 5.0, "max": 40.0, "typical": 20.0},
}

# Typical reduction potential by strategy (% of addressable emissions)
REDUCTION_POTENTIAL: Dict[str, float] = {
    "energy_efficiency": 15.0,
    "fuel_switching": 25.0,
    "renewable_energy": 40.0,
    "process_optimization": 10.0,
    "supply_chain": 20.0,
    "behavioral_change": 5.0,
    "technology_upgrade": 30.0,
    "circular_economy": 10.0,
    "electrification": 35.0,
    "green_procurement": 15.0,
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


class ReductionTarget(BaseModel):
    target_id: str = Field(default="")
    description: str = Field(default="")
    scope: str = Field(default="")
    base_year: int = Field(default=2020)
    target_year: int = Field(default=2030)
    reduction_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    annual_reduction_rate: float = Field(default=0.0, ge=0.0)
    base_year_emissions_tco2e: float = Field(default=0.0, ge=0.0)
    target_emissions_tco2e: float = Field(default=0.0, ge=0.0)
    is_science_aligned: bool = Field(default=False)
    alignment_framework: str = Field(default="")


class AbatementAction(BaseModel):
    action_id: str = Field(default="")
    name: str = Field(default="")
    strategy: ReductionStrategy = Field(default=ReductionStrategy.ENERGY_EFFICIENCY)
    priority: AbatementPriority = Field(default=AbatementPriority.MEDIUM)
    time_horizon: TimeHorizon = Field(default=TimeHorizon.SHORT_TERM)
    reduction_potential_tco2e: float = Field(default=0.0, ge=0.0)
    reduction_potential_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    cost_usd: float = Field(default=0.0)
    cost_per_tco2e: float = Field(default=0.0)
    payback_years: float = Field(default=0.0, ge=0.0)
    implementation_start: str = Field(default="")
    implementation_end: str = Field(default="")
    responsible_party: str = Field(default="")
    status: str = Field(default="planned")
    dependencies: List[str] = Field(default_factory=list)
    co_benefits: List[str] = Field(default_factory=list)
    risks: List[str] = Field(default_factory=list)


class ResidualProfile(BaseModel):
    year: int = Field(...)
    projected_emissions_tco2e: float = Field(default=0.0, ge=0.0)
    reductions_achieved_tco2e: float = Field(default=0.0, ge=0.0)
    residual_emissions_tco2e: float = Field(default=0.0, ge=0.0)
    credits_needed_tco2e: float = Field(default=0.0, ge=0.0)
    reduction_pct_from_base: float = Field(default=0.0, ge=0.0, le=100.0)
    cumulative_reduction_pct: float = Field(default=0.0, ge=0.0, le=100.0)


class MgmtPlanTimeline(BaseModel):
    start_year: int = Field(default=2025)
    end_year: int = Field(default=2030)
    milestones: List[Dict[str, Any]] = Field(default_factory=list)
    review_frequency: str = Field(default="annual")
    update_triggers: List[str] = Field(default_factory=list)


class CarbonMgmtPlanConfig(BaseModel):
    org_name: str = Field(default="")
    reporting_year: int = Field(default=2025, ge=2015, le=2050)
    base_year: int = Field(default=2020, ge=2015, le=2050)
    base_year_emissions_tco2e: float = Field(default=0.0, ge=0.0)
    scope1_tco2e: float = Field(default=0.0, ge=0.0)
    scope2_tco2e: float = Field(default=0.0, ge=0.0)
    scope3_tco2e: float = Field(default=0.0, ge=0.0)
    target_year: int = Field(default=2030, ge=2025, le=2060)
    target_reduction_pct: float = Field(default=50.0, ge=0.0, le=100.0)
    strategies: List[ReductionStrategy] = Field(
        default_factory=lambda: [
            ReductionStrategy.ENERGY_EFFICIENCY,
            ReductionStrategy.RENEWABLE_ENERGY,
        ]
    )
    pas2060_compliance: bool = Field(default=True)
    vcmi_claim_target: str = Field(default="silver")
    budget_usd: float = Field(default=0.0, ge=0.0)
    entity_id: str = Field(default="")
    tenant_id: str = Field(default="")


class CarbonMgmtPlanResult(BaseModel):
    workflow_id: str = Field(...)
    workflow_name: str = Field(default="carbon_mgmt_plan")
    status: WorkflowStatus = Field(...)
    phases: List[PhaseResult] = Field(default_factory=list)
    total_duration_seconds: float = Field(default=0.0)
    reduction_targets: List[ReductionTarget] = Field(default_factory=list)
    abatement_actions: List[AbatementAction] = Field(default_factory=list)
    residual_profile: List[ResidualProfile] = Field(default_factory=list)
    timeline: Optional[MgmtPlanTimeline] = Field(None)
    total_abatement_potential_tco2e: float = Field(default=0.0)
    total_investment_usd: float = Field(default=0.0)
    weighted_avg_cost_per_tco2e: float = Field(default=0.0)
    meets_pas2060_reduction: bool = Field(default=False)
    vcmi_claim_eligible: str = Field(default="none")
    provenance_hash: str = Field(default="")


# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================


class CarbonMgmtPlanWorkflow:
    """
    5-phase carbon management plan workflow for PACK-024.

    PAS 2060 requires a documented carbon management plan with reduction
    targets before carbon credits can be used for neutralization.  This
    workflow develops the reduction-first strategy, abatement plan,
    residual emissions forecast, and implementation timeline.

    Attributes:
        workflow_id: Unique execution identifier.
    """

    def __init__(self) -> None:
        self.workflow_id: str = _new_uuid()
        self._phase_results: List[PhaseResult] = []
        self._targets: List[ReductionTarget] = []
        self._actions: List[AbatementAction] = []
        self._residuals: List[ResidualProfile] = []
        self._timeline: Optional[MgmtPlanTimeline] = None
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    async def execute(self, config: CarbonMgmtPlanConfig) -> CarbonMgmtPlanResult:
        """Execute the 5-phase carbon management plan workflow."""
        started_at = _utcnow()
        self.logger.info(
            "Starting carbon mgmt plan %s, org=%s, target_year=%d",
            self.workflow_id, config.org_name, config.target_year,
        )
        self._phase_results = []
        overall_status = WorkflowStatus.RUNNING

        try:
            phase1 = await self._phase_baseline_analysis(config)
            self._phase_results.append(phase1)
            if phase1.status == PhaseStatus.FAILED:
                raise ValueError("Baseline analysis failed")

            phase2 = await self._phase_reduction_targeting(config)
            self._phase_results.append(phase2)

            phase3 = await self._phase_abatement_planning(config)
            self._phase_results.append(phase3)

            phase4 = await self._phase_residual_forecasting(config)
            self._phase_results.append(phase4)

            phase5 = await self._phase_plan_compilation(config)
            self._phase_results.append(phase5)

            failed = [p for p in self._phase_results if p.status == PhaseStatus.FAILED]
            overall_status = WorkflowStatus.COMPLETED if not failed else WorkflowStatus.PARTIAL

        except Exception as exc:
            self.logger.error("Carbon mgmt plan failed: %s", exc, exc_info=True)
            overall_status = WorkflowStatus.FAILED
            self._phase_results.append(PhaseResult(
                phase_name="error", status=PhaseStatus.FAILED, errors=[str(exc)],
            ))

        elapsed = (_utcnow() - started_at).total_seconds()
        total_abatement = sum(a.reduction_potential_tco2e for a in self._actions)
        total_investment = sum(a.cost_usd for a in self._actions)
        avg_cost = (total_investment / max(total_abatement, 1.0))

        # Check PAS 2060 minimum reduction
        total_base = config.base_year_emissions_tco2e or 1.0
        years = max(config.target_year - config.base_year, 1)
        annual_rate = (total_abatement / total_base) / years * 100.0
        meets_pas2060 = annual_rate >= PAS2060_MIN_ANNUAL_REDUCTION_PCT

        # Check VCMI claim eligibility
        total_reduction_pct = (total_abatement / total_base) * 100.0
        vcmi_claim = "none"
        if total_reduction_pct >= VCMI_GOLD_REDUCTION_PCT:
            vcmi_claim = "gold"
        elif total_reduction_pct >= VCMI_SILVER_REDUCTION_PCT:
            vcmi_claim = "silver"

        result = CarbonMgmtPlanResult(
            workflow_id=self.workflow_id,
            status=overall_status,
            phases=self._phase_results,
            total_duration_seconds=round(elapsed, 4),
            reduction_targets=self._targets,
            abatement_actions=self._actions,
            residual_profile=self._residuals,
            timeline=self._timeline,
            total_abatement_potential_tco2e=round(total_abatement, 2),
            total_investment_usd=round(total_investment, 2),
            weighted_avg_cost_per_tco2e=round(avg_cost, 2),
            meets_pas2060_reduction=meets_pas2060,
            vcmi_claim_eligible=vcmi_claim,
        )
        result.provenance_hash = _compute_hash(
            result.model_dump_json(exclude={"provenance_hash"})
        )
        return result

    # -------------------------------------------------------------------------
    # Phase 1: Baseline Analysis
    # -------------------------------------------------------------------------

    async def _phase_baseline_analysis(self, config: CarbonMgmtPlanConfig) -> PhaseResult:
        """Analyze current emissions profile and historical trends."""
        started = _utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []
        errors: List[str] = []

        total = config.scope1_tco2e + config.scope2_tco2e + config.scope3_tco2e
        if total <= 0:
            if config.base_year_emissions_tco2e > 0:
                total = config.base_year_emissions_tco2e
            else:
                errors.append("No emissions data provided for baseline analysis")

        outputs["base_year"] = config.base_year
        outputs["reporting_year"] = config.reporting_year
        outputs["scope1_tco2e"] = round(config.scope1_tco2e, 2)
        outputs["scope2_tco2e"] = round(config.scope2_tco2e, 2)
        outputs["scope3_tco2e"] = round(config.scope3_tco2e, 2)
        outputs["total_emissions_tco2e"] = round(total, 2)
        outputs["scope1_pct"] = round((config.scope1_tco2e / max(total, 1.0)) * 100.0, 1)
        outputs["scope2_pct"] = round((config.scope2_tco2e / max(total, 1.0)) * 100.0, 1)
        outputs["scope3_pct"] = round((config.scope3_tco2e / max(total, 1.0)) * 100.0, 1)

        # Intensity metrics
        outputs["years_since_base"] = config.reporting_year - config.base_year
        if config.base_year_emissions_tco2e > 0:
            change = ((total - config.base_year_emissions_tco2e) / config.base_year_emissions_tco2e) * 100.0
            outputs["change_from_base_pct"] = round(change, 1)
        else:
            outputs["change_from_base_pct"] = 0.0

        status = PhaseStatus.FAILED if errors else PhaseStatus.COMPLETED
        elapsed = (_utcnow() - started).total_seconds()
        return PhaseResult(
            phase_name=MgmtPlanPhase.BASELINE_ANALYSIS.value,
            status=status, duration_seconds=round(elapsed, 4),
            outputs=outputs, warnings=warnings, errors=errors,
            provenance_hash=_compute_hash(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 2: Reduction Targeting
    # -------------------------------------------------------------------------

    async def _phase_reduction_targeting(self, config: CarbonMgmtPlanConfig) -> PhaseResult:
        """Set reduction targets aligned with science-based pathways."""
        started = _utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []
        errors: List[str] = []

        years = max(config.target_year - config.base_year, 1)
        total_base = config.base_year_emissions_tco2e or (
            config.scope1_tco2e + config.scope2_tco2e + config.scope3_tco2e
        )
        target_emissions = total_base * (1.0 - config.target_reduction_pct / 100.0)
        annual_rate = config.target_reduction_pct / years

        # Scope-level targets
        scope_splits = [
            ("scope_1_2", config.scope1_tco2e + config.scope2_tco2e, min(config.target_reduction_pct + 5, 100)),
            ("scope_3", config.scope3_tco2e, max(config.target_reduction_pct - 5, 0)),
        ]

        for scope_label, base_val, red_pct in scope_splits:
            target = ReductionTarget(
                target_id=_new_uuid(),
                description=f"{scope_label.replace('_', ' ').title()} reduction target",
                scope=scope_label,
                base_year=config.base_year,
                target_year=config.target_year,
                reduction_pct=red_pct,
                annual_reduction_rate=round(red_pct / years, 2),
                base_year_emissions_tco2e=round(base_val, 2),
                target_emissions_tco2e=round(base_val * (1.0 - red_pct / 100.0), 2),
                is_science_aligned=annual_rate >= 4.2,
                alignment_framework="SBTi 1.5C" if annual_rate >= 4.2 else "PAS 2060",
            )
            self._targets.append(target)

        outputs["targets_defined"] = len(self._targets)
        outputs["overall_reduction_pct"] = config.target_reduction_pct
        outputs["annual_reduction_rate"] = round(annual_rate, 2)
        outputs["target_emissions_tco2e"] = round(target_emissions, 2)
        outputs["is_science_aligned"] = annual_rate >= 4.2

        if annual_rate < PAS2060_MIN_ANNUAL_REDUCTION_PCT:
            warnings.append(
                f"Annual reduction rate {annual_rate:.1f}% is below "
                f"PAS 2060 minimum of {PAS2060_MIN_ANNUAL_REDUCTION_PCT}%"
            )

        status = PhaseStatus.COMPLETED
        elapsed = (_utcnow() - started).total_seconds()
        return PhaseResult(
            phase_name=MgmtPlanPhase.REDUCTION_TARGETING.value,
            status=status, duration_seconds=round(elapsed, 4),
            outputs=outputs, warnings=warnings, errors=errors,
            provenance_hash=_compute_hash(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 3: Abatement Planning
    # -------------------------------------------------------------------------

    async def _phase_abatement_planning(self, config: CarbonMgmtPlanConfig) -> PhaseResult:
        """Identify and prioritize emission reduction actions."""
        started = _utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []
        errors: List[str] = []

        total_base = config.base_year_emissions_tco2e or (
            config.scope1_tco2e + config.scope2_tco2e + config.scope3_tco2e
        )
        actions: List[AbatementAction] = []

        for i, strategy in enumerate(config.strategies):
            cost_info = ABATEMENT_COST_RANGES.get(strategy.value, {"typical": 50.0})
            potential_pct = REDUCTION_POTENTIAL.get(strategy.value, 10.0)
            reduction_tco2e = total_base * (potential_pct / 100.0)
            cost_per_t = cost_info["typical"]
            total_cost = reduction_tco2e * max(cost_per_t, 0)
            payback = total_cost / max(reduction_tco2e * cost_per_t * 0.1, 1.0) if cost_per_t > 0 else 0.0

            action = AbatementAction(
                action_id=_new_uuid(),
                name=f"{strategy.value.replace('_', ' ').title()} Programme",
                strategy=strategy,
                priority=AbatementPriority.HIGH if i < 2 else AbatementPriority.MEDIUM,
                time_horizon=TimeHorizon.SHORT_TERM if i < 2 else TimeHorizon.MEDIUM_TERM,
                reduction_potential_tco2e=round(reduction_tco2e, 2),
                reduction_potential_pct=round(potential_pct, 1),
                cost_usd=round(total_cost, 2),
                cost_per_tco2e=round(cost_per_t, 2),
                payback_years=round(min(payback, 20.0), 1),
                status="planned",
                co_benefits=["Cost savings", "Regulatory compliance"],
            )
            actions.append(action)

        # Sort by cost-effectiveness (negative cost first = savings)
        actions.sort(key=lambda a: a.cost_per_tco2e)
        self._actions = actions

        total_abatement = sum(a.reduction_potential_tco2e for a in actions)
        total_cost = sum(a.cost_usd for a in actions)

        outputs["actions_count"] = len(actions)
        outputs["total_abatement_tco2e"] = round(total_abatement, 2)
        outputs["total_investment_usd"] = round(total_cost, 2)
        outputs["abatement_pct_of_base"] = round(
            (total_abatement / max(total_base, 1.0)) * 100.0, 1
        )

        if config.budget_usd > 0 and total_cost > config.budget_usd:
            warnings.append(
                f"Total investment ${total_cost:,.0f} exceeds budget ${config.budget_usd:,.0f}"
            )

        status = PhaseStatus.COMPLETED
        elapsed = (_utcnow() - started).total_seconds()
        return PhaseResult(
            phase_name=MgmtPlanPhase.ABATEMENT_PLANNING.value,
            status=status, duration_seconds=round(elapsed, 4),
            outputs=outputs, warnings=warnings, errors=errors,
            provenance_hash=_compute_hash(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 4: Residual Forecasting
    # -------------------------------------------------------------------------

    async def _phase_residual_forecasting(self, config: CarbonMgmtPlanConfig) -> PhaseResult:
        """Forecast residual emissions after planned reductions."""
        started = _utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []
        errors: List[str] = []

        total_base = config.base_year_emissions_tco2e or (
            config.scope1_tco2e + config.scope2_tco2e + config.scope3_tco2e
        )
        total_abatement = sum(a.reduction_potential_tco2e for a in self._actions)
        years = max(config.target_year - config.reporting_year, 1)

        residuals: List[ResidualProfile] = []
        for yr_offset in range(years + 1):
            year = config.reporting_year + yr_offset
            progress = yr_offset / max(years, 1)
            reductions = total_abatement * min(progress, 1.0)
            residual = max(total_base - reductions, 0.0)
            credits_needed = residual  # Carbon neutral requires offsetting all residual

            residuals.append(ResidualProfile(
                year=year,
                projected_emissions_tco2e=round(total_base, 2),
                reductions_achieved_tco2e=round(reductions, 2),
                residual_emissions_tco2e=round(residual, 2),
                credits_needed_tco2e=round(credits_needed, 2),
                reduction_pct_from_base=round((reductions / max(total_base, 1.0)) * 100.0, 1),
                cumulative_reduction_pct=round(progress * 100.0, 1),
            ))

        self._residuals = residuals

        final = residuals[-1] if residuals else None
        outputs["forecast_years"] = len(residuals)
        outputs["final_year"] = config.target_year
        outputs["final_residual_tco2e"] = round(final.residual_emissions_tco2e, 2) if final else 0.0
        outputs["final_credits_needed_tco2e"] = round(final.credits_needed_tco2e, 2) if final else 0.0
        outputs["total_credits_over_period_tco2e"] = round(
            sum(r.credits_needed_tco2e for r in residuals), 2
        )

        status = PhaseStatus.COMPLETED
        elapsed = (_utcnow() - started).total_seconds()
        return PhaseResult(
            phase_name=MgmtPlanPhase.RESIDUAL_FORECASTING.value,
            status=status, duration_seconds=round(elapsed, 4),
            outputs=outputs, warnings=warnings, errors=errors,
            provenance_hash=_compute_hash(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 5: Plan Compilation
    # -------------------------------------------------------------------------

    async def _phase_plan_compilation(self, config: CarbonMgmtPlanConfig) -> PhaseResult:
        """Compile management plan with milestones and timelines."""
        started = _utcnow()
        outputs: Dict[str, Any] = {}
        warnings: List[str] = []
        errors: List[str] = []

        milestones = []
        for yr in range(config.reporting_year, config.target_year + 1):
            milestone = {
                "year": yr,
                "type": "annual_review",
                "description": f"Annual review and progress assessment for {yr}",
            }
            milestones.append(milestone)

        # Add key milestones
        mid_year = config.reporting_year + (config.target_year - config.reporting_year) // 2
        milestones.append({
            "year": mid_year,
            "type": "mid_term_review",
            "description": "Mid-term strategy review and plan adjustment",
        })
        milestones.sort(key=lambda m: m["year"])

        self._timeline = MgmtPlanTimeline(
            start_year=config.reporting_year,
            end_year=config.target_year,
            milestones=milestones,
            review_frequency="annual",
            update_triggers=[
                "Significant change in emissions (>5%)",
                "Organizational restructuring",
                "Change in methodology or emission factors",
                "New regulatory requirements",
                "Material acquisition or divestiture",
            ],
        )

        outputs["plan_start_year"] = config.reporting_year
        outputs["plan_end_year"] = config.target_year
        outputs["milestones_count"] = len(milestones)
        outputs["review_frequency"] = "annual"
        outputs["actions_count"] = len(self._actions)
        outputs["targets_count"] = len(self._targets)
        outputs["pas2060_section_7_compliant"] = True

        status = PhaseStatus.COMPLETED
        elapsed = (_utcnow() - started).total_seconds()
        return PhaseResult(
            phase_name=MgmtPlanPhase.PLAN_COMPILATION.value,
            status=status, duration_seconds=round(elapsed, 4),
            outputs=outputs, warnings=warnings, errors=errors,
            provenance_hash=_compute_hash(outputs),
        )
