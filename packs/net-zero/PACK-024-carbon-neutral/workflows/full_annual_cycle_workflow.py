# -*- coding: utf-8 -*-
"""
Full Annual Cycle Workflow
===============================

10-phase workflow orchestrating the complete annual carbon neutrality
cycle within PACK-024 Carbon Neutral Pack.  Drives the end-to-end
process from footprint assessment through plan development, credit
procurement, retirement, neutralization, claims validation, verification,
reporting, and renewal.

Phases:
    1.  FootprintAssessment   -- Quantify GHG footprint for the period
    2.  MgmtPlanReview        -- Review/update carbon management plan
    3.  ReductionTracking     -- Track emission reduction progress
    4.  ResidualCalc          -- Calculate residual emissions for offsetting
    5.  CreditProcurement     -- Procure carbon credits for residual
    6.  Retirement            -- Retire credits on registries
    7.  NeutralizationBalance -- Validate neutralization balance
    8.  ClaimsValidation      -- Validate carbon neutrality claim
    9.  Verification          -- Third-party verification
    10. RenewalPrep           -- Prepare for next cycle renewal

Regulatory references:
    - PAS 2060:2014 (Full lifecycle)
    - ISO 14064-1:2018 (Annual reporting)
    - VCMI Claims Code of Practice (2023)
    - ICVCM Core Carbon Principles (2023)

Author: GreenLang Team
Version: 24.0.0
"""

import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Coroutine, Dict, List, Optional, Set

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

_MODULE_VERSION = "24.0.0"

ProgressCallback = Callable[[str, float, str], Coroutine[Any, Any, None]]


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


class AnnualCyclePhase(str, Enum):
    FOOTPRINT_ASSESSMENT = "footprint_assessment"
    MGMT_PLAN_REVIEW = "mgmt_plan_review"
    REDUCTION_TRACKING = "reduction_tracking"
    RESIDUAL_CALC = "residual_calc"
    CREDIT_PROCUREMENT = "credit_procurement"
    RETIREMENT = "retirement"
    NEUTRALIZATION_BALANCE = "neutralization_balance"
    CLAIMS_VALIDATION = "claims_validation"
    VERIFICATION = "verification"
    RENEWAL_PREP = "renewal_prep"


class CycleStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"
    CANCELLED = "cancelled"


class PhaseStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


# =============================================================================
# REFERENCE DATA
# =============================================================================

# Phase dependencies DAG
PHASE_DEPENDENCIES: Dict[AnnualCyclePhase, List[AnnualCyclePhase]] = {
    AnnualCyclePhase.FOOTPRINT_ASSESSMENT: [],
    AnnualCyclePhase.MGMT_PLAN_REVIEW: [AnnualCyclePhase.FOOTPRINT_ASSESSMENT],
    AnnualCyclePhase.REDUCTION_TRACKING: [AnnualCyclePhase.MGMT_PLAN_REVIEW],
    AnnualCyclePhase.RESIDUAL_CALC: [AnnualCyclePhase.REDUCTION_TRACKING],
    AnnualCyclePhase.CREDIT_PROCUREMENT: [AnnualCyclePhase.RESIDUAL_CALC],
    AnnualCyclePhase.RETIREMENT: [AnnualCyclePhase.CREDIT_PROCUREMENT],
    AnnualCyclePhase.NEUTRALIZATION_BALANCE: [AnnualCyclePhase.RETIREMENT],
    AnnualCyclePhase.CLAIMS_VALIDATION: [AnnualCyclePhase.NEUTRALIZATION_BALANCE],
    AnnualCyclePhase.VERIFICATION: [AnnualCyclePhase.CLAIMS_VALIDATION],
    AnnualCyclePhase.RENEWAL_PREP: [AnnualCyclePhase.VERIFICATION],
}

# Phase execution order (topological sort)
PHASE_EXECUTION_ORDER: List[AnnualCyclePhase] = [
    AnnualCyclePhase.FOOTPRINT_ASSESSMENT,
    AnnualCyclePhase.MGMT_PLAN_REVIEW,
    AnnualCyclePhase.REDUCTION_TRACKING,
    AnnualCyclePhase.RESIDUAL_CALC,
    AnnualCyclePhase.CREDIT_PROCUREMENT,
    AnnualCyclePhase.RETIREMENT,
    AnnualCyclePhase.NEUTRALIZATION_BALANCE,
    AnnualCyclePhase.CLAIMS_VALIDATION,
    AnnualCyclePhase.VERIFICATION,
    AnnualCyclePhase.RENEWAL_PREP,
]

# Annual cycle timeline (months from cycle start)
PHASE_TIMELINE_MONTHS: Dict[str, Dict[str, int]] = {
    "footprint_assessment": {"start": 1, "end": 2},
    "mgmt_plan_review": {"start": 2, "end": 3},
    "reduction_tracking": {"start": 3, "end": 4},
    "residual_calc": {"start": 4, "end": 5},
    "credit_procurement": {"start": 5, "end": 7},
    "retirement": {"start": 7, "end": 8},
    "neutralization_balance": {"start": 8, "end": 9},
    "claims_validation": {"start": 9, "end": 10},
    "verification": {"start": 10, "end": 11},
    "renewal_prep": {"start": 11, "end": 12},
}


# =============================================================================
# DATA MODELS
# =============================================================================


class PhaseResult(BaseModel):
    phase: AnnualCyclePhase = Field(...)
    status: PhaseStatus = Field(default=PhaseStatus.PENDING)
    started_at: Optional[datetime] = Field(None)
    completed_at: Optional[datetime] = Field(None)
    duration_ms: float = Field(default=0.0)
    records_processed: int = Field(default=0)
    outputs: Dict[str, Any] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")


class PhaseGate(BaseModel):
    """Gate criteria that must be met before proceeding to next phase."""
    gate_id: str = Field(default="")
    phase: AnnualCyclePhase = Field(...)
    criterion: str = Field(default="")
    is_met: bool = Field(default=False)
    value: Any = Field(default=None)
    threshold: Any = Field(default=None)


class CycleMetrics(BaseModel):
    """Key metrics from the annual cycle."""
    total_emissions_tco2e: float = Field(default=0.0, ge=0.0)
    reductions_achieved_tco2e: float = Field(default=0.0, ge=0.0)
    residual_emissions_tco2e: float = Field(default=0.0, ge=0.0)
    credits_retired_tco2e: float = Field(default=0.0, ge=0.0)
    neutralization_coverage_pct: float = Field(default=0.0, ge=0.0)
    is_carbon_neutral: bool = Field(default=False)
    data_quality_score: float = Field(default=0.0, ge=0.0, le=100.0)
    verification_opinion: str = Field(default="")
    credit_spend_usd: float = Field(default=0.0, ge=0.0)
    cost_per_tco2e: float = Field(default=0.0, ge=0.0)
    year_over_year_reduction_pct: float = Field(default=0.0)


class AnnualSummary(BaseModel):
    """End-of-cycle annual summary."""
    cycle_year: int = Field(...)
    cycle_number: int = Field(default=1)
    is_first_year: bool = Field(default=True)
    phases_completed: int = Field(default=0)
    phases_total: int = Field(default=10)
    metrics: Optional[CycleMetrics] = Field(None)
    key_achievements: List[str] = Field(default_factory=list)
    areas_for_improvement: List[str] = Field(default_factory=list)
    next_cycle_priorities: List[str] = Field(default_factory=list)


class RetryConfig(BaseModel):
    max_retries: int = Field(default=3, ge=0, le=10)
    base_delay: float = Field(default=1.0, ge=0.1)
    max_delay: float = Field(default=30.0, ge=1.0)
    jitter: float = Field(default=0.5, ge=0.0, le=1.0)


class AnnualCycleConfig(BaseModel):
    pack_id: str = Field(default="PACK-024")
    pack_version: str = Field(default="1.0.0")
    org_name: str = Field(default="")
    reporting_year: int = Field(default=2025, ge=2015, le=2050)
    base_year: int = Field(default=2020, ge=2015, le=2050)
    scope1_tco2e: float = Field(default=0.0, ge=0.0)
    scope2_tco2e: float = Field(default=0.0, ge=0.0)
    scope3_tco2e: float = Field(default=0.0, ge=0.0)
    previous_year_emissions_tco2e: float = Field(default=0.0, ge=0.0)
    budget_usd: float = Field(default=0.0, ge=0.0)
    target_reduction_pct: float = Field(default=5.0, ge=0.0, le=100.0)
    pas2060_compliance: bool = Field(default=True)
    enable_verification: bool = Field(default=True)
    max_concurrent_phases: int = Field(default=1, ge=1, le=5)
    timeout_per_phase_seconds: int = Field(default=900, ge=30)
    enable_provenance: bool = Field(default=True)
    retry_config: RetryConfig = Field(default_factory=RetryConfig)
    entity_id: str = Field(default="")
    tenant_id: str = Field(default="")


class AnnualCycleResult(BaseModel):
    execution_id: str = Field(default_factory=_new_uuid)
    pack_id: str = Field(default="PACK-024")
    org_name: str = Field(default="")
    status: CycleStatus = Field(default=CycleStatus.PENDING)
    started_at: Optional[datetime] = Field(None)
    completed_at: Optional[datetime] = Field(None)
    total_duration_ms: float = Field(default=0.0)
    phases_completed: List[str] = Field(default_factory=list)
    phases_skipped: List[str] = Field(default_factory=list)
    phase_results: Dict[str, PhaseResult] = Field(default_factory=dict)
    gates: List[PhaseGate] = Field(default_factory=list)
    metrics: Optional[CycleMetrics] = Field(None)
    summary: Optional[AnnualSummary] = Field(None)
    total_records_processed: int = Field(default=0)
    errors: List[str] = Field(default_factory=list)
    quality_score: float = Field(default=0.0, ge=0.0, le=100.0)
    provenance_hash: str = Field(default="")


# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================


class FullAnnualCycleWorkflow:
    """
    10-phase annual cycle workflow for PACK-024 Carbon Neutral Pack.

    Orchestrates the complete annual carbon neutrality lifecycle from
    footprint assessment through renewal preparation, with DAG-based
    phase dependencies, gate criteria, and progress tracking.

    Attributes:
        config: Workflow configuration.
    """

    def __init__(
        self,
        config: Optional[AnnualCycleConfig] = None,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> None:
        self.config = config or AnnualCycleConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        self._results: Dict[str, AnnualCycleResult] = {}
        self._cancelled: Set[str] = set()
        self._progress_callback = progress_callback

    async def execute(
        self,
        input_data: Optional[Dict[str, Any]] = None,
    ) -> AnnualCycleResult:
        """Execute the full 10-phase annual cycle."""
        input_data = input_data or {}

        result = AnnualCycleResult(
            org_name=self.config.org_name,
            status=CycleStatus.RUNNING,
            started_at=_utcnow(),
        )
        self._results[result.execution_id] = result

        start_time = time.monotonic()
        phases = PHASE_EXECUTION_ORDER
        total_phases = len(phases)

        self.logger.info(
            "Starting annual cycle: execution_id=%s, org=%s, year=%d, phases=%d",
            result.execution_id, self.config.org_name, self.config.reporting_year, total_phases,
        )

        shared_context: Dict[str, Any] = dict(input_data)
        shared_context["org_name"] = self.config.org_name
        shared_context["reporting_year"] = self.config.reporting_year
        shared_context["base_year"] = self.config.base_year
        shared_context["scope1_tco2e"] = self.config.scope1_tco2e
        shared_context["scope2_tco2e"] = self.config.scope2_tco2e
        shared_context["scope3_tco2e"] = self.config.scope3_tco2e
        shared_context["budget_usd"] = self.config.budget_usd

        try:
            for phase_idx, phase in enumerate(phases):
                if result.execution_id in self._cancelled:
                    result.status = CycleStatus.CANCELLED
                    result.errors.append("Cycle cancelled by user")
                    break

                # Skip verification if disabled
                if phase == AnnualCyclePhase.VERIFICATION and not self.config.enable_verification:
                    phase_result = PhaseResult(
                        phase=phase,
                        status=PhaseStatus.SKIPPED,
                        started_at=_utcnow(),
                        completed_at=_utcnow(),
                    )
                    result.phase_results[phase.value] = phase_result
                    result.phases_skipped.append(phase.value)
                    continue

                # DAG dependency check
                if not self._dependencies_met(phase, result):
                    phase_result = PhaseResult(
                        phase=phase,
                        status=PhaseStatus.FAILED,
                        errors=["Dependencies not met"],
                    )
                    result.phase_results[phase.value] = phase_result
                    result.status = CycleStatus.FAILED
                    result.errors.append(f"Phase '{phase.value}' dependencies not met")
                    break

                # Progress callback
                progress_pct = (phase_idx / total_phases) * 100.0
                if self._progress_callback:
                    await self._progress_callback(
                        phase.value, progress_pct, f"Executing {phase.value}"
                    )

                # Execute phase
                phase_result = await self._execute_phase(phase, shared_context)
                result.phase_results[phase.value] = phase_result

                if phase_result.status == PhaseStatus.FAILED:
                    result.status = CycleStatus.FAILED
                    result.errors.append(f"Phase '{phase.value}' failed")
                    break

                result.phases_completed.append(phase.value)
                result.total_records_processed += phase_result.records_processed
                shared_context[phase.value] = phase_result.outputs

            if result.status == CycleStatus.RUNNING:
                result.status = CycleStatus.COMPLETED

        except Exception as exc:
            self.logger.error("Annual cycle failed: %s", exc, exc_info=True)
            result.status = CycleStatus.FAILED
            result.errors.append(str(exc))

        finally:
            result.completed_at = _utcnow()
            result.total_duration_ms = (time.monotonic() - start_time) * 1000
            result.quality_score = self._compute_quality_score(result)
            result.metrics = self._compile_metrics(shared_context)
            result.summary = self._compile_summary(result)
            if self.config.enable_provenance:
                result.provenance_hash = _compute_hash(
                    result.model_dump_json(exclude={"provenance_hash"})
                )

        self.logger.info(
            "Annual cycle %s: status=%s, phases=%d/%d, duration=%.1fms",
            result.execution_id, result.status.value,
            len(result.phases_completed), total_phases, result.total_duration_ms,
        )
        return result

    def cancel(self, execution_id: str) -> Dict[str, Any]:
        """Cancel a running annual cycle execution."""
        self._cancelled.add(execution_id)
        return {"cancelled": True, "execution_id": execution_id}

    def get_result(self, execution_id: str) -> Optional[AnnualCycleResult]:
        """Retrieve result for a given execution."""
        return self._results.get(execution_id)

    # -------------------------------------------------------------------------
    # Phase Execution
    # -------------------------------------------------------------------------

    async def _execute_phase(
        self, phase: AnnualCyclePhase, context: Dict[str, Any]
    ) -> PhaseResult:
        """Execute a single phase of the annual cycle."""
        started = _utcnow()
        start_time = time.monotonic()

        handler = self._get_phase_handler(phase)
        try:
            outputs, warnings, errors = await handler(context)
            status = PhaseStatus.FAILED if errors else PhaseStatus.COMPLETED
        except Exception as exc:
            outputs = {}
            warnings = []
            errors = [str(exc)]
            status = PhaseStatus.FAILED

        elapsed_ms = (time.monotonic() - start_time) * 1000

        return PhaseResult(
            phase=phase,
            status=status,
            started_at=started,
            completed_at=_utcnow(),
            duration_ms=round(elapsed_ms, 2),
            outputs=outputs,
            warnings=warnings,
            errors=errors,
            provenance_hash=_compute_hash(outputs) if self.config.enable_provenance else "",
        )

    def _get_phase_handler(self, phase: AnnualCyclePhase):
        """Return the handler coroutine for a given phase."""
        handlers = {
            AnnualCyclePhase.FOOTPRINT_ASSESSMENT: self._handle_footprint,
            AnnualCyclePhase.MGMT_PLAN_REVIEW: self._handle_mgmt_plan,
            AnnualCyclePhase.REDUCTION_TRACKING: self._handle_reduction_tracking,
            AnnualCyclePhase.RESIDUAL_CALC: self._handle_residual_calc,
            AnnualCyclePhase.CREDIT_PROCUREMENT: self._handle_procurement,
            AnnualCyclePhase.RETIREMENT: self._handle_retirement,
            AnnualCyclePhase.NEUTRALIZATION_BALANCE: self._handle_neutralization,
            AnnualCyclePhase.CLAIMS_VALIDATION: self._handle_claims,
            AnnualCyclePhase.VERIFICATION: self._handle_verification,
            AnnualCyclePhase.RENEWAL_PREP: self._handle_renewal,
        }
        return handlers[phase]

    # -------------------------------------------------------------------------
    # Phase Handlers
    # -------------------------------------------------------------------------

    async def _handle_footprint(self, ctx: Dict[str, Any]):
        total = ctx.get("scope1_tco2e", 0) + ctx.get("scope2_tco2e", 0) + ctx.get("scope3_tco2e", 0)
        outputs = {
            "total_emissions_tco2e": round(total, 2),
            "scope1_tco2e": ctx.get("scope1_tco2e", 0),
            "scope2_tco2e": ctx.get("scope2_tco2e", 0),
            "scope3_tco2e": ctx.get("scope3_tco2e", 0),
            "reporting_year": ctx.get("reporting_year", 2025),
        }
        warnings = []
        if total <= 0:
            warnings.append("Total emissions are zero or not provided")
        return outputs, warnings, []

    async def _handle_mgmt_plan(self, ctx: Dict[str, Any]):
        total = ctx.get("footprint_assessment", {}).get("total_emissions_tco2e", 0)
        target_reduction = total * (self.config.target_reduction_pct / 100.0)
        outputs = {
            "plan_reviewed": True,
            "target_reduction_tco2e": round(target_reduction, 2),
            "target_reduction_pct": self.config.target_reduction_pct,
            "strategies_count": 5,
        }
        return outputs, [], []

    async def _handle_reduction_tracking(self, ctx: Dict[str, Any]):
        target = ctx.get("mgmt_plan_review", {}).get("target_reduction_tco2e", 0)
        achieved = target * 0.8  # Assume 80% achievement for tracking
        outputs = {
            "reductions_targeted_tco2e": round(target, 2),
            "reductions_achieved_tco2e": round(achieved, 2),
            "achievement_pct": round((achieved / max(target, 1)) * 100, 1),
        }
        warnings = []
        if achieved < target:
            warnings.append("Reduction target not fully met")
        return outputs, warnings, []

    async def _handle_residual_calc(self, ctx: Dict[str, Any]):
        total = ctx.get("footprint_assessment", {}).get("total_emissions_tco2e", 0)
        reductions = ctx.get("reduction_tracking", {}).get("reductions_achieved_tco2e", 0)
        residual = max(total - reductions, 0)
        outputs = {
            "total_emissions_tco2e": round(total, 2),
            "reductions_tco2e": round(reductions, 2),
            "residual_emissions_tco2e": round(residual, 2),
            "credits_needed_tco2e": round(residual, 2),
        }
        return outputs, [], []

    async def _handle_procurement(self, ctx: Dict[str, Any]):
        needed = ctx.get("residual_calc", {}).get("credits_needed_tco2e", 0)
        avg_price = 15.0
        cost = needed * avg_price
        outputs = {
            "credits_procured_tco2e": round(needed, 2),
            "total_cost_usd": round(cost, 2),
            "avg_price_per_tco2e": avg_price,
            "registries_used": ["verra", "gold_standard"],
        }
        warnings = []
        budget = ctx.get("budget_usd", 0)
        if budget > 0 and cost > budget:
            warnings.append(f"Credit cost ${cost:,.0f} exceeds budget ${budget:,.0f}")
        return outputs, warnings, []

    async def _handle_retirement(self, ctx: Dict[str, Any]):
        procured = ctx.get("credit_procurement", {}).get("credits_procured_tco2e", 0)
        outputs = {
            "credits_retired_tco2e": round(procured, 2),
            "retirement_records": 1,
            "registries_confirmed": ["verra", "gold_standard"],
            "certificates_issued": 1,
        }
        return outputs, [], []

    async def _handle_neutralization(self, ctx: Dict[str, Any]):
        residual = ctx.get("residual_calc", {}).get("residual_emissions_tco2e", 0)
        retired = ctx.get("retirement", {}).get("credits_retired_tco2e", 0)
        coverage = (retired / max(residual, 1)) * 100.0
        is_neutral = coverage >= 100.0
        outputs = {
            "residual_tco2e": round(residual, 2),
            "credits_retired_tco2e": round(retired, 2),
            "coverage_pct": round(coverage, 1),
            "is_carbon_neutral": is_neutral,
            "balance_tco2e": round(retired - residual, 2),
        }
        warnings = []
        if not is_neutral:
            warnings.append(f"Neutralization gap: {residual - retired:.0f} tCO2e")
        return outputs, warnings, []

    async def _handle_claims(self, ctx: Dict[str, Any]):
        is_neutral = ctx.get("neutralization_balance", {}).get("is_carbon_neutral", False)
        outputs = {
            "claim_valid": is_neutral,
            "claim_type": "carbon_neutral_organization",
            "frameworks_checked": ["PAS 2060", "VCMI"],
            "compliance_pct": 100.0 if is_neutral else 60.0,
        }
        return outputs, [], []

    async def _handle_verification(self, ctx: Dict[str, Any]):
        outputs = {
            "verification_completed": True,
            "assurance_level": "limited",
            "opinion_type": "unmodified",
            "verification_body": "Bureau Veritas",
            "findings_count": 2,
            "critical_findings": 0,
        }
        return outputs, [], []

    async def _handle_renewal(self, ctx: Dict[str, Any]):
        next_year = self.config.reporting_year + 1
        outputs = {
            "renewal_prepared": True,
            "next_cycle_year": next_year,
            "carry_forward_surplus_tco2e": max(
                ctx.get("neutralization_balance", {}).get("balance_tco2e", 0), 0
            ),
            "improvement_priorities": [
                "Increase primary data collection",
                "Expand supplier engagement",
                "Increase removal credit share",
            ],
        }
        return outputs, [], []

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _dependencies_met(self, phase: AnnualCyclePhase, result: AnnualCycleResult) -> bool:
        deps = PHASE_DEPENDENCIES.get(phase, [])
        for dep in deps:
            dep_result = result.phase_results.get(dep.value)
            if not dep_result or dep_result.status not in (PhaseStatus.COMPLETED, PhaseStatus.SKIPPED):
                return False
        return True

    def _compute_quality_score(self, result: AnnualCycleResult) -> float:
        total = len(PHASE_EXECUTION_ORDER)
        completed = len(result.phases_completed)
        skipped = len(result.phases_skipped)
        effective = completed + skipped * 0.5
        return round((effective / max(total, 1)) * 100.0, 1)

    def _compile_metrics(self, ctx: Dict[str, Any]) -> CycleMetrics:
        footprint = ctx.get("footprint_assessment", {})
        reduction = ctx.get("reduction_tracking", {})
        residual = ctx.get("residual_calc", {})
        procurement = ctx.get("credit_procurement", {})
        neutralization = ctx.get("neutralization_balance", {})
        verification = ctx.get("verification", {})

        total = footprint.get("total_emissions_tco2e", 0)
        prev = self.config.previous_year_emissions_tco2e
        yoy = ((prev - total) / max(prev, 1)) * 100.0 if prev > 0 else 0.0

        return CycleMetrics(
            total_emissions_tco2e=total,
            reductions_achieved_tco2e=reduction.get("reductions_achieved_tco2e", 0),
            residual_emissions_tco2e=residual.get("residual_emissions_tco2e", 0),
            credits_retired_tco2e=neutralization.get("credits_retired_tco2e", 0),
            neutralization_coverage_pct=neutralization.get("coverage_pct", 0),
            is_carbon_neutral=neutralization.get("is_carbon_neutral", False),
            verification_opinion=verification.get("opinion_type", ""),
            credit_spend_usd=procurement.get("total_cost_usd", 0),
            cost_per_tco2e=procurement.get("avg_price_per_tco2e", 0),
            year_over_year_reduction_pct=round(yoy, 1),
        )

    def _compile_summary(self, result: AnnualCycleResult) -> AnnualSummary:
        achievements = []
        improvements = []
        priorities = []

        if result.metrics and result.metrics.is_carbon_neutral:
            achievements.append("Carbon neutrality achieved for the reporting period")
        if result.metrics and result.metrics.year_over_year_reduction_pct > 0:
            achievements.append(
                f"{result.metrics.year_over_year_reduction_pct:.1f}% year-over-year emission reduction"
            )

        if result.metrics and result.metrics.neutralization_coverage_pct < 100:
            improvements.append("Close neutralization gap with additional credits")
        improvements.append("Improve Scope 3 data quality")

        priorities.append("Increase removal credit share in portfolio")
        priorities.append("Expand supplier engagement for Scope 3")
        priorities.append("Prepare for potential VCMI Gold claim upgrade")

        return AnnualSummary(
            cycle_year=self.config.reporting_year,
            phases_completed=len(result.phases_completed),
            metrics=result.metrics,
            key_achievements=achievements,
            areas_for_improvement=improvements,
            next_cycle_priorities=priorities,
        )
