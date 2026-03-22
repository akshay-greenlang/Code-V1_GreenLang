# -*- coding: utf-8 -*-
"""
Rate Optimization Workflow
===================================

4-phase rate optimization workflow within PACK-036 Utility Analysis Pack.
Orchestrates load profile analysis, rate modelling, rate comparison, and
optimization report generation to identify the most cost-effective tariff
structure for each utility account.

Phases:
    1. LoadProfileAnalysis  -- Analyse interval data to build load profile,
                               calculate load factor, peak/off-peak ratios
    2. RateModeling         -- Model costs under each candidate rate schedule
                               using deterministic tariff formulas
    3. RateComparison       -- Compare modelled costs, rank alternatives by
                               annual savings, calculate payback periods
    4. OptimizationReport   -- Generate optimization report with recommended
                               rate schedule and migration plan

The workflow follows GreenLang zero-hallucination principles: all rate
calculations use published tariff formulas, deterministic arithmetic, and
lookup tables. No LLM calls in the numeric computation path.

Schedule: quarterly / on rate change
Estimated duration: 20 minutes

Regulatory References:
    - FERC Form 1 rate schedule filings
    - State PUC tariff books
    - NARUC cost-of-service manual
    - EPRI load research data standards

Author: GreenLang Team
Version: 36.0.0
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import time
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

_MODULE_VERSION = "36.0.0"


def _utcnow() -> datetime:
    """Return current UTC timestamp with zero microseconds."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _new_uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())


def _compute_hash(data: Any) -> str:
    """Compute SHA-256 hash for provenance tracking."""
    if hasattr(data, "model_dump"):
        s = data.model_dump(mode="json")
    elif isinstance(data, dict):
        s = data
    else:
        s = str(data)
    if isinstance(s, dict):
        s = {k: v for k, v in s.items()
             if k not in ("calculated_at", "processing_time_ms", "provenance_hash")}
    return hashlib.sha256(
        json.dumps(s, sort_keys=True, default=str).encode()
    ).hexdigest()


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


class RateStructure(str, Enum):
    """Utility rate structure classification."""
    FLAT = "flat"
    TIERED = "tiered"
    TIME_OF_USE = "time_of_use"
    DEMAND = "demand"
    REAL_TIME_PRICING = "real_time_pricing"
    CRITICAL_PEAK = "critical_peak"
    SEASONAL = "seasonal"
    INTERRUPTIBLE = "interruptible"


class TouPeriod(str, Enum):
    """Time-of-use period classification."""
    ON_PEAK = "on_peak"
    MID_PEAK = "mid_peak"
    OFF_PEAK = "off_peak"
    SUPER_OFF_PEAK = "super_off_peak"
    CRITICAL_PEAK = "critical_peak"


class SeasonType(str, Enum):
    """Season classification for seasonal rates."""
    SUMMER = "summer"
    WINTER = "winter"
    SPRING = "spring"
    FALL = "fall"


# =============================================================================
# REFERENCE DATA (Zero-Hallucination)
# =============================================================================

# Typical TOU period definitions (hours, weekday)
DEFAULT_TOU_PERIODS: Dict[str, Dict[str, Any]] = {
    "on_peak": {"hours_start": 12, "hours_end": 18, "weekday_only": True},
    "mid_peak": {"hours_start": 8, "hours_end": 12, "weekday_only": True},
    "off_peak": {"hours_start": 18, "hours_end": 8, "weekday_only": False},
}

# Rate structure complexity weights (for comparison scoring)
STRUCTURE_COMPLEXITY: Dict[str, float] = {
    "flat": 1.0,
    "tiered": 1.5,
    "time_of_use": 2.0,
    "demand": 2.5,
    "real_time_pricing": 3.5,
    "critical_peak": 3.0,
    "seasonal": 2.0,
    "interruptible": 3.0,
}

# Load factor benchmarks by building type
LOAD_FACTOR_BENCHMARKS: Dict[str, Tuple[float, float]] = {
    "commercial_office": (0.45, 0.65),
    "retail": (0.35, 0.55),
    "industrial_1_shift": (0.40, 0.60),
    "industrial_2_shift": (0.55, 0.75),
    "industrial_3_shift": (0.70, 0.90),
    "hospital": (0.60, 0.80),
    "data_centre": (0.75, 0.95),
    "school": (0.25, 0.45),
    "warehouse": (0.30, 0.50),
    "restaurant": (0.35, 0.55),
    "hotel": (0.50, 0.70),
    "default": (0.40, 0.60),
}


# =============================================================================
# DATA MODELS
# =============================================================================


class PhaseResult(BaseModel):
    """Result from a single workflow phase."""
    phase_name: str = Field(..., description="Phase identifier")
    phase_number: int = Field(default=0, description="Phase sequence number")
    status: PhaseStatus = Field(..., description="Phase completion status")
    duration_seconds: float = Field(default=0.0)
    outputs: Dict[str, Any] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")


class IntervalRecord(BaseModel):
    """Interval metering data record (typically 15-min or hourly).

    Attributes:
        timestamp: ISO 8601 timestamp.
        consumption_kwh: Energy consumption in the interval.
        demand_kw: Average demand during the interval.
        hour_of_day: Hour (0-23).
        day_of_week: Day (0=Mon, 6=Sun).
        month: Month number (1-12).
        is_weekend: Whether interval falls on weekend.
    """
    timestamp: str = Field(default="")
    consumption_kwh: float = Field(default=0.0, ge=0.0)
    demand_kw: float = Field(default=0.0, ge=0.0)
    hour_of_day: int = Field(default=0, ge=0, le=23)
    day_of_week: int = Field(default=0, ge=0, le=6)
    month: int = Field(default=1, ge=1, le=12)
    is_weekend: bool = Field(default=False)


class RateSchedule(BaseModel):
    """A candidate rate schedule for comparison.

    Attributes:
        schedule_id: Unique rate schedule identifier.
        schedule_name: Display name of the rate schedule.
        utility_name: Utility provider name.
        rate_structure: Rate structure classification.
        customer_charge_monthly: Fixed monthly customer charge.
        energy_rates: Energy rates by tier/period ($/kWh).
        demand_rates: Demand charges by tier/period ($/kW).
        tier_thresholds_kwh: Tier boundary thresholds (kWh).
        tou_periods: TOU period definitions.
        seasonal_adjustments: Seasonal rate multipliers.
        minimum_bill: Minimum monthly bill amount.
        power_factor_adjustment: Power factor penalty/credit rate.
        effective_date: Rate schedule effective date.
    """
    schedule_id: str = Field(default_factory=lambda: f"rs-{uuid.uuid4().hex[:8]}")
    schedule_name: str = Field(default="")
    utility_name: str = Field(default="")
    rate_structure: RateStructure = Field(default=RateStructure.FLAT)
    customer_charge_monthly: float = Field(default=0.0, ge=0.0)
    energy_rates: Dict[str, float] = Field(
        default_factory=lambda: {"flat": 0.10}, description="$/kWh by period/tier"
    )
    demand_rates: Dict[str, float] = Field(
        default_factory=dict, description="$/kW by period"
    )
    tier_thresholds_kwh: List[float] = Field(
        default_factory=list, description="Tier breakpoints"
    )
    tou_periods: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    seasonal_adjustments: Dict[str, float] = Field(default_factory=dict)
    minimum_bill: float = Field(default=0.0, ge=0.0)
    power_factor_adjustment: float = Field(default=0.0)
    effective_date: str = Field(default="")


class RateRecommendation(BaseModel):
    """A rate schedule recommendation with savings estimate.

    Attributes:
        rank: Recommendation rank (1 = best).
        schedule_id: Recommended rate schedule identifier.
        schedule_name: Rate schedule display name.
        rate_structure: Rate structure type.
        annual_cost: Modelled annual cost under this rate.
        annual_savings: Savings vs. current rate.
        savings_pct: Savings as percentage of current cost.
        monthly_cost_avg: Average monthly cost.
        migration_effort: Estimated effort to switch (low/medium/high).
        notes: Additional context.
    """
    rank: int = Field(default=0, ge=0)
    schedule_id: str = Field(default="")
    schedule_name: str = Field(default="")
    rate_structure: str = Field(default="")
    annual_cost: float = Field(default=0.0)
    annual_savings: float = Field(default=0.0)
    savings_pct: float = Field(default=0.0)
    monthly_cost_avg: float = Field(default=0.0)
    migration_effort: str = Field(default="low")
    notes: str = Field(default="")


class RateOptimizationInput(BaseModel):
    """Input data model for RateOptimizationWorkflow.

    Attributes:
        account_id: Utility account identifier.
        facility_name: Facility name.
        current_rate_schedule: Currently applied rate schedule.
        candidate_schedules: Alternative rate schedules to evaluate.
        interval_data: Interval metering data (15-min or hourly).
        monthly_consumption_kwh: Monthly consumption totals (if no interval data).
        monthly_demand_kw: Monthly peak demand values.
        annual_consumption_kwh: Total annual consumption.
        building_type: Building type for load factor benchmarking.
        power_factor: Average power factor (0.0-1.0).
        entity_id: Multi-tenant entity identifier.
        tenant_id: Multi-tenant tenant identifier.
    """
    account_id: str = Field(default="")
    facility_name: str = Field(default="")
    current_rate_schedule: RateSchedule = Field(default_factory=RateSchedule)
    candidate_schedules: List[RateSchedule] = Field(default_factory=list)
    interval_data: List[IntervalRecord] = Field(default_factory=list)
    monthly_consumption_kwh: Dict[int, float] = Field(
        default_factory=dict, description="Month -> kWh"
    )
    monthly_demand_kw: Dict[int, float] = Field(
        default_factory=dict, description="Month -> peak kW"
    )
    annual_consumption_kwh: float = Field(default=0.0, ge=0.0)
    building_type: str = Field(default="default")
    power_factor: float = Field(default=0.95, ge=0.0, le=1.0)
    entity_id: str = Field(default="")
    tenant_id: str = Field(default="")


class RateOptimizationResult(BaseModel):
    """Complete result from rate optimization workflow."""
    workflow_id: str = Field(..., description="Unique execution ID")
    workflow_name: str = Field(default="rate_optimization")
    status: WorkflowStatus = Field(..., description="Overall status")
    phases: List[PhaseResult] = Field(default_factory=list)
    account_id: str = Field(default="")
    current_annual_cost: float = Field(default=0.0)
    best_annual_cost: float = Field(default=0.0)
    best_annual_savings: float = Field(default=0.0)
    best_savings_pct: float = Field(default=0.0)
    recommendations: List[RateRecommendation] = Field(default_factory=list)
    load_profile: Dict[str, Any] = Field(default_factory=dict)
    rate_comparison: Dict[str, Any] = Field(default_factory=dict)
    duration_seconds: float = Field(default=0.0)
    provenance_hash: str = Field(default="")


# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================


class RateOptimizationWorkflow:
    """
    4-phase rate optimization workflow.

    Analyses load profiles and evaluates candidate rate schedules to identify
    optimal tariff structure. Each phase produces a PhaseResult with SHA-256
    provenance hash.

    Phases:
        1. LoadProfileAnalysis - Build load profile, calculate load factor
        2. RateModeling        - Model costs under each candidate rate
        3. RateComparison      - Compare and rank alternatives by savings
        4. OptimizationReport  - Generate optimization report

    Zero-hallucination: all rate calculations use deterministic formulas
    (consumption * rate + demand * demand_charge + fixed charges).

    Example:
        >>> wf = RateOptimizationWorkflow()
        >>> inp = RateOptimizationInput(
        ...     account_id="acct-001",
        ...     current_rate_schedule=RateSchedule(schedule_name="SC-1"),
        ...     candidate_schedules=[RateSchedule(schedule_name="TOU-3")],
        ...     annual_consumption_kwh=500000.0,
        ... )
        >>> result = wf.execute(inp)
        >>> assert result.status == WorkflowStatus.COMPLETED
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialise RateOptimizationWorkflow."""
        self.workflow_id: str = _new_uuid()
        self.config: Dict[str, Any] = config or {}
        self._load_profile: Dict[str, Any] = {}
        self._modelled_costs: Dict[str, Dict[str, Any]] = {}
        self._current_cost: float = 0.0
        self._recommendations: List[RateRecommendation] = []
        self._phase_results: List[PhaseResult] = []
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def execute(self, input_data: RateOptimizationInput) -> RateOptimizationResult:
        """Execute the 4-phase rate optimization workflow.

        Args:
            input_data: Validated rate optimization input.

        Returns:
            RateOptimizationResult with recommendations and savings.
        """
        t_start = time.perf_counter()
        self.logger.info(
            "Starting rate optimization workflow %s for account=%s",
            self.workflow_id, input_data.account_id,
        )

        self._phase_results = []
        self._load_profile = {}
        self._modelled_costs = {}
        self._current_cost = 0.0
        self._recommendations = []
        overall_status = WorkflowStatus.RUNNING

        try:
            phase1 = self._phase_1_load_profile_analysis(input_data)
            self._phase_results.append(phase1)
            if phase1.status == PhaseStatus.FAILED:
                raise ValueError(f"Phase 1 failed: {phase1.errors}")

            phase2 = self._phase_2_rate_modeling(input_data)
            self._phase_results.append(phase2)

            phase3 = self._phase_3_rate_comparison(input_data)
            self._phase_results.append(phase3)

            phase4 = self._phase_4_optimization_report(input_data)
            self._phase_results.append(phase4)

            failed_count = sum(
                1 for p in self._phase_results if p.status == PhaseStatus.FAILED
            )
            if failed_count == 0:
                overall_status = WorkflowStatus.COMPLETED
            elif failed_count < len(self._phase_results):
                overall_status = WorkflowStatus.PARTIAL
            else:
                overall_status = WorkflowStatus.FAILED

        except Exception as exc:
            self.logger.error("Rate optimization failed: %s", exc, exc_info=True)
            overall_status = WorkflowStatus.FAILED
            self._phase_results.append(PhaseResult(
                phase_name="error", phase_number=0,
                status=PhaseStatus.FAILED, errors=[str(exc)],
            ))

        elapsed = time.perf_counter() - t_start

        best_savings = self._recommendations[0].annual_savings if self._recommendations else 0.0
        best_cost = self._recommendations[0].annual_cost if self._recommendations else self._current_cost
        best_pct = self._recommendations[0].savings_pct if self._recommendations else 0.0

        result = RateOptimizationResult(
            workflow_id=self.workflow_id,
            status=overall_status,
            phases=self._phase_results,
            account_id=input_data.account_id,
            current_annual_cost=round(self._current_cost, 2),
            best_annual_cost=round(best_cost, 2),
            best_annual_savings=round(best_savings, 2),
            best_savings_pct=round(best_pct, 2),
            recommendations=self._recommendations,
            load_profile=self._load_profile,
            rate_comparison={
                sid: {k: round(v, 2) if isinstance(v, float) else v
                      for k, v in costs.items()}
                for sid, costs in self._modelled_costs.items()
            },
            duration_seconds=round(elapsed, 4),
        )
        result.provenance_hash = _compute_hash(result)

        self.logger.info(
            "Rate optimization %s completed in %.2fs: current=$%.2f best=$%.2f "
            "savings=$%.2f (%.1f%%)",
            self.workflow_id, elapsed, self._current_cost, best_cost,
            best_savings, best_pct,
        )
        return result

    # -------------------------------------------------------------------------
    # Phase 1: Load Profile Analysis
    # -------------------------------------------------------------------------

    def _phase_1_load_profile_analysis(
        self, input_data: RateOptimizationInput
    ) -> PhaseResult:
        """Analyse interval data to build load profile and calculate load factor.

        Args:
            input_data: Rate optimization input.

        Returns:
            PhaseResult with load profile outputs.
        """
        t_start = time.perf_counter()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        annual_kwh = input_data.annual_consumption_kwh
        peak_demand = 0.0

        if input_data.interval_data:
            # Calculate from interval data
            total_kwh = sum(r.consumption_kwh for r in input_data.interval_data)
            if annual_kwh == 0.0:
                annual_kwh = total_kwh

            peak_demand = max(
                (r.demand_kw for r in input_data.interval_data), default=0.0
            )

            # Time-of-use distribution
            on_peak_kwh = sum(
                r.consumption_kwh for r in input_data.interval_data
                if not r.is_weekend and 12 <= r.hour_of_day < 18
            )
            mid_peak_kwh = sum(
                r.consumption_kwh for r in input_data.interval_data
                if not r.is_weekend and 8 <= r.hour_of_day < 12
            )
            off_peak_kwh = total_kwh - on_peak_kwh - mid_peak_kwh

            on_peak_pct = (on_peak_kwh / total_kwh * 100.0) if total_kwh > 0 else 0.0
            mid_peak_pct = (mid_peak_kwh / total_kwh * 100.0) if total_kwh > 0 else 0.0
            off_peak_pct = (off_peak_kwh / total_kwh * 100.0) if total_kwh > 0 else 0.0

            # Monthly distribution
            monthly_kwh: Dict[int, float] = {}
            for r in input_data.interval_data:
                monthly_kwh[r.month] = monthly_kwh.get(r.month, 0.0) + r.consumption_kwh

            outputs["data_source"] = "interval_data"
            outputs["interval_count"] = len(input_data.interval_data)
            outputs["tou_distribution"] = {
                "on_peak_pct": round(on_peak_pct, 1),
                "mid_peak_pct": round(mid_peak_pct, 1),
                "off_peak_pct": round(off_peak_pct, 1),
            }
        elif input_data.monthly_consumption_kwh:
            # Calculate from monthly data
            if annual_kwh == 0.0:
                annual_kwh = sum(input_data.monthly_consumption_kwh.values())
            monthly_kwh = input_data.monthly_consumption_kwh

            if input_data.monthly_demand_kw:
                peak_demand = max(input_data.monthly_demand_kw.values())

            # Estimate TOU distribution (typical commercial profile)
            outputs["data_source"] = "monthly_totals"
            outputs["tou_distribution"] = {
                "on_peak_pct": 35.0,
                "mid_peak_pct": 25.0,
                "off_peak_pct": 40.0,
            }
            warnings.append(
                "No interval data; TOU distribution estimated from typical profile"
            )
        else:
            outputs["data_source"] = "annual_total"
            monthly_kwh = {m: annual_kwh / 12.0 for m in range(1, 13)}
            warnings.append(
                "No interval or monthly data; using flat monthly distribution"
            )
            outputs["tou_distribution"] = {
                "on_peak_pct": 35.0,
                "mid_peak_pct": 25.0,
                "off_peak_pct": 40.0,
            }

        # Calculate load factor
        hours_per_year = 8760.0
        if peak_demand > 0:
            load_factor = annual_kwh / (peak_demand * hours_per_year)
        else:
            # Estimate from building type
            lf_range = LOAD_FACTOR_BENCHMARKS.get(
                input_data.building_type,
                LOAD_FACTOR_BENCHMARKS["default"],
            )
            load_factor = (lf_range[0] + lf_range[1]) / 2.0
            peak_demand = annual_kwh / (load_factor * hours_per_year) if load_factor > 0 else 0.0
            warnings.append(
                f"Peak demand estimated from load factor benchmark ({load_factor:.2f})"
            )

        load_factor = min(1.0, max(0.0, load_factor))

        self._load_profile = {
            "annual_consumption_kwh": round(annual_kwh, 2),
            "peak_demand_kw": round(peak_demand, 2),
            "load_factor": round(load_factor, 4),
            "monthly_kwh": {str(k): round(v, 2) for k, v in monthly_kwh.items()},
            "avg_monthly_kwh": round(annual_kwh / 12.0, 2),
            "building_type": input_data.building_type,
        }
        self._load_profile.update(outputs.get("tou_distribution", {}))

        outputs["annual_consumption_kwh"] = round(annual_kwh, 2)
        outputs["peak_demand_kw"] = round(peak_demand, 2)
        outputs["load_factor"] = round(load_factor, 4)
        outputs["avg_monthly_kwh"] = round(annual_kwh / 12.0, 2)

        elapsed = time.perf_counter() - t_start
        self.logger.info(
            "Phase 1 LoadProfileAnalysis: %.0f kWh/yr, %.0f kW peak, LF=%.2f (%.3fs)",
            annual_kwh, peak_demand, load_factor, elapsed,
        )
        return PhaseResult(
            phase_name="load_profile_analysis", phase_number=1,
            status=PhaseStatus.COMPLETED, duration_seconds=round(elapsed, 4),
            outputs=outputs, warnings=warnings,
            provenance_hash=_compute_hash(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 2: Rate Modeling
    # -------------------------------------------------------------------------

    def _phase_2_rate_modeling(
        self, input_data: RateOptimizationInput
    ) -> PhaseResult:
        """Model annual costs under each candidate rate schedule.

        Args:
            input_data: Rate optimization input.

        Returns:
            PhaseResult with modelled costs.
        """
        t_start = time.perf_counter()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        annual_kwh = self._load_profile.get("annual_consumption_kwh", 0.0)
        peak_kw = self._load_profile.get("peak_demand_kw", 0.0)
        monthly_kwh_data = self._load_profile.get("monthly_kwh", {})

        # Model current rate
        current = input_data.current_rate_schedule
        current_cost = self._calculate_annual_cost(
            current, annual_kwh, peak_kw, monthly_kwh_data, input_data.power_factor
        )
        self._current_cost = current_cost
        self._modelled_costs[current.schedule_id] = {
            "schedule_name": current.schedule_name,
            "rate_structure": current.rate_structure.value,
            "annual_cost": current_cost,
            "is_current": True,
        }

        # Model each candidate
        for schedule in input_data.candidate_schedules:
            cost = self._calculate_annual_cost(
                schedule, annual_kwh, peak_kw, monthly_kwh_data, input_data.power_factor
            )
            self._modelled_costs[schedule.schedule_id] = {
                "schedule_name": schedule.schedule_name,
                "rate_structure": schedule.rate_structure.value,
                "annual_cost": cost,
                "is_current": False,
            }

        outputs["schedules_modelled"] = len(self._modelled_costs)
        outputs["current_schedule"] = current.schedule_name
        outputs["current_annual_cost"] = round(current_cost, 2)
        outputs["modelled_costs"] = {
            sid: round(data["annual_cost"], 2)
            for sid, data in self._modelled_costs.items()
        }

        elapsed = time.perf_counter() - t_start
        self.logger.info(
            "Phase 2 RateModeling: %d schedules, current=$%.2f (%.3fs)",
            len(self._modelled_costs), current_cost, elapsed,
        )
        return PhaseResult(
            phase_name="rate_modeling", phase_number=2,
            status=PhaseStatus.COMPLETED, duration_seconds=round(elapsed, 4),
            outputs=outputs, warnings=warnings,
            provenance_hash=_compute_hash(outputs),
        )

    def _calculate_annual_cost(
        self,
        schedule: RateSchedule,
        annual_kwh: float,
        peak_kw: float,
        monthly_kwh: Dict[str, float],
        power_factor: float,
    ) -> float:
        """Calculate annual cost under a given rate schedule (deterministic).

        Args:
            schedule: Rate schedule to evaluate.
            annual_kwh: Annual energy consumption (kWh).
            peak_kw: Peak demand (kW).
            monthly_kwh: Monthly consumption breakdown.
            power_factor: Average power factor.

        Returns:
            Estimated annual cost in dollars.
        """
        # Fixed charges (12 months)
        fixed_cost = schedule.customer_charge_monthly * 12.0

        # Energy charges
        energy_cost = 0.0
        if schedule.rate_structure == RateStructure.FLAT:
            rate = schedule.energy_rates.get("flat", 0.10)
            energy_cost = annual_kwh * rate
        elif schedule.rate_structure == RateStructure.TIERED:
            energy_cost = self._calculate_tiered_cost(
                annual_kwh, schedule.energy_rates, schedule.tier_thresholds_kwh
            )
        elif schedule.rate_structure == RateStructure.TIME_OF_USE:
            tou_dist = self._load_profile.get("tou_distribution", {})
            on_peak_pct = tou_dist.get("on_peak_pct", 35.0) / 100.0
            mid_peak_pct = tou_dist.get("mid_peak_pct", 25.0) / 100.0
            off_peak_pct = tou_dist.get("off_peak_pct", 40.0) / 100.0

            on_rate = schedule.energy_rates.get("on_peak", 0.15)
            mid_rate = schedule.energy_rates.get("mid_peak", 0.10)
            off_rate = schedule.energy_rates.get("off_peak", 0.06)

            energy_cost = (
                annual_kwh * on_peak_pct * on_rate
                + annual_kwh * mid_peak_pct * mid_rate
                + annual_kwh * off_peak_pct * off_rate
            )
        elif schedule.rate_structure == RateStructure.SEASONAL:
            summer_rate = schedule.energy_rates.get("summer", 0.12)
            winter_rate = schedule.energy_rates.get("winter", 0.08)
            summer_months = {6, 7, 8, 9}
            for m_str, m_kwh in monthly_kwh.items():
                m_int = int(m_str)
                rate = summer_rate if m_int in summer_months else winter_rate
                energy_cost += m_kwh * rate
            if not monthly_kwh:
                energy_cost = annual_kwh * (summer_rate + winter_rate) / 2.0
        else:
            rate = schedule.energy_rates.get("flat", 0.10)
            energy_cost = annual_kwh * rate

        # Demand charges
        demand_cost = 0.0
        if schedule.demand_rates:
            demand_rate = schedule.demand_rates.get("flat", 0.0)
            on_peak_demand_rate = schedule.demand_rates.get("on_peak", demand_rate)
            demand_cost = peak_kw * on_peak_demand_rate * 12.0

        # Power factor adjustment
        pf_adjustment = 0.0
        if schedule.power_factor_adjustment != 0.0 and power_factor < 0.90:
            pf_penalty_pct = (0.90 - power_factor) / 0.90
            pf_adjustment = (energy_cost + demand_cost) * pf_penalty_pct * schedule.power_factor_adjustment

        # Minimum bill
        total = fixed_cost + energy_cost + demand_cost + pf_adjustment
        if schedule.minimum_bill > 0:
            monthly_avg = total / 12.0
            if monthly_avg < schedule.minimum_bill:
                total = schedule.minimum_bill * 12.0

        return max(0.0, total)

    def _calculate_tiered_cost(
        self,
        annual_kwh: float,
        rates: Dict[str, float],
        thresholds: List[float],
    ) -> float:
        """Calculate tiered energy cost (deterministic).

        Args:
            annual_kwh: Annual consumption.
            rates: Tier rates (tier_1, tier_2, etc.).
            thresholds: Tier boundaries in kWh.

        Returns:
            Annual energy cost under tiered structure.
        """
        # Apply tiers on monthly basis
        monthly_kwh = annual_kwh / 12.0
        monthly_cost = 0.0
        remaining = monthly_kwh

        sorted_tiers = sorted(
            [(k, v) for k, v in rates.items() if k.startswith("tier_")],
            key=lambda x: x[0],
        )

        for i, (tier_name, rate) in enumerate(sorted_tiers):
            if i < len(thresholds):
                tier_kwh = min(remaining, thresholds[i])
            else:
                tier_kwh = remaining

            monthly_cost += tier_kwh * rate
            remaining -= tier_kwh
            if remaining <= 0:
                break

        if remaining > 0 and sorted_tiers:
            monthly_cost += remaining * sorted_tiers[-1][1]

        return monthly_cost * 12.0

    # -------------------------------------------------------------------------
    # Phase 3: Rate Comparison
    # -------------------------------------------------------------------------

    def _phase_3_rate_comparison(
        self, input_data: RateOptimizationInput
    ) -> PhaseResult:
        """Compare modelled costs and rank alternatives by savings.

        Args:
            input_data: Rate optimization input.

        Returns:
            PhaseResult with comparison outputs.
        """
        t_start = time.perf_counter()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        current_cost = self._current_cost
        alternatives: List[Dict[str, Any]] = []

        for sid, data in self._modelled_costs.items():
            if data.get("is_current"):
                continue
            annual_cost = data["annual_cost"]
            savings = current_cost - annual_cost
            savings_pct = (savings / current_cost * 100.0) if current_cost > 0 else 0.0

            complexity = STRUCTURE_COMPLEXITY.get(data["rate_structure"], 2.0)
            if complexity > 2.5:
                effort = "high"
            elif complexity > 1.5:
                effort = "medium"
            else:
                effort = "low"

            alternatives.append({
                "schedule_id": sid,
                "schedule_name": data["schedule_name"],
                "rate_structure": data["rate_structure"],
                "annual_cost": annual_cost,
                "savings": savings,
                "savings_pct": savings_pct,
                "effort": effort,
            })

        # Sort by savings descending
        alternatives.sort(key=lambda a: a["savings"], reverse=True)

        # Build recommendations
        recommendations: List[RateRecommendation] = []
        for rank, alt in enumerate(alternatives, start=1):
            recommendations.append(RateRecommendation(
                rank=rank,
                schedule_id=alt["schedule_id"],
                schedule_name=alt["schedule_name"],
                rate_structure=alt["rate_structure"],
                annual_cost=round(alt["annual_cost"], 2),
                annual_savings=round(alt["savings"], 2),
                savings_pct=round(alt["savings_pct"], 2),
                monthly_cost_avg=round(alt["annual_cost"] / 12.0, 2),
                migration_effort=alt["effort"],
                notes=(
                    f"Switch to {alt['rate_structure']} rate for "
                    f"${alt['savings']:.0f}/yr savings"
                    if alt["savings"] > 0
                    else "No savings vs current rate"
                ),
            ))

        self._recommendations = recommendations

        outputs["alternatives_evaluated"] = len(alternatives)
        outputs["recommendations_with_savings"] = sum(
            1 for r in recommendations if r.annual_savings > 0
        )
        if recommendations:
            outputs["best_schedule"] = recommendations[0].schedule_name
            outputs["best_savings"] = round(recommendations[0].annual_savings, 2)
            outputs["best_savings_pct"] = round(recommendations[0].savings_pct, 2)

        elapsed = time.perf_counter() - t_start
        self.logger.info(
            "Phase 3 RateComparison: %d alternatives, best savings=$%.2f (%.3fs)",
            len(alternatives),
            recommendations[0].annual_savings if recommendations else 0.0,
            elapsed,
        )
        return PhaseResult(
            phase_name="rate_comparison", phase_number=3,
            status=PhaseStatus.COMPLETED, duration_seconds=round(elapsed, 4),
            outputs=outputs, warnings=warnings,
            provenance_hash=_compute_hash(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 4: Optimization Report
    # -------------------------------------------------------------------------

    def _phase_4_optimization_report(
        self, input_data: RateOptimizationInput
    ) -> PhaseResult:
        """Generate optimization report with migration plan.

        Args:
            input_data: Rate optimization input.

        Returns:
            PhaseResult with report outputs.
        """
        t_start = time.perf_counter()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        report_id = f"rpt-{uuid.uuid4().hex[:8]}"

        # Migration steps for top recommendation
        migration_steps: List[Dict[str, str]] = []
        if self._recommendations and self._recommendations[0].annual_savings > 0:
            top = self._recommendations[0]
            migration_steps = [
                {"step": "1", "action": "Contact utility to confirm rate eligibility",
                 "timeline": "1-2 weeks"},
                {"step": "2", "action": f"Submit rate change request to {top.schedule_name}",
                 "timeline": "1 billing cycle"},
                {"step": "3", "action": "Verify first bill under new rate schedule",
                 "timeline": "1 billing cycle"},
                {"step": "4", "action": "Monitor monthly costs for 3 billing cycles",
                 "timeline": "3 months"},
            ]

        outputs["report_id"] = report_id
        outputs["generated_at"] = _utcnow().isoformat()
        outputs["account_id"] = input_data.account_id
        outputs["facility_name"] = input_data.facility_name
        outputs["current_rate"] = input_data.current_rate_schedule.schedule_name
        outputs["current_annual_cost"] = round(self._current_cost, 2)
        outputs["recommendation_count"] = len(self._recommendations)
        outputs["migration_steps"] = migration_steps
        outputs["methodology"] = [
            "Load profile analysis from interval/monthly data",
            "Deterministic rate schedule cost modeling",
            "Annual cost comparison with savings ranking",
            "Power factor adjustment per utility tariff",
        ]

        elapsed = time.perf_counter() - t_start
        self.logger.info(
            "Phase 4 OptimizationReport: report=%s, %d steps (%.3fs)",
            report_id, len(migration_steps), elapsed,
        )
        return PhaseResult(
            phase_name="optimization_report", phase_number=4,
            status=PhaseStatus.COMPLETED, duration_seconds=round(elapsed, 4),
            outputs=outputs, warnings=warnings,
            provenance_hash=_compute_hash(outputs),
        )
