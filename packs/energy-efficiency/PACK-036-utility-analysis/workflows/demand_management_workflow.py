# -*- coding: utf-8 -*-
"""
Demand Management Workflow
===================================

4-phase demand management workflow within PACK-036 Utility Analysis Pack.
Orchestrates demand profiling, peak identification, strategy development,
and demand reduction report generation to minimise peak demand charges
and improve load factor.

Phases:
    1. DemandProfiling       -- Analyse interval data to build demand profile,
                                calculate coincident/non-coincident peaks, load
                                duration curves, and demand variability metrics
    2. PeakIdentification    -- Identify peak demand events, classify by cause
                                (HVAC, process, lighting), determine ratchet
                                exposure and demand response eligibility
    3. StrategyDevelopment   -- Develop demand reduction strategies: load
                                shifting, peak shaving, demand response,
                                battery storage, and operational scheduling
    4. DemandReport          -- Generate demand management report with strategy
                                rankings, cost-benefit analysis, and
                                implementation roadmap

The workflow follows GreenLang zero-hallucination principles: all demand
calculations use deterministic arithmetic (kW measurements, load factor
formulas, cost-benefit ratios). No LLM calls in the numeric path.

Schedule: monthly / on-demand
Estimated duration: 25 minutes

Regulatory References:
    - FERC demand response technical standards
    - NAESB demand response communications protocol
    - ASHRAE Guideline 14-2014 for M&V of demand savings
    - IEEE 1547-2018 for DER interconnection

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


class PeakCause(str, Enum):
    """Root cause classification for peak demand events."""
    HVAC_STARTUP = "hvac_startup"
    HVAC_COOLING = "hvac_cooling"
    HVAC_HEATING = "hvac_heating"
    PROCESS_LOAD = "process_load"
    LIGHTING = "lighting"
    EQUIPMENT_STARTUP = "equipment_startup"
    COINCIDENT_LOADS = "coincident_loads"
    WEATHER_DRIVEN = "weather_driven"
    OCCUPANCY_DRIVEN = "occupancy_driven"
    UNKNOWN = "unknown"


class StrategyType(str, Enum):
    """Demand reduction strategy classification."""
    LOAD_SHIFTING = "load_shifting"
    PEAK_SHAVING = "peak_shaving"
    DEMAND_RESPONSE = "demand_response"
    BATTERY_STORAGE = "battery_storage"
    OPERATIONAL_SCHEDULING = "operational_scheduling"
    HVAC_OPTIMIZATION = "hvac_optimization"
    MOTOR_SOFT_START = "motor_soft_start"
    LOAD_SEQUENCING = "load_sequencing"
    THERMAL_STORAGE = "thermal_storage"
    ON_SITE_GENERATION = "on_site_generation"


class StrategyPriority(str, Enum):
    """Strategy implementation priority."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


# =============================================================================
# REFERENCE DATA (Zero-Hallucination)
# =============================================================================

# Typical demand reduction potential by strategy type (percentage of peak)
STRATEGY_REDUCTION_POTENTIAL: Dict[str, Tuple[float, float]] = {
    "load_shifting": (0.05, 0.15),
    "peak_shaving": (0.10, 0.25),
    "demand_response": (0.10, 0.30),
    "battery_storage": (0.15, 0.40),
    "operational_scheduling": (0.05, 0.15),
    "hvac_optimization": (0.08, 0.20),
    "motor_soft_start": (0.03, 0.10),
    "load_sequencing": (0.05, 0.12),
    "thermal_storage": (0.10, 0.25),
    "on_site_generation": (0.20, 0.50),
}

# Typical implementation costs per kW of demand reduction
STRATEGY_COST_PER_KW: Dict[str, Tuple[float, float]] = {
    "load_shifting": (5.0, 25.0),
    "peak_shaving": (50.0, 200.0),
    "demand_response": (10.0, 50.0),
    "battery_storage": (400.0, 800.0),
    "operational_scheduling": (2.0, 15.0),
    "hvac_optimization": (20.0, 80.0),
    "motor_soft_start": (30.0, 100.0),
    "load_sequencing": (10.0, 40.0),
    "thermal_storage": (200.0, 500.0),
    "on_site_generation": (500.0, 1500.0),
}

# Demand charge rate benchmarks by utility size class ($/kW/month)
DEMAND_CHARGE_BENCHMARKS: Dict[str, Tuple[float, float]] = {
    "small_commercial": (5.0, 15.0),
    "medium_commercial": (8.0, 20.0),
    "large_commercial": (10.0, 25.0),
    "industrial": (12.0, 30.0),
    "default": (8.0, 20.0),
}

# Load factor improvement targets by current load factor
LOAD_FACTOR_TARGETS: Dict[str, float] = {
    "below_30": 0.45,
    "30_to_50": 0.60,
    "50_to_70": 0.75,
    "above_70": 0.85,
}


# =============================================================================
# DATA MODELS
# =============================================================================


class PhaseResult(BaseModel):
    """Result from a single workflow phase."""
    phase_name: str = Field(..., description="Phase identifier")
    phase_number: int = Field(default=0)
    status: PhaseStatus = Field(...)
    duration_seconds: float = Field(default=0.0)
    outputs: Dict[str, Any] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    provenance_hash: str = Field(default="")


class DemandInterval(BaseModel):
    """Demand interval measurement record.

    Attributes:
        timestamp: ISO 8601 timestamp.
        demand_kw: Measured demand in kW.
        consumption_kwh: Energy in interval.
        hour_of_day: Hour (0-23).
        day_of_week: Day (0=Mon, 6=Sun).
        month: Month (1-12).
        is_weekend: Weekend flag.
        temperature_c: Outdoor temperature (optional).
    """
    timestamp: str = Field(default="")
    demand_kw: float = Field(default=0.0, ge=0.0)
    consumption_kwh: float = Field(default=0.0, ge=0.0)
    hour_of_day: int = Field(default=0, ge=0, le=23)
    day_of_week: int = Field(default=0, ge=0, le=6)
    month: int = Field(default=1, ge=1, le=12)
    is_weekend: bool = Field(default=False)
    temperature_c: Optional[float] = Field(default=None)


class PeakEvent(BaseModel):
    """A detected peak demand event.

    Attributes:
        event_id: Unique event identifier.
        timestamp: When the peak occurred.
        demand_kw: Peak demand value.
        duration_minutes: Duration of the peak event.
        cause: Classified root cause.
        month: Month of occurrence.
        hour_of_day: Hour of occurrence.
        is_billing_peak: Whether this set the billing peak.
        estimated_cost_impact: Estimated demand charge impact.
    """
    event_id: str = Field(default_factory=lambda: f"pk-{uuid.uuid4().hex[:8]}")
    timestamp: str = Field(default="")
    demand_kw: float = Field(default=0.0, ge=0.0)
    duration_minutes: int = Field(default=15, ge=1)
    cause: PeakCause = Field(default=PeakCause.UNKNOWN)
    month: int = Field(default=1, ge=1, le=12)
    hour_of_day: int = Field(default=12, ge=0, le=23)
    is_billing_peak: bool = Field(default=False)
    estimated_cost_impact: float = Field(default=0.0)


class DemandReductionStrategy(BaseModel):
    """A demand reduction strategy recommendation.

    Attributes:
        strategy_id: Unique strategy identifier.
        strategy_type: Strategy classification.
        priority: Implementation priority.
        description: Strategy description.
        target_reduction_kw: Expected peak demand reduction.
        target_reduction_pct: Reduction as percentage of current peak.
        annual_savings: Estimated annual cost savings.
        implementation_cost: Estimated implementation cost.
        simple_payback_years: Simple payback period.
        load_factor_improvement: Expected load factor increase.
        implementation_timeline: Estimated timeline.
        prerequisites: Required preconditions.
    """
    strategy_id: str = Field(default_factory=lambda: f"strat-{uuid.uuid4().hex[:8]}")
    strategy_type: StrategyType = Field(default=StrategyType.LOAD_SHIFTING)
    priority: StrategyPriority = Field(default=StrategyPriority.MEDIUM)
    description: str = Field(default="")
    target_reduction_kw: float = Field(default=0.0, ge=0.0)
    target_reduction_pct: float = Field(default=0.0, ge=0.0)
    annual_savings: float = Field(default=0.0)
    implementation_cost: float = Field(default=0.0, ge=0.0)
    simple_payback_years: float = Field(default=0.0, ge=0.0)
    load_factor_improvement: float = Field(default=0.0)
    implementation_timeline: str = Field(default="")
    prerequisites: List[str] = Field(default_factory=list)


class DemandManagementInput(BaseModel):
    """Input data model for DemandManagementWorkflow.

    Attributes:
        facility_id: Facility identifier.
        facility_name: Facility name.
        interval_data: Demand interval measurements.
        monthly_peak_demand_kw: Monthly billing peak demands.
        annual_consumption_kwh: Annual energy consumption.
        demand_charge_rate: Current demand charge rate ($/kW/month).
        ratchet_percentage: Demand ratchet percentage (0-100).
        ratchet_months: Number of months for ratchet lookback.
        building_type: Building type for benchmarking.
        utility_class: Utility customer class.
        has_bms: Whether building has a BMS/EMS.
        has_interval_meters: Whether interval metering is available.
        entity_id: Multi-tenant entity identifier.
        tenant_id: Multi-tenant tenant identifier.
    """
    facility_id: str = Field(default="")
    facility_name: str = Field(default="")
    interval_data: List[DemandInterval] = Field(default_factory=list)
    monthly_peak_demand_kw: Dict[int, float] = Field(
        default_factory=dict, description="Month -> peak kW"
    )
    annual_consumption_kwh: float = Field(default=0.0, ge=0.0)
    demand_charge_rate: float = Field(default=12.0, ge=0.0, description="$/kW/month")
    ratchet_percentage: float = Field(default=80.0, ge=0.0, le=100.0)
    ratchet_months: int = Field(default=11, ge=0, le=24)
    building_type: str = Field(default="default")
    utility_class: str = Field(default="medium_commercial")
    has_bms: bool = Field(default=False)
    has_interval_meters: bool = Field(default=True)
    entity_id: str = Field(default="")
    tenant_id: str = Field(default="")


class DemandManagementResult(BaseModel):
    """Complete result from demand management workflow."""
    workflow_id: str = Field(..., description="Unique execution ID")
    workflow_name: str = Field(default="demand_management")
    status: WorkflowStatus = Field(...)
    phases: List[PhaseResult] = Field(default_factory=list)
    facility_id: str = Field(default="")
    current_peak_kw: float = Field(default=0.0)
    current_load_factor: float = Field(default=0.0)
    target_peak_kw: float = Field(default=0.0)
    target_load_factor: float = Field(default=0.0)
    annual_demand_charges: float = Field(default=0.0)
    peak_events: List[PeakEvent] = Field(default_factory=list)
    strategies: List[DemandReductionStrategy] = Field(default_factory=list)
    total_savings_potential: float = Field(default=0.0)
    total_implementation_cost: float = Field(default=0.0)
    demand_profile: Dict[str, Any] = Field(default_factory=dict)
    duration_seconds: float = Field(default=0.0)
    provenance_hash: str = Field(default="")


# =============================================================================
# WORKFLOW IMPLEMENTATION
# =============================================================================


class DemandManagementWorkflow:
    """
    4-phase demand management workflow.

    Analyses demand profiles, identifies peak events, develops reduction
    strategies, and generates implementation reports. Each phase produces
    a PhaseResult with SHA-256 provenance hash.

    Phases:
        1. DemandProfiling      - Build demand profile, load duration curve
        2. PeakIdentification   - Detect and classify peak events
        3. StrategyDevelopment  - Develop demand reduction strategies
        4. DemandReport         - Generate implementation report

    Zero-hallucination: all demand calculations use deterministic arithmetic.
    Reduction potentials from published ASHRAE/EPRI benchmarks.

    Example:
        >>> wf = DemandManagementWorkflow()
        >>> inp = DemandManagementInput(
        ...     facility_id="fac-001",
        ...     monthly_peak_demand_kw={1: 500, 2: 480, 7: 750},
        ...     annual_consumption_kwh=2000000,
        ... )
        >>> result = wf.execute(inp)
        >>> assert result.status == WorkflowStatus.COMPLETED
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialise DemandManagementWorkflow."""
        self.workflow_id: str = _new_uuid()
        self.config: Dict[str, Any] = config or {}
        self._demand_profile: Dict[str, Any] = {}
        self._peak_events: List[PeakEvent] = []
        self._strategies: List[DemandReductionStrategy] = []
        self._phase_results: List[PhaseResult] = []
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def execute(self, input_data: DemandManagementInput) -> DemandManagementResult:
        """Execute the 4-phase demand management workflow.

        Args:
            input_data: Validated demand management input.

        Returns:
            DemandManagementResult with strategies and savings.
        """
        t_start = time.perf_counter()
        self.logger.info(
            "Starting demand management workflow %s for facility=%s",
            self.workflow_id, input_data.facility_id,
        )

        self._phase_results = []
        self._demand_profile = {}
        self._peak_events = []
        self._strategies = []
        overall_status = WorkflowStatus.RUNNING

        try:
            phase1 = self._phase_1_demand_profiling(input_data)
            self._phase_results.append(phase1)
            if phase1.status == PhaseStatus.FAILED:
                raise ValueError(f"Phase 1 failed: {phase1.errors}")

            phase2 = self._phase_2_peak_identification(input_data)
            self._phase_results.append(phase2)

            phase3 = self._phase_3_strategy_development(input_data)
            self._phase_results.append(phase3)

            phase4 = self._phase_4_demand_report(input_data)
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
            self.logger.error("Demand management failed: %s", exc, exc_info=True)
            overall_status = WorkflowStatus.FAILED
            self._phase_results.append(PhaseResult(
                phase_name="error", phase_number=0,
                status=PhaseStatus.FAILED, errors=[str(exc)],
            ))

        elapsed = time.perf_counter() - t_start

        current_peak = self._demand_profile.get("overall_peak_kw", 0.0)
        load_factor = self._demand_profile.get("load_factor", 0.0)
        annual_demand = current_peak * input_data.demand_charge_rate * 12.0
        total_savings = sum(s.annual_savings for s in self._strategies)
        total_impl_cost = sum(s.implementation_cost for s in self._strategies)

        # Target load factor
        if load_factor < 0.30:
            target_lf = LOAD_FACTOR_TARGETS["below_30"]
        elif load_factor < 0.50:
            target_lf = LOAD_FACTOR_TARGETS["30_to_50"]
        elif load_factor < 0.70:
            target_lf = LOAD_FACTOR_TARGETS["50_to_70"]
        else:
            target_lf = LOAD_FACTOR_TARGETS["above_70"]

        target_peak = (
            input_data.annual_consumption_kwh / (target_lf * 8760.0)
            if target_lf > 0 and input_data.annual_consumption_kwh > 0
            else current_peak
        )

        result = DemandManagementResult(
            workflow_id=self.workflow_id,
            status=overall_status,
            phases=self._phase_results,
            facility_id=input_data.facility_id,
            current_peak_kw=round(current_peak, 2),
            current_load_factor=round(load_factor, 4),
            target_peak_kw=round(target_peak, 2),
            target_load_factor=round(target_lf, 4),
            annual_demand_charges=round(annual_demand, 2),
            peak_events=self._peak_events,
            strategies=self._strategies,
            total_savings_potential=round(total_savings, 2),
            total_implementation_cost=round(total_impl_cost, 2),
            demand_profile=self._demand_profile,
            duration_seconds=round(elapsed, 4),
        )
        result.provenance_hash = _compute_hash(result)

        self.logger.info(
            "Demand management %s completed in %.2fs: peak=%.0f kW, LF=%.2f, "
            "savings=$%.2f",
            self.workflow_id, elapsed, current_peak, load_factor, total_savings,
        )
        return result

    # -------------------------------------------------------------------------
    # Phase 1: Demand Profiling
    # -------------------------------------------------------------------------

    def _phase_1_demand_profiling(
        self, input_data: DemandManagementInput
    ) -> PhaseResult:
        """Build demand profile with load factor and variability metrics.

        Args:
            input_data: Demand management input.

        Returns:
            PhaseResult with demand profile outputs.
        """
        t_start = time.perf_counter()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        if input_data.interval_data:
            demands = [r.demand_kw for r in input_data.interval_data if r.demand_kw > 0]
            if not demands:
                return PhaseResult(
                    phase_name="demand_profiling", phase_number=1,
                    status=PhaseStatus.FAILED,
                    errors=["No valid demand readings in interval data"],
                    duration_seconds=round(time.perf_counter() - t_start, 4),
                )

            overall_peak = max(demands)
            avg_demand = sum(demands) / len(demands)
            min_demand = min(demands)
            std_dev = math.sqrt(
                sum((d - avg_demand) ** 2 for d in demands) / len(demands)
            )

            # Hourly demand profile
            hourly_avg: Dict[int, List[float]] = {}
            for r in input_data.interval_data:
                if r.hour_of_day not in hourly_avg:
                    hourly_avg[r.hour_of_day] = []
                hourly_avg[r.hour_of_day].append(r.demand_kw)

            hourly_profile = {
                str(h): round(sum(vals) / len(vals), 2)
                for h, vals in sorted(hourly_avg.items())
            }

            # Monthly peaks from interval data
            monthly_peaks: Dict[int, float] = {}
            for r in input_data.interval_data:
                if r.month not in monthly_peaks or r.demand_kw > monthly_peaks[r.month]:
                    monthly_peaks[r.month] = r.demand_kw

            outputs["data_source"] = "interval_data"
            outputs["interval_count"] = len(input_data.interval_data)

        elif input_data.monthly_peak_demand_kw:
            peaks = list(input_data.monthly_peak_demand_kw.values())
            overall_peak = max(peaks)
            avg_demand = sum(peaks) / len(peaks)
            min_demand = min(peaks)
            std_dev = math.sqrt(
                sum((p - avg_demand) ** 2 for p in peaks) / len(peaks)
            ) if len(peaks) > 1 else 0.0
            monthly_peaks = dict(input_data.monthly_peak_demand_kw)
            hourly_profile = {}
            outputs["data_source"] = "monthly_peaks"
            warnings.append("No interval data; limited demand profile resolution")
        else:
            return PhaseResult(
                phase_name="demand_profiling", phase_number=1,
                status=PhaseStatus.FAILED,
                errors=["No demand data provided (interval or monthly)"],
                duration_seconds=round(time.perf_counter() - t_start, 4),
            )

        # Load factor calculation
        hours_per_year = 8760.0
        annual_kwh = input_data.annual_consumption_kwh
        if annual_kwh > 0 and overall_peak > 0:
            load_factor = annual_kwh / (overall_peak * hours_per_year)
        else:
            load_factor = avg_demand / overall_peak if overall_peak > 0 else 0.0

        load_factor = min(1.0, max(0.0, load_factor))

        # Demand variability coefficient
        cv = std_dev / avg_demand if avg_demand > 0 else 0.0

        # Ratchet analysis
        ratchet_demand = overall_peak * (input_data.ratchet_percentage / 100.0)
        ratchet_exposure = sum(
            1 for pk in monthly_peaks.values() if pk < ratchet_demand
        )

        self._demand_profile = {
            "overall_peak_kw": round(overall_peak, 2),
            "average_demand_kw": round(avg_demand, 2),
            "minimum_demand_kw": round(min_demand, 2),
            "std_dev_kw": round(std_dev, 2),
            "coefficient_of_variation": round(cv, 4),
            "load_factor": round(load_factor, 4),
            "monthly_peaks": {str(k): round(v, 2) for k, v in monthly_peaks.items()},
            "hourly_profile": hourly_profile,
            "peak_to_average_ratio": round(
                overall_peak / avg_demand, 2
            ) if avg_demand > 0 else 0.0,
            "ratchet_demand_kw": round(ratchet_demand, 2),
            "ratchet_exposure_months": ratchet_exposure,
        }

        outputs.update({
            "overall_peak_kw": round(overall_peak, 2),
            "average_demand_kw": round(avg_demand, 2),
            "load_factor": round(load_factor, 4),
            "coefficient_of_variation": round(cv, 4),
            "ratchet_exposure_months": ratchet_exposure,
        })

        elapsed = time.perf_counter() - t_start
        self.logger.info(
            "Phase 1 DemandProfiling: peak=%.0f kW, avg=%.0f kW, LF=%.2f (%.3fs)",
            overall_peak, avg_demand, load_factor, elapsed,
        )
        return PhaseResult(
            phase_name="demand_profiling", phase_number=1,
            status=PhaseStatus.COMPLETED, duration_seconds=round(elapsed, 4),
            outputs=outputs, warnings=warnings,
            provenance_hash=_compute_hash(outputs),
        )

    # -------------------------------------------------------------------------
    # Phase 2: Peak Identification
    # -------------------------------------------------------------------------

    def _phase_2_peak_identification(
        self, input_data: DemandManagementInput
    ) -> PhaseResult:
        """Identify and classify peak demand events.

        Args:
            input_data: Demand management input.

        Returns:
            PhaseResult with peak event outputs.
        """
        t_start = time.perf_counter()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}
        peak_events: List[PeakEvent] = []

        overall_peak = self._demand_profile.get("overall_peak_kw", 0.0)
        avg_demand = self._demand_profile.get("average_demand_kw", 0.0)
        monthly_peaks = self._demand_profile.get("monthly_peaks", {})

        # Identify peak events from monthly peaks
        peak_threshold = avg_demand * 1.3 if avg_demand > 0 else overall_peak * 0.8

        for month_str, peak_kw in monthly_peaks.items():
            month = int(month_str)
            if peak_kw >= peak_threshold:
                # Classify cause based on month and time patterns
                cause = self._classify_peak_cause(month, peak_kw, overall_peak)
                is_billing = abs(peak_kw - overall_peak) < 1.0
                cost_impact = peak_kw * input_data.demand_charge_rate

                peak_events.append(PeakEvent(
                    timestamp=f"2025-{month:02d}-15T14:00:00Z",
                    demand_kw=round(peak_kw, 2),
                    duration_minutes=15,
                    cause=cause,
                    month=month,
                    hour_of_day=14,
                    is_billing_peak=is_billing,
                    estimated_cost_impact=round(cost_impact, 2),
                ))

        # From interval data: identify top N peaks
        if input_data.interval_data:
            sorted_intervals = sorted(
                input_data.interval_data,
                key=lambda r: r.demand_kw,
                reverse=True,
            )
            top_n = min(10, len(sorted_intervals))
            for r in sorted_intervals[:top_n]:
                if r.demand_kw >= peak_threshold:
                    cause = self._classify_peak_cause(
                        r.month, r.demand_kw, overall_peak
                    )
                    peak_events.append(PeakEvent(
                        timestamp=r.timestamp,
                        demand_kw=round(r.demand_kw, 2),
                        duration_minutes=15,
                        cause=cause,
                        month=r.month,
                        hour_of_day=r.hour_of_day,
                        is_billing_peak=abs(r.demand_kw - overall_peak) < 1.0,
                        estimated_cost_impact=round(
                            r.demand_kw * input_data.demand_charge_rate, 2
                        ),
                    ))

        # Deduplicate by keeping highest per month
        seen_months: Dict[int, PeakEvent] = {}
        for evt in peak_events:
            if evt.month not in seen_months or evt.demand_kw > seen_months[evt.month].demand_kw:
                seen_months[evt.month] = evt
        peak_events = sorted(seen_months.values(), key=lambda e: e.demand_kw, reverse=True)

        self._peak_events = peak_events

        # Peak cause distribution
        cause_dist: Dict[str, int] = {}
        for evt in peak_events:
            cause_dist[evt.cause.value] = cause_dist.get(evt.cause.value, 0) + 1

        outputs["peak_events_identified"] = len(peak_events)
        outputs["peak_threshold_kw"] = round(peak_threshold, 2)
        outputs["cause_distribution"] = cause_dist
        outputs["billing_peak_count"] = sum(1 for e in peak_events if e.is_billing_peak)
        outputs["total_peak_cost_impact"] = round(
            sum(e.estimated_cost_impact for e in peak_events), 2
        )

        elapsed = time.perf_counter() - t_start
        self.logger.info(
            "Phase 2 PeakIdentification: %d events above %.0f kW threshold (%.3fs)",
            len(peak_events), peak_threshold, elapsed,
        )
        return PhaseResult(
            phase_name="peak_identification", phase_number=2,
            status=PhaseStatus.COMPLETED, duration_seconds=round(elapsed, 4),
            outputs=outputs, warnings=warnings,
            provenance_hash=_compute_hash(outputs),
        )

    def _classify_peak_cause(
        self, month: int, peak_kw: float, overall_peak: float
    ) -> PeakCause:
        """Classify peak demand event cause based on temporal patterns.

        Args:
            month: Month of peak event.
            peak_kw: Peak demand value.
            overall_peak: Annual overall peak.

        Returns:
            PeakCause classification.
        """
        summer_months = {6, 7, 8}
        winter_months = {12, 1, 2}

        if month in summer_months and peak_kw > overall_peak * 0.90:
            return PeakCause.HVAC_COOLING
        elif month in winter_months and peak_kw > overall_peak * 0.85:
            return PeakCause.HVAC_HEATING
        elif peak_kw > overall_peak * 0.95:
            return PeakCause.COINCIDENT_LOADS
        elif month in summer_months:
            return PeakCause.WEATHER_DRIVEN
        else:
            return PeakCause.OCCUPANCY_DRIVEN

    # -------------------------------------------------------------------------
    # Phase 3: Strategy Development
    # -------------------------------------------------------------------------

    def _phase_3_strategy_development(
        self, input_data: DemandManagementInput
    ) -> PhaseResult:
        """Develop demand reduction strategies with cost-benefit analysis.

        Args:
            input_data: Demand management input.

        Returns:
            PhaseResult with strategy outputs.
        """
        t_start = time.perf_counter()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}
        strategies: List[DemandReductionStrategy] = []

        overall_peak = self._demand_profile.get("overall_peak_kw", 0.0)
        load_factor = self._demand_profile.get("load_factor", 0.0)
        demand_rate = input_data.demand_charge_rate

        if overall_peak <= 0:
            warnings.append("Peak demand is zero; no strategies applicable")
            elapsed = time.perf_counter() - t_start
            return PhaseResult(
                phase_name="strategy_development", phase_number=3,
                status=PhaseStatus.COMPLETED, duration_seconds=round(elapsed, 4),
                outputs=outputs, warnings=warnings,
                provenance_hash=_compute_hash(outputs),
            )

        # Evaluate each strategy type
        for strat_type in StrategyType:
            reduction_range = STRATEGY_REDUCTION_POTENTIAL.get(
                strat_type.value, (0.05, 0.15)
            )
            cost_range = STRATEGY_COST_PER_KW.get(
                strat_type.value, (50.0, 200.0)
            )

            # Use midpoint estimates
            reduction_pct = (reduction_range[0] + reduction_range[1]) / 2.0
            reduction_kw = overall_peak * reduction_pct
            cost_per_kw = (cost_range[0] + cost_range[1]) / 2.0

            annual_savings = reduction_kw * demand_rate * 12.0
            impl_cost = reduction_kw * cost_per_kw
            payback = impl_cost / annual_savings if annual_savings > 0 else 999.0

            # Load factor improvement estimate
            new_peak = overall_peak - reduction_kw
            new_lf = (
                input_data.annual_consumption_kwh / (new_peak * 8760.0)
                if new_peak > 0 and input_data.annual_consumption_kwh > 0
                else load_factor
            )
            lf_improvement = min(1.0, new_lf) - load_factor

            # Priority based on payback
            if payback <= 1.0:
                priority = StrategyPriority.CRITICAL
            elif payback <= 3.0:
                priority = StrategyPriority.HIGH
            elif payback <= 7.0:
                priority = StrategyPriority.MEDIUM
            else:
                priority = StrategyPriority.LOW

            # Prerequisites
            prereqs = self._get_strategy_prerequisites(strat_type, input_data)

            # Timeline
            timelines: Dict[str, str] = {
                "load_shifting": "1-3 months",
                "peak_shaving": "3-6 months",
                "demand_response": "1-2 months",
                "battery_storage": "6-12 months",
                "operational_scheduling": "1-2 months",
                "hvac_optimization": "2-4 months",
                "motor_soft_start": "1-3 months",
                "load_sequencing": "1-2 months",
                "thermal_storage": "6-12 months",
                "on_site_generation": "12-18 months",
            }

            strategies.append(DemandReductionStrategy(
                strategy_type=strat_type,
                priority=priority,
                description=self._get_strategy_description(strat_type),
                target_reduction_kw=round(reduction_kw, 2),
                target_reduction_pct=round(reduction_pct * 100.0, 2),
                annual_savings=round(annual_savings, 2),
                implementation_cost=round(impl_cost, 2),
                simple_payback_years=round(payback, 2),
                load_factor_improvement=round(lf_improvement, 4),
                implementation_timeline=timelines.get(strat_type.value, "3-6 months"),
                prerequisites=prereqs,
            ))

        # Sort by payback ascending
        strategies.sort(key=lambda s: s.simple_payback_years)
        self._strategies = strategies

        outputs["strategies_developed"] = len(strategies)
        outputs["total_potential_savings"] = round(
            sum(s.annual_savings for s in strategies), 2
        )
        outputs["quick_wins"] = sum(
            1 for s in strategies if s.simple_payback_years <= 2.0
        )
        if strategies:
            outputs["best_strategy"] = strategies[0].strategy_type.value
            outputs["best_payback_years"] = strategies[0].simple_payback_years

        elapsed = time.perf_counter() - t_start
        self.logger.info(
            "Phase 3 StrategyDevelopment: %d strategies, %d quick wins (%.3fs)",
            len(strategies),
            outputs.get("quick_wins", 0),
            elapsed,
        )
        return PhaseResult(
            phase_name="strategy_development", phase_number=3,
            status=PhaseStatus.COMPLETED, duration_seconds=round(elapsed, 4),
            outputs=outputs, warnings=warnings,
            provenance_hash=_compute_hash(outputs),
        )

    def _get_strategy_description(self, strategy_type: StrategyType) -> str:
        """Get deterministic description for a strategy type.

        Args:
            strategy_type: Strategy classification.

        Returns:
            Description string.
        """
        descriptions: Dict[str, str] = {
            "load_shifting": "Shift non-critical loads to off-peak periods to reduce coincident peak demand",
            "peak_shaving": "Use demand limiting controls to cap peak demand at a target setpoint",
            "demand_response": "Enrol in utility demand response programme for curtailment payments",
            "battery_storage": "Deploy battery energy storage system for peak demand reduction",
            "operational_scheduling": "Stagger equipment start-up sequences to avoid coincident peaks",
            "hvac_optimization": "Optimise HVAC setpoints and staging to reduce demand spikes",
            "motor_soft_start": "Install variable frequency drives or soft starters on large motors",
            "load_sequencing": "Implement automated load sequencing to prevent simultaneous starts",
            "thermal_storage": "Use ice or chilled water storage to shift cooling demand off-peak",
            "on_site_generation": "Deploy on-site generation (CHP, solar+storage) for peak reduction",
        }
        return descriptions.get(strategy_type.value, "Implement demand reduction measures")

    def _get_strategy_prerequisites(
        self, strategy_type: StrategyType, input_data: DemandManagementInput
    ) -> List[str]:
        """Get prerequisites for a strategy type.

        Args:
            strategy_type: Strategy classification.
            input_data: Demand management input.

        Returns:
            List of prerequisite strings.
        """
        prereqs: List[str] = []
        if not input_data.has_interval_meters:
            prereqs.append("Install interval metering for demand monitoring")
        if not input_data.has_bms and strategy_type in (
            StrategyType.PEAK_SHAVING, StrategyType.LOAD_SEQUENCING,
            StrategyType.HVAC_OPTIMIZATION,
        ):
            prereqs.append("Install building management system (BMS/EMS)")
        if strategy_type == StrategyType.BATTERY_STORAGE:
            prereqs.append("Complete electrical infrastructure assessment")
            prereqs.append("Obtain interconnection agreement")
        if strategy_type == StrategyType.ON_SITE_GENERATION:
            prereqs.append("Complete feasibility study and permitting")
        return prereqs

    # -------------------------------------------------------------------------
    # Phase 4: Demand Report
    # -------------------------------------------------------------------------

    def _phase_4_demand_report(
        self, input_data: DemandManagementInput
    ) -> PhaseResult:
        """Generate demand management report.

        Args:
            input_data: Demand management input.

        Returns:
            PhaseResult with report outputs.
        """
        t_start = time.perf_counter()
        warnings: List[str] = []
        outputs: Dict[str, Any] = {}

        report_id = f"rpt-{uuid.uuid4().hex[:8]}"

        # Implementation roadmap
        roadmap: List[Dict[str, Any]] = []
        for i, strat in enumerate(self._strategies[:5], start=1):
            roadmap.append({
                "phase": i,
                "strategy": strat.strategy_type.value,
                "priority": strat.priority.value,
                "timeline": strat.implementation_timeline,
                "savings": round(strat.annual_savings, 2),
                "cost": round(strat.implementation_cost, 2),
                "payback_years": strat.simple_payback_years,
            })

        outputs["report_id"] = report_id
        outputs["generated_at"] = _utcnow().isoformat()
        outputs["facility_id"] = input_data.facility_id
        outputs["current_peak_kw"] = self._demand_profile.get("overall_peak_kw", 0.0)
        outputs["current_load_factor"] = self._demand_profile.get("load_factor", 0.0)
        outputs["demand_charge_rate"] = input_data.demand_charge_rate
        outputs["peak_events_count"] = len(self._peak_events)
        outputs["strategies_count"] = len(self._strategies)
        outputs["implementation_roadmap"] = roadmap
        outputs["methodology"] = [
            "Interval demand profile analysis",
            "Statistical peak event detection and classification",
            "ASHRAE/EPRI demand reduction benchmarks",
            "Deterministic cost-benefit analysis",
            "Load factor improvement calculation",
        ]

        elapsed = time.perf_counter() - t_start
        self.logger.info(
            "Phase 4 DemandReport: report=%s, %d roadmap items (%.3fs)",
            report_id, len(roadmap), elapsed,
        )
        return PhaseResult(
            phase_name="demand_report", phase_number=4,
            status=PhaseStatus.COMPLETED, duration_seconds=round(elapsed, 4),
            outputs=outputs, warnings=warnings,
            provenance_hash=_compute_hash(outputs),
        )
