"""
GL-003 UNIFIEDSTEAM - Extended Trap Maintenance Optimizer

Provides advanced trap maintenance optimization capabilities:
- PredictiveMaintenanceScheduler: ML-free predictive scheduling
- TrapReplacementPrioritizer: Enhanced prioritization with cost-benefit
- CostBenefitAnalyzer: Detailed ROI analysis
- SparePartsOptimizer: Inventory optimization with ABC analysis

All optimizations include confidence scores and explainability.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import hashlib
import logging
import math
import time

from pydantic import BaseModel, Field, validator

from .trap_maintenance_optimizer import (
    TrapType,
    TrapStatus,
    TrapCriticality,
    TrapData,
    TrapFleet,
    FailurePrediction,
    DowntimeConstraint,
    InspectionTask,
    InspectionSchedule,
    ReplacementTask,
    ReplacementPriority,
    SparePartItem,
    SparesPlan,
    TrapMaintenanceOptimizer,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Predictive Maintenance Models
# =============================================================================


class MaintenanceActionType(str, Enum):
    """Type of maintenance action."""

    INSPECTION = "inspection"
    MINOR_REPAIR = "minor_repair"
    MAJOR_REPAIR = "major_repair"
    REPLACEMENT = "replacement"
    RECALIBRATION = "recalibration"


class MaintenanceWindow(BaseModel):
    """Maintenance window definition."""

    window_id: str = Field(..., description="Window identifier")
    start_time: datetime = Field(..., description="Window start time")
    end_time: datetime = Field(..., description="Window end time")
    available_labor_hours: float = Field(default=8.0, description="Available labor hours")
    crew_size: int = Field(default=2, description="Crew size")
    equipment_available: List[str] = Field(
        default_factory=list, description="Available equipment"
    )


class PredictiveMaintenanceTask(BaseModel):
    """A predicted maintenance task."""

    task_id: str = Field(..., description="Task identifier")
    trap_id: str = Field(..., description="Trap identifier")
    action_type: MaintenanceActionType = Field(..., description="Action type")
    predicted_failure_date: datetime = Field(..., description="Predicted failure date")
    optimal_maintenance_date: datetime = Field(..., description="Optimal maintenance date")
    confidence: float = Field(..., ge=0, le=1, description="Prediction confidence")
    priority: int = Field(..., ge=1, le=5, description="Priority (1=highest)")

    # Cost analysis
    cost_if_reactive: float = Field(default=0.0, description="Cost if failure occurs")
    cost_if_proactive: float = Field(default=0.0, description="Cost of proactive maintenance")
    savings_potential: float = Field(default=0.0, description="Potential savings")

    # Explainability
    prediction_factors: List[str] = Field(
        default_factory=list, description="Factors driving prediction"
    )
    explanation: str = Field(default="", description="Human-readable explanation")

    # Provenance
    provenance_hash: str = Field(default="", description="SHA-256 hash")


class PredictiveMaintenanceSchedule(BaseModel):
    """Predictive maintenance schedule."""

    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    schedule_horizon_days: int = Field(default=90, description="Schedule horizon (days)")
    tasks: List[PredictiveMaintenanceTask] = Field(
        default_factory=list, description="Scheduled tasks"
    )

    # Summary metrics
    total_tasks: int = Field(default=0, description="Total tasks")
    total_proactive_cost: float = Field(default=0.0, description="Total proactive cost ($)")
    total_avoided_cost: float = Field(default=0.0, description="Total avoided cost ($)")
    net_savings: float = Field(default=0.0, description="Net savings ($)")
    average_confidence: float = Field(default=0.0, description="Average confidence")

    # By time period
    tasks_by_week: Dict[str, int] = Field(
        default_factory=dict, description="Tasks per week"
    )

    provenance_hash: str = Field(default="", description="SHA-256 hash")


# =============================================================================
# Cost-Benefit Analysis Models
# =============================================================================


class CostCategory(str, Enum):
    """Cost category."""

    STEAM_LOSS = "steam_loss"
    LABOR = "labor"
    MATERIALS = "materials"
    DOWNTIME = "downtime"
    ENERGY = "energy"
    ENVIRONMENTAL = "environmental"


class CostComponent(BaseModel):
    """A single cost component."""

    category: CostCategory = Field(..., description="Cost category")
    description: str = Field(..., description="Cost description")
    amount: float = Field(..., description="Cost amount ($)")
    is_one_time: bool = Field(default=False, description="One-time vs recurring")
    recurrence_period: str = Field(default="annual", description="Recurrence period")
    confidence: float = Field(default=0.90, ge=0, le=1, description="Cost confidence")
    assumptions: List[str] = Field(default_factory=list, description="Assumptions")


class CostBenefitAnalysis(BaseModel):
    """Complete cost-benefit analysis."""

    analysis_id: str = Field(..., description="Analysis identifier")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    subject_id: str = Field(..., description="Subject (trap_id, project_id, etc.)")
    subject_type: str = Field(default="trap", description="Subject type")

    # Costs
    costs: List[CostComponent] = Field(
        default_factory=list, description="Cost components"
    )
    total_one_time_cost: float = Field(default=0.0, description="Total one-time costs ($)")
    total_annual_cost: float = Field(default=0.0, description="Total annual costs ($)")

    # Benefits
    benefits: List[CostComponent] = Field(
        default_factory=list, description="Benefit components"
    )
    total_annual_benefit: float = Field(default=0.0, description="Total annual benefits ($)")

    # ROI metrics
    simple_payback_months: float = Field(default=0.0, description="Simple payback (months)")
    roi_percent: float = Field(default=0.0, description="ROI percentage")
    npv_10_years: float = Field(default=0.0, description="10-year NPV ($)")
    irr_percent: float = Field(default=0.0, description="IRR percentage")

    # Sensitivity
    best_case_roi: float = Field(default=0.0, description="Best case ROI")
    worst_case_roi: float = Field(default=0.0, description="Worst case ROI")
    breakeven_assumptions: List[str] = Field(
        default_factory=list, description="Breakeven assumptions"
    )

    # Recommendation
    recommendation: str = Field(default="", description="Investment recommendation")
    confidence: float = Field(default=0.85, ge=0, le=1, description="Analysis confidence")

    provenance_hash: str = Field(default="", description="SHA-256 hash")


# =============================================================================
# Predictive Maintenance Scheduler
# =============================================================================


class PredictiveMaintenanceScheduler:
    """
    Schedules predictive maintenance using deterministic failure modeling.

    Uses Weibull-based failure prediction (no ML):
    - Historical failure rates by trap type
    - Operating condition adjustments
    - Age-based degradation curves
    - Criticality-weighted scheduling

    All predictions include confidence scores and explainability.
    """

    # Weibull parameters by trap type (shape, scale in operating hours)
    WEIBULL_PARAMS = {
        TrapType.THERMODYNAMIC: (2.0, 15000),
        TrapType.THERMOSTATIC: (2.2, 20000),
        TrapType.MECHANICAL: (1.8, 25000),
        TrapType.INVERTED_BUCKET: (2.5, 30000),
        TrapType.FLOAT_THERMOSTATIC: (2.3, 22000),
        TrapType.BIMETALLIC: (2.0, 18000),
        TrapType.UNKNOWN: (2.0, 20000),
    }

    # Operating hours per year assumption
    DEFAULT_OPERATING_HOURS_PER_YEAR = 8000

    # Cost factors
    REACTIVE_COST_MULTIPLIER = 3.0  # Reactive maintenance costs 3x more

    def __init__(
        self,
        base_optimizer: Optional[TrapMaintenanceOptimizer] = None,
        operating_hours_per_year: int = DEFAULT_OPERATING_HOURS_PER_YEAR,
        planning_horizon_days: int = 90,
    ) -> None:
        """
        Initialize predictive maintenance scheduler.

        Args:
            base_optimizer: Base trap maintenance optimizer
            operating_hours_per_year: Annual operating hours
            planning_horizon_days: Planning horizon in days
        """
        self.base_optimizer = base_optimizer or TrapMaintenanceOptimizer()
        self.operating_hours_per_year = operating_hours_per_year
        self.planning_horizon_days = planning_horizon_days

        logger.info("PredictiveMaintenanceScheduler initialized")

    def create_predictive_schedule(
        self,
        trap_fleet: TrapFleet,
        maintenance_windows: List[MaintenanceWindow],
        budget: float = 50000.0,
    ) -> PredictiveMaintenanceSchedule:
        """
        Create a predictive maintenance schedule.

        Args:
            trap_fleet: Current trap fleet
            maintenance_windows: Available maintenance windows
            budget: Available budget ($)

        Returns:
            PredictiveMaintenanceSchedule
        """
        start_time = time.perf_counter()
        tasks: List[PredictiveMaintenanceTask] = []
        now = datetime.now(timezone.utc)

        # Generate failure predictions for all traps
        for trap in trap_fleet.traps:
            prediction = self._predict_failure(trap, now)

            # Only schedule if predicted failure within horizon
            if prediction.days_to_failure <= self.planning_horizon_days:
                task = self._create_maintenance_task(trap, prediction, now)
                tasks.append(task)

        # Sort by priority and optimal date
        tasks.sort(key=lambda t: (t.priority, t.optimal_maintenance_date))

        # Assign to maintenance windows
        tasks = self._assign_to_windows(tasks, maintenance_windows)

        # Apply budget constraint
        cumulative_cost = 0.0
        within_budget_tasks = []
        for task in tasks:
            if cumulative_cost + task.cost_if_proactive <= budget:
                within_budget_tasks.append(task)
                cumulative_cost += task.cost_if_proactive

        # Calculate metrics
        total_proactive_cost = sum(t.cost_if_proactive for t in within_budget_tasks)
        total_avoided_cost = sum(t.cost_if_reactive for t in within_budget_tasks)
        net_savings = total_avoided_cost - total_proactive_cost
        avg_confidence = (
            sum(t.confidence for t in within_budget_tasks) / len(within_budget_tasks)
            if within_budget_tasks else 0.0
        )

        # Group by week
        tasks_by_week: Dict[str, int] = {}
        for task in within_budget_tasks:
            week = task.optimal_maintenance_date.strftime("%Y-W%W")
            tasks_by_week[week] = tasks_by_week.get(week, 0) + 1

        schedule = PredictiveMaintenanceSchedule(
            schedule_horizon_days=self.planning_horizon_days,
            tasks=within_budget_tasks,
            total_tasks=len(within_budget_tasks),
            total_proactive_cost=round(total_proactive_cost, 2),
            total_avoided_cost=round(total_avoided_cost, 2),
            net_savings=round(net_savings, 2),
            average_confidence=round(avg_confidence, 3),
            tasks_by_week=tasks_by_week,
        )

        # Generate provenance hash
        hash_data = f"{len(tasks)}{total_proactive_cost}{schedule.timestamp.isoformat()}"
        schedule.provenance_hash = hashlib.sha256(hash_data.encode()).hexdigest()

        computation_time = (time.perf_counter() - start_time) * 1000
        logger.info(
            f"Predictive schedule: {len(within_budget_tasks)} tasks, "
            f"net savings=${net_savings:.0f}, in {computation_time:.1f}ms"
        )

        return schedule

    def _predict_failure(
        self,
        trap: TrapData,
        now: datetime,
    ) -> "FailurePredictionResult":
        """Predict failure using Weibull distribution."""
        # Get Weibull parameters
        shape, scale = self.WEIBULL_PARAMS.get(
            trap.trap_type, self.WEIBULL_PARAMS[TrapType.UNKNOWN]
        )

        # Adjust scale for operating conditions
        condition_factor = 1.0
        if trap.inlet_pressure_psig > 200:
            condition_factor *= 0.85  # Higher pressure = shorter life
        if trap.condensate_load_lb_hr > 500:
            condition_factor *= 0.90  # Higher load = shorter life
        if trap.failure_count > 2:
            condition_factor *= 0.80  # History of failures

        adjusted_scale = scale * condition_factor

        # Calculate current age in operating hours
        if trap.install_date:
            years_installed = (now - trap.install_date).days / 365
            current_hours = years_installed * self.operating_hours_per_year
        else:
            current_hours = trap.operating_hours

        # Calculate remaining useful life (RUL) using Weibull reliability
        # R(t) = exp(-(t/scale)^shape)
        # Find time where R(t) = 0.1 (90% probability of failure)
        target_reliability = 0.1
        time_to_90_pct_failure = adjusted_scale * (
            -math.log(target_reliability)
        ) ** (1 / shape)

        remaining_hours = max(0, time_to_90_pct_failure - current_hours)
        days_to_failure = remaining_hours / (self.operating_hours_per_year / 365)

        # Calculate confidence based on data quality
        confidence = 0.85
        if trap.operating_hours > 5000:
            confidence = 0.90
        if trap.operating_hours < 1000:
            confidence = 0.70
        if trap.failure_count > 0:
            confidence -= 0.05 * trap.failure_count

        # Build explanation
        factors = []
        if years_installed > 5:
            factors.append(f"Age ({years_installed:.1f} years)")
        if condition_factor < 1.0:
            factors.append(f"Operating stress (factor: {condition_factor:.2f})")
        if trap.failure_count > 0:
            factors.append(f"Prior failures ({trap.failure_count})")
        if trap.status == TrapStatus.SUSPECT:
            factors.append("Currently suspect")

        return FailurePredictionResult(
            trap_id=trap.trap_id,
            days_to_failure=days_to_failure,
            remaining_hours=remaining_hours,
            failure_probability_horizon=1 - math.exp(
                -((current_hours + remaining_hours) / adjusted_scale) ** shape
            ),
            confidence=max(0.5, min(0.95, confidence)),
            factors=factors,
        )

    def _create_maintenance_task(
        self,
        trap: TrapData,
        prediction: "FailurePredictionResult",
        now: datetime,
    ) -> PredictiveMaintenanceTask:
        """Create a maintenance task from prediction."""
        import uuid

        # Determine action type based on trap condition
        if trap.status in (TrapStatus.FAILED_OPEN, TrapStatus.FAILED_CLOSED):
            action_type = MaintenanceActionType.REPLACEMENT
        elif trap.status == TrapStatus.LEAKING:
            action_type = MaintenanceActionType.MAJOR_REPAIR
        elif trap.status == TrapStatus.SUSPECT:
            action_type = MaintenanceActionType.INSPECTION
        else:
            action_type = MaintenanceActionType.INSPECTION

        # Calculate optimal maintenance date (before predicted failure)
        lead_time_days = max(7, prediction.days_to_failure * 0.7)
        optimal_date = now + timedelta(days=lead_time_days)

        # Calculate costs
        proactive_cost = trap.replacement_cost + trap.labor_hours * self.base_optimizer.labor_rate
        reactive_cost = proactive_cost * self.REACTIVE_COST_MULTIPLIER

        # Add steam loss cost for reactive scenario
        if trap.status != TrapStatus.GOOD:
            steam_loss = self.base_optimizer._calculate_steam_loss(trap)
            annual_loss_cost = (
                steam_loss / 1000 *
                self.base_optimizer.steam_cost_per_klb *
                self.base_optimizer.operating_hours
            )
            reactive_cost += annual_loss_cost * (prediction.days_to_failure / 365)

        # Calculate priority
        priority = self._calculate_priority(trap, prediction)

        # Generate explanation
        explanation = (
            f"Predicted failure in {prediction.days_to_failure:.0f} days "
            f"with {prediction.confidence:.0%} confidence. "
            f"Recommended {action_type.value} by {optimal_date.strftime('%Y-%m-%d')}. "
            f"Savings potential: ${reactive_cost - proactive_cost:.0f}"
        )

        # Generate provenance hash
        hash_data = f"{trap.trap_id}{prediction.days_to_failure}{now.isoformat()}"
        provenance = hashlib.sha256(hash_data.encode()).hexdigest()

        return PredictiveMaintenanceTask(
            task_id=f"PMT-{uuid.uuid4().hex[:8].upper()}",
            trap_id=trap.trap_id,
            action_type=action_type,
            predicted_failure_date=now + timedelta(days=prediction.days_to_failure),
            optimal_maintenance_date=optimal_date,
            confidence=prediction.confidence,
            priority=priority,
            cost_if_reactive=round(reactive_cost, 2),
            cost_if_proactive=round(proactive_cost, 2),
            savings_potential=round(reactive_cost - proactive_cost, 2),
            prediction_factors=prediction.factors,
            explanation=explanation,
            provenance_hash=provenance,
        )

    def _calculate_priority(
        self,
        trap: TrapData,
        prediction: "FailurePredictionResult",
    ) -> int:
        """Calculate task priority (1-5)."""
        score = 0

        # Criticality contribution
        crit_scores = {
            TrapCriticality.CRITICAL: 40,
            TrapCriticality.HIGH: 30,
            TrapCriticality.MEDIUM: 20,
            TrapCriticality.LOW: 10,
        }
        score += crit_scores.get(trap.criticality, 20)

        # Time to failure contribution
        if prediction.days_to_failure < 7:
            score += 40
        elif prediction.days_to_failure < 14:
            score += 30
        elif prediction.days_to_failure < 30:
            score += 20
        elif prediction.days_to_failure < 60:
            score += 10

        # Status contribution
        if trap.status in (TrapStatus.FAILED_OPEN, TrapStatus.FAILED_CLOSED):
            score += 20
        elif trap.status == TrapStatus.LEAKING:
            score += 15
        elif trap.status == TrapStatus.SUSPECT:
            score += 10

        # Convert to priority (1-5)
        if score >= 80:
            return 1
        elif score >= 60:
            return 2
        elif score >= 40:
            return 3
        elif score >= 20:
            return 4
        else:
            return 5

    def _assign_to_windows(
        self,
        tasks: List[PredictiveMaintenanceTask],
        windows: List[MaintenanceWindow],
    ) -> List[PredictiveMaintenanceTask]:
        """Assign tasks to maintenance windows."""
        if not windows:
            return tasks

        # Simple assignment: assign tasks to nearest available window
        window_loads: Dict[str, float] = {w.window_id: 0.0 for w in windows}
        window_capacity: Dict[str, float] = {
            w.window_id: w.available_labor_hours * w.crew_size
            for w in windows
        }

        for task in tasks:
            # Find best window for this task
            best_window = None
            best_score = float('inf')

            for window in windows:
                # Check capacity
                estimated_hours = 2.0  # Default task duration
                if window_loads[window.window_id] + estimated_hours > window_capacity[window.window_id]:
                    continue

                # Score based on proximity to optimal date
                days_diff = abs(
                    (window.start_time - task.optimal_maintenance_date).days
                )
                score = days_diff + window_loads[window.window_id] * 0.1

                if score < best_score:
                    best_score = score
                    best_window = window

            if best_window:
                window_loads[best_window.window_id] += 2.0
                # Update task date to window
                task.optimal_maintenance_date = best_window.start_time

        return tasks


@dataclass
class FailurePredictionResult:
    """Result of failure prediction."""

    trap_id: str
    days_to_failure: float
    remaining_hours: float
    failure_probability_horizon: float
    confidence: float
    factors: List[str] = field(default_factory=list)


# =============================================================================
# Cost-Benefit Analyzer
# =============================================================================


class CostBenefitAnalyzer:
    """
    Performs detailed cost-benefit analysis for trap maintenance decisions.

    Analyzes:
    - Steam loss costs
    - Labor costs
    - Material costs
    - Downtime costs
    - Environmental costs
    - ROI metrics
    """

    DEFAULT_DISCOUNT_RATE = 0.08  # 8% annual

    def __init__(
        self,
        steam_cost_per_klb: float = 10.0,
        labor_rate: float = 75.0,
        co2_cost_per_ton: float = 50.0,
        operating_hours: int = 8000,
    ) -> None:
        """
        Initialize cost-benefit analyzer.

        Args:
            steam_cost_per_klb: Steam cost ($/klb)
            labor_rate: Labor rate ($/hr)
            co2_cost_per_ton: CO2 cost ($/ton)
            operating_hours: Annual operating hours
        """
        self.steam_cost_per_klb = steam_cost_per_klb
        self.labor_rate = labor_rate
        self.co2_cost_per_ton = co2_cost_per_ton
        self.operating_hours = operating_hours

    def analyze_trap_replacement(
        self,
        trap: TrapData,
        current_steam_loss_lb_hr: float,
    ) -> CostBenefitAnalysis:
        """
        Analyze cost-benefit of replacing a trap.

        Args:
            trap: Trap data
            current_steam_loss_lb_hr: Current steam loss rate

        Returns:
            CostBenefitAnalysis
        """
        import uuid
        start_time = time.perf_counter()

        analysis_id = f"CBA-{uuid.uuid4().hex[:8].upper()}"
        costs: List[CostComponent] = []
        benefits: List[CostComponent] = []

        # === COSTS ===

        # Material cost (one-time)
        costs.append(CostComponent(
            category=CostCategory.MATERIALS,
            description="Replacement trap",
            amount=trap.replacement_cost,
            is_one_time=True,
            confidence=0.95,
            assumptions=["Standard OEM replacement"],
        ))

        # Labor cost (one-time)
        labor_cost = trap.labor_hours * self.labor_rate
        costs.append(CostComponent(
            category=CostCategory.LABOR,
            description="Installation labor",
            amount=labor_cost,
            is_one_time=True,
            confidence=0.90,
            assumptions=[f"Labor rate: ${self.labor_rate}/hr"],
        ))

        # === BENEFITS ===

        # Steam savings (annual)
        annual_steam_savings = (
            current_steam_loss_lb_hr / 1000 *
            self.steam_cost_per_klb *
            self.operating_hours
        )
        benefits.append(CostComponent(
            category=CostCategory.STEAM_LOSS,
            description="Eliminated steam loss",
            amount=annual_steam_savings,
            is_one_time=False,
            recurrence_period="annual",
            confidence=0.85,
            assumptions=[
                f"Steam cost: ${self.steam_cost_per_klb}/klb",
                f"Operating hours: {self.operating_hours}/year",
                f"Steam loss: {current_steam_loss_lb_hr:.1f} lb/hr",
            ],
        ))

        # Energy savings (from reduced boiler load)
        fuel_savings_mmbtu = current_steam_loss_lb_hr * self.operating_hours / 1000 * 1.1  # ~1.1 MMBTU/klb
        fuel_cost_per_mmbtu = 8.0  # Assume $8/MMBTU
        annual_fuel_savings = fuel_savings_mmbtu * fuel_cost_per_mmbtu
        benefits.append(CostComponent(
            category=CostCategory.ENERGY,
            description="Reduced boiler fuel",
            amount=annual_fuel_savings,
            is_one_time=False,
            recurrence_period="annual",
            confidence=0.80,
            assumptions=[f"Fuel cost: ${fuel_cost_per_mmbtu}/MMBTU"],
        ))

        # Environmental benefit (CO2 reduction)
        co2_tons = fuel_savings_mmbtu * 0.0585  # ~0.0585 tons CO2/MMBTU natural gas
        annual_co2_savings = co2_tons * self.co2_cost_per_ton
        if annual_co2_savings > 0:
            benefits.append(CostComponent(
                category=CostCategory.ENVIRONMENTAL,
                description="CO2 reduction value",
                amount=annual_co2_savings,
                is_one_time=False,
                recurrence_period="annual",
                confidence=0.70,
                assumptions=[f"CO2 cost: ${self.co2_cost_per_ton}/ton"],
            ))

        # === CALCULATE METRICS ===

        total_one_time = sum(c.amount for c in costs if c.is_one_time)
        total_annual_cost = sum(c.amount for c in costs if not c.is_one_time)
        total_annual_benefit = sum(b.amount for b in benefits if not b.is_one_time)
        net_annual_benefit = total_annual_benefit - total_annual_cost

        # Simple payback
        simple_payback_months = (
            total_one_time / (net_annual_benefit / 12)
            if net_annual_benefit > 0 else float('inf')
        )

        # ROI
        roi_percent = (
            (net_annual_benefit - total_one_time / 10) / total_one_time * 100
            if total_one_time > 0 else 0
        )

        # NPV (10 years)
        npv = -total_one_time
        for year in range(1, 11):
            npv += net_annual_benefit / (1 + self.DEFAULT_DISCOUNT_RATE) ** year

        # Simplified IRR estimation
        irr = 0.0
        if total_one_time > 0 and net_annual_benefit > 0:
            # Quick IRR estimate using payback
            irr = (1 / (simple_payback_months / 12)) - 1 if simple_payback_months < 120 else 0
            irr = max(0, min(1, irr)) * 100

        # Sensitivity analysis
        best_case_roi = roi_percent * 1.3  # 30% better
        worst_case_roi = roi_percent * 0.7  # 30% worse

        # Recommendation
        if simple_payback_months < 6:
            recommendation = "STRONGLY RECOMMENDED: Excellent payback under 6 months"
        elif simple_payback_months < 12:
            recommendation = "RECOMMENDED: Good payback within 1 year"
        elif simple_payback_months < 24:
            recommendation = "CONSIDER: Reasonable payback within 2 years"
        else:
            recommendation = "LOW PRIORITY: Payback exceeds 2 years"

        # Overall confidence
        confidence = sum(c.confidence * c.amount for c in costs + benefits) / sum(
            c.amount for c in costs + benefits
        ) if costs or benefits else 0.85

        # Generate provenance hash
        hash_data = f"{trap.trap_id}{total_one_time}{net_annual_benefit}{datetime.now().isoformat()}"
        provenance = hashlib.sha256(hash_data.encode()).hexdigest()

        analysis = CostBenefitAnalysis(
            analysis_id=analysis_id,
            subject_id=trap.trap_id,
            subject_type="trap_replacement",
            costs=costs,
            total_one_time_cost=round(total_one_time, 2),
            total_annual_cost=round(total_annual_cost, 2),
            benefits=benefits,
            total_annual_benefit=round(total_annual_benefit, 2),
            simple_payback_months=round(simple_payback_months, 1),
            roi_percent=round(roi_percent, 1),
            npv_10_years=round(npv, 2),
            irr_percent=round(irr, 1),
            best_case_roi=round(best_case_roi, 1),
            worst_case_roi=round(worst_case_roi, 1),
            breakeven_assumptions=[
                f"Steam loss maintained at {current_steam_loss_lb_hr:.1f} lb/hr",
                f"Steam cost stable at ${self.steam_cost_per_klb}/klb",
                f"Operating hours: {self.operating_hours}/year",
            ],
            recommendation=recommendation,
            confidence=round(confidence, 2),
            provenance_hash=provenance,
        )

        computation_time = (time.perf_counter() - start_time) * 1000
        logger.info(
            f"Cost-benefit analysis for {trap.trap_id}: "
            f"payback={simple_payback_months:.1f} months, ROI={roi_percent:.0f}%, "
            f"in {computation_time:.1f}ms"
        )

        return analysis


# =============================================================================
# Spare Parts Optimizer
# =============================================================================


class InventoryClassification(str, Enum):
    """ABC inventory classification."""

    A = "A"  # High value, tight control
    B = "B"  # Medium value, moderate control
    C = "C"  # Low value, loose control


class OptimizedSparePartItem(BaseModel):
    """Enhanced spare part item with optimization details."""

    part_number: str = Field(..., description="Part number")
    trap_type: TrapType = Field(..., description="Applicable trap type")
    description: str = Field(..., description="Part description")

    # Current state
    current_quantity: int = Field(default=0, ge=0, description="Current stock")
    current_value: float = Field(default=0.0, ge=0, description="Current inventory value ($)")

    # Optimized recommendations
    recommended_quantity: int = Field(default=0, ge=0, description="Recommended stock")
    reorder_point: int = Field(default=0, ge=0, description="Reorder point")
    economic_order_quantity: int = Field(default=0, ge=0, description="EOQ")
    safety_stock: int = Field(default=0, ge=0, description="Safety stock")

    # Cost analysis
    unit_cost: float = Field(default=0.0, ge=0, description="Unit cost ($)")
    annual_holding_cost: float = Field(default=0.0, ge=0, description="Annual holding cost ($)")
    annual_ordering_cost: float = Field(default=0.0, ge=0, description="Annual ordering cost ($)")
    total_annual_cost: float = Field(default=0.0, ge=0, description="Total annual inventory cost ($)")

    # Classification
    classification: InventoryClassification = Field(
        default=InventoryClassification.C, description="ABC classification"
    )
    annual_usage: int = Field(default=0, ge=0, description="Annual usage")
    lead_time_days: int = Field(default=14, ge=0, description="Lead time (days)")

    # Risk
    stockout_risk: str = Field(default="low", description="Stockout risk level")
    criticality_score: float = Field(default=0.0, description="Criticality score (0-100)")

    # Explainability
    optimization_rationale: str = Field(default="", description="Optimization rationale")


class SparePartsOptimizer:
    """
    Optimizes spare parts inventory with ABC analysis.

    Features:
    - ABC classification based on value and criticality
    - EOQ calculation with safety stock
    - Service level optimization
    - Criticality-weighted stocking
    """

    def __init__(
        self,
        target_service_level: float = 0.95,
        holding_cost_rate: float = 0.25,
        ordering_cost: float = 50.0,
    ) -> None:
        """
        Initialize spare parts optimizer.

        Args:
            target_service_level: Target service level (0-1)
            holding_cost_rate: Annual holding cost as % of item value
            ordering_cost: Cost per order ($)
        """
        self.target_service_level = target_service_level
        self.holding_cost_rate = holding_cost_rate
        self.ordering_cost = ordering_cost

    def optimize_inventory(
        self,
        current_inventory: Dict[str, int],
        trap_fleet_composition: Dict[TrapType, int],
        failure_rates: Dict[TrapType, float],
        part_costs: Dict[str, float],
        lead_times: Dict[str, int],
    ) -> List[OptimizedSparePartItem]:
        """
        Optimize spare parts inventory.

        Args:
            current_inventory: Current stock by part number
            trap_fleet_composition: Number of traps by type
            failure_rates: Annual failure rates by trap type
            part_costs: Unit costs by part number
            lead_times: Lead times by part number (days)

        Returns:
            List of optimized spare part items
        """
        start_time = time.perf_counter()
        items: List[OptimizedSparePartItem] = []

        # Map trap types to parts
        type_to_part = {
            TrapType.THERMODYNAMIC: "TD-001",
            TrapType.THERMOSTATIC: "TS-001",
            TrapType.MECHANICAL: "MC-001",
            TrapType.INVERTED_BUCKET: "IB-001",
            TrapType.FLOAT_THERMOSTATIC: "FT-001",
            TrapType.BIMETALLIC: "BM-001",
        }

        # Calculate annual demand and value for each part
        part_demands: Dict[str, float] = {}
        part_values: Dict[str, float] = {}

        for trap_type, count in trap_fleet_composition.items():
            if trap_type == TrapType.UNKNOWN or count == 0:
                continue

            part_number = type_to_part.get(trap_type)
            if not part_number:
                continue

            # Annual demand based on failure rate
            failure_rate = failure_rates.get(trap_type, 0.10)
            annual_demand = count * failure_rate
            part_demands[part_number] = annual_demand

            # Annual value
            unit_cost = part_costs.get(part_number, 300.0)
            part_values[part_number] = annual_demand * unit_cost

        # ABC Classification
        total_value = sum(part_values.values())
        sorted_parts = sorted(part_values.items(), key=lambda x: x[1], reverse=True)

        cumulative_value = 0.0
        classifications: Dict[str, InventoryClassification] = {}

        for part_number, value in sorted_parts:
            cumulative_value += value
            pct = cumulative_value / total_value if total_value > 0 else 0

            if pct <= 0.70:
                classifications[part_number] = InventoryClassification.A
            elif pct <= 0.90:
                classifications[part_number] = InventoryClassification.B
            else:
                classifications[part_number] = InventoryClassification.C

        # Optimize each part
        for trap_type, count in trap_fleet_composition.items():
            if trap_type == TrapType.UNKNOWN or count == 0:
                continue

            part_number = type_to_part.get(trap_type)
            if not part_number:
                continue

            annual_demand = part_demands.get(part_number, 0)
            unit_cost = part_costs.get(part_number, 300.0)
            lead_time = lead_times.get(part_number, 14)
            classification = classifications.get(part_number, InventoryClassification.C)

            # Calculate EOQ
            eoq = self._calculate_eoq(annual_demand, unit_cost)

            # Calculate safety stock based on service level and classification
            safety_stock = self._calculate_safety_stock(
                annual_demand, lead_time, classification
            )

            # Reorder point
            daily_demand = annual_demand / 365
            reorder_point = int(math.ceil(daily_demand * lead_time + safety_stock))

            # Recommended quantity
            recommended_qty = max(safety_stock + 1, reorder_point, int(eoq * 0.5))

            # Costs
            holding_cost = recommended_qty * unit_cost * self.holding_cost_rate
            orders_per_year = annual_demand / eoq if eoq > 0 else 1
            ordering_cost = orders_per_year * self.ordering_cost
            total_annual_cost = holding_cost + ordering_cost

            # Current state
            current_qty = current_inventory.get(part_number, 0)
            current_value = current_qty * unit_cost

            # Stockout risk
            if current_qty < safety_stock:
                stockout_risk = "high"
            elif current_qty < reorder_point:
                stockout_risk = "medium"
            else:
                stockout_risk = "low"

            # Criticality score (higher for A items and critical applications)
            criticality_score = 0.0
            if classification == InventoryClassification.A:
                criticality_score = 80.0
            elif classification == InventoryClassification.B:
                criticality_score = 50.0
            else:
                criticality_score = 25.0

            # Generate rationale
            rationale = self._generate_rationale(
                part_number, classification, stockout_risk,
                current_qty, recommended_qty, safety_stock
            )

            items.append(OptimizedSparePartItem(
                part_number=part_number,
                trap_type=trap_type,
                description=f"{trap_type.value} trap replacement",
                current_quantity=current_qty,
                current_value=round(current_value, 2),
                recommended_quantity=recommended_qty,
                reorder_point=reorder_point,
                economic_order_quantity=int(eoq),
                safety_stock=safety_stock,
                unit_cost=unit_cost,
                annual_holding_cost=round(holding_cost, 2),
                annual_ordering_cost=round(ordering_cost, 2),
                total_annual_cost=round(total_annual_cost, 2),
                classification=classification,
                annual_usage=int(annual_demand),
                lead_time_days=lead_time,
                stockout_risk=stockout_risk,
                criticality_score=criticality_score,
                optimization_rationale=rationale,
            ))

        computation_time = (time.perf_counter() - start_time) * 1000
        logger.info(
            f"Spare parts optimization: {len(items)} items optimized "
            f"in {computation_time:.1f}ms"
        )

        return items

    def _calculate_eoq(self, annual_demand: float, unit_cost: float) -> float:
        """Calculate Economic Order Quantity."""
        if annual_demand <= 0 or unit_cost <= 0:
            return 1.0

        holding_cost = unit_cost * self.holding_cost_rate
        eoq = math.sqrt(2 * annual_demand * self.ordering_cost / holding_cost)
        return max(1.0, eoq)

    def _calculate_safety_stock(
        self,
        annual_demand: float,
        lead_time_days: int,
        classification: InventoryClassification,
    ) -> int:
        """Calculate safety stock based on service level and classification."""
        # Z-scores for service levels
        z_scores = {
            0.90: 1.28,
            0.95: 1.65,
            0.98: 2.05,
            0.99: 2.33,
        }

        # Adjust service level by classification
        if classification == InventoryClassification.A:
            service_level = min(0.99, self.target_service_level + 0.02)
        elif classification == InventoryClassification.B:
            service_level = self.target_service_level
        else:
            service_level = max(0.90, self.target_service_level - 0.03)

        # Get z-score
        z = z_scores.get(service_level, 1.65)

        # Calculate safety stock
        daily_demand = annual_demand / 365
        demand_std = daily_demand * 0.2  # Assume 20% CV
        safety_stock = z * math.sqrt(lead_time_days) * demand_std

        return int(math.ceil(safety_stock))

    def _generate_rationale(
        self,
        part_number: str,
        classification: InventoryClassification,
        stockout_risk: str,
        current_qty: int,
        recommended_qty: int,
        safety_stock: int,
    ) -> str:
        """Generate optimization rationale."""
        parts = []

        parts.append(f"ABC Classification: {classification.value}")

        if stockout_risk == "high":
            parts.append(
                f"HIGH STOCKOUT RISK: Current stock ({current_qty}) "
                f"below safety stock ({safety_stock})"
            )
        elif stockout_risk == "medium":
            parts.append("REORDER NEEDED: Stock below reorder point")

        if current_qty < recommended_qty:
            parts.append(
                f"Recommend increasing stock by {recommended_qty - current_qty} units"
            )
        elif current_qty > recommended_qty * 1.5:
            parts.append("Consider reducing excess inventory")
        else:
            parts.append("Current stock level appropriate")

        return ". ".join(parts) + "."
