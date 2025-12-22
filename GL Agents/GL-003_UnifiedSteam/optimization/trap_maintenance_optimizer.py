"""
GL-003 UNIFIEDSTEAM - Trap Maintenance Optimizer

Provides optimization for steam trap maintenance:
- Inspection schedule optimization
- Replacement prioritization
- Spares inventory optimization

Objectives:
- Minimize steam losses from failed traps
- Minimize maintenance costs
- Maximize trap uptime and reliability
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

logger = logging.getLogger(__name__)


# =============================================================================
# Data Models
# =============================================================================


class TrapType(str, Enum):
    """Steam trap type."""

    THERMODYNAMIC = "thermodynamic"
    THERMOSTATIC = "thermostatic"
    MECHANICAL = "mechanical"
    INVERTED_BUCKET = "inverted_bucket"
    FLOAT_THERMOSTATIC = "float_thermostatic"
    BIMETALLIC = "bimetallic"
    UNKNOWN = "unknown"


class TrapStatus(str, Enum):
    """Current trap status."""

    GOOD = "good"
    LEAKING = "leaking"
    FAILED_OPEN = "failed_open"
    FAILED_CLOSED = "failed_closed"
    SUSPECT = "suspect"
    UNKNOWN = "unknown"


class TrapCriticality(str, Enum):
    """Trap criticality level."""

    CRITICAL = "critical"  # Production-critical process
    HIGH = "high"  # Important process
    MEDIUM = "medium"  # Standard process
    LOW = "low"  # Non-essential


class TrapData(BaseModel):
    """Data for a single steam trap."""

    trap_id: str = Field(..., description="Trap identifier")
    trap_type: TrapType = Field(..., description="Trap type")
    location: str = Field(default="", description="Physical location")
    application: str = Field(default="", description="Application type")
    criticality: TrapCriticality = Field(
        default=TrapCriticality.MEDIUM, description="Criticality level"
    )

    # Operating conditions
    inlet_pressure_psig: float = Field(
        default=100.0, description="Inlet pressure (psig)"
    )
    outlet_pressure_psig: float = Field(
        default=0.0, description="Outlet pressure (psig)"
    )
    condensate_load_lb_hr: float = Field(
        default=100.0, ge=0, description="Design condensate load (lb/hr)"
    )

    # Status
    status: TrapStatus = Field(
        default=TrapStatus.GOOD, description="Current status"
    )
    last_inspection_date: Optional[datetime] = Field(
        default=None, description="Last inspection date"
    )
    last_replacement_date: Optional[datetime] = Field(
        default=None, description="Last replacement date"
    )
    install_date: Optional[datetime] = Field(
        default=None, description="Installation date"
    )

    # Failure data
    failure_count: int = Field(
        default=0, ge=0, description="Total failure count"
    )
    operating_hours: float = Field(
        default=0.0, ge=0, description="Total operating hours"
    )

    # Cost data
    replacement_cost: float = Field(
        default=500.0, ge=0, description="Replacement cost ($)"
    )
    labor_hours: float = Field(
        default=2.0, ge=0, description="Labor hours for replacement"
    )


class TrapFleet(BaseModel):
    """Collection of steam traps."""

    traps: List[TrapData] = Field(
        default_factory=list, description="List of traps"
    )
    total_count: int = Field(default=0, description="Total trap count")
    failed_count: int = Field(default=0, description="Failed trap count")
    suspect_count: int = Field(default=0, description="Suspect trap count")
    total_steam_loss_lb_hr: float = Field(
        default=0.0, description="Total steam loss (lb/hr)"
    )


class FailurePrediction(BaseModel):
    """Failure prediction for a trap."""

    trap_id: str = Field(..., description="Trap identifier")
    failure_probability_30d: float = Field(
        ..., ge=0, le=1, description="30-day failure probability"
    )
    failure_probability_90d: float = Field(
        ..., ge=0, le=1, description="90-day failure probability"
    )
    failure_probability_365d: float = Field(
        ..., ge=0, le=1, description="365-day failure probability"
    )
    expected_remaining_life_days: float = Field(
        ..., ge=0, description="Expected remaining life (days)"
    )
    confidence: float = Field(
        default=0.80, ge=0, le=1, description="Prediction confidence"
    )
    risk_factors: List[str] = Field(
        default_factory=list, description="Contributing risk factors"
    )


class DowntimeConstraint(BaseModel):
    """Downtime constraints for maintenance."""

    max_downtime_hours_per_week: float = Field(
        default=8.0, description="Maximum downtime per week (hours)"
    )
    blackout_periods: List[Tuple[datetime, datetime]] = Field(
        default_factory=list, description="Periods when maintenance not allowed"
    )
    preferred_days: List[int] = Field(
        default_factory=lambda: [5, 6],  # Saturday, Sunday
        description="Preferred maintenance days (0=Monday)"
    )
    require_redundancy: bool = Field(
        default=True, description="Require process redundancy during maintenance"
    )


# =============================================================================
# Result Models
# =============================================================================


class InspectionTask(BaseModel):
    """A single inspection task."""

    trap_id: str = Field(..., description="Trap identifier")
    location: str = Field(..., description="Physical location")
    scheduled_date: datetime = Field(..., description="Scheduled inspection date")
    priority: int = Field(..., ge=1, le=5, description="Priority (1=highest)")
    estimated_duration_min: int = Field(
        default=15, description="Estimated duration (minutes)"
    )
    reason: str = Field(..., description="Reason for inspection")
    required_equipment: List[str] = Field(
        default_factory=list, description="Required inspection equipment"
    )


class InspectionSchedule(BaseModel):
    """Optimized inspection schedule."""

    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    schedule_horizon_days: int = Field(
        default=90, description="Schedule horizon (days)"
    )
    tasks: List[InspectionTask] = Field(
        default_factory=list, description="Scheduled inspection tasks"
    )
    total_inspection_hours: float = Field(
        default=0.0, description="Total inspection hours"
    )
    coverage_percent: float = Field(
        default=0.0, description="Fleet coverage (%)"
    )
    expected_failure_prevention: int = Field(
        default=0, description="Expected failures prevented"
    )
    optimization_score: float = Field(
        default=0.0, description="Schedule optimization score"
    )
    provenance_hash: str = Field(default="")


class ReplacementTask(BaseModel):
    """A single replacement task."""

    trap_id: str = Field(..., description="Trap identifier")
    location: str = Field(..., description="Physical location")
    trap_type: TrapType = Field(..., description="Trap type")
    current_status: TrapStatus = Field(..., description="Current status")
    priority: int = Field(..., ge=1, le=5, description="Priority (1=highest)")
    steam_loss_lb_hr: float = Field(
        default=0.0, description="Current steam loss (lb/hr)"
    )
    annual_loss_cost: float = Field(
        default=0.0, description="Annual loss cost ($)"
    )
    replacement_cost: float = Field(
        default=0.0, description="Replacement cost ($)"
    )
    payback_days: float = Field(
        default=0.0, description="Simple payback (days)"
    )
    recommended_date: Optional[datetime] = Field(
        default=None, description="Recommended replacement date"
    )


class ReplacementPriority(BaseModel):
    """Prioritized replacement list."""

    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    tasks: List[ReplacementTask] = Field(
        default_factory=list, description="Prioritized replacement tasks"
    )
    total_tasks: int = Field(default=0, description="Total replacement tasks")
    within_budget_tasks: int = Field(
        default=0, description="Tasks within budget"
    )
    total_steam_loss_lb_hr: float = Field(
        default=0.0, description="Total current steam loss (lb/hr)"
    )
    total_annual_loss_cost: float = Field(
        default=0.0, description="Total annual loss cost ($)"
    )
    total_replacement_cost: float = Field(
        default=0.0, description="Total replacement cost ($)"
    )
    budget_utilization_percent: float = Field(
        default=0.0, description="Budget utilization (%)"
    )
    annual_savings_if_complete: float = Field(
        default=0.0, description="Annual savings if all completed ($)"
    )
    provenance_hash: str = Field(default="")


class SparePartItem(BaseModel):
    """Spare part inventory item."""

    part_number: str = Field(..., description="Part number")
    trap_type: TrapType = Field(..., description="Applicable trap type")
    description: str = Field(..., description="Part description")
    current_quantity: int = Field(default=0, ge=0, description="Current stock")
    recommended_quantity: int = Field(
        default=0, ge=0, description="Recommended stock"
    )
    reorder_point: int = Field(default=0, ge=0, description="Reorder point")
    lead_time_days: int = Field(default=14, ge=0, description="Lead time (days)")
    unit_cost: float = Field(default=0.0, ge=0, description="Unit cost ($)")
    annual_usage: int = Field(default=0, ge=0, description="Annual usage")
    stockout_risk: str = Field(
        default="low", description="Stockout risk level"
    )


class SparesPlan(BaseModel):
    """Optimized spares inventory plan."""

    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    items: List[SparePartItem] = Field(
        default_factory=list, description="Spare part recommendations"
    )
    total_current_value: float = Field(
        default=0.0, description="Current inventory value ($)"
    )
    total_recommended_value: float = Field(
        default=0.0, description="Recommended inventory value ($)"
    )
    investment_required: float = Field(
        default=0.0, description="Additional investment required ($)"
    )
    service_level_percent: float = Field(
        default=95.0, description="Target service level (%)"
    )
    expected_stockouts_per_year: float = Field(
        default=0.0, description="Expected stockouts per year"
    )
    provenance_hash: str = Field(default="")


# =============================================================================
# Trap Maintenance Optimizer
# =============================================================================


class TrapMaintenanceOptimizer:
    """
    Optimizes steam trap maintenance activities.

    Objectives:
    - Minimize steam losses and associated costs
    - Optimize inspection and replacement schedules
    - Maintain adequate spares inventory
    - Balance maintenance costs with reliability

    Uses deterministic calculations:
    - Failure rate analysis (MTBF/MTTF)
    - Economic analysis (payback, ROI)
    - Inventory optimization (EOQ, safety stock)
    """

    # Default cost assumptions
    DEFAULT_STEAM_COST_PER_KLB = 10.0  # $/klb
    DEFAULT_LABOR_RATE = 75.0  # $/hr
    DEFAULT_OPERATING_HOURS = 8000  # hours/year

    # Failure rates by trap type (failures per 1000 operating hours)
    TYPICAL_FAILURE_RATES = {
        TrapType.THERMODYNAMIC: 0.15,
        TrapType.THERMOSTATIC: 0.10,
        TrapType.MECHANICAL: 0.08,
        TrapType.INVERTED_BUCKET: 0.05,
        TrapType.FLOAT_THERMOSTATIC: 0.07,
        TrapType.BIMETALLIC: 0.12,
        TrapType.UNKNOWN: 0.10,
    }

    # Steam loss rates by failure mode (lb/hr per psi differential)
    STEAM_LOSS_FACTORS = {
        TrapStatus.LEAKING: 0.1,  # lb/hr per psi
        TrapStatus.FAILED_OPEN: 0.5,  # lb/hr per psi (significant)
        TrapStatus.SUSPECT: 0.05,  # lb/hr per psi (potential)
    }

    def __init__(
        self,
        steam_cost_per_klb: float = DEFAULT_STEAM_COST_PER_KLB,
        labor_rate: float = DEFAULT_LABOR_RATE,
        operating_hours: int = DEFAULT_OPERATING_HOURS,
    ) -> None:
        """
        Initialize trap maintenance optimizer.

        Args:
            steam_cost_per_klb: Steam cost ($/klb)
            labor_rate: Labor rate ($/hr)
            operating_hours: Annual operating hours
        """
        self.steam_cost_per_klb = steam_cost_per_klb
        self.labor_rate = labor_rate
        self.operating_hours = operating_hours

        logger.info("TrapMaintenanceOptimizer initialized")

    def optimize_inspection_schedule(
        self,
        trap_fleet: TrapFleet,
        failure_predictions: List[FailurePrediction],
        horizon_days: int = 90,
        max_inspections_per_week: int = 50,
    ) -> InspectionSchedule:
        """
        Optimize trap inspection schedule.

        Creates a prioritized inspection schedule based on:
        - Failure predictions
        - Time since last inspection
        - Trap criticality
        - Location grouping for efficiency

        Args:
            trap_fleet: Current trap fleet data
            failure_predictions: Failure predictions by trap
            horizon_days: Schedule horizon (days)
            max_inspections_per_week: Maximum weekly inspections

        Returns:
            Optimized inspection schedule
        """
        start_time = time.perf_counter()

        # Create prediction lookup
        pred_lookup = {p.trap_id: p for p in failure_predictions}

        # Score each trap for inspection priority
        trap_scores: List[Tuple[TrapData, float, str]] = []

        now = datetime.now(timezone.utc)

        for trap in trap_fleet.traps:
            score, reason = self._calculate_inspection_priority(
                trap, pred_lookup.get(trap.trap_id), now
            )
            trap_scores.append((trap, score, reason))

        # Sort by score (higher = more urgent)
        trap_scores.sort(key=lambda x: x[1], reverse=True)

        # Create schedule
        tasks: List[InspectionTask] = []
        current_date = now + timedelta(days=1)  # Start tomorrow
        weekly_count = 0
        week_start = current_date

        for trap, score, reason in trap_scores:
            if (current_date - now).days > horizon_days:
                break

            # Check weekly limit
            if weekly_count >= max_inspections_per_week:
                # Move to next week
                days_to_monday = (7 - current_date.weekday()) % 7
                if days_to_monday == 0:
                    days_to_monday = 7
                current_date = current_date + timedelta(days=days_to_monday)
                week_start = current_date
                weekly_count = 0

            # Assign priority level (1-5)
            if score >= 80:
                priority = 1
            elif score >= 60:
                priority = 2
            elif score >= 40:
                priority = 3
            elif score >= 20:
                priority = 4
            else:
                priority = 5

            tasks.append(
                InspectionTask(
                    trap_id=trap.trap_id,
                    location=trap.location,
                    scheduled_date=current_date,
                    priority=priority,
                    estimated_duration_min=15,
                    reason=reason,
                    required_equipment=self._get_inspection_equipment(trap.trap_type),
                )
            )

            weekly_count += 1

            # Group nearby traps on same day
            if weekly_count % 10 == 0:
                current_date = current_date + timedelta(days=1)

        # Calculate metrics
        total_hours = len(tasks) * 15 / 60  # 15 min per inspection
        coverage = len(tasks) / len(trap_fleet.traps) * 100 if trap_fleet.traps else 0

        # Estimate failures prevented
        failures_prevented = sum(
            1 for trap, score, _ in trap_scores[:len(tasks)]
            if score >= 40 and pred_lookup.get(trap.trap_id) and
            pred_lookup[trap.trap_id].failure_probability_90d > 0.3
        )

        result = InspectionSchedule(
            schedule_horizon_days=horizon_days,
            tasks=tasks,
            total_inspection_hours=total_hours,
            coverage_percent=coverage,
            expected_failure_prevention=failures_prevented,
            optimization_score=self._calculate_schedule_score(tasks, trap_fleet),
        )

        result.provenance_hash = self._generate_provenance_hash(
            trap_fleet, failure_predictions, result
        )

        computation_time = (time.perf_counter() - start_time) * 1000
        logger.info(
            f"Inspection schedule: {len(tasks)} tasks over {horizon_days} days, "
            f"coverage={coverage:.1f}%, in {computation_time:.1f}ms"
        )

        return result

    def _calculate_inspection_priority(
        self,
        trap: TrapData,
        prediction: Optional[FailurePrediction],
        now: datetime,
    ) -> Tuple[float, str]:
        """Calculate inspection priority score for a trap."""
        score = 0.0
        reasons = []

        # Factor 1: Criticality (0-25 points)
        criticality_scores = {
            TrapCriticality.CRITICAL: 25,
            TrapCriticality.HIGH: 20,
            TrapCriticality.MEDIUM: 10,
            TrapCriticality.LOW: 5,
        }
        score += criticality_scores.get(trap.criticality, 10)

        # Factor 2: Current status (0-30 points)
        status_scores = {
            TrapStatus.SUSPECT: 30,
            TrapStatus.LEAKING: 25,
            TrapStatus.UNKNOWN: 20,
            TrapStatus.GOOD: 5,
        }
        status_score = status_scores.get(trap.status, 10)
        score += status_score
        if trap.status in (TrapStatus.SUSPECT, TrapStatus.LEAKING):
            reasons.append(f"Status: {trap.status.value}")

        # Factor 3: Time since inspection (0-25 points)
        if trap.last_inspection_date:
            days_since = (now - trap.last_inspection_date).days
            if days_since > 365:
                score += 25
                reasons.append(f"Overdue: {days_since} days since inspection")
            elif days_since > 180:
                score += 15
                reasons.append("Due for inspection")
            elif days_since > 90:
                score += 10
            else:
                score += 5
        else:
            score += 20
            reasons.append("No inspection history")

        # Factor 4: Failure prediction (0-20 points)
        if prediction:
            if prediction.failure_probability_90d > 0.5:
                score += 20
                reasons.append(f"High failure risk: {prediction.failure_probability_90d:.0%}")
            elif prediction.failure_probability_90d > 0.3:
                score += 15
            elif prediction.failure_probability_90d > 0.1:
                score += 10
            else:
                score += 5

        # Create reason string
        if not reasons:
            reasons.append("Routine inspection")
        reason = "; ".join(reasons)

        return score, reason

    def _get_inspection_equipment(self, trap_type: TrapType) -> List[str]:
        """Get required inspection equipment by trap type."""
        base_equipment = ["Ultrasonic detector", "Temperature gun"]

        if trap_type in (TrapType.THERMOSTATIC, TrapType.BIMETALLIC):
            return base_equipment + ["Thermometer probe"]
        elif trap_type == TrapType.THERMODYNAMIC:
            return base_equipment + ["Stethoscope"]
        else:
            return base_equipment

    def _calculate_schedule_score(
        self,
        tasks: List[InspectionTask],
        fleet: TrapFleet,
    ) -> float:
        """Calculate optimization score for schedule."""
        if not tasks or not fleet.traps:
            return 0.0

        # Score based on coverage and priority alignment
        coverage = len(tasks) / len(fleet.traps)
        priority_weight = sum(6 - t.priority for t in tasks) / (5 * len(tasks))

        return (coverage * 50 + priority_weight * 50)

    def prioritize_replacements(
        self,
        failed_traps: List[TrapData],
        budget: float,
        downtime_constraints: Optional[DowntimeConstraint] = None,
    ) -> ReplacementPriority:
        """
        Prioritize trap replacements within budget.

        Prioritizes based on:
        - Steam loss cost
        - Criticality
        - Payback period
        - Downtime constraints

        Args:
            failed_traps: List of failed/leaking traps
            budget: Available budget ($)
            downtime_constraints: Downtime constraints

        Returns:
            Prioritized replacement list
        """
        start_time = time.perf_counter()

        tasks: List[ReplacementTask] = []
        total_loss = 0.0
        total_annual_cost = 0.0

        for trap in failed_traps:
            # Calculate steam loss
            loss_lb_hr = self._calculate_steam_loss(trap)
            annual_loss_cost = (
                loss_lb_hr / 1000 * self.steam_cost_per_klb * self.operating_hours
            )

            # Calculate replacement cost
            labor_cost = trap.labor_hours * self.labor_rate
            total_replacement = trap.replacement_cost + labor_cost

            # Calculate payback
            payback_days = (
                total_replacement / (annual_loss_cost / 365)
                if annual_loss_cost > 0 else float('inf')
            )

            # Assign priority
            priority = self._calculate_replacement_priority(
                trap, loss_lb_hr, payback_days
            )

            tasks.append(
                ReplacementTask(
                    trap_id=trap.trap_id,
                    location=trap.location,
                    trap_type=trap.trap_type,
                    current_status=trap.status,
                    priority=priority,
                    steam_loss_lb_hr=round(loss_lb_hr, 1),
                    annual_loss_cost=round(annual_loss_cost, 0),
                    replacement_cost=round(total_replacement, 0),
                    payback_days=round(payback_days, 0),
                )
            )

            total_loss += loss_lb_hr
            total_annual_cost += annual_loss_cost

        # Sort by priority
        tasks.sort(key=lambda t: t.priority)

        # Apply budget constraint
        remaining_budget = budget
        within_budget = 0
        total_replacement_cost = 0.0
        savings_if_complete = 0.0

        for task in tasks:
            if task.replacement_cost <= remaining_budget:
                remaining_budget -= task.replacement_cost
                within_budget += 1
                total_replacement_cost += task.replacement_cost
                savings_if_complete += task.annual_loss_cost

        budget_utilization = (
            (budget - remaining_budget) / budget * 100
            if budget > 0 else 0
        )

        result = ReplacementPriority(
            tasks=tasks,
            total_tasks=len(tasks),
            within_budget_tasks=within_budget,
            total_steam_loss_lb_hr=total_loss,
            total_annual_loss_cost=total_annual_cost,
            total_replacement_cost=total_replacement_cost,
            budget_utilization_percent=budget_utilization,
            annual_savings_if_complete=savings_if_complete,
        )

        result.provenance_hash = hashlib.sha256(
            f"{len(tasks)}{budget}{total_loss}{result.timestamp}".encode()
        ).hexdigest()

        computation_time = (time.perf_counter() - start_time) * 1000
        logger.info(
            f"Replacement prioritization: {len(tasks)} traps, "
            f"{within_budget} within ${budget:.0f} budget, "
            f"savings=${savings_if_complete:.0f}/yr in {computation_time:.1f}ms"
        )

        return result

    def _calculate_steam_loss(self, trap: TrapData) -> float:
        """Calculate steam loss for a failed trap."""
        loss_factor = self.STEAM_LOSS_FACTORS.get(trap.status, 0.0)
        delta_p = trap.inlet_pressure_psig - trap.outlet_pressure_psig
        return loss_factor * delta_p

    def _calculate_replacement_priority(
        self,
        trap: TrapData,
        loss_lb_hr: float,
        payback_days: float,
    ) -> int:
        """Calculate replacement priority (1=highest, 5=lowest)."""
        score = 0

        # Criticality contribution
        crit_scores = {
            TrapCriticality.CRITICAL: 30,
            TrapCriticality.HIGH: 20,
            TrapCriticality.MEDIUM: 10,
            TrapCriticality.LOW: 5,
        }
        score += crit_scores.get(trap.criticality, 10)

        # Status contribution
        status_scores = {
            TrapStatus.FAILED_OPEN: 30,
            TrapStatus.LEAKING: 25,
            TrapStatus.FAILED_CLOSED: 20,  # Process impact
            TrapStatus.SUSPECT: 10,
        }
        score += status_scores.get(trap.status, 5)

        # Payback contribution
        if payback_days < 30:
            score += 25
        elif payback_days < 90:
            score += 20
        elif payback_days < 180:
            score += 15
        elif payback_days < 365:
            score += 10
        else:
            score += 5

        # Loss magnitude contribution
        if loss_lb_hr > 50:
            score += 15
        elif loss_lb_hr > 20:
            score += 10
        elif loss_lb_hr > 10:
            score += 5

        # Convert score to priority
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

    def compute_optimal_spares_inventory(
        self,
        failure_rates: Dict[TrapType, float],
        lead_times: Dict[str, int],
        current_inventory: Dict[str, int],
        trap_fleet_composition: Dict[TrapType, int],
        service_level: float = 0.95,
    ) -> SparesPlan:
        """
        Compute optimal spares inventory using EOQ and safety stock.

        Args:
            failure_rates: Failure rates by trap type (per 1000 hours)
            lead_times: Lead times by part number (days)
            current_inventory: Current inventory by part number
            trap_fleet_composition: Number of traps by type
            service_level: Target service level (0-1)

        Returns:
            Optimized spares plan
        """
        start_time = time.perf_counter()

        items: List[SparePartItem] = []
        total_current = 0.0
        total_recommended = 0.0

        # Map trap types to part numbers (simplified)
        type_to_part = {
            TrapType.THERMODYNAMIC: "TD-001",
            TrapType.THERMOSTATIC: "TS-001",
            TrapType.MECHANICAL: "MC-001",
            TrapType.INVERTED_BUCKET: "IB-001",
            TrapType.FLOAT_THERMOSTATIC: "FT-001",
            TrapType.BIMETALLIC: "BM-001",
        }

        # Default costs
        part_costs = {
            "TD-001": 150.0,
            "TS-001": 200.0,
            "MC-001": 350.0,
            "IB-001": 450.0,
            "FT-001": 400.0,
            "BM-001": 175.0,
        }

        for trap_type, count in trap_fleet_composition.items():
            if trap_type == TrapType.UNKNOWN or count == 0:
                continue

            part_number = type_to_part.get(trap_type)
            if not part_number:
                continue

            # Get failure rate
            failure_rate = failure_rates.get(
                trap_type, self.TYPICAL_FAILURE_RATES.get(trap_type, 0.10)
            )

            # Calculate annual demand
            annual_failures = count * failure_rate * self.operating_hours / 1000
            annual_demand = int(math.ceil(annual_failures))

            # Get lead time
            lead_time = lead_times.get(part_number, 14)

            # Calculate safety stock for service level
            # Using simplified approach: safety stock = z * sqrt(lead_time * variance)
            z_score = self._get_z_score(service_level)
            demand_variance = annual_demand * 0.2  # Assume 20% coefficient of variation
            daily_demand = annual_demand / 365
            safety_stock = int(math.ceil(
                z_score * math.sqrt(lead_time * demand_variance / 365)
            ))

            # Reorder point
            reorder_point = int(math.ceil(daily_demand * lead_time + safety_stock))

            # Recommended quantity (EOQ approximation + safety stock)
            unit_cost = part_costs.get(part_number, 300.0)
            ordering_cost = 50.0  # $ per order
            holding_cost_rate = 0.25  # 25% of unit cost per year

            eoq = math.sqrt(
                2 * annual_demand * ordering_cost /
                (unit_cost * holding_cost_rate)
            ) if annual_demand > 0 else 0

            recommended_qty = int(max(
                safety_stock + 1,
                reorder_point,
                eoq * 0.5  # Keep ~half EOQ on hand
            ))

            # Current inventory
            current_qty = current_inventory.get(part_number, 0)

            # Stockout risk
            if current_qty < safety_stock:
                risk = "high"
            elif current_qty < reorder_point:
                risk = "medium"
            else:
                risk = "low"

            items.append(
                SparePartItem(
                    part_number=part_number,
                    trap_type=trap_type,
                    description=f"{trap_type.value} trap replacement",
                    current_quantity=current_qty,
                    recommended_quantity=recommended_qty,
                    reorder_point=reorder_point,
                    lead_time_days=lead_time,
                    unit_cost=unit_cost,
                    annual_usage=annual_demand,
                    stockout_risk=risk,
                )
            )

            total_current += current_qty * unit_cost
            total_recommended += recommended_qty * unit_cost

        # Calculate expected stockouts
        stockouts = sum(
            1 for item in items if item.stockout_risk == "high"
        ) * 2  # Estimate 2 per high-risk item

        result = SparesPlan(
            items=items,
            total_current_value=total_current,
            total_recommended_value=total_recommended,
            investment_required=max(0, total_recommended - total_current),
            service_level_percent=service_level * 100,
            expected_stockouts_per_year=stockouts,
        )

        result.provenance_hash = hashlib.sha256(
            f"{len(items)}{total_recommended}{result.timestamp}".encode()
        ).hexdigest()

        computation_time = (time.perf_counter() - start_time) * 1000
        logger.info(
            f"Spares plan: {len(items)} items, "
            f"current=${total_current:.0f}, recommended=${total_recommended:.0f}, "
            f"investment=${result.investment_required:.0f} in {computation_time:.1f}ms"
        )

        return result

    def _get_z_score(self, service_level: float) -> float:
        """Get z-score for service level (normal distribution approximation)."""
        # Common service levels
        z_scores = {
            0.90: 1.28,
            0.95: 1.65,
            0.98: 2.05,
            0.99: 2.33,
        }
        # Find closest
        closest = min(z_scores.keys(), key=lambda x: abs(x - service_level))
        return z_scores[closest]

    def _generate_provenance_hash(
        self,
        fleet: TrapFleet,
        predictions: List[FailurePrediction],
        result: InspectionSchedule,
    ) -> str:
        """Generate SHA-256 provenance hash."""
        data = (
            f"{len(fleet.traps)}"
            f"{len(predictions)}"
            f"{len(result.tasks)}"
            f"{result.timestamp.isoformat()}"
        )
        return hashlib.sha256(data.encode()).hexdigest()

    def generate_failure_predictions(
        self,
        trap_fleet: TrapFleet,
    ) -> List[FailurePrediction]:
        """
        Generate failure predictions for trap fleet.

        Uses historical failure rates and operating conditions
        to predict failure probability.

        Args:
            trap_fleet: Current trap fleet data

        Returns:
            List of failure predictions
        """
        predictions = []
        now = datetime.now(timezone.utc)

        for trap in trap_fleet.traps:
            # Get base failure rate
            base_rate = self.TYPICAL_FAILURE_RATES.get(
                trap.trap_type, 0.10
            )

            # Adjust for age
            age_factor = 1.0
            if trap.install_date:
                age_years = (now - trap.install_date).days / 365
                if age_years > 5:
                    age_factor = 1.5 + (age_years - 5) * 0.1
                elif age_years > 3:
                    age_factor = 1.2

            # Adjust for operating conditions
            condition_factor = 1.0
            if trap.inlet_pressure_psig > 200:
                condition_factor *= 1.2
            if trap.condensate_load_lb_hr > 500:
                condition_factor *= 1.1

            # Adjust for failure history
            history_factor = 1.0 + trap.failure_count * 0.2

            # Effective failure rate
            effective_rate = base_rate * age_factor * condition_factor * history_factor

            # Calculate probabilities (exponential distribution)
            prob_30d = 1 - math.exp(-effective_rate * 30 * 24 / 1000)
            prob_90d = 1 - math.exp(-effective_rate * 90 * 24 / 1000)
            prob_365d = 1 - math.exp(-effective_rate * 365 * 24 / 1000)

            # MTTF
            mttf_hours = 1000 / effective_rate if effective_rate > 0 else float('inf')
            remaining_life_days = mttf_hours / 24

            # Risk factors
            risk_factors = []
            if age_factor > 1.2:
                risk_factors.append("Age > 5 years")
            if trap.failure_count > 2:
                risk_factors.append(f"Multiple failures ({trap.failure_count})")
            if condition_factor > 1.0:
                risk_factors.append("High operating stress")
            if trap.status == TrapStatus.SUSPECT:
                risk_factors.append("Currently suspect")

            predictions.append(
                FailurePrediction(
                    trap_id=trap.trap_id,
                    failure_probability_30d=round(min(1.0, prob_30d), 3),
                    failure_probability_90d=round(min(1.0, prob_90d), 3),
                    failure_probability_365d=round(min(1.0, prob_365d), 3),
                    expected_remaining_life_days=round(remaining_life_days, 0),
                    confidence=0.80 if trap.operating_hours > 1000 else 0.65,
                    risk_factors=risk_factors,
                )
            )

        return predictions
