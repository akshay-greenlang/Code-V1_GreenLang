"""
GL-013 PREDICTMAINT - Maintenance Scheduling Optimizer

This module implements optimal maintenance scheduling using
reliability-centered maintenance (RCM) principles and cost optimization.

Key Features:
- Cost optimization: min(C_p + C_f * P(f))
- Age replacement policy
- Block replacement policy
- Condition-based maintenance triggers
- Resource constraint handling
- Work order prioritization

Reference Standards:
- IEC 60300-3-11: Reliability centred maintenance
- SAE JA1011: RCM Criteria
- ISO 14224: Failure and maintenance data

Author: GL-CalculatorEngineer
Version: 1.0.0
License: Proprietary - GreenLang
"""

from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Any
from enum import Enum, auto
import math
from datetime import datetime, timedelta

from .constants import (
    WEIBULL_PARAMETERS,
    MAINTENANCE_COST_RATIOS,
    MaintenanceCostParameters,
    DEFAULT_DECIMAL_PRECISION,
)
from .provenance import (
    ProvenanceBuilder,
    ProvenanceRecord,
    CalculationType,
    store_provenance,
)


# =============================================================================
# ENUMS
# =============================================================================

class MaintenancePolicy(Enum):
    """Maintenance policy types."""
    AGE_REPLACEMENT = auto()      # Replace at fixed age
    BLOCK_REPLACEMENT = auto()    # Replace at fixed intervals
    CONDITION_BASED = auto()      # Replace based on condition
    FAILURE_BASED = auto()        # Run to failure (corrective only)
    PREDICTIVE = auto()           # Based on RUL prediction


class PriorityLevel(Enum):
    """Work order priority levels."""
    CRITICAL = 1    # Immediate action required
    HIGH = 2        # Within 24 hours
    MEDIUM = 3      # Within 1 week
    LOW = 4         # Within 1 month
    ROUTINE = 5     # Next scheduled opportunity


class MaintenanceType(Enum):
    """Types of maintenance activities."""
    PREVENTIVE = auto()
    PREDICTIVE = auto()
    CORRECTIVE = auto()
    CONDITION_MONITORING = auto()
    INSPECTION = auto()
    OVERHAUL = auto()


class ResourceType(Enum):
    """Types of maintenance resources."""
    LABOR = auto()
    SPARE_PARTS = auto()
    EQUIPMENT = auto()
    CONTRACTOR = auto()


# =============================================================================
# RESULT DATA CLASSES
# =============================================================================

@dataclass(frozen=True)
class OptimalIntervalResult:
    """
    Result of optimal maintenance interval calculation.

    Attributes:
        optimal_interval_hours: Optimal time between maintenance
        optimal_interval_days: Same in days
        expected_cost_per_hour: Expected cost rate at optimal
        availability: Expected availability at optimal interval
        reliability_at_interval: Reliability when maintenance is due
        cost_ratio: Preventive to corrective cost ratio
        policy_used: Maintenance policy
        provenance_hash: SHA-256 hash
    """
    optimal_interval_hours: Decimal
    optimal_interval_days: Decimal
    expected_cost_per_hour: Decimal
    availability: Decimal
    reliability_at_interval: Decimal
    cost_ratio: Decimal
    policy_used: str
    provenance_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "optimal_interval_hours": str(self.optimal_interval_hours),
            "optimal_interval_days": str(self.optimal_interval_days),
            "expected_cost_per_hour": str(self.expected_cost_per_hour),
            "availability": str(self.availability),
            "reliability_at_interval": str(self.reliability_at_interval),
            "cost_ratio": str(self.cost_ratio),
            "policy_used": self.policy_used,
            "provenance_hash": self.provenance_hash,
        }


@dataclass(frozen=True)
class CostOptimizationResult:
    """Result of maintenance cost optimization."""
    total_expected_cost: Decimal
    preventive_cost_component: Decimal
    failure_cost_component: Decimal
    downtime_cost_component: Decimal
    optimal_strategy: str
    savings_vs_run_to_failure: Decimal
    provenance_hash: str = ""


@dataclass(frozen=True)
class WorkOrderPriority:
    """Prioritized work order."""
    equipment_id: str
    description: str
    priority: PriorityLevel
    priority_score: Decimal
    due_date: Optional[str]
    estimated_duration_hours: Decimal
    estimated_cost: Decimal
    maintenance_type: MaintenanceType
    risk_score: Decimal
    provenance_hash: str = ""


@dataclass(frozen=True)
class ScheduleResult:
    """Result of maintenance scheduling."""
    schedule_id: str
    start_date: str
    end_date: str
    work_orders: Tuple[WorkOrderPriority, ...]
    total_labor_hours: Decimal
    total_cost: Decimal
    resource_utilization: Dict[str, Decimal]
    conflicts: Tuple[str, ...]
    provenance_hash: str = ""


@dataclass(frozen=True)
class CBMTrigger:
    """Condition-based maintenance trigger definition."""
    parameter: str
    threshold_value: Decimal
    current_value: Decimal
    trend_direction: str
    time_to_threshold_hours: Optional[Decimal]
    trigger_activated: bool
    confidence: Decimal
    recommended_action: str


# =============================================================================
# MAINTENANCE SCHEDULER
# =============================================================================

class MaintenanceScheduler:
    """
    Optimal maintenance scheduling with cost optimization.

    Implements reliability-centered maintenance (RCM) principles
    to determine optimal maintenance intervals and priorities.

    Key optimization model:
        C(t) = C_p / t + C_f * P(f|t)

    Where:
        C(t) = Expected cost per unit time
        C_p = Preventive maintenance cost
        C_f = Failure/corrective cost
        P(f|t) = Probability of failure by time t

    Reference: Barlow & Proschan (1965), Mathematical Theory of Reliability

    Example:
        >>> scheduler = MaintenanceScheduler()
        >>> result = scheduler.calculate_optimal_interval(
        ...     equipment_type="pump_centrifugal",
        ...     preventive_cost=Decimal("2500"),
        ...     failure_cost=Decimal("15000")
        ... )
        >>> print(f"Optimal interval: {result.optimal_interval_days} days")
    """

    def __init__(
        self,
        precision: int = DEFAULT_DECIMAL_PRECISION,
        store_provenance_records: bool = True
    ):
        """
        Initialize Maintenance Scheduler.

        Args:
            precision: Decimal precision for calculations
            store_provenance_records: Whether to store provenance
        """
        self._precision = precision
        self._store_provenance = store_provenance_records

    # =========================================================================
    # OPTIMAL INTERVAL CALCULATION
    # =========================================================================

    def calculate_optimal_interval(
        self,
        equipment_type: str,
        preventive_cost: Union[Decimal, float, str],
        failure_cost: Union[Decimal, float, str],
        preventive_duration_hours: Union[Decimal, float, str] = "4",
        corrective_duration_hours: Union[Decimal, float, str] = "24",
        downtime_cost_per_hour: Union[Decimal, float, str] = "0",
        beta: Optional[Union[Decimal, float, str]] = None,
        eta: Optional[Union[Decimal, float, str]] = None
    ) -> OptimalIntervalResult:
        """
        Calculate optimal preventive maintenance interval.

        Uses the age replacement model to find the interval that
        minimizes expected cost per unit time.

        For Weibull distribution with beta > 1 (wear-out):
            Optimal interval minimizes:
            C(t) = (C_p + C_d * t_p) / (t + t_p) +
                   (C_f + C_d * t_c) * F(t) / (t + t_p)

        Args:
            equipment_type: Equipment type for Weibull parameters
            preventive_cost: Cost of preventive maintenance ($)
            failure_cost: Cost of corrective maintenance ($)
            preventive_duration_hours: Duration of preventive maintenance
            corrective_duration_hours: Duration of corrective maintenance
            downtime_cost_per_hour: Cost per hour of downtime
            beta: Override Weibull shape parameter
            eta: Override Weibull scale parameter

        Returns:
            OptimalIntervalResult

        Reference:
            Barlow & Proschan (1965), Eq. 3.7

        Example:
            >>> scheduler = MaintenanceScheduler()
            >>> result = scheduler.calculate_optimal_interval(
            ...     equipment_type="pump_centrifugal",
            ...     preventive_cost="2500",
            ...     failure_cost="15000"
            ... )
        """
        builder = ProvenanceBuilder(CalculationType.MAINTENANCE_OPTIMIZATION)

        # Convert inputs
        C_p = self._to_decimal(preventive_cost)
        C_f = self._to_decimal(failure_cost)
        t_p = self._to_decimal(preventive_duration_hours)
        t_c = self._to_decimal(corrective_duration_hours)
        C_d = self._to_decimal(downtime_cost_per_hour)

        # Get Weibull parameters
        if beta and eta:
            beta_val = self._to_decimal(beta)
            eta_val = self._to_decimal(eta)
        else:
            if equipment_type not in WEIBULL_PARAMETERS:
                raise ValueError(f"Unknown equipment type: {equipment_type}")
            params = WEIBULL_PARAMETERS[equipment_type]
            beta_val = params.beta
            eta_val = params.eta

        builder.add_input("equipment_type", equipment_type)
        builder.add_input("preventive_cost", C_p)
        builder.add_input("failure_cost", C_f)
        builder.add_input("beta", beta_val)
        builder.add_input("eta", eta_val)

        # Step 1: Check if preventive maintenance is worthwhile
        # Condition: C_p < C_f and beta > 1
        cost_ratio = C_p / C_f

        builder.add_step(
            step_number=1,
            operation="calculate",
            description="Calculate cost ratio",
            inputs={"C_p": C_p, "C_f": C_f},
            output_name="cost_ratio",
            output_value=cost_ratio,
            formula="cost_ratio = C_p / C_f"
        )

        if beta_val <= Decimal("1"):
            # Constant or decreasing failure rate - run to failure is optimal
            builder.add_step(
                step_number=2,
                operation="check",
                description="Check failure rate pattern",
                inputs={"beta": beta_val},
                output_name="policy_recommendation",
                output_value="Run to failure (beta <= 1)",
                reference="Barlow & Proschan"
            )

            # MTTF for exponential
            mttf = eta_val

            builder.add_output("optimal_interval", mttf)
            builder.add_output("policy", "RUN_TO_FAILURE")

            provenance = builder.build()
            if self._store_provenance:
                store_provenance(provenance)

            return OptimalIntervalResult(
                optimal_interval_hours=mttf,
                optimal_interval_days=self._apply_precision(mttf / Decimal("24"), 1),
                expected_cost_per_hour=C_f / mttf,
                availability=Decimal("1") - t_c / mttf,
                reliability_at_interval=Decimal("0.368"),  # exp(-1)
                cost_ratio=cost_ratio,
                policy_used="Run to Failure (constant/decreasing failure rate)",
                provenance_hash=provenance.final_hash
            )

        # Step 2: Calculate optimal interval using numerical search
        # For Weibull, optimal t* satisfies:
        # beta * (t/eta)^beta * [C_f - C_p] = (C_p + C_f * F(t)) * (t/eta)

        # Use bisection search
        t_low = eta_val * Decimal("0.01")
        t_high = eta_val * Decimal("3")

        def expected_cost_rate(t: Decimal) -> Decimal:
            """Calculate expected cost per unit time."""
            # F(t) = 1 - exp(-(t/eta)^beta)
            F_t = Decimal("1") - self._exp(-self._power(t / eta_val, beta_val))
            R_t = Decimal("1") - F_t

            # Expected cycle length
            # E[cycle] = integral from 0 to t of R(x)dx + F(t)*t_c + R(t)*t_p
            # Approximate: E[cycle] ~ t * R(t) + MTTF * F(t) + t_p*R(t) + t_c*F(t)
            # Simplified: t + t_p for preventive, MTTF_truncated + t_c for failure

            # Total cost per cycle
            C_total = C_p * R_t + C_f * F_t + C_d * (t_p * R_t + t_c * F_t)

            # Expected cycle time (simplified)
            cycle_time = t + t_p * R_t + t_c * F_t

            if cycle_time > Decimal("0"):
                return C_total / cycle_time
            return Decimal("Infinity")

        # Find minimum using golden section search
        golden_ratio = Decimal("1.618033988749895")
        tolerance = eta_val * Decimal("0.001")

        a, b = t_low, t_high
        c = b - (b - a) / golden_ratio
        d = a + (b - a) / golden_ratio

        iterations = 0
        max_iterations = 100

        while abs(b - a) > tolerance and iterations < max_iterations:
            if expected_cost_rate(c) < expected_cost_rate(d):
                b = d
            else:
                a = c

            c = b - (b - a) / golden_ratio
            d = a + (b - a) / golden_ratio
            iterations += 1

        t_optimal = (a + b) / Decimal("2")

        builder.add_step(
            step_number=2,
            operation="optimize",
            description="Find optimal interval using golden section search",
            inputs={"t_low": t_low, "t_high": t_high, "iterations": iterations},
            output_name="t_optimal",
            output_value=t_optimal,
            formula="Minimize C(t) = E[cost]/E[cycle_time]",
            reference="Golden section optimization"
        )

        # Step 3: Calculate metrics at optimal interval
        F_optimal = Decimal("1") - self._exp(-self._power(t_optimal / eta_val, beta_val))
        R_optimal = Decimal("1") - F_optimal
        cost_rate_optimal = expected_cost_rate(t_optimal)

        # Availability
        cycle_time = t_optimal + t_p * R_optimal + t_c * F_optimal
        downtime = t_p * R_optimal + t_c * F_optimal
        availability = (cycle_time - downtime) / cycle_time if cycle_time > 0 else Decimal("0")

        builder.add_step(
            step_number=3,
            operation="calculate",
            description="Calculate metrics at optimal interval",
            inputs={"t_optimal": t_optimal},
            output_name="metrics",
            output_value={
                "reliability": R_optimal,
                "availability": availability,
                "cost_rate": cost_rate_optimal
            }
        )

        # Finalize
        t_optimal_days = t_optimal / Decimal("24")

        builder.add_output("optimal_interval_hours", t_optimal)
        builder.add_output("optimal_interval_days", t_optimal_days)
        builder.add_output("expected_cost_per_hour", cost_rate_optimal)

        provenance = builder.build()
        if self._store_provenance:
            store_provenance(provenance)

        return OptimalIntervalResult(
            optimal_interval_hours=self._apply_precision(t_optimal, 0),
            optimal_interval_days=self._apply_precision(t_optimal_days, 1),
            expected_cost_per_hour=self._apply_precision(cost_rate_optimal, 4),
            availability=self._apply_precision(availability, 4),
            reliability_at_interval=self._apply_precision(R_optimal, 4),
            cost_ratio=self._apply_precision(cost_ratio, 4),
            policy_used="Age Replacement (Weibull wear-out)",
            provenance_hash=provenance.final_hash
        )

    # =========================================================================
    # COST OPTIMIZATION
    # =========================================================================

    def optimize_maintenance_costs(
        self,
        equipment_id: str,
        equipment_type: str,
        annual_operating_hours: Union[Decimal, float, int, str] = "8760",
        planning_horizon_years: Union[Decimal, float, int, str] = "5",
        preventive_cost: Optional[Union[Decimal, float, str]] = None,
        failure_cost: Optional[Union[Decimal, float, str]] = None,
        inspection_cost: Optional[Union[Decimal, float, str]] = None,
        downtime_cost_per_hour: Optional[Union[Decimal, float, str]] = None
    ) -> CostOptimizationResult:
        """
        Optimize total maintenance costs over planning horizon.

        Compares different maintenance strategies and calculates
        expected costs for each.

        Strategies compared:
        1. Run to failure (corrective only)
        2. Time-based preventive at optimal interval
        3. Condition-based with inspections

        Args:
            equipment_id: Equipment identifier
            equipment_type: Equipment type for parameters
            annual_operating_hours: Hours per year of operation
            planning_horizon_years: Planning horizon
            preventive_cost: Override preventive cost
            failure_cost: Override failure cost
            inspection_cost: Override inspection cost
            downtime_cost_per_hour: Override downtime cost

        Returns:
            CostOptimizationResult

        Example:
            >>> scheduler = MaintenanceScheduler()
            >>> result = scheduler.optimize_maintenance_costs(
            ...     equipment_id="PUMP-001",
            ...     equipment_type="pump_centrifugal",
            ...     planning_horizon_years=10
            ... )
        """
        builder = ProvenanceBuilder(CalculationType.MAINTENANCE_OPTIMIZATION)

        # Get cost parameters
        if equipment_type in MAINTENANCE_COST_RATIOS:
            cost_params = MAINTENANCE_COST_RATIOS[equipment_type]
            C_p = self._to_decimal(preventive_cost) if preventive_cost else cost_params.preventive_cost
            C_f = self._to_decimal(failure_cost) if failure_cost else cost_params.corrective_cost
            C_i = self._to_decimal(inspection_cost) if inspection_cost else cost_params.inspection_cost
            C_d = self._to_decimal(downtime_cost_per_hour) if downtime_cost_per_hour else cost_params.downtime_cost_per_hour
        else:
            C_p = self._to_decimal(preventive_cost or "1000")
            C_f = self._to_decimal(failure_cost or "5000")
            C_i = self._to_decimal(inspection_cost or "200")
            C_d = self._to_decimal(downtime_cost_per_hour or "1000")

        annual_hours = self._to_decimal(annual_operating_hours)
        horizon = self._to_decimal(planning_horizon_years)
        total_hours = annual_hours * horizon

        builder.add_input("equipment_id", equipment_id)
        builder.add_input("equipment_type", equipment_type)
        builder.add_input("planning_horizon_years", horizon)
        builder.add_input("preventive_cost", C_p)
        builder.add_input("failure_cost", C_f)

        # Get reliability parameters
        if equipment_type in WEIBULL_PARAMETERS:
            params = WEIBULL_PARAMETERS[equipment_type]
            beta = params.beta
            eta = params.eta
        else:
            beta = Decimal("2.0")
            eta = Decimal("50000")

        # Strategy 1: Run to failure
        # Expected failures = total_hours / MTTF
        mttf = eta * self._gamma_function(Decimal("1") + Decimal("1") / beta)
        expected_failures_rtf = total_hours / mttf
        cost_rtf = expected_failures_rtf * (C_f + C_d * Decimal("24"))  # Assume 24h downtime

        builder.add_step(
            step_number=1,
            operation="calculate",
            description="Calculate run-to-failure cost",
            inputs={"mttf": mttf, "total_hours": total_hours, "C_f": C_f},
            output_name="cost_rtf",
            output_value=cost_rtf,
            formula="Cost_RTF = (total_hours / MTTF) * C_f"
        )

        # Strategy 2: Optimal preventive maintenance
        optimal_result = self.calculate_optimal_interval(
            equipment_type=equipment_type,
            preventive_cost=C_p,
            failure_cost=C_f,
            downtime_cost_per_hour=C_d
        )
        t_opt = optimal_result.optimal_interval_hours
        cost_rate = optimal_result.expected_cost_per_hour
        cost_pm = cost_rate * total_hours

        builder.add_step(
            step_number=2,
            operation="calculate",
            description="Calculate preventive maintenance cost",
            inputs={"t_optimal": t_opt, "cost_rate": cost_rate},
            output_name="cost_pm",
            output_value=cost_pm,
            formula="Cost_PM = cost_rate * total_hours"
        )

        # Strategy 3: Condition-based (estimate: 70-80% of optimal PM cost)
        cbm_efficiency = Decimal("0.75")
        cost_cbm = cost_pm * cbm_efficiency + (total_hours / (Decimal("720"))) * C_i  # Monthly inspections

        builder.add_step(
            step_number=3,
            operation="calculate",
            description="Calculate condition-based maintenance cost",
            inputs={"cost_pm": cost_pm, "efficiency": cbm_efficiency},
            output_name="cost_cbm",
            output_value=cost_cbm
        )

        # Determine optimal strategy
        costs = {
            "Run to Failure": cost_rtf,
            "Preventive Maintenance": cost_pm,
            "Condition-Based Maintenance": cost_cbm
        }
        optimal_strategy = min(costs, key=costs.get)
        optimal_cost = costs[optimal_strategy]

        # Calculate savings vs run to failure
        savings = cost_rtf - optimal_cost
        savings_pct = (savings / cost_rtf) * Decimal("100") if cost_rtf > 0 else Decimal("0")

        # Break down optimal cost
        if optimal_strategy == "Preventive Maintenance":
            num_pm = total_hours / t_opt
            F_opt = Decimal("1") - self._exp(-self._power(t_opt / eta, beta))
            pm_component = num_pm * C_p * (Decimal("1") - F_opt)
            failure_component = num_pm * C_f * F_opt
            downtime_component = optimal_cost - pm_component - failure_component
        else:
            pm_component = optimal_cost * Decimal("0.6")
            failure_component = optimal_cost * Decimal("0.3")
            downtime_component = optimal_cost * Decimal("0.1")

        builder.add_output("optimal_strategy", optimal_strategy)
        builder.add_output("total_expected_cost", optimal_cost)
        builder.add_output("savings_vs_rtf", savings)

        provenance = builder.build()
        if self._store_provenance:
            store_provenance(provenance)

        return CostOptimizationResult(
            total_expected_cost=self._apply_precision(optimal_cost, 2),
            preventive_cost_component=self._apply_precision(pm_component, 2),
            failure_cost_component=self._apply_precision(failure_component, 2),
            downtime_cost_component=self._apply_precision(downtime_component, 2),
            optimal_strategy=optimal_strategy,
            savings_vs_run_to_failure=self._apply_precision(savings, 2),
            provenance_hash=provenance.final_hash
        )

    # =========================================================================
    # WORK ORDER PRIORITIZATION
    # =========================================================================

    def prioritize_work_orders(
        self,
        work_orders: List[Dict[str, Any]]
    ) -> List[WorkOrderPriority]:
        """
        Prioritize maintenance work orders.

        Uses multi-criteria scoring based on:
        - Equipment criticality
        - Failure probability / RUL
        - Safety impact
        - Production impact
        - Cost of delay

        Priority score = w1*criticality + w2*risk + w3*cost_impact

        Args:
            work_orders: List of work order dicts with fields:
                - equipment_id: Equipment identifier
                - description: Work description
                - equipment_criticality: 1-5 scale
                - failure_probability: 0-1
                - safety_impact: 1-5 scale
                - production_impact: 1-5 scale
                - estimated_duration_hours: Hours to complete
                - estimated_cost: Dollar cost
                - maintenance_type: MaintenanceType
                - due_date: Optional due date string

        Returns:
            List of WorkOrderPriority sorted by priority

        Example:
            >>> scheduler = MaintenanceScheduler()
            >>> orders = [
            ...     {"equipment_id": "PUMP-001", "description": "Replace seal",
            ...      "equipment_criticality": 4, "failure_probability": 0.8,
            ...      "safety_impact": 2, "production_impact": 4,
            ...      "estimated_duration_hours": 4, "estimated_cost": 500,
            ...      "maintenance_type": "PREVENTIVE"},
            ... ]
            >>> prioritized = scheduler.prioritize_work_orders(orders)
        """
        builder = ProvenanceBuilder(CalculationType.MAINTENANCE_OPTIMIZATION)

        builder.add_input("num_work_orders", len(work_orders))

        # Weights for priority calculation
        w_criticality = Decimal("0.25")
        w_risk = Decimal("0.30")
        w_safety = Decimal("0.25")
        w_production = Decimal("0.20")

        prioritized = []

        for i, wo in enumerate(work_orders):
            # Extract and convert values
            eq_id = wo.get("equipment_id", f"UNKNOWN-{i}")
            description = wo.get("description", "Unspecified maintenance")
            criticality = self._to_decimal(wo.get("equipment_criticality", "3"))
            failure_prob = self._to_decimal(wo.get("failure_probability", "0.5"))
            safety = self._to_decimal(wo.get("safety_impact", "3"))
            production = self._to_decimal(wo.get("production_impact", "3"))
            duration = self._to_decimal(wo.get("estimated_duration_hours", "4"))
            cost = self._to_decimal(wo.get("estimated_cost", "1000"))
            maint_type_str = wo.get("maintenance_type", "PREVENTIVE")
            due_date = wo.get("due_date")

            # Convert maintenance type
            try:
                maint_type = MaintenanceType[maint_type_str]
            except KeyError:
                maint_type = MaintenanceType.PREVENTIVE

            # Normalize scores to 0-1
            criticality_norm = criticality / Decimal("5")
            risk_score = failure_prob  # Already 0-1
            safety_norm = safety / Decimal("5")
            production_norm = production / Decimal("5")

            # Calculate priority score (higher = more urgent)
            priority_score = (
                w_criticality * criticality_norm +
                w_risk * risk_score +
                w_safety * safety_norm +
                w_production * production_norm
            )

            # Determine priority level
            if priority_score >= Decimal("0.8") or safety >= Decimal("5"):
                priority = PriorityLevel.CRITICAL
            elif priority_score >= Decimal("0.6"):
                priority = PriorityLevel.HIGH
            elif priority_score >= Decimal("0.4"):
                priority = PriorityLevel.MEDIUM
            elif priority_score >= Decimal("0.2"):
                priority = PriorityLevel.LOW
            else:
                priority = PriorityLevel.ROUTINE

            prioritized.append(WorkOrderPriority(
                equipment_id=eq_id,
                description=description,
                priority=priority,
                priority_score=self._apply_precision(priority_score, 4),
                due_date=due_date,
                estimated_duration_hours=duration,
                estimated_cost=cost,
                maintenance_type=maint_type,
                risk_score=self._apply_precision(risk_score, 4),
                provenance_hash=""
            ))

        # Sort by priority (lower enum value = higher priority)
        prioritized.sort(key=lambda x: (x.priority.value, -float(x.priority_score)))

        builder.add_step(
            step_number=1,
            operation="prioritize",
            description="Calculate priority scores and sort",
            inputs={"num_orders": len(work_orders)},
            output_name="prioritized_orders",
            output_value=[wo.equipment_id for wo in prioritized[:5]]
        )

        builder.add_output("num_critical", sum(1 for wo in prioritized if wo.priority == PriorityLevel.CRITICAL))
        builder.add_output("num_high", sum(1 for wo in prioritized if wo.priority == PriorityLevel.HIGH))

        provenance = builder.build()
        if self._store_provenance:
            store_provenance(provenance)

        return prioritized

    # =========================================================================
    # CONDITION-BASED MAINTENANCE TRIGGERS
    # =========================================================================

    def evaluate_cbm_triggers(
        self,
        equipment_id: str,
        condition_parameters: List[Dict[str, Any]]
    ) -> List[CBMTrigger]:
        """
        Evaluate condition-based maintenance triggers.

        Checks current condition parameters against thresholds
        and predicts time to threshold.

        Args:
            equipment_id: Equipment identifier
            condition_parameters: List of parameter dicts:
                - name: Parameter name
                - current_value: Current reading
                - threshold_value: Trigger threshold
                - unit: Unit of measurement
                - trend_rate: Rate of change (per day)
                - baseline_value: Normal operating value

        Returns:
            List of CBMTrigger results

        Example:
            >>> scheduler = MaintenanceScheduler()
            >>> params = [
            ...     {"name": "vibration_mm_s", "current_value": 4.5,
            ...      "threshold_value": 7.1, "trend_rate": 0.1},
            ...     {"name": "temperature_c", "current_value": 75,
            ...      "threshold_value": 85, "trend_rate": 0.5},
            ... ]
            >>> triggers = scheduler.evaluate_cbm_triggers("MOTOR-001", params)
        """
        builder = ProvenanceBuilder(CalculationType.MAINTENANCE_OPTIMIZATION)

        builder.add_input("equipment_id", equipment_id)
        builder.add_input("num_parameters", len(condition_parameters))

        triggers = []

        for param in condition_parameters:
            name = param.get("name", "unknown")
            current = self._to_decimal(param.get("current_value", "0"))
            threshold = self._to_decimal(param.get("threshold_value", "100"))
            trend_rate = self._to_decimal(param.get("trend_rate", "0"))
            baseline = self._to_decimal(param.get("baseline_value", current))

            # Determine trend direction
            if trend_rate > Decimal("0.01"):
                trend_direction = "increasing"
            elif trend_rate < Decimal("-0.01"):
                trend_direction = "decreasing"
            else:
                trend_direction = "stable"

            # Check if trigger is activated
            trigger_activated = current >= threshold

            # Calculate time to threshold
            if trend_rate > Decimal("0") and current < threshold:
                time_to_threshold = (threshold - current) / trend_rate
            elif trend_rate <= Decimal("0") or current >= threshold:
                time_to_threshold = None
            else:
                time_to_threshold = None

            # Calculate confidence based on how far from baseline
            if baseline > Decimal("0"):
                deviation = abs(current - baseline) / baseline
                confidence = min(Decimal("1"), deviation * Decimal("2"))
            else:
                confidence = Decimal("0.5")

            # Recommend action
            if trigger_activated:
                recommended_action = "Immediate maintenance required"
            elif time_to_threshold and time_to_threshold < Decimal("168"):  # 1 week
                recommended_action = f"Schedule maintenance within {int(time_to_threshold)} hours"
            elif time_to_threshold and time_to_threshold < Decimal("720"):  # 1 month
                recommended_action = "Add to next maintenance window"
            else:
                recommended_action = "Continue monitoring"

            triggers.append(CBMTrigger(
                parameter=name,
                threshold_value=threshold,
                current_value=current,
                trend_direction=trend_direction,
                time_to_threshold_hours=self._apply_precision(time_to_threshold, 0) if time_to_threshold else None,
                trigger_activated=trigger_activated,
                confidence=self._apply_precision(confidence, 2),
                recommended_action=recommended_action
            ))

        builder.add_step(
            step_number=1,
            operation="evaluate",
            description="Evaluate CBM trigger conditions",
            inputs={"num_parameters": len(condition_parameters)},
            output_name="triggered",
            output_value=[t.parameter for t in triggers if t.trigger_activated]
        )

        provenance = builder.build()
        if self._store_provenance:
            store_provenance(provenance)

        return triggers

    # =========================================================================
    # RESOURCE SCHEDULING
    # =========================================================================

    def schedule_maintenance(
        self,
        work_orders: List[WorkOrderPriority],
        available_labor_hours: Union[Decimal, float, str],
        available_budget: Union[Decimal, float, str],
        schedule_window_days: int = 30
    ) -> ScheduleResult:
        """
        Schedule maintenance work orders within constraints.

        Uses priority-based scheduling with resource constraints.

        Args:
            work_orders: Prioritized work orders
            available_labor_hours: Available labor hours
            available_budget: Available budget
            schedule_window_days: Planning window in days

        Returns:
            ScheduleResult with scheduled work orders

        Example:
            >>> scheduler = MaintenanceScheduler()
            >>> prioritized = scheduler.prioritize_work_orders([...])
            >>> schedule = scheduler.schedule_maintenance(
            ...     work_orders=prioritized,
            ...     available_labor_hours=160,
            ...     available_budget=50000
            ... )
        """
        builder = ProvenanceBuilder(CalculationType.MAINTENANCE_OPTIMIZATION)

        labor_available = self._to_decimal(available_labor_hours)
        budget_available = self._to_decimal(available_budget)

        builder.add_input("num_work_orders", len(work_orders))
        builder.add_input("available_labor_hours", labor_available)
        builder.add_input("available_budget", budget_available)
        builder.add_input("schedule_window_days", schedule_window_days)

        # Schedule work orders by priority until constraints are hit
        scheduled = []
        labor_used = Decimal("0")
        cost_used = Decimal("0")
        conflicts = []

        for wo in work_orders:
            # Check constraints
            if labor_used + wo.estimated_duration_hours > labor_available:
                conflicts.append(f"{wo.equipment_id}: Insufficient labor")
                continue

            if cost_used + wo.estimated_cost > budget_available:
                conflicts.append(f"{wo.equipment_id}: Insufficient budget")
                continue

            # Schedule the work order
            scheduled.append(wo)
            labor_used += wo.estimated_duration_hours
            cost_used += wo.estimated_cost

        # Calculate resource utilization
        labor_util = (labor_used / labor_available) * Decimal("100") if labor_available > 0 else Decimal("0")
        budget_util = (cost_used / budget_available) * Decimal("100") if budget_available > 0 else Decimal("0")

        # Generate schedule dates
        start_date = datetime.now().strftime("%Y-%m-%d")
        end_date = (datetime.now() + timedelta(days=schedule_window_days)).strftime("%Y-%m-%d")
        schedule_id = f"SCH-{datetime.now().strftime('%Y%m%d%H%M%S')}"

        builder.add_step(
            step_number=1,
            operation="schedule",
            description="Schedule work orders within constraints",
            inputs={
                "num_orders": len(work_orders),
                "labor_available": labor_available,
                "budget_available": budget_available
            },
            output_name="scheduled_count",
            output_value=len(scheduled)
        )

        builder.add_output("labor_utilization", labor_util)
        builder.add_output("budget_utilization", budget_util)
        builder.add_output("conflicts", len(conflicts))

        provenance = builder.build()
        if self._store_provenance:
            store_provenance(provenance)

        return ScheduleResult(
            schedule_id=schedule_id,
            start_date=start_date,
            end_date=end_date,
            work_orders=tuple(scheduled),
            total_labor_hours=self._apply_precision(labor_used, 1),
            total_cost=self._apply_precision(cost_used, 2),
            resource_utilization={
                "labor_percent": self._apply_precision(labor_util, 1),
                "budget_percent": self._apply_precision(budget_util, 1)
            },
            conflicts=tuple(conflicts),
            provenance_hash=provenance.final_hash
        )

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    def _to_decimal(self, value: Union[Decimal, float, int, str]) -> Decimal:
        """Convert value to Decimal."""
        if isinstance(value, Decimal):
            return value
        try:
            return Decimal(str(value))
        except InvalidOperation as e:
            raise ValueError(f"Cannot convert {value} to Decimal: {e}")

    def _apply_precision(
        self,
        value: Decimal,
        precision: Optional[int] = None
    ) -> Decimal:
        """Apply precision rounding."""
        prec = precision if precision is not None else self._precision
        if prec == 0:
            return value.quantize(Decimal("1"), rounding=ROUND_HALF_UP)
        quantize_str = "0." + "0" * prec
        return value.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP)

    def _exp(self, x: Decimal) -> Decimal:
        """Calculate e^x."""
        if x == Decimal("0"):
            return Decimal("1")
        if x < Decimal("-700"):
            return Decimal("0")
        return Decimal(str(math.exp(float(x))))

    def _power(self, base: Decimal, exponent: Decimal) -> Decimal:
        """Calculate base^exponent."""
        if base == Decimal("0"):
            return Decimal("0") if exponent > Decimal("0") else Decimal("1")
        if exponent == Decimal("0"):
            return Decimal("1")
        return Decimal(str(math.pow(float(base), float(exponent))))

    def _gamma_function(self, x: Decimal) -> Decimal:
        """Calculate Gamma function."""
        return Decimal(str(math.gamma(float(x))))

    def get_supported_equipment_types(self) -> List[str]:
        """Get list of equipment types with cost parameters."""
        return list(MAINTENANCE_COST_RATIOS.keys())


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "MaintenancePolicy",
    "PriorityLevel",
    "MaintenanceType",
    "ResourceType",

    # Data classes
    "OptimalIntervalResult",
    "CostOptimizationResult",
    "WorkOrderPriority",
    "ScheduleResult",
    "CBMTrigger",

    # Main class
    "MaintenanceScheduler",
]
