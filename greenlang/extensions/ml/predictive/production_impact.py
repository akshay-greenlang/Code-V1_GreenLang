# -*- coding: utf-8 -*-
"""
Production Impact Modeling Module (TASK-110)

Zero-Hallucination Production Impact Analysis for Process Heat Systems

This module implements deterministic production impact modeling including:
- Downtime cost estimation
- Throughput impact of failures
- Energy cost impact calculations
- Quality impact modeling
- Financial impact aggregation
- Monte Carlo simulation for uncertainty

Integrates with Process Heat agents (GL-003 through GL-018) for comprehensive
operational and financial impact assessment.

References:
    - ISO 55000: Asset Management
    - API 689: Collection and Exchange of Reliability Data
    - IEC 60300-3-11: Reliability Analysis
    - Monte Carlo Methods (Metropolis & Ulam, 1949)

Author: GreenLang Engineering Team
License: MIT
"""

from dataclasses import dataclass, field
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from enum import Enum
from datetime import datetime, timedelta
import hashlib
import logging
import math
import random

from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)


class FailureMode(str, Enum):
    """Equipment failure modes."""
    MECHANICAL = "mechanical"
    ELECTRICAL = "electrical"
    INSTRUMENTATION = "instrumentation"
    PROCESS = "process"
    STRUCTURAL = "structural"
    CONTROL_SYSTEM = "control_system"
    HUMAN_ERROR = "human_error"
    EXTERNAL = "external"


class ImpactSeverity(str, Enum):
    """Impact severity classification."""
    NEGLIGIBLE = "negligible"  # No production impact
    MINOR = "minor"            # <5% production loss
    MODERATE = "moderate"      # 5-25% production loss
    MAJOR = "major"            # 25-75% production loss
    CATASTROPHIC = "catastrophic"  # >75% production loss


class ProductionState(str, Enum):
    """Production system states."""
    NORMAL = "normal"
    DEGRADED = "degraded"
    PARTIAL_OUTAGE = "partial_outage"
    FULL_OUTAGE = "full_outage"
    MAINTENANCE = "maintenance"
    STARTUP = "startup"
    SHUTDOWN = "shutdown"


class CostCategory(str, Enum):
    """Cost impact categories."""
    LOST_PRODUCTION = "lost_production"
    ENERGY_WASTE = "energy_waste"
    QUALITY_LOSS = "quality_loss"
    REPAIR_COST = "repair_cost"
    LABOR_OVERTIME = "labor_overtime"
    EXPEDITED_PARTS = "expedited_parts"
    ENVIRONMENTAL = "environmental"
    SAFETY = "safety"
    CUSTOMER_PENALTY = "customer_penalty"


# ============================================================================
# Pydantic Models
# ============================================================================

class EquipmentProfile(BaseModel):
    """Equipment profile for impact modeling."""

    equipment_id: str = Field(..., description="Equipment identifier")
    equipment_name: str = Field(..., description="Equipment name/description")
    equipment_type: str = Field(..., description="Equipment type")
    capacity_units: str = Field(default="ton/hr", description="Capacity units")
    rated_capacity: float = Field(..., gt=0, description="Rated capacity")
    current_efficiency: float = Field(
        default=0.85,
        ge=0.5,
        le=1.0,
        description="Current operating efficiency"
    )

    # Reliability data
    mtbf_hours: float = Field(default=8760, gt=0, description="Mean Time Between Failures")
    mttr_hours: float = Field(default=4, gt=0, description="Mean Time To Repair")
    failure_rate_per_year: Optional[float] = Field(None, ge=0, description="Annual failure rate")

    # Cost parameters
    hourly_operating_cost: float = Field(default=0, ge=0, description="Hourly operating cost")
    hourly_labor_cost: float = Field(default=75, ge=0, description="Hourly labor rate")
    energy_consumption_kwh: float = Field(default=0, ge=0, description="Energy consumption kWh")

    @validator('failure_rate_per_year', pre=True, always=True)
    def calculate_failure_rate(cls, v, values):
        """Calculate failure rate from MTBF if not provided."""
        if v is not None:
            return v
        mtbf = values.get('mtbf_hours', 8760)
        return 8760 / mtbf  # Failures per year


class ProductionParameters(BaseModel):
    """Production system parameters."""

    production_value_per_unit: float = Field(
        ...,
        gt=0,
        description="Value per unit of production ($)"
    )
    hourly_production_rate: float = Field(
        ...,
        gt=0,
        description="Production rate per hour"
    )
    production_unit: str = Field(default="ton", description="Production unit")

    # Operating parameters
    operating_hours_per_day: float = Field(default=24, ge=0, le=24)
    operating_days_per_year: int = Field(default=350, ge=0, le=365)
    current_utilization: float = Field(default=0.85, ge=0, le=1)

    # Quality parameters
    quality_yield: float = Field(default=0.98, ge=0.5, le=1.0, description="Quality yield rate")
    rework_cost_per_unit: float = Field(default=0, ge=0, description="Rework cost per unit")
    scrap_cost_per_unit: float = Field(default=0, ge=0, description="Scrap cost per unit")

    # Energy parameters
    energy_cost_per_kwh: float = Field(default=0.08, ge=0, description="Energy cost $/kWh")
    natural_gas_cost_per_mmbtu: float = Field(default=4.0, ge=0, description="NG cost $/MMBtu")

    # Penalty parameters
    customer_penalty_per_day: float = Field(default=0, ge=0, description="Daily customer penalty")
    environmental_fine_risk: float = Field(default=0, ge=0, description="Environmental fine risk")


class FailureScenario(BaseModel):
    """Failure scenario definition."""

    scenario_id: str = Field(..., description="Scenario identifier")
    scenario_name: str = Field(..., description="Scenario description")
    failure_mode: FailureMode = Field(..., description="Type of failure")

    # Probability
    probability: float = Field(..., ge=0, le=1, description="Annual probability of occurrence")

    # Duration
    duration_hours_min: float = Field(..., ge=0, description="Minimum downtime hours")
    duration_hours_max: float = Field(..., ge=0, description="Maximum downtime hours")
    duration_hours_mode: Optional[float] = Field(None, ge=0, description="Most likely duration")

    # Production impact
    capacity_loss_pct: float = Field(
        default=100,
        ge=0,
        le=100,
        description="Capacity loss percentage"
    )
    quality_impact_pct: float = Field(
        default=0,
        ge=0,
        le=100,
        description="Quality degradation percentage"
    )

    # Repair costs
    repair_cost_min: float = Field(default=0, ge=0, description="Minimum repair cost")
    repair_cost_max: float = Field(default=0, ge=0, description="Maximum repair cost")

    # Affected equipment
    affected_equipment: List[str] = Field(
        default_factory=list,
        description="List of affected equipment IDs"
    )


class ProductionImpactInput(BaseModel):
    """Input for production impact analysis."""

    analysis_name: str = Field(..., description="Analysis name/identifier")

    # Equipment
    equipment: List[EquipmentProfile] = Field(
        ...,
        min_items=1,
        description="Equipment profiles"
    )

    # Production parameters
    production: ProductionParameters = Field(..., description="Production parameters")

    # Failure scenarios
    scenarios: List[FailureScenario] = Field(
        default_factory=list,
        description="Failure scenarios to analyze"
    )

    # Analysis parameters
    analysis_period_years: float = Field(
        default=1.0,
        gt=0,
        le=10,
        description="Analysis period in years"
    )
    monte_carlo_iterations: int = Field(
        default=10000,
        ge=100,
        le=100000,
        description="Monte Carlo iterations"
    )
    confidence_level: float = Field(
        default=0.95,
        ge=0.8,
        le=0.99,
        description="Confidence level for estimates"
    )
    random_seed: Optional[int] = Field(
        default=42,
        description="Random seed for reproducibility"
    )


class DowntimeCostResult(BaseModel):
    """Downtime cost breakdown."""

    total_downtime_hours: float = Field(..., description="Total downtime hours")
    lost_production_units: float = Field(..., description="Lost production units")
    lost_production_value: float = Field(..., description="Lost production value $")
    quality_loss_value: float = Field(..., description="Quality loss value $")
    energy_waste_value: float = Field(..., description="Energy waste during startup $")
    repair_cost: float = Field(..., description="Direct repair costs $")
    labor_overtime_cost: float = Field(..., description="Overtime labor cost $")
    expedited_parts_cost: float = Field(..., description="Expedited parts cost $")
    customer_penalty: float = Field(..., description="Customer penalties $")
    total_cost: float = Field(..., description="Total downtime cost $")


class ScenarioResult(BaseModel):
    """Result for a single failure scenario."""

    scenario_id: str = Field(..., description="Scenario identifier")
    scenario_name: str = Field(..., description="Scenario name")
    probability: float = Field(..., description="Annual probability")
    expected_frequency: float = Field(..., description="Expected occurrences per year")
    expected_duration_hours: float = Field(..., description="Expected downtime hours")
    expected_cost: float = Field(..., description="Expected annual cost $")
    cost_per_occurrence: float = Field(..., description="Cost per occurrence $")
    severity: ImpactSeverity = Field(..., description="Impact severity")
    risk_priority_number: float = Field(..., description="Risk Priority Number (RPN)")


class MonteCarloResult(BaseModel):
    """Monte Carlo simulation results."""

    iterations: int = Field(..., description="Number of iterations")
    mean_annual_cost: float = Field(..., description="Mean annual cost $")
    std_annual_cost: float = Field(..., description="Std dev of annual cost $")
    percentile_5: float = Field(..., description="5th percentile cost $")
    percentile_25: float = Field(..., description="25th percentile cost $")
    percentile_50: float = Field(..., description="Median cost $")
    percentile_75: float = Field(..., description="75th percentile cost $")
    percentile_95: float = Field(..., description="95th percentile cost $")
    percentile_99: float = Field(..., description="99th percentile cost $")
    var_95: float = Field(..., description="Value at Risk (95%) $")
    expected_shortfall: float = Field(..., description="Expected Shortfall (CVaR) $")
    cost_distribution: Dict[str, int] = Field(
        default_factory=dict,
        description="Cost distribution histogram"
    )


class ProductionImpactOutput(BaseModel):
    """Output from production impact analysis."""

    analysis_name: str = Field(..., description="Analysis name")
    analysis_period_years: float = Field(..., description="Analysis period")

    # Summary metrics
    expected_annual_downtime_hours: float = Field(..., description="Expected annual downtime")
    expected_annual_availability: float = Field(..., description="Expected availability %")
    expected_annual_production_loss: float = Field(..., description="Expected production loss")
    expected_annual_cost: float = Field(..., description="Total expected annual cost $")

    # Breakdown by cost category
    cost_by_category: Dict[str, float] = Field(
        default_factory=dict,
        description="Cost breakdown by category"
    )

    # Downtime cost detail
    downtime_cost_detail: DowntimeCostResult = Field(
        ...,
        description="Detailed downtime cost breakdown"
    )

    # Scenario results
    scenario_results: List[ScenarioResult] = Field(
        default_factory=list,
        description="Results by scenario"
    )
    top_risk_scenarios: List[str] = Field(
        default_factory=list,
        description="Top 5 scenarios by expected cost"
    )

    # Monte Carlo results
    monte_carlo_result: MonteCarloResult = Field(
        ...,
        description="Monte Carlo simulation results"
    )

    # Risk metrics
    maximum_credible_loss: float = Field(..., description="Maximum credible loss (99th pct)")
    risk_adjusted_cost: float = Field(..., description="Risk-adjusted annual cost")

    # Recommendations
    recommendations: List[str] = Field(
        default_factory=list,
        description="Risk mitigation recommendations"
    )

    # Provenance
    provenance_hash: str = Field(..., description="SHA-256 hash for audit")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    processing_time_ms: float = Field(..., description="Processing time")

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


# ============================================================================
# Core Production Impact Engine
# ============================================================================

class ProductionImpactModeler:
    """
    Production Impact Modeling Engine.

    ZERO-HALLUCINATION GUARANTEE:
    - All calculations use deterministic formulas
    - Monte Carlo uses seeded random for reproducibility
    - No LLM involvement in financial calculations
    - Complete provenance tracking

    Capabilities:
    - Downtime cost estimation
    - Throughput impact analysis
    - Energy cost impact modeling
    - Quality impact assessment
    - Monte Carlo simulation
    - Risk prioritization

    Integration:
    - Process Heat agents (GL-003 through GL-018)
    - GL-013 PredictiveMaint for failure prediction
    - GL-011 Fuel Module for energy costs

    Example:
        >>> modeler = ProductionImpactModeler()
        >>> result = modeler.analyze(input_data)
        >>> print(f"Expected annual cost: ${result.expected_annual_cost:,.2f}")
    """

    # Severity thresholds (% production loss)
    SEVERITY_THRESHOLDS = {
        ImpactSeverity.NEGLIGIBLE: 0.01,
        ImpactSeverity.MINOR: 0.05,
        ImpactSeverity.MODERATE: 0.25,
        ImpactSeverity.MAJOR: 0.75,
        ImpactSeverity.CATASTROPHIC: 1.0
    }

    # Startup energy multiplier (extra energy during restart)
    STARTUP_ENERGY_MULTIPLIER = 1.5

    # Overtime labor multiplier
    OVERTIME_MULTIPLIER = 1.5

    # Expedited shipping premium
    EXPEDITED_PREMIUM = 2.0

    def __init__(self, precision: int = 2):
        """
        Initialize production impact modeler.

        Args:
            precision: Decimal precision for outputs
        """
        self.precision = precision
        logger.info("ProductionImpactModeler initialized")

    def _apply_precision(self, value: float) -> float:
        """Apply precision rounding."""
        if self.precision == 0:
            return round(value)
        return round(value, self.precision)

    def _calculate_provenance(self, inputs: Dict, outputs: Dict) -> str:
        """Calculate SHA-256 hash for audit trail."""
        provenance_data = {
            "method": "ProductionImpactModeler",
            "inputs": {k: str(v) for k, v in inputs.items()},
            "outputs": {k: str(v) for k, v in outputs.items()}
        }
        provenance_str = str(sorted(provenance_data.items()))
        return hashlib.sha256(provenance_str.encode()).hexdigest()

    def _triangular_sample(
        self,
        low: float,
        mode: Optional[float],
        high: float,
        rng: random.Random
    ) -> float:
        """
        Sample from triangular distribution.

        DETERMINISTIC with seeded RNG.
        """
        if mode is None:
            mode = (low + high) / 2

        # Ensure proper ordering
        low = min(low, mode, high)
        high = max(low, mode, high)
        mode = max(low, min(mode, high))

        if high == low:
            return low

        u = rng.random()
        fc = (mode - low) / (high - low)

        if u < fc:
            return low + math.sqrt(u * (high - low) * (mode - low))
        else:
            return high - math.sqrt((1 - u) * (high - low) * (high - mode))

    def _poisson_sample(self, rate: float, rng: random.Random) -> int:
        """
        Sample from Poisson distribution.

        DETERMINISTIC with seeded RNG.
        Using inverse transform method.
        """
        if rate <= 0:
            return 0

        l_exp = math.exp(-rate)
        k = 0
        p = 1.0

        while p > l_exp:
            k += 1
            p *= rng.random()

        return k - 1

    def _classify_severity(self, capacity_loss_pct: float) -> ImpactSeverity:
        """Classify impact severity based on capacity loss."""
        loss_fraction = capacity_loss_pct / 100

        if loss_fraction < self.SEVERITY_THRESHOLDS[ImpactSeverity.NEGLIGIBLE]:
            return ImpactSeverity.NEGLIGIBLE
        elif loss_fraction < self.SEVERITY_THRESHOLDS[ImpactSeverity.MINOR]:
            return ImpactSeverity.MINOR
        elif loss_fraction < self.SEVERITY_THRESHOLDS[ImpactSeverity.MODERATE]:
            return ImpactSeverity.MODERATE
        elif loss_fraction < self.SEVERITY_THRESHOLDS[ImpactSeverity.MAJOR]:
            return ImpactSeverity.MAJOR
        else:
            return ImpactSeverity.CATASTROPHIC

    def _calculate_lost_production_cost(
        self,
        duration_hours: float,
        production: ProductionParameters,
        capacity_loss_pct: float
    ) -> float:
        """Calculate lost production value."""
        hourly_production = production.hourly_production_rate * production.current_utilization
        lost_units = hourly_production * duration_hours * (capacity_loss_pct / 100)
        return lost_units * production.production_value_per_unit

    def _calculate_quality_loss_cost(
        self,
        duration_hours: float,
        production: ProductionParameters,
        quality_impact_pct: float
    ) -> float:
        """Calculate quality degradation cost."""
        if quality_impact_pct <= 0:
            return 0

        hourly_production = production.hourly_production_rate * production.current_utilization
        affected_units = hourly_production * duration_hours * (quality_impact_pct / 100)

        # Split between rework and scrap
        rework_fraction = 0.7  # 70% reworkable
        scrap_fraction = 0.3

        rework_cost = affected_units * rework_fraction * production.rework_cost_per_unit
        scrap_cost = affected_units * scrap_fraction * production.scrap_cost_per_unit

        return rework_cost + scrap_cost

    def _calculate_energy_waste_cost(
        self,
        duration_hours: float,
        equipment: List[EquipmentProfile],
        production: ProductionParameters
    ) -> float:
        """Calculate energy wasted during startup/restart."""
        # Total energy consumption of affected equipment
        total_energy_kwh = sum(eq.energy_consumption_kwh for eq in equipment)

        # Startup energy (assume 2 hours of extra energy consumption)
        startup_hours = min(2.0, duration_hours * 0.1)
        startup_energy = total_energy_kwh * startup_hours * (self.STARTUP_ENERGY_MULTIPLIER - 1)

        return startup_energy * production.energy_cost_per_kwh

    def _calculate_labor_overtime_cost(
        self,
        duration_hours: float,
        equipment: List[EquipmentProfile],
        is_emergency: bool = True
    ) -> float:
        """Calculate overtime labor cost for repairs."""
        if not is_emergency:
            return 0

        # Assume 2 technicians for repair
        technicians = 2
        avg_hourly_rate = sum(eq.hourly_labor_cost for eq in equipment) / len(equipment)

        # Overtime for hours beyond 8
        regular_hours = min(8, duration_hours)
        overtime_hours = max(0, duration_hours - 8)

        regular_cost = regular_hours * technicians * avg_hourly_rate
        overtime_cost = overtime_hours * technicians * avg_hourly_rate * self.OVERTIME_MULTIPLIER

        return regular_cost + overtime_cost

    def _calculate_expedited_parts_cost(
        self,
        repair_cost: float,
        duration_hours: float
    ) -> float:
        """Calculate expedited parts premium."""
        # If repair needed quickly, assume 40% of parts cost expedited
        if duration_hours < 24:
            parts_fraction = 0.5  # 50% of repair is parts
            expedited_fraction = 0.4
            return repair_cost * parts_fraction * expedited_fraction * (self.EXPEDITED_PREMIUM - 1)
        return 0

    def _calculate_customer_penalty(
        self,
        duration_hours: float,
        production: ProductionParameters
    ) -> float:
        """Calculate customer penalty costs."""
        if production.customer_penalty_per_day <= 0:
            return 0

        # Penalty typically kicks in after 24 hours
        if duration_hours > 24:
            penalty_days = (duration_hours - 24) / 24
            return penalty_days * production.customer_penalty_per_day

        return 0

    def _calculate_scenario_cost(
        self,
        scenario: FailureScenario,
        duration_hours: float,
        repair_cost: float,
        production: ProductionParameters,
        equipment: List[EquipmentProfile]
    ) -> DowntimeCostResult:
        """Calculate total cost for a scenario occurrence."""
        # Lost production
        lost_production_value = self._calculate_lost_production_cost(
            duration_hours, production, scenario.capacity_loss_pct
        )

        lost_production_units = (
            production.hourly_production_rate *
            production.current_utilization *
            duration_hours *
            (scenario.capacity_loss_pct / 100)
        )

        # Quality loss
        quality_loss = self._calculate_quality_loss_cost(
            duration_hours, production, scenario.quality_impact_pct
        )

        # Energy waste
        energy_waste = self._calculate_energy_waste_cost(
            duration_hours, equipment, production
        )

        # Labor overtime
        labor_overtime = self._calculate_labor_overtime_cost(
            duration_hours, equipment, is_emergency=True
        )

        # Expedited parts
        expedited_parts = self._calculate_expedited_parts_cost(
            repair_cost, duration_hours
        )

        # Customer penalty
        customer_penalty = self._calculate_customer_penalty(
            duration_hours, production
        )

        # Total
        total_cost = (
            lost_production_value +
            quality_loss +
            energy_waste +
            repair_cost +
            labor_overtime +
            expedited_parts +
            customer_penalty
        )

        return DowntimeCostResult(
            total_downtime_hours=duration_hours,
            lost_production_units=lost_production_units,
            lost_production_value=lost_production_value,
            quality_loss_value=quality_loss,
            energy_waste_value=energy_waste,
            repair_cost=repair_cost,
            labor_overtime_cost=labor_overtime,
            expedited_parts_cost=expedited_parts,
            customer_penalty=customer_penalty,
            total_cost=total_cost
        )

    def _run_monte_carlo(
        self,
        scenarios: List[FailureScenario],
        production: ProductionParameters,
        equipment: List[EquipmentProfile],
        iterations: int,
        analysis_period_years: float,
        random_seed: Optional[int]
    ) -> MonteCarloResult:
        """
        Run Monte Carlo simulation for uncertainty analysis.

        DETERMINISTIC with seeded RNG for reproducibility.

        Args:
            scenarios: Failure scenarios
            production: Production parameters
            equipment: Equipment list
            iterations: Number of iterations
            analysis_period_years: Analysis period
            random_seed: Random seed

        Returns:
            MonteCarloResult with distribution
        """
        rng = random.Random(random_seed if random_seed is not None else 42)

        annual_costs = []

        for _ in range(iterations):
            iteration_cost = 0.0

            for scenario in scenarios:
                # Sample number of occurrences (Poisson)
                annual_rate = scenario.probability * analysis_period_years
                occurrences = self._poisson_sample(annual_rate, rng)

                for _ in range(occurrences):
                    # Sample duration (triangular)
                    duration = self._triangular_sample(
                        scenario.duration_hours_min,
                        scenario.duration_hours_mode,
                        scenario.duration_hours_max,
                        rng
                    )

                    # Sample repair cost (triangular)
                    repair_cost = self._triangular_sample(
                        scenario.repair_cost_min,
                        None,
                        scenario.repair_cost_max,
                        rng
                    )

                    # Calculate cost
                    cost_result = self._calculate_scenario_cost(
                        scenario, duration, repair_cost, production, equipment
                    )

                    iteration_cost += cost_result.total_cost

            annual_costs.append(iteration_cost / analysis_period_years)

        # Calculate statistics
        annual_costs.sort()
        n = len(annual_costs)

        mean_cost = sum(annual_costs) / n
        variance = sum((c - mean_cost)**2 for c in annual_costs) / n
        std_cost = math.sqrt(variance)

        # Percentiles
        def percentile(data: List[float], p: float) -> float:
            idx = int(p * (len(data) - 1))
            return data[idx]

        pct_5 = percentile(annual_costs, 0.05)
        pct_25 = percentile(annual_costs, 0.25)
        pct_50 = percentile(annual_costs, 0.50)
        pct_75 = percentile(annual_costs, 0.75)
        pct_95 = percentile(annual_costs, 0.95)
        pct_99 = percentile(annual_costs, 0.99)

        # VaR and ES
        var_95 = pct_95
        tail_start = int(0.95 * n)
        tail_costs = annual_costs[tail_start:]
        expected_shortfall = sum(tail_costs) / len(tail_costs) if tail_costs else pct_95

        # Cost distribution histogram
        if max(annual_costs) > 0:
            bin_width = max(annual_costs) / 10
            distribution = {}
            for c in annual_costs:
                bin_idx = int(c / bin_width) if bin_width > 0 else 0
                bin_label = f"${int(bin_idx * bin_width):,}-${int((bin_idx+1) * bin_width):,}"
                distribution[bin_label] = distribution.get(bin_label, 0) + 1
        else:
            distribution = {"$0": n}

        return MonteCarloResult(
            iterations=iterations,
            mean_annual_cost=self._apply_precision(mean_cost),
            std_annual_cost=self._apply_precision(std_cost),
            percentile_5=self._apply_precision(pct_5),
            percentile_25=self._apply_precision(pct_25),
            percentile_50=self._apply_precision(pct_50),
            percentile_75=self._apply_precision(pct_75),
            percentile_95=self._apply_precision(pct_95),
            percentile_99=self._apply_precision(pct_99),
            var_95=self._apply_precision(var_95),
            expected_shortfall=self._apply_precision(expected_shortfall),
            cost_distribution=distribution
        )

    def _generate_recommendations(
        self,
        scenario_results: List[ScenarioResult],
        monte_carlo: MonteCarloResult,
        total_annual_cost: float
    ) -> List[str]:
        """Generate risk mitigation recommendations."""
        recommendations = []

        # Top risk scenarios
        sorted_scenarios = sorted(
            scenario_results,
            key=lambda x: x.expected_cost,
            reverse=True
        )

        if sorted_scenarios:
            top = sorted_scenarios[0]
            recommendations.append(
                f"PRIORITY: Address '{top.scenario_name}' - highest expected cost "
                f"(${top.expected_cost:,.0f}/year, {top.expected_frequency:.1f} occurrences/year)"
            )

        # High RPN scenarios
        high_rpn = [s for s in scenario_results if s.risk_priority_number > 100]
        if high_rpn:
            recommendations.append(
                f"ATTENTION: {len(high_rpn)} scenarios have high Risk Priority Numbers (>100) - "
                f"consider failure mode analysis"
            )

        # Catastrophic scenarios
        catastrophic = [s for s in scenario_results if s.severity == ImpactSeverity.CATASTROPHIC]
        if catastrophic:
            recommendations.append(
                f"CRITICAL: {len(catastrophic)} catastrophic failure scenarios identified - "
                f"implement redundancy or backup systems"
            )

        # High variance
        if monte_carlo.std_annual_cost > monte_carlo.mean_annual_cost * 0.5:
            recommendations.append(
                "HIGH UNCERTAINTY: Cost variance is high - consider insurance or "
                "reserve fund for unexpected failures"
            )

        # 99th percentile exposure
        if monte_carlo.percentile_99 > total_annual_cost * 3:
            recommendations.append(
                f"TAIL RISK: 1% probability of annual costs exceeding "
                f"${monte_carlo.percentile_99:,.0f} - ensure adequate contingency budget"
            )

        # General recommendations
        recommendations.append(
            "Implement condition-based monitoring to detect failures early and reduce downtime"
        )
        recommendations.append(
            "Review spare parts inventory for critical failure scenarios to minimize MTTR"
        )
        recommendations.append(
            "Consider preventive maintenance scheduling to reduce failure probability"
        )

        return recommendations[:8]

    def analyze(self, input_data: ProductionImpactInput) -> ProductionImpactOutput:
        """
        Perform comprehensive production impact analysis.

        ZERO-HALLUCINATION: All calculations are deterministic.
        Monte Carlo simulation uses seeded RNG for reproducibility.

        Args:
            input_data: Validated input with scenarios and parameters

        Returns:
            ProductionImpactOutput with analysis results

        Example:
            >>> modeler = ProductionImpactModeler()
            >>> result = modeler.analyze(input_data)
            >>> print(f"Expected annual cost: ${result.expected_annual_cost:,.2f}")
        """
        start_time = datetime.now()

        logger.info(
            f"Starting production impact analysis: {input_data.analysis_name}, "
            f"{len(input_data.scenarios)} scenarios, "
            f"{input_data.monte_carlo_iterations} Monte Carlo iterations"
        )

        # Calculate expected values for each scenario
        scenario_results = []
        total_expected_cost = 0
        total_expected_downtime = 0

        for scenario in input_data.scenarios:
            # Expected frequency
            expected_frequency = scenario.probability * input_data.analysis_period_years

            # Expected duration
            expected_duration = (
                scenario.duration_hours_min +
                (scenario.duration_hours_mode or (scenario.duration_hours_min + scenario.duration_hours_max) / 2) +
                scenario.duration_hours_max
            ) / 3  # Triangular mean

            # Expected repair cost
            expected_repair = (scenario.repair_cost_min + scenario.repair_cost_max) / 2

            # Calculate expected cost per occurrence
            cost_result = self._calculate_scenario_cost(
                scenario,
                expected_duration,
                expected_repair,
                input_data.production,
                input_data.equipment
            )

            expected_annual_cost = cost_result.total_cost * expected_frequency
            total_expected_cost += expected_annual_cost
            total_expected_downtime += expected_duration * expected_frequency

            # Severity and RPN
            severity = self._classify_severity(scenario.capacity_loss_pct)

            # RPN = Probability * Severity * Duration
            severity_score = {
                ImpactSeverity.NEGLIGIBLE: 1,
                ImpactSeverity.MINOR: 2,
                ImpactSeverity.MODERATE: 4,
                ImpactSeverity.MAJOR: 7,
                ImpactSeverity.CATASTROPHIC: 10
            }.get(severity, 5)

            duration_score = min(10, expected_duration / 4)  # 4 hours = score of 1
            probability_score = min(10, scenario.probability * 10)

            rpn = severity_score * duration_score * probability_score

            scenario_results.append(ScenarioResult(
                scenario_id=scenario.scenario_id,
                scenario_name=scenario.scenario_name,
                probability=scenario.probability,
                expected_frequency=self._apply_precision(expected_frequency),
                expected_duration_hours=self._apply_precision(expected_duration),
                expected_cost=self._apply_precision(expected_annual_cost),
                cost_per_occurrence=self._apply_precision(cost_result.total_cost),
                severity=severity,
                risk_priority_number=self._apply_precision(rpn)
            ))

        # Top risk scenarios
        sorted_by_cost = sorted(scenario_results, key=lambda x: x.expected_cost, reverse=True)
        top_risk_scenarios = [s.scenario_id for s in sorted_by_cost[:5]]

        # Aggregate downtime cost detail
        total_hours = input_data.production.operating_hours_per_day * input_data.production.operating_days_per_year
        availability = (total_hours - total_expected_downtime) / total_hours if total_hours > 0 else 1

        # Cost breakdown by category (estimated proportions)
        cost_by_category = {
            CostCategory.LOST_PRODUCTION.value: total_expected_cost * 0.60,
            CostCategory.QUALITY_LOSS.value: total_expected_cost * 0.08,
            CostCategory.ENERGY_WASTE.value: total_expected_cost * 0.05,
            CostCategory.REPAIR_COST.value: total_expected_cost * 0.15,
            CostCategory.LABOR_OVERTIME.value: total_expected_cost * 0.05,
            CostCategory.EXPEDITED_PARTS.value: total_expected_cost * 0.03,
            CostCategory.CUSTOMER_PENALTY.value: total_expected_cost * 0.04
        }

        # Downtime cost detail
        lost_production = (
            input_data.production.hourly_production_rate *
            input_data.production.current_utilization *
            total_expected_downtime
        )

        downtime_detail = DowntimeCostResult(
            total_downtime_hours=self._apply_precision(total_expected_downtime),
            lost_production_units=self._apply_precision(lost_production),
            lost_production_value=self._apply_precision(cost_by_category[CostCategory.LOST_PRODUCTION.value]),
            quality_loss_value=self._apply_precision(cost_by_category[CostCategory.QUALITY_LOSS.value]),
            energy_waste_value=self._apply_precision(cost_by_category[CostCategory.ENERGY_WASTE.value]),
            repair_cost=self._apply_precision(cost_by_category[CostCategory.REPAIR_COST.value]),
            labor_overtime_cost=self._apply_precision(cost_by_category[CostCategory.LABOR_OVERTIME.value]),
            expedited_parts_cost=self._apply_precision(cost_by_category[CostCategory.EXPEDITED_PARTS.value]),
            customer_penalty=self._apply_precision(cost_by_category[CostCategory.CUSTOMER_PENALTY.value]),
            total_cost=self._apply_precision(total_expected_cost)
        )

        # Monte Carlo simulation
        monte_carlo = self._run_monte_carlo(
            input_data.scenarios,
            input_data.production,
            input_data.equipment,
            input_data.monte_carlo_iterations,
            input_data.analysis_period_years,
            input_data.random_seed
        )

        # Risk metrics
        maximum_credible_loss = monte_carlo.percentile_99
        risk_adjusted_cost = monte_carlo.mean_annual_cost + 0.5 * monte_carlo.std_annual_cost

        # Recommendations
        recommendations = self._generate_recommendations(
            scenario_results, monte_carlo, total_expected_cost
        )

        # Provenance
        provenance_inputs = {
            "analysis_name": input_data.analysis_name,
            "n_scenarios": len(input_data.scenarios),
            "n_equipment": len(input_data.equipment),
            "monte_carlo_iterations": input_data.monte_carlo_iterations
        }
        provenance_outputs = {
            "expected_annual_cost": total_expected_cost,
            "expected_downtime": total_expected_downtime,
            "monte_carlo_mean": monte_carlo.mean_annual_cost
        }
        provenance_hash = self._calculate_provenance(provenance_inputs, provenance_outputs)

        processing_time = (datetime.now() - start_time).total_seconds() * 1000

        logger.info(
            f"Analysis complete: expected_cost=${total_expected_cost:,.0f}/year, "
            f"availability={availability*100:.1f}%, processing={processing_time:.1f}ms"
        )

        return ProductionImpactOutput(
            analysis_name=input_data.analysis_name,
            analysis_period_years=input_data.analysis_period_years,
            expected_annual_downtime_hours=self._apply_precision(total_expected_downtime),
            expected_annual_availability=self._apply_precision(availability * 100),
            expected_annual_production_loss=self._apply_precision(lost_production),
            expected_annual_cost=self._apply_precision(total_expected_cost),
            cost_by_category={k: self._apply_precision(v) for k, v in cost_by_category.items()},
            downtime_cost_detail=downtime_detail,
            scenario_results=scenario_results,
            top_risk_scenarios=top_risk_scenarios,
            monte_carlo_result=monte_carlo,
            maximum_credible_loss=self._apply_precision(maximum_credible_loss),
            risk_adjusted_cost=self._apply_precision(risk_adjusted_cost),
            recommendations=recommendations,
            provenance_hash=provenance_hash,
            processing_time_ms=processing_time
        )

    def quick_downtime_cost(
        self,
        downtime_hours: float,
        hourly_production_rate: float,
        production_value_per_unit: float,
        repair_cost: float = 0
    ) -> float:
        """
        Quick downtime cost calculation.

        Args:
            downtime_hours: Downtime duration
            hourly_production_rate: Production rate per hour
            production_value_per_unit: Value per unit
            repair_cost: Direct repair cost

        Returns:
            Total downtime cost
        """
        lost_production = downtime_hours * hourly_production_rate * production_value_per_unit
        return lost_production + repair_cost


# ============================================================================
# Convenience Functions
# ============================================================================

def estimate_downtime_cost(
    downtime_hours: float,
    hourly_production_value: float,
    repair_cost: float = 0,
    energy_waste: float = 0,
    overtime_labor: float = 0
) -> Dict[str, float]:
    """
    Quick downtime cost estimation.

    Example:
        >>> result = estimate_downtime_cost(
        ...     downtime_hours=8,
        ...     hourly_production_value=5000,
        ...     repair_cost=2500
        ... )
        >>> print(f"Total cost: ${result['total_cost']:,.2f}")
    """
    lost_production = downtime_hours * hourly_production_value

    total = lost_production + repair_cost + energy_waste + overtime_labor

    return {
        "downtime_hours": downtime_hours,
        "lost_production_cost": lost_production,
        "repair_cost": repair_cost,
        "energy_waste_cost": energy_waste,
        "overtime_labor_cost": overtime_labor,
        "total_cost": total
    }


def calculate_availability(
    mtbf_hours: float,
    mttr_hours: float
) -> Dict[str, float]:
    """
    Calculate availability from MTBF and MTTR.

    Availability = MTBF / (MTBF + MTTR)

    Example:
        >>> result = calculate_availability(mtbf_hours=1000, mttr_hours=4)
        >>> print(f"Availability: {result['availability_pct']:.2f}%")
    """
    availability = mtbf_hours / (mtbf_hours + mttr_hours)
    failure_rate = 8760 / mtbf_hours  # Failures per year
    expected_downtime = failure_rate * mttr_hours

    return {
        "mtbf_hours": mtbf_hours,
        "mttr_hours": mttr_hours,
        "availability": availability,
        "availability_pct": availability * 100,
        "failure_rate_per_year": failure_rate,
        "expected_downtime_hours_per_year": expected_downtime
    }


def calculate_risk_priority_number(
    probability: float,
    severity: int,
    detectability: int
) -> Dict[str, Any]:
    """
    Calculate Risk Priority Number (RPN).

    RPN = Severity x Occurrence x Detection

    Scale: 1-10 for each factor, RPN range: 1-1000

    Args:
        probability: Annual probability (0-1)
        severity: Severity rating (1-10)
        detectability: Detection difficulty (1-10, higher = harder to detect)

    Returns:
        RPN analysis dictionary
    """
    # Convert probability to occurrence scale
    if probability >= 0.5:
        occurrence = 10
    elif probability >= 0.2:
        occurrence = 8
    elif probability >= 0.1:
        occurrence = 6
    elif probability >= 0.05:
        occurrence = 4
    elif probability >= 0.01:
        occurrence = 2
    else:
        occurrence = 1

    rpn = severity * occurrence * detectability

    if rpn >= 200:
        priority = "HIGH - Immediate action required"
    elif rpn >= 100:
        priority = "MEDIUM - Action should be planned"
    elif rpn >= 50:
        priority = "LOW - Monitor and review"
    else:
        priority = "ACCEPTABLE - No immediate action"

    return {
        "severity": severity,
        "occurrence": occurrence,
        "detectability": detectability,
        "rpn": rpn,
        "priority": priority,
        "max_rpn": 1000
    }


# ============================================================================
# Unit Test Stubs
# ============================================================================

class TestProductionImpactModeler:
    """Unit tests for ProductionImpactModeler."""

    def test_init(self):
        """Test initialization."""
        modeler = ProductionImpactModeler()
        assert modeler.precision == 2

    def test_triangular_sample(self):
        """Test triangular distribution sampling."""
        modeler = ProductionImpactModeler()
        rng = random.Random(42)

        samples = [
            modeler._triangular_sample(1, 3, 5, rng)
            for _ in range(100)
        ]

        assert all(1 <= s <= 5 for s in samples)
        mean = sum(samples) / len(samples)
        assert 2 < mean < 4  # Should be around 3

    def test_poisson_sample(self):
        """Test Poisson distribution sampling."""
        modeler = ProductionImpactModeler()
        rng = random.Random(42)

        # Rate = 5, should average around 5
        samples = [modeler._poisson_sample(5, rng) for _ in range(1000)]

        mean = sum(samples) / len(samples)
        assert 4 < mean < 6

    def test_severity_classification(self):
        """Test severity classification."""
        modeler = ProductionImpactModeler()

        assert modeler._classify_severity(0) == ImpactSeverity.NEGLIGIBLE
        assert modeler._classify_severity(3) == ImpactSeverity.MINOR
        assert modeler._classify_severity(15) == ImpactSeverity.MODERATE
        assert modeler._classify_severity(50) == ImpactSeverity.MAJOR
        assert modeler._classify_severity(100) == ImpactSeverity.CATASTROPHIC

    def test_lost_production_cost(self):
        """Test lost production calculation."""
        modeler = ProductionImpactModeler()

        production = ProductionParameters(
            production_value_per_unit=100,
            hourly_production_rate=10,
            current_utilization=0.9
        )

        cost = modeler._calculate_lost_production_cost(
            duration_hours=8,
            production=production,
            capacity_loss_pct=100
        )

        # 8 hours * 10 units/hr * 0.9 utilization * 100% loss * $100/unit = $7200
        assert cost == 7200

    def test_monte_carlo_determinism(self):
        """Test Monte Carlo is deterministic with seed."""
        modeler = ProductionImpactModeler()

        scenarios = [
            FailureScenario(
                scenario_id="S1",
                scenario_name="Test Scenario",
                failure_mode=FailureMode.MECHANICAL,
                probability=0.1,
                duration_hours_min=2,
                duration_hours_max=8,
                capacity_loss_pct=100,
                repair_cost_min=1000,
                repair_cost_max=5000
            )
        ]

        production = ProductionParameters(
            production_value_per_unit=100,
            hourly_production_rate=10
        )

        equipment = [
            EquipmentProfile(
                equipment_id="EQ1",
                equipment_name="Test Equipment",
                equipment_type="pump",
                rated_capacity=100
            )
        ]

        result1 = modeler._run_monte_carlo(
            scenarios, production, equipment, 1000, 1.0, 42
        )
        result2 = modeler._run_monte_carlo(
            scenarios, production, equipment, 1000, 1.0, 42
        )

        assert result1.mean_annual_cost == result2.mean_annual_cost

    def test_provenance_hash(self):
        """Test provenance hash is deterministic."""
        modeler = ProductionImpactModeler()

        inputs = {"test": "value"}
        outputs = {"cost": 1000}

        hash1 = modeler._calculate_provenance(inputs, outputs)
        hash2 = modeler._calculate_provenance(inputs, outputs)

        assert hash1 == hash2
        assert len(hash1) == 64  # SHA-256

    def test_convenience_functions(self):
        """Test convenience functions."""
        # Downtime cost
        result = estimate_downtime_cost(
            downtime_hours=8,
            hourly_production_value=5000,
            repair_cost=2500
        )
        assert result['total_cost'] == 8 * 5000 + 2500

        # Availability
        avail = calculate_availability(mtbf_hours=1000, mttr_hours=4)
        assert 0.99 < avail['availability'] < 1.0

        # RPN
        rpn = calculate_risk_priority_number(
            probability=0.1,
            severity=8,
            detectability=6
        )
        assert rpn['rpn'] == 8 * 6 * 6  # occurrence = 6 for p=0.1
