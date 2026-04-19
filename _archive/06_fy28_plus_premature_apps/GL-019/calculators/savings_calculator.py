"""
GL-019 HEATSCHEDULER - Savings Calculator

Zero-hallucination, deterministic calculations for cost savings analysis
comparing baseline vs optimized heating schedules.

This module provides:
- Baseline vs optimized schedule comparison
- Savings breakdown by category (load shifting, demand reduction, peak avoidance)
- ROI calculations for schedule optimization
- Annual savings projections
- Payback period calculations

Standards Reference:
- ISO 50001 - Energy Management Systems
- ISO 50006 - Measuring Energy Performance Using Baselines
- ASHRAE Guideline 14 - Measurement of Energy and Demand Savings
- IPMVP - International Performance Measurement and Verification Protocol

Author: GL-CalculatorEngineer
Version: 1.0.0
"""

from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
import math
import statistics

from .provenance import ProvenanceTracker, ProvenanceRecord


# =============================================================================
# CONSTANTS AND REFERENCE DATA
# =============================================================================

# Standard operating hours per year
HOURS_PER_YEAR = 8760

# Business days per year (typical)
BUSINESS_DAYS_PER_YEAR = 260

# Typical operating hours per business day
OPERATING_HOURS_PER_DAY = 16

# Currency precision
CURRENCY_PRECISION = 2

# NPV discount rate default
DEFAULT_DISCOUNT_RATE = 0.08  # 8%

# Project lifetime default (years)
DEFAULT_PROJECT_LIFE_YEARS = 10


class SavingsCategory(str, Enum):
    """Categories of energy cost savings."""
    LOAD_SHIFTING = "load_shifting"
    DEMAND_REDUCTION = "demand_reduction"
    PEAK_AVOIDANCE = "peak_avoidance"
    EFFICIENCY_IMPROVEMENT = "efficiency_improvement"
    THERMAL_STORAGE = "thermal_storage"
    TOTAL = "total"


class VerificationMethod(str, Enum):
    """IPMVP verification methods."""
    OPTION_A = "option_a"  # Retrofit isolation, key parameter measurement
    OPTION_B = "option_b"  # Retrofit isolation, all parameter measurement
    OPTION_C = "option_c"  # Whole facility, utility bill analysis
    OPTION_D = "option_d"  # Calibrated simulation


# =============================================================================
# INPUT/OUTPUT DATA STRUCTURES
# =============================================================================

@dataclass(frozen=True)
class HourlyScheduleData:
    """
    Hourly schedule data for comparison.

    Attributes:
        hour: Hour index
        energy_kwh: Energy consumption (kWh)
        demand_kw: Peak demand (kW)
        energy_rate: Energy rate (currency/kWh)
        demand_rate: Demand rate (currency/kW)
        is_peak_hour: Whether this is a peak hour
    """
    hour: int
    energy_kwh: float
    demand_kw: float
    energy_rate: float
    demand_rate: float = 0.0
    is_peak_hour: bool = False


@dataclass(frozen=True)
class ScheduleComparison:
    """
    Baseline and optimized schedule data.

    Attributes:
        baseline_schedule: Baseline (unoptimized) schedule
        optimized_schedule: Optimized schedule
        analysis_period_days: Number of days in analysis
        baseline_description: Description of baseline scenario
        optimized_description: Description of optimized scenario
    """
    baseline_schedule: List[HourlyScheduleData]
    optimized_schedule: List[HourlyScheduleData]
    analysis_period_days: int = 30
    baseline_description: str = "Current operation"
    optimized_description: str = "Optimized schedule"


@dataclass(frozen=True)
class ProjectCosts:
    """
    Implementation costs for ROI calculation.

    Attributes:
        capital_cost: One-time capital investment
        implementation_cost: Implementation/installation cost
        annual_maintenance_cost: Annual maintenance cost
        software_license_annual: Annual software license cost
        training_cost: One-time training cost
    """
    capital_cost: float = 0.0
    implementation_cost: float = 0.0
    annual_maintenance_cost: float = 0.0
    software_license_annual: float = 0.0
    training_cost: float = 0.0


@dataclass(frozen=True)
class SavingsCalculatorInput:
    """
    Input for savings calculations.

    Attributes:
        schedule_comparison: Baseline vs optimized schedules
        project_costs: Implementation costs
        projection_years: Years for annual projection
        discount_rate: Discount rate for NPV
        verification_method: IPMVP verification method
    """
    schedule_comparison: ScheduleComparison
    project_costs: Optional[ProjectCosts] = None
    projection_years: int = DEFAULT_PROJECT_LIFE_YEARS
    discount_rate: float = DEFAULT_DISCOUNT_RATE
    verification_method: VerificationMethod = VerificationMethod.OPTION_C


@dataclass(frozen=True)
class SavingsByCategory:
    """
    Savings breakdown by category.

    Attributes:
        load_shifting_savings: From shifting load to off-peak
        demand_reduction_savings: From reducing peak demand
        peak_avoidance_savings: From avoiding peak hours
        efficiency_savings: From efficiency improvements
        thermal_storage_savings: From thermal storage use
        total_savings: Total savings
    """
    load_shifting_savings: float
    demand_reduction_savings: float
    peak_avoidance_savings: float
    efficiency_savings: float
    thermal_storage_savings: float
    total_savings: float


@dataclass(frozen=True)
class ROIMetrics:
    """
    Return on investment metrics.

    Attributes:
        simple_payback_years: Simple payback period
        npv: Net present value
        irr: Internal rate of return
        roi_percentage: ROI percentage
        benefit_cost_ratio: Benefit-to-cost ratio
        annual_net_savings: Net annual savings after costs
    """
    simple_payback_years: float
    npv: float
    irr: float
    roi_percentage: float
    benefit_cost_ratio: float
    annual_net_savings: float


@dataclass(frozen=True)
class SavingsCalculatorOutput:
    """
    Output from savings calculations.

    Attributes:
        period_savings: Savings for analysis period
        annual_savings_projection: Projected annual savings
        savings_by_category: Breakdown by savings category
        roi_metrics: ROI calculations
        baseline_cost: Total baseline cost
        optimized_cost: Total optimized cost
        savings_percentage: Percentage savings
        energy_savings_kwh: Energy saved (kWh)
        demand_savings_kw: Peak demand reduction (kW)
        verification_confidence: Verification confidence level
        monthly_savings: Month-by-month projection
    """
    period_savings: float
    annual_savings_projection: float
    savings_by_category: SavingsByCategory
    roi_metrics: Optional[ROIMetrics]
    baseline_cost: float
    optimized_cost: float
    savings_percentage: float
    energy_savings_kwh: float
    demand_savings_kw: float
    verification_confidence: str
    monthly_savings: List[float]


# =============================================================================
# SAVINGS CALCULATOR CLASS
# =============================================================================

class SavingsCalculator:
    """
    Zero-hallucination savings calculator for heating schedules.

    Implements deterministic calculations for cost savings analysis
    following IPMVP protocols. All calculations produce bit-perfect
    reproducible results with complete provenance tracking.

    Features:
    - Baseline vs optimized comparison
    - Category-wise savings breakdown
    - ROI and payback calculations
    - Annual savings projections
    - IPMVP verification support

    Guarantees:
    - DETERMINISTIC: Same input always produces same output
    - REPRODUCIBLE: SHA-256 verified calculation chain
    - AUDITABLE: Complete step-by-step provenance trail
    - ZERO HALLUCINATION: No LLM in calculation path

    Example:
        >>> calculator = SavingsCalculator()
        >>> baseline = [HourlyScheduleData(h, 100, 50, 0.15) for h in range(24)]
        >>> optimized = [HourlyScheduleData(h, 90, 45, 0.15) for h in range(24)]
        >>> comparison = ScheduleComparison(baseline, optimized)
        >>> inputs = SavingsCalculatorInput(comparison)
        >>> result, provenance = calculator.calculate(inputs)
        >>> print(f"Annual Savings: ${result.annual_savings_projection:.2f}")
    """

    VERSION = "1.0.0"
    NAME = "SavingsCalculator"

    def __init__(self):
        """Initialize the savings calculator."""
        self._tracker: Optional[ProvenanceTracker] = None
        self._step_counter = 0

    def calculate(
        self,
        inputs: SavingsCalculatorInput
    ) -> Tuple[SavingsCalculatorOutput, ProvenanceRecord]:
        """
        Calculate savings from schedule optimization.

        Args:
            inputs: SavingsCalculatorInput with schedules and costs

        Returns:
            Tuple of (SavingsCalculatorOutput, ProvenanceRecord)

        Raises:
            ValueError: If inputs are invalid
        """
        # Initialize provenance tracking
        self._tracker = ProvenanceTracker(
            calculator_name=self.NAME,
            calculator_version=self.VERSION,
            metadata={
                "standards": ["ISO 50001", "ISO 50006", "ASHRAE Guideline 14", "IPMVP"],
                "domain": "Energy Cost Savings Analysis"
            }
        )
        self._step_counter = 0

        # Prepare inputs for provenance
        comparison = inputs.schedule_comparison
        input_dict = {
            "baseline_hours": len(comparison.baseline_schedule),
            "optimized_hours": len(comparison.optimized_schedule),
            "analysis_period_days": comparison.analysis_period_days,
            "projection_years": inputs.projection_years,
            "discount_rate": inputs.discount_rate,
            "verification_method": inputs.verification_method.value,
            "has_project_costs": inputs.project_costs is not None
        }
        self._tracker.set_inputs(input_dict)

        # Validate inputs
        self._validate_inputs(inputs)

        # Calculate baseline costs
        baseline_cost = self._calculate_schedule_cost(
            comparison.baseline_schedule,
            "baseline"
        )

        # Calculate optimized costs
        optimized_cost = self._calculate_schedule_cost(
            comparison.optimized_schedule,
            "optimized"
        )

        # Calculate period savings
        period_savings = baseline_cost - optimized_cost
        self._add_step(
            "Calculate period savings",
            "subtract",
            {"baseline_cost": baseline_cost, "optimized_cost": optimized_cost},
            period_savings,
            "period_savings",
            "Savings = Baseline - Optimized"
        )

        # Calculate savings percentage
        savings_percentage = (period_savings / baseline_cost * 100) if baseline_cost > 0 else 0.0
        self._add_step(
            "Calculate savings percentage",
            "percentage",
            {"savings": period_savings, "baseline": baseline_cost},
            savings_percentage,
            "savings_percentage",
            "Percentage = (Savings / Baseline) * 100"
        )

        # Calculate energy and demand savings
        energy_savings = self._calculate_energy_savings(comparison)
        demand_savings = self._calculate_demand_savings(comparison)

        # Calculate savings by category
        savings_by_category = self._calculate_savings_by_category(
            comparison,
            period_savings
        )

        # Project annual savings
        annual_savings = self._project_annual_savings(
            period_savings,
            comparison.analysis_period_days
        )

        # Calculate ROI metrics
        roi_metrics = None
        if inputs.project_costs:
            roi_metrics = self._calculate_roi_metrics(
                annual_savings,
                inputs.project_costs,
                inputs.projection_years,
                inputs.discount_rate
            )

        # Calculate monthly savings
        monthly_savings = self._calculate_monthly_savings(
            annual_savings,
            inputs.projection_years
        )

        # Determine verification confidence
        verification_confidence = self._get_verification_confidence(
            inputs.verification_method,
            len(comparison.baseline_schedule)
        )

        # Create output
        output = SavingsCalculatorOutput(
            period_savings=round(period_savings, CURRENCY_PRECISION),
            annual_savings_projection=round(annual_savings, CURRENCY_PRECISION),
            savings_by_category=savings_by_category,
            roi_metrics=roi_metrics,
            baseline_cost=round(baseline_cost, CURRENCY_PRECISION),
            optimized_cost=round(optimized_cost, CURRENCY_PRECISION),
            savings_percentage=round(savings_percentage, 2),
            energy_savings_kwh=round(energy_savings, 2),
            demand_savings_kw=round(demand_savings, 2),
            verification_confidence=verification_confidence,
            monthly_savings=[round(m, CURRENCY_PRECISION) for m in monthly_savings]
        )

        # Set outputs and finalize provenance
        self._tracker.set_outputs({
            "period_savings": output.period_savings,
            "annual_savings_projection": output.annual_savings_projection,
            "savings_percentage": output.savings_percentage,
            "energy_savings_kwh": output.energy_savings_kwh,
            "demand_savings_kw": output.demand_savings_kw,
            "baseline_cost": output.baseline_cost,
            "optimized_cost": output.optimized_cost
        })

        provenance = self._tracker.finalize()
        return output, provenance

    def _add_step(
        self,
        description: str,
        operation: str,
        inputs: Dict[str, Any],
        output_value: Union[float, str, Dict, List],
        output_name: str,
        formula: str = ""
    ) -> None:
        """Add a calculation step to provenance tracking."""
        self._step_counter += 1
        self._tracker.add_step(
            step_number=self._step_counter,
            description=description,
            operation=operation,
            inputs=inputs,
            output_value=output_value,
            output_name=output_name,
            formula=formula
        )

    def _validate_inputs(self, inputs: SavingsCalculatorInput) -> None:
        """Validate input parameters."""
        comparison = inputs.schedule_comparison

        if not comparison.baseline_schedule:
            raise ValueError("Baseline schedule required")

        if not comparison.optimized_schedule:
            raise ValueError("Optimized schedule required")

        if len(comparison.baseline_schedule) != len(comparison.optimized_schedule):
            raise ValueError("Baseline and optimized schedules must have same length")

        if comparison.analysis_period_days <= 0:
            raise ValueError("Analysis period must be positive")

        if inputs.discount_rate < 0 or inputs.discount_rate > 1:
            raise ValueError("Discount rate must be between 0 and 1")

    def _calculate_schedule_cost(
        self,
        schedule: List[HourlyScheduleData],
        schedule_name: str
    ) -> float:
        """
        Calculate total cost for a schedule.

        Args:
            schedule: Hourly schedule data
            schedule_name: Name for tracking

        Returns:
            Total cost
        """
        energy_cost = sum(s.energy_kwh * s.energy_rate for s in schedule)

        # Demand cost based on peak demand
        peak_demand = max(s.demand_kw for s in schedule) if schedule else 0.0
        avg_demand_rate = statistics.mean(s.demand_rate for s in schedule) if schedule else 0.0
        demand_cost = peak_demand * avg_demand_rate

        total_cost = energy_cost + demand_cost

        self._add_step(
            f"Calculate {schedule_name} schedule cost",
            "cost_sum",
            {
                "energy_cost": energy_cost,
                "demand_cost": demand_cost,
                "peak_demand_kw": peak_demand
            },
            total_cost,
            f"{schedule_name}_total_cost",
            "Total = Energy Cost + Demand Cost"
        )

        return total_cost

    def _calculate_energy_savings(
        self,
        comparison: ScheduleComparison
    ) -> float:
        """
        Calculate energy savings (kWh).

        Args:
            comparison: Schedule comparison

        Returns:
            Energy savings (kWh)
        """
        baseline_energy = sum(s.energy_kwh for s in comparison.baseline_schedule)
        optimized_energy = sum(s.energy_kwh for s in comparison.optimized_schedule)

        energy_savings = baseline_energy - optimized_energy

        self._add_step(
            "Calculate energy savings",
            "subtract",
            {"baseline_energy_kwh": baseline_energy, "optimized_energy_kwh": optimized_energy},
            energy_savings,
            "energy_savings_kwh",
            "Energy Savings = Baseline - Optimized"
        )

        return energy_savings

    def _calculate_demand_savings(
        self,
        comparison: ScheduleComparison
    ) -> float:
        """
        Calculate peak demand reduction (kW).

        Args:
            comparison: Schedule comparison

        Returns:
            Peak demand reduction (kW)
        """
        baseline_peak = max(s.demand_kw for s in comparison.baseline_schedule)
        optimized_peak = max(s.demand_kw for s in comparison.optimized_schedule)

        demand_savings = baseline_peak - optimized_peak

        self._add_step(
            "Calculate demand savings",
            "subtract",
            {"baseline_peak_kw": baseline_peak, "optimized_peak_kw": optimized_peak},
            demand_savings,
            "demand_savings_kw",
            "Demand Savings = Baseline Peak - Optimized Peak"
        )

        return demand_savings

    def _calculate_savings_by_category(
        self,
        comparison: ScheduleComparison,
        total_savings: float
    ) -> SavingsByCategory:
        """
        Break down savings by category.

        Args:
            comparison: Schedule comparison
            total_savings: Total savings amount

        Returns:
            Savings breakdown by category
        """
        # Calculate load shifting savings (from moving load to lower-rate hours)
        baseline_peak_cost = sum(
            s.energy_kwh * s.energy_rate
            for s in comparison.baseline_schedule
            if s.is_peak_hour
        )
        optimized_peak_cost = sum(
            s.energy_kwh * s.energy_rate
            for s in comparison.optimized_schedule
            if s.is_peak_hour
        )
        load_shifting_savings = baseline_peak_cost - optimized_peak_cost

        # Calculate demand reduction savings
        baseline_demand_cost = max(s.demand_kw * s.demand_rate for s in comparison.baseline_schedule)
        optimized_demand_cost = max(s.demand_kw * s.demand_rate for s in comparison.optimized_schedule)
        demand_reduction_savings = baseline_demand_cost - optimized_demand_cost

        # Peak avoidance savings (from avoiding peak periods entirely)
        baseline_peak_hours = sum(1 for s in comparison.baseline_schedule if s.is_peak_hour and s.energy_kwh > 0)
        optimized_peak_hours = sum(1 for s in comparison.optimized_schedule if s.is_peak_hour and s.energy_kwh > 0)
        peak_hours_avoided = baseline_peak_hours - optimized_peak_hours
        avg_peak_cost = baseline_peak_cost / max(baseline_peak_hours, 1)
        peak_avoidance_savings = peak_hours_avoided * avg_peak_cost * 0.5

        # Efficiency savings (remaining savings)
        accounted_savings = load_shifting_savings + demand_reduction_savings + peak_avoidance_savings
        efficiency_savings = max(0, total_savings - accounted_savings)

        # Thermal storage savings (placeholder - would require storage data)
        thermal_storage_savings = 0.0

        self._add_step(
            "Calculate savings by category",
            "category_breakdown",
            {
                "load_shifting": load_shifting_savings,
                "demand_reduction": demand_reduction_savings,
                "peak_avoidance": peak_avoidance_savings
            },
            total_savings,
            "savings_breakdown",
            "Breakdown by: Load Shifting, Demand, Peak Avoidance, Efficiency"
        )

        return SavingsByCategory(
            load_shifting_savings=round(max(0, load_shifting_savings), CURRENCY_PRECISION),
            demand_reduction_savings=round(max(0, demand_reduction_savings), CURRENCY_PRECISION),
            peak_avoidance_savings=round(max(0, peak_avoidance_savings), CURRENCY_PRECISION),
            efficiency_savings=round(efficiency_savings, CURRENCY_PRECISION),
            thermal_storage_savings=round(thermal_storage_savings, CURRENCY_PRECISION),
            total_savings=round(total_savings, CURRENCY_PRECISION)
        )

    def _project_annual_savings(
        self,
        period_savings: float,
        analysis_period_days: int
    ) -> float:
        """
        Project savings to annual basis.

        Args:
            period_savings: Savings in analysis period
            analysis_period_days: Number of days in period

        Returns:
            Projected annual savings
        """
        daily_savings = period_savings / analysis_period_days
        annual_savings = daily_savings * 365

        self._add_step(
            "Project annual savings",
            "annualize",
            {"period_savings": period_savings, "period_days": analysis_period_days},
            annual_savings,
            "annual_savings",
            f"Annual = (Period Savings / {analysis_period_days}) * 365"
        )

        return annual_savings

    def _calculate_roi_metrics(
        self,
        annual_savings: float,
        costs: ProjectCosts,
        years: int,
        discount_rate: float
    ) -> ROIMetrics:
        """
        Calculate ROI metrics.

        Args:
            annual_savings: Annual savings amount
            costs: Project costs
            years: Project lifetime
            discount_rate: Discount rate for NPV

        Returns:
            ROI metrics
        """
        # Total initial investment
        initial_investment = (
            costs.capital_cost +
            costs.implementation_cost +
            costs.training_cost
        )

        # Annual costs
        annual_costs = (
            costs.annual_maintenance_cost +
            costs.software_license_annual
        )

        # Net annual savings
        net_annual_savings = annual_savings - annual_costs

        # Simple payback
        if net_annual_savings > 0:
            simple_payback = initial_investment / net_annual_savings
        else:
            simple_payback = float('inf')

        self._add_step(
            "Calculate simple payback",
            "divide",
            {"initial_investment": initial_investment, "net_annual_savings": net_annual_savings},
            simple_payback,
            "simple_payback_years",
            "Payback = Initial Investment / Net Annual Savings"
        )

        # NPV calculation
        npv = -initial_investment
        for year in range(1, years + 1):
            npv += net_annual_savings / ((1 + discount_rate) ** year)

        self._add_step(
            "Calculate NPV",
            "npv_sum",
            {"initial_investment": initial_investment, "discount_rate": discount_rate, "years": years},
            npv,
            "npv",
            "NPV = -Investment + sum(Annual Savings / (1+r)^t)"
        )

        # ROI percentage
        total_savings_over_life = net_annual_savings * years
        if initial_investment > 0:
            roi_percentage = (total_savings_over_life - initial_investment) / initial_investment * 100
        else:
            roi_percentage = float('inf') if net_annual_savings > 0 else 0.0

        # Benefit-cost ratio
        if initial_investment > 0:
            benefit_cost_ratio = total_savings_over_life / initial_investment
        else:
            benefit_cost_ratio = float('inf') if total_savings_over_life > 0 else 0.0

        # IRR calculation (simplified - Newton-Raphson)
        irr = self._calculate_irr(initial_investment, net_annual_savings, years)

        self._add_step(
            "Calculate ROI metrics",
            "roi_calculation",
            {
                "total_savings": total_savings_over_life,
                "initial_investment": initial_investment
            },
            roi_percentage,
            "roi_percentage",
            "ROI = (Total Savings - Investment) / Investment * 100"
        )

        return ROIMetrics(
            simple_payback_years=round(simple_payback, 2),
            npv=round(npv, CURRENCY_PRECISION),
            irr=round(irr, 4),
            roi_percentage=round(roi_percentage, 2),
            benefit_cost_ratio=round(benefit_cost_ratio, 2),
            annual_net_savings=round(net_annual_savings, CURRENCY_PRECISION)
        )

    def _calculate_irr(
        self,
        initial_investment: float,
        annual_cash_flow: float,
        years: int,
        max_iterations: int = 100,
        tolerance: float = 0.0001
    ) -> float:
        """
        Calculate Internal Rate of Return using Newton-Raphson method.

        Args:
            initial_investment: Initial investment
            annual_cash_flow: Annual cash flow
            years: Number of years
            max_iterations: Maximum iterations
            tolerance: Convergence tolerance

        Returns:
            IRR as decimal (0.15 = 15%)
        """
        if initial_investment <= 0 or annual_cash_flow <= 0:
            return 0.0

        # Initial guess
        rate = 0.1

        for _ in range(max_iterations):
            # Calculate NPV at current rate
            npv = -initial_investment
            npv_derivative = 0.0

            for t in range(1, years + 1):
                factor = (1 + rate) ** t
                npv += annual_cash_flow / factor
                npv_derivative -= t * annual_cash_flow / (factor * (1 + rate))

            if abs(npv_derivative) < 1e-10:
                break

            # Newton-Raphson step
            new_rate = rate - npv / npv_derivative

            if abs(new_rate - rate) < tolerance:
                return new_rate

            rate = new_rate

        return rate

    def _calculate_monthly_savings(
        self,
        annual_savings: float,
        years: int
    ) -> List[float]:
        """
        Calculate monthly savings projection.

        Args:
            annual_savings: Annual savings
            years: Number of years

        Returns:
            List of monthly savings
        """
        monthly_savings = annual_savings / 12
        total_months = years * 12

        return [monthly_savings] * total_months

    def _get_verification_confidence(
        self,
        method: VerificationMethod,
        data_points: int
    ) -> str:
        """
        Determine verification confidence level.

        Args:
            method: IPMVP verification method
            data_points: Number of data points

        Returns:
            Confidence level description
        """
        confidence_map = {
            VerificationMethod.OPTION_A: "Moderate (Key Parameters Only)",
            VerificationMethod.OPTION_B: "High (Full Measurement)",
            VerificationMethod.OPTION_C: "Moderate (Utility Analysis)",
            VerificationMethod.OPTION_D: "High (Calibrated Simulation)"
        }

        base_confidence = confidence_map.get(method, "Unknown")

        # Adjust based on data points
        if data_points < 168:  # Less than 1 week
            return f"{base_confidence} - Limited Data"
        elif data_points < 720:  # Less than 1 month
            return base_confidence
        else:
            return f"{base_confidence} - Strong Data"


# =============================================================================
# STANDALONE SAVINGS FUNCTIONS
# =============================================================================

def calculate_simple_payback(
    initial_investment: float,
    annual_savings: float
) -> float:
    """
    Calculate simple payback period.

    Formula:
        Payback = Initial Investment / Annual Savings

    Args:
        initial_investment: Initial investment cost
        annual_savings: Annual savings amount

    Returns:
        Payback period in years

    Example:
        >>> payback = calculate_simple_payback(50000, 12000)
        >>> print(f"Payback: {payback:.1f} years")  # 4.2 years
    """
    if annual_savings <= 0:
        return float('inf')
    return initial_investment / annual_savings


def calculate_npv(
    initial_investment: float,
    annual_cash_flows: List[float],
    discount_rate: float
) -> float:
    """
    Calculate Net Present Value.

    Formula:
        NPV = -Investment + sum(CF_t / (1+r)^t)

    Args:
        initial_investment: Initial investment
        annual_cash_flows: List of annual cash flows
        discount_rate: Discount rate (e.g., 0.08 for 8%)

    Returns:
        Net Present Value

    Example:
        >>> cash_flows = [10000, 10000, 10000, 10000, 10000]
        >>> npv = calculate_npv(30000, cash_flows, 0.08)
        >>> print(f"NPV: ${npv:,.2f}")
    """
    npv = -initial_investment

    for t, cash_flow in enumerate(annual_cash_flows, start=1):
        npv += cash_flow / ((1 + discount_rate) ** t)

    return npv


def calculate_levelized_cost(
    total_cost: float,
    total_energy_kwh: float
) -> float:
    """
    Calculate levelized cost of energy.

    Formula:
        LCOE = Total Cost / Total Energy

    Args:
        total_cost: Total lifecycle cost
        total_energy_kwh: Total energy produced/saved

    Returns:
        Levelized cost (currency/kWh)

    Example:
        >>> lcoe = calculate_levelized_cost(100000, 500000)
        >>> print(f"LCOE: ${lcoe:.4f}/kWh")  # $0.2000/kWh
    """
    if total_energy_kwh <= 0:
        return 0.0
    return total_cost / total_energy_kwh


def calculate_savings_percentage(
    baseline_cost: float,
    optimized_cost: float
) -> float:
    """
    Calculate savings as percentage of baseline.

    Formula:
        Savings% = (Baseline - Optimized) / Baseline * 100

    Args:
        baseline_cost: Baseline cost
        optimized_cost: Optimized cost

    Returns:
        Savings percentage

    Example:
        >>> pct = calculate_savings_percentage(1000, 800)
        >>> print(f"Savings: {pct:.1f}%")  # 20.0%
    """
    if baseline_cost <= 0:
        return 0.0
    return (baseline_cost - optimized_cost) / baseline_cost * 100


def annualize_savings(
    period_savings: float,
    period_days: int
) -> float:
    """
    Annualize savings from a measurement period.

    Formula:
        Annual = Period_Savings * (365 / Period_Days)

    Args:
        period_savings: Savings during measurement period
        period_days: Number of days in period

    Returns:
        Annualized savings

    Example:
        >>> annual = annualize_savings(5000, 30)
        >>> print(f"Annual: ${annual:,.2f}")  # ~$60,833
    """
    if period_days <= 0:
        return 0.0
    return period_savings * (365 / period_days)


def calculate_demand_charge_savings(
    baseline_peak_kw: float,
    optimized_peak_kw: float,
    demand_rate_per_kw: float
) -> float:
    """
    Calculate savings from peak demand reduction.

    Formula:
        Savings = (Baseline_Peak - Optimized_Peak) * Demand_Rate

    Args:
        baseline_peak_kw: Baseline peak demand (kW)
        optimized_peak_kw: Optimized peak demand (kW)
        demand_rate_per_kw: Demand charge rate (currency/kW)

    Returns:
        Demand charge savings

    Example:
        >>> savings = calculate_demand_charge_savings(500, 400, 15)
        >>> print(f"Monthly Savings: ${savings:.2f}")  # $1,500.00
    """
    demand_reduction = baseline_peak_kw - optimized_peak_kw
    return demand_reduction * demand_rate_per_kw


def calculate_load_shift_savings(
    shifted_energy_kwh: float,
    peak_rate: float,
    off_peak_rate: float
) -> float:
    """
    Calculate savings from load shifting.

    Formula:
        Savings = Shifted_Energy * (Peak_Rate - Off_Peak_Rate)

    Args:
        shifted_energy_kwh: Energy shifted from peak to off-peak
        peak_rate: Peak period rate (currency/kWh)
        off_peak_rate: Off-peak rate (currency/kWh)

    Returns:
        Load shifting savings

    Example:
        >>> savings = calculate_load_shift_savings(1000, 0.25, 0.08)
        >>> print(f"Savings: ${savings:.2f}")  # $170.00
    """
    rate_differential = peak_rate - off_peak_rate
    return shifted_energy_kwh * rate_differential


def estimate_annual_savings_range(
    measured_savings: float,
    measurement_uncertainty: float
) -> Tuple[float, float]:
    """
    Estimate savings range accounting for uncertainty.

    Args:
        measured_savings: Measured savings amount
        measurement_uncertainty: Uncertainty percentage (e.g., 0.10 for 10%)

    Returns:
        Tuple of (low_estimate, high_estimate)

    Example:
        >>> low, high = estimate_annual_savings_range(10000, 0.15)
        >>> print(f"Range: ${low:,.2f} - ${high:,.2f}")
    """
    margin = measured_savings * measurement_uncertainty
    return (measured_savings - margin, measured_savings + margin)
