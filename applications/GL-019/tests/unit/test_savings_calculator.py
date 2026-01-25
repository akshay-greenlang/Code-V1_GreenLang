"""
GL-019 HEATSCHEDULER - Savings Calculator Unit Tests

Comprehensive unit tests for SavingsCalculator with 95%+ coverage target.
Tests baseline vs optimized comparison, savings breakdown, and ROI projections.

Target Coverage: 95%+
Author: GL-TestEngineer
Version: 1.0.0
"""

import pytest
import sys
import os
from datetime import datetime, timedelta, timezone, date
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import hashlib
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import test fixtures from conftest
from tests.conftest import (
    EnergyTariff, RateType, PeriodType, TimePeriod,
    HeatingEquipment, ProductionJob, ScheduleSlot, OptimizedSchedule
)


# =============================================================================
# DATA CLASSES FOR SAVINGS CALCULATIONS
# =============================================================================

@dataclass
class SavingsBreakdown:
    """Detailed breakdown of savings by category."""
    period_shift_savings: float  # Savings from shifting to lower-rate periods
    demand_reduction_savings: float  # Savings from reducing peak demand
    efficiency_savings: float  # Savings from improved efficiency
    demand_response_savings: float  # Savings from demand response participation
    total_savings: float
    savings_by_period: Dict[str, float] = field(default_factory=dict)


@dataclass
class AnnualProjection:
    """Annual savings projection."""
    annual_savings: float
    annual_energy_cost_baseline: float
    annual_energy_cost_optimized: float
    annual_demand_cost_baseline: float
    annual_demand_cost_optimized: float
    monthly_projections: List[Dict[str, float]] = field(default_factory=list)
    confidence_interval: Tuple[float, float] = (0.0, 0.0)


@dataclass
class ROIAnalysis:
    """Return on Investment analysis."""
    implementation_cost: float
    annual_savings: float
    simple_payback_years: float
    roi_pct: float
    npv_10_year: float
    irr_pct: float
    break_even_date: Optional[date] = None


@dataclass
class ProvenanceRecord:
    """Provenance tracking for calculations."""
    calculator_name: str
    calculator_version: str
    provenance_hash: str
    calculation_steps: List[dict]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# =============================================================================
# MOCK CALCULATOR CLASS FOR TESTING
# =============================================================================

class SavingsCalculator:
    """
    Savings calculator for process heating scheduling.

    Calculates and projects savings from optimized scheduling:
    - Baseline vs optimized comparison
    - Savings breakdown by category
    - Annual projections
    - ROI analysis
    """

    VERSION = "1.0.0"
    NAME = "SavingsCalculator"

    # Constants for financial calculations
    DISCOUNT_RATE = 0.08  # 8% annual discount rate
    MONTHS_PER_YEAR = 12
    WORKING_DAYS_PER_MONTH = 22

    def __init__(self, discount_rate: float = 0.08):
        self.discount_rate = discount_rate
        self._tracker = None

    def calculate_savings(
        self,
        baseline_schedule: OptimizedSchedule,
        optimized_schedule: OptimizedSchedule
    ) -> Tuple[SavingsBreakdown, ProvenanceRecord]:
        """
        Calculate savings between baseline and optimized schedules.

        Returns:
            Tuple of (SavingsBreakdown, ProvenanceRecord)
        """
        if baseline_schedule.total_cost < 0:
            raise ValueError("Baseline cost cannot be negative")

        if optimized_schedule.total_cost < 0:
            raise ValueError("Optimized cost cannot be negative")

        # Calculate period shift savings
        baseline_period_costs = self._get_period_costs(baseline_schedule)
        optimized_period_costs = self._get_period_costs(optimized_schedule)

        period_shift_savings = 0.0
        savings_by_period = {}

        for period in PeriodType:
            baseline_cost = baseline_period_costs.get(period.value, 0.0)
            optimized_cost = optimized_period_costs.get(period.value, 0.0)
            period_savings = baseline_cost - optimized_cost
            savings_by_period[period.value] = period_savings
            period_shift_savings += period_savings

        # Calculate demand reduction savings (simplified)
        demand_reduction_savings = self._calculate_demand_savings(
            baseline_schedule, optimized_schedule
        )

        # Calculate efficiency savings (if applicable)
        efficiency_savings = self._calculate_efficiency_savings(
            baseline_schedule, optimized_schedule
        )

        # Demand response savings (if applicable)
        demand_response_savings = 0.0  # Calculated separately in DR events

        total_savings = baseline_schedule.total_cost - optimized_schedule.total_cost

        breakdown = SavingsBreakdown(
            period_shift_savings=period_shift_savings,
            demand_reduction_savings=demand_reduction_savings,
            efficiency_savings=efficiency_savings,
            demand_response_savings=demand_response_savings,
            total_savings=total_savings,
            savings_by_period=savings_by_period
        )

        provenance = self._create_provenance(baseline_schedule, optimized_schedule, breakdown)

        return breakdown, provenance

    def calculate_savings_percentage(
        self,
        baseline_cost: float,
        optimized_cost: float
    ) -> float:
        """Calculate savings as percentage of baseline."""
        if baseline_cost <= 0:
            if optimized_cost <= 0:
                return 0.0
            raise ValueError("Baseline cost must be positive")

        savings = baseline_cost - optimized_cost
        return (savings / baseline_cost) * 100.0

    def project_annual_savings(
        self,
        daily_savings: float,
        working_days_per_month: int = 22,
        seasonal_factors: Dict[int, float] = None
    ) -> AnnualProjection:
        """
        Project annual savings from daily savings data.

        Args:
            daily_savings: Average daily savings
            working_days_per_month: Working days per month (default 22)
            seasonal_factors: Dict of month (1-12) to seasonal adjustment factor

        Returns:
            AnnualProjection with monthly breakdown
        """
        if daily_savings < 0:
            # Allow negative "savings" (cost increase) but warn
            pass

        if working_days_per_month <= 0 or working_days_per_month > 31:
            raise ValueError("Working days per month must be between 1 and 31")

        if seasonal_factors is None:
            # Default: summer peak (higher savings opportunity)
            seasonal_factors = {
                1: 0.9, 2: 0.9, 3: 0.95, 4: 1.0,
                5: 1.05, 6: 1.15, 7: 1.2, 8: 1.2,
                9: 1.1, 10: 1.0, 11: 0.95, 12: 0.9
            }

        monthly_projections = []
        annual_savings = 0.0

        for month in range(1, 13):
            factor = seasonal_factors.get(month, 1.0)
            monthly_savings = daily_savings * working_days_per_month * factor
            annual_savings += monthly_savings

            monthly_projections.append({
                "month": month,
                "savings": monthly_savings,
                "seasonal_factor": factor,
                "working_days": working_days_per_month
            })

        # Calculate confidence interval (simplified: +/- 15%)
        confidence_low = annual_savings * 0.85
        confidence_high = annual_savings * 1.15

        # Estimate baseline and optimized annual costs
        # Assuming daily savings represents ~15% cost reduction
        daily_cost_baseline = daily_savings / 0.15 if daily_savings > 0 else 1000.0
        daily_cost_optimized = daily_cost_baseline - daily_savings

        annual_cost_baseline = daily_cost_baseline * working_days_per_month * 12
        annual_cost_optimized = daily_cost_optimized * working_days_per_month * 12

        return AnnualProjection(
            annual_savings=annual_savings,
            annual_energy_cost_baseline=annual_cost_baseline * 0.85,  # Energy portion
            annual_energy_cost_optimized=annual_cost_optimized * 0.85,
            annual_demand_cost_baseline=annual_cost_baseline * 0.15,  # Demand portion
            annual_demand_cost_optimized=annual_cost_optimized * 0.15,
            monthly_projections=monthly_projections,
            confidence_interval=(confidence_low, confidence_high)
        )

    def calculate_roi(
        self,
        implementation_cost: float,
        annual_savings: float,
        analysis_period_years: int = 10
    ) -> ROIAnalysis:
        """
        Calculate Return on Investment for scheduling optimization.

        Args:
            implementation_cost: Initial implementation cost
            annual_savings: Expected annual savings
            analysis_period_years: Period for NPV/IRR calculations

        Returns:
            ROIAnalysis with payback, ROI, NPV, and IRR
        """
        if implementation_cost < 0:
            raise ValueError("Implementation cost cannot be negative")

        if annual_savings <= 0:
            # No positive savings - no meaningful ROI
            return ROIAnalysis(
                implementation_cost=implementation_cost,
                annual_savings=annual_savings,
                simple_payback_years=float('inf'),
                roi_pct=0.0,
                npv_10_year=-implementation_cost,
                irr_pct=0.0,
                break_even_date=None
            )

        # Simple payback period
        simple_payback = implementation_cost / annual_savings

        # ROI percentage
        roi_pct = (annual_savings / implementation_cost) * 100.0 if implementation_cost > 0 else float('inf')

        # NPV calculation
        npv = -implementation_cost
        for year in range(1, analysis_period_years + 1):
            npv += annual_savings / ((1 + self.discount_rate) ** year)

        # IRR calculation (simplified Newton-Raphson)
        irr = self._calculate_irr(implementation_cost, annual_savings, analysis_period_years)

        # Break-even date
        if simple_payback <= analysis_period_years:
            today = date.today()
            break_even_date = today + timedelta(days=int(simple_payback * 365))
        else:
            break_even_date = None

        return ROIAnalysis(
            implementation_cost=implementation_cost,
            annual_savings=annual_savings,
            simple_payback_years=simple_payback,
            roi_pct=roi_pct,
            npv_10_year=npv,
            irr_pct=irr,
            break_even_date=break_even_date
        )

    def compare_scenarios(
        self,
        scenarios: Dict[str, Tuple[float, float]]  # scenario_name -> (baseline, optimized)
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare multiple scheduling scenarios.

        Returns comparison table with savings for each scenario.
        """
        if not scenarios:
            raise ValueError("No scenarios provided")

        results = {}

        for name, (baseline, optimized) in scenarios.items():
            savings = baseline - optimized
            savings_pct = self.calculate_savings_percentage(baseline, optimized)

            results[name] = {
                "baseline_cost": baseline,
                "optimized_cost": optimized,
                "savings": savings,
                "savings_pct": savings_pct
            }

        # Rank scenarios by savings
        sorted_scenarios = sorted(results.items(), key=lambda x: x[1]["savings"], reverse=True)
        for rank, (name, _) in enumerate(sorted_scenarios, 1):
            results[name]["rank"] = rank

        return results

    def calculate_demand_response_value(
        self,
        curtailable_load_kw: float,
        event_hours: float,
        incentive_per_kwh: float,
        avoided_energy_cost_per_kwh: float = 0.0
    ) -> Dict[str, float]:
        """
        Calculate value of demand response participation.

        Returns value breakdown for DR event.
        """
        if curtailable_load_kw < 0:
            raise ValueError("Curtailable load cannot be negative")

        if event_hours <= 0:
            raise ValueError("Event hours must be positive")

        if incentive_per_kwh < 0:
            raise ValueError("Incentive cannot be negative")

        curtailed_energy_kwh = curtailable_load_kw * event_hours

        incentive_value = curtailed_energy_kwh * incentive_per_kwh
        avoided_cost = curtailed_energy_kwh * avoided_energy_cost_per_kwh
        total_value = incentive_value + avoided_cost

        return {
            "curtailed_energy_kwh": curtailed_energy_kwh,
            "incentive_value": incentive_value,
            "avoided_energy_cost": avoided_cost,
            "total_value": total_value,
            "value_per_kw": total_value / curtailable_load_kw if curtailable_load_kw > 0 else 0.0
        }

    def _get_period_costs(self, schedule: OptimizedSchedule) -> Dict[str, float]:
        """Get costs broken down by period type."""
        period_costs = {}
        for slot in schedule.slots:
            period_key = slot.period_type.value
            if period_key not in period_costs:
                period_costs[period_key] = 0.0
            period_costs[period_key] += slot.estimated_cost
        return period_costs

    def _calculate_demand_savings(
        self,
        baseline: OptimizedSchedule,
        optimized: OptimizedSchedule
    ) -> float:
        """Calculate demand charge savings."""
        # Simplified: compare peak power
        baseline_peak = max(s.power_kw for s in baseline.slots) if baseline.slots else 0
        optimized_peak = max(s.power_kw for s in optimized.slots) if optimized.slots else 0

        # Assume $15/kW demand charge
        demand_rate = 15.0
        return (baseline_peak - optimized_peak) * demand_rate

    def _calculate_efficiency_savings(
        self,
        baseline: OptimizedSchedule,
        optimized: OptimizedSchedule
    ) -> float:
        """Calculate efficiency-related savings."""
        # Simplified: assume efficiency gains from better scheduling
        baseline_energy = baseline.total_energy_kwh
        optimized_energy = optimized.total_energy_kwh

        # If optimized uses less energy, there's efficiency savings
        if optimized_energy < baseline_energy:
            # Assume average rate
            avg_rate = baseline.total_cost / baseline_energy if baseline_energy > 0 else 0.10
            return (baseline_energy - optimized_energy) * avg_rate

        return 0.0

    def _calculate_irr(
        self,
        investment: float,
        annual_cashflow: float,
        years: int
    ) -> float:
        """Calculate Internal Rate of Return using Newton-Raphson method."""
        if investment <= 0 or annual_cashflow <= 0:
            return 0.0

        # Initial guess
        irr = 0.1

        for _ in range(100):  # Max iterations
            npv = -investment
            npv_derivative = 0.0

            for year in range(1, years + 1):
                discount = (1 + irr) ** year
                npv += annual_cashflow / discount
                npv_derivative -= year * annual_cashflow / ((1 + irr) ** (year + 1))

            if abs(npv_derivative) < 1e-10:
                break

            new_irr = irr - npv / npv_derivative

            if abs(new_irr - irr) < 1e-6:
                break

            irr = new_irr

        return max(0.0, irr * 100.0)  # Return as percentage

    def _create_provenance(
        self,
        baseline: OptimizedSchedule,
        optimized: OptimizedSchedule,
        breakdown: SavingsBreakdown
    ) -> ProvenanceRecord:
        """Create provenance record for savings calculation."""
        data = {
            "calculator": self.NAME,
            "version": self.VERSION,
            "baseline_cost": baseline.total_cost,
            "optimized_cost": optimized.total_cost,
            "total_savings": breakdown.total_savings
        }
        hash_input = json.dumps(data, sort_keys=True)
        provenance_hash = hashlib.sha256(hash_input.encode()).hexdigest()

        steps = [
            {"step_number": 1, "description": "Validate inputs", "operation": "validation"},
            {"step_number": 2, "description": "Calculate period costs", "operation": "period_breakdown"},
            {"step_number": 3, "description": "Calculate period shift savings", "operation": "shift_savings"},
            {"step_number": 4, "description": "Calculate demand savings", "operation": "demand_savings"},
            {"step_number": 5, "description": "Calculate efficiency savings", "operation": "efficiency_savings"},
            {"step_number": 6, "description": "Compute total savings", "operation": "total"},
        ]

        return ProvenanceRecord(
            calculator_name=self.NAME,
            calculator_version=self.VERSION,
            provenance_hash=provenance_hash,
            calculation_steps=steps
        )


# =============================================================================
# HELPER FUNCTIONS FOR CREATING TEST DATA
# =============================================================================

def create_test_schedule(
    slots: List[Tuple[str, float, float, PeriodType]],
    total_cost: float = None
) -> OptimizedSchedule:
    """
    Create a test schedule from slot definitions.

    Args:
        slots: List of (job_id, energy_kwh, cost, period_type)
        total_cost: Override total cost (otherwise sum of slot costs)
    """
    now = datetime.now(timezone.utc)
    schedule_slots = []

    for i, (job_id, energy, cost, period) in enumerate(slots):
        slot = ScheduleSlot(
            slot_id=f"SLOT-{i}",
            equipment_id="EQUIP-001",
            job_id=job_id,
            start_time=now + timedelta(hours=i),
            end_time=now + timedelta(hours=i + 1),
            power_kw=100.0,
            energy_kwh=energy,
            estimated_cost=cost,
            period_type=period
        )
        schedule_slots.append(slot)

    if total_cost is None:
        total_cost = sum(s.estimated_cost for s in schedule_slots)

    return OptimizedSchedule(
        schedule_id="TEST-SCHEDULE",
        created_at=now,
        slots=schedule_slots,
        total_energy_kwh=sum(s.energy_kwh for s in schedule_slots),
        total_cost=total_cost,
        baseline_cost=total_cost * 1.2,  # Assume 20% savings opportunity
        savings=total_cost * 0.2,
        savings_pct=20.0,
        optimization_time_ms=100.0,
        constraints_satisfied=True,
        provenance_hash="test_hash" * 8
    )


# =============================================================================
# UNIT TESTS
# =============================================================================

@pytest.mark.unit
@pytest.mark.calculator
@pytest.mark.critical
class TestSavingsCalculator:
    """Comprehensive test suite for SavingsCalculator."""

    # =========================================================================
    # INITIALIZATION TESTS
    # =========================================================================

    def test_initialization_default(self):
        """Test SavingsCalculator initializes with default discount rate."""
        calculator = SavingsCalculator()

        assert calculator.VERSION == "1.0.0"
        assert calculator.NAME == "SavingsCalculator"
        assert calculator.discount_rate == 0.08

    def test_initialization_custom_discount_rate(self):
        """Test SavingsCalculator with custom discount rate."""
        calculator = SavingsCalculator(discount_rate=0.10)

        assert calculator.discount_rate == 0.10

    # =========================================================================
    # BASELINE VS OPTIMIZED COMPARISON TESTS
    # =========================================================================

    def test_calculate_savings_basic(self):
        """Test basic savings calculation."""
        calculator = SavingsCalculator()

        baseline = create_test_schedule([
            ("JOB-1", 100.0, 25.0, PeriodType.ON_PEAK),
            ("JOB-2", 100.0, 25.0, PeriodType.ON_PEAK),
        ], total_cost=50.0)

        optimized = create_test_schedule([
            ("JOB-1", 100.0, 12.0, PeriodType.OFF_PEAK),
            ("JOB-2", 100.0, 12.0, PeriodType.OFF_PEAK),
        ], total_cost=24.0)

        baseline.baseline_cost = 50.0
        optimized.baseline_cost = 50.0

        breakdown, provenance = calculator.calculate_savings(baseline, optimized)

        assert breakdown.total_savings == pytest.approx(26.0, rel=0.001)
        assert len(provenance.provenance_hash) == 64

    def test_calculate_savings_by_period(self):
        """Test savings breakdown by period."""
        calculator = SavingsCalculator()

        baseline = create_test_schedule([
            ("JOB-1", 100.0, 30.0, PeriodType.ON_PEAK),
            ("JOB-2", 100.0, 15.0, PeriodType.MID_PEAK),
        ], total_cost=45.0)

        optimized = create_test_schedule([
            ("JOB-1", 100.0, 10.0, PeriodType.OFF_PEAK),
            ("JOB-2", 100.0, 10.0, PeriodType.OFF_PEAK),
        ], total_cost=20.0)

        breakdown, _ = calculator.calculate_savings(baseline, optimized)

        assert PeriodType.ON_PEAK.value in breakdown.savings_by_period
        assert PeriodType.OFF_PEAK.value in breakdown.savings_by_period

    def test_calculate_savings_no_change(self):
        """Test savings when baseline equals optimized."""
        calculator = SavingsCalculator()

        schedule = create_test_schedule([
            ("JOB-1", 100.0, 20.0, PeriodType.MID_PEAK),
        ], total_cost=20.0)

        breakdown, _ = calculator.calculate_savings(schedule, schedule)

        assert breakdown.total_savings == 0.0

    def test_calculate_savings_negative(self):
        """Test when optimized is more expensive (negative savings)."""
        calculator = SavingsCalculator()

        baseline = create_test_schedule([
            ("JOB-1", 100.0, 15.0, PeriodType.OFF_PEAK),
        ], total_cost=15.0)

        optimized = create_test_schedule([
            ("JOB-1", 100.0, 25.0, PeriodType.ON_PEAK),
        ], total_cost=25.0)

        breakdown, _ = calculator.calculate_savings(baseline, optimized)

        assert breakdown.total_savings == -10.0

    def test_calculate_savings_invalid_baseline(self):
        """Test savings calculation with negative baseline cost."""
        calculator = SavingsCalculator()

        baseline = create_test_schedule([], total_cost=-10.0)
        optimized = create_test_schedule([], total_cost=20.0)

        with pytest.raises(ValueError, match="Baseline cost cannot be negative"):
            calculator.calculate_savings(baseline, optimized)

    def test_calculate_savings_invalid_optimized(self):
        """Test savings calculation with negative optimized cost."""
        calculator = SavingsCalculator()

        baseline = create_test_schedule([], total_cost=50.0)
        optimized = create_test_schedule([], total_cost=-10.0)

        with pytest.raises(ValueError, match="Optimized cost cannot be negative"):
            calculator.calculate_savings(baseline, optimized)

    # =========================================================================
    # SAVINGS PERCENTAGE TESTS
    # =========================================================================

    @pytest.mark.parametrize("baseline,optimized,expected_pct", [
        (1000.0, 850.0, 15.0),
        (5000.0, 4000.0, 20.0),
        (10000.0, 7500.0, 25.0),
        (2500.0, 2500.0, 0.0),
        (1000.0, 1100.0, -10.0),  # Negative savings
    ])
    def test_calculate_savings_percentage(self, baseline, optimized, expected_pct):
        """Test savings percentage calculation."""
        calculator = SavingsCalculator()

        pct = calculator.calculate_savings_percentage(baseline, optimized)

        assert pct == pytest.approx(expected_pct, rel=0.001)

    def test_calculate_savings_percentage_zero_baseline(self):
        """Test savings percentage with zero baseline."""
        calculator = SavingsCalculator()

        # Zero baseline with zero optimized = 0%
        pct = calculator.calculate_savings_percentage(0.0, 0.0)
        assert pct == 0.0

        # Zero baseline with positive optimized = error
        with pytest.raises(ValueError, match="Baseline cost must be positive"):
            calculator.calculate_savings_percentage(0.0, 100.0)

    # =========================================================================
    # ANNUAL PROJECTION TESTS
    # =========================================================================

    def test_project_annual_savings_basic(self):
        """Test basic annual savings projection."""
        calculator = SavingsCalculator()

        projection = calculator.project_annual_savings(
            daily_savings=100.0,
            working_days_per_month=22
        )

        # Base annual = 100 * 22 * 12 = 26,400
        # With seasonal factors, will vary slightly
        assert projection.annual_savings > 25000.0
        assert projection.annual_savings < 30000.0
        assert len(projection.monthly_projections) == 12

    def test_project_annual_savings_seasonal(self):
        """Test annual projection with seasonal factors."""
        calculator = SavingsCalculator()

        # Higher summer savings
        seasonal = {
            1: 0.5, 2: 0.5, 3: 0.6, 4: 0.7,
            5: 0.9, 6: 1.2, 7: 1.5, 8: 1.5,
            9: 1.1, 10: 0.8, 11: 0.6, 12: 0.5
        }

        projection = calculator.project_annual_savings(
            daily_savings=100.0,
            working_days_per_month=22,
            seasonal_factors=seasonal
        )

        # Summer months should have higher savings
        july_savings = projection.monthly_projections[6]["savings"]
        january_savings = projection.monthly_projections[0]["savings"]

        assert july_savings > january_savings

    def test_project_annual_savings_confidence_interval(self):
        """Test confidence interval is calculated."""
        calculator = SavingsCalculator()

        projection = calculator.project_annual_savings(
            daily_savings=100.0,
            working_days_per_month=22
        )

        low, high = projection.confidence_interval

        assert low < projection.annual_savings
        assert high > projection.annual_savings
        assert low == pytest.approx(projection.annual_savings * 0.85, rel=0.01)
        assert high == pytest.approx(projection.annual_savings * 1.15, rel=0.01)

    def test_project_annual_savings_invalid_working_days(self):
        """Test rejection of invalid working days."""
        calculator = SavingsCalculator()

        with pytest.raises(ValueError, match="Working days per month"):
            calculator.project_annual_savings(daily_savings=100.0, working_days_per_month=0)

        with pytest.raises(ValueError, match="Working days per month"):
            calculator.project_annual_savings(daily_savings=100.0, working_days_per_month=35)

    def test_project_annual_savings_negative(self):
        """Test projection with negative savings (cost increase)."""
        calculator = SavingsCalculator()

        # Should not raise, but produce negative projection
        projection = calculator.project_annual_savings(
            daily_savings=-50.0,
            working_days_per_month=22
        )

        assert projection.annual_savings < 0

    # =========================================================================
    # ROI CALCULATION TESTS
    # =========================================================================

    def test_calculate_roi_basic(self):
        """Test basic ROI calculation."""
        calculator = SavingsCalculator()

        roi = calculator.calculate_roi(
            implementation_cost=50000.0,
            annual_savings=25000.0,
            analysis_period_years=10
        )

        # Simple payback = 50000 / 25000 = 2 years
        assert roi.simple_payback_years == pytest.approx(2.0, rel=0.01)

        # ROI = 25000 / 50000 * 100 = 50%
        assert roi.roi_pct == pytest.approx(50.0, rel=0.01)

        # NPV should be positive
        assert roi.npv_10_year > 0

        # IRR should be significant
        assert roi.irr_pct > 20.0

    def test_calculate_roi_payback_period(self):
        """Test various payback period scenarios."""
        calculator = SavingsCalculator()

        # Fast payback
        roi_fast = calculator.calculate_roi(
            implementation_cost=10000.0,
            annual_savings=20000.0
        )
        assert roi_fast.simple_payback_years == pytest.approx(0.5, rel=0.01)

        # Slow payback
        roi_slow = calculator.calculate_roi(
            implementation_cost=100000.0,
            annual_savings=10000.0
        )
        assert roi_slow.simple_payback_years == pytest.approx(10.0, rel=0.01)

    def test_calculate_roi_break_even_date(self):
        """Test break-even date calculation."""
        calculator = SavingsCalculator()

        roi = calculator.calculate_roi(
            implementation_cost=30000.0,
            annual_savings=15000.0
        )

        # 2 year payback - break-even date should exist
        assert roi.break_even_date is not None
        assert roi.break_even_date > date.today()

        # Very long payback - no break-even within analysis period
        roi_long = calculator.calculate_roi(
            implementation_cost=200000.0,
            annual_savings=10000.0,
            analysis_period_years=10
        )
        assert roi_long.break_even_date is None  # 20 years > 10 year analysis

    def test_calculate_roi_no_savings(self):
        """Test ROI with zero/negative savings."""
        calculator = SavingsCalculator()

        roi = calculator.calculate_roi(
            implementation_cost=50000.0,
            annual_savings=0.0
        )

        assert roi.simple_payback_years == float('inf')
        assert roi.roi_pct == 0.0
        assert roi.npv_10_year < 0  # Negative NPV

    def test_calculate_roi_negative_implementation(self):
        """Test ROI rejects negative implementation cost."""
        calculator = SavingsCalculator()

        with pytest.raises(ValueError, match="Implementation cost cannot be negative"):
            calculator.calculate_roi(
                implementation_cost=-10000.0,
                annual_savings=5000.0
            )

    def test_calculate_roi_npv_with_discount(self):
        """Test NPV calculation respects discount rate."""
        # Lower discount rate = higher NPV
        calc_low = SavingsCalculator(discount_rate=0.05)
        calc_high = SavingsCalculator(discount_rate=0.12)

        roi_low = calc_low.calculate_roi(50000.0, 15000.0, 10)
        roi_high = calc_high.calculate_roi(50000.0, 15000.0, 10)

        assert roi_low.npv_10_year > roi_high.npv_10_year

    # =========================================================================
    # SCENARIO COMPARISON TESTS
    # =========================================================================

    def test_compare_scenarios_basic(self):
        """Test basic scenario comparison."""
        calculator = SavingsCalculator()

        scenarios = {
            "Scenario A": (1000.0, 850.0),
            "Scenario B": (1000.0, 750.0),
            "Scenario C": (1000.0, 900.0),
        }

        results = calculator.compare_scenarios(scenarios)

        assert len(results) == 3
        assert results["Scenario B"]["rank"] == 1  # Best savings
        assert results["Scenario A"]["rank"] == 2
        assert results["Scenario C"]["rank"] == 3

    def test_compare_scenarios_savings_values(self):
        """Test scenario comparison savings values."""
        calculator = SavingsCalculator()

        scenarios = {
            "High Savings": (2000.0, 1500.0),
            "Low Savings": (2000.0, 1900.0),
        }

        results = calculator.compare_scenarios(scenarios)

        assert results["High Savings"]["savings"] == 500.0
        assert results["Low Savings"]["savings"] == 100.0
        assert results["High Savings"]["savings_pct"] == pytest.approx(25.0, rel=0.01)
        assert results["Low Savings"]["savings_pct"] == pytest.approx(5.0, rel=0.01)

    def test_compare_scenarios_empty(self):
        """Test scenario comparison with empty input."""
        calculator = SavingsCalculator()

        with pytest.raises(ValueError, match="No scenarios provided"):
            calculator.compare_scenarios({})

    def test_compare_scenarios_single(self):
        """Test scenario comparison with single scenario."""
        calculator = SavingsCalculator()

        scenarios = {
            "Only Scenario": (1000.0, 800.0),
        }

        results = calculator.compare_scenarios(scenarios)

        assert len(results) == 1
        assert results["Only Scenario"]["rank"] == 1

    # =========================================================================
    # DEMAND RESPONSE VALUE TESTS
    # =========================================================================

    def test_demand_response_value_basic(self):
        """Test basic demand response value calculation."""
        calculator = SavingsCalculator()

        value = calculator.calculate_demand_response_value(
            curtailable_load_kw=500.0,
            event_hours=4.0,
            incentive_per_kwh=0.50
        )

        # 500 kW * 4 hours = 2000 kWh
        # 2000 * $0.50 = $1000
        assert value["curtailed_energy_kwh"] == 2000.0
        assert value["incentive_value"] == 1000.0
        assert value["total_value"] == 1000.0

    def test_demand_response_value_with_avoided_cost(self):
        """Test demand response with avoided energy cost."""
        calculator = SavingsCalculator()

        value = calculator.calculate_demand_response_value(
            curtailable_load_kw=500.0,
            event_hours=4.0,
            incentive_per_kwh=0.50,
            avoided_energy_cost_per_kwh=0.25  # Also save on energy
        )

        # Incentive: 2000 * 0.50 = $1000
        # Avoided: 2000 * 0.25 = $500
        # Total: $1500
        assert value["incentive_value"] == 1000.0
        assert value["avoided_energy_cost"] == 500.0
        assert value["total_value"] == 1500.0

    def test_demand_response_value_per_kw(self):
        """Test value per kW calculation."""
        calculator = SavingsCalculator()

        value = calculator.calculate_demand_response_value(
            curtailable_load_kw=200.0,
            event_hours=2.0,
            incentive_per_kwh=0.40
        )

        # Total value: 200 * 2 * 0.40 = $160
        # Per kW: 160 / 200 = $0.80
        assert value["value_per_kw"] == pytest.approx(0.80, rel=0.01)

    def test_demand_response_value_invalid_load(self):
        """Test DR value rejects negative load."""
        calculator = SavingsCalculator()

        with pytest.raises(ValueError, match="Curtailable load cannot be negative"):
            calculator.calculate_demand_response_value(
                curtailable_load_kw=-100.0,
                event_hours=2.0,
                incentive_per_kwh=0.50
            )

    def test_demand_response_value_invalid_hours(self):
        """Test DR value rejects non-positive hours."""
        calculator = SavingsCalculator()

        with pytest.raises(ValueError, match="Event hours must be positive"):
            calculator.calculate_demand_response_value(
                curtailable_load_kw=100.0,
                event_hours=0.0,
                incentive_per_kwh=0.50
            )

    def test_demand_response_value_invalid_incentive(self):
        """Test DR value rejects negative incentive."""
        calculator = SavingsCalculator()

        with pytest.raises(ValueError, match="Incentive cannot be negative"):
            calculator.calculate_demand_response_value(
                curtailable_load_kw=100.0,
                event_hours=2.0,
                incentive_per_kwh=-0.10
            )

    # =========================================================================
    # PROVENANCE TESTS
    # =========================================================================

    def test_provenance_determinism(self):
        """Test provenance hash is deterministic."""
        calculator = SavingsCalculator()

        baseline = create_test_schedule([
            ("JOB-1", 100.0, 30.0, PeriodType.ON_PEAK),
        ], total_cost=30.0)

        optimized = create_test_schedule([
            ("JOB-1", 100.0, 15.0, PeriodType.OFF_PEAK),
        ], total_cost=15.0)

        _, prov1 = calculator.calculate_savings(baseline, optimized)
        _, prov2 = calculator.calculate_savings(baseline, optimized)

        assert prov1.provenance_hash == prov2.provenance_hash

    def test_provenance_completeness(self):
        """Test provenance includes all required fields."""
        calculator = SavingsCalculator()

        baseline = create_test_schedule([
            ("JOB-1", 100.0, 30.0, PeriodType.ON_PEAK),
        ], total_cost=30.0)

        optimized = create_test_schedule([
            ("JOB-1", 100.0, 15.0, PeriodType.OFF_PEAK),
        ], total_cost=15.0)

        _, provenance = calculator.calculate_savings(baseline, optimized)

        assert provenance.calculator_name == "SavingsCalculator"
        assert provenance.calculator_version == "1.0.0"
        assert len(provenance.provenance_hash) == 64
        assert len(provenance.calculation_steps) >= 5

    # =========================================================================
    # PERFORMANCE TESTS
    # =========================================================================

    @pytest.mark.performance
    def test_calculation_speed(self):
        """Test savings calculation meets performance target."""
        import time

        calculator = SavingsCalculator()

        # Create larger schedules
        baseline_slots = [
            (f"JOB-{i}", 100.0, 25.0, PeriodType.ON_PEAK)
            for i in range(100)
        ]
        optimized_slots = [
            (f"JOB-{i}", 100.0, 12.0, PeriodType.OFF_PEAK)
            for i in range(100)
        ]

        baseline = create_test_schedule(baseline_slots)
        optimized = create_test_schedule(optimized_slots)

        start = time.time()
        for _ in range(100):
            calculator.calculate_savings(baseline, optimized)
        duration_ms = (time.time() - start) * 1000

        # 100 calculations should take <1 second
        assert duration_ms < 1000

    @pytest.mark.performance
    def test_roi_calculation_speed(self):
        """Test ROI calculation performance."""
        import time

        calculator = SavingsCalculator()

        start = time.time()
        for i in range(1000):
            calculator.calculate_roi(
                implementation_cost=50000.0 + i * 100,
                annual_savings=10000.0 + i * 50,
                analysis_period_years=10
            )
        duration_ms = (time.time() - start) * 1000

        # 1000 calculations should take <2 seconds
        assert duration_ms < 2000


# =============================================================================
# EDGE CASE TESTS
# =============================================================================

@pytest.mark.unit
class TestSavingsEdgeCases:
    """Edge case tests for savings calculations."""

    def test_very_small_savings(self):
        """Test calculation with very small savings."""
        calculator = SavingsCalculator()

        baseline = create_test_schedule([
            ("JOB-1", 100.0, 100.001, PeriodType.ON_PEAK),
        ], total_cost=100.001)

        optimized = create_test_schedule([
            ("JOB-1", 100.0, 100.0, PeriodType.OFF_PEAK),
        ], total_cost=100.0)

        breakdown, _ = calculator.calculate_savings(baseline, optimized)

        assert breakdown.total_savings == pytest.approx(0.001, rel=0.01)

    def test_very_large_costs(self):
        """Test calculation with very large costs."""
        calculator = SavingsCalculator()

        pct = calculator.calculate_savings_percentage(
            baseline=1000000000.0,  # $1 billion
            optimized=850000000.0
        )

        assert pct == pytest.approx(15.0, rel=0.001)

    def test_project_very_small_daily_savings(self):
        """Test projection with very small daily savings."""
        calculator = SavingsCalculator()

        projection = calculator.project_annual_savings(
            daily_savings=0.01,  # 1 cent per day
            working_days_per_month=22
        )

        assert projection.annual_savings > 0
        assert projection.annual_savings < 5  # < $5/year

    def test_roi_zero_implementation_cost(self):
        """Test ROI with zero implementation cost."""
        calculator = SavingsCalculator()

        roi = calculator.calculate_roi(
            implementation_cost=0.0,
            annual_savings=10000.0
        )

        # Infinite ROI
        assert roi.roi_pct == float('inf')
        assert roi.simple_payback_years == 0.0
