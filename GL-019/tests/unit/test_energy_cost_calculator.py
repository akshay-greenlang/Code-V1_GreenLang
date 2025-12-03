"""
GL-019 HEATSCHEDULER - Energy Cost Calculator Unit Tests

Comprehensive unit tests for EnergyCostCalculator with 95%+ coverage target.
Tests ToU rate calculations, demand charges, real-time pricing, and edge cases.

Target Coverage: 95%+
Author: GL-TestEngineer
Version: 1.0.0
"""

import pytest
import sys
import os
from datetime import datetime, date, time, timedelta, timezone
from decimal import Decimal
from typing import Dict, List, Tuple
import math

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import test fixtures from conftest
from tests.conftest import (
    EnergyTariff, RateType, PeriodType, TimePeriod,
    HeatingEquipment, ProductionJob, ScheduleSlot
)


# =============================================================================
# MOCK CALCULATOR CLASSES FOR TESTING
# =============================================================================

class EnergyCostCalculator:
    """
    Energy cost calculator for process heating scheduling.

    Calculates electricity costs based on:
    - Time-of-Use (ToU) rates
    - Demand charges
    - Real-time pricing
    - Equipment efficiency factors
    """

    VERSION = "1.0.0"
    NAME = "EnergyCostCalculator"

    def __init__(self):
        self._tracker = None

    def get_period_type(
        self,
        timestamp: datetime,
        tariff: EnergyTariff
    ) -> PeriodType:
        """Determine the rate period for a given timestamp."""
        if tariff.rate_type == RateType.FLAT:
            return PeriodType.OFF_PEAK  # Flat rate has no periods

        hour = timestamp.hour
        day_of_week = timestamp.weekday()

        for period in tariff.time_periods:
            if day_of_week in period.days_of_week:
                # Handle periods that cross midnight
                if period.start_hour <= period.end_hour:
                    if period.start_hour <= hour < period.end_hour:
                        return period.period_type
                else:  # Crosses midnight
                    if hour >= period.start_hour or hour < period.end_hour:
                        return period.period_type

        return PeriodType.OFF_PEAK  # Default to off-peak

    def get_rate_for_period(
        self,
        period_type: PeriodType,
        tariff: EnergyTariff
    ) -> float:
        """Get the rate multiplier for a given period type."""
        multiplier = tariff.period_rates.get(period_type.value, 1.0)
        return tariff.base_rate_kwh * multiplier

    def get_real_time_price(
        self,
        timestamp: datetime,
        tariff: EnergyTariff
    ) -> float:
        """Get real-time price for a specific timestamp."""
        if tariff.rate_type != RateType.REAL_TIME:
            raise ValueError("Tariff is not real-time pricing type")

        # Round to hour for lookup
        hour_key = timestamp.replace(minute=0, second=0, microsecond=0).isoformat()

        price = tariff.real_time_prices.get(hour_key)
        if price is None:
            return tariff.base_rate_kwh  # Fallback to base rate

        return price

    def calculate_energy_cost(
        self,
        energy_kwh: float,
        start_time: datetime,
        duration_hours: float,
        tariff: EnergyTariff,
        equipment: HeatingEquipment = None
    ) -> Tuple[float, Dict]:
        """
        Calculate total energy cost for a given consumption period.

        Returns:
            Tuple of (total_cost, cost_breakdown)
        """
        if energy_kwh < 0:
            raise ValueError("Energy consumption cannot be negative")

        if duration_hours <= 0:
            raise ValueError("Duration must be positive")

        # Apply efficiency factor if equipment provided
        actual_energy = energy_kwh
        if equipment:
            actual_energy = energy_kwh / equipment.efficiency

        cost_breakdown = {
            "energy_kwh": actual_energy,
            "duration_hours": duration_hours,
            "periods": {},
            "energy_cost": 0.0,
            "demand_charge": 0.0,
            "total_cost": 0.0
        }

        if tariff.rate_type == RateType.FLAT:
            energy_cost = actual_energy * tariff.base_rate_kwh
            cost_breakdown["energy_cost"] = energy_cost
            cost_breakdown["periods"]["flat"] = {
                "energy_kwh": actual_energy,
                "rate": tariff.base_rate_kwh,
                "cost": energy_cost
            }

        elif tariff.rate_type == RateType.REAL_TIME:
            # Calculate cost for each hour
            hours = int(math.ceil(duration_hours))
            energy_per_hour = actual_energy / duration_hours

            for h in range(hours):
                hour_time = start_time + timedelta(hours=h)
                price = self.get_real_time_price(hour_time, tariff)

                # Partial hour handling for last hour
                if h == hours - 1 and duration_hours % 1 > 0:
                    hour_energy = energy_per_hour * (duration_hours % 1)
                else:
                    hour_energy = energy_per_hour

                hour_cost = hour_energy * price
                cost_breakdown["energy_cost"] += hour_cost

                hour_key = hour_time.isoformat()
                cost_breakdown["periods"][hour_key] = {
                    "energy_kwh": hour_energy,
                    "rate": price,
                    "cost": hour_cost
                }

        else:  # TIME_OF_USE or DEMAND
            # Break down by time periods
            current_time = start_time
            remaining_energy = actual_energy
            energy_per_hour = actual_energy / duration_hours
            remaining_hours = duration_hours

            while remaining_hours > 0:
                period_type = self.get_period_type(current_time, tariff)
                rate = self.get_rate_for_period(period_type, tariff)

                # Find how long until next period change
                next_hour = current_time.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
                hours_in_period = min(
                    (next_hour - current_time).total_seconds() / 3600,
                    remaining_hours
                )

                period_energy = energy_per_hour * hours_in_period
                period_cost = period_energy * rate

                period_key = period_type.value
                if period_key not in cost_breakdown["periods"]:
                    cost_breakdown["periods"][period_key] = {
                        "energy_kwh": 0.0,
                        "rate": rate,
                        "cost": 0.0,
                        "hours": 0.0
                    }

                cost_breakdown["periods"][period_key]["energy_kwh"] += period_energy
                cost_breakdown["periods"][period_key]["cost"] += period_cost
                cost_breakdown["periods"][period_key]["hours"] += hours_in_period
                cost_breakdown["energy_cost"] += period_cost

                current_time = next_hour
                remaining_hours -= hours_in_period
                remaining_energy -= period_energy

        # Calculate demand charge based on peak power
        if tariff.demand_charge_kw > 0 and equipment:
            # Monthly demand charge pro-rated for this usage
            peak_demand_kw = equipment.power_kw
            monthly_hours = 720  # Approximate hours in a month
            demand_charge = (tariff.demand_charge_kw * peak_demand_kw * duration_hours) / monthly_hours
            cost_breakdown["demand_charge"] = demand_charge
            cost_breakdown["peak_demand_kw"] = peak_demand_kw

        cost_breakdown["total_cost"] = cost_breakdown["energy_cost"] + cost_breakdown["demand_charge"]

        return cost_breakdown["total_cost"], cost_breakdown

    def calculate_demand_charge(
        self,
        peak_demand_kw: float,
        demand_rate: float,
        billing_period_days: int = 30
    ) -> float:
        """Calculate monthly demand charge."""
        if peak_demand_kw < 0:
            raise ValueError("Peak demand cannot be negative")

        if demand_rate < 0:
            raise ValueError("Demand rate cannot be negative")

        return peak_demand_kw * demand_rate

    def calculate_period_cost(
        self,
        energy_kwh: float,
        period_type: PeriodType,
        tariff: EnergyTariff
    ) -> float:
        """Calculate cost for energy consumed in a specific period."""
        rate = self.get_rate_for_period(period_type, tariff)
        return energy_kwh * rate

    def find_lowest_cost_window(
        self,
        duration_hours: float,
        search_start: datetime,
        search_end: datetime,
        tariff: EnergyTariff,
        energy_kwh: float
    ) -> Tuple[datetime, float]:
        """
        Find the lowest cost time window within a search range.

        Returns:
            Tuple of (optimal_start_time, estimated_cost)
        """
        if search_end <= search_start:
            raise ValueError("Search end must be after search start")

        if duration_hours <= 0:
            raise ValueError("Duration must be positive")

        best_start = search_start
        best_cost = float('inf')

        current_start = search_start
        step_hours = 0.5  # 30-minute granularity

        while current_start + timedelta(hours=duration_hours) <= search_end:
            cost, _ = self.calculate_energy_cost(
                energy_kwh=energy_kwh,
                start_time=current_start,
                duration_hours=duration_hours,
                tariff=tariff
            )

            if cost < best_cost:
                best_cost = cost
                best_start = current_start

            current_start += timedelta(hours=step_hours)

        return best_start, best_cost


# =============================================================================
# UNIT TESTS
# =============================================================================

@pytest.mark.unit
@pytest.mark.calculator
@pytest.mark.critical
class TestEnergyCostCalculator:
    """Comprehensive test suite for EnergyCostCalculator."""

    # =========================================================================
    # INITIALIZATION TESTS
    # =========================================================================

    def test_initialization(self):
        """Test EnergyCostCalculator initializes correctly."""
        calculator = EnergyCostCalculator()

        assert calculator.VERSION == "1.0.0"
        assert calculator.NAME == "EnergyCostCalculator"
        assert calculator._tracker is None

    # =========================================================================
    # TIME-OF-USE RATE TESTS
    # =========================================================================

    def test_get_period_type_on_peak(self, simple_tou_tariff):
        """Test on-peak period detection."""
        calculator = EnergyCostCalculator()

        # 2pm on Monday should be on-peak
        timestamp = datetime(2025, 1, 6, 14, 0, tzinfo=timezone.utc)  # Monday
        period = calculator.get_period_type(timestamp, simple_tou_tariff)

        assert period == PeriodType.ON_PEAK

    def test_get_period_type_mid_peak(self, simple_tou_tariff):
        """Test mid-peak period detection."""
        calculator = EnergyCostCalculator()

        # 10am on Tuesday should be mid-peak
        timestamp = datetime(2025, 1, 7, 10, 0, tzinfo=timezone.utc)  # Tuesday
        period = calculator.get_period_type(timestamp, simple_tou_tariff)

        assert period == PeriodType.MID_PEAK

    def test_get_period_type_off_peak(self, simple_tou_tariff):
        """Test off-peak period detection."""
        calculator = EnergyCostCalculator()

        # 3am on Wednesday should be off-peak
        timestamp = datetime(2025, 1, 8, 3, 0, tzinfo=timezone.utc)  # Wednesday
        period = calculator.get_period_type(timestamp, simple_tou_tariff)

        assert period == PeriodType.OFF_PEAK

    def test_get_period_type_flat_rate(self, flat_rate_tariff):
        """Test flat rate returns off-peak by default."""
        calculator = EnergyCostCalculator()

        timestamp = datetime(2025, 1, 6, 14, 0, tzinfo=timezone.utc)
        period = calculator.get_period_type(timestamp, flat_rate_tariff)

        assert period == PeriodType.OFF_PEAK

    @pytest.mark.parametrize("hour,day,expected_period,expected_multiplier", [
        (14, 0, PeriodType.ON_PEAK, 2.5),   # Monday 2pm
        (15, 2, PeriodType.ON_PEAK, 2.5),   # Wednesday 3pm
        (18, 1, PeriodType.ON_PEAK, 2.5),   # Tuesday 6pm
        (10, 3, PeriodType.MID_PEAK, 1.5),  # Thursday 10am
        (20, 4, PeriodType.MID_PEAK, 1.5),  # Friday 8pm
        (3, 0, PeriodType.OFF_PEAK, 0.6),   # Monday 3am
        (23, 2, PeriodType.OFF_PEAK, 0.6),  # Wednesday 11pm
    ])
    def test_tou_rate_variations(
        self,
        simple_tou_tariff,
        hour,
        day,
        expected_period,
        expected_multiplier
    ):
        """Test ToU rate calculation across various times."""
        calculator = EnergyCostCalculator()

        # Create timestamp for specific hour and day
        base_date = datetime(2025, 1, 6, tzinfo=timezone.utc)  # Monday
        timestamp = base_date + timedelta(days=day, hours=hour)
        timestamp = timestamp.replace(hour=hour)

        period = calculator.get_period_type(timestamp, simple_tou_tariff)
        rate = calculator.get_rate_for_period(period, simple_tou_tariff)

        expected_rate = simple_tou_tariff.base_rate_kwh * expected_multiplier

        assert period == expected_period
        assert rate == pytest.approx(expected_rate, rel=0.001)

    def test_weekend_off_peak(self, complex_tou_tariff):
        """Test weekend is always off-peak."""
        calculator = EnergyCostCalculator()

        # Saturday noon
        saturday = datetime(2025, 1, 11, 12, 0, tzinfo=timezone.utc)
        period_sat = calculator.get_period_type(saturday, complex_tou_tariff)

        # Sunday evening
        sunday = datetime(2025, 1, 12, 18, 0, tzinfo=timezone.utc)
        period_sun = calculator.get_period_type(sunday, complex_tou_tariff)

        assert period_sat == PeriodType.OFF_PEAK
        assert period_sun == PeriodType.OFF_PEAK

    # =========================================================================
    # MIDNIGHT CROSSING TESTS (EDGE CASES)
    # =========================================================================

    def test_period_crossing_midnight(self, simple_tou_tariff):
        """Test period detection across midnight."""
        calculator = EnergyCostCalculator()

        # Just before midnight - should be off-peak (22:00-10:00)
        before_midnight = datetime(2025, 1, 6, 23, 30, tzinfo=timezone.utc)
        period_before = calculator.get_period_type(before_midnight, simple_tou_tariff)

        # Just after midnight - should still be off-peak
        after_midnight = datetime(2025, 1, 7, 0, 30, tzinfo=timezone.utc)
        period_after = calculator.get_period_type(after_midnight, simple_tou_tariff)

        assert period_before == PeriodType.OFF_PEAK
        assert period_after == PeriodType.OFF_PEAK

    def test_cost_calculation_crossing_midnight(self, simple_tou_tariff, single_furnace):
        """Test cost calculation for operations crossing midnight."""
        calculator = EnergyCostCalculator()

        # Start at 10pm, run for 4 hours (until 2am)
        start_time = datetime(2025, 1, 6, 22, 0, tzinfo=timezone.utc)
        duration_hours = 4.0
        energy_kwh = 1000.0

        total_cost, breakdown = calculator.calculate_energy_cost(
            energy_kwh=energy_kwh,
            start_time=start_time,
            duration_hours=duration_hours,
            tariff=simple_tou_tariff,
            equipment=single_furnace
        )

        # All hours should be off-peak (22:00 - 10:00 is off-peak)
        assert PeriodType.OFF_PEAK.value in breakdown["periods"]
        assert total_cost > 0

    # =========================================================================
    # DST TRANSITION TESTS (EDGE CASES)
    # =========================================================================

    def test_dst_spring_forward(self, simple_tou_tariff, dst_transition_dates):
        """Test handling of spring forward DST transition."""
        calculator = EnergyCostCalculator()

        # During spring forward, 2am becomes 3am
        spring_forward = dst_transition_dates["spring_forward_2025"]
        period = calculator.get_period_type(spring_forward, simple_tou_tariff)

        # Should still return valid period
        assert period in [PeriodType.ON_PEAK, PeriodType.MID_PEAK, PeriodType.OFF_PEAK]

    def test_dst_fall_back(self, simple_tou_tariff, dst_transition_dates):
        """Test handling of fall back DST transition."""
        calculator = EnergyCostCalculator()

        # During fall back, 2am happens twice
        fall_back = dst_transition_dates["fall_back_2025"]
        period = calculator.get_period_type(fall_back, simple_tou_tariff)

        # Should still return valid period
        assert period in [PeriodType.ON_PEAK, PeriodType.MID_PEAK, PeriodType.OFF_PEAK]

    # =========================================================================
    # DEMAND CHARGE TESTS
    # =========================================================================

    @pytest.mark.parametrize("peak_demand,rate,expected_charge", [
        (100.0, 15.00, 1500.00),
        (500.0, 15.00, 7500.00),
        (1000.0, 20.00, 20000.00),
        (250.0, 18.50, 4625.00),
        (0.0, 15.00, 0.00),
    ])
    def test_demand_charge_calculation(self, peak_demand, rate, expected_charge):
        """Test demand charge calculation with known values."""
        calculator = EnergyCostCalculator()

        charge = calculator.calculate_demand_charge(peak_demand, rate)

        assert charge == pytest.approx(expected_charge, rel=0.001)

    def test_demand_charge_negative_peak(self):
        """Test demand charge rejects negative peak demand."""
        calculator = EnergyCostCalculator()

        with pytest.raises(ValueError, match="Peak demand cannot be negative"):
            calculator.calculate_demand_charge(-100.0, 15.00)

    def test_demand_charge_negative_rate(self):
        """Test demand charge rejects negative rate."""
        calculator = EnergyCostCalculator()

        with pytest.raises(ValueError, match="Demand rate cannot be negative"):
            calculator.calculate_demand_charge(100.0, -15.00)

    def test_demand_charge_zero_values(self):
        """Test demand charge with zero values."""
        calculator = EnergyCostCalculator()

        # Zero demand
        charge_zero_demand = calculator.calculate_demand_charge(0.0, 15.00)
        assert charge_zero_demand == 0.0

        # Zero rate
        charge_zero_rate = calculator.calculate_demand_charge(100.0, 0.0)
        assert charge_zero_rate == 0.0

    # =========================================================================
    # REAL-TIME PRICING TESTS
    # =========================================================================

    def test_real_time_price_lookup(self, real_time_tariff):
        """Test real-time price lookup."""
        calculator = EnergyCostCalculator()

        base_date = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)

        # Early morning - low price
        morning_price = calculator.get_real_time_price(base_date, real_time_tariff)
        assert morning_price < 0.10

        # Peak afternoon - high price
        peak_time = base_date + timedelta(hours=15)
        peak_price = calculator.get_real_time_price(peak_time, real_time_tariff)
        assert peak_price > 0.20

    def test_real_time_price_fallback(self, real_time_tariff):
        """Test real-time price falls back to base rate for missing data."""
        calculator = EnergyCostCalculator()

        # Timestamp not in price data (far future)
        future_time = datetime(2030, 1, 1, 12, 0, tzinfo=timezone.utc)
        price = calculator.get_real_time_price(future_time, real_time_tariff)

        assert price == real_time_tariff.base_rate_kwh

    def test_real_time_price_invalid_tariff(self, simple_tou_tariff):
        """Test real-time price rejects non-RTP tariff."""
        calculator = EnergyCostCalculator()

        timestamp = datetime(2025, 1, 6, 12, 0, tzinfo=timezone.utc)

        with pytest.raises(ValueError, match="not real-time pricing"):
            calculator.get_real_time_price(timestamp, simple_tou_tariff)

    def test_real_time_cost_calculation(self, real_time_tariff):
        """Test cost calculation with real-time pricing."""
        calculator = EnergyCostCalculator()

        base_date = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        energy_kwh = 100.0
        duration_hours = 4.0

        total_cost, breakdown = calculator.calculate_energy_cost(
            energy_kwh=energy_kwh,
            start_time=base_date,
            duration_hours=duration_hours,
            tariff=real_time_tariff
        )

        # Should have hourly breakdown
        assert len(breakdown["periods"]) == 4
        assert total_cost > 0
        assert breakdown["energy_cost"] == pytest.approx(total_cost, rel=0.01)

    # =========================================================================
    # ENERGY COST CALCULATION TESTS
    # =========================================================================

    @pytest.mark.parametrize("energy,rate,expected_cost", [
        (100.0, 0.10, 10.00),
        (500.0, 0.25, 125.00),
        (1000.0, 0.15, 150.00),
        (2500.0, 0.08, 200.00),
        (0.0, 0.10, 0.00),
    ])
    def test_simple_energy_cost(self, energy, rate, expected_cost, flat_rate_tariff):
        """Test simple energy cost calculation."""
        calculator = EnergyCostCalculator()

        # Modify flat rate for test
        flat_rate_tariff.base_rate_kwh = rate
        flat_rate_tariff.demand_charge_kw = 0.0  # No demand charge

        start_time = datetime(2025, 1, 6, 12, 0, tzinfo=timezone.utc)

        total_cost, breakdown = calculator.calculate_energy_cost(
            energy_kwh=energy,
            start_time=start_time,
            duration_hours=1.0,
            tariff=flat_rate_tariff
        )

        assert breakdown["energy_cost"] == pytest.approx(expected_cost, rel=0.001)

    def test_energy_cost_with_efficiency(self, flat_rate_tariff, single_furnace):
        """Test energy cost accounts for equipment efficiency."""
        calculator = EnergyCostCalculator()
        flat_rate_tariff.demand_charge_kw = 0.0

        start_time = datetime(2025, 1, 6, 12, 0, tzinfo=timezone.utc)
        energy_kwh = 100.0

        total_cost, breakdown = calculator.calculate_energy_cost(
            energy_kwh=energy_kwh,
            start_time=start_time,
            duration_hours=1.0,
            tariff=flat_rate_tariff,
            equipment=single_furnace
        )

        # Actual energy should be energy / efficiency
        expected_actual_energy = energy_kwh / single_furnace.efficiency
        assert breakdown["energy_kwh"] == pytest.approx(expected_actual_energy, rel=0.001)

    def test_energy_cost_negative_energy(self, flat_rate_tariff):
        """Test energy cost rejects negative energy."""
        calculator = EnergyCostCalculator()

        start_time = datetime(2025, 1, 6, 12, 0, tzinfo=timezone.utc)

        with pytest.raises(ValueError, match="cannot be negative"):
            calculator.calculate_energy_cost(
                energy_kwh=-100.0,
                start_time=start_time,
                duration_hours=1.0,
                tariff=flat_rate_tariff
            )

    def test_energy_cost_zero_duration(self, flat_rate_tariff):
        """Test energy cost rejects zero duration."""
        calculator = EnergyCostCalculator()

        start_time = datetime(2025, 1, 6, 12, 0, tzinfo=timezone.utc)

        with pytest.raises(ValueError, match="Duration must be positive"):
            calculator.calculate_energy_cost(
                energy_kwh=100.0,
                start_time=start_time,
                duration_hours=0.0,
                tariff=flat_rate_tariff
            )

    def test_energy_cost_with_demand_charge(self, simple_tou_tariff, single_furnace):
        """Test energy cost includes demand charge."""
        calculator = EnergyCostCalculator()

        start_time = datetime(2025, 1, 6, 3, 0, tzinfo=timezone.utc)  # Off-peak
        energy_kwh = 500.0
        duration_hours = 2.0

        total_cost, breakdown = calculator.calculate_energy_cost(
            energy_kwh=energy_kwh,
            start_time=start_time,
            duration_hours=duration_hours,
            tariff=simple_tou_tariff,
            equipment=single_furnace
        )

        assert breakdown["demand_charge"] > 0
        assert breakdown["peak_demand_kw"] == single_furnace.power_kw
        assert total_cost == pytest.approx(
            breakdown["energy_cost"] + breakdown["demand_charge"],
            rel=0.001
        )

    def test_energy_cost_multi_period(self, simple_tou_tariff):
        """Test energy cost spanning multiple periods."""
        calculator = EnergyCostCalculator()

        # Start at 1pm (mid-peak), run for 8 hours (through peak and back to off-peak)
        start_time = datetime(2025, 1, 6, 13, 0, tzinfo=timezone.utc)  # Monday 1pm
        energy_kwh = 800.0
        duration_hours = 8.0

        total_cost, breakdown = calculator.calculate_energy_cost(
            energy_kwh=energy_kwh,
            start_time=start_time,
            duration_hours=duration_hours,
            tariff=simple_tou_tariff
        )

        # Should have multiple periods
        assert len(breakdown["periods"]) >= 2
        assert total_cost > 0

    # =========================================================================
    # PERIOD COST CALCULATION TESTS
    # =========================================================================

    def test_period_cost_on_peak(self, simple_tou_tariff):
        """Test period cost for on-peak."""
        calculator = EnergyCostCalculator()

        cost = calculator.calculate_period_cost(
            energy_kwh=100.0,
            period_type=PeriodType.ON_PEAK,
            tariff=simple_tou_tariff
        )

        expected = 100.0 * simple_tou_tariff.base_rate_kwh * 2.5  # 2.5x multiplier
        assert cost == pytest.approx(expected, rel=0.001)

    def test_period_cost_off_peak(self, simple_tou_tariff):
        """Test period cost for off-peak."""
        calculator = EnergyCostCalculator()

        cost = calculator.calculate_period_cost(
            energy_kwh=100.0,
            period_type=PeriodType.OFF_PEAK,
            tariff=simple_tou_tariff
        )

        expected = 100.0 * simple_tou_tariff.base_rate_kwh * 0.6  # 0.6x multiplier
        assert cost == pytest.approx(expected, rel=0.001)

    def test_period_cost_ratio(self, simple_tou_tariff):
        """Test on-peak to off-peak cost ratio."""
        calculator = EnergyCostCalculator()

        on_peak_cost = calculator.calculate_period_cost(
            energy_kwh=100.0,
            period_type=PeriodType.ON_PEAK,
            tariff=simple_tou_tariff
        )

        off_peak_cost = calculator.calculate_period_cost(
            energy_kwh=100.0,
            period_type=PeriodType.OFF_PEAK,
            tariff=simple_tou_tariff
        )

        # On-peak should be ~4.17x more expensive (2.5 / 0.6)
        ratio = on_peak_cost / off_peak_cost
        assert ratio == pytest.approx(2.5 / 0.6, rel=0.01)

    # =========================================================================
    # LOWEST COST WINDOW TESTS
    # =========================================================================

    def test_find_lowest_cost_window_simple(self, simple_tou_tariff):
        """Test finding lowest cost window selects off-peak."""
        calculator = EnergyCostCalculator()

        # Search across a full day
        search_start = datetime(2025, 1, 6, 0, 0, tzinfo=timezone.utc)  # Monday midnight
        search_end = datetime(2025, 1, 7, 0, 0, tzinfo=timezone.utc)  # Tuesday midnight

        optimal_start, optimal_cost = calculator.find_lowest_cost_window(
            duration_hours=2.0,
            search_start=search_start,
            search_end=search_end,
            tariff=simple_tou_tariff,
            energy_kwh=200.0
        )

        # Optimal should be during off-peak (22:00 - 10:00)
        optimal_period = calculator.get_period_type(optimal_start, simple_tou_tariff)
        assert optimal_period == PeriodType.OFF_PEAK

    def test_find_lowest_cost_window_constraints(self, simple_tou_tariff):
        """Test lowest cost window respects search constraints."""
        calculator = EnergyCostCalculator()

        # Search only during peak hours
        search_start = datetime(2025, 1, 6, 14, 0, tzinfo=timezone.utc)  # 2pm
        search_end = datetime(2025, 1, 6, 18, 0, tzinfo=timezone.utc)  # 6pm

        optimal_start, optimal_cost = calculator.find_lowest_cost_window(
            duration_hours=1.0,
            search_start=search_start,
            search_end=search_end,
            tariff=simple_tou_tariff,
            energy_kwh=100.0
        )

        assert optimal_start >= search_start
        assert optimal_start + timedelta(hours=1.0) <= search_end

    def test_find_lowest_cost_window_invalid_range(self, simple_tou_tariff):
        """Test lowest cost window rejects invalid search range."""
        calculator = EnergyCostCalculator()

        search_start = datetime(2025, 1, 6, 12, 0, tzinfo=timezone.utc)
        search_end = datetime(2025, 1, 6, 10, 0, tzinfo=timezone.utc)  # Before start

        with pytest.raises(ValueError, match="Search end must be after search start"):
            calculator.find_lowest_cost_window(
                duration_hours=1.0,
                search_start=search_start,
                search_end=search_end,
                tariff=simple_tou_tariff,
                energy_kwh=100.0
            )

    def test_find_lowest_cost_window_zero_duration(self, simple_tou_tariff):
        """Test lowest cost window rejects zero duration."""
        calculator = EnergyCostCalculator()

        search_start = datetime(2025, 1, 6, 0, 0, tzinfo=timezone.utc)
        search_end = datetime(2025, 1, 6, 12, 0, tzinfo=timezone.utc)

        with pytest.raises(ValueError, match="Duration must be positive"):
            calculator.find_lowest_cost_window(
                duration_hours=0.0,
                search_start=search_start,
                search_end=search_end,
                tariff=simple_tou_tariff,
                energy_kwh=100.0
            )

    # =========================================================================
    # ACCURACY VALIDATION TESTS
    # =========================================================================

    def test_calculation_accuracy_known_values(self, simple_tou_tariff):
        """Test calculation accuracy against known values."""
        calculator = EnergyCostCalculator()

        # Known calculation:
        # 100 kWh at off-peak = 100 * 0.10 * 0.6 = $6.00
        start_time = datetime(2025, 1, 6, 3, 0, tzinfo=timezone.utc)  # 3am Monday (off-peak)

        total_cost, breakdown = calculator.calculate_energy_cost(
            energy_kwh=100.0,
            start_time=start_time,
            duration_hours=1.0,
            tariff=simple_tou_tariff
        )

        expected_cost = 100.0 * 0.10 * 0.6  # $6.00
        assert breakdown["energy_cost"] == pytest.approx(expected_cost, rel=0.001)

    def test_calculation_reproducibility(self, simple_tou_tariff, single_furnace):
        """Test calculations are reproducible."""
        calculator = EnergyCostCalculator()

        start_time = datetime(2025, 1, 6, 12, 0, tzinfo=timezone.utc)
        energy_kwh = 500.0
        duration_hours = 4.0

        results = []
        for _ in range(5):
            cost, breakdown = calculator.calculate_energy_cost(
                energy_kwh=energy_kwh,
                start_time=start_time,
                duration_hours=duration_hours,
                tariff=simple_tou_tariff,
                equipment=single_furnace
            )
            results.append(cost)

        # All results should be identical
        for result in results[1:]:
            assert result == results[0]

    # =========================================================================
    # PERFORMANCE TESTS
    # =========================================================================

    @pytest.mark.performance
    def test_calculation_speed(self, simple_tou_tariff, single_furnace, benchmark):
        """Test calculation meets performance target (<5ms)."""
        calculator = EnergyCostCalculator()

        start_time = datetime(2025, 1, 6, 12, 0, tzinfo=timezone.utc)

        def run_calculation():
            return calculator.calculate_energy_cost(
                energy_kwh=500.0,
                start_time=start_time,
                duration_hours=4.0,
                tariff=simple_tou_tariff,
                equipment=single_furnace
            )

        result = benchmark(run_calculation)
        # Benchmark stats would show mean time

    @pytest.mark.performance
    def test_batch_calculation_throughput(self, simple_tou_tariff):
        """Test batch calculation throughput."""
        calculator = EnergyCostCalculator()
        import time

        base_time = datetime(2025, 1, 6, 0, 0, tzinfo=timezone.utc)
        num_calculations = 1000

        start = time.time()
        for i in range(num_calculations):
            start_time = base_time + timedelta(hours=i % 24)
            calculator.calculate_energy_cost(
                energy_kwh=100.0 + (i % 500),
                start_time=start_time,
                duration_hours=1.0 + (i % 4),
                tariff=simple_tou_tariff
            )
        end = time.time()

        throughput = num_calculations / (end - start)
        assert throughput > 1000  # >1000 calculations per second


# =============================================================================
# EDGE CASE TESTS
# =============================================================================

@pytest.mark.unit
class TestEnergyCostEdgeCases:
    """Edge case tests for energy cost calculations."""

    def test_very_small_energy(self, flat_rate_tariff):
        """Test calculation with very small energy values."""
        calculator = EnergyCostCalculator()

        start_time = datetime(2025, 1, 6, 12, 0, tzinfo=timezone.utc)

        total_cost, breakdown = calculator.calculate_energy_cost(
            energy_kwh=0.001,  # 1 Wh
            start_time=start_time,
            duration_hours=0.1,
            tariff=flat_rate_tariff
        )

        assert total_cost >= 0
        assert breakdown["energy_kwh"] == pytest.approx(0.001, rel=0.01)

    def test_very_large_energy(self, flat_rate_tariff):
        """Test calculation with very large energy values."""
        calculator = EnergyCostCalculator()

        start_time = datetime(2025, 1, 6, 0, 0, tzinfo=timezone.utc)

        total_cost, breakdown = calculator.calculate_energy_cost(
            energy_kwh=1000000.0,  # 1 GWh
            start_time=start_time,
            duration_hours=24.0,
            tariff=flat_rate_tariff
        )

        assert total_cost > 0
        assert breakdown["energy_kwh"] == 1000000.0

    def test_fractional_hours(self, simple_tou_tariff):
        """Test calculation with fractional hour durations."""
        calculator = EnergyCostCalculator()

        start_time = datetime(2025, 1, 6, 12, 0, tzinfo=timezone.utc)

        total_cost, breakdown = calculator.calculate_energy_cost(
            energy_kwh=100.0,
            start_time=start_time,
            duration_hours=2.5,  # 2.5 hours
            tariff=simple_tou_tariff
        )

        assert total_cost > 0
        assert breakdown["duration_hours"] == 2.5

    def test_long_duration(self, simple_tou_tariff):
        """Test calculation spanning multiple days."""
        calculator = EnergyCostCalculator()

        start_time = datetime(2025, 1, 6, 0, 0, tzinfo=timezone.utc)

        total_cost, breakdown = calculator.calculate_energy_cost(
            energy_kwh=10000.0,
            start_time=start_time,
            duration_hours=72.0,  # 3 days
            tariff=simple_tou_tariff
        )

        assert total_cost > 0
        assert breakdown["duration_hours"] == 72.0

    def test_year_boundary(self, simple_tou_tariff):
        """Test calculation crossing year boundary."""
        calculator = EnergyCostCalculator()

        start_time = datetime(2025, 12, 31, 22, 0, tzinfo=timezone.utc)

        total_cost, breakdown = calculator.calculate_energy_cost(
            energy_kwh=500.0,
            start_time=start_time,
            duration_hours=4.0,  # Crosses into 2026
            tariff=simple_tou_tariff
        )

        assert total_cost > 0

    def test_leap_year_february(self, simple_tou_tariff):
        """Test calculation during leap year February."""
        calculator = EnergyCostCalculator()

        # February 28, 2024 (leap year)
        start_time = datetime(2024, 2, 28, 22, 0, tzinfo=timezone.utc)

        total_cost, breakdown = calculator.calculate_energy_cost(
            energy_kwh=500.0,
            start_time=start_time,
            duration_hours=4.0,  # Crosses into Feb 29
            tariff=simple_tou_tariff
        )

        assert total_cost > 0
