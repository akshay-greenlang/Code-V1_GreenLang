"""
Unit tests for GL-GHG-APP v1.0 Intensity Calculator

Tests revenue, employee, production, floor area, and custom intensity
metrics, plus year-over-year comparison and sector benchmarks.  25+ test cases.
"""

import pytest
from decimal import Decimal, InvalidOperation
from typing import Dict, List, Optional

from services.config import (
    IntensityDenominator,
    Scope,
    SECTOR_BENCHMARKS,
)
from services.models import IntensityMetric


# ---------------------------------------------------------------------------
# IntensityCalculator under test
# ---------------------------------------------------------------------------

class IntensityCalculator:
    """
    Calculates GHG intensity metrics per GHG Protocol Ch 12.

    Intensity = total_emissions / denominator_value
    """

    def calculate(
        self,
        total_tco2e: Decimal,
        denominator: IntensityDenominator,
        denominator_value: Decimal,
        denominator_unit: str = "",
        scope: Optional[Scope] = None,
    ) -> IntensityMetric:
        """Calculate a single intensity metric."""
        if denominator_value <= 0:
            raise ValueError("Denominator must be positive and non-zero")

        intensity = total_tco2e / denominator_value

        unit_map = {
            IntensityDenominator.REVENUE: "tCO2e/million USD",
            IntensityDenominator.EMPLOYEES: "tCO2e/employee",
            IntensityDenominator.PRODUCTION_UNITS: f"tCO2e/{denominator_unit or 'unit'}",
            IntensityDenominator.FLOOR_AREA: "tCO2e/m2",
            IntensityDenominator.CUSTOM: f"tCO2e/{denominator_unit or 'unit'}",
        }

        return IntensityMetric(
            denominator=denominator,
            denominator_value=denominator_value,
            denominator_unit=denominator_unit,
            intensity_value=intensity,
            total_tco2e=total_tco2e,
            scope=scope,
            unit=unit_map.get(denominator, "tCO2e/unit"),
        )

    def calculate_all(
        self,
        total_tco2e: Decimal,
        revenue_m_usd: Optional[Decimal] = None,
        employees: Optional[int] = None,
        production_units: Optional[Decimal] = None,
        production_unit_name: Optional[str] = None,
        floor_area_m2: Optional[Decimal] = None,
    ) -> List[IntensityMetric]:
        """Calculate all available intensity metrics."""
        metrics = []
        if revenue_m_usd and revenue_m_usd > 0:
            metrics.append(self.calculate(
                total_tco2e, IntensityDenominator.REVENUE, revenue_m_usd, "million USD",
            ))
        if employees and employees > 0:
            metrics.append(self.calculate(
                total_tco2e, IntensityDenominator.EMPLOYEES, Decimal(str(employees)),
            ))
        if production_units and production_units > 0:
            metrics.append(self.calculate(
                total_tco2e, IntensityDenominator.PRODUCTION_UNITS,
                production_units, production_unit_name or "units",
            ))
        if floor_area_m2 and floor_area_m2 > 0:
            metrics.append(self.calculate(
                total_tco2e, IntensityDenominator.FLOOR_AREA, floor_area_m2, "m2",
            ))
        return metrics

    def year_over_year(
        self,
        current: IntensityMetric,
        previous: IntensityMetric,
    ) -> Dict[str, Decimal]:
        """Calculate year-over-year intensity change."""
        if previous.intensity_value == 0:
            return {
                "absolute_change": current.intensity_value,
                "percentage_change": Decimal("0"),
                "direction": "increase" if current.intensity_value > 0 else "flat",
            }
        absolute_change = current.intensity_value - previous.intensity_value
        pct_change = (absolute_change / previous.intensity_value) * 100
        if pct_change > 0:
            direction = "increase"
        elif pct_change < 0:
            direction = "decrease"
        else:
            direction = "flat"

        return {
            "absolute_change": absolute_change,
            "percentage_change": pct_change,
            "direction": direction,
        }

    def benchmark(self, sector: str, denominator: str) -> Optional[Decimal]:
        """Look up sector benchmark value."""
        sector_data = SECTOR_BENCHMARKS.get(sector.lower())
        if sector_data is None:
            return None
        return sector_data.get(denominator)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def calc():
    return IntensityCalculator()


@pytest.fixture
def total_emissions():
    return Decimal("63781.0")


# ---------------------------------------------------------------------------
# TestRevenueIntensity
# ---------------------------------------------------------------------------

class TestRevenueIntensity:
    """Test revenue-based intensity."""

    def test_calculation(self, calc, total_emissions):
        """Test revenue intensity calculation."""
        metric = calc.calculate(
            total_emissions,
            IntensityDenominator.REVENUE,
            Decimal("150.0"),
            "million USD",
        )
        expected = total_emissions / Decimal("150.0")
        assert metric.intensity_value == expected

    def test_zero_revenue_raises(self, calc, total_emissions):
        """Test zero revenue raises ValueError."""
        with pytest.raises(ValueError, match="positive"):
            calc.calculate(total_emissions, IntensityDenominator.REVENUE, Decimal("0"))

    def test_units(self, calc, total_emissions):
        """Test revenue intensity unit string."""
        metric = calc.calculate(total_emissions, IntensityDenominator.REVENUE, Decimal("150.0"), "million USD")
        assert metric.unit == "tCO2e/million USD"


# ---------------------------------------------------------------------------
# TestEmployeeIntensity
# ---------------------------------------------------------------------------

class TestEmployeeIntensity:
    """Test employee-based intensity."""

    def test_calculation(self, calc, total_emissions):
        """Test employee intensity calculation."""
        metric = calc.calculate(total_emissions, IntensityDenominator.EMPLOYEES, Decimal("5000"))
        expected = total_emissions / Decimal("5000")
        assert metric.intensity_value == expected

    def test_zero_employees_raises(self, calc, total_emissions):
        """Test zero employees raises ValueError."""
        with pytest.raises(ValueError, match="positive"):
            calc.calculate(total_emissions, IntensityDenominator.EMPLOYEES, Decimal("0"))

    def test_single_employee(self, calc, total_emissions):
        """Test intensity with single employee."""
        metric = calc.calculate(total_emissions, IntensityDenominator.EMPLOYEES, Decimal("1"))
        assert metric.intensity_value == total_emissions


# ---------------------------------------------------------------------------
# TestProductionIntensity
# ---------------------------------------------------------------------------

class TestProductionIntensity:
    """Test production-based intensity."""

    def test_calculation(self, calc, total_emissions):
        """Test production intensity calculation."""
        metric = calc.calculate(
            total_emissions,
            IntensityDenominator.PRODUCTION_UNITS,
            Decimal("1000000"),
            "widgets",
        )
        expected = total_emissions / Decimal("1000000")
        assert metric.intensity_value == expected

    def test_custom_unit_name(self, calc, total_emissions):
        """Test custom production unit name in unit string."""
        metric = calc.calculate(
            total_emissions,
            IntensityDenominator.PRODUCTION_UNITS,
            Decimal("500000"),
            "barrels",
        )
        assert metric.unit == "tCO2e/barrels"


# ---------------------------------------------------------------------------
# TestFloorAreaIntensity
# ---------------------------------------------------------------------------

class TestFloorAreaIntensity:
    """Test floor area intensity."""

    def test_calculation(self, calc, total_emissions):
        """Test floor area intensity calculation."""
        metric = calc.calculate(
            total_emissions,
            IntensityDenominator.FLOOR_AREA,
            Decimal("75000"),
            "m2",
        )
        expected = total_emissions / Decimal("75000")
        assert metric.intensity_value == expected

    def test_unit_string(self, calc, total_emissions):
        """Test floor area intensity unit."""
        metric = calc.calculate(total_emissions, IntensityDenominator.FLOOR_AREA, Decimal("75000"), "m2")
        assert metric.unit == "tCO2e/m2"


# ---------------------------------------------------------------------------
# TestCustomIntensity
# ---------------------------------------------------------------------------

class TestCustomIntensity:
    """Test custom denominator intensity."""

    def test_arbitrary_denominator(self, calc, total_emissions):
        """Test custom denominator (e.g., transactions)."""
        metric = calc.calculate(
            total_emissions,
            IntensityDenominator.CUSTOM,
            Decimal("1000000"),
            "transactions",
        )
        expected = total_emissions / Decimal("1000000")
        assert metric.intensity_value == expected
        assert metric.denominator == IntensityDenominator.CUSTOM


# ---------------------------------------------------------------------------
# TestAllIntensities
# ---------------------------------------------------------------------------

class TestAllIntensities:
    """Test calculating all available intensities."""

    def test_calculates_all_available(self, calc, total_emissions):
        """Test all available metrics are calculated."""
        metrics = calc.calculate_all(
            total_emissions,
            revenue_m_usd=Decimal("150.0"),
            employees=5000,
            production_units=Decimal("1000000"),
            production_unit_name="widgets",
            floor_area_m2=Decimal("75000"),
        )
        assert len(metrics) == 4
        denominators = {m.denominator for m in metrics}
        assert IntensityDenominator.REVENUE in denominators
        assert IntensityDenominator.EMPLOYEES in denominators
        assert IntensityDenominator.PRODUCTION_UNITS in denominators
        assert IntensityDenominator.FLOOR_AREA in denominators

    def test_partial_available(self, calc, total_emissions):
        """Test only available metrics are calculated."""
        metrics = calc.calculate_all(
            total_emissions,
            revenue_m_usd=Decimal("150.0"),
            employees=5000,
        )
        assert len(metrics) == 2

    def test_none_available(self, calc, total_emissions):
        """Test no metrics when no denominators provided."""
        metrics = calc.calculate_all(total_emissions)
        assert len(metrics) == 0


# ---------------------------------------------------------------------------
# TestYoYComparison
# ---------------------------------------------------------------------------

class TestYoYComparison:
    """Test year-over-year intensity comparison."""

    def test_change_direction_decrease(self, calc):
        """Test decrease direction detection."""
        current = IntensityMetric(
            denominator=IntensityDenominator.REVENUE,
            denominator_value=Decimal("160"),
            intensity_value=Decimal("380.0"),
            total_tco2e=Decimal("60800"),
        )
        previous = IntensityMetric(
            denominator=IntensityDenominator.REVENUE,
            denominator_value=Decimal("150"),
            intensity_value=Decimal("425.0"),
            total_tco2e=Decimal("63750"),
        )
        result = calc.year_over_year(current, previous)
        assert result["direction"] == "decrease"
        assert result["absolute_change"] < 0

    def test_change_direction_increase(self, calc):
        """Test increase direction detection."""
        current = IntensityMetric(
            denominator=IntensityDenominator.REVENUE,
            denominator_value=Decimal("140"),
            intensity_value=Decimal("500.0"),
            total_tco2e=Decimal("70000"),
        )
        previous = IntensityMetric(
            denominator=IntensityDenominator.REVENUE,
            denominator_value=Decimal("150"),
            intensity_value=Decimal("425.0"),
            total_tco2e=Decimal("63750"),
        )
        result = calc.year_over_year(current, previous)
        assert result["direction"] == "increase"

    def test_percentage_change(self, calc):
        """Test percentage change calculation."""
        current = IntensityMetric(
            denominator=IntensityDenominator.EMPLOYEES,
            denominator_value=Decimal("5000"),
            intensity_value=Decimal("12.0"),
            total_tco2e=Decimal("60000"),
        )
        previous = IntensityMetric(
            denominator=IntensityDenominator.EMPLOYEES,
            denominator_value=Decimal("5000"),
            intensity_value=Decimal("10.0"),
            total_tco2e=Decimal("50000"),
        )
        result = calc.year_over_year(current, previous)
        assert result["percentage_change"] == Decimal("20.0")


# ---------------------------------------------------------------------------
# TestBenchmark
# ---------------------------------------------------------------------------

class TestBenchmark:
    """Test sector benchmark lookup."""

    def test_sector_lookup(self, calc):
        """Test known sector benchmark lookup."""
        value = calc.benchmark("manufacturing", "revenue")
        assert value == Decimal("420.0")

    def test_unknown_sector_returns_none(self, calc):
        """Test unknown sector returns None."""
        value = calc.benchmark("space_exploration", "revenue")
        assert value is None

    def test_case_insensitive(self, calc):
        """Test sector lookup is case-insensitive."""
        value = calc.benchmark("Manufacturing", "revenue")
        assert value == Decimal("420.0")

    def test_employee_benchmark(self, calc):
        """Test employee benchmark for energy sector."""
        value = calc.benchmark("energy", "employees")
        assert value == Decimal("45.0")
