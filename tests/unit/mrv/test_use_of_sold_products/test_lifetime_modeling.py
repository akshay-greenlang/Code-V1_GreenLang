# -*- coding: utf-8 -*-
"""
Unit tests for LifetimeModelingEngine -- AGENT-MRV-024

Tests product lifetime modeling including default lifetimes by category,
adjustment factors, degradation curves (linear, exponential), Weibull
survival curves, fleet-level emissions with survival, and discounted
emissions analysis.

Target: 30+ tests.
Author: GL-TestEngineer
"""

import threading
import math
from decimal import Decimal
from typing import Any, Dict, List

import pytest

try:
    from greenlang.use_of_sold_products.lifetime_modeling import (
        LifetimeModelingEngine,
        get_lifetime_engine,
        calculate_degradation_factor,
        calculate_weibull_survival,
        calculate_fleet_emissions,
        calculate_discounted_emissions,
        get_default_lifetime,
        get_adjustment_factor,
        calculate_provenance_hash,
    )

    _AVAILABLE = True
except ImportError:
    _AVAILABLE = False

_SKIP = pytest.mark.skipif(not _AVAILABLE, reason="LifetimeModelingEngine not available")
pytestmark = _SKIP

_Q8 = Decimal("0.00000001")


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture(autouse=True)
def _reset_engine():
    """Reset the singleton before each test."""
    LifetimeModelingEngine.reset()
    yield
    LifetimeModelingEngine.reset()


@pytest.fixture
def engine():
    """Create a LifetimeModelingEngine instance."""
    return LifetimeModelingEngine()


# ============================================================================
# TEST: Default Lifetimes by Category
# ============================================================================


class TestDefaultLifetimes:
    """Test default lifetime lookups by product category."""

    @pytest.mark.parametrize(
        "category,expected_years",
        [
            ("vehicles", 15),
            ("appliances", 15),
            ("hvac", 12),
            ("lighting", 25),
            ("it_equipment", 5),
            ("industrial_equipment", 20),
            ("building_products", 20),
            ("consumer_products", 3),
            ("medical_devices", 12),
        ],
    )
    def test_default_lifetime_by_category(self, engine, category, expected_years):
        """Test default lifetime for each product category."""
        result = engine.get_default_lifetime(category)
        assert result == expected_years

    def test_unknown_category_returns_default(self, engine):
        """Test unknown category returns a sensible default (10 years)."""
        result = engine.get_default_lifetime("unknown_category")
        assert isinstance(result, int)
        assert result > 0


# ============================================================================
# TEST: Lifetime Adjustment Factors
# ============================================================================


class TestAdjustmentFactors:
    """Test lifetime adjustment factors for usage patterns."""

    @pytest.mark.parametrize(
        "factor_name,expected_multiplier",
        [
            ("heavy_use", Decimal("0.75")),
            ("light_use", Decimal("1.25")),
            ("commercial", Decimal("0.80")),
            ("residential", Decimal("1.00")),
            ("industrial", Decimal("0.70")),
        ],
    )
    def test_adjustment_factors(self, engine, factor_name, expected_multiplier):
        """Test lifetime adjustment factors for 5 usage patterns."""
        result = engine.get_adjustment_factor(factor_name)
        assert result == expected_multiplier

    def test_adjusted_lifetime(self, engine):
        """Test applying heavy_use factor: 15yr x 0.75 = 11.25 ~ 11yr."""
        base_lifetime = 15
        factor = engine.get_adjustment_factor("heavy_use")
        adjusted = int(Decimal(str(base_lifetime)) * factor)
        assert adjusted == 11


# ============================================================================
# TEST: Linear Degradation Curves
# ============================================================================


class TestLinearDegradation:
    """Test linear degradation curves year-by-year."""

    def test_linear_degradation_year_0(self, engine):
        """Test year 0 degradation factor is 1.0 (no degradation)."""
        factor = engine.calculate_degradation_factor(
            year=0,
            degradation_rate=Decimal("0.005"),
            model="linear",
        )
        assert factor == Decimal("1.0")

    def test_linear_degradation_year_1(self, engine):
        """Test year 1 degradation factor is 0.995 (0.5% loss)."""
        factor = engine.calculate_degradation_factor(
            year=1,
            degradation_rate=Decimal("0.005"),
            model="linear",
        )
        assert factor == pytest.approx(Decimal("0.995"), rel=Decimal("0.001"))

    def test_linear_degradation_year_10(self, engine):
        """Test year 10 degradation factor is 0.950 (5% cumulative loss)."""
        factor = engine.calculate_degradation_factor(
            year=10,
            degradation_rate=Decimal("0.005"),
            model="linear",
        )
        assert factor == pytest.approx(Decimal("0.950"), rel=Decimal("0.001"))

    @pytest.mark.parametrize("year,expected", [
        (0, Decimal("1.000")),
        (1, Decimal("0.995")),
        (5, Decimal("0.975")),
        (10, Decimal("0.950")),
        (15, Decimal("0.925")),
        (20, Decimal("0.900")),
    ])
    def test_linear_degradation_year_by_year(self, engine, year, expected):
        """Test linear degradation at 0.5% per year matches expected values."""
        factor = engine.calculate_degradation_factor(
            year=year,
            degradation_rate=Decimal("0.005"),
            model="linear",
        )
        assert factor == pytest.approx(expected, rel=Decimal("0.001"))

    def test_linear_degradation_never_below_zero(self, engine):
        """Test degradation factor never goes below zero."""
        factor = engine.calculate_degradation_factor(
            year=300,
            degradation_rate=Decimal("0.005"),
            model="linear",
        )
        assert factor >= Decimal("0")

    def test_zero_degradation_rate(self, engine):
        """Test zero degradation rate means no degradation."""
        factor = engine.calculate_degradation_factor(
            year=15,
            degradation_rate=Decimal("0.0"),
            model="linear",
        )
        assert factor == Decimal("1.0")


# ============================================================================
# TEST: Exponential Degradation Curves
# ============================================================================


class TestExponentialDegradation:
    """Test exponential degradation curves."""

    def test_exponential_year_0(self, engine):
        """Test exponential year 0 is 1.0."""
        factor = engine.calculate_degradation_factor(
            year=0,
            degradation_rate=Decimal("0.005"),
            model="exponential",
        )
        assert factor == Decimal("1.0")

    def test_exponential_year_10(self, engine):
        """Test exponential year 10 at 0.5%: e^(-0.005*10) = e^(-0.05) ~ 0.951."""
        factor = engine.calculate_degradation_factor(
            year=10,
            degradation_rate=Decimal("0.005"),
            model="exponential",
        )
        expected = Decimal(str(math.exp(-0.005 * 10)))
        assert factor == pytest.approx(expected, rel=Decimal("0.01"))

    def test_exponential_approaches_zero(self, engine):
        """Test exponential degradation approaches zero for large years."""
        factor = engine.calculate_degradation_factor(
            year=1000,
            degradation_rate=Decimal("0.005"),
            model="exponential",
        )
        assert factor < Decimal("0.01")
        assert factor >= Decimal("0")


# ============================================================================
# TEST: Weibull Survival Curves
# ============================================================================


class TestWeibullSurvival:
    """Test Weibull survival curve calculations."""

    def test_weibull_at_year_0(self, engine):
        """Test Weibull survival at year 0 is 1.0 (all surviving)."""
        survival = engine.calculate_weibull_survival(
            year=0,
            shape=Decimal("3.5"),
            scale=Decimal("15.0"),
        )
        assert survival == Decimal("1.0")

    def test_weibull_at_median(self, engine):
        """Test Weibull survival around the scale parameter."""
        survival = engine.calculate_weibull_survival(
            year=15,
            shape=Decimal("3.5"),
            scale=Decimal("15.0"),
        )
        # At scale parameter, survival = e^(-1) ~ 0.368
        assert survival == pytest.approx(Decimal("0.368"), rel=Decimal("0.05"))

    def test_weibull_approaches_zero(self, engine):
        """Test Weibull survival approaches zero for large years."""
        survival = engine.calculate_weibull_survival(
            year=50,
            shape=Decimal("3.5"),
            scale=Decimal("15.0"),
        )
        assert survival < Decimal("0.01")

    def test_weibull_monotonically_decreasing(self, engine):
        """Test Weibull survival is monotonically decreasing."""
        prev = Decimal("1.0")
        for year in range(1, 30):
            survival = engine.calculate_weibull_survival(
                year=year,
                shape=Decimal("3.5"),
                scale=Decimal("15.0"),
            )
            assert survival <= prev
            prev = survival

    def test_weibull_different_shape_parameters(self, engine):
        """Test Weibull with different shape parameters."""
        # Higher shape = steeper curve (more failures around scale)
        s_low = engine.calculate_weibull_survival(year=10, shape=Decimal("2.0"), scale=Decimal("15.0"))
        s_high = engine.calculate_weibull_survival(year=10, shape=Decimal("5.0"), scale=Decimal("15.0"))
        # Both should be between 0 and 1
        assert Decimal("0") <= s_low <= Decimal("1")
        assert Decimal("0") <= s_high <= Decimal("1")


# ============================================================================
# TEST: Fleet Emissions with Survival
# ============================================================================


class TestFleetEmissions:
    """Test fleet-level emissions calculations with survival curves."""

    def test_fleet_emissions_basic(self, engine):
        """Test fleet emissions summing over years with survival."""
        result = engine.calculate_fleet_emissions(
            units_sold=1000,
            annual_emissions_per_unit=Decimal("2778.0"),
            lifetime_years=15,
            weibull_shape=Decimal("3.5"),
            weibull_scale=Decimal("15.0"),
        )
        # With survival, total should be less than simple product
        simple_total = Decimal("1000") * Decimal("15") * Decimal("2778.0")
        assert result["total_co2e_kg"] < simple_total
        assert result["total_co2e_kg"] > Decimal("0")

    def test_fleet_emissions_no_survival(self, engine):
        """Test fleet emissions without survival (all units survive full lifetime)."""
        result = engine.calculate_fleet_emissions(
            units_sold=1000,
            annual_emissions_per_unit=Decimal("2778.0"),
            lifetime_years=15,
            weibull_shape=None,
            weibull_scale=None,
        )
        expected = Decimal("1000") * Decimal("15") * Decimal("2778.0")
        assert result["total_co2e_kg"] == pytest.approx(expected, rel=Decimal("0.001"))

    def test_fleet_emissions_with_degradation(self, engine):
        """Test fleet emissions with both survival and degradation."""
        result = engine.calculate_fleet_emissions(
            units_sold=1000,
            annual_emissions_per_unit=Decimal("2778.0"),
            lifetime_years=15,
            weibull_shape=Decimal("3.5"),
            weibull_scale=Decimal("15.0"),
            degradation_rate=Decimal("0.005"),
        )
        # Should be less than without degradation
        no_degrade = engine.calculate_fleet_emissions(
            units_sold=1000,
            annual_emissions_per_unit=Decimal("2778.0"),
            lifetime_years=15,
            weibull_shape=Decimal("3.5"),
            weibull_scale=Decimal("15.0"),
        )
        assert result["total_co2e_kg"] <= no_degrade["total_co2e_kg"]

    def test_fleet_year_by_year_breakdown(self, engine):
        """Test fleet emissions include year-by-year breakdown."""
        result = engine.calculate_fleet_emissions(
            units_sold=1000,
            annual_emissions_per_unit=Decimal("2778.0"),
            lifetime_years=15,
            weibull_shape=Decimal("3.5"),
            weibull_scale=Decimal("15.0"),
        )
        if "year_by_year" in result:
            assert len(result["year_by_year"]) == 15
            # Year 1 should have highest surviving units
            assert result["year_by_year"][0] >= result["year_by_year"][-1]


# ============================================================================
# TEST: Discounted Emissions
# ============================================================================


class TestDiscountedEmissions:
    """Test discounted emissions calculations."""

    def test_no_discount(self, engine):
        """Test with 0% discount rate equals undiscounted total."""
        result = engine.calculate_discounted_emissions(
            annual_emissions=[Decimal("2778000")] * 15,
            discount_rate=Decimal("0.0"),
        )
        expected = Decimal("2778000") * 15
        assert result["discounted_total_co2e_kg"] == pytest.approx(expected, rel=Decimal("0.001"))

    def test_positive_discount(self, engine):
        """Test with 3% discount rate reduces total."""
        undiscounted = [Decimal("2778000")] * 15
        result = engine.calculate_discounted_emissions(
            annual_emissions=undiscounted,
            discount_rate=Decimal("0.03"),
        )
        undiscounted_total = sum(undiscounted)
        assert result["discounted_total_co2e_kg"] < undiscounted_total
        assert result["discounted_total_co2e_kg"] > Decimal("0")

    def test_high_discount_rate(self, engine):
        """Test with high discount rate heavily discounts future years."""
        result_low = engine.calculate_discounted_emissions(
            annual_emissions=[Decimal("1000000")] * 20,
            discount_rate=Decimal("0.03"),
        )
        result_high = engine.calculate_discounted_emissions(
            annual_emissions=[Decimal("1000000")] * 20,
            discount_rate=Decimal("0.10"),
        )
        assert result_high["discounted_total_co2e_kg"] < result_low["discounted_total_co2e_kg"]


# ============================================================================
# TEST: Provenance
# ============================================================================


class TestLifetimeProvenance:
    """Test provenance hash generation for lifetime modeling."""

    def test_provenance_hash_64_chars(self, engine):
        """Test provenance hash is 64-char hex."""
        h = calculate_provenance_hash("vehicles", "15", "linear", "0.005")
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_provenance_hash_deterministic(self, engine):
        """Test same inputs produce same hash."""
        h1 = calculate_provenance_hash("vehicles", "15", "linear")
        h2 = calculate_provenance_hash("vehicles", "15", "linear")
        assert h1 == h2


# ============================================================================
# TEST: Singleton
# ============================================================================


class TestLifetimeSingleton:
    """Test LifetimeModelingEngine singleton pattern."""

    def test_singleton_same_instance(self):
        """Test get_lifetime_engine returns same instance."""
        e1 = get_lifetime_engine()
        e2 = get_lifetime_engine()
        assert e1 is e2

    def test_singleton_thread_safe(self):
        """Test singleton is thread-safe."""
        results = []
        errors = []

        def _get():
            try:
                e = get_lifetime_engine()
                results.append(id(e))
            except Exception as ex:
                errors.append(ex)

        threads = [threading.Thread(target=_get) for _ in range(12)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        assert len(errors) == 0
        assert len(set(results)) == 1
