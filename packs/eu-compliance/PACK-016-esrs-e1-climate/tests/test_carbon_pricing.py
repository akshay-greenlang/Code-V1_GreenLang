# -*- coding: utf-8 -*-
"""
PACK-016 ESRS E1 Climate Pack - Carbon Pricing Engine Tests
===============================================================

Unit tests for CarbonPricingEngine (Engine 7) covering carbon price
registration, shadow price scenarios, disclosure building, weighted
average price, coverage calculation, completeness, and E1-8 data points.

ESRS E1-8: Internal carbon pricing.

Target: 40+ tests.

Author:  GreenLang Platform Team (GL-TestEngineer)
Pack:    PACK-016 ESRS E1 Climate Change
Date:    March 2026
"""

from decimal import Decimal

import pytest

from .conftest import _load_engine


# ---------------------------------------------------------------------------
# Module-scoped engine loading
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def mod():
    """Load the carbon_pricing engine module."""
    return _load_engine("carbon_pricing")


@pytest.fixture
def engine(mod):
    """Create a fresh CarbonPricingEngine instance."""
    return mod.CarbonPricingEngine()


@pytest.fixture
def shadow_price(mod):
    """Create a sample shadow carbon price."""
    return mod.CarbonPrice(
        mechanism=mod.PricingMechanism.SHADOW_PRICE,
        price_per_tco2e=Decimal("100.00"),
        currency=mod.CurrencyCode.EUR,
        scope=mod.PricingScope.INVESTMENT_DECISIONS,
        emissions_covered_tco2e=Decimal("50000"),
        emissions_covered_pct=Decimal("85.0"),
        covers_scope_1=True,
        covers_scope_2=True,
        covers_scope_3=False,
    )


@pytest.fixture
def internal_fee(mod):
    """Create a sample internal carbon fee."""
    return mod.CarbonPrice(
        mechanism=mod.PricingMechanism.INTERNAL_FEE,
        price_per_tco2e=Decimal("50.00"),
        currency=mod.CurrencyCode.EUR,
        scope=mod.PricingScope.ALL,
        emissions_covered_tco2e=Decimal("30000"),
        emissions_covered_pct=Decimal("51.0"),
        covers_scope_1=True,
        covers_scope_2=True,
        covers_scope_3=True,
        revenue_generated=Decimal("1500000"),
    )


@pytest.fixture
def sample_scenario(mod):
    """Create a sample shadow price scenario."""
    return mod.ShadowPriceScenario(
        scenario_name="IEA NZE Trajectory",
        price_trajectory={
            2025: Decimal("85.00"),
            2030: Decimal("130.00"),
            2035: Decimal("175.00"),
            2040: Decimal("220.00"),
            2050: Decimal("310.00"),
        },
        source="IEA World Energy Outlook 2025",
    )


# ===========================================================================
# Enum Tests
# ===========================================================================


class TestPricingEnums:
    """Tests for carbon pricing enums."""

    def test_pricing_mechanism_values(self, mod):
        """PricingMechanism has at least 3 values."""
        assert len(mod.PricingMechanism) >= 3
        values = {m.value for m in mod.PricingMechanism}
        assert "shadow_price" in values
        assert "internal_fee" in values
        assert "implicit_price" in values

    def test_pricing_scope_values(self, mod):
        """PricingScope has expected values."""
        assert len(mod.PricingScope) >= 4
        values = {m.value for m in mod.PricingScope}
        assert "investment_decisions" in values
        assert "procurement" in values
        assert "all" in values

    def test_currency_code_values(self, mod):
        """CurrencyCode includes EUR and USD."""
        values = {m.value for m in mod.CurrencyCode}
        assert "EUR" in values
        assert "USD" in values


# ===========================================================================
# Constants Tests
# ===========================================================================


class TestPricingConstants:
    """Tests for carbon pricing constants."""

    def test_e1_8_datapoints_exist(self, mod):
        """E1_8_DATAPOINTS is a non-empty dict."""
        assert len(mod.E1_8_DATAPOINTS) >= 10

    def test_reference_carbon_prices_exist(self, mod):
        """REFERENCE_CARBON_PRICES has entries."""
        assert len(mod.REFERENCE_CARBON_PRICES) >= 3
        assert "eu_ets" in mod.REFERENCE_CARBON_PRICES

    def test_eu_ets_price(self, mod):
        """EU ETS reference price is a positive Decimal."""
        eu_ets = mod.REFERENCE_CARBON_PRICES["eu_ets"]
        assert eu_ets["price_eur"] > Decimal("0")

    def test_shadow_price_benchmarks_exist(self, mod):
        """SHADOW_PRICE_BENCHMARKS has low/central/high."""
        assert "low" in mod.SHADOW_PRICE_BENCHMARKS
        assert "central" in mod.SHADOW_PRICE_BENCHMARKS
        assert "high" in mod.SHADOW_PRICE_BENCHMARKS

    def test_shadow_price_trajectory_years(self, mod):
        """Shadow price benchmarks cover 2025-2050."""
        for scenario in ["low", "central", "high"]:
            trajectory = mod.SHADOW_PRICE_BENCHMARKS[scenario]
            assert 2025 in trajectory
            assert 2050 in trajectory

    def test_mechanism_descriptions_exist(self, mod):
        """MECHANISM_DESCRIPTIONS has entries for all mechanisms."""
        for mech in mod.PricingMechanism:
            assert mech.value in mod.MECHANISM_DESCRIPTIONS

    def test_scope_descriptions_exist(self, mod):
        """SCOPE_DESCRIPTIONS has entries for all scopes."""
        for scope in mod.PricingScope:
            assert scope.value in mod.SCOPE_DESCRIPTIONS


# ===========================================================================
# CarbonPrice Model Tests
# ===========================================================================


class TestCarbonPriceModel:
    """Tests for CarbonPrice Pydantic model."""

    def test_create_valid_price(self, mod):
        """Create a valid CarbonPrice."""
        price = mod.CarbonPrice(
            mechanism=mod.PricingMechanism.SHADOW_PRICE,
            price_per_tco2e=Decimal("100.00"),
        )
        assert price.mechanism == mod.PricingMechanism.SHADOW_PRICE
        assert price.price_per_tco2e == Decimal("100.00")
        assert len(price.price_id) > 0

    def test_eur_currency(self, mod):
        """Default currency is EUR."""
        price = mod.CarbonPrice(
            mechanism=mod.PricingMechanism.SHADOW_PRICE,
            price_per_tco2e=Decimal("80.00"),
        )
        assert price.currency == mod.CurrencyCode.EUR

    def test_usd_currency(self, mod):
        """Can set USD currency."""
        price = mod.CarbonPrice(
            mechanism=mod.PricingMechanism.INTERNAL_FEE,
            price_per_tco2e=Decimal("45.00"),
            currency=mod.CurrencyCode.USD,
        )
        assert price.currency == mod.CurrencyCode.USD


# ===========================================================================
# Shadow Price Scenario Tests
# ===========================================================================


class TestShadowPriceScenario:
    """Tests for ShadowPriceScenario model."""

    def test_create_valid_scenario(self, mod):
        """Create a valid ShadowPriceScenario."""
        scenario = mod.ShadowPriceScenario(
            scenario_name="High Carbon World",
            price_trajectory={
                2025: Decimal("120.00"),
                2030: Decimal("200.00"),
                2050: Decimal("520.00"),
            },
        )
        assert scenario.scenario_name == "High Carbon World"
        assert len(scenario.price_trajectory) == 3

    def test_trajectory_over_years(self, sample_scenario):
        """Trajectory has prices over multiple years."""
        assert len(sample_scenario.price_trajectory) >= 5
        # Prices should increase over time
        assert sample_scenario.price_trajectory[2050] > sample_scenario.price_trajectory[2025]

    def test_empty_name_rejected(self, mod):
        """Empty scenario name is rejected."""
        with pytest.raises(Exception):
            mod.ShadowPriceScenario(
                scenario_name="",
                price_trajectory={2025: Decimal("100")},
            )


# ===========================================================================
# Build Disclosure Tests
# ===========================================================================


class TestBuildDisclosure:
    """Tests for build_pricing_disclosure method."""

    def test_basic_disclosure(self, engine, shadow_price):
        """Build basic pricing disclosure."""
        engine.set_carbon_price(shadow_price)
        result = engine.build_pricing_disclosure(
            total_emissions=Decimal("58823.5")
        )
        assert result is not None
        assert result.applies_carbon_pricing is True
        assert result.total_mechanisms >= 1

    def test_multiple_mechanisms(self, mod):
        """Disclosure with multiple pricing mechanisms."""
        eng = mod.CarbonPricingEngine()
        eng.set_carbon_price(mod.CarbonPrice(
            mechanism=mod.PricingMechanism.SHADOW_PRICE,
            price_per_tco2e=Decimal("100.00"),
            emissions_covered_tco2e=Decimal("40000"),
            emissions_covered_pct=Decimal("68.0"),
        ))
        eng.set_carbon_price(mod.CarbonPrice(
            mechanism=mod.PricingMechanism.INTERNAL_FEE,
            price_per_tco2e=Decimal("50.00"),
            emissions_covered_tco2e=Decimal("20000"),
            emissions_covered_pct=Decimal("34.0"),
        ))
        result = eng.build_pricing_disclosure(
            total_emissions=Decimal("58823.5")
        )
        assert result.total_mechanisms >= 2

    def test_disclosure_coverage(self, mod):
        """Disclosure includes coverage calculation."""
        eng = mod.CarbonPricingEngine()
        eng.set_carbon_price(mod.CarbonPrice(
            mechanism=mod.PricingMechanism.SHADOW_PRICE,
            price_per_tco2e=Decimal("100.00"),
            emissions_covered_tco2e=Decimal("50000"),
            emissions_covered_pct=Decimal("85.0"),
        ))
        result = eng.build_pricing_disclosure(
            total_emissions=Decimal("58823.5")
        )
        assert result.total_emissions_covered_tco2e > Decimal("0")

    def test_disclosure_provenance(self, mod):
        """Disclosure has a provenance hash."""
        eng = mod.CarbonPricingEngine()
        eng.set_carbon_price(mod.CarbonPrice(
            mechanism=mod.PricingMechanism.SHADOW_PRICE,
            price_per_tco2e=Decimal("80.00"),
        ))
        result = eng.build_pricing_disclosure(
            total_emissions=Decimal("50000")
        )
        assert len(result.provenance_hash) == 64

    def test_no_pricing_disclosure(self, mod):
        """Disclosure without any prices."""
        eng = mod.CarbonPricingEngine()
        result = eng.build_pricing_disclosure(
            total_emissions=Decimal("50000")
        )
        assert result.applies_carbon_pricing is False


# ===========================================================================
# Weighted Average Tests
# ===========================================================================


class TestWeightedAverage:
    """Tests for weighted average price calculation."""

    def test_single_price(self, engine, shadow_price):
        """Single price mechanism gives the price as average."""
        engine.set_carbon_price(shadow_price)
        result = engine.calculate_weighted_average_price([shadow_price])
        assert float(result) == pytest.approx(100.0, abs=1.0)

    def test_multiple_prices(self, mod):
        """Multiple mechanisms produce emission-weighted average."""
        prices = [
            mod.CarbonPrice(
                mechanism=mod.PricingMechanism.SHADOW_PRICE,
                price_per_tco2e=Decimal("100.00"),
                emissions_covered_tco2e=Decimal("30000"),
            ),
            mod.CarbonPrice(
                mechanism=mod.PricingMechanism.INTERNAL_FEE,
                price_per_tco2e=Decimal("50.00"),
                emissions_covered_tco2e=Decimal("20000"),
            ),
        ]
        eng = mod.CarbonPricingEngine()
        result = eng.calculate_weighted_average_price(prices)
        # Weighted: (100*30000 + 50*20000) / (30000+20000) = 4000000/50000 = 80
        assert float(result) == pytest.approx(80.0, abs=1.0)


# ===========================================================================
# Coverage Tests
# ===========================================================================


class TestCoverage:
    """Tests for emission coverage calculation."""

    def test_full_coverage(self, mod):
        """Full coverage when all emissions are covered."""
        prices = [
            mod.CarbonPrice(
                mechanism=mod.PricingMechanism.SHADOW_PRICE,
                price_per_tco2e=Decimal("100.00"),
                emissions_covered_tco2e=Decimal("50000"),
                emissions_covered_pct=Decimal("100.0"),
            ),
        ]
        eng = mod.CarbonPricingEngine()
        result = eng.calculate_coverage(
            prices=prices,
            total_emissions=Decimal("50000"),
        )
        assert float(result) == pytest.approx(100.0, abs=1.0)

    def test_partial_coverage(self, mod):
        """Partial coverage calculation."""
        prices = [
            mod.CarbonPrice(
                mechanism=mod.PricingMechanism.SHADOW_PRICE,
                price_per_tco2e=Decimal("100.00"),
                emissions_covered_tco2e=Decimal("25000"),
                emissions_covered_pct=Decimal("50.0"),
            ),
        ]
        eng = mod.CarbonPricingEngine()
        result = eng.calculate_coverage(
            prices=prices,
            total_emissions=Decimal("50000"),
        )
        assert float(result) == pytest.approx(50.0, abs=1.0)


# ===========================================================================
# Shadow Analysis Tests
# ===========================================================================


class TestShadowAnalysis:
    """Tests for shadow price scenario analysis."""

    def test_add_benchmark_scenarios(self, mod):
        """Add benchmark scenarios from constants."""
        eng = mod.CarbonPricingEngine()
        scenarios = eng.add_benchmark_scenarios()
        assert len(scenarios) >= 3

    def test_custom_scenario(self, mod, sample_scenario):
        """Add a custom shadow price scenario."""
        eng = mod.CarbonPricingEngine()
        result = eng.add_scenario(sample_scenario)
        assert result is not None

    def test_scenario_in_disclosure(self, mod, sample_scenario):
        """Scenarios appear in the disclosure."""
        eng = mod.CarbonPricingEngine()
        eng.set_carbon_price(mod.CarbonPrice(
            mechanism=mod.PricingMechanism.SHADOW_PRICE,
            price_per_tco2e=Decimal("100.00"),
        ))
        eng.add_scenario(sample_scenario)
        result = eng.build_pricing_disclosure(
            total_emissions=Decimal("50000")
        )
        assert len(result.shadow_price_scenarios) >= 1


# ===========================================================================
# Completeness Tests
# ===========================================================================


class TestCompleteness:
    """Tests for E1-8 completeness validation."""

    def test_complete(self, engine, shadow_price, sample_scenario):
        """Complete pricing disclosure."""
        engine.set_carbon_price(shadow_price)
        engine.add_scenario(sample_scenario)
        result = engine.build_pricing_disclosure(
            total_emissions=Decimal("58823.5")
        )
        completeness = engine.validate_completeness(result)
        assert isinstance(completeness, dict)
        assert len(completeness) > 0

    def test_incomplete(self, mod):
        """Empty pricing has lower completeness."""
        eng = mod.CarbonPricingEngine()
        result = eng.build_pricing_disclosure(
            total_emissions=Decimal("50000")
        )
        completeness = eng.validate_completeness(result)
        assert isinstance(completeness, dict)


# ===========================================================================
# E1-8 Data Points Tests
# ===========================================================================


class TestE18Datapoints:
    """Tests for E1-8 required data point extraction."""

    def test_returns_datapoints(self, engine, shadow_price):
        """get_e1_8_datapoints returns required data points."""
        engine.set_carbon_price(shadow_price)
        result = engine.build_pricing_disclosure(
            total_emissions=Decimal("58823.5")
        )
        datapoints = engine.get_e1_8_datapoints(result)
        assert isinstance(datapoints, dict)
        assert len(datapoints) >= 8

    def test_e1_8_datapoints_constant(self, mod):
        """E1_8_DATAPOINTS dict has at least 10 entries."""
        assert len(mod.E1_8_DATAPOINTS) >= 10


# ===========================================================================
# Additional Mechanism Tests
# ===========================================================================


class TestAdditionalMechanisms:
    """Tests for additional pricing mechanism types."""

    def test_implicit_price(self, mod):
        """Create an implicit carbon price."""
        price = mod.CarbonPrice(
            mechanism=mod.PricingMechanism.IMPLICIT_PRICE,
            price_per_tco2e=Decimal("35.00"),
        )
        assert price.mechanism == mod.PricingMechanism.IMPLICIT_PRICE

    def test_offset_price(self, mod):
        """Create an offset-based carbon price."""
        price = mod.CarbonPrice(
            mechanism=mod.PricingMechanism.OFFSET_PRICE,
            price_per_tco2e=Decimal("25.00"),
            currency=mod.CurrencyCode.USD,
        )
        assert price.mechanism == mod.PricingMechanism.OFFSET_PRICE
        assert price.currency == mod.CurrencyCode.USD

    def test_all_mechanisms_have_description(self, mod):
        """Every mechanism has a description."""
        for mech in mod.PricingMechanism:
            desc = mod.MECHANISM_DESCRIPTIONS[mech.value]
            assert len(desc) > 0


# ===========================================================================
# Benchmark Scenario Advanced Tests
# ===========================================================================


class TestBenchmarkScenariosAdvanced:
    """Advanced tests for benchmark shadow price scenarios."""

    def test_central_trajectory_increasing(self, mod):
        """Central trajectory prices increase over time."""
        traj = mod.SHADOW_PRICE_BENCHMARKS["central"]
        years = sorted(traj.keys())
        for i in range(len(years) - 1):
            assert traj[years[i]] <= traj[years[i + 1]]

    def test_high_exceeds_central(self, mod):
        """High trajectory exceeds central at every year."""
        high = mod.SHADOW_PRICE_BENCHMARKS["high"]
        central = mod.SHADOW_PRICE_BENCHMARKS["central"]
        for year in high:
            if year in central:
                assert high[year] >= central[year]

    def test_low_below_central(self, mod):
        """Low trajectory is at or below central at every year."""
        low = mod.SHADOW_PRICE_BENCHMARKS["low"]
        central = mod.SHADOW_PRICE_BENCHMARKS["central"]
        for year in low:
            if year in central:
                assert low[year] <= central[year]
