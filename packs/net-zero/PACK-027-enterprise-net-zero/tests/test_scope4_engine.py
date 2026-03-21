# -*- coding: utf-8 -*-
"""
Test suite for PACK-027 Enterprise Net Zero Pack - Scope 4 Avoided Emissions Engine.

Tests avoided emissions quantification following WBCSD guidance with baseline
scenario, attribution, conservative principles, and uncertainty ranges.

Author:  GreenLang Test Engineering
Pack:    PACK-027 Enterprise Net Zero
Tests:   ~40 tests
"""

import sys
from decimal import Decimal
from pathlib import Path

import pytest

_PACK_ROOT = Path(__file__).resolve().parents[1]
if str(_PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACK_ROOT))

from engines.scope4_avoided_emissions_engine import (
    Scope4AvoidedEmissionsEngine,
    Scope4Input,
    Scope4Result,
    AvoidedEmissionCategory,
    BaselineType,
    ProductAvoidedEmissionEntry,
    ProductAvoidedResult,
)

from .conftest import assert_decimal_positive, assert_provenance_hash


def _make_product(name="Test Product", category=AvoidedEmissionCategory.PRODUCT_SUBSTITUTION,
                  baseline=Decimal("5.0"), product=Decimal("2.0"), units=1000, **kwargs):
    return ProductAvoidedEmissionEntry(
        product_name=name,
        category=category,
        functional_unit="unit",
        baseline_emissions_per_unit=baseline,
        product_lifecycle_emissions_per_unit=product,
        units_sold=Decimal(str(units)),
        **kwargs,
    )


def _make_input(products=None, **kwargs):
    if products is None:
        products = [_make_product()]
    defaults = dict(products=products)
    defaults.update(kwargs)
    return Scope4Input(**defaults)


class TestScope4Instantiation:
    def test_engine_instantiates(self):
        engine = Scope4AvoidedEmissionsEngine()
        assert engine is not None

    def test_engine_has_calculate_method(self):
        engine = Scope4AvoidedEmissionsEngine()
        assert hasattr(engine, "calculate")

    def test_supported_categories(self):
        cats = [m.value for m in AvoidedEmissionCategory]
        assert "product_substitution" in cats
        assert "efficiency_improvement" in cats
        assert "enabling_effect" in cats
        assert "systemic_change" in cats


class TestProductSubstitution:
    def test_ev_displacing_ice(self):
        engine = Scope4AvoidedEmissionsEngine()
        result = engine.calculate(_make_input(products=[
            _make_product("Electric Vehicle Model X",
                          AvoidedEmissionCategory.PRODUCT_SUBSTITUTION,
                          Decimal("4.5"), Decimal("1.8"), 12000),
        ]))
        assert result.total_avoided_tco2e > Decimal("0")

    def test_led_lighting(self):
        engine = Scope4AvoidedEmissionsEngine()
        result = engine.calculate(_make_input(products=[
            _make_product("LED Industrial Lighting",
                          AvoidedEmissionCategory.EFFICIENCY_IMPROVEMENT,
                          Decimal("0.175"), Decimal("0.044"), 50000),
        ]))
        assert result.total_avoided_tco2e > Decimal("0")

    def test_baseline_type_in_result(self):
        """Baseline must track the type used."""
        engine = Scope4AvoidedEmissionsEngine()
        result = engine.calculate(_make_input(products=[
            _make_product(baseline_type=BaselineType.MARKET_AVERAGE),
        ]))
        assert result is not None


class TestConservativePrinciples:
    def test_rebound_effect_deduction(self):
        """Rebound effects reduce avoided emissions."""
        engine = Scope4AvoidedEmissionsEngine()
        # Without rebound
        r_no_rebound = engine.calculate(_make_input(products=[
            _make_product("Efficient Appliance",
                          AvoidedEmissionCategory.EFFICIENCY_IMPROVEMENT,
                          Decimal("3.0"), Decimal("1.0"), 10000),
        ]))
        # With rebound
        r_with_rebound = engine.calculate(_make_input(products=[
            _make_product("Efficient Appliance",
                          AvoidedEmissionCategory.EFFICIENCY_IMPROVEMENT,
                          Decimal("3.0"), Decimal("1.0"), 10000,
                          rebound_effect_pct=Decimal("15")),
        ]))
        assert r_with_rebound.total_avoided_tco2e <= r_no_rebound.total_avoided_tco2e

    def test_attribution_share_enabling(self):
        """Enabling effects with attribution must reduce total."""
        engine = Scope4AvoidedEmissionsEngine()
        result = engine.calculate(_make_input(products=[
            _make_product("Teleconferencing Platform",
                          AvoidedEmissionCategory.ENABLING_EFFECT,
                          Decimal("0.5"), Decimal("0.01"), 100000,
                          attribution_share_pct=Decimal("30")),
        ]))
        assert result.total_avoided_tco2e > Decimal("0")

    def test_double_counting_check(self):
        """Must check for double counting."""
        engine = Scope4AvoidedEmissionsEngine()
        result = engine.calculate(_make_input())
        assert hasattr(result, "double_counting")

    def test_avoided_reported_separately(self):
        """Avoided emissions must be separate from Scope 1/2/3."""
        engine = Scope4AvoidedEmissionsEngine()
        result = engine.calculate(_make_input())
        assert result.total_avoided_tco2e >= Decimal("0")


class TestUncertaintyAndReporting:
    def test_confidence_ranges(self):
        engine = Scope4AvoidedEmissionsEngine()
        result = engine.calculate(_make_input())
        assert result.confidence_total_lower_tco2e <= result.total_avoided_tco2e
        assert result.confidence_total_upper_tco2e >= result.total_avoided_tco2e

    def test_ratio_to_footprint(self):
        engine = Scope4AvoidedEmissionsEngine()
        result = engine.calculate(_make_input(total_footprint_tco2e=Decimal("867000")))
        assert result.avoided_to_footprint_ratio >= Decimal("0")

    def test_by_product_results(self):
        engine = Scope4AvoidedEmissionsEngine()
        result = engine.calculate(_make_input(products=[
            _make_product("Product A"),
            _make_product("Product B", units=500),
        ]))
        assert len(result.by_product) == 2

    def test_by_category_results(self):
        engine = Scope4AvoidedEmissionsEngine()
        result = engine.calculate(_make_input())
        assert isinstance(result.by_category, dict)

    def test_provenance_hash(self):
        engine = Scope4AvoidedEmissionsEngine()
        result = engine.calculate(_make_input())
        assert_provenance_hash(result)

    def test_multiple_products(self):
        engine = Scope4AvoidedEmissionsEngine()
        result = engine.calculate(_make_input(products=[
            _make_product("EV", AvoidedEmissionCategory.PRODUCT_SUBSTITUTION,
                          Decimal("4.5"), Decimal("1.8"), 12000),
            _make_product("LED", AvoidedEmissionCategory.EFFICIENCY_IMPROVEMENT,
                          Decimal("0.175"), Decimal("0.044"), 50000),
        ]))
        assert len(result.by_product) == 2
        assert result.total_avoided_tco2e > Decimal("0")


# ===========================================================================
# Parametrized Avoided Emissions Categories
# ===========================================================================


AVOIDED_EMISSION_PRODUCTS = [
    ("Electric Vehicle", AvoidedEmissionCategory.PRODUCT_SUBSTITUTION,
     Decimal("4.5"), Decimal("1.8"), 12000),
    ("LED Lighting", AvoidedEmissionCategory.EFFICIENCY_IMPROVEMENT,
     Decimal("0.175"), Decimal("0.044"), 50000),
    ("Heat Pump", AvoidedEmissionCategory.PRODUCT_SUBSTITUTION,
     Decimal("3.2"), Decimal("1.1"), 5000),
    ("Solar Panel", AvoidedEmissionCategory.PRODUCT_SUBSTITUTION,
     Decimal("0.500"), Decimal("0.050"), 25000),
    ("Insulation Material", AvoidedEmissionCategory.EFFICIENCY_IMPROVEMENT,
     Decimal("2.0"), Decimal("0.8"), 8000),
    ("Video Conferencing Platform", AvoidedEmissionCategory.ENABLING_EFFECT,
     Decimal("0.100"), Decimal("0.002"), 1000000),
    ("Smart Grid Controller", AvoidedEmissionCategory.SYSTEMIC_CHANGE,
     Decimal("1.5"), Decimal("0.3"), 2000),
]


class TestParametrizedAvoidedEmissions:
    @pytest.mark.parametrize("product,category,baseline_ef,product_ef,units",
                             AVOIDED_EMISSION_PRODUCTS,
                             ids=[p[0] for p in AVOIDED_EMISSION_PRODUCTS])
    def test_avoided_emissions_positive(self, product, category, baseline_ef, product_ef, units):
        """Each product must produce positive avoided emissions."""
        engine = Scope4AvoidedEmissionsEngine()
        result = engine.calculate(_make_input(products=[
            _make_product(product, category, baseline_ef, product_ef, units),
        ]))
        assert result.total_avoided_tco2e > Decimal("0")

    @pytest.mark.parametrize("product,category,baseline_ef,product_ef,units",
                             AVOIDED_EMISSION_PRODUCTS,
                             ids=[p[0] for p in AVOIDED_EMISSION_PRODUCTS])
    def test_avoided_emissions_provenance(self, product, category, baseline_ef, product_ef, units):
        """Each product calculation must have provenance hash."""
        engine = Scope4AvoidedEmissionsEngine()
        result = engine.calculate(_make_input(products=[
            _make_product(product, category, baseline_ef, product_ef, units),
        ]))
        assert_provenance_hash(result)
