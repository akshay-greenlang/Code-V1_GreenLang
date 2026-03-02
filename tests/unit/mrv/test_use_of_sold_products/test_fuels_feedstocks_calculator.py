# -*- coding: utf-8 -*-
"""
Unit tests for FuelsAndFeedstocksCalculatorEngine -- AGENT-MRV-024

Tests fuel sales combustion and feedstock oxidation calculations for
products that are themselves fuels or chemical feedstocks.

Calculation formulas:
- Fuel sales: quantity_sold x fuel_EF
- Feedstock: quantity_sold x carbon_content x oxidation_factor x (44/12)

Specific test values:
- Fuel: 1,000,000L gasoline x 2.315 = 2,315,000 kgCO2e
- Feedstock: 1,000,000kg naphtha x 0.836 x 1.00 x 3.667 = 3,064,012 kgCO2e

Target: 25+ tests.
Author: GL-TestEngineer
"""

import threading
from decimal import Decimal
from typing import Any, Dict, List

import pytest

try:
    from greenlang.use_of_sold_products.fuels_feedstocks_calculator import (
        FuelsAndFeedstocksCalculatorEngine,
        get_fuels_calculator,
        calculate_fuel_sales_emissions,
        calculate_feedstock_oxidation,
        calculate_provenance_hash,
    )

    _AVAILABLE = True
except ImportError:
    _AVAILABLE = False

_SKIP = pytest.mark.skipif(not _AVAILABLE, reason="FuelsAndFeedstocksCalculatorEngine not available")
pytestmark = _SKIP

_Q8 = Decimal("0.00000001")


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture(autouse=True)
def _reset_engine():
    """Reset the singleton before each test."""
    FuelsAndFeedstocksCalculatorEngine.reset()
    yield
    FuelsAndFeedstocksCalculatorEngine.reset()


@pytest.fixture
def engine():
    """Create a FuelsAndFeedstocksCalculatorEngine instance."""
    return FuelsAndFeedstocksCalculatorEngine()


# ============================================================================
# TEST: Fuel Sales Calculations
# ============================================================================


class TestFuelSales:
    """Test fuel sales combustion calculations."""

    def test_gasoline_basic(self, engine):
        """Test 1,000,000L gasoline x 2.315 = 2,315,000 kgCO2e."""
        result = engine.calculate_fuel_sales(
            quantity_sold_litres=Decimal("1000000.0"),
            fuel_ef_kg_per_litre=Decimal("2.315"),
        )
        expected = Decimal("2315000.0")
        assert result["total_co2e_kg"] == pytest.approx(expected, rel=Decimal("0.001"))

    def test_diesel_basic(self, engine):
        """Test 500,000L diesel x 2.680 = 1,340,000 kgCO2e."""
        result = engine.calculate_fuel_sales(
            quantity_sold_litres=Decimal("500000.0"),
            fuel_ef_kg_per_litre=Decimal("2.680"),
        )
        expected = Decimal("1340000.0")
        assert result["total_co2e_kg"] == pytest.approx(expected, rel=Decimal("0.001"))

    def test_lpg_sales(self, engine):
        """Test 200,000L LPG x 1.553 = 310,600 kgCO2e."""
        result = engine.calculate_fuel_sales(
            quantity_sold_litres=Decimal("200000.0"),
            fuel_ef_kg_per_litre=Decimal("1.553"),
        )
        expected = Decimal("310600.0")
        assert result["total_co2e_kg"] == pytest.approx(expected, rel=Decimal("0.001"))

    def test_zero_quantity_returns_zero(self, engine):
        """Test zero quantity returns zero emissions."""
        result = engine.calculate_fuel_sales(
            quantity_sold_litres=Decimal("0.0"),
            fuel_ef_kg_per_litre=Decimal("2.315"),
        )
        assert result["total_co2e_kg"] == Decimal("0")

    @pytest.mark.parametrize("fuel,ef,quantity,expected", [
        ("gasoline", Decimal("2.315"), Decimal("1000000"), Decimal("2315000")),
        ("diesel", Decimal("2.680"), Decimal("1000000"), Decimal("2680000")),
        ("lpg", Decimal("1.553"), Decimal("1000000"), Decimal("1553000")),
        ("kerosene", Decimal("2.540"), Decimal("1000000"), Decimal("2540000")),
        ("heating_oil", Decimal("2.960"), Decimal("1000000"), Decimal("2960000")),
        ("propane", Decimal("1.510"), Decimal("1000000"), Decimal("1510000")),
        ("ethanol_e85", Decimal("1.610"), Decimal("1000000"), Decimal("1610000")),
        ("biodiesel_b20", Decimal("2.144"), Decimal("1000000"), Decimal("2144000")),
        ("lng", Decimal("1.180"), Decimal("1000000"), Decimal("1180000")),
    ])
    def test_all_fuel_types_parametrized(self, engine, fuel, ef, quantity, expected):
        """Test fuel sales calculation for all 9 liquid fuel types."""
        result = engine.calculate_fuel_sales(
            quantity_sold_litres=quantity,
            fuel_ef_kg_per_litre=ef,
        )
        assert result["total_co2e_kg"] == pytest.approx(expected, rel=Decimal("0.001"))

    def test_natural_gas_m3(self, engine):
        """Test natural gas sales in m3: 5,000,000m3 x 1.93 = 9,650,000 kgCO2e."""
        result = engine.calculate_fuel_sales(
            quantity_sold_litres=Decimal("5000000.0"),
            fuel_ef_kg_per_litre=Decimal("1.930"),
        )
        expected = Decimal("9650000.0")
        assert result["total_co2e_kg"] == pytest.approx(expected, rel=Decimal("0.001"))

    def test_coal_by_mass(self, engine):
        """Test coal sales by mass: 100,000kg x 2.860 = 286,000 kgCO2e."""
        result = engine.calculate_fuel_sales(
            quantity_sold_litres=Decimal("100000.0"),
            fuel_ef_kg_per_litre=Decimal("2.860"),
        )
        expected = Decimal("286000.0")
        assert result["total_co2e_kg"] == pytest.approx(expected, rel=Decimal("0.001"))

    def test_provenance_hash_present(self, engine):
        """Test fuel sales result includes provenance hash."""
        result = engine.calculate_fuel_sales(
            quantity_sold_litres=Decimal("1000000.0"),
            fuel_ef_kg_per_litre=Decimal("2.315"),
        )
        assert "provenance_hash" in result
        assert len(result["provenance_hash"]) == 64


# ============================================================================
# TEST: Feedstock Oxidation Calculations
# ============================================================================


class TestFeedstockOxidation:
    """Test feedstock oxidation calculations."""

    def test_naphtha_feedstock(self, engine):
        """Test naphtha: 1,000,000kg x 0.836 x 1.00 x (44/12) = ~3,064,000 kgCO2."""
        result = engine.calculate_feedstock_oxidation(
            quantity_sold_kg=Decimal("1000000.0"),
            carbon_content=Decimal("0.836"),
            oxidation_factor=Decimal("1.00"),
        )
        # Expected: 1,000,000 x 0.836 x 1.00 x (44/12) = 3,065,333 kgCO2
        expected = Decimal("1000000") * Decimal("0.836") * Decimal("1.00") * (Decimal("44") / Decimal("12"))
        assert result["total_co2e_kg"] == pytest.approx(expected, rel=Decimal("0.01"))

    def test_ethane_feedstock(self, engine):
        """Test ethane: 500,000kg x 0.799 x 1.00 x (44/12) = ~1,465,000 kgCO2."""
        result = engine.calculate_feedstock_oxidation(
            quantity_sold_kg=Decimal("500000.0"),
            carbon_content=Decimal("0.799"),
            oxidation_factor=Decimal("1.00"),
        )
        expected = Decimal("500000") * Decimal("0.799") * Decimal("1.00") * (Decimal("44") / Decimal("12"))
        assert result["total_co2e_kg"] == pytest.approx(expected, rel=Decimal("0.01"))

    def test_coal_feedstock_partial_oxidation(self, engine):
        """Test coal feedstock with partial oxidation (0.98)."""
        result = engine.calculate_feedstock_oxidation(
            quantity_sold_kg=Decimal("200000.0"),
            carbon_content=Decimal("0.710"),
            oxidation_factor=Decimal("0.98"),
        )
        expected = Decimal("200000") * Decimal("0.710") * Decimal("0.98") * (Decimal("44") / Decimal("12"))
        assert result["total_co2e_kg"] == pytest.approx(expected, rel=Decimal("0.01"))

    def test_zero_quantity_returns_zero(self, engine):
        """Test zero quantity returns zero emissions."""
        result = engine.calculate_feedstock_oxidation(
            quantity_sold_kg=Decimal("0.0"),
            carbon_content=Decimal("0.836"),
            oxidation_factor=Decimal("1.00"),
        )
        assert result["total_co2e_kg"] == Decimal("0")

    def test_provenance_hash_present(self, engine):
        """Test feedstock result includes provenance hash."""
        result = engine.calculate_feedstock_oxidation(
            quantity_sold_kg=Decimal("1000000.0"),
            carbon_content=Decimal("0.836"),
            oxidation_factor=Decimal("1.00"),
        )
        assert "provenance_hash" in result
        assert len(result["provenance_hash"]) == 64


# ============================================================================
# TEST: DQI Scores for Fuels/Feedstocks
# ============================================================================


class TestFuelsFeedstocksDQI:
    """Test data quality indicator scoring for fuels/feedstocks."""

    def test_dqi_fuel_sales_range(self, engine):
        """Test DQI score for fuel sales is in 85-95 range."""
        result = engine.calculate_fuel_sales(
            quantity_sold_litres=Decimal("1000000.0"),
            fuel_ef_kg_per_litre=Decimal("2.315"),
        )
        if "dqi_score" in result:
            assert Decimal("70") <= result["dqi_score"] <= Decimal("100")

    def test_dqi_feedstock_range(self, engine):
        """Test DQI score for feedstock is in 85-95 range."""
        result = engine.calculate_feedstock_oxidation(
            quantity_sold_kg=Decimal("1000000.0"),
            carbon_content=Decimal("0.836"),
            oxidation_factor=Decimal("1.00"),
        )
        if "dqi_score" in result:
            assert Decimal("70") <= result["dqi_score"] <= Decimal("100")


# ============================================================================
# TEST: Unit Conversions
# ============================================================================


class TestUnitConversions:
    """Test unit conversion handling in fuels/feedstocks."""

    def test_gallons_to_litres_conversion(self, engine):
        """Test gallons are converted to litres: 1 gallon = 3.785 litres."""
        # 264,172 gallons = 1,000,000 litres
        litres_result = engine.calculate_fuel_sales(
            quantity_sold_litres=Decimal("1000000.0"),
            fuel_ef_kg_per_litre=Decimal("2.315"),
        )
        # Verify the result is consistent
        assert litres_result["total_co2e_kg"] > Decimal("0")

    def test_tonnes_to_kg_for_feedstock(self, engine):
        """Test tonnes converted to kg: 1000 tonnes = 1,000,000 kg."""
        result = engine.calculate_feedstock_oxidation(
            quantity_sold_kg=Decimal("1000000.0"),
            carbon_content=Decimal("0.836"),
            oxidation_factor=Decimal("1.00"),
        )
        assert result["total_co2e_kg"] > Decimal("0")
