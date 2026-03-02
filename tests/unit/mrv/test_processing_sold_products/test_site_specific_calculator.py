# -*- coding: utf-8 -*-
"""
Unit tests for SiteSpecificCalculatorEngine -- AGENT-MRV-023

Tests the three site-specific calculation methods (direct, energy-based, fuel-based)
including known-value calculations, DQI scoring, uncertainty ranges, validation
warnings, multi-product aggregation, and the method dispatcher.

Target: 35+ tests.
Author: GL-TestEngineer
"""

import threading
from decimal import Decimal
from typing import Any, Dict, List

import pytest

try:
    from greenlang.processing_sold_products.site_specific_calculator import (
        SiteSpecificCalculatorEngine,
        get_site_specific_engine,
        CalculationResult,
        ProductBreakdown,
        DataQualityScore,
        UncertaintyResult,
        METHOD_DIRECT,
        METHOD_ENERGY,
        METHOD_FUEL,
        DQI_BASE_SCORES,
        UNCERTAINTY_FRACTIONS,
    )

    _AVAILABLE = True
except ImportError:
    _AVAILABLE = False

_SKIP = pytest.mark.skipif(not _AVAILABLE, reason="SiteSpecificCalculatorEngine not available")
pytestmark = _SKIP

_Q8 = Decimal("0.00000001")


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def engine():
    """Create a SiteSpecificCalculatorEngine instance."""
    return SiteSpecificCalculatorEngine()


@pytest.fixture
def direct_product():
    """Single product with customer-reported emission factor."""
    return {
        "product_id": "STEEL-001",
        "product_name": "Hot-rolled steel coil",
        "category": "METALS_FERROUS",
        "quantity_tonnes": "1000",
        "customer_ef": "280",
        "customer_id": "CUST-AUTO-01",
        "country": "US",
    }


@pytest.fixture
def energy_product():
    """Single product with energy consumption data."""
    return {
        "product_id": "STEEL-002",
        "product_name": "Steel sheet",
        "category": "METALS_FERROUS",
        "quantity_tonnes": "500",
        "energy_per_unit_kwh": "280",
        "customer_id": "CUST-MFG-01",
        "country": "US",
    }


@pytest.fixture
def fuel_product():
    """Single product with fuel consumption data."""
    return {
        "product_id": "STEEL-003",
        "product_name": "Steel billets",
        "category": "METALS_FERROUS",
        "quantity_tonnes": "500",
        "fuel_per_unit": "15",
        "fuel_type": "NATURAL_GAS",
        "customer_id": "CUST-FORGE-01",
        "country": "US",
    }


# ============================================================================
# TEST: Direct Method -- E = Q * EF_customer
# ============================================================================


class TestDirectMethod:
    """Test site-specific direct calculation method."""

    def test_direct_known_value_steel_1000t(self, engine, direct_product):
        """Known-value: 1000t x 280 kgCO2e/t = 280,000 kgCO2e."""
        result = engine.calculate_direct([direct_product], "ORG-001", 2024)
        assert isinstance(result, CalculationResult)
        assert result.total_co2e == Decimal("280000").quantize(_Q8)
        assert result.total_co2e_tonnes == Decimal("280").quantize(_Q8)

    def test_direct_method_field(self, engine, direct_product):
        """Test that the method field is correctly set."""
        result = engine.calculate_direct([direct_product], "ORG-001", 2024)
        assert result.method == METHOD_DIRECT

    def test_direct_product_breakdown_count(self, engine, direct_product):
        """Test that there is exactly one product breakdown."""
        result = engine.calculate_direct([direct_product], "ORG-001", 2024)
        assert result.product_count == 1
        assert len(result.product_breakdowns) == 1

    def test_direct_product_breakdown_fields(self, engine, direct_product):
        """Test product breakdown contains all expected fields."""
        result = engine.calculate_direct([direct_product], "ORG-001", 2024)
        bd = result.product_breakdowns[0]
        assert bd.product_id == "STEEL-001"
        assert bd.category == "METALS_FERROUS"
        assert bd.emissions_kg_co2e == Decimal("280000").quantize(_Q8)
        assert bd.method == METHOD_DIRECT
        assert len(bd.provenance_hash) == 64

    def test_direct_multi_product(self, engine):
        """Test direct calculation with multiple products."""
        products = [
            {"product_id": "P1", "category": "METALS_FERROUS",
             "quantity_tonnes": "100", "customer_ef": "280"},
            {"product_id": "P2", "category": "ELECTRONICS",
             "quantity_tonnes": "50", "customer_ef": "950"},
        ]
        result = engine.calculate_direct(products, "ORG-001", 2024)
        # P1: 100 * 280 = 28000, P2: 50 * 950 = 47500, Total = 75500
        assert result.total_co2e == Decimal("75500").quantize(_Q8)
        assert result.product_count == 2

    def test_direct_empty_products_raises(self, engine):
        """Test that empty products list raises ValueError."""
        with pytest.raises(ValueError, match="must not be empty"):
            engine.calculate_direct([], "ORG-001", 2024)

    def test_direct_missing_customer_ef_raises(self, engine):
        """Test that missing customer_ef raises ValueError."""
        product = {
            "product_id": "P1",
            "category": "METALS_FERROUS",
            "quantity_tonnes": "100",
        }
        with pytest.raises(ValueError, match="customer_ef"):
            engine.calculate_direct([product], "ORG-001", 2024)

    def test_direct_negative_quantity_raises(self, engine):
        """Test that negative quantity raises ValueError."""
        product = {
            "product_id": "P1",
            "category": "METALS_FERROUS",
            "quantity_tonnes": "-100",
            "customer_ef": "280",
        }
        with pytest.raises(ValueError, match="positive"):
            engine.calculate_direct([product], "ORG-001", 2024)

    def test_direct_zero_ef_produces_zero_emissions(self, engine):
        """Test that zero customer EF produces zero emissions."""
        product = {
            "product_id": "P1",
            "category": "METALS_FERROUS",
            "quantity_tonnes": "100",
            "customer_ef": "0",
        }
        result = engine.calculate_direct([product], "ORG-001", 2024)
        assert result.total_co2e == Decimal("0").quantize(_Q8)


# ============================================================================
# TEST: Energy-Based Method -- E = Q * EP * GridEF
# ============================================================================


class TestEnergyBasedMethod:
    """Test site-specific energy-based calculation method."""

    def test_energy_known_value_500t(self, engine, energy_product):
        """Known-value: 500t x 280 kWh/t x 0.417 kgCO2e/kWh = 58,380 kgCO2e."""
        result = engine.calculate_energy_based([energy_product], "ORG-001", 2024)
        assert isinstance(result, CalculationResult)
        assert result.total_co2e == Decimal("58380").quantize(_Q8)

    def test_energy_method_field(self, engine, energy_product):
        """Test that the method field is correctly set."""
        result = engine.calculate_energy_based([energy_product], "ORG-001", 2024)
        assert result.method == METHOD_ENERGY

    def test_energy_global_fallback_when_no_country(self, engine):
        """Test that GLOBAL grid EF is used when no country is provided."""
        product = {
            "product_id": "P1",
            "category": "METALS_FERROUS",
            "quantity_tonnes": "100",
            "energy_per_unit_kwh": "100",
        }
        result = engine.calculate_energy_based([product], "ORG-001", 2024)
        # 100 * 100 * 0.475 (GLOBAL) = 4750
        assert result.total_co2e == Decimal("4750").quantize(_Q8)

    def test_energy_de_grid_ef(self, engine):
        """Test calculation with Germany grid EF."""
        product = {
            "product_id": "P1",
            "category": "PLASTICS_THERMOPLASTIC",
            "quantity_tonnes": "200",
            "energy_per_unit_kwh": "520",
            "country": "DE",
        }
        result = engine.calculate_energy_based([product], "ORG-001", 2024)
        # 200 * 520 * 0.348 = 36,192
        assert result.total_co2e == Decimal("36192").quantize(_Q8)

    def test_energy_empty_products_raises(self, engine):
        """Test that empty products list raises ValueError."""
        with pytest.raises(ValueError, match="must not be empty"):
            engine.calculate_energy_based([], "ORG-001", 2024)

    def test_energy_missing_field_raises(self, engine):
        """Test that missing energy_per_unit_kwh raises ValueError."""
        product = {
            "product_id": "P1",
            "category": "METALS_FERROUS",
            "quantity_tonnes": "100",
        }
        with pytest.raises(ValueError, match="energy_per_unit_kwh"):
            engine.calculate_energy_based([product], "ORG-001", 2024)


# ============================================================================
# TEST: Fuel-Based Method -- E = Q * FP * FuelEF
# ============================================================================


class TestFuelBasedMethod:
    """Test site-specific fuel-based calculation method."""

    def test_fuel_known_value_500t_natural_gas(self, engine, fuel_product):
        """Known-value: 500t x 15 units/t x 2.024 kgCO2e/unit = 15,180 kgCO2e."""
        result = engine.calculate_fuel_based([fuel_product], "ORG-001", 2024)
        assert isinstance(result, CalculationResult)
        assert result.total_co2e == Decimal("15180").quantize(_Q8)

    def test_fuel_method_field(self, engine, fuel_product):
        """Test that the method field is correctly set."""
        result = engine.calculate_fuel_based([fuel_product], "ORG-001", 2024)
        assert result.method == METHOD_FUEL

    def test_fuel_diesel_ef(self, engine):
        """Test fuel-based calculation with diesel."""
        product = {
            "product_id": "P1",
            "category": "METALS_FERROUS",
            "quantity_tonnes": "100",
            "fuel_per_unit": "10",
            "fuel_type": "DIESEL",
        }
        result = engine.calculate_fuel_based([product], "ORG-001", 2024)
        # 100 * 10 * 2.706 = 2706
        assert result.total_co2e == Decimal("2706").quantize(_Q8)

    def test_fuel_default_natural_gas_when_no_type(self, engine):
        """Test that default fuel type is NATURAL_GAS."""
        product = {
            "product_id": "P1",
            "category": "METALS_FERROUS",
            "quantity_tonnes": "100",
            "fuel_per_unit": "10",
        }
        result = engine.calculate_fuel_based([product], "ORG-001", 2024)
        # 100 * 10 * 2.024 = 2024
        assert result.total_co2e == Decimal("2024").quantize(_Q8)

    def test_fuel_empty_products_raises(self, engine):
        """Test that empty products list raises ValueError."""
        with pytest.raises(ValueError, match="must not be empty"):
            engine.calculate_fuel_based([], "ORG-001", 2024)


# ============================================================================
# TEST: DQI Scoring
# ============================================================================


class TestDQIScoring:
    """Test data quality indicator scoring."""

    def test_dqi_direct_score_90(self, engine):
        """Test that direct method has DQI score of at least 90."""
        product = {"product_id": "P1", "country": "US"}
        dqi = engine.compute_dqi_score(product, METHOD_DIRECT)
        assert dqi.overall_score >= Decimal("90")
        assert dqi.classification == "Excellent"
        assert dqi.tier == "Tier 1"

    def test_dqi_energy_score_80(self, engine):
        """Test that energy method has DQI score of at least 80."""
        product = {"product_id": "P1", "country": "US"}
        dqi = engine.compute_dqi_score(product, METHOD_ENERGY)
        assert dqi.overall_score >= Decimal("80")

    def test_dqi_fuel_score_75(self, engine):
        """Test that fuel method has DQI score of at least 75."""
        product = {"product_id": "P1", "country": "US"}
        dqi = engine.compute_dqi_score(product, METHOD_FUEL)
        assert dqi.overall_score >= Decimal("75")

    def test_dqi_dimensions_present(self, engine):
        """Test that all 5 DQI dimensions are present."""
        product = {"product_id": "P1"}
        dqi = engine.compute_dqi_score(product, METHOD_DIRECT)
        expected_dims = {"representativeness", "completeness", "temporal",
                         "geographical", "technological"}
        assert set(dqi.dimensions.keys()) == expected_dims


# ============================================================================
# TEST: Uncertainty
# ============================================================================


class TestUncertainty:
    """Test uncertainty quantification."""

    def test_uncertainty_direct_10_pct(self, engine):
        """Test direct method uncertainty: +/-10% at 95% CI."""
        unc = engine.compute_uncertainty(Decimal("1000"), METHOD_DIRECT)
        assert unc.ci_lower == Decimal("900").quantize(_Q8)
        assert unc.ci_upper == Decimal("1100").quantize(_Q8)
        assert unc.half_width_fraction == Decimal("0.10").quantize(_Q8)

    def test_uncertainty_energy_15_pct(self, engine):
        """Test energy method uncertainty: +/-15% at 95% CI."""
        unc = engine.compute_uncertainty(Decimal("1000"), METHOD_ENERGY)
        assert unc.ci_lower == Decimal("850").quantize(_Q8)
        assert unc.ci_upper == Decimal("1150").quantize(_Q8)

    def test_uncertainty_fuel_20_pct(self, engine):
        """Test fuel method uncertainty: +/-20% at 95% CI."""
        unc = engine.compute_uncertainty(Decimal("1000"), METHOD_FUEL)
        assert unc.ci_lower == Decimal("800").quantize(_Q8)
        assert unc.ci_upper == Decimal("1200").quantize(_Q8)

    def test_uncertainty_lower_bound_floors_at_zero(self, engine):
        """Test that CI lower bound does not go below zero."""
        unc = engine.compute_uncertainty(Decimal("1"), METHOD_FUEL)
        assert unc.ci_lower >= Decimal("0")


# ============================================================================
# TEST: Validation Warnings
# ============================================================================


class TestValidationWarnings:
    """Test the validate_site_specific_data method."""

    def test_empty_list_warning(self, engine):
        """Test that empty product list produces a warning."""
        warnings = engine.validate_site_specific_data([])
        assert any("empty" in w.lower() for w in warnings)

    def test_missing_product_id_warning(self, engine):
        """Test warning for missing product_id."""
        products = [{"category": "METALS_FERROUS", "quantity_tonnes": "100"}]
        warnings = engine.validate_site_specific_data(products)
        assert any("product_id" in w.lower() for w in warnings)

    def test_duplicate_product_id_warning(self, engine):
        """Test warning for duplicate product_ids."""
        products = [
            {"product_id": "P1", "category": "METALS_FERROUS", "quantity_tonnes": "100"},
            {"product_id": "P1", "category": "METALS_FERROUS", "quantity_tonnes": "200"},
        ]
        warnings = engine.validate_site_specific_data(products)
        assert any("duplicate" in w.lower() for w in warnings)

    def test_negative_quantity_warning(self, engine):
        """Test warning for negative quantity."""
        products = [
            {"product_id": "P1", "category": "METALS_FERROUS", "quantity_tonnes": "-100"}
        ]
        warnings = engine.validate_site_specific_data(products)
        assert any("positive" in w.lower() for w in warnings)

    def test_unknown_country_warning(self, engine):
        """Test warning for unmapped country code."""
        products = [
            {"product_id": "P1", "category": "METALS_FERROUS",
             "quantity_tonnes": "100", "country": "ZZ"}
        ]
        warnings = engine.validate_site_specific_data(products)
        assert any("not mapped" in w.lower() for w in warnings)

    def test_valid_product_no_warnings(self, engine, direct_product):
        """Test that a valid product produces no warnings."""
        warnings = engine.validate_site_specific_data([direct_product])
        assert len(warnings) == 0


# ============================================================================
# TEST: Method Dispatcher
# ============================================================================


class TestMethodDispatcher:
    """Test the calculate() method dispatcher."""

    def test_dispatch_to_direct(self, engine, direct_product):
        """Test dispatching to direct method via short name."""
        result = engine.calculate([direct_product], "direct", "ORG-001", 2024)
        assert result.method == METHOD_DIRECT

    def test_dispatch_to_energy(self, engine, energy_product):
        """Test dispatching to energy method."""
        result = engine.calculate([energy_product], "energy", "ORG-001", 2024)
        assert result.method == METHOD_ENERGY

    def test_dispatch_to_fuel(self, engine, fuel_product):
        """Test dispatching to fuel method."""
        result = engine.calculate([fuel_product], "fuel", "ORG-001", 2024)
        assert result.method == METHOD_FUEL

    def test_dispatch_unknown_raises(self, engine, direct_product):
        """Test that unknown method raises ValueError."""
        with pytest.raises(ValueError, match="Unknown calculation method"):
            engine.calculate([direct_product], "unknown_method", "ORG-001", 2024)


# ============================================================================
# TEST: Engine Status and Singleton
# ============================================================================


class TestEngineStatus:
    """Test engine status and singleton behavior."""

    def test_singleton(self, engine):
        """Test that SiteSpecificCalculatorEngine is a singleton."""
        engine2 = SiteSpecificCalculatorEngine()
        assert engine is engine2

    def test_engine_status_fields(self, engine):
        """Test engine status contains required fields."""
        status = engine.get_engine_status()
        assert status["agent_id"] == "GL-MRV-S3-010"
        assert status["version"] == "1.0.0"
        assert "supported_methods" in status
        assert len(status["supported_methods"]) == 3

    def test_calculation_count_increments(self, engine, direct_product):
        """Test that calculation_count increments on calculate_direct."""
        initial = engine.calculation_count
        engine.calculate_direct([direct_product], "ORG-001", 2024)
        assert engine.calculation_count == initial + 1

    def test_provenance_hash_is_valid(self, engine, direct_product):
        """Test that result provenance_hash is 64-char hex."""
        result = engine.calculate_direct([direct_product], "ORG-001", 2024)
        h = result.provenance_hash
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_processing_time_positive(self, engine, direct_product):
        """Test that processing_time_ms is positive."""
        result = engine.calculate_direct([direct_product], "ORG-001", 2024)
        assert result.processing_time_ms > Decimal("0")
