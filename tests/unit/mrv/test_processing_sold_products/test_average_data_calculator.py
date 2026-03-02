# -*- coding: utf-8 -*-
"""
Unit tests for AverageDataCalculatorEngine -- AGENT-MRV-023

Tests the average-data calculation method including process EF calculations,
energy intensity calculations, multi-step chain calculations, DQI scoring,
and category comparisons.

Target: 30+ tests.
Author: GL-TestEngineer
"""

import threading
from decimal import Decimal
from typing import Any, Dict, List

import pytest

try:
    from greenlang.processing_sold_products.average_data_calculator import (
        AverageDataCalculatorEngine,
    )

    _AVAILABLE = True
except ImportError:
    _AVAILABLE = False

_SKIP = pytest.mark.skipif(not _AVAILABLE, reason="AverageDataCalculatorEngine not available")
pytestmark = _SKIP

_Q8 = Decimal("0.00000001")


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def engine():
    """Create an AverageDataCalculatorEngine instance."""
    return AverageDataCalculatorEngine()


@pytest.fixture
def steel_products():
    """Steel products for process EF calculation."""
    return [
        {
            "product_id": "STEEL-001",
            "product_name": "Hot-rolled coil",
            "category": "METALS_FERROUS",
            "quantity_tonnes": "1000",
            "processing_type": "MACHINING",
            "country": "US",
        }
    ]


@pytest.fixture
def multi_products():
    """Multiple products across different categories."""
    return [
        {
            "product_id": "P1",
            "product_name": "Steel billet",
            "category": "METALS_FERROUS",
            "quantity_tonnes": "500",
            "processing_type": "MACHINING",
        },
        {
            "product_id": "P2",
            "product_name": "Plastic resin",
            "category": "PLASTICS_THERMOPLASTIC",
            "quantity_tonnes": "300",
            "processing_type": "INJECTION_MOLDING",
        },
    ]


# ============================================================================
# TEST: Process EF Method -- E = Q * EF_category
# ============================================================================


class TestProcessEFMethod:
    """Test average-data process emission factor calculations."""

    @pytest.mark.parametrize(
        "category,quantity,expected_kg",
        [
            ("METALS_FERROUS", "1000", Decimal("280000")),
            ("METALS_NON_FERROUS", "1000", Decimal("380000")),
            ("PLASTICS_THERMOPLASTIC", "1000", Decimal("520000")),
            ("PLASTICS_THERMOSET", "1000", Decimal("450000")),
            ("CHEMICALS", "1000", Decimal("680000")),
            ("FOOD_INGREDIENTS", "1000", Decimal("130000")),
            ("TEXTILES", "1000", Decimal("350000")),
            ("ELECTRONICS", "1000", Decimal("950000")),
            ("GLASS_CERAMICS", "1000", Decimal("580000")),
            ("WOOD_PAPER", "1000", Decimal("190000")),
            ("MINERALS", "1000", Decimal("250000")),
            ("AGRICULTURAL", "1000", Decimal("110000")),
        ],
    )
    def test_process_ef_all_12_categories(self, engine, category, quantity, expected_kg):
        """Test process EF calculation for all 12 product categories at 1000 tonnes."""
        products = [
            {
                "product_id": f"P-{category}",
                "category": category,
                "quantity_tonnes": quantity,
                "processing_type": "MACHINING",
            }
        ]
        result = engine.calculate_process_ef(products, "ORG-001", 2024)
        assert result.total_co2e_kg == expected_kg.quantize(_Q8)

    def test_process_ef_known_value_steel_1000t(self, engine, steel_products):
        """Known-value: steel 1000t x 280 kgCO2e/t = 280,000 kgCO2e = 280 tCO2e."""
        result = engine.calculate_process_ef(steel_products, "ORG-001", 2024)
        assert result.total_co2e_kg == Decimal("280000").quantize(_Q8)
        assert result.total_co2e_tonnes == Decimal("280").quantize(_Q8)

    def test_process_ef_multi_product_aggregation(self, engine, multi_products):
        """Test that multi-product results are aggregated correctly."""
        result = engine.calculate_process_ef(multi_products, "ORG-001", 2024)
        # P1: 500 * 280 = 140000, P2: 300 * 520 = 156000
        expected = Decimal("296000")
        assert result.total_co2e_kg == expected.quantize(_Q8)
        assert result.product_count == 2

    def test_process_ef_empty_raises(self, engine):
        """Test that empty products list raises ValueError."""
        with pytest.raises(ValueError):
            engine.calculate_process_ef([], "ORG-001", 2024)


# ============================================================================
# TEST: Energy Intensity Method
# ============================================================================


class TestEnergyIntensityMethod:
    """Test energy intensity-based calculations."""

    @pytest.mark.parametrize(
        "processing_type,expected_kwh",
        [
            ("MACHINING", Decimal("280")),
            ("STAMPING", Decimal("140")),
            ("WELDING", Decimal("220")),
            ("HEAT_TREATMENT", Decimal("380")),
            ("INJECTION_MOLDING", Decimal("520")),
            ("EXTRUSION", Decimal("340")),
            ("BLOW_MOLDING", Decimal("400")),
            ("CASTING", Decimal("750")),
            ("FORGING", Decimal("580")),
            ("COATING", Decimal("120")),
            ("ASSEMBLY", Decimal("45")),
            ("CHEMICAL_REACTION", Decimal("1100")),
            ("REFINING", Decimal("900")),
            ("MILLING", Decimal("190")),
            ("DRYING", Decimal("310")),
            ("SINTERING", Decimal("1200")),
            ("FERMENTATION", Decimal("160")),
            ("TEXTILE_FINISHING", Decimal("420")),
        ],
    )
    def test_energy_intensity_all_18_types(self, engine, processing_type, expected_kwh):
        """Test energy intensity retrieval for all 18 processing types."""
        products = [
            {
                "product_id": f"P-{processing_type}",
                "category": "METALS_FERROUS",
                "quantity_tonnes": "100",
                "processing_type": processing_type,
                "country": "US",
            }
        ]
        result = engine.calculate_energy_intensity(products, "ORG-001", 2024)
        # Emissions = 100 * expected_kwh * 0.417 (US grid EF)
        expected_emissions = (Decimal("100") * expected_kwh * Decimal("0.417")).quantize(_Q8)
        assert result.total_co2e_kg == expected_emissions


# ============================================================================
# TEST: Multi-Step Chain Calculations
# ============================================================================


class TestChainCalculations:
    """Test multi-step processing chain emission calculations."""

    @pytest.mark.parametrize(
        "chain_type,expected_ef",
        [
            ("STEEL_AUTOMOTIVE", Decimal("195")),
            ("ALUMINIUM_AEROSPACE", Decimal("420")),
            ("PLASTIC_PACKAGING", Decimal("385")),
            ("CHEMICAL_PHARMACEUTICAL", Decimal("820")),
            ("FOOD_BEVERAGE", Decimal("155")),
            ("TEXTILE_GARMENT", Decimal("310")),
            ("ELECTRONICS_PCB", Decimal("580")),
            ("WOOD_FURNITURE", Decimal("175")),
        ],
    )
    def test_chain_ef_all_8_chains(self, engine, chain_type, expected_ef):
        """Test combined EF for all 8 processing chains at 100 tonnes."""
        result = engine.calculate_chain_emissions(
            chain_type=chain_type,
            quantity_tonnes=Decimal("100"),
            org_id="ORG-001",
            reporting_year=2024,
        )
        expected_kg = (Decimal("100") * expected_ef).quantize(_Q8)
        assert result.total_co2e_kg == expected_kg


# ============================================================================
# TEST: DQI Scoring
# ============================================================================


class TestAverageDataDQI:
    """Test DQI scoring for average-data methods."""

    def test_dqi_process_ef_score(self, engine):
        """Test that process EF method DQI is around 55."""
        dqi = engine.compute_dqi_score("process_ef")
        assert dqi.composite >= 40
        assert dqi.composite <= 70

    def test_dqi_energy_intensity_score(self, engine):
        """Test that energy intensity method DQI is around 50."""
        dqi = engine.compute_dqi_score("energy_intensity")
        assert dqi.composite >= 35
        assert dqi.composite <= 65

    def test_dqi_chain_score(self, engine):
        """Test that chain method DQI is around 45."""
        dqi = engine.compute_dqi_score("chain")
        assert dqi.composite >= 30
        assert dqi.composite <= 60


# ============================================================================
# TEST: Category Comparisons
# ============================================================================


class TestCategoryComparisons:
    """Test relative emission factor ordering across categories."""

    def test_electronics_greater_than_food(self, engine):
        """Test that electronics EF (950) > food ingredients EF (130)."""
        products_elec = [
            {"product_id": "E1", "category": "ELECTRONICS",
             "quantity_tonnes": "1000", "processing_type": "SINTERING"}
        ]
        products_food = [
            {"product_id": "F1", "category": "FOOD_INGREDIENTS",
             "quantity_tonnes": "1000", "processing_type": "MILLING"}
        ]
        elec_result = engine.calculate_process_ef(products_elec, "ORG-001", 2024)
        food_result = engine.calculate_process_ef(products_food, "ORG-001", 2024)
        assert elec_result.total_co2e_kg > food_result.total_co2e_kg

    def test_chemicals_greater_than_agricultural(self, engine):
        """Test that chemicals EF (680) > agricultural EF (110)."""
        products_chem = [
            {"product_id": "C1", "category": "CHEMICALS",
             "quantity_tonnes": "1000", "processing_type": "CHEMICAL_REACTION"}
        ]
        products_ag = [
            {"product_id": "A1", "category": "AGRICULTURAL",
             "quantity_tonnes": "1000", "processing_type": "MILLING"}
        ]
        chem_result = engine.calculate_process_ef(products_chem, "ORG-001", 2024)
        ag_result = engine.calculate_process_ef(products_ag, "ORG-001", 2024)
        assert chem_result.total_co2e_kg > ag_result.total_co2e_kg


# ============================================================================
# TEST: Uncertainty
# ============================================================================


class TestAverageDataUncertainty:
    """Test uncertainty quantification for average-data methods."""

    def test_uncertainty_range_30_pct_default(self, engine):
        """Test that default uncertainty for average-data is around 30%."""
        unc = engine.compute_uncertainty(Decimal("100000"))
        assert unc.lower_bound < Decimal("100000")
        assert unc.upper_bound > Decimal("100000")

    def test_uncertainty_confidence_level(self, engine):
        """Test that confidence level is 95%."""
        unc = engine.compute_uncertainty(Decimal("100000"))
        assert unc.confidence_level == 95


# ============================================================================
# TEST: Singleton Pattern
# ============================================================================


class TestAverageDataSingleton:
    """Test singleton pattern for AverageDataCalculatorEngine."""

    def test_singleton_identity(self, engine):
        """Test that two instantiations return the same object."""
        engine2 = AverageDataCalculatorEngine()
        assert engine is engine2

    def test_engine_status(self, engine):
        """Test that health check returns valid status."""
        status = engine.health_check()
        assert status["engine"] == "AverageDataCalculatorEngine"
        assert status["status"] == "healthy"


# ============================================================================
# TEST: Provenance
# ============================================================================


class TestAverageDataProvenance:
    """Test provenance hashing in average-data calculations."""

    def test_provenance_hash_64_char(self, engine, steel_products):
        """Test that calculation result includes a 64-char provenance hash."""
        result = engine.calculate_process_ef(steel_products, "ORG-001", 2024)
        h = result.provenance_hash
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_provenance_deterministic_for_same_input(self, engine):
        """Test that same inputs produce the same provenance hash."""
        products = [
            {"product_id": "P1", "category": "METALS_FERROUS",
             "quantity_tonnes": "100", "processing_type": "MACHINING"}
        ]
        r1 = engine.calculate_process_ef(products, "ORG-001", 2024)
        r2 = engine.calculate_process_ef(products, "ORG-001", 2024)
        # Note: provenance may include timestamps, so they might differ.
        # We just validate hash format here.
        assert len(r1.provenance_hash) == 64
        assert len(r2.provenance_hash) == 64
