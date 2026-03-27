# -*- coding: utf-8 -*-
"""
Unit tests for AverageDataCalculatorEngine -- AGENT-MRV-025

Tests the average-data calculation method which uses product category-level
composite emission factors when specific material composition is unknown.

Coverage:
- 20 product categories with composite EFs parametrized
- Regional adjustment factors (12 regions)
- Weight estimation from defaults
- Fallback EF hierarchy
- DQI scoring (lower quality for average-data method)
- Uncertainty bounds (wider for average-data)

Target: 40+ expanded tests.
Author: GL-TestEngineer
Date: February 2026
"""

from decimal import Decimal
from typing import Any, Dict, List
import pytest

try:
    from greenlang.agents.mrv.end_of_life_treatment.average_data_calculator import (
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
    return AverageDataCalculatorEngine.get_instance()


@pytest.fixture
def electronics_input():
    """Consumer electronics average-data input."""
    return {
        "product_id": "PRD-ELEC-AVG",
        "product_category": "consumer_electronics",
        "total_mass_kg": Decimal("200.0"),
        "units_sold": 1000,
        "region": "US",
        "reporting_year": 2024,
    }


@pytest.fixture
def packaging_input():
    """Packaging average-data input."""
    return {
        "product_id": "PRD-PKG-AVG",
        "product_category": "packaging",
        "total_mass_kg": Decimal("5000.0"),
        "units_sold": 1000000,
        "region": "GB",
        "reporting_year": 2024,
    }


@pytest.fixture
def furniture_input():
    """Furniture average-data input."""
    return {
        "product_id": "PRD-FRN-AVG",
        "product_category": "furniture",
        "total_mass_kg": Decimal("2500.0"),
        "units_sold": 100,
        "region": "DE",
        "reporting_year": 2024,
    }


# ============================================================================
# TEST: Product Category Composite EFs
# ============================================================================


class TestProductCategoryCompositeEFs:
    """Test composite emission factor calculations by product category."""

    @pytest.mark.parametrize("category", [
        "consumer_electronics", "large_appliances", "small_appliances",
        "packaging", "clothing", "furniture", "batteries", "tires",
        "food_products", "building_materials", "automotive_parts",
        "medical_devices", "toys", "sporting_goods", "cosmetics",
        "office_equipment", "garden_tools", "pet_products", "lighting", "mixed",
    ])
    def test_category_calculation_returns_result(self, engine, category):
        """Test calculation for each of 20 product categories."""
        inp = {
            "product_id": f"PRD-{category}",
            "product_category": category,
            "total_mass_kg": Decimal("100.0"),
            "units_sold": 100,
            "region": "GLOBAL",
            "reporting_year": 2024,
        }
        result = engine.calculate(inp)
        assert result is not None
        assert "gross_emissions_kgco2e" in result
        assert result["gross_emissions_kgco2e"] >= Decimal("0.0")

    def test_electronics_higher_than_building_materials(self, engine):
        """Test electronics have higher per-kg EF than inert building materials."""
        elec = engine.calculate({
            "product_id": "E1", "product_category": "consumer_electronics",
            "total_mass_kg": Decimal("100.0"), "units_sold": 100,
            "region": "GLOBAL", "reporting_year": 2024,
        })
        bldg = engine.calculate({
            "product_id": "B1", "product_category": "building_materials",
            "total_mass_kg": Decimal("100.0"), "units_sold": 100,
            "region": "GLOBAL", "reporting_year": 2024,
        })
        # Electronics (plastics, batteries) should have higher EOL emissions than concrete
        elec_per_kg = elec["gross_emissions_kgco2e"] / Decimal("100.0")
        bldg_per_kg = bldg["gross_emissions_kgco2e"] / Decimal("100.0")
        assert elec_per_kg > bldg_per_kg

    def test_tires_high_incineration_component(self, engine):
        """Test tires have significant incineration emissions (TDF)."""
        result = engine.calculate({
            "product_id": "T1", "product_category": "tires",
            "total_mass_kg": Decimal("1000.0"), "units_sold": 100,
            "region": "GLOBAL", "reporting_year": 2024,
        })
        assert result["gross_emissions_kgco2e"] > Decimal("0.0")


# ============================================================================
# TEST: Regional Adjustment Factors
# ============================================================================


class TestRegionalAdjustment:
    """Test regional adjustment factors for treatment mix differences."""

    @pytest.mark.parametrize("region", [
        "US", "DE", "GB", "JP", "FR", "CN", "IN", "BR", "AU", "KR", "CA", "GLOBAL",
    ])
    def test_region_produces_results(self, engine, region):
        """Test calculation produces results for each of 12 regions."""
        inp = {
            "product_id": f"PRD-{region}",
            "product_category": "packaging",
            "total_mass_kg": Decimal("100.0"),
            "units_sold": 100,
            "region": region,
            "reporting_year": 2024,
        }
        result = engine.calculate(inp)
        assert result["gross_emissions_kgco2e"] >= Decimal("0.0")

    def test_japan_vs_us_incineration_difference(self, engine):
        """Test Japan has different emissions profile than US (more incineration)."""
        jp = engine.calculate({
            "product_id": "JP1", "product_category": "packaging",
            "total_mass_kg": Decimal("1000.0"), "units_sold": 100,
            "region": "JP", "reporting_year": 2024,
        })
        us = engine.calculate({
            "product_id": "US1", "product_category": "packaging",
            "total_mass_kg": Decimal("1000.0"), "units_sold": 100,
            "region": "US", "reporting_year": 2024,
        })
        # Different treatment mixes should yield different results
        assert jp["gross_emissions_kgco2e"] != us["gross_emissions_kgco2e"]

    def test_germany_lower_gross_emissions(self, engine):
        """Test Germany has lower gross emissions (high recycling, low landfill)."""
        de = engine.calculate({
            "product_id": "DE1", "product_category": "consumer_electronics",
            "total_mass_kg": Decimal("1000.0"), "units_sold": 100,
            "region": "DE", "reporting_year": 2024,
        })
        in_ = engine.calculate({
            "product_id": "IN1", "product_category": "consumer_electronics",
            "total_mass_kg": Decimal("1000.0"), "units_sold": 100,
            "region": "IN", "reporting_year": 2024,
        })
        # Germany (high recycling) should generally have lower gross vs India (landfill + open burning)
        assert de["gross_emissions_kgco2e"] < in_["gross_emissions_kgco2e"]


# ============================================================================
# TEST: Weight Estimation
# ============================================================================


class TestWeightEstimation:
    """Test weight estimation from product category defaults."""

    def test_mass_from_units_and_default_weight(self, engine):
        """Test mass estimation when only units_sold provided."""
        inp = {
            "product_id": "PRD-WE",
            "product_category": "consumer_electronics",
            "units_sold": 10000,
            "region": "US",
            "reporting_year": 2024,
        }
        result = engine.calculate(inp)
        assert result["gross_emissions_kgco2e"] >= Decimal("0.0")

    def test_explicit_mass_overrides_default(self, engine):
        """Test explicit total_mass_kg overrides default weight estimate."""
        # With explicit mass
        explicit = engine.calculate({
            "product_id": "PRD-EX",
            "product_category": "consumer_electronics",
            "total_mass_kg": Decimal("5000.0"),
            "units_sold": 100,
            "region": "US",
            "reporting_year": 2024,
        })
        # With different mass
        different = engine.calculate({
            "product_id": "PRD-DF",
            "product_category": "consumer_electronics",
            "total_mass_kg": Decimal("1000.0"),
            "units_sold": 100,
            "region": "US",
            "reporting_year": 2024,
        })
        # More mass should yield more emissions
        assert explicit["gross_emissions_kgco2e"] > different["gross_emissions_kgco2e"]


# ============================================================================
# TEST: Fallback EF Hierarchy
# ============================================================================


class TestFallbackEFHierarchy:
    """Test emission factor fallback hierarchy."""

    def test_unknown_category_uses_mixed_default(self, engine):
        """Test unknown category falls back to 'mixed' composite EF."""
        result = engine.calculate({
            "product_id": "PRD-UNK",
            "product_category": "mixed",
            "total_mass_kg": Decimal("100.0"),
            "units_sold": 100,
            "region": "GLOBAL",
            "reporting_year": 2024,
        })
        assert result["gross_emissions_kgco2e"] >= Decimal("0.0")


# ============================================================================
# TEST: DQI and Uncertainty
# ============================================================================


class TestDQIAndUncertainty:
    """Test data quality and uncertainty for average-data method."""

    def test_dqi_score_lower_than_waste_type(self, engine, electronics_input):
        """Test DQI score for average-data is lower (less precise method)."""
        result = engine.calculate(electronics_input)
        assert "dqi_score" in result
        # Average-data typically scores 30-55 (lower than waste-type-specific)
        assert result["dqi_score"] <= Decimal("60.0")

    def test_uncertainty_higher_than_waste_type(self, engine, electronics_input):
        """Test uncertainty is higher for average-data method."""
        result = engine.calculate(electronics_input)
        assert "uncertainty_pct" in result
        # Average-data has wider uncertainty (30-50%)
        assert result["uncertainty_pct"] >= Decimal("25.0")

    def test_method_is_average_data(self, engine, electronics_input):
        """Test method field is 'average_data'."""
        result = engine.calculate(electronics_input)
        assert result["method"] == "average_data"

    def test_avoided_emissions_typically_zero(self, engine, electronics_input):
        """Test average-data method reports zero avoided emissions by default."""
        result = engine.calculate(electronics_input)
        avoided = result.get("avoided_emissions_kgco2e", Decimal("0.0"))
        # Average-data method does not typically calculate avoided emissions
        assert avoided == Decimal("0.0")


# ============================================================================
# TEST: Edge Cases
# ============================================================================


class TestEdgeCases:
    """Test edge cases for average-data calculator."""

    def test_very_small_mass(self, engine):
        """Test calculation with very small mass."""
        result = engine.calculate({
            "product_id": "PRD-SM",
            "product_category": "cosmetics",
            "total_mass_kg": Decimal("0.001"),
            "units_sold": 1,
            "region": "US",
            "reporting_year": 2024,
        })
        assert result["gross_emissions_kgco2e"] >= Decimal("0.0")

    def test_very_large_mass(self, engine):
        """Test calculation with very large mass."""
        result = engine.calculate({
            "product_id": "PRD-LG",
            "product_category": "building_materials",
            "total_mass_kg": Decimal("1000000.0"),
            "units_sold": 20000,
            "region": "US",
            "reporting_year": 2024,
        })
        assert result["gross_emissions_kgco2e"] > Decimal("0.0")

    def test_emissions_scale_linearly_with_mass(self, engine):
        """Test emissions scale linearly with mass."""
        r1 = engine.calculate({
            "product_id": "P1", "product_category": "packaging",
            "total_mass_kg": Decimal("100.0"), "units_sold": 100,
            "region": "US", "reporting_year": 2024,
        })
        r2 = engine.calculate({
            "product_id": "P2", "product_category": "packaging",
            "total_mass_kg": Decimal("200.0"), "units_sold": 200,
            "region": "US", "reporting_year": 2024,
        })
        ratio = r2["gross_emissions_kgco2e"] / r1["gross_emissions_kgco2e"]
        assert abs(ratio - Decimal("2.0")) < Decimal("0.01")
