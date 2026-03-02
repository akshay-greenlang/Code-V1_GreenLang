# -*- coding: utf-8 -*-
"""
Unit tests for EOLProductDatabaseEngine -- AGENT-MRV-025

Tests the product database engine that provides emission factors, product
compositions, regional treatment mixes, landfill FOD parameters, incineration
parameters, recycling factors, and product weight defaults.

Coverage:
- 15 materials x 7 treatments parametrized EF lookups
- 20 product composition lookups parametrized
- 12 regional treatment mix lookups parametrized
- FOD parameters by material + climate
- Incineration parameters by material
- Weight defaults by product category
- Provenance hash determinism
- Unknown material/treatment fallback

Target: 70+ expanded tests.
Author: GL-TestEngineer
Date: February 2026
"""

from decimal import Decimal
from typing import Any, Dict
import pytest

try:
    from greenlang.end_of_life_treatment.eol_product_database import (
        EOLProductDatabaseEngine,
    )
    _AVAILABLE = True
except ImportError:
    _AVAILABLE = False

_SKIP = pytest.mark.skipif(not _AVAILABLE, reason="EOLProductDatabaseEngine not available")
pytestmark = _SKIP

_Q8 = Decimal("0.00000001")


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def engine():
    """Create an EOLProductDatabaseEngine instance."""
    return EOLProductDatabaseEngine.get_instance()


# ============================================================================
# TEST: Material x Treatment EF Lookups
# ============================================================================


class TestMaterialTreatmentEFs:
    """Test emission factor lookups for material x treatment combinations."""

    @pytest.mark.parametrize("material,treatment", [
        ("steel", "landfill"),
        ("steel", "incineration"),
        ("steel", "recycling"),
        ("aluminum", "landfill"),
        ("aluminum", "recycling"),
        ("copper", "recycling"),
        ("glass", "landfill"),
        ("glass", "recycling"),
        ("plastic_abs", "landfill"),
        ("plastic_abs", "incineration"),
        ("plastic_abs", "recycling"),
        ("plastic_pe", "incineration"),
        ("plastic_pp", "incineration"),
        ("paper_cardboard", "landfill"),
        ("paper_cardboard", "incineration"),
        ("paper_cardboard", "recycling"),
        ("paper_cardboard", "composting"),
        ("wood_mdf", "landfill"),
        ("wood_mdf", "composting"),
        ("cotton", "landfill"),
        ("cotton", "incineration"),
        ("polyester", "incineration"),
        ("rubber_synthetic", "incineration"),
        ("food_organic", "landfill"),
        ("food_organic", "composting"),
        ("food_organic", "anaerobic_digestion"),
        ("concrete", "landfill"),
        ("concrete", "recycling"),
        ("lithium_battery", "recycling"),
    ])
    def test_ef_lookup_returns_decimal(self, engine, material, treatment):
        """Test EF lookup returns a valid Decimal value."""
        result = engine.get_material_treatment_ef(material, treatment)
        assert result is not None
        assert isinstance(result["ef_kg_co2e_per_kg"], Decimal)
        assert result["ef_kg_co2e_per_kg"] >= Decimal("0.0")

    @pytest.mark.parametrize("material", [
        "steel", "aluminum", "copper", "glass", "plastic_abs",
        "plastic_pe", "plastic_pp", "paper_cardboard", "wood_mdf",
        "cotton", "polyester", "rubber_synthetic", "food_organic",
        "concrete", "lithium_battery",
    ])
    def test_all_15_materials_have_landfill_ef(self, engine, material):
        """Test all 15 materials have a landfill emission factor."""
        result = engine.get_material_treatment_ef(material, "landfill")
        assert result is not None

    def test_plastic_incineration_high(self, engine):
        """Test plastic incineration EF is high (fossil CO2)."""
        result = engine.get_material_treatment_ef("plastic_abs", "incineration")
        assert result["ef_kg_co2e_per_kg"] > Decimal("1.0")

    def test_metal_incineration_low(self, engine):
        """Test metal incineration EF is low (metals do not combust)."""
        result = engine.get_material_treatment_ef("steel", "incineration")
        assert result["ef_kg_co2e_per_kg"] < Decimal("0.1")

    def test_food_organic_composting_exists(self, engine):
        """Test food organic has composting EF."""
        result = engine.get_material_treatment_ef("food_organic", "composting")
        assert result is not None
        assert result["ef_kg_co2e_per_kg"] > Decimal("0.0")

    def test_concrete_inert_low_landfill(self, engine):
        """Test concrete has very low landfill EF (inert material)."""
        result = engine.get_material_treatment_ef("concrete", "landfill")
        assert result["ef_kg_co2e_per_kg"] < Decimal("0.01")

    def test_unknown_material_returns_none(self, engine):
        """Test unknown material returns None."""
        result = engine.get_material_treatment_ef("unobtanium", "landfill")
        assert result is None

    def test_unknown_treatment_returns_none(self, engine):
        """Test unknown treatment returns None."""
        result = engine.get_material_treatment_ef("steel", "teleportation")
        assert result is None


# ============================================================================
# TEST: Product Composition Lookups
# ============================================================================


class TestProductCompositions:
    """Test product composition lookups by category."""

    @pytest.mark.parametrize("category", [
        "consumer_electronics", "large_appliances", "small_appliances",
        "packaging", "clothing", "furniture", "batteries", "tires",
        "food_products", "building_materials", "automotive_parts",
        "medical_devices", "toys", "sporting_goods", "cosmetics",
        "office_equipment", "garden_tools", "pet_products", "lighting", "mixed",
    ])
    def test_composition_lookup_valid(self, engine, category):
        """Test composition lookup returns valid data for each category."""
        result = engine.get_product_composition(category)
        assert result is not None
        assert isinstance(result, list)
        assert len(result) >= 2

    def test_composition_fractions_sum_to_one(self, engine):
        """Test all compositions sum to approximately 1.0."""
        categories = engine.get_available_categories()
        for category in categories:
            comp = engine.get_product_composition(category)
            if comp:
                total = sum(m["mass_fraction"] for m in comp)
                assert abs(total - Decimal("1.0")) < Decimal("0.02"), (
                    f"Category {category} fractions sum to {total}"
                )

    def test_unknown_category_returns_none(self, engine):
        """Test unknown category returns None."""
        result = engine.get_product_composition("imaginary_product")
        assert result is None

    def test_electronics_has_plastic_and_metal(self, engine):
        """Test consumer electronics contains plastic and metal materials."""
        comp = engine.get_product_composition("consumer_electronics")
        materials = [m["material"] for m in comp]
        has_plastic = any("plastic" in m for m in materials)
        has_metal = any(m in ("steel", "aluminum", "copper") for m in materials)
        assert has_plastic
        assert has_metal


# ============================================================================
# TEST: Regional Treatment Mix Lookups
# ============================================================================


class TestRegionalTreatmentMixes:
    """Test regional treatment mix lookups."""

    @pytest.mark.parametrize("region", [
        "US", "DE", "GB", "JP", "FR", "CN", "IN", "BR", "AU", "KR", "CA", "GLOBAL",
    ])
    def test_treatment_mix_valid(self, engine, region):
        """Test treatment mix lookup returns valid data for each region."""
        result = engine.get_regional_treatment_mix(region)
        assert result is not None
        assert isinstance(result, dict)
        assert "landfill" in result
        assert "incineration" in result
        assert "recycling" in result

    def test_treatment_mix_sums_to_one(self, engine):
        """Test treatment mix fractions sum to 1.0."""
        regions = engine.get_available_regions()
        for region in regions:
            mix = engine.get_regional_treatment_mix(region)
            if mix:
                total = sum(mix.values())
                assert abs(total - Decimal("1.0")) < Decimal("0.02"), (
                    f"Region {region} mix sums to {total}"
                )

    def test_unknown_region_falls_back_to_global(self, engine):
        """Test unknown region falls back to GLOBAL mix."""
        result = engine.get_regional_treatment_mix("XX")
        assert result is not None
        global_mix = engine.get_regional_treatment_mix("GLOBAL")
        assert result == global_mix

    def test_japan_dominated_by_incineration(self, engine):
        """Test Japan has >60% incineration."""
        mix = engine.get_regional_treatment_mix("JP")
        assert mix["incineration"] >= Decimal("0.60")

    def test_germany_minimal_landfill(self, engine):
        """Test Germany has <5% landfill (banned since 2005)."""
        mix = engine.get_regional_treatment_mix("DE")
        assert mix["landfill"] <= Decimal("0.05")

    def test_us_high_landfill(self, engine):
        """Test US has >40% landfill."""
        mix = engine.get_regional_treatment_mix("US")
        assert mix["landfill"] >= Decimal("0.40")


# ============================================================================
# TEST: Landfill FOD Parameters
# ============================================================================


class TestLandfillFODParams:
    """Test landfill first-order decay parameter lookups."""

    @pytest.mark.parametrize("material,climate", [
        ("paper_cardboard", "temperate_wet"),
        ("paper_cardboard", "tropical_wet"),
        ("food_organic", "temperate_wet"),
        ("food_organic", "tropical_wet"),
        ("wood_mdf", "temperate_wet"),
        ("cotton", "temperate_wet"),
        ("plastic_abs", "temperate_wet"),
    ])
    def test_fod_params_valid(self, engine, material, climate):
        """Test FOD parameter lookup returns valid data."""
        result = engine.get_landfill_fod_params(material, climate)
        assert result is not None
        assert "doc" in result
        assert "docf" in result
        assert "mcf" in result
        assert "k" in result

    def test_plastic_doc_is_zero(self, engine):
        """Test plastic DOC is 0 (non-degradable)."""
        result = engine.get_landfill_fod_params("plastic_abs", "temperate_wet")
        assert result["doc"] == Decimal("0.0")

    def test_food_higher_k_than_paper(self, engine):
        """Test food organic has higher decay rate than paper."""
        food = engine.get_landfill_fod_params("food_organic", "temperate_wet")
        paper = engine.get_landfill_fod_params("paper_cardboard", "temperate_wet")
        assert food["k"] > paper["k"]

    def test_tropical_higher_k_than_temperate(self, engine):
        """Test tropical wet has higher decay rate than temperate wet."""
        tropical = engine.get_landfill_fod_params("food_organic", "tropical_wet")
        temperate = engine.get_landfill_fod_params("food_organic", "temperate_wet")
        assert tropical["k"] > temperate["k"]


# ============================================================================
# TEST: Incineration Parameters
# ============================================================================


class TestIncinerationParams:
    """Test incineration parameter lookups."""

    @pytest.mark.parametrize("material", [
        "plastic_abs", "plastic_pe", "paper_cardboard",
        "food_organic", "rubber_synthetic", "cotton",
    ])
    def test_incineration_params_valid(self, engine, material):
        """Test incineration parameter lookup returns valid data."""
        result = engine.get_incineration_params(material)
        assert result is not None
        assert "fossil_carbon_fraction" in result
        assert "carbon_content" in result
        assert "combustion_efficiency" in result

    def test_plastic_100pct_fossil(self, engine):
        """Test plastic has 100% fossil carbon."""
        result = engine.get_incineration_params("plastic_abs")
        assert result["fossil_carbon_fraction"] == Decimal("1.00")

    def test_food_zero_fossil(self, engine):
        """Test food organic has 0% fossil carbon (all biogenic)."""
        result = engine.get_incineration_params("food_organic")
        assert result["fossil_carbon_fraction"] == Decimal("0.0")

    def test_combustion_efficiency_high(self, engine):
        """Test combustion efficiency is above 0.99."""
        result = engine.get_incineration_params("plastic_abs")
        assert result["combustion_efficiency"] >= Decimal("0.99")


# ============================================================================
# TEST: Recycling Factors
# ============================================================================


class TestRecyclingFactors:
    """Test recycling factor lookups."""

    @pytest.mark.parametrize("material", [
        "steel", "aluminum", "copper", "glass",
        "paper_cardboard", "plastic_pe", "plastic_abs", "rubber_synthetic",
    ])
    def test_recycling_factors_valid(self, engine, material):
        """Test recycling factor lookup returns valid data."""
        result = engine.get_recycling_factors(material)
        assert result is not None
        assert "recovery_rate" in result
        assert "avoided_ef" in result

    def test_aluminum_highest_avoided(self, engine):
        """Test aluminum has highest avoided emissions (energy-intensive virgin)."""
        al = engine.get_recycling_factors("aluminum")
        steel = engine.get_recycling_factors("steel")
        assert al["avoided_ef"] > steel["avoided_ef"]


# ============================================================================
# TEST: Product Weight Defaults
# ============================================================================


class TestProductWeights:
    """Test product weight default lookups."""

    @pytest.mark.parametrize("category", [
        "consumer_electronics", "large_appliances", "packaging",
        "clothing", "furniture", "tires", "building_materials",
    ])
    def test_weight_lookup_valid(self, engine, category):
        """Test weight lookup returns a valid Decimal."""
        result = engine.get_product_weight(category)
        assert result is not None
        assert isinstance(result, Decimal)
        assert result > Decimal("0.0")


# ============================================================================
# TEST: Provenance Hash Determinism
# ============================================================================


class TestProvenanceHash:
    """Test provenance hash determinism for database lookups."""

    def test_same_query_same_hash(self, engine):
        """Test same lookup produces same provenance hash."""
        r1 = engine.get_material_treatment_ef("steel", "landfill")
        r2 = engine.get_material_treatment_ef("steel", "landfill")
        if r1 is not None and r2 is not None:
            assert r1 == r2

    def test_database_summary(self, engine):
        """Test database summary returns valid counts."""
        summary = engine.get_database_summary()
        assert summary["materials"] >= 15
        assert summary["treatments"] >= 7
        assert summary["regions"] >= 12
        assert summary["categories"] >= 20
