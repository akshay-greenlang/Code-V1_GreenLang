# -*- coding: utf-8 -*-
"""
Unit tests for RetailScope3Engine -- PACK-014 CSRD Retail Engine 2
====================================================================

Tests all 15 Scope 3 categories with retail-specific prioritization,
calculation methods, data-quality scoring, and hotspot analysis.

Coverage target: 85%+
Total tests: ~41
"""

import importlib.util
import os
import sys

import pytest

# ---------------------------------------------------------------------------
# Dynamic module loading
# ---------------------------------------------------------------------------
ENGINES_DIR = os.path.join(os.path.dirname(__file__), "..", "engines")


def _load(name: str):
    path = os.path.join(ENGINES_DIR, f"{name}.py")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_m = _load("retail_scope3_engine")

RetailScope3Engine = _m.RetailScope3Engine
RetailScope3Input = _m.RetailScope3Input
RetailScope3Result = _m.RetailScope3Result
Scope3Category = _m.Scope3Category
CalculationMethod = _m.CalculationMethod
DataQualityLevel = _m.DataQualityLevel
ProductCategory = _m.ProductCategory
TransportMode = _m.TransportMode
WasteDisposalMethod = _m.WasteDisposalMethod
TravelMode = _m.TravelMode
CommuteMode = _m.CommuteMode
PurchasedGoodsData = _m.PurchasedGoodsData
CapitalGoodsData = _m.CapitalGoodsData
TransportData = _m.TransportData
WasteData = _m.WasteData
BusinessTravelData = _m.BusinessTravelData
CommuteData = _m.CommuteData
UsePhaseData = _m.UsePhaseData
EndOfLifeData = _m.EndOfLifeData
FranchiseData = _m.FranchiseData
CategoryBreakdown = _m.CategoryBreakdown
HotspotResult = _m.HotspotResult
DataQualitySummary = _m.DataQualitySummary
SPEND_EMISSION_FACTORS = _m.SPEND_EMISSION_FACTORS
TRANSPORT_EMISSION_FACTORS = _m.TRANSPORT_EMISSION_FACTORS
WASTE_EMISSION_FACTORS = _m.WASTE_EMISSION_FACTORS
TRAVEL_EMISSION_FACTORS = _m.TRAVEL_EMISSION_FACTORS
COMMUTE_EMISSION_FACTORS = _m.COMMUTE_EMISSION_FACTORS
USE_PHASE_ELECTRICITY = _m.USE_PHASE_ELECTRICITY
EOL_EMISSION_FACTORS = _m.EOL_EMISSION_FACTORS
RETAIL_SCOPE3_PRIORITY = _m.RETAIL_SCOPE3_PRIORITY
CATEGORY_NAMES = _m.CATEGORY_NAMES


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_input(**overrides) -> RetailScope3Input:
    """Build a minimal valid RetailScope3Input."""
    defaults = dict(
        organisation_id="ORG001",
        reporting_year=2025,
        retail_sub_sector="general_retail",
    )
    defaults.update(overrides)
    return RetailScope3Input(**defaults)


# ===================================================================
# TestInitialization
# ===================================================================


class TestInitialization:
    """Engine instantiation tests."""

    def test_default(self):
        engine = RetailScope3Engine()
        assert engine is not None

    def test_config_spend_factors(self):
        engine = RetailScope3Engine()
        assert len(engine._spend_factors) > 0

    def test_config_transport_factors(self):
        engine = RetailScope3Engine()
        assert TransportMode.ROAD in engine._transport_factors

    def test_config_priority_matrix(self):
        engine = RetailScope3Engine()
        assert "supermarket" in engine._priority_matrix


# ===================================================================
# TestScope3Categories
# ===================================================================


class TestScope3Categories:
    """Scope 3 category enum tests."""

    def test_all_15_defined(self):
        assert len(Scope3Category) == 15

    def test_enum_values(self):
        assert Scope3Category.CAT_1.value == "cat_1_purchased_goods_services"
        assert Scope3Category.CAT_15.value == "cat_15_investments"

    def test_cat1_dominant_for_supermarket(self):
        priority = RETAIL_SCOPE3_PRIORITY["supermarket"]
        assert priority[Scope3Category.CAT_1] == "CRITICAL"

    def test_retail_priority_matrix(self):
        assert len(RETAIL_SCOPE3_PRIORITY) >= 5

    def test_priority_by_subsector(self):
        assert "electronics_retail" in RETAIL_SCOPE3_PRIORITY
        priority = RETAIL_SCOPE3_PRIORITY["electronics_retail"]
        assert priority[Scope3Category.CAT_11] == "CRITICAL"


# ===================================================================
# TestCat1PurchasedGoods
# ===================================================================


class TestCat1PurchasedGoods:
    """Cat 1: Purchased Goods and Services tests."""

    def test_spend_based(self):
        engine = RetailScope3Engine()
        inp = _make_input(
            purchased_goods=[
                PurchasedGoodsData(
                    product_category=ProductCategory.FOOD_FRESH,
                    spend_eur=1_000_000.0,
                    calculation_method=CalculationMethod.SPEND_BASED,
                ),
            ],
        )
        result = engine.calculate_scope3(inp)
        cat1 = [c for c in result.category_breakdown if c.category == Scope3Category.CAT_1.value][0]
        # 1M EUR * 1850 tCO2e/M = 1850
        assert cat1.emissions_tco2e == pytest.approx(1850.0, rel=1e-2)

    def test_product_level(self):
        engine = RetailScope3Engine()
        inp = _make_input(
            purchased_goods=[
                PurchasedGoodsData(
                    product_category=ProductCategory.FOOD_FRESH,
                    sub_category="beef",
                    quantity_units=10_000.0,
                    quantity_unit_type="kg",
                    calculation_method=CalculationMethod.AVERAGE_DATA,
                    data_quality=DataQualityLevel.SCORE_3,
                ),
            ],
        )
        result = engine.calculate_scope3(inp)
        cat1 = [c for c in result.category_breakdown if c.category == Scope3Category.CAT_1.value][0]
        # 10000 kg * 0.0272 tCO2e/kg = 272
        assert cat1.emissions_tco2e == pytest.approx(272.0, rel=1e-2)

    def test_supplier_specific(self):
        engine = RetailScope3Engine()
        inp = _make_input(
            purchased_goods=[
                PurchasedGoodsData(
                    product_category=ProductCategory.ELECTRONICS,
                    supplier_id="SUP001",
                    supplier_name="ElectroCo",
                    supplier_emissions_tco2e=500.0,
                    calculation_method=CalculationMethod.SUPPLIER_SPECIFIC,
                    data_quality=DataQualityLevel.SCORE_1,
                ),
            ],
        )
        result = engine.calculate_scope3(inp)
        cat1 = [c for c in result.category_breakdown if c.category == Scope3Category.CAT_1.value][0]
        assert cat1.emissions_tco2e == pytest.approx(500.0, rel=1e-4)

    def test_zero_spend(self):
        engine = RetailScope3Engine()
        inp = _make_input(
            purchased_goods=[
                PurchasedGoodsData(
                    product_category=ProductCategory.OTHER,
                    spend_eur=0.0,
                ),
            ],
        )
        result = engine.calculate_scope3(inp)
        cat1 = [c for c in result.category_breakdown if c.category == Scope3Category.CAT_1.value][0]
        assert cat1.emissions_tco2e == pytest.approx(0.0, abs=1e-6)

    def test_multiple_categories(self):
        engine = RetailScope3Engine()
        inp = _make_input(
            purchased_goods=[
                PurchasedGoodsData(
                    product_category=ProductCategory.FOOD_FRESH,
                    spend_eur=500_000.0,
                ),
                PurchasedGoodsData(
                    product_category=ProductCategory.APPAREL,
                    spend_eur=300_000.0,
                ),
            ],
        )
        result = engine.calculate_scope3(inp)
        cat1 = [c for c in result.category_breakdown if c.category == Scope3Category.CAT_1.value][0]
        # 0.5M * 1850 + 0.3M * 820 = 925 + 246 = 1171
        assert cat1.emissions_tco2e == pytest.approx(1171.0, rel=1e-2)


# ===================================================================
# TestTransportEmissions
# ===================================================================


class TestTransportEmissions:
    """Cat 4 and Cat 9 transport emission tests."""

    def test_cat4_road(self):
        engine = RetailScope3Engine()
        inp = _make_input(
            upstream_transport=[
                TransportData(mode=TransportMode.ROAD, distance_km=500.0, weight_tonnes=20.0),
            ],
        )
        result = engine.calculate_scope3(inp)
        cat4 = [c for c in result.category_breakdown if c.category == Scope3Category.CAT_4.value][0]
        # 500 * 20 * 0.000062 = 0.62
        assert cat4.emissions_tco2e == pytest.approx(0.62, rel=1e-2)

    def test_cat9_last_mile(self):
        engine = RetailScope3Engine()
        inp = _make_input(
            downstream_transport=[
                TransportData(mode=TransportMode.LAST_MILE_VAN, distance_km=100.0, weight_tonnes=5.0),
            ],
        )
        result = engine.calculate_scope3(inp)
        cat9 = [c for c in result.category_breakdown if c.category == Scope3Category.CAT_9.value][0]
        # 100 * 5 * 0.000248 = 0.124
        assert cat9.emissions_tco2e == pytest.approx(0.124, rel=1e-2)

    def test_multimodal(self):
        engine = RetailScope3Engine()
        inp = _make_input(
            upstream_transport=[
                TransportData(mode=TransportMode.SEA, distance_km=5000.0, weight_tonnes=100.0),
                TransportData(mode=TransportMode.ROAD, distance_km=200.0, weight_tonnes=100.0),
            ],
        )
        result = engine.calculate_scope3(inp)
        cat4 = [c for c in result.category_breakdown if c.category == Scope3Category.CAT_4.value][0]
        # sea: 5000*100*0.000008 = 4.0; road: 200*100*0.000062 = 1.24; total = 5.24
        assert cat4.emissions_tco2e == pytest.approx(5.24, rel=1e-2)

    def test_air_freight(self):
        engine = RetailScope3Engine()
        inp = _make_input(
            upstream_transport=[
                TransportData(mode=TransportMode.AIR, distance_km=8000.0, weight_tonnes=2.0),
            ],
        )
        result = engine.calculate_scope3(inp)
        cat4 = [c for c in result.category_breakdown if c.category == Scope3Category.CAT_4.value][0]
        # 8000 * 2 * 0.000602 = 9.632
        assert cat4.emissions_tco2e == pytest.approx(9.632, rel=1e-2)


# ===================================================================
# TestUsePhaseCat11
# ===================================================================


class TestUsePhaseCat11:
    """Cat 11: Use of Sold Products tests."""

    def test_electronics_use(self):
        engine = RetailScope3Engine()
        inp = _make_input(
            use_phase=[
                UsePhaseData(
                    product_type="laptop",
                    units_sold=1000,
                    grid_factor_tco2e_per_kwh=0.000230,
                ),
            ],
        )
        result = engine.calculate_scope3(inp)
        cat11 = [c for c in result.category_breakdown if c.category == Scope3Category.CAT_11.value][0]
        # 1000 * 50 kWh/yr * 5 yr * 0.000230 = 57.5
        assert cat11.emissions_tco2e == pytest.approx(57.5, rel=1e-2)

    def test_zero_energy_product(self):
        engine = RetailScope3Engine()
        inp = _make_input(
            use_phase=[
                UsePhaseData(
                    product_type="unknown_product",
                    units_sold=500,
                ),
            ],
        )
        result = engine.calculate_scope3(inp)
        cat11 = [c for c in result.category_breakdown if c.category == Scope3Category.CAT_11.value][0]
        # unknown product has no entry in USE_PHASE_ELECTRICITY -> 0
        assert cat11.emissions_tco2e == pytest.approx(0.0, abs=1e-6)

    def test_lifetime_calc(self):
        # Verify the use-phase data constants
        laptop_info = USE_PHASE_ELECTRICITY["laptop"]
        assert laptop_info["kwh_per_year"] == 50.0
        assert laptop_info["lifetime_years"] == 5.0


# ===================================================================
# TestEndOfLifeCat12
# ===================================================================


class TestEndOfLifeCat12:
    """Cat 12: End-of-Life Treatment tests."""

    def test_landfill(self):
        engine = RetailScope3Engine()
        inp = _make_input(
            end_of_life=[
                EndOfLifeData(material="plastic", weight_tonnes=100.0, disposal_method="landfill"),
            ],
        )
        result = engine.calculate_scope3(inp)
        cat12 = [c for c in result.category_breakdown if c.category == Scope3Category.CAT_12.value][0]
        # 100 * 0.040 = 4.0
        assert cat12.emissions_tco2e == pytest.approx(4.0, rel=1e-2)

    def test_recycling(self):
        engine = RetailScope3Engine()
        inp = _make_input(
            end_of_life=[
                EndOfLifeData(material="metal_aluminium", weight_tonnes=10.0, disposal_method="recycling"),
            ],
        )
        result = engine.calculate_scope3(inp)
        cat12 = [c for c in result.category_breakdown if c.category == Scope3Category.CAT_12.value][0]
        # 10 * -9.1 = -91.0 (credit from recycling)
        assert cat12.emissions_tco2e == pytest.approx(-91.0, rel=1e-2)

    def test_mixed_disposal(self):
        engine = RetailScope3Engine()
        inp = _make_input(
            end_of_life=[
                EndOfLifeData(material="paper", weight_tonnes=50.0, disposal_method="landfill"),
                EndOfLifeData(material="paper", weight_tonnes=50.0, disposal_method="recycling"),
            ],
        )
        result = engine.calculate_scope3(inp)
        cat12 = [c for c in result.category_breakdown if c.category == Scope3Category.CAT_12.value][0]
        # 50 * 1.070 + 50 * -0.700 = 53.5 - 35 = 18.5
        assert cat12.emissions_tco2e == pytest.approx(18.5, rel=1e-2)


# ===================================================================
# TestHotspotAnalysis
# ===================================================================


class TestHotspotAnalysis:
    """Hotspot analysis tests."""

    def test_pareto_top20(self):
        engine = RetailScope3Engine()
        inp = _make_input(
            purchased_goods=[
                PurchasedGoodsData(
                    product_category=ProductCategory.FOOD_FRESH,
                    spend_eur=5_000_000.0,
                ),
            ],
            upstream_transport=[
                TransportData(mode=TransportMode.ROAD, distance_km=1000.0, weight_tonnes=500.0),
            ],
        )
        result = engine.calculate_scope3(inp)
        assert len(result.hotspot_analysis.pareto_categories) >= 1

    def test_top_suppliers(self):
        engine = RetailScope3Engine()
        inp = _make_input(
            purchased_goods=[
                PurchasedGoodsData(
                    product_category=ProductCategory.FOOD_FRESH,
                    supplier_name="FreshCo",
                    spend_eur=1_000_000.0,
                ),
                PurchasedGoodsData(
                    product_category=ProductCategory.BEVERAGES,
                    supplier_name="DrinkCo",
                    spend_eur=500_000.0,
                ),
            ],
        )
        result = engine.calculate_scope3(inp)
        assert len(result.hotspot_analysis.top_suppliers) >= 2

    def test_top_products(self):
        engine = RetailScope3Engine()
        inp = _make_input(
            purchased_goods=[
                PurchasedGoodsData(
                    product_category=ProductCategory.ELECTRONICS,
                    spend_eur=2_000_000.0,
                ),
            ],
        )
        result = engine.calculate_scope3(inp)
        assert len(result.hotspot_analysis.top_products) >= 1

    def test_improvement_potential(self):
        engine = RetailScope3Engine()
        inp = _make_input(
            purchased_goods=[
                PurchasedGoodsData(
                    product_category=ProductCategory.FOOD_FRESH,
                    spend_eur=10_000_000.0,
                ),
            ],
        )
        result = engine.calculate_scope3(inp)
        # spend_based items have 15% improvement potential
        assert result.hotspot_analysis.improvement_potential_tco2e > 0


# ===================================================================
# TestDataQuality
# ===================================================================


class TestDataQuality:
    """Data quality summary tests."""

    def test_weighted_average(self):
        engine = RetailScope3Engine()
        inp = _make_input(
            purchased_goods=[
                PurchasedGoodsData(
                    product_category=ProductCategory.FOOD_FRESH,
                    spend_eur=1_000_000.0,
                    data_quality=DataQualityLevel.SCORE_5,
                ),
            ],
        )
        result = engine.calculate_scope3(inp)
        assert result.data_quality_summary.weighted_score >= 1.0

    def test_score_range_1_5(self):
        engine = RetailScope3Engine()
        inp = _make_input(
            purchased_goods=[
                PurchasedGoodsData(
                    product_category=ProductCategory.FOOD_FRESH,
                    spend_eur=1_000_000.0,
                    data_quality=DataQualityLevel.SCORE_5,
                ),
            ],
        )
        result = engine.calculate_scope3(inp)
        assert 1.0 <= result.data_quality_summary.weighted_score <= 5.0

    def test_engagement_recommendations(self):
        engine = RetailScope3Engine()
        inp = _make_input(
            purchased_goods=[
                PurchasedGoodsData(
                    product_category=ProductCategory.FOOD_FRESH,
                    spend_eur=1_000_000.0,
                    data_quality=DataQualityLevel.SCORE_5,
                ),
            ],
        )
        result = engine.calculate_scope3(inp)
        assert len(result.data_quality_summary.recommendations) >= 1

    def test_dq_by_method(self):
        engine = RetailScope3Engine()
        inp = _make_input(
            purchased_goods=[
                PurchasedGoodsData(
                    product_category=ProductCategory.FOOD_FRESH,
                    spend_eur=1_000_000.0,
                    data_quality=DataQualityLevel.SCORE_5,
                ),
                PurchasedGoodsData(
                    product_category=ProductCategory.ELECTRONICS,
                    supplier_emissions_tco2e=100.0,
                    calculation_method=CalculationMethod.SUPPLIER_SPECIFIC,
                    data_quality=DataQualityLevel.SCORE_1,
                ),
            ],
        )
        result = engine.calculate_scope3(inp)
        dist = result.data_quality_summary.score_distribution
        assert "score_5" in dist
        assert "score_1" in dist


# ===================================================================
# TestCategoryBreakdown
# ===================================================================


class TestCategoryBreakdown:
    """Category breakdown structure tests."""

    def test_all_categories_present(self):
        engine = RetailScope3Engine()
        inp = _make_input()
        result = engine.calculate_scope3(inp)
        assert len(result.category_breakdown) == 15

    def test_pct_sums_to_100(self):
        engine = RetailScope3Engine()
        inp = _make_input(
            purchased_goods=[
                PurchasedGoodsData(
                    product_category=ProductCategory.FOOD_FRESH,
                    spend_eur=5_000_000.0,
                ),
            ],
            upstream_transport=[
                TransportData(mode=TransportMode.ROAD, distance_km=1000.0, weight_tonnes=100.0),
            ],
        )
        result = engine.calculate_scope3(inp)
        total_pct = sum(c.pct_of_total for c in result.category_breakdown)
        assert total_pct == pytest.approx(100.0, abs=1.0)

    def test_dominant_category(self):
        engine = RetailScope3Engine()
        inp = _make_input(
            purchased_goods=[
                PurchasedGoodsData(
                    product_category=ProductCategory.FOOD_FRESH,
                    spend_eur=10_000_000.0,
                ),
            ],
        )
        result = engine.calculate_scope3(inp)
        sorted_cats = sorted(result.category_breakdown, key=lambda c: c.emissions_tco2e, reverse=True)
        assert sorted_cats[0].category == Scope3Category.CAT_1.value


# ===================================================================
# TestProvenance
# ===================================================================


class TestProvenance:
    """Provenance hash tests."""

    def test_hash(self):
        engine = RetailScope3Engine()
        inp = _make_input(
            purchased_goods=[
                PurchasedGoodsData(product_category=ProductCategory.FOOD_FRESH, spend_eur=1_000_000.0),
            ],
        )
        result = engine.calculate_scope3(inp)
        assert len(result.provenance_hash) == 64

    def test_deterministic(self):
        engine = RetailScope3Engine()
        inp = _make_input(
            purchased_goods=[
                PurchasedGoodsData(product_category=ProductCategory.FOOD_FRESH, spend_eur=1_000_000.0),
            ],
        )
        r1 = engine.calculate_scope3(inp)
        r2 = engine.calculate_scope3(inp)
        assert r1.provenance_hash == r2.provenance_hash

    def test_different_input(self):
        engine = RetailScope3Engine()
        inp1 = _make_input(
            purchased_goods=[
                PurchasedGoodsData(product_category=ProductCategory.FOOD_FRESH, spend_eur=1_000_000.0),
            ],
        )
        inp2 = _make_input(
            organisation_id="ORG002",
            purchased_goods=[
                PurchasedGoodsData(product_category=ProductCategory.FOOD_FRESH, spend_eur=2_000_000.0),
            ],
        )
        r1 = engine.calculate_scope3(inp1)
        r2 = engine.calculate_scope3(inp2)
        assert r1.provenance_hash != r2.provenance_hash


# ===================================================================
# TestEdgeCases
# ===================================================================


class TestEdgeCases:
    """Edge case and boundary tests."""

    def test_empty_suppliers(self):
        engine = RetailScope3Engine()
        inp = _make_input()
        result = engine.calculate_scope3(inp)
        assert result.total_scope3_tco2e == pytest.approx(0.0, abs=1e-6)

    def test_large_dataset(self):
        engine = RetailScope3Engine()
        goods = [
            PurchasedGoodsData(
                product_category=ProductCategory.FOOD_FRESH,
                spend_eur=100_000.0,
                supplier_name=f"Supplier_{i}",
            )
            for i in range(200)
        ]
        inp = _make_input(purchased_goods=goods)
        result = engine.calculate_scope3(inp)
        assert result.total_scope3_tco2e > 0
        assert len(result.hotspot_analysis.top_suppliers) <= 10

    def test_result_fields(self):
        engine = RetailScope3Engine()
        inp = _make_input(
            purchased_goods=[
                PurchasedGoodsData(product_category=ProductCategory.FOOD_FRESH, spend_eur=1_000_000.0),
            ],
        )
        result = engine.calculate_scope3(inp)
        assert isinstance(result, RetailScope3Result)
        assert result.organisation_id == "ORG001"
        assert result.reporting_year == 2025
        assert result.engine_version == "1.0.0"
        assert result.processing_time_ms >= 0
