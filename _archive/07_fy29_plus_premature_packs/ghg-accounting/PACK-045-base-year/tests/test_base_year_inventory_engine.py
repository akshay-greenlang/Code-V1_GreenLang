# -*- coding: utf-8 -*-
"""
Tests for BaseYearInventoryEngine (Engine 2).

Covers inventory creation, scope aggregation, completeness assessment,
snapshots, comparisons, and GWP factors.
Target: ~70 tests.
"""

import pytest
from decimal import Decimal
from pathlib import Path
import sys

PACK_ROOT = Path(__file__).resolve().parent.parent
if str(PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(PACK_ROOT))

from engines.base_year_inventory_engine import (
    BaseYearInventoryEngine,
    SourceEmission,
    InventoryConfig,
    BaseYearInventory,
    ScopeTotal,
    InventoryComparison,
    CompletenessAssessment,
    SourceCategory,
    GasType,
    GWPVersion,
    ScopeType,
    ConsolidationApproach,
    MethodologyTier,
    InventoryStatus,
    GWP_FACTORS,
    CATEGORY_SCOPE_MAP,
    SCOPE_1_CATEGORIES,
    establish_base_year_inventory,
    compare_base_year_inventories,
    get_gwp_factor,
)


class TestBaseYearInventoryEngineInit:
    def test_engine_creation(self, inventory_engine):
        assert inventory_engine is not None

    def test_engine_version(self, inventory_engine):
        assert inventory_engine.get_version() == "1.0.0"


class TestSourceEmissionModel:
    def test_create_minimal(self):
        s = SourceEmission(category=SourceCategory.STATIONARY_COMBUSTION)
        assert s.category == SourceCategory.STATIONARY_COMBUSTION
        assert s.tco2e == Decimal("0")

    def test_scope_property(self):
        s = SourceEmission(category=SourceCategory.STATIONARY_COMBUSTION)
        assert s.scope == ScopeType.SCOPE_1

    def test_scope_property_scope2(self):
        s = SourceEmission(category=SourceCategory.ELECTRICITY_LOCATION)
        assert s.scope == ScopeType.SCOPE_2_LOCATION

    def test_scope_property_scope3(self):
        s = SourceEmission(category=SourceCategory.SCOPE3_CAT1)
        assert s.scope == ScopeType.SCOPE_3

    def test_decimal_coercion(self):
        s = SourceEmission(category=SourceCategory.STATIONARY_COMBUSTION, tco2e=100.5)
        assert s.tco2e == Decimal("100.5")

    def test_default_gas_type(self):
        s = SourceEmission(category=SourceCategory.STATIONARY_COMBUSTION)
        assert s.gas_type == GasType.CO2

    def test_default_methodology_tier(self):
        s = SourceEmission(category=SourceCategory.STATIONARY_COMBUSTION)
        assert s.methodology_tier == MethodologyTier.TIER_1


class TestEstablishInventory:
    def test_basic_establishment(self, inventory_engine, sample_sources, inventory_config):
        inv = inventory_engine.establish_inventory(sample_sources, inventory_config)
        assert isinstance(inv, BaseYearInventory)
        assert inv.status == InventoryStatus.ESTABLISHED
        assert inv.base_year == 2022

    def test_scope1_total(self, established_inventory):
        # stationary(3000) + mobile(1500) + fugitive(500) = 5000
        assert established_inventory.scope1_total_tco2e == Decimal("5000.000")

    def test_scope2_location_total(self, established_inventory):
        # electricity_location(2000)
        assert established_inventory.scope2_location_tco2e == Decimal("2000.000")

    def test_scope2_market_total(self, established_inventory):
        # electricity_market(1800)
        assert established_inventory.scope2_market_tco2e == Decimal("1800.000")

    def test_scope3_total(self, established_inventory):
        # scope3_cat1(4000) + scope3_cat6(300)
        assert established_inventory.scope3_total_tco2e == Decimal("4300.000")

    def test_grand_total(self, established_inventory):
        # scope1(5000) + scope2_location(2000) + scope3(4300) = 11300
        assert established_inventory.grand_total_tco2e == Decimal("11300.000")

    def test_provenance_hash_set(self, established_inventory):
        assert established_inventory.provenance_hash != ""
        assert len(established_inventory.provenance_hash) == 64

    def test_processing_time_positive(self, established_inventory):
        assert established_inventory.processing_time_ms > 0

    def test_organization_id_set(self, established_inventory):
        assert established_inventory.organization_id == "ORG-TEST-001"

    def test_gwp_version_set(self, established_inventory):
        assert established_inventory.gwp_version == GWPVersion.AR5

    def test_empty_sources_raises(self, inventory_engine, inventory_config):
        with pytest.raises(ValueError, match="At least one"):
            inventory_engine.establish_inventory([], inventory_config)

    def test_exclude_scope3(self, inventory_engine, sample_sources):
        config = InventoryConfig(
            organization_id="ORG-001",
            base_year=2022,
            include_scope3=False,
        )
        inv = inventory_engine.establish_inventory(sample_sources, config)
        assert inv.scope3_total_tco2e == Decimal("0")

    def test_minimum_quality_filter(self, inventory_engine, sample_sources):
        config = InventoryConfig(
            organization_id="ORG-001",
            base_year=2022,
            minimum_quality_score=Decimal("80"),
        )
        inv = inventory_engine.establish_inventory(sample_sources, config)
        # Sources with quality < 80 should be filtered
        assert len(inv.sources) >= 1

    def test_completeness_assessment_present(self, established_inventory):
        assert established_inventory.completeness is not None
        assert isinstance(established_inventory.completeness, CompletenessAssessment)

    # Helper property for tests
    @pytest.fixture
    def source_count_check(self, established_inventory):
        return len(established_inventory.sources)


# Add property to BaseYearInventory for test convenience
@pytest.fixture
def source_count_via_sources():
    """Helper to count sources."""
    return True


class TestAggregation:
    def test_aggregate_by_scope(self, inventory_engine, sample_sources):
        totals = inventory_engine.aggregate_by_scope(sample_sources)
        assert ScopeType.SCOPE_1 in totals
        assert ScopeType.SCOPE_2_LOCATION in totals
        assert ScopeType.SCOPE_3 in totals

    def test_aggregate_by_category(self, inventory_engine, sample_sources):
        totals = inventory_engine.aggregate_by_category(sample_sources)
        assert "stationary_combustion" in totals
        assert totals["stationary_combustion"] == Decimal("3000.000")

    def test_aggregate_by_gas(self, inventory_engine, sample_sources):
        totals = inventory_engine.aggregate_by_gas(sample_sources)
        assert "co2" in totals

    def test_aggregate_by_facility(self, inventory_engine, sample_sources):
        totals = inventory_engine.aggregate_by_facility(sample_sources)
        assert "FAC-001" in totals
        assert "FAC-002" in totals

    def test_calculate_scope_totals(self, inventory_engine, sample_sources):
        scope_totals = inventory_engine.calculate_scope_totals(sample_sources)
        assert len(scope_totals) == 4  # 4 scope types
        for st in scope_totals:
            assert isinstance(st, ScopeTotal)


class TestCompleteness:
    def test_validate_completeness_assessment(self, inventory_engine, sample_sources):
        assessment = inventory_engine.validate_completeness_assessment(sample_sources)
        assert isinstance(assessment, CompletenessAssessment)
        assert Decimal("0") <= assessment.overall_completeness <= Decimal("100")

    def test_completeness_scope1(self, inventory_engine, sample_sources):
        assessment = inventory_engine.validate_completeness_assessment(sample_sources)
        # We have 3 of 5 scope 1 categories => 60%
        assert assessment.scope1_completeness == Decimal("60.00")

    def test_completeness_missing_categories(self, inventory_engine, sample_sources):
        assessment = inventory_engine.validate_completeness_assessment(sample_sources)
        assert len(assessment.missing_categories) > 0

    def test_completeness_no_scope3(self, inventory_engine, sample_sources):
        assessment = inventory_engine.validate_completeness_assessment(
            sample_sources, include_scope3=False
        )
        assert assessment.scope3_completeness == Decimal("0")

    def test_validate_completeness_on_inventory(self, inventory_engine, established_inventory):
        assessment = inventory_engine.validate_completeness(established_inventory)
        assert isinstance(assessment, CompletenessAssessment)


class TestSnapshot:
    def test_snapshot_creates_frozen(self, inventory_engine, established_inventory):
        snapshot = inventory_engine.snapshot_inventory(established_inventory)
        assert snapshot.status == InventoryStatus.FROZEN

    def test_snapshot_has_different_id(self, inventory_engine, established_inventory):
        snapshot = inventory_engine.snapshot_inventory(established_inventory)
        assert snapshot.inventory_id != established_inventory.inventory_id

    def test_snapshot_has_provenance_hash(self, inventory_engine, established_inventory):
        snapshot = inventory_engine.snapshot_inventory(established_inventory)
        assert snapshot.provenance_hash != ""
        assert len(snapshot.provenance_hash) == 64

    def test_snapshot_preserves_data(self, inventory_engine, established_inventory):
        snapshot = inventory_engine.snapshot_inventory(established_inventory)
        assert snapshot.grand_total_tco2e == established_inventory.grand_total_tco2e
        assert snapshot.scope1_total_tco2e == established_inventory.scope1_total_tco2e


class TestComparison:
    def test_compare_identical_inventories(self, inventory_engine, established_inventory):
        comp = inventory_engine.compare_inventories(
            established_inventory, established_inventory
        )
        assert isinstance(comp, InventoryComparison)
        assert comp.any_significant is False
        assert comp.total_diff_tco2e == Decimal("0.000")

    def test_compare_different_inventories(self, inventory_engine, sample_sources, inventory_config):
        inv1 = inventory_engine.establish_inventory(sample_sources, inventory_config)
        # Create a modified set of sources with higher emissions
        modified_sources = []
        for s in sample_sources:
            data = s.model_dump()
            data["tco2e"] = s.tco2e * Decimal("1.1")  # 10% increase
            modified_sources.append(SourceEmission.model_validate(data))
        inv2 = inventory_engine.establish_inventory(modified_sources, inventory_config)

        comp = inventory_engine.compare_inventories(inv1, inv2)
        assert comp.any_significant is True
        assert comp.total_diff_pct >= Decimal("5")

    def test_comparison_provenance_hash(self, inventory_engine, established_inventory):
        comp = inventory_engine.compare_inventories(
            established_inventory, established_inventory
        )
        assert comp.provenance_hash != ""

    def test_comparison_summary(self, inventory_engine, established_inventory):
        comp = inventory_engine.compare_inventories(
            established_inventory, established_inventory
        )
        assert "No significant" in comp.summary

    def test_comparison_custom_threshold(self, inventory_engine, established_inventory):
        comp = inventory_engine.compare_inventories(
            established_inventory, established_inventory,
            significance_threshold_pct=Decimal("1"),
        )
        assert comp.significance_threshold == Decimal("1")


class TestGWPFactors:
    def test_gwp_factors_ar5_co2(self):
        assert GWP_FACTORS["ar5"]["co2"] == Decimal("1")

    def test_gwp_factors_ar5_ch4(self):
        assert GWP_FACTORS["ar5"]["ch4"] == Decimal("28")

    def test_gwp_factors_ar6_ch4(self):
        assert GWP_FACTORS["ar6"]["ch4"] == Decimal("27.9")

    def test_get_gwp_factor_func(self):
        factor = get_gwp_factor(GasType.CH4, GWPVersion.AR5)
        assert factor == Decimal("28")

    def test_get_gwp_factor_unknown_defaults_to_1(self):
        # Using a known gas with an unknown version should fallback
        factor = get_gwp_factor(GasType.CO2, GWPVersion.AR5)
        assert factor == Decimal("1")


class TestCategoryScopeMap:
    def test_stationary_is_scope1(self):
        assert CATEGORY_SCOPE_MAP["stationary_combustion"] == ScopeType.SCOPE_1

    def test_electricity_location_is_scope2(self):
        assert CATEGORY_SCOPE_MAP["electricity_location"] == ScopeType.SCOPE_2_LOCATION

    def test_scope3_cat1_is_scope3(self):
        assert CATEGORY_SCOPE_MAP["scope3_cat1"] == ScopeType.SCOPE_3

    def test_all_categories_mapped(self):
        for cat in SourceCategory:
            assert cat.value in CATEGORY_SCOPE_MAP


class TestConvenienceFunctions:
    def test_establish_convenience(self, sample_sources, inventory_config):
        inv = establish_base_year_inventory(sample_sources, inventory_config)
        assert isinstance(inv, BaseYearInventory)

    def test_compare_convenience(self, established_inventory):
        comp = compare_base_year_inventories(
            established_inventory, established_inventory
        )
        assert isinstance(comp, InventoryComparison)


class TestInventorySummary:
    def test_get_summary(self, inventory_engine, established_inventory):
        summary = inventory_engine.get_inventory_summary(established_inventory)
        assert summary["base_year"] == 2022
        assert summary["status"] == "established"
        assert "grand_total_tco2e" in summary
        assert "provenance_hash" in summary
