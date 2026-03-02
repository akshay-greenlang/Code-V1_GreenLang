# -*- coding: utf-8 -*-
"""
Unit tests for CompletenessScreenerEngine (AGENT-MRV-029, Engine 5)

70 tests covering:
- Relevance matrix: 8 company types x 5 tests each (40 tests)
- Completeness screening: gap identification, scoring, actions (20 tests)
- Industry benchmarks: per-type validation, sum-to-100 (10 tests)

The engine evaluates which GHG Protocol Scope 3 categories are relevant
based on company type, identifies data gaps, estimates materiality using
industry benchmarks, and generates prioritized data collection
recommendations. All relevance mappings are deterministic lookup tables.

Author: GL-TestEngineer
Date: March 2026
"""

import pytest
from decimal import Decimal
from typing import Any, Dict, List
from unittest.mock import MagicMock

from greenlang.scope3_category_mapper.models import (
    ALL_SCOPE3_CATEGORIES,
    BenchmarkComparison,
    CategoryCompletenessEntry,
    CategoryRelevance,
    CompanyType,
    CompletenessReport,
    DataQualityTier,
    Scope3Category,
    ScreeningResult,
)
from greenlang.scope3_category_mapper.completeness_screener import (
    BENCHMARK_TOLERANCE_PCT,
    COMPANY_TYPE_RELEVANCE,
    CompletenessScreenerEngine,
    INDUSTRY_BENCHMARKS,
)


# ==============================================================================
# FIXTURES
# ==============================================================================


@pytest.fixture
def engine() -> CompletenessScreenerEngine:
    """Create a fresh CompletenessScreenerEngine instance."""
    CompletenessScreenerEngine.reset_instance()
    return CompletenessScreenerEngine.get_instance()


@pytest.fixture
def all_categories_reported() -> List[Scope3Category]:
    """All 15 Scope 3 categories reported."""
    return list(ALL_SCOPE3_CATEGORIES)


@pytest.fixture
def no_categories_reported() -> List[Scope3Category]:
    """No categories reported."""
    return []


@pytest.fixture
def manufacturer_material_only() -> List[Scope3Category]:
    """Only material categories for a manufacturer."""
    return [
        Scope3Category.CAT_1_PURCHASED_GOODS_SERVICES,
        Scope3Category.CAT_2_CAPITAL_GOODS,
        Scope3Category.CAT_4_UPSTREAM_TRANSPORTATION,
        Scope3Category.CAT_9_DOWNSTREAM_TRANSPORTATION,
        Scope3Category.CAT_10_PROCESSING_SOLD_PRODUCTS,
        Scope3Category.CAT_11_USE_OF_SOLD_PRODUCTS,
        Scope3Category.CAT_12_END_OF_LIFE_TREATMENT,
    ]


@pytest.fixture
def sample_data_by_category() -> Dict[Scope3Category, Dict[str, Any]]:
    """Category data payloads for quality assessment."""
    return {
        Scope3Category.CAT_1_PURCHASED_GOODS_SERVICES: {
            "emissions_kg": 500000,
            "calculation_method": "spend_based",
        },
        Scope3Category.CAT_4_UPSTREAM_TRANSPORTATION: {
            "emissions_tco2e": 1200.5,
            "calculation_method": "activity_based",
        },
        Scope3Category.CAT_6_BUSINESS_TRAVEL: {
            "total_emissions": 350,
            "calculation_method": "supplier_specific",
        },
    }


# ==============================================================================
# MANUFACTURER RELEVANCE TESTS
# ==============================================================================


class TestManufacturerRelevance:
    """Test category relevance for CompanyType.MANUFACTURER."""

    def test_cat1_material(self, engine):
        """Cat 1 (Purchased Goods & Services) is MATERIAL for manufacturer."""
        r = engine.assess_category_relevance(
            CompanyType.MANUFACTURER,
            Scope3Category.CAT_1_PURCHASED_GOODS_SERVICES,
        )
        assert r == CategoryRelevance.MATERIAL

    def test_cat2_material(self, engine):
        """Cat 2 (Capital Goods) is MATERIAL for manufacturer."""
        r = engine.assess_category_relevance(
            CompanyType.MANUFACTURER, Scope3Category.CAT_2_CAPITAL_GOODS
        )
        assert r == CategoryRelevance.MATERIAL

    def test_cat4_material(self, engine):
        """Cat 4 (Upstream Transportation) is MATERIAL for manufacturer."""
        r = engine.assess_category_relevance(
            CompanyType.MANUFACTURER,
            Scope3Category.CAT_4_UPSTREAM_TRANSPORTATION,
        )
        assert r == CategoryRelevance.MATERIAL

    def test_cat9_material(self, engine):
        """Cat 9 (Downstream Transportation) is MATERIAL for manufacturer."""
        r = engine.assess_category_relevance(
            CompanyType.MANUFACTURER,
            Scope3Category.CAT_9_DOWNSTREAM_TRANSPORTATION,
        )
        assert r == CategoryRelevance.MATERIAL

    def test_cat14_not_relevant(self, engine):
        """Cat 14 (Franchises) is NOT_RELEVANT for manufacturer."""
        r = engine.assess_category_relevance(
            CompanyType.MANUFACTURER, Scope3Category.CAT_14_FRANCHISES
        )
        assert r == CategoryRelevance.NOT_RELEVANT


# ==============================================================================
# FINANCIAL INSTITUTION RELEVANCE TESTS
# ==============================================================================


class TestFinancialRelevance:
    """Test category relevance for CompanyType.FINANCIAL."""

    def test_cat15_material(self, engine):
        """Cat 15 (Investments) is MATERIAL for financial."""
        r = engine.assess_category_relevance(
            CompanyType.FINANCIAL, Scope3Category.CAT_15_INVESTMENTS
        )
        assert r == CategoryRelevance.MATERIAL

    def test_cat6_material(self, engine):
        """Cat 6 (Business Travel) is MATERIAL for financial."""
        r = engine.assess_category_relevance(
            CompanyType.FINANCIAL, Scope3Category.CAT_6_BUSINESS_TRAVEL
        )
        assert r == CategoryRelevance.MATERIAL

    def test_cat4_not_relevant(self, engine):
        """Cat 4 (Upstream Transportation) is NOT_RELEVANT for financial."""
        r = engine.assess_category_relevance(
            CompanyType.FINANCIAL,
            Scope3Category.CAT_4_UPSTREAM_TRANSPORTATION,
        )
        assert r == CategoryRelevance.NOT_RELEVANT

    def test_cat8_material(self, engine):
        """Cat 8 (Upstream Leased Assets) is MATERIAL for financial."""
        r = engine.assess_category_relevance(
            CompanyType.FINANCIAL,
            Scope3Category.CAT_8_UPSTREAM_LEASED_ASSETS,
        )
        assert r == CategoryRelevance.MATERIAL

    def test_cat10_not_relevant(self, engine):
        """Cat 10 (Processing Sold Products) is NOT_RELEVANT for financial."""
        r = engine.assess_category_relevance(
            CompanyType.FINANCIAL,
            Scope3Category.CAT_10_PROCESSING_SOLD_PRODUCTS,
        )
        assert r == CategoryRelevance.NOT_RELEVANT


# ==============================================================================
# RETAILER RELEVANCE TESTS
# ==============================================================================


class TestRetailerRelevance:
    """Test category relevance for CompanyType.RETAILER."""

    def test_cat1_material(self, engine):
        """Cat 1 (Purchased Goods & Services) is MATERIAL for retailer."""
        r = engine.assess_category_relevance(
            CompanyType.RETAILER,
            Scope3Category.CAT_1_PURCHASED_GOODS_SERVICES,
        )
        assert r == CategoryRelevance.MATERIAL

    def test_cat4_material(self, engine):
        """Cat 4 (Upstream Transportation) is MATERIAL for retailer."""
        r = engine.assess_category_relevance(
            CompanyType.RETAILER,
            Scope3Category.CAT_4_UPSTREAM_TRANSPORTATION,
        )
        assert r == CategoryRelevance.MATERIAL

    def test_cat8_material(self, engine):
        """Cat 8 (Upstream Leased Assets) is MATERIAL for retailer."""
        r = engine.assess_category_relevance(
            CompanyType.RETAILER,
            Scope3Category.CAT_8_UPSTREAM_LEASED_ASSETS,
        )
        assert r == CategoryRelevance.MATERIAL

    def test_cat9_material(self, engine):
        """Cat 9 (Downstream Transportation) is MATERIAL for retailer."""
        r = engine.assess_category_relevance(
            CompanyType.RETAILER,
            Scope3Category.CAT_9_DOWNSTREAM_TRANSPORTATION,
        )
        assert r == CategoryRelevance.MATERIAL

    def test_cat10_not_relevant(self, engine):
        """Cat 10 (Processing Sold Products) is NOT_RELEVANT for retailer."""
        r = engine.assess_category_relevance(
            CompanyType.RETAILER,
            Scope3Category.CAT_10_PROCESSING_SOLD_PRODUCTS,
        )
        assert r == CategoryRelevance.NOT_RELEVANT


# ==============================================================================
# ENERGY COMPANY RELEVANCE TESTS
# ==============================================================================


class TestEnergyRelevance:
    """Test category relevance for CompanyType.ENERGY."""

    def test_cat3_material(self, engine):
        """Cat 3 (Fuel & Energy Activities) is MATERIAL for energy."""
        r = engine.assess_category_relevance(
            CompanyType.ENERGY,
            Scope3Category.CAT_3_FUEL_ENERGY_ACTIVITIES,
        )
        assert r == CategoryRelevance.MATERIAL

    def test_cat11_material(self, engine):
        """Cat 11 (Use of Sold Products) is MATERIAL for energy."""
        r = engine.assess_category_relevance(
            CompanyType.ENERGY,
            Scope3Category.CAT_11_USE_OF_SOLD_PRODUCTS,
        )
        assert r == CategoryRelevance.MATERIAL

    def test_cat1_material(self, engine):
        """Cat 1 (Purchased Goods & Services) is MATERIAL for energy."""
        r = engine.assess_category_relevance(
            CompanyType.ENERGY,
            Scope3Category.CAT_1_PURCHASED_GOODS_SERVICES,
        )
        assert r == CategoryRelevance.MATERIAL

    def test_cat14_not_relevant(self, engine):
        """Cat 14 (Franchises) is NOT_RELEVANT for energy."""
        r = engine.assess_category_relevance(
            CompanyType.ENERGY, Scope3Category.CAT_14_FRANCHISES
        )
        assert r == CategoryRelevance.NOT_RELEVANT

    def test_cat9_material(self, engine):
        """Cat 9 (Downstream Transportation) is MATERIAL for energy."""
        r = engine.assess_category_relevance(
            CompanyType.ENERGY,
            Scope3Category.CAT_9_DOWNSTREAM_TRANSPORTATION,
        )
        assert r == CategoryRelevance.MATERIAL


# ==============================================================================
# MINING COMPANY RELEVANCE TESTS
# ==============================================================================


class TestMiningRelevance:
    """Test category relevance for CompanyType.MINING."""

    def test_cat5_material(self, engine):
        """Cat 5 (Waste Generated) is MATERIAL for mining."""
        r = engine.assess_category_relevance(
            CompanyType.MINING, Scope3Category.CAT_5_WASTE_GENERATED
        )
        assert r == CategoryRelevance.MATERIAL

    def test_cat10_material(self, engine):
        """Cat 10 (Processing Sold Products) is MATERIAL for mining."""
        r = engine.assess_category_relevance(
            CompanyType.MINING,
            Scope3Category.CAT_10_PROCESSING_SOLD_PRODUCTS,
        )
        assert r == CategoryRelevance.MATERIAL

    def test_cat11_not_relevant(self, engine):
        """Cat 11 (Use of Sold Products) is NOT_RELEVANT for mining."""
        r = engine.assess_category_relevance(
            CompanyType.MINING,
            Scope3Category.CAT_11_USE_OF_SOLD_PRODUCTS,
        )
        assert r == CategoryRelevance.NOT_RELEVANT

    def test_cat14_not_relevant(self, engine):
        """Cat 14 (Franchises) is NOT_RELEVANT for mining."""
        r = engine.assess_category_relevance(
            CompanyType.MINING, Scope3Category.CAT_14_FRANCHISES
        )
        assert r == CategoryRelevance.NOT_RELEVANT

    def test_cat3_material(self, engine):
        """Cat 3 (Fuel & Energy Activities) is MATERIAL for mining."""
        r = engine.assess_category_relevance(
            CompanyType.MINING,
            Scope3Category.CAT_3_FUEL_ENERGY_ACTIVITIES,
        )
        assert r == CategoryRelevance.MATERIAL


# ==============================================================================
# SERVICES COMPANY RELEVANCE TESTS
# ==============================================================================


class TestServicesRelevance:
    """Test category relevance for CompanyType.SERVICES."""

    def test_cat6_material(self, engine):
        """Cat 6 (Business Travel) is MATERIAL for services."""
        r = engine.assess_category_relevance(
            CompanyType.SERVICES, Scope3Category.CAT_6_BUSINESS_TRAVEL
        )
        assert r == CategoryRelevance.MATERIAL

    def test_cat7_material(self, engine):
        """Cat 7 (Employee Commuting) is MATERIAL for services."""
        r = engine.assess_category_relevance(
            CompanyType.SERVICES,
            Scope3Category.CAT_7_EMPLOYEE_COMMUTING,
        )
        assert r == CategoryRelevance.MATERIAL

    def test_cat8_material(self, engine):
        """Cat 8 (Upstream Leased Assets) is MATERIAL for services."""
        r = engine.assess_category_relevance(
            CompanyType.SERVICES,
            Scope3Category.CAT_8_UPSTREAM_LEASED_ASSETS,
        )
        assert r == CategoryRelevance.MATERIAL

    def test_cat4_not_relevant(self, engine):
        """Cat 4 (Upstream Transportation) is NOT_RELEVANT for services."""
        r = engine.assess_category_relevance(
            CompanyType.SERVICES,
            Scope3Category.CAT_4_UPSTREAM_TRANSPORTATION,
        )
        assert r == CategoryRelevance.NOT_RELEVANT

    def test_cat10_not_relevant(self, engine):
        """Cat 10 (Processing Sold Products) is NOT_RELEVANT for services."""
        r = engine.assess_category_relevance(
            CompanyType.SERVICES,
            Scope3Category.CAT_10_PROCESSING_SOLD_PRODUCTS,
        )
        assert r == CategoryRelevance.NOT_RELEVANT


# ==============================================================================
# AGRICULTURE RELEVANCE TESTS
# ==============================================================================


class TestAgricultureRelevance:
    """Test category relevance for CompanyType.AGRICULTURE."""

    def test_cat1_material(self, engine):
        """Cat 1 (Purchased Goods & Services) is MATERIAL for agriculture."""
        r = engine.assess_category_relevance(
            CompanyType.AGRICULTURE,
            Scope3Category.CAT_1_PURCHASED_GOODS_SERVICES,
        )
        assert r == CategoryRelevance.MATERIAL

    def test_cat4_material(self, engine):
        """Cat 4 (Upstream Transportation) is MATERIAL for agriculture."""
        r = engine.assess_category_relevance(
            CompanyType.AGRICULTURE,
            Scope3Category.CAT_4_UPSTREAM_TRANSPORTATION,
        )
        assert r == CategoryRelevance.MATERIAL

    def test_cat10_material(self, engine):
        """Cat 10 (Processing Sold Products) is MATERIAL for agriculture."""
        r = engine.assess_category_relevance(
            CompanyType.AGRICULTURE,
            Scope3Category.CAT_10_PROCESSING_SOLD_PRODUCTS,
        )
        assert r == CategoryRelevance.MATERIAL

    def test_cat6_not_relevant(self, engine):
        """Cat 6 (Business Travel) is NOT_RELEVANT for agriculture."""
        r = engine.assess_category_relevance(
            CompanyType.AGRICULTURE,
            Scope3Category.CAT_6_BUSINESS_TRAVEL,
        )
        assert r == CategoryRelevance.NOT_RELEVANT

    def test_cat8_not_relevant(self, engine):
        """Cat 8 (Upstream Leased Assets) is NOT_RELEVANT for agriculture."""
        r = engine.assess_category_relevance(
            CompanyType.AGRICULTURE,
            Scope3Category.CAT_8_UPSTREAM_LEASED_ASSETS,
        )
        assert r == CategoryRelevance.NOT_RELEVANT


# ==============================================================================
# TRANSPORT COMPANY RELEVANCE TESTS
# ==============================================================================


class TestTransportRelevance:
    """Test category relevance for CompanyType.TRANSPORT."""

    def test_cat2_material(self, engine):
        """Cat 2 (Capital Goods) is MATERIAL for transport (fleet)."""
        r = engine.assess_category_relevance(
            CompanyType.TRANSPORT, Scope3Category.CAT_2_CAPITAL_GOODS
        )
        assert r == CategoryRelevance.MATERIAL

    def test_cat3_material(self, engine):
        """Cat 3 (Fuel & Energy Activities) is MATERIAL for transport."""
        r = engine.assess_category_relevance(
            CompanyType.TRANSPORT,
            Scope3Category.CAT_3_FUEL_ENERGY_ACTIVITIES,
        )
        assert r == CategoryRelevance.MATERIAL

    def test_cat8_material(self, engine):
        """Cat 8 (Upstream Leased Assets) is MATERIAL for transport."""
        r = engine.assess_category_relevance(
            CompanyType.TRANSPORT,
            Scope3Category.CAT_8_UPSTREAM_LEASED_ASSETS,
        )
        assert r == CategoryRelevance.MATERIAL

    def test_cat4_not_relevant(self, engine):
        """Cat 4 (Upstream Transportation) is NOT_RELEVANT for transport."""
        r = engine.assess_category_relevance(
            CompanyType.TRANSPORT,
            Scope3Category.CAT_4_UPSTREAM_TRANSPORTATION,
        )
        assert r == CategoryRelevance.NOT_RELEVANT

    def test_cat13_material(self, engine):
        """Cat 13 (Downstream Leased Assets) is MATERIAL for transport."""
        r = engine.assess_category_relevance(
            CompanyType.TRANSPORT,
            Scope3Category.CAT_13_DOWNSTREAM_LEASED_ASSETS,
        )
        assert r == CategoryRelevance.MATERIAL


# ==============================================================================
# COMPLETENESS SCREENING TESTS
# ==============================================================================


class TestCompletenessScreening:
    """Test the main screen_completeness method."""

    def test_screen_all_categories_reported(
        self, engine, all_categories_reported
    ):
        """All 15 categories reported -> high completeness score."""
        report = engine.screen_completeness(
            company_type=CompanyType.MANUFACTURER,
            categories_reported=all_categories_reported,
        )
        assert report.overall_score == Decimal("100.00")
        assert report.categories_reported == 15
        assert len(report.gaps) == 0

    def test_screen_missing_material_category(self, engine):
        """Missing material categories -> lower score."""
        reported = [
            Scope3Category.CAT_1_PURCHASED_GOODS_SERVICES,
            Scope3Category.CAT_6_BUSINESS_TRAVEL,
        ]
        report = engine.screen_completeness(
            company_type=CompanyType.MANUFACTURER,
            categories_reported=reported,
        )
        assert report.overall_score < Decimal("50.00")
        assert len(report.gaps) > 0

    def test_screen_missing_non_relevant(self, engine):
        """Reporting all material + relevant, missing only not-relevant -> 90%."""
        # Get all material and relevant categories for manufacturer
        material_relevant = []
        for cat in ALL_SCOPE3_CATEGORIES:
            rel = COMPANY_TYPE_RELEVANCE[CompanyType.MANUFACTURER].get(cat)
            if rel in (CategoryRelevance.MATERIAL, CategoryRelevance.RELEVANT):
                material_relevant.append(cat)

        report = engine.screen_completeness(
            company_type=CompanyType.MANUFACTURER,
            categories_reported=material_relevant,
        )
        # Material=60% + Relevant=30% = 90% base, missing not-relevant=10%
        assert report.overall_score == Decimal("90.00")

    def test_screen_manufacturer_report_fields(
        self, engine, all_categories_reported
    ):
        """Manufacturer with all categories -> report has correct fields."""
        report = engine.screen_completeness(
            company_type=CompanyType.MANUFACTURER,
            categories_reported=all_categories_reported,
        )
        assert report.company_type == CompanyType.MANUFACTURER
        assert len(report.entries) == 15

    def test_identify_gaps_none(self, engine, all_categories_reported):
        """All categories reported -> no gaps."""
        gaps = engine.identify_gaps(
            company_type=CompanyType.MANUFACTURER,
            categories_reported=all_categories_reported,
        )
        assert len(gaps) == 0

    def test_identify_gaps_missing(self, engine):
        """Missing material category -> gap list returned."""
        reported = [Scope3Category.CAT_1_PURCHASED_GOODS_SERVICES]
        gaps = engine.identify_gaps(
            company_type=CompanyType.MANUFACTURER,
            categories_reported=reported,
        )
        assert len(gaps) > 0
        # Should mention at least Cat 2 (Capital Goods)
        assert any("Capital Goods" in g for g in gaps)

    def test_calculate_completeness_score_100(
        self, engine, all_categories_reported
    ):
        """Perfect coverage -> score = 100."""
        report = engine.screen_completeness(
            company_type=CompanyType.MANUFACTURER,
            categories_reported=all_categories_reported,
        )
        score = engine.calculate_completeness_score(report)
        assert score == Decimal("100.00")

    def test_calculate_completeness_score_0(
        self, engine, no_categories_reported
    ):
        """No categories reported -> score = 0."""
        report = engine.screen_completeness(
            company_type=CompanyType.MANUFACTURER,
            categories_reported=no_categories_reported,
        )
        # Nothing reported -> material 0/7, relevant 0/5, nr 0/3 -> all zero
        assert report.overall_score == Decimal("0.00")

    def test_materiality_estimate_manufacturer_cat1(self, engine):
        """Manufacturer Cat 1 benchmark is 60%."""
        mat = engine.estimate_materiality(
            CompanyType.MANUFACTURER,
            Scope3Category.CAT_1_PURCHASED_GOODS_SERVICES,
        )
        assert mat == Decimal("60.00")

    def test_materiality_estimate_financial_cat15(self, engine):
        """Financial Cat 15 benchmark is 80%."""
        mat = engine.estimate_materiality(
            CompanyType.FINANCIAL, Scope3Category.CAT_15_INVESTMENTS
        )
        assert mat == Decimal("80.00")

    def test_recommend_actions_prioritized(self, engine):
        """Recommendations: CRITICAL (missing material) comes first."""
        reported = [Scope3Category.CAT_1_PURCHASED_GOODS_SERVICES]
        report = engine.screen_completeness(
            company_type=CompanyType.MANUFACTURER,
            categories_reported=reported,
        )
        actions = engine.recommend_actions(report)
        assert len(actions) > 0
        assert actions[0].startswith("[CRITICAL]")

    def test_compare_to_benchmark_within_tolerance(self, engine):
        """Actual distribution close to benchmark -> within tolerance."""
        benchmarks = INDUSTRY_BENCHMARKS[CompanyType.MANUFACTURER]
        actual = dict(benchmarks)  # exact match
        comparisons = engine.compare_to_benchmark(
            CompanyType.MANUFACTURER, actual
        )
        for cat, comp in comparisons.items():
            assert comp.within_tolerance is True
            assert comp.flag is None

    def test_compare_to_benchmark_above_tolerance(self, engine):
        """Large deviation above benchmark -> flagged."""
        actual = {cat: Decimal("0.00") for cat in ALL_SCOPE3_CATEGORIES}
        actual[Scope3Category.CAT_1_PURCHASED_GOODS_SERVICES] = Decimal("95.00")
        actual[Scope3Category.CAT_2_CAPITAL_GOODS] = Decimal("5.00")
        comparisons = engine.compare_to_benchmark(
            CompanyType.MANUFACTURER, actual
        )
        cat1_comp = comparisons[Scope3Category.CAT_1_PURCHASED_GOODS_SERVICES]
        # 95 - 60 = 35% deviation -> outside 10% tolerance
        assert cat1_comp.within_tolerance is False
        assert cat1_comp.flag is not None
        assert "ABOVE" in cat1_comp.flag

    def test_screen_with_data_quality_supplier_specific(
        self, engine, sample_data_by_category
    ):
        """Category with supplier_specific method -> Tier 1 quality."""
        report = engine.screen_completeness(
            company_type=CompanyType.MANUFACTURER,
            categories_reported=list(sample_data_by_category.keys()),
            data_by_category=sample_data_by_category,
        )
        cat6_entry = None
        for e in report.entries:
            if e.category == Scope3Category.CAT_6_BUSINESS_TRAVEL:
                cat6_entry = e
                break
        assert cat6_entry is not None
        assert cat6_entry.data_quality_tier == DataQualityTier.TIER_1

    def test_screen_with_data_quality_spend_based(
        self, engine, sample_data_by_category
    ):
        """Category with spend_based method -> Tier 4 quality."""
        report = engine.screen_completeness(
            company_type=CompanyType.MANUFACTURER,
            categories_reported=list(sample_data_by_category.keys()),
            data_by_category=sample_data_by_category,
        )
        cat1_entry = None
        for e in report.entries:
            if e.category == Scope3Category.CAT_1_PURCHASED_GOODS_SERVICES:
                cat1_entry = e
                break
        assert cat1_entry is not None
        assert cat1_entry.data_quality_tier == DataQualityTier.TIER_4

    def test_screen_provenance_hash_present(
        self, engine, all_categories_reported
    ):
        """Report includes a 64-char SHA-256 provenance hash."""
        report = engine.screen_completeness(
            company_type=CompanyType.MANUFACTURER,
            categories_reported=all_categories_reported,
        )
        assert len(report.provenance_hash) == 64

    def test_screen_provenance_deterministic(
        self, engine, all_categories_reported
    ):
        """Same input produces same provenance hash."""
        r1 = engine.screen_completeness(
            company_type=CompanyType.MANUFACTURER,
            categories_reported=all_categories_reported,
        )
        r2 = engine.screen_completeness(
            company_type=CompanyType.MANUFACTURER,
            categories_reported=all_categories_reported,
        )
        assert r1.provenance_hash == r2.provenance_hash

    def test_screen_entries_count(self, engine, all_categories_reported):
        """Report entries list has exactly 15 items (one per category)."""
        report = engine.screen_completeness(
            company_type=CompanyType.MANUFACTURER,
            categories_reported=all_categories_reported,
        )
        assert len(report.entries) == 15

    def test_screen_invalid_company_type_raises(self, engine):
        """Invalid company type raises an error."""
        # CompanyType enum is strict; a raw string will fail when the engine
        # tries to access .value or look it up in the relevance matrix.
        with pytest.raises((ValueError, KeyError, AttributeError)):
            engine.screen_completeness(
                company_type="INVALID_TYPE",  # type: ignore[arg-type]
                categories_reported=[],
            )

    def test_screen_material_count_manufacturer(
        self, engine, all_categories_reported
    ):
        """Manufacturer has 7 material categories."""
        report = engine.screen_completeness(
            company_type=CompanyType.MANUFACTURER,
            categories_reported=all_categories_reported,
        )
        assert report.categories_material == 7


# ==============================================================================
# INDUSTRY BENCHMARK TESTS
# ==============================================================================


class TestIndustryBenchmarks:
    """Test industry benchmark data integrity."""

    @pytest.mark.parametrize("company_type", list(CompanyType))
    def test_benchmark_sum_close_to_100(self, company_type):
        """Each company type's benchmark percentages sum to ~100% (+/- 1%)."""
        benchmarks = INDUSTRY_BENCHMARKS[company_type]
        total = sum(benchmarks.values())
        # Allow a small tolerance for rounding in source data
        assert Decimal("99.00") <= total <= Decimal("101.00"), (
            f"{company_type.value} benchmarks sum to {total}, expected ~100.00"
        )

    def test_benchmark_manufacturer_cat1_highest(self):
        """For manufacturer, Cat 1 has the highest benchmark (60%)."""
        bm = INDUSTRY_BENCHMARKS[CompanyType.MANUFACTURER]
        cat1_pct = bm[Scope3Category.CAT_1_PURCHASED_GOODS_SERVICES]
        assert cat1_pct == Decimal("60.00")
        for cat, pct in bm.items():
            if cat != Scope3Category.CAT_1_PURCHASED_GOODS_SERVICES:
                assert pct <= cat1_pct

    def test_benchmark_financial_cat15_highest(self):
        """For financial, Cat 15 has the highest benchmark (80%)."""
        bm = INDUSTRY_BENCHMARKS[CompanyType.FINANCIAL]
        cat15_pct = bm[Scope3Category.CAT_15_INVESTMENTS]
        assert cat15_pct == Decimal("80.00")
        for cat, pct in bm.items():
            if cat != Scope3Category.CAT_15_INVESTMENTS:
                assert pct <= cat15_pct

    def test_benchmark_retailer_cat1_highest(self):
        """For retailer, Cat 1 has the highest benchmark (70%)."""
        bm = INDUSTRY_BENCHMARKS[CompanyType.RETAILER]
        cat1_pct = bm[Scope3Category.CAT_1_PURCHASED_GOODS_SERVICES]
        assert cat1_pct == Decimal("70.00")
        for cat, pct in bm.items():
            if cat != Scope3Category.CAT_1_PURCHASED_GOODS_SERVICES:
                assert pct <= cat1_pct

    def test_benchmark_energy_cat11_highest(self):
        """For energy, Cat 11 has the highest benchmark (55%)."""
        bm = INDUSTRY_BENCHMARKS[CompanyType.ENERGY]
        cat11_pct = bm[Scope3Category.CAT_11_USE_OF_SOLD_PRODUCTS]
        assert cat11_pct == Decimal("55.00")
        for cat, pct in bm.items():
            if cat != Scope3Category.CAT_11_USE_OF_SOLD_PRODUCTS:
                assert pct <= cat11_pct

    def test_benchmark_transport_cat3_highest(self):
        """For transport, Cat 3 (Fuel & Energy) has the highest benchmark (30%)."""
        bm = INDUSTRY_BENCHMARKS[CompanyType.TRANSPORT]
        cat3_pct = bm[Scope3Category.CAT_3_FUEL_ENERGY_ACTIVITIES]
        assert cat3_pct == Decimal("30.00")

    def test_benchmark_all_types_have_15_categories(self):
        """Every company type benchmark has exactly 15 category entries."""
        for ct in CompanyType:
            bm = INDUSTRY_BENCHMARKS[ct]
            assert len(bm) == 15, f"{ct.value} has {len(bm)} categories"

    def test_get_industry_benchmark_returns_copy(self, engine):
        """get_industry_benchmark returns a new dict (not mutable reference)."""
        bm1 = engine.get_industry_benchmark(CompanyType.MANUFACTURER)
        bm2 = engine.get_industry_benchmark(CompanyType.MANUFACTURER)
        assert bm1 == bm2
        assert bm1 is not bm2

    def test_get_industry_benchmark_invalid_type_raises(self, engine):
        """Requesting benchmark for invalid type raises an error."""
        with pytest.raises((ValueError, AttributeError)):
            engine.get_industry_benchmark("INVALID")  # type: ignore[arg-type]

    def test_relevance_matrix_all_types_have_15_categories(self):
        """Every company type relevance matrix has 15 category entries."""
        for ct in CompanyType:
            rel = COMPANY_TYPE_RELEVANCE[ct]
            assert len(rel) == 15, f"{ct.value} has {len(rel)} relevance entries"
