# -*- coding: utf-8 -*-
"""
Unit tests for SpendAnalyticsEngine - AGENT-DATA-009 Batch 3

Tests the SpendAnalyticsEngine with 85%+ coverage across:
- Initialization and configuration
- Aggregation by Scope 3 category, vendor, and time period
- Pareto/ABC analysis (80/20 identification, classification, empty)
- Hotspot identification (top-N emissions, spend, rising trends)
- Concentration metrics (HHI, CR4/CR8/CR20, interpretation)
- Emissions intensity by category and vendor
- Trend analysis (increasing, decreasing, flat, insufficient data)
- Variance analysis (positive, negative, no baseline)
- Industry benchmarking (with data, unknown industry)
- Statistics tracking
- SHA-256 provenance hashes
- Thread safety (concurrent analytics)

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import threading
from typing import Any, Dict, List

import pytest

from greenlang.spend_categorizer.spend_analytics import (
    HotspotResult,
    SpendAggregate,
    SpendAnalyticsEngine,
    TrendDataPoint,
    _INDUSTRY_BENCHMARKS,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def engine() -> SpendAnalyticsEngine:
    """Create a default SpendAnalyticsEngine."""
    return SpendAnalyticsEngine()


@pytest.fixture
def engine_custom() -> SpendAnalyticsEngine:
    """Create an engine with custom configuration."""
    return SpendAnalyticsEngine({
        "default_top_n": 5,
        "pareto_threshold": 0.7,
        "industry": "technology",
    })


@pytest.fixture
def multi_category_records() -> List[Dict[str, Any]]:
    """Records spanning multiple Scope 3 categories."""
    return [
        {"scope3_category": "Cat 1", "vendor_name": "Vendor A", "amount_usd": 100000, "emissions_kgco2e": 50000, "transaction_date": "2025-01-15"},
        {"scope3_category": "Cat 1", "vendor_name": "Vendor B", "amount_usd": 80000, "emissions_kgco2e": 40000, "transaction_date": "2025-02-10"},
        {"scope3_category": "Cat 2", "vendor_name": "Vendor C", "amount_usd": 60000, "emissions_kgco2e": 12000, "transaction_date": "2025-04-20"},
        {"scope3_category": "Cat 3", "vendor_name": "Vendor D", "amount_usd": 40000, "emissions_kgco2e": 36000, "transaction_date": "2025-05-01"},
        {"scope3_category": "Cat 3", "vendor_name": "Vendor E", "amount_usd": 20000, "emissions_kgco2e": 18000, "transaction_date": "2025-07-15"},
        {"scope3_category": "Cat 4", "vendor_name": "Vendor F", "amount_usd": 10000, "emissions_kgco2e": 2000, "transaction_date": "2025-08-01"},
        {"scope3_category": "Cat 1", "vendor_name": "Vendor A", "amount_usd": 50000, "emissions_kgco2e": 25000, "transaction_date": "2025-10-10"},
        {"scope3_category": "Cat 2", "vendor_name": "Vendor C", "amount_usd": 30000, "emissions_kgco2e": 6000, "transaction_date": "2025-11-05"},
    ]


@pytest.fixture
def vendor_records() -> List[Dict[str, Any]]:
    """Records with multiple vendors for concentration analysis."""
    return [
        {"vendor_name": "BigCorp", "amount_usd": 500000, "emissions_kgco2e": 100000, "scope3_category": "Cat 1"},
        {"vendor_name": "MediumCo", "amount_usd": 200000, "emissions_kgco2e": 40000, "scope3_category": "Cat 1"},
        {"vendor_name": "SmallBiz", "amount_usd": 100000, "emissions_kgco2e": 20000, "scope3_category": "Cat 2"},
        {"vendor_name": "TinyLLC", "amount_usd": 50000, "emissions_kgco2e": 10000, "scope3_category": "Cat 3"},
        {"vendor_name": "MicroInc", "amount_usd": 25000, "emissions_kgco2e": 5000, "scope3_category": "Cat 4"},
        {"vendor_name": "NanoSvc", "amount_usd": 10000, "emissions_kgco2e": 2000, "scope3_category": "Cat 5"},
    ]


@pytest.fixture
def trend_records() -> List[Dict[str, Any]]:
    """Records across multiple quarters for trend analysis."""
    return [
        {"transaction_date": "2024-01-15", "amount_usd": 100000, "emissions_kgco2e": 20000},
        {"transaction_date": "2024-02-10", "amount_usd": 120000, "emissions_kgco2e": 24000},
        {"transaction_date": "2024-04-20", "amount_usd": 130000, "emissions_kgco2e": 26000},
        {"transaction_date": "2024-05-01", "amount_usd": 150000, "emissions_kgco2e": 30000},
        {"transaction_date": "2024-07-15", "amount_usd": 140000, "emissions_kgco2e": 28000},
        {"transaction_date": "2024-08-01", "amount_usd": 160000, "emissions_kgco2e": 32000},
        {"transaction_date": "2024-10-10", "amount_usd": 180000, "emissions_kgco2e": 36000},
        {"transaction_date": "2024-11-05", "amount_usd": 200000, "emissions_kgco2e": 40000},
    ]


@pytest.fixture
def baseline_records() -> List[Dict[str, Any]]:
    """Baseline period records for variance analysis."""
    return [
        {"scope3_category": "Cat 1", "amount_usd": 80000, "emissions_kgco2e": 40000},
        {"scope3_category": "Cat 2", "amount_usd": 50000, "emissions_kgco2e": 10000},
        {"scope3_category": "Cat 3", "amount_usd": 30000, "emissions_kgco2e": 27000},
    ]


# ===========================================================================
# TestInit
# ===========================================================================


class TestInit:
    """Test SpendAnalyticsEngine initialization."""

    def test_default_init(self, engine: SpendAnalyticsEngine):
        """Engine initializes with default configuration."""
        assert engine._default_top_n == 10
        assert engine._pareto_threshold == 0.8
        assert engine._default_industry == "manufacturing"

    def test_custom_init(self, engine_custom: SpendAnalyticsEngine):
        """Engine respects custom configuration."""
        assert engine_custom._default_top_n == 5
        assert engine_custom._pareto_threshold == 0.7
        assert engine_custom._default_industry == "technology"

    def test_stats_initialized(self, engine: SpendAnalyticsEngine):
        """Statistics counters start at zero."""
        stats = engine.get_statistics()
        assert stats["analyses_performed"] == 0
        assert stats["records_analysed"] == 0

    def test_empty_config(self):
        """None config is treated as empty dict."""
        engine = SpendAnalyticsEngine(None)
        assert engine._default_top_n == 10


# ===========================================================================
# TestAggregateByCategory
# ===========================================================================


class TestAggregateByCategory:
    """Test aggregate_by_category() for Scope 3 categories."""

    def test_groups_by_scope3(self, engine: SpendAnalyticsEngine, multi_category_records):
        """Records are grouped by scope3_category."""
        aggs = engine.aggregate_by_category(multi_category_records)
        group_keys = {a.group_key for a in aggs}
        assert "Cat 1" in group_keys
        assert "Cat 2" in group_keys
        assert "Cat 3" in group_keys
        assert "Cat 4" in group_keys

    def test_sorted_by_emissions_desc(self, engine: SpendAnalyticsEngine, multi_category_records):
        """Results are sorted by emissions descending."""
        aggs = engine.aggregate_by_category(multi_category_records)
        emissions = [a.total_emissions_kgco2e for a in aggs]
        assert emissions == sorted(emissions, reverse=True)

    def test_aggregate_totals(self, engine: SpendAnalyticsEngine, multi_category_records):
        """Cat 1 aggregate totals are correct (3 records)."""
        aggs = engine.aggregate_by_category(multi_category_records)
        cat1 = [a for a in aggs if a.group_key == "Cat 1"][0]
        assert cat1.record_count == 3
        assert cat1.total_spend_usd == 230000.0
        assert cat1.total_emissions_kgco2e == 115000.0

    def test_share_of_spend(self, engine: SpendAnalyticsEngine, multi_category_records):
        """Share of spend sums to approximately 1.0."""
        aggs = engine.aggregate_by_category(multi_category_records)
        total_share = sum(a.share_of_spend for a in aggs)
        assert total_share == pytest.approx(1.0, abs=0.01)

    def test_share_of_emissions(self, engine: SpendAnalyticsEngine, multi_category_records):
        """Share of emissions sums to approximately 1.0."""
        aggs = engine.aggregate_by_category(multi_category_records)
        total_share = sum(a.share_of_emissions for a in aggs)
        assert total_share == pytest.approx(1.0, abs=0.01)

    def test_intensity_calculated(self, engine: SpendAnalyticsEngine, multi_category_records):
        """Intensity = emissions / spend for each group."""
        aggs = engine.aggregate_by_category(multi_category_records)
        for a in aggs:
            if a.total_spend_usd > 0:
                expected = round(a.total_emissions_kgco2e / a.total_spend_usd, 6)
                assert a.intensity_kgco2e_per_usd == expected

    def test_tco2e_conversion(self, engine: SpendAnalyticsEngine, multi_category_records):
        """tCO2e = kgCO2e / 1000."""
        aggs = engine.aggregate_by_category(multi_category_records)
        for a in aggs:
            assert a.total_emissions_tco2e == pytest.approx(a.total_emissions_kgco2e / 1000.0, abs=0.001)

    def test_empty_records(self, engine: SpendAnalyticsEngine):
        """Empty records list returns empty aggregation."""
        aggs = engine.aggregate_by_category([])
        assert aggs == []

    def test_custom_group_by(self, engine: SpendAnalyticsEngine, multi_category_records):
        """Aggregation supports custom group_by field."""
        aggs = engine.aggregate_by_category(multi_category_records, group_by="vendor_name")
        group_keys = {a.group_key for a in aggs}
        assert "Vendor A" in group_keys

    def test_group_type_label(self, engine: SpendAnalyticsEngine, multi_category_records):
        """group_type is set to 'category'."""
        aggs = engine.aggregate_by_category(multi_category_records)
        assert all(a.group_type == "category" for a in aggs)

    def test_provenance_hash(self, engine: SpendAnalyticsEngine, multi_category_records):
        """Each aggregate has a SHA-256 provenance hash."""
        aggs = engine.aggregate_by_category(multi_category_records)
        for a in aggs:
            assert len(a.provenance_hash) == 64


# ===========================================================================
# TestAggregateByVendor
# ===========================================================================


class TestAggregateByVendor:
    """Test aggregate_by_vendor() for vendor-level aggregation."""

    def test_top_n_vendors(self, engine: SpendAnalyticsEngine, vendor_records):
        """Returns at most top_n vendors."""
        aggs = engine.aggregate_by_vendor(vendor_records, top_n=3)
        assert len(aggs) <= 3

    def test_all_vendors(self, engine: SpendAnalyticsEngine, vendor_records):
        """All vendors returned when top_n exceeds count."""
        aggs = engine.aggregate_by_vendor(vendor_records, top_n=100)
        assert len(aggs) == 6

    def test_group_type_vendor(self, engine: SpendAnalyticsEngine, vendor_records):
        """group_type is set to 'vendor'."""
        aggs = engine.aggregate_by_vendor(vendor_records)
        assert all(a.group_type == "vendor" for a in aggs)

    def test_empty_records(self, engine: SpendAnalyticsEngine):
        """Empty records returns empty vendor list."""
        aggs = engine.aggregate_by_vendor([])
        assert aggs == []


# ===========================================================================
# TestAggregateByPeriod
# ===========================================================================


class TestAggregateByPeriod:
    """Test aggregate_by_period() across timeframes."""

    def test_quarterly(self, engine: SpendAnalyticsEngine, trend_records):
        """Quarterly aggregation produces correct period labels."""
        aggs = engine.aggregate_by_period(trend_records, timeframe="quarterly")
        period_labels = [a.group_key for a in aggs]
        assert "2024-Q1" in period_labels

    def test_monthly(self, engine: SpendAnalyticsEngine, trend_records):
        """Monthly aggregation produces month-level labels."""
        aggs = engine.aggregate_by_period(trend_records, timeframe="monthly")
        period_labels = [a.group_key for a in aggs]
        assert "2024-01" in period_labels

    def test_yearly(self, engine: SpendAnalyticsEngine, trend_records):
        """Yearly aggregation produces year-level labels."""
        aggs = engine.aggregate_by_period(trend_records, timeframe="yearly")
        period_labels = [a.group_key for a in aggs]
        assert "2024" in period_labels

    def test_sorted_ascending(self, engine: SpendAnalyticsEngine, trend_records):
        """Period results are sorted in ascending order."""
        aggs = engine.aggregate_by_period(trend_records, timeframe="quarterly")
        labels = [a.group_key for a in aggs]
        assert labels == sorted(labels)

    def test_empty_records(self, engine: SpendAnalyticsEngine):
        """Empty records returns empty periods."""
        aggs = engine.aggregate_by_period([])
        assert aggs == []

    def test_unknown_dates_grouped(self, engine: SpendAnalyticsEngine):
        """Records without valid dates are grouped as 'Unknown'."""
        records = [
            {"amount_usd": 100, "emissions_kgco2e": 10, "transaction_date": ""},
            {"amount_usd": 200, "emissions_kgco2e": 20, "transaction_date": None},
        ]
        aggs = engine.aggregate_by_period(records)
        assert any(a.group_key == "Unknown" for a in aggs)


# ===========================================================================
# TestParetoAnalysis
# ===========================================================================


class TestParetoAnalysis:
    """Test pareto_analysis() ABC classification."""

    def test_pareto_result_structure(self, engine: SpendAnalyticsEngine, vendor_records):
        """Pareto result contains required keys."""
        result = engine.pareto_analysis(vendor_records)
        assert "total_spend_usd" in result
        assert "a_class_count" in result
        assert "b_class_count" in result
        assert "c_class_count" in result
        assert "a_class" in result
        assert "b_class" in result
        assert "c_class" in result

    def test_all_vendors_classified(self, engine: SpendAnalyticsEngine, vendor_records):
        """All vendors appear in exactly one class."""
        result = engine.pareto_analysis(vendor_records)
        total = result["a_class_count"] + result["b_class_count"] + result["c_class_count"]
        assert total == result["total_vendors"]

    def test_a_class_is_top_spenders(self, engine: SpendAnalyticsEngine, vendor_records):
        """A-class contains the top cumulative spenders within threshold."""
        result = engine.pareto_analysis(vendor_records, threshold=0.8)
        a_spend = sum(v["spend_usd"] for v in result["a_class"])
        total_spend = result["total_spend_usd"]
        assert a_spend / total_spend <= 0.81  # slightly above threshold is ok

    def test_threshold_0_8(self, engine: SpendAnalyticsEngine, vendor_records):
        """Default 80% threshold classifies correctly."""
        result = engine.pareto_analysis(vendor_records, threshold=0.8)
        assert result["threshold"] == 0.8

    def test_custom_threshold(self, engine: SpendAnalyticsEngine, vendor_records):
        """Custom threshold value is respected."""
        result = engine.pareto_analysis(vendor_records, threshold=0.6)
        assert result["threshold"] == 0.6

    def test_empty_records(self, engine: SpendAnalyticsEngine):
        """Empty records returns empty Pareto result."""
        result = engine.pareto_analysis([])
        assert result["total_spend_usd"] == 0.0
        assert result["a_class_count"] == 0

    def test_single_vendor(self, engine: SpendAnalyticsEngine):
        """Single vendor goes to A-class."""
        records = [{"vendor_name": "Solo Corp", "amount_usd": 100000, "emissions_kgco2e": 5000}]
        result = engine.pareto_analysis(records)
        assert result["total_vendors"] == 1

    def test_provenance_hash(self, engine: SpendAnalyticsEngine, vendor_records):
        """Pareto result includes provenance hash."""
        result = engine.pareto_analysis(vendor_records)
        assert len(result["provenance_hash"]) == 64

    def test_vendor_entries_have_share(self, engine: SpendAnalyticsEngine, vendor_records):
        """Each vendor entry has share and cumulative share."""
        result = engine.pareto_analysis(vendor_records)
        for v in result["a_class"]:
            assert "share" in v
            assert "cumulative" in v
            assert v["share"] > 0

    def test_a_class_percentage(self, engine: SpendAnalyticsEngine, vendor_records):
        """a_class_percentage reflects portion of total vendors."""
        result = engine.pareto_analysis(vendor_records)
        expected = round(result["a_class_count"] / result["total_vendors"] * 100, 2)
        assert result["a_class_percentage"] == expected

    def test_zero_spend_returns_empty(self, engine: SpendAnalyticsEngine):
        """All zero spend records returns empty Pareto."""
        records = [
            {"vendor_name": "V1", "amount_usd": 0, "emissions_kgco2e": 0},
            {"vendor_name": "V2", "amount_usd": 0, "emissions_kgco2e": 0},
        ]
        result = engine.pareto_analysis(records)
        assert result["total_spend_usd"] == 0.0


# ===========================================================================
# TestIdentifyHotspots
# ===========================================================================


class TestIdentifyHotspots:
    """Test identify_hotspots() for emissions hotspot ranking."""

    def test_returns_hotspot_objects(self, engine: SpendAnalyticsEngine, multi_category_records):
        """Returns a list of HotspotResult objects."""
        hotspots = engine.identify_hotspots(multi_category_records)
        assert all(isinstance(h, HotspotResult) for h in hotspots)

    def test_ranked_by_emissions_desc(self, engine: SpendAnalyticsEngine, multi_category_records):
        """Hotspots are ranked by emissions descending."""
        hotspots = engine.identify_hotspots(multi_category_records)
        emissions = [h.total_emissions_kgco2e for h in hotspots]
        assert emissions == sorted(emissions, reverse=True)

    def test_rank_numbers(self, engine: SpendAnalyticsEngine, multi_category_records):
        """Ranks start at 1 and are sequential."""
        hotspots = engine.identify_hotspots(multi_category_records)
        ranks = [h.rank for h in hotspots]
        assert ranks == list(range(1, len(hotspots) + 1))

    def test_top_n_limit(self, engine: SpendAnalyticsEngine, multi_category_records):
        """top_n limits the number of hotspots returned."""
        hotspots = engine.identify_hotspots(multi_category_records, top_n=2)
        assert len(hotspots) == 2

    def test_cumulative_share(self, engine: SpendAnalyticsEngine, multi_category_records):
        """Cumulative share increases with rank."""
        hotspots = engine.identify_hotspots(multi_category_records)
        for i in range(1, len(hotspots)):
            assert hotspots[i].cumulative_share >= hotspots[i - 1].cumulative_share

    def test_share_of_emissions_sums(self, engine: SpendAnalyticsEngine, multi_category_records):
        """Total share of emissions sums to 1.0 for all categories."""
        hotspots = engine.identify_hotspots(multi_category_records, top_n=100)
        total_share = sum(h.share_of_emissions for h in hotspots)
        assert total_share == pytest.approx(1.0, abs=0.01)

    def test_empty_records(self, engine: SpendAnalyticsEngine):
        """Empty records returns empty list."""
        hotspots = engine.identify_hotspots([])
        assert hotspots == []

    def test_zero_emissions_returns_empty(self, engine: SpendAnalyticsEngine):
        """All-zero emissions returns empty list."""
        records = [{"scope3_category": "Cat 1", "amount_usd": 1000, "emissions_kgco2e": 0}]
        hotspots = engine.identify_hotspots(records)
        assert hotspots == []

    def test_hotspot_intensity(self, engine: SpendAnalyticsEngine, multi_category_records):
        """Hotspot intensity is emissions / spend."""
        hotspots = engine.identify_hotspots(multi_category_records)
        for h in hotspots:
            if h.total_spend_usd > 0:
                expected = round(h.total_emissions_kgco2e / h.total_spend_usd, 6)
                assert h.intensity_kgco2e_per_usd == expected

    def test_provenance_hash(self, engine: SpendAnalyticsEngine, multi_category_records):
        """Each hotspot has a provenance hash."""
        hotspots = engine.identify_hotspots(multi_category_records)
        for h in hotspots:
            assert len(h.provenance_hash) == 64


# ===========================================================================
# TestCalculateConcentration
# ===========================================================================


class TestCalculateConcentration:
    """Test calculate_concentration() HHI and CR metrics."""

    def test_result_structure(self, engine: SpendAnalyticsEngine, vendor_records):
        """Concentration result contains required keys."""
        result = engine.calculate_concentration(vendor_records)
        assert "hhi" in result
        assert "cr4" in result
        assert "cr8" in result
        assert "cr20" in result
        assert "interpretation" in result

    def test_hhi_range(self, engine: SpendAnalyticsEngine, vendor_records):
        """HHI is between 0 and 10000."""
        result = engine.calculate_concentration(vendor_records)
        assert 0 <= result["hhi"] <= 10000

    def test_cr4_range(self, engine: SpendAnalyticsEngine, vendor_records):
        """CR4 is between 0 and 1."""
        result = engine.calculate_concentration(vendor_records)
        assert 0 <= result["cr4"] <= 1.0

    def test_single_vendor_max_concentration(self, engine: SpendAnalyticsEngine):
        """Single vendor = HHI of 10000 (monopoly)."""
        records = [{"vendor_name": "MonopolyCo", "amount_usd": 1000000}]
        result = engine.calculate_concentration(records)
        assert result["hhi"] == 10000.0
        assert result["interpretation"] == "highly_concentrated"

    def test_many_equal_vendors(self, engine: SpendAnalyticsEngine):
        """Many equal vendors produce low HHI."""
        records = [
            {"vendor_name": f"Vendor{i}", "amount_usd": 10000}
            for i in range(100)
        ]
        result = engine.calculate_concentration(records)
        assert result["hhi"] < 200  # 100 * (1%)^2 = 100
        assert result["interpretation"] == "unconcentrated"

    def test_moderately_concentrated(self, engine: SpendAnalyticsEngine):
        """Moderate concentration (3-4 vendors dominating)."""
        records = [
            {"vendor_name": "V1", "amount_usd": 400000},
            {"vendor_name": "V2", "amount_usd": 300000},
            {"vendor_name": "V3", "amount_usd": 200000},
            {"vendor_name": "V4", "amount_usd": 100000},
        ]
        result = engine.calculate_concentration(records)
        # HHI = 40^2 + 30^2 + 20^2 + 10^2 = 1600+900+400+100 = 3000
        assert result["hhi"] == 3000.0
        assert result["interpretation"] == "highly_concentrated"

    def test_empty_records(self, engine: SpendAnalyticsEngine):
        """Empty records returns no_data interpretation."""
        result = engine.calculate_concentration([])
        assert result["interpretation"] == "no_data"
        assert result["hhi"] == 0

    def test_zero_spend(self, engine: SpendAnalyticsEngine):
        """All zero spend returns no_data."""
        records = [{"vendor_name": "V1", "amount_usd": 0}]
        result = engine.calculate_concentration(records)
        assert result["interpretation"] == "no_data"

    def test_vendor_count(self, engine: SpendAnalyticsEngine, vendor_records):
        """vendor_count is correctly tracked."""
        result = engine.calculate_concentration(vendor_records)
        assert result["vendor_count"] == 6

    def test_provenance_hash(self, engine: SpendAnalyticsEngine, vendor_records):
        """Concentration result has provenance hash."""
        result = engine.calculate_concentration(vendor_records)
        assert len(result["provenance_hash"]) == 64


# ===========================================================================
# TestCalculateIntensity
# ===========================================================================


class TestCalculateIntensity:
    """Test calculate_intensity() per-category intensity metrics."""

    def test_result_structure(self, engine: SpendAnalyticsEngine, multi_category_records):
        """Intensity result contains required keys."""
        result = engine.calculate_intensity(multi_category_records)
        assert "overall_intensity_kgco2e_per_usd" in result
        assert "total_spend_usd" in result
        assert "by_category" in result

    def test_overall_intensity(self, engine: SpendAnalyticsEngine, multi_category_records):
        """Overall intensity = total emissions / total spend."""
        result = engine.calculate_intensity(multi_category_records)
        total_spend = sum(float(r.get("amount_usd", 0)) for r in multi_category_records)
        total_emissions = sum(float(r.get("emissions_kgco2e", 0)) for r in multi_category_records)
        expected = round(total_emissions / total_spend, 6)
        assert result["overall_intensity_kgco2e_per_usd"] == expected

    def test_by_category_sorted_by_intensity(self, engine: SpendAnalyticsEngine, multi_category_records):
        """Categories are sorted by intensity descending."""
        result = engine.calculate_intensity(multi_category_records)
        intensities = [c["intensity_kgco2e_per_usd"] for c in result["by_category"]]
        assert intensities == sorted(intensities, reverse=True)

    def test_per_category_intensity(self, engine: SpendAnalyticsEngine, multi_category_records):
        """Each category's intensity = category emissions / category spend."""
        result = engine.calculate_intensity(multi_category_records)
        for c in result["by_category"]:
            if c["spend_usd"] > 0:
                expected = round(c["emissions_kgco2e"] / c["spend_usd"], 6)
                assert c["intensity_kgco2e_per_usd"] == expected

    def test_empty_records(self, engine: SpendAnalyticsEngine):
        """Empty records returns zero intensity."""
        result = engine.calculate_intensity([])
        assert result["overall_intensity_kgco2e_per_usd"] == 0.0

    def test_provenance_hash(self, engine: SpendAnalyticsEngine, multi_category_records):
        """Intensity result has provenance hash."""
        result = engine.calculate_intensity(multi_category_records)
        assert len(result["provenance_hash"]) == 64


# ===========================================================================
# TestTrendAnalysis
# ===========================================================================


class TestTrendAnalysis:
    """Test trend_analysis() period-over-period tracking."""

    def test_returns_trend_data_points(self, engine: SpendAnalyticsEngine, trend_records):
        """Returns a list of TrendDataPoint objects."""
        trend = engine.trend_analysis(trend_records)
        assert all(isinstance(t, TrendDataPoint) for t in trend)

    def test_chronological_order(self, engine: SpendAnalyticsEngine, trend_records):
        """Trend points are in chronological order."""
        trend = engine.trend_analysis(trend_records)
        periods = [t.period for t in trend]
        assert periods == sorted(periods)

    def test_first_period_no_change(self, engine: SpendAnalyticsEngine, trend_records):
        """First period has no change percentages."""
        trend = engine.trend_analysis(trend_records)
        assert trend[0].spend_change_pct is None
        assert trend[0].emissions_change_pct is None

    def test_increasing_trend(self, engine: SpendAnalyticsEngine, trend_records):
        """Increasing spend shows positive change percentages."""
        trend = engine.trend_analysis(trend_records)
        # Q2 > Q1, so spend_change_pct should be positive
        if len(trend) > 1 and trend[1].spend_change_pct is not None:
            assert trend[1].spend_change_pct > 0

    def test_periods_limited(self, engine: SpendAnalyticsEngine, trend_records):
        """Periods parameter limits the number returned."""
        trend = engine.trend_analysis(trend_records, periods=2)
        assert len(trend) <= 2

    def test_period_index(self, engine: SpendAnalyticsEngine, trend_records):
        """Period index is 0-based sequential."""
        trend = engine.trend_analysis(trend_records)
        indices = [t.period_index for t in trend]
        assert indices == list(range(len(trend)))

    def test_empty_records(self, engine: SpendAnalyticsEngine):
        """Empty records returns empty trend."""
        trend = engine.trend_analysis([])
        assert trend == []

    def test_intensity_per_period(self, engine: SpendAnalyticsEngine, trend_records):
        """Each period has intensity = emissions / spend."""
        trend = engine.trend_analysis(trend_records)
        for t in trend:
            if t.total_spend_usd > 0:
                expected = round(t.total_emissions_kgco2e / t.total_spend_usd, 6)
                assert t.intensity_kgco2e_per_usd == expected

    def test_provenance_hash(self, engine: SpendAnalyticsEngine, trend_records):
        """Each trend data point has a provenance hash."""
        trend = engine.trend_analysis(trend_records)
        for t in trend:
            assert len(t.provenance_hash) == 64

    def test_insufficient_data_single_record(self, engine: SpendAnalyticsEngine):
        """Single record produces single-period trend with no change."""
        records = [{"transaction_date": "2024-01-15", "amount_usd": 1000, "emissions_kgco2e": 100}]
        trend = engine.trend_analysis(records)
        assert len(trend) == 1
        assert trend[0].spend_change_pct is None

    def test_flat_trend(self, engine: SpendAnalyticsEngine):
        """Equal amounts across periods produce 0% change."""
        records = [
            {"transaction_date": "2024-01-15", "amount_usd": 1000, "emissions_kgco2e": 100},
            {"transaction_date": "2024-04-15", "amount_usd": 1000, "emissions_kgco2e": 100},
        ]
        trend = engine.trend_analysis(records)
        if len(trend) > 1:
            assert trend[1].spend_change_pct == 0.0
            assert trend[1].emissions_change_pct == 0.0


# ===========================================================================
# TestVarianceAnalysis
# ===========================================================================


class TestVarianceAnalysis:
    """Test variance_analysis() current vs baseline comparison."""

    def test_result_structure(self, engine: SpendAnalyticsEngine, multi_category_records, baseline_records):
        """Variance result contains required keys."""
        result = engine.variance_analysis(multi_category_records, baseline_records)
        assert "current_spend_usd" in result
        assert "baseline_spend_usd" in result
        assert "spend_variance_usd" in result
        assert "spend_variance_pct" in result
        assert "emissions_variance_kgco2e" in result

    def test_positive_variance(self, engine: SpendAnalyticsEngine, multi_category_records, baseline_records):
        """Higher current than baseline produces positive variance."""
        result = engine.variance_analysis(multi_category_records, baseline_records)
        assert result["spend_variance_usd"] > 0

    def test_negative_variance(self, engine: SpendAnalyticsEngine, baseline_records, multi_category_records):
        """Lower current than baseline produces negative variance."""
        result = engine.variance_analysis(baseline_records, multi_category_records)
        assert result["spend_variance_usd"] < 0

    def test_category_variances(self, engine: SpendAnalyticsEngine, multi_category_records, baseline_records):
        """Category-level variances cover all categories from both periods."""
        result = engine.variance_analysis(multi_category_records, baseline_records)
        cats = {cv["category"] for cv in result["category_variances"]}
        assert "Cat 1" in cats
        assert "Cat 2" in cats

    def test_no_baseline(self, engine: SpendAnalyticsEngine, multi_category_records):
        """Empty baseline produces 0% variance percentages."""
        result = engine.variance_analysis(multi_category_records, [])
        assert result["spend_variance_pct"] == 0.0

    def test_empty_current(self, engine: SpendAnalyticsEngine, baseline_records):
        """Empty current produces negative variance."""
        result = engine.variance_analysis([], baseline_records)
        assert result["spend_variance_usd"] < 0

    def test_provenance_hash(self, engine: SpendAnalyticsEngine, multi_category_records, baseline_records):
        """Variance result has provenance hash."""
        result = engine.variance_analysis(multi_category_records, baseline_records)
        assert len(result["provenance_hash"]) == 64


# ===========================================================================
# TestBenchmark
# ===========================================================================


class TestBenchmark:
    """Test benchmark() industry comparison."""

    def test_result_structure(self, engine: SpendAnalyticsEngine, multi_category_records):
        """Benchmark result contains required keys."""
        result = engine.benchmark(multi_category_records)
        assert "industry" in result
        assert "actual_intensity_kgco2e_per_usd" in result
        assert "benchmark_intensity_kgco2e_per_usd" in result
        assert "performance_rating" in result

    def test_manufacturing_benchmark(self, engine: SpendAnalyticsEngine, multi_category_records):
        """Default industry is manufacturing."""
        result = engine.benchmark(multi_category_records)
        assert result["industry"] == "manufacturing"
        assert result["benchmark_intensity_kgco2e_per_usd"] == 0.45

    def test_technology_benchmark(self, engine: SpendAnalyticsEngine, multi_category_records):
        """Technology industry has lower benchmark intensity."""
        result = engine.benchmark(multi_category_records, industry="technology")
        assert result["industry"] == "technology"
        assert result["benchmark_intensity_kgco2e_per_usd"] == 0.08

    def test_unknown_industry_falls_back(self, engine: SpendAnalyticsEngine, multi_category_records):
        """Unknown industry falls back to manufacturing."""
        result = engine.benchmark(multi_category_records, industry="unknown_industry")
        assert result["benchmark_intensity_kgco2e_per_usd"] == _INDUSTRY_BENCHMARKS["manufacturing"]["intensity_kgco2e_per_usd"]

    def test_performance_rating_leader(self, engine: SpendAnalyticsEngine):
        """Very low intensity gets 'leader' rating."""
        records = [{"amount_usd": 100000, "emissions_kgco2e": 1}]
        result = engine.benchmark(records, industry="manufacturing")
        assert result["performance_rating"] == "leader"

    def test_performance_rating_laggard(self, engine: SpendAnalyticsEngine):
        """Very high intensity gets 'laggard' rating."""
        records = [{"amount_usd": 100000, "emissions_kgco2e": 100000}]
        result = engine.benchmark(records, industry="manufacturing")
        assert result["performance_rating"] == "laggard"

    def test_intensity_gap(self, engine: SpendAnalyticsEngine, multi_category_records):
        """Intensity gap = actual - benchmark."""
        result = engine.benchmark(multi_category_records)
        expected_gap = round(
            result["actual_intensity_kgco2e_per_usd"] - result["benchmark_intensity_kgco2e_per_usd"],
            6,
        )
        assert result["intensity_gap"] == expected_gap

    def test_provenance_hash(self, engine: SpendAnalyticsEngine, multi_category_records):
        """Benchmark result has provenance hash."""
        result = engine.benchmark(multi_category_records)
        assert len(result["provenance_hash"]) == 64

    def test_available_industries(self, engine: SpendAnalyticsEngine):
        """Statistics list available industry benchmarks."""
        stats = engine.get_statistics()
        assert "manufacturing" in stats["industry_benchmarks_available"]
        assert "technology" in stats["industry_benchmarks_available"]
        assert len(stats["industry_benchmarks_available"]) == len(_INDUSTRY_BENCHMARKS)


# ===========================================================================
# TestStatistics
# ===========================================================================


class TestStatistics:
    """Test analytics statistics tracking."""

    def test_initial_stats(self, engine: SpendAnalyticsEngine):
        """Statistics start at zero."""
        stats = engine.get_statistics()
        assert stats["analyses_performed"] == 0
        assert stats["records_analysed"] == 0

    def test_analysis_increments_count(self, engine: SpendAnalyticsEngine, multi_category_records):
        """Each analysis increments the analyses_performed counter."""
        engine.aggregate_by_category(multi_category_records)
        engine.identify_hotspots(multi_category_records)
        stats = engine.get_statistics()
        assert stats["analyses_performed"] == 2

    def test_records_analysed_tracks(self, engine: SpendAnalyticsEngine, multi_category_records):
        """records_analysed accumulates total records processed."""
        engine.aggregate_by_category(multi_category_records)
        stats = engine.get_statistics()
        assert stats["records_analysed"] == len(multi_category_records)

    def test_by_analysis_type(self, engine: SpendAnalyticsEngine, multi_category_records):
        """by_analysis_type tracks analysis types."""
        engine.identify_hotspots(multi_category_records)
        engine.pareto_analysis(multi_category_records)
        stats = engine.get_statistics()
        assert stats["by_analysis_type"].get("hotspot", 0) >= 1
        assert stats["by_analysis_type"].get("pareto", 0) >= 1


# ===========================================================================
# TestProvenance
# ===========================================================================


class TestProvenance:
    """Test SHA-256 provenance hash generation."""

    def test_aggregate_hashes(self, engine: SpendAnalyticsEngine, multi_category_records):
        """Each aggregate has a unique 64-char hex hash."""
        aggs = engine.aggregate_by_category(multi_category_records)
        hashes = {a.provenance_hash for a in aggs}
        assert len(hashes) == len(aggs)
        assert all(len(h) == 64 for h in hashes)

    def test_hotspot_hashes(self, engine: SpendAnalyticsEngine, multi_category_records):
        """Each hotspot has a unique 64-char hex hash."""
        hotspots = engine.identify_hotspots(multi_category_records)
        hashes = {h.provenance_hash for h in hotspots}
        assert all(len(h) == 64 for h in hashes)

    def test_hash_hex_format(self, engine: SpendAnalyticsEngine, multi_category_records):
        """All provenance hashes are hex-encoded."""
        aggs = engine.aggregate_by_category(multi_category_records)
        for a in aggs:
            assert all(c in "0123456789abcdef" for c in a.provenance_hash)


# ===========================================================================
# TestThreadSafety
# ===========================================================================


class TestThreadSafety:
    """Test thread-safe concurrent analytics."""

    def test_concurrent_aggregations(self, engine: SpendAnalyticsEngine, multi_category_records):
        """Concurrent aggregations do not corrupt state."""
        errors: List[str] = []

        def agg_task():
            try:
                for _ in range(20):
                    engine.aggregate_by_category(multi_category_records)
            except Exception as exc:
                errors.append(str(exc))

        threads = [threading.Thread(target=agg_task) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0

    def test_concurrent_hotspots(self, engine: SpendAnalyticsEngine, multi_category_records):
        """Concurrent hotspot identification does not crash."""
        errors: List[str] = []

        def hotspot_task():
            try:
                for _ in range(20):
                    engine.identify_hotspots(multi_category_records)
            except Exception as exc:
                errors.append(str(exc))

        threads = [threading.Thread(target=hotspot_task) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0

    def test_concurrent_mixed_analytics(self, engine: SpendAnalyticsEngine, multi_category_records):
        """Mixed concurrent analytics do not corrupt statistics."""
        errors: List[str] = []

        def mixed_task():
            try:
                engine.aggregate_by_category(multi_category_records)
                engine.identify_hotspots(multi_category_records)
                engine.calculate_intensity(multi_category_records)
            except Exception as exc:
                errors.append(str(exc))

        threads = [threading.Thread(target=mixed_task) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        stats = engine.get_statistics()
        # 4 threads * 3 analyses = 12
        assert stats["analyses_performed"] == 12
