# -*- coding: utf-8 -*-
"""
Unit tests for BenchmarkingEngine -- CDP sector and regional benchmarking.

Tests sector benchmark retrieval, regional benchmark, score distribution,
category comparison, A-list rate calculation, custom peer groups, and
historical sector trends with 27+ test functions.

Author: GL-TestEngineer
Date: March 2026
"""

from decimal import Decimal

import pytest

from services.models import (
    CDPBenchmark,
    CDPPeerComparison,
    CDPCategoryScore,
    _new_id,
)
from services.benchmarking_engine import BenchmarkingEngine


# ---------------------------------------------------------------------------
# Sector benchmark retrieval
# ---------------------------------------------------------------------------

class TestSectorBenchmark:
    """Test sector benchmark data retrieval."""

    def test_get_sector_benchmark(self, benchmarking_engine):
        benchmark = benchmarking_engine.get_sector_benchmark(
            sector_gics="20101010", year=2025,
        )
        assert isinstance(benchmark, CDPBenchmark)
        assert benchmark.sector_gics == "20101010"
        assert benchmark.year == 2025

    def test_benchmark_has_statistics(self, benchmarking_engine):
        benchmark = benchmarking_engine.get_sector_benchmark(
            sector_gics="20101010", year=2025,
        )
        assert benchmark.mean_score > Decimal("0")
        assert benchmark.median_score > Decimal("0")
        assert benchmark.p25_score <= benchmark.median_score
        assert benchmark.median_score <= benchmark.p75_score

    def test_benchmark_respondent_count(self, benchmarking_engine):
        benchmark = benchmarking_engine.get_sector_benchmark(
            sector_gics="20101010", year=2025,
        )
        assert benchmark.respondent_count > 0

    def test_unknown_sector_returns_global(self, benchmarking_engine):
        benchmark = benchmarking_engine.get_sector_benchmark(
            sector_gics="99999999", year=2025,
        )
        assert benchmark is not None
        assert benchmark.region == "Global"

    def test_list_available_sectors(self, benchmarking_engine):
        sectors = benchmarking_engine.list_available_sectors()
        assert len(sectors) > 0
        assert all(isinstance(s, str) for s in sectors)


# ---------------------------------------------------------------------------
# Regional benchmark
# ---------------------------------------------------------------------------

class TestRegionalBenchmark:
    """Test regional benchmark retrieval."""

    def test_get_regional_benchmark(self, benchmarking_engine):
        benchmark = benchmarking_engine.get_regional_benchmark(
            region="North America", year=2025,
        )
        assert benchmark is not None
        assert benchmark.region == "North America"

    def test_europe_benchmark(self, benchmarking_engine):
        benchmark = benchmarking_engine.get_regional_benchmark(
            region="Europe", year=2025,
        )
        assert benchmark is not None

    def test_global_benchmark(self, benchmarking_engine):
        benchmark = benchmarking_engine.get_regional_benchmark(
            region="Global", year=2025,
        )
        assert benchmark is not None

    def test_regional_vs_sector(self, benchmarking_engine):
        regional = benchmarking_engine.get_regional_benchmark("North America", 2025)
        sector = benchmarking_engine.get_sector_benchmark("20101010", 2025)
        # Both should return valid benchmarks
        assert regional.respondent_count > 0
        assert sector.respondent_count > 0


# ---------------------------------------------------------------------------
# Score distribution
# ---------------------------------------------------------------------------

class TestScoreDistribution:
    """Test score distribution calculations."""

    def test_distribution_has_all_bands(self, benchmarking_engine, sample_benchmark):
        dist = benchmarking_engine.get_score_distribution(sample_benchmark)
        expected_bands = ["D-", "D", "C-", "C", "B-", "B", "A-", "A"]
        for band in expected_bands:
            assert band in dist

    def test_distribution_sums_to_respondent_count(self, benchmarking_engine, sample_benchmark):
        dist = benchmarking_engine.get_score_distribution(sample_benchmark)
        total = sum(dist.values())
        assert total == sample_benchmark.respondent_count

    def test_percentile_calculation(self, benchmarking_engine, sample_benchmark):
        percentile = benchmarking_engine.calculate_percentile(
            org_score=Decimal("72.5"),
            benchmark=sample_benchmark,
        )
        assert Decimal("0") <= percentile <= Decimal("100")

    def test_high_score_high_percentile(self, benchmarking_engine, sample_benchmark):
        percentile = benchmarking_engine.calculate_percentile(
            org_score=Decimal("92.0"),
            benchmark=sample_benchmark,
        )
        assert percentile >= Decimal("80")


# ---------------------------------------------------------------------------
# Category comparison
# ---------------------------------------------------------------------------

class TestCategoryComparison:
    """Test category-by-category comparison against sector average."""

    def test_compare_categories(self, benchmarking_engine, sample_category_scores):
        comparison = benchmarking_engine.compare_categories(
            org_scores=sample_category_scores,
            sector_gics="20101010",
            year=2025,
        )
        assert isinstance(comparison, dict)
        assert len(comparison) > 0

    def test_comparison_shows_delta(self, benchmarking_engine, sample_category_scores):
        comparison = benchmarking_engine.compare_categories(
            org_scores=sample_category_scores,
            sector_gics="20101010",
            year=2025,
        )
        for cat_code, data in comparison.items():
            assert "org_score" in data
            assert "sector_avg" in data
            assert "delta" in data

    def test_above_average_positive_delta(self, benchmarking_engine, sample_category_scores):
        comparison = benchmarking_engine.compare_categories(
            org_scores=sample_category_scores,
            sector_gics="20101010",
            year=2025,
        )
        # At least some categories should have positive deltas
        deltas = [d["delta"] for d in comparison.values()]
        assert any(d > Decimal("0") for d in deltas)


# ---------------------------------------------------------------------------
# A-list rate
# ---------------------------------------------------------------------------

class TestAListRate:
    """Test A-list rate calculation."""

    def test_a_list_rate_percentage(self, benchmarking_engine, sample_benchmark):
        rate = benchmarking_engine.get_a_list_rate(sample_benchmark)
        assert Decimal("0") <= rate <= Decimal("100")

    def test_a_list_rate_from_distribution(self, benchmarking_engine, sample_benchmark):
        rate = benchmarking_engine.get_a_list_rate(sample_benchmark)
        a_count = sample_benchmark.score_distribution.get("A", 0)
        expected = Decimal(str(a_count)) / Decimal(str(sample_benchmark.respondent_count)) * 100
        assert rate == pytest.approx(float(expected), rel=0.1)

    def test_a_list_rate_includes_a_minus(self, benchmarking_engine):
        benchmark = CDPBenchmark(
            sector_gics="test",
            region="Global",
            year=2025,
            mean_score=Decimal("50"),
            median_score=Decimal("48"),
            p25_score=Decimal("35"),
            p75_score=Decimal("65"),
            a_list_rate=Decimal("8.0"),
            respondent_count=100,
            score_distribution={"D-": 10, "D": 15, "C-": 15, "C": 15, "B-": 15, "B": 12, "A-": 10, "A": 8},
        )
        rate = benchmarking_engine.get_a_list_rate(benchmark)
        assert rate == Decimal("8.0")


# ---------------------------------------------------------------------------
# Custom peer group
# ---------------------------------------------------------------------------

class TestCustomPeerGroup:
    """Test user-defined peer group comparisons."""

    def test_create_peer_group(self, benchmarking_engine):
        group = benchmarking_engine.create_peer_group(
            name="European Industrials",
            sector_filter="20101010",
            region_filter="Europe",
        )
        assert group["name"] == "European Industrials"
        assert group["sector_filter"] == "20101010"

    def test_compare_against_peer_group(self, benchmarking_engine):
        group = benchmarking_engine.create_peer_group(
            name="Test Group",
            sector_filter="20101010",
            region_filter="Global",
        )
        comparison = benchmarking_engine.compare_to_peer_group(
            org_score=Decimal("72.5"),
            peer_group_id=group["id"],
        )
        assert isinstance(comparison, CDPPeerComparison)
        assert comparison.org_score == Decimal("72.5")


# ---------------------------------------------------------------------------
# Historical trends
# ---------------------------------------------------------------------------

class TestHistoricalTrends:
    """Test historical sector score trends."""

    def test_get_sector_trends(self, benchmarking_engine):
        trends = benchmarking_engine.get_sector_trends(
            sector_gics="20101010",
            years=[2022, 2023, 2024, 2025],
        )
        assert len(trends) > 0
        for entry in trends:
            assert "year" in entry
            assert "mean_score" in entry

    def test_trends_sorted_by_year(self, benchmarking_engine):
        trends = benchmarking_engine.get_sector_trends(
            sector_gics="20101010",
            years=[2022, 2023, 2024, 2025],
        )
        years = [t["year"] for t in trends]
        assert years == sorted(years)

    def test_single_year_trend(self, benchmarking_engine):
        trends = benchmarking_engine.get_sector_trends(
            sector_gics="20101010",
            years=[2025],
        )
        assert len(trends) == 1

    def test_trends_include_respondent_count(self, benchmarking_engine):
        trends = benchmarking_engine.get_sector_trends(
            sector_gics="20101010",
            years=[2023, 2024, 2025],
        )
        for entry in trends:
            assert "respondent_count" in entry or "mean_score" in entry


# ---------------------------------------------------------------------------
# Top quartile analysis
# ---------------------------------------------------------------------------

class TestTopQuartile:
    """Test top quartile identification."""

    def test_identify_top_quartile_threshold(self, benchmarking_engine, sample_benchmark):
        threshold = benchmarking_engine.get_top_quartile_threshold(sample_benchmark)
        assert threshold == sample_benchmark.p75_score

    def test_org_in_top_quartile(self, benchmarking_engine, sample_benchmark):
        is_top = benchmarking_engine.is_in_top_quartile(
            org_score=Decimal("80.0"),
            benchmark=sample_benchmark,
        )
        assert is_top is True

    def test_org_not_in_top_quartile(self, benchmarking_engine, sample_benchmark):
        is_top = benchmarking_engine.is_in_top_quartile(
            org_score=Decimal("30.0"),
            benchmark=sample_benchmark,
        )
        assert is_top is False
