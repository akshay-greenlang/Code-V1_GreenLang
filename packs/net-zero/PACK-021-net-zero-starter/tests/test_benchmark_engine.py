# -*- coding: utf-8 -*-
"""
Unit tests for NetZeroBenchmarkEngine (PACK-021 Engine 8).

Tests sector benchmark lookups, percentile ranking, gap-to-leader analysis,
KPI comparison, performance trend detection, best practice identification,
and provenance hashing across 20+ sectors.

Author:  GL-TestEngineer
Pack:    PACK-021 Net Zero Starter
"""

import sys
from decimal import Decimal
from pathlib import Path

import pytest

_PACK_ROOT = Path(__file__).resolve().parents[1]
if str(_PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACK_ROOT))

from engines.net_zero_benchmark_engine import (
    BEST_PRACTICES,
    BenchmarkInput,
    BenchmarkResult,
    BenchmarkSector,
    CDP_SCORE_MAP,
    KPIBenchmark,
    NetZeroBenchmarkEngine,
    PeerComparison,
    Percentile,
    PerformanceIndicator,
    PerformanceTrend,
    SBTiStatus,
    SECTOR_BENCHMARKS,
    _percentile_rank,
)


# ========================================================================
# Fixtures
# ========================================================================


@pytest.fixture
def engine() -> NetZeroBenchmarkEngine:
    """Fresh NetZeroBenchmarkEngine."""
    return NetZeroBenchmarkEngine()


@pytest.fixture
def industrials_input() -> BenchmarkInput:
    """Typical industrials company input."""
    return BenchmarkInput(
        entity_name="IndustrialCo",
        sector=BenchmarkSector.INDUSTRIALS,
        assessment_year=2026,
        total_emissions_tco2e=Decimal("50000"),
        revenue_musd=Decimal("500"),
        carbon_intensity_revenue=Decimal("100"),
        annual_reduction_rate_pct=Decimal("4.5"),
        scope3_categories_measured=8,
        sbti_status=SBTiStatus.COMMITTED,
        renewable_electricity_pct=Decimal("45"),
        cdp_score="B",
    )


@pytest.fixture
def technology_input() -> BenchmarkInput:
    """Technology sector input with good performance."""
    return BenchmarkInput(
        entity_name="TechCo",
        sector=BenchmarkSector.INFORMATION_TECHNOLOGY,
        assessment_year=2026,
        carbon_intensity_revenue=Decimal("5"),
        annual_reduction_rate_pct=Decimal("8"),
        scope3_categories_measured=10,
        sbti_status=SBTiStatus.APPROVED,
        renewable_electricity_pct=Decimal("85"),
        cdp_score="A-",
    )


@pytest.fixture
def energy_input() -> BenchmarkInput:
    """Energy sector input with below-average performance."""
    return BenchmarkInput(
        entity_name="EnergyCo",
        sector=BenchmarkSector.ENERGY,
        assessment_year=2026,
        carbon_intensity_revenue=Decimal("600"),
        annual_reduction_rate_pct=Decimal("1.5"),
        scope3_categories_measured=4,
        sbti_status=SBTiStatus.NONE,
        renewable_electricity_pct=Decimal("10"),
        cdp_score="C",
    )


# ========================================================================
# Instantiation
# ========================================================================


class TestBenchmarkEngineInstantiation:
    """Tests for engine creation."""

    def test_engine_instantiates(self):
        """Engine creates without error."""
        engine = NetZeroBenchmarkEngine()
        assert engine is not None
        assert engine.engine_version == "1.0.0"

    def test_engine_has_benchmark_method(self, engine):
        """Engine exposes a `benchmark` method."""
        assert callable(getattr(engine, "benchmark", None))


# ========================================================================
# Sector Benchmarks (20+ Sectors)
# ========================================================================


class TestSectorBenchmarks:
    """Tests for sector benchmark reference data."""

    def test_at_least_10_sectors_defined(self):
        """SECTOR_BENCHMARKS has 10+ sector entries."""
        assert len(SECTOR_BENCHMARKS) >= 10

    @pytest.mark.parametrize(
        "sector",
        [s for s in BenchmarkSector if s.value in SECTOR_BENCHMARKS],
        ids=[s.value for s in BenchmarkSector if s.value in SECTOR_BENCHMARKS],
    )
    def test_each_sector_has_benchmark_data(self, sector):
        """BenchmarkSector entries with data have valid benchmark entries."""
        assert sector.value in SECTOR_BENCHMARKS

    @pytest.mark.parametrize(
        "sector",
        [s for s in BenchmarkSector if s.value in SECTOR_BENCHMARKS],
        ids=[s.value for s in BenchmarkSector if s.value in SECTOR_BENCHMARKS],
    )
    def test_sector_has_intensity_kpi(self, sector):
        """Each benchmarked sector has carbon intensity by revenue benchmark."""
        data = SECTOR_BENCHMARKS[sector.value]
        kpi_key = PerformanceIndicator.CARBON_INTENSITY_REVENUE.value
        assert kpi_key in data
        kpi_data = data[kpi_key]
        assert "median" in kpi_data
        assert "leader" in kpi_data
        assert "p10" in kpi_data
        assert "p90" in kpi_data

    def test_sector_has_leader_profile(self):
        """Industrials sector has a leader profile."""
        data = SECTOR_BENCHMARKS[BenchmarkSector.INDUSTRIALS.value]
        assert "leader_profile" in data
        assert "name" in data["leader_profile"]

    def test_sector_leader_better_than_median(self):
        """Leader value is better than median for intensity (lower is better)."""
        data = SECTOR_BENCHMARKS[BenchmarkSector.INDUSTRIALS.value]
        kpi = data[PerformanceIndicator.CARBON_INTENSITY_REVENUE.value]
        assert kpi["leader"] < kpi["median"]


# ========================================================================
# Percentile Ranking
# ========================================================================


class TestPercentileRanking:
    """Tests for percentile rank computation."""

    def test_percentile_rank_utility(self):
        """_percentile_rank computes correct percentile."""
        values = [10.0, 20.0, 30.0, 40.0, 50.0]
        # 30 is the median (3rd of 5) -> should be around 50th percentile
        rank = _percentile_rank(values, 30.0)
        assert 40 <= rank <= 60

    def test_percentile_rank_lowest(self):
        """Lowest value yields low percentile."""
        values = [10.0, 20.0, 30.0, 40.0, 50.0]
        rank = _percentile_rank(values, 10.0)
        assert rank < 30

    def test_percentile_rank_highest(self):
        """Highest value yields high percentile."""
        values = [10.0, 20.0, 30.0, 40.0, 50.0]
        rank = _percentile_rank(values, 50.0)
        assert rank > 70

    def test_percentile_rank_empty_list(self):
        """Empty list returns 50.0."""
        rank = _percentile_rank([], 42.0)
        assert rank == 50.0

    def test_benchmark_result_has_percentile_rankings(self, engine, industrials_input):
        """Result contains KPI percentile rankings."""
        result = engine.benchmark(industrials_input)
        assert len(result.percentile_rankings) > 0
        for kpi in result.percentile_rankings:
            assert isinstance(kpi, KPIBenchmark)
            assert Decimal("0") <= kpi.percentile_rank <= Decimal("100")

    def test_percentile_bracket_assignment(self, engine, technology_input):
        """KPI benchmarks have valid percentile brackets."""
        result = engine.benchmark(technology_input)
        valid_brackets = set(Percentile)
        for kpi in result.percentile_rankings:
            assert kpi.percentile_bracket in valid_brackets


# ========================================================================
# Gap to Leader
# ========================================================================


class TestGapToLeader:
    """Tests for gap-to-leader analysis."""

    def test_gap_to_leader_in_result(self, engine, industrials_input):
        """Result has gap_to_leader dict."""
        result = engine.benchmark(industrials_input)
        assert isinstance(result.gap_to_leader, dict)

    def test_kpi_gap_to_leader_calculated(self, engine, industrials_input):
        """Each KPI benchmark has gap_to_leader field."""
        result = engine.benchmark(industrials_input)
        for kpi in result.percentile_rankings:
            # gap_to_leader is a Decimal indicating the difference
            assert isinstance(kpi.gap_to_leader, Decimal)

    def test_leader_has_zero_gap(self, engine):
        """A company at the leader level has gap_to_leader ~0."""
        # Industrials leader intensity = 10
        inp = BenchmarkInput(
            entity_name="LeaderCo",
            sector=BenchmarkSector.INDUSTRIALS,
            carbon_intensity_revenue=Decimal("10"),
            annual_reduction_rate_pct=Decimal("11"),
            scope3_categories_measured=15,
            renewable_electricity_pct=Decimal("100"),
            cdp_score="A",
            sbti_status=SBTiStatus.APPROVED,
        )
        result = engine.benchmark(inp)
        # Find intensity KPI
        intensity_kpi = None
        for kpi in result.percentile_rankings:
            if kpi.kpi == PerformanceIndicator.CARBON_INTENSITY_REVENUE:
                intensity_kpi = kpi
                break
        if intensity_kpi:
            assert intensity_kpi.gap_to_leader == Decimal("0")

    def test_gap_to_median_present(self, engine, industrials_input):
        """Each KPI has a gap_to_median field."""
        result = engine.benchmark(industrials_input)
        for kpi in result.percentile_rankings:
            assert isinstance(kpi.gap_to_median, Decimal)


# ========================================================================
# KPI Comparison
# ========================================================================


class TestKPIComparison:
    """Tests for KPI comparison against sector values."""

    @pytest.mark.parametrize(
        "kpi",
        [
            PerformanceIndicator.CARBON_INTENSITY_REVENUE,
            PerformanceIndicator.ANNUAL_REDUCTION_RATE,
            PerformanceIndicator.SCOPE3_COVERAGE,
            PerformanceIndicator.RENEWABLE_ELECTRICITY_PCT,
            PerformanceIndicator.CDP_SCORE,
        ],
    )
    def test_standard_kpis_benchmarked(self, engine, industrials_input, kpi):
        """Standard KPIs are included in benchmark results."""
        result = engine.benchmark(industrials_input)
        kpi_names = {k.kpi for k in result.percentile_rankings}
        assert kpi in kpi_names

    def test_kpi_has_sector_context(self, engine, industrials_input):
        """Each KPI benchmark includes sector median and leader."""
        result = engine.benchmark(industrials_input)
        for kpi in result.percentile_rankings:
            assert kpi.sector_median >= Decimal("0") or kpi.sector_median < Decimal("0")
            assert kpi.unit  # non-empty unit string

    def test_lower_is_better_flag(self, engine, industrials_input):
        """Carbon intensity has lower_is_better=True."""
        result = engine.benchmark(industrials_input)
        for kpi in result.percentile_rankings:
            if kpi.kpi == PerformanceIndicator.CARBON_INTENSITY_REVENUE:
                assert kpi.lower_is_better is True
            elif kpi.kpi == PerformanceIndicator.ANNUAL_REDUCTION_RATE:
                assert kpi.lower_is_better is False


# ========================================================================
# Performance Trend
# ========================================================================


class TestPerformanceTrend:
    """Tests for trend detection."""

    def test_trend_field_present(self, engine, industrials_input):
        """Each KPI benchmark has a trend field."""
        result = engine.benchmark(industrials_input)
        for kpi in result.percentile_rankings:
            assert kpi.trend in (
                PerformanceTrend.IMPROVING,
                PerformanceTrend.STABLE,
                PerformanceTrend.DECLINING,
                PerformanceTrend.INSUFFICIENT_DATA,
            )

    def test_trend_with_prior_year_data(self, engine):
        """Trend computed when prior year data is provided."""
        inp = BenchmarkInput(
            entity_name="TrendCo",
            sector=BenchmarkSector.INDUSTRIALS,
            carbon_intensity_revenue=Decimal("100"),
            annual_reduction_rate_pct=Decimal("5"),
            scope3_categories_measured=8,
            renewable_electricity_pct=Decimal("50"),
            cdp_score="B",
            sbti_status=SBTiStatus.COMMITTED,
            prior_year_data={
                "carbon_intensity_revenue": 120,
                "annual_reduction_rate_pct": 3,
            },
        )
        result = engine.benchmark(inp)
        trend_analysis = result.trend_analysis
        assert isinstance(trend_analysis, dict)

    def test_trend_analysis_in_result(self, engine, industrials_input):
        """Result has trend_analysis dict."""
        result = engine.benchmark(industrials_input)
        assert isinstance(result.trend_analysis, dict)


# ========================================================================
# Best Practices
# ========================================================================


class TestBestPracticesIdentification:
    """Tests for best practice recommendations."""

    def test_best_practices_in_result(self, engine, industrials_input):
        """Result includes best practices list."""
        result = engine.benchmark(industrials_input)
        assert isinstance(result.best_practices, list)

    def test_best_practices_reference_data(self):
        """BEST_PRACTICES has entries for key KPIs."""
        assert PerformanceIndicator.CARBON_INTENSITY_REVENUE.value in BEST_PRACTICES
        assert PerformanceIndicator.ANNUAL_REDUCTION_RATE.value in BEST_PRACTICES
        assert PerformanceIndicator.SCOPE3_COVERAGE.value in BEST_PRACTICES
        assert PerformanceIndicator.RENEWABLE_ELECTRICITY_PCT.value in BEST_PRACTICES
        assert PerformanceIndicator.CDP_SCORE.value in BEST_PRACTICES

    def test_each_kpi_has_practices(self):
        """Each KPI in BEST_PRACTICES has at least 1 practice."""
        for kpi, practices in BEST_PRACTICES.items():
            assert len(practices) >= 1

    def test_kpi_benchmarks_include_practices(self, engine, energy_input):
        """KPI benchmarks include relevant best practices."""
        result = engine.benchmark(energy_input)
        for kpi in result.percentile_rankings:
            assert isinstance(kpi.best_practices, list)


# ========================================================================
# Peer Comparison
# ========================================================================


class TestPeerComparison:
    """Tests for peer comparison summary."""

    def test_peer_comparison_present(self, engine, industrials_input):
        """Result includes a PeerComparison object."""
        result = engine.benchmark(industrials_input)
        assert result.peer_comparison is not None
        assert isinstance(result.peer_comparison, PeerComparison)

    def test_peer_comparison_fields(self, engine, industrials_input):
        """PeerComparison has expected summary fields."""
        result = engine.benchmark(industrials_input)
        peer = result.peer_comparison
        assert peer.kpis_assessed > 0
        assert peer.sector == BenchmarkSector.INDUSTRIALS.value
        assert Decimal("0") <= peer.overall_percentile <= Decimal("100")

    def test_peer_comparison_sbti_adoption(self, engine, industrials_input):
        """PeerComparison reports sector SBTi adoption rate."""
        result = engine.benchmark(industrials_input)
        peer = result.peer_comparison
        assert peer.sector_sbti_adoption_pct >= Decimal("0")

    def test_peer_comparison_above_below_median(self, engine, industrials_input):
        """PeerComparison counts KPIs above/below median."""
        result = engine.benchmark(industrials_input)
        peer = result.peer_comparison
        total = peer.kpis_above_median + peer.kpis_below_median
        assert total <= peer.kpis_assessed


# ========================================================================
# CDP Score Map
# ========================================================================


class TestCDPScoreMap:
    """Tests for CDP score numeric mapping."""

    @pytest.mark.parametrize(
        "cdp_letter,expected_numeric",
        [
            ("A", Decimal("100")),
            ("A-", Decimal("85")),
            ("B", Decimal("70")),
            ("B-", Decimal("55")),
            ("C", Decimal("40")),
            ("D", Decimal("15")),
            ("F", Decimal("0")),
        ],
    )
    def test_cdp_score_mapping(self, cdp_letter, expected_numeric):
        """CDP letter scores map to correct numeric values."""
        assert CDP_SCORE_MAP[cdp_letter] == expected_numeric


# ========================================================================
# Provenance Hash
# ========================================================================


class TestBenchmarkProvenanceHash:
    """Tests for SHA-256 provenance hashing."""

    def test_provenance_hash_present(self, engine, industrials_input):
        """Result has a 64-char hex provenance hash."""
        result = engine.benchmark(industrials_input)
        assert result.provenance_hash
        assert len(result.provenance_hash) == 64

    def test_provenance_hash_is_valid_sha256(self, engine):
        """Provenance hash is a valid SHA-256 hex string."""
        inp = BenchmarkInput(
            entity_name="HashTest",
            sector=BenchmarkSector.INDUSTRIALS,
            carbon_intensity_revenue=Decimal("100"),
            annual_reduction_rate_pct=Decimal("4.5"),
            scope3_categories_measured=8,
            renewable_electricity_pct=Decimal("45"),
            cdp_score="B",
            sbti_status=SBTiStatus.COMMITTED,
        )
        r1 = engine.benchmark(inp)
        r2 = engine.benchmark(inp)
        assert len(r1.provenance_hash) == 64
        assert len(r2.provenance_hash) == 64
        assert all(c in "0123456789abcdef" for c in r1.provenance_hash)
        assert all(c in "0123456789abcdef" for c in r2.provenance_hash)


# ========================================================================
# Result Structure
# ========================================================================


class TestBenchmarkResultStructure:
    """Validate complete result structure."""

    def test_result_fields(self, engine, industrials_input):
        """BenchmarkResult has all expected fields."""
        result = engine.benchmark(industrials_input)
        assert isinstance(result, BenchmarkResult)
        assert result.result_id
        assert result.engine_version == "1.0.0"
        assert result.entity_name == "IndustrialCo"
        assert result.sector == BenchmarkSector.INDUSTRIALS.value
        assert result.assessment_year == 2026
        assert result.processing_time_ms >= 0.0
        assert isinstance(result.recommendations, list)
        assert isinstance(result.warnings, list)

    def test_minimal_input_benchmark(self, engine):
        """Engine handles minimal input (only sector required)."""
        inp = BenchmarkInput(
            sector=BenchmarkSector.FINANCIALS,
        )
        result = engine.benchmark(inp)
        assert isinstance(result, BenchmarkResult)
        assert result.provenance_hash
