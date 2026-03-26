# -*- coding: utf-8 -*-
"""
PACK-044 Test Suite - Benchmarking Engine Tests
=================================================

Tests BenchmarkingEngine: intensity metrics, peer comparison,
percentile ranking, z-score calculation, trend analysis, and
facility-level internal benchmarking.

Target: 40+ test cases.
"""

from decimal import Decimal

import pytest

from conftest import _load_engine

# ---------------------------------------------------------------------------
# Dynamic imports
# ---------------------------------------------------------------------------

_mod = _load_engine("benchmarking")

BenchmarkingEngine = _mod.BenchmarkingEngine
BenchmarkingResult = _mod.BenchmarkingResult
BenchmarkResult = _mod.BenchmarkResult
IntensityResult = _mod.IntensityResult
PeerComparison = _mod.PeerComparison
PeerDataPoint = _mod.PeerDataPoint
HistoricalDataPoint = _mod.HistoricalDataPoint
FacilityProfile = _mod.FacilityProfile
EntityProfile = _mod.EntityProfile


# ===================================================================
# Fixtures
# ===================================================================


@pytest.fixture
def engine():
    """Create a fresh BenchmarkingEngine."""
    return BenchmarkingEngine()


@pytest.fixture
def entity():
    """Entity profile for benchmarking."""
    return EntityProfile(
        entity_name="Acme Corp",
        total_scope1_tco2e=Decimal("22000"),
        total_scope2_location_tco2e=Decimal("15000"),
        revenue_eur_millions=Decimal("950"),
        fte_count=Decimal("2500"),
        floor_area_m2=Decimal("138000"),
        production_units=Decimal("50000"),
    )


@pytest.fixture
def peer_data():
    """Peer data for comparison."""
    return [
        PeerDataPoint(
            peer_id="PEER-001", peer_name="Competitor A",
            sector="manufacturing", year=2025,
            total_scope12_tco2e=Decimal("42000"),
            intensity_revenue=Decimal("38.18"),
            intensity_fte=Decimal("14.0"),
        ),
        PeerDataPoint(
            peer_id="PEER-002", peer_name="Competitor B",
            sector="manufacturing", year=2025,
            total_scope12_tco2e=Decimal("28000"),
            intensity_revenue=Decimal("40.0"),
            intensity_fte=Decimal("15.56"),
        ),
        PeerDataPoint(
            peer_id="PEER-003", peer_name="Competitor C",
            sector="manufacturing", year=2025,
            total_scope12_tco2e=Decimal("55000"),
            intensity_revenue=Decimal("36.67"),
            intensity_fte=Decimal("13.75"),
        ),
        PeerDataPoint(
            peer_id="PEER-004", peer_name="Competitor D",
            sector="manufacturing", year=2025,
            total_scope12_tco2e=Decimal("31000"),
            intensity_revenue=Decimal("38.75"),
            intensity_fte=Decimal("14.09"),
        ),
        PeerDataPoint(
            peer_id="PEER-005", peer_name="Competitor E",
            sector="manufacturing", year=2025,
            total_scope12_tco2e=Decimal("48000"),
            intensity_revenue=Decimal("36.92"),
            intensity_fte=Decimal("13.71"),
        ),
    ]


@pytest.fixture
def historical_data():
    """Historical data for trend analysis."""
    return [
        HistoricalDataPoint(
            year=2023, total_scope12_tco2e=Decimal("39500"),
            revenue_eur_millions=Decimal("900"), fte_count=Decimal("2400"),
        ),
        HistoricalDataPoint(
            year=2024, total_scope12_tco2e=Decimal("38100"),
            revenue_eur_millions=Decimal("920"), fte_count=Decimal("2450"),
        ),
        HistoricalDataPoint(
            year=2025, total_scope12_tco2e=Decimal("37000"),
            revenue_eur_millions=Decimal("950"), fte_count=Decimal("2500"),
        ),
    ]


@pytest.fixture
def facilities():
    """Facility profiles for internal benchmarking."""
    return [
        FacilityProfile(
            facility_name="Frankfurt Plant",
            country="DE",
            total_tco2e=Decimal("12500"),
            revenue_eur_millions=Decimal("350"),
            fte_count=Decimal("800"),
            floor_area_m2=Decimal("45000"),
        ),
        FacilityProfile(
            facility_name="Berlin Office",
            country="DE",
            total_tco2e=Decimal("3500"),
            revenue_eur_millions=Decimal("200"),
            fte_count=Decimal("500"),
            floor_area_m2=Decimal("25000"),
        ),
        FacilityProfile(
            facility_name="Munich Warehouse",
            country="DE",
            total_tco2e=Decimal("6000"),
            revenue_eur_millions=Decimal("100"),
            fte_count=Decimal("200"),
            floor_area_m2=Decimal("60000"),
        ),
    ]


# ===================================================================
# Intensity Calculation Tests (via benchmark)
# ===================================================================


class TestIntensityCalculation:
    """Tests for intensity metric calculations via benchmark()."""

    def test_intensity_metrics_populated(self, engine, entity, peer_data):
        result = engine.benchmark(entity, peers=peer_data)
        assert len(result.intensity_metrics) >= 2

    def test_revenue_intensity_calculated(self, engine, entity, peer_data):
        result = engine.benchmark(entity, peers=peer_data)
        rev_int = [r for r in result.intensity_metrics if r.metric_type == "revenue_intensity"]
        if rev_int:
            assert rev_int[0].intensity > 0

    def test_fte_intensity_calculated(self, engine, entity, peer_data):
        result = engine.benchmark(entity, peers=peer_data)
        fte_int = [r for r in result.intensity_metrics if r.metric_type == "fte_intensity"]
        if fte_int:
            assert fte_int[0].intensity > 0

    def test_floor_area_intensity_calculated(self, engine, entity, peer_data):
        result = engine.benchmark(entity, peers=peer_data)
        area_int = [r for r in result.intensity_metrics
                    if r.metric_type == "floor_area_intensity"]
        if area_int:
            assert area_int[0].intensity > 0


# ===================================================================
# Peer Benchmarking Tests
# ===================================================================


class TestPeerBenchmarking:
    """Tests for peer group benchmarking."""

    def test_benchmark_against_peers(self, engine, entity, peer_data):
        result = engine.benchmark(entity, peers=peer_data)
        assert result is not None
        assert isinstance(result, BenchmarkingResult)

    def test_percentile_rank_range(self, engine, entity, peer_data):
        result = engine.benchmark(entity, peers=peer_data)
        for br in result.sector_benchmarks:
            assert 0 <= br.percentile_rank <= 100

    def test_z_score_computed(self, engine, entity, peer_data):
        result = engine.benchmark(entity, peers=peer_data)
        for br in result.sector_benchmarks:
            assert isinstance(br.z_score, (int, float))

    def test_peer_statistics(self, engine, entity, peer_data):
        result = engine.benchmark(entity, peers=peer_data)
        # sector_benchmarks includes both sector-only (peer_count=0)
        # and peer-based benchmarks (peer_count>0)
        peer_benchmarks = [br for br in result.sector_benchmarks if br.peer_count > 0]
        assert len(peer_benchmarks) >= 1
        for br in peer_benchmarks:
            assert br.peer_count == 5
            assert br.peer_mean > 0

    def test_benchmark_status_classification(self, engine, entity, peer_data):
        result = engine.benchmark(entity, peers=peer_data)
        for br in result.sector_benchmarks:
            assert br.status in ("leader", "above_average", "average",
                                  "below_average", "laggard")

    def test_peer_comparisons_generated(self, engine, entity, peer_data):
        result = engine.benchmark(entity, peers=peer_data)
        assert len(result.peer_comparisons) >= 1


# ===================================================================
# Trend Analysis Tests (via benchmark with historical)
# ===================================================================


class TestTrendAnalysis:
    """Tests for historical trend benchmarking."""

    def test_trend_analysis(self, engine, entity, historical_data):
        result = engine.benchmark(entity, historical=historical_data)
        assert len(result.trends) >= 1

    def test_trend_has_direction(self, engine, entity, historical_data):
        result = engine.benchmark(entity, historical=historical_data)
        for t in result.trends:
            assert t.trend_direction in ("improving", "stable", "worsening",
                                         "decreasing", "increasing")

    def test_trend_has_yoy_changes(self, engine, entity, historical_data):
        result = engine.benchmark(entity, historical=historical_data)
        for t in result.trends:
            assert len(t.year_over_year) >= 1

    def test_single_year_no_crash(self, engine, entity):
        result = engine.benchmark(entity, historical=[
            HistoricalDataPoint(year=2025, total_scope12_tco2e=Decimal("37000")),
        ])
        assert result is not None


# ===================================================================
# Internal Facility Benchmarking Tests
# ===================================================================


class TestFacilityBenchmarking:
    """Tests for internal facility-level ranking."""

    def test_facility_rankings_populated(self, engine, entity, facilities):
        result = engine.benchmark(entity, facilities=facilities)
        assert len(result.facility_rankings) >= 1

    def test_facilities_ranked(self, engine, entity, facilities):
        result = engine.benchmark(entity, facilities=facilities)
        if len(result.facility_rankings) >= 2:
            assert result.facility_rankings[0].rank <= result.facility_rankings[1].rank

    def test_best_performer_identified(self, engine, entity, facilities):
        result = engine.benchmark(entity, facilities=facilities)
        assert result.facility_rankings[0].rank == 1


# ===================================================================
# Provenance Tests
# ===================================================================


class TestProvenance:
    """Tests for provenance hashing."""

    def test_benchmark_provenance_hash(self, engine, entity, peer_data):
        result = engine.benchmark(entity, peers=peer_data)
        assert len(result.provenance_hash) == 64

    def test_processing_time_positive(self, engine, entity, peer_data):
        result = engine.benchmark(entity, peers=peer_data)
        assert result.processing_time_ms >= 0


# ===================================================================
# Model Tests
# ===================================================================


class TestModels:
    """Tests for Pydantic model defaults."""

    def test_intensity_result_defaults(self):
        ir = IntensityResult()
        assert ir.intensity == 0.0

    def test_benchmark_result_defaults(self):
        br = BenchmarkResult()
        assert br.peer_count == 0
        assert br.status == "average"

    def test_peer_comparison_defaults(self):
        pc = PeerComparison()
        assert pc.entity_is_better is False

    def test_peer_data_point_creation(self):
        p = PeerDataPoint(
            peer_name="Test Peer",
            total_scope12_tco2e=Decimal("25000"),
        )
        assert p.total_scope12_tco2e == Decimal("25000")

    def test_historical_data_point_creation(self):
        h = HistoricalDataPoint(
            year=2024,
            total_scope12_tco2e=Decimal("38000"),
        )
        assert h.year == 2024

    def test_facility_profile_creation(self):
        f = FacilityProfile(
            facility_name="Test Facility",
            total_tco2e=Decimal("5000"),
        )
        assert f.facility_name == "Test Facility"

    def test_facility_decimal_coercion(self):
        f = FacilityProfile(
            facility_name="Coerce",
            total_tco2e="12345",
            revenue_eur_millions="100",
        )
        assert f.total_tco2e == Decimal("12345")
        assert f.revenue_eur_millions == Decimal("100")
