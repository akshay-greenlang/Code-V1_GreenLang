# -*- coding: utf-8 -*-
"""
Test suite for PACK-028 Sector Pathway Pack - Sector Benchmark Engine.

Tests peer benchmarking, percentile ranking, gap-to-leader analysis,
SBTi/IEA benchmark comparisons, and composite scoring.

Author:  GreenLang Test Engineering
Pack:    PACK-028 Sector Pathway Pack
Engine:  7 of 8 - sector_benchmark_engine.py
"""

import sys
from decimal import Decimal
from pathlib import Path

import pytest

_PACK_ROOT = Path(__file__).resolve().parents[1]
if str(_PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACK_ROOT))

from engines.sector_benchmark_engine import (
    SectorBenchmarkEngine,
    BenchmarkInput,
    BenchmarkResult,
    PercentileRanking,
    GapToLeader,
    SBTiBenchmarkResult,
    IEAPathwayBenchmark,
    CompositeBenchmarkScore,
    PeerCompanyEntry,
    SBTiTargetStatus,
)

from .conftest import (
    assert_decimal_close,
    assert_decimal_positive,
    assert_provenance_hash,
    assert_processing_time,
    INTENSITY_METRICS,
    SDA_SECTORS,
    timed_block,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_input(sector="steel", current_intensity=Decimal("2.1"),
                entity_name="TestCo", base_year=2020, current_year=2024,
                base_year_intensity=None, annual_reduction=Decimal("3.0"),
                custom_peers=None, include_sbti=True, include_iea=True,
                include_regulatory=True, sbti_status=None):
    kw = dict(
        entity_name=entity_name,
        sector=sector,
        current_intensity=current_intensity,
        base_year=base_year,
        current_year=current_year,
        annual_reduction_rate_pct=annual_reduction,
        include_sbti_comparison=include_sbti,
        include_iea_comparison=include_iea,
        include_regulatory_comparison=include_regulatory,
    )
    if base_year_intensity is not None:
        kw["base_year_intensity"] = base_year_intensity
    if custom_peers is not None:
        kw["custom_peers"] = custom_peers
    if sbti_status is not None:
        kw["sbti_status"] = sbti_status
    return BenchmarkInput(**kw)


def _make_peer(name, intensity, region="EU"):
    return PeerCompanyEntry(
        company_name=name,
        intensity=intensity,
        region=region,
    )


# ===========================================================================
# Engine Instantiation
# ===========================================================================


class TestBenchmarkInstantiation:
    """Engine instantiation tests."""

    def test_engine_instantiates(self):
        engine = SectorBenchmarkEngine()
        assert engine is not None

    def test_engine_has_calculate(self):
        engine = SectorBenchmarkEngine()
        assert hasattr(engine, "calculate")

    def test_engine_version(self):
        engine = SectorBenchmarkEngine()
        assert engine.engine_version == "1.0.0"


# ===========================================================================
# Basic Benchmarking
# ===========================================================================


class TestBasicBenchmarking:
    """Test basic benchmark calculations."""

    def test_steel_benchmark(self):
        engine = SectorBenchmarkEngine()
        result = engine.calculate(_make_input(sector="steel", current_intensity=Decimal("2.1")))
        assert result.sector == "steel"
        assert result.current_intensity == Decimal("2.1")
        assert result.sector_average > Decimal("0")

    def test_power_benchmark(self):
        engine = SectorBenchmarkEngine()
        result = engine.calculate(_make_input(
            sector="power_generation", current_intensity=Decimal("450")))
        assert result.sector_average > Decimal("0")

    def test_cement_benchmark(self):
        engine = SectorBenchmarkEngine()
        result = engine.calculate(_make_input(
            sector="cement", current_intensity=Decimal("0.7")))
        assert result is not None


# ===========================================================================
# Percentile Ranking
# ===========================================================================


class TestPercentileRanking:
    """Test percentile ranking calculations."""

    def test_percentile_exists(self):
        engine = SectorBenchmarkEngine()
        result = engine.calculate(_make_input())
        assert result.percentile_ranking is not None

    def test_percentile_range(self):
        engine = SectorBenchmarkEngine()
        result = engine.calculate(_make_input())
        if result.percentile_ranking:
            assert Decimal("0") <= result.percentile_ranking.percentile <= Decimal("100")


# ===========================================================================
# Gap to Leader
# ===========================================================================


class TestGapToLeader:
    """Test gap-to-leader analysis."""

    def test_gap_to_leader_exists(self):
        engine = SectorBenchmarkEngine()
        result = engine.calculate(_make_input())
        assert result.gap_to_leader is not None

    def test_gap_when_above_leader(self):
        engine = SectorBenchmarkEngine()
        result = engine.calculate(_make_input(current_intensity=Decimal("3.0")))
        if result.gap_to_leader:
            assert result.gap_to_leader.gap_absolute > Decimal("0")


# ===========================================================================
# SBTi Benchmark
# ===========================================================================


class TestSBTiBenchmark:
    """Test SBTi benchmark comparison."""

    def test_sbti_benchmark_exists(self):
        engine = SectorBenchmarkEngine()
        result = engine.calculate(_make_input(include_sbti=True))
        assert result.sbti_benchmark is not None

    def test_sbti_benchmark_disabled(self):
        engine = SectorBenchmarkEngine()
        result = engine.calculate(_make_input(include_sbti=False))
        assert result.sbti_benchmark is None


# ===========================================================================
# IEA Benchmark
# ===========================================================================


class TestIEABenchmark:
    """Test IEA pathway benchmark comparison."""

    def test_iea_benchmark_exists(self):
        engine = SectorBenchmarkEngine()
        result = engine.calculate(_make_input(include_iea=True))
        assert result.iea_benchmark is not None

    def test_iea_benchmark_disabled(self):
        engine = SectorBenchmarkEngine()
        result = engine.calculate(_make_input(include_iea=False))
        assert result.iea_benchmark is None


# ===========================================================================
# Composite Score
# ===========================================================================


class TestCompositeScore:
    """Test composite benchmark scoring."""

    def test_composite_score_exists(self):
        engine = SectorBenchmarkEngine()
        result = engine.calculate(_make_input())
        assert result.composite_score is not None

    def test_composite_score_range(self):
        engine = SectorBenchmarkEngine()
        result = engine.calculate(_make_input())
        if result.composite_score:
            assert result.composite_score.overall_score >= Decimal("0")
            assert result.composite_score.overall_score <= Decimal("100")


# ===========================================================================
# Custom Peers
# ===========================================================================


class TestCustomPeers:
    """Test custom peer comparisons."""

    def test_with_custom_peers(self):
        engine = SectorBenchmarkEngine()
        peers = [
            _make_peer("CompA", Decimal("1.8")),
            _make_peer("CompB", Decimal("2.3")),
            _make_peer("CompC", Decimal("1.5")),
        ]
        result = engine.calculate(_make_input(custom_peers=peers))
        assert result.custom_peer_ranking is not None

    def test_without_custom_peers(self):
        engine = SectorBenchmarkEngine()
        result = engine.calculate(_make_input())
        # No custom peer ranking when no peers provided
        assert isinstance(result.custom_peer_ranking, (dict, type(None)))


# ===========================================================================
# Sector Parametrized Tests
# ===========================================================================


class TestSectorBenchmarks:
    """Test benchmark across sectors."""

    @pytest.mark.parametrize("sector", SDA_SECTORS)
    def test_sector_benchmark(self, sector):
        engine = SectorBenchmarkEngine()
        metrics = INTENSITY_METRICS.get(sector)
        if metrics is None:
            pytest.skip(f"No metrics for {sector}")
        result = engine.calculate(_make_input(
            sector=sector, current_intensity=metrics["base_2020"]))
        assert result.sector == sector
        assert result.sector_average >= Decimal("0")

    @pytest.mark.parametrize("sector", SDA_SECTORS[:6])
    def test_sector_has_iea_benchmark(self, sector):
        engine = SectorBenchmarkEngine()
        metrics = INTENSITY_METRICS.get(sector)
        if metrics is None:
            pytest.skip(f"No metrics for {sector}")
        result = engine.calculate(_make_input(
            sector=sector, current_intensity=metrics["base_2020"]))
        assert result.iea_benchmark is not None


# ===========================================================================
# Result Structure & Provenance
# ===========================================================================


class TestBenchmarkResultStructure:
    """Test result structure and provenance."""

    def test_result_provenance(self):
        engine = SectorBenchmarkEngine()
        result = engine.calculate(_make_input())
        assert_provenance_hash(result)

    def test_result_processing_time(self):
        engine = SectorBenchmarkEngine()
        result = engine.calculate(_make_input())
        assert_processing_time(result)

    def test_result_entity_name(self):
        engine = SectorBenchmarkEngine()
        result = engine.calculate(_make_input(entity_name="BenchCo"))
        assert result.entity_name == "BenchCo"

    def test_result_has_recommendations(self):
        engine = SectorBenchmarkEngine()
        result = engine.calculate(_make_input())
        assert isinstance(result.recommendations, list)

    def test_result_deterministic(self):
        engine = SectorBenchmarkEngine()
        inp = _make_input()
        r1 = engine.calculate(inp)
        r2 = engine.calculate(inp)
        assert r1.sector_average == r2.sector_average


# ===========================================================================
# Performance Tests
# ===========================================================================


class TestBenchmarkPerformance:
    """Performance tests."""

    def test_single_benchmark_under_100ms(self):
        engine = SectorBenchmarkEngine()
        with timed_block("single_benchmark", max_seconds=0.1):
            engine.calculate(_make_input())

    def test_100_benchmarks_under_2s(self):
        engine = SectorBenchmarkEngine()
        with timed_block("100_benchmarks", max_seconds=2.0):
            for _ in range(100):
                engine.calculate(_make_input())

    @pytest.mark.parametrize("sector", SDA_SECTORS)
    def test_each_sector_under_200ms(self, sector):
        engine = SectorBenchmarkEngine()
        metrics = INTENSITY_METRICS.get(sector)
        if metrics is None:
            pytest.skip(f"No metrics for {sector}")
        with timed_block(f"benchmark_{sector}", max_seconds=0.2):
            engine.calculate(_make_input(
                sector=sector, current_intensity=metrics["base_2020"]))
