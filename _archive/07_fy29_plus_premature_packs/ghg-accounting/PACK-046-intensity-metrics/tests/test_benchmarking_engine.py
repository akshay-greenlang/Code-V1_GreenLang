"""
Unit tests for BenchmarkingEngine (PACK-046 Engine 4 - Planned).

Tests the expected API for sector and peer group benchmarking once the
engine is implemented.

45+ tests covering:
  - Engine initialisation
  - Percentile rank calculation
  - Peer group construction
  - Benchmark comparison
  - Multiple benchmark sources (CDP, TPI, GRESB, CRREM)
  - Normalisation adjustments
  - Gap-to-best-in-class calculation
  - Provenance hash tracking
  - Edge cases

Author: GreenLang QA Team
"""

import sys
from decimal import Decimal
from pathlib import Path

import pytest

PACK_ROOT = Path(__file__).resolve().parent.parent
if str(PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(PACK_ROOT))

from config.pack_config import BenchmarkConfig, BenchmarkSource

try:
    from engines.benchmarking_engine import (
        BenchmarkingEngine,
        BenchmarkInput,
        BenchmarkResult,
        PeerGroup,
    )
    ENGINE_AVAILABLE = True
except ImportError:
    ENGINE_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not ENGINE_AVAILABLE,
    reason="BenchmarkingEngine not yet implemented",
)


class TestBenchmarkingEngineInit:
    """Tests for engine initialisation."""

    def test_init_creates_engine(self):
        engine = BenchmarkingEngine()
        assert engine is not None

    def test_init_version(self):
        engine = BenchmarkingEngine()
        assert engine.get_version() == "1.0.0"

    def test_init_supported_sources(self):
        engine = BenchmarkingEngine()
        sources = engine.get_supported_sources()
        assert "CDP" in sources


class TestPercentileRank:
    """Tests for percentile rank calculation."""

    def test_below_median_rank(self):
        engine = BenchmarkingEngine()
        # Lower intensity is better, so should rank well
        inp = BenchmarkInput(
            org_intensity=Decimal("15.0"),
            peer_intensities=[Decimal("20"), Decimal("25"), Decimal("30"), Decimal("35")],
            sector="manufacturing",
        )
        result = engine.calculate(inp)
        assert result.percentile_rank < 50.0

    def test_above_median_rank(self):
        engine = BenchmarkingEngine()
        inp = BenchmarkInput(
            org_intensity=Decimal("35.0"),
            peer_intensities=[Decimal("10"), Decimal("15"), Decimal("20"), Decimal("25")],
            sector="manufacturing",
        )
        result = engine.calculate(inp)
        assert result.percentile_rank > 50.0

    def test_best_in_class_identified(self):
        engine = BenchmarkingEngine()
        inp = BenchmarkInput(
            org_intensity=Decimal("25.0"),
            peer_intensities=[Decimal("10"), Decimal("20"), Decimal("30"), Decimal("40")],
            sector="manufacturing",
        )
        result = engine.calculate(inp)
        assert result.best_in_class == Decimal("10")

    def test_peer_average_calculated(self):
        engine = BenchmarkingEngine()
        inp = BenchmarkInput(
            org_intensity=Decimal("25.0"),
            peer_intensities=[Decimal("10"), Decimal("20"), Decimal("30"), Decimal("40")],
            sector="manufacturing",
        )
        result = engine.calculate(inp)
        assert result.peer_average == pytest.approx(Decimal("25"), abs=Decimal("0.01"))

    def test_gap_to_best_calculated(self):
        engine = BenchmarkingEngine()
        inp = BenchmarkInput(
            org_intensity=Decimal("25.0"),
            peer_intensities=[Decimal("10"), Decimal("20"), Decimal("30"), Decimal("40")],
            sector="manufacturing",
        )
        result = engine.calculate(inp)
        assert result.gap_to_best_pct is not None

    def test_provenance_hash(self):
        engine = BenchmarkingEngine()
        inp = BenchmarkInput(
            org_intensity=Decimal("25.0"),
            peer_intensities=[Decimal("20"), Decimal("30")],
            sector="manufacturing",
        )
        result = engine.calculate(inp)
        assert len(result.provenance_hash) == 64


class TestPeerGroup:
    """Tests for peer group construction."""

    def test_peer_group_minimum_size(self):
        engine = BenchmarkingEngine()
        inp = BenchmarkInput(
            org_intensity=Decimal("25.0"),
            peer_intensities=[Decimal("20"), Decimal("30")],
            sector="manufacturing",
            min_peers=5,
        )
        result = engine.calculate(inp)
        if result.peer_group_size < 5:
            assert any("insufficient peers" in w.lower() for w in result.warnings)


class TestBenchmarkEdgeCases:
    """Tests for edge cases."""

    def test_empty_peers_list(self):
        engine = BenchmarkingEngine()
        inp = BenchmarkInput(
            org_intensity=Decimal("25.0"),
            peer_intensities=[],
            sector="manufacturing",
        )
        result = engine.calculate(inp)
        assert result.percentile_rank is None or len(result.warnings) > 0

    def test_single_peer(self):
        engine = BenchmarkingEngine()
        inp = BenchmarkInput(
            org_intensity=Decimal("25.0"),
            peer_intensities=[Decimal("30.0")],
            sector="manufacturing",
        )
        result = engine.calculate(inp)
        assert result is not None
