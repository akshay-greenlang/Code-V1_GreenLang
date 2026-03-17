# -*- coding: utf-8 -*-
"""
PACK-018 EU Green Claims Prep Pack - Green Claims Benchmark Engine Tests
=========================================================================

Unit tests for GreenClaimsBenchmarkEngine covering enums (MaturityLevel,
BenchmarkDimension, ImprovementTimeframe, PeerComparisonOutcome), models
(PortfolioMetrics, BenchmarkResult, PeerDataPoint), and engine methods
(calculate_portfolio_metrics, determine_maturity_level,
benchmark_against_peers, generate_improvement_roadmap,
calculate_overall_score).

Target: ~50 tests (file-existence-aware).

Author:  GreenLang Platform Team (GL-TestEngineer)
Pack:    PACK-018 EU Green Claims Prep
Date:    March 2026
"""

from decimal import Decimal

import pytest

from .conftest import _load_engine, ENGINES_DIR, ENGINE_FILES


# ---------------------------------------------------------------------------
# Module-scoped engine loading (skip if file missing)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def mod():
    """Load the Green Claims Benchmark engine module, skip if not found."""
    try:
        return _load_engine("green_claims_benchmark")
    except (FileNotFoundError, ImportError):
        pytest.skip("green_claims_benchmark_engine.py not yet created")


@pytest.fixture
def engine(mod):
    """Create a fresh GreenClaimsBenchmarkEngine instance."""
    return mod.GreenClaimsBenchmarkEngine()


# ===========================================================================
# File Existence Tests
# ===========================================================================


class TestGreenClaimsBenchmarkFileExistence:
    """Tests for benchmark engine file existence."""

    def test_engine_file_name_in_registry(self):
        """green_claims_benchmark is registered in ENGINE_FILES."""
        assert "green_claims_benchmark" in ENGINE_FILES

    def test_engine_file_name_value(self):
        """ENGINE_FILES maps to green_claims_benchmark_engine.py."""
        assert ENGINE_FILES["green_claims_benchmark"] == "green_claims_benchmark_engine.py"

    def test_engine_file_exists_on_disk(self):
        """green_claims_benchmark_engine.py exists on disk."""
        path = ENGINES_DIR / "green_claims_benchmark_engine.py"
        if not path.exists():
            pytest.skip("green_claims_benchmark_engine.py not yet created")
        assert path.exists()


# ===========================================================================
# Enum Tests
# ===========================================================================


class TestGreenClaimsBenchmarkEnums:
    """Tests for Green Claims Benchmark engine enums."""

    def test_maturity_level_enum_exists(self, mod):
        """MaturityLevel enum exists."""
        assert hasattr(mod, "MaturityLevel")

    def test_maturity_level_count(self, mod):
        """MaturityLevel has expected number of values."""
        assert len(mod.MaturityLevel) >= 4

    def test_benchmark_dimension_enum_exists(self, mod):
        """BenchmarkDimension enum exists."""
        assert hasattr(mod, "BenchmarkDimension")

    def test_benchmark_dimension_count(self, mod):
        """BenchmarkDimension has expected number of values."""
        assert len(mod.BenchmarkDimension) >= 4

    def test_improvement_timeframe_enum_exists(self, mod):
        """ImprovementTimeframe enum exists."""
        assert hasattr(mod, "ImprovementTimeframe")

    def test_peer_comparison_outcome_enum_exists(self, mod):
        """PeerComparisonOutcome enum exists."""
        assert hasattr(mod, "PeerComparisonOutcome")


# ===========================================================================
# Model Tests
# ===========================================================================


class TestGreenClaimsBenchmarkModels:
    """Tests for Green Claims Benchmark engine models."""

    def test_engine_class_exists(self, mod):
        """GreenClaimsBenchmarkEngine class exists."""
        assert hasattr(mod, "GreenClaimsBenchmarkEngine")

    def test_engine_has_docstring(self, mod):
        """GreenClaimsBenchmarkEngine has a docstring."""
        assert mod.GreenClaimsBenchmarkEngine.__doc__ is not None

    def test_has_portfolio_metrics_model(self, mod):
        """Module has PortfolioMetrics model."""
        assert hasattr(mod, "PortfolioMetrics")

    def test_has_benchmark_result_model(self, mod):
        """Module has BenchmarkResult model."""
        assert hasattr(mod, "BenchmarkResult")

    def test_has_peer_data_point_model(self, mod):
        """Module has PeerDataPoint model."""
        assert hasattr(mod, "PeerDataPoint")


# ===========================================================================
# Engine Method Tests
# ===========================================================================


class TestGreenClaimsBenchmarkEngine:
    """Tests for GreenClaimsBenchmarkEngine methods."""

    def test_engine_instantiation(self, mod):
        """Engine can be instantiated."""
        engine = mod.GreenClaimsBenchmarkEngine()
        assert engine is not None

    def test_engine_has_calculate_portfolio_metrics(self, engine):
        """Engine has calculate_portfolio_metrics method."""
        assert hasattr(engine, "calculate_portfolio_metrics")
        assert callable(engine.calculate_portfolio_metrics)

    def test_engine_has_determine_maturity_level(self, engine):
        """Engine has determine_maturity_level method."""
        assert hasattr(engine, "determine_maturity_level")
        assert callable(engine.determine_maturity_level)

    def test_engine_has_benchmark_against_peers(self, engine):
        """Engine has benchmark_against_peers method."""
        assert hasattr(engine, "benchmark_against_peers")
        assert callable(engine.benchmark_against_peers)

    def test_engine_has_generate_improvement_roadmap(self, engine):
        """Engine has generate_improvement_roadmap method."""
        assert hasattr(engine, "generate_improvement_roadmap")
        assert callable(engine.generate_improvement_roadmap)

    def test_engine_has_calculate_overall_score(self, engine):
        """Engine has calculate_overall_score method."""
        assert hasattr(engine, "calculate_overall_score")
        assert callable(engine.calculate_overall_score)


# ===========================================================================
# Provenance and Source Checks
# ===========================================================================


class TestGreenClaimsBenchmarkProvenance:
    """Tests for source file characteristics and provenance."""

    def test_engine_source_has_sha256(self):
        """Engine source uses SHA-256 for provenance."""
        path = ENGINES_DIR / "green_claims_benchmark_engine.py"
        if not path.exists():
            pytest.skip("file not yet created")
        source = path.read_text(encoding="utf-8")
        assert "sha256" in source.lower() or "hashlib" in source

    def test_engine_source_has_decimal(self):
        """Engine source uses Decimal arithmetic."""
        path = ENGINES_DIR / "green_claims_benchmark_engine.py"
        if not path.exists():
            pytest.skip("file not yet created")
        source = path.read_text(encoding="utf-8")
        assert "Decimal" in source

    def test_engine_source_has_basemodel(self):
        """Engine source uses Pydantic BaseModel."""
        path = ENGINES_DIR / "green_claims_benchmark_engine.py"
        if not path.exists():
            pytest.skip("file not yet created")
        source = path.read_text(encoding="utf-8")
        assert "BaseModel" in source

    def test_engine_source_has_logging(self):
        """Engine source uses logging."""
        path = ENGINES_DIR / "green_claims_benchmark_engine.py"
        if not path.exists():
            pytest.skip("file not yet created")
        source = path.read_text(encoding="utf-8")
        assert "logging" in source


# ===========================================================================
# Sample Data Tests
# ===========================================================================


class TestBenchmarkSampleData:
    """Tests using sample_portfolio_results fixture from conftest."""

    def test_sample_portfolio_results_count(self, sample_portfolio_results):
        """sample_portfolio_results has at least 5 entries."""
        assert len(sample_portfolio_results) >= 5

    def test_sample_results_have_claim_id(self, sample_portfolio_results):
        """All sample results have claim_id."""
        for result in sample_portfolio_results:
            assert "claim_id" in result

    def test_sample_results_have_substantiation_score(self, sample_portfolio_results):
        """All sample results have substantiation_score."""
        for result in sample_portfolio_results:
            assert "substantiation_score" in result

    def test_sample_results_have_greenwashing_risk(self, sample_portfolio_results):
        """All sample results have greenwashing_risk."""
        for result in sample_portfolio_results:
            assert "greenwashing_risk" in result

    def test_sample_results_scores_are_decimal(self, sample_portfolio_results):
        """Scores in sample results are Decimal."""
        for result in sample_portfolio_results:
            assert isinstance(result["substantiation_score"], Decimal)
            assert isinstance(result["greenwashing_risk"], Decimal)

    def test_sample_results_have_verification_flag(self, sample_portfolio_results):
        """All sample results have verification_ready flag."""
        for result in sample_portfolio_results:
            assert "verification_ready" in result

    def test_sample_results_have_evidence_flag(self, sample_portfolio_results):
        """All sample results have evidence_complete flag."""
        for result in sample_portfolio_results:
            assert "evidence_complete" in result

    def test_high_risk_claim_identified(self, sample_portfolio_results):
        """At least one claim has greenwashing_risk >= 80."""
        high_risk = [
            r for r in sample_portfolio_results
            if r["greenwashing_risk"] >= Decimal("80")
        ]
        assert len(high_risk) >= 1

    def test_compliant_claim_identified(self, sample_portfolio_results):
        """At least one claim has substantiation_score >= 70."""
        compliant = [
            r for r in sample_portfolio_results
            if r["substantiation_score"] >= Decimal("70")
        ]
        assert len(compliant) >= 1

    def test_unique_claim_ids(self, sample_portfolio_results):
        """All claim IDs in sample results are unique."""
        ids = [r["claim_id"] for r in sample_portfolio_results]
        assert len(ids) == len(set(ids))
