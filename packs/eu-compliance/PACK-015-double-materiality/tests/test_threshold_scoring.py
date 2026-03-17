# -*- coding: utf-8 -*-
"""
PACK-015 Double Materiality Assessment Pack - Threshold Scoring Engine Tests
==============================================================================

Unit tests for ThresholdScoringEngine (Engine 7) covering scoring profiles,
matter scoring, batch scoring, sector adjustment, normalization, sensitivity
analysis, industry thresholds, methodology comparison, and provenance hashing.

5 methodologies: geometric_mean, arithmetic_mean, weighted_sum, max_score, product.

Target: 40+ tests.

Author:  GreenLang Platform Team (GL-TestEngineer)
Pack:    PACK-015 Double Materiality Assessment
Date:    March 2026
"""

from decimal import Decimal

import pytest

from .conftest import _load_engine


# ---------------------------------------------------------------------------
# Module-scoped engine loading
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def mod():
    """Load the threshold_scoring engine module."""
    return _load_engine("threshold_scoring")


@pytest.fixture
def engine(mod):
    """Create a ThresholdScoringEngine with default sector."""
    return mod.ThresholdScoringEngine()


@pytest.fixture
def manufacturing_engine(mod):
    """Create a ThresholdScoringEngine for manufacturing sector."""
    return mod.ThresholdScoringEngine(sector="manufacturing")


@pytest.fixture
def default_profile(engine):
    """Create a default scoring profile."""
    return engine.create_scoring_profile()


@pytest.fixture
def sample_raw_score(mod):
    """Create a sample RawScoreInput for testing.

    Uses the actual model fields: matter_id, esrs_topic, sub_scores.
    sub_scores is a Dict[str, Decimal] of criterion_name -> score (0-5).
    """
    return mod.RawScoreInput(
        matter_id="TS-001",
        esrs_topic="E1",
        sub_scores={
            "scale": Decimal("3.5"),
            "scope": Decimal("4.0"),
            "irremediable_character": Decimal("3.0"),
        },
    )


# ===========================================================================
# Enum Tests
# ===========================================================================


class TestThresholdScoringEnums:
    """Tests for threshold scoring enums."""

    def test_scoring_methodology_count(self, mod):
        """ScoringMethodology has 5 values."""
        assert len(mod.ScoringMethodology) == 5
        names = {m.name for m in mod.ScoringMethodology}
        expected = {
            "GEOMETRIC_MEAN", "ARITHMETIC_MEAN",
            "WEIGHTED_SUM", "MAX_SCORE", "PRODUCT",
        }
        assert names == expected

    def test_normalization_method_count(self, mod):
        """NormalizationMethod has 4 values."""
        assert len(mod.NormalizationMethod) == 4
        names = {m.name for m in mod.NormalizationMethod}
        expected = {"MIN_MAX", "Z_SCORE", "PERCENTILE", "NONE"}
        assert names == expected

    def test_threshold_source_count(self, mod):
        """ThresholdSource has 4 values."""
        assert len(mod.ThresholdSource) == 4


# ===========================================================================
# Constants Tests
# ===========================================================================


class TestThresholdScoringConstants:
    """Tests for threshold scoring constants."""

    def test_industry_thresholds_has_default(self, mod):
        """INDUSTRY_THRESHOLDS has a 'default' entry."""
        assert "default" in mod.INDUSTRY_THRESHOLDS

    def test_industry_thresholds_count(self, mod):
        """INDUSTRY_THRESHOLDS has at least 9 sectors plus default."""
        assert len(mod.INDUSTRY_THRESHOLDS) >= 10

    def test_industry_thresholds_structure(self, mod):
        """Each threshold entry has impact_threshold key."""
        for sector, thresholds in mod.INDUSTRY_THRESHOLDS.items():
            assert "impact_threshold" in thresholds, (
                f"Sector {sector} missing impact_threshold"
            )

    def test_sector_adjustment_factors_has_default(self, mod):
        """SECTOR_ADJUSTMENT_FACTORS has a 'default' entry."""
        assert "default" in mod.SECTOR_ADJUSTMENT_FACTORS

    def test_sector_adjustment_factors_count(self, mod):
        """SECTOR_ADJUSTMENT_FACTORS has at least 7 sectors."""
        assert len(mod.SECTOR_ADJUSTMENT_FACTORS) >= 7


# ===========================================================================
# Engine Initialization Tests
# ===========================================================================


class TestThresholdEngineInit:
    """Tests for ThresholdScoringEngine initialization."""

    def test_default_sector(self, engine):
        """Default engine uses 'default' sector."""
        assert engine.sector == "default"

    def test_manufacturing_sector(self, manufacturing_engine):
        """Manufacturing engine uses 'manufacturing' sector."""
        assert manufacturing_engine.sector == "manufacturing"

    def test_custom_sector(self, mod):
        """Custom sector is accepted."""
        engine = mod.ThresholdScoringEngine(sector="energy")
        assert engine.sector == "energy"


# ===========================================================================
# Create Scoring Profile Tests
# ===========================================================================


class TestCreateScoringProfile:
    """Tests for create_scoring_profile method."""

    def test_create_default_profile(self, engine):
        """Create scoring profile with default settings."""
        profile = engine.create_scoring_profile()
        assert profile is not None
        assert hasattr(profile, "methodology")
        assert profile.methodology.value == "arithmetic_mean"

    def test_create_profile_with_methodology(self, engine, mod):
        """Create scoring profile with specific methodology."""
        profile = engine.create_scoring_profile(
            methodology=mod.ScoringMethodology.GEOMETRIC_MEAN,
        )
        assert profile is not None
        assert profile.methodology == mod.ScoringMethodology.GEOMETRIC_MEAN


# ===========================================================================
# Score Matter Tests
# ===========================================================================


class TestScoreMatter:
    """Tests for score_matter method."""

    def test_score_matter_basic(self, engine, sample_raw_score, default_profile):
        """score_matter returns a ScoringResult."""
        result = engine.score_matter(sample_raw_score, default_profile)
        assert result is not None
        assert hasattr(result, "weighted_score")
        assert hasattr(result, "passes_threshold")

    def test_score_matter_passes_threshold_high_score(self, engine, mod, default_profile):
        """High scores pass the threshold."""
        raw = mod.RawScoreInput(
            matter_id="HIGH-1",
            esrs_topic="E1",
            sub_scores={
                "scale": Decimal("4.5"),
                "scope": Decimal("4.5"),
                "irremediable_character": Decimal("4.5"),
            },
        )
        result = engine.score_matter(raw, default_profile)
        assert result.passes_threshold is True

    def test_score_matter_fails_threshold_low_score(self, engine, mod, default_profile):
        """Low scores fail the threshold."""
        raw = mod.RawScoreInput(
            matter_id="LOW-1",
            esrs_topic="E1",
            sub_scores={
                "scale": Decimal("1.0"),
                "scope": Decimal("1.0"),
                "irremediable_character": Decimal("1.0"),
            },
        )
        result = engine.score_matter(raw, default_profile)
        assert result.passes_threshold is False

    def test_score_matter_deterministic(self, engine, sample_raw_score, default_profile):
        """Same inputs produce same weighted score."""
        r1 = engine.score_matter(sample_raw_score, default_profile)
        r2 = engine.score_matter(sample_raw_score, default_profile)
        assert r1.weighted_score == r2.weighted_score

    def test_score_matter_provenance_hash(self, engine, sample_raw_score, default_profile):
        """Scoring result has provenance hash."""
        result = engine.score_matter(sample_raw_score, default_profile)
        assert hasattr(result, "provenance_hash")
        assert len(result.provenance_hash) == 64


# ===========================================================================
# Batch Score Tests
# ===========================================================================


class TestBatchScore:
    """Tests for batch_score method."""

    def test_batch_score_basic(self, engine, mod, default_profile):
        """batch_score processes multiple raw scores."""
        raws = [
            mod.RawScoreInput(
                matter_id=f"BS-{i}",
                esrs_topic="E1",
                sub_scores={
                    "scale": Decimal(str(1.0 + i * 0.8)),
                    "scope": Decimal(str(1.0 + i * 0.7)),
                    "irremediable_character": Decimal(str(1.0 + i * 0.6)),
                },
            )
            for i in range(5)
        ]
        results = engine.batch_score(raws, default_profile)
        assert isinstance(results, list)
        assert len(results) == 5

    def test_batch_score_ranking(self, engine, mod, default_profile):
        """batch_score assigns percentile ranks."""
        raws = [
            mod.RawScoreInput(
                matter_id=f"RANK-{i}",
                esrs_topic="E1",
                sub_scores={
                    "scale": Decimal(str(1.0 + i * 0.8)),
                    "scope": Decimal(str(1.0 + i * 0.7)),
                },
            )
            for i in range(5)
        ]
        results = engine.batch_score(raws, default_profile)
        # Check that percentile is present on each result
        for r in results:
            assert hasattr(r, "percentile")
            assert r.percentile >= Decimal("0")

    def test_batch_score_ordering(self, engine, mod, default_profile):
        """Highest score should have highest percentile."""
        raws = [
            mod.RawScoreInput(
                matter_id="ORD-LOW",
                esrs_topic="E1",
                sub_scores={"scale": Decimal("1.0"), "scope": Decimal("1.0")},
            ),
            mod.RawScoreInput(
                matter_id="ORD-HIGH",
                esrs_topic="E1",
                sub_scores={"scale": Decimal("4.5"), "scope": Decimal("4.5")},
            ),
        ]
        results = engine.batch_score(raws, default_profile)
        assert len(results) == 2
        # Find the high and low results
        low_result = next(r for r in results if r.matter_id == "ORD-LOW")
        high_result = next(r for r in results if r.matter_id == "ORD-HIGH")
        assert high_result.percentile > low_result.percentile


# ===========================================================================
# Sector Adjustment Tests
# ===========================================================================


class TestSectorAdjustment:
    """Tests for apply_sector_adjustment method."""

    def test_sector_adjustment_default(self, engine):
        """Default sector adjustment is neutral (1.0 multiplier)."""
        adjusted = engine.apply_sector_adjustment(
            Decimal("3.0"), "default", "E1",
        )
        assert adjusted is not None
        assert isinstance(adjusted, Decimal)
        # Default adjustment factor for E1 is 1.0, so score stays 3.0
        assert adjusted == Decimal("3.000000")

    def test_sector_adjustment_manufacturing(self, manufacturing_engine):
        """Manufacturing sector amplifies E1 scores (factor 1.20)."""
        adjusted = manufacturing_engine.apply_sector_adjustment(
            Decimal("3.0"), "manufacturing", "E1",
        )
        assert adjusted is not None
        # 3.0 * 1.20 = 3.6
        assert adjusted == Decimal("3.600000")

    def test_sector_adjustment_preserves_bounds(self, engine):
        """Adjusted score is capped at 5.0."""
        adjusted = engine.apply_sector_adjustment(
            Decimal("4.8"), "energy", "E1",
        )
        assert adjusted >= Decimal("0")
        assert adjusted <= Decimal("5.000000")


# ===========================================================================
# Normalization Tests
# ===========================================================================


class TestNormalization:
    """Tests for normalize_scores method."""

    def test_normalize_min_max(self, engine, mod):
        """MIN_MAX normalization scales to 0-1."""
        scores = [
            Decimal("1.0"), Decimal("2.0"), Decimal("3.0"),
            Decimal("4.0"), Decimal("5.0"),
        ]
        normalized = engine.normalize_scores(
            scores, method=mod.NormalizationMethod.MIN_MAX,
        )
        assert isinstance(normalized, list)
        assert len(normalized) == 5
        # Min should be 0, max should be 1
        assert float(min(normalized)) == pytest.approx(0.0, abs=0.01)
        assert float(max(normalized)) == pytest.approx(1.0, abs=0.01)

    def test_normalize_none(self, engine, mod):
        """NONE normalization returns unchanged scores."""
        scores = [Decimal("1.0"), Decimal("3.0"), Decimal("5.0")]
        normalized = engine.normalize_scores(
            scores, method=mod.NormalizationMethod.NONE,
        )
        for orig, norm in zip(scores, normalized):
            assert float(norm) == pytest.approx(float(orig), abs=0.01)


# ===========================================================================
# Sensitivity Analysis Tests
# ===========================================================================


class TestSensitivityAnalysis:
    """Tests for run_sensitivity_analysis method."""

    def test_sensitivity_analysis_basic(self, engine):
        """run_sensitivity_analysis returns analysis result."""
        analysis = engine.run_sensitivity_analysis(
            matter_id="SA-1",
            score=Decimal("3.2"),
        )
        assert analysis is not None
        assert hasattr(analysis, "sensitivity_results")
        assert len(analysis.sensitivity_results) > 0

    def test_sensitivity_analysis_identifies_borderline(self, engine):
        """Sensitivity analysis identifies borderline matters."""
        # Score close to the default threshold of 3.0
        analysis = engine.run_sensitivity_analysis(
            matter_id="BORDER-1",
            score=Decimal("3.1"),
            base_threshold=Decimal("3.0"),
        )
        assert analysis is not None
        # Score is within +/- 0.5 of threshold and has a breakpoint
        assert analysis.is_borderline is True

    def test_sensitivity_analysis_not_borderline_far_above(self, engine):
        """Matters far above threshold are not borderline."""
        analysis = engine.run_sensitivity_analysis(
            matter_id="FAR-ABOVE-1",
            score=Decimal("4.8"),
            base_threshold=Decimal("3.0"),
        )
        assert analysis is not None
        assert analysis.is_borderline is False


# ===========================================================================
# Industry Threshold Tests
# ===========================================================================


class TestIndustryThreshold:
    """Tests for get_industry_threshold method."""

    def test_get_default_threshold(self, engine):
        """get_industry_threshold returns ThresholdSet for default sector."""
        threshold = engine.get_industry_threshold()
        assert threshold is not None
        assert hasattr(threshold, "impact_threshold")
        assert threshold.impact_threshold == Decimal("3.000")

    def test_get_manufacturing_threshold(self, engine):
        """get_industry_threshold returns ThresholdSet for manufacturing."""
        threshold = engine.get_industry_threshold(sector="manufacturing")
        assert threshold is not None
        assert threshold.impact_threshold == Decimal("3.000")

    def test_get_energy_threshold(self, engine):
        """get_industry_threshold returns ThresholdSet for energy."""
        threshold = engine.get_industry_threshold(sector="energy")
        assert threshold is not None
        # Energy has lower impact threshold
        assert threshold.impact_threshold == Decimal("2.500")


# ===========================================================================
# Methodology Comparison Tests
# ===========================================================================


class TestMethodologyComparison:
    """Tests for compare_methodologies method."""

    def test_compare_methodologies_basic(self, engine):
        """compare_methodologies returns comparison of all 5 methodologies."""
        comparison = engine.compare_methodologies(
            matter_id="CMP-1",
            sub_scores={
                "scale": Decimal("3.5"),
                "scope": Decimal("4.0"),
                "irremediable_character": Decimal("3.0"),
            },
        )
        assert comparison is not None
        assert isinstance(comparison, dict)
        assert len(comparison) == 5

    def test_compare_methodologies_all_present(self, engine):
        """All 5 methodologies appear in comparison results."""
        comparison = engine.compare_methodologies(
            matter_id="CMA-1",
            sub_scores={
                "scale": Decimal("3.0"),
                "scope": Decimal("3.5"),
            },
        )
        expected_keys = {
            "geometric_mean", "arithmetic_mean",
            "weighted_sum", "max_score", "product",
        }
        assert set(comparison.keys()) == expected_keys

    def test_compare_methodologies_max_score_highest(self, engine):
        """MAX_SCORE methodology should return the highest sub-score."""
        comparison = engine.compare_methodologies(
            matter_id="MAX-CHECK",
            sub_scores={
                "criterion_a": Decimal("2.0"),
                "criterion_b": Decimal("4.5"),
            },
        )
        # max_score should be 4.5
        assert comparison["max_score"]["score"] == pytest.approx(4.5, abs=0.01)


# ===========================================================================
# Provenance Hash Tests
# ===========================================================================


class TestThresholdProvenanceHash:
    """Tests for threshold scoring provenance hashing."""

    def test_hash_is_64_chars(self, engine, sample_raw_score, default_profile):
        """Provenance hash is a 64-character SHA-256 hex string."""
        result = engine.score_matter(sample_raw_score, default_profile)
        assert len(result.provenance_hash) == 64
        int(result.provenance_hash, 16)

    def test_hash_deterministic(self, engine, mod, default_profile):
        """Same inputs produce same provenance hash."""
        raw = mod.RawScoreInput(
            matter_id="DET-TH",
            esrs_topic="E1",
            sub_scores={
                "scale": Decimal("3.0"),
                "scope": Decimal("3.5"),
            },
        )
        r1 = engine.score_matter(raw, default_profile)
        r2 = engine.score_matter(raw, default_profile)
        assert r1.provenance_hash == r2.provenance_hash
