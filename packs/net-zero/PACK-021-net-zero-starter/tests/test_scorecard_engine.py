# -*- coding: utf-8 -*-
"""
Unit tests for NetZeroScorecardEngine (PACK-021 Engine 7).

Tests 8-dimension scoring, maturity level assignment, weighted overall score,
recommendation generation, priority ranking, gap analysis, and provenance
hashing across all five maturity levels.

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

from engines.net_zero_scorecard_engine import (
    BENCHMARK_CONTEXT,
    DEFAULT_DIMENSION_WEIGHTS,
    DIMENSION_INDICATORS,
    DimensionInput,
    DimensionScore,
    MATURITY_LEVELS,
    MaturityLevel,
    NetZeroScorecardEngine,
    RecommendationPriority,
    ScorecardDimension,
    ScorecardInput,
    ScorecardRecommendation,
    ScorecardResult,
)


# ========================================================================
# Helper functions
# ========================================================================


def _build_dimension_input(
    dimension: ScorecardDimension,
    fill_pct: Decimal = Decimal("50"),
) -> DimensionInput:
    """Create a DimensionInput with all indicators filled at fill_pct% of max.

    Args:
        dimension: The scorecard dimension to build input for.
        fill_pct: Percentage (0-100) of max points to assign per indicator.

    Returns:
        DimensionInput with computed indicator scores.
    """
    indicators = DIMENSION_INDICATORS.get(dimension.value, [])
    scores = {}
    for ind in indicators:
        max_pts = ind["max_points"]
        scores[ind["id"]] = max_pts * fill_pct / Decimal("100")
    return DimensionInput(dimension=dimension, indicator_scores=scores)


def _build_full_input(
    fill_pct: Decimal = Decimal("50"),
    entity_name: str = "TestCo",
    sector: str = "manufacturing",
) -> ScorecardInput:
    """Build a full ScorecardInput with all 8 dimensions at the given percentage."""
    dims = [
        _build_dimension_input(d, fill_pct)
        for d in ScorecardDimension
    ]
    return ScorecardInput(
        entity_name=entity_name,
        sector=sector,
        assessment_year=2026,
        dimensions=dims,
    )


# ========================================================================
# Fixtures
# ========================================================================


@pytest.fixture
def engine() -> NetZeroScorecardEngine:
    """Fresh NetZeroScorecardEngine."""
    return NetZeroScorecardEngine()


@pytest.fixture
def full_input_50pct() -> ScorecardInput:
    """All dimensions at 50% of max points (overall ~50, Level 3)."""
    return _build_full_input(fill_pct=Decimal("50"))


@pytest.fixture
def full_input_0pct() -> ScorecardInput:
    """All dimensions at 0% (overall ~0, Level 1)."""
    return _build_full_input(fill_pct=Decimal("0"))


@pytest.fixture
def full_input_100pct() -> ScorecardInput:
    """All dimensions at 100% (overall ~100, Level 5)."""
    return _build_full_input(fill_pct=Decimal("100"))


# ========================================================================
# Instantiation
# ========================================================================


class TestScorecardEngineInstantiation:
    """Tests for engine creation."""

    def test_engine_instantiates(self):
        """Engine creates without error."""
        engine = NetZeroScorecardEngine()
        assert engine is not None
        assert engine.engine_version == "1.0.0"

    def test_engine_has_assess_method(self, engine):
        """Engine exposes an `assess` method."""
        assert callable(getattr(engine, "assess", None))


# ========================================================================
# 8 Dimensions Scored
# ========================================================================


class TestEightDimensionsScored:
    """Validate that all 8 dimensions are scored."""

    def test_8_dimensions_in_result(self, engine, full_input_50pct):
        """Result contains scores for all 8 dimensions."""
        result = engine.assess(full_input_50pct)
        assert len(result.dimension_scores) == 8

    def test_all_dimension_enums_present(self, engine, full_input_50pct):
        """Every ScorecardDimension enum appears in the result."""
        result = engine.assess(full_input_50pct)
        result_dims = {ds.dimension for ds in result.dimension_scores}
        expected_dims = set(ScorecardDimension)
        assert result_dims == expected_dims

    @pytest.mark.parametrize(
        "dim",
        list(ScorecardDimension),
        ids=[d.value for d in ScorecardDimension],
    )
    def test_each_dimension_has_indicators(self, dim):
        """Each dimension has indicator definitions in DIMENSION_INDICATORS."""
        assert dim.value in DIMENSION_INDICATORS
        assert len(DIMENSION_INDICATORS[dim.value]) >= 1

    def test_dimension_scores_are_dimension_score_objects(self, engine, full_input_50pct):
        """Each dimension score is a DimensionScore instance."""
        result = engine.assess(full_input_50pct)
        for ds in result.dimension_scores:
            assert isinstance(ds, DimensionScore)

    def test_dimension_raw_scores_0_to_100(self, engine, full_input_50pct):
        """All raw scores are between 0 and 100."""
        result = engine.assess(full_input_50pct)
        for ds in result.dimension_scores:
            assert Decimal("0") <= ds.raw_score <= Decimal("100")


# ========================================================================
# Maturity Levels
# ========================================================================


class TestMaturityLevels:
    """Tests for the 5-level maturity model."""

    def test_five_maturity_levels_defined(self):
        """MATURITY_LEVELS has exactly 5 entries."""
        assert len(MATURITY_LEVELS) == 5

    @pytest.mark.parametrize(
        "level,expected_min,expected_max",
        [
            (MaturityLevel.AWARENESS, Decimal("0"), Decimal("20")),
            (MaturityLevel.FOUNDATION, Decimal("21"), Decimal("40")),
            (MaturityLevel.DEVELOPING, Decimal("41"), Decimal("60")),
            (MaturityLevel.ADVANCED, Decimal("61"), Decimal("80")),
            (MaturityLevel.LEADING, Decimal("81"), Decimal("100")),
        ],
    )
    def test_maturity_level_thresholds(self, level, expected_min, expected_max):
        """Each maturity level has correct min/max score thresholds."""
        level_data = MATURITY_LEVELS[level.value]
        assert level_data["min_score"] == expected_min
        assert level_data["max_score"] == expected_max


# ========================================================================
# Overall Weighted Score
# ========================================================================


class TestOverallWeightedScore:
    """Tests for overall score calculation."""

    def test_weights_sum_to_one(self):
        """Default dimension weights sum to 1.0."""
        total = sum(DEFAULT_DIMENSION_WEIGHTS.values())
        assert total == Decimal("1.00")

    def test_overall_score_at_50_pct(self, engine, full_input_50pct):
        """All dimensions at 50% -> overall ~50."""
        result = engine.assess(full_input_50pct)
        assert Decimal("45") <= result.overall_score <= Decimal("55")

    def test_overall_score_at_0_pct(self, engine, full_input_0pct):
        """All dimensions at 0% -> overall ~0."""
        result = engine.assess(full_input_0pct)
        assert result.overall_score <= Decimal("5")

    def test_overall_score_at_100_pct(self, engine, full_input_100pct):
        """All dimensions at 100% -> overall ~100."""
        result = engine.assess(full_input_100pct)
        assert result.overall_score >= Decimal("95")

    def test_overall_score_between_0_and_100(self, engine, full_input_50pct):
        """Overall score is in valid range."""
        result = engine.assess(full_input_50pct)
        assert Decimal("0") <= result.overall_score <= Decimal("100")


# ========================================================================
# Level 1 / 3 / 5 Maturity
# ========================================================================


class TestMaturityLevelAssignment:
    """Tests for maturity level assignment from overall score."""

    def test_level_1_awareness(self, engine, full_input_0pct):
        """Score ~0 -> Level 1: Awareness."""
        result = engine.assess(full_input_0pct)
        assert result.maturity_level == MaturityLevel.AWARENESS
        assert "Awareness" in result.maturity_label

    def test_level_3_developing(self, engine, full_input_50pct):
        """Score ~50 -> Level 3: Developing."""
        result = engine.assess(full_input_50pct)
        assert result.maturity_level == MaturityLevel.DEVELOPING
        assert "Developing" in result.maturity_label

    def test_level_5_leading(self, engine, full_input_100pct):
        """Score ~100 -> Level 5: Leading."""
        result = engine.assess(full_input_100pct)
        assert result.maturity_level == MaturityLevel.LEADING
        assert "Leading" in result.maturity_label

    def test_level_2_foundation(self, engine):
        """Score ~30 -> Level 2: Foundation."""
        inp = _build_full_input(fill_pct=Decimal("30"))
        result = engine.assess(inp)
        assert result.maturity_level == MaturityLevel.FOUNDATION

    def test_level_4_advanced(self, engine):
        """Score ~70 -> Level 4: Advanced."""
        inp = _build_full_input(fill_pct=Decimal("70"))
        result = engine.assess(inp)
        assert result.maturity_level == MaturityLevel.ADVANCED


# ========================================================================
# Recommendations
# ========================================================================


class TestRecommendationsGenerated:
    """Tests for recommendation generation."""

    def test_recommendations_generated(self, engine, full_input_50pct):
        """Recommendations list is non-empty for a partial score."""
        result = engine.assess(full_input_50pct)
        assert isinstance(result.recommendations, list)
        assert len(result.recommendations) >= 1

    def test_recommendation_structure(self, engine, full_input_50pct):
        """Each recommendation is a ScorecardRecommendation with required fields."""
        result = engine.assess(full_input_50pct)
        for rec in result.recommendations:
            assert isinstance(rec, ScorecardRecommendation)
            assert rec.title
            assert rec.priority in (
                RecommendationPriority.CRITICAL,
                RecommendationPriority.HIGH,
                RecommendationPriority.MEDIUM,
                RecommendationPriority.LOW,
            )

    def test_no_recommendations_for_perfect_score(self, engine, full_input_100pct):
        """Perfect score may generate fewer or zero critical recommendations."""
        result = engine.assess(full_input_100pct)
        critical_recs = [
            r for r in result.recommendations
            if r.priority == RecommendationPriority.CRITICAL
        ]
        # Should be zero critical recommendations at 100%
        assert len(critical_recs) == 0


# ========================================================================
# Priority Ranking
# ========================================================================


class TestPriorityRanking:
    """Tests for action priority grouping."""

    def test_action_priorities_dict(self, engine, full_input_50pct):
        """action_priorities is a dict keyed by priority level."""
        result = engine.assess(full_input_50pct)
        assert isinstance(result.action_priorities, dict)

    def test_strongest_and_weakest_dimensions(self, engine, full_input_50pct):
        """Result identifies strongest and weakest dimensions."""
        result = engine.assess(full_input_50pct)
        assert len(result.strongest_dimensions) > 0
        assert len(result.weakest_dimensions) > 0

    def test_gap_summary_populated(self, engine, full_input_50pct):
        """Gap summary is populated for each dimension."""
        result = engine.assess(full_input_50pct)
        assert isinstance(result.gap_summary, dict)


# ========================================================================
# Provenance Hash
# ========================================================================


class TestScorecardProvenanceHash:
    """Tests for SHA-256 provenance hashing."""

    def test_provenance_hash_present(self, engine, full_input_50pct):
        """Result has a 64-char hex provenance hash."""
        result = engine.assess(full_input_50pct)
        assert result.provenance_hash
        assert len(result.provenance_hash) == 64

    def test_provenance_hash_is_valid_sha256(self, engine):
        """Provenance hash is a valid SHA-256 hex string."""
        inp = _build_full_input(fill_pct=Decimal("50"), entity_name="HashTestCo")
        r1 = engine.assess(inp)
        r2 = engine.assess(inp)
        assert len(r1.provenance_hash) == 64
        assert len(r2.provenance_hash) == 64
        assert all(c in "0123456789abcdef" for c in r1.provenance_hash)
        assert all(c in "0123456789abcdef" for c in r2.provenance_hash)

    def test_different_input_different_hash(self, engine):
        """Different inputs produce different hashes."""
        inp1 = _build_full_input(fill_pct=Decimal("50"))
        inp2 = _build_full_input(fill_pct=Decimal("70"))
        r1 = engine.assess(inp1)
        r2 = engine.assess(inp2)
        assert r1.provenance_hash != r2.provenance_hash


# ========================================================================
# Custom Weights
# ========================================================================


class TestCustomWeights:
    """Tests for custom dimension weights."""

    def test_custom_weights_applied(self, engine):
        """Custom weights change the overall score."""
        custom_weights = {
            ScorecardDimension.GHG_INVENTORY.value: Decimal("1.00"),
            ScorecardDimension.TARGET_AMBITION.value: Decimal("0.00"),
            ScorecardDimension.REDUCTION_PROGRESS.value: Decimal("0.00"),
            ScorecardDimension.DECARBONIZATION_ACTIONS.value: Decimal("0.00"),
            ScorecardDimension.GOVERNANCE_STRATEGY.value: Decimal("0.00"),
            ScorecardDimension.FINANCIAL_PLANNING.value: Decimal("0.00"),
            ScorecardDimension.OFFSET_NEUTRALIZATION.value: Decimal("0.00"),
            ScorecardDimension.REPORTING_DISCLOSURE.value: Decimal("0.00"),
        }
        # Only GHG inventory at 80%, everything else at 20%
        dims = []
        for d in ScorecardDimension:
            if d == ScorecardDimension.GHG_INVENTORY:
                dims.append(_build_dimension_input(d, Decimal("80")))
            else:
                dims.append(_build_dimension_input(d, Decimal("20")))
        inp = ScorecardInput(
            entity_name="WeightTest",
            sector="services",
            dimensions=dims,
            custom_weights=custom_weights,
        )
        result = engine.assess(inp)
        # With 100% weight on GHG inventory at 80%, overall ~ 80
        assert result.overall_score >= Decimal("70")


# ========================================================================
# Result Structure
# ========================================================================


class TestScorecardResultStructure:
    """Validate complete result structure."""

    def test_result_has_all_required_fields(self, engine, full_input_50pct):
        """ScorecardResult has all expected fields."""
        result = engine.assess(full_input_50pct)
        assert isinstance(result, ScorecardResult)
        assert result.result_id
        assert result.engine_version == "1.0.0"
        assert result.entity_name == "TestCo"
        assert result.sector == "manufacturing"
        assert result.assessment_year == 2026
        assert result.processing_time_ms >= 0.0
        assert result.maturity_description

    def test_benchmark_context_for_all_dimensions(self):
        """BENCHMARK_CONTEXT has entries for all 8 dimensions."""
        for dim in ScorecardDimension:
            assert dim.value in BENCHMARK_CONTEXT
            ctx = BENCHMARK_CONTEXT[dim.value]
            assert "leading" in ctx
            assert "awareness" in ctx
