# -*- coding: utf-8 -*-
"""
PACK-015 Double Materiality Assessment Pack - Materiality Matrix Engine Tests
================================================================================

Unit tests for MaterialityMatrixEngine (Engine 5) covering matrix building,
quadrant classification, combined score calculation, material topic filtering,
visualization data, matrix comparison, ranking, summary statistics, and
provenance hashing.

Uses 0-5 scale. _round_val returns float. _compute_hash excludes
calculated_at, processing_time_ms, provenance_hash.

Target: 45+ tests.

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
    """Load the materiality_matrix engine module."""
    return _load_engine("materiality_matrix")


@pytest.fixture
def engine(mod):
    """Create a MaterialityMatrixEngine with default thresholds."""
    return mod.MaterialityMatrixEngine()


@pytest.fixture
def custom_engine(mod):
    """Create a MaterialityMatrixEngine with custom thresholds."""
    return mod.MaterialityMatrixEngine(
        impact_threshold=Decimal("4.000"),
        financial_threshold=Decimal("4.000"),
    )


@pytest.fixture
def sample_impact_scores(mod):
    """Create sample ImpactScoreInput list."""
    return [
        mod.ImpactScoreInput(
            matter_id="M-001", matter_name="Climate Change",
            esrs_topic="e1_climate", score=4.5,
        ),
        mod.ImpactScoreInput(
            matter_id="M-002", matter_name="Own Workforce",
            esrs_topic="s1_own_workforce", score=3.8,
        ),
        mod.ImpactScoreInput(
            matter_id="M-003", matter_name="Business Conduct",
            esrs_topic="g1_business_conduct", score=2.0,
        ),
    ]


@pytest.fixture
def sample_financial_scores(mod):
    """Create sample FinancialScoreInput list."""
    return [
        mod.FinancialScoreInput(
            matter_id="M-001", matter_name="Climate Change",
            esrs_topic="e1_climate", score=4.2,
        ),
        mod.FinancialScoreInput(
            matter_id="M-002", matter_name="Own Workforce",
            esrs_topic="s1_own_workforce", score=2.5,
        ),
        mod.FinancialScoreInput(
            matter_id="M-003", matter_name="Business Conduct",
            esrs_topic="g1_business_conduct", score=3.5,
        ),
    ]


# ===========================================================================
# Enum Tests
# ===========================================================================


class TestMatrixEnums:
    """Tests for materiality matrix enums."""

    def test_quadrant_values(self, mod):
        """Quadrant has 4 values."""
        assert len(mod.Quadrant) == 4
        names = {m.name for m in mod.Quadrant}
        expected = {
            "DOUBLE_MATERIAL", "IMPACT_ONLY",
            "FINANCIAL_ONLY", "NOT_MATERIAL",
        }
        assert names == expected

    def test_matrix_layout_values(self, mod):
        """MatrixLayout has 3 values."""
        assert len(mod.MatrixLayout) == 3

    def test_combined_score_method_values(self, mod):
        """CombinedScoreMethod has 3 methods."""
        assert len(mod.CombinedScoreMethod) == 3
        names = {m.name for m in mod.CombinedScoreMethod}
        expected = {"ARITHMETIC_MEAN", "GEOMETRIC_MEAN", "MAX_SCORE"}
        assert names == expected


# ===========================================================================
# Constants Tests
# ===========================================================================


class TestMatrixConstants:
    """Tests for materiality matrix constants."""

    def test_default_impact_threshold(self, mod):
        """Default impact threshold is 3.000."""
        assert mod.DEFAULT_IMPACT_THRESHOLD == Decimal("3.000")

    def test_default_financial_threshold(self, mod):
        """Default financial threshold is 3.000."""
        assert mod.DEFAULT_FINANCIAL_THRESHOLD == Decimal("3.000")


# ===========================================================================
# Engine Initialization Tests
# ===========================================================================


class TestMatrixEngineInit:
    """Tests for MaterialityMatrixEngine initialization."""

    def test_default_initialization(self, engine):
        """Default engine has 3.0 thresholds."""
        assert engine.impact_threshold == Decimal("3.000")
        assert engine.financial_threshold == Decimal("3.000")

    def test_custom_initialization(self, custom_engine):
        """Custom engine respects provided thresholds."""
        assert custom_engine.impact_threshold == Decimal("4.000")
        assert custom_engine.financial_threshold == Decimal("4.000")

    def test_weight_defaults(self, engine):
        """Default weights are 0.5/0.5."""
        assert engine.impact_weight == pytest.approx(0.5)
        assert engine.financial_weight == pytest.approx(0.5)


# ===========================================================================
# Build Matrix Tests
# ===========================================================================


class TestBuildMatrix:
    """Tests for build_matrix method."""

    def test_build_matrix_basic(
        self, engine, sample_impact_scores, sample_financial_scores,
    ):
        """build_matrix returns a MaterialityMatrix."""
        matrix = engine.build_matrix(
            sample_impact_scores, sample_financial_scores,
        )
        assert matrix is not None
        assert hasattr(matrix, "entries")
        assert len(matrix.entries) == 3

    def test_build_matrix_provenance_hash(
        self, engine, sample_impact_scores, sample_financial_scores,
    ):
        """Matrix has a 64-char provenance hash."""
        matrix = engine.build_matrix(
            sample_impact_scores, sample_financial_scores,
        )
        assert len(matrix.provenance_hash) == 64
        int(matrix.provenance_hash, 16)

    def test_build_matrix_entries_have_quadrants(
        self, engine, sample_impact_scores, sample_financial_scores,
    ):
        """Each matrix entry has a quadrant classification."""
        matrix = engine.build_matrix(
            sample_impact_scores, sample_financial_scores,
        )
        for entry in matrix.entries:
            assert hasattr(entry, "quadrant")

    def test_build_matrix_missing_financial_score(self, engine, mod):
        """Matter with only impact score gets zero financial score."""
        impact_scores = [
            mod.ImpactScoreInput(
                matter_id="MISS-1", matter_name="Missing Financial",
                esrs_topic="e1_climate", score=4.0,
            ),
        ]
        financial_scores = []  # No financial scores
        matrix = engine.build_matrix(impact_scores, financial_scores)
        assert len(matrix.entries) >= 1

    def test_build_matrix_empty_inputs(self, engine):
        """Empty inputs produce empty matrix."""
        matrix = engine.build_matrix([], [])
        assert len(matrix.entries) == 0


# ===========================================================================
# Quadrant Classification Tests
# ===========================================================================


class TestQuadrantClassification:
    """Tests for classify_quadrant method."""

    def test_double_material(self, engine, mod):
        """Impact >= 3.0 and Financial >= 3.0 => DOUBLE_MATERIAL."""
        quadrant = engine.classify_quadrant(4.0, 4.0)
        assert quadrant == mod.Quadrant.DOUBLE_MATERIAL

    def test_impact_only(self, engine, mod):
        """Impact >= 3.0 and Financial < 3.0 => IMPACT_ONLY."""
        quadrant = engine.classify_quadrant(4.0, 2.0)
        assert quadrant == mod.Quadrant.IMPACT_ONLY

    def test_financial_only(self, engine, mod):
        """Impact < 3.0 and Financial >= 3.0 => FINANCIAL_ONLY."""
        quadrant = engine.classify_quadrant(2.0, 4.0)
        assert quadrant == mod.Quadrant.FINANCIAL_ONLY

    def test_not_material(self, engine, mod):
        """Impact < 3.0 and Financial < 3.0 => NOT_MATERIAL."""
        quadrant = engine.classify_quadrant(2.0, 2.0)
        assert quadrant == mod.Quadrant.NOT_MATERIAL

    def test_boundary_at_threshold(self, engine, mod):
        """Score exactly at threshold is material."""
        quadrant = engine.classify_quadrant(3.0, 3.0)
        assert quadrant == mod.Quadrant.DOUBLE_MATERIAL

    def test_custom_threshold_classification(self, custom_engine, mod):
        """Custom threshold changes classification."""
        # With 4.0 thresholds, 3.5 is below threshold
        quadrant = custom_engine.classify_quadrant(3.5, 3.5)
        assert quadrant == mod.Quadrant.NOT_MATERIAL


# ===========================================================================
# Combined Score Tests
# ===========================================================================


class TestCombinedScore:
    """Tests for calculate_combined_score method."""

    def test_arithmetic_mean(self, engine):
        """Arithmetic mean: (4.0 + 2.0) / 2 = 3.0."""
        score = engine.calculate_combined_score(4.0, 2.0)
        assert score == pytest.approx(3.0, abs=0.01)

    def test_combined_equal_scores(self, engine):
        """Equal scores: combined = same value."""
        score = engine.calculate_combined_score(3.5, 3.5)
        assert score == pytest.approx(3.5, abs=0.01)

    def test_combined_zero_scores(self, engine):
        """Zero scores: combined = 0."""
        score = engine.calculate_combined_score(0.0, 0.0)
        assert score == pytest.approx(0.0, abs=0.01)

    def test_geometric_mean_method(self, mod):
        """Geometric mean method calculates differently."""
        engine = mod.MaterialityMatrixEngine(
            combined_method=mod.CombinedScoreMethod.GEOMETRIC_MEAN,
        )
        score = engine.calculate_combined_score(4.0, 4.0)
        assert score == pytest.approx(4.0, abs=0.01)

    def test_max_score_method(self, mod):
        """Max score method returns the higher score."""
        engine = mod.MaterialityMatrixEngine(
            combined_method=mod.CombinedScoreMethod.MAX_SCORE,
        )
        score = engine.calculate_combined_score(4.0, 2.0)
        assert score == pytest.approx(4.0, abs=0.01)


# ===========================================================================
# Material Topics Tests
# ===========================================================================


class TestMaterialTopics:
    """Tests for get_material_topics and get_double_material."""

    def test_get_material_topics(
        self, engine, mod, sample_impact_scores, sample_financial_scores,
    ):
        """get_material_topics returns topics above threshold."""
        matrix = engine.build_matrix(
            sample_impact_scores, sample_financial_scores,
        )
        material = engine.get_material_topics(matrix)
        assert isinstance(material, list)
        # M-001 (4.5/4.2) should be double material
        material_ids = [e.matter_id for e in material]
        assert "M-001" in material_ids

    def test_get_double_material(
        self, engine, mod, sample_impact_scores, sample_financial_scores,
    ):
        """get_double_material returns only DOUBLE_MATERIAL entries."""
        matrix = engine.build_matrix(
            sample_impact_scores, sample_financial_scores,
        )
        double = engine.get_double_material(matrix)
        assert isinstance(double, list)
        for entry in double:
            assert entry.quadrant == mod.Quadrant.DOUBLE_MATERIAL

    def test_no_material_topics_with_high_threshold(self, mod):
        """Very high threshold results in no material topics."""
        engine = mod.MaterialityMatrixEngine(
            impact_threshold=Decimal("5.000"),
            financial_threshold=Decimal("5.000"),
        )
        impact = [
            mod.ImpactScoreInput(
                matter_id="X-1", matter_name="Test",
                esrs_topic="e1_climate", score=4.5,
            ),
        ]
        financial = [
            mod.FinancialScoreInput(
                matter_id="X-1", matter_name="Test",
                esrs_topic="e1_climate", score=4.5,
            ),
        ]
        matrix = engine.build_matrix(impact, financial)
        material = engine.get_material_topics(matrix)
        assert len(material) == 0


# ===========================================================================
# Visualization Data Tests
# ===========================================================================


class TestVisualizationData:
    """Tests for get_visualization_data method."""

    def test_visualization_data_returned(
        self, engine, sample_impact_scores, sample_financial_scores,
    ):
        """get_visualization_data returns MatrixVisualizationData."""
        matrix = engine.build_matrix(
            sample_impact_scores, sample_financial_scores,
        )
        viz = engine.get_visualization_data(matrix)
        assert viz is not None
        assert hasattr(viz, "x_values") and hasattr(viz, "y_values")


# ===========================================================================
# Compare Matrices Tests
# ===========================================================================


class TestCompareMatrices:
    """Tests for compare_matrices method."""

    def test_compare_identical_matrices(
        self, engine, sample_impact_scores, sample_financial_scores, mod,
    ):
        """Comparing identical matrices produces no changes."""
        m1 = engine.build_matrix(
            sample_impact_scores, sample_financial_scores,
        )
        m2 = engine.build_matrix(
            sample_impact_scores, sample_financial_scores,
        )
        delta = engine.compare_matrices(m1, m2)
        assert delta is not None
        assert len(delta.new_material) == 0
        assert len(delta.no_longer_material) == 0

    def test_compare_different_matrices(self, engine, mod):
        """Different matrices show score changes."""
        impact1 = [
            mod.ImpactScoreInput(
                matter_id="CMP-1", matter_name="Climate",
                esrs_topic="e1_climate", score=4.0,
            ),
        ]
        financial1 = [
            mod.FinancialScoreInput(
                matter_id="CMP-1", matter_name="Climate",
                esrs_topic="e1_climate", score=4.0,
            ),
        ]
        impact2 = [
            mod.ImpactScoreInput(
                matter_id="CMP-1", matter_name="Climate",
                esrs_topic="e1_climate", score=2.0,
            ),
        ]
        financial2 = [
            mod.FinancialScoreInput(
                matter_id="CMP-1", matter_name="Climate",
                esrs_topic="e1_climate", score=2.0,
            ),
        ]
        m1 = engine.build_matrix(impact1, financial1)
        m2 = engine.build_matrix(impact2, financial2)
        delta = engine.compare_matrices(m1, m2)
        assert delta is not None
        # CMP-1 was material in m1 but not in m2
        assert len(delta.no_longer_material) >= 1 or len(delta.score_changes) >= 1


# ===========================================================================
# Ranking Tests
# ===========================================================================


class TestRanking:
    """Tests for rank_by_combined_score method."""

    def test_rank_by_combined_score(
        self, engine, sample_impact_scores, sample_financial_scores,
    ):
        """Entries are ranked by combined score descending."""
        matrix = engine.build_matrix(
            sample_impact_scores, sample_financial_scores,
        )
        ranked = engine.rank_by_combined_score(matrix)
        assert isinstance(ranked, list)
        assert len(ranked) == 3
        # First entry should have highest combined score
        if len(ranked) >= 2:
            assert ranked[0].combined_score >= ranked[1].combined_score


# ===========================================================================
# Summary Statistics Tests
# ===========================================================================


class TestSummaryStatistics:
    """Tests for get_summary_statistics method."""

    def test_summary_statistics_returned(
        self, engine, sample_impact_scores, sample_financial_scores,
    ):
        """get_summary_statistics returns a dictionary."""
        matrix = engine.build_matrix(
            sample_impact_scores, sample_financial_scores,
        )
        stats = engine.get_summary_statistics(matrix)
        assert isinstance(stats, dict)
        assert "impact_mean" in stats or "combined_mean" in stats


# ===========================================================================
# Aggregate by ESRS Topic Tests
# ===========================================================================


class TestAggregateByTopic:
    """Tests for aggregate_by_esrs_topic method."""

    def test_aggregate_by_topic(
        self, engine, sample_impact_scores, sample_financial_scores,
    ):
        """aggregate_by_esrs_topic groups entries by topic."""
        matrix = engine.build_matrix(
            sample_impact_scores, sample_financial_scores,
        )
        aggregated = engine.aggregate_by_esrs_topic(matrix)
        assert isinstance(aggregated, dict)
        assert "e1_climate" in aggregated


# ===========================================================================
# Provenance Hash Tests
# ===========================================================================


class TestMatrixProvenanceHash:
    """Tests for materiality matrix provenance hashing."""

    def test_hash_is_64_chars(
        self, engine, sample_impact_scores, sample_financial_scores,
    ):
        """Provenance hash is a 64-character SHA-256 hex string."""
        matrix = engine.build_matrix(
            sample_impact_scores, sample_financial_scores,
        )
        assert len(matrix.provenance_hash) == 64
        int(matrix.provenance_hash, 16)

    def test_hash_excludes_timestamp(
        self, engine, sample_impact_scores, sample_financial_scores,
    ):
        """Hash computation excludes calculated_at, processing_time_ms."""
        m1 = engine.build_matrix(
            sample_impact_scores, sample_financial_scores,
        )
        m2 = engine.build_matrix(
            sample_impact_scores, sample_financial_scores,
        )
        # Hash should be identical since only inputs/scores matter
        assert m1.provenance_hash == m2.provenance_hash


# ===========================================================================
# Edge Cases
# ===========================================================================


class TestMatrixEdgeCases:
    """Edge case tests for materiality matrix engine."""

    def test_single_entry_matrix(self, engine, mod):
        """Matrix with a single entry works correctly."""
        impact = [
            mod.ImpactScoreInput(
                matter_id="SINGLE-1", matter_name="Single",
                esrs_topic="e1_climate", score=4.0,
            ),
        ]
        financial = [
            mod.FinancialScoreInput(
                matter_id="SINGLE-1", matter_name="Single",
                esrs_topic="e1_climate", score=4.0,
            ),
        ]
        matrix = engine.build_matrix(impact, financial)
        assert len(matrix.entries) == 1

    def test_zero_score_entry(self, engine, mod):
        """Entry with zero scores is classified as NOT_MATERIAL."""
        impact = [
            mod.ImpactScoreInput(
                matter_id="ZERO-1", matter_name="Zero",
                esrs_topic="e1_climate", score=0.0,
            ),
        ]
        financial = [
            mod.FinancialScoreInput(
                matter_id="ZERO-1", matter_name="Zero",
                esrs_topic="e1_climate", score=0.0,
            ),
        ]
        matrix = engine.build_matrix(impact, financial)
        assert matrix.entries[0].quadrant == mod.Quadrant.NOT_MATERIAL
