# -*- coding: utf-8 -*-
"""
PACK-044 Test Suite - Gap Analysis Engine Tests
=================================================

Tests GapAnalysisEngine: gap identification, methodology tier scoring,
Pareto analysis, improvement prioritisation, and roadmap generation.

Target: 50+ test cases.
"""

from decimal import Decimal

import pytest

from conftest import _load_engine

# ---------------------------------------------------------------------------
# Dynamic imports
# ---------------------------------------------------------------------------

_mod = _load_engine("gap_analysis")

GapAnalysisEngine = _mod.GapAnalysisEngine
GapAnalysisResult = _mod.GapAnalysisResult
DataGap = _mod.DataGap
MethodologyGap = _mod.MethodologyGap
ImprovementRecommendation = _mod.ImprovementRecommendation
ImprovementRoadmap = _mod.ImprovementRoadmap
MethodologyTier = _mod.MethodologyTier
GapAnalysisConfig = _mod.GapAnalysisConfig
SourceCategoryAssessment = _mod.SourceCategoryAssessment
TIER_SCORES = _mod.TIER_SCORES


# ===================================================================
# Fixtures
# ===================================================================


@pytest.fixture
def engine():
    """Create a fresh GapAnalysisEngine."""
    return GapAnalysisEngine()


@pytest.fixture
def categories_high_quality():
    """Categories with high methodology tiers and data quality."""
    return [
        SourceCategoryAssessment(
            category_name="stationary_combustion",
            scope=1,
            emissions_tco2e=Decimal("12800"),
            methodology_tier=MethodologyTier.TIER_3,
            has_activity_data=True,
            has_emission_factors=True,
            temporal_coverage_pct=Decimal("100"),
            documentation_complete=True,
            data_source="Continuous metering + supplier EFs",
        ),
        SourceCategoryAssessment(
            category_name="purchased_electricity",
            scope=2,
            emissions_tco2e=Decimal("14700"),
            methodology_tier=MethodologyTier.TIER_3,
            has_activity_data=True,
            has_emission_factors=True,
            temporal_coverage_pct=Decimal("100"),
            documentation_complete=True,
            data_source="Smart meters + market-based EFs",
        ),
    ]


@pytest.fixture
def categories_mixed_quality():
    """Categories with mixed tiers and quality levels."""
    return [
        SourceCategoryAssessment(
            category_name="stationary_combustion",
            scope=1,
            emissions_tco2e=Decimal("12800"),
            methodology_tier=MethodologyTier.TIER_2,
            has_activity_data=True,
            has_emission_factors=True,
            data_source="Fuel invoices + DEFRA EFs",
        ),
        SourceCategoryAssessment(
            category_name="mobile_combustion",
            scope=1,
            emissions_tco2e=Decimal("2500"),
            methodology_tier=MethodologyTier.TIER_2,
            has_activity_data=True,
            has_emission_factors=True,
            data_source="Fleet fuel cards",
        ),
        SourceCategoryAssessment(
            category_name="process_emissions",
            scope=1,
            emissions_tco2e=Decimal("4200"),
            methodology_tier=MethodologyTier.TIER_1,
            has_activity_data=True,
            has_emission_factors=True,
            data_source="Industry average EFs",
        ),
        SourceCategoryAssessment(
            category_name="fugitive_emissions",
            scope=1,
            emissions_tco2e=Decimal("350"),
            methodology_tier=MethodologyTier.TIER_1,
            has_activity_data=True,
            has_emission_factors=True,
            data_source="Estimated from equipment type",
        ),
        SourceCategoryAssessment(
            category_name="purchased_electricity",
            scope=2,
            emissions_tco2e=Decimal("14700"),
            methodology_tier=MethodologyTier.TIER_2,
            has_activity_data=True,
            has_emission_factors=True,
            data_source="Utility bills + IEA grid factors",
        ),
        SourceCategoryAssessment(
            category_name="purchased_goods",
            scope=3,
            emissions_tco2e=Decimal("8000"),
            methodology_tier=MethodologyTier.TIER_1,
            has_activity_data=True,
            has_emission_factors=True,
            data_source="Spend-based screening",
        ),
        SourceCategoryAssessment(
            category_name="employee_commuting",
            scope=3,
            emissions_tco2e=Decimal("0"),
            methodology_tier=MethodologyTier.MISSING,
            has_activity_data=False,
            has_emission_factors=False,
            data_source="",
        ),
    ]


# ===================================================================
# Basic Analysis Tests
# ===================================================================


class TestBasicAnalysis:
    """Tests for basic gap analysis execution."""

    def test_analyse_returns_result(self, engine, categories_mixed_quality):
        result = engine.analyse(categories=categories_mixed_quality)
        assert isinstance(result, GapAnalysisResult)

    def test_categories_assessed_count(self, engine, categories_mixed_quality):
        result = engine.analyse(categories=categories_mixed_quality)
        assert result.categories_assessed == 7

    def test_total_emissions_stored(self, engine, categories_mixed_quality):
        result = engine.analyse(categories=categories_mixed_quality)
        assert result.total_emissions_tco2e > 0

    def test_provenance_hash(self, engine, categories_mixed_quality):
        result = engine.analyse(categories=categories_mixed_quality)
        assert len(result.provenance_hash) == 64

    def test_processing_time_positive(self, engine, categories_mixed_quality):
        result = engine.analyse(categories=categories_mixed_quality)
        assert result.processing_time_ms >= 0


# ===================================================================
# Gap Identification Tests
# ===================================================================


class TestGapIdentification:
    """Tests for gap detection."""

    def test_missing_data_gap_detected(self, engine, categories_mixed_quality):
        result = engine.analyse(categories_mixed_quality)
        missing = [g for g in result.data_gaps
                   if g.category_name == "employee_commuting"]
        assert len(missing) >= 1

    def test_low_tier_gaps_detected(self, engine, categories_mixed_quality):
        result = engine.analyse(categories_mixed_quality)
        low_tier = [g for g in result.data_gaps
                    if "tier" in g.gap_category.lower()
                    or g.current_tier in ("tier_1", "estimated", "missing")]
        assert len(low_tier) >= 1

    def test_high_quality_no_critical_gaps(self, engine, categories_high_quality):
        result = engine.analyse(categories_high_quality)
        assert result.critical_gaps == 0

    def test_gaps_have_severity(self, engine, categories_mixed_quality):
        result = engine.analyse(categories_mixed_quality)
        for gap in result.data_gaps:
            assert gap.severity in ("critical", "high", "medium", "low", "info")

    def test_gaps_have_scope(self, engine, categories_mixed_quality):
        result = engine.analyse(categories_mixed_quality)
        for gap in result.data_gaps:
            assert gap.scope in (1, 2, 3)


# ===================================================================
# Methodology Gap Tests
# ===================================================================


class TestMethodologyGaps:
    """Tests for methodology tier gap analysis."""

    def test_methodology_gaps_populated(self, engine, categories_mixed_quality):
        result = engine.analyse(categories_mixed_quality)
        assert len(result.methodology_gaps) >= 1

    def test_tier_gap_calculation(self, engine, categories_mixed_quality):
        result = engine.analyse(
            categories_mixed_quality,
            config=GapAnalysisConfig(target_tier=MethodologyTier.TIER_2),
        )
        for mg in result.methodology_gaps:
            assert mg.tier_gap >= 0

    def test_effort_estimate_populated(self, engine, categories_mixed_quality):
        result = engine.analyse(categories_mixed_quality)
        for mg in result.methodology_gaps:
            assert mg.estimated_effort in ("low", "medium", "high", "very_high")


# ===================================================================
# Tier Scoring Tests
# ===================================================================


class TestTierScoring:
    """Tests for methodology tier score constants."""

    def test_tier_3_highest(self):
        assert TIER_SCORES[MethodologyTier.TIER_3.value] == 100

    def test_tier_2_score(self):
        assert TIER_SCORES[MethodologyTier.TIER_2.value] == 70

    def test_tier_1_score(self):
        assert TIER_SCORES[MethodologyTier.TIER_1.value] == 40

    def test_estimated_score(self):
        assert TIER_SCORES[MethodologyTier.ESTIMATED.value] == 15

    def test_missing_score(self):
        assert TIER_SCORES[MethodologyTier.MISSING.value] == 0

    def test_scores_monotonically_decreasing(self):
        tiers = [
            MethodologyTier.TIER_3,
            MethodologyTier.TIER_2,
            MethodologyTier.TIER_1,
            MethodologyTier.ESTIMATED,
            MethodologyTier.MISSING,
        ]
        scores = [TIER_SCORES[t.value] for t in tiers]
        for i in range(len(scores) - 1):
            assert scores[i] > scores[i + 1]


# ===================================================================
# Quality Score Tests
# ===================================================================


class TestQualityScore:
    """Tests for overall quality score calculation."""

    def test_quality_score_range(self, engine, categories_mixed_quality):
        result = engine.analyse(categories_mixed_quality)
        assert 0 <= result.overall_quality_score <= 100

    def test_high_quality_high_score(self, engine, categories_high_quality):
        result = engine.analyse(categories_high_quality)
        assert result.overall_quality_score >= 80

    def test_mixed_quality_moderate_score(self, engine, categories_mixed_quality):
        result = engine.analyse(categories_mixed_quality)
        assert result.overall_quality_score < 90


# ===================================================================
# Improvement Recommendations Tests
# ===================================================================


class TestRecommendations:
    """Tests for prioritised improvement recommendations."""

    def test_recommendations_generated(self, engine, categories_mixed_quality):
        result = engine.analyse(categories_mixed_quality)
        assert len(result.recommendations) >= 1

    def test_recommendations_prioritised(self, engine, categories_mixed_quality):
        result = engine.analyse(categories_mixed_quality)
        if len(result.recommendations) >= 2:
            ranks = [r.priority_rank for r in result.recommendations]
            assert ranks == sorted(ranks)

    def test_recommendation_has_action(self, engine, categories_mixed_quality):
        result = engine.analyse(categories_mixed_quality)
        for rec in result.recommendations:
            assert rec.action != ""

    def test_recommendation_roi_score(self, engine, categories_mixed_quality):
        result = engine.analyse(categories_mixed_quality)
        for rec in result.recommendations:
            assert rec.roi_score >= 0

    def test_max_recommendations_respected(self, engine, categories_mixed_quality):
        config = GapAnalysisConfig(max_recommendations=3)
        result = engine.analyse(categories_mixed_quality, config=config)
        assert len(result.recommendations) <= 3


# ===================================================================
# Improvement Roadmap Tests
# ===================================================================


class TestRoadmap:
    """Tests for phased improvement roadmap."""

    def test_roadmap_populated(self, engine, categories_mixed_quality):
        result = engine.analyse(categories_mixed_quality)
        assert result.roadmap is not None

    def test_roadmap_has_phases(self, engine, categories_mixed_quality):
        result = engine.analyse(categories_mixed_quality)
        total_actions = (
            len(result.roadmap.phase_1_quick_wins)
            + len(result.roadmap.phase_2_medium_term)
            + len(result.roadmap.phase_3_long_term)
        )
        assert total_actions >= 1

    def test_roadmap_projected_score(self, engine, categories_mixed_quality):
        result = engine.analyse(categories_mixed_quality)
        assert result.roadmap.projected_score_after >= result.roadmap.current_overall_score


# ===================================================================
# Pareto Analysis Tests
# ===================================================================


class TestParetoAnalysis:
    """Tests for Pareto-based prioritisation."""

    def test_pareto_coverage_populated(self, engine, categories_mixed_quality):
        result = engine.analyse(categories_mixed_quality)
        assert result.pareto_coverage_pct >= 0


# ===================================================================
# Configuration Tests
# ===================================================================


class TestConfiguration:
    """Tests for gap analysis configuration."""

    def test_custom_target_tier(self, engine, categories_mixed_quality):
        config = GapAnalysisConfig(target_tier=MethodologyTier.TIER_3)
        result = engine.analyse(categories_mixed_quality, config=config)
        assert result.gaps_identified >= 1

    def test_exclude_scope3(self, engine, categories_mixed_quality):
        config = GapAnalysisConfig(include_scope3=False)
        result = engine.analyse(categories_mixed_quality, config=config)
        scope3_gaps = [g for g in result.data_gaps if g.scope == 3]
        assert len(scope3_gaps) == 0

    def test_custom_materiality_threshold(self, engine, categories_mixed_quality):
        config = GapAnalysisConfig(materiality_threshold_pct=Decimal("5.0"))
        result = engine.analyse(categories_mixed_quality, config=config)
        assert result is not None


# ===================================================================
# Edge Cases Tests
# ===================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_categories_raises(self, engine):
        with pytest.raises(ValueError):
            engine.analyse([])

    def test_single_category(self, engine):
        result = engine.analyse([
            SourceCategoryAssessment(
                category_name="stationary",
                scope=1,
                emissions_tco2e=Decimal("10000"),
                methodology_tier=MethodologyTier.TIER_2,
                has_activity_data=True,
                has_emission_factors=True,
                data_source="Invoices",
            ),
        ])
        assert result.categories_assessed == 1


# ===================================================================
# Model Tests
# ===================================================================


class TestModels:
    """Tests for Pydantic model defaults."""

    @pytest.mark.parametrize("tier", list(MethodologyTier))
    def test_methodology_tiers(self, tier):
        assert tier.value in TIER_SCORES

    def test_data_gap_defaults(self):
        dg = DataGap()
        assert dg.severity == "low"

    def test_methodology_gap_defaults(self):
        mg = MethodologyGap()
        assert mg.estimated_effort == "medium"

    def test_improvement_recommendation_defaults(self):
        ir = ImprovementRecommendation()
        assert ir.priority_rank == 0

    def test_result_defaults(self):
        r = GapAnalysisResult()
        assert r.gaps_identified == 0
        assert r.overall_quality_score == 0.0
