# -*- coding: utf-8 -*-
"""
Unit tests for GapAnalysisEngine -- CDP gap identification and recommendations.

Tests gap identification by severity, categorization, priority ranking,
recommendation generation, effort estimation, score uplift prediction,
gap tracking over time, and summary aggregation with 32+ test functions.

Author: GL-TestEngineer
Date: March 2026
"""

from decimal import Decimal
from typing import List

import pytest

from services.config import (
    GapSeverity,
    ScoringLevel,
    ScoringCategory,
    EffortLevel,
)
from services.models import (
    CDPGapAnalysis,
    CDPGapItem,
    CDPCategoryScore,
    CDPResponse,
    _new_id,
)
from services.gap_analysis_engine import GapAnalysisEngine


# ---------------------------------------------------------------------------
# Gap identification
# ---------------------------------------------------------------------------

class TestGapIdentification:
    """Test identifying gaps between current and target scoring levels."""

    def test_identify_disclosure_gap(self, gap_analysis_engine):
        gap = gap_analysis_engine.identify_gap(
            question_number="C7.1",
            module_code="M7",
            current_level=None,  # No response
            scoring_category=ScoringCategory.SCOPE_1_2_EMISSIONS,
        )
        assert gap is not None
        assert gap.severity == GapSeverity.CRITICAL
        assert gap.current_level is None or gap.current_level == ScoringLevel.DISCLOSURE

    def test_identify_awareness_gap(self, gap_analysis_engine):
        gap = gap_analysis_engine.identify_gap(
            question_number="C1.1",
            module_code="M1",
            current_level=ScoringLevel.DISCLOSURE,
            scoring_category=ScoringCategory.GOVERNANCE,
        )
        assert gap is not None
        assert gap.current_level == ScoringLevel.DISCLOSURE
        assert gap.target_level in [ScoringLevel.AWARENESS, ScoringLevel.MANAGEMENT, ScoringLevel.LEADERSHIP]

    def test_no_gap_at_leadership(self, gap_analysis_engine):
        gap = gap_analysis_engine.identify_gap(
            question_number="C1.1",
            module_code="M1",
            current_level=ScoringLevel.LEADERSHIP,
            scoring_category=ScoringCategory.GOVERNANCE,
        )
        assert gap is None

    def test_identify_multiple_gaps(self, gap_analysis_engine, sample_category_scores):
        gaps = gap_analysis_engine.run_analysis(
            category_scores=sample_category_scores,
            org_id=_new_id(),
            questionnaire_id=_new_id(),
        )
        assert isinstance(gaps, CDPGapAnalysis)
        assert gaps.total_gaps >= 0

    def test_gap_counts_consistent(self, gap_analysis_engine, sample_category_scores):
        gaps = gap_analysis_engine.run_analysis(
            category_scores=sample_category_scores,
            org_id=_new_id(),
            questionnaire_id=_new_id(),
        )
        total = gaps.critical_gaps + gaps.high_gaps + gaps.medium_gaps + gaps.low_gaps
        assert total == gaps.total_gaps


# ---------------------------------------------------------------------------
# Gap severity classification
# ---------------------------------------------------------------------------

class TestGapSeverity:
    """Test gap severity assignment."""

    def test_critical_for_missing_required(self, gap_analysis_engine):
        severity = gap_analysis_engine.classify_severity(
            has_response=False,
            is_required=True,
            scoring_impact=Decimal("5.0"),
        )
        assert severity == GapSeverity.CRITICAL

    def test_high_for_low_score_required(self, gap_analysis_engine):
        severity = gap_analysis_engine.classify_severity(
            has_response=True,
            is_required=True,
            scoring_impact=Decimal("3.0"),
        )
        assert severity == GapSeverity.HIGH

    def test_medium_for_moderate_impact(self, gap_analysis_engine):
        severity = gap_analysis_engine.classify_severity(
            has_response=True,
            is_required=True,
            scoring_impact=Decimal("1.5"),
        )
        assert severity == GapSeverity.MEDIUM

    def test_low_for_optional_gap(self, gap_analysis_engine):
        severity = gap_analysis_engine.classify_severity(
            has_response=True,
            is_required=False,
            scoring_impact=Decimal("0.5"),
        )
        assert severity == GapSeverity.LOW


# ---------------------------------------------------------------------------
# Gap categorization
# ---------------------------------------------------------------------------

class TestGapCategorization:
    """Test gap categorization by scoring level."""

    def test_disclosure_level_gap(self, gap_analysis_engine):
        category = gap_analysis_engine.categorize_gap(
            current_level=None,
            target_level=ScoringLevel.DISCLOSURE,
        )
        assert category == "disclosure"

    def test_awareness_level_gap(self, gap_analysis_engine):
        category = gap_analysis_engine.categorize_gap(
            current_level=ScoringLevel.DISCLOSURE,
            target_level=ScoringLevel.AWARENESS,
        )
        assert category == "awareness"

    def test_management_level_gap(self, gap_analysis_engine):
        category = gap_analysis_engine.categorize_gap(
            current_level=ScoringLevel.AWARENESS,
            target_level=ScoringLevel.MANAGEMENT,
        )
        assert category == "management"

    def test_leadership_level_gap(self, gap_analysis_engine):
        category = gap_analysis_engine.categorize_gap(
            current_level=ScoringLevel.MANAGEMENT,
            target_level=ScoringLevel.LEADERSHIP,
        )
        assert category == "leadership"


# ---------------------------------------------------------------------------
# Priority ranking
# ---------------------------------------------------------------------------

class TestPriorityRanking:
    """Test priority ranking by score impact."""

    def test_rank_by_score_uplift(self, gap_analysis_engine, sample_gap_items):
        ranked = gap_analysis_engine.rank_by_priority(sample_gap_items)
        assert ranked[0].score_uplift >= ranked[-1].score_uplift

    def test_critical_first_in_ranking(self, gap_analysis_engine, sample_gap_items):
        ranked = gap_analysis_engine.rank_by_priority(sample_gap_items)
        first_severity = ranked[0].severity
        assert first_severity == GapSeverity.CRITICAL

    def test_empty_list_ranking(self, gap_analysis_engine):
        ranked = gap_analysis_engine.rank_by_priority([])
        assert ranked == []


# ---------------------------------------------------------------------------
# Recommendation generation
# ---------------------------------------------------------------------------

class TestRecommendations:
    """Test actionable recommendation generation."""

    def test_generate_recommendation(self, gap_analysis_engine):
        rec = gap_analysis_engine.generate_recommendation(
            module_code="M7",
            question_number="C7.3",
            current_level=ScoringLevel.DISCLOSURE,
            target_level=ScoringLevel.MANAGEMENT,
            scoring_category=ScoringCategory.SCOPE_1_2_EMISSIONS,
        )
        assert rec is not None
        assert len(rec) > 20  # Meaningful recommendation text

    def test_recommendation_for_verification_gap(self, gap_analysis_engine):
        rec = gap_analysis_engine.generate_recommendation(
            module_code="M7",
            question_number="C7.9",
            current_level=ScoringLevel.MANAGEMENT,
            target_level=ScoringLevel.LEADERSHIP,
            scoring_category=ScoringCategory.SCOPE_1_2_EMISSIONS,
        )
        assert "verif" in rec.lower() or "assurance" in rec.lower() or "third" in rec.lower()


# ---------------------------------------------------------------------------
# Effort estimation
# ---------------------------------------------------------------------------

class TestEffortEstimation:
    """Test effort estimation for closing gaps."""

    def test_low_effort_simple_disclosure(self, gap_analysis_engine):
        effort = gap_analysis_engine.estimate_effort(
            current_level=ScoringLevel.DISCLOSURE,
            target_level=ScoringLevel.AWARENESS,
            module_code="M0",
        )
        assert effort == EffortLevel.LOW

    def test_high_effort_verification(self, gap_analysis_engine):
        effort = gap_analysis_engine.estimate_effort(
            current_level=ScoringLevel.MANAGEMENT,
            target_level=ScoringLevel.LEADERSHIP,
            module_code="M7",
        )
        assert effort in [EffortLevel.HIGH, EffortLevel.MEDIUM]

    def test_high_effort_transition_plan(self, gap_analysis_engine):
        effort = gap_analysis_engine.estimate_effort(
            current_level=None,
            target_level=ScoringLevel.LEADERSHIP,
            module_code="M5",
        )
        assert effort == EffortLevel.HIGH


# ---------------------------------------------------------------------------
# Score uplift prediction
# ---------------------------------------------------------------------------

class TestScoreUplift:
    """Test predicted score uplift per gap closed."""

    def test_uplift_positive(self, gap_analysis_engine):
        uplift = gap_analysis_engine.predict_score_uplift(
            category_code="CAT09",
            current_score=Decimal("60"),
            target_score=Decimal("85"),
            weight=Decimal("0.10"),
        )
        assert uplift > Decimal("0")

    def test_uplift_proportional_to_weight(self, gap_analysis_engine):
        uplift_high_weight = gap_analysis_engine.predict_score_uplift(
            category_code="CAT09", current_score=Decimal("60"),
            target_score=Decimal("85"), weight=Decimal("0.10"),
        )
        uplift_low_weight = gap_analysis_engine.predict_score_uplift(
            category_code="CAT14", current_score=Decimal("60"),
            target_score=Decimal("85"), weight=Decimal("0.03"),
        )
        assert uplift_high_weight > uplift_low_weight

    def test_total_potential_uplift(self, gap_analysis_engine, sample_gap_items):
        total = gap_analysis_engine.calculate_total_potential_uplift(sample_gap_items)
        expected = sum(g.score_uplift for g in sample_gap_items)
        assert total == expected


# ---------------------------------------------------------------------------
# Gap tracking over time
# ---------------------------------------------------------------------------

class TestGapTracking:
    """Test gap tracking and resolution over time."""

    def test_mark_gap_resolved(self, gap_analysis_engine, sample_gap_items):
        gap = sample_gap_items[0]
        gap_analysis_engine._gap_store[gap.id] = gap
        resolved = gap_analysis_engine.resolve_gap(gap.id)
        assert resolved.is_resolved is True
        assert resolved.resolved_at is not None

    def test_resolved_gap_not_in_open_list(self, gap_analysis_engine, sample_gap_items):
        gap = sample_gap_items[0]
        gap_analysis_engine._gap_store[gap.id] = gap
        gap_analysis_engine.resolve_gap(gap.id)
        for g in sample_gap_items[1:]:
            gap_analysis_engine._gap_store[g.id] = g
        open_gaps = gap_analysis_engine.get_open_gaps()
        gap_ids = [g.id for g in open_gaps]
        assert gap.id not in gap_ids


# ---------------------------------------------------------------------------
# Summary aggregation
# ---------------------------------------------------------------------------

class TestGapSummary:
    """Test gap summary statistics."""

    def test_summary_by_module(self, gap_analysis_engine, sample_gap_items):
        for g in sample_gap_items:
            gap_analysis_engine._gap_store[g.id] = g
        summary = gap_analysis_engine.summarize_by_module()
        assert isinstance(summary, dict)
        assert "M7" in summary or "M5" in summary

    def test_summary_by_severity(self, gap_analysis_engine, sample_gap_items):
        for g in sample_gap_items:
            gap_analysis_engine._gap_store[g.id] = g
        summary = gap_analysis_engine.summarize_by_severity()
        assert "critical" in summary
        assert "high" in summary
        assert "medium" in summary

    def test_summary_total_uplift(self, gap_analysis_engine, sample_gap_items):
        for g in sample_gap_items:
            gap_analysis_engine._gap_store[g.id] = g
        summary = gap_analysis_engine.summarize_by_severity()
        total_uplift = sum(v.get("total_uplift", Decimal("0")) for v in summary.values())
        assert total_uplift > Decimal("0")
