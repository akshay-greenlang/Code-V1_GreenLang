# -*- coding: utf-8 -*-
"""
Unit tests for TCFD Gap Analysis Engine.

Tests maturity assessment, pillar scoring, gap identification, peer
benchmarking, action plan generation, and progress tracking
with 22+ test functions.

Author: GL-TestEngineer
Date: March 2026
"""

from decimal import Decimal

import pytest

from services.config import (
    MaturityLevel,
    TCFDPillar,
    MATURITY_SCORES,
)
from services.models import (
    GapAssessment,
    _new_id,
)


# ===========================================================================
# Maturity Assessment
# ===========================================================================

class TestMaturityAssessment:
    """Test organizational maturity assessment."""

    def test_overall_maturity(self, sample_gap_assessment):
        assert sample_gap_assessment.overall_maturity == MaturityLevel.DEVELOPING

    def test_initial_maturity(self):
        gap = GapAssessment(
            org_id=_new_id(),
            overall_maturity=MaturityLevel.INITIAL,
        )
        assert gap.overall_maturity == MaturityLevel.INITIAL

    def test_optimized_maturity(self):
        gap = GapAssessment(
            org_id=_new_id(),
            overall_maturity=MaturityLevel.OPTIMIZED,
            pillar_scores={
                "governance": 5, "strategy": 5,
                "risk_management": 5, "metrics_targets": 5,
            },
        )
        assert gap.overall_maturity == MaturityLevel.OPTIMIZED

    @pytest.mark.parametrize("level", list(MaturityLevel))
    def test_all_maturity_levels(self, level):
        gap = GapAssessment(
            org_id=_new_id(),
            overall_maturity=level,
        )
        assert gap.overall_maturity == level


# ===========================================================================
# Pillar Scoring
# ===========================================================================

class TestPillarScoring:
    """Test per-pillar maturity scoring."""

    def test_pillar_scores_present(self, sample_gap_assessment):
        scores = sample_gap_assessment.pillar_scores
        assert "governance" in scores
        assert "strategy" in scores
        assert "risk_management" in scores
        assert "metrics_targets" in scores

    def test_pillar_score_range(self, sample_gap_assessment):
        for pillar, score in sample_gap_assessment.pillar_scores.items():
            assert 1 <= score <= 5, f"Pillar {pillar} score {score} out of range"

    def test_governance_score(self, sample_gap_assessment):
        assert sample_gap_assessment.pillar_scores["governance"] == 3

    def test_strategy_score(self, sample_gap_assessment):
        assert sample_gap_assessment.pillar_scores["strategy"] == 2

    def test_average_pillar_score(self, sample_gap_assessment):
        scores = sample_gap_assessment.pillar_scores.values()
        avg = sum(scores) / len(scores)
        assert avg == 2.5  # (3+2+3+2)/4


# ===========================================================================
# Gap Identification
# ===========================================================================

class TestGapIdentification:
    """Test gap identification."""

    def test_gaps_identified(self, sample_gap_assessment):
        assert len(sample_gap_assessment.gaps) == 2

    def test_gap_content(self, sample_gap_assessment):
        gap = sample_gap_assessment.gaps[0]
        assert "pillar" in gap
        assert "disclosure" in gap
        assert "gap" in gap

    def test_scenario_analysis_gap(self, sample_gap_assessment):
        scenario_gap = next(
            (g for g in sample_gap_assessment.gaps if g["disclosure"] == "str_c"),
            None,
        )
        assert scenario_gap is not None
        assert "scenario" in scenario_gap["gap"].lower()

    def test_no_gaps(self):
        gap = GapAssessment(
            org_id=_new_id(),
            overall_maturity=MaturityLevel.OPTIMIZED,
            gaps=[],
        )
        assert len(gap.gaps) == 0


# ===========================================================================
# Peer Benchmarking
# ===========================================================================

class TestPeerBenchmarking:
    """Test peer benchmarking percentile."""

    def test_peer_percentile(self, sample_gap_assessment):
        assert sample_gap_assessment.peer_benchmark_percentile == 35

    def test_high_percentile(self):
        gap = GapAssessment(
            org_id=_new_id(),
            peer_benchmark_percentile=90,
        )
        assert gap.peer_benchmark_percentile == 90

    def test_no_benchmark(self):
        gap = GapAssessment(
            org_id=_new_id(),
        )
        assert gap.peer_benchmark_percentile is None


# ===========================================================================
# Action Plan Generation
# ===========================================================================

class TestActionPlanGeneration:
    """Test action plan generation."""

    def test_actions_defined(self, sample_gap_assessment):
        assert len(sample_gap_assessment.actions) == 2

    def test_action_content(self, sample_gap_assessment):
        action = sample_gap_assessment.actions[0]
        assert "action" in action
        assert "priority" in action
        assert "timeline" in action

    def test_action_priority_high(self, sample_gap_assessment):
        for action in sample_gap_assessment.actions:
            assert action["priority"] == "high"

    def test_empty_action_plan(self):
        gap = GapAssessment(
            org_id=_new_id(),
            actions=[],
        )
        assert len(gap.actions) == 0


# ===========================================================================
# Progress Tracking
# ===========================================================================

class TestProgressTracking:
    """Test gap assessment progress tracking."""

    def test_gap_assessment_provenance(self, sample_gap_assessment):
        assert len(sample_gap_assessment.provenance_hash) == 64

    def test_gap_assessment_date(self, sample_gap_assessment):
        assert sample_gap_assessment.assessment_date is not None

    def test_provenance_deterministic(self):
        org_id = _new_id()
        from datetime import date
        assessment_date = date(2025, 12, 31)
        g1 = GapAssessment(
            org_id=org_id,
            assessment_date=assessment_date,
            overall_maturity=MaturityLevel.DEVELOPING,
        )
        g2 = GapAssessment(
            org_id=org_id,
            assessment_date=assessment_date,
            overall_maturity=MaturityLevel.DEVELOPING,
        )
        assert g1.provenance_hash == g2.provenance_hash
