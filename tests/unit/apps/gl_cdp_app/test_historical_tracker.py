# -*- coding: utf-8 -*-
"""
Unit tests for HistoricalTracker -- year-over-year comparison and trends.

Tests year-over-year comparison, score progression, category trend analysis,
response carry-forward, and change log generation with 22+ test functions.

Author: GL-TestEngineer
Date: March 2026
"""

from decimal import Decimal
from datetime import datetime

import pytest

from services.config import ScoreBand
from services.models import (
    CDPScoringResult,
    CDPCategoryScore,
    CDPResponse,
    _new_id,
)
from services.historical_tracker import HistoricalTracker


# ---------------------------------------------------------------------------
# Year-over-year comparison
# ---------------------------------------------------------------------------

class TestYearOverYearComparison:
    """Test YoY score comparison."""

    def test_compare_two_years(self, historical_tracker):
        current = CDPScoringResult(
            questionnaire_id=_new_id(),
            org_id=_new_id(),
            overall_score=Decimal("72.5"),
            score_band="A-",
        )
        previous = CDPScoringResult(
            questionnaire_id=_new_id(),
            org_id=current.org_id,
            overall_score=Decimal("58.0"),
            score_band="B-",
        )
        comparison = historical_tracker.compare_years(
            current_result=current,
            previous_result=previous,
        )
        assert comparison["score_delta"] == Decimal("14.5")
        assert comparison["current_band"] == "A-"
        assert comparison["previous_band"] == "B-"
        assert comparison["improved"] is True

    def test_compare_decline(self, historical_tracker):
        current = CDPScoringResult(
            questionnaire_id=_new_id(), org_id=_new_id(),
            overall_score=Decimal("45.0"), score_band="C",
        )
        previous = CDPScoringResult(
            questionnaire_id=_new_id(), org_id=current.org_id,
            overall_score=Decimal("60.0"), score_band="B",
        )
        comparison = historical_tracker.compare_years(current, previous)
        assert comparison["score_delta"] == Decimal("-15.0")
        assert comparison["improved"] is False

    def test_compare_no_change(self, historical_tracker):
        org_id = _new_id()
        current = CDPScoringResult(
            questionnaire_id=_new_id(), org_id=org_id,
            overall_score=Decimal("55.0"), score_band="B-",
        )
        previous = CDPScoringResult(
            questionnaire_id=_new_id(), org_id=org_id,
            overall_score=Decimal("55.0"), score_band="B-",
        )
        comparison = historical_tracker.compare_years(current, previous)
        assert comparison["score_delta"] == Decimal("0")

    def test_compare_no_previous(self, historical_tracker):
        current = CDPScoringResult(
            questionnaire_id=_new_id(), org_id=_new_id(),
            overall_score=Decimal("55.0"), score_band="B-",
        )
        comparison = historical_tracker.compare_years(current, None)
        assert comparison["is_first_year"] is True
        assert comparison["previous_band"] is None


# ---------------------------------------------------------------------------
# Score progression
# ---------------------------------------------------------------------------

class TestScoreProgression:
    """Test multi-year score progression."""

    def test_score_progression_multiple_years(self, historical_tracker):
        org_id = _new_id()
        history = [
            {"year": 2021, "score": Decimal("35.0"), "band": "C-"},
            {"year": 2022, "score": Decimal("42.0"), "band": "C"},
            {"year": 2023, "score": Decimal("55.0"), "band": "B-"},
            {"year": 2024, "score": Decimal("65.0"), "band": "B"},
            {"year": 2025, "score": Decimal("72.5"), "band": "A-"},
        ]
        progression = historical_tracker.get_score_progression(org_id, history)
        assert len(progression) == 5
        assert progression[0]["year"] == 2021
        assert progression[-1]["year"] == 2025

    def test_progression_improvement_rate(self, historical_tracker):
        org_id = _new_id()
        history = [
            {"year": 2022, "score": Decimal("40.0"), "band": "C"},
            {"year": 2025, "score": Decimal("70.0"), "band": "A-"},
        ]
        progression = historical_tracker.get_score_progression(org_id, history)
        rate = historical_tracker.calculate_improvement_rate(org_id, history)
        assert rate > Decimal("0")

    def test_progression_with_band_transitions(self, historical_tracker):
        org_id = _new_id()
        history = [
            {"year": 2023, "score": Decimal("55.0"), "band": "B-"},
            {"year": 2024, "score": Decimal("62.0"), "band": "B"},
            {"year": 2025, "score": Decimal("71.0"), "band": "A-"},
        ]
        transitions = historical_tracker.get_band_transitions(org_id, history)
        assert len(transitions) == 2
        assert transitions[0]["from"] == "B-"
        assert transitions[0]["to"] == "B"


# ---------------------------------------------------------------------------
# Category trend analysis
# ---------------------------------------------------------------------------

class TestCategoryTrends:
    """Test category-level trend analysis."""

    def test_category_trend_improving(self, historical_tracker):
        org_id = _new_id()
        category_history = [
            {"year": 2023, "category_code": "CAT09", "score": Decimal("60")},
            {"year": 2024, "category_code": "CAT09", "score": Decimal("70")},
            {"year": 2025, "category_code": "CAT09", "score": Decimal("85")},
        ]
        trend = historical_tracker.get_category_trend(org_id, "CAT09", category_history)
        assert trend["direction"] == "improving"
        assert trend["avg_annual_change"] > Decimal("0")

    def test_category_trend_declining(self, historical_tracker):
        org_id = _new_id()
        category_history = [
            {"year": 2023, "category_code": "CAT12", "score": Decimal("70")},
            {"year": 2024, "category_code": "CAT12", "score": Decimal("55")},
            {"year": 2025, "category_code": "CAT12", "score": Decimal("40")},
        ]
        trend = historical_tracker.get_category_trend(org_id, "CAT12", category_history)
        assert trend["direction"] == "declining"

    def test_category_trend_stable(self, historical_tracker):
        org_id = _new_id()
        category_history = [
            {"year": 2023, "category_code": "CAT01", "score": Decimal("65")},
            {"year": 2024, "category_code": "CAT01", "score": Decimal("66")},
            {"year": 2025, "category_code": "CAT01", "score": Decimal("65")},
        ]
        trend = historical_tracker.get_category_trend(org_id, "CAT01", category_history)
        assert trend["direction"] == "stable"

    def test_all_categories_trend(self, historical_tracker):
        org_id = _new_id()
        all_history = {}
        for i in range(1, 18):
            cat = f"CAT{i:02d}"
            all_history[cat] = [
                {"year": y, "category_code": cat, "score": Decimal(str(40 + i + (y - 2023) * 3))}
                for y in [2023, 2024, 2025]
            ]
        trends = historical_tracker.get_all_category_trends(org_id, all_history)
        assert len(trends) == 17


# ---------------------------------------------------------------------------
# Response carry-forward
# ---------------------------------------------------------------------------

class TestResponseCarryForward:
    """Test response carry-forward from previous year."""

    def test_carry_forward_responses(self, historical_tracker):
        previous_responses = [
            CDPResponse(
                question_id=_new_id(),
                questionnaire_id=_new_id(),
                org_id=_new_id(),
                response_content={"answer": f"prev_{i}"},
                response_text=f"Previous response {i}",
                response_status="submitted",
            )
            for i in range(10)
        ]
        carried = historical_tracker.carry_forward(
            previous_responses=previous_responses,
            target_questionnaire_id=_new_id(),
        )
        assert len(carried) == 10
        for r in carried:
            assert r.response_status.value == "draft" or r.response_status == "draft"

    def test_carry_forward_preserves_content(self, historical_tracker):
        original = CDPResponse(
            question_id=_new_id(),
            questionnaire_id=_new_id(),
            org_id=_new_id(),
            response_content={"scope1": 5000, "scope2": 3000},
            response_text="Original text",
            response_status="submitted",
        )
        carried = historical_tracker.carry_forward(
            previous_responses=[original],
            target_questionnaire_id=_new_id(),
        )
        assert carried[0].response_content == {"scope1": 5000, "scope2": 3000}

    def test_carry_forward_empty_returns_empty(self, historical_tracker):
        carried = historical_tracker.carry_forward(
            previous_responses=[],
            target_questionnaire_id=_new_id(),
        )
        assert carried == []


# ---------------------------------------------------------------------------
# Change log
# ---------------------------------------------------------------------------

class TestChangeLog:
    """Test change log generation between submissions."""

    def test_generate_change_log(self, historical_tracker):
        current_responses = {
            "Q1": {"content": {"answer": "yes"}, "text": "Yes"},
            "Q2": {"content": {"value": 5000}, "text": "5000"},
        }
        previous_responses = {
            "Q1": {"content": {"answer": "no"}, "text": "No"},
            "Q2": {"content": {"value": 5000}, "text": "5000"},
        }
        changelog = historical_tracker.generate_change_log(
            current=current_responses,
            previous=previous_responses,
        )
        assert len(changelog) >= 1  # Q1 changed
        assert any(c["question_id"] == "Q1" for c in changelog)

    def test_no_changes_empty_log(self, historical_tracker):
        responses = {
            "Q1": {"content": {"answer": "yes"}, "text": "Yes"},
        }
        changelog = historical_tracker.generate_change_log(
            current=responses,
            previous=responses,
        )
        assert len(changelog) == 0

    def test_new_question_in_changelog(self, historical_tracker):
        current = {
            "Q1": {"content": {"answer": "yes"}, "text": "Yes"},
            "Q_NEW": {"content": {"answer": "new"}, "text": "New response"},
        }
        previous = {
            "Q1": {"content": {"answer": "yes"}, "text": "Yes"},
        }
        changelog = historical_tracker.generate_change_log(current, previous)
        assert any(c["question_id"] == "Q_NEW" and c["change_type"] == "added" for c in changelog)

    def test_removed_question_in_changelog(self, historical_tracker):
        current = {
            "Q1": {"content": {"answer": "yes"}, "text": "Yes"},
        }
        previous = {
            "Q1": {"content": {"answer": "yes"}, "text": "Yes"},
            "Q_OLD": {"content": {"answer": "old"}, "text": "Old response"},
        }
        changelog = historical_tracker.generate_change_log(current, previous)
        assert any(c["question_id"] == "Q_OLD" and c["change_type"] == "removed" for c in changelog)


# ---------------------------------------------------------------------------
# Emissions comparison
# ---------------------------------------------------------------------------

class TestEmissionsComparison:
    """Test year-over-year emissions data comparison."""

    def test_emissions_yoy_increase(self, historical_tracker):
        comparison = historical_tracker.compare_emissions(
            current_emissions={"scope1": Decimal("5500"), "scope2": Decimal("3200")},
            previous_emissions={"scope1": Decimal("5000"), "scope2": Decimal("3000")},
        )
        assert comparison["scope1"]["delta"] == Decimal("500")
        assert comparison["scope1"]["direction"] == "increase"

    def test_emissions_yoy_decrease(self, historical_tracker):
        comparison = historical_tracker.compare_emissions(
            current_emissions={"scope1": Decimal("4500")},
            previous_emissions={"scope1": Decimal("5000")},
        )
        assert comparison["scope1"]["delta"] == Decimal("-500")
        assert comparison["scope1"]["direction"] == "decrease"

    def test_emissions_no_change(self, historical_tracker):
        comparison = historical_tracker.compare_emissions(
            current_emissions={"scope1": Decimal("5000")},
            previous_emissions={"scope1": Decimal("5000")},
        )
        assert comparison["scope1"]["delta"] == Decimal("0")
        assert comparison["scope1"]["direction"] == "stable"

    def test_submission_timeline(self, historical_tracker):
        org_id = _new_id()
        history = [
            {"year": 2022, "submitted_at": "2022-07-15"},
            {"year": 2023, "submitted_at": "2023-07-10"},
            {"year": 2024, "submitted_at": "2024-07-20"},
        ]
        timeline = historical_tracker.get_submission_timeline(org_id, history)
        assert len(timeline) == 3
        assert timeline[0]["year"] == 2022
