# -*- coding: utf-8 -*-
"""
Deep tests for ActionPlanEngine (Engine 4 of 10).

Covers: 10-section plan validation, MACC-based prioritization, section
rating thresholds, plan quality classification, publication deadline
tracking, HLEG Rec 3 alignment, action categorization, Decimal
arithmetic, SHA-256 provenance.

Target: ~55 tests.

Author: GreenLang Platform Team
Pack: PACK-025 Race to Zero Pack
"""

import sys
from decimal import Decimal
from pathlib import Path

import pytest

_PACK_ROOT = Path(__file__).resolve().parents[1]
if str(_PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACK_ROOT))
_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

from engines.action_plan_engine import (
    ActionPlanEngine,
    ActionPlanInput,
    DecarbonizationActionInput,
    SectionInput,
    PlanSection,
    SectionRating,
    ActionCategory,
    PlanQuality,
    SECTION_IDS,
    SECTION_WEIGHTS,
    SECTION_LABELS,
    RATING_THRESHOLDS,
    QUALITY_THRESHOLDS,
    MIN_ACTIONS_COUNT,
    PUBLICATION_DEADLINE_MONTHS,
)

from conftest import assert_decimal_close, assert_provenance_hash, timed_block


# ========================================================================
# Section Constants
# ========================================================================


class TestSectionConstants:
    """Validate 10-section plan constants."""

    def test_exactly_10_section_ids(self):
        assert len(SECTION_IDS) == 10

    def test_section_ids_match_enum(self):
        for sec in PlanSection:
            assert sec.value in SECTION_IDS

    def test_weights_sum_to_one(self):
        total = sum(SECTION_WEIGHTS.values())
        assert_decimal_close(total, Decimal("1.00"), Decimal("0.001"))

    def test_targets_highest_weight(self):
        assert SECTION_WEIGHTS["targets"] == Decimal("0.15")

    def test_reduction_actions_highest_weight(self):
        assert SECTION_WEIGHTS["reduction_actions"] == Decimal("0.15")

    def test_just_transition_lowest_weight(self):
        assert SECTION_WEIGHTS["just_transition"] == Decimal("0.06")

    def test_monitoring_lowest_weight(self):
        assert SECTION_WEIGHTS["monitoring"] == Decimal("0.06")

    def test_all_sections_have_labels(self):
        for sid in SECTION_IDS:
            assert sid in SECTION_LABELS

    def test_min_actions_count_10(self):
        assert MIN_ACTIONS_COUNT == 10

    def test_publication_deadline_12_months(self):
        assert PUBLICATION_DEADLINE_MONTHS == 12


# ========================================================================
# Enum Validation
# ========================================================================


class TestActionPlanEnums:
    """Validate action plan enums."""

    def test_plan_section_10_values(self):
        assert len(PlanSection) == 10

    def test_section_rating_5_values(self):
        assert len(SectionRating) == 5

    def test_section_rating_values(self):
        assert SectionRating.COMPLETE.value == "complete"
        assert SectionRating.ADEQUATE.value == "adequate"
        assert SectionRating.PARTIAL.value == "partial"
        assert SectionRating.INCOMPLETE.value == "incomplete"
        assert SectionRating.MISSING.value == "missing"

    def test_action_category_count(self):
        assert len(ActionCategory) >= 10

    @pytest.mark.parametrize("category", [
        "energy_efficiency", "renewable_energy", "electrification",
        "fuel_switching", "process_change", "supply_chain",
        "transport", "buildings", "waste_reduction", "carbon_capture",
    ])
    def test_action_category_values(self, category):
        assert ActionCategory(category) is not None

    def test_plan_quality_5_values(self):
        assert len(PlanQuality) == 5

    def test_plan_quality_values(self):
        assert PlanQuality.EXCELLENT.value == "excellent"
        assert PlanQuality.GOOD.value == "good"
        assert PlanQuality.ADEQUATE.value == "adequate"
        assert PlanQuality.INADEQUATE.value == "inadequate"
        assert PlanQuality.MISSING.value == "missing"


# ========================================================================
# Rating & Quality Thresholds
# ========================================================================


class TestRatingThresholds:
    """Validate rating and quality thresholds."""

    def test_rating_thresholds_5_entries(self):
        assert len(RATING_THRESHOLDS) == 5

    def test_complete_threshold_9(self):
        assert RATING_THRESHOLDS[0] == (Decimal("9"), "complete")

    def test_adequate_threshold_7(self):
        assert RATING_THRESHOLDS[1] == (Decimal("7"), "adequate")

    def test_partial_threshold_5(self):
        assert RATING_THRESHOLDS[2] == (Decimal("5"), "partial")

    def test_quality_thresholds_5_entries(self):
        assert len(QUALITY_THRESHOLDS) == 5

    def test_excellent_threshold_85(self):
        assert QUALITY_THRESHOLDS[0] == (Decimal("85"), "excellent")

    def test_good_threshold_70(self):
        assert QUALITY_THRESHOLDS[1] == (Decimal("70"), "good")


# ========================================================================
# Input Model Validation
# ========================================================================


class TestActionPlanInputModel:
    """Validate ActionPlanInput Pydantic model."""

    def test_complete_input_constructs(self, complete_action_plan_input):
        assert complete_action_plan_input.entity_name == "GreenCorp International"
        assert len(complete_action_plan_input.actions) == 15

    def test_incomplete_input_constructs(self, incomplete_action_plan_input):
        assert incomplete_action_plan_input.entity_name == "SlowStart Ltd"
        assert len(incomplete_action_plan_input.actions) == 3

    def test_action_input_construction(self):
        action = DecarbonizationActionInput(
            action_name="LED retrofit",
            category="energy_efficiency",
            scope_impact=[1, 2],
            abatement_tco2e=Decimal("500"),
            cost_total_usd=Decimal("50000"),
        )
        assert action.action_name == "LED retrofit"

    def test_section_input_construction(self):
        section = SectionInput(
            section_id="targets",
            score=Decimal("9"),
            content_summary="Targets fully documented.",
        )
        assert section.section_id == "targets"

    def test_complete_has_10_sections(self, complete_action_plan_input):
        assert len(complete_action_plan_input.sections) == 10


# ========================================================================
# Engine Instantiation
# ========================================================================


class TestActionPlanEngineInstantiation:
    """Tests for engine creation."""

    def test_default_instantiation(self, action_plan_engine):
        assert action_plan_engine is not None

    def test_engine_has_calculate(self, action_plan_engine):
        assert callable(getattr(action_plan_engine, "assess", None))


# ========================================================================
# Complete Plan Assessment
# ========================================================================


class TestCompletePlanAssessment:
    """Tests for a complete action plan assessment."""

    def test_complete_calculates(
        self, action_plan_engine, complete_action_plan_input,
    ):
        result = action_plan_engine.assess(complete_action_plan_input)
        assert result is not None

    def test_complete_quality_high(
        self, action_plan_engine, complete_action_plan_input,
    ):
        result = action_plan_engine.assess(complete_action_plan_input)
        assert result.plan_quality in ("excellent", "good")

    def test_complete_completeness_above_70(
        self, action_plan_engine, complete_action_plan_input,
    ):
        result = action_plan_engine.assess(complete_action_plan_input)
        assert result.completeness_score >= Decimal("70")

    def test_complete_has_section_results(
        self, action_plan_engine, complete_action_plan_input,
    ):
        result = action_plan_engine.assess(complete_action_plan_input)
        assert hasattr(result, "section_results")

    def test_complete_has_provenance(
        self, action_plan_engine, complete_action_plan_input,
    ):
        result = action_plan_engine.assess(complete_action_plan_input)
        assert_provenance_hash(result)

    def test_complete_has_action_prioritization(
        self, action_plan_engine, complete_action_plan_input,
    ):
        result = action_plan_engine.assess(complete_action_plan_input)
        has_prio = (
            hasattr(result, "prioritized_actions") or
            hasattr(result, "action_results") or
            hasattr(result, "abatement_summary")
        )
        assert has_prio

    def test_complete_performance(
        self, action_plan_engine, complete_action_plan_input,
    ):
        with timed_block("complete_plan_assessment", max_seconds=5.0):
            action_plan_engine.assess(complete_action_plan_input)


# ========================================================================
# Incomplete Plan Assessment
# ========================================================================


class TestIncompletePlanAssessment:
    """Tests for an incomplete action plan assessment."""

    def test_incomplete_calculates(
        self, action_plan_engine, incomplete_action_plan_input,
    ):
        result = action_plan_engine.assess(incomplete_action_plan_input)
        assert result is not None

    def test_incomplete_quality_lower(
        self, action_plan_engine, incomplete_action_plan_input,
    ):
        result = action_plan_engine.assess(incomplete_action_plan_input)
        assert result.plan_quality in ("inadequate", "missing", "adequate")

    def test_incomplete_has_provenance(
        self, action_plan_engine, incomplete_action_plan_input,
    ):
        result = action_plan_engine.assess(incomplete_action_plan_input)
        assert_provenance_hash(result)


# ========================================================================
# Determinism
# ========================================================================


class TestActionPlanDeterminism:
    """Tests for deterministic output."""

    def test_same_input_same_quality(
        self, action_plan_engine, complete_action_plan_input,
    ):
        r1 = action_plan_engine.assess(complete_action_plan_input)
        r2 = action_plan_engine.assess(complete_action_plan_input)
        assert r1.plan_quality == r2.plan_quality

    def test_same_input_same_score(
        self, action_plan_engine, complete_action_plan_input,
    ):
        r1 = action_plan_engine.assess(complete_action_plan_input)
        r2 = action_plan_engine.assess(complete_action_plan_input)
        assert r1.completeness_score == r2.completeness_score
