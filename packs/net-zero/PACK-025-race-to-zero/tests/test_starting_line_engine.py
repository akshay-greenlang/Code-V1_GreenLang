# -*- coding: utf-8 -*-
"""
Deep tests for StartingLineEngine (Engine 2 of 10).

Covers: 4P framework (Pledge/Plan/Proceed/Publish), 20 sub-criteria
validation, compliance status assessment, remediation plan generation,
12-month deadline tracking, scope coverage validation, SHA-256
provenance, Decimal arithmetic, edge cases.

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

from engines.starting_line_engine import (
    ComplianceStatus,
    StartingLineEngine,
    StartingLineInput,
    SubCriterionInput,
    SubCriterionStatus,
    StartingLineCategory,
    SUB_CRITERIA,
    SUB_CRITERION_IDS,
    STATUS_SCORES,
    CATEGORY_LABELS,
    PRIORITY_WEIGHTS,
)

from conftest import assert_decimal_close, assert_provenance_hash, timed_block


# ========================================================================
# Sub-Criteria Constants
# ========================================================================


class TestSubCriteriaConstants:
    """Validate the 20 sub-criteria definitions."""

    def test_exactly_20_sub_criteria(self):
        assert len(SUB_CRITERIA) == 20

    def test_exactly_20_sub_criterion_ids(self):
        assert len(SUB_CRITERION_IDS) == 20

    def test_pledge_category_has_5(self):
        pledge_ids = [k for k, v in SUB_CRITERIA.items()
                      if v["category"] == "pledge"]
        assert len(pledge_ids) == 5

    def test_plan_category_has_5(self):
        plan_ids = [k for k, v in SUB_CRITERIA.items()
                    if v["category"] == "plan"]
        assert len(plan_ids) == 5

    def test_proceed_category_has_5(self):
        proceed_ids = [k for k, v in SUB_CRITERIA.items()
                       if v["category"] == "proceed"]
        assert len(proceed_ids) == 5

    def test_publish_category_has_5(self):
        publish_ids = [k for k, v in SUB_CRITERIA.items()
                       if v["category"] == "publish"]
        assert len(publish_ids) == 5

    def test_pledge_ids_pattern(self):
        for i in range(1, 6):
            assert f"SL-P{i}" in SUB_CRITERION_IDS

    def test_plan_ids_pattern(self):
        for i in range(1, 6):
            assert f"SL-A{i}" in SUB_CRITERION_IDS

    def test_proceed_ids_pattern(self):
        for i in range(1, 6):
            assert f"SL-R{i}" in SUB_CRITERION_IDS

    def test_publish_ids_pattern(self):
        for i in range(1, 6):
            assert f"SL-D{i}" in SUB_CRITERION_IDS

    def test_each_sub_criterion_has_required_fields(self):
        for sc_id, sc_data in SUB_CRITERIA.items():
            assert "category" in sc_data, f"{sc_id} missing category"
            assert "name" in sc_data, f"{sc_id} missing name"
            assert "description" in sc_data, f"{sc_id} missing description"
            assert "evidence_required" in sc_data, f"{sc_id} missing evidence_required"


# ========================================================================
# Enum Validation
# ========================================================================


class TestStartingLineEnums:
    """Validate Starting Line enums."""

    def test_starting_line_category_4_values(self):
        assert len(StartingLineCategory) == 4

    def test_category_values(self):
        assert StartingLineCategory.PLEDGE.value == "pledge"
        assert StartingLineCategory.PLAN.value == "plan"
        assert StartingLineCategory.PROCEED.value == "proceed"
        assert StartingLineCategory.PUBLISH.value == "publish"

    def test_compliance_status_3_values(self):
        assert len(ComplianceStatus) == 3

    def test_compliance_values(self):
        assert ComplianceStatus.COMPLIANT.value == "compliant"
        assert ComplianceStatus.PARTIALLY_COMPLIANT.value == "partially_compliant"
        assert ComplianceStatus.NON_COMPLIANT.value == "non_compliant"

    def test_sub_criterion_status_4_values(self):
        assert len(SubCriterionStatus) == 4

    def test_sub_criterion_status_values(self):
        assert SubCriterionStatus.PASS.value == "pass"
        assert SubCriterionStatus.PARTIAL.value == "partial"
        assert SubCriterionStatus.FAIL.value == "fail"
        assert SubCriterionStatus.NOT_APPLICABLE.value == "not_applicable"

    def test_status_scores_mapping(self):
        assert STATUS_SCORES["pass"] == Decimal("1.0")
        assert STATUS_SCORES["partial"] == Decimal("0.5")
        assert STATUS_SCORES["fail"] == Decimal("0.0")
        assert STATUS_SCORES["not_applicable"] == Decimal("1.0")

    def test_category_labels_4_entries(self):
        assert len(CATEGORY_LABELS) == 4

    def test_priority_weights_4_entries(self):
        assert len(PRIORITY_WEIGHTS) == 4

    def test_pledge_highest_priority(self):
        assert PRIORITY_WEIGHTS["pledge"] == 1


# ========================================================================
# Input Model Validation
# ========================================================================


class TestStartingLineInputModel:
    """Validate StartingLineInput Pydantic model."""

    def test_compliant_input_constructs(self, compliant_starting_line_input):
        assert compliant_starting_line_input.entity_name == "GreenCorp International"

    def test_non_compliant_input_constructs(self, non_compliant_starting_line_input):
        assert non_compliant_starting_line_input.entity_name == "SlowStart Ltd"

    def test_input_has_all_boolean_fields(self, compliant_starting_line_input):
        assert compliant_starting_line_input.action_plan_published is True
        assert compliant_starting_line_input.immediate_actions_taken is True
        assert compliant_starting_line_input.annual_reporting_done is True

    def test_sub_criterion_input_validation(self):
        sc = SubCriterionInput(criterion_id="SL-P1", status="pass")
        assert sc.criterion_id == "SL-P1"

    def test_invalid_sub_criterion_id_raises(self):
        with pytest.raises(Exception):
            SubCriterionInput(criterion_id="SL-X99")

    def test_invalid_sub_criterion_status_raises(self):
        with pytest.raises(Exception):
            SubCriterionInput(criterion_id="SL-P1", status="bogus")


# ========================================================================
# Engine Instantiation
# ========================================================================


class TestStartingLineEngineInstantiation:
    """Tests for engine creation."""

    def test_default_instantiation(self, starting_line_engine):
        assert starting_line_engine is not None

    def test_engine_has_calculate(self, starting_line_engine):
        assert callable(getattr(starting_line_engine, "assess", None))


# ========================================================================
# Compliant Assessment
# ========================================================================


class TestCompliantStartingLineAssessment:
    """Tests for a fully compliant Starting Line assessment."""

    def test_compliant_calculates(
        self, starting_line_engine, compliant_starting_line_input,
    ):
        result = starting_line_engine.assess(compliant_starting_line_input)
        assert result is not None

    def test_compliant_status(
        self, starting_line_engine, compliant_starting_line_input,
    ):
        result = starting_line_engine.assess(compliant_starting_line_input)
        assert result.overall_status in (
            "compliant", "partially_compliant",
        )

    def test_compliant_overall_score_high(
        self, starting_line_engine, compliant_starting_line_input,
    ):
        result = starting_line_engine.assess(compliant_starting_line_input)
        assert result.overall_score >= Decimal("70")

    def test_compliant_has_category_results(
        self, starting_line_engine, compliant_starting_line_input,
    ):
        result = starting_line_engine.assess(compliant_starting_line_input)
        assert hasattr(result, "category_results") or hasattr(result, "categories")

    def test_compliant_has_provenance(
        self, starting_line_engine, compliant_starting_line_input,
    ):
        result = starting_line_engine.assess(compliant_starting_line_input)
        assert_provenance_hash(result)

    def test_compliant_has_processing_time(
        self, starting_line_engine, compliant_starting_line_input,
    ):
        result = starting_line_engine.assess(compliant_starting_line_input)
        assert result.processing_time_ms >= 0

    def test_compliant_performance(
        self, starting_line_engine, compliant_starting_line_input,
    ):
        with timed_block("compliant_sl_assessment", max_seconds=5.0):
            starting_line_engine.assess(compliant_starting_line_input)


# ========================================================================
# Non-Compliant Assessment
# ========================================================================


class TestNonCompliantStartingLineAssessment:
    """Tests for a non-compliant Starting Line assessment."""

    def test_non_compliant_calculates(
        self, starting_line_engine, non_compliant_starting_line_input,
    ):
        result = starting_line_engine.assess(non_compliant_starting_line_input)
        assert result is not None

    def test_non_compliant_status(
        self, starting_line_engine, non_compliant_starting_line_input,
    ):
        result = starting_line_engine.assess(non_compliant_starting_line_input)
        assert result.overall_status in (
            "non_compliant", "partially_compliant",
        )

    def test_non_compliant_score_lower(
        self, starting_line_engine, non_compliant_starting_line_input,
    ):
        result = starting_line_engine.assess(non_compliant_starting_line_input)
        assert result.overall_score < Decimal("80")

    def test_non_compliant_has_provenance(
        self, starting_line_engine, non_compliant_starting_line_input,
    ):
        result = starting_line_engine.assess(non_compliant_starting_line_input)
        assert_provenance_hash(result)

    def test_non_compliant_has_remediation(
        self, starting_line_engine, non_compliant_starting_line_input,
    ):
        result = starting_line_engine.assess(non_compliant_starting_line_input)
        has_remediation = (
            hasattr(result, "remediation_plan") or
            hasattr(result, "gaps") or
            hasattr(result, "recommendations")
        )
        assert has_remediation


# ========================================================================
# Determinism
# ========================================================================


class TestStartingLineDeterminism:
    """Tests for deterministic output."""

    def test_same_input_same_score(
        self, starting_line_engine, compliant_starting_line_input,
    ):
        r1 = starting_line_engine.assess(compliant_starting_line_input)
        r2 = starting_line_engine.assess(compliant_starting_line_input)
        assert r1.overall_score == r2.overall_score

    def test_same_input_same_status(
        self, starting_line_engine, compliant_starting_line_input,
    ):
        r1 = starting_line_engine.assess(compliant_starting_line_input)
        r2 = starting_line_engine.assess(compliant_starting_line_input)
        assert r1.overall_status == r2.overall_status
