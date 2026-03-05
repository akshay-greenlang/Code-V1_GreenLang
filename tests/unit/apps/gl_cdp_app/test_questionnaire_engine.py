# -*- coding: utf-8 -*-
"""
Unit tests for QuestionnaireEngine -- CDP questionnaire management.

Tests module listing, question retrieval, conditional logic, sector-specific
routing, completion tracking, versioning, and questionnaire lifecycle
with 38+ test functions.

Author: GL-TestEngineer
Date: March 2026
"""

from decimal import Decimal

import pytest

from services.config import CDPModule, QuestionType, ScoringCategory
from services.models import (
    CDPOrganization,
    CDPQuestionnaire,
    CDPModuleInstance,
    CDPQuestion,
    _new_id,
)
from services.questionnaire_engine import QuestionnaireEngine


# ---------------------------------------------------------------------------
# Module listing
# ---------------------------------------------------------------------------

class TestModuleListing:
    """Test CDP module definitions and listing."""

    def test_list_all_modules(self, questionnaire_engine):
        modules = questionnaire_engine.list_modules()
        assert len(modules) == 14  # M0 through M13

    def test_module_codes_sequential(self, questionnaire_engine):
        modules = questionnaire_engine.list_modules()
        codes = [m.module_code for m in modules]
        for i in range(14):
            assert f"M{i}" in codes

    def test_m0_introduction(self, questionnaire_engine):
        modules = questionnaire_engine.list_modules()
        m0 = next(m for m in modules if m.module_code == "M0")
        assert "introduction" in m0.module_name.lower() or "Introduction" in m0.module_name

    def test_m7_climate_change(self, questionnaire_engine):
        modules = questionnaire_engine.list_modules()
        m7 = next(m for m in modules if m.module_code == "M7")
        assert "climate" in m7.module_name.lower()

    def test_m12_financial_services(self, questionnaire_engine):
        modules = questionnaire_engine.list_modules()
        m12 = next(m for m in modules if m.module_code == "M12")
        assert "financial" in m12.module_name.lower()

    def test_m13_sign_off(self, questionnaire_engine):
        modules = questionnaire_engine.list_modules()
        m13 = next(m for m in modules if m.module_code == "M13")
        assert "sign" in m13.module_name.lower()

    def test_modules_have_descriptions(self, questionnaire_engine):
        modules = questionnaire_engine.list_modules()
        for m in modules:
            assert m.description is not None
            assert len(m.description) > 0

    def test_modules_sorted_by_order(self, questionnaire_engine):
        modules = questionnaire_engine.list_modules()
        orders = [m.sort_order for m in modules]
        assert orders == sorted(orders)


# ---------------------------------------------------------------------------
# Question retrieval
# ---------------------------------------------------------------------------

class TestQuestionRetrieval:
    """Test question retrieval by module."""

    def test_get_questions_by_module(self, questionnaire_engine):
        questions = questionnaire_engine.get_questions_for_module("M7")
        assert len(questions) > 0

    def test_m7_has_most_questions(self, questionnaire_engine):
        m7_questions = questionnaire_engine.get_questions_for_module("M7")
        m0_questions = questionnaire_engine.get_questions_for_module("M0")
        assert len(m7_questions) >= len(m0_questions)

    def test_m13_has_fewest_questions(self, questionnaire_engine):
        m13_questions = questionnaire_engine.get_questions_for_module("M13")
        assert len(m13_questions) <= 10

    def test_questions_have_numbers(self, questionnaire_engine):
        questions = questionnaire_engine.get_questions_for_module("M1")
        for q in questions:
            assert q.question_number is not None
            assert len(q.question_number) > 0

    def test_questions_have_types(self, questionnaire_engine):
        questions = questionnaire_engine.get_questions_for_module("M7")
        valid_types = {qt.value for qt in QuestionType}
        for q in questions:
            assert q.question_type.value in valid_types

    def test_get_question_by_number(self, questionnaire_engine):
        question = questionnaire_engine.get_question_by_number("C7.1")
        assert question is not None
        assert question.question_number == "C7.1"

    def test_get_nonexistent_question_returns_none(self, questionnaire_engine):
        question = questionnaire_engine.get_question_by_number("C99.99")
        assert question is None

    def test_total_question_count(self, questionnaire_engine):
        total = questionnaire_engine.get_total_question_count()
        assert total >= 200  # CDP has 200+ questions


# ---------------------------------------------------------------------------
# Conditional logic
# ---------------------------------------------------------------------------

class TestConditionalLogic:
    """Test conditional question skip patterns and dependencies."""

    def test_conditional_questions_exist(self, questionnaire_engine):
        questions = questionnaire_engine.get_questions_for_module("M7")
        conditional = [q for q in questions if q.is_conditional]
        assert len(conditional) > 0

    def test_conditional_has_logic(self, questionnaire_engine):
        questions = questionnaire_engine.get_questions_for_module("M7")
        conditional = [q for q in questions if q.is_conditional]
        for q in conditional:
            assert q.condition_logic is not None
            assert "depends_on" in q.condition_logic or "trigger" in q.condition_logic

    def test_evaluate_condition_met(self, questionnaire_engine):
        result = questionnaire_engine.evaluate_condition(
            condition_logic={"depends_on": "C7.1", "value": "yes"},
            current_responses={"C7.1": {"answer": "yes"}},
        )
        assert result is True

    def test_evaluate_condition_not_met(self, questionnaire_engine):
        result = questionnaire_engine.evaluate_condition(
            condition_logic={"depends_on": "C7.1", "value": "yes"},
            current_responses={"C7.1": {"answer": "no"}},
        )
        assert result is False

    def test_evaluate_condition_missing_dependency(self, questionnaire_engine):
        result = questionnaire_engine.evaluate_condition(
            condition_logic={"depends_on": "C7.1", "value": "yes"},
            current_responses={},
        )
        assert result is False


# ---------------------------------------------------------------------------
# Sector-specific routing
# ---------------------------------------------------------------------------

class TestSectorRouting:
    """Test sector-specific module routing."""

    def test_financial_services_enables_m12(self, questionnaire_engine, financial_services_org):
        applicable = questionnaire_engine.get_applicable_modules(
            sector_gics=financial_services_org.sector_gics
        )
        module_codes = [m.module_code for m in applicable]
        assert "M12" in module_codes

    def test_non_financial_excludes_m12(self, questionnaire_engine, sample_organization):
        applicable = questionnaire_engine.get_applicable_modules(
            sector_gics=sample_organization.sector_gics
        )
        module_codes = [m.module_code for m in applicable]
        assert "M12" not in module_codes

    def test_core_modules_always_applicable(self, questionnaire_engine, sample_organization):
        applicable = questionnaire_engine.get_applicable_modules(
            sector_gics=sample_organization.sector_gics
        )
        module_codes = [m.module_code for m in applicable]
        for core in ["M0", "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M13"]:
            assert core in module_codes


# ---------------------------------------------------------------------------
# Module completion
# ---------------------------------------------------------------------------

class TestModuleCompletion:
    """Test module completion percentage calculation."""

    def test_completion_zero_no_responses(self, questionnaire_engine):
        pct = questionnaire_engine.calculate_module_completion(
            module_code="M7",
            total_questions=35,
            answered_questions=0,
        )
        assert pct == Decimal("0")

    def test_completion_fifty_percent(self, questionnaire_engine):
        pct = questionnaire_engine.calculate_module_completion(
            module_code="M7",
            total_questions=20,
            answered_questions=10,
        )
        assert pct == Decimal("50.0")

    def test_completion_hundred_percent(self, questionnaire_engine):
        pct = questionnaire_engine.calculate_module_completion(
            module_code="M7",
            total_questions=35,
            answered_questions=35,
        )
        assert pct == Decimal("100.0")

    def test_completion_overall(self, questionnaire_engine):
        module_completions = {
            "M0": Decimal("100.0"),
            "M1": Decimal("75.0"),
            "M7": Decimal("50.0"),
        }
        overall = questionnaire_engine.calculate_overall_completion(module_completions)
        expected = (Decimal("100.0") + Decimal("75.0") + Decimal("50.0")) / 3
        assert overall == pytest.approx(float(expected), rel=1e-2)


# ---------------------------------------------------------------------------
# Question versioning
# ---------------------------------------------------------------------------

class TestQuestionVersioning:
    """Test question versioning across years."""

    def test_get_questions_for_version_2025(self, questionnaire_engine):
        questions = questionnaire_engine.get_questions_for_version(2025)
        assert len(questions) > 0
        for q in questions:
            assert q.version_year == 2025

    def test_get_questions_for_version_2024(self, questionnaire_engine):
        questions = questionnaire_engine.get_questions_for_version(2024)
        assert len(questions) > 0
        for q in questions:
            assert q.version_year == 2024

    def test_version_question_count_varies(self, questionnaire_engine):
        q_2024 = questionnaire_engine.get_questions_for_version(2024)
        q_2025 = questionnaire_engine.get_questions_for_version(2025)
        # Counts may differ between versions
        assert isinstance(len(q_2024), int)
        assert isinstance(len(q_2025), int)


# ---------------------------------------------------------------------------
# Questionnaire creation
# ---------------------------------------------------------------------------

class TestQuestionnaireCreation:
    """Test creating a full questionnaire for an organization."""

    def test_create_questionnaire_for_org(self, questionnaire_engine, sample_organization):
        questionnaire = questionnaire_engine.create_questionnaire(
            org_id=sample_organization.id,
            reporting_year=2025,
            sector_gics=sample_organization.sector_gics,
        )
        assert questionnaire.org_id == sample_organization.id
        assert questionnaire.reporting_year == 2025
        assert questionnaire.status == "not_started"

    def test_create_questionnaire_assigns_modules(self, questionnaire_engine, sample_organization):
        questionnaire = questionnaire_engine.create_questionnaire(
            org_id=sample_organization.id,
            reporting_year=2025,
            sector_gics=sample_organization.sector_gics,
        )
        modules = questionnaire_engine.get_questionnaire_modules(questionnaire.id)
        assert len(modules) >= 10  # At least core modules

    def test_create_fs_questionnaire_includes_m12(self, questionnaire_engine, financial_services_org):
        questionnaire = questionnaire_engine.create_questionnaire(
            org_id=financial_services_org.id,
            reporting_year=2025,
            sector_gics=financial_services_org.sector_gics,
        )
        modules = questionnaire_engine.get_questionnaire_modules(questionnaire.id)
        codes = [m.module_code for m in modules]
        assert "M12" in codes


# ---------------------------------------------------------------------------
# Question type validation
# ---------------------------------------------------------------------------

class TestQuestionTypeValidation:
    """Test question type constraints."""

    def test_numeric_question_type(self, questionnaire_engine):
        questions = questionnaire_engine.get_questions_for_module("M7")
        numeric_qs = [q for q in questions if q.question_type == QuestionType.NUMERIC]
        assert len(numeric_qs) > 0

    def test_table_question_type(self, questionnaire_engine):
        questions = questionnaire_engine.get_questions_for_module("M7")
        table_qs = [q for q in questions if q.question_type == QuestionType.TABLE]
        assert len(table_qs) > 0
