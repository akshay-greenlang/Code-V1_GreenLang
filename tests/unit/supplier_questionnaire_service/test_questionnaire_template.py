# -*- coding: utf-8 -*-
"""
Unit tests for QuestionnaireTemplateEngine
==========================================

AGENT-DATA-008: Supplier Questionnaire Processor
Tests all methods of QuestionnaireTemplateEngine with comprehensive
coverage.  Validates template CRUD, versioning, cloning, validation,
import/export, built-in frameworks, multi-language, conditional rules,
and SHA-256 provenance hashing.

Total: ~80 tests
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
import uuid
from typing import Any, Dict, List

import pytest

from greenlang.supplier_questionnaire.questionnaire_template import (
    QuestionnaireTemplateEngine,
)
from greenlang.supplier_questionnaire.models import (
    Framework,
    QuestionnaireStatus,
    QuestionnaireTemplate,
    QuestionType,
    TemplateQuestion,
    TemplateSection,
    ValidationCheck,
    ValidationSeverity,
    ValidationSummary,
)


# ===================================================================
# Fixtures
# ===================================================================

@pytest.fixture
def engine() -> QuestionnaireTemplateEngine:
    """Fresh engine instance per test."""
    return QuestionnaireTemplateEngine()


@pytest.fixture
def engine_no_auto() -> QuestionnaireTemplateEngine:
    """Engine with auto_populate_framework disabled."""
    return QuestionnaireTemplateEngine({"auto_populate_framework": False})


@pytest.fixture
def cdp_template(engine: QuestionnaireTemplateEngine) -> QuestionnaireTemplate:
    """Create a CDP Climate template for reuse."""
    return engine.create_template(
        name="CDP Climate 2025",
        framework="cdp_climate",
        language="en",
        description="Annual CDP climate disclosure",
        tags=["cdp", "climate"],
        created_by="test-user",
    )


@pytest.fixture
def custom_section() -> TemplateSection:
    """A standalone section with two questions."""
    return TemplateSection(
        name="Custom Section",
        order=99,
        questions=[
            TemplateQuestion(
                code="CS.1", text="Custom question 1",
                question_type=QuestionType.TEXT, order=0,
            ),
            TemplateQuestion(
                code="CS.2", text="Custom question 2",
                question_type=QuestionType.NUMERIC, order=1,
            ),
        ],
    )


@pytest.fixture
def choice_question() -> TemplateQuestion:
    """A SINGLE_CHOICE question with choices defined."""
    return TemplateQuestion(
        code="SC.1",
        text="Select an option",
        question_type=QuestionType.SINGLE_CHOICE,
        choices=["Alpha", "Beta", "Gamma"],
        order=0,
    )


@pytest.fixture
def empty_choice_question() -> TemplateQuestion:
    """A SINGLE_CHOICE question WITHOUT choices (invalid)."""
    return TemplateQuestion(
        code="SC.BAD",
        text="Select an option (no choices)",
        question_type=QuestionType.SINGLE_CHOICE,
        choices=[],
        order=0,
    )


# ===================================================================
# TEST CLASS: Create Template
# ===================================================================

class TestCreateTemplate:
    """Tests for QuestionnaireTemplateEngine.create_template()."""

    def test_create_template_basic_returns_template(self, engine):
        tmpl = engine.create_template(name="Basic", framework="custom")
        assert isinstance(tmpl, QuestionnaireTemplate)

    def test_create_template_name_preserved(self, engine):
        tmpl = engine.create_template(name="My Template", framework="custom")
        assert tmpl.name == "My Template"

    def test_create_template_framework_resolved(self, engine):
        tmpl = engine.create_template(name="T", framework="cdp_climate")
        assert tmpl.framework == Framework.CDP_CLIMATE

    def test_create_template_version_starts_at_one(self, engine):
        tmpl = engine.create_template(name="T", framework="custom")
        assert tmpl.version == 1

    def test_create_template_status_is_draft(self, engine):
        tmpl = engine.create_template(name="T", framework="custom")
        assert tmpl.status == QuestionnaireStatus.DRAFT

    def test_create_template_language_default_en(self, engine):
        tmpl = engine.create_template(name="T", framework="custom")
        assert tmpl.language == "en"

    def test_create_template_language_override(self, engine):
        tmpl = engine.create_template(name="T", framework="custom", language="de")
        assert tmpl.language == "de"

    def test_create_template_description_stored(self, engine):
        tmpl = engine.create_template(
            name="T", framework="custom", description="Desc",
        )
        assert tmpl.description == "Desc"

    def test_create_template_tags_stored(self, engine):
        tmpl = engine.create_template(
            name="T", framework="custom", tags=["a", "b"],
        )
        assert tmpl.tags == ["a", "b"]

    def test_create_template_created_by_stored(self, engine):
        tmpl = engine.create_template(
            name="T", framework="custom", created_by="alice",
        )
        assert tmpl.created_by == "alice"

    def test_create_template_provenance_hash_is_sha256(self, engine):
        tmpl = engine.create_template(name="T", framework="custom")
        assert len(tmpl.provenance_hash) == 64
        assert re.fullmatch(r"[0-9a-f]{64}", tmpl.provenance_hash)

    def test_create_template_unique_ids(self, engine):
        t1 = engine.create_template(name="A", framework="custom")
        t2 = engine.create_template(name="B", framework="custom")
        assert t1.template_id != t2.template_id

    def test_create_template_empty_name_raises(self, engine):
        with pytest.raises(ValueError, match="non-empty"):
            engine.create_template(name="", framework="custom")

    def test_create_template_whitespace_name_raises(self, engine):
        with pytest.raises(ValueError, match="non-empty"):
            engine.create_template(name="   ", framework="custom")

    def test_create_template_invalid_framework_raises(self, engine):
        with pytest.raises(ValueError, match="[Uu]nknown framework"):
            engine.create_template(name="T", framework="nonexistent")

    def test_create_template_max_limit_enforced(self):
        eng = QuestionnaireTemplateEngine({"max_templates": 2})
        eng.create_template(name="A", framework="custom")
        eng.create_template(name="B", framework="custom")
        with pytest.raises(ValueError, match="[Mm]aximum templates"):
            eng.create_template(name="C", framework="custom")

    def test_create_template_with_explicit_sections(self, engine_no_auto):
        sec = TemplateSection(name="S1", order=0)
        tmpl = engine_no_auto.create_template(
            name="T", framework="custom", sections=[sec],
        )
        assert len(tmpl.sections) == 1
        assert tmpl.sections[0].name == "S1"

    def test_create_template_statistics_increment(self, engine):
        engine.create_template(name="T", framework="custom")
        stats = engine.get_statistics()
        assert stats["templates_created"] == 1

    def test_create_template_supported_languages_includes_primary(self, engine):
        tmpl = engine.create_template(name="T", framework="custom", language="fr")
        assert "fr" in tmpl.supported_languages


# ===================================================================
# TEST CLASS: Built-in Framework Templates (auto-populate)
# ===================================================================

class TestBuiltInFrameworks:
    """Tests for auto-populated CDP, EcoVadis, and DJSI templates."""

    def test_cdp_climate_auto_populates_11_sections(self, engine):
        tmpl = engine.create_template(name="CDP", framework="cdp_climate")
        assert len(tmpl.sections) == 11

    def test_cdp_climate_first_section_is_c0(self, engine):
        tmpl = engine.create_template(name="CDP", framework="cdp_climate")
        assert "C0" in tmpl.sections[0].name

    def test_cdp_climate_c6_has_numeric_questions(self, engine):
        tmpl = engine.create_template(name="CDP", framework="cdp_climate")
        c6 = [s for s in tmpl.sections if "C6" in s.name][0]
        for q in c6.questions:
            assert q.question_type == QuestionType.NUMERIC

    def test_ecovadis_auto_populates_4_sections(self, engine):
        tmpl = engine.create_template(name="EV", framework="ecovadis")
        assert len(tmpl.sections) == 4

    def test_ecovadis_has_environment_section(self, engine):
        tmpl = engine.create_template(name="EV", framework="ecovadis")
        names = [s.name for s in tmpl.sections]
        assert "Environment" in names

    def test_djsi_auto_populates_3_sections(self, engine):
        tmpl = engine.create_template(name="DJ", framework="djsi")
        assert len(tmpl.sections) == 3

    def test_djsi_has_economic_dimension(self, engine):
        tmpl = engine.create_template(name="DJ", framework="djsi")
        names = [s.name for s in tmpl.sections]
        assert "Economic Dimension" in names

    def test_custom_framework_no_auto_sections(self, engine):
        tmpl = engine.create_template(name="Custom", framework="custom")
        assert len(tmpl.sections) == 0

    def test_auto_populate_disabled_yields_empty_sections(self, engine_no_auto):
        tmpl = engine_no_auto.create_template(name="CDP", framework="cdp_climate")
        assert len(tmpl.sections) == 0

    def test_cdp_climate_questions_have_framework_ref(self, engine):
        tmpl = engine.create_template(name="CDP", framework="cdp_climate")
        for section in tmpl.sections:
            for q in section.questions:
                assert q.framework_ref != ""

    def test_cdp_choice_questions_have_choices(self, engine):
        tmpl = engine.create_template(name="CDP", framework="cdp_climate")
        for section in tmpl.sections:
            for q in section.questions:
                if q.question_type in (
                    QuestionType.SINGLE_CHOICE, QuestionType.MULTI_CHOICE,
                ):
                    assert len(q.choices) > 0

    def test_cdp_section_ordering_sequential(self, engine):
        tmpl = engine.create_template(name="CDP", framework="cdp_climate")
        orders = [s.order for s in tmpl.sections]
        assert orders == sorted(orders)


# ===================================================================
# TEST CLASS: Get Template
# ===================================================================

class TestGetTemplate:
    """Tests for QuestionnaireTemplateEngine.get_template()."""

    def test_get_template_existing_returns_template(self, engine, cdp_template):
        fetched = engine.get_template(cdp_template.template_id)
        assert fetched.template_id == cdp_template.template_id

    def test_get_template_non_existing_raises(self, engine):
        with pytest.raises(ValueError, match="[Uu]nknown template"):
            engine.get_template("non-existent-id")

    def test_get_template_preserves_all_fields(self, engine, cdp_template):
        fetched = engine.get_template(cdp_template.template_id)
        assert fetched.name == cdp_template.name
        assert fetched.framework == cdp_template.framework
        assert fetched.version == cdp_template.version


# ===================================================================
# TEST CLASS: List Templates
# ===================================================================

class TestListTemplates:
    """Tests for QuestionnaireTemplateEngine.list_templates()."""

    def test_list_templates_empty_engine(self, engine):
        assert engine.list_templates() == []

    def test_list_templates_returns_all(self, engine):
        engine.create_template(name="A", framework="custom")
        engine.create_template(name="B", framework="cdp_climate")
        assert len(engine.list_templates()) == 2

    def test_list_templates_filter_by_framework(self, engine):
        engine.create_template(name="A", framework="custom")
        engine.create_template(name="B", framework="cdp_climate")
        filtered = engine.list_templates(framework="cdp_climate")
        assert len(filtered) == 1
        assert filtered[0].framework == Framework.CDP_CLIMATE

    def test_list_templates_filter_by_status(self, engine):
        t = engine.create_template(name="A", framework="custom")
        engine.update_template(
            t.template_id, {"status": QuestionnaireStatus.ACTIVE},
        )
        actives = engine.list_templates(status="active")
        assert len(actives) == 1

    def test_list_templates_combined_filters(self, engine):
        engine.create_template(name="A", framework="cdp_climate")
        engine.create_template(name="B", framework="custom")
        result = engine.list_templates(framework="cdp_climate", status="draft")
        assert len(result) == 1

    def test_list_templates_no_match_returns_empty(self, engine):
        engine.create_template(name="A", framework="custom")
        assert engine.list_templates(framework="djsi") == []


# ===================================================================
# TEST CLASS: Update Template
# ===================================================================

class TestUpdateTemplate:
    """Tests for QuestionnaireTemplateEngine.update_template()."""

    def test_update_template_increments_version(self, engine, cdp_template):
        updated = engine.update_template(
            cdp_template.template_id, {"name": "Updated"},
        )
        assert updated.version == 2

    def test_update_template_name_changed(self, engine, cdp_template):
        updated = engine.update_template(
            cdp_template.template_id, {"name": "New Name"},
        )
        assert updated.name == "New Name"

    def test_update_template_description_changed(self, engine, cdp_template):
        updated = engine.update_template(
            cdp_template.template_id, {"description": "New desc"},
        )
        assert updated.description == "New desc"

    def test_update_template_tags_changed(self, engine, cdp_template):
        updated = engine.update_template(
            cdp_template.template_id, {"tags": ["new_tag"]},
        )
        assert updated.tags == ["new_tag"]

    def test_update_template_provenance_changes(self, engine, cdp_template):
        old_hash = cdp_template.provenance_hash
        updated = engine.update_template(
            cdp_template.template_id, {"name": "v2"},
        )
        assert updated.provenance_hash != old_hash

    def test_update_template_non_existing_raises(self, engine):
        with pytest.raises(ValueError, match="[Uu]nknown template"):
            engine.update_template("fake-id", {"name": "X"})

    def test_update_template_ignores_disallowed_fields(self, engine, cdp_template):
        updated = engine.update_template(
            cdp_template.template_id, {"template_id": "hacked-id"},
        )
        assert updated.template_id == cdp_template.template_id

    def test_update_template_multiple_updates_version_monotonic(self, engine, cdp_template):
        v2 = engine.update_template(cdp_template.template_id, {"name": "v2"})
        v3 = engine.update_template(cdp_template.template_id, {"name": "v3"})
        assert v2.version == 2
        assert v3.version == 3

    def test_update_template_stats_increment(self, engine, cdp_template):
        engine.update_template(cdp_template.template_id, {"name": "v2"})
        stats = engine.get_statistics()
        assert stats["templates_updated"] >= 1


# ===================================================================
# TEST CLASS: Clone Template
# ===================================================================

class TestCloneTemplate:
    """Tests for QuestionnaireTemplateEngine.clone_template()."""

    def test_clone_template_new_id(self, engine, cdp_template):
        cloned = engine.clone_template(cdp_template.template_id, "Clone")
        assert cloned.template_id != cdp_template.template_id

    def test_clone_template_new_name(self, engine, cdp_template):
        cloned = engine.clone_template(cdp_template.template_id, "Clone Name")
        assert cloned.name == "Clone Name"

    def test_clone_template_version_reset_to_one(self, engine, cdp_template):
        engine.update_template(cdp_template.template_id, {"name": "v2"})
        cloned = engine.clone_template(cdp_template.template_id, "Clone")
        assert cloned.version == 1

    def test_clone_template_status_is_draft(self, engine, cdp_template):
        cloned = engine.clone_template(cdp_template.template_id, "Clone")
        assert cloned.status == QuestionnaireStatus.DRAFT

    def test_clone_template_preserves_section_count(self, engine, cdp_template):
        cloned = engine.clone_template(cdp_template.template_id, "Clone")
        assert len(cloned.sections) == len(cdp_template.sections)

    def test_clone_template_fresh_section_ids(self, engine, cdp_template):
        cloned = engine.clone_template(cdp_template.template_id, "Clone")
        original_ids = {s.section_id for s in cdp_template.sections}
        cloned_ids = {s.section_id for s in cloned.sections}
        assert original_ids.isdisjoint(cloned_ids)

    def test_clone_template_fresh_question_ids(self, engine, cdp_template):
        cloned = engine.clone_template(cdp_template.template_id, "Clone")
        original_q_ids = set()
        for s in cdp_template.sections:
            for q in s.questions:
                original_q_ids.add(q.question_id)
        cloned_q_ids = set()
        for s in cloned.sections:
            for q in s.questions:
                cloned_q_ids.add(q.question_id)
        assert original_q_ids.isdisjoint(cloned_q_ids)

    def test_clone_template_description_references_source(self, engine, cdp_template):
        cloned = engine.clone_template(cdp_template.template_id, "Clone")
        assert "Cloned from" in cloned.description
        assert cdp_template.name in cloned.description

    def test_clone_template_preserves_tags(self, engine, cdp_template):
        cloned = engine.clone_template(cdp_template.template_id, "Clone")
        assert cloned.tags == cdp_template.tags

    def test_clone_template_non_existing_raises(self, engine):
        with pytest.raises(ValueError, match="[Uu]nknown template"):
            engine.clone_template("bad-id", "Clone")

    def test_clone_template_stats_increment(self, engine, cdp_template):
        engine.clone_template(cdp_template.template_id, "Clone")
        stats = engine.get_statistics()
        assert stats["templates_cloned"] == 1

    def test_clone_template_provenance_hash_set(self, engine, cdp_template):
        cloned = engine.clone_template(cdp_template.template_id, "Clone")
        assert len(cloned.provenance_hash) == 64


# ===================================================================
# TEST CLASS: Add Section
# ===================================================================

class TestAddSection:
    """Tests for QuestionnaireTemplateEngine.add_section()."""

    def test_add_section_appends(self, engine, cdp_template, custom_section):
        before_count = len(cdp_template.sections)
        engine.add_section(cdp_template.template_id, custom_section)
        refreshed = engine.get_template(cdp_template.template_id)
        assert len(refreshed.sections) == before_count + 1

    def test_add_section_auto_order_when_zero(self, engine, cdp_template):
        max_existing = max(s.order for s in cdp_template.sections)
        sec = TemplateSection(name="New Sec", order=0)
        result = engine.add_section(cdp_template.template_id, sec)
        assert result.order == max_existing + 1

    def test_add_section_explicit_order_preserved(self, engine, cdp_template):
        sec = TemplateSection(name="Explicit", order=50)
        result = engine.add_section(cdp_template.template_id, sec)
        assert result.order == 50

    def test_add_section_non_existing_template_raises(self, engine, custom_section):
        with pytest.raises(ValueError, match="[Uu]nknown template"):
            engine.add_section("bad-id", custom_section)

    def test_add_section_updates_provenance(self, engine, cdp_template, custom_section):
        old_hash = engine.get_template(cdp_template.template_id).provenance_hash
        engine.add_section(cdp_template.template_id, custom_section)
        new_hash = engine.get_template(cdp_template.template_id).provenance_hash
        assert new_hash != old_hash


# ===================================================================
# TEST CLASS: Add Question
# ===================================================================

class TestAddQuestion:
    """Tests for QuestionnaireTemplateEngine.add_question()."""

    def test_add_question_to_section(self, engine, cdp_template):
        section_id = cdp_template.sections[0].section_id
        q = TemplateQuestion(code="NEW.1", text="New question", order=99)
        result = engine.add_question(cdp_template.template_id, section_id, q)
        assert result.code == "NEW.1"

    def test_add_question_auto_order(self, engine, cdp_template):
        section = cdp_template.sections[0]
        max_existing = max(qq.order for qq in section.questions) if section.questions else -1
        q = TemplateQuestion(code="NEW.2", text="New Q", order=0)
        result = engine.add_question(
            cdp_template.template_id, section.section_id, q,
        )
        assert result.order == max_existing + 1

    def test_add_question_bad_section_raises(self, engine, cdp_template):
        q = TemplateQuestion(code="X", text="X")
        with pytest.raises(ValueError, match="not found"):
            engine.add_question(cdp_template.template_id, "bad-section", q)

    def test_add_question_bad_template_raises(self, engine):
        q = TemplateQuestion(code="X", text="X")
        with pytest.raises(ValueError, match="[Uu]nknown template"):
            engine.add_question("bad-tmpl", "bad-sec", q)

    def test_add_question_updates_provenance(self, engine, cdp_template):
        old_hash = engine.get_template(cdp_template.template_id).provenance_hash
        section_id = cdp_template.sections[0].section_id
        q = TemplateQuestion(code="P.1", text="Provenance Q")
        engine.add_question(cdp_template.template_id, section_id, q)
        new_hash = engine.get_template(cdp_template.template_id).provenance_hash
        assert new_hash != old_hash

    def test_add_question_all_question_types(self, engine, cdp_template):
        section_id = cdp_template.sections[0].section_id
        for qt in [QuestionType.TEXT, QuestionType.NUMERIC,
                    QuestionType.SINGLE_CHOICE, QuestionType.MULTI_CHOICE,
                    QuestionType.DATE, QuestionType.FILE_UPLOAD,
                    QuestionType.YES_NO, QuestionType.TABLE]:
            q = TemplateQuestion(
                code=f"QT.{qt.value}", text=f"Question {qt.value}",
                question_type=qt,
            )
            result = engine.add_question(
                cdp_template.template_id, section_id, q,
            )
            assert result.question_type == qt


# ===================================================================
# TEST CLASS: Validate Template
# ===================================================================

class TestValidateTemplate:
    """Tests for QuestionnaireTemplateEngine.validate_template()."""

    def test_validate_cdp_template_is_valid(self, engine, cdp_template):
        summary = engine.validate_template(cdp_template.template_id)
        assert summary.is_valid is True

    def test_validate_template_returns_summary(self, engine, cdp_template):
        summary = engine.validate_template(cdp_template.template_id)
        assert isinstance(summary, ValidationSummary)

    def test_validate_template_total_checks_positive(self, engine, cdp_template):
        summary = engine.validate_template(cdp_template.template_id)
        assert summary.total_checks > 0

    def test_validate_template_no_sections_is_error(self, engine):
        tmpl = engine.create_template(name="Empty", framework="custom")
        summary = engine.validate_template(tmpl.template_id)
        assert summary.error_count > 0
        assert summary.is_valid is False

    def test_validate_template_empty_question_text_is_error(self, engine):
        sec = TemplateSection(
            name="Sec",
            order=0,
            questions=[TemplateQuestion(code="Q1", text="", order=0)],
        )
        tmpl = engine.create_template(
            name="T", framework="custom", sections=[sec],
        )
        summary = engine.validate_template(tmpl.template_id)
        assert summary.error_count > 0

    def test_validate_template_choice_without_choices_is_error(
        self, engine, empty_choice_question,
    ):
        sec = TemplateSection(
            name="Sec", order=0, questions=[empty_choice_question],
        )
        tmpl = engine.create_template(
            name="T", framework="custom", sections=[sec],
        )
        summary = engine.validate_template(tmpl.template_id)
        error_msgs = [c.message for c in summary.checks if not c.passed]
        assert any("no choices" in m.lower() for m in error_msgs)

    def test_validate_template_data_quality_score(self, engine, cdp_template):
        summary = engine.validate_template(cdp_template.template_id)
        assert 0.0 <= summary.data_quality_score <= 100.0

    def test_validate_template_provenance_hash_set(self, engine, cdp_template):
        summary = engine.validate_template(cdp_template.template_id)
        assert len(summary.provenance_hash) == 64

    def test_validate_template_non_existing_raises(self, engine):
        with pytest.raises(ValueError, match="[Uu]nknown template"):
            engine.validate_template("bad-id")

    def test_validate_template_stats_increment(self, engine, cdp_template):
        engine.validate_template(cdp_template.template_id)
        stats = engine.get_statistics()
        assert stats["validations_run"] == 1

    def test_validate_template_section_ordering_check(self, engine):
        sec1 = TemplateSection(
            name="S1", order=5,
            questions=[TemplateQuestion(code="Q1", text="T1")],
        )
        sec2 = TemplateSection(
            name="S2", order=2,
            questions=[TemplateQuestion(code="Q2", text="T2")],
        )
        tmpl = engine.create_template(
            name="T", framework="custom", sections=[sec1, sec2],
        )
        summary = engine.validate_template(tmpl.template_id)
        ordering_checks = [
            c for c in summary.checks
            if "ordering" in c.message.lower()
        ]
        assert len(ordering_checks) > 0
        assert ordering_checks[0].passed is False

    def test_validate_template_duplicate_codes_detected(self, engine):
        q1 = TemplateQuestion(code="DUP", text="Q1")
        q2 = TemplateQuestion(code="DUP", text="Q2")
        sec = TemplateSection(name="S", order=0, questions=[q1, q2])
        tmpl = engine.create_template(
            name="T", framework="custom", sections=[sec],
        )
        summary = engine.validate_template(tmpl.template_id)
        dup_checks = [
            c for c in summary.checks if "duplicate" in c.message.lower()
        ]
        assert len(dup_checks) > 0
        assert dup_checks[0].passed is False

    def test_validate_template_metadata_check_passes(self, engine, cdp_template):
        summary = engine.validate_template(cdp_template.template_id)
        meta_checks = [
            c for c in summary.checks if "metadata" in c.message.lower()
        ]
        assert len(meta_checks) > 0
        assert meta_checks[0].passed is True


# ===================================================================
# TEST CLASS: Export / Import
# ===================================================================

class TestExportImport:
    """Tests for export_template() and import_template()."""

    def test_export_returns_json_string(self, engine, cdp_template):
        exported = engine.export_template(cdp_template.template_id)
        parsed = json.loads(exported)
        assert isinstance(parsed, dict)

    def test_export_contains_template_name(self, engine, cdp_template):
        exported = engine.export_template(cdp_template.template_id)
        parsed = json.loads(exported)
        assert parsed["name"] == cdp_template.name

    def test_export_unsupported_format_raises(self, engine, cdp_template):
        with pytest.raises(ValueError, match="[Uu]nsupported"):
            engine.export_template(cdp_template.template_id, format="xml")

    def test_export_non_existing_raises(self, engine):
        with pytest.raises(ValueError, match="[Uu]nknown template"):
            engine.export_template("bad-id")

    def test_import_roundtrip_preserves_name(self, engine, cdp_template):
        exported = engine.export_template(cdp_template.template_id)
        imported = engine.import_template(exported)
        assert imported.name == cdp_template.name

    def test_import_assigns_new_id(self, engine, cdp_template):
        exported = engine.export_template(cdp_template.template_id)
        imported = engine.import_template(exported)
        assert imported.template_id != cdp_template.template_id

    def test_import_resets_version_to_one(self, engine, cdp_template):
        engine.update_template(cdp_template.template_id, {"name": "v2"})
        exported = engine.export_template(cdp_template.template_id)
        imported = engine.import_template(exported)
        assert imported.version == 1

    def test_import_sets_status_to_draft(self, engine, cdp_template):
        exported = engine.export_template(cdp_template.template_id)
        imported = engine.import_template(exported)
        assert imported.status == QuestionnaireStatus.DRAFT

    def test_import_assigns_fresh_section_ids(self, engine, cdp_template):
        exported = engine.export_template(cdp_template.template_id)
        imported = engine.import_template(exported)
        orig_ids = {s.section_id for s in cdp_template.sections}
        imp_ids = {s.section_id for s in imported.sections}
        assert orig_ids.isdisjoint(imp_ids)

    def test_import_invalid_json_raises(self, engine):
        with pytest.raises(ValueError, match="[Ii]nvalid JSON"):
            engine.import_template("{bad json")

    def test_import_unsupported_format_raises(self, engine):
        with pytest.raises(ValueError, match="[Uu]nsupported"):
            engine.import_template("{}", format="yaml")

    def test_import_provenance_hash_set(self, engine, cdp_template):
        exported = engine.export_template(cdp_template.template_id)
        imported = engine.import_template(exported)
        assert len(imported.provenance_hash) == 64

    def test_export_stats_increment(self, engine, cdp_template):
        engine.export_template(cdp_template.template_id)
        stats = engine.get_statistics()
        assert stats["templates_exported"] == 1

    def test_import_stats_increment(self, engine, cdp_template):
        exported = engine.export_template(cdp_template.template_id)
        engine.import_template(exported)
        stats = engine.get_statistics()
        assert stats["templates_imported"] == 1


# ===================================================================
# TEST CLASS: Multi-language Support
# ===================================================================

class TestMultiLanguage:
    """Tests for multi-language template features."""

    def test_template_primary_language(self, engine):
        tmpl = engine.create_template(name="T", framework="custom", language="de")
        assert tmpl.language == "de"

    def test_template_supported_languages_list(self, engine):
        tmpl = engine.create_template(name="T", framework="custom", language="fr")
        assert "fr" in tmpl.supported_languages

    def test_question_translations_field(self, engine, cdp_template):
        section_id = cdp_template.sections[0].section_id
        q = TemplateQuestion(
            code="ML.1", text="English text",
            translations={"de": "Deutscher Text", "fr": "Texte francais"},
        )
        result = engine.add_question(cdp_template.template_id, section_id, q)
        assert "de" in result.translations
        assert "fr" in result.translations


# ===================================================================
# TEST CLASS: Template Versioning
# ===================================================================

class TestVersioning:
    """Tests for template version management."""

    def test_initial_version_is_one(self, engine):
        tmpl = engine.create_template(name="V", framework="custom")
        assert tmpl.version == 1

    def test_version_increments_on_update(self, engine, cdp_template):
        v2 = engine.update_template(cdp_template.template_id, {"name": "v2"})
        assert v2.version == 2

    def test_version_monotonically_increasing(self, engine, cdp_template):
        versions = [cdp_template.version]
        for i in range(5):
            u = engine.update_template(
                cdp_template.template_id, {"name": f"v{i+2}"},
            )
            versions.append(u.version)
        assert versions == sorted(versions)
        assert versions == list(range(1, 7))


# ===================================================================
# TEST CLASS: Conditional Rules
# ===================================================================

class TestConditionalRules:
    """Tests for conditional rules on questions."""

    def test_question_with_validation_rules(self, engine, cdp_template):
        section_id = cdp_template.sections[0].section_id
        q = TemplateQuestion(
            code="COND.1", text="Conditional question",
            validation_rules={"min": 0, "max": 100},
        )
        result = engine.add_question(cdp_template.template_id, section_id, q)
        assert result.validation_rules is not None

    def test_question_without_validation_rules(self, engine, cdp_template):
        section_id = cdp_template.sections[0].section_id
        q = TemplateQuestion(code="PLAIN.1", text="Plain question")
        result = engine.add_question(cdp_template.template_id, section_id, q)
        assert result.code == "PLAIN.1"


# ===================================================================
# TEST CLASS: Provenance
# ===================================================================

class TestProvenance:
    """Tests for SHA-256 provenance hashing on all operations."""

    def test_create_provenance_is_sha256(self, engine):
        tmpl = engine.create_template(name="P", framework="custom")
        assert len(tmpl.provenance_hash) == 64

    def test_update_provenance_changes(self, engine, cdp_template):
        old = cdp_template.provenance_hash
        updated = engine.update_template(cdp_template.template_id, {"name": "X"})
        assert updated.provenance_hash != old

    def test_clone_provenance_set(self, engine, cdp_template):
        cloned = engine.clone_template(cdp_template.template_id, "C")
        assert len(cloned.provenance_hash) == 64

    def test_add_section_provenance_changes(self, engine, cdp_template, custom_section):
        old = engine.get_template(cdp_template.template_id).provenance_hash
        engine.add_section(cdp_template.template_id, custom_section)
        new = engine.get_template(cdp_template.template_id).provenance_hash
        assert new != old

    def test_validate_provenance_set(self, engine, cdp_template):
        summary = engine.validate_template(cdp_template.template_id)
        assert len(summary.provenance_hash) == 64

    def test_import_provenance_set(self, engine, cdp_template):
        exported = engine.export_template(cdp_template.template_id)
        imported = engine.import_template(exported)
        assert len(imported.provenance_hash) == 64


# ===================================================================
# TEST CLASS: Statistics
# ===================================================================

class TestStatistics:
    """Tests for get_statistics()."""

    def test_statistics_initial_zeros(self, engine):
        stats = engine.get_statistics()
        assert stats["templates_created"] == 0
        assert stats["templates_updated"] == 0
        assert stats["active_templates"] == 0

    def test_statistics_active_templates_count(self, engine):
        engine.create_template(name="A", framework="custom")
        engine.create_template(name="B", framework="custom")
        stats = engine.get_statistics()
        assert stats["active_templates"] == 2

    def test_statistics_has_timestamp(self, engine):
        stats = engine.get_statistics()
        assert "timestamp" in stats


# ===================================================================
# TEST CLASS: Thread Safety
# ===================================================================

class TestThreadSafety:
    """Tests for thread-safe operations."""

    def test_concurrent_creates(self, engine):
        errors = []

        def create(idx):
            try:
                engine.create_template(name=f"T-{idx}", framework="custom")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=create, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(engine.list_templates()) == 20
