# -*- coding: utf-8 -*-
"""
Unit tests for ResponseCollectorEngine
========================================

AGENT-DATA-008: Supplier Questionnaire Processor
Tests all methods of ResponseCollectorEngine with comprehensive coverage.
Validates response submission, partial save, finalisation, bulk import,
deduplication, acknowledgement, progress tracking, answer normalisation,
status transitions, and SHA-256 provenance hashing.

Total: ~70 tests
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List

import pytest

from greenlang.supplier_questionnaire.response_collector import (
    ResponseCollectorEngine,
)
from greenlang.supplier_questionnaire.models import (
    Answer,
    QuestionnaireResponse,
    QuestionnaireTemplate,
    ResponseStatus,
    TemplateQuestion,
    TemplateSection,
)


# ===================================================================
# Fixtures
# ===================================================================

@pytest.fixture
def engine() -> ResponseCollectorEngine:
    """Fresh ResponseCollectorEngine per test."""
    return ResponseCollectorEngine()


@pytest.fixture
def engine_no_normalize() -> ResponseCollectorEngine:
    """Engine with normalisation disabled."""
    return ResponseCollectorEngine({"normalize_values": False})


@pytest.fixture
def engine_no_ack() -> ResponseCollectorEngine:
    """Engine with auto_acknowledge disabled."""
    return ResponseCollectorEngine({"auto_acknowledge": False})


@pytest.fixture
def basic_answers() -> List[Dict[str, Any]]:
    """Simple set of answer dicts."""
    return [
        {"question_id": "q1", "value": "42"},
        {"question_id": "q2", "value": "Yes"},
        {"question_id": "q3", "value": "Some text answer"},
    ]


@pytest.fixture
def basic_response(engine, basic_answers) -> QuestionnaireResponse:
    """Submit a basic response and return it."""
    return engine.submit_response(
        distribution_id="dist-001",
        answers=basic_answers,
        template_id="tmpl-001",
        supplier_id="SUP001",
    )


@pytest.fixture
def sample_template() -> QuestionnaireTemplate:
    """A small template with 2 sections and 5 questions for progress testing."""
    q1 = TemplateQuestion(question_id="q1", code="Q1", text="Q1", required=True)
    q2 = TemplateQuestion(question_id="q2", code="Q2", text="Q2", required=True)
    q3 = TemplateQuestion(question_id="q3", code="Q3", text="Q3", required=False)
    q4 = TemplateQuestion(question_id="q4", code="Q4", text="Q4", required=True)
    q5 = TemplateQuestion(question_id="q5", code="Q5", text="Q5", required=True)

    sec1 = TemplateSection(
        section_id="sec1", name="Section 1",
        questions=[q1, q2, q3],
    )
    sec2 = TemplateSection(
        section_id="sec2", name="Section 2",
        questions=[q4, q5],
    )

    return QuestionnaireTemplate(
        template_id="tmpl-001", name="Test Template",
        framework="custom",
        sections=[sec1, sec2],
    )


# ===================================================================
# TEST CLASS: Submit Response
# ===================================================================

class TestSubmitResponse:
    """Tests for ResponseCollectorEngine.submit_response()."""

    def test_submit_returns_response(self, engine, basic_answers):
        resp = engine.submit_response(
            distribution_id="dist-001", answers=basic_answers,
        )
        assert isinstance(resp, QuestionnaireResponse)

    def test_submit_response_id_generated(self, engine, basic_answers):
        resp = engine.submit_response(
            distribution_id="dist-001", answers=basic_answers,
        )
        assert len(resp.response_id) == 36  # UUID format

    def test_submit_status_is_in_progress(self, engine, basic_answers):
        resp = engine.submit_response(
            distribution_id="dist-001", answers=basic_answers,
        )
        assert resp.status == ResponseStatus.IN_PROGRESS

    def test_submit_distribution_id_stored(self, engine, basic_answers):
        resp = engine.submit_response(
            distribution_id="dist-001", answers=basic_answers,
        )
        assert resp.distribution_id == "dist-001"

    def test_submit_template_id_stored(self, engine, basic_answers):
        resp = engine.submit_response(
            distribution_id="dist-001", answers=basic_answers,
            template_id="tmpl-001",
        )
        assert resp.template_id == "tmpl-001"

    def test_submit_supplier_id_stored(self, engine, basic_answers):
        resp = engine.submit_response(
            distribution_id="dist-001", answers=basic_answers,
            supplier_id="SUP001",
        )
        assert resp.supplier_id == "SUP001"

    def test_submit_supplier_id_auto_generated(self, engine, basic_answers):
        resp = engine.submit_response(
            distribution_id="dist-001", answers=basic_answers,
        )
        assert resp.supplier_id.startswith("supplier-")

    def test_submit_language_default_en(self, engine, basic_answers):
        resp = engine.submit_response(
            distribution_id="dist-001", answers=basic_answers,
        )
        assert resp.language == "en"

    def test_submit_language_override(self, engine, basic_answers):
        resp = engine.submit_response(
            distribution_id="dist-001", answers=basic_answers,
            language="de",
        )
        assert resp.language == "de"

    def test_submit_empty_distribution_raises(self, engine):
        with pytest.raises(ValueError, match="non-empty"):
            engine.submit_response(distribution_id="", answers=[])

    def test_submit_whitespace_distribution_raises(self, engine):
        with pytest.raises(ValueError, match="non-empty"):
            engine.submit_response(distribution_id="  ", answers=[])

    def test_submit_max_responses_enforced(self):
        eng = ResponseCollectorEngine({"max_responses": 2})
        eng.submit_response(
            distribution_id="d1",
            answers=[{"question_id": "q1", "value": "a"}],
        )
        eng.submit_response(
            distribution_id="d2",
            answers=[{"question_id": "q2", "value": "b"}],
        )
        with pytest.raises(ValueError, match="[Mm]aximum responses"):
            eng.submit_response(
                distribution_id="d3",
                answers=[{"question_id": "q3", "value": "c"}],
            )

    def test_submit_provenance_hash_set(self, engine, basic_answers):
        resp = engine.submit_response(
            distribution_id="dist-001", answers=basic_answers,
        )
        assert len(resp.provenance_hash) == 64

    def test_submit_skips_empty_question_id(self, engine):
        answers = [
            {"question_id": "", "value": "skip me"},
            {"question_id": "q1", "value": "keep me"},
        ]
        resp = engine.submit_response(
            distribution_id="dist-001", answers=answers,
        )
        assert len(resp.answers) == 1
        assert resp.answers[0].question_id == "q1"

    def test_submit_stats_increment(self, engine, basic_answers):
        engine.submit_response(distribution_id="d1", answers=basic_answers)
        stats = engine.get_statistics()
        assert stats["responses_submitted"] == 1


# ===================================================================
# TEST CLASS: Get Response
# ===================================================================

class TestGetResponse:
    """Tests for ResponseCollectorEngine.get_response()."""

    def test_get_response_existing(self, engine, basic_response):
        fetched = engine.get_response(basic_response.response_id)
        assert fetched.response_id == basic_response.response_id

    def test_get_response_non_existing_raises(self, engine):
        with pytest.raises(ValueError, match="[Uu]nknown response"):
            engine.get_response("non-existent-id")

    def test_get_response_preserves_answers(self, engine, basic_response):
        fetched = engine.get_response(basic_response.response_id)
        assert len(fetched.answers) == len(basic_response.answers)


# ===================================================================
# TEST CLASS: List Responses
# ===================================================================

class TestListResponses:
    """Tests for ResponseCollectorEngine.list_responses()."""

    def test_list_responses_empty(self, engine):
        assert engine.list_responses() == []

    def test_list_responses_returns_all(self, engine, basic_answers):
        engine.submit_response(distribution_id="d1", answers=basic_answers)
        engine.submit_response(distribution_id="d2", answers=basic_answers)
        assert len(engine.list_responses()) == 2

    def test_list_responses_filter_template(self, engine, basic_answers):
        engine.submit_response(
            distribution_id="d1", answers=basic_answers, template_id="t1",
        )
        engine.submit_response(
            distribution_id="d2", answers=basic_answers, template_id="t2",
        )
        filtered = engine.list_responses(template_id="t1")
        assert len(filtered) == 1
        assert filtered[0].template_id == "t1"

    def test_list_responses_filter_supplier(self, engine, basic_answers):
        engine.submit_response(
            distribution_id="d1", answers=basic_answers, supplier_id="S1",
        )
        engine.submit_response(
            distribution_id="d2", answers=basic_answers, supplier_id="S2",
        )
        filtered = engine.list_responses(supplier_id="S1")
        assert len(filtered) == 1

    def test_list_responses_filter_status(self, engine, basic_answers):
        r = engine.submit_response(
            distribution_id="d1", answers=basic_answers,
        )
        engine.finalize_response(r.response_id)
        engine.submit_response(distribution_id="d2", answers=basic_answers)
        submitted = engine.list_responses(status="submitted")
        assert len(submitted) == 1

    def test_list_responses_no_match(self, engine, basic_response):
        assert engine.list_responses(template_id="no-such") == []


# ===================================================================
# TEST CLASS: Update Response (Partial Save)
# ===================================================================

class TestUpdateResponse:
    """Tests for ResponseCollectorEngine.update_response()."""

    def test_update_response_adds_answers(self, engine, basic_response):
        updated = engine.update_response(
            basic_response.response_id,
            [{"question_id": "q4", "value": "new answer"}],
        )
        qids = {a.question_id for a in updated.answers}
        assert "q4" in qids

    def test_update_response_replaces_existing(self, engine, basic_response):
        updated = engine.update_response(
            basic_response.response_id,
            [{"question_id": "q1", "value": "replaced"}],
        )
        q1_answer = next(a for a in updated.answers if a.question_id == "q1")
        assert q1_answer.value == "replaced"

    def test_update_response_sets_status_in_progress(self, engine, basic_response):
        updated = engine.update_response(
            basic_response.response_id,
            [{"question_id": "q4", "value": "x"}],
        )
        assert updated.status == ResponseStatus.IN_PROGRESS

    def test_update_response_non_editable_raises(self, engine, basic_response):
        engine.finalize_response(basic_response.response_id)
        with pytest.raises(ValueError, match="not editable"):
            engine.update_response(
                basic_response.response_id,
                [{"question_id": "q1", "value": "oops"}],
            )

    def test_update_response_non_existing_raises(self, engine):
        with pytest.raises(ValueError, match="[Uu]nknown response"):
            engine.update_response("bad-id", [])

    def test_update_response_provenance_changes(self, engine, basic_response):
        old_hash = basic_response.provenance_hash
        updated = engine.update_response(
            basic_response.response_id,
            [{"question_id": "q5", "value": "v"}],
        )
        assert updated.provenance_hash != old_hash

    def test_update_response_stats_increment(self, engine, basic_response):
        engine.update_response(
            basic_response.response_id,
            [{"question_id": "q5", "value": "v"}],
        )
        stats = engine.get_statistics()
        assert stats["responses_updated"] >= 1


# ===================================================================
# TEST CLASS: Finalize Response
# ===================================================================

class TestFinalizeResponse:
    """Tests for ResponseCollectorEngine.finalize_response()."""

    def test_finalize_sets_submitted_status(self, engine, basic_response):
        finalized = engine.finalize_response(basic_response.response_id)
        assert finalized.status == ResponseStatus.SUBMITTED

    def test_finalize_sets_submitted_at(self, engine, basic_response):
        finalized = engine.finalize_response(basic_response.response_id)
        assert finalized.submitted_at is not None

    def test_finalize_auto_acknowledge(self, engine, basic_response):
        finalized = engine.finalize_response(basic_response.response_id)
        assert finalized.confirmation_token != ""

    def test_finalize_no_auto_acknowledge(self, engine_no_ack):
        resp = engine_no_ack.submit_response(
            distribution_id="d1",
            answers=[{"question_id": "q1", "value": "v"}],
        )
        finalized = engine_no_ack.finalize_response(resp.response_id)
        assert finalized.confirmation_token == ""

    def test_finalize_already_submitted_raises(self, engine, basic_response):
        engine.finalize_response(basic_response.response_id)
        with pytest.raises(ValueError, match="cannot be finalized"):
            engine.finalize_response(basic_response.response_id)

    def test_finalize_non_existing_raises(self, engine):
        with pytest.raises(ValueError, match="[Uu]nknown response"):
            engine.finalize_response("bad-id")

    def test_finalize_stats_increment(self, engine, basic_response):
        engine.finalize_response(basic_response.response_id)
        stats = engine.get_statistics()
        assert stats["responses_finalized"] == 1


# ===================================================================
# TEST CLASS: Import Responses Bulk
# ===================================================================

class TestImportResponsesBulk:
    """Tests for ResponseCollectorEngine.import_responses_bulk()."""

    def test_import_bulk_returns_list(self, engine):
        file_data = [
            {"distribution_id": "d1", "supplier_id": "S1",
             "answers": [{"question_id": "q1", "value": "a"}]},
            {"distribution_id": "d2", "supplier_id": "S2",
             "answers": [{"question_id": "q1", "value": "b"}]},
        ]
        results = engine.import_responses_bulk(file_data, "tmpl-001")
        assert len(results) == 2

    def test_import_bulk_creates_responses(self, engine):
        file_data = [
            {"distribution_id": "d1", "supplier_id": "S1",
             "answers": [{"question_id": "q1", "value": "a"}]},
        ]
        results = engine.import_responses_bulk(file_data, "tmpl-001")
        assert results[0].template_id == "tmpl-001"

    def test_import_bulk_empty_raises(self, engine):
        with pytest.raises(ValueError, match="non-empty"):
            engine.import_responses_bulk([], "tmpl-001")

    def test_import_bulk_stats_increment(self, engine):
        file_data = [
            {"distribution_id": "d1", "supplier_id": "S1",
             "answers": [{"question_id": "q1", "value": "a"}]},
            {"distribution_id": "d2", "supplier_id": "S2",
             "answers": [{"question_id": "q1", "value": "b"}]},
        ]
        engine.import_responses_bulk(file_data, "tmpl-001")
        stats = engine.get_statistics()
        assert stats["responses_imported"] == 2

    def test_import_bulk_with_language(self, engine):
        file_data = [
            {"distribution_id": "d1", "supplier_id": "S1",
             "answers": [{"question_id": "q1", "value": "a"}],
             "language": "de"},
        ]
        results = engine.import_responses_bulk(file_data, "tmpl-001")
        assert results[0].language == "de"

    def test_import_bulk_generates_dist_id_if_missing(self, engine):
        file_data = [
            {"supplier_id": "S1",
             "answers": [{"question_id": "q1", "value": "a"}]},
        ]
        results = engine.import_responses_bulk(file_data, "tmpl-001")
        assert results[0].distribution_id != ""


# ===================================================================
# TEST CLASS: Reopen Response
# ===================================================================

class TestReopenResponse:
    """Tests for ResponseCollectorEngine.reopen_response()."""

    def test_reopen_submitted_response(self, engine, basic_response):
        engine.finalize_response(basic_response.response_id)
        reopened = engine.reopen_response(
            basic_response.response_id, "Data correction needed",
        )
        assert reopened.status == ResponseStatus.REOPENED

    def test_reopen_non_submitted_raises(self, engine, basic_response):
        with pytest.raises(ValueError, match="cannot be reopened"):
            engine.reopen_response(basic_response.response_id)

    def test_reopen_non_existing_raises(self, engine):
        with pytest.raises(ValueError, match="[Uu]nknown response"):
            engine.reopen_response("bad-id")

    def test_reopen_stats_increment(self, engine, basic_response):
        engine.finalize_response(basic_response.response_id)
        engine.reopen_response(basic_response.response_id)
        stats = engine.get_statistics()
        assert stats["responses_reopened"] == 1

    def test_reopen_allows_subsequent_update(self, engine, basic_response):
        engine.finalize_response(basic_response.response_id)
        engine.reopen_response(basic_response.response_id)
        updated = engine.update_response(
            basic_response.response_id,
            [{"question_id": "q99", "value": "new data"}],
        )
        qids = {a.question_id for a in updated.answers}
        assert "q99" in qids


# ===================================================================
# TEST CLASS: Get Response Progress
# ===================================================================

class TestGetResponseProgress:
    """Tests for ResponseCollectorEngine.get_response_progress()."""

    def test_progress_without_template(self, engine, basic_response):
        progress = engine.get_response_progress(basic_response.response_id)
        assert "total_answers" in progress
        assert progress["total_answers"] == len(basic_response.answers)

    def test_progress_with_template(self, engine, basic_response, sample_template):
        progress = engine.get_response_progress(
            basic_response.response_id, template=sample_template,
        )
        assert "total_questions" in progress
        assert "required_questions" in progress
        assert "completion_pct" in progress

    def test_progress_completion_pct_correct(self, engine, sample_template):
        answers = [
            {"question_id": "q1", "value": "a"},
            {"question_id": "q2", "value": "b"},
            {"question_id": "q4", "value": "c"},
            {"question_id": "q5", "value": "d"},
        ]
        resp = engine.submit_response(
            distribution_id="d1", answers=answers,
        )
        progress = engine.get_response_progress(
            resp.response_id, template=sample_template,
        )
        assert progress["completion_pct"] == 100.0

    def test_progress_partial_completion(self, engine, sample_template):
        answers = [
            {"question_id": "q1", "value": "a"},
            {"question_id": "q2", "value": "b"},
        ]
        resp = engine.submit_response(
            distribution_id="d1", answers=answers,
        )
        progress = engine.get_response_progress(
            resp.response_id, template=sample_template,
        )
        assert progress["completion_pct"] == 50.0

    def test_progress_section_breakdown(self, engine, basic_response, sample_template):
        progress = engine.get_response_progress(
            basic_response.response_id, template=sample_template,
        )
        assert "section_progress" in progress
        assert "sec1" in progress["section_progress"]
        assert "sec2" in progress["section_progress"]

    def test_progress_non_existing_raises(self, engine):
        with pytest.raises(ValueError, match="[Uu]nknown response"):
            engine.get_response_progress("bad-id")

    def test_progress_updates_stored_completion(self, engine, sample_template):
        answers = [
            {"question_id": "q1", "value": "a"},
            {"question_id": "q2", "value": "b"},
        ]
        resp = engine.submit_response(
            distribution_id="d1", answers=answers,
        )
        engine.get_response_progress(resp.response_id, template=sample_template)
        refreshed = engine.get_response(resp.response_id)
        assert refreshed.completion_pct == 50.0


# ===================================================================
# TEST CLASS: Deduplicate Responses
# ===================================================================

class TestDeduplicateResponses:
    """Tests for ResponseCollectorEngine.deduplicate_responses()."""

    def test_deduplicate_finds_duplicates(self, engine, basic_answers):
        engine.submit_response(
            distribution_id="d1", answers=basic_answers,
            template_id="t1", supplier_id="S1",
        )
        engine.submit_response(
            distribution_id="d2", answers=basic_answers,
            template_id="t1", supplier_id="S1",
        )
        dupes = engine.deduplicate_responses("S1", "t1")
        assert len(dupes) == 2

    def test_deduplicate_sorted_by_updated_at(self, engine, basic_answers):
        engine.submit_response(
            distribution_id="d1", answers=basic_answers,
            template_id="t1", supplier_id="S1",
        )
        engine.submit_response(
            distribution_id="d2", answers=basic_answers,
            template_id="t1", supplier_id="S1",
        )
        dupes = engine.deduplicate_responses("S1", "t1")
        assert len(dupes) == 2

    def test_deduplicate_no_duplicates(self, engine, basic_answers):
        engine.submit_response(
            distribution_id="d1", answers=basic_answers,
            template_id="t1", supplier_id="S1",
        )
        dupes = engine.deduplicate_responses("S1", "t1")
        assert len(dupes) == 1

    def test_deduplicate_different_supplier_no_match(self, engine, basic_answers):
        engine.submit_response(
            distribution_id="d1", answers=basic_answers,
            template_id="t1", supplier_id="S1",
        )
        engine.submit_response(
            distribution_id="d2", answers=basic_answers,
            template_id="t1", supplier_id="S2",
        )
        dupes = engine.deduplicate_responses("S1", "t1")
        assert len(dupes) == 1

    def test_deduplicate_stats_increment_on_found(self, engine, basic_answers):
        engine.submit_response(
            distribution_id="d1", answers=basic_answers,
            template_id="t1", supplier_id="S1",
        )
        engine.submit_response(
            distribution_id="d2", answers=basic_answers,
            template_id="t1", supplier_id="S1",
        )
        engine.deduplicate_responses("S1", "t1")
        stats = engine.get_statistics()
        assert stats["responses_deduplicated"] >= 1

    def test_deduplicate_empty_results(self, engine):
        dupes = engine.deduplicate_responses("nobody", "nothing")
        assert dupes == []


# ===================================================================
# TEST CLASS: Acknowledge Response
# ===================================================================

class TestAcknowledgeResponse:
    """Tests for ResponseCollectorEngine.acknowledge_response()."""

    def test_acknowledge_returns_token(self, engine, basic_response):
        token = engine.acknowledge_response(basic_response.response_id)
        assert len(token) == 64
        assert re.fullmatch(r"[0-9a-f]{64}", token)

    def test_acknowledge_token_stored(self, engine, basic_response):
        token = engine.acknowledge_response(basic_response.response_id)
        refreshed = engine.get_response(basic_response.response_id)
        assert refreshed.confirmation_token == token

    def test_acknowledge_non_existing_raises(self, engine):
        with pytest.raises(ValueError, match="[Uu]nknown response"):
            engine.acknowledge_response("bad-id")

    def test_acknowledge_stats_increment(self, engine, basic_response):
        engine.acknowledge_response(basic_response.response_id)
        stats = engine.get_statistics()
        assert stats["acknowledgements_sent"] >= 1


# ===================================================================
# TEST CLASS: Answer Normalisation
# ===================================================================

class TestAnswerNormalisation:
    """Tests for value normalisation in _parse_answers and _normalize_value."""

    def test_normalize_yes_to_true(self, engine):
        resp = engine.submit_response(
            distribution_id="d1",
            answers=[{"question_id": "q1", "value": "Yes"}],
        )
        assert resp.answers[0].value is True

    def test_normalize_no_to_false(self, engine):
        resp = engine.submit_response(
            distribution_id="d1",
            answers=[{"question_id": "q1", "value": "No"}],
        )
        assert resp.answers[0].value is False

    def test_normalize_y_to_true(self, engine):
        resp = engine.submit_response(
            distribution_id="d1",
            answers=[{"question_id": "q1", "value": "y"}],
        )
        assert resp.answers[0].value is True

    def test_normalize_n_to_false(self, engine):
        resp = engine.submit_response(
            distribution_id="d1",
            answers=[{"question_id": "q1", "value": "n"}],
        )
        assert resp.answers[0].value is False

    def test_normalize_true_string(self, engine):
        resp = engine.submit_response(
            distribution_id="d1",
            answers=[{"question_id": "q1", "value": "true"}],
        )
        assert resp.answers[0].value is True

    def test_normalize_false_string(self, engine):
        resp = engine.submit_response(
            distribution_id="d1",
            answers=[{"question_id": "q1", "value": "false"}],
        )
        assert resp.answers[0].value is False

    def test_normalize_numeric_integer(self, engine):
        resp = engine.submit_response(
            distribution_id="d1",
            answers=[{"question_id": "q1", "value": "42"}],
        )
        assert resp.answers[0].value == 42

    def test_normalize_numeric_float(self, engine):
        resp = engine.submit_response(
            distribution_id="d1",
            answers=[{"question_id": "q1", "value": "3.14"}],
        )
        assert resp.answers[0].value == pytest.approx(3.14)

    def test_normalize_numeric_with_commas(self, engine):
        resp = engine.submit_response(
            distribution_id="d1",
            answers=[{"question_id": "q1", "value": "1,234,567"}],
        )
        assert resp.answers[0].value == 1234567

    def test_normalize_numeric_with_unit_tco2e(self, engine):
        resp = engine.submit_response(
            distribution_id="d1",
            answers=[{"question_id": "q1", "value": "500 tCO2e"}],
        )
        assert resp.answers[0].value == 500
        assert resp.answers[0].unit == "tCO2e"

    def test_normalize_numeric_with_unit_mwh(self, engine):
        resp = engine.submit_response(
            distribution_id="d1",
            answers=[{"question_id": "q1", "value": "1200 MWh"}],
        )
        assert resp.answers[0].value == 1200
        assert resp.answers[0].unit == "MWh"

    def test_normalize_percentage(self, engine):
        resp = engine.submit_response(
            distribution_id="d1",
            answers=[{"question_id": "q1", "value": "85.5%"}],
        )
        assert resp.answers[0].value == pytest.approx(85.5)
        assert resp.answers[0].unit == "%"

    def test_normalize_none_value(self, engine):
        resp = engine.submit_response(
            distribution_id="d1",
            answers=[{"question_id": "q1", "value": None}],
        )
        assert resp.answers[0].value == ""

    def test_normalize_int_passthrough(self, engine):
        resp = engine.submit_response(
            distribution_id="d1",
            answers=[{"question_id": "q1", "value": 42}],
        )
        assert resp.answers[0].value == 42

    def test_normalize_float_passthrough(self, engine):
        resp = engine.submit_response(
            distribution_id="d1",
            answers=[{"question_id": "q1", "value": 3.14}],
        )
        assert resp.answers[0].value == pytest.approx(3.14)

    def test_normalize_list_passthrough(self, engine):
        resp = engine.submit_response(
            distribution_id="d1",
            answers=[{"question_id": "q1", "value": ["a", "b"]}],
        )
        assert resp.answers[0].value == ["a", "b"]

    def test_normalize_plain_text_passthrough(self, engine):
        resp = engine.submit_response(
            distribution_id="d1",
            answers=[{"question_id": "q1", "value": "Hello world"}],
        )
        assert resp.answers[0].value == "Hello world"

    def test_normalize_disabled(self, engine_no_normalize):
        resp = engine_no_normalize.submit_response(
            distribution_id="d1",
            answers=[{"question_id": "q1", "value": "Yes"}],
        )
        assert resp.answers[0].value == "Yes"

    def test_confidence_clamped_above_one(self, engine):
        resp = engine.submit_response(
            distribution_id="d1",
            answers=[{"question_id": "q1", "value": "v", "confidence": 5.0}],
        )
        assert resp.answers[0].confidence == 1.0

    def test_confidence_clamped_below_zero(self, engine):
        resp = engine.submit_response(
            distribution_id="d1",
            answers=[{"question_id": "q1", "value": "v", "confidence": -2.0}],
        )
        assert resp.answers[0].confidence == 0.0

    def test_evidence_refs_stored(self, engine):
        resp = engine.submit_response(
            distribution_id="d1",
            answers=[{"question_id": "q1", "value": "v",
                       "evidence_refs": ["ref1", "ref2"]}],
        )
        assert resp.answers[0].evidence_refs == ["ref1", "ref2"]

    def test_notes_stored(self, engine):
        resp = engine.submit_response(
            distribution_id="d1",
            answers=[{"question_id": "q1", "value": "v",
                       "notes": "Some notes"}],
        )
        assert resp.answers[0].notes == "Some notes"

    def test_unit_preserved_when_provided(self, engine):
        resp = engine.submit_response(
            distribution_id="d1",
            answers=[{"question_id": "q1", "value": "100", "unit": "kg"}],
        )
        assert resp.answers[0].unit == "kg"


# ===================================================================
# TEST CLASS: Status Transitions
# ===================================================================

class TestStatusTransitions:
    """Tests for valid and invalid status transitions."""

    def test_in_progress_to_submitted_via_finalize(self, engine, basic_response):
        finalized = engine.finalize_response(basic_response.response_id)
        assert finalized.status == ResponseStatus.SUBMITTED

    def test_submitted_to_reopened(self, engine, basic_response):
        engine.finalize_response(basic_response.response_id)
        reopened = engine.reopen_response(basic_response.response_id)
        assert reopened.status == ResponseStatus.REOPENED

    def test_reopened_to_submitted(self, engine, basic_response):
        engine.finalize_response(basic_response.response_id)
        engine.reopen_response(basic_response.response_id)
        finalized_again = engine.finalize_response(basic_response.response_id)
        assert finalized_again.status == ResponseStatus.SUBMITTED

    def test_reopened_allows_update(self, engine, basic_response):
        engine.finalize_response(basic_response.response_id)
        engine.reopen_response(basic_response.response_id)
        updated = engine.update_response(
            basic_response.response_id,
            [{"question_id": "q_new", "value": "data"}],
        )
        assert updated.status == ResponseStatus.IN_PROGRESS


# ===================================================================
# TEST CLASS: Provenance on All Operations
# ===================================================================

class TestProvenance:
    """Tests for SHA-256 provenance hashing on all operations."""

    def test_submit_provenance(self, engine, basic_answers):
        resp = engine.submit_response(
            distribution_id="d1", answers=basic_answers,
        )
        assert len(resp.provenance_hash) == 64
        assert re.fullmatch(r"[0-9a-f]{64}", resp.provenance_hash)

    def test_update_provenance_changes(self, engine, basic_response):
        old = basic_response.provenance_hash
        updated = engine.update_response(
            basic_response.response_id,
            [{"question_id": "q99", "value": "v"}],
        )
        assert updated.provenance_hash != old

    def test_finalize_provenance_set(self, engine, basic_response):
        finalized = engine.finalize_response(basic_response.response_id)
        assert len(finalized.provenance_hash) == 64

    def test_reopen_provenance_set(self, engine, basic_response):
        engine.finalize_response(basic_response.response_id)
        reopened = engine.reopen_response(basic_response.response_id)
        assert len(reopened.provenance_hash) == 64


# ===================================================================
# TEST CLASS: Statistics
# ===================================================================

class TestStatistics:
    """Tests for get_statistics()."""

    def test_statistics_initial_zeros(self, engine):
        stats = engine.get_statistics()
        assert stats["responses_submitted"] == 0
        assert stats["active_responses"] == 0

    def test_statistics_active_responses(self, engine, basic_answers):
        engine.submit_response(distribution_id="d1", answers=basic_answers)
        engine.submit_response(distribution_id="d2", answers=basic_answers)
        stats = engine.get_statistics()
        assert stats["active_responses"] == 2

    def test_statistics_has_timestamp(self, engine):
        stats = engine.get_statistics()
        assert "timestamp" in stats


# ===================================================================
# TEST CLASS: Thread Safety
# ===================================================================

class TestThreadSafety:
    """Tests for thread-safe operations."""

    def test_concurrent_submits(self, engine):
        errors = []

        def submit(idx):
            try:
                engine.submit_response(
                    distribution_id=f"d-{idx}",
                    answers=[{"question_id": "q1", "value": f"v{idx}"}],
                )
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=submit, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(engine.list_responses()) == 20
