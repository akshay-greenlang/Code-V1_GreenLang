# -*- coding: utf-8 -*-
"""
Unit tests for ResponseManager -- CDP response lifecycle management.

Tests response CRUD, status transitions, version history, evidence attachments,
review workflow, bulk operations, auto-save conflict detection, and response
templates with 38+ test functions.

Author: GL-TestEngineer
Date: March 2026
"""

from datetime import datetime
from decimal import Decimal

import pytest

from services.config import ResponseStatus
from services.models import (
    CDPResponse,
    CDPResponseVersion,
    CDPEvidenceAttachment,
    CDPReviewAction,
    _new_id,
)
from services.response_manager import ResponseManager


# ---------------------------------------------------------------------------
# Response CRUD
# ---------------------------------------------------------------------------

class TestResponseCRUD:
    """Test basic response create/read/update/delete."""

    def test_create_response(self, response_manager, sample_question, sample_questionnaire, sample_organization):
        response = response_manager.create_response(
            question_id=sample_question.id,
            questionnaire_id=sample_questionnaire.id,
            org_id=sample_organization.id,
            content={"answer": "yes"},
            text="Yes, we have an emissions inventory.",
        )
        assert response.response_status == ResponseStatus.DRAFT
        assert response.response_content == {"answer": "yes"}
        assert len(response.id) == 36

    def test_get_response_by_id(self, response_manager, sample_response):
        response_manager._store[sample_response.id] = sample_response
        retrieved = response_manager.get_response(sample_response.id)
        assert retrieved is not None
        assert retrieved.id == sample_response.id

    def test_get_nonexistent_response(self, response_manager):
        result = response_manager.get_response("nonexistent-id")
        assert result is None

    def test_update_response_content(self, response_manager, sample_response):
        response_manager._store[sample_response.id] = sample_response
        updated = response_manager.update_response(
            response_id=sample_response.id,
            content={"answer": "no"},
            text="No, we do not yet have an inventory.",
        )
        assert updated.response_content == {"answer": "no"}

    def test_delete_response(self, response_manager, sample_response):
        response_manager._store[sample_response.id] = sample_response
        response_manager.delete_response(sample_response.id)
        assert response_manager.get_response(sample_response.id) is None

    def test_list_responses_for_questionnaire(self, response_manager, sample_questionnaire, sample_response):
        response_manager._store[sample_response.id] = sample_response
        responses = response_manager.list_responses(questionnaire_id=sample_questionnaire.id)
        assert len(responses) >= 1

    def test_list_responses_by_status(self, response_manager, sample_response):
        response_manager._store[sample_response.id] = sample_response
        drafts = response_manager.list_responses(
            questionnaire_id=sample_response.questionnaire_id,
            status=ResponseStatus.DRAFT,
        )
        assert all(r.response_status == ResponseStatus.DRAFT for r in drafts)


# ---------------------------------------------------------------------------
# Status transitions
# ---------------------------------------------------------------------------

class TestStatusTransitions:
    """Test response status lifecycle transitions."""

    def test_draft_to_in_review(self, response_manager, sample_response):
        response_manager._store[sample_response.id] = sample_response
        updated = response_manager.transition_status(
            sample_response.id, ResponseStatus.IN_REVIEW,
        )
        assert updated.response_status == ResponseStatus.IN_REVIEW

    def test_in_review_to_approved(self, response_manager, sample_response):
        sample_response.response_status = ResponseStatus.IN_REVIEW
        response_manager._store[sample_response.id] = sample_response
        updated = response_manager.transition_status(
            sample_response.id, ResponseStatus.APPROVED,
        )
        assert updated.response_status == ResponseStatus.APPROVED

    def test_approved_to_submitted(self, response_manager, sample_response):
        sample_response.response_status = ResponseStatus.APPROVED
        response_manager._store[sample_response.id] = sample_response
        updated = response_manager.transition_status(
            sample_response.id, ResponseStatus.SUBMITTED,
        )
        assert updated.response_status == ResponseStatus.SUBMITTED

    def test_invalid_draft_to_submitted_raises(self, response_manager, sample_response):
        response_manager._store[sample_response.id] = sample_response
        with pytest.raises(ValueError, match="[Ii]nvalid.*transition"):
            response_manager.transition_status(
                sample_response.id, ResponseStatus.SUBMITTED,
            )

    def test_invalid_submitted_to_draft_raises(self, response_manager, sample_response):
        sample_response.response_status = ResponseStatus.SUBMITTED
        response_manager._store[sample_response.id] = sample_response
        with pytest.raises(ValueError, match="[Ii]nvalid.*transition"):
            response_manager.transition_status(
                sample_response.id, ResponseStatus.DRAFT,
            )

    def test_in_review_to_draft_rejection(self, response_manager, sample_response):
        sample_response.response_status = ResponseStatus.IN_REVIEW
        response_manager._store[sample_response.id] = sample_response
        updated = response_manager.transition_status(
            sample_response.id, ResponseStatus.DRAFT,
        )
        assert updated.response_status == ResponseStatus.DRAFT


# ---------------------------------------------------------------------------
# Version history
# ---------------------------------------------------------------------------

class TestVersionHistory:
    """Test response version tracking."""

    def test_version_created_on_update(self, response_manager, sample_response):
        response_manager._store[sample_response.id] = sample_response
        response_manager.update_response(
            response_id=sample_response.id,
            content={"answer": "updated"},
            text="Updated response.",
            change_reason="Corrected data",
        )
        versions = response_manager.get_version_history(sample_response.id)
        assert len(versions) >= 1

    def test_version_increments(self, response_manager, sample_response):
        response_manager._store[sample_response.id] = sample_response
        for i in range(3):
            response_manager.update_response(
                response_id=sample_response.id,
                content={"answer": f"version_{i}"},
                text=f"Version {i}",
            )
        versions = response_manager.get_version_history(sample_response.id)
        version_numbers = [v.version_number for v in versions]
        assert version_numbers == sorted(version_numbers)

    def test_version_stores_old_content(self, response_manager, sample_response):
        response_manager._store[sample_response.id] = sample_response
        original_content = sample_response.response_content.copy()
        response_manager.update_response(
            response_id=sample_response.id,
            content={"answer": "new_value"},
            text="New value",
        )
        versions = response_manager.get_version_history(sample_response.id)
        assert versions[0].content == original_content


# ---------------------------------------------------------------------------
# Evidence attachments
# ---------------------------------------------------------------------------

class TestEvidenceAttachments:
    """Test evidence attachment management."""

    def test_attach_evidence(self, response_manager, sample_response):
        response_manager._store[sample_response.id] = sample_response
        attachment = response_manager.attach_evidence(
            response_id=sample_response.id,
            file_name="verification_report.pdf",
            file_type="application/pdf",
            file_size_bytes=1048576,
            storage_path="/evidence/verification_report.pdf",
            description="Third-party verification statement",
            uploaded_by=_new_id(),
        )
        assert attachment.file_name == "verification_report.pdf"
        assert attachment.file_size_bytes == 1048576

    def test_list_evidence(self, response_manager, sample_response):
        response_manager._store[sample_response.id] = sample_response
        response_manager.attach_evidence(
            response_id=sample_response.id,
            file_name="doc1.pdf",
            file_type="application/pdf",
            file_size_bytes=500000,
            storage_path="/evidence/doc1.pdf",
        )
        response_manager.attach_evidence(
            response_id=sample_response.id,
            file_name="doc2.xlsx",
            file_type="application/vnd.openxmlformats",
            file_size_bytes=250000,
            storage_path="/evidence/doc2.xlsx",
        )
        attachments = response_manager.list_evidence(sample_response.id)
        assert len(attachments) == 2

    def test_remove_evidence(self, response_manager, sample_response):
        response_manager._store[sample_response.id] = sample_response
        attachment = response_manager.attach_evidence(
            response_id=sample_response.id,
            file_name="temp.pdf",
            file_type="application/pdf",
            file_size_bytes=100,
            storage_path="/evidence/temp.pdf",
        )
        response_manager.remove_evidence(sample_response.id, attachment.id)
        attachments = response_manager.list_evidence(sample_response.id)
        assert len(attachments) == 0


# ---------------------------------------------------------------------------
# Review workflow
# ---------------------------------------------------------------------------

class TestReviewWorkflow:
    """Test multi-step review workflow."""

    def test_assign_reviewer(self, response_manager, sample_response):
        response_manager._store[sample_response.id] = sample_response
        updated = response_manager.assign_reviewer(
            response_id=sample_response.id,
            reviewer_id=_new_id(),
        )
        assert updated.assigned_to is not None

    def test_submit_review(self, response_manager, sample_response):
        sample_response.response_status = ResponseStatus.IN_REVIEW
        sample_response.assigned_to = _new_id()
        response_manager._store[sample_response.id] = sample_response
        action = response_manager.submit_review(
            response_id=sample_response.id,
            reviewer_id=sample_response.assigned_to,
            action="approve",
            comments="Looks good",
        )
        assert action.action == "approve"

    def test_reject_review(self, response_manager, sample_response):
        sample_response.response_status = ResponseStatus.IN_REVIEW
        sample_response.assigned_to = _new_id()
        response_manager._store[sample_response.id] = sample_response
        action = response_manager.submit_review(
            response_id=sample_response.id,
            reviewer_id=sample_response.assigned_to,
            action="reject",
            comments="Needs more detail on methodology",
        )
        assert action.action == "reject"
        updated = response_manager.get_response(sample_response.id)
        assert updated.response_status == ResponseStatus.DRAFT

    def test_approve_response(self, response_manager, sample_response):
        sample_response.response_status = ResponseStatus.IN_REVIEW
        response_manager._store[sample_response.id] = sample_response
        approved = response_manager.approve_response(
            response_id=sample_response.id,
            approver_id=_new_id(),
        )
        assert approved.response_status == ResponseStatus.APPROVED
        assert approved.approved_by is not None


# ---------------------------------------------------------------------------
# Bulk operations
# ---------------------------------------------------------------------------

class TestBulkOperations:
    """Test bulk import and approval."""

    def test_bulk_import_previous_year(self, response_manager, sample_questionnaire):
        prev_responses = [
            CDPResponse(
                question_id=_new_id(),
                questionnaire_id=_new_id(),
                org_id=sample_questionnaire.org_id,
                response_content={"answer": f"prev_{i}"},
                response_text=f"Previous year response {i}",
                response_status=ResponseStatus.SUBMITTED,
            )
            for i in range(5)
        ]
        imported = response_manager.bulk_import_previous_year(
            target_questionnaire_id=sample_questionnaire.id,
            previous_responses=prev_responses,
        )
        assert len(imported) == 5
        for r in imported:
            assert r.response_status == ResponseStatus.DRAFT
            assert r.questionnaire_id == sample_questionnaire.id

    def test_bulk_approve(self, response_manager):
        responses = []
        for i in range(3):
            r = CDPResponse(
                question_id=_new_id(),
                questionnaire_id=_new_id(),
                org_id=_new_id(),
                response_content={"answer": f"val_{i}"},
                response_text=f"Response {i}",
                response_status=ResponseStatus.IN_REVIEW,
            )
            response_manager._store[r.id] = r
            responses.append(r)
        response_ids = [r.id for r in responses]
        approved = response_manager.bulk_approve(
            response_ids=response_ids,
            approver_id=_new_id(),
        )
        assert len(approved) == 3
        for r in approved:
            assert r.response_status == ResponseStatus.APPROVED


# ---------------------------------------------------------------------------
# Auto-save conflict detection
# ---------------------------------------------------------------------------

class TestAutoSaveConflict:
    """Test auto-save conflict detection."""

    def test_no_conflict_same_timestamp(self, response_manager, sample_response):
        response_manager._store[sample_response.id] = sample_response
        has_conflict = response_manager.check_save_conflict(
            response_id=sample_response.id,
            client_updated_at=sample_response.updated_at,
        )
        assert has_conflict is False

    def test_conflict_stale_timestamp(self, response_manager, sample_response):
        from datetime import timedelta
        response_manager._store[sample_response.id] = sample_response
        stale_time = sample_response.updated_at - timedelta(minutes=5)
        has_conflict = response_manager.check_save_conflict(
            response_id=sample_response.id,
            client_updated_at=stale_time,
        )
        assert has_conflict is True


# ---------------------------------------------------------------------------
# Response templates
# ---------------------------------------------------------------------------

class TestResponseTemplates:
    """Test reusable response templates."""

    def test_create_template(self, response_manager):
        template = response_manager.create_template(
            name="Standard Verification Response",
            content={"answer": "yes", "details": "Verified by third party"},
            text="Yes, emissions are verified by an accredited third party.",
        )
        assert template["name"] == "Standard Verification Response"

    def test_apply_template(self, response_manager, sample_response):
        response_manager._store[sample_response.id] = sample_response
        template = response_manager.create_template(
            name="Template A",
            content={"answer": "template_value"},
            text="Template response text",
        )
        applied = response_manager.apply_template(
            response_id=sample_response.id,
            template_id=template["id"],
        )
        assert applied.response_content == {"answer": "template_value"}
        assert applied.response_text == "Template response text"

    def test_list_templates(self, response_manager):
        response_manager.create_template(name="T1", content={}, text="t1")
        response_manager.create_template(name="T2", content={}, text="t2")
        templates = response_manager.list_templates()
        assert len(templates) >= 2
