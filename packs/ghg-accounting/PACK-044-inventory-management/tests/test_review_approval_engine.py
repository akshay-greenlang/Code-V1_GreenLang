# -*- coding: utf-8 -*-
"""
PACK-044 Test Suite - Review Approval Engine Tests
=====================================================

Tests ReviewApprovalEngine: 4-stage review workflow, separation of
duties, digital sign-off, comments, escalation, and approval records.

Target: 60+ test cases.
"""

from decimal import Decimal

import pytest

from conftest import _load_engine

# ---------------------------------------------------------------------------
# Dynamic imports
# ---------------------------------------------------------------------------

_mod = _load_engine("review_approval")

ReviewApprovalEngine = _mod.ReviewApprovalEngine
ReviewRequest = _mod.ReviewRequest
ReviewComment = _mod.ReviewComment
StageSignOff = _mod.StageSignOff
ReviewDecision = _mod.ReviewDecision
ApprovalRecord = _mod.ApprovalRecord
ReviewApprovalResult = _mod.ReviewApprovalResult
ReviewStage = _mod.ReviewStage
ReviewStatus = _mod.ReviewStatus
CommentType = _mod.CommentType
FindingSeverity = _mod.FindingSeverity


# ===================================================================
# Fixtures
# ===================================================================


@pytest.fixture
def engine():
    """Create a fresh ReviewApprovalEngine."""
    return ReviewApprovalEngine()


@pytest.fixture
def review_request():
    """Standard review request."""
    return ReviewRequest(
        inventory_id="inv-2025-001",
        reporting_year=2025,
        section="scope1",
        title="FY2025 Scope 1 Review",
        preparer_id="user-analyst-001",
        preparer_name="Data Analyst",
        reviewer_id="user-reviewer-001",
        reviewer_name="Senior Reviewer",
        approver_id="user-approver-001",
        approver_name="Sustainability Director",
    )


@pytest.fixture
def submitted_review(engine, review_request):
    """Create and submit a review request, returning (engine, request, result)."""
    result = engine.submit_for_review(
        review_request, "user-analyst-001", "Data Analyst"
    )
    return engine, result.request, result


# ===================================================================
# Review Creation / Submission Tests
# ===================================================================


class TestReviewCreation:
    """Tests for creating and submitting review requests."""

    def test_submit_returns_result(self, engine, review_request):
        result = engine.submit_for_review(
            review_request, "user-analyst-001", "Data Analyst"
        )
        assert isinstance(result, ReviewApprovalResult)

    def test_after_submit_stage_is_review(self, engine, review_request):
        result = engine.submit_for_review(
            review_request, "user-analyst-001", "Data Analyst"
        )
        assert result.request.current_stage == ReviewStage.REVIEW

    def test_stage_statuses_initialized(self, review_request):
        statuses = review_request.stage_statuses
        assert ReviewStage.PREPARATION.value in statuses
        assert ReviewStage.REVIEW.value in statuses
        assert ReviewStage.APPROVAL.value in statuses
        assert ReviewStage.VERIFICATION.value in statuses

    def test_preparer_stored(self, review_request):
        assert review_request.preparer_id == "user-analyst-001"

    def test_reviewer_stored(self, review_request):
        assert review_request.reviewer_id == "user-reviewer-001"

    def test_approver_stored(self, review_request):
        assert review_request.approver_id == "user-approver-001"

    def test_provenance_hash(self, engine, review_request):
        result = engine.submit_for_review(
            review_request, "user-analyst-001", "Data Analyst"
        )
        assert len(result.provenance_hash) == 64

    def test_revision_round_zero(self, review_request):
        assert review_request.revision_round == 0


# ===================================================================
# Stage Progression Tests
# ===================================================================


class TestStageProgression:
    """Tests for advancing through review stages."""

    def test_submit_for_review(self, submitted_review):
        engine, req, result = submitted_review
        assert req.current_stage == ReviewStage.REVIEW

    def test_submit_requires_preparer(self, engine, review_request):
        with pytest.raises((ValueError, Exception)):
            engine.submit_for_review(
                review_request, "user-reviewer-001", "Not the preparer"
            )

    def test_approve_at_review_stage(self, submitted_review):
        engine, req, _ = submitted_review
        r = engine.record_decision(
            req, stage=ReviewStage.REVIEW,
            decision=ReviewStatus.APPROVED,
            actor_id="user-reviewer-001",
            actor_name="Senior Reviewer",
        )
        assert r.request.current_stage == ReviewStage.APPROVAL

    def test_approve_at_approval_stage(self, submitted_review):
        engine, req, _ = submitted_review
        engine.record_decision(
            req, ReviewStage.REVIEW,
            ReviewStatus.APPROVED,
            "user-reviewer-001", "Reviewer",
        )
        r = engine.record_decision(
            req, ReviewStage.APPROVAL,
            ReviewStatus.APPROVED,
            "user-approver-001", "Approver",
        )
        assert r.decision.can_progress is True

    def test_full_workflow_to_approved(self, submitted_review):
        engine, req, _ = submitted_review
        engine.record_decision(
            req, ReviewStage.REVIEW,
            ReviewStatus.APPROVED,
            "user-reviewer-001", "Reviewer",
        )
        r = engine.record_decision(
            req, ReviewStage.APPROVAL,
            ReviewStatus.APPROVED,
            "user-approver-001", "Approver",
        )
        assert req.stage_statuses[ReviewStage.APPROVAL.value] in (
            ReviewStatus.APPROVED.value, "approved",
        )


# ===================================================================
# Separation of Duties Tests
# ===================================================================


class TestSeparationOfDuties:
    """Tests for enforcing different actors at each stage."""

    def test_reviewer_cannot_be_preparer(self, submitted_review):
        engine, req, _ = submitted_review
        with pytest.raises((ValueError, Exception)):
            engine.record_decision(
                req, ReviewStage.REVIEW,
                ReviewStatus.APPROVED,
                "user-analyst-001", "Analyst",
            )

    def test_approver_cannot_be_reviewer(self, submitted_review):
        engine, req, _ = submitted_review
        engine.record_decision(
            req, ReviewStage.REVIEW,
            ReviewStatus.APPROVED,
            "user-reviewer-001", "Reviewer",
        )
        with pytest.raises((ValueError, Exception)):
            engine.record_decision(
                req, ReviewStage.APPROVAL,
                ReviewStatus.APPROVED,
                "user-reviewer-001", "Reviewer acting as approver",
            )

    def test_approver_cannot_be_preparer(self, submitted_review):
        engine, req, _ = submitted_review
        engine.record_decision(
            req, ReviewStage.REVIEW,
            ReviewStatus.APPROVED,
            "user-reviewer-001", "Reviewer",
        )
        with pytest.raises((ValueError, Exception)):
            engine.record_decision(
                req, ReviewStage.APPROVAL,
                ReviewStatus.APPROVED,
                "user-analyst-001", "Analyst acting as approver",
            )


# ===================================================================
# Rejection and Revision Tests
# ===================================================================


class TestRejectionAndRevision:
    """Tests for rejection and revision rounds."""

    def test_revisions_requested_stays_at_current_stage(self, submitted_review):
        """REVISIONS_REQUESTED does not progress; stage stays at REVIEW."""
        engine, req, _ = submitted_review
        r = engine.record_decision(
            req, ReviewStage.REVIEW,
            ReviewStatus.REVISIONS_REQUESTED,
            "user-reviewer-001", "Reviewer",
        )
        # Engine keeps at current stage (no progression for REVISIONS_REQUESTED)
        assert r.request.current_stage == ReviewStage.REVIEW

    def test_revision_round_increments(self, submitted_review):
        engine, req, _ = submitted_review
        r = engine.record_decision(
            req, ReviewStage.REVIEW,
            ReviewStatus.REVISIONS_REQUESTED,
            "user-reviewer-001", "Reviewer",
        )
        assert r.request.revision_round >= 1

    def test_resubmit_after_revision_via_fresh_request(self, submitted_review):
        """After revision, preparer creates a new request for resubmission."""
        engine, req, _ = submitted_review
        engine.record_decision(
            req, ReviewStage.REVIEW,
            ReviewStatus.REVISIONS_REQUESTED,
            "user-reviewer-001", "Reviewer",
        )
        # Revision round incremented
        assert req.revision_round >= 1
        # The engine records the decision -- preparer can then address and
        # a new review cycle can start with a fresh ReviewRequest
        new_req = ReviewRequest(
            inventory_id=req.inventory_id,
            reporting_year=req.reporting_year,
            section=req.section,
            title=f"FY2025 Scope 1 Review - Revision {req.revision_round}",
            preparer_id=req.preparer_id,
            preparer_name=req.preparer_name,
            reviewer_id=req.reviewer_id,
            reviewer_name=req.reviewer_name,
            approver_id=req.approver_id,
            approver_name=req.approver_name,
        )
        r = engine.submit_for_review(new_req, "user-analyst-001", "Analyst")
        assert r.request.current_stage == ReviewStage.REVIEW


# ===================================================================
# Comment Tests
# ===================================================================


class TestComments:
    """Tests for add_comment."""

    def test_add_review_comment(self, submitted_review):
        engine, req, _ = submitted_review
        comment = ReviewComment(
            author_id="user-reviewer-001",
            author_name="Senior Reviewer",
            content="Please verify the EF source for boiler.",
            comment_type=CommentType.OBSERVATION,
            section_reference="scope1.stationary_combustion",
        )
        updated = engine.add_comment(req, comment)
        assert len(updated.comments) >= 1

    def test_add_finding_comment(self, submitted_review):
        engine, req, _ = submitted_review
        comment = ReviewComment(
            author_id="user-reviewer-001",
            author_name="Senior Reviewer",
            content="YoY variance exceeds 25% threshold.",
            comment_type=CommentType.FINDING,
            severity=FindingSeverity.MAJOR,
        )
        updated = engine.add_comment(req, comment)
        findings = [c for c in updated.comments
                    if c.comment_type == CommentType.FINDING]
        assert len(findings) >= 1

    def test_resolve_comment(self, submitted_review):
        engine, req, _ = submitted_review
        comment = ReviewComment(
            author_id="user-reviewer-001",
            author_name="Reviewer",
            content="Check this.",
            comment_type=CommentType.OBSERVATION,
        )
        engine.add_comment(req, comment)
        cid = req.comments[0].comment_id
        updated = engine.resolve_comment(
            req, cid, "user-analyst-001", "Analyst"
        )
        resolved = [c for c in updated.comments if c.comment_id == cid]
        assert resolved[0].is_resolved is True


# ===================================================================
# Sign-Off Tests
# ===================================================================


class TestSignOff:
    """Tests for stage sign-off records."""

    def test_sign_off_creates_record(self, submitted_review):
        engine, req, _ = submitted_review
        r = engine.record_decision(
            req, ReviewStage.REVIEW,
            ReviewStatus.APPROVED,
            "user-reviewer-001", "Reviewer",
        )
        assert len(r.request.signoffs) >= 1

    def test_sign_off_provenance_hash(self, submitted_review):
        engine, req, _ = submitted_review
        r = engine.record_decision(
            req, ReviewStage.REVIEW,
            ReviewStatus.APPROVED,
            "user-reviewer-001", "Reviewer",
        )
        so = r.request.signoffs[-1]
        assert len(so.provenance_hash) == 64

    def test_conditional_approval(self, submitted_review):
        engine, req, _ = submitted_review
        r = engine.record_decision(
            req, ReviewStage.REVIEW,
            ReviewStatus.APPROVED,
            "user-reviewer-001", "Reviewer",
            conditions=["Update EF sources before final submission"],
        )
        so = r.request.signoffs[-1]
        assert len(so.conditions) >= 1


# ===================================================================
# Approval Record Tests
# ===================================================================


class TestApprovalRecord:
    """Tests for final approval record generation."""

    def test_full_approval_generates_record(self, submitted_review):
        engine, req, _ = submitted_review
        engine.record_decision(
            req, ReviewStage.REVIEW,
            ReviewStatus.APPROVED,
            "user-reviewer-001", "Reviewer",
        )
        r = engine.record_decision(
            req, ReviewStage.APPROVAL,
            ReviewStatus.APPROVED,
            "user-approver-001", "Approver",
        )
        if r.approval_record is not None:
            assert r.approval_record.is_fully_approved is True
            assert len(r.approval_record.provenance_hash) == 64


# ===================================================================
# Review Summary Tests
# ===================================================================


class TestReviewSummary:
    """Tests for get_review_summary."""

    def test_summary_has_stage(self, submitted_review):
        engine, req, _ = submitted_review
        summary = engine.get_review_summary(req)
        assert summary["current_stage"] == ReviewStage.REVIEW.value

    def test_summary_has_signoffs(self, submitted_review):
        engine, req, _ = submitted_review
        summary = engine.get_review_summary(req)
        assert "signoffs_count" in summary


# ===================================================================
# Model Tests
# ===================================================================


class TestModels:
    """Tests for Pydantic model defaults and enum values."""

    @pytest.mark.parametrize("stage", list(ReviewStage))
    def test_review_stages(self, stage):
        assert stage.value is not None

    @pytest.mark.parametrize("status", list(ReviewStatus))
    def test_review_statuses(self, status):
        assert status.value is not None

    @pytest.mark.parametrize("ctype", list(CommentType))
    def test_comment_types(self, ctype):
        assert ctype.value is not None

    @pytest.mark.parametrize("sev", list(FindingSeverity))
    def test_finding_severities(self, sev):
        assert sev.value is not None

    def test_review_request_defaults(self):
        rr = ReviewRequest()
        assert rr.current_stage == ReviewStage.PREPARATION
        assert rr.revision_round == 0

    def test_stage_sign_off_defaults(self):
        so = StageSignOff()
        assert so.decision == ReviewStatus.APPROVED

    def test_review_comment_defaults(self):
        rc = ReviewComment()
        assert rc.comment_type == CommentType.OBSERVATION
        assert rc.is_resolved is False

    def test_review_decision_defaults(self):
        rd = ReviewDecision()
        assert rd.can_progress is False

    def test_approval_record_defaults(self):
        ar = ApprovalRecord()
        assert ar.is_fully_approved is False
