"""
CDP Response Manager -- Response Lifecycle Management

This module implements the full response lifecycle for CDP questionnaire
responses including CRUD operations, status transitions (draft -> in_review ->
approved -> submitted), version control with change tracking, evidence
attachment management, review workflows, bulk operations, auto-save with
conflict detection, and response templates.

Key capabilities:
  - CRUD for responses per question
  - Status transitions with validation
  - Version control with full change history
  - Evidence attachment management
  - Review workflow (assign, review, approve, reject)
  - Bulk operations (import previous year, bulk approve)
  - Auto-save with conflict detection via timestamps
  - Response templates for recurring answers

Example:
    >>> manager = ResponseManager(config, questionnaire_engine)
    >>> response = manager.save_response(questionnaire_id, question_id, "My answer")
    >>> manager.submit_for_review(response.id, "reviewer-1")
"""

from __future__ import annotations

import logging
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

from .config import CDPAppConfig, CDPModule, ResponseStatus
from .models import (
    EvidenceAttachment,
    Response,
    ResponseVersion,
    ReviewComment,
    ReviewWorkflow,
    _new_id,
    _now,
    _sha256,
)
from .questionnaire_engine import QuestionnaireEngine

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Valid State Transitions
# ---------------------------------------------------------------------------

VALID_TRANSITIONS: Dict[ResponseStatus, List[ResponseStatus]] = {
    ResponseStatus.NOT_STARTED: [ResponseStatus.DRAFT],
    ResponseStatus.DRAFT: [ResponseStatus.IN_REVIEW],
    ResponseStatus.IN_REVIEW: [ResponseStatus.DRAFT, ResponseStatus.APPROVED],
    ResponseStatus.APPROVED: [ResponseStatus.IN_REVIEW, ResponseStatus.SUBMITTED],
    ResponseStatus.SUBMITTED: [],
    ResponseStatus.RETURNED: [ResponseStatus.DRAFT],
}


class ResponseTemplate(object):
    """A reusable response template for recurring question types."""

    def __init__(
        self,
        template_id: str,
        name: str,
        question_type: str,
        content: str,
        table_data: Optional[List[Dict[str, Any]]] = None,
        created_by: Optional[str] = None,
    ) -> None:
        self.id = template_id
        self.name = name
        self.question_type = question_type
        self.content = content
        self.table_data = table_data
        self.created_by = created_by
        self.created_at = _now()


class ResponseManager:
    """
    CDP Response Manager -- manages response lifecycle and workflows.

    Provides CRUD for responses, status transitions with validation,
    version control, evidence attachments, review workflows, and
    bulk operations.

    Attributes:
        config: Application configuration.
        questionnaire_engine: Reference to questionnaire engine for validation.
        _responses: In-memory response store keyed by response ID.
        _templates: In-memory template store.

    Example:
        >>> manager = ResponseManager(config, questionnaire_engine)
        >>> resp = manager.save_response("q-1", "C1.1", content="Yes")
    """

    def __init__(
        self,
        config: CDPAppConfig,
        questionnaire_engine: QuestionnaireEngine,
    ) -> None:
        """Initialize the Response Manager."""
        self.config = config
        self.questionnaire_engine = questionnaire_engine
        self._responses: Dict[str, Response] = {}
        self._by_questionnaire: Dict[str, Dict[str, Response]] = {}  # q_id -> {question_id: resp}
        self._templates: Dict[str, ResponseTemplate] = {}
        logger.info("ResponseManager initialized")

    # ------------------------------------------------------------------
    # Response CRUD
    # ------------------------------------------------------------------

    def save_response(
        self,
        questionnaire_id: str,
        question_id: str,
        content: Optional[str] = None,
        table_data: Optional[List[Dict[str, Any]]] = None,
        numeric_value: Optional[Decimal] = None,
        selected_options: Optional[List[str]] = None,
        user_id: Optional[str] = None,
    ) -> Response:
        """
        Save or update a response to a question.

        Creates a new response if none exists, or updates the existing one
        with a new version entry for change tracking.

        Args:
            questionnaire_id: Questionnaire ID.
            question_id: Question ID or question number.
            content: Text content of the response.
            table_data: Tabular data for table-type questions.
            numeric_value: Numeric value for numeric-type questions.
            selected_options: Selected option values.
            user_id: User making the change.

        Returns:
            Saved Response instance.
        """
        start_time = datetime.utcnow()

        existing = self._find_response(questionnaire_id, question_id)
        if existing:
            response = self._update_existing_response(
                existing, content, table_data, numeric_value,
                selected_options, user_id,
            )
        else:
            response = self._create_new_response(
                questionnaire_id, question_id, content,
                table_data, numeric_value, selected_options, user_id,
            )

        # Update questionnaire module completion
        self._update_module_completion(questionnaire_id, response.module_code.value)

        elapsed = (datetime.utcnow() - start_time).total_seconds() * 1000
        logger.info(
            "Saved response for question %s in questionnaire %s (%.1f ms)",
            question_id, questionnaire_id, elapsed,
        )
        return response

    def get_response(
        self,
        questionnaire_id: str,
        question_id: str,
    ) -> Optional[Response]:
        """Get the response for a specific question."""
        return self._find_response(questionnaire_id, question_id)

    def get_all_responses(self, questionnaire_id: str) -> List[Response]:
        """Get all responses for a questionnaire."""
        q_responses = self._by_questionnaire.get(questionnaire_id, {})
        return list(q_responses.values())

    def get_module_responses(
        self,
        questionnaire_id: str,
        module_code: str,
    ) -> List[Response]:
        """Get all responses for a specific module."""
        all_resp = self.get_all_responses(questionnaire_id)
        return [r for r in all_resp if r.module_code.value == module_code]

    def delete_response(
        self,
        questionnaire_id: str,
        question_id: str,
    ) -> bool:
        """Delete a response (only if in DRAFT or NOT_STARTED status)."""
        response = self._find_response(questionnaire_id, question_id)
        if not response:
            return False
        if response.status not in (ResponseStatus.NOT_STARTED, ResponseStatus.DRAFT):
            logger.warning(
                "Cannot delete response %s in status %s",
                response.id, response.status.value,
            )
            return False

        self._responses.pop(response.id, None)
        q_map = self._by_questionnaire.get(questionnaire_id, {})
        q_map.pop(question_id, None)
        self._update_module_completion(questionnaire_id, response.module_code.value)
        logger.info("Deleted response %s", response.id)
        return True

    # ------------------------------------------------------------------
    # Status Transitions
    # ------------------------------------------------------------------

    def submit_for_review(
        self,
        response_id: str,
        reviewer: str,
        comment: Optional[str] = None,
    ) -> Response:
        """
        Submit a response for review.

        Transitions status from DRAFT to IN_REVIEW and creates a review workflow.

        Args:
            response_id: Response ID.
            reviewer: Reviewer user ID.
            comment: Optional submission comment.

        Returns:
            Updated Response.

        Raises:
            ValueError: If transition is not valid.
        """
        response = self._get_or_raise(response_id)
        self._validate_transition(response.status, ResponseStatus.IN_REVIEW)

        response.status = ResponseStatus.IN_REVIEW
        response.workflow = ReviewWorkflow(
            response_id=response_id,
            assigned_to=reviewer,
            status=ResponseStatus.IN_REVIEW,
            review_started_at=_now(),
        )

        if comment:
            review_comment = ReviewComment(
                response_id=response_id,
                reviewer="submitter",
                comment=comment,
                action="submit",
            )
            response.workflow.comments.append(review_comment)

        response.updated_at = _now()
        logger.info("Response %s submitted for review by %s", response_id, reviewer)
        return response

    def approve_response(
        self,
        response_id: str,
        approved_by: str,
        comment: Optional[str] = None,
    ) -> Response:
        """
        Approve a response.

        Transitions status from IN_REVIEW to APPROVED.

        Args:
            response_id: Response ID.
            approved_by: Approver user ID.
            comment: Optional approval comment.

        Returns:
            Updated Response.
        """
        response = self._get_or_raise(response_id)
        self._validate_transition(response.status, ResponseStatus.APPROVED)

        response.status = ResponseStatus.APPROVED

        if response.workflow:
            response.workflow.status = ResponseStatus.APPROVED
            response.workflow.approved_by = approved_by
            response.workflow.approved_at = _now()
            response.workflow.review_completed_at = _now()
            if comment:
                response.workflow.comments.append(ReviewComment(
                    response_id=response_id,
                    reviewer=approved_by,
                    comment=comment,
                    action="approve",
                ))

        response.updated_at = _now()
        self._update_module_completion(
            response.questionnaire_id, response.module_code.value,
        )
        logger.info("Response %s approved by %s", response_id, approved_by)
        return response

    def reject_response(
        self,
        response_id: str,
        rejected_by: str,
        reason: str,
    ) -> Response:
        """
        Reject a response back to DRAFT status.

        Args:
            response_id: Response ID.
            rejected_by: Rejector user ID.
            reason: Rejection reason.

        Returns:
            Updated Response.
        """
        response = self._get_or_raise(response_id)
        self._validate_transition(response.status, ResponseStatus.DRAFT)

        response.status = ResponseStatus.DRAFT

        if response.workflow:
            response.workflow.status = ResponseStatus.DRAFT
            response.workflow.review_completed_at = _now()
            response.workflow.comments.append(ReviewComment(
                response_id=response_id,
                reviewer=rejected_by,
                comment=reason,
                action="reject",
            ))

        response.updated_at = _now()
        logger.info("Response %s rejected by %s: %s", response_id, rejected_by, reason)
        return response

    def mark_submitted(self, questionnaire_id: str) -> int:
        """
        Mark all approved responses as submitted.

        Returns the number of responses marked as submitted.
        """
        count = 0
        responses = self.get_all_responses(questionnaire_id)
        for response in responses:
            if response.status == ResponseStatus.APPROVED:
                response.status = ResponseStatus.SUBMITTED
                response.updated_at = _now()
                count += 1

        logger.info(
            "Marked %d responses as submitted for questionnaire %s",
            count, questionnaire_id,
        )
        return count

    # ------------------------------------------------------------------
    # Evidence Attachments
    # ------------------------------------------------------------------

    def add_evidence(
        self,
        response_id: str,
        file_name: str,
        file_type: str,
        file_size_bytes: int,
        file_path: Optional[str] = None,
        description: Optional[str] = None,
        uploaded_by: Optional[str] = None,
    ) -> EvidenceAttachment:
        """
        Attach evidence to a response.

        Validates file type and size against configuration limits.

        Args:
            response_id: Response ID.
            file_name: Name of the file.
            file_type: File extension type.
            file_size_bytes: File size in bytes.
            file_path: Storage path.
            description: Evidence description.
            uploaded_by: User uploading the file.

        Returns:
            Created EvidenceAttachment.

        Raises:
            ValueError: If file type or size exceeds limits.
        """
        response = self._get_or_raise(response_id)

        # Validate file type
        if file_type.lower() not in self.config.evidence_allowed_types:
            raise ValueError(
                f"File type '{file_type}' not allowed. "
                f"Allowed: {self.config.evidence_allowed_types}"
            )

        # Validate file size
        max_bytes = self.config.evidence_max_size_mb * 1024 * 1024
        if file_size_bytes > max_bytes:
            raise ValueError(
                f"File size {file_size_bytes} exceeds maximum "
                f"{self.config.evidence_max_size_mb} MB"
            )

        attachment = EvidenceAttachment(
            response_id=response_id,
            file_name=file_name,
            file_type=file_type.lower(),
            file_size_bytes=file_size_bytes,
            file_path=file_path,
            description=description,
            uploaded_by=uploaded_by,
        )

        response.evidence.append(attachment)
        response.updated_at = _now()
        logger.info(
            "Added evidence '%s' to response %s", file_name, response_id,
        )
        return attachment

    def remove_evidence(self, response_id: str, attachment_id: str) -> bool:
        """Remove an evidence attachment from a response."""
        response = self._get_or_raise(response_id)
        original_len = len(response.evidence)
        response.evidence = [e for e in response.evidence if e.id != attachment_id]
        removed = len(response.evidence) < original_len
        if removed:
            response.updated_at = _now()
            logger.info("Removed evidence %s from response %s", attachment_id, response_id)
        return removed

    def get_evidence(self, response_id: str) -> List[EvidenceAttachment]:
        """Get all evidence attachments for a response."""
        response = self._responses.get(response_id)
        if response:
            return response.evidence
        return []

    # ------------------------------------------------------------------
    # Review Workflow
    # ------------------------------------------------------------------

    def assign_question(
        self,
        questionnaire_id: str,
        question_id: str,
        assignee: str,
    ) -> Response:
        """Assign a question to a team member."""
        response = self._find_response(questionnaire_id, question_id)
        if not response:
            # Create a placeholder response
            response = self.save_response(
                questionnaire_id, question_id, content="",
            )

        response.assigned_to = assignee
        response.updated_at = _now()
        logger.info("Assigned question %s to %s", question_id, assignee)
        return response

    def add_review_comment(
        self,
        response_id: str,
        reviewer: str,
        comment: str,
        action: str = "comment",
    ) -> ReviewComment:
        """Add a review comment to a response."""
        response = self._get_or_raise(response_id)

        review_comment = ReviewComment(
            response_id=response_id,
            reviewer=reviewer,
            comment=comment,
            action=action,
        )

        if not response.workflow:
            response.workflow = ReviewWorkflow(
                response_id=response_id,
                status=response.status,
            )

        response.workflow.comments.append(review_comment)
        response.workflow.updated_at = _now()
        return review_comment

    # ------------------------------------------------------------------
    # Bulk Operations
    # ------------------------------------------------------------------

    def bulk_save(
        self,
        questionnaire_id: str,
        response_data: List[Dict[str, Any]],
        user_id: Optional[str] = None,
    ) -> List[Response]:
        """
        Save multiple responses at once.

        Args:
            questionnaire_id: Questionnaire ID.
            response_data: List of dicts with question_id, content, etc.
            user_id: User performing the bulk save.

        Returns:
            List of saved Responses.
        """
        results = []
        for data in response_data:
            response = self.save_response(
                questionnaire_id=questionnaire_id,
                question_id=data.get("question_id", ""),
                content=data.get("content"),
                table_data=data.get("table_data"),
                numeric_value=data.get("numeric_value"),
                selected_options=data.get("selected_options"),
                user_id=user_id,
            )
            results.append(response)

        logger.info(
            "Bulk saved %d responses for questionnaire %s",
            len(results), questionnaire_id,
        )
        return results

    def bulk_approve(
        self,
        questionnaire_id: str,
        approved_by: str,
        module_code: Optional[str] = None,
    ) -> int:
        """
        Bulk approve all IN_REVIEW responses.

        Args:
            questionnaire_id: Questionnaire ID.
            approved_by: Approver user ID.
            module_code: Optional module filter.

        Returns:
            Number of responses approved.
        """
        responses = self.get_all_responses(questionnaire_id)
        if module_code:
            responses = [r for r in responses if r.module_code.value == module_code]

        count = 0
        for response in responses:
            if response.status == ResponseStatus.IN_REVIEW:
                try:
                    self.approve_response(response.id, approved_by)
                    count += 1
                except ValueError:
                    continue

        logger.info(
            "Bulk approved %d responses for questionnaire %s",
            count, questionnaire_id,
        )
        return count

    def import_previous_year(
        self,
        target_questionnaire_id: str,
        source_questionnaire_id: str,
        overwrite: bool = False,
    ) -> Tuple[int, int]:
        """
        Import responses from a previous year questionnaire.

        Args:
            target_questionnaire_id: Target questionnaire ID.
            source_questionnaire_id: Source questionnaire ID.
            overwrite: Whether to overwrite existing responses.

        Returns:
            Tuple of (imported_count, skipped_count).
        """
        source_responses = self.get_all_responses(source_questionnaire_id)
        imported = 0
        skipped = 0

        for src_resp in source_responses:
            existing = self._find_response(
                target_questionnaire_id, src_resp.question_id,
            )

            if existing and not overwrite:
                skipped += 1
                continue

            self.save_response(
                questionnaire_id=target_questionnaire_id,
                question_id=src_resp.question_id,
                content=src_resp.content,
                table_data=src_resp.table_data,
                numeric_value=src_resp.numeric_value,
                selected_options=src_resp.selected_options,
                user_id="import_previous_year",
            )
            imported += 1

        logger.info(
            "Imported %d responses from %s to %s (skipped %d)",
            imported, source_questionnaire_id, target_questionnaire_id, skipped,
        )
        return imported, skipped

    # ------------------------------------------------------------------
    # Auto-Save with Conflict Detection
    # ------------------------------------------------------------------

    def auto_save(
        self,
        questionnaire_id: str,
        question_id: str,
        content: Optional[str] = None,
        table_data: Optional[List[Dict[str, Any]]] = None,
        numeric_value: Optional[Decimal] = None,
        selected_options: Optional[List[str]] = None,
        client_timestamp: Optional[datetime] = None,
        user_id: Optional[str] = None,
    ) -> Tuple[Response, bool]:
        """
        Auto-save with conflict detection.

        Checks if the server version has been modified since the client's
        last known timestamp. If a conflict is detected, returns the
        server version without saving.

        Args:
            questionnaire_id: Questionnaire ID.
            question_id: Question ID.
            content: Text content.
            table_data: Table data.
            numeric_value: Numeric value.
            selected_options: Selected options.
            client_timestamp: Client's last known save timestamp.
            user_id: User performing the save.

        Returns:
            Tuple of (Response, conflict_detected: bool).
        """
        existing = self._find_response(questionnaire_id, question_id)

        if existing and client_timestamp:
            if existing.last_saved_at and existing.last_saved_at > client_timestamp:
                logger.warning(
                    "Auto-save conflict detected for question %s. "
                    "Server: %s > Client: %s",
                    question_id, existing.last_saved_at, client_timestamp,
                )
                return existing, True

        response = self.save_response(
            questionnaire_id, question_id, content,
            table_data, numeric_value, selected_options, user_id,
        )
        return response, False

    # ------------------------------------------------------------------
    # Response Templates
    # ------------------------------------------------------------------

    def create_template(
        self,
        name: str,
        question_type: str,
        content: str,
        table_data: Optional[List[Dict[str, Any]]] = None,
        created_by: Optional[str] = None,
    ) -> ResponseTemplate:
        """Create a reusable response template."""
        template = ResponseTemplate(
            template_id=_new_id(),
            name=name,
            question_type=question_type,
            content=content,
            table_data=table_data,
            created_by=created_by,
        )
        self._templates[template.id] = template
        logger.info("Created response template '%s' (%s)", name, template.id)
        return template

    def apply_template(
        self,
        template_id: str,
        questionnaire_id: str,
        question_id: str,
        user_id: Optional[str] = None,
    ) -> Optional[Response]:
        """Apply a template to a question response."""
        template = self._templates.get(template_id)
        if not template:
            return None

        return self.save_response(
            questionnaire_id=questionnaire_id,
            question_id=question_id,
            content=template.content,
            table_data=template.table_data,
            user_id=user_id,
        )

    def list_templates(self) -> List[ResponseTemplate]:
        """List all available response templates."""
        return list(self._templates.values())

    # ------------------------------------------------------------------
    # Completion Statistics
    # ------------------------------------------------------------------

    def get_completion_stats(
        self,
        questionnaire_id: str,
    ) -> Dict[str, Any]:
        """
        Calculate completion statistics for a questionnaire.

        Returns a summary with counts and percentages.
        """
        responses = self.get_all_responses(questionnaire_id)
        total = len(responses)

        if total == 0:
            return {
                "total_responses": 0,
                "not_started": 0,
                "draft": 0,
                "in_review": 0,
                "approved": 0,
                "submitted": 0,
                "completion_pct": 0.0,
                "approval_pct": 0.0,
            }

        status_counts = {}
        for status in ResponseStatus:
            status_counts[status.value] = sum(
                1 for r in responses if r.status == status
            )

        answered = total - status_counts.get("not_started", 0)
        approved = status_counts.get("approved", 0) + status_counts.get("submitted", 0)

        questionnaire = self.questionnaire_engine.get_questionnaire(questionnaire_id)
        total_questions = questionnaire.total_questions if questionnaire else total

        return {
            "total_responses": total,
            "total_questions": total_questions,
            "not_started": status_counts.get("not_started", 0),
            "draft": status_counts.get("draft", 0),
            "in_review": status_counts.get("in_review", 0),
            "approved": status_counts.get("approved", 0),
            "submitted": status_counts.get("submitted", 0),
            "completion_pct": round(answered / total_questions * 100, 1) if total_questions > 0 else 0.0,
            "approval_pct": round(approved / total_questions * 100, 1) if total_questions > 0 else 0.0,
        }

    # ------------------------------------------------------------------
    # Internal Helpers
    # ------------------------------------------------------------------

    def _find_response(
        self,
        questionnaire_id: str,
        question_id: str,
    ) -> Optional[Response]:
        """Find an existing response by questionnaire and question ID."""
        q_map = self._by_questionnaire.get(questionnaire_id, {})
        return q_map.get(question_id)

    def _get_or_raise(self, response_id: str) -> Response:
        """Get a response by ID or raise ValueError."""
        response = self._responses.get(response_id)
        if not response:
            raise ValueError(f"Response {response_id} not found")
        return response

    def _validate_transition(
        self,
        current: ResponseStatus,
        target: ResponseStatus,
    ) -> None:
        """Validate a status transition is allowed."""
        allowed = VALID_TRANSITIONS.get(current, [])
        if target not in allowed:
            raise ValueError(
                f"Invalid transition from {current.value} to {target.value}. "
                f"Allowed transitions: {[s.value for s in allowed]}"
            )

    def _create_new_response(
        self,
        questionnaire_id: str,
        question_id: str,
        content: Optional[str],
        table_data: Optional[List[Dict[str, Any]]],
        numeric_value: Optional[Decimal],
        selected_options: Optional[List[str]],
        user_id: Optional[str],
    ) -> Response:
        """Create a new response."""
        # Look up question metadata
        question = self.questionnaire_engine.get_question(question_id)
        module_code = question.module_code if question else CDPModule.M0_INTRODUCTION
        q_number = question.question_number if question else question_id

        response = Response(
            questionnaire_id=questionnaire_id,
            question_id=question_id,
            question_number=q_number,
            module_code=module_code,
            status=ResponseStatus.DRAFT,
            content=content or "",
            table_data=table_data,
            numeric_value=numeric_value,
            selected_options=selected_options or [],
            current_version=1,
            last_saved_at=_now(),
        )

        # Create initial version
        version = ResponseVersion(
            response_id=response.id,
            version_number=1,
            content=content or "",
            table_data=table_data,
            numeric_value=numeric_value,
            selected_options=selected_options or [],
            changed_by=user_id,
            change_reason="Initial creation",
        )
        response.versions.append(version)

        # Store
        self._responses[response.id] = response
        if questionnaire_id not in self._by_questionnaire:
            self._by_questionnaire[questionnaire_id] = {}
        self._by_questionnaire[questionnaire_id][question_id] = response

        return response

    def _update_existing_response(
        self,
        response: Response,
        content: Optional[str],
        table_data: Optional[List[Dict[str, Any]]],
        numeric_value: Optional[Decimal],
        selected_options: Optional[List[str]],
        user_id: Optional[str],
    ) -> Response:
        """Update an existing response with a new version."""
        # Check if content actually changed
        new_content = content if content is not None else response.content
        new_table = table_data if table_data is not None else response.table_data
        new_numeric = numeric_value if numeric_value is not None else response.numeric_value
        new_options = selected_options if selected_options is not None else response.selected_options

        content_changed = (
            new_content != response.content
            or new_table != response.table_data
            or new_numeric != response.numeric_value
            or new_options != response.selected_options
        )

        if content_changed:
            response.current_version += 1
            version = ResponseVersion(
                response_id=response.id,
                version_number=response.current_version,
                content=new_content,
                table_data=new_table,
                numeric_value=new_numeric,
                selected_options=new_options,
                changed_by=user_id,
                change_reason="Content updated",
            )
            response.versions.append(version)

        response.content = new_content
        response.table_data = new_table
        response.numeric_value = new_numeric
        response.selected_options = new_options
        response.last_saved_at = _now()
        response.updated_at = _now()

        # Reset to draft if was not_started
        if response.status == ResponseStatus.NOT_STARTED and new_content:
            response.status = ResponseStatus.DRAFT

        return response

    def _update_module_completion(
        self,
        questionnaire_id: str,
        module_code: str,
    ) -> None:
        """Update module completion counts after a response change."""
        responses = self.get_module_responses(questionnaire_id, module_code)
        answered = sum(
            1 for r in responses
            if r.status not in (ResponseStatus.NOT_STARTED,) and r.content
        )
        approved = sum(
            1 for r in responses
            if r.status in (ResponseStatus.APPROVED, ResponseStatus.SUBMITTED)
        )

        self.questionnaire_engine.update_module_completion(
            questionnaire_id, module_code, answered, approved,
        )
