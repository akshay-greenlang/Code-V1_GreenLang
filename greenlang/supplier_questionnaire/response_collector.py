# -*- coding: utf-8 -*-
"""
Response Collector Engine - AGENT-DATA-008: Supplier Questionnaire Processor
=============================================================================

Collects and manages supplier questionnaire responses including submission,
partial save, finalisation, bulk import, deduplication, acknowledgement,
and progress tracking.

Supports:
    - Response submission with answer normalisation
    - Partial save / incremental update
    - Response finalisation (mark as submitted)
    - Bulk response import from structured data
    - Response reopening with reason tracking
    - Response deduplication per supplier/template
    - Acknowledgement token generation
    - Completion percentage tracking
    - Answer normalisation (numeric parsing, choice mapping)
    - Thread-safe in-memory storage
    - SHA-256 provenance hashes on all operations

Zero-Hallucination Guarantees:
    - All normalisation is rule-based (regex/arithmetic)
    - No LLM involvement in response collection
    - SHA-256 provenance hashes for audit trails
    - Completion percentages are deterministic calculations

Example:
    >>> from greenlang.supplier_questionnaire.response_collector import (
    ...     ResponseCollectorEngine,
    ... )
    >>> engine = ResponseCollectorEngine()
    >>> response = engine.submit_response(
    ...     distribution_id="dist-001",
    ...     answers=[{"question_id": "q1", "value": "42"}],
    ...     language="en",
    ... )
    >>> assert response.status.value == "in_progress"

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-008 Supplier Questionnaire Processor
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from greenlang.supplier_questionnaire.models import (
    Answer,
    QuestionnaireResponse,
    QuestionnaireTemplate,
    ResponseStatus,
)

logger = logging.getLogger(__name__)

__all__ = [
    "ResponseCollectorEngine",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


# Regex for numeric value extraction
_NUMERIC_PATTERN = re.compile(
    r"^[^\d]*(-?\d[\d,]*\.?\d*)\s*(%|tCO2e|MWh|kg|t|m3|ML|GJ)?$"
)


# ---------------------------------------------------------------------------
# ResponseCollectorEngine
# ---------------------------------------------------------------------------


class ResponseCollectorEngine:
    """Questionnaire response collection and management engine.

    Manages the lifecycle of supplier questionnaire responses from
    initial submission through finalisation, including partial saves,
    deduplication, acknowledgement, and progress tracking.

    Attributes:
        _responses: In-memory response storage keyed by response_id.
        _config: Configuration dictionary.
        _lock: Threading lock for mutations.
        _stats: Aggregate statistics counters.

    Example:
        >>> engine = ResponseCollectorEngine()
        >>> resp = engine.submit_response("dist-001", [{"question_id": "q1", "value": "yes"}])
        >>> progress = engine.get_response_progress(resp.response_id)
        >>> assert progress["total_answers"] == 1
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize ResponseCollectorEngine.

        Args:
            config: Optional configuration dict. Recognised keys:
                - ``max_responses``: int (default 50000)
                - ``normalize_values``: bool (default True)
                - ``auto_acknowledge``: bool (default True)
        """
        self._config = config or {}
        self._responses: Dict[str, QuestionnaireResponse] = {}
        self._lock = threading.Lock()
        self._max_responses: int = self._config.get("max_responses", 50000)
        self._normalize: bool = self._config.get("normalize_values", True)
        self._auto_ack: bool = self._config.get("auto_acknowledge", True)
        self._stats: Dict[str, int] = {
            "responses_submitted": 0,
            "responses_updated": 0,
            "responses_finalized": 0,
            "responses_reopened": 0,
            "responses_imported": 0,
            "responses_deduplicated": 0,
            "acknowledgements_sent": 0,
            "errors": 0,
        }
        logger.info(
            "ResponseCollectorEngine initialised: max_responses=%d, "
            "normalize=%s, auto_ack=%s",
            self._max_responses,
            self._normalize,
            self._auto_ack,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def submit_response(
        self,
        distribution_id: str,
        answers: List[Dict[str, Any]],
        language: str = "en",
        template_id: str = "",
        supplier_id: str = "",
    ) -> QuestionnaireResponse:
        """Submit a new questionnaire response.

        Creates a response record from the provided answers. If
        normalisation is enabled, numeric values are parsed and
        standardised.

        Args:
            distribution_id: Distribution being responded to.
            answers: List of answer dicts with question_id and value.
            language: Language used in the response.
            template_id: Template the response is for.
            supplier_id: Responding supplier identifier.

        Returns:
            Created QuestionnaireResponse.

        Raises:
            ValueError: If distribution_id is empty or max responses reached.
        """
        start = time.monotonic()

        if not distribution_id or not distribution_id.strip():
            raise ValueError("distribution_id must be non-empty")

        with self._lock:
            if len(self._responses) >= self._max_responses:
                raise ValueError(
                    f"Maximum responses ({self._max_responses}) reached"
                )

        # Parse and normalise answers
        parsed_answers = self._parse_answers(answers)

        response_id = str(uuid.uuid4())
        provenance_hash = self._compute_provenance(
            "submit_response", response_id, distribution_id,
        )

        response = QuestionnaireResponse(
            response_id=response_id,
            distribution_id=distribution_id,
            template_id=template_id,
            supplier_id=supplier_id or f"supplier-{distribution_id[:8]}",
            answers=parsed_answers,
            status=ResponseStatus.IN_PROGRESS,
            language=language,
            completion_pct=0.0,  # Will be recalculated when template known
            provenance_hash=provenance_hash,
        )

        with self._lock:
            self._responses[response_id] = response
            self._stats["responses_submitted"] += 1

        elapsed_ms = (time.monotonic() - start) * 1000
        logger.info(
            "Submitted response %s: distribution=%s answers=%d (%.1f ms)",
            response_id[:8], distribution_id[:8],
            len(parsed_answers), elapsed_ms,
        )
        return response

    def get_response(self, response_id: str) -> QuestionnaireResponse:
        """Get a response by ID.

        Args:
            response_id: Response identifier.

        Returns:
            QuestionnaireResponse.

        Raises:
            ValueError: If response_id is not found.
        """
        return self._get_response_or_raise(response_id)

    def list_responses(
        self,
        template_id: Optional[str] = None,
        supplier_id: Optional[str] = None,
        status: Optional[str] = None,
    ) -> List[QuestionnaireResponse]:
        """List responses with optional filtering.

        Args:
            template_id: Filter by template.
            supplier_id: Filter by supplier.
            status: Filter by status value.

        Returns:
            List of matching QuestionnaireResponse records.
        """
        with self._lock:
            responses = list(self._responses.values())

        if template_id is not None:
            responses = [
                r for r in responses if r.template_id == template_id
            ]
        if supplier_id is not None:
            responses = [
                r for r in responses if r.supplier_id == supplier_id
            ]
        if status is not None:
            responses = [
                r for r in responses if r.status.value == status
            ]

        logger.debug(
            "Listed %d responses (template=%s, supplier=%s, status=%s)",
            len(responses), template_id, supplier_id, status,
        )
        return responses

    def update_response(
        self,
        response_id: str,
        additional_answers: List[Dict[str, Any]],
    ) -> QuestionnaireResponse:
        """Update a response with additional answers (partial save).

        Merges new answers with existing ones. If a question_id
        already has an answer, the new value replaces it.

        Args:
            response_id: Response to update.
            additional_answers: New answers to merge.

        Returns:
            Updated QuestionnaireResponse.

        Raises:
            ValueError: If response_id not found or response not editable.
        """
        start = time.monotonic()
        response = self._get_response_or_raise(response_id)

        editable_statuses = {
            ResponseStatus.DRAFT,
            ResponseStatus.IN_PROGRESS,
            ResponseStatus.REOPENED,
        }
        if response.status not in editable_statuses:
            raise ValueError(
                f"Response {response_id} is not editable "
                f"(status: {response.status.value})"
            )

        # Parse new answers
        new_answers = self._parse_answers(additional_answers)

        # Merge: existing answers keyed by question_id
        answer_map: Dict[str, Answer] = {
            a.question_id: a for a in response.answers
        }
        for new_a in new_answers:
            answer_map[new_a.question_id] = new_a

        merged_answers = list(answer_map.values())

        with self._lock:
            record = self._responses[response_id]
            record.answers = merged_answers
            record.updated_at = _utcnow()
            record.status = ResponseStatus.IN_PROGRESS
            record.provenance_hash = self._compute_provenance(
                "update_response", response_id, str(len(merged_answers)),
            )
            self._stats["responses_updated"] += 1

        elapsed_ms = (time.monotonic() - start) * 1000
        logger.info(
            "Updated response %s: merged to %d answers (%.1f ms)",
            response_id[:8], len(merged_answers), elapsed_ms,
        )
        return self._responses[response_id]

    def finalize_response(self, response_id: str) -> QuestionnaireResponse:
        """Finalize a response, marking it as submitted.

        Sets the status to SUBMITTED and records the submission
        timestamp.

        Args:
            response_id: Response to finalize.

        Returns:
            Finalized QuestionnaireResponse.

        Raises:
            ValueError: If response_id not found or not finalizable.
        """
        response = self._get_response_or_raise(response_id)

        finalizable = {
            ResponseStatus.DRAFT,
            ResponseStatus.IN_PROGRESS,
            ResponseStatus.REOPENED,
        }
        if response.status not in finalizable:
            raise ValueError(
                f"Response {response_id} cannot be finalized "
                f"(status: {response.status.value})"
            )

        now = _utcnow()
        with self._lock:
            record = self._responses[response_id]
            record.status = ResponseStatus.SUBMITTED
            record.submitted_at = now
            record.updated_at = now
            record.provenance_hash = self._compute_provenance(
                "finalize_response", response_id,
            )
            self._stats["responses_finalized"] += 1

        # Auto-acknowledge if configured
        if self._auto_ack:
            self.acknowledge_response(response_id)

        logger.info(
            "Finalized response %s", response_id[:8],
        )
        return self._responses[response_id]

    def import_responses_bulk(
        self,
        file_data: List[Dict[str, Any]],
        template_id: str,
    ) -> List[QuestionnaireResponse]:
        """Bulk import responses from structured data.

        Each entry in file_data should have 'distribution_id',
        'supplier_id', and 'answers' keys.

        Args:
            file_data: List of response dicts to import.
            template_id: Template these responses belong to.

        Returns:
            List of created QuestionnaireResponse records.

        Raises:
            ValueError: If file_data is empty.
        """
        start = time.monotonic()

        if not file_data:
            raise ValueError("file_data must be non-empty")

        imported: List[QuestionnaireResponse] = []
        for entry in file_data:
            dist_id = entry.get("distribution_id", str(uuid.uuid4()))
            supplier_id = entry.get("supplier_id", "")
            answers = entry.get("answers", [])
            language = entry.get("language", "en")

            response = self.submit_response(
                distribution_id=dist_id,
                answers=answers,
                language=language,
                template_id=template_id,
                supplier_id=supplier_id,
            )
            imported.append(response)

        with self._lock:
            self._stats["responses_imported"] += len(imported)

        elapsed_ms = (time.monotonic() - start) * 1000
        logger.info(
            "Bulk imported %d responses for template %s (%.1f ms)",
            len(imported), template_id[:8], elapsed_ms,
        )
        return imported

    def reopen_response(
        self,
        response_id: str,
        reason: str = "",
    ) -> QuestionnaireResponse:
        """Reopen a submitted response for editing.

        Args:
            response_id: Response to reopen.
            reason: Reason for reopening.

        Returns:
            Reopened QuestionnaireResponse.

        Raises:
            ValueError: If response_id not found or not reopenable.
        """
        response = self._get_response_or_raise(response_id)

        reopenable = {
            ResponseStatus.SUBMITTED,
            ResponseStatus.VALIDATED,
            ResponseStatus.SCORED,
        }
        if response.status not in reopenable:
            raise ValueError(
                f"Response {response_id} cannot be reopened "
                f"(status: {response.status.value})"
            )

        with self._lock:
            record = self._responses[response_id]
            record.status = ResponseStatus.REOPENED
            record.updated_at = _utcnow()
            record.provenance_hash = self._compute_provenance(
                "reopen_response", response_id, reason,
            )
            self._stats["responses_reopened"] += 1

        logger.info(
            "Reopened response %s: reason='%s'",
            response_id[:8], reason[:50] if reason else "none",
        )
        return self._responses[response_id]

    def get_response_progress(
        self,
        response_id: str,
        template: Optional[QuestionnaireTemplate] = None,
    ) -> Dict[str, Any]:
        """Get completion progress for a response.

        If a template is provided, calculates completion as answered
        questions / total required questions. Otherwise reports raw
        answer count.

        Args:
            response_id: Response to check progress for.
            template: Optional template for accurate completion calc.

        Returns:
            Dictionary with progress metrics.

        Raises:
            ValueError: If response_id is not found.
        """
        response = self._get_response_or_raise(response_id)

        total_answers = len(response.answers)
        answered_ids = {a.question_id for a in response.answers}

        result: Dict[str, Any] = {
            "response_id": response_id,
            "status": response.status.value,
            "total_answers": total_answers,
            "language": response.language,
        }

        if template is not None:
            # Count total and required questions
            total_questions = 0
            required_questions = 0
            required_answered = 0
            section_progress: Dict[str, Dict[str, Any]] = {}

            for section in template.sections:
                section_total = len(section.questions)
                section_required = sum(
                    1 for q in section.questions if q.required
                )
                section_answered = sum(
                    1 for q in section.questions
                    if q.question_id in answered_ids
                )
                section_req_answered = sum(
                    1 for q in section.questions
                    if q.required and q.question_id in answered_ids
                )

                total_questions += section_total
                required_questions += section_required
                required_answered += section_req_answered

                section_pct = (
                    round(section_answered / section_total * 100, 1)
                    if section_total > 0
                    else 0.0
                )
                section_progress[section.section_id] = {
                    "name": section.name,
                    "total": section_total,
                    "answered": section_answered,
                    "required": section_required,
                    "required_answered": section_req_answered,
                    "completion_pct": section_pct,
                }

            completion_pct = (
                round(required_answered / required_questions * 100, 1)
                if required_questions > 0
                else 0.0
            )

            result["total_questions"] = total_questions
            result["required_questions"] = required_questions
            result["required_answered"] = required_answered
            result["completion_pct"] = completion_pct
            result["section_progress"] = section_progress

            # Update stored completion percentage
            with self._lock:
                self._responses[response_id].completion_pct = completion_pct
        else:
            result["completion_pct"] = response.completion_pct

        return result

    def deduplicate_responses(
        self,
        supplier_id: str,
        template_id: str,
    ) -> List[QuestionnaireResponse]:
        """Find and return duplicate responses for a supplier/template.

        If multiple responses exist for the same supplier and template,
        returns them sorted by most recently updated first. The first
        one is considered the canonical response.

        Args:
            supplier_id: Supplier to check for duplicates.
            template_id: Template to check for duplicates.

        Returns:
            List of duplicate responses (sorted by updated_at desc).
        """
        matches = self.list_responses(
            template_id=template_id, supplier_id=supplier_id,
        )

        # Sort by updated_at descending (most recent first)
        sorted_matches = sorted(
            matches,
            key=lambda r: r.updated_at,
            reverse=True,
        )

        if len(sorted_matches) > 1:
            with self._lock:
                self._stats["responses_deduplicated"] += 1
            logger.info(
                "Found %d responses for supplier %s / template %s",
                len(sorted_matches), supplier_id, template_id[:8],
            )

        return sorted_matches

    def acknowledge_response(self, response_id: str) -> str:
        """Generate an acknowledgement token for a response.

        Args:
            response_id: Response to acknowledge.

        Returns:
            64-character hex confirmation token.

        Raises:
            ValueError: If response_id is not found.
        """
        response = self._get_response_or_raise(response_id)

        token_seed = (
            f"ack:{response_id}:{response.supplier_id}:"
            f"{_utcnow().isoformat()}"
        )
        token = hashlib.sha256(token_seed.encode("utf-8")).hexdigest()

        with self._lock:
            self._responses[response_id].confirmation_token = token
            self._stats["acknowledgements_sent"] += 1

        logger.info(
            "Acknowledged response %s: token=%s...",
            response_id[:8], token[:16],
        )
        return token

    def get_statistics(self) -> Dict[str, Any]:
        """Return engine statistics.

        Returns:
            Dictionary of counter values.
        """
        with self._lock:
            return {
                **self._stats,
                "active_responses": len(self._responses),
                "timestamp": _utcnow().isoformat(),
            }

    # ------------------------------------------------------------------
    # Answer parsing and normalisation
    # ------------------------------------------------------------------

    def _parse_answers(
        self,
        raw_answers: List[Dict[str, Any]],
    ) -> List[Answer]:
        """Parse raw answer dicts into Answer objects with normalisation.

        Args:
            raw_answers: List of answer dictionaries.

        Returns:
            List of Answer objects.
        """
        parsed: List[Answer] = []
        for raw in raw_answers:
            qid = raw.get("question_id", "")
            if not qid:
                logger.warning("Skipping answer with empty question_id")
                continue

            value = raw.get("value", "")
            unit = raw.get("unit", "")
            confidence = raw.get("confidence", 1.0)
            evidence_refs = raw.get("evidence_refs", [])
            notes = raw.get("notes", "")

            # Normalise value if enabled
            if self._normalize:
                value, detected_unit = self._normalize_value(value)
                if detected_unit and not unit:
                    unit = detected_unit

            # Clamp confidence
            confidence = max(0.0, min(1.0, float(confidence)))

            parsed.append(Answer(
                question_id=qid,
                value=value,
                unit=unit,
                confidence=confidence,
                evidence_refs=evidence_refs if isinstance(evidence_refs, list) else [],
                notes=str(notes),
            ))

        return parsed

    def _normalize_value(self, value: Any) -> tuple:
        """Normalize an answer value.

        Attempts to parse numeric strings, clean whitespace, and
        detect units from value text.

        Args:
            value: Raw answer value.

        Returns:
            Tuple of (normalized_value, detected_unit).
        """
        if value is None:
            return "", ""

        if isinstance(value, (int, float)):
            return value, ""

        if isinstance(value, bool):
            return value, ""

        if isinstance(value, list):
            return value, ""

        # String normalisation
        text = str(value).strip()

        # Yes/No normalisation
        lower = text.lower()
        if lower in ("yes", "y", "true", "1"):
            return True, ""
        if lower in ("no", "n", "false", "0"):
            return False, ""

        # Numeric with optional unit
        match = _NUMERIC_PATTERN.match(text)
        if match:
            num_str = match.group(1).replace(",", "")
            detected_unit = match.group(2) or ""
            try:
                if "." in num_str:
                    return float(num_str), detected_unit
                return int(num_str), detected_unit
            except (ValueError, OverflowError):
                pass

        # Percentage normalisation
        if text.endswith("%"):
            pct_str = text[:-1].strip().replace(",", "")
            try:
                return float(pct_str), "%"
            except ValueError:
                pass

        return text, ""

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_response_or_raise(
        self,
        response_id: str,
    ) -> QuestionnaireResponse:
        """Retrieve a response or raise ValueError.

        Args:
            response_id: Response identifier.

        Returns:
            QuestionnaireResponse.

        Raises:
            ValueError: If response_id is not found.
        """
        with self._lock:
            response = self._responses.get(response_id)
        if response is None:
            raise ValueError(f"Unknown response: {response_id}")
        return response

    def _compute_provenance(self, *parts: str) -> str:
        """Compute SHA-256 provenance hash from parts.

        Args:
            *parts: Strings to include in the hash.

        Returns:
            Hex-encoded SHA-256 digest.
        """
        combined = json.dumps(
            {"parts": list(parts), "timestamp": _utcnow().isoformat()},
            sort_keys=True,
        )
        return hashlib.sha256(combined.encode("utf-8")).hexdigest()
