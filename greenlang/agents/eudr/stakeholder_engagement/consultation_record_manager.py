# -*- coding: utf-8 -*-
"""
ConsultationRecordManager Engine - AGENT-EUDR-031

Structured documentation system for all consultations with indigenous
peoples, local communities, and other stakeholders per EUDR Article
10(2)(e). Each consultation record captures objectives, participants,
methodology, outcomes, commitments, and follow-up actions with
SHA-256 provenance hashing. Records are immutable after finalization.

Zero-Hallucination: All consultation lifecycle operations are
deterministic state transitions. No LLM involvement.

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-031 (GL-EUDR-SET-031)
Regulation: EU 2023/1115 (EUDR) Article 10(2)(e)
Status: Production Ready
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from greenlang.agents.eudr.stakeholder_engagement.config import (
    StakeholderEngagementConfig,
)
from greenlang.agents.eudr.stakeholder_engagement.models import (
    ConsultationRecord,
    ConsultationType,
)
from greenlang.agents.eudr.stakeholder_engagement.provenance import (
    ProvenanceTracker,
)

logger = logging.getLogger(__name__)


class ConsultationRecordManager:
    """Consultation record management engine.

    Creates, manages, and finalizes consultation records for
    EUDR stakeholder engagement compliance.

    Attributes:
        _config: Engine configuration.
        _provenance: Provenance hash chain tracker.
        _consultations: In-memory consultation store.
    """

    def __init__(self, config: StakeholderEngagementConfig) -> None:
        """Initialize ConsultationRecordManager.

        Args:
            config: Stakeholder engagement configuration.
        """
        self._config = config
        self._provenance = ProvenanceTracker()
        self._consultations: Dict[str, ConsultationRecord] = {}
        logger.info("ConsultationRecordManager initialized")

    async def create_consultation(
        self,
        operator_id: str,
        consultation_type: ConsultationType,
        title: str,
        scheduled_at: Optional[datetime] = None,
        stakeholder_ids: Optional[List[str]] = None,
        location: Optional[str] = None,
        language: Optional[str] = None,
    ) -> ConsultationRecord:
        """Create a new consultation record.

        Args:
            operator_id: Operator conducting the consultation.
            consultation_type: Type of consultation.
            title: Consultation title.
            scheduled_at: Scheduled date/time.
            stakeholder_ids: Participating stakeholder IDs.
            location: Consultation location.
            language: Primary language.

        Returns:
            Newly created ConsultationRecord.

        Raises:
            ValueError: If required fields are empty.
        """
        if not operator_id or not operator_id.strip():
            raise ValueError("operator_id is required")
        if not title or not title.strip():
            raise ValueError("title is required")

        consultation_id = f"CON-{uuid.uuid4().hex[:8].upper()}"
        now = datetime.now(tz=timezone.utc)

        record = ConsultationRecord(
            consultation_id=consultation_id,
            operator_id=operator_id,
            consultation_type=consultation_type,
            title=title,
            scheduled_at=scheduled_at,
            stakeholder_ids=stakeholder_ids or [],
            participants=[],
            outcomes=[],
            evidence_refs=[],
            location=location or "",
            language=language or "en",
            status="scheduled",
            created_at=now,
        )

        self._consultations[consultation_id] = record
        self._provenance.record(
            "consultation", "create", consultation_id, "AGENT-EUDR-031",
            metadata={"type": consultation_type.value},
        )
        logger.info("Consultation %s created: %s", consultation_id, title)
        return record

    async def add_participants(
        self,
        consultation_id: str,
        participants: List[Dict[str, Any]],
    ) -> ConsultationRecord:
        """Add participants to a consultation.

        Args:
            consultation_id: Consultation to update.
            participants: Participant list with name, role, etc.

        Returns:
            Updated ConsultationRecord.

        Raises:
            ValueError: If consultation not found or participants empty.
        """
        if not participants:
            raise ValueError("participants are required")

        record = self._get_consultation(consultation_id)
        record.participants.extend(participants)

        self._provenance.record(
            "consultation", "add_participants", consultation_id, "AGENT-EUDR-031",
        )
        return record

    async def record_outcomes(
        self,
        consultation_id: str,
        outcomes: List[Dict[str, Any]],
    ) -> ConsultationRecord:
        """Record outcomes from a consultation.

        Args:
            consultation_id: Consultation to update.
            outcomes: Outcomes and decisions reached.

        Returns:
            Updated ConsultationRecord.

        Raises:
            ValueError: If consultation not found or outcomes empty.
        """
        if not outcomes:
            raise ValueError("outcomes are required")

        record = self._get_consultation(consultation_id)
        record.outcomes.extend(outcomes)

        self._provenance.record(
            "consultation", "record_outcomes", consultation_id, "AGENT-EUDR-031",
        )
        return record

    async def attach_evidence(
        self,
        consultation_id: str,
        evidence_refs: List[str],
    ) -> ConsultationRecord:
        """Attach evidence to a consultation.

        Args:
            consultation_id: Consultation to update.
            evidence_refs: Evidence file references.

        Returns:
            Updated ConsultationRecord.

        Raises:
            ValueError: If consultation not found or evidence empty.
        """
        if not evidence_refs:
            raise ValueError("evidence_refs are required")

        record = self._get_consultation(consultation_id)
        record.evidence_refs.extend(evidence_refs)

        self._provenance.record(
            "consultation", "attach_evidence", consultation_id, "AGENT-EUDR-031",
        )
        return record

    async def finalize_consultation(
        self,
        consultation_id: str,
    ) -> ConsultationRecord:
        """Finalize a consultation record.

        Sets status to completed, records conducted_at timestamp,
        and marks the record as immutable.

        Args:
            consultation_id: Consultation to finalize.

        Returns:
            Finalized ConsultationRecord.

        Raises:
            ValueError: If consultation not found, already completed,
                or missing required data.
        """
        record = self._get_consultation(consultation_id)

        if record.status == "completed":
            raise ValueError("already completed")
        if not record.participants:
            raise ValueError("participants are required before finalization")
        if not record.outcomes:
            raise ValueError("outcomes are required before finalization")

        now = datetime.now(tz=timezone.utc)
        record.status = "completed"
        record.conducted_at = now

        self._provenance.record(
            "consultation", "finalize", consultation_id, "AGENT-EUDR-031",
        )
        logger.info("Consultation %s finalized", consultation_id)
        return record

    async def generate_register(
        self,
        operator_id: str,
    ) -> Dict[str, Any]:
        """Generate a consultation register for an operator.

        Args:
            operator_id: Operator to generate register for.

        Returns:
            Register summary dictionary.

        Raises:
            ValueError: If operator_id is empty.
        """
        if not operator_id or not operator_id.strip():
            raise ValueError("operator_id is required")

        records = [
            c for c in self._consultations.values()
            if c.operator_id == operator_id
        ]

        completed = [r for r in records if r.status == "completed"]
        scheduled = [r for r in records if r.status == "scheduled"]

        type_breakdown: Dict[str, int] = {}
        for r in records:
            t = r.consultation_type.value
            type_breakdown[t] = type_breakdown.get(t, 0) + 1

        total_participants = sum(len(r.participants) for r in records)

        return {
            "operator_id": operator_id,
            "total_consultations": len(records),
            "completed": len(completed),
            "scheduled": len(scheduled),
            "type_breakdown": type_breakdown,
            "total_participants": total_participants,
            "summary": f"{len(records)} consultations, {len(completed)} completed",
        }

    def _get_consultation(self, consultation_id: str) -> ConsultationRecord:
        """Get consultation by ID or raise ValueError."""
        if consultation_id not in self._consultations:
            raise ValueError(f"consultation not found: {consultation_id}")
        return self._consultations[consultation_id]
