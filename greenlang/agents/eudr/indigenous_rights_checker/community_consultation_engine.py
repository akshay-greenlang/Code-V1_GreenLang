# -*- coding: utf-8 -*-
"""
CommunityConsultationEngine - Feature 4: Community Consultation Tracker

Tracks community consultation lifecycle through 7 stages (IDENTIFIED,
NOTIFIED, INFORMATION_SHARED, CONSULTATION_HELD, RESPONSE_RECORDED,
AGREEMENT_REACHED, MONITORING_ACTIVE). Records meeting details, manages
grievances with SLA tracking, and handles benefit-sharing agreements.

Per PRD F4.1-F4.10: Complete audit trail for all consultation activities.

Example:
    >>> engine = CommunityConsultationEngine(config, provenance)
    >>> record = await engine.record_consultation(
    ...     community_id="c-001", stage="consultation_held",
    ...     meeting_date="2026-03-01", attendees=[...],
    ... )

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-021 (Feature 4)
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from greenlang.agents.eudr.indigenous_rights_checker.config import (
    IndigenousRightsCheckerConfig,
)
from greenlang.agents.eudr.indigenous_rights_checker.models import (
    ConsultationRecord,
    ConsultationStage,
    GrievanceRecord,
    GrievanceStatus,
    AlertSeverity,
    BenefitSharingAgreement,
)
from greenlang.agents.eudr.indigenous_rights_checker.provenance import (
    ProvenanceTracker,
)
from greenlang.agents.eudr.indigenous_rights_checker.metrics import (
    record_consultation_recorded,
)

logger = logging.getLogger(__name__)

_SQL_INSERT_CONSULTATION = """
    INSERT INTO eudr_indigenous_rights_checker.gl_eudr_irc_consultations (
        consultation_id, community_id, plot_id, territory_id,
        consultation_stage, meeting_date, meeting_location,
        attendees, agenda, minutes, outcomes, follow_up_actions,
        documents_shared, community_response, grievance_id,
        provenance_hash
    ) VALUES (
        %(consultation_id)s, %(community_id)s, %(plot_id)s,
        %(territory_id)s, %(consultation_stage)s, %(meeting_date)s,
        %(meeting_location)s, %(attendees)s, %(agenda)s, %(minutes)s,
        %(outcomes)s, %(follow_up_actions)s, %(documents_shared)s,
        %(community_response)s, %(grievance_id)s, %(provenance_hash)s
    )
"""

_SQL_INSERT_GRIEVANCE = """
    INSERT INTO eudr_indigenous_rights_checker.gl_eudr_irc_grievances (
        grievance_id, community_id, territory_id, grievance_type,
        description, severity, status, submitted_at,
        investigation_deadline, resolution_deadline, provenance_hash
    ) VALUES (
        %(grievance_id)s, %(community_id)s, %(territory_id)s,
        %(grievance_type)s, %(description)s, %(severity)s,
        %(status)s, %(submitted_at)s,
        %(investigation_deadline)s, %(resolution_deadline)s,
        %(provenance_hash)s
    )
"""

_SQL_GET_CONSULTATIONS = """
    SELECT consultation_id, community_id, plot_id, territory_id,
           consultation_stage, meeting_date, meeting_location,
           attendees, agenda, minutes, outcomes, provenance_hash,
           created_at
    FROM eudr_indigenous_rights_checker.gl_eudr_irc_consultations
    WHERE community_id = %(community_id)s
    ORDER BY created_at DESC
    LIMIT %(limit)s
"""


class CommunityConsultationEngine:
    """Engine for tracking community consultation lifecycle.

    Manages consultation records, grievances, and benefit-sharing
    agreements with full audit trail and SLA tracking.

    Attributes:
        _config: Agent configuration with SLA timelines.
        _provenance: Provenance tracker for audit trail.
        _pool: Async database connection pool.
    """

    def __init__(
        self,
        config: IndigenousRightsCheckerConfig,
        provenance: ProvenanceTracker,
    ) -> None:
        """Initialize CommunityConsultationEngine."""
        self._config = config
        self._provenance = provenance
        self._pool: Any = None
        logger.info("CommunityConsultationEngine initialized")

    async def startup(self, pool: Any) -> None:
        """Set the database connection pool."""
        self._pool = pool

    async def shutdown(self) -> None:
        """Clean up engine resources."""
        self._pool = None

    async def record_consultation(
        self,
        community_id: str,
        consultation_stage: str,
        plot_id: Optional[str] = None,
        territory_id: Optional[str] = None,
        meeting_date: Optional[str] = None,
        meeting_location: Optional[str] = None,
        attendees: Optional[List[Dict[str, Any]]] = None,
        agenda: Optional[str] = None,
        minutes: Optional[str] = None,
        outcomes: Optional[str] = None,
        follow_up_actions: Optional[List[Dict[str, Any]]] = None,
        documents_shared: Optional[List[Dict[str, Any]]] = None,
        community_response: Optional[str] = None,
    ) -> ConsultationRecord:
        """Record a consultation activity with full audit trail.

        Args:
            community_id: Community identifier.
            consultation_stage: Lifecycle stage (ConsultationStage value).
            plot_id: Optional related plot identifier.
            territory_id: Optional related territory identifier.
            meeting_date: Meeting date string (ISO 8601).
            meeting_location: Meeting location description.
            attendees: List of attendee dictionaries with roles.
            agenda: Meeting agenda text.
            minutes: Meeting minutes text.
            outcomes: Meeting outcomes summary.
            follow_up_actions: List of follow-up action items.
            documents_shared: List of documents shared with community.
            community_response: Community's response/feedback.

        Returns:
            ConsultationRecord with provenance hash.
        """
        consultation_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)

        provenance_hash = self._provenance.compute_data_hash({
            "consultation_id": consultation_id,
            "community_id": community_id,
            "stage": consultation_stage,
            "created_at": now.isoformat(),
        })

        record = ConsultationRecord(
            consultation_id=consultation_id,
            community_id=community_id,
            plot_id=plot_id,
            territory_id=territory_id,
            consultation_stage=ConsultationStage(consultation_stage),
            meeting_date=(
                datetime.fromisoformat(meeting_date).date()
                if meeting_date else None
            ),
            meeting_location=meeting_location,
            attendees=attendees or [],
            agenda=agenda,
            minutes=minutes,
            outcomes=outcomes,
            follow_up_actions=follow_up_actions or [],
            documents_shared=documents_shared or [],
            community_response=community_response,
            provenance_hash=provenance_hash,
            created_at=now,
        )

        self._provenance.record(
            "consultation", "create", consultation_id,
            metadata={
                "community_id": community_id,
                "stage": consultation_stage,
            },
        )

        record_consultation_recorded(consultation_stage)
        await self._persist_consultation(record)

        logger.info(
            f"Consultation recorded: {consultation_id} "
            f"community={community_id} stage={consultation_stage}"
        )

        return record

    async def submit_grievance(
        self,
        community_id: str,
        grievance_type: str,
        description: str,
        severity: str = "medium",
        territory_id: Optional[str] = None,
    ) -> GrievanceRecord:
        """Submit a community grievance with SLA deadlines.

        Automatically calculates SLA deadlines based on configured
        grievance_sla_days (acknowledge=5d, investigate=30d, resolve=90d).

        Args:
            community_id: Community identifier.
            grievance_type: Type of grievance.
            description: Detailed grievance description.
            severity: Severity level (low, medium, high, critical).
            territory_id: Optional related territory identifier.

        Returns:
            GrievanceRecord with calculated SLA deadlines.
        """
        grievance_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)

        sla = self._config.grievance_sla_days
        investigation_deadline = now + timedelta(days=sla.get("investigate", 30))
        resolution_deadline = now + timedelta(days=sla.get("resolve", 90))

        provenance_hash = self._provenance.compute_data_hash({
            "grievance_id": grievance_id,
            "community_id": community_id,
            "grievance_type": grievance_type,
            "submitted_at": now.isoformat(),
        })

        record = GrievanceRecord(
            grievance_id=grievance_id,
            community_id=community_id,
            territory_id=territory_id,
            grievance_type=grievance_type,
            description=description,
            severity=AlertSeverity(severity),
            status=GrievanceStatus.SUBMITTED,
            submitted_at=now,
            investigation_deadline=investigation_deadline,
            resolution_deadline=resolution_deadline,
            provenance_hash=provenance_hash,
        )

        self._provenance.record(
            "grievance", "create", grievance_id,
            metadata={
                "community_id": community_id,
                "severity": severity,
            },
        )

        await self._persist_grievance(record)

        logger.info(
            f"Grievance submitted: {grievance_id} "
            f"community={community_id} type={grievance_type}"
        )

        return record

    async def get_consultations_for_community(
        self, community_id: str, limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Get consultation records for a community.

        Args:
            community_id: Community identifier.
            limit: Maximum records to return.

        Returns:
            List of consultation summary dictionaries.
        """
        if self._pool is None:
            return []

        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    _SQL_GET_CONSULTATIONS,
                    {"community_id": community_id, "limit": limit},
                )
                rows = await cur.fetchall()

        return [
            {
                "consultation_id": str(row[0]),
                "community_id": str(row[1]),
                "plot_id": str(row[2]) if row[2] else None,
                "territory_id": str(row[3]) if row[3] else None,
                "consultation_stage": row[4],
                "meeting_date": str(row[5]) if row[5] else None,
                "meeting_location": row[6],
                "provenance_hash": row[11],
                "created_at": row[12].isoformat() if row[12] else None,
            }
            for row in rows
        ]

    async def _persist_consultation(self, record: ConsultationRecord) -> None:
        """Persist consultation record to database."""
        if self._pool is None:
            return

        import json
        params = {
            "consultation_id": record.consultation_id,
            "community_id": record.community_id,
            "plot_id": record.plot_id,
            "territory_id": record.territory_id,
            "consultation_stage": record.consultation_stage.value,
            "meeting_date": record.meeting_date,
            "meeting_location": record.meeting_location,
            "attendees": json.dumps(record.attendees),
            "agenda": record.agenda,
            "minutes": record.minutes,
            "outcomes": record.outcomes,
            "follow_up_actions": json.dumps(record.follow_up_actions),
            "documents_shared": json.dumps(record.documents_shared),
            "community_response": record.community_response,
            "grievance_id": record.grievance_id,
            "provenance_hash": record.provenance_hash,
        }

        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(_SQL_INSERT_CONSULTATION, params)
            await conn.commit()

    async def _persist_grievance(self, record: GrievanceRecord) -> None:
        """Persist grievance record to database."""
        if self._pool is None:
            return

        params = {
            "grievance_id": record.grievance_id,
            "community_id": record.community_id,
            "territory_id": record.territory_id,
            "grievance_type": record.grievance_type,
            "description": record.description,
            "severity": record.severity.value,
            "status": record.status.value,
            "submitted_at": record.submitted_at,
            "investigation_deadline": record.investigation_deadline,
            "resolution_deadline": record.resolution_deadline,
            "provenance_hash": record.provenance_hash,
        }

        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(_SQL_INSERT_GRIEVANCE, params)
            await conn.commit()
