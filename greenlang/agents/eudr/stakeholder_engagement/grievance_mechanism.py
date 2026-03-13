# -*- coding: utf-8 -*-
"""
GrievanceMechanism Engine - AGENT-EUDR-031

Structured complaint management system compliant with UN Guiding
Principles Principle 31 effectiveness criteria. Provides intake,
automated triage, investigation, resolution, satisfaction assessment,
and appeal handling.

Zero-Hallucination: All triage classification and SLA computations
are deterministic. No LLM involvement in grievance lifecycle logic.

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-031 (GL-EUDR-SET-031)
Regulation: EU 2023/1115 (EUDR), UNGP Principle 31
Status: Production Ready
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional

from greenlang.agents.eudr.stakeholder_engagement.config import (
    StakeholderEngagementConfig,
)
from greenlang.agents.eudr.stakeholder_engagement.models import (
    GrievanceRecord,
    GrievanceSeverity,
    GrievanceStatus,
)
from greenlang.agents.eudr.stakeholder_engagement.provenance import (
    ProvenanceTracker,
)

logger = logging.getLogger(__name__)


class GrievanceMechanism:
    """Grievance mechanism engine for stakeholder complaints.

    Manages the full grievance lifecycle: submission, triage,
    investigation, resolution, satisfaction assessment, and appeal.

    Attributes:
        _config: Engine configuration.
        _provenance: Provenance hash chain tracker.
        _grievances: In-memory grievance store (keyed by grievance_id).
    """

    def __init__(self, config: StakeholderEngagementConfig) -> None:
        """Initialize GrievanceMechanism.

        Args:
            config: Stakeholder engagement configuration.
        """
        self._config = config
        self._provenance = ProvenanceTracker()
        self._grievances: Dict[str, GrievanceRecord] = {}
        logger.info("GrievanceMechanism initialized")

    async def submit_grievance(
        self,
        stakeholder_id: str,
        operator_id: str,
        title: str,
        description: str,
        severity: GrievanceSeverity,
        channel: str,
        category: Optional[str] = None,
    ) -> GrievanceRecord:
        """Submit a new grievance.

        Args:
            stakeholder_id: Stakeholder filing the complaint.
            operator_id: Operator against whom complaint is filed.
            title: Grievance title.
            description: Detailed description.
            severity: Severity classification.
            channel: Intake channel.
            category: Optional grievance category.

        Returns:
            Newly created GrievanceRecord.

        Raises:
            ValueError: If required fields are empty.
        """
        if not stakeholder_id or not stakeholder_id.strip():
            raise ValueError("stakeholder_id is required")
        if not operator_id or not operator_id.strip():
            raise ValueError("operator_id is required")
        if not title or not title.strip():
            raise ValueError("title is required")

        now = datetime.now(tz=timezone.utc)
        grievance_id = f"GRV-{uuid.uuid4().hex[:8].upper()}"
        sla_deadline = self._calculate_sla_deadline(severity, now)

        grievance = GrievanceRecord(
            grievance_id=grievance_id,
            stakeholder_id=stakeholder_id,
            operator_id=operator_id,
            title=title,
            description=description,
            severity=severity,
            status=GrievanceStatus.SUBMITTED,
            channel=channel,
            category=category or "",
            submitted_at=now,
            sla_deadline=sla_deadline,
            investigation_notes=[],
            resolution_actions=[],
        )

        self._grievances[grievance_id] = grievance
        self._provenance.record(
            "grievance", "submit", grievance_id, "AGENT-EUDR-031",
            metadata={"severity": severity.value},
        )
        logger.info("Grievance %s submitted (severity=%s)", grievance_id, severity.value)
        return grievance

    async def triage_grievance(
        self,
        grievance_id: str,
        assigned_to: str,
        priority_notes: Optional[str] = None,
        escalate_to: Optional[GrievanceSeverity] = None,
    ) -> GrievanceRecord:
        """Triage a submitted grievance.

        Args:
            grievance_id: Grievance to triage.
            assigned_to: Investigator assignment.
            priority_notes: Optional priority notes.
            escalate_to: Optional severity escalation.

        Returns:
            Updated GrievanceRecord.

        Raises:
            ValueError: If grievance not found or fields empty.
        """
        if not grievance_id or not grievance_id.strip():
            raise ValueError("grievance_id is required")
        if not assigned_to or not assigned_to.strip():
            raise ValueError("assigned_to is required")

        grievance = self._get_grievance(grievance_id)
        now = datetime.now(tz=timezone.utc)

        grievance.status = GrievanceStatus.TRIAGED
        grievance.assigned_to = assigned_to
        grievance.triaged_at = now

        if escalate_to is not None:
            grievance.severity = escalate_to
            grievance.sla_deadline = self._calculate_sla_deadline(escalate_to, grievance.submitted_at)

        self._provenance.record(
            "grievance", "triage", grievance_id, "AGENT-EUDR-031",
            metadata={"assigned_to": assigned_to},
        )
        logger.info("Grievance %s triaged, assigned to %s", grievance_id, assigned_to)
        return grievance

    async def investigate(
        self,
        grievance_id: str,
        notes: List[Dict[str, Any]],
    ) -> GrievanceRecord:
        """Record investigation notes for a grievance.

        Args:
            grievance_id: Grievance under investigation.
            notes: Investigation notes to add.

        Returns:
            Updated GrievanceRecord.

        Raises:
            ValueError: If grievance not found or notes empty.
        """
        if not notes:
            raise ValueError("notes are required")

        grievance = self._get_grievance(grievance_id)
        grievance.status = GrievanceStatus.INVESTIGATING
        grievance.investigation_notes.extend(notes)

        self._provenance.record(
            "grievance", "investigate", grievance_id, "AGENT-EUDR-031",
        )
        logger.info("Grievance %s investigation updated (%d notes)", grievance_id, len(notes))
        return grievance

    async def resolve(
        self,
        grievance_id: str,
        resolution_actions: List[Dict[str, Any]],
        resolution_summary: str,
    ) -> GrievanceRecord:
        """Resolve a grievance with corrective actions.

        Args:
            grievance_id: Grievance to resolve.
            resolution_actions: Actions taken to resolve.
            resolution_summary: Summary of resolution.

        Returns:
            Updated GrievanceRecord.

        Raises:
            ValueError: If grievance not found or required fields empty.
        """
        grievance = self._get_grievance(grievance_id)

        if not resolution_actions:
            raise ValueError("resolution_actions are required")
        if not resolution_summary or not resolution_summary.strip():
            raise ValueError("resolution_summary is required")
        now = datetime.now(tz=timezone.utc)

        grievance.status = GrievanceStatus.RESOLVED
        grievance.resolution_actions.extend(resolution_actions)
        grievance.resolution_summary = resolution_summary
        grievance.resolved_at = now

        self._provenance.record(
            "grievance", "resolve", grievance_id, "AGENT-EUDR-031",
        )
        logger.info("Grievance %s resolved", grievance_id)
        return grievance

    async def assess_satisfaction(
        self,
        grievance_id: str,
        satisfaction_score: Decimal,
        feedback: str,
    ) -> Dict[str, Any]:
        """Assess post-resolution satisfaction.

        Args:
            grievance_id: Resolved grievance.
            satisfaction_score: Score between 0 and 100.
            feedback: Satisfaction feedback text.

        Returns:
            Satisfaction assessment dictionary.

        Raises:
            ValueError: If grievance not found or score out of bounds.
        """
        if satisfaction_score < Decimal("0") or satisfaction_score > Decimal("100"):
            raise ValueError("score must be between 0 and 100")

        grievance = self._get_grievance(grievance_id)

        grievance.satisfaction_score = satisfaction_score
        grievance.satisfaction_feedback = feedback

        self._provenance.record(
            "grievance", "assess_satisfaction", grievance_id, "AGENT-EUDR-031",
        )

        return {
            "grievance_id": grievance_id,
            "satisfaction_score": satisfaction_score,
            "feedback": feedback,
            "assessed_at": datetime.now(tz=timezone.utc).isoformat(),
        }

    async def appeal(
        self,
        grievance_id: str,
        reason: str,
    ) -> GrievanceRecord:
        """Appeal a resolved grievance.

        Args:
            grievance_id: Grievance to appeal.
            reason: Reason for the appeal.

        Returns:
            Updated GrievanceRecord.

        Raises:
            ValueError: If grievance not found, not resolved, or reason empty.
        """
        if not reason or not reason.strip():
            raise ValueError("reason is required")

        grievance = self._get_grievance(grievance_id)

        if grievance.status not in (
            GrievanceStatus.RESOLVED,
            GrievanceStatus.CLOSED,
            GrievanceStatus.APPEALED,
        ):
            raise ValueError("must be resolved before appeal")

        grievance.status = GrievanceStatus.APPEALED
        grievance.appeal_reason = reason

        self._provenance.record(
            "grievance", "appeal", grievance_id, "AGENT-EUDR-031",
            metadata={"reason": reason},
        )
        logger.info("Grievance %s appealed", grievance_id)
        return grievance

    def _calculate_sla_deadline(
        self,
        severity: GrievanceSeverity,
        from_time: datetime,
    ) -> datetime:
        """Calculate SLA deadline based on severity.

        Args:
            severity: Grievance severity level.
            from_time: Base time for SLA calculation.

        Returns:
            SLA deadline datetime.
        """
        sla_map = {
            GrievanceSeverity.CRITICAL: timedelta(hours=self._config.grievance_sla_critical_hours),
            GrievanceSeverity.HIGH: timedelta(hours=self._config.grievance_sla_high_hours),
            GrievanceSeverity.STANDARD: timedelta(days=self._config.grievance_sla_standard_days),
            GrievanceSeverity.MINOR: timedelta(days=self._config.grievance_sla_minor_days),
        }
        delta = sla_map.get(severity, timedelta(days=14))
        return from_time + delta

    def _get_grievance(self, grievance_id: str) -> GrievanceRecord:
        """Get grievance by ID or raise ValueError."""
        if grievance_id not in self._grievances:
            raise ValueError(f"grievance not found: {grievance_id}")
        return self._grievances[grievance_id]
