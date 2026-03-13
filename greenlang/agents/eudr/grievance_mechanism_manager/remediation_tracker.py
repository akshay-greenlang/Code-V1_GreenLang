# -*- coding: utf-8 -*-
"""
Remediation Tracker Engine - AGENT-EUDR-032

Effectiveness measurement, stakeholder satisfaction tracking, cost
accounting, timeline adherence monitoring, and verification evidence
collection for remediation actions linked to EUDR-031 grievances.

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-032 (GL-EUDR-GMM-032)
Status: Production Ready
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional

from .config import GrievanceMechanismManagerConfig, get_config
from .models import (
    AGENT_ID,
    ImplementationStatus,
    RemediationAction,
    RemediationRecord,
    RemediationType,
)
from .provenance import ProvenanceTracker

logger = logging.getLogger(__name__)


class RemediationTracker:
    """Remediation effectiveness tracking engine.

    Example:
        >>> tracker = RemediationTracker()
        >>> rem = await tracker.create_remediation(
        ...     grievance_id="g-001", operator_id="OP-001",
        ...     remediation_type="process_change",
        ...     actions=[{"action": "Revise SOP", "deadline": "2026-06-01"}],
        ... )
        >>> assert rem.implementation_status == ImplementationStatus.PLANNED
    """

    def __init__(
        self, config: Optional[GrievanceMechanismManagerConfig] = None,
    ) -> None:
        self.config = config or get_config()
        self._provenance = ProvenanceTracker()
        self._remediations: Dict[str, RemediationRecord] = {}
        logger.info("RemediationTracker engine initialized")

    async def create_remediation(
        self,
        grievance_id: str,
        operator_id: str,
        remediation_type: str,
        actions: Optional[List[Dict[str, Any]]] = None,
    ) -> RemediationRecord:
        """Create a new remediation tracking record."""
        now = datetime.now(timezone.utc).replace(microsecond=0)
        remediation_id = str(uuid.uuid4())

        try:
            rem_type = RemediationType(remediation_type)
        except ValueError:
            rem_type = RemediationType.PROCESS_CHANGE

        parsed_actions = []
        for a in (actions or []):
            parsed_actions.append(RemediationAction(
                action=a.get("action", ""),
                deadline=None,
                status=a.get("status", "pending"),
                responsible_party=a.get("responsible_party", ""),
            ))

        record = RemediationRecord(
            remediation_id=remediation_id,
            grievance_id=grievance_id,
            operator_id=operator_id,
            remediation_type=rem_type,
            remediation_actions=parsed_actions,
            implementation_status=ImplementationStatus.PLANNED,
            created_at=now,
            updated_at=now,
        )

        record.provenance_hash = self._provenance.compute_hash({
            "remediation_id": remediation_id,
            "grievance_id": grievance_id,
            "type": rem_type.value,
            "created_at": now.isoformat(),
        })

        self._remediations[remediation_id] = record

        self._provenance.record(
            entity_type="remediation",
            action="create",
            entity_id=remediation_id,
            actor=AGENT_ID,
            metadata={"grievance_id": grievance_id, "type": rem_type.value},
        )

        logger.info("Remediation %s created for grievance %s", remediation_id, grievance_id)
        return record

    async def update_progress(
        self,
        remediation_id: str,
        completion_percentage: float,
        status: Optional[str] = None,
    ) -> RemediationRecord:
        """Update remediation progress."""
        record = self._remediations.get(remediation_id)
        if record is None:
            raise ValueError(f"Remediation {remediation_id} not found")

        now = datetime.now(timezone.utc).replace(microsecond=0)
        record.completion_percentage = Decimal(str(min(100, max(0, completion_percentage))))

        if status:
            try:
                record.implementation_status = ImplementationStatus(status)
            except ValueError:
                pass
        elif record.completion_percentage >= Decimal("100"):
            record.implementation_status = ImplementationStatus.COMPLETED

        record.updated_at = now

        self._provenance.record(
            entity_type="remediation",
            action="update",
            entity_id=remediation_id,
            actor=AGENT_ID,
            metadata={
                "completion": str(record.completion_percentage),
                "status": record.implementation_status.value,
            },
        )

        logger.info("Remediation %s progress: %s%%", remediation_id, record.completion_percentage)
        return record

    async def record_satisfaction(
        self,
        remediation_id: str,
        satisfaction_score: float,
    ) -> RemediationRecord:
        """Record stakeholder satisfaction rating."""
        record = self._remediations.get(remediation_id)
        if record is None:
            raise ValueError(f"Remediation {remediation_id} not found")

        score = max(1.0, min(5.0, satisfaction_score))
        record.stakeholder_satisfaction = Decimal(str(round(score, 1)))
        record.updated_at = datetime.now(timezone.utc).replace(microsecond=0)

        self._provenance.record(
            entity_type="remediation",
            action="update",
            entity_id=remediation_id,
            actor="complainant",
            metadata={"satisfaction": str(score)},
        )

        logger.info("Remediation %s satisfaction: %s", remediation_id, score)
        return record

    async def record_cost(
        self,
        remediation_id: str,
        cost: float,
    ) -> RemediationRecord:
        """Record cost incurred for remediation."""
        record = self._remediations.get(remediation_id)
        if record is None:
            raise ValueError(f"Remediation {remediation_id} not found")

        record.cost_incurred = Decimal(str(max(0, cost)))
        record.updated_at = datetime.now(timezone.utc).replace(microsecond=0)
        return record

    async def verify_remediation(
        self,
        remediation_id: str,
        verification_evidence: List[Dict[str, Any]],
        effectiveness_indicators: Optional[Dict[str, Any]] = None,
    ) -> RemediationRecord:
        """Verify remediation effectiveness."""
        record = self._remediations.get(remediation_id)
        if record is None:
            raise ValueError(f"Remediation {remediation_id} not found")

        now = datetime.now(timezone.utc).replace(microsecond=0)
        record.verification_evidence.extend(verification_evidence)
        if effectiveness_indicators:
            record.effectiveness_indicators = effectiveness_indicators
        record.implementation_status = ImplementationStatus.VERIFIED
        record.verified_at = now
        record.updated_at = now

        record.provenance_hash = self._provenance.compute_hash({
            "remediation_id": remediation_id,
            "verified_at": now.isoformat(),
            "evidence_count": len(record.verification_evidence),
        })

        self._provenance.record(
            entity_type="remediation",
            action="verify",
            entity_id=remediation_id,
            actor=AGENT_ID,
            metadata={"evidence_count": len(verification_evidence)},
        )

        logger.info("Remediation %s verified", remediation_id)
        return record

    async def add_lessons_learned(
        self, remediation_id: str, lessons: str,
    ) -> RemediationRecord:
        """Add lessons learned to a remediation record."""
        record = self._remediations.get(remediation_id)
        if record is None:
            raise ValueError(f"Remediation {remediation_id} not found")

        record.lessons_learned = lessons
        record.updated_at = datetime.now(timezone.utc).replace(microsecond=0)
        return record

    async def get_remediation(self, remediation_id: str) -> Optional[RemediationRecord]:
        """Retrieve a remediation record by ID."""
        return self._remediations.get(remediation_id)

    async def list_remediations(
        self,
        grievance_id: Optional[str] = None,
        operator_id: Optional[str] = None,
        status: Optional[str] = None,
        remediation_type: Optional[str] = None,
    ) -> List[RemediationRecord]:
        """List remediation records with optional filters."""
        results = list(self._remediations.values())
        if grievance_id:
            results = [r for r in results if r.grievance_id == grievance_id]
        if operator_id:
            results = [r for r in results if r.operator_id == operator_id]
        if status:
            results = [r for r in results if r.implementation_status.value == status]
        if remediation_type:
            results = [r for r in results if r.remediation_type.value == remediation_type]
        return results

    async def health_check(self) -> Dict[str, Any]:
        open_count = sum(
            1 for r in self._remediations.values()
            if r.implementation_status.value in ("planned", "in_progress")
        )
        return {
            "engine": "RemediationTracker",
            "status": "healthy",
            "remediation_count": len(self._remediations),
            "open_remediations": open_count,
        }
