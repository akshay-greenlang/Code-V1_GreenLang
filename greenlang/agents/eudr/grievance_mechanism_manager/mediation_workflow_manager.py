# -*- coding: utf-8 -*-
"""
Mediation Workflow Manager Engine - AGENT-EUDR-032

Multi-party mediation state machine with 7-stage workflow: initiated ->
preparation -> dialogue -> negotiation -> settlement -> implementation ->
closed. Tracks session records, agreements, settlement terms, and
compliance with CSDDD Article 8 mediation requirements.

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-032 (GL-EUDR-GMM-032)
Status: Production Ready
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .config import GrievanceMechanismManagerConfig, get_config
from .models import (
    AGENT_ID,
    MEDIATION_STAGES_ORDERED,
    MediationRecord,
    MediationSession,
    MediationStage,
    MediatorType,
    SettlementStatus,
)
from .provenance import ProvenanceTracker

logger = logging.getLogger(__name__)


class MediationWorkflowManager:
    """Multi-party mediation workflow state machine.

    Example:
        >>> manager = MediationWorkflowManager()
        >>> med = await manager.initiate_mediation(
        ...     grievance_id="g-001", operator_id="OP-001",
        ...     parties=[{"role": "complainant", "name": "Community A"}],
        ... )
        >>> assert med.mediation_stage == MediationStage.INITIATED
    """

    def __init__(
        self, config: Optional[GrievanceMechanismManagerConfig] = None,
    ) -> None:
        self.config = config or get_config()
        self._provenance = ProvenanceTracker()
        self._mediations: Dict[str, MediationRecord] = {}
        logger.info("MediationWorkflowManager engine initialized")

    async def initiate_mediation(
        self,
        grievance_id: str,
        operator_id: str,
        parties: List[Dict[str, Any]],
        mediator_type: str = "internal",
        mediator_id: Optional[str] = None,
    ) -> MediationRecord:
        """Initiate a new mediation workflow."""
        now = datetime.now(timezone.utc).replace(microsecond=0)
        mediation_id = str(uuid.uuid4())

        try:
            med_type = MediatorType(mediator_type)
        except ValueError:
            med_type = MediatorType.INTERNAL

        record = MediationRecord(
            mediation_id=mediation_id,
            grievance_id=grievance_id,
            operator_id=operator_id,
            mediation_stage=MediationStage.INITIATED,
            parties=parties,
            mediator_id=mediator_id,
            mediator_type=med_type,
            settlement_status=SettlementStatus.PENDING,
            created_at=now,
            updated_at=now,
        )

        record.provenance_hash = self._provenance.compute_hash({
            "mediation_id": mediation_id,
            "grievance_id": grievance_id,
            "created_at": now.isoformat(),
        })

        self._mediations[mediation_id] = record

        self._provenance.record(
            entity_type="mediation",
            action="create",
            entity_id=mediation_id,
            actor=AGENT_ID,
            metadata={"grievance_id": grievance_id, "mediator_type": med_type.value},
        )

        logger.info("Mediation %s initiated for grievance %s", mediation_id, grievance_id)
        return record

    async def advance_stage(
        self, mediation_id: str, target_stage: Optional[str] = None,
    ) -> MediationRecord:
        """Advance mediation to the next stage."""
        record = self._mediations.get(mediation_id)
        if record is None:
            raise ValueError(f"Mediation {mediation_id} not found")

        current_idx = MEDIATION_STAGES_ORDERED.index(record.mediation_stage)

        if target_stage:
            try:
                target = MediationStage(target_stage)
            except ValueError:
                raise ValueError(f"Invalid mediation stage: {target_stage}")
            target_idx = MEDIATION_STAGES_ORDERED.index(target)
            if target_idx <= current_idx:
                raise ValueError(
                    f"Cannot move from {record.mediation_stage.value} to "
                    f"{target_stage} (must advance forward)"
                )
        else:
            if current_idx >= len(MEDIATION_STAGES_ORDERED) - 1:
                raise ValueError("Mediation is already in final stage (closed)")
            target = MEDIATION_STAGES_ORDERED[current_idx + 1]

        now = datetime.now(timezone.utc).replace(microsecond=0)
        old_stage = record.mediation_stage
        record.mediation_stage = target
        record.updated_at = now

        if target == MediationStage.CLOSED:
            record.completed_at = now

        self._provenance.record(
            entity_type="mediation",
            action="advance_stage",
            entity_id=mediation_id,
            actor=AGENT_ID,
            metadata={"from": old_stage.value, "to": target.value},
        )

        logger.info(
            "Mediation %s advanced: %s -> %s",
            mediation_id, old_stage.value, target.value,
        )
        return record

    async def record_session(
        self,
        mediation_id: str,
        session_data: Dict[str, Any],
    ) -> MediationRecord:
        """Record a mediation session."""
        record = self._mediations.get(mediation_id)
        if record is None:
            raise ValueError(f"Mediation {mediation_id} not found")

        if record.session_count >= self.config.mediation_max_sessions:
            raise ValueError(
                f"Maximum sessions ({self.config.mediation_max_sessions}) reached"
            )

        now = datetime.now(timezone.utc).replace(microsecond=0)
        session_num = record.session_count + 1
        duration = session_data.get("duration_minutes", self.config.mediation_default_session_minutes)

        session = MediationSession(
            session_number=session_num,
            date=now,
            duration_minutes=duration,
            summary=session_data.get("summary", ""),
            attendees=session_data.get("attendees", []),
            outcomes=session_data.get("outcomes", []),
        )

        record.sessions.append(session)
        record.session_count = session_num
        record.total_duration_minutes += duration
        record.updated_at = now

        self._provenance.record(
            entity_type="mediation",
            action="update",
            entity_id=mediation_id,
            actor=AGENT_ID,
            metadata={"session_number": session_num, "duration_minutes": duration},
        )

        logger.info("Mediation %s session %d recorded (%d min)", mediation_id, session_num, duration)
        return record

    async def record_agreement(
        self,
        mediation_id: str,
        agreement_data: Dict[str, Any],
    ) -> MediationRecord:
        """Record a mediation agreement."""
        record = self._mediations.get(mediation_id)
        if record is None:
            raise ValueError(f"Mediation {mediation_id} not found")

        now = datetime.now(timezone.utc).replace(microsecond=0)
        agreement_data["recorded_at"] = now.isoformat()
        record.agreements.append(agreement_data)
        record.updated_at = now

        self._provenance.record(
            entity_type="mediation",
            action="update",
            entity_id=mediation_id,
            actor=AGENT_ID,
            metadata={"agreement_count": len(record.agreements)},
        )

        logger.info("Mediation %s agreement recorded", mediation_id)
        return record

    async def set_settlement(
        self,
        mediation_id: str,
        settlement_terms: Dict[str, Any],
        status: str = "accepted",
    ) -> MediationRecord:
        """Set settlement terms and status."""
        record = self._mediations.get(mediation_id)
        if record is None:
            raise ValueError(f"Mediation {mediation_id} not found")

        try:
            settlement_status = SettlementStatus(status)
        except ValueError:
            settlement_status = SettlementStatus.PENDING

        now = datetime.now(timezone.utc).replace(microsecond=0)
        record.settlement_terms = settlement_terms
        record.settlement_status = settlement_status
        record.updated_at = now

        self._provenance.record(
            entity_type="mediation",
            action="update",
            entity_id=mediation_id,
            actor=AGENT_ID,
            metadata={"settlement_status": settlement_status.value},
        )

        logger.info("Mediation %s settlement: %s", mediation_id, settlement_status.value)
        return record

    async def get_mediation(self, mediation_id: str) -> Optional[MediationRecord]:
        """Retrieve a mediation record by ID."""
        return self._mediations.get(mediation_id)

    async def list_mediations(
        self,
        operator_id: Optional[str] = None,
        grievance_id: Optional[str] = None,
        stage: Optional[str] = None,
    ) -> List[MediationRecord]:
        """List mediation records with optional filters."""
        results = list(self._mediations.values())
        if operator_id:
            results = [r for r in results if r.operator_id == operator_id]
        if grievance_id:
            results = [r for r in results if r.grievance_id == grievance_id]
        if stage:
            results = [r for r in results if r.mediation_stage.value == stage]
        return results

    async def health_check(self) -> Dict[str, Any]:
        active = sum(
            1 for m in self._mediations.values()
            if m.mediation_stage != MediationStage.CLOSED
        )
        return {
            "engine": "MediationWorkflowManager",
            "status": "healthy",
            "mediation_count": len(self._mediations),
            "active_mediations": active,
        }
