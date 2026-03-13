# -*- coding: utf-8 -*-
"""
Collective Grievance Handler Engine - AGENT-EUDR-032

Class-action/collective complaint management with demand tracking,
negotiation workflow, representative body coordination, and
stakeholder consolidation.

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
    CollectiveDemand,
    CollectiveGrievanceRecord,
    CollectiveStatus,
    NegotiationStatus,
)
from .provenance import ProvenanceTracker

logger = logging.getLogger(__name__)


class CollectiveGrievanceHandler:
    """Collective/class-action grievance management engine.

    Example:
        >>> handler = CollectiveGrievanceHandler()
        >>> cg = await handler.create_collective(
        ...     operator_id="OP-001",
        ...     title="Community Water Rights",
        ...     individual_ids=["g-001", "g-002", "g-003"],
        ... )
        >>> assert cg.collective_status == CollectiveStatus.FORMING
    """

    def __init__(
        self, config: Optional[GrievanceMechanismManagerConfig] = None,
    ) -> None:
        self.config = config or get_config()
        self._provenance = ProvenanceTracker()
        self._collectives: Dict[str, CollectiveGrievanceRecord] = {}
        logger.info("CollectiveGrievanceHandler engine initialized")

    async def create_collective(
        self,
        operator_id: str,
        title: str,
        individual_ids: Optional[List[str]] = None,
        description: str = "",
        category: str = "process",
        lead_complainant_id: Optional[str] = None,
        affected_count: int = 1,
    ) -> CollectiveGrievanceRecord:
        """Create a new collective grievance."""
        now = datetime.now(timezone.utc).replace(microsecond=0)
        collective_id = str(uuid.uuid4())

        ids = individual_ids or []
        count = max(affected_count, len(ids), 1)

        record = CollectiveGrievanceRecord(
            collective_id=collective_id,
            operator_id=operator_id,
            title=title,
            description=description,
            grievance_category=category,
            lead_complainant_id=lead_complainant_id,
            affected_stakeholder_count=count,
            individual_grievance_ids=ids,
            collective_status=CollectiveStatus.FORMING,
            negotiation_status=NegotiationStatus.NOT_STARTED,
            created_at=now,
            updated_at=now,
        )

        record.provenance_hash = self._provenance.compute_hash({
            "collective_id": collective_id,
            "operator_id": operator_id,
            "created_at": now.isoformat(),
        })

        self._collectives[collective_id] = record

        self._provenance.record(
            entity_type="collective_grievance",
            action="create",
            entity_id=collective_id,
            actor=AGENT_ID,
            metadata={"individual_count": len(ids), "affected": count},
        )

        logger.info("Collective %s created: %s (%d individuals)", collective_id, title, len(ids))
        return record

    async def add_individual_grievances(
        self,
        collective_id: str,
        grievance_ids: List[str],
    ) -> CollectiveGrievanceRecord:
        """Add individual grievances to a collective."""
        record = self._collectives.get(collective_id)
        if record is None:
            raise ValueError(f"Collective {collective_id} not found")

        for gid in grievance_ids:
            if gid not in record.individual_grievance_ids:
                record.individual_grievance_ids.append(gid)

        record.affected_stakeholder_count = max(
            record.affected_stakeholder_count,
            len(record.individual_grievance_ids),
        )
        record.updated_at = datetime.now(timezone.utc).replace(microsecond=0)

        logger.info("Collective %s: added %d grievances", collective_id, len(grievance_ids))
        return record

    async def set_spokesperson(
        self,
        collective_id: str,
        spokesperson: str,
        representative_body: Optional[str] = None,
    ) -> CollectiveGrievanceRecord:
        """Set spokesperson and representative body."""
        record = self._collectives.get(collective_id)
        if record is None:
            raise ValueError(f"Collective {collective_id} not found")

        record.spokesperson = spokesperson
        if representative_body:
            record.representative_body = representative_body
        record.updated_at = datetime.now(timezone.utc).replace(microsecond=0)

        logger.info("Collective %s spokesperson: %s", collective_id, spokesperson)
        return record

    async def add_demands(
        self,
        collective_id: str,
        demands: List[Dict[str, Any]],
    ) -> CollectiveGrievanceRecord:
        """Add demands to a collective grievance."""
        record = self._collectives.get(collective_id)
        if record is None:
            raise ValueError(f"Collective {collective_id} not found")

        for d in demands:
            demand = CollectiveDemand(
                demand=d.get("demand", ""),
                priority=d.get("priority", "medium"),
                negotiable=d.get("negotiable", True),
            )
            record.collective_demands.append(demand)

        record.updated_at = datetime.now(timezone.utc).replace(microsecond=0)
        logger.info("Collective %s: %d demands added", collective_id, len(demands))
        return record

    async def update_status(
        self,
        collective_id: str,
        status: str,
    ) -> CollectiveGrievanceRecord:
        """Update collective grievance status."""
        record = self._collectives.get(collective_id)
        if record is None:
            raise ValueError(f"Collective {collective_id} not found")

        try:
            new_status = CollectiveStatus(status)
        except ValueError:
            raise ValueError(f"Invalid status: {status}")

        now = datetime.now(timezone.utc).replace(microsecond=0)
        old_status = record.collective_status
        record.collective_status = new_status
        record.updated_at = now

        if new_status in (CollectiveStatus.RESOLVED, CollectiveStatus.CLOSED):
            record.resolved_at = now

        self._provenance.record(
            entity_type="collective_grievance",
            action="update",
            entity_id=collective_id,
            actor=AGENT_ID,
            metadata={"from": old_status.value, "to": new_status.value},
        )

        logger.info("Collective %s status: %s -> %s", collective_id, old_status.value, new_status.value)
        return record

    async def update_negotiation_status(
        self,
        collective_id: str,
        negotiation_status: str,
    ) -> CollectiveGrievanceRecord:
        """Update negotiation status."""
        record = self._collectives.get(collective_id)
        if record is None:
            raise ValueError(f"Collective {collective_id} not found")

        try:
            new_neg = NegotiationStatus(negotiation_status)
        except ValueError:
            raise ValueError(f"Invalid negotiation status: {negotiation_status}")

        record.negotiation_status = new_neg
        record.updated_at = datetime.now(timezone.utc).replace(microsecond=0)

        logger.info("Collective %s negotiation: %s", collective_id, new_neg.value)
        return record

    async def get_collective(self, collective_id: str) -> Optional[CollectiveGrievanceRecord]:
        """Retrieve a collective grievance by ID."""
        return self._collectives.get(collective_id)

    async def list_collectives(
        self,
        operator_id: Optional[str] = None,
        status: Optional[str] = None,
        category: Optional[str] = None,
    ) -> List[CollectiveGrievanceRecord]:
        """List collective grievances with optional filters."""
        results = list(self._collectives.values())
        if operator_id:
            results = [r for r in results if r.operator_id == operator_id]
        if status:
            results = [r for r in results if r.collective_status.value == status]
        if category:
            results = [r for r in results if r.grievance_category == category]
        return results

    async def health_check(self) -> Dict[str, Any]:
        active = sum(
            1 for c in self._collectives.values()
            if c.collective_status.value not in ("resolved", "closed")
        )
        return {
            "engine": "CollectiveGrievanceHandler",
            "status": "healthy",
            "collective_count": len(self._collectives),
            "active_collectives": active,
        }
