# -*- coding: utf-8 -*-
"""
Pilot partner registry (F090).

Manages design partner enrollment, configuration, and lifecycle.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class PilotStatus(str, Enum):
    INVITED = "invited"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    CHURNED = "churned"


class PilotTier(str, Enum):
    COMMUNITY = "community"
    PRO = "pro"
    ENTERPRISE = "enterprise"


@dataclass
class PilotPartner:
    """A design partner enrolled in the Factors pilot."""

    partner_id: str
    name: str
    contact_email: str
    organization: str
    tier: PilotTier = PilotTier.PRO
    status: PilotStatus = PilotStatus.INVITED
    tenant_id: str = ""
    api_key: str = ""
    enrolled_at: str = ""
    activated_at: str = ""
    target_use_cases: List[str] = field(default_factory=list)
    max_api_calls_per_day: int = 10000
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "partner_id": self.partner_id,
            "name": self.name,
            "contact_email": self.contact_email,
            "organization": self.organization,
            "tier": self.tier.value,
            "status": self.status.value,
            "tenant_id": self.tenant_id,
            "enrolled_at": self.enrolled_at,
            "activated_at": self.activated_at,
            "target_use_cases": self.target_use_cases,
            "max_api_calls_per_day": self.max_api_calls_per_day,
        }


class PilotRegistry:
    """
    Registry of design partners for the Factors pilot program.

    Manages partner enrollment, activation, and status tracking.
    """

    def __init__(self) -> None:
        self._partners: Dict[str, PilotPartner] = {}

    def enroll(
        self,
        name: str,
        contact_email: str,
        organization: str,
        tier: PilotTier = PilotTier.PRO,
        use_cases: Optional[List[str]] = None,
    ) -> PilotPartner:
        """Enroll a new design partner."""
        partner_id = f"pilot_{uuid.uuid4().hex[:12]}"
        tenant_id = f"tenant_{uuid.uuid4().hex[:8]}"
        partner = PilotPartner(
            partner_id=partner_id,
            name=name,
            contact_email=contact_email,
            organization=organization,
            tier=tier,
            tenant_id=tenant_id,
            enrolled_at=datetime.now(timezone.utc).isoformat(),
            target_use_cases=use_cases or [],
        )
        self._partners[partner_id] = partner
        logger.info("Enrolled pilot partner: %s (%s)", name, partner_id)
        return partner

    def activate(self, partner_id: str, api_key: str) -> Optional[PilotPartner]:
        """Activate a partner with an API key."""
        partner = self._partners.get(partner_id)
        if not partner:
            return None
        partner.api_key = api_key
        partner.status = PilotStatus.ACTIVE
        partner.activated_at = datetime.now(timezone.utc).isoformat()
        logger.info("Activated pilot partner: %s", partner_id)
        return partner

    def pause(self, partner_id: str) -> Optional[PilotPartner]:
        partner = self._partners.get(partner_id)
        if partner:
            partner.status = PilotStatus.PAUSED
        return partner

    def complete(self, partner_id: str) -> Optional[PilotPartner]:
        partner = self._partners.get(partner_id)
        if partner:
            partner.status = PilotStatus.COMPLETED
        return partner

    def get(self, partner_id: str) -> Optional[PilotPartner]:
        return self._partners.get(partner_id)

    def get_by_tenant(self, tenant_id: str) -> Optional[PilotPartner]:
        for p in self._partners.values():
            if p.tenant_id == tenant_id:
                return p
        return None

    def list_active(self) -> List[PilotPartner]:
        return [p for p in self._partners.values() if p.status == PilotStatus.ACTIVE]

    def list_all(self) -> List[PilotPartner]:
        return list(self._partners.values())

    def summary(self) -> Dict[str, Any]:
        by_status: Dict[str, int] = {}
        for p in self._partners.values():
            by_status[p.status.value] = by_status.get(p.status.value, 0) + 1
        return {
            "total": len(self._partners),
            "by_status": by_status,
        }
