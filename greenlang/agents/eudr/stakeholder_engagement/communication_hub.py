# -*- coding: utf-8 -*-
"""
CommunicationHub Engine - AGENT-EUDR-031

Multi-channel stakeholder communication platform supporting email,
SMS, letter, radio, in-person, phone, and digital platform channels.
Provides communication sending, scheduling, delivery tracking,
response management, and campaign coordination.

Zero-Hallucination: All delivery tracking and campaign logic is
deterministic. No LLM involvement in communication operations.

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-031 (GL-EUDR-SET-031)
Regulation: EU 2023/1115 (EUDR)
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
    CommunicationChannel,
    CommunicationRecord,
    DeliveryStatus,
)
from greenlang.agents.eudr.stakeholder_engagement.provenance import (
    ProvenanceTracker,
)

logger = logging.getLogger(__name__)


class CommunicationHub:
    """Multi-channel communication hub for stakeholder outreach.

    Manages sending, scheduling, tracking, and response management
    for stakeholder communications across all supported channels.

    Attributes:
        _config: Engine configuration.
        _provenance: Provenance hash chain tracker.
        _communications: In-memory communication store.
        _responses: In-memory response store.
    """

    def __init__(self, config: StakeholderEngagementConfig) -> None:
        """Initialize CommunicationHub.

        Args:
            config: Stakeholder engagement configuration.
        """
        self._config = config
        self._provenance = ProvenanceTracker()
        self._communications: Dict[str, CommunicationRecord] = {}
        self._responses: Dict[str, List[Dict[str, Any]]] = {}
        logger.info("CommunicationHub initialized")

    async def send_communication(
        self,
        operator_id: str,
        stakeholder_ids: List[str],
        channel: CommunicationChannel,
        subject: str,
        body: str,
        template_id: Optional[str] = None,
        language: Optional[str] = None,
    ) -> CommunicationRecord:
        """Send a communication to stakeholders.

        Args:
            operator_id: Operator sending the communication.
            stakeholder_ids: Target stakeholder identifiers.
            channel: Communication channel.
            subject: Communication subject.
            body: Communication content.
            template_id: Optional template reference.
            language: Communication language.

        Returns:
            Created CommunicationRecord.

        Raises:
            ValueError: If required fields are empty.
        """
        if not subject or not subject.strip():
            raise ValueError("subject is required")
        if not body or not body.strip():
            raise ValueError("body is required")
        if not stakeholder_ids:
            raise ValueError("stakeholder_ids are required")

        now = datetime.now(tz=timezone.utc)
        communication_id = f"COMM-{uuid.uuid4().hex[:8].upper()}"

        record = CommunicationRecord(
            communication_id=communication_id,
            operator_id=operator_id,
            stakeholder_ids=stakeholder_ids,
            channel=channel,
            subject=subject,
            body=body,
            language=language or self._config.communication_default_language,
            template_id=template_id,
            delivery_status=DeliveryStatus.SENT,
            sent_at=now,
            created_at=now,
        )

        self._communications[communication_id] = record
        self._provenance.record(
            "communication", "send", communication_id, "AGENT-EUDR-031",
            metadata={"channel": channel.value},
        )
        logger.info("Communication %s sent via %s", communication_id, channel.value)
        return record

    async def schedule_communication(
        self,
        operator_id: str,
        stakeholder_ids: List[str],
        channel: CommunicationChannel,
        subject: str,
        body: str,
        send_at: Optional[datetime] = None,
        template_id: Optional[str] = None,
        language: Optional[str] = None,
    ) -> CommunicationRecord:
        """Schedule a communication for future delivery.

        Args:
            operator_id: Operator scheduling the communication.
            stakeholder_ids: Target stakeholder identifiers.
            channel: Communication channel.
            subject: Communication subject.
            body: Communication content.
            send_at: Scheduled delivery time (must be in the future).
            template_id: Optional template reference.
            language: Communication language.

        Returns:
            Scheduled CommunicationRecord.

        Raises:
            ValueError: If send_at is missing or in the past.
        """
        if send_at is None:
            raise ValueError("send_at is required for scheduling")

        now = datetime.now(tz=timezone.utc)
        if send_at <= now:
            raise ValueError("send_at must be in the future")

        communication_id = f"COMM-{uuid.uuid4().hex[:8].upper()}"

        record = CommunicationRecord(
            communication_id=communication_id,
            operator_id=operator_id,
            stakeholder_ids=stakeholder_ids,
            channel=channel,
            subject=subject,
            body=body,
            language=language or self._config.communication_default_language,
            template_id=template_id,
            delivery_status=DeliveryStatus.SCHEDULED,
            scheduled_at=send_at,
            created_at=now,
        )

        self._communications[communication_id] = record
        self._provenance.record(
            "communication", "schedule", communication_id, "AGENT-EUDR-031",
        )
        logger.info("Communication %s scheduled for %s", communication_id, send_at.isoformat())
        return record

    async def send_campaign(
        self,
        campaign_id: str,
        name: str,
        operator_id: str,
        stakeholder_ids: List[str],
        channels: List[CommunicationChannel],
        subject: str,
        body: str,
        language: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Send a communication campaign across channels.

        Args:
            campaign_id: Campaign identifier.
            name: Campaign name.
            operator_id: Operator running the campaign.
            stakeholder_ids: Target stakeholders.
            channels: Channels to use.
            subject: Communication subject.
            body: Communication body.
            language: Optional language override.

        Returns:
            Campaign summary dictionary.

        Raises:
            ValueError: If required fields are empty.
        """
        if not name or not name.strip():
            raise ValueError("name is required")
        if not stakeholder_ids:
            raise ValueError("stakeholder_ids are required")
        if not channels:
            raise ValueError("channels are required")

        communications: List[CommunicationRecord] = []
        now = datetime.now(tz=timezone.utc)

        for stk_id in stakeholder_ids:
            for channel in channels:
                comm_id = f"COMM-{uuid.uuid4().hex[:8].upper()}"
                record = CommunicationRecord(
                    communication_id=comm_id,
                    operator_id=operator_id,
                    stakeholder_ids=[stk_id],
                    channel=channel,
                    subject=subject,
                    body=body,
                    language=language or self._config.communication_default_language,
                    delivery_status=DeliveryStatus.SENT,
                    sent_at=now,
                    campaign_id=campaign_id,
                    created_at=now,
                )
                self._communications[comm_id] = record
                communications.append(record)

        total_sent = len(communications)
        self._provenance.record(
            "communication", "campaign", campaign_id, "AGENT-EUDR-031",
            metadata={"total": total_sent},
        )
        logger.info("Campaign %s: %d communications sent", campaign_id, total_sent)

        return {
            "campaign_id": campaign_id,
            "name": name,
            "communications_sent": total_sent,
            "total": total_sent,
            "channels_used": len(channels),
            "communications": [c.communication_id for c in communications],
        }

    async def track_delivery(
        self,
        communication_id: str,
    ) -> Dict[str, Any]:
        """Track delivery status of a communication.

        Args:
            communication_id: Communication to track.

        Returns:
            Delivery status dictionary.

        Raises:
            ValueError: If communication not found or ID empty.
        """
        if not communication_id or not communication_id.strip():
            raise ValueError("communication_id is required")

        record = self._get_communication(communication_id)

        result: Dict[str, Any] = {
            "communication_id": communication_id,
            "delivery_status": record.delivery_status.value,
            "channel": record.channel.value,
        }

        if record.sent_at:
            result["sent_at"] = record.sent_at.isoformat()
        if record.delivered_at:
            result["delivered_at"] = record.delivered_at.isoformat()
        if record.scheduled_at:
            result["scheduled_at"] = record.scheduled_at.isoformat()

        result["updated_at"] = datetime.now(tz=timezone.utc).isoformat()
        return result

    async def update_delivery_status(
        self,
        communication_id: str,
        status: DeliveryStatus,
    ) -> CommunicationRecord:
        """Update delivery status of a communication.

        Args:
            communication_id: Communication to update.
            status: New delivery status.

        Returns:
            Updated CommunicationRecord.

        Raises:
            ValueError: If communication not found.
        """
        record = self._get_communication(communication_id)
        record.delivery_status = status

        if status == DeliveryStatus.DELIVERED:
            record.delivered_at = datetime.now(tz=timezone.utc)

        self._provenance.record(
            "communication", "update_status", communication_id, "AGENT-EUDR-031",
            metadata={"status": status.value},
        )
        return record

    async def record_response(
        self,
        communication_id: str,
        stakeholder_id: str,
        response_type: str,
        response_text: str,
    ) -> Dict[str, Any]:
        """Record a stakeholder response to a communication.

        Args:
            communication_id: Communication responded to.
            stakeholder_id: Responding stakeholder.
            response_type: Type of response (confirmed, declined, etc.).
            response_text: Response content.

        Returns:
            Response record dictionary.

        Raises:
            ValueError: If communication not found or fields empty.
        """
        if not stakeholder_id or not stakeholder_id.strip():
            raise ValueError("stakeholder_id is required")

        self._get_communication(communication_id)  # Validate exists

        now = datetime.now(tz=timezone.utc)
        response = {
            "communication_id": communication_id,
            "stakeholder_id": stakeholder_id,
            "response_type": response_type,
            "response_text": response_text,
            "responded_at": now.isoformat(),
        }

        if communication_id not in self._responses:
            self._responses[communication_id] = []
        self._responses[communication_id].append(response)

        self._provenance.record(
            "communication", "record_response", communication_id, "AGENT-EUDR-031",
            metadata={"stakeholder_id": stakeholder_id, "type": response_type},
        )
        return response

    def _get_communication(self, communication_id: str) -> CommunicationRecord:
        """Get communication by ID or raise ValueError."""
        if communication_id not in self._communications:
            raise ValueError(f"communication not found: {communication_id}")
        return self._communications[communication_id]
