# -*- coding: utf-8 -*-
"""
Escalation Engine - SEC-010

Manages incident escalation based on severity levels and SLA timers.
Integrates with PagerDuty for on-call responder lookup and automatically
escalates incidents that breach SLA thresholds.

Example:
    >>> from greenlang.infrastructure.incident_response.escalator import (
    ...     EscalationEngine,
    ... )
    >>> engine = EscalationEngine(config)
    >>> await engine.escalate(incident)

Author: GreenLang Security Team
Date: February 2026
Status: Production Ready
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Set
from uuid import UUID, uuid4

import httpx

from greenlang.infrastructure.incident_response.config import (
    IncidentResponseConfig,
    get_config,
)
from greenlang.infrastructure.incident_response.models import (
    Incident,
    IncidentStatus,
    EscalationLevel,
    TimelineEvent,
    TimelineEventType,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class OnCallResponder:
    """On-call responder information.

    Attributes:
        id: Responder unique identifier.
        name: Responder full name.
        email: Responder email address.
        phone: Phone number for SMS/calls.
        slack_id: Slack user ID.
        schedule_name: On-call schedule name.
        escalation_level: Current escalation level in schedule.
    """

    id: str
    name: str
    email: str
    phone: Optional[str] = None
    slack_id: Optional[str] = None
    schedule_name: Optional[str] = None
    escalation_level: int = 1


@dataclass
class EscalationRecord:
    """Record of an escalation action.

    Attributes:
        id: Unique record identifier.
        incident_id: Associated incident ID.
        escalation_level: Level escalated to.
        reason: Reason for escalation.
        responder: Target responder.
        escalated_at: Escalation timestamp.
        acknowledged_at: When acknowledged.
        notification_channels: Channels used.
    """

    id: UUID = field(default_factory=uuid4)
    incident_id: UUID = field(default_factory=uuid4)
    escalation_level: int = 1
    reason: str = ""
    responder: Optional[OnCallResponder] = None
    escalated_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    acknowledged_at: Optional[datetime] = None
    notification_channels: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Escalation Engine
# ---------------------------------------------------------------------------


class EscalationEngine:
    """Manages incident escalation workflow.

    Handles automatic escalation based on SLA timers, integrates with
    PagerDuty for on-call lookup, and tracks acknowledgment status.

    Attributes:
        config: Incident response configuration.
        http_client: Async HTTP client for API calls.
        pending_escalations: Incidents awaiting acknowledgment.
        escalation_history: History of escalation records.

    Example:
        >>> engine = EscalationEngine(config)
        >>> await engine.escalate(incident)
        >>> responder = await engine.get_on_call_responder()
    """

    def __init__(
        self,
        config: Optional[IncidentResponseConfig] = None,
        http_client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        """Initialize the escalation engine.

        Args:
            config: Incident response configuration.
            http_client: Optional HTTP client (created if not provided).
        """
        self.config = config or get_config()
        self._http_client = http_client
        self._owns_client = http_client is None
        self.pending_escalations: Dict[UUID, EscalationRecord] = {}
        self.escalation_history: Dict[UUID, List[EscalationRecord]] = {}
        self._escalation_tasks: Dict[UUID, asyncio.Task] = {}

        logger.info("EscalationEngine initialized")

    async def _get_http_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client.

        Returns:
            Async HTTP client instance.
        """
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(timeout=30.0)
        return self._http_client

    async def close(self) -> None:
        """Close HTTP client and cancel pending tasks."""
        # Cancel all pending escalation tasks
        for task in self._escalation_tasks.values():
            task.cancel()
        self._escalation_tasks.clear()

        if self._owns_client and self._http_client is not None:
            await self._http_client.aclose()
            self._http_client = None

    async def escalate(
        self,
        incident: Incident,
        reason: Optional[str] = None,
    ) -> EscalationRecord:
        """Escalate an incident.

        Looks up the on-call responder and creates an escalation record.
        Triggers notifications through configured channels.

        Args:
            incident: Incident to escalate.
            reason: Optional reason for escalation.

        Returns:
            Created escalation record.
        """
        logger.info(
            "Escalating incident %s (severity=%s)",
            incident.incident_number,
            incident.severity.value,
        )

        # Get on-call responder
        responder = await self.get_on_call_responder(incident.severity)

        # Determine escalation reason
        if not reason:
            reason = f"Severity {incident.severity.value} incident requires immediate attention"

        # Determine notification channels
        channels = self._get_notification_channels(incident.severity)

        # Create escalation record
        record = EscalationRecord(
            id=uuid4(),
            incident_id=incident.id,
            escalation_level=self._get_escalation_level_number(incident.severity),
            reason=reason,
            responder=responder,
            escalated_at=datetime.now(timezone.utc),
            notification_channels=channels,
        )

        # Store record
        self.pending_escalations[incident.id] = record
        if incident.id not in self.escalation_history:
            self.escalation_history[incident.id] = []
        self.escalation_history[incident.id].append(record)

        # Assign responder to incident if available
        if responder:
            incident.assignee_id = UUID(responder.id) if _is_valid_uuid(responder.id) else None
            incident.assignee_name = responder.name

        logger.info(
            "Escalation record created for incident %s -> %s",
            incident.incident_number,
            responder.name if responder else "unassigned",
        )

        return record

    async def get_on_call_responder(
        self,
        severity: EscalationLevel,
    ) -> Optional[OnCallResponder]:
        """Get the current on-call responder from PagerDuty.

        Args:
            severity: Incident severity for schedule selection.

        Returns:
            On-call responder or None if unavailable.
        """
        if not self.config.pagerduty.enabled:
            logger.debug("PagerDuty disabled, returning None")
            return None

        api_key = self.config.pagerduty.get_api_key()
        if not api_key:
            logger.warning("PagerDuty API key not configured")
            return None

        try:
            client = await self._get_http_client()

            # Get on-call users
            headers = {
                "Authorization": f"Token token={api_key}",
                "Content-Type": "application/json",
            }

            # Get oncalls endpoint
            response = await client.get(
                f"{self.config.pagerduty.api_base_url}/oncalls",
                headers=headers,
                params={
                    "time_zone": "UTC",
                    "include[]": ["users"],
                },
            )
            response.raise_for_status()

            data = response.json()
            oncalls = data.get("oncalls", [])

            if not oncalls:
                logger.warning("No on-call responders found in PagerDuty")
                return None

            # Get first on-call user
            oncall = oncalls[0]
            user = oncall.get("user", {})

            return OnCallResponder(
                id=user.get("id", str(uuid4())),
                name=user.get("name", "Unknown"),
                email=user.get("email", ""),
                schedule_name=oncall.get("schedule", {}).get("summary"),
                escalation_level=oncall.get("escalation_level", 1),
            )

        except httpx.HTTPError as e:
            logger.error("Failed to get on-call from PagerDuty: %s", e)
            return None
        except Exception as e:
            logger.error("Unexpected error getting on-call: %s", e)
            return None

    def track_acknowledgment(
        self,
        incident_id: UUID,
        acknowledger_id: Optional[UUID] = None,
        acknowledger_name: Optional[str] = None,
    ) -> Optional[EscalationRecord]:
        """Track incident acknowledgment.

        Updates the escalation record with acknowledgment timestamp
        and removes from pending escalations.

        Args:
            incident_id: Incident ID.
            acknowledger_id: ID of user acknowledging.
            acknowledger_name: Name of acknowledger.

        Returns:
            Updated escalation record or None.
        """
        if incident_id not in self.pending_escalations:
            logger.debug("No pending escalation for incident %s", incident_id)
            return None

        record = self.pending_escalations.pop(incident_id)
        record.acknowledged_at = datetime.now(timezone.utc)

        # Cancel any auto-escalation task
        if incident_id in self._escalation_tasks:
            self._escalation_tasks[incident_id].cancel()
            del self._escalation_tasks[incident_id]

        # Calculate response time
        response_time = (record.acknowledged_at - record.escalated_at).total_seconds()

        logger.info(
            "Incident %s acknowledged in %.1f seconds by %s",
            incident_id,
            response_time,
            acknowledger_name or "unknown",
        )

        return record

    async def auto_escalate(
        self,
        incident: Incident,
        delay_seconds: Optional[int] = None,
    ) -> None:
        """Schedule automatic escalation on SLA breach.

        Sets up a timer to automatically escalate the incident if it
        isn't acknowledged within the SLA threshold.

        Args:
            incident: Incident to monitor.
            delay_seconds: Delay before escalation (uses config if None).
        """
        # Cancel any existing escalation task
        if incident.id in self._escalation_tasks:
            self._escalation_tasks[incident.id].cancel()

        # Determine delay
        if delay_seconds is None:
            delay_seconds = self.config.get_escalation_threshold_minutes(
                incident.severity.value
            ) * 60

        logger.info(
            "Scheduling auto-escalation for %s in %d seconds",
            incident.incident_number,
            delay_seconds,
        )

        # Create escalation task
        task = asyncio.create_task(
            self._auto_escalation_worker(incident, delay_seconds)
        )
        self._escalation_tasks[incident.id] = task

    async def _auto_escalation_worker(
        self,
        incident: Incident,
        delay_seconds: int,
    ) -> None:
        """Background worker for auto-escalation.

        Args:
            incident: Incident to potentially escalate.
            delay_seconds: Delay before escalation.
        """
        try:
            await asyncio.sleep(delay_seconds)

            # Check if still pending
            if incident.id in self.pending_escalations:
                # Check if acknowledged
                record = self.pending_escalations[incident.id]
                if record.acknowledged_at is None:
                    # Perform escalation
                    new_record = await self.escalate(
                        incident,
                        reason=f"SLA breach: No acknowledgment within {delay_seconds // 60} minutes",
                    )
                    new_record.escalation_level += 1

                    logger.warning(
                        "Auto-escalated incident %s due to SLA breach",
                        incident.incident_number,
                    )

        except asyncio.CancelledError:
            logger.debug("Auto-escalation cancelled for %s", incident.incident_number)
        except Exception as e:
            logger.error("Auto-escalation failed for %s: %s", incident.incident_number, e)

    def get_pending_escalations(self) -> List[EscalationRecord]:
        """Get all pending escalation records.

        Returns:
            List of pending escalation records.
        """
        return list(self.pending_escalations.values())

    def get_escalation_history(
        self,
        incident_id: UUID,
    ) -> List[EscalationRecord]:
        """Get escalation history for an incident.

        Args:
            incident_id: Incident ID.

        Returns:
            List of escalation records.
        """
        return self.escalation_history.get(incident_id, [])

    def check_sla_breach(self, incident: Incident) -> bool:
        """Check if incident has breached SLA.

        Args:
            incident: Incident to check.

        Returns:
            True if SLA is breached.
        """
        if incident.status in (IncidentStatus.RESOLVED, IncidentStatus.CLOSED):
            return False

        if incident.acknowledged_at:
            return False

        sla_minutes = self.config.get_response_sla_minutes(incident.severity.value)
        threshold = incident.detected_at + timedelta(minutes=sla_minutes)

        return datetime.now(timezone.utc) > threshold

    def get_time_until_breach(self, incident: Incident) -> Optional[timedelta]:
        """Get time remaining until SLA breach.

        Args:
            incident: Incident to check.

        Returns:
            Time until breach, or None if already breached/N/A.
        """
        if incident.status in (IncidentStatus.RESOLVED, IncidentStatus.CLOSED):
            return None

        if incident.acknowledged_at:
            return None

        sla_minutes = self.config.get_response_sla_minutes(incident.severity.value)
        threshold = incident.detected_at + timedelta(minutes=sla_minutes)
        remaining = threshold - datetime.now(timezone.utc)

        if remaining.total_seconds() < 0:
            return None  # Already breached

        return remaining

    def _get_notification_channels(
        self,
        severity: EscalationLevel,
    ) -> List[str]:
        """Get notification channels for severity level.

        Args:
            severity: Incident severity.

        Returns:
            List of notification channel names.
        """
        channels = []

        if severity == EscalationLevel.P0:
            if self.config.pagerduty.enabled:
                channels.append("pagerduty")
            if self.config.slack.enabled:
                channels.append("slack")
            if self.config.sms.enabled:
                channels.append("sms")
            if self.config.email.enabled:
                channels.append("email")
        elif severity == EscalationLevel.P1:
            if self.config.pagerduty.enabled:
                channels.append("pagerduty")
            if self.config.slack.enabled:
                channels.append("slack")
            if self.config.email.enabled:
                channels.append("email")
        elif severity == EscalationLevel.P2:
            if self.config.slack.enabled:
                channels.append("slack")
            if self.config.email.enabled:
                channels.append("email")
        else:
            if self.config.email.enabled:
                channels.append("email")

        return channels

    def _get_escalation_level_number(
        self,
        severity: EscalationLevel,
    ) -> int:
        """Convert severity to numeric escalation level.

        Args:
            severity: Escalation level.

        Returns:
            Numeric level (1-4).
        """
        mapping = {
            EscalationLevel.P0: 1,
            EscalationLevel.P1: 2,
            EscalationLevel.P2: 3,
            EscalationLevel.P3: 4,
        }
        return mapping.get(severity, 3)

    def create_timeline_event(
        self,
        incident: Incident,
        record: EscalationRecord,
    ) -> TimelineEvent:
        """Create a timeline event for an escalation.

        Args:
            incident: Associated incident.
            record: Escalation record.

        Returns:
            TimelineEvent for the escalation.
        """
        responder_name = record.responder.name if record.responder else "unassigned"

        return TimelineEvent(
            id=uuid4(),
            incident_id=incident.id,
            event_type=TimelineEventType.ESCALATED,
            timestamp=record.escalated_at,
            description=f"Escalated to {responder_name}: {record.reason}",
            new_value=f"Level {record.escalation_level}",
            metadata={
                "responder_id": record.responder.id if record.responder else None,
                "responder_email": record.responder.email if record.responder else None,
                "notification_channels": record.notification_channels,
            },
        )


def _is_valid_uuid(value: str) -> bool:
    """Check if string is a valid UUID.

    Args:
        value: String to check.

    Returns:
        True if valid UUID.
    """
    try:
        UUID(value)
        return True
    except (ValueError, TypeError):
        return False


# ---------------------------------------------------------------------------
# Global Engine Instance
# ---------------------------------------------------------------------------

_global_engine: Optional[EscalationEngine] = None


def get_escalation_engine(
    config: Optional[IncidentResponseConfig] = None,
) -> EscalationEngine:
    """Get or create the global escalation engine.

    Args:
        config: Optional configuration override.

    Returns:
        The global EscalationEngine instance.
    """
    global _global_engine

    if _global_engine is None:
        _global_engine = EscalationEngine(config)

    return _global_engine


async def reset_escalation_engine() -> None:
    """Reset and close the global escalation engine."""
    global _global_engine

    if _global_engine is not None:
        await _global_engine.close()
        _global_engine = None


__all__ = [
    "EscalationEngine",
    "OnCallResponder",
    "EscalationRecord",
    "get_escalation_engine",
    "reset_escalation_engine",
]
