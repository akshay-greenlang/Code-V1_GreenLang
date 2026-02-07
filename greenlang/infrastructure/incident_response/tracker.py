# -*- coding: utf-8 -*-
"""
Incident Tracker - SEC-010

Manages incident lifecycle including creation, status updates, timeline
tracking, assignment, and post-mortem generation. Integrates with Jira
for ticket creation.

Example:
    >>> from greenlang.infrastructure.incident_response.tracker import (
    ...     IncidentTracker,
    ... )
    >>> tracker = IncidentTracker(config)
    >>> incident = await tracker.create_incident(title="Data breach", ...)
    >>> await tracker.update_status(incident.id, IncidentStatus.INVESTIGATING)

Author: GreenLang Security Team
Date: February 2026
Status: Production Ready
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

import httpx

from greenlang.infrastructure.incident_response.config import (
    IncidentResponseConfig,
    get_config,
)
from greenlang.infrastructure.incident_response.models import (
    Alert,
    AlertSource,
    Incident,
    IncidentStatus,
    IncidentType,
    EscalationLevel,
    TimelineEvent,
    TimelineEventType,
    PostMortem,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Incident Tracker
# ---------------------------------------------------------------------------


class IncidentTracker:
    """Manages incident lifecycle and tracking.

    Handles incident creation, status transitions, timeline events,
    responder assignment, and post-mortem generation.

    Attributes:
        config: Incident response configuration.
        incidents: In-memory incident store (production uses database).
        timeline_events: In-memory timeline store.
        post_mortems: In-memory post-mortem store.
        incident_counter: Counter for incident number generation.

    Example:
        >>> tracker = IncidentTracker(config)
        >>> incident = await tracker.create_incident(...)
        >>> await tracker.update_status(incident.id, IncidentStatus.INVESTIGATING)
    """

    def __init__(
        self,
        config: Optional[IncidentResponseConfig] = None,
        http_client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        """Initialize the incident tracker.

        Args:
            config: Incident response configuration.
            http_client: Optional HTTP client for external integrations.
        """
        self.config = config or get_config()
        self._http_client = http_client
        self._owns_client = http_client is None

        # In-memory stores (production would use database)
        self.incidents: Dict[UUID, Incident] = {}
        self.timeline_events: Dict[UUID, List[TimelineEvent]] = {}
        self.post_mortems: Dict[UUID, PostMortem] = {}

        self._incident_counter = 0

        logger.info("IncidentTracker initialized")

    async def _get_http_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client.

        Returns:
            Async HTTP client instance.
        """
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(timeout=30.0)
        return self._http_client

    async def close(self) -> None:
        """Close HTTP client if owned by this instance."""
        if self._owns_client and self._http_client is not None:
            await self._http_client.aclose()
            self._http_client = None

    def _generate_incident_number(self) -> str:
        """Generate a unique incident number.

        Returns:
            Incident number string (e.g., INC-2026-0001).
        """
        self._incident_counter += 1
        year = datetime.now(timezone.utc).year
        return f"{self.config.incident_number_prefix}-{year}-{self._incident_counter:04d}"

    async def create_incident(
        self,
        title: str,
        severity: EscalationLevel,
        source: AlertSource,
        incident_type: IncidentType = IncidentType.UNKNOWN,
        description: Optional[str] = None,
        affected_systems: Optional[List[str]] = None,
        affected_users: Optional[int] = None,
        related_alerts: Optional[List[UUID]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        created_by: Optional[UUID] = None,
    ) -> Incident:
        """Create a new incident.

        Args:
            title: Incident title.
            severity: Severity level.
            source: Alert source.
            incident_type: Type of incident.
            description: Detailed description.
            affected_systems: List of affected systems.
            affected_users: Number of affected users.
            related_alerts: Associated alert IDs.
            metadata: Additional metadata.
            created_by: Creator user ID.

        Returns:
            Created Incident object.
        """
        incident = Incident(
            id=uuid4(),
            incident_number=self._generate_incident_number(),
            title=title,
            description=description,
            severity=severity,
            status=IncidentStatus.DETECTED,
            incident_type=incident_type,
            source=source,
            detected_at=datetime.now(timezone.utc),
            affected_systems=affected_systems or [],
            affected_users=affected_users,
            related_alerts=related_alerts or [],
            metadata=metadata or {},
        )

        # Calculate provenance hash
        incident.provenance_hash = incident.calculate_provenance_hash()

        # Store incident
        self.incidents[incident.id] = incident
        self.timeline_events[incident.id] = []

        # Add creation event
        await self.add_timeline_event(
            incident_id=incident.id,
            event_type=TimelineEventType.CREATED,
            description=f"Incident created: {title}",
            actor_id=created_by,
            metadata={"source": source.value, "severity": severity.value},
        )

        logger.info(
            "Created incident %s: %s (severity=%s)",
            incident.incident_number,
            title,
            severity.value,
        )

        return incident

    async def create_incident_from_alert(
        self,
        alert: Alert,
        title: Optional[str] = None,
        created_by: Optional[UUID] = None,
    ) -> Incident:
        """Create an incident from an alert.

        Args:
            alert: Alert to create incident from.
            title: Override title (uses alert message if not provided).
            created_by: Creator user ID.

        Returns:
            Created Incident object.
        """
        return await self.create_incident(
            title=title or alert.message,
            severity=alert.severity,
            source=alert.source,
            description=alert.description,
            related_alerts=[alert.id],
            metadata={"original_alert": str(alert.id)},
            created_by=created_by,
        )

    async def get_incident(self, incident_id: UUID) -> Optional[Incident]:
        """Get an incident by ID.

        Args:
            incident_id: Incident UUID.

        Returns:
            Incident or None if not found.
        """
        return self.incidents.get(incident_id)

    async def get_incident_by_number(
        self,
        incident_number: str,
    ) -> Optional[Incident]:
        """Get an incident by number.

        Args:
            incident_number: Incident number string.

        Returns:
            Incident or None if not found.
        """
        for incident in self.incidents.values():
            if incident.incident_number == incident_number:
                return incident
        return None

    async def update_status(
        self,
        incident_id: UUID,
        new_status: IncidentStatus,
        actor_id: Optional[UUID] = None,
        actor_name: Optional[str] = None,
        comment: Optional[str] = None,
    ) -> Optional[Incident]:
        """Update incident status.

        Args:
            incident_id: Incident UUID.
            new_status: New status to set.
            actor_id: User making the change.
            actor_name: Actor display name.
            comment: Optional status change comment.

        Returns:
            Updated Incident or None if not found.

        Raises:
            ValueError: If status transition is invalid.
        """
        incident = self.incidents.get(incident_id)
        if not incident:
            logger.warning("Incident %s not found", incident_id)
            return None

        # Validate transition
        if not incident.can_transition_to(new_status):
            raise ValueError(
                f"Invalid status transition: {incident.status.value} -> {new_status.value}"
            )

        old_status = incident.status
        incident.status = new_status

        # Update timestamps
        now = datetime.now(timezone.utc)
        if new_status == IncidentStatus.ACKNOWLEDGED:
            incident.acknowledged_at = now
        elif new_status == IncidentStatus.RESOLVED:
            incident.resolved_at = now
        elif new_status == IncidentStatus.CLOSED:
            incident.closed_at = now

        # Update provenance hash
        incident.provenance_hash = incident.calculate_provenance_hash()

        # Add timeline event
        description = f"Status changed: {old_status.value} -> {new_status.value}"
        if comment:
            description += f" - {comment}"

        await self.add_timeline_event(
            incident_id=incident_id,
            event_type=TimelineEventType.STATUS_CHANGE,
            description=description,
            actor_id=actor_id,
            actor_name=actor_name,
            old_value=old_status.value,
            new_value=new_status.value,
        )

        logger.info(
            "Updated incident %s status: %s -> %s",
            incident.incident_number,
            old_status.value,
            new_status.value,
        )

        return incident

    async def add_timeline_event(
        self,
        incident_id: UUID,
        event_type: TimelineEventType,
        description: str,
        actor_id: Optional[UUID] = None,
        actor_name: Optional[str] = None,
        old_value: Optional[str] = None,
        new_value: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> TimelineEvent:
        """Add an event to the incident timeline.

        Args:
            incident_id: Incident UUID.
            event_type: Type of event.
            description: Event description.
            actor_id: Actor UUID.
            actor_name: Actor display name.
            old_value: Previous value (for changes).
            new_value: New value (for changes).
            metadata: Additional event data.

        Returns:
            Created TimelineEvent.
        """
        event = TimelineEvent(
            id=uuid4(),
            incident_id=incident_id,
            event_type=event_type,
            timestamp=datetime.now(timezone.utc),
            actor_id=actor_id,
            actor_name=actor_name,
            description=description,
            old_value=old_value,
            new_value=new_value,
            metadata=metadata or {},
        )

        if incident_id not in self.timeline_events:
            self.timeline_events[incident_id] = []

        self.timeline_events[incident_id].append(event)

        logger.debug(
            "Added timeline event for %s: %s",
            incident_id,
            event_type.value,
        )

        return event

    async def get_timeline(self, incident_id: UUID) -> List[TimelineEvent]:
        """Get incident timeline.

        Args:
            incident_id: Incident UUID.

        Returns:
            List of timeline events in chronological order.
        """
        events = self.timeline_events.get(incident_id, [])
        return sorted(events, key=lambda e: e.timestamp)

    async def assign_responder(
        self,
        incident_id: UUID,
        assignee_id: UUID,
        assignee_name: str,
        assigned_by: Optional[UUID] = None,
        assigned_by_name: Optional[str] = None,
    ) -> Optional[Incident]:
        """Assign a responder to an incident.

        Args:
            incident_id: Incident UUID.
            assignee_id: Assignee UUID.
            assignee_name: Assignee display name.
            assigned_by: Assigner UUID.
            assigned_by_name: Assigner display name.

        Returns:
            Updated Incident or None if not found.
        """
        incident = self.incidents.get(incident_id)
        if not incident:
            return None

        old_assignee = incident.assignee_name
        incident.assignee_id = assignee_id
        incident.assignee_name = assignee_name

        await self.add_timeline_event(
            incident_id=incident_id,
            event_type=TimelineEventType.ASSIGNED,
            description=f"Assigned to {assignee_name}",
            actor_id=assigned_by,
            actor_name=assigned_by_name,
            old_value=old_assignee,
            new_value=assignee_name,
        )

        logger.info(
            "Assigned incident %s to %s",
            incident.incident_number,
            assignee_name,
        )

        return incident

    async def generate_post_mortem(
        self,
        incident_id: UUID,
        created_by: Optional[UUID] = None,
    ) -> Optional[PostMortem]:
        """Generate a post-mortem template for an incident.

        Args:
            incident_id: Incident UUID.
            created_by: Creator UUID.

        Returns:
            Generated PostMortem or None if incident not found.
        """
        incident = self.incidents.get(incident_id)
        if not incident:
            return None

        timeline = await self.get_timeline(incident_id)

        # Build timeline for post-mortem
        pm_timeline = [
            {
                "time": event.timestamp.isoformat(),
                "event": event.description,
                "type": event.event_type.value,
            }
            for event in timeline
        ]

        # Generate action items template
        action_items = [
            {
                "title": "Review detection mechanisms",
                "owner": "",
                "status": "pending",
                "due_date": "",
            },
            {
                "title": "Update runbooks",
                "owner": "",
                "status": "pending",
                "due_date": "",
            },
            {
                "title": "Implement preventive controls",
                "owner": "",
                "status": "pending",
                "due_date": "",
            },
        ]

        post_mortem = PostMortem(
            id=uuid4(),
            incident_id=incident_id,
            incident_number=incident.incident_number,
            title=f"Post-Mortem: {incident.title}",
            summary=f"Post-mortem analysis for {incident.incident_type.value} incident",
            timeline=pm_timeline,
            root_cause="[To be determined during analysis]",
            impact=f"Affected systems: {', '.join(incident.affected_systems)}. "
            f"Affected users: {incident.affected_users or 'Unknown'}",
            detection=f"Detected via {incident.source.value} at {incident.detected_at.isoformat()}",
            response=(
                f"MTTD: {self._format_duration(incident.get_mttd_seconds())}, "
                f"MTTR: {self._format_duration(incident.get_mttr_seconds())}"
            ),
            lessons_learned=[
                "What went well?",
                "What could be improved?",
                "What will we do differently?",
            ],
            action_items=action_items,
            created_by=created_by,
            status="draft",
        )

        self.post_mortems[incident_id] = post_mortem

        await self.add_timeline_event(
            incident_id=incident_id,
            event_type=TimelineEventType.COMMENT,
            description="Post-mortem template generated",
            actor_id=created_by,
        )

        logger.info(
            "Generated post-mortem for incident %s",
            incident.incident_number,
        )

        return post_mortem

    async def get_post_mortem(
        self,
        incident_id: UUID,
    ) -> Optional[PostMortem]:
        """Get post-mortem for an incident.

        Args:
            incident_id: Incident UUID.

        Returns:
            PostMortem or None if not found.
        """
        return self.post_mortems.get(incident_id)

    async def create_jira_ticket(
        self,
        incident: Incident,
    ) -> Optional[Dict[str, Any]]:
        """Create a Jira ticket for the incident.

        Args:
            incident: Incident to create ticket for.

        Returns:
            Jira ticket info or None if failed.
        """
        if not self.config.jira.enabled:
            logger.debug("Jira integration disabled")
            return None

        base_url = self.config.jira.base_url
        if not base_url:
            logger.warning("Jira base URL not configured")
            return None

        api_token = self.config.jira.get_api_token()
        if not api_token:
            logger.warning("Jira API token not configured")
            return None

        try:
            client = await self._get_http_client()

            # Map severity to Jira priority
            priority_map = {
                EscalationLevel.P0: "Highest",
                EscalationLevel.P1: "High",
                EscalationLevel.P2: "Medium",
                EscalationLevel.P3: "Low",
            }

            # Build issue payload
            payload = {
                "fields": {
                    "project": {"key": self.config.jira.project_key},
                    "summary": f"[{incident.severity.value}] {incident.title}",
                    "description": self._build_jira_description(incident),
                    "issuetype": {"name": self.config.jira.issue_type},
                    "priority": {"name": priority_map.get(incident.severity, "Medium")},
                    "labels": [
                        "security-incident",
                        incident.incident_type.value,
                        incident.severity.value,
                    ],
                }
            }

            # Add custom fields
            for field_name, field_id in self.config.jira.custom_fields.items():
                if field_name == "incident_number":
                    payload["fields"][field_id] = incident.incident_number

            # Create issue
            auth = (self.config.jira.user_email or "", api_token)
            response = await client.post(
                f"{base_url}/rest/api/3/issue",
                json=payload,
                auth=auth,
            )
            response.raise_for_status()

            result = response.json()
            ticket_key = result.get("key")
            ticket_url = f"{base_url}/browse/{ticket_key}"

            # Update incident metadata
            incident.metadata["jira_ticket"] = ticket_key
            incident.metadata["jira_url"] = ticket_url

            await self.add_timeline_event(
                incident_id=incident.id,
                event_type=TimelineEventType.COMMENT,
                description=f"Jira ticket created: {ticket_key}",
                metadata={"ticket_key": ticket_key, "ticket_url": ticket_url},
            )

            logger.info(
                "Created Jira ticket %s for incident %s",
                ticket_key,
                incident.incident_number,
            )

            return {"key": ticket_key, "url": ticket_url}

        except httpx.HTTPError as e:
            logger.error("Failed to create Jira ticket: %s", e)
            return None
        except Exception as e:
            logger.error("Unexpected error creating Jira ticket: %s", e)
            return None

    def _build_jira_description(self, incident: Incident) -> str:
        """Build Jira issue description from incident.

        Args:
            incident: Incident to describe.

        Returns:
            Formatted description string.
        """
        description_parts = [
            f"h2. Security Incident: {incident.incident_number}",
            "",
            f"*Severity:* {incident.severity.value}",
            f"*Type:* {incident.incident_type.value}",
            f"*Status:* {incident.status.value}",
            f"*Source:* {incident.source.value}",
            f"*Detected:* {incident.detected_at.strftime('%Y-%m-%d %H:%M:%S UTC')}",
            "",
            "h3. Description",
            incident.description or "No description provided.",
            "",
        ]

        if incident.affected_systems:
            description_parts.extend([
                "h3. Affected Systems",
                "* " + "\n* ".join(incident.affected_systems),
                "",
            ])

        if incident.affected_users:
            description_parts.extend([
                f"h3. Affected Users: {incident.affected_users}",
                "",
            ])

        description_parts.extend([
            "h3. Dashboard Link",
            f"[View in Dashboard|https://dashboard.greenlang.io/incidents/{incident.id}]",
        ])

        return "\n".join(description_parts)

    def _format_duration(self, seconds: Optional[float]) -> str:
        """Format duration in seconds to human-readable string.

        Args:
            seconds: Duration in seconds.

        Returns:
            Formatted duration string.
        """
        if seconds is None:
            return "N/A"

        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            return f"{seconds / 60:.1f}m"
        else:
            return f"{seconds / 3600:.1f}h"

    async def list_incidents(
        self,
        status: Optional[IncidentStatus] = None,
        severity: Optional[EscalationLevel] = None,
        incident_type: Optional[IncidentType] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[Incident]:
        """List incidents with optional filters.

        Args:
            status: Filter by status.
            severity: Filter by severity.
            incident_type: Filter by type.
            limit: Maximum results.
            offset: Result offset.

        Returns:
            List of matching incidents.
        """
        filtered = list(self.incidents.values())

        if status:
            filtered = [i for i in filtered if i.status == status]
        if severity:
            filtered = [i for i in filtered if i.severity == severity]
        if incident_type:
            filtered = [i for i in filtered if i.incident_type == incident_type]

        # Sort by detected_at descending
        filtered.sort(key=lambda i: i.detected_at, reverse=True)

        return filtered[offset : offset + limit]

    async def get_active_incidents(self) -> List[Incident]:
        """Get all active (non-closed) incidents.

        Returns:
            List of active incidents.
        """
        return [
            i
            for i in self.incidents.values()
            if i.status != IncidentStatus.CLOSED
        ]

    async def add_comment(
        self,
        incident_id: UUID,
        comment: str,
        actor_id: Optional[UUID] = None,
        actor_name: Optional[str] = None,
    ) -> TimelineEvent:
        """Add a comment to an incident.

        Args:
            incident_id: Incident UUID.
            comment: Comment text.
            actor_id: Commenter UUID.
            actor_name: Commenter display name.

        Returns:
            Created TimelineEvent.
        """
        return await self.add_timeline_event(
            incident_id=incident_id,
            event_type=TimelineEventType.COMMENT,
            description=comment,
            actor_id=actor_id,
            actor_name=actor_name,
        )


# ---------------------------------------------------------------------------
# Global Tracker Instance
# ---------------------------------------------------------------------------

_global_tracker: Optional[IncidentTracker] = None


def get_tracker(
    config: Optional[IncidentResponseConfig] = None,
) -> IncidentTracker:
    """Get or create the global incident tracker.

    Args:
        config: Optional configuration override.

    Returns:
        The global IncidentTracker instance.
    """
    global _global_tracker

    if _global_tracker is None:
        _global_tracker = IncidentTracker(config)

    return _global_tracker


async def reset_tracker() -> None:
    """Reset and close the global tracker."""
    global _global_tracker

    if _global_tracker is not None:
        await _global_tracker.close()
        _global_tracker = None


__all__ = [
    "IncidentTracker",
    "get_tracker",
    "reset_tracker",
]
