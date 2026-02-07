# -*- coding: utf-8 -*-
"""
Alerting Service Models - OBS-004: Unified Alerting Service

Core data models for the unified alerting service including Alert,
NotificationResult, EscalationPolicy, and OnCall schedule structures.
All models use dataclasses for configuration types and provide full
serialization support.

Example:
    >>> from greenlang.infrastructure.alerting_service.models import (
    ...     Alert, AlertSeverity, AlertStatus, NotificationChannel,
    ... )
    >>> alert = Alert(
    ...     source="prometheus",
    ...     name="HighCPU",
    ...     severity=AlertSeverity.WARNING,
    ...     title="CPU usage above 90%",
    ...     description="Node cpu-1 is running hot.",
    ... )
    >>> print(alert.alert_id)

Author: GreenLang Platform Team
Date: February 2026
PRD: OBS-004 Unified Alerting Service
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class AlertSeverity(str, Enum):
    """Alert severity levels aligned with Prometheus / PagerDuty conventions."""

    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"


class AlertStatus(str, Enum):
    """Alert lifecycle status values."""

    FIRING = "firing"
    ACKNOWLEDGED = "acknowledged"
    INVESTIGATING = "investigating"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"


class NotificationStatus(str, Enum):
    """Outcome of a single notification delivery attempt."""

    SENT = "sent"
    FAILED = "failed"
    RATE_LIMITED = "rate_limited"
    SKIPPED = "skipped"


class NotificationChannel(str, Enum):
    """Supported notification delivery channels."""

    PAGERDUTY = "pagerduty"
    OPSGENIE = "opsgenie"
    SLACK = "slack"
    EMAIL = "email"
    TEAMS = "teams"
    WEBHOOK = "webhook"


# ---------------------------------------------------------------------------
# Alert
# ---------------------------------------------------------------------------


@dataclass
class Alert:
    """Core alert model representing a single alerting event.

    Attributes:
        alert_id: Unique identifier (UUID4 string).
        fingerprint: Deduplication fingerprint derived from source+name+labels.
        source: Originating system (e.g. ``prometheus``, ``loki``, ``app``).
        name: Alert rule name.
        severity: One of CRITICAL, WARNING, INFO.
        status: Current lifecycle status.
        title: Human-readable one-line summary.
        description: Detailed description of the alert condition.
        labels: Arbitrary key-value labels (e.g. ``instance``, ``job``).
        annotations: Arbitrary annotations (e.g. ``summary``, ``runbook_url``).
        tenant_id: Multi-tenant identifier.
        team: Owning team name.
        service: Affected service name.
        environment: Deployment environment (dev/staging/prod).
        fired_at: Timestamp when the alert first fired.
        acknowledged_at: Timestamp when the alert was acknowledged.
        acknowledged_by: User who acknowledged.
        resolved_at: Timestamp when the alert was resolved.
        resolved_by: User who resolved.
        escalation_level: Current escalation step (0 = not escalated).
        notification_count: Number of notifications sent for this alert.
        runbook_url: Link to the operational runbook.
        dashboard_url: Link to the relevant Grafana dashboard.
        related_trace_id: OpenTelemetry trace ID for correlation.
    """

    source: str
    name: str
    severity: AlertSeverity
    title: str
    description: str = ""
    alert_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    fingerprint: str = ""
    status: AlertStatus = AlertStatus.FIRING
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)
    tenant_id: str = ""
    team: str = ""
    service: str = ""
    environment: str = ""
    fired_at: Optional[datetime] = None
    acknowledged_at: Optional[datetime] = None
    acknowledged_by: str = ""
    resolved_at: Optional[datetime] = None
    resolved_by: str = ""
    escalation_level: int = 0
    notification_count: int = 0
    runbook_url: str = ""
    dashboard_url: str = ""
    related_trace_id: str = ""

    def __post_init__(self) -> None:
        """Set computed defaults after initialization."""
        if self.fired_at is None:
            self.fired_at = datetime.now(timezone.utc)
        if not self.fingerprint:
            self.fingerprint = self.generate_fingerprint(
                self.source, self.name, self.labels,
            )

    # ------------------------------------------------------------------
    # Fingerprint
    # ------------------------------------------------------------------

    @staticmethod
    def generate_fingerprint(
        source: str,
        name: str,
        labels: Dict[str, str],
    ) -> str:
        """Generate a stable deduplication fingerprint.

        Uses MD5 over ``source + name + sorted(labels)`` to produce a
        compact fingerprint suitable for deduplication lookups.

        Args:
            source: Alert source system.
            name: Alert rule name.
            labels: Label key-value pairs.

        Returns:
            Hex-encoded MD5 digest.
        """
        sorted_labels = "&".join(
            f"{k}={v}" for k, v in sorted(labels.items())
        )
        raw = f"{source}|{name}|{sorted_labels}"
        return hashlib.md5(raw.encode("utf-8")).hexdigest()

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the alert to a plain dictionary.

        Returns:
            Dictionary representation suitable for JSON encoding.
        """
        return {
            "alert_id": self.alert_id,
            "fingerprint": self.fingerprint,
            "source": self.source,
            "name": self.name,
            "severity": self.severity.value,
            "status": self.status.value,
            "title": self.title,
            "description": self.description,
            "labels": dict(self.labels),
            "annotations": dict(self.annotations),
            "tenant_id": self.tenant_id,
            "team": self.team,
            "service": self.service,
            "environment": self.environment,
            "fired_at": self.fired_at.isoformat() if self.fired_at else None,
            "acknowledged_at": (
                self.acknowledged_at.isoformat()
                if self.acknowledged_at
                else None
            ),
            "acknowledged_by": self.acknowledged_by,
            "resolved_at": (
                self.resolved_at.isoformat() if self.resolved_at else None
            ),
            "resolved_by": self.resolved_by,
            "escalation_level": self.escalation_level,
            "notification_count": self.notification_count,
            "runbook_url": self.runbook_url,
            "dashboard_url": self.dashboard_url,
            "related_trace_id": self.related_trace_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Alert:
        """Deserialize an alert from a plain dictionary.

        Args:
            data: Dictionary (e.g. from JSON).

        Returns:
            Alert instance.
        """
        severity = data.get("severity", "info")
        if isinstance(severity, str):
            severity = AlertSeverity(severity)

        status = data.get("status", "firing")
        if isinstance(status, str):
            status = AlertStatus(status)

        fired_at = data.get("fired_at")
        if isinstance(fired_at, str):
            fired_at = datetime.fromisoformat(fired_at)

        acknowledged_at = data.get("acknowledged_at")
        if isinstance(acknowledged_at, str):
            acknowledged_at = datetime.fromisoformat(acknowledged_at)

        resolved_at = data.get("resolved_at")
        if isinstance(resolved_at, str):
            resolved_at = datetime.fromisoformat(resolved_at)

        return cls(
            alert_id=data.get("alert_id", str(uuid.uuid4())),
            fingerprint=data.get("fingerprint", ""),
            source=data.get("source", "unknown"),
            name=data.get("name", "unknown"),
            severity=severity,
            status=status,
            title=data.get("title", ""),
            description=data.get("description", ""),
            labels=data.get("labels", {}),
            annotations=data.get("annotations", {}),
            tenant_id=data.get("tenant_id", ""),
            team=data.get("team", ""),
            service=data.get("service", ""),
            environment=data.get("environment", ""),
            fired_at=fired_at,
            acknowledged_at=acknowledged_at,
            acknowledged_by=data.get("acknowledged_by", ""),
            resolved_at=resolved_at,
            resolved_by=data.get("resolved_by", ""),
            escalation_level=data.get("escalation_level", 0),
            notification_count=data.get("notification_count", 0),
            runbook_url=data.get("runbook_url", ""),
            dashboard_url=data.get("dashboard_url", ""),
            related_trace_id=data.get("related_trace_id", ""),
        )


# ---------------------------------------------------------------------------
# Notification Result
# ---------------------------------------------------------------------------


@dataclass
class NotificationResult:
    """Outcome of a single notification delivery attempt.

    Attributes:
        channel: Which channel was used.
        status: Delivery outcome.
        recipient: Target address / identifier.
        duration_ms: Round-trip time in milliseconds.
        response_code: HTTP status code (or provider-specific code).
        error_message: Human-readable error detail on failure.
        sent_at: Timestamp of the delivery attempt.
    """

    channel: NotificationChannel
    status: NotificationStatus
    recipient: str = ""
    duration_ms: float = 0.0
    response_code: int = 0
    error_message: str = ""
    sent_at: Optional[datetime] = None

    def __post_init__(self) -> None:
        """Set sent_at to now if not provided."""
        if self.sent_at is None:
            self.sent_at = datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Escalation Models
# ---------------------------------------------------------------------------


@dataclass
class EscalationStep:
    """Single step within an escalation policy.

    Attributes:
        delay_minutes: Minutes to wait before executing this step.
        channels: Notification channels to use at this step.
        oncall_schedule_id: Optional on-call schedule to notify.
        notify_users: Explicit user IDs to notify.
        repeat: Number of times to repeat this step.
    """

    delay_minutes: int
    channels: List[str] = field(default_factory=list)
    oncall_schedule_id: str = ""
    notify_users: List[str] = field(default_factory=list)
    repeat: int = 1


@dataclass
class EscalationPolicy:
    """Ordered sequence of escalation steps for a severity class.

    Attributes:
        name: Human-readable policy name (e.g. ``critical_default``).
        steps: Ordered list of escalation steps.
    """

    name: str
    steps: List[EscalationStep] = field(default_factory=list)


# ---------------------------------------------------------------------------
# On-Call Models
# ---------------------------------------------------------------------------


@dataclass
class OnCallUser:
    """A single on-call responder.

    Attributes:
        user_id: Unique user identifier in the on-call provider.
        name: Display name.
        email: Contact e-mail.
        phone: Contact phone number (E.164 format).
        provider: On-call provider (``pagerduty`` or ``opsgenie``).
        schedule_id: Schedule this user is on-call for.
    """

    user_id: str
    name: str
    email: str = ""
    phone: str = ""
    provider: str = ""
    schedule_id: str = ""


@dataclass
class OnCallSchedule:
    """An on-call schedule with the current responder.

    Attributes:
        schedule_id: Unique schedule identifier.
        name: Human-readable schedule name.
        provider: On-call provider.
        current_oncall: The user currently on-call.
        timezone: Schedule timezone (e.g. ``UTC``).
    """

    schedule_id: str
    name: str
    provider: str = ""
    current_oncall: Optional[OnCallUser] = None
    timezone: str = "UTC"
