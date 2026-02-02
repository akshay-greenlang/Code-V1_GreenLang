"""
FurnacePulse Alerts Module

This module provides centralized alert management, notification services,
and response playbook management for the FurnacePulse furnace monitoring system.

Components:
    - AlertOrchestrator: Centralized alert taxonomy, scoring, and escalation
    - NotificationService: Multi-channel notification delivery and tracking
    - ResponsePlaybookManager: Guided response procedures for each alert type

Alert Taxonomy:
    - A-001: Hotspot Advisory (TMT approaching threshold)
    - A-002: Hotspot Warning (Sustained TMT exceedance)
    - A-003: Hotspot Urgent (High-confidence hotspot risk)
    - A-010: Efficiency Degradation
    - A-020: Draft Instability
    - A-030: Sensor Drift/Stuck

Example:
    >>> from alerts import AlertOrchestrator, NotificationService
    >>> orchestrator = AlertOrchestrator(config)
    >>> notification_service = NotificationService(config)
    >>> alert = orchestrator.create_alert("A-001", sensor_data)
    >>> notification_service.send(alert)
"""

from alerts.alert_orchestrator import (
    AlertOrchestrator,
    Alert,
    AlertCode,
    AlertSeverity,
    OwnerRole,
)
from alerts.notification_service import (
    NotificationService,
    NotificationChannel,
    NotificationStatus,
)
from alerts.response_playbooks import (
    ResponsePlaybookManager,
    Playbook,
    PlaybookStep,
)

__all__ = [
    # Alert Orchestrator
    "AlertOrchestrator",
    "Alert",
    "AlertCode",
    "AlertSeverity",
    "OwnerRole",
    # Notification Service
    "NotificationService",
    "NotificationChannel",
    "NotificationStatus",
    # Response Playbooks
    "ResponsePlaybookManager",
    "Playbook",
    "PlaybookStep",
]

__version__ = "1.0.0"
