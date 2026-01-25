# -*- coding: utf-8 -*-
"""
GL-010 EmissionsGuardian - Alert Management Module

This module provides comprehensive alert management including definitions,
routing, throttling, deduplication, escalation policies, and acknowledgment
tracking for emissions compliance monitoring.

Features:
    - Alert definitions with warning, error, critical severities
    - Multi-channel routing (email, SMS, webhook, PagerDuty)
    - Alert throttling and deduplication
    - Escalation policies with time-based triggers
    - Alert acknowledgment tracking
    - Provenance tracking with SHA-256 hashes

Example:
    >>> from monitoring.alerts import AlertManager
    >>> manager = AlertManager()
    >>> manager.send_alert(Alert(
    ...     alert_type=AlertType.COMPLIANCE_EXCEEDANCE,
    ...     severity=AlertSeverity.CRITICAL,
    ...     title="NOx Exceedance Detected",
    ...     message="Unit 1 NOx exceeded permit limit"
    ... ))
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import smtplib
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set
from urllib.request import Request, urlopen
from urllib.error import URLError

logger = logging.getLogger(__name__)


# =============================================================================
# Enums
# =============================================================================

class AlertSeverity(str, Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertType(str, Enum):
    """Alert types for EmissionsGuardian."""
    COMPLIANCE_EXCEEDANCE = "compliance_exceedance"
    COMPLIANCE_WARNING = "compliance_warning"
    FUGITIVE_DETECTION = "fugitive_detection"
    CEMS_FAILURE = "cems_failure"
    CEMS_DATA_QUALITY = "cems_data_quality"
    CALIBRATION_DUE = "calibration_due"
    CALIBRATION_FAILURE = "calibration_failure"
    RATA_DUE = "rata_due"
    RATA_FAILURE = "rata_failure"
    DATA_AVAILABILITY = "data_availability"
    SYSTEM_ERROR = "system_error"
    DEADLINE_APPROACHING = "deadline_approaching"
    ALLOWANCE_LOW = "allowance_low"
    SAFETY_INTERLOCK = "safety_interlock"


class AlertState(str, Enum):
    """Alert lifecycle states."""
    OPEN = "open"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"


class NotificationChannel(str, Enum):
    """Notification delivery channels."""
    EMAIL = "email"
    SMS = "sms"
    WEBHOOK = "webhook"
    PAGERDUTY = "pagerduty"
    SLACK = "slack"
    TEAMS = "teams"
    LOG = "log"


# =============================================================================
# Data Models
# =============================================================================

@dataclass
class Alert:
    """Alert data model."""
    alert_type: AlertType
    severity: AlertSeverity
    title: str
    message: str
    alert_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    state: AlertState = AlertState.OPEN
    source: str = "EmissionsGuardian"
    facility_id: Optional[str] = None
    unit_id: Optional[str] = None
    pollutant: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    acknowledged_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None
    resolved_at: Optional[datetime] = None
    resolved_by: Optional[str] = None
    escalation_level: int = 0
    notification_count: int = 0
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    provenance_hash: str = ""

    def __post_init__(self):
        if not self.provenance_hash:
            self.provenance_hash = self.calculate_provenance()

    def calculate_provenance(self) -> str:
        content = f"{self.alert_id}|{self.alert_type.value}|{self.severity.value}|{self.title}|{self.created_at.isoformat()}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "alert_id": self.alert_id,
            "alert_type": self.alert_type.value,
            "severity": self.severity.value,
            "state": self.state.value,
            "title": self.title,
            "message": self.message,
            "source": self.source,
            "facility_id": self.facility_id,
            "unit_id": self.unit_id,
            "pollutant": self.pollutant,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "acknowledged_at": self.acknowledged_at.isoformat() if self.acknowledged_at else None,
            "acknowledged_by": self.acknowledged_by,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "resolved_by": self.resolved_by,
            "escalation_level": self.escalation_level,
            "notification_count": self.notification_count,
            "tags": self.tags,
            "metadata": self.metadata,
            "provenance_hash": self.provenance_hash
        }

    def acknowledge(self, by: str) -> None:
        self.state = AlertState.ACKNOWLEDGED
        self.acknowledged_at = datetime.utcnow()
        self.acknowledged_by = by
        self.updated_at = datetime.utcnow()

    def resolve(self, by: str) -> None:
        self.state = AlertState.RESOLVED
        self.resolved_at = datetime.utcnow()
        self.resolved_by = by
        self.updated_at = datetime.utcnow()


@dataclass
class NotificationTarget:
    """Target for alert notifications."""
    channel: NotificationChannel
    address: str
    name: Optional[str] = None
    enabled: bool = True
    config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EscalationPolicy:
    """Escalation policy for alerts."""
    name: str
    alert_types: List[AlertType]
    severities: List[AlertSeverity]
    levels: List[Dict[str, Any]] = field(default_factory=list)
    enabled: bool = True

    def get_targets_for_level(self, level: int) -> List[NotificationTarget]:
        if level < len(self.levels):
            return self.levels[level].get("targets", [])
        return []

    def get_delay_for_level(self, level: int) -> int:
        if level < len(self.levels):
            return self.levels[level].get("delay_minutes", 15)
        return 15



# =============================================================================
# Notification Handlers
# =============================================================================

class NotificationHandler:
    """Base class for notification handlers."""

    def __init__(self, channel: NotificationChannel):
        self.channel = channel

    def send(self, alert: Alert, target: NotificationTarget) -> bool:
        raise NotImplementedError


class EmailNotificationHandler(NotificationHandler):
    """Email notification handler."""

    def __init__(self, smtp_host: str = 'localhost', smtp_port: int = 25, username: Optional[str] = None, password: Optional[str] = None, use_tls: bool = True):
        super().__init__(NotificationChannel.EMAIL)
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.use_tls = use_tls

    def send(self, alert: Alert, target: NotificationTarget) -> bool:
        try:
            msg = MIMEMultipart()
            msg['Subject'] = f'[{alert.severity.value.upper()}] {alert.title}'
            msg['From'] = target.config.get('from_address', 'emissionsguardian@greenlang.io')
            msg['To'] = target.address

            body = f"""
Alert: {alert.title}
Severity: {alert.severity.value}
Type: {alert.alert_type.value}
Time: {alert.created_at.isoformat()}
Facility: {alert.facility_id or 'N/A'}
Unit: {alert.unit_id or 'N/A'}

Message:
{alert.message}

Alert ID: {alert.alert_id}
Provenance: {alert.provenance_hash}
"""
            msg.attach(MIMEText(body, 'plain'))

            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                if self.use_tls:
                    server.starttls()
                if self.username and self.password:
                    server.login(self.username, self.password)
                server.send_message(msg)
            logger.info(f'Email sent for alert {alert.alert_id} to {target.address}')
            return True
        except Exception as e:
            logger.error(f'Failed to send email for alert {alert.alert_id}: {e}')
            return False


class WebhookNotificationHandler(NotificationHandler):
    """Webhook notification handler."""

    def __init__(self):
        super().__init__(NotificationChannel.WEBHOOK)

    def send(self, alert: Alert, target: NotificationTarget) -> bool:
        try:
            payload = json.dumps(alert.to_dict()).encode('utf-8')
            headers = {'Content-Type': 'application/json'}
            headers.update(target.config.get('headers', {}))

            req = Request(target.address, data=payload, headers=headers, method='POST')
            with urlopen(req, timeout=10) as response:
                if response.status == 200:
                    logger.info(f'Webhook sent for alert {alert.alert_id} to {target.address}')
                    return True
                else:
                    logger.warning(f'Webhook returned status {response.status} for alert {alert.alert_id}')
                    return False
        except URLError as e:
            logger.error(f'Failed to send webhook for alert {alert.alert_id}: {e}')
            return False
        except Exception as e:
            logger.error(f'Failed to send webhook for alert {alert.alert_id}: {e}')
            return False


class PagerDutyNotificationHandler(NotificationHandler):
    """PagerDuty notification handler."""

    def __init__(self):
        super().__init__(NotificationChannel.PAGERDUTY)
        self.events_url = 'https://events.pagerduty.com/v2/enqueue'

    def send(self, alert: Alert, target: NotificationTarget) -> bool:
        try:
            severity_map = {
                AlertSeverity.INFO: 'info',
                AlertSeverity.WARNING: 'warning',
                AlertSeverity.ERROR: 'error',
                AlertSeverity.CRITICAL: 'critical'
            }

            payload = {
                'routing_key': target.address,
                'event_action': 'trigger',
                'dedup_key': alert.alert_id,
                'payload': {
                    'summary': alert.title,
                    'severity': severity_map.get(alert.severity, 'error'),
                    'source': alert.source,
                    'custom_details': {
                        'message': alert.message,
                        'alert_type': alert.alert_type.value,
                        'facility_id': alert.facility_id,
                        'unit_id': alert.unit_id,
                        'pollutant': alert.pollutant,
                        'provenance_hash': alert.provenance_hash
                    }
                }
            }

            data = json.dumps(payload).encode('utf-8')
            req = Request(self.events_url, data=data, headers={'Content-Type': 'application/json'}, method='POST')
            with urlopen(req, timeout=10) as response:
                if response.status == 202:
                    logger.info(f'PagerDuty event sent for alert {alert.alert_id}')
                    return True
                return False
        except Exception as e:
            logger.error(f'Failed to send PagerDuty event for alert {alert.alert_id}: {e}')
            return False


class LogNotificationHandler(NotificationHandler):
    """Log notification handler for testing/local development."""

    def __init__(self):
        super().__init__(NotificationChannel.LOG)

    def send(self, alert: Alert, target: NotificationTarget) -> bool:
        log_msg = f'ALERT [{alert.severity.value}] {alert.title}: {alert.message} (ID: {alert.alert_id})'
        if alert.severity == AlertSeverity.CRITICAL:
            logger.critical(log_msg)
        elif alert.severity == AlertSeverity.ERROR:
            logger.error(log_msg)
        elif alert.severity == AlertSeverity.WARNING:
            logger.warning(log_msg)
        else:
            logger.info(log_msg)
        return True



# =============================================================================
# Alert Throttling and Deduplication
# =============================================================================

class AlertThrottler:
    """Throttles alert notifications to prevent spam."""

    def __init__(self, window_seconds: int = 300, max_alerts: int = 5):
        self.window_seconds = window_seconds
        self.max_alerts = max_alerts
        self._alert_times: Dict[str, List[datetime]] = {}
        self._lock = threading.Lock()

    def _get_throttle_key(self, alert: Alert) -> str:
        return f"{alert.alert_type.value}:{alert.facility_id}:{alert.unit_id}:{alert.pollutant}"

    def should_throttle(self, alert: Alert) -> bool:
        key = self._get_throttle_key(alert)
        now = datetime.utcnow()
        cutoff = now - timedelta(seconds=self.window_seconds)

        with self._lock:
            if key not in self._alert_times:
                self._alert_times[key] = []

            # Remove old entries
            self._alert_times[key] = [t for t in self._alert_times[key] if t > cutoff]

            # Check if throttled
            if len(self._alert_times[key]) >= self.max_alerts:
                logger.debug(f'Alert throttled: {key}')
                return True

            # Record this alert
            self._alert_times[key].append(now)
            return False

    def clear(self) -> None:
        with self._lock:
            self._alert_times.clear()


class AlertDeduplicator:
    """Deduplicates alerts to prevent duplicates."""

    def __init__(self, ttl_seconds: int = 3600):
        self.ttl_seconds = ttl_seconds
        self._seen_alerts: Dict[str, datetime] = {}
        self._lock = threading.Lock()

    def _get_dedup_key(self, alert: Alert) -> str:
        content = f"{alert.alert_type.value}|{alert.severity.value}|{alert.title}|{alert.facility_id}|{alert.unit_id}"
        return hashlib.sha256(content.encode()).hexdigest()[:32]

    def is_duplicate(self, alert: Alert) -> bool:
        key = self._get_dedup_key(alert)
        now = datetime.utcnow()
        cutoff = now - timedelta(seconds=self.ttl_seconds)

        with self._lock:
            # Clean old entries
            self._seen_alerts = {k: v for k, v in self._seen_alerts.items() if v > cutoff}

            if key in self._seen_alerts:
                logger.debug(f'Duplicate alert detected: {alert.alert_id}')
                return True

            self._seen_alerts[key] = now
            return False

    def clear(self) -> None:
        with self._lock:
            self._seen_alerts.clear()



# =============================================================================
# Alert Manager
# =============================================================================

class AlertManager:
    """Central alert management system."""

    _instance: Optional['AlertManager'] = None
    _lock = threading.Lock()

    def __new__(cls) -> 'AlertManager':
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if getattr(self, '_initialized', False):
            return
        self._initialized = True
        self._alerts: Dict[str, Alert] = {}
        self._handlers: Dict[NotificationChannel, NotificationHandler] = {}
        self._policies: List[EscalationPolicy] = []
        self._default_targets: List[NotificationTarget] = []
        self._throttler = AlertThrottler()
        self._deduplicator = AlertDeduplicator()
        self._alert_lock = threading.Lock()

        # Register default handlers
        self._handlers[NotificationChannel.LOG] = LogNotificationHandler()
        self._handlers[NotificationChannel.WEBHOOK] = WebhookNotificationHandler()
        self._handlers[NotificationChannel.EMAIL] = EmailNotificationHandler()
        self._handlers[NotificationChannel.PAGERDUTY] = PagerDutyNotificationHandler()

        logger.info('AlertManager initialized')

    def register_handler(self, handler: NotificationHandler) -> None:
        self._handlers[handler.channel] = handler
        logger.info(f'Registered notification handler: {handler.channel.value}')

    def add_default_target(self, target: NotificationTarget) -> None:
        self._default_targets.append(target)
        logger.info(f'Added default notification target: {target.channel.value} -> {target.address}')

    def add_escalation_policy(self, policy: EscalationPolicy) -> None:
        self._policies.append(policy)
        logger.info(f'Added escalation policy: {policy.name}')

    def send_alert(self, alert: Alert, targets: Optional[List[NotificationTarget]] = None) -> bool:
        # Check for duplicates
        if self._deduplicator.is_duplicate(alert):
            logger.debug(f'Alert {alert.alert_id} is a duplicate, skipping')
            return False

        # Check for throttling
        if self._throttler.should_throttle(alert):
            logger.debug(f'Alert {alert.alert_id} is throttled, skipping')
            return False

        # Store alert
        with self._alert_lock:
            self._alerts[alert.alert_id] = alert

        # Determine targets
        notification_targets = targets or self._default_targets
        if not notification_targets:
            notification_targets = [NotificationTarget(channel=NotificationChannel.LOG, address='')]

        # Send notifications
        success = False
        for target in notification_targets:
            if not target.enabled:
                continue
            handler = self._handlers.get(target.channel)
            if handler:
                try:
                    if handler.send(alert, target):
                        success = True
                        alert.notification_count += 1
                except Exception as e:
                    logger.error(f'Error sending notification: {e}')

        return success

    def acknowledge_alert(self, alert_id: str, by: str) -> Optional[Alert]:
        with self._alert_lock:
            alert = self._alerts.get(alert_id)
            if alert:
                alert.acknowledge(by)
                logger.info(f'Alert {alert_id} acknowledged by {by}')
            return alert

    def resolve_alert(self, alert_id: str, by: str) -> Optional[Alert]:
        with self._alert_lock:
            alert = self._alerts.get(alert_id)
            if alert:
                alert.resolve(by)
                logger.info(f'Alert {alert_id} resolved by {by}')
            return alert

    def get_alert(self, alert_id: str) -> Optional[Alert]:
        with self._alert_lock:
            return self._alerts.get(alert_id)

    def get_open_alerts(self) -> List[Alert]:
        with self._alert_lock:
            return [a for a in self._alerts.values() if a.state == AlertState.OPEN]

    def get_alerts_by_severity(self, severity: AlertSeverity) -> List[Alert]:
        with self._alert_lock:
            return [a for a in self._alerts.values() if a.severity == severity]

    def get_alerts_by_facility(self, facility_id: str) -> List[Alert]:
        with self._alert_lock:
            return [a for a in self._alerts.values() if a.facility_id == facility_id]

    def clear_resolved_alerts(self, older_than_hours: int = 24) -> int:
        cutoff = datetime.utcnow() - timedelta(hours=older_than_hours)
        count = 0
        with self._alert_lock:
            to_remove = [
                aid for aid, alert in self._alerts.items()
                if alert.state == AlertState.RESOLVED and alert.resolved_at and alert.resolved_at < cutoff
            ]
            for aid in to_remove:
                del self._alerts[aid]
                count += 1
        logger.info(f'Cleared {count} resolved alerts older than {older_than_hours} hours')
        return count


# =============================================================================
# Convenience Functions
# =============================================================================

_default_manager: Optional[AlertManager] = None


def get_alert_manager() -> AlertManager:
    global _default_manager
    if _default_manager is None:
        _default_manager = AlertManager()
    return _default_manager


def send_alert(alert: Alert, targets: Optional[List[NotificationTarget]] = None) -> bool:
    return get_alert_manager().send_alert(alert, targets)


def send_compliance_alert(
    title: str,
    message: str,
    severity: AlertSeverity = AlertSeverity.WARNING,
    facility_id: Optional[str] = None,
    unit_id: Optional[str] = None,
    pollutant: Optional[str] = None
) -> Alert:
    alert = Alert(
        alert_type=AlertType.COMPLIANCE_EXCEEDANCE if severity in [AlertSeverity.ERROR, AlertSeverity.CRITICAL] else AlertType.COMPLIANCE_WARNING,
        severity=severity,
        title=title,
        message=message,
        facility_id=facility_id,
        unit_id=unit_id,
        pollutant=pollutant
    )
    get_alert_manager().send_alert(alert)
    return alert


def send_fugitive_alert(
    title: str,
    message: str,
    confidence: float,
    facility_id: Optional[str] = None,
    source_type: Optional[str] = None
) -> Alert:
    severity = AlertSeverity.CRITICAL if confidence >= 0.9 else AlertSeverity.WARNING
    alert = Alert(
        alert_type=AlertType.FUGITIVE_DETECTION,
        severity=severity,
        title=title,
        message=message,
        facility_id=facility_id,
        metadata={"confidence": confidence, "source_type": source_type}
    )
    get_alert_manager().send_alert(alert)
    return alert
