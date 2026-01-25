"""
Security Event Monitoring Service - Anomaly Detection and SIEM Integration

This module provides comprehensive security event monitoring including failed
login tracking, anomaly detection for data access patterns, alerting for
suspicious activity, and SIEM integration for centralized security monitoring.

SOC2 Controls Addressed:
    - CC7.1: System operations monitoring
    - CC7.2: Incident detection and monitoring
    - CC7.3: Incident response

ISO27001 Controls Addressed:
    - A.12.4.1: Event logging
    - A.12.4.3: Administrator and operator logs
    - A.16.1.2: Reporting information security events

Example:
    >>> config = SecurityMonitorConfig()
    >>> monitor = SecurityMonitor(config)
    >>> await monitor.initialize()
    >>> await monitor.record_login_attempt(
    ...     user_id="user-123",
    ...     success=False,
    ...     ip_address="192.168.1.100",
    ... )
    >>> alerts = await monitor.get_active_alerts()
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import statistics
import uuid
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ThreatLevel(str, Enum):
    """Threat severity levels."""

    INFO = "INFO"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class EventCategory(str, Enum):
    """Categories of security events."""

    AUTHENTICATION = "AUTHENTICATION"
    AUTHORIZATION = "AUTHORIZATION"
    DATA_ACCESS = "DATA_ACCESS"
    DATA_EXFILTRATION = "DATA_EXFILTRATION"
    BRUTE_FORCE = "BRUTE_FORCE"
    PRIVILEGE_ESCALATION = "PRIVILEGE_ESCALATION"
    ANOMALY = "ANOMALY"
    POLICY_VIOLATION = "POLICY_VIOLATION"
    MALWARE = "MALWARE"
    INTRUSION = "INTRUSION"
    CONFIGURATION_CHANGE = "CONFIGURATION_CHANGE"


class AlertStatus(str, Enum):
    """Status of security alerts."""

    OPEN = "OPEN"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    INVESTIGATING = "INVESTIGATING"
    RESOLVED = "RESOLVED"
    FALSE_POSITIVE = "FALSE_POSITIVE"
    ESCALATED = "ESCALATED"


class SecurityEvent(BaseModel):
    """
    Security event for monitoring and analysis.

    Captures security-relevant events for threat detection
    and compliance monitoring.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    category: EventCategory
    threat_level: ThreatLevel = Field(default=ThreatLevel.INFO)

    # Event details
    event_type: str = Field(..., description="Specific event type")
    description: str = Field(..., description="Human-readable description")
    details: Dict[str, Any] = Field(default_factory=dict)

    # Actor information
    actor_id: Optional[str] = Field(None)
    actor_type: str = Field(default="user")
    tenant_id: Optional[str] = Field(None)

    # Source information
    source_ip: Optional[str] = Field(None)
    source_port: Optional[int] = Field(None)
    user_agent: Optional[str] = Field(None)
    geo_location: Optional[Dict[str, Any]] = Field(None)

    # Target information
    target_resource: Optional[str] = Field(None)
    target_action: Optional[str] = Field(None)

    # Correlation
    correlation_id: Optional[str] = Field(None)
    related_events: List[str] = Field(default_factory=list)

    # Processing status
    processed: bool = Field(default=False)
    alerts_generated: List[str] = Field(default_factory=list)

    class Config:
        """Pydantic configuration."""

        json_encoders = {
            datetime: lambda v: v.isoformat(),
        }


class SecurityAlert(BaseModel):
    """
    Security alert requiring attention.

    Generated from security events that match threat detection rules.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # Alert classification
    category: EventCategory
    threat_level: ThreatLevel
    title: str = Field(..., description="Alert title")
    description: str = Field(..., description="Detailed description")

    # Status tracking
    status: AlertStatus = Field(default=AlertStatus.OPEN)
    assigned_to: Optional[str] = Field(None)

    # Context
    tenant_id: Optional[str] = Field(None)
    affected_users: List[str] = Field(default_factory=list)
    affected_resources: List[str] = Field(default_factory=list)

    # Evidence
    source_events: List[str] = Field(default_factory=list)
    indicators: Dict[str, Any] = Field(default_factory=dict)
    raw_data: Optional[Dict[str, Any]] = Field(None)

    # Response
    response_actions: List[str] = Field(default_factory=list)
    resolution_notes: Optional[str] = Field(None)
    resolved_at: Optional[datetime] = Field(None)
    resolved_by: Optional[str] = Field(None)

    # SIEM integration
    siem_ticket_id: Optional[str] = Field(None)
    external_references: Dict[str, str] = Field(default_factory=dict)

    class Config:
        """Pydantic configuration."""

        json_encoders = {
            datetime: lambda v: v.isoformat(),
        }


class AnomalyRule(BaseModel):
    """
    Rule for detecting anomalous behavior.

    Defines thresholds and conditions for anomaly detection.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: str
    enabled: bool = Field(default=True)
    category: EventCategory

    # Trigger conditions
    metric: str = Field(..., description="Metric to monitor")
    aggregation: str = Field(default="count", description="count, sum, avg, max, min")
    window_seconds: int = Field(default=300)
    threshold: float = Field(...)
    comparison: str = Field(default="gt", description="gt, lt, gte, lte, eq")

    # Alert configuration
    threat_level: ThreatLevel = Field(default=ThreatLevel.MEDIUM)
    alert_title_template: str
    alert_description_template: str

    # Rate limiting
    cooldown_seconds: int = Field(default=300)
    max_alerts_per_hour: int = Field(default=10)


class SIEMConfig(BaseModel):
    """Configuration for SIEM integration."""

    enabled: bool = Field(default=False)
    provider: str = Field(default="splunk", description="splunk, elastic, sentinel, qradar")
    endpoint: Optional[str] = Field(None)
    api_key: Optional[str] = Field(None)
    index: str = Field(default="greenlang-security")
    batch_size: int = Field(default=100)
    flush_interval_seconds: int = Field(default=10)


class SecurityMonitorConfig(BaseModel):
    """Configuration for the Security Monitor."""

    # Failed login tracking
    failed_login_threshold: int = Field(default=5, description="Failures before lockout")
    failed_login_window_seconds: int = Field(default=300)
    lockout_duration_seconds: int = Field(default=900)

    # Data access anomaly detection
    data_access_baseline_days: int = Field(default=30)
    data_access_deviation_threshold: float = Field(default=3.0, description="Standard deviations")

    # Alerting
    alert_retention_days: int = Field(default=90)
    auto_escalate_critical: bool = Field(default=True)
    escalation_timeout_minutes: int = Field(default=15)

    # SIEM integration
    siem: SIEMConfig = Field(default_factory=SIEMConfig)

    # Event retention
    event_retention_days: int = Field(default=90)
    event_buffer_size: int = Field(default=10000)


class SecurityMonitor:
    """
    Production-grade security event monitoring service.

    Provides comprehensive security monitoring including:
    - Failed login tracking with automatic lockout
    - Data access anomaly detection
    - Threat detection rules engine
    - Alert management and escalation
    - SIEM integration for centralized monitoring

    Example:
        >>> config = SecurityMonitorConfig()
        >>> monitor = SecurityMonitor(config)
        >>> await monitor.initialize()
        >>>
        >>> # Record authentication events
        >>> await monitor.record_login_attempt(
        ...     user_id="user-123",
        ...     success=False,
        ...     ip_address="192.168.1.100",
        ... )
        >>>
        >>> # Check for anomalies
        >>> anomalies = await monitor.detect_anomalies("tenant-123")
        >>>
        >>> # Get alerts
        >>> alerts = await monitor.get_active_alerts()

    Attributes:
        config: Monitor configuration
        _events: Security event buffer
        _alerts: Active alerts
        _failed_logins: Failed login tracking
        _data_access_stats: Data access statistics for anomaly detection
    """

    def __init__(self, config: Optional[SecurityMonitorConfig] = None):
        """
        Initialize the Security Monitor.

        Args:
            config: Monitor configuration
        """
        self.config = config or SecurityMonitorConfig()
        self._events: List[SecurityEvent] = []
        self._alerts: Dict[str, SecurityAlert] = {}
        self._failed_logins: Dict[str, List[datetime]] = defaultdict(list)
        self._locked_accounts: Dict[str, datetime] = {}
        self._data_access_stats: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
        self._anomaly_rules: Dict[str, AnomalyRule] = {}
        self._rule_last_fired: Dict[str, datetime] = {}
        self._siem_buffer: List[SecurityEvent] = []
        self._processing_task: Optional[asyncio.Task] = None
        self._siem_task: Optional[asyncio.Task] = None
        self._initialized = False

        logger.info(
            "SecurityMonitor initialized",
            extra={
                "failed_login_threshold": self.config.failed_login_threshold,
                "siem_enabled": self.config.siem.enabled,
            },
        )

    async def initialize(self) -> None:
        """Initialize the security monitor and start background tasks."""
        if self._initialized:
            logger.warning("SecurityMonitor already initialized")
            return

        try:
            # Load default anomaly rules
            self._load_default_rules()

            # Start background processing
            self._processing_task = asyncio.create_task(self._process_events_loop())

            # Start SIEM forwarding if enabled
            if self.config.siem.enabled:
                self._siem_task = asyncio.create_task(self._siem_forward_loop())

            self._initialized = True
            logger.info("SecurityMonitor initialization complete")

        except Exception as e:
            logger.error(f"Failed to initialize SecurityMonitor: {e}", exc_info=True)
            raise

    async def shutdown(self) -> None:
        """Gracefully shutdown the security monitor."""
        logger.info("Shutting down SecurityMonitor...")

        if self._processing_task:
            self._processing_task.cancel()
            try:
                await self._processing_task
            except asyncio.CancelledError:
                pass

        if self._siem_task:
            self._siem_task.cancel()
            try:
                await self._siem_task
            except asyncio.CancelledError:
                pass

        # Flush SIEM buffer
        if self._siem_buffer:
            await self._flush_to_siem()

        self._initialized = False
        logger.info("SecurityMonitor shutdown complete")

    async def record_event(self, event: SecurityEvent) -> SecurityEvent:
        """
        Record a security event for monitoring.

        Args:
            event: Security event to record

        Returns:
            The recorded event
        """
        # Add to event buffer
        self._events.append(event)

        # Trim buffer if needed
        if len(self._events) > self.config.event_buffer_size:
            self._events = self._events[-self.config.event_buffer_size:]

        # Add to SIEM buffer
        if self.config.siem.enabled:
            self._siem_buffer.append(event)

        logger.debug(
            f"Security event recorded: {event.event_type}",
            extra={
                "event_id": event.id,
                "category": event.category.value,
                "threat_level": event.threat_level.value,
            },
        )

        return event

    async def record_login_attempt(
        self,
        user_id: str,
        success: bool,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        tenant_id: Optional[str] = None,
        failure_reason: Optional[str] = None,
    ) -> Tuple[SecurityEvent, Optional[SecurityAlert]]:
        """
        Record a login attempt and check for brute force attacks.

        Args:
            user_id: User attempting to login
            success: Whether login was successful
            ip_address: Client IP address
            user_agent: Client user agent
            tenant_id: Tenant context
            failure_reason: Reason for failure if unsuccessful

        Returns:
            Tuple of (event, alert if generated)
        """
        now = datetime.now(timezone.utc)

        # Create event
        event = SecurityEvent(
            category=EventCategory.AUTHENTICATION,
            event_type="LOGIN_SUCCESS" if success else "LOGIN_FAILURE",
            description=f"Login {'successful' if success else 'failed'} for user {user_id}",
            threat_level=ThreatLevel.INFO if success else ThreatLevel.LOW,
            actor_id=user_id,
            tenant_id=tenant_id,
            source_ip=ip_address,
            user_agent=user_agent,
            details={
                "success": success,
                "failure_reason": failure_reason,
            },
        )

        await self.record_event(event)

        alert = None

        if not success:
            # Track failed login
            key = f"{user_id}:{ip_address or 'unknown'}"
            self._failed_logins[key].append(now)

            # Clean old entries
            cutoff = now - timedelta(seconds=self.config.failed_login_window_seconds)
            self._failed_logins[key] = [
                t for t in self._failed_logins[key] if t > cutoff
            ]

            # Check threshold
            if len(self._failed_logins[key]) >= self.config.failed_login_threshold:
                # Check if already locked
                if key not in self._locked_accounts:
                    # Lock account
                    self._locked_accounts[key] = now + timedelta(
                        seconds=self.config.lockout_duration_seconds
                    )

                    # Generate alert
                    alert = await self._create_alert(
                        category=EventCategory.BRUTE_FORCE,
                        threat_level=ThreatLevel.HIGH,
                        title=f"Brute Force Attack Detected: {user_id}",
                        description=(
                            f"Multiple failed login attempts detected for user {user_id} "
                            f"from IP {ip_address}. Account has been locked for "
                            f"{self.config.lockout_duration_seconds} seconds."
                        ),
                        tenant_id=tenant_id,
                        affected_users=[user_id],
                        source_events=[event.id],
                        indicators={
                            "failed_attempts": len(self._failed_logins[key]),
                            "source_ip": ip_address,
                            "lockout_until": self._locked_accounts[key].isoformat(),
                        },
                    )

                    logger.warning(
                        f"Account locked due to failed login attempts: {user_id}",
                        extra={
                            "user_id": user_id,
                            "ip_address": ip_address,
                            "failed_attempts": len(self._failed_logins[key]),
                        },
                    )

        else:
            # Clear failed login history on success
            key = f"{user_id}:{ip_address or 'unknown'}"
            self._failed_logins.pop(key, None)
            self._locked_accounts.pop(key, None)

        return event, alert

    async def is_account_locked(
        self,
        user_id: str,
        ip_address: Optional[str] = None,
    ) -> bool:
        """
        Check if an account is locked due to failed login attempts.

        Args:
            user_id: User to check
            ip_address: IP address context

        Returns:
            True if account is locked
        """
        key = f"{user_id}:{ip_address or 'unknown'}"

        if key not in self._locked_accounts:
            return False

        lockout_until = self._locked_accounts[key]

        if datetime.now(timezone.utc) >= lockout_until:
            # Lockout expired
            del self._locked_accounts[key]
            return False

        return True

    async def record_data_access(
        self,
        user_id: str,
        tenant_id: str,
        resource_type: str,
        resource_count: int = 1,
        ip_address: Optional[str] = None,
    ) -> Tuple[SecurityEvent, Optional[SecurityAlert]]:
        """
        Record data access for anomaly detection.

        Args:
            user_id: User accessing data
            tenant_id: Tenant context
            resource_type: Type of resource accessed
            resource_count: Number of resources accessed
            ip_address: Client IP address

        Returns:
            Tuple of (event, alert if anomaly detected)
        """
        now = datetime.now(timezone.utc)

        event = SecurityEvent(
            category=EventCategory.DATA_ACCESS,
            event_type="DATA_ACCESS",
            description=f"User {user_id} accessed {resource_count} {resource_type} resources",
            threat_level=ThreatLevel.INFO,
            actor_id=user_id,
            tenant_id=tenant_id,
            source_ip=ip_address,
            target_resource=resource_type,
            details={
                "resource_count": resource_count,
            },
        )

        await self.record_event(event)

        # Update access statistics
        stats_key = f"{user_id}:{resource_type}"
        self._data_access_stats[tenant_id][stats_key].append(resource_count)

        # Keep only recent data
        max_samples = self.config.data_access_baseline_days * 24  # Hourly samples
        if len(self._data_access_stats[tenant_id][stats_key]) > max_samples:
            self._data_access_stats[tenant_id][stats_key] = \
                self._data_access_stats[tenant_id][stats_key][-max_samples:]

        # Check for anomaly
        alert = await self._check_data_access_anomaly(
            user_id=user_id,
            tenant_id=tenant_id,
            resource_type=resource_type,
            resource_count=resource_count,
            event=event,
        )

        return event, alert

    async def _check_data_access_anomaly(
        self,
        user_id: str,
        tenant_id: str,
        resource_type: str,
        resource_count: int,
        event: SecurityEvent,
    ) -> Optional[SecurityAlert]:
        """Check if data access is anomalous."""
        stats_key = f"{user_id}:{resource_type}"
        samples = self._data_access_stats[tenant_id][stats_key]

        # Need enough samples for baseline
        if len(samples) < 10:
            return None

        # Calculate statistics
        mean = statistics.mean(samples[:-1])  # Exclude current
        if len(samples) > 2:
            stdev = statistics.stdev(samples[:-1])
        else:
            stdev = mean * 0.5  # Fallback

        # Check deviation
        if stdev > 0:
            z_score = (resource_count - mean) / stdev
        else:
            z_score = 0

        if abs(z_score) > self.config.data_access_deviation_threshold:
            # Anomaly detected
            threat_level = ThreatLevel.MEDIUM
            if resource_count > mean * 10:
                threat_level = ThreatLevel.HIGH
            if resource_count > mean * 100:
                threat_level = ThreatLevel.CRITICAL

            alert = await self._create_alert(
                category=EventCategory.ANOMALY,
                threat_level=threat_level,
                title=f"Anomalous Data Access: {user_id}",
                description=(
                    f"User {user_id} accessed {resource_count} {resource_type} resources, "
                    f"which is {abs(z_score):.1f} standard deviations from normal "
                    f"(baseline: {mean:.1f} +/- {stdev:.1f})."
                ),
                tenant_id=tenant_id,
                affected_users=[user_id],
                affected_resources=[resource_type],
                source_events=[event.id],
                indicators={
                    "resource_count": resource_count,
                    "baseline_mean": mean,
                    "baseline_stdev": stdev,
                    "z_score": z_score,
                },
            )

            logger.warning(
                f"Anomalous data access detected",
                extra={
                    "user_id": user_id,
                    "resource_type": resource_type,
                    "resource_count": resource_count,
                    "z_score": z_score,
                },
            )

            return alert

        return None

    async def detect_anomalies(
        self,
        tenant_id: str,
        time_window_hours: int = 24,
    ) -> List[SecurityAlert]:
        """
        Run anomaly detection for a tenant.

        Args:
            tenant_id: Tenant to analyze
            time_window_hours: Time window for analysis

        Returns:
            List of alerts for detected anomalies
        """
        alerts = []
        cutoff = datetime.now(timezone.utc) - timedelta(hours=time_window_hours)

        # Get recent events for tenant
        tenant_events = [
            e for e in self._events
            if e.tenant_id == tenant_id and e.timestamp > cutoff
        ]

        # Apply each anomaly rule
        for rule in self._anomaly_rules.values():
            if not rule.enabled:
                continue

            # Check cooldown
            if rule.id in self._rule_last_fired:
                cooldown_until = self._rule_last_fired[rule.id] + timedelta(seconds=rule.cooldown_seconds)
                if datetime.now(timezone.utc) < cooldown_until:
                    continue

            # Apply rule
            alert = await self._apply_anomaly_rule(rule, tenant_id, tenant_events)
            if alert:
                alerts.append(alert)
                self._rule_last_fired[rule.id] = datetime.now(timezone.utc)

        return alerts

    async def _apply_anomaly_rule(
        self,
        rule: AnomalyRule,
        tenant_id: str,
        events: List[SecurityEvent],
    ) -> Optional[SecurityAlert]:
        """Apply an anomaly detection rule to events."""
        # Filter events by category and time window
        now = datetime.now(timezone.utc)
        window_start = now - timedelta(seconds=rule.window_seconds)

        relevant_events = [
            e for e in events
            if e.category == rule.category and e.timestamp >= window_start
        ]

        if not relevant_events:
            return None

        # Calculate metric
        if rule.aggregation == "count":
            value = len(relevant_events)
        elif rule.aggregation == "sum":
            value = sum(e.details.get(rule.metric, 0) for e in relevant_events)
        elif rule.aggregation == "avg":
            values = [e.details.get(rule.metric, 0) for e in relevant_events]
            value = statistics.mean(values) if values else 0
        elif rule.aggregation == "max":
            value = max(e.details.get(rule.metric, 0) for e in relevant_events)
        elif rule.aggregation == "min":
            value = min(e.details.get(rule.metric, 0) for e in relevant_events)
        else:
            value = len(relevant_events)

        # Compare against threshold
        triggered = False
        if rule.comparison == "gt" and value > rule.threshold:
            triggered = True
        elif rule.comparison == "lt" and value < rule.threshold:
            triggered = True
        elif rule.comparison == "gte" and value >= rule.threshold:
            triggered = True
        elif rule.comparison == "lte" and value <= rule.threshold:
            triggered = True
        elif rule.comparison == "eq" and value == rule.threshold:
            triggered = True

        if triggered:
            return await self._create_alert(
                category=rule.category,
                threat_level=rule.threat_level,
                title=rule.alert_title_template.format(value=value, threshold=rule.threshold),
                description=rule.alert_description_template.format(
                    value=value,
                    threshold=rule.threshold,
                    window=rule.window_seconds,
                ),
                tenant_id=tenant_id,
                source_events=[e.id for e in relevant_events[:10]],
                indicators={
                    "rule_id": rule.id,
                    "metric": rule.metric,
                    "value": value,
                    "threshold": rule.threshold,
                    "window_seconds": rule.window_seconds,
                },
            )

        return None

    async def _create_alert(
        self,
        category: EventCategory,
        threat_level: ThreatLevel,
        title: str,
        description: str,
        tenant_id: Optional[str] = None,
        affected_users: Optional[List[str]] = None,
        affected_resources: Optional[List[str]] = None,
        source_events: Optional[List[str]] = None,
        indicators: Optional[Dict[str, Any]] = None,
    ) -> SecurityAlert:
        """Create and store a security alert."""
        alert = SecurityAlert(
            category=category,
            threat_level=threat_level,
            title=title,
            description=description,
            tenant_id=tenant_id,
            affected_users=affected_users or [],
            affected_resources=affected_resources or [],
            source_events=source_events or [],
            indicators=indicators or {},
        )

        self._alerts[alert.id] = alert

        # Auto-escalate critical alerts
        if self.config.auto_escalate_critical and threat_level == ThreatLevel.CRITICAL:
            alert.status = AlertStatus.ESCALATED
            # Would trigger external notification here

        logger.warning(
            f"Security alert created: {title}",
            extra={
                "alert_id": alert.id,
                "category": category.value,
                "threat_level": threat_level.value,
            },
        )

        return alert

    async def get_active_alerts(
        self,
        tenant_id: Optional[str] = None,
        threat_level: Optional[ThreatLevel] = None,
        category: Optional[EventCategory] = None,
    ) -> List[SecurityAlert]:
        """
        Get active security alerts.

        Args:
            tenant_id: Filter by tenant
            threat_level: Filter by minimum threat level
            category: Filter by category

        Returns:
            List of matching alerts
        """
        threat_order = {
            ThreatLevel.INFO: 0,
            ThreatLevel.LOW: 1,
            ThreatLevel.MEDIUM: 2,
            ThreatLevel.HIGH: 3,
            ThreatLevel.CRITICAL: 4,
        }

        results = []
        for alert in self._alerts.values():
            # Filter by status
            if alert.status in (AlertStatus.RESOLVED, AlertStatus.FALSE_POSITIVE):
                continue

            # Filter by tenant
            if tenant_id and alert.tenant_id != tenant_id:
                continue

            # Filter by threat level
            if threat_level:
                if threat_order[alert.threat_level] < threat_order[threat_level]:
                    continue

            # Filter by category
            if category and alert.category != category:
                continue

            results.append(alert)

        # Sort by threat level (highest first) then by time
        results.sort(
            key=lambda a: (-threat_order[a.threat_level], a.created_at),
            reverse=True,
        )

        return results

    async def update_alert_status(
        self,
        alert_id: str,
        status: AlertStatus,
        assigned_to: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> Optional[SecurityAlert]:
        """
        Update the status of an alert.

        Args:
            alert_id: Alert to update
            status: New status
            assigned_to: User assigned to the alert
            notes: Status notes

        Returns:
            Updated alert or None if not found
        """
        alert = self._alerts.get(alert_id)
        if not alert:
            return None

        alert.status = status
        alert.updated_at = datetime.now(timezone.utc)

        if assigned_to:
            alert.assigned_to = assigned_to

        if status == AlertStatus.RESOLVED:
            alert.resolved_at = datetime.now(timezone.utc)
            alert.resolution_notes = notes

        logger.info(
            f"Alert status updated: {alert_id} -> {status.value}",
            extra={"alert_id": alert_id, "status": status.value},
        )

        return alert

    async def get_security_metrics(
        self,
        tenant_id: Optional[str] = None,
        time_window_hours: int = 24,
    ) -> Dict[str, Any]:
        """
        Get security metrics for monitoring dashboards.

        Args:
            tenant_id: Filter by tenant
            time_window_hours: Time window for metrics

        Returns:
            Security metrics dictionary
        """
        cutoff = datetime.now(timezone.utc) - timedelta(hours=time_window_hours)

        # Filter events
        events = [
            e for e in self._events
            if e.timestamp > cutoff and (not tenant_id or e.tenant_id == tenant_id)
        ]

        # Count by category
        events_by_category = defaultdict(int)
        for e in events:
            events_by_category[e.category.value] += 1

        # Count by threat level
        events_by_threat = defaultdict(int)
        for e in events:
            events_by_threat[e.threat_level.value] += 1

        # Active alerts summary
        active_alerts = await self.get_active_alerts(tenant_id)
        alerts_by_level = defaultdict(int)
        for a in active_alerts:
            alerts_by_level[a.threat_level.value] += 1

        # Failed logins
        failed_login_count = sum(
            len(attempts) for key, attempts in self._failed_logins.items()
            if any(t > cutoff for t in attempts)
        )

        return {
            "time_window_hours": time_window_hours,
            "tenant_id": tenant_id,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "events": {
                "total": len(events),
                "by_category": dict(events_by_category),
                "by_threat_level": dict(events_by_threat),
            },
            "alerts": {
                "active_total": len(active_alerts),
                "by_threat_level": dict(alerts_by_level),
                "critical": alerts_by_level.get("CRITICAL", 0),
                "high": alerts_by_level.get("HIGH", 0),
            },
            "authentication": {
                "failed_attempts": failed_login_count,
                "locked_accounts": len(self._locked_accounts),
            },
        }

    def _load_default_rules(self) -> None:
        """Load default anomaly detection rules."""
        default_rules = [
            AnomalyRule(
                id="rule-high-auth-failures",
                name="High Authentication Failures",
                description="Detect unusually high rate of authentication failures",
                category=EventCategory.AUTHENTICATION,
                metric="count",
                aggregation="count",
                window_seconds=300,
                threshold=50,
                comparison="gt",
                threat_level=ThreatLevel.HIGH,
                alert_title_template="High Authentication Failure Rate: {value} failures",
                alert_description_template=(
                    "Detected {value} authentication failures in the last {window} seconds, "
                    "exceeding threshold of {threshold}."
                ),
            ),
            AnomalyRule(
                id="rule-mass-data-access",
                name="Mass Data Access",
                description="Detect potential data exfiltration via mass data access",
                category=EventCategory.DATA_ACCESS,
                metric="resource_count",
                aggregation="sum",
                window_seconds=600,
                threshold=1000,
                comparison="gt",
                threat_level=ThreatLevel.CRITICAL,
                alert_title_template="Potential Data Exfiltration: {value} resources accessed",
                alert_description_template=(
                    "A total of {value} resources were accessed in the last {window} seconds, "
                    "which may indicate data exfiltration. Threshold: {threshold}."
                ),
            ),
            AnomalyRule(
                id="rule-privilege-escalation",
                name="Privilege Escalation Attempts",
                description="Detect attempts to access privileged resources",
                category=EventCategory.AUTHORIZATION,
                metric="count",
                aggregation="count",
                window_seconds=300,
                threshold=10,
                comparison="gt",
                threat_level=ThreatLevel.HIGH,
                alert_title_template="Privilege Escalation Attempts: {value} denials",
                alert_description_template=(
                    "Detected {value} authorization denials in the last {window} seconds, "
                    "which may indicate privilege escalation attempts."
                ),
            ),
        ]

        for rule in default_rules:
            self._anomaly_rules[rule.id] = rule

    async def _process_events_loop(self) -> None:
        """Background task to process events."""
        while True:
            try:
                await asyncio.sleep(60)
                await self._cleanup_old_data()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Event processing error: {e}", exc_info=True)

    async def _cleanup_old_data(self) -> None:
        """Clean up old events and resolved alerts."""
        now = datetime.now(timezone.utc)

        # Clean old events
        event_cutoff = now - timedelta(days=self.config.event_retention_days)
        self._events = [e for e in self._events if e.timestamp > event_cutoff]

        # Clean old resolved alerts
        alert_cutoff = now - timedelta(days=self.config.alert_retention_days)
        for alert_id in list(self._alerts.keys()):
            alert = self._alerts[alert_id]
            if alert.status == AlertStatus.RESOLVED and alert.resolved_at:
                if alert.resolved_at < alert_cutoff:
                    del self._alerts[alert_id]

        # Clean expired lockouts
        for key in list(self._locked_accounts.keys()):
            if now >= self._locked_accounts[key]:
                del self._locked_accounts[key]

    async def _siem_forward_loop(self) -> None:
        """Background task to forward events to SIEM."""
        while True:
            try:
                await asyncio.sleep(self.config.siem.flush_interval_seconds)
                await self._flush_to_siem()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"SIEM forwarding error: {e}", exc_info=True)

    async def _flush_to_siem(self) -> None:
        """Flush event buffer to SIEM."""
        if not self._siem_buffer:
            return

        events_to_send = self._siem_buffer[:self.config.siem.batch_size]
        self._siem_buffer = self._siem_buffer[self.config.siem.batch_size:]

        try:
            # Format events for SIEM
            formatted = [self._format_for_siem(e) for e in events_to_send]

            # Send to SIEM (placeholder - would use actual SIEM client)
            # await self._siem_client.send(formatted)

            logger.debug(f"Forwarded {len(formatted)} events to SIEM")

        except Exception as e:
            logger.error(f"Failed to forward events to SIEM: {e}")
            # Re-add to buffer for retry
            self._siem_buffer = events_to_send + self._siem_buffer

    def _format_for_siem(self, event: SecurityEvent) -> Dict[str, Any]:
        """Format a security event for SIEM ingestion."""
        return {
            "timestamp": event.timestamp.isoformat(),
            "event_id": event.id,
            "category": event.category.value,
            "threat_level": event.threat_level.value,
            "event_type": event.event_type,
            "description": event.description,
            "actor_id": event.actor_id,
            "tenant_id": event.tenant_id,
            "source_ip": event.source_ip,
            "target_resource": event.target_resource,
            "details": event.details,
            "index": self.config.siem.index,
        }
