# -*- coding: utf-8 -*-
"""
GL-012 STEAMQUAL Steam Quality Controller Alert Management
===========================================================

Production-ready alerting system for the GL-012 STEAMQUAL
SteamQualityController agent. Provides comprehensive alert management
for steam quality issues, control system faults, and operational
anomalies.

Alert Types:
- STEAM_QUALITY_LOW: Steam dryness below threshold
- PRESSURE_DEVIATION: Steam pressure outside tolerance
- TEMPERATURE_DEVIATION: Steam temperature outside tolerance
- DESUPERHEATER_FAULT: Injection system failure
- VALVE_STUCK: Control valve not responding
- MOISTURE_HIGH: Excessive moisture content
- CONDENSATION_RISK: Condensation predicted in steam lines

Severity Levels:
- INFO: Informational alerts for awareness
- WARNING: Conditions requiring attention
- CRITICAL: Immediate action required

Features:
- Alert lifecycle management (raise, acknowledge, clear)
- Configurable thresholds per alert type
- Alert history tracking with retention
- Rate limiting to prevent alert storms
- Alert escalation support
- Webhook/callback notification support

Usage:
    >>> from monitoring.alerting import AlertManager, AlertType, Severity
    >>>
    >>> manager = AlertManager()
    >>>
    >>> # Raise an alert
    >>> alert = manager.raise_alert(
    ...     alert_type=AlertType.STEAM_QUALITY_LOW,
    ...     severity=Severity.WARNING,
    ...     details={"dryness": 0.85, "threshold": 0.92}
    ... )
    >>>
    >>> # Acknowledge alert
    >>> manager.acknowledge_alert(alert.alert_id, user="operator1")
    >>>
    >>> # Clear alert when resolved
    >>> manager.clear_alert(alert.alert_id)
    >>>
    >>> # Get all active alerts
    >>> active = manager.get_active_alerts()

Author: GreenLang Team
License: Proprietary
"""

import hashlib
import logging
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import (
    Dict, List, Optional, Any, Callable, Set
)
from uuid import uuid4

logger = logging.getLogger(__name__)


class Severity(Enum):
    """Alert severity levels following industry standards."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"

    def __str__(self) -> str:
        return self.value

    @property
    def priority(self) -> int:
        """Get numeric priority (higher = more severe)."""
        priorities = {
            Severity.INFO: 1,
            Severity.WARNING: 2,
            Severity.CRITICAL: 3,
        }
        return priorities[self]

    def __lt__(self, other: "Severity") -> bool:
        return self.priority < other.priority

    def __gt__(self, other: "Severity") -> bool:
        return self.priority > other.priority


class AlertType(Enum):
    """Types of alerts for steam quality control."""

    # Steam Quality Alerts
    STEAM_QUALITY_LOW = "steam_quality_low"
    MOISTURE_HIGH = "moisture_high"
    CONDENSATION_RISK = "condensation_risk"

    # Process Parameter Alerts
    PRESSURE_DEVIATION = "pressure_deviation"
    TEMPERATURE_DEVIATION = "temperature_deviation"

    # Equipment Alerts
    DESUPERHEATER_FAULT = "desuperheater_fault"
    VALVE_STUCK = "valve_stuck"

    # System Alerts
    SENSOR_FAILURE = "sensor_failure"
    COMMUNICATION_LOSS = "communication_loss"
    CALCULATION_ERROR = "calculation_error"

    def __str__(self) -> str:
        return self.value

    @property
    def default_severity(self) -> Severity:
        """Get default severity for this alert type."""
        critical_types = {
            AlertType.DESUPERHEATER_FAULT,
            AlertType.VALVE_STUCK,
            AlertType.SENSOR_FAILURE,
        }
        warning_types = {
            AlertType.STEAM_QUALITY_LOW,
            AlertType.PRESSURE_DEVIATION,
            AlertType.TEMPERATURE_DEVIATION,
            AlertType.MOISTURE_HIGH,
            AlertType.CONDENSATION_RISK,
        }

        if self in critical_types:
            return Severity.CRITICAL
        elif self in warning_types:
            return Severity.WARNING
        else:
            return Severity.INFO


class AlertState(Enum):
    """Alert lifecycle states."""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"

    def __str__(self) -> str:
        return self.value


@dataclass
class Alert:
    """
    Represents a single alert instance.

    Attributes:
        alert_id: Unique identifier for this alert instance
        alert_type: Type of alert (from AlertType enum)
        severity: Alert severity level
        state: Current alert state
        message: Human-readable alert message
        details: Additional diagnostic information
        source: Component that raised the alert
        raised_at: Timestamp when alert was raised
        acknowledged_at: Timestamp when acknowledged (if applicable)
        acknowledged_by: User who acknowledged (if applicable)
        resolved_at: Timestamp when resolved (if applicable)
        resolved_by: User/system that resolved (if applicable)
        occurrence_count: Number of times this alert has occurred
        fingerprint: Hash for deduplication
    """
    alert_id: str
    alert_type: AlertType
    severity: Severity
    state: AlertState = AlertState.ACTIVE
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    source: str = "GL-012-STEAMQUAL"
    raised_at: float = field(default_factory=time.time)
    acknowledged_at: Optional[float] = None
    acknowledged_by: Optional[str] = None
    resolved_at: Optional[float] = None
    resolved_by: Optional[str] = None
    occurrence_count: int = 1
    fingerprint: str = ""
    tags: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.fingerprint:
            self.fingerprint = self._calculate_fingerprint()
        if not self.message:
            self.message = self._default_message()

    def _calculate_fingerprint(self) -> str:
        """Calculate fingerprint for deduplication."""
        fp_data = f"{self.alert_type.value}:{self.source}"
        return hashlib.sha256(fp_data.encode()).hexdigest()[:16]

    def _default_message(self) -> str:
        """Generate default message based on alert type."""
        messages = {
            AlertType.STEAM_QUALITY_LOW: "Steam dryness fraction below threshold",
            AlertType.PRESSURE_DEVIATION: "Steam pressure outside acceptable range",
            AlertType.TEMPERATURE_DEVIATION: "Steam temperature outside acceptable range",
            AlertType.DESUPERHEATER_FAULT: "Desuperheater injection system fault detected",
            AlertType.VALVE_STUCK: "Control valve not responding to commands",
            AlertType.MOISTURE_HIGH: "Excessive moisture content in steam",
            AlertType.CONDENSATION_RISK: "Condensation risk predicted in steam lines",
            AlertType.SENSOR_FAILURE: "Steam quality sensor failure detected",
            AlertType.COMMUNICATION_LOSS: "Communication loss with control system",
            AlertType.CALCULATION_ERROR: "Steam quality calculation error occurred",
        }
        return messages.get(self.alert_type, f"Alert: {self.alert_type.value}")

    @property
    def is_active(self) -> bool:
        """Check if alert is currently active."""
        return self.state == AlertState.ACTIVE

    @property
    def is_acknowledged(self) -> bool:
        """Check if alert has been acknowledged."""
        return self.state == AlertState.ACKNOWLEDGED

    @property
    def is_resolved(self) -> bool:
        """Check if alert has been resolved."""
        return self.state == AlertState.RESOLVED

    @property
    def duration_seconds(self) -> float:
        """Get alert duration in seconds."""
        end_time = self.resolved_at or time.time()
        return end_time - self.raised_at

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "alert_id": self.alert_id,
            "alert_type": self.alert_type.value,
            "severity": self.severity.value,
            "state": self.state.value,
            "message": self.message,
            "details": self.details,
            "source": self.source,
            "raised_at": self.raised_at,
            "raised_at_iso": datetime.fromtimestamp(
                self.raised_at, tz=timezone.utc
            ).isoformat(),
            "acknowledged_at": self.acknowledged_at,
            "acknowledged_by": self.acknowledged_by,
            "resolved_at": self.resolved_at,
            "resolved_by": self.resolved_by,
            "occurrence_count": self.occurrence_count,
            "fingerprint": self.fingerprint,
            "tags": self.tags,
            "is_active": self.is_active,
            "duration_seconds": self.duration_seconds,
        }


@dataclass
class AlertThreshold:
    """Configuration for alert thresholds."""
    alert_type: AlertType
    warning_threshold: Optional[float] = None
    critical_threshold: Optional[float] = None
    comparison: str = "less_than"  # less_than, greater_than, outside_range
    range_min: Optional[float] = None
    range_max: Optional[float] = None
    enabled: bool = True
    cooldown_seconds: int = 300  # Minimum time between repeat alerts

    def evaluate(self, value: float) -> Optional[Severity]:
        """
        Evaluate a value against thresholds.

        Args:
            value: Value to evaluate

        Returns:
            Severity if threshold exceeded, None otherwise
        """
        if not self.enabled:
            return None

        if self.comparison == "less_than":
            if self.critical_threshold and value < self.critical_threshold:
                return Severity.CRITICAL
            if self.warning_threshold and value < self.warning_threshold:
                return Severity.WARNING
        elif self.comparison == "greater_than":
            if self.critical_threshold and value > self.critical_threshold:
                return Severity.CRITICAL
            if self.warning_threshold and value > self.warning_threshold:
                return Severity.WARNING
        elif self.comparison == "outside_range":
            if self.range_min is not None and self.range_max is not None:
                if value < self.range_min or value > self.range_max:
                    # Check how far outside range
                    deviation = min(
                        abs(value - self.range_min) if value < self.range_min else 0,
                        abs(value - self.range_max) if value > self.range_max else 0
                    )
                    if self.critical_threshold and deviation > self.critical_threshold:
                        return Severity.CRITICAL
                    return Severity.WARNING

        return None


# Type alias for notification callbacks
NotificationCallback = Callable[[Alert], None]


class AlertManager:
    """
    Main alert management system for GL-012 STEAMQUAL agent.

    Provides comprehensive alert lifecycle management including
    raising, acknowledging, clearing, and tracking alerts. Supports
    configurable thresholds, rate limiting, and notification callbacks.

    Attributes:
        active_alerts: Dictionary of currently active alerts
        alert_history: List of all historical alerts
        thresholds: Configured alert thresholds
        notification_callbacks: Registered notification handlers

    Example:
        >>> manager = AlertManager()
        >>>
        >>> # Configure thresholds
        >>> manager.configure_thresholds({
        ...     AlertType.STEAM_QUALITY_LOW: AlertThreshold(
        ...         alert_type=AlertType.STEAM_QUALITY_LOW,
        ...         warning_threshold=0.92,
        ...         critical_threshold=0.85
        ...     )
        ... })
        >>>
        >>> # Raise alert
        >>> alert = manager.raise_alert(
        ...     AlertType.STEAM_QUALITY_LOW,
        ...     Severity.WARNING,
        ...     {"dryness": 0.89}
        ... )
    """

    # Default thresholds for steam quality alerts
    DEFAULT_THRESHOLDS = {
        AlertType.STEAM_QUALITY_LOW: AlertThreshold(
            alert_type=AlertType.STEAM_QUALITY_LOW,
            warning_threshold=0.92,
            critical_threshold=0.85,
            comparison="less_than",
        ),
        AlertType.MOISTURE_HIGH: AlertThreshold(
            alert_type=AlertType.MOISTURE_HIGH,
            warning_threshold=0.08,
            critical_threshold=0.15,
            comparison="greater_than",
        ),
        AlertType.PRESSURE_DEVIATION: AlertThreshold(
            alert_type=AlertType.PRESSURE_DEVIATION,
            range_min=8.0,
            range_max=12.0,
            warning_threshold=0.5,
            critical_threshold=1.0,
            comparison="outside_range",
        ),
        AlertType.TEMPERATURE_DEVIATION: AlertThreshold(
            alert_type=AlertType.TEMPERATURE_DEVIATION,
            range_min=170.0,
            range_max=190.0,
            warning_threshold=5.0,
            critical_threshold=10.0,
            comparison="outside_range",
        ),
    }

    def __init__(
        self,
        max_history_size: int = 10000,
        rate_limit_per_minute: int = 100,
        default_cooldown_seconds: int = 300,
    ):
        """
        Initialize alert manager.

        Args:
            max_history_size: Maximum number of alerts to keep in history
            rate_limit_per_minute: Maximum alerts per minute (prevents storms)
            default_cooldown_seconds: Default cooldown between repeat alerts
        """
        self.max_history_size = max_history_size
        self.rate_limit_per_minute = rate_limit_per_minute
        self.default_cooldown_seconds = default_cooldown_seconds

        # Alert storage
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []

        # Fingerprint -> last alert time for deduplication
        self._last_alert_time: Dict[str, float] = {}

        # Rate limiting
        self._alert_timestamps: List[float] = []

        # Thresholds
        self.thresholds: Dict[AlertType, AlertThreshold] = dict(
            self.DEFAULT_THRESHOLDS
        )

        # Notification callbacks
        self.notification_callbacks: List[NotificationCallback] = []

        # Suppressed alert types
        self._suppressed_types: Set[AlertType] = set()

        # Thread safety
        self.lock = threading.Lock()

        # Statistics
        self._stats = {
            "total_raised": 0,
            "total_acknowledged": 0,
            "total_resolved": 0,
            "total_suppressed": 0,
            "total_deduplicated": 0,
            "total_rate_limited": 0,
        }

        logger.info(
            f"AlertManager initialized (max_history={max_history_size}, "
            f"rate_limit={rate_limit_per_minute}/min)"
        )

    def raise_alert(
        self,
        alert_type: AlertType,
        severity: Severity,
        details: Optional[Dict[str, Any]] = None,
        message: Optional[str] = None,
        source: str = "GL-012-STEAMQUAL",
        tags: Optional[List[str]] = None,
    ) -> Optional[Alert]:
        """
        Raise a new alert.

        Args:
            alert_type: Type of alert to raise
            severity: Alert severity level
            details: Additional diagnostic information
            message: Custom alert message (uses default if not provided)
            source: Component raising the alert
            tags: Optional tags for categorization

        Returns:
            Alert object if raised, None if deduplicated/rate-limited
        """
        with self.lock:
            # Check if alert type is suppressed
            if alert_type in self._suppressed_types:
                self._stats["total_suppressed"] += 1
                logger.debug(f"Alert suppressed: {alert_type.value}")
                return None

            # Check rate limit
            if not self._check_rate_limit():
                self._stats["total_rate_limited"] += 1
                logger.warning(
                    f"Alert rate limited: {alert_type.value} "
                    f"(limit: {self.rate_limit_per_minute}/min)"
                )
                return None

            # Create alert
            alert = Alert(
                alert_id=str(uuid4()),
                alert_type=alert_type,
                severity=severity,
                message=message or "",
                details=details or {},
                source=source,
                tags=tags or [],
            )

            # Check for deduplication
            threshold = self.thresholds.get(alert_type)
            cooldown = (
                threshold.cooldown_seconds if threshold
                else self.default_cooldown_seconds
            )

            last_time = self._last_alert_time.get(alert.fingerprint, 0)
            if time.time() - last_time < cooldown:
                # Update existing alert instead
                existing = self._find_by_fingerprint(alert.fingerprint)
                if existing:
                    existing.occurrence_count += 1
                    existing.details.update(details or {})
                    self._stats["total_deduplicated"] += 1
                    logger.debug(
                        f"Alert deduplicated: {alert_type.value} "
                        f"(count: {existing.occurrence_count})"
                    )
                    return existing
                return None

            # Store alert
            self.active_alerts[alert.alert_id] = alert
            self.alert_history.append(alert)
            self._last_alert_time[alert.fingerprint] = time.time()
            self._alert_timestamps.append(time.time())
            self._stats["total_raised"] += 1

            # Trim history if needed
            self._trim_history()

            # Send notifications
            self._notify(alert)

            logger.info(
                f"Alert raised: {alert.alert_id} - "
                f"{alert_type.value} ({severity.value})"
            )

            return alert

    def clear_alert(
        self,
        alert_id: str,
        resolved_by: str = "system",
    ) -> bool:
        """
        Clear/resolve an alert.

        Args:
            alert_id: ID of alert to clear
            resolved_by: User or system that resolved the alert

        Returns:
            True if alert was cleared, False if not found
        """
        with self.lock:
            if alert_id not in self.active_alerts:
                logger.warning(f"Alert not found for clearing: {alert_id}")
                return False

            alert = self.active_alerts[alert_id]
            alert.state = AlertState.RESOLVED
            alert.resolved_at = time.time()
            alert.resolved_by = resolved_by

            del self.active_alerts[alert_id]
            self._stats["total_resolved"] += 1

            logger.info(
                f"Alert cleared: {alert_id} by {resolved_by}"
            )

            return True

    def acknowledge_alert(
        self,
        alert_id: str,
        user: str,
        notes: Optional[str] = None,
    ) -> bool:
        """
        Acknowledge an alert.

        Args:
            alert_id: ID of alert to acknowledge
            user: User acknowledging the alert
            notes: Optional notes about acknowledgment

        Returns:
            True if acknowledged, False if not found or already acknowledged
        """
        with self.lock:
            if alert_id not in self.active_alerts:
                logger.warning(f"Alert not found for acknowledgment: {alert_id}")
                return False

            alert = self.active_alerts[alert_id]

            if alert.state == AlertState.ACKNOWLEDGED:
                logger.debug(f"Alert already acknowledged: {alert_id}")
                return False

            alert.state = AlertState.ACKNOWLEDGED
            alert.acknowledged_at = time.time()
            alert.acknowledged_by = user

            if notes:
                alert.details["acknowledgment_notes"] = notes

            self._stats["total_acknowledged"] += 1

            logger.info(
                f"Alert acknowledged: {alert_id} by {user}"
            )

            return True

    def get_active_alerts(
        self,
        severity: Optional[Severity] = None,
        alert_type: Optional[AlertType] = None,
        source: Optional[str] = None,
    ) -> List[Alert]:
        """
        Get all active alerts, optionally filtered.

        Args:
            severity: Filter by severity level
            alert_type: Filter by alert type
            source: Filter by source component

        Returns:
            List of active Alert objects
        """
        with self.lock:
            alerts = list(self.active_alerts.values())

        # Apply filters
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        if alert_type:
            alerts = [a for a in alerts if a.alert_type == alert_type]
        if source:
            alerts = [a for a in alerts if a.source == source]

        # Sort by severity (highest first) then by time (oldest first)
        alerts.sort(key=lambda a: (-a.severity.priority, a.raised_at))

        return alerts

    def get_alert_by_id(self, alert_id: str) -> Optional[Alert]:
        """
        Get a specific alert by ID.

        Args:
            alert_id: Alert identifier

        Returns:
            Alert object or None if not found
        """
        with self.lock:
            # Check active alerts first
            if alert_id in self.active_alerts:
                return self.active_alerts[alert_id]

            # Check history
            for alert in reversed(self.alert_history):
                if alert.alert_id == alert_id:
                    return alert

        return None

    def configure_thresholds(
        self,
        thresholds: Dict[AlertType, AlertThreshold]
    ) -> None:
        """
        Configure alert thresholds.

        Args:
            thresholds: Dictionary mapping AlertType to AlertThreshold
        """
        with self.lock:
            self.thresholds.update(thresholds)

        logger.info(
            f"Updated thresholds for {len(thresholds)} alert types"
        )

    def get_threshold(self, alert_type: AlertType) -> Optional[AlertThreshold]:
        """
        Get threshold configuration for an alert type.

        Args:
            alert_type: Type of alert

        Returns:
            AlertThreshold or None if not configured
        """
        return self.thresholds.get(alert_type)

    def suppress_alert_type(self, alert_type: AlertType) -> None:
        """
        Suppress an alert type (no new alerts will be raised).

        Args:
            alert_type: Type of alert to suppress
        """
        with self.lock:
            self._suppressed_types.add(alert_type)

        logger.info(f"Suppressed alert type: {alert_type.value}")

    def unsuppress_alert_type(self, alert_type: AlertType) -> None:
        """
        Remove suppression for an alert type.

        Args:
            alert_type: Type of alert to unsuppress
        """
        with self.lock:
            self._suppressed_types.discard(alert_type)

        logger.info(f"Unsuppressed alert type: {alert_type.value}")

    def register_notification_callback(
        self,
        callback: NotificationCallback
    ) -> None:
        """
        Register a callback for alert notifications.

        Args:
            callback: Function to call when alerts are raised
        """
        with self.lock:
            self.notification_callbacks.append(callback)

        logger.info("Registered notification callback")

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get alert statistics.

        Returns:
            Dictionary of alert statistics
        """
        with self.lock:
            active_by_severity = defaultdict(int)
            active_by_type = defaultdict(int)

            for alert in self.active_alerts.values():
                active_by_severity[alert.severity.value] += 1
                active_by_type[alert.alert_type.value] += 1

            return {
                "total_raised": self._stats["total_raised"],
                "total_acknowledged": self._stats["total_acknowledged"],
                "total_resolved": self._stats["total_resolved"],
                "total_suppressed": self._stats["total_suppressed"],
                "total_deduplicated": self._stats["total_deduplicated"],
                "total_rate_limited": self._stats["total_rate_limited"],
                "currently_active": len(self.active_alerts),
                "history_size": len(self.alert_history),
                "suppressed_types": [t.value for t in self._suppressed_types],
                "active_by_severity": dict(active_by_severity),
                "active_by_type": dict(active_by_type),
            }

    def clear_all_alerts(self, resolved_by: str = "system") -> int:
        """
        Clear all active alerts.

        Args:
            resolved_by: User or system clearing the alerts

        Returns:
            Number of alerts cleared
        """
        with self.lock:
            count = len(self.active_alerts)

            for alert in self.active_alerts.values():
                alert.state = AlertState.RESOLVED
                alert.resolved_at = time.time()
                alert.resolved_by = resolved_by

            self.active_alerts.clear()
            self._stats["total_resolved"] += count

        logger.info(f"Cleared all alerts ({count} total) by {resolved_by}")
        return count

    def get_alert_history(
        self,
        limit: int = 100,
        offset: int = 0,
        severity: Optional[Severity] = None,
        alert_type: Optional[AlertType] = None,
        since_timestamp: Optional[float] = None,
    ) -> List[Alert]:
        """
        Get alert history with optional filters.

        Args:
            limit: Maximum number of alerts to return
            offset: Number of alerts to skip
            severity: Filter by severity level
            alert_type: Filter by alert type
            since_timestamp: Only return alerts after this timestamp

        Returns:
            List of historical Alert objects
        """
        with self.lock:
            alerts = list(self.alert_history)

        # Apply filters
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        if alert_type:
            alerts = [a for a in alerts if a.alert_type == alert_type]
        if since_timestamp:
            alerts = [a for a in alerts if a.raised_at >= since_timestamp]

        # Sort by time (newest first)
        alerts.sort(key=lambda a: a.raised_at, reverse=True)

        # Apply pagination
        return alerts[offset:offset + limit]

    # =========================================================================
    # PRIVATE METHODS
    # =========================================================================

    def _check_rate_limit(self) -> bool:
        """Check if we're within rate limits."""
        now = time.time()
        minute_ago = now - 60

        # Remove old timestamps
        self._alert_timestamps = [
            t for t in self._alert_timestamps if t > minute_ago
        ]

        return len(self._alert_timestamps) < self.rate_limit_per_minute

    def _find_by_fingerprint(self, fingerprint: str) -> Optional[Alert]:
        """Find active alert by fingerprint."""
        for alert in self.active_alerts.values():
            if alert.fingerprint == fingerprint:
                return alert
        return None

    def _trim_history(self) -> None:
        """Trim alert history to max size."""
        if len(self.alert_history) > self.max_history_size:
            excess = len(self.alert_history) - self.max_history_size
            self.alert_history = self.alert_history[excess:]

    def _notify(self, alert: Alert) -> None:
        """Send notifications for an alert."""
        for callback in self.notification_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(
                    f"Notification callback failed: {e}",
                    exc_info=True
                )


# Module-level convenience functions

_default_manager: Optional[AlertManager] = None


def get_alert_manager() -> Optional[AlertManager]:
    """
    Get the default alert manager instance.

    Returns:
        Default AlertManager or None if not initialized
    """
    return _default_manager


def init_alert_manager(**kwargs) -> AlertManager:
    """
    Initialize and return the default alert manager.

    Args:
        **kwargs: Arguments for AlertManager constructor

    Returns:
        Initialized AlertManager instance
    """
    global _default_manager
    _default_manager = AlertManager(**kwargs)
    return _default_manager


__all__ = [
    "Severity",
    "AlertType",
    "AlertState",
    "Alert",
    "AlertThreshold",
    "NotificationCallback",
    "AlertManager",
    "get_alert_manager",
    "init_alert_manager",
]
