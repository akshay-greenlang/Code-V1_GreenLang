"""
GL-012 STEAMQUAL SteamQualityController - Alert Management

This module implements comprehensive alert management for steam quality monitoring,
including alert creation, acknowledgment, escalation, and notification routing
with alarm hysteresis and rate limiting to prevent alarm flooding.

Alert Types:
    - LOW_DRYNESS: Dryness fraction below threshold
    - HIGH_MOISTURE: Moisture carryover detected
    - CARRYOVER_RISK: Carryover risk above threshold
    - SEPARATOR_FLOODING: Separator drain issues
    - WATER_HAMMER_RISK: Condensate accumulation risk
    - DATA_QUALITY_DEGRADED: Sensor/data issues

Severity Levels:
    - S0 (INFO): Informational, no action required
    - S1 (ADVISORY): Advisory, awareness needed
    - S2 (WARNING): Warning, action recommended
    - S3 (CRITICAL): Critical, immediate action required

Features:
    - Alarm hysteresis to prevent alarm chatter
    - Rate limiting to prevent alarm flooding
    - Alert deduplication using fingerprinting
    - Multi-level escalation (L1-L4)
    - Alert acknowledgment with audit trail
    - Notification channel routing

Example:
    >>> manager = SteamQualityAlertManager()
    >>> alert = manager.evaluate_dryness(0.92, separator_id="SEP-001")
    >>> if alert:
    ...     print(f"Alert: {alert.message}")
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum, IntEnum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import asyncio
import hashlib
import logging
import threading
import uuid

logger = logging.getLogger(__name__)


# =============================================================================
# Enumerations
# =============================================================================

class QualityAlertType(Enum):
    """Steam quality alert types."""
    LOW_DRYNESS = "low_dryness"
    HIGH_MOISTURE = "high_moisture"
    CARRYOVER_RISK = "carryover_risk"
    SEPARATOR_FLOODING = "separator_flooding"
    WATER_HAMMER_RISK = "water_hammer_risk"
    DATA_QUALITY_DEGRADED = "data_quality_degraded"


class AlertSeverity(IntEnum):
    """
    Alert severity levels following industrial alarm standards.

    S0: Info - Informational, logged only
    S1: Advisory - Operator awareness needed
    S2: Warning - Action recommended within timeframe
    S3: Critical - Immediate action required
    """
    S0_INFO = 0
    S1_ADVISORY = 1
    S2_WARNING = 2
    S3_CRITICAL = 3

    @property
    def name_display(self) -> str:
        """Get display name for severity."""
        names = {
            AlertSeverity.S0_INFO: "INFO",
            AlertSeverity.S1_ADVISORY: "ADVISORY",
            AlertSeverity.S2_WARNING: "WARNING",
            AlertSeverity.S3_CRITICAL: "CRITICAL",
        }
        return names.get(self, "UNKNOWN")

    @property
    def response_time_minutes(self) -> int:
        """Get expected response time for severity level."""
        times = {
            AlertSeverity.S0_INFO: 0,  # No response required
            AlertSeverity.S1_ADVISORY: 60,
            AlertSeverity.S2_WARNING: 15,
            AlertSeverity.S3_CRITICAL: 5,
        }
        return times.get(self, 60)


class AlertState(Enum):
    """Alert lifecycle states."""
    PENDING = "pending"  # Created but not yet fired
    FIRING = "firing"  # Active alert
    ACKNOWLEDGED = "acknowledged"  # Acknowledged by operator
    RESOLVED = "resolved"  # Condition cleared
    SUPPRESSED = "suppressed"  # Suppressed by rule


class EscalationLevel(Enum):
    """Alert escalation levels."""
    L1_OPERATOR = "L1_operator"
    L2_SUPERVISOR = "L2_supervisor"
    L3_ENGINEER = "L3_engineer"
    L4_MANAGEMENT = "L4_management"

    @property
    def timeout_minutes(self) -> int:
        """Default escalation timeout for this level."""
        timeouts = {
            EscalationLevel.L1_OPERATOR: 10,
            EscalationLevel.L2_SUPERVISOR: 20,
            EscalationLevel.L3_ENGINEER: 45,
            EscalationLevel.L4_MANAGEMENT: 90,
        }
        return timeouts.get(self, 60)


# =============================================================================
# Hysteresis Configuration
# =============================================================================

@dataclass
class HysteresisConfig:
    """
    Hysteresis configuration for alarm thresholds.

    Hysteresis prevents alarm chatter by requiring the measured value
    to cross different thresholds for alarm activation vs. clearing.

    Example for dryness fraction:
        - Alarm activates when dryness < 0.95 (alarm_threshold)
        - Alarm clears when dryness > 0.97 (clear_threshold)
        - This 2% dead band prevents rapid on/off cycling
    """
    alarm_threshold: float  # Value that triggers the alarm
    clear_threshold: float  # Value that clears the alarm
    dead_time_seconds: float = 30.0  # Minimum time in state before transition
    confirmation_samples: int = 3  # Number of consecutive samples to confirm


@dataclass
class ThresholdConfig:
    """Complete threshold configuration for an alert type."""
    alert_type: QualityAlertType
    s1_hysteresis: Optional[HysteresisConfig] = None  # Advisory level
    s2_hysteresis: Optional[HysteresisConfig] = None  # Warning level
    s3_hysteresis: Optional[HysteresisConfig] = None  # Critical level
    rate_limit_per_hour: int = 10  # Max alerts per hour for this type
    cooldown_seconds: float = 60.0  # Minimum time between alerts


# Default threshold configurations
DEFAULT_THRESHOLDS: Dict[QualityAlertType, ThresholdConfig] = {
    QualityAlertType.LOW_DRYNESS: ThresholdConfig(
        alert_type=QualityAlertType.LOW_DRYNESS,
        s1_hysteresis=HysteresisConfig(
            alarm_threshold=0.96,  # Alarm when x < 0.96
            clear_threshold=0.97,  # Clear when x > 0.97
            dead_time_seconds=30.0,
        ),
        s2_hysteresis=HysteresisConfig(
            alarm_threshold=0.94,
            clear_threshold=0.96,
            dead_time_seconds=20.0,
        ),
        s3_hysteresis=HysteresisConfig(
            alarm_threshold=0.90,
            clear_threshold=0.93,
            dead_time_seconds=10.0,
        ),
        rate_limit_per_hour=20,
    ),
    QualityAlertType.HIGH_MOISTURE: ThresholdConfig(
        alert_type=QualityAlertType.HIGH_MOISTURE,
        s1_hysteresis=HysteresisConfig(
            alarm_threshold=0.04,  # Alarm when moisture > 4%
            clear_threshold=0.03,  # Clear when moisture < 3%
        ),
        s2_hysteresis=HysteresisConfig(
            alarm_threshold=0.06,
            clear_threshold=0.045,
        ),
        s3_hysteresis=HysteresisConfig(
            alarm_threshold=0.10,
            clear_threshold=0.07,
        ),
        rate_limit_per_hour=15,
    ),
    QualityAlertType.CARRYOVER_RISK: ThresholdConfig(
        alert_type=QualityAlertType.CARRYOVER_RISK,
        s1_hysteresis=HysteresisConfig(
            alarm_threshold=0.3,  # Alarm when risk > 0.3
            clear_threshold=0.2,  # Clear when risk < 0.2
        ),
        s2_hysteresis=HysteresisConfig(
            alarm_threshold=0.5,
            clear_threshold=0.35,
        ),
        s3_hysteresis=HysteresisConfig(
            alarm_threshold=0.7,
            clear_threshold=0.5,
        ),
        rate_limit_per_hour=10,
    ),
    QualityAlertType.SEPARATOR_FLOODING: ThresholdConfig(
        alert_type=QualityAlertType.SEPARATOR_FLOODING,
        s1_hysteresis=HysteresisConfig(
            alarm_threshold=70.0,  # Alarm when level > 70%
            clear_threshold=60.0,  # Clear when level < 60%
        ),
        s2_hysteresis=HysteresisConfig(
            alarm_threshold=85.0,
            clear_threshold=75.0,
        ),
        s3_hysteresis=HysteresisConfig(
            alarm_threshold=95.0,
            clear_threshold=85.0,
        ),
        rate_limit_per_hour=10,
    ),
    QualityAlertType.WATER_HAMMER_RISK: ThresholdConfig(
        alert_type=QualityAlertType.WATER_HAMMER_RISK,
        s1_hysteresis=HysteresisConfig(
            alarm_threshold=0.3,
            clear_threshold=0.2,
        ),
        s2_hysteresis=HysteresisConfig(
            alarm_threshold=0.5,
            clear_threshold=0.35,
        ),
        s3_hysteresis=HysteresisConfig(
            alarm_threshold=0.7,
            clear_threshold=0.5,
        ),
        rate_limit_per_hour=10,
    ),
    QualityAlertType.DATA_QUALITY_DEGRADED: ThresholdConfig(
        alert_type=QualityAlertType.DATA_QUALITY_DEGRADED,
        s1_hysteresis=HysteresisConfig(
            alarm_threshold=0.8,  # Alarm when quality < 80%
            clear_threshold=0.9,  # Clear when quality > 90%
        ),
        s2_hysteresis=HysteresisConfig(
            alarm_threshold=0.6,
            clear_threshold=0.75,
        ),
        s3_hysteresis=HysteresisConfig(
            alarm_threshold=0.4,
            clear_threshold=0.55,
        ),
        rate_limit_per_hour=5,
    ),
}


# =============================================================================
# Data Models
# =============================================================================

@dataclass
class AlertContext:
    """Contextual information for an alert."""
    # Location identifiers
    system_id: Optional[str] = None
    separator_id: Optional[str] = None
    site_id: Optional[str] = None

    # Measured values
    current_value: Optional[float] = None
    threshold_value: Optional[float] = None
    deviation_percent: Optional[float] = None

    # Steam quality context
    dryness_fraction: Optional[float] = None
    moisture_content: Optional[float] = None
    carryover_risk: Optional[float] = None
    separator_level_percent: Optional[float] = None
    separator_efficiency: Optional[float] = None

    # Operating conditions
    pressure_bar: Optional[float] = None
    temperature_c: Optional[float] = None
    steam_flow_kg_h: Optional[float] = None

    # Data quality
    data_quality_score: Optional[float] = None
    sensor_status: Optional[str] = None

    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "system_id": self.system_id,
            "separator_id": self.separator_id,
            "site_id": self.site_id,
            "current_value": self.current_value,
            "threshold_value": self.threshold_value,
            "deviation_percent": self.deviation_percent,
            "dryness_fraction": self.dryness_fraction,
            "moisture_content": self.moisture_content,
            "carryover_risk": self.carryover_risk,
            "separator_level_percent": self.separator_level_percent,
            "separator_efficiency": self.separator_efficiency,
            "pressure_bar": self.pressure_bar,
            "temperature_c": self.temperature_c,
            "steam_flow_kg_h": self.steam_flow_kg_h,
            "data_quality_score": self.data_quality_score,
            "sensor_status": self.sensor_status,
            "metadata": self.metadata,
        }


@dataclass
class Alert:
    """Alert instance representing a detected condition."""
    alert_id: str
    alert_type: QualityAlertType
    severity: AlertSeverity
    source: str  # Equipment/separator ID
    message: str
    state: AlertState = AlertState.PENDING
    context: AlertContext = field(default_factory=AlertContext)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    fired_at: Optional[datetime] = None
    acknowledged_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None
    acknowledgment_notes: Optional[str] = None
    resolved_at: Optional[datetime] = None
    escalation_level: EscalationLevel = EscalationLevel.L1_OPERATOR
    escalated_at: Optional[datetime] = None
    fingerprint: str = ""
    labels: Dict[str, str] = field(default_factory=dict)
    notification_count: int = 0

    def __post_init__(self) -> None:
        """Compute fingerprint after initialization."""
        if not self.fingerprint:
            self.fingerprint = self._compute_fingerprint()

    def _compute_fingerprint(self) -> str:
        """Compute unique fingerprint for alert deduplication."""
        data = (
            f"{self.alert_type.value}:"
            f"{self.source}:"
            f"{self.severity.value}:"
            f"{self.context.separator_id}:"
            f"{sorted(self.labels.items())}"
        )
        return hashlib.sha256(data.encode()).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "alert_id": self.alert_id,
            "alert_type": self.alert_type.value,
            "severity": self.severity.name_display,
            "severity_code": f"S{self.severity.value}",
            "source": self.source,
            "message": self.message,
            "state": self.state.value,
            "context": self.context.to_dict(),
            "created_at": self.created_at.isoformat(),
            "fired_at": self.fired_at.isoformat() if self.fired_at else None,
            "acknowledged_at": (
                self.acknowledged_at.isoformat() if self.acknowledged_at else None
            ),
            "acknowledged_by": self.acknowledged_by,
            "acknowledgment_notes": self.acknowledgment_notes,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "escalation_level": self.escalation_level.value,
            "escalated_at": (
                self.escalated_at.isoformat() if self.escalated_at else None
            ),
            "fingerprint": self.fingerprint,
            "labels": self.labels,
            "notification_count": self.notification_count,
        }


@dataclass
class AlertFilter:
    """Filter criteria for querying alerts."""
    alert_types: Optional[List[QualityAlertType]] = None
    severities: Optional[List[AlertSeverity]] = None
    states: Optional[List[AlertState]] = None
    source_pattern: Optional[str] = None
    separator_id: Optional[str] = None
    site_id: Optional[str] = None
    from_time: Optional[datetime] = None
    to_time: Optional[datetime] = None
    escalation_levels: Optional[List[EscalationLevel]] = None


@dataclass
class AcknowledgmentResult:
    """Result of an alert acknowledgment operation."""
    success: bool
    alert_id: str
    acknowledged_by: str
    acknowledged_at: datetime
    notes: Optional[str] = None
    error_message: Optional[str] = None


@dataclass
class EscalationResult:
    """Result of an alert escalation operation."""
    success: bool
    alert_id: str
    previous_level: EscalationLevel
    new_level: EscalationLevel
    escalated_at: datetime
    escalation_reason: Optional[str] = None
    error_message: Optional[str] = None


# =============================================================================
# Hysteresis State Tracking
# =============================================================================

@dataclass
class HysteresisState:
    """Tracks the hysteresis state for a single alarm point."""
    is_alarmed: bool = False
    last_transition_time: Optional[datetime] = None
    confirmation_count: int = 0
    last_value: Optional[float] = None
    severity: Optional[AlertSeverity] = None


# =============================================================================
# Notification Channels
# =============================================================================

class NotificationChannel:
    """Base class for notification channels."""

    async def send(self, alert: Alert) -> bool:
        """Send alert notification. Override in subclasses."""
        raise NotImplementedError("Subclasses must implement send()")


class LogNotificationChannel(NotificationChannel):
    """Log-based notification channel for development and debugging."""

    async def send(self, alert: Alert) -> bool:
        """Log alert to standard logging."""
        log_level_map = {
            AlertSeverity.S0_INFO: logging.INFO,
            AlertSeverity.S1_ADVISORY: logging.INFO,
            AlertSeverity.S2_WARNING: logging.WARNING,
            AlertSeverity.S3_CRITICAL: logging.ERROR,
        }
        log_level = log_level_map.get(alert.severity, logging.WARNING)

        logger.log(
            log_level,
            "[ALERT] %s | S%d-%s | %s: %s (source=%s)",
            alert.alert_type.value.upper(),
            alert.severity.value,
            alert.severity.name_display,
            alert.alert_id[:8],
            alert.message,
            alert.source,
        )
        return True


# =============================================================================
# Main Alert Manager
# =============================================================================

class SteamQualityAlertManager:
    """
    Alert management system for steam quality monitoring.

    This class provides comprehensive alert lifecycle management including
    creation, acknowledgment, escalation, and notification routing with
    hysteresis and rate limiting capabilities to prevent alarm flooding.

    Attributes:
        namespace: Namespace for alert identification
        thresholds: Threshold configurations for each alert type

    Example:
        >>> manager = SteamQualityAlertManager(namespace="steamqual")
        >>> alert = manager.evaluate_dryness(0.92, separator_id="SEP-001")
        >>> if alert:
        ...     print(f"Alert: {alert.message}")
    """

    def __init__(
        self,
        namespace: str = "steamqual",
        on_alert_callback: Optional[Callable[[Alert], None]] = None,
        deduplication_window_s: float = 300.0,
        thresholds: Optional[Dict[QualityAlertType, ThresholdConfig]] = None,
    ) -> None:
        """
        Initialize SteamQualityAlertManager.

        Args:
            namespace: Namespace for alert identification
            on_alert_callback: Optional callback invoked on new alerts
            deduplication_window_s: Window for alert deduplication (seconds)
            thresholds: Custom threshold configurations
        """
        self.namespace = namespace
        self._on_alert_callback = on_alert_callback
        self._deduplication_window_s = deduplication_window_s
        self._thresholds = thresholds or DEFAULT_THRESHOLDS
        self._lock = threading.Lock()

        # Alert storage
        self._active_alerts: Dict[str, Alert] = {}  # alert_id -> Alert
        self._alert_by_fingerprint: Dict[str, str] = {}  # fingerprint -> alert_id
        self._alert_history: List[Alert] = []
        self._max_history_size = 10000

        # Hysteresis state: (separator_id, alert_type) -> HysteresisState
        self._hysteresis_states: Dict[Tuple[str, QualityAlertType], HysteresisState] = {}

        # Rate limiting: (separator_id, alert_type) -> list of timestamps
        self._rate_limit_buckets: Dict[Tuple[str, QualityAlertType], List[datetime]] = {}

        # Suppression rules
        self._suppressed_fingerprints: Dict[str, datetime] = {}  # fingerprint -> expires_at

        # Notification channels
        self._channels: List[NotificationChannel] = [LogNotificationChannel()]

        # Escalation rules by alert type and severity
        self._escalation_rules = self._default_escalation_rules()

        # Statistics
        self._stats = {
            "alerts_created": 0,
            "alerts_acknowledged": 0,
            "alerts_resolved": 0,
            "alerts_escalated": 0,
            "alerts_deduplicated": 0,
            "alerts_rate_limited": 0,
            "alerts_hysteresis_blocked": 0,
            "notifications_sent": 0,
        }

        # Background task control
        self._escalation_running = False

        logger.info(
            "SteamQualityAlertManager initialized: namespace=%s, dedup_window=%ss",
            namespace,
            deduplication_window_s,
        )

    def _default_escalation_rules(self) -> Dict[QualityAlertType, Dict[AlertSeverity, int]]:
        """Define default escalation timeout rules (minutes)."""
        return {
            QualityAlertType.LOW_DRYNESS: {
                AlertSeverity.S3_CRITICAL: 5,
                AlertSeverity.S2_WARNING: 15,
                AlertSeverity.S1_ADVISORY: 45,
                AlertSeverity.S0_INFO: 0,
            },
            QualityAlertType.HIGH_MOISTURE: {
                AlertSeverity.S3_CRITICAL: 5,
                AlertSeverity.S2_WARNING: 15,
                AlertSeverity.S1_ADVISORY: 45,
                AlertSeverity.S0_INFO: 0,
            },
            QualityAlertType.CARRYOVER_RISK: {
                AlertSeverity.S3_CRITICAL: 5,
                AlertSeverity.S2_WARNING: 10,
                AlertSeverity.S1_ADVISORY: 30,
                AlertSeverity.S0_INFO: 0,
            },
            QualityAlertType.SEPARATOR_FLOODING: {
                AlertSeverity.S3_CRITICAL: 3,
                AlertSeverity.S2_WARNING: 10,
                AlertSeverity.S1_ADVISORY: 30,
                AlertSeverity.S0_INFO: 0,
            },
            QualityAlertType.WATER_HAMMER_RISK: {
                AlertSeverity.S3_CRITICAL: 3,
                AlertSeverity.S2_WARNING: 10,
                AlertSeverity.S1_ADVISORY: 30,
                AlertSeverity.S0_INFO: 0,
            },
            QualityAlertType.DATA_QUALITY_DEGRADED: {
                AlertSeverity.S3_CRITICAL: 10,
                AlertSeverity.S2_WARNING: 30,
                AlertSeverity.S1_ADVISORY: 60,
                AlertSeverity.S0_INFO: 0,
            },
        }

    # =========================================================================
    # Hysteresis Evaluation
    # =========================================================================

    def _evaluate_hysteresis(
        self,
        separator_id: str,
        alert_type: QualityAlertType,
        current_value: float,
        high_is_alarm: bool = True,
    ) -> Optional[Tuple[bool, AlertSeverity]]:
        """
        Evaluate hysteresis for an alarm point.

        Args:
            separator_id: Separator identifier
            alert_type: Type of alert
            current_value: Current measured value
            high_is_alarm: True if high values trigger alarm, False for low

        Returns:
            Tuple of (should_transition, new_severity) or None if no transition
        """
        threshold_config = self._thresholds.get(alert_type)
        if not threshold_config:
            return None

        key = (separator_id, alert_type)
        now = datetime.now(timezone.utc)

        with self._lock:
            # Get or create hysteresis state
            if key not in self._hysteresis_states:
                self._hysteresis_states[key] = HysteresisState()

            state = self._hysteresis_states[key]

            # Determine current severity based on value
            new_severity: Optional[AlertSeverity] = None

            for severity, hysteresis in [
                (AlertSeverity.S3_CRITICAL, threshold_config.s3_hysteresis),
                (AlertSeverity.S2_WARNING, threshold_config.s2_hysteresis),
                (AlertSeverity.S1_ADVISORY, threshold_config.s1_hysteresis),
            ]:
                if hysteresis is None:
                    continue

                if high_is_alarm:
                    # High value triggers alarm (e.g., carryover risk)
                    if state.is_alarmed:
                        # Check clear threshold
                        if current_value < hysteresis.clear_threshold:
                            pass  # Could clear at this level
                        else:
                            new_severity = severity
                            break
                    else:
                        # Check alarm threshold
                        if current_value >= hysteresis.alarm_threshold:
                            new_severity = severity
                            break
                else:
                    # Low value triggers alarm (e.g., dryness fraction)
                    if state.is_alarmed:
                        # Check clear threshold
                        if current_value > hysteresis.clear_threshold:
                            pass  # Could clear at this level
                        else:
                            new_severity = severity
                            break
                    else:
                        # Check alarm threshold
                        if current_value <= hysteresis.alarm_threshold:
                            new_severity = severity
                            break

            # Determine if we should transition
            should_alarm = new_severity is not None
            current_alarmed = state.is_alarmed

            # Check dead time
            if state.last_transition_time:
                dead_time = threshold_config.s1_hysteresis.dead_time_seconds if threshold_config.s1_hysteresis else 30.0
                time_since_transition = (now - state.last_transition_time).total_seconds()
                if time_since_transition < dead_time:
                    self._stats["alerts_hysteresis_blocked"] += 1
                    return None

            # Check for state change
            if should_alarm != current_alarmed:
                # Confirmation logic
                if state.last_value is not None:
                    if should_alarm:
                        state.confirmation_count += 1
                    else:
                        state.confirmation_count = 0

                    confirmation_required = (
                        threshold_config.s1_hysteresis.confirmation_samples
                        if threshold_config.s1_hysteresis else 3
                    )

                    if state.confirmation_count < confirmation_required:
                        state.last_value = current_value
                        return None

                # Transition confirmed
                state.is_alarmed = should_alarm
                state.last_transition_time = now
                state.confirmation_count = 0
                state.severity = new_severity
                state.last_value = current_value

                return (should_alarm, new_severity)

            state.last_value = current_value
            return None

    # =========================================================================
    # Rate Limiting
    # =========================================================================

    def _check_rate_limit(
        self,
        separator_id: str,
        alert_type: QualityAlertType,
    ) -> bool:
        """
        Check if alert is rate limited.

        Returns:
            True if rate limited (should not create alert)
        """
        threshold_config = self._thresholds.get(alert_type)
        if not threshold_config:
            return False

        key = (separator_id, alert_type)
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(hours=1)

        with self._lock:
            if key not in self._rate_limit_buckets:
                self._rate_limit_buckets[key] = []

            # Clean old entries
            self._rate_limit_buckets[key] = [
                t for t in self._rate_limit_buckets[key] if t > cutoff
            ]

            # Check limit
            if len(self._rate_limit_buckets[key]) >= threshold_config.rate_limit_per_hour:
                self._stats["alerts_rate_limited"] += 1
                logger.debug(
                    "Alert rate limited: separator=%s, type=%s",
                    separator_id,
                    alert_type.value,
                )
                return True

            # Check cooldown
            if self._rate_limit_buckets[key]:
                last_alert = max(self._rate_limit_buckets[key])
                if (now - last_alert).total_seconds() < threshold_config.cooldown_seconds:
                    self._stats["alerts_rate_limited"] += 1
                    return True

            # Add current timestamp
            self._rate_limit_buckets[key].append(now)
            return False

    # =========================================================================
    # Alert Evaluation Methods
    # =========================================================================

    def evaluate_dryness(
        self,
        dryness_fraction: float,
        separator_id: str,
        context: Optional[AlertContext] = None,
    ) -> Optional[Alert]:
        """
        Evaluate dryness fraction and create alert if threshold exceeded.

        Args:
            dryness_fraction: Current dryness fraction (0.0-1.0)
            separator_id: Separator identifier
            context: Optional additional context

        Returns:
            Alert if condition triggers, None otherwise
        """
        result = self._evaluate_hysteresis(
            separator_id=separator_id,
            alert_type=QualityAlertType.LOW_DRYNESS,
            current_value=dryness_fraction,
            high_is_alarm=False,  # Low dryness triggers alarm
        )

        if result is None:
            return None

        should_alarm, severity = result

        if not should_alarm:
            # Clear any existing alert
            self._auto_resolve_alert(separator_id, QualityAlertType.LOW_DRYNESS)
            return None

        # Check rate limit
        if self._check_rate_limit(separator_id, QualityAlertType.LOW_DRYNESS):
            return None

        # Create alert
        threshold_config = self._thresholds[QualityAlertType.LOW_DRYNESS]
        threshold = getattr(threshold_config, f"s{severity.value}_hysteresis").alarm_threshold

        ctx = context or AlertContext()
        ctx.separator_id = separator_id
        ctx.dryness_fraction = dryness_fraction
        ctx.current_value = dryness_fraction
        ctx.threshold_value = threshold

        message = (
            f"Low steam dryness detected: x={dryness_fraction:.3f} "
            f"(threshold: {threshold:.3f}) on separator {separator_id}"
        )

        return self.create_alert(
            alert_type=QualityAlertType.LOW_DRYNESS,
            severity=severity,
            source=separator_id,
            message=message,
            context=ctx,
        )

    def evaluate_moisture(
        self,
        moisture_content: float,
        separator_id: str,
        context: Optional[AlertContext] = None,
    ) -> Optional[Alert]:
        """
        Evaluate moisture content and create alert if threshold exceeded.

        Args:
            moisture_content: Current moisture content (0.0-1.0)
            separator_id: Separator identifier
            context: Optional additional context

        Returns:
            Alert if condition triggers, None otherwise
        """
        result = self._evaluate_hysteresis(
            separator_id=separator_id,
            alert_type=QualityAlertType.HIGH_MOISTURE,
            current_value=moisture_content,
            high_is_alarm=True,  # High moisture triggers alarm
        )

        if result is None:
            return None

        should_alarm, severity = result

        if not should_alarm:
            self._auto_resolve_alert(separator_id, QualityAlertType.HIGH_MOISTURE)
            return None

        if self._check_rate_limit(separator_id, QualityAlertType.HIGH_MOISTURE):
            return None

        threshold_config = self._thresholds[QualityAlertType.HIGH_MOISTURE]
        threshold = getattr(threshold_config, f"s{severity.value}_hysteresis").alarm_threshold

        ctx = context or AlertContext()
        ctx.separator_id = separator_id
        ctx.moisture_content = moisture_content
        ctx.current_value = moisture_content
        ctx.threshold_value = threshold

        message = (
            f"High moisture carryover detected: {moisture_content*100:.1f}% "
            f"(threshold: {threshold*100:.1f}%) on separator {separator_id}"
        )

        return self.create_alert(
            alert_type=QualityAlertType.HIGH_MOISTURE,
            severity=severity,
            source=separator_id,
            message=message,
            context=ctx,
        )

    def evaluate_carryover_risk(
        self,
        carryover_risk: float,
        separator_id: str,
        context: Optional[AlertContext] = None,
    ) -> Optional[Alert]:
        """
        Evaluate carryover risk and create alert if threshold exceeded.

        Args:
            carryover_risk: Current carryover risk score (0.0-1.0)
            separator_id: Separator identifier
            context: Optional additional context

        Returns:
            Alert if condition triggers, None otherwise
        """
        result = self._evaluate_hysteresis(
            separator_id=separator_id,
            alert_type=QualityAlertType.CARRYOVER_RISK,
            current_value=carryover_risk,
            high_is_alarm=True,
        )

        if result is None:
            return None

        should_alarm, severity = result

        if not should_alarm:
            self._auto_resolve_alert(separator_id, QualityAlertType.CARRYOVER_RISK)
            return None

        if self._check_rate_limit(separator_id, QualityAlertType.CARRYOVER_RISK):
            return None

        threshold_config = self._thresholds[QualityAlertType.CARRYOVER_RISK]
        threshold = getattr(threshold_config, f"s{severity.value}_hysteresis").alarm_threshold

        ctx = context or AlertContext()
        ctx.separator_id = separator_id
        ctx.carryover_risk = carryover_risk
        ctx.current_value = carryover_risk
        ctx.threshold_value = threshold

        message = (
            f"Elevated carryover risk: {carryover_risk:.2f} "
            f"(threshold: {threshold:.2f}) on separator {separator_id}"
        )

        return self.create_alert(
            alert_type=QualityAlertType.CARRYOVER_RISK,
            severity=severity,
            source=separator_id,
            message=message,
            context=ctx,
        )

    def evaluate_separator_level(
        self,
        level_percent: float,
        separator_id: str,
        context: Optional[AlertContext] = None,
    ) -> Optional[Alert]:
        """
        Evaluate separator level and create flooding alert if threshold exceeded.

        Args:
            level_percent: Current separator level (0-100%)
            separator_id: Separator identifier
            context: Optional additional context

        Returns:
            Alert if condition triggers, None otherwise
        """
        result = self._evaluate_hysteresis(
            separator_id=separator_id,
            alert_type=QualityAlertType.SEPARATOR_FLOODING,
            current_value=level_percent,
            high_is_alarm=True,
        )

        if result is None:
            return None

        should_alarm, severity = result

        if not should_alarm:
            self._auto_resolve_alert(separator_id, QualityAlertType.SEPARATOR_FLOODING)
            return None

        if self._check_rate_limit(separator_id, QualityAlertType.SEPARATOR_FLOODING):
            return None

        threshold_config = self._thresholds[QualityAlertType.SEPARATOR_FLOODING]
        threshold = getattr(threshold_config, f"s{severity.value}_hysteresis").alarm_threshold

        ctx = context or AlertContext()
        ctx.separator_id = separator_id
        ctx.separator_level_percent = level_percent
        ctx.current_value = level_percent
        ctx.threshold_value = threshold

        message = (
            f"Separator flooding risk: level={level_percent:.1f}% "
            f"(threshold: {threshold:.1f}%) on separator {separator_id}"
        )

        return self.create_alert(
            alert_type=QualityAlertType.SEPARATOR_FLOODING,
            severity=severity,
            source=separator_id,
            message=message,
            context=ctx,
        )

    def evaluate_water_hammer_risk(
        self,
        water_hammer_risk: float,
        separator_id: str,
        context: Optional[AlertContext] = None,
    ) -> Optional[Alert]:
        """
        Evaluate water hammer risk and create alert if threshold exceeded.

        Args:
            water_hammer_risk: Current water hammer risk score (0.0-1.0)
            separator_id: Separator identifier
            context: Optional additional context

        Returns:
            Alert if condition triggers, None otherwise
        """
        result = self._evaluate_hysteresis(
            separator_id=separator_id,
            alert_type=QualityAlertType.WATER_HAMMER_RISK,
            current_value=water_hammer_risk,
            high_is_alarm=True,
        )

        if result is None:
            return None

        should_alarm, severity = result

        if not should_alarm:
            self._auto_resolve_alert(separator_id, QualityAlertType.WATER_HAMMER_RISK)
            return None

        if self._check_rate_limit(separator_id, QualityAlertType.WATER_HAMMER_RISK):
            return None

        threshold_config = self._thresholds[QualityAlertType.WATER_HAMMER_RISK]
        threshold = getattr(threshold_config, f"s{severity.value}_hysteresis").alarm_threshold

        ctx = context or AlertContext()
        ctx.separator_id = separator_id
        ctx.current_value = water_hammer_risk
        ctx.threshold_value = threshold

        message = (
            f"Water hammer risk detected: risk={water_hammer_risk:.2f} "
            f"(threshold: {threshold:.2f}) on separator {separator_id}"
        )

        return self.create_alert(
            alert_type=QualityAlertType.WATER_HAMMER_RISK,
            severity=severity,
            source=separator_id,
            message=message,
            context=ctx,
        )

    def evaluate_data_quality(
        self,
        quality_score: float,
        separator_id: str,
        sensor_status: Optional[str] = None,
        context: Optional[AlertContext] = None,
    ) -> Optional[Alert]:
        """
        Evaluate data quality and create alert if degraded.

        Args:
            quality_score: Data quality score (0.0-1.0, 1.0 = perfect)
            separator_id: Separator identifier
            sensor_status: Optional sensor status description
            context: Optional additional context

        Returns:
            Alert if condition triggers, None otherwise
        """
        result = self._evaluate_hysteresis(
            separator_id=separator_id,
            alert_type=QualityAlertType.DATA_QUALITY_DEGRADED,
            current_value=quality_score,
            high_is_alarm=False,  # Low quality triggers alarm
        )

        if result is None:
            return None

        should_alarm, severity = result

        if not should_alarm:
            self._auto_resolve_alert(separator_id, QualityAlertType.DATA_QUALITY_DEGRADED)
            return None

        if self._check_rate_limit(separator_id, QualityAlertType.DATA_QUALITY_DEGRADED):
            return None

        threshold_config = self._thresholds[QualityAlertType.DATA_QUALITY_DEGRADED]
        threshold = getattr(threshold_config, f"s{severity.value}_hysteresis").alarm_threshold

        ctx = context or AlertContext()
        ctx.separator_id = separator_id
        ctx.data_quality_score = quality_score
        ctx.sensor_status = sensor_status
        ctx.current_value = quality_score
        ctx.threshold_value = threshold

        message = (
            f"Data quality degraded: score={quality_score:.2f} "
            f"(threshold: {threshold:.2f}) for separator {separator_id}"
        )
        if sensor_status:
            message += f" - {sensor_status}"

        return self.create_alert(
            alert_type=QualityAlertType.DATA_QUALITY_DEGRADED,
            severity=severity,
            source=separator_id,
            message=message,
            context=ctx,
        )

    # =========================================================================
    # Alert Lifecycle Management
    # =========================================================================

    def _auto_resolve_alert(
        self,
        separator_id: str,
        alert_type: QualityAlertType,
    ) -> None:
        """Automatically resolve alert when condition clears."""
        with self._lock:
            for alert_id, alert in list(self._active_alerts.items()):
                if (
                    alert.alert_type == alert_type
                    and alert.source == separator_id
                    and alert.state in [AlertState.FIRING, AlertState.ACKNOWLEDGED]
                ):
                    self.resolve_alert(alert_id, "Auto-resolved: condition cleared")

    def create_alert(
        self,
        alert_type: QualityAlertType,
        severity: AlertSeverity,
        source: str,
        message: str,
        context: Optional[AlertContext] = None,
        labels: Optional[Dict[str, str]] = None,
    ) -> Alert:
        """
        Create a new alert.

        Args:
            alert_type: Type of alert
            severity: Severity level
            source: Source identifier (separator, equipment, etc.)
            message: Human-readable alert message
            context: Optional contextual information
            labels: Optional labels for categorization

        Returns:
            Created Alert instance (may be existing if deduplicated)
        """
        context = context or AlertContext()
        labels = labels or {}

        # Create alert with temporary fingerprint
        alert = Alert(
            alert_id=str(uuid.uuid4()),
            alert_type=alert_type,
            severity=severity,
            source=source,
            message=message,
            context=context,
            labels=labels,
        )

        with self._lock:
            # Check suppression
            if alert.fingerprint in self._suppressed_fingerprints:
                expires_at = self._suppressed_fingerprints[alert.fingerprint]
                if datetime.now(timezone.utc) <= expires_at:
                    logger.debug("Alert suppressed: %s", alert.fingerprint[:8])
                    alert.state = AlertState.SUPPRESSED
                    return alert
                del self._suppressed_fingerprints[alert.fingerprint]

            # Check deduplication
            if alert.fingerprint in self._alert_by_fingerprint:
                existing_id = self._alert_by_fingerprint[alert.fingerprint]
                if existing_id in self._active_alerts:
                    existing = self._active_alerts[existing_id]
                    time_since_fired = (
                        datetime.now(timezone.utc) - (existing.fired_at or existing.created_at)
                    ).total_seconds()

                    if time_since_fired < self._deduplication_window_s:
                        self._stats["alerts_deduplicated"] += 1
                        logger.debug(
                            "Alert deduplicated: %s -> %s",
                            alert.fingerprint[:8],
                            existing.alert_id[:8],
                        )
                        return existing

            # Fire the alert
            alert.state = AlertState.FIRING
            alert.fired_at = datetime.now(timezone.utc)

            # Store alert
            self._active_alerts[alert.alert_id] = alert
            self._alert_by_fingerprint[alert.fingerprint] = alert.alert_id
            self._stats["alerts_created"] += 1

        logger.info(
            "Alert created: %s | S%d | %s: %s",
            alert_type.value,
            severity.value,
            alert.alert_id[:8],
            message,
        )

        # Invoke callback
        if self._on_alert_callback:
            try:
                self._on_alert_callback(alert)
            except Exception as e:
                logger.error("Alert callback failed: %s", e)

        # Send notifications asynchronously
        try:
            asyncio.create_task(self._notify_async(alert))
        except RuntimeError:
            # No event loop running, skip async notification
            pass

        return alert

    async def _notify_async(self, alert: Alert) -> None:
        """Send notifications to all channels asynchronously."""
        for channel in self._channels:
            try:
                success = await channel.send(alert)
                if success:
                    alert.notification_count += 1
                    self._stats["notifications_sent"] += 1
            except Exception as e:
                logger.error("Notification channel failed: %s", e)

    def acknowledge_alert(
        self,
        alert_id: str,
        user_id: str,
        notes: Optional[str] = None,
    ) -> AcknowledgmentResult:
        """
        Acknowledge an active alert.

        Args:
            alert_id: Alert identifier
            user_id: User performing acknowledgment
            notes: Optional acknowledgment notes

        Returns:
            AcknowledgmentResult with operation status
        """
        now = datetime.now(timezone.utc)

        with self._lock:
            if alert_id not in self._active_alerts:
                return AcknowledgmentResult(
                    success=False,
                    alert_id=alert_id,
                    acknowledged_by=user_id,
                    acknowledged_at=now,
                    error_message=f"Alert not found: {alert_id}",
                )

            alert = self._active_alerts[alert_id]

            if alert.state not in [AlertState.FIRING, AlertState.PENDING]:
                return AcknowledgmentResult(
                    success=False,
                    alert_id=alert_id,
                    acknowledged_by=user_id,
                    acknowledged_at=now,
                    error_message=f"Alert not in acknowledgeble state: {alert.state.value}",
                )

            # Update alert
            alert.state = AlertState.ACKNOWLEDGED
            alert.acknowledged_at = now
            alert.acknowledged_by = user_id
            alert.acknowledgment_notes = notes

            self._stats["alerts_acknowledged"] += 1

        logger.info(
            "Alert acknowledged: %s by %s%s",
            alert_id[:8],
            user_id,
            f" - {notes}" if notes else "",
        )

        return AcknowledgmentResult(
            success=True,
            alert_id=alert_id,
            acknowledged_by=user_id,
            acknowledged_at=now,
            notes=notes,
        )

    def resolve_alert(
        self,
        alert_id: str,
        resolution_notes: Optional[str] = None,
    ) -> bool:
        """
        Resolve an active alert.

        Args:
            alert_id: Alert identifier
            resolution_notes: Optional resolution notes

        Returns:
            True if successfully resolved
        """
        with self._lock:
            if alert_id not in self._active_alerts:
                return False

            alert = self._active_alerts[alert_id]
            alert.state = AlertState.RESOLVED
            alert.resolved_at = datetime.now(timezone.utc)

            # Move to history
            self._alert_history.append(alert)
            if len(self._alert_history) > self._max_history_size:
                self._alert_history = self._alert_history[-self._max_history_size:]

            # Clean up active storage
            del self._active_alerts[alert_id]
            if alert.fingerprint in self._alert_by_fingerprint:
                del self._alert_by_fingerprint[alert.fingerprint]

            self._stats["alerts_resolved"] += 1

        logger.info(
            "Alert resolved: %s%s",
            alert_id[:8],
            f" - {resolution_notes}" if resolution_notes else "",
        )

        return True

    def escalate_alert(
        self,
        alert_id: str,
        escalation_level: EscalationLevel,
        reason: Optional[str] = None,
    ) -> EscalationResult:
        """
        Escalate an alert to a higher level.

        Args:
            alert_id: Alert identifier
            escalation_level: Target escalation level
            reason: Optional escalation reason

        Returns:
            EscalationResult with operation status
        """
        now = datetime.now(timezone.utc)

        with self._lock:
            if alert_id not in self._active_alerts:
                return EscalationResult(
                    success=False,
                    alert_id=alert_id,
                    previous_level=EscalationLevel.L1_OPERATOR,
                    new_level=escalation_level,
                    escalated_at=now,
                    error_message=f"Alert not found: {alert_id}",
                )

            alert = self._active_alerts[alert_id]
            previous_level = alert.escalation_level

            # Validate escalation direction
            level_order = [
                EscalationLevel.L1_OPERATOR,
                EscalationLevel.L2_SUPERVISOR,
                EscalationLevel.L3_ENGINEER,
                EscalationLevel.L4_MANAGEMENT,
            ]

            if level_order.index(escalation_level) <= level_order.index(previous_level):
                return EscalationResult(
                    success=False,
                    alert_id=alert_id,
                    previous_level=previous_level,
                    new_level=escalation_level,
                    escalated_at=now,
                    error_message=(
                        f"Cannot escalate from {previous_level.value} to {escalation_level.value}"
                    ),
                )

            # Update alert
            alert.escalation_level = escalation_level
            alert.escalated_at = now

            self._stats["alerts_escalated"] += 1

        logger.warning(
            "Alert escalated: %s from %s to %s%s",
            alert_id[:8],
            previous_level.value,
            escalation_level.value,
            f" - {reason}" if reason else "",
        )

        return EscalationResult(
            success=True,
            alert_id=alert_id,
            previous_level=previous_level,
            new_level=escalation_level,
            escalated_at=now,
            escalation_reason=reason,
        )

    # =========================================================================
    # Query Methods
    # =========================================================================

    def get_active_alerts(
        self,
        filters: Optional[AlertFilter] = None,
    ) -> List[Alert]:
        """
        Get active alerts with optional filtering.

        Args:
            filters: Optional filter criteria

        Returns:
            List of matching active alerts sorted by severity and time
        """
        with self._lock:
            alerts = list(self._active_alerts.values())

        if filters:
            if filters.alert_types:
                alerts = [a for a in alerts if a.alert_type in filters.alert_types]

            if filters.severities:
                alerts = [a for a in alerts if a.severity in filters.severities]

            if filters.states:
                alerts = [a for a in alerts if a.state in filters.states]

            if filters.source_pattern:
                alerts = [a for a in alerts if filters.source_pattern in a.source]

            if filters.separator_id:
                alerts = [
                    a for a in alerts
                    if a.context.separator_id == filters.separator_id
                ]

            if filters.site_id:
                alerts = [a for a in alerts if a.context.site_id == filters.site_id]

            if filters.escalation_levels:
                alerts = [
                    a for a in alerts if a.escalation_level in filters.escalation_levels
                ]

            if filters.from_time:
                alerts = [a for a in alerts if a.created_at >= filters.from_time]

            if filters.to_time:
                alerts = [a for a in alerts if a.created_at <= filters.to_time]

        # Sort by severity (higher value = more severe) then by created time
        return sorted(
            alerts,
            key=lambda a: (-a.severity.value, a.created_at),
        )

    def get_alert_history(
        self,
        time_window: Optional[timedelta] = None,
        limit: int = 100,
    ) -> List[Alert]:
        """
        Get alert history within time window.

        Args:
            time_window: Optional time window to filter
            limit: Maximum number of alerts to return

        Returns:
            List of historical alerts
        """
        with self._lock:
            alerts = list(self._alert_history)

        if time_window:
            cutoff = datetime.now(timezone.utc) - time_window
            alerts = [a for a in alerts if a.created_at >= cutoff]

        return sorted(alerts, key=lambda a: a.created_at, reverse=True)[:limit]

    def get_statistics(self) -> Dict[str, Any]:
        """Get alerting statistics."""
        with self._lock:
            return {
                **self._stats,
                "active_alerts": len(self._active_alerts),
                "suppressed_patterns": len(self._suppressed_fingerprints),
                "history_size": len(self._alert_history),
                "notification_channels": len(self._channels),
                "hysteresis_states": len(self._hysteresis_states),
            }

    def add_notification_channel(self, channel: NotificationChannel) -> None:
        """Add a notification channel."""
        self._channels.append(channel)
        logger.info("Added notification channel: %s", type(channel).__name__)

    def suppress_alert_pattern(
        self,
        fingerprint: str,
        duration_minutes: int = 60,
    ) -> bool:
        """
        Suppress alerts matching a fingerprint pattern.

        Args:
            fingerprint: Alert fingerprint to suppress
            duration_minutes: Suppression duration

        Returns:
            True if suppression was set
        """
        with self._lock:
            expires_at = datetime.now(timezone.utc) + timedelta(minutes=duration_minutes)
            self._suppressed_fingerprints[fingerprint] = expires_at

        logger.info(
            "Alert pattern suppressed: %s until %s",
            fingerprint[:8],
            expires_at,
        )
        return True
