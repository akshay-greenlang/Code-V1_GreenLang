"""
Violation Handler for GL-001 ThermalCommand Safety System

This module handles safety boundary violations by:
- Blocking actuation on violation
- Emitting alarms and events
- Creating immutable audit records
- Executing escalation procedures
- Notifying operators and systems

All violations are logged with cryptographic hashes for audit compliance.

Example:
    >>> from violation_handler import ViolationHandler
    >>> handler = ViolationHandler()
    >>> handler.handle_violation(violation)
"""

from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set
import hashlib
import json
import logging
import threading
import uuid
from collections import defaultdict

from pydantic import BaseModel, Field

from .safety_schemas import (
    BoundaryViolation,
    PolicyType,
    ViolationType,
    ViolationSeverity,
    ActionGateResult,
    SafetyAuditRecord,
    GateDecision,
)

logger = logging.getLogger(__name__)


class EscalationLevel(int, Enum):
    """Escalation levels for violation handling."""

    LEVEL_0 = 0  # Logged only
    LEVEL_1 = 1  # Local alarm
    LEVEL_2 = 2  # Operator notification
    LEVEL_3 = 3  # Supervisor notification
    LEVEL_4 = 4  # Emergency response


class NotificationType(str, Enum):
    """Types of notifications."""

    LOG = "log"
    ALARM = "alarm"
    EVENT = "event"
    EMAIL = "email"
    SMS = "sms"
    WEBHOOK = "webhook"
    HISTORIAN = "historian"


class NotificationTarget(BaseModel):
    """Target for violation notifications."""

    target_id: str = Field(..., description="Unique target identifier")
    notification_type: NotificationType = Field(..., description="Type of notification")
    endpoint: str = Field(..., description="Endpoint/address for notification")
    enabled: bool = Field(default=True, description="Whether target is enabled")
    min_severity: ViolationSeverity = Field(
        default=ViolationSeverity.WARNING,
        description="Minimum severity to notify"
    )
    escalation_level: EscalationLevel = Field(
        default=EscalationLevel.LEVEL_1,
        description="Minimum escalation level to notify"
    )


class ViolationEvent(BaseModel):
    """Event generated from a violation."""

    event_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique event identifier"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Event timestamp"
    )
    violation: BoundaryViolation = Field(..., description="Source violation")
    escalation_level: EscalationLevel = Field(..., description="Escalation level")
    notifications_sent: List[str] = Field(
        default_factory=list,
        description="List of notification target IDs that were notified"
    )
    action_taken: str = Field(..., description="Action taken in response")
    provenance_hash: str = Field(default="", description="SHA-256 hash for audit")


class EscalationRule(BaseModel):
    """Rule for determining escalation level."""

    rule_id: str = Field(..., description="Rule identifier")
    violation_type: Optional[ViolationType] = Field(
        None,
        description="Violation type to match"
    )
    severity: Optional[ViolationSeverity] = Field(
        None,
        description="Severity to match"
    )
    policy_id_pattern: Optional[str] = Field(
        None,
        description="Policy ID pattern to match"
    )
    tag_pattern: Optional[str] = Field(
        None,
        description="Tag pattern to match"
    )
    escalation_level: EscalationLevel = Field(
        ...,
        description="Escalation level to apply"
    )
    description: str = Field(default="", description="Rule description")


# Default escalation rules
DEFAULT_ESCALATION_RULES: List[EscalationRule] = [
    # SIS violations - highest escalation
    EscalationRule(
        rule_id="ESC_SIS_001",
        violation_type=ViolationType.SIS_VIOLATION,
        escalation_level=EscalationLevel.LEVEL_4,
        description="SIS violations require emergency response",
    ),
    # Emergency severity - high escalation
    EscalationRule(
        rule_id="ESC_EMERGENCY_001",
        severity=ViolationSeverity.EMERGENCY,
        escalation_level=EscalationLevel.LEVEL_3,
        description="Emergency severity requires supervisor notification",
    ),
    # Critical severity - operator notification
    EscalationRule(
        rule_id="ESC_CRITICAL_001",
        severity=ViolationSeverity.CRITICAL,
        escalation_level=EscalationLevel.LEVEL_2,
        description="Critical severity requires operator notification",
    ),
    # Interlock violations - high escalation
    EscalationRule(
        rule_id="ESC_INTERLOCK_001",
        violation_type=ViolationType.INTERLOCK_ACTIVE,
        escalation_level=EscalationLevel.LEVEL_3,
        description="Interlock violations require supervisor notification",
    ),
    # Unauthorized tag access - high escalation
    EscalationRule(
        rule_id="ESC_UNAUTHORIZED_001",
        violation_type=ViolationType.UNAUTHORIZED_TAG,
        escalation_level=EscalationLevel.LEVEL_2,
        description="Unauthorized tag access requires operator notification",
    ),
    # Warning severity - local alarm only
    EscalationRule(
        rule_id="ESC_WARNING_001",
        severity=ViolationSeverity.WARNING,
        escalation_level=EscalationLevel.LEVEL_1,
        description="Warning severity triggers local alarm",
    ),
]


class ViolationHandler:
    """
    Handler for safety boundary violations.

    Processes violations by:
    1. Determining escalation level
    2. Sending notifications to appropriate targets
    3. Creating immutable audit records
    4. Tracking violation statistics

    All violations result in blocked actuation - this handler
    manages the post-block response and notification.

    Attributes:
        notification_targets: Configured notification targets
        escalation_rules: Rules for determining escalation
        audit_records: Immutable audit record chain

    Example:
        >>> handler = ViolationHandler()
        >>> handler.add_notification_target(email_target)
        >>> handler.handle_violation(violation)
    """

    def __init__(
        self,
        notification_targets: Optional[List[NotificationTarget]] = None,
        escalation_rules: Optional[List[EscalationRule]] = None,
        audit_callback: Optional[Callable[[SafetyAuditRecord], None]] = None,
    ) -> None:
        """
        Initialize Violation Handler.

        Args:
            notification_targets: Initial notification targets
            escalation_rules: Custom escalation rules
            audit_callback: Callback for audit record creation
        """
        self._notification_targets: Dict[str, NotificationTarget] = {}
        self._escalation_rules = escalation_rules or DEFAULT_ESCALATION_RULES
        self._audit_callback = audit_callback

        # Register initial targets
        for target in (notification_targets or []):
            self.add_notification_target(target)

        # Audit trail
        self._audit_records: List[SafetyAuditRecord] = []
        self._audit_lock = threading.Lock()

        # Violation events
        self._violation_events: List[ViolationEvent] = []
        self._events_lock = threading.Lock()

        # Statistics
        self._stats = {
            "violations_handled": 0,
            "escalations_level_0": 0,
            "escalations_level_1": 0,
            "escalations_level_2": 0,
            "escalations_level_3": 0,
            "escalations_level_4": 0,
            "notifications_sent": 0,
            "notifications_failed": 0,
        }
        self._stats_lock = threading.Lock()

        # Violation rate tracking
        self._violation_times: Dict[str, List[datetime]] = defaultdict(list)
        self._rate_lock = threading.Lock()

        # Notification handlers (for testing/extensibility)
        self._notification_handlers: Dict[NotificationType, Callable] = {
            NotificationType.LOG: self._send_log_notification,
            NotificationType.ALARM: self._send_alarm_notification,
            NotificationType.EVENT: self._send_event_notification,
            NotificationType.WEBHOOK: self._send_webhook_notification,
            NotificationType.HISTORIAN: self._send_historian_notification,
        }

        logger.info("ViolationHandler initialized")

    def handle_violation(
        self,
        violation: BoundaryViolation,
        gate_result: Optional[ActionGateResult] = None,
    ) -> ViolationEvent:
        """
        Handle a safety boundary violation.

        This is the main entry point for violation handling.
        Always blocks actuation and creates audit trail.

        Args:
            violation: The violation to handle
            gate_result: Associated gate result if available

        Returns:
            ViolationEvent with handling details
        """
        start_time = datetime.utcnow()

        with self._stats_lock:
            self._stats["violations_handled"] += 1

        logger.warning(
            f"Handling violation: policy={violation.policy_id}, "
            f"tag={violation.tag_id}, type={violation.violation_type}, "
            f"severity={violation.severity}"
        )

        # Step 1: Determine escalation level
        escalation = self._determine_escalation(violation)

        # Update escalation stats
        with self._stats_lock:
            self._stats[f"escalations_level_{escalation.value}"] += 1

        # Step 2: Send notifications
        notified_targets = self._send_notifications(violation, escalation)

        # Step 3: Create audit record
        audit_record = self._create_audit_record(
            violation, gate_result, escalation, notified_targets
        )

        # Step 4: Create violation event
        event = ViolationEvent(
            violation=violation,
            escalation_level=escalation,
            notifications_sent=notified_targets,
            action_taken="ACTUATION_BLOCKED",
        )

        # Compute provenance hash
        hash_input = (
            f"{event.timestamp}|{violation.provenance_hash}|"
            f"{escalation.value}|{','.join(notified_targets)}"
        )
        event = ViolationEvent(
            **{**event.dict(), "provenance_hash": hashlib.sha256(hash_input.encode()).hexdigest()}
        )

        # Store event
        with self._events_lock:
            self._violation_events.append(event)

        # Track violation rate
        self._track_violation_rate(violation)

        # Check for violation storm
        self._check_violation_storm(violation)

        logger.info(
            f"Violation handled: event_id={event.event_id}, "
            f"escalation={escalation.name}, "
            f"notified={len(notified_targets)} targets"
        )

        return event

    def _determine_escalation(
        self,
        violation: BoundaryViolation
    ) -> EscalationLevel:
        """
        Determine escalation level for a violation.

        Args:
            violation: The violation

        Returns:
            EscalationLevel to apply
        """
        max_escalation = EscalationLevel.LEVEL_0

        for rule in self._escalation_rules:
            # Check if rule matches
            if self._rule_matches(rule, violation):
                if rule.escalation_level.value > max_escalation.value:
                    max_escalation = rule.escalation_level
                    logger.debug(
                        f"Rule {rule.rule_id} matched, "
                        f"escalation now {max_escalation.name}"
                    )

        return max_escalation

    def _rule_matches(
        self,
        rule: EscalationRule,
        violation: BoundaryViolation
    ) -> bool:
        """
        Check if an escalation rule matches a violation.

        Args:
            rule: Escalation rule
            violation: Violation to check

        Returns:
            True if rule matches
        """
        # Check violation type
        if rule.violation_type is not None:
            if violation.violation_type != rule.violation_type:
                return False

        # Check severity
        if rule.severity is not None:
            if violation.severity != rule.severity:
                return False

        # Check policy ID pattern
        if rule.policy_id_pattern is not None:
            from fnmatch import fnmatch
            if not fnmatch(violation.policy_id, rule.policy_id_pattern):
                return False

        # Check tag pattern
        if rule.tag_pattern is not None:
            from fnmatch import fnmatch
            if not fnmatch(violation.tag_id, rule.tag_pattern):
                return False

        return True

    def _send_notifications(
        self,
        violation: BoundaryViolation,
        escalation: EscalationLevel,
    ) -> List[str]:
        """
        Send notifications for a violation.

        Args:
            violation: The violation
            escalation: Escalation level

        Returns:
            List of target IDs that were notified
        """
        notified: List[str] = []

        for target_id, target in self._notification_targets.items():
            if not target.enabled:
                continue

            # Check minimum severity
            severity_order = {
                ViolationSeverity.WARNING: 0,
                ViolationSeverity.CRITICAL: 1,
                ViolationSeverity.EMERGENCY: 2,
            }
            if severity_order.get(violation.severity, 0) < severity_order.get(target.min_severity, 0):
                continue

            # Check escalation level
            if escalation.value < target.escalation_level.value:
                continue

            # Send notification
            handler = self._notification_handlers.get(target.notification_type)
            if handler:
                try:
                    handler(target, violation, escalation)
                    notified.append(target_id)
                    with self._stats_lock:
                        self._stats["notifications_sent"] += 1
                except Exception as e:
                    logger.error(f"Notification failed for {target_id}: {e}")
                    with self._stats_lock:
                        self._stats["notifications_failed"] += 1

        return notified

    def _send_log_notification(
        self,
        target: NotificationTarget,
        violation: BoundaryViolation,
        escalation: EscalationLevel,
    ) -> None:
        """Send log notification."""
        log_level = {
            ViolationSeverity.WARNING: logging.WARNING,
            ViolationSeverity.CRITICAL: logging.ERROR,
            ViolationSeverity.EMERGENCY: logging.CRITICAL,
        }.get(violation.severity, logging.ERROR)

        logger.log(
            log_level,
            f"[SAFETY VIOLATION] policy={violation.policy_id}, "
            f"tag={violation.tag_id}, type={violation.violation_type}, "
            f"severity={violation.severity}, escalation={escalation.name}, "
            f"message={violation.message}"
        )

    def _send_alarm_notification(
        self,
        target: NotificationTarget,
        violation: BoundaryViolation,
        escalation: EscalationLevel,
    ) -> None:
        """Send alarm notification."""
        # In production, this would integrate with alarm management system
        logger.info(
            f"[ALARM] {target.endpoint}: {violation.policy_id} - "
            f"{violation.message} (escalation: {escalation.name})"
        )

    def _send_event_notification(
        self,
        target: NotificationTarget,
        violation: BoundaryViolation,
        escalation: EscalationLevel,
    ) -> None:
        """Send event notification."""
        # In production, this would publish to event bus
        logger.info(
            f"[EVENT] {target.endpoint}: Safety violation event published"
        )

    def _send_webhook_notification(
        self,
        target: NotificationTarget,
        violation: BoundaryViolation,
        escalation: EscalationLevel,
    ) -> None:
        """Send webhook notification."""
        # In production, this would make HTTP POST to webhook
        payload = {
            "violation_id": violation.violation_id,
            "timestamp": violation.timestamp.isoformat(),
            "policy_id": violation.policy_id,
            "tag_id": violation.tag_id,
            "violation_type": violation.violation_type,
            "severity": violation.severity,
            "escalation_level": escalation.value,
            "message": violation.message,
        }
        logger.info(
            f"[WEBHOOK] {target.endpoint}: {json.dumps(payload)}"
        )

    def _send_historian_notification(
        self,
        target: NotificationTarget,
        violation: BoundaryViolation,
        escalation: EscalationLevel,
    ) -> None:
        """Send historian notification."""
        # In production, this would write to process historian
        logger.info(
            f"[HISTORIAN] {target.endpoint}: Recording violation "
            f"{violation.violation_id}"
        )

    def _create_audit_record(
        self,
        violation: BoundaryViolation,
        gate_result: Optional[ActionGateResult],
        escalation: EscalationLevel,
        notified_targets: List[str],
    ) -> SafetyAuditRecord:
        """
        Create immutable audit record.

        Args:
            violation: The violation
            gate_result: Associated gate result
            escalation: Escalation level
            notified_targets: List of notified targets

        Returns:
            Created audit record
        """
        with self._audit_lock:
            # Get previous hash for chain
            previous_hash = ""
            if self._audit_records:
                previous_hash = self._audit_records[-1].provenance_hash

            record = SafetyAuditRecord(
                event_type=f"VIOLATION_{violation.violation_type}",
                violation=violation,
                gate_result=gate_result,
                action_taken="ACTUATION_BLOCKED",
                operator_notified=escalation.value >= EscalationLevel.LEVEL_2.value,
                escalation_level=escalation.value,
                previous_hash=previous_hash,
            )

            self._audit_records.append(record)

            # Call audit callback if provided
            if self._audit_callback:
                try:
                    self._audit_callback(record)
                except Exception as e:
                    logger.error(f"Audit callback failed: {e}")

            return record

    def _track_violation_rate(self, violation: BoundaryViolation) -> None:
        """Track violation rate for storm detection."""
        with self._rate_lock:
            now = datetime.utcnow()
            key = f"{violation.policy_id}:{violation.tag_id}"
            self._violation_times[key].append(now)

            # Clean old entries (keep last 10 minutes)
            cutoff = now - timedelta(minutes=10)
            self._violation_times[key] = [
                t for t in self._violation_times[key] if t > cutoff
            ]

    def _check_violation_storm(
        self,
        violation: BoundaryViolation,
        threshold: int = 10,
        window_seconds: int = 60,
    ) -> None:
        """
        Check for violation storm condition.

        Args:
            violation: Current violation
            threshold: Number of violations to trigger storm
            window_seconds: Time window to check
        """
        with self._rate_lock:
            key = f"{violation.policy_id}:{violation.tag_id}"
            now = datetime.utcnow()
            cutoff = now - timedelta(seconds=window_seconds)

            recent = [t for t in self._violation_times[key] if t > cutoff]

            if len(recent) >= threshold:
                logger.critical(
                    f"VIOLATION STORM DETECTED: {len(recent)} violations "
                    f"in {window_seconds}s for {key}"
                )
                # In production, trigger additional escalation

    def add_notification_target(self, target: NotificationTarget) -> None:
        """
        Add a notification target.

        Args:
            target: Notification target to add
        """
        self._notification_targets[target.target_id] = target
        logger.info(f"Added notification target: {target.target_id}")

    def remove_notification_target(self, target_id: str) -> bool:
        """
        Remove a notification target.

        Args:
            target_id: Target ID to remove

        Returns:
            True if removed
        """
        if target_id in self._notification_targets:
            del self._notification_targets[target_id]
            logger.info(f"Removed notification target: {target_id}")
            return True
        return False

    def add_escalation_rule(self, rule: EscalationRule) -> None:
        """
        Add an escalation rule.

        Args:
            rule: Rule to add
        """
        self._escalation_rules.append(rule)
        logger.info(f"Added escalation rule: {rule.rule_id}")

    def get_audit_records(
        self,
        since: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[SafetyAuditRecord]:
        """
        Get audit records.

        Args:
            since: Only records after this timestamp
            limit: Maximum records to return

        Returns:
            List of audit records
        """
        with self._audit_lock:
            records = self._audit_records
            if since:
                records = [r for r in records if r.timestamp > since]
            return records[-limit:]

    def get_violation_events(
        self,
        since: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[ViolationEvent]:
        """
        Get violation events.

        Args:
            since: Only events after this timestamp
            limit: Maximum events to return

        Returns:
            List of violation events
        """
        with self._events_lock:
            events = self._violation_events
            if since:
                events = [e for e in events if e.timestamp > since]
            return events[-limit:]

    def verify_audit_chain(self) -> bool:
        """
        Verify integrity of audit record chain.

        Returns:
            True if chain is valid
        """
        with self._audit_lock:
            for i, record in enumerate(self._audit_records):
                if i == 0:
                    if record.previous_hash != "":
                        logger.error("First record should have empty previous_hash")
                        return False
                else:
                    expected = self._audit_records[i - 1].provenance_hash
                    if record.previous_hash != expected:
                        logger.error(f"Audit chain broken at record {i}")
                        return False
            return True

    def get_statistics(self) -> Dict[str, int]:
        """
        Get handler statistics.

        Returns:
            Dict of statistics
        """
        with self._stats_lock:
            return dict(self._stats)

    def get_violation_rate(
        self,
        policy_id: Optional[str] = None,
        tag_id: Optional[str] = None,
        window_seconds: int = 60,
    ) -> float:
        """
        Get violation rate per minute.

        Args:
            policy_id: Filter by policy ID
            tag_id: Filter by tag ID
            window_seconds: Time window

        Returns:
            Violations per minute
        """
        with self._rate_lock:
            now = datetime.utcnow()
            cutoff = now - timedelta(seconds=window_seconds)
            count = 0

            for key, times in self._violation_times.items():
                if policy_id and not key.startswith(policy_id):
                    continue
                if tag_id and not key.endswith(tag_id):
                    continue
                count += len([t for t in times if t > cutoff])

            # Convert to per minute
            return count * (60.0 / window_seconds)

    def export_audit_trail(self, format: str = "json") -> str:
        """
        Export audit trail.

        Args:
            format: Export format (json)

        Returns:
            Exported audit trail
        """
        with self._audit_lock:
            records = [r.dict() for r in self._audit_records]

        if format == "json":
            return json.dumps(records, default=str, indent=2)

        raise ValueError(f"Unsupported format: {format}")


class ViolationHandlerFactory:
    """
    Factory for creating Violation Handler instances.

    Example:
        >>> factory = ViolationHandlerFactory()
        >>> handler = factory.create_handler()
    """

    @staticmethod
    def create_handler(
        notification_targets: Optional[List[NotificationTarget]] = None,
        escalation_rules: Optional[List[EscalationRule]] = None,
    ) -> ViolationHandler:
        """
        Create a configured Violation Handler.

        Args:
            notification_targets: Notification targets
            escalation_rules: Escalation rules

        Returns:
            Configured ViolationHandler
        """
        # Add default log target
        targets = notification_targets or []
        targets.append(NotificationTarget(
            target_id="default_log",
            notification_type=NotificationType.LOG,
            endpoint="safety.violations",
            min_severity=ViolationSeverity.WARNING,
            escalation_level=EscalationLevel.LEVEL_0,
        ))

        return ViolationHandler(
            notification_targets=targets,
            escalation_rules=escalation_rules,
        )

    @staticmethod
    def create_production_handler() -> ViolationHandler:
        """
        Create production-ready handler with standard targets.

        Returns:
            Production ViolationHandler
        """
        targets = [
            NotificationTarget(
                target_id="log",
                notification_type=NotificationType.LOG,
                endpoint="safety.violations",
                min_severity=ViolationSeverity.WARNING,
                escalation_level=EscalationLevel.LEVEL_0,
            ),
            NotificationTarget(
                target_id="alarm_system",
                notification_type=NotificationType.ALARM,
                endpoint="alarm_manager",
                min_severity=ViolationSeverity.CRITICAL,
                escalation_level=EscalationLevel.LEVEL_1,
            ),
            NotificationTarget(
                target_id="historian",
                notification_type=NotificationType.HISTORIAN,
                endpoint="process_historian",
                min_severity=ViolationSeverity.WARNING,
                escalation_level=EscalationLevel.LEVEL_0,
            ),
            NotificationTarget(
                target_id="event_bus",
                notification_type=NotificationType.EVENT,
                endpoint="event_broker",
                min_severity=ViolationSeverity.WARNING,
                escalation_level=EscalationLevel.LEVEL_0,
            ),
        ]

        return ViolationHandler(notification_targets=targets)
