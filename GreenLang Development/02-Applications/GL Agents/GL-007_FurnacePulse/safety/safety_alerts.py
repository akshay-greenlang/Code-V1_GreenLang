"""
Safety Alert Manager - GL-007_FurnacePulse Safety Module

This module implements a comprehensive safety alert system with a defined
taxonomy, severity levels, detection logic, escalation workflows, and
response playbook integration.

Alert Taxonomy:
    - A-001: Hotspot Advisory (low severity, early indication)
    - A-002: Hotspot Warning (medium severity, requires attention)
    - A-003: Hotspot Urgent (high severity, immediate action required)
    - A-010: Efficiency Degradation
    - A-020: Draft Instability
    - A-030: Sensor Drift

Each alert includes severity, detection logic, confidence score,
recommended actions, and owner role assignments.

Example:
    >>> config = SafetyAlertConfig(...)
    >>> manager = SafetyAlertManager(config)
    >>> alert = manager.create_alert(
    ...     alert_code="A-002",
    ...     furnace_id="FRN-001",
    ...     detection_data={"hotspot_temp": 1250, "baseline_temp": 1150}
    ... )
    >>> manager.escalate_alert(alert.alert_id, "SHIFT_SUPERVISOR")
"""

from typing import Dict, List, Optional, Any, Callable
from pydantic import BaseModel, Field, validator
from enum import Enum
from datetime import datetime, timezone, timedelta
from uuid import uuid4
import hashlib
import logging
import json

logger = logging.getLogger(__name__)


# =============================================================================
# Enums and Constants
# =============================================================================

class AlertSeverity(str, Enum):
    """Alert severity levels."""
    ADVISORY = "advisory"  # Information only
    WARNING = "warning"  # Requires monitoring
    URGENT = "urgent"  # Immediate attention
    CRITICAL = "critical"  # Emergency response


class AlertStatus(str, Enum):
    """Alert lifecycle status."""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    IN_PROGRESS = "in_progress"
    RESOLVED = "resolved"
    ESCALATED = "escalated"
    SUPPRESSED = "suppressed"
    EXPIRED = "expired"


class OwnerRole(str, Enum):
    """Owner role assignments."""
    OPERATOR = "operator"
    SHIFT_SUPERVISOR = "shift_supervisor"
    MAINTENANCE_TECH = "maintenance_tech"
    PROCESS_ENGINEER = "process_engineer"
    SAFETY_ENGINEER = "safety_engineer"
    PLANT_MANAGER = "plant_manager"
    EMERGENCY_RESPONSE = "emergency_response"


class EscalationLevel(int, Enum):
    """Escalation levels."""
    LEVEL_0 = 0  # Initial
    LEVEL_1 = 1  # Supervisor
    LEVEL_2 = 2  # Engineer
    LEVEL_3 = 3  # Manager
    LEVEL_4 = 4  # Emergency


# =============================================================================
# Alert Definitions
# =============================================================================

ALERT_TAXONOMY: Dict[str, Dict[str, Any]] = {
    "A-001": {
        "name": "Hotspot Advisory",
        "description": "Early indication of potential hotspot development",
        "category": "thermal",
        "severity": AlertSeverity.ADVISORY,
        "default_owner": OwnerRole.OPERATOR,
        "auto_escalation_minutes": 60,
        "detection_parameters": {
            "temp_delta_threshold": 50,  # degrees above baseline
            "confidence_threshold": 0.6,
        },
        "recommended_actions": [
            "Monitor affected zone on IR display",
            "Review thermal trend for past 4 hours",
            "Check for feed rate or fuel changes",
            "Note in shift log",
        ],
        "playbook_reference": "PB-THERMAL-001",
    },
    "A-002": {
        "name": "Hotspot Warning",
        "description": "Confirmed hotspot requiring operator attention",
        "category": "thermal",
        "severity": AlertSeverity.WARNING,
        "default_owner": OwnerRole.OPERATOR,
        "auto_escalation_minutes": 30,
        "detection_parameters": {
            "temp_delta_threshold": 100,
            "confidence_threshold": 0.75,
            "sustained_duration_minutes": 5,
        },
        "recommended_actions": [
            "Reduce firing rate in affected zone by 10%",
            "Adjust combustion air to affected burners",
            "Check refractory inspection schedule",
            "Notify shift supervisor if persists >15 min",
            "Prepare thermal scan report",
        ],
        "playbook_reference": "PB-THERMAL-002",
    },
    "A-003": {
        "name": "Hotspot Urgent",
        "description": "Critical hotspot requiring immediate action",
        "category": "thermal",
        "severity": AlertSeverity.URGENT,
        "default_owner": OwnerRole.SHIFT_SUPERVISOR,
        "auto_escalation_minutes": 15,
        "detection_parameters": {
            "temp_delta_threshold": 150,
            "confidence_threshold": 0.85,
            "sustained_duration_minutes": 2,
        },
        "recommended_actions": [
            "IMMEDIATE: Reduce zone firing rate by 25%",
            "Shut down affected burner if temp continues rising",
            "Notify maintenance for emergency refractory inspection",
            "Prepare for controlled shutdown if > 200C delta",
            "Activate emergency response team standby",
            "Document all actions with timestamps",
        ],
        "playbook_reference": "PB-THERMAL-003",
    },
    "A-010": {
        "name": "Efficiency Degradation",
        "description": "Furnace efficiency below acceptable threshold",
        "category": "efficiency",
        "severity": AlertSeverity.WARNING,
        "default_owner": OwnerRole.PROCESS_ENGINEER,
        "auto_escalation_minutes": 120,
        "detection_parameters": {
            "efficiency_drop_percent": 5,
            "baseline_period_hours": 168,  # 1 week
            "minimum_confidence": 0.7,
        },
        "recommended_actions": [
            "Review combustion analysis data",
            "Check fuel/air ratio calibration",
            "Inspect for air infiltration",
            "Review waste heat recovery performance",
            "Schedule burner tune-up if persistent",
        ],
        "playbook_reference": "PB-EFFICIENCY-001",
    },
    "A-020": {
        "name": "Draft Instability",
        "description": "Furnace draft pressure outside stable range",
        "category": "combustion",
        "severity": AlertSeverity.WARNING,
        "default_owner": OwnerRole.OPERATOR,
        "auto_escalation_minutes": 20,
        "detection_parameters": {
            "draft_variance_threshold": 0.1,  # inches WC
            "oscillation_frequency_hz": 0.5,
            "sustained_duration_seconds": 30,
        },
        "recommended_actions": [
            "Check ID/FD fan operation",
            "Verify damper positions",
            "Inspect for air leaks in furnace shell",
            "Check barometric damper operation",
            "Review recent combustion changes",
        ],
        "playbook_reference": "PB-DRAFT-001",
    },
    "A-030": {
        "name": "Sensor Drift",
        "description": "Sensor readings showing abnormal drift pattern",
        "category": "instrumentation",
        "severity": AlertSeverity.ADVISORY,
        "default_owner": OwnerRole.MAINTENANCE_TECH,
        "auto_escalation_minutes": 240,
        "detection_parameters": {
            "drift_rate_threshold": 0.5,  # % per hour
            "correlation_threshold": 0.3,  # vs reference sensors
            "confidence_threshold": 0.65,
        },
        "recommended_actions": [
            "Compare with redundant sensors",
            "Check sensor calibration date",
            "Schedule calibration verification",
            "Review environmental factors",
            "Update data quality flags if confirmed",
        ],
        "playbook_reference": "PB-SENSOR-001",
    },
}


# =============================================================================
# Pydantic Models
# =============================================================================

class SafetyAlertConfig(BaseModel):
    """Configuration for Safety Alert Manager."""

    site_id: str = Field(..., description="Site identifier")
    enable_auto_escalation: bool = Field(
        default=True, description="Enable automatic escalation"
    )
    enable_auto_suppression: bool = Field(
        default=False, description="Enable duplicate suppression"
    )
    suppression_window_minutes: int = Field(
        default=15, ge=1, le=60,
        description="Window for duplicate suppression"
    )
    notification_channels: List[str] = Field(
        default_factory=lambda: ["email", "sms", "control_room"],
        description="Enabled notification channels"
    )
    shift_schedule_path: Optional[str] = Field(
        None, description="Path to shift schedule for routing"
    )
    audit_retention_days: int = Field(
        default=365, ge=30, le=3650,
        description="Audit record retention period"
    )


class DetectionResult(BaseModel):
    """Result of alert detection logic."""

    detected: bool = Field(..., description="Whether condition was detected")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Detection confidence"
    )
    detection_data: Dict[str, Any] = Field(
        ..., description="Data used in detection"
    )
    threshold_data: Dict[str, Any] = Field(
        ..., description="Threshold configuration"
    )
    explanation: str = Field(..., description="Human-readable explanation")


class RecommendedAction(BaseModel):
    """Recommended action with tracking."""

    action_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique action ID"
    )
    sequence: int = Field(..., description="Action sequence order")
    description: str = Field(..., description="Action description")
    completed: bool = Field(default=False, description="Completion status")
    completed_by: Optional[str] = Field(None, description="Completer user ID")
    completed_at: Optional[datetime] = Field(None, description="Completion time")
    notes: Optional[str] = Field(None, description="Action notes")


class EscalationRecord(BaseModel):
    """Record of alert escalation."""

    escalation_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique escalation ID"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Escalation timestamp"
    )
    from_level: EscalationLevel = Field(..., description="Previous level")
    to_level: EscalationLevel = Field(..., description="New level")
    from_owner: OwnerRole = Field(..., description="Previous owner role")
    to_owner: OwnerRole = Field(..., description="New owner role")
    reason: str = Field(..., description="Escalation reason")
    escalated_by: str = Field(..., description="User or SYSTEM")
    is_auto: bool = Field(..., description="Automatic escalation flag")


class SafetyAlert(BaseModel):
    """Safety alert record."""

    alert_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique alert ID"
    )
    alert_code: str = Field(..., description="Alert taxonomy code")
    name: str = Field(..., description="Alert name")
    description: str = Field(..., description="Alert description")
    category: str = Field(..., description="Alert category")
    furnace_id: str = Field(..., description="Associated furnace ID")
    asset_id: Optional[str] = Field(None, description="Specific asset ID")
    zone_id: Optional[str] = Field(None, description="Furnace zone ID")

    # Severity and Status
    severity: AlertSeverity = Field(..., description="Alert severity")
    status: AlertStatus = Field(
        default=AlertStatus.ACTIVE, description="Current status"
    )
    escalation_level: EscalationLevel = Field(
        default=EscalationLevel.LEVEL_0, description="Current escalation level"
    )

    # Detection
    detection_result: DetectionResult = Field(..., description="Detection details")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence")

    # Ownership
    owner_role: OwnerRole = Field(..., description="Responsible role")
    assigned_to: Optional[str] = Field(None, description="Specific assignee user ID")

    # Actions
    recommended_actions: List[RecommendedAction] = Field(
        ..., description="Recommended actions"
    )
    playbook_reference: str = Field(..., description="Response playbook ID")

    # Timeline
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Creation timestamp"
    )
    acknowledged_at: Optional[datetime] = Field(
        None, description="Acknowledgement timestamp"
    )
    acknowledged_by: Optional[str] = Field(None, description="Acknowledger user ID")
    resolved_at: Optional[datetime] = Field(None, description="Resolution timestamp")
    resolved_by: Optional[str] = Field(None, description="Resolver user ID")
    resolution_notes: Optional[str] = Field(None, description="Resolution notes")

    # Escalation
    escalation_history: List[EscalationRecord] = Field(
        default_factory=list, description="Escalation history"
    )
    auto_escalation_due: Optional[datetime] = Field(
        None, description="Auto-escalation deadline"
    )

    # Audit
    data_hash: str = Field(..., description="SHA-256 hash for integrity")


class AlertAuditEntry(BaseModel):
    """Audit log entry for alert activities."""

    audit_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique audit ID"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Event timestamp"
    )
    alert_id: str = Field(..., description="Associated alert ID")
    action: str = Field(..., description="Action performed")
    actor_id: str = Field(..., description="Actor user or system ID")
    details: Dict[str, Any] = Field(default_factory=dict, description="Action details")
    data_hash: str = Field(..., description="SHA-256 hash of change data")


class AlertSummaryDashboard(BaseModel):
    """Dashboard summary of alerts."""

    furnace_id: str = Field(..., description="Furnace ID")
    generated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Generation timestamp"
    )
    active_alerts_count: int = Field(..., description="Count of active alerts")
    by_severity: Dict[str, int] = Field(..., description="Counts by severity")
    by_category: Dict[str, int] = Field(..., description="Counts by category")
    by_status: Dict[str, int] = Field(..., description="Counts by status")
    escalated_count: int = Field(..., description="Count of escalated alerts")
    overdue_acknowledgements: List[Dict[str, Any]] = Field(
        ..., description="Alerts awaiting acknowledgement"
    )
    recent_alerts: List[Dict[str, Any]] = Field(
        ..., description="Most recent alerts"
    )
    mean_time_to_acknowledge_minutes: Optional[float] = Field(
        None, description="MTTA in minutes"
    )
    mean_time_to_resolve_minutes: Optional[float] = Field(
        None, description="MTTR in minutes"
    )


# =============================================================================
# Escalation Workflow
# =============================================================================

ESCALATION_MATRIX: Dict[EscalationLevel, Dict[str, Any]] = {
    EscalationLevel.LEVEL_0: {
        "owner_mapping": {
            AlertSeverity.ADVISORY: OwnerRole.OPERATOR,
            AlertSeverity.WARNING: OwnerRole.OPERATOR,
            AlertSeverity.URGENT: OwnerRole.SHIFT_SUPERVISOR,
            AlertSeverity.CRITICAL: OwnerRole.SHIFT_SUPERVISOR,
        },
        "next_level": EscalationLevel.LEVEL_1,
    },
    EscalationLevel.LEVEL_1: {
        "owner_mapping": {
            AlertSeverity.ADVISORY: OwnerRole.SHIFT_SUPERVISOR,
            AlertSeverity.WARNING: OwnerRole.SHIFT_SUPERVISOR,
            AlertSeverity.URGENT: OwnerRole.PROCESS_ENGINEER,
            AlertSeverity.CRITICAL: OwnerRole.SAFETY_ENGINEER,
        },
        "next_level": EscalationLevel.LEVEL_2,
    },
    EscalationLevel.LEVEL_2: {
        "owner_mapping": {
            AlertSeverity.ADVISORY: OwnerRole.PROCESS_ENGINEER,
            AlertSeverity.WARNING: OwnerRole.PROCESS_ENGINEER,
            AlertSeverity.URGENT: OwnerRole.SAFETY_ENGINEER,
            AlertSeverity.CRITICAL: OwnerRole.PLANT_MANAGER,
        },
        "next_level": EscalationLevel.LEVEL_3,
    },
    EscalationLevel.LEVEL_3: {
        "owner_mapping": {
            AlertSeverity.ADVISORY: OwnerRole.PLANT_MANAGER,
            AlertSeverity.WARNING: OwnerRole.PLANT_MANAGER,
            AlertSeverity.URGENT: OwnerRole.PLANT_MANAGER,
            AlertSeverity.CRITICAL: OwnerRole.EMERGENCY_RESPONSE,
        },
        "next_level": EscalationLevel.LEVEL_4,
    },
    EscalationLevel.LEVEL_4: {
        "owner_mapping": {
            AlertSeverity.ADVISORY: OwnerRole.EMERGENCY_RESPONSE,
            AlertSeverity.WARNING: OwnerRole.EMERGENCY_RESPONSE,
            AlertSeverity.URGENT: OwnerRole.EMERGENCY_RESPONSE,
            AlertSeverity.CRITICAL: OwnerRole.EMERGENCY_RESPONSE,
        },
        "next_level": None,  # Maximum level
    },
}


# =============================================================================
# Safety Alert Manager
# =============================================================================

class SafetyAlertManager:
    """
    Safety Alert Manager for industrial furnace operations.

    This manager handles alert creation, acknowledgement, escalation,
    resolution, and audit logging. It implements a configurable alert
    taxonomy with severity levels and automated escalation workflows.

    Attributes:
        config: Alert manager configuration
        alerts: Active and historical alerts
        audit_log: Audit trail for all alert activities

    Example:
        >>> config = SafetyAlertConfig(site_id="SITE-001")
        >>> manager = SafetyAlertManager(config)
        >>> alert = manager.create_alert(
        ...     alert_code="A-002",
        ...     furnace_id="FRN-001",
        ...     detection_data={"hotspot_temp": 1250, "baseline_temp": 1150}
        ... )
    """

    def __init__(self, config: SafetyAlertConfig):
        """
        Initialize SafetyAlertManager.

        Args:
            config: Alert manager configuration
        """
        self.config = config
        self.alerts: Dict[str, SafetyAlert] = {}
        self.audit_log: List[AlertAuditEntry] = []
        self._suppression_cache: Dict[str, datetime] = {}

        logger.info(f"SafetyAlertManager initialized for site {config.site_id}")

    def create_alert(
        self,
        alert_code: str,
        furnace_id: str,
        detection_data: Dict[str, Any],
        asset_id: Optional[str] = None,
        zone_id: Optional[str] = None,
        created_by: str = "SYSTEM"
    ) -> SafetyAlert:
        """
        Create a new safety alert.

        Args:
            alert_code: Alert taxonomy code (e.g., "A-001")
            furnace_id: Furnace identifier
            detection_data: Data from detection logic
            asset_id: Optional specific asset ID
            zone_id: Optional furnace zone ID
            created_by: Creator user or system ID

        Returns:
            Created SafetyAlert

        Raises:
            ValueError: If alert code not found in taxonomy
        """
        if alert_code not in ALERT_TAXONOMY:
            raise ValueError(f"Unknown alert code: {alert_code}")

        # Check for suppression
        suppression_key = f"{alert_code}:{furnace_id}:{zone_id or 'all'}"
        if self._should_suppress(suppression_key):
            logger.info(f"Alert suppressed (duplicate): {suppression_key}")
            # Return the existing active alert
            for alert in self.alerts.values():
                if (alert.alert_code == alert_code and
                    alert.furnace_id == furnace_id and
                    alert.status == AlertStatus.ACTIVE):
                    return alert

        taxonomy = ALERT_TAXONOMY[alert_code]
        start_time = datetime.now(timezone.utc)

        # Run detection logic
        detection_result = self._run_detection_logic(alert_code, detection_data)

        # Create recommended actions
        recommended_actions = [
            RecommendedAction(sequence=i+1, description=action)
            for i, action in enumerate(taxonomy["recommended_actions"])
        ]

        # Calculate auto-escalation deadline
        auto_escalation_due = None
        if self.config.enable_auto_escalation and taxonomy.get("auto_escalation_minutes"):
            auto_escalation_due = start_time + timedelta(
                minutes=taxonomy["auto_escalation_minutes"]
            )

        # Create alert data for hash
        alert_data = {
            "alert_code": alert_code,
            "furnace_id": furnace_id,
            "detection_data": detection_data,
            "created_at": start_time.isoformat(),
        }
        data_hash = hashlib.sha256(
            json.dumps(alert_data, sort_keys=True, default=str).encode()
        ).hexdigest()

        # Create alert
        alert = SafetyAlert(
            alert_code=alert_code,
            name=taxonomy["name"],
            description=taxonomy["description"],
            category=taxonomy["category"],
            furnace_id=furnace_id,
            asset_id=asset_id,
            zone_id=zone_id,
            severity=taxonomy["severity"],
            detection_result=detection_result,
            confidence=detection_result.confidence,
            owner_role=taxonomy["default_owner"],
            recommended_actions=recommended_actions,
            playbook_reference=taxonomy["playbook_reference"],
            auto_escalation_due=auto_escalation_due,
            data_hash=data_hash,
        )

        self.alerts[alert.alert_id] = alert

        # Update suppression cache
        if self.config.enable_auto_suppression:
            self._suppression_cache[suppression_key] = start_time

        # Log audit
        self._log_audit(
            alert_id=alert.alert_id,
            action="ALERT_CREATED",
            actor_id=created_by,
            details={
                "alert_code": alert_code,
                "severity": taxonomy["severity"].value,
                "confidence": detection_result.confidence,
                "furnace_id": furnace_id,
            }
        )

        processing_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
        logger.warning(
            f"Alert created: {alert_code} ({taxonomy['name']}) for {furnace_id} "
            f"[{taxonomy['severity'].value}] in {processing_time:.2f}ms"
        )

        return alert

    def acknowledge_alert(
        self,
        alert_id: str,
        acknowledged_by: str,
        notes: Optional[str] = None
    ) -> SafetyAlert:
        """
        Acknowledge a safety alert.

        Args:
            alert_id: Alert identifier
            acknowledged_by: User acknowledging
            notes: Optional acknowledgement notes

        Returns:
            Updated SafetyAlert

        Raises:
            ValueError: If alert not found
        """
        if alert_id not in self.alerts:
            raise ValueError(f"Alert {alert_id} not found")

        alert = self.alerts[alert_id]
        if alert.status not in [AlertStatus.ACTIVE, AlertStatus.ESCALATED]:
            logger.warning(f"Alert {alert_id} cannot be acknowledged (status: {alert.status})")
            return alert

        old_status = alert.status
        alert.status = AlertStatus.ACKNOWLEDGED
        alert.acknowledged_at = datetime.now(timezone.utc)
        alert.acknowledged_by = acknowledged_by

        self._log_audit(
            alert_id=alert_id,
            action="ALERT_ACKNOWLEDGED",
            actor_id=acknowledged_by,
            details={
                "old_status": old_status.value,
                "notes": notes,
                "time_to_acknowledge_minutes": (
                    (alert.acknowledged_at - alert.created_at).total_seconds() / 60
                ),
            }
        )

        logger.info(f"Alert acknowledged: {alert_id} by {acknowledged_by}")
        return alert

    def start_work_on_alert(
        self,
        alert_id: str,
        worker_id: str
    ) -> SafetyAlert:
        """
        Mark alert as work in progress.

        Args:
            alert_id: Alert identifier
            worker_id: User working on alert

        Returns:
            Updated SafetyAlert

        Raises:
            ValueError: If alert not found
        """
        if alert_id not in self.alerts:
            raise ValueError(f"Alert {alert_id} not found")

        alert = self.alerts[alert_id]
        old_status = alert.status
        alert.status = AlertStatus.IN_PROGRESS
        alert.assigned_to = worker_id

        self._log_audit(
            alert_id=alert_id,
            action="ALERT_WORK_STARTED",
            actor_id=worker_id,
            details={"old_status": old_status.value}
        )

        logger.info(f"Alert work started: {alert_id} by {worker_id}")
        return alert

    def complete_action(
        self,
        alert_id: str,
        action_id: str,
        completed_by: str,
        notes: Optional[str] = None
    ) -> SafetyAlert:
        """
        Mark a recommended action as completed.

        Args:
            alert_id: Alert identifier
            action_id: Action identifier
            completed_by: User completing action
            notes: Optional completion notes

        Returns:
            Updated SafetyAlert

        Raises:
            ValueError: If alert or action not found
        """
        if alert_id not in self.alerts:
            raise ValueError(f"Alert {alert_id} not found")

        alert = self.alerts[alert_id]
        action_found = False

        for action in alert.recommended_actions:
            if action.action_id == action_id:
                action.completed = True
                action.completed_by = completed_by
                action.completed_at = datetime.now(timezone.utc)
                action.notes = notes
                action_found = True
                break

        if not action_found:
            raise ValueError(f"Action {action_id} not found in alert {alert_id}")

        self._log_audit(
            alert_id=alert_id,
            action="ACTION_COMPLETED",
            actor_id=completed_by,
            details={
                "action_id": action_id,
                "notes": notes,
                "completed_actions": sum(
                    1 for a in alert.recommended_actions if a.completed
                ),
                "total_actions": len(alert.recommended_actions),
            }
        )

        logger.info(f"Action completed: {action_id} for alert {alert_id}")
        return alert

    def resolve_alert(
        self,
        alert_id: str,
        resolved_by: str,
        resolution_notes: str
    ) -> SafetyAlert:
        """
        Resolve a safety alert.

        Args:
            alert_id: Alert identifier
            resolved_by: User resolving
            resolution_notes: Resolution notes (required)

        Returns:
            Updated SafetyAlert

        Raises:
            ValueError: If alert not found
        """
        if alert_id not in self.alerts:
            raise ValueError(f"Alert {alert_id} not found")

        alert = self.alerts[alert_id]
        old_status = alert.status
        alert.status = AlertStatus.RESOLVED
        alert.resolved_at = datetime.now(timezone.utc)
        alert.resolved_by = resolved_by
        alert.resolution_notes = resolution_notes

        # Calculate metrics
        time_to_resolve = (alert.resolved_at - alert.created_at).total_seconds() / 60
        time_to_acknowledge = None
        if alert.acknowledged_at:
            time_to_acknowledge = (
                alert.acknowledged_at - alert.created_at
            ).total_seconds() / 60

        self._log_audit(
            alert_id=alert_id,
            action="ALERT_RESOLVED",
            actor_id=resolved_by,
            details={
                "old_status": old_status.value,
                "resolution_notes": resolution_notes,
                "time_to_resolve_minutes": time_to_resolve,
                "time_to_acknowledge_minutes": time_to_acknowledge,
                "actions_completed": sum(
                    1 for a in alert.recommended_actions if a.completed
                ),
            }
        )

        logger.info(
            f"Alert resolved: {alert_id} by {resolved_by} "
            f"(TTR: {time_to_resolve:.1f} min)"
        )
        return alert

    def escalate_alert(
        self,
        alert_id: str,
        reason: str,
        escalated_by: str = "SYSTEM",
        is_auto: bool = False
    ) -> SafetyAlert:
        """
        Escalate an alert to the next level.

        Args:
            alert_id: Alert identifier
            reason: Escalation reason
            escalated_by: User or SYSTEM
            is_auto: Whether this is automatic escalation

        Returns:
            Updated SafetyAlert

        Raises:
            ValueError: If alert not found or at max escalation
        """
        if alert_id not in self.alerts:
            raise ValueError(f"Alert {alert_id} not found")

        alert = self.alerts[alert_id]

        # Get current and next escalation level
        current_level = alert.escalation_level
        current_config = ESCALATION_MATRIX.get(current_level)

        if not current_config or current_config["next_level"] is None:
            logger.warning(f"Alert {alert_id} already at maximum escalation level")
            return alert

        next_level = current_config["next_level"]
        next_config = ESCALATION_MATRIX[next_level]

        # Determine new owner
        old_owner = alert.owner_role
        new_owner = next_config["owner_mapping"].get(alert.severity, OwnerRole.PLANT_MANAGER)

        # Create escalation record
        escalation = EscalationRecord(
            from_level=current_level,
            to_level=next_level,
            from_owner=old_owner,
            to_owner=new_owner,
            reason=reason,
            escalated_by=escalated_by,
            is_auto=is_auto,
        )

        # Update alert
        alert.escalation_level = next_level
        alert.owner_role = new_owner
        alert.status = AlertStatus.ESCALATED
        alert.escalation_history.append(escalation)
        alert.assigned_to = None  # Reset assignment for new owner

        # Update auto-escalation deadline
        taxonomy = ALERT_TAXONOMY.get(alert.alert_code, {})
        if self.config.enable_auto_escalation and taxonomy.get("auto_escalation_minutes"):
            alert.auto_escalation_due = datetime.now(timezone.utc) + timedelta(
                minutes=taxonomy["auto_escalation_minutes"]
            )

        self._log_audit(
            alert_id=alert_id,
            action="ALERT_ESCALATED",
            actor_id=escalated_by,
            details={
                "from_level": current_level.value,
                "to_level": next_level.value,
                "from_owner": old_owner.value,
                "to_owner": new_owner.value,
                "reason": reason,
                "is_auto": is_auto,
            }
        )

        logger.warning(
            f"Alert escalated: {alert_id} to Level {next_level.value} "
            f"(Owner: {new_owner.value}) - {reason}"
        )
        return alert

    def check_auto_escalations(self) -> List[SafetyAlert]:
        """
        Check for alerts needing automatic escalation.

        Returns:
            List of alerts that were auto-escalated
        """
        now = datetime.now(timezone.utc)
        escalated = []

        for alert in self.alerts.values():
            if alert.status not in [AlertStatus.ACTIVE, AlertStatus.ESCALATED]:
                continue

            if alert.auto_escalation_due and alert.auto_escalation_due <= now:
                if alert.status != AlertStatus.ACKNOWLEDGED:
                    self.escalate_alert(
                        alert_id=alert.alert_id,
                        reason="Auto-escalation: Acknowledgement timeout",
                        is_auto=True,
                    )
                    escalated.append(alert)

        if escalated:
            logger.info(f"Auto-escalated {len(escalated)} alerts")

        return escalated

    def suppress_alert(
        self,
        alert_id: str,
        suppressed_by: str,
        reason: str,
        duration_minutes: int = 60
    ) -> SafetyAlert:
        """
        Suppress an alert for a specified duration.

        Args:
            alert_id: Alert identifier
            suppressed_by: User suppressing
            reason: Suppression reason
            duration_minutes: Suppression duration

        Returns:
            Updated SafetyAlert

        Raises:
            ValueError: If alert not found
        """
        if alert_id not in self.alerts:
            raise ValueError(f"Alert {alert_id} not found")

        alert = self.alerts[alert_id]
        old_status = alert.status
        alert.status = AlertStatus.SUPPRESSED

        self._log_audit(
            alert_id=alert_id,
            action="ALERT_SUPPRESSED",
            actor_id=suppressed_by,
            details={
                "old_status": old_status.value,
                "reason": reason,
                "duration_minutes": duration_minutes,
            }
        )

        logger.info(
            f"Alert suppressed: {alert_id} for {duration_minutes} min - {reason}"
        )
        return alert

    def get_alert(self, alert_id: str) -> Optional[SafetyAlert]:
        """Get alert by ID."""
        return self.alerts.get(alert_id)

    def get_active_alerts(
        self,
        furnace_id: Optional[str] = None,
        severity: Optional[AlertSeverity] = None,
        category: Optional[str] = None
    ) -> List[SafetyAlert]:
        """
        Get active alerts with optional filters.

        Args:
            furnace_id: Optional furnace filter
            severity: Optional severity filter
            category: Optional category filter

        Returns:
            List of matching active alerts
        """
        active_statuses = [AlertStatus.ACTIVE, AlertStatus.ACKNOWLEDGED,
                          AlertStatus.IN_PROGRESS, AlertStatus.ESCALATED]

        alerts = [
            a for a in self.alerts.values()
            if a.status in active_statuses
        ]

        if furnace_id:
            alerts = [a for a in alerts if a.furnace_id == furnace_id]
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        if category:
            alerts = [a for a in alerts if a.category == category]

        return sorted(alerts, key=lambda a: a.created_at, reverse=True)

    def get_alert_summary_dashboard(self, furnace_id: str) -> AlertSummaryDashboard:
        """
        Generate alert summary dashboard data.

        Args:
            furnace_id: Furnace identifier

        Returns:
            AlertSummaryDashboard with comprehensive metrics
        """
        furnace_alerts = [
            a for a in self.alerts.values()
            if a.furnace_id == furnace_id
        ]

        # Count active alerts
        active_statuses = [AlertStatus.ACTIVE, AlertStatus.ACKNOWLEDGED,
                          AlertStatus.IN_PROGRESS, AlertStatus.ESCALATED]
        active_alerts = [a for a in furnace_alerts if a.status in active_statuses]

        # By severity
        by_severity = {}
        for severity in AlertSeverity:
            count = sum(1 for a in active_alerts if a.severity == severity)
            if count > 0:
                by_severity[severity.value] = count

        # By category
        by_category = {}
        for alert in active_alerts:
            by_category[alert.category] = by_category.get(alert.category, 0) + 1

        # By status
        by_status = {}
        for status in AlertStatus:
            count = sum(1 for a in furnace_alerts if a.status == status)
            if count > 0:
                by_status[status.value] = count

        # Escalated count
        escalated_count = sum(
            1 for a in active_alerts
            if a.escalation_level.value > 0
        )

        # Overdue acknowledgements (active but not acknowledged within threshold)
        overdue_acks = []
        now = datetime.now(timezone.utc)
        for alert in active_alerts:
            if alert.status == AlertStatus.ACTIVE and alert.auto_escalation_due:
                if now > alert.auto_escalation_due:
                    overdue_acks.append({
                        "alert_id": alert.alert_id,
                        "alert_code": alert.alert_code,
                        "name": alert.name,
                        "created_at": alert.created_at.isoformat(),
                        "overdue_minutes": (now - alert.auto_escalation_due).total_seconds() / 60,
                    })

        # Recent alerts (last 24 hours)
        twenty_four_hours_ago = now - timedelta(hours=24)
        recent_alerts = [
            {
                "alert_id": a.alert_id,
                "alert_code": a.alert_code,
                "name": a.name,
                "severity": a.severity.value,
                "status": a.status.value,
                "created_at": a.created_at.isoformat(),
            }
            for a in furnace_alerts
            if a.created_at >= twenty_four_hours_ago
        ][:10]  # Limit to 10

        # Calculate MTTA and MTTR
        resolved_alerts = [a for a in furnace_alerts if a.status == AlertStatus.RESOLVED]

        mtta = None
        if resolved_alerts:
            ack_times = [
                (a.acknowledged_at - a.created_at).total_seconds() / 60
                for a in resolved_alerts
                if a.acknowledged_at
            ]
            if ack_times:
                mtta = sum(ack_times) / len(ack_times)

        mttr = None
        if resolved_alerts:
            resolve_times = [
                (a.resolved_at - a.created_at).total_seconds() / 60
                for a in resolved_alerts
                if a.resolved_at
            ]
            if resolve_times:
                mttr = sum(resolve_times) / len(resolve_times)

        return AlertSummaryDashboard(
            furnace_id=furnace_id,
            active_alerts_count=len(active_alerts),
            by_severity=by_severity,
            by_category=by_category,
            by_status=by_status,
            escalated_count=escalated_count,
            overdue_acknowledgements=overdue_acks,
            recent_alerts=recent_alerts,
            mean_time_to_acknowledge_minutes=round(mtta, 2) if mtta else None,
            mean_time_to_resolve_minutes=round(mttr, 2) if mttr else None,
        )

    def get_playbook_reference(self, alert_code: str) -> Optional[Dict[str, Any]]:
        """
        Get playbook reference for an alert code.

        Args:
            alert_code: Alert taxonomy code

        Returns:
            Playbook reference information or None
        """
        if alert_code not in ALERT_TAXONOMY:
            return None

        taxonomy = ALERT_TAXONOMY[alert_code]
        return {
            "playbook_id": taxonomy["playbook_reference"],
            "alert_code": alert_code,
            "alert_name": taxonomy["name"],
            "severity": taxonomy["severity"].value,
            "recommended_actions": taxonomy["recommended_actions"],
            "detection_parameters": taxonomy["detection_parameters"],
        }

    def get_all_playbooks(self) -> List[Dict[str, Any]]:
        """Get all playbook references."""
        return [
            self.get_playbook_reference(code)
            for code in ALERT_TAXONOMY.keys()
        ]

    def _run_detection_logic(
        self,
        alert_code: str,
        detection_data: Dict[str, Any]
    ) -> DetectionResult:
        """
        Run detection logic for alert.

        Args:
            alert_code: Alert taxonomy code
            detection_data: Input data for detection

        Returns:
            DetectionResult with confidence and explanation
        """
        taxonomy = ALERT_TAXONOMY[alert_code]
        thresholds = taxonomy["detection_parameters"]

        # Default detection result
        detected = True
        confidence = detection_data.get("confidence", 0.7)
        explanation_parts = []

        # Check specific thresholds based on category
        if taxonomy["category"] == "thermal":
            temp_delta = detection_data.get("temp_delta", 0)
            threshold = thresholds.get("temp_delta_threshold", 100)
            confidence_threshold = thresholds.get("confidence_threshold", 0.7)

            detected = temp_delta >= threshold and confidence >= confidence_threshold
            explanation_parts.append(
                f"Temperature delta {temp_delta}C vs threshold {threshold}C"
            )

        elif taxonomy["category"] == "efficiency":
            efficiency_drop = detection_data.get("efficiency_drop_percent", 0)
            threshold = thresholds.get("efficiency_drop_percent", 5)

            detected = efficiency_drop >= threshold
            explanation_parts.append(
                f"Efficiency drop {efficiency_drop}% vs threshold {threshold}%"
            )

        elif taxonomy["category"] == "combustion":
            variance = detection_data.get("draft_variance", 0)
            threshold = thresholds.get("draft_variance_threshold", 0.1)

            detected = variance >= threshold
            explanation_parts.append(
                f"Draft variance {variance} vs threshold {threshold}"
            )

        elif taxonomy["category"] == "instrumentation":
            drift_rate = detection_data.get("drift_rate", 0)
            threshold = thresholds.get("drift_rate_threshold", 0.5)

            detected = drift_rate >= threshold
            explanation_parts.append(
                f"Drift rate {drift_rate}%/hr vs threshold {threshold}%/hr"
            )

        explanation = "; ".join(explanation_parts) if explanation_parts else "Detection criteria met"

        return DetectionResult(
            detected=detected,
            confidence=confidence,
            detection_data=detection_data,
            threshold_data=thresholds,
            explanation=explanation,
        )

    def _should_suppress(self, suppression_key: str) -> bool:
        """Check if alert should be suppressed as duplicate."""
        if not self.config.enable_auto_suppression:
            return False

        if suppression_key in self._suppression_cache:
            last_time = self._suppression_cache[suppression_key]
            window = timedelta(minutes=self.config.suppression_window_minutes)
            if datetime.now(timezone.utc) - last_time < window:
                return True

        return False

    def _log_audit(
        self,
        alert_id: str,
        action: str,
        actor_id: str,
        details: Dict[str, Any]
    ) -> None:
        """Add entry to audit log with integrity hash."""
        details_str = json.dumps(details, sort_keys=True, default=str)
        data_hash = hashlib.sha256(details_str.encode()).hexdigest()

        entry = AlertAuditEntry(
            alert_id=alert_id,
            action=action,
            actor_id=actor_id,
            details=details,
            data_hash=data_hash,
        )
        self.audit_log.append(entry)

    def get_audit_log(
        self,
        alert_id: Optional[str] = None,
        action_filter: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[AlertAuditEntry]:
        """
        Retrieve audit log with optional filters.

        Args:
            alert_id: Optional alert ID filter
            action_filter: Optional action type filter
            start_time: Optional start time filter
            end_time: Optional end time filter

        Returns:
            List of matching AlertAuditEntry records
        """
        filtered = self.audit_log

        if alert_id:
            filtered = [e for e in filtered if e.alert_id == alert_id]
        if action_filter:
            filtered = [e for e in filtered if e.action == action_filter]
        if start_time:
            filtered = [e for e in filtered if e.timestamp >= start_time]
        if end_time:
            filtered = [e for e in filtered if e.timestamp <= end_time]

        return filtered
