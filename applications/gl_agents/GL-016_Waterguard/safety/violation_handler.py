"""
GL-016 Waterguard Violation Handler - IEC 61511 SIL-3 Compliant

This module implements safety violation detection, escalation, and
response for the WATERGUARD system. It provides tiered escalation
and automatic CMMS work order generation.

Key Features:
    - Tiered violation severity (INFO, LOW, MEDIUM, HIGH, CRITICAL)
    - Automatic escalation based on persistence and severity
    - CMMS work order generation for maintenance tracking
    - Violation report generation for compliance audits
    - Complete audit trail with SHA-256 hashes
    - Integration with emergency shutdown handler

Escalation Tiers:
    1. Operator notification (immediate)
    2. Shift supervisor alert (5 minutes unresolved)
    3. Plant manager notification (15 minutes)
    4. Automatic protective action (severity-dependent)
    5. CMMS work order generation (all violations)

Reference Standards:
    - IEC 61511-1:2016 Functional Safety
    - ISA-18.2 Alarm Management
    - OSHA Process Safety Management

Author: GreenLang Safety Engineering Team
Version: 1.0.0
SIL Level: 3
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum, IntEnum
from typing import Any, Callable, Dict, List, Optional, Tuple
import uuid

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMERATIONS
# =============================================================================


class ViolationSeverity(IntEnum):
    """Severity levels for violations."""
    INFO = 1          # Informational only
    LOW = 2           # Minor deviation
    MEDIUM = 3        # Significant deviation
    HIGH = 4          # Serious violation
    CRITICAL = 5      # Critical safety violation


class ViolationType(str, Enum):
    """Types of safety violations."""
    CONSTRAINT_EXCEEDED = "constraint_exceeded"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    BOUNDARY_VIOLATION = "boundary_violation"
    COMMUNICATION_FAILURE = "communication_failure"
    SENSOR_FAULT = "sensor_fault"
    ACTUATOR_FAULT = "actuator_fault"
    WATCHDOG_TIMEOUT = "watchdog_timeout"
    MANUAL_OVERRIDE = "manual_override"
    CHEMISTRY_LIMIT = "chemistry_limit"
    OEM_LIMIT_EXCEEDED = "oem_limit_exceeded"
    CALIBRATION_OVERDUE = "calibration_overdue"
    MAINTENANCE_OVERDUE = "maintenance_overdue"
    PROCEDURE_DEVIATION = "procedure_deviation"
    CONFIGURATION_ERROR = "configuration_error"
    SECURITY_VIOLATION = "security_violation"


class ViolationState(str, Enum):
    """State of a violation."""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"
    ESCALATED = "escalated"


class EscalationLevel(IntEnum):
    """Escalation levels."""
    NONE = 0
    OPERATOR = 1
    SUPERVISOR = 2
    MANAGER = 3
    EMERGENCY = 4


class WorkOrderPriority(IntEnum):
    """CMMS work order priority."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    URGENT = 4
    EMERGENCY = 5


class WorkOrderStatus(str, Enum):
    """Status of a CMMS work order."""
    DRAFT = "draft"
    SUBMITTED = "submitted"
    ACCEPTED = "accepted"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


# =============================================================================
# DATA MODELS
# =============================================================================


class ViolationRecord(BaseModel):
    """A safety violation record."""

    violation_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique violation identifier"
    )
    violation_type: ViolationType = Field(
        ...,
        description="Type of violation"
    )
    severity: ViolationSeverity = Field(
        ...,
        description="Severity level"
    )
    state: ViolationState = Field(
        default=ViolationState.ACTIVE,
        description="Current state"
    )
    source: str = Field(
        default="WATERGUARD",
        description="Violation source"
    )
    tag: str = Field(
        default="",
        description="Related equipment tag"
    )
    parameter: str = Field(
        default="",
        description="Parameter name"
    )
    actual_value: Optional[float] = Field(
        default=None,
        description="Actual value"
    )
    limit_value: Optional[float] = Field(
        default=None,
        description="Limit that was exceeded"
    )
    engineering_units: str = Field(
        default="",
        description="Engineering units"
    )
    message: str = Field(
        default="",
        description="Violation message"
    )
    recommended_action: str = Field(
        default="",
        description="Recommended action"
    )

    # Timing
    first_occurrence: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="First occurrence time"
    )
    last_occurrence: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Last occurrence time"
    )
    occurrence_count: int = Field(
        default=1,
        ge=1,
        description="Number of occurrences"
    )
    duration_seconds: float = Field(
        default=0.0,
        ge=0.0,
        description="Duration of violation"
    )

    # Escalation
    escalation_level: EscalationLevel = Field(
        default=EscalationLevel.NONE,
        description="Current escalation level"
    )
    escalated_at: Optional[datetime] = Field(
        default=None,
        description="Last escalation time"
    )
    escalated_to: List[str] = Field(
        default_factory=list,
        description="People escalated to"
    )

    # Resolution
    acknowledged_by: Optional[str] = Field(
        default=None,
        description="Person who acknowledged"
    )
    acknowledged_at: Optional[datetime] = Field(
        default=None,
        description="Acknowledgement time"
    )
    resolved_by: Optional[str] = Field(
        default=None,
        description="Person who resolved"
    )
    resolved_at: Optional[datetime] = Field(
        default=None,
        description="Resolution time"
    )
    resolution_notes: str = Field(
        default="",
        description="Resolution notes"
    )

    # Work order
    work_order_id: Optional[str] = Field(
        default=None,
        description="CMMS work order ID"
    )

    # Audit
    provenance_hash: str = Field(
        default="",
        description="SHA-256 hash for audit"
    )

    def model_post_init(self, __context: Any) -> None:
        """Calculate provenance hash."""
        if not self.provenance_hash:
            hash_str = (
                f"{self.violation_id}|{self.violation_type.value}|"
                f"{self.severity}|{self.tag}|"
                f"{self.first_occurrence.isoformat()}"
            )
            self.provenance_hash = hashlib.sha256(hash_str.encode()).hexdigest()


class CMMSWorkOrder(BaseModel):
    """A CMMS work order for maintenance tracking."""

    work_order_id: str = Field(
        default_factory=lambda: f"WO-{str(uuid.uuid4())[:8].upper()}",
        description="Work order ID"
    )
    violation_id: str = Field(
        ...,
        description="Related violation ID"
    )
    title: str = Field(
        ...,
        description="Work order title"
    )
    description: str = Field(
        default="",
        description="Detailed description"
    )
    priority: WorkOrderPriority = Field(
        default=WorkOrderPriority.MEDIUM,
        description="Priority level"
    )
    status: WorkOrderStatus = Field(
        default=WorkOrderStatus.DRAFT,
        description="Current status"
    )

    # Equipment
    equipment_tag: str = Field(
        default="",
        description="Equipment tag"
    )
    equipment_description: str = Field(
        default="",
        description="Equipment description"
    )
    location: str = Field(
        default="",
        description="Equipment location"
    )

    # Work details
    work_type: str = Field(
        default="corrective",
        description="Work type (corrective, preventive, inspection)"
    )
    estimated_hours: float = Field(
        default=1.0,
        ge=0.0,
        description="Estimated work hours"
    )
    required_skills: List[str] = Field(
        default_factory=list,
        description="Required skills/certifications"
    )
    spare_parts: List[str] = Field(
        default_factory=list,
        description="Required spare parts"
    )

    # Timing
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Creation time"
    )
    due_date: Optional[datetime] = Field(
        default=None,
        description="Due date"
    )
    started_at: Optional[datetime] = Field(
        default=None,
        description="Start time"
    )
    completed_at: Optional[datetime] = Field(
        default=None,
        description="Completion time"
    )

    # Assignment
    assigned_to: Optional[str] = Field(
        default=None,
        description="Assigned technician"
    )
    created_by: str = Field(
        default="WATERGUARD",
        description="Created by"
    )

    # Completion
    completion_notes: str = Field(
        default="",
        description="Completion notes"
    )
    actual_hours: Optional[float] = Field(
        default=None,
        description="Actual hours worked"
    )

    # Audit
    provenance_hash: str = Field(
        default="",
        description="SHA-256 hash for audit"
    )

    def model_post_init(self, __context: Any) -> None:
        """Calculate provenance hash."""
        if not self.provenance_hash:
            hash_str = (
                f"{self.work_order_id}|{self.violation_id}|"
                f"{self.priority}|{self.created_at.isoformat()}"
            )
            self.provenance_hash = hashlib.sha256(hash_str.encode()).hexdigest()


class ViolationReport(BaseModel):
    """A violation report for compliance audits."""

    report_id: str = Field(
        default_factory=lambda: f"VR-{str(uuid.uuid4())[:8].upper()}",
        description="Report ID"
    )
    report_type: str = Field(
        default="incident",
        description="Report type"
    )
    title: str = Field(
        ...,
        description="Report title"
    )
    generated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Generation time"
    )
    generated_by: str = Field(
        default="WATERGUARD",
        description="Generated by"
    )

    # Scope
    start_time: datetime = Field(
        ...,
        description="Report start time"
    )
    end_time: datetime = Field(
        ...,
        description="Report end time"
    )

    # Summary
    total_violations: int = Field(
        default=0,
        ge=0,
        description="Total violations"
    )
    violations_by_severity: Dict[str, int] = Field(
        default_factory=dict,
        description="Count by severity"
    )
    violations_by_type: Dict[str, int] = Field(
        default_factory=dict,
        description="Count by type"
    )

    # Details
    violations: List[ViolationRecord] = Field(
        default_factory=list,
        description="Violation records"
    )
    work_orders: List[CMMSWorkOrder] = Field(
        default_factory=list,
        description="Work orders generated"
    )

    # Analysis
    root_causes: List[str] = Field(
        default_factory=list,
        description="Identified root causes"
    )
    corrective_actions: List[str] = Field(
        default_factory=list,
        description="Corrective actions taken"
    )
    recommendations: List[str] = Field(
        default_factory=list,
        description="Recommendations"
    )

    # Audit
    provenance_hash: str = Field(
        default="",
        description="SHA-256 hash for audit"
    )

    def model_post_init(self, __context: Any) -> None:
        """Calculate provenance hash."""
        if not self.provenance_hash:
            hash_str = (
                f"{self.report_id}|{self.total_violations}|"
                f"{self.generated_at.isoformat()}"
            )
            self.provenance_hash = hashlib.sha256(hash_str.encode()).hexdigest()


class EscalationConfig(BaseModel):
    """Configuration for escalation behavior."""

    severity: ViolationSeverity = Field(
        ...,
        description="Severity level"
    )
    initial_level: EscalationLevel = Field(
        default=EscalationLevel.OPERATOR,
        description="Initial escalation level"
    )
    supervisor_delay_minutes: int = Field(
        default=5,
        ge=1,
        description="Minutes before supervisor escalation"
    )
    manager_delay_minutes: int = Field(
        default=15,
        ge=1,
        description="Minutes before manager escalation"
    )
    emergency_delay_minutes: int = Field(
        default=30,
        ge=1,
        description="Minutes before emergency escalation"
    )
    auto_work_order: bool = Field(
        default=True,
        description="Auto-generate work order"
    )


# =============================================================================
# VIOLATION HANDLER
# =============================================================================


class ViolationHandler:
    """
    Handles safety violation detection, escalation, and response.

    This handler provides comprehensive violation management including:
        - Detection and logging of safety violations
        - Tiered escalation based on severity and persistence
        - Automatic CMMS work order generation
        - Violation reporting for compliance audits
        - Integration with emergency shutdown handler

    Escalation Logic:
        - INFO: Log only, no escalation
        - LOW: Operator notification, work order after 30 minutes
        - MEDIUM: Immediate operator, supervisor at 5 minutes
        - HIGH: Immediate supervisor, manager at 10 minutes
        - CRITICAL: Immediate manager, emergency action

    Example:
        >>> handler = ViolationHandler()
        >>> violation = handler.record_violation(
        ...     ViolationType.CHEMISTRY_LIMIT,
        ...     ViolationSeverity.HIGH,
        ...     tag="CT-001",
        ...     parameter="conductivity",
        ...     actual_value=3600,
        ...     limit_value=3500,
        ...     message="Conductivity exceeds limit"
        ... )
        >>> handler.escalate_violation(violation.violation_id)
    """

    # Default escalation configurations by severity
    DEFAULT_ESCALATION_CONFIGS = {
        ViolationSeverity.INFO: EscalationConfig(
            severity=ViolationSeverity.INFO,
            initial_level=EscalationLevel.NONE,
            auto_work_order=False
        ),
        ViolationSeverity.LOW: EscalationConfig(
            severity=ViolationSeverity.LOW,
            initial_level=EscalationLevel.OPERATOR,
            supervisor_delay_minutes=30,
            manager_delay_minutes=60,
            auto_work_order=True
        ),
        ViolationSeverity.MEDIUM: EscalationConfig(
            severity=ViolationSeverity.MEDIUM,
            initial_level=EscalationLevel.OPERATOR,
            supervisor_delay_minutes=5,
            manager_delay_minutes=15,
            auto_work_order=True
        ),
        ViolationSeverity.HIGH: EscalationConfig(
            severity=ViolationSeverity.HIGH,
            initial_level=EscalationLevel.SUPERVISOR,
            supervisor_delay_minutes=0,
            manager_delay_minutes=10,
            auto_work_order=True
        ),
        ViolationSeverity.CRITICAL: EscalationConfig(
            severity=ViolationSeverity.CRITICAL,
            initial_level=EscalationLevel.MANAGER,
            supervisor_delay_minutes=0,
            manager_delay_minutes=0,
            emergency_delay_minutes=5,
            auto_work_order=True
        ),
    }

    def __init__(
        self,
        emergency_handler: Optional[Any] = None,
        cmms_callback: Optional[Callable[[CMMSWorkOrder], None]] = None
    ) -> None:
        """
        Initialize ViolationHandler.

        Args:
            emergency_handler: EmergencyShutdownHandler for critical violations
            cmms_callback: Callback for CMMS work order submission
        """
        self._emergency_handler = emergency_handler
        self._cmms_callback = cmms_callback

        self._lock = threading.Lock()

        # Active violations (violation_id -> ViolationRecord)
        self._active_violations: Dict[str, ViolationRecord] = {}

        # History
        self._violation_history: List[ViolationRecord] = []
        self._work_order_history: List[CMMSWorkOrder] = []
        self._max_history = 1000

        # Escalation configs
        self._escalation_configs = dict(self.DEFAULT_ESCALATION_CONFIGS)

        # Statistics
        self._stats = {
            "violations_total": 0,
            "violations_resolved": 0,
            "work_orders_generated": 0,
            "escalations_total": 0,
            "by_severity": {},
            "by_type": {},
        }

        # Callbacks
        self._on_violation: Optional[Callable[[ViolationRecord], None]] = None
        self._on_escalation: Optional[Callable[[ViolationRecord, EscalationLevel], None]] = None
        self._on_resolution: Optional[Callable[[ViolationRecord], None]] = None

        logger.info("ViolationHandler initialized")

    def record_violation(
        self,
        violation_type: ViolationType,
        severity: ViolationSeverity,
        tag: str = "",
        parameter: str = "",
        actual_value: Optional[float] = None,
        limit_value: Optional[float] = None,
        engineering_units: str = "",
        message: str = "",
        recommended_action: str = ""
    ) -> ViolationRecord:
        """
        Record a safety violation.

        Args:
            violation_type: Type of violation
            severity: Severity level
            tag: Equipment tag
            parameter: Parameter name
            actual_value: Actual value
            limit_value: Limit that was exceeded
            engineering_units: Engineering units
            message: Violation message
            recommended_action: Recommended action

        Returns:
            ViolationRecord
        """
        # Check for existing active violation (same type, tag, parameter)
        existing = self._find_existing_violation(violation_type, tag, parameter)

        if existing:
            # Update existing violation
            with self._lock:
                existing.occurrence_count += 1
                existing.last_occurrence = datetime.now(timezone.utc)
                if actual_value is not None:
                    existing.actual_value = actual_value
                existing.duration_seconds = (
                    existing.last_occurrence - existing.first_occurrence
                ).total_seconds()

                logger.warning(
                    "Violation recurring: %s on %s (count: %d, duration: %.1fs)",
                    violation_type.value, tag,
                    existing.occurrence_count, existing.duration_seconds
                )

                # Check if escalation needed
                self._check_auto_escalation(existing)

                return existing

        # Create new violation
        violation = ViolationRecord(
            violation_type=violation_type,
            severity=severity,
            tag=tag,
            parameter=parameter,
            actual_value=actual_value,
            limit_value=limit_value,
            engineering_units=engineering_units,
            message=message,
            recommended_action=recommended_action
        )

        with self._lock:
            self._active_violations[violation.violation_id] = violation
            self._violation_history.append(violation)
            if len(self._violation_history) > self._max_history:
                self._violation_history = self._violation_history[-self._max_history:]

            # Update statistics
            self._stats["violations_total"] += 1
            sev_key = str(severity)
            self._stats["by_severity"][sev_key] = (
                self._stats["by_severity"].get(sev_key, 0) + 1
            )
            type_key = violation_type.value
            self._stats["by_type"][type_key] = (
                self._stats["by_type"].get(type_key, 0) + 1
            )

        # Log violation
        log_level = {
            ViolationSeverity.INFO: logging.INFO,
            ViolationSeverity.LOW: logging.WARNING,
            ViolationSeverity.MEDIUM: logging.WARNING,
            ViolationSeverity.HIGH: logging.ERROR,
            ViolationSeverity.CRITICAL: logging.CRITICAL,
        }.get(severity, logging.WARNING)

        logger.log(
            log_level,
            "VIOLATION [%s]: %s - %s (tag: %s, value: %s, limit: %s) - %s",
            severity.name,
            violation_type.value,
            message,
            tag,
            actual_value,
            limit_value,
            violation.provenance_hash[:16]
        )

        # Initial escalation based on severity
        config = self._escalation_configs.get(severity)
        if config and config.initial_level != EscalationLevel.NONE:
            self._do_escalation(violation, config.initial_level)

        # Callback
        if self._on_violation:
            try:
                self._on_violation(violation)
            except Exception as e:
                logger.error("Violation callback failed: %s", e)

        # Check for emergency escalation
        if severity == ViolationSeverity.CRITICAL and self._emergency_handler:
            from .emergency_shutdown import EmergencyType, EmergencySeverity
            self._emergency_handler.trigger_estop(
                EmergencyType.CONSTRAINT_VIOLATION,
                EmergencySeverity.CRITICAL,
                message,
                trigger_tag=tag,
                trigger_value=actual_value,
                trigger_limit=limit_value
            )

        return violation

    def _find_existing_violation(
        self,
        violation_type: ViolationType,
        tag: str,
        parameter: str
    ) -> Optional[ViolationRecord]:
        """Find an existing active violation matching criteria."""
        with self._lock:
            for violation in self._active_violations.values():
                if (
                    violation.violation_type == violation_type and
                    violation.tag == tag and
                    violation.parameter == parameter and
                    violation.state == ViolationState.ACTIVE
                ):
                    return violation
        return None

    def _check_auto_escalation(self, violation: ViolationRecord) -> None:
        """Check if violation should be auto-escalated."""
        config = self._escalation_configs.get(violation.severity)
        if not config:
            return

        duration_minutes = violation.duration_seconds / 60

        # Check escalation thresholds
        if (
            violation.escalation_level < EscalationLevel.SUPERVISOR and
            duration_minutes >= config.supervisor_delay_minutes
        ):
            self._do_escalation(violation, EscalationLevel.SUPERVISOR)

        elif (
            violation.escalation_level < EscalationLevel.MANAGER and
            duration_minutes >= config.manager_delay_minutes
        ):
            self._do_escalation(violation, EscalationLevel.MANAGER)

        elif (
            violation.escalation_level < EscalationLevel.EMERGENCY and
            duration_minutes >= config.emergency_delay_minutes
        ):
            self._do_escalation(violation, EscalationLevel.EMERGENCY)

    def escalate_violation(
        self,
        violation_id: str,
        target_level: Optional[EscalationLevel] = None
    ) -> bool:
        """
        Manually escalate a violation.

        Args:
            violation_id: Violation ID
            target_level: Target escalation level (or next level if None)

        Returns:
            True if escalated
        """
        with self._lock:
            violation = self._active_violations.get(violation_id)
            if not violation:
                logger.warning("Violation not found: %s", violation_id[:8])
                return False

            if target_level is None:
                target_level = EscalationLevel(
                    min(violation.escalation_level + 1, EscalationLevel.EMERGENCY)
                )

            if target_level <= violation.escalation_level:
                logger.warning(
                    "Cannot escalate to same or lower level: %s",
                    target_level.name
                )
                return False

            self._do_escalation(violation, target_level)
            return True

    def _do_escalation(
        self,
        violation: ViolationRecord,
        level: EscalationLevel
    ) -> None:
        """Perform escalation to specified level."""
        old_level = violation.escalation_level
        violation.escalation_level = level
        violation.escalated_at = datetime.now(timezone.utc)
        violation.state = ViolationState.ESCALATED

        self._stats["escalations_total"] += 1

        logger.warning(
            "ESCALATION: %s -> %s for violation %s (%s on %s)",
            old_level.name, level.name,
            violation.violation_id[:8],
            violation.violation_type.value,
            violation.tag
        )

        # Callback
        if self._on_escalation:
            try:
                self._on_escalation(violation, level)
            except Exception as e:
                logger.error("Escalation callback failed: %s", e)

        # Generate work order if configured
        config = self._escalation_configs.get(violation.severity)
        if config and config.auto_work_order and not violation.work_order_id:
            self.generate_work_order(violation.violation_id)

    def acknowledge_violation(
        self,
        violation_id: str,
        acknowledged_by: str
    ) -> bool:
        """
        Acknowledge a violation.

        Args:
            violation_id: Violation ID
            acknowledged_by: Person acknowledging

        Returns:
            True if acknowledged
        """
        with self._lock:
            violation = self._active_violations.get(violation_id)
            if not violation:
                logger.warning("Violation not found: %s", violation_id[:8])
                return False

            if violation.state not in (ViolationState.ACTIVE, ViolationState.ESCALATED):
                logger.warning(
                    "Cannot acknowledge violation in state: %s",
                    violation.state.value
                )
                return False

            violation.state = ViolationState.ACKNOWLEDGED
            violation.acknowledged_by = acknowledged_by
            violation.acknowledged_at = datetime.now(timezone.utc)

            logger.info(
                "Violation %s acknowledged by %s",
                violation_id[:8], acknowledged_by
            )

            return True

    def resolve_violation(
        self,
        violation_id: str,
        resolved_by: str,
        resolution_notes: str = ""
    ) -> bool:
        """
        Resolve a violation.

        Args:
            violation_id: Violation ID
            resolved_by: Person resolving
            resolution_notes: Resolution notes

        Returns:
            True if resolved
        """
        with self._lock:
            violation = self._active_violations.get(violation_id)
            if not violation:
                logger.warning("Violation not found: %s", violation_id[:8])
                return False

            if violation.state not in (
                ViolationState.ACTIVE,
                ViolationState.ACKNOWLEDGED,
                ViolationState.ESCALATED
            ):
                logger.warning(
                    "Cannot resolve violation in state: %s",
                    violation.state.value
                )
                return False

            violation.state = ViolationState.RESOLVED
            violation.resolved_by = resolved_by
            violation.resolved_at = datetime.now(timezone.utc)
            violation.resolution_notes = resolution_notes

            # Remove from active
            del self._active_violations[violation_id]
            self._stats["violations_resolved"] += 1

            logger.info(
                "Violation %s resolved by %s: %s",
                violation_id[:8], resolved_by, resolution_notes[:100]
            )

            # Callback
            if self._on_resolution:
                try:
                    self._on_resolution(violation)
                except Exception as e:
                    logger.error("Resolution callback failed: %s", e)

            return True

    def generate_work_order(
        self,
        violation_id: str,
        additional_notes: str = ""
    ) -> Optional[CMMSWorkOrder]:
        """
        Generate a CMMS work order for a violation.

        Args:
            violation_id: Violation ID
            additional_notes: Additional notes

        Returns:
            CMMSWorkOrder or None
        """
        with self._lock:
            violation = self._active_violations.get(violation_id)
            if not violation:
                # Check history
                for v in self._violation_history:
                    if v.violation_id == violation_id:
                        violation = v
                        break

            if not violation:
                logger.warning("Violation not found: %s", violation_id[:8])
                return None

            if violation.work_order_id:
                logger.warning(
                    "Work order already exists: %s",
                    violation.work_order_id
                )
                return None

        # Map severity to priority
        priority_map = {
            ViolationSeverity.INFO: WorkOrderPriority.LOW,
            ViolationSeverity.LOW: WorkOrderPriority.LOW,
            ViolationSeverity.MEDIUM: WorkOrderPriority.MEDIUM,
            ViolationSeverity.HIGH: WorkOrderPriority.HIGH,
            ViolationSeverity.CRITICAL: WorkOrderPriority.EMERGENCY,
        }

        # Calculate due date based on priority
        due_days = {
            WorkOrderPriority.LOW: 30,
            WorkOrderPriority.MEDIUM: 14,
            WorkOrderPriority.HIGH: 7,
            WorkOrderPriority.URGENT: 3,
            WorkOrderPriority.EMERGENCY: 1,
        }

        priority = priority_map.get(violation.severity, WorkOrderPriority.MEDIUM)
        due_date = datetime.now(timezone.utc) + timedelta(days=due_days.get(priority, 14))

        # Build description
        description = (
            f"Safety Violation: {violation.violation_type.value}\n"
            f"Severity: {violation.severity.name}\n"
            f"Equipment: {violation.tag}\n"
            f"Parameter: {violation.parameter}\n"
            f"Actual Value: {violation.actual_value} {violation.engineering_units}\n"
            f"Limit Value: {violation.limit_value} {violation.engineering_units}\n"
            f"\nMessage: {violation.message}\n"
            f"\nRecommended Action: {violation.recommended_action}\n"
            f"\nFirst Occurrence: {violation.first_occurrence.isoformat()}\n"
            f"Occurrence Count: {violation.occurrence_count}\n"
            f"\nViolation ID: {violation.violation_id}\n"
            f"Provenance Hash: {violation.provenance_hash}\n"
        )
        if additional_notes:
            description += f"\nAdditional Notes: {additional_notes}\n"

        work_order = CMMSWorkOrder(
            violation_id=violation.violation_id,
            title=f"[{violation.severity.name}] {violation.violation_type.value} - {violation.tag}",
            description=description,
            priority=priority,
            equipment_tag=violation.tag,
            due_date=due_date,
            work_type="corrective"
        )

        with self._lock:
            violation.work_order_id = work_order.work_order_id
            self._work_order_history.append(work_order)
            if len(self._work_order_history) > self._max_history:
                self._work_order_history = self._work_order_history[-self._max_history:]
            self._stats["work_orders_generated"] += 1

        logger.info(
            "Work order generated: %s for violation %s (priority: %s)",
            work_order.work_order_id,
            violation_id[:8],
            priority.name
        )

        # Callback to CMMS
        if self._cmms_callback:
            try:
                self._cmms_callback(work_order)
            except Exception as e:
                logger.error("CMMS callback failed: %s", e)

        return work_order

    def generate_violation_report(
        self,
        start_time: datetime,
        end_time: datetime,
        report_type: str = "incident"
    ) -> ViolationReport:
        """
        Generate a violation report for a time period.

        Args:
            start_time: Report start time
            end_time: Report end time
            report_type: Report type

        Returns:
            ViolationReport
        """
        with self._lock:
            # Filter violations by time
            violations = [
                v for v in self._violation_history
                if start_time <= v.first_occurrence <= end_time
            ]

            # Filter work orders by time
            work_orders = [
                wo for wo in self._work_order_history
                if start_time <= wo.created_at <= end_time
            ]

        # Calculate summaries
        by_severity = {}
        by_type = {}
        for v in violations:
            sev_key = v.severity.name
            by_severity[sev_key] = by_severity.get(sev_key, 0) + 1
            type_key = v.violation_type.value
            by_type[type_key] = by_type.get(type_key, 0) + 1

        report = ViolationReport(
            report_type=report_type,
            title=f"Violation Report {start_time.date()} to {end_time.date()}",
            start_time=start_time,
            end_time=end_time,
            total_violations=len(violations),
            violations_by_severity=by_severity,
            violations_by_type=by_type,
            violations=violations,
            work_orders=work_orders
        )

        logger.info(
            "Violation report generated: %s (%d violations, %d work orders)",
            report.report_id,
            report.total_violations,
            len(work_orders)
        )

        return report

    def get_active_violations(
        self,
        severity: Optional[ViolationSeverity] = None,
        violation_type: Optional[ViolationType] = None
    ) -> List[ViolationRecord]:
        """Get active violations with optional filtering."""
        with self._lock:
            violations = list(self._active_violations.values())
            if severity:
                violations = [v for v in violations if v.severity == severity]
            if violation_type:
                violations = [v for v in violations if v.violation_type == violation_type]
            return violations

    def get_violation_history(
        self,
        limit: int = 100,
        severity: Optional[ViolationSeverity] = None
    ) -> List[ViolationRecord]:
        """Get violation history."""
        with self._lock:
            violations = self._violation_history
            if severity:
                violations = [v for v in violations if v.severity == severity]
            return list(reversed(violations[-limit:]))

    def get_work_order_history(self, limit: int = 100) -> List[CMMSWorkOrder]:
        """Get work order history."""
        with self._lock:
            return list(reversed(self._work_order_history[-limit:]))

    def get_statistics(self) -> Dict[str, Any]:
        """Get handler statistics."""
        with self._lock:
            return {
                **self._stats,
                "active_violations": len(self._active_violations),
                "pending_work_orders": len([
                    wo for wo in self._work_order_history
                    if wo.status not in (
                        WorkOrderStatus.COMPLETED,
                        WorkOrderStatus.CANCELLED
                    )
                ]),
            }

    def set_escalation_config(
        self,
        severity: ViolationSeverity,
        config: EscalationConfig
    ) -> None:
        """Set escalation configuration for a severity level."""
        self._escalation_configs[severity] = config

    def set_on_violation(
        self,
        callback: Callable[[ViolationRecord], None]
    ) -> None:
        """Set callback for new violations."""
        self._on_violation = callback

    def set_on_escalation(
        self,
        callback: Callable[[ViolationRecord, EscalationLevel], None]
    ) -> None:
        """Set callback for escalations."""
        self._on_escalation = callback

    def set_on_resolution(
        self,
        callback: Callable[[ViolationRecord], None]
    ) -> None:
        """Set callback for resolutions."""
        self._on_resolution = callback

    def set_cmms_callback(
        self,
        callback: Callable[[CMMSWorkOrder], None]
    ) -> None:
        """Set callback for CMMS work order submission."""
        self._cmms_callback = callback


# =============================================================================
# MODULE EXPORTS
# =============================================================================


__all__ = [
    # Enums
    "ViolationSeverity",
    "ViolationType",
    "ViolationState",
    "EscalationLevel",
    "WorkOrderPriority",
    "WorkOrderStatus",
    # Models
    "ViolationRecord",
    "CMMSWorkOrder",
    "ViolationReport",
    "EscalationConfig",
    # Classes
    "ViolationHandler",
]
