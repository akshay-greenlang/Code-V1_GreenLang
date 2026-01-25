"""
AuditLogger - Comprehensive audit logging for BURNMASTER operations.

This module implements the AuditLogger for GL-004 BURNMASTER, providing
immutable audit records for all system operations including recommendations,
setpoint writes, mode transitions, safety events, and operator actions.

All audit records include cryptographic hashes for integrity verification
and support regulatory compliance requirements for industrial combustion systems.

Example:
    >>> logger = AuditLogger(config)
    >>> record = logger.log_recommendation(rec, context)
    >>> assert record.integrity_hash is not None
"""

from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field, validator
from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
import logging
import uuid

logger = logging.getLogger(__name__)


# =============================================================================
# Enums
# =============================================================================

class AuditEventType(str, Enum):
    """Types of auditable events in BURNMASTER."""
    RECOMMENDATION = "recommendation"
    SETPOINT_WRITE = "setpoint_write"
    MODE_TRANSITION = "mode_transition"
    SAFETY_EVENT = "safety_event"
    OPERATOR_ACTION = "operator_action"
    SYSTEM_EVENT = "system_event"


class AuditSeverity(str, Enum):
    """Severity levels for audit events."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class SafetyEventType(str, Enum):
    """Types of safety events."""
    LIMIT_EXCEEDED = "limit_exceeded"
    EMERGENCY_SHUTDOWN = "emergency_shutdown"
    INTERLOCK_TRIGGERED = "interlock_triggered"
    ALARM_ACTIVATED = "alarm_activated"
    CONSTRAINT_VIOLATION = "constraint_violation"


class OperatorActionType(str, Enum):
    """Types of operator actions."""
    MANUAL_OVERRIDE = "manual_override"
    SETPOINT_CHANGE = "setpoint_change"
    MODE_CHANGE = "mode_change"
    ACKNOWLEDGMENT = "acknowledgment"
    CONFIGURATION_CHANGE = "configuration_change"


# =============================================================================
# Input Models
# =============================================================================

class Recommendation(BaseModel):
    """Recommendation from the optimization engine."""

    recommendation_id: str = Field(..., description="Unique recommendation identifier")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    recommendation_type: str = Field(..., description="Type of recommendation")
    target_setpoints: Dict[str, float] = Field(..., description="Recommended setpoint values")
    expected_improvement: Dict[str, float] = Field(..., description="Expected metric improvements")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Confidence in recommendation")
    model_id: str = Field(..., description="Model that generated recommendation")
    rationale: str = Field(..., description="Explanation for recommendation")
    constraints_satisfied: bool = Field(..., description="Whether all constraints are satisfied")

    @validator('recommendation_id')
    def validate_recommendation_id(cls, v: str) -> str:
        """Validate recommendation ID format."""
        if not v or len(v) < 8:
            raise ValueError("Recommendation ID must be at least 8 characters")
        return v


class SetpointWrite(BaseModel):
    """Record of a setpoint write operation."""

    write_id: str = Field(..., description="Unique write operation identifier")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    setpoint_name: str = Field(..., description="Name of setpoint being written")
    previous_value: float = Field(..., description="Value before write")
    new_value: float = Field(..., description="Value being written")
    recommendation_id: Optional[str] = Field(None, description="Linked recommendation if any")
    write_source: str = Field(..., description="Source of write: auto or manual")
    validation_passed: bool = Field(..., description="Whether validation checks passed")


class ModeTransition(BaseModel):
    """Record of a system mode transition."""

    transition_id: str = Field(..., description="Unique transition identifier")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    from_mode: str = Field(..., description="Mode transitioning from")
    to_mode: str = Field(..., description="Mode transitioning to")
    trigger: str = Field(..., description="What triggered the transition")
    operator_id: Optional[str] = Field(None, description="Operator if manual transition")
    automatic: bool = Field(..., description="Whether transition was automatic")
    safety_checks_passed: bool = Field(..., description="Whether all safety checks passed")


class SafetyEvent(BaseModel):
    """Record of a safety-related event."""

    event_id: str = Field(..., description="Unique event identifier")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    event_type: SafetyEventType = Field(..., description="Type of safety event")
    severity: AuditSeverity = Field(..., description="Event severity")
    description: str = Field(..., description="Event description")
    affected_equipment: List[str] = Field(default_factory=list, description="Equipment affected")
    sensor_readings: Dict[str, float] = Field(default_factory=dict, description="Relevant sensor values")
    action_taken: str = Field(..., description="Action taken in response")
    resolution_status: str = Field(..., description="Current resolution status")


class OperatorAction(BaseModel):
    """Record of an operator action."""

    action_id: str = Field(..., description="Unique action identifier")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    operator_id: str = Field(..., description="Operator identifier")
    action_type: OperatorActionType = Field(..., description="Type of action")
    description: str = Field(..., description="Action description")
    target: str = Field(..., description="Target of action")
    previous_state: Optional[Dict[str, Any]] = Field(None, description="State before action")
    new_state: Optional[Dict[str, Any]] = Field(None, description="State after action")
    authorization_level: str = Field(..., description="Authorization level required")
    justification: Optional[str] = Field(None, description="Operator justification")


class AuditContext(BaseModel):
    """Context information for audit records."""

    session_id: str = Field(..., description="Current session identifier")
    user_id: Optional[str] = Field(None, description="User identifier if applicable")
    system_state: Dict[str, Any] = Field(default_factory=dict, description="Current system state")
    data_snapshot_hash: Optional[str] = Field(None, description="Hash of data snapshot")
    model_version: Optional[str] = Field(None, description="Current model version")
    code_version: Optional[str] = Field(None, description="Current code version")
    constraint_set_hash: Optional[str] = Field(None, description="Hash of constraint set")
    correlation_id: Optional[str] = Field(None, description="Correlation ID for tracing")


class AuditFilters(BaseModel):
    """Filters for querying audit logs."""

    start_time: Optional[datetime] = Field(None, description="Filter from this time")
    end_time: Optional[datetime] = Field(None, description="Filter until this time")
    event_types: Optional[List[AuditEventType]] = Field(None, description="Filter by event types")
    severity_levels: Optional[List[AuditSeverity]] = Field(None, description="Filter by severity")
    user_ids: Optional[List[str]] = Field(None, description="Filter by user IDs")
    session_ids: Optional[List[str]] = Field(None, description="Filter by session IDs")
    recommendation_ids: Optional[List[str]] = Field(None, description="Filter by recommendation IDs")
    search_text: Optional[str] = Field(None, description="Full-text search")
    limit: int = Field(1000, ge=1, le=10000, description="Maximum records to return")
    offset: int = Field(0, ge=0, description="Offset for pagination")


# =============================================================================
# Output Models
# =============================================================================

class AuditRecord(BaseModel):
    """Immutable audit record with cryptographic integrity."""

    record_id: str = Field(..., description="Unique record identifier")
    sequence_number: int = Field(..., ge=0, description="Monotonic sequence number")
    timestamp: datetime = Field(..., description="Record creation timestamp")
    event_type: AuditEventType = Field(..., description="Type of audited event")
    severity: AuditSeverity = Field(..., description="Event severity")
    event_data: Dict[str, Any] = Field(..., description="Event-specific data")
    context: AuditContext = Field(..., description="Audit context")
    integrity_hash: str = Field(..., description="SHA-256 hash for integrity")
    previous_hash: Optional[str] = Field(None, description="Hash of previous record (chain)")

    class Config:
        """Pydantic configuration for immutability."""
        frozen = True  # Makes the model immutable

    @validator('integrity_hash')
    def validate_hash(cls, v: str) -> str:
        """Validate hash format."""
        if len(v) != 64:
            raise ValueError("Integrity hash must be 64-character SHA-256 hex string")
        try:
            int(v, 16)
        except ValueError:
            raise ValueError("Integrity hash must be valid hexadecimal")
        return v


# =============================================================================
# Configuration
# =============================================================================

class AuditLoggerConfig(BaseModel):
    """Configuration for AuditLogger."""

    storage_backend: str = Field("file", description="Storage backend: file, database, both")
    log_directory: str = Field("./audit_logs", description="Directory for file-based logs")
    database_url: Optional[str] = Field(None, description="Database connection URL")
    enable_chain_verification: bool = Field(True, description="Enable hash chain verification")
    buffer_size: int = Field(100, ge=1, description="Buffer size before flush")
    flush_interval_seconds: int = Field(5, ge=1, description="Flush interval in seconds")
    enable_compression: bool = Field(True, description="Enable log compression")
    retention_days: int = Field(2555, ge=1, description="Retention period (7 years default)")


# =============================================================================
# AuditLogger Implementation
# =============================================================================

class AuditLogger:
    """
    AuditLogger implementation for BURNMASTER.

    This class provides comprehensive audit logging for all BURNMASTER operations.
    It follows GreenLang's zero-hallucination principle by using deterministic
    hash calculations and immutable record storage.

    All records are immutable once created and include cryptographic hashes
    for integrity verification, supporting regulatory compliance requirements.

    Attributes:
        config: Logger configuration
        _sequence_number: Monotonically increasing sequence counter
        _last_hash: Hash of the last record for chain verification
        _buffer: In-memory buffer for pending records

    Example:
        >>> config = AuditLoggerConfig()
        >>> logger = AuditLogger(config)
        >>> record = logger.log_recommendation(rec, context)
        >>> assert record.integrity_hash is not None
    """

    def __init__(self, config: AuditLoggerConfig):
        """
        Initialize AuditLogger.

        Args:
            config: Logger configuration
        """
        self.config = config
        self._sequence_number: int = 0
        self._last_hash: Optional[str] = None
        self._buffer: List[AuditRecord] = []
        self._storage: List[AuditRecord] = []  # In-memory storage for now

        logger.info(
            f"AuditLogger initialized with backend={config.storage_backend}, "
            f"chain_verification={config.enable_chain_verification}"
        )

    def log_recommendation(
        self,
        rec: Recommendation,
        context: AuditContext
    ) -> AuditRecord:
        """
        Log a recommendation event.

        Args:
            rec: Recommendation to log
            context: Audit context

        Returns:
            Immutable audit record with integrity hash

        Raises:
            ValueError: If recommendation data is invalid
        """
        start_time = datetime.now(timezone.utc)

        try:
            event_data = {
                "recommendation": rec.dict(),
                "constraints_satisfied": rec.constraints_satisfied,
                "confidence_score": rec.confidence_score
            }

            record = self._create_audit_record(
                event_type=AuditEventType.RECOMMENDATION,
                severity=AuditSeverity.INFO,
                event_data=event_data,
                context=context
            )

            self._store_record(record)

            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            logger.info(
                f"Logged recommendation {rec.recommendation_id} in {processing_time_ms:.2f}ms, "
                f"record_id={record.record_id}"
            )

            return record

        except Exception as e:
            logger.error(f"Failed to log recommendation: {str(e)}", exc_info=True)
            raise

    def log_setpoint_write(
        self,
        write: SetpointWrite,
        context: AuditContext
    ) -> AuditRecord:
        """
        Log a setpoint write operation.

        Args:
            write: Setpoint write to log
            context: Audit context

        Returns:
            Immutable audit record with integrity hash

        Raises:
            ValueError: If write data is invalid
        """
        start_time = datetime.now(timezone.utc)

        try:
            # Determine severity based on write characteristics
            severity = AuditSeverity.INFO
            if not write.validation_passed:
                severity = AuditSeverity.WARNING
            if write.write_source == "manual":
                severity = AuditSeverity.WARNING  # Manual overrides are notable

            event_data = {
                "setpoint_write": write.dict(),
                "delta": write.new_value - write.previous_value,
                "delta_percent": (
                    ((write.new_value - write.previous_value) / write.previous_value * 100)
                    if write.previous_value != 0 else None
                )
            }

            record = self._create_audit_record(
                event_type=AuditEventType.SETPOINT_WRITE,
                severity=severity,
                event_data=event_data,
                context=context
            )

            self._store_record(record)

            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            logger.info(
                f"Logged setpoint write {write.write_id}: {write.setpoint_name} "
                f"{write.previous_value} -> {write.new_value} in {processing_time_ms:.2f}ms"
            )

            return record

        except Exception as e:
            logger.error(f"Failed to log setpoint write: {str(e)}", exc_info=True)
            raise

    def log_mode_transition(
        self,
        transition: ModeTransition,
        context: AuditContext
    ) -> AuditRecord:
        """
        Log a mode transition event.

        Args:
            transition: Mode transition to log
            context: Audit context

        Returns:
            Immutable audit record with integrity hash

        Raises:
            ValueError: If transition data is invalid
        """
        start_time = datetime.now(timezone.utc)

        try:
            # Determine severity based on transition type
            severity = AuditSeverity.INFO
            if not transition.safety_checks_passed:
                severity = AuditSeverity.CRITICAL
            elif transition.to_mode in ["emergency", "shutdown"]:
                severity = AuditSeverity.WARNING

            event_data = {
                "mode_transition": transition.dict(),
                "transition_path": f"{transition.from_mode} -> {transition.to_mode}"
            }

            record = self._create_audit_record(
                event_type=AuditEventType.MODE_TRANSITION,
                severity=severity,
                event_data=event_data,
                context=context
            )

            self._store_record(record)

            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            logger.info(
                f"Logged mode transition {transition.transition_id}: "
                f"{transition.from_mode} -> {transition.to_mode} in {processing_time_ms:.2f}ms"
            )

            return record

        except Exception as e:
            logger.error(f"Failed to log mode transition: {str(e)}", exc_info=True)
            raise

    def log_safety_event(
        self,
        event: SafetyEvent,
        context: AuditContext
    ) -> AuditRecord:
        """
        Log a safety event.

        Safety events are always logged with elevated priority and may trigger
        immediate notification systems.

        Args:
            event: Safety event to log
            context: Audit context

        Returns:
            Immutable audit record with integrity hash

        Raises:
            ValueError: If event data is invalid
        """
        start_time = datetime.now(timezone.utc)

        try:
            event_data = {
                "safety_event": event.dict(),
                "requires_acknowledgment": event.severity in [
                    AuditSeverity.CRITICAL,
                    AuditSeverity.EMERGENCY
                ]
            }

            record = self._create_audit_record(
                event_type=AuditEventType.SAFETY_EVENT,
                severity=event.severity,
                event_data=event_data,
                context=context
            )

            # Safety events bypass buffer and are stored immediately
            self._store_record(record, immediate=True)

            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            logger.warning(
                f"Logged safety event {event.event_id}: {event.event_type.value} "
                f"severity={event.severity.value} in {processing_time_ms:.2f}ms"
            )

            return record

        except Exception as e:
            logger.error(f"Failed to log safety event: {str(e)}", exc_info=True)
            raise

    def log_operator_action(
        self,
        action: OperatorAction,
        context: AuditContext
    ) -> AuditRecord:
        """
        Log an operator action.

        Args:
            action: Operator action to log
            context: Audit context

        Returns:
            Immutable audit record with integrity hash

        Raises:
            ValueError: If action data is invalid
        """
        start_time = datetime.now(timezone.utc)

        try:
            # Determine severity based on action type
            severity = AuditSeverity.INFO
            if action.action_type == OperatorActionType.MANUAL_OVERRIDE:
                severity = AuditSeverity.WARNING
            elif action.action_type == OperatorActionType.CONFIGURATION_CHANGE:
                severity = AuditSeverity.WARNING

            event_data = {
                "operator_action": action.dict(),
                "requires_review": action.action_type in [
                    OperatorActionType.MANUAL_OVERRIDE,
                    OperatorActionType.CONFIGURATION_CHANGE
                ]
            }

            record = self._create_audit_record(
                event_type=AuditEventType.OPERATOR_ACTION,
                severity=severity,
                event_data=event_data,
                context=context
            )

            self._store_record(record)

            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            logger.info(
                f"Logged operator action {action.action_id}: {action.action_type.value} "
                f"by {action.operator_id} in {processing_time_ms:.2f}ms"
            )

            return record

        except Exception as e:
            logger.error(f"Failed to log operator action: {str(e)}", exc_info=True)
            raise

    def query_audit_log(self, filters: AuditFilters) -> List[AuditRecord]:
        """
        Query audit log with filters.

        Args:
            filters: Query filters

        Returns:
            List of matching audit records

        Raises:
            ValueError: If filters are invalid
        """
        start_time = datetime.now(timezone.utc)

        try:
            results: List[AuditRecord] = []

            for record in self._storage:
                if self._matches_filters(record, filters):
                    results.append(record)

            # Apply pagination
            paginated_results = results[filters.offset:filters.offset + filters.limit]

            processing_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            logger.info(
                f"Query returned {len(paginated_results)} records "
                f"(total matching: {len(results)}) in {processing_time_ms:.2f}ms"
            )

            return paginated_results

        except Exception as e:
            logger.error(f"Failed to query audit log: {str(e)}", exc_info=True)
            raise

    def verify_chain_integrity(self) -> bool:
        """
        Verify the integrity of the audit record chain.

        Returns:
            True if chain is valid, False if corruption detected
        """
        if not self._storage:
            return True

        previous_hash: Optional[str] = None

        for record in self._storage:
            # Verify previous hash link
            if record.previous_hash != previous_hash:
                logger.error(
                    f"Chain integrity violation at record {record.record_id}: "
                    f"expected previous_hash={previous_hash}, got {record.previous_hash}"
                )
                return False

            # Verify record integrity hash
            computed_hash = self._compute_integrity_hash(record)
            if computed_hash != record.integrity_hash:
                logger.error(
                    f"Integrity hash mismatch at record {record.record_id}: "
                    f"computed={computed_hash}, stored={record.integrity_hash}"
                )
                return False

            previous_hash = record.integrity_hash

        logger.info(f"Chain integrity verified for {len(self._storage)} records")
        return True

    def _create_audit_record(
        self,
        event_type: AuditEventType,
        severity: AuditSeverity,
        event_data: Dict[str, Any],
        context: AuditContext
    ) -> AuditRecord:
        """
        Create an immutable audit record with integrity hash.

        Args:
            event_type: Type of event
            severity: Event severity
            event_data: Event-specific data
            context: Audit context

        Returns:
            Immutable audit record
        """
        record_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc)
        sequence_number = self._get_next_sequence()

        # Create preliminary record for hash calculation
        prelim_data = {
            "record_id": record_id,
            "sequence_number": sequence_number,
            "timestamp": timestamp.isoformat(),
            "event_type": event_type.value,
            "severity": severity.value,
            "event_data": event_data,
            "context": context.dict(),
            "previous_hash": self._last_hash
        }

        # Compute integrity hash
        integrity_hash = self._compute_hash_from_dict(prelim_data)

        # Create immutable record
        record = AuditRecord(
            record_id=record_id,
            sequence_number=sequence_number,
            timestamp=timestamp,
            event_type=event_type,
            severity=severity,
            event_data=event_data,
            context=context,
            integrity_hash=integrity_hash,
            previous_hash=self._last_hash
        )

        return record

    def _store_record(self, record: AuditRecord, immediate: bool = False) -> None:
        """
        Store an audit record.

        Args:
            record: Record to store
            immediate: If True, bypass buffer and store immediately
        """
        if immediate:
            self._storage.append(record)
            self._last_hash = record.integrity_hash
        else:
            self._buffer.append(record)
            self._last_hash = record.integrity_hash

            if len(self._buffer) >= self.config.buffer_size:
                self._flush_buffer()

    def _flush_buffer(self) -> None:
        """Flush buffered records to storage."""
        if self._buffer:
            self._storage.extend(self._buffer)
            logger.debug(f"Flushed {len(self._buffer)} records to storage")
            self._buffer = []

    def _get_next_sequence(self) -> int:
        """Get next sequence number (monotonically increasing)."""
        self._sequence_number += 1
        return self._sequence_number

    def _compute_hash_from_dict(self, data: Dict[str, Any]) -> str:
        """
        Compute SHA-256 hash from dictionary data.

        Args:
            data: Dictionary to hash

        Returns:
            64-character hexadecimal hash string
        """
        # Sort keys for deterministic serialization
        json_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode('utf-8')).hexdigest()

    def _compute_integrity_hash(self, record: AuditRecord) -> str:
        """
        Recompute integrity hash for verification.

        Args:
            record: Record to verify

        Returns:
            Computed hash
        """
        prelim_data = {
            "record_id": record.record_id,
            "sequence_number": record.sequence_number,
            "timestamp": record.timestamp.isoformat(),
            "event_type": record.event_type.value,
            "severity": record.severity.value,
            "event_data": record.event_data,
            "context": record.context.dict(),
            "previous_hash": record.previous_hash
        }
        return self._compute_hash_from_dict(prelim_data)

    def _matches_filters(self, record: AuditRecord, filters: AuditFilters) -> bool:
        """
        Check if a record matches the given filters.

        Args:
            record: Record to check
            filters: Filters to apply

        Returns:
            True if record matches all filters
        """
        # Time range filter
        if filters.start_time and record.timestamp < filters.start_time:
            return False
        if filters.end_time and record.timestamp > filters.end_time:
            return False

        # Event type filter
        if filters.event_types and record.event_type not in filters.event_types:
            return False

        # Severity filter
        if filters.severity_levels and record.severity not in filters.severity_levels:
            return False

        # User ID filter
        if filters.user_ids:
            user_id = record.context.user_id
            if user_id not in filters.user_ids:
                return False

        # Session ID filter
        if filters.session_ids:
            if record.context.session_id not in filters.session_ids:
                return False

        # Recommendation ID filter
        if filters.recommendation_ids:
            rec_id = record.event_data.get("recommendation", {}).get("recommendation_id")
            if rec_id not in filters.recommendation_ids:
                return False

        # Text search filter
        if filters.search_text:
            search_text = filters.search_text.lower()
            record_json = json.dumps(record.event_data, default=str).lower()
            if search_text not in record_json:
                return False

        return True

    def flush(self) -> None:
        """Manually flush all buffered records to storage."""
        self._flush_buffer()
        logger.info("Manual flush completed")

    def get_record_count(self) -> int:
        """Get total number of stored records."""
        return len(self._storage) + len(self._buffer)
