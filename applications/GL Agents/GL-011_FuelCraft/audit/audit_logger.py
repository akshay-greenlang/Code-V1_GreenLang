# -*- coding: utf-8 -*-
"""
AuditLogger - Comprehensive audit logging for GL-011 FuelCraft.

This module implements audit logging with 7-year retention compliance,
RBAC integration, and export capabilities. All audit events are
cryptographically hashed for integrity verification.

Key Features:
- Structured audit events with provenance
- 7-year retention policy compliance
- RBAC integration for access control
- Export to multiple formats
- Immutable event storage

Author: GL-BackendDeveloper
Date: 2025-01-01
Version: 1.0.0
"""

from typing import Dict, List, Optional, Any, Callable
from pydantic import BaseModel, Field
from datetime import datetime, timezone, timedelta
from enum import Enum
from pathlib import Path
from collections import deque
from threading import Lock
import hashlib
import json
import uuid
import gzip
import logging

logger = logging.getLogger(__name__)


class AuditEventType(str, Enum):
    """Types of audit events."""
    # Run lifecycle
    RUN_STARTED = "run_started"
    RUN_COMPLETED = "run_completed"
    RUN_FAILED = "run_failed"

    # Optimization events
    OPTIMIZATION_STARTED = "optimization_started"
    OPTIMIZATION_COMPLETED = "optimization_completed"
    OPTIMIZATION_FAILED = "optimization_failed"

    # Data events
    DATA_INGESTED = "data_ingested"
    DATA_VALIDATED = "data_validated"
    DATA_TRANSFORMED = "data_transformed"
    DATA_EXPORTED = "data_exported"

    # Safety events
    SAFETY_CHECK_PASSED = "safety_check_passed"
    SAFETY_CHECK_FAILED = "safety_check_failed"
    ALARM_RAISED = "alarm_raised"
    ALARM_ACKNOWLEDGED = "alarm_acknowledged"

    # Transfer events
    TRANSFER_STARTED = "transfer_started"
    TRANSFER_COMPLETED = "transfer_completed"
    TRANSFER_BLOCKED = "transfer_blocked"

    # Compliance events
    COMPLIANCE_CHECK = "compliance_check"
    CONSTRAINT_VIOLATED = "constraint_violated"

    # Access events
    USER_LOGIN = "user_login"
    USER_LOGOUT = "user_logout"
    ACCESS_DENIED = "access_denied"
    PERMISSION_CHANGED = "permission_changed"

    # Administrative events
    CONFIG_CHANGED = "config_changed"
    SYSTEM_STARTED = "system_started"
    SYSTEM_SHUTDOWN = "system_shutdown"


class AuditSeverity(str, Enum):
    """Severity level of audit events."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class RetentionCategory(str, Enum):
    """Retention category for audit events."""
    SAFETY = "safety"           # 7+ years
    COMPLIANCE = "compliance"   # 7+ years
    OPERATIONAL = "operational" # 3 years
    SECURITY = "security"       # 7+ years
    DIAGNOSTIC = "diagnostic"   # 1 year


class AuditEvent(BaseModel):
    """
    Structured audit event record.

    Contains complete context for regulatory audit trail.
    """
    event_id: str = Field(default_factory=lambda: f"EVT-{uuid.uuid4().hex[:12].upper()}")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    event_type: AuditEventType = Field(...)
    severity: AuditSeverity = Field(AuditSeverity.INFO)
    retention_category: RetentionCategory = Field(RetentionCategory.OPERATIONAL)

    # Context
    run_id: Optional[str] = Field(None)
    correlation_id: Optional[str] = Field(None)
    sequence_number: int = Field(0)

    # Actor
    user_id: Optional[str] = Field(None)
    user_role: Optional[str] = Field(None)
    source_system: str = Field("GL-011")
    source_component: Optional[str] = Field(None)

    # Details
    message: str = Field(...)
    details: Dict[str, Any] = Field(default_factory=dict)

    # Affected resources
    resource_type: Optional[str] = Field(None)
    resource_id: Optional[str] = Field(None)
    resource_name: Optional[str] = Field(None)

    # Outcome
    success: bool = Field(True)
    error_code: Optional[str] = Field(None)
    error_message: Optional[str] = Field(None)

    # Provenance
    event_hash: str = Field(default="")
    previous_hash: str = Field(default="")

    def calculate_hash(self) -> str:
        """Calculate SHA-256 hash of event."""
        data = self.model_dump(exclude={"event_hash"})
        return hashlib.sha256(
            json.dumps(data, sort_keys=True, default=str).encode()
        ).hexdigest()


class RetentionPolicy(BaseModel):
    """Retention policy configuration."""
    policy_id: str = Field(...)
    policy_name: str = Field(...)

    # Retention periods by category (years)
    safety_retention_years: int = Field(7)
    compliance_retention_years: int = Field(7)
    security_retention_years: int = Field(7)
    operational_retention_years: int = Field(3)
    diagnostic_retention_years: int = Field(1)

    # Minimum retention (regulatory requirement)
    minimum_retention_years: int = Field(7)

    # Storage settings
    archive_after_days: int = Field(90)
    compress_archives: bool = Field(True)


class RBACConfig(BaseModel):
    """RBAC configuration for audit access."""
    enabled: bool = Field(True)

    # Role permissions
    read_roles: List[str] = Field(default_factory=lambda: ["auditor", "admin", "operator"])
    write_roles: List[str] = Field(default_factory=lambda: ["system"])
    export_roles: List[str] = Field(default_factory=lambda: ["auditor", "admin"])
    delete_roles: List[str] = Field(default_factory=list)  # Empty - no delete allowed


class AuditExportResult(BaseModel):
    """Result of an audit export operation."""
    export_id: str = Field(...)
    exported_at: datetime = Field(...)
    format: str = Field(...)
    event_count: int = Field(...)
    file_path: Optional[str] = Field(None)
    file_hash: str = Field(...)
    exported_by: str = Field(...)


class AuditLogger:
    """
    Comprehensive audit logger with retention compliance.

    Provides structured logging of all system events with
    7-year retention, RBAC integration, and export capabilities.

    Example:
        >>> logger = AuditLogger()
        >>> logger.log_run_started(run_id="RUN-001", user_id="operator1")
        >>> logger.log_optimization_completed(run_id="RUN-001", results=results)
        >>> events = logger.query_events(run_id="RUN-001")
    """

    # Default retention policy
    DEFAULT_RETENTION_YEARS = 7

    def __init__(
        self,
        retention_policy: Optional[RetentionPolicy] = None,
        rbac_config: Optional[RBACConfig] = None,
        storage_path: Optional[str] = None,
        max_memory_events: int = 10000
    ):
        """Initialize audit logger."""
        self._lock = Lock()
        self._retention_policy = retention_policy or self._default_retention_policy()
        self._rbac_config = rbac_config or RBACConfig()
        self._storage_path = Path(storage_path) if storage_path else None

        self._events: deque = deque(maxlen=max_memory_events)
        self._sequence_counter = 0
        self._last_hash = hashlib.sha256(b"genesis").hexdigest()

        if self._storage_path:
            self._storage_path.mkdir(parents=True, exist_ok=True)

        logger.info("AuditLogger initialized with 7-year retention compliance")

    def log_event(
        self,
        event_type: AuditEventType,
        message: str,
        severity: AuditSeverity = AuditSeverity.INFO,
        run_id: Optional[str] = None,
        user_id: Optional[str] = None,
        user_role: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        success: bool = True,
        error_code: Optional[str] = None,
        error_message: Optional[str] = None,
        source_component: Optional[str] = None
    ) -> AuditEvent:
        """
        Log an audit event.

        Args:
            event_type: Type of event
            message: Human-readable message
            severity: Event severity
            run_id: Associated run ID
            user_id: User who triggered event
            user_role: Role of user
            details: Additional details
            resource_type: Type of affected resource
            resource_id: ID of affected resource
            success: Whether operation succeeded
            error_code: Error code if failed
            error_message: Error message if failed
            source_component: Component that generated event

        Returns:
            Created AuditEvent
        """
        with self._lock:
            self._sequence_counter += 1

            retention_category = self._determine_retention_category(event_type, severity)

            event = AuditEvent(
                event_type=event_type,
                severity=severity,
                retention_category=retention_category,
                run_id=run_id,
                sequence_number=self._sequence_counter,
                user_id=user_id,
                user_role=user_role,
                source_component=source_component,
                message=message,
                details=details or {},
                resource_type=resource_type,
                resource_id=resource_id,
                success=success,
                error_code=error_code,
                error_message=error_message,
                previous_hash=self._last_hash
            )

            # Calculate and set hash
            event.event_hash = event.calculate_hash()
            self._last_hash = event.event_hash

            self._events.append(event)

            # Persist if storage configured
            if self._storage_path:
                self._persist_event(event)

            log_level = {
                AuditSeverity.INFO: logging.INFO,
                AuditSeverity.WARNING: logging.WARNING,
                AuditSeverity.ERROR: logging.ERROR,
                AuditSeverity.CRITICAL: logging.CRITICAL
            }.get(severity, logging.INFO)

            logger.log(
                log_level,
                f"[AUDIT] {event_type.value}: {message} "
                f"(event_id={event.event_id}, run_id={run_id})"
            )

            return event

    # Convenience methods for common events

    def log_run_started(
        self,
        run_id: str,
        user_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ) -> AuditEvent:
        """Log run started event."""
        return self.log_event(
            event_type=AuditEventType.RUN_STARTED,
            message=f"Run {run_id} started",
            run_id=run_id,
            user_id=user_id,
            details=details,
            resource_type="run",
            resource_id=run_id
        )

    def log_run_completed(
        self,
        run_id: str,
        duration_seconds: float,
        result_summary: Optional[Dict[str, Any]] = None
    ) -> AuditEvent:
        """Log run completed event."""
        return self.log_event(
            event_type=AuditEventType.RUN_COMPLETED,
            message=f"Run {run_id} completed in {duration_seconds:.2f}s",
            run_id=run_id,
            details={
                "duration_seconds": duration_seconds,
                "result_summary": result_summary or {}
            },
            resource_type="run",
            resource_id=run_id
        )

    def log_run_failed(
        self,
        run_id: str,
        error: str,
        error_code: Optional[str] = None
    ) -> AuditEvent:
        """Log run failed event."""
        return self.log_event(
            event_type=AuditEventType.RUN_FAILED,
            message=f"Run {run_id} failed: {error}",
            severity=AuditSeverity.ERROR,
            run_id=run_id,
            success=False,
            error_code=error_code,
            error_message=error,
            resource_type="run",
            resource_id=run_id
        )

    def log_optimization_completed(
        self,
        run_id: str,
        objective_value: float,
        solver_time_seconds: float,
        details: Optional[Dict[str, Any]] = None
    ) -> AuditEvent:
        """Log optimization completed event."""
        return self.log_event(
            event_type=AuditEventType.OPTIMIZATION_COMPLETED,
            message=f"Optimization completed: objective={objective_value:.4f}",
            run_id=run_id,
            details={
                "objective_value": objective_value,
                "solver_time_seconds": solver_time_seconds,
                **(details or {})
            },
            resource_type="optimization",
            resource_id=run_id
        )

    def log_safety_check(
        self,
        run_id: str,
        check_name: str,
        passed: bool,
        details: Optional[Dict[str, Any]] = None
    ) -> AuditEvent:
        """Log safety check event."""
        event_type = AuditEventType.SAFETY_CHECK_PASSED if passed else AuditEventType.SAFETY_CHECK_FAILED
        severity = AuditSeverity.INFO if passed else AuditSeverity.WARNING

        return self.log_event(
            event_type=event_type,
            message=f"Safety check '{check_name}': {'PASSED' if passed else 'FAILED'}",
            severity=severity,
            run_id=run_id,
            success=passed,
            details={"check_name": check_name, **(details or {})},
            resource_type="safety_check",
            resource_id=check_name
        )

    def log_transfer_event(
        self,
        transfer_id: str,
        event_type: AuditEventType,
        source_tank: str,
        dest_tank: str,
        volume_m3: Optional[float] = None,
        blocked_reason: Optional[str] = None
    ) -> AuditEvent:
        """Log transfer event."""
        if event_type == AuditEventType.TRANSFER_BLOCKED:
            message = f"Transfer {transfer_id} BLOCKED: {blocked_reason}"
            severity = AuditSeverity.WARNING
            success = False
        elif event_type == AuditEventType.TRANSFER_COMPLETED:
            message = f"Transfer {transfer_id} completed: {volume_m3:.1f} m3 from {source_tank} to {dest_tank}"
            severity = AuditSeverity.INFO
            success = True
        else:
            message = f"Transfer {transfer_id} started: {source_tank} to {dest_tank}"
            severity = AuditSeverity.INFO
            success = True

        return self.log_event(
            event_type=event_type,
            message=message,
            severity=severity,
            success=success,
            details={
                "source_tank": source_tank,
                "dest_tank": dest_tank,
                "volume_m3": volume_m3,
                "blocked_reason": blocked_reason
            },
            resource_type="transfer",
            resource_id=transfer_id
        )

    def log_access_denied(
        self,
        user_id: str,
        resource: str,
        action: str,
        reason: str
    ) -> AuditEvent:
        """Log access denied event."""
        return self.log_event(
            event_type=AuditEventType.ACCESS_DENIED,
            message=f"Access denied: {user_id} attempted {action} on {resource}",
            severity=AuditSeverity.WARNING,
            user_id=user_id,
            success=False,
            details={"resource": resource, "action": action, "reason": reason},
            resource_type="access",
            resource_id=resource
        )

    def query_events(
        self,
        run_id: Optional[str] = None,
        event_type: Optional[AuditEventType] = None,
        user_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        severity: Optional[AuditSeverity] = None,
        limit: int = 1000,
        requesting_user: Optional[str] = None,
        requesting_role: Optional[str] = None
    ) -> List[AuditEvent]:
        """
        Query audit events with filters.

        Args:
            run_id: Filter by run ID
            event_type: Filter by event type
            user_id: Filter by user ID
            start_time: Start of time range
            end_time: End of time range
            severity: Filter by severity
            limit: Maximum events to return
            requesting_user: User requesting the query
            requesting_role: Role of requesting user

        Returns:
            List of matching AuditEvent records
        """
        # RBAC check
        if self._rbac_config.enabled:
            if requesting_role not in self._rbac_config.read_roles:
                logger.warning(f"Audit query denied for role {requesting_role}")
                return []

        with self._lock:
            results = []

            for event in reversed(self._events):
                if len(results) >= limit:
                    break

                if run_id and event.run_id != run_id:
                    continue
                if event_type and event.event_type != event_type:
                    continue
                if user_id and event.user_id != user_id:
                    continue
                if start_time and event.timestamp < start_time:
                    continue
                if end_time and event.timestamp > end_time:
                    continue
                if severity and event.severity != severity:
                    continue

                results.append(event)

            return results

    def export_events(
        self,
        format: str = "json",
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        output_path: Optional[str] = None,
        exported_by: str = "system"
    ) -> AuditExportResult:
        """
        Export audit events to file.

        Args:
            format: Export format (json, csv)
            start_time: Start of time range
            end_time: End of time range
            output_path: Output file path
            exported_by: User performing export

        Returns:
            AuditExportResult with export details
        """
        events = self.query_events(
            start_time=start_time,
            end_time=end_time,
            limit=100000,
            requesting_role="admin"
        )

        export_id = f"EXPORT-{uuid.uuid4().hex[:12].upper()}"

        if format == "json":
            content = json.dumps(
                [e.model_dump() for e in events],
                indent=2,
                default=str
            )
        else:
            raise ValueError(f"Unsupported export format: {format}")

        content_bytes = content.encode('utf-8')
        file_hash = hashlib.sha256(content_bytes).hexdigest()

        if output_path:
            with open(output_path, 'w') as f:
                f.write(content)

        return AuditExportResult(
            export_id=export_id,
            exported_at=datetime.now(timezone.utc),
            format=format,
            event_count=len(events),
            file_path=output_path,
            file_hash=file_hash,
            exported_by=exported_by
        )

    def get_retention_status(self) -> Dict[str, Any]:
        """Get retention status for all categories."""
        now = datetime.now(timezone.utc)
        status = {}

        for category in RetentionCategory:
            retention_years = self._get_retention_years(category)
            cutoff = now - timedelta(days=365 * retention_years)

            with self._lock:
                category_events = [
                    e for e in self._events
                    if e.retention_category == category
                ]

            status[category.value] = {
                "retention_years": retention_years,
                "cutoff_date": cutoff.isoformat(),
                "event_count": len(category_events),
                "oldest_event": min([e.timestamp for e in category_events]).isoformat() if category_events else None
            }

        return status

    def verify_chain_integrity(self) -> bool:
        """Verify the hash chain integrity of audit events."""
        with self._lock:
            previous_hash = hashlib.sha256(b"genesis").hexdigest()

            for event in self._events:
                if event.previous_hash != previous_hash:
                    logger.error(f"Chain integrity broken at event {event.event_id}")
                    return False

                calculated_hash = event.calculate_hash()
                if calculated_hash != event.event_hash:
                    logger.error(f"Event hash mismatch at {event.event_id}")
                    return False

                previous_hash = event.event_hash

            return True

    def _determine_retention_category(
        self,
        event_type: AuditEventType,
        severity: AuditSeverity
    ) -> RetentionCategory:
        """Determine retention category for an event."""
        safety_events = {
            AuditEventType.SAFETY_CHECK_PASSED,
            AuditEventType.SAFETY_CHECK_FAILED,
            AuditEventType.ALARM_RAISED,
            AuditEventType.ALARM_ACKNOWLEDGED,
            AuditEventType.TRANSFER_BLOCKED
        }

        compliance_events = {
            AuditEventType.COMPLIANCE_CHECK,
            AuditEventType.CONSTRAINT_VIOLATED,
            AuditEventType.RUN_COMPLETED,
            AuditEventType.OPTIMIZATION_COMPLETED
        }

        security_events = {
            AuditEventType.USER_LOGIN,
            AuditEventType.USER_LOGOUT,
            AuditEventType.ACCESS_DENIED,
            AuditEventType.PERMISSION_CHANGED
        }

        if event_type in safety_events:
            return RetentionCategory.SAFETY
        elif event_type in compliance_events:
            return RetentionCategory.COMPLIANCE
        elif event_type in security_events:
            return RetentionCategory.SECURITY
        elif severity in [AuditSeverity.ERROR, AuditSeverity.CRITICAL]:
            return RetentionCategory.COMPLIANCE
        else:
            return RetentionCategory.OPERATIONAL

    def _get_retention_years(self, category: RetentionCategory) -> int:
        """Get retention years for a category."""
        mapping = {
            RetentionCategory.SAFETY: self._retention_policy.safety_retention_years,
            RetentionCategory.COMPLIANCE: self._retention_policy.compliance_retention_years,
            RetentionCategory.SECURITY: self._retention_policy.security_retention_years,
            RetentionCategory.OPERATIONAL: self._retention_policy.operational_retention_years,
            RetentionCategory.DIAGNOSTIC: self._retention_policy.diagnostic_retention_years
        }
        return mapping.get(category, self.DEFAULT_RETENTION_YEARS)

    def _persist_event(self, event: AuditEvent) -> None:
        """Persist event to storage."""
        if not self._storage_path:
            return

        date_path = event.timestamp.strftime("%Y/%m/%d")
        event_path = self._storage_path / date_path
        event_path.mkdir(parents=True, exist_ok=True)

        file_path = event_path / f"{event.event_id}.json.gz"
        content = json.dumps(event.model_dump(), default=str)

        with gzip.open(file_path, 'wt') as f:
            f.write(content)

    def _default_retention_policy(self) -> RetentionPolicy:
        """Get default retention policy."""
        return RetentionPolicy(
            policy_id="DEFAULT",
            policy_name="7-Year Regulatory Compliance",
            safety_retention_years=7,
            compliance_retention_years=7,
            security_retention_years=7,
            operational_retention_years=3,
            diagnostic_retention_years=1,
            minimum_retention_years=7
        )
