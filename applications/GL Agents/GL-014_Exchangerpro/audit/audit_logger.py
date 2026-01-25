# -*- coding: utf-8 -*-
"""
GL-014 Exchangerpro - Audit Logger

Immutable audit logging system for heat exchanger operations with
microsecond precision timestamps, SHA-256 hash chain integrity,
and comprehensive event tracking.

Features:
    - Immutable audit log entries
    - Timestamp with microsecond precision (ISO 8601)
    - User action logging with full context
    - System event logging for all operations
    - SHA-256 hash chain for tamper detection
    - Thread-safe singleton pattern
    - Multiple storage backends support

Standards:
    - ISO 27001: Information security management
    - SOC 2 Type II: Service organization controls
    - 21 CFR Part 11: Electronic records and signatures
    - TEMA: Heat exchanger standards

Example:
    >>> from audit.audit_logger import AuditLogger, get_audit_logger
    >>> logger = get_audit_logger()
    >>> logger.log_user_action(
    ...     user_id="engineer@company.com",
    ...     action="modify_threshold",
    ...     resource_type="exchanger_config",
    ...     resource_id="HEX-001"
    ... )
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
import uuid
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union, Generator

from .schemas import (
    AuditRecord,
    ComputationType,
    AuditEventType,
    ActorType,
    SeverityLevel,
    compute_sha256,
)

logger = logging.getLogger(__name__)


# =============================================================================
# AUDIT EVENT ENUMS
# =============================================================================

class AuditOutcome(str, Enum):
    """Outcome of audited operation."""

    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL = "partial"
    SKIPPED = "skipped"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


# =============================================================================
# IMMUTABLE AUDIT CONTEXT
# =============================================================================

@dataclass(frozen=True)
class AuditContext:
    """
    Immutable context for audit events.

    Provides correlation and tracing information that follows
    a request through the entire system.
    """

    correlation_id: str
    session_id: str
    user_id: Optional[str] = None
    tenant_id: Optional[str] = None
    source_ip: Optional[str] = None
    user_agent: Optional[str] = None
    request_id: Optional[str] = None
    exchanger_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "correlation_id": self.correlation_id,
            "session_id": self.session_id,
            "user_id": self.user_id,
            "tenant_id": self.tenant_id,
            "source_ip": self.source_ip,
            "user_agent": self.user_agent,
            "request_id": self.request_id,
            "exchanger_id": self.exchanger_id,
        }


# =============================================================================
# IMMUTABLE AUDIT EVENT
# =============================================================================

@dataclass(frozen=True)
class AuditEvent:
    """
    Immutable audit event record.

    Captures all details of an auditable operation before
    being persisted to the audit log.
    """

    event_type: AuditEventType
    action: str
    severity: SeverityLevel = SeverityLevel.INFO
    outcome: AuditOutcome = AuditOutcome.SUCCESS
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "event_type": self.event_type.value if isinstance(self.event_type, AuditEventType) else str(self.event_type),
            "action": self.action,
            "severity": self.severity.value if isinstance(self.severity, SeverityLevel) else str(self.severity),
            "outcome": self.outcome.value if isinstance(self.outcome, AuditOutcome) else str(self.outcome),
            "message": self.message,
            "details": dict(self.details),
            "metadata": dict(self.metadata),
        }


# =============================================================================
# AUDIT LOG ENTRY
# =============================================================================

@dataclass
class AuditLogEntry:
    """
    Complete audit log entry with provenance.

    This is the final, immutable form of an audit event that
    is persisted to storage with full hash chain linking.
    """

    entry_id: str
    timestamp: str  # ISO 8601 with microsecond precision
    event_type: str
    action: str
    severity: str
    outcome: str
    message: str
    details: Dict[str, Any]
    metadata: Dict[str, Any]
    context: Dict[str, Any]
    provenance: Dict[str, str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    def to_json(self) -> str:
        """Convert to JSON string with deterministic ordering."""
        return json.dumps(self.to_dict(), sort_keys=True, default=str)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AuditLogEntry":
        """Create from dictionary."""
        return cls(**data)


# =============================================================================
# AUDIT STORAGE INTERFACE
# =============================================================================

class AuditStorage(ABC):
    """Abstract base class for audit storage backends."""

    @abstractmethod
    def store(self, entry: AuditLogEntry) -> None:
        """Store a single audit entry."""
        pass

    @abstractmethod
    def store_batch(self, entries: List[AuditLogEntry]) -> None:
        """Store multiple audit entries."""
        pass

    @abstractmethod
    def query(
        self,
        event_type: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        correlation_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[AuditLogEntry]:
        """Query audit entries."""
        pass

    @abstractmethod
    def get_by_id(self, entry_id: str) -> Optional[AuditLogEntry]:
        """Get a single entry by ID."""
        pass

    @abstractmethod
    def verify_integrity(self) -> bool:
        """Verify hash chain integrity."""
        pass


# =============================================================================
# FILE AUDIT STORAGE
# =============================================================================

class FileAuditStorage(AuditStorage):
    """
    File-based audit storage using JSON Lines format.

    Suitable for development and smaller deployments.
    Each entry is stored as a single JSON line for append-only writes.
    """

    def __init__(
        self,
        directory: str = "audit_logs",
        rotate_size_mb: float = 100.0,
        compress_old: bool = True,
    ):
        """
        Initialize file storage.

        Args:
            directory: Directory for audit files
            rotate_size_mb: Rotate file when size exceeds this
            compress_old: Compress rotated files
        """
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.rotate_size_bytes = int(rotate_size_mb * 1024 * 1024)
        self.compress_old = compress_old
        self._current_file: Optional[Path] = None
        self._lock = threading.Lock()

        logger.info(f"FileAuditStorage initialized at {self.directory}")

    def _get_current_file(self) -> Path:
        """Get current audit log file."""
        if self._current_file is None:
            date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            self._current_file = self.directory / f"audit_gl014_{date_str}.jsonl"
        return self._current_file

    def _check_rotation(self) -> None:
        """Check if file rotation is needed."""
        current_file = self._get_current_file()
        if current_file.exists():
            size = current_file.stat().st_size
            if size >= self.rotate_size_bytes:
                self._rotate_file(current_file)

    def _rotate_file(self, file_path: Path) -> None:
        """Rotate the current audit file."""
        timestamp = datetime.now(timezone.utc).strftime("%H%M%S")
        new_name = file_path.with_suffix(f".{timestamp}.jsonl")
        file_path.rename(new_name)
        self._current_file = None

        if self.compress_old:
            import gzip
            with open(new_name, "rb") as f_in:
                with gzip.open(f"{new_name}.gz", "wb") as f_out:
                    f_out.write(f_in.read())
            new_name.unlink()

        logger.info(f"Rotated audit log: {file_path} -> {new_name}")

    def store(self, entry: AuditLogEntry) -> None:
        """Store a single audit entry."""
        with self._lock:
            self._check_rotation()
            file_path = self._get_current_file()

            with open(file_path, "a", encoding="utf-8") as f:
                f.write(entry.to_json() + "\n")

    def store_batch(self, entries: List[AuditLogEntry]) -> None:
        """Store multiple audit entries."""
        with self._lock:
            self._check_rotation()
            file_path = self._get_current_file()

            with open(file_path, "a", encoding="utf-8") as f:
                for entry in entries:
                    f.write(entry.to_json() + "\n")

    def query(
        self,
        event_type: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        correlation_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[AuditLogEntry]:
        """Query audit entries from files."""
        entries = []
        files = sorted(self.directory.glob("audit_gl014_*.jsonl"), reverse=True)

        for file_path in files:
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        entry = AuditLogEntry.from_dict(data)

                        if self._matches_query(entry, event_type, start_time, end_time, correlation_id):
                            entries.append(entry)
                            if len(entries) >= limit:
                                return entries
                    except json.JSONDecodeError:
                        continue

        return entries

    def _matches_query(
        self,
        entry: AuditLogEntry,
        event_type: Optional[str],
        start_time: Optional[datetime],
        end_time: Optional[datetime],
        correlation_id: Optional[str],
    ) -> bool:
        """Check if entry matches query filters."""
        if event_type and entry.event_type != event_type:
            return False

        if correlation_id:
            if entry.context.get("correlation_id") != correlation_id:
                return False

        if start_time:
            entry_time = datetime.fromisoformat(entry.timestamp.replace("Z", "+00:00"))
            if entry_time < start_time:
                return False

        if end_time:
            entry_time = datetime.fromisoformat(entry.timestamp.replace("Z", "+00:00"))
            if entry_time > end_time:
                return False

        return True

    def get_by_id(self, entry_id: str) -> Optional[AuditLogEntry]:
        """Get a single entry by ID."""
        files = sorted(self.directory.glob("audit_gl014_*.jsonl"), reverse=True)

        for file_path in files:
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        if data.get("entry_id") == entry_id:
                            return AuditLogEntry.from_dict(data)
                    except json.JSONDecodeError:
                        continue

        return None

    def verify_integrity(self) -> bool:
        """Verify hash chain integrity of all entries."""
        files = sorted(self.directory.glob("audit_gl014_*.jsonl"))
        previous_hash = None

        for file_path in files:
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        entry = AuditLogEntry.from_dict(data)

                        stored_previous = entry.provenance.get("previous_hash")
                        if previous_hash is not None:
                            if stored_previous != previous_hash:
                                logger.error(
                                    f"Hash chain broken at {entry.entry_id}: "
                                    f"expected {previous_hash}, got {stored_previous}"
                                )
                                return False

                        previous_hash = entry.provenance.get("entry_hash")

                    except json.JSONDecodeError:
                        continue

        logger.info("Hash chain integrity verified successfully")
        return True


# =============================================================================
# MAIN AUDIT LOGGER
# =============================================================================

class AuditLogger:
    """
    Enterprise-grade audit logger for GL-014 Exchangerpro.

    Provides comprehensive audit logging with:
    - Immutable log entries with microsecond precision timestamps
    - SHA-256 hash chain for tamper detection
    - Correlation IDs for request tracing
    - Multiple storage backends
    - Thread-safe singleton pattern
    - Context managers for scoped logging

    Usage:
        >>> logger = AuditLogger()
        >>> logger.log_user_action(
        ...     user_id="engineer@company.com",
        ...     action="update_threshold",
        ...     resource_type="exchanger_config",
        ...     resource_id="HEX-001"
        ... )
    """

    _instance: Optional["AuditLogger"] = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs) -> "AuditLogger":
        """Singleton pattern for global audit logger."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        storage: Optional[AuditStorage] = None,
        app_name: str = "GL-014-EXCHANGERPRO",
        environment: str = "production",
        enable_hash_chain: bool = True,
    ):
        """
        Initialize audit logger.

        Args:
            storage: Audit storage backend
            app_name: Application identifier
            environment: Deployment environment
            enable_hash_chain: Enable SHA-256 hash chain for integrity
        """
        if hasattr(self, "_initialized") and self._initialized:
            return

        self.app_name = app_name
        self.environment = environment
        self.enable_hash_chain = enable_hash_chain
        self._storage = storage
        self._previous_hash: Optional[str] = None
        self._entry_count = 0
        self._local = threading.local()
        self._initialized = True

        logger.info(
            f"AuditLogger initialized: app={app_name}, env={environment}, "
            f"hash_chain={enable_hash_chain}"
        )

    def set_storage(self, storage: AuditStorage) -> None:
        """Set storage backend."""
        self._storage = storage

    def _get_context(self) -> AuditContext:
        """Get current audit context from thread-local storage."""
        return getattr(
            self._local,
            "context",
            AuditContext(
                correlation_id=str(uuid.uuid4()),
                session_id=str(uuid.uuid4()),
            )
        )

    def set_context(self, context: AuditContext) -> None:
        """Set audit context for current thread."""
        self._local.context = context

    @contextmanager
    def correlation_context(
        self,
        correlation_id: Optional[str] = None,
        user_id: Optional[str] = None,
        exchanger_id: Optional[str] = None,
        **kwargs,
    ) -> Generator[AuditContext, None, None]:
        """
        Context manager for correlation tracking.

        Args:
            correlation_id: Optional correlation ID (generated if not provided)
            user_id: Optional user identifier
            exchanger_id: Optional exchanger identifier
            **kwargs: Additional context fields

        Yields:
            AuditContext for this correlation scope
        """
        old_context = getattr(self._local, "context", None)

        new_context = AuditContext(
            correlation_id=correlation_id or str(uuid.uuid4()),
            session_id=str(uuid.uuid4()),
            user_id=user_id,
            exchanger_id=exchanger_id,
            **kwargs,
        )
        self._local.context = new_context

        try:
            yield new_context
        finally:
            if old_context:
                self._local.context = old_context
            else:
                if hasattr(self._local, "context"):
                    delattr(self._local, "context")

    def _get_timestamp(self) -> str:
        """Get current timestamp with microsecond precision in ISO 8601 format."""
        return datetime.now(timezone.utc).isoformat(timespec='microseconds')

    def _compute_entry_hash(self, entry: AuditLogEntry) -> str:
        """Compute SHA-256 hash for audit entry."""
        content = entry.to_json()
        if self._previous_hash:
            content = self._previous_hash + content
        return hashlib.sha256(content.encode()).hexdigest()

    def _create_entry(self, event: AuditEvent) -> AuditLogEntry:
        """Create complete audit log entry from event."""
        context = self._get_context()

        entry = AuditLogEntry(
            entry_id=str(uuid.uuid4()),
            timestamp=self._get_timestamp(),
            event_type=event.event_type.value if isinstance(event.event_type, AuditEventType) else str(event.event_type),
            action=event.action,
            severity=event.severity.value if isinstance(event.severity, SeverityLevel) else str(event.severity),
            outcome=event.outcome.value if isinstance(event.outcome, AuditOutcome) else str(event.outcome),
            message=event.message,
            details=dict(event.details),
            metadata={
                "app_name": self.app_name,
                "environment": self.environment,
                "entry_sequence": self._entry_count,
                **event.metadata,
            },
            context=context.to_dict(),
            provenance={},
        )

        # Compute hash chain
        if self.enable_hash_chain:
            entry_hash = self._compute_entry_hash(entry)
            entry.provenance["entry_hash"] = entry_hash
            entry.provenance["previous_hash"] = self._previous_hash or "genesis"
            entry.provenance["hash_algorithm"] = "SHA-256"
            self._previous_hash = entry_hash

        self._entry_count += 1
        return entry

    def log(self, event: AuditEvent) -> AuditLogEntry:
        """
        Log an audit event.

        Args:
            event: Audit event to log

        Returns:
            Complete audit log entry
        """
        entry = self._create_entry(event)

        # Persist to storage if configured
        if self._storage:
            self._storage.store(entry)

        # Also log to standard logger
        log_method = getattr(logger, event.severity.value, logger.info)
        log_method(
            f"[AUDIT] {event.event_type.value}:{event.action} - {event.message} "
            f"(correlation_id={entry.context['correlation_id']})"
        )

        return entry

    # =========================================================================
    # USER ACTION LOGGING
    # =========================================================================

    def log_user_action(
        self,
        user_id: str,
        action: str,
        resource_type: str,
        resource_id: str,
        success: bool = True,
        details: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> AuditLogEntry:
        """
        Log a user action.

        Args:
            user_id: User identifier
            action: Action performed
            resource_type: Type of resource affected
            resource_id: Identifier of resource
            success: Whether action succeeded
            details: Additional details
            **kwargs: Extra metadata

        Returns:
            Audit log entry
        """
        return self.log(AuditEvent(
            event_type=AuditEventType.USER_ACTION,
            action=action,
            severity=SeverityLevel.INFO,
            outcome=AuditOutcome.SUCCESS if success else AuditOutcome.FAILURE,
            message=f"User {user_id} performed {action} on {resource_type}/{resource_id}",
            details={
                "user_id": user_id,
                "resource_type": resource_type,
                "resource_id": resource_id,
                **(details or {}),
            },
            metadata=kwargs,
        ))

    def log_user_login(
        self,
        user_id: str,
        success: bool,
        auth_method: str = "password",
        source_ip: Optional[str] = None,
    ) -> AuditLogEntry:
        """Log user login attempt."""
        return self.log(AuditEvent(
            event_type=AuditEventType.USER_ACTION,
            action="login",
            severity=SeverityLevel.INFO if success else SeverityLevel.WARNING,
            outcome=AuditOutcome.SUCCESS if success else AuditOutcome.FAILURE,
            message=f"User {user_id} login {'successful' if success else 'failed'}",
            details={
                "user_id": user_id,
                "auth_method": auth_method,
                "source_ip": source_ip,
            },
        ))

    def log_user_logout(self, user_id: str, session_duration_seconds: float = 0.0) -> AuditLogEntry:
        """Log user logout."""
        return self.log(AuditEvent(
            event_type=AuditEventType.USER_ACTION,
            action="logout",
            severity=SeverityLevel.INFO,
            outcome=AuditOutcome.SUCCESS,
            message=f"User {user_id} logged out",
            details={
                "user_id": user_id,
                "session_duration_seconds": session_duration_seconds,
            },
        ))

    def log_approval(
        self,
        approver_id: str,
        item_type: str,
        item_id: str,
        approved: bool,
        comments: Optional[str] = None,
    ) -> AuditLogEntry:
        """Log approval decision."""
        return self.log(AuditEvent(
            event_type=AuditEventType.USER_ACTION,
            action="approve" if approved else "reject",
            severity=SeverityLevel.INFO,
            outcome=AuditOutcome.SUCCESS,
            message=f"{item_type} {item_id} {'approved' if approved else 'rejected'} by {approver_id}",
            details={
                "approver_id": approver_id,
                "item_type": item_type,
                "item_id": item_id,
                "approved": approved,
                "comments": comments,
            },
        ))

    # =========================================================================
    # SYSTEM EVENT LOGGING
    # =========================================================================

    def log_system_startup(
        self,
        version: str,
        config_hash: Optional[str] = None,
        components: Optional[Dict[str, str]] = None,
    ) -> AuditLogEntry:
        """Log system startup."""
        return self.log(AuditEvent(
            event_type=AuditEventType.SYSTEM_EVENT,
            action="startup",
            severity=SeverityLevel.INFO,
            outcome=AuditOutcome.SUCCESS,
            message=f"GL-014 Exchangerpro started, version {version}",
            details={
                "version": version,
                "config_hash": config_hash,
                "components": components or {},
            },
        ))

    def log_system_shutdown(
        self,
        reason: str = "normal",
        uptime_seconds: Optional[float] = None,
    ) -> AuditLogEntry:
        """Log system shutdown."""
        return self.log(AuditEvent(
            event_type=AuditEventType.SYSTEM_EVENT,
            action="shutdown",
            severity=SeverityLevel.INFO,
            outcome=AuditOutcome.SUCCESS,
            message=f"GL-014 Exchangerpro shutdown: {reason}",
            details={
                "reason": reason,
                "uptime_seconds": uptime_seconds,
            },
        ))

    def log_system_error(
        self,
        error_type: str,
        error_message: str,
        component: str,
        stack_trace: Optional[str] = None,
        exchanger_id: Optional[str] = None,
    ) -> AuditLogEntry:
        """Log system error."""
        return self.log(AuditEvent(
            event_type=AuditEventType.SYSTEM_EVENT,
            action="error",
            severity=SeverityLevel.ERROR,
            outcome=AuditOutcome.FAILURE,
            message=f"System error in {component}: {error_type}",
            details={
                "error_type": error_type,
                "error_message": error_message,
                "component": component,
                "stack_trace": stack_trace,
                "exchanger_id": exchanger_id,
            },
        ))

    def log_health_check(
        self,
        status: str,
        components: Dict[str, str],
        latency_ms: Optional[float] = None,
    ) -> AuditLogEntry:
        """Log health check result."""
        return self.log(AuditEvent(
            event_type=AuditEventType.SYSTEM_EVENT,
            action="health_check",
            severity=SeverityLevel.DEBUG,
            outcome=AuditOutcome.SUCCESS if status == "healthy" else AuditOutcome.PARTIAL,
            message=f"Health check: {status}",
            details={
                "status": status,
                "components": components,
                "latency_ms": latency_ms,
            },
        ))

    # =========================================================================
    # CONFIGURATION CHANGE LOGGING
    # =========================================================================

    def log_config_change(
        self,
        config_key: str,
        old_value: Any,
        new_value: Any,
        changed_by: Optional[str] = None,
        reason: Optional[str] = None,
        exchanger_id: Optional[str] = None,
    ) -> AuditLogEntry:
        """Log configuration change."""
        return self.log(AuditEvent(
            event_type=AuditEventType.CONFIGURATION_CHANGED,
            action="update_config",
            severity=SeverityLevel.INFO,
            outcome=AuditOutcome.SUCCESS,
            message=f"Configuration '{config_key}' changed",
            details={
                "config_key": config_key,
                "old_value": old_value,
                "new_value": new_value,
                "old_value_hash": compute_sha256(old_value),
                "new_value_hash": compute_sha256(new_value),
                "changed_by": changed_by,
                "reason": reason,
                "exchanger_id": exchanger_id,
            },
        ))

    def log_threshold_update(
        self,
        threshold_name: str,
        old_value: float,
        new_value: float,
        unit: str,
        exchanger_id: str,
        standard_reference: Optional[str] = None,
    ) -> AuditLogEntry:
        """Log threshold update."""
        return self.log(AuditEvent(
            event_type=AuditEventType.CONFIGURATION_CHANGED,
            action="update_threshold",
            severity=SeverityLevel.WARNING,
            outcome=AuditOutcome.SUCCESS,
            message=f"Threshold '{threshold_name}' updated: {old_value} -> {new_value} {unit}",
            details={
                "threshold_name": threshold_name,
                "old_value": old_value,
                "new_value": new_value,
                "unit": unit,
                "exchanger_id": exchanger_id,
                "standard_reference": standard_reference,
            },
        ))

    # =========================================================================
    # CALCULATION LOGGING
    # =========================================================================

    def log_calculation(
        self,
        calculation_type: ComputationType,
        exchanger_id: str,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
        algorithm_version: str,
        duration_ms: float,
        success: bool = True,
        formula_reference: Optional[str] = None,
    ) -> AuditLogEntry:
        """
        Log a calculation with full input/output provenance.

        Args:
            calculation_type: Type of calculation performed
            exchanger_id: Heat exchanger identifier
            inputs: Input parameters
            outputs: Calculation results
            algorithm_version: Version of algorithm used
            duration_ms: Calculation duration
            success: Whether calculation succeeded
            formula_reference: Reference to formula used

        Returns:
            Audit log entry
        """
        return self.log(AuditEvent(
            event_type=AuditEventType.CALCULATION_EXECUTED,
            action=f"calculate_{calculation_type.value}",
            severity=SeverityLevel.INFO if success else SeverityLevel.ERROR,
            outcome=AuditOutcome.SUCCESS if success else AuditOutcome.FAILURE,
            message=f"Calculation {calculation_type.value} for {exchanger_id}",
            details={
                "calculation_type": calculation_type.value,
                "exchanger_id": exchanger_id,
                "inputs_hash": compute_sha256(inputs),
                "outputs_hash": compute_sha256(outputs),
                "inputs_summary": self._summarize_data(inputs),
                "outputs_summary": self._summarize_data(outputs),
                "algorithm_version": algorithm_version,
                "formula_reference": formula_reference,
                "duration_ms": duration_ms,
            },
        ))

    def _summarize_data(self, data: Dict[str, Any], max_items: int = 10) -> Dict[str, Any]:
        """Create a summary of data for audit logging."""
        if not data:
            return {}

        summary = {}
        for key, value in list(data.items())[:max_items]:
            if isinstance(value, (list, tuple)):
                summary[key] = f"<{len(value)} items>"
            elif isinstance(value, dict):
                summary[key] = f"<dict with {len(value)} keys>"
            elif isinstance(value, (int, float, str, bool)):
                summary[key] = value
            else:
                summary[key] = f"<{type(value).__name__}>"
        return summary

    # =========================================================================
    # PREDICTION AND RECOMMENDATION LOGGING
    # =========================================================================

    def log_prediction(
        self,
        prediction_type: str,
        exchanger_id: str,
        model_id: str,
        model_version: str,
        inputs_hash: str,
        prediction_value: Any,
        confidence: float,
        duration_ms: float,
    ) -> AuditLogEntry:
        """Log a model prediction."""
        return self.log(AuditEvent(
            event_type=AuditEventType.PREDICTION_GENERATED,
            action=f"predict_{prediction_type}",
            severity=SeverityLevel.INFO,
            outcome=AuditOutcome.SUCCESS,
            message=f"Prediction {prediction_type} for {exchanger_id} (confidence: {confidence:.2%})",
            details={
                "prediction_type": prediction_type,
                "exchanger_id": exchanger_id,
                "model_id": model_id,
                "model_version": model_version,
                "inputs_hash": inputs_hash,
                "prediction_value": prediction_value,
                "prediction_hash": compute_sha256(prediction_value),
                "confidence": confidence,
                "duration_ms": duration_ms,
            },
        ))

    def log_recommendation(
        self,
        recommendation_type: str,
        exchanger_id: str,
        recommendation: str,
        priority: str,
        supporting_data_hash: str,
        work_order_id: Optional[str] = None,
    ) -> AuditLogEntry:
        """Log a recommendation issued to the user or CMMS."""
        return self.log(AuditEvent(
            event_type=AuditEventType.RECOMMENDATION_ISSUED,
            action=f"recommend_{recommendation_type}",
            severity=SeverityLevel.INFO,
            outcome=AuditOutcome.SUCCESS,
            message=f"Recommendation for {exchanger_id}: {recommendation[:100]}...",
            details={
                "recommendation_type": recommendation_type,
                "exchanger_id": exchanger_id,
                "recommendation": recommendation,
                "priority": priority,
                "supporting_data_hash": supporting_data_hash,
                "work_order_id": work_order_id,
            },
        ))

    def log_work_order_created(
        self,
        work_order_id: str,
        exchanger_id: str,
        recommendation_id: str,
        cmms_system: str,
        work_order_type: str,
    ) -> AuditLogEntry:
        """Log work order creation in CMMS."""
        return self.log(AuditEvent(
            event_type=AuditEventType.WORK_ORDER_CREATED,
            action="create_work_order",
            severity=SeverityLevel.INFO,
            outcome=AuditOutcome.SUCCESS,
            message=f"Work order {work_order_id} created for {exchanger_id}",
            details={
                "work_order_id": work_order_id,
                "exchanger_id": exchanger_id,
                "recommendation_id": recommendation_id,
                "cmms_system": cmms_system,
                "work_order_type": work_order_type,
            },
        ))

    # =========================================================================
    # SAFETY AND COMPLIANCE LOGGING
    # =========================================================================

    def log_safety_check(
        self,
        exchanger_id: str,
        check_type: str,
        passed: bool,
        violations: Optional[List[Dict[str, Any]]] = None,
        standard_reference: Optional[str] = None,
    ) -> AuditLogEntry:
        """Log safety check result."""
        severity = SeverityLevel.INFO if passed else SeverityLevel.WARNING
        if violations and any(v.get("severity") == "critical" for v in violations):
            severity = SeverityLevel.CRITICAL

        return self.log(AuditEvent(
            event_type=AuditEventType.SAFETY_CHECK_PERFORMED,
            action=f"check_{check_type}",
            severity=severity,
            outcome=AuditOutcome.SUCCESS if passed else AuditOutcome.FAILURE,
            message=f"Safety check '{check_type}' for {exchanger_id}: {'PASS' if passed else 'FAIL'}",
            details={
                "exchanger_id": exchanger_id,
                "check_type": check_type,
                "passed": passed,
                "violations": violations or [],
                "standard_reference": standard_reference,
            },
        ))

    def log_compliance_verification(
        self,
        framework: str,
        exchanger_id: str,
        is_compliant: bool,
        compliance_score: float,
        checks: Dict[str, bool],
        evidence_pack_id: Optional[str] = None,
    ) -> AuditLogEntry:
        """Log compliance verification result."""
        return self.log(AuditEvent(
            event_type=AuditEventType.COMPLIANCE_VERIFIED,
            action=f"verify_{framework}",
            severity=SeverityLevel.INFO if is_compliant else SeverityLevel.WARNING,
            outcome=AuditOutcome.SUCCESS if is_compliant else AuditOutcome.PARTIAL,
            message=f"Compliance verification for {exchanger_id}: {framework} - {'COMPLIANT' if is_compliant else 'NON-COMPLIANT'}",
            details={
                "framework": framework,
                "exchanger_id": exchanger_id,
                "is_compliant": is_compliant,
                "compliance_score": compliance_score,
                "checks": checks,
                "evidence_pack_id": evidence_pack_id,
            },
        ))

    # =========================================================================
    # DATA INGESTION LOGGING
    # =========================================================================

    def log_data_ingestion(
        self,
        source: str,
        exchanger_id: str,
        record_count: int,
        data_hash: str,
        validation_passed: bool,
        errors: Optional[List[str]] = None,
    ) -> AuditLogEntry:
        """Log data ingestion event."""
        return self.log(AuditEvent(
            event_type=AuditEventType.DATA_INGESTION,
            action="ingest_data",
            severity=SeverityLevel.INFO if validation_passed else SeverityLevel.WARNING,
            outcome=AuditOutcome.SUCCESS if validation_passed else AuditOutcome.PARTIAL,
            message=f"Ingested {record_count} records from {source} for {exchanger_id}",
            details={
                "source": source,
                "exchanger_id": exchanger_id,
                "record_count": record_count,
                "data_hash": data_hash,
                "validation_passed": validation_passed,
                "errors": errors or [],
            },
        ))

    def log_threshold_exceeded(
        self,
        exchanger_id: str,
        metric_name: str,
        current_value: float,
        threshold_value: float,
        unit: str,
        severity_level: str = "warning",
    ) -> AuditLogEntry:
        """Log threshold exceedance event."""
        severity = SeverityLevel.CRITICAL if severity_level == "critical" else SeverityLevel.WARNING

        return self.log(AuditEvent(
            event_type=AuditEventType.THRESHOLD_EXCEEDED,
            action="threshold_exceeded",
            severity=severity,
            outcome=AuditOutcome.SUCCESS,  # Event was logged successfully
            message=f"Threshold exceeded for {exchanger_id}: {metric_name} = {current_value} {unit} (limit: {threshold_value})",
            details={
                "exchanger_id": exchanger_id,
                "metric_name": metric_name,
                "current_value": current_value,
                "threshold_value": threshold_value,
                "unit": unit,
                "exceedance_ratio": current_value / threshold_value if threshold_value else None,
            },
        ))

    # =========================================================================
    # QUERY INTERFACE
    # =========================================================================

    def get_entries(
        self,
        event_type: Optional[AuditEventType] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        correlation_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[AuditLogEntry]:
        """
        Query audit log entries.

        Args:
            event_type: Filter by event type
            start_time: Filter by start time
            end_time: Filter by end time
            correlation_id: Filter by correlation ID
            limit: Maximum entries to return

        Returns:
            List of matching audit log entries
        """
        if self._storage is None:
            logger.warning("No storage configured, cannot query entries")
            return []

        return self._storage.query(
            event_type=event_type.value if event_type else None,
            start_time=start_time,
            end_time=end_time,
            correlation_id=correlation_id,
            limit=limit,
        )


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def get_audit_logger() -> AuditLogger:
    """Get the singleton audit logger instance."""
    return AuditLogger()


def create_audit_logger(
    storage_dir: str = "audit_logs",
    app_name: str = "GL-014-EXCHANGERPRO",
    environment: str = "production",
) -> AuditLogger:
    """
    Create and configure an audit logger with file storage.

    Args:
        storage_dir: Directory for audit log files
        app_name: Application name
        environment: Deployment environment

    Returns:
        Configured AuditLogger instance
    """
    storage = FileAuditStorage(directory=storage_dir)
    audit_logger = AuditLogger(
        storage=storage,
        app_name=app_name,
        environment=environment,
    )
    return audit_logger
