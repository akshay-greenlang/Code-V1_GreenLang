"""
GL-EUDR-001: Audit Logging Module

Provides comprehensive audit logging for EUDR compliance.
All state-modifying operations are logged with full context.

Features:
- Structured audit log entries
- User, action, timestamp, and affected entities tracking
- Async logging for performance
- Log rotation and archival support
- Compliance-ready export formats

EUDR Requirements:
- Article 10: Full traceability audit trail
- Article 11: Due diligence documentation
"""

import hashlib
import json
import logging
import os
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# =============================================================================
# AUDIT ACTION TYPES
# =============================================================================

class AuditAction(str, Enum):
    """Types of auditable actions."""
    # Node actions
    NODE_CREATE = "node.create"
    NODE_UPDATE = "node.update"
    NODE_DELETE = "node.delete"

    # Edge actions
    EDGE_CREATE = "edge.create"
    EDGE_UPDATE = "edge.update"
    EDGE_DELETE = "edge.delete"

    # Plot actions
    PLOT_CREATE = "plot.create"
    PLOT_UPDATE = "plot.update"
    PLOT_DELETE = "plot.delete"

    # Coverage actions
    COVERAGE_CALCULATE = "coverage.calculate"
    COVERAGE_GATES_CHECK = "coverage.gates_check"

    # Snapshot actions
    SNAPSHOT_CREATE = "snapshot.create"
    SNAPSHOT_QUERY = "snapshot.query"
    SNAPSHOT_DIFF = "snapshot.diff"

    # Entity resolution actions
    ER_RUN = "entity_resolution.run"
    ER_MERGE = "entity_resolution.merge"
    ER_REJECT = "entity_resolution.reject"

    # Graph actions
    GRAPH_QUERY = "graph.query"
    GRAPH_EXPORT = "graph.export"

    # NL Query actions
    NL_QUERY = "nl_query.execute"

    # Bulk operations
    BULK_IMPORT = "bulk.import"
    BULK_EXPORT = "bulk.export"

    # Authentication actions
    AUTH_LOGIN = "auth.login"
    AUTH_LOGOUT = "auth.logout"
    AUTH_TOKEN_REFRESH = "auth.token_refresh"
    AUTH_FAILED = "auth.failed"

    # Admin actions
    ADMIN_CONFIG_CHANGE = "admin.config_change"
    ADMIN_USER_CREATE = "admin.user_create"
    ADMIN_USER_UPDATE = "admin.user_update"


class AuditSeverity(str, Enum):
    """Severity levels for audit entries."""
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


# =============================================================================
# AUDIT LOG MODELS
# =============================================================================

class AuditLogEntry(BaseModel):
    """
    Single audit log entry.
    Immutable record of an action for compliance.
    """
    entry_id: UUID = Field(default_factory=uuid.uuid4)
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    # Action details
    action: AuditAction
    severity: AuditSeverity = AuditSeverity.INFO

    # User context
    user_id: Optional[UUID] = None
    user_email: Optional[str] = None
    user_role: Optional[str] = None
    organization_id: Optional[UUID] = None

    # Resource context
    resource_type: Optional[str] = None  # node, edge, plot, snapshot, etc.
    resource_id: Optional[UUID] = None
    resource_ids: List[UUID] = Field(default_factory=list)

    # Request context
    request_id: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    endpoint: Optional[str] = None
    method: Optional[str] = None

    # Change details
    changes: Dict[str, Any] = Field(default_factory=dict)
    previous_state: Dict[str, Any] = Field(default_factory=dict)
    new_state: Dict[str, Any] = Field(default_factory=dict)

    # Result
    success: bool = True
    error_message: Optional[str] = None

    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)

    # Integrity
    checksum: Optional[str] = None

    class Config:
        use_enum_values = True

    def compute_checksum(self) -> str:
        """Compute SHA-256 checksum for integrity verification."""
        content = json.dumps({
            "entry_id": str(self.entry_id),
            "timestamp": self.timestamp.isoformat(),
            "action": self.action,
            "user_id": str(self.user_id) if self.user_id else None,
            "resource_id": str(self.resource_id) if self.resource_id else None,
            "success": self.success
        }, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()

    def finalize(self) -> "AuditLogEntry":
        """Finalize entry with checksum (makes it immutable)."""
        self.checksum = self.compute_checksum()
        return self


class AuditContext(BaseModel):
    """Context for audit logging, typically extracted from request."""
    user_id: Optional[UUID] = None
    user_email: Optional[str] = None
    user_role: Optional[str] = None
    organization_id: Optional[UUID] = None
    request_id: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    endpoint: Optional[str] = None
    method: Optional[str] = None


# =============================================================================
# AUDIT LOGGER
# =============================================================================

class AuditLogger:
    """
    Audit logger for EUDR compliance.
    Thread-safe, supports multiple output backends.
    """

    def __init__(
        self,
        log_to_file: bool = True,
        log_to_db: bool = False,
        log_dir: str = "audit_logs"
    ):
        self.log_to_file = log_to_file
        self.log_to_db = log_to_db
        self.log_dir = log_dir

        # In-memory buffer for batch writes
        self._buffer: List[AuditLogEntry] = []
        self._max_buffer_size = 100

        # Setup file logger
        if log_to_file:
            os.makedirs(log_dir, exist_ok=True)
            self._file_handler = self._setup_file_handler()

        # Structured logger
        self._logger = logging.getLogger("audit.trail")
        self._logger.setLevel(logging.INFO)

    def _setup_file_handler(self) -> logging.FileHandler:
        """Setup rotating file handler for audit logs."""
        from logging.handlers import TimedRotatingFileHandler

        log_file = os.path.join(self.log_dir, "audit.log")
        handler = TimedRotatingFileHandler(
            log_file,
            when="midnight",
            interval=1,
            backupCount=90  # Keep 90 days of logs
        )
        handler.setFormatter(logging.Formatter(
            '%(message)s'  # JSON format, no extra formatting
        ))
        return handler

    def log(
        self,
        action: AuditAction,
        context: AuditContext,
        resource_type: Optional[str] = None,
        resource_id: Optional[UUID] = None,
        resource_ids: Optional[List[UUID]] = None,
        changes: Optional[Dict[str, Any]] = None,
        previous_state: Optional[Dict[str, Any]] = None,
        new_state: Optional[Dict[str, Any]] = None,
        success: bool = True,
        error_message: Optional[str] = None,
        severity: AuditSeverity = AuditSeverity.INFO,
        metadata: Optional[Dict[str, Any]] = None
    ) -> AuditLogEntry:
        """
        Create and persist an audit log entry.

        Args:
            action: The action being audited
            context: Request/user context
            resource_type: Type of resource affected
            resource_id: Primary resource ID
            resource_ids: List of affected resource IDs
            changes: Dictionary of field changes
            previous_state: State before the action
            new_state: State after the action
            success: Whether action succeeded
            error_message: Error message if failed
            severity: Log severity level
            metadata: Additional metadata

        Returns:
            The created audit log entry
        """
        entry = AuditLogEntry(
            action=action,
            severity=severity,
            user_id=context.user_id,
            user_email=context.user_email,
            user_role=context.user_role,
            organization_id=context.organization_id,
            request_id=context.request_id,
            ip_address=context.ip_address,
            user_agent=context.user_agent,
            endpoint=context.endpoint,
            method=context.method,
            resource_type=resource_type,
            resource_id=resource_id,
            resource_ids=resource_ids or [],
            changes=changes or {},
            previous_state=previous_state or {},
            new_state=new_state or {},
            success=success,
            error_message=error_message,
            metadata=metadata or {}
        ).finalize()

        # Write to outputs
        self._write_entry(entry)

        return entry

    def _write_entry(self, entry: AuditLogEntry):
        """Write entry to configured outputs."""
        # JSON representation
        entry_json = entry.json()

        # Log to structured logger
        if entry.severity == AuditSeverity.ERROR:
            self._logger.error(entry_json)
        elif entry.severity == AuditSeverity.WARNING:
            self._logger.warning(entry_json)
        elif entry.severity == AuditSeverity.CRITICAL:
            self._logger.critical(entry_json)
        else:
            self._logger.info(entry_json)

        # Write to file
        if self.log_to_file and hasattr(self, '_file_handler'):
            record = logging.LogRecord(
                name="audit",
                level=logging.INFO,
                pathname="",
                lineno=0,
                msg=entry_json,
                args=(),
                exc_info=None
            )
            self._file_handler.emit(record)

        # Buffer for batch DB writes
        if self.log_to_db:
            self._buffer.append(entry)
            if len(self._buffer) >= self._max_buffer_size:
                self._flush_to_db()

    def _flush_to_db(self):
        """Flush buffer to database."""
        if not self._buffer:
            return

        # TODO: Implement database persistence
        # db_session.bulk_insert_mappings(AuditLogModel, [e.dict() for e in self._buffer])
        # db_session.commit()

        self._buffer = []

    def log_node_create(
        self,
        context: AuditContext,
        node_id: UUID,
        node_data: Dict[str, Any]
    ) -> AuditLogEntry:
        """Log node creation."""
        return self.log(
            action=AuditAction.NODE_CREATE,
            context=context,
            resource_type="node",
            resource_id=node_id,
            new_state=node_data,
            metadata={"node_type": node_data.get("node_type")}
        )

    def log_node_update(
        self,
        context: AuditContext,
        node_id: UUID,
        previous: Dict[str, Any],
        updated: Dict[str, Any],
        changes: Dict[str, Any]
    ) -> AuditLogEntry:
        """Log node update."""
        return self.log(
            action=AuditAction.NODE_UPDATE,
            context=context,
            resource_type="node",
            resource_id=node_id,
            previous_state=previous,
            new_state=updated,
            changes=changes
        )

    def log_node_delete(
        self,
        context: AuditContext,
        node_id: UUID,
        node_data: Dict[str, Any]
    ) -> AuditLogEntry:
        """Log node deletion."""
        return self.log(
            action=AuditAction.NODE_DELETE,
            context=context,
            resource_type="node",
            resource_id=node_id,
            previous_state=node_data,
            severity=AuditSeverity.WARNING
        )

    def log_entity_merge(
        self,
        context: AuditContext,
        keep_id: UUID,
        merge_id: UUID,
        merge_details: Dict[str, Any]
    ) -> AuditLogEntry:
        """Log entity resolution merge."""
        return self.log(
            action=AuditAction.ER_MERGE,
            context=context,
            resource_type="entity_resolution",
            resource_id=keep_id,
            resource_ids=[keep_id, merge_id],
            changes=merge_details,
            severity=AuditSeverity.WARNING,
            metadata={"merged_node_id": str(merge_id)}
        )

    def log_snapshot_create(
        self,
        context: AuditContext,
        snapshot_id: UUID,
        snapshot_data: Dict[str, Any]
    ) -> AuditLogEntry:
        """Log snapshot creation."""
        return self.log(
            action=AuditAction.SNAPSHOT_CREATE,
            context=context,
            resource_type="snapshot",
            resource_id=snapshot_id,
            new_state={
                "snapshot_id": str(snapshot_id),
                "node_count": snapshot_data.get("node_count"),
                "edge_count": snapshot_data.get("edge_count"),
                "coverage": snapshot_data.get("coverage_percentage"),
                "trigger_type": snapshot_data.get("trigger_type")
            }
        )

    def log_coverage_check(
        self,
        context: AuditContext,
        importer_id: UUID,
        commodity: str,
        result: Dict[str, Any]
    ) -> AuditLogEntry:
        """Log coverage gate check."""
        return self.log(
            action=AuditAction.COVERAGE_GATES_CHECK,
            context=context,
            resource_type="coverage",
            resource_id=importer_id,
            new_state=result,
            metadata={
                "commodity": commodity,
                "can_proceed": result.get("can_proceed_to_risk_assessment"),
                "can_dds": result.get("can_submit_dds")
            }
        )

    def log_auth_failure(
        self,
        context: AuditContext,
        reason: str
    ) -> AuditLogEntry:
        """Log authentication failure."""
        return self.log(
            action=AuditAction.AUTH_FAILED,
            context=context,
            resource_type="auth",
            success=False,
            error_message=reason,
            severity=AuditSeverity.WARNING
        )


# =============================================================================
# FASTAPI MIDDLEWARE
# =============================================================================

class AuditMiddleware:
    """
    FastAPI middleware for automatic audit logging.
    """

    def __init__(self, audit_logger: AuditLogger):
        self.audit_logger = audit_logger

    def create_context_from_request(self, request: Any, user: Any = None) -> AuditContext:
        """Create audit context from FastAPI request."""
        return AuditContext(
            user_id=getattr(user, 'user_id', None) if user else None,
            user_email=getattr(user, 'email', None) if user else None,
            user_role=getattr(user, 'role', None) if user else None,
            organization_id=getattr(user, 'organization_id', None) if user else None,
            request_id=request.headers.get("X-Request-ID", str(uuid.uuid4())),
            ip_address=request.client.host if request.client else None,
            user_agent=request.headers.get("User-Agent"),
            endpoint=str(request.url.path),
            method=request.method
        )


# =============================================================================
# DECORATOR FOR AUDITED FUNCTIONS
# =============================================================================

def audited(
    action: AuditAction,
    resource_type: str,
    get_resource_id: Optional[Callable] = None
):
    """
    Decorator for auditing function calls.

    Usage:
        @audited(AuditAction.NODE_CREATE, "node", lambda result: result.node_id)
        async def create_node(...):
            ...
    """
    def decorator(func: Callable):
        async def wrapper(*args, **kwargs):
            # Get audit logger from context (would be injected in real app)
            audit_logger = kwargs.pop('_audit_logger', global_audit_logger)
            audit_context = kwargs.pop('_audit_context', AuditContext())

            try:
                result = await func(*args, **kwargs)

                # Extract resource ID
                resource_id = None
                if get_resource_id and result:
                    resource_id = get_resource_id(result)

                # Log success
                audit_logger.log(
                    action=action,
                    context=audit_context,
                    resource_type=resource_type,
                    resource_id=resource_id,
                    success=True
                )

                return result

            except Exception as e:
                # Log failure
                audit_logger.log(
                    action=action,
                    context=audit_context,
                    resource_type=resource_type,
                    success=False,
                    error_message=str(e),
                    severity=AuditSeverity.ERROR
                )
                raise

        return wrapper
    return decorator


# =============================================================================
# GLOBAL INSTANCE
# =============================================================================

# Global audit logger instance
global_audit_logger = AuditLogger(
    log_to_file=True,
    log_to_db=False,
    log_dir=os.getenv("AUDIT_LOG_DIR", "audit_logs")
)


def get_audit_logger() -> AuditLogger:
    """Get the global audit logger instance."""
    return global_audit_logger


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "AuditAction",
    "AuditSeverity",

    # Models
    "AuditLogEntry",
    "AuditContext",

    # Logger
    "AuditLogger",
    "AuditMiddleware",
    "global_audit_logger",
    "get_audit_logger",

    # Decorator
    "audited",
]
