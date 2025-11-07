"""
GreenLang Security Audit Logger
================================

Enterprise-grade audit logging for security-sensitive operations.
Provides structured logging for authentication, authorization, data access,
and agent execution with SIEM-friendly JSON output.

Author: GreenLang Security Team
Phase: 3 - Security Hardening
"""

import json
import logging
import time
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, Union

from pydantic import BaseModel, Field


class AuditEventType(str, Enum):
    """Types of audit events."""

    # Authentication events
    AUTH_SUCCESS = "auth.success"
    AUTH_FAILURE = "auth.failure"
    AUTH_LOGOUT = "auth.logout"
    AUTH_TOKEN_CREATED = "auth.token.created"
    AUTH_TOKEN_REVOKED = "auth.token.revoked"

    # Authorization events
    AUTHZ_ALLOWED = "authz.allowed"
    AUTHZ_DENIED = "authz.denied"
    AUTHZ_POLICY_EVALUATED = "authz.policy.evaluated"

    # Configuration events
    CONFIG_CHANGED = "config.changed"
    CONFIG_LOADED = "config.loaded"
    CONFIG_VALIDATED = "config.validated"

    # Data access events
    DATA_READ = "data.read"
    DATA_WRITE = "data.write"
    DATA_DELETE = "data.delete"
    DATA_EXPORT = "data.export"

    # Agent execution events
    AGENT_STARTED = "agent.started"
    AGENT_COMPLETED = "agent.completed"
    AGENT_FAILED = "agent.failed"
    AGENT_CANCELLED = "agent.cancelled"

    # Pack management events
    PACK_INSTALLED = "pack.installed"
    PACK_REMOVED = "pack.removed"
    PACK_VERIFIED = "pack.verified"
    PACK_SIGNATURE_FAILED = "pack.signature.failed"

    # Security events
    SECURITY_VIOLATION = "security.violation"
    SECURITY_SCAN_COMPLETED = "security.scan.completed"
    SECURITY_ALERT = "security.alert"

    # System events
    SYSTEM_STARTED = "system.started"
    SYSTEM_STOPPED = "system.stopped"
    SYSTEM_ERROR = "system.error"


class AuditSeverity(str, Enum):
    """Severity levels for audit events."""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AuditEvent(BaseModel):
    """Structured audit event model."""

    # Core fields
    timestamp: str = Field(description="ISO 8601 timestamp with timezone")
    event_type: AuditEventType = Field(description="Type of audit event")
    severity: AuditSeverity = Field(description="Severity level")

    # Actor information
    user_id: Optional[str] = Field(None, description="User ID if authenticated")
    username: Optional[str] = Field(None, description="Username if known")
    session_id: Optional[str] = Field(None, description="Session ID")
    ip_address: Optional[str] = Field(None, description="Source IP address")

    # Resource information
    resource_type: Optional[str] = Field(None, description="Type of resource accessed")
    resource_id: Optional[str] = Field(None, description="Resource identifier")
    resource_name: Optional[str] = Field(None, description="Resource name")

    # Action details
    action: str = Field(description="Action performed")
    result: str = Field(description="Result of action (success/failure)")
    reason: Optional[str] = Field(None, description="Reason for result")

    # Additional context
    details: Dict[str, Any] = Field(default_factory=dict, description="Additional event details")

    # Tracing
    trace_id: Optional[str] = Field(None, description="Distributed trace ID")
    span_id: Optional[str] = Field(None, description="Span ID within trace")

    # System metadata
    hostname: Optional[str] = Field(None, description="Hostname of system")
    service: str = Field(default="greenlang", description="Service name")
    version: Optional[str] = Field(None, description="Service version")

    class Config:
        """Pydantic config."""

        use_enum_values = True


class AuditLogger:
    """
    Enterprise audit logger for GreenLang.

    Provides structured logging for security-sensitive operations with
    JSON output suitable for SIEM integration.

    Example:
        >>> audit = AuditLogger()
        >>> audit.log_auth_success(
        ...     user_id="user123",
        ...     username="john.doe",
        ...     ip_address="192.168.1.100"
        ... )
        >>> audit.log_agent_execution(
        ...     agent_name="FuelAgent",
        ...     user_id="user123",
        ...     result="success",
        ...     details={"emissions_kg_co2": 123.45}
        ... )
    """

    def __init__(
        self,
        log_file: Optional[Union[str, Path]] = None,
        service_name: str = "greenlang",
        service_version: Optional[str] = None,
        enable_console: bool = False,
    ):
        """
        Initialize audit logger.

        Args:
            log_file: Path to audit log file. If None, uses default location.
            service_name: Name of service for audit events.
            service_version: Version of service.
            enable_console: If True, also log to console.
        """
        self.service_name = service_name
        self.service_version = service_version or self._get_version()

        # Set up audit log file
        if log_file is None:
            log_file = Path.home() / ".greenlang" / "logs" / "audit.jsonl"

        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

        # Set up Python logger
        self.logger = logging.getLogger(f"{__name__}.{service_name}")
        self.logger.setLevel(logging.INFO)

        # Remove existing handlers
        self.logger.handlers.clear()

        # Add file handler for JSON logs
        file_handler = logging.FileHandler(self.log_file, encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter("%(message)s"))
        self.logger.addHandler(file_handler)

        # Optionally add console handler
        if enable_console:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            console_handler.setFormatter(
                logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            )
            self.logger.addHandler(console_handler)

    def _get_version(self) -> str:
        """Get GreenLang version."""
        try:
            from greenlang import __version__

            return __version__
        except ImportError:
            return "unknown"

    def _create_event(
        self,
        event_type: AuditEventType,
        severity: AuditSeverity,
        action: str,
        result: str,
        **kwargs,
    ) -> AuditEvent:
        """Create audit event with common fields."""
        return AuditEvent(
            timestamp=datetime.now(timezone.utc).isoformat(),
            event_type=event_type,
            severity=severity,
            action=action,
            result=result,
            service=self.service_name,
            version=self.service_version,
            **kwargs,
        )

    def log_event(self, event: AuditEvent) -> None:
        """Log an audit event."""
        # Convert to JSON
        event_json = event.model_dump_json()

        # Log to file
        self.logger.info(event_json)

    # Authentication methods

    def log_auth_success(
        self,
        user_id: str,
        username: Optional[str] = None,
        ip_address: Optional[str] = None,
        session_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log successful authentication."""
        event = self._create_event(
            event_type=AuditEventType.AUTH_SUCCESS,
            severity=AuditSeverity.INFO,
            action="authenticate",
            result="success",
            user_id=user_id,
            username=username,
            ip_address=ip_address,
            session_id=session_id,
            details=details or {},
        )
        self.log_event(event)

    def log_auth_failure(
        self,
        username: Optional[str] = None,
        ip_address: Optional[str] = None,
        reason: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log failed authentication attempt."""
        event = self._create_event(
            event_type=AuditEventType.AUTH_FAILURE,
            severity=AuditSeverity.WARNING,
            action="authenticate",
            result="failure",
            username=username,
            ip_address=ip_address,
            reason=reason,
            details=details or {},
        )
        self.log_event(event)

    # Authorization methods

    def log_authz_decision(
        self,
        user_id: str,
        resource_type: str,
        resource_id: str,
        action: str,
        allowed: bool,
        reason: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log authorization decision."""
        event = self._create_event(
            event_type=AuditEventType.AUTHZ_ALLOWED if allowed else AuditEventType.AUTHZ_DENIED,
            severity=AuditSeverity.INFO if allowed else AuditSeverity.WARNING,
            action=action,
            result="allowed" if allowed else "denied",
            user_id=user_id,
            resource_type=resource_type,
            resource_id=resource_id,
            reason=reason,
            details=details or {},
        )
        self.log_event(event)

    # Configuration methods

    def log_config_change(
        self,
        user_id: str,
        config_key: str,
        old_value: Any,
        new_value: Any,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log configuration change."""
        event = self._create_event(
            event_type=AuditEventType.CONFIG_CHANGED,
            severity=AuditSeverity.INFO,
            action="update_config",
            result="success",
            user_id=user_id,
            resource_type="configuration",
            resource_id=config_key,
            details={
                "old_value": str(old_value),
                "new_value": str(new_value),
                **(details or {}),
            },
        )
        self.log_event(event)

    # Data access methods

    def log_data_access(
        self,
        user_id: str,
        data_type: str,
        data_id: str,
        operation: str,  # read/write/delete/export
        result: str = "success",
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log data access operation."""
        event_type_map = {
            "read": AuditEventType.DATA_READ,
            "write": AuditEventType.DATA_WRITE,
            "delete": AuditEventType.DATA_DELETE,
            "export": AuditEventType.DATA_EXPORT,
        }

        event = self._create_event(
            event_type=event_type_map.get(operation, AuditEventType.DATA_READ),
            severity=AuditSeverity.INFO,
            action=f"{operation}_data",
            result=result,
            user_id=user_id,
            resource_type=data_type,
            resource_id=data_id,
            details=details or {},
        )
        self.log_event(event)

    # Agent execution methods

    def log_agent_execution(
        self,
        agent_name: str,
        user_id: Optional[str] = None,
        result: str = "success",
        execution_time_ms: Optional[float] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log agent execution."""
        event_type_map = {
            "success": AuditEventType.AGENT_COMPLETED,
            "failure": AuditEventType.AGENT_FAILED,
            "cancelled": AuditEventType.AGENT_CANCELLED,
        }

        severity_map = {
            "success": AuditSeverity.INFO,
            "failure": AuditSeverity.ERROR,
            "cancelled": AuditSeverity.WARNING,
        }

        event_details = details or {}
        if execution_time_ms is not None:
            event_details["execution_time_ms"] = execution_time_ms

        event = self._create_event(
            event_type=event_type_map.get(result, AuditEventType.AGENT_COMPLETED),
            severity=severity_map.get(result, AuditSeverity.INFO),
            action="execute_agent",
            result=result,
            user_id=user_id,
            resource_type="agent",
            resource_name=agent_name,
            details=event_details,
        )
        self.log_event(event)

    # Security methods

    def log_security_violation(
        self,
        violation_type: str,
        description: str,
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log security violation."""
        event = self._create_event(
            event_type=AuditEventType.SECURITY_VIOLATION,
            severity=AuditSeverity.CRITICAL,
            action="security_check",
            result="violation",
            user_id=user_id,
            ip_address=ip_address,
            reason=description,
            details={"violation_type": violation_type, **(details or {})},
        )
        self.log_event(event)


# Global audit logger instance
_global_audit_logger: Optional[AuditLogger] = None


def get_audit_logger() -> AuditLogger:
    """Get global audit logger instance."""
    global _global_audit_logger
    if _global_audit_logger is None:
        _global_audit_logger = AuditLogger()
    return _global_audit_logger


def configure_audit_logger(
    log_file: Optional[Union[str, Path]] = None,
    service_name: str = "greenlang",
    service_version: Optional[str] = None,
    enable_console: bool = False,
) -> AuditLogger:
    """Configure global audit logger."""
    global _global_audit_logger
    _global_audit_logger = AuditLogger(
        log_file=log_file,
        service_name=service_name,
        service_version=service_version,
        enable_console=enable_console,
    )
    return _global_audit_logger
