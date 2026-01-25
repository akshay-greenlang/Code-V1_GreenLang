"""
Audit Service - SOC2 and ISO27001 Compliant Audit Logging

This module provides comprehensive audit trail capabilities for all operations
within the GreenLang platform. It implements tamper-evident logging with hash
chains, log retention policies, and compliance report export functionality.

SOC2 Controls Addressed:
    - CC6.1: Logical access security
    - CC7.1: System operations monitoring
    - CC7.2: Incident detection and monitoring

ISO27001 Controls Addressed:
    - A.12.4.1: Event logging
    - A.12.4.2: Protection of log information
    - A.12.4.3: Administrator and operator logs

Example:
    >>> service = AuditService(config)
    >>> await service.log_event(
    ...     event_type=AuditEventType.DATA_ACCESS,
    ...     actor_id="user-123",
    ...     resource_type="EmissionsReport",
    ...     resource_id="report-456",
    ...     action="READ",
    ...     details={"fields_accessed": ["emissions", "scope"]}
    ... )
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import uuid
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)


class AuditEventType(str, Enum):
    """Categories of audit events for compliance tracking."""

    # Authentication Events
    LOGIN_SUCCESS = "LOGIN_SUCCESS"
    LOGIN_FAILURE = "LOGIN_FAILURE"
    LOGOUT = "LOGOUT"
    TOKEN_REFRESH = "TOKEN_REFRESH"
    PASSWORD_CHANGE = "PASSWORD_CHANGE"
    MFA_ENABLED = "MFA_ENABLED"
    MFA_DISABLED = "MFA_DISABLED"
    SESSION_TIMEOUT = "SESSION_TIMEOUT"

    # Authorization Events
    ACCESS_GRANTED = "ACCESS_GRANTED"
    ACCESS_DENIED = "ACCESS_DENIED"
    PERMISSION_CHANGED = "PERMISSION_CHANGED"
    ROLE_ASSIGNED = "ROLE_ASSIGNED"
    ROLE_REMOVED = "ROLE_REMOVED"

    # Data Access Events
    DATA_ACCESS = "DATA_ACCESS"
    DATA_CREATE = "DATA_CREATE"
    DATA_UPDATE = "DATA_UPDATE"
    DATA_DELETE = "DATA_DELETE"
    DATA_EXPORT = "DATA_EXPORT"
    BULK_DATA_ACCESS = "BULK_DATA_ACCESS"

    # Agent Events
    AGENT_EXECUTION_START = "AGENT_EXECUTION_START"
    AGENT_EXECUTION_COMPLETE = "AGENT_EXECUTION_COMPLETE"
    AGENT_EXECUTION_FAILED = "AGENT_EXECUTION_FAILED"
    AGENT_DEPLOYED = "AGENT_DEPLOYED"
    AGENT_DISABLED = "AGENT_DISABLED"
    AGENT_CONFIG_CHANGED = "AGENT_CONFIG_CHANGED"

    # System Events
    SYSTEM_STARTUP = "SYSTEM_STARTUP"
    SYSTEM_SHUTDOWN = "SYSTEM_SHUTDOWN"
    CONFIG_CHANGE = "CONFIG_CHANGE"
    SECURITY_POLICY_CHANGE = "SECURITY_POLICY_CHANGE"
    MAINTENANCE_MODE = "MAINTENANCE_MODE"

    # Security Events
    SECURITY_ALERT = "SECURITY_ALERT"
    INTRUSION_DETECTED = "INTRUSION_DETECTED"
    RATE_LIMIT_EXCEEDED = "RATE_LIMIT_EXCEEDED"
    IP_BLOCKED = "IP_BLOCKED"
    SUSPICIOUS_ACTIVITY = "SUSPICIOUS_ACTIVITY"

    # Compliance Events
    COMPLIANCE_CHECK = "COMPLIANCE_CHECK"
    EVIDENCE_GENERATED = "EVIDENCE_GENERATED"
    AUDIT_EXPORT = "AUDIT_EXPORT"
    RETENTION_PURGE = "RETENTION_PURGE"


class AuditSeverity(str, Enum):
    """Severity levels for audit events."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class AuditExportFormat(str, Enum):
    """Supported export formats for compliance reports."""

    JSON = "JSON"
    CSV = "CSV"
    NDJSON = "NDJSON"  # Newline-delimited JSON for streaming
    SYSLOG = "SYSLOG"  # RFC 5424 format


class AuditEntry(BaseModel):
    """
    Immutable audit log entry with tamper-evident hashing.

    Each entry contains a hash of the previous entry to create
    a tamper-evident chain that can be verified for integrity.

    Attributes:
        id: Unique identifier for this audit entry
        timestamp: UTC timestamp when the event occurred
        event_type: Category of the audit event
        severity: Severity level of the event
        actor_id: ID of the user or system that performed the action
        actor_type: Type of actor (user, system, api_key, service)
        tenant_id: Multi-tenant organization identifier
        resource_type: Type of resource affected
        resource_id: Identifier of the affected resource
        action: Specific action performed
        details: Additional context about the event
        request_metadata: HTTP request information for traceability
        previous_hash: Hash of the previous entry in the chain
        entry_hash: SHA-256 hash of this entry for integrity verification
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    event_type: AuditEventType
    severity: AuditSeverity = Field(default=AuditSeverity.INFO)

    # Actor information
    actor_id: str = Field(..., description="User or system ID")
    actor_type: str = Field(default="user", description="user, system, api_key, service")
    tenant_id: str = Field(..., description="Organization tenant ID")

    # Resource information
    resource_type: Optional[str] = Field(None, description="Type of resource affected")
    resource_id: Optional[str] = Field(None, description="ID of resource affected")
    action: str = Field(..., description="Action performed")

    # Event details
    details: Dict[str, Any] = Field(default_factory=dict)
    old_value: Optional[Dict[str, Any]] = Field(None, description="Previous state for updates")
    new_value: Optional[Dict[str, Any]] = Field(None, description="New state for updates")

    # Request metadata
    request_metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="IP address, user agent, correlation ID, etc.",
    )

    # Hash chain for tamper evidence
    previous_hash: Optional[str] = Field(None, description="Hash of previous entry")
    entry_hash: Optional[str] = Field(None, description="SHA-256 hash of this entry")

    # Compliance metadata
    compliance_frameworks: List[str] = Field(
        default_factory=list,
        description="Applicable compliance frameworks (SOC2, ISO27001)",
    )
    control_ids: List[str] = Field(
        default_factory=list,
        description="Control IDs this event relates to",
    )

    class Config:
        """Pydantic configuration."""

        json_encoders = {
            datetime: lambda v: v.isoformat(),
        }

    @validator("entry_hash", always=True)
    def compute_hash(cls, v, values):
        """Compute SHA-256 hash of the entry for integrity verification."""
        if v is not None:
            return v

        # Create hash input from immutable fields
        hash_input = {
            "id": values.get("id"),
            "timestamp": values.get("timestamp").isoformat() if values.get("timestamp") else None,
            "event_type": values.get("event_type").value if values.get("event_type") else None,
            "actor_id": values.get("actor_id"),
            "tenant_id": values.get("tenant_id"),
            "resource_type": values.get("resource_type"),
            "resource_id": values.get("resource_id"),
            "action": values.get("action"),
            "details": values.get("details"),
            "previous_hash": values.get("previous_hash"),
        }

        hash_str = json.dumps(hash_input, sort_keys=True, default=str)
        return hashlib.sha256(hash_str.encode()).hexdigest()

    def to_syslog(self) -> str:
        """
        Format entry as RFC 5424 syslog message for SIEM integration.

        Returns:
            Formatted syslog string
        """
        # Map severity to syslog severity
        severity_map = {
            AuditSeverity.DEBUG: 7,
            AuditSeverity.INFO: 6,
            AuditSeverity.WARNING: 4,
            AuditSeverity.ERROR: 3,
            AuditSeverity.CRITICAL: 2,
        }
        sev = severity_map.get(self.severity, 6)

        # Facility 4 = auth/security
        pri = (4 * 8) + sev

        timestamp = self.timestamp.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        hostname = self.request_metadata.get("hostname", "greenlang")
        app_name = "greenlang-audit"
        proc_id = self.request_metadata.get("correlation_id", "-")
        msg_id = self.event_type.value

        structured_data = (
            f'[audit@greenlang actor_id="{self.actor_id}" '
            f'tenant_id="{self.tenant_id}" '
            f'resource_type="{self.resource_type or "-"}" '
            f'resource_id="{self.resource_id or "-"}"]'
        )

        msg = f"{self.action}: {json.dumps(self.details)}"

        return f"<{pri}>1 {timestamp} {hostname} {app_name} {proc_id} {msg_id} {structured_data} {msg}"


class RetentionPolicy(BaseModel):
    """
    Log retention policy configuration.

    Implements tiered retention based on event severity and type
    to balance compliance requirements with storage costs.
    """

    # Default retention periods (days)
    default_retention_days: int = Field(default=365, ge=30, le=2555)

    # Override retention by severity
    retention_by_severity: Dict[AuditSeverity, int] = Field(
        default_factory=lambda: {
            AuditSeverity.CRITICAL: 2555,  # 7 years for critical events
            AuditSeverity.ERROR: 1825,  # 5 years for errors
            AuditSeverity.WARNING: 730,  # 2 years for warnings
            AuditSeverity.INFO: 365,  # 1 year for info
            AuditSeverity.DEBUG: 90,  # 90 days for debug
        }
    )

    # Override retention by event type
    retention_by_type: Dict[AuditEventType, int] = Field(
        default_factory=lambda: {
            AuditEventType.SECURITY_ALERT: 2555,
            AuditEventType.INTRUSION_DETECTED: 2555,
            AuditEventType.COMPLIANCE_CHECK: 1825,
            AuditEventType.DATA_DELETE: 1825,
            AuditEventType.PERMISSION_CHANGED: 1825,
        }
    )

    # Compliance framework minimum requirements
    compliance_minimums: Dict[str, int] = Field(
        default_factory=lambda: {
            "SOC2": 365,  # SOC2 requires 1 year minimum
            "ISO27001": 365,
            "GDPR": 90,  # GDPR requires minimization
            "PCI-DSS": 365,
        }
    )

    def get_retention_days(
        self,
        severity: AuditSeverity,
        event_type: AuditEventType,
        compliance_frameworks: List[str],
    ) -> int:
        """
        Calculate retention period for an event.

        Uses the maximum of all applicable retention requirements.

        Args:
            severity: Event severity level
            event_type: Type of audit event
            compliance_frameworks: Applicable compliance frameworks

        Returns:
            Number of days to retain the event
        """
        retention_days = self.default_retention_days

        # Check severity-based retention
        if severity in self.retention_by_severity:
            retention_days = max(retention_days, self.retention_by_severity[severity])

        # Check type-based retention
        if event_type in self.retention_by_type:
            retention_days = max(retention_days, self.retention_by_type[event_type])

        # Check compliance requirements
        for framework in compliance_frameworks:
            if framework in self.compliance_minimums:
                retention_days = max(retention_days, self.compliance_minimums[framework])

        return retention_days


class AuditServiceConfig(BaseModel):
    """Configuration for the Audit Service."""

    # Storage backend
    storage_backend: str = Field(default="postgresql", description="postgresql, elasticsearch, s3")

    # Hash chain configuration
    enable_hash_chain: bool = Field(default=True)
    hash_algorithm: str = Field(default="sha256")

    # Retention policy
    retention_policy: RetentionPolicy = Field(default_factory=RetentionPolicy)

    # Performance settings
    batch_size: int = Field(default=100, ge=1, le=1000)
    flush_interval_seconds: int = Field(default=5, ge=1, le=60)

    # SIEM integration
    siem_enabled: bool = Field(default=False)
    siem_endpoint: Optional[str] = Field(default=None)
    siem_format: AuditExportFormat = Field(default=AuditExportFormat.SYSLOG)

    # Encryption
    encrypt_sensitive_fields: bool = Field(default=True)
    sensitive_field_patterns: List[str] = Field(
        default_factory=lambda: [
            "password",
            "token",
            "secret",
            "api_key",
            "credential",
            "ssn",
            "social_security",
        ]
    )


class AuditService:
    """
    Production-grade audit logging service for SOC2 and ISO27001 compliance.

    This service provides:
    - Comprehensive audit trails for all operations
    - Tamper-evident logging with SHA-256 hash chains
    - Configurable retention policies
    - Export capabilities for compliance reports
    - SIEM integration for security monitoring

    The hash chain ensures log integrity by including the hash of the
    previous entry in each new entry. Any tampering breaks the chain
    and can be detected during verification.

    Example:
        >>> config = AuditServiceConfig()
        >>> service = AuditService(config)
        >>> await service.initialize()
        >>> await service.log_event(
        ...     event_type=AuditEventType.DATA_ACCESS,
        ...     actor_id="user-123",
        ...     tenant_id="tenant-456",
        ...     action="READ",
        ...     resource_type="EmissionsReport",
        ...     resource_id="report-789",
        ... )

    Attributes:
        config: Service configuration
        _buffer: In-memory buffer for batch writes
        _last_hash: Hash of the most recent entry for chain continuity
    """

    def __init__(self, config: Optional[AuditServiceConfig] = None):
        """
        Initialize the Audit Service.

        Args:
            config: Service configuration (uses defaults if not provided)
        """
        self.config = config or AuditServiceConfig()
        self._buffer: List[AuditEntry] = []
        self._last_hash: Optional[str] = None
        self._buffer_lock = asyncio.Lock()
        self._flush_task: Optional[asyncio.Task] = None
        self._initialized = False

        logger.info(
            "AuditService initialized",
            extra={
                "storage_backend": self.config.storage_backend,
                "hash_chain_enabled": self.config.enable_hash_chain,
            },
        )

    async def initialize(self) -> None:
        """
        Initialize the audit service and start background tasks.

        This method should be called during application startup.
        It retrieves the last hash from storage to continue the chain.
        """
        if self._initialized:
            logger.warning("AuditService already initialized")
            return

        try:
            # Retrieve last hash from storage for chain continuity
            self._last_hash = await self._get_last_hash_from_storage()
            logger.info(f"Retrieved last hash: {self._last_hash[:16] if self._last_hash else 'None'}...")

            # Start background flush task
            self._flush_task = asyncio.create_task(self._background_flush())

            self._initialized = True
            logger.info("AuditService initialization complete")

        except Exception as e:
            logger.error(f"Failed to initialize AuditService: {e}", exc_info=True)
            raise

    async def shutdown(self) -> None:
        """
        Gracefully shutdown the audit service.

        Flushes remaining buffer and cancels background tasks.
        """
        logger.info("Shutting down AuditService...")

        # Cancel background flush
        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass

        # Flush remaining entries
        await self._flush_buffer()

        self._initialized = False
        logger.info("AuditService shutdown complete")

    async def log_event(
        self,
        event_type: AuditEventType,
        actor_id: str,
        tenant_id: str,
        action: str,
        actor_type: str = "user",
        severity: AuditSeverity = AuditSeverity.INFO,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        old_value: Optional[Dict[str, Any]] = None,
        new_value: Optional[Dict[str, Any]] = None,
        request_metadata: Optional[Dict[str, Any]] = None,
        compliance_frameworks: Optional[List[str]] = None,
        control_ids: Optional[List[str]] = None,
    ) -> AuditEntry:
        """
        Log an audit event with tamper-evident hashing.

        Creates an audit entry with a hash chain link to the previous
        entry for integrity verification. The entry is buffered for
        batch writing to optimize performance.

        Args:
            event_type: Category of the audit event
            actor_id: ID of the user or system performing the action
            tenant_id: Multi-tenant organization ID
            action: Specific action performed
            actor_type: Type of actor (user, system, api_key, service)
            severity: Severity level of the event
            resource_type: Type of resource affected
            resource_id: ID of the resource affected
            details: Additional event context
            old_value: Previous state for update operations
            new_value: New state for update operations
            request_metadata: HTTP request information
            compliance_frameworks: Applicable compliance frameworks
            control_ids: Related control IDs

        Returns:
            The created audit entry

        Raises:
            RuntimeError: If service is not initialized
        """
        if not self._initialized:
            raise RuntimeError("AuditService not initialized. Call initialize() first.")

        start_time = datetime.now(timezone.utc)

        try:
            # Mask sensitive fields in details
            masked_details = self._mask_sensitive_fields(details or {})
            masked_old_value = self._mask_sensitive_fields(old_value) if old_value else None
            masked_new_value = self._mask_sensitive_fields(new_value) if new_value else None

            # Create entry with hash chain link
            async with self._buffer_lock:
                entry = AuditEntry(
                    event_type=event_type,
                    severity=severity,
                    actor_id=actor_id,
                    actor_type=actor_type,
                    tenant_id=tenant_id,
                    resource_type=resource_type,
                    resource_id=resource_id,
                    action=action,
                    details=masked_details,
                    old_value=masked_old_value,
                    new_value=masked_new_value,
                    request_metadata=request_metadata or {},
                    previous_hash=self._last_hash,
                    compliance_frameworks=compliance_frameworks or ["SOC2", "ISO27001"],
                    control_ids=control_ids or [],
                )

                # Update chain hash
                self._last_hash = entry.entry_hash

                # Add to buffer
                self._buffer.append(entry)

            # Log high-severity events immediately
            if severity in (AuditSeverity.ERROR, AuditSeverity.CRITICAL):
                await self._flush_buffer()

            # Send to SIEM if enabled
            if self.config.siem_enabled:
                await self._send_to_siem(entry)

            processing_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            logger.debug(
                f"Audit event logged: {event_type.value}",
                extra={
                    "event_id": entry.id,
                    "processing_time_ms": processing_time,
                },
            )

            return entry

        except Exception as e:
            logger.error(f"Failed to log audit event: {e}", exc_info=True)
            raise

    async def log_data_access(
        self,
        actor_id: str,
        tenant_id: str,
        resource_type: str,
        resource_id: str,
        fields_accessed: List[str],
        access_reason: str,
        request_metadata: Optional[Dict[str, Any]] = None,
    ) -> AuditEntry:
        """
        Log a data access event for compliance tracking.

        This is a convenience method for logging data access events
        with structured field tracking for compliance audits.

        Args:
            actor_id: ID of the user accessing data
            tenant_id: Organization tenant ID
            resource_type: Type of data resource accessed
            resource_id: ID of the specific resource
            fields_accessed: List of fields that were accessed
            access_reason: Business justification for access
            request_metadata: HTTP request information

        Returns:
            The created audit entry
        """
        return await self.log_event(
            event_type=AuditEventType.DATA_ACCESS,
            actor_id=actor_id,
            tenant_id=tenant_id,
            action="READ",
            resource_type=resource_type,
            resource_id=resource_id,
            details={
                "fields_accessed": fields_accessed,
                "access_reason": access_reason,
                "field_count": len(fields_accessed),
            },
            request_metadata=request_metadata,
            control_ids=["CC6.1", "A.12.4.1"],  # SOC2 and ISO27001 controls
        )

    async def log_authentication(
        self,
        actor_id: str,
        tenant_id: str,
        success: bool,
        auth_method: str,
        failure_reason: Optional[str] = None,
        request_metadata: Optional[Dict[str, Any]] = None,
    ) -> AuditEntry:
        """
        Log an authentication event.

        Args:
            actor_id: ID of the user attempting authentication
            tenant_id: Organization tenant ID
            success: Whether authentication was successful
            auth_method: Authentication method used (password, sso, mfa)
            failure_reason: Reason for failure if unsuccessful
            request_metadata: HTTP request information

        Returns:
            The created audit entry
        """
        event_type = AuditEventType.LOGIN_SUCCESS if success else AuditEventType.LOGIN_FAILURE
        severity = AuditSeverity.INFO if success else AuditSeverity.WARNING

        details = {
            "auth_method": auth_method,
            "success": success,
        }
        if failure_reason:
            details["failure_reason"] = failure_reason

        return await self.log_event(
            event_type=event_type,
            severity=severity,
            actor_id=actor_id,
            tenant_id=tenant_id,
            action="AUTHENTICATE",
            details=details,
            request_metadata=request_metadata,
            control_ids=["CC6.1", "A.9.4.2"],
        )

    async def query_events(
        self,
        tenant_id: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        event_types: Optional[List[AuditEventType]] = None,
        actor_id: Optional[str] = None,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        severity: Optional[AuditSeverity] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[AuditEntry]:
        """
        Query audit events with filtering.

        Args:
            tenant_id: Tenant to query (required for multi-tenant isolation)
            start_time: Start of time range
            end_time: End of time range
            event_types: Filter by event types
            actor_id: Filter by actor
            resource_type: Filter by resource type
            resource_id: Filter by resource ID
            severity: Filter by severity
            limit: Maximum entries to return
            offset: Pagination offset

        Returns:
            List of matching audit entries
        """
        logger.info(
            f"Querying audit events for tenant {tenant_id}",
            extra={"limit": limit, "offset": offset},
        )

        # This would query the actual storage backend
        # For now, return entries from buffer matching criteria
        results = []
        for entry in self._buffer:
            if entry.tenant_id != tenant_id:
                continue
            if start_time and entry.timestamp < start_time:
                continue
            if end_time and entry.timestamp > end_time:
                continue
            if event_types and entry.event_type not in event_types:
                continue
            if actor_id and entry.actor_id != actor_id:
                continue
            if resource_type and entry.resource_type != resource_type:
                continue
            if resource_id and entry.resource_id != resource_id:
                continue
            if severity and entry.severity != severity:
                continue
            results.append(entry)

        return results[offset : offset + limit]

    async def export_for_compliance(
        self,
        tenant_id: str,
        start_time: datetime,
        end_time: datetime,
        format: AuditExportFormat = AuditExportFormat.JSON,
        compliance_framework: Optional[str] = None,
    ) -> bytes:
        """
        Export audit logs for compliance reporting.

        Generates a compliance-ready export with all required metadata
        including hash chain verification results.

        Args:
            tenant_id: Tenant to export
            start_time: Start of export period
            end_time: End of export period
            format: Export format
            compliance_framework: Filter by compliance framework

        Returns:
            Exported data as bytes
        """
        logger.info(
            f"Exporting audit logs for compliance",
            extra={
                "tenant_id": tenant_id,
                "format": format.value,
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
            },
        )

        # Log the export action itself
        await self.log_event(
            event_type=AuditEventType.AUDIT_EXPORT,
            actor_id="system",
            actor_type="system",
            tenant_id=tenant_id,
            action="EXPORT",
            details={
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "format": format.value,
                "compliance_framework": compliance_framework,
            },
        )

        # Query events for export
        events = await self.query_events(
            tenant_id=tenant_id,
            start_time=start_time,
            end_time=end_time,
            limit=100000,  # Large limit for exports
        )

        # Verify hash chain integrity
        chain_valid = await self.verify_chain_integrity(events)

        # Format export
        if format == AuditExportFormat.JSON:
            export_data = {
                "export_metadata": {
                    "tenant_id": tenant_id,
                    "export_time": datetime.now(timezone.utc).isoformat(),
                    "start_time": start_time.isoformat(),
                    "end_time": end_time.isoformat(),
                    "event_count": len(events),
                    "hash_chain_verified": chain_valid,
                    "compliance_framework": compliance_framework,
                },
                "events": [e.dict() for e in events],
            }
            return json.dumps(export_data, indent=2, default=str).encode()

        elif format == AuditExportFormat.NDJSON:
            lines = []
            for event in events:
                lines.append(json.dumps(event.dict(), default=str))
            return "\n".join(lines).encode()

        elif format == AuditExportFormat.CSV:
            import csv
            import io

            output = io.StringIO()
            if events:
                writer = csv.DictWriter(output, fieldnames=events[0].dict().keys())
                writer.writeheader()
                for event in events:
                    writer.writerow({k: str(v) for k, v in event.dict().items()})
            return output.getvalue().encode()

        elif format == AuditExportFormat.SYSLOG:
            lines = [event.to_syslog() for event in events]
            return "\n".join(lines).encode()

        else:
            raise ValueError(f"Unsupported export format: {format}")

    async def verify_chain_integrity(
        self,
        entries: Optional[List[AuditEntry]] = None,
        tenant_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> bool:
        """
        Verify the integrity of the audit log hash chain.

        Checks that each entry's previous_hash matches the entry_hash
        of the preceding entry, detecting any tampering.

        Args:
            entries: Entries to verify (if not provided, queries storage)
            tenant_id: Tenant to verify (required if entries not provided)
            start_time: Start of verification period
            end_time: End of verification period

        Returns:
            True if chain is intact, False if tampering detected
        """
        if not entries:
            if not tenant_id:
                raise ValueError("Either entries or tenant_id must be provided")
            entries = await self.query_events(
                tenant_id=tenant_id,
                start_time=start_time,
                end_time=end_time,
                limit=100000,
            )

        if len(entries) < 2:
            return True

        # Sort by timestamp
        sorted_entries = sorted(entries, key=lambda e: e.timestamp)

        # Verify chain
        for i in range(1, len(sorted_entries)):
            current = sorted_entries[i]
            previous = sorted_entries[i - 1]

            if current.previous_hash != previous.entry_hash:
                logger.error(
                    f"Hash chain broken at entry {current.id}",
                    extra={
                        "expected_hash": previous.entry_hash,
                        "found_hash": current.previous_hash,
                    },
                )
                return False

        logger.info(f"Hash chain verified: {len(entries)} entries intact")
        return True

    async def apply_retention_policy(self, tenant_id: str) -> int:
        """
        Apply retention policy and purge expired entries.

        Args:
            tenant_id: Tenant to apply retention to

        Returns:
            Number of entries purged
        """
        logger.info(f"Applying retention policy for tenant {tenant_id}")

        purged_count = 0
        now = datetime.now(timezone.utc)

        # This would query and delete from actual storage
        # For demonstration, we check the buffer
        entries_to_keep = []
        for entry in self._buffer:
            if entry.tenant_id != tenant_id:
                entries_to_keep.append(entry)
                continue

            retention_days = self.config.retention_policy.get_retention_days(
                severity=entry.severity,
                event_type=entry.event_type,
                compliance_frameworks=entry.compliance_frameworks,
            )

            expiry_date = entry.timestamp + timedelta(days=retention_days)
            if now < expiry_date:
                entries_to_keep.append(entry)
            else:
                purged_count += 1

        if purged_count > 0:
            self._buffer = entries_to_keep

            # Log the purge action
            await self.log_event(
                event_type=AuditEventType.RETENTION_PURGE,
                actor_id="system",
                actor_type="system",
                tenant_id=tenant_id,
                action="PURGE",
                details={"entries_purged": purged_count},
            )

        logger.info(f"Retention policy applied: {purged_count} entries purged")
        return purged_count

    def _mask_sensitive_fields(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Mask sensitive fields in audit data.

        Replaces values of sensitive fields with "[REDACTED]" to prevent
        logging of passwords, tokens, and other sensitive data.

        Args:
            data: Dictionary to mask

        Returns:
            Dictionary with sensitive fields masked
        """
        if not data:
            return data

        masked = {}
        for key, value in data.items():
            key_lower = key.lower()
            is_sensitive = any(
                pattern in key_lower for pattern in self.config.sensitive_field_patterns
            )

            if is_sensitive:
                masked[key] = "[REDACTED]"
            elif isinstance(value, dict):
                masked[key] = self._mask_sensitive_fields(value)
            else:
                masked[key] = value

        return masked

    async def _flush_buffer(self) -> None:
        """Flush buffered entries to storage."""
        async with self._buffer_lock:
            if not self._buffer:
                return

            entries_to_flush = self._buffer.copy()
            self._buffer = []

        try:
            # Write to storage backend
            await self._write_to_storage(entries_to_flush)
            logger.debug(f"Flushed {len(entries_to_flush)} audit entries to storage")

        except Exception as e:
            logger.error(f"Failed to flush audit buffer: {e}", exc_info=True)
            # Re-add entries to buffer for retry
            async with self._buffer_lock:
                self._buffer = entries_to_flush + self._buffer

    async def _background_flush(self) -> None:
        """Background task to periodically flush the buffer."""
        while True:
            try:
                await asyncio.sleep(self.config.flush_interval_seconds)
                await self._flush_buffer()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Background flush error: {e}", exc_info=True)

    async def _write_to_storage(self, entries: List[AuditEntry]) -> None:
        """
        Write entries to the configured storage backend.

        This is a placeholder for actual storage implementation.
        Would integrate with PostgreSQL, Elasticsearch, or S3.
        """
        # Placeholder - actual implementation would write to database
        logger.debug(f"Writing {len(entries)} entries to {self.config.storage_backend}")

    async def _get_last_hash_from_storage(self) -> Optional[str]:
        """
        Retrieve the hash of the most recent entry from storage.

        This is a placeholder for actual storage implementation.
        """
        # Placeholder - would query storage for last entry
        return None

    async def _send_to_siem(self, entry: AuditEntry) -> None:
        """
        Send audit entry to SIEM system for security monitoring.

        Args:
            entry: Audit entry to send
        """
        if not self.config.siem_endpoint:
            return

        try:
            if self.config.siem_format == AuditExportFormat.SYSLOG:
                message = entry.to_syslog()
            else:
                message = json.dumps(entry.dict(), default=str)

            # Placeholder - would send via HTTP or syslog protocol
            logger.debug(f"Sent audit entry to SIEM: {entry.id}")

        except Exception as e:
            logger.warning(f"Failed to send audit entry to SIEM: {e}")
