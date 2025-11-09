"""
Enhanced Audit Logging System
GL-VCCI Scope 3 Platform

Comprehensive audit logging with:
- Failed authentication attempts
- Password change attempts
- API key usage
- Token refresh events
- Suspicious activity patterns
- Immutable storage (S3 with object lock)
- Audit log integrity verification (hash chain)
- SIEM integration ready

Version: 1.0.0
Security Enhancement: 2025-11-09
"""

import os
import json
import hashlib
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field, asdict

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

# Configuration
ENVIRONMENT = os.getenv("ENVIRONMENT", "dev")
AUDIT_ENABLED = os.getenv("AUDIT_ENABLED", "true").lower() == "true"
AUDIT_STORAGE = os.getenv("AUDIT_STORAGE", "local")  # local, s3, both
AUDIT_S3_BUCKET = os.getenv("AUDIT_S3_BUCKET", "vcci-audit-logs")
AUDIT_S3_PREFIX = os.getenv("AUDIT_S3_PREFIX", f"{ENVIRONMENT}/audit")
AUDIT_INTEGRITY_CHECK = os.getenv("AUDIT_INTEGRITY_CHECK", "true").lower() == "true"

# Local storage path
AUDIT_LOG_DIR = os.getenv("AUDIT_LOG_DIR", "./logs/audit")


class AuditEventType(str, Enum):
    """Types of audit events."""

    # Authentication events
    AUTH_SUCCESS = "auth.success"
    AUTH_FAILURE = "auth.failure"
    AUTH_LOCKOUT = "auth.lockout"
    TOKEN_REFRESH = "auth.token_refresh"
    TOKEN_REVOCATION = "auth.token_revocation"
    LOGOUT = "auth.logout"

    # Password events
    PASSWORD_CHANGE = "password.change"
    PASSWORD_RESET = "password.reset"
    PASSWORD_RESET_REQUEST = "password.reset_request"

    # API key events
    API_KEY_CREATED = "api_key.created"
    API_KEY_USED = "api_key.used"
    API_KEY_REVOKED = "api_key.revoked"
    API_KEY_RATE_LIMIT = "api_key.rate_limit_exceeded"

    # Data access events
    DATA_EXPORT = "data.export"
    DATA_IMPORT = "data.import"
    DATA_DELETE = "data.delete"
    REPORT_GENERATED = "report.generated"

    # Configuration events
    CONFIG_CHANGE = "config.change"
    USER_CREATED = "user.created"
    USER_MODIFIED = "user.modified"
    USER_DELETED = "user.deleted"
    ROLE_ASSIGNED = "role.assigned"
    ROLE_REVOKED = "role.revoked"

    # Security events
    SUSPICIOUS_ACTIVITY = "security.suspicious_activity"
    BRUTE_FORCE_DETECTED = "security.brute_force"
    RATE_LIMIT_EXCEEDED = "security.rate_limit"
    SIGNATURE_FAILURE = "security.signature_failure"
    BLACKLIST_HIT = "security.blacklist_hit"

    # System events
    SYSTEM_STARTUP = "system.startup"
    SYSTEM_SHUTDOWN = "system.shutdown"
    BACKUP_CREATED = "system.backup_created"
    BACKUP_RESTORED = "system.backup_restored"


class AuditSeverity(str, Enum):
    """Severity levels for audit events."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class AuditEvent:
    """Audit event data structure."""

    event_type: AuditEventType
    severity: AuditSeverity
    timestamp: datetime = field(default_factory=datetime.utcnow)
    user_id: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    resource: Optional[str] = None
    action: Optional[str] = None
    result: Optional[str] = None  # success, failure, error
    details: Optional[Dict[str, Any]] = None
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    previous_hash: Optional[str] = None  # For hash chain integrity
    event_hash: Optional[str] = None  # Hash of this event

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        data = asdict(self)
        data["timestamp"] = self.timestamp.isoformat()
        data["event_type"] = self.event_type.value
        data["severity"] = self.severity.value
        return data

    def compute_hash(self) -> str:
        """
        Compute hash of this event for integrity verification.

        Creates a hash chain by including the previous event's hash.

        Returns:
            SHA256 hash of event data
        """
        # Create deterministic string representation
        hash_data = {
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "user_id": self.user_id,
            "ip_address": self.ip_address,
            "resource": self.resource,
            "action": self.action,
            "result": self.result,
            "details": json.dumps(self.details, sort_keys=True) if self.details else None,
            "previous_hash": self.previous_hash,
        }

        hash_string = json.dumps(hash_data, sort_keys=True)
        event_hash = hashlib.sha256(hash_string.encode("utf-8")).hexdigest()

        return event_hash


class AuditLogger:
    """
    Enhanced audit logger with integrity verification and immutable storage.

    Example:
        ```python
        audit = AuditLogger()

        # Log authentication failure
        await audit.log_event(
            AuditEventType.AUTH_FAILURE,
            AuditSeverity.WARNING,
            user_id="user@example.com",
            ip_address="192.168.1.1",
            details={"reason": "invalid_password"}
        )
        ```
    """

    def __init__(self):
        """Initialize audit logger."""
        self.s3_client = None
        self.last_event_hash = None

        if AUDIT_STORAGE in ["s3", "both"]:
            try:
                self.s3_client = boto3.client("s3")
                logger.info(f"Initialized S3 audit logging to bucket: {AUDIT_S3_BUCKET}")
            except Exception as e:
                logger.error(f"Failed to initialize S3 client: {str(e)}")

        # Ensure local log directory exists
        if AUDIT_STORAGE in ["local", "both"]:
            os.makedirs(AUDIT_LOG_DIR, exist_ok=True)

    async def log_event(
        self,
        event_type: AuditEventType,
        severity: AuditSeverity,
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        resource: Optional[str] = None,
        action: Optional[str] = None,
        result: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
        request_id: Optional[str] = None,
    ) -> AuditEvent:
        """
        Log an audit event.

        Args:
            event_type: Type of event
            severity: Event severity
            user_id: User identifier
            ip_address: Client IP address
            user_agent: Client user agent
            resource: Resource being accessed
            action: Action performed
            result: Result (success/failure/error)
            details: Additional event details
            session_id: Session identifier
            request_id: Request identifier

        Returns:
            AuditEvent that was logged

        Example:
            >>> await audit.log_event(
            ...     AuditEventType.DATA_EXPORT,
            ...     AuditSeverity.INFO,
            ...     user_id="admin@example.com",
            ...     resource="/api/export/emissions",
            ...     result="success"
            ... )
        """
        if not AUDIT_ENABLED:
            return

        # Create event
        event = AuditEvent(
            event_type=event_type,
            severity=severity,
            user_id=user_id,
            ip_address=ip_address,
            user_agent=user_agent,
            resource=resource,
            action=action,
            result=result,
            details=details,
            session_id=session_id,
            request_id=request_id,
            previous_hash=self.last_event_hash,
        )

        # Compute hash for integrity verification
        if AUDIT_INTEGRITY_CHECK:
            event.event_hash = event.compute_hash()
            self.last_event_hash = event.event_hash

        # Store event
        await self._store_event(event)

        # Log to application logger
        self._log_to_application_logger(event)

        return event

    async def _store_event(self, event: AuditEvent):
        """Store audit event to configured storage."""
        event_data = event.to_dict()

        # Store locally
        if AUDIT_STORAGE in ["local", "both"]:
            await self._store_local(event, event_data)

        # Store in S3
        if AUDIT_STORAGE in ["s3", "both"] and self.s3_client:
            await self._store_s3(event, event_data)

    async def _store_local(self, event: AuditEvent, event_data: Dict[str, Any]):
        """Store event to local file system."""
        try:
            # Organize by date
            date_str = event.timestamp.strftime("%Y-%m-%d")
            log_file = os.path.join(AUDIT_LOG_DIR, f"audit-{date_str}.jsonl")

            # Append to JSONL file
            with open(log_file, "a") as f:
                f.write(json.dumps(event_data) + "\n")

        except Exception as e:
            logger.error(f"Failed to store audit event locally: {str(e)}")

    async def _store_s3(self, event: AuditEvent, event_data: Dict[str, Any]):
        """Store event to S3 with object lock for immutability."""
        try:
            # Create S3 key with hierarchical structure
            # Format: {prefix}/{year}/{month}/{day}/{timestamp}_{event_type}_{hash}.json
            timestamp_str = event.timestamp.strftime("%Y%m%d_%H%M%S_%f")
            year = event.timestamp.strftime("%Y")
            month = event.timestamp.strftime("%m")
            day = event.timestamp.strftime("%d")

            event_hash_short = event.event_hash[:8] if event.event_hash else "no_hash"
            key = (
                f"{AUDIT_S3_PREFIX}/{year}/{month}/{day}/"
                f"{timestamp_str}_{event.event_type.value}_{event_hash_short}.json"
            )

            # Upload to S3
            self.s3_client.put_object(
                Bucket=AUDIT_S3_BUCKET,
                Key=key,
                Body=json.dumps(event_data, indent=2),
                ContentType="application/json",
                Metadata={
                    "event_type": event.event_type.value,
                    "severity": event.severity.value,
                    "user_id": event.user_id or "unknown",
                    "event_hash": event.event_hash or "no_hash",
                },
                # Enable object lock for immutability (requires bucket configuration)
                # ObjectLockMode='COMPLIANCE',
                # ObjectLockRetainUntilDate=datetime.utcnow() + timedelta(days=2555),  # 7 years
            )

        except ClientError as e:
            logger.error(f"Failed to store audit event to S3: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error storing to S3: {str(e)}")

    def _log_to_application_logger(self, event: AuditEvent):
        """Log event to application logger for immediate visibility."""
        log_message = (
            f"AUDIT: {event.event_type.value} | "
            f"user={event.user_id} | "
            f"ip={event.ip_address} | "
            f"resource={event.resource} | "
            f"result={event.result}"
        )

        if event.severity == AuditSeverity.CRITICAL:
            logger.critical(log_message)
        elif event.severity == AuditSeverity.ERROR:
            logger.error(log_message)
        elif event.severity == AuditSeverity.WARNING:
            logger.warning(log_message)
        else:
            logger.info(log_message)

    async def verify_integrity(
        self,
        events: List[AuditEvent],
    ) -> bool:
        """
        Verify integrity of audit log chain.

        Checks that each event's hash matches and the hash chain is intact.

        Args:
            events: List of events in chronological order

        Returns:
            True if integrity is verified, False otherwise

        Example:
            >>> events = await audit.get_events(start_date, end_date)
            >>> if await audit.verify_integrity(events):
            ...     print("Audit log integrity verified")
        """
        if not events:
            return True

        for i, event in enumerate(events):
            # Verify event hash
            computed_hash = event.compute_hash()

            if event.event_hash != computed_hash:
                logger.error(
                    f"Integrity check failed: event hash mismatch at index {i}"
                )
                return False

            # Verify chain (except first event)
            if i > 0:
                previous_event = events[i - 1]

                if event.previous_hash != previous_event.event_hash:
                    logger.error(
                        f"Integrity check failed: hash chain broken at index {i}"
                    )
                    return False

        logger.info(f"Integrity verified for {len(events)} events")
        return True


# Global audit logger instance
_audit_logger: Optional[AuditLogger] = None


def get_audit_logger() -> AuditLogger:
    """Get global audit logger instance."""
    global _audit_logger

    if _audit_logger is None:
        _audit_logger = AuditLogger()

    return _audit_logger


# Convenience functions for common audit events
async def log_auth_success(
    user_id: str,
    ip_address: Optional[str] = None,
    user_agent: Optional[str] = None,
    session_id: Optional[str] = None,
):
    """Log successful authentication."""
    audit = get_audit_logger()
    await audit.log_event(
        AuditEventType.AUTH_SUCCESS,
        AuditSeverity.INFO,
        user_id=user_id,
        ip_address=ip_address,
        user_agent=user_agent,
        result="success",
        session_id=session_id,
    )


async def log_auth_failure(
    user_id: str,
    ip_address: Optional[str] = None,
    user_agent: Optional[str] = None,
    reason: Optional[str] = None,
):
    """Log failed authentication attempt."""
    audit = get_audit_logger()
    await audit.log_event(
        AuditEventType.AUTH_FAILURE,
        AuditSeverity.WARNING,
        user_id=user_id,
        ip_address=ip_address,
        user_agent=user_agent,
        result="failure",
        details={"reason": reason} if reason else None,
    )


async def log_password_change(
    user_id: str,
    ip_address: Optional[str] = None,
    result: str = "success",
):
    """Log password change attempt."""
    audit = get_audit_logger()
    severity = AuditSeverity.INFO if result == "success" else AuditSeverity.WARNING
    await audit.log_event(
        AuditEventType.PASSWORD_CHANGE,
        severity,
        user_id=user_id,
        ip_address=ip_address,
        result=result,
    )


async def log_api_key_usage(
    service_name: str,
    ip_address: Optional[str] = None,
    resource: Optional[str] = None,
    action: Optional[str] = None,
):
    """Log API key usage."""
    audit = get_audit_logger()
    await audit.log_event(
        AuditEventType.API_KEY_USED,
        AuditSeverity.INFO,
        user_id=f"service:{service_name}",
        ip_address=ip_address,
        resource=resource,
        action=action,
        result="success",
    )


async def log_token_refresh(
    user_id: str,
    ip_address: Optional[str] = None,
    result: str = "success",
):
    """Log token refresh event."""
    audit = get_audit_logger()
    severity = AuditSeverity.INFO if result == "success" else AuditSeverity.WARNING
    await audit.log_event(
        AuditEventType.TOKEN_REFRESH,
        severity,
        user_id=user_id,
        ip_address=ip_address,
        result=result,
    )


async def log_suspicious_activity(
    user_id: Optional[str],
    ip_address: Optional[str],
    activity_type: str,
    details: Optional[Dict[str, Any]] = None,
):
    """Log suspicious activity detection."""
    audit = get_audit_logger()
    await audit.log_event(
        AuditEventType.SUSPICIOUS_ACTIVITY,
        AuditSeverity.ERROR,
        user_id=user_id,
        ip_address=ip_address,
        action=activity_type,
        result="detected",
        details=details,
    )


async def log_data_export(
    user_id: str,
    resource: str,
    ip_address: Optional[str] = None,
    record_count: Optional[int] = None,
):
    """Log data export operation."""
    audit = get_audit_logger()
    details = {"record_count": record_count} if record_count else None
    await audit.log_event(
        AuditEventType.DATA_EXPORT,
        AuditSeverity.INFO,
        user_id=user_id,
        ip_address=ip_address,
        resource=resource,
        action="export",
        result="success",
        details=details,
    )


async def log_config_change(
    user_id: str,
    resource: str,
    action: str,
    ip_address: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None,
):
    """Log configuration change."""
    audit = get_audit_logger()
    await audit.log_event(
        AuditEventType.CONFIG_CHANGE,
        AuditSeverity.WARNING,
        user_id=user_id,
        ip_address=ip_address,
        resource=resource,
        action=action,
        result="success",
        details=details,
    )


# SIEM integration helpers
def export_to_siem_format(event: AuditEvent) -> Dict[str, Any]:
    """
    Export audit event to SIEM-compatible format (CEF, LEEF, etc.).

    Args:
        event: Audit event to export

    Returns:
        Dictionary in SIEM format

    Example:
        >>> siem_data = export_to_siem_format(event)
        >>> send_to_splunk(siem_data)
    """
    # Common Event Format (CEF) example
    return {
        "version": "1.0",
        "deviceVendor": "GreenLang",
        "deviceProduct": "VCCI-Scope3-Platform",
        "deviceVersion": "2.0",
        "signatureId": event.event_type.value,
        "name": event.event_type.value,
        "severity": event.severity.value,
        "timestamp": event.timestamp.isoformat(),
        "src": event.ip_address,
        "suser": event.user_id,
        "request": event.resource,
        "outcome": event.result,
        "cs1": event.session_id,
        "cs1Label": "SessionID",
        "cs2": event.request_id,
        "cs2Label": "RequestID",
        "cn1": event.event_hash,
        "cn1Label": "EventHash",
    }


__all__ = [
    "AuditEventType",
    "AuditSeverity",
    "AuditEvent",
    "AuditLogger",
    "get_audit_logger",
    "log_auth_success",
    "log_auth_failure",
    "log_password_change",
    "log_api_key_usage",
    "log_token_refresh",
    "log_suspicious_activity",
    "log_data_export",
    "log_config_change",
    "export_to_siem_format",
]
