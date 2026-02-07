# -*- coding: utf-8 -*-
"""
Auth Audit Logger - JWT Authentication Service (SEC-001)

Emits structured authentication events as JSON log records for Loki
ingestion.  Every auth event (login, token lifecycle, lockout, permission
denial) is captured with a consistent schema that includes correlation IDs,
tenant isolation, and client metadata.

**Security invariant:** This module NEVER logs sensitive data including
passwords, token values, MFA secrets, or bearer credentials.  Only opaque
identifiers (JTI, user ID, tenant ID) are included.

Classes:
    - AuthEventType: Enumeration of all auditable auth events.
    - AuthEvent: Structured auth event data class.
    - AuthAuditLogger: Stateless logger that emits JSON-structured events.

Example:
    >>> audit = AuthAuditLogger()
    >>> audit.log_login(
    ...     user_id="u-abc",
    ...     tenant_id="t-corp",
    ...     success=True,
    ...     ip_address="10.0.0.1",
    ... )

Author: GreenLang Framework Team
Date: February 2026
Status: Production Ready
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Event types
# ---------------------------------------------------------------------------


class AuthEventType(str, Enum):
    """Enumeration of all auditable authentication events.

    Values are lowercase snake_case strings suitable for use as Loki labels.
    """

    LOGIN_SUCCESS = "login_success"
    LOGIN_FAILURE = "login_failure"
    TOKEN_ISSUED = "token_issued"
    TOKEN_VALIDATED = "token_validated"
    TOKEN_REVOKED = "token_revoked"
    TOKEN_REFRESHED = "token_refreshed"
    LOGOUT = "logout"
    PASSWORD_CHANGED = "password_changed"
    PASSWORD_RESET_REQUESTED = "password_reset_requested"
    MFA_SETUP = "mfa_setup"
    MFA_VERIFIED = "mfa_verified"
    MFA_FAILED = "mfa_failed"
    ACCOUNT_LOCKED = "account_locked"
    ACCOUNT_UNLOCKED = "account_unlocked"
    PERMISSION_DENIED = "permission_denied"
    SESSION_EXPIRED = "session_expired"
    RATE_LIMITED = "rate_limited"


# ---------------------------------------------------------------------------
# Event data class
# ---------------------------------------------------------------------------

# Fields that must NEVER appear in log output
_REDACTED_FIELDS = frozenset({
    "password",
    "new_password",
    "old_password",
    "token",
    "access_token",
    "refresh_token",
    "mfa_secret",
    "totp_code",
    "bearer",
    "authorization",
    "secret",
    "private_key",
})


@dataclass
class AuthEvent:
    """Structured representation of an authentication event.

    Attributes:
        event_type: The type of auth event.
        user_id: UUID of the user (``None`` for pre-auth events).
        tenant_id: UUID of the tenant.
        ip_address: Client IP address.
        user_agent: Client User-Agent header.
        correlation_id: Request correlation / trace ID.
        details: Additional event-specific key-value pairs.
        result: Outcome string (``success``, ``failure``, ``denied``).
        timestamp: UTC datetime of the event.
    """

    event_type: AuthEventType
    user_id: Optional[str] = None
    tenant_id: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    correlation_id: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    result: str = "success"
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the event to a log-safe dictionary.

        Strips any keys in ``details`` that match the redaction set.

        Returns:
            Dictionary safe for JSON serialization and log emission.
        """
        safe_details = {
            k: v
            for k, v in self.details.items()
            if k.lower() not in _REDACTED_FIELDS
        }

        return {
            "event_type": self.event_type.value,
            "user_id": self.user_id,
            "tenant_id": self.tenant_id,
            "ip_address": self.ip_address,
            "user_agent": self.user_agent,
            "correlation_id": self.correlation_id,
            "result": self.result,
            "details": safe_details,
            "timestamp": self.timestamp.isoformat(),
        }


# ---------------------------------------------------------------------------
# Audit logger
# ---------------------------------------------------------------------------


class AuthAuditLogger:
    """Emits structured auth audit events as JSON log records.

    Uses a dedicated ``greenlang.auth.audit`` logger so that Loki pipeline
    stages can match on the logger name and apply appropriate labels.

    The logger is stateless and safe for concurrent use.

    Example:
        >>> audit = AuthAuditLogger()
        >>> audit.log_login("u-1", "t-1", success=True, ip_address="10.0.0.1")
    """

    def __init__(self) -> None:
        """Initialize the audit logger."""
        self._logger = logging.getLogger("greenlang.auth.audit")

    # ------------------------------------------------------------------
    # Core log method
    # ------------------------------------------------------------------

    def log_event(self, event: AuthEvent) -> None:
        """Log an auth event as a structured JSON record.

        The log level is determined by the event result:
        - ``success`` -> INFO
        - ``failure`` -> WARNING
        - ``denied``  -> WARNING

        Args:
            event: The auth event to log.
        """
        payload = event.to_dict()

        level = logging.INFO
        if event.result in ("failure", "denied"):
            level = logging.WARNING

        # Emit as a JSON string with extra fields for Loki labels
        self._logger.log(
            level,
            json.dumps(payload, default=str),
            extra={
                "event_type": event.event_type.value,
                "auth_result": event.result,
                "tenant_id": event.tenant_id or "",
                "user_id": event.user_id or "",
            },
        )

    # ------------------------------------------------------------------
    # Convenience: login events
    # ------------------------------------------------------------------

    def log_login(
        self,
        user_id: str,
        tenant_id: str,
        success: bool,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        method: str = "password",
        correlation_id: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Log a login attempt.

        Args:
            user_id: UUID of the user.
            tenant_id: UUID of the tenant.
            success: Whether the login succeeded.
            ip_address: Client IP address.
            user_agent: Client User-Agent header.
            method: Authentication method (``password``, ``sso``, ``api_key``).
            correlation_id: Request correlation ID.
            **kwargs: Additional details (sensitive keys are auto-redacted).
        """
        event_type = (
            AuthEventType.LOGIN_SUCCESS if success else AuthEventType.LOGIN_FAILURE
        )
        details: Dict[str, Any] = {"method": method}
        details.update(kwargs)

        event = AuthEvent(
            event_type=event_type,
            user_id=user_id,
            tenant_id=tenant_id,
            ip_address=ip_address,
            user_agent=user_agent,
            correlation_id=correlation_id,
            details=details,
            result="success" if success else "failure",
        )
        self.log_event(event)

    # ------------------------------------------------------------------
    # Convenience: token lifecycle events
    # ------------------------------------------------------------------

    def log_token_event(
        self,
        event_type: AuthEventType,
        user_id: str,
        tenant_id: Optional[str] = None,
        jti: Optional[str] = None,
        ip_address: Optional[str] = None,
        correlation_id: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Log a token lifecycle event (issued, validated, revoked, refreshed).

        Args:
            event_type: One of the token-related ``AuthEventType`` values.
            user_id: UUID of the user.
            tenant_id: UUID of the tenant.
            jti: JWT ID (opaque identifier -- NOT the token value).
            ip_address: Client IP address.
            correlation_id: Request correlation ID.
            **kwargs: Additional details.
        """
        details: Dict[str, Any] = {}
        if jti:
            details["jti"] = jti
        details.update(kwargs)

        event = AuthEvent(
            event_type=event_type,
            user_id=user_id,
            tenant_id=tenant_id,
            ip_address=ip_address,
            correlation_id=correlation_id,
            details=details,
            result="success",
        )
        self.log_event(event)

    # ------------------------------------------------------------------
    # Convenience: permission denied
    # ------------------------------------------------------------------

    def log_permission_denied(
        self,
        user_id: str,
        permission: str,
        resource: str,
        tenant_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        correlation_id: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Log a permission denial event.

        Args:
            user_id: UUID of the user whose request was denied.
            permission: Permission that was required (e.g. ``reports:write``).
            resource: The resource that was accessed.
            tenant_id: UUID of the tenant.
            ip_address: Client IP address.
            correlation_id: Request correlation ID.
            **kwargs: Additional details.
        """
        details: Dict[str, Any] = {
            "permission": permission,
            "resource": resource,
        }
        details.update(kwargs)

        event = AuthEvent(
            event_type=AuthEventType.PERMISSION_DENIED,
            user_id=user_id,
            tenant_id=tenant_id,
            ip_address=ip_address,
            correlation_id=correlation_id,
            details=details,
            result="denied",
        )
        self.log_event(event)

    # ------------------------------------------------------------------
    # Convenience: account lockout events
    # ------------------------------------------------------------------

    def log_account_locked(
        self,
        user_id: str,
        tenant_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        lockout_duration_s: Optional[int] = None,
        failed_attempts: Optional[int] = None,
        correlation_id: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Log an account lockout event.

        Args:
            user_id: UUID of the locked user.
            tenant_id: UUID of the tenant.
            ip_address: Client IP that triggered the lockout.
            lockout_duration_s: Duration of the lockout in seconds.
            failed_attempts: Number of failed attempts that triggered lockout.
            correlation_id: Request correlation ID.
            **kwargs: Additional details.
        """
        details: Dict[str, Any] = {}
        if lockout_duration_s is not None:
            details["lockout_duration_s"] = lockout_duration_s
        if failed_attempts is not None:
            details["failed_attempts"] = failed_attempts
        details.update(kwargs)

        event = AuthEvent(
            event_type=AuthEventType.ACCOUNT_LOCKED,
            user_id=user_id,
            tenant_id=tenant_id,
            ip_address=ip_address,
            correlation_id=correlation_id,
            details=details,
            result="failure",
        )
        self.log_event(event)

    def log_account_unlocked(
        self,
        user_id: str,
        unlocked_by: str = "admin",
        tenant_id: Optional[str] = None,
        correlation_id: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Log an account unlock event.

        Args:
            user_id: UUID of the unlocked user.
            unlocked_by: Identity of who performed the unlock.
            tenant_id: UUID of the tenant.
            correlation_id: Request correlation ID.
            **kwargs: Additional details.
        """
        details: Dict[str, Any] = {"unlocked_by": unlocked_by}
        details.update(kwargs)

        event = AuthEvent(
            event_type=AuthEventType.ACCOUNT_UNLOCKED,
            user_id=user_id,
            tenant_id=tenant_id,
            correlation_id=correlation_id,
            details=details,
            result="success",
        )
        self.log_event(event)

    # ------------------------------------------------------------------
    # Convenience: password events
    # ------------------------------------------------------------------

    def log_password_changed(
        self,
        user_id: str,
        tenant_id: Optional[str] = None,
        changed_by: str = "self",
        reason: str = "user_change",
        ip_address: Optional[str] = None,
        correlation_id: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Log a password change event.

        Args:
            user_id: UUID of the user.
            tenant_id: UUID of the tenant.
            changed_by: Who changed the password (``self``, ``admin``, ``system``).
            reason: Reason for the change.
            ip_address: Client IP address.
            correlation_id: Request correlation ID.
            **kwargs: Additional details.
        """
        details: Dict[str, Any] = {
            "changed_by": changed_by,
            "reason": reason,
        }
        details.update(kwargs)

        event = AuthEvent(
            event_type=AuthEventType.PASSWORD_CHANGED,
            user_id=user_id,
            tenant_id=tenant_id,
            ip_address=ip_address,
            correlation_id=correlation_id,
            details=details,
            result="success",
        )
        self.log_event(event)

    # ------------------------------------------------------------------
    # Convenience: MFA events
    # ------------------------------------------------------------------

    def log_mfa_event(
        self,
        event_type: AuthEventType,
        user_id: str,
        tenant_id: Optional[str] = None,
        success: bool = True,
        ip_address: Optional[str] = None,
        correlation_id: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Log an MFA-related event.

        Args:
            event_type: One of ``MFA_SETUP``, ``MFA_VERIFIED``, ``MFA_FAILED``.
            user_id: UUID of the user.
            tenant_id: UUID of the tenant.
            success: Whether the MFA operation succeeded.
            ip_address: Client IP address.
            correlation_id: Request correlation ID.
            **kwargs: Additional details.
        """
        event = AuthEvent(
            event_type=event_type,
            user_id=user_id,
            tenant_id=tenant_id,
            ip_address=ip_address,
            correlation_id=correlation_id,
            details=dict(kwargs),
            result="success" if success else "failure",
        )
        self.log_event(event)

    # ------------------------------------------------------------------
    # Convenience: rate limit events
    # ------------------------------------------------------------------

    def log_rate_limited(
        self,
        ip_address: str,
        endpoint: str,
        limit: int,
        window_seconds: int,
        user_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        correlation_id: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Log a rate-limit enforcement event.

        Args:
            ip_address: Client IP that was rate-limited.
            endpoint: The endpoint that was rate-limited.
            limit: The configured limit.
            window_seconds: The window in which the limit applies.
            user_id: UUID of the user (if known).
            tenant_id: UUID of the tenant (if known).
            correlation_id: Request correlation ID.
            **kwargs: Additional details.
        """
        details: Dict[str, Any] = {
            "endpoint": endpoint,
            "limit": limit,
            "window_seconds": window_seconds,
        }
        details.update(kwargs)

        event = AuthEvent(
            event_type=AuthEventType.RATE_LIMITED,
            user_id=user_id,
            tenant_id=tenant_id,
            ip_address=ip_address,
            correlation_id=correlation_id,
            details=details,
            result="denied",
        )
        self.log_event(event)
