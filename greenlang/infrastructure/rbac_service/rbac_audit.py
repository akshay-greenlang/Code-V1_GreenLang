# -*- coding: utf-8 -*-
"""
RBAC Audit Logger - Authorization Service (SEC-002)

Emits structured RBAC authorization events as JSON log records for Loki
ingestion.  Every RBAC event (role lifecycle, permission grants/revocations,
role assignments, authorization decisions) is captured with a consistent
schema that includes correlation IDs, tenant isolation, and actor metadata.

**Security invariant:** This module NEVER logs PII such as email addresses,
display names, or other personal information.  Only opaque identifiers
(user_id, tenant_id, role_id, permission_id) are included.

Classes:
    - RBACAuditEventType: Enumeration of all auditable RBAC events.
    - RBACAuditEvent: Structured RBAC event data class.
    - RBACAuditLogger: Logger that emits JSON-structured events and
      optionally writes to ``security.rbac_audit_log`` via async DB.

Example:
    >>> audit = RBACAuditLogger()
    >>> await audit.log_event(
    ...     tenant_id="t-corp",
    ...     actor_id="u-admin",
    ...     event_type=RBACAuditEventType.ROLE_CREATED,
    ...     target_type="role",
    ...     target_id="r-analyst",
    ...     action="create_role",
    ... )

Author: GreenLang Framework Team
Date: February 2026
Status: Production Ready
"""

from __future__ import annotations

import asyncio
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


class RBACAuditEventType(str, Enum):
    """Enumeration of all auditable RBAC events.

    Values are lowercase snake_case strings suitable for use as Loki labels.
    """

    ROLE_CREATED = "role_created"
    ROLE_UPDATED = "role_updated"
    ROLE_DELETED = "role_deleted"
    ROLE_ENABLED = "role_enabled"
    ROLE_DISABLED = "role_disabled"
    PERMISSION_GRANTED = "permission_granted"
    PERMISSION_REVOKED = "permission_revoked"
    ROLE_ASSIGNED = "role_assigned"
    ROLE_REVOKED = "role_revoked"
    ROLE_EXPIRED = "role_expired"
    AUTHORIZATION_ALLOWED = "authorization_allowed"
    AUTHORIZATION_DENIED = "authorization_denied"
    CACHE_INVALIDATED = "cache_invalidated"


# ---------------------------------------------------------------------------
# Redacted fields
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
    "email",
    "name",
    "display_name",
    "full_name",
    "first_name",
    "last_name",
    "phone",
    "phone_number",
})


# ---------------------------------------------------------------------------
# Event data class
# ---------------------------------------------------------------------------


@dataclass
class RBACAuditEvent:
    """Structured representation of an RBAC audit event.

    Attributes:
        event_type: The type of RBAC event.
        tenant_id: UUID of the tenant scope.
        actor_id: UUID of the actor who triggered the event.
        target_type: Type of the target entity (``role``, ``permission``,
            ``assignment``, ``user``).
        target_id: Identifier of the target entity.
        action: Human-readable action description (e.g. ``create_role``).
        old_value: Previous state before the change (serializable dict).
        new_value: New state after the change (serializable dict).
        ip_address: Client IP address.
        correlation_id: Request correlation / trace ID.
        details: Additional event-specific key-value pairs.
        result: Outcome string (``success``, ``failure``, ``denied``).
        timestamp: UTC datetime of the event.
    """

    event_type: RBACAuditEventType
    tenant_id: Optional[str] = None
    actor_id: Optional[str] = None
    target_type: Optional[str] = None
    target_id: Optional[str] = None
    action: Optional[str] = None
    old_value: Optional[Dict[str, Any]] = None
    new_value: Optional[Dict[str, Any]] = None
    ip_address: Optional[str] = None
    correlation_id: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    result: str = "success"
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the event to a log-safe dictionary.

        Strips any keys in ``details``, ``old_value``, and ``new_value``
        that match the PII redaction set.

        Returns:
            Dictionary safe for JSON serialization and log emission.
        """
        safe_details = {
            k: v
            for k, v in self.details.items()
            if k.lower() not in _REDACTED_FIELDS
        }

        safe_old = None
        if self.old_value is not None:
            safe_old = {
                k: v
                for k, v in self.old_value.items()
                if k.lower() not in _REDACTED_FIELDS
            }

        safe_new = None
        if self.new_value is not None:
            safe_new = {
                k: v
                for k, v in self.new_value.items()
                if k.lower() not in _REDACTED_FIELDS
            }

        return {
            "event_type": self.event_type.value,
            "event_category": "rbac",
            "tenant_id": self.tenant_id,
            "actor_id": self.actor_id,
            "target_type": self.target_type,
            "target_id": self.target_id,
            "action": self.action,
            "old_value": safe_old,
            "new_value": safe_new,
            "ip_address": self.ip_address,
            "correlation_id": self.correlation_id,
            "result": self.result,
            "details": safe_details,
            "timestamp": self.timestamp.isoformat(),
        }


# ---------------------------------------------------------------------------
# Audit logger
# ---------------------------------------------------------------------------


class RBACAuditLogger:
    """RBAC audit logger that writes to both structured JSON logs (for Loki)
    and optionally to ``security.rbac_audit_log`` in PostgreSQL.

    Uses a dedicated ``greenlang.rbac.audit`` logger so that Loki pipeline
    stages can match on the logger name and apply appropriate labels.

    The logger is safe for concurrent use.  Database writes are async
    fire-and-forget so they never block the request path.

    Example:
        >>> audit = RBACAuditLogger()
        >>> await audit.log_event(
        ...     tenant_id="t-1",
        ...     actor_id="u-admin",
        ...     event_type=RBACAuditEventType.ROLE_CREATED,
        ...     target_type="role",
        ...     target_id="r-analyst",
        ...     action="create_role",
        ... )
    """

    def __init__(self, db_pool: Any = None) -> None:
        """Initialize the RBAC audit logger.

        Args:
            db_pool: Optional async database connection pool (psycopg_pool).
                If provided, audit events are also written to the
                ``security.rbac_audit_log`` table.
        """
        self._logger = logging.getLogger("greenlang.rbac.audit")
        self._pool = db_pool

    # ------------------------------------------------------------------
    # Core log method
    # ------------------------------------------------------------------

    async def log_event(
        self,
        *,
        tenant_id: Optional[str] = None,
        actor_id: Optional[str] = None,
        event_type: RBACAuditEventType,
        target_type: Optional[str] = None,
        target_id: Optional[str] = None,
        action: Optional[str] = None,
        old_value: Optional[Dict[str, Any]] = None,
        new_value: Optional[Dict[str, Any]] = None,
        ip_address: Optional[str] = None,
        correlation_id: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Write an RBAC audit event to structured logs and the database.

        The structured JSON log is always emitted (even if the DB is
        unavailable).  The database write is async fire-and-forget so it
        never blocks the calling request.

        Args:
            tenant_id: UUID of the tenant scope.
            actor_id: UUID of the actor who triggered the event.
            event_type: The type of RBAC event.
            target_type: Type of the target entity.
            target_id: Identifier of the target entity.
            action: Human-readable action description.
            old_value: Previous state before the change.
            new_value: New state after the change.
            ip_address: Client IP address.
            correlation_id: Request correlation / trace ID.
            **kwargs: Additional details (PII keys are auto-redacted).
        """
        event = RBACAuditEvent(
            event_type=event_type,
            tenant_id=tenant_id,
            actor_id=actor_id,
            target_type=target_type,
            target_id=target_id,
            action=action,
            old_value=old_value,
            new_value=new_value,
            ip_address=ip_address,
            correlation_id=correlation_id,
            details=dict(kwargs),
            result="denied" if event_type == RBACAuditEventType.AUTHORIZATION_DENIED else "success",
        )

        # 1. Always emit structured JSON log
        self._emit_log(event)

        # 2. Async fire-and-forget DB write (if pool is available)
        if self._pool is not None:
            asyncio.ensure_future(self._write_to_db(event))

    def _emit_log(self, event: RBACAuditEvent) -> None:
        """Emit an audit event as a structured JSON log record.

        The log level is determined by the event result:
        - ``success`` -> INFO
        - ``failure`` -> WARNING
        - ``denied``  -> WARNING

        Args:
            event: The RBAC audit event to log.
        """
        payload = event.to_dict()

        level = logging.INFO
        if event.result in ("failure", "denied"):
            level = logging.WARNING

        self._logger.log(
            level,
            json.dumps(payload, default=str),
            extra={
                "event_type": event.event_type.value,
                "event_category": "rbac",
                "rbac_result": event.result,
                "tenant_id": event.tenant_id or "",
                "actor_id": event.actor_id or "",
            },
        )

    async def _write_to_db(self, event: RBACAuditEvent) -> None:
        """Write an audit event to ``security.rbac_audit_log``.

        This is fire-and-forget; errors are logged but never propagated
        to the caller.

        Args:
            event: The RBAC audit event to persist.
        """
        try:
            async with self._pool.connection() as conn:
                await conn.execute(
                    """
                    INSERT INTO security.rbac_audit_log (
                        tenant_id, actor_id, event_type, target_type,
                        target_id, action, old_value, new_value,
                        ip_address, correlation_id, result, created_at
                    ) VALUES (
                        %(tenant_id)s, %(actor_id)s, %(event_type)s,
                        %(target_type)s, %(target_id)s, %(action)s,
                        %(old_value)s, %(new_value)s, %(ip_address)s,
                        %(correlation_id)s, %(result)s, %(created_at)s
                    )
                    """,
                    {
                        "tenant_id": event.tenant_id,
                        "actor_id": event.actor_id,
                        "event_type": event.event_type.value,
                        "target_type": event.target_type,
                        "target_id": event.target_id,
                        "action": event.action,
                        "old_value": json.dumps(event.old_value, default=str)
                        if event.old_value
                        else None,
                        "new_value": json.dumps(event.new_value, default=str)
                        if event.new_value
                        else None,
                        "ip_address": event.ip_address,
                        "correlation_id": event.correlation_id,
                        "result": event.result,
                        "created_at": event.timestamp.isoformat(),
                    },
                )
        except Exception:
            logger.exception(
                "Failed to write RBAC audit event to database: %s",
                event.event_type.value,
            )

    # ------------------------------------------------------------------
    # Convenience: role lifecycle events
    # ------------------------------------------------------------------

    async def log_role_created(
        self,
        *,
        tenant_id: Optional[str] = None,
        actor_id: str,
        role_id: str,
        role_name: str,
        ip_address: Optional[str] = None,
        correlation_id: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Log a role creation event.

        Args:
            tenant_id: UUID of the tenant.
            actor_id: UUID of the actor who created the role.
            role_id: UUID of the created role.
            role_name: Name of the created role.
            ip_address: Client IP address.
            correlation_id: Request correlation ID.
            **kwargs: Additional details.
        """
        await self.log_event(
            tenant_id=tenant_id,
            actor_id=actor_id,
            event_type=RBACAuditEventType.ROLE_CREATED,
            target_type="role",
            target_id=role_id,
            action="create_role",
            new_value={"role_id": role_id, "role_name": role_name},
            ip_address=ip_address,
            correlation_id=correlation_id,
            **kwargs,
        )

    async def log_role_updated(
        self,
        *,
        tenant_id: Optional[str] = None,
        actor_id: str,
        role_id: str,
        old_value: Optional[Dict[str, Any]] = None,
        new_value: Optional[Dict[str, Any]] = None,
        ip_address: Optional[str] = None,
        correlation_id: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Log a role update event.

        Args:
            tenant_id: UUID of the tenant.
            actor_id: UUID of the actor who updated the role.
            role_id: UUID of the updated role.
            old_value: Previous role state.
            new_value: New role state.
            ip_address: Client IP address.
            correlation_id: Request correlation ID.
            **kwargs: Additional details.
        """
        await self.log_event(
            tenant_id=tenant_id,
            actor_id=actor_id,
            event_type=RBACAuditEventType.ROLE_UPDATED,
            target_type="role",
            target_id=role_id,
            action="update_role",
            old_value=old_value,
            new_value=new_value,
            ip_address=ip_address,
            correlation_id=correlation_id,
            **kwargs,
        )

    async def log_role_deleted(
        self,
        *,
        tenant_id: Optional[str] = None,
        actor_id: str,
        role_id: str,
        role_name: str,
        ip_address: Optional[str] = None,
        correlation_id: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Log a role deletion event.

        Args:
            tenant_id: UUID of the tenant.
            actor_id: UUID of the actor who deleted the role.
            role_id: UUID of the deleted role.
            role_name: Name of the deleted role.
            ip_address: Client IP address.
            correlation_id: Request correlation ID.
            **kwargs: Additional details.
        """
        await self.log_event(
            tenant_id=tenant_id,
            actor_id=actor_id,
            event_type=RBACAuditEventType.ROLE_DELETED,
            target_type="role",
            target_id=role_id,
            action="delete_role",
            old_value={"role_id": role_id, "role_name": role_name},
            ip_address=ip_address,
            correlation_id=correlation_id,
            **kwargs,
        )

    async def log_role_enabled(
        self,
        *,
        tenant_id: Optional[str] = None,
        actor_id: str,
        role_id: str,
        ip_address: Optional[str] = None,
        correlation_id: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Log a role enable event.

        Args:
            tenant_id: UUID of the tenant.
            actor_id: UUID of the actor.
            role_id: UUID of the enabled role.
            ip_address: Client IP address.
            correlation_id: Request correlation ID.
            **kwargs: Additional details.
        """
        await self.log_event(
            tenant_id=tenant_id,
            actor_id=actor_id,
            event_type=RBACAuditEventType.ROLE_ENABLED,
            target_type="role",
            target_id=role_id,
            action="enable_role",
            ip_address=ip_address,
            correlation_id=correlation_id,
            **kwargs,
        )

    async def log_role_disabled(
        self,
        *,
        tenant_id: Optional[str] = None,
        actor_id: str,
        role_id: str,
        ip_address: Optional[str] = None,
        correlation_id: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Log a role disable event.

        Args:
            tenant_id: UUID of the tenant.
            actor_id: UUID of the actor.
            role_id: UUID of the disabled role.
            ip_address: Client IP address.
            correlation_id: Request correlation ID.
            **kwargs: Additional details.
        """
        await self.log_event(
            tenant_id=tenant_id,
            actor_id=actor_id,
            event_type=RBACAuditEventType.ROLE_DISABLED,
            target_type="role",
            target_id=role_id,
            action="disable_role",
            ip_address=ip_address,
            correlation_id=correlation_id,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Convenience: permission events
    # ------------------------------------------------------------------

    async def log_permission_granted(
        self,
        *,
        tenant_id: Optional[str] = None,
        actor_id: str,
        role_id: str,
        permission_id: str,
        effect: str = "allow",
        ip_address: Optional[str] = None,
        correlation_id: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Log a permission grant event.

        Args:
            tenant_id: UUID of the tenant.
            actor_id: UUID of the actor who granted the permission.
            role_id: UUID of the role receiving the permission.
            permission_id: UUID of the granted permission.
            effect: Permission effect (``allow`` or ``deny``).
            ip_address: Client IP address.
            correlation_id: Request correlation ID.
            **kwargs: Additional details.
        """
        await self.log_event(
            tenant_id=tenant_id,
            actor_id=actor_id,
            event_type=RBACAuditEventType.PERMISSION_GRANTED,
            target_type="role_permission",
            target_id=f"{role_id}:{permission_id}",
            action="grant_permission",
            new_value={
                "role_id": role_id,
                "permission_id": permission_id,
                "effect": effect,
            },
            ip_address=ip_address,
            correlation_id=correlation_id,
            **kwargs,
        )

    async def log_permission_revoked(
        self,
        *,
        tenant_id: Optional[str] = None,
        actor_id: str,
        role_id: str,
        permission_id: str,
        ip_address: Optional[str] = None,
        correlation_id: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Log a permission revocation event.

        Args:
            tenant_id: UUID of the tenant.
            actor_id: UUID of the actor who revoked the permission.
            role_id: UUID of the role losing the permission.
            permission_id: UUID of the revoked permission.
            ip_address: Client IP address.
            correlation_id: Request correlation ID.
            **kwargs: Additional details.
        """
        await self.log_event(
            tenant_id=tenant_id,
            actor_id=actor_id,
            event_type=RBACAuditEventType.PERMISSION_REVOKED,
            target_type="role_permission",
            target_id=f"{role_id}:{permission_id}",
            action="revoke_permission",
            old_value={
                "role_id": role_id,
                "permission_id": permission_id,
            },
            ip_address=ip_address,
            correlation_id=correlation_id,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Convenience: assignment events
    # ------------------------------------------------------------------

    async def log_role_assigned(
        self,
        *,
        tenant_id: Optional[str] = None,
        actor_id: str,
        user_id: str,
        role_id: str,
        assignment_id: Optional[str] = None,
        expires_at: Optional[str] = None,
        ip_address: Optional[str] = None,
        correlation_id: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Log a role assignment event.

        Args:
            tenant_id: UUID of the tenant.
            actor_id: UUID of the actor who made the assignment.
            user_id: UUID of the user receiving the role.
            role_id: UUID of the assigned role.
            assignment_id: UUID of the assignment record.
            expires_at: Optional expiration time (ISO format).
            ip_address: Client IP address.
            correlation_id: Request correlation ID.
            **kwargs: Additional details.
        """
        await self.log_event(
            tenant_id=tenant_id,
            actor_id=actor_id,
            event_type=RBACAuditEventType.ROLE_ASSIGNED,
            target_type="assignment",
            target_id=assignment_id or f"{user_id}:{role_id}",
            action="assign_role",
            new_value={
                "user_id": user_id,
                "role_id": role_id,
                "expires_at": expires_at,
            },
            ip_address=ip_address,
            correlation_id=correlation_id,
            **kwargs,
        )

    async def log_role_revoked_assignment(
        self,
        *,
        tenant_id: Optional[str] = None,
        actor_id: str,
        assignment_id: str,
        user_id: Optional[str] = None,
        role_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        correlation_id: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Log a role assignment revocation event.

        Args:
            tenant_id: UUID of the tenant.
            actor_id: UUID of the actor who revoked the assignment.
            assignment_id: UUID of the revoked assignment.
            user_id: UUID of the user (if known).
            role_id: UUID of the role (if known).
            ip_address: Client IP address.
            correlation_id: Request correlation ID.
            **kwargs: Additional details.
        """
        await self.log_event(
            tenant_id=tenant_id,
            actor_id=actor_id,
            event_type=RBACAuditEventType.ROLE_REVOKED,
            target_type="assignment",
            target_id=assignment_id,
            action="revoke_assignment",
            old_value={
                "assignment_id": assignment_id,
                "user_id": user_id,
                "role_id": role_id,
            },
            ip_address=ip_address,
            correlation_id=correlation_id,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Convenience: authorization decision events
    # ------------------------------------------------------------------

    async def log_authorization_decision(
        self,
        *,
        tenant_id: Optional[str] = None,
        user_id: str,
        resource: str,
        action: str,
        allowed: bool,
        matched_permissions: Optional[list] = None,
        evaluation_time_ms: Optional[float] = None,
        cache_hit: bool = False,
        ip_address: Optional[str] = None,
        correlation_id: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Log an authorization decision (allowed or denied).

        Args:
            tenant_id: UUID of the tenant.
            user_id: UUID of the user being evaluated.
            resource: Resource being accessed.
            action: Action being attempted.
            allowed: Whether the action was authorized.
            matched_permissions: List of permissions that matched.
            evaluation_time_ms: Evaluation duration in milliseconds.
            cache_hit: Whether the result came from cache.
            ip_address: Client IP address.
            correlation_id: Request correlation ID.
            **kwargs: Additional details.
        """
        event_type = (
            RBACAuditEventType.AUTHORIZATION_ALLOWED
            if allowed
            else RBACAuditEventType.AUTHORIZATION_DENIED
        )

        await self.log_event(
            tenant_id=tenant_id,
            actor_id=user_id,
            event_type=event_type,
            target_type="authorization",
            target_id=f"{resource}:{action}",
            action="check_permission",
            ip_address=ip_address,
            correlation_id=correlation_id,
            resource=resource,
            checked_action=action,
            matched_permissions=matched_permissions or [],
            evaluation_time_ms=evaluation_time_ms,
            cache_hit=cache_hit,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Convenience: cache events
    # ------------------------------------------------------------------

    async def log_cache_invalidated(
        self,
        *,
        tenant_id: Optional[str] = None,
        actor_id: Optional[str] = None,
        reason: str = "manual",
        scope: str = "tenant",
        correlation_id: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Log a cache invalidation event.

        Args:
            tenant_id: UUID of the tenant whose cache was invalidated.
            actor_id: UUID of the actor who triggered the invalidation.
            reason: Reason for invalidation (``manual``, ``role_change``,
                ``permission_change``, ``assignment_change``).
            scope: Invalidation scope (``tenant``, ``user``, ``global``).
            correlation_id: Request correlation ID.
            **kwargs: Additional details.
        """
        await self.log_event(
            tenant_id=tenant_id,
            actor_id=actor_id,
            event_type=RBACAuditEventType.CACHE_INVALIDATED,
            target_type="cache",
            target_id=tenant_id or "global",
            action="invalidate_cache",
            correlation_id=correlation_id,
            reason=reason,
            scope=scope,
            **kwargs,
        )
