# -*- coding: utf-8 -*-
"""
Unified Audit Event Model - SEC-005: Centralized Audit Logging Service

Defines the UnifiedAuditEvent dataclass with 25+ fields for comprehensive
audit trail capture. Provides converters from legacy event types (auth, RBAC,
encryption) and builder pattern for fluent construction.

**Security Invariant:** This module NEVER logs sensitive data including:
- Passwords, tokens, secrets, keys
- PII (email, names, phone numbers)
- Ciphertext or plaintext content

Only opaque identifiers and metadata are included in audit records.

Author: GreenLang Framework Team
Date: February 2026
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from .event_types import (
    AuditEventCategory,
    AuditSeverity,
    AuditAction,
    AuditResult,
    UnifiedAuditEventType,
    AUTH_EVENT_TYPE_MAP,
    RBAC_EVENT_TYPE_MAP,
    ENCRYPTION_EVENT_TYPE_MAP,
)

if TYPE_CHECKING:
    from ..auth_service.auth_audit import AuthEvent
    from ..rbac_service.rbac_audit import RBACAuditEvent
    from ..encryption_service.encryption_audit import EncryptionAuditEvent


# ---------------------------------------------------------------------------
# Fields that must NEVER appear in log output (PII + Secrets)
# ---------------------------------------------------------------------------

_REDACTED_FIELDS = frozenset({
    # Authentication secrets
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
    "api_key",
    "api_secret",
    # Encryption secrets
    "private_key",
    "key",
    "dek",
    "kek",
    "data_key",
    "encryption_key",
    "decryption_key",
    "key_material",
    "raw_key",
    "plaintext",
    "ciphertext",
    "nonce",
    "auth_tag",
    "iv",
    "aad",
    # PII fields
    "email",
    "email_address",
    "name",
    "display_name",
    "full_name",
    "first_name",
    "last_name",
    "phone",
    "phone_number",
    "ssn",
    "social_security",
    "credit_card",
    "card_number",
    "cvv",
    "address",
    "street_address",
    "date_of_birth",
    "dob",
})


def _redact_dict(data: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively redact sensitive fields from a dictionary.

    Args:
        data: Dictionary potentially containing sensitive fields.

    Returns:
        Dictionary with sensitive fields redacted.
    """
    if not data:
        return {}

    result = {}
    for key, value in data.items():
        if key.lower() in _REDACTED_FIELDS:
            result[key] = "[REDACTED]"
        elif isinstance(value, dict):
            result[key] = _redact_dict(value)
        elif isinstance(value, list):
            result[key] = [
                _redact_dict(item) if isinstance(item, dict) else item
                for item in value
            ]
        else:
            result[key] = value
    return result


# ---------------------------------------------------------------------------
# Unified Audit Event Dataclass
# ---------------------------------------------------------------------------


@dataclass
class UnifiedAuditEvent:
    """Unified representation of an audit event with 25+ fields.

    Designed for comprehensive audit trail capture with multi-destination
    routing (PostgreSQL, Loki, Redis pub/sub).

    Attributes:
        event_id: Unique identifier for this event (UUID v4).
        correlation_id: Request correlation / trace ID for distributed tracing.
        event_type: The unified event type from UnifiedAuditEventType.
        category: Top-level event category (auth, rbac, encryption, etc.).
        severity: Event severity level (debug, info, warning, error, critical).
        user_id: UUID of the user who triggered the event.
        username: Username (for display, never logged with PII).
        tenant_id: UUID of the tenant scope.
        session_id: User session identifier.
        client_ip: Client IP address.
        user_agent: Client User-Agent header.
        geo_location: Geo-IP derived location (country, region, city).
        resource_type: Type of resource accessed (role, permission, key, etc.).
        resource_id: Identifier of the target resource.
        resource_name: Human-readable resource name.
        action: CRUD action performed (create, read, update, delete, etc.).
        result: Outcome of the operation (success, failure, denied, error).
        error_message: Error message if operation failed (sanitized).
        request_path: HTTP request path.
        request_method: HTTP method (GET, POST, PUT, DELETE).
        response_status: HTTP response status code.
        duration_ms: Operation duration in milliseconds.
        metadata: Additional event-specific key-value pairs.
        tags: Searchable tags for filtering.
        occurred_at: UTC datetime when the event occurred.
        recorded_at: UTC datetime when the event was recorded.
    """

    # Required fields
    event_type: UnifiedAuditEventType

    # Identity fields
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    correlation_id: Optional[str] = None

    # Event classification
    category: Optional[AuditEventCategory] = None
    severity: Optional[AuditSeverity] = None

    # Actor fields
    user_id: Optional[str] = None
    username: Optional[str] = None
    tenant_id: Optional[str] = None
    session_id: Optional[str] = None

    # Client context
    client_ip: Optional[str] = None
    user_agent: Optional[str] = None
    geo_location: Optional[Dict[str, str]] = None

    # Resource fields
    resource_type: Optional[str] = None
    resource_id: Optional[str] = None
    resource_name: Optional[str] = None

    # Action fields
    action: Optional[AuditAction] = None
    result: AuditResult = AuditResult.SUCCESS
    error_message: Optional[str] = None

    # Request context
    request_path: Optional[str] = None
    request_method: Optional[str] = None
    response_status: Optional[int] = None
    duration_ms: Optional[float] = None

    # Extensibility
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)

    # Timestamps
    occurred_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    recorded_at: Optional[datetime] = None

    def __post_init__(self) -> None:
        """Set derived fields after initialization."""
        # Auto-set category from event type if not provided
        if self.category is None:
            self.category = UnifiedAuditEventType.get_category(self.event_type)

        # Auto-set severity from event type if not provided
        if self.severity is None:
            self.severity = UnifiedAuditEventType.get_default_severity(
                self.event_type
            )

        # Set recorded_at to occurred_at if not provided
        if self.recorded_at is None:
            self.recorded_at = self.occurred_at

    def to_dict(self, redact_pii: bool = True) -> Dict[str, Any]:
        """Serialize the event to a log-safe dictionary.

        Strips any keys in ``metadata`` that match the redaction set.
        Ensures no sensitive data is included in the output.

        Args:
            redact_pii: Whether to redact PII fields (default True).

        Returns:
            Dictionary safe for JSON serialization and log emission.
        """
        safe_metadata = (
            _redact_dict(self.metadata) if redact_pii else self.metadata
        )

        return {
            "event_id": self.event_id,
            "correlation_id": self.correlation_id,
            "event_type": self.event_type.value,
            "category": self.category.value if self.category else None,
            "severity": self.severity.value if self.severity else None,
            "user_id": self.user_id,
            "username": self.username if not redact_pii else None,
            "tenant_id": self.tenant_id,
            "session_id": self.session_id,
            "client_ip": self.client_ip,
            "user_agent": self.user_agent,
            "geo_location": self.geo_location,
            "resource_type": self.resource_type,
            "resource_id": self.resource_id,
            "resource_name": self.resource_name,
            "action": self.action.value if self.action else None,
            "result": self.result.value,
            "error_message": self.error_message,
            "request_path": self.request_path,
            "request_method": self.request_method,
            "response_status": self.response_status,
            "duration_ms": self.duration_ms,
            "metadata": safe_metadata,
            "tags": self.tags,
            "occurred_at": self.occurred_at.isoformat(),
            "recorded_at": (
                self.recorded_at.isoformat() if self.recorded_at else None
            ),
        }

    def to_json(self, redact_pii: bool = True) -> str:
        """Serialize the event to a JSON string for Loki ingestion.

        Args:
            redact_pii: Whether to redact PII fields (default True).

        Returns:
            JSON string safe for log emission.
        """
        return json.dumps(self.to_dict(redact_pii=redact_pii), default=str)

    def compute_hash(self) -> str:
        """Compute SHA-256 hash for tamper detection.

        Creates a deterministic hash of the event for audit integrity.

        Returns:
            SHA-256 hex digest of the event.
        """
        # Create a canonical JSON representation
        canonical = json.dumps(
            self.to_dict(redact_pii=False),
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    @classmethod
    def from_auth_event(
        cls,
        auth_event: "AuthEvent",
        tenant_id: Optional[str] = None,
    ) -> "UnifiedAuditEvent":
        """Convert a legacy AuthEvent to UnifiedAuditEvent.

        Args:
            auth_event: Legacy auth event from auth_audit module.
            tenant_id: Optional tenant ID override.

        Returns:
            Unified audit event.
        """
        # Map legacy event type to unified type
        legacy_type = auth_event.event_type.value
        unified_type = AUTH_EVENT_TYPE_MAP.get(
            legacy_type, UnifiedAuditEventType.AUTH_LOGIN_SUCCESS
        )

        # Map result
        result_map = {
            "success": AuditResult.SUCCESS,
            "failure": AuditResult.FAILURE,
            "denied": AuditResult.DENIED,
        }
        result = result_map.get(auth_event.result, AuditResult.SUCCESS)

        return cls(
            event_type=unified_type,
            correlation_id=auth_event.correlation_id,
            category=AuditEventCategory.AUTH,
            user_id=auth_event.user_id,
            tenant_id=tenant_id or auth_event.tenant_id,
            client_ip=auth_event.ip_address,
            user_agent=auth_event.user_agent,
            result=result,
            metadata=auth_event.details,
            occurred_at=auth_event.timestamp,
        )

    @classmethod
    def from_rbac_event(
        cls,
        rbac_event: "RBACAuditEvent",
    ) -> "UnifiedAuditEvent":
        """Convert a legacy RBACAuditEvent to UnifiedAuditEvent.

        Args:
            rbac_event: Legacy RBAC event from rbac_audit module.

        Returns:
            Unified audit event.
        """
        # Map legacy event type to unified type
        legacy_type = rbac_event.event_type.value
        unified_type = RBAC_EVENT_TYPE_MAP.get(
            legacy_type, UnifiedAuditEventType.RBAC_AUTHORIZATION_ALLOWED
        )

        # Map result
        result_map = {
            "success": AuditResult.SUCCESS,
            "failure": AuditResult.FAILURE,
            "denied": AuditResult.DENIED,
        }
        result = result_map.get(rbac_event.result, AuditResult.SUCCESS)

        # Map action
        action_map = {
            "create_role": AuditAction.CREATE,
            "update_role": AuditAction.UPDATE,
            "delete_role": AuditAction.DELETE,
            "enable_role": AuditAction.UPDATE,
            "disable_role": AuditAction.UPDATE,
            "grant_permission": AuditAction.GRANT,
            "revoke_permission": AuditAction.REVOKE,
            "assign_role": AuditAction.GRANT,
            "revoke_assignment": AuditAction.REVOKE,
            "check_permission": AuditAction.READ,
            "invalidate_cache": AuditAction.DELETE,
        }
        action = action_map.get(rbac_event.action or "", None)

        # Merge old_value and new_value into metadata
        metadata = dict(rbac_event.details)
        if rbac_event.old_value:
            metadata["old_value"] = rbac_event.old_value
        if rbac_event.new_value:
            metadata["new_value"] = rbac_event.new_value

        return cls(
            event_type=unified_type,
            correlation_id=rbac_event.correlation_id,
            category=AuditEventCategory.RBAC,
            user_id=rbac_event.actor_id,
            tenant_id=rbac_event.tenant_id,
            client_ip=rbac_event.ip_address,
            resource_type=rbac_event.target_type,
            resource_id=rbac_event.target_id,
            action=action,
            result=result,
            metadata=metadata,
            occurred_at=rbac_event.timestamp,
        )

    @classmethod
    def from_encryption_event(
        cls,
        encryption_event: "EncryptionAuditEvent",
    ) -> "UnifiedAuditEvent":
        """Convert a legacy EncryptionAuditEvent to UnifiedAuditEvent.

        Args:
            encryption_event: Legacy encryption event from encryption_audit module.

        Returns:
            Unified audit event.
        """
        # Map legacy event type to unified type
        legacy_type = encryption_event.event_type.value
        unified_type = ENCRYPTION_EVENT_TYPE_MAP.get(
            legacy_type, UnifiedAuditEventType.ENCRYPTION_PERFORMED
        )

        # Map result
        result = AuditResult.SUCCESS if encryption_event.success else AuditResult.FAILURE

        # Map action from operation
        action_map = {
            "encrypt": AuditAction.ENCRYPT,
            "decrypt": AuditAction.DECRYPT,
            "generate_key": AuditAction.CREATE,
            "rotate_key": AuditAction.UPDATE,
            "cache_lookup": AuditAction.READ,
            "cache_invalidate": AuditAction.DELETE,
        }
        action = action_map.get(encryption_event.operation, None)

        # Build metadata
        metadata = dict(encryption_event.metadata)
        if encryption_event.key_version:
            metadata["key_version"] = encryption_event.key_version
        if encryption_event.data_class:
            metadata["data_class"] = encryption_event.data_class

        return cls(
            event_type=unified_type,
            correlation_id=encryption_event.correlation_id,
            category=AuditEventCategory.ENCRYPTION,
            tenant_id=encryption_event.tenant_id,
            client_ip=encryption_event.client_ip,
            action=action,
            result=result,
            error_message=encryption_event.error_message,
            duration_ms=encryption_event.duration_ms,
            metadata=metadata,
            occurred_at=encryption_event.timestamp,
        )


# ---------------------------------------------------------------------------
# Event Builder for Fluent Construction
# ---------------------------------------------------------------------------


class EventBuilder:
    """Builder pattern for constructing UnifiedAuditEvent instances.

    Provides a fluent interface for building audit events with method chaining.

    Example:
        >>> event = (
        ...     EventBuilder(UnifiedAuditEventType.AUTH_LOGIN_SUCCESS)
        ...     .with_user(user_id="u-123", username="john")
        ...     .with_tenant(tenant_id="t-corp")
        ...     .with_client(ip="10.0.0.1", user_agent="Mozilla/5.0")
        ...     .with_result(AuditResult.SUCCESS)
        ...     .build()
        ... )
    """

    def __init__(self, event_type: UnifiedAuditEventType) -> None:
        """Initialize builder with event type.

        Args:
            event_type: The type of event to build.
        """
        self._event_type = event_type
        self._data: Dict[str, Any] = {}

    def with_correlation_id(self, correlation_id: str) -> "EventBuilder":
        """Set correlation ID for distributed tracing.

        Args:
            correlation_id: Request correlation / trace ID.

        Returns:
            Self for method chaining.
        """
        self._data["correlation_id"] = correlation_id
        return self

    def with_user(
        self,
        user_id: Optional[str] = None,
        username: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> "EventBuilder":
        """Set user/actor information.

        Args:
            user_id: UUID of the user.
            username: Username (display only).
            session_id: Session identifier.

        Returns:
            Self for method chaining.
        """
        if user_id:
            self._data["user_id"] = user_id
        if username:
            self._data["username"] = username
        if session_id:
            self._data["session_id"] = session_id
        return self

    def with_tenant(self, tenant_id: str) -> "EventBuilder":
        """Set tenant scope.

        Args:
            tenant_id: UUID of the tenant.

        Returns:
            Self for method chaining.
        """
        self._data["tenant_id"] = tenant_id
        return self

    def with_client(
        self,
        ip: Optional[str] = None,
        user_agent: Optional[str] = None,
        geo_location: Optional[Dict[str, str]] = None,
    ) -> "EventBuilder":
        """Set client context.

        Args:
            ip: Client IP address.
            user_agent: Client User-Agent header.
            geo_location: Geo-IP derived location.

        Returns:
            Self for method chaining.
        """
        if ip:
            self._data["client_ip"] = ip
        if user_agent:
            self._data["user_agent"] = user_agent
        if geo_location:
            self._data["geo_location"] = geo_location
        return self

    def with_resource(
        self,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        resource_name: Optional[str] = None,
    ) -> "EventBuilder":
        """Set resource information.

        Args:
            resource_type: Type of resource (role, permission, key, etc.).
            resource_id: Identifier of the resource.
            resource_name: Human-readable resource name.

        Returns:
            Self for method chaining.
        """
        if resource_type:
            self._data["resource_type"] = resource_type
        if resource_id:
            self._data["resource_id"] = resource_id
        if resource_name:
            self._data["resource_name"] = resource_name
        return self

    def with_action(self, action: AuditAction) -> "EventBuilder":
        """Set the action performed.

        Args:
            action: CRUD action type.

        Returns:
            Self for method chaining.
        """
        self._data["action"] = action
        return self

    def with_result(
        self,
        result: AuditResult,
        error_message: Optional[str] = None,
    ) -> "EventBuilder":
        """Set the operation result.

        Args:
            result: Outcome of the operation.
            error_message: Error message if operation failed.

        Returns:
            Self for method chaining.
        """
        self._data["result"] = result
        if error_message:
            self._data["error_message"] = error_message
        return self

    def with_request(
        self,
        path: Optional[str] = None,
        method: Optional[str] = None,
        status: Optional[int] = None,
        duration_ms: Optional[float] = None,
    ) -> "EventBuilder":
        """Set HTTP request context.

        Args:
            path: Request path.
            method: HTTP method.
            status: Response status code.
            duration_ms: Request duration in milliseconds.

        Returns:
            Self for method chaining.
        """
        if path:
            self._data["request_path"] = path
        if method:
            self._data["request_method"] = method
        if status:
            self._data["response_status"] = status
        if duration_ms:
            self._data["duration_ms"] = duration_ms
        return self

    def with_metadata(self, **kwargs: Any) -> "EventBuilder":
        """Add metadata key-value pairs.

        Args:
            **kwargs: Metadata key-value pairs.

        Returns:
            Self for method chaining.
        """
        if "metadata" not in self._data:
            self._data["metadata"] = {}
        self._data["metadata"].update(kwargs)
        return self

    def with_tags(self, *tags: str) -> "EventBuilder":
        """Add searchable tags.

        Args:
            *tags: Tag strings to add.

        Returns:
            Self for method chaining.
        """
        if "tags" not in self._data:
            self._data["tags"] = []
        self._data["tags"].extend(tags)
        return self

    def with_severity(self, severity: AuditSeverity) -> "EventBuilder":
        """Override the default severity.

        Args:
            severity: Severity level.

        Returns:
            Self for method chaining.
        """
        self._data["severity"] = severity
        return self

    def build(self) -> UnifiedAuditEvent:
        """Build the UnifiedAuditEvent instance.

        Returns:
            Constructed audit event.
        """
        return UnifiedAuditEvent(
            event_type=self._event_type,
            **self._data,
        )


__all__ = [
    "UnifiedAuditEvent",
    "EventBuilder",
    "_REDACTED_FIELDS",
    "_redact_dict",
]
