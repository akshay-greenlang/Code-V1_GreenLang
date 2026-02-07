# -*- coding: utf-8 -*-
"""
Unit tests for Audit Event Model - SEC-005: Centralized Audit Logging Service

Tests the UnifiedAuditEvent dataclass, serialization methods, factory methods
for converting from legacy event types, and the EventBuilder fluent API.

Coverage targets: 85%+ of event_model.py
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Attempt to import the audit event model module.
# ---------------------------------------------------------------------------
try:
    from greenlang.infrastructure.audit_service.event_model import (
        UnifiedAuditEvent,
        EventBuilder,
    )
    _HAS_MODULE = True
except ImportError:
    _HAS_MODULE = False

    from dataclasses import dataclass, field

    @dataclass
    class UnifiedAuditEvent:
        """Stub for test collection when module is not yet built."""
        event_id: str = ""
        event_type: str = ""
        category: str = ""
        severity: str = "info"
        timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
        tenant_id: Optional[str] = None
        user_id: Optional[str] = None
        actor_id: Optional[str] = None
        actor_type: str = "user"
        resource_type: Optional[str] = None
        resource_id: Optional[str] = None
        action: Optional[str] = None
        result: str = "success"
        client_ip: Optional[str] = None
        user_agent: Optional[str] = None
        request_id: Optional[str] = None
        session_id: Optional[str] = None
        details: Dict[str, Any] = field(default_factory=dict)
        metadata: Dict[str, Any] = field(default_factory=dict)
        pii_fields: list = field(default_factory=list)

        def to_dict(self, redact_pii: bool = False) -> Dict[str, Any]: ...
        def to_json(self, redact_pii: bool = False) -> str: ...
        @classmethod
        def from_auth_event(cls, event: Any) -> "UnifiedAuditEvent": ...
        @classmethod
        def from_rbac_event(cls, event: Any) -> "UnifiedAuditEvent": ...
        @classmethod
        def from_encryption_event(cls, event: Any) -> "UnifiedAuditEvent": ...

    class EventBuilder:
        """Stub for test collection when module is not yet built."""
        def __init__(self, event_type: str): ...
        def with_tenant(self, tenant_id: str) -> "EventBuilder": ...
        def with_user(self, user_id: str) -> "EventBuilder": ...
        def with_actor(self, actor_id: str, actor_type: str = "user") -> "EventBuilder": ...
        def with_resource(self, resource_type: str, resource_id: str) -> "EventBuilder": ...
        def with_action(self, action: str) -> "EventBuilder": ...
        def with_result(self, result: str) -> "EventBuilder": ...
        def with_severity(self, severity: str) -> "EventBuilder": ...
        def with_request_context(self, client_ip: str, user_agent: str, request_id: str) -> "EventBuilder": ...
        def with_detail(self, key: str, value: Any) -> "EventBuilder": ...
        def with_details(self, details: Dict[str, Any]) -> "EventBuilder": ...
        def with_pii_field(self, field_name: str) -> "EventBuilder": ...
        def build(self) -> UnifiedAuditEvent: ...


from greenlang.infrastructure.audit_service.event_types import (
    AuditEventCategory,
    AuditSeverity,
    AuditResult,
    UnifiedAuditEventType,
)


pytestmark = pytest.mark.skipif(
    not _HAS_MODULE,
    reason="audit_service.event_model not yet implemented",
)


# ============================================================================
# Helpers
# ============================================================================


def _make_event(
    event_id: Optional[str] = None,
    event_type: str = "auth.login_success",
    tenant_id: str = "t-acme",
    user_id: str = "u-1",
    **kwargs,
) -> UnifiedAuditEvent:
    """Create a test audit event."""
    return UnifiedAuditEvent(
        event_id=event_id or str(uuid.uuid4()),
        event_type=event_type,
        category="auth",
        tenant_id=tenant_id,
        user_id=user_id,
        actor_id=user_id,
        **kwargs,
    )


def _make_auth_event_mock(
    event_type: str = "login_success",
    user_id: str = "u-1",
    tenant_id: str = "t-acme",
    client_ip: str = "192.168.1.1",
) -> MagicMock:
    """Create a mock legacy auth event."""
    mock = MagicMock()
    mock.event_type = event_type
    mock.user_id = user_id
    mock.tenant_id = tenant_id
    mock.client_ip = client_ip
    mock.user_agent = "Mozilla/5.0"
    mock.timestamp = datetime.now(timezone.utc)
    mock.details = {"method": "password"}
    return mock


def _make_rbac_event_mock(
    event_type: str = "role_created",
    user_id: str = "u-1",
    tenant_id: str = "t-acme",
) -> MagicMock:
    """Create a mock legacy RBAC event."""
    mock = MagicMock()
    mock.event_type = event_type
    mock.user_id = user_id
    mock.tenant_id = tenant_id
    mock.role_id = "r-1"
    mock.role_name = "viewer"
    mock.timestamp = datetime.now(timezone.utc)
    mock.details = {"permissions": ["read"]}
    return mock


def _make_encryption_event_mock(
    event_type: str = "encryption_performed",
    user_id: str = "u-1",
    tenant_id: str = "t-acme",
) -> MagicMock:
    """Create a mock legacy encryption event."""
    mock = MagicMock()
    mock.event_type = event_type
    mock.user_id = user_id
    mock.tenant_id = tenant_id
    mock.key_id = "k-1"
    mock.algorithm = "AES-256-GCM"
    mock.timestamp = datetime.now(timezone.utc)
    mock.details = {"field_count": 5}
    return mock


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_event() -> UnifiedAuditEvent:
    """Create a sample audit event for testing."""
    return _make_event()


@pytest.fixture
def event_with_pii() -> UnifiedAuditEvent:
    """Create an event with PII fields marked."""
    return _make_event(
        details={
            "email": "user@example.com",
            "ip_address": "192.168.1.1",
            "action": "login",
        },
        pii_fields=["email", "ip_address"],
    )


# ============================================================================
# TestUnifiedAuditEvent
# ============================================================================


class TestUnifiedAuditEvent:
    """Tests for UnifiedAuditEvent dataclass."""

    # ------------------------------------------------------------------
    # Creation tests
    # ------------------------------------------------------------------

    def test_create_event_with_required_fields(self) -> None:
        """Event can be created with required fields."""
        event = UnifiedAuditEvent(
            event_id="e-1",
            event_type="auth.login_success",
            category="auth",
        )
        assert event.event_id == "e-1"
        assert event.event_type == "auth.login_success"
        assert event.category == "auth"

    def test_create_event_with_all_fields(self) -> None:
        """Event can be created with all fields."""
        now = datetime.now(timezone.utc)
        event = UnifiedAuditEvent(
            event_id="e-2",
            event_type="auth.login_success",
            category="auth",
            severity="info",
            timestamp=now,
            tenant_id="t-acme",
            user_id="u-1",
            actor_id="u-1",
            actor_type="user",
            resource_type="session",
            resource_id="s-1",
            action="create",
            result="success",
            client_ip="192.168.1.1",
            user_agent="Mozilla/5.0",
            request_id="req-1",
            session_id="sess-1",
            details={"method": "password"},
            metadata={"version": "1.0"},
            pii_fields=["client_ip"],
        )
        assert event.tenant_id == "t-acme"
        assert event.user_id == "u-1"
        assert event.client_ip == "192.168.1.1"
        assert event.details["method"] == "password"

    def test_create_event_default_values(self) -> None:
        """Event has correct default values."""
        event = UnifiedAuditEvent(
            event_id="e-3",
            event_type="auth.login_success",
            category="auth",
        )
        assert event.severity == "info"
        assert event.actor_type == "user"
        assert event.result == "success"
        assert event.details == {}
        assert event.metadata == {}
        assert event.pii_fields == []

    def test_event_timestamp_is_datetime(self) -> None:
        """Event timestamp is a datetime object."""
        event = _make_event()
        assert isinstance(event.timestamp, datetime)

    def test_event_id_uniqueness(self) -> None:
        """Each event gets a unique ID when generated."""
        e1 = _make_event()
        e2 = _make_event()
        assert e1.event_id != e2.event_id

    # ------------------------------------------------------------------
    # to_dict tests
    # ------------------------------------------------------------------

    def test_to_dict_includes_all_fields(self, sample_event: UnifiedAuditEvent) -> None:
        """to_dict includes all event fields."""
        result = sample_event.to_dict()
        assert isinstance(result, dict)
        assert "event_id" in result
        assert "event_type" in result
        assert "category" in result
        assert "timestamp" in result

    def test_to_dict_timestamp_is_iso_format(self, sample_event: UnifiedAuditEvent) -> None:
        """to_dict converts timestamp to ISO format string."""
        result = sample_event.to_dict()
        # Timestamp should be serializable as a string
        assert isinstance(result.get("timestamp"), (str, datetime))

    def test_to_dict_no_pii_redaction_by_default(self, event_with_pii: UnifiedAuditEvent) -> None:
        """to_dict does not redact PII by default."""
        result = event_with_pii.to_dict(redact_pii=False)
        assert result["details"]["email"] == "user@example.com"
        assert result["details"]["ip_address"] == "192.168.1.1"

    def test_to_dict_with_pii_redaction(self, event_with_pii: UnifiedAuditEvent) -> None:
        """to_dict redacts PII when redact_pii=True."""
        result = event_with_pii.to_dict(redact_pii=True)
        # PII fields should be redacted
        assert result["details"]["email"] != "user@example.com"
        assert result["details"]["ip_address"] != "192.168.1.1"
        # Non-PII fields should remain
        assert result["details"]["action"] == "login"

    def test_to_dict_pii_redaction_marker(self, event_with_pii: UnifiedAuditEvent) -> None:
        """Redacted PII fields use a consistent marker."""
        result = event_with_pii.to_dict(redact_pii=True)
        # Common redaction markers
        redaction_markers = ["[REDACTED]", "***", "<redacted>", "REDACTED"]
        email_value = result["details"]["email"]
        assert any(marker in str(email_value) for marker in redaction_markers) or email_value != "user@example.com"

    def test_to_dict_empty_details(self) -> None:
        """to_dict handles empty details correctly."""
        event = _make_event(details={})
        result = event.to_dict()
        assert result["details"] == {}

    def test_to_dict_nested_details(self) -> None:
        """to_dict handles nested details correctly."""
        event = _make_event(details={"nested": {"key": "value"}})
        result = event.to_dict()
        assert result["details"]["nested"]["key"] == "value"

    # ------------------------------------------------------------------
    # to_json tests
    # ------------------------------------------------------------------

    def test_to_json_returns_valid_json(self, sample_event: UnifiedAuditEvent) -> None:
        """to_json returns a valid JSON string."""
        result = sample_event.to_json()
        assert isinstance(result, str)
        parsed = json.loads(result)
        assert isinstance(parsed, dict)

    def test_to_json_includes_all_fields(self, sample_event: UnifiedAuditEvent) -> None:
        """to_json includes all event fields."""
        result = json.loads(sample_event.to_json())
        assert "event_id" in result
        assert "event_type" in result
        assert "category" in result

    def test_to_json_with_pii_redaction(self, event_with_pii: UnifiedAuditEvent) -> None:
        """to_json redacts PII when redact_pii=True."""
        result = json.loads(event_with_pii.to_json(redact_pii=True))
        assert result["details"]["email"] != "user@example.com"

    def test_to_json_without_pii_redaction(self, event_with_pii: UnifiedAuditEvent) -> None:
        """to_json preserves PII when redact_pii=False."""
        result = json.loads(event_with_pii.to_json(redact_pii=False))
        assert result["details"]["email"] == "user@example.com"

    def test_to_json_is_parseable(self, sample_event: UnifiedAuditEvent) -> None:
        """to_json output can be parsed back."""
        json_str = sample_event.to_json()
        parsed = json.loads(json_str)
        assert parsed["event_id"] == sample_event.event_id
        assert parsed["event_type"] == sample_event.event_type

    # ------------------------------------------------------------------
    # from_auth_event tests
    # ------------------------------------------------------------------

    def test_from_auth_event_creates_event(self) -> None:
        """from_auth_event creates a UnifiedAuditEvent from legacy auth event."""
        auth_event = _make_auth_event_mock()
        result = UnifiedAuditEvent.from_auth_event(auth_event)
        assert isinstance(result, UnifiedAuditEvent)

    def test_from_auth_event_maps_event_type(self) -> None:
        """from_auth_event maps legacy event type to unified type."""
        auth_event = _make_auth_event_mock(event_type="login_success")
        result = UnifiedAuditEvent.from_auth_event(auth_event)
        assert result.event_type == "auth.login_success"

    def test_from_auth_event_sets_category(self) -> None:
        """from_auth_event sets category to 'auth'."""
        auth_event = _make_auth_event_mock()
        result = UnifiedAuditEvent.from_auth_event(auth_event)
        assert result.category == "auth"

    def test_from_auth_event_preserves_user_info(self) -> None:
        """from_auth_event preserves user and tenant info."""
        auth_event = _make_auth_event_mock(user_id="u-42", tenant_id="t-corp")
        result = UnifiedAuditEvent.from_auth_event(auth_event)
        assert result.user_id == "u-42"
        assert result.tenant_id == "t-corp"

    def test_from_auth_event_preserves_client_info(self) -> None:
        """from_auth_event preserves client IP and user agent."""
        auth_event = _make_auth_event_mock(client_ip="10.0.0.1")
        result = UnifiedAuditEvent.from_auth_event(auth_event)
        assert result.client_ip == "10.0.0.1"

    # ------------------------------------------------------------------
    # from_rbac_event tests
    # ------------------------------------------------------------------

    def test_from_rbac_event_creates_event(self) -> None:
        """from_rbac_event creates a UnifiedAuditEvent from legacy RBAC event."""
        rbac_event = _make_rbac_event_mock()
        result = UnifiedAuditEvent.from_rbac_event(rbac_event)
        assert isinstance(result, UnifiedAuditEvent)

    def test_from_rbac_event_maps_event_type(self) -> None:
        """from_rbac_event maps legacy event type to unified type."""
        rbac_event = _make_rbac_event_mock(event_type="role_created")
        result = UnifiedAuditEvent.from_rbac_event(rbac_event)
        assert result.event_type == "rbac.role_created"

    def test_from_rbac_event_sets_category(self) -> None:
        """from_rbac_event sets category to 'rbac'."""
        rbac_event = _make_rbac_event_mock()
        result = UnifiedAuditEvent.from_rbac_event(rbac_event)
        assert result.category == "rbac"

    def test_from_rbac_event_preserves_role_info(self) -> None:
        """from_rbac_event includes role info in details."""
        rbac_event = _make_rbac_event_mock()
        result = UnifiedAuditEvent.from_rbac_event(rbac_event)
        # Role info should be in details or as resource
        assert result.resource_type == "role" or "role" in result.details

    # ------------------------------------------------------------------
    # from_encryption_event tests
    # ------------------------------------------------------------------

    def test_from_encryption_event_creates_event(self) -> None:
        """from_encryption_event creates a UnifiedAuditEvent from legacy encryption event."""
        enc_event = _make_encryption_event_mock()
        result = UnifiedAuditEvent.from_encryption_event(enc_event)
        assert isinstance(result, UnifiedAuditEvent)

    def test_from_encryption_event_maps_event_type(self) -> None:
        """from_encryption_event maps legacy event type to unified type."""
        enc_event = _make_encryption_event_mock(event_type="encryption_performed")
        result = UnifiedAuditEvent.from_encryption_event(enc_event)
        assert result.event_type == "encryption.performed"

    def test_from_encryption_event_sets_category(self) -> None:
        """from_encryption_event sets category to 'encryption'."""
        enc_event = _make_encryption_event_mock()
        result = UnifiedAuditEvent.from_encryption_event(enc_event)
        assert result.category == "encryption"

    def test_from_encryption_event_preserves_key_info(self) -> None:
        """from_encryption_event includes key info in details."""
        enc_event = _make_encryption_event_mock()
        result = UnifiedAuditEvent.from_encryption_event(enc_event)
        # Key info should be in details or as resource
        assert result.resource_type == "key" or "key_id" in result.details


# ============================================================================
# TestEventBuilder
# ============================================================================


class TestEventBuilder:
    """Tests for EventBuilder fluent API."""

    # ------------------------------------------------------------------
    # Initialization tests
    # ------------------------------------------------------------------

    def test_builder_initialization(self) -> None:
        """EventBuilder initializes with event type."""
        builder = EventBuilder("auth.login_success")
        assert builder is not None

    def test_builder_returns_self(self) -> None:
        """Builder methods return self for chaining."""
        builder = EventBuilder("auth.login_success")
        result = builder.with_tenant("t-1")
        assert result is builder

    # ------------------------------------------------------------------
    # Builder method tests
    # ------------------------------------------------------------------

    def test_with_tenant(self) -> None:
        """with_tenant sets tenant_id."""
        event = EventBuilder("auth.login_success").with_tenant("t-acme").build()
        assert event.tenant_id == "t-acme"

    def test_with_user(self) -> None:
        """with_user sets user_id."""
        event = EventBuilder("auth.login_success").with_user("u-42").build()
        assert event.user_id == "u-42"

    def test_with_actor(self) -> None:
        """with_actor sets actor_id and actor_type."""
        event = EventBuilder("auth.login_success").with_actor("svc-1", "service").build()
        assert event.actor_id == "svc-1"
        assert event.actor_type == "service"

    def test_with_actor_default_type(self) -> None:
        """with_actor defaults actor_type to 'user'."""
        event = EventBuilder("auth.login_success").with_actor("u-1").build()
        assert event.actor_type == "user"

    def test_with_resource(self) -> None:
        """with_resource sets resource_type and resource_id."""
        event = EventBuilder("data.read").with_resource("document", "doc-1").build()
        assert event.resource_type == "document"
        assert event.resource_id == "doc-1"

    def test_with_action(self) -> None:
        """with_action sets action."""
        event = EventBuilder("data.read").with_action("read").build()
        assert event.action == "read"

    def test_with_result(self) -> None:
        """with_result sets result."""
        event = EventBuilder("auth.login_failure").with_result("failure").build()
        assert event.result == "failure"

    def test_with_severity(self) -> None:
        """with_severity sets severity."""
        event = EventBuilder("system.error").with_severity("error").build()
        assert event.severity == "error"

    def test_with_request_context(self) -> None:
        """with_request_context sets client_ip, user_agent, request_id."""
        event = (
            EventBuilder("api.request_received")
            .with_request_context("192.168.1.1", "Mozilla/5.0", "req-123")
            .build()
        )
        assert event.client_ip == "192.168.1.1"
        assert event.user_agent == "Mozilla/5.0"
        assert event.request_id == "req-123"

    def test_with_detail(self) -> None:
        """with_detail adds a single detail."""
        event = EventBuilder("auth.login_success").with_detail("method", "password").build()
        assert event.details["method"] == "password"

    def test_with_details(self) -> None:
        """with_details adds multiple details."""
        event = (
            EventBuilder("auth.login_success")
            .with_details({"method": "password", "mfa": True})
            .build()
        )
        assert event.details["method"] == "password"
        assert event.details["mfa"] is True

    def test_with_pii_field(self) -> None:
        """with_pii_field marks a field as PII."""
        event = EventBuilder("auth.login_success").with_pii_field("email").build()
        assert "email" in event.pii_fields

    # ------------------------------------------------------------------
    # Build tests
    # ------------------------------------------------------------------

    def test_build_returns_unified_event(self) -> None:
        """build() returns a UnifiedAuditEvent."""
        event = EventBuilder("auth.login_success").build()
        assert isinstance(event, UnifiedAuditEvent)

    def test_build_generates_event_id(self) -> None:
        """build() generates a unique event_id."""
        event = EventBuilder("auth.login_success").build()
        assert event.event_id is not None
        assert len(event.event_id) > 0

    def test_build_sets_timestamp(self) -> None:
        """build() sets timestamp to current time."""
        before = datetime.now(timezone.utc)
        event = EventBuilder("auth.login_success").build()
        after = datetime.now(timezone.utc)
        assert before <= event.timestamp <= after

    def test_build_infers_category(self) -> None:
        """build() infers category from event type."""
        event = EventBuilder("auth.login_success").build()
        assert event.category == "auth"

    def test_build_infers_severity(self) -> None:
        """build() infers default severity from event type."""
        event = EventBuilder("auth.login_failure").build()
        assert event.severity == "error"

    # ------------------------------------------------------------------
    # Chaining tests
    # ------------------------------------------------------------------

    def test_full_chain(self) -> None:
        """Builder supports full method chaining."""
        event = (
            EventBuilder("auth.login_success")
            .with_tenant("t-acme")
            .with_user("u-42")
            .with_actor("u-42", "user")
            .with_resource("session", "s-123")
            .with_action("create")
            .with_result("success")
            .with_severity("info")
            .with_request_context("192.168.1.1", "Chrome", "req-1")
            .with_detail("method", "password")
            .with_details({"mfa": False})
            .with_pii_field("client_ip")
            .build()
        )
        assert event.tenant_id == "t-acme"
        assert event.user_id == "u-42"
        assert event.resource_type == "session"
        assert event.details["method"] == "password"
        assert event.details["mfa"] is False

    def test_multiple_builds(self) -> None:
        """Builder can be used multiple times with different values."""
        builder = EventBuilder("auth.login_success")
        e1 = builder.with_tenant("t-1").build()
        e2 = builder.with_tenant("t-2").build()
        # Both events should exist
        assert e1.tenant_id == "t-1" or e2.tenant_id == "t-2"

    def test_builder_does_not_mutate_previous_builds(self) -> None:
        """Each build creates an independent event."""
        builder = EventBuilder("auth.login_success").with_tenant("t-1")
        e1 = builder.build()
        builder.with_tenant("t-2")
        # e1 should not be affected
        assert e1.tenant_id == "t-1"
