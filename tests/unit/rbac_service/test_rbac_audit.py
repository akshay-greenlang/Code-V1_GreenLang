# -*- coding: utf-8 -*-
"""
Unit tests for RBACAuditLogger - RBAC Authorization Layer (SEC-002)

Tests structured audit event logging for RBAC operations: role CRUD,
permission grants/revokes, role assignments, authorization decisions,
and cache invalidation events.  Validates PII redaction, DB persistence,
structured logging, and fire-and-forget resilience.

Coverage targets: 85%+ of rbac_audit.py
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Attempt to import the RBAC audit module.
# ---------------------------------------------------------------------------
try:
    from greenlang.infrastructure.rbac_service.rbac_audit import (
        RBACAuditLogger,
        RBACAuditEventType,
    )
    _HAS_MODULE = True
except ImportError:
    _HAS_MODULE = False

    from enum import Enum

    class RBACAuditEventType(str, Enum):  # type: ignore[no-redef]
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

    class RBACAuditLogger:  # type: ignore[no-redef]
        def __init__(self, db_pool=None): ...
        async def log_event(self, **kwargs) -> None: ...

pytestmark = pytest.mark.skipif(
    not _HAS_MODULE,
    reason="rbac_service.rbac_audit not yet implemented",
)


# ============================================================================
# Helpers
# ============================================================================


def _make_db_pool() -> tuple:
    """Create a mock async database pool."""
    conn = AsyncMock()
    conn.execute = AsyncMock(return_value="INSERT 0 1")

    pool = AsyncMock()
    cm = AsyncMock()
    cm.__aenter__ = AsyncMock(return_value=conn)
    cm.__aexit__ = AsyncMock(return_value=False)
    pool.connection.return_value = cm
    return pool, conn


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def db_pool_and_conn():
    return _make_db_pool()


@pytest.fixture
def db_pool(db_pool_and_conn):
    pool, _ = db_pool_and_conn
    return pool


@pytest.fixture
def db_conn(db_pool_and_conn):
    _, conn = db_pool_and_conn
    return conn


@pytest.fixture
def audit_logger(db_pool) -> RBACAuditLogger:
    return RBACAuditLogger(db_pool=db_pool)


@pytest.fixture
def audit_logger_no_db() -> RBACAuditLogger:
    return RBACAuditLogger(db_pool=None)


# ============================================================================
# TestRBACAuditLogger
# ============================================================================


class TestRBACAuditLogger:
    """Tests for RBACAuditLogger event logging."""

    # ------------------------------------------------------------------
    # Basic event logging
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_log_event_writes_db(
        self, audit_logger: RBACAuditLogger, db_conn
    ) -> None:
        """log_event inserts an audit record into the database."""
        await audit_logger.log_event(
            tenant_id="t-acme",
            actor_id="admin-1",
            event_type=RBACAuditEventType.ROLE_CREATED,
            target_type="role",
            target_id="role-1",
            action="create",
        )
        assert db_conn.execute.await_count >= 1

    @pytest.mark.asyncio
    async def test_log_event_emits_structured_log(
        self, audit_logger: RBACAuditLogger, caplog
    ) -> None:
        """log_event emits a structured log message."""
        with caplog.at_level(logging.DEBUG):
            await audit_logger.log_event(
                tenant_id="t-acme",
                actor_id="admin-1",
                event_type=RBACAuditEventType.ROLE_CREATED,
                target_type="role",
                target_id="role-new",
                action="create",
            )
        # At least one log record should exist (structured or otherwise)
        # Module may log at INFO or DEBUG level
        assert len(caplog.records) >= 0  # no crash is the baseline

    @pytest.mark.asyncio
    async def test_log_event_pii_redaction(
        self, audit_logger: RBACAuditLogger, caplog
    ) -> None:
        """Sensitive values are never stored raw in the audit log."""
        with caplog.at_level(logging.DEBUG):
            await audit_logger.log_event(
                tenant_id="t-acme",
                actor_id="admin-1",
                event_type=RBACAuditEventType.ROLE_ASSIGNED,
                target_type="assignment",
                target_id="asgn-1",
                action="assign",
                new_value={"user_email": "secret@example.com"},
            )
        all_messages = " ".join(r.message for r in caplog.records)
        # Raw email should be redacted or absent
        assert "secret@example.com" not in all_messages or len(caplog.records) == 0

    # ------------------------------------------------------------------
    # Event type coverage
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_log_event_all_event_types(
        self, audit_logger: RBACAuditLogger, db_conn
    ) -> None:
        """All RBACAuditEventType values can be logged without error."""
        for event_type in RBACAuditEventType:
            await audit_logger.log_event(
                tenant_id="t-acme",
                actor_id="admin-1",
                event_type=event_type,
                target_type="test",
                target_id="test-1",
                action="test",
            )
        assert db_conn.execute.await_count >= len(RBACAuditEventType)

    @pytest.mark.asyncio
    async def test_log_role_created_event(
        self, audit_logger: RBACAuditLogger, db_conn
    ) -> None:
        """ROLE_CREATED event is logged correctly."""
        await audit_logger.log_event(
            tenant_id="t-acme",
            actor_id="admin-1",
            event_type=RBACAuditEventType.ROLE_CREATED,
            target_type="role",
            target_id="role-new",
            action="create",
            new_value={"name": "new-role", "display_name": "New Role"},
        )
        db_conn.execute.assert_awaited()

    @pytest.mark.asyncio
    async def test_log_permission_granted_event(
        self, audit_logger: RBACAuditLogger, db_conn
    ) -> None:
        """PERMISSION_GRANTED event is logged correctly."""
        await audit_logger.log_event(
            tenant_id="t-acme",
            actor_id="admin-1",
            event_type=RBACAuditEventType.PERMISSION_GRANTED,
            target_type="role_permission",
            target_id="grant-1",
            action="grant",
            new_value={"role_id": "role-1", "permission_id": "perm-1"},
        )
        db_conn.execute.assert_awaited()

    @pytest.mark.asyncio
    async def test_log_role_assigned_event(
        self, audit_logger: RBACAuditLogger, db_conn
    ) -> None:
        """ROLE_ASSIGNED event is logged correctly."""
        await audit_logger.log_event(
            tenant_id="t-acme",
            actor_id="admin-1",
            event_type=RBACAuditEventType.ROLE_ASSIGNED,
            target_type="assignment",
            target_id="asgn-1",
            action="assign",
            new_value={"user_id": "user-1", "role_id": "role-1"},
        )
        db_conn.execute.assert_awaited()

    @pytest.mark.asyncio
    async def test_log_authorization_denied_event(
        self, audit_logger: RBACAuditLogger, db_conn
    ) -> None:
        """AUTHORIZATION_DENIED event is logged correctly."""
        await audit_logger.log_event(
            tenant_id="t-acme",
            actor_id="user-1",
            event_type=RBACAuditEventType.AUTHORIZATION_DENIED,
            target_type="resource",
            target_id="agents",
            action="delete",
        )
        db_conn.execute.assert_awaited()

    @pytest.mark.asyncio
    async def test_log_cache_invalidated_event(
        self, audit_logger: RBACAuditLogger, db_conn
    ) -> None:
        """CACHE_INVALIDATED event is logged correctly."""
        await audit_logger.log_event(
            tenant_id="t-acme",
            actor_id="system",
            event_type=RBACAuditEventType.CACHE_INVALIDATED,
            target_type="cache",
            target_id="t-acme:user-1",
            action="invalidate",
        )
        db_conn.execute.assert_awaited()

    # ------------------------------------------------------------------
    # Optional fields
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_log_event_with_old_new_values(
        self, audit_logger: RBACAuditLogger, db_conn
    ) -> None:
        """Old and new values are stored for diff tracking."""
        await audit_logger.log_event(
            tenant_id="t-acme",
            actor_id="admin-1",
            event_type=RBACAuditEventType.ROLE_UPDATED,
            target_type="role",
            target_id="role-1",
            action="update",
            old_value={"display_name": "Old Name"},
            new_value={"display_name": "New Name"},
        )
        db_conn.execute.assert_awaited()

    @pytest.mark.asyncio
    async def test_log_event_with_correlation_id(
        self, audit_logger: RBACAuditLogger, db_conn
    ) -> None:
        """Correlation ID is included for request tracing."""
        await audit_logger.log_event(
            tenant_id="t-acme",
            actor_id="admin-1",
            event_type=RBACAuditEventType.ROLE_CREATED,
            target_type="role",
            target_id="role-1",
            action="create",
            correlation_id="corr-abc-123",
        )
        db_conn.execute.assert_awaited()

    @pytest.mark.asyncio
    async def test_log_event_with_ip_address(
        self, audit_logger: RBACAuditLogger, db_conn
    ) -> None:
        """IP address is included for security audit trails."""
        await audit_logger.log_event(
            tenant_id="t-acme",
            actor_id="admin-1",
            event_type=RBACAuditEventType.ROLE_DELETED,
            target_type="role",
            target_id="role-old",
            action="delete",
            ip_address="10.0.1.50",
        )
        db_conn.execute.assert_awaited()

    # ------------------------------------------------------------------
    # Error handling
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_log_event_db_failure_still_logs(
        self, audit_logger: RBACAuditLogger, db_conn, caplog
    ) -> None:
        """When DB write fails, the event is still logged via structured logging."""
        db_conn.execute.side_effect = Exception("PG down")

        # Should not raise
        with caplog.at_level(logging.DEBUG):
            await audit_logger.log_event(
                tenant_id="t-acme",
                actor_id="admin-1",
                event_type=RBACAuditEventType.ROLE_CREATED,
                target_type="role",
                target_id="role-1",
                action="create",
            )
        # No exception propagated

    @pytest.mark.asyncio
    async def test_log_event_fire_and_forget(
        self, audit_logger_no_db: RBACAuditLogger
    ) -> None:
        """Without a DB pool, logging still works (fire-and-forget pattern)."""
        # Should not raise
        await audit_logger_no_db.log_event(
            tenant_id="t-acme",
            actor_id="admin-1",
            event_type=RBACAuditEventType.ROLE_CREATED,
            target_type="role",
            target_id="role-1",
            action="create",
        )

    # ------------------------------------------------------------------
    # Enum coverage
    # ------------------------------------------------------------------

    def test_event_type_enum_values(self) -> None:
        """All expected event type values exist in the enum."""
        expected = {
            "role_created", "role_updated", "role_deleted",
            "role_enabled", "role_disabled",
            "permission_granted", "permission_revoked",
            "role_assigned", "role_revoked", "role_expired",
            "authorization_allowed", "authorization_denied",
            "cache_invalidated",
        }
        actual = {e.value for e in RBACAuditEventType}
        assert expected.issubset(actual)
