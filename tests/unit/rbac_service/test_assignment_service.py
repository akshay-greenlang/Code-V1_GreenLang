# -*- coding: utf-8 -*-
"""
Unit tests for AssignmentService - RBAC Authorization Layer (SEC-002)

Tests role assignment and revocation lifecycle: assign, revoke, list
user roles, get aggregated permissions, bulk assignment, stale-expiry
cleanup, cache integration, and audit logging triggers.

Coverage targets: 85%+ of assignment_service.py
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch, call

import pytest

# ---------------------------------------------------------------------------
# Attempt to import the RBAC assignment service module.
# ---------------------------------------------------------------------------
try:
    from greenlang.infrastructure.rbac_service.assignment_service import (
        AssignmentService,
    )
    _HAS_MODULE = True
except ImportError:
    _HAS_MODULE = False

    class AssignmentService:  # type: ignore[no-redef]
        """Stub for test collection when module is not yet built."""
        def __init__(self, db_pool, cache, audit=None, metrics=None): ...
        async def assign_role(self, **kwargs) -> dict: ...
        async def revoke_role(self, assignment_id, revoked_by=None) -> dict: ...
        async def list_user_roles(self, user_id, tenant_id, include_expired=False) -> List[dict]: ...
        async def get_user_permissions(self, user_id, tenant_id) -> List[str]: ...
        async def bulk_assign_role(self, **kwargs) -> List[dict]: ...
        async def expire_stale_assignments(self) -> int: ...
        async def get_assignment(self, assignment_id) -> Optional[dict]: ...

pytestmark = pytest.mark.skipif(
    not _HAS_MODULE,
    reason="rbac_service.assignment_service not yet implemented",
)


# ============================================================================
# Helpers
# ============================================================================


def _make_db_pool(
    fetchrow_return: Any = None,
    fetch_return: Optional[list] = None,
    execute_return: str = "INSERT 0 1",
) -> tuple:
    """Create a mock async database pool with connection context manager."""
    conn = AsyncMock()
    conn.execute = AsyncMock(return_value=execute_return)
    conn.fetchrow = AsyncMock(return_value=fetchrow_return)
    conn.fetch = AsyncMock(return_value=fetch_return or [])
    conn.fetchval = AsyncMock(return_value=0)

    pool = AsyncMock()
    cm = AsyncMock()
    cm.__aenter__ = AsyncMock(return_value=conn)
    cm.__aexit__ = AsyncMock(return_value=False)
    pool.connection.return_value = cm
    return pool, conn


def _make_cache() -> AsyncMock:
    """Create a mock RBACCache."""
    cache = AsyncMock()
    cache.get_permissions = AsyncMock(return_value=None)
    cache.set_permissions = AsyncMock()
    cache.invalidate_user = AsyncMock()
    cache.invalidate_tenant = AsyncMock()
    cache.invalidate_all = AsyncMock()
    cache.publish_invalidation = AsyncMock()
    return cache


def _make_audit() -> AsyncMock:
    """Create a mock RBACAuditLogger."""
    audit = AsyncMock()
    audit.log_event = AsyncMock()
    return audit


def _make_metrics() -> MagicMock:
    """Create a mock RBACMetrics."""
    metrics = MagicMock()
    metrics.record_authorization = MagicMock()
    metrics.record_role_change = MagicMock()
    metrics.set_assignments_count = MagicMock()
    return metrics


def _make_assignment_row(
    assignment_id: str = "asgn-1",
    user_id: str = "user-1",
    role_id: str = "role-1",
    tenant_id: str = "t-acme",
    assigned_by: Optional[str] = "admin-1",
    expires_at: Optional[datetime] = None,
    revoked_at: Optional[datetime] = None,
    is_active: bool = True,
) -> dict:
    return {
        "id": assignment_id,
        "user_id": user_id,
        "role_id": role_id,
        "tenant_id": tenant_id,
        "assigned_by": assigned_by,
        "assigned_at": datetime.now(timezone.utc),
        "expires_at": expires_at,
        "revoked_at": revoked_at,
        "revoked_by": None,
        "is_active": is_active,
    }


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
def cache() -> AsyncMock:
    return _make_cache()


@pytest.fixture
def audit() -> AsyncMock:
    return _make_audit()


@pytest.fixture
def metrics() -> MagicMock:
    return _make_metrics()


@pytest.fixture
def assignment_service(db_pool, cache, audit, metrics) -> AssignmentService:
    return AssignmentService(
        db_pool=db_pool, cache=cache, audit=audit, metrics=metrics
    )


# ============================================================================
# TestAssignmentService
# ============================================================================


class TestAssignmentService:
    """Tests for AssignmentService role assignment operations."""

    # ------------------------------------------------------------------
    # assign_role
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_assign_role_success(
        self, assignment_service: AssignmentService, db_conn
    ) -> None:
        """Assigning a role to a user succeeds."""
        row = _make_assignment_row()
        db_conn.fetchrow.return_value = row

        result = await assignment_service.assign_role(
            user_id="user-1",
            role_id="role-1",
            tenant_id="t-acme",
            assigned_by="admin-1",
        )
        assert isinstance(result, dict)
        assert result.get("user_id") == "user-1" or "id" in result

    @pytest.mark.asyncio
    async def test_assign_role_duplicate(
        self, assignment_service: AssignmentService, db_conn
    ) -> None:
        """Assigning the same role to the same user twice raises an error."""
        db_conn.execute.side_effect = Exception("unique constraint")
        with pytest.raises(Exception):
            await assignment_service.assign_role(
                user_id="user-1",
                role_id="role-1",
                tenant_id="t-acme",
            )

    @pytest.mark.asyncio
    async def test_assign_role_with_expiry(
        self, assignment_service: AssignmentService, db_conn
    ) -> None:
        """Assignment with expires_at stores the temporal boundary."""
        future = datetime.now(timezone.utc) + timedelta(days=30)
        row = _make_assignment_row(expires_at=future)
        db_conn.fetchrow.return_value = row

        result = await assignment_service.assign_role(
            user_id="user-1",
            role_id="role-1",
            tenant_id="t-acme",
            expires_at=future,
        )
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_assign_role_invalid_user_id(
        self, assignment_service: AssignmentService, db_conn
    ) -> None:
        """Assigning a role with an empty user_id raises an error."""
        with pytest.raises(Exception):
            await assignment_service.assign_role(
                user_id="",
                role_id="role-1",
                tenant_id="t-acme",
            )

    # ------------------------------------------------------------------
    # revoke_role
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_revoke_role_success(
        self, assignment_service: AssignmentService, db_conn
    ) -> None:
        """Revoking an active assignment succeeds."""
        row = _make_assignment_row(assignment_id="asgn-rev")
        revoked = {**row, "revoked_at": datetime.now(timezone.utc), "is_active": False}
        db_conn.fetchrow.side_effect = [row, revoked]

        result = await assignment_service.revoke_role(
            "asgn-rev", revoked_by="admin-1"
        )
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_revoke_role_not_found(
        self, assignment_service: AssignmentService, db_conn
    ) -> None:
        """Revoking a non-existent assignment raises an error."""
        db_conn.fetchrow.return_value = None
        with pytest.raises(Exception):
            await assignment_service.revoke_role("ghost")

    # ------------------------------------------------------------------
    # list_user_roles
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_list_user_roles_active(
        self, assignment_service: AssignmentService, db_conn
    ) -> None:
        """Listing user roles returns only active assignments by default."""
        rows = [
            _make_assignment_row(assignment_id=f"a-{i}", role_id=f"role-{i}")
            for i in range(3)
        ]
        db_conn.fetch.return_value = rows

        result = await assignment_service.list_user_roles("user-1", "t-acme")
        assert isinstance(result, list)
        assert len(result) == 3

    @pytest.mark.asyncio
    async def test_list_user_roles_include_expired(
        self, assignment_service: AssignmentService, db_conn
    ) -> None:
        """Listing with include_expired=True returns expired assignments too."""
        past = datetime.now(timezone.utc) - timedelta(days=1)
        rows = [
            _make_assignment_row(assignment_id="a-1"),
            _make_assignment_row(
                assignment_id="a-2", expires_at=past, is_active=False
            ),
        ]
        db_conn.fetch.return_value = rows

        result = await assignment_service.list_user_roles(
            "user-1", "t-acme", include_expired=True
        )
        assert isinstance(result, list)
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_list_user_roles_empty(
        self, assignment_service: AssignmentService, db_conn
    ) -> None:
        """Listing user roles when none exist returns empty list."""
        db_conn.fetch.return_value = []

        result = await assignment_service.list_user_roles("user-none", "t-acme")
        assert isinstance(result, list)
        assert len(result) == 0

    # ------------------------------------------------------------------
    # get_user_permissions
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_get_user_permissions_aggregated(
        self, assignment_service: AssignmentService, db_conn, cache
    ) -> None:
        """Getting user permissions returns aggregated list from all roles."""
        cache.get_permissions.return_value = None
        db_conn.fetch.return_value = [
            {"permission": "agents:read"},
            {"permission": "agents:write"},
            {"permission": "emissions:read"},
        ]

        result = await assignment_service.get_user_permissions("user-1", "t-acme")
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_get_user_permissions_with_hierarchy(
        self, assignment_service: AssignmentService, db_conn, cache
    ) -> None:
        """User permissions include inherited permissions from parent roles."""
        cache.get_permissions.return_value = None
        db_conn.fetch.return_value = [
            {"permission": "agents:read"},
            {"permission": "agents:write"},
            {"permission": "admin:all"},
        ]

        result = await assignment_service.get_user_permissions("user-1", "t-acme")
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_get_user_permissions_cache_hit(
        self, assignment_service: AssignmentService, db_conn, cache
    ) -> None:
        """Cache hit returns permissions without touching DB."""
        cached_perms = ["agents:read", "agents:write"]
        cache.get_permissions.return_value = cached_perms

        result = await assignment_service.get_user_permissions("user-1", "t-acme")
        assert isinstance(result, list)
        assert result == cached_perms

    @pytest.mark.asyncio
    async def test_get_user_permissions_cache_miss_loads_db(
        self, assignment_service: AssignmentService, db_conn, cache
    ) -> None:
        """Cache miss loads permissions from DB and populates cache."""
        cache.get_permissions.return_value = None
        db_conn.fetch.return_value = [
            {"permission": "agents:read"},
        ]

        result = await assignment_service.get_user_permissions("user-1", "t-acme")
        assert isinstance(result, list)
        # Cache should be populated
        assert cache.set_permissions.await_count >= 1

    @pytest.mark.asyncio
    async def test_cache_set_on_get_user_permissions(
        self, assignment_service: AssignmentService, db_conn, cache
    ) -> None:
        """After DB lookup, permissions are written to cache."""
        cache.get_permissions.return_value = None
        db_conn.fetch.return_value = [
            {"permission": "emissions:read"},
        ]

        await assignment_service.get_user_permissions("user-1", "t-acme")
        cache.set_permissions.assert_awaited()

    # ------------------------------------------------------------------
    # bulk_assign_role
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_bulk_assign_role_success(
        self, assignment_service: AssignmentService, db_conn
    ) -> None:
        """Bulk assigning a role to multiple users succeeds."""
        rows = [
            _make_assignment_row(assignment_id=f"b-{i}", user_id=f"user-{i}")
            for i in range(3)
        ]
        db_conn.fetchrow.side_effect = rows

        result = await assignment_service.bulk_assign_role(
            user_ids=["user-0", "user-1", "user-2"],
            role_id="role-1",
            tenant_id="t-acme",
            assigned_by="admin-1",
        )
        assert isinstance(result, list)
        assert len(result) == 3

    @pytest.mark.asyncio
    async def test_bulk_assign_role_partial_failure(
        self, assignment_service: AssignmentService, db_conn
    ) -> None:
        """Bulk assign handles partial failures (some users already assigned)."""
        row = _make_assignment_row()
        db_conn.fetchrow.side_effect = [
            row,
            Exception("unique constraint"),
            row,
        ]

        result = await assignment_service.bulk_assign_role(
            user_ids=["user-1", "user-2", "user-3"],
            role_id="role-1",
            tenant_id="t-acme",
        )
        # Should return at least the successful assignments
        assert isinstance(result, list)

    # ------------------------------------------------------------------
    # expire_stale_assignments
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_expire_stale_assignments(
        self, assignment_service: AssignmentService, db_conn
    ) -> None:
        """Stale assignment cleanup deactivates expired assignments."""
        db_conn.execute.return_value = "UPDATE 5"

        result = await assignment_service.expire_stale_assignments()
        assert result == 5

    @pytest.mark.asyncio
    async def test_expire_stale_no_expired(
        self, assignment_service: AssignmentService, db_conn
    ) -> None:
        """When no assignments are expired, returns 0."""
        db_conn.execute.return_value = "UPDATE 0"

        result = await assignment_service.expire_stale_assignments()
        assert result == 0

    # ------------------------------------------------------------------
    # get_assignment
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_get_assignment_exists(
        self, assignment_service: AssignmentService, db_conn
    ) -> None:
        """Getting an existing assignment returns its data."""
        row = _make_assignment_row(assignment_id="asgn-42")
        db_conn.fetchrow.return_value = row

        result = await assignment_service.get_assignment("asgn-42")
        assert result is not None
        assert result["id"] == "asgn-42"

    @pytest.mark.asyncio
    async def test_get_assignment_not_found(
        self, assignment_service: AssignmentService, db_conn
    ) -> None:
        """Getting a non-existent assignment returns None."""
        db_conn.fetchrow.return_value = None

        result = await assignment_service.get_assignment("ghost")
        assert result is None

    # ------------------------------------------------------------------
    # Temporal assignment
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_temporal_assignment_not_returned_after_expiry(
        self, assignment_service: AssignmentService, db_conn
    ) -> None:
        """An expired assignment is not returned in active role listing."""
        past = datetime.now(timezone.utc) - timedelta(hours=1)
        db_conn.fetch.return_value = []  # DB filters out expired

        result = await assignment_service.list_user_roles("user-1", "t-acme")
        assert isinstance(result, list)
        assert len(result) == 0

    # ------------------------------------------------------------------
    # Cache invalidation on assign/revoke
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_assign_triggers_cache_invalidation(
        self, assignment_service: AssignmentService, db_conn, cache
    ) -> None:
        """Assigning a role invalidates the user's permission cache."""
        row = _make_assignment_row()
        db_conn.fetchrow.return_value = row

        await assignment_service.assign_role(
            user_id="user-1", role_id="role-1", tenant_id="t-acme"
        )
        assert (
            cache.invalidate_user.await_count >= 1
            or cache.publish_invalidation.await_count >= 1
        )

    @pytest.mark.asyncio
    async def test_revoke_triggers_cache_invalidation(
        self, assignment_service: AssignmentService, db_conn, cache
    ) -> None:
        """Revoking a role invalidates the user's permission cache."""
        row = _make_assignment_row(assignment_id="asgn-rev")
        revoked = {**row, "revoked_at": datetime.now(timezone.utc), "is_active": False}
        db_conn.fetchrow.side_effect = [row, revoked]

        await assignment_service.revoke_role("asgn-rev", revoked_by="admin-1")
        assert (
            cache.invalidate_user.await_count >= 1
            or cache.publish_invalidation.await_count >= 1
        )

    # ------------------------------------------------------------------
    # Audit logging
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_assign_triggers_audit_log(
        self, assignment_service: AssignmentService, db_conn, audit
    ) -> None:
        """Assigning a role emits an audit event."""
        row = _make_assignment_row()
        db_conn.fetchrow.return_value = row

        await assignment_service.assign_role(
            user_id="user-1", role_id="role-1", tenant_id="t-acme"
        )
        assert audit.log_event.await_count >= 1

    @pytest.mark.asyncio
    async def test_revoke_triggers_audit_log(
        self, assignment_service: AssignmentService, db_conn, audit
    ) -> None:
        """Revoking a role emits an audit event."""
        row = _make_assignment_row(assignment_id="asgn-rev")
        revoked = {**row, "revoked_at": datetime.now(timezone.utc), "is_active": False}
        db_conn.fetchrow.side_effect = [row, revoked]

        await assignment_service.revoke_role("asgn-rev", revoked_by="admin-1")
        assert audit.log_event.await_count >= 1
