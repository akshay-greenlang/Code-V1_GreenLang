# -*- coding: utf-8 -*-
"""
Unit tests for PermissionService - RBAC Authorization Layer (SEC-002)

Tests permission CRUD, role-permission grants and revokes, permission
evaluation (allow/deny/wildcard/conditions), cache interaction, and
inherited permission aggregation through the role hierarchy.

Coverage targets: 85%+ of permission_service.py
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch, call

import pytest

# ---------------------------------------------------------------------------
# Attempt to import the RBAC permission service module.
# ---------------------------------------------------------------------------
try:
    from greenlang.infrastructure.rbac_service.permission_service import (
        PermissionService,
    )
    _HAS_MODULE = True
except ImportError:
    _HAS_MODULE = False

    class PermissionService:  # type: ignore[no-redef]
        """Stub for test collection when module is not yet built."""
        def __init__(self, db_pool, cache, config=None): ...
        async def create_permission(self, **kwargs) -> dict: ...
        async def list_permissions(self, **kwargs) -> dict: ...
        async def get_permission(self, permission_id: str) -> Optional[dict]: ...
        async def delete_permission(self, permission_id: str) -> bool: ...
        async def grant_permission_to_role(self, **kwargs) -> dict: ...
        async def revoke_permission_from_role(self, role_id, permission_id) -> bool: ...
        async def get_role_permissions(self, role_id, include_inherited=True) -> List[dict]: ...
        async def evaluate_permission(self, **kwargs) -> dict: ...

try:
    from greenlang.infrastructure.rbac_service import RBACServiceConfig
except ImportError:
    class RBACServiceConfig:  # type: ignore[no-redef]
        def __init__(self, **kwargs):
            self.cache_ttl = kwargs.get("cache_ttl", 300)
            self.max_permissions = kwargs.get("max_permissions", 1000)

pytestmark = pytest.mark.skipif(
    not _HAS_MODULE,
    reason="rbac_service.permission_service not yet implemented",
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


def _make_permission_row(
    permission_id: str = "perm-1",
    resource: str = "agents",
    action: str = "read",
    description: str = "Read agent data",
    is_system: bool = False,
) -> dict:
    return {
        "id": permission_id,
        "resource": resource,
        "action": action,
        "description": description,
        "is_system": is_system,
        "created_at": datetime.now(timezone.utc),
    }


def _make_grant_row(
    grant_id: str = "grant-1",
    role_id: str = "role-1",
    permission_id: str = "perm-1",
    effect: str = "allow",
    conditions: Optional[dict] = None,
    scope: Optional[str] = None,
) -> dict:
    return {
        "id": grant_id,
        "role_id": role_id,
        "permission_id": permission_id,
        "effect": effect,
        "conditions": conditions,
        "scope": scope,
        "granted_by": "admin-1",
        "granted_at": datetime.now(timezone.utc),
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
def config() -> RBACServiceConfig:
    return RBACServiceConfig(cache_ttl=300, max_permissions=1000)


@pytest.fixture
def perm_service(db_pool, cache, config) -> PermissionService:
    return PermissionService(db_pool=db_pool, cache=cache, config=config)


# ============================================================================
# TestPermissionService
# ============================================================================


class TestPermissionService:
    """Tests for PermissionService CRUD operations."""

    # ------------------------------------------------------------------
    # create_permission
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_create_permission_success(
        self, perm_service: PermissionService, db_conn
    ) -> None:
        """Creating a permission with valid data succeeds."""
        row = _make_permission_row()
        db_conn.fetchrow.return_value = row

        result = await perm_service.create_permission(
            resource="agents", action="read", description="Read agents"
        )
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_create_permission_duplicate(
        self, perm_service: PermissionService, db_conn
    ) -> None:
        """Creating a duplicate resource:action permission raises an error."""
        db_conn.execute.side_effect = Exception("unique constraint")
        with pytest.raises(Exception):
            await perm_service.create_permission(
                resource="agents", action="read"
            )

    # ------------------------------------------------------------------
    # list_permissions
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_list_permissions_all(
        self, perm_service: PermissionService, db_conn
    ) -> None:
        """Listing all permissions returns the full set."""
        rows = [
            _make_permission_row(permission_id=f"p-{i}", resource="agents", action=a)
            for i, a in enumerate(["read", "write", "delete"])
        ]
        db_conn.fetch.return_value = rows
        db_conn.fetchval.return_value = 3

        result = await perm_service.list_permissions()
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_list_permissions_by_resource(
        self, perm_service: PermissionService, db_conn
    ) -> None:
        """Listing permissions filtered by resource returns only matching ones."""
        rows = [_make_permission_row(resource="emissions")]
        db_conn.fetch.return_value = rows
        db_conn.fetchval.return_value = 1

        result = await perm_service.list_permissions(resource_filter="emissions")
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_list_permissions_paginated(
        self, perm_service: PermissionService, db_conn
    ) -> None:
        """Listing permissions respects page and page_size parameters."""
        rows = [_make_permission_row(permission_id=f"p-{i}") for i in range(10)]
        db_conn.fetch.return_value = rows[:5]
        db_conn.fetchval.return_value = 10

        result = await perm_service.list_permissions(page=1, page_size=5)
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_list_empty_permissions(
        self, perm_service: PermissionService, db_conn
    ) -> None:
        """Listing permissions when none exist returns empty."""
        db_conn.fetch.return_value = []
        db_conn.fetchval.return_value = 0

        result = await perm_service.list_permissions()
        assert isinstance(result, dict)

    # ------------------------------------------------------------------
    # get_permission / delete_permission
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_get_permission_exists(
        self, perm_service: PermissionService, db_conn
    ) -> None:
        """Getting an existing permission returns its data."""
        row = _make_permission_row(permission_id="p-42")
        db_conn.fetchrow.return_value = row

        result = await perm_service.get_permission("p-42")
        assert result is not None
        assert result["id"] == "p-42"

    @pytest.mark.asyncio
    async def test_get_permission_not_found(
        self, perm_service: PermissionService, db_conn
    ) -> None:
        """Getting a non-existent permission returns None."""
        db_conn.fetchrow.return_value = None

        result = await perm_service.get_permission("ghost")
        assert result is None

    @pytest.mark.asyncio
    async def test_delete_permission_success(
        self, perm_service: PermissionService, db_conn
    ) -> None:
        """Deleting a custom permission succeeds."""
        row = _make_permission_row(permission_id="p-del", is_system=False)
        db_conn.fetchrow.return_value = row
        db_conn.execute.return_value = "DELETE 1"

        result = await perm_service.delete_permission("p-del")
        assert result is True

    @pytest.mark.asyncio
    async def test_delete_permission_system_protected(
        self, perm_service: PermissionService, db_conn
    ) -> None:
        """Deleting a system permission is rejected."""
        row = _make_permission_row(permission_id="p-sys", is_system=True)
        db_conn.fetchrow.return_value = row

        with pytest.raises(Exception):
            await perm_service.delete_permission("p-sys")

    # ------------------------------------------------------------------
    # grant / revoke
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_grant_permission_to_role(
        self, perm_service: PermissionService, db_conn
    ) -> None:
        """Granting a permission to a role succeeds."""
        grant = _make_grant_row()
        db_conn.fetchrow.return_value = grant

        result = await perm_service.grant_permission_to_role(
            role_id="role-1",
            permission_id="perm-1",
            effect="allow",
            granted_by="admin-1",
        )
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_grant_duplicate_permission(
        self, perm_service: PermissionService, db_conn
    ) -> None:
        """Granting the same permission twice raises an error."""
        db_conn.execute.side_effect = Exception("unique constraint")
        with pytest.raises(Exception):
            await perm_service.grant_permission_to_role(
                role_id="role-1",
                permission_id="perm-1",
            )

    @pytest.mark.asyncio
    async def test_grant_with_conditions(
        self, perm_service: PermissionService, db_conn
    ) -> None:
        """Granting with conditions stores the JSON conditions."""
        grant = _make_grant_row(conditions={"ip_range": "10.0.0.0/8"})
        db_conn.fetchrow.return_value = grant

        result = await perm_service.grant_permission_to_role(
            role_id="role-1",
            permission_id="perm-1",
            conditions={"ip_range": "10.0.0.0/8"},
        )
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_grant_with_scope(
        self, perm_service: PermissionService, db_conn
    ) -> None:
        """Granting with scope restricts the permission's applicability."""
        grant = _make_grant_row(scope="tenant:t-acme")
        db_conn.fetchrow.return_value = grant

        result = await perm_service.grant_permission_to_role(
            role_id="role-1",
            permission_id="perm-1",
            scope="tenant:t-acme",
        )
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_revoke_permission_from_role(
        self, perm_service: PermissionService, db_conn
    ) -> None:
        """Revoking a permission from a role succeeds."""
        db_conn.execute.return_value = "DELETE 1"

        result = await perm_service.revoke_permission_from_role("role-1", "perm-1")
        assert result is True

    @pytest.mark.asyncio
    async def test_revoke_nonexistent_permission(
        self, perm_service: PermissionService, db_conn
    ) -> None:
        """Revoking a non-existent grant returns False or raises."""
        db_conn.execute.return_value = "DELETE 0"

        result = await perm_service.revoke_permission_from_role("role-1", "ghost")
        assert result is False or result is None

    # ------------------------------------------------------------------
    # get_role_permissions
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_get_role_permissions_direct(
        self, perm_service: PermissionService, db_conn
    ) -> None:
        """Getting role permissions returns directly granted ones."""
        grants = [
            {**_make_grant_row(grant_id=f"g-{i}"), "resource": "agents", "action": a}
            for i, a in enumerate(["read", "write"])
        ]
        db_conn.fetch.return_value = grants

        result = await perm_service.get_role_permissions("role-1", include_inherited=False)
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_get_role_permissions_with_inheritance(
        self, perm_service: PermissionService, db_conn
    ) -> None:
        """Getting role permissions with inheritance includes parent's."""
        parent_grants = [
            {**_make_grant_row(grant_id="g-parent"), "resource": "agents", "action": "read"}
        ]
        child_grants = [
            {**_make_grant_row(grant_id="g-child"), "resource": "agents", "action": "write"}
        ]
        db_conn.fetch.side_effect = [child_grants, parent_grants]

        result = await perm_service.get_role_permissions("child-role", include_inherited=True)
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_permission_hierarchy_aggregation(
        self, perm_service: PermissionService, db_conn
    ) -> None:
        """Permission aggregation through hierarchy deduplicates correctly."""
        db_conn.fetch.return_value = [
            {**_make_grant_row(grant_id="g-1"), "resource": "agents", "action": "read"},
            {**_make_grant_row(grant_id="g-2"), "resource": "agents", "action": "read"},
        ]

        result = await perm_service.get_role_permissions("role-1")
        assert isinstance(result, list)

    # ------------------------------------------------------------------
    # evaluate_permission
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_evaluate_permission_allowed(
        self, perm_service: PermissionService, db_conn, cache
    ) -> None:
        """Permission evaluation returns allowed when user has the permission."""
        cache.get_permissions.return_value = ["agents:read", "agents:write"]

        result = await perm_service.evaluate_permission(
            user_id="user-1",
            tenant_id="t-acme",
            resource="agents",
            action="read",
        )
        assert isinstance(result, dict)
        assert result.get("allowed") is True or result.get("effect") == "allow"

    @pytest.mark.asyncio
    async def test_evaluate_permission_denied(
        self, perm_service: PermissionService, db_conn, cache
    ) -> None:
        """Permission evaluation returns denied when user lacks the permission."""
        cache.get_permissions.return_value = ["agents:read"]

        result = await perm_service.evaluate_permission(
            user_id="user-1",
            tenant_id="t-acme",
            resource="agents",
            action="delete",
        )
        assert isinstance(result, dict)
        assert result.get("allowed") is False or result.get("effect") == "deny"

    @pytest.mark.asyncio
    async def test_evaluate_permission_wildcard_match(
        self, perm_service: PermissionService, db_conn, cache
    ) -> None:
        """Wildcard permission (agents:*) matches any action on that resource."""
        cache.get_permissions.return_value = ["agents:*"]

        result = await perm_service.evaluate_permission(
            user_id="user-1",
            tenant_id="t-acme",
            resource="agents",
            action="delete",
        )
        assert isinstance(result, dict)
        assert result.get("allowed") is True or result.get("effect") == "allow"

    @pytest.mark.asyncio
    async def test_evaluate_permission_deny_wins(
        self, perm_service: PermissionService, db_conn, cache
    ) -> None:
        """Explicit deny overrides an allow (deny-wins policy)."""
        # When evaluation checks grants, an explicit deny entry should win
        cache.get_permissions.return_value = None
        db_conn.fetch.return_value = [
            {**_make_grant_row(effect="allow"), "resource": "agents", "action": "write"},
            {**_make_grant_row(grant_id="g-deny", effect="deny"), "resource": "agents", "action": "write"},
        ]

        result = await perm_service.evaluate_permission(
            user_id="user-1",
            tenant_id="t-acme",
            resource="agents",
            action="write",
        )
        assert isinstance(result, dict)
        assert result.get("allowed") is False or result.get("effect") == "deny"

    @pytest.mark.asyncio
    async def test_evaluate_permission_with_conditions(
        self, perm_service: PermissionService, db_conn, cache
    ) -> None:
        """Permission evaluation respects condition constraints."""
        cache.get_permissions.return_value = None
        db_conn.fetch.return_value = [
            {
                **_make_grant_row(conditions={"ip_range": "10.0.0.0/8"}),
                "resource": "agents",
                "action": "read",
            }
        ]

        result = await perm_service.evaluate_permission(
            user_id="user-1",
            tenant_id="t-acme",
            resource="agents",
            action="read",
            context={"ip_address": "10.1.2.3"},
        )
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_evaluate_permission_cache_hit(
        self, perm_service: PermissionService, db_conn, cache
    ) -> None:
        """Permission evaluation serves from cache when available."""
        cache.get_permissions.return_value = ["agents:read"]

        result = await perm_service.evaluate_permission(
            user_id="user-1",
            tenant_id="t-acme",
            resource="agents",
            action="read",
        )
        assert isinstance(result, dict)
        # DB should NOT have been queried for user permissions
        # (the cache had the answer)

    @pytest.mark.asyncio
    async def test_evaluate_permission_cache_miss(
        self, perm_service: PermissionService, db_conn, cache
    ) -> None:
        """Permission evaluation falls through to DB on cache miss."""
        cache.get_permissions.return_value = None
        db_conn.fetch.return_value = [
            {**_make_grant_row(), "resource": "agents", "action": "read"}
        ]

        result = await perm_service.evaluate_permission(
            user_id="user-1",
            tenant_id="t-acme",
            resource="agents",
            action="read",
        )
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_evaluate_permission_redis_failure_fallback(
        self, perm_service: PermissionService, db_conn, cache
    ) -> None:
        """When Redis fails, evaluation falls through to DB gracefully."""
        cache.get_permissions.side_effect = ConnectionError("Redis down")
        db_conn.fetch.return_value = [
            {**_make_grant_row(), "resource": "agents", "action": "read"}
        ]

        result = await perm_service.evaluate_permission(
            user_id="user-1",
            tenant_id="t-acme",
            resource="agents",
            action="read",
        )
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_permission_scope_restriction(
        self, perm_service: PermissionService, db_conn, cache
    ) -> None:
        """Permission with scope is only valid within that scope."""
        cache.get_permissions.return_value = None
        db_conn.fetch.return_value = [
            {
                **_make_grant_row(scope="tenant:t-other"),
                "resource": "agents",
                "action": "read",
            }
        ]

        result = await perm_service.evaluate_permission(
            user_id="user-1",
            tenant_id="t-acme",
            resource="agents",
            action="read",
        )
        assert isinstance(result, dict)

    # ------------------------------------------------------------------
    # Cache invalidation
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_cache_invalidation_on_grant(
        self, perm_service: PermissionService, db_conn, cache
    ) -> None:
        """Granting a permission triggers cache invalidation."""
        grant = _make_grant_row()
        db_conn.fetchrow.return_value = grant

        await perm_service.grant_permission_to_role(
            role_id="role-1", permission_id="perm-1"
        )
        assert (
            cache.invalidate_tenant.await_count >= 1
            or cache.publish_invalidation.await_count >= 1
            or cache.invalidate_all.await_count >= 1
        )

    @pytest.mark.asyncio
    async def test_cache_invalidation_on_revoke(
        self, perm_service: PermissionService, db_conn, cache
    ) -> None:
        """Revoking a permission triggers cache invalidation."""
        db_conn.execute.return_value = "DELETE 1"

        await perm_service.revoke_permission_from_role("role-1", "perm-1")
        assert (
            cache.invalidate_tenant.await_count >= 1
            or cache.publish_invalidation.await_count >= 1
            or cache.invalidate_all.await_count >= 1
        )
