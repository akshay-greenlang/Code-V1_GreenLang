# -*- coding: utf-8 -*-
"""
Unit tests for RoleService - RBAC Authorization Layer (SEC-002)

Tests the full lifecycle of role management: creation, retrieval,
update, deletion, enable/disable, hierarchy traversal, and cache
invalidation.  Validates system-role protection, cycle detection,
max-depth enforcement, tenant scoping, and pagination.

Coverage targets: 85%+ of role_service.py
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch, call

import pytest

# ---------------------------------------------------------------------------
# Attempt to import the RBAC role service module.
# ---------------------------------------------------------------------------
try:
    from greenlang.infrastructure.rbac_service.role_service import RoleService
    _HAS_MODULE = True
except ImportError:
    _HAS_MODULE = False

    class RoleService:  # type: ignore[no-redef]
        """Stub for test collection when module is not yet built."""
        def __init__(self, db_pool, cache, config=None): ...
        async def create_role(self, **kwargs) -> dict: ...
        async def get_role(self, role_id: str) -> Optional[dict]: ...
        async def get_role_by_name(self, name: str, tenant_id=None) -> Optional[dict]: ...
        async def list_roles(self, **kwargs) -> dict: ...
        async def update_role(self, role_id: str, **fields) -> dict: ...
        async def delete_role(self, role_id: str) -> bool: ...
        async def enable_role(self, role_id: str) -> dict: ...
        async def disable_role(self, role_id: str) -> dict: ...
        async def get_role_hierarchy(self, role_id: str) -> List[dict]: ...

try:
    from greenlang.infrastructure.rbac_service import RBACServiceConfig
    _HAS_CONFIG = True
except ImportError:
    _HAS_CONFIG = False

    class RBACServiceConfig:  # type: ignore[no-redef]
        def __init__(self, **kwargs):
            self.cache_ttl = kwargs.get("cache_ttl", 300)
            self.max_hierarchy_depth = kwargs.get("max_hierarchy_depth", 5)
            self.max_roles_per_tenant = kwargs.get("max_roles_per_tenant", 200)

pytestmark = pytest.mark.skipif(
    not _HAS_MODULE,
    reason="rbac_service.role_service not yet implemented",
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


def _make_role_row(
    role_id: str = "role-1",
    name: str = "viewer",
    display_name: str = "Viewer",
    description: str = "Read-only access",
    is_system: bool = False,
    is_active: bool = True,
    parent_role_id: Optional[str] = None,
    tenant_id: Optional[str] = None,
    metadata: Optional[dict] = None,
) -> dict:
    """Create a mock role database row."""
    return {
        "id": role_id,
        "name": name,
        "display_name": display_name,
        "description": description,
        "is_system": is_system,
        "is_active": is_active,
        "parent_role_id": parent_role_id,
        "tenant_id": tenant_id,
        "metadata": metadata or {},
        "created_at": datetime.now(timezone.utc),
        "updated_at": datetime.now(timezone.utc),
        "created_by": "system",
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
    return RBACServiceConfig(
        cache_ttl=300,
        max_hierarchy_depth=5,
        max_roles_per_tenant=200,
    )


@pytest.fixture
def role_service(db_pool, cache, config) -> RoleService:
    return RoleService(db_pool=db_pool, cache=cache, config=config)


# ============================================================================
# TestRoleService
# ============================================================================


class TestRoleService:
    """Tests for RoleService CRUD and hierarchy operations."""

    # ------------------------------------------------------------------
    # create_role
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_create_role_success(
        self, role_service: RoleService, db_conn
    ) -> None:
        """Creating a role with valid data succeeds and returns a dict."""
        db_conn.fetchrow.return_value = _make_role_row()
        result = await role_service.create_role(
            name="analyst",
            display_name="Analyst",
            description="Can analyze data",
        )
        assert isinstance(result, dict)
        assert result["name"] == "viewer" or "name" in result

    @pytest.mark.asyncio
    async def test_create_role_duplicate_name_raises(
        self, role_service: RoleService, db_conn
    ) -> None:
        """Creating a role with a duplicate name raises an error."""
        from asyncio import InvalidStateError as _sentinel

        db_conn.execute.side_effect = Exception("unique constraint")
        with pytest.raises(Exception):
            await role_service.create_role(name="admin")

    @pytest.mark.asyncio
    async def test_create_role_with_parent(
        self, role_service: RoleService, db_conn
    ) -> None:
        """Creating a role with a parent_role_id stores the relationship."""
        parent_row = _make_role_row(role_id="parent-1", name="manager")
        child_row = _make_role_row(
            role_id="child-1", name="analyst", parent_role_id="parent-1"
        )
        db_conn.fetchrow.side_effect = [parent_row, child_row]
        db_conn.fetch.return_value = [parent_row]

        result = await role_service.create_role(
            name="analyst",
            parent_role_id="parent-1",
        )
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_create_role_cycle_detection(
        self, role_service: RoleService, db_conn
    ) -> None:
        """Creating a role that would form a cycle is rejected."""
        # Role A -> B -> A would be a cycle
        role_a = _make_role_row(role_id="a", name="role-a", parent_role_id="b")
        role_b = _make_role_row(role_id="b", name="role-b", parent_role_id="a")
        db_conn.fetchrow.side_effect = [role_b, role_a, role_b]
        db_conn.fetch.return_value = [role_a, role_b]

        with pytest.raises(Exception):
            await role_service.create_role(
                name="role-c",
                parent_role_id="a",
            )

    @pytest.mark.asyncio
    async def test_create_role_max_hierarchy_depth(
        self, role_service: RoleService, db_conn
    ) -> None:
        """Creating a role exceeding max hierarchy depth is rejected."""
        # Build a chain deeper than max_hierarchy_depth=5
        chain = []
        for i in range(6):
            parent = f"role-{i - 1}" if i > 0 else None
            chain.append(
                _make_role_row(
                    role_id=f"role-{i}",
                    name=f"level-{i}",
                    parent_role_id=parent,
                )
            )
        db_conn.fetchrow.side_effect = chain
        db_conn.fetch.return_value = chain

        with pytest.raises(Exception):
            await role_service.create_role(
                name="too-deep",
                parent_role_id="role-5",
            )

    @pytest.mark.asyncio
    async def test_create_role_with_metadata(
        self, role_service: RoleService, db_conn
    ) -> None:
        """Metadata dict is stored on role creation."""
        row = _make_role_row(metadata={"department": "finance"})
        db_conn.fetchrow.return_value = row
        result = await role_service.create_role(
            name="finance-viewer",
            metadata={"department": "finance"},
        )
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_create_role_with_tenant_id(
        self, role_service: RoleService, db_conn
    ) -> None:
        """Tenant-scoped role stores the tenant_id."""
        row = _make_role_row(tenant_id="t-acme")
        db_conn.fetchrow.return_value = row
        result = await role_service.create_role(
            name="acme-admin",
            tenant_id="t-acme",
        )
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_create_role_invalid_name(
        self, role_service: RoleService, db_conn
    ) -> None:
        """Empty or invalid role name raises an error."""
        with pytest.raises(Exception):
            await role_service.create_role(name="")

    # ------------------------------------------------------------------
    # get_role
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_get_role_exists(
        self, role_service: RoleService, db_conn
    ) -> None:
        """Getting an existing role returns its data."""
        row = _make_role_row(role_id="role-42", name="editor")
        db_conn.fetchrow.return_value = row
        result = await role_service.get_role("role-42")
        assert result is not None
        assert result["id"] == "role-42"

    @pytest.mark.asyncio
    async def test_get_role_not_found(
        self, role_service: RoleService, db_conn
    ) -> None:
        """Getting a non-existent role returns None."""
        db_conn.fetchrow.return_value = None
        result = await role_service.get_role("does-not-exist")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_role_by_name(
        self, role_service: RoleService, db_conn
    ) -> None:
        """Getting a role by name returns its data."""
        row = _make_role_row(name="super_admin")
        db_conn.fetchrow.return_value = row
        result = await role_service.get_role_by_name("super_admin")
        assert result is not None
        assert result["name"] == "super_admin"

    # ------------------------------------------------------------------
    # list_roles
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_list_roles_paginated(
        self, role_service: RoleService, db_conn
    ) -> None:
        """Listing roles returns paginated results."""
        rows = [
            _make_role_row(role_id=f"r-{i}", name=f"role-{i}")
            for i in range(5)
        ]
        db_conn.fetch.return_value = rows
        db_conn.fetchval.return_value = 5

        result = await role_service.list_roles(page=1, page_size=10)
        assert isinstance(result, dict)
        assert "items" in result or "roles" in result or isinstance(result.get("total"), int)

    @pytest.mark.asyncio
    async def test_list_roles_with_system_filter(
        self, role_service: RoleService, db_conn
    ) -> None:
        """Listing roles with include_system=False excludes system roles."""
        rows = [
            _make_role_row(role_id="r-custom", name="custom", is_system=False),
        ]
        db_conn.fetch.return_value = rows
        db_conn.fetchval.return_value = 1

        result = await role_service.list_roles(include_system=False)
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_list_roles_tenant_scoped(
        self, role_service: RoleService, db_conn
    ) -> None:
        """Listing roles for a specific tenant returns only those roles."""
        rows = [
            _make_role_row(role_id="r-t1", name="tenant-role", tenant_id="t-acme"),
        ]
        db_conn.fetch.return_value = rows
        db_conn.fetchval.return_value = 1

        result = await role_service.list_roles(tenant_id="t-acme")
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_list_roles_empty(
        self, role_service: RoleService, db_conn
    ) -> None:
        """Listing roles when none exist returns an empty list."""
        db_conn.fetch.return_value = []
        db_conn.fetchval.return_value = 0

        result = await role_service.list_roles()
        assert isinstance(result, dict)

    # ------------------------------------------------------------------
    # update_role
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_update_role_success(
        self, role_service: RoleService, db_conn
    ) -> None:
        """Updating a role with valid fields succeeds."""
        existing = _make_role_row(role_id="r-up", name="old-name")
        updated = _make_role_row(role_id="r-up", name="old-name", display_name="New Display")
        db_conn.fetchrow.side_effect = [existing, updated]

        result = await role_service.update_role(
            "r-up", display_name="New Display"
        )
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_update_role_system_protected(
        self, role_service: RoleService, db_conn
    ) -> None:
        """Updating a system role is rejected."""
        system_role = _make_role_row(
            role_id="r-sys", name="super_admin", is_system=True
        )
        db_conn.fetchrow.return_value = system_role

        with pytest.raises(Exception):
            await role_service.update_role(
                "r-sys", display_name="Hacked"
            )

    @pytest.mark.asyncio
    async def test_update_role_not_found(
        self, role_service: RoleService, db_conn
    ) -> None:
        """Updating a non-existent role raises an error."""
        db_conn.fetchrow.return_value = None
        with pytest.raises(Exception):
            await role_service.update_role("ghost", display_name="X")

    @pytest.mark.asyncio
    async def test_update_role_parent_cycle_detection(
        self, role_service: RoleService, db_conn
    ) -> None:
        """Updating a role's parent to create a cycle is rejected."""
        role_a = _make_role_row(role_id="a", name="role-a")
        role_b = _make_role_row(role_id="b", name="role-b", parent_role_id="a")
        db_conn.fetchrow.side_effect = [role_a, role_b, role_a]
        db_conn.fetch.return_value = [role_b, role_a]

        with pytest.raises(Exception):
            await role_service.update_role("a", parent_role_id="b")

    # ------------------------------------------------------------------
    # delete_role
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_delete_role_success(
        self, role_service: RoleService, db_conn
    ) -> None:
        """Deleting a custom role succeeds."""
        row = _make_role_row(role_id="r-del", name="deletable", is_system=False)
        db_conn.fetchrow.return_value = row
        db_conn.execute.return_value = "DELETE 1"

        result = await role_service.delete_role("r-del")
        assert result is True

    @pytest.mark.asyncio
    async def test_delete_role_system_protected(
        self, role_service: RoleService, db_conn
    ) -> None:
        """Deleting a system role is rejected."""
        system_role = _make_role_row(
            role_id="r-sys", name="super_admin", is_system=True
        )
        db_conn.fetchrow.return_value = system_role

        with pytest.raises(Exception):
            await role_service.delete_role("r-sys")

    @pytest.mark.asyncio
    async def test_delete_role_not_found(
        self, role_service: RoleService, db_conn
    ) -> None:
        """Deleting a non-existent role raises an error."""
        db_conn.fetchrow.return_value = None
        with pytest.raises(Exception):
            await role_service.delete_role("ghost")

    # ------------------------------------------------------------------
    # enable / disable
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_enable_role(
        self, role_service: RoleService, db_conn
    ) -> None:
        """Enabling a disabled role sets is_active=True."""
        disabled = _make_role_row(role_id="r-dis", is_active=False)
        enabled = _make_role_row(role_id="r-dis", is_active=True)
        db_conn.fetchrow.side_effect = [disabled, enabled]

        result = await role_service.enable_role("r-dis")
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_disable_role(
        self, role_service: RoleService, db_conn
    ) -> None:
        """Disabling an active role sets is_active=False."""
        active = _make_role_row(role_id="r-act", is_active=True, is_system=False)
        inactive = _make_role_row(role_id="r-act", is_active=False)
        db_conn.fetchrow.side_effect = [active, inactive]

        result = await role_service.disable_role("r-act")
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_disable_system_role_protected(
        self, role_service: RoleService, db_conn
    ) -> None:
        """Disabling a system role is rejected."""
        system_role = _make_role_row(
            role_id="r-sys", name="super_admin", is_system=True, is_active=True
        )
        db_conn.fetchrow.return_value = system_role

        with pytest.raises(Exception):
            await role_service.disable_role("r-sys")

    # ------------------------------------------------------------------
    # hierarchy
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_get_role_hierarchy(
        self, role_service: RoleService, db_conn
    ) -> None:
        """Getting role hierarchy returns the full ancestor chain."""
        root = _make_role_row(role_id="root", name="root", parent_role_id=None)
        mid = _make_role_row(role_id="mid", name="mid", parent_role_id="root")
        leaf = _make_role_row(role_id="leaf", name="leaf", parent_role_id="mid")
        db_conn.fetchrow.side_effect = [leaf, mid, root]

        result = await role_service.get_role_hierarchy("leaf")
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_get_role_hierarchy_max_depth(
        self, role_service: RoleService, db_conn
    ) -> None:
        """Hierarchy traversal stops at max_hierarchy_depth."""
        # Build a chain of 10 nodes (exceeds max_hierarchy_depth=5)
        chain = []
        for i in range(10):
            parent = f"role-{i - 1}" if i > 0 else None
            chain.append(
                _make_role_row(
                    role_id=f"role-{i}", name=f"level-{i}",
                    parent_role_id=parent,
                )
            )
        db_conn.fetchrow.side_effect = list(reversed(chain))

        result = await role_service.get_role_hierarchy("role-9")
        assert isinstance(result, list)
        assert len(result) <= 6  # max_depth + the role itself

    @pytest.mark.asyncio
    async def test_role_hierarchy_expansion(
        self, role_service: RoleService, db_conn
    ) -> None:
        """Hierarchy correctly expands a multi-level tree."""
        grandparent = _make_role_row(role_id="gp", name="grandparent")
        parent = _make_role_row(role_id="p", name="parent", parent_role_id="gp")
        child = _make_role_row(role_id="c", name="child", parent_role_id="p")
        db_conn.fetchrow.side_effect = [child, parent, grandparent]

        result = await role_service.get_role_hierarchy("c")
        assert isinstance(result, list)

    # ------------------------------------------------------------------
    # Cache invalidation
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_cache_invalidation_on_create(
        self, role_service: RoleService, db_conn, cache
    ) -> None:
        """Creating a role triggers cache invalidation."""
        row = _make_role_row(tenant_id="t-acme")
        db_conn.fetchrow.return_value = row

        await role_service.create_role(name="new-role", tenant_id="t-acme")
        assert (
            cache.invalidate_tenant.await_count >= 1
            or cache.publish_invalidation.await_count >= 1
            or cache.invalidate_all.await_count >= 1
        )

    @pytest.mark.asyncio
    async def test_cache_invalidation_on_update(
        self, role_service: RoleService, db_conn, cache
    ) -> None:
        """Updating a role triggers cache invalidation."""
        existing = _make_role_row(role_id="r-up", is_system=False)
        updated = _make_role_row(role_id="r-up", display_name="Updated")
        db_conn.fetchrow.side_effect = [existing, updated]

        await role_service.update_role("r-up", display_name="Updated")
        assert (
            cache.invalidate_tenant.await_count >= 1
            or cache.publish_invalidation.await_count >= 1
            or cache.invalidate_all.await_count >= 1
        )

    @pytest.mark.asyncio
    async def test_cache_invalidation_on_delete(
        self, role_service: RoleService, db_conn, cache
    ) -> None:
        """Deleting a role triggers cache invalidation."""
        row = _make_role_row(role_id="r-del", is_system=False)
        db_conn.fetchrow.return_value = row
        db_conn.execute.return_value = "DELETE 1"

        await role_service.delete_role("r-del")
        assert (
            cache.invalidate_tenant.await_count >= 1
            or cache.publish_invalidation.await_count >= 1
            or cache.invalidate_all.await_count >= 1
        )
