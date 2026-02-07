# -*- coding: utf-8 -*-
"""
Role Service - RBAC Authorization Service (SEC-002)

Full CRUD management for hierarchical roles in the ``security.roles``
table.  Provides cycle detection on parent-child relationships,
system-role protection, tenant-scoped queries, and cache invalidation
on every mutation.

All methods are async and operate against an async ``psycopg``
connection pool.

Features:
    - Create, read, update, delete roles
    - Hierarchical parent-child relationships with max depth of 5
    - Cycle detection on parent_role_id changes
    - System role protection (cannot delete/modify is_system_role=true)
    - Tenant scoping: system roles (tenant_id=NULL) visible to all
    - Cache invalidation via ``RBACCache.publish_invalidation()``

Example:
    >>> svc = RoleService(db_pool, cache, config)
    >>> role = await svc.create_role(
    ...     name="team_lead",
    ...     display_name="Team Lead",
    ...     tenant_id="t-1",
    ...     parent_role_id=manager_id,
    ...     created_by="admin-user",
    ... )
    >>> print(role["name"])
    'team_lead'

Author: GreenLang Framework Team
Date: February 2026
Status: Production Ready
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import UUID

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class RoleNotFoundError(Exception):
    """Raised when a role lookup fails."""


class SystemRoleProtectionError(Exception):
    """Raised when attempting to modify or delete a system role."""


class RoleHierarchyCycleError(Exception):
    """Raised when a parent assignment would create a cycle."""


class RoleHierarchyDepthError(Exception):
    """Raised when the hierarchy exceeds the maximum allowed depth."""


class DuplicateRoleError(Exception):
    """Raised when a role name conflicts within the same tenant."""


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class RoleService:
    """CRUD service for RBAC roles with hierarchy support.

    Args:
        db_pool: Async ``psycopg_pool.AsyncConnectionPool``.
        cache: ``RBACCache`` instance for invalidation.
        config: ``RBACServiceConfig`` providing ``max_hierarchy_depth``
            and ``max_roles_per_tenant``.
    """

    def __init__(self, db_pool: Any, cache: Any, config: Any) -> None:
        """Initialize the role service.

        Args:
            db_pool: Async PostgreSQL connection pool.
            cache: RBACCache for invalidation broadcasts.
            config: RBACServiceConfig instance.
        """
        self._pool = db_pool
        self._cache = cache
        self._max_depth = getattr(config, "max_hierarchy_depth", 5)
        self._max_per_tenant = getattr(config, "max_roles_per_tenant", 200)

    # ------------------------------------------------------------------
    # Create
    # ------------------------------------------------------------------

    async def create_role(
        self,
        name: str,
        display_name: str,
        tenant_id: Optional[str] = None,
        description: Optional[str] = None,
        parent_role_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        created_by: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create a new role.

        Args:
            name: Unique role name within the tenant.
            display_name: Human-readable display name.
            tenant_id: Tenant UUID (``None`` for system roles).
            description: Optional description.
            parent_role_id: UUID of the parent role for hierarchy.
            metadata: Optional JSONB metadata.
            created_by: Identity of the creator.

        Returns:
            Dictionary representation of the created role.

        Raises:
            RoleHierarchyDepthError: If adding this role would exceed
                the maximum hierarchy depth.
            DuplicateRoleError: If a role with the same name already
                exists in the tenant.
        """
        import json as _json

        if parent_role_id:
            depth = await self._get_hierarchy_depth(parent_role_id)
            if depth + 1 > self._max_depth:
                raise RoleHierarchyDepthError(
                    f"Adding child would exceed max depth of {self._max_depth} "
                    f"(parent depth={depth})"
                )

        meta_json = _json.dumps(metadata or {})

        async with self._pool.connection() as conn:
            try:
                row = await conn.execute(
                    """
                    INSERT INTO security.roles
                        (name, display_name, description, tenant_id,
                         parent_role_id, metadata, created_by)
                    VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s)
                    RETURNING id, name, display_name, description, tenant_id,
                              parent_role_id, is_system_role, is_enabled,
                              metadata, created_at, created_by, updated_at
                    """,
                    (
                        name,
                        display_name,
                        description,
                        tenant_id,
                        parent_role_id,
                        meta_json,
                        created_by,
                    ),
                )
                record = await row.fetchone()
            except Exception as exc:
                error_msg = str(exc)
                if "uq_role_name_tenant" in error_msg:
                    raise DuplicateRoleError(
                        f"Role '{name}' already exists in tenant {tenant_id}"
                    ) from exc
                raise

        role = self._row_to_dict(record)
        logger.info(
            "Created role  id=%s  name=%s  tenant=%s",
            role["id"],
            role["name"],
            role.get("tenant_id"),
        )

        await self._invalidate_tenant(tenant_id)
        return role

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    async def get_role(self, role_id: str) -> Dict[str, Any]:
        """Get a role by its UUID.

        Args:
            role_id: UUID of the role.

        Returns:
            Dictionary representation of the role.

        Raises:
            RoleNotFoundError: If the role does not exist.
        """
        async with self._pool.connection() as conn:
            row = await conn.execute(
                """
                SELECT id, name, display_name, description, tenant_id,
                       parent_role_id, is_system_role, is_enabled,
                       metadata, created_at, created_by, updated_at
                FROM security.roles
                WHERE id = %s
                """,
                (role_id,),
            )
            record = await row.fetchone()

        if record is None:
            raise RoleNotFoundError(f"Role not found: {role_id}")

        return self._row_to_dict(record)

    async def get_role_by_name(
        self,
        name: str,
        tenant_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get a role by name within a tenant scope.

        System roles (``tenant_id IS NULL``) are matched when
        ``tenant_id`` is ``None``.

        Args:
            name: Role name.
            tenant_id: Tenant UUID or ``None`` for system roles.

        Returns:
            Dictionary representation of the role.

        Raises:
            RoleNotFoundError: If the role does not exist.
        """
        async with self._pool.connection() as conn:
            if tenant_id is None:
                row = await conn.execute(
                    """
                    SELECT id, name, display_name, description, tenant_id,
                           parent_role_id, is_system_role, is_enabled,
                           metadata, created_at, created_by, updated_at
                    FROM security.roles
                    WHERE name = %s AND tenant_id IS NULL
                    """,
                    (name,),
                )
            else:
                row = await conn.execute(
                    """
                    SELECT id, name, display_name, description, tenant_id,
                           parent_role_id, is_system_role, is_enabled,
                           metadata, created_at, created_by, updated_at
                    FROM security.roles
                    WHERE name = %s AND tenant_id = %s
                    """,
                    (name, tenant_id),
                )
            record = await row.fetchone()

        if record is None:
            raise RoleNotFoundError(
                f"Role not found: name={name}, tenant_id={tenant_id}"
            )

        return self._row_to_dict(record)

    async def list_roles(
        self,
        tenant_id: Optional[str] = None,
        include_system: bool = True,
        enabled_only: bool = True,
    ) -> List[Dict[str, Any]]:
        """List roles visible to a tenant.

        Returns both system roles (``tenant_id IS NULL``) and
        tenant-scoped roles when ``include_system`` is ``True``.

        Args:
            tenant_id: Tenant UUID to scope results.
            include_system: Include system roles in the result.
            enabled_only: Only return enabled roles.

        Returns:
            List of role dictionaries.
        """
        conditions: List[str] = []
        params: List[Any] = []

        if tenant_id is not None:
            if include_system:
                conditions.append("(tenant_id = %s OR tenant_id IS NULL)")
                params.append(tenant_id)
            else:
                conditions.append("tenant_id = %s")
                params.append(tenant_id)
        elif not include_system:
            conditions.append("tenant_id IS NOT NULL")

        if enabled_only:
            conditions.append("is_enabled = true")

        where = ""
        if conditions:
            where = "WHERE " + " AND ".join(conditions)

        query = f"""
            SELECT id, name, display_name, description, tenant_id,
                   parent_role_id, is_system_role, is_enabled,
                   metadata, created_at, created_by, updated_at
            FROM security.roles
            {where}
            ORDER BY is_system_role DESC, name ASC
        """

        async with self._pool.connection() as conn:
            cursor = await conn.execute(query, tuple(params))
            records = await cursor.fetchall()

        return [self._row_to_dict(r) for r in records]

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    async def update_role(
        self,
        role_id: str,
        display_name: Optional[str] = None,
        description: Optional[str] = None,
        parent_role_id: Optional[str] = ...,  # type: ignore[assignment]
        is_enabled: Optional[bool] = None,
        metadata: Optional[Dict[str, Any]] = None,
        updated_by: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Update a role's attributes.

        System roles cannot be modified (raises ``SystemRoleProtectionError``).
        Changing ``parent_role_id`` triggers cycle detection.

        Args:
            role_id: UUID of the role to update.
            display_name: New display name (if provided).
            description: New description (if provided).
            parent_role_id: New parent role UUID. Pass ``None`` to
                remove the parent. Use the default sentinel ``...``
                to leave unchanged.
            is_enabled: Enable or disable the role.
            metadata: New metadata (replaces existing).
            updated_by: Identity of the updater.

        Returns:
            Updated role dictionary.

        Raises:
            RoleNotFoundError: If the role does not exist.
            SystemRoleProtectionError: If the role is a system role.
            RoleHierarchyCycleError: If the parent change creates a cycle.
            RoleHierarchyDepthError: If the parent change exceeds max depth.
        """
        import json as _json

        # Fetch current role to check system protection
        current = await self.get_role(role_id)
        if current["is_system_role"]:
            raise SystemRoleProtectionError(
                f"Cannot modify system role: {current['name']}"
            )

        # Build SET clause dynamically
        updates: List[str] = []
        params: List[Any] = []

        if display_name is not None:
            updates.append("display_name = %s")
            params.append(display_name)

        if description is not None:
            updates.append("description = %s")
            params.append(description)

        if parent_role_id is not ...:
            # Validate hierarchy if setting a new parent
            if parent_role_id is not None:
                await self._validate_no_cycle(role_id, parent_role_id)
                depth = await self._get_hierarchy_depth(parent_role_id)
                if depth + 1 > self._max_depth:
                    raise RoleHierarchyDepthError(
                        f"Parent change would exceed max depth of {self._max_depth}"
                    )
            updates.append("parent_role_id = %s")
            params.append(parent_role_id)

        if is_enabled is not None:
            updates.append("is_enabled = %s")
            params.append(is_enabled)

        if metadata is not None:
            updates.append("metadata = %s::jsonb")
            params.append(_json.dumps(metadata))

        if not updates:
            return current

        params.append(role_id)

        query = f"""
            UPDATE security.roles
            SET {', '.join(updates)}
            WHERE id = %s
            RETURNING id, name, display_name, description, tenant_id,
                      parent_role_id, is_system_role, is_enabled,
                      metadata, created_at, created_by, updated_at
        """

        async with self._pool.connection() as conn:
            cursor = await conn.execute(query, tuple(params))
            record = await cursor.fetchone()

        if record is None:
            raise RoleNotFoundError(f"Role not found after update: {role_id}")

        role = self._row_to_dict(record)
        logger.info("Updated role  id=%s  name=%s", role["id"], role["name"])

        await self._invalidate_tenant(role.get("tenant_id"))
        return role

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------

    async def delete_role(self, role_id: str, deleted_by: Optional[str] = None) -> bool:
        """Delete a role.

        System roles cannot be deleted.  Deleting a role cascades to
        ``role_permissions`` and ``user_roles`` via FK ``ON DELETE CASCADE``.

        Args:
            role_id: UUID of the role to delete.
            deleted_by: Identity of the deleter (for audit logging).

        Returns:
            ``True`` if the role was deleted.

        Raises:
            RoleNotFoundError: If the role does not exist.
            SystemRoleProtectionError: If the role is a system role.
        """
        current = await self.get_role(role_id)
        if current["is_system_role"]:
            raise SystemRoleProtectionError(
                f"Cannot delete system role: {current['name']}"
            )

        async with self._pool.connection() as conn:
            result = await conn.execute(
                "DELETE FROM security.roles WHERE id = %s",
                (role_id,),
            )

        logger.info(
            "Deleted role  id=%s  name=%s  by=%s",
            role_id,
            current["name"],
            deleted_by,
        )

        await self._invalidate_tenant(current.get("tenant_id"))
        return True

    # ------------------------------------------------------------------
    # Hierarchy helpers
    # ------------------------------------------------------------------

    async def get_role_hierarchy(self, role_id: str) -> List[Dict[str, Any]]:
        """Walk up the role hierarchy from child to root.

        Args:
            role_id: UUID of the starting role.

        Returns:
            List of role dictionaries from the role itself up to the
            root (inclusive), ordered child-first.
        """
        chain: List[Dict[str, Any]] = []
        visited: set[str] = set()
        current_id: Optional[str] = role_id

        while current_id and len(chain) < self._max_depth + 1:
            if current_id in visited:
                logger.warning("Cycle detected in role hierarchy at %s", current_id)
                break
            visited.add(current_id)

            role = await self.get_role(current_id)
            chain.append(role)
            current_id = role.get("parent_role_id")

        return chain

    async def get_child_roles(self, role_id: str) -> List[Dict[str, Any]]:
        """Get all direct child roles of a given role.

        Args:
            role_id: UUID of the parent role.

        Returns:
            List of child role dictionaries.
        """
        async with self._pool.connection() as conn:
            cursor = await conn.execute(
                """
                SELECT id, name, display_name, description, tenant_id,
                       parent_role_id, is_system_role, is_enabled,
                       metadata, created_at, created_by, updated_at
                FROM security.roles
                WHERE parent_role_id = %s
                ORDER BY name
                """,
                (role_id,),
            )
            records = await cursor.fetchall()

        return [self._row_to_dict(r) for r in records]

    # ------------------------------------------------------------------
    # Internal: hierarchy validation
    # ------------------------------------------------------------------

    async def _get_hierarchy_depth(self, role_id: str) -> int:
        """Compute the depth of a role in the hierarchy (0 = root).

        Args:
            role_id: UUID of the role.

        Returns:
            Integer depth from root.
        """
        depth = 0
        visited: set[str] = set()
        current_id: Optional[str] = role_id

        async with self._pool.connection() as conn:
            while current_id and depth < self._max_depth + 2:
                if current_id in visited:
                    break
                visited.add(current_id)

                cursor = await conn.execute(
                    "SELECT parent_role_id FROM security.roles WHERE id = %s",
                    (current_id,),
                )
                record = await cursor.fetchone()
                if record is None or record[0] is None:
                    break
                current_id = str(record[0])
                depth += 1

        return depth

    async def _validate_no_cycle(self, role_id: str, new_parent_id: str) -> None:
        """Verify that setting new_parent_id does not create a cycle.

        Walks from ``new_parent_id`` up to the root.  If ``role_id``
        is encountered during the walk, a cycle would be created.

        Args:
            role_id: The role being updated.
            new_parent_id: The proposed new parent.

        Raises:
            RoleHierarchyCycleError: If a cycle would be created.
        """
        if role_id == new_parent_id:
            raise RoleHierarchyCycleError(
                f"Role {role_id} cannot be its own parent"
            )

        visited: set[str] = set()
        current_id: Optional[str] = new_parent_id

        async with self._pool.connection() as conn:
            while current_id and len(visited) < self._max_depth + 2:
                if current_id in visited:
                    break
                if current_id == role_id:
                    raise RoleHierarchyCycleError(
                        f"Setting parent {new_parent_id} on role {role_id} "
                        f"would create a cycle"
                    )
                visited.add(current_id)

                cursor = await conn.execute(
                    "SELECT parent_role_id FROM security.roles WHERE id = %s",
                    (current_id,),
                )
                record = await cursor.fetchone()
                if record is None or record[0] is None:
                    break
                current_id = str(record[0])

    # ------------------------------------------------------------------
    # Internal: cache invalidation
    # ------------------------------------------------------------------

    async def _invalidate_tenant(self, tenant_id: Optional[str]) -> None:
        """Broadcast cache invalidation for a tenant.

        Args:
            tenant_id: UUID of the affected tenant.
        """
        if self._cache is None:
            return

        effective_tenant = tenant_id or "__system__"
        try:
            await self._cache.publish_invalidation(
                event_type="role_changed",
                tenant_id=effective_tenant,
            )
        except Exception as exc:
            logger.warning("Cache invalidation failed: %s", exc)

    # ------------------------------------------------------------------
    # Internal: row mapping
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_dict(record: Any) -> Dict[str, Any]:
        """Convert a database row tuple to a dictionary.

        Args:
            record: Row tuple from psycopg cursor.

        Returns:
            Dictionary with column names as keys.
        """
        if record is None:
            return {}

        return {
            "id": str(record[0]) if record[0] else None,
            "name": record[1],
            "display_name": record[2],
            "description": record[3],
            "tenant_id": str(record[4]) if record[4] else None,
            "parent_role_id": str(record[5]) if record[5] else None,
            "is_system_role": record[6],
            "is_enabled": record[7],
            "metadata": record[8] if record[8] else {},
            "created_at": record[9],
            "created_by": record[10],
            "updated_at": record[11],
        }
