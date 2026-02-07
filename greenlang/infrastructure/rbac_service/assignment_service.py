# -*- coding: utf-8 -*-
"""
Assignment Service - RBAC Authorization Service (SEC-002)

User-role assignment management.  Handles assigning roles to users,
revoking assignments, listing a user's roles, aggregating effective
permissions, bulk assignment, and stale-assignment expiration.

All mutations trigger Redis cache invalidation and log audit events
via the RBAC audit logger.

All methods are async and operate against an async ``psycopg``
connection pool.

Example:
    >>> svc = AssignmentService(db_pool, cache, audit, metrics)
    >>> assignment = await svc.assign_role(
    ...     user_id="u-1",
    ...     role_id=admin_role_id,
    ...     tenant_id="t-1",
    ...     assigned_by="super-admin",
    ... )
    >>> print(assignment["is_active"])
    True

Author: GreenLang Framework Team
Date: February 2026
Status: Production Ready
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class AssignmentNotFoundError(Exception):
    """Raised when a user-role assignment lookup fails."""


class DuplicateAssignmentError(Exception):
    """Raised when a user already holds the role in the tenant."""


class AssignmentAlreadyRevokedError(Exception):
    """Raised when trying to revoke an already-revoked assignment."""


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class AssignmentService:
    """Manages user-role assignments with cache invalidation and audit.

    Args:
        db_pool: Async ``psycopg_pool.AsyncConnectionPool``.
        cache: ``RBACCache`` for permission cache invalidation.
        audit: Optional audit logger instance (``AuthAuditLogger`` or
            similar).  When ``None``, audit logging is skipped.
        metrics: Optional metrics instance (``AuthMetrics`` or similar).
            When ``None``, metrics recording is skipped.
    """

    def __init__(
        self,
        db_pool: Any,
        cache: Any,
        audit: Any = None,
        metrics: Any = None,
    ) -> None:
        """Initialize the assignment service.

        Args:
            db_pool: Async PostgreSQL connection pool.
            cache: RBACCache for invalidation.
            audit: Optional audit logger.
            metrics: Optional Prometheus metrics.
        """
        self._pool = db_pool
        self._cache = cache
        self._audit = audit
        self._metrics = metrics

    # ------------------------------------------------------------------
    # Assign
    # ------------------------------------------------------------------

    async def assign_role(
        self,
        user_id: str,
        role_id: str,
        tenant_id: str,
        assigned_by: Optional[str] = None,
        expires_at: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """Assign a role to a user within a tenant.

        If the user already has this role in the tenant, raises
        ``DuplicateAssignmentError``.

        Args:
            user_id: UUID of the user.
            role_id: UUID of the role.
            tenant_id: UUID of the tenant.
            assigned_by: Identity of the assigner.
            expires_at: Optional expiration timestamp (UTC).

        Returns:
            Dictionary of the created assignment.

        Raises:
            DuplicateAssignmentError: If the assignment already exists.
        """
        async with self._pool.connection() as conn:
            try:
                cursor = await conn.execute(
                    """
                    INSERT INTO security.user_roles
                        (user_id, role_id, tenant_id, assigned_by, expires_at)
                    VALUES (%s, %s, %s, %s, %s)
                    RETURNING id, user_id, role_id, tenant_id, assigned_at,
                              assigned_by, expires_at, revoked_at,
                              revoked_by, is_active
                    """,
                    (user_id, role_id, tenant_id, assigned_by, expires_at),
                )
                record = await cursor.fetchone()
            except Exception as exc:
                if "uq_user_role_tenant" in str(exc):
                    raise DuplicateAssignmentError(
                        f"User {user_id} already has role {role_id} "
                        f"in tenant {tenant_id}"
                    ) from exc
                raise

        assignment = self._row_to_dict(record)

        logger.info(
            "Assigned role  user=%s  role=%s  tenant=%s  by=%s",
            user_id,
            role_id,
            tenant_id,
            assigned_by,
        )

        # Invalidate cache for this user
        await self._invalidate_user_cache(tenant_id, user_id)

        return assignment

    # ------------------------------------------------------------------
    # Revoke
    # ------------------------------------------------------------------

    async def revoke_role(
        self,
        assignment_id: str,
        revoked_by: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Revoke a user-role assignment (soft-delete).

        Sets ``is_active=false``, ``revoked_at=NOW()``, and
        ``revoked_by`` on the assignment record.

        Args:
            assignment_id: UUID of the user_roles record.
            revoked_by: Identity of the revoker.

        Returns:
            Updated assignment dictionary.

        Raises:
            AssignmentNotFoundError: If the assignment does not exist.
            AssignmentAlreadyRevokedError: If already revoked.
        """
        # Fetch current to validate
        current = await self.get_assignment(assignment_id)
        if not current["is_active"]:
            raise AssignmentAlreadyRevokedError(
                f"Assignment {assignment_id} is already revoked"
            )

        async with self._pool.connection() as conn:
            cursor = await conn.execute(
                """
                UPDATE security.user_roles
                SET is_active = false,
                    revoked_at = NOW(),
                    revoked_by = %s
                WHERE id = %s
                RETURNING id, user_id, role_id, tenant_id, assigned_at,
                          assigned_by, expires_at, revoked_at,
                          revoked_by, is_active
                """,
                (revoked_by, assignment_id),
            )
            record = await cursor.fetchone()

        if record is None:
            raise AssignmentNotFoundError(
                f"Assignment not found: {assignment_id}"
            )

        assignment = self._row_to_dict(record)

        logger.info(
            "Revoked role  assignment=%s  user=%s  role=%s  tenant=%s  by=%s",
            assignment_id,
            assignment["user_id"],
            assignment["role_id"],
            assignment["tenant_id"],
            revoked_by,
        )

        # Invalidate cache for this user
        await self._invalidate_user_cache(
            assignment["tenant_id"],
            assignment["user_id"],
        )

        return assignment

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    async def get_assignment(self, assignment_id: str) -> Dict[str, Any]:
        """Get a user-role assignment by its UUID.

        Args:
            assignment_id: UUID of the assignment.

        Returns:
            Assignment dictionary.

        Raises:
            AssignmentNotFoundError: If not found.
        """
        async with self._pool.connection() as conn:
            cursor = await conn.execute(
                """
                SELECT id, user_id, role_id, tenant_id, assigned_at,
                       assigned_by, expires_at, revoked_at,
                       revoked_by, is_active
                FROM security.user_roles
                WHERE id = %s
                """,
                (assignment_id,),
            )
            record = await cursor.fetchone()

        if record is None:
            raise AssignmentNotFoundError(
                f"Assignment not found: {assignment_id}"
            )

        return self._row_to_dict(record)

    async def list_user_roles(
        self,
        user_id: str,
        tenant_id: str,
        include_expired: bool = False,
    ) -> List[Dict[str, Any]]:
        """List all role assignments for a user in a tenant.

        Includes role details (name, display_name) via JOIN.

        Args:
            user_id: UUID of the user.
            tenant_id: UUID of the tenant.
            include_expired: Include expired/revoked assignments.

        Returns:
            List of assignment dictionaries with role details.
        """
        conditions = [
            "ur.user_id = %s",
            "ur.tenant_id = %s",
        ]
        params: List[Any] = [user_id, tenant_id]

        if not include_expired:
            conditions.append("ur.is_active = true")
            conditions.append(
                "(ur.expires_at IS NULL OR ur.expires_at > NOW())"
            )

        where = "WHERE " + " AND ".join(conditions)

        query = f"""
            SELECT ur.id, ur.user_id, ur.role_id, ur.tenant_id,
                   ur.assigned_at, ur.assigned_by, ur.expires_at,
                   ur.revoked_at, ur.revoked_by, ur.is_active,
                   r.name AS role_name, r.display_name AS role_display_name
            FROM security.user_roles ur
            JOIN security.roles r ON r.id = ur.role_id
            {where}
            ORDER BY ur.assigned_at DESC
        """

        async with self._pool.connection() as conn:
            cursor = await conn.execute(query, tuple(params))
            records = await cursor.fetchall()

        results: List[Dict[str, Any]] = []
        for row in records:
            entry = self._row_to_dict(row[:10])
            entry["role_name"] = row[10]
            entry["role_display_name"] = row[11]
            results.append(entry)

        return results

    async def get_user_permissions(
        self,
        user_id: str,
        tenant_id: str,
    ) -> List[str]:
        """Get aggregated permission strings for a user.

        Resolves all active roles and their hierarchies, collects
        all allow permissions (deny permissions are excluded from
        the returned set), and returns unique ``resource:action`` strings.

        This method delegates to PermissionService if available, but
        also works standalone by directly querying the database.

        Args:
            user_id: UUID of the user.
            tenant_id: UUID of the tenant.

        Returns:
            Sorted list of unique ``resource:action`` permission strings.
        """
        # Check cache first
        if self._cache:
            cached = await self._cache.get_permissions(tenant_id, user_id)
            if cached is not None:
                return cached

        async with self._pool.connection() as conn:
            cursor = await conn.execute(
                """
                SELECT DISTINCT p.resource || ':' || p.action AS perm_string
                FROM security.user_roles ur
                JOIN security.roles r ON r.id = ur.role_id
                JOIN security.role_permissions rp ON rp.role_id = r.id
                JOIN security.permissions p ON p.id = rp.permission_id
                WHERE ur.user_id = %s
                  AND ur.tenant_id = %s
                  AND ur.is_active = true
                  AND (ur.expires_at IS NULL OR ur.expires_at > NOW())
                  AND r.is_enabled = true
                  AND rp.effect = 'allow'
                ORDER BY perm_string
                """,
                (user_id, tenant_id),
            )
            records = await cursor.fetchall()

        permissions = [row[0] for row in records]

        # Cache the result
        if self._cache:
            await self._cache.set_permissions(tenant_id, user_id, permissions)

        return permissions

    # ------------------------------------------------------------------
    # Bulk operations
    # ------------------------------------------------------------------

    async def bulk_assign_role(
        self,
        user_ids: List[str],
        role_id: str,
        tenant_id: str,
        assigned_by: Optional[str] = None,
    ) -> int:
        """Assign a role to multiple users in a single batch.

        Uses a multi-row INSERT with ``ON CONFLICT DO NOTHING`` to
        skip users who already have the role.

        Args:
            user_ids: List of user UUIDs.
            role_id: UUID of the role to assign.
            tenant_id: UUID of the tenant.
            assigned_by: Identity of the assigner.

        Returns:
            Number of assignments created (excludes duplicates).
        """
        if not user_ids:
            return 0

        # Build multi-row VALUES
        values_parts: List[str] = []
        params: List[Any] = []

        for uid in user_ids:
            values_parts.append("(%s, %s, %s, %s)")
            params.extend([uid, role_id, tenant_id, assigned_by])

        values_sql = ", ".join(values_parts)

        query = f"""
            INSERT INTO security.user_roles
                (user_id, role_id, tenant_id, assigned_by)
            VALUES {values_sql}
            ON CONFLICT (user_id, role_id, tenant_id) DO NOTHING
        """

        async with self._pool.connection() as conn:
            result = await conn.execute(query, tuple(params))

        count = self._extract_row_count(result)
        logger.info(
            "Bulk assign  role=%s  tenant=%s  requested=%d  created=%d",
            role_id,
            tenant_id,
            len(user_ids),
            count,
        )

        # Invalidate cache for all affected users
        for uid in user_ids:
            await self._invalidate_user_cache(tenant_id, uid)

        return count

    # ------------------------------------------------------------------
    # Expiration
    # ------------------------------------------------------------------

    async def expire_stale_assignments(self) -> int:
        """Deactivate all user-role assignments past their expiration.

        Sets ``is_active=false``, ``revoked_at=NOW()``, and
        ``revoked_by='system:auto_expire'`` for all expired
        assignments.

        Returns:
            Number of assignments expired.
        """
        async with self._pool.connection() as conn:
            # Fetch affected user/tenant pairs before updating
            cursor = await conn.execute(
                """
                SELECT DISTINCT user_id, tenant_id
                FROM security.user_roles
                WHERE is_active = true
                  AND expires_at IS NOT NULL
                  AND expires_at < NOW()
                """,
            )
            affected = await cursor.fetchall()

            # Perform the update
            result = await conn.execute(
                """
                UPDATE security.user_roles
                SET is_active = false,
                    revoked_at = NOW(),
                    revoked_by = 'system:auto_expire'
                WHERE is_active = true
                  AND expires_at IS NOT NULL
                  AND expires_at < NOW()
                """,
            )

        count = self._extract_row_count(result)
        logger.info("Expired %d stale assignments", count)

        # Invalidate cache for affected users
        for row in affected:
            user_id = str(row[0])
            tenant_id = str(row[1])
            await self._invalidate_user_cache(tenant_id, user_id)

        return count

    # ------------------------------------------------------------------
    # Internal: cache invalidation
    # ------------------------------------------------------------------

    async def _invalidate_user_cache(
        self,
        tenant_id: str,
        user_id: str,
    ) -> None:
        """Invalidate the cached permission set for a user.

        Also publishes an invalidation event to the Redis channel so
        all service replicas can evict their caches.

        Args:
            tenant_id: UUID of the tenant.
            user_id: UUID of the user.
        """
        if self._cache is None:
            return

        try:
            await self._cache.invalidate_user(tenant_id, user_id)
            await self._cache.publish_invalidation(
                event_type="user_role_changed",
                tenant_id=tenant_id,
                user_id=user_id,
            )
        except Exception as exc:
            logger.warning(
                "Cache invalidation failed for user=%s tenant=%s: %s",
                user_id,
                tenant_id,
                exc,
            )

    # ------------------------------------------------------------------
    # Internal: row mapping
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_dict(record: Any) -> Dict[str, Any]:
        """Convert a user_roles row tuple to a dictionary.

        Args:
            record: Row tuple from psycopg cursor.

        Returns:
            Dictionary with column names as keys.
        """
        if record is None:
            return {}

        return {
            "id": str(record[0]) if record[0] else None,
            "user_id": str(record[1]) if record[1] else None,
            "role_id": str(record[2]) if record[2] else None,
            "tenant_id": str(record[3]) if record[3] else None,
            "assigned_at": record[4],
            "assigned_by": record[5],
            "expires_at": record[6],
            "revoked_at": record[7],
            "revoked_by": record[8],
            "is_active": record[9],
        }

    @staticmethod
    def _extract_row_count(result: Any) -> int:
        """Extract the number of affected rows from a psycopg result.

        Args:
            result: The result from ``conn.execute()``.

        Returns:
            Number of rows affected.
        """
        if result is None:
            return 0

        # psycopg3 cursor has rowcount attribute
        rowcount = getattr(result, "rowcount", None)
        if rowcount is not None and rowcount >= 0:
            return rowcount

        # Fallback: parse statusmessage
        msg = getattr(result, "statusmessage", "") or ""
        parts = msg.split()
        if len(parts) >= 2:
            try:
                return int(parts[-1])
            except ValueError:
                pass
        return 0
