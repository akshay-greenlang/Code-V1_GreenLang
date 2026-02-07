# -*- coding: utf-8 -*-
"""
RBAC Seeder - RBAC Authorization Service (SEC-002)

Seeds default system roles, standard permissions, and role-permission
mappings into the ``security`` schema tables created by V010.

Used for:
    - Development and testing environments where the database is
      recreated frequently without running Flyway migrations.
    - Programmatic re-seeding when the V010 seed data needs to be
      refreshed (e.g. after a schema-only restore).
    - Integration tests that need a fully populated RBAC catalogue.

All operations are idempotent: running the seeder multiple times
produces the same result thanks to ``ON CONFLICT DO NOTHING``.

Example:
    >>> seeder = RBACSeeder(db_pool)
    >>> result = await seeder.seed_all()
    >>> print(result)
    {'roles': 10, 'permissions': 61, 'role_permissions': 215}

Author: GreenLang Framework Team
Date: February 2026
Status: Production Ready
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Seed data definitions
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SystemRoleDef:
    """Definition of a system role to be seeded."""

    name: str
    display_name: str
    description: str


@dataclass(frozen=True)
class PermissionDef:
    """Definition of a standard permission to be seeded."""

    resource: str
    action: str
    description: str


# ---------------------------------------------------------------------------
# System roles (matches V010 seed exactly)
# ---------------------------------------------------------------------------

SYSTEM_ROLES: List[SystemRoleDef] = [
    SystemRoleDef(
        name="super_admin",
        display_name="Super Administrator",
        description="Full system access across all tenants. Reserved for platform operators.",
    ),
    SystemRoleDef(
        name="admin",
        display_name="Administrator",
        description="Tenant-level administrator with full access to tenant resources.",
    ),
    SystemRoleDef(
        name="manager",
        display_name="Manager",
        description=(
            "Manages agents, emissions, jobs, compliance, and factory "
            "resources within a tenant."
        ),
    ),
    SystemRoleDef(
        name="developer",
        display_name="Developer",
        description=(
            "Develops and configures agents, manages emissions data, "
            "and factory resources."
        ),
    ),
    SystemRoleDef(
        name="operator",
        display_name="Operator",
        description=(
            "Executes agents, calculates emissions, manages jobs, and "
            "operates factory pipelines."
        ),
    ),
    SystemRoleDef(
        name="analyst",
        display_name="Analyst",
        description=(
            "Read-only access to agents and factory, full access to "
            "emissions and compliance data."
        ),
    ),
    SystemRoleDef(
        name="viewer",
        display_name="Viewer",
        description=(
            "Read-only and list access to all resources. Cannot modify "
            "or execute anything."
        ),
    ),
    SystemRoleDef(
        name="auditor",
        display_name="Auditor",
        description=(
            "Compliance auditor with read access to audit logs, sessions, "
            "compliance, and RBAC metadata."
        ),
    ),
    SystemRoleDef(
        name="service_account",
        display_name="Service Account",
        description=(
            "Machine-to-machine identity for automated agent execution "
            "and emissions calculations."
        ),
    ),
    SystemRoleDef(
        name="guest",
        display_name="Guest",
        description="Unauthenticated or minimal-access identity. No default permissions.",
    ),
]

# ---------------------------------------------------------------------------
# Standard permissions (matches V010 seed exactly)
# ---------------------------------------------------------------------------

STANDARD_PERMISSIONS: List[PermissionDef] = [
    # agents
    PermissionDef("agents", "list", "List agents"),
    PermissionDef("agents", "read", "View agent details"),
    PermissionDef("agents", "execute", "Execute an agent"),
    PermissionDef("agents", "configure", "Configure agent settings"),
    PermissionDef("agents", "create", "Create a new agent"),
    PermissionDef("agents", "update", "Update an existing agent"),
    PermissionDef("agents", "delete", "Delete an agent"),
    # emissions
    PermissionDef("emissions", "list", "List emission records"),
    PermissionDef("emissions", "read", "View emission record details"),
    PermissionDef("emissions", "calculate", "Execute emissions calculation"),
    PermissionDef("emissions", "create", "Create emission records"),
    PermissionDef("emissions", "update", "Update emission records"),
    PermissionDef("emissions", "delete", "Delete emission records"),
    PermissionDef("emissions", "export", "Export emission data"),
    # jobs
    PermissionDef("jobs", "list", "List jobs"),
    PermissionDef("jobs", "read", "View job details"),
    PermissionDef("jobs", "create", "Create a new job"),
    PermissionDef("jobs", "cancel", "Cancel a running job"),
    PermissionDef("jobs", "delete", "Delete a job"),
    # compliance
    PermissionDef("compliance", "list", "List compliance reports"),
    PermissionDef("compliance", "read", "View compliance report details"),
    PermissionDef("compliance", "create", "Create compliance reports"),
    PermissionDef("compliance", "update", "Update compliance reports"),
    PermissionDef("compliance", "delete", "Delete compliance reports"),
    PermissionDef("compliance", "approve", "Approve compliance reports"),
    # factory
    PermissionDef("factory", "list", "List factory agents"),
    PermissionDef("factory", "read", "View factory agent details"),
    PermissionDef("factory", "create", "Create factory agent entries"),
    PermissionDef("factory", "update", "Update factory agent entries"),
    PermissionDef("factory", "delete", "Delete factory agent entries"),
    PermissionDef("factory", "execute", "Execute factory agent pipelines"),
    PermissionDef("factory", "metrics", "View factory agent metrics"),
    PermissionDef("factory", "deploy", "Deploy factory agents"),
    PermissionDef("factory", "rollback", "Rollback factory agent deployments"),
    # flags
    PermissionDef("flags", "list", "List feature flags"),
    PermissionDef("flags", "read", "View feature flag details"),
    PermissionDef("flags", "create", "Create feature flags"),
    PermissionDef("flags", "update", "Update feature flags"),
    PermissionDef("flags", "delete", "Delete feature flags"),
    PermissionDef("flags", "evaluate", "Evaluate feature flags"),
    PermissionDef("flags", "rollout", "Manage flag rollout percentages"),
    PermissionDef("flags", "kill", "Activate kill switch on a flag"),
    PermissionDef("flags", "restore", "Restore a killed flag"),
    # admin
    PermissionDef("admin:users", "list", "List users"),
    PermissionDef("admin:users", "read", "View user details"),
    PermissionDef("admin:users", "unlock", "Unlock locked accounts"),
    PermissionDef("admin:users", "revoke", "Revoke user tokens"),
    PermissionDef("admin:users", "reset", "Force password reset"),
    PermissionDef("admin:users", "mfa", "Manage user MFA settings"),
    PermissionDef("admin:sessions", "list", "List active sessions"),
    PermissionDef("admin:sessions", "terminate", "Terminate sessions"),
    PermissionDef("admin:audit", "read", "Read audit logs"),
    PermissionDef("admin:lockouts", "list", "List account lockouts"),
    # rbac
    PermissionDef("rbac:roles", "list", "List RBAC roles"),
    PermissionDef("rbac:roles", "read", "View RBAC role details"),
    PermissionDef("rbac:roles", "create", "Create RBAC roles"),
    PermissionDef("rbac:roles", "update", "Update RBAC roles"),
    PermissionDef("rbac:roles", "delete", "Delete RBAC roles"),
    PermissionDef("rbac:permissions", "list", "List RBAC permissions"),
    PermissionDef("rbac:permissions", "read", "View RBAC permission details"),
    PermissionDef("rbac:assignments", "list", "List role assignments"),
    PermissionDef("rbac:assignments", "read", "View role assignment details"),
    PermissionDef("rbac:assignments", "create", "Assign roles to users"),
    PermissionDef("rbac:assignments", "revoke", "Revoke role assignments"),
]

# ---------------------------------------------------------------------------
# Default role-permission mappings
# ---------------------------------------------------------------------------
# Each key is a role name.  The value is either:
#   - "__all__"   : grant ALL permissions
#   - A list of (resource, action) tuples that should be granted
#   - "__all_except__": a tuple of (list_to_grant_as_all, exclusions)
# ---------------------------------------------------------------------------

_ALL_MARKER = "__all__"

# Mapping from role name to a filter function that selects which
# permissions to grant.  Each function receives (resource, action)
# and returns True if the permission should be granted.

DEFAULT_MAPPINGS: Dict[str, Any] = {
    "super_admin": _ALL_MARKER,
    "admin": "__all_except__",  # handled specially
    "manager": [
        ("agents", None),           # all agents actions
        ("emissions", None),        # all emissions actions
        ("jobs", None),             # all jobs actions
        ("compliance", None),       # all compliance actions
        ("factory", "list"),
        ("factory", "read"),
        ("factory", "execute"),
    ],
    "developer": [
        ("agents", "list"),
        ("agents", "read"),
        ("agents", "execute"),
        ("agents", "configure"),
        ("emissions", None),        # all emissions actions
        ("factory", "list"),
        ("factory", "read"),
        ("factory", "create"),
        ("factory", "update"),
        ("factory", "execute"),
    ],
    "operator": [
        ("agents", "list"),
        ("agents", "read"),
        ("agents", "execute"),
        ("emissions", "list"),
        ("emissions", "read"),
        ("emissions", "calculate"),
        ("jobs", None),             # all jobs actions
        ("factory", "list"),
        ("factory", "read"),
        ("factory", "execute"),
    ],
    "analyst": [
        ("agents", "list"),
        ("agents", "read"),
        ("emissions", None),        # all emissions actions
        ("compliance", "list"),
        ("compliance", "read"),
        ("factory", "list"),
        ("factory", "read"),
        ("factory", "metrics"),
    ],
    "viewer": "__viewer__",         # all list + read permissions
    "auditor": [
        ("admin:audit", "read"),
        ("admin:sessions", "list"),
        ("compliance", None),       # all compliance actions
        ("rbac:roles", "list"),
        ("rbac:permissions", "list"),
        ("rbac:assignments", "list"),
    ],
    "service_account": [
        ("agents", "execute"),
        ("emissions", "calculate"),
        ("factory", "execute"),
    ],
    "guest": [],                    # no permissions
}


# ---------------------------------------------------------------------------
# Seeder class
# ---------------------------------------------------------------------------


class RBACSeeder:
    """Seeds default roles and permissions into the RBAC tables.

    Used for development/testing and as a fallback when V010 migration
    seeding needs to be re-run programmatically.

    All operations are idempotent (use ``ON CONFLICT ... DO NOTHING``).

    Args:
        db_pool: An async ``psycopg_pool.AsyncConnectionPool`` or any
            object exposing an ``async with pool.connection() as conn``
            interface.

    Example:
        >>> seeder = RBACSeeder(db_pool)
        >>> result = await seeder.seed_all()
        >>> print(result)
        {'roles': 10, 'permissions': 61, 'role_permissions': 215}
    """

    def __init__(self, db_pool: Any) -> None:
        """Initialize the RBAC seeder.

        Args:
            db_pool: Async PostgreSQL connection pool.
        """
        self._pool = db_pool

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def seed_all(self) -> Dict[str, int]:
        """Seed roles, permissions, and role-permission mappings.

        Runs all three seed operations in sequence within a single
        transaction for atomicity.

        Returns:
            Dictionary with counts of inserted rows per category.
        """
        logger.info("Starting RBAC seed operation")

        async with self._pool.connection() as conn:
            async with conn.transaction():
                roles_count = await self._seed_roles(conn)
                perms_count = await self._seed_permissions(conn)
                mappings_count = await self._seed_role_permissions(conn)

        result = {
            "roles": roles_count,
            "permissions": perms_count,
            "role_permissions": mappings_count,
        }
        logger.info("RBAC seed complete: %s", result)
        return result

    async def seed_roles(self) -> int:
        """Seed system roles only.

        Returns:
            Number of roles inserted.
        """
        async with self._pool.connection() as conn:
            async with conn.transaction():
                return await self._seed_roles(conn)

    async def seed_permissions(self) -> int:
        """Seed standard permissions only.

        Returns:
            Number of permissions inserted.
        """
        async with self._pool.connection() as conn:
            async with conn.transaction():
                return await self._seed_permissions(conn)

    async def seed_role_permissions(self) -> int:
        """Seed default role-permission mappings only.

        Requires roles and permissions to already exist.

        Returns:
            Number of role-permission mappings inserted.
        """
        async with self._pool.connection() as conn:
            async with conn.transaction():
                return await self._seed_role_permissions(conn)

    # ------------------------------------------------------------------
    # Internal: seed operations
    # ------------------------------------------------------------------

    async def _seed_roles(self, conn: Any) -> int:
        """Insert system roles into security.roles."""
        count = 0
        for role_def in SYSTEM_ROLES:
            result = await conn.execute(
                """
                INSERT INTO security.roles
                    (name, display_name, description, is_system_role, created_by)
                VALUES (%s, %s, %s, true, 'system')
                ON CONFLICT (tenant_id, name) DO NOTHING
                """,
                (role_def.name, role_def.display_name, role_def.description),
            )
            if result.statusmessage and "INSERT 0 1" in result.statusmessage:
                count += 1

        logger.info("Seeded %d system roles (of %d)", count, len(SYSTEM_ROLES))
        return count

    async def _seed_permissions(self, conn: Any) -> int:
        """Insert standard permissions into security.permissions."""
        count = 0
        for perm_def in STANDARD_PERMISSIONS:
            result = await conn.execute(
                """
                INSERT INTO security.permissions
                    (resource, action, description, is_system_permission)
                VALUES (%s, %s, %s, true)
                ON CONFLICT (resource, action) DO NOTHING
                """,
                (perm_def.resource, perm_def.action, perm_def.description),
            )
            if result.statusmessage and "INSERT 0 1" in result.statusmessage:
                count += 1

        logger.info(
            "Seeded %d standard permissions (of %d)",
            count,
            len(STANDARD_PERMISSIONS),
        )
        return count

    async def _seed_role_permissions(self, conn: Any) -> int:
        """Insert default role-permission mappings."""
        total = 0

        for role_name, mapping in DEFAULT_MAPPINGS.items():
            perm_filters = self._resolve_permission_filters(role_name, mapping)
            if perm_filters is None:
                # All permissions
                result = await conn.execute(
                    """
                    INSERT INTO security.role_permissions
                        (role_id, permission_id, effect, granted_by)
                    SELECT r.id, p.id, 'allow', 'system'
                    FROM security.roles r
                    CROSS JOIN security.permissions p
                    WHERE r.name = %s AND r.is_system_role = true
                    ON CONFLICT (role_id, permission_id) DO NOTHING
                    """,
                    (role_name,),
                )
                total += self._extract_row_count(result)

            elif role_name == "admin":
                # All except rbac:roles:delete
                result = await conn.execute(
                    """
                    INSERT INTO security.role_permissions
                        (role_id, permission_id, effect, granted_by)
                    SELECT r.id, p.id, 'allow', 'system'
                    FROM security.roles r
                    CROSS JOIN security.permissions p
                    WHERE r.name = 'admin' AND r.is_system_role = true
                      AND NOT (p.resource = 'rbac:roles' AND p.action = 'delete')
                    ON CONFLICT (role_id, permission_id) DO NOTHING
                    """,
                )
                total += self._extract_row_count(result)

            elif mapping == "__viewer__":
                # All list + read permissions
                result = await conn.execute(
                    """
                    INSERT INTO security.role_permissions
                        (role_id, permission_id, effect, granted_by)
                    SELECT r.id, p.id, 'allow', 'system'
                    FROM security.roles r
                    CROSS JOIN security.permissions p
                    WHERE r.name = %s AND r.is_system_role = true
                      AND p.action IN ('list', 'read')
                    ON CONFLICT (role_id, permission_id) DO NOTHING
                    """,
                    (role_name,),
                )
                total += self._extract_row_count(result)

            elif isinstance(perm_filters, list):
                for resource, action in perm_filters:
                    if action is None:
                        # All actions for this resource
                        result = await conn.execute(
                            """
                            INSERT INTO security.role_permissions
                                (role_id, permission_id, effect, granted_by)
                            SELECT r.id, p.id, 'allow', 'system'
                            FROM security.roles r
                            CROSS JOIN security.permissions p
                            WHERE r.name = %s AND r.is_system_role = true
                              AND p.resource = %s
                            ON CONFLICT (role_id, permission_id) DO NOTHING
                            """,
                            (role_name, resource),
                        )
                    else:
                        # Specific resource:action
                        result = await conn.execute(
                            """
                            INSERT INTO security.role_permissions
                                (role_id, permission_id, effect, granted_by)
                            SELECT r.id, p.id, 'allow', 'system'
                            FROM security.roles r
                            CROSS JOIN security.permissions p
                            WHERE r.name = %s AND r.is_system_role = true
                              AND p.resource = %s AND p.action = %s
                            ON CONFLICT (role_id, permission_id) DO NOTHING
                            """,
                            (role_name, resource, action),
                        )
                    total += self._extract_row_count(result)

        logger.info("Seeded %d role-permission mappings", total)
        return total

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_permission_filters(
        role_name: str,
        mapping: Any,
    ) -> Optional[List[Tuple[str, Optional[str]]]]:
        """Resolve mapping value to a list of permission filters.

        Returns:
            ``None`` for ALL permissions, or a list of
            ``(resource, action_or_none)`` tuples.
        """
        if mapping == _ALL_MARKER:
            return None
        if mapping == "__all_except__":
            # Handled specially in caller
            return None  # pragma: no cover -- sentinel
        if mapping == "__viewer__":
            return mapping  # type: ignore[return-value]
        if isinstance(mapping, list):
            return mapping
        return []

    @staticmethod
    def _extract_row_count(result: Any) -> int:
        """Extract the number of affected rows from a psycopg cursor result.

        Args:
            result: The result from ``conn.execute()``.

        Returns:
            Number of rows inserted (0 if not determinable).
        """
        if result is None:
            return 0
        msg = getattr(result, "statusmessage", "") or ""
        # Format: "INSERT 0 N" where N is the count
        parts = msg.split()
        if len(parts) == 3 and parts[0] == "INSERT":
            try:
                return int(parts[2])
            except ValueError:
                return 0
        return 0
