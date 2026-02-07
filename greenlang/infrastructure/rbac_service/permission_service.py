# -*- coding: utf-8 -*-
"""
Permission Service - RBAC Authorization Service (SEC-002)

Permission CRUD operations and the core permission evaluation engine.
Aggregates permissions across role hierarchies, applies deny-wins
conflict resolution, and supports wildcard matching on resource:action
pairs using ``fnmatch``.

Key method -- ``evaluate_permission()``:
    1. Check cache for pre-computed permission set.
    2. On miss: query DB (user_roles -> roles -> role_permissions -> permissions).
    3. Walk role hierarchy to aggregate inherited permissions.
    4. Apply deny-wins conflict resolution.
    5. Cache the result.
    6. Match requested ``resource:action`` against aggregated set
       (wildcard-aware via ``fnmatch``).

All methods are async and operate against an async ``psycopg``
connection pool.

Example:
    >>> svc = PermissionService(db_pool, cache, config)
    >>> allowed = await svc.evaluate_permission(
    ...     user_id="u-1", tenant_id="t-1",
    ...     resource="agents", action="execute",
    ... )
    >>> print(allowed)
    True

Author: GreenLang Framework Team
Date: February 2026
Status: Production Ready
"""

from __future__ import annotations

import fnmatch
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class PermissionEntry:
    """A resolved permission with its effect.

    Attributes:
        resource: Resource identifier (e.g. ``"agents"``).
        action: Action identifier (e.g. ``"execute"``).
        effect: ``"allow"`` or ``"deny"``.
        conditions: JSONB conditions from role_permissions.
        scope: Optional scope string.
        source_role_id: The role that granted this permission.
        source_role_name: Human-readable name of the source role.
    """

    resource: str
    action: str
    effect: str = "allow"
    conditions: Dict[str, Any] = field(default_factory=dict)
    scope: Optional[str] = None
    source_role_id: Optional[str] = None
    source_role_name: Optional[str] = None

    @property
    def permission_string(self) -> str:
        """Return the ``resource:action`` string."""
        return f"{self.resource}:{self.action}"


@dataclass
class EvaluationResult:
    """Result of a permission evaluation.

    Attributes:
        allowed: Whether the action is allowed.
        matched_permissions: Permissions that matched the request.
        denied_by: The deny permission that blocked access (if any).
        reason: Human-readable explanation.
    """

    allowed: bool
    matched_permissions: List[PermissionEntry] = field(default_factory=list)
    denied_by: Optional[PermissionEntry] = None
    reason: str = ""


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class PermissionNotFoundError(Exception):
    """Raised when a permission lookup fails."""


class DuplicatePermissionError(Exception):
    """Raised when a resource:action pair already exists."""


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class PermissionService:
    """Permission CRUD and evaluation engine.

    Args:
        db_pool: Async ``psycopg_pool.AsyncConnectionPool``.
        cache: ``RBACCache`` instance for permission caching.
        config: ``RBACServiceConfig`` providing ``max_hierarchy_depth``.
    """

    def __init__(self, db_pool: Any, cache: Any, config: Any) -> None:
        """Initialize the permission service.

        Args:
            db_pool: Async PostgreSQL connection pool.
            cache: RBACCache for permission caching.
            config: RBACServiceConfig instance.
        """
        self._pool = db_pool
        self._cache = cache
        self._max_depth = getattr(config, "max_hierarchy_depth", 5)

    # ------------------------------------------------------------------
    # CRUD: Create
    # ------------------------------------------------------------------

    async def create_permission(
        self,
        resource: str,
        action: str,
        description: Optional[str] = None,
        is_system: bool = False,
    ) -> Dict[str, Any]:
        """Create a new permission entry.

        Args:
            resource: Resource identifier.
            action: Action identifier.
            description: Optional description.
            is_system: Whether this is a system permission.

        Returns:
            Dictionary of the created permission.

        Raises:
            DuplicatePermissionError: If the resource:action already exists.
        """
        async with self._pool.connection() as conn:
            try:
                cursor = await conn.execute(
                    """
                    INSERT INTO security.permissions
                        (resource, action, description, is_system_permission)
                    VALUES (%s, %s, %s, %s)
                    RETURNING id, resource, action, description,
                              is_system_permission, created_at
                    """,
                    (resource, action, description, is_system),
                )
                record = await cursor.fetchone()
            except Exception as exc:
                if "uq_permission" in str(exc):
                    raise DuplicatePermissionError(
                        f"Permission already exists: {resource}:{action}"
                    ) from exc
                raise

        perm = self._perm_row_to_dict(record)
        logger.info("Created permission: %s:%s", resource, action)
        return perm

    # ------------------------------------------------------------------
    # CRUD: Read
    # ------------------------------------------------------------------

    async def get_permission(self, permission_id: str) -> Dict[str, Any]:
        """Get a permission by its UUID.

        Args:
            permission_id: UUID of the permission.

        Returns:
            Permission dictionary.

        Raises:
            PermissionNotFoundError: If not found.
        """
        async with self._pool.connection() as conn:
            cursor = await conn.execute(
                """
                SELECT id, resource, action, description,
                       is_system_permission, created_at
                FROM security.permissions
                WHERE id = %s
                """,
                (permission_id,),
            )
            record = await cursor.fetchone()

        if record is None:
            raise PermissionNotFoundError(f"Permission not found: {permission_id}")

        return self._perm_row_to_dict(record)

    async def list_permissions(
        self,
        resource: Optional[str] = None,
        action: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """List permissions with optional filters.

        Args:
            resource: Filter by resource.
            action: Filter by action.

        Returns:
            List of permission dictionaries.
        """
        conditions: List[str] = []
        params: List[Any] = []

        if resource:
            conditions.append("resource = %s")
            params.append(resource)
        if action:
            conditions.append("action = %s")
            params.append(action)

        where = ""
        if conditions:
            where = "WHERE " + " AND ".join(conditions)

        query = f"""
            SELECT id, resource, action, description,
                   is_system_permission, created_at
            FROM security.permissions
            {where}
            ORDER BY resource, action
        """

        async with self._pool.connection() as conn:
            cursor = await conn.execute(query, tuple(params))
            records = await cursor.fetchall()

        return [self._perm_row_to_dict(r) for r in records]

    # ------------------------------------------------------------------
    # Role permission management
    # ------------------------------------------------------------------

    async def get_role_permissions(
        self,
        role_id: str,
        include_inherited: bool = True,
    ) -> List[PermissionEntry]:
        """Get all permissions for a role, optionally including inherited.

        When ``include_inherited`` is True, walks up the parent
        hierarchy and aggregates permissions from all ancestor roles.

        Args:
            role_id: UUID of the role.
            include_inherited: Whether to include parent role permissions.

        Returns:
            List of ``PermissionEntry`` objects.
        """
        role_ids = [role_id]

        if include_inherited:
            ancestors = await self._walk_hierarchy(role_id)
            role_ids.extend(ancestors)

        return await self._fetch_permissions_for_roles(role_ids)

    # ------------------------------------------------------------------
    # Evaluation: core method
    # ------------------------------------------------------------------

    async def evaluate_permission(
        self,
        user_id: str,
        tenant_id: str,
        resource: str,
        action: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> EvaluationResult:
        """Evaluate whether a user has a specific permission.

        Evaluation pipeline:
            1. Check cache for ``(tenant_id, user_id)`` permission set.
            2. On miss: query DB for user's active roles + hierarchy.
            3. Aggregate all permissions (allow/deny).
            4. Apply deny-wins conflict resolution.
            5. Cache the result set.
            6. Check ``resource:action`` against aggregated set
               (wildcard-aware).

        Args:
            user_id: UUID of the user.
            tenant_id: UUID of the tenant.
            resource: Resource being accessed (e.g. ``"agents"``).
            action: Action being performed (e.g. ``"execute"``).
            context: Optional context dict for condition evaluation.

        Returns:
            ``EvaluationResult`` with the decision and details.
        """
        context = context or {}
        requested = f"{resource}:{action}"

        # Step 1: Check cache
        cached = await self._get_cached_permissions(tenant_id, user_id)

        if cached is not None:
            # Fast path: check cached permission strings
            allowed = self._match_permission(requested, cached)
            return EvaluationResult(
                allowed=allowed,
                reason="cache_hit" if allowed else "no_matching_permission (cached)",
            )

        # Step 2: Query DB for all resolved permissions
        all_entries = await self._resolve_user_permissions(user_id, tenant_id)

        # Step 3-4: Separate allow/deny and apply deny-wins
        allow_entries: List[PermissionEntry] = []
        deny_entries: List[PermissionEntry] = []

        for entry in all_entries:
            if entry.effect == "deny":
                deny_entries.append(entry)
            else:
                allow_entries.append(entry)

        # Step 5: Cache the resolved permission strings (allow only)
        permission_strings = [e.permission_string for e in allow_entries]
        await self._set_cached_permissions(tenant_id, user_id, permission_strings)

        # Step 6: Check for explicit deny first
        for deny in deny_entries:
            if self._entry_matches(deny, resource, action, context):
                return EvaluationResult(
                    allowed=False,
                    matched_permissions=[deny],
                    denied_by=deny,
                    reason=f"Denied by {deny.permission_string} "
                    f"from role {deny.source_role_name}",
                )

        # Check for allow
        matching_allows = [
            e
            for e in allow_entries
            if self._entry_matches(e, resource, action, context)
        ]

        if matching_allows:
            return EvaluationResult(
                allowed=True,
                matched_permissions=matching_allows,
                reason=f"Allowed by {len(matching_allows)} permission(s)",
            )

        # Default deny
        return EvaluationResult(
            allowed=False,
            reason=f"No matching permission for {requested}",
        )

    async def get_user_permissions(
        self,
        user_id: str,
        tenant_id: str,
    ) -> List[str]:
        """Get the aggregated permission strings for a user.

        First checks cache, then resolves from DB if needed.

        Args:
            user_id: UUID of the user.
            tenant_id: UUID of the tenant.

        Returns:
            List of ``resource:action`` permission strings.
        """
        # Check cache
        cached = await self._get_cached_permissions(tenant_id, user_id)
        if cached is not None:
            return cached

        # Resolve from DB
        entries = await self._resolve_user_permissions(user_id, tenant_id)
        allow_strings = [
            e.permission_string for e in entries if e.effect == "allow"
        ]

        # Cache
        await self._set_cached_permissions(tenant_id, user_id, allow_strings)
        return allow_strings

    # ------------------------------------------------------------------
    # Internal: DB queries
    # ------------------------------------------------------------------

    async def _resolve_user_permissions(
        self,
        user_id: str,
        tenant_id: str,
    ) -> List[PermissionEntry]:
        """Resolve all permissions for a user from the database.

        Joins user_roles -> roles -> role_permissions -> permissions,
        then walks role hierarchy for each active role.

        Args:
            user_id: UUID of the user.
            tenant_id: UUID of the tenant.

        Returns:
            List of all ``PermissionEntry`` objects (both allow and deny).
        """
        # Fetch active role IDs for the user
        async with self._pool.connection() as conn:
            cursor = await conn.execute(
                """
                SELECT ur.role_id, r.name, r.parent_role_id
                FROM security.user_roles ur
                JOIN security.roles r ON r.id = ur.role_id
                WHERE ur.user_id = %s
                  AND ur.tenant_id = %s
                  AND ur.is_active = true
                  AND (ur.expires_at IS NULL OR ur.expires_at > NOW())
                  AND r.is_enabled = true
                """,
                (user_id, tenant_id),
            )
            role_rows = await cursor.fetchall()

        if not role_rows:
            return []

        # Collect all role IDs including ancestors
        all_role_ids: List[str] = []
        for row in role_rows:
            role_id = str(row[0])
            all_role_ids.append(role_id)
            # Walk hierarchy
            ancestors = await self._walk_hierarchy(role_id)
            all_role_ids.extend(ancestors)

        # Deduplicate
        unique_role_ids = list(dict.fromkeys(all_role_ids))

        return await self._fetch_permissions_for_roles(unique_role_ids)

    async def _fetch_permissions_for_roles(
        self,
        role_ids: List[str],
    ) -> List[PermissionEntry]:
        """Fetch all permission entries for a set of role IDs.

        Args:
            role_ids: List of role UUIDs.

        Returns:
            List of ``PermissionEntry`` objects.
        """
        if not role_ids:
            return []

        # Build placeholders for IN clause
        placeholders = ", ".join(["%s"] * len(role_ids))

        query = f"""
            SELECT p.resource, p.action, rp.effect, rp.conditions,
                   rp.scope, r.id, r.name
            FROM security.role_permissions rp
            JOIN security.permissions p ON p.id = rp.permission_id
            JOIN security.roles r ON r.id = rp.role_id
            WHERE rp.role_id IN ({placeholders})
            ORDER BY rp.effect DESC, p.resource, p.action
        """

        async with self._pool.connection() as conn:
            cursor = await conn.execute(query, tuple(role_ids))
            records = await cursor.fetchall()

        entries: List[PermissionEntry] = []
        for row in records:
            entries.append(
                PermissionEntry(
                    resource=row[0],
                    action=row[1],
                    effect=row[2],
                    conditions=row[3] if row[3] else {},
                    scope=row[4],
                    source_role_id=str(row[5]) if row[5] else None,
                    source_role_name=row[6],
                )
            )

        return entries

    async def _walk_hierarchy(self, role_id: str) -> List[str]:
        """Walk up the role hierarchy and return ancestor role IDs.

        Does not include the starting ``role_id`` itself.

        Args:
            role_id: UUID of the starting role.

        Returns:
            List of ancestor role UUIDs (parent-first).
        """
        ancestors: List[str] = []
        visited: set[str] = {role_id}

        current_id: Optional[str] = role_id
        async with self._pool.connection() as conn:
            for _ in range(self._max_depth):
                cursor = await conn.execute(
                    "SELECT parent_role_id FROM security.roles WHERE id = %s",
                    (current_id,),
                )
                record = await cursor.fetchone()
                if record is None or record[0] is None:
                    break
                parent_id = str(record[0])
                if parent_id in visited:
                    logger.warning(
                        "Cycle in role hierarchy at %s", parent_id
                    )
                    break
                visited.add(parent_id)
                ancestors.append(parent_id)
                current_id = parent_id

        return ancestors

    # ------------------------------------------------------------------
    # Internal: matching
    # ------------------------------------------------------------------

    @staticmethod
    def _match_permission(
        requested: str,
        permission_strings: List[str],
    ) -> bool:
        """Check if a requested permission matches any in the list.

        Uses ``fnmatch`` for wildcard-aware matching.  Both the
        requested string and each permission string may contain
        wildcards.

        Args:
            requested: The ``resource:action`` being requested.
            permission_strings: List of granted permission strings.

        Returns:
            ``True`` if any permission matches.
        """
        for perm in permission_strings:
            # Check both directions for wildcard matching
            if fnmatch.fnmatch(requested, perm):
                return True
            if fnmatch.fnmatch(perm, requested):
                return True
        return False

    @staticmethod
    def _entry_matches(
        entry: PermissionEntry,
        resource: str,
        action: str,
        context: Dict[str, Any],
    ) -> bool:
        """Check if a PermissionEntry matches a resource:action request.

        Supports wildcard patterns via ``fnmatch`` and evaluates
        JSONB conditions if present.

        Args:
            entry: The permission entry to check.
            resource: Requested resource.
            action: Requested action.
            context: Request context for condition evaluation.

        Returns:
            ``True`` if the entry matches.
        """
        # Resource match (wildcard-aware)
        if not fnmatch.fnmatch(resource, entry.resource):
            if not fnmatch.fnmatch(entry.resource, resource):
                return False

        # Action match (wildcard-aware)
        if not fnmatch.fnmatch(action, entry.action):
            if not fnmatch.fnmatch(entry.action, action):
                return False

        # Condition evaluation
        if entry.conditions and context:
            for key, expected_value in entry.conditions.items():
                actual_value = context.get(key)
                if actual_value != expected_value:
                    return False

        return True

    # ------------------------------------------------------------------
    # Internal: caching
    # ------------------------------------------------------------------

    async def _get_cached_permissions(
        self,
        tenant_id: str,
        user_id: str,
    ) -> Optional[List[str]]:
        """Retrieve cached permission strings.

        Args:
            tenant_id: UUID of the tenant.
            user_id: UUID of the user.

        Returns:
            Cached permission list or ``None``.
        """
        if self._cache is None:
            return None
        try:
            return await self._cache.get_permissions(tenant_id, user_id)
        except Exception as exc:
            logger.warning("Cache read failed: %s", exc)
            return None

    async def _set_cached_permissions(
        self,
        tenant_id: str,
        user_id: str,
        permissions: List[str],
    ) -> None:
        """Store permission strings in cache.

        Args:
            tenant_id: UUID of the tenant.
            user_id: UUID of the user.
            permissions: Permission strings to cache.
        """
        if self._cache is None:
            return
        try:
            await self._cache.set_permissions(tenant_id, user_id, permissions)
        except Exception as exc:
            logger.warning("Cache write failed: %s", exc)

    # ------------------------------------------------------------------
    # Internal: row mapping
    # ------------------------------------------------------------------

    @staticmethod
    def _perm_row_to_dict(record: Any) -> Dict[str, Any]:
        """Convert a permission row to a dictionary.

        Args:
            record: Row tuple from psycopg cursor.

        Returns:
            Permission dictionary.
        """
        if record is None:
            return {}

        return {
            "id": str(record[0]) if record[0] else None,
            "resource": record[1],
            "action": record[2],
            "description": record[3],
            "is_system_permission": record[4],
            "created_at": record[5],
        }
