# -*- coding: utf-8 -*-
"""
GreenLang RBAC Service - SEC-002: RBAC Authorization Layer

Production-grade role-based access control service built on top of the
``security`` schema created by V009 (auth) and V010 (RBAC).  Provides
hierarchical roles, fine-grained permissions, deny-wins evaluation,
Redis-backed caching with pub/sub invalidation, and full audit trail.

Depends on:
    - SEC-001 (JWT Authentication) for AuthContext and token service.
    - V009 migration for the ``security`` schema.
    - V010 migration for RBAC tables and seed data.

Sub-modules:
    role_service        - CRUD for hierarchical roles.
    permission_service  - Permission CRUD + evaluation engine.
    assignment_service  - User-role assignment management.
    rbac_cache          - Redis permission cache with pub/sub.
    rbac_seeder         - Programmatic seeder for roles/permissions.

Quick start:
    >>> from greenlang.infrastructure.rbac_service import (
    ...     RBACServiceConfig,
    ...     RoleService,
    ...     PermissionService,
    ...     AssignmentService,
    ...     RBACCache,
    ...     RBACSeeder,
    ... )
    >>> config = RBACServiceConfig()
    >>> cache = RBACCache(redis_client, config)
    >>> roles = RoleService(db_pool, cache, config)
    >>> permissions = PermissionService(db_pool, cache, config)
    >>> assignments = AssignmentService(db_pool, cache)

Author: GreenLang Framework Team
Date: February 2026
Status: Production Ready
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class RBACServiceConfig:
    """Top-level configuration for the RBAC Authorization Service (SEC-002).

    Attributes:
        cache_ttl: TTL in seconds for cached permission sets in Redis.
            Default 300 (5 minutes).
        max_hierarchy_depth: Maximum depth of the role hierarchy tree.
            Prevents infinite recursion from malformed data.
            Default 5.
        max_roles_per_tenant: Maximum number of custom roles a tenant
            can create.  System roles do not count toward this limit.
            Default 200.
        max_permissions: Maximum number of custom permissions.  System
            permissions do not count.  Default 1000.
        redis_key_prefix: Prefix for all RBAC-related Redis keys.
            Default ``"gl:rbac"``.
        invalidation_channel: Redis pub/sub channel for cache
            invalidation events.  Default ``"gl:rbac:invalidate"``.
    """

    cache_ttl: int = 300
    max_hierarchy_depth: int = 5
    max_roles_per_tenant: int = 200
    max_permissions: int = 1000
    redis_key_prefix: str = "gl:rbac"
    invalidation_channel: str = "gl:rbac:invalidate"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

from greenlang.infrastructure.rbac_service.role_service import (  # noqa: E402
    RoleService,
    RoleNotFoundError,
    SystemRoleProtectionError,
    RoleHierarchyCycleError,
    RoleHierarchyDepthError,
    DuplicateRoleError,
)
from greenlang.infrastructure.rbac_service.permission_service import (  # noqa: E402
    PermissionService,
    PermissionEntry,
    EvaluationResult,
    PermissionNotFoundError,
    DuplicatePermissionError,
)
from greenlang.infrastructure.rbac_service.assignment_service import (  # noqa: E402
    AssignmentService,
    AssignmentNotFoundError,
    DuplicateAssignmentError,
    AssignmentAlreadyRevokedError,
)
from greenlang.infrastructure.rbac_service.rbac_cache import (  # noqa: E402
    RBACCache,
)
from greenlang.infrastructure.rbac_service.rbac_seeder import (  # noqa: E402
    RBACSeeder,
    SYSTEM_ROLES,
    STANDARD_PERMISSIONS,
)

__all__ = [
    # Config
    "RBACServiceConfig",
    # Role Service
    "RoleService",
    "RoleNotFoundError",
    "SystemRoleProtectionError",
    "RoleHierarchyCycleError",
    "RoleHierarchyDepthError",
    "DuplicateRoleError",
    # Permission Service
    "PermissionService",
    "PermissionEntry",
    "EvaluationResult",
    "PermissionNotFoundError",
    "DuplicatePermissionError",
    # Assignment Service
    "AssignmentService",
    "AssignmentNotFoundError",
    "DuplicateAssignmentError",
    "AssignmentAlreadyRevokedError",
    # Cache
    "RBACCache",
    # Seeder
    "RBACSeeder",
    "SYSTEM_ROLES",
    "STANDARD_PERMISSIONS",
]
