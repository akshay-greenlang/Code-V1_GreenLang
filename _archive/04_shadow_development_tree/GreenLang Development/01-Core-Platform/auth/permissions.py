# -*- coding: utf-8 -*-
"""
Fine-Grained Permission Model for GreenLang

This module provides resource-level access control with a comprehensive permission
evaluation engine. Supports hierarchical resources, wildcard patterns, and complex
permission rules.

Permission Format:
    resource:action (e.g., "agent:read", "workflow:execute", "data:export")

Resource Hierarchy:
    - agent:* (all agent operations)
    - agent:<agent_id>:* (specific agent operations)
    - workflow:* (all workflow operations)
    - data:<dataset_id>:export (specific data export)

Features:
    - Resource-level access control
    - Wildcard pattern matching
    - Permission evaluation engine
    - PostgreSQL storage support
    - Permission caching
    - Attribute-based conditions

Author: GreenLang Framework Team - Phase 4
Date: November 2025
Status: Production Ready
"""

import logging
import fnmatch
import hashlib
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple
from pathlib import Path
import json
import uuid

from pydantic import BaseModel, Field, validator
from greenlang.utilities.determinism import deterministic_uuid, DeterministicClock

logger = logging.getLogger(__name__)


# ==============================================================================
# Permission Actions and Resources
# ==============================================================================

class PermissionAction(str, Enum):
    """Standard permission actions across GreenLang resources."""

    # CRUD operations
    CREATE = "create"
    READ = "read"
    UPDATE = "update"
    DELETE = "delete"

    # Execution operations
    EXECUTE = "execute"
    RUN = "run"
    TRIGGER = "trigger"

    # Data operations
    EXPORT = "export"
    IMPORT = "import"
    SHARE = "share"

    # Administrative operations
    ADMIN = "admin"
    MANAGE = "manage"
    CONFIGURE = "configure"

    # Listing operations
    LIST = "list"
    SEARCH = "search"

    # Approval operations
    APPROVE = "approve"
    REJECT = "reject"

    # Special operations
    ALL = "*"


class ResourceType(str, Enum):
    """Resource types in GreenLang that can be protected by permissions."""

    # Core resources
    AGENT = "agent"
    WORKFLOW = "workflow"
    PIPELINE = "pipeline"

    # Data resources
    DATASET = "dataset"
    DATA = "data"
    MODEL = "model"

    # System resources
    PACK = "pack"
    TENANT = "tenant"
    USER = "user"
    ROLE = "role"

    # Infrastructure resources
    API_KEY = "api_key"
    CLUSTER = "cluster"
    NAMESPACE = "namespace"

    # Configuration resources
    SECRET = "secret"
    CONFIG = "config"
    POLICY = "policy"

    # Special
    ALL = "*"


class PermissionEffect(str, Enum):
    """Effect of a permission rule (allow or deny)."""
    ALLOW = "allow"
    DENY = "deny"


# ==============================================================================
# Permission Data Models
# ==============================================================================

class PermissionCondition(BaseModel):
    """Condition that must be satisfied for permission to apply."""

    attribute: str = Field(..., description="Attribute to check (e.g., 'user.department')")
    operator: str = Field(
        ...,
        description="Comparison operator (eq, ne, in, not_in, gt, lt, gte, lte, contains, matches)"
    )
    value: Any = Field(..., description="Value to compare against")

    @validator('operator')
    def validate_operator(cls, v):
        valid_operators = {'eq', 'ne', 'in', 'not_in', 'gt', 'lt', 'gte', 'lte', 'contains', 'matches'}
        if v not in valid_operators:
            raise ValueError(f"Invalid operator: {v}. Must be one of {valid_operators}")
        return v

    def evaluate(self, context: Dict[str, Any]) -> bool:
        """
        Evaluate condition against context.

        Args:
            context: Context dictionary with attributes

        Returns:
            True if condition is satisfied, False otherwise
        """
        # Get attribute value from nested context
        attr_value = self._get_nested_value(context, self.attribute)

        if attr_value is None:
            return False

        # Evaluate based on operator
        if self.operator == 'eq':
            return attr_value == self.value
        elif self.operator == 'ne':
            return attr_value != self.value
        elif self.operator == 'in':
            return attr_value in self.value
        elif self.operator == 'not_in':
            return attr_value not in self.value
        elif self.operator == 'gt':
            return attr_value > self.value
        elif self.operator == 'lt':
            return attr_value < self.value
        elif self.operator == 'gte':
            return attr_value >= self.value
        elif self.operator == 'lte':
            return attr_value <= self.value
        elif self.operator == 'contains':
            return self.value in attr_value
        elif self.operator == 'matches':
            return fnmatch.fnmatch(str(attr_value), str(self.value))

        return False

    def _get_nested_value(self, context: Dict[str, Any], path: str) -> Any:
        """Get nested value from context using dot notation."""
        parts = path.split('.')
        value = context

        for part in parts:
            if isinstance(value, dict):
                value = value.get(part)
            else:
                return None

            if value is None:
                return None

        return value


class Permission(BaseModel):
    """
    Fine-grained permission definition.

    A permission grants (or denies) the ability to perform an action on a resource,
    optionally subject to conditions.
    """

    permission_id: str = Field(
        default_factory=lambda: str(deterministic_uuid(__name__, str(DeterministicClock.now()))),
        description="Unique permission identifier"
    )
    resource: str = Field(
        ...,
        description="Resource pattern (e.g., 'agent:*', 'workflow:carbon-audit')"
    )
    action: str = Field(
        ...,
        description="Action pattern (e.g., 'read', 'execute', '*')"
    )
    effect: PermissionEffect = Field(
        default=PermissionEffect.ALLOW,
        description="Whether this permission allows or denies access"
    )
    conditions: List[PermissionCondition] = Field(
        default_factory=list,
        description="Conditions that must be satisfied"
    )
    scope: Optional[str] = Field(
        None,
        description="Optional scope (e.g., 'tenant:123', 'namespace:prod')"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata about the permission"
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="When the permission was created"
    )
    created_by: Optional[str] = Field(
        None,
        description="User who created the permission"
    )

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            PermissionEffect: lambda v: v.value
        }

    def matches_request(
        self,
        resource: str,
        action: str,
        context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Check if this permission matches the requested resource and action.

        Args:
            resource: Requested resource (e.g., 'agent:carbon-calculator')
            action: Requested action (e.g., 'execute')
            context: Optional context for condition evaluation

        Returns:
            True if permission matches request
        """
        context = context or {}

        # Check resource pattern match
        if not fnmatch.fnmatch(resource, self.resource):
            return False

        # Check action pattern match
        if not fnmatch.fnmatch(action, self.action):
            return False

        # Check scope if specified
        if self.scope and context:
            scope_parts = self.scope.split(':', 1)
            if len(scope_parts) == 2:
                scope_type, scope_value = scope_parts
                context_value = context.get(scope_type)

                # Support wildcard in scope value
                if scope_value != '*':
                    if context_value != scope_value:
                        return False

        # Evaluate all conditions
        for condition in self.conditions:
            if not condition.evaluate(context):
                return False

        return True

    def to_string(self) -> str:
        """Convert permission to string representation."""
        perm_str = f"{self.effect.value}:{self.resource}:{self.action}"
        if self.scope:
            perm_str += f"@{self.scope}"
        if self.conditions:
            perm_str += f" (with {len(self.conditions)} conditions)"
        return perm_str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            'permission_id': self.permission_id,
            'resource': self.resource,
            'action': self.action,
            'effect': self.effect.value,
            'conditions': [c.dict() for c in self.conditions],
            'scope': self.scope,
            'metadata': self.metadata,
            'created_at': self.created_at.isoformat(),
            'created_by': self.created_by
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Permission':
        """Create Permission from dictionary."""
        if 'conditions' in data and data['conditions']:
            data['conditions'] = [PermissionCondition(**c) if isinstance(c, dict) else c
                                 for c in data['conditions']]
        if 'created_at' in data and isinstance(data['created_at'], str):
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        if 'effect' in data and isinstance(data['effect'], str):
            data['effect'] = PermissionEffect(data['effect'])
        return cls(**data)


# ==============================================================================
# Permission Evaluation Engine
# ==============================================================================

@dataclass
class EvaluationResult:
    """Result of permission evaluation."""

    allowed: bool
    matched_permissions: List[Permission] = field(default_factory=list)
    denied_by: Optional[Permission] = None
    evaluation_time_ms: float = 0.0
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'allowed': self.allowed,
            'matched_permissions': [p.to_string() for p in self.matched_permissions],
            'denied_by': self.denied_by.to_string() if self.denied_by else None,
            'evaluation_time_ms': round(self.evaluation_time_ms, 2),
            'reason': self.reason
        }


class PermissionEvaluator:
    """
    Permission evaluation engine.

    Evaluates whether a given action on a resource is allowed based on a set of
    permissions. Implements "explicit deny wins" conflict resolution.
    """

    def __init__(self, cache_ttl_seconds: int = 300):
        """
        Initialize permission evaluator.

        Args:
            cache_ttl_seconds: TTL for evaluation cache (default 5 minutes)
        """
        self.cache_ttl_seconds = cache_ttl_seconds
        self._evaluation_cache: Dict[str, Tuple[EvaluationResult, datetime]] = {}
        self._stats = {
            'evaluations': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'allows': 0,
            'denies': 0
        }

    def evaluate(
        self,
        permissions: List[Permission],
        resource: str,
        action: str,
        context: Optional[Dict[str, Any]] = None,
        use_cache: bool = True
    ) -> EvaluationResult:
        """
        Evaluate whether action on resource is allowed.

        Resolution strategy:
        1. Explicit DENY wins over ALLOW
        2. If no DENY and at least one ALLOW -> allowed
        3. If no matching permissions -> denied (default deny)

        Args:
            permissions: List of permissions to evaluate
            resource: Resource being accessed
            action: Action being performed
            context: Optional context for condition evaluation
            use_cache: Whether to use evaluation cache

        Returns:
            EvaluationResult with decision and details
        """
        import time
        start_time = time.time()

        self._stats['evaluations'] += 1
        context = context or {}

        # Check cache
        if use_cache:
            cache_key = self._get_cache_key(permissions, resource, action, context)
            cached_result = self._get_from_cache(cache_key)
            if cached_result:
                self._stats['cache_hits'] += 1
                return cached_result
            self._stats['cache_misses'] += 1

        # Find all matching permissions
        matching_allow: List[Permission] = []
        matching_deny: List[Permission] = []

        for perm in permissions:
            if perm.matches_request(resource, action, context):
                if perm.effect == PermissionEffect.DENY:
                    matching_deny.append(perm)
                else:
                    matching_allow.append(perm)

        # Apply resolution strategy: explicit deny wins
        if matching_deny:
            result = EvaluationResult(
                allowed=False,
                matched_permissions=matching_allow + matching_deny,
                denied_by=matching_deny[0],
                evaluation_time_ms=(time.time() - start_time) * 1000,
                reason=f"Explicitly denied by: {matching_deny[0].to_string()}"
            )
            self._stats['denies'] += 1
        elif matching_allow:
            result = EvaluationResult(
                allowed=True,
                matched_permissions=matching_allow,
                evaluation_time_ms=(time.time() - start_time) * 1000,
                reason=f"Allowed by {len(matching_allow)} permission(s)"
            )
            self._stats['allows'] += 1
        else:
            # Default deny
            result = EvaluationResult(
                allowed=False,
                matched_permissions=[],
                evaluation_time_ms=(time.time() - start_time) * 1000,
                reason="No matching permissions (default deny)"
            )
            self._stats['denies'] += 1

        # Cache result
        if use_cache:
            self._put_in_cache(cache_key, result)

        return result

    def batch_evaluate(
        self,
        permissions: List[Permission],
        requests: List[Tuple[str, str, Optional[Dict[str, Any]]]],
        use_cache: bool = True
    ) -> Dict[str, EvaluationResult]:
        """
        Evaluate multiple permission requests in batch.

        Args:
            permissions: List of permissions to evaluate
            requests: List of (resource, action, context) tuples
            use_cache: Whether to use evaluation cache

        Returns:
            Dictionary mapping request key to EvaluationResult
        """
        results = {}

        for i, (resource, action, context) in enumerate(requests):
            key = f"{i}:{resource}:{action}"
            results[key] = self.evaluate(permissions, resource, action, context, use_cache)

        return results

    def clear_cache(self):
        """Clear evaluation cache."""
        self._evaluation_cache.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get evaluator statistics."""
        cache_hit_rate = 0.0
        if self._stats['evaluations'] > 0:
            cache_hit_rate = (self._stats['cache_hits'] / self._stats['evaluations']) * 100

        return {
            **self._stats,
            'cache_hit_rate': round(cache_hit_rate, 2),
            'cache_size': len(self._evaluation_cache)
        }

    def _get_cache_key(
        self,
        permissions: List[Permission],
        resource: str,
        action: str,
        context: Dict[str, Any]
    ) -> str:
        """Generate cache key for evaluation."""
        # Hash permissions
        perm_ids = sorted([p.permission_id for p in permissions])
        perm_hash = hashlib.sha256(json.dumps(perm_ids).encode()).hexdigest()[:16]

        # Hash context
        context_str = json.dumps(context, sort_keys=True)
        context_hash = hashlib.sha256(context_str.encode()).hexdigest()[:16]

        return f"{perm_hash}:{resource}:{action}:{context_hash}"

    def _get_from_cache(self, key: str) -> Optional[EvaluationResult]:
        """Get result from cache if not expired."""
        if key in self._evaluation_cache:
            result, cached_at = self._evaluation_cache[key]
            age = (DeterministicClock.utcnow() - cached_at).total_seconds()

            if age < self.cache_ttl_seconds:
                return result
            else:
                # Expired, remove from cache
                del self._evaluation_cache[key]

        return None

    def _put_in_cache(self, key: str, result: EvaluationResult):
        """Put result in cache."""
        self._evaluation_cache[key] = (result, DeterministicClock.utcnow())


# ==============================================================================
# Permission Storage (PostgreSQL Support)
# ==============================================================================

class PermissionStore:
    """
    Storage layer for permissions with PostgreSQL support.

    Supports:
    - CRUD operations for permissions
    - Querying by resource, action, scope
    - Permission indexing for fast lookup
    """

    def __init__(self, storage_backend: str = "memory", db_config: Optional[Dict[str, Any]] = None):
        """
        Initialize permission store.

        Args:
            storage_backend: Storage backend ("memory" or "postgresql")
            db_config: Database configuration for PostgreSQL backend
        """
        self.storage_backend = storage_backend
        self._memory_store: Dict[str, Permission] = {}
        self._indices: Dict[str, Set[str]] = {
            'by_resource': {},
            'by_action': {},
            'by_scope': {},
            'by_effect': {}
        }

        # Initialize PostgreSQL backend if selected
        if storage_backend == "postgresql":
            if not db_config:
                raise ValueError("Database configuration required for PostgreSQL backend")
            from greenlang.auth.backends.postgresql import PostgreSQLBackend, DatabaseConfig
            self._pg_backend = PostgreSQLBackend(DatabaseConfig(**db_config))
            logger.info("Initialized PostgreSQL backend for PermissionStore")

        logger.info(f"Initialized PermissionStore with backend: {storage_backend}")

    def create(self, permission: Permission) -> Permission:
        """
        Create a new permission.

        Args:
            permission: Permission to create

        Returns:
            Created permission
        """
        if self.storage_backend == "memory":
            self._memory_store[permission.permission_id] = permission
            self._update_indices(permission)
        elif self.storage_backend == "postgresql":
            # Use PostgreSQL backend
            from greenlang.auth.backends.postgresql import PostgreSQLBackend
            if not hasattr(self, '_pg_backend'):
                raise RuntimeError("PostgreSQL backend not initialized")
            return self._pg_backend.create_permission(permission)
        else:
            raise ValueError(f"Unknown storage backend: {self.storage_backend}")

        logger.info(f"Created permission: {permission.to_string()}")
        return permission

    def get(self, permission_id: str) -> Optional[Permission]:
        """Get permission by ID."""
        if self.storage_backend == "memory":
            return self._memory_store.get(permission_id)
        elif self.storage_backend == "postgresql":
            # Use PostgreSQL backend
            from greenlang.auth.backends.postgresql import PostgreSQLBackend
            if not hasattr(self, '_pg_backend'):
                raise RuntimeError("PostgreSQL backend not initialized")
            return self._pg_backend.get_permission(permission_id)
        else:
            raise ValueError(f"Unknown storage backend: {self.storage_backend}")

    def update(self, permission: Permission) -> Permission:
        """Update existing permission."""
        if self.storage_backend == "memory":
            if permission.permission_id not in self._memory_store:
                raise ValueError(f"Permission {permission.permission_id} not found")

            # Remove old indices
            old_perm = self._memory_store[permission.permission_id]
            self._remove_from_indices(old_perm)

            # Update and re-index
            self._memory_store[permission.permission_id] = permission
            self._update_indices(permission)
        elif self.storage_backend == "postgresql":
            # Use PostgreSQL backend
            from greenlang.auth.backends.postgresql import PostgreSQLBackend
            if not hasattr(self, '_pg_backend'):
                raise RuntimeError("PostgreSQL backend not initialized")
            return self._pg_backend.update_permission(permission)
        else:
            raise ValueError(f"Unknown storage backend: {self.storage_backend}")

        logger.info(f"Updated permission: {permission.to_string()}")
        return permission

    def delete(self, permission_id: str) -> bool:
        """Delete permission by ID."""
        if self.storage_backend == "memory":
            if permission_id in self._memory_store:
                perm = self._memory_store[permission_id]
                self._remove_from_indices(perm)
                del self._memory_store[permission_id]
                logger.info(f"Deleted permission: {permission_id}")
                return True
            return False
        elif self.storage_backend == "postgresql":
            # Use PostgreSQL backend
            from greenlang.auth.backends.postgresql import PostgreSQLBackend
            if not hasattr(self, '_pg_backend'):
                raise RuntimeError("PostgreSQL backend not initialized")
            return self._pg_backend.delete_permission(permission_id)
        else:
            raise ValueError(f"Unknown storage backend: {self.storage_backend}")

    def list(
        self,
        resource_pattern: Optional[str] = None,
        action_pattern: Optional[str] = None,
        scope: Optional[str] = None,
        effect: Optional[PermissionEffect] = None
    ) -> List[Permission]:
        """
        List permissions matching criteria.

        Args:
            resource_pattern: Filter by resource pattern
            action_pattern: Filter by action pattern
            scope: Filter by scope
            effect: Filter by effect

        Returns:
            List of matching permissions
        """
        if self.storage_backend == "memory":
            permissions = list(self._memory_store.values())

            # Apply filters
            if resource_pattern:
                permissions = [p for p in permissions if fnmatch.fnmatch(p.resource, resource_pattern)]
            if action_pattern:
                permissions = [p for p in permissions if fnmatch.fnmatch(p.action, action_pattern)]
            if scope:
                permissions = [p for p in permissions if p.scope == scope]
            if effect:
                permissions = [p for p in permissions if p.effect == effect]

            return permissions
        elif self.storage_backend == "postgresql":
            # Use PostgreSQL backend
            from greenlang.auth.backends.postgresql import PostgreSQLBackend
            if not hasattr(self, '_pg_backend'):
                raise RuntimeError("PostgreSQL backend not initialized")
            return self._pg_backend.list_permissions(
                resource_pattern=resource_pattern,
                action_pattern=action_pattern,
                scope=scope,
                effect=effect
            )
        else:
            raise ValueError(f"Unknown storage backend: {self.storage_backend}")

    def _update_indices(self, permission: Permission):
        """Update search indices for permission."""
        # Index by resource
        if permission.resource not in self._indices['by_resource']:
            self._indices['by_resource'][permission.resource] = set()
        self._indices['by_resource'][permission.resource].add(permission.permission_id)

        # Index by action
        if permission.action not in self._indices['by_action']:
            self._indices['by_action'][permission.action] = set()
        self._indices['by_action'][permission.action].add(permission.permission_id)

        # Index by scope
        if permission.scope:
            if permission.scope not in self._indices['by_scope']:
                self._indices['by_scope'][permission.scope] = set()
            self._indices['by_scope'][permission.scope].add(permission.permission_id)

        # Index by effect
        effect_key = permission.effect.value
        if effect_key not in self._indices['by_effect']:
            self._indices['by_effect'][effect_key] = set()
        self._indices['by_effect'][effect_key].add(permission.permission_id)

    def _remove_from_indices(self, permission: Permission):
        """Remove permission from search indices."""
        # Remove from resource index
        if permission.resource in self._indices['by_resource']:
            self._indices['by_resource'][permission.resource].discard(permission.permission_id)

        # Remove from action index
        if permission.action in self._indices['by_action']:
            self._indices['by_action'][permission.action].discard(permission.permission_id)

        # Remove from scope index
        if permission.scope and permission.scope in self._indices['by_scope']:
            self._indices['by_scope'][permission.scope].discard(permission.permission_id)

        # Remove from effect index
        effect_key = permission.effect.value
        if effect_key in self._indices['by_effect']:
            self._indices['by_effect'][effect_key].discard(permission.permission_id)


# ==============================================================================
# Convenience Functions
# ==============================================================================

def create_permission(
    resource: str,
    action: str,
    effect: PermissionEffect = PermissionEffect.ALLOW,
    conditions: Optional[List[Dict[str, Any]]] = None,
    scope: Optional[str] = None,
    **kwargs
) -> Permission:
    """
    Convenience function to create a permission.

    Args:
        resource: Resource pattern
        action: Action pattern
        effect: Permission effect (allow or deny)
        conditions: List of condition dictionaries
        scope: Optional scope
        **kwargs: Additional permission attributes

    Returns:
        Created Permission object
    """
    condition_objs = []
    if conditions:
        condition_objs = [PermissionCondition(**c) for c in conditions]

    return Permission(
        resource=resource,
        action=action,
        effect=effect,
        conditions=condition_objs,
        scope=scope,
        **kwargs
    )


def parse_permission_string(perm_string: str) -> Permission:
    """
    Parse permission from string format.

    Format: [effect:]resource:action[@scope]
    Example: "allow:agent:*:execute@tenant:123"

    Args:
        perm_string: Permission string

    Returns:
        Permission object
    """
    # Extract scope if present
    scope = None
    if '@' in perm_string:
        perm_string, scope = perm_string.split('@', 1)

    # Split into parts
    parts = perm_string.split(':')

    # Determine effect and resource/action
    if len(parts) == 3:
        effect_str, resource, action = parts
        effect = PermissionEffect(effect_str)
    elif len(parts) == 2:
        resource, action = parts
        effect = PermissionEffect.ALLOW
    else:
        raise ValueError(f"Invalid permission string format: {perm_string}")

    return Permission(
        resource=resource,
        action=action,
        effect=effect,
        scope=scope
    )


__all__ = [
    'PermissionAction',
    'ResourceType',
    'PermissionEffect',
    'PermissionCondition',
    'Permission',
    'EvaluationResult',
    'PermissionEvaluator',
    'PermissionStore',
    'create_permission',
    'parse_permission_string'
]
