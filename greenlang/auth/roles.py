# -*- coding: utf-8 -*-
"""
Role Hierarchy with Inheritance for GreenLang

This module provides a hierarchical role-based access control system with
permission inheritance and conflict resolution.

Role Hierarchy Example:
    Admin (all permissions)
      ├── Manager (manage workflows, agents)
      │   ├── Analyst (read/execute workflows)
      │   └── Operator (execute only)
      └── Viewer (read only)

Features:
    - Hierarchical role trees
    - Permission aggregation from parent roles
    - Conflict resolution (explicit deny wins)
    - Role assignment and revocation
    - Built-in role templates

Author: GreenLang Framework Team - Phase 4
Date: November 2025
Status: Production Ready
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Set, Any
from collections import defaultdict
import uuid

from pydantic import BaseModel, Field, validator

from greenlang.determinism import deterministic_uuid, DeterministicClock
from greenlang.auth.permissions import (
    Permission,
    PermissionEffect,
    PermissionEvaluator,
    create_permission
)

logger = logging.getLogger(__name__)


# ==============================================================================
# Built-in Role Types
# ==============================================================================

class BuiltInRole(str, Enum):
    """Pre-defined roles in GreenLang."""

    # Administrative roles
    SUPER_ADMIN = "super_admin"
    ADMIN = "admin"
    MANAGER = "manager"

    # Operational roles
    OPERATOR = "operator"
    ANALYST = "analyst"
    DEVELOPER = "developer"

    # Read-only roles
    VIEWER = "viewer"
    AUDITOR = "auditor"

    # Special roles
    SERVICE_ACCOUNT = "service_account"
    GUEST = "guest"


# ==============================================================================
# Role Data Models
# ==============================================================================

class Role(BaseModel):
    """
    Role definition with hierarchical support.

    A role is a collection of permissions that can inherit from parent roles.
    """

    role_id: str = Field(
        default_factory=lambda: str(deterministic_uuid(__name__, str(DeterministicClock.now()))),
        description="Unique role identifier"
    )
    name: str = Field(
        ...,
        description="Role name (unique within tenant)"
    )
    display_name: str = Field(
        ...,
        description="Human-readable role name"
    )
    description: str = Field(
        default="",
        description="Role description"
    )
    permissions: List[Permission] = Field(
        default_factory=list,
        description="Permissions directly assigned to this role"
    )
    parent_role_ids: List[str] = Field(
        default_factory=list,
        description="IDs of parent roles (for inheritance)"
    )
    is_built_in: bool = Field(
        default=False,
        description="Whether this is a built-in system role"
    )
    is_enabled: bool = Field(
        default=True,
        description="Whether this role is currently enabled"
    )
    tenant_id: Optional[str] = Field(
        None,
        description="Tenant this role belongs to (None for global)"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional role metadata"
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="When the role was created"
    )
    created_by: Optional[str] = Field(
        None,
        description="User who created the role"
    )
    updated_at: Optional[datetime] = Field(
        None,
        description="When the role was last updated"
    )

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

    @validator('name')
    def validate_name(cls, v):
        """Validate role name format."""
        if not v or len(v) < 2:
            raise ValueError("Role name must be at least 2 characters")
        if not v.replace('_', '').replace('-', '').isalnum():
            raise ValueError("Role name must be alphanumeric (with _ and -)")
        return v.lower()

    def add_permission(self, permission: Permission):
        """Add a permission to this role."""
        self.permissions.append(permission)
        self.updated_at = DeterministicClock.utcnow()

    def remove_permission(self, permission_id: str) -> bool:
        """Remove a permission from this role."""
        original_length = len(self.permissions)
        self.permissions = [p for p in self.permissions if p.permission_id != permission_id]
        if len(self.permissions) < original_length:
            self.updated_at = DeterministicClock.utcnow()
            return True
        return False

    def add_parent(self, parent_role_id: str):
        """Add a parent role for inheritance."""
        if parent_role_id not in self.parent_role_ids:
            self.parent_role_ids.append(parent_role_id)
            self.updated_at = DeterministicClock.utcnow()

    def remove_parent(self, parent_role_id: str) -> bool:
        """Remove a parent role."""
        if parent_role_id in self.parent_role_ids:
            self.parent_role_ids.remove(parent_role_id)
            self.updated_at = DeterministicClock.utcnow()
            return True
        return False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            'role_id': self.role_id,
            'name': self.name,
            'display_name': self.display_name,
            'description': self.description,
            'permissions': [p.to_dict() for p in self.permissions],
            'parent_role_ids': self.parent_role_ids,
            'is_built_in': self.is_built_in,
            'is_enabled': self.is_enabled,
            'tenant_id': self.tenant_id,
            'metadata': self.metadata,
            'created_at': self.created_at.isoformat(),
            'created_by': self.created_by,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Role':
        """Create Role from dictionary."""
        if 'permissions' in data and data['permissions']:
            data['permissions'] = [
                Permission.from_dict(p) if isinstance(p, dict) else p
                for p in data['permissions']
            ]
        if 'created_at' in data and isinstance(data['created_at'], str):
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        if 'updated_at' in data and isinstance(data['updated_at'], str):
            data['updated_at'] = datetime.fromisoformat(data['updated_at'])
        return cls(**data)


@dataclass
class RoleAssignment:
    """Assignment of a role to a user or service."""

    assignment_id: str = field(default_factory=lambda: str(deterministic_uuid(__name__, str(DeterministicClock.now()))))
    role_id: str = ""
    principal_id: str = ""  # User or service account ID
    principal_type: str = "user"  # "user" or "service_account"
    tenant_id: Optional[str] = None
    scope: Optional[str] = None  # Optional scope limitation
    assigned_at: datetime = field(default_factory=datetime.utcnow)
    assigned_by: Optional[str] = None
    expires_at: Optional[datetime] = None

    def is_expired(self) -> bool:
        """Check if assignment has expired."""
        if self.expires_at:
            return DeterministicClock.utcnow() > self.expires_at
        return False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'assignment_id': self.assignment_id,
            'role_id': self.role_id,
            'principal_id': self.principal_id,
            'principal_type': self.principal_type,
            'tenant_id': self.tenant_id,
            'scope': self.scope,
            'assigned_at': self.assigned_at.isoformat(),
            'assigned_by': self.assigned_by,
            'expires_at': self.expires_at.isoformat() if self.expires_at else None
        }


# ==============================================================================
# Role Hierarchy Manager
# ==============================================================================

class RoleHierarchy:
    """
    Manages role hierarchy and permission inheritance.

    Implements:
    - Tree structure for role relationships
    - Permission aggregation from parent roles
    - Conflict resolution (explicit deny wins)
    - Cycle detection
    """

    def __init__(self):
        """Initialize role hierarchy."""
        self._roles: Dict[str, Role] = {}
        self._children_map: Dict[str, Set[str]] = defaultdict(set)
        logger.info("Initialized RoleHierarchy")

    def add_role(self, role: Role):
        """
        Add a role to the hierarchy.

        Args:
            role: Role to add

        Raises:
            ValueError: If adding role would create a cycle
        """
        # Check for cycles
        for parent_id in role.parent_role_ids:
            if self._would_create_cycle(role.role_id, parent_id):
                raise ValueError(
                    f"Adding role {role.name} with parent {parent_id} would create a cycle"
                )

        # Add role
        self._roles[role.role_id] = role

        # Update children map
        for parent_id in role.parent_role_ids:
            self._children_map[parent_id].add(role.role_id)

        logger.info(f"Added role: {role.name} (id={role.role_id})")

    def get_role(self, role_id: str) -> Optional[Role]:
        """Get role by ID."""
        return self._roles.get(role_id)

    def remove_role(self, role_id: str) -> bool:
        """
        Remove a role from the hierarchy.

        Args:
            role_id: Role ID to remove

        Returns:
            True if removed, False if not found
        """
        if role_id not in self._roles:
            return False

        role = self._roles[role_id]

        # Remove from children maps
        for parent_id in role.parent_role_ids:
            self._children_map[parent_id].discard(role_id)

        # Remove from roles
        del self._roles[role_id]

        logger.info(f"Removed role: {role.name} (id={role_id})")
        return True

    def get_effective_permissions(self, role_id: str) -> List[Permission]:
        """
        Get all effective permissions for a role, including inherited permissions.

        Resolution strategy:
        1. Collect all permissions from role and ancestors
        2. Apply conflict resolution (explicit deny wins)

        Args:
            role_id: Role ID

        Returns:
            List of effective permissions
        """
        if role_id not in self._roles:
            return []

        # Collect all permissions from role hierarchy
        all_permissions: Dict[str, Permission] = {}
        deny_permissions: Set[str] = set()

        # BFS traversal up the hierarchy
        visited = set()
        queue = [role_id]

        while queue:
            current_id = queue.pop(0)
            if current_id in visited:
                continue
            visited.add(current_id)

            current_role = self._roles.get(current_id)
            if not current_role or not current_role.is_enabled:
                continue

            # Add permissions from current role
            for perm in current_role.permissions:
                key = f"{perm.resource}:{perm.action}"

                if perm.effect == PermissionEffect.DENY:
                    deny_permissions.add(key)
                    all_permissions[key] = perm
                elif key not in deny_permissions:
                    # Only add allow if not already denied
                    if key not in all_permissions:
                        all_permissions[key] = perm

            # Add parent roles to queue
            queue.extend(current_role.parent_role_ids)

        return list(all_permissions.values())

    def get_role_ancestors(self, role_id: str) -> List[Role]:
        """
        Get all ancestor roles in the hierarchy.

        Args:
            role_id: Role ID

        Returns:
            List of ancestor roles
        """
        if role_id not in self._roles:
            return []

        ancestors = []
        visited = set()
        queue = [role_id]

        while queue:
            current_id = queue.pop(0)
            if current_id in visited or current_id == role_id:
                if current_id != role_id:  # Don't include self
                    ancestors.append(self._roles[current_id])
                continue

            visited.add(current_id)

            current_role = self._roles.get(current_id)
            if current_role:
                queue.extend(current_role.parent_role_ids)

        return ancestors

    def get_role_descendants(self, role_id: str) -> List[Role]:
        """
        Get all descendant roles in the hierarchy.

        Args:
            role_id: Role ID

        Returns:
            List of descendant roles
        """
        if role_id not in self._roles:
            return []

        descendants = []
        visited = set()
        queue = [role_id]

        while queue:
            current_id = queue.pop(0)
            if current_id in visited:
                continue

            visited.add(current_id)

            # Add children
            for child_id in self._children_map.get(current_id, []):
                if child_id not in visited:
                    descendants.append(self._roles[child_id])
                    queue.append(child_id)

        return descendants

    def _would_create_cycle(self, role_id: str, parent_id: str) -> bool:
        """
        Check if adding parent would create a cycle.

        Args:
            role_id: Child role ID
            parent_id: Potential parent role ID

        Returns:
            True if cycle would be created
        """
        if parent_id == role_id:
            return True

        # Check if parent_id is a descendant of role_id
        visited = set()
        queue = [role_id]

        while queue:
            current_id = queue.pop(0)
            if current_id in visited:
                continue

            visited.add(current_id)

            if current_id == parent_id:
                return True

            # Check children
            for child_id in self._children_map.get(current_id, []):
                queue.append(child_id)

        return False

    def list_roles(
        self,
        tenant_id: Optional[str] = None,
        include_built_in: bool = True,
        enabled_only: bool = True
    ) -> List[Role]:
        """
        List roles matching criteria.

        Args:
            tenant_id: Filter by tenant ID
            include_built_in: Include built-in roles
            enabled_only: Only include enabled roles

        Returns:
            List of matching roles
        """
        roles = list(self._roles.values())

        if tenant_id is not None:
            roles = [r for r in roles if r.tenant_id == tenant_id]

        if not include_built_in:
            roles = [r for r in roles if not r.is_built_in]

        if enabled_only:
            roles = [r for r in roles if r.is_enabled]

        return roles

    def get_hierarchy_tree(self) -> Dict[str, Any]:
        """
        Get role hierarchy as a tree structure.

        Returns:
            Dictionary representing the role tree
        """
        # Find root roles (no parents)
        roots = [r for r in self._roles.values() if not r.parent_role_ids]

        def build_tree(role: Role) -> Dict[str, Any]:
            children = [
                build_tree(self._roles[child_id])
                for child_id in self._children_map.get(role.role_id, [])
                if child_id in self._roles
            ]

            return {
                'role_id': role.role_id,
                'name': role.name,
                'display_name': role.display_name,
                'permission_count': len(role.permissions),
                'is_enabled': role.is_enabled,
                'children': children
            }

        return {
            'roots': [build_tree(root) for root in roots],
            'total_roles': len(self._roles)
        }


# ==============================================================================
# Role Manager
# ==============================================================================

class RoleManager:
    """
    High-level role management with assignment tracking.

    Provides:
    - Role CRUD operations
    - Role assignment to users
    - Permission evaluation for users
    - Role hierarchy management
    """

    def __init__(self):
        """Initialize role manager."""
        self.hierarchy = RoleHierarchy()
        self._assignments: Dict[str, RoleAssignment] = {}
        self._user_assignments: Dict[str, Set[str]] = defaultdict(set)  # user_id -> assignment_ids
        self.evaluator = PermissionEvaluator()

        # Initialize built-in roles
        self._initialize_built_in_roles()

        logger.info("Initialized RoleManager")

    def create_role(
        self,
        name: str,
        display_name: str,
        description: str = "",
        permissions: Optional[List[Permission]] = None,
        parent_role_ids: Optional[List[str]] = None,
        tenant_id: Optional[str] = None,
        created_by: Optional[str] = None
    ) -> Role:
        """
        Create a new role.

        Args:
            name: Role name (unique identifier)
            display_name: Human-readable name
            description: Role description
            permissions: Initial permissions
            parent_role_ids: Parent role IDs for inheritance
            tenant_id: Tenant this role belongs to
            created_by: User creating the role

        Returns:
            Created role
        """
        role = Role(
            name=name,
            display_name=display_name,
            description=description,
            permissions=permissions or [],
            parent_role_ids=parent_role_ids or [],
            tenant_id=tenant_id,
            created_by=created_by
        )

        self.hierarchy.add_role(role)
        logger.info(f"Created role: {role.name}")
        return role

    def get_role(self, role_id: str) -> Optional[Role]:
        """Get role by ID."""
        return self.hierarchy.get_role(role_id)

    def get_role_by_name(self, name: str, tenant_id: Optional[str] = None) -> Optional[Role]:
        """Get role by name."""
        for role in self.hierarchy.list_roles(tenant_id=tenant_id):
            if role.name == name.lower():
                return role
        return None

    def update_role(self, role: Role) -> Role:
        """Update existing role."""
        existing = self.hierarchy.get_role(role.role_id)
        if not existing:
            raise ValueError(f"Role {role.role_id} not found")

        if existing.is_built_in:
            raise ValueError("Cannot modify built-in roles")

        role.updated_at = DeterministicClock.utcnow()
        self.hierarchy.add_role(role)  # Updates existing
        logger.info(f"Updated role: {role.name}")
        return role

    def delete_role(self, role_id: str) -> bool:
        """Delete a role."""
        role = self.hierarchy.get_role(role_id)
        if not role:
            return False

        if role.is_built_in:
            raise ValueError("Cannot delete built-in roles")

        # Remove all assignments
        assignments_to_remove = [
            a for a in self._assignments.values() if a.role_id == role_id
        ]
        for assignment in assignments_to_remove:
            self.unassign_role(assignment.assignment_id)

        # Remove from hierarchy
        return self.hierarchy.remove_role(role_id)

    def assign_role(
        self,
        role_id: str,
        principal_id: str,
        principal_type: str = "user",
        tenant_id: Optional[str] = None,
        scope: Optional[str] = None,
        assigned_by: Optional[str] = None,
        expires_at: Optional[datetime] = None
    ) -> RoleAssignment:
        """
        Assign a role to a user or service account.

        Args:
            role_id: Role to assign
            principal_id: User or service account ID
            principal_type: Type of principal ("user" or "service_account")
            tenant_id: Tenant context
            scope: Optional scope limitation
            assigned_by: User making the assignment
            expires_at: Optional expiration time

        Returns:
            Role assignment
        """
        # Verify role exists
        if not self.hierarchy.get_role(role_id):
            raise ValueError(f"Role {role_id} not found")

        assignment = RoleAssignment(
            role_id=role_id,
            principal_id=principal_id,
            principal_type=principal_type,
            tenant_id=tenant_id,
            scope=scope,
            assigned_by=assigned_by,
            expires_at=expires_at
        )

        self._assignments[assignment.assignment_id] = assignment
        self._user_assignments[principal_id].add(assignment.assignment_id)

        logger.info(f"Assigned role {role_id} to {principal_type} {principal_id}")
        return assignment

    def unassign_role(self, assignment_id: str) -> bool:
        """Unassign a role."""
        if assignment_id not in self._assignments:
            return False

        assignment = self._assignments[assignment_id]
        self._user_assignments[assignment.principal_id].discard(assignment_id)
        del self._assignments[assignment_id]

        logger.info(f"Unassigned role assignment {assignment_id}")
        return True

    def get_user_roles(self, principal_id: str, include_expired: bool = False) -> List[Role]:
        """
        Get all roles assigned to a user.

        Args:
            principal_id: User or service account ID
            include_expired: Include expired assignments

        Returns:
            List of assigned roles
        """
        roles = []
        assignment_ids = self._user_assignments.get(principal_id, set())

        for assignment_id in assignment_ids:
            assignment = self._assignments.get(assignment_id)
            if not assignment:
                continue

            if not include_expired and assignment.is_expired():
                continue

            role = self.hierarchy.get_role(assignment.role_id)
            if role and role.is_enabled:
                roles.append(role)

        return roles

    def get_user_permissions(self, principal_id: str) -> List[Permission]:
        """
        Get all effective permissions for a user.

        Aggregates permissions from all assigned roles and their ancestors.

        Args:
            principal_id: User or service account ID

        Returns:
            List of effective permissions
        """
        all_permissions: Dict[str, Permission] = {}
        deny_permissions: Set[str] = set()

        # Get all assigned roles
        roles = self.get_user_roles(principal_id)

        for role in roles:
            # Get effective permissions for each role (includes inherited)
            perms = self.hierarchy.get_effective_permissions(role.role_id)

            for perm in perms:
                key = f"{perm.resource}:{perm.action}"

                if perm.effect == PermissionEffect.DENY:
                    deny_permissions.add(key)
                    all_permissions[key] = perm
                elif key not in deny_permissions:
                    if key not in all_permissions:
                        all_permissions[key] = perm

        return list(all_permissions.values())

    def check_permission(
        self,
        principal_id: str,
        resource: str,
        action: str,
        context: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Check if user has permission to perform action on resource.

        Args:
            principal_id: User or service account ID
            resource: Resource being accessed
            action: Action being performed
            context: Optional context for evaluation

        Returns:
            True if permitted, False otherwise
        """
        permissions = self.get_user_permissions(principal_id)
        result = self.evaluator.evaluate(permissions, resource, action, context)
        return result.allowed

    def _initialize_built_in_roles(self):
        """Initialize built-in system roles."""
        # Super Admin - all permissions
        super_admin = Role(
            name=BuiltInRole.SUPER_ADMIN.value,
            display_name="Super Administrator",
            description="Full system access with all permissions",
            permissions=[
                create_permission("*", "*", metadata={"built_in": True})
            ],
            is_built_in=True
        )
        self.hierarchy.add_role(super_admin)

        # Admin - most permissions
        admin = Role(
            name=BuiltInRole.ADMIN.value,
            display_name="Administrator",
            description="Administrative access to tenant resources",
            permissions=[
                create_permission("agent:*", "*"),
                create_permission("workflow:*", "*"),
                create_permission("data:*", "*"),
                create_permission("user:*", "read"),
                create_permission("role:*", "*")
            ],
            parent_role_ids=[],
            is_built_in=True
        )
        self.hierarchy.add_role(admin)

        # Manager - manage workflows and agents
        manager = Role(
            name=BuiltInRole.MANAGER.value,
            display_name="Manager",
            description="Manage workflows and agents",
            permissions=[
                create_permission("agent:*", "read"),
                create_permission("agent:*", "execute"),
                create_permission("workflow:*", "read"),
                create_permission("workflow:*", "execute"),
                create_permission("workflow:*", "update"),
                create_permission("data:*", "read")
            ],
            parent_role_ids=[],
            is_built_in=True
        )
        self.hierarchy.add_role(manager)

        # Analyst - read and execute
        analyst = Role(
            name=BuiltInRole.ANALYST.value,
            display_name="Analyst",
            description="Read and execute workflows",
            permissions=[
                create_permission("agent:*", "read"),
                create_permission("agent:*", "execute"),
                create_permission("workflow:*", "read"),
                create_permission("workflow:*", "execute"),
                create_permission("data:*", "read")
            ],
            parent_role_ids=[manager.role_id],
            is_built_in=True
        )
        self.hierarchy.add_role(analyst)

        # Viewer - read only
        viewer = Role(
            name=BuiltInRole.VIEWER.value,
            display_name="Viewer",
            description="Read-only access",
            permissions=[
                create_permission("agent:*", "read"),
                create_permission("workflow:*", "read"),
                create_permission("data:*", "read")
            ],
            parent_role_ids=[],
            is_built_in=True
        )
        self.hierarchy.add_role(viewer)

        logger.info("Initialized built-in roles")


__all__ = [
    'BuiltInRole',
    'Role',
    'RoleAssignment',
    'RoleHierarchy',
    'RoleManager'
]
