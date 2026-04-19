# -*- coding: utf-8 -*-
"""
Role-Based Access Control (RBAC) for GreenLang
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Set, Optional, Any
import fnmatch
from greenlang.utilities.determinism import DeterministicClock

logger = logging.getLogger(__name__)


class PermissionAction(Enum):
    """Standard permission actions"""

    CREATE = "create"
    READ = "read"
    UPDATE = "update"
    DELETE = "delete"
    EXECUTE = "execute"
    LIST = "list"
    APPROVE = "approve"
    ADMIN = "admin"
    ALL = "*"


class ResourceType(Enum):
    """Resource types in GreenLang"""

    PIPELINE = "pipeline"
    PACK = "pack"
    DATASET = "dataset"
    MODEL = "model"
    AGENT = "agent"
    WORKFLOW = "workflow"
    TENANT = "tenant"
    USER = "user"
    ROLE = "role"
    API_KEY = "api_key"
    CLUSTER = "cluster"
    NAMESPACE = "namespace"
    SECRET = "secret"
    CONFIG = "config"
    ALL = "*"


@dataclass
class Permission:
    """Permission definition"""

    resource: str  # Resource type or pattern (e.g., "pipeline", "pack:*", "pipeline:carbon-*")
    action: str  # Action or pattern (e.g., "read", "execute", "*")
    scope: Optional[str] = None  # Optional scope (e.g., "tenant:123", "namespace:prod")
    conditions: Dict[str, Any] = field(default_factory=dict)  # Additional conditions

    def matches(
        self, resource: str, action: str, context: Dict[str, Any] = None
    ) -> bool:
        """
        Check if permission matches request

        Args:
            resource: Requested resource
            action: Requested action
            context: Request context for condition evaluation

        Returns:
            True if permission matches
        """
        # Check resource match (with wildcards)
        if not fnmatch.fnmatch(resource, self.resource):
            return False

        # Check action match (with wildcards)
        if not fnmatch.fnmatch(action, self.action):
            return False

        # Check scope if specified
        if self.scope and context:
            scope_parts = self.scope.split(":")
            if len(scope_parts) == 2:
                scope_type, scope_value = scope_parts
                context_value = context.get(scope_type)
                if context_value != scope_value and scope_value != "*":
                    return False

        # Check conditions
        if self.conditions and context:
            for key, expected_value in self.conditions.items():
                actual_value = context.get(key)
                if actual_value != expected_value:
                    return False

        return True

    def to_string(self) -> str:
        """Convert permission to string format"""
        perm_str = f"{self.resource}:{self.action}"
        if self.scope:
            perm_str += f":{self.scope}"
        return perm_str

    @classmethod
    def from_string(cls, perm_str: str) -> "Permission":
        """Create permission from string format"""
        parts = perm_str.split(":")
        if len(parts) < 2:
            raise ValueError(f"Invalid permission string: {perm_str}")

        resource = parts[0]
        action = parts[1]
        scope = parts[2] if len(parts) > 2 else None

        return cls(resource=resource, action=action, scope=scope)


@dataclass
class Role:
    """Role definition"""

    name: str
    description: str = ""
    permissions: List[Permission] = field(default_factory=list)
    parent_roles: List[str] = field(default_factory=list)  # Role inheritance
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    def add_permission(self, permission: Permission):
        """Add permission to role"""
        self.permissions.append(permission)
        self.updated_at = DeterministicClock.utcnow()

    def remove_permission(self, permission: Permission):
        """Remove permission from role"""
        self.permissions = [p for p in self.permissions if p != permission]
        self.updated_at = DeterministicClock.utcnow()

    def has_permission(
        self, resource: str, action: str, context: Dict[str, Any] = None
    ) -> bool:
        """Check if role has permission"""
        for permission in self.permissions:
            if permission.matches(resource, action, context):
                return True
        return False

    def get_all_permissions(
        self, rbac_manager: "RBACManager" = None
    ) -> List[Permission]:
        """Get all permissions including inherited ones"""
        all_permissions = list(self.permissions)

        if rbac_manager and self.parent_roles:
            for parent_role_name in self.parent_roles:
                parent_role = rbac_manager.get_role(parent_role_name)
                if parent_role:
                    all_permissions.extend(
                        parent_role.get_all_permissions(rbac_manager)
                    )

        return all_permissions

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "name": self.name,
            "description": self.description,
            "permissions": [p.to_string() for p in self.permissions],
            "parent_roles": self.parent_roles,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


class RBACManager:
    """RBAC management system"""

    def __init__(self):
        """Initialize RBAC manager"""
        self.roles: Dict[str, Role] = {}
        self.user_roles: Dict[str, Set[str]] = {}  # user_id -> set of role names
        self.resource_policies: Dict[str, List[Dict[str, Any]]] = (
            {}
        )  # Resource-specific policies

        # Initialize default roles
        self._init_default_roles()

        logger.info("RBACManager initialized")

    def _init_default_roles(self):
        """Initialize default system roles"""

        # Super Admin role
        self.create_role(
            "super_admin",
            "Super Administrator with full access",
            [Permission(resource="*", action="*")],
        )

        # Admin role
        self.create_role(
            "admin",
            "Administrator with most permissions",
            [
                Permission(resource="*", action="*", scope="tenant:*"),
                Permission(resource="tenant", action="read"),
                Permission(resource="user", action="*", scope="tenant:*"),
            ],
        )

        # Developer role
        self.create_role(
            "developer",
            "Developer with pipeline and pack permissions",
            [
                Permission(resource="pipeline", action="*"),
                Permission(resource="pack", action="*"),
                Permission(resource="dataset", action="*"),
                Permission(resource="model", action="read"),
                Permission(resource="agent", action="execute"),
            ],
        )

        # Operator role
        self.create_role(
            "operator",
            "Operator with execution permissions",
            [
                Permission(resource="pipeline", action="execute"),
                Permission(resource="pipeline", action="read"),
                Permission(resource="pack", action="read"),
                Permission(resource="dataset", action="read"),
                Permission(resource="cluster", action="read"),
            ],
        )

        # Viewer role
        self.create_role(
            "viewer",
            "Read-only access",
            [
                Permission(resource="*", action="read"),
                Permission(resource="*", action="list"),
            ],
        )

        # Auditor role
        self.create_role(
            "auditor",
            "Audit and compliance access",
            [
                Permission(resource="*", action="read"),
                Permission(resource="audit", action="*"),
                Permission(resource="compliance", action="*"),
            ],
        )

    def create_role(
        self,
        name: str,
        description: str = "",
        permissions: List[Permission] = None,
        parent_roles: List[str] = None,
    ) -> Role:
        """
        Create a new role

        Args:
            name: Role name
            description: Role description
            permissions: List of permissions
            parent_roles: Parent roles for inheritance

        Returns:
            Created role
        """
        role = Role(
            name=name,
            description=description,
            permissions=permissions or [],
            parent_roles=parent_roles or [],
        )

        self.roles[name] = role
        logger.info(f"Created role: {name}")

        return role

    def get_role(self, name: str) -> Optional[Role]:
        """Get role by name"""
        return self.roles.get(name)

    def update_role(self, name: str, updates: Dict[str, Any]) -> Optional[Role]:
        """Update role"""
        role = self.roles.get(name)
        if not role:
            return None

        if "description" in updates:
            role.description = updates["description"]

        if "permissions" in updates:
            role.permissions = updates["permissions"]

        if "parent_roles" in updates:
            role.parent_roles = updates["parent_roles"]

        role.updated_at = DeterministicClock.utcnow()

        logger.info(f"Updated role: {name}")
        return role

    def delete_role(self, name: str) -> bool:
        """Delete role"""
        if name in ["super_admin", "admin", "developer", "operator", "viewer"]:
            logger.warning(f"Cannot delete system role: {name}")
            return False

        if name in self.roles:
            del self.roles[name]

            # Remove from user assignments
            for user_id in list(self.user_roles.keys()):
                self.user_roles[user_id].discard(name)

            logger.info(f"Deleted role: {name}")
            return True

        return False

    def assign_role(self, user_id: str, role_name: str) -> bool:
        """
        Assign role to user

        Args:
            user_id: User ID
            role_name: Role name

        Returns:
            True if successful
        """
        if role_name not in self.roles:
            logger.error(f"Role not found: {role_name}")
            return False

        if user_id not in self.user_roles:
            self.user_roles[user_id] = set()

        self.user_roles[user_id].add(role_name)
        logger.info(f"Assigned role {role_name} to user {user_id}")

        return True

    def revoke_role(self, user_id: str, role_name: str) -> bool:
        """
        Revoke role from user

        Args:
            user_id: User ID
            role_name: Role name

        Returns:
            True if successful
        """
        if user_id in self.user_roles:
            self.user_roles[user_id].discard(role_name)
            logger.info(f"Revoked role {role_name} from user {user_id}")
            return True

        return False

    def get_user_roles(self, user_id: str) -> List[str]:
        """Get roles assigned to user"""
        return list(self.user_roles.get(user_id, set()))

    def get_user_permissions(self, user_id: str) -> List[Permission]:
        """Get all permissions for user"""
        permissions = []

        for role_name in self.user_roles.get(user_id, set()):
            role = self.roles.get(role_name)
            if role:
                permissions.extend(role.get_all_permissions(self))

        # Remove duplicates
        unique_permissions = []
        seen = set()
        for perm in permissions:
            perm_str = perm.to_string()
            if perm_str not in seen:
                seen.add(perm_str)
                unique_permissions.append(perm)

        return unique_permissions

    def check_permission(
        self, user_id: str, resource: str, action: str, context: Dict[str, Any] = None
    ) -> bool:
        """
        Check if user has permission

        Args:
            user_id: User ID
            resource: Resource to access
            action: Action to perform
            context: Request context

        Returns:
            True if user has permission
        """
        # Get user's roles
        role_names = self.user_roles.get(user_id, set())

        # Check each role
        for role_name in role_names:
            role = self.roles.get(role_name)
            if role:
                # Check role permissions (including inherited)
                all_permissions = role.get_all_permissions(self)
                for permission in all_permissions:
                    if permission.matches(resource, action, context):
                        logger.debug(
                            f"User {user_id} granted {action} on {resource} via role {role_name}"
                        )
                        return True

        # Check resource-specific policies
        if resource in self.resource_policies:
            for policy in self.resource_policies[resource]:
                if self._evaluate_policy(policy, user_id, action, context):
                    logger.debug(
                        f"User {user_id} granted {action} on {resource} via policy"
                    )
                    return True

        logger.debug(f"User {user_id} denied {action} on {resource}")
        return False

    def add_resource_policy(self, resource: str, policy: Dict[str, Any]):
        """
        Add resource-specific policy

        Args:
            resource: Resource identifier
            policy: Policy definition
        """
        if resource not in self.resource_policies:
            self.resource_policies[resource] = []

        self.resource_policies[resource].append(policy)
        logger.info(f"Added policy for resource: {resource}")

    def _evaluate_policy(
        self, policy: Dict[str, Any], user_id: str, action: str, context: Dict[str, Any]
    ) -> bool:
        """Evaluate resource policy"""
        # Check if policy applies to user
        if "users" in policy:
            if user_id not in policy["users"] and "*" not in policy["users"]:
                return False

        # Check if policy allows action
        if "actions" in policy:
            if action not in policy["actions"] and "*" not in policy["actions"]:
                return False

        # Check conditions
        if "conditions" in policy:
            for key, expected_value in policy["conditions"].items():
                actual_value = context.get(key) if context else None
                if actual_value != expected_value:
                    return False

        return True


class AccessControl:
    """Access control decorator and utilities"""

    def __init__(self, rbac_manager: RBACManager):
        """
        Initialize access control

        Args:
            rbac_manager: RBAC manager instance
        """
        self.rbac_manager = rbac_manager

    def require_permission(self, resource: str, action: str):
        """
        Decorator to require permission for function

        Args:
            resource: Required resource
            action: Required action
        """

        def decorator(func):
            def wrapper(*args, **kwargs):
                # Extract user_id and context from arguments
                user_id = kwargs.get("user_id")
                context = kwargs.get("context", {})

                if not user_id:
                    raise PermissionError("User ID required for access control")

                # Check permission
                if not self.rbac_manager.check_permission(
                    user_id, resource, action, context
                ):
                    raise PermissionError(
                        f"User {user_id} lacks permission: {resource}:{action}"
                    )

                # Execute function
                return func(*args, **kwargs)

            return wrapper

        return decorator

    def filter_resources(
        self,
        user_id: str,
        resources: List[Any],
        resource_type: str,
        action: str = "read",
    ) -> List[Any]:
        """
        Filter resources based on user permissions

        Args:
            user_id: User ID
            resources: List of resources
            resource_type: Type of resources
            action: Action to check

        Returns:
            Filtered list of resources user can access
        """
        filtered = []

        for resource in resources:
            # Build resource identifier
            if hasattr(resource, "id"):
                resource_id = f"{resource_type}:{resource.id}"
            elif hasattr(resource, "name"):
                resource_id = f"{resource_type}:{resource.name}"
            else:
                resource_id = resource_type

            # Check permission
            if self.rbac_manager.check_permission(user_id, resource_id, action):
                filtered.append(resource)

        return filtered

    def get_allowed_actions(self, user_id: str, resource: str) -> List[str]:
        """
        Get list of allowed actions for user on resource

        Args:
            user_id: User ID
            resource: Resource identifier

        Returns:
            List of allowed actions
        """
        allowed_actions = []
        permissions = self.rbac_manager.get_user_permissions(user_id)

        for permission in permissions:
            if fnmatch.fnmatch(resource, permission.resource):
                if permission.action == "*":
                    # User has all permissions
                    return [a.value for a in PermissionAction]
                else:
                    allowed_actions.append(permission.action)

        return list(set(allowed_actions))  # Remove duplicates
