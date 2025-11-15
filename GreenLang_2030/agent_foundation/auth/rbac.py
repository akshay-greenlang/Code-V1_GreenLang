"""
Role-Based Access Control (RBAC) System

This module implements enterprise-grade RBAC for the GreenLang Agent Foundation.
Provides 8 predefined roles with granular permissions, role hierarchy, and
permission checking decorators.

Features:
- 8 predefined roles (super_admin to api_service)
- Resource-based permission model
- Role inheritance and composition
- Permission checking decorators
- Dynamic permission evaluation
- Audit trail for all access decisions

Example:
    >>> rbac = RBACManager()
    >>> rbac.assign_role(user_id="user123", role=Role.ANALYST)
    >>> has_access = rbac.check_permission(user_id="user123", resource="reports", action="read")
"""

from typing import Dict, List, Optional, Set, Callable, Any
from pydantic import BaseModel, Field, validator
from datetime import datetime
from enum import Enum
import logging
from functools import wraps
import asyncio

logger = logging.getLogger(__name__)


class Role(str, Enum):
    """Predefined system roles"""
    SUPER_ADMIN = "super_admin"          # Platform administrator (full access)
    TENANT_ADMIN = "tenant_admin"        # Customer administrator (tenant scope)
    DEVELOPER = "developer"               # Build and deploy agents
    OPERATOR = "operator"                 # Monitor and operate platform
    ANALYST = "analyst"                   # View data and generate reports
    AUDITOR = "auditor"                   # Read-only compliance access
    VIEWER = "viewer"                     # Dashboard viewing only
    API_SERVICE = "api_service"           # Programmatic API access


class Action(str, Enum):
    """Standard CRUD actions"""
    CREATE = "create"
    READ = "read"
    UPDATE = "update"
    DELETE = "delete"
    EXECUTE = "execute"
    ADMIN = "admin"


class ResourceType(str, Enum):
    """System resources"""
    AGENT = "agent"
    WORKFLOW = "workflow"
    DATA = "data"
    REPORT = "report"
    USER = "user"
    ROLE = "role"
    API_KEY = "api_key"
    SECRET = "secret"
    AUDIT_LOG = "audit_log"
    CONFIG = "config"
    DASHBOARD = "dashboard"
    INTEGRATION = "integration"


class Permission(BaseModel):
    """Permission model (resource + action)"""
    resource: ResourceType = Field(..., description="Resource type")
    action: Action = Field(..., description="Action allowed")
    scope: Optional[str] = Field(None, description="Scope limitation (e.g., 'tenant:123')")
    conditions: Dict[str, Any] = Field(default_factory=dict, description="Dynamic conditions")

    def matches(self, resource: ResourceType, action: Action, context: Optional[Dict] = None) -> bool:
        """Check if permission matches request"""
        # Basic match
        if self.resource != resource or self.action != action:
            return False

        # Check conditions
        if self.conditions and context:
            for key, expected_value in self.conditions.items():
                if context.get(key) != expected_value:
                    return False

        return True

    def __str__(self) -> str:
        return f"{self.resource.value}:{self.action.value}"


class RoleDefinition(BaseModel):
    """Role definition with permissions"""
    role: Role = Field(..., description="Role identifier")
    name: str = Field(..., description="Human-readable name")
    description: str = Field(..., description="Role description")
    permissions: List[Permission] = Field(..., description="Granted permissions")
    inherits_from: List[Role] = Field(default_factory=list, description="Parent roles")
    is_system_role: bool = Field(default=True, description="System-defined role")

    def has_permission(self, resource: ResourceType, action: Action, context: Optional[Dict] = None) -> bool:
        """Check if role has specific permission"""
        return any(perm.matches(resource, action, context) for perm in self.permissions)


class UserRole(BaseModel):
    """User role assignment"""
    user_id: str = Field(..., description="User identifier")
    role: Role = Field(..., description="Assigned role")
    tenant_id: Optional[str] = Field(None, description="Tenant scope")
    assigned_at: datetime = Field(default_factory=datetime.utcnow)
    assigned_by: str = Field(..., description="Who assigned the role")
    expires_at: Optional[datetime] = Field(None, description="Role expiration")
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def is_active(self) -> bool:
        """Check if role assignment is active"""
        if self.expires_at and self.expires_at < datetime.utcnow():
            return False
        return True


class PermissionCheck(BaseModel):
    """Permission check result"""
    user_id: str
    resource: ResourceType
    action: Action
    granted: bool
    reason: str
    roles: List[Role]
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# Predefined role permissions
ROLE_PERMISSIONS: Dict[Role, List[Permission]] = {
    Role.SUPER_ADMIN: [
        # Full access to everything
        Permission(resource=ResourceType.AGENT, action=Action.ADMIN),
        Permission(resource=ResourceType.WORKFLOW, action=Action.ADMIN),
        Permission(resource=ResourceType.DATA, action=Action.ADMIN),
        Permission(resource=ResourceType.REPORT, action=Action.ADMIN),
        Permission(resource=ResourceType.USER, action=Action.ADMIN),
        Permission(resource=ResourceType.ROLE, action=Action.ADMIN),
        Permission(resource=ResourceType.API_KEY, action=Action.ADMIN),
        Permission(resource=ResourceType.SECRET, action=Action.ADMIN),
        Permission(resource=ResourceType.AUDIT_LOG, action=Action.READ),
        Permission(resource=ResourceType.CONFIG, action=Action.ADMIN),
        Permission(resource=ResourceType.DASHBOARD, action=Action.ADMIN),
        Permission(resource=ResourceType.INTEGRATION, action=Action.ADMIN),
    ],

    Role.TENANT_ADMIN: [
        # Tenant-scoped administration
        Permission(resource=ResourceType.AGENT, action=Action.CREATE),
        Permission(resource=ResourceType.AGENT, action=Action.READ),
        Permission(resource=ResourceType.AGENT, action=Action.UPDATE),
        Permission(resource=ResourceType.AGENT, action=Action.DELETE),
        Permission(resource=ResourceType.WORKFLOW, action=Action.CREATE),
        Permission(resource=ResourceType.WORKFLOW, action=Action.READ),
        Permission(resource=ResourceType.WORKFLOW, action=Action.UPDATE),
        Permission(resource=ResourceType.WORKFLOW, action=Action.DELETE),
        Permission(resource=ResourceType.DATA, action=Action.READ),
        Permission(resource=ResourceType.DATA, action=Action.CREATE),
        Permission(resource=ResourceType.DATA, action=Action.UPDATE),
        Permission(resource=ResourceType.REPORT, action=Action.READ),
        Permission(resource=ResourceType.REPORT, action=Action.CREATE),
        Permission(resource=ResourceType.USER, action=Action.CREATE),
        Permission(resource=ResourceType.USER, action=Action.READ),
        Permission(resource=ResourceType.USER, action=Action.UPDATE),
        Permission(resource=ResourceType.ROLE, action=Action.READ),
        Permission(resource=ResourceType.API_KEY, action=Action.CREATE),
        Permission(resource=ResourceType.API_KEY, action=Action.READ),
        Permission(resource=ResourceType.API_KEY, action=Action.DELETE),
        Permission(resource=ResourceType.AUDIT_LOG, action=Action.READ),
        Permission(resource=ResourceType.DASHBOARD, action=Action.READ),
        Permission(resource=ResourceType.INTEGRATION, action=Action.READ),
    ],

    Role.DEVELOPER: [
        # Agent development and deployment
        Permission(resource=ResourceType.AGENT, action=Action.CREATE),
        Permission(resource=ResourceType.AGENT, action=Action.READ),
        Permission(resource=ResourceType.AGENT, action=Action.UPDATE),
        Permission(resource=ResourceType.AGENT, action=Action.EXECUTE),
        Permission(resource=ResourceType.WORKFLOW, action=Action.CREATE),
        Permission(resource=ResourceType.WORKFLOW, action=Action.READ),
        Permission(resource=ResourceType.WORKFLOW, action=Action.UPDATE),
        Permission(resource=ResourceType.DATA, action=Action.READ),
        Permission(resource=ResourceType.DATA, action=Action.CREATE),
        Permission(resource=ResourceType.REPORT, action=Action.READ),
        Permission(resource=ResourceType.API_KEY, action=Action.READ),
        Permission(resource=ResourceType.DASHBOARD, action=Action.READ),
        Permission(resource=ResourceType.INTEGRATION, action=Action.READ),
    ],

    Role.OPERATOR: [
        # Operations and monitoring
        Permission(resource=ResourceType.AGENT, action=Action.READ),
        Permission(resource=ResourceType.AGENT, action=Action.EXECUTE),
        Permission(resource=ResourceType.WORKFLOW, action=Action.READ),
        Permission(resource=ResourceType.WORKFLOW, action=Action.EXECUTE),
        Permission(resource=ResourceType.DATA, action=Action.READ),
        Permission(resource=ResourceType.REPORT, action=Action.READ),
        Permission(resource=ResourceType.AUDIT_LOG, action=Action.READ),
        Permission(resource=ResourceType.DASHBOARD, action=Action.READ),
        Permission(resource=ResourceType.INTEGRATION, action=Action.READ),
    ],

    Role.ANALYST: [
        # Data analysis and reporting
        Permission(resource=ResourceType.AGENT, action=Action.READ),
        Permission(resource=ResourceType.WORKFLOW, action=Action.READ),
        Permission(resource=ResourceType.DATA, action=Action.READ),
        Permission(resource=ResourceType.DATA, action=Action.CREATE),
        Permission(resource=ResourceType.REPORT, action=Action.READ),
        Permission(resource=ResourceType.REPORT, action=Action.CREATE),
        Permission(resource=ResourceType.DASHBOARD, action=Action.READ),
    ],

    Role.AUDITOR: [
        # Read-only compliance access
        Permission(resource=ResourceType.AGENT, action=Action.READ),
        Permission(resource=ResourceType.WORKFLOW, action=Action.READ),
        Permission(resource=ResourceType.DATA, action=Action.READ),
        Permission(resource=ResourceType.REPORT, action=Action.READ),
        Permission(resource=ResourceType.USER, action=Action.READ),
        Permission(resource=ResourceType.ROLE, action=Action.READ),
        Permission(resource=ResourceType.AUDIT_LOG, action=Action.READ),
        Permission(resource=ResourceType.DASHBOARD, action=Action.READ),
    ],

    Role.VIEWER: [
        # Dashboard viewing only
        Permission(resource=ResourceType.DASHBOARD, action=Action.READ),
        Permission(resource=ResourceType.REPORT, action=Action.READ),
    ],

    Role.API_SERVICE: [
        # Programmatic API access
        Permission(resource=ResourceType.AGENT, action=Action.CREATE),
        Permission(resource=ResourceType.AGENT, action=Action.READ),
        Permission(resource=ResourceType.AGENT, action=Action.EXECUTE),
        Permission(resource=ResourceType.WORKFLOW, action=Action.CREATE),
        Permission(resource=ResourceType.WORKFLOW, action=Action.READ),
        Permission(resource=ResourceType.WORKFLOW, action=Action.EXECUTE),
        Permission(resource=ResourceType.DATA, action=Action.CREATE),
        Permission(resource=ResourceType.DATA, action=Action.READ),
        Permission(resource=ResourceType.REPORT, action=Action.READ),
    ],
}

# Role inheritance hierarchy
ROLE_INHERITANCE: Dict[Role, List[Role]] = {
    Role.SUPER_ADMIN: [],
    Role.TENANT_ADMIN: [Role.DEVELOPER, Role.OPERATOR, Role.ANALYST],
    Role.DEVELOPER: [Role.ANALYST],
    Role.OPERATOR: [Role.VIEWER],
    Role.ANALYST: [Role.VIEWER],
    Role.AUDITOR: [Role.VIEWER],
    Role.VIEWER: [],
    Role.API_SERVICE: [],
}


class RBACManager:
    """
    Role-Based Access Control Manager

    Manages role assignments, permission checking, and access control decisions.
    Supports role hierarchy and dynamic permission evaluation.

    Attributes:
        role_definitions: Predefined role definitions
        user_roles: User role assignments
        permission_cache: Cached permission checks

    Example:
        >>> rbac = RBACManager()
        >>> rbac.assign_role(user_id="user123", role=Role.ANALYST, assigned_by="admin")
        >>> has_access = rbac.check_permission(
        ...     user_id="user123",
        ...     resource=ResourceType.REPORT,
        ...     action=Action.READ
        ... )
    """

    def __init__(self):
        """Initialize RBAC manager"""
        self.role_definitions: Dict[Role, RoleDefinition] = self._initialize_roles()
        self.user_roles: Dict[str, List[UserRole]] = {}
        self.permission_cache: Dict[str, PermissionCheck] = {}

    def _initialize_roles(self) -> Dict[Role, RoleDefinition]:
        """Initialize predefined roles"""
        definitions = {}

        for role in Role:
            definitions[role] = RoleDefinition(
                role=role,
                name=role.value.replace("_", " ").title(),
                description=self._get_role_description(role),
                permissions=ROLE_PERMISSIONS.get(role, []),
                inherits_from=ROLE_INHERITANCE.get(role, [])
            )

        return definitions

    def _get_role_description(self, role: Role) -> str:
        """Get role description"""
        descriptions = {
            Role.SUPER_ADMIN: "Platform administrator with full system access",
            Role.TENANT_ADMIN: "Customer administrator with tenant-scoped access",
            Role.DEVELOPER: "Build, deploy, and manage agents and workflows",
            Role.OPERATOR: "Monitor and operate the platform",
            Role.ANALYST: "Analyze data and generate reports",
            Role.AUDITOR: "Read-only access for compliance and auditing",
            Role.VIEWER: "View dashboards and reports only",
            Role.API_SERVICE: "Programmatic API access for integrations"
        }
        return descriptions.get(role, "")

    def assign_role(
        self,
        user_id: str,
        role: Role,
        assigned_by: str,
        tenant_id: Optional[str] = None,
        expires_at: Optional[datetime] = None
    ) -> UserRole:
        """
        Assign role to user

        Args:
            user_id: User identifier
            role: Role to assign
            assigned_by: Who is assigning the role
            tenant_id: Tenant scope (optional)
            expires_at: Role expiration (optional)

        Returns:
            UserRole assignment

        Example:
            >>> user_role = rbac.assign_role(
            ...     user_id="user123",
            ...     role=Role.ANALYST,
            ...     assigned_by="admin@example.com"
            ... )
        """
        user_role = UserRole(
            user_id=user_id,
            role=role,
            tenant_id=tenant_id,
            assigned_by=assigned_by,
            expires_at=expires_at
        )

        if user_id not in self.user_roles:
            self.user_roles[user_id] = []

        self.user_roles[user_id].append(user_role)

        # Clear permission cache for this user
        self._clear_user_cache(user_id)

        logger.info(f"Assigned role {role.value} to user {user_id} by {assigned_by}")

        return user_role

    def revoke_role(self, user_id: str, role: Role) -> bool:
        """
        Revoke role from user

        Args:
            user_id: User identifier
            role: Role to revoke

        Returns:
            True if role was revoked, False if not found

        Example:
            >>> rbac.revoke_role(user_id="user123", role=Role.ANALYST)
        """
        if user_id not in self.user_roles:
            return False

        initial_count = len(self.user_roles[user_id])
        self.user_roles[user_id] = [
            ur for ur in self.user_roles[user_id]
            if ur.role != role
        ]

        revoked = len(self.user_roles[user_id]) < initial_count

        if revoked:
            self._clear_user_cache(user_id)
            logger.info(f"Revoked role {role.value} from user {user_id}")

        return revoked

    def get_user_roles(self, user_id: str, tenant_id: Optional[str] = None) -> List[Role]:
        """
        Get active roles for user

        Args:
            user_id: User identifier
            tenant_id: Filter by tenant (optional)

        Returns:
            List of active roles

        Example:
            >>> roles = rbac.get_user_roles(user_id="user123")
            >>> print(roles)
        """
        if user_id not in self.user_roles:
            return []

        roles = []
        for user_role in self.user_roles[user_id]:
            if not user_role.is_active():
                continue

            if tenant_id and user_role.tenant_id != tenant_id:
                continue

            roles.append(user_role.role)

        return roles

    def get_user_permissions(
        self,
        user_id: str,
        tenant_id: Optional[str] = None
    ) -> Set[Permission]:
        """
        Get all permissions for user (including inherited)

        Args:
            user_id: User identifier
            tenant_id: Filter by tenant (optional)

        Returns:
            Set of permissions

        Example:
            >>> permissions = rbac.get_user_permissions(user_id="user123")
        """
        roles = self.get_user_roles(user_id, tenant_id)
        permissions: Set[Permission] = set()

        for role in roles:
            # Add direct permissions
            role_def = self.role_definitions[role]
            permissions.update(role_def.permissions)

            # Add inherited permissions
            for parent_role in role_def.inherits_from:
                parent_def = self.role_definitions[parent_role]
                permissions.update(parent_def.permissions)

        return permissions

    def check_permission(
        self,
        user_id: str,
        resource: ResourceType,
        action: Action,
        tenant_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> PermissionCheck:
        """
        Check if user has permission

        Args:
            user_id: User identifier
            resource: Resource type
            action: Action to perform
            tenant_id: Tenant context (optional)
            context: Additional context for dynamic checks (optional)

        Returns:
            PermissionCheck result

        Example:
            >>> check = rbac.check_permission(
            ...     user_id="user123",
            ...     resource=ResourceType.REPORT,
            ...     action=Action.READ
            ... )
            >>> if check.granted:
            ...     print("Access granted")
        """
        # Check cache first
        cache_key = f"{user_id}:{resource.value}:{action.value}:{tenant_id}"
        if cache_key in self.permission_cache:
            cached = self.permission_cache[cache_key]
            if (datetime.utcnow() - cached.timestamp).total_seconds() < 60:
                return cached

        # Get user roles and permissions
        roles = self.get_user_roles(user_id, tenant_id)
        permissions = self.get_user_permissions(user_id, tenant_id)

        # Check if any permission matches
        granted = any(
            perm.matches(resource, action, context)
            for perm in permissions
        )

        # Create result
        result = PermissionCheck(
            user_id=user_id,
            resource=resource,
            action=action,
            granted=granted,
            reason=f"User has {len(permissions)} permissions from roles: {[r.value for r in roles]}" if granted else "No matching permission",
            roles=roles
        )

        # Cache result
        self.permission_cache[cache_key] = result

        logger.debug(f"Permission check for {user_id}: {resource.value}:{action.value} = {granted}")

        return result

    def require_permission(
        self,
        resource: ResourceType,
        action: Action
    ) -> Callable:
        """
        Decorator to require permission for function

        Args:
            resource: Resource type
            action: Required action

        Returns:
            Decorator function

        Example:
            >>> @rbac.require_permission(ResourceType.REPORT, Action.CREATE)
            ... def create_report(user_id: str, data: dict):
            ...     return {"report_id": "123"}
        """
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                user_id = kwargs.get("user_id") or (args[0] if args else None)
                if not user_id:
                    raise ValueError("user_id required for permission check")

                check = self.check_permission(user_id, resource, action)
                if not check.granted:
                    raise PermissionError(
                        f"User {user_id} does not have permission {resource.value}:{action.value}"
                    )

                return await func(*args, **kwargs)

            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                user_id = kwargs.get("user_id") or (args[0] if args else None)
                if not user_id:
                    raise ValueError("user_id required for permission check")

                check = self.check_permission(user_id, resource, action)
                if not check.granted:
                    raise PermissionError(
                        f"User {user_id} does not have permission {resource.value}:{action.value}"
                    )

                return func(*args, **kwargs)

            # Return appropriate wrapper based on function type
            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            else:
                return sync_wrapper

        return decorator

    def _clear_user_cache(self, user_id: str) -> None:
        """Clear permission cache for user"""
        keys_to_remove = [k for k in self.permission_cache.keys() if k.startswith(f"{user_id}:")]
        for key in keys_to_remove:
            del self.permission_cache[key]


# Global RBAC instance
_rbac_instance: Optional[RBACManager] = None


def get_rbac_manager() -> RBACManager:
    """Get global RBAC manager instance"""
    global _rbac_instance
    if _rbac_instance is None:
        _rbac_instance = RBACManager()
    return _rbac_instance


# Convenience decorators
def require_admin(func: Callable) -> Callable:
    """Require super_admin or tenant_admin role"""
    rbac = get_rbac_manager()

    @wraps(func)
    def wrapper(*args, **kwargs):
        user_id = kwargs.get("user_id") or (args[0] if args else None)
        if not user_id:
            raise ValueError("user_id required")

        roles = rbac.get_user_roles(user_id)
        if Role.SUPER_ADMIN not in roles and Role.TENANT_ADMIN not in roles:
            raise PermissionError(f"User {user_id} is not an administrator")

        return func(*args, **kwargs)

    return wrapper


def require_role(*required_roles: Role) -> Callable:
    """Require specific role(s)"""
    rbac = get_rbac_manager()

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            user_id = kwargs.get("user_id") or (args[0] if args else None)
            if not user_id:
                raise ValueError("user_id required")

            roles = rbac.get_user_roles(user_id)
            if not any(role in roles for role in required_roles):
                raise PermissionError(
                    f"User {user_id} does not have required role(s): {[r.value for r in required_roles]}"
                )

            return func(*args, **kwargs)

        return wrapper

    return decorator


# Example usage
if __name__ == "__main__":
    # Initialize RBAC
    rbac = RBACManager()

    # Assign roles
    rbac.assign_role(
        user_id="user123",
        role=Role.ANALYST,
        assigned_by="admin@example.com"
    )

    rbac.assign_role(
        user_id="admin456",
        role=Role.TENANT_ADMIN,
        assigned_by="system@example.com",
        tenant_id="tenant1"
    )

    # Check permissions
    check1 = rbac.check_permission(
        user_id="user123",
        resource=ResourceType.REPORT,
        action=Action.READ
    )
    print(f"User123 can read reports: {check1.granted}")

    check2 = rbac.check_permission(
        user_id="user123",
        resource=ResourceType.USER,
        action=Action.DELETE
    )
    print(f"User123 can delete users: {check2.granted}")

    # Get user permissions
    permissions = rbac.get_user_permissions(user_id="admin456", tenant_id="tenant1")
    print(f"\nAdmin456 has {len(permissions)} permissions")

    # Use decorator
    @rbac.require_permission(ResourceType.REPORT, Action.CREATE)
    def create_report(user_id: str, report_data: dict):
        return {"report_id": "new-report", "status": "created"}

    # Test decorator
    try:
        result = create_report(user_id="user123", report_data={"title": "Q4 Report"})
        print(f"\nReport created: {result}")
    except PermissionError as e:
        print(f"Permission denied: {e}")
