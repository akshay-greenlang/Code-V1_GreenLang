"""
GraphQL Authentication Middleware

Implements JWT-based authentication and role-based authorization
for the GreenLang GraphQL API.

Features:
- JWT token verification
- Role-based access control (RBAC)
- Permission-based authorization
- Multi-tenant isolation
- API key support

Example:
    @require_permission(Permission.AGENT_EXECUTE)
    async def run_agent(info: Info, id: str, input: JSON):
        ...
"""

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Set

from strawberry.extensions import Extension
from strawberry.types import Info

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================


# JWT Configuration (should be loaded from environment/secrets in production)
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "greenlang-dev-secret-key-change-in-production")
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = int(os.getenv("JWT_EXPIRATION_HOURS", "24"))


# =============================================================================
# Enums
# =============================================================================


class Permission(str, Enum):
    """API permissions for authorization."""

    # Agent permissions
    AGENT_READ = "agent:read"
    AGENT_WRITE = "agent:write"
    AGENT_EXECUTE = "agent:execute"
    AGENT_CONFIGURE = "agent:configure"
    AGENT_DELETE = "agent:delete"

    # Calculation permissions
    CALCULATION_READ = "calculation:read"
    CALCULATION_WRITE = "calculation:write"
    CALCULATION_DELETE = "calculation:delete"

    # Admin permissions
    ADMIN_READ = "admin:read"
    ADMIN_WRITE = "admin:write"
    ADMIN_USERS = "admin:users"
    ADMIN_TENANTS = "admin:tenants"

    # Audit permissions
    AUDIT_READ = "audit:read"
    AUDIT_EXPORT = "audit:export"

    # Subscription permissions
    SUBSCRIPTION_EVENTS = "subscription:events"
    SUBSCRIPTION_METRICS = "subscription:metrics"


class Role(str, Enum):
    """User roles with associated permissions."""

    VIEWER = "viewer"           # Read-only access
    OPERATOR = "operator"       # Execute agents
    ANALYST = "analyst"         # Read + execute + calculations
    DEVELOPER = "developer"     # Full agent management
    ADMIN = "admin"             # Full access
    SUPER_ADMIN = "super_admin" # Cross-tenant access


# Role to permission mapping
ROLE_PERMISSIONS: Dict[Role, Set[Permission]] = {
    Role.VIEWER: {
        Permission.AGENT_READ,
        Permission.CALCULATION_READ,
        Permission.SUBSCRIPTION_EVENTS,
    },
    Role.OPERATOR: {
        Permission.AGENT_READ,
        Permission.AGENT_EXECUTE,
        Permission.CALCULATION_READ,
        Permission.SUBSCRIPTION_EVENTS,
        Permission.SUBSCRIPTION_METRICS,
    },
    Role.ANALYST: {
        Permission.AGENT_READ,
        Permission.AGENT_EXECUTE,
        Permission.CALCULATION_READ,
        Permission.CALCULATION_WRITE,
        Permission.AUDIT_READ,
        Permission.SUBSCRIPTION_EVENTS,
        Permission.SUBSCRIPTION_METRICS,
    },
    Role.DEVELOPER: {
        Permission.AGENT_READ,
        Permission.AGENT_WRITE,
        Permission.AGENT_EXECUTE,
        Permission.AGENT_CONFIGURE,
        Permission.CALCULATION_READ,
        Permission.CALCULATION_WRITE,
        Permission.AUDIT_READ,
        Permission.SUBSCRIPTION_EVENTS,
        Permission.SUBSCRIPTION_METRICS,
    },
    Role.ADMIN: {
        Permission.AGENT_READ,
        Permission.AGENT_WRITE,
        Permission.AGENT_EXECUTE,
        Permission.AGENT_CONFIGURE,
        Permission.AGENT_DELETE,
        Permission.CALCULATION_READ,
        Permission.CALCULATION_WRITE,
        Permission.CALCULATION_DELETE,
        Permission.ADMIN_READ,
        Permission.ADMIN_WRITE,
        Permission.ADMIN_USERS,
        Permission.AUDIT_READ,
        Permission.AUDIT_EXPORT,
        Permission.SUBSCRIPTION_EVENTS,
        Permission.SUBSCRIPTION_METRICS,
    },
    Role.SUPER_ADMIN: set(Permission),  # All permissions
}


# =============================================================================
# Auth Context
# =============================================================================


@dataclass
class AuthContext:
    """
    Authentication context for GraphQL requests.

    Contains user identity, roles, permissions, and tenant information.
    """

    # Identity
    user_id: Optional[str] = None
    email: Optional[str] = None
    name: Optional[str] = None

    # Multi-tenancy
    tenant_id: str = "default"
    tenant_name: Optional[str] = None

    # Authorization
    roles: List[Role] = field(default_factory=list)
    permissions: Set[Permission] = field(default_factory=set)

    # Authentication method
    auth_method: str = "none"  # jwt, api_key, anonymous
    authenticated: bool = False

    # Token info
    token_exp: Optional[datetime] = None
    token_iat: Optional[datetime] = None

    # Request context
    request_id: Optional[str] = None
    ip_address: Optional[str] = None

    def has_permission(self, permission: Permission) -> bool:
        """Check if context has a specific permission."""
        return permission in self.permissions

    def has_any_permission(self, permissions: List[Permission]) -> bool:
        """Check if context has any of the specified permissions."""
        return any(p in self.permissions for p in permissions)

    def has_all_permissions(self, permissions: List[Permission]) -> bool:
        """Check if context has all specified permissions."""
        return all(p in self.permissions for p in permissions)

    def has_role(self, role: Role) -> bool:
        """Check if context has a specific role."""
        return role in self.roles

    def is_super_admin(self) -> bool:
        """Check if context has super admin role."""
        return Role.SUPER_ADMIN in self.roles


# =============================================================================
# JWT Token Functions
# =============================================================================


def create_jwt_token(
    user_id: str,
    email: str,
    tenant_id: str,
    roles: List[str],
    name: Optional[str] = None,
    expiration_hours: int = JWT_EXPIRATION_HOURS,
) -> str:
    """
    Create a JWT token for a user.

    Args:
        user_id: User ID
        email: User email
        tenant_id: Tenant ID
        roles: List of role names
        name: Optional user name
        expiration_hours: Token expiration in hours

    Returns:
        JWT token string
    """
    try:
        import jwt
    except ImportError:
        logger.error("PyJWT not installed, cannot create tokens")
        raise ImportError("PyJWT is required for JWT authentication")

    now = datetime.now(timezone.utc)
    exp = now + timedelta(hours=expiration_hours)

    payload = {
        "sub": user_id,
        "email": email,
        "name": name,
        "tenant_id": tenant_id,
        "roles": roles,
        "iat": now,
        "exp": exp,
        "iss": "greenlang-api",
    }

    return jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)


def verify_jwt_token(token: str) -> Optional[Dict[str, Any]]:
    """
    Verify and decode a JWT token.

    Args:
        token: JWT token string

    Returns:
        Decoded payload or None if invalid
    """
    try:
        import jwt
    except ImportError:
        logger.error("PyJWT not installed, cannot verify tokens")
        return None

    try:
        payload = jwt.decode(
            token,
            JWT_SECRET_KEY,
            algorithms=[JWT_ALGORITHM],
            options={"require": ["exp", "sub", "tenant_id"]},
        )
        return payload

    except jwt.ExpiredSignatureError:
        logger.warning("JWT token expired")
        return None

    except jwt.InvalidTokenError as e:
        logger.warning(f"Invalid JWT token: {e}")
        return None


def verify_api_key(api_key: str) -> Optional[Dict[str, Any]]:
    """
    Verify an API key and return associated user info.

    Args:
        api_key: API key string

    Returns:
        User info dict or None if invalid
    """
    # In production, this would validate against database
    # For development, accept keys with specific prefix
    if api_key.startswith("gl_dev_"):
        return {
            "user_id": "api-user",
            "email": "api@greenlang.io",
            "tenant_id": "default",
            "roles": ["operator"],
        }

    if api_key.startswith("gl_admin_"):
        return {
            "user_id": "admin-user",
            "email": "admin@greenlang.io",
            "tenant_id": "default",
            "roles": ["admin"],
        }

    return None


# =============================================================================
# Context Factory
# =============================================================================


async def get_context_with_auth(request) -> AuthContext:
    """
    Create AuthContext from HTTP request.

    Extracts authentication from:
    1. Authorization header (Bearer token)
    2. X-API-Key header
    3. Cookie (for web clients)

    Args:
        request: FastAPI/Starlette request

    Returns:
        AuthContext with authentication state
    """
    import uuid

    context = AuthContext(
        request_id=str(uuid.uuid4()),
        ip_address=getattr(request.client, "host", None) if hasattr(request, "client") else None,
    )

    # Try Authorization header (JWT)
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        token = auth_header[7:]
        payload = verify_jwt_token(token)

        if payload:
            context.authenticated = True
            context.auth_method = "jwt"
            context.user_id = payload.get("sub")
            context.email = payload.get("email")
            context.name = payload.get("name")
            context.tenant_id = payload.get("tenant_id", "default")
            context.token_exp = datetime.fromtimestamp(payload.get("exp", 0), tz=timezone.utc)
            context.token_iat = datetime.fromtimestamp(payload.get("iat", 0), tz=timezone.utc)

            # Convert role strings to Role enums
            role_names = payload.get("roles", [])
            context.roles = [Role(r) for r in role_names if r in Role.__members__.values()]

            # Compute permissions from roles
            for role in context.roles:
                context.permissions.update(ROLE_PERMISSIONS.get(role, set()))

            logger.debug(f"JWT auth successful: user={context.user_id}, tenant={context.tenant_id}")
            return context

    # Try X-API-Key header
    api_key = request.headers.get("X-API-Key", "")
    if api_key:
        user_info = verify_api_key(api_key)

        if user_info:
            context.authenticated = True
            context.auth_method = "api_key"
            context.user_id = user_info.get("user_id")
            context.email = user_info.get("email")
            context.tenant_id = user_info.get("tenant_id", "default")

            role_names = user_info.get("roles", [])
            context.roles = [Role(r) for r in role_names if r in Role.__members__.values()]

            for role in context.roles:
                context.permissions.update(ROLE_PERMISSIONS.get(role, set()))

            logger.debug(f"API key auth successful: user={context.user_id}")
            return context

    # Try cookie (for web clients)
    token_cookie = request.cookies.get("greenlang_token")
    if token_cookie:
        payload = verify_jwt_token(token_cookie)

        if payload:
            context.authenticated = True
            context.auth_method = "cookie"
            context.user_id = payload.get("sub")
            context.email = payload.get("email")
            context.tenant_id = payload.get("tenant_id", "default")

            role_names = payload.get("roles", [])
            context.roles = [Role(r) for r in role_names if r in Role.__members__.values()]

            for role in context.roles:
                context.permissions.update(ROLE_PERMISSIONS.get(role, set()))

            return context

    # No authentication - anonymous access
    logger.debug("Anonymous access - no authentication provided")
    context.auth_method = "anonymous"

    # Grant basic read permissions for anonymous users in development
    if os.getenv("ENVIRONMENT", "development") == "development":
        context.permissions = {Permission.AGENT_READ, Permission.CALCULATION_READ}

    return context


# =============================================================================
# Authorization Decorators
# =============================================================================


def require_permission(permission: Permission):
    """
    Decorator to require a specific permission for a resolver.

    Args:
        permission: Required permission

    Example:
        @require_permission(Permission.AGENT_EXECUTE)
        async def run_agent(info: Info, ...):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Find info in args or kwargs
            info = None
            for arg in args:
                if isinstance(arg, Info):
                    info = arg
                    break
            if not info:
                info = kwargs.get("info")

            if not info:
                raise PermissionError("Cannot verify permissions without GraphQL info")

            context: AuthContext = info.context

            if not context.has_permission(permission):
                logger.warning(
                    f"Permission denied: user={context.user_id}, "
                    f"required={permission.value}, has={[p.value for p in context.permissions]}"
                )
                raise PermissionError(f"Permission denied: {permission.value} required")

            return await func(*args, **kwargs)

        return wrapper
    return decorator


def require_role(role: Role):
    """
    Decorator to require a specific role for a resolver.

    Args:
        role: Required role

    Example:
        @require_role(Role.ADMIN)
        async def delete_agent(info: Info, ...):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            info = None
            for arg in args:
                if isinstance(arg, Info):
                    info = arg
                    break
            if not info:
                info = kwargs.get("info")

            if not info:
                raise PermissionError("Cannot verify role without GraphQL info")

            context: AuthContext = info.context

            if not context.has_role(role):
                logger.warning(
                    f"Role denied: user={context.user_id}, "
                    f"required={role.value}, has={[r.value for r in context.roles]}"
                )
                raise PermissionError(f"Role denied: {role.value} required")

            return await func(*args, **kwargs)

        return wrapper
    return decorator


def require_authenticated(func: Callable) -> Callable:
    """
    Decorator to require authentication for a resolver.

    Example:
        @require_authenticated
        async def my_profile(info: Info):
            ...
    """
    @wraps(func)
    async def wrapper(*args, **kwargs):
        info = None
        for arg in args:
            if isinstance(arg, Info):
                info = arg
                break
        if not info:
            info = kwargs.get("info")

        if not info:
            raise PermissionError("Cannot verify authentication without GraphQL info")

        context: AuthContext = info.context

        if not context.authenticated:
            raise PermissionError("Authentication required")

        return await func(*args, **kwargs)

    return wrapper


# =============================================================================
# Strawberry Extension
# =============================================================================


class AuthMiddleware(Extension):
    """
    Strawberry extension for authentication.

    Validates authentication on every request and injects
    AuthContext into the GraphQL context.
    """

    async def on_request_start(self) -> None:
        """Called at the start of each request."""
        request = self.execution_context.context.get("request")

        if request:
            auth_context = await get_context_with_auth(request)
            # Update the context with auth info
            self.execution_context.context["auth"] = auth_context
            self.execution_context.context["tenant_id"] = auth_context.tenant_id
            self.execution_context.context["user_id"] = auth_context.user_id

    async def on_request_end(self) -> None:
        """Called at the end of each request."""
        pass

    def on_operation(self) -> None:
        """Called before executing an operation."""
        # Could add operation-level auth checks here
        pass
