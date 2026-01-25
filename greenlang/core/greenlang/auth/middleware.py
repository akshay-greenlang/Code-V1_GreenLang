# -*- coding: utf-8 -*-
"""
Authentication Middleware for GreenLang FastAPI Applications

Provides middleware and dependencies for authenticating requests
using JWT tokens or API keys.

Features:
- JWT token validation from Authorization header
- API key validation from X-API-Key header
- Tenant context injection
- Role-based access control decorators
- Permission-based access control decorators
- Request context enrichment

Security Compliance:
- SOC 2 CC6.1 (Logical Access)
- ISO 27001 A.9.4 (System and Application Access Control)
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Set
from functools import wraps
from contextvars import ContextVar

logger = logging.getLogger(__name__)

# Context variable for current authenticated user
_auth_context: ContextVar[Optional["AuthContext"]] = ContextVar(
    "auth_context", default=None
)


@dataclass
class AuthContext:
    """
    Authentication context for the current request.

    This is injected into the request after successful authentication
    and provides access to user identity and permissions.
    """

    # Identity
    user_id: str
    tenant_id: str

    # Authentication method
    auth_method: str  # "jwt" or "api_key"
    auth_token_id: Optional[str] = None  # JTI for JWT, key_id for API key

    # Access control
    roles: List[str] = field(default_factory=list)
    permissions: List[str] = field(default_factory=list)
    scopes: List[str] = field(default_factory=list)

    # Optional user info
    email: Optional[str] = None
    name: Optional[str] = None
    org_id: Optional[str] = None

    # Request metadata
    client_ip: Optional[str] = None
    user_agent: Optional[str] = None

    def has_role(self, role: str) -> bool:
        """Check if context has specified role"""
        return role in self.roles

    def has_any_role(self, roles: List[str]) -> bool:
        """Check if context has any of the specified roles"""
        return bool(set(self.roles) & set(roles))

    def has_all_roles(self, roles: List[str]) -> bool:
        """Check if context has all specified roles"""
        return set(roles).issubset(set(self.roles))

    def has_permission(self, permission: str) -> bool:
        """Check if context has specified permission"""
        return permission in self.permissions

    def has_any_permission(self, permissions: List[str]) -> bool:
        """Check if context has any of the specified permissions"""
        return bool(set(self.permissions) & set(permissions))

    def has_scope(self, scope: str) -> bool:
        """Check if context has specified scope (for API keys)"""
        return scope in self.scopes or "admin" in self.scopes

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "user_id": self.user_id,
            "tenant_id": self.tenant_id,
            "auth_method": self.auth_method,
            "roles": self.roles,
            "permissions": self.permissions,
            "scopes": self.scopes,
            "email": self.email,
            "name": self.name,
            "org_id": self.org_id,
        }


def get_current_user() -> Optional[AuthContext]:
    """
    Get the current authenticated user context.

    Returns:
        AuthContext or None if not authenticated
    """
    return _auth_context.get()


def set_auth_context(context: Optional[AuthContext]) -> None:
    """Set the authentication context for the current request"""
    _auth_context.set(context)


# FastAPI Integration
try:
    from fastapi import Request, HTTPException, Depends
    from fastapi.security import HTTPBearer, APIKeyHeader
    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.responses import JSONResponse

    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    logger.warning("FastAPI not available. Install with: pip install fastapi")


if FASTAPI_AVAILABLE:
    # Security schemes
    bearer_scheme = HTTPBearer(auto_error=False)
    api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

    class JWTAuthBackend:
        """
        JWT Authentication backend for FastAPI.

        Validates JWT tokens from the Authorization header.
        """

        def __init__(self, jwt_handler: Any):
            """
            Initialize JWT auth backend.

            Args:
                jwt_handler: JWTHandler instance
            """
            self.jwt_handler = jwt_handler

        async def authenticate(self, request: Request) -> Optional[AuthContext]:
            """
            Authenticate request using JWT token.

            Args:
                request: FastAPI request

            Returns:
                AuthContext if authenticated, None otherwise
            """
            # Get authorization header
            auth_header = request.headers.get("Authorization")
            if not auth_header:
                return None

            # Parse Bearer token
            parts = auth_header.split()
            if len(parts) != 2 or parts[0].lower() != "bearer":
                return None

            token = parts[1]

            try:
                # Import here to avoid circular dependency
                from .jwt_handler import (
                    JWTClaims,
                    TokenExpiredError,
                    InvalidTokenError,
                    InvalidSignatureError,
                )

                # Validate token
                claims = self.jwt_handler.validate_token(token)

                # Create auth context
                return AuthContext(
                    user_id=claims.sub,
                    tenant_id=claims.tenant_id,
                    auth_method="jwt",
                    auth_token_id=claims.jti,
                    roles=claims.roles,
                    permissions=claims.permissions,
                    email=claims.email,
                    name=claims.name,
                    org_id=claims.org_id,
                    client_ip=self._get_client_ip(request),
                    user_agent=request.headers.get("User-Agent"),
                )

            except Exception as e:
                logger.debug(f"JWT authentication failed: {e}")
                return None

        def _get_client_ip(self, request: Request) -> str:
            """Extract client IP from request"""
            # Check X-Forwarded-For header (for proxied requests)
            forwarded = request.headers.get("X-Forwarded-For")
            if forwarded:
                return forwarded.split(",")[0].strip()
            return request.client.host if request.client else "unknown"

    class APIKeyAuthBackend:
        """
        API Key Authentication backend for FastAPI.

        Validates API keys from the X-API-Key header.
        """

        def __init__(self, api_key_manager: Any):
            """
            Initialize API key auth backend.

            Args:
                api_key_manager: APIKeyManager instance
            """
            self.api_key_manager = api_key_manager

        async def authenticate(
            self,
            request: Request,
            required_scopes: Optional[List[str]] = None,
        ) -> Optional[AuthContext]:
            """
            Authenticate request using API key.

            Args:
                request: FastAPI request
                required_scopes: Required scopes for this request

            Returns:
                AuthContext if authenticated, None otherwise
            """
            # Get API key from header
            api_key = request.headers.get("X-API-Key")
            if not api_key:
                return None

            try:
                # Import here to avoid circular dependency
                from .api_key_manager import (
                    InvalidAPIKeyError,
                    ExpiredAPIKeyError,
                    RateLimitExceededError,
                )

                # Get client info
                client_ip = self._get_client_ip(request)
                origin = request.headers.get("Origin")

                # Validate API key
                record = self.api_key_manager.validate_api_key(
                    api_key,
                    required_scopes=required_scopes,
                    client_ip=client_ip,
                    origin=origin,
                )

                # Create auth context
                return AuthContext(
                    user_id=record.user_id,
                    tenant_id=record.tenant_id,
                    auth_method="api_key",
                    auth_token_id=record.key_id,
                    scopes=record.scopes,
                    client_ip=client_ip,
                    user_agent=request.headers.get("User-Agent"),
                )

            except Exception as e:
                logger.debug(f"API key authentication failed: {e}")
                return None

        def _get_client_ip(self, request: Request) -> str:
            """Extract client IP from request"""
            forwarded = request.headers.get("X-Forwarded-For")
            if forwarded:
                return forwarded.split(",")[0].strip()
            return request.client.host if request.client else "unknown"

    class AuthenticationMiddleware(BaseHTTPMiddleware):
        """
        FastAPI middleware for authentication.

        Supports both JWT tokens and API keys.
        Sets AuthContext for authenticated requests.
        """

        def __init__(
            self,
            app,
            jwt_handler: Optional[Any] = None,
            api_key_manager: Optional[Any] = None,
            exclude_paths: Optional[List[str]] = None,
            require_auth: bool = False,
        ):
            """
            Initialize authentication middleware.

            Args:
                app: FastAPI application
                jwt_handler: JWTHandler instance (optional)
                api_key_manager: APIKeyManager instance (optional)
                exclude_paths: Paths to exclude from auth (e.g., ["/health", "/docs"])
                require_auth: Whether to require authentication for all requests
            """
            super().__init__(app)
            self.jwt_backend = JWTAuthBackend(jwt_handler) if jwt_handler else None
            self.api_key_backend = APIKeyAuthBackend(api_key_manager) if api_key_manager else None
            self.exclude_paths = set(exclude_paths or ["/health", "/docs", "/openapi.json"])
            self.require_auth = require_auth

        async def dispatch(self, request: Request, call_next):
            """Process request through authentication"""
            # Skip excluded paths
            if request.url.path in self.exclude_paths:
                return await call_next(request)

            # Try JWT authentication first
            auth_context = None
            if self.jwt_backend:
                auth_context = await self.jwt_backend.authenticate(request)

            # Fall back to API key authentication
            if not auth_context and self.api_key_backend:
                auth_context = await self.api_key_backend.authenticate(request)

            # Check if authentication is required
            if self.require_auth and not auth_context:
                return JSONResponse(
                    status_code=401,
                    content={"detail": "Authentication required"},
                    headers={"WWW-Authenticate": "Bearer"},
                )

            # Set auth context
            set_auth_context(auth_context)

            # Add context to request state for easy access
            request.state.auth = auth_context

            try:
                response = await call_next(request)
                return response
            finally:
                # Clear auth context after request
                set_auth_context(None)

    # Dependency injection functions
    async def get_current_user_dep(request: Request) -> AuthContext:
        """
        FastAPI dependency to get current authenticated user.

        Raises HTTPException 401 if not authenticated.
        """
        auth = getattr(request.state, "auth", None)
        if not auth:
            raise HTTPException(
                status_code=401,
                detail="Not authenticated",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return auth

    async def get_optional_user_dep(request: Request) -> Optional[AuthContext]:
        """
        FastAPI dependency to optionally get current user.

        Returns None if not authenticated.
        """
        return getattr(request.state, "auth", None)

    def require_auth(func: Callable) -> Callable:
        """
        Decorator to require authentication for an endpoint.

        Example:
            @app.get("/protected")
            @require_auth
            async def protected_endpoint(request: Request):
                user = request.state.auth
                return {"user": user.user_id}
        """
        @wraps(func)
        async def wrapper(*args, **kwargs):
            request = kwargs.get("request")
            if not request:
                for arg in args:
                    if isinstance(arg, Request):
                        request = arg
                        break

            if request and not getattr(request.state, "auth", None):
                raise HTTPException(
                    status_code=401,
                    detail="Authentication required",
                    headers={"WWW-Authenticate": "Bearer"},
                )
            return await func(*args, **kwargs)
        return wrapper

    def require_roles(*roles: str) -> Callable:
        """
        Decorator to require specific roles for an endpoint.

        Example:
            @app.get("/admin")
            @require_roles("admin", "super_admin")
            async def admin_endpoint(request: Request):
                return {"message": "Admin access granted"}
        """
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            async def wrapper(*args, **kwargs):
                request = kwargs.get("request")
                if not request:
                    for arg in args:
                        if isinstance(arg, Request):
                            request = arg
                            break

                auth = getattr(request.state, "auth", None) if request else None
                if not auth:
                    raise HTTPException(
                        status_code=401,
                        detail="Authentication required",
                        headers={"WWW-Authenticate": "Bearer"},
                    )

                if not auth.has_any_role(list(roles)):
                    raise HTTPException(
                        status_code=403,
                        detail=f"Required roles: {roles}",
                    )

                return await func(*args, **kwargs)
            return wrapper
        return decorator

    def require_permissions(*permissions: str) -> Callable:
        """
        Decorator to require specific permissions for an endpoint.

        Example:
            @app.post("/agents")
            @require_permissions("agent:create")
            async def create_agent(request: Request):
                return {"message": "Agent created"}
        """
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            async def wrapper(*args, **kwargs):
                request = kwargs.get("request")
                if not request:
                    for arg in args:
                        if isinstance(arg, Request):
                            request = arg
                            break

                auth = getattr(request.state, "auth", None) if request else None
                if not auth:
                    raise HTTPException(
                        status_code=401,
                        detail="Authentication required",
                        headers={"WWW-Authenticate": "Bearer"},
                    )

                if not auth.has_any_permission(list(permissions)):
                    raise HTTPException(
                        status_code=403,
                        detail=f"Required permissions: {permissions}",
                    )

                return await func(*args, **kwargs)
            return wrapper
        return decorator

    def require_tenant(tenant_id_param: str = "tenant_id") -> Callable:
        """
        Decorator to validate tenant access.

        Ensures the authenticated user belongs to the tenant specified
        in the request path or query parameter.

        Example:
            @app.get("/tenants/{tenant_id}/agents")
            @require_tenant("tenant_id")
            async def get_tenant_agents(request: Request, tenant_id: str):
                return {"tenant": tenant_id}
        """
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            async def wrapper(*args, **kwargs):
                request = kwargs.get("request")
                if not request:
                    for arg in args:
                        if isinstance(arg, Request):
                            request = arg
                            break

                auth = getattr(request.state, "auth", None) if request else None
                if not auth:
                    raise HTTPException(
                        status_code=401,
                        detail="Authentication required",
                    )

                # Get tenant_id from path parameters or kwargs
                requested_tenant = kwargs.get(tenant_id_param)
                if not requested_tenant and request:
                    requested_tenant = request.path_params.get(tenant_id_param)

                if requested_tenant and requested_tenant != auth.tenant_id:
                    # Allow super_admin to access any tenant
                    if not auth.has_role("super_admin"):
                        raise HTTPException(
                            status_code=403,
                            detail="Access denied to this tenant",
                        )

                return await func(*args, **kwargs)
            return wrapper
        return decorator

    class TenantContextMiddleware(BaseHTTPMiddleware):
        """
        Middleware to extract and validate tenant context.

        Extracts tenant_id from:
        1. JWT token claims
        2. X-Tenant-ID header
        3. Query parameter
        """

        def __init__(self, app, header_name: str = "X-Tenant-ID"):
            super().__init__(app)
            self.header_name = header_name

        async def dispatch(self, request: Request, call_next):
            """Extract tenant context"""
            tenant_id = None

            # First, check auth context (set by AuthenticationMiddleware)
            auth = getattr(request.state, "auth", None)
            if auth:
                tenant_id = auth.tenant_id

            # Fall back to header
            if not tenant_id:
                tenant_id = request.headers.get(self.header_name)

            # Fall back to query param
            if not tenant_id:
                tenant_id = request.query_params.get("tenant_id")

            # Store in request state
            request.state.tenant_id = tenant_id

            return await call_next(request)

else:
    # Stubs when FastAPI is not available
    class JWTAuthBackend:
        def __init__(self, *args, **kwargs):
            raise ImportError("FastAPI is required for JWTAuthBackend")

    class APIKeyAuthBackend:
        def __init__(self, *args, **kwargs):
            raise ImportError("FastAPI is required for APIKeyAuthBackend")

    class AuthenticationMiddleware:
        def __init__(self, *args, **kwargs):
            raise ImportError("FastAPI is required for AuthenticationMiddleware")

    def require_auth(func):
        raise ImportError("FastAPI is required for require_auth decorator")

    def require_roles(*roles):
        raise ImportError("FastAPI is required for require_roles decorator")

    def require_permissions(*permissions):
        raise ImportError("FastAPI is required for require_permissions decorator")
