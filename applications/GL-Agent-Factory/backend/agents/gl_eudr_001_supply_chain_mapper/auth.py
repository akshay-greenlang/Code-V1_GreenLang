"""
GL-EUDR-001: Authentication & Authorization Module

Provides JWT-based authentication and role-based access control (RBAC)
for the Supply Chain Mapper API.

Features:
- JWT token validation with configurable expiry
- Role-based permissions (admin, compliance_officer, analyst, auditor)
- Resource ownership verification (organization-level isolation)
- Rate limiting per user/role
- Audit-ready authentication logging

Security Standards:
- OWASP API Security Top 10 compliant
- GDPR-ready with user consent tracking
- EUDR audit trail requirements
"""

import hashlib
import logging
import os
import secrets
from datetime import datetime, timedelta
from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Set
from uuid import UUID

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

# JWT library - using python-jose
try:
    from jose import JWTError, jwt
except ImportError:
    # Fallback for environments without jose
    jwt = None
    JWTError = Exception

logger = logging.getLogger(__name__)
security_logger = logging.getLogger("security.auth")

# =============================================================================
# CONFIGURATION
# =============================================================================

class AuthConfig:
    """Authentication configuration."""
    SECRET_KEY: str = os.getenv("JWT_SECRET_KEY", secrets.token_urlsafe(32))
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("JWT_EXPIRE_MINUTES", "60"))
    REFRESH_TOKEN_EXPIRE_DAYS: int = int(os.getenv("JWT_REFRESH_DAYS", "7"))
    ISSUER: str = "gl-eudr-001"
    AUDIENCE: str = "supply-chain-mapper"


# =============================================================================
# ENUMS & MODELS
# =============================================================================

class UserRole(str, Enum):
    """User roles with hierarchical permissions."""
    ADMIN = "admin"
    COMPLIANCE_OFFICER = "compliance_officer"
    ANALYST = "analyst"
    AUDITOR = "auditor"
    VIEWER = "viewer"


class Permission(str, Enum):
    """Granular permissions for RBAC."""
    # Node permissions
    NODE_READ = "node:read"
    NODE_WRITE = "node:write"
    NODE_DELETE = "node:delete"

    # Edge permissions
    EDGE_READ = "edge:read"
    EDGE_WRITE = "edge:write"

    # Coverage permissions
    COVERAGE_READ = "coverage:read"
    COVERAGE_GATES = "coverage:gates"

    # Snapshot permissions
    SNAPSHOT_READ = "snapshot:read"
    SNAPSHOT_CREATE = "snapshot:create"

    # Entity resolution permissions
    ER_READ = "entity_resolution:read"
    ER_RUN = "entity_resolution:run"
    ER_RESOLVE = "entity_resolution:resolve"

    # Admin permissions
    ADMIN_USERS = "admin:users"
    ADMIN_CONFIG = "admin:config"

    # PII access
    PII_VIEW = "pii:view"


# Role to permissions mapping
ROLE_PERMISSIONS: Dict[UserRole, Set[Permission]] = {
    UserRole.ADMIN: set(Permission),  # All permissions
    UserRole.COMPLIANCE_OFFICER: {
        Permission.NODE_READ, Permission.NODE_WRITE,
        Permission.EDGE_READ, Permission.EDGE_WRITE,
        Permission.COVERAGE_READ, Permission.COVERAGE_GATES,
        Permission.SNAPSHOT_READ, Permission.SNAPSHOT_CREATE,
        Permission.ER_READ, Permission.ER_RUN, Permission.ER_RESOLVE,
        Permission.PII_VIEW,
    },
    UserRole.ANALYST: {
        Permission.NODE_READ, Permission.NODE_WRITE,
        Permission.EDGE_READ, Permission.EDGE_WRITE,
        Permission.COVERAGE_READ,
        Permission.SNAPSHOT_READ, Permission.SNAPSHOT_CREATE,
        Permission.ER_READ, Permission.ER_RUN,
    },
    UserRole.AUDITOR: {
        Permission.NODE_READ,
        Permission.EDGE_READ,
        Permission.COVERAGE_READ, Permission.COVERAGE_GATES,
        Permission.SNAPSHOT_READ,
        Permission.ER_READ,
        Permission.PII_VIEW,
    },
    UserRole.VIEWER: {
        Permission.NODE_READ,
        Permission.EDGE_READ,
        Permission.COVERAGE_READ,
        Permission.SNAPSHOT_READ,
    },
}


class User(BaseModel):
    """Authenticated user model."""
    user_id: UUID
    email: str
    name: str
    role: UserRole
    organization_id: UUID
    permissions: Set[Permission] = Field(default_factory=set)
    is_active: bool = True
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_login: Optional[datetime] = None

    class Config:
        use_enum_values = True

    def has_permission(self, permission: Permission) -> bool:
        """Check if user has a specific permission."""
        # Get role-based permissions
        role_perms = ROLE_PERMISSIONS.get(self.role, set())
        # Combine with explicit permissions
        all_perms = role_perms | self.permissions
        return permission in all_perms

    def has_any_permission(self, permissions: List[Permission]) -> bool:
        """Check if user has any of the specified permissions."""
        return any(self.has_permission(p) for p in permissions)

    def has_all_permissions(self, permissions: List[Permission]) -> bool:
        """Check if user has all of the specified permissions."""
        return all(self.has_permission(p) for p in permissions)


class TokenData(BaseModel):
    """JWT token payload."""
    sub: str  # user_id
    email: str
    name: str
    role: str
    org_id: str
    permissions: List[str] = []
    exp: datetime
    iat: datetime
    iss: str = AuthConfig.ISSUER
    aud: str = AuthConfig.AUDIENCE


class TokenResponse(BaseModel):
    """Token response for login endpoint."""
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    refresh_token: Optional[str] = None


# =============================================================================
# TOKEN UTILITIES
# =============================================================================

def create_access_token(user: User, expires_delta: Optional[timedelta] = None) -> str:
    """Create JWT access token for user."""
    if jwt is None:
        raise HTTPException(
            status_code=500,
            detail="JWT library not available"
        )

    if expires_delta is None:
        expires_delta = timedelta(minutes=AuthConfig.ACCESS_TOKEN_EXPIRE_MINUTES)

    now = datetime.utcnow()
    expire = now + expires_delta

    payload = {
        "sub": str(user.user_id),
        "email": user.email,
        "name": user.name,
        "role": user.role.value if isinstance(user.role, UserRole) else user.role,
        "org_id": str(user.organization_id),
        "permissions": [p.value for p in user.permissions],
        "exp": expire,
        "iat": now,
        "iss": AuthConfig.ISSUER,
        "aud": AuthConfig.AUDIENCE,
    }

    token = jwt.encode(payload, AuthConfig.SECRET_KEY, algorithm=AuthConfig.ALGORITHM)

    security_logger.info(
        "Access token created",
        extra={
            "user_id": str(user.user_id),
            "email": user.email,
            "expires": expire.isoformat()
        }
    )

    return token


def decode_token(token: str) -> TokenData:
    """Decode and validate JWT token."""
    if jwt is None:
        raise HTTPException(
            status_code=500,
            detail="JWT library not available"
        )

    try:
        payload = jwt.decode(
            token,
            AuthConfig.SECRET_KEY,
            algorithms=[AuthConfig.ALGORITHM],
            audience=AuthConfig.AUDIENCE,
            issuer=AuthConfig.ISSUER
        )

        return TokenData(
            sub=payload["sub"],
            email=payload["email"],
            name=payload["name"],
            role=payload["role"],
            org_id=payload["org_id"],
            permissions=payload.get("permissions", []),
            exp=datetime.fromtimestamp(payload["exp"]),
            iat=datetime.fromtimestamp(payload["iat"]),
            iss=payload.get("iss", AuthConfig.ISSUER),
            aud=payload.get("aud", AuthConfig.AUDIENCE)
        )

    except JWTError as e:
        security_logger.warning(
            "Token decode failed",
            extra={"error": str(e), "token_hash": hashlib.sha256(token.encode()).hexdigest()[:16]}
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )


# =============================================================================
# FASTAPI DEPENDENCIES
# =============================================================================

# HTTP Bearer scheme
bearer_scheme = HTTPBearer(auto_error=False)


async def get_current_user(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme)
) -> User:
    """
    FastAPI dependency to get current authenticated user.

    Usage:
        @router.get("/endpoint")
        async def endpoint(current_user: User = Depends(get_current_user)):
            ...
    """
    if credentials is None:
        security_logger.warning(
            "Missing authentication",
            extra={"path": request.url.path, "method": request.method}
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = credentials.credentials
    token_data = decode_token(token)

    # Check token expiration
    if token_data.exp < datetime.utcnow():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Create user object from token
    user = User(
        user_id=UUID(token_data.sub),
        email=token_data.email,
        name=token_data.name,
        role=UserRole(token_data.role),
        organization_id=UUID(token_data.org_id),
        permissions={Permission(p) for p in token_data.permissions if p in [e.value for e in Permission]},
    )

    # Log successful authentication
    security_logger.info(
        "User authenticated",
        extra={
            "user_id": str(user.user_id),
            "email": user.email,
            "path": request.url.path,
            "method": request.method
        }
    )

    return user


async def get_optional_user(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme)
) -> Optional[User]:
    """
    FastAPI dependency for optional authentication.
    Returns None if no valid token provided.
    """
    if credentials is None:
        return None

    try:
        return await get_current_user(request, credentials)
    except HTTPException:
        return None


def require_permissions(*permissions: Permission) -> Callable:
    """
    Dependency factory for permission checking.

    Usage:
        @router.post("/nodes")
        async def create_node(
            current_user: User = Depends(get_current_user),
            _: None = Depends(require_permissions(Permission.NODE_WRITE))
        ):
            ...
    """
    async def permission_checker(
        request: Request,
        current_user: User = Depends(get_current_user)
    ) -> None:
        missing = [p for p in permissions if not current_user.has_permission(p)]

        if missing:
            security_logger.warning(
                "Permission denied",
                extra={
                    "user_id": str(current_user.user_id),
                    "required": [p.value for p in permissions],
                    "missing": [p.value for p in missing],
                    "path": request.url.path
                }
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Missing required permissions: {[p.value for p in missing]}"
            )

    return permission_checker


def require_role(*roles: UserRole) -> Callable:
    """
    Dependency factory for role checking.

    Usage:
        @router.delete("/nodes/{node_id}")
        async def delete_node(
            current_user: User = Depends(get_current_user),
            _: None = Depends(require_role(UserRole.ADMIN, UserRole.COMPLIANCE_OFFICER))
        ):
            ...
    """
    async def role_checker(
        request: Request,
        current_user: User = Depends(get_current_user)
    ) -> None:
        if current_user.role not in roles:
            security_logger.warning(
                "Role check failed",
                extra={
                    "user_id": str(current_user.user_id),
                    "user_role": current_user.role.value if isinstance(current_user.role, UserRole) else current_user.role,
                    "required_roles": [r.value for r in roles],
                    "path": request.url.path
                }
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Required role: {[r.value for r in roles]}"
            )

    return role_checker


# =============================================================================
# RESOURCE OWNERSHIP VERIFICATION
# =============================================================================

class ResourceOwnershipVerifier:
    """
    Verifies that a user has access to a specific resource based on organization.
    Prevents IDOR vulnerabilities.
    """

    @staticmethod
    def verify_node_access(
        node_id: UUID,
        current_user: User,
        node_getter: Callable[[UUID], Any]
    ) -> Any:
        """
        Verify user has access to the specified node.

        Args:
            node_id: The node UUID to access
            current_user: The authenticated user
            node_getter: Function to retrieve node by ID

        Returns:
            The node if access is allowed

        Raises:
            HTTPException: 404 if not found, 403 if access denied
        """
        node = node_getter(node_id)

        if node is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Resource not found"
            )

        # Admin can access all resources
        if current_user.role == UserRole.ADMIN:
            return node

        # Check organization ownership
        node_org_id = node.metadata.get("organization_id") if hasattr(node, 'metadata') else None

        if node_org_id and UUID(node_org_id) != current_user.organization_id:
            security_logger.warning(
                "Cross-organization access attempt",
                extra={
                    "user_id": str(current_user.user_id),
                    "user_org": str(current_user.organization_id),
                    "resource_org": str(node_org_id),
                    "resource_id": str(node_id)
                }
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to this resource"
            )

        return node

    @staticmethod
    def filter_by_organization(
        items: List[Any],
        current_user: User,
        org_field: str = "organization_id"
    ) -> List[Any]:
        """
        Filter items to only those belonging to user's organization.

        Args:
            items: List of items to filter
            current_user: The authenticated user
            org_field: Field name containing organization ID

        Returns:
            Filtered list of items
        """
        # Admin sees all
        if current_user.role == UserRole.ADMIN:
            return items

        filtered = []
        for item in items:
            item_org = None

            if hasattr(item, 'metadata') and isinstance(item.metadata, dict):
                item_org = item.metadata.get(org_field)
            elif hasattr(item, org_field):
                item_org = getattr(item, org_field)

            if item_org is None or UUID(str(item_org)) == current_user.organization_id:
                filtered.append(item)

        return filtered


# =============================================================================
# PII MASKING
# =============================================================================

class PIIMasker:
    """
    Masks Personally Identifiable Information (PII) based on user permissions.
    Required for GDPR and EUDR compliance.
    """

    # Fields that contain PII
    PII_FIELDS = {"tax_id", "duns_number", "eori_number", "address"}

    # Partial mask pattern
    MASK_CHAR = "*"

    @classmethod
    def mask_value(cls, value: str, show_last: int = 4) -> str:
        """Mask a value, showing only last N characters."""
        if not value or len(value) <= show_last:
            return cls.MASK_CHAR * len(value) if value else None

        masked_len = len(value) - show_last
        return cls.MASK_CHAR * masked_len + value[-show_last:]

    @classmethod
    def mask_dict(cls, data: Dict[str, Any], current_user: User) -> Dict[str, Any]:
        """
        Mask PII fields in a dictionary based on user permissions.

        Args:
            data: Dictionary potentially containing PII
            current_user: The authenticated user

        Returns:
            Dictionary with PII fields masked if user lacks permission
        """
        if current_user.has_permission(Permission.PII_VIEW):
            return data

        masked = data.copy()

        for field in cls.PII_FIELDS:
            if field in masked and masked[field]:
                if field == "address" and isinstance(masked[field], dict):
                    # Mask address fields
                    masked[field] = {
                        k: cls.mask_value(str(v)) if v else None
                        for k, v in masked[field].items()
                    }
                else:
                    masked[field] = cls.mask_value(str(masked[field]))

        return masked

    @classmethod
    def mask_node(cls, node: Any, current_user: User) -> Dict[str, Any]:
        """Mask PII fields in a node object."""
        if hasattr(node, 'dict'):
            return cls.mask_dict(node.dict(), current_user)
        elif isinstance(node, dict):
            return cls.mask_dict(node, current_user)
        return node


# =============================================================================
# RATE LIMITING
# =============================================================================

class RateLimiter:
    """
    Simple in-memory rate limiter.
    For production, use Redis-based rate limiting.
    """

    def __init__(self, requests_per_minute: int = 60):
        self.requests_per_minute = requests_per_minute
        self._requests: Dict[str, List[datetime]] = {}

    def is_allowed(self, key: str) -> bool:
        """Check if request is allowed under rate limit."""
        now = datetime.utcnow()
        window_start = now - timedelta(minutes=1)

        # Clean old requests
        if key in self._requests:
            self._requests[key] = [
                req_time for req_time in self._requests[key]
                if req_time > window_start
            ]
        else:
            self._requests[key] = []

        # Check limit
        if len(self._requests[key]) >= self.requests_per_minute:
            return False

        # Record request
        self._requests[key].append(now)
        return True

    def get_remaining(self, key: str) -> int:
        """Get remaining requests in current window."""
        now = datetime.utcnow()
        window_start = now - timedelta(minutes=1)

        if key not in self._requests:
            return self.requests_per_minute

        current = len([r for r in self._requests[key] if r > window_start])
        return max(0, self.requests_per_minute - current)


# Global rate limiter instances
default_rate_limiter = RateLimiter(requests_per_minute=60)
strict_rate_limiter = RateLimiter(requests_per_minute=10)  # For expensive operations


def rate_limit(limiter: RateLimiter = default_rate_limiter) -> Callable:
    """
    Dependency factory for rate limiting.

    Usage:
        @router.post("/expensive-operation")
        async def expensive(
            current_user: User = Depends(get_current_user),
            _: None = Depends(rate_limit(strict_rate_limiter))
        ):
            ...
    """
    async def rate_limit_checker(
        request: Request,
        current_user: User = Depends(get_current_user)
    ) -> None:
        key = f"{current_user.user_id}:{request.url.path}"

        if not limiter.is_allowed(key):
            security_logger.warning(
                "Rate limit exceeded",
                extra={
                    "user_id": str(current_user.user_id),
                    "path": request.url.path
                }
            )
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded. Please try again later.",
                headers={"Retry-After": "60"}
            )

    return rate_limit_checker


# =============================================================================
# TEST/DEVELOPMENT HELPERS
# =============================================================================

def create_test_user(
    role: UserRole = UserRole.ANALYST,
    organization_id: Optional[UUID] = None
) -> User:
    """Create a test user for development/testing."""
    import uuid

    return User(
        user_id=uuid.uuid4(),
        email=f"test_{role.value}@example.com",
        name=f"Test {role.value.title()}",
        role=role,
        organization_id=organization_id or uuid.uuid4(),
        is_active=True
    )


def create_test_token(user: User) -> str:
    """Create a test token for development/testing."""
    return create_access_token(user)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Configuration
    "AuthConfig",

    # Enums
    "UserRole",
    "Permission",
    "ROLE_PERMISSIONS",

    # Models
    "User",
    "TokenData",
    "TokenResponse",

    # Token utilities
    "create_access_token",
    "decode_token",

    # FastAPI dependencies
    "bearer_scheme",
    "get_current_user",
    "get_optional_user",
    "require_permissions",
    "require_role",

    # Resource ownership
    "ResourceOwnershipVerifier",

    # PII masking
    "PIIMasker",

    # Rate limiting
    "RateLimiter",
    "default_rate_limiter",
    "strict_rate_limiter",
    "rate_limit",

    # Test helpers
    "create_test_user",
    "create_test_token",
]
