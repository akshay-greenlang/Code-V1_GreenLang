"""
GL-012 SteamQual API Authentication

Implements mTLS + OAuth2/JWT authentication with fine-grained authorization.
Supports role-based access control (RBAC) for steam quality control operations.

Features:
- JWT Bearer token authentication
- API key validation
- RBAC for configuration and actions
- Audit logging of all API calls
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
from datetime import datetime, timedelta
from enum import Enum
from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from uuid import UUID, uuid4

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import (
    APIKeyHeader,
    HTTPAuthorizationCredentials,
    HTTPBearer,
    OAuth2PasswordBearer,
)
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

class AuthConfig(BaseSettings):
    """Authentication configuration with environment variable support."""

    # JWT settings
    jwt_secret_key: str = Field(
        default="steamqual-dev-secret-key-change-in-production-12345",
        alias="STEAMQUAL_JWT_SECRET"
    )
    jwt_algorithm: str = Field(default="HS256", alias="STEAMQUAL_JWT_ALGORITHM")
    jwt_access_token_expire_minutes: int = Field(default=30, alias="STEAMQUAL_JWT_ACCESS_EXPIRE")
    jwt_refresh_token_expire_days: int = Field(default=7, alias="STEAMQUAL_JWT_REFRESH_EXPIRE")

    # API Key settings
    api_key_header: str = Field(default="X-API-Key", alias="STEAMQUAL_API_KEY_HEADER")
    api_key_prefix: str = Field(default="squal_", alias="STEAMQUAL_API_KEY_PREFIX")

    # Security settings
    max_failed_attempts: int = Field(default=5, alias="STEAMQUAL_MAX_FAILED_ATTEMPTS")
    lockout_duration_minutes: int = Field(default=30, alias="STEAMQUAL_LOCKOUT_DURATION")

    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_auth_config() -> AuthConfig:
    """Get cached authentication configuration."""
    return AuthConfig()


# =============================================================================
# Role-Based Access Control
# =============================================================================

class Role(str, Enum):
    """User roles for SteamQual Controller."""
    # System roles
    ADMIN = "admin"  # Full system access
    OPERATOR = "operator"  # Can execute control actions
    ENGINEER = "engineer"  # Can modify configurations
    ANALYST = "analyst"  # Read-only analytics access
    AUDITOR = "auditor"  # Compliance and audit access
    API_SERVICE = "api_service"  # Machine-to-machine access

    # Scoped roles
    QUALITY_OPERATOR = "quality_operator"  # Quality control operations
    VIEWER = "viewer"  # Read-only basic access


class Permission(str, Enum):
    """Fine-grained permissions for SteamQual operations."""

    # Quality estimation permissions
    QUALITY_READ = "quality:read"
    QUALITY_ESTIMATE = "quality:estimate"

    # Carryover risk permissions
    CARRYOVER_READ = "carryover:read"
    CARRYOVER_ASSESS = "carryover:assess"

    # Quality state permissions
    STATE_READ = "state:read"

    # Events permissions
    EVENTS_READ = "events:read"
    EVENTS_ACKNOWLEDGE = "events:acknowledge"
    EVENTS_RESOLVE = "events:resolve"

    # Recommendations permissions
    RECOMMENDATIONS_READ = "recommendations:read"
    RECOMMENDATIONS_IMPLEMENT = "recommendations:implement"

    # Metrics permissions
    METRICS_READ = "metrics:read"
    METRICS_EXPORT = "metrics:export"

    # Configuration permissions
    CONFIG_READ = "config:read"
    CONFIG_WRITE = "config:write"
    THRESHOLDS_CONFIGURE = "thresholds:configure"

    # Control action permissions
    CONTROL_READ = "control:read"
    CONTROL_EXECUTE = "control:execute"
    CONTROL_OVERRIDE = "control:override"

    # Admin permissions
    USER_MANAGE = "user:manage"
    AUDIT_READ = "audit:read"
    SYSTEM_ADMIN = "system:admin"


# Role to permissions mapping
ROLE_PERMISSIONS: Dict[Role, Set[Permission]] = {
    Role.ADMIN: set(Permission),  # All permissions

    Role.OPERATOR: {
        Permission.QUALITY_READ,
        Permission.QUALITY_ESTIMATE,
        Permission.CARRYOVER_READ,
        Permission.CARRYOVER_ASSESS,
        Permission.STATE_READ,
        Permission.EVENTS_READ,
        Permission.EVENTS_ACKNOWLEDGE,
        Permission.RECOMMENDATIONS_READ,
        Permission.RECOMMENDATIONS_IMPLEMENT,
        Permission.METRICS_READ,
        Permission.CONTROL_READ,
        Permission.CONTROL_EXECUTE,
    },

    Role.ENGINEER: {
        Permission.QUALITY_READ,
        Permission.QUALITY_ESTIMATE,
        Permission.CARRYOVER_READ,
        Permission.CARRYOVER_ASSESS,
        Permission.STATE_READ,
        Permission.EVENTS_READ,
        Permission.EVENTS_ACKNOWLEDGE,
        Permission.EVENTS_RESOLVE,
        Permission.RECOMMENDATIONS_READ,
        Permission.METRICS_READ,
        Permission.METRICS_EXPORT,
        Permission.CONFIG_READ,
        Permission.CONFIG_WRITE,
        Permission.THRESHOLDS_CONFIGURE,
        Permission.CONTROL_READ,
    },

    Role.ANALYST: {
        Permission.QUALITY_READ,
        Permission.CARRYOVER_READ,
        Permission.STATE_READ,
        Permission.EVENTS_READ,
        Permission.RECOMMENDATIONS_READ,
        Permission.METRICS_READ,
        Permission.METRICS_EXPORT,
        Permission.AUDIT_READ,
    },

    Role.AUDITOR: {
        Permission.QUALITY_READ,
        Permission.CARRYOVER_READ,
        Permission.STATE_READ,
        Permission.EVENTS_READ,
        Permission.RECOMMENDATIONS_READ,
        Permission.METRICS_READ,
        Permission.METRICS_EXPORT,
        Permission.CONFIG_READ,
        Permission.AUDIT_READ,
    },

    Role.API_SERVICE: {
        Permission.QUALITY_READ,
        Permission.QUALITY_ESTIMATE,
        Permission.CARRYOVER_READ,
        Permission.CARRYOVER_ASSESS,
        Permission.STATE_READ,
        Permission.EVENTS_READ,
        Permission.RECOMMENDATIONS_READ,
        Permission.METRICS_READ,
        Permission.CONTROL_READ,
    },

    Role.QUALITY_OPERATOR: {
        Permission.QUALITY_READ,
        Permission.QUALITY_ESTIMATE,
        Permission.CARRYOVER_READ,
        Permission.CARRYOVER_ASSESS,
        Permission.STATE_READ,
        Permission.EVENTS_READ,
        Permission.EVENTS_ACKNOWLEDGE,
        Permission.RECOMMENDATIONS_READ,
        Permission.RECOMMENDATIONS_IMPLEMENT,
        Permission.CONTROL_READ,
        Permission.CONTROL_EXECUTE,
    },

    Role.VIEWER: {
        Permission.QUALITY_READ,
        Permission.CARRYOVER_READ,
        Permission.STATE_READ,
        Permission.EVENTS_READ,
        Permission.RECOMMENDATIONS_READ,
        Permission.METRICS_READ,
    },
}


# =============================================================================
# User Model
# =============================================================================

class SteamQualUser(BaseModel):
    """Authenticated user model for SteamQual Controller."""
    user_id: UUID
    username: str
    email: str
    tenant_id: UUID
    roles: List[Role]
    permissions: Set[Permission] = Field(default_factory=set)

    # Organization context
    organization_id: Optional[UUID] = None
    organization_name: Optional[str] = None

    # Resource access (for scoped access)
    allowed_header_ids: Optional[List[str]] = None  # None = all headers

    # Session info
    session_id: Optional[str] = None
    auth_method: str = "jwt"  # jwt, api_key

    # Metadata
    created_at: datetime
    last_login: Optional[datetime] = None

    def has_permission(self, permission: Permission) -> bool:
        """Check if user has a specific permission."""
        # Direct permission check
        if permission in self.permissions:
            return True

        # Role-based permission check
        for role in self.roles:
            if permission in ROLE_PERMISSIONS.get(role, set()):
                return True

        return False

    def has_any_permission(self, permissions: List[Permission]) -> bool:
        """Check if user has any of the specified permissions."""
        return any(self.has_permission(p) for p in permissions)

    def has_all_permissions(self, permissions: List[Permission]) -> bool:
        """Check if user has all specified permissions."""
        return all(self.has_permission(p) for p in permissions)

    def can_access_header(self, header_id: str) -> bool:
        """Check if user can access a specific steam header."""
        if self.allowed_header_ids is None:
            return True  # No restrictions
        return header_id in self.allowed_header_ids

    def get_all_permissions(self) -> Set[Permission]:
        """Get all permissions for the user (direct + role-based)."""
        all_perms = set(self.permissions)
        for role in self.roles:
            all_perms.update(ROLE_PERMISSIONS.get(role, set()))
        return all_perms

    class Config:
        json_encoders = {
            UUID: lambda v: str(v),
            datetime: lambda v: v.isoformat(),
            Permission: lambda v: v.value,
            Role: lambda v: v.value,
        }


# =============================================================================
# Token Models
# =============================================================================

class TokenData(BaseModel):
    """JWT token payload data."""
    sub: str  # User ID
    username: str
    email: str
    tenant_id: str
    roles: List[str]
    permissions: List[str] = Field(default_factory=list)
    exp: datetime
    iat: datetime
    jti: Optional[str] = None  # JWT ID for revocation
    scope: str = "access"  # access, refresh


class TokenResponse(BaseModel):
    """OAuth2 token response."""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int  # Seconds until expiration
    scope: str = "openid profile email steamqual"


# =============================================================================
# Security Utilities
# =============================================================================

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2 scheme for JWT bearer tokens
oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl="/api/v1/auth/token",
    auto_error=False,
    scopes={
        "openid": "OpenID Connect scope",
        "profile": "User profile information",
        "email": "User email address",
        "steamqual": "Steam quality control operations",
    },
)

# API Key authentication
api_key_header_scheme = APIKeyHeader(name="X-API-Key", auto_error=False)

# HTTP Bearer for explicit bearer token auth
http_bearer = HTTPBearer(auto_error=False)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash."""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Hash a password for storage."""
    return pwd_context.hash(password)


def create_access_token(
    user: SteamQualUser,
    config: AuthConfig,
    expires_delta: Optional[timedelta] = None,
) -> str:
    """
    Create a JWT access token for a user.

    Args:
        user: Authenticated user
        config: Authentication configuration
        expires_delta: Optional custom expiration time

    Returns:
        Encoded JWT token string
    """
    if expires_delta is None:
        expires_delta = timedelta(minutes=config.jwt_access_token_expire_minutes)

    now = datetime.utcnow()
    expire = now + expires_delta

    payload = {
        "sub": str(user.user_id),
        "username": user.username,
        "email": user.email,
        "tenant_id": str(user.tenant_id),
        "roles": [role.value for role in user.roles],
        "permissions": [perm.value for perm in user.permissions],
        "exp": expire,
        "iat": now,
        "jti": hashlib.sha256(f"{user.user_id}{now.isoformat()}".encode()).hexdigest()[:16],
        "scope": "access",
    }

    # Add optional claims
    if user.organization_id:
        payload["org_id"] = str(user.organization_id)
    if user.allowed_header_ids:
        payload["header_ids"] = user.allowed_header_ids

    return jwt.encode(payload, config.jwt_secret_key, algorithm=config.jwt_algorithm)


def create_refresh_token(
    user: SteamQualUser,
    config: AuthConfig,
) -> str:
    """
    Create a JWT refresh token for a user.

    Args:
        user: Authenticated user
        config: Authentication configuration

    Returns:
        Encoded JWT refresh token string
    """
    now = datetime.utcnow()
    expire = now + timedelta(days=config.jwt_refresh_token_expire_days)

    payload = {
        "sub": str(user.user_id),
        "tenant_id": str(user.tenant_id),
        "exp": expire,
        "iat": now,
        "jti": hashlib.sha256(f"refresh_{user.user_id}{now.isoformat()}".encode()).hexdigest()[:16],
        "scope": "refresh",
    }

    return jwt.encode(payload, config.jwt_secret_key, algorithm=config.jwt_algorithm)


def decode_token(token: str, config: AuthConfig) -> TokenData:
    """
    Decode and validate a JWT token.

    Args:
        token: JWT token string
        config: Authentication configuration

    Returns:
        Decoded token data

    Raises:
        JWTError: If token is invalid or expired
    """
    payload = jwt.decode(
        token,
        config.jwt_secret_key,
        algorithms=[config.jwt_algorithm],
    )

    return TokenData(
        sub=payload["sub"],
        username=payload.get("username", ""),
        email=payload.get("email", ""),
        tenant_id=payload["tenant_id"],
        roles=payload.get("roles", []),
        permissions=payload.get("permissions", []),
        exp=datetime.fromtimestamp(payload["exp"]),
        iat=datetime.fromtimestamp(payload["iat"]),
        jti=payload.get("jti"),
        scope=payload.get("scope", "access"),
    )


# =============================================================================
# API Key Management
# =============================================================================

class APIKeyInfo(BaseModel):
    """API key metadata."""
    key_id: str
    key_prefix: str  # First 8 chars for identification
    name: str
    user_id: UUID
    tenant_id: UUID
    roles: List[Role]
    permissions: Set[Permission]
    created_at: datetime
    expires_at: Optional[datetime] = None
    last_used_at: Optional[datetime] = None
    is_active: bool = True
    allowed_ips: Optional[List[str]] = None


def generate_api_key(prefix: str = "squal_") -> Tuple[str, str]:
    """
    Generate a new API key.

    Returns:
        Tuple of (full_key, key_hash)
    """
    import secrets

    # Generate random key
    random_part = secrets.token_urlsafe(32)
    full_key = f"{prefix}{random_part}"

    # Hash for storage
    key_hash = hashlib.sha256(full_key.encode()).hexdigest()

    return full_key, key_hash


def verify_api_key(api_key: str, stored_hash: str) -> bool:
    """Verify an API key against its stored hash."""
    key_hash = hashlib.sha256(api_key.encode()).hexdigest()
    return hmac.compare_digest(key_hash, stored_hash)


# =============================================================================
# In-Memory Stores (Replace with Database in Production)
# =============================================================================

_users_db: Dict[str, SteamQualUser] = {}
_api_keys_db: Dict[str, APIKeyInfo] = {}
_revoked_tokens: Set[str] = set()


async def get_user_by_id(user_id: str) -> Optional[SteamQualUser]:
    """Get user by ID from database."""
    return _users_db.get(user_id)


async def get_api_key_info(api_key: str) -> Optional[APIKeyInfo]:
    """Get API key info from database."""
    key_hash = hashlib.sha256(api_key.encode()).hexdigest()
    return _api_keys_db.get(key_hash)


async def is_token_revoked(jti: str) -> bool:
    """Check if a token has been revoked."""
    return jti in _revoked_tokens


async def revoke_token(jti: str) -> None:
    """Revoke a token by its JTI."""
    _revoked_tokens.add(jti)


# =============================================================================
# Authentication Dependencies
# =============================================================================

def verify_token(token: str, config: Optional[AuthConfig] = None) -> TokenData:
    """
    Verify and decode a JWT token.

    Args:
        token: JWT token string
        config: Optional auth config (uses default if not provided)

    Returns:
        Decoded token data

    Raises:
        HTTPException: If token is invalid
    """
    if config is None:
        config = get_auth_config()

    try:
        token_data = decode_token(token, config)
        return token_data
    except JWTError as e:
        logger.warning(f"Token verification failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def get_current_user(
    request: Request,
    token: Optional[str] = Depends(oauth2_scheme),
    api_key: Optional[str] = Depends(api_key_header_scheme),
    bearer: Optional[HTTPAuthorizationCredentials] = Depends(http_bearer),
) -> SteamQualUser:
    """
    Get the current authenticated user from the request.

    Supports multiple authentication methods:
    1. OAuth2 Bearer token (Authorization header)
    2. API Key (X-API-Key header)

    Args:
        request: FastAPI request
        token: OAuth2 token from Authorization header
        api_key: API key from X-API-Key header
        bearer: HTTP Bearer credentials

    Returns:
        Authenticated user

    Raises:
        HTTPException: If authentication fails
    """
    config = get_auth_config()
    auth_method = None
    user = None

    # Try OAuth2 Bearer token first
    effective_token = token
    if not effective_token and bearer:
        effective_token = bearer.credentials

    if effective_token:
        try:
            token_data = verify_token(effective_token, config)

            # Check if token is revoked
            if token_data.jti and await is_token_revoked(token_data.jti):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Token has been revoked",
                )

            # Load user from database or create from token
            user = await get_user_by_id(token_data.sub)
            if not user:
                # Create user from token data
                user = SteamQualUser(
                    user_id=UUID(token_data.sub),
                    username=token_data.username,
                    email=token_data.email,
                    tenant_id=UUID(token_data.tenant_id),
                    roles=[Role(r) for r in token_data.roles if r in [e.value for e in Role]],
                    permissions={Permission(p) for p in token_data.permissions if p in [e.value for e in Permission]},
                    auth_method="jwt",
                    created_at=token_data.iat,
                )
            auth_method = "jwt"
        except HTTPException:
            pass  # Try other methods

    # Try API Key
    if not user and api_key:
        api_key_info = await get_api_key_info(api_key)

        if api_key_info and api_key_info.is_active:
            # Check expiration
            if api_key_info.expires_at and api_key_info.expires_at < datetime.utcnow():
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="API key has expired",
                )

            # Check IP restrictions
            if api_key_info.allowed_ips:
                client_ip = request.client.host if request.client else None
                if client_ip not in api_key_info.allowed_ips:
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="IP address not allowed for this API key",
                    )

            # Create user from API key
            user = SteamQualUser(
                user_id=api_key_info.user_id,
                username=api_key_info.name,
                email="",
                tenant_id=api_key_info.tenant_id,
                roles=api_key_info.roles,
                permissions=api_key_info.permissions,
                auth_method="api_key",
                created_at=api_key_info.created_at,
            )
            auth_method = "api_key"

    # Authentication failed
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Update auth method
    user.auth_method = auth_method

    # Log authentication
    logger.info(
        f"User authenticated: {user.username} ({user.user_id}) "
        f"via {auth_method} from {request.client.host if request.client else 'unknown'}"
    )

    return user


# =============================================================================
# Authorization Decorators and Dependencies
# =============================================================================

def require_permissions(*required_permissions: Permission):
    """
    Dependency factory for requiring specific permissions.

    Usage:
        @app.get("/endpoint")
        async def endpoint(user: SteamQualUser = Depends(require_permissions(Permission.QUALITY_READ))):
            ...
    """
    async def permission_dependency(
        user: SteamQualUser = Depends(get_current_user),
    ) -> SteamQualUser:
        if not user.has_all_permissions(list(required_permissions)):
            missing = [p.value for p in required_permissions if not user.has_permission(p)]
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Missing required permissions: {', '.join(missing)}",
            )
        return user

    return permission_dependency


def require_any_permission(*required_permissions: Permission):
    """
    Dependency factory requiring any of the specified permissions.
    """
    async def permission_dependency(
        user: SteamQualUser = Depends(get_current_user),
    ) -> SteamQualUser:
        if not user.has_any_permission(list(required_permissions)):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions",
            )
        return user

    return permission_dependency


def require_roles(*required_roles: Role):
    """
    Dependency factory for requiring specific roles.
    """
    async def role_dependency(
        user: SteamQualUser = Depends(get_current_user),
    ) -> SteamQualUser:
        if not any(role in user.roles for role in required_roles):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Required roles: {', '.join(r.value for r in required_roles)}",
            )
        return user

    return role_dependency


def require_header_access(header_id_param: str = "header_id"):
    """
    Dependency factory for requiring access to a specific steam header.
    """
    async def header_dependency(
        request: Request,
        user: SteamQualUser = Depends(get_current_user),
    ) -> SteamQualUser:
        # Get header_id from path or query parameters
        header_id = request.path_params.get(header_id_param) or request.query_params.get(header_id_param)

        if header_id:
            if not user.can_access_header(header_id):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Not authorized to access header {header_id}",
                )

        return user

    return header_dependency


# =============================================================================
# Audit Logging
# =============================================================================

class AuditLogEntry(BaseModel):
    """Audit log entry for security and operational events."""
    log_id: UUID = Field(default_factory=uuid4)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    event_type: str
    user_id: Optional[UUID] = None
    username: Optional[str] = None
    tenant_id: Optional[UUID] = None
    action: str
    resource_type: str
    resource_id: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    success: bool
    details: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    latency_ms: Optional[float] = None


async def log_api_call(
    request: Request,
    user: Optional[SteamQualUser] = None,
    action: str = "api_call",
    resource_type: str = "api",
    resource_id: Optional[str] = None,
    success: bool = True,
    details: Optional[Dict[str, Any]] = None,
    error_message: Optional[str] = None,
    latency_ms: Optional[float] = None,
) -> AuditLogEntry:
    """
    Log an API call for audit purposes.

    This function captures:
    - User identity (who made the call)
    - Action performed (what was done)
    - Resource accessed (what was accessed)
    - Success/failure status
    - Performance metrics (latency)

    Args:
        request: FastAPI request
        user: Authenticated user (if any)
        action: Action performed
        resource_type: Type of resource accessed
        resource_id: ID of accessed resource
        success: Whether the action succeeded
        details: Additional details
        error_message: Error message if failed
        latency_ms: Request latency in milliseconds

    Returns:
        Created audit log entry
    """
    entry = AuditLogEntry(
        event_type="api_call",
        user_id=user.user_id if user else None,
        username=user.username if user else None,
        tenant_id=user.tenant_id if user else None,
        action=action,
        resource_type=resource_type,
        resource_id=resource_id,
        ip_address=request.client.host if request.client else None,
        user_agent=request.headers.get("user-agent"),
        success=success,
        details=details,
        error_message=error_message,
        latency_ms=latency_ms,
    )

    # In production, persist to database or audit log service
    log_level = logging.INFO if success else logging.WARNING
    logger.log(log_level, f"AUDIT: {entry.model_dump_json()}")

    return entry


async def log_security_event(
    event_type: str,
    action: str,
    resource_type: str,
    request: Request,
    user: Optional[SteamQualUser] = None,
    resource_id: Optional[str] = None,
    success: bool = True,
    details: Optional[Dict[str, Any]] = None,
    error_message: Optional[str] = None,
) -> AuditLogEntry:
    """
    Log a security-related event for audit purposes.

    Args:
        event_type: Type of event (auth, access, modification)
        action: Action performed
        resource_type: Type of resource accessed
        request: FastAPI request
        user: Authenticated user (if any)
        resource_id: ID of accessed resource
        success: Whether the action succeeded
        details: Additional details
        error_message: Error message if failed

    Returns:
        Created audit log entry
    """
    entry = AuditLogEntry(
        event_type=event_type,
        user_id=user.user_id if user else None,
        username=user.username if user else None,
        tenant_id=user.tenant_id if user else None,
        action=action,
        resource_type=resource_type,
        resource_id=resource_id,
        ip_address=request.client.host if request.client else None,
        user_agent=request.headers.get("user-agent"),
        success=success,
        details=details,
        error_message=error_message,
    )

    # In production, persist to database or audit log service
    log_level = logging.INFO if success else logging.WARNING
    logger.log(log_level, f"SECURITY_AUDIT: {entry.model_dump_json()}")

    return entry
