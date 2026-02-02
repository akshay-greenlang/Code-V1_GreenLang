"""
GL-003 UnifiedSteam API Authentication

Implements mTLS + OAuth2/JWT authentication with fine-grained authorization.
Supports role-based access control (RBAC) for steam system operations.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
import ssl
from datetime import datetime, timedelta
from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from uuid import UUID

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import (
    APIKeyHeader,
    HTTPAuthorizationCredentials,
    HTTPBearer,
    OAuth2PasswordBearer,
)
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

class AuthConfig(BaseModel):
    """Authentication configuration."""
    # JWT settings
    jwt_secret_key: str = Field(..., min_length=32)
    jwt_algorithm: str = "HS256"
    jwt_access_token_expire_minutes: int = 30
    jwt_refresh_token_expire_days: int = 7

    # mTLS settings
    mtls_enabled: bool = True
    mtls_ca_cert_path: Optional[str] = None
    mtls_server_cert_path: Optional[str] = None
    mtls_server_key_path: Optional[str] = None
    mtls_verify_client: bool = True

    # API Key settings
    api_key_header: str = "X-API-Key"
    api_key_prefix: str = "steam_"

    # Security settings
    max_failed_attempts: int = 5
    lockout_duration_minutes: int = 30
    password_min_length: int = 12
    require_mfa: bool = False


# Load config from environment (with defaults for development)
def get_auth_config() -> AuthConfig:
    """Load authentication configuration from environment."""
    return AuthConfig(
        jwt_secret_key=os.getenv("STEAM_JWT_SECRET", "development-secret-key-change-in-production-12345"),
        jwt_algorithm=os.getenv("STEAM_JWT_ALGORITHM", "HS256"),
        jwt_access_token_expire_minutes=int(os.getenv("STEAM_JWT_ACCESS_EXPIRE", "30")),
        jwt_refresh_token_expire_days=int(os.getenv("STEAM_JWT_REFRESH_EXPIRE", "7")),
        mtls_enabled=os.getenv("STEAM_MTLS_ENABLED", "false").lower() == "true",
        mtls_ca_cert_path=os.getenv("STEAM_MTLS_CA_CERT"),
        mtls_server_cert_path=os.getenv("STEAM_MTLS_SERVER_CERT"),
        mtls_server_key_path=os.getenv("STEAM_MTLS_SERVER_KEY"),
        api_key_header=os.getenv("STEAM_API_KEY_HEADER", "X-API-Key"),
    )


# =============================================================================
# Role-Based Access Control
# =============================================================================

class Role(str, Enum):
    """User roles for SteamSystemOptimizer."""
    # System roles
    ADMIN = "admin"  # Full system access
    OPERATOR = "operator"  # Can execute optimization actions
    ENGINEER = "engineer"  # Can modify configurations and view detailed analytics
    ANALYST = "analyst"  # Read-only analytics access
    AUDITOR = "auditor"  # Compliance and audit access
    API_SERVICE = "api_service"  # Machine-to-machine access

    # Scoped roles
    STEAM_OPERATOR = "steam_operator"  # Steam operations only
    TRAP_TECHNICIAN = "trap_technician"  # Trap diagnostics and maintenance
    VIEWER = "viewer"  # Read-only basic access


class Permission(str, Enum):
    """Fine-grained permissions for SteamSystemOptimizer operations."""
    # Steam properties permissions
    STEAM_PROPERTIES_READ = "steam:properties:read"
    STEAM_PROPERTIES_COMPUTE = "steam:properties:compute"

    # Balance permissions
    BALANCE_READ = "balance:read"
    BALANCE_COMPUTE = "balance:compute"

    # Optimization permissions
    OPTIMIZATION_READ = "optimization:read"
    OPTIMIZATION_REQUEST = "optimization:request"
    OPTIMIZATION_EXECUTE = "optimization:execute"
    OPTIMIZATION_CONFIGURE = "optimization:configure"

    # Trap permissions
    TRAP_READ = "trap:read"
    TRAP_DIAGNOSE = "trap:diagnose"
    TRAP_MAINTAIN = "trap:maintain"
    TRAP_CONFIGURE = "trap:configure"

    # RCA permissions
    RCA_READ = "rca:read"
    RCA_ANALYZE = "rca:analyze"

    # KPI permissions
    KPI_READ = "kpi:read"
    KPI_WRITE = "kpi:write"
    KPI_CONFIGURE = "kpi:configure"

    # Climate/Emissions permissions
    CLIMATE_READ = "climate:read"
    CLIMATE_REPORT = "climate:report"

    # Explainability permissions
    EXPLAINABILITY_READ = "explainability:read"

    # Recommendation permissions
    RECOMMENDATION_READ = "recommendation:read"
    RECOMMENDATION_ACKNOWLEDGE = "recommendation:acknowledge"
    RECOMMENDATION_IMPLEMENT = "recommendation:implement"

    # Configuration permissions
    CONFIG_READ = "config:read"
    CONFIG_WRITE = "config:write"

    # Admin permissions
    USER_MANAGE = "user:manage"
    AUDIT_READ = "audit:read"
    SYSTEM_ADMIN = "system:admin"


# Role to permissions mapping
ROLE_PERMISSIONS: Dict[Role, Set[Permission]] = {
    Role.ADMIN: set(Permission),  # All permissions

    Role.OPERATOR: {
        Permission.STEAM_PROPERTIES_READ,
        Permission.STEAM_PROPERTIES_COMPUTE,
        Permission.BALANCE_READ,
        Permission.BALANCE_COMPUTE,
        Permission.OPTIMIZATION_READ,
        Permission.OPTIMIZATION_REQUEST,
        Permission.OPTIMIZATION_EXECUTE,
        Permission.TRAP_READ,
        Permission.TRAP_DIAGNOSE,
        Permission.RCA_READ,
        Permission.KPI_READ,
        Permission.CLIMATE_READ,
        Permission.EXPLAINABILITY_READ,
        Permission.RECOMMENDATION_READ,
        Permission.RECOMMENDATION_ACKNOWLEDGE,
        Permission.RECOMMENDATION_IMPLEMENT,
    },

    Role.ENGINEER: {
        Permission.STEAM_PROPERTIES_READ,
        Permission.STEAM_PROPERTIES_COMPUTE,
        Permission.BALANCE_READ,
        Permission.BALANCE_COMPUTE,
        Permission.OPTIMIZATION_READ,
        Permission.OPTIMIZATION_REQUEST,
        Permission.OPTIMIZATION_CONFIGURE,
        Permission.TRAP_READ,
        Permission.TRAP_DIAGNOSE,
        Permission.TRAP_CONFIGURE,
        Permission.RCA_READ,
        Permission.RCA_ANALYZE,
        Permission.KPI_READ,
        Permission.KPI_WRITE,
        Permission.KPI_CONFIGURE,
        Permission.CLIMATE_READ,
        Permission.CLIMATE_REPORT,
        Permission.EXPLAINABILITY_READ,
        Permission.RECOMMENDATION_READ,
        Permission.CONFIG_READ,
        Permission.CONFIG_WRITE,
    },

    Role.ANALYST: {
        Permission.STEAM_PROPERTIES_READ,
        Permission.BALANCE_READ,
        Permission.OPTIMIZATION_READ,
        Permission.TRAP_READ,
        Permission.RCA_READ,
        Permission.KPI_READ,
        Permission.CLIMATE_READ,
        Permission.CLIMATE_REPORT,
        Permission.EXPLAINABILITY_READ,
        Permission.RECOMMENDATION_READ,
        Permission.AUDIT_READ,
    },

    Role.AUDITOR: {
        Permission.STEAM_PROPERTIES_READ,
        Permission.BALANCE_READ,
        Permission.OPTIMIZATION_READ,
        Permission.TRAP_READ,
        Permission.RCA_READ,
        Permission.KPI_READ,
        Permission.CLIMATE_READ,
        Permission.CLIMATE_REPORT,
        Permission.RECOMMENDATION_READ,
        Permission.AUDIT_READ,
        Permission.CONFIG_READ,
    },

    Role.API_SERVICE: {
        Permission.STEAM_PROPERTIES_READ,
        Permission.STEAM_PROPERTIES_COMPUTE,
        Permission.BALANCE_READ,
        Permission.BALANCE_COMPUTE,
        Permission.OPTIMIZATION_READ,
        Permission.OPTIMIZATION_REQUEST,
        Permission.OPTIMIZATION_EXECUTE,
        Permission.TRAP_READ,
        Permission.TRAP_DIAGNOSE,
        Permission.RCA_READ,
        Permission.RCA_ANALYZE,
        Permission.KPI_READ,
        Permission.CLIMATE_READ,
        Permission.EXPLAINABILITY_READ,
        Permission.RECOMMENDATION_READ,
    },

    Role.STEAM_OPERATOR: {
        Permission.STEAM_PROPERTIES_READ,
        Permission.STEAM_PROPERTIES_COMPUTE,
        Permission.BALANCE_READ,
        Permission.BALANCE_COMPUTE,
        Permission.OPTIMIZATION_READ,
        Permission.OPTIMIZATION_REQUEST,
        Permission.KPI_READ,
        Permission.RECOMMENDATION_READ,
        Permission.RECOMMENDATION_ACKNOWLEDGE,
    },

    Role.TRAP_TECHNICIAN: {
        Permission.TRAP_READ,
        Permission.TRAP_DIAGNOSE,
        Permission.TRAP_MAINTAIN,
        Permission.KPI_READ,
        Permission.RECOMMENDATION_READ,
        Permission.RECOMMENDATION_ACKNOWLEDGE,
    },

    Role.VIEWER: {
        Permission.STEAM_PROPERTIES_READ,
        Permission.BALANCE_READ,
        Permission.OPTIMIZATION_READ,
        Permission.TRAP_READ,
        Permission.RCA_READ,
        Permission.KPI_READ,
        Permission.CLIMATE_READ,
        Permission.EXPLAINABILITY_READ,
        Permission.RECOMMENDATION_READ,
    },
}


# =============================================================================
# User Model
# =============================================================================

class SteamSystemUser(BaseModel):
    """Authenticated user model for SteamSystemOptimizer."""
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
    allowed_equipment_ids: Optional[List[str]] = None  # None = all equipment
    allowed_areas: Optional[List[str]] = None

    # Session info
    session_id: Optional[str] = None
    is_mfa_verified: bool = False
    auth_method: str = "jwt"  # jwt, api_key, mtls

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

    def can_access_equipment(self, equipment_id: str) -> bool:
        """Check if user can access a specific equipment."""
        if self.allowed_equipment_ids is None:
            return True  # No restrictions
        return equipment_id in self.allowed_equipment_ids

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
    scope: str = "openid profile email"


class TokenRefreshRequest(BaseModel):
    """Token refresh request."""
    refresh_token: str
    grant_type: str = "refresh_token"


# =============================================================================
# Security Utilities
# =============================================================================

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2 scheme for JWT bearer tokens
oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl="/api/v1/auth/token",
    scopes={
        "openid": "OpenID Connect scope",
        "profile": "User profile information",
        "email": "User email address",
        "steam": "Steam properties operations",
        "optimization": "Optimization operations",
        "traps": "Trap diagnostics operations",
        "analytics": "Analytics and reporting",
    },
)

# API Key authentication
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

# HTTP Bearer for explicit bearer token auth
http_bearer = HTTPBearer(auto_error=False)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash."""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Hash a password for storage."""
    return pwd_context.hash(password)


def create_access_token(
    user: SteamSystemUser,
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
    if user.allowed_equipment_ids:
        payload["equipment_ids"] = user.allowed_equipment_ids

    return jwt.encode(payload, config.jwt_secret_key, algorithm=config.jwt_algorithm)


def create_refresh_token(
    user: SteamSystemUser,
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
# mTLS Support
# =============================================================================

class MTLSContext:
    """Mutual TLS context management."""

    def __init__(self, config: AuthConfig):
        self.config = config
        self._ssl_context: Optional[ssl.SSLContext] = None

    def create_ssl_context(self) -> ssl.SSLContext:
        """Create SSL context for mTLS."""
        if not self.config.mtls_enabled:
            raise ValueError("mTLS is not enabled in configuration")

        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)

        # Load server certificate and key
        if self.config.mtls_server_cert_path and self.config.mtls_server_key_path:
            context.load_cert_chain(
                self.config.mtls_server_cert_path,
                self.config.mtls_server_key_path,
            )

        # Load CA certificate for client verification
        if self.config.mtls_ca_cert_path:
            context.load_verify_locations(self.config.mtls_ca_cert_path)

        # Require client certificate
        if self.config.mtls_verify_client:
            context.verify_mode = ssl.CERT_REQUIRED
        else:
            context.verify_mode = ssl.CERT_OPTIONAL

        self._ssl_context = context
        return context

    def get_ssl_context(self) -> Optional[ssl.SSLContext]:
        """Get the SSL context, creating if necessary."""
        if self._ssl_context is None and self.config.mtls_enabled:
            return self.create_ssl_context()
        return self._ssl_context


def extract_client_cert_info(request: Request) -> Optional[Dict[str, Any]]:
    """
    Extract client certificate information from request.

    Args:
        request: FastAPI request object

    Returns:
        Dictionary with client certificate info, or None if not present
    """
    client_cert = request.scope.get("transport", {}).get("peercert")

    if not client_cert:
        return None

    return {
        "subject": dict(x[0] for x in client_cert.get("subject", [])),
        "issuer": dict(x[0] for x in client_cert.get("issuer", [])),
        "serial_number": client_cert.get("serialNumber"),
        "not_before": client_cert.get("notBefore"),
        "not_after": client_cert.get("notAfter"),
    }


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


def generate_api_key(prefix: str = "steam_") -> Tuple[str, str]:
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
# Authentication Dependencies
# =============================================================================

# In-memory stores (replace with database in production)
_users_db: Dict[str, SteamSystemUser] = {}
_api_keys_db: Dict[str, APIKeyInfo] = {}
_revoked_tokens: Set[str] = set()


async def get_user_by_id(user_id: str) -> Optional[SteamSystemUser]:
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
    api_key: Optional[str] = Depends(api_key_header),
    bearer: Optional[HTTPAuthorizationCredentials] = Depends(http_bearer),
) -> SteamSystemUser:
    """
    Get the current authenticated user from the request.

    Supports multiple authentication methods:
    1. OAuth2 Bearer token (Authorization header)
    2. API Key (X-API-Key header)
    3. mTLS client certificate

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

            # Load user from database
            user = await get_user_by_id(token_data.sub)
            if user:
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
            user = SteamSystemUser(
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

    # Try mTLS client certificate
    if not user and config.mtls_enabled:
        cert_info = extract_client_cert_info(request)
        if cert_info:
            subject = cert_info.get("subject", {})
            cn = subject.get("commonName")
            if cn:
                user = await get_user_by_id(cn)
                if user:
                    auth_method = "mtls"
                    user.is_mfa_verified = True

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
        async def endpoint(user: SteamSystemUser = Depends(require_permissions(Permission.STEAM_PROPERTIES_READ))):
            ...
    """
    async def permission_dependency(
        user: SteamSystemUser = Depends(get_current_user),
    ) -> SteamSystemUser:
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
        user: SteamSystemUser = Depends(get_current_user),
    ) -> SteamSystemUser:
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
        user: SteamSystemUser = Depends(get_current_user),
    ) -> SteamSystemUser:
        if not any(role in user.roles for role in required_roles):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Required roles: {', '.join(r.value for r in required_roles)}",
            )
        return user

    return role_dependency


def require_equipment_access(equipment_id_param: str = "equipment_id"):
    """
    Dependency factory for requiring access to a specific equipment.
    """
    async def equipment_dependency(
        request: Request,
        user: SteamSystemUser = Depends(get_current_user),
    ) -> SteamSystemUser:
        # Get equipment_id from path or query parameters
        equipment_id = request.path_params.get(equipment_id_param) or request.query_params.get(equipment_id_param)

        if equipment_id:
            if not user.can_access_equipment(equipment_id):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Not authorized to access equipment {equipment_id}",
                )

        return user

    return equipment_dependency


async def authorize_action(
    user: SteamSystemUser,
    action: str,
    resource_type: str,
    resource_id: Optional[str] = None,
) -> bool:
    """
    Authorize a specific action on a resource.

    Args:
        user: The authenticated user
        action: Action to perform (read, compute, request, etc.)
        resource_type: Type of resource (steam, optimization, trap, etc.)
        resource_id: Optional specific resource ID

    Returns:
        True if authorized

    Raises:
        HTTPException: If not authorized
    """
    # Build permission from action and resource type
    permission_str = f"{resource_type}:{action}"

    try:
        permission = Permission(permission_str)
    except ValueError:
        logger.warning(f"Unknown permission requested: {permission_str}")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Unknown permission: {permission_str}",
        )

    if not user.has_permission(permission):
        logger.warning(
            f"Authorization denied: user {user.username} lacks {permission_str} "
            f"for resource {resource_id}"
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Not authorized for action '{action}' on {resource_type}",
        )

    # Check resource-specific access
    if resource_id and resource_type in ["trap", "equipment"]:
        if not user.can_access_equipment(resource_id):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Not authorized to access {resource_type} {resource_id}",
            )

    logger.debug(f"Authorization granted: {user.username} -> {permission_str}")
    return True


# =============================================================================
# Audit Logging
# =============================================================================

class AuditLogEntry(BaseModel):
    """Audit log entry for security events."""
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


async def log_security_event(
    event_type: str,
    action: str,
    resource_type: str,
    request: Request,
    user: Optional[SteamSystemUser] = None,
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
    logger.info(f"AUDIT: {entry.model_dump_json()}")

    return entry
