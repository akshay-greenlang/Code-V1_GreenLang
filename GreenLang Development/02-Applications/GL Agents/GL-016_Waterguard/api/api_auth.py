"""
GL-016_Waterguard API Authentication

JWT-based authentication and authorization for the Waterguard API.
Implements role-based access control (RBAC) and comprehensive audit logging.

Author: GL-APIDeveloper
Version: 1.0.0
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timedelta
from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Set

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer, OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel, Field

from api.config import get_api_config

logger = logging.getLogger(__name__)

# =============================================================================
# Security Configuration
# =============================================================================

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2 scheme for token authentication
oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl="/api/v1/auth/token",
    auto_error=False
)

# HTTP Bearer scheme for API key authentication
bearer_scheme = HTTPBearer(auto_error=False)


# =============================================================================
# Enumerations
# =============================================================================

class UserRole(str, Enum):
    """User roles for RBAC."""
    OPERATOR = "operator"      # Basic operations, view data, approve recommendations
    ENGINEER = "engineer"      # All operator + modify setpoints, run optimizations
    ADMIN = "admin"           # Full access including user management
    SERVICE = "service"       # Service account for integrations
    READONLY = "readonly"     # Read-only access for auditors


class Permission(str, Enum):
    """Granular permissions for fine-grained access control."""
    # Read permissions
    READ_CHEMISTRY = "read:chemistry"
    READ_RECOMMENDATIONS = "read:recommendations"
    READ_COMPLIANCE = "read:compliance"
    READ_SAVINGS = "read:savings"
    READ_BLOWDOWN = "read:blowdown"
    READ_DOSING = "read:dosing"

    # Write permissions
    WRITE_RECOMMENDATIONS = "write:recommendations"
    WRITE_SETPOINTS = "write:setpoints"
    WRITE_OPERATING_MODE = "write:operating_mode"

    # Execute permissions
    EXECUTE_OPTIMIZATION = "execute:optimization"
    APPROVE_RECOMMENDATIONS = "approve:recommendations"

    # Admin permissions
    MANAGE_USERS = "manage:users"
    MANAGE_CONFIG = "manage:config"
    VIEW_AUDIT_LOGS = "view:audit_logs"


# Role to permissions mapping
ROLE_PERMISSIONS: Dict[UserRole, Set[Permission]] = {
    UserRole.READONLY: {
        Permission.READ_CHEMISTRY,
        Permission.READ_RECOMMENDATIONS,
        Permission.READ_COMPLIANCE,
        Permission.READ_SAVINGS,
        Permission.READ_BLOWDOWN,
        Permission.READ_DOSING,
    },
    UserRole.OPERATOR: {
        Permission.READ_CHEMISTRY,
        Permission.READ_RECOMMENDATIONS,
        Permission.READ_COMPLIANCE,
        Permission.READ_SAVINGS,
        Permission.READ_BLOWDOWN,
        Permission.READ_DOSING,
        Permission.APPROVE_RECOMMENDATIONS,
    },
    UserRole.ENGINEER: {
        Permission.READ_CHEMISTRY,
        Permission.READ_RECOMMENDATIONS,
        Permission.READ_COMPLIANCE,
        Permission.READ_SAVINGS,
        Permission.READ_BLOWDOWN,
        Permission.READ_DOSING,
        Permission.WRITE_RECOMMENDATIONS,
        Permission.WRITE_SETPOINTS,
        Permission.WRITE_OPERATING_MODE,
        Permission.EXECUTE_OPTIMIZATION,
        Permission.APPROVE_RECOMMENDATIONS,
    },
    UserRole.ADMIN: set(Permission),  # All permissions
    UserRole.SERVICE: {
        Permission.READ_CHEMISTRY,
        Permission.READ_RECOMMENDATIONS,
        Permission.READ_COMPLIANCE,
        Permission.READ_SAVINGS,
        Permission.READ_BLOWDOWN,
        Permission.READ_DOSING,
        Permission.EXECUTE_OPTIMIZATION,
    },
}


# =============================================================================
# Data Models
# =============================================================================

class TokenPayload(BaseModel):
    """JWT token payload."""
    sub: str = Field(..., description="Subject (user ID)")
    email: str = Field(..., description="User email")
    name: str = Field(..., description="User display name")
    roles: List[str] = Field(default_factory=list, description="User roles")
    permissions: List[str] = Field(default_factory=list, description="Explicit permissions")
    tenant_id: Optional[str] = Field(None, description="Tenant ID for multi-tenancy")
    tower_ids: List[str] = Field(default_factory=list, description="Authorized tower IDs")
    exp: datetime = Field(..., description="Token expiration")
    iat: datetime = Field(..., description="Token issued at")
    jti: str = Field(..., description="JWT ID for revocation")


class User(BaseModel):
    """Authenticated user model."""
    id: str = Field(..., description="User ID")
    email: str = Field(..., description="User email")
    name: str = Field(..., description="Display name")
    roles: List[UserRole] = Field(default_factory=list)
    permissions: Set[Permission] = Field(default_factory=set)
    tenant_id: Optional[str] = None
    tower_ids: List[str] = Field(default_factory=list)
    is_active: bool = True

    class Config:
        use_enum_values = True

    def has_role(self, role: UserRole) -> bool:
        """Check if user has a specific role."""
        return role in self.roles or UserRole.ADMIN in self.roles

    def has_permission(self, permission: Permission) -> bool:
        """Check if user has a specific permission."""
        # Check explicit permissions
        if permission in self.permissions:
            return True

        # Check role-based permissions
        for role in self.roles:
            if permission in ROLE_PERMISSIONS.get(role, set()):
                return True

        return False

    def can_access_tower(self, tower_id: str) -> bool:
        """Check if user can access a specific tower."""
        # Admin can access all towers
        if UserRole.ADMIN in self.roles:
            return True

        # Empty list means access to all towers in tenant
        if not self.tower_ids:
            return True

        return tower_id in self.tower_ids


class TokenResponse(BaseModel):
    """Token response model."""
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    refresh_token: Optional[str] = None
    scope: Optional[str] = None


class AuditLogEntry(BaseModel):
    """Audit log entry for API calls."""
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: Optional[str] = None
    user_email: Optional[str] = None
    user_roles: List[str] = Field(default_factory=list)
    action: str = Field(..., description="API action performed")
    resource: str = Field(..., description="Resource accessed")
    resource_id: Optional[str] = None
    method: str = Field(..., description="HTTP method")
    path: str = Field(..., description="Request path")
    status_code: int = Field(..., description="Response status code")
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    duration_ms: Optional[float] = None
    request_body: Optional[Dict[str, Any]] = None
    response_summary: Optional[str] = None
    error_message: Optional[str] = None


# =============================================================================
# JWT Authenticator
# =============================================================================

class JWTAuthenticator:
    """
    JWT authentication handler for the Waterguard API.

    Provides token creation, validation, and user extraction from JWT tokens.
    """

    def __init__(self):
        """Initialize the JWT authenticator."""
        self.config = get_api_config().jwt
        self._revoked_tokens: Set[str] = set()  # In production, use Redis
        self._audit_logs: List[AuditLogEntry] = []  # In production, use database

    def create_access_token(
        self,
        user_id: str,
        email: str,
        name: str,
        roles: List[str],
        permissions: Optional[List[str]] = None,
        tenant_id: Optional[str] = None,
        tower_ids: Optional[List[str]] = None,
        expires_delta: Optional[timedelta] = None,
    ) -> str:
        """
        Create a new JWT access token.

        Args:
            user_id: User identifier
            email: User email
            name: User display name
            roles: List of user roles
            permissions: Explicit permissions (optional)
            tenant_id: Tenant ID for multi-tenancy
            tower_ids: Authorized tower IDs
            expires_delta: Custom expiration time

        Returns:
            Encoded JWT token string
        """
        now = datetime.utcnow()

        if expires_delta:
            expire = now + expires_delta
        else:
            expire = now + timedelta(seconds=self.config.access_token_expire_seconds)

        payload = {
            "sub": user_id,
            "email": email,
            "name": name,
            "roles": roles,
            "permissions": permissions or [],
            "tenant_id": tenant_id,
            "tower_ids": tower_ids or [],
            "exp": expire,
            "iat": now,
            "jti": str(uuid.uuid4()),
            "iss": self.config.issuer,
            "aud": self.config.audience,
        }

        token = jwt.encode(
            payload,
            self.config.secret_key,
            algorithm=self.config.algorithm
        )

        logger.info(f"Created access token for user {user_id}")
        return token

    def create_refresh_token(
        self,
        user_id: str,
        expires_delta: Optional[timedelta] = None,
    ) -> str:
        """
        Create a refresh token.

        Args:
            user_id: User identifier
            expires_delta: Custom expiration time

        Returns:
            Encoded refresh token string
        """
        now = datetime.utcnow()

        if expires_delta:
            expire = now + expires_delta
        else:
            expire = now + timedelta(seconds=self.config.refresh_token_expire_seconds)

        payload = {
            "sub": user_id,
            "type": "refresh",
            "exp": expire,
            "iat": now,
            "jti": str(uuid.uuid4()),
        }

        return jwt.encode(
            payload,
            self.config.secret_key,
            algorithm=self.config.algorithm
        )

    def validate_token(self, token: str) -> TokenPayload:
        """
        Validate a JWT token and extract payload.

        Args:
            token: JWT token string

        Returns:
            TokenPayload with user information

        Raises:
            HTTPException: If token is invalid or expired
        """
        try:
            payload = jwt.decode(
                token,
                self.config.secret_key,
                algorithms=[self.config.algorithm],
                audience=self.config.audience,
                issuer=self.config.issuer,
            )

            # Check if token is revoked
            jti = payload.get("jti")
            if jti and jti in self._revoked_tokens:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Token has been revoked",
                    headers={"WWW-Authenticate": "Bearer"},
                )

            return TokenPayload(
                sub=payload["sub"],
                email=payload["email"],
                name=payload["name"],
                roles=payload.get("roles", []),
                permissions=payload.get("permissions", []),
                tenant_id=payload.get("tenant_id"),
                tower_ids=payload.get("tower_ids", []),
                exp=datetime.fromtimestamp(payload["exp"]),
                iat=datetime.fromtimestamp(payload["iat"]),
                jti=payload["jti"],
            )

        except jwt.ExpiredSignatureError:
            logger.warning("Token has expired")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired",
                headers={"WWW-Authenticate": "Bearer"},
            )

        except JWTError as e:
            logger.warning(f"Invalid token: {e}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token",
                headers={"WWW-Authenticate": "Bearer"},
            )

    def revoke_token(self, jti: str) -> None:
        """
        Revoke a token by its JTI.

        Args:
            jti: JWT ID to revoke
        """
        self._revoked_tokens.add(jti)
        logger.info(f"Revoked token {jti}")

    def get_user_from_token(self, token_payload: TokenPayload) -> User:
        """
        Create User object from token payload.

        Args:
            token_payload: Validated token payload

        Returns:
            User object with roles and permissions
        """
        roles = [UserRole(r) for r in token_payload.roles if r in UserRole.__members__.values()]

        # Build permission set from roles and explicit permissions
        permissions: Set[Permission] = set()
        for role in roles:
            permissions.update(ROLE_PERMISSIONS.get(role, set()))

        # Add explicit permissions
        for perm in token_payload.permissions:
            try:
                permissions.add(Permission(perm))
            except ValueError:
                pass  # Ignore invalid permissions

        return User(
            id=token_payload.sub,
            email=token_payload.email,
            name=token_payload.name,
            roles=roles,
            permissions=permissions,
            tenant_id=token_payload.tenant_id,
            tower_ids=token_payload.tower_ids,
        )


# Global authenticator instance
_authenticator: Optional[JWTAuthenticator] = None


def get_authenticator() -> JWTAuthenticator:
    """Get or create the JWT authenticator instance."""
    global _authenticator
    if _authenticator is None:
        _authenticator = JWTAuthenticator()
    return _authenticator


# =============================================================================
# Audit Logger
# =============================================================================

class AuditLogger:
    """
    Audit logger for tracking all API calls.

    Logs user actions, resource access, and security events for compliance.
    """

    def __init__(self):
        """Initialize the audit logger."""
        self._logs: List[AuditLogEntry] = []
        self._logger = logging.getLogger("waterguard.audit")

    async def log(
        self,
        request: Request,
        user: Optional[User],
        action: str,
        resource: str,
        resource_id: Optional[str] = None,
        status_code: int = 200,
        duration_ms: Optional[float] = None,
        request_body: Optional[Dict[str, Any]] = None,
        response_summary: Optional[str] = None,
        error_message: Optional[str] = None,
    ) -> AuditLogEntry:
        """
        Log an API call to the audit trail.

        Args:
            request: FastAPI request object
            user: Authenticated user (if any)
            action: Action performed
            resource: Resource accessed
            resource_id: Specific resource ID
            status_code: HTTP response status
            duration_ms: Request duration in milliseconds
            request_body: Sanitized request body
            response_summary: Brief response summary
            error_message: Error message if applicable

        Returns:
            Created audit log entry
        """
        # Get client IP
        client_ip = request.client.host if request.client else None

        # Get request ID from headers or generate
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))

        entry = AuditLogEntry(
            request_id=request_id,
            user_id=user.id if user else None,
            user_email=user.email if user else None,
            user_roles=[r.value for r in user.roles] if user else [],
            action=action,
            resource=resource,
            resource_id=resource_id,
            method=request.method,
            path=str(request.url.path),
            status_code=status_code,
            ip_address=client_ip,
            user_agent=request.headers.get("User-Agent"),
            duration_ms=duration_ms,
            request_body=self._sanitize_request_body(request_body),
            response_summary=response_summary,
            error_message=error_message,
        )

        # Store in memory (in production, persist to database)
        self._logs.append(entry)

        # Log to file/stdout
        log_message = (
            f"AUDIT | {entry.timestamp.isoformat()} | "
            f"{entry.method} {entry.path} | "
            f"User: {entry.user_email or 'anonymous'} | "
            f"Action: {entry.action} | "
            f"Status: {entry.status_code}"
        )

        if entry.error_message:
            self._logger.warning(log_message + f" | Error: {entry.error_message}")
        else:
            self._logger.info(log_message)

        return entry

    def _sanitize_request_body(
        self,
        body: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """
        Sanitize request body by removing sensitive fields.

        Args:
            body: Raw request body

        Returns:
            Sanitized body with sensitive fields redacted
        """
        if not body:
            return None

        sensitive_fields = {"password", "token", "secret", "api_key", "authorization"}
        sanitized = {}

        for key, value in body.items():
            if key.lower() in sensitive_fields:
                sanitized[key] = "[REDACTED]"
            elif isinstance(value, dict):
                sanitized[key] = self._sanitize_request_body(value)
            else:
                sanitized[key] = value

        return sanitized

    def get_logs(
        self,
        user_id: Optional[str] = None,
        action: Optional[str] = None,
        since: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[AuditLogEntry]:
        """
        Retrieve audit logs with optional filtering.

        Args:
            user_id: Filter by user ID
            action: Filter by action
            since: Filter logs after this timestamp
            limit: Maximum entries to return

        Returns:
            List of matching audit log entries
        """
        logs = self._logs

        if user_id:
            logs = [l for l in logs if l.user_id == user_id]
        if action:
            logs = [l for l in logs if l.action == action]
        if since:
            logs = [l for l in logs if l.timestamp >= since]

        return logs[-limit:]


# Global audit logger instance
_audit_logger: Optional[AuditLogger] = None


def get_audit_logger() -> AuditLogger:
    """Get or create the audit logger instance."""
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = AuditLogger()
    return _audit_logger


# =============================================================================
# Dependency Injection Functions
# =============================================================================

async def get_token_from_header(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
    token: Optional[str] = Depends(oauth2_scheme),
) -> str:
    """
    Extract JWT token from request headers.

    Supports both Bearer token and OAuth2 password bearer schemes.

    Args:
        credentials: HTTP Bearer credentials
        token: OAuth2 token

    Returns:
        JWT token string

    Raises:
        HTTPException: If no valid token found
    """
    if credentials and credentials.credentials:
        return credentials.credentials
    if token:
        return token

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Not authenticated",
        headers={"WWW-Authenticate": "Bearer"},
    )


async def get_current_user(
    token: str = Depends(get_token_from_header),
) -> User:
    """
    Get the current authenticated user from JWT token.

    Args:
        token: JWT token from request

    Returns:
        Authenticated User object

    Raises:
        HTTPException: If authentication fails
    """
    authenticator = get_authenticator()
    token_payload = authenticator.validate_token(token)
    user = authenticator.get_user_from_token(token_payload)

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is disabled",
        )

    return user


async def get_current_user_optional(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
    token: Optional[str] = Depends(oauth2_scheme),
) -> Optional[User]:
    """
    Get the current user if authenticated, otherwise return None.

    Useful for endpoints that have optional authentication.

    Args:
        credentials: HTTP Bearer credentials
        token: OAuth2 token

    Returns:
        User object or None
    """
    try:
        token_str = None
        if credentials and credentials.credentials:
            token_str = credentials.credentials
        elif token:
            token_str = token

        if not token_str:
            return None

        authenticator = get_authenticator()
        token_payload = authenticator.validate_token(token_str)
        return authenticator.get_user_from_token(token_payload)
    except HTTPException:
        return None


def require_role(*roles: UserRole) -> Callable:
    """
    Dependency that requires the user to have one of the specified roles.

    Args:
        roles: Required roles (user must have at least one)

    Returns:
        Dependency function

    Example:
        @app.get("/admin")
        async def admin_endpoint(user: User = Depends(require_role(UserRole.ADMIN))):
            ...
    """
    async def role_checker(user: User = Depends(get_current_user)) -> User:
        if not any(user.has_role(role) for role in roles):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Requires one of roles: {[r.value for r in roles]}",
            )
        return user

    return role_checker


def require_permission(*permissions: Permission) -> Callable:
    """
    Dependency that requires the user to have all specified permissions.

    Args:
        permissions: Required permissions

    Returns:
        Dependency function

    Example:
        @app.post("/optimize")
        async def optimize(user: User = Depends(require_permission(Permission.EXECUTE_OPTIMIZATION))):
            ...
    """
    async def permission_checker(user: User = Depends(get_current_user)) -> User:
        missing = [p for p in permissions if not user.has_permission(p)]
        if missing:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Missing permissions: {[p.value for p in missing]}",
            )
        return user

    return permission_checker


def require_tower_access(tower_id_param: str = "tower_id") -> Callable:
    """
    Dependency that requires the user to have access to the specified tower.

    Args:
        tower_id_param: Name of the path/query parameter containing tower ID

    Returns:
        Dependency function
    """
    async def tower_access_checker(
        request: Request,
        user: User = Depends(get_current_user),
    ) -> User:
        # Get tower_id from path or query parameters
        tower_id = request.path_params.get(tower_id_param)
        if not tower_id:
            tower_id = request.query_params.get(tower_id_param)

        if tower_id and not user.can_access_tower(tower_id):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"No access to tower: {tower_id}",
            )

        return user

    return tower_access_checker


# =============================================================================
# Utility Functions
# =============================================================================

def hash_password(password: str) -> str:
    """
    Hash a password using bcrypt.

    Args:
        password: Plain text password

    Returns:
        Hashed password string
    """
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify a password against its hash.

    Args:
        plain_password: Plain text password to verify
        hashed_password: Stored password hash

    Returns:
        True if password matches
    """
    return pwd_context.verify(plain_password, hashed_password)


def generate_api_key() -> str:
    """
    Generate a new API key.

    Returns:
        API key string with prefix
    """
    config = get_api_config().jwt
    key = uuid.uuid4().hex + uuid.uuid4().hex
    return f"{config.api_key_prefix}{key}"
