"""
GL-004 BURNMASTER API Authentication and Authorization

JWT token validation, role-based access control, API key authentication,
and audit logging for all authenticated requests.
"""

from fastapi import Depends, HTTPException, status, Request, Security
from fastapi.security import OAuth2PasswordBearer, APIKeyHeader, HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Callable
from datetime import datetime, timedelta
from functools import wraps
import hashlib
import secrets
import logging
import uuid
from enum import Enum

from .config import get_settings
from .api_schemas import UserRole

logger = logging.getLogger(__name__)
settings = get_settings()

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/token", auto_error=False)

# API Key header
api_key_header = APIKeyHeader(name=settings.security.api_key_header, auto_error=False)

# HTTP Bearer for JWT
http_bearer = HTTPBearer(auto_error=False)


# ============================================================================
# Models
# ============================================================================

class Token(BaseModel):
    """OAuth2 token response."""
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    refresh_token: Optional[str] = None


class TokenData(BaseModel):
    """Decoded token data."""
    user_id: str
    email: str
    roles: List[UserRole]
    tenant_id: Optional[str] = None
    exp: datetime
    iat: datetime
    jti: str


class User(BaseModel):
    """User model for authentication."""
    id: str
    email: str
    full_name: str
    roles: List[UserRole]
    tenant_id: Optional[str] = None
    is_active: bool = True
    is_verified: bool = True
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_login: Optional[datetime] = None


class APIKey(BaseModel):
    """API key model."""
    key_id: str
    key_hash: str
    name: str
    description: Optional[str] = None
    owner_id: str
    tenant_id: Optional[str] = None
    roles: List[UserRole]
    is_active: bool = True
    rate_limit: Optional[str] = None
    expires_at: Optional[datetime] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_used: Optional[datetime] = None


class AuditLogEntry(BaseModel):
    """Audit log entry."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    user_id: str
    user_email: str
    tenant_id: Optional[str] = None
    action: str
    resource_type: str
    resource_id: Optional[str] = None
    request_method: str
    request_path: str
    request_id: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    status_code: int
    response_time_ms: float
    details: Dict[str, Any] = Field(default_factory=dict)


# ============================================================================
# Authentication Functions
# ============================================================================

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a plain password against its hash."""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Hash a password."""
    return pwd_context.hash(password)


def create_access_token(
    user: User,
    expires_delta: Optional[timedelta] = None
) -> str:
    """
    Create JWT access token.

    Args:
        user: User object
        expires_delta: Optional custom expiration time

    Returns:
        Encoded JWT token
    """
    if expires_delta is None:
        expires_delta = timedelta(minutes=settings.security.jwt_access_token_expire_minutes)

    now = datetime.utcnow()
    expire = now + expires_delta

    to_encode = {
        "sub": user.id,
        "email": user.email,
        "roles": [role.value for role in user.roles],
        "tenant_id": user.tenant_id,
        "exp": expire,
        "iat": now,
        "jti": str(uuid.uuid4()),
        "type": "access"
    }

    encoded_jwt = jwt.encode(
        to_encode,
        settings.security.jwt_secret_key.get_secret_value(),
        algorithm=settings.security.jwt_algorithm
    )

    return encoded_jwt


def create_refresh_token(user: User) -> str:
    """Create JWT refresh token."""
    expires_delta = timedelta(days=settings.security.jwt_refresh_token_expire_days)
    now = datetime.utcnow()
    expire = now + expires_delta

    to_encode = {
        "sub": user.id,
        "exp": expire,
        "iat": now,
        "jti": str(uuid.uuid4()),
        "type": "refresh"
    }

    encoded_jwt = jwt.encode(
        to_encode,
        settings.security.jwt_secret_key.get_secret_value(),
        algorithm=settings.security.jwt_algorithm
    )

    return encoded_jwt


def decode_token(token: str) -> TokenData:
    """
    Decode and validate JWT token.

    Args:
        token: JWT token string

    Returns:
        TokenData object

    Raises:
        HTTPException: If token is invalid or expired
    """
    try:
        payload = jwt.decode(
            token,
            settings.security.jwt_secret_key.get_secret_value(),
            algorithms=[settings.security.jwt_algorithm]
        )

        user_id: str = payload.get("sub")
        if user_id is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token: missing subject",
                headers={"WWW-Authenticate": "Bearer"},
            )

        token_data = TokenData(
            user_id=user_id,
            email=payload.get("email", ""),
            roles=[UserRole(r) for r in payload.get("roles", [])],
            tenant_id=payload.get("tenant_id"),
            exp=datetime.fromtimestamp(payload.get("exp", 0)),
            iat=datetime.fromtimestamp(payload.get("iat", 0)),
            jti=payload.get("jti", "")
        )

        return token_data

    except JWTError as e:
        logger.warning(f"JWT decode error: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )


# ============================================================================
# API Key Functions
# ============================================================================

def generate_api_key() -> tuple[str, str]:
    """
    Generate a new API key.

    Returns:
        Tuple of (key_id, api_key)
    """
    key_id = f"{settings.security.api_key_prefix}{secrets.token_hex(8)}"
    api_key = secrets.token_urlsafe(32)
    return key_id, api_key


def hash_api_key(api_key: str) -> str:
    """Hash an API key for storage."""
    return hashlib.sha256(api_key.encode()).hexdigest()


async def validate_api_key(api_key: str) -> Optional[APIKey]:
    """
    Validate an API key.

    Args:
        api_key: The API key to validate

    Returns:
        APIKey object if valid, None otherwise
    """
    # In production, this would query a database
    # For now, return a mock API key for demonstration
    if api_key.startswith(settings.security.api_key_prefix):
        # Mock validation - replace with actual database lookup
        return APIKey(
            key_id=api_key[:16],
            key_hash=hash_api_key(api_key),
            name="Service API Key",
            owner_id="system",
            roles=[UserRole.OPERATOR]
        )
    return None


# ============================================================================
# Dependency Functions
# ============================================================================

async def get_current_user_from_token(
    credentials: Optional[HTTPAuthorizationCredentials] = Security(http_bearer)
) -> Optional[User]:
    """Get current user from JWT token."""
    if credentials is None:
        return None

    token_data = decode_token(credentials.credentials)

    # In production, look up user from database
    # For now, construct user from token data
    user = User(
        id=token_data.user_id,
        email=token_data.email,
        full_name="",
        roles=token_data.roles,
        tenant_id=token_data.tenant_id
    )

    return user


async def get_current_user_from_api_key(
    api_key: Optional[str] = Security(api_key_header)
) -> Optional[User]:
    """Get current user from API key."""
    if api_key is None or not settings.security.enable_api_key_auth:
        return None

    api_key_obj = await validate_api_key(api_key)
    if api_key_obj is None or not api_key_obj.is_active:
        return None

    # Check expiration
    if api_key_obj.expires_at and api_key_obj.expires_at < datetime.utcnow():
        return None

    # Create user from API key
    user = User(
        id=api_key_obj.owner_id,
        email=f"{api_key_obj.name}@api.greenlang.io",
        full_name=api_key_obj.name,
        roles=api_key_obj.roles,
        tenant_id=api_key_obj.tenant_id
    )

    return user


async def get_current_user(
    token_user: Optional[User] = Depends(get_current_user_from_token),
    api_key_user: Optional[User] = Depends(get_current_user_from_api_key)
) -> User:
    """
    Get current authenticated user from either JWT token or API key.

    Args:
        token_user: User from JWT token
        api_key_user: User from API key

    Returns:
        Authenticated user

    Raises:
        HTTPException: If not authenticated
    """
    user = token_user or api_key_user

    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is disabled"
        )

    return user


async def get_optional_user(
    token_user: Optional[User] = Depends(get_current_user_from_token),
    api_key_user: Optional[User] = Depends(get_current_user_from_api_key)
) -> Optional[User]:
    """Get current user if authenticated, None otherwise."""
    return token_user or api_key_user


# ============================================================================
# Role-Based Access Control
# ============================================================================

class RoleChecker:
    """
    Role-based access control checker.

    Usage:
        @app.get("/admin")
        async def admin_endpoint(user: User = Depends(require_roles([UserRole.ADMIN]))):
            return {"message": "Admin access granted"}
    """

    def __init__(self, required_roles: List[UserRole]):
        self.required_roles = required_roles

    async def __call__(self, user: User = Depends(get_current_user)) -> User:
        """Check if user has required roles."""
        user_roles = set(user.roles)
        required = set(self.required_roles)

        if not user_roles.intersection(required):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient permissions. Required roles: {[r.value for r in self.required_roles]}"
            )

        return user


def require_roles(roles: List[UserRole]) -> RoleChecker:
    """Create a role checker dependency."""
    return RoleChecker(roles)


def require_operator() -> RoleChecker:
    """Require operator or higher role."""
    return RoleChecker([UserRole.OPERATOR, UserRole.ENGINEER, UserRole.ADMIN])


def require_engineer() -> RoleChecker:
    """Require engineer or higher role."""
    return RoleChecker([UserRole.ENGINEER, UserRole.ADMIN])


def require_admin() -> RoleChecker:
    """Require admin role."""
    return RoleChecker([UserRole.ADMIN])


# ============================================================================
# Audit Logging
# ============================================================================

class AuditLogger:
    """Audit logger for tracking authenticated requests."""

    def __init__(self):
        self._entries: List[AuditLogEntry] = []

    async def log(
        self,
        user: User,
        request: Request,
        action: str,
        resource_type: str,
        resource_id: Optional[str] = None,
        status_code: int = 200,
        response_time_ms: float = 0,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Log an audit entry.

        Args:
            user: Authenticated user
            request: FastAPI request object
            action: Action performed (e.g., "read", "update", "delete")
            resource_type: Type of resource accessed
            resource_id: Optional resource identifier
            status_code: HTTP status code
            response_time_ms: Response time in milliseconds
            details: Additional details to log
        """
        if not settings.audit.enabled:
            return

        entry = AuditLogEntry(
            user_id=user.id,
            user_email=user.email,
            tenant_id=user.tenant_id,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            request_method=request.method,
            request_path=str(request.url.path),
            request_id=request.headers.get("X-Request-ID"),
            ip_address=request.client.host if request.client else None,
            user_agent=request.headers.get("User-Agent"),
            status_code=status_code,
            response_time_ms=response_time_ms,
            details=self._sanitize_details(details or {})
        )

        # In production, persist to database or external service
        self._entries.append(entry)

        logger.info(
            f"AUDIT: user={user.email} action={action} resource={resource_type}/{resource_id} "
            f"status={status_code} time={response_time_ms:.2f}ms"
        )

    def _sanitize_details(self, details: Dict[str, Any]) -> Dict[str, Any]:
        """Remove sensitive fields from audit details."""
        sanitized = {}
        for key, value in details.items():
            if any(sensitive in key.lower() for sensitive in settings.audit.sensitive_fields):
                sanitized[key] = "[REDACTED]"
            else:
                sanitized[key] = value
        return sanitized

    def get_entries(
        self,
        user_id: Optional[str] = None,
        action: Optional[str] = None,
        resource_type: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100
    ) -> List[AuditLogEntry]:
        """Query audit log entries."""
        entries = self._entries

        if user_id:
            entries = [e for e in entries if e.user_id == user_id]
        if action:
            entries = [e for e in entries if e.action == action]
        if resource_type:
            entries = [e for e in entries if e.resource_type == resource_type]
        if start_time:
            entries = [e for e in entries if e.timestamp >= start_time]
        if end_time:
            entries = [e for e in entries if e.timestamp <= end_time]

        return entries[-limit:]


# Global audit logger instance
audit_logger = AuditLogger()


# ============================================================================
# Middleware and Utilities
# ============================================================================

async def get_request_id(request: Request) -> str:
    """Get or generate request ID."""
    return request.headers.get("X-Request-ID", str(uuid.uuid4()))


def audit_action(action: str, resource_type: str):
    """
    Decorator for auditing API actions.

    Usage:
        @app.post("/units/{unit_id}/mode")
        @audit_action("change_mode", "unit")
        async def change_mode(unit_id: str, user: User = Depends(get_current_user)):
            ...
    """
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            request = kwargs.get("request")
            user = kwargs.get("current_user") or kwargs.get("user")
            resource_id = kwargs.get("unit_id") or kwargs.get("rec_id") or kwargs.get("alert_id")

            start_time = datetime.utcnow()
            try:
                result = await func(*args, **kwargs)
                status_code = 200
                return result
            except HTTPException as e:
                status_code = e.status_code
                raise
            except Exception as e:
                status_code = 500
                raise
            finally:
                if request and user:
                    response_time = (datetime.utcnow() - start_time).total_seconds() * 1000
                    await audit_logger.log(
                        user=user,
                        request=request,
                        action=action,
                        resource_type=resource_type,
                        resource_id=resource_id,
                        status_code=status_code,
                        response_time_ms=response_time
                    )
        return wrapper
    return decorator
