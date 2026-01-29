"""
Authentication middleware for GL Normalizer Service.

This module provides authentication middleware that validates API keys and
JWT tokens before requests reach route handlers. It supports both API key
and Bearer token authentication methods.

Authentication Methods:
    1. API Key: X-API-Key header or "ApiKey" Authorization scheme
    2. JWT Bearer: "Bearer" Authorization scheme

Usage:
    >>> from fastapi import FastAPI
    >>> from gl_normalizer_service.middleware.auth import AuthMiddleware
    >>>
    >>> app = FastAPI()
    >>> app.add_middleware(AuthMiddleware)

Configuration:
    Environment variables:
    - GL_NORMALIZER_SECRET_KEY: JWT signing key
    - GL_NORMALIZER_API_KEY_HEADER: Custom API key header name
"""

import time
from typing import Callable, Optional
from uuid import uuid4

import structlog
from fastapi import Request, Response
from jose import JWTError, jwt
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from gl_normalizer_service.config import Settings, get_settings

logger = structlog.get_logger(__name__)


class AuthMiddleware(BaseHTTPMiddleware):
    """
    Authentication middleware for API key and JWT validation.

    This middleware intercepts all requests and validates authentication
    credentials before passing to route handlers. Public endpoints can be
    configured to bypass authentication.

    Attributes:
        public_paths: Set of paths that don't require authentication
        settings: Application settings

    Example:
        >>> app.add_middleware(
        ...     AuthMiddleware,
        ...     public_paths={"/v1/health", "/v1/ready", "/v1/live", "/docs"}
        ... )
    """

    def __init__(
        self,
        app,
        public_paths: Optional[set[str]] = None,
        settings: Optional[Settings] = None,
    ):
        """
        Initialize authentication middleware.

        Args:
            app: ASGI application
            public_paths: Paths that don't require authentication
            settings: Application settings (uses default if not provided)
        """
        super().__init__(app)
        self.public_paths = public_paths or {
            "/v1/health",
            "/v1/ready",
            "/v1/live",
            "/docs",
            "/redoc",
            "/openapi.json",
            "/api/docs",
            "/api/redoc",
            "/api/openapi.json",
        }
        self.settings = settings or get_settings()

    async def dispatch(
        self, request: Request, call_next: Callable
    ) -> Response:
        """
        Process request through authentication middleware.

        Args:
            request: Incoming HTTP request
            call_next: Next middleware/handler in chain

        Returns:
            Response from handler or authentication error
        """
        # Generate request ID if not present
        request_id = request.headers.get("X-Request-ID", f"req_{uuid4().hex[:12]}")
        request.state.request_id = request_id

        # Check if path is public
        path = request.url.path
        if self._is_public_path(path):
            return await call_next(request)

        # Extract credentials
        api_key = self._extract_api_key(request)
        bearer_token = self._extract_bearer_token(request)

        # Validate credentials
        if api_key:
            user = await self._validate_api_key(api_key)
            if user:
                request.state.user = user
                request.state.auth_method = "api_key"
                return await call_next(request)

        if bearer_token:
            user = await self._validate_jwt(bearer_token)
            if user:
                request.state.user = user
                request.state.auth_method = "jwt"
                return await call_next(request)

        # Authentication failed
        logger.warning(
            "auth_failed",
            request_id=request_id,
            path=path,
            method=request.method,
            has_api_key=bool(api_key),
            has_bearer=bool(bearer_token),
        )

        return JSONResponse(
            status_code=401,
            content={
                "api_revision": self.settings.api_revision,
                "error": {
                    "code": "GLNORM-007",
                    "message": "Authentication required. Provide X-API-Key or Bearer token.",
                },
                "request_id": request_id,
            },
            headers={"WWW-Authenticate": "Bearer"},
        )

    def _is_public_path(self, path: str) -> bool:
        """
        Check if path is public (no auth required).

        Args:
            path: Request path

        Returns:
            True if path is public
        """
        # Exact match
        if path in self.public_paths:
            return True

        # Prefix match for docs
        for public_path in self.public_paths:
            if path.startswith(public_path):
                return True

        return False

    def _extract_api_key(self, request: Request) -> Optional[str]:
        """
        Extract API key from request headers.

        Args:
            request: HTTP request

        Returns:
            API key string or None
        """
        # Check X-API-Key header
        api_key = request.headers.get(self.settings.api_key_header)
        if api_key:
            return api_key

        # Check Authorization header with ApiKey scheme
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("ApiKey "):
            return auth_header[7:]

        return None

    def _extract_bearer_token(self, request: Request) -> Optional[str]:
        """
        Extract JWT bearer token from Authorization header.

        Args:
            request: HTTP request

        Returns:
            JWT token string or None
        """
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            return auth_header[7:]
        return None

    async def _validate_api_key(self, api_key: str) -> Optional[dict]:
        """
        Validate API key and return user info.

        Args:
            api_key: API key to validate

        Returns:
            User dict or None if invalid
        """
        # Development mode: accept test keys
        if self.settings.env == "development" and api_key.startswith("dev_"):
            return {
                "id": "dev_user_001",
                "email": "developer@greenlang.io",
                "tenant_id": "dev_tenant",
                "roles": ["developer", "normalizer:read", "normalizer:write"],
                "api_key_id": api_key[:20],
            }

        # Production: validate against key store
        # TODO: Implement production API key validation
        # Example:
        # key_data = await self.key_store.get(api_key)
        # if key_data and not key_data.is_expired():
        #     return key_data.user

        return None

    async def _validate_jwt(self, token: str) -> Optional[dict]:
        """
        Validate JWT token and return user claims.

        Args:
            token: JWT token string

        Returns:
            User dict or None if invalid
        """
        try:
            payload = jwt.decode(
                token,
                self.settings.secret_key,
                algorithms=[self.settings.algorithm],
            )

            user_id = payload.get("sub")
            if not user_id:
                return None

            return {
                "id": user_id,
                "email": payload.get("email", ""),
                "tenant_id": payload.get("tenant_id", ""),
                "roles": payload.get("roles", []),
            }

        except JWTError as e:
            logger.debug("jwt_validation_failed", error=str(e))
            return None


class APIKeyManager:
    """
    API key management utilities.

    Provides methods for generating, validating, and revoking API keys.
    In production, backed by Redis or database storage.

    Example:
        >>> manager = APIKeyManager(settings)
        >>> key = await manager.create_key(user_id="user_123", scopes=["read"])
        >>> is_valid = await manager.validate_key(key)
    """

    def __init__(self, settings: Settings):
        """
        Initialize API key manager.

        Args:
            settings: Application settings
        """
        self.settings = settings
        # In production: connect to Redis/database
        self._keys: dict[str, dict] = {}

    async def create_key(
        self,
        user_id: str,
        tenant_id: str,
        scopes: list[str],
        name: Optional[str] = None,
        expires_in_days: int = 365,
    ) -> str:
        """
        Create a new API key.

        Args:
            user_id: User ID to associate with key
            tenant_id: Tenant ID
            scopes: Authorized scopes
            name: Optional key name/description
            expires_in_days: Key expiration in days

        Returns:
            Generated API key string
        """
        import hashlib
        import secrets

        # Generate key
        key = f"glnorm_{secrets.token_urlsafe(32)}"
        key_hash = hashlib.sha256(key.encode()).hexdigest()

        # Store key metadata
        self._keys[key_hash] = {
            "user_id": user_id,
            "tenant_id": tenant_id,
            "scopes": scopes,
            "name": name,
            "created_at": time.time(),
            "expires_at": time.time() + (expires_in_days * 86400),
        }

        logger.info(
            "api_key_created",
            user_id=user_id,
            tenant_id=tenant_id,
            key_name=name,
            expires_in_days=expires_in_days,
        )

        return key

    async def validate_key(self, key: str) -> Optional[dict]:
        """
        Validate an API key.

        Args:
            key: API key to validate

        Returns:
            Key metadata or None if invalid
        """
        import hashlib

        key_hash = hashlib.sha256(key.encode()).hexdigest()
        key_data = self._keys.get(key_hash)

        if not key_data:
            return None

        # Check expiration
        if key_data["expires_at"] < time.time():
            logger.info("api_key_expired", key_hash=key_hash[:16])
            return None

        return key_data

    async def revoke_key(self, key: str) -> bool:
        """
        Revoke an API key.

        Args:
            key: API key to revoke

        Returns:
            True if key was revoked, False if not found
        """
        import hashlib

        key_hash = hashlib.sha256(key.encode()).hexdigest()

        if key_hash in self._keys:
            del self._keys[key_hash]
            logger.info("api_key_revoked", key_hash=key_hash[:16])
            return True

        return False


class JWTManager:
    """
    JWT token management utilities.

    Provides methods for creating and validating JWT tokens.

    Example:
        >>> manager = JWTManager(settings)
        >>> token = manager.create_token(user_id="user_123", roles=["admin"])
        >>> claims = manager.decode_token(token)
    """

    def __init__(self, settings: Settings):
        """
        Initialize JWT manager.

        Args:
            settings: Application settings
        """
        self.settings = settings

    def create_token(
        self,
        user_id: str,
        email: str,
        tenant_id: str,
        roles: list[str],
        expires_minutes: Optional[int] = None,
    ) -> str:
        """
        Create a JWT access token.

        Args:
            user_id: User identifier
            email: User email
            tenant_id: Tenant identifier
            roles: User roles
            expires_minutes: Token expiration (uses default if not specified)

        Returns:
            Encoded JWT token
        """
        from datetime import datetime, timedelta

        expires = expires_minutes or self.settings.access_token_expire_minutes
        expire_time = datetime.utcnow() + timedelta(minutes=expires)

        payload = {
            "sub": user_id,
            "email": email,
            "tenant_id": tenant_id,
            "roles": roles,
            "exp": expire_time,
            "iat": datetime.utcnow(),
            "iss": "gl-normalizer-service",
        }

        return jwt.encode(
            payload,
            self.settings.secret_key,
            algorithm=self.settings.algorithm,
        )

    def decode_token(self, token: str) -> Optional[dict]:
        """
        Decode and validate a JWT token.

        Args:
            token: JWT token string

        Returns:
            Token claims or None if invalid
        """
        try:
            return jwt.decode(
                token,
                self.settings.secret_key,
                algorithms=[self.settings.algorithm],
            )
        except JWTError:
            return None

    def refresh_token(self, token: str) -> Optional[str]:
        """
        Refresh an existing token.

        Args:
            token: Current JWT token

        Returns:
            New token or None if current token is invalid
        """
        claims = self.decode_token(token)
        if not claims:
            return None

        return self.create_token(
            user_id=claims["sub"],
            email=claims.get("email", ""),
            tenant_id=claims.get("tenant_id", ""),
            roles=claims.get("roles", []),
        )
