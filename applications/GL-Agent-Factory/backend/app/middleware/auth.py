"""
JWT Authentication Middleware

This module provides JWT-based authentication for all API endpoints.
It implements secure token validation, API key authentication,
and user context injection for the GreenLang Agent Factory.

Features:
- JWT token decoding and validation using python-jose
- Token expiration, issuer, and audience validation
- API key authentication as fallback
- User claims extraction (user_id, tenant_id, roles)
- Token refresh detection and handling
- Comprehensive error responses with proper HTTP status codes

Example:
    >>> from app.middleware.auth import JWTAuthMiddleware
    >>> app.add_middleware(
    ...     JWTAuthMiddleware,
    ...     secret_key=settings.jwt_secret,
    ...     algorithm=settings.jwt_algorithm,
    ... )
"""

import hashlib
import logging
import re
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Set

from fastapi import Request, Response
from jose import ExpiredSignatureError, JWTError, jwt
from jose.exceptions import JWTClaimsError
from pydantic import BaseModel, Field, validator
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

logger = logging.getLogger(__name__)


# =============================================================================
# Pydantic Models for Token Claims
# =============================================================================


class TokenClaims(BaseModel):
    """
    JWT Token Claims Model.

    Represents the standard and custom claims extracted from a JWT token.
    Follows RFC 7519 for standard claims and adds GreenLang-specific claims.

    Attributes:
        sub: Subject (user identifier)
        tenant_id: Tenant identifier for multi-tenancy
        roles: List of user roles for RBAC
        permissions: List of specific permissions
        iat: Issued at timestamp
        exp: Expiration timestamp
        iss: Token issuer
        aud: Token audience
        jti: JWT ID for token tracking/revocation
        token_type: Type of token (access, refresh)
    """

    sub: str = Field(..., description="Subject (user ID)")
    tenant_id: str = Field(..., description="Tenant identifier")
    roles: List[str] = Field(default_factory=list, description="User roles")
    permissions: List[str] = Field(default_factory=list, description="User permissions")
    iat: Optional[int] = Field(None, description="Issued at timestamp")
    exp: Optional[int] = Field(None, description="Expiration timestamp")
    iss: Optional[str] = Field(None, description="Token issuer")
    aud: Optional[str] = Field(None, description="Token audience")
    jti: Optional[str] = Field(None, description="JWT ID")
    token_type: str = Field(default="access", description="Token type")

    @validator("roles", pre=True, always=True)
    def ensure_roles_list(cls, v: Any) -> List[str]:
        """Ensure roles is always a list."""
        if v is None:
            return []
        if isinstance(v, str):
            return [v]
        return list(v)

    @validator("permissions", pre=True, always=True)
    def ensure_permissions_list(cls, v: Any) -> List[str]:
        """Ensure permissions is always a list."""
        if v is None:
            return []
        if isinstance(v, str):
            return [v]
        return list(v)


class APIKeyInfo(BaseModel):
    """
    API Key Information Model.

    Represents validated API key metadata.

    Attributes:
        key_id: Unique identifier for the API key
        tenant_id: Associated tenant
        user_id: Associated user (service account)
        roles: Roles assigned to this API key
        permissions: Specific permissions for this key
        rate_limit: Custom rate limit for this key
        expires_at: Optional expiration datetime
    """

    key_id: str = Field(..., description="API key identifier")
    tenant_id: str = Field(..., description="Tenant identifier")
    user_id: str = Field(..., description="Service account user ID")
    roles: List[str] = Field(default_factory=list, description="Assigned roles")
    permissions: List[str] = Field(default_factory=list, description="Specific permissions")
    rate_limit: Optional[int] = Field(None, description="Custom rate limit")
    expires_at: Optional[datetime] = Field(None, description="Key expiration")


class AuthError(BaseModel):
    """
    Authentication Error Response Model.

    Standardized error response for authentication failures.
    """

    code: str = Field(..., description="Error code")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")


# =============================================================================
# API Key Store (In-Memory Cache with Database Fallback)
# =============================================================================


class APIKeyStore:
    """
    API Key Storage and Validation.

    Provides API key lookup with caching. In production, this would
    integrate with a database and Redis cache.

    This implementation uses an in-memory store for development
    and can be extended for production use.
    """

    def __init__(self) -> None:
        """Initialize the API key store."""
        # In-memory cache for API keys (development/testing)
        # Format: {key_hash: APIKeyInfo}
        self._cache: Dict[str, APIKeyInfo] = {}
        self._cache_ttl: int = 300  # 5 minutes
        self._cache_timestamps: Dict[str, float] = {}

    def _hash_key(self, api_key: str) -> str:
        """
        Hash API key for secure storage/lookup.

        Args:
            api_key: Raw API key string

        Returns:
            SHA-256 hash of the API key
        """
        return hashlib.sha256(api_key.encode()).hexdigest()

    async def validate_api_key(self, api_key: str) -> Optional[APIKeyInfo]:
        """
        Validate an API key and return associated information.

        Checks the in-memory cache first, then falls back to database.
        Invalid or expired keys return None.

        Args:
            api_key: The API key to validate

        Returns:
            APIKeyInfo if valid, None if invalid or not found
        """
        if not api_key:
            return None

        # Validate API key format (prefix_randomstring)
        if not self._validate_key_format(api_key):
            logger.warning("Invalid API key format")
            return None

        key_hash = self._hash_key(api_key)

        # Check cache first
        cached = self._get_from_cache(key_hash)
        if cached is not None:
            return cached

        # In production, query database here
        # For now, check if it's a development key
        key_info = await self._lookup_key_in_database(api_key, key_hash)

        if key_info:
            self._set_in_cache(key_hash, key_info)

        return key_info

    def _validate_key_format(self, api_key: str) -> bool:
        """
        Validate API key format.

        Expected format: gl_{environment}_{32_char_random}
        Example: gl_prod_a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6

        Args:
            api_key: API key to validate

        Returns:
            True if format is valid
        """
        # Pattern: gl_<env>_<32 alphanumeric chars>
        pattern = r"^gl_(prod|staging|dev|test)_[a-zA-Z0-9]{32}$"
        return bool(re.match(pattern, api_key))

    def _get_from_cache(self, key_hash: str) -> Optional[APIKeyInfo]:
        """Get API key info from cache if not expired."""
        if key_hash not in self._cache:
            return None

        timestamp = self._cache_timestamps.get(key_hash, 0)
        if time.time() - timestamp > self._cache_ttl:
            # Cache expired
            del self._cache[key_hash]
            del self._cache_timestamps[key_hash]
            return None

        return self._cache[key_hash]

    def _set_in_cache(self, key_hash: str, key_info: APIKeyInfo) -> None:
        """Store API key info in cache."""
        self._cache[key_hash] = key_info
        self._cache_timestamps[key_hash] = time.time()

    async def _lookup_key_in_database(
        self, api_key: str, key_hash: str
    ) -> Optional[APIKeyInfo]:
        """
        Look up API key in database.

        In production, this would query the database.
        For development, we allow specific test keys.

        Args:
            api_key: Raw API key
            key_hash: Hashed API key

        Returns:
            APIKeyInfo if found, None otherwise
        """
        # Development/testing: Allow specific test keys
        # In production, replace with actual database query
        test_keys = {
            "gl_dev_00000000000000000000000000000001": APIKeyInfo(
                key_id="dev-key-001",
                tenant_id="default",
                user_id="dev-service-account",
                roles=["developer"],
                permissions=["read", "write", "execute"],
            ),
            "gl_test_00000000000000000000000000000001": APIKeyInfo(
                key_id="test-key-001",
                tenant_id="test-tenant",
                user_id="test-service-account",
                roles=["tester"],
                permissions=["read", "execute"],
            ),
        }

        return test_keys.get(api_key)

    def register_api_key(self, api_key: str, key_info: APIKeyInfo) -> None:
        """
        Register a new API key (for testing/development).

        Args:
            api_key: The API key string
            key_info: Associated key information
        """
        key_hash = self._hash_key(api_key)
        self._set_in_cache(key_hash, key_info)


# =============================================================================
# Token Blacklist (for Revocation)
# =============================================================================


class TokenBlacklist:
    """
    Token Blacklist for Revocation.

    Tracks revoked tokens by their JTI (JWT ID) claim.
    In production, this should use Redis for distributed state.
    """

    def __init__(self) -> None:
        """Initialize the token blacklist."""
        # In-memory blacklist: {jti: expiration_timestamp}
        self._blacklist: Dict[str, int] = {}

    def is_blacklisted(self, jti: str) -> bool:
        """
        Check if a token is blacklisted.

        Args:
            jti: JWT ID to check

        Returns:
            True if token is revoked
        """
        if jti in self._blacklist:
            # Check if blacklist entry has expired
            if self._blacklist[jti] > int(time.time()):
                return True
            else:
                # Clean up expired entry
                del self._blacklist[jti]
        return False

    def blacklist_token(self, jti: str, exp: int) -> None:
        """
        Add a token to the blacklist.

        Args:
            jti: JWT ID to blacklist
            exp: Token expiration timestamp (blacklist until this time)
        """
        self._blacklist[jti] = exp

    def cleanup_expired(self) -> int:
        """
        Remove expired entries from blacklist.

        Returns:
            Number of entries removed
        """
        current_time = int(time.time())
        expired = [jti for jti, exp in self._blacklist.items() if exp <= current_time]
        for jti in expired:
            del self._blacklist[jti]
        return len(expired)


# =============================================================================
# JWT Authentication Middleware
# =============================================================================


class JWTAuthMiddleware(BaseHTTPMiddleware):
    """
    JWT Authentication Middleware.

    Validates JWT tokens from the Authorization header and injects
    user context into the request state.

    Features:
    - Bearer token extraction and validation
    - Token signature verification
    - Expiration, issuer, and audience validation
    - User claims extraction
    - API key fallback authentication
    - Token blacklist checking for revocation
    - Comprehensive error responses

    Public endpoints (like /health) are excluded from authentication.

    Attributes:
        secret_key: JWT signing secret
        algorithm: JWT algorithm (HS256, RS256, etc.)
        issuer: Expected token issuer
        audience: Expected token audience
        api_key_store: API key validation store
        token_blacklist: Token revocation blacklist

    Example:
        >>> app.add_middleware(
        ...     JWTAuthMiddleware,
        ...     secret_key="your-secret-key",
        ...     algorithm="HS256",
        ...     issuer="greenlang-auth",
        ...     audience="greenlang-api",
        ... )
    """

    # Paths that don't require authentication
    PUBLIC_PATHS: Set[str] = {
        "/health",
        "/ready",
        "/docs",
        "/redoc",
        "/openapi.json",
        "/favicon.ico",
    }

    # Path prefixes that don't require authentication
    PUBLIC_PATH_PREFIXES: Set[str] = {
        "/docs",
        "/redoc",
    }

    def __init__(
        self,
        app: Any,
        secret_key: str,
        algorithm: str = "HS256",
        issuer: Optional[str] = None,
        audience: Optional[str] = None,
        verify_exp: bool = True,
        leeway: int = 0,
    ) -> None:
        """
        Initialize the JWT authentication middleware.

        Args:
            app: The ASGI application
            secret_key: JWT signing secret (must be secure in production)
            algorithm: JWT algorithm (default: HS256)
            issuer: Expected token issuer (optional validation)
            audience: Expected token audience (optional validation)
            verify_exp: Whether to verify token expiration (default: True)
            leeway: Seconds of leeway for expiration checking (default: 0)
        """
        super().__init__(app)
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.issuer = issuer
        self.audience = audience
        self.verify_exp = verify_exp
        self.leeway = leeway

        # Initialize API key store and token blacklist
        self.api_key_store = APIKeyStore()
        self.token_blacklist = TokenBlacklist()

        # Validate configuration
        self._validate_config()

        logger.info(
            "JWTAuthMiddleware initialized",
            extra={
                "algorithm": algorithm,
                "issuer": issuer,
                "audience": audience,
                "verify_exp": verify_exp,
            },
        )

    def _validate_config(self) -> None:
        """Validate middleware configuration."""
        if not self.secret_key:
            raise ValueError("JWT secret_key is required")

        if self.secret_key == "change-me-in-production":
            logger.warning(
                "Using default JWT secret - CHANGE THIS IN PRODUCTION!",
                extra={"security_warning": True},
            )

        supported_algorithms = {"HS256", "HS384", "HS512", "RS256", "RS384", "RS512"}
        if self.algorithm not in supported_algorithms:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Response],
    ) -> Response:
        """
        Process the request through authentication.

        Workflow:
        1. Check if path is public (skip auth)
        2. Extract token from Authorization header
        3. If no token, try API key authentication
        4. Validate token (signature, expiration, claims)
        5. Check token blacklist for revocation
        6. Inject user context into request.state
        7. Pass to next middleware/handler

        Args:
            request: The incoming request
            call_next: Next middleware/handler in chain

        Returns:
            Response from the handler or error response
        """
        # Skip auth for public paths
        if self._is_public_path(request.url.path):
            return await call_next(request)

        # Extract Authorization header
        auth_header = request.headers.get("Authorization")

        # Try Bearer token authentication
        if auth_header and auth_header.startswith("Bearer "):
            return await self._handle_bearer_auth(request, call_next, auth_header)

        # Try API key authentication as fallback
        api_key = request.headers.get("X-API-Key")
        if api_key:
            return await self._handle_api_key_auth(request, call_next, api_key)

        # No authentication provided
        logger.warning(
            "Authentication required but no credentials provided",
            extra={
                "path": request.url.path,
                "method": request.method,
                "client_ip": self._get_client_ip(request),
            },
        )

        return self._create_error_response(
            status_code=401,
            code="UNAUTHORIZED",
            message="Authentication required. Provide Bearer token or API key.",
            headers={"WWW-Authenticate": 'Bearer realm="greenlang-api"'},
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
        if path in self.PUBLIC_PATHS:
            return True

        # Prefix match
        for prefix in self.PUBLIC_PATH_PREFIXES:
            if path.startswith(prefix):
                return True

        return False

    async def _handle_bearer_auth(
        self,
        request: Request,
        call_next: Callable[[Request], Response],
        auth_header: str,
    ) -> Response:
        """
        Handle Bearer token authentication.

        Args:
            request: The incoming request
            call_next: Next handler
            auth_header: Authorization header value

        Returns:
            Response from handler or error response
        """
        # Extract token
        try:
            token = auth_header.split(" ", 1)[1]
        except IndexError:
            return self._create_error_response(
                status_code=401,
                code="INVALID_AUTH_HEADER",
                message="Invalid Authorization header format. Expected: Bearer <token>",
            )

        if not token:
            return self._create_error_response(
                status_code=401,
                code="MISSING_TOKEN",
                message="Bearer token is empty",
            )

        # Validate token
        claims_result = self._validate_token(token)
        if isinstance(claims_result, Response):
            return claims_result

        claims: TokenClaims = claims_result

        # Check token blacklist
        if claims.jti and self.token_blacklist.is_blacklisted(claims.jti):
            logger.warning(
                "Revoked token used",
                extra={
                    "jti": claims.jti,
                    "user_id": claims.sub,
                    "client_ip": self._get_client_ip(request),
                },
            )
            return self._create_error_response(
                status_code=401,
                code="TOKEN_REVOKED",
                message="Token has been revoked",
            )

        # Check if token is about to expire (for refresh warning)
        token_expires_soon = self._check_token_expiring_soon(claims)

        # Inject user context into request state
        request.state.user_id = claims.sub
        request.state.tenant_id = claims.tenant_id
        request.state.roles = claims.roles
        request.state.permissions = claims.permissions
        request.state.token_claims = claims
        request.state.auth_method = "jwt"

        logger.debug(
            "JWT authentication successful",
            extra={
                "user_id": claims.sub,
                "tenant_id": claims.tenant_id,
                "roles": claims.roles,
            },
        )

        # Process request
        response = await call_next(request)

        # Add token expiration warning header if expiring soon
        if token_expires_soon:
            response.headers["X-Token-Expiring-Soon"] = "true"

        return response

    def _validate_token(self, token: str) -> TokenClaims | Response:
        """
        Validate JWT token and extract claims.

        Performs:
        - Signature verification
        - Expiration validation
        - Issuer validation (if configured)
        - Audience validation (if configured)
        - Required claims validation

        Args:
            token: JWT token string

        Returns:
            TokenClaims if valid, error Response if invalid
        """
        try:
            # Build decode options
            options = {
                "verify_signature": True,
                "verify_exp": self.verify_exp,
                "verify_iat": True,
                "verify_nbf": True,
                "require_exp": True,
                "require_iat": False,
                "leeway": self.leeway,
            }

            # Build expected claims
            expected_claims: Dict[str, Any] = {}
            if self.issuer:
                expected_claims["iss"] = self.issuer
                options["verify_iss"] = True
            if self.audience:
                expected_claims["aud"] = self.audience
                options["verify_aud"] = True

            # Decode and verify token
            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=[self.algorithm],
                options=options,
                audience=self.audience,
                issuer=self.issuer,
            )

            # Validate required claims
            if "sub" not in payload:
                return self._create_error_response(
                    status_code=401,
                    code="MISSING_CLAIM",
                    message="Token missing required 'sub' claim",
                )

            # Extract tenant_id (required for multi-tenancy)
            tenant_id = payload.get("tenant_id")
            if not tenant_id:
                # Allow default tenant for backward compatibility
                tenant_id = "default"
                logger.debug("Token missing tenant_id, using default")

            # Build claims model
            claims = TokenClaims(
                sub=payload["sub"],
                tenant_id=tenant_id,
                roles=payload.get("roles", []),
                permissions=payload.get("permissions", []),
                iat=payload.get("iat"),
                exp=payload.get("exp"),
                iss=payload.get("iss"),
                aud=payload.get("aud"),
                jti=payload.get("jti"),
                token_type=payload.get("token_type", "access"),
            )

            return claims

        except ExpiredSignatureError:
            logger.info("Token expired")
            return self._create_error_response(
                status_code=401,
                code="TOKEN_EXPIRED",
                message="Token has expired. Please refresh your token.",
                headers={"X-Token-Expired": "true"},
            )

        except JWTClaimsError as e:
            logger.warning(f"JWT claims validation failed: {e}")
            return self._create_error_response(
                status_code=401,
                code="INVALID_CLAIMS",
                message=f"Token claims validation failed: {str(e)}",
            )

        except JWTError as e:
            logger.warning(f"JWT validation failed: {e}")
            return self._create_error_response(
                status_code=401,
                code="INVALID_TOKEN",
                message="Invalid token. Please authenticate again.",
            )

        except Exception as e:
            logger.error(f"Unexpected error during token validation: {e}", exc_info=True)
            return self._create_error_response(
                status_code=401,
                code="AUTH_ERROR",
                message="Authentication failed due to an internal error",
            )

    def _check_token_expiring_soon(
        self, claims: TokenClaims, threshold_seconds: int = 300
    ) -> bool:
        """
        Check if token is expiring soon (for refresh warning).

        Args:
            claims: Token claims
            threshold_seconds: Seconds before expiration to warn (default: 5 min)

        Returns:
            True if token expires within threshold
        """
        if claims.exp is None:
            return False

        current_time = int(time.time())
        time_until_expiry = claims.exp - current_time

        return 0 < time_until_expiry <= threshold_seconds

    async def _handle_api_key_auth(
        self,
        request: Request,
        call_next: Callable[[Request], Response],
        api_key: str,
    ) -> Response:
        """
        Handle API key authentication.

        Args:
            request: The incoming request
            call_next: Next handler
            api_key: API key from header

        Returns:
            Response from handler or error response
        """
        # Validate API key
        key_info = await self.api_key_store.validate_api_key(api_key)

        if key_info is None:
            logger.warning(
                "Invalid API key",
                extra={
                    "client_ip": self._get_client_ip(request),
                    "path": request.url.path,
                },
            )
            return self._create_error_response(
                status_code=401,
                code="INVALID_API_KEY",
                message="Invalid or expired API key",
            )

        # Check if API key is expired
        if key_info.expires_at:
            if datetime.now(timezone.utc) > key_info.expires_at:
                logger.warning(
                    "Expired API key used",
                    extra={"key_id": key_info.key_id},
                )
                return self._create_error_response(
                    status_code=401,
                    code="API_KEY_EXPIRED",
                    message="API key has expired",
                )

        # Inject user context into request state
        request.state.user_id = key_info.user_id
        request.state.tenant_id = key_info.tenant_id
        request.state.roles = key_info.roles
        request.state.permissions = key_info.permissions
        request.state.api_key_id = key_info.key_id
        request.state.auth_method = "api_key"

        # Set custom rate limit if configured
        if key_info.rate_limit:
            request.state.custom_rate_limit = key_info.rate_limit

        logger.debug(
            "API key authentication successful",
            extra={
                "key_id": key_info.key_id,
                "tenant_id": key_info.tenant_id,
                "user_id": key_info.user_id,
            },
        )

        return await call_next(request)

    def _create_error_response(
        self,
        status_code: int,
        code: str,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> JSONResponse:
        """
        Create a standardized error response.

        Args:
            status_code: HTTP status code
            code: Error code
            message: Human-readable error message
            details: Additional error details
            headers: Additional response headers

        Returns:
            JSONResponse with error payload
        """
        error = AuthError(code=code, message=message, details=details)

        response_headers = {"X-Error-Code": code}
        if headers:
            response_headers.update(headers)

        return JSONResponse(
            status_code=status_code,
            content={"error": error.model_dump(exclude_none=True)},
            headers=response_headers,
        )

    def _get_client_ip(self, request: Request) -> str:
        """
        Extract client IP from request.

        Handles X-Forwarded-For header for proxied requests.

        Args:
            request: The incoming request

        Returns:
            Client IP address
        """
        # Check X-Forwarded-For header (from reverse proxy)
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            # Take the first IP (original client)
            return forwarded_for.split(",")[0].strip()

        # Check X-Real-IP header
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip

        # Fallback to direct client
        if request.client:
            return request.client.host

        return "unknown"


# =============================================================================
# Utility Functions
# =============================================================================


def create_jwt_token(
    user_id: str,
    tenant_id: str,
    secret_key: str,
    algorithm: str = "HS256",
    expires_in_hours: int = 24,
    roles: Optional[List[str]] = None,
    permissions: Optional[List[str]] = None,
    issuer: Optional[str] = None,
    audience: Optional[str] = None,
    token_type: str = "access",
) -> str:
    """
    Create a JWT token.

    Utility function for creating tokens (used in auth service/tests).

    Args:
        user_id: Subject (user identifier)
        tenant_id: Tenant identifier
        secret_key: Signing secret
        algorithm: JWT algorithm
        expires_in_hours: Token validity period
        roles: User roles
        permissions: User permissions
        issuer: Token issuer
        audience: Token audience
        token_type: Type of token

    Returns:
        Encoded JWT token string

    Example:
        >>> token = create_jwt_token(
        ...     user_id="user-123",
        ...     tenant_id="tenant-abc",
        ...     secret_key="my-secret",
        ...     roles=["admin"],
        ... )
    """
    import uuid

    current_time = int(time.time())
    expiration_time = current_time + (expires_in_hours * 3600)

    payload = {
        "sub": user_id,
        "tenant_id": tenant_id,
        "roles": roles or [],
        "permissions": permissions or [],
        "iat": current_time,
        "exp": expiration_time,
        "jti": str(uuid.uuid4()),
        "token_type": token_type,
    }

    if issuer:
        payload["iss"] = issuer
    if audience:
        payload["aud"] = audience

    return jwt.encode(payload, secret_key, algorithm=algorithm)


def decode_jwt_token(
    token: str,
    secret_key: str,
    algorithm: str = "HS256",
    verify_exp: bool = True,
) -> Dict[str, Any]:
    """
    Decode a JWT token without the middleware context.

    Utility function for token inspection/debugging.

    Args:
        token: JWT token string
        secret_key: Signing secret
        algorithm: JWT algorithm
        verify_exp: Whether to verify expiration

    Returns:
        Token payload as dictionary

    Raises:
        JWTError: If token is invalid
    """
    options = {
        "verify_exp": verify_exp,
        "verify_signature": True,
    }

    return jwt.decode(token, secret_key, algorithms=[algorithm], options=options)
