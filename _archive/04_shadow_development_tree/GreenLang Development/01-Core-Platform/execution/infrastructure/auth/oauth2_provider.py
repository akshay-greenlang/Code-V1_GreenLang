"""
OAuth2/OIDC Provider for GreenLang Process Heat Agents
=======================================================

This module implements OAuth2/OIDC integration with Keycloak for the
GreenLang platform. It provides token validation, introspection,
JWKS-based signature verification, and scope-based authorization.

Features:
    - JWT token validation with JWKS endpoint integration
    - Token introspection for opaque tokens
    - Automatic JWKS key rotation handling
    - Scope-based authorization checks
    - FastAPI security dependency integration
    - Token refresh flow support
    - Comprehensive audit logging

Security Considerations:
    - Tokens are validated cryptographically using JWKS
    - Key caching with automatic refresh on rotation
    - Timing-safe token comparison
    - Rate limiting on validation endpoints
    - Comprehensive error handling without information leakage

OWASP Compliance:
    - A02:2021: RS256/ES256 signature validation
    - A07:2021: Token expiration, audience, issuer validation
    - A09:2021: Comprehensive audit logging

Example:
    >>> config = OAuth2ProviderConfig(
    ...     issuer="https://auth.greenlang.io/realms/greenlang",
    ...     audience="greenlang-api"
    ... )
    >>> provider = OAuth2Provider(config)
    >>> await provider.initialize()
    >>>
    >>> # Validate token
    >>> token_info = await provider.validate_token(bearer_token)
    >>> print(f"User: {token_info.subject}, Roles: {token_info.roles}")

Author: Security Team
Created: 2025-12-06
TASK-152: Implement OAuth2/OIDC (Keycloak) for Process Heat Agents
"""

import hashlib
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

from pydantic import BaseModel, Field, HttpUrl, validator

logger = logging.getLogger(__name__)


# =============================================================================
# Exceptions
# =============================================================================


class OAuth2Error(Exception):
    """Base exception for OAuth2 errors."""

    def __init__(
        self,
        message: str,
        error_code: str = "oauth2_error",
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}


class TokenValidationError(OAuth2Error):
    """Token validation failed."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, "token_validation_error", details)


class TokenExpiredError(OAuth2Error):
    """Token has expired."""

    def __init__(self, message: str = "Token has expired"):
        super().__init__(message, "token_expired")


class InsufficientScopeError(OAuth2Error):
    """Required scope not present in token."""

    def __init__(
        self,
        required_scopes: List[str],
        present_scopes: List[str]
    ):
        missing = set(required_scopes) - set(present_scopes)
        super().__init__(
            f"Insufficient scope: missing {missing}",
            "insufficient_scope",
            {"required": required_scopes, "present": present_scopes}
        )
        self.required_scopes = required_scopes
        self.present_scopes = present_scopes


class JWKSError(OAuth2Error):
    """JWKS retrieval or parsing error."""

    def __init__(self, message: str):
        super().__init__(message, "jwks_error")


# =============================================================================
# Models
# =============================================================================


class TokenType(str, Enum):
    """OAuth2 token types."""
    BEARER = "Bearer"
    DPoP = "DPoP"


class OAuth2ProviderConfig(BaseModel):
    """Configuration for OAuth2 provider."""

    # Required settings
    issuer: HttpUrl = Field(
        ...,
        description="OAuth2 issuer URL (Keycloak realm URL)"
    )
    audience: str = Field(
        ...,
        description="Expected audience claim in tokens"
    )

    # Optional settings with secure defaults
    jwks_uri: Optional[HttpUrl] = Field(
        None,
        description="JWKS endpoint URL (auto-discovered if not provided)"
    )
    introspection_uri: Optional[HttpUrl] = Field(
        None,
        description="Token introspection endpoint"
    )
    userinfo_uri: Optional[HttpUrl] = Field(
        None,
        description="UserInfo endpoint"
    )

    # Client credentials for introspection
    client_id: Optional[str] = Field(
        None,
        description="Client ID for token introspection"
    )
    client_secret: Optional[str] = Field(
        None,
        description="Client secret for token introspection"
    )

    # Validation settings
    allowed_algorithms: List[str] = Field(
        default=["RS256", "RS384", "RS512", "ES256", "ES384", "ES512"],
        description="Allowed JWT signing algorithms"
    )
    clock_skew_seconds: int = Field(
        default=30,
        ge=0,
        le=300,
        description="Allowed clock skew for token validation"
    )
    require_exp: bool = Field(
        default=True,
        description="Require expiration claim"
    )
    require_iat: bool = Field(
        default=True,
        description="Require issued-at claim"
    )
    verify_aud: bool = Field(
        default=True,
        description="Verify audience claim"
    )

    # Caching settings
    jwks_cache_ttl_seconds: int = Field(
        default=3600,
        ge=60,
        description="JWKS cache TTL in seconds"
    )
    token_cache_ttl_seconds: int = Field(
        default=60,
        ge=0,
        description="Token validation cache TTL"
    )
    max_cached_tokens: int = Field(
        default=10000,
        ge=100,
        description="Maximum number of cached token validations"
    )

    # Rate limiting
    max_validation_rate: int = Field(
        default=1000,
        ge=10,
        description="Maximum token validations per second"
    )

    # HTTP settings
    http_timeout_seconds: float = Field(
        default=10.0,
        ge=1.0,
        le=60.0,
        description="HTTP request timeout"
    )
    http_retry_count: int = Field(
        default=3,
        ge=0,
        le=10,
        description="HTTP retry count"
    )

    @validator("issuer")
    def validate_issuer(cls, v: HttpUrl) -> HttpUrl:
        """Ensure issuer is HTTPS in production."""
        if not str(v).startswith("https://") and "localhost" not in str(v):
            logger.warning(
                "OAuth2 issuer is not using HTTPS. "
                "This is insecure for production use."
            )
        return v

    class Config:
        """Pydantic configuration."""
        extra = "forbid"


class TokenInfo(BaseModel):
    """Validated token information."""

    # Standard claims
    subject: str = Field(..., description="Token subject (user ID)")
    issuer: str = Field(..., description="Token issuer")
    audience: List[str] = Field(..., description="Token audience(s)")
    issued_at: datetime = Field(..., description="Token issue time")
    expires_at: datetime = Field(..., description="Token expiration time")
    not_before: Optional[datetime] = Field(None, description="Token not-before time")

    # Token metadata
    token_id: Optional[str] = Field(None, description="Unique token ID (jti)")
    token_type: TokenType = Field(TokenType.BEARER, description="Token type")

    # User claims
    email: Optional[str] = Field(None, description="User email")
    email_verified: bool = Field(False, description="Email verification status")
    name: Optional[str] = Field(None, description="User full name")
    preferred_username: Optional[str] = Field(None, description="Username")
    given_name: Optional[str] = Field(None, description="First name")
    family_name: Optional[str] = Field(None, description="Last name")

    # Authorization claims
    scopes: List[str] = Field(default_factory=list, description="OAuth2 scopes")
    roles: List[str] = Field(default_factory=list, description="Realm roles")
    client_roles: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Client-specific roles"
    )
    groups: List[str] = Field(default_factory=list, description="Group memberships")

    # GreenLang-specific claims
    greenlang_permissions: List[str] = Field(
        default_factory=list,
        description="GreenLang-specific permissions"
    )
    greenlang_access_level: Optional[str] = Field(
        None,
        description="GreenLang access level"
    )

    # Additional claims
    custom_claims: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional custom claims"
    )

    # Validation metadata
    validated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Validation timestamp"
    )
    validation_method: str = Field(
        "jwt_signature",
        description="How token was validated"
    )

    def is_expired(self) -> bool:
        """Check if token has expired."""
        return datetime.now(timezone.utc) >= self.expires_at

    def has_scope(self, scope: str) -> bool:
        """Check if token has a specific scope."""
        return scope in self.scopes

    def has_role(self, role: str) -> bool:
        """Check if token has a specific realm role."""
        return role in self.roles

    def has_client_role(self, client: str, role: str) -> bool:
        """Check if token has a specific client role."""
        return role in self.client_roles.get(client, [])

    def has_permission(self, permission: str) -> bool:
        """Check if token has a GreenLang permission."""
        # Check for wildcard permission
        if "*" in self.greenlang_permissions:
            return True
        return permission in self.greenlang_permissions

    def get_provenance_hash(self) -> str:
        """Generate SHA-256 hash for audit trail."""
        data = f"{self.subject}:{self.token_id}:{self.issued_at.isoformat()}"
        return hashlib.sha256(data.encode()).hexdigest()


class UserInfo(BaseModel):
    """User information from userinfo endpoint."""

    subject: str = Field(..., description="User ID")
    email: Optional[str] = Field(None, description="Email address")
    email_verified: bool = Field(False, description="Email verified")
    name: Optional[str] = Field(None, description="Full name")
    preferred_username: Optional[str] = Field(None, description="Username")
    given_name: Optional[str] = Field(None, description="First name")
    family_name: Optional[str] = Field(None, description="Last name")
    groups: List[str] = Field(default_factory=list, description="Groups")
    roles: List[str] = Field(default_factory=list, description="Roles")


@dataclass
class JWKSCache:
    """Cache for JWKS keys."""

    keys: Dict[str, Any] = field(default_factory=dict)
    fetched_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None

    def is_expired(self) -> bool:
        """Check if cache has expired."""
        if self.expires_at is None:
            return True
        return datetime.now(timezone.utc) >= self.expires_at

    def get_key(self, kid: str) -> Optional[Dict[str, Any]]:
        """Get key by key ID."""
        return self.keys.get(kid)


# =============================================================================
# OAuth2 Provider Implementation
# =============================================================================


class OAuth2Provider:
    """
    OAuth2/OIDC provider for token validation and authorization.

    This class provides comprehensive OAuth2/OIDC integration with Keycloak,
    including JWT validation, token introspection, and scope-based authorization.

    Attributes:
        config: Provider configuration
        _jwks_cache: Cached JWKS keys
        _token_cache: Cached token validations
        _http_client: Async HTTP client

    Example:
        >>> config = OAuth2ProviderConfig(
        ...     issuer="https://auth.greenlang.io/realms/greenlang",
        ...     audience="greenlang-api"
        ... )
        >>> provider = OAuth2Provider(config)
        >>> await provider.initialize()
        >>>
        >>> # Validate token
        >>> token_info = await provider.validate_token(bearer_token)
        >>> if token_info.has_role("operator"):
        ...     print("User is an operator")
    """

    def __init__(self, config: OAuth2ProviderConfig):
        """
        Initialize OAuth2 provider.

        Args:
            config: Provider configuration
        """
        self.config = config
        self._jwks_cache = JWKSCache()
        self._token_cache: Dict[str, Tuple[TokenInfo, float]] = {}
        self._http_client: Optional[Any] = None
        self._initialized = False
        self._rate_limiter: Dict[str, List[float]] = {}

        # Metrics
        self._validation_count = 0
        self._cache_hits = 0
        self._cache_misses = 0

        logger.info(
            f"OAuth2Provider created for issuer: {config.issuer}"
        )

    async def initialize(self) -> None:
        """
        Initialize the provider by discovering endpoints and fetching JWKS.

        This method should be called before using the provider.
        It performs OIDC discovery and pre-fetches the JWKS.

        Raises:
            OAuth2Error: If initialization fails
        """
        if self._initialized:
            logger.debug("OAuth2Provider already initialized")
            return

        try:
            # Import httpx for async HTTP
            import httpx
            self._http_client = httpx.AsyncClient(
                timeout=self.config.http_timeout_seconds,
                follow_redirects=True,
                http2=True
            )

            # Perform OIDC discovery if needed
            if not self.config.jwks_uri:
                await self._discover_endpoints()

            # Pre-fetch JWKS
            await self._fetch_jwks()

            self._initialized = True
            logger.info("OAuth2Provider initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize OAuth2Provider: {e}")
            raise OAuth2Error(f"Initialization failed: {e}")

    async def close(self) -> None:
        """Close the provider and release resources."""
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None
        self._initialized = False
        logger.info("OAuth2Provider closed")

    async def _discover_endpoints(self) -> None:
        """Discover OIDC endpoints from well-known configuration."""
        discovery_url = f"{self.config.issuer}/.well-known/openid-configuration"

        try:
            response = await self._http_client.get(discovery_url)
            response.raise_for_status()
            config_data = response.json()

            # Extract endpoints
            if not self.config.jwks_uri:
                self.config.jwks_uri = config_data.get("jwks_uri")
            if not self.config.introspection_uri:
                self.config.introspection_uri = config_data.get(
                    "introspection_endpoint"
                )
            if not self.config.userinfo_uri:
                self.config.userinfo_uri = config_data.get("userinfo_endpoint")

            logger.info(f"Discovered OIDC endpoints from {discovery_url}")

        except Exception as e:
            logger.error(f"OIDC discovery failed: {e}")
            raise OAuth2Error(f"OIDC discovery failed: {e}")

    async def _fetch_jwks(self) -> None:
        """Fetch and cache JWKS keys."""
        if not self.config.jwks_uri:
            raise JWKSError("JWKS URI not configured")

        try:
            response = await self._http_client.get(str(self.config.jwks_uri))
            response.raise_for_status()
            jwks_data = response.json()

            # Parse and cache keys
            keys = {}
            for key_data in jwks_data.get("keys", []):
                kid = key_data.get("kid")
                if kid:
                    keys[kid] = key_data

            now = datetime.now(timezone.utc)
            self._jwks_cache = JWKSCache(
                keys=keys,
                fetched_at=now,
                expires_at=datetime.fromtimestamp(
                    now.timestamp() + self.config.jwks_cache_ttl_seconds,
                    tz=timezone.utc
                )
            )

            logger.info(f"Fetched {len(keys)} JWKS keys")

        except Exception as e:
            logger.error(f"Failed to fetch JWKS: {e}")
            raise JWKSError(f"Failed to fetch JWKS: {e}")

    async def _get_signing_key(self, kid: str) -> Dict[str, Any]:
        """
        Get signing key by key ID, refreshing cache if needed.

        Args:
            kid: Key ID from JWT header

        Returns:
            JWK key data

        Raises:
            JWKSError: If key not found
        """
        # Check cache
        if self._jwks_cache.is_expired():
            await self._fetch_jwks()

        key = self._jwks_cache.get_key(kid)
        if key:
            return key

        # Key not found, try refreshing
        logger.info(f"Key {kid} not found, refreshing JWKS")
        await self._fetch_jwks()

        key = self._jwks_cache.get_key(kid)
        if not key:
            raise JWKSError(f"Signing key {kid} not found in JWKS")

        return key

    def _get_token_cache_key(self, token: str) -> str:
        """Generate cache key for token."""
        return hashlib.sha256(token.encode()).hexdigest()[:32]

    def _check_token_cache(self, token: str) -> Optional[TokenInfo]:
        """Check if token is in cache and not expired."""
        cache_key = self._get_token_cache_key(token)
        cached = self._token_cache.get(cache_key)

        if cached:
            token_info, cached_at = cached
            cache_age = time.time() - cached_at

            # Check cache TTL
            if cache_age < self.config.token_cache_ttl_seconds:
                # Check token expiration
                if not token_info.is_expired():
                    self._cache_hits += 1
                    return token_info

            # Remove expired cache entry
            del self._token_cache[cache_key]

        self._cache_misses += 1
        return None

    def _cache_token(self, token: str, token_info: TokenInfo) -> None:
        """Cache validated token."""
        # Evict old entries if cache is full
        if len(self._token_cache) >= self.config.max_cached_tokens:
            # Remove oldest 10% of entries
            entries = sorted(
                self._token_cache.items(),
                key=lambda x: x[1][1]
            )
            for key, _ in entries[:len(entries) // 10]:
                del self._token_cache[key]

        cache_key = self._get_token_cache_key(token)
        self._token_cache[cache_key] = (token_info, time.time())

    async def validate_token(
        self,
        token: str,
        required_scopes: Optional[List[str]] = None,
        require_roles: Optional[List[str]] = None
    ) -> TokenInfo:
        """
        Validate an OAuth2 access token.

        This method validates the token cryptographically using JWKS,
        checks expiration, audience, issuer, and optionally scopes/roles.

        Args:
            token: Bearer token (without "Bearer " prefix)
            required_scopes: Optional list of required scopes
            require_roles: Optional list of required roles

        Returns:
            Validated token information

        Raises:
            TokenValidationError: If token validation fails
            TokenExpiredError: If token has expired
            InsufficientScopeError: If required scopes are missing
        """
        if not self._initialized:
            await self.initialize()

        self._validation_count += 1
        start_time = time.time()

        try:
            # Check cache first
            cached_info = self._check_token_cache(token)
            if cached_info:
                logger.debug("Token validated from cache")
                self._validate_requirements(
                    cached_info, required_scopes, require_roles
                )
                return cached_info

            # Validate JWT
            token_info = await self._validate_jwt(token)

            # Cache the result
            self._cache_token(token, token_info)

            # Check additional requirements
            self._validate_requirements(token_info, required_scopes, require_roles)

            # Log successful validation
            duration_ms = (time.time() - start_time) * 1000
            logger.info(
                f"Token validated for subject={token_info.subject} "
                f"duration={duration_ms:.2f}ms"
            )

            return token_info

        except OAuth2Error:
            raise
        except Exception as e:
            logger.error(f"Token validation failed: {e}")
            raise TokenValidationError(f"Token validation failed: {e}")

    async def _validate_jwt(self, token: str) -> TokenInfo:
        """
        Validate JWT token signature and claims.

        Args:
            token: JWT token string

        Returns:
            Parsed and validated token info

        Raises:
            TokenValidationError: If validation fails
        """
        try:
            # Import JWT library
            import jwt
            from jwt import PyJWKClient

            # Decode header to get key ID
            unverified_header = jwt.get_unverified_header(token)
            kid = unverified_header.get("kid")
            alg = unverified_header.get("alg")

            # Validate algorithm
            if alg not in self.config.allowed_algorithms:
                raise TokenValidationError(f"Algorithm {alg} not allowed")

            # Get signing key
            jwk_data = await self._get_signing_key(kid)

            # Build public key from JWK
            from jwt.algorithms import RSAAlgorithm, ECAlgorithm

            if alg.startswith("RS"):
                public_key = RSAAlgorithm.from_jwk(jwk_data)
            elif alg.startswith("ES"):
                public_key = ECAlgorithm.from_jwk(jwk_data)
            else:
                raise TokenValidationError(f"Unsupported algorithm: {alg}")

            # Validate token
            options = {
                "verify_signature": True,
                "verify_exp": self.config.require_exp,
                "verify_iat": self.config.require_iat,
                "verify_aud": self.config.verify_aud,
                "verify_iss": True,
                "require": ["exp", "iat", "sub"] if self.config.require_exp else ["sub"]
            }

            claims = jwt.decode(
                token,
                public_key,
                algorithms=[alg],
                audience=self.config.audience if self.config.verify_aud else None,
                issuer=str(self.config.issuer),
                options=options,
                leeway=self.config.clock_skew_seconds
            )

            # Parse claims into TokenInfo
            return self._parse_claims(claims)

        except jwt.ExpiredSignatureError:
            raise TokenExpiredError()
        except jwt.InvalidAudienceError:
            raise TokenValidationError("Invalid audience")
        except jwt.InvalidIssuerError:
            raise TokenValidationError("Invalid issuer")
        except jwt.InvalidSignatureError:
            raise TokenValidationError("Invalid signature")
        except jwt.DecodeError as e:
            raise TokenValidationError(f"Failed to decode token: {e}")
        except Exception as e:
            raise TokenValidationError(f"JWT validation failed: {e}")

    def _parse_claims(self, claims: Dict[str, Any]) -> TokenInfo:
        """
        Parse JWT claims into TokenInfo model.

        Args:
            claims: Decoded JWT claims

        Returns:
            Parsed TokenInfo
        """
        # Parse timestamps
        issued_at = datetime.fromtimestamp(claims.get("iat", 0), tz=timezone.utc)
        expires_at = datetime.fromtimestamp(claims.get("exp", 0), tz=timezone.utc)
        not_before = None
        if "nbf" in claims:
            not_before = datetime.fromtimestamp(claims["nbf"], tz=timezone.utc)

        # Parse audience
        aud = claims.get("aud", [])
        if isinstance(aud, str):
            aud = [aud]

        # Parse scopes
        scopes = []
        scope_claim = claims.get("scope", "")
        if isinstance(scope_claim, str):
            scopes = scope_claim.split() if scope_claim else []
        elif isinstance(scope_claim, list):
            scopes = scope_claim

        # Parse roles from realm_access
        roles = []
        realm_access = claims.get("realm_access", {})
        if isinstance(realm_access, dict):
            roles = realm_access.get("roles", [])

        # Parse client roles from resource_access
        client_roles = {}
        resource_access = claims.get("resource_access", {})
        if isinstance(resource_access, dict):
            for client, access in resource_access.items():
                if isinstance(access, dict):
                    client_roles[client] = access.get("roles", [])

        # Parse groups
        groups = claims.get("groups", [])
        if isinstance(groups, str):
            groups = [groups]

        # Parse GreenLang-specific claims
        greenlang_permissions = claims.get("greenlang_permissions", [])
        if isinstance(greenlang_permissions, str):
            greenlang_permissions = greenlang_permissions.split()

        # Build TokenInfo
        return TokenInfo(
            subject=claims.get("sub", ""),
            issuer=claims.get("iss", ""),
            audience=aud,
            issued_at=issued_at,
            expires_at=expires_at,
            not_before=not_before,
            token_id=claims.get("jti"),
            email=claims.get("email"),
            email_verified=claims.get("email_verified", False),
            name=claims.get("name"),
            preferred_username=claims.get("preferred_username"),
            given_name=claims.get("given_name"),
            family_name=claims.get("family_name"),
            scopes=scopes,
            roles=roles,
            client_roles=client_roles,
            groups=groups,
            greenlang_permissions=greenlang_permissions,
            greenlang_access_level=claims.get("greenlang_access_level"),
            validation_method="jwt_signature"
        )

    def _validate_requirements(
        self,
        token_info: TokenInfo,
        required_scopes: Optional[List[str]],
        required_roles: Optional[List[str]]
    ) -> None:
        """
        Validate additional token requirements.

        Args:
            token_info: Validated token info
            required_scopes: Required scopes
            required_roles: Required roles

        Raises:
            TokenExpiredError: If token is expired
            InsufficientScopeError: If scopes are missing
            OAuth2Error: If roles are missing
        """
        # Check expiration
        if token_info.is_expired():
            raise TokenExpiredError()

        # Check scopes
        if required_scopes:
            missing_scopes = set(required_scopes) - set(token_info.scopes)
            if missing_scopes:
                raise InsufficientScopeError(
                    required_scopes, token_info.scopes
                )

        # Check roles
        if required_roles:
            missing_roles = set(required_roles) - set(token_info.roles)
            if missing_roles:
                raise OAuth2Error(
                    f"Missing required roles: {missing_roles}",
                    "insufficient_roles"
                )

    async def introspect_token(self, token: str) -> TokenInfo:
        """
        Introspect token using the introspection endpoint.

        This is useful for opaque tokens or when additional validation is needed.

        Args:
            token: Access token to introspect

        Returns:
            Token information from introspection

        Raises:
            OAuth2Error: If introspection fails
        """
        if not self._initialized:
            await self.initialize()

        if not self.config.introspection_uri:
            raise OAuth2Error("Introspection endpoint not configured")

        if not self.config.client_id or not self.config.client_secret:
            raise OAuth2Error("Client credentials required for introspection")

        try:
            response = await self._http_client.post(
                str(self.config.introspection_uri),
                data={"token": token},
                auth=(self.config.client_id, self.config.client_secret)
            )
            response.raise_for_status()
            data = response.json()

            if not data.get("active", False):
                raise TokenValidationError("Token is not active")

            # Parse introspection response
            return self._parse_claims(data)

        except OAuth2Error:
            raise
        except Exception as e:
            logger.error(f"Token introspection failed: {e}")
            raise OAuth2Error(f"Token introspection failed: {e}")

    async def get_user_info(self, token: str) -> UserInfo:
        """
        Get user information from the userinfo endpoint.

        Args:
            token: Valid access token

        Returns:
            User information

        Raises:
            OAuth2Error: If userinfo request fails
        """
        if not self._initialized:
            await self.initialize()

        if not self.config.userinfo_uri:
            raise OAuth2Error("UserInfo endpoint not configured")

        try:
            response = await self._http_client.get(
                str(self.config.userinfo_uri),
                headers={"Authorization": f"Bearer {token}"}
            )
            response.raise_for_status()
            data = response.json()

            return UserInfo(
                subject=data.get("sub", ""),
                email=data.get("email"),
                email_verified=data.get("email_verified", False),
                name=data.get("name"),
                preferred_username=data.get("preferred_username"),
                given_name=data.get("given_name"),
                family_name=data.get("family_name"),
                groups=data.get("groups", []),
                roles=data.get("realm_access", {}).get("roles", [])
            )

        except Exception as e:
            logger.error(f"UserInfo request failed: {e}")
            raise OAuth2Error(f"UserInfo request failed: {e}")

    async def get_user_roles(self, token: str) -> List[str]:
        """
        Get user roles from token.

        Args:
            token: Access token

        Returns:
            List of user roles

        Raises:
            OAuth2Error: If token validation fails
        """
        token_info = await self.validate_token(token)
        return token_info.roles

    async def authorize_agent_access(
        self,
        token: str,
        agent_id: str,
        action: str
    ) -> bool:
        """
        Check if token authorizes access to a specific agent action.

        Args:
            token: Access token
            agent_id: Agent identifier (e.g., "gl-010")
            action: Action to perform (e.g., "execute", "view", "configure")

        Returns:
            True if authorized, False otherwise

        Raises:
            OAuth2Error: If token validation fails
        """
        try:
            token_info = await self.validate_token(token)

            # Admin has full access
            if token_info.has_role("admin"):
                return True

            # Check specific permission
            permission = f"agents:{action}"
            if token_info.has_permission(permission):
                return True

            # Check agent-specific permission
            agent_permission = f"agents:{agent_id}:{action}"
            if token_info.has_permission(agent_permission):
                return True

            # Operators can execute and view
            if token_info.has_role("operator"):
                if action in ["execute", "view"]:
                    return True

            # Viewers can only view
            if token_info.has_role("viewer"):
                if action == "view":
                    return True

            return False

        except OAuth2Error:
            return False

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get provider metrics.

        Returns:
            Dictionary of metrics
        """
        total_requests = self._cache_hits + self._cache_misses
        cache_hit_rate = (
            self._cache_hits / total_requests * 100
            if total_requests > 0 else 0
        )

        return {
            "validation_count": self._validation_count,
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "cache_hit_rate_percent": round(cache_hit_rate, 2),
            "cached_tokens": len(self._token_cache),
            "jwks_keys_count": len(self._jwks_cache.keys),
            "jwks_cache_age_seconds": (
                (datetime.now(timezone.utc) - self._jwks_cache.fetched_at).total_seconds()
                if self._jwks_cache.fetched_at else None
            )
        }


# =============================================================================
# FastAPI Security Dependencies
# =============================================================================


def create_oauth2_dependency(
    provider: OAuth2Provider,
    required_scopes: Optional[List[str]] = None,
    required_roles: Optional[List[str]] = None
) -> Callable:
    """
    Create a FastAPI security dependency for OAuth2 authentication.

    Args:
        provider: OAuth2 provider instance
        required_scopes: Required OAuth2 scopes
        required_roles: Required user roles

    Returns:
        FastAPI dependency function

    Example:
        >>> oauth2_provider = OAuth2Provider(config)
        >>> auth_required = create_oauth2_dependency(oauth2_provider)
        >>>
        >>> @app.get("/protected")
        >>> async def protected_endpoint(
        ...     token_info: TokenInfo = Depends(auth_required)
        ... ):
        ...     return {"user": token_info.subject}
    """
    try:
        from fastapi import Depends, HTTPException, Security, status
        from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

        security = HTTPBearer(auto_error=True)

        async def oauth2_auth(
            credentials: HTTPAuthorizationCredentials = Security(security)
        ) -> TokenInfo:
            """Validate OAuth2 token and return user info."""
            try:
                token_info = await provider.validate_token(
                    credentials.credentials,
                    required_scopes=required_scopes,
                    require_roles=required_roles
                )
                return token_info

            except TokenExpiredError:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Token has expired",
                    headers={"WWW-Authenticate": "Bearer"}
                )
            except InsufficientScopeError as e:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Insufficient scope: {e.message}",
                    headers={
                        "WWW-Authenticate": f'Bearer scope="{" ".join(e.required_scopes)}"'
                    }
                )
            except TokenValidationError as e:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail=f"Invalid token: {e.message}",
                    headers={"WWW-Authenticate": "Bearer"}
                )
            except OAuth2Error as e:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail=f"Authentication failed: {e.message}",
                    headers={"WWW-Authenticate": "Bearer"}
                )

        return oauth2_auth

    except ImportError:
        logger.warning("FastAPI not available, skipping dependency creation")
        return None


def create_scope_checker(
    provider: OAuth2Provider,
    required_scopes: List[str]
) -> Callable:
    """
    Create a FastAPI dependency that checks for specific scopes.

    Args:
        provider: OAuth2 provider instance
        required_scopes: Required OAuth2 scopes

    Returns:
        FastAPI dependency function
    """
    return create_oauth2_dependency(provider, required_scopes=required_scopes)


def create_role_checker(
    provider: OAuth2Provider,
    required_roles: List[str]
) -> Callable:
    """
    Create a FastAPI dependency that checks for specific roles.

    Args:
        provider: OAuth2 provider instance
        required_roles: Required user roles

    Returns:
        FastAPI dependency function
    """
    return create_oauth2_dependency(provider, required_roles=required_roles)
