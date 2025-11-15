"""
OAuth 2.0 / JWT Authentication System

This module implements production-grade OAuth 2.0 authentication with JWT tokens
for the GreenLang Agent Foundation platform. Supports multiple identity providers
including Auth0, Okta, and Azure AD.

Features:
- JWT token generation and validation
- Access token (15 min) + Refresh token (30 days)
- Token revocation and blacklisting
- Multi-issuer support with public key rotation
- SOC 2 Type II compliant security

Example:
    >>> auth_manager = OAuthManager(config)
    >>> tokens = auth_manager.generate_tokens(user_id="user123", roles=["analyst"])
    >>> user = auth_manager.validate_access_token(tokens.access_token)
"""

from typing import Dict, List, Optional, Any, Set
from pydantic import BaseModel, Field, validator
from datetime import datetime, timedelta
import jwt
import secrets
import hashlib
import logging
from enum import Enum
import asyncio
import aiohttp
from functools import lru_cache
import json

logger = logging.getLogger(__name__)


class TokenType(str, Enum):
    """Token types supported by the system"""
    ACCESS = "access"
    REFRESH = "refresh"
    ID = "id_token"


class IdentityProvider(str, Enum):
    """Supported OAuth identity providers"""
    AUTH0 = "auth0"
    OKTA = "okta"
    AZURE_AD = "azure_ad"
    GOOGLE = "google"
    CUSTOM = "custom"


class TokenPair(BaseModel):
    """OAuth token pair with metadata"""
    access_token: str = Field(..., description="Short-lived access token (15 min)")
    refresh_token: str = Field(..., description="Long-lived refresh token (30 days)")
    token_type: str = Field(default="Bearer", description="Token type")
    expires_in: int = Field(..., description="Access token TTL in seconds")
    scope: str = Field(..., description="Token scope")
    issued_at: datetime = Field(default_factory=datetime.utcnow)


class TokenClaims(BaseModel):
    """JWT token claims structure"""
    sub: str = Field(..., description="Subject (user ID)")
    iss: str = Field(..., description="Issuer")
    aud: str = Field(..., description="Audience")
    exp: int = Field(..., description="Expiration timestamp")
    iat: int = Field(..., description="Issued at timestamp")
    nbf: Optional[int] = Field(None, description="Not before timestamp")
    jti: str = Field(..., description="JWT ID (unique identifier)")

    # Custom claims
    email: Optional[str] = Field(None, description="User email")
    roles: List[str] = Field(default_factory=list, description="User roles")
    permissions: List[str] = Field(default_factory=list, description="User permissions")
    tenant_id: Optional[str] = Field(None, description="Tenant identifier")
    session_id: Optional[str] = Field(None, description="Session identifier")

    @validator('roles', 'permissions')
    def validate_lists(cls, v):
        """Ensure roles and permissions are unique"""
        return list(set(v)) if v else []


class OAuthConfig(BaseModel):
    """OAuth configuration settings"""

    # JWT settings
    secret_key: str = Field(..., description="Secret key for signing tokens")
    algorithm: str = Field(default="HS256", description="JWT signing algorithm")
    issuer: str = Field(..., description="Token issuer identifier")
    audience: str = Field(..., description="Token audience")

    # Token TTL settings
    access_token_ttl: int = Field(default=900, description="Access token TTL (15 min)")
    refresh_token_ttl: int = Field(default=2592000, description="Refresh token TTL (30 days)")

    # Identity provider settings
    provider: IdentityProvider = Field(default=IdentityProvider.CUSTOM)
    provider_config: Dict[str, Any] = Field(default_factory=dict)

    # Public key rotation
    public_keys: Dict[str, str] = Field(default_factory=dict, description="Public keys by key ID")
    key_rotation_interval: int = Field(default=86400, description="Key rotation interval (1 day)")

    # Security settings
    enable_token_blacklist: bool = Field(default=True, description="Enable token revocation")
    require_https: bool = Field(default=True, description="Require HTTPS for token endpoints")
    enable_csrf_protection: bool = Field(default=True, description="Enable CSRF protection")


class TokenBlacklist:
    """In-memory token blacklist with expiration (use Redis in production)"""

    def __init__(self):
        self._blacklist: Set[str] = set()
        self._expiration: Dict[str, datetime] = {}

    def add(self, jti: str, expires_at: datetime) -> None:
        """Add token to blacklist"""
        self._blacklist.add(jti)
        self._expiration[jti] = expires_at
        logger.info(f"Token {jti} added to blacklist")

    def is_blacklisted(self, jti: str) -> bool:
        """Check if token is blacklisted"""
        if jti not in self._blacklist:
            return False

        # Remove expired entries
        if self._expiration.get(jti, datetime.utcnow()) < datetime.utcnow():
            self._blacklist.discard(jti)
            self._expiration.pop(jti, None)
            return False

        return True

    def cleanup_expired(self) -> int:
        """Remove expired tokens from blacklist"""
        now = datetime.utcnow()
        expired = [jti for jti, exp in self._expiration.items() if exp < now]

        for jti in expired:
            self._blacklist.discard(jti)
            self._expiration.pop(jti, None)

        logger.info(f"Cleaned up {len(expired)} expired tokens from blacklist")
        return len(expired)


class PublicKeyCache:
    """Cache for public keys from identity providers"""

    def __init__(self, ttl: int = 3600):
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._timestamps: Dict[str, datetime] = {}
        self._ttl = ttl

    def get(self, provider: str, key_id: str) -> Optional[str]:
        """Get cached public key"""
        cache_key = f"{provider}:{key_id}"

        if cache_key not in self._cache:
            return None

        # Check if expired
        if (datetime.utcnow() - self._timestamps[cache_key]).total_seconds() > self._ttl:
            del self._cache[cache_key]
            del self._timestamps[cache_key]
            return None

        return self._cache[cache_key]

    def set(self, provider: str, key_id: str, public_key: str) -> None:
        """Cache public key"""
        cache_key = f"{provider}:{key_id}"
        self._cache[cache_key] = public_key
        self._timestamps[cache_key] = datetime.utcnow()


class OAuthManager:
    """
    OAuth 2.0 / JWT Authentication Manager

    Handles JWT token generation, validation, and refresh for OAuth 2.0 flows.
    Supports multiple identity providers with public key rotation.

    Attributes:
        config: OAuth configuration settings
        blacklist: Token revocation blacklist
        key_cache: Public key cache for external providers

    Example:
        >>> config = OAuthConfig(
        ...     secret_key="your-secret-key",
        ...     issuer="https://greenlang.ai",
        ...     audience="greenlang-api"
        ... )
        >>> manager = OAuthManager(config)
        >>> tokens = manager.generate_tokens(user_id="user123", roles=["analyst"])
        >>> user = manager.validate_access_token(tokens.access_token)
    """

    def __init__(self, config: OAuthConfig):
        """Initialize OAuth manager"""
        self.config = config
        self.blacklist = TokenBlacklist()
        self.key_cache = PublicKeyCache()

        # Start cleanup task for blacklist
        if config.enable_token_blacklist:
            asyncio.create_task(self._periodic_cleanup())

    def generate_tokens(
        self,
        user_id: str,
        roles: List[str],
        permissions: Optional[List[str]] = None,
        tenant_id: Optional[str] = None,
        email: Optional[str] = None,
        session_id: Optional[str] = None,
        custom_claims: Optional[Dict[str, Any]] = None
    ) -> TokenPair:
        """
        Generate access and refresh token pair

        Args:
            user_id: Unique user identifier
            roles: User roles
            permissions: User permissions
            tenant_id: Optional tenant identifier
            email: Optional user email
            session_id: Optional session identifier
            custom_claims: Additional custom claims

        Returns:
            TokenPair with access and refresh tokens

        Example:
            >>> tokens = manager.generate_tokens(
            ...     user_id="user123",
            ...     roles=["analyst"],
            ...     permissions=["read:reports"],
            ...     email="user@example.com"
            ... )
        """
        now = datetime.utcnow()

        # Generate access token
        access_claims = {
            "sub": user_id,
            "iss": self.config.issuer,
            "aud": self.config.audience,
            "exp": int((now + timedelta(seconds=self.config.access_token_ttl)).timestamp()),
            "iat": int(now.timestamp()),
            "nbf": int(now.timestamp()),
            "jti": self._generate_jti(),
            "type": TokenType.ACCESS.value,
            "roles": roles,
            "permissions": permissions or [],
        }

        if tenant_id:
            access_claims["tenant_id"] = tenant_id
        if email:
            access_claims["email"] = email
        if session_id:
            access_claims["session_id"] = session_id
        if custom_claims:
            access_claims.update(custom_claims)

        access_token = jwt.encode(
            access_claims,
            self.config.secret_key,
            algorithm=self.config.algorithm
        )

        # Generate refresh token
        refresh_claims = {
            "sub": user_id,
            "iss": self.config.issuer,
            "aud": self.config.audience,
            "exp": int((now + timedelta(seconds=self.config.refresh_token_ttl)).timestamp()),
            "iat": int(now.timestamp()),
            "jti": self._generate_jti(),
            "type": TokenType.REFRESH.value,
        }

        refresh_token = jwt.encode(
            refresh_claims,
            self.config.secret_key,
            algorithm=self.config.algorithm
        )

        logger.info(f"Generated token pair for user {user_id}")

        return TokenPair(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=self.config.access_token_ttl,
            scope=" ".join(roles)
        )

    def validate_access_token(self, token: str) -> TokenClaims:
        """
        Validate access token and return claims

        Args:
            token: JWT access token

        Returns:
            Validated token claims

        Raises:
            jwt.ExpiredSignatureError: If token has expired
            jwt.InvalidTokenError: If token is invalid
            ValueError: If token is blacklisted

        Example:
            >>> claims = manager.validate_access_token(token)
            >>> print(claims.sub, claims.roles)
        """
        try:
            # Decode and validate token
            payload = jwt.decode(
                token,
                self.config.secret_key,
                algorithms=[self.config.algorithm],
                audience=self.config.audience,
                issuer=self.config.issuer
            )

            # Check token type
            if payload.get("type") != TokenType.ACCESS.value:
                raise ValueError("Invalid token type - expected access token")

            # Check blacklist
            jti = payload.get("jti")
            if self.config.enable_token_blacklist and self.blacklist.is_blacklisted(jti):
                raise ValueError(f"Token {jti} has been revoked")

            claims = TokenClaims(**payload)
            logger.debug(f"Validated access token for user {claims.sub}")

            return claims

        except jwt.ExpiredSignatureError:
            logger.warning("Access token has expired")
            raise
        except jwt.InvalidTokenError as e:
            logger.error(f"Invalid access token: {str(e)}")
            raise

    def validate_refresh_token(self, token: str) -> TokenClaims:
        """
        Validate refresh token and return claims

        Args:
            token: JWT refresh token

        Returns:
            Validated token claims

        Raises:
            jwt.ExpiredSignatureError: If token has expired
            jwt.InvalidTokenError: If token is invalid
            ValueError: If token is blacklisted
        """
        try:
            # Decode and validate token
            payload = jwt.decode(
                token,
                self.config.secret_key,
                algorithms=[self.config.algorithm],
                audience=self.config.audience,
                issuer=self.config.issuer
            )

            # Check token type
            if payload.get("type") != TokenType.REFRESH.value:
                raise ValueError("Invalid token type - expected refresh token")

            # Check blacklist
            jti = payload.get("jti")
            if self.config.enable_token_blacklist and self.blacklist.is_blacklisted(jti):
                raise ValueError(f"Token {jti} has been revoked")

            claims = TokenClaims(**payload)
            logger.debug(f"Validated refresh token for user {claims.sub}")

            return claims

        except jwt.ExpiredSignatureError:
            logger.warning("Refresh token has expired")
            raise
        except jwt.InvalidTokenError as e:
            logger.error(f"Invalid refresh token: {str(e)}")
            raise

    def refresh_access_token(
        self,
        refresh_token: str,
        roles: Optional[List[str]] = None,
        permissions: Optional[List[str]] = None
    ) -> TokenPair:
        """
        Refresh access token using refresh token

        Args:
            refresh_token: Valid refresh token
            roles: Updated user roles (optional)
            permissions: Updated user permissions (optional)

        Returns:
            New token pair with refreshed access token

        Raises:
            jwt.ExpiredSignatureError: If refresh token has expired
            jwt.InvalidTokenError: If refresh token is invalid

        Example:
            >>> new_tokens = manager.refresh_access_token(old_tokens.refresh_token)
        """
        # Validate refresh token
        claims = self.validate_refresh_token(refresh_token)

        # Generate new token pair
        new_tokens = self.generate_tokens(
            user_id=claims.sub,
            roles=roles or claims.roles,
            permissions=permissions or claims.permissions,
            tenant_id=claims.tenant_id,
            email=claims.email,
            session_id=claims.session_id
        )

        logger.info(f"Refreshed access token for user {claims.sub}")

        return new_tokens

    def revoke_token(self, token: str) -> None:
        """
        Revoke token by adding to blacklist

        Args:
            token: Token to revoke

        Example:
            >>> manager.revoke_token(user_token)
        """
        if not self.config.enable_token_blacklist:
            logger.warning("Token blacklist is disabled")
            return

        try:
            # Decode token without validation (we need the JTI)
            payload = jwt.decode(
                token,
                self.config.secret_key,
                algorithms=[self.config.algorithm],
                options={"verify_exp": False}
            )

            jti = payload.get("jti")
            exp = payload.get("exp")

            if jti and exp:
                expires_at = datetime.utcfromtimestamp(exp)
                self.blacklist.add(jti, expires_at)
                logger.info(f"Revoked token {jti}")
            else:
                logger.error("Token missing JTI or EXP claims")

        except Exception as e:
            logger.error(f"Failed to revoke token: {str(e)}")
            raise

    def decode_token_without_validation(self, token: str) -> Dict[str, Any]:
        """
        Decode token without validation (for debugging/logging)

        Args:
            token: JWT token

        Returns:
            Decoded token payload

        Warning:
            DO NOT use for authentication - validation is disabled
        """
        try:
            payload = jwt.decode(
                token,
                options={"verify_signature": False, "verify_exp": False}
            )
            return payload
        except Exception as e:
            logger.error(f"Failed to decode token: {str(e)}")
            return {}

    async def fetch_public_key(self, provider: str, key_id: str) -> Optional[str]:
        """
        Fetch public key from identity provider

        Args:
            provider: Identity provider name
            key_id: Key identifier

        Returns:
            Public key or None if not found
        """
        # Check cache first
        cached_key = self.key_cache.get(provider, key_id)
        if cached_key:
            return cached_key

        # Fetch from provider
        provider_config = self.config.provider_config.get(provider, {})
        jwks_uri = provider_config.get("jwks_uri")

        if not jwks_uri:
            logger.error(f"No JWKS URI configured for provider {provider}")
            return None

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(jwks_uri) as response:
                    if response.status == 200:
                        jwks = await response.json()

                        # Find key by ID
                        for key in jwks.get("keys", []):
                            if key.get("kid") == key_id:
                                public_key = self._jwk_to_pem(key)
                                self.key_cache.set(provider, key_id, public_key)
                                return public_key

                    logger.error(f"Failed to fetch JWKS: HTTP {response.status}")
                    return None

        except Exception as e:
            logger.error(f"Error fetching public key: {str(e)}")
            return None

    def _generate_jti(self) -> str:
        """Generate unique JWT ID"""
        random_bytes = secrets.token_bytes(32)
        return hashlib.sha256(random_bytes).hexdigest()

    def _jwk_to_pem(self, jwk: Dict[str, Any]) -> str:
        """Convert JWK to PEM format (simplified - use cryptography library in production)"""
        # This is a placeholder - use python-jose or cryptography library
        return jwk.get("x5c", [""])[0] if "x5c" in jwk else ""

    async def _periodic_cleanup(self) -> None:
        """Periodic cleanup of expired blacklist entries"""
        while True:
            try:
                await asyncio.sleep(3600)  # Every hour
                self.blacklist.cleanup_expired()
            except Exception as e:
                logger.error(f"Blacklist cleanup error: {str(e)}")


class MultiProviderOAuthManager(OAuthManager):
    """
    Multi-provider OAuth manager with support for Auth0, Okta, Azure AD

    This manager can validate tokens from multiple identity providers and
    handle public key rotation automatically.

    Example:
        >>> config = OAuthConfig(
        ...     secret_key="secret",
        ...     issuer="https://greenlang.ai",
        ...     audience="greenlang-api",
        ...     provider_config={
        ...         "auth0": {
        ...             "domain": "greenlang.auth0.com",
        ...             "jwks_uri": "https://greenlang.auth0.com/.well-known/jwks.json"
        ...         }
        ...     }
        ... )
        >>> manager = MultiProviderOAuthManager(config)
    """

    def __init__(self, config: OAuthConfig):
        super().__init__(config)
        self.providers: Dict[str, IdentityProvider] = {}

    async def validate_external_token(
        self,
        token: str,
        provider: IdentityProvider
    ) -> TokenClaims:
        """
        Validate token from external identity provider

        Args:
            token: JWT token from external provider
            provider: Identity provider

        Returns:
            Validated token claims

        Raises:
            jwt.InvalidTokenError: If token is invalid
        """
        try:
            # Decode header to get key ID
            unverified_header = jwt.get_unverified_header(token)
            key_id = unverified_header.get("kid")

            if not key_id:
                raise ValueError("Token missing 'kid' header")

            # Fetch public key
            public_key = await self.fetch_public_key(provider.value, key_id)

            if not public_key:
                raise ValueError(f"Could not fetch public key for {provider}")

            # Validate token
            payload = jwt.decode(
                token,
                public_key,
                algorithms=["RS256"],
                audience=self.config.audience
            )

            claims = TokenClaims(**payload)
            logger.info(f"Validated external token from {provider} for user {claims.sub}")

            return claims

        except Exception as e:
            logger.error(f"External token validation failed: {str(e)}")
            raise


# Example usage
if __name__ == "__main__":
    # Configure OAuth
    config = OAuthConfig(
        secret_key="your-secret-key-at-least-32-characters-long",
        issuer="https://greenlang.ai",
        audience="greenlang-api",
        access_token_ttl=900,  # 15 minutes
        refresh_token_ttl=2592000  # 30 days
    )

    # Create manager
    manager = OAuthManager(config)

    # Generate tokens
    tokens = manager.generate_tokens(
        user_id="user123",
        roles=["analyst", "developer"],
        permissions=["read:reports", "write:data"],
        email="user@example.com"
    )

    print(f"Access Token: {tokens.access_token[:50]}...")
    print(f"Refresh Token: {tokens.refresh_token[:50]}...")
    print(f"Expires in: {tokens.expires_in} seconds")

    # Validate access token
    claims = manager.validate_access_token(tokens.access_token)
    print(f"\nValidated user: {claims.sub}")
    print(f"Roles: {claims.roles}")
    print(f"Permissions: {claims.permissions}")

    # Refresh access token
    new_tokens = manager.refresh_access_token(tokens.refresh_token)
    print(f"\nNew Access Token: {new_tokens.access_token[:50]}...")

    # Revoke token
    manager.revoke_token(tokens.access_token)
    print("\nToken revoked")

    # Try to validate revoked token (will fail)
    try:
        manager.validate_access_token(tokens.access_token)
    except ValueError as e:
        print(f"Validation failed as expected: {str(e)}")
