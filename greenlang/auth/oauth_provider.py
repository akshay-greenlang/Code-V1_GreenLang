# -*- coding: utf-8 -*-
"""
OAuth 2.0 / OpenID Connect (OIDC) Authentication Provider for GreenLang

This module implements OAuth 2.0 and OIDC authentication flows for enterprise SSO.
Supports multiple providers including Google, GitHub, Azure, and generic OAuth/OIDC.

Features:
- OAuth 2.0 flows: authorization code, client credentials, refresh token
- OIDC ID token validation and user info extraction
- Multiple provider support with dynamic registration
- PKCE (Proof Key for Code Exchange) support
- Token exchange and refresh
- JWT validation and signing
- Discovery document support

Security:
- State parameter for CSRF protection
- Nonce validation for ID tokens
- Token signature verification
- Secure token storage
- Automatic token refresh
"""

import logging
import secrets
import hashlib
import base64
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from urllib.parse import urlencode, parse_qs, urlparse
import jwt
from greenlang.determinism import DeterministicClock

try:
    from authlib.integrations.requests_client import OAuth2Session
    from authlib.oauth2.rfc6749 import OAuth2Token
    from authlib.oidc.core import CodeIDToken
    from authlib.jose import JsonWebKey, jwt as authlib_jwt
    AUTHLIB_AVAILABLE = True
except ImportError:
    AUTHLIB_AVAILABLE = False

import requests

logger = logging.getLogger(__name__)


class OAuthGrantType(Enum):
    """OAuth 2.0 grant types"""
    AUTHORIZATION_CODE = "authorization_code"
    CLIENT_CREDENTIALS = "client_credentials"
    REFRESH_TOKEN = "refresh_token"
    IMPLICIT = "implicit"
    PASSWORD = "password"  # Not recommended
    DEVICE_CODE = "urn:ietf:params:oauth:grant-type:device_code"


class OAuthResponseType(Enum):
    """OAuth 2.0 response types"""
    CODE = "code"
    TOKEN = "token"
    ID_TOKEN = "id_token"
    CODE_TOKEN = "code token"
    CODE_ID_TOKEN = "code id_token"
    ID_TOKEN_TOKEN = "id_token token"
    CODE_ID_TOKEN_TOKEN = "code id_token token"


class TokenType(Enum):
    """OAuth token types"""
    BEARER = "Bearer"
    MAC = "MAC"
    DPoP = "DPoP"


@dataclass
class OAuthConfig:
    """OAuth 2.0 / OIDC Provider Configuration"""
    # Client credentials
    client_id: str
    client_secret: str
    redirect_uri: str

    # Provider endpoints
    authorization_endpoint: str
    token_endpoint: str
    userinfo_endpoint: Optional[str] = None
    jwks_uri: Optional[str] = None
    discovery_url: Optional[str] = None

    # OIDC specific
    issuer: Optional[str] = None
    id_token_signing_alg: str = "RS256"

    # OAuth settings
    scope: List[str] = field(default_factory=lambda: ["openid", "profile", "email"])
    grant_type: OAuthGrantType = OAuthGrantType.AUTHORIZATION_CODE
    response_type: OAuthResponseType = OAuthResponseType.CODE

    # Security
    use_pkce: bool = True
    pkce_challenge_method: str = "S256"  # S256 or plain
    validate_issuer: bool = True
    validate_audience: bool = True
    validate_nonce: bool = True

    # Token settings
    token_endpoint_auth_method: str = "client_secret_post"  # client_secret_post, client_secret_basic
    access_token_lifetime: int = 3600
    refresh_token_lifetime: int = 86400

    # Provider-specific
    provider_name: str = "generic"
    provider_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OAuthTokens:
    """OAuth token set"""
    access_token: str
    token_type: str = "Bearer"
    expires_in: Optional[int] = None
    refresh_token: Optional[str] = None
    id_token: Optional[str] = None
    scope: Optional[str] = None

    # Metadata
    issued_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None

    def __post_init__(self):
        if self.expires_in and not self.expires_at:
            self.expires_at = self.issued_at + timedelta(seconds=self.expires_in)

    def is_expired(self) -> bool:
        """Check if access token is expired"""
        if not self.expires_at:
            return False
        return DeterministicClock.utcnow() >= self.expires_at

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "access_token": self.access_token,
            "token_type": self.token_type,
            "expires_in": self.expires_in,
            "refresh_token": self.refresh_token,
            "id_token": self.id_token,
            "scope": self.scope,
        }


@dataclass
class OIDCIDToken:
    """Parsed OIDC ID Token"""
    iss: str  # Issuer
    sub: str  # Subject
    aud: str  # Audience
    exp: int  # Expiration
    iat: int  # Issued at
    nonce: Optional[str] = None
    auth_time: Optional[int] = None
    azp: Optional[str] = None  # Authorized party

    # User claims
    email: Optional[str] = None
    email_verified: Optional[bool] = None
    name: Optional[str] = None
    given_name: Optional[str] = None
    family_name: Optional[str] = None
    picture: Optional[str] = None
    locale: Optional[str] = None

    # Additional claims
    claims: Dict[str, Any] = field(default_factory=dict)

    def is_expired(self) -> bool:
        """Check if ID token is expired"""
        return time.time() >= self.exp


@dataclass
class OAuthUser:
    """User object created from OAuth/OIDC response"""
    user_id: str
    email: str
    username: str
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    display_name: Optional[str] = None
    picture: Optional[str] = None
    email_verified: bool = False
    locale: Optional[str] = None

    # OAuth metadata
    provider: str = "oauth"
    provider_user_id: str = ""

    # Additional attributes
    groups: List[str] = field(default_factory=list)
    roles: List[str] = field(default_factory=list)
    attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OAuthSession:
    """OAuth session"""
    session_id: str
    user: OAuthUser
    tokens: OAuthTokens
    created_at: datetime
    expires_at: datetime
    provider: str


class PKCEHelper:
    """PKCE (Proof Key for Code Exchange) helper"""

    @staticmethod
    def generate_code_verifier(length: int = 128) -> str:
        """Generate code verifier (43-128 characters)"""
        if length < 43 or length > 128:
            raise ValueError("Code verifier length must be between 43 and 128")

        verifier = secrets.token_urlsafe(length)
        return verifier[:length]

    @staticmethod
    def generate_code_challenge(
        code_verifier: str,
        method: str = "S256"
    ) -> str:
        """Generate code challenge from verifier"""
        if method == "plain":
            return code_verifier
        elif method == "S256":
            digest = hashlib.sha256(code_verifier.encode('utf-8')).digest()
            challenge = base64.urlsafe_b64encode(digest).decode('utf-8').rstrip('=')
            return challenge
        else:
            raise ValueError(f"Invalid PKCE method: {method}")


class OIDCDiscovery:
    """OIDC Discovery helper"""

    @staticmethod
    def fetch_configuration(discovery_url: str) -> Dict[str, Any]:
        """Fetch OIDC discovery configuration"""
        try:
            response = requests.get(discovery_url, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to fetch OIDC configuration: {e}")
            raise OAuthError(f"Discovery failed: {e}")

    @staticmethod
    def create_config_from_discovery(
        discovery_url: str,
        client_id: str,
        client_secret: str,
        redirect_uri: str,
        **kwargs
    ) -> OAuthConfig:
        """Create OAuth config from discovery document"""
        config_data = OIDCDiscovery.fetch_configuration(discovery_url)

        return OAuthConfig(
            client_id=client_id,
            client_secret=client_secret,
            redirect_uri=redirect_uri,
            authorization_endpoint=config_data["authorization_endpoint"],
            token_endpoint=config_data["token_endpoint"],
            userinfo_endpoint=config_data.get("userinfo_endpoint"),
            jwks_uri=config_data.get("jwks_uri"),
            issuer=config_data.get("issuer"),
            discovery_url=discovery_url,
            provider_metadata=config_data,
            **kwargs
        )


class JWTValidator:
    """JWT token validator"""

    def __init__(self, config: OAuthConfig):
        self.config = config
        self._jwks_cache: Optional[Dict[str, Any]] = None
        self._jwks_cache_time: Optional[datetime] = None
        self._jwks_cache_ttl = 3600  # 1 hour

    def validate_id_token(
        self,
        id_token: str,
        nonce: Optional[str] = None
    ) -> OIDCIDToken:
        """Validate and decode ID token"""
        try:
            # Get signing keys
            jwks = self._get_jwks()

            # Decode and validate
            claims = jwt.decode(
                id_token,
                key=self._get_signing_key(id_token, jwks),
                algorithms=[self.config.id_token_signing_alg],
                audience=self.config.client_id if self.config.validate_audience else None,
                issuer=self.config.issuer if self.config.validate_issuer else None,
            )

            # Validate nonce
            if self.config.validate_nonce and nonce:
                if claims.get("nonce") != nonce:
                    raise OAuthError("Nonce validation failed")

            # Create ID token object
            id_token_obj = OIDCIDToken(
                iss=claims["iss"],
                sub=claims["sub"],
                aud=claims["aud"],
                exp=claims["exp"],
                iat=claims["iat"],
                nonce=claims.get("nonce"),
                auth_time=claims.get("auth_time"),
                azp=claims.get("azp"),
                email=claims.get("email"),
                email_verified=claims.get("email_verified", False),
                name=claims.get("name"),
                given_name=claims.get("given_name"),
                family_name=claims.get("family_name"),
                picture=claims.get("picture"),
                locale=claims.get("locale"),
                claims=claims
            )

            return id_token_obj

        except jwt.InvalidTokenError as e:
            logger.error(f"ID token validation failed: {e}")
            raise OAuthError(f"Invalid ID token: {e}")

    def _get_jwks(self) -> Dict[str, Any]:
        """Get JWKS from cache or fetch from endpoint"""
        # Check cache
        if self._jwks_cache and self._jwks_cache_time:
            age = (DeterministicClock.utcnow() - self._jwks_cache_time).total_seconds()
            if age < self._jwks_cache_ttl:
                return self._jwks_cache

        # Fetch JWKS
        if not self.config.jwks_uri:
            raise OAuthError("JWKS URI not configured")

        try:
            response = requests.get(self.config.jwks_uri, timeout=10)
            response.raise_for_status()
            jwks = response.json()

            # Cache
            self._jwks_cache = jwks
            self._jwks_cache_time = DeterministicClock.utcnow()

            return jwks

        except Exception as e:
            logger.error(f"Failed to fetch JWKS: {e}")
            raise OAuthError(f"JWKS fetch failed: {e}")

    def _get_signing_key(self, token: str, jwks: Dict[str, Any]) -> str:
        """Get signing key for token"""
        # Get key ID from token header
        unverified_header = jwt.get_unverified_header(token)
        kid = unverified_header.get("kid")

        if not kid:
            raise OAuthError("No 'kid' in token header")

        # Find matching key in JWKS
        for key in jwks.get("keys", []):
            if key.get("kid") == kid:
                # Convert JWK to PEM
                return self._jwk_to_pem(key)

        raise OAuthError(f"Signing key not found: {kid}")

    def _jwk_to_pem(self, jwk: Dict[str, Any]) -> str:
        """Convert JWK to PEM format"""
        try:
            if AUTHLIB_AVAILABLE:
                key = JsonWebKey.import_key(jwk)
                return key
            else:
                # Fallback: use PyJWT's built-in JWK support
                from jwt.algorithms import RSAAlgorithm
                return RSAAlgorithm.from_jwk(json.dumps(jwk))
        except Exception as e:
            logger.error(f"Failed to convert JWK to PEM: {e}")
            raise OAuthError(f"Key conversion failed: {e}")


class OAuthProvider:
    """
    OAuth 2.0 / OIDC Provider Implementation

    Handles OAuth flows, token management, and user authentication.
    """

    def __init__(self, config: OAuthConfig):
        if not AUTHLIB_AVAILABLE:
            logger.warning(
                "authlib is not installed. Some features may be limited. "
                "Install it with: pip install authlib"
            )

        self.config = config
        self.jwt_validator = JWTValidator(config)
        self.sessions: Dict[str, OAuthSession] = {}
        self._state_cache: Dict[str, Dict[str, Any]] = {}

        logger.info(f"Initialized OAuth provider: {config.provider_name}")

    def get_authorization_url(
        self,
        state: Optional[str] = None,
        nonce: Optional[str] = None,
        extra_params: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, str, Optional[str], Optional[str]]:
        """
        Generate OAuth authorization URL

        Returns:
            Tuple of (auth_url, state, code_verifier, nonce)
        """
        # Generate state for CSRF protection
        if not state:
            state = secrets.token_urlsafe(32)

        # Generate nonce for OIDC
        if not nonce and "openid" in self.config.scope:
            nonce = secrets.token_urlsafe(32)

        # PKCE
        code_verifier = None
        code_challenge = None
        if self.config.use_pkce:
            code_verifier = PKCEHelper.generate_code_verifier()
            code_challenge = PKCEHelper.generate_code_challenge(
                code_verifier,
                self.config.pkce_challenge_method
            )

        # Build authorization URL
        params = {
            "client_id": self.config.client_id,
            "redirect_uri": self.config.redirect_uri,
            "response_type": self.config.response_type.value,
            "scope": " ".join(self.config.scope),
            "state": state,
        }

        if nonce:
            params["nonce"] = nonce

        if code_challenge:
            params["code_challenge"] = code_challenge
            params["code_challenge_method"] = self.config.pkce_challenge_method

        if extra_params:
            params.update(extra_params)

        auth_url = f"{self.config.authorization_endpoint}?{urlencode(params)}"

        # Cache state data
        self._state_cache[state] = {
            "nonce": nonce,
            "code_verifier": code_verifier,
            "created_at": DeterministicClock.utcnow(),
        }

        logger.info(f"Generated authorization URL with state: {state}")
        return auth_url, state, code_verifier, nonce

    def exchange_code_for_tokens(
        self,
        code: str,
        state: Optional[str] = None,
        code_verifier: Optional[str] = None
    ) -> OAuthTokens:
        """
        Exchange authorization code for tokens

        Args:
            code: Authorization code
            state: State parameter (for validation)
            code_verifier: PKCE code verifier

        Returns:
            OAuthTokens
        """
        try:
            # Validate state
            state_data = None
            if state:
                state_data = self._state_cache.get(state)
                if not state_data:
                    raise OAuthError("Invalid or expired state")

                # Use cached code verifier if not provided
                if not code_verifier and state_data.get("code_verifier"):
                    code_verifier = state_data["code_verifier"]

            # Prepare token request
            data = {
                "grant_type": OAuthGrantType.AUTHORIZATION_CODE.value,
                "code": code,
                "redirect_uri": self.config.redirect_uri,
                "client_id": self.config.client_id,
            }

            if code_verifier:
                data["code_verifier"] = code_verifier

            # Add client authentication
            if self.config.token_endpoint_auth_method == "client_secret_post":
                data["client_secret"] = self.config.client_secret
            elif self.config.token_endpoint_auth_method == "client_secret_basic":
                # Will be added as Basic auth header
                pass

            # Make token request
            auth = None
            if self.config.token_endpoint_auth_method == "client_secret_basic":
                auth = (self.config.client_id, self.config.client_secret)

            response = requests.post(
                self.config.token_endpoint,
                data=data,
                auth=auth,
                timeout=10
            )
            response.raise_for_status()

            token_data = response.json()

            # Create token object
            tokens = OAuthTokens(
                access_token=token_data["access_token"],
                token_type=token_data.get("token_type", "Bearer"),
                expires_in=token_data.get("expires_in"),
                refresh_token=token_data.get("refresh_token"),
                id_token=token_data.get("id_token"),
                scope=token_data.get("scope"),
            )

            # Validate ID token if present
            if tokens.id_token and state_data:
                nonce = state_data.get("nonce")
                self.jwt_validator.validate_id_token(tokens.id_token, nonce)

            # Clean up state cache
            if state:
                del self._state_cache[state]

            logger.info("Successfully exchanged code for tokens")
            return tokens

        except requests.RequestException as e:
            logger.error(f"Token exchange failed: {e}")
            raise OAuthError(f"Token exchange failed: {e}")

    def refresh_access_token(self, refresh_token: str) -> OAuthTokens:
        """Refresh access token using refresh token"""
        try:
            data = {
                "grant_type": OAuthGrantType.REFRESH_TOKEN.value,
                "refresh_token": refresh_token,
                "client_id": self.config.client_id,
            }

            # Add client authentication
            auth = None
            if self.config.token_endpoint_auth_method == "client_secret_post":
                data["client_secret"] = self.config.client_secret
            elif self.config.token_endpoint_auth_method == "client_secret_basic":
                auth = (self.config.client_id, self.config.client_secret)

            response = requests.post(
                self.config.token_endpoint,
                data=data,
                auth=auth,
                timeout=10
            )
            response.raise_for_status()

            token_data = response.json()

            tokens = OAuthTokens(
                access_token=token_data["access_token"],
                token_type=token_data.get("token_type", "Bearer"),
                expires_in=token_data.get("expires_in"),
                refresh_token=token_data.get("refresh_token", refresh_token),
                id_token=token_data.get("id_token"),
                scope=token_data.get("scope"),
            )

            logger.info("Successfully refreshed access token")
            return tokens

        except requests.RequestException as e:
            logger.error(f"Token refresh failed: {e}")
            raise OAuthError(f"Token refresh failed: {e}")

    def get_user_info(self, access_token: str) -> Dict[str, Any]:
        """Get user info from userinfo endpoint"""
        if not self.config.userinfo_endpoint:
            raise OAuthError("Userinfo endpoint not configured")

        try:
            headers = {
                "Authorization": f"Bearer {access_token}"
            }

            response = requests.get(
                self.config.userinfo_endpoint,
                headers=headers,
                timeout=10
            )
            response.raise_for_status()

            return response.json()

        except requests.RequestException as e:
            logger.error(f"Failed to get user info: {e}")
            raise OAuthError(f"User info request failed: {e}")

    def create_user_from_tokens(self, tokens: OAuthTokens) -> OAuthUser:
        """Create user object from OAuth tokens"""
        user_data = {}

        # Extract from ID token if available
        if tokens.id_token:
            id_token = self.jwt_validator.validate_id_token(tokens.id_token)
            user_data = {
                "user_id": id_token.sub,
                "email": id_token.email or "",
                "email_verified": id_token.email_verified or False,
                "first_name": id_token.given_name,
                "last_name": id_token.family_name,
                "display_name": id_token.name,
                "picture": id_token.picture,
                "locale": id_token.locale,
                "provider_user_id": id_token.sub,
            }
        else:
            # Fetch from userinfo endpoint
            userinfo = self.get_user_info(tokens.access_token)
            user_data = self._map_userinfo_to_user(userinfo)

        # Create user object
        user = OAuthUser(
            user_id=user_data.get("user_id", ""),
            email=user_data.get("email", ""),
            username=user_data.get("username") or user_data.get("email", "").split("@")[0],
            first_name=user_data.get("first_name"),
            last_name=user_data.get("last_name"),
            display_name=user_data.get("display_name"),
            picture=user_data.get("picture"),
            email_verified=user_data.get("email_verified", False),
            locale=user_data.get("locale"),
            provider=self.config.provider_name,
            provider_user_id=user_data.get("provider_user_id", ""),
            groups=user_data.get("groups", []),
            roles=user_data.get("roles", []),
            attributes=user_data.get("attributes", {}),
        )

        return user

    def _map_userinfo_to_user(self, userinfo: Dict[str, Any]) -> Dict[str, Any]:
        """Map userinfo response to user data"""
        return {
            "user_id": userinfo.get("sub") or userinfo.get("id"),
            "email": userinfo.get("email"),
            "email_verified": userinfo.get("email_verified", False),
            "username": userinfo.get("preferred_username") or userinfo.get("login"),
            "first_name": userinfo.get("given_name"),
            "last_name": userinfo.get("family_name"),
            "display_name": userinfo.get("name"),
            "picture": userinfo.get("picture") or userinfo.get("avatar_url"),
            "locale": userinfo.get("locale"),
            "provider_user_id": userinfo.get("sub") or userinfo.get("id"),
            "groups": userinfo.get("groups", []),
            "roles": userinfo.get("roles", []),
            "attributes": userinfo,
        }

    def create_session(self, user: OAuthUser, tokens: OAuthTokens) -> OAuthSession:
        """Create OAuth session"""
        session_id = secrets.token_urlsafe(32)
        now = DeterministicClock.utcnow()

        session = OAuthSession(
            session_id=session_id,
            user=user,
            tokens=tokens,
            created_at=now,
            expires_at=tokens.expires_at or (now + timedelta(seconds=self.config.access_token_lifetime)),
            provider=self.config.provider_name
        )

        self.sessions[session_id] = session
        logger.info(f"Created session for user: {user.email}")

        return session

    def validate_session(self, session_id: str, auto_refresh: bool = True) -> bool:
        """Validate and optionally refresh session"""
        session = self.sessions.get(session_id)
        if not session:
            return False

        # Check if expired
        if DeterministicClock.utcnow() >= session.expires_at:
            # Try to refresh if enabled and refresh token available
            if auto_refresh and session.tokens.refresh_token:
                try:
                    new_tokens = self.refresh_access_token(session.tokens.refresh_token)
                    session.tokens = new_tokens
                    session.expires_at = new_tokens.expires_at
                    logger.info(f"Refreshed session: {session_id}")
                    return True
                except OAuthError:
                    del self.sessions[session_id]
                    return False
            else:
                del self.sessions[session_id]
                return False

        return True

    def get_session(self, session_id: str) -> Optional[OAuthSession]:
        """Get session by ID"""
        if self.validate_session(session_id):
            return self.sessions.get(session_id)
        return None

    def revoke_session(self, session_id: str) -> bool:
        """Revoke session"""
        if session_id in self.sessions:
            del self.sessions[session_id]
            logger.info(f"Revoked session: {session_id}")
            return True
        return False

    def cleanup_expired_sessions(self) -> int:
        """Clean up expired sessions"""
        now = DeterministicClock.utcnow()
        expired = [
            sid for sid, session in self.sessions.items()
            if now >= session.expires_at
        ]

        for sid in expired:
            del self.sessions[sid]

        logger.info(f"Cleaned up {len(expired)} expired sessions")
        return len(expired)


class OAuthError(Exception):
    """OAuth-specific error"""
    pass


# Helper functions for common OAuth providers

def create_google_config(
    client_id: str,
    client_secret: str,
    redirect_uri: str,
    **kwargs
) -> OAuthConfig:
    """Create OAuth config for Google"""
    return OIDCDiscovery.create_config_from_discovery(
        discovery_url="https://accounts.google.com/.well-known/openid-configuration",
        client_id=client_id,
        client_secret=client_secret,
        redirect_uri=redirect_uri,
        provider_name="google",
        **kwargs
    )


def create_github_config(
    client_id: str,
    client_secret: str,
    redirect_uri: str,
    **kwargs
) -> OAuthConfig:
    """Create OAuth config for GitHub"""
    return OAuthConfig(
        client_id=client_id,
        client_secret=client_secret,
        redirect_uri=redirect_uri,
        authorization_endpoint="https://github.com/login/oauth/authorize",
        token_endpoint="https://github.com/login/oauth/access_token",
        userinfo_endpoint="https://api.github.com/user",
        scope=["user:email"],
        provider_name="github",
        use_pkce=False,  # GitHub doesn't support PKCE
        **kwargs
    )


def create_azure_config(
    tenant_id: str,
    client_id: str,
    client_secret: str,
    redirect_uri: str,
    **kwargs
) -> OAuthConfig:
    """Create OAuth config for Azure AD"""
    return OIDCDiscovery.create_config_from_discovery(
        discovery_url=f"https://login.microsoftonline.com/{tenant_id}/v2.0/.well-known/openid-configuration",
        client_id=client_id,
        client_secret=client_secret,
        redirect_uri=redirect_uri,
        provider_name="azure",
        **kwargs
    )


__all__ = [
    "OAuthProvider",
    "OAuthConfig",
    "OAuthTokens",
    "OIDCIDToken",
    "OAuthUser",
    "OAuthSession",
    "OAuthError",
    "OAuthGrantType",
    "OAuthResponseType",
    "TokenType",
    "PKCEHelper",
    "OIDCDiscovery",
    "JWTValidator",
    "create_google_config",
    "create_github_config",
    "create_azure_config",
]
