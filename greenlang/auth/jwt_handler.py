# -*- coding: utf-8 -*-
"""
JWT Handler for GreenLang Authentication

Implements RS256 (asymmetric) JWT token generation and validation
following enterprise security best practices.

Features:
- RS256 asymmetric signing with 2048-bit RSA keys
- 1-hour token expiry by default
- Multi-tenant support via tenant_id claim
- Role-based access control via roles claim
- Permission-based access control via permissions claim
- JTI (JWT ID) for token revocation support
- Audience validation
- Key rotation support via JWKS endpoint

Security Compliance:
- SOC 2 CC6.1 (Logical Access)
- ISO 27001 A.10.1 (Cryptographic Controls)
"""

import os
import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any, Set
from pathlib import Path

logger = logging.getLogger(__name__)

# Token expiry constants
DEFAULT_TOKEN_EXPIRY_SECONDS = 3600  # 1 hour
DEFAULT_KEY_SIZE = 2048
ALGORITHM = "RS256"


class JWTError(Exception):
    """Base exception for JWT operations"""
    pass


class TokenExpiredError(JWTError):
    """Token has expired"""
    pass


class InvalidTokenError(JWTError):
    """Token is invalid or malformed"""
    pass


class InvalidSignatureError(JWTError):
    """Token signature verification failed"""
    pass


class KeyNotFoundError(JWTError):
    """RSA key not found"""
    pass


@dataclass
class JWTConfig:
    """Configuration for JWT Handler"""

    # Key management
    private_key_path: Optional[str] = None
    public_key_path: Optional[str] = None
    private_key_pem: Optional[bytes] = None
    public_key_pem: Optional[bytes] = None

    # Token settings
    issuer: str = "greenlang"
    audience: str = "greenlang-api"
    token_expiry_seconds: int = DEFAULT_TOKEN_EXPIRY_SECONDS

    # Key rotation
    key_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

    # JWKS endpoint (for key distribution)
    jwks_url: Optional[str] = None

    # Security settings
    require_tenant_id: bool = True
    allowed_algorithms: List[str] = field(default_factory=lambda: [ALGORITHM])


@dataclass
class JWTClaims:
    """JWT Claims structure following RFC 7519"""

    # Standard claims (RFC 7519)
    sub: str  # Subject (user_id)
    iss: str  # Issuer
    aud: str  # Audience
    exp: datetime  # Expiration time
    iat: datetime  # Issued at
    nbf: datetime  # Not before
    jti: str  # JWT ID (for revocation)

    # Custom claims for GreenLang
    tenant_id: str  # Multi-tenancy support
    roles: List[str] = field(default_factory=list)
    permissions: List[str] = field(default_factory=list)

    # Optional claims
    org_id: Optional[str] = None
    email: Optional[str] = None
    name: Optional[str] = None

    # Token metadata
    token_type: str = "access"
    scope: Optional[str] = None

    def is_expired(self) -> bool:
        """Check if token is expired"""
        return datetime.now(timezone.utc) > self.exp

    def is_valid_time(self) -> bool:
        """Check if token is within valid time window"""
        now = datetime.now(timezone.utc)
        return self.nbf <= now <= self.exp

    def has_role(self, role: str) -> bool:
        """Check if claims include specified role"""
        return role in self.roles

    def has_permission(self, permission: str) -> bool:
        """Check if claims include specified permission"""
        return permission in self.permissions

    def has_any_role(self, roles: List[str]) -> bool:
        """Check if claims include any of the specified roles"""
        return bool(set(self.roles) & set(roles))

    def has_all_roles(self, roles: List[str]) -> bool:
        """Check if claims include all specified roles"""
        return set(roles).issubset(set(self.roles))

    def to_dict(self) -> Dict[str, Any]:
        """Convert claims to dictionary for JWT encoding"""
        return {
            "sub": self.sub,
            "iss": self.iss,
            "aud": self.aud,
            "exp": int(self.exp.timestamp()),
            "iat": int(self.iat.timestamp()),
            "nbf": int(self.nbf.timestamp()),
            "jti": self.jti,
            "tenant_id": self.tenant_id,
            "roles": self.roles,
            "permissions": self.permissions,
            "org_id": self.org_id,
            "email": self.email,
            "name": self.name,
            "token_type": self.token_type,
            "scope": self.scope,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "JWTClaims":
        """Create claims from dictionary (decoded JWT payload)"""
        return cls(
            sub=data["sub"],
            iss=data["iss"],
            aud=data["aud"],
            exp=datetime.fromtimestamp(data["exp"], tz=timezone.utc),
            iat=datetime.fromtimestamp(data["iat"], tz=timezone.utc),
            nbf=datetime.fromtimestamp(data["nbf"], tz=timezone.utc),
            jti=data["jti"],
            tenant_id=data["tenant_id"],
            roles=data.get("roles", []),
            permissions=data.get("permissions", []),
            org_id=data.get("org_id"),
            email=data.get("email"),
            name=data.get("name"),
            token_type=data.get("token_type", "access"),
            scope=data.get("scope"),
        )


class JWTHandler:
    """
    JWT Handler for generating and validating tokens.

    Uses RS256 (RSA with SHA-256) asymmetric signing for enterprise security.
    Private key signs tokens, public key validates them.

    Example:
        ```python
        config = JWTConfig(
            private_key_path="/path/to/private.pem",
            public_key_path="/path/to/public.pem",
            issuer="greenlang",
            audience="greenlang-api"
        )
        handler = JWTHandler(config)

        # Generate token
        token = handler.generate_token(
            user_id="user123",
            tenant_id="tenant456",
            roles=["agent_creator", "viewer"]
        )

        # Validate token
        claims = handler.validate_token(token)
        ```
    """

    def __init__(self, config: Optional[JWTConfig] = None):
        """
        Initialize JWT Handler.

        Args:
            config: JWT configuration. If not provided, will attempt to
                   load from environment variables.
        """
        self.config = config or self._load_config_from_env()
        self._private_key = None
        self._public_key = None
        self._revoked_tokens: Set[str] = set()  # JTI blacklist

        # Load keys
        self._load_keys()

        logger.info(f"JWTHandler initialized with issuer={self.config.issuer}")

    def _load_config_from_env(self) -> JWTConfig:
        """Load configuration from environment variables"""
        return JWTConfig(
            private_key_path=os.getenv("GL_JWT_PRIVATE_KEY_PATH"),
            public_key_path=os.getenv("GL_JWT_PUBLIC_KEY_PATH"),
            issuer=os.getenv("GL_JWT_ISSUER", "greenlang"),
            audience=os.getenv("GL_JWT_AUDIENCE", "greenlang-api"),
            token_expiry_seconds=int(os.getenv("GL_JWT_EXPIRY_SECONDS", DEFAULT_TOKEN_EXPIRY_SECONDS)),
            jwks_url=os.getenv("GL_JWKS_URL"),
        )

    def _load_keys(self) -> None:
        """Load RSA keys from config or generate new ones"""
        try:
            from cryptography.hazmat.primitives import serialization
            from cryptography.hazmat.primitives.asymmetric import rsa
            from cryptography.hazmat.backends import default_backend
        except ImportError:
            raise ImportError(
                "cryptography package is required for JWT operations. "
                "Install with: pip install cryptography"
            )

        # Try loading from PEM bytes first
        if self.config.private_key_pem:
            self._private_key = serialization.load_pem_private_key(
                self.config.private_key_pem,
                password=None,
                backend=default_backend()
            )

        if self.config.public_key_pem:
            self._public_key = serialization.load_pem_public_key(
                self.config.public_key_pem,
                backend=default_backend()
            )

        # Try loading from file paths
        if not self._private_key and self.config.private_key_path:
            key_path = Path(self.config.private_key_path)
            if key_path.exists():
                with open(key_path, "rb") as f:
                    self._private_key = serialization.load_pem_private_key(
                        f.read(),
                        password=None,
                        backend=default_backend()
                    )
                logger.info(f"Loaded private key from {key_path}")

        if not self._public_key and self.config.public_key_path:
            key_path = Path(self.config.public_key_path)
            if key_path.exists():
                with open(key_path, "rb") as f:
                    self._public_key = serialization.load_pem_public_key(
                        f.read(),
                        backend=default_backend()
                    )
                logger.info(f"Loaded public key from {key_path}")

        # If we have private key but no public, derive public from private
        if self._private_key and not self._public_key:
            self._public_key = self._private_key.public_key()

        # Generate keys if none provided (for development only)
        if not self._private_key and not self._public_key:
            logger.warning(
                "No RSA keys provided. Generating temporary keys. "
                "This should NOT be used in production!"
            )
            self._private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=DEFAULT_KEY_SIZE,
                backend=default_backend()
            )
            self._public_key = self._private_key.public_key()

    def generate_token(
        self,
        user_id: str,
        tenant_id: str,
        roles: Optional[List[str]] = None,
        permissions: Optional[List[str]] = None,
        org_id: Optional[str] = None,
        email: Optional[str] = None,
        name: Optional[str] = None,
        expiry_seconds: Optional[int] = None,
        additional_claims: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Generate a JWT token.

        Args:
            user_id: User identifier (becomes 'sub' claim)
            tenant_id: Tenant identifier for multi-tenancy
            roles: List of role names
            permissions: List of permission strings
            org_id: Optional organization ID
            email: Optional user email
            name: Optional user display name
            expiry_seconds: Custom expiry (defaults to config value)
            additional_claims: Additional custom claims to include

        Returns:
            Signed JWT token string

        Raises:
            KeyNotFoundError: If private key is not available
            JWTError: If token generation fails
        """
        try:
            import jwt
        except ImportError:
            raise ImportError(
                "PyJWT package is required for JWT operations. "
                "Install with: pip install PyJWT"
            )

        if not self._private_key:
            raise KeyNotFoundError("Private key is required to generate tokens")

        now = datetime.now(timezone.utc)
        expiry = expiry_seconds or self.config.token_expiry_seconds

        # Build claims
        claims = JWTClaims(
            sub=user_id,
            iss=self.config.issuer,
            aud=self.config.audience,
            exp=now + timedelta(seconds=expiry),
            iat=now,
            nbf=now,
            jti=str(uuid.uuid4()),
            tenant_id=tenant_id,
            roles=roles or [],
            permissions=permissions or [],
            org_id=org_id,
            email=email,
            name=name,
        )

        # Convert to dict and add any additional claims
        payload = claims.to_dict()
        if additional_claims:
            payload.update(additional_claims)

        # Add key ID header for key rotation support
        headers = {"kid": self.config.key_id}

        try:
            token = jwt.encode(
                payload,
                self._private_key,
                algorithm=ALGORITHM,
                headers=headers
            )

            logger.debug(f"Generated token for user={user_id}, tenant={tenant_id}")
            return token

        except Exception as e:
            logger.error(f"Failed to generate token: {e}")
            raise JWTError(f"Token generation failed: {e}")

    def validate_token(self, token: str) -> JWTClaims:
        """
        Validate a JWT token and return its claims.

        Args:
            token: JWT token string

        Returns:
            JWTClaims object with validated claims

        Raises:
            TokenExpiredError: If token has expired
            InvalidTokenError: If token is invalid or malformed
            InvalidSignatureError: If signature verification fails
        """
        try:
            import jwt
            from jwt.exceptions import (
                ExpiredSignatureError,
                InvalidSignatureError as JWTInvalidSignatureError,
                DecodeError,
                InvalidAudienceError,
                InvalidIssuerError,
            )
        except ImportError:
            raise ImportError(
                "PyJWT package is required for JWT operations. "
                "Install with: pip install PyJWT"
            )

        if not self._public_key:
            raise KeyNotFoundError("Public key is required to validate tokens")

        try:
            # Decode and verify token
            payload = jwt.decode(
                token,
                self._public_key,
                algorithms=self.config.allowed_algorithms,
                issuer=self.config.issuer,
                audience=self.config.audience,
                options={
                    "require": ["sub", "iss", "aud", "exp", "iat", "jti", "tenant_id"],
                    "verify_exp": True,
                    "verify_iss": True,
                    "verify_aud": True,
                }
            )

            # Check if token is revoked
            jti = payload.get("jti")
            if jti and jti in self._revoked_tokens:
                raise InvalidTokenError("Token has been revoked")

            # Parse claims
            claims = JWTClaims.from_dict(payload)

            logger.debug(f"Validated token for user={claims.sub}, tenant={claims.tenant_id}")
            return claims

        except ExpiredSignatureError:
            logger.warning("Token validation failed: expired")
            raise TokenExpiredError("Token has expired")

        except JWTInvalidSignatureError:
            logger.warning("Token validation failed: invalid signature")
            raise InvalidSignatureError("Token signature verification failed")

        except (DecodeError, InvalidAudienceError, InvalidIssuerError) as e:
            logger.warning(f"Token validation failed: {e}")
            raise InvalidTokenError(f"Token validation failed: {e}")

        except KeyError as e:
            logger.warning(f"Token validation failed: missing claim {e}")
            raise InvalidTokenError(f"Token missing required claim: {e}")

        except Exception as e:
            logger.error(f"Token validation error: {e}")
            raise InvalidTokenError(f"Token validation failed: {e}")

    def revoke_token(self, jti: str) -> None:
        """
        Add a token to the revocation list.

        In production, this should be backed by Redis or a database
        for persistence across instances.

        Args:
            jti: JWT ID of the token to revoke
        """
        self._revoked_tokens.add(jti)
        logger.info(f"Revoked token with JTI: {jti}")

    def is_token_revoked(self, jti: str) -> bool:
        """
        Check if a token has been revoked.

        Args:
            jti: JWT ID to check

        Returns:
            True if token is revoked
        """
        return jti in self._revoked_tokens

    def get_jwks(self) -> Dict[str, Any]:
        """
        Get the JSON Web Key Set (JWKS) for public key distribution.

        This should be exposed at /.well-known/jwks.json endpoint.

        Returns:
            JWKS dictionary containing public keys
        """
        try:
            from cryptography.hazmat.primitives.asymmetric.rsa import RSAPublicKey
            import base64
        except ImportError:
            raise ImportError("cryptography package required for JWKS generation")

        if not self._public_key:
            return {"keys": []}

        # Get public key numbers
        public_numbers = self._public_key.public_numbers()

        # Convert to base64url encoding
        def _int_to_base64url(val: int, length: int) -> str:
            """Convert integer to base64url encoded string"""
            val_bytes = val.to_bytes(length, byteorder='big')
            return base64.urlsafe_b64encode(val_bytes).rstrip(b'=').decode('ascii')

        # RSA 2048 has 256-byte modulus and variable exponent
        n_bytes = (public_numbers.n.bit_length() + 7) // 8

        jwk = {
            "kty": "RSA",
            "use": "sig",
            "alg": ALGORITHM,
            "kid": self.config.key_id,
            "n": _int_to_base64url(public_numbers.n, n_bytes),
            "e": _int_to_base64url(public_numbers.e, 3),
        }

        return {"keys": [jwk]}

    def refresh_token(self, token: str, extend_by_seconds: Optional[int] = None) -> str:
        """
        Refresh a valid token with extended expiry.

        Args:
            token: Current valid token
            extend_by_seconds: New expiry duration (defaults to config value)

        Returns:
            New token with extended expiry

        Raises:
            TokenExpiredError: If current token is expired
            InvalidTokenError: If current token is invalid
        """
        # Validate current token
        claims = self.validate_token(token)

        # Revoke old token
        self.revoke_token(claims.jti)

        # Generate new token with same claims
        return self.generate_token(
            user_id=claims.sub,
            tenant_id=claims.tenant_id,
            roles=claims.roles,
            permissions=claims.permissions,
            org_id=claims.org_id,
            email=claims.email,
            name=claims.name,
            expiry_seconds=extend_by_seconds,
        )

    def export_public_key_pem(self) -> bytes:
        """
        Export the public key in PEM format.

        Useful for distributing to services that need to validate tokens.

        Returns:
            Public key in PEM format
        """
        from cryptography.hazmat.primitives import serialization

        if not self._public_key:
            raise KeyNotFoundError("No public key available")

        return self._public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )

    @classmethod
    def generate_key_pair(
        cls,
        private_key_path: str,
        public_key_path: str,
        key_size: int = DEFAULT_KEY_SIZE
    ) -> None:
        """
        Generate a new RSA key pair and save to files.

        Args:
            private_key_path: Path to save private key
            public_key_path: Path to save public key
            key_size: RSA key size (default 2048)
        """
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric import rsa
        from cryptography.hazmat.backends import default_backend

        # Generate key pair
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=key_size,
            backend=default_backend()
        )
        public_key = private_key.public_key()

        # Save private key
        private_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )

        private_path = Path(private_key_path)
        private_path.parent.mkdir(parents=True, exist_ok=True)
        with open(private_path, "wb") as f:
            f.write(private_pem)

        # Set restrictive permissions on Unix
        if os.name != "nt":
            os.chmod(private_path, 0o600)

        # Save public key
        public_pem = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )

        public_path = Path(public_key_path)
        public_path.parent.mkdir(parents=True, exist_ok=True)
        with open(public_path, "wb") as f:
            f.write(public_pem)

        logger.info(f"Generated RSA key pair: {private_key_path}, {public_key_path}")
