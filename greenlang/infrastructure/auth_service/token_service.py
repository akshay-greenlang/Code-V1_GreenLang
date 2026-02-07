# -*- coding: utf-8 -*-
"""
Token Service - JWT Authentication Service (SEC-001)

Manages the full lifecycle of JWT access tokens: issuance, validation,
decoding (introspection), and JWKS public-key distribution.  Wraps the
existing ``greenlang.auth.jwt_handler.JWTHandler`` with production
concerns that the low-level handler does not own:

* **JTI tracking** -- every issued token is recorded so it can be
  individually revoked later.
* **Revocation checking** -- on every validation call the JTI is
  checked against the ``RevocationService`` *before* returning claims.
* **Claims standardisation** -- a thin ``TokenClaims`` dataclass keeps
  the public surface uniform regardless of how the underlying handler
  evolves.
* **Audit event emission** -- issuance and validation outcomes are
  logged at ``INFO`` / ``WARNING`` level for the Loki pipeline.

All public methods are ``async`` so they integrate cleanly with
FastAPI and the wider async infrastructure.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Try importing the existing JWTHandler; fall back to standalone path
# ---------------------------------------------------------------------------
try:
    from greenlang.auth.jwt_handler import (
        JWTHandler,
        JWTConfig,
        JWTClaims,
        JWTError,
        TokenExpiredError,
        InvalidTokenError,
        InvalidSignatureError,
        KeyNotFoundError,
    )

    _HAS_JWT_HANDLER = True
except ImportError:  # pragma: no cover
    _HAS_JWT_HANDLER = False
    logger.warning(
        "greenlang.auth.jwt_handler not available; "
        "TokenService will operate in standalone mode"
    )

# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------


@dataclass
class TokenClaims:
    """Standard JWT claims used throughout the Auth Service.

    Attributes:
        sub: User identifier (``sub`` claim).
        tenant_id: Tenant scope for multi-tenancy isolation.
        roles: RBAC role names assigned to the user.
        permissions: Fine-grained permission strings.
        scopes: OAuth2-style scopes.
        email: Optional user email.
        name: Optional display name.
    """

    sub: str
    tenant_id: str
    roles: List[str] = field(default_factory=list)
    permissions: List[str] = field(default_factory=list)
    scopes: List[str] = field(default_factory=list)
    email: Optional[str] = None
    name: Optional[str] = None


@dataclass
class IssuedToken:
    """Represents an issued JWT access token returned to the caller.

    Attributes:
        access_token: The signed JWT string.
        token_type: Always ``"Bearer"``.
        expires_in: Lifetime in seconds from the moment of issuance.
        expires_at: Absolute UTC expiry timestamp.
        jti: Unique token identifier for revocation tracking.
        scope: Space-separated scope string.
    """

    access_token: str
    token_type: str = "Bearer"
    expires_in: int = 3600
    expires_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    jti: str = field(default_factory=lambda: str(uuid.uuid4()))
    scope: str = ""


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class TokenService:
    """JWT access-token lifecycle management.

    Wraps ``greenlang.auth.jwt_handler.JWTHandler`` with:

    * JTI tracking for every issued token.
    * Revocation check during validation.
    * Claims standardisation via ``TokenClaims``.
    * Audit-event emission (structured logging).

    Args:
        jwt_handler: An existing ``JWTHandler`` instance.  When *None* a
            fresh handler is created using the provided ``issuer``,
            ``audience``, and ``access_token_ttl`` values.
        revocation_service: The ``RevocationService`` used for JTI
            blacklist checks.  May be *None* in development/test
            environments (revocation checking is then skipped).
        redis_client: Optional async Redis client for JTI caching.
        issuer: JWT ``iss`` claim.
        audience: JWT ``aud`` claim.
        access_token_ttl: Token lifetime in seconds.

    Example:
        >>> svc = TokenService(revocation_service=revocation)
        >>> token = await svc.issue_token(TokenClaims(
        ...     sub="user-1", tenant_id="t-acme", roles=["viewer"]
        ... ))
        >>> claims = await svc.validate_token(token.access_token)
    """

    def __init__(
        self,
        jwt_handler: Any = None,
        revocation_service: Any = None,
        redis_client: Any = None,
        issuer: str = "greenlang",
        audience: str = "greenlang-api",
        access_token_ttl: int = 3600,
    ) -> None:
        self._issuer = issuer
        self._audience = audience
        self._access_token_ttl = access_token_ttl
        self._revocation_service = revocation_service
        self._redis = redis_client

        # Initialise the underlying JWT handler
        if jwt_handler is not None:
            self._jwt_handler = jwt_handler
        elif _HAS_JWT_HANDLER:
            config = JWTConfig(
                issuer=issuer,
                audience=audience,
                token_expiry_seconds=access_token_ttl,
            )
            self._jwt_handler = JWTHandler(config)
        else:  # pragma: no cover
            raise RuntimeError(
                "No jwt_handler supplied and greenlang.auth.jwt_handler "
                "is not importable.  Cannot initialise TokenService."
            )

        # In-memory JTI set for fast lookup when Redis is unavailable
        self._issued_jtis: Set[str] = set()

        logger.info(
            "TokenService initialised  issuer=%s  audience=%s  ttl=%ds",
            issuer,
            audience,
            access_token_ttl,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def issue_token(self, claims: TokenClaims) -> IssuedToken:
        """Issue a new JWT access token with JTI tracking.

        Args:
            claims: The ``TokenClaims`` to embed in the JWT.

        Returns:
            An ``IssuedToken`` containing the signed JWT, JTI, and expiry
            metadata.

        Raises:
            ValueError: If required claim fields are missing or empty.
            RuntimeError: If the underlying handler fails to sign.
        """
        self._validate_claims(claims)

        start = datetime.now(timezone.utc)
        jti = str(uuid.uuid4())
        expires_at = start + timedelta(seconds=self._access_token_ttl)

        # Delegate signing to the existing JWTHandler
        token_str = self._jwt_handler.generate_token(
            user_id=claims.sub,
            tenant_id=claims.tenant_id,
            roles=claims.roles,
            permissions=claims.permissions,
            email=claims.email,
            name=claims.name,
            expiry_seconds=self._access_token_ttl,
            additional_claims={
                "jti": jti,
                "scopes": claims.scopes,
            },
        )

        # Track JTI
        self._issued_jtis.add(jti)
        await self._cache_jti(jti, self._access_token_ttl)

        elapsed_ms = (datetime.now(timezone.utc) - start).total_seconds() * 1000
        logger.info(
            "Token issued  sub=%s  tenant=%s  jti=%s  ttl=%ds  elapsed=%.1fms",
            claims.sub,
            claims.tenant_id,
            jti,
            self._access_token_ttl,
            elapsed_ms,
        )

        return IssuedToken(
            access_token=token_str,
            token_type="Bearer",
            expires_in=self._access_token_ttl,
            expires_at=expires_at,
            jti=jti,
            scope=" ".join(claims.scopes),
        )

    async def validate_token(self, token: str) -> Optional[TokenClaims]:
        """Validate a JWT: signature, expiry, and JTI revocation status.

        The validation pipeline is:

        1. Decode & verify signature + standard claims via ``JWTHandler``.
        2. Extract ``jti`` and check revocation via ``RevocationService``.
        3. Map the raw claims into a ``TokenClaims`` dataclass.

        Args:
            token: The raw JWT string (without ``Bearer`` prefix).

        Returns:
            ``TokenClaims`` when the token is valid, or *None* when
            validation fails for any reason.
        """
        start = datetime.now(timezone.utc)

        try:
            jwt_claims = self._jwt_handler.validate_token(token)
        except Exception as exc:
            logger.warning("Token validation failed: %s", exc)
            return None

        # Revocation check
        jti = jwt_claims.jti
        if self._revocation_service is not None:
            try:
                revoked = await self._revocation_service.is_revoked(jti)
            except Exception as exc:
                logger.error(
                    "Revocation check error for jti=%s: %s", jti, exc
                )
                # Fail-closed: treat as revoked if the check itself errors
                revoked = True

            if revoked:
                logger.warning("Token revoked  jti=%s  sub=%s", jti, jwt_claims.sub)
                return None

        elapsed_ms = (datetime.now(timezone.utc) - start).total_seconds() * 1000
        logger.debug(
            "Token validated  sub=%s  tenant=%s  jti=%s  elapsed=%.1fms",
            jwt_claims.sub,
            jwt_claims.tenant_id,
            jti,
            elapsed_ms,
        )

        scopes_raw = getattr(jwt_claims, "scope", None) or ""
        scopes_list: List[str] = []
        if isinstance(scopes_raw, str) and scopes_raw:
            scopes_list = scopes_raw.split()
        elif isinstance(scopes_raw, list):
            scopes_list = scopes_raw

        return TokenClaims(
            sub=jwt_claims.sub,
            tenant_id=jwt_claims.tenant_id,
            roles=list(jwt_claims.roles),
            permissions=list(jwt_claims.permissions),
            scopes=scopes_list,
            email=jwt_claims.email,
            name=jwt_claims.name,
        )

    async def decode_token(self, token: str) -> Dict[str, Any]:
        """Decode a token *without* full validation (for introspection).

        This deliberately skips the revocation check so that callers can
        inspect an already-revoked token's payload (e.g., to extract the
        JTI or expiry).

        Args:
            token: The raw JWT string.

        Returns:
            A dictionary of the decoded payload claims, or an empty dict
            on decode failure.
        """
        try:
            import jwt as pyjwt

            # Decode without verification -- introspection only
            payload: Dict[str, Any] = pyjwt.decode(
                token,
                options={
                    "verify_signature": False,
                    "verify_exp": False,
                    "verify_aud": False,
                },
                algorithms=["RS256"],
            )
            return payload
        except Exception as exc:
            logger.warning("Token decode (introspection) failed: %s", exc)
            return {}

    async def get_public_key_jwks(self) -> Dict[str, Any]:
        """Return the public key in JWKS format.

        Delegates to ``JWTHandler.get_jwks()`` which produces a
        standards-compliant JWKS dictionary suitable for exposure at
        ``/.well-known/jwks.json``.

        Returns:
            A JWKS dictionary ``{"keys": [...]}``.
        """
        return self._jwt_handler.get_jwks()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_claims(claims: TokenClaims) -> None:
        """Raise ``ValueError`` when required fields are missing."""
        if not claims.sub:
            raise ValueError("TokenClaims.sub (user_id) must not be empty")
        if not claims.tenant_id:
            raise ValueError("TokenClaims.tenant_id must not be empty")

    async def _cache_jti(self, jti: str, ttl: int) -> None:
        """Optionally cache the JTI in Redis for distributed tracking."""
        if self._redis is None:
            return
        try:
            key = f"gl:auth:jti:{jti}"
            await self._redis.set(key, "1", ex=ttl)
        except Exception as exc:
            logger.warning("Failed to cache JTI in Redis: %s", exc)
