# -*- coding: utf-8 -*-
"""
GreenLang Auth Service - SEC-001: Core JWT Authentication Service

Production-grade JWT authentication service wrapping the existing
``greenlang.auth`` modules with service-level concerns: JTI tracking,
two-layer token revocation (Redis + PostgreSQL), opaque refresh-token
rotation with family-based reuse detection, and FastAPI REST endpoints.

Sub-modules:
    token_service   - JWT access-token lifecycle (issue, validate, JWKS).
    revocation      - Two-layer JTI blacklist (Redis L1, PostgreSQL L2).
    refresh_tokens  - Opaque refresh-token rotation with family tracking.
    api             - FastAPI routers for /auth/* and user self-service.

Quick start:
    >>> from greenlang.infrastructure.auth_service import (
    ...     TokenService,
    ...     RevocationService,
    ...     RefreshTokenManager,
    ...     AuthServiceConfig,
    ... )
    >>> revocation = RevocationService(redis_client=redis, db_pool=pool)
    >>> token_svc = TokenService(revocation_service=revocation)
    >>> issued = await token_svc.issue_token(claims)

Author: GreenLang Framework Team
Date: February 2026
Status: Production Ready
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class AuthServiceConfig:
    """Top-level configuration for the Auth Service (SEC-001).

    Attributes:
        issuer: JWT ``iss`` claim value.
        audience: JWT ``aud`` claim value.
        access_token_ttl: Access-token lifetime in seconds (default 3600).
        refresh_token_ttl_days: Refresh-token lifetime in days (default 7).
        max_refresh_family_size: Maximum tokens in a single refresh family.
        reuse_grace_seconds: Grace period for concurrent refresh requests.
        redis_url: Redis connection URL used by revocation and caching.
        database_url: PostgreSQL connection URL for durable storage.
    """

    issuer: str = "greenlang"
    audience: str = "greenlang-api"
    access_token_ttl: int = 3600
    refresh_token_ttl_days: int = 7
    max_refresh_family_size: int = 30
    reuse_grace_seconds: int = 5
    redis_url: Optional[str] = None
    database_url: Optional[str] = None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

from greenlang.infrastructure.auth_service.token_service import (  # noqa: E402
    IssuedToken,
    TokenClaims,
    TokenService,
)
from greenlang.infrastructure.auth_service.revocation import (  # noqa: E402
    RevocationEntry,
    RevocationService,
)
from greenlang.infrastructure.auth_service.refresh_tokens import (  # noqa: E402
    RefreshTokenManager,
    RefreshTokenRecord,
    RefreshTokenResult,
)
from greenlang.infrastructure.auth_service.auth_setup import (  # noqa: E402
    configure_auth,
)

__all__ = [
    # Config
    "AuthServiceConfig",
    # Token Service
    "TokenService",
    "TokenClaims",
    "IssuedToken",
    # Revocation
    "RevocationService",
    "RevocationEntry",
    # Refresh Tokens
    "RefreshTokenManager",
    "RefreshTokenResult",
    "RefreshTokenRecord",
    # Auth Setup
    "configure_auth",
]
