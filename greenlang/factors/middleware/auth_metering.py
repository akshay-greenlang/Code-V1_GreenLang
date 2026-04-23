# -*- coding: utf-8 -*-
"""
FastAPI middleware: Factors API auth + tier gate + credit metering.

Installs on any FastAPI app that exposes the Factors routes.  The
middleware runs **before** the route handler (auth + tier check) and
**after** (usage event emission).  Existing route-level
``Depends(get_current_user)`` continues to work; this middleware simply
populates ``request.state.user`` so routes can skip the JWT decode when
the caller presented an API key.

Example::

    from fastapi import FastAPI
    from greenlang.factors.middleware.auth_metering import install_factors_middleware

    app = FastAPI()
    install_factors_middleware(app, protected_prefix="/api/v1/factors")

This module does NOT handle tenant overrides, which are applied later in
the route via :mod:`greenlang.factors.tier_enforcement`.
"""
from __future__ import annotations

import logging
from typing import Callable, Optional

from greenlang.factors.api_auth import (
    APIKeyValidator,
    authenticate_headers,
    default_validator,
    tier_allows_endpoint,
    min_tier_for_endpoint,
)
from greenlang.factors.billing.metering import record_usage_event

logger = logging.getLogger(__name__)


def _default_jwt_decoder():
    """Return a callable that decodes a JWT using the existing API config."""
    try:
        import jwt  # type: ignore[import-not-found]

        from greenlang.integration.api.dependencies import (
            JWT_ALGORITHM,
            JWT_SECRET,
        )

        def decode(token: str):
            return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])

        return decode
    except ImportError:  # pragma: no cover - optional dependency
        return None


def install_factors_middleware(
    app,
    *,
    protected_prefix: str = "/api/v1/factors",
    validator: Optional[APIKeyValidator] = None,
    jwt_decode: Optional[Callable] = None,
) -> None:
    """Attach the auth + tier + metering middleware to a FastAPI app.

    Args:
        app: FastAPI application instance.
        protected_prefix: Only requests whose path starts with this
            prefix are subject to auth / metering.
        validator: Custom :class:`APIKeyValidator` (uses the default
            keyring when omitted).
        jwt_decode: Custom JWT decode callable (uses the platform
            default when omitted).
    """
    from fastapi import Request
    from fastapi.responses import JSONResponse

    key_validator = validator or default_validator()
    decoder = jwt_decode or _default_jwt_decoder()

    @app.middleware("http")
    async def factors_auth_metering(request: Request, call_next):
        path = request.url.path
        if not path.startswith(protected_prefix):
            return await call_next(request)

        authorization = request.headers.get("Authorization")
        api_key = request.headers.get("X-API-Key")
        user = authenticate_headers(
            authorization,
            api_key,
            jwt_decode=decoder,
            validator=key_validator,
        )

        if user is None:
            return JSONResponse(
                status_code=401,
                content={
                    "detail": (
                        "Missing or invalid credentials. Provide either "
                        "Authorization: Bearer <jwt> or X-API-Key: <key>."
                    )
                },
            )

        # Tier gate. JWT users default to community unless the token
        # carries a 'tier' claim.  API-key users always carry a tier.
        tier = user.get("tier") or "community"
        if not tier_allows_endpoint(tier, path):
            required = min_tier_for_endpoint(path)
            return JSONResponse(
                status_code=403,
                content={
                    "detail": (
                        f"Endpoint {path} requires tier '{required}'+ "
                        f"(caller tier: '{tier}')."
                    )
                },
            )

        # Expose to route handlers that want to skip the Depends() lookup.
        request.state.user = user
        request.state.tier = tier

        response = await call_next(request)

        # Metering: record after response so row_count is known for
        # endpoints that expose it via the 'X-Row-Count' response header.
        row_count = 1
        try:
            row_count = int(response.headers.get("X-Row-Count", "1") or 1)
        except (TypeError, ValueError):
            row_count = 1

        try:
            record_usage_event(
                tier=tier,
                endpoint=path,
                method=request.method,
                user=user,
                row_count=row_count,
                status_code=response.status_code,
                api_key=api_key,
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("metering failed: %s", exc)

        return response

    logger.info(
        "Factors middleware installed: prefix=%s validator_keys=%d",
        protected_prefix,
        len(key_validator.list_records()),
    )


# ---------------------------------------------------------------------------
# Class-shape middleware (BaseHTTPMiddleware) for use with
# `app.add_middleware(AuthMeteringMiddleware)` from factors_app.py.
# Mirrors the decorator above but applies to /v1 by default.
# ---------------------------------------------------------------------------

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request as _StarletteRequest
from starlette.responses import JSONResponse as _JSONResponse


class AuthMeteringMiddleware(BaseHTTPMiddleware):
    """Class wrapper around the same auth + tier + metering logic.

    Reads protected prefix from ``GL_FACTORS_PROTECTED_PREFIX`` env (default
    ``/v1``). Public routes (/v1/health, /openapi.json, /docs, /redoc,
    /metrics) bypass the gate so monitoring + docs work unauthenticated.
    """

    PUBLIC_PATHS = {
        "/v1/health",
        "/openapi.json",
        "/docs",
        "/redoc",
        "/metrics",
        "/",
    }

    def __init__(self, app, *, protected_prefix: Optional[str] = None,
                 validator: Optional[APIKeyValidator] = None,
                 jwt_decode: Optional[Callable] = None) -> None:
        import os
        super().__init__(app)
        self.protected_prefix = protected_prefix or os.getenv(
            "GL_FACTORS_PROTECTED_PREFIX", "/v1"
        )
        self.validator = validator or default_validator()
        self.decoder = jwt_decode or _default_jwt_decoder()

    async def dispatch(self, request: _StarletteRequest, call_next):
        path = request.url.path
        if path in self.PUBLIC_PATHS:
            return await call_next(request)
        if not path.startswith(self.protected_prefix):
            return await call_next(request)

        authorization = request.headers.get("Authorization")
        api_key = request.headers.get("X-API-Key")
        user = authenticate_headers(
            authorization, api_key,
            jwt_decode=self.decoder, validator=self.validator,
        )
        if user is None:
            return _JSONResponse(
                status_code=401,
                content={
                    "error": "unauthenticated",
                    "message": (
                        "Missing or invalid credentials. Provide either "
                        "Authorization: Bearer <jwt> or X-API-Key: <key>."
                    ),
                },
            )

        tier = user.get("tier") or "community"
        if not tier_allows_endpoint(tier, path):
            required = min_tier_for_endpoint(path)
            return _JSONResponse(
                status_code=403,
                content={
                    "error": "tier_forbidden",
                    "message": (
                        f"Endpoint {path} requires tier '{required}'+ "
                        f"(caller tier: '{tier}')."
                    ),
                    "required_tier": required,
                    "caller_tier": tier,
                    "upgrade_url": "https://greenlang.ai/pricing",
                },
            )

        request.state.user = user
        request.state.tier = tier
        response = await call_next(request)

        row_count = 1
        try:
            row_count = int(response.headers.get("X-Row-Count", "1") or 1)
        except (TypeError, ValueError):
            row_count = 1
        try:
            record_usage_event(
                tier=tier, endpoint=path, method=request.method,
                user=user, row_count=row_count,
                status_code=response.status_code, api_key=api_key,
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("metering failed: %s", exc)
        return response
