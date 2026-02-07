# -*- coding: utf-8 -*-
"""
Auth Service REST API - SEC-001

FastAPI routers for authentication, user self-service, and
administrative endpoints.

Routers:
    auth_router  - /auth/login, /auth/token, /auth/refresh, /auth/revoke,
                   /auth/logout, /auth/validate, /auth/me, /auth/jwks
    user_router  - /auth/password/change, /auth/password/reset,
                   /auth/mfa/setup, /auth/mfa/verify
    admin_router - /auth/admin/users, /auth/admin/sessions,
                   /auth/admin/audit-log, /auth/admin/lockouts
"""

from __future__ import annotations

from greenlang.infrastructure.auth_service.api.admin_routes import router as admin_router

# NOTE: auth_routes and user_routes are created in a separate SEC-001 phase.
# They are imported conditionally to avoid ImportError during phased rollout.
try:
    from greenlang.infrastructure.auth_service.api.auth_routes import router as auth_router
except ImportError:
    auth_router = None  # type: ignore[assignment]

try:
    from greenlang.infrastructure.auth_service.api.user_routes import router as user_router
except ImportError:
    user_router = None  # type: ignore[assignment]

__all__ = [
    "admin_router",
    "auth_router",
    "user_router",
]
