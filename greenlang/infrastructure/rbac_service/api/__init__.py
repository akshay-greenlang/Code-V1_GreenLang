# -*- coding: utf-8 -*-
"""
RBAC REST API - SEC-002

FastAPI routers, schemas, and combined router for the RBAC authorization
REST API.  Assembles all sub-routers (roles, permissions, assignments, check)
into a single ``rbac_router`` for easy inclusion in the GreenLang application.

Provides:
    - Role management endpoints (CRUD, enable/disable, hierarchy)
    - Permission management endpoints (list, grant, revoke)
    - Assignment management endpoints (assign, revoke, user roles/permissions)
    - Permission check endpoint (runtime authorization evaluation)

Example:
    >>> from fastapi import FastAPI
    >>> from greenlang.infrastructure.rbac_service.api import rbac_router
    >>> app = FastAPI()
    >>> app.include_router(rbac_router)

Author: GreenLang Framework Team
Date: February 2026
Status: Production Ready
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

try:
    from fastapi import APIRouter

    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    APIRouter = None  # type: ignore[misc, assignment]

from greenlang.infrastructure.rbac_service.api.roles_routes import roles_router
from greenlang.infrastructure.rbac_service.api.permissions_routes import (
    permissions_router,
)
from greenlang.infrastructure.rbac_service.api.assignments_routes import (
    assignments_router,
)
from greenlang.infrastructure.rbac_service.api.check_routes import check_router

# Combined router
if FASTAPI_AVAILABLE and all(
    r is not None
    for r in (roles_router, permissions_router, assignments_router, check_router)
):
    rbac_router = APIRouter()
    rbac_router.include_router(roles_router)
    rbac_router.include_router(permissions_router)
    rbac_router.include_router(assignments_router)
    rbac_router.include_router(check_router)
else:
    rbac_router = None  # type: ignore[assignment]
    logger.warning(
        "FastAPI not available or sub-routers failed to initialize - "
        "rbac_router is None"
    )

__all__ = [
    "rbac_router",
    "roles_router",
    "permissions_router",
    "assignments_router",
    "check_router",
]
