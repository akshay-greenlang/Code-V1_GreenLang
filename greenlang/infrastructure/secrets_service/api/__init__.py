# -*- coding: utf-8 -*-
"""
Secrets Service REST API - SEC-006

FastAPI routers for the Secrets Management Service.

Routers:
    secrets_router   - CRUD operations for secrets
    rotation_router  - Rotation management
    health_router    - Health and status endpoints

Example:
    >>> from fastapi import FastAPI
    >>> from greenlang.infrastructure.secrets_service.api import secrets_router
    >>> app = FastAPI()
    >>> app.include_router(secrets_router, prefix="/api/v1/secrets")

Author: GreenLang Framework Team
Date: February 2026
Status: Production Ready
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Import routers conditionally
try:
    from greenlang.infrastructure.secrets_service.api.secrets_routes import (
        secrets_router,
    )
except ImportError as e:
    secrets_router = None  # type: ignore[assignment]
    logger.warning("secrets_routes import failed: %s", e)

try:
    from greenlang.infrastructure.secrets_service.api.rotation_routes import (
        rotation_router,
    )
except ImportError as e:
    rotation_router = None  # type: ignore[assignment]
    logger.warning("rotation_routes import failed: %s", e)

try:
    from greenlang.infrastructure.secrets_service.api.health_routes import (
        health_router,
    )
except ImportError as e:
    health_router = None  # type: ignore[assignment]
    logger.warning("health_routes import failed: %s", e)


# Combined router for mounting at /api/v1/secrets
def create_combined_router():
    """Create a combined router with all secrets API endpoints.

    Returns:
        APIRouter with secrets, rotation, and health routes.
        Returns None if FastAPI is not available.
    """
    try:
        from fastapi import APIRouter

        combined = APIRouter()

        if secrets_router is not None:
            combined.include_router(secrets_router)

        if rotation_router is not None:
            combined.include_router(rotation_router)

        if health_router is not None:
            combined.include_router(health_router)

        return combined

    except ImportError:
        return None


__all__ = [
    "secrets_router",
    "rotation_router",
    "health_router",
    "create_combined_router",
]
