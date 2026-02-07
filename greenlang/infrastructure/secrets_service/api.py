# -*- coding: utf-8 -*-
"""
Secrets Service REST API - SEC-006

Re-exports the secrets API router for inclusion in the main application.
The actual implementation is in api/secrets_routes.py.

Routes:
    secrets_router - /api/v1/secrets/* endpoints including:
        - CRUD operations (GET, POST, PUT, DELETE)
        - Version management (GET /versions)
        - Undelete (POST /undelete)
        - Rotation triggers (POST /rotate)
        - Health and status endpoints

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Re-export the secrets router from the implementation module
try:
    from greenlang.infrastructure.secrets_service.api.secrets_routes import (
        secrets_router,
    )
except ImportError:
    # Fall back to a stub if the implementation is not available
    secrets_router = None  # type: ignore[assignment]
    logger.warning(
        "secrets_routes module not available; secrets_router is None"
    )


__all__ = [
    "secrets_router",
]
