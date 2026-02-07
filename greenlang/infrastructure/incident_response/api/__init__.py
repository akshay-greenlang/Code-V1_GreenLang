# -*- coding: utf-8 -*-
"""
Incident Response API Package - SEC-010

FastAPI routers for incident response automation endpoints.

This package provides REST API endpoints for:
    - Incident management (CRUD, status updates)
    - Playbook execution
    - Timeline and metrics retrieval

Example:
    >>> from greenlang.infrastructure.incident_response.api import incident_router
    >>> app.include_router(incident_router, prefix="/api/v1/secops")

Author: GreenLang Security Team
Date: February 2026
Status: Production Ready
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Try to import the router
try:
    from greenlang.infrastructure.incident_response.api.incident_routes import (
        incident_router,
        FASTAPI_AVAILABLE,
    )
except ImportError as e:
    logger.warning("Failed to import incident_routes: %s", e)
    incident_router = None  # type: ignore[assignment]
    FASTAPI_AVAILABLE = False


__all__ = [
    "incident_router",
    "FASTAPI_AVAILABLE",
]
