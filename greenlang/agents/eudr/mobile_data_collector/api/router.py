# -*- coding: utf-8 -*-
"""
Mobile Data Collector API Router - AGENT-EUDR-015

Main router aggregating 8 domain-specific sub-routers plus a health
endpoint for the Mobile Data Collector Agent.

Prefix: /v1/eudr-mdc
Tags: eudr-mobile-data-collector

Sub-routers:
    - form_routes: Form management (7 endpoints)
    - gps_routes: GPS capture (7 endpoints)
    - photo_routes: Photo evidence (7 endpoints)
    - sync_routes: Sync management (6 endpoints)
    - template_routes: Form templates (8 endpoints)
    - signature_routes: Digital signatures (7 endpoints)
    - package_routes: Data packages (11 endpoints)
    - device_routes: Device fleet (11 endpoints)
    + health (1 endpoint) = 65 total

Auth & RBAC:
    All endpoints (except health) require JWT auth via SEC-001 and
    check eudr-mdc:* permissions via SEC-002.

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-015, Section 7.4
Agent ID: GL-EUDR-MDC-015
Status: Production Ready
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from fastapi import APIRouter

# ---------------------------------------------------------------------------
# Sub-router imports (try/except for import safety during startup)
# ---------------------------------------------------------------------------

try:
    from greenlang.agents.eudr.mobile_data_collector.api.form_routes import (
        router as form_router,
    )
except ImportError:
    form_router = None  # type: ignore[assignment]

try:
    from greenlang.agents.eudr.mobile_data_collector.api.gps_routes import (
        router as gps_router,
    )
except ImportError:
    gps_router = None  # type: ignore[assignment]

try:
    from greenlang.agents.eudr.mobile_data_collector.api.photo_routes import (
        router as photo_router,
    )
except ImportError:
    photo_router = None  # type: ignore[assignment]

try:
    from greenlang.agents.eudr.mobile_data_collector.api.sync_routes import (
        router as sync_router,
    )
except ImportError:
    sync_router = None  # type: ignore[assignment]

try:
    from greenlang.agents.eudr.mobile_data_collector.api.template_routes import (
        router as template_router,
    )
except ImportError:
    template_router = None  # type: ignore[assignment]

try:
    from greenlang.agents.eudr.mobile_data_collector.api.signature_routes import (
        router as signature_router,
    )
except ImportError:
    signature_router = None  # type: ignore[assignment]

try:
    from greenlang.agents.eudr.mobile_data_collector.api.package_routes import (
        router as package_router,
    )
except ImportError:
    package_router = None  # type: ignore[assignment]

try:
    from greenlang.agents.eudr.mobile_data_collector.api.device_routes import (
        router as device_router,
    )
except ImportError:
    device_router = None  # type: ignore[assignment]

from greenlang.agents.eudr.mobile_data_collector.api.schemas import (
    HealthSchema,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Startup time for uptime tracking
# ---------------------------------------------------------------------------

_startup_time = datetime.now(timezone.utc).replace(microsecond=0)

# ---------------------------------------------------------------------------
# Main router with /v1/eudr-mdc prefix
# ---------------------------------------------------------------------------

router = APIRouter(
    prefix="/v1/eudr-mdc",
    tags=["eudr-mobile-data-collector"],
    responses={
        401: {"description": "Authentication required"},
        403: {"description": "Insufficient permissions"},
        429: {"description": "Rate limit exceeded"},
        500: {"description": "Internal server error"},
    },
)

# Include all sub-routers that were successfully imported
if form_router is not None:
    router.include_router(form_router)
if gps_router is not None:
    router.include_router(gps_router)
if photo_router is not None:
    router.include_router(photo_router)
if sync_router is not None:
    router.include_router(sync_router)
if template_router is not None:
    router.include_router(template_router)
if signature_router is not None:
    router.include_router(signature_router)
if package_router is not None:
    router.include_router(package_router)
if device_router is not None:
    router.include_router(device_router)


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------


@router.get(
    "/health",
    response_model=HealthSchema,
    summary="Health check",
    description=(
        "Check EUDR Mobile Data Collector API health and component "
        "status. No authentication required."
    ),
    tags=["System"],
)
async def health_check() -> HealthSchema:
    """Health check endpoint for load balancers and monitoring.

    Returns:
        HealthSchema with service health and component status.
    """
    now = datetime.now(timezone.utc).replace(microsecond=0)
    uptime_seconds = (now - _startup_time).total_seconds()

    return HealthSchema(
        uptime_seconds=uptime_seconds,
        checked_at=now,
    )


# ---------------------------------------------------------------------------
# Router factory
# ---------------------------------------------------------------------------


def get_router() -> APIRouter:
    """Return the EUDR Mobile Data Collector API router for mounting.

    Usage:
        >>> from greenlang.agents.eudr.mobile_data_collector.api import get_router
        >>> app.include_router(get_router(), prefix="/api")

    Returns:
        Configured APIRouter with all mobile data collector endpoints.
    """
    return router


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "router",
    "get_router",
]
