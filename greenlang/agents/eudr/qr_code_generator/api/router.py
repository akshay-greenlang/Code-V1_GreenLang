# -*- coding: utf-8 -*-
"""
QR Code Generator API Router - AGENT-EUDR-014

Main router aggregating 8 domain-specific sub-routers plus a health
endpoint for the QR Code Generator Agent.

Prefix: /v1/eudr-qrg
Tags: eudr-qr-code-generator

Sub-routers:
    - qr_routes: QR code generation (5 endpoints)
    - payload_routes: Payload composition (4 endpoints)
    - label_routes: Label rendering (5 endpoints)
    - batch_code_routes: Batch code management (4 endpoints)
    - verification_routes: Verification URL & signature (4 endpoints)
    - counterfeit_routes: Anti-counterfeiting (4 endpoints)
    - bulk_routes: Bulk generation jobs (5 endpoints)
    - lifecycle_routes: QR code lifecycle (5 endpoints)
    + health (1 endpoint) = 37 total

Auth & RBAC:
    All endpoints (except health) require JWT auth via SEC-001 and
    check eudr-qrg:* permissions via SEC-002.

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-014, Section 7.4
Agent ID: GL-EUDR-QRG-014
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
    from greenlang.agents.eudr.qr_code_generator.api.qr_routes import (
        router as qr_router,
    )
except ImportError:
    qr_router = None  # type: ignore[assignment]

try:
    from greenlang.agents.eudr.qr_code_generator.api.payload_routes import (
        router as payload_router,
    )
except ImportError:
    payload_router = None  # type: ignore[assignment]

try:
    from greenlang.agents.eudr.qr_code_generator.api.label_routes import (
        router as label_router,
    )
except ImportError:
    label_router = None  # type: ignore[assignment]

try:
    from greenlang.agents.eudr.qr_code_generator.api.batch_code_routes import (
        router as batch_code_router,
    )
except ImportError:
    batch_code_router = None  # type: ignore[assignment]

try:
    from greenlang.agents.eudr.qr_code_generator.api.verification_routes import (
        router as verification_router,
    )
except ImportError:
    verification_router = None  # type: ignore[assignment]

try:
    from greenlang.agents.eudr.qr_code_generator.api.counterfeit_routes import (
        router as counterfeit_router,
    )
except ImportError:
    counterfeit_router = None  # type: ignore[assignment]

try:
    from greenlang.agents.eudr.qr_code_generator.api.bulk_routes import (
        router as bulk_router,
    )
except ImportError:
    bulk_router = None  # type: ignore[assignment]

try:
    from greenlang.agents.eudr.qr_code_generator.api.lifecycle_routes import (
        router as lifecycle_router,
    )
except ImportError:
    lifecycle_router = None  # type: ignore[assignment]

from greenlang.agents.eudr.qr_code_generator.api.schemas import (
    HealthResponse,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Startup time for uptime tracking
# ---------------------------------------------------------------------------

_startup_time = datetime.now(timezone.utc).replace(microsecond=0)

# ---------------------------------------------------------------------------
# Main router with /v1/eudr-qrg prefix
# ---------------------------------------------------------------------------

router = APIRouter(
    prefix="/v1/eudr-qrg",
    tags=["eudr-qr-code-generator"],
    responses={
        401: {"description": "Authentication required"},
        403: {"description": "Insufficient permissions"},
        429: {"description": "Rate limit exceeded"},
        500: {"description": "Internal server error"},
    },
)

# Include all sub-routers that were successfully imported
if qr_router is not None:
    router.include_router(qr_router)
if payload_router is not None:
    router.include_router(payload_router)
if label_router is not None:
    router.include_router(label_router)
if batch_code_router is not None:
    router.include_router(batch_code_router)
if verification_router is not None:
    router.include_router(verification_router)
if counterfeit_router is not None:
    router.include_router(counterfeit_router)
if bulk_router is not None:
    router.include_router(bulk_router)
if lifecycle_router is not None:
    router.include_router(lifecycle_router)


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description=(
        "Check EUDR QR Code Generator API health and component "
        "status. No authentication required."
    ),
    tags=["System"],
)
async def health_check() -> HealthResponse:
    """Health check endpoint for load balancers and monitoring.

    Returns:
        HealthResponse with service health and component status.
    """
    now = datetime.now(timezone.utc).replace(microsecond=0)
    uptime_seconds = (now - _startup_time).total_seconds()

    return HealthResponse(
        uptime_seconds=uptime_seconds,
        checked_at=now,
    )


# ---------------------------------------------------------------------------
# Router factory
# ---------------------------------------------------------------------------


def get_router() -> APIRouter:
    """Return the EUDR QR Code Generator API router for mounting.

    Usage:
        >>> from greenlang.agents.eudr.qr_code_generator.api import get_router
        >>> app.include_router(get_router(), prefix="/api")

    Returns:
        Configured APIRouter with all QR code generator endpoints.
    """
    return router


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "router",
    "get_router",
]
