# -*- coding: utf-8 -*-
"""
Main Router Registration - AGENT-EUDR-007 GPS Coordinate Validator API

Aggregates all route modules under the ``/v1/eudr-gcv`` prefix and
provides ``get_router()`` for integration with the GreenLang platform.

Route Module Summary (28 endpoints):
    parsing_routes:      4 endpoints (POST parse, POST batch, POST detect, POST normalize)
    validation_routes:   5 endpoints (POST validate, POST batch, POST range, POST swap, POST duplicates)
    plausibility_routes: 5 endpoints (POST full, POST land-ocean, POST country, POST commodity, POST elevation)
    assessment_routes:   4 endpoints (POST assess, POST batch, GET result, POST precision)
    report_routes:       5 endpoints (POST compliance, POST summary, POST remediation, GET report, GET download)
    batch_routes:        9 endpoints (POST reverse, POST reverse/batch, POST country, POST datum/transform,
                                      POST datum/batch, GET datum/list, POST batch/submit, GET batch/{id},
                                      DELETE batch/{id})
    + health check:      1 endpoint (GET health)
    -------
    Total: 32 unique endpoints + health = 33 base

Auth & RBAC:
    All endpoints (except health) require JWT auth via SEC-001 and
    check eudr-gcv:* permissions via SEC-002.

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-007 GPS Coordinate Validator (GL-EUDR-GPS-007)
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from fastapi import APIRouter

from greenlang.agents.eudr.gps_coordinate_validator.api.parsing_routes import (
    router as parsing_router,
)
from greenlang.agents.eudr.gps_coordinate_validator.api.validation_routes import (
    router as validation_router,
)
from greenlang.agents.eudr.gps_coordinate_validator.api.plausibility_routes import (
    router as plausibility_router,
)
from greenlang.agents.eudr.gps_coordinate_validator.api.assessment_routes import (
    router as assessment_router,
)
from greenlang.agents.eudr.gps_coordinate_validator.api.report_routes import (
    router as report_router,
)
from greenlang.agents.eudr.gps_coordinate_validator.api.batch_routes import (
    router as batch_router,
)
from greenlang.agents.eudr.gps_coordinate_validator.api.schemas import (
    HealthResponseSchema,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Main router with /v1/eudr-gcv prefix
# ---------------------------------------------------------------------------

router = APIRouter(
    prefix="/v1/eudr-gcv",
    tags=["EUDR GPS Coordinate Validator"],
    responses={
        401: {"description": "Authentication required"},
        403: {"description": "Insufficient permissions"},
        429: {"description": "Rate limit exceeded"},
        500: {"description": "Internal server error"},
    },
)

# Include all sub-routers
router.include_router(
    parsing_router,
    prefix="/parse",
    tags=["Coordinate Parsing"],
)
router.include_router(
    validation_router,
    prefix="/validate",
    tags=["Coordinate Validation"],
)
router.include_router(
    plausibility_router,
    prefix="/plausibility",
    tags=["Plausibility Analysis"],
)
router.include_router(
    assessment_router,
    prefix="/assess",
    tags=["Accuracy Assessment"],
)
router.include_router(
    report_router,
    prefix="/report",
    tags=["Compliance Reporting"],
)
router.include_router(
    batch_router,
    tags=["Batch & Geocoding"],
)


# ---------------------------------------------------------------------------
# Health Check
# ---------------------------------------------------------------------------


@router.get(
    "/health",
    response_model=HealthResponseSchema,
    summary="Health check",
    description="Check EUDR GPS Coordinate Validator API health.",
    tags=["System"],
)
async def health_check() -> HealthResponseSchema:
    """Health check endpoint for load balancers and monitoring.

    Returns service identity, version, supported formats, and
    current timestamp. No authentication required.
    """
    return HealthResponseSchema(
        status="healthy",
        agent_id="GL-EUDR-GPS-007",
        agent_name="EUDR GPS Coordinate Validator",
        version="1.0.0",
        timestamp=datetime.now(timezone.utc).replace(microsecond=0),
    )


def get_router() -> APIRouter:
    """Return the GPS Coordinate Validator API router for mounting.

    Usage:
        >>> from greenlang.agents.eudr.gps_coordinate_validator.api import get_router
        >>> app.include_router(get_router(), prefix="/api")

    Returns:
        Configured APIRouter with all GPS validation endpoints.
    """
    return router


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "router",
    "get_router",
]
