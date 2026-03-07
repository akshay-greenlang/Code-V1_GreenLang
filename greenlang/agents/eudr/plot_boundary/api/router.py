# -*- coding: utf-8 -*-
"""
Main Router Registration - AGENT-EUDR-006 Plot Boundary Manager API

Aggregates all route modules under the ``/v1/eudr-pbm`` prefix and
provides ``get_router()`` for integration with the GreenLang platform.

Route Module Summary (29 endpoints):
    boundary_routes:     6 endpoints (POST create, GET get, PUT update,
                                      DELETE delete, POST batch, POST search)
    validation_routes:   4 endpoints (POST validate, POST validate/batch,
                                      POST repair, POST repair/batch)
    area_routes:         3 endpoints (POST calculate, POST batch, POST threshold)
    overlap_routes:      4 endpoints (POST detect, POST scan, GET records,
                                      POST resolve)
    version_routes:      4 endpoints (GET history, GET at, GET diff, GET lineage)
    export_routes:       8 endpoints (POST geojson, POST kml, POST shapefile,
                                      POST eudr-xml, POST batch, GET result,
                                      POST split, POST merge, GET genealogy)
    + health check:      1 endpoint  (GET health)
    -------
    Total: 29 unique endpoints + health = 30 base

Auth & RBAC:
    All endpoints (except health) require JWT auth via SEC-001 and
    check eudr-boundary:* permissions via SEC-002.

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-006 Plot Boundary Manager Agent (GL-EUDR-PBM-006)
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from fastapi import APIRouter

from greenlang.agents.eudr.plot_boundary.api.area_routes import (
    router as area_router,
)
from greenlang.agents.eudr.plot_boundary.api.boundary_routes import (
    router as boundary_router,
)
from greenlang.agents.eudr.plot_boundary.api.export_routes import (
    router as export_router,
)
from greenlang.agents.eudr.plot_boundary.api.overlap_routes import (
    router as overlap_router,
)
from greenlang.agents.eudr.plot_boundary.api.validation_routes import (
    router as validation_router,
)
from greenlang.agents.eudr.plot_boundary.api.version_routes import (
    router as version_router,
)
from greenlang.agents.eudr.plot_boundary.api.schemas import (
    HealthResponseSchema,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Main router with /v1/eudr-pbm prefix
# ---------------------------------------------------------------------------

router = APIRouter(
    prefix="/v1/eudr-pbm",
    tags=["EUDR Plot Boundary Manager"],
    responses={
        401: {"description": "Authentication required"},
        403: {"description": "Insufficient permissions"},
        429: {"description": "Rate limit exceeded"},
        500: {"description": "Internal server error"},
    },
)

# Include all sub-routers
router.include_router(
    boundary_router,
    tags=["Boundary CRUD"],
)
router.include_router(
    validation_router,
    tags=["Validation"],
)
router.include_router(
    area_router,
    tags=["Area Calculation"],
)
router.include_router(
    overlap_router,
    tags=["Overlap Detection"],
)
router.include_router(
    version_router,
    tags=["Version History"],
)
router.include_router(
    export_router,
    tags=["Export & Split/Merge"],
)


# ---------------------------------------------------------------------------
# Health Check
# ---------------------------------------------------------------------------


@router.get(
    "/health",
    response_model=HealthResponseSchema,
    summary="Health check",
    description="Check EUDR Plot Boundary Manager API health and engine status.",
    tags=["System"],
)
async def health_check() -> HealthResponseSchema:
    """Health check endpoint for load balancers and monitoring.

    Returns service identity, version, engine status, and current
    timestamp. No authentication required.

    Returns:
        HealthResponseSchema with service health information.
    """
    # Check engine availability
    engine_status = {}
    engine_names = [
        ("geometry_validator", "greenlang.agents.eudr.plot_boundary.api.dependencies", "get_geometry_validator"),
        ("area_calculator", "greenlang.agents.eudr.plot_boundary.api.dependencies", "get_area_calculator"),
        ("overlap_detector", "greenlang.agents.eudr.plot_boundary.api.dependencies", "get_overlap_detector"),
        ("boundary_versioner", "greenlang.agents.eudr.plot_boundary.api.dependencies", "get_boundary_versioner"),
        ("simplification_engine", "greenlang.agents.eudr.plot_boundary.api.dependencies", "get_simplification_engine"),
        ("split_merge_engine", "greenlang.agents.eudr.plot_boundary.api.dependencies", "get_split_merge_engine"),
        ("export_engine", "greenlang.agents.eudr.plot_boundary.api.dependencies", "get_export_engine"),
        ("compliance_reporter", "greenlang.agents.eudr.plot_boundary.api.dependencies", "get_compliance_reporter"),
    ]

    for name, _, _ in engine_names:
        try:
            from greenlang.agents.eudr.plot_boundary.api import dependencies as deps

            getter = getattr(deps, f"get_{name}", None)
            if getter:
                engine = getter()
                if hasattr(engine, "name") and engine.name == name.replace("_", " ").title():
                    engine_status[name] = "stub"
                else:
                    engine_status[name] = "healthy"
            else:
                engine_status[name] = "unknown"
        except Exception:
            engine_status[name] = "error"

    return HealthResponseSchema(
        status="healthy",
        agent="AGENT-EUDR-006",
        agent_name="Plot Boundary Manager",
        component="Plot Boundary Manager",
        version="1.0.0",
        engines=engine_status,
        boundary_count=0,
        timestamp=datetime.now(timezone.utc).replace(microsecond=0),
    )


def get_router() -> APIRouter:
    """Return the EUDR Plot Boundary Manager API router for mounting.

    Usage:
        >>> from greenlang.agents.eudr.plot_boundary.api import get_router
        >>> app.include_router(get_router(), prefix="/api")

    Returns:
        Configured APIRouter with all plot boundary management endpoints.
    """
    return router


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "router",
    "get_router",
]
