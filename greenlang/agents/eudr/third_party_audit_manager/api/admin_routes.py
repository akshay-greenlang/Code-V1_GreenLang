# -*- coding: utf-8 -*-
"""
Admin Routes - AGENT-EUDR-024 Third-Party Audit Manager API

Endpoints for health check and service statistics.

Endpoints (2):
    GET /health   - Health check (public)
    GET /stats    - Service statistics

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-024, Administration
"""

from __future__ import annotations

import hashlib
import logging
import time
from decimal import Decimal
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status

from greenlang.agents.eudr.third_party_audit_manager.api.dependencies import (
    AuthUser,
    get_analytics_engine,
    rate_limit_standard,
    require_permission,
)
from greenlang.agents.eudr.third_party_audit_manager.api.schemas import (
    ErrorResponse,
    HealthResponse,
    ProvenanceInfo,
    StatsResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Admin"])


def _compute_provenance(input_data: Any, output_data: Any) -> str:
    """Compute SHA-256 provenance hash for audit trail."""
    data_str = f"{input_data}{output_data}"
    return hashlib.sha256(data_str.encode("utf-8")).hexdigest()


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Returns service health status for AGENT-EUDR-024.",
    tags=["health"],
)
async def health_check() -> HealthResponse:
    """Return health status for the Third-Party Audit Manager agent.

    Returns:
        HealthResponse with status, agent ID, and component name.
    """
    return HealthResponse(
        status="healthy",
        agent_id="GL-EUDR-TAM-024",
        agent="EUDR-024",
        component="third-party-audit-manager",
        version="1.0.0",
    )


@router.get(
    "/stats",
    response_model=StatsResponse,
    summary="Service statistics",
    description="Retrieve service-level statistics for the audit manager.",
    responses={
        200: {"description": "Statistics retrieved"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
    },
)
async def get_stats(
    user: AuthUser = Depends(require_permission("eudr-tam:analytics:read")),
    _rate: None = Depends(rate_limit_standard),
) -> StatsResponse:
    """Retrieve service-level statistics.

    Args:
        user: Authenticated user with analytics:read permission.

    Returns:
        StatsResponse with service metrics.
    """
    start = time.monotonic()
    try:
        engine = get_analytics_engine()
        result = engine.get_stats()
        elapsed_ms = Decimal(str((time.monotonic() - start) * 1000))
        return StatsResponse(
            total_audits=result.get("total_audits", 0),
            total_auditors=result.get("total_auditors", 0),
            total_ncs=result.get("total_ncs", 0),
            total_cars=result.get("total_cars", 0),
            total_certificates=result.get("total_certificates", 0),
            total_reports=result.get("total_reports", 0),
            total_authority_interactions=result.get("total_authority_interactions", 0),
            provenance=ProvenanceInfo(
                provenance_hash=_compute_provenance("stats", ""),
                processing_time_ms=elapsed_ms.quantize(Decimal("0.01")),
            ),
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Stats retrieval failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve service statistics",
        )
