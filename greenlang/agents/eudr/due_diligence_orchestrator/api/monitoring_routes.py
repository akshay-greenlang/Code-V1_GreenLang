# -*- coding: utf-8 -*-
"""
Monitoring Routes - AGENT-EUDR-026 Due Diligence Orchestrator API

Health check, metrics, version, circuit breaker management, and dead
letter queue endpoints for operational monitoring of the orchestrator.

Endpoints (6):
    GET  /health                              - Service health check
    GET  /metrics                             - Prometheus metrics summary
    GET  /version                             - Agent version info
    GET  /circuit-breakers                    - Circuit breaker states
    POST /circuit-breakers/{agent_id}/reset   - Reset a circuit breaker
    GET  /dead-letter-queue                   - Dead letter queue entries

RBAC Permissions:
    eudr-ddo:workflows:read        - Health, metrics, version
    eudr-ddo:circuit-breakers:read - View circuit breaker states
    eudr-ddo:circuit-breakers:manage - Reset circuit breakers
    eudr-ddo:dlq:read              - View dead letter queue

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-026 Due Diligence Orchestrator (GL-EUDR-DDO-026)
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status

from greenlang.agents.eudr.due_diligence_orchestrator.api.dependencies import (
    AuthUser,
    ErrorResponse,
    PaginationParams,
    get_ddo_service,
    get_pagination,
    rate_limit_standard,
    require_permission,
)
from greenlang.agents.eudr.due_diligence_orchestrator.models import (
    VERSION,
    CircuitBreakerState,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Monitoring & Operations"])


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


@router.get(
    "/health",
    summary="Service health check",
    description=(
        "Returns the health status of the Due Diligence Orchestrator service "
        "including connectivity to PostgreSQL, Redis, and S3 dependencies."
    ),
    response_model=Dict[str, Any],
    responses={503: {"model": ErrorResponse}},
)
async def health_check(
    _rate: None = Depends(rate_limit_standard),
) -> Dict[str, Any]:
    """Check orchestrator service health.

    Returns:
        Health status with component availability.
    """
    try:
        service = get_ddo_service()
        health = await service.health_check()
        return {
            "status": "healthy",
            "agent": "GL-EUDR-DDO-026",
            "version": VERSION,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "components": health,
        }
    except Exception as exc:
        logger.error("Health check failed: %s", exc)
        return {
            "status": "degraded",
            "agent": "GL-EUDR-DDO-026",
            "version": VERSION,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "error": str(exc),
        }


# ---------------------------------------------------------------------------
# Metrics summary
# ---------------------------------------------------------------------------


@router.get(
    "/metrics",
    summary="Prometheus metrics summary",
    description=(
        "Returns a JSON summary of the 20 Prometheus metrics tracked by the "
        "orchestrator, including workflow counts, agent execution stats, "
        "quality gate evaluations, and circuit breaker state."
    ),
    response_model=Dict[str, Any],
)
async def metrics_summary(
    user: AuthUser = Depends(require_permission("eudr-ddo:workflows:read")),
    _rate: None = Depends(rate_limit_standard),
) -> Dict[str, Any]:
    """Get orchestrator metrics summary.

    Returns:
        JSON summary of Prometheus metrics.
    """
    try:
        service = get_ddo_service()
        metrics = await service.get_metrics()
        return {
            "agent": "GL-EUDR-DDO-026",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metrics": metrics,
        }
    except Exception as exc:
        logger.error("Metrics retrieval failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve metrics: {exc}",
        )


# ---------------------------------------------------------------------------
# Version info
# ---------------------------------------------------------------------------


@router.get(
    "/version",
    summary="Agent version information",
    description="Returns the agent version, build date, and supported features.",
    response_model=Dict[str, Any],
)
async def version_info(
    _rate: None = Depends(rate_limit_standard),
) -> Dict[str, Any]:
    """Get orchestrator version information.

    Returns:
        Version, build date, and feature list.
    """
    return {
        "agent_id": "GL-EUDR-DDO-026",
        "agent_name": "Due Diligence Orchestrator",
        "version": VERSION,
        "prd": "PRD-AGENT-EUDR-026",
        "regulation": "EU 2023/1115 (EUDR)",
        "articles": ["4", "8", "9", "10", "11", "12", "13", "31"],
        "features": [
            "workflow_definition_engine",
            "information_gathering_coordinator",
            "risk_assessment_coordinator",
            "risk_mitigation_coordinator",
            "quality_gate_engine",
            "workflow_state_manager",
            "parallel_execution_engine",
            "error_recovery_manager",
            "due_diligence_package_generator",
        ],
        "supported_commodities": [
            "cattle", "cocoa", "coffee", "palm_oil",
            "rubber", "soya", "wood",
        ],
        "workflow_types": ["standard", "simplified", "custom"],
        "upstream_agents": 25,
        "quality_gates": 3,
    }


# ---------------------------------------------------------------------------
# Circuit breaker management
# ---------------------------------------------------------------------------


@router.get(
    "/circuit-breakers",
    summary="Get circuit breaker states",
    description=(
        "Returns the current circuit breaker state for each of the 25 "
        "upstream EUDR agents. States: CLOSED (normal), OPEN (rejecting), "
        "HALF_OPEN (probing recovery)."
    ),
    response_model=Dict[str, Any],
)
async def get_circuit_breakers(
    user: AuthUser = Depends(require_permission("eudr-ddo:circuit-breakers:read")),
    _rate: None = Depends(rate_limit_standard),
) -> Dict[str, Any]:
    """Get all circuit breaker states.

    Returns:
        Circuit breaker states keyed by agent ID.
    """
    try:
        service = get_ddo_service()
        states = service._error_manager.get_all_circuit_breaker_states()
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_agents": 25,
            "circuit_breakers": {
                agent_id: {
                    "state": record.state.value if hasattr(record, "state") else str(record),
                    "failure_count": getattr(record, "failure_count", 0),
                    "last_failure_at": (
                        record.last_failure_at.isoformat()
                        if hasattr(record, "last_failure_at") and record.last_failure_at
                        else None
                    ),
                }
                for agent_id, record in states.items()
            },
            "summary": {
                "closed": sum(
                    1 for r in states.values()
                    if (hasattr(r, "state") and r.state == CircuitBreakerState.CLOSED)
                    or r == CircuitBreakerState.CLOSED
                ),
                "open": sum(
                    1 for r in states.values()
                    if (hasattr(r, "state") and r.state == CircuitBreakerState.OPEN)
                    or r == CircuitBreakerState.OPEN
                ),
                "half_open": sum(
                    1 for r in states.values()
                    if (hasattr(r, "state") and r.state == CircuitBreakerState.HALF_OPEN)
                    or r == CircuitBreakerState.HALF_OPEN
                ),
            },
        }
    except Exception as exc:
        logger.error("Circuit breaker query failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve circuit breaker states: {exc}",
        )


@router.post(
    "/circuit-breakers/{agent_id}/reset",
    summary="Reset a circuit breaker",
    description=(
        "Manually reset an OPEN or HALF_OPEN circuit breaker to CLOSED. "
        "Use when an upstream agent has recovered and you want to force "
        "the orchestrator to resume calling it."
    ),
    response_model=Dict[str, Any],
    responses={404: {"model": ErrorResponse}},
)
async def reset_circuit_breaker(
    agent_id: str,
    user: AuthUser = Depends(require_permission("eudr-ddo:circuit-breakers:manage")),
    _rate: None = Depends(rate_limit_standard),
) -> Dict[str, Any]:
    """Reset a circuit breaker to CLOSED state.

    Args:
        agent_id: Agent whose circuit breaker to reset.

    Returns:
        Confirmation with new circuit breaker state.
    """
    try:
        service = get_ddo_service()
        service._error_manager.reset_circuit_breaker(agent_id)
        return {
            "agent_id": agent_id,
            "new_state": "closed",
            "reset_by": user.user_id,
            "reset_at": datetime.now(timezone.utc).isoformat(),
        }
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Agent not found: {agent_id}",
        )
    except Exception as exc:
        logger.error("Circuit breaker reset failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to reset circuit breaker: {exc}",
        )


# ---------------------------------------------------------------------------
# Dead letter queue
# ---------------------------------------------------------------------------


@router.get(
    "/dead-letter-queue",
    summary="List dead letter queue entries",
    description=(
        "Returns entries from the dead letter queue containing permanently "
        "failed agent invocations that require manual investigation."
    ),
    response_model=Dict[str, Any],
)
async def list_dead_letter_queue(
    user: AuthUser = Depends(require_permission("eudr-ddo:dlq:read")),
    resolved: Optional[bool] = Query(
        None, description="Filter by resolution status"
    ),
    agent_id: Optional[str] = Query(
        None, description="Filter by agent ID"
    ),
    pagination: PaginationParams = Depends(get_pagination),
    _rate: None = Depends(rate_limit_standard),
) -> Dict[str, Any]:
    """List dead letter queue entries.

    Args:
        resolved: Filter by resolution status.
        agent_id: Filter by agent ID.
        pagination: Pagination parameters.

    Returns:
        Paginated dead letter queue entries.
    """
    try:
        service = get_ddo_service()
        entries = service._error_manager.get_dead_letter_entries()

        # Apply filters
        if resolved is not None:
            entries = [e for e in entries if e.resolved == resolved]
        if agent_id is not None:
            entries = [e for e in entries if e.agent_id == agent_id]

        total = len(entries)
        start = pagination.offset
        end = start + pagination.limit
        page = entries[start:end]

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total": total,
            "offset": pagination.offset,
            "limit": pagination.limit,
            "entries": [
                {
                    "dlq_id": e.dlq_id,
                    "workflow_id": e.workflow_id,
                    "agent_id": e.agent_id,
                    "error_type": e.error_type,
                    "error_message": e.error_message,
                    "attempt_count": e.attempt_count,
                    "resolved": e.resolved,
                    "created_at": e.created_at.isoformat() if hasattr(e, "created_at") else None,
                }
                for e in page
            ],
        }
    except Exception as exc:
        logger.error("DLQ query failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve dead letter queue: {exc}",
        )
