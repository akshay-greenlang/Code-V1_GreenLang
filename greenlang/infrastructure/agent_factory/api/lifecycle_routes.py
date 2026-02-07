# -*- coding: utf-8 -*-
"""
Lifecycle Routes - Agent lifecycle management endpoints.

Router prefix: /api/v1/factory/lifecycle

Endpoints:
    POST /agents/{key}/deploy  - Deploy an agent.
    POST /agents/{key}/rollback - Rollback to a previous version.
    POST /agents/{key}/drain   - Drain an agent (stop accepting new work).
    POST /agents/{key}/retire  - Retire an agent permanently.
    GET  /agents/{key}/health  - Get agent health status.
    POST /agents/{key}/restart - Restart an agent.
    GET  /agents/{key}/history - Get lifecycle transition history.

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field

from greenlang.infrastructure.agent_factory.api.operations_routes import (
    OperationResponse,
    operation_manager,
    _build_poll_url,
    _to_response,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/factory/lifecycle", tags=["Agent Lifecycle"])


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class DeployRequest(BaseModel):
    """Request body for deploying an agent."""

    version: str = Field(..., description="Version to deploy.")
    environment: str = Field("dev", description="Target environment: dev, staging, prod.")
    strategy: str = Field("rolling", description="Deployment strategy: rolling, canary, blue-green.")
    canary_percent: int = Field(5, ge=1, le=50, description="Canary traffic percentage.")
    config_overrides: Dict[str, Any] = Field(default_factory=dict, description="Per-deploy config overrides.")


class DeployResponse(BaseModel):
    """Deployment result."""

    deployment_id: str
    agent_key: str
    version: str
    environment: str
    strategy: str
    status: str
    started_at: str


class RollbackRequest(BaseModel):
    """Request body for rolling back an agent."""

    target_version: Optional[str] = Field(None, description="Version to roll back to (default: previous).")
    reason: str = Field("", description="Rollback reason for audit log.")


class RollbackResponse(BaseModel):
    """Rollback result."""

    rollback_id: str
    agent_key: str
    from_version: str
    to_version: str
    status: str
    reason: str
    completed_at: str


class DrainRequest(BaseModel):
    """Request body for draining an agent."""

    timeout_seconds: int = Field(60, ge=5, le=600, description="Max drain timeout.")
    reason: str = Field("", description="Drain reason.")


class DrainResponse(BaseModel):
    """Drain result."""

    agent_key: str
    status: str
    reason: str
    timeout_seconds: int
    started_at: str


class RetireRequest(BaseModel):
    """Request body for retiring an agent."""

    reason: str = Field("", description="Retirement reason.")
    archive: bool = Field(True, description="Archive agent data before retiring.")


class RetireResponse(BaseModel):
    """Retire result."""

    agent_key: str
    status: str
    reason: str
    archived: bool
    retired_at: str


class HealthResponse(BaseModel):
    """Agent health status."""

    agent_key: str
    status: str
    liveness: str
    readiness: str
    checks: Dict[str, Any] = Field(default_factory=dict)
    last_check_at: str


class RestartRequest(BaseModel):
    """Request body for restarting an agent."""

    reason: str = Field("", description="Restart reason.")
    force: bool = Field(False, description="Force immediate restart (skip graceful drain).")


class RestartResponse(BaseModel):
    """Restart result."""

    agent_key: str
    status: str
    reason: str
    restarted_at: str


class LifecycleEvent(BaseModel):
    """Single lifecycle transition event."""

    from_state: str
    to_state: str
    reason: str
    actor: str
    timestamp: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class HistoryResponse(BaseModel):
    """Lifecycle transition history."""

    agent_key: str
    events: List[LifecycleEvent]
    total: int


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# In production these would query the lifecycle manager and state machine.
_lifecycle_store: Dict[str, List[Dict[str, Any]]] = {}


def _record_event(agent_key: str, event: Dict[str, Any]) -> None:
    """Append a lifecycle event to the in-memory store."""
    _lifecycle_store.setdefault(agent_key, []).append(event)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/agents/{key}/deploy", response_model=OperationResponse, status_code=202)
async def deploy_agent(key: str, body: DeployRequest, request: Request) -> OperationResponse:
    """Deploy an agent to the target environment (async 202 pattern).

    Creates an async deploy operation and returns immediately with a
    poll URL. The actual deployment runs as a background task.
    """
    now = _now_iso()

    _record_event(key, {
        "from_state": "validated",
        "to_state": "deploying",
        "reason": f"deploy v{body.version} to {body.environment}",
        "actor": "api",
        "timestamp": now,
        "metadata": {"strategy": body.strategy, "canary_pct": body.canary_percent},
    })

    try:
        record = operation_manager.create_operation(
            operation_type="deploy",
            agent_key=key,
            params={
                "version": body.version,
                "environment": body.environment,
                "strategy": body.strategy,
                "canary_percent": body.canary_percent,
                "config_overrides": body.config_overrides,
            },
            idempotency_key=None,
            created_by=request.headers.get("X-User-Id", "api"),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if record["status"] == "pending":
        operation_manager.dispatch(record["operation_id"])

    logger.info("Deploy delegated to operation %s: %s v%s -> %s (%s)",
                record["operation_id"], key, body.version, body.environment, body.strategy)

    poll_url = _build_poll_url(request, record["operation_id"])
    return _to_response(record, poll_url)


@router.post("/agents/{key}/rollback", response_model=OperationResponse, status_code=202)
async def rollback_agent(key: str, body: RollbackRequest, request: Request) -> OperationResponse:
    """Rollback an agent to a previous version (async 202 pattern).

    Creates an async rollback operation and returns immediately with a
    poll URL. The actual rollback runs as a background task.
    """
    to_version = body.target_version or "previous"
    now = _now_iso()

    _record_event(key, {
        "from_state": "running",
        "to_state": "deploying",
        "reason": body.reason or f"rollback to {to_version}",
        "actor": "api",
        "timestamp": now,
        "metadata": {"rollback_to": to_version},
    })

    try:
        record = operation_manager.create_operation(
            operation_type="rollback",
            agent_key=key,
            params={
                "target_version": to_version,
                "reason": body.reason,
            },
            idempotency_key=None,
            created_by=request.headers.get("X-User-Id", "api"),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if record["status"] == "pending":
        operation_manager.dispatch(record["operation_id"])

    logger.info("Rollback delegated to operation %s: %s -> %s",
                record["operation_id"], key, to_version)

    poll_url = _build_poll_url(request, record["operation_id"])
    return _to_response(record, poll_url)


@router.post("/agents/{key}/drain", response_model=DrainResponse)
async def drain_agent(key: str, body: DrainRequest) -> DrainResponse:
    """Drain an agent -- stop accepting new work and finish in-flight tasks."""
    now = _now_iso()

    _record_event(key, {
        "from_state": "running",
        "to_state": "draining",
        "reason": body.reason or "drain via API",
        "actor": "api",
        "timestamp": now,
        "metadata": {"timeout_s": body.timeout_seconds},
    })

    logger.info("Drain started: %s (timeout=%ds)", key, body.timeout_seconds)

    return DrainResponse(
        agent_key=key,
        status="draining",
        reason=body.reason,
        timeout_seconds=body.timeout_seconds,
        started_at=now,
    )


@router.post("/agents/{key}/retire", response_model=RetireResponse)
async def retire_agent(key: str, body: RetireRequest) -> RetireResponse:
    """Permanently retire an agent."""
    now = _now_iso()

    _record_event(key, {
        "from_state": "draining",
        "to_state": "retired",
        "reason": body.reason or "retire via API",
        "actor": "api",
        "timestamp": now,
        "metadata": {"archived": body.archive},
    })

    logger.info("Agent retired: %s (archived=%s)", key, body.archive)

    return RetireResponse(
        agent_key=key,
        status="retired",
        reason=body.reason,
        archived=body.archive,
        retired_at=now,
    )


@router.get("/agents/{key}/health", response_model=HealthResponse)
async def get_agent_health(key: str) -> HealthResponse:
    """Get the health status of an agent."""
    now = _now_iso()

    # In production, query HealthCheckRegistry
    return HealthResponse(
        agent_key=key,
        status="healthy",
        liveness="pass",
        readiness="pass",
        checks={
            "database": {"status": "pass", "latency_ms": 2.1},
            "redis": {"status": "pass", "latency_ms": 0.8},
        },
        last_check_at=now,
    )


@router.post("/agents/{key}/restart", response_model=RestartResponse)
async def restart_agent(key: str, body: RestartRequest) -> RestartResponse:
    """Restart an agent instance."""
    now = _now_iso()

    _record_event(key, {
        "from_state": "running",
        "to_state": "warming_up",
        "reason": body.reason or "restart via API",
        "actor": "api",
        "timestamp": now,
        "metadata": {"force": body.force},
    })

    logger.info("Agent restart: %s (force=%s)", key, body.force)

    return RestartResponse(
        agent_key=key,
        status="restarting",
        reason=body.reason,
        restarted_at=now,
    )


@router.get("/agents/{key}/history", response_model=HistoryResponse)
async def get_lifecycle_history(
    key: str,
    limit: int = Query(50, ge=1, le=500, description="Max events to return."),
) -> HistoryResponse:
    """Get lifecycle transition history for an agent."""
    events = _lifecycle_store.get(key, [])
    recent = events[-limit:]

    return HistoryResponse(
        agent_key=key,
        events=[
            LifecycleEvent(
                from_state=e.get("from_state", ""),
                to_state=e.get("to_state", ""),
                reason=e.get("reason", ""),
                actor=e.get("actor", ""),
                timestamp=e.get("timestamp", ""),
                metadata=e.get("metadata", {}),
            )
            for e in recent
        ],
        total=len(events),
    )
