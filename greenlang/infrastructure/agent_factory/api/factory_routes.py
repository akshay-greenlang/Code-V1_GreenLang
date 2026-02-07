# -*- coding: utf-8 -*-
"""
Factory Routes - Core agent CRUD and execution endpoints.

Router prefix: /api/v1/factory

Endpoints:
    POST   /agents              - Register a new agent.
    GET    /agents              - List agents (paginated, filterable).
    GET    /agents/{key}        - Get agent details.
    PUT    /agents/{key}        - Update agent configuration.
    DELETE /agents/{key}        - Deregister an agent.
    POST   /agents/{key}/execute - Trigger agent execution.
    GET    /agents/{key}/metrics - Get agent metrics.
    POST   /agents/batch-execute - Execute multiple agents.

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/factory", tags=["Agent Factory"])


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class AgentCreateRequest(BaseModel):
    """Request body for registering a new agent."""

    agent_key: str = Field(..., min_length=3, max_length=64, description="Unique agent key.")
    version: str = Field("0.1.0", description="Semantic version.")
    agent_type: str = Field("deterministic", description="Agent type: deterministic, reasoning, insight.")
    description: str = Field("", description="Agent description.")
    entry_point: str = Field("agent.py", description="Entry-point module.")
    config: Dict[str, Any] = Field(default_factory=dict, description="Agent configuration.")


class AgentUpdateRequest(BaseModel):
    """Request body for updating an agent."""

    version: Optional[str] = Field(None, description="New version.")
    description: Optional[str] = Field(None, description="Updated description.")
    config: Optional[Dict[str, Any]] = Field(None, description="Updated configuration.")


class AgentResponse(BaseModel):
    """Standard agent response."""

    agent_key: str
    version: str
    agent_type: str
    description: str
    status: str
    created_at: str
    updated_at: str
    config: Dict[str, Any] = Field(default_factory=dict)


class AgentListResponse(BaseModel):
    """Paginated list of agents."""

    agents: List[AgentResponse]
    total: int
    page: int
    page_size: int


class ExecuteRequest(BaseModel):
    """Request body for triggering agent execution."""

    input_data: Dict[str, Any] = Field(..., description="Input payload for the agent.")
    priority: int = Field(0, ge=0, le=10, description="Execution priority (0 = highest).")
    timeout_seconds: int = Field(30, ge=1, le=300, description="Execution timeout.")
    correlation_id: Optional[str] = Field(None, description="Optional correlation ID.")


class ExecuteResponse(BaseModel):
    """Response from agent execution."""

    task_id: str
    agent_key: str
    status: str
    result: Optional[Dict[str, Any]] = None
    provenance_hash: Optional[str] = None
    duration_ms: Optional[float] = None
    correlation_id: str


class BatchExecuteRequest(BaseModel):
    """Request body for batch agent execution."""

    tasks: List[ExecuteRequest]
    agent_keys: List[str] = Field(..., min_length=1, description="Agent keys to execute.")


class BatchExecuteResponse(BaseModel):
    """Batch execution response."""

    batch_id: str
    tasks: List[ExecuteResponse]
    total: int
    succeeded: int
    failed: int


class AgentMetricsResponse(BaseModel):
    """Agent metrics response."""

    agent_key: str
    execution_count: int
    error_count: int
    success_rate: float
    avg_duration_ms: float
    p95_duration_ms: float
    p99_duration_ms: float
    total_cost_usd: float
    queue_depth: int
    active_instances: int


# ---------------------------------------------------------------------------
# In-memory store (replaced by DB in production)
# ---------------------------------------------------------------------------

_agent_store: Dict[str, Dict[str, Any]] = {}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/agents", response_model=AgentResponse, status_code=201)
async def create_agent(body: AgentCreateRequest) -> AgentResponse:
    """Register a new agent in the factory."""
    if body.agent_key in _agent_store:
        raise HTTPException(status_code=409, detail=f"Agent '{body.agent_key}' already registered.")

    now = _now_iso()
    record = {
        "agent_key": body.agent_key,
        "version": body.version,
        "agent_type": body.agent_type,
        "description": body.description,
        "status": "created",
        "entry_point": body.entry_point,
        "config": body.config,
        "created_at": now,
        "updated_at": now,
    }
    _agent_store[body.agent_key] = record
    logger.info("Agent registered: %s v%s", body.agent_key, body.version)
    return AgentResponse(**record)


@router.get("/agents", response_model=AgentListResponse)
async def list_agents(
    page: int = Query(1, ge=1, description="Page number."),
    page_size: int = Query(20, ge=1, le=100, description="Items per page."),
    status: Optional[str] = Query(None, description="Filter by status."),
    agent_type: Optional[str] = Query(None, description="Filter by agent type."),
) -> AgentListResponse:
    """List all registered agents with pagination and filtering."""
    agents = list(_agent_store.values())
    if status:
        agents = [a for a in agents if a["status"] == status]
    if agent_type:
        agents = [a for a in agents if a["agent_type"] == agent_type]

    total = len(agents)
    start = (page - 1) * page_size
    page_items = agents[start : start + page_size]

    return AgentListResponse(
        agents=[AgentResponse(**a) for a in page_items],
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get("/agents/{key}", response_model=AgentResponse)
async def get_agent(key: str) -> AgentResponse:
    """Get details for a specific agent."""
    record = _agent_store.get(key)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Agent '{key}' not found.")
    return AgentResponse(**record)


@router.put("/agents/{key}", response_model=AgentResponse)
async def update_agent(key: str, body: AgentUpdateRequest) -> AgentResponse:
    """Update an agent's configuration."""
    record = _agent_store.get(key)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Agent '{key}' not found.")

    if body.version is not None:
        record["version"] = body.version
    if body.description is not None:
        record["description"] = body.description
    if body.config is not None:
        record["config"] = body.config
    record["updated_at"] = _now_iso()

    logger.info("Agent updated: %s", key)
    return AgentResponse(**record)


@router.delete("/agents/{key}", status_code=204)
async def delete_agent(key: str) -> None:
    """Deregister an agent."""
    if key not in _agent_store:
        raise HTTPException(status_code=404, detail=f"Agent '{key}' not found.")
    del _agent_store[key]
    logger.info("Agent deregistered: %s", key)


@router.post("/agents/{key}/execute", response_model=ExecuteResponse)
async def execute_agent(key: str, body: ExecuteRequest) -> ExecuteResponse:
    """Trigger an agent execution."""
    record = _agent_store.get(key)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Agent '{key}' not found.")

    task_id = uuid.uuid4().hex[:16]
    correlation_id = body.correlation_id or uuid.uuid4().hex

    logger.info("Execution triggered for %s (task=%s)", key, task_id)

    return ExecuteResponse(
        task_id=task_id,
        agent_key=key,
        status="queued",
        correlation_id=correlation_id,
    )


@router.get("/agents/{key}/metrics", response_model=AgentMetricsResponse)
async def get_agent_metrics(key: str) -> AgentMetricsResponse:
    """Get execution metrics for an agent."""
    if key not in _agent_store:
        raise HTTPException(status_code=404, detail=f"Agent '{key}' not found.")

    # In production, fetch from AgentMetricsCollector
    return AgentMetricsResponse(
        agent_key=key,
        execution_count=0,
        error_count=0,
        success_rate=0.0,
        avg_duration_ms=0.0,
        p95_duration_ms=0.0,
        p99_duration_ms=0.0,
        total_cost_usd=0.0,
        queue_depth=0,
        active_instances=0,
    )


@router.post("/agents/batch-execute", response_model=BatchExecuteResponse)
async def batch_execute(body: BatchExecuteRequest) -> BatchExecuteResponse:
    """Execute multiple agents in batch."""
    batch_id = uuid.uuid4().hex[:16]
    results: List[ExecuteResponse] = []
    succeeded = 0
    failed = 0

    for agent_key, task in zip(body.agent_keys, body.tasks):
        if agent_key not in _agent_store:
            results.append(ExecuteResponse(
                task_id=uuid.uuid4().hex[:16],
                agent_key=agent_key,
                status="failed",
                correlation_id=task.correlation_id or uuid.uuid4().hex,
            ))
            failed += 1
        else:
            task_id = uuid.uuid4().hex[:16]
            results.append(ExecuteResponse(
                task_id=task_id,
                agent_key=agent_key,
                status="queued",
                correlation_id=task.correlation_id or uuid.uuid4().hex,
            ))
            succeeded += 1

    return BatchExecuteResponse(
        batch_id=batch_id,
        tasks=results,
        total=len(results),
        succeeded=succeeded,
        failed=failed,
    )


# SEC-001: Apply authentication and permission protection
try:
    from greenlang.infrastructure.auth_service.route_protector import protect_router

    protect_router(router)
except ImportError:
    pass  # auth_service not available
