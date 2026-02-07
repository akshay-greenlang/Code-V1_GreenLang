# -*- coding: utf-8 -*-
"""
Queue Routes - Execution queue management endpoints.

Router prefix: /api/v1/factory/queue

Endpoints:
    GET  /status          - Queue status (depth, workers, throughput).
    GET  /tasks           - List tasks (pagination, filtering).
    GET  /tasks/{id}      - Get task details.
    POST /tasks/{id}/retry  - Retry a failed task.
    POST /tasks/{id}/cancel - Cancel a queued task.
    GET  /dlq             - List dead-letter queue items.
    POST /dlq/{id}/reprocess - Reprocess a DLQ item.

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/factory/queue", tags=["Execution Queue"])


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class QueueStatusResponse(BaseModel):
    """Aggregate queue status."""

    total_depth: int = Field(description="Total tasks currently in queue.")
    active_workers: int = Field(description="Workers processing tasks.")
    idle_workers: int = Field(description="Workers waiting for tasks.")
    throughput_per_min: float = Field(description="Tasks completed per minute (last 5 min).")
    avg_wait_ms: float = Field(description="Average wait time in ms.")
    oldest_task_age_s: float = Field(description="Age of oldest queued task in seconds.")
    dlq_depth: int = Field(description="Items in the dead-letter queue.")


class TaskStatus(BaseModel):
    """Status of a single queued task."""

    task_id: str
    agent_key: str
    status: str = Field(description="queued, running, completed, failed, cancelled.")
    priority: int
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    wait_time_ms: Optional[float] = None
    duration_ms: Optional[float] = None
    error: Optional[str] = None
    correlation_id: str = ""


class TaskListResponse(BaseModel):
    """Paginated task list."""

    tasks: List[TaskStatus]
    total: int
    page: int
    page_size: int


class TaskRetryRequest(BaseModel):
    """Request body for retrying a task."""

    priority: Optional[int] = Field(None, ge=0, le=10, description="Override priority for retry.")


class TaskRetryResponse(BaseModel):
    """Retry result."""

    original_task_id: str
    new_task_id: str
    agent_key: str
    status: str
    retried_at: str


class TaskCancelResponse(BaseModel):
    """Cancel result."""

    task_id: str
    status: str
    cancelled_at: str


class DLQItem(BaseModel):
    """Dead-letter queue item."""

    dlq_id: str
    original_task_id: str
    agent_key: str
    error: str
    failed_at: str
    retry_count: int
    input_data: Dict[str, Any] = Field(default_factory=dict)


class DLQListResponse(BaseModel):
    """Paginated DLQ list."""

    items: List[DLQItem]
    total: int
    page: int
    page_size: int


class DLQReprocessResponse(BaseModel):
    """DLQ reprocess result."""

    dlq_id: str
    new_task_id: str
    status: str
    reprocessed_at: str


# ---------------------------------------------------------------------------
# In-memory stores (replaced by Redis/DB in production)
# ---------------------------------------------------------------------------

_task_store: Dict[str, Dict[str, Any]] = {}
_dlq_store: Dict[str, Dict[str, Any]] = {}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _seed_demo_data() -> None:
    """Populate demo data if empty."""
    if _task_store:
        return

    now = _now_iso()
    statuses = ["queued", "running", "completed", "failed", "completed"]
    agents = ["carbon-calc", "eudr-compliance", "csrd-disclosure", "scope3-mapper", "sbti-validator"]
    for i, (s, a) in enumerate(zip(statuses, agents)):
        tid = uuid.uuid4().hex[:16]
        _task_store[tid] = {
            "task_id": tid,
            "agent_key": a,
            "status": s,
            "priority": i % 3,
            "created_at": now,
            "started_at": now if s != "queued" else None,
            "completed_at": now if s in ("completed", "failed") else None,
            "wait_time_ms": 120.5 + i * 30,
            "duration_ms": 450.0 + i * 100 if s == "completed" else None,
            "error": "ERP connection timeout" if s == "failed" else None,
            "correlation_id": uuid.uuid4().hex,
        }

    # One DLQ item
    dlq_id = uuid.uuid4().hex[:16]
    _dlq_store[dlq_id] = {
        "dlq_id": dlq_id,
        "original_task_id": list(_task_store.keys())[-1],
        "agent_key": "scope3-mapper",
        "error": "Max retries exceeded: ERP connection timeout",
        "failed_at": now,
        "retry_count": 3,
        "input_data": {"facility_id": "FAC-001"},
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/status", response_model=QueueStatusResponse)
async def queue_status() -> QueueStatusResponse:
    """Get aggregate queue status."""
    _seed_demo_data()

    queued = sum(1 for t in _task_store.values() if t["status"] == "queued")
    running = sum(1 for t in _task_store.values() if t["status"] == "running")

    return QueueStatusResponse(
        total_depth=queued,
        active_workers=running,
        idle_workers=max(0, 4 - running),
        throughput_per_min=42.5,
        avg_wait_ms=155.2,
        oldest_task_age_s=12.3,
        dlq_depth=len(_dlq_store),
    )


@router.get("/tasks", response_model=TaskListResponse)
async def list_tasks(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    status: Optional[str] = Query(None, description="Filter by status."),
    agent_key: Optional[str] = Query(None, description="Filter by agent key."),
) -> TaskListResponse:
    """List tasks with optional filtering and pagination."""
    _seed_demo_data()

    tasks = list(_task_store.values())
    if status:
        tasks = [t for t in tasks if t["status"] == status]
    if agent_key:
        tasks = [t for t in tasks if t["agent_key"] == agent_key]

    total = len(tasks)
    start = (page - 1) * page_size
    page_items = tasks[start : start + page_size]

    return TaskListResponse(
        tasks=[TaskStatus(**t) for t in page_items],
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get("/tasks/{task_id}", response_model=TaskStatus)
async def get_task(task_id: str) -> TaskStatus:
    """Get details for a specific task."""
    _seed_demo_data()

    task = _task_store.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found.")
    return TaskStatus(**task)


@router.post("/tasks/{task_id}/retry", response_model=TaskRetryResponse)
async def retry_task(task_id: str, body: TaskRetryRequest | None = None) -> TaskRetryResponse:
    """Retry a failed task."""
    _seed_demo_data()

    task = _task_store.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found.")
    if task["status"] != "failed":
        raise HTTPException(status_code=400, detail="Only failed tasks can be retried.")

    new_id = uuid.uuid4().hex[:16]
    priority = body.priority if body and body.priority is not None else task["priority"]
    now = _now_iso()

    _task_store[new_id] = {
        **task,
        "task_id": new_id,
        "status": "queued",
        "priority": priority,
        "created_at": now,
        "started_at": None,
        "completed_at": None,
        "error": None,
    }

    logger.info("Task retried: %s -> %s (agent=%s)", task_id, new_id, task["agent_key"])

    return TaskRetryResponse(
        original_task_id=task_id,
        new_task_id=new_id,
        agent_key=task["agent_key"],
        status="queued",
        retried_at=now,
    )


@router.post("/tasks/{task_id}/cancel", response_model=TaskCancelResponse)
async def cancel_task(task_id: str) -> TaskCancelResponse:
    """Cancel a queued task."""
    _seed_demo_data()

    task = _task_store.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found.")
    if task["status"] not in ("queued",):
        raise HTTPException(status_code=400, detail="Only queued tasks can be cancelled.")

    task["status"] = "cancelled"
    now = _now_iso()
    task["completed_at"] = now

    logger.info("Task cancelled: %s", task_id)

    return TaskCancelResponse(task_id=task_id, status="cancelled", cancelled_at=now)


@router.get("/dlq", response_model=DLQListResponse)
async def list_dlq(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
) -> DLQListResponse:
    """List items in the dead-letter queue."""
    _seed_demo_data()

    items = list(_dlq_store.values())
    total = len(items)
    start = (page - 1) * page_size
    page_items = items[start : start + page_size]

    return DLQListResponse(
        items=[DLQItem(**i) for i in page_items],
        total=total,
        page=page,
        page_size=page_size,
    )


@router.post("/dlq/{dlq_id}/reprocess", response_model=DLQReprocessResponse)
async def reprocess_dlq(dlq_id: str) -> DLQReprocessResponse:
    """Reprocess a dead-letter queue item by creating a new task."""
    _seed_demo_data()

    item = _dlq_store.get(dlq_id)
    if item is None:
        raise HTTPException(status_code=404, detail=f"DLQ item '{dlq_id}' not found.")

    new_task_id = uuid.uuid4().hex[:16]
    now = _now_iso()

    _task_store[new_task_id] = {
        "task_id": new_task_id,
        "agent_key": item["agent_key"],
        "status": "queued",
        "priority": 0,
        "created_at": now,
        "started_at": None,
        "completed_at": None,
        "wait_time_ms": None,
        "duration_ms": None,
        "error": None,
        "correlation_id": uuid.uuid4().hex,
    }

    # Remove from DLQ
    del _dlq_store[dlq_id]

    logger.info("DLQ reprocessed: %s -> task %s", dlq_id, new_task_id)

    return DLQReprocessResponse(
        dlq_id=dlq_id,
        new_task_id=new_task_id,
        status="queued",
        reprocessed_at=now,
    )
