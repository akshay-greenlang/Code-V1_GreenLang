"""
Pipeline Orchestration API Routes for GL-EUDR-APP v1.0

Manages EUDR compliance pipeline runs. A pipeline executes a sequence
of stages: data collection, geospatial analysis, deforestation check,
risk assessment, and DDS generation. Supports start, status polling,
history, retry, and cancellation.

Prefix: /api/v1/pipeline
Tags: Pipeline
"""

import uuid
import math
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional
from enum import Enum

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/pipeline", tags=["Pipeline"])

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class PipelineStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class StageStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


# ---------------------------------------------------------------------------
# Pipeline Stage Definitions
# ---------------------------------------------------------------------------

PIPELINE_STAGES = [
    {"name": "data_collection", "display_name": "Data Collection", "order": 1},
    {"name": "geospatial_analysis", "display_name": "Geospatial Analysis", "order": 2},
    {"name": "deforestation_check", "display_name": "Deforestation Check", "order": 3},
    {"name": "risk_assessment", "display_name": "Risk Assessment", "order": 4},
    {"name": "dds_generation", "display_name": "DDS Generation", "order": 5},
]


# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------


class PipelineStartRequest(BaseModel):
    """Request to start a compliance pipeline run.

    Example::

        {
            "supplier_id": "sup_abc123",
            "commodity": "soya",
            "plot_ids": ["plot_001", "plot_002"]
        }
    """

    supplier_id: str = Field(..., description="Target supplier ID")
    commodity: str = Field(..., description="EUDR-regulated commodity")
    plot_ids: List[str] = Field(
        ..., min_length=1, description="Plot IDs to include in analysis"
    )
    priority: Optional[str] = Field(
        "normal", description="Priority: low | normal | high"
    )


class StageProgress(BaseModel):
    """Progress details for a single pipeline stage."""

    name: str = Field(..., description="Stage internal name")
    display_name: str
    order: int
    status: StageStatus
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    error_message: Optional[str] = None
    output_summary: Optional[Dict] = None


class PipelineRunResponse(BaseModel):
    """Response model for a pipeline run."""

    run_id: str = Field(..., description="Unique pipeline run identifier")
    supplier_id: str
    commodity: str
    plot_ids: List[str]
    status: PipelineStatus
    priority: str
    progress_percent: float = Field(
        ..., ge=0, le=100, description="Overall completion percentage"
    )
    stages: List[StageProgress]
    started_at: datetime
    completed_at: Optional[datetime] = None
    total_duration_seconds: Optional[float] = None
    error_message: Optional[str] = None
    retry_count: int = Field(0, description="Number of retries attempted")
    created_at: datetime


class PipelineListResponse(BaseModel):
    """Paginated list of pipeline runs."""

    items: List[PipelineRunResponse]
    page: int
    limit: int
    total: int
    total_pages: int


class PipelineActionResponse(BaseModel):
    """Response for pipeline actions (retry, cancel)."""

    run_id: str
    status: PipelineStatus
    message: str


# ---------------------------------------------------------------------------
# In-Memory Storage (v1.0)
# ---------------------------------------------------------------------------

_pipeline_runs: Dict[str, dict] = {}


def _create_stages() -> List[dict]:
    """Create initial stage records for a new pipeline run."""
    return [
        {
            "name": stage["name"],
            "display_name": stage["display_name"],
            "order": stage["order"],
            "status": StageStatus.PENDING,
            "started_at": None,
            "completed_at": None,
            "duration_seconds": None,
            "error_message": None,
            "output_summary": None,
        }
        for stage in PIPELINE_STAGES
    ]


def _simulate_pipeline_progress(record: dict) -> None:
    """
    Simulate pipeline execution progress for v1.0.

    In production, each stage would be executed asynchronously by
    dedicated workers. Here we simulate immediate completion of the
    first few stages to demonstrate the status model.
    """
    now = datetime.now(timezone.utc)
    stages = record["stages"]

    # Simulate: first 3 stages complete, 4th running, 5th pending
    for i, stage in enumerate(stages):
        if i < 3:
            stage["status"] = StageStatus.COMPLETED
            stage["started_at"] = record["started_at"] + timedelta(seconds=i * 10)
            stage["completed_at"] = record["started_at"] + timedelta(seconds=(i + 1) * 10)
            stage["duration_seconds"] = 10.0
            stage["output_summary"] = {"records_processed": 100 + i * 50}
        elif i == 3:
            stage["status"] = StageStatus.RUNNING
            stage["started_at"] = record["started_at"] + timedelta(seconds=30)
        # else: remains PENDING

    completed = sum(1 for s in stages if s["status"] == StageStatus.COMPLETED)
    record["progress_percent"] = round((completed / len(stages)) * 100, 1)
    record["status"] = PipelineStatus.RUNNING


def _compute_progress(record: dict) -> float:
    """Recompute progress percentage from stage statuses."""
    stages = record["stages"]
    completed = sum(1 for s in stages if s["status"] in (StageStatus.COMPLETED, StageStatus.SKIPPED))
    return round((completed / len(stages)) * 100, 1) if stages else 0.0


def _build_run_response(data: dict) -> PipelineRunResponse:
    return PipelineRunResponse(**data)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post(
    "/start",
    response_model=PipelineRunResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Start pipeline",
    description="Start a new EUDR compliance pipeline run for a supplier.",
)
async def start_pipeline(body: PipelineStartRequest) -> PipelineRunResponse:
    """
    Initiate a compliance pipeline run.

    Creates stage records for each pipeline phase and begins execution.
    In v1.0, progress is simulated; production would dispatch to async
    task workers.

    Returns:
        201 with pipeline run details.
    """
    now = datetime.now(timezone.utc)
    run_id = f"run_{uuid.uuid4().hex[:12]}"

    record = {
        "run_id": run_id,
        "supplier_id": body.supplier_id,
        "commodity": body.commodity.lower(),
        "plot_ids": body.plot_ids,
        "status": PipelineStatus.PENDING,
        "priority": body.priority or "normal",
        "progress_percent": 0.0,
        "stages": _create_stages(),
        "started_at": now,
        "completed_at": None,
        "total_duration_seconds": None,
        "error_message": None,
        "retry_count": 0,
        "created_at": now,
    }

    # Simulate initial progress
    _simulate_pipeline_progress(record)

    _pipeline_runs[run_id] = record
    logger.info(
        "Pipeline started: %s (supplier=%s, commodity=%s)",
        run_id,
        body.supplier_id,
        body.commodity,
    )
    return _build_run_response(record)


@router.get(
    "/{run_id}",
    response_model=PipelineRunResponse,
    summary="Get pipeline status",
    description="Retrieve pipeline run status with per-stage progress.",
)
async def get_pipeline_status(run_id: str) -> PipelineRunResponse:
    """
    Fetch the current status of a pipeline run including per-stage
    progress, timing, and any error messages.

    Returns:
        200 with pipeline run details.

    Raises:
        404 if run not found.
    """
    record = _pipeline_runs.get(run_id)
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Pipeline run '{run_id}' not found",
        )
    return _build_run_response(record)


@router.get(
    "/history",
    response_model=PipelineListResponse,
    summary="Pipeline run history",
    description="List pipeline runs with filtering and pagination.",
)
async def list_pipeline_runs(
    supplier_id: Optional[str] = Query(None, description="Filter by supplier ID"),
    pipeline_status: Optional[PipelineStatus] = Query(
        None, alias="status", description="Filter by pipeline status"
    ),
    page: int = Query(1, ge=1, description="Page number"),
    limit: int = Query(20, ge=1, le=100, description="Items per page"),
) -> PipelineListResponse:
    """
    Retrieve paginated list of pipeline runs.

    Returns:
        200 with paginated pipeline run list.
    """
    results = list(_pipeline_runs.values())

    if supplier_id:
        results = [r for r in results if r["supplier_id"] == supplier_id]
    if pipeline_status:
        results = [
            r for r in results
            if r["status"] == pipeline_status
            or (isinstance(r["status"], str) and r["status"] == pipeline_status.value)
        ]

    results.sort(key=lambda r: r["created_at"], reverse=True)

    total = len(results)
    total_pages = max(1, math.ceil(total / limit))
    start = (page - 1) * limit
    page_items = results[start : start + limit]

    return PipelineListResponse(
        items=[_build_run_response(r) for r in page_items],
        page=page,
        limit=limit,
        total=total,
        total_pages=total_pages,
    )


@router.post(
    "/{run_id}/retry",
    response_model=PipelineActionResponse,
    summary="Retry pipeline",
    description="Retry a failed pipeline run from the point of failure.",
)
async def retry_pipeline(run_id: str) -> PipelineActionResponse:
    """
    Retry a failed pipeline run.

    Resets failed stages to PENDING and restarts execution from the
    first failed stage. Increments the retry counter.

    Returns:
        200 with retry confirmation.

    Raises:
        400 if pipeline is not in a failed state.
        404 if run not found.
    """
    record = _pipeline_runs.get(run_id)
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Pipeline run '{run_id}' not found",
        )

    current_status = record["status"]
    if isinstance(current_status, PipelineStatus):
        current_status = current_status.value

    if current_status != PipelineStatus.FAILED.value:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Can only retry failed pipelines. Current status: {current_status}",
        )

    # Reset failed stages
    for stage in record["stages"]:
        stage_status = stage["status"]
        if isinstance(stage_status, StageStatus):
            stage_status = stage_status.value
        if stage_status == StageStatus.FAILED.value:
            stage["status"] = StageStatus.PENDING
            stage["error_message"] = None
            stage["started_at"] = None
            stage["completed_at"] = None
            stage["duration_seconds"] = None

    record["status"] = PipelineStatus.RUNNING
    record["retry_count"] = record.get("retry_count", 0) + 1
    record["error_message"] = None
    record["progress_percent"] = _compute_progress(record)

    logger.info("Pipeline retried: %s (retry #%d)", run_id, record["retry_count"])

    return PipelineActionResponse(
        run_id=run_id,
        status=PipelineStatus.RUNNING,
        message=f"Pipeline retry initiated (attempt #{record['retry_count']})",
    )


@router.post(
    "/{run_id}/cancel",
    response_model=PipelineActionResponse,
    summary="Cancel pipeline",
    description="Cancel an active pipeline run.",
)
async def cancel_pipeline(run_id: str) -> PipelineActionResponse:
    """
    Cancel an active (pending or running) pipeline run.

    Marks all non-completed stages as SKIPPED and sets the overall
    status to CANCELLED.

    Returns:
        200 with cancellation confirmation.

    Raises:
        400 if pipeline is not in an active state.
        404 if run not found.
    """
    record = _pipeline_runs.get(run_id)
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Pipeline run '{run_id}' not found",
        )

    current_status = record["status"]
    if isinstance(current_status, PipelineStatus):
        current_status = current_status.value

    cancellable = {PipelineStatus.PENDING.value, PipelineStatus.RUNNING.value}
    if current_status not in cancellable:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot cancel pipeline in status '{current_status}'",
        )

    now = datetime.now(timezone.utc)

    for stage in record["stages"]:
        stage_status = stage["status"]
        if isinstance(stage_status, StageStatus):
            stage_status = stage_status.value
        if stage_status in (StageStatus.PENDING.value, StageStatus.RUNNING.value):
            stage["status"] = StageStatus.SKIPPED

    record["status"] = PipelineStatus.CANCELLED
    record["completed_at"] = now
    if record.get("started_at"):
        started = record["started_at"]
        delta = (now - started).total_seconds()
        record["total_duration_seconds"] = round(delta, 2)
    record["progress_percent"] = _compute_progress(record)

    logger.info("Pipeline cancelled: %s", run_id)

    return PipelineActionResponse(
        run_id=run_id,
        status=PipelineStatus.CANCELLED,
        message="Pipeline run cancelled successfully",
    )
