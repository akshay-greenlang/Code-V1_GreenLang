"""
GreenLang Jobs REST API Routes

This module provides REST API endpoints for job queue management,
including listing jobs, getting status, and canceling running jobs.

Endpoints:
    GET  /api/v1/jobs          - List calculation jobs
    GET  /api/v1/jobs/{id}     - Get job status and results
    POST /api/v1/jobs/{id}/cancel - Cancel running job

Features:
    - Job lifecycle management
    - Progress tracking
    - Result retrieval
    - Cancellation support
    - Filtering and pagination

Example:
    >>> from fastapi import FastAPI
    >>> from greenlang.infrastructure.api.routes.jobs_routes import jobs_router
    >>>
    >>> app = FastAPI()
    >>> app.include_router(jobs_router, prefix="/api/v1")
"""

import logging
import uuid
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, validator

try:
    from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
    from fastapi.responses import JSONResponse
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    APIRouter = object
    Depends = None
    HTTPException = Exception
    Query = None
    Request = None
    status = None
    JSONResponse = None

logger = logging.getLogger(__name__)

# =============================================================================
# ENUMS
# =============================================================================


class JobStatus(str, Enum):
    """Job execution status."""
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class JobType(str, Enum):
    """Types of calculation jobs."""
    EMISSION_CALCULATION = "emission_calculation"
    BATCH_CALCULATION = "batch_calculation"
    COMPLIANCE_REPORT = "compliance_report"
    DATA_IMPORT = "data_import"
    DATA_EXPORT = "data_export"
    AGENT_EXECUTION = "agent_execution"
    OPTIMIZATION = "optimization"
    FORECAST = "forecast"


class JobPriority(str, Enum):
    """Job priority levels."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


# =============================================================================
# REQUEST MODELS
# =============================================================================


class JobCancelRequest(BaseModel):
    """
    Request model for job cancellation.

    Attributes:
        reason: Reason for cancellation
        force: Force cancellation even if job is in critical section
    """
    reason: Optional[str] = Field(
        default=None,
        max_length=500,
        description="Reason for cancellation"
    )
    force: bool = Field(
        default=False,
        description="Force cancellation even if job is in critical section"
    )

    class Config:
        schema_extra = {
            "example": {
                "reason": "User requested cancellation",
                "force": False
            }
        }


# =============================================================================
# RESPONSE MODELS
# =============================================================================


class JobProgress(BaseModel):
    """
    Job progress information.

    Attributes:
        percent_complete: Completion percentage
        current_step: Current processing step
        total_steps: Total number of steps
        items_processed: Number of items processed
        items_total: Total items to process
        estimated_remaining_seconds: Estimated time remaining
    """
    percent_complete: float = Field(..., ge=0, le=100, description="Completion percentage")
    current_step: str = Field(..., description="Current processing step")
    total_steps: int = Field(..., ge=1, description="Total steps")
    items_processed: int = Field(..., ge=0, description="Items processed")
    items_total: int = Field(..., ge=0, description="Total items")
    estimated_remaining_seconds: Optional[int] = Field(default=None, description="Estimated time remaining")


class JobResult(BaseModel):
    """
    Job execution result.

    Attributes:
        output: Output data from the job
        artifacts: List of artifact URLs/paths
        metrics: Execution metrics
        warnings: Any warnings generated
    """
    output: Optional[Dict[str, Any]] = Field(default=None, description="Job output data")
    artifacts: List[str] = Field(default_factory=list, description="Artifact URLs/paths")
    metrics: Dict[str, Any] = Field(default_factory=dict, description="Execution metrics")
    warnings: List[str] = Field(default_factory=list, description="Warnings generated")


class JobError(BaseModel):
    """
    Job error information.

    Attributes:
        code: Error code
        message: Error message
        details: Additional error details
        stack_trace: Stack trace (if available)
        recoverable: Whether error is recoverable
    """
    code: str = Field(..., description="Error code")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Error details")
    stack_trace: Optional[str] = Field(default=None, description="Stack trace")
    recoverable: bool = Field(default=False, description="Whether error is recoverable")


class JobSummary(BaseModel):
    """
    Summary information about a job.

    Attributes:
        job_id: Unique job identifier
        type: Job type
        status: Current job status
        priority: Job priority
        progress_percent: Completion percentage
        created_at: Job creation timestamp
        started_at: Job start timestamp
        completed_at: Job completion timestamp
    """
    job_id: str = Field(..., description="Unique job ID")
    type: JobType = Field(..., description="Job type")
    status: JobStatus = Field(..., description="Current status")
    priority: JobPriority = Field(..., description="Job priority")
    progress_percent: float = Field(..., ge=0, le=100, description="Progress percentage")
    created_at: datetime = Field(..., description="Creation timestamp")
    started_at: Optional[datetime] = Field(default=None, description="Start timestamp")
    completed_at: Optional[datetime] = Field(default=None, description="Completion timestamp")


class JobDetail(BaseModel):
    """
    Detailed job information.

    Attributes:
        job_id: Unique job identifier
        type: Job type
        status: Current job status
        priority: Job priority
        progress: Progress information
        input_params: Input parameters for the job
        result: Job result (if completed)
        error: Error information (if failed)
        submitted_by: User who submitted the job
        agent_id: Agent processing the job
        retry_count: Number of retry attempts
        max_retries: Maximum retry attempts
        timeout_seconds: Job timeout in seconds
        created_at: Job creation timestamp
        started_at: Job start timestamp
        completed_at: Job completion timestamp
        updated_at: Last update timestamp
    """
    job_id: str = Field(..., description="Unique job ID")
    type: JobType = Field(..., description="Job type")
    status: JobStatus = Field(..., description="Current status")
    priority: JobPriority = Field(..., description="Job priority")
    progress: JobProgress = Field(..., description="Progress information")
    input_params: Dict[str, Any] = Field(..., description="Input parameters")
    result: Optional[JobResult] = Field(default=None, description="Job result")
    error: Optional[JobError] = Field(default=None, description="Error information")
    submitted_by: str = Field(..., description="Submitting user")
    agent_id: Optional[str] = Field(default=None, description="Processing agent")
    retry_count: int = Field(default=0, ge=0, description="Retry count")
    max_retries: int = Field(default=3, ge=0, description="Max retries")
    timeout_seconds: int = Field(..., ge=1, description="Timeout in seconds")
    created_at: datetime = Field(..., description="Creation timestamp")
    started_at: Optional[datetime] = Field(default=None, description="Start timestamp")
    completed_at: Optional[datetime] = Field(default=None, description="Completion timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")

    class Config:
        schema_extra = {
            "example": {
                "job_id": "job_abc123xyz",
                "type": "emission_calculation",
                "status": "completed",
                "priority": "normal",
                "progress": {
                    "percent_complete": 100.0,
                    "current_step": "completed",
                    "total_steps": 5,
                    "items_processed": 100,
                    "items_total": 100
                },
                "input_params": {
                    "fuel_type": "diesel",
                    "activity_amount": 1000
                },
                "result": {
                    "output": {"emissions_kg_co2e": 10210.0},
                    "artifacts": [],
                    "metrics": {"execution_time_ms": 245}
                },
                "submitted_by": "user@example.com",
                "agent_id": "gl-010-emissions-guardian",
                "retry_count": 0,
                "max_retries": 3,
                "timeout_seconds": 300,
                "created_at": "2025-12-07T10:00:00Z",
                "started_at": "2025-12-07T10:00:01Z",
                "completed_at": "2025-12-07T10:00:03Z",
                "updated_at": "2025-12-07T10:00:03Z"
            }
        }


class JobListResponse(BaseModel):
    """
    Paginated list of jobs.

    Attributes:
        items: List of job summaries
        total: Total number of jobs
        page: Current page number
        page_size: Items per page
        total_pages: Total pages
        has_next: Has next page
        has_prev: Has previous page
        queue_stats: Queue statistics
    """
    items: List[JobSummary] = Field(..., description="Job summaries")
    total: int = Field(..., description="Total jobs")
    page: int = Field(..., ge=1, description="Current page")
    page_size: int = Field(..., ge=1, le=100, description="Items per page")
    total_pages: int = Field(..., ge=0, description="Total pages")
    has_next: bool = Field(..., description="Has next page")
    has_prev: bool = Field(..., description="Has previous page")
    queue_stats: Optional[Dict[str, int]] = Field(default=None, description="Queue statistics")


class JobCancelResponse(BaseModel):
    """
    Response for job cancellation request.

    Attributes:
        job_id: Job that was cancelled
        success: Whether cancellation was successful
        previous_status: Status before cancellation
        message: Cancellation message
        cancelled_at: Cancellation timestamp
    """
    job_id: str = Field(..., description="Job ID")
    success: bool = Field(..., description="Cancellation success")
    previous_status: JobStatus = Field(..., description="Previous status")
    message: str = Field(..., description="Result message")
    cancelled_at: datetime = Field(..., description="Cancellation timestamp")


class ErrorResponse(BaseModel):
    """Standard error response."""
    error_code: str = Field(..., description="Error code")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Error details")
    correlation_id: Optional[str] = Field(default=None, description="Correlation ID")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# =============================================================================
# STORAGE (In-memory for demonstration)
# =============================================================================

# Sample jobs for demonstration
_jobs: Dict[str, JobDetail] = {}


def _initialize_sample_jobs():
    """Initialize sample jobs for demonstration."""
    now = datetime.now(timezone.utc)

    sample_jobs = [
        JobDetail(
            job_id="job_001abc",
            type=JobType.EMISSION_CALCULATION,
            status=JobStatus.COMPLETED,
            priority=JobPriority.NORMAL,
            progress=JobProgress(
                percent_complete=100.0,
                current_step="completed",
                total_steps=5,
                items_processed=100,
                items_total=100
            ),
            input_params={"fuel_type": "diesel", "activity_amount": 1000},
            result=JobResult(
                output={"emissions_kg_co2e": 10210.0, "emissions_tonnes_co2e": 10.21},
                artifacts=[],
                metrics={"execution_time_ms": 245, "factor_lookups": 3}
            ),
            submitted_by="user@example.com",
            agent_id="gl-010-emissions-guardian",
            timeout_seconds=300,
            created_at=now - timedelta(hours=2),
            started_at=now - timedelta(hours=2) + timedelta(seconds=1),
            completed_at=now - timedelta(hours=2) + timedelta(seconds=3),
            updated_at=now - timedelta(hours=2) + timedelta(seconds=3)
        ),
        JobDetail(
            job_id="job_002def",
            type=JobType.BATCH_CALCULATION,
            status=JobStatus.RUNNING,
            priority=JobPriority.HIGH,
            progress=JobProgress(
                percent_complete=65.0,
                current_step="processing_records",
                total_steps=5,
                items_processed=650,
                items_total=1000,
                estimated_remaining_seconds=120
            ),
            input_params={"batch_file": "emissions_q4_2025.csv", "record_count": 1000},
            submitted_by="admin@example.com",
            agent_id="gl-018-unified-combustion",
            timeout_seconds=600,
            created_at=now - timedelta(minutes=5),
            started_at=now - timedelta(minutes=5) + timedelta(seconds=2),
            updated_at=now
        ),
        JobDetail(
            job_id="job_003ghi",
            type=JobType.COMPLIANCE_REPORT,
            status=JobStatus.QUEUED,
            priority=JobPriority.NORMAL,
            progress=JobProgress(
                percent_complete=0.0,
                current_step="queued",
                total_steps=8,
                items_processed=0,
                items_total=0
            ),
            input_params={"report_type": "annual", "year": 2025, "framework": "GHG_Protocol"},
            submitted_by="compliance@example.com",
            timeout_seconds=1800,
            created_at=now - timedelta(minutes=1),
            updated_at=now - timedelta(minutes=1)
        ),
        JobDetail(
            job_id="job_004jkl",
            type=JobType.DATA_IMPORT,
            status=JobStatus.FAILED,
            priority=JobPriority.LOW,
            progress=JobProgress(
                percent_complete=45.0,
                current_step="validation",
                total_steps=4,
                items_processed=450,
                items_total=1000
            ),
            input_params={"source": "erp_system", "data_type": "activity_data"},
            error=JobError(
                code="VALIDATION_ERROR",
                message="Invalid data format in row 451",
                details={"row": 451, "field": "activity_unit", "value": "INVALID"},
                recoverable=True
            ),
            submitted_by="data@example.com",
            timeout_seconds=900,
            created_at=now - timedelta(hours=1),
            started_at=now - timedelta(hours=1) + timedelta(seconds=5),
            completed_at=now - timedelta(minutes=30),
            updated_at=now - timedelta(minutes=30)
        )
    ]

    for job in sample_jobs:
        _jobs[job.job_id] = job


# Initialize sample data
_initialize_sample_jobs()


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def get_queue_stats() -> Dict[str, int]:
    """Get queue statistics."""
    stats = {
        "queued": 0,
        "running": 0,
        "completed": 0,
        "failed": 0,
        "cancelled": 0,
        "timeout": 0
    }
    for job in _jobs.values():
        stats[job.status.value] = stats.get(job.status.value, 0) + 1
    return stats


# =============================================================================
# ROUTER DEFINITION
# =============================================================================

if FASTAPI_AVAILABLE:
    jobs_router = APIRouter(
        prefix="/jobs",
        tags=["Jobs"],
        responses={
            400: {"model": ErrorResponse, "description": "Bad Request"},
            401: {"model": ErrorResponse, "description": "Unauthorized"},
            403: {"model": ErrorResponse, "description": "Forbidden"},
            404: {"model": ErrorResponse, "description": "Not Found"},
            409: {"model": ErrorResponse, "description": "Conflict"},
            429: {"model": ErrorResponse, "description": "Rate Limited"},
            500: {"model": ErrorResponse, "description": "Internal Server Error"},
        }
    )


    @jobs_router.get(
        "",
        response_model=JobListResponse,
        summary="List calculation jobs",
        description="""
        Retrieve a paginated list of calculation jobs.

        Supports filtering by:
        - Status (queued, running, completed, failed, cancelled)
        - Type (emission_calculation, batch_calculation, etc.)
        - Priority (low, normal, high, critical)
        - Date range

        Includes queue statistics in the response.
        """,
        operation_id="list_jobs"
    )
    async def list_jobs(
        request: Request,
        page: int = Query(1, ge=1, description="Page number"),
        page_size: int = Query(20, ge=1, le=100, description="Items per page"),
        status_filter: Optional[JobStatus] = Query(None, alias="status", description="Filter by status"),
        job_type: Optional[JobType] = Query(None, alias="type", description="Filter by job type"),
        priority: Optional[JobPriority] = Query(None, description="Filter by priority"),
        from_date: Optional[datetime] = Query(None, description="Filter from date"),
        to_date: Optional[datetime] = Query(None, description="Filter to date"),
        include_stats: bool = Query(True, description="Include queue statistics"),
    ) -> JobListResponse:
        """
        List calculation jobs with pagination and filtering.

        Args:
            request: FastAPI request object
            page: Page number
            page_size: Items per page
            status_filter: Optional status filter
            job_type: Optional job type filter
            priority: Optional priority filter
            from_date: Optional start date filter
            to_date: Optional end date filter
            include_stats: Include queue statistics

        Returns:
            Paginated list of jobs with optional statistics
        """
        logger.info(f"Listing jobs: page={page}, page_size={page_size}")

        # Filter jobs
        jobs = list(_jobs.values())

        if status_filter:
            jobs = [j for j in jobs if j.status == status_filter]

        if job_type:
            jobs = [j for j in jobs if j.type == job_type]

        if priority:
            jobs = [j for j in jobs if j.priority == priority]

        if from_date:
            jobs = [j for j in jobs if j.created_at >= from_date]

        if to_date:
            jobs = [j for j in jobs if j.created_at <= to_date]

        # Sort by creation date (most recent first)
        jobs.sort(key=lambda x: x.created_at, reverse=True)

        # Convert to summaries
        summaries = [
            JobSummary(
                job_id=j.job_id,
                type=j.type,
                status=j.status,
                priority=j.priority,
                progress_percent=j.progress.percent_complete,
                created_at=j.created_at,
                started_at=j.started_at,
                completed_at=j.completed_at
            )
            for j in jobs
        ]

        # Paginate
        total = len(summaries)
        total_pages = (total + page_size - 1) // page_size if total > 0 else 0
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        paginated = summaries[start_idx:end_idx]

        return JobListResponse(
            items=paginated,
            total=total,
            page=page,
            page_size=page_size,
            total_pages=total_pages,
            has_next=page < total_pages,
            has_prev=page > 1,
            queue_stats=get_queue_stats() if include_stats else None
        )


    @jobs_router.get(
        "/{job_id}",
        response_model=JobDetail,
        summary="Get job status and results",
        description="""
        Retrieve detailed information about a specific job.

        Returns:
        - Job status and progress
        - Input parameters
        - Results (if completed)
        - Error information (if failed)
        - Execution metrics
        """,
        operation_id="get_job"
    )
    async def get_job(
        request: Request,
        job_id: str,
    ) -> JobDetail:
        """
        Get detailed job information.

        Args:
            request: FastAPI request object
            job_id: Job identifier

        Returns:
            Detailed job information

        Raises:
            HTTPException: If job not found
        """
        logger.info(f"Getting job details: {job_id}")

        job = _jobs.get(job_id)

        if not job:
            logger.warning(f"Job not found: {job_id}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "error_code": "JOB_NOT_FOUND",
                    "message": f"Job '{job_id}' not found"
                }
            )

        return job


    @jobs_router.post(
        "/{job_id}/cancel",
        response_model=JobCancelResponse,
        summary="Cancel running job",
        description="""
        Cancel a running or queued job.

        Cancellation behavior:
        - Queued jobs: Immediately removed from queue
        - Running jobs: Graceful shutdown requested
        - Completed/Failed jobs: Cannot be cancelled

        Force cancellation will interrupt critical sections (use with caution).
        """,
        operation_id="cancel_job"
    )
    async def cancel_job(
        request: Request,
        job_id: str,
        cancel_request: Optional[JobCancelRequest] = None,
    ) -> JobCancelResponse:
        """
        Cancel a running or queued job.

        Args:
            request: FastAPI request object
            job_id: Job identifier
            cancel_request: Optional cancellation parameters

        Returns:
            Cancellation result

        Raises:
            HTTPException: If job not found or cannot be cancelled
        """
        logger.info(f"Cancelling job: {job_id}")

        job = _jobs.get(job_id)

        if not job:
            logger.warning(f"Job not found: {job_id}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "error_code": "JOB_NOT_FOUND",
                    "message": f"Job '{job_id}' not found"
                }
            )

        # Check if job can be cancelled
        if job.status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail={
                    "error_code": "CANNOT_CANCEL",
                    "message": f"Job '{job_id}' is already in terminal state '{job.status.value}'"
                }
            )

        previous_status = job.status
        now = datetime.now(timezone.utc)

        # Update job status
        job.status = JobStatus.CANCELLED
        job.completed_at = now
        job.updated_at = now

        reason = cancel_request.reason if cancel_request else "User requested cancellation"

        if job.error is None:
            job.error = JobError(
                code="CANCELLED",
                message=f"Job cancelled: {reason}",
                recoverable=False
            )

        logger.info(f"Job cancelled: {job_id}, previous_status={previous_status.value}")

        return JobCancelResponse(
            job_id=job_id,
            success=True,
            previous_status=previous_status,
            message=f"Job successfully cancelled. Reason: {reason}",
            cancelled_at=now
        )

else:
    # Provide stub router when FastAPI is not available
    jobs_router = None
    logger.warning("FastAPI not available - jobs_router is None")
