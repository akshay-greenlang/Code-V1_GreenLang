"""
Calculation Service for GreenLang GraphQL API

This module provides the CalculationService class that manages calculation
job orchestration, progress tracking, and result retrieval.

Features:
    - Async calculation job execution
    - Job status tracking and progress updates
    - Result caching and retrieval
    - Provenance tracking with SHA-256 hashes
    - Integration with SSE for real-time updates

Example:
    >>> service = CalculationService()
    >>> job = await service.run_calculation(params)
    >>> status = await service.get_job_status(job.id)
    >>> jobs = await service.list_jobs(status="running")
"""

import asyncio
import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)


class JobStatusEnum(str, Enum):
    """Calculation job status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobPriorityEnum(str, Enum):
    """Job priority levels."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class EmissionResult:
    """GHG emission calculation result."""
    id: str
    facility_id: str
    co2_tonnes: float
    ch4_tonnes: float
    n2o_tonnes: float
    total_co2e_tonnes: float
    provenance_hash: str
    calculation_method: str
    timestamp: datetime
    confidence_score: float


@dataclass
class CalculationJob:
    """Calculation job record."""
    id: str
    status: JobStatusEnum
    progress_percent: int
    agent_id: str
    facility_id: str
    input_summary: str
    start_date: date
    end_date: date
    priority: JobPriorityEnum
    parameters: Dict[str, Any]
    results: Optional[List[EmissionResult]]
    error_details: Optional[str]
    execution_time_ms: float
    created_at: datetime
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    provenance_hash: str


class CalculationInput(BaseModel):
    """Input for starting a calculation job."""
    agent_id: str = Field(..., description="Target agent ID")
    facility_id: str = Field(..., description="Facility identifier")
    start_date: date = Field(..., description="Calculation start date")
    end_date: date = Field(..., description="Calculation end date")
    parameters: Optional[Dict[str, Any]] = Field(
        default=None, description="Additional parameters"
    )
    priority: JobPriorityEnum = Field(
        default=JobPriorityEnum.NORMAL, description="Job priority"
    )

    @validator('end_date')
    def validate_date_range(cls, v: date, values: Dict[str, Any]) -> date:
        """Validate end_date is after start_date."""
        if 'start_date' in values and v < values['start_date']:
            raise ValueError("end_date must be after start_date")
        return v


class CalculationServiceError(Exception):
    """Base exception for calculation service errors."""
    pass


class JobNotFoundError(CalculationServiceError):
    """Raised when a job is not found."""
    pass


class JobExecutionError(CalculationServiceError):
    """Raised when job execution fails."""
    pass


class CalculationService:
    """
    Calculation Service for managing emission calculation jobs.

    Provides job orchestration, progress tracking, and result management
    with full provenance tracking.

    Attributes:
        _jobs: In-memory job registry
        _lock: Asyncio lock for thread-safe operations
        _progress_callbacks: Registered progress callbacks

    Example:
        >>> service = CalculationService()
        >>> params = CalculationInput(
        ...     agent_id="agent-001",
        ...     facility_id="facility-001",
        ...     start_date=date(2025, 1, 1),
        ...     end_date=date(2025, 1, 31)
        ... )
        >>> job = await service.run_calculation(params)
        >>> print(f"Job {job.id} status: {job.status}")
    """

    def __init__(self) -> None:
        """Initialize CalculationService."""
        self._jobs: Dict[str, CalculationJob] = {}
        self._lock = asyncio.Lock()
        self._progress_callbacks: List[Callable[[str, int, str], None]] = []
        self._running_tasks: Dict[str, asyncio.Task] = {}
        self._initialize_sample_jobs()
        logger.info("CalculationService initialized")

    def _initialize_sample_jobs(self) -> None:
        """Initialize sample jobs for demonstration."""
        now = datetime.utcnow()
        today = now.date()

        sample_jobs = [
            CalculationJob(
                id="job-001",
                status=JobStatusEnum.COMPLETED,
                progress_percent=100,
                agent_id="agent-001",
                facility_id="facility-001",
                input_summary="Facility 001, January 2025",
                start_date=date(2025, 1, 1),
                end_date=date(2025, 1, 31),
                priority=JobPriorityEnum.NORMAL,
                parameters={"include_scope3": True},
                results=[
                    EmissionResult(
                        id="result-001",
                        facility_id="facility-001",
                        co2_tonnes=1250.0,
                        ch4_tonnes=5.2,
                        n2o_tonnes=0.8,
                        total_co2e_tonnes=1256.5,
                        provenance_hash=hashlib.sha256(b"result-001").hexdigest(),
                        calculation_method="IPCC AR6",
                        timestamp=now,
                        confidence_score=0.95
                    )
                ],
                error_details=None,
                execution_time_ms=5432.1,
                created_at=now - timedelta(hours=2),
                started_at=now - timedelta(hours=2),
                completed_at=now - timedelta(hours=1),
                provenance_hash=hashlib.sha256(b"job-001").hexdigest()
            ),
            CalculationJob(
                id="job-002",
                status=JobStatusEnum.RUNNING,
                progress_percent=45,
                agent_id="agent-002",
                facility_id="facility-002",
                input_summary="Facility 002, Q1 2025",
                start_date=date(2025, 1, 1),
                end_date=date(2025, 3, 31),
                priority=JobPriorityEnum.HIGH,
                parameters={"emission_factors": "custom"},
                results=None,
                error_details=None,
                execution_time_ms=12500.0,
                created_at=now - timedelta(minutes=30),
                started_at=now - timedelta(minutes=25),
                completed_at=None,
                provenance_hash=hashlib.sha256(b"job-002").hexdigest()
            ),
            CalculationJob(
                id="job-003",
                status=JobStatusEnum.PENDING,
                progress_percent=0,
                agent_id="agent-001",
                facility_id="facility-003",
                input_summary="Facility 003, February 2025",
                start_date=date(2025, 2, 1),
                end_date=date(2025, 2, 28),
                priority=JobPriorityEnum.LOW,
                parameters={},
                results=None,
                error_details=None,
                execution_time_ms=0.0,
                created_at=now - timedelta(minutes=5),
                started_at=None,
                completed_at=None,
                provenance_hash=hashlib.sha256(b"job-003").hexdigest()
            ),
        ]

        for job in sample_jobs:
            self._jobs[job.id] = job

    async def run_calculation(
        self,
        input_params: CalculationInput
    ) -> CalculationJob:
        """
        Start a new calculation job.

        Args:
            input_params: Calculation input parameters

        Returns:
            Created CalculationJob

        Raises:
            CalculationServiceError: If job creation fails
        """
        try:
            job_id = str(uuid4())
            now = datetime.utcnow()

            # Create job record
            job = CalculationJob(
                id=job_id,
                status=JobStatusEnum.PENDING,
                progress_percent=0,
                agent_id=input_params.agent_id,
                facility_id=input_params.facility_id,
                input_summary=self._create_input_summary(input_params),
                start_date=input_params.start_date,
                end_date=input_params.end_date,
                priority=input_params.priority,
                parameters=input_params.parameters or {},
                results=None,
                error_details=None,
                execution_time_ms=0.0,
                created_at=now,
                started_at=None,
                completed_at=None,
                provenance_hash=self._calculate_job_hash(job_id, input_params)
            )

            async with self._lock:
                self._jobs[job_id] = job

            # Start background execution
            task = asyncio.create_task(self._execute_job(job_id))
            self._running_tasks[job_id] = task

            logger.info(
                f"Created calculation job {job_id} for facility {input_params.facility_id}"
            )
            return job

        except Exception as e:
            logger.error(f"Failed to create calculation job: {e}", exc_info=True)
            raise CalculationServiceError(
                f"Failed to create calculation job: {str(e)}"
            ) from e

    async def get_job_status(self, job_id: str) -> Optional[CalculationJob]:
        """
        Get job status and details.

        Args:
            job_id: Job identifier

        Returns:
            CalculationJob if found, None otherwise
        """
        async with self._lock:
            job = self._jobs.get(job_id)

        if job:
            logger.debug(f"Retrieved job {job_id}: {job.status}")
        else:
            logger.debug(f"Job not found: {job_id}")

        return job

    async def list_jobs(
        self,
        status: Optional[str] = None,
        agent_id: Optional[str] = None,
        facility_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[CalculationJob]:
        """
        List calculation jobs with optional filters.

        Args:
            status: Filter by job status
            agent_id: Filter by agent ID
            facility_id: Filter by facility ID
            limit: Maximum number of results
            offset: Number of results to skip

        Returns:
            List of matching CalculationJob records
        """
        async with self._lock:
            jobs = list(self._jobs.values())

        # Apply filters
        if status:
            try:
                status_enum = JobStatusEnum(status.lower())
                jobs = [j for j in jobs if j.status == status_enum]
            except ValueError:
                logger.warning(f"Invalid status filter: {status}")
                return []

        if agent_id:
            jobs = [j for j in jobs if j.agent_id == agent_id]

        if facility_id:
            jobs = [j for j in jobs if j.facility_id == facility_id]

        # Sort by creation time (newest first)
        jobs.sort(key=lambda j: j.created_at, reverse=True)

        # Apply pagination
        jobs = jobs[offset:offset + limit]

        logger.debug(f"Listed {len(jobs)} jobs (status={status})")
        return jobs

    async def cancel_job(self, job_id: str) -> CalculationJob:
        """
        Cancel a running or pending job.

        Args:
            job_id: Job identifier

        Returns:
            Updated CalculationJob

        Raises:
            JobNotFoundError: If job is not found
            CalculationServiceError: If cancellation fails
        """
        async with self._lock:
            if job_id not in self._jobs:
                raise JobNotFoundError(f"Job not found: {job_id}")

            job = self._jobs[job_id]

            if job.status not in (JobStatusEnum.PENDING, JobStatusEnum.RUNNING):
                raise CalculationServiceError(
                    f"Cannot cancel job in {job.status} status"
                )

            # Cancel running task
            if job_id in self._running_tasks:
                self._running_tasks[job_id].cancel()
                del self._running_tasks[job_id]

            job.status = JobStatusEnum.CANCELLED
            job.completed_at = datetime.utcnow()

            logger.info(f"Cancelled job {job_id}")
            return job

    async def get_job_results(
        self,
        job_id: str
    ) -> Optional[List[EmissionResult]]:
        """
        Get results for a completed job.

        Args:
            job_id: Job identifier

        Returns:
            List of EmissionResult if job is completed, None otherwise
        """
        job = await self.get_job_status(job_id)

        if not job:
            return None

        if job.status != JobStatusEnum.COMPLETED:
            logger.debug(f"Job {job_id} not completed: {job.status}")
            return None

        return job.results

    def register_progress_callback(
        self,
        callback: Callable[[str, int, str], None]
    ) -> None:
        """
        Register a callback for job progress updates.

        Args:
            callback: Function(job_id, progress_percent, message)
        """
        self._progress_callbacks.append(callback)
        logger.debug("Registered progress callback")

    async def _execute_job(self, job_id: str) -> None:
        """
        Execute a calculation job (background task).

        Args:
            job_id: Job identifier
        """
        try:
            async with self._lock:
                job = self._jobs.get(job_id)
                if not job:
                    return

                job.status = JobStatusEnum.RUNNING
                job.started_at = datetime.utcnow()

            start_time = datetime.utcnow()

            # Simulate calculation steps
            steps = [
                (0, "Initializing calculation engine"),
                (10, "Loading facility data"),
                (25, "Retrieving emission factors"),
                (40, "Processing activity data"),
                (60, "Calculating Scope 1 emissions"),
                (75, "Calculating Scope 2 emissions"),
                (90, "Validating results"),
                (100, "Generating provenance hash"),
            ]

            for progress, message in steps:
                await asyncio.sleep(0.5)  # Simulate work

                async with self._lock:
                    job = self._jobs.get(job_id)
                    if not job or job.status == JobStatusEnum.CANCELLED:
                        return

                    job.progress_percent = progress

                # Notify callbacks
                await self._notify_progress(job_id, progress, message)

            # Generate results
            now = datetime.utcnow()
            execution_time = (now - start_time).total_seconds() * 1000

            results = [
                EmissionResult(
                    id=str(uuid4()),
                    facility_id=job.facility_id,
                    co2_tonnes=1250.0 + (hash(job_id) % 500),
                    ch4_tonnes=5.2 + (hash(job_id) % 10) * 0.1,
                    n2o_tonnes=0.8 + (hash(job_id) % 5) * 0.05,
                    total_co2e_tonnes=0.0,  # Calculated below
                    provenance_hash="",
                    calculation_method="IPCC AR6 GWP100",
                    timestamp=now,
                    confidence_score=0.95
                )
            ]

            # Calculate CO2e using GWP values
            for r in results:
                r.total_co2e_tonnes = (
                    r.co2_tonnes * 1.0 +
                    r.ch4_tonnes * 28.0 +
                    r.n2o_tonnes * 265.0
                )
                r.provenance_hash = self._calculate_result_hash(r)

            async with self._lock:
                job = self._jobs.get(job_id)
                if job and job.status != JobStatusEnum.CANCELLED:
                    job.status = JobStatusEnum.COMPLETED
                    job.progress_percent = 100
                    job.results = results
                    job.completed_at = now
                    job.execution_time_ms = execution_time

            logger.info(f"Job {job_id} completed in {execution_time:.1f}ms")

        except asyncio.CancelledError:
            logger.info(f"Job {job_id} was cancelled")
        except Exception as e:
            logger.error(f"Job {job_id} failed: {e}", exc_info=True)

            async with self._lock:
                job = self._jobs.get(job_id)
                if job:
                    job.status = JobStatusEnum.FAILED
                    job.error_details = str(e)
                    job.completed_at = datetime.utcnow()

    async def _notify_progress(
        self,
        job_id: str,
        progress: int,
        message: str
    ) -> None:
        """Notify all registered progress callbacks."""
        for callback in self._progress_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(job_id, progress, message)
                else:
                    callback(job_id, progress, message)
            except Exception as e:
                logger.error(f"Progress callback error: {e}")

    def _create_input_summary(self, params: CalculationInput) -> str:
        """Create human-readable input summary."""
        duration = (params.end_date - params.start_date).days + 1
        return (
            f"Facility {params.facility_id}, "
            f"{params.start_date.strftime('%b %Y')} - "
            f"{params.end_date.strftime('%b %Y')} ({duration} days)"
        )

    def _calculate_job_hash(
        self,
        job_id: str,
        params: CalculationInput
    ) -> str:
        """Calculate SHA-256 hash for job provenance."""
        content = json.dumps({
            "job_id": job_id,
            "agent_id": params.agent_id,
            "facility_id": params.facility_id,
            "start_date": params.start_date.isoformat(),
            "end_date": params.end_date.isoformat(),
            "parameters": params.parameters,
        }, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()

    def _calculate_result_hash(self, result: EmissionResult) -> str:
        """Calculate SHA-256 hash for result provenance."""
        content = f"{result.id}:{result.co2_tonnes}:{result.ch4_tonnes}:{result.n2o_tonnes}"
        return hashlib.sha256(content.encode()).hexdigest()


# Singleton instance
_calculation_service_instance: Optional[CalculationService] = None


def get_calculation_service() -> CalculationService:
    """
    Get the global CalculationService instance.

    Returns:
        CalculationService singleton instance
    """
    global _calculation_service_instance
    if _calculation_service_instance is None:
        _calculation_service_instance = CalculationService()
    return _calculation_service_instance


__all__ = [
    "CalculationService",
    "CalculationJob",
    "CalculationInput",
    "EmissionResult",
    "JobStatusEnum",
    "JobPriorityEnum",
    "CalculationServiceError",
    "JobNotFoundError",
    "JobExecutionError",
    "get_calculation_service",
]
