# -*- coding: utf-8 -*-
"""
Time Series Gap Filler REST API Router - AGENT-DATA-014

FastAPI router providing 20 endpoints for gap detection,
frequency analysis, gap filling, fill validation, cross-series
correlation, calendar management, job management, and
health/statistics.

All endpoints are mounted under ``/api/v1/gap-filler``.

Endpoints:
    1.  POST   /jobs                        - Create gap filling job
    2.  GET    /jobs                        - List jobs
    3.  GET    /jobs/{job_id}               - Get job details
    4.  DELETE /jobs/{job_id}               - Delete/cancel job
    5.  POST   /detect                     - Detect gaps in series
    6.  POST   /detect/batch               - Batch gap detection
    7.  GET    /detections                  - List detections
    8.  GET    /detections/{detection_id}   - Get detection result
    9.  POST   /frequency                  - Analyze frequency
    10. GET    /frequency/{analysis_id}    - Get frequency analysis
    11. POST   /fill                       - Fill detected gaps
    12. GET    /fills/{fill_id}            - Get fill details
    13. POST   /validate                   - Validate filled values
    14. GET    /validations/{validation_id} - Get validation result
    15. POST   /correlations               - Compute correlations
    16. GET    /correlations               - List correlations
    17. POST   /calendars                  - Create calendar
    18. GET    /calendars                  - List calendars
    19. GET    /health                     - Health check
    20. GET    /stats                      - Statistics

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-014 Time Series Gap Filler (GL-DATA-X-017)
Status: Production Ready
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional FastAPI import (no `from __future__ import annotations` here)
# ---------------------------------------------------------------------------

try:
    from fastapi import APIRouter, HTTPException, Query, Request
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field

    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    APIRouter = None  # type: ignore[assignment, misc]
    logger.warning(
        "FastAPI not available; time series gap filler router is None"
    )


# ---------------------------------------------------------------------------
# Pydantic request/response models (only when FastAPI is available)
# ---------------------------------------------------------------------------

if FASTAPI_AVAILABLE:

    # === Request Bodies ===

    class CreateGapFillerJobRequest(BaseModel):
        """Request body for creating a gap filling job."""
        series_id: str = Field(
            default="", description="Identifier of the series to process",
        )
        series: List[Any] = Field(
            default_factory=list,
            description="Time series data values (None/null for gaps)",
        )
        timestamps: List[Any] = Field(
            default_factory=list,
            description="Timestamps corresponding to series values",
        )
        strategy: str = Field(
            default="auto",
            description="Fill strategy (auto, linear, spline, seasonal, "
            "forward_fill, backward_fill, kalman)",
        )
        options: Optional[Dict[str, Any]] = Field(
            None, description="Additional job configuration options",
        )

    class DetectGapsRequest(BaseModel):
        """Request body for detecting gaps in a time series."""
        series: List[Any] = Field(
            ..., description="Time series data values",
        )
        timestamps: List[Any] = Field(
            ..., description="Timestamps corresponding to series values",
        )
        frequency: Optional[str] = Field(
            None,
            description="Expected frequency (auto-detected if omitted). "
            "Values: sub_minute, minutely, hourly, daily, weekly, "
            "monthly, quarterly, yearly",
        )
        name: Optional[str] = Field(
            None, description="Optional series name for labeling",
        )

    class AnalyzeFrequencyRequest(BaseModel):
        """Request body for analyzing time series frequency."""
        timestamps: List[Any] = Field(
            ...,
            description="Timestamps to analyze (ISO strings, epoch "
            "floats, or datetime objects)",
        )

    class FillGapsRequest(BaseModel):
        """Request body for filling gaps in a time series."""
        series: List[Any] = Field(
            ..., description="Time series data values (None/null for gaps)",
        )
        timestamps: List[Any] = Field(
            ..., description="Timestamps corresponding to series values",
        )
        gaps: Optional[List[Dict[str, Any]]] = Field(
            None,
            description="Pre-detected gap descriptors (auto-detected "
            "from None/NaN if omitted)",
        )
        strategy: Optional[str] = Field(
            None,
            description="Fill strategy (auto, linear, spline, seasonal, "
            "forward_fill, backward_fill, kalman)",
        )

    class ValidateFillsRequest(BaseModel):
        """Request body for validating filled values."""
        fills: List[Dict[str, Any]] = Field(
            ..., description="List of filled value dicts from a FillResult",
        )
        original_series: List[Any] = Field(
            ..., description="Original series values before filling",
        )

    class ComputeCorrelationsRequest(BaseModel):
        """Request body for computing cross-series correlations."""
        target: List[Any] = Field(
            ..., description="Target series values",
        )
        references: List[List[Any]] = Field(
            ..., description="List of reference series values",
        )
        method: str = Field(
            default="pearson",
            description="Correlation method (pearson, spearman, kendall)",
        )

    class CreateCalendarRequest(BaseModel):
        """Request body for creating a calendar definition."""
        name: str = Field(
            ..., description="Human-readable calendar name",
        )
        calendar_type: str = Field(
            default="business",
            description="Calendar type (business, fiscal, custom)",
        )
        timezone: str = Field(
            default="UTC",
            description="Timezone name (e.g., UTC, US/Eastern)",
        )
        business_days: List[int] = Field(
            default_factory=lambda: [0, 1, 2, 3, 4],
            description="Business day indices (0=Mon, 6=Sun)",
        )
        holidays: List[str] = Field(
            default_factory=list,
            description="Holiday dates as ISO-8601 strings",
        )
        fiscal_year_start_month: int = Field(
            default=1,
            description="Month number (1-12) for fiscal year start",
        )
        reporting_periods: List[Dict[str, Any]] = Field(
            default_factory=list,
            description="Custom reporting period definitions",
        )

    # === Response Models ===

    class JobResponse(BaseModel):
        """Response model for a gap filling job."""
        job_id: str = Field(default="")
        series_id: str = Field(default="")
        status: str = Field(default="pending")
        strategy: str = Field(default="auto")
        total_points: int = Field(default=0)
        gaps_detected: int = Field(default=0)
        gaps_filled: int = Field(default=0)
        fill_confidence: float = Field(default=0.0)
        config: Dict[str, Any] = Field(default_factory=dict)
        error_message: Optional[str] = Field(default=None)
        created_at: str = Field(default="")
        started_at: Optional[str] = Field(default=None)
        completed_at: Optional[str] = Field(default=None)
        provenance_hash: str = Field(default="")

    class JobListResponse(BaseModel):
        """Response model for listing jobs."""
        jobs: List[Dict[str, Any]] = Field(default_factory=list)
        count: int = Field(default=0)
        total: int = Field(default=0)
        limit: int = Field(default=50)
        offset: int = Field(default=0)

    class GapDetectionResponse(BaseModel):
        """Response model for gap detection results."""
        detection_id: str = Field(default="")
        series_name: str = Field(default="")
        total_points: int = Field(default=0)
        total_gaps: int = Field(default=0)
        gap_pct: float = Field(default=0.0)
        gaps: List[Dict[str, Any]] = Field(default_factory=list)
        gap_types: Dict[str, int] = Field(default_factory=dict)
        avg_gap_length: float = Field(default=0.0)
        max_gap_length: int = Field(default=0)
        processing_time_ms: float = Field(default=0.0)
        provenance_hash: str = Field(default="")

    class FrequencyResponse(BaseModel):
        """Response model for frequency analysis results."""
        analysis_id: str = Field(default="")
        detected_frequency: str = Field(default="unknown")
        frequency_seconds: float = Field(default=0.0)
        regularity_score: float = Field(default=0.0)
        confidence: float = Field(default=0.0)
        num_points: int = Field(default=0)
        median_interval: float = Field(default=0.0)
        std_interval: float = Field(default=0.0)
        is_regular: bool = Field(default=False)
        processing_time_ms: float = Field(default=0.0)
        provenance_hash: str = Field(default="")

    class FillResponse(BaseModel):
        """Response model for gap filling results."""
        fill_id: str = Field(default="")
        series_name: str = Field(default="")
        strategy: str = Field(default="linear")
        total_filled: int = Field(default=0)
        total_gaps: int = Field(default=0)
        fill_rate: float = Field(default=0.0)
        filled_values: List[Dict[str, Any]] = Field(default_factory=list)
        avg_confidence: float = Field(default=0.0)
        min_confidence: float = Field(default=1.0)
        distribution_preserved: bool = Field(default=True)
        processing_time_ms: float = Field(default=0.0)
        provenance_hash: str = Field(default="")

    class ValidationResponse(BaseModel):
        """Response model for fill validation results."""
        validation_id: str = Field(default="")
        fill_id: str = Field(default="")
        passed: bool = Field(default=True)
        total_checks: int = Field(default=0)
        passed_checks: int = Field(default=0)
        failed_checks: int = Field(default=0)
        checks: List[Dict[str, Any]] = Field(default_factory=list)
        overall_confidence: float = Field(default=0.0)
        distribution_test: str = Field(default="not_run")
        processing_time_ms: float = Field(default=0.0)
        provenance_hash: str = Field(default="")

    class CorrelationResponse(BaseModel):
        """Response model for correlation analysis results."""
        correlation_id: str = Field(default="")
        target_series: str = Field(default="")
        reference_series: str = Field(default="")
        method: str = Field(default="pearson")
        coefficient: float = Field(default=0.0)
        p_value: float = Field(default=1.0)
        sample_size: int = Field(default=0)
        is_significant: bool = Field(default=False)
        suitable_for_fill: bool = Field(default=False)
        processing_time_ms: float = Field(default=0.0)
        provenance_hash: str = Field(default="")

    class CalendarResponse(BaseModel):
        """Response model for calendar definitions."""
        calendar_id: str = Field(default="")
        name: str = Field(default="")
        calendar_type: str = Field(default="business")
        timezone_name: str = Field(default="UTC")
        business_days: List[int] = Field(default_factory=list)
        holidays: List[str] = Field(default_factory=list)
        fiscal_year_start_month: int = Field(default=1)
        reporting_periods: List[Dict[str, Any]] = Field(default_factory=list)
        active: bool = Field(default=True)
        created_at: str = Field(default="")
        provenance_hash: str = Field(default="")

    class HealthResponse(BaseModel):
        """Response model for health check."""
        status: str = Field(default="healthy")
        service: str = Field(default="time-series-gap-filler")
        started: bool = Field(default=False)
        engine_count: int = Field(default=0)
        engines: Dict[str, bool] = Field(default_factory=dict)
        stores: Dict[str, int] = Field(default_factory=dict)
        uptime_seconds: float = Field(default=0.0)
        provenance_entries: int = Field(default=0)
        prometheus_available: bool = Field(default=False)

    class StatsResponse(BaseModel):
        """Response model for aggregate statistics."""
        total_jobs: int = Field(default=0)
        completed_jobs: int = Field(default=0)
        failed_jobs: int = Field(default=0)
        cancelled_jobs: int = Field(default=0)
        total_gaps_detected: int = Field(default=0)
        total_gaps_filled: int = Field(default=0)
        total_validations: int = Field(default=0)
        total_frequency_analyses: int = Field(default=0)
        total_correlations: int = Field(default=0)
        total_calendars: int = Field(default=0)
        active_jobs: int = Field(default=0)
        avg_gap_pct: float = Field(default=0.0)
        avg_fill_confidence: float = Field(default=0.0)
        by_strategy: Dict[str, int] = Field(default_factory=dict)
        by_gap_type: Dict[str, int] = Field(default_factory=dict)
        by_frequency: Dict[str, int] = Field(default_factory=dict)
        provenance_entries: int = Field(default=0)

    class ErrorResponse(BaseModel):
        """Error response model."""
        error: str = Field(default="")
        detail: str = Field(default="")
        status_code: int = Field(default=500)


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

if FASTAPI_AVAILABLE:
    router = APIRouter(
        prefix="/api/v1/gap-filler",
        tags=["gap-filler"],
    )
else:
    router = None  # type: ignore[assignment]


def _get_service(request: Request) -> Any:
    """Extract TimeSeriesGapFillerService from app state.

    Args:
        request: FastAPI request object.

    Returns:
        TimeSeriesGapFillerService instance.

    Raises:
        HTTPException: If service is not configured.
    """
    service = getattr(
        request.app.state, "time_series_gap_filler_service", None,
    )
    if service is None:
        raise HTTPException(
            status_code=503,
            detail="Time series gap filler service not configured",
        )
    return service


if FASTAPI_AVAILABLE:

    # ------------------------------------------------------------------
    # 1. Create gap filling job
    # ------------------------------------------------------------------
    @router.post("/jobs")
    async def create_job(
        body: CreateGapFillerJobRequest,
        request: Request,
    ) -> Dict[str, Any]:
        """Create a new gap filling job."""
        service = _get_service(request)
        try:
            result = service.create_job(
                config={
                    "series_id": body.series_id,
                    "series": body.series,
                    "timestamps": body.timestamps,
                    "strategy": body.strategy,
                    "options": body.options,
                },
            )
            return result
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except Exception as exc:
            logger.error("Create job failed: %s", exc, exc_info=True)
            raise HTTPException(status_code=500, detail=str(exc))

    # ------------------------------------------------------------------
    # 2. List jobs
    # ------------------------------------------------------------------
    @router.get("/jobs")
    async def list_jobs(
        limit: int = Query(50, ge=1, le=1000),
        offset: int = Query(0, ge=0),
        request: Request = None,  # type: ignore[assignment]
    ) -> Dict[str, Any]:
        """List gap filling jobs with pagination."""
        service = _get_service(request)
        try:
            return service.list_jobs(limit=limit, offset=offset)
        except Exception as exc:
            logger.error("List jobs failed: %s", exc, exc_info=True)
            raise HTTPException(status_code=500, detail=str(exc))

    # ------------------------------------------------------------------
    # 3. Get job details
    # ------------------------------------------------------------------
    @router.get("/jobs/{job_id}")
    async def get_job(
        job_id: str,
        request: Request,
    ) -> Dict[str, Any]:
        """Get a gap filling job by ID."""
        service = _get_service(request)
        try:
            job = service.get_job(job_id)
            if job is None:
                raise HTTPException(
                    status_code=404,
                    detail=f"Job {job_id} not found",
                )
            return job
        except HTTPException:
            raise
        except Exception as exc:
            logger.error("Get job failed: %s", exc, exc_info=True)
            raise HTTPException(status_code=500, detail=str(exc))

    # ------------------------------------------------------------------
    # 4. Delete/cancel job
    # ------------------------------------------------------------------
    @router.delete("/jobs/{job_id}")
    async def delete_job(
        job_id: str,
        request: Request,
    ) -> Dict[str, Any]:
        """Delete (cancel) a gap filling job."""
        service = _get_service(request)
        try:
            return service.delete_job(job_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        except Exception as exc:
            logger.error("Delete job failed: %s", exc, exc_info=True)
            raise HTTPException(status_code=500, detail=str(exc))

    # ------------------------------------------------------------------
    # 5. Detect gaps
    # ------------------------------------------------------------------
    @router.post("/detect")
    async def detect_gaps(
        body: DetectGapsRequest,
        request: Request,
    ) -> Dict[str, Any]:
        """Detect gaps in a time series."""
        service = _get_service(request)
        try:
            result = service.detect_gaps(
                series=body.series,
                timestamps=body.timestamps,
                frequency=body.frequency,
            )
            return result.model_dump(mode="json")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except Exception as exc:
            logger.error("Detect gaps failed: %s", exc, exc_info=True)
            raise HTTPException(status_code=500, detail=str(exc))

    # ------------------------------------------------------------------
    # 6. Batch gap detection
    # ------------------------------------------------------------------
    @router.post("/detect/batch")
    async def detect_gaps_batch(
        body: List[DetectGapsRequest],
        request: Request,
    ) -> Dict[str, Any]:
        """Detect gaps across multiple series in batch."""
        service = _get_service(request)
        try:
            series_list = [
                {
                    "series": item.series,
                    "timestamps": item.timestamps,
                    "frequency": item.frequency,
                    "name": item.name,
                }
                for item in body
            ]
            result = service.detect_gaps_batch(series_list=series_list)
            return result.model_dump(mode="json")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except Exception as exc:
            logger.error("Batch detect failed: %s", exc, exc_info=True)
            raise HTTPException(status_code=500, detail=str(exc))

    # ------------------------------------------------------------------
    # 7. List detections
    # ------------------------------------------------------------------
    @router.get("/detections")
    async def list_detections(
        limit: int = Query(50, ge=1, le=1000),
        offset: int = Query(0, ge=0),
        request: Request = None,  # type: ignore[assignment]
    ) -> Dict[str, Any]:
        """List gap detection results with pagination."""
        service = _get_service(request)
        try:
            return service.list_detections(limit=limit, offset=offset)
        except Exception as exc:
            logger.error("List detections failed: %s", exc, exc_info=True)
            raise HTTPException(status_code=500, detail=str(exc))

    # ------------------------------------------------------------------
    # 8. Get detection result
    # ------------------------------------------------------------------
    @router.get("/detections/{detection_id}")
    async def get_detection(
        detection_id: str,
        request: Request,
    ) -> Dict[str, Any]:
        """Get a gap detection result by ID."""
        service = _get_service(request)
        try:
            result = service.get_detection(detection_id)
            if result is None:
                raise HTTPException(
                    status_code=404,
                    detail=f"Detection {detection_id} not found",
                )
            return result.model_dump(mode="json")
        except HTTPException:
            raise
        except Exception as exc:
            logger.error("Get detection failed: %s", exc, exc_info=True)
            raise HTTPException(status_code=500, detail=str(exc))

    # ------------------------------------------------------------------
    # 9. Analyze frequency
    # ------------------------------------------------------------------
    @router.post("/frequency")
    async def analyze_frequency(
        body: AnalyzeFrequencyRequest,
        request: Request,
    ) -> Dict[str, Any]:
        """Analyze the frequency of a time series."""
        service = _get_service(request)
        try:
            result = service.analyze_frequency(
                timestamps=body.timestamps,
            )
            return result.model_dump(mode="json")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except Exception as exc:
            logger.error("Frequency analysis failed: %s", exc, exc_info=True)
            raise HTTPException(status_code=500, detail=str(exc))

    # ------------------------------------------------------------------
    # 10. Get frequency analysis
    # ------------------------------------------------------------------
    @router.get("/frequency/{analysis_id}")
    async def get_frequency_analysis(
        analysis_id: str,
        request: Request,
    ) -> Dict[str, Any]:
        """Get a frequency analysis result by ID."""
        service = _get_service(request)
        try:
            result = service.get_frequency_analysis(analysis_id)
            if result is None:
                raise HTTPException(
                    status_code=404,
                    detail=f"Frequency analysis {analysis_id} not found",
                )
            return result
        except HTTPException:
            raise
        except Exception as exc:
            logger.error(
                "Get frequency analysis failed: %s", exc, exc_info=True,
            )
            raise HTTPException(status_code=500, detail=str(exc))

    # ------------------------------------------------------------------
    # 11. Fill gaps
    # ------------------------------------------------------------------
    @router.post("/fill")
    async def fill_gaps(
        body: FillGapsRequest,
        request: Request,
    ) -> Dict[str, Any]:
        """Fill detected gaps in a time series."""
        service = _get_service(request)
        try:
            result = service.fill_gaps(
                series=body.series,
                timestamps=body.timestamps,
                gaps=body.gaps,
                strategy=body.strategy,
            )
            return result.model_dump(mode="json")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except Exception as exc:
            logger.error("Fill gaps failed: %s", exc, exc_info=True)
            raise HTTPException(status_code=500, detail=str(exc))

    # ------------------------------------------------------------------
    # 12. Get fill details
    # ------------------------------------------------------------------
    @router.get("/fills/{fill_id}")
    async def get_fill(
        fill_id: str,
        request: Request,
    ) -> Dict[str, Any]:
        """Get a fill result by ID."""
        service = _get_service(request)
        try:
            result = service.get_fill(fill_id)
            if result is None:
                raise HTTPException(
                    status_code=404,
                    detail=f"Fill {fill_id} not found",
                )
            return result
        except HTTPException:
            raise
        except Exception as exc:
            logger.error("Get fill failed: %s", exc, exc_info=True)
            raise HTTPException(status_code=500, detail=str(exc))

    # ------------------------------------------------------------------
    # 13. Validate fills
    # ------------------------------------------------------------------
    @router.post("/validate")
    async def validate_fills(
        body: ValidateFillsRequest,
        request: Request,
    ) -> Dict[str, Any]:
        """Validate filled values against the original series."""
        service = _get_service(request)
        try:
            results = service.validate_fills(
                fills=body.fills,
                original_series=body.original_series,
            )
            return {
                "validations": [r.model_dump(mode="json") for r in results],
                "count": len(results),
                "all_passed": all(r.passed for r in results),
            }
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except Exception as exc:
            logger.error("Validate fills failed: %s", exc, exc_info=True)
            raise HTTPException(status_code=500, detail=str(exc))

    # ------------------------------------------------------------------
    # 14. Get validation result
    # ------------------------------------------------------------------
    @router.get("/validations/{validation_id}")
    async def get_validation(
        validation_id: str,
        request: Request,
    ) -> Dict[str, Any]:
        """Get a validation result by ID."""
        service = _get_service(request)
        try:
            result = service.get_validation(validation_id)
            if result is None:
                raise HTTPException(
                    status_code=404,
                    detail=f"Validation {validation_id} not found",
                )
            return result
        except HTTPException:
            raise
        except Exception as exc:
            logger.error("Get validation failed: %s", exc, exc_info=True)
            raise HTTPException(status_code=500, detail=str(exc))

    # ------------------------------------------------------------------
    # 15. Compute correlations
    # ------------------------------------------------------------------
    @router.post("/correlations")
    async def compute_correlations(
        body: ComputeCorrelationsRequest,
        request: Request,
    ) -> Dict[str, Any]:
        """Compute cross-series correlations for gap filling."""
        service = _get_service(request)
        try:
            results = service.compute_correlations(
                target=body.target,
                references=body.references,
                method=body.method,
            )
            return {
                "correlations": [r.model_dump(mode="json") for r in results],
                "count": len(results),
                "suitable_count": sum(
                    1 for r in results if r.suitable_for_fill
                ),
            }
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except Exception as exc:
            logger.error(
                "Compute correlations failed: %s", exc, exc_info=True,
            )
            raise HTTPException(status_code=500, detail=str(exc))

    # ------------------------------------------------------------------
    # 16. List correlations
    # ------------------------------------------------------------------
    @router.get("/correlations")
    async def list_correlations(
        limit: int = Query(50, ge=1, le=1000),
        offset: int = Query(0, ge=0),
        request: Request = None,  # type: ignore[assignment]
    ) -> Dict[str, Any]:
        """List correlation analysis results with pagination."""
        service = _get_service(request)
        try:
            return service.list_correlations(limit=limit, offset=offset)
        except Exception as exc:
            logger.error(
                "List correlations failed: %s", exc, exc_info=True,
            )
            raise HTTPException(status_code=500, detail=str(exc))

    # ------------------------------------------------------------------
    # 17. Create calendar
    # ------------------------------------------------------------------
    @router.post("/calendars")
    async def create_calendar(
        body: CreateCalendarRequest,
        request: Request,
    ) -> Dict[str, Any]:
        """Create a calendar definition for calendar-aware filling."""
        service = _get_service(request)
        try:
            result = service.create_calendar(
                calendar={
                    "name": body.name,
                    "calendar_type": body.calendar_type,
                    "timezone": body.timezone,
                    "business_days": body.business_days,
                    "holidays": body.holidays,
                    "fiscal_year_start_month": body.fiscal_year_start_month,
                    "reporting_periods": body.reporting_periods,
                },
            )
            return result.model_dump(mode="json")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except Exception as exc:
            logger.error("Create calendar failed: %s", exc, exc_info=True)
            raise HTTPException(status_code=500, detail=str(exc))

    # ------------------------------------------------------------------
    # 18. List calendars
    # ------------------------------------------------------------------
    @router.get("/calendars")
    async def list_calendars(
        limit: int = Query(50, ge=1, le=1000),
        offset: int = Query(0, ge=0),
        request: Request = None,  # type: ignore[assignment]
    ) -> Dict[str, Any]:
        """List calendar definitions with pagination."""
        service = _get_service(request)
        try:
            return service.list_calendars(limit=limit, offset=offset)
        except Exception as exc:
            logger.error("List calendars failed: %s", exc, exc_info=True)
            raise HTTPException(status_code=500, detail=str(exc))

    # ------------------------------------------------------------------
    # 19. Health check
    # ------------------------------------------------------------------
    @router.get("/health")
    async def health_check(
        request: Request,
    ) -> Dict[str, Any]:
        """Get gap filler service health status."""
        service = _get_service(request)
        try:
            return service.health_check()
        except Exception as exc:
            logger.error("Health check failed: %s", exc, exc_info=True)
            raise HTTPException(status_code=500, detail=str(exc))

    # ------------------------------------------------------------------
    # 20. Statistics
    # ------------------------------------------------------------------
    @router.get("/stats")
    async def get_stats(
        request: Request,
    ) -> Dict[str, Any]:
        """Get aggregate gap filler service statistics."""
        service = _get_service(request)
        try:
            stats = service.get_statistics()
            return stats.model_dump(mode="json")
        except Exception as exc:
            logger.error("Get stats failed: %s", exc, exc_info=True)
            raise HTTPException(status_code=500, detail=str(exc))
