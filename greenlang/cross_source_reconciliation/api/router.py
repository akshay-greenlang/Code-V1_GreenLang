# -*- coding: utf-8 -*-
"""
Cross-Source Reconciliation REST API Router - AGENT-DATA-015

FastAPI router providing 20 endpoints for cross-source data
reconciliation: job management, source registration, record matching,
field comparison, discrepancy detection, conflict resolution, golden
record assembly, pipeline orchestration, and health/statistics.

All endpoints are mounted under ``/api/v1/reconciliation``.

Endpoints:
    1.  POST   /jobs                               - Create reconciliation job
    2.  GET    /jobs                               - List jobs
    3.  GET    /jobs/{job_id}                      - Get job details
    4.  DELETE /jobs/{job_id}                      - Cancel/delete job
    5.  POST   /sources                            - Register data source
    6.  GET    /sources                            - List registered sources
    7.  GET    /sources/{source_id}                - Get source details
    8.  PUT    /sources/{source_id}                - Update source metadata
    9.  POST   /match                              - Match records across sources
    10. GET    /matches                            - List match results
    11. GET    /matches/{match_id}                 - Get match details
    12. POST   /compare                            - Compare matched records
    13. GET    /discrepancies                      - List discrepancies
    14. GET    /discrepancies/{discrepancy_id}     - Get discrepancy details
    15. POST   /resolve                            - Resolve discrepancies
    16. GET    /golden-records                     - List golden records
    17. GET    /golden-records/{record_id}         - Get golden record details
    18. POST   /pipeline                           - Run full reconciliation pipeline
    19. GET    /health                             - Health check
    20. GET    /stats                              - Service statistics

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-015 Cross-Source Reconciliation (GL-DATA-X-018)
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
        "FastAPI not available; cross-source reconciliation router is None"
    )


# ---------------------------------------------------------------------------
# Pydantic request/response models (only when FastAPI is available)
# ---------------------------------------------------------------------------

if FASTAPI_AVAILABLE:

    # === Request Bodies ===

    class CreateJobRequest(BaseModel):
        """Request body for creating a reconciliation job."""
        name: str = Field(
            default="", description="Human-readable job name",
        )
        source_ids: List[str] = Field(
            default_factory=list,
            description="List of source IDs to reconcile",
        )
        strategy: str = Field(
            default="auto",
            description="Reconciliation strategy (auto, priority_wins, "
            "most_recent, weighted_average, consensus)",
        )
        config: Optional[Dict[str, Any]] = Field(
            None, description="Additional job configuration overrides",
        )

    class RegisterSourceRequest(BaseModel):
        """Request body for registering a data source."""
        name: str = Field(
            ..., description="Human-readable source name",
        )
        source_type: str = Field(
            default="manual",
            description="Source type (erp, utility, meter, questionnaire, "
            "registry, manual, api)",
        )
        schema_info: Optional[Dict[str, Any]] = Field(
            None, description="Source schema definition with column types",
        )
        priority: int = Field(
            default=5, ge=1, le=10,
            description="Source priority (1=highest, 10=lowest)",
        )
        credibility_score: float = Field(
            default=0.8, ge=0.0, le=1.0,
            description="Source credibility score (0.0 to 1.0)",
        )
        refresh_cadence: str = Field(
            default="monthly",
            description="Data refresh frequency (daily, weekly, monthly, "
            "quarterly, annual, real_time)",
        )
        metadata: Optional[Dict[str, Any]] = Field(
            None, description="Additional source metadata",
        )

    class UpdateSourceRequest(BaseModel):
        """Request body for updating a data source."""
        name: Optional[str] = Field(
            None, description="New source name",
        )
        priority: Optional[int] = Field(
            None, ge=1, le=10,
            description="New source priority",
        )
        credibility_score: Optional[float] = Field(
            None, ge=0.0, le=1.0,
            description="New credibility score",
        )
        refresh_cadence: Optional[str] = Field(
            None, description="New refresh cadence",
        )
        schema_info: Optional[Dict[str, Any]] = Field(
            None, description="New schema definition",
        )
        metadata: Optional[Dict[str, Any]] = Field(
            None, description="Additional metadata to merge",
        )

    class MatchRecordsRequest(BaseModel):
        """Request body for matching records across sources."""
        source_ids: List[str] = Field(
            default_factory=list,
            description="Source IDs to match across",
        )
        records_a: List[Dict[str, Any]] = Field(
            default_factory=list,
            description="First set of records for matching",
        )
        records_b: List[Dict[str, Any]] = Field(
            default_factory=list,
            description="Second set of records for matching",
        )
        match_keys: List[str] = Field(
            default_factory=lambda: ["entity_id", "period"],
            description="Fields to use as composite match key",
        )
        threshold: float = Field(
            default=0.85, ge=0.0, le=1.0,
            description="Minimum match confidence threshold",
        )
        strategy: str = Field(
            default="composite",
            description="Matching strategy (exact, fuzzy, composite, "
            "rule_based)",
        )

    class CompareRecordsRequest(BaseModel):
        """Request body for comparing matched records."""
        match_id: Optional[str] = Field(
            None,
            description="Stored match ID to compare (loads records from "
            "match result)",
        )
        record_a: Optional[Dict[str, Any]] = Field(
            None, description="First record for inline comparison",
        )
        record_b: Optional[Dict[str, Any]] = Field(
            None, description="Second record for inline comparison",
        )
        fields: Optional[List[str]] = Field(
            None,
            description="Fields to compare (all shared fields if None)",
        )
        tolerance_pct: float = Field(
            default=5.0, ge=0.0,
            description="Relative tolerance as percentage",
        )
        tolerance_abs: float = Field(
            default=0.01, ge=0.0,
            description="Absolute tolerance for numeric fields",
        )

    class ResolveDiscrepanciesRequest(BaseModel):
        """Request body for resolving discrepancies."""
        discrepancy_ids: List[str] = Field(
            default_factory=list,
            description="IDs of discrepancies to resolve",
        )
        strategy: str = Field(
            default="priority_wins",
            description="Resolution strategy (priority_wins, most_recent, "
            "weighted_average, most_complete, consensus, "
            "manual_override)",
        )
        source_priorities: Optional[Dict[str, int]] = Field(
            None,
            description="Source priority map for priority_wins strategy",
        )
        manual_values: Optional[Dict[str, Any]] = Field(
            None,
            description="Manual override values for manual_override "
            "strategy",
        )

    class RunPipelineRequest(BaseModel):
        """Request body for running the full reconciliation pipeline."""
        source_ids: List[str] = Field(
            default_factory=list,
            description="Source IDs to reconcile",
        )
        records_a: List[Dict[str, Any]] = Field(
            default_factory=list,
            description="First set of records",
        )
        records_b: List[Dict[str, Any]] = Field(
            default_factory=list,
            description="Second set of records",
        )
        match_keys: List[str] = Field(
            default_factory=lambda: ["entity_id", "period"],
            description="Fields for record matching",
        )
        match_threshold: float = Field(
            default=0.85, ge=0.0, le=1.0,
            description="Match confidence threshold",
        )
        tolerance_pct: float = Field(
            default=5.0, ge=0.0,
            description="Relative comparison tolerance (%)",
        )
        tolerance_abs: float = Field(
            default=0.01, ge=0.0,
            description="Absolute comparison tolerance",
        )
        resolution_strategy: str = Field(
            default="priority_wins",
            description="Conflict resolution strategy",
        )
        generate_golden_records: bool = Field(
            default=True,
            description="Whether to assemble golden records",
        )

    # === Response Models ===

    class JobResponse(BaseModel):
        """Response model for a reconciliation job."""
        job_id: str = Field(default="")
        name: str = Field(default="")
        source_ids: List[str] = Field(default_factory=list)
        strategy: str = Field(default="auto")
        status: str = Field(default="pending")
        match_count: int = Field(default=0)
        discrepancy_count: int = Field(default=0)
        golden_record_count: int = Field(default=0)
        created_at: str = Field(default="")
        started_at: Optional[str] = Field(default=None)
        completed_at: Optional[str] = Field(default=None)
        provenance_hash: str = Field(default="")

    class SourceResponse(BaseModel):
        """Response model for a registered data source."""
        source_id: str = Field(default="")
        name: str = Field(default="")
        source_type: str = Field(default="manual")
        priority: int = Field(default=5)
        credibility_score: float = Field(default=0.8)
        refresh_cadence: str = Field(default="monthly")
        record_count: int = Field(default=0)
        status: str = Field(default="active")
        created_at: str = Field(default="")
        updated_at: str = Field(default="")
        provenance_hash: str = Field(default="")

    class MatchResponse(BaseModel):
        """Response model for record matching results."""
        match_id: str = Field(default="")
        source_ids: List[str] = Field(default_factory=list)
        strategy: str = Field(default="composite")
        threshold: float = Field(default=0.85)
        total_matched: int = Field(default=0)
        total_unmatched_a: int = Field(default=0)
        total_unmatched_b: int = Field(default=0)
        avg_confidence: float = Field(default=0.0)
        processing_time_ms: float = Field(default=0.0)
        provenance_hash: str = Field(default="")

    class ComparisonResponse(BaseModel):
        """Response model for field comparison results."""
        comparison_id: str = Field(default="")
        match_id: str = Field(default="")
        total_fields: int = Field(default=0)
        matching_fields: int = Field(default=0)
        mismatching_fields: int = Field(default=0)
        missing_fields: int = Field(default=0)
        match_rate: float = Field(default=0.0)
        processing_time_ms: float = Field(default=0.0)
        provenance_hash: str = Field(default="")

    class DiscrepancyResponse(BaseModel):
        """Response model for a detected discrepancy."""
        discrepancy_id: str = Field(default="")
        detection_id: str = Field(default="")
        field: str = Field(default="")
        type: str = Field(default="value_mismatch")
        severity: str = Field(default="medium")
        value_a: Optional[Any] = Field(default=None)
        value_b: Optional[Any] = Field(default=None)
        abs_diff: float = Field(default=0.0)
        rel_diff_pct: float = Field(default=0.0)
        status: str = Field(default="open")

    class ResolutionResponse(BaseModel):
        """Response model for conflict resolution results."""
        resolution_id: str = Field(default="")
        strategy: str = Field(default="priority_wins")
        total_resolved: int = Field(default=0)
        resolutions: List[Dict[str, Any]] = Field(default_factory=list)
        processing_time_ms: float = Field(default=0.0)
        provenance_hash: str = Field(default="")

    class GoldenRecordResponse(BaseModel):
        """Response model for a golden record."""
        record_id: str = Field(default="")
        entity_id: str = Field(default="")
        period: str = Field(default="")
        field_values: Dict[str, Any] = Field(default_factory=dict)
        field_sources: Dict[str, str] = Field(default_factory=dict)
        field_confidence: Dict[str, float] = Field(default_factory=dict)
        overall_confidence: float = Field(default=0.0)
        status: str = Field(default="active")
        created_at: str = Field(default="")
        provenance_hash: str = Field(default="")

    class HealthResponse(BaseModel):
        """Response model for health check."""
        status: str = Field(default="healthy")
        service: str = Field(default="cross_source_reconciliation")
        engines: Dict[str, bool] = Field(default_factory=dict)
        stores: Dict[str, int] = Field(default_factory=dict)
        timestamp: str = Field(default="")

    class StatsResponse(BaseModel):
        """Response model for aggregate statistics."""
        total_jobs: int = Field(default=0)
        total_sources: int = Field(default=0)
        total_matches: int = Field(default=0)
        total_comparisons: int = Field(default=0)
        total_discrepancies: int = Field(default=0)
        total_resolutions: int = Field(default=0)
        total_golden_records: int = Field(default=0)
        total_pipelines: int = Field(default=0)
        provenance_entries: int = Field(default=0)
        timestamp: str = Field(default="")

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
        prefix="/api/v1/reconciliation",
        tags=["reconciliation"],
    )
else:
    router = None  # type: ignore[assignment]


def _get_service(request: Request) -> Any:
    """Extract CrossSourceReconciliationService from app state.

    Args:
        request: FastAPI request object.

    Returns:
        CrossSourceReconciliationService instance.

    Raises:
        HTTPException: If service is not configured.
    """
    service = getattr(
        request.app.state, "cross_source_reconciliation_service", None,
    )
    if service is None:
        raise HTTPException(
            status_code=503,
            detail="Cross-source reconciliation service not configured",
        )
    return service


if FASTAPI_AVAILABLE:

    # ------------------------------------------------------------------
    # 1. Create reconciliation job
    # ------------------------------------------------------------------
    @router.post("/jobs")
    async def create_job(
        body: CreateJobRequest,
        request: Request,
    ) -> JSONResponse:
        """Create a new reconciliation job."""
        service = _get_service(request)
        try:
            result = service.create_job(
                name=body.name,
                source_ids=body.source_ids,
                strategy=body.strategy,
                config=body.config,
            )
            return JSONResponse(
                status_code=201,
                content={"status": "created", "data": result},
            )
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
        status: Optional[str] = Query(None, description="Filter by status"),
        limit: int = Query(50, ge=1, le=1000),
        offset: int = Query(0, ge=0),
        request: Request = None,  # type: ignore[assignment]
    ) -> JSONResponse:
        """List reconciliation jobs with optional status filter."""
        service = _get_service(request)
        try:
            result = service.list_jobs(
                status=status, limit=limit, offset=offset,
            )
            return JSONResponse(
                content={"status": "ok", "data": result},
            )
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
    ) -> JSONResponse:
        """Get a reconciliation job by ID."""
        service = _get_service(request)
        try:
            job = service.get_job(job_id)
            if job is None:
                raise HTTPException(
                    status_code=404,
                    detail=f"Job {job_id} not found",
                )
            return JSONResponse(
                content={"status": "ok", "data": job},
            )
        except HTTPException:
            raise
        except Exception as exc:
            logger.error("Get job failed: %s", exc, exc_info=True)
            raise HTTPException(status_code=500, detail=str(exc))

    # ------------------------------------------------------------------
    # 4. Cancel/delete job
    # ------------------------------------------------------------------
    @router.delete("/jobs/{job_id}")
    async def delete_job(
        job_id: str,
        request: Request,
    ) -> JSONResponse:
        """Cancel and delete a reconciliation job."""
        service = _get_service(request)
        try:
            result = service.delete_job(job_id)
            return JSONResponse(
                content={"status": "ok", "data": result},
            )
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        except Exception as exc:
            logger.error("Delete job failed: %s", exc, exc_info=True)
            raise HTTPException(status_code=500, detail=str(exc))

    # ------------------------------------------------------------------
    # 5. Register data source
    # ------------------------------------------------------------------
    @router.post("/sources")
    async def register_source(
        body: RegisterSourceRequest,
        request: Request,
    ) -> JSONResponse:
        """Register a new data source for reconciliation."""
        service = _get_service(request)
        try:
            result = service.register_source(
                name=body.name,
                source_type=body.source_type,
                schema=body.schema_info,
                priority=body.priority,
                credibility_score=body.credibility_score,
                refresh_cadence=body.refresh_cadence,
                metadata=body.metadata,
            )
            return JSONResponse(
                status_code=201,
                content={"status": "created", "data": result},
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except Exception as exc:
            logger.error(
                "Register source failed: %s", exc, exc_info=True,
            )
            raise HTTPException(status_code=500, detail=str(exc))

    # ------------------------------------------------------------------
    # 6. List registered sources
    # ------------------------------------------------------------------
    @router.get("/sources")
    async def list_sources(
        limit: int = Query(50, ge=1, le=1000),
        offset: int = Query(0, ge=0),
        request: Request = None,  # type: ignore[assignment]
    ) -> JSONResponse:
        """List registered data sources with pagination."""
        service = _get_service(request)
        try:
            result = service.list_sources(limit=limit, offset=offset)
            return JSONResponse(
                content={"status": "ok", "data": result},
            )
        except Exception as exc:
            logger.error("List sources failed: %s", exc, exc_info=True)
            raise HTTPException(status_code=500, detail=str(exc))

    # ------------------------------------------------------------------
    # 7. Get source details
    # ------------------------------------------------------------------
    @router.get("/sources/{source_id}")
    async def get_source(
        source_id: str,
        request: Request,
    ) -> JSONResponse:
        """Get a registered data source by ID."""
        service = _get_service(request)
        try:
            source = service.get_source(source_id)
            if source is None:
                raise HTTPException(
                    status_code=404,
                    detail=f"Source {source_id} not found",
                )
            return JSONResponse(
                content={"status": "ok", "data": source},
            )
        except HTTPException:
            raise
        except Exception as exc:
            logger.error("Get source failed: %s", exc, exc_info=True)
            raise HTTPException(status_code=500, detail=str(exc))

    # ------------------------------------------------------------------
    # 8. Update source metadata
    # ------------------------------------------------------------------
    @router.put("/sources/{source_id}")
    async def update_source(
        source_id: str,
        body: UpdateSourceRequest,
        request: Request,
    ) -> JSONResponse:
        """Update a registered data source."""
        service = _get_service(request)
        try:
            result = service.update_source(
                source_id=source_id,
                name=body.name,
                priority=body.priority,
                credibility_score=body.credibility_score,
                refresh_cadence=body.refresh_cadence,
                schema=body.schema_info,
                metadata=body.metadata,
            )
            return JSONResponse(
                content={"status": "ok", "data": result},
            )
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        except Exception as exc:
            logger.error(
                "Update source failed: %s", exc, exc_info=True,
            )
            raise HTTPException(status_code=500, detail=str(exc))

    # ------------------------------------------------------------------
    # 9. Match records across sources
    # ------------------------------------------------------------------
    @router.post("/match")
    async def match_records(
        body: MatchRecordsRequest,
        request: Request,
    ) -> JSONResponse:
        """Match records across data sources."""
        service = _get_service(request)
        try:
            result = service.match_records(
                source_ids=body.source_ids,
                records_a=body.records_a,
                records_b=body.records_b,
                match_keys=body.match_keys,
                threshold=body.threshold,
                strategy=body.strategy,
            )
            return JSONResponse(
                content={"status": "ok", "data": result},
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except Exception as exc:
            logger.error(
                "Match records failed: %s", exc, exc_info=True,
            )
            raise HTTPException(status_code=500, detail=str(exc))

    # ------------------------------------------------------------------
    # 10. List match results
    # ------------------------------------------------------------------
    @router.get("/matches")
    async def list_matches(
        limit: int = Query(50, ge=1, le=1000),
        offset: int = Query(0, ge=0),
        request: Request = None,  # type: ignore[assignment]
    ) -> JSONResponse:
        """List record match results with pagination."""
        service = _get_service(request)
        try:
            result = service.list_matches(limit=limit, offset=offset)
            return JSONResponse(
                content={"status": "ok", "data": result},
            )
        except Exception as exc:
            logger.error("List matches failed: %s", exc, exc_info=True)
            raise HTTPException(status_code=500, detail=str(exc))

    # ------------------------------------------------------------------
    # 11. Get match details
    # ------------------------------------------------------------------
    @router.get("/matches/{match_id}")
    async def get_match(
        match_id: str,
        request: Request,
    ) -> JSONResponse:
        """Get a record match result by ID."""
        service = _get_service(request)
        try:
            match = service.get_match(match_id)
            if match is None:
                raise HTTPException(
                    status_code=404,
                    detail=f"Match {match_id} not found",
                )
            return JSONResponse(
                content={"status": "ok", "data": match},
            )
        except HTTPException:
            raise
        except Exception as exc:
            logger.error("Get match failed: %s", exc, exc_info=True)
            raise HTTPException(status_code=500, detail=str(exc))

    # ------------------------------------------------------------------
    # 12. Compare matched records
    # ------------------------------------------------------------------
    @router.post("/compare")
    async def compare_records(
        body: CompareRecordsRequest,
        request: Request,
    ) -> JSONResponse:
        """Compare matched records field by field."""
        service = _get_service(request)
        try:
            result = service.compare_records(
                match_id=body.match_id,
                record_a=body.record_a,
                record_b=body.record_b,
                fields=body.fields,
                tolerance_pct=body.tolerance_pct,
                tolerance_abs=body.tolerance_abs,
            )
            return JSONResponse(
                content={"status": "ok", "data": result},
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except Exception as exc:
            logger.error(
                "Compare records failed: %s", exc, exc_info=True,
            )
            raise HTTPException(status_code=500, detail=str(exc))

    # ------------------------------------------------------------------
    # 13. List discrepancies
    # ------------------------------------------------------------------
    @router.get("/discrepancies")
    async def list_discrepancies(
        severity: Optional[str] = Query(
            None, description="Filter by severity (critical, high, "
            "medium, low, info)",
        ),
        status: Optional[str] = Query(
            None, description="Filter by status (open, resolved, "
            "dismissed)",
        ),
        limit: int = Query(50, ge=1, le=1000),
        offset: int = Query(0, ge=0),
        request: Request = None,  # type: ignore[assignment]
    ) -> JSONResponse:
        """List detected discrepancies with optional filters."""
        service = _get_service(request)
        try:
            result = service.list_discrepancies(
                severity=severity,
                status=status,
                limit=limit,
                offset=offset,
            )
            return JSONResponse(
                content={"status": "ok", "data": result},
            )
        except Exception as exc:
            logger.error(
                "List discrepancies failed: %s", exc, exc_info=True,
            )
            raise HTTPException(status_code=500, detail=str(exc))

    # ------------------------------------------------------------------
    # 14. Get discrepancy details
    # ------------------------------------------------------------------
    @router.get("/discrepancies/{discrepancy_id}")
    async def get_discrepancy(
        discrepancy_id: str,
        request: Request,
    ) -> JSONResponse:
        """Get a discrepancy by ID."""
        service = _get_service(request)
        try:
            disc = service.get_discrepancy(discrepancy_id)
            if disc is None:
                raise HTTPException(
                    status_code=404,
                    detail=f"Discrepancy {discrepancy_id} not found",
                )
            return JSONResponse(
                content={"status": "ok", "data": disc},
            )
        except HTTPException:
            raise
        except Exception as exc:
            logger.error(
                "Get discrepancy failed: %s", exc, exc_info=True,
            )
            raise HTTPException(status_code=500, detail=str(exc))

    # ------------------------------------------------------------------
    # 15. Resolve discrepancies
    # ------------------------------------------------------------------
    @router.post("/resolve")
    async def resolve_discrepancies(
        body: ResolveDiscrepanciesRequest,
        request: Request,
    ) -> JSONResponse:
        """Resolve discrepancies using a configurable strategy."""
        service = _get_service(request)
        try:
            result = service.resolve_discrepancies(
                discrepancy_ids=body.discrepancy_ids,
                strategy=body.strategy,
                source_priorities=body.source_priorities,
                manual_values=body.manual_values,
            )
            return JSONResponse(
                content={"status": "ok", "data": result},
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except Exception as exc:
            logger.error(
                "Resolve discrepancies failed: %s", exc, exc_info=True,
            )
            raise HTTPException(status_code=500, detail=str(exc))

    # ------------------------------------------------------------------
    # 16. List golden records
    # ------------------------------------------------------------------
    @router.get("/golden-records")
    async def list_golden_records(
        limit: int = Query(50, ge=1, le=1000),
        offset: int = Query(0, ge=0),
        request: Request = None,  # type: ignore[assignment]
    ) -> JSONResponse:
        """List assembled golden records with pagination."""
        service = _get_service(request)
        try:
            result = service.get_golden_records(
                limit=limit, offset=offset,
            )
            return JSONResponse(
                content={"status": "ok", "data": result},
            )
        except Exception as exc:
            logger.error(
                "List golden records failed: %s", exc, exc_info=True,
            )
            raise HTTPException(status_code=500, detail=str(exc))

    # ------------------------------------------------------------------
    # 17. Get golden record details
    # ------------------------------------------------------------------
    @router.get("/golden-records/{record_id}")
    async def get_golden_record(
        record_id: str,
        request: Request,
    ) -> JSONResponse:
        """Get a golden record by ID."""
        service = _get_service(request)
        try:
            record = service.get_golden_record(record_id)
            if record is None:
                raise HTTPException(
                    status_code=404,
                    detail=f"Golden record {record_id} not found",
                )
            return JSONResponse(
                content={"status": "ok", "data": record},
            )
        except HTTPException:
            raise
        except Exception as exc:
            logger.error(
                "Get golden record failed: %s", exc, exc_info=True,
            )
            raise HTTPException(status_code=500, detail=str(exc))

    # ------------------------------------------------------------------
    # 18. Run full reconciliation pipeline
    # ------------------------------------------------------------------
    @router.post("/pipeline")
    async def run_pipeline(
        body: RunPipelineRequest,
        request: Request,
    ) -> JSONResponse:
        """Run the full reconciliation pipeline end-to-end."""
        service = _get_service(request)
        try:
            result = service.run_pipeline(
                source_ids=body.source_ids,
                records_a=body.records_a,
                records_b=body.records_b,
                match_keys=body.match_keys,
                match_threshold=body.match_threshold,
                tolerance_pct=body.tolerance_pct,
                tolerance_abs=body.tolerance_abs,
                resolution_strategy=body.resolution_strategy,
                generate_golden_records=body.generate_golden_records,
            )
            return JSONResponse(
                content={"status": "ok", "data": result},
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except Exception as exc:
            logger.error(
                "Run pipeline failed: %s", exc, exc_info=True,
            )
            raise HTTPException(status_code=500, detail=str(exc))

    # ------------------------------------------------------------------
    # 19. Health check
    # ------------------------------------------------------------------
    @router.get("/health")
    async def health_check(
        request: Request,
    ) -> JSONResponse:
        """Get reconciliation service health status."""
        service = _get_service(request)
        try:
            result = service.get_health()
            return JSONResponse(
                content={"status": "ok", "data": result},
            )
        except Exception as exc:
            logger.error("Health check failed: %s", exc, exc_info=True)
            raise HTTPException(status_code=500, detail=str(exc))

    # ------------------------------------------------------------------
    # 20. Statistics
    # ------------------------------------------------------------------
    @router.get("/stats")
    async def get_stats(
        request: Request,
    ) -> JSONResponse:
        """Get aggregate reconciliation service statistics."""
        service = _get_service(request)
        try:
            result = service.get_statistics()
            return JSONResponse(
                content={"status": "ok", "data": result},
            )
        except Exception as exc:
            logger.error("Get stats failed: %s", exc, exc_info=True)
            raise HTTPException(status_code=500, detail=str(exc))
