# -*- coding: utf-8 -*-
"""
Data Freshness Monitor REST API Router - AGENT-DATA-016

FastAPI router providing 20 endpoints for data freshness monitoring:
dataset registration, SLA definition, freshness checking, batch checking,
SLA breach management, alert listing, refresh predictions, pipeline
orchestration, and health/statistics.

All endpoints are mounted under ``/api/v1/freshness``.

Endpoints:
    1.  POST   /datasets                  - Register dataset
    2.  GET    /datasets                  - List datasets
    3.  GET    /datasets/{id}             - Get dataset details
    4.  PUT    /datasets/{id}             - Update dataset metadata
    5.  DELETE /datasets/{id}             - Remove dataset from monitoring
    6.  POST   /sla                       - Create SLA definition
    7.  GET    /sla                       - List SLA definitions
    8.  GET    /sla/{id}                  - Get SLA details
    9.  PUT    /sla/{id}                  - Update SLA definition
    10. POST   /check                     - Run freshness check
    11. POST   /check/batch               - Run batch freshness check
    12. GET    /checks                    - List check results
    13. GET    /breaches                  - List SLA breaches
    14. GET    /breaches/{id}             - Get breach details
    15. PUT    /breaches/{id}             - Update breach status
    16. GET    /alerts                    - List alerts
    17. GET    /predictions               - Get refresh predictions
    18. POST   /pipeline                  - Run full monitoring pipeline
    19. GET    /health                    - Health check
    20. GET    /stats                     - Service statistics

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-016 Data Freshness Monitor (GL-DATA-X-019)
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
        "FastAPI not available; data freshness monitor router is None"
    )


# ---------------------------------------------------------------------------
# Pydantic request/response models (only when FastAPI is available)
# ---------------------------------------------------------------------------

if FASTAPI_AVAILABLE:

    # === Request Bodies ===

    class RegisterDatasetRequest(BaseModel):
        """Request body for registering a dataset."""
        name: str = Field(
            ..., description="Human-readable dataset name",
        )
        source: str = Field(
            default="", description="Source system identifier",
        )
        owner: str = Field(
            default="", description="Dataset owner (team or individual)",
        )
        refresh_cadence: str = Field(
            default="daily",
            description="Expected refresh frequency (realtime, hourly, "
            "daily, weekly, monthly, quarterly, annual)",
        )
        priority: int = Field(
            default=5, ge=1, le=10,
            description="Dataset priority (1=highest, 10=lowest)",
        )
        tags: List[str] = Field(
            default_factory=list,
            description="Tags for grouping and filtering",
        )
        metadata: Optional[Dict[str, Any]] = Field(
            None, description="Additional dataset metadata",
        )

    class UpdateDatasetRequest(BaseModel):
        """Request body for updating a dataset."""
        name: Optional[str] = Field(
            None, description="New dataset name",
        )
        source: Optional[str] = Field(
            None, description="New source identifier",
        )
        owner: Optional[str] = Field(
            None, description="New owner",
        )
        refresh_cadence: Optional[str] = Field(
            None, description="New refresh cadence",
        )
        priority: Optional[int] = Field(
            None, ge=1, le=10,
            description="New priority",
        )
        tags: Optional[List[str]] = Field(
            None, description="Replacement tags",
        )
        metadata: Optional[Dict[str, Any]] = Field(
            None, description="Additional metadata to merge",
        )
        status: Optional[str] = Field(
            None,
            description="New status (active, inactive, archived)",
        )

    class CreateSLARequest(BaseModel):
        """Request body for creating an SLA definition."""
        dataset_id: str = Field(
            default="",
            description="Dataset this SLA applies to (empty for default)",
        )
        name: str = Field(
            default="", description="Human-readable SLA name",
        )
        warning_hours: float = Field(
            default=24.0, ge=0.0,
            description="Hours before warning-level alert fires",
        )
        critical_hours: float = Field(
            default=72.0, ge=0.0,
            description="Hours before critical-level alert fires",
        )
        severity: str = Field(
            default="high",
            description="Default breach severity classification "
            "(info, low, medium, high, critical)",
        )
        escalation_policy: Optional[Dict[str, Any]] = Field(
            None, description="Escalation chain configuration",
        )
        metadata: Optional[Dict[str, Any]] = Field(
            None, description="Additional SLA metadata",
        )

    class UpdateSLARequest(BaseModel):
        """Request body for updating an SLA definition."""
        name: Optional[str] = Field(
            None, description="New SLA name",
        )
        warning_hours: Optional[float] = Field(
            None, ge=0.0,
            description="New warning threshold in hours",
        )
        critical_hours: Optional[float] = Field(
            None, ge=0.0,
            description="New critical threshold in hours",
        )
        severity: Optional[str] = Field(
            None, description="New severity classification",
        )
        escalation_policy: Optional[Dict[str, Any]] = Field(
            None, description="New escalation policy",
        )
        metadata: Optional[Dict[str, Any]] = Field(
            None, description="Additional metadata to merge",
        )
        status: Optional[str] = Field(
            None, description="New status (active, inactive)",
        )

    class RunCheckRequest(BaseModel):
        """Request body for running a freshness check."""
        dataset_id: str = Field(
            ..., description="Dataset to check",
        )
        last_refreshed_at: Optional[str] = Field(
            None,
            description="ISO timestamp of last refresh (uses stored "
            "metadata if not provided)",
        )

    class RunBatchCheckRequest(BaseModel):
        """Request body for running a batch freshness check."""
        dataset_ids: Optional[List[str]] = Field(
            None,
            description="Dataset IDs to check (all if not provided)",
        )

    class UpdateBreachRequest(BaseModel):
        """Request body for updating a breach status."""
        status: Optional[str] = Field(
            None,
            description="New status (detected, acknowledged, "
            "investigating, resolved)",
        )
        resolution_notes: Optional[str] = Field(
            None, description="Resolution notes",
        )
        metadata: Optional[Dict[str, Any]] = Field(
            None, description="Additional metadata to merge",
        )

    class RunPipelineRequest(BaseModel):
        """Request body for running the full monitoring pipeline."""
        dataset_ids: Optional[List[str]] = Field(
            None,
            description="Dataset IDs to monitor (all if not provided)",
        )
        run_predictions: bool = Field(
            default=True,
            description="Whether to run refresh predictions",
        )
        generate_alerts: bool = Field(
            default=True,
            description="Whether to generate alerts for breaches",
        )

    # === Response Models ===

    class DatasetResponse(BaseModel):
        """Response model for a registered dataset."""
        dataset_id: str = Field(default="")
        name: str = Field(default="")
        source: str = Field(default="")
        owner: str = Field(default="")
        refresh_cadence: str = Field(default="daily")
        priority: int = Field(default=5)
        tags: List[str] = Field(default_factory=list)
        status: str = Field(default="active")
        last_refreshed_at: Optional[str] = Field(default=None)
        last_checked_at: Optional[str] = Field(default=None)
        freshness_score: Optional[float] = Field(default=None)
        freshness_level: Optional[str] = Field(default=None)
        sla_status: str = Field(default="unknown")
        created_at: str = Field(default="")
        updated_at: str = Field(default="")
        provenance_hash: str = Field(default="")

    class SLAResponse(BaseModel):
        """Response model for an SLA definition."""
        sla_id: str = Field(default="")
        dataset_id: str = Field(default="")
        name: str = Field(default="")
        warning_hours: float = Field(default=24.0)
        critical_hours: float = Field(default=72.0)
        severity: str = Field(default="high")
        status: str = Field(default="active")
        created_at: str = Field(default="")
        updated_at: str = Field(default="")
        provenance_hash: str = Field(default="")

    class CheckResponse(BaseModel):
        """Response model for a freshness check result."""
        check_id: str = Field(default="")
        dataset_id: str = Field(default="")
        checked_at: str = Field(default="")
        age_hours: float = Field(default=0.0)
        freshness_score: float = Field(default=0.0)
        freshness_level: str = Field(default="")
        sla_status: str = Field(default="")
        processing_time_ms: float = Field(default=0.0)
        provenance_hash: str = Field(default="")

    class BreachResponse(BaseModel):
        """Response model for an SLA breach."""
        breach_id: str = Field(default="")
        dataset_id: str = Field(default="")
        sla_id: str = Field(default="")
        severity: str = Field(default="warning")
        age_hours: float = Field(default=0.0)
        threshold_hours: float = Field(default=0.0)
        status: str = Field(default="detected")
        detected_at: str = Field(default="")
        resolved_at: Optional[str] = Field(default=None)
        provenance_hash: str = Field(default="")

    class AlertResponse(BaseModel):
        """Response model for a generated alert."""
        alert_id: str = Field(default="")
        breach_id: str = Field(default="")
        dataset_id: str = Field(default="")
        severity: str = Field(default="warning")
        message: str = Field(default="")
        status: str = Field(default="open")
        created_at: str = Field(default="")
        acknowledged_at: Optional[str] = Field(default=None)
        resolved_at: Optional[str] = Field(default=None)
        provenance_hash: str = Field(default="")

    class PredictionResponse(BaseModel):
        """Response model for a refresh prediction."""
        prediction_id: str = Field(default="")
        dataset_id: str = Field(default="")
        predicted_refresh_at: Optional[str] = Field(default=None)
        confidence: float = Field(default=0.0)
        provenance_hash: str = Field(default="")

    class HealthResponse(BaseModel):
        """Response model for health check."""
        status: str = Field(default="healthy")
        service: str = Field(default="data_freshness_monitor")
        engines: Dict[str, bool] = Field(default_factory=dict)
        stores: Dict[str, int] = Field(default_factory=dict)
        timestamp: str = Field(default="")

    class StatsResponse(BaseModel):
        """Response model for aggregate statistics."""
        total_datasets: int = Field(default=0)
        total_sla_definitions: int = Field(default=0)
        total_checks: int = Field(default=0)
        total_breaches: int = Field(default=0)
        total_alerts: int = Field(default=0)
        total_predictions: int = Field(default=0)
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
        prefix="/api/v1/freshness",
        tags=["freshness"],
    )
else:
    router = None  # type: ignore[assignment]


def _get_service(request: Request) -> Any:
    """Extract DataFreshnessMonitorService from app state.

    Args:
        request: FastAPI request object.

    Returns:
        DataFreshnessMonitorService instance.

    Raises:
        HTTPException: If service is not configured.
    """
    service = getattr(
        request.app.state, "data_freshness_monitor_service", None,
    )
    if service is None:
        raise HTTPException(
            status_code=503,
            detail="Data freshness monitor service not configured",
        )
    return service


if FASTAPI_AVAILABLE:

    # ------------------------------------------------------------------
    # 1. Register dataset
    # ------------------------------------------------------------------
    @router.post("/datasets")
    async def register_dataset(
        body: RegisterDatasetRequest,
        request: Request,
    ) -> JSONResponse:
        """Register a new dataset for freshness monitoring."""
        service = _get_service(request)
        try:
            result = service.register_dataset(
                name=body.name,
                source=body.source,
                owner=body.owner,
                refresh_cadence=body.refresh_cadence,
                priority=body.priority,
                tags=body.tags,
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
                "Register dataset failed: %s", exc, exc_info=True,
            )
            raise HTTPException(status_code=500, detail=str(exc))

    # ------------------------------------------------------------------
    # 2. List datasets
    # ------------------------------------------------------------------
    @router.get("/datasets")
    async def list_datasets(
        status: Optional[str] = Query(
            None, description="Filter by status (active, inactive, archived)",
        ),
        source: Optional[str] = Query(
            None, description="Filter by source",
        ),
        limit: int = Query(50, ge=1, le=1000),
        offset: int = Query(0, ge=0),
        request: Request = None,  # type: ignore[assignment]
    ) -> JSONResponse:
        """List registered datasets with optional filters."""
        service = _get_service(request)
        try:
            result = service.list_datasets(
                status=status, source=source,
                limit=limit, offset=offset,
            )
            return JSONResponse(
                content={"status": "ok", "data": result},
            )
        except Exception as exc:
            logger.error(
                "List datasets failed: %s", exc, exc_info=True,
            )
            raise HTTPException(status_code=500, detail=str(exc))

    # ------------------------------------------------------------------
    # 3. Get dataset details
    # ------------------------------------------------------------------
    @router.get("/datasets/{dataset_id}")
    async def get_dataset(
        dataset_id: str,
        request: Request,
    ) -> JSONResponse:
        """Get a registered dataset by ID."""
        service = _get_service(request)
        try:
            dataset = service.get_dataset(dataset_id)
            if dataset is None:
                raise HTTPException(
                    status_code=404,
                    detail=f"Dataset {dataset_id} not found",
                )
            return JSONResponse(
                content={"status": "ok", "data": dataset},
            )
        except HTTPException:
            raise
        except Exception as exc:
            logger.error(
                "Get dataset failed: %s", exc, exc_info=True,
            )
            raise HTTPException(status_code=500, detail=str(exc))

    # ------------------------------------------------------------------
    # 4. Update dataset metadata
    # ------------------------------------------------------------------
    @router.put("/datasets/{dataset_id}")
    async def update_dataset(
        dataset_id: str,
        body: UpdateDatasetRequest,
        request: Request,
    ) -> JSONResponse:
        """Update a registered dataset."""
        service = _get_service(request)
        try:
            result = service.update_dataset(
                dataset_id=dataset_id,
                name=body.name,
                source=body.source,
                owner=body.owner,
                refresh_cadence=body.refresh_cadence,
                priority=body.priority,
                tags=body.tags,
                metadata=body.metadata,
                status=body.status,
            )
            return JSONResponse(
                content={"status": "ok", "data": result},
            )
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        except Exception as exc:
            logger.error(
                "Update dataset failed: %s", exc, exc_info=True,
            )
            raise HTTPException(status_code=500, detail=str(exc))

    # ------------------------------------------------------------------
    # 5. Remove dataset from monitoring
    # ------------------------------------------------------------------
    @router.delete("/datasets/{dataset_id}")
    async def delete_dataset(
        dataset_id: str,
        request: Request,
    ) -> JSONResponse:
        """Remove a dataset from monitoring."""
        service = _get_service(request)
        try:
            result = service.delete_dataset(dataset_id)
            return JSONResponse(
                content={"status": "ok", "data": result},
            )
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        except Exception as exc:
            logger.error(
                "Delete dataset failed: %s", exc, exc_info=True,
            )
            raise HTTPException(status_code=500, detail=str(exc))

    # ------------------------------------------------------------------
    # 6. Create SLA definition
    # ------------------------------------------------------------------
    @router.post("/sla")
    async def create_sla(
        body: CreateSLARequest,
        request: Request,
    ) -> JSONResponse:
        """Create a new SLA definition."""
        service = _get_service(request)
        try:
            result = service.create_sla(
                dataset_id=body.dataset_id,
                name=body.name,
                warning_hours=body.warning_hours,
                critical_hours=body.critical_hours,
                severity=body.severity,
                escalation_policy=body.escalation_policy,
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
                "Create SLA failed: %s", exc, exc_info=True,
            )
            raise HTTPException(status_code=500, detail=str(exc))

    # ------------------------------------------------------------------
    # 7. List SLA definitions
    # ------------------------------------------------------------------
    @router.get("/sla")
    async def list_slas(
        dataset_id: Optional[str] = Query(
            None, description="Filter by dataset ID",
        ),
        limit: int = Query(50, ge=1, le=1000),
        offset: int = Query(0, ge=0),
        request: Request = None,  # type: ignore[assignment]
    ) -> JSONResponse:
        """List SLA definitions with optional dataset filter."""
        service = _get_service(request)
        try:
            result = service.list_slas(
                dataset_id=dataset_id,
                limit=limit, offset=offset,
            )
            return JSONResponse(
                content={"status": "ok", "data": result},
            )
        except Exception as exc:
            logger.error(
                "List SLAs failed: %s", exc, exc_info=True,
            )
            raise HTTPException(status_code=500, detail=str(exc))

    # ------------------------------------------------------------------
    # 8. Get SLA details
    # ------------------------------------------------------------------
    @router.get("/sla/{sla_id}")
    async def get_sla(
        sla_id: str,
        request: Request,
    ) -> JSONResponse:
        """Get an SLA definition by ID."""
        service = _get_service(request)
        try:
            sla = service.get_sla(sla_id)
            if sla is None:
                raise HTTPException(
                    status_code=404,
                    detail=f"SLA {sla_id} not found",
                )
            return JSONResponse(
                content={"status": "ok", "data": sla},
            )
        except HTTPException:
            raise
        except Exception as exc:
            logger.error(
                "Get SLA failed: %s", exc, exc_info=True,
            )
            raise HTTPException(status_code=500, detail=str(exc))

    # ------------------------------------------------------------------
    # 9. Update SLA definition
    # ------------------------------------------------------------------
    @router.put("/sla/{sla_id}")
    async def update_sla(
        sla_id: str,
        body: UpdateSLARequest,
        request: Request,
    ) -> JSONResponse:
        """Update an SLA definition."""
        service = _get_service(request)
        try:
            result = service.update_sla(
                sla_id=sla_id,
                name=body.name,
                warning_hours=body.warning_hours,
                critical_hours=body.critical_hours,
                severity=body.severity,
                escalation_policy=body.escalation_policy,
                metadata=body.metadata,
                status=body.status,
            )
            return JSONResponse(
                content={"status": "ok", "data": result},
            )
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        except Exception as exc:
            logger.error(
                "Update SLA failed: %s", exc, exc_info=True,
            )
            raise HTTPException(status_code=500, detail=str(exc))

    # ------------------------------------------------------------------
    # 10. Run freshness check
    # ------------------------------------------------------------------
    @router.post("/check")
    async def run_check(
        body: RunCheckRequest,
        request: Request,
    ) -> JSONResponse:
        """Run a freshness check on a single dataset."""
        service = _get_service(request)
        try:
            result = service.run_check(
                dataset_id=body.dataset_id,
                last_refreshed_at=body.last_refreshed_at,
            )
            return JSONResponse(
                content={"status": "ok", "data": result},
            )
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        except Exception as exc:
            logger.error(
                "Run check failed: %s", exc, exc_info=True,
            )
            raise HTTPException(status_code=500, detail=str(exc))

    # ------------------------------------------------------------------
    # 11. Run batch freshness check
    # ------------------------------------------------------------------
    @router.post("/check/batch")
    async def run_batch_check(
        body: RunBatchCheckRequest,
        request: Request,
    ) -> JSONResponse:
        """Run freshness checks on multiple datasets."""
        service = _get_service(request)
        try:
            result = service.run_batch_check(
                dataset_ids=body.dataset_ids,
            )
            return JSONResponse(
                content={"status": "ok", "data": result},
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except Exception as exc:
            logger.error(
                "Run batch check failed: %s", exc, exc_info=True,
            )
            raise HTTPException(status_code=500, detail=str(exc))

    # ------------------------------------------------------------------
    # 12. List check results
    # ------------------------------------------------------------------
    @router.get("/checks")
    async def list_checks(
        dataset_id: Optional[str] = Query(
            None, description="Filter by dataset ID",
        ),
        limit: int = Query(50, ge=1, le=1000),
        offset: int = Query(0, ge=0),
        request: Request = None,  # type: ignore[assignment]
    ) -> JSONResponse:
        """List freshness check results with optional dataset filter."""
        service = _get_service(request)
        try:
            result = service.list_checks(
                dataset_id=dataset_id,
                limit=limit, offset=offset,
            )
            return JSONResponse(
                content={"status": "ok", "data": result},
            )
        except Exception as exc:
            logger.error(
                "List checks failed: %s", exc, exc_info=True,
            )
            raise HTTPException(status_code=500, detail=str(exc))

    # ------------------------------------------------------------------
    # 13. List SLA breaches
    # ------------------------------------------------------------------
    @router.get("/breaches")
    async def list_breaches(
        severity: Optional[str] = Query(
            None,
            description="Filter by severity (info, low, medium, "
            "high, critical, warning)",
        ),
        status: Optional[str] = Query(
            None,
            description="Filter by status (detected, acknowledged, "
            "investigating, resolved)",
        ),
        limit: int = Query(50, ge=1, le=1000),
        offset: int = Query(0, ge=0),
        request: Request = None,  # type: ignore[assignment]
    ) -> JSONResponse:
        """List SLA breaches with optional filters."""
        service = _get_service(request)
        try:
            result = service.list_breaches(
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
                "List breaches failed: %s", exc, exc_info=True,
            )
            raise HTTPException(status_code=500, detail=str(exc))

    # ------------------------------------------------------------------
    # 14. Get breach details
    # ------------------------------------------------------------------
    @router.get("/breaches/{breach_id}")
    async def get_breach(
        breach_id: str,
        request: Request,
    ) -> JSONResponse:
        """Get an SLA breach by ID."""
        service = _get_service(request)
        try:
            breach = service.get_breach(breach_id)
            if breach is None:
                raise HTTPException(
                    status_code=404,
                    detail=f"Breach {breach_id} not found",
                )
            return JSONResponse(
                content={"status": "ok", "data": breach},
            )
        except HTTPException:
            raise
        except Exception as exc:
            logger.error(
                "Get breach failed: %s", exc, exc_info=True,
            )
            raise HTTPException(status_code=500, detail=str(exc))

    # ------------------------------------------------------------------
    # 15. Update breach status
    # ------------------------------------------------------------------
    @router.put("/breaches/{breach_id}")
    async def update_breach(
        breach_id: str,
        body: UpdateBreachRequest,
        request: Request,
    ) -> JSONResponse:
        """Update an SLA breach status."""
        service = _get_service(request)
        try:
            result = service.update_breach(
                breach_id=breach_id,
                status=body.status,
                resolution_notes=body.resolution_notes,
                metadata=body.metadata,
            )
            return JSONResponse(
                content={"status": "ok", "data": result},
            )
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        except Exception as exc:
            logger.error(
                "Update breach failed: %s", exc, exc_info=True,
            )
            raise HTTPException(status_code=500, detail=str(exc))

    # ------------------------------------------------------------------
    # 16. List alerts
    # ------------------------------------------------------------------
    @router.get("/alerts")
    async def list_alerts(
        severity: Optional[str] = Query(
            None, description="Filter by severity",
        ),
        status: Optional[str] = Query(
            None,
            description="Filter by status (open, acknowledged, resolved)",
        ),
        limit: int = Query(50, ge=1, le=1000),
        offset: int = Query(0, ge=0),
        request: Request = None,  # type: ignore[assignment]
    ) -> JSONResponse:
        """List generated alerts with optional filters."""
        service = _get_service(request)
        try:
            result = service.list_alerts(
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
                "List alerts failed: %s", exc, exc_info=True,
            )
            raise HTTPException(status_code=500, detail=str(exc))

    # ------------------------------------------------------------------
    # 17. Get refresh predictions
    # ------------------------------------------------------------------
    @router.get("/predictions")
    async def get_predictions(
        dataset_id: Optional[str] = Query(
            None, description="Filter by dataset ID",
        ),
        limit: int = Query(50, ge=1, le=1000),
        offset: int = Query(0, ge=0),
        request: Request = None,  # type: ignore[assignment]
    ) -> JSONResponse:
        """Get refresh predictions with optional dataset filter."""
        service = _get_service(request)
        try:
            result = service.get_predictions(
                dataset_id=dataset_id,
                limit=limit,
                offset=offset,
            )
            return JSONResponse(
                content={"status": "ok", "data": result},
            )
        except Exception as exc:
            logger.error(
                "Get predictions failed: %s", exc, exc_info=True,
            )
            raise HTTPException(status_code=500, detail=str(exc))

    # ------------------------------------------------------------------
    # 18. Run full monitoring pipeline
    # ------------------------------------------------------------------
    @router.post("/pipeline")
    async def run_pipeline(
        body: RunPipelineRequest,
        request: Request,
    ) -> JSONResponse:
        """Run the full freshness monitoring pipeline end-to-end."""
        service = _get_service(request)
        try:
            result = service.run_pipeline(
                dataset_ids=body.dataset_ids,
                run_predictions=body.run_predictions,
                generate_alerts=body.generate_alerts,
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
        """Get freshness monitor service health status."""
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
        """Get aggregate freshness monitor service statistics."""
        service = _get_service(request)
        try:
            result = service.get_statistics()
            return JSONResponse(
                content={"status": "ok", "data": result},
            )
        except Exception as exc:
            logger.error("Get stats failed: %s", exc, exc_info=True)
            raise HTTPException(status_code=500, detail=str(exc))
