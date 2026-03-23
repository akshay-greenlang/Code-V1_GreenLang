# -*- coding: utf-8 -*-
"""
Outlier Detection REST API Router - AGENT-DATA-013

FastAPI router providing 20 endpoints for outlier detection,
classification, treatment, threshold management, feedback,
pipeline execution, impact analysis, job management, and
health/statistics.

All endpoints are mounted under ``/api/v1/outlier``.

Endpoints:
    1.  POST   /jobs                       - Create detection job
    2.  GET    /jobs                       - List jobs
    3.  GET    /jobs/{job_id}              - Get job details
    4.  DELETE /jobs/{job_id}              - Delete job
    5.  POST   /detect                    - Detect outliers (single column)
    6.  POST   /detect/batch              - Batch detect (multiple columns)
    7.  GET    /detections                - List detections
    8.  GET    /detections/{detection_id} - Get detection result
    9.  POST   /classify                  - Classify outliers
    10. GET    /classify/{id}             - Get classification result
    11. POST   /treat                     - Apply treatment
    12. GET    /treat/{treatment_id}      - Get treatment result
    13. POST   /treat/{treatment_id}/undo - Undo treatment
    14. POST   /thresholds                - Create threshold
    15. GET    /thresholds                - List thresholds
    16. POST   /feedback                  - Submit feedback
    17. POST   /impact                    - Analyze impact
    18. POST   /pipeline                  - Run full pipeline
    19. GET    /health                    - Health check
    20. GET    /stats                     - Statistics

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-013 Outlier Detection (GL-DATA-X-016)
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
        "FastAPI not available; outlier detector router is None"
    )


# ---------------------------------------------------------------------------
# Pydantic request/response models (only when FastAPI is available)
# ---------------------------------------------------------------------------

if FASTAPI_AVAILABLE:

    class CreateJobBody(BaseModel):
        """Request body for creating a detection job."""
        dataset_id: str = Field(
            default="", description="Identifier of the dataset to analyze",
        )
        records: List[Dict[str, Any]] = Field(
            ..., description="List of record dicts to analyze",
        )
        pipeline_config: Optional[Dict[str, Any]] = Field(
            None, description="Optional pipeline configuration",
        )

    class DetectBody(BaseModel):
        """Request body for detecting outliers in a single column."""
        records: List[Dict[str, Any]] = Field(
            ..., description="List of record dicts to analyze",
        )
        column: str = Field(
            ..., description="Column name to analyze for outliers",
        )
        methods: Optional[List[str]] = Field(
            None,
            description="Detection methods to use (iqr, zscore, "
            "modified_zscore, mad, grubbs, tukey, percentile, "
            "lof, isolation_forest, mahalanobis, dbscan)",
        )
        options: Optional[Dict[str, Any]] = Field(
            None, description="Additional detection options",
        )

    class BatchDetectBody(BaseModel):
        """Request body for batch detection across columns."""
        records: List[Dict[str, Any]] = Field(
            ..., description="List of record dicts to analyze",
        )
        columns: Optional[List[str]] = Field(
            None,
            description="Columns to analyze (auto-detect numeric if omitted)",
        )

    class ClassifyBody(BaseModel):
        """Request body for classifying detected outliers."""
        detections: List[Dict[str, Any]] = Field(
            ..., description="List of outlier detection score dicts",
        )
        records: List[Dict[str, Any]] = Field(
            ..., description="Original record dicts for context",
        )

    class TreatBody(BaseModel):
        """Request body for treating detected outliers."""
        records: List[Dict[str, Any]] = Field(
            ..., description="Original record dicts",
        )
        detections: List[Dict[str, Any]] = Field(
            ..., description="List of outlier detection score dicts",
        )
        strategy: str = Field(
            default="flag",
            description="Treatment strategy (cap, winsorize, flag, "
            "remove, replace, investigate)",
        )
        options: Optional[Dict[str, Any]] = Field(
            None, description="Additional treatment options",
        )

    class CreateThresholdBody(BaseModel):
        """Request body for creating a domain threshold."""
        column: str = Field(
            ..., description="Column name this threshold applies to",
        )
        min_val: Optional[float] = Field(
            None, description="Lower acceptable bound",
        )
        max_val: Optional[float] = Field(
            None, description="Upper acceptable bound",
        )
        source: str = Field(
            default="domain",
            description="Source of threshold (domain, statistical, "
            "regulatory, custom, learned)",
        )
        context: str = Field(
            default="",
            description="Additional description or context",
        )

    class SubmitFeedbackBody(BaseModel):
        """Request body for submitting feedback."""
        detection_id: str = Field(
            ..., description="Identifier of the detection being reviewed",
        )
        feedback_type: str = Field(
            default="confirmed_outlier",
            description="Feedback type (confirmed_outlier, false_positive, "
            "reclassified, unknown)",
        )
        notes: str = Field(
            default="",
            description="Human notes or justification",
        )

    class ImpactBody(BaseModel):
        """Request body for impact analysis."""
        original: List[Dict[str, Any]] = Field(
            ..., description="Original record dicts before treatment",
        )
        treated: List[Dict[str, Any]] = Field(
            ..., description="Record dicts after treatment",
        )

    class RunPipelineBody(BaseModel):
        """Request body for running the full detection pipeline."""
        records: List[Dict[str, Any]] = Field(
            ..., description="List of record dicts to process",
        )
        config: Optional[Dict[str, Any]] = Field(
            None, description="Optional pipeline configuration overrides",
        )


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

if FASTAPI_AVAILABLE:
    router = APIRouter(
        prefix="/api/v1/outlier",
        tags=["Outlier Detection"],
    )
else:
    router = None  # type: ignore[assignment]


def _get_service(request: Request) -> Any:
    """Extract OutlierDetectorService from app state.

    Args:
        request: FastAPI request object.

    Returns:
        OutlierDetectorService instance.

    Raises:
        HTTPException: If service is not configured.
    """
    service = getattr(
        request.app.state, "outlier_detector_service", None,
    )
    if service is None:
        raise HTTPException(
            status_code=503,
            detail="Outlier detector service not configured",
        )
    return service


if FASTAPI_AVAILABLE:

    # ------------------------------------------------------------------
    # 1. Create detection job
    # ------------------------------------------------------------------
    @router.post("/jobs")
    async def create_job(
        body: CreateJobBody,
        request: Request,
    ) -> Dict[str, Any]:
        """Create a new outlier detection job."""
        service = _get_service(request)
        try:
            result = service.create_job(
                request={
                    "dataset_id": body.dataset_id,
                    "records": body.records,
                    "pipeline_config": body.pipeline_config,
                },
            )
            return result
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    # ------------------------------------------------------------------
    # 2. List jobs
    # ------------------------------------------------------------------
    @router.get("/jobs")
    async def list_jobs(
        status: Optional[str] = Query(None),
        limit: int = Query(50, ge=1, le=1000),
        offset: int = Query(0, ge=0),
        request: Request = None,  # type: ignore[assignment]
    ) -> Dict[str, Any]:
        """List outlier detection jobs with optional status filter."""
        service = _get_service(request)
        jobs = service.list_jobs(
            status=status,
            limit=limit,
            offset=offset,
        )
        return {
            "jobs": jobs,
            "count": len(jobs),
            "limit": limit,
            "offset": offset,
        }

    # ------------------------------------------------------------------
    # 3. Get job details
    # ------------------------------------------------------------------
    @router.get("/jobs/{job_id}")
    async def get_job(
        job_id: str,
        request: Request,
    ) -> Dict[str, Any]:
        """Get an outlier detection job by ID."""
        service = _get_service(request)
        job = service.get_job(job_id)
        if job is None:
            raise HTTPException(
                status_code=404,
                detail=f"Detection job {job_id} not found",
            )
        return job

    # ------------------------------------------------------------------
    # 4. Delete job
    # ------------------------------------------------------------------
    @router.delete("/jobs/{job_id}")
    async def delete_job(
        job_id: str,
        request: Request,
    ) -> Dict[str, Any]:
        """Delete (cancel) an outlier detection job."""
        service = _get_service(request)
        deleted = service.delete_job(job_id)
        if not deleted:
            raise HTTPException(
                status_code=404,
                detail=f"Detection job {job_id} not found",
            )
        return {"deleted": True, "job_id": job_id}

    # ------------------------------------------------------------------
    # 5. Detect outliers (single column)
    # ------------------------------------------------------------------
    @router.post("/detect")
    async def detect_outliers(
        body: DetectBody,
        request: Request,
    ) -> Dict[str, Any]:
        """Detect outliers in a single column."""
        service = _get_service(request)
        try:
            result = service.detect_outliers(
                records=body.records,
                column=body.column,
                methods=body.methods,
                options=body.options,
            )
            return result.model_dump(mode="json")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    # ------------------------------------------------------------------
    # 6. Batch detect (multiple columns)
    # ------------------------------------------------------------------
    @router.post("/detect/batch")
    async def detect_batch(
        body: BatchDetectBody,
        request: Request,
    ) -> Dict[str, Any]:
        """Detect outliers across multiple columns."""
        service = _get_service(request)
        try:
            result = service.detect_batch(
                records=body.records,
                columns=body.columns,
            )
            return result.model_dump(mode="json")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    # ------------------------------------------------------------------
    # 7. List detections
    # ------------------------------------------------------------------
    @router.get("/detections")
    async def list_detections(
        limit: int = Query(50, ge=1, le=1000),
        offset: int = Query(0, ge=0),
        request: Request = None,  # type: ignore[assignment]
    ) -> Dict[str, Any]:
        """List all stored detection results."""
        service = _get_service(request)
        detections = service.get_detections()
        sliced = detections[offset:offset + limit]
        return {
            "detections": [d.model_dump(mode="json") for d in sliced],
            "count": len(sliced),
            "total": len(detections),
            "limit": limit,
            "offset": offset,
        }

    # ------------------------------------------------------------------
    # 8. Get detection result
    # ------------------------------------------------------------------
    @router.get("/detections/{detection_id}")
    async def get_detection(
        detection_id: str,
        request: Request,
    ) -> Dict[str, Any]:
        """Get a detection result by ID."""
        service = _get_service(request)
        result = service.get_detection(detection_id)
        if result is None:
            raise HTTPException(
                status_code=404,
                detail=f"Detection {detection_id} not found",
            )
        return result.model_dump(mode="json")

    # ------------------------------------------------------------------
    # 9. Classify outliers
    # ------------------------------------------------------------------
    @router.post("/classify")
    async def classify_outliers(
        body: ClassifyBody,
        request: Request,
    ) -> Dict[str, Any]:
        """Classify detected outliers by root cause."""
        service = _get_service(request)
        try:
            result = service.classify_outliers(
                detections=body.detections,
                records=body.records,
            )
            return result.model_dump(mode="json")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    # ------------------------------------------------------------------
    # 10. Get classification result
    # ------------------------------------------------------------------
    @router.get("/classify/{classification_id}")
    async def get_classification(
        classification_id: str,
        request: Request,
    ) -> Dict[str, Any]:
        """Get a classification result by ID."""
        service = _get_service(request)
        result = service.get_classification(classification_id)
        if result is None:
            raise HTTPException(
                status_code=404,
                detail=f"Classification {classification_id} not found",
            )
        return result.model_dump(mode="json")

    # ------------------------------------------------------------------
    # 11. Apply treatment
    # ------------------------------------------------------------------
    @router.post("/treat")
    async def apply_treatment(
        body: TreatBody,
        request: Request,
    ) -> Dict[str, Any]:
        """Apply treatment to detected outliers."""
        service = _get_service(request)
        try:
            result = service.apply_treatment(
                records=body.records,
                detections=body.detections,
                strategy=body.strategy,
                options=body.options,
            )
            return result.model_dump(mode="json")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    # ------------------------------------------------------------------
    # 12. Get treatment result
    # ------------------------------------------------------------------
    @router.get("/treat/{treatment_id}")
    async def get_treatment(
        treatment_id: str,
        request: Request,
    ) -> Dict[str, Any]:
        """Get a treatment result by ID."""
        service = _get_service(request)
        result = service.get_treatment(treatment_id)
        if result is None:
            raise HTTPException(
                status_code=404,
                detail=f"Treatment {treatment_id} not found",
            )
        return result.model_dump(mode="json")

    # ------------------------------------------------------------------
    # 13. Undo treatment
    # ------------------------------------------------------------------
    @router.post("/treat/{treatment_id}/undo")
    async def undo_treatment(
        treatment_id: str,
        request: Request,
    ) -> Dict[str, Any]:
        """Undo a previously applied treatment."""
        service = _get_service(request)
        success = service.undo_treatment(treatment_id)
        if not success:
            raise HTTPException(
                status_code=404,
                detail=f"Treatment {treatment_id} not found or not reversible",
            )
        return {"undone": True, "treatment_id": treatment_id}

    # ------------------------------------------------------------------
    # 14. Create threshold
    # ------------------------------------------------------------------
    @router.post("/thresholds")
    async def create_threshold(
        body: CreateThresholdBody,
        request: Request,
    ) -> Dict[str, Any]:
        """Create a domain-specific detection threshold."""
        service = _get_service(request)
        try:
            result = service.create_threshold(
                column=body.column,
                min_val=body.min_val,
                max_val=body.max_val,
                source=body.source,
                context=body.context,
            )
            return result.model_dump(mode="json")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    # ------------------------------------------------------------------
    # 15. List thresholds
    # ------------------------------------------------------------------
    @router.get("/thresholds")
    async def list_thresholds(
        request: Request,
    ) -> Dict[str, Any]:
        """List all domain thresholds."""
        service = _get_service(request)
        thresholds = service.list_thresholds()
        return {
            "thresholds": [t.model_dump(mode="json") for t in thresholds],
            "count": len(thresholds),
        }

    # ------------------------------------------------------------------
    # 16. Submit feedback
    # ------------------------------------------------------------------
    @router.post("/feedback")
    async def submit_feedback(
        body: SubmitFeedbackBody,
        request: Request,
    ) -> Dict[str, Any]:
        """Submit human feedback on an outlier detection."""
        service = _get_service(request)
        result = service.submit_feedback(
            detection_id=body.detection_id,
            feedback_type=body.feedback_type,
            notes=body.notes,
        )
        return result.model_dump(mode="json")

    # ------------------------------------------------------------------
    # 17. Analyze impact
    # ------------------------------------------------------------------
    @router.post("/impact")
    async def analyze_impact(
        body: ImpactBody,
        request: Request,
    ) -> Dict[str, Any]:
        """Analyze the statistical impact of outlier treatment."""
        service = _get_service(request)
        try:
            result = service.analyze_impact(
                original=body.original,
                treated=body.treated,
            )
            return result
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    # ------------------------------------------------------------------
    # 18. Run full pipeline
    # ------------------------------------------------------------------
    @router.post("/pipeline")
    async def run_pipeline(
        body: RunPipelineBody,
        request: Request,
    ) -> Dict[str, Any]:
        """Run the full outlier detection pipeline."""
        service = _get_service(request)
        try:
            result = service.run_pipeline(
                records=body.records,
                config=body.config,
            )
            return result.model_dump(mode="json")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    # ------------------------------------------------------------------
    # 19. Health check
    # ------------------------------------------------------------------
    @router.get("/health")
    async def health_check(
        request: Request,
    ) -> Dict[str, Any]:
        """Get outlier detector service health status."""
        service = _get_service(request)
        return service.health_check()

    # ------------------------------------------------------------------
    # 20. Statistics
    # ------------------------------------------------------------------
    @router.get("/stats")
    async def get_stats(
        request: Request,
    ) -> Dict[str, Any]:
        """Get aggregate outlier detection statistics."""
        service = _get_service(request)
        stats = service.get_statistics()
        return stats.model_dump(mode="json")
