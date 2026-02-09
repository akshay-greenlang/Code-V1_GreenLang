# -*- coding: utf-8 -*-
"""
Data Quality Profiler REST API Router - AGENT-DATA-010

FastAPI router providing 20 endpoints for dataset profiling,
quality assessment, anomaly detection, freshness checking,
rule management, quality gates, trends, reporting, and health.

All endpoints are mounted under ``/api/v1/data-quality``.

Endpoints:
    1.  POST   /v1/profile                    - Profile a dataset
    2.  POST   /v1/profile/batch              - Batch profile datasets
    3.  GET    /v1/profiles                    - List profiles
    4.  GET    /v1/profiles/{profile_id}       - Get single profile
    5.  POST   /v1/assess                      - Assess dataset quality
    6.  POST   /v1/assess/batch                - Batch quality assessment
    7.  GET    /v1/assessments                  - List assessments
    8.  GET    /v1/assessments/{assessment_id}  - Get single assessment
    9.  POST   /v1/validate                    - Validate dataset with rules
    10. POST   /v1/detect-anomalies            - Detect anomalies
    11. GET    /v1/anomalies                    - List anomaly results
    12. POST   /v1/check-freshness             - Check dataset freshness
    13. POST   /v1/rules                        - Create quality rule
    14. GET    /v1/rules                        - List quality rules
    15. PUT    /v1/rules/{rule_id}              - Update quality rule
    16. DELETE /v1/rules/{rule_id}              - Delete quality rule
    17. POST   /v1/gates                        - Evaluate quality gate
    18. GET    /v1/trends                       - Get quality trends
    19. POST   /v1/reports                      - Generate report
    20. GET    /health                          - Health check

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-010 Data Quality Profiler (GL-DATA-X-013)
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
        "FastAPI not available; data quality profiler router is None"
    )


# ---------------------------------------------------------------------------
# Pydantic request/response models (only when FastAPI is available)
# ---------------------------------------------------------------------------

if FASTAPI_AVAILABLE:

    class ProfileDatasetBody(BaseModel):
        """Request body for profiling a dataset."""
        data: List[Dict[str, Any]] = Field(
            ..., description="List of row dicts representing the dataset",
        )
        dataset_name: str = Field(
            default="unnamed",
            description="Name of the dataset being profiled",
        )
        columns: Optional[List[str]] = Field(
            None, description="Optional list of columns to profile (all if omitted)",
        )
        source: str = Field(
            default="manual",
            description="Data source identifier",
        )

    class ProfileBatchBody(BaseModel):
        """Request body for batch profiling datasets."""
        datasets: List[Dict[str, Any]] = Field(
            ..., description="List of dicts with data, dataset_name, columns, source",
        )

    class AssessQualityBody(BaseModel):
        """Request body for assessing dataset quality."""
        data: List[Dict[str, Any]] = Field(
            ..., description="List of row dicts representing the dataset",
        )
        dataset_name: str = Field(
            default="unnamed",
            description="Name of the dataset",
        )
        dimensions: Optional[List[str]] = Field(
            None,
            description="Dimensions to assess (completeness, validity, consistency, timeliness, uniqueness, accuracy)",
        )

    class AssessBatchBody(BaseModel):
        """Request body for batch quality assessment."""
        datasets: List[Dict[str, Any]] = Field(
            ..., description="List of dicts with data, dataset_name, dimensions",
        )

    class ValidateDatasetBody(BaseModel):
        """Request body for validating a dataset."""
        data: List[Dict[str, Any]] = Field(
            ..., description="List of row dicts representing the dataset",
        )
        dataset_name: str = Field(
            default="unnamed",
            description="Name of the dataset",
        )
        rule_ids: Optional[List[str]] = Field(
            None, description="Optional list of rule IDs to evaluate (all active if omitted)",
        )

    class DetectAnomaliesBody(BaseModel):
        """Request body for anomaly detection."""
        data: List[Dict[str, Any]] = Field(
            ..., description="List of row dicts representing the dataset",
        )
        dataset_name: str = Field(
            default="unnamed",
            description="Name of the dataset",
        )
        columns: Optional[List[str]] = Field(
            None, description="Columns to analyse (all numeric if omitted)",
        )
        method: Optional[str] = Field(
            None,
            description="Detection method (iqr, zscore, percentile)",
        )

    class CheckFreshnessBody(BaseModel):
        """Request body for checking dataset freshness."""
        dataset_name: str = Field(
            ..., description="Name of the dataset",
        )
        last_updated: str = Field(
            ..., description="ISO 8601 timestamp of last data update",
        )
        sla_hours: Optional[float] = Field(
            None, description="SLA threshold in hours (uses config default if omitted)",
        )

    class CreateRuleBody(BaseModel):
        """Request body for creating a quality rule."""
        name: str = Field(
            ..., description="Rule display name",
        )
        rule_type: str = Field(
            ..., description="Rule type (not_null, unique, range, regex, custom, referential)",
        )
        column: str = Field(
            default="", description="Target column name",
        )
        operator: str = Field(
            default="eq", description="Comparison operator",
        )
        threshold: Optional[Any] = Field(
            None, description="Threshold value for the rule",
        )
        parameters: Optional[Dict[str, Any]] = Field(
            None, description="Additional rule parameters",
        )
        priority: int = Field(
            default=100, description="Rule priority (lower = higher priority)",
        )

    class UpdateRuleBody(BaseModel):
        """Request body for updating a quality rule."""
        name: Optional[str] = Field(
            None, description="New rule name",
        )
        rule_type: Optional[str] = Field(
            None, description="New rule type",
        )
        column: Optional[str] = Field(
            None, description="New column name",
        )
        operator: Optional[str] = Field(
            None, description="New operator",
        )
        threshold: Optional[Any] = Field(
            None, description="New threshold",
        )
        parameters: Optional[Dict[str, Any]] = Field(
            None, description="New parameters",
        )
        priority: Optional[int] = Field(
            None, description="New priority",
        )
        is_active: Optional[bool] = Field(
            None, description="New active status",
        )

    class EvaluateGateBody(BaseModel):
        """Request body for evaluating a quality gate."""
        conditions: List[Dict[str, Any]] = Field(
            ..., description="List of condition dicts with dimension, operator, threshold",
        )
        dimension_scores: Optional[Dict[str, float]] = Field(
            None, description="Dict of dimension name to score",
        )

    class GenerateReportBody(BaseModel):
        """Request body for generating a report."""
        dataset_name: Optional[str] = Field(
            None, description="Optional dataset name filter",
        )
        report_type: str = Field(
            default="scorecard",
            description="Report type (scorecard, detailed, executive, issues, anomaly)",
        )
        format: str = Field(
            default="json",
            description="Report format (json, markdown, html, text, csv)",
        )


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

if FASTAPI_AVAILABLE:
    router = APIRouter(
        prefix="/api/v1/data-quality",
        tags=["Data Quality Profiler"],
    )
else:
    router = None  # type: ignore[assignment]


def _get_service(request: Request) -> Any:
    """Extract DataQualityProfilerService from app state.

    Args:
        request: FastAPI request object.

    Returns:
        DataQualityProfilerService instance.

    Raises:
        HTTPException: If service is not configured.
    """
    service = getattr(
        request.app.state, "data_quality_profiler_service", None,
    )
    if service is None:
        raise HTTPException(
            status_code=503,
            detail="Data quality profiler service not configured",
        )
    return service


if FASTAPI_AVAILABLE:

    # ------------------------------------------------------------------
    # 1. Profile a dataset
    # ------------------------------------------------------------------
    @router.post("/v1/profile")
    async def profile_dataset(
        body: ProfileDatasetBody,
        request: Request,
    ) -> Dict[str, Any]:
        """Profile a dataset and generate column-level statistics."""
        service = _get_service(request)
        try:
            result = service.profile_dataset(
                data=body.data,
                dataset_name=body.dataset_name,
                columns=body.columns,
                source=body.source,
            )
            return result.model_dump(mode="json")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    # ------------------------------------------------------------------
    # 2. Batch profile datasets
    # ------------------------------------------------------------------
    @router.post("/v1/profile/batch")
    async def profile_dataset_batch(
        body: ProfileBatchBody,
        request: Request,
    ) -> Dict[str, Any]:
        """Profile multiple datasets in batch."""
        service = _get_service(request)
        results = service.profile_dataset_batch(datasets=body.datasets)
        return {
            "profiles": [r.model_dump(mode="json") for r in results],
            "count": len(results),
            "requested": len(body.datasets),
        }

    # ------------------------------------------------------------------
    # 3. List profiles
    # ------------------------------------------------------------------
    @router.get("/v1/profiles")
    async def list_profiles(
        limit: int = Query(50, ge=1, le=1000),
        offset: int = Query(0, ge=0),
        request: Request = None,  # type: ignore[assignment]
    ) -> Dict[str, Any]:
        """List dataset profiles."""
        service = _get_service(request)
        profiles = service.list_profiles(limit=limit, offset=offset)
        return {
            "profiles": [p.model_dump(mode="json") for p in profiles],
            "count": len(profiles),
            "limit": limit,
            "offset": offset,
        }

    # ------------------------------------------------------------------
    # 4. Get single profile
    # ------------------------------------------------------------------
    @router.get("/v1/profiles/{profile_id}")
    async def get_profile(
        profile_id: str,
        request: Request,
    ) -> Dict[str, Any]:
        """Get a dataset profile by ID."""
        service = _get_service(request)
        profile = service.get_profile(profile_id)
        if profile is None:
            raise HTTPException(
                status_code=404,
                detail=f"Profile {profile_id} not found",
            )
        return profile.model_dump(mode="json")

    # ------------------------------------------------------------------
    # 5. Assess dataset quality
    # ------------------------------------------------------------------
    @router.post("/v1/assess")
    async def assess_quality(
        body: AssessQualityBody,
        request: Request,
    ) -> Dict[str, Any]:
        """Assess the quality of a dataset across 6 dimensions."""
        service = _get_service(request)
        try:
            result = service.assess_quality(
                data=body.data,
                dataset_name=body.dataset_name,
                dimensions=body.dimensions,
            )
            return result.model_dump(mode="json")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    # ------------------------------------------------------------------
    # 6. Batch quality assessment
    # ------------------------------------------------------------------
    @router.post("/v1/assess/batch")
    async def assess_quality_batch(
        body: AssessBatchBody,
        request: Request,
    ) -> Dict[str, Any]:
        """Assess quality for multiple datasets in batch."""
        service = _get_service(request)
        results = service.assess_quality_batch(datasets=body.datasets)
        return {
            "assessments": [r.model_dump(mode="json") for r in results],
            "count": len(results),
            "requested": len(body.datasets),
        }

    # ------------------------------------------------------------------
    # 7. List assessments
    # ------------------------------------------------------------------
    @router.get("/v1/assessments")
    async def list_assessments(
        limit: int = Query(50, ge=1, le=1000),
        offset: int = Query(0, ge=0),
        request: Request = None,  # type: ignore[assignment]
    ) -> Dict[str, Any]:
        """List quality assessments."""
        service = _get_service(request)
        assessments = service.list_assessments(limit=limit, offset=offset)
        return {
            "assessments": [a.model_dump(mode="json") for a in assessments],
            "count": len(assessments),
            "limit": limit,
            "offset": offset,
        }

    # ------------------------------------------------------------------
    # 8. Get single assessment
    # ------------------------------------------------------------------
    @router.get("/v1/assessments/{assessment_id}")
    async def get_assessment(
        assessment_id: str,
        request: Request,
    ) -> Dict[str, Any]:
        """Get a quality assessment by ID."""
        service = _get_service(request)
        assessment = service.get_assessment(assessment_id)
        if assessment is None:
            raise HTTPException(
                status_code=404,
                detail=f"Assessment {assessment_id} not found",
            )
        return assessment.model_dump(mode="json")

    # ------------------------------------------------------------------
    # 9. Validate dataset with rules
    # ------------------------------------------------------------------
    @router.post("/v1/validate")
    async def validate_dataset(
        body: ValidateDatasetBody,
        request: Request,
    ) -> Dict[str, Any]:
        """Validate a dataset against quality rules."""
        service = _get_service(request)
        try:
            result = service.validate_dataset(
                data=body.data,
                dataset_name=body.dataset_name,
                rule_ids=body.rule_ids,
            )
            return result
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    # ------------------------------------------------------------------
    # 10. Detect anomalies
    # ------------------------------------------------------------------
    @router.post("/v1/detect-anomalies")
    async def detect_anomalies(
        body: DetectAnomaliesBody,
        request: Request,
    ) -> Dict[str, Any]:
        """Detect anomalies in a dataset."""
        service = _get_service(request)
        try:
            result = service.detect_anomalies(
                data=body.data,
                dataset_name=body.dataset_name,
                columns=body.columns,
                method=body.method,
            )
            return result.model_dump(mode="json")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    # ------------------------------------------------------------------
    # 11. List anomaly results
    # ------------------------------------------------------------------
    @router.get("/v1/anomalies")
    async def list_anomalies(
        limit: int = Query(50, ge=1, le=1000),
        offset: int = Query(0, ge=0),
        request: Request = None,  # type: ignore[assignment]
    ) -> Dict[str, Any]:
        """List anomaly detection results."""
        service = _get_service(request)
        results = service.list_anomalies(limit=limit, offset=offset)
        return {
            "anomaly_results": [r.model_dump(mode="json") for r in results],
            "count": len(results),
            "limit": limit,
            "offset": offset,
        }

    # ------------------------------------------------------------------
    # 12. Check dataset freshness
    # ------------------------------------------------------------------
    @router.post("/v1/check-freshness")
    async def check_freshness(
        body: CheckFreshnessBody,
        request: Request,
    ) -> Dict[str, Any]:
        """Check the freshness of a dataset."""
        service = _get_service(request)
        result = service.check_freshness(
            dataset_name=body.dataset_name,
            last_updated=body.last_updated,
            sla_hours=body.sla_hours,
        )
        return result.model_dump(mode="json")

    # ------------------------------------------------------------------
    # 13. Create quality rule
    # ------------------------------------------------------------------
    @router.post("/v1/rules")
    async def create_rule(
        body: CreateRuleBody,
        request: Request,
    ) -> Dict[str, Any]:
        """Create a new quality rule."""
        service = _get_service(request)
        try:
            rule = service.create_rule(
                name=body.name,
                rule_type=body.rule_type,
                column=body.column,
                operator=body.operator,
                threshold=body.threshold,
                parameters=body.parameters,
                priority=body.priority,
            )
            return rule.model_dump(mode="json")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    # ------------------------------------------------------------------
    # 14. List quality rules
    # ------------------------------------------------------------------
    @router.get("/v1/rules")
    async def list_rules(
        active_only: bool = Query(False),
        rule_type: Optional[str] = Query(None),
        request: Request = None,  # type: ignore[assignment]
    ) -> Dict[str, Any]:
        """List quality rules with optional filters."""
        service = _get_service(request)
        rules = service.list_rules(
            active_only=active_only,
            rule_type=rule_type,
        )
        return {
            "rules": [r.model_dump(mode="json") for r in rules],
            "count": len(rules),
        }

    # ------------------------------------------------------------------
    # 15. Update quality rule
    # ------------------------------------------------------------------
    @router.put("/v1/rules/{rule_id}")
    async def update_rule(
        rule_id: str,
        body: UpdateRuleBody,
        request: Request,
    ) -> Dict[str, Any]:
        """Update an existing quality rule."""
        service = _get_service(request)
        try:
            updates = body.model_dump(exclude_none=True)
            rule = service.update_rule(rule_id=rule_id, updates=updates)
            return rule.model_dump(mode="json")
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc))

    # ------------------------------------------------------------------
    # 16. Delete quality rule
    # ------------------------------------------------------------------
    @router.delete("/v1/rules/{rule_id}")
    async def delete_rule(
        rule_id: str,
        request: Request,
    ) -> Dict[str, Any]:
        """Delete a quality rule."""
        service = _get_service(request)
        deleted = service.delete_rule(rule_id)
        if not deleted:
            raise HTTPException(
                status_code=404,
                detail=f"Rule {rule_id} not found",
            )
        return {
            "deleted": True,
            "rule_id": rule_id,
        }

    # ------------------------------------------------------------------
    # 17. Evaluate quality gate
    # ------------------------------------------------------------------
    @router.post("/v1/gates")
    async def evaluate_gate(
        body: EvaluateGateBody,
        request: Request,
    ) -> Dict[str, Any]:
        """Evaluate a quality gate against dimension scores."""
        service = _get_service(request)
        try:
            result = service.evaluate_gate(
                conditions=body.conditions,
                dimension_scores=body.dimension_scores,
            )
            return result.model_dump(mode="json")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    # ------------------------------------------------------------------
    # 18. Get quality trends
    # ------------------------------------------------------------------
    @router.get("/v1/trends")
    async def get_trends(
        dataset_name: Optional[str] = Query(None),
        periods: int = Query(10, ge=1, le=100),
        request: Request = None,  # type: ignore[assignment]
    ) -> Dict[str, Any]:
        """Get quality score trends over recent assessments."""
        service = _get_service(request)
        trends = service.get_trends(
            dataset_name=dataset_name,
            periods=periods,
        )
        return {
            "trends": trends,
            "count": len(trends),
            "dataset_name": dataset_name or "all",
        }

    # ------------------------------------------------------------------
    # 19. Generate report
    # ------------------------------------------------------------------
    @router.post("/v1/reports")
    async def generate_report(
        body: GenerateReportBody,
        request: Request,
    ) -> Dict[str, Any]:
        """Generate a data quality report."""
        service = _get_service(request)
        report = service.generate_report(
            dataset_name=body.dataset_name,
            report_type=body.report_type,
            report_format=body.format,
        )
        return report

    # ------------------------------------------------------------------
    # 20. Health check
    # ------------------------------------------------------------------
    @router.get("/health")
    async def health(
        request: Request,
    ) -> Dict[str, Any]:
        """Data quality profiler service health check endpoint."""
        service = _get_service(request)
        return service.health_check()


__all__ = [
    "router",
]
