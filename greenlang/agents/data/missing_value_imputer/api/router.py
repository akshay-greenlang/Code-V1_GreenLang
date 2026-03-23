# -*- coding: utf-8 -*-
"""
Missing Value Imputer REST API Router - AGENT-DATA-012

FastAPI router providing 20 endpoints for missing value imputation,
missingness analysis, strategy selection, batch imputation, validation,
rule management, template management, pipeline execution, job management,
and health/statistics.

All endpoints are mounted under ``/api/v1/imputer``.

Endpoints:
    1.  POST   /jobs                  - Create imputation job
    2.  GET    /jobs                  - List jobs
    3.  GET    /jobs/{job_id}         - Get job details
    4.  DELETE /jobs/{job_id}         - Delete job
    5.  POST   /analyze              - Analyze missingness
    6.  GET    /analyze/{analysis_id} - Get analysis result
    7.  POST   /impute               - Impute values
    8.  POST   /impute/batch         - Batch impute
    9.  GET    /results/{result_id}  - Get imputation result
    10. POST   /validate             - Validate imputation
    11. GET    /validate/{id}        - Get validation result
    12. POST   /rules                - Create rule
    13. GET    /rules                - List rules
    14. PUT    /rules/{rule_id}      - Update rule
    15. DELETE /rules/{rule_id}      - Delete rule
    16. POST   /templates            - Create template
    17. GET    /templates            - List templates
    18. POST   /pipeline             - Run pipeline
    19. GET    /health               - Health check
    20. GET    /stats                - Statistics

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-012 Missing Value Imputer (GL-DATA-X-015)
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
        "FastAPI not available; missing value imputer router is None"
    )


# ---------------------------------------------------------------------------
# Pydantic request/response models (only when FastAPI is available)
# ---------------------------------------------------------------------------

if FASTAPI_AVAILABLE:

    class CreateJobBody(BaseModel):
        """Request body for creating an imputation job."""
        dataset_id: str = Field(
            default="", description="Identifier of the dataset to impute",
        )
        records: List[Dict[str, Any]] = Field(
            ..., description="List of record dicts to impute",
        )
        pipeline_config: Optional[Dict[str, Any]] = Field(
            None, description="Optional pipeline configuration",
        )
        template_id: Optional[str] = Field(
            None, description="Optional template ID to apply",
        )

    class AnalyzeBody(BaseModel):
        """Request body for analyzing missingness."""
        records: List[Dict[str, Any]] = Field(
            ..., description="List of record dicts to analyze",
        )
        columns: Optional[List[str]] = Field(
            None, description="Columns to analyze (all if omitted)",
        )

    class ImputeBody(BaseModel):
        """Request body for imputing values in a single column."""
        records: List[Dict[str, Any]] = Field(
            ..., description="List of record dicts to impute",
        )
        column: str = Field(
            ..., description="Column name to impute",
        )
        strategy: Optional[str] = Field(
            None,
            description="Imputation strategy (mean, median, mode, knn, "
            "regression, mice, random_forest, gradient_boosting, "
            "linear_interpolation, spline_interpolation, "
            "seasonal_decomposition, rule_based, lookup_table, "
            "regulatory_default)",
        )
        options: Optional[Dict[str, Any]] = Field(
            None, description="Additional imputation options",
        )

    class BatchImputeBody(BaseModel):
        """Request body for batch imputation across columns."""
        records: List[Dict[str, Any]] = Field(
            ..., description="List of record dicts to impute",
        )
        strategies: Optional[Dict[str, str]] = Field(
            None,
            description="Column-to-strategy mapping (auto-select if omitted)",
        )

    class ValidateBody(BaseModel):
        """Request body for validating imputation results."""
        original_records: List[Dict[str, Any]] = Field(
            ..., description="Original records before imputation",
        )
        imputed_records: List[Dict[str, Any]] = Field(
            ..., description="Records after imputation",
        )
        method: str = Field(
            default="plausibility_range",
            description="Validation method (ks_test, chi_square, "
            "plausibility_range, distribution_preservation, "
            "cross_validation)",
        )

    class CreateRuleBody(BaseModel):
        """Request body for creating an imputation rule."""
        name: str = Field(
            ..., description="Human-readable rule name",
        )
        target_column: str = Field(
            ..., description="Column whose missing values this rule imputes",
        )
        conditions: Optional[List[Dict[str, Any]]] = Field(
            None, description="List of rule condition dicts",
        )
        impute_value: Optional[Any] = Field(
            None, description="Static value to impute when conditions are met",
        )
        priority: str = Field(
            default="medium",
            description="Rule priority (critical, high, medium, low, default)",
        )
        justification: str = Field(
            default="",
            description="Justification for the rule",
        )

    class UpdateRuleBody(BaseModel):
        """Request body for updating an imputation rule."""
        name: Optional[str] = Field(
            None, description="Updated rule name",
        )
        target_column: Optional[str] = Field(
            None, description="Updated target column",
        )
        conditions: Optional[List[Dict[str, Any]]] = Field(
            None, description="Updated conditions",
        )
        impute_value: Optional[Any] = Field(
            None, description="Updated impute value",
        )
        priority: Optional[str] = Field(
            None, description="Updated priority",
        )
        is_active: Optional[bool] = Field(
            None, description="Updated active status",
        )
        justification: Optional[str] = Field(
            None, description="Updated justification",
        )

    class CreateTemplateBody(BaseModel):
        """Request body for creating an imputation template."""
        name: str = Field(
            ..., description="Human-readable template name",
        )
        description: str = Field(
            default="",
            description="Template description",
        )
        strategies: Optional[Dict[str, str]] = Field(
            None,
            description="Column-to-strategy mapping",
        )

    class RunPipelineBody(BaseModel):
        """Request body for running the full imputation pipeline."""
        records: List[Dict[str, Any]] = Field(
            ..., description="List of record dicts to impute",
        )
        config: Optional[Dict[str, Any]] = Field(
            None, description="Optional pipeline configuration overrides",
        )


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

if FASTAPI_AVAILABLE:
    router = APIRouter(
        prefix="/api/v1/imputer",
        tags=["Missing Value Imputer"],
    )
else:
    router = None  # type: ignore[assignment]


def _get_service(request: Request) -> Any:
    """Extract MissingValueImputerService from app state.

    Args:
        request: FastAPI request object.

    Returns:
        MissingValueImputerService instance.

    Raises:
        HTTPException: If service is not configured.
    """
    service = getattr(
        request.app.state, "missing_value_imputer_service", None,
    )
    if service is None:
        raise HTTPException(
            status_code=503,
            detail="Missing value imputer service not configured",
        )
    return service


if FASTAPI_AVAILABLE:

    # ------------------------------------------------------------------
    # 1. Create imputation job
    # ------------------------------------------------------------------
    @router.post("/jobs")
    async def create_job(
        body: CreateJobBody,
        request: Request,
    ) -> Dict[str, Any]:
        """Create a new imputation job."""
        service = _get_service(request)
        try:
            result = service.create_job(
                request={
                    "dataset_id": body.dataset_id,
                    "records": body.records,
                    "pipeline_config": body.pipeline_config,
                    "template_id": body.template_id,
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
        """List imputation jobs with optional status filter."""
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
        """Get an imputation job by ID."""
        service = _get_service(request)
        job = service.get_job(job_id)
        if job is None:
            raise HTTPException(
                status_code=404,
                detail=f"Job {job_id} not found",
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
        """Delete (cancel) an imputation job."""
        service = _get_service(request)
        deleted = service.delete_job(job_id)
        if not deleted:
            raise HTTPException(
                status_code=404,
                detail=f"Job {job_id} not found",
            )
        return {"job_id": job_id, "status": "cancelled"}

    # ------------------------------------------------------------------
    # 5. Analyze missingness
    # ------------------------------------------------------------------
    @router.post("/analyze")
    async def analyze_missingness(
        body: AnalyzeBody,
        request: Request,
    ) -> Dict[str, Any]:
        """Analyze missingness patterns in a dataset."""
        service = _get_service(request)
        try:
            result = service.analyze_missingness(
                records=body.records,
                columns=body.columns,
            )
            return result.model_dump(mode="json")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    # ------------------------------------------------------------------
    # 6. Get analysis result
    # ------------------------------------------------------------------
    @router.get("/analyze/{analysis_id}")
    async def get_analysis(
        analysis_id: str,
        request: Request,
    ) -> Dict[str, Any]:
        """Get a missingness analysis result by ID."""
        service = _get_service(request)
        result = service.get_analysis(analysis_id)
        if result is None:
            raise HTTPException(
                status_code=404,
                detail=f"Analysis {analysis_id} not found",
            )
        return result.model_dump(mode="json")

    # ------------------------------------------------------------------
    # 7. Impute values
    # ------------------------------------------------------------------
    @router.post("/impute")
    async def impute_values(
        body: ImputeBody,
        request: Request,
    ) -> Dict[str, Any]:
        """Impute missing values in a single column."""
        service = _get_service(request)
        try:
            result = service.impute_values(
                records=body.records,
                column=body.column,
                strategy=body.strategy,
                options=body.options,
            )
            return result.model_dump(mode="json")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    # ------------------------------------------------------------------
    # 8. Batch impute
    # ------------------------------------------------------------------
    @router.post("/impute/batch")
    async def impute_batch(
        body: BatchImputeBody,
        request: Request,
    ) -> Dict[str, Any]:
        """Batch impute missing values across multiple columns."""
        service = _get_service(request)
        try:
            result = service.impute_batch(
                records=body.records,
                strategies=body.strategies,
            )
            return result.model_dump(mode="json")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    # ------------------------------------------------------------------
    # 9. Get imputation result
    # ------------------------------------------------------------------
    @router.get("/results/{result_id}")
    async def get_results(
        result_id: str,
        request: Request,
    ) -> Dict[str, Any]:
        """Get an imputation result by ID."""
        service = _get_service(request)
        result = service.get_results(result_id)
        if result is None:
            raise HTTPException(
                status_code=404,
                detail=f"Result {result_id} not found",
            )
        return result.model_dump(mode="json")

    # ------------------------------------------------------------------
    # 10. Validate imputation
    # ------------------------------------------------------------------
    @router.post("/validate")
    async def validate_imputation(
        body: ValidateBody,
        request: Request,
    ) -> Dict[str, Any]:
        """Validate imputation quality using statistical tests."""
        service = _get_service(request)
        try:
            result = service.validate_imputation(
                original=body.original_records,
                imputed=body.imputed_records,
                method=body.method,
            )
            return result.model_dump(mode="json")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    # ------------------------------------------------------------------
    # 11. Get validation result
    # ------------------------------------------------------------------
    @router.get("/validate/{validation_id}")
    async def get_validation(
        validation_id: str,
        request: Request,
    ) -> Dict[str, Any]:
        """Get a validation result by ID."""
        service = _get_service(request)
        result = service.get_validation(validation_id)
        if result is None:
            raise HTTPException(
                status_code=404,
                detail=f"Validation {validation_id} not found",
            )
        return result.model_dump(mode="json")

    # ------------------------------------------------------------------
    # 12. Create rule
    # ------------------------------------------------------------------
    @router.post("/rules")
    async def create_rule(
        body: CreateRuleBody,
        request: Request,
    ) -> Dict[str, Any]:
        """Create a new imputation rule."""
        service = _get_service(request)
        try:
            result = service.create_rule(
                name=body.name,
                target_column=body.target_column,
                conditions=body.conditions,
                impute_value=body.impute_value,
                priority=body.priority,
                justification=body.justification,
            )
            return result.model_dump(mode="json")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    # ------------------------------------------------------------------
    # 13. List rules
    # ------------------------------------------------------------------
    @router.get("/rules")
    async def list_rules(
        is_active: Optional[bool] = Query(None),
        request: Request = None,  # type: ignore[assignment]
    ) -> Dict[str, Any]:
        """List imputation rules with optional active filter."""
        service = _get_service(request)
        rules = service.list_rules(is_active=is_active)
        return {
            "rules": rules,
            "count": len(rules),
        }

    # ------------------------------------------------------------------
    # 14. Update rule
    # ------------------------------------------------------------------
    @router.put("/rules/{rule_id}")
    async def update_rule(
        rule_id: str,
        body: UpdateRuleBody,
        request: Request,
    ) -> Dict[str, Any]:
        """Update an existing imputation rule."""
        service = _get_service(request)
        updates = body.model_dump(exclude_none=True)
        result = service.update_rule(rule_id, **updates)
        if result is None:
            raise HTTPException(
                status_code=404,
                detail=f"Rule {rule_id} not found",
            )
        return result.model_dump(mode="json")

    # ------------------------------------------------------------------
    # 15. Delete rule
    # ------------------------------------------------------------------
    @router.delete("/rules/{rule_id}")
    async def delete_rule(
        rule_id: str,
        request: Request,
    ) -> Dict[str, Any]:
        """Delete (deactivate) an imputation rule."""
        service = _get_service(request)
        deleted = service.delete_rule(rule_id)
        if not deleted:
            raise HTTPException(
                status_code=404,
                detail=f"Rule {rule_id} not found",
            )
        return {"rule_id": rule_id, "status": "deleted"}

    # ------------------------------------------------------------------
    # 16. Create template
    # ------------------------------------------------------------------
    @router.post("/templates")
    async def create_template(
        body: CreateTemplateBody,
        request: Request,
    ) -> Dict[str, Any]:
        """Create a new imputation template."""
        service = _get_service(request)
        try:
            result = service.create_template(
                name=body.name,
                description=body.description,
                strategies=body.strategies,
            )
            return result.model_dump(mode="json")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    # ------------------------------------------------------------------
    # 17. List templates
    # ------------------------------------------------------------------
    @router.get("/templates")
    async def list_templates(
        request: Request,
    ) -> Dict[str, Any]:
        """List all imputation templates."""
        service = _get_service(request)
        templates = service.list_templates()
        return {
            "templates": templates,
            "count": len(templates),
        }

    # ------------------------------------------------------------------
    # 18. Run pipeline
    # ------------------------------------------------------------------
    @router.post("/pipeline")
    async def run_pipeline(
        body: RunPipelineBody,
        request: Request,
    ) -> Dict[str, Any]:
        """Run the full imputation pipeline end-to-end."""
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
    async def health(
        request: Request,
    ) -> Dict[str, Any]:
        """Missing value imputer service health check endpoint."""
        service = _get_service(request)
        return service.health_check()

    # ------------------------------------------------------------------
    # 20. Statistics
    # ------------------------------------------------------------------
    @router.get("/stats")
    async def stats(
        request: Request,
    ) -> Dict[str, Any]:
        """Get aggregated missing value imputer statistics."""
        service = _get_service(request)
        result = service.get_statistics()
        return result.model_dump(mode="json")


__all__ = [
    "router",
]
