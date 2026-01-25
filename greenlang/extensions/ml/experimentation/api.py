# -*- coding: utf-8 -*-
"""
FastAPI Integration for A/B Testing Framework

Provides RESTful API endpoints for creating, managing, and analyzing
A/B experiments for Process Heat agents.

Endpoints:
    POST /experiments - Create new experiment
    GET /experiments/{id}/assign - Get variant assignment
    POST /experiments/{id}/metrics - Record metric
    GET /experiments/{id}/results - Get analysis results
    GET /experiments/{id}/status - Get experiment status
    GET /experiments/{id}/prometheus - Export Prometheus metrics

Example:
    >>> from fastapi import FastAPI
    >>> from greenlang.ml.experimentation.api import create_ab_testing_routes
    >>> app = FastAPI()
    >>> manager = ABTestManager()
    >>> create_ab_testing_routes(app, manager)
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from fastapi import APIRouter, HTTPException, Depends
from greenlang.ml.experimentation.ab_testing import (
    ABTestManager,
    MetricType,
    TestType,
    ExperimentResult,
)
import logging

logger = logging.getLogger(__name__)


class CreateExperimentRequest(BaseModel):
    """Request to create experiment."""

    name: str = Field(..., description="Experiment name")
    variants: List[str] = Field(..., description="Variant names")
    traffic_split: Optional[Dict[str, float]] = Field(
        None, description="Traffic allocation"
    )
    metric_type: str = Field("continuous", description="Metric type")
    test_type: str = Field("welch_t", description="Test type")


class AssignVariantRequest(BaseModel):
    """Request to assign variant."""

    user_id: str = Field(..., description="User identifier")
    experiment_id: str = Field(..., description="Experiment ID")


class RecordMetricRequest(BaseModel):
    """Request to record metric."""

    experiment_id: str = Field(..., description="Experiment ID")
    variant: str = Field(..., description="Variant name")
    metric_name: str = Field(..., description="Metric name")
    value: float = Field(..., description="Metric value")


class AssignVariantResponse(BaseModel):
    """Response with assigned variant."""

    variant: str = Field(..., description="Assigned variant")
    experiment_id: str = Field(..., description="Experiment ID")


class ExperimentStatusResponse(BaseModel):
    """Experiment status response."""

    experiment_id: str
    name: str
    status: str
    variants: List[str]
    sample_counts: Dict[str, int]
    winner: Optional[str]
    is_significant: bool
    p_value: float


def create_ab_testing_routes(app, manager: ABTestManager) -> None:
    """
    Register A/B testing routes with FastAPI app.

    Args:
        app: FastAPI application
        manager: ABTestManager instance
    """
    router = APIRouter(prefix="/api/v1/experiments", tags=["A/B Testing"])

    @router.post("", response_model=Dict[str, str])
    async def create_experiment(request: CreateExperimentRequest):
        """Create a new experiment."""
        try:
            traffic_split = request.traffic_split
            if traffic_split:
                # Normalize traffic split
                total = sum(traffic_split.values())
                traffic_split = {k: v / total for k, v in traffic_split.items()}

            exp_id = manager.create_experiment(
                name=request.name,
                variants=request.variants,
                traffic_split=traffic_split,
                metric_type=MetricType(request.metric_type),
                test_type=TestType(request.test_type),
            )

            logger.info(f"Created experiment: {exp_id}")
            return {"experiment_id": exp_id}

        except Exception as e:
            logger.error(f"Error creating experiment: {str(e)}")
            raise HTTPException(status_code=400, detail=str(e))

    @router.get("/{experiment_id}/assign", response_model=AssignVariantResponse)
    async def assign_variant(experiment_id: str, user_id: str):
        """Get variant assignment for user."""
        try:
            variant = manager.assign_variant(user_id, experiment_id)
            return AssignVariantResponse(
                variant=variant, experiment_id=experiment_id
            )

        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            logger.error(f"Error assigning variant: {str(e)}")
            raise HTTPException(status_code=400, detail=str(e))

    @router.post("/{experiment_id}/metrics", response_model=Dict[str, str])
    async def record_metric(experiment_id: str, request: RecordMetricRequest):
        """Record metric for variant."""
        try:
            manager.record_metric(
                experiment_id=experiment_id,
                variant=request.variant,
                metric_name=request.metric_name,
                value=request.value,
            )

            logger.info(
                f"Recorded metric for exp {experiment_id}, "
                f"variant {request.variant}"
            )
            return {"status": "recorded"}

        except Exception as e:
            logger.error(f"Error recording metric: {str(e)}")
            raise HTTPException(status_code=400, detail=str(e))

    @router.get("/{experiment_id}/results", response_model=Dict[str, Any])
    async def get_results(experiment_id: str):
        """Get experiment analysis results."""
        try:
            result = manager.analyze_results(experiment_id)
            return result.dict()

        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            logger.error(f"Error analyzing results: {str(e)}")
            raise HTTPException(status_code=400, detail=str(e))

    @router.get("/{experiment_id}/status", response_model=ExperimentStatusResponse)
    async def get_status(experiment_id: str):
        """Get experiment status."""
        try:
            status = manager.get_status(experiment_id)
            return ExperimentStatusResponse(**status)

        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            logger.error(f"Error getting status: {str(e)}")
            raise HTTPException(status_code=400, detail=str(e))

    @router.get("/{experiment_id}/prometheus")
    async def get_prometheus_metrics(experiment_id: str):
        """Get Prometheus metrics export."""
        try:
            metrics = manager.export_prometheus_metrics(experiment_id)
            return {"metrics": metrics}

        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            logger.error(f"Error exporting metrics: {str(e)}")
            raise HTTPException(status_code=400, detail=str(e))

    app.include_router(router)
