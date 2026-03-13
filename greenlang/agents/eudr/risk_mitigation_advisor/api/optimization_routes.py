# -*- coding: utf-8 -*-
"""
Cost-Benefit Optimization Routes - AGENT-EUDR-025 Risk Mitigation Advisor API

Endpoints for budget allocation optimization using linear programming,
Pareto frontier generation, and sensitivity analysis.

Endpoints (4):
    POST /optimization/run                              - Run budget optimization
    GET  /optimization/{optimization_id}                - Get optimization results
    GET  /optimization/{optimization_id}/pareto         - Get Pareto frontier
    GET  /optimization/{optimization_id}/sensitivity    - Get sensitivity analysis

RBAC Permissions:
    eudr-rma:optimization:execute - Run optimizations
    eudr-rma:optimization:read    - View results

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-025, Engine 7: Cost-Benefit Optimizer
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status

from greenlang.agents.eudr.risk_mitigation_advisor.api.dependencies import (
    AuthUser,
    get_rma_service,
    rate_limit_optimize,
    rate_limit_standard,
    require_permission,
    validate_uuid,
)
from greenlang.agents.eudr.risk_mitigation_advisor.api.schemas import (
    ErrorResponse,
    OptimizationResultResponse,
    OptimizeBudgetRequest,
    ParetoFrontierResponse,
    SensitivityAnalysisResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/optimization", tags=["Cost-Benefit Optimization"])


# ---------------------------------------------------------------------------
# POST /optimization/run
# ---------------------------------------------------------------------------


@router.post(
    "/run",
    response_model=OptimizationResultResponse,
    status_code=status.HTTP_200_OK,
    summary="Run budget optimization",
    description=(
        "Execute linear programming optimization to allocate mitigation "
        "budget across suppliers and measures for maximum risk reduction. "
        "Uses PuLP CBC deterministic solver. Supports constraints: total "
        "budget, per-supplier cap, per-risk-category limits. Generates "
        "optimal allocation with predicted risk reduction and budget "
        "utilization metrics."
    ),
    responses={
        200: {"description": "Optimization completed"},
        400: {"model": ErrorResponse, "description": "Invalid optimization parameters"},
        408: {"model": ErrorResponse, "description": "Solver timeout (> 30s)"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded (10/min)"},
    },
)
async def run_optimization(
    request: Request,
    body: OptimizeBudgetRequest,
    user: AuthUser = Depends(require_permission("eudr-rma:optimization:execute")),
    _rate: None = Depends(rate_limit_optimize),
    service: Any = Depends(get_rma_service),
) -> OptimizationResultResponse:
    """Run budget allocation optimization."""
    start = time.monotonic()

    try:
        from greenlang.agents.eudr.risk_mitigation_advisor.models import (
            OptimizeBudgetRequest as EngineRequest,
        )

        engine_request = EngineRequest(
            operator_id=body.operator_id,
            total_budget=body.total_budget,
            per_supplier_cap=body.per_supplier_cap,
            category_budgets=body.category_budgets,
            supplier_ids=body.supplier_ids,
            scenario_name=body.scenario_name,
        )

        result = await service.optimize_budget(engine_request)

        data = result if isinstance(result, dict) else {}
        elapsed_ms = int((time.monotonic() - start) * 1000)

        logger.info(
            "Optimization completed: operator=%s budget=%s elapsed_ms=%d solver=%s",
            body.operator_id, body.total_budget, elapsed_ms,
            data.get("solver_status", "unknown"),
        )

        return OptimizationResultResponse(
            optimization_id=data.get("optimization_id", ""),
            operator_id=body.operator_id,
            scenario_name=body.scenario_name,
            total_budget=body.total_budget,
            total_cost=Decimal(str(data.get("total_cost", 0))),
            budget_utilization_pct=Decimal(str(data.get("budget_utilization_pct", 0))),
            total_predicted_risk_reduction=Decimal(str(data.get("total_predicted_risk_reduction", 0))),
            allocations=data.get("allocations", []),
            solver_status=data.get("solver_status", ""),
            computation_time_ms=elapsed_ms,
            created_at=datetime.now(timezone.utc),
        )

    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except TimeoutError:
        raise HTTPException(
            status_code=status.HTTP_408_REQUEST_TIMEOUT,
            detail="Optimization solver timed out (> 30s). Reduce portfolio size or relax constraints.",
        )
    except Exception as e:
        logger.error("Optimization failed: %s", e, exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Optimization processing error")


# ---------------------------------------------------------------------------
# GET /optimization/{optimization_id}
# ---------------------------------------------------------------------------


@router.get(
    "/{optimization_id}",
    response_model=OptimizationResultResponse,
    summary="Get optimization results",
    description="Retrieve results of a previously executed budget optimization.",
    responses={
        200: {"description": "Results retrieved"},
        404: {"model": ErrorResponse, "description": "Optimization result not found"},
    },
)
async def get_optimization_result(
    request: Request,
    optimization_id: str,
    user: AuthUser = Depends(require_permission("eudr-rma:optimization:read")),
    _rate: None = Depends(rate_limit_standard),
    service: Any = Depends(get_rma_service),
) -> OptimizationResultResponse:
    """Get optimization results."""
    validate_uuid(optimization_id, "optimization_id")

    try:
        result = await service.get_optimization_result(
            optimization_id=optimization_id,
            operator_id=user.operator_id,
        )

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Optimization result {optimization_id} not found",
            )

        data = result if isinstance(result, dict) else {}
        return OptimizationResultResponse(
            optimization_id=optimization_id,
            operator_id=data.get("operator_id", ""),
            scenario_name=data.get("scenario_name", ""),
            total_budget=Decimal(str(data.get("total_budget", 0))),
            total_cost=Decimal(str(data.get("total_cost", 0))),
            budget_utilization_pct=Decimal(str(data.get("budget_utilization_pct", 0))),
            total_predicted_risk_reduction=Decimal(str(data.get("total_predicted_risk_reduction", 0))),
            allocations=data.get("allocations", []),
            solver_status=data.get("solver_status", ""),
            computation_time_ms=data.get("computation_time_ms", 0),
            created_at=data.get("created_at"),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Optimization result retrieval failed: %s", e, exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to retrieve optimization result")


# ---------------------------------------------------------------------------
# GET /optimization/{optimization_id}/pareto
# ---------------------------------------------------------------------------


@router.get(
    "/{optimization_id}/pareto",
    response_model=ParetoFrontierResponse,
    summary="Get Pareto frontier data",
    description=(
        "Retrieve the Pareto-optimal frontier showing budget vs. risk "
        "reduction trade-offs. Generated by varying the budget constraint "
        "and re-solving the LP problem at each step."
    ),
    responses={
        200: {"description": "Pareto frontier retrieved"},
        404: {"model": ErrorResponse, "description": "Optimization result not found"},
    },
)
async def get_pareto_frontier(
    request: Request,
    optimization_id: str,
    user: AuthUser = Depends(require_permission("eudr-rma:optimization:read")),
    _rate: None = Depends(rate_limit_standard),
    service: Any = Depends(get_rma_service),
) -> ParetoFrontierResponse:
    """Get Pareto frontier for an optimization."""
    validate_uuid(optimization_id, "optimization_id")

    try:
        result = await service.get_pareto_frontier(
            optimization_id=optimization_id,
            operator_id=user.operator_id,
        )

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Pareto frontier for optimization {optimization_id} not found",
            )

        data = result if isinstance(result, dict) else {}
        budget_range = data.get("budget_range", {})
        budget_range_dec = {
            k: Decimal(str(v)) for k, v in budget_range.items()
        } if budget_range else {}

        return ParetoFrontierResponse(
            optimization_id=optimization_id,
            points=data.get("points", []),
            budget_range=budget_range_dec,
            recommended_point=data.get("recommended_point"),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Pareto frontier failed: %s", e, exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to retrieve Pareto frontier")


# ---------------------------------------------------------------------------
# GET /optimization/{optimization_id}/sensitivity
# ---------------------------------------------------------------------------


@router.get(
    "/{optimization_id}/sensitivity",
    response_model=SensitivityAnalysisResponse,
    summary="Get sensitivity analysis",
    description=(
        "Retrieve sensitivity analysis showing how changes in budget, "
        "measure availability, and supplier priority affect optimal "
        "allocation. Identifies which constraints are binding and "
        "highest-impact budget decisions."
    ),
    responses={
        200: {"description": "Sensitivity analysis retrieved"},
        404: {"model": ErrorResponse, "description": "Optimization result not found"},
    },
)
async def get_sensitivity_analysis(
    request: Request,
    optimization_id: str,
    user: AuthUser = Depends(require_permission("eudr-rma:optimization:read")),
    _rate: None = Depends(rate_limit_standard),
    service: Any = Depends(get_rma_service),
) -> SensitivityAnalysisResponse:
    """Get sensitivity analysis for an optimization."""
    validate_uuid(optimization_id, "optimization_id")

    try:
        result = await service.get_sensitivity_analysis(
            optimization_id=optimization_id,
            operator_id=user.operator_id,
        )

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Sensitivity analysis for optimization {optimization_id} not found",
            )

        data = result if isinstance(result, dict) else {}
        return SensitivityAnalysisResponse(
            optimization_id=optimization_id,
            budget_sensitivity=data.get("budget_sensitivity", []),
            measure_sensitivity=data.get("measure_sensitivity", []),
            supplier_priority_ranking=data.get("supplier_priority_ranking", []),
            constraint_binding=data.get("constraint_binding", {}),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Sensitivity analysis failed: %s", e, exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to retrieve sensitivity analysis")
