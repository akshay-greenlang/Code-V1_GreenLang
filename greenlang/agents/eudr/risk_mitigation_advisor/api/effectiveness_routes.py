# -*- coding: utf-8 -*-
"""
Effectiveness Tracking Routes - AGENT-EUDR-025 Risk Mitigation Advisor API

Endpoints for mitigation effectiveness measurement including before/after
risk scoring, ROI analysis, trend detection, and portfolio-level summaries.

Endpoints (4):
    GET /effectiveness/{plan_id}                - Plan effectiveness metrics
    GET /effectiveness/supplier/{supplier_id}   - Supplier effectiveness history
    GET /effectiveness/portfolio                - Portfolio-level summary
    GET /effectiveness/roi                      - ROI analysis

RBAC Permissions:
    eudr-rma:effectiveness:read - View effectiveness metrics

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-025, Engine 5: Effectiveness Tracking
"""

from __future__ import annotations

import logging
import time
from decimal import Decimal
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status

from greenlang.agents.eudr.risk_mitigation_advisor.api.dependencies import (
    AuthUser,
    get_rma_service,
    rate_limit_standard,
    require_permission,
    validate_uuid,
)
from greenlang.agents.eudr.risk_mitigation_advisor.api.schemas import (
    EffectivenessEntry,
    EffectivenessPlanResponse,
    EffectivenessSupplierResponse,
    ErrorResponse,
    PortfolioEffectivenessResponse,
    ROIAnalysisResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/effectiveness", tags=["Effectiveness Tracking"])


def _record_dict_to_entry(r: Dict[str, Any]) -> EffectivenessEntry:
    """Convert effectiveness record dictionary to EffectivenessEntry."""
    return EffectivenessEntry(
        record_id=r.get("record_id", ""),
        plan_id=r.get("plan_id", ""),
        supplier_id=r.get("supplier_id", ""),
        baseline_risk_scores=r.get("baseline_risk_scores", {}),
        current_risk_scores=r.get("current_risk_scores", {}),
        dimension_reductions=r.get("dimension_reductions", r.get("risk_reduction_pct", {})),
        composite_reduction_pct=Decimal(str(r.get("composite_reduction_pct", 0))),
        predicted_reduction_pct=Decimal(str(r["predicted_reduction_pct"])) if r.get("predicted_reduction_pct") is not None else None,
        deviation_pct=Decimal(str(r["deviation_pct"])) if r.get("deviation_pct") is not None else None,
        roi=Decimal(str(r["roi"])) if r.get("roi") is not None else None,
        cost_to_date=Decimal(str(r.get("cost_to_date", 0))),
        statistical_significance=r.get("statistical_significance", False),
        p_value=Decimal(str(r["p_value"])) if r.get("p_value") is not None else None,
        measured_at=r.get("measured_at"),
        provenance_hash=r.get("provenance_hash", ""),
    )


# ---------------------------------------------------------------------------
# GET /effectiveness/portfolio (must be before /{plan_id})
# ---------------------------------------------------------------------------


@router.get(
    "/portfolio",
    response_model=PortfolioEffectivenessResponse,
    summary="Get portfolio-level effectiveness summary",
    description=(
        "Retrieve aggregated effectiveness metrics across all active "
        "mitigation plans for the operator. Includes average risk "
        "reduction, portfolio ROI, top/underperforming plans, and "
        "monthly trend data."
    ),
    responses={200: {"description": "Portfolio summary generated"}},
)
async def get_portfolio_effectiveness(
    request: Request,
    user: AuthUser = Depends(require_permission("eudr-rma:effectiveness:read")),
    _rate: None = Depends(rate_limit_standard),
    service: Any = Depends(get_rma_service),
) -> PortfolioEffectivenessResponse:
    """Get portfolio-level effectiveness summary."""
    try:
        result = await service.get_portfolio_effectiveness(
            operator_id=user.operator_id,
        )

        data = result if isinstance(result, dict) else {}
        return PortfolioEffectivenessResponse(
            total_plans_active=data.get("total_plans_active", 0),
            total_suppliers_under_mitigation=data.get("total_suppliers_under_mitigation", 0),
            average_composite_reduction_pct=Decimal(str(data.get("average_composite_reduction_pct", 0))),
            average_roi=Decimal(str(data["average_roi"])) if data.get("average_roi") is not None else None,
            top_performing_plans=data.get("top_performing_plans", []),
            underperforming_plans=data.get("underperforming_plans", []),
            trend_data=data.get("trend_data", []),
        )

    except Exception as e:
        logger.error("Portfolio effectiveness failed: %s", e, exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to get portfolio effectiveness")


# ---------------------------------------------------------------------------
# GET /effectiveness/roi
# ---------------------------------------------------------------------------


@router.get(
    "/roi",
    response_model=ROIAnalysisResponse,
    summary="Get ROI analysis across portfolio",
    description=(
        "Retrieve comprehensive ROI analysis across the mitigation portfolio "
        "including investment totals, risk reduction value, ROI by risk "
        "category and commodity, and cost-effectiveness ranking of measures."
    ),
    responses={200: {"description": "ROI analysis generated"}},
)
async def get_roi_analysis(
    request: Request,
    user: AuthUser = Depends(require_permission("eudr-rma:effectiveness:read")),
    _rate: None = Depends(rate_limit_standard),
    service: Any = Depends(get_rma_service),
) -> ROIAnalysisResponse:
    """Get ROI analysis."""
    try:
        result = await service.get_roi_analysis(
            operator_id=user.operator_id,
        )

        data = result if isinstance(result, dict) else {}
        return ROIAnalysisResponse(
            total_investment_eur=Decimal(str(data.get("total_investment_eur", 0))),
            total_risk_reduction_value_eur=Decimal(str(data.get("total_risk_reduction_value_eur", 0))),
            portfolio_roi_pct=Decimal(str(data.get("portfolio_roi_pct", 0))),
            by_risk_category=data.get("by_risk_category", {}),
            by_commodity=data.get("by_commodity", {}),
            cost_effectiveness_ranking=data.get("cost_effectiveness_ranking", []),
        )

    except Exception as e:
        logger.error("ROI analysis failed: %s", e, exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to generate ROI analysis")


# ---------------------------------------------------------------------------
# GET /effectiveness/supplier/{supplier_id}
# ---------------------------------------------------------------------------


@router.get(
    "/supplier/{supplier_id}",
    response_model=EffectivenessSupplierResponse,
    summary="Get supplier effectiveness history",
    description="Retrieve effectiveness tracking history for a specific supplier across all active mitigation plans.",
    responses={
        200: {"description": "Supplier effectiveness retrieved"},
        404: {"model": ErrorResponse, "description": "Supplier not found or no effectiveness data"},
    },
)
async def get_supplier_effectiveness(
    request: Request,
    supplier_id: str,
    user: AuthUser = Depends(require_permission("eudr-rma:effectiveness:read")),
    _rate: None = Depends(rate_limit_standard),
    service: Any = Depends(get_rma_service),
) -> EffectivenessSupplierResponse:
    """Get supplier effectiveness history."""
    try:
        result = await service.get_supplier_effectiveness(
            supplier_id=supplier_id,
            operator_id=user.operator_id,
        )

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No effectiveness data for supplier {supplier_id}",
            )

        data = result if isinstance(result, dict) else {}
        records = [_record_dict_to_entry(r) for r in data.get("records", [])]

        return EffectivenessSupplierResponse(
            supplier_id=supplier_id,
            records=records,
            aggregate_reduction_pct=Decimal(str(data.get("aggregate_reduction_pct", 0))),
            aggregate_roi=Decimal(str(data["aggregate_roi"])) if data.get("aggregate_roi") is not None else None,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Supplier effectiveness failed: %s", e, exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to get supplier effectiveness")


# ---------------------------------------------------------------------------
# GET /effectiveness/{plan_id}
# ---------------------------------------------------------------------------


@router.get(
    "/{plan_id}",
    response_model=EffectivenessPlanResponse,
    summary="Get effectiveness metrics for a plan",
    description=(
        "Retrieve effectiveness tracking metrics for a specific remediation "
        "plan including baseline/current risk scores, per-dimension risk "
        "reduction, composite reduction, predicted vs actual comparison, "
        "ROI, and statistical significance of the measured improvement."
    ),
    responses={
        200: {"description": "Effectiveness metrics retrieved"},
        404: {"model": ErrorResponse, "description": "Plan not found or no effectiveness data"},
    },
)
async def get_plan_effectiveness(
    request: Request,
    plan_id: str,
    user: AuthUser = Depends(require_permission("eudr-rma:effectiveness:read")),
    _rate: None = Depends(rate_limit_standard),
    service: Any = Depends(get_rma_service),
) -> EffectivenessPlanResponse:
    """Get effectiveness metrics for a plan."""
    validate_uuid(plan_id, "plan_id")

    try:
        result = await service.get_plan_effectiveness(
            plan_id=plan_id,
            operator_id=user.operator_id,
        )

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No effectiveness data for plan {plan_id}",
            )

        data = result if isinstance(result, dict) else {}
        records = [_record_dict_to_entry(r) for r in data.get("records", [])]
        latest = _record_dict_to_entry(data["latest"]) if data.get("latest") else None

        return EffectivenessPlanResponse(
            plan_id=plan_id,
            records=records,
            latest=latest,
            trend=data.get("trend", "stable"),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Plan effectiveness failed: %s", e, exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to get plan effectiveness")
