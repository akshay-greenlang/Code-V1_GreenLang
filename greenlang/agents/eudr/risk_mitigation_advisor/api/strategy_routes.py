# -*- coding: utf-8 -*-
"""
Strategy Recommendation Routes - AGENT-EUDR-025 Risk Mitigation Advisor API

Endpoints for ML-powered mitigation strategy recommendation, strategy listing,
strategy detail retrieval with SHAP explainability, strategy selection for
implementation, and detailed explainability reports.

Endpoints (5):
    POST /strategies/recommend         - Generate strategy recommendations
    GET  /strategies                   - List recommended strategies
    GET  /strategies/{strategy_id}     - Get strategy detail
    POST /strategies/{strategy_id}/select  - Select strategy for implementation
    GET  /strategies/{strategy_id}/explain - Get SHAP explainability report

RBAC Permissions:
    eudr-rma:strategies:read    - View strategies
    eudr-rma:strategies:execute - Generate recommendations
    eudr-rma:plans:write        - Select strategy for implementation

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-025, Engine 1: Strategy Selection
"""

from __future__ import annotations

import hashlib
import logging
import time
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status

from greenlang.agents.eudr.risk_mitigation_advisor.api.dependencies import (
    AuthUser,
    PaginationParams,
    get_pagination,
    get_rma_service,
    rate_limit_ml,
    rate_limit_standard,
    require_permission,
    validate_uuid,
)
from greenlang.agents.eudr.risk_mitigation_advisor.api.schemas import (
    ErrorResponse,
    PaginatedMeta,
    ProvenanceInfo,
    StrategyEntry,
    StrategyExplainResponse,
    StrategyListResponse,
    StrategyRecommendRequest,
    StrategyRecommendResponse,
    StrategySelectRequest,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/strategies", tags=["Strategy Recommendation"])


def _compute_provenance(input_data: Any, output_data: Any) -> str:
    """Compute SHA-256 provenance hash for audit trail."""
    data_str = f"{input_data}{output_data}"
    return hashlib.sha256(data_str.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# POST /strategies/recommend
# ---------------------------------------------------------------------------


@router.post(
    "/recommend",
    response_model=StrategyRecommendResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate mitigation strategy recommendations",
    description=(
        "Analyze risk assessment outputs from 9 upstream EUDR agents "
        "(EUDR-016 through EUDR-024) and recommend context-appropriate "
        "mitigation strategies. Uses ML-powered XGBoost/LightGBM recommendation "
        "engine with SHAP explainability. Falls back to deterministic rule-based "
        "engine when ML confidence < 0.7. Returns ranked list of top-K strategies "
        "with predicted effectiveness, cost, complexity, and time-to-effect. "
        "Per EUDR Article 11: strategies are adequate and proportionate to reduce "
        "risk to negligible level."
    ),
    responses={
        200: {"description": "Strategy recommendations generated successfully"},
        400: {"model": ErrorResponse, "description": "Invalid risk input parameters"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded (30/min for ML inference)"},
    },
)
async def recommend_strategies(
    request: Request,
    body: StrategyRecommendRequest,
    user: AuthUser = Depends(
        require_permission("eudr-rma:strategies:execute")
    ),
    _rate: None = Depends(rate_limit_ml),
    service: Any = Depends(get_rma_service),
) -> StrategyRecommendResponse:
    """Generate ML-powered or deterministic strategy recommendations.

    Consumes risk scores from 9 upstream agents, extracts 45+ features,
    and runs XGBoost inference (or deterministic fallback) to produce
    ranked strategy recommendations with SHAP explanations.

    Args:
        body: Risk input context with scores from 9 upstream agents.
        user: Authenticated user with eudr-rma:strategies:execute permission.
        service: RiskMitigationAdvisorSetup singleton.

    Returns:
        StrategyRecommendResponse with ranked strategies and provenance.
    """
    start = time.monotonic()

    try:
        # Build risk input for engine
        from greenlang.agents.eudr.risk_mitigation_advisor.models import (
            RecommendStrategiesRequest as EngineRequest,
            RiskInput,
        )

        risk_input = RiskInput(
            operator_id=body.operator_id,
            supplier_id=body.supplier_id,
            country_code=body.country_code,
            commodity=body.commodity,
            country_risk_score=body.country_risk_score,
            supplier_risk_score=body.supplier_risk_score,
            commodity_risk_score=body.commodity_risk_score,
            corruption_risk_score=body.corruption_risk_score,
            deforestation_risk_score=body.deforestation_risk_score,
            indigenous_rights_score=body.indigenous_rights_score,
            protected_areas_score=body.protected_areas_score,
            legal_compliance_score=body.legal_compliance_score,
            audit_risk_score=body.audit_risk_score,
            due_diligence_level=body.due_diligence_level,
            risk_factors=body.risk_factors,
        )

        engine_request = EngineRequest(
            risk_input=risk_input,
            mode=body.mode,
            top_k=body.top_k,
        )

        result = await service.recommend_strategies(engine_request)

        # Build response
        strategies = []
        raw_strategies = getattr(result, "strategies", []) if result else []
        for s in raw_strategies:
            strategies.append(
                StrategyEntry(
                    strategy_id=getattr(s, "strategy_id", ""),
                    name=getattr(s, "name", ""),
                    description=getattr(s, "description", ""),
                    risk_categories=[str(c) for c in getattr(s, "risk_categories", [])],
                    iso_31000_type=str(getattr(s, "iso_31000_type", "reduce")),
                    predicted_effectiveness=getattr(s, "predicted_effectiveness", Decimal("0")),
                    confidence_score=getattr(s, "confidence_score", Decimal("0")),
                    cost_estimate=getattr(s, "cost_estimate", {}) if isinstance(getattr(s, "cost_estimate", None), dict) else {},
                    implementation_complexity=str(getattr(s, "implementation_complexity", "medium")),
                    time_to_effect_weeks=getattr(s, "time_to_effect_weeks", 8),
                    eudr_articles=getattr(s, "eudr_articles", []),
                    shap_explanation=getattr(s, "shap_explanation", {}),
                    measure_ids=getattr(s, "measure_ids", []),
                    model_version=getattr(s, "model_version", ""),
                    provenance_hash=getattr(s, "provenance_hash", ""),
                    status=getattr(s, "status", "recommended") if hasattr(s, "status") else "recommended",
                    created_at=getattr(s, "created_at", None),
                )
            )

        elapsed_ms = int((time.monotonic() - start) * 1000)
        provenance_hash = _compute_provenance(body.model_dump(), [s.model_dump() for s in strategies])

        logger.info(
            "Strategy recommendation completed: user=%s supplier=%s mode=%s count=%d elapsed_ms=%d",
            user.user_id, body.supplier_id, body.mode, len(strategies), elapsed_ms,
        )

        return StrategyRecommendResponse(
            strategies=strategies,
            risk_profile_summary={
                "operator_id": body.operator_id,
                "supplier_id": body.supplier_id,
                "country_code": body.country_code,
                "commodity": body.commodity,
                "composite_risk": str(
                    (body.country_risk_score + body.supplier_risk_score + body.commodity_risk_score) / 3
                ),
            },
            computation_mode=body.mode,
            provenance=ProvenanceInfo(
                provenance_hash=provenance_hash,
                model_version=strategies[0].model_version if strategies else "",
                computation_mode=body.mode,
                timestamp=datetime.now(timezone.utc),
            ),
        )

    except ValueError as e:
        logger.warning("Strategy recommendation validation error: %s", e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid risk input: {e}",
        )
    except Exception as e:
        logger.error("Strategy recommendation failed: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Strategy recommendation processing error",
        )


# ---------------------------------------------------------------------------
# GET /strategies
# ---------------------------------------------------------------------------


@router.get(
    "",
    response_model=StrategyListResponse,
    summary="List recommended strategies",
    description=(
        "Retrieve a paginated list of strategy recommendations with optional "
        "filters for status, supplier, risk category, and date range."
    ),
    responses={
        200: {"description": "Strategies listed successfully"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
    },
)
async def list_strategies(
    request: Request,
    strategy_status: Optional[str] = Query(
        None, alias="status",
        description="Filter by status: recommended, selected, active, completed, rejected",
    ),
    supplier_id: Optional[str] = Query(None, description="Filter by supplier ID"),
    risk_category: Optional[str] = Query(None, description="Filter by risk category"),
    commodity: Optional[str] = Query(None, description="Filter by EUDR commodity"),
    pagination: PaginationParams = Depends(get_pagination),
    user: AuthUser = Depends(
        require_permission("eudr-rma:strategies:read")
    ),
    _rate: None = Depends(rate_limit_standard),
    service: Any = Depends(get_rma_service),
) -> StrategyListResponse:
    """List strategy recommendations with filters and pagination.

    Args:
        strategy_status: Optional status filter.
        supplier_id: Optional supplier filter.
        risk_category: Optional risk category filter.
        commodity: Optional commodity filter.
        pagination: Pagination parameters.
        user: Authenticated user.
        service: RMA service singleton.

    Returns:
        StrategyListResponse with paginated strategy list.
    """
    start = time.monotonic()

    try:
        result = await service.list_strategies(
            operator_id=user.operator_id,
            status=strategy_status,
            supplier_id=supplier_id,
            risk_category=risk_category,
            commodity=commodity,
            limit=pagination.limit,
            offset=pagination.offset,
        )

        strategies_raw = result.get("strategies", []) if isinstance(result, dict) else []
        total = result.get("total", 0) if isinstance(result, dict) else 0

        strategies = []
        for s in strategies_raw:
            strategies.append(
                StrategyEntry(
                    strategy_id=s.get("strategy_id", ""),
                    name=s.get("name", ""),
                    description=s.get("description", ""),
                    risk_categories=s.get("risk_categories", []),
                    iso_31000_type=s.get("iso_31000_type", "reduce"),
                    predicted_effectiveness=Decimal(str(s.get("predicted_effectiveness", 0))),
                    confidence_score=Decimal(str(s.get("confidence_score", 0))),
                    implementation_complexity=s.get("implementation_complexity", "medium"),
                    time_to_effect_weeks=s.get("time_to_effect_weeks", 8),
                    eudr_articles=s.get("eudr_articles", []),
                    model_version=s.get("model_version", ""),
                    provenance_hash=s.get("provenance_hash", ""),
                    status=s.get("status", "recommended"),
                    created_at=s.get("created_at"),
                )
            )

        logger.info(
            "Strategy list: user=%s count=%d elapsed_ms=%d",
            user.user_id, len(strategies), int((time.monotonic() - start) * 1000),
        )

        return StrategyListResponse(
            strategies=strategies,
            meta=PaginatedMeta(
                total=total,
                limit=pagination.limit,
                offset=pagination.offset,
                has_more=(pagination.offset + pagination.limit) < total,
            ),
        )

    except Exception as e:
        logger.error("Strategy list failed: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve strategies",
        )


# ---------------------------------------------------------------------------
# GET /strategies/{strategy_id}
# ---------------------------------------------------------------------------


@router.get(
    "/{strategy_id}",
    response_model=StrategyEntry,
    summary="Get strategy detail with SHAP explanation",
    description=(
        "Retrieve full details of a single strategy recommendation "
        "including SHAP feature importance values for explainability."
    ),
    responses={
        200: {"description": "Strategy details retrieved"},
        404: {"model": ErrorResponse, "description": "Strategy not found"},
    },
)
async def get_strategy_detail(
    request: Request,
    strategy_id: str,
    user: AuthUser = Depends(
        require_permission("eudr-rma:strategies:read")
    ),
    _rate: None = Depends(rate_limit_standard),
    service: Any = Depends(get_rma_service),
) -> StrategyEntry:
    """Get full details of a specific strategy recommendation.

    Args:
        strategy_id: Strategy UUID.
        user: Authenticated user.
        service: RMA service singleton.

    Returns:
        StrategyEntry with full details.
    """
    validate_uuid(strategy_id, "strategy_id")

    try:
        result = await service.get_strategy(strategy_id, operator_id=user.operator_id)

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Strategy {strategy_id} not found",
            )

        s = result
        return StrategyEntry(
            strategy_id=getattr(s, "strategy_id", strategy_id),
            name=getattr(s, "name", ""),
            description=getattr(s, "description", ""),
            risk_categories=[str(c) for c in getattr(s, "risk_categories", [])],
            iso_31000_type=str(getattr(s, "iso_31000_type", "reduce")),
            predicted_effectiveness=getattr(s, "predicted_effectiveness", Decimal("0")),
            confidence_score=getattr(s, "confidence_score", Decimal("0")),
            cost_estimate=getattr(s, "cost_estimate", {}) if isinstance(getattr(s, "cost_estimate", None), dict) else {},
            implementation_complexity=str(getattr(s, "implementation_complexity", "medium")),
            time_to_effect_weeks=getattr(s, "time_to_effect_weeks", 8),
            eudr_articles=getattr(s, "eudr_articles", []),
            shap_explanation=getattr(s, "shap_explanation", {}),
            measure_ids=getattr(s, "measure_ids", []),
            model_version=getattr(s, "model_version", ""),
            provenance_hash=getattr(s, "provenance_hash", ""),
            status=getattr(s, "status", "recommended") if hasattr(s, "status") else "recommended",
            created_at=getattr(s, "created_at", None),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Strategy detail retrieval failed: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve strategy details",
        )


# ---------------------------------------------------------------------------
# POST /strategies/{strategy_id}/select
# ---------------------------------------------------------------------------


@router.post(
    "/{strategy_id}/select",
    response_model=StrategyEntry,
    summary="Select strategy for implementation",
    description=(
        "Mark a recommended strategy as selected for implementation. "
        "This transitions the strategy status from 'recommended' to 'selected' "
        "and triggers remediation plan generation via Engine 2."
    ),
    responses={
        200: {"description": "Strategy selected successfully"},
        404: {"model": ErrorResponse, "description": "Strategy not found"},
        409: {"model": ErrorResponse, "description": "Strategy already selected or invalid state"},
    },
)
async def select_strategy(
    request: Request,
    strategy_id: str,
    body: StrategySelectRequest,
    user: AuthUser = Depends(
        require_permission("eudr-rma:plans:write")
    ),
    _rate: None = Depends(rate_limit_standard),
    service: Any = Depends(get_rma_service),
) -> StrategyEntry:
    """Select a strategy for implementation.

    Args:
        strategy_id: Strategy UUID to select.
        body: Selection details including rationale.
        user: Authenticated user.
        service: RMA service singleton.

    Returns:
        Updated StrategyEntry with 'selected' status.
    """
    validate_uuid(strategy_id, "strategy_id")

    try:
        result = await service.select_strategy(
            strategy_id=strategy_id,
            selected_by=body.selected_by or user.user_id,
            notes=body.notes,
            operator_id=user.operator_id,
        )

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Strategy {strategy_id} not found",
            )

        s = result
        logger.info(
            "Strategy selected: strategy_id=%s user=%s",
            strategy_id, user.user_id,
        )

        return StrategyEntry(
            strategy_id=getattr(s, "strategy_id", strategy_id),
            name=getattr(s, "name", ""),
            description=getattr(s, "description", ""),
            risk_categories=[str(c) for c in getattr(s, "risk_categories", [])],
            iso_31000_type=str(getattr(s, "iso_31000_type", "reduce")),
            predicted_effectiveness=getattr(s, "predicted_effectiveness", Decimal("0")),
            confidence_score=getattr(s, "confidence_score", Decimal("0")),
            implementation_complexity=str(getattr(s, "implementation_complexity", "medium")),
            time_to_effect_weeks=getattr(s, "time_to_effect_weeks", 8),
            eudr_articles=getattr(s, "eudr_articles", []),
            model_version=getattr(s, "model_version", ""),
            provenance_hash=getattr(s, "provenance_hash", ""),
            status="selected",
            created_at=getattr(s, "created_at", None),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Strategy selection failed: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to select strategy",
        )


# ---------------------------------------------------------------------------
# GET /strategies/{strategy_id}/explain
# ---------------------------------------------------------------------------


@router.get(
    "/{strategy_id}/explain",
    response_model=StrategyExplainResponse,
    summary="Get detailed SHAP explainability report",
    description=(
        "Retrieve detailed SHAP (SHapley Additive exPlanations) values "
        "showing which risk factors from the 9 upstream agents drove "
        "this specific strategy recommendation. Features are ranked by "
        "absolute SHAP value (most influential first). Supports audit "
        "trail requirements per EUDR Articles 10-11."
    ),
    responses={
        200: {"description": "Explainability report generated"},
        404: {"model": ErrorResponse, "description": "Strategy not found"},
    },
)
async def explain_strategy(
    request: Request,
    strategy_id: str,
    user: AuthUser = Depends(
        require_permission("eudr-rma:strategies:read")
    ),
    _rate: None = Depends(rate_limit_standard),
    service: Any = Depends(get_rma_service),
) -> StrategyExplainResponse:
    """Get SHAP-based explainability report for a strategy.

    Args:
        strategy_id: Strategy UUID.
        user: Authenticated user.
        service: RMA service singleton.

    Returns:
        StrategyExplainResponse with SHAP values and feature ranking.
    """
    validate_uuid(strategy_id, "strategy_id")

    try:
        result = await service.explain_strategy(
            strategy_id=strategy_id,
            operator_id=user.operator_id,
        )

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Strategy {strategy_id} not found",
            )

        shap_values = result.get("shap_values", {}) if isinstance(result, dict) else {}
        # Rank features by absolute SHAP value
        ranked = sorted(
            [{"feature": k, "shap_value": v, "abs_value": abs(v)} for k, v in shap_values.items()],
            key=lambda x: x["abs_value"],
            reverse=True,
        )

        provenance_hash = _compute_provenance(strategy_id, shap_values)

        return StrategyExplainResponse(
            strategy_id=strategy_id,
            strategy_name=result.get("strategy_name", "") if isinstance(result, dict) else "",
            shap_values=shap_values,
            feature_importance_ranked=ranked,
            model_version=result.get("model_version", "") if isinstance(result, dict) else "",
            computation_mode=result.get("computation_mode", "deterministic") if isinstance(result, dict) else "deterministic",
            provenance=ProvenanceInfo(
                provenance_hash=provenance_hash,
                model_version=result.get("model_version", "") if isinstance(result, dict) else "",
                computation_mode=result.get("computation_mode", "deterministic") if isinstance(result, dict) else "deterministic",
                timestamp=datetime.now(timezone.utc),
            ),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Strategy explain failed: %s", e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate explainability report",
        )
