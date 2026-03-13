# -*- coding: utf-8 -*-
"""
Mitigation Measure Library Routes - AGENT-EUDR-025 Risk Mitigation Advisor API

Endpoints for the searchable 500+ mitigation measure knowledge base including
full-text search with faceted filtering, measure detail retrieval, side-by-side
comparison, and risk-scenario-based measure package recommendations.

Endpoints (4):
    GET /measures                               - Search/list measures
    GET /measures/{measure_id}                  - Get measure detail
    GET /measures/compare                       - Compare measures side-by-side
    GET /measures/packages/{risk_scenario}      - Get recommended measure package

RBAC Permissions:
    eudr-rma:library:read   - Browse and search measures

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-025, Engine 4: Measure Library
"""

from __future__ import annotations

import logging
import time
from decimal import Decimal
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status

from greenlang.agents.eudr.risk_mitigation_advisor.api.dependencies import (
    AuthUser,
    PaginationParams,
    get_pagination,
    get_rma_service,
    rate_limit_standard,
    require_permission,
    validate_uuid,
)
from greenlang.agents.eudr.risk_mitigation_advisor.api.schemas import (
    ErrorResponse,
    MeasureCompareResponse,
    MeasureDetailResponse,
    MeasureEntry,
    MeasurePackageResponse,
    MeasureSearchResponse,
    PaginatedMeta,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/measures", tags=["Mitigation Measure Library"])


def _measure_dict_to_entry(m: Dict[str, Any]) -> MeasureEntry:
    """Convert measure dictionary to MeasureEntry schema."""
    return MeasureEntry(
        measure_id=m.get("measure_id", ""),
        name=m.get("name", ""),
        description=m.get("description", ""),
        risk_category=m.get("risk_category", ""),
        sub_category=m.get("sub_category", ""),
        effectiveness_rating=Decimal(str(m.get("effectiveness_rating", 0))),
        cost_estimate_min=Decimal(str(m["cost_estimate_min"])) if m.get("cost_estimate_min") is not None else None,
        cost_estimate_max=Decimal(str(m["cost_estimate_max"])) if m.get("cost_estimate_max") is not None else None,
        implementation_complexity=m.get("implementation_complexity", "medium"),
        time_to_effect_weeks=m.get("time_to_effect_weeks", 8),
        expected_risk_reduction_min=Decimal(str(m["expected_risk_reduction_min"])) if m.get("expected_risk_reduction_min") is not None else None,
        expected_risk_reduction_max=Decimal(str(m["expected_risk_reduction_max"])) if m.get("expected_risk_reduction_max") is not None else None,
        iso_31000_type=m.get("iso_31000_type", "reduce"),
        eudr_articles=m.get("eudr_articles", []),
        certification_schemes=m.get("certification_schemes", []),
        tags=m.get("tags", []),
        applicability=m.get("applicability", {}),
    )


# ---------------------------------------------------------------------------
# GET /measures
# ---------------------------------------------------------------------------


@router.get(
    "",
    response_model=MeasureSearchResponse,
    summary="Search mitigation measures",
    description=(
        "Search the 500+ mitigation measure library with full-text search "
        "and faceted filtering by risk category, commodity, country, cost range, "
        "complexity, ISO 31000 treatment type, and effectiveness rating. "
        "Uses PostgreSQL GIN index for sub-500ms response. Results are ranked "
        "by relevance (if query provided) or effectiveness rating."
    ),
    responses={
        200: {"description": "Measures retrieved"},
    },
)
async def search_measures(
    request: Request,
    query: Optional[str] = Query(None, description="Full-text search query"),
    risk_category: Optional[str] = Query(None, description="Filter by risk category: country, supplier, commodity, corruption, deforestation, indigenous_rights, protected_areas, legal_compliance"),
    commodity: Optional[str] = Query(None, description="Filter by EUDR commodity"),
    country_code: Optional[str] = Query(None, description="Filter by country code"),
    cost_max: Optional[float] = Query(None, ge=0, description="Maximum cost (EUR)"),
    complexity: Optional[str] = Query(None, description="Filter by complexity: low, medium, high, very_high"),
    iso_31000_type: Optional[str] = Query(None, description="Filter by ISO 31000 type: avoid, reduce, share, retain"),
    min_effectiveness: Optional[float] = Query(None, ge=0, le=100, description="Minimum effectiveness rating"),
    pagination: PaginationParams = Depends(get_pagination),
    user: AuthUser = Depends(require_permission("eudr-rma:library:read")),
    _rate: None = Depends(rate_limit_standard),
    service: Any = Depends(get_rma_service),
) -> MeasureSearchResponse:
    """Search and filter mitigation measures."""
    start = time.monotonic()

    try:
        result = await service.search_measures(
            query=query,
            risk_category=risk_category,
            commodity=commodity,
            country_code=country_code,
            cost_max=Decimal(str(cost_max)) if cost_max is not None else None,
            complexity=complexity,
            iso_31000_type=iso_31000_type,
            min_effectiveness=Decimal(str(min_effectiveness)) if min_effectiveness is not None else None,
            limit=pagination.limit,
            offset=pagination.offset,
        )

        measures_raw = result.get("measures", []) if isinstance(result, dict) else []
        total = result.get("total", 0) if isinstance(result, dict) else 0
        facets = result.get("facets", {}) if isinstance(result, dict) else {}
        measures = [_measure_dict_to_entry(m) for m in measures_raw]

        elapsed_ms = int((time.monotonic() - start) * 1000)
        logger.info(
            "Measure search: query=%s results=%d elapsed_ms=%d",
            query, len(measures), elapsed_ms,
        )

        return MeasureSearchResponse(
            measures=measures,
            meta=PaginatedMeta(
                total=total, limit=pagination.limit, offset=pagination.offset,
                has_more=(pagination.offset + pagination.limit) < total,
            ),
            facets=facets,
        )

    except Exception as e:
        logger.error("Measure search failed: %s", e, exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to search measures")


# ---------------------------------------------------------------------------
# GET /measures/compare (must be before /{measure_id} to avoid path conflict)
# ---------------------------------------------------------------------------


@router.get(
    "/compare",
    response_model=MeasureCompareResponse,
    summary="Compare measures side-by-side",
    description=(
        "Compare 2-5 mitigation measures side-by-side on effectiveness, "
        "cost, complexity, time-to-effect, and applicability criteria."
    ),
    responses={
        200: {"description": "Comparison generated"},
        400: {"model": ErrorResponse, "description": "Must provide 2-5 measure IDs"},
    },
)
async def compare_measures(
    request: Request,
    ids: str = Query(..., description="Comma-separated measure IDs (2-5)"),
    user: AuthUser = Depends(require_permission("eudr-rma:library:read")),
    _rate: None = Depends(rate_limit_standard),
    service: Any = Depends(get_rma_service),
) -> MeasureCompareResponse:
    """Compare measures side-by-side."""
    measure_ids = [mid.strip() for mid in ids.split(",") if mid.strip()]

    if len(measure_ids) < 2 or len(measure_ids) > 5:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Provide 2-5 measure IDs for comparison",
        )

    try:
        result = await service.compare_measures(
            measure_ids=measure_ids,
            operator_id=user.operator_id,
        )

        data = result if isinstance(result, dict) else {}
        measures = [_measure_dict_to_entry(m) for m in data.get("measures", [])]

        return MeasureCompareResponse(
            measures=measures,
            comparison_matrix=data.get("comparison_matrix", {}),
            recommendation=data.get("recommendation"),
        )

    except Exception as e:
        logger.error("Measure comparison failed: %s", e, exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to compare measures")


# ---------------------------------------------------------------------------
# GET /measures/packages/{risk_scenario}
# ---------------------------------------------------------------------------


@router.get(
    "/packages/{risk_scenario}",
    response_model=MeasurePackageResponse,
    summary="Get recommended measure package for risk scenario",
    description=(
        "Get a recommended package of mitigation measures for a specific "
        "risk scenario. Scenarios include: high_country_risk, high_supplier_risk, "
        "deforestation_critical, indigenous_rights_violation, protected_area_encroachment, "
        "legal_gap, corruption_high, multi_dimensional."
    ),
    responses={
        200: {"description": "Package generated"},
        404: {"model": ErrorResponse, "description": "Unknown risk scenario"},
    },
)
async def get_measure_package(
    request: Request,
    risk_scenario: str,
    commodity: Optional[str] = Query(None, description="EUDR commodity context"),
    country_code: Optional[str] = Query(None, description="Country context"),
    user: AuthUser = Depends(require_permission("eudr-rma:library:read")),
    _rate: None = Depends(rate_limit_standard),
    service: Any = Depends(get_rma_service),
) -> MeasurePackageResponse:
    """Get recommended measure package for a risk scenario."""
    try:
        result = await service.get_measure_package(
            risk_scenario=risk_scenario,
            commodity=commodity,
            country_code=country_code,
        )

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Unknown risk scenario: {risk_scenario}",
            )

        data = result if isinstance(result, dict) else {}
        measures = [_measure_dict_to_entry(m) for m in data.get("recommended_measures", [])]

        return MeasurePackageResponse(
            risk_scenario=risk_scenario,
            description=data.get("description", ""),
            recommended_measures=measures,
            total_estimated_cost=data.get("total_estimated_cost", {}),
            expected_risk_reduction=Decimal(str(data.get("expected_risk_reduction", 0))),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Measure package retrieval failed: %s", e, exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to retrieve measure package")


# ---------------------------------------------------------------------------
# GET /measures/{measure_id}
# ---------------------------------------------------------------------------


@router.get(
    "/{measure_id}",
    response_model=MeasureDetailResponse,
    summary="Get measure detail with evidence",
    description="Retrieve full details of a mitigation measure including effectiveness evidence, prerequisites, and applicability criteria.",
    responses={
        200: {"description": "Measure detail retrieved"},
        404: {"model": ErrorResponse, "description": "Measure not found"},
    },
)
async def get_measure_detail(
    request: Request,
    measure_id: str,
    user: AuthUser = Depends(require_permission("eudr-rma:library:read")),
    _rate: None = Depends(rate_limit_standard),
    service: Any = Depends(get_rma_service),
) -> MeasureDetailResponse:
    """Get full details of a specific mitigation measure."""
    validate_uuid(measure_id, "measure_id")

    try:
        result = await service.get_measure(measure_id)

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Measure {measure_id} not found",
            )

        data = result if isinstance(result, dict) else {}
        measure = _measure_dict_to_entry(data)

        return MeasureDetailResponse(
            measure=measure,
            effectiveness_evidence=data.get("effectiveness_evidence", []),
            prerequisite_conditions=data.get("prerequisite_conditions", []),
            target_risk_factors=data.get("target_risk_factors", []),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Measure detail failed: %s", e, exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to retrieve measure")
