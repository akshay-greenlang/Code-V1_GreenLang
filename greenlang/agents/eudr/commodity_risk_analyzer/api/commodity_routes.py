# -*- coding: utf-8 -*-
"""
Commodity Profile Routes - AGENT-EUDR-018 Commodity Risk Analyzer API

Endpoints for EUDR commodity profiling including single commodity profiling,
batch profiling, risk scoring, risk history, cross-commodity comparison,
and summary overview.

Endpoints:
    POST /commodities/profile          - Profile a single commodity
    POST /commodities/profile-batch    - Batch profile multiple commodities
    GET  /commodities/{commodity_id}/risk    - Get commodity risk score
    GET  /commodities/{commodity_id}/history - Get risk history
    GET  /commodities/compare          - Compare commodities
    GET  /commodities/summary          - Summary of all commodities

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-018, Commodity Profiling Engine
"""

from __future__ import annotations

import hashlib
import logging
import time
import uuid
from datetime import date, datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status

from greenlang.agents.eudr.commodity_risk_analyzer.api.dependencies import (
    AuthUser,
    ErrorResponse,
    PaginationParams,
    get_commodity_profiler,
    get_pagination,
    rate_limit_heavy,
    rate_limit_standard,
    rate_limit_write,
    require_permission,
    validate_commodity_type,
    validate_date_range,
)
from greenlang.agents.eudr.commodity_risk_analyzer.api.schemas import (
    CommodityComparisonEntry,
    CommodityComparisonResponse,
    CommodityProfileBatchRequest,
    CommodityProfileBatchResponse,
    CommodityProfileRequest,
    CommodityProfileResponse,
    CommodityRiskHistoryEntry,
    CommodityRiskHistoryResponse,
    CommoditySummaryEntry,
    CommoditySummaryResponse,
    CommodityTypeEnum,
    HealthResponse,
    PaginatedMeta,
    ProvenanceInfo,
    RiskLevelEnum,
    VolatilityTrendEnum,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Commodity Profiling"])


# ---------------------------------------------------------------------------
# In-memory stores (replaced by database in production)
# ---------------------------------------------------------------------------

_profile_store: Dict[str, CommodityProfileResponse] = {}


def _compute_provenance(input_data: Any, output_data: Any) -> str:
    """Compute SHA-256 provenance hash for audit trail."""
    data_str = f"{input_data}{output_data}"
    return hashlib.sha256(data_str.encode("utf-8")).hexdigest()


def _classify_risk(score: Decimal) -> RiskLevelEnum:
    """Classify risk score into a risk level."""
    if score >= Decimal("75.0"):
        return RiskLevelEnum.CRITICAL
    elif score >= Decimal("50.0"):
        return RiskLevelEnum.HIGH
    elif score >= Decimal("25.0"):
        return RiskLevelEnum.MEDIUM
    return RiskLevelEnum.LOW


# ---------------------------------------------------------------------------
# POST /commodities/profile
# ---------------------------------------------------------------------------


@router.post(
    "/commodities/profile",
    response_model=CommodityProfileResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Profile a single EUDR commodity",
    description=(
        "Generate a comprehensive risk profile for an EUDR-regulated commodity "
        "including risk scoring (0-100), deforestation risk classification, "
        "supply chain complexity analysis, and traceability scoring. "
        "Per EU 2023/1115 Articles 4, 9, and 10."
    ),
    responses={
        201: {"description": "Commodity profile created successfully"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        422: {"model": ErrorResponse, "description": "Validation error"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def profile_commodity(
    request: Request,
    body: CommodityProfileRequest,
    user: AuthUser = Depends(
        require_permission("eudr-commodity-risk:commodities:create")
    ),
    _rate: None = Depends(rate_limit_write),
) -> CommodityProfileResponse:
    """Profile a single EUDR commodity with risk analysis.

    Args:
        body: Commodity profiling request with country and supply chain data.
        user: Authenticated user with commodities:create permission.

    Returns:
        CommodityProfileResponse with risk scores and provenance hash.
    """
    start = time.monotonic()
    try:
        profile_id = str(uuid.uuid4())

        # Compute risk score based on country distribution and supply chain data
        country_risk = sum(
            (c.risk_score or Decimal("50.0")) * c.share
            for c in body.country_data
        )
        sc_data = body.supply_chain_data
        complexity = Decimal("0.5")
        traceability = Decimal("0.5")
        if sc_data:
            complexity = min(Decimal("1.0"), Decimal(str(sc_data.depth)) / Decimal("10"))
            traceability = sc_data.traceability_coverage

        risk_score = min(
            Decimal("100.0"),
            country_risk * Decimal("0.6") + complexity * Decimal("40.0"),
        )

        country_dist = {c.country_code: c.share for c in body.country_data}

        elapsed_ms = Decimal(str((time.monotonic() - start) * 1000))
        provenance_hash = _compute_provenance(
            body.model_dump_json(), f"{profile_id}:{risk_score}"
        )

        profile = CommodityProfileResponse(
            profile_id=profile_id,
            commodity_type=body.commodity_type,
            risk_score=risk_score.quantize(Decimal("0.01")),
            risk_level=_classify_risk(risk_score),
            deforestation_risk=_classify_risk(country_risk),
            supply_chain_complexity=complexity.quantize(Decimal("0.01")),
            traceability_score=traceability,
            country_distribution=country_dist,
            provenance=ProvenanceInfo(
                provenance_hash=provenance_hash,
                processing_time_ms=elapsed_ms.quantize(Decimal("0.01")),
            ),
        )

        _profile_store[profile_id] = profile

        logger.info(
            "Commodity profiled: id=%s type=%s risk_score=%s operator=%s",
            profile_id,
            body.commodity_type.value,
            risk_score,
            user.operator_id or user.user_id,
        )

        return profile

    except Exception as exc:
        logger.error("Commodity profiling failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Commodity profiling failed",
        )


# ---------------------------------------------------------------------------
# POST /commodities/profile-batch
# ---------------------------------------------------------------------------


@router.post(
    "/commodities/profile-batch",
    response_model=CommodityProfileBatchResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Batch profile multiple EUDR commodities",
    description=(
        "Profile up to 50 EUDR commodities in a single batch request. "
        "Each commodity is profiled independently with its own risk scores."
    ),
    responses={
        201: {"description": "Batch profiling completed"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def profile_commodity_batch(
    request: Request,
    body: CommodityProfileBatchRequest,
    user: AuthUser = Depends(
        require_permission("eudr-commodity-risk:commodities:create")
    ),
    _rate: None = Depends(rate_limit_heavy),
) -> CommodityProfileBatchResponse:
    """Batch profile multiple EUDR commodities.

    Args:
        body: Batch request containing up to 50 commodity profiles.
        user: Authenticated user with commodities:create permission.

    Returns:
        CommodityProfileBatchResponse with all profiling results.
    """
    start = time.monotonic()
    profiles: List[CommodityProfileResponse] = []
    failed = 0

    for commodity_req in body.commodities:
        try:
            # Reuse single profile logic inline
            profile_id = str(uuid.uuid4())
            country_risk = sum(
                (c.risk_score or Decimal("50.0")) * c.share
                for c in commodity_req.country_data
            )
            sc_data = commodity_req.supply_chain_data
            complexity = Decimal("0.5")
            traceability = Decimal("0.5")
            if sc_data:
                complexity = min(
                    Decimal("1.0"),
                    Decimal(str(sc_data.depth)) / Decimal("10"),
                )
                traceability = sc_data.traceability_coverage

            risk_score = min(
                Decimal("100.0"),
                country_risk * Decimal("0.6") + complexity * Decimal("40.0"),
            )

            country_dist = {c.country_code: c.share for c in commodity_req.country_data}
            provenance_hash = _compute_provenance(
                commodity_req.model_dump_json(), f"{profile_id}:{risk_score}"
            )

            profile = CommodityProfileResponse(
                profile_id=profile_id,
                commodity_type=commodity_req.commodity_type,
                risk_score=risk_score.quantize(Decimal("0.01")),
                risk_level=_classify_risk(risk_score),
                deforestation_risk=_classify_risk(country_risk),
                supply_chain_complexity=complexity.quantize(Decimal("0.01")),
                traceability_score=traceability,
                country_distribution=country_dist,
                provenance=ProvenanceInfo(
                    provenance_hash=provenance_hash,
                    processing_time_ms=Decimal("0.0"),
                ),
            )
            profiles.append(profile)
            _profile_store[profile_id] = profile

        except Exception as exc:
            logger.warning("Batch item failed: %s", exc)
            failed += 1

    elapsed_ms = Decimal(str((time.monotonic() - start) * 1000))

    logger.info(
        "Batch profiling completed: total=%d succeeded=%d failed=%d",
        len(body.commodities),
        len(profiles),
        failed,
    )

    return CommodityProfileBatchResponse(
        profiles=profiles,
        total_processed=len(profiles),
        total_failed=failed,
        processing_time_ms=elapsed_ms.quantize(Decimal("0.01")),
    )


# ---------------------------------------------------------------------------
# GET /commodities/{commodity_id}/risk
# ---------------------------------------------------------------------------


@router.get(
    "/commodities/{commodity_id}/risk",
    response_model=CommodityProfileResponse,
    summary="Get commodity risk score",
    description="Retrieve the current risk profile for a specific commodity.",
    responses={
        200: {"description": "Commodity risk profile"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "Commodity not found"},
    },
)
async def get_commodity_risk(
    commodity_id: str,
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-commodity-risk:commodities:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> CommodityProfileResponse:
    """Get the current risk profile for a commodity.

    Args:
        commodity_id: Unique commodity profile identifier.
        user: Authenticated user with commodities:read permission.

    Returns:
        CommodityProfileResponse with current risk scores.

    Raises:
        HTTPException: 404 if commodity not found.
    """
    profile = _profile_store.get(commodity_id)
    if profile is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Commodity profile {commodity_id} not found",
        )
    return profile


# ---------------------------------------------------------------------------
# GET /commodities/{commodity_id}/history
# ---------------------------------------------------------------------------


@router.get(
    "/commodities/{commodity_id}/history",
    response_model=CommodityRiskHistoryResponse,
    summary="Get commodity risk history",
    description="Retrieve historical risk scores for a commodity over time.",
    responses={
        200: {"description": "Risk history time series"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "Commodity not found"},
    },
)
async def get_commodity_history(
    commodity_id: str,
    request: Request,
    date_range: Dict[str, Optional[date]] = Depends(validate_date_range),
    pagination: PaginationParams = Depends(get_pagination),
    user: AuthUser = Depends(
        require_permission("eudr-commodity-risk:commodities:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> CommodityRiskHistoryResponse:
    """Get risk history for a commodity.

    Args:
        commodity_id: Unique commodity profile identifier.
        date_range: Optional date range filter.
        pagination: Pagination parameters.
        user: Authenticated user with commodities:read permission.

    Returns:
        CommodityRiskHistoryResponse with time series data.

    Raises:
        HTTPException: 404 if commodity not found.
    """
    profile = _profile_store.get(commodity_id)
    if profile is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Commodity profile {commodity_id} not found",
        )

    # In production, fetch from time-series database.
    # Return current snapshot as single history point.
    today = date.today()
    history_entry = CommodityRiskHistoryEntry(
        date=today,
        risk_score=profile.risk_score,
        risk_level=profile.risk_level,
        deforestation_risk=profile.deforestation_risk,
    )

    return CommodityRiskHistoryResponse(
        commodity_id=commodity_id,
        commodity_type=profile.commodity_type,
        history=[history_entry],
        period_start=date_range.get("start_date") or today,
        period_end=date_range.get("end_date") or today,
        trend=VolatilityTrendEnum.STABLE,
    )


# ---------------------------------------------------------------------------
# GET /commodities/compare
# ---------------------------------------------------------------------------


@router.get(
    "/commodities/compare",
    response_model=CommodityComparisonResponse,
    summary="Compare commodities",
    description=(
        "Compare risk profiles across multiple commodities side by side, "
        "including a pairwise risk differential matrix and recommendations."
    ),
    responses={
        200: {"description": "Commodity comparison results"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
    },
)
async def compare_commodities(
    request: Request,
    commodity_ids: str = Query(
        ...,
        description="Comma-separated list of commodity profile IDs to compare",
    ),
    user: AuthUser = Depends(
        require_permission("eudr-commodity-risk:commodities:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> CommodityComparisonResponse:
    """Compare multiple commodity risk profiles.

    Args:
        commodity_ids: Comma-separated commodity profile IDs.
        user: Authenticated user with commodities:read permission.

    Returns:
        CommodityComparisonResponse with comparison matrix.

    Raises:
        HTTPException: 400 if fewer than 2 commodities provided.
    """
    ids = [cid.strip() for cid in commodity_ids.split(",") if cid.strip()]
    if len(ids) < 2:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least 2 commodity IDs required for comparison",
        )

    entries: List[CommodityComparisonEntry] = []
    for cid in ids:
        profile = _profile_store.get(cid)
        if profile is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Commodity profile {cid} not found",
            )
        entries.append(
            CommodityComparisonEntry(
                commodity_type=profile.commodity_type,
                risk_score=profile.risk_score,
                deforestation_risk=profile.deforestation_risk,
                supply_chain_complexity=profile.supply_chain_complexity,
                traceability_score=profile.traceability_score,
                price_volatility=profile.price_volatility_index,
            )
        )

    # Build pairwise risk differential matrix
    matrix: Dict[str, Dict[str, Decimal]] = {}
    for i, e1 in enumerate(entries):
        key1 = e1.commodity_type.value
        matrix[key1] = {}
        for j, e2 in enumerate(entries):
            key2 = e2.commodity_type.value
            matrix[key1][key2] = abs(e1.risk_score - e2.risk_score)

    # Generate basic recommendations
    recommendations: List[str] = []
    highest = max(entries, key=lambda e: e.risk_score)
    lowest = min(entries, key=lambda e: e.risk_score)
    if highest.risk_score - lowest.risk_score > Decimal("30.0"):
        recommendations.append(
            f"Consider increasing due diligence for {highest.commodity_type.value} "
            f"which has significantly higher risk than {lowest.commodity_type.value}"
        )

    return CommodityComparisonResponse(
        commodities=entries,
        comparison_matrix=matrix,
        recommendations=recommendations,
    )


# ---------------------------------------------------------------------------
# GET /commodities/summary
# ---------------------------------------------------------------------------


@router.get(
    "/commodities/summary",
    response_model=CommoditySummaryResponse,
    summary="Summary of all monitored commodities",
    description=(
        "Get a high-level overview of all monitored EUDR commodities "
        "including average risk scores and the highest-risk commodity."
    ),
    responses={
        200: {"description": "Commodity summary"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
    },
)
async def get_commodity_summary(
    request: Request,
    user: AuthUser = Depends(
        require_permission("eudr-commodity-risk:commodities:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> CommoditySummaryResponse:
    """Get summary overview of all monitored commodities.

    Args:
        user: Authenticated user with commodities:read permission.

    Returns:
        CommoditySummaryResponse with aggregated metrics.
    """
    # Aggregate by commodity type
    type_scores: Dict[str, List[Decimal]] = {}
    for profile in _profile_store.values():
        ct = profile.commodity_type.value
        if ct not in type_scores:
            type_scores[ct] = []
        type_scores[ct].append(profile.risk_score)

    entries: List[CommoditySummaryEntry] = []
    for ct, scores in type_scores.items():
        avg_score = sum(scores) / len(scores)
        entries.append(
            CommoditySummaryEntry(
                commodity_type=CommodityTypeEnum(ct),
                risk_score=avg_score.quantize(Decimal("0.01")),
                risk_level=_classify_risk(avg_score),
                active_suppliers=len(scores),
                compliance_rate=Decimal("0.75"),  # Placeholder
            )
        )

    total = len(entries)
    avg_risk = Decimal("0.0")
    highest: Optional[CommodityTypeEnum] = None
    if entries:
        avg_risk = (sum(e.risk_score for e in entries) / total).quantize(
            Decimal("0.01")
        )
        highest_entry = max(entries, key=lambda e: e.risk_score)
        highest = highest_entry.commodity_type

    return CommoditySummaryResponse(
        commodities=entries,
        total_commodities=total,
        average_risk_score=avg_risk,
        highest_risk_commodity=highest,
    )


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Check EUDR Commodity Risk Analyzer API health.",
    tags=["System"],
)
async def health_check() -> HealthResponse:
    """Health check endpoint for load balancers and monitoring."""
    return HealthResponse()
