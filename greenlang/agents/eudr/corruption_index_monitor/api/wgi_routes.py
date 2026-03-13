# -*- coding: utf-8 -*-
"""
WGI Analysis Routes - AGENT-EUDR-019 Corruption Index Monitor API

Endpoints for World Bank Worldwide Governance Indicators (WGI) analysis
across 6 governance dimensions: voice_accountability, political_stability,
government_effectiveness, regulatory_quality, rule_of_law, and
control_of_corruption on a -2.5 to +2.5 scale.

Endpoints:
    GET  /wgi/{country_code}/indicators  - All 6 WGI dimensions for a country
    GET  /wgi/{country_code}/history     - WGI indicator history for a dimension
    GET  /wgi/dimension/{dimension}      - Cross-country analysis for a dimension
    POST /wgi/compare                    - Compare countries across WGI dimensions
    GET  /wgi/rankings                   - Rankings by dimension

WGI Scale: -2.5 to +2.5 (lower=weaker governance, higher=stronger)
Data Source: World Bank
Coverage: 200+ countries, 6 dimensions

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-019, WGI Analyzer Engine
"""

from __future__ import annotations

import hashlib
import logging
import time
from decimal import Decimal
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status

from greenlang.agents.eudr.corruption_index_monitor.api.dependencies import (
    AuthUser,
    ErrorResponse,
    PaginationParams,
    get_pagination,
    get_wgi_engine,
    rate_limit_heavy,
    rate_limit_standard,
    require_permission,
    validate_country_code,
    validate_year_range,
)
from greenlang.agents.eudr.corruption_index_monitor.api.schemas import (
    DataSourceEnum,
    ErrorResponse as SchemaErrorResponse,
    GovernanceRatingEnum,
    MetadataSchema,
    PaginatedMeta,
    ProvenanceInfo,
    RiskLevelEnum,
    TrendDirectionEnum,
    WGIComparisonCountryEntry,
    WGIComparisonRequest,
    WGIComparisonResponse,
    WGIDimensionCountryEntry,
    WGIDimensionEnum,
    WGIDimensionResponse,
    WGIDimensionScore,
    WGIHistoryEntry,
    WGIHistoryResponse,
    WGIIndicatorsResponse,
    WGIRankingEntry,
    WGIRankingsResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/wgi", tags=["WGI Analysis"])


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _compute_provenance(input_data: Any, output_data: Any) -> str:
    """Compute SHA-256 provenance hash for audit trail."""
    data_str = f"{input_data}{output_data}"
    return hashlib.sha256(data_str.encode("utf-8")).hexdigest()


def _classify_wgi_risk(estimate: Decimal) -> RiskLevelEnum:
    """Classify WGI estimate into a risk level.

    Args:
        estimate: WGI governance estimate on -2.5 to +2.5 scale.

    Returns:
        Risk level classification.
    """
    if estimate <= Decimal("-0.5"):
        return RiskLevelEnum.CRITICAL
    elif estimate <= Decimal("0.0"):
        return RiskLevelEnum.HIGH
    elif estimate <= Decimal("0.5"):
        return RiskLevelEnum.MODERATE
    return RiskLevelEnum.LOW


def _rate_governance(composite: Decimal) -> GovernanceRatingEnum:
    """Rate governance quality based on composite WGI score.

    Args:
        composite: Weighted composite WGI score.

    Returns:
        Governance quality rating.
    """
    if composite >= Decimal("1.5"):
        return GovernanceRatingEnum.EXCELLENT
    elif composite >= Decimal("0.5"):
        return GovernanceRatingEnum.GOOD
    elif composite >= Decimal("0.0"):
        return GovernanceRatingEnum.ADEQUATE
    elif composite >= Decimal("-1.0"):
        return GovernanceRatingEnum.POOR
    return GovernanceRatingEnum.CRITICAL


# ---------------------------------------------------------------------------
# GET /wgi/{country_code}/indicators
# ---------------------------------------------------------------------------


@router.get(
    "/{country_code}/indicators",
    response_model=WGIIndicatorsResponse,
    summary="Get all 6 WGI dimensions for a country",
    description=(
        "Retrieve all 6 World Bank Worldwide Governance Indicators for a "
        "specific country: voice_accountability, political_stability, "
        "government_effectiveness, regulatory_quality, rule_of_law, and "
        "control_of_corruption. Each dimension scored on -2.5 to +2.5 scale."
    ),
    responses={
        200: {"description": "WGI indicators retrieved successfully"},
        400: {"model": SchemaErrorResponse, "description": "Invalid country code"},
        401: {"model": SchemaErrorResponse, "description": "Authentication required"},
        403: {"model": SchemaErrorResponse, "description": "Insufficient permissions"},
        404: {"model": SchemaErrorResponse, "description": "Country not found"},
    },
)
async def get_wgi_indicators(
    country_code: str,
    request: Request,
    year: Optional[int] = Query(None, ge=1996, le=2030, description="Indicator year"),
    user: AuthUser = Depends(
        require_permission("eudr-corruption-index:wgi:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> WGIIndicatorsResponse:
    """Get all 6 WGI dimension indicators for a specific country.

    Args:
        country_code: ISO 3166-1 alpha-2 country code.
        year: Indicator year (default: latest).
        user: Authenticated user with wgi:read permission.

    Returns:
        WGIIndicatorsResponse with all 6 dimensions and composite score.
    """
    start = time.monotonic()
    normalized_code = validate_country_code(country_code)

    try:
        engine = get_wgi_engine()
        result = engine.get_indicators(normalized_code, year=year)

        if result is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"WGI indicators not found for country {normalized_code}",
            )

        dimensions = []
        for dim_data in result.get("dimensions", []):
            dimensions.append(
                WGIDimensionScore(
                    dimension=WGIDimensionEnum(dim_data.get("dimension", "control_of_corruption")),
                    estimate=Decimal(str(dim_data.get("estimate", 0))),
                    standard_error=Decimal(str(dim_data.get("std_error", 0))) if dim_data.get("std_error") else None,
                    percentile_rank=Decimal(str(dim_data.get("percentile", 0))) if dim_data.get("percentile") else None,
                    num_sources=dim_data.get("num_sources"),
                )
            )

        composite = Decimal(str(result.get("composite_score", 0)))
        result_year = result.get("year", year or 2024)

        elapsed_ms = Decimal(str((time.monotonic() - start) * 1000))
        provenance_hash = _compute_provenance(
            f"{normalized_code}:{result_year}", str(composite)
        )

        logger.info(
            "WGI indicators retrieved: country=%s year=%d composite=%s operator=%s",
            normalized_code,
            result_year,
            composite,
            user.operator_id or user.user_id,
        )

        return WGIIndicatorsResponse(
            country_code=normalized_code,
            country_name=result.get("country_name", ""),
            year=result_year,
            dimensions=dimensions,
            composite_score=composite,
            governance_rating=_rate_governance(composite),
            risk_level=_classify_wgi_risk(composite),
            data_source=DataSourceEnum.WORLD_BANK,
            provenance=ProvenanceInfo(
                provenance_hash=provenance_hash,
                processing_time_ms=elapsed_ms.quantize(Decimal("0.01")),
            ),
            metadata=MetadataSchema(
                data_sources=["World Bank WGI"],
            ),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("WGI indicators retrieval failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="WGI indicators retrieval failed",
        )


# ---------------------------------------------------------------------------
# GET /wgi/{country_code}/history
# ---------------------------------------------------------------------------


@router.get(
    "/{country_code}/history",
    response_model=WGIHistoryResponse,
    summary="Get WGI indicator history for a dimension",
    description=(
        "Retrieve historical WGI indicator values for a specific country "
        "and dimension. Returns time series with year-over-year changes, "
        "average estimate, and trend direction."
    ),
    responses={
        200: {"description": "WGI history retrieved"},
        400: {"model": SchemaErrorResponse, "description": "Invalid parameters"},
        401: {"model": SchemaErrorResponse, "description": "Authentication required"},
        404: {"model": SchemaErrorResponse, "description": "Country not found"},
    },
)
async def get_wgi_history(
    country_code: str,
    request: Request,
    dimension: WGIDimensionEnum = Query(
        ...,
        description="WGI dimension to query",
    ),
    year_range: Dict[str, Optional[int]] = Depends(validate_year_range),
    user: AuthUser = Depends(
        require_permission("eudr-corruption-index:wgi:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> WGIHistoryResponse:
    """Get WGI indicator history for a country and dimension.

    Args:
        country_code: ISO 3166-1 alpha-2 country code.
        dimension: WGI dimension to retrieve history for.
        year_range: Optional start/end year range.
        user: Authenticated user with wgi:read permission.

    Returns:
        WGIHistoryResponse with time series data and trend analysis.
    """
    start = time.monotonic()
    normalized_code = validate_country_code(country_code)

    try:
        engine = get_wgi_engine()
        result = engine.get_dimension_history(
            normalized_code,
            dimension=dimension.value,
            start_year=year_range.get("start_year"),
            end_year=year_range.get("end_year"),
        )

        if result is None or not result.get("history"):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"WGI history not found for {normalized_code}/{dimension.value}",
            )

        history_entries = []
        prev_estimate = None
        for entry in result.get("history", []):
            estimate = Decimal(str(entry.get("estimate", 0)))
            change = None
            if prev_estimate is not None:
                change = estimate - prev_estimate
            history_entries.append(
                WGIHistoryEntry(
                    year=entry.get("year", 2024),
                    estimate=estimate,
                    percentile_rank=Decimal(str(entry.get("percentile", 0))) if entry.get("percentile") else None,
                    change_from_prior=change,
                )
            )
            prev_estimate = estimate

        estimates = [e.estimate for e in history_entries]
        avg_estimate = sum(estimates) / len(estimates) if estimates else Decimal("0")
        total_change = estimates[-1] - estimates[0] if len(estimates) >= 2 else Decimal("0")

        if total_change > Decimal("0.1"):
            trend = TrendDirectionEnum.IMPROVING
        elif total_change < Decimal("-0.1"):
            trend = TrendDirectionEnum.DETERIORATING
        else:
            trend = TrendDirectionEnum.STABLE

        period_start = history_entries[0].year if history_entries else 2015
        period_end = history_entries[-1].year if history_entries else 2024

        elapsed_ms = Decimal(str((time.monotonic() - start) * 1000))
        provenance_hash = _compute_provenance(
            f"{normalized_code}:{dimension.value}:{period_start}-{period_end}",
            str(avg_estimate),
        )

        logger.info(
            "WGI history retrieved: country=%s dimension=%s entries=%d operator=%s",
            normalized_code,
            dimension.value,
            len(history_entries),
            user.operator_id or user.user_id,
        )

        return WGIHistoryResponse(
            country_code=normalized_code,
            country_name=result.get("country_name", ""),
            dimension=dimension,
            history=history_entries,
            period_start=period_start,
            period_end=period_end,
            average_estimate=avg_estimate.quantize(Decimal("0.001")),
            trend_direction=trend,
            total_change=total_change.quantize(Decimal("0.001")),
            provenance=ProvenanceInfo(
                provenance_hash=provenance_hash,
                processing_time_ms=elapsed_ms.quantize(Decimal("0.01")),
            ),
            metadata=MetadataSchema(data_sources=["World Bank WGI"]),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("WGI history retrieval failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="WGI history retrieval failed",
        )


# ---------------------------------------------------------------------------
# GET /wgi/dimension/{dimension}
# ---------------------------------------------------------------------------


@router.get(
    "/dimension/{dimension}",
    response_model=WGIDimensionResponse,
    summary="Get cross-country WGI analysis for a dimension",
    description=(
        "Retrieve a specific WGI dimension score across all countries. "
        "Supports pagination and returns global statistics for the dimension."
    ),
    responses={
        200: {"description": "Dimension analysis retrieved"},
        401: {"model": SchemaErrorResponse, "description": "Authentication required"},
        403: {"model": SchemaErrorResponse, "description": "Insufficient permissions"},
    },
)
async def get_wgi_dimension(
    dimension: WGIDimensionEnum,
    request: Request,
    year: Optional[int] = Query(None, ge=1996, le=2030, description="Analysis year"),
    pagination: PaginationParams = Depends(get_pagination),
    user: AuthUser = Depends(
        require_permission("eudr-corruption-index:wgi:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> WGIDimensionResponse:
    """Get cross-country analysis for a specific WGI dimension.

    Args:
        dimension: WGI dimension to analyze.
        year: Analysis year (default: latest).
        pagination: Pagination parameters.
        user: Authenticated user with wgi:read permission.

    Returns:
        WGIDimensionResponse with country scores for the dimension.
    """
    start = time.monotonic()

    try:
        engine = get_wgi_engine()
        result = engine.get_dimension_analysis(
            dimension=dimension.value,
            year=year,
            limit=pagination.limit,
            offset=pagination.offset,
        )

        countries = []
        for entry in result.get("countries", []):
            estimate = Decimal(str(entry.get("estimate", 0)))
            countries.append(
                WGIDimensionCountryEntry(
                    country_code=entry.get("country_code", ""),
                    country_name=entry.get("country_name", ""),
                    estimate=estimate,
                    percentile_rank=Decimal(str(entry.get("percentile", 0))) if entry.get("percentile") else None,
                    risk_level=_classify_wgi_risk(estimate),
                )
            )

        total = result.get("total", len(countries))
        result_year = result.get("year", year or 2024)

        elapsed_ms = Decimal(str((time.monotonic() - start) * 1000))
        provenance_hash = _compute_provenance(
            f"dimension:{dimension.value}:{result_year}", str(total)
        )

        logger.info(
            "WGI dimension analysis: dimension=%s year=%d countries=%d operator=%s",
            dimension.value,
            result_year,
            total,
            user.operator_id or user.user_id,
        )

        return WGIDimensionResponse(
            dimension=dimension,
            year=result_year,
            countries=countries,
            global_average=Decimal(str(result.get("global_average", 0))),
            global_median=Decimal(str(result.get("global_median", 0))),
            total_countries=total,
            pagination=PaginatedMeta(
                total=total,
                limit=pagination.limit,
                offset=pagination.offset,
                has_more=(pagination.offset + pagination.limit) < total,
            ),
            provenance=ProvenanceInfo(
                provenance_hash=provenance_hash,
                processing_time_ms=elapsed_ms.quantize(Decimal("0.01")),
            ),
            metadata=MetadataSchema(data_sources=["World Bank WGI"]),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("WGI dimension analysis failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="WGI dimension analysis failed",
        )


# ---------------------------------------------------------------------------
# POST /wgi/compare
# ---------------------------------------------------------------------------


@router.post(
    "/compare",
    response_model=WGIComparisonResponse,
    summary="Compare countries across WGI dimensions",
    description=(
        "Compare 2-20 countries across WGI governance dimensions. Returns "
        "per-country dimension scores, composite scores, governance ratings, "
        "pairwise differential matrix, and best/worst performer identification."
    ),
    responses={
        200: {"description": "WGI comparison completed"},
        400: {"model": SchemaErrorResponse, "description": "Invalid request"},
        401: {"model": SchemaErrorResponse, "description": "Authentication required"},
        403: {"model": SchemaErrorResponse, "description": "Insufficient permissions"},
        429: {"model": SchemaErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def compare_wgi(
    request: Request,
    body: WGIComparisonRequest,
    user: AuthUser = Depends(
        require_permission("eudr-corruption-index:wgi:read")
    ),
    _rate: None = Depends(rate_limit_heavy),
) -> WGIComparisonResponse:
    """Compare countries across WGI governance dimensions.

    Args:
        body: Comparison request with country codes and optional dimension filter.
        user: Authenticated user with wgi:read permission.

    Returns:
        WGIComparisonResponse with comparison results and differential matrix.
    """
    start = time.monotonic()

    try:
        engine = get_wgi_engine()
        result = engine.compare_countries(
            country_codes=body.country_codes,
            year=body.year,
            dimensions=[d.value for d in body.dimensions] if body.dimensions else None,
        )

        countries = []
        for entry in result.get("countries", []):
            dims = []
            for dim_data in entry.get("dimensions", []):
                dims.append(
                    WGIDimensionScore(
                        dimension=WGIDimensionEnum(dim_data.get("dimension", "control_of_corruption")),
                        estimate=Decimal(str(dim_data.get("estimate", 0))),
                        standard_error=Decimal(str(dim_data.get("std_error", 0))) if dim_data.get("std_error") else None,
                        percentile_rank=Decimal(str(dim_data.get("percentile", 0))) if dim_data.get("percentile") else None,
                    )
                )
            composite = Decimal(str(entry.get("composite_score", 0)))
            countries.append(
                WGIComparisonCountryEntry(
                    country_code=entry.get("country_code", ""),
                    country_name=entry.get("country_name", ""),
                    dimensions=dims,
                    composite_score=composite,
                    governance_rating=_rate_governance(composite),
                )
            )

        # Build differential matrix
        matrix: Dict[str, Dict[str, Decimal]] = {}
        for c1 in countries:
            matrix[c1.country_code] = {}
            for c2 in countries:
                matrix[c1.country_code][c2.country_code] = abs(
                    c1.composite_score - c2.composite_score
                ).quantize(Decimal("0.001"))

        best = max(countries, key=lambda c: c.composite_score).country_code if countries else None
        worst = min(countries, key=lambda c: c.composite_score).country_code if countries else None

        result_year = result.get("year", body.year or 2024)

        elapsed_ms = Decimal(str((time.monotonic() - start) * 1000))
        provenance_hash = _compute_provenance(
            f"compare:{','.join(body.country_codes)}:{result_year}",
            str(len(countries)),
        )

        logger.info(
            "WGI comparison completed: countries=%d year=%d operator=%s",
            len(countries),
            result_year,
            user.operator_id or user.user_id,
        )

        return WGIComparisonResponse(
            year=result_year,
            countries=countries,
            differential_matrix=matrix,
            best_performer=best,
            worst_performer=worst,
            provenance=ProvenanceInfo(
                provenance_hash=provenance_hash,
                processing_time_ms=elapsed_ms.quantize(Decimal("0.01")),
            ),
            metadata=MetadataSchema(data_sources=["World Bank WGI"]),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("WGI comparison failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="WGI comparison failed",
        )


# ---------------------------------------------------------------------------
# GET /wgi/rankings
# ---------------------------------------------------------------------------


@router.get(
    "/rankings",
    response_model=WGIRankingsResponse,
    summary="Get WGI rankings by dimension",
    description=(
        "Retrieve country rankings for a specific WGI governance dimension. "
        "Countries are ranked by governance estimate in descending order."
    ),
    responses={
        200: {"description": "WGI rankings retrieved"},
        401: {"model": SchemaErrorResponse, "description": "Authentication required"},
        403: {"model": SchemaErrorResponse, "description": "Insufficient permissions"},
    },
)
async def get_wgi_rankings(
    request: Request,
    dimension: WGIDimensionEnum = Query(
        ...,
        description="WGI dimension to rank by",
    ),
    year: Optional[int] = Query(None, ge=1996, le=2030, description="Rankings year"),
    pagination: PaginationParams = Depends(get_pagination),
    user: AuthUser = Depends(
        require_permission("eudr-corruption-index:wgi:read")
    ),
    _rate: None = Depends(rate_limit_standard),
) -> WGIRankingsResponse:
    """Get WGI rankings by a specific governance dimension.

    Args:
        dimension: WGI dimension to rank by.
        year: Rankings year (default: latest).
        pagination: Pagination parameters.
        user: Authenticated user with wgi:read permission.

    Returns:
        WGIRankingsResponse with ranked country list.
    """
    start = time.monotonic()

    try:
        engine = get_wgi_engine()
        result = engine.get_rankings(
            dimension=dimension.value,
            year=year,
            limit=pagination.limit,
            offset=pagination.offset,
        )

        rankings = []
        for entry in result.get("rankings", []):
            rankings.append(
                WGIRankingEntry(
                    rank=entry.get("rank", 0),
                    country_code=entry.get("country_code", ""),
                    country_name=entry.get("country_name", ""),
                    estimate=Decimal(str(entry.get("estimate", 0))),
                    percentile_rank=Decimal(str(entry.get("percentile", 0))) if entry.get("percentile") else None,
                )
            )

        total = result.get("total", len(rankings))
        result_year = result.get("year", year or 2024)

        elapsed_ms = Decimal(str((time.monotonic() - start) * 1000))
        provenance_hash = _compute_provenance(
            f"wgi_rankings:{dimension.value}:{result_year}", str(total)
        )

        logger.info(
            "WGI rankings retrieved: dimension=%s year=%d total=%d operator=%s",
            dimension.value,
            result_year,
            total,
            user.operator_id or user.user_id,
        )

        return WGIRankingsResponse(
            dimension=dimension,
            year=result_year,
            rankings=rankings,
            total_countries=total,
            global_average=Decimal(str(result.get("global_average", 0))),
            pagination=PaginatedMeta(
                total=total,
                limit=pagination.limit,
                offset=pagination.offset,
                has_more=(pagination.offset + pagination.limit) < total,
            ),
            provenance=ProvenanceInfo(
                provenance_hash=provenance_hash,
                processing_time_ms=elapsed_ms.quantize(Decimal("0.01")),
            ),
            metadata=MetadataSchema(data_sources=["World Bank WGI"]),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("WGI rankings retrieval failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="WGI rankings retrieval failed",
        )
